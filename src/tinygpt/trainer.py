"""Training loop and checkpoint helpers for the tiny GPT project."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import asdict
import json
import math
from pathlib import Path
import random
import time

import torch
from torch import optim

from tinygpt.config import ModelConfig, TrainingConfig
from tinygpt.dataset import ChatStream
from tinygpt.model import TinyGPT
from tinygpt.tokenizer import BaseTokenizer, load_tokenizer_from_dict


def set_seed(seed: int) -> None:
    """Seed Python and PyTorch for reproducibility."""

    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device_name: str) -> torch.device:
    """Resolve a requested device while gracefully falling back to CPU."""

    if device_name == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def learning_rate_at(step: int, config: TrainingConfig) -> float:
    """Linear warmup then cosine decay to `min_learning_rate_ratio * lr`."""

    peak = config.learning_rate
    if step < config.warmup_steps:
        return peak * (step + 1) / config.warmup_steps
    progress = (step - config.warmup_steps) / max(1, config.max_steps - config.warmup_steps)
    floor = peak * config.min_learning_rate_ratio
    return floor + 0.5 * (peak - floor) * (1 + math.cos(math.pi * min(1.0, progress)))


def _optimizer(model: TinyGPT, config: TrainingConfig, device: torch.device) -> optim.Optimizer:
    # No weight decay on norms (1-D params); decay everything else.
    decay = [parameter for parameter in model.parameters() if parameter.dim() >= 2]
    no_decay = [parameter for parameter in model.parameters() if parameter.dim() < 2]
    return optim.AdamW(
        [{"params": decay, "weight_decay": config.weight_decay}, {"params": no_decay, "weight_decay": 0.0}],
        lr=config.learning_rate,
        betas=(0.9, 0.95),
        fused=device.type == "cuda",
    )


def train_model(
    training_config: TrainingConfig,
    model_config: ModelConfig,
    tokenizer: BaseTokenizer,
    train_stream: ChatStream,
    val_stream: ChatStream,
    *,
    init_state: dict[str, torch.Tensor] | None = None,
) -> Path:
    """Train the model and save best/latest checkpoints.

    Returns:
        Path to the best checkpoint file.
    """

    set_seed(training_config.seed)
    device = resolve_device(training_config.device)
    run_dir = Path(training_config.output_dir) / training_config.run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    model = TinyGPT(model_config).to(device)
    if init_state is not None:
        model.load_state_dict(init_state)
    optimizer = _optimizer(model, training_config, device)

    # Turing GPUs (e.g. T2000) have no bf16, so use fp16 autocast + loss scaling.
    use_amp = device.type == "cuda"
    amp_dtype = torch.bfloat16 if use_amp and torch.cuda.is_bf16_supported() else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and amp_dtype == torch.float16)
    autocast = (lambda: torch.autocast("cuda", dtype=amp_dtype)) if use_amp else nullcontext

    print(
        f"device={device} params={model.num_parameters() / 1e6:.2f}M "
        f"train_tokens={train_stream.num_tokens} learn_tokens={train_stream.num_learn_tokens} "
        f"val_tokens={val_stream.num_tokens}",
        flush=True,
    )

    best_val_loss = float("inf")
    best_checkpoint = run_dir / "best.pt"
    evals_without_improvement = 0
    started = time.time()

    for step in range(training_config.max_steps):
        for group in optimizer.param_groups:
            group["lr"] = learning_rate_at(step, training_config)

        model.train()
        for _ in range(training_config.gradient_accumulation):
            inputs, targets = train_stream.get_batch(training_config.batch_size, model_config.context_length, device)
            with autocast():
                _, loss = model(inputs, targets)
            if loss is None:
                raise RuntimeError("Loss was unexpectedly None during training")
            scaler.scale(loss / training_config.gradient_accumulation).backward()

        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), training_config.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        last_step = step + 1 == training_config.max_steps
        if (step + 1) % training_config.eval_interval == 0 or last_step:
            train_loss = estimate_loss(model, train_stream, model_config, training_config, device, autocast)
            val_loss = estimate_loss(model, val_stream, model_config, training_config, device, autocast)
            elapsed = time.time() - started
            print(
                f"step={step + 1} lr={learning_rate_at(step, training_config):.2e} "
                f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} elapsed={elapsed:.0f}s",
                flush=True,
            )

            checkpoint_payload = _checkpoint_payload(
                model=model,
                model_config=model_config,
                training_config=training_config,
                tokenizer=tokenizer,
                step=step + 1,
                val_loss=val_loss,
            )
            torch.save(checkpoint_payload, run_dir / "latest.pt")
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                evals_without_improvement = 0
                torch.save(checkpoint_payload, best_checkpoint)
            else:
                evals_without_improvement += 1
                # Personal datasets are small; stop before the model memorizes them.
                if training_config.patience is not None and evals_without_improvement >= training_config.patience:
                    print(f"early stop: val loss has not improved for {evals_without_improvement} evals")
                    break

    metadata_path = run_dir / "run_config.json"
    metadata_path.write_text(
        json.dumps(
            {
                "model_config": asdict(model_config),
                "training_config": asdict(training_config),
                "best_val_loss": best_val_loss,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    return best_checkpoint


@torch.no_grad()
def estimate_loss(
    model: TinyGPT,
    stream: ChatStream,
    model_config: ModelConfig,
    training_config: TrainingConfig,
    device: torch.device,
    autocast=nullcontext,
) -> float:
    """Estimate average loss over a fixed set of random batches."""

    model.eval()
    generator = torch.Generator().manual_seed(training_config.seed)
    losses = []
    for _ in range(training_config.eval_batches):
        inputs, targets = stream.get_batch(training_config.batch_size, model_config.context_length, device, generator)
        with autocast():
            _, loss = model(inputs, targets)
        if loss is None:
            raise RuntimeError("Loss was unexpectedly None during evaluation")
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)


def load_checkpoint(checkpoint_path: str | Path, device_name: str = "cuda") -> tuple[TinyGPT, BaseTokenizer, ModelConfig, dict[str, object]]:
    """Load a checkpoint and reconstruct the model and tokenizer."""

    device = resolve_device(device_name)
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_config = ModelConfig(**payload["model_config"])
    tokenizer_payload = payload["tokenizer"]
    if not isinstance(tokenizer_payload, dict):
        raise TypeError("Checkpoint tokenizer payload must be a dictionary")
    tokenizer = load_tokenizer_from_dict(tokenizer_payload)
    model = TinyGPT(model_config).to(device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    return model, tokenizer, model_config, payload


def _checkpoint_payload(
    *,
    model: TinyGPT,
    model_config: ModelConfig,
    training_config: TrainingConfig,
    tokenizer: BaseTokenizer,
    step: int,
    val_loss: float,
) -> dict[str, object]:
    return {
        "model_state_dict": model.state_dict(),
        "model_config": asdict(model_config),
        "training_config": asdict(training_config),
        "tokenizer": tokenizer.to_dict(),
        "step": step,
        "val_loss": val_loss,
    }
