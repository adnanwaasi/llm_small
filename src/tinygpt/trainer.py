"""Training loop and checkpoint helpers for the tiny GPT project."""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import random

import torch
from torch import optim

from tinygpt.config import ModelConfig, TrainingConfig
from tinygpt.dataset import get_batch, split_token_stream
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


def train_model(training_config: TrainingConfig, model_config: ModelConfig, tokenizer: BaseTokenizer, corpus_text: str) -> Path:
    """Train the model and save best/latest checkpoints.

    Returns:
        Path to the best checkpoint file.
    """

    set_seed(training_config.seed)
    device = resolve_device(training_config.device)
    token_ids = tokenizer.encode(corpus_text)
    train_data, val_data = split_token_stream(token_ids, training_config.train_split)

    run_dir = Path(training_config.output_dir) / training_config.run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    model = TinyGPT(model_config).to(device)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=training_config.learning_rate,
        weight_decay=training_config.weight_decay,
    )

    best_val_loss = float("inf")
    best_checkpoint = run_dir / "best.pt"

    for step in range(1, training_config.max_steps + 1):
        model.train()
        inputs, targets = get_batch(
            train_data,
            batch_size=training_config.batch_size,
            context_length=model_config.context_length,
            device=device,
        )
        _, loss = model(inputs, targets)
        if loss is None:
            raise RuntimeError("Loss was unexpectedly None during training")

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if step % training_config.eval_interval == 0 or step == training_config.max_steps:
            train_loss = estimate_loss(model, train_data, model_config, training_config, device)
            val_loss = estimate_loss(model, val_data, model_config, training_config, device)
            print(f"step={step} train_loss={train_loss:.4f} val_loss={val_loss:.4f}")

            checkpoint_payload = _checkpoint_payload(
                model=model,
                model_config=model_config,
                training_config=training_config,
                tokenizer=tokenizer,
                step=step,
                val_loss=val_loss,
            )
            torch.save(checkpoint_payload, run_dir / "latest.pt")
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(checkpoint_payload, best_checkpoint)

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
    data: torch.Tensor,
    model_config: ModelConfig,
    training_config: TrainingConfig,
    device: torch.device,
) -> float:
    """Estimate average loss over several random batches."""

    model.eval()
    losses = []
    for _ in range(training_config.eval_batches):
        inputs, targets = get_batch(
            data,
            batch_size=training_config.batch_size,
            context_length=model_config.context_length,
            device=device,
        )
        _, loss = model(inputs, targets)
        if loss is None:
            raise RuntimeError("Loss was unexpectedly None during evaluation")
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)


def load_checkpoint(checkpoint_path: str | Path, device_name: str = "cuda") -> tuple[TinyGPT, BaseTokenizer, ModelConfig, dict[str, object]]:
    """Load a checkpoint and reconstruct the model and tokenizer."""

    device = resolve_device(device_name)
    payload = torch.load(checkpoint_path, map_location=device)
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
