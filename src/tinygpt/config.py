"""Configuration objects used across parsing, training, and generation."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass(slots=True)
class ParserConfig:
    """Controls how Instagram exports are turned into training text."""

    input_dir: str = "fuck"
    output_text: str = "data/processed/corpus.txt"
    output_jsonl: str = "data/processed/messages.jsonl"
    speaker: str | None = None
    include_message_requests: bool = True
    include_reactions: bool = False
    lowercase: bool = False
    clean_training_text: bool = False

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(slots=True)
class ModelConfig:
    """Model hyperparameters for the tiny decoder-only GPT.

    Defaults (~16M params with an 8k vocab) fit comfortably on a 4 GB GPU.
    """

    vocab_size: int
    context_length: int = 256
    embedding_dim: int = 384
    num_heads: int = 6
    num_layers: int = 6
    dropout: float = 0.1
    rope_base: float = 10000.0

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(slots=True)
class TrainingConfig:
    """Training hyperparameters and runtime settings.

    `stage="pretrain"` learns general chat from public dialogues (loss on every
    token). `stage="finetune"` starts from a pretrained checkpoint and learns
    only your side of conversations.
    """

    data_paths: list[str] = field(default_factory=list)
    tokenizer_path: str = "artifacts/tokenizer.json"
    stage: str = "pretrain"
    init_from: str | None = None
    run_name: str = "tiny-gpt-run"
    batch_size: int = 32
    gradient_accumulation: int = 1
    max_steps: int = 3000
    warmup_steps: int = 200
    eval_interval: int = 200
    eval_batches: int = 20
    learning_rate: float = 3e-4
    min_learning_rate_ratio: float = 0.1
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    val_fraction: float = 0.05
    patience: int | None = None
    seed: int = 42
    device: str = "cuda"
    output_dir: str = "artifacts"

    @property
    def train_on(self) -> str:
        return "all" if self.stage == "pretrain" else "me"

    def to_dict(self) -> dict[str, object]:
        return asdict(self)
