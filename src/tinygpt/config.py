"""Configuration objects used across parsing, training, and generation."""

from __future__ import annotations

from dataclasses import asdict, dataclass


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
    """Model hyperparameters for the tiny decoder-only GPT."""

    vocab_size: int
    context_length: int = 128
    embedding_dim: int = 128
    num_heads: int = 4
    num_layers: int = 4
    dropout: float = 0.1

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(slots=True)
class TrainingConfig:
    """Training hyperparameters and runtime settings."""

    input_text: str
    run_name: str = "tiny-gpt-run"
    batch_size: int = 32
    max_steps: int = 3000
    eval_interval: int = 200
    eval_batches: int = 20
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    train_split: float = 0.9
    sample_length: int = 200
    seed: int = 42
    device: str = "cuda"
    output_dir: str = "artifacts"

    def to_dict(self) -> dict[str, object]:
        return asdict(self)
