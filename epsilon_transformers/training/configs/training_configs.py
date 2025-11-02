

"""
CORRECT training_configs.py for epsilon-transformers with KL analysis support.

This file extends the existing training_configs.py with KL analysis capabilities.
Uses Pydantic Config inheritance (not dataclasses) to match the repo structure.
"""

from typing import Literal, Optional
from pydantic import BaseModel, field_validator, model_validator
import pathlib
import torch
from torch.utils.data import DataLoader
import wandb
import os
import dotenv
import math
from dataclasses import dataclass, asdict, field

from epsilon_transformers.persistence import Persister
from epsilon_transformers.process.processes import PROCESS_REGISTRY
from epsilon_transformers.process.dataset import (
    ProcessDataset,
    process_dataset_collate_fn,
)
from epsilon_transformers.training.configs.base_config import Config
from epsilon_transformers.training.configs.model_configs import RawModelConfig


# ============================================================================
# EXISTING CONFIGS (from repo - DO NOT MODIFY)
# ============================================================================

Optimizer = torch.optim.Adam | torch.optim.SGD


class OptimizerConfig(Config):
    optimizer_type: Literal["sgd", "adam"]
    learning_rate: float
    weight_decay: float

    def from_model(self, model: torch.nn.Module, device: torch.device) -> Optimizer:
        if self.optimizer_type == "adam":
            optimizer = torch.optim.Adam
        elif self.optimizer_type == "sgd":
            optimizer = torch.optim.SGD
        else:
            raise ValueError(
                f"{self.optimizer_type} is not a valid optimizer_type. Must be 'adam' or 'sgd'"
            )
        return optimizer(
            model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )


class PersistanceConfig(Config):
    location: Literal["local", "s3"]
    collection_location: pathlib.Path | str
    checkpoint_every_n_tokens: int

    def init(self) -> Persister:
        use_s3 = self.location == "s3"
        save_dir = str(self.collection_location)
        return Persister(save_dir=save_dir, use_s3=use_s3) 


class ProcessDatasetConfig(Config):
    """Dataset configuration."""
    process: str
    process_params: dict[str, float]
    batch_size: int
    sequence_length: int
    num_tokens: int
    test_split: float

    @field_validator("batch_size")
    @classmethod
    def validate_batch_size(cls, v):
        """Validate batch size."""
        if v <= 0:
            raise ValueError("batch_size must be > 0")
        return v

    def to_dataloader(self, sequence_length: int, train: bool) -> DataLoader:
        """Create dataloader from config."""
        # Use sequence_length from config by default
        seq_len = sequence_length
        
        dataset = ProcessDataset(
            process_name=self.process,
            process_params=self.process_params,
            sequence_length=seq_len,
            num_samples=(
                self.num_tokens
                if train
                else math.ceil(self.num_tokens * self.test_split)
            ),
        )
        return DataLoader(
            dataset=dataset,
            collate_fn=process_dataset_collate_fn,
            batch_size=self.batch_size,
        )


@dataclass
class Log:
    train_loss: float | None
    test_loss: float | None
    config: "LoggingConfig"
    
    metrics: dict[str, dict[str, float]] = field(default_factory=lambda: {"train": {}, "test": {}})


    def reset(self):
        if self.config.train_loss:
            self.train_loss = 0.0
        else:
            self.train_loss = None

        if self.config.test_loss:
            self.test_loss = 0.0
        else:
            self.test_loss = None

        self.metrics = {"train": {}, "test": {}}    

    def update_metrics(self, train_or_test: Literal["train", "test"], loss: float, metric_name: Optional[str] = 'loss'):
        if metric_name == 'loss':
            if train_or_test == "train" and self.config.train_loss:
                assert self.train_loss is not None
                self.train_loss += loss
            elif train_or_test == "test" and self.config.test_loss:
                assert self.test_loss is not None
                self.test_loss += loss
            else:
                raise ValueError(f"Invalid train_or_test: {train_or_test}")
            
        if train_or_test not in self.metrics:
            self.metrics[train_or_test] = {}
        if metric_name not in self.metrics[train_or_test]:
            self.metrics[train_or_test][metric_name] = 0.0
        self.metrics[train_or_test][metric_name] += float(loss)   

    def persist(self):
        if self.config.wandb:
            wandb.log(
                {
                    k: v
                    for k, v in asdict(self).items()
                    if v is not None and not isinstance(v, LoggingConfig)
                }
            )

        if self.metrics:
                flat = {}
                for split, md in self.metrics.items():
                    for name, val in md.items():
                        flat[f"{split}/{name}"] = val
                if flat:
                    wandb.log(flat)
    
        if self.config.local is not None:
            raise NotImplementedError


class LoggingConfig(Config):
    local: pathlib.Path | None = None
    wandb: bool = True
    project_name: str | None = None
    wandb_api_key: str | None = None  # NEW: Explicit API key field
    train_loss: bool = True
    test_loss: bool = True

    @field_validator("project_name")
    @classmethod
    def validate_wandb_config(cls, v, info):
        """Validate that project_name is set if wandb is enabled."""
        wandb_enabled = info.data.get("wandb", False)
        if wandb_enabled and not v:
            raise ValueError("project_name must be provided if wandb logging is enabled")
        return v

    def close(self):
        if self.wandb:
            wandb.finish()
        if self.local is not None:
            raise NotImplementedError

    def init(self) -> Log:
        return Log(
            config=self,
            train_loss=0.0 if self.train_loss else None,
            test_loss=0.0 if self.test_loss else None,
        )


# ============================================================================
# NEW: KL ANALYSIS CONFIGS
# ============================================================================

@dataclass
class NGramAnalysisConfig:
    """Configuration for n-gram KL divergence analysis during validation."""
    enabled: bool = True
    n_values: list[int] = field(default_factory=lambda: [1, 2, 3])
    return_per_position: bool = True

    def __post_init__(self):
        """Validate n_values."""
        if not all(isinstance(n, int) and n >= 1 for n in self.n_values):
            raise ValueError("n_values must be list of positive integers")
        if max(self.n_values) > 5:
            raise ValueError("n_values > 5 may be too slow, recommended max is 3")


@dataclass
class MarkovKLAnalysisConfig:
    """Configuration for Markov process KL divergence analysis during validation."""
    enabled: bool = True
    return_per_position: bool = True


@dataclass
class KLAnalysisConfig:
    """Configuration for all KL analysis metrics."""
    ngram_analysis: NGramAnalysisConfig = field(default_factory=NGramAnalysisConfig)
    markov_kl_analysis: MarkovKLAnalysisConfig = field(default_factory=MarkovKLAnalysisConfig)


# ============================================================================
# EXTENDED: TrainConfig WITH KL Analysis
# ============================================================================

class TrainConfig(Config):
    model: RawModelConfig
    optimizer: OptimizerConfig
    dataset: ProcessDatasetConfig
    persistance: PersistanceConfig
    logging: LoggingConfig
    seed: int
    verbose: bool
    
    # NEW: KL Analysis configuration
    kl_analysis: KLAnalysisConfig = field(default_factory=KLAnalysisConfig)

    @model_validator(mode="after")
    def validate_model(self):
        """Validate model vocab matches process vocab (if process is registered)."""
        dataset_process = self.dataset.process
        
        # Only validate if process is in PROCESS_REGISTRY
        if dataset_process and dataset_process in PROCESS_REGISTRY:
            try:
                process_vocab_len = PROCESS_REGISTRY[dataset_process]().vocab_len
                if self.model.d_vocab != process_vocab_len:
                    raise ValueError(
                        f"Model's d_vocab ({self.model.d_vocab}) doesn't match "
                        f"dataset process's vocab_len ({process_vocab_len})"
                    )
            except KeyError:
                # Process not registered, skip validation
                print(f"[Warning] Process '{dataset_process}' not in PROCESS_REGISTRY, skipping vocab validation")
        elif dataset_process:
            print(f"[Warning] Process '{dataset_process}' not found in PROCESS_REGISTRY")
            print(f"[Warning] Available processes: {list(PROCESS_REGISTRY.keys())}")
        
        return self

    def init_logger(self) -> Log:
        """Initialize logger with optional wandb support."""
        if self.logging.wandb:
            dotenv.load_dotenv()
            
            # Try to get API key from config first, then environment
            wandb_api_key = self.logging.wandb_api_key or os.environ.get("WANDB_API_KEY", None)
            
            if wandb_api_key is None:
                raise ValueError(
                    "To use wandb, provide wandb_api_key in config or set WANDB_API_KEY environment variable"
                )

            wandb.login(key=wandb_api_key)
            wandb.init(project=self.logging.project_name, config=self.model_dump())
        
        if self.logging.local is not None:
            raise NotImplementedError()
        
        return self.logging.init()
