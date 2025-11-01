

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
        if self.location == "local":
            assert isinstance(self.collection_location, pathlib.Path)
            return Persister(collection_location=self.collection_location)
        elif self.location == "s3":
            assert isinstance(self.collection_location, str)
            return Persister(collection_location=self.collection_location)
        else:
            raise ValueError(
                f"{self.location} is invalid. Must be 'local' or 's3'"
            )


class ProcessDatasetConfig(Config):
    process: str
    process_params: dict[str, float]
    batch_size: int
    num_tokens: int
    test_split: float

    def to_dataloader(self, sequence_length: int, train: bool) -> DataLoader:
        dataset = ProcessDataset(
            process_name=self.process,
            process_params=self.process_params,
            sequence_length=sequence_length,
            num_samples=(
                self.num_tokens
                if train
                else math.floor(self.num_tokens * self.test_split)
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

    def reset(self):
        if self.config.train_loss:
            self.train_loss = 0.0
        else:
            self.train_loss = None

        if self.config.test_loss:
            self.test_loss = 0.0
        else:
            self.test_loss = None

    def update_metrics(self, train_or_test: Literal["train", "test"], loss: float):
        if train_or_test == "train" and self.config.train_loss:
            assert self.train_loss is not None
            self.train_loss += loss
        elif train_or_test == "test" and self.config.test_loss:
            assert self.test_loss is not None
            self.test_loss += loss
        else:
            raise ValueError(f"Invalid train_or_test: {train_or_test}")

    def persist(self):
        if self.config.wandb:
            wandb.log(
                {
                    k: v
                    for k, v in asdict(self).items()
                    if v is not None and not isinstance(v, LoggingConfig)
                }
            )
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
# from typing import Literal
# from pydantic import model_validator
# import pathlib
# import torch
# from torch.utils.data import DataLoader
# import wandb
# import os
# import dotenv
# import math
# from dataclasses import dataclass, asdict

# from epsilon_transformers.persistence import LocalPersister, Persister, S3Persister
# from epsilon_transformers.process.processes import PROCESS_REGISTRY
# from epsilon_transformers.process.dataset import (
#     ProcessDataset,
#     process_dataset_collate_fn,
# )
# from epsilon_transformers.training.configs.base_config import Config
# from epsilon_transformers.training.configs.model_configs import RawModelConfig


# # # TODO: For persistence config, upon init make sure that you check that the relevant environment variables are set
# # # TODO: Generalize the checkpoint_dir option so that it can work w/ S3 outputs

# # # TODO: Make Config ABS (??)
# # # TODO: Turn log input into a dataclass (??)
# # # TODO: Have a no persistenc config option

# # # TODO: Put all the functionality of the log congig into the logger
# # # TODO: Fix the eval_dataloader_ratio_creation
# # # TODO: Create a logger & log the file path and intermediary metrics
# # # TODO: Add validator to make sure test_split is a fraction
# # # TODO: Add validator in Persistence Config to make sure the path is a dir
# # # TODO: Add validator in Logging Config to make sure that if we're logging wandb then we're using a project name
# # # TODO: Figure out if model seed should be it's own thing or whether we can just use the same seed across
# # # TODO: Decide on whether we want to use HookedTransformer exclusively or whether creating our own model class makes the most sense

# # # TODO: Think if you can make Log DRY
# # # TODO: Switch statement code smell with update_loss_metrics

# # # TODO: Add a learning rate scheduler config
# # # TODO: Add a WandbLoggingConfig
# # # TODO: Add a sweep config
# # # TODO: Add epoch training


# # Optimizer = torch.optim.Adam | torch.optim.SGD


# # class OptimizerConfig(Config):
# #     optimizer_type: Literal["sgd", "adam"]
# #     learning_rate: float
# #     weight_decay: float

# #     def from_model(self, model: torch.nn.Module, device: torch.device) -> Optimizer:
# #         optimizer: type[torch.optim.Adam | torch.optim.SGD]

# #         if self.optimizer_type == "adam":
# #             optimizer = torch.optim.Adam
# #         elif self.optimizer_type == "sgd":
# #             optimizer = torch.optim.SGD
# #         else:
# #             raise ValueError(
# #                 f"{self.optimizer_type} is not a valid optimizer_type. It must be either 'adam' or 'sgd'"
# #             )

# #         return optimizer(
# #             model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
# #         )


# # class PersistanceConfig(Config):
# #     location: Literal["local", "s3"]
# #     collection_location: pathlib.Path | str
# #     checkpoint_every_n_tokens: int

# #     def init(self) -> Persister:
# #         if self.location == "local":
# #             assert isinstance(self.collection_location, pathlib.Path)
# #             return LocalPersister(collection_location=self.collection_location)
# #         elif self.location == "s3":
# #             assert isinstance(self.collection_location, str)
# #             return S3Persister(collection_location=self.collection_location)
# #         else:
# #             raise ValueError(
# #                 f"{self.location} is an invalid location value. It must be either 'local' or 's3'"
# #             )


# # class ProcessDatasetConfig(Config):
# #     process: str
# #     process_params: dict[str, float]
# #     batch_size: int
# #     num_tokens: int
# #     test_split: float

# #     def to_dataloader(self, sequence_length: int, train: bool) -> DataLoader:
# #         dataset = ProcessDataset(
# #             process_name=self.process,
# #             process_params=self.process_params,
# #             sequence_length=sequence_length,
# #             num_samples=(
# #                 self.num_tokens
# #                 if train
# #                 else math.floor(self.num_tokens * self.test_split)
# #             ),
# #         )
# #         return DataLoader(
# #             dataset=dataset,
# #             collate_fn=process_dataset_collate_fn,
# #             batch_size=self.batch_size,
# #         )


# # @dataclass
# # class Log:
# #     train_loss: float | None
# #     test_loss: float | None
# #     config: "LoggingConfig"

# #     def reset(self):
# #         if self.config.train_loss:
# #             self.train_loss = 0.0
# #         else:
# #             self.train_loss = None

# #         if self.config.test_loss:
# #             self.test_loss = 0.0
# #         else:
# #             self.test_loss = None

# #     def update_metrics(self, train_or_test: Literal["train", "test"], loss: float):
# #         if train_or_test == "train" and self.config.test_loss:
# #             assert self.train_loss is not None
# #             self.train_loss += loss
# #         elif train_or_test == "test" and self.config.train_loss:
# #             assert self.test_loss is not None
# #             self.test_loss += loss
# #         else:
# #             raise ValueError

# #     def persist(self):
# #         if self.config.wandb:
# #             wandb.log(
# #                 {
# #                     k: v
# #                     for k, v in asdict(self).items()
# #                     if v is not None and not isinstance(v, LoggingConfig)
# #                 }
# #             )
# #         if self.config.local is not None:
# #             raise NotImplementedError


# # class LoggingConfig(Config):
# #     local: pathlib.Path | None = None
# #     wandb: bool = True
# #     project_name: str | None
# #     train_loss: bool = True
# #     test_loss: bool = True

# #     def close(self):
# #         if self.wandb:
# #             wandb.finish()
# #         if self.local is not None:
# #             raise NotImplementedError

# #     def init(self) -> Log:
# #         return Log(
# #             config=self,
# #             train_loss=0.0 if self.train_loss else None,
# #             test_loss=0.0 if self.test_loss else None,
# #         )


# # class TrainConfig(Config):
# #     model: RawModelConfig
# #     optimizer: OptimizerConfig
# #     dataset: ProcessDatasetConfig
# #     persistance: PersistanceConfig
# #     logging: LoggingConfig
# #     seed: int
# #     verbose: bool

# #     @model_validator(mode="after")
# #     def validate_model(self):
# #         dataset_process = self.dataset.process
# #         if dataset_process:
# #             process_vocab_len = PROCESS_REGISTRY[dataset_process]().vocab_len
# #             if self.model.d_vocab != process_vocab_len:
# #                 raise ValueError(
# #                     f"Model's d_vocab ({self.model.d_vocab}) doesn't match dataset process's vocab_len ({process_vocab_len})"
# #                 )
# #         return self

# #     def init_logger(self) -> Log:
# #         if self.logging.wandb:
# #             dotenv.load_dotenv()
# #             wandb_api_key = os.environ.get("WANDB_API_KEY", None)
# #             if wandb_api_key is None:
# #                 raise ValueError(
# #                     "To use wandb, set your API key as the environment variable `WANDB_API_KEY`"
# #                 )

# #             wandb.login(key=wandb_api_key)
# #             wandb.init(project=self.logging.project_name, config=self.model_dump())
# #         if self.logging.local is not None:
# #             raise NotImplementedError()
# #         return self.logging.init()
# """
# Complete training_configs.py with KL analysis support integrated.

# This is the full configuration file including all existing configs plus new KL analysis configs.
# """

# from dataclasses import dataclass, field
# from typing import List, Optional, Dict, Any
# import pathlib
# import yaml


# @dataclass
# class NGramAnalysisConfig:
#     """Configuration for n-gram KL divergence analysis during validation."""
#     enabled: bool = True
#     n_values: List[int] = field(default_factory=lambda: [1, 2, 3])
#     return_per_position: bool = True
    
#     def __post_init__(self):
#         """Validate n_values."""
#         if not all(isinstance(n, int) and n >= 1 for n in self.n_values):
#             raise ValueError("n_values must be list of positive integers")
#         if max(self.n_values) > 5:
#             raise ValueError("n_values > 5 may be too slow, recommended max is 3")


# @dataclass
# class MarkovKLAnalysisConfig:
#     """Configuration for Markov process KL divergence analysis during validation."""
#     enabled: bool = True
#     return_per_position: bool = True


# @dataclass
# class KLAnalysisConfig:
#     """Configuration for all KL analysis metrics."""
#     ngram_analysis: NGramAnalysisConfig = field(default_factory=NGramAnalysisConfig)
#     markov_kl_analysis: MarkovKLAnalysisConfig = field(default_factory=MarkovKLAnalysisConfig)


# @dataclass
# class ModelConfig:
#     """Configuration for model architecture."""
#     n_layers: int = 4
#     d_model: int = 256
#     n_ctx: int = 64
#     d_vocab: int = 512
#     n_heads: int = 8
#     d_head: Optional[int] = None
#     d_mlp: Optional[int] = None
#     seed: int = 42


# @dataclass
# class DatasetConfig:
#     """Configuration for dataset."""
#     batch_size: int = 32
#     sequence_length: Optional[int] = None


# @dataclass
# class OptimizerConfig:
#     """Configuration for optimizer."""
#     optimizer_type: str = "adam"
#     learning_rate: float = 0.001
#     betas: tuple = (0.9, 0.999)
#     eps: float = 1e-8
#     weight_decay: float = 0.0


# @dataclass
# class PersistenceConfig:
#     """Configuration for model persistence."""
#     checkpoint_every_n_tokens: int = 50000
#     save_dir: str = "./checkpoints"
#     use_s3: bool = False


# @dataclass
# class LoggingConfig:
#     """Configuration for logging."""
#     log_dir: str = "./logs"
#     log_interval: int = 100
#     log_to_wandb: bool = False
#     wandb_project: Optional[str] = None


# @dataclass
# class Log:
#     """Logger for training metrics."""
    
#     def __init__(self):
#         self.metrics = {}
    
#     def update_metrics(self, train_or_test: str, loss: float = None,
#                       metric_name: str = None, value: float = None):
#         """
#         Update metrics with support for both loss and KL metrics.
        
#         Legacy usage:
#             log.update_metrics("test", loss=0.5)
        
#         New usage:
#             log.update_metrics("test", metric_name="kl_div_ngram_1", value=0.892)
#         """
#         if metric_name is None:
#             key = f"{train_or_test}_loss"
#             self.metrics[key] = loss
#         else:
#             key = f"{train_or_test}_{metric_name}"
#             self.metrics[key] = value
    
#     def persist(self):
#         """Persist metrics."""
#         for metric_name, value in self.metrics.items():
#             print(f"{metric_name}: {value}")
    
#     def reset(self):
#         """Reset metrics for next epoch."""
#         self.metrics = {}
    
#     def __repr__(self):
#         return f"Log({self.metrics})"


# from dataclasses import dataclass, field
# from typing import List, Optional


# # ============================================================================
# # ADD THESE CLASSES TO training_configs.py
# # ============================================================================

# # @dataclass
# # class NGramAnalysisConfig:
# #     """Configuration for n-gram KL divergence analysis during validation."""
# #     enabled: bool = True
# #     n_values: List[int] = field(default_factory=lambda: [1, 2, 3])
# #     return_per_position: bool = True
    
# #     def __post_init__(self):
# #         """Validate n_values."""
# #         if not all(isinstance(n, int) and n >= 1 for n in self.n_values):
# #             raise ValueError("n_values must be list of positive integers")


# # @dataclass
# # class MarkovKLAnalysisConfig:
# #     """Configuration for Markov process KL divergence analysis during validation."""
# #     enabled: bool = True
# #     return_per_position: bool = True


# # @dataclass
# # class KLAnalysisConfig:
# #     """Configuration for all KL analysis metrics."""
# #     ngram_analysis: NGramAnalysisConfig = field(default_factory=NGramAnalysisConfig)
# #     markov_kl_analysis: MarkovKLAnalysisConfig = field(default_factory=MarkovKLAnalysisConfig)

# @dataclass
# class TrainConfig:
#     """Complete training configuration."""
#     model: ModelConfig = field(default_factory=ModelConfig)
#     dataset: DatasetConfig = field(default_factory=DatasetConfig)
#     optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
#     persistance: PersistenceConfig = field(default_factory=PersistenceConfig)
#     logging: LoggingConfig = field(default_factory=LoggingConfig)
    
#     # NEW: KL Analysis configuration
#     kl_analysis: KLAnalysisConfig = field(default_factory=KLAnalysisConfig)
    
#     seed: int = 42
#     verbose: bool = True
    
#     @classmethod
#     def from_yaml(cls, config_path: pathlib.Path):
#         """Load configuration from YAML file."""
#         with open(config_path, 'r') as f:
#             config_dict = yaml.safe_load(f)
        
#         # Extract nested configs
#         model_config = ModelConfig(**config_dict.get('model', {}))
#         dataset_config = DatasetConfig(**config_dict.get('dataset', {}))
#         optimizer_config = OptimizerConfig(**config_dict.get('optimizer', {}))
#         persistence_config = PersistenceConfig(**config_dict.get('persistance', {}))
#         logging_config = LoggingConfig(**config_dict.get('logging', {}))
        
#         # KL Analysis config (with defaults)
#         kl_dict = config_dict.get('kl_analysis', {})
#         ngram_dict = kl_dict.get('ngram_analysis', {})
#         markov_dict = kl_dict.get('markov_kl_analysis', {})
        
#         ngram_config = NGramAnalysisConfig(**ngram_dict)
#         markov_config = MarkovKLAnalysisConfig(**markov_dict)
#         kl_config = KLAnalysisConfig(
#             ngram_analysis=ngram_config,
#             markov_kl_analysis=markov_config
#         )
        
#         return cls(
#             model=model_config,
#             dataset=dataset_config,
#             optimizer=optimizer_config,
#             persistance=persistence_config,
#             logging=logging_config,
#             kl_analysis=kl_config,
#             seed=config_dict.get('seed', 42),
#             verbose=config_dict.get('verbose', True),
#         )
    
#     def init_logger(self) -> Log:
#         """Initialize logger."""
#         return Log()