"""
Updated train_mess3.py - Config only, uses corrected training_configs.py

Includes wandb_api_key support and proper KL analysis configuration.
"""

from transformers.models import FalconForSequenceClassification
from epsilon_transformers.training.configs.model_configs import RawModelConfig
from epsilon_transformers.training.configs.training_configs import (
    LoggingConfig,
    OptimizerConfig,
    PersistanceConfig,
    ProcessDatasetConfig,
    TrainConfig,
    NGramAnalysisConfig,
    MarkovKLAnalysisConfig,
    KLAnalysisConfig,
)
from epsilon_transformers.training.train import train_model


# ============================================================================
# Model Configuration
# ============================================================================

model_config = RawModelConfig(
    d_vocab=3,
    d_model=64,
    n_ctx=10,
    d_head=8,
    n_head=1,
    d_mlp=12,
    n_layers=4,
)


# ============================================================================
# Optimizer Configuration
# ============================================================================

optimizer_config = OptimizerConfig(
    optimizer_type='sgd',
    learning_rate=1e-2,
    weight_decay=0
)


# ============================================================================
# Dataset Configuration
# ============================================================================

dataset_config = ProcessDatasetConfig(
    process='Mess3',
    process_params={'x': 0.5, 'a': 0.85},
    batch_size=64,
    num_tokens=100000,
    sequence_length=10,
    test_split=0.1
)


# ============================================================================
# Persistence Configuration
# ============================================================================
from pathlib import Path

persistance_config = PersistanceConfig(
    location='local',
    collection_location=Path('models/mess3'),
    checkpoint_every_n_tokens=1000
)


# ============================================================================
# Logging Configuration - UPDATED with wandb_api_key
# ============================================================================

logging_config = LoggingConfig(
    project_name="epstrans",
    wandb=True,
    # NEW: Option 1 - Pass API key directly (recommended for testing)
    wandb_api_key="9df77e7cbad36f3323af2ea208aa4027a970df97",  # NEW!
    # OR use environment variable: export WANDB_API_KEY="YOUR_KEY"
    train_loss=True,
    test_loss=True,
)


# ============================================================================
# KL Analysis Configuration
# ============================================================================

kl_analysis_config = KLAnalysisConfig(
    ngram_analysis=NGramAnalysisConfig(
        enabled=False,
        n_values=[1, 2, 3],
        return_per_position=True,
    ),
    markov_kl_analysis=MarkovKLAnalysisConfig(
        enabled=False,
        return_per_position=True,
    ),
)


# ============================================================================
# Complete Training Configuration
# ============================================================================

mock_config = TrainConfig(
    model=model_config,
    optimizer=optimizer_config,
    dataset=dataset_config,
    persistance=persistance_config,
    logging=logging_config,
    kl_analysis=kl_analysis_config,
    verbose=True,
    seed=42
)


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    try:
        train_model(mock_config)
    except ValueError as e:
        print(f"Configuration Error: {e}")
        print("\nFix: Either:")
        print("  1. Set wandb_api_key in LoggingConfig")
        print("  2. OR set WANDB_API_KEY environment variable")
        raise