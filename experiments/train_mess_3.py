#one training batch of size 128, seq len 10 takes 0.01 s on average




from epsilon_transformers.training.configs.model_configs import RawModelConfig
from epsilon_transformers.training.configs.training_configs import (
    LoggingConfig,
    OptimizerConfig,
    PersistanceConfig,
    ProcessDatasetConfig,
    TrainConfig,
    NGramAnalysisConfig,
    MarkovKLAnalysisConfig,
    SimplexAnalysisConfig,
    AnalysisConfig,
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
    n_head=2,
    d_mlp=256,
    n_layers=1,
)


# ============================================================================
# Optimizer Configuration
# ============================================================================

optimizer_config = OptimizerConfig(
    optimizer_type='sgd',
    learning_rate=1e-4,
    weight_decay=0
)


# ============================================================================
# Dataset Configuration
# ============================================================================

dataset_config = ProcessDatasetConfig(
    process='Trun_Mess3',
    process_params={'x': 0.05, 'a': 0.85},
    batch_size=128,
    chunk_size=2048,
    num_tokens=150000,
    sequence_length=10,
    test_split=0.0625,
    test_batch_size=512,
)


# ============================================================================
# Persistence Configuration
# ============================================================================
from pathlib import Path

persistance_config = PersistanceConfig(
    location='local',
    collection_location=Path('models/mess3/single_layer'),
    checkpoint_every_n_tokens=125000
)


# ============================================================================
# Logging Configuration - UPDATED with wandb_api_key
# ============================================================================
logging_config = LoggingConfig(
    project_name="epstrans",
    wandb=True,
    # NEW: Option 1 - Pass API key directly (recommended for testing)
    wandb_api_key="9df77e7cbad36f3323af2ea208aa4027a970df97",
    run_name="trial",# NEW!
    # OR use environment variable: export WANDB_API_KEY="YOUR_KEY"
)



# ============================================================================
# KL Analysis Configuration
# ============================================================================

# kl_analysis_config = KLAnalysisConfig(
#     ngram_analysis=NGramAnalysisConfig(
#         enabled=True,
#         n_values=[1, 2, 3],
#         return_per_position=False,
#     ),
#     markov_kl_analysis=MarkovKLAnalysisConfig(
#         enabled=True,
#         return_per_position=True,
#     ),
# )
analysis_config = AnalysisConfig(
    ngram_analysis=NGramAnalysisConfig(
        enabled=True,
        n_values=[1, 2, 3],
        return_per_position=False,
    ),
    markov_kl_analysis=MarkovKLAnalysisConfig(
        enabled=True,
        return_per_position=True,
    ),
    # NEW: Simplex Analysis Configuration
    simplex_analysis=SimplexAnalysisConfig(
        enabled=True,
        hook_point="blocks.0.hook_resid_post", # Adjust layer index if n_layers changes
        num_samples_for_probe=None
    )
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
    analysis=analysis_config,
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