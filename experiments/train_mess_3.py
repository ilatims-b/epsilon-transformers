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
import torch

# ============================================================================
# Model Configuration
# ============================================================================

model_config = RawModelConfig(
    d_vocab=6,
    d_model=128,
    n_ctx=10,
    d_head=16,
    n_head=1,
    d_mlp=512,
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

# dataset_config = ProcessDatasetConfig(
#     process='Mixed_Mess3',
#     process_params={'x1': 0.15, 'a1': 0.6, 'x2': 0.15, 'a2': 0.75},
#     batch_size=128,
#     num_tokens=150000,
#     sequence_length=10,
#     test_split=0.0625,
#     chunk_size=2048,
#     test_batch_size=1250,
#     steady_state=[1/6, 1/6, 1/6, 1/6, 1/6, 1/6]
# )
dataset_config = ProcessDatasetConfig(
    process='Mess3',  # Placeholder name, actual logic driven by mixing=True
    process_params={}, # Ignored for sub-processes when mixing=True
    batch_size=10,   
    num_tokens=5000,  
    sequence_length=10,
    test_split=0.2,
    chunk_size=1000,
    test_batch_size=10,
    steady_state=None, # Let processes compute their own steady states
    
    # --- Mixing Configuration ---
    mixing=True,
    mixing_params={
        'processes': [
            ('Mess3', {'x': 0.15, 'a': 0.6}, {0: 0, 1: 1, 2: 2}),
            ('Mess3', {'x': 0.15, 'a': 0.75}, {0: 3, 1: 4, 2: 5})
        ],
        'switch_times': [5,10], 
        'switch_prob': [0.5, 1.0],
        'state_mode': 'same' 
    }
)


# ============================================================================
# Persistence Configuration
# ============================================================================
from pathlib import Path

persistance_config = PersistanceConfig(
    location='local',
    collection_location=Path('models/trial'),
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
    run_name="trial_mixed",
    relative_loss=False
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
        enabled=False,
        n_values=[1, 2, 3],
        return_per_position=False,
    ),
    markov_kl_analysis=MarkovKLAnalysisConfig(
        enabled=False,
        return_per_position=False,
    ),
    # NEW: Simplex Analysis Configuration
    simplex_analysis=SimplexAnalysisConfig(
        enabled=False,
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
    seed=42,
    do_eval=False
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