"""
Complete train.py with integrated KL analysis metrics support.

This is the main training script that combines model training with KL analysis logging.
"""

import fire
import pathlib
import random
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from typing import Tuple, Optional

from epsilon_transformers.persistence import Persister, MetricLogger
from epsilon_transformers.training.configs.training_configs import (
    TrainConfig, Log
)

from epsilon_transformers.analysis.kl_analysis import MarkovKLAnalyzer, compute_markov_kl_divergence
from epsilon_transformers.analysis.ngram_analysis import NGramAnalyzer, compute_ngram_kl_divergence


def _set_random_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _calculate_tokens_trained(batch_size: int, sequence_len: int, batch_idx: int) -> int:
    """Calculate total tokens trained up to this batch."""
    tokens_per_batch = batch_size * sequence_len
    total_tokens_trained = (batch_idx + 1) * tokens_per_batch
    return total_tokens_trained


def _check_if_action_batch(
    perform_action_every_n_tokens: int,
    batch_size: int,
    sequence_len: int,
    batch_idx: int,
) -> bool:
    """Check if this batch should trigger a checkpoint/evaluation."""
    tokens_per_batch = batch_size * sequence_len
    assert (
        perform_action_every_n_tokens >= tokens_per_batch
    ), "perform_action_every_n_tokens must be >= tokens_per_batch"
    perform_action_every_n_batches = perform_action_every_n_tokens // tokens_per_batch
    return (batch_idx + 1) % perform_action_every_n_batches == 0


def _setup_kl_analyzers(
    config: TrainConfig,
    vocab_size: int,
    eval_dataloader,
    val_process: Optional[object] = None,
) -> Tuple[Optional[NGramAnalyzer], Optional[MarkovKLAnalyzer]]:
    """Initialize KL divergence analyzers."""
    ngram_analyzer = None
    markov_analyzer = None
    
    # Check config for KL analysis enabled
    ngram_enabled = (
        hasattr(config, 'kl_analysis') and 
        config.kl_analysis.ngram_analysis.enabled
    )
    markov_enabled = (
        hasattr(config, 'kl_analysis') and
        config.kl_analysis.markov_kl_analysis.enabled
    )
    
    # Initialize N-gram analyzer
    if ngram_enabled:
        n_values = config.kl_analysis.ngram_analysis.n_values
        ngram_analyzer = NGramAnalyzer(vocab_size=vocab_size, n_grams=n_values)
        
        # Build n-gram frequencies from eval data
        eval_sequences = []
        for batch in eval_dataloader:
            if isinstance(batch, tuple):
                sequences = batch[0]
            elif isinstance(batch, dict):
                sequences = batch.get('input_ids', batch.get('sequences', batch[0]))
            else:
                sequences = batch
            eval_sequences.append(sequences)
        
        eval_sequences_tensor = torch.cat(eval_sequences, dim=0)
        ngram_analyzer.build_from_sequences(eval_sequences_tensor)
        print(f"[KL Analysis] N-gram analyzer initialized with n_values={n_values}")
    
    # Initialize Markov analyzer
    if markov_enabled and val_process is not None:
        markov_analyzer = MarkovKLAnalyzer(vocab_size=vocab_size)
        print("[KL Analysis] Markov KL analyzer initialized")
    
    return ngram_analyzer, markov_analyzer


def _compute_validation_metrics(
    model,
    eval_dataloader,
    device: torch.device,
    log: Log,
    ngram_analyzer: Optional[NGramAnalyzer] = None,
    markov_analyzer: Optional[MarkovKLAnalyzer] = None,
    val_process: Optional[object] = None,
    return_per_position: bool = True,
) -> Log:
    """Compute validation metrics including loss and KL divergences."""
    model.eval()
    
    all_logits = []
    all_sequences = []
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch in tqdm(eval_dataloader, desc="Eval Loop", leave=False):
            if isinstance(batch, tuple):
                input_data, target_data = batch
            elif isinstance(batch, dict):
                input_data = batch.get('input_ids', batch.get('sequences'))
                target_data = batch.get('target_ids', batch.get('labels', input_data))
            else:
                input_data = batch
                target_data = batch
            
            input_data = input_data.to(device)
            target_data = target_data.to(device)
            
            # Compute loss
            loss = model(input_data, return_type="loss")
            total_loss += loss.item()
            num_batches += 1
            
            # Get logits for KL analysis
            logits = model(input_data, return_type="logits")
            all_logits.append(logits.cpu())
            all_sequences.append(input_data.cpu())
    
    # Update log with validation loss
    avg_loss = total_loss / max(num_batches, 1)
    log.update_metrics("test", loss=avg_loss)
    
    # Compute KL metrics if analyzers available
    if ngram_analyzer is not None or markov_analyzer is not None:
        if len(all_logits) > 0:
            all_logits_tensor = torch.cat(all_logits, dim=0)
            all_sequences_tensor = torch.cat(all_sequences, dim=0)
            
            # N-gram KL divergences
            if ngram_analyzer is not None:
                ngram_metrics = compute_ngram_kl_divergence(
                    all_logits_tensor,
                    all_sequences_tensor,
                    ngram_analyzer,
                    n_values=ngram_analyzer.n_grams,
                    return_per_position=return_per_position,
                )
                
                for metric_name, metric_value in ngram_metrics.items():
                    log.update_metrics("test", metric_name=metric_name, value=metric_value)
            
            # Markov KL divergence
            if markov_analyzer is not None and val_process is not None:
                markov_metrics = compute_markov_kl_divergence(
                    all_logits_tensor,
                    all_sequences_tensor,
                    val_process,
                    analyzer=markov_analyzer,
                    return_per_position=return_per_position,
                )
                
                for metric_name, metric_value in markov_metrics.items():
                    log.update_metrics("test", metric_name=metric_name, value=metric_value)
    
    return log


def _evaluate_log_and_persist(
    persister: Persister,
    model,
    verbose: bool,
    log: Log,
    device: torch.device,
    tokens_trained: int,
    eval_dataloader,
    ngram_analyzer: Optional[NGramAnalyzer] = None,
    markov_analyzer: Optional[MarkovKLAnalyzer] = None,
    val_process: Optional[object] = None,
    return_per_position: bool = True,
):
    """Evaluate model, log metrics, and persist checkpoint."""
    _compute_validation_metrics(
        model=model,
        eval_dataloader=eval_dataloader,
        device=device,
        log=log,
        ngram_analyzer=ngram_analyzer,
        markov_analyzer=markov_analyzer,
        val_process=val_process,
        return_per_position=return_per_position,
    )
    
    if verbose:
        print(f"Metrics: {log.metrics}")
    
    log.persist()
    persister.save_model(model, tokens_trained)
    log.reset()


def train_model(config: TrainConfig, return_per_position: bool = True) -> Tuple:
    """Train transformer model with KL analysis metrics."""
    device = torch.device(
        "mps" if torch.backends.mps.is_available()
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    
    print(f"[Training] Using device: {device}")
    
    _set_random_seed(config.seed)
    
    # Initialize model and optimizer
    model = config.model.to_hooked_transformer(device=device, seed=config.seed)
    optimizer = config.optimizer.from_model(model=model, device=device)
    
    # Create data loaders
    train_dataloader = config.dataset.to_dataloader(
        sequence_length=model.cfg.n_ctx, train=True
    )
    eval_dataloader = config.dataset.to_dataloader(
        sequence_length=model.cfg.n_ctx, train=False
    )
    
    # Initialize persistence and logging
    persister = Persister(
        save_dir=config.persistance.save_dir,
        use_s3=config.persistance.use_s3
    )
    log = config.init_logger()
    
    # Initialize KL analyzers
    ngram_analyzer, markov_analyzer = _setup_kl_analyzers(
        config=config,
        vocab_size=model.cfg.d_vocab,
        eval_dataloader=eval_dataloader,
        val_process=getattr(config, 'val_process', None),
    )
    
    model.train()
    tokens_trained_so_far = 0
    
    # Training loop
    for batch_idx, (input_data, target_data) in enumerate(
        tqdm(train_dataloader, desc="Train Loop")
    ):
        input_data, target_data = input_data.to(device), target_data.to(device)
        loss = model(input_data, return_type="loss")
        log.update_metrics(train_or_test="train", loss=loss.item())
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        tokens_trained_so_far = _calculate_tokens_trained(
            batch_size=config.dataset.batch_size,
            sequence_len=model.cfg.n_ctx,
            batch_idx=batch_idx,
        )
        
        # Checkpoint and evaluation
        if _check_if_action_batch(
            perform_action_every_n_tokens=config.persistance.checkpoint_every_n_tokens,
            batch_size=config.dataset.batch_size,
            batch_idx=batch_idx,
            sequence_len=model.cfg.n_ctx,
        ):
            model.eval()
            _evaluate_log_and_persist(
                persister=persister,
                model=model,
                log=log,
                verbose=config.verbose,
                device=device,
                tokens_trained=tokens_trained_so_far,
                eval_dataloader=eval_dataloader,
                ngram_analyzer=ngram_analyzer,
                markov_analyzer=markov_analyzer,
                val_process=getattr(config, 'val_process', None),
                return_per_position=return_per_position,
            )
            model.train()
    
    # Final evaluation
    model.eval()
    _evaluate_log_and_persist(
        persister=persister,
        model=model,
        log=log,
        verbose=config.verbose,
        device=device,
        tokens_trained=tokens_trained_so_far,
        eval_dataloader=eval_dataloader,
        ngram_analyzer=ngram_analyzer,
        markov_analyzer=markov_analyzer,
        val_process=getattr(config, 'val_process', None),
        return_per_position=return_per_position,
    )
    
    return model, log


def _main(config_path: pathlib.Path):
    """Main entry point."""
    config = TrainConfig.from_yaml(config_path)
    train_model(config)


if __name__ == "__main__":
    fire.Fire(_main)
# """
# Modified train.py with integrated KL divergence analysis metrics (per-position enabled).

# This version uses the eval dataloader for both loss and KL divergence metric logging,
# maintaining the same checkpoint intervals as the original implementation.

# Now returns per-position KL divergences in addition to overall metrics.
# """



# import fire  # type: ignore
# import pathlib
# import random
# import numpy as np
# import torch
# import torch.nn.functional as F
# from tqdm import tqdm
# from typing import Tuple, Optional
# from torch.utils.data import DataLoader
# from transformer_lens import HookedTransformer  # type: ignore

# from epsilon_transformers.persistence import Persister
# from epsilon_transformers.training.configs.training_configs import (
#     TrainConfig,
#     ProcessDatasetConfig,
#     Log,
# )
# from epsilon_transformers.analysis.kl_analysis import MarkovKLAnalyzer, compute_markov_kl_divergence
# from epsilon_transformers.analysis.ngram_analysis import NGramAnalyzer, compute_ngram_kl_divergence


# # TODO: Bug of outputting num_of_tokens_trained on rather than num_of_tokens_seen
# # TODO: Put flag for overwriting (either don't do it, or have a logger throw a warning)
# # TODO: Bug where the last final loss outputs train_loss of 0
# # TODO: Use logger library for logging
# # TODO: Make Log into a singleton
# # TODO: Add TQDM to all of this
# # TODO: Generalize train_model so that it doesn't depend on the HookedTransformer internal loss function
# # TODO: move _check_if_action_batch asserts to a config validator
# # TODO: Add option to resume from checkpoint
# # TODO: Review best practices regarding seed setting
# # TODO: Test on GPUs
# # TODO: Add DP


# def _set_random_seed(seed):
#     """Set random seeds for reproducibility."""
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)


# def _calculate_tokens_trained(
#     batch_size: int,
#     sequence_len: int,
#     batch_idx: int,
# ) -> int:
#     """Calculate total tokens trained up to this batch."""
#     tokens_per_batch = batch_size * sequence_len
#     total_tokens_trained = (batch_idx + 1) * tokens_per_batch
#     return total_tokens_trained


# def _check_if_action_batch(
#     perform_action_every_n_tokens: int,
#     batch_size: int,
#     sequence_len: int,
#     batch_idx: int,
# ) -> bool:
#     """Check if this batch should trigger a checkpoint/evaluation."""
#     tokens_per_batch = batch_size * sequence_len
#     assert (
#         perform_action_every_n_tokens >= tokens_per_batch
#     ), "perform_action_every_n_tokens must be greater than or equal to tokens_per_batch"
#     perform_action_every_n_batches = perform_action_every_n_tokens // tokens_per_batch
#     return (batch_idx + 1) % perform_action_every_n_batches == 0


# def _setup_kl_analyzers(
#     config: TrainConfig,
#     vocab_size: int,
#     eval_dataloader: DataLoader,
#     val_process: Optional[object] = None,
# ) -> Tuple[Optional[NGramAnalyzer], Optional[MarkovKLAnalyzer]]:
#     """
#     Initialize KL divergence analyzers.
    
#     Args:
#         config: Training configuration
#         vocab_size: Size of model vocabulary
#         eval_dataloader: Evaluation dataloader for building n-gram frequencies
#         val_process: Process object for Markov analysis (optional)
        
#     Returns:
#         tuple: (ngram_analyzer, markov_analyzer)
#     """
#     ngram_analyzer = None
#     markov_analyzer = None
    
#     # Check if KL analysis is enabled in config
#     ngram_enabled = (
#         hasattr(config, 'ngram_analysis') and 
#         getattr(config.ngram_analysis, 'enabled', False)
#     )
#     markov_enabled = (
#         hasattr(config, 'markov_kl_analysis') and 
#         getattr(config.markov_kl_analysis, 'enabled', False)
#     )
    
#     # Initialize N-gram analyzer and build frequencies from eval data
#     if ngram_enabled:
#         n_values = getattr(config.ngram_analysis, 'n_values', [1, 2, 3])
#         ngram_analyzer = NGramAnalyzer(
#             vocab_size=vocab_size,
#             n_grams=n_values
#         )
        
#         # Build n-gram frequencies from eval data
#         eval_sequences = []
        
#         for batch in eval_dataloader:
#             if isinstance(batch, tuple):
#                 sequences = batch[0]  # (input_data, target_data)
#             elif isinstance(batch, dict):
#                 sequences = batch.get('input_ids', batch.get('sequences', batch[0]))
#             else:
#                 sequences = batch
            
#             eval_sequences.append(sequences)
        
#         # Concatenate all sequences
#         eval_sequences_tensor = torch.cat(eval_sequences, dim=0)
        
#         # Build the n-gram model
#         ngram_analyzer.build_from_sequences(eval_sequences_tensor)
        
#         print(f"[KL Analysis] N-gram analyzer initialized with n_values={n_values}")
    
#     # Initialize Markov analyzer
#     if markov_enabled and val_process is not None:
#         markov_analyzer = MarkovKLAnalyzer(vocab_size=vocab_size)
#         print("[KL Analysis] Markov KL analyzer initialized")
    
#     return ngram_analyzer, markov_analyzer


# def _compute_validation_metrics(
#     model: HookedTransformer,
#     eval_dataloader: DataLoader,
#     device: torch.device,
#     log: Log,
#     ngram_analyzer: Optional[NGramAnalyzer] = None,
#     markov_analyzer: Optional[MarkovKLAnalyzer] = None,
#     val_process: Optional[object] = None,
#     return_per_position: bool = True,
# ) -> Log:
#     """
#     Compute validation metrics including loss and KL divergences.
    
#     This function processes the entire eval dataloader once to compute:
#     - Validation loss
#     - N-gram KL divergences (if enabled)
#     - Markov process KL divergence (if enabled)
#     - Per-position KL divergences (if enabled)
    
#     Args:
#         model: The transformer model
#         eval_dataloader: Evaluation dataloader
#         device: Device (CPU/GPU)
#         log: Logger object
#         ngram_analyzer: Optional NGramAnalyzer instance
#         markov_analyzer: Optional MarkovKLAnalyzer instance
#         val_process: Optional Process object for Markov analysis
#         return_per_position: Whether to compute and log per-position metrics
        
#     Returns:
#         Updated log object with metrics
#     """
#     model.eval()
    
#     all_logits = []
#     all_sequences = []
#     total_loss = 0.0
#     num_batches = 0
    
#     with torch.no_grad():
#         for batch in tqdm(eval_dataloader, desc="Eval Loop", leave=False):
#             # Extract data from batch
#             if isinstance(batch, tuple):
#                 input_data, target_data = batch
#             elif isinstance(batch, dict):
#                 input_data = batch.get('input_ids', batch.get('sequences'))
#                 target_data = batch.get('target_ids', batch.get('labels', input_data))
#             else:
#                 input_data = batch
#                 target_data = batch
            
#             input_data = input_data.to(device)
#             target_data = target_data.to(device)
            
#             # Compute loss using model's built-in loss computation
#             loss = model(input_data, return_type="loss")
#             total_loss += loss.item()
#             num_batches += 1
            
#             # Get logits for KL analysis
#             logits = model(input_data, return_type="logits")
#             all_logits.append(logits.cpu())
#             all_sequences.append(input_data.cpu())
    
#     # Update log with validation loss
#     avg_loss = total_loss / max(num_batches, 1)
#     log.update_metrics("test", avg_loss)
    
#     # Compute KL divergence metrics if analyzers are available
#     if ngram_analyzer is not None or markov_analyzer is not None:
#         if len(all_logits) > 0:
#             all_logits_tensor = torch.cat(all_logits, dim=0)
#             all_sequences_tensor = torch.cat(all_sequences, dim=0)
            
#             # Compute n-gram KL divergences
#             if ngram_analyzer is not None:
#                 ngram_metrics = compute_ngram_kl_divergence(
#                     all_logits_tensor,
#                     all_sequences_tensor,
#                     ngram_analyzer,
#                     n_values=ngram_analyzer.n_grams,
#                     return_per_position=return_per_position,
#                 )
                
#                 # Log n-gram metrics
#                 for metric_name, metric_value in ngram_metrics.items():
#                     log.update_metrics("test", metric_value, metric_name=metric_name)
            
#             # Compute Markov KL divergence
#             if markov_analyzer is not None and val_process is not None:
#                 markov_metrics = compute_markov_kl_divergence(
#                     all_logits_tensor,
#                     all_sequences_tensor,
#                     val_process,
#                     analyzer=markov_analyzer,
#                     return_per_position=return_per_position,
#                 )
                
#                 # Log Markov metrics
#                 for metric_name, metric_value in markov_metrics.items():
#                     log.update_metrics("test", metric_value, metric_name=metric_name)
    
#     return log


# def _evaluate_log_and_persist(
#     dataset_config: ProcessDatasetConfig,
#     persister: Persister,
#     model: HookedTransformer,
#     verbose: bool,
#     log: Log,
#     device: torch.device,
#     tokens_trained: int,
#     eval_dataloader: DataLoader,
#     ngram_analyzer: Optional[NGramAnalyzer] = None,
#     markov_analyzer: Optional[MarkovKLAnalyzer] = None,
#     val_process: Optional[object] = None,
#     return_per_position: bool = True,
# ):
#     """
#     Evaluate model, log metrics, and persist checkpoint.
    
#     This uses the pre-initialized eval_dataloader for all metric computation.
    
#     Args:
#         dataset_config: Dataset configuration (kept for compatibility)
#         persister: Model persister
#         model: The transformer model
#         verbose: Whether to print verbose output
#         log: Logger object
#         device: Device (CPU/GPU)
#         tokens_trained: Number of tokens trained so far
#         eval_dataloader: Pre-initialized evaluation dataloader
#         ngram_analyzer: Optional NGramAnalyzer instance
#         markov_analyzer: Optional MarkovKLAnalyzer instance
#         val_process: Optional Process object for Markov analysis
#         return_per_position: Whether to compute and log per-position metrics
#     """
#     # Compute all metrics using eval dataloader
#     _compute_validation_metrics(
#         model=model,
#         eval_dataloader=eval_dataloader,
#         device=device,
#         log=log,
#         ngram_analyzer=ngram_analyzer,
#         markov_analyzer=markov_analyzer,
#         val_process=val_process,
#         return_per_position=return_per_position,
#     )
    
#     if verbose:
#         print(f"This is the log\n{log}")
    
#     log.persist()
#     log.reset()
#     persister.save_model(model, tokens_trained)


# def train_model(config: TrainConfig, return_per_position: bool = True) -> Tuple[HookedTransformer, Log]:
#     """
#     Train a transformer model with optional KL divergence analysis.
    
#     Args:
#         config: Training configuration
#         return_per_position: Whether to compute and log per-position KL metrics
        
#     Returns:
#         tuple: (trained model, logger)
#     """
#     device = torch.device(
#         "mps"
#         if torch.backends.mps.is_available()
#         else ("cuda" if torch.cuda.is_available() else "cpu")
#     )
    
#     print(f"[Training] Using device: {device}")
    
#     _set_random_seed(config.seed)
    
#     model = config.model.to_hooked_transformer(device=device, seed=config.seed)
#     optimizer = config.optimizer.from_model(model=model, device=device)
#     train_dataloader = config.dataset.to_dataloader(
#         sequence_length=model.cfg.n_ctx, train=True
#     )
    
#     # Create eval dataloader once for reuse throughout training
#     eval_dataloader = config.dataset.to_dataloader(
#         sequence_length=model.cfg.n_ctx, train=False
#     )
    
#     persister = config.persistance.init()
#     log = config.init_logger()
    
#     # Initialize KL analyzers if enabled
#     ngram_analyzer, markov_analyzer = _setup_kl_analyzers(
#         config=config,
#         vocab_size=model.cfg.d_vocab,
#         eval_dataloader=eval_dataloader,
#         val_process=getattr(config, 'val_process', None),
#     )
    
#     model.train()
#     tokens_trained_so_far = 0
    
#     for batch_idx, (input_data, target_data) in enumerate(
#         tqdm(train_dataloader, desc="Train Loop")
#     ):
#         input_data, target_data = input_data.to(device), target_data.to(device)
#         loss = model(input_data, return_type="loss")
#         log.update_metrics(train_or_test="train", loss=loss.item())
        
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
        
#         tokens_trained_so_far = _calculate_tokens_trained(
#             batch_size=config.dataset.batch_size,
#             sequence_len=model.cfg.n_ctx,
#             batch_idx=batch_idx,
#         )
        
#         if _check_if_action_batch(
#             perform_action_every_n_tokens=config.persistance.checkpoint_every_n_tokens,
#             batch_size=config.dataset.batch_size,
#             batch_idx=batch_idx,
#             sequence_len=model.cfg.n_ctx,
#         ):
#             model.eval()
#             _evaluate_log_and_persist(
#                 dataset_config=config.dataset,
#                 persister=persister,
#                 model=model,
#                 log=log,
#                 verbose=config.verbose,
#                 device=device,
#                 tokens_trained=tokens_trained_so_far,
#                 eval_dataloader=eval_dataloader,
#                 ngram_analyzer=ngram_analyzer,
#                 markov_analyzer=markov_analyzer,
#                 val_process=getattr(config, 'val_process', None),
#                 return_per_position=return_per_position,
#             )
#             log.reset()
#             model.train()
    
#     # Final evaluation
#     model.eval()
#     _evaluate_log_and_persist(
#         dataset_config=config.dataset,
#         persister=persister,
#         model=model,
#         log=log,
#         verbose=config.verbose,
#         device=device,
#         tokens_trained=tokens_trained_so_far,
#         eval_dataloader=eval_dataloader,
#         ngram_analyzer=ngram_analyzer,
#         markov_analyzer=markov_analyzer,
#         val_process=getattr(config, 'val_process', None),
#         return_per_position=return_per_position,
#     )
    
#     config.logging.close()
#     return model, log


# def _main(config_path: pathlib.Path):
#     """Main entry point for training."""
#     config: TrainConfig = TrainConfig.from_yaml(config_path)
#     train_model(config)


# if __name__ == "__main__":
#     fire.Fire(_main)