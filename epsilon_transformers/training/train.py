"""
Updated train.py - Works with corrected training_configs.py

Key changes:
- Uses Log class from training_configs (not external)
- Supports KL analysis with updated config
- Proper error handling for wandb setup
"""

import fire
import pathlib
import random
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from typing import Tuple, Optional
from torch.utils.data import DataLoader

from epsilon_transformers.persistence import Persister
from epsilon_transformers.training.configs.training_configs import (
    TrainConfig,
    Log,
    ProcessDatasetConfig,
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


def _setup_persister(config: TrainConfig):
    return config.persistance.init()


def _setup_kl_analyzers(
    config: TrainConfig,
    vocab_size: int,
    # eval_dataloader,
    # val_process: Optional[object] = None,
    # dataset_config: ProcessDatasetConfig = None,
) -> Tuple[Optional[NGramAnalyzer], Optional[MarkovKLAnalyzer]]:
    """Initialize KL divergence analyzers if enabled."""
    ngram_analyzer = None
    markov_analyzer = None
    
    # Check config for KL analysis enabled
    if not hasattr(config, 'kl_analysis'):
        return None, None
    if config.kl_analysis.ngram_analysis.enabled:
        n_values = config.kl_analysis.ngram_analysis.n_values
        ngram_analyzer = NGramAnalyzer(vocab_size=vocab_size, n_grams=n_values)
    
    if config.kl_analysis.markov_kl_analysis.enabled:
        markov_analyzer = MarkovKLAnalyzer(vocab_size=vocab_size)
    
    # ngram_enabled = config.kl_analysis.ngram_analysis.enabled
    # markov_enabled = config.kl_analysis.markov_kl_analysis.enabled
    
    # # Initialize N-gram analyzer
    # if ngram_enabled:
    #     n_values = config.kl_analysis.ngram_analysis.n_values
    #     ngram_analyzer = NGramAnalyzer(vocab_size=vocab_size, n_grams=n_values)
    #     eval_dataloader = dataset_config.to_dataloader(
    #     sequence_length=dataset_config.sequence_length, train=False
    # )
    #     # Build n-gram frequencies from eval data
    #     eval_sequences = []
    #     for batch in eval_dataloader:
    #         if isinstance(batch, tuple):
    #             sequences = batch[0]
    #         elif isinstance(batch, dict):
    #             sequences = batch.get('input_ids', batch.get('sequences', batch[0]))
    #         else:
    #             sequences = batch
    #         eval_sequences.append(sequences)
        
    #     eval_sequences_tensor = torch.cat(eval_sequences, dim=0)
    #     ngram_analyzer.build_from_sequences(eval_sequences_tensor)
    #     print(f"[KL Analysis] N-gram analyzer initialized with n_values={n_values}")
    
    # # Initialize Markov analyzer
    # if markov_enabled and val_process is not None:
    #     markov_analyzer = MarkovKLAnalyzer(vocab_size=vocab_size)
    #     print("[KL Analysis] Markov KL analyzer initialized")
    
    return ngram_analyzer, markov_analyzer


def _compute_validation_metrics(
    model,
    eval_dataloader:DataLoader,
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
        # for input_data, target_data in tqdm(eval_dataloader, desc="Eval Loop"):
        #     input_data, target_data = input_data.to(device), target_data.to(device)
        #     loss = model(input_data, return_type="loss")
        #     log.update_metrics("test", loss.item())
        for batch in tqdm(eval_dataloader, desc="Eval Loop", leave=False):
            # if isinstance(batch, tuple):
            #     input_data, target_data = batch
                # print("is tuple")
            # elif isinstance(batch, dict):
            #     input_data = batch.get('input_ids', batch.get('sequences'))
            #     target_data = batch.get('target_ids', batch.get('labels', input_data))
            #     print("is dict")
            # else:
            #     input_data = batch
            #     target_data = batch
            input_data, target_data= batch
            input_data = input_data.to(device)
            
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
    if (ngram_analyzer is not None or markov_analyzer is not None) and len(all_logits) > 0:
        all_logits_tensor = torch.cat(all_logits, dim=0)
        all_sequences_tensor = torch.cat(all_sequences, dim=0)
        ngram_analyzer.build_from_sequences(all_sequences_tensor)
        print(f"[KL Analysis] N-gram analyzer (rebuilt) on current eval dataset")
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
                log.update_metrics("test", metric_name=metric_name, loss=metric_value)
        
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
                log.update_metrics("test", metric_name=metric_name, loss=metric_value)
    
    return log


def _evaluate_log_and_persist(
    persister,
    model,
    verbose: bool,
    log: Log,
    device: torch.device,
    tokens_trained: int,
    dataset_config: ProcessDatasetConfig,
    ngram_analyzer: Optional[NGramAnalyzer] = None,
    markov_analyzer: Optional[MarkovKLAnalyzer] = None,
    val_process: Optional[object] = None,
    return_per_position: bool = True,
):
    """Evaluate model, log metrics, and persist checkpoint."""
    eval_dataloader = dataset_config.to_dataloader(
        sequence_length=model.cfg.n_ctx, train=False
    )
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
        print(f"[Step {tokens_trained}] Training loss: {log.train_loss:.6f}") 

    metadata = {
        "train_loss": log.train_loss,
        "test_loss": log.test_loss,
    }
    persister.save_model(model, tokens_trained, metadata=metadata) 
    #     print(f"[Step {tokens_trained}] Metrics: {log.metrics}")
    
    # log.persist()
    # persister.save_model(model, tokens_trained, metadata=log.metrics)
    log.reset()


def train_model(config: TrainConfig, return_per_position: bool = True) -> Tuple:
    """Train transformer model with KL analysis metrics."""
    device = torch.device(
        "mps" if torch.backends.mps.is_available()
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    
    print(f"[Training] Using device: {device}")
    
    _set_random_seed(config.seed)
    
    # Initialize logger (handles wandb setup)
    log = config.init_logger()
    
    # Initialize model and optimizer
    model = config.model.to_hooked_transformer(device=device, seed=config.seed)
    optimizer = config.optimizer.from_model(model=model, device=device)
    print(f"[Training] Creating dataloaders...")
    # Create data loaders
    train_dataloader = config.dataset.to_dataloader(
        sequence_length=model.cfg.n_ctx, train=True
    )

    print(f"[Training] Dataloaders created")
    # Initialize persistence
    persister = _setup_persister(config)
    
    # Initialize KL analyzers
    val_process = getattr(config.dataset, 'process', None)
    ngram_analyzer, markov_analyzer = _setup_kl_analyzers(
        config=config,
        vocab_size=model.cfg.d_vocab)
    
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
                dataset_config=config.dataset,
                tokens_trained=tokens_trained_so_far,
                ngram_analyzer=ngram_analyzer,
                markov_analyzer=markov_analyzer,
                val_process=val_process,
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
        dataset_config=config.dataset,
        ngram_analyzer=ngram_analyzer,
        markov_analyzer=markov_analyzer,
        val_process=val_process,
        return_per_position=return_per_position,
    )
    
    # Close logger
    config.logging.close()
    
    return model, log


def _main(config_path: pathlib.Path):
    """Main entry point."""
    config = TrainConfig.from_yaml(config_path)
    train_model(config)


if __name__ == "__main__":
    fire.Fire(_main)