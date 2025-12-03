import fire
import pathlib
import random
import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
import dotenv
from tqdm import tqdm
from typing import Tuple, Optional
from torch.utils.data import DataLoader
from epsilon_transformers.process.MixedStateTree import MixedStateTree

from epsilon_transformers.persistence import Persister
from epsilon_transformers.training.configs.training_configs import (
    TrainConfig,
    Log,
    ProcessDatasetConfig,
)
from epsilon_transformers.analysis.kl_analysis import MarkovKLAnalyzer, compute_markov_kl_divergence
from epsilon_transformers.analysis.ngram_analysis import NGramAnalyzer, compute_ngram_kl_divergence

from epsilon_transformers.process import Process

from epsilon_transformers.process.processes import PROCESS_REGISTRY

# def torch_to_cupy(tensor):
#     import cupy as cp
#     from torch.utils.dlpack import to_dlpack
#     from cupy import from_dlpack
#     return from_dlpack(to_dlpack(tensor))
    
def get_process_object(process_name: str, process_params: dict):
    """Return an instantiated Process object given name and parameters."""
    process_class = PROCESS_REGISTRY.get(process_name, None)
    if process_class is None:
        raise ValueError(
            f"{process_name!r} is not a recognized process. "
            f"Available processes: {list(PROCESS_REGISTRY.keys())}"
        )
    return process_class(**process_params)

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
    vocab_size: int
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
    
    return ngram_analyzer, markov_analyzer

def _compute_myopic_entropy(val_process:object, n_ctx: int, device: torch.device) -> torch.Tensor:
    """Compute theoretical minimum (myopic) cross-entropy per position for given process."""
    #MSP_tree = mixed_state_tree(process, n_ctx + 1)
    mixed_state_tree = val_process.derive_mixed_state_presentation(depth=n_ctx + 1)
    #MSP_transition_matrix = mixed_state_tree.build_msp_transition_matrix()
    #block_entropy = mixed_state_tree.block_entropy
    myopic_entropy_rate = mixed_state_tree.myopic_entropy
    minimum_cross_entropy = myopic_entropy_rate 
    print(f"myopic entropy rates:{minimum_cross_entropy}")
    return torch.tensor(minimum_cross_entropy, dtype=torch.float32, device=device)

def _compute_relative_losses(loss_tensor: torch.Tensor, minimum_cross_entropy: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute mean loss and relative loss per position.
    loss_tensor: (batch, seq_len)
    """
    per_position_loss = loss_tensor.mean(dim=0)
    #print(per_position_loss)
    relative_loss = per_position_loss / minimum_cross_entropy
    mean_loss = per_position_loss.mean()
    return mean_loss, relative_loss


def _compute_validation_metrics(
    model,
    eval_dataloader:DataLoader,
    device: torch.device,
    log: Log,
    ngram_analyzer: Optional[NGramAnalyzer] = None,
    markov_analyzer: Optional[MarkovKLAnalyzer] = None,
    val_process: Optional[object] = None,
    return_per_position: bool = True,
    minimum_cross_entropy: torch.Tensor=None
) -> Log:
    """Compute validation metrics including loss and KL divergences."""

    model.eval()

    criterion=nn.CrossEntropyLoss(reduction="none")

    all_logits = []
    all_sequences = []

    total_loss = 0.0
    total_relative_loss_per_pos = None
    total_relative_loss = 0.0
    num_batches = 0 

    t_start_eval=time.time()
    print("[eval] starting loop")

    with torch.no_grad():
        t_model_fwd=0.0
        for i, batch in enumerate(eval_dataloader):
            t0 = time.time()

            input_data, target_data= batch
            input_data,target_data = input_data.to(device), target_data.to(device)
            
            logits=model(input_data, return_type="logits")
            loss=criterion(logits.view(-1,logits.size(-1)),target_data.view(-1))
            loss=loss.view(input_data.shape[0],input_data.shape[1])
            
            mean_loss, relative_loss=_compute_relative_losses(loss,minimum_cross_entropy)
            total_loss+=mean_loss.item()
            total_relative_loss = relative_loss.mean().item()

            if total_relative_loss_per_pos is None:
                total_relative_loss_per_pos = torch.zeros_like(relative_loss)
            
            total_relative_loss_per_pos += relative_loss
            num_batches += 1

            all_logits.append(logits)
            all_sequences.append(input_data)
            t1 = time.time()
            t_model_fwd += (t1 - t0)
        print(f"[eval] model forward time: {t_model_fwd:.3f} seconds over {num_batches} batches")    

        if num_batches > 0:
            avg_loss = total_loss / num_batches
            avg_relative_loss_per_pos = total_relative_loss_per_pos / num_batches
            avg_relative_loss=total_relative_loss/num_batches    
            log.update_metrics("test", metric_name="loss", loss=avg_loss)
            log.update_metrics("test", metric_name="relative_loss", loss=avg_relative_loss)
            for i, rel_val in enumerate(avg_relative_loss_per_pos):
                log.update_metrics("test", metric_name=f"relative_loss_{i}", loss=rel_val.item())

            #for kl
            all_logits.append(logits)
            all_sequences.append(input_data)
    
    avg_loss = total_loss / max(num_batches, 1)
    log.update_metrics("test", loss=avg_loss)
    
    if (ngram_analyzer is not None or markov_analyzer is not None) and len(all_logits)>0:
        t_concat_start=time.time()
        all_logits_tensor=torch.cat(all_logits,dim=0)
        all_sequences_tensor=torch.cat(all_sequences,dim=0)
        print(f"[eval] concat: {time.time()-t_concat_start:.3f}s")
        print(f'[kl]computing kl metrics on {len(all_sequences_tensor)} sequences')
        if ngram_analyzer is not None:
            if not ngram_analyzer.prob_tables and ngram_analyzer.count_tables:
                print("[kl] warning: prob_tables are empty, rebuilding from count_tables before computing kl")
                ngram_analyzer.build_prob_tables_from_counts()
            t_ngram_start=time.time()
            if ngram_analyzer.device != device:
                print(f"[kl] moving ngram analyzer to {device}")
                for n in ngram_analyzer.prob_tables:
                    ngram_analyzer.prob_tables[n]=ngram_analyzer.prob_tables[n].to(device)
                ngram_analyzer.device=device
            ngram_metrics=compute_ngram_kl_divergence(
                all_logits_tensor,
                all_sequences_tensor,
                ngram_analyzer=ngram_analyzer,
                n_values=ngram_analyzer.n_grams,
                return_per_position=return_per_position,
            )    
            for metric_name, metric_value in ngram_metrics.items():
                log.update_metrics("test", metric_name=metric_name, loss=metric_value)
            print(f"[kl] ngram kl time: {time.time()-t_ngram_start:.3f}s")

        if markov_analyzer is not None and val_process is not None:
            t_markov_start=time.time()
            markov_metrics=compute_markov_kl_divergence(
                all_logits_tensor,
                all_sequences_tensor,
                process=val_process,
                analyzer=markov_analyzer,
                return_per_position=return_per_position,
            )
            for metric_name, metric_value in markov_metrics.items():
                log.update_metrics("test", metric_name=metric_name, loss=metric_value)
            print(f"[kl] markov kl time: {time.time()-t_markov_start:.3f}s")
    print(f"[eval] total eval time: {time.time()-t_start_eval:.3f}s")
    return log                    



    
    # Compute KL metrics if analyzers available
    # if (ngram_analyzer is not None or markov_analyzer is not None) and len(all_logits) > 0:
    #     all_logits_tensor = torch.cat(all_logits, dim=0)
    #     all_sequences_tensor = torch.cat(all_sequences, dim=0)
    #     ngram_analyzer.build_from_sequences(all_sequences_tensor)
    #     print(f"[KL Analysis] N-gram analyzer (rebuilt) on current eval dataset")
    #     if device.type == 'cuda' :
    #         try:
    #             import cupy as cp
    #             print("[TIMING] Starting CuPy conversion")
    #             torch.cuda.synchronize()
    #             t0 = time.time()
    #             backend_logits = torch_to_cupy(all_logits_tensor)
    #             torch.cuda.synchronize()
    #             t1 = time.time()
    #             print(f"[TIMING] CuPy conversion took {t1 - t0:.3f} seconds for backend_logits")
    #             torch.cuda.synchronize()
    #             t0 = time.time()
    #             backend_sequences = torch_to_cupy(all_sequences_tensor)
    #             torch.cuda.synchronize()
    #             t1 = time.time()
    #             print(f"[TIMING] CuPy conversion took {t1 - t0:.3f} seconds for backend_sequences")
    #             print(f"[KL Analysis] Data moved to CuPy for KL computations")
    #         except Exception as e:
    #             print(f"[KL Analysis] Failed to move data to CuPy: {e}. Using numpy on CPU.")
    #             backend_logits = all_logits_tensor.cpu().numpy()
    #             backend_sequences = all_sequences_tensor.cpu().numpy()  

    #     else:
    #         backend_logits = all_logits_tensor.cpu().numpy()
    #         backend_sequences = all_sequences_tensor.cpu().numpy()
                      
        
    #     # N-gram KL divergences
    #     if ngram_analyzer is not None:
    #         ngram_metrics = compute_ngram_kl_divergence(
    #             backend_logits,
    #             backend_sequences,
    #             ngram_analyzer,
    #             n_values=ngram_analyzer.n_grams,
    #             return_per_position=return_per_position,
    #         )
            
    #         for metric_name, metric_value in ngram_metrics.items():
    #             log.update_metrics("test", metric_name=metric_name, loss=metric_value)
        
    #     # Markov KL divergence
    #     if markov_analyzer is not None and val_process is not None:
    #         markov_metrics = compute_markov_kl_divergence(
    #             backend_logits,
    #             backend_sequences,
    #             process=val_process,
    #             analyzer=markov_analyzer,
    #             return_per_position=return_per_position,
    #         )
            
    #         for metric_name, metric_value in markov_metrics.items():
    #             log.update_metrics("test", metric_name=metric_name, loss=metric_value)
    
    # return log


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
    minimum_cross_entropy: torch.Tensor=None
):
    """Evaluate model, log metrics, and persist checkpoint."""
    eval_dataloader = dataset_config.to_dataloader(
        sequence_length=model.cfg.n_ctx, train=False
    )
    with torch.no_grad():
        _compute_validation_metrics(
            model=model,
            eval_dataloader=eval_dataloader,
            device=device,
            log=log,
            ngram_analyzer=ngram_analyzer,
            markov_analyzer=markov_analyzer,
            val_process=val_process,
            return_per_position=return_per_position,
            minimum_cross_entropy=minimum_cross_entropy
        )
    
    if verbose:
        print(f"[Step {tokens_trained}] Training loss: {log.train_loss:.6f}") 

    # metadata = {
    #     "train_loss": log.train_loss,
    #     "test_loss": log.test_loss,
    # }
    # persister.save_model(model, tokens_trained, metadata=metadata) 
    #print(f"[Step {tokens_trained}] Metrics: {log.metrics}")

    if "train" in log.metrics and log.metrics["train"]:
        persister.save_metrics_to_csv("train", log.metrics["train"], tokens_trained)
    if "test" in log.metrics and log.metrics["test"]:
        persister.save_metrics_to_csv("test", log.metrics["test"], tokens_trained)

    log.persist()
    persister.save_model(model, tokens_trained, metadata=log.metrics)
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
    persister.save_config(config.model_dump())
    
    # Initialize KL analyzers
    val_process = get_process_object(config.dataset.process, config.dataset.process_params)
    minimum_cross_entropy = _compute_myopic_entropy(val_process, model.cfg.n_ctx, device)
    ngram_analyzer, markov_analyzer = _setup_kl_analyzers(
        config=config,
        vocab_size=model.cfg.d_vocab)
    last_action_batch_tokens=0#for ngram analyzer

    # BUILD N-GRAM ANALYZER ON TRAIN DATA
    # if ngram_analyzer is not None:
    #     print(f"[Training] Building N-Gram Analyzer statistics from training data...")
    #     t_ngram_build_start=time.time()
    #     # 1. Collect all training data into one tensor
    #     all_train_sequences = []
    #     for input_data, _ in tqdm(train_dataloader, desc="Collecting Train Data"):
    #         all_train_sequences.append(input_data)
        
    #     # shape[Total_Train_Samples, Seq_Len]
    #     full_train_tensor = torch.cat(all_train_sequences, dim=0)
        
    #     #Move to GPU for fast build (if it fits)
    #     #If OOM, move to 'cpu' instead.
    #     build_device = device
    #     try:
    #         full_train_tensor = full_train_tensor.to(build_device)
    #         ngram_analyzer.build_from_sequences(full_train_tensor)
    #         persister.save_ngram_data(ngram_analyzer, tokens_trained=(full_train_tensor.shape[0]*full_train_tensor.shape[1]))
    #     except RuntimeError as e:
    #         if "out of memory" in str(e):
    #             print("[Warning] OOM during N-Gram build on GPU. Falling back to CPU.")
    #             torch.cuda.empty_cache()
    #             ngram_analyzer.build_from_sequences(full_train_tensor.to('cpu'))
    #             # Move tables back to GPU for eval later
    #             for n in ngram_analyzer.prob_tables:
    #                 ngram_analyzer.prob_tables[n] = ngram_analyzer.prob_tables[n].to(device)
    #                 ngram_analyzer.device = device
    #         else:
    #             raise e
    #     del full_train_tensor
    #     del all_train_sequences
    #     if device.type == 'cuda':
    #         torch.cuda.empty_cache()
    #     print(f"[Training] N-Gram build time: {time.time()-t_ngram_build_start:.3f}s")
    #     _set_random_seed(config.seed)
    #     train_dataloader=config.dataset.to_dataloader(
    #         sequence_length=model.cfg.n_ctx, train=True
    #     )

    model.train()
    tokens_trained_so_far = 0
    train_sequences_since_last_action=[]
    
    # Training loop
    for batch_idx, (input_data, target_data) in enumerate(
        tqdm(train_dataloader, desc="Train Loop")
    ):
        t0 = time.time()
        input_data = input_data.to(device)
        print(input_data.size(0))
        target_data = target_data.to(device)
        if ngram_analyzer is not None:
            train_sequences_since_last_action.append(input_data)
        logits = model(input_data, return_type="logits")
        criterion=nn.CrossEntropyLoss(reduction="none")
        loss_per_token = criterion(logits.view(-1, logits.size(-1)), target_data.view(-1))
        loss_per_token = loss_per_token.view(input_data.size(0), input_data.size(1))
        mean_loss, relative_loss = _compute_relative_losses(loss_per_token, minimum_cross_entropy)
        mean_loss=mean_loss/input_data.size(0)

        log.update_metrics(train_or_test="train", loss=mean_loss.item())
        optimizer.zero_grad()
        mean_loss.backward()
        optimizer.step()
        t1 = time.time()
        print(f"[TIMING] Train batch took {t1 - t0:.3f} seconds")

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
            t0 = time.time()
            if ngram_analyzer is not None and len(train_sequences_since_last_action)>0:
                print(f"[train] merging ngram analyzer count tables from {last_action_batch_tokens} to {tokens_trained_so_far}")
                new_train_tensor=torch.cat(train_sequences_since_last_action,dim=0)
                build_device=device
                try:
                    new_train_tensor=new_train_tensor.to(build_device)
                    temp_analyzer=NGramAnalyzer(vocab_size=model.cfg.d_vocab,n_grams=ngram_analyzer.n_grams)
                    temp_analyzer.build_from_sequences(new_train_tensor)
                    if last_action_batch_tokens==0:
                        prev_data=persister.load_ngram_data(tokens_trained=0,device=device)
                        if prev_data is not None:
                            ngram_analyzer.count_tables=prev_data['count_tables']
                            ngram_analyzer.n_grams=prev_data['n_grams']
                            ngram_analyzer.vocab_size=prev_data['vocab_size']
                            ngram_analyzer.device=device
                            ngram_analyzer.merge_ngram_tables(temp_analyzer.count_tables)
                        else:
                            print("[train] no previous ngram data found, using temp as current")
                            ngram_analyzer.count_tables=temp_analyzer.count_tables
                            ngram_analyzer.n_grams=temp_analyzer.n_grams
                            ngram_analyzer.vocab_size=temp_analyzer.vocab_size
                            ngram_analyzer.device=device
                    else:
                        ngram_analyzer.count_tables=temp_analyzer.count_tables
                        ngram_analyzer.n_grams=temp_analyzer.n_grams
                        ngram_analyzer.vocab_size=temp_analyzer.vocab_size
                        ngram_analyzer.device=device
                    persister.save_ngram_data(ngram_analyzer,tokens_trained=tokens_trained_so_far)
                    print(f"[train] ngram analyzer count tables merged and saved")
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        print("warning: oom during ngram build, falling back to cpu") 
                        torch.cuda.empty_cache()
                        new_train_tensor=new_train_tensor.to('cpu')
                        temp_analyzer = NGramAnalyzer(vocab_size=model.cfg.d_vocab, n_grams=ngram_analyzer.n_grams)
                        temp_analyzer.build_from_sequences(new_train_tensor)
                        
                        if last_action_batch_tokens > 0:
                            prev_data = persister.load_ngram_data(last_action_batch_tokens, device='cpu')
                            if prev_data is not None:
                                ngram_analyzer.count_tables = prev_data['count_tables']
                                ngram_analyzer.n_grams = prev_data['n_grams']
                                ngram_analyzer.vocab_size = prev_data['vocab_size']
                                ngram_analyzer.device = torch.device('cpu')
                                ngram_analyzer.merge_ngram_tables(temp_analyzer.count_tables)
                            else:
                                ngram_analyzer.count_tables = temp_analyzer.count_tables
                                ngram_analyzer.prob_tables = temp_analyzer.prob_tables
                                ngram_analyzer.device = torch.device('cpu')
                        else:
                            ngram_analyzer.count_tables = temp_analyzer.count_tables
                            ngram_analyzer.prob_tables = temp_analyzer.prob_tables
                            ngram_analyzer.device = torch.device('cpu')
                        
                        # Move tables back to GPU for eval
                        for n in ngram_analyzer.prob_tables:
                            ngram_analyzer.prob_tables[n] = ngram_analyzer.prob_tables[n].to(device)
                        ngram_analyzer.device = device
                        
                        persister.save_ngram_data(ngram_analyzer, tokens_trained=tokens_trained_so_far)
                    else:
                        raise e
                
                # Free memory
                del new_train_tensor
                del train_sequences_since_last_action
                train_sequences_since_last_action = []
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
                
                # Update last action batch tracker
                last_action_batch_tokens = tokens_trained_so_far   

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
                minimum_cross_entropy=minimum_cross_entropy)
            t1 = time.time()
            #print(f"[TIMING] Full evaluation step took {t1 - t0:.3f} seconds")
            model.train()
    
    # Final evaluation
    model.eval()
    #build final ngram table for remaining sequences
    if ngram_analyzer is not None and len(train_sequences_since_last_action) > 0:
        print(f"[Training] Building final N-Gram table...")
        new_train_tensor = torch.cat(train_sequences_since_last_action, dim=0)
        
        try:
            new_train_tensor = new_train_tensor.to(device)
            temp_analyzer = NGramAnalyzer(vocab_size=model.cfg.d_vocab, n_grams=ngram_analyzer.n_grams)
            temp_analyzer.build_from_sequences(new_train_tensor)
            
            if last_action_batch_tokens > 0:
                prev_data = persister.load_ngram_data(last_action_batch_tokens, device=str(device))
                if prev_data is not None:
                    ngram_analyzer.count_tables = prev_data['count_tables']
                    ngram_analyzer.n_grams = prev_data['n_grams']
                    ngram_analyzer.vocab_size = prev_data['vocab_size']
                    ngram_analyzer.device = device
                    ngram_analyzer.merge_ngram_tables(temp_analyzer.count_tables)
                else:
                    ngram_analyzer.count_tables = temp_analyzer.count_tables
                    ngram_analyzer.prob_tables = temp_analyzer.prob_tables
            else:
                ngram_analyzer.count_tables = temp_analyzer.count_tables
                ngram_analyzer.prob_tables = temp_analyzer.prob_tables
            
            persister.save_ngram_data(ngram_analyzer, tokens_trained=tokens_trained_so_far)
        except RuntimeError as e:
            if "out of memory" in str(e):
                torch.cuda.empty_cache()
                new_train_tensor = new_train_tensor.to('cpu')
                temp_analyzer = NGramAnalyzer(vocab_size=model.cfg.d_vocab, n_grams=ngram_analyzer.n_grams)
                temp_analyzer.build_from_sequences(new_train_tensor)
                
                if last_action_batch_tokens > 0:
                    prev_data = persister.load_ngram_data(last_action_batch_tokens, device='cpu')
                    if prev_data is not None:
                        ngram_analyzer.count_tables = prev_data['count_tables']
                        ngram_analyzer.n_grams = prev_data['n_grams']
                        ngram_analyzer.vocab_size = prev_data['vocab_size']
                        ngram_analyzer.device = torch.device('cpu')
                        ngram_analyzer.merge_ngram_tables(temp_analyzer.count_tables)
                
                for n in ngram_analyzer.prob_tables:
                    ngram_analyzer.prob_tables[n] = ngram_analyzer.prob_tables[n].to(device)
                ngram_analyzer.device = device
                
                persister.save_ngram_data(ngram_analyzer, tokens_trained=tokens_trained_so_far)
    
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
        minimum_cross_entropy=minimum_cross_entropy
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


# import re
# from epsilon_transformers.process.datasets import ProcessDataset

# def finetune(
#     checkpoint_path: str,
#     config_path:str,
#     seed: int,
#     wandb_run_id: Optional[str]=None
# ):
#     """
#         Finetune a model from a checkpoint.
        
#         Args:
#             checkpoint_path: Path to the .pt checkpoint file
#             config_path: Path to the config.yaml or train_config.json
#             seed: Random seed for the training dataloader (must match original run for correct resumption)
#             wandb_run_id: Optional WandB run ID to resume logging to the same run
#         """
#     checkpoint_path = pathlib.Path(checkpoint_path)
#     match=re.search(r'tokens_(\d+)',checkpoint_path.name)
#     if not match:
#         raise ValueError(f"could not parse tokens trained from chkpt name: {checkpoint_path.name}")
    
#     tokens_trained_start=int(match.group(1))

#     if str(config_path).endswith('.yaml') or str(config_path).endswith('.yml'):
#         config=TrainConfig.from_yaml(config_path)
#     else:
#         import json
#         with open(config_path, 'r') as f:
#             config_dict=json.load(f)
#             config=TrainConfig(**config_dict)
#         device=torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
#         print(f"[Finetune] Using device: {device}")
#         print(f"[finetune] resuming from {tokens_trained_start} tokens trained")

#     if wandb_run_id:
#         print("[finetune] resuming wandb run id:", wandb_run_id)
#         dotenv.load_dotenv()
#         wandb.login(key=config.logging.wandb_api_key or os.environ.get("WANDB_API_KEY"))     

#         wandb.init(
#             project=config.logging.project_name,
#             id=wandb_run_id,
#             resume="must",
#             config=config.model_dump()

#         )     
#         log=Log(
#             config=config.logging,
#             train_loss=0.0,
#             test_loss=0.0

#         )
#     else:
#         print("[finetune] starting new wandb run")  
#         log = config.init_logger()
#     persister=Persister(save_dir=config.persistance.collection_location)
#     print(f"[finetune] loading model from {checkpoint_path}")
#     model=persister.load_model(checkpoint_path,device=device)
#     model.train()
#     optimizer=config.optimizer.from_model(model=model, device=device)
#     print(f"[finetune] creating dataloaders")
#     _set_random_seed(seed)
#     train_dataloader = config.dataset.to_dataloader(
#         sequence_length=model.cfg.n_ctx, train=True
#     )
#     seq_len=config.dataset.sequence_length
#     samples_processed = tokens_trained_start // seq_len
#     emissions_to_burn = samples_processed * (seq_len + 1)
    
#     dataset = train_dataloader.dataset
#     if isinstance(dataset, ProcessDataset):
#         print(f"[Finetune] Fast-forwarding generator by {emissions_to_burn} emissions (this may take a moment)...")
#         # Consume the iterator efficiently
#         for _ in tqdm(range(emissions_to_burn), desc="Burning dataset history"):
#             next(dataset.samples)
#     else:
#         print("[Warning] Dataset is not ProcessDataset. Deterministic resumption cannot be guaranteed.")

#     # 5. Setup Analyzers
#     val_process = get_process_object(config.dataset.process, config.dataset.process_params)
#     minimum_cross_entropy = _compute_myopic_entropy(val_process, model.cfg.n_ctx, device)
#     ngram_analyzer, markov_analyzer = _setup_kl_analyzers(config=config, vocab_size=model.cfg.d_vocab)

#     if ngram_analyzer:
#         print("[Finetune] Note: N-Gram statistics are empty. Rebuild from full history if strictly needed.")

#     # 6. Training Loop
#     print("[Finetune] Starting training loop...")
#     tokens_per_batch = config.dataset.batch_size * seq_len
#     start_batch_idx = tokens_trained_start // tokens_per_batch

#     # Iterate through the now fast-forwarded dataloader
#     for i, (input_data, target_data) in enumerate(tqdm(train_dataloader, desc="Finetune")):
#         batch_idx = start_batch_idx + i
        
#         t0 = time.time()
#         input_data = input_data.to(device)
#         target_data = target_data.to(device)

#         # Forward
#         logits = model(input_data, return_type="logits")
#         criterion = nn.CrossEntropyLoss(reduction="none")
#         loss = criterion(logits.view(-1, logits.size(-1)), target_data.view(-1))
#         loss = loss.view(input_data.size(0), input_data.size(1))
        
#         # Loss Calculation
#         mean_loss, relative_loss = _compute_relative_losses(loss, minimum_cross_entropy)
#         mean_loss = mean_loss / input_data.size(0) # Normalize by batch size if needed by optimizer
        
#         # Log
#         log.update_metrics(train_or_test="train", loss=mean_loss.item())
        
#         # Backward
#         optimizer.zero_grad()
#         mean_loss.backward()
#         optimizer.step()
        
#         # Calculate Tokens Trained
#         tokens_trained_so_far = _calculate_tokens_trained(
#             batch_size=config.dataset.batch_size,
#             sequence_len=model.cfg.n_ctx,
#             batch_idx=batch_idx
#         )

#         # Checkpoint & Eval
#         if _check_if_action_batch(
#             perform_action_every_n_tokens=config.persistance.checkpoint_every_n_tokens,
#             batch_size=config.dataset.batch_size,
#             batch_idx=batch_idx,
#             sequence_len=model.cfg.n_ctx,
#         ):
#             model.eval()
#             _evaluate_log_and_persist(
#                 persister=persister,
#                 model=model,
#                 log=log,
#                 verbose=config.verbose,
#                 device=device,
#                 dataset_config=config.dataset,
#                 tokens_trained=tokens_trained_so_far,
#                 ngram_analyzer=ngram_analyzer,
#                 markov_analyzer=markov_analyzer,
#                 val_process=val_process,
#                 minimum_cross_entropy=minimum_cross_entropy
#             )
#             model.train()

#     # Final Save
#     config.logging.close()
#     return model, log  
