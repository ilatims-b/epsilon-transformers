#
"""
Complete persistance.py with KL analysis metric logging support.
"""

import torch
import pathlib
from typing import Optional, Dict, Any


class Persister:
    """Handles model persistence and checkpoint management."""
    
    def __init__(self, save_dir: str = "./checkpoints", use_s3: bool = False):
        """
        Initialize persister.
        
        Args:
            save_dir: Directory to save checkpoints
            use_s3: Whether to use S3 for storage
        """
        self.save_dir = pathlib.Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.use_s3 = use_s3
        self.checkpoint_count = 0
    
    def save_model(self, model: Any, tokens_trained: int, metadata: Optional[Dict] = None):
        """
        Save model checkpoint.
        
        Args:
            model: Model to save
            tokens_trained: Number of tokens trained so far
            metadata: Optional metadata to save with checkpoint
        """
        checkpoint_num = self.checkpoint_count
        checkpoint_path = self.save_dir / f"checkpoint_{checkpoint_num}_tokens_{tokens_trained}.pt"
        
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'tokens_trained': tokens_trained,
            'checkpoint_number': checkpoint_num,
            'metadata': metadata or {}
        }
        
        torch.save(checkpoint, checkpoint_path)
        self.checkpoint_count += 1
        
        print(f"[Persister] Saved checkpoint {checkpoint_num} at {checkpoint_path}")
    
    def load_model(self, model: Any, checkpoint_path: str) -> Dict[str, Any]:
        """
        Load model from checkpoint.
        
        Args:
            model: Model to load weights into
            checkpoint_path: Path to checkpoint file
            
        Returns:
            Checkpoint dictionary with metadata
        """
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        print(f"[Persister] Loaded checkpoint from {checkpoint_path}")
        print(f"[Persister] Tokens trained: {checkpoint['tokens_trained']}")
        
        return checkpoint
    
    def get_latest_checkpoint(self) -> Optional[pathlib.Path]:
        """Get path to latest checkpoint if it exists."""
        checkpoints = list(self.save_dir.glob("checkpoint_*.pt"))
        if not checkpoints:
            return None
        return max(checkpoints, key=lambda x: x.stat().st_mtime)


class MetricLogger:
    """Handles metric logging for training."""
    
    def __init__(self, log_dir: str = "./logs", log_to_wandb: bool = False,
                 wandb_project: Optional[str] = None):
        """
        Initialize metric logger.
        
        Args:
            log_dir: Directory to save logs
            log_to_wandb: Whether to log to wandb
            wandb_project: Wandb project name
        """
        self.log_dir = pathlib.Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_to_wandb = log_to_wandb
        self.wandb_project = wandb_project
        
        self.metrics_history = []
        
        if log_to_wandb:
            try:
                import wandb
                wandb.init(project=wandb_project)
                self.wandb = wandb
                print(f"[MetricLogger] Initialized wandb logging to project '{wandb_project}'")
            except ImportError:
                print("[MetricLogger] wandb not installed, disabling wandb logging")
                self.log_to_wandb = False
    
    def log_metrics(self, metrics: Dict[str, float], step: int):
        """
        Log metrics.
        
        Args:
            metrics: Dictionary of metric names to values
            step: Current training step
        """
        # Store in memory
        log_entry = {'step': step, 'metrics': metrics}
        self.metrics_history.append(log_entry)
        
        # Log to wandb if enabled
        if self.log_to_wandb:
            self.wandb.log(metrics, step=step)
        
        # Print to console
        metric_str = " | ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        print(f"[Step {step}] {metric_str}")
    
    def save_metrics(self):
        """Save metrics to file."""
        import json
        metrics_path = self.log_dir / "metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(self.metrics_history, f, indent=2)
        print(f"[MetricLogger] Saved metrics to {metrics_path}")
    
    def close(self):
        """Close logger."""
        self.save_metrics()
        if self.log_to_wandb:
            self.wandb.finish()


# For compatibility with existing code
def init_persister(config) -> Persister:
    """Initialize persister from config."""
    return Persister(
        save_dir=config.persistance.save_dir,
        use_s3=config.persistance.use_s3
    )


def init_metric_logger(config) -> MetricLogger:
    """Initialize metric logger from config."""
    return MetricLogger(
        log_dir=config.logging.log_dir,
        log_to_wandb=config.logging.log_to_wandb,
        wandb_project=config.logging.wandb_project
    )
# #
# """
# Complete persistance.py with KL analysis metric logging support.
# """

# import torch
# import pathlib
# from typing import Optional, Dict, Any


# class Persister:
#     """Handles model persistence and checkpoint management."""
    
#     def __init__(self, save_dir: str = "./checkpoints", use_s3: bool = False):
#         """
#         Initialize persister.
        
#         Args:
#             save_dir: Directory to save checkpoints
#             use_s3: Whether to use S3 for storage
#         """
#         self.save_dir = pathlib.Path(save_dir)
#         self.save_dir.mkdir(parents=True, exist_ok=True)
#         self.use_s3 = use_s3
#         self.checkpoint_count = 0
    
#     def save_model(self, model: Any, tokens_trained: int, metadata: Optional[Dict] = None):
#         """
#         Save model checkpoint.
        
#         Args:
#             model: Model to save
#             tokens_trained: Number of tokens trained so far
#             metadata: Optional metadata to save with checkpoint
#         """
#         checkpoint_num = self.checkpoint_count
#         checkpoint_path = self.save_dir / f"checkpoint_{checkpoint_num}_tokens_{tokens_trained}.pt"
        
#         checkpoint = {
#             'model_state_dict': model.state_dict(),
#             'tokens_trained': tokens_trained,
#             'checkpoint_number': checkpoint_num,
#             'metadata': metadata or {}
#         }
        
#         torch.save(checkpoint, checkpoint_path)
#         self.checkpoint_count += 1
        
#         print(f"[Persister] Saved checkpoint {checkpoint_num} at {checkpoint_path}")
    
#     def load_model(self, model: Any, checkpoint_path: str) -> Dict[str, Any]:
#         """
#         Load model from checkpoint.
        
#         Args:
#             model: Model to load weights into
#             checkpoint_path: Path to checkpoint file
            
#         Returns:
#             Checkpoint dictionary with metadata
#         """
#         checkpoint = torch.load(checkpoint_path)
#         model.load_state_dict(checkpoint['model_state_dict'])
        
#         print(f"[Persister] Loaded checkpoint from {checkpoint_path}")
#         print(f"[Persister] Tokens trained: {checkpoint['tokens_trained']}")
        
#         return checkpoint
    
#     def get_latest_checkpoint(self) -> Optional[pathlib.Path]:
#         """Get path to latest checkpoint if it exists."""
#         checkpoints = list(self.save_dir.glob("checkpoint_*.pt"))
#         if not checkpoints:
#             return None
#         return max(checkpoints, key=lambda x: x.stat().st_mtime)

#     def save_metrics_json(self, metrics: Dict[str,Any]) :
#         with open(self.metrics_json, "w") as f:
#             import json
#             json.dump(metrics,f, indent=2)
#         print(f"[Persister] Saved metrics to {self.metrics_json}")

#     def append_csv_metric(self, csv_path:pathlib.Path,step: int, metrics: Dict[str,float]):
#         with open(csv_path,"a", newline="") as f:
#             writer=csv.writer(f)
#             for name, val in metrics.items():
#                 writer.writerrow([step,name,val])
#     def log_kl_metrics(self,step:int, kl_metrics:Dict[str, float]):
#         self.append_csv_metric(self.kl_csv, step, kl_metrics)
#     def log_ngram_metrics(self, step: int, ngram_metrics: Dict[str, float]):
#         """Save n-gram metrics to ngram_metrics.csv."""
#         self.append_csv_metric(self.ngram_csv, step, ngram_metrics)                 


# class MetricLogger:
#     """Handles metric logging for console+wandb"""
    
#     def __init__(self, log_dir: str = "./logs", log_to_wandb: bool = False,
#                  wandb_project: Optional[str] = None):
#         """
#         Initialize metric logger.
        
#         Args:
#             log_dir: Directory to save logs
#             log_to_wandb: Whether to log to wandb
#             wandb_project: Wandb project name
#         """
#         self.log_dir = pathlib.Path(log_dir)
#         self.log_dir.mkdir(parents=True, exist_ok=True)
#         self.log_to_wandb = log_to_wandb
#         self.wandb_project = wandb_project
        
#         self.metrics_history = []
        
#         if log_to_wandb:
#             try:
#                 import wandb
#                 wandb.init(project=wandb_project)
#                 self.wandb = wandb
#                 print(f"[MetricLogger] Initialized wandb logging to project '{wandb_project}'")
#             except ImportError:
#                 print("[MetricLogger] wandb not installed, disabling wandb logging")
#                 self.log_to_wandb = False
    
#     def log_metrics(self, metrics: Dict[str, float], step: int):
#         """
#         Log metrics.
        
#         Args:
#             metrics: Dictionary of metric names to values
#             step: Current training step
#         """
#         # Store in memory
#         log_entry = {'step': step, 'metrics': metrics}
#         self.metrics_history.append(log_entry)
        
#         # Log to wandb if enabled
#         if self.log_to_wandb:
#             self.wandb.log(metrics, step=step)
        
#         # Print to console
#         metric_str = " | ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
#         print(f"[Step {step}] {metric_str}")
    
#     def flush_to_json(self):
#         """Save metrics.json snapshot."""
#         path = self.log_dir / "metrics.json"
#         with open(path, "w") as f:
#             json.dump(self.metrics_history, f, indent=2)
#         print(f"[MetricLogger] metrics.json saved at {path}")

#     def close(self):
#         """Close wandb session."""
#         self.flush_to_json()
#         if self.log_to_wandb:
#             self.wandb.finish()



# # For compatibility with existing code
# def init_persister(config) -> Persister:
#     """Initialize persister from config."""
#     return Persister(
#         save_dir=config.persistance.save_dir,
#         use_s3=config.persistance.use_s3
#     )


# def init_metric_logger(config) -> MetricLogger:
#     """Initialize metric logger from config."""
#     return MetricLogger(
#         log_dir=config.logging.log_dir,
#         log_to_wandb=config.logging.log_to_wandb,
#         wandb_project=config.logging.wandb_project
#     )