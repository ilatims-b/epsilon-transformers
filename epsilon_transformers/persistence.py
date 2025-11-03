import torch
import pathlib
from typing import Optional, Dict, Any
import csv
import pandas as pd  

class Persister:
    """Handles model persistence and checkpoint management."""
    
    def __init__(self, save_dir: str = "./checkpoints"):
        """
        Initialize persister.
        
        Args:
            save_dir: Directory to save checkpoints
        """
        self.save_dir = pathlib.Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
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

    def save_metrics_to_csv(self, split: str, metrics: Dict[str, float], step: int):
        """
        Append metrics to train_logs.csv or test_logs.csv.

        Args:
            split: 'train' or 'test'
            metrics: dictionary of metric_name -> value
            step: current training step (e.g., tokens_trained)
        """
        assert split in ["train", "test"], "split must be 'train' or 'test'"

        filename = self.save_dir / f"{split}_logs.csv"
        fieldnames = ["step", "metric_name", "value"]

        # Create file with header if it doesn't exist
        file_exists = filename.exists()
        with open(filename, mode="a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()

            for metric_name, val in metrics.items():
                writer.writerow({
                    "step": step,
                    "metric_name": metric_name,
                    "value": val
                })

        print(f"[Persister] Logged {len(metrics)} {split} metrics to {filename.name}")

    def load_metrics_csv(self, split: str) -> Optional[pd.DataFrame]:
        """
        Load persisted CSV metrics as a DataFrame.
        Returns None if file doesn't exist.
        """
        filename = self.save_dir / f"{split}_logs.csv"
        if not filename.exists():
            print(f"[Persister] No {split}_logs.csv found at {self.save_dir}")
            return None
        df = pd.read_csv(filename)
        print(f"[Persister] Loaded {len(df)} entries from {filename.name}")
        return df    