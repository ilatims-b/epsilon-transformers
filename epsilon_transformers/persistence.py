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