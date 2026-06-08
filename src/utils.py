from pathlib import Path
import random
import numpy as np
import torch


def set_global_seed(random_state: int = 42) -> None:
    """
    Sets random seeds for reproducibility.

    If PyTorch is installed, it also sets torch seeds.
    """
    random.seed(random_state)
    np.random.seed(random_state)

   
    torch.manual_seed(random_state)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(random_state)
        torch.cuda.manual_seed_all(random_state)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def ensure_dir(path: str | Path) -> Path:
    """
    Creates a directory if it does not exist.
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path