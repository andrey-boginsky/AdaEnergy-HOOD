import os
import random

import numpy as np
import torch

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"  # или ":16:8"

def set_seed(seed: int = 42):
    """
    Configure full reproducibility for:
        - Python
        - NumPy
        - PyTorch
        - CUDA
        - DataLoader workers

    IMPORTANT:
        Full determinism may slightly reduce performance,
        but is strongly recommended for research and OOD
        benchmarking.
    """

    # ========================================================
    # PYTHON RANDOM
    # ========================================================

    random.seed(seed)

    # ========================================================
    # NUMPY RANDOM
    # ========================================================

    np.random.seed(seed)

    # ========================================================
    # PYTORCH RANDOM
    # ========================================================

    torch.manual_seed(seed)

    # ========================================================
    # CUDA RANDOM
    # ========================================================

    torch.cuda.manual_seed(seed)

    torch.cuda.manual_seed_all(seed)

    # ========================================================
    # PYTHON HASH SEED
    # ========================================================

    os.environ["PYTHONHASHSEED"] = str(seed)

    # ========================================================
    # CUDNN SETTINGS
    # ========================================================

    # Force deterministic CUDA algorithms
    torch.backends.cudnn.deterministic = True

    # Disable auto-benchmarking because it introduces
    # non-determinism
    torch.backends.cudnn.benchmark = False

    # ========================================================
    # PYTORCH DETERMINISTIC ALGORITHMS
    # ========================================================

    torch.use_deterministic_algorithms(True)

    print(f"Global seed set to: {seed}")


# ============================================================
# DATALOADER WORKER SEEDING
# ============================================================

def seed_worker(worker_id: int):
    """
    Ensure deterministic behavior inside each DataLoader worker.

    This is important when:
        - num_workers > 0
        - shuffling is enabled
        - augmentations are used
    """

    worker_seed = torch.initial_seed() % 2**32

    np.random.seed(worker_seed)

    random.seed(worker_seed)