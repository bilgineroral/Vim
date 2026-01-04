from typing import Tuple, Union
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pytorch_lightning as pl
import json, random

def random_spanning_mask(
    grid_shape: Tuple[int, int],
    n: int,
    k: Tuple[int, int] = (1, 1),
    seed: int = None,
) -> np.ndarray:
    """
    Random visible patches; True=masked, False=visible.

    Args:
        grid_shape: (Hp, Wp)
        n: exact number of visible patches (clamped to [0, Hp*Wp])
        k: (k1, k2): spanning block size
        device: torch device

    Returns:
        mask: boolean NumPy array of shape (Hp, Wp), True = masked, False = visible
    """
    if k != (1, 1):
        raise NotImplementedError("Currently only k=(1,1) is supported.")
    if n < 0:
        raise ValueError("n must be non-negative")

    Hp, Wp = grid_shape
    total = Hp * Wp

    if total == 0:
        return np.ones((Hp, Wp), dtype=bool)

    n = min(n, total)
    if n == 0:
        return np.ones((Hp, Wp), dtype=bool)
    if n == total:
        return np.zeros((Hp, Wp), dtype=bool)

    # Choose n unique visible positions using NumPy
    if seed is not None:
        rng = np.random.default_rng(seed)
        idx = rng.choice(total, size=n, replace=False)
    else:
        idx = np.random.choice(total, size=n, replace=False)

    visible_flat = np.zeros(total, dtype=bool)
    visible_flat[idx] = True

    visible = visible_flat.reshape(Hp, Wp)
    mask = ~visible
    return mask

def repeat(x: torch.Tensor, rH: int, rW: int) -> torch.Tensor:
    """
    Args:
        x: (B, H, W) or (H, W), either torch.Tensor or np.ndarray
        rH: repeat multiplier for H
        rW: repeat multiplier for W

    Returns:
        Tiled object of the same type as input:
        - torch.Tensor: (B, H*rH, W*rW)
        - np.ndarray:   (B, H*rH, W*rW)
    """
    if torch.is_tensor(x):
        if rW > 1:
            x = x.repeat_interleave(rW, dim=-1)
        if rH > 1:
            x = x.repeat_interleave(rH, dim=-2)
        return x

    elif isinstance(x, np.ndarray):
        if rW > 1:
            x = np.repeat(x, repeats=rW, axis=-1)
        if rH > 1:
            x = np.repeat(x, repeats=rH, axis=-2)
        return x

    else:
        raise TypeError(f"Unsupported type: {type(x)}. Expected torch.Tensor or np.ndarray.")

def seed_worker(worker_id: int) -> None:
    ws = torch.initial_seed() % (2**32)
    np.random.seed(ws)
    random.seed(ws)

def load_data(dataset_path: str):
    X_np = np.load(dataset_path, mmap_mode="r")
    if X_np.ndim != 4:
        raise ValueError(f"Expected (N,C,H,W), got {X_np.shape}")
    
    return X_np

class VectorFieldDataset(Dataset):
    def __init__(self, X: Union[torch.Tensor, np.ndarray], 
                 stats_path: str,
                 mask_shape: Tuple[int, int] = (16, 16),
                 vrl: int = 4, vrh: int = 16, 
                 alpha=0.0, gamma=0.0, k=(1,1),
                 base_seed: int = None):
        if not ((torch.is_tensor(X) or isinstance(X, np.ndarray)) and X.ndim == 4):
            raise ValueError(f"Expected 4D tensor or numpy array (N,C,H,W), got {getattr(X,'shape',None)}")
        self.X = X
        with open(stats_path, "r") as f:
            stats = json.load(f)
        self.mean = torch.tensor(stats["mean"], dtype=torch.float32).view(-1, 1, 1) # (C, 1, 1)
        self.std = torch.tensor(stats["std"], dtype=torch.float32).view(-1, 1, 1) # (C, 1, 1)
        self.N = X.shape[0]
        self.mask_shape = mask_shape
        self.k = k
        self.vrl = vrl
        self.vrh = vrh
        self.vr = torch.arange(self.vrl, self.vrh + 1)
        mid = (self.vrl + self.vrh) / 2.0
        weights = (self.vrh - self.vr + 1).float().pow(alpha) * torch.exp(gamma * (self.vr - mid))
        self.probs = weights / weights.sum()
        self.base_seed = base_seed

    def __len__(self):
        return self.N

    def __getitem__(self, i: int):
        if self.base_seed is not None:
            g = torch.Generator()
            g.manual_seed(self.base_seed + i)
        else:
            g = None
        idx = torch.multinomial(self.probs, num_samples=1, generator=g).item()

        x = self.X[i] # (C, H, W)
        x = torch.from_numpy(x.copy()).to(torch.float32)
        x.sub_(self.mean).div_(self.std + 1e-6)
        
        v = self.vr[idx].item()
        coarse_mask = random_spanning_mask(
            grid_shape=self.mask_shape,
            n=v,
            k=self.k,
            seed=(self.base_seed + i) if self.base_seed is not None else None,
        ) # bool, True=masked, False=visible
        coarse_mask = torch.from_numpy(coarse_mask)
        ups = x.shape[1] // coarse_mask.shape[0]
        fine_mask = repeat(coarse_mask, ups, ups)
        masked_data = x * (~fine_mask).to(x.dtype)

        return {"x": x, "x_masked": masked_data, "mask": fine_mask, "idx": i}

class VFDataModule(pl.LightningDataModule):
    def __init__(self, train_ds, val_ds, seed: int, loader_kwargs: dict):
        super().__init__()
        self.train_ds = train_ds
        self.val_ds = val_ds
        self.seed = seed
        self.loader_kwargs = loader_kwargs

    def train_dataloader(self):
        rank = int(self.trainer.global_rank)
        g = torch.Generator()
        g.manual_seed(self.seed + 10_000 * rank)
        return DataLoader(
            self.train_ds,
            shuffle=True,
            drop_last=True,
            worker_init_fn=seed_worker,
            generator=g,
            **self.loader_kwargs
        )

    def val_dataloader(self):
        rank = int(self.trainer.global_rank)
        g = torch.Generator()
        g.manual_seed(self.seed + 10_000 * rank)
        return DataLoader(
            self.val_ds,
            shuffle=False,
            drop_last=False,
            worker_init_fn=seed_worker,
            generator=g,
            **self.loader_kwargs
        )

class VFEvalModule(pl.LightningDataModule):
    def __init__(self, eval_ds, seed: int, loader_kwargs: dict):
        super().__init__()
        self.eval_ds = eval_ds
        self.seed = seed
        self.loader_kwargs = loader_kwargs

    def test_dataloader(self):
        rank = int(self.trainer.global_rank)
        g = torch.Generator()
        g.manual_seed(self.seed + 10_000 * rank)
        return DataLoader(
            self.eval_ds,
            shuffle=False,
            drop_last=False,
            worker_init_fn=seed_worker,
            generator=g,
            **self.loader_kwargs
        )
