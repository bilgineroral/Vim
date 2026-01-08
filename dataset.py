from typing import Tuple, Optional, Dict, Any
import math, torch
from torch.utils.data import Dataset, DataLoader, DistributedSampler
import torchvision.transforms.functional as TVF
import numpy as np
import pytorch_lightning as pl
import json

def random_spanning_mask(
    grid_shape: Tuple[int, int],
    n: int,
    seed: int = None
) -> np.ndarray:
    """
    Random visible patches; True=masked, False=visible.

    Key property:
      mask1 is a deterministic function of (seed, grid_shape, n1)
    """
    if n < 0:
        raise ValueError("n1 must be non-negative")

    Hp, Wp = grid_shape
    total = Hp * Wp

    if total == 0:
        m = np.ones((Hp, Wp), dtype=bool)
        return m

    n = min(n, total)

    # deterministic RNG
    rng = np.random.default_rng(seed) if seed is not None else np.random.default_rng()

    # one ordering drives everything -> mask1 independent of n2 presence/value
    perm = rng.permutation(total)

    idx1 = perm[:n]
    visible_flat = np.ones(total, dtype=bool)
    visible_flat[idx1] = False
    mask = visible_flat.reshape(Hp, Wp)
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

def aug_rotate_vectors(x: torch.Tensor, theta: float) -> torch.Tensor:
    c, s = float(np.cos(theta)), float(np.sin(theta))
    u, v = x[0], x[1]
    up = c * u - s * v
    vp = s * u + c * v
    return torch.stack([up, vp], dim=0)

def aug_add_mag_noise(x: torch.Tensor, std: float, g: Optional[torch.Generator] = None) -> torch.Tensor:
    r = torch.linalg.norm(x, dim=0, keepdim=True).clamp_min(1e-6)
    noise = std * torch.randn(r.shape, generator=g, dtype=x.dtype, device=x.device)
    r_new = r + noise
    return x * (r_new / r)

def aug_gaussian_blur(x: torch.Tensor, ksize: int, sigma: float) -> torch.Tensor:
    if ksize % 2 == 0: ksize += 1
    return TVF.gaussian_blur(x, kernel_size=[ksize, ksize], sigma=[sigma, sigma])

def aug_background(x: torch.Tensor, du: float, dv: float) -> torch.Tensor:
    return torch.cat([(x[0] + du).unsqueeze(0), (x[1] + dv).unsqueeze(0)], dim=0)

class VectorFieldDataset(Dataset):
    def __init__(
        self,
        dataset_path: str, 
        stats_path: str,
        mask_shape: Tuple[int, int] = (16, 16),
        vrl: int = 4, vrh: int = 16, 
        base_seed: Optional[int] = None,
        aug: Optional[Dict[str, Any]] = None,
    ):
        self.dataset_path = dataset_path
        self.stats_path = stats_path
        self.mask_shape = mask_shape
        self.vrl, self.vrh = vrl, vrh
        self.base_seed = base_seed
        self._epoch = torch.zeros((), dtype=torch.int64).share_memory_()

        X = np.load(self.dataset_path, mmap_mode="r")
        if X.ndim != 4:
            raise ValueError(f"Expected (N,C,H,W), got {X.shape}")
        self.X = X
        self.N, self.C, self.H, self.W = X.shape

        with open(self.stats_path, "r") as f:
            stats = json.load(f)
        self.mean = torch.tensor(stats["mean"], dtype=torch.float32).view(self.C, 1, 1)
        self.std = torch.tensor(stats["std"], dtype=torch.float32).view(self.C, 1, 1)

        self.vr = np.arange(self.vrl, self.vrh + 1, dtype=np.int64)
        self.probs = np.ones_like(self.vr, dtype=np.float64)
        self.probs /= self.probs.sum()

        self.aug = aug or {}
        self.rot_prob = float(self.aug.get("rot_prob", 0.0))
        self.rot_max_rad = float(self.aug.get("rot_max_deg", 30.0)) * (np.pi / 180.0)

        self.noise_prob = float(self.aug.get("noise_prob", 0.0))
        self.noise_std = float(self.aug.get("noise_std", 1.0))

        self.blur_prob = float(self.aug.get("blur_prob", 0.0))
        self.blur_ksize = int(self.aug.get("blur_ksize", 9))
        self.blur_sigma = tuple(self.aug.get("blur_sigma", (0.6, 1.2)))

        self.bg_prob = float(self.aug.get("bg_prob", 0.0))
        self.bg_factor = float(self.aug.get("bg_factor", 0.5))
    
    def set_epoch(self, epoch: int) -> None:
        self._epoch.fill_(int(epoch))

    def _mix_seed(self, base_seed: int, epoch: int, idx: int, off: int) -> int:
        ss = np.random.SeedSequence([base_seed, epoch, idx, off])
        return int(ss.generate_state(1, dtype=np.uint64)[0])

    def _apply_augmentations(self, x: torch.Tensor, seed_aug: Optional[int]) -> torch.Tensor:
        rng = np.random.default_rng(seed_aug) if seed_aug is not None else np.random.default_rng()
        g = None
        if seed_aug is not None:
            g = torch.Generator(device="cpu")
            g.manual_seed(seed_aug)

        if rng.random() < self.rot_prob and self.rot_max_rad > 0.0:
            theta = rng.uniform(-self.rot_max_rad, self.rot_max_rad)
            x = aug_rotate_vectors(x, theta)

        if rng.random() < self.bg_prob and self.bg_factor > 0.0:
            umean = x[0].mean().item()
            vmean = x[1].mean().item()
            du = rng.uniform(-self.bg_factor * math.fabs(umean), self.bg_factor * math.fabs(umean))
            dv = rng.uniform(-self.bg_factor * math.fabs(vmean), self.bg_factor * math.fabs(vmean))
            x = aug_background(x, du, dv)

        if rng.random() < self.noise_prob and self.noise_std > 0.0:
            x = aug_add_mag_noise(x, self.noise_std, g=g)

        if rng.random() < self.blur_prob:
            sigma = rng.uniform(self.blur_sigma[0], self.blur_sigma[1])
            x = aug_gaussian_blur(x, self.blur_ksize, sigma)

        return x

    def __len__(self):
        return self.N

    def __getitem__(self, i: int):
        x_np = self.X[i]
        x = torch.from_numpy(x_np.copy()).to(torch.float32)
        x = (x - self.mean) / (self.std + 1e-6)

        epoch = int(self._epoch.item())
        seed_mask = None if self.base_seed is None else self._mix_seed(self.base_seed, epoch, i, off=0)
        rng_mask = np.random.default_rng(seed_mask) if seed_mask is not None else np.random.default_rng()
        
        v = int(rng_mask.choice(self.vr, p=self.probs))
        mask = random_spanning_mask(self.mask_shape, n=v, seed=seed_mask)

        seed_aug = None if self.base_seed is None else self._mix_seed(self.base_seed, epoch, i, off=2)
        x = self._apply_augmentations(x, seed_aug)

        coarse_mask = torch.from_numpy(mask)
        ups_h = x.shape[1] // coarse_mask.shape[0]
        ups_w = x.shape[2] // coarse_mask.shape[1]
        fine_mask = repeat(coarse_mask, ups_h, ups_w)
        masked_data = x * (~fine_mask).to(x.dtype)
        return {"x": x, "x_masked": masked_data, "mask": fine_mask, "idx": i}

class VFDataModule(pl.LightningDataModule):
    def __init__(self, train_ds, val_ds, seed: int, batch_size: int, num_workers: int):
        super().__init__()
        self.train_ds = train_ds
        self.val_ds = val_ds
        self.seed = seed
        self.batch_size = batch_size
        self.num_workers = num_workers

        self._g = torch.Generator()
        self._g.manual_seed(self.seed)

    def _get_sampler(self, ds, shuffle: bool, drop_last: bool):
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            return DistributedSampler(ds, shuffle=shuffle, seed=self.seed, drop_last=drop_last)
        return None

    def train_dataloader(self):
        sampler = self._get_sampler(self.train_ds, shuffle=True, drop_last=True)
        kwargs = dict(
            dataset=self.train_ds,
            sampler=sampler,
            drop_last=True,
            persistent_workers=(self.num_workers > 0),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )
        if sampler is None:
            kwargs["shuffle"] = True
            kwargs["generator"] = self._g

        return DataLoader(**kwargs)

    def val_dataloader(self):
        sampler = self._get_sampler(self.val_ds, shuffle=False, drop_last=False)
        kwargs = dict(
            dataset=self.val_ds,
            sampler=sampler,
            drop_last=False,
            persistent_workers=(self.num_workers > 0),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )
        if sampler is None:
            kwargs["shuffle"] = False
            kwargs["generator"] = self._g

        return DataLoader(**kwargs)

class VFEvalModule(pl.LightningDataModule):
    """ Evaluation is run on one process only. """
    def __init__(self, eval_ds, seed: int, batch_size: int, num_workers: int):
        assert not (torch.distributed.is_available() and torch.distributed.is_initialized()), \
            "VFEvalModule should be used in single-process evaluation only."
        super().__init__()
        self.eval_ds = eval_ds
        self.seed = seed
        self.batch_size = batch_size
        self.num_workers = num_workers

    def test_dataloader(self):
        return DataLoader(
            self.eval_ds,
            shuffle=False,
            drop_last=False,
            persistent_workers=False,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )