from typing import Tuple
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import time

def random_spanning_mask(
    grid_shape: Tuple[int, int],
    n: int,
    k: Tuple[int, int] = (1, 1),
) -> torch.Tensor:
    if k != (1, 1):
        raise NotImplementedError("Currently only k=(1,1) is supported.")
    if n < 0:
        raise ValueError("n must be non-negative")

    Hp, Wp = grid_shape
    total = Hp * Wp

    if total == 0:
        return torch.ones((Hp, Wp), dtype=torch.bool)

    n = min(n, total)
    if n == 0:
        return torch.ones((Hp, Wp), dtype=torch.bool)
    if n == total:
        return torch.zeros((Hp, Wp), dtype=torch.bool)

    scores = torch.rand(total)
    _, idx = scores.topk(n, dim=0)

    visible_flat = torch.zeros(total, dtype=torch.bool)
    visible_flat.scatter_(0, idx, True)

    visible = visible_flat.view(Hp, Wp)
    mask = ~visible
    return mask

def tile_tensor(x: torch.Tensor, rH: int, rW: int) -> torch.Tensor:
    if rW > 1:
        x = x.repeat_interleave(rW, dim=-1)
    if rH > 1:
        x = x.repeat_interleave(rH, dim=-2)
    return x

def load_data(dataset_path: str, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    X_np = np.load(dataset_path, mmap_mode="r")
    if X_np.ndim != 4:
        raise ValueError(f"Expected (N,C,H,W), got {X_np.shape}")
    X = torch.tensor(X_np, dtype=dtype, device="cpu")
    X.share_memory_()
    return X

class VectorFieldDataset(Dataset):
    def __init__(self, X: torch.Tensor, 
                 mask_shape: Tuple[int, int] = (16, 16),
                 vrl: int = 4, vrh: int = 16, 
                 alpha=2.0, gamma=0.10, k=(1,1)):
        if not (torch.is_tensor(X) and X.ndim == 4):
            raise ValueError(f"Expected 4D tensor (N,C,H,W), got {getattr(X,'shape',None)}")
        self.X = X
        self.N = X.shape[0]
        self.mask_shape = mask_shape
        self.ups = X.shape[2] // mask_shape[0]
        assert self.ups == X.shape[3] // mask_shape[1], "Height and width upscaling factors must match."
        self.k = k
        self.vrl = vrl
        self.vrh = vrh
        self.vr = torch.arange(self.vrl, self.vrh + 1)
        mid = (self.vrl + self.vrh) / 2.0
        weights = (self.vrh - self.vr + 1).float().pow(alpha) * torch.exp(gamma * (self.vr - mid))
        self.probs = weights / weights.sum()

    def __shape__(self) -> Tuple[int, int, int]:
        return self.X[0].shape

    def __len__(self):
        return self.N

    def __getitem__(self, i: int):
        x = self.X[i]
        idx = torch.multinomial(self.probs, num_samples=1).item()
        visible_regions = self.vr[idx].item()
        mask = random_spanning_mask(
            grid_shape=self.mask_shape,
            n=visible_regions,
            k=self.k
        ) # True=masked, False=visible
        mask = tile_tensor(mask, self.ups, self.ups)
        masked_data = x * (~mask).to(self.X.dtype)
        return {"x": x, "x_masked": masked_data, "mask": mask, "idx": i}
    
if __name__ == "__main__":
    dataset_path = "/home/bilginer/dataset_hrrr/val/vec2d.npy"
    batch_size = 32
    workers = 0
    dtype = torch.float32
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X = load_data(dataset_path, dtype=dtype)
    ds = VectorFieldDataset(X, vrl=4, vrh=16, alpha=0.0, gamma=0.0)
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
        persistent_workers=(workers > 0),
        prefetch_factor=2 if workers > 0 else None
    )
    start = time.time()
    total_samples = 0
    for step, batch in enumerate(loader):
        x = batch["x"].to(device=device, non_blocking=True)
        x_masked = batch["x_masked"].to(device=device, non_blocking=True)
        mask = batch["mask"].to(device=device, non_blocking=True)
        total_samples += x.size(0)
    
    end = time.time()
    print(f"Processed {len(loader)} batches in {end - start:.2f} seconds (batch_size={batch_size}, total_samples={total_samples}).")
    print(f"Throughput: {total_samples / (end - start):.2f} samples/second.")
