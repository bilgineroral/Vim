import os
import random
from dataclasses import dataclass, asdict
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

from dataset import VectorFieldDataset, load_data
from vmd import VisionMambaDecoder

import wandb

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

from contextlib import nullcontext

def is_dist_avail_and_initialized() -> bool:
    return dist.is_available() and dist.is_initialized()

def get_rank() -> int:
    return dist.get_rank() if is_dist_avail_and_initialized() else 0

def is_main_process() -> bool:
    return get_rank() == 0

@dataclass
class TrainConfig:
    train_path: str = "/home/bilginer/dataset_hrrr/train/vec2d.npy"
    val_path: str = "/home/bilginer/dataset_hrrr/val/vec2d.npy"
    img_size: int = 512
    patch_size: int = 16
    mask_shape: Tuple[int, int] = (16, 16)
    vrl: int = 4
    vrh: int = 16
    alpha: float = 2.0
    gamma: float = 0.10
    k: Tuple[int, int] = (1, 1)

    batch_size: int = 32 # per-GPU batch size
    num_epochs: int = 5
    lr: float = 5e-4
    weight_decay: float = 0.05
    grad_clip_norm: float = 1.0
    grad_accum_steps: int = 1

    num_workers: int = 4
    pin_memory: bool = True

    wandb_project: str = "vmdecoder"
    wandb_run_name: str = None
    log_every: int = 10 # in optimizer steps
    validate_every: int = 20 # epochs
    ckpt_dir: str = "./checkpoints"
    keep_top_k: int = 3

    seed: int = 42
    compile_model: bool = False # doesn't work with the kernels

def masked_l1_loss(recon, target, mask_tokens):
    B, C, H, W = recon.shape
    mask_tokens = mask_tokens.to(dtype=recon.dtype)
    m_exp = mask_tokens.unsqueeze(1)

    err = F.l1_loss(recon, target, reduction='none')
    masked_err = err * m_exp

    denom = (mask_tokens.sum() * C).clamp_min(1.0)
    loss = masked_err.sum() / denom
    return loss

def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def create_dataloaders(config: TrainConfig, distributed: bool):
    train_X = load_data(config.train_path, dtype=torch.float32)
    val_X = load_data(config.val_path, dtype=torch.float32)

    train_ds = VectorFieldDataset(
        train_X,
        mask_shape=config.mask_shape,
        vrl=config.vrl,
        vrh=config.vrh,
        alpha=config.alpha,
        gamma=config.gamma,
        k=config.k,
    )
    val_ds = VectorFieldDataset(
        val_X,
        mask_shape=config.mask_shape,
        vrl=config.vrl,
        vrh=config.vrh,
        alpha=config.alpha,
        gamma=config.gamma,
        k=config.k,
    )

    train_sampler = DistributedSampler(train_ds) if distributed else None
    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        drop_last=True,
    )

    return train_loader, val_loader, train_sampler

def save_checkpoint(
    config: TrainConfig,
    epoch: int,
    global_step: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    best_val_loss: float,
    suffix: str,
) -> str:
    os.makedirs(config.ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(config.ckpt_dir, f"checkpoint_{suffix}.pt")
    state = {
        "epoch": epoch,
        "global_step": global_step,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "best_val_loss": best_val_loss,
        "config": asdict(config),
    }
    torch.save(state, ckpt_path)
    return ckpt_path

@torch.no_grad()
def evaluate(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
) -> float:
    model.eval()
    total_loss = 0.0
    total_samples = 0

    for batch in val_loader:
        x = batch["x"].to(device, non_blocking=True) # (B, C, H, W)
        x_masked = batch["x_masked"].to(device, non_blocking=True) # (B, C, H, W)
        mask = batch["mask"].to(device, non_blocking=True) # (B, H, W)
        
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            x_hat = model(x_masked)
            loss = masked_l1_loss(x_hat, x, mask_tokens=mask)

        bs = x_masked.size(0)
        total_loss += loss.item() * bs
        total_samples += bs

    mean_loss = total_loss / max(1, total_samples)
    return mean_loss

def train(config: TrainConfig):
    assert torch.cuda.is_available(), "cuda is required"

    num_gpus = torch.cuda.device_count()
    use_ddp = num_gpus > 1

    if use_ddp:
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        dist.init_process_group(backend="nccl")
    else:
        local_rank = 0
        device = torch.device("cuda")

    seed_everything(config.seed + get_rank())

    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")

    train_loader, val_loader, train_sampler = create_dataloaders(
        config, distributed=use_ddp
    )
    model = VisionMambaDecoder(img_size=config.img_size, patch_size=config.patch_size)
    model.to(device)

    if config.compile_model:
        model = torch.compile(model)

    if use_ddp:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )

    if is_main_process():
        wandb.init(
            project=config.wandb_project,
            name=config.wandb_run_name,
            config=asdict(config),
        )
        wandb.watch(model, log="none")

    best_val_loss = float("inf")
    global_step = 0
    best_checkpoints = []

    if is_main_process():
        print("Training config:")
        for k, v in asdict(config).items():
            print(f"  {k}: {v}")

    grad_accum = max(1, config.grad_accum_steps)

    for epoch in range(1, config.num_epochs + 1):
        model.train()
        if use_ddp and isinstance(train_sampler, DistributedSampler):
            train_sampler.set_epoch(epoch)

        running_loss = 0.0
        micro_batches_since_log = 0

        optimizer.zero_grad(set_to_none=True)
        num_batches = len(train_loader)

        full_batch_total = num_batches // grad_accum
        max_step = full_batch_total * grad_accum # drop last partial batch

        for step, batch in enumerate(train_loader, start=1):
            if step > max_step:
                break
            x = batch["x"].to(device, non_blocking=True)
            x_masked = batch["x_masked"].to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)

            micro_step = (step - 1) % grad_accum
            is_last_micro_in_block = (micro_step == grad_accum - 1)

            if use_ddp and isinstance(model, DDP) and not is_last_micro_in_block:
                context = model.no_sync()
            else:
                context = nullcontext()

            with context:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    x_hat = model(x_masked)
                    loss = masked_l1_loss(x_hat, x, mask_tokens=mask)

                loss_item = loss.item()
                running_loss += loss_item
                micro_batches_since_log += 1
                loss = loss / grad_accum
                loss.backward()

            if is_last_micro_in_block:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip_norm)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

                global_step += 1
                if config.log_every > 0 and global_step % config.log_every == 0:
                    avg_loss_local = running_loss / max(1, micro_batches_since_log)

                    if use_ddp and is_dist_avail_and_initialized():
                        loss_tensor = torch.tensor(
                            avg_loss_local, device=device, dtype=torch.float32
                        )
                        dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
                        world_size = dist.get_world_size()
                        avg_loss = (loss_tensor / world_size).item()
                    else:
                        avg_loss = avg_loss_local

                    if is_main_process():
                        wandb.log(
                            {
                                "train/loss": avg_loss,
                                "epoch": epoch,
                                "lr": optimizer.param_groups[0]["lr"],
                            },
                            step=global_step,
                        )
                        print(
                            f"[Epoch {epoch} | OptStep {global_step}] "
                            f"train/loss={avg_loss:.6f}"
                        )

                    running_loss = 0.0
                    micro_batches_since_log = 0

        if is_main_process() and (epoch % config.validate_every == 0):
            eval_model = model.module if isinstance(model, DDP) else model
            val_loss = evaluate(eval_model, val_loader, device)
            wandb.log({"val/loss": val_loss, "epoch": epoch}, step=global_step)
            print(f"[Epoch {epoch}] val/loss={val_loss:.6f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss

            should_save = False
            if len(best_checkpoints) < config.keep_top_k:
                should_save = True
            else:
                worst_loss, _ = max(best_checkpoints, key=lambda x: x[0])
                if val_loss < worst_loss:
                    should_save = True

            if should_save:
                suffix = f"val_{val_loss:.4f}_epoch_{epoch:03d}"
                ckpt_path = save_checkpoint(
                    config,
                    epoch=epoch,
                    global_step=global_step,
                    model=eval_model,
                    optimizer=optimizer,
                    best_val_loss=best_val_loss,
                    suffix=suffix,
                )

                best_checkpoints.append((val_loss, ckpt_path))

                if len(best_checkpoints) > config.keep_top_k:
                    worst_loss, worst_path = max(
                        best_checkpoints, key=lambda x: x[0]
                    )
                    try:
                        os.remove(worst_path)
                    except OSError as e:
                        pass

                    best_checkpoints = [
                        (l, p) for (l, p) in best_checkpoints if p != worst_path
                    ]

    if is_main_process():
        print("Training finished!")
        wandb.finish()

    if use_ddp:
        dist.destroy_process_group()

if __name__ == "__main__":
    cfg = TrainConfig(
        train_path="/home/bilginer/dataset_hrrr/train/vec2d.npy",
        val_path="/home/bilginer/dataset_hrrr/val/vec2d.npy",
        alpha=0.0,
        gamma=0.0,
        num_epochs=5,
        batch_size=8,
        grad_accum_steps=2,
        log_every=10,
        validate_every=1,
        ckpt_every=1,
        keep_top_k=3
    )
    train(cfg)
