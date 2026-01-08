import argparse
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.utilities import rank_zero_only

from vmd import LitVMD
from dataset import VectorFieldDataset, VFDataModule

from config import config as cfg

@rank_zero_only
def rank_zero_print(*args, **kwargs):
    print(*args, **kwargs)

def main():
    p = argparse.ArgumentParser(description="Train MaskGFTransformer on 2D vector fields")
    p.add_argument("--train-data", type=str, required=True, help="Path to training set (.npy file)")
    p.add_argument("--val-data", type=str, required=True, help="Path to validation set (.npy file)")
    p.add_argument("--stats", type=str, default=None, help="Path to statistics (.json file)")
    p.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    args = p.parse_args()

    pl.seed_everything(cfg["seed"], workers=True)

    torch.set_float32_matmul_precision(cfg["matmul_precision"])

    rank_zero_print("\nTraining configuration:")
    for key, value in cfg.items():
        rank_zero_print(f"  {key}: {value}")
    rank_zero_print("")

    augmentations = {
        "rot_prob": cfg["rot_prob"],
        "noise_prob": cfg["noise_prob"],
        "blur_prob": cfg["blur_prob"],
        "bg_prob": cfg["bg_prob"],
    }

    train_dataset = VectorFieldDataset(args.train_data, args.stats, 
                                       vrl=cfg["vrl"], vrh=cfg["vrh"],
                                       base_seed=cfg["seed"],
                                       aug=augmentations)
    val_dataset = VectorFieldDataset(args.val_data, args.stats,
                                     vrl=cfg["vrl"], vrh=cfg["vrh"],
                                     base_seed=(cfg["seed"] + 10_000_000_000),
                                     aug=None)
    dm = VFDataModule(
        train_ds=train_dataset,
        val_ds=val_dataset,
        seed=cfg["seed"],
        batch_size=cfg["batch_size"],
        num_workers=cfg["num_workers"],
    )
    
    model = LitVMD(
        img_size=512, patch_size=16, lr=cfg["lr"],
        warmup_epochs=cfg["warmup_epochs"],
        weight_decay=cfg["weight_decay"],
        drop_rate=cfg["drop_rate"],
        drop_path_rate=cfg["drop_path_rate"],
    )

    if cfg["wandb"]:
        logger = WandbLogger(
            project=cfg["wandb_project"],
            name=cfg["wandb_run_name"],
            save_dir="./wandb",
            log_model=True,
        )
    else:
        logger = True

    checkpoint_callback = ModelCheckpoint(
        dirpath=cfg["output_dir"],
        filename=f"vmd_epoch_{{epoch:03d}}_{{val/l1_loss:.4f}}",
        monitor=f"val/l1_loss",
        mode="min",
        save_top_k=cfg["save_top_k"],
        save_last=cfg["save_last"],
        every_n_epochs=cfg["ckpt_every"],
        auto_insert_metric_name=False,
        save_on_train_epoch_end=False,
    )

    trainer = pl.Trainer(
        max_epochs=cfg["max_epochs"],
        accelerator="auto",
        devices="auto",
        precision="bf16-mixed" if cfg["amp_dtype"] == "bf16" else "16-mixed",
        gradient_clip_val=cfg["grad_clip"],
        log_every_n_steps=1,
        check_val_every_n_epoch=cfg["val_every"],
        logger=logger,
        callbacks=[checkpoint_callback],
        accumulate_grad_batches=cfg["grad_accum_steps"],
        use_distributed_sampler=False,
        enable_progress_bar=(not cfg["wandb"]),
    )

    trainer.fit(
        model, datamodule=dm,
        ckpt_path=(args.resume if args.resume is not None else None)
    )

if __name__ == "__main__":
    main()
