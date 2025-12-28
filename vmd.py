from vim.models_mamba import VisionMamba
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

def vector_metric_sums(recon, orig, include_mask):
    """
    Returns sums + counts so you can sum-reduce across batches/ranks and divide once.
    """
    def _masked_err_sum_and_count(recon, target, include_mask, kind="abs"):
        """
        recon/target:   (B, C, H, W)
        include_mask:   (B, H, W) bool, True where INCLUDED in metric
        Returns:
        err_sum: scalar (sum over included pixels * channels)
        count:   scalar (#included pixels * channels)
        """
        B, C, H, W = recon.shape
        m = include_mask.to(dtype=recon.dtype).unsqueeze(1)  # (B,1,H,W)

        diff = recon - target
        if kind == "abs":
            err = diff.abs()
        elif kind == "sq":
            err = diff.square()
        else:
            raise ValueError(kind)

        err_sum = (err * m).sum()
        count = (include_mask.sum() * C).to(err_sum.dtype)
        return err_sum, count

    dev = recon.device
    eps = torch.finfo(recon.dtype).eps

    # component-wise
    abs_sum, count_c = _masked_err_sum_and_count(recon, orig, include_mask, "abs")
    sq_sum,  _       = _masked_err_sum_and_count(recon, orig, include_mask, "sq")

    # magnitudes
    mag_r = recon.square().sum(dim=1).sqrt()  # (B,H,W)
    mag_o = orig.square().sum(dim=1).sqrt()   # (B,H,W)
    err_mag = mag_r - mag_o

    vm = include_mask
    mag_abs_sum = torch.abs(err_mag[vm]).sum() if vm.any() else torch.tensor(0.0, device=dev)
    mag_sq_sum  = (err_mag[vm].square()).sum() if vm.any() else torch.tensor(0.0, device=dev)
    mag_count   = vm.sum().to(dtype=abs_sum.dtype)

    # relative magnitude (only where target magnitude > eps)
    rel_mask = vm & (mag_o > eps)
    if rel_mask.any():
        rel_err = err_mag[rel_mask] / mag_o[rel_mask]
        rel_abs_sum = rel_err.abs().sum()
        rel_sq_sum  = rel_err.square().sum()
        rel_count   = rel_mask.sum().to(dtype=abs_sum.dtype)
    else:
        rel_abs_sum = torch.tensor(0.0, device=dev)
        rel_sq_sum  = torch.tensor(0.0, device=dev)
        rel_count   = torch.tensor(0.0, device=dev, dtype=abs_sum.dtype)

    # angle
    dot = (recon * orig).sum(dim=1)                          # (B,H,W)
    denom = (mag_r * mag_o).clamp_min(eps)
    ok = vm & (mag_r > eps) & (mag_o > eps)
    if ok.any():
        cos_ok = torch.clamp(dot[ok] / denom[ok], -1.0, 1.0)
        ang_deg = torch.acos(cos_ok) * (180.0 / torch.pi)
        ang_sum = ang_deg.sum()
        ang_count = ok.sum().to(dtype=abs_sum.dtype)
    else:
        ang_sum = torch.tensor(0.0, device=dev)
        ang_count = torch.tensor(0.0, device=dev, dtype=abs_sum.dtype)

    return {
        "abs_sum": abs_sum,
        "sq_sum": sq_sum,
        "count_c": count_c,

        "mag_abs_sum": mag_abs_sum,
        "mag_sq_sum": mag_sq_sum,
        "mag_count": mag_count,

        "rel_abs_sum": rel_abs_sum,
        "rel_sq_sum": rel_sq_sum,
        "rel_count": rel_count,

        "ang_sum_deg": ang_sum,
        "ang_count": ang_count,
    }

class Decoder(nn.Module):
    def __init__(self, in_channels: int = 1024, out_channels: int = 2):
        super().__init__()
        layer_info = [
            # (in_ch, out_ch), stride, kernel
            ((in_channels, 512), 1, 3),
            ((512, 256), 2, 4),
            ((256, 128), 2, 4),
            ((128, 128), 2, 4),
            ((128, 64),  2, 4),
            ((64, 64),   1, 3),
            ((64, out_channels), 1, 3)
        ]
        blocks = []
        for i, ((in_ch, out_ch), stride, kernel_size) in enumerate(layer_info):
            is_last = (i == len(layer_info) - 1)
            layers = [
                nn.ConvTranspose2d(
                    in_ch, out_ch,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=1,
                    output_padding=0,
                )
            ]

            # GroupNorm + ReLU for intermediate blocks only
            if not is_last:
                layers += [nn.GroupNorm(num_groups=32, num_channels=out_ch),
                           nn.ReLU(inplace=True)]

            blocks.append(nn.Sequential(*layers))

        self.deconv_blocks = nn.ModuleList(blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.deconv_blocks:
            x = block(x)
        W_hat = x
        return W_hat

class VisionMambaDecoder(nn.Module):
    def __init__(self, img_size=512, patch_size=16):
        super().__init__()
        self.encoder = VisionMamba(
            img_size=img_size,
            patch_size=patch_size,
            stride=patch_size,
            depth=24,
            embed_dim=256,
            d_state=16,
            channels=2,
            num_classes=0,
            bimamba_type="v2",
            if_cls_token=False,
            use_middle_cls_token=False,
            final_pool_type='all',
            drop_path_rate=0,
        )
        self.proj = nn.Linear(self.encoder.embed_dim, 2*self.encoder.embed_dim)
        self.decoder = Decoder(in_channels=self.proj.out_features, out_channels=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        z = self.proj(z)
        z = z.transpose(1, 2).reshape(z.size(0), -1, self.encoder.feat_gran, self.encoder.feat_gran)
        W_hat = self.decoder(z)
        return W_hat

class LitVMD(pl.LightningModule):
    def __init__(self, img_size=512, patch_size=16, lr=1e-4, warmup_epochs=5, weight_decay=0.0):
        super().__init__()
        self.model = VisionMambaDecoder(img_size=img_size, patch_size=patch_size)
        self.save_hyperparameters()
        self.criterion = F.l1_loss

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, x_masked = batch["x"], batch["x_masked"]
        mask = batch["mask"]

        x_hat = self(x_masked)
        loss = self.criterion(x_hat, x, reduction='mean')

        self.log("train/loss", loss, on_step=False, on_epoch=True, sync_dist=True, prog_bar=True)
        self.log("train/mask_ratio", mask.float().mean(), on_step=False, on_epoch=True, sync_dist=True, prog_bar=False)
        self.log("train/lr", self.trainer.optimizers[0].param_groups[0]["lr"], on_step=False, on_epoch=True, sync_dist=True, prog_bar=False)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, x_masked = batch["x"], batch["x_masked"]
        mask = batch["mask"]

        x_hat = self(x_masked)
        x_hat = torch.where(mask.unsqueeze(1), x_hat, x)

        l1_loss = F.l1_loss(x_hat, x, reduction='mean')
        l2_loss = F.mse_loss(x_hat, x, reduction='mean')

        self.log(f"val/l1_loss", l1_loss, on_step=False, on_epoch=True, sync_dist=True, prog_bar=True)
        self.log(f"val/l2_loss", l2_loss, on_step=False, on_epoch=True, sync_dist=True, prog_bar=False)
        self.log("val/mask_ratio", mask.float().mean(), on_step=False, on_epoch=True, sync_dist=True, prog_bar=False)
        return {"l1_loss": l1_loss, "l2_loss": l2_loss}

    def test_step(self, batch, batch_idx):
        x, x_masked = batch["x"], batch["x_masked"]
        mask = batch["mask"]

        x_hat = self(x_masked)

        # Hard copy observed parts
        mask_expanded = mask.unsqueeze(1) # (B, 1, H, W)
        recon = torch.where(mask_expanded, x_hat, x)

        include_mask = torch.ones_like(mask, dtype=torch.bool)  # include all pixels
        s = vector_metric_sums(recon, x, include_mask)

        # sum-reduce across steps and ranks
        for k, v in s.items():
            self.log(
                f"test_agg/{k}",
                v,
                on_step=False,
                on_epoch=True,
                sync_dist=True,
                reduce_fx="sum",
                logger=False,
                prog_bar=False,
            )

    def on_test_epoch_end(self):
        m = self.trainer.callback_metrics

        def safe_div(num, den):
            return num / den.clamp_min(1.0)

        count_c = m["test_agg/count_c"]
        mae  = safe_div(m["test_agg/abs_sum"], count_c)
        mse  = safe_div(m["test_agg/sq_sum"],  count_c)
        rmse = torch.sqrt(mse)   # IMPORTANT: sqrt AFTER global mean-square

        mag_count = m["test_agg/mag_count"]
        mag_mae  = safe_div(m["test_agg/mag_abs_sum"], mag_count)
        mag_mse  = safe_div(m["test_agg/mag_sq_sum"],  mag_count)
        mag_rmse = torch.sqrt(mag_mse)

        rel_count = m["test_agg/rel_count"]
        rel_mae  = safe_div(m["test_agg/rel_abs_sum"], rel_count)
        rel_mse  = safe_div(m["test_agg/rel_sq_sum"],  rel_count)
        rel_rmse = torch.sqrt(rel_mse)

        ang_count = m["test_agg/ang_count"]
        ang_mean_deg = safe_div(m["test_agg/ang_sum_deg"], ang_count)

        self.log_dict(
            {
                "test/mae": mae,
                "test/mse": mse,
                "test/rmse": rmse,
                "test/mag_mae": mag_mae,
                "test/mag_mse": mag_mse,
                "test/mag_rmse": mag_rmse,
                "test/rel_mae": rel_mae,
                "test/rel_mse": rel_mse,
                "test/rel_rmse": rel_rmse,
                "test/ang_mean_deg": ang_mean_deg,
            },
            sync_dist=True,
            prog_bar=False,
        )

    def configure_optimizers(self):
        decay, no_decay = [], []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if name.endswith(".bias") or "norm" in name.lower():
                no_decay.append(param)
            else:
                decay.append(param)

        optimizer = torch.optim.AdamW(
            [
                {"params": decay, "weight_decay": self.hparams.weight_decay},
                {"params": no_decay, "weight_decay": 0.0},
            ],
            lr=self.hparams.lr,
        )

        total_steps = self.trainer.estimated_stepping_batches
        total_steps = int(max(1, total_steps))

        # map epoch-based warmup to steps by ratio
        max_epochs = int(max(1, self.trainer.max_epochs or 1))
        warmup_ratio = float(self.hparams.warmup_epochs) / float(max_epochs)
        warmup_steps = int(max(0, round(warmup_ratio * total_steps)))
        cosine_steps = int(max(1, total_steps - warmup_steps))

        # --- schedulers: step-based warmup then cosine ---
        if warmup_steps > 0:
            warmup = torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=1e-3,   # start at 0.1% of target LR
                end_factor=1.0,
                total_iters=warmup_steps,  # in optimizer steps (not epochs)
            )
            cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=cosine_steps,       # in optimizer steps
                eta_min=1e-6,
            )
            scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer,
                schedulers=[warmup, cosine],
                milestones=[warmup_steps],  # switch exactly after warmup_steps
            )
        else:
            # no warmup requested -> pure cosine over all steps
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=total_steps,
                eta_min=1e-6,
            )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",   # run after each optimizer step (respects grad accumulation)
                "frequency": 1,
                "name": "warmup+cosine",
            },
        }

if __name__ == "__main__":
    # simple test
    model = LitVMD(img_size=512, patch_size=16)
    
    # number of params
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of trainable parameters: {num_params/1e6:.2f} M")

    # number of encoder params
    num_enc_params = sum(p.numel() for p in model.model.encoder.parameters() if p.requires_grad)
    print(f"Number of encoder trainable parameters: {num_enc_params/1e6:.2f} M")

    # number of decoder params
    num_dec_params = sum(p.numel() for p in model.model.decoder.parameters() if p.requires_grad)
    print(f"Number of decoder trainable parameters: {num_dec_params/1e6:.2f} M")
