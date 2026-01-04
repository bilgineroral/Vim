import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import time

from dataset import VFEvalModule, VectorFieldDataset, load_data


SEEDS_DEFAULT = [2026, 7000, 12345, 54321, 99999]


def load_model(ckpt_path):
    from vmd import LitVMD
    
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state_dict = ckpt["state_dict"]

    new_state_dict = {}
    for k, v in state_dict.items():
        new_key = k.replace("_orig_mod.", "")
        new_state_dict[new_key] = v

    hparams = ckpt.get("hyper_parameters", {})
    model = LitVMD(**hparams)
    model.load_state_dict(new_state_dict)
    return model


def _to_float(v):
    if isinstance(v, torch.Tensor):
        return float(v.detach().cpu().item())
    if isinstance(v, (np.generic,)):
        return float(v.item())
    if isinstance(v, (float, int)):
        return float(v)
    return v


def _strip_test_prefix(metrics: dict) -> dict:
    out = {}
    for k, v in metrics.items():
        if isinstance(k, str) and k.startswith("test/"):
            out[k[len("test/"):]] = v
        else:
            out[k] = v
    return out


def eval_once(
    *,
    model_ckpt: str,
    eval_data: str,
    stats: str,
    vr: int,
    batch_size: int,
    seed: int,
    num_workers: int,
    devices: int,
):
    pl.seed_everything(seed, workers=False)

    model = load_model(model_ckpt)
    model.eval()

    X = load_data(eval_data)

    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": True,
        "persistent_workers": (num_workers > 0),
    }

    eval_set = VectorFieldDataset(
        X,
        stats,
        mask_shape=(16, 16),
        vrl=vr,
        vrh=vr,
        alpha=0.0,
        gamma=0.0,
        base_seed=seed,
    )
    datamodule = VFEvalModule(eval_set, seed=seed, loader_kwargs=loader_kwargs)

    trainer = pl.Trainer(
        devices=devices,
        num_nodes=1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
    )

    results = trainer.test(model, datamodule=datamodule, verbose=False)
    if not results:
        raise RuntimeError("trainer.test returned no results.")
    metrics = results[0]

    metrics = {k: _to_float(v) for k, v in metrics.items()}
    metrics = _strip_test_prefix(metrics)
    return metrics


def sweep(
    *,
    model_ckpt: str,
    eval_data: str,
    stats: str,
    vr_min: int,
    vr_max: int,
    seeds: list[int],
    batch_size: int,
    num_workers: int,
    devices: int,
):
    run_rows = []

    total_runs = (vr_max - vr_min + 1) * len(seeds)
    run_idx = 0
    t0 = time.time()

    for vr in range(vr_min, vr_max + 1):
        vr_t0 = time.time()
        print(f"[vr={vr}] starting ({len(seeds)} seeds)")

        for j, seed in enumerate(seeds, start=1):
            run_idx += 1
            one_t0 = time.time()

            metrics = eval_once(
                model_ckpt=model_ckpt,
                eval_data=eval_data,
                stats=stats,
                vr=vr,
                batch_size=batch_size,
                seed=seed,
                num_workers=num_workers,
                devices=devices,
            )
            run_rows.append({"vr": vr, "seed": seed, **metrics})

            dt = time.time() - one_t0
            elapsed = time.time() - t0
            # crude ETA based on average so far
            avg = elapsed / run_idx
            eta = avg * (total_runs - run_idx)

            # Print a couple of headline metrics if present
            headline = []
            for key in ("rmse", "mae", "ang_mean_deg"):
                if key in metrics:
                    headline.append(f"{key}={metrics[key]:.6g}")
            headline_str = (" | " + ", ".join(headline)) if headline else ""

            print(
                f"  [{run_idx:>3}/{total_runs}] vr={vr} seed={seed} "
                f"done in {dt:.1f}s (ETA ~ {eta/60:.1f} min){headline_str}"
            )

        vr_dt = time.time() - vr_t0
        print(f"[vr={vr}] done in {vr_dt:.1f}s\n")

    df_runs = pd.DataFrame(run_rows)
    metric_cols = [c for c in df_runs.columns if c not in ("vr", "seed")]

    grouped = df_runs.groupby("vr")[metric_cols]
    df_mean = grouped.mean(numeric_only=True).add_suffix("_mean")
    df_std = grouped.std(ddof=1, numeric_only=True).add_suffix("_std")  # sample std
    df_summary = pd.concat([df_mean, df_std], axis=1).reset_index()

    total_dt = time.time() - t0
    print(f"All runs complete: {total_runs} runs in {total_dt/60:.1f} min")

    return df_runs, df_summary


def write_excel(df_runs: pd.DataFrame, df_summary: pd.DataFrame, out_path: str):
    out = Path(out_path)
    if out.suffix.lower() != ".xlsx":
        raise ValueError("Excel output only. Please provide --out with a .xlsx extension.")

    out.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(out, engine="openpyxl") as writer:
        df_summary.to_excel(writer, index=False, sheet_name="summary")
        df_runs.to_excel(writer, index=False, sheet_name="runs")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--eval-data", type=str, required=True)
    p.add_argument("--stats", type=str, required=True)
    p.add_argument("--model-ckpt", type=str, required=True)

    p.add_argument("--vr-min", type=int, default=4)
    p.add_argument("--vr-max", type=int, default=16)
    p.add_argument("--seeds", type=int, nargs="+", default=SEEDS_DEFAULT)

    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--devices", type=int, default=1)

    p.add_argument("--out", type=str, required=True, help="Excel output path (.xlsx)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    df_runs, df_summary = sweep(
        model_ckpt=args.model_ckpt,
        eval_data=args.eval_data,
        stats=args.stats,
        vr_min=args.vr_min,
        vr_max=args.vr_max,
        seeds=args.seeds,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        devices=args.devices,
    )

    write_excel(df_runs, df_summary, args.out)
    print(f"Wrote Excel with summary + runs to: {args.out}")