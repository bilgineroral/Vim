r = "grf/run1"
ckpt_dir = f"checkpoints/{r}"
config = {
    "batch_size": 8,
    "warmup_epochs": 5,
    "max_epochs": 1000,

    "grad_clip": 1.0,
    "grad_accum_steps": 4,
    "lr": 2e-4,
    "weight_decay": 0.0,

    "num_workers": 4,
    "prefetch_factor": 2,
    "seed": 42,
    
    "vrl": 4, # min. number of visible regions
    "vrh": 16, # max. number of visible regions
    "k": (1, 1), # spanning mask block size

    "compile": False,
    "matmul_precision": False,
    "amp_dtype": "fp16",

    "output_dir": ckpt_dir,
    "save_top_k": 3,
    "log_every": 1, # epochs
    "ckpt_every": 10, # epochs
    "val_every": 10, # epochs
    "save_last": True,

    "wandb": True,
    "wandb_run_name": r,
    "wandb_project": "vmdecoder",
}