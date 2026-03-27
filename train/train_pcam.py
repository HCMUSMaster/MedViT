"""
train_pcam.py — Train MedViT (small/base/large) on the PCAM dataset.

PCAM: 327,680 RGB histopathology patches (96×96), binary classification.
Requires: pip install h5py

Outputs (outputs/{model}/pcam/):
  best_val.pth  — checkpoint with best validation AUC
  last.pth      — checkpoint from the final epoch
  train.log     — per-epoch log

results_summary_pcam.csv — final test AUC/ACC for every model

Usage:
    python train_pcam.py
    python train_pcam.py --data_root /path/to/pcam/data
    python train_pcam.py --models small base   # run only specific variants
"""

import argparse
import csv
import logging
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision
from torch.utils import data
from torch.utils.data import Subset
import torchvision.transforms as transforms
from sklearn.metrics import accuracy_score, roc_auc_score
from tqdm import tqdm
from torchvision.datasets import PCAM

from MedViT import MedViT_small, MedViT_base, MedViT_large

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

NUM_EPOCHS  = 100
BATCH_SIZE  = 24
LR          = 0.005
MOMENTUM    = 0.9
INPUT_SIZE  = 224
PATIENCE    = 5
N_CLASSES   = 2
DOWNLOAD    = True

ALL_MODELS = {
    "small": (MedViT_small, "ckpt/MedViT_small_im1k.pth"),
    "pathmnist": (MedViT_small, "outputs/small/pathmnist/best_val.pth"),
    "base":  (MedViT_base,  "ckpt/MedViT_base_im1k.pth"),
    "large": (MedViT_large, "ckpt/MedViT_large_im1k.pth"),
}

OUTPUT_ROOT = "outputs"
SUMMARY_CSV = "results_summary_pcam.csv"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_logger(log_path: str) -> logging.Logger:
    logger = logging.getLogger(log_path)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s  %(message)s", datefmt="%H:%M:%S")
    fh = logging.FileHandler(log_path)
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    return logger


def build_loaders(data_root: str):
    train_transform = transforms.Compose([
        transforms.Resize(INPUT_SIZE),
        transforms.AugMix(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    eval_transform = transforms.Compose([
        transforms.Resize(INPUT_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    train_ds = PCAM(root=data_root, split="train", transform=train_transform, download=DOWNLOAD)
    val_ds   = PCAM(root=data_root, split="val",   transform=eval_transform,  download=DOWNLOAD)
    test_ds  = PCAM(root=data_root, split="test",  transform=eval_transform,  download=DOWNLOAD)

    # 🔹 Compute subset sizes
    train_size = int(0.1 * len(train_ds))
    val_size   = int(0.5 * len(val_ds))

    # 🔹 Random indices
    train_indices = torch.randperm(len(train_ds))[:train_size]
    val_indices   = torch.randperm(len(val_ds))[:val_size]

    # 🔹 Subsets
    train_ds = Subset(train_ds, train_indices)
    val_ds   = Subset(val_ds, val_indices)

    print(f"Train samples: {len(train_ds):_}  Val samples: {len(val_ds):_}  Test samples: {len(test_ds):_}")

    train_loader = data.DataLoader(train_ds, batch_size=BATCH_SIZE,     shuffle=True,  num_workers=4, pin_memory=True)
    val_loader   = data.DataLoader(val_ds,   batch_size=BATCH_SIZE * 2, shuffle=False, num_workers=4, pin_memory=True)
    test_loader  = data.DataLoader(test_ds,  batch_size=BATCH_SIZE * 2, shuffle=False, num_workers=4, pin_memory=True)

    return train_loader, val_loader, test_loader


def build_model(model_name: str, ckpt_path: str) -> nn.Module:
    model = ALL_MODELS[model_name][0]()
    # Set the correct head first so fine-tuned checkpoints load fully
    model.proj_head[0] = nn.Linear(1024, N_CLASSES, bias=True)

    ckpt  = torch.load(ckpt_path, map_location="cpu")
    state = ckpt.get("model_state_dict", ckpt.get("model", ckpt))

    # Skip any keys whose shape doesn't match (e.g. proj_head from ImageNet ckpt with 1000 classes)
    model_state = model.state_dict()
    for k in list(state.keys()):
        if k in model_state and state[k].shape != model_state[k].shape:
            print(f"  [INFO] shape mismatch, skipping key: {k} "
                  f"({state[k].shape} → {model_state[k].shape})")
            del state[k]

    missing, _ = model.load_state_dict(state, strict=False)
    non_head_missing = [k for k in missing if not k.startswith("proj_head")]
    if non_head_missing:
        print(f"  [WARN] Unexpected missing keys: {non_head_missing}")

    return model.cuda()


def freeze_backbone(model: nn.Module) -> None:
    """Freeze all parameters except proj_head."""
    for name, param in model.named_parameters():
        if not name.startswith("proj_head"):
            param.requires_grad = False


def evaluate(model: nn.Module, loader: data.DataLoader, desc: str = "eval") -> tuple[float, float]:
    """Return (auc, acc) using sklearn — no medmnist dependency."""
    model.eval()
    all_targets = []
    all_scores  = []

    with torch.no_grad():
        for inputs, targets in tqdm(loader, desc=f"  {desc:5s}", leave=False):
            inputs = inputs.cuda()
            scores = model(inputs).softmax(dim=-1).cpu().numpy()
            all_scores.append(scores)
            # PCAM targets: shape (B,1) or (B,) with values 0/1
            t = targets.squeeze().numpy().astype(int)
            all_targets.append(t)

    y_true  = np.concatenate(all_targets)
    y_score = np.concatenate(all_scores)           # shape (N, 2)

    auc  = roc_auc_score(y_true, y_score[:, 1])
    acc  = accuracy_score(y_true, y_score.argmax(axis=1))
    return float(auc), float(acc)


def save_checkpoint(path: str, model: nn.Module, optimizer: optim.Optimizer,
                    epoch: int, val_auc: float, val_acc: float):
    torch.save({
        "epoch": epoch,
        "model_state_dict":     model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "val_auc": val_auc,
        "val_acc": val_acc,
    }, path)


# ---------------------------------------------------------------------------
# Single run
# ---------------------------------------------------------------------------

def run_one(model_name: str, ckpt_path: str, data_root: str, summary_writer,
            freeze: bool = False):
    mode     = "frozen" if freeze else "finetune"
    ds_dir   = "pcam_freeze" if freeze else "pcam"
    run_tag  = f"{model_name}_{ds_dir}"
    out_dir  = os.path.join(OUTPUT_ROOT, model_name, ds_dir)
    os.makedirs(out_dir, exist_ok=True)

    logger = get_logger(os.path.join(out_dir, f"{run_tag}.log"))
    logger.info(f"=== run={run_tag} ===")
    logger.info(f"model={model_name}  dataset=pcam  mode={mode}  init_ckpt={ckpt_path}")

    train_loader, val_loader, test_loader = build_loaders(data_root)

    model = build_model(model_name, ckpt_path)

    if freeze:
        freeze_backbone(model)
        trainable = [p for p in model.parameters() if p.requires_grad]
        logger.info(f"backbone frozen — trainable params: {sum(p.numel() for p in trainable):_}")
    else:
        trainable = list(model.parameters())
        logger.info(f"total params: {sum(p.numel() for p in trainable):_}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(trainable, lr=LR, momentum=MOMENTUM)

    best_val_auc   = -1.0
    patience_count = 0
    best_ckpt_path = os.path.join(out_dir, "best_val.pth")
    last_ckpt_path = os.path.join(out_dir, "last.pth")

    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0
        n_batches    = 0

        for inputs, targets in tqdm(train_loader,
                                    desc=f"[{model_name}/pcam] Epoch {epoch+1}/{NUM_EPOCHS}",
                                    leave=False):
            inputs  = inputs.cuda()
            targets = targets.squeeze().long().cuda()

            optimizer.zero_grad()
            loss = criterion(model(inputs), targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            n_batches    += 1

        avg_loss = running_loss / max(n_batches, 1)
        val_auc, val_acc = evaluate(model, val_loader, desc="val")

        logger.info(
            f"Epoch [{epoch+1:3d}/{NUM_EPOCHS}]  loss={avg_loss:.4f}  "
            f"val_auc={val_auc:.4f}  val_acc={val_acc:.4f}"
        )

        if val_auc > best_val_auc:
            best_val_auc   = val_auc
            patience_count = 0
            save_checkpoint(best_ckpt_path, model, optimizer, epoch, val_auc, val_acc)
            logger.info(f"  --> new best val AUC={val_auc:.4f}  (epoch {epoch+1})")
        else:
            patience_count += 1
            logger.info(f"  no improvement ({patience_count}/{PATIENCE})")
            if patience_count >= PATIENCE:
                logger.info(f"  early stopping at epoch {epoch+1}")
                break

    # Save last checkpoint
    val_auc, val_acc = evaluate(model, val_loader, desc="val")
    save_checkpoint(last_ckpt_path, model, optimizer, epoch, val_auc, val_acc)

    # Final test evaluation
    test_auc, test_acc = evaluate(model, test_loader, desc="test")
    logger.info(f"FINAL  test_auc={test_auc:.4f}  test_acc={test_acc:.4f}  best_val_auc={best_val_auc:.4f}")

    summary_writer.writerow({
        "model":        model_name,
        "dataset":      "pcam",
        "mode":         mode,
        "best_val_auc": f"{best_val_auc:.4f}",
        "test_auc":     f"{test_auc:.4f}",
        "test_acc":     f"{test_acc:.4f}",
    })

    del model
    torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", default="./data",
                   help="Root dir for PCAM download (default: ./data)")
    p.add_argument("--models", nargs="+", choices=list(ALL_MODELS),
                   default=list(ALL_MODELS),
                   help="Which model variants to train (default: all)")
    p.add_argument("--ckpt", default=None,
                   help="Custom checkpoint to initialize from (overrides the default "
                        "ImageNet pretrained ckpt for all selected models). "
                        "Accepts both ImageNet ckpts and fine-tuned ckpts.")
    p.add_argument("--freeze_backbone", action="store_true",
                   help="Freeze all layers except proj_head (linear probing mode)")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    models_to_run = {k: v for k, v in ALL_MODELS.items() if k in args.models}

    with open(SUMMARY_CSV, "w", newline="") as csv_file:
        fieldnames = ["model", "dataset", "mode", "best_val_auc", "test_auc", "test_acc"]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        for model_name, (_, default_ckpt) in models_to_run.items():
            ckpt_path = args.ckpt if args.ckpt else default_ckpt
            t0 = time.time()
            print(f"\n{'='*60}")
            print(f"  START  model={model_name}  dataset=pcam  ckpt={ckpt_path}")
            print(f"{'='*60}")
            try:
                run_one(model_name, ckpt_path, args.data_root, writer,
                        freeze=args.freeze_backbone)
                csv_file.flush()
            except Exception as exc:
                print(f"  [ERROR] {model_name}/pcam: {exc}")
                writer.writerow({
                    "model": model_name, "dataset": "pcam",
                    "best_val_auc": "ERROR", "test_auc": "ERROR",
                    "test_acc": str(exc)[:80],
                })
                csv_file.flush()
            elapsed = time.time() - t0
            print(f"  DONE  ({elapsed/60:.1f} min)")

    print(f"\nAll runs complete. Summary saved to {SUMMARY_CSV}")


if __name__ == "__main__":
    main()
