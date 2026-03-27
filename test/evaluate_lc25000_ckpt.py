"""
evaluate_lc25000_ckpt.py — Load a checkpoint and evaluate AUC/ACC on
train / val / test splits of the LC25000 dataset.

Usage:
    # Single checkpoint:
    python evaluate_lc25000_ckpt.py \
        --ckpt outputs/small/lc25000/best_val.pth --model small

    # Auto-sweep all outputs/*/lc25000*/:
    python evaluate_lc25000_ckpt.py

    # Use last.pth instead of best_val.pth:
    python evaluate_lc25000_ckpt.py --ckpt_name last.pth

    # Custom data root:
    python evaluate_lc25000_ckpt.py --data_root /path/to/lc25000
"""

import argparse
import csv
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision.transforms as transforms
from sklearn.metrics import accuracy_score, roc_auc_score
from torch.utils.data import random_split
from torchvision.datasets import ImageFolder
from tqdm import tqdm

from MedViT import MedViT_small, MedViT_base, MedViT_large

# ---------------------------------------------------------------------------
# Constants (must match train_lc25000.py)
# ---------------------------------------------------------------------------

INPUT_SIZE  = 224
BATCH_SIZE  = 32
VAL_SPLIT   = 0.2
OUTPUT_ROOT = "outputs"
# SPLITS      = ["train", "val", "test"]
SPLITS      = ["test"]

TRAINVAL_DIR = "lung_colon_image_set/Train and Validation Set"
TEST_DIR     = "lung_colon_image_set/Test Set"

# Directories that contain LC25000 checkpoints (both finetune and frozen)
LC_DS_DIRS = ["lc25000", "lc25000_freeze"]

MODELS = {
    "small": MedViT_small,
    "base":  MedViT_base,
    "large": MedViT_large,
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def build_loaders(data_root: str):
    eval_transform = transforms.Compose([
        transforms.Resize(INPUT_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    trainval_path = os.path.join(data_root, TRAINVAL_DIR)
    test_path     = os.path.join(data_root, TEST_DIR)

    full_ds   = ImageFolder(trainval_path, transform=eval_transform)
    classes   = full_ds.classes
    n_classes = len(classes)

    n_val   = int(len(full_ds) * VAL_SPLIT)
    n_train = len(full_ds) - n_val
    train_ds, val_ds = random_split(
        full_ds, [n_train, n_val],
        generator=torch.Generator().manual_seed(42)
    )

    test_ds = ImageFolder(test_path, transform=eval_transform)

    loaders = {
        "train": data.DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=False,
                                 num_workers=4, pin_memory=True),
        "val":   data.DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                                 num_workers=4, pin_memory=True),
        "test":  data.DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False,
                                 num_workers=4, pin_memory=True),
    }
    return loaders, classes, n_classes


def load_model(model_name: str, ckpt_path: str, n_classes: int) -> nn.Module:
    model = MODELS[model_name]()
    model.proj_head[0] = nn.Linear(1024, n_classes, bias=True)

    ckpt  = torch.load(ckpt_path, map_location="cpu")
    state = ckpt.get("model_state_dict", ckpt.get("model", ckpt))
    model.load_state_dict(state)
    model.cuda().eval()
    return model


def evaluate_split(model: nn.Module, loader: data.DataLoader,
                   n_classes: int, split: str) -> tuple[float, float]:
    all_targets = []
    all_scores  = []

    with torch.no_grad():
        for inputs, targets in tqdm(loader, desc=f"  {split:5s}", leave=False):
            inputs = inputs.cuda()
            scores = model(inputs).softmax(dim=-1).cpu().numpy()
            all_scores.append(scores)
            all_targets.append(targets.numpy())

    y_true  = np.concatenate(all_targets)
    y_score = np.concatenate(all_scores)

    auc = roc_auc_score(y_true, y_score,
                        multi_class="ovr", average="macro")
    acc = accuracy_score(y_true, y_score.argmax(axis=1))
    return float(auc), float(acc)


# ---------------------------------------------------------------------------
# Single checkpoint evaluation
# ---------------------------------------------------------------------------

def evaluate_one(model_name: str, ckpt_path: str, data_root: str) -> dict:
    print(f"\n  model={model_name}  ckpt={ckpt_path}")

    loaders, classes, n_classes = build_loaders(data_root)
    print(f"  classes: {classes}")

    model = load_model(model_name, ckpt_path, n_classes)
    row   = {"model": model_name, "dataset": "lc25000", "ckpt": ckpt_path}

    for split in SPLITS:
        auc, acc = evaluate_split(model, loaders[split], n_classes, split)
        row[f"{split}_auc"] = f"{auc:.4f}"
        row[f"{split}_acc"] = f"{acc:.4f}"
        print(f"    {split:5s}  AUC={auc:.4f}  ACC={acc:.4f}")

    del model
    torch.cuda.empty_cache()
    return row


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt",      default=None,
                   help="Path to a specific .pth checkpoint file")
    p.add_argument("--model",     choices=list(MODELS), default=None,
                   help="Model variant (required with --ckpt)")
    p.add_argument("--ckpt_name", default="best_val.pth",
                   help="Checkpoint filename for auto-sweep (default: best_val.pth)")
    p.add_argument("--data_root", default="./dataset/lc25000",
                   help="Root of LC25000 dataset (default: ./dataset/lc25000)")
    p.add_argument("--out_csv",   default="eval_lc25000_results.csv",
                   help="Output CSV path (default: eval_lc25000_results.csv)")
    return p.parse_args()


def main():
    args = parse_args()
    jobs = []  # list of (model_name, ckpt_path)

    if args.ckpt:
        if not args.model:
            print("ERROR: --model is required when using --ckpt")
            sys.exit(1)
        if not os.path.isfile(args.ckpt):
            print(f"ERROR: checkpoint not found: {args.ckpt}")
            sys.exit(1)
        jobs.append((args.model, args.ckpt))
    else:
        # Auto-sweep outputs/{model}/{lc25000,lc25000_freeze}/{ckpt_name}
        print(f"Sweeping {OUTPUT_ROOT}/*/{{lc25000,lc25000_freeze}}/{args.ckpt_name} ...")
        for model_name in MODELS:
            for ds_dir in LC_DS_DIRS:
                ckpt_path = os.path.join(OUTPUT_ROOT, model_name, ds_dir, args.ckpt_name)
                if os.path.isfile(ckpt_path):
                    jobs.append((model_name, ckpt_path))

    if not jobs:
        print("No LC25000 checkpoints found. Nothing to evaluate.")
        sys.exit(0)

    print(f"Found {len(jobs)} checkpoint(s).")

    fieldnames = ["model", "dataset", "ckpt",
                  "train_auc", "train_acc",
                  "val_auc",   "val_acc",
                  "test_auc",  "test_acc"]
    rows = []

    for model_name, ckpt_path in jobs:
        try:
            row = evaluate_one(model_name, ckpt_path, args.data_root)
            rows.append(row)
        except Exception as exc:
            print(f"  [ERROR] {model_name}: {exc}")
            rows.append({"model": model_name, "dataset": "lc25000",
                         "ckpt": ckpt_path, "test_auc": f"ERROR: {exc}"})

    with open(args.out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nResults saved to {args.out_csv}")

    print(f"\n{'Model':<8}  {'Ckpt':<30}  "
          f"{'Train AUC':>9}  {'Val AUC':>7}  {'Test AUC':>8}  {'Test ACC':>8}")
    print("-" * 80)
    for r in rows:
        ckpt_short = os.path.join(*r.get("ckpt", "").split(os.sep)[-3:])
        print(f"{r.get('model',''):<8}  {ckpt_short:<30}  "
              f"{r.get('train_auc','N/A'):>9}  {r.get('val_auc','N/A'):>7}  "
              f"{r.get('test_auc','N/A'):>8}  {r.get('test_acc','N/A'):>8}")


if __name__ == "__main__":
    main()
