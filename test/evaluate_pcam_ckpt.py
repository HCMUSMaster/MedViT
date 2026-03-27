"""
evaluate_pcam_ckpt.py — Load a PCAM checkpoint and evaluate AUC/ACC on train/val/test.

Usage:
    # Single checkpoint:
    python evaluate_pcam_ckpt.py --ckpt outputs/small/pcam/best_val.pth --model small

    # Sweep all outputs/*/pcam/ automatically:
    python evaluate_pcam_ckpt.py

    # Use last.pth instead of best_val.pth:
    python evaluate_pcam_ckpt.py --ckpt_name last.pth

    # Custom data root:
    python evaluate_pcam_ckpt.py --data_root /path/to/pcam
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
from tqdm import tqdm
from torchvision.datasets import PCAM

from MedViT import MedViT_small, MedViT_base, MedViT_large

# ---------------------------------------------------------------------------
# Constants (must match train_pcam.py)
# ---------------------------------------------------------------------------

INPUT_SIZE = 224
N_CLASSES = 2
DOWNLOAD = True
# SPLITS      = ["train", "val", "test"]
SPLITS = ["test"]

MODELS = {
    "small": MedViT_small,
    "base": MedViT_base,
    "large": MedViT_large,
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def build_loader(data_root: str, split: str, batch_size: int) -> data.DataLoader:
    transform = transforms.Compose(
        [
            transforms.Resize(INPUT_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )
    ds = PCAM(root=data_root, split=split, transform=transform, download=DOWNLOAD)
    return data.DataLoader(
        ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True
    )


def load_model(model_name: str, ckpt_path: str, device: torch.device) -> nn.Module:
    model = MODELS[model_name]()
    model.proj_head[0] = nn.Linear(1024, N_CLASSES, bias=True)

    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt.get("model_state_dict", ckpt.get("model", ckpt))
    model.load_state_dict(state)
    model.to(device).eval()
    return model


def evaluate_split(model: nn.Module, loader: data.DataLoader, split: str):
    all_targets = []
    all_scores = []
    device = next(model.parameters()).device
    with torch.no_grad():
        for inputs, targets in tqdm(loader, desc=f"  {split:5s}", leave=False):
            inputs = inputs.to(device)
            scores = model(inputs).softmax(dim=-1).cpu().numpy()
            all_scores.append(scores)
            all_targets.append(targets.squeeze().numpy().astype(int))

    y_true = np.concatenate(all_targets)
    y_score = np.concatenate(all_scores)

    auc = roc_auc_score(y_true, y_score[:, 1])
    acc = accuracy_score(y_true, y_score.argmax(axis=1))
    return float(auc), float(acc)


# ---------------------------------------------------------------------------
# Single checkpoint evaluation
# ---------------------------------------------------------------------------


def evaluate_one(
    model_name: str, ckpt_path: str, data_root: str, batch_size: int
) -> dict:
    print(f"\n  model={model_name}  dataset=pcam")
    print(f"  ckpt={ckpt_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(model_name, ckpt_path, device)
    row = {"model": model_name, "dataset": "pcam", "ckpt": ckpt_path}

    for split in SPLITS:
        loader = build_loader(data_root, split, batch_size)
        auc, acc = evaluate_split(model, loader, split)
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
    p.add_argument(
        "--ckpt", default=None, help="Path to a specific .pth checkpoint file"
    )
    p.add_argument(
        "--model",
        choices=list(MODELS),
        default=None,
        help="Model variant (required with --ckpt)",
    )
    p.add_argument(
        "--ckpt_name",
        default="best_val.pth",
        help="Checkpoint filename for auto-sweep (default: best_val.pth)",
    )
    p.add_argument(
        "--data_root",
        default="./dataset",
        help="Root dir for PCAM data (default: ./dataset)",
    )
    p.add_argument(
        "--out_csv",
        default="eval_pcam_results.csv",
        help="Output CSV path (default: eval_pcam_results.csv)",
    )
    p.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for DataLoader (default: 32)",
    )
    p.add_argument(
        "--output_root",
        type=str,
        default="outputs",
        help="Root directory for outputs (default: outputs)",
    )
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
        # Auto-sweep outputs/{model}/pcam/{ckpt_name}
        print(f"Sweeping {args.output_root}/*/pcam/{args.ckpt_name} ...")
        for model_name in MODELS:
            ckpt_path = os.path.join(
                args.output_root, model_name, "pcam", args.ckpt_name
            )
            if os.path.isfile(ckpt_path):
                jobs.append((model_name, ckpt_path))

    if not jobs:
        print("No PCAM checkpoints found. Nothing to evaluate.")
        sys.exit(0)

    print(f"Found {len(jobs)} checkpoint(s).")

    fieldnames = [
        "model",
        "dataset",
        "ckpt",
        "train_auc",
        "train_acc",
        "val_auc",
        "val_acc",
        "test_auc",
        "test_acc",
    ]
    rows = []

    for model_name, ckpt_path in jobs:
        try:
            row = evaluate_one(model_name, ckpt_path, args.data_root, args.batch_size)
            rows.append(row)
        except Exception as exc:
            print(f"  [ERROR] {model_name}/pcam: {exc}")
            rows.append(
                {
                    "model": model_name,
                    "dataset": "pcam",
                    "ckpt": ckpt_path,
                    "test_auc": f"ERROR: {exc}",
                }
            )

    with open(args.out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nResults saved to {args.out_csv}")

    print(
        f"\n{'Model':<8}  {'Train AUC':>9}  {'Val AUC':>7}  {'Test AUC':>8}  {'Test ACC':>8}"
    )
    print("-" * 50)
    for r in rows:
        print(
            f"{r.get('model',''):<8}  "
            f"{r.get('train_auc','N/A'):>9}  {r.get('val_auc','N/A'):>7}  "
            f"{r.get('test_auc','N/A'):>8}  {r.get('test_acc','N/A'):>8}"
        )


if __name__ == "__main__":
    main()
