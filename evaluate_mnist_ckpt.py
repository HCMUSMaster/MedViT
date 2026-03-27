"""
evaluate_ckpt.py — Load a checkpoint and evaluate AUC/ACC on train/val/test.

Usage:
    # Single checkpoint:
    python evaluate_ckpt.py --ckpt outputs/small/breastmnist/best_val.pth \
                            --model small --dataset breastmnist

    # Sweep all outputs/ automatically (no args needed):
    python evaluate_ckpt.py

    # Use last.pth instead of best_val.pth:
    python evaluate_ckpt.py --ckpt_name last.pth
"""

import argparse
import csv
import os
import sys

import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision.transforms as transforms
from tqdm import tqdm

import medmnist
from medmnist import INFO, Evaluator

from MedViT import MedViT_small, MedViT_base, MedViT_large

# ---------------------------------------------------------------------------
# Constants (must match train_all_medmnist.py)
# ---------------------------------------------------------------------------

INPUT_SIZE = 224
BATCH_SIZE = 256
DOWNLOAD   = True
OUTPUT_ROOT = "outputs"

MODELS = {
    "small": MedViT_small,
    "base":  MedViT_base,
    "large": MedViT_large,
}

DATASETS = [
    "tissuemnist", "pathmnist", "chestmnist", "dermamnist", "octmnist",
    "pneumoniamnist", "retinamnist", "breastmnist", "bloodmnist",
    "organamnist", "organcmnist", "organsmnist",
]

# SPLITS = ["train", "val", "test"]
SPLITS = ["val", "test"]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def build_loader(data_flag: str, split: str):
    info = INFO[data_flag]
    DataClass = getattr(medmnist, info["python_class"])
    transform = transforms.Compose([
        transforms.Resize(INPUT_SIZE),
        transforms.Lambda(lambda img: img.convert("RGB")),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])
    ds = DataClass(split=split, transform=transform, download=DOWNLOAD)
    return data.DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False,
                           num_workers=4, pin_memory=True), info


def load_model(model_name: str, ckpt_path: str, n_classes: int) -> nn.Module:
    model_fn = MODELS[model_name]
    model = model_fn()
    model.proj_head[0] = nn.Linear(1024, n_classes, bias=True)

    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt.get("model_state_dict", ckpt.get("model", ckpt))
    model.load_state_dict(state)
    model.cuda().eval()
    return model


def evaluate_split(model, loader, data_flag: str, split: str, task: str):
    y_true  = torch.tensor([]).cuda()
    y_score = torch.tensor([]).cuda()

    with torch.no_grad():
        for inputs, targets in tqdm(loader, desc=f"  {split:5s}", leave=False):
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs = model(inputs).softmax(dim=-1)

            if task == "multi-label, binary-class":
                targets = targets.float()
            else:
                targets = targets.squeeze().float().reshape(-1, 1)

            y_true  = torch.cat((y_true,  targets), 0)
            y_score = torch.cat((y_score, outputs), 0)

    y_true  = y_true.cpu().numpy()
    y_score = y_score.detach().cpu().numpy()

    evaluator = Evaluator(data_flag, split)
    auc, acc = evaluator.evaluate(y_score)
    return float(auc), float(acc)


# ---------------------------------------------------------------------------
# Single checkpoint evaluation
# ---------------------------------------------------------------------------

def evaluate_one(model_name: str, data_flag: str, ckpt_path: str) -> dict:
    print(f"\n  model={model_name}  dataset={data_flag}")
    print(f"  ckpt={ckpt_path}")

    info      = INFO[data_flag]
    task      = info["task"]
    n_classes = len(info["label"])

    model = load_model(model_name, ckpt_path, n_classes)

    row = {"model": model_name, "dataset": data_flag, "ckpt": ckpt_path}
    for split in SPLITS:
        loader, _ = build_loader(data_flag, split)
        auc, acc  = evaluate_split(model, loader, data_flag, split, task)
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
    p.add_argument("--dataset",   choices=DATASETS, default=None,
                   help="MedMNIST dataset flag (required with --ckpt)")
    p.add_argument("--ckpt_name", default="best_val.pth",
                   help="Checkpoint filename to look for in sweep mode "
                        "(default: best_val.pth)")
    p.add_argument("--out_csv",   default="eval_results.csv",
                   help="Output CSV path (default: eval_results.csv)")
    return p.parse_args()


def main():
    args = parse_args()
    jobs = []  # list of (model_name, data_flag, ckpt_path)

    if args.ckpt:
        # Single explicit checkpoint
        if not args.model or not args.dataset:
            print("ERROR: --model and --dataset are required when using --ckpt")
            sys.exit(1)
        if not os.path.isfile(args.ckpt):
            print(f"ERROR: checkpoint not found: {args.ckpt}")
            sys.exit(1)
        jobs.append((args.model, args.dataset, args.ckpt))
    else:
        # Auto-sweep outputs/{model}/{dataset}/{ckpt_name}
        print(f"Sweeping {OUTPUT_ROOT}/ for '{args.ckpt_name}' ...")
        for model_name in MODELS:
            model_dir = os.path.join(OUTPUT_ROOT, model_name)
            if not os.path.isdir(model_dir):
                continue
            for data_flag in os.listdir(model_dir):
                ckpt_path = os.path.join(model_dir, data_flag, args.ckpt_name)
                if os.path.isfile(ckpt_path):
                    jobs.append((model_name, data_flag, ckpt_path))

    if not jobs:
        print("No checkpoints found. Nothing to evaluate.")
        sys.exit(0)

    print(f"Found {len(jobs)} checkpoint(s) to evaluate.")

    fieldnames = ["model", "dataset", "ckpt",
                  "train_auc", "train_acc",
                  "val_auc",   "val_acc",
                  "test_auc",  "test_acc"]

    rows = []
    for model_name, data_flag, ckpt_path in jobs:
        try:
            row = evaluate_one(model_name, data_flag, ckpt_path)
            rows.append(row)
        except Exception as exc:
            print(f"  [ERROR] {model_name}/{data_flag}: {exc}")
            rows.append({"model": model_name, "dataset": data_flag,
                         "ckpt": ckpt_path, "test_auc": f"ERROR: {exc}"})

    with open(args.out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nResults saved to {args.out_csv}")

    # Pretty-print summary table
    print(f"\n{'Model':<8}  {'Dataset':<14}  "
          f"{'Train AUC':>9}  {'Val AUC':>7}  {'Test AUC':>8}  {'Test ACC':>8}")
    print("-" * 65)
    for r in rows:
        print(f"{r.get('model',''):<8}  {r.get('dataset',''):<14}  "
              f"{r.get('train_auc','N/A'):>9}  {r.get('val_auc','N/A'):>7}  "
              f"{r.get('test_auc','N/A'):>8}  {r.get('test_acc','N/A'):>8}")


if __name__ == "__main__":
    main()
