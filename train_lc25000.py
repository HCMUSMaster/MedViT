"""
train_lc25000.py — Train MedViT on the LC25000 lung/colon histopathology dataset.

Dataset structure expected:
  <data_root>/lung_colon_image_set/Train and Validation Set/{class}/
  <data_root>/lung_colon_image_set/Test Set/{class}/

5 classes: colon_aca, colon_n, lung_aca, lung_n, lung_scc

The "Train and Validation Set" folder is split 80/20 into train and val.

Outputs (outputs/{model}/lc25000/{mode}/):
  best_val.pth               — checkpoint with best validation AUC
  last.pth                   — checkpoint from the final epoch
  {model}_lc25000_{mode}.log — per-epoch log

results_summary_lc25000.csv — final test AUC/ACC for every run

Usage:
    python train_lc25000.py
    python train_lc25000.py --models small --freeze_backbone
    python train_lc25000.py --ckpt outputs/small/lc25000/finetune/best_val.pth --models small
    python train_lc25000.py --data_root /path/to/lc25000
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
import torchvision.transforms as transforms
from sklearn.metrics import accuracy_score, roc_auc_score
from torch.utils.data import random_split
from torchvision.datasets import ImageFolder
from tqdm import tqdm

from MedViT import MedViT_small, MedViT_base, MedViT_large

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

NUM_EPOCHS  = 2
BATCH_SIZE  = 24
LR          = 0.005
MOMENTUM    = 0.9
INPUT_SIZE  = 224
PATIENCE    = 2
VAL_SPLIT   = 0.2          # fraction of train_and_val used for validation

ALL_MODELS = {
    "small": (MedViT_small, "ckpt/MedViT_small_im1k.pth"),
    "pathmnist": (MedViT_small, "outputs/small/pathmnist/best_val.pth"),
    "base":  (MedViT_base,  "ckpt/MedViT_base_im1k.pth"),
    "large": (MedViT_large, "ckpt/MedViT_large_im1k.pth"),
}

OUTPUT_ROOT = "outputs"
SUMMARY_CSV = "results_summary_lc25000.csv"

TRAINVAL_DIR = "lung_colon_image_set/Train and Validation Set"
TEST_DIR     = "lung_colon_image_set/Test Set"

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

    trainval_path = os.path.join(data_root, TRAINVAL_DIR)
    test_path     = os.path.join(data_root, TEST_DIR)

    # Load full train+val folder with eval transform first to get class info
    full_ds = ImageFolder(trainval_path, transform=eval_transform)
    classes   = full_ds.classes
    n_classes = len(classes)

    # Split into train / val
    n_val   = int(len(full_ds) * VAL_SPLIT)
    n_train = len(full_ds) - n_val
    train_ds_eval, val_ds = random_split(
        full_ds, [n_train, n_val],
        generator=torch.Generator().manual_seed(42)
    )

    # Re-wrap train split with augmentation transform
    train_ds_aug = ImageFolder(trainval_path, transform=train_transform)
    train_ds = data.Subset(train_ds_aug, train_ds_eval.indices)

    test_ds = ImageFolder(test_path, transform=eval_transform)

    train_loader = data.DataLoader(train_ds, batch_size=BATCH_SIZE,     shuffle=True,  num_workers=4, pin_memory=True)
    val_loader   = data.DataLoader(val_ds,   batch_size=BATCH_SIZE * 2, shuffle=False, num_workers=4, pin_memory=True)
    test_loader  = data.DataLoader(test_ds,  batch_size=BATCH_SIZE * 2, shuffle=False, num_workers=4, pin_memory=True)

    return train_loader, val_loader, test_loader, classes, n_classes


def build_model(model_name: str, ckpt_path: str, n_classes: int) -> nn.Module:
    model = ALL_MODELS[model_name][0]()
    model.proj_head[0] = nn.Linear(1024, n_classes, bias=True)

    ckpt  = torch.load(ckpt_path, map_location="cpu")
    state = ckpt.get("model_state_dict", ckpt.get("model", ckpt))

    # Skip shape-mismatched keys (e.g. ImageNet head with 1000 classes)
    model_state = model.state_dict()
    for k in list(state.keys()):
        if k in model_state and state[k].shape != model_state[k].shape:
            print(f"  [INFO] shape mismatch, skipping: {k} "
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


def evaluate(model: nn.Module, loader: data.DataLoader,
             n_classes: int, desc: str = "eval") -> tuple[float, float]:
    """Return (macro-OvR AUC, accuracy) using sklearn."""
    model.eval()
    all_targets = []
    all_scores  = []

    with torch.no_grad():
        for inputs, targets in tqdm(loader, desc=f"  {desc:5s}", leave=False):
            inputs = inputs.cuda()
            scores = model(inputs).softmax(dim=-1).cpu().numpy()
            all_scores.append(scores)
            all_targets.append(targets.numpy())

    y_true  = np.concatenate(all_targets)
    y_score = np.concatenate(all_scores)   # shape (N, n_classes)

    auc = roc_auc_score(y_true, y_score,
                        multi_class="ovr", average="macro")
    acc = accuracy_score(y_true, y_score.argmax(axis=1))
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
    mode    = "frozen" if freeze else "finetune"
    ds_dir  = "lc25000_freeze" if freeze else "lc25000"
    run_tag = f"{model_name}_{ds_dir}"
    out_dir = os.path.join(OUTPUT_ROOT, model_name, ds_dir)
    os.makedirs(out_dir, exist_ok=True)

    logger = get_logger(os.path.join(out_dir, f"{run_tag}.log"))
    logger.info(f"=== run={run_tag} ===")
    logger.info(f"model={model_name}  dataset=lc25000  mode={mode}  init_ckpt={ckpt_path}")

    train_loader, val_loader, test_loader, classes, n_classes = build_loaders(data_root)
    logger.info(f"classes ({n_classes}): {classes}")
    logger.info(f"train={len(train_loader.dataset):_}  "
                f"val={len(val_loader.dataset):_}  "
                f"test={len(test_loader.dataset):_}")

    model = build_model(model_name, ckpt_path, n_classes)

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
                                    desc=f"[{run_tag}] Epoch {epoch+1}/{NUM_EPOCHS}",
                                    leave=False):
            inputs, targets = inputs.cuda(), targets.cuda()
            optimizer.zero_grad()
            loss = criterion(model(inputs), targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            n_batches    += 1

        avg_loss = running_loss / max(n_batches, 1)
        val_auc, val_acc = evaluate(model, val_loader, n_classes, desc="val")

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
    val_auc, val_acc = evaluate(model, val_loader, n_classes, desc="val")
    save_checkpoint(last_ckpt_path, model, optimizer, epoch, val_auc, val_acc)

    # Final test evaluation
    test_auc, test_acc = evaluate(model, test_loader, n_classes, desc="test")
    logger.info(f"FINAL  test_auc={test_auc:.4f}  test_acc={test_acc:.4f}  best_val_auc={best_val_auc:.4f}")

    summary_writer.writerow({
        "model":        model_name,
        "dataset":      "lc25000",
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
    p.add_argument("--data_root", default="./dataset/lc25000",
                   help="Root of LC25000 dataset "
                        "(default: ./dataset/lc25000)")
    p.add_argument("--models", nargs="+", choices=list(ALL_MODELS),
                   default=list(ALL_MODELS),
                   help="Model variants to train (default: all)")
    p.add_argument("--ckpt", default=None,
                   help="Custom init checkpoint, overrides the default ImageNet "
                        "pretrained ckpt. Accepts both ImageNet and fine-tuned ckpts.")
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
            mode_label = "frozen" if args.freeze_backbone else "finetune"
            print(f"\n{'='*60}")
            print(f"  START  model={model_name}  dataset=lc25000  mode={mode_label}")
            print(f"  ckpt={ckpt_path}")
            print(f"{'='*60}")
            t0 = time.time()
            try:
                run_one(model_name, ckpt_path, args.data_root, writer,
                        freeze=args.freeze_backbone)
                csv_file.flush()
            except Exception as exc:
                print(f"  [ERROR] {model_name}/lc25000: {exc}")
                writer.writerow({
                    "model": model_name, "dataset": "lc25000", "mode": mode_label,
                    "best_val_auc": "ERROR", "test_auc": "ERROR",
                    "test_acc": str(exc)[:80],
                })
                csv_file.flush()
            print(f"  DONE  ({(time.time()-t0)/60:.1f} min)")

    print(f"\nAll runs complete. Summary saved to {SUMMARY_CSV}")


if __name__ == "__main__":
    main()
