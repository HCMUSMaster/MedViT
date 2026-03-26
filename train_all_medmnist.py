"""
train_all.py — Train MedViT (small/base/large) on all MedMNIST datasets.

Outputs per run (outputs/{model}/{dataset}/):
  best_val.pth  — checkpoint with best validation AUC
  last.pth      — checkpoint from the final epoch
  train.log     — per-epoch log

results_summary.csv — final test AUC/ACC for every (model, dataset) pair
"""

import os
import csv
import logging
import sys
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm

import medmnist
from medmnist import INFO, Evaluator

from MedViT import MedViT_small, MedViT_base, MedViT_large

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

NUM_EPOCHS = 100
BATCH_SIZE = 24
LR = 0.005
MOMENTUM = 0.9
INPUT_SIZE = 224
DOWNLOAD = True

MODELS = {
    "small": (MedViT_small, "ckpt/MedViT_small_im1k.pth"),
    # "base":  (MedViT_base,  "ckpt/MedViT_base_im1k.pth"),
    # "large": (MedViT_large, "ckpt/MedViT_large_im1k.pth"),
}

DATASETS = [
    "bloodmnist",
    "pathmnist",
    "chestmnist",
    "dermamnist",
    "octmnist",
    "pneumoniamnist",
    "retinamnist",
    "breastmnist",
    "tissuemnist",
    "organamnist",
    "organcmnist",
    "organsmnist",
]

OUTPUT_ROOT = "outputs"
SUMMARY_CSV = "results_summary.csv"


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


def build_loaders(data_flag: str):
    info = INFO[data_flag]
    DataClass = getattr(medmnist, info["python_class"])

    train_transform = transforms.Compose([
        transforms.Resize(INPUT_SIZE),
        transforms.Lambda(lambda img: img.convert("RGB")),
        torchvision.transforms.AugMix(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])
    eval_transform = transforms.Compose([
        transforms.Resize(INPUT_SIZE),
        transforms.Lambda(lambda img: img.convert("RGB")),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])

    train_ds = DataClass(split="train", transform=train_transform, download=DOWNLOAD)
    val_ds   = DataClass(split="val",   transform=eval_transform,  download=DOWNLOAD)
    test_ds  = DataClass(split="test",  transform=eval_transform,  download=DOWNLOAD)

    train_loader = data.DataLoader(train_ds, batch_size=BATCH_SIZE,     shuffle=True,  num_workers=4, pin_memory=True)
    val_loader   = data.DataLoader(val_ds,   batch_size=BATCH_SIZE * 2, shuffle=False, num_workers=4, pin_memory=True)
    test_loader  = data.DataLoader(test_ds,  batch_size=BATCH_SIZE * 2, shuffle=False, num_workers=4, pin_memory=True)

    return train_loader, val_loader, test_loader, info


def build_model(model_fn, pretrained_path: str, n_classes: int) -> nn.Module:
    model = model_fn()

    # Load pretrained ImageNet weights, skipping proj_head (wrong num_classes)
    ckpt = torch.load(pretrained_path, map_location="cpu")
    state_dict = ckpt["model"] if "model" in ckpt else ckpt
    for k in list(state_dict.keys()):
        if k.startswith("proj_head"):
            del state_dict[k]
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    # Only proj_head keys should be missing — everything else must load
    non_head_missing = [k for k in missing if not k.startswith("proj_head")]
    if non_head_missing:
        print(f"  [WARN] Unexpected missing keys: {non_head_missing}")

    # Replace classification head
    model.proj_head[0] = nn.Linear(in_features=1024, out_features=n_classes, bias=True)
    model = model.cuda()
    return model


def evaluate(model, loader, data_flag: str, split: str, task: str) -> tuple[float, float]:
    """Return (auc, acc) using medmnist Evaluator."""
    model.eval()
    y_true  = torch.tensor([]).cuda()
    y_score = torch.tensor([]).cuda()

    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs = model(inputs)

            if task == "multi-label, binary-class":
                targets = targets.to(torch.float32)
                outputs = outputs.softmax(dim=-1)
            else:
                targets = targets.squeeze().long()
                outputs = outputs.softmax(dim=-1)
                targets = targets.float().reshape(-1, 1)

            y_true  = torch.cat((y_true,  targets), 0)
            y_score = torch.cat((y_score, outputs), 0)

    y_true  = y_true.cpu().numpy()
    y_score = y_score.detach().cpu().numpy()

    evaluator = Evaluator(data_flag, split)
    auc, acc = evaluator.evaluate(y_score)
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

def run_one(model_name: str, model_fn, pretrained_path: str, data_flag: str,
            summary_writer):
    out_dir = os.path.join(OUTPUT_ROOT, model_name, data_flag)
    os.makedirs(out_dir, exist_ok=True)

    log_path = os.path.join(out_dir, "train.log")
    logger   = get_logger(log_path)
    logger.info(f"=== model={model_name}  dataset={data_flag} ===")

    # Data
    train_loader, val_loader, test_loader, info = build_loaders(data_flag)
    task      = info["task"]
    n_classes = len(info["label"])
    logger.info(f"task={task}  n_classes={n_classes}")

    # Model
    model = build_model(model_fn, pretrained_path, n_classes)
    logger.info(f"params: {sum(p.numel() for p in model.parameters()):_}")

    # Loss & optimizer
    if task == "multi-label, binary-class":
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM)

    best_val_auc   = -1.0
    patience       = 5
    no_improve     = 0
    best_ckpt_path = os.path.join(out_dir, "best_val.pth")
    last_ckpt_path = os.path.join(out_dir, "last.pth")

    # Training loop
    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0
        n_batches    = 0

        for inputs, targets in tqdm(train_loader,
                                    desc=f"[{model_name}/{data_flag}] Epoch {epoch+1}/{NUM_EPOCHS}",
                                    leave=False):
            inputs, targets = inputs.cuda(), targets.cuda()
            optimizer.zero_grad()
            outputs = model(inputs)

            if task == "multi-label, binary-class":
                targets = targets.to(torch.float32)
                loss = criterion(outputs, targets)
            else:
                targets = targets.squeeze().long()
                loss = criterion(outputs, targets)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            n_batches    += 1

        avg_loss = running_loss / max(n_batches, 1)

        # Validation
        val_auc, val_acc = evaluate(model, val_loader, data_flag, "val", task)
        logger.info(
            f"Epoch [{epoch+1:3d}/{NUM_EPOCHS}]  loss={avg_loss:.4f}  "
            f"val_auc={val_auc:.4f}  val_acc={val_acc:.4f}"
        )

        # Save best checkpoint / early stopping
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            no_improve   = 0
            save_checkpoint(best_ckpt_path, model, optimizer, epoch, val_auc, val_acc)
            logger.info(f"  --> new best val AUC={val_auc:.4f}  (epoch {epoch+1})")
        else:
            no_improve += 1
            logger.info(f"  no improvement ({no_improve}/{patience})")
            if no_improve >= patience:
                logger.info(f"  early stopping at epoch {epoch+1}")
                break

    # Save last checkpoint
    val_auc, val_acc = evaluate(model, val_loader, data_flag, "val", task)
    save_checkpoint(last_ckpt_path, model, optimizer, epoch, val_auc, val_acc)

    # Final test evaluation
    test_auc, test_acc = evaluate(model, test_loader, data_flag, "test", task)
    logger.info(f"FINAL  test_auc={test_auc:.4f}  test_acc={test_acc:.4f}  best_val_auc={best_val_auc:.4f}")

    summary_writer.writerow({
        "model":        model_name,
        "dataset":      data_flag,
        "best_val_auc": f"{best_val_auc:.4f}",
        "test_auc":     f"{test_auc:.4f}",
        "test_acc":     f"{test_acc:.4f}",
    })

    # Clean up CUDA memory between runs
    del model
    torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    with open(SUMMARY_CSV, "w", newline="") as csv_file:
        fieldnames = ["model", "dataset", "best_val_auc", "test_auc", "test_acc"]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        for model_name, (model_fn, pretrained_path) in MODELS.items():
            for data_flag in DATASETS:
                t0 = time.time()
                print(f"\n{'='*60}")
                print(f"  START  model={model_name}  dataset={data_flag}")
                print(f"{'='*60}")
                try:
                    run_one(model_name, model_fn, pretrained_path, data_flag, writer)
                    csv_file.flush()
                except Exception as exc:
                    print(f"  [ERROR] {model_name}/{data_flag}: {exc}")
                    writer.writerow({
                        "model":        model_name,
                        "dataset":      data_flag,
                        "best_val_auc": "ERROR",
                        "test_auc":     "ERROR",
                        "test_acc":     str(exc)[:80],
                    })
                    csv_file.flush()
                elapsed = time.time() - t0
                print(f"  DONE  ({elapsed/60:.1f} min)")

    print(f"\nAll runs complete. Summary saved to {SUMMARY_CSV}")


if __name__ == "__main__":
    main()
