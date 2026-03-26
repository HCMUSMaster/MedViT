#!/usr/bin/env bash

set -euo pipefail

CHECKPOINT_ID="14wcH5cm8P63cMZAUHA1lhhJgMVOw_5VQ"
CHECKPOINT_NAME="MedViT_small_im1k.pth"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGET_DIR="${SCRIPT_DIR}/../ckpt"

mkdir -p "${TARGET_DIR}"

if [[ -f "${TARGET_DIR}/${CHECKPOINT_NAME}" ]]; then
	echo "Checkpoint already exists: ${TARGET_DIR}/${CHECKPOINT_NAME}"
	exit 0
fi

gdown --id "${CHECKPOINT_ID}" -O "${TARGET_DIR}/${CHECKPOINT_NAME}"

echo "Saved checkpoint to ${TARGET_DIR}/${CHECKPOINT_NAME}"
