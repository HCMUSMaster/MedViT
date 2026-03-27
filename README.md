# MedViT: A Robust Vision Transformer for Generalized Medical Image Classification

[![Paper](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/2302.09462)
[![Paper](https://img.shields.io/badge/Elsevier-CIBM-blue)](https://doi.org/10.1016/j.compbiomed.2023.106791)

## Project Members

| Student ID | Full Name           |
|------------|---------------------|
| 25C11057   | Dương Tấn Phát      |
| 25C11027   | Bùi Quốc Việt       |
| 25C11034   | Đinh Hoàng Dương    |
| 25C11022   | Đặng Anh Tiến       |

## Introduction
This project is part of the Artificial Intelligence course (K35) at HCMUS. We reproduce the **MedViT** model and extend experiments to two additional datasets:

* **PCAM**: [https://github.com/basveeling/pcam](https://github.com/basveeling/pcam)
* **LC25000**: [https://github.com/tampapath/lung_colon_image_set](https://github.com/tampapath/lung_colon_image_set)

---

## Installation

Install dependencies using:

```bash
pip install -e .
```

---

## Training

Run training scripts for each dataset as follows:

### 1. MedMNIST

```bash
python3 train/train_all_medmnist.py
```

### 2. PCAM

```bash
python3 train/train_pcam.py
```

### 3. LC25000

```bash
python3 train/train_lc25000.py
```

> 💡 You can customize training via additional arguments such as:
>
> * `--ckpt` (load pretrained checkpoint)
> * `--freeze-backbone` (freeze feature extractor)

---

## Evaluation

Evaluate trained models using the corresponding scripts:

### 1. MedMNIST

```bash
python3 test/evaluate_mnist_ckpt.py \
    --model <model_name> \
    --dataset <dataset_name> \
    --data_root <path_to_data> \
    --ckpt <checkpoint_path>
```

### 2. PCAM

```bash
python3 test/evaluate_pcam_ckpt.py \
    --model <model_name> \
    --dataset <dataset_name> \
    --data_root <path_to_data> \
    --ckpt <checkpoint_path>
```

### 3. LC25000

```bash
python3 test/evaluate_lc25000_ckpt.py \
    --model <model_name> \
    --dataset <dataset_name> \
    --data_root <path_to_data> \
    --ckpt <checkpoint_path>
```

## Overview

<div style="text-align: center">
<img src="images/structure.png" title="MedViT-S" height="75%" width="75%">
</div>
Figure 2. The overall hierarchical architecture of MedViT.</center>

## ImageNet Pre-train
We provide a series of MedViT models pretrained on ILSVRC2012 ImageNet-1K dataset.

| Model      |   Dataset   | Resolution  | Acc@1 | ckpt   |  
|------------|:-----------:|:----------:|:--------:|:--------:|
| MedViT_small | ImageNet-1K |    224   | 83.70 | [ckpt](https://drive.google.com/file/d/14wcH5cm8P63cMZAUHA1lhhJgMVOw_5VQ/view?usp=sharing) | 
| MedViT_base | ImageNet-1K |    224    | 83.92 |[ckpt](https://drive.google.com/file/d/1Lrfzjf3CK7YOztKa8D6lTUZjYJIiT7_s/view?usp=sharing) | 
| MedViT_large | ImageNet-1K |    224   | 83.96 |[ckpt](https://drive.google.com/file/d/1sU-nLpYuCI65h7MjFJKG0yphNAlUFSKG/view?usp=sharing) | 



## Acknowledgement
We acknowledge the authors of MedViT and reuse parts of their implementation from the official repository: [Omid-Nejati/MedViT](https://github.com/Omid-Nejati/MedViT)

```
@article{manzari2023medvit,
  title={MedViT: A robust vision transformer for generalized medical image classification},
  author={Manzari, Omid Nejati and Ahmadabadi, Hamid and Kashiani, Hossein and Shokouhi, Shahriar B and Ayatollahi, Ahmad},
  journal={Computers in Biology and Medicine},
  volume={157},
  pages={106791},
  year={2023},
  publisher={Elsevier}
}
```
