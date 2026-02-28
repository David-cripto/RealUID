# RealUID

Official PyTorch implementation and pretrained checkpoints for our **ICLR 2026 Oral** paper, **“Universal Inverse Distillation for Matching Models with Real-Data Supervision (No GANs)”**, which introduces **RealUID**, an adversarial flow-matching distillation approach for **one-step image generation**.

---

## What’s in this repository

This codebase supports:

- **Teacher Flow Matching (FM)** models
- **One-step distillation** into generators using **RealUID**
- Optional **RealUID + GAN** training
- **Sampling** and **FID evaluation** scripts

Datasets:

- **CIFAR-10**: unconditional + class-conditional
- **CelebA**: unconditional (expects a preprocessed folder of images)

---

## Checkpoints and data

1) Create the expected folders:

```bash
mkdir -p model_checkpoints data
```

2) Download pretrained checkpoints to `./model_checkpoints/`:

- **Google Drive (checkpoints):**
  https://drive.google.com/drive/folders/1PjZPmnk31eUTjbLHddwl8tjGt2apNO_P?usp=sharing

3) CelebA (only if you run CelebA experiments):

- Download `cut_celeba_full.zip` and extract it to:

```bash
unzip cut_celeba_full.zip -d data/cut_celeba_full
```

Expected structure:

```text
data/cut_celeba_full/
  0.jpg
  1.jpg
  ...
  202598.jpg
```

---

## Installation

This repo is pure Python. You’ll need PyTorch + a few common libraries.

A minimal setup (example):

```bash
# Python 3.9+ recommended
pip install torch torchvision
pip install torchcfm torchdyn clean-fid tqdm numpy pillow

# Optional (logging)
pip install tensorboard wandb
```

---

## Quickstart

### Generate samples

**One-step generator (distilled checkpoint):**

```bash
python3 generate_samples.py \
  --model_ckpt_path ./model_checkpoints/cifar10_uncond_distil.pt \
  --mode one_step \
  --dataset cifar10 \
  --cond uncond \
  --output_dir ./results \
  --num_save_image 100
```

**Multi-step sampling from a teacher FM vector field:**

```bash
python3 generate_samples.py \
  --model_ckpt_path ./model_checkpoints/cifar10_uncond_FM_400000.pt \
  --mode multi_step \
  --dataset cifar10 \
  --cond uncond \
  --output_dir ./results \
  --num_save_image 100
```

### Evaluate FID

**One-step generator (distilled checkpoint):**

```bash
python3 eval_model.py \
  --model_ckpt_path ./model_checkpoints/cifar10_uncond_distil.pt \
  --mode one_step \
  --dataset cifar10 \
  --cond uncond \
  --output_dir ./results
```

**Teacher FM (multi-step):**

```bash
python3 eval_model.py \
  --model_ckpt_path ./model_checkpoints/cifar10_uncond_FM_400000.pt \
  --mode multi_step \
  --dataset cifar10 \
  --cond uncond \
  --output_dir ./results
```

**CelebA (requires `--celeba_dir`):**

```bash
python3 eval_model.py \
  --model_ckpt_path ./model_checkpoints/celeba_uncond_distil.pt \
  --mode one_step \
  --dataset celeba \
  --cond uncond \
  --celeba_dir ./data/cut_celeba_full \
  --output_dir ./results
```

---

## Training

You can either:

- Use the provided `run.sh` convenience launcher, or
- Run `train_teacher.py` / `train_distil.py` directly.

### Common options

- Adjust memory:
  - `--batch_size`
  - `--fid_batch_size`
- Track EMA models:
  - `--ema_decays 0.999 0.9996 0.9999`
- Logging:
  - TensorBoard: `--logger_name tensorboard`
  - Weights & Biases: `--logger_name wandb --wandb_user ... --wandb_api_key ...`
- Resume training:
  - `--resume_ckpt_path /path/to/previous_checkpoint.pt`

### 1) Train teacher Flow Matching models

```bash
bash run.sh cifar10-uncond-teacher
bash run.sh cifar10-cond-teacher
bash run.sh celeba-teacher
```

### 2) Distill one-step generators with RealUID

```bash
bash run.sh cifar10-uncond-distil
bash run.sh cifar10-cond-distil
bash run.sh celeba-distil
```

Tuning notes:

- You can vary `--alpha` and `--beta` (typically in **[0.85, 1.0]**).
- We recommend keeping **`beta/alpha ≈ 1`** (e.g. **1.02** or **0.98**).
- The setting `alpha = beta = 1.0` corresponds to the **data-free** case.

### 3) Distill with GAN loss (optional)

```bash
bash run.sh cifar10-uncond-distil-gan
bash run.sh cifar10-cond-distil-gan
bash run.sh celeba-distil-gan
```

- We recommend keeping **`disc_coef/gen_coef ≈ 3`**.
- You can combine GAN + RealUID by setting `--alpha` and `--beta` to values different from 1.0.

### 4) Fine-tune the best checkpoint with RealUID

1) Put your best distilled checkpoint into `./model_checkpoints/` and name it:

- `cifar10_uncond_distil.pt` or
- `cifar10_cond_distil.pt`

2) Run fine-tuning:

```bash
bash run.sh cifar10-uncond-distil-finetune
bash run.sh cifar10-cond-distil-finetune
```

---

## Checkpoint formats

This matters for sampling/evaluation:

- **Teacher checkpoints** (saved by `train_teacher.py`) contain:
  - `net_model`, `ema_models`
  - Use with `--mode multi_step`

- **Distilled checkpoints** (saved by `train_distil.py`) contain:
  - `gen`, `u`, `ema_gens`
  - Use with `--mode one_step`

---

## Citation

If you find this work useful or build upon its results in your research, please consider citing the paper:

```bibtex
@inproceedings{
    kornilov2026universal,
    title={Universal Inverse Distillation for Matching Models with Real-Data Supervision (No {GAN}s)},
    author={Nikita Maksimovich Kornilov and David Li and Tikhon Mavrin and Aleksei Leonov and Nikita Gushchin and Evgeny Burnaev and Iaroslav Sergeevich Koshelev and Alexander Korotin},
    booktitle={The Fourteenth International Conference on Learning Representations},
    year={2026},
    url={https://openreview.net/forum?id=8NuN5UzXLC}
}
```

---

## Contact

Please feel free to contact the authors should you have any questions regarding the paper:

- **Nikita Kornilov**: jhomanik14@gmail.com
- **David Li**: David.Li@mbzuai.ac.ae
- **Tikhon Mavrin**: tixonmavrin@gmail.com
- **Alexander Korotin**: iamalexkorotin@gmail.com

