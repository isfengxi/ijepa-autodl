

# I-JEPA (AutoDL Adapted Version)

This repository is an AutoDL-adapted version of the official I-JEPA implementation from Meta AI.

Original repository:
🔗 [https://github.com/facebookresearch/ijepa](https://github.com/facebookresearch/ijepa)

This version includes:

* AutoDL-compatible configs
* CSI dataset support
* Simplified single-GPU training
* Cleaner project structure for reproducibility

---

## 📂 Project Structure

```text
ijepa/
├── configs/
├── src/
│   ├── datasets/
│   ├── masks/
│   ├── models/
│   ├── utils/
│   ├── helper.py
│   ├── train.py
│   └── transforms.py
├── main.py
├── main_distributed.py
└── README.md
```

---

## 🧠 What is I-JEPA?

I-JEPA (Image Joint Embedding Predictive Architecture) is a self-supervised representation learning framework introduced by Meta AI.

Instead of reconstructing pixels (like MAE), I-JEPA predicts representations of masked image blocks in embedding space.

---

## ⚙️ Environment Setup (AutoDL)

### 1️⃣ Create environment

```bash
conda create -n ijepa python=3.10 -y
conda activate ijepa
```

### 2️⃣ Install dependencies

```bash
pip install torch torchvision timm numpy pyyaml
```

If using ImageNet training, make sure:

* torchvision supports ImageFolder
* CUDA is correctly configured

---

## 📊 Dataset Preparation

### CSI Dataset

Place dataset under:

```bash
/root/autodl-tmp/csi_preprocessed/
```

Then modify config:

```yaml
data:
  root_path: /root/autodl-tmp
```

---

### ImageNet (Optional)

Place ImageNet root under:

```bash
/root/autodl-tmp/ImageNet/
```

Ensure folder structure:

```text
ImageNet/
├── train/
│   ├── class1/
│   ├── class2/
│   └── ...
└── val/
```

---

## 🚀 Training

### 🔹 Single GPU

```bash
python -u main.py \
  --fname configs/csi_autodl.yaml \
  --devices cuda:0
```

---

### 🔹 Distributed Training

```bash
python -m torch.distributed.run \
  --nproc_per_node=2 \
  main_distributed.py \
  --fname configs/in1k_vith14_ep300_autodl.yaml
```

---

## 📁 Output

Training outputs are saved to:

```yaml
logging:
  folder: /root/autodl-tmp/ijepa_runs/
```

Checkpoints and logs are automatically generated.

---

## 🔬 Differences from Official Repo

This version:

* Removes unnecessary distributed complexity for single-GPU
* Adds CSI dataset loader
* Provides AutoDL-ready configs
* Cleans git structure and ignores training artifacts

---

## 📌 Notes

* Do not commit datasets or checkpoints.
* Ensure `.gitignore` excludes large files.
* Recommended batch size depends on GPU memory.

---

## 📜 License

This project follows the original license from Meta AI.

---

## 🙌 Acknowledgement

Based on:

Meta AI – I-JEPA
[https://github.com/facebookresearch/ijepa](https://github.com/facebookresearch/ijepa)

---

# Optional: Add Experiment Tracking Section (Recommended)

If you want, we can add:

* Experiment logs table
* Model comparison (MAE vs I-JEPA)
* Pretraining + reconstruction explanation
* Citation section

---

如果你愿意，我可以下一步帮你：

* ✅ 写一个 **更学术版本 README（含公式解释）**
* ✅ 加一个 “MAE vs I-JEPA” 对比表
* ✅ 增加 Reproducibility checklist
* ✅ 加自动训练脚本目录 `scripts/`
* ✅ 让仓库达到“论文级开源质量”

你现在这个项目，已经可以往“科研可展示仓库”方向进化了。
