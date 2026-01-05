# Multimodal Medical Mini-Research (ROCOv2)

This repository contains a small educational research project exploring **multimodal learning**
on **radiology images and captions** using the ROCOv2 dataset.

The project builds a **non-diagnostic derived task** (medical image modality classification)
and compares image-only, text-only, and multimodal approaches.

---

## Task Definition

**Input:**
- Medical image
- Corresponding radiology caption

**Output (derived labels):**
- `ct`
- `xray`
- `mri`
- `ultrasound`
- `other`

⚠️ This project does **NOT** perform disease diagnosis.

---

## Dataset

- Dataset: **ROCOv2 (Radiology Objects in Context)**
- Source: Hugging Face
- Modality: image + text

The dataset is **streamed**, and a small subset (e.g. 2000 samples) is saved locally
as `dataset.npz`.

The generated dataset file is **not included** in this repository.

---

## Methods

The following baselines are implemented:

1. **Image-only**
   - CLIP image embeddings
   - Logistic Regression

2. **Text-only**
   - TF-IDF on captions
   - Logistic Regression

3. **Multimodal**
   - CLIP image embeddings + CLIP text embeddings
   - Feature concatenation + Logistic Regression

---

## Setup

### Option 1: pip
```bash
pip install -r requirements.txt
