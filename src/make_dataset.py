import argparse
import re
from datasets import load_dataset
from tqdm import tqdm
import numpy as np

DATASET_NAME = "eltorio/ROCOv2-radiology"  # HF dataset (image + caption) :contentReference[oaicite:1]{index=1}

# Safe, non-diagnostic label rules (keyword → modality-ish bucket)
LABEL_RULES = [
    ("xray", r"\b(x[\s-]?ray|radiograph|cxr)\b"),
    ("ct", r"\b(ct|computed tomography)\b"),
    ("mri", r"\b(mri|magnetic resonance)\b"),
    ("ultrasound", r"\b(ultrasound|sonograph|sonography)\b"),
]

def label_from_caption(caption: str) -> str:
    c = caption.lower()
    for name, pattern in LABEL_RULES:
        if re.search(pattern, c):
            return name
    return "other"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", type=str, default="train", help="train/validation/test if available")
    ap.add_argument("--n", type=int, default=2000, help="number of examples to sample")
    ap.add_argument("--out", type=str, default="dataset.npz")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    # Streaming avoids downloading everything
    ds = load_dataset(DATASET_NAME, split=args.split, streaming=True)

    images = []
    captions = []
    labels = []

    rng = np.random.default_rng(args.seed)
    # Simple reservoir-ish sampling from stream: take first N*2 then randomly choose N
    buffer = []
    for i, ex in enumerate(tqdm(ds, desc="Streaming ROCOv2")):
        if "image" not in ex:
            continue
        cap = ex.get("caption") or ex.get("text") or ex.get("sentences")
        if cap is None:
            continue
        if isinstance(cap, list):
            cap = cap[0] if cap else ""
        cap = str(cap).strip()
        if not cap:
            continue

        buffer.append((ex["image"], cap))
        if len(buffer) >= args.n * 2:
            break

    if len(buffer) < 100:
        raise SystemExit("Not enough examples pulled from stream. Try a different split or increase buffer.")

    idx = rng.choice(len(buffer), size=min(args.n, len(buffer)), replace=False)

    for j in idx:
        img, cap = buffer[j]
        lab = label_from_caption(cap)
        images.append(img)      # PIL image objects
        captions.append(cap)
        labels.append(lab)

    # Save images as uint8 arrays (224 resize happens later)
    # Keep raw small to avoid huge file: store as PNG bytes
    import io
    png_bytes = []
    for img in tqdm(images, desc="Encoding PNG"):
        bio = io.BytesIO()
        img.save(bio, format="PNG")
        png_bytes.append(bio.getvalue())

    np.savez_compressed(
        args.out,
        png_bytes=np.array(png_bytes, dtype=object),
        captions=np.array(captions, dtype=object),
        labels=np.array(labels, dtype=object),
    )
    print(f"Saved {len(labels)} samples to {args.out}")
    unique, counts = np.unique(labels, return_counts=True)
    print("Label counts:", dict(zip(unique, counts)))

if __name__ == "__main__":
    main()
