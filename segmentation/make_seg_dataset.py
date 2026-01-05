import argparse
import os
import numpy as np
from tqdm import tqdm
from datasets import load_dataset
from PIL import Image
from itertools import islice

DATASET_NAME = "Angelou0516/kvasir-seg"  # or your chosen repo id

def img_to_np(img: Image.Image, size: int):
    img = img.convert("RGB").resize((size, size))
    return np.array(img, dtype=np.uint8)

def mask_to_np(mask: Image.Image, size: int):
    mask = mask.convert("L").resize((size, size))
    arr = np.array(mask, dtype=np.uint8)
    return (arr > 127).astype(np.uint8)  # 0/1

def save_split_streaming(split: str, out_path: str, size: int, N: int):
    # Streaming dataset: no random indexing, must iterate
    ds = load_dataset(DATASET_NAME, split=split, streaming=True)

    images, masks = [], []
    iterator = islice(ds, N)  # take first N samples from the stream

    for item in tqdm(iterator, total=N, desc=f"Streaming {split}"):
        # depending on dataset fields, usually "image" and "mask"
        img = item["image"]
        msk = item["mask"]

        images.append(img_to_np(img, size))
        masks.append(mask_to_np(msk, size))

    images = np.stack(images)  # (N,H,W,3)
    masks = np.stack(masks)    # (N,H,W)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez_compressed(out_path, images=images, masks=masks)
    print(f"[OK] saved {split} -> {out_path}")
    print(" images:", images.shape, "masks:", masks.shape)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", type=str, default="train", choices=["train", "validation", "test"])
    ap.add_argument("--out_dir", type=str, default="dataprocessing/data/processed")
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--N", type=int, default=1000, help="how many samples to save from the stream")
    args = ap.parse_args()

    out_path = os.path.join(args.out_dir, f"kvasir_{args.split}.npz")
    save_split_streaming(args.split, out_path, args.size, args.N)

if __name__ == "__main__":
    main()
