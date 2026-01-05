import io
import numpy as np
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel
from sklearn.feature_extraction.text import TfidfVectorizer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_ID = "openai/clip-vit-base-patch32"

class ClipFeaturizer:
    def __init__(self):
        self.model = CLIPModel.from_pretrained(MODEL_ID).to(DEVICE).eval()
        self.processor = CLIPProcessor.from_pretrained(MODEL_ID)

    @torch.no_grad()
    def image_features(self, pil_images):
        inputs = self.processor(images=pil_images, return_tensors="pt").to(DEVICE)
        feats = self.model.get_image_features(**inputs)  # (B, D)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats.detach().cpu().numpy()

    @torch.no_grad()
    def text_features(self, texts):
        inputs = self.processor(text=texts, return_tensors="pt", padding=True, truncation=True).to(DEVICE)
        feats = self.model.get_text_features(**inputs)  # (B, D)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats.detach().cpu().numpy()

def load_npz(npz_path: str):
    data = np.load(npz_path, allow_pickle=True)
    png_bytes = data["png_bytes"]
    captions = data["captions"].tolist()
    labels = data["labels"].tolist()
    images = [Image.open(io.BytesIO(b)).convert("RGB") for b in png_bytes]
    return images, captions, labels

def make_tfidf(captions):
    vec = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X = vec.fit_transform(captions)
    return X, vec
