import argparse
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer

from src.features import load_npz, ClipFeaturizer

def print_results(name, y_true, y_pred, le: LabelEncoder):
    print(f"\n=== {name} ===")
    print(classification_report(y_true, y_pred, target_names=list(le.classes_)))
    print("Confusion matrix:")
    print(confusion_matrix(y_true, y_pred))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default="dataset.npz")
    ap.add_argument("--test_size", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    images, captions, labels = load_npz(args.data)

    # Encode labels
    le = LabelEncoder()
    y = le.fit_transform(labels)

    # Split
    idx = np.arange(len(y))
    train_idx, test_idx = train_test_split(idx, test_size=args.test_size, random_state=args.seed, stratify=y)

    imgs_train = [images[i] for i in train_idx]
    imgs_test = [images[i] for i in test_idx]
    caps_train = [captions[i] for i in train_idx]
    caps_test = [captions[i] for i in test_idx]
    y_train = y[train_idx]
    y_test = y[test_idx]

    # Features
    clip = ClipFeaturizer()

    Ximg_train = clip.image_features(imgs_train)
    Ximg_test = clip.image_features(imgs_test)

    Xtxtclip_train = clip.text_features(caps_train)
    Xtxtclip_test = clip.text_features(caps_test)

    # 1) Image-only (CLIP image emb)
    clf_img = LogisticRegression(max_iter=2000)
    clf_img.fit(Ximg_train, y_train)
    pred_img = clf_img.predict(Ximg_test)
    print_results("Image-only (CLIP image embeddings)", y_test, pred_img, le)

    # 2) Text-only (TF-IDF on captions)
    tfidf_clf = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
        ("clf", LogisticRegression(max_iter=2000)),
    ])
    tfidf_clf.fit(caps_train, y_train)
    pred_txt = tfidf_clf.predict(caps_test)
    print_results("Text-only (TF-IDF captions)", y_test, pred_txt, le)

    # 3) Multimodal (concat image emb + text emb)
    Xmm_train = np.concatenate([Ximg_train, Xtxtclip_train], axis=1)
    Xmm_test = np.concatenate([Ximg_test, Xtxtclip_test], axis=1)

    clf_mm = LogisticRegression(max_iter=2000)
    clf_mm.fit(Xmm_train, y_train)
    pred_mm = clf_mm.predict(Xmm_test)
    print_results("Multimodal (CLIP image + CLIP text embeddings)", y_test, pred_mm, le)

    print("\nLabels:", list(le.classes_))
    print("Tip: if 'other' dominates, increase N or tweak keyword rules in make_dataset.py.")

if __name__ == "__main__":
    main()
