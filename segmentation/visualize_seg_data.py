import numpy as np
import matplotlib.pyplot as plt

data = np.load("dataprocessing/data/processed/kvasir_train.npz")
images = data["images"]
masks = data["masks"]

idx = 0  # change this to see different samples

img = images[idx]
mask = masks[idx]

plt.figure(figsize=(10, 4))

plt.subplot(1, 3, 1)
plt.title("Original Image")
plt.imshow(img)
plt.axis("off")

plt.subplot(1, 3, 2)
plt.title("Segmentation Mask")
plt.imshow(mask, cmap="gray")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.title("Overlay")
plt.imshow(img)
plt.imshow(mask, alpha=0.4, cmap="Reds")
plt.axis("off")

plt.show()

