import io
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

data = np.load("dataset.npz", allow_pickle=True)

png_bytes = data["png_bytes"]
captions = data["captions"]
labels = data["labels"]

# Change this index to browse
idx = 1000

img = Image.open(io.BytesIO(png_bytes[idx])).convert("RGB")

plt.imshow(img)
plt.axis("off")
plt.title(f"Label: {labels[idx]}")
plt.show()

print("Caption:\n", captions[idx])
