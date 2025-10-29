import numpy as np
import matplotlib.pyplot as plt
import os
import random

# path to your preprocessed npz files
DATA_DIR = r"C:\Users\Eslam-PC\Downloads\preprocessed_test_frames"

# randomly pick one .npz file
npz_files = [f for f in os.listdir(DATA_DIR) if f.endswith('.npz')]
chosen = random.choice(npz_files)
path = os.path.join(DATA_DIR, chosen)

# load it
data = np.load(path)
frames = data['frames']  # shape (num_frames, 224, 224, 3)

print(f"Loaded {chosen}")
print("dtype:", frames.dtype)
print("min:", np.min(frames))
print("max:", np.max(frames))
print("mean:", np.mean(frames))
print("shape:", frames.shape)

# convert to float32 for matplotlib
frames_vis = frames.astype(np.float32)

# rescale just for visualization (not for model)
frames_vis = (frames_vis - frames_vis.min()) / (frames_vis.max() - frames_vis.min())

# show a few sample frames
plt.figure(figsize=(12, 6))
for i in range(8):
    idx = min(i * (len(frames_vis) // 8), len(frames_vis) - 1)
    plt.subplot(2, 4, i + 1)
    plt.imshow(frames_vis[idx])
    plt.axis("off")
    plt.title(f"Frame {idx}")
plt.suptitle(f"Raw normalized visualization of: {chosen}")
plt.tight_layout()
plt.show()
