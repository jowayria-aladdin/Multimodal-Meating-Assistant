import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# === CONFIG ===
NPZ_PATH = r"C:\Users\Eslam-PC\Downloads\preprocessed_test_frames\_G0RrDVpOZ4_4-5-rgb_front.npz"
STATS_PATH = r"C:\Users\Eslam-PC\OneDrive\Desktop\dataset_mean_std.json"  # for de-normalization (optional)

# === LOAD DATA ===
data = np.load(NPZ_PATH)
frames = data["frames"]  # (N, H, W, C)
print(f"Loaded {NPZ_PATH}")
print(f"Shape: {frames.shape}, dtype: {frames.dtype}")

# === OPTIONAL: load mean/std to reverse normalization ===
mean, std = None, None
if os.path.exists(STATS_PATH):
    import json
    with open(STATS_PATH, "r") as f:
        stats = json.load(f)
        mean = np.array(stats["mean"], dtype=np.float32)
        std = np.array(stats["std"], dtype=np.float32)
    print("Loaded dataset mean/std for de-normalization.")
else:
    print("No mean/std file found — showing normalized frames directly.")

# === DE-NORMALIZE ===
if mean is not None and std is not None:
    frames = (frames * std) + mean  # reverse normalization
frames = np.clip(frames, 0, 1)  # clip to valid range

# === DISPLAY FRAMES ===
def show_video_grid(frames, ncols=8):
    n = len(frames)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, nrows * 2))
    for i, ax in enumerate(axes.flat):
        if i < n:
            img = frames[i][:, :, ::-1]  # BGR → RGB
            ax.imshow(img)
            ax.axis("off")
            ax.set_title(f"Frame {i+1}")
        else:
            ax.axis("off")
    plt.tight_layout()
    plt.show()

show_video_grid(frames)
