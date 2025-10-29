import os
import cv2
import numpy as np
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

# === CONFIG ===
FRAMES_ROOT = r"C:\Users\Eslam-PC\OneDrive\Desktop\Sign Language\Frames\test_frames"
OUTPUT_ROOT = r"C:\Users\Eslam-PC\Downloads\preprocessed_test_frames"
N_FRAMES = 32
IMG_SIZE = 224

# === FIXED mean/std (precomputed) ===
mean = np.array([0.4576356977208257, 0.631635347978813, 0.46978979367693396], dtype=np.float32)
std  = np.array([0.13932171079485092, 0.18834881894988587, 0.16424638687165657], dtype=np.float32)
print("✅ Using fixed precomputed dataset mean/std values.\n")

# === STEP 1: Frame sampling ===
def sample_frames(frames, n):
    if len(frames) <= n:
        return frames
    idxs = np.linspace(0, len(frames) - 1, n).astype(int)
    return [frames[i] for i in idxs]

# === STEP 2: Process one video ===
def process_video(video_folder, output_folder, mean, std):
    frames = sorted([
        os.path.join(video_folder, f)
        for f in os.listdir(video_folder)
        if f.lower().endswith((".jpg", ".png"))
    ])

    if len(frames) == 0:
        return f"No frames found for {video_folder}"

    frames = sample_frames(frames, N_FRAMES)

    tensors = []
    for frame_path in frames:
        img = cv2.imread(frame_path)
        if img is None:
            continue
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img = img.astype(np.float32) / 255.0
        img = (img - mean) / std
        tensors.append(img)

    tensor = np.stack(tensors).astype(np.float16)

    os.makedirs(output_folder, exist_ok=True)
    out_name = os.path.basename(video_folder) + ".npz"
    out_path = os.path.join(output_folder, out_name)
    np.savez_compressed(out_path, frames=tensor)

    return f"Saved {out_name} → {tensor.shape}"

# === STEP 3: Main ===
def main():
    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    video_folders = [
        os.path.join(FRAMES_ROOT, d)
        for d in os.listdir(FRAMES_ROOT)
        if os.path.isdir(os.path.join(FRAMES_ROOT, d))
    ]

    print(f"Found {len(video_folders)} videos to preprocess\n")

    with ProcessPoolExecutor(max_workers=4) as executor:
        futures = {
            executor.submit(process_video, folder, OUTPUT_ROOT, mean, std): folder
            for folder in video_folders
        }
        for fut in tqdm(as_completed(futures), total=len(video_folders), desc="Preprocessing", unit="video"):
            try:
                msg = fut.result()
                print(msg)
            except Exception as e:
                print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
