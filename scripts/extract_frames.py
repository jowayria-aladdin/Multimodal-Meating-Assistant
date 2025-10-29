import os
import cv2
import pandas as pd
from tqdm import tqdm

# === CONFIG ===
CSV_PATH = r"C:\Users\Eslam-PC\OneDrive\Desktop\Graduation Project\captions\how2sign_val.csv"  # change to training csv path
VIDEOS_DIR = r"C:\Users\Eslam-PC\OneDrive\Desktop\Sign Language\Videos\validation_videos" # change to training videos path
OUTPUT_DIR = r"C:\Users\Eslam-PC\OneDrive\Desktop\Sign Language\Videos\validation_frames" 

os.makedirs(OUTPUT_DIR, exist_ok=True)

# === LOAD METADATA ===
df = pd.read_csv(CSV_PATH, sep='\t')
print("Loaded", len(df), "sentence videos from CSV")

# === EXTRACT FRAMES ===
for _, row in tqdm(df.iterrows(), total=len(df)):
    sentence_name = row['SENTENCE_NAME']   # e.g., "-fZc293MpJk_0-1-rgb_front"
    video_file = f"{sentence_name}.mp4"
    video_path = os.path.join(VIDEOS_DIR, video_file)

    out_dir = os.path.join(OUTPUT_DIR, sentence_name)
    os.makedirs(out_dir, exist_ok=True)

    if not os.path.exists(video_path):
        print(f"Missing video: {video_path}")
        continue

    cap = cv2.VideoCapture(video_path)
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (224, 224))
        frame_path = os.path.join(out_dir, f"frame_{frame_count:04d}.jpg")
        cv2.imwrite(frame_path, frame)
        frame_count += 1

    cap.release()

print("Frame extraction complete.")
