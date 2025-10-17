"""
How2Sign landmark extractor (local, optimized for VS Code)
- Uses MediaPipe Hands + Pose (face landmarks skipped)
- Downsamples video to target_fps to save time and data
- Multiprocessing with safe default workers
- Saves per-video .npy files containing (T, D) landmark arrays
- Checkpointing/resume: skips videos that already have output .npy

Usage (example):
python landmark_extraction.py --input_dir "C:\\Users\\Eslam-PC\\Downloads\\raw_videos_validation" --output_dir "C:\\Users\\Eslam-PC\\Downloads\\landmarks_validation" --target_fps 10 --workers 4

Notes:
- Each frame vector contains normalized landmarks: pose (33*3), left_hand (21*3), right_hand (21*3).
- If a hand is missing, its 21*3 block is zeros.
- Normalization: x/image_width, y/image_height, z/(image_width+image_height)/2 (approx)
- Tune target_fps for speed/accuracy tradeoff (8-12 recommended for sign language)
"""

import os
import sys
import argparse
import numpy as np
import cv2
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import time
import logging
from tqdm import tqdm

def get_default_workers():
    """Leave a few cores free to avoid overheating."""
    try:
        cpu = multiprocessing.cpu_count()
        return max(1, min(4, cpu - 4))
    except Exception:
        return 2


def list_videos(input_dir, exts={".mp4", ".avi", ".mov", ".mkv"}):
    videos = []
    for root, _, files in os.walk(input_dir):
        for f in files:
            if os.path.splitext(f)[1].lower() in exts:
                videos.append(os.path.join(root, f))
    return sorted(videos)


def process_video(args):
    """Extract landmarks from a single video and save as .npy."""
    import mediapipe as mp

    video_path, out_path, target_fps, resize_longer_side = args
    mp_hands = mp.solutions.hands
    mp_pose = mp.solutions.pose

    if os.path.exists(out_path):
        return (video_path, "skipped")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return (video_path, "error_open")

    orig_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    interval = max(1, int(round(orig_fps / float(target_fps))))
    frames_feats = []

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as hands, mp_pose.Pose(
        static_image_mode=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as pose:
        idx = 0
        while True:
            success, frame = cap.read()
            if not success:
                break
            if idx % interval != 0:
                idx += 1
                continue

            if resize_longer_side is not None:
                h, w = frame.shape[:2]
                longer = max(h, w)
                if longer > resize_longer_side:
                    scale = resize_longer_side / float(longer)
                    frame = cv2.resize(frame, (int(w * scale), int(h * scale)))

            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w = img_rgb.shape[:2]

            pose_res = pose.process(img_rgb)
            hands_res = hands.process(img_rgb)
            feat = []

            # Pose landmarks
            if pose_res.pose_landmarks:
                for lm in pose_res.pose_landmarks.landmark:
                    feat.extend([lm.x, lm.y, lm.z / ((w + h) / 2.0)])
            else:
                feat.extend([0.0] * 33 * 3)

            # Hands landmarks
            left_hand, right_hand = None, None
            if hands_res.multi_handedness and hands_res.multi_hand_landmarks:
                for idx_h, hand_landmarks in enumerate(hands_res.multi_hand_landmarks):
                    label = hands_res.multi_handedness[idx_h].classification[0].label
                    if label == "Left":
                        left_hand = hand_landmarks
                    elif label == "Right":
                        right_hand = hand_landmarks
                if left_hand is None and right_hand is None and hands_res.multi_hand_landmarks:
                    xs = [np.mean([lm.x for lm in h.landmark]) for h in hands_res.multi_hand_landmarks]
                    if len(xs) == 1:
                        right_hand = hands_res.multi_hand_landmarks[0]
                    else:
                        order = np.argsort(xs)
                        left_hand = hands_res.multi_hand_landmarks[order[0]]
                        right_hand = hands_res.multi_hand_landmarks[order[1]]

            for hand in [left_hand, right_hand]:
                if hand is not None:
                    for lm in hand.landmark:
                        feat.extend([lm.x, lm.y, lm.z / ((w + h) / 2.0)])
                else:
                    feat.extend([0.0] * 21 * 3)

            frames_feats.append(feat)
            idx += 1

        cap.release()

    if len(frames_feats) == 0:
        return (video_path, "no_frames")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.save(out_path, np.asarray(frames_feats, dtype=np.float32))
    return (video_path, "done")


def main():
    parser = argparse.ArgumentParser(description="How2Sign landmark extractor (local)")
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--target_fps", type=int, default=10)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--resize_longer_side", type=int, default=960)
    parser.add_argument("--ext", type=str, default=".npy")
    args = parser.parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir
    target_fps = args.target_fps
    resize_longer_side = args.resize_longer_side if args.resize_longer_side > 0 else None
    workers = args.workers if args.workers is not None else get_default_workers()

    videos = list_videos(input_dir)
    if len(videos) == 0:
        logging.error("No videos found in %s", input_dir)
        sys.exit(1)

    tasks = []
    for v in videos:
        rel = os.path.relpath(v, input_dir)
        out_file = os.path.splitext(rel)[0] + args.ext
        out_path = os.path.join(output_dir, out_file)
        if not os.path.exists(os.path.dirname(out_path)):
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
        if os.path.exists(out_path):
            continue
        tasks.append((v, out_path, target_fps, resize_longer_side))

    total = len(tasks)
    logging.info("Found %d videos. Target FPS=%d, workers=%d", total, target_fps, workers)

    start = time.time()
    results = {"done": 0, "skipped": 0, "error_open": 0, "no_frames": 0, "error_save": 0}

    with ProcessPoolExecutor(max_workers=workers) as exe:
        futures = {exe.submit(process_video, t): t[0] for t in tasks}
        for fut in tqdm(as_completed(futures), total=total, desc="Overall progress", unit="video"):
            video = futures[fut]
            try:
                vp, status = fut.result()
                if status in results:
                    results[status] += 1
            except Exception as e:
                logging.error("Worker crashed for %s: %s", video, str(e))

    elapsed = time.time() - start
    logging.info("Finished in %.1f sec. Summary: %s", elapsed, results)


if __name__ == "__main__":
    main()
