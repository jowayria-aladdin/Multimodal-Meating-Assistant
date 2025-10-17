"""
Landmark Dataset Verifier
-------------------------
Verifies extracted landmark.npy files for integrity and consistency.

Checks performed:
1. File readability (no corrupted .npy files)
2. Shape consistency (must be 2D: [frames, features])
3. NaN / Inf detection
4. Empty or all-zero arrays
5. Frame count statistics

Usage:
python verify_landmarks.py --input_dir "C:\\path\\to\\landmarks_validation"
python verify_landmarks.py --input_dir "C:\\path\\to\\landmarks_test"
"""

import os
import argparse
import numpy as np
from tqdm import tqdm


def verify_landmarks_folder(folder):
    files = [f for f in os.listdir(folder) if f.endswith('.npy')]
    if len(files) == 0:
        print(f"❌ No .npy files found in {folder}")
        return

    print(f"🔍 Checking {len(files)} files in {folder}\n")

    results = {
        "total": len(files),
        "loaded": 0,
        "empty": 0,
        "shape_mismatch": 0,
        "nan_inf": 0,
        "all_zero": 0,
    }

    feature_dim = None
    frame_counts = []

    for f in tqdm(files, desc="Progress", unit="file"):
        path = os.path.join(folder, f)
        try:
            arr = np.load(path)

            if arr.size == 0:
                results["empty"] += 1
                continue

            if arr.ndim != 2:
                results["shape_mismatch"] += 1
                continue

            T, D = arr.shape
            frame_counts.append(T)

            if feature_dim is None:
                feature_dim = D
            elif D != feature_dim:
                results["shape_mismatch"] += 1

            if np.isnan(arr).any() or np.isinf(arr).any():
                results["nan_inf"] += 1

            if np.allclose(arr, 0):
                results["all_zero"] += 1

            results["loaded"] += 1

        except Exception as e:
            print(f"❌ Failed to load {f}: {e}")

    print("\n📋 Summary:")
    print(f"✅ Successfully loaded: {results['loaded']}/{results['total']}")
    print(f"⚠️ Empty files: {results['empty']}")
    print(f"⚠️ Shape mismatches: {results['shape_mismatch']}")
    print(f"⚠️ NaN/Inf values: {results['nan_inf']}")
    print(f"⚠️ All-zero files: {results['all_zero']}")

    if frame_counts:
        print(f"\n📈 Frame count stats:")
        print(f"  Min: {np.min(frame_counts)}")
        print(f"  Max: {np.max(frame_counts)}")
        print(f"  Mean: {np.mean(frame_counts):.1f}")
    else:
        print("No valid files to analyze frame counts.")


def main():
    parser = argparse.ArgumentParser(description="Verify How2Sign landmark .npy files")
    parser.add_argument("--input_dir", type=str, required=True, help="Path to folder containing .npy files")
    args = parser.parse_args()

    folder = args.input_dir
    if not os.path.exists(folder):
        print(f"❌ Input folder does not exist: {folder}")
        return

    verify_landmarks_folder(folder)


if __name__ == "__main__":
    main()
