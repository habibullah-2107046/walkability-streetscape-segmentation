# preprocess_panoramas.py
import os
import cv2
import numpy as np
import py360convert as p360
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root
INPUT_DIR  = os.path.join(BASE_DIR, "panoramas")
OUTPUT_DIR = os.path.join(BASE_DIR, "preprocessed")

FOV        = 60      # field of view, matches reference paper
OUT_SIZE   = 1024    # output image size (square)
YAWS       = [0, 60, 120, 180, 240, 300]   # six views covering 360°

def process_panorama(path, out_dir, base_id):
    pano = cv2.imread(path)
    if pano is None:
        print(f"  WARNING: could not read {path}, skipping.")
        return
    pano = cv2.cvtColor(pano, cv2.COLOR_BGR2RGB)
    for i, yaw in enumerate(YAWS):
        persp = p360.e2p(
            pano,
            fov_deg=(FOV, FOV),
            u_deg=yaw,
            v_deg=0,
            out_hw=(OUT_SIZE, OUT_SIZE)
        )
        out_path = os.path.join(out_dir, f"{base_id}_view{i}.jpg")
        cv2.imwrite(out_path, cv2.cvtColor(persp, cv2.COLOR_RGB2BGR))

if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    print(f"Found {len(files)} panoramas in {INPUT_DIR}")

    for fname in files:
        base_id = os.path.splitext(fname)[0]
        process_panorama(os.path.join(INPUT_DIR, fname), OUTPUT_DIR, base_id)
        print(f"Processed {base_id}")

    print("Done.")