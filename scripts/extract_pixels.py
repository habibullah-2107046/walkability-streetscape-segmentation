# extract_pixels.py
import os, glob
import numpy as np
import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MASK_DIR = os.path.join(BASE_DIR, "masks")
OUT_CSV  = os.path.join(BASE_DIR, "pixel_statistics", "pixel_counts.csv")

CLASS_NAMES = {
    0:"road",1:"sidewalk",2:"building",3:"wall",4:"fence",5:"pole",
    6:"traffic_light",7:"traffic_sign",8:"vegetation",9:"terrain",
    10:"sky",11:"person",12:"rider",13:"car",14:"truck",15:"bus",
    16:"train",17:"motorcycle",18:"bicycle"
}

rows = []
mask_files = glob.glob(os.path.join(MASK_DIR, "*_mask.npy"))
print(f"Found {len(mask_files)} mask files.")

for fpath in mask_files:
    base_id = os.path.basename(fpath).replace("_mask.npy", "")
    mask = np.load(fpath)
    total = mask.size
    counts = np.bincount(mask.flatten(), minlength=19)
    row = {"image_id": base_id, "total_pixels": total}
    for cid, cname in CLASS_NAMES.items():
        row[cname] = int(counts[cid])
    rows.append(row)

df = pd.DataFrame(rows)
os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
df.to_csv(OUT_CSV, index=False)
print(f"Saved {len(df)} rows to {OUT_CSV}")
print(df.head())