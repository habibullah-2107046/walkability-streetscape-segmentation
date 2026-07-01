# Streetscape Segmentation Pipeline for Walkability Assessment

A deep learning pipeline that converts 360° Google Street View (GSV) panoramas into pedestrian-level visual streetscape indices using semantic image segmentation.

It computes five indices per study point:

- **GVI** — Green View Index
- **SVI** — Sky View Index (Openness)
- **EI** — Enclosure Index
- **Sidewalk Visibility Index**
- **Crowdedness Index**

The pipeline is built around a pretrained **SegFormer** model (`nvidia/segformer-b5-finetuned-cityscapes-1024-1024`, via Hugging Face `transformers`), free and open-source, requiring no custom training.

---

## Repository Structure

```
├── panoramas/          # 97 sample 360° GSV panoramas (equirectangular), named by FID
├── study_points/
│   └── Study_Points.xlsx   # coordinates (POINT_X, POINT_Y) and route info for each panorama
├── scripts/
│   ├── test_model.py           # (optional) verifies the segmentation model downloads/loads correctly
│   ├── preprocess_panoramas.py # Step 1: splits each panorama into six 60° perspective views
│   ├── run_inference.py        # Step 2: runs SegFormer segmentation on every perspective view
│   ├── extract_pixels.py       # Step 3: counts per-class pixels from segmentation masks
│   └── compute_indices.py      # Step 4: aggregates views per point and computes the 5 indices
└── .gitignore
```

**Note:** folders like `preprocessed/`, `masks/`, `segmented/`, `pixel_statistics/`, and `indices/` are **not included in this repo** — they are automatically created by the scripts when you run them (see below).

---

## How `panoramas/` maps to `study_points/Study_Points.xlsx`

Each panorama filename is a plain number (e.g. `50.jpg`) corresponding directly to the `FID` column in `Study_Points.xlsx`. That row also contains:
- `Route_Name` — which study road the point belongs to
- `POINT_X`, `POINT_Y` — the longitude/latitude the panorama was captured at

So `panoramas/50.jpg` ↔ the row where `FID = 50` in `Study_Points.xlsx`.

The panoramas were downloaded using the free **[Street View Download 360](https://svd360.rive.cz/)** desktop tool, using the coordinates in `Study_Points.xlsx` as input.

---

## Prerequisites

- Windows, macOS, or Linux (tested on Windows 10/11)
- [Anaconda](https://www.anaconda.com/download) (recommended, for environment management)
- Python 3.10
- ~2 GB free disk space (for model weights + generated outputs)
- No GPU required — the pipeline runs on CPU (slower, but fully functional). If you have an NVIDIA GPU with CUDA support, inference will be significantly faster.

---

## Setup

1. **Clone this repository:**
```bash
   git clone https://github.com/<your-username>/walkability-streetscape-segmentation.git
   cd walkability-streetscape-segmentation
```

2. **Create and activate a Python environment:**
```bash
   conda create -n walkability python=3.10 -y
   conda activate walkability
```

3. **Install PyTorch.**
   - If you have an NVIDIA GPU:
```bash
     pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```
   - If you do NOT have an NVIDIA GPU (CPU-only):
```bash
     pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

4. **Install remaining dependencies:**
```bash
   pip install transformers accelerate opencv-python pillow numpy pandas matplotlib py360convert tqdm openpyxl
```

5. **(Optional) Verify the segmentation model loads correctly:**
```bash
   python scripts/test_model.py
```
   This downloads the SegFormer model (~340 MB, cached locally after the first run) and prints its 19 Cityscapes class names. This step is optional — `run_inference.py` will also download the model automatically if you skip this.

---

## Running the Pipeline

Run the four scripts in order from the repository root:

```bash
python scripts/preprocess_panoramas.py
python scripts/run_inference.py
python scripts/extract_pixels.py
python scripts/compute_indices.py
```

Each script's role, input, and output:

| Script | What it does | Input | Output |
|---|---|---|---|
| `preprocess_panoramas.py` | Splits each 360° panorama into six 60° perspective images (matching pedestrian eye-level FoV) | `panoramas/*.jpg` | `preprocessed/<id>_view0.jpg` … `_view5.jpg` |
| `run_inference.py` | Runs SegFormer semantic segmentation on every perspective image | `preprocessed/*.jpg` | `masks/<id>_mask.npy` (class-ID arrays), `segmented/<id>_seg.jpg` (colorized visualizations) |
| `extract_pixels.py` | Counts per-class pixel totals from each mask | `masks/*.npy` | `pixel_statistics/pixel_counts.csv` |
| `compute_indices.py` | Aggregates the six views per point and computes the five walkability indices | `pixel_statistics/pixel_counts.csv` | `indices/streetscape_indices.csv` |

All intermediate folders (`preprocessed/`, `masks/`, `segmented/`, `pixel_statistics/`, `indices/`) are created automatically — no manual folder setup required.

**Expected runtime:** on a CPU-only machine, segmentation takes roughly 2–6 seconds per perspective image. For 97 panoramas (582 perspective images), expect the full pipeline to take **approximately 1–2 hours**, mostly spent in `run_inference.py`. A GPU reduces this to a few minutes.

---

## Final Output

`indices/streetscape_indices.csv` contains one row per study point:

| Column | Meaning |
|---|---|
| `point_id` | Matches the panorama filename / `FID` in `Study_Points.xlsx` |
| `num_views` | Number of perspective views aggregated (should always be 6) |
| `GVI` | Green View Index — vegetation pixels ÷ total pixels |
| `SVI` | Sky View Index — sky pixels ÷ total pixels |
| `EI` | Enclosure Index — (building + vegetation pixels) ÷ (sidewalk + road + fence pixels) |
| `Sidewalk_Visibility` | Sidewalk pixels ÷ total pixels |
| `Crowdedness` | Obstacle-class pixels (poles, signs, vehicles, people, etc.) ÷ total pixels |

This CSV can be joined to a GIS point layer using `point_id` (↔ `FID` in `Study_Points.xlsx`) for further spatial analysis.

---

## Using Your Own Panoramas

To run this pipeline on your own Street View panoramas instead of the provided sample set:

1. Replace the contents of `panoramas/` with your own 360° equirectangular panorama images (`.jpg`/`.png`).
2. Ensure filenames are unique identifiers — these become each point's `point_id` in the final output.
3. Run the four pipeline scripts as described above, in order.

No code changes are required — all scripts use paths relative to the repository root.

---

## Model & Methodology Notes

- **Model:** [`nvidia/segformer-b5-finetuned-cityscapes-1024-1024`](https://huggingface.co/nvidia/segformer-b5-finetuned-cityscapes-1024-1024), pretrained on the Cityscapes dataset (19 semantic classes: road, sidewalk, building, wall, fence, pole, traffic light, traffic sign, vegetation, terrain, sky, person, rider, car, truck, bus, train, motorcycle, bicycle).
- **Panorama splitting:** each panorama is split into six 60° field-of-view perspective images at 0° pitch (eye level), to approximate pedestrian-level visual perception from a full 360° view.
- **Index formulas** follow standard streetscape visual-perception methods established in the walkability literature (e.g., greenness as vegetation pixel ratio, openness as sky pixel ratio, enclosure as vertical-to-horizontal element ratio).

---

## License

*(Add your preferred license here — e.g. MIT, or leave as "All rights reserved" for now.)*
