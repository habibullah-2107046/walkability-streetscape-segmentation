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

## Background

This pipeline was built to support a broader research project conducted as part of academic coursework at the **Department of Urban & Regional Planning, Rajshahi University of Engineering & Technology (RUET)**.

**Course:** Transportation Planning Studio (Course Code: URP 4128)
**Program:** Fourth Year, Department of Urban & Regional Planning, RUET

**Submitted To (Course Instructors):**
- Muhammad Waresul Hassan Nipun, Assistant Professor, Dept. of Urban & Regional Planning, RUET
- Jahid Hasan, Assistant Professor, Dept. of Urban & Regional Planning, RUET

**Project Group (Group-01):**
- Jubaer Al Hasan (2107031)
- Md. Habibullah Masbah (2107046)
- Nafis Sadman (2107052)
- Md. Jehan Rahman (2107053)

**Title:** *Assessing Walkability Potential Using Streetscape Perception, Remote Sensing and GIS Connectivity Measures: Evidence from Rajshahi City Corporation*

**Study Objectives:**
1. To extract and quantify pedestrian-level visual streetscape attributes from Google Street View imagery using deep learning-based image segmentation.
2. To develop a composite walkability index by integrating streetscape-derived indicators with remote sensing-derived environmental characteristics and GIS-based road connectivity measures.
3. To analyze and map the spatial distribution of walkability along selected urban roads and propose targeted interventions for enhancing pedestrian accessibility, safety, and comfort.

**Study Area:** Three major roads in Rajshahi City Corporation (RCC), Bangladesh:

| Route | Segment | Approx. Length |
|---|---|---|
| Route 01 | C&B Mor → Shaheb Bazar Zero Point → Binodpur Mor | 7.04 km |
| Route 02 | C&B Mor → Ghora Chattar → Vodra Mor → Talaimari Mor | 6.30 km |
| Route 03 | Shaheb Bazar Zero Point → Railgate → Amchattar | 5.09 km |

**Scope of this repository:** This repo implements **Objective 1 only** — turning raw 360° street-level panoramas into standardized visual streetscape indices (GVI, SVI, EI, Sidewalk Visibility, Crowdedness) for each sampled study point. Objectives 2 and 3 (integrating remote sensing data, GIS connectivity analysis, composite index calculation, and spatial mapping) are separate downstream stages not covered here — this repo's final output (`indices/streetscape_indices.csv`) is the input feeding into that later work.

Walkability is difficult to assess objectively because pedestrian experience depends not just on road geometry but also on visual streetscape quality — greenery, sky openness, enclosure, sidewalk presence, and visual crowding. This pipeline addresses that specific piece: turning raw panoramic street photography into standardized, reproducible numeric indicators, using a pretrained semantic segmentation model rather than manual/subjective rating.

This repository specifically documents and shares the **code implementation** of Objective 1, developed individually as part of the group's overall project.

---

## Repository Structure

```
├── panoramas/          # 5 sample 360° GSV panoramas (equirectangular), named by FID
├── study_points/
│   └── Study_Points.xlsx   # coordinates (POINT_X, POINT_Y) and route info for all study points
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

## Panorama Dataset

Only **5 sample panoramas** are included directly in this repository (in `panoramas/`), enough to test-run the full pipeline immediately after cloning.

The **full set of panoramas** (97 images, covering all sampled study points across the three routes) is hosted separately on Google Drive:

🔗 **[Full panorama dataset — Google Drive](PASTE_YOUR_GOOGLE_DRIVE_LINK_HERE)**

To use the full dataset: download the images from the link above and place them into the `panoramas/` folder (replacing or supplementing the 5 sample images), then run the pipeline as described below.

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

**Expected runtime:** on a CPU-only machine, segmentation takes roughly 2–6 seconds per perspective image.
- With the 5 sample panoramas (30 perspective images): a few minutes.
- With the full 97-panorama dataset (582 perspective images): approximately **1–2 hours**, mostly spent in `run_inference.py`. A GPU reduces this to a few minutes.

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

This CSV is the direct input for **Objective 2** of the broader study (integrating with remote sensing indicators and GIS road connectivity to build a Composite Walkability Index), and can be joined to a GIS point layer using `point_id` (↔ `FID` in `Study_Points.xlsx`).

---

## Using Your Own Panoramas

To run this pipeline on your own Street View panoramas instead of the provided sample/dataset:

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

This project is licensed under the [MIT License](LICENSE).
