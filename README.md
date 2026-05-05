# Pit Analyzer

A web application for analyzing particle ejection from corrosion pits in time-lapse microscopy images.

## Overview

Pit Analyzer provides a full image-analysis pipeline:

**Crop → Preprocess → Detect → Track → Pit Analysis**

It runs locally in your browser via a Flask server, requires no cloud service, and supports both classical blob detection and deep-learning segmentation with StarDist.

---

## Features

| Step | What it does |
|------|-------------|
| **0 – Load dataset** | Point to a folder of TIFF/PNG/JPG frames; get an instant preview |
| **1 – Crop & slice** | Draw a ROI on the canvas; select a frame range; frames are saved as 8-bit TIFFs |
| **2 – Preprocess** | Morphological background flattening → motion ratio → median cleanup → contrast stretch |
| **3 – Detect** | Classical blob detector (Otsu / sigma / fixed threshold + shape filter) **or** StarDist instance segmentation |
| **4 – Track** | Hungarian optimal assignment tracker or greedy nearest-neighbour fallback |
| **5 – Pit analysis** | Assign trajectories to pit markers; compute r(t), α(t), I(t) per pit; export CSV & figures |

---

## Requirements

- Python 3.9+
- The packages listed below (all installed automatically via the virtual environment)

### Core dependencies

```
flask
numpy
scipy
Pillow
scikit-image
tifffile
opencv-python
pandas
matplotlib
```

### Optional — StarDist deep-learning detector

```
stardist
tensorflow
csbdeep
```

---

## Installation

```bash
# Clone the repository
git clone https://github.com/yasmin-elwardi/Pit-Analyzer.git
cd Pit-Analyzer

# Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# Install core dependencies
pip install flask numpy scipy pillow scikit-image tifffile opencv-python pandas matplotlib

# (Optional) Install StarDist support
pip install stardist tensorflow csbdeep
```

---

## Usage

```bash
source .venv/bin/activate
python app.py
```

Then open [http://127.0.0.1:5000](http://127.0.0.1:5000) in your browser.

---

## Pipeline walkthrough

### Step 0 — Load dataset
Enter the absolute path to a folder containing your image sequence (TIFF, PNG, JPG, or BMP).  
A preview of the last frame is displayed immediately.

### Step 1 — Crop & slice
- Draw a rectangle on the canvas to define the spatial ROI (leave empty for full frame).  
- Set frame range `f0` / `f1` to restrict the temporal window.  
- Cropped frames are saved to `outputs/<name>/`.

### Step 2 — Preprocess
Applies the morphological pipeline:
1. Gaussian background flattening (per-frame division by blurred version)
2. Frame-to-frame motion ratio (`future / current`)
3. Binary erosion to remove bright artefacts
4. Median filter to smooth noise
5. Percentile contrast stretch → 8-bit TIFFs

### Step 3 — Detect
**Classical detector**
- Threshold methods: `otsu` (recommended), `sigma`, `fixed`
- Shape filters: area range + minimum circularity

**StarDist detector** (requires optional install)
- Load a pre-trained or custom `StarDist2D` model from disk
- Outputs coloured instance-segmentation masks saved as uint16 TIFFs

### Step 4 — Track
- **Hungarian tracker** (default): globally optimal frame-to-frame assignment via `scipy.optimize.linear_sum_assignment`
- **Greedy tracker**: fast nearest-neighbour fallback
- Parameters: `max_dist_px`, `max_missed`, `min_track_length`

### Step 5 — Pit analysis
1. Place pit markers on the image canvas (click to add, drag to reposition)
2. Each particle track is assigned to its nearest pit
3. Per-pit time series are computed:
   - **r(t)** — distance from pit centre [µm]
   - **α(t)** — ejection angle [°]
   - **I(t)** — raw intensity from the original frames
4. Export results as CSV or download figures as PNG

---

## Outputs

```
outputs/
├── cropped/            # Step 1 – cropped 8-bit TIFFs
├── preprocessed/       # Step 2 – motion-ratio TIFFs
├── seg_masks/          # Step 3 – uint16 StarDist label masks
└── <export_name>/
    ├── detections/     # Per-frame detection overlays (PNG)
    ├── trajectories/   # Trajectory overlays per frame (PNG)
    ├── analysis/       # Per-pit r(t), α(t), I(t) figures (PNG)
    └── pit_analysis.csv
```

---

## Project structure

```
Pit-Analyzer/
├── app.py          # Flask server + full analysis pipeline
├── index.html      # Single-page frontend (vanilla JS)
└── README.md
```

---

## License

MIT License — see [LICENSE](LICENSE) for details.
