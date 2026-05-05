"""
Pit Analyzer Web Application
============================
Full pipeline: crop → preprocess (morphological) → detect → track → pit analysis

Run:  pip install flask numpy scipy pillow scikit-image tifffile opencv-python tqdm
      python app.py
"""

import os, sys, json, pickle, threading, traceback, io, base64, time
from pathlib import Path
from flask import Flask, render_template, request, jsonify, send_file, send_from_directory

import numpy as np
from PIL import Image

# ── optional heavy imports ────────────────────────────────────────────────────
try:
    from scipy.ndimage import gaussian_filter, binary_erosion, median_filter
    SCIPY_OK = True
except ImportError:
    SCIPY_OK = False

try:
    import tifffile
    TIFFFILE_OK = True
except ImportError:
    TIFFFILE_OK = False

try:
    import cv2
    CV2_OK = True
except ImportError:
    CV2_OK = False

try:
    from skimage import measure, morphology as sk_morphology
    SKIMAGE_OK = True
except ImportError:
    SKIMAGE_OK = False

try:
    from stardist.models import StarDist2D
    from csbdeep.utils import normalize as csbdeep_normalize
    STARDIST_OK = True
except ImportError:
    STARDIST_OK = False

try:
    from scipy.optimize import linear_sum_assignment
    HUNGARIAN_OK = True
except ImportError:
    HUNGARIAN_OK = False

# ── Flask app ─────────────────────────────────────────────────────────────────
app = Flask(__name__, template_folder=".")
app.config['MAX_CONTENT_LENGTH'] = 2 * 1024 * 1024 * 1024  # 2 GB

BASE_DIR    = Path(__file__).parent
UPLOAD_DIR  = BASE_DIR / "uploads"
OUTPUT_DIR  = BASE_DIR / "outputs"
for d in [UPLOAD_DIR, OUTPUT_DIR]: d.mkdir(exist_ok=True)

# ── global session state (single-user demo) ───────────────────────────────────
STATE = {
    "raw_folder":    None,   # Path to raw TIFF folder (original or after crop)
    "crop_folder":   None,   # Path to cropped frames
    "prep_folder":   None,   # Path to morphologically preprocessed frames
    "detections":    None,   # list of detection dicts
    "df_tracks":     None,   # pandas DataFrame
    "all_tracks":    None,
    "pit_list":      [],
    "px_um":         0.5,
    "dt_s":          10.0,
    "last_frame_b64": None,
    "progress":       {"stage": "idle", "pct": 0, "msg": ""},
    "stats":          {},
    # StarDist model
    "stardist_model":      None,   # loaded StarDist2D instance
    "stardist_model_dir":  None,
    "stardist_model_name": None,
    "stardist_prob_thresh": 0.4,
    "stardist_nms_thresh":  0.3,
    "seg_masks":           None,   # list of uint16 label arrays (one per frame)
    "seg_renders":         None,   # list of RGB uint8 render_label() arrays
    "seg_masks_folder":    None,   # path to saved uint16 mask TIFs
}

# ─────────────────────────────────────────────────────────────────────────────
# Image helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_gray(fp: Path) -> np.ndarray:
    fp = Path(fp)
    if fp.suffix.lower() in (".tif", ".tiff") and TIFFFILE_OK:
        img = tifffile.imread(str(fp))
    elif CV2_OK:
        img = cv2.imread(str(fp), cv2.IMREAD_GRAYSCALE)
    else:
        img = np.array(Image.open(fp).convert("L"))
    if img is None:
        raise RuntimeError(f"Cannot read: {fp}")
    if img.ndim == 3:
        img = img.mean(axis=-1)
    return img.astype(np.float32)


def collect_images(folder: Path):
    files = []
    for e in ["*.tif", "*.tiff", "*.png", "*.jpg", "*.bmp"]:
        files += list(Path(folder).glob(e))
    return sorted(set(files))


def arr_to_b64(arr: np.ndarray) -> str:
    """Convert numpy float/uint array to base64 PNG string."""
    if arr.dtype != np.uint8:
        lo, hi = arr.min(), arr.max()
        if hi > lo:
            arr = ((arr - lo) / (hi - lo) * 255).astype(np.uint8)
        else:
            arr = np.zeros_like(arr, dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def frame_b64(folder, idx=-1):
    files = collect_images(Path(folder))
    if not files:
        return None
    fp = files[idx]
    img = load_gray(fp)
    return arr_to_b64(img.astype(np.float32))


# ─────────────────────────────────────────────────────────────────────────────
# Morphological preprocessing (from morphologie_detection.py)
# ─────────────────────────────────────────────────────────────────────────────

def run_morphological_preprocessing(
    input_folder: Path, output_folder: Path,
    gaussian_sigma=10, motion_gap=1,
    threshold_value=1.0, morph_kernel_size=3,
    median_kernel_size=3, bright_spot_threshold=1.05,
    pct_low=1, pct_high=99,
    src_files=None,          # pre-collected list of Path objects (skips re-glob)
    progress_cb=None
):
    """Morphological background-flatten → motion-ratio → cleanup → save."""
    if not SCIPY_OK:
        raise RuntimeError("scipy not installed — run: pip install scipy")

    output_folder.mkdir(parents=True, exist_ok=True)

    # Use pre-collected file list if provided, else fall back to collect_images
    if src_files is not None:
        tif_files = list(src_files)
    else:
        tif_files = collect_images(input_folder)   # handles all extensions & cases

    if not tif_files:
        raise FileNotFoundError(
            f"No image files found in:\n  {input_folder}\n"
            "The crop step may not have completed yet."
        )

    n = len(tif_files)
    print(f"[preprocess] {n} files from {input_folder}")

    # Load stack using load_gray (handles 16-bit TIF, grayscale, etc.)
    frames = []
    for i, fp in enumerate(tif_files):
        img = load_gray(Path(fp)).astype(np.float64)
        frames.append(img)
        if progress_cb: progress_cb(int(i / n * 20), f"Loading {i+1}/{n}")

    image_stack = np.stack(frames, axis=2)   # (H, W, T)
    h, w, total = image_stack.shape
    print(f"[preprocess] stack shape: {image_stack.shape}")

    # Background flattening
    flat = np.zeros_like(image_stack)
    for i in range(total):
        frame   = image_stack[:, :, i]
        blurred = gaussian_filter(frame, sigma=gaussian_sigma)
        blurred[blurred == 0] = 1e-10
        flat[:, :, i] = frame / blurred
        if progress_cb: progress_cb(20 + int(i / total * 20), f"Background flatten {i+1}/{total}")

    # Motion ratios
    num_ratios = total - motion_gap
    if num_ratios <= 0:
        raise ValueError(
            f"Not enough frames for motion_gap={motion_gap}. "
            f"Got {total} frames, need at least {motion_gap + 1}."
        )
    ratio_stack = np.zeros((h, w, num_ratios))
    for i in range(num_ratios):
        cur = flat[:, :, i].copy()
        fut = flat[:, :, i + motion_gap]
        cur[cur == 0] = 1e-10
        ratio_stack[:, :, i] = fut / cur
        if progress_cb: progress_cb(40 + int(i / num_ratios * 20), f"Motion ratio {i+1}/{num_ratios}")

    # Morphological cleanup
    kernel    = np.ones((morph_kernel_size, morph_kernel_size), dtype=bool)
    processed = ratio_stack.copy()
    for i in range(num_ratios):
        frame   = processed[:, :, i]
        bright  = frame > threshold_value * bright_spot_threshold
        eroded  = binary_erosion(bright, kernel)
        removed = bright & ~eroded
        frame[removed] = threshold_value
        frame           = median_filter(frame, size=median_kernel_size)
        processed[:, :, i] = frame
        if progress_cb: progress_cb(60 + int(i / num_ratios * 20), f"Morphology {i+1}/{num_ratios}")

    # Contrast stretch and save
    vmin = np.percentile(processed, pct_low)
    vmax = np.percentile(processed, pct_high)
    saved_files = []
    for i in range(num_ratios):
        frame   = processed[:, :, i]
        clipped = np.clip(frame, vmin, vmax)
        scaled  = ((clipped - vmin) / (vmax - vmin) * 255).astype(np.uint8)
        out_fp  = output_folder / f"particle_{i:04d}.tif"
        Image.fromarray(scaled).save(out_fp)
        saved_files.append(out_fp)
        if progress_cb: progress_cb(80 + int(i / num_ratios * 19), f"Saving {i+1}/{num_ratios}")

    if progress_cb: progress_cb(100, f"Done — {len(saved_files)} frames saved")
    print(f"[preprocess] saved {len(saved_files)} files to {output_folder}")
    return saved_files


# ─────────────────────────────────────────────────────────────────────────────
# Simple blob detector (replaces the notebook's detection step)
# ─────────────────────────────────────────────────────────────────────────────

def detect_particles_simple(prep_folder: Path, threshold_factor=0.7, min_area=3, max_area=500,
                              threshold_method="otsu", min_circularity=0.3,
                              progress_cb=None):
    """
    Blob detector: threshold → connected components → shape/size filter.

    threshold_method:
      "otsu"  — Otsu automatic threshold (best for bimodal ratio images).
                threshold_factor is a multiplier applied to the Otsu value (0.8–1.2).
      "sigma" — threshold = mean + threshold_factor * std  (good when SNR is known).
      "fixed" — threshold on per-frame min-max normalised [0,1] image (original behaviour,
                most prone to false positives on low-contrast frames).

    min_circularity: 4π·area/perimeter² ∈ (0,1].  Rejects streaks/noise blobs.
                     Typical round particles: >0.5.  Set 0 to disable.
    """
    files = collect_images(prep_folder)
    if not files:
        raise FileNotFoundError(f"No images in {prep_folder}")

    detections = []
    n = len(files)
    for fi, fp in enumerate(files):
        img = load_gray(fp)

        # ── threshold ────────────────────────────────────────────────────────
        if threshold_method == "otsu" and SKIMAGE_OK:
            from skimage.filters import threshold_otsu
            try:
                t = threshold_otsu(img) * threshold_factor
            except Exception:
                t = img.mean() + threshold_factor * img.std()
            mask = img > t

        elif threshold_method == "sigma":
            t = img.mean() + threshold_factor * img.std()
            mask = img > t

        else:  # "fixed" — original normalised approach
            lo, hi = img.min(), img.max()
            norm = (img - lo) / (hi - lo) if hi > lo else np.zeros_like(img)
            mask = norm > threshold_factor

        # ── connected components ─────────────────────────────────────────────
        cents, probs = [], []
        if SKIMAGE_OK:
            labeled = measure.label(mask)
            props   = measure.regionprops(labeled, intensity_image=img)
            for p in props:
                if not (min_area <= p.area <= max_area):
                    continue
                # circularity filter (skip if perimeter is 0)
                if min_circularity > 0 and p.perimeter > 0:
                    circ = 4 * np.pi * p.area / (p.perimeter ** 2)
                    if circ < min_circularity:
                        continue
                cents.append([p.centroid[0], p.centroid[1]])
                # probability = normalised mean intensity of the blob in the original image
                global_max = img.max()
                probs.append(float(p.mean_intensity / global_max) if global_max > 0 else 0.8)
        else:
            ys, xs = np.where(mask)
            cents = [[float(y), float(x)] for y, x in zip(ys[::5], xs[::5])]
            probs = [0.8] * len(cents)

        detections.append({"frame": fi, "centroids": np.array(cents), "probs": probs})
        if progress_cb: progress_cb(int(fi / n * 100), f"Detecting frame {fi+1}/{n}")

    if progress_cb: progress_cb(100, f"Detection done — {sum(len(d['centroids']) for d in detections)} detections")
    return detections


# ─────────────────────────────────────────────────────────────────────────────
# StarDist instance segmentation detector
# ─────────────────────────────────────────────────────────────────────────────

def load_stardist_model(model_basedir: str, model_name: str):
    """Load a saved StarDist2D model from disk."""
    if not STARDIST_OK:
        raise RuntimeError(
            "stardist / csbdeep not installed.\n"
            "Run:  pip install stardist tensorflow csbdeep"
        )
    model = StarDist2D(None, name=model_name, basedir=str(model_basedir))
    return model


def detect_particles_stardist(
    prep_folder: Path,
    model,
    prob_thresh:   float = 0.4,
    nms_thresh:    float = 0.3,
    norm_pct_lo:   float = 1.0,
    norm_pct_hi:   float = 99.8,
    masks_out_dir: Path  = None,
    progress_cb=None,
):
    """
    Exact notebook pipeline:
      load_gray → csbdeep_normalize → predict_instances → render_label → save masks

    Returns
    -------
    detections  : list of {frame, centroids (N,2) [row,col], probs (N,)}
    seg_masks   : list of uint16 label arrays (one per frame, in RAM)
    seg_renders : list of RGB uint8 arrays   (render_label output, for the viewer)
    """
    if not STARDIST_OK:
        raise RuntimeError("stardist not installed — run: pip install stardist tensorflow csbdeep")
    if not SKIMAGE_OK:
        raise RuntimeError("scikit-image not installed — run: pip install scikit-image")

    from stardist.plot import render_label   # same import as notebook

    files = collect_images(prep_folder)
    if not files:
        raise FileNotFoundError(
            f"No images found in preprocessed folder:\n  {prep_folder}\n"
            "Make sure Step 2 (Preprocess) completed and saved files there."
        )

    if masks_out_dir is not None:
        masks_out_dir.mkdir(parents=True, exist_ok=True)

    detections, seg_masks, seg_renders = [], [], []
    n = len(files)
    print(f"[stardist] running on {n} files from {prep_folder}")

    for fi, fp in enumerate(files):
        # 1. Load & normalise (identical to notebook)
        raw      = load_gray(fp).astype(np.float32)
        img_norm = csbdeep_normalize(raw, norm_pct_lo, norm_pct_hi, axis=(0, 1))

        # 2. StarDist predict_instances (identical to notebook)
        labels, details = model.predict_instances(
            img_norm, prob_thresh=prob_thresh, nms_thresh=nms_thresh
        )
        # labels  : (H,W) uint16  0=background, 1..N=instances
        # details : {'coord':(N,rays,2), 'points':(N,2), 'prob':(N,)}

        # 3. render_label — produces the proper coloured mask overlay (identical to notebook)
        rendered_rgba = render_label(labels, img=img_norm)       # (H,W,4) float32 RGBA
        rendered_rgb  = (np.clip(rendered_rgba[..., :3], 0, 1) * 255).astype(np.uint8)
        seg_renders.append(rendered_rgb)

        # 4. Save uint16 label mask to disk
        if masks_out_dir is not None:
            mask_fp = masks_out_dir / f"mask_{fi:04d}.tif"
            if TIFFFILE_OK:
                tifffile.imwrite(str(mask_fp), labels.astype(np.uint16))
            else:
                Image.fromarray(np.clip(labels, 0, 255).astype(np.uint8)).save(
                    mask_fp.with_suffix(".png"))

        # 5. Centroids from regionprops
        props = measure.regionprops(labels)
        cents = np.array([[p.centroid[0], p.centroid[1]] for p in props]) \
                if props else np.empty((0, 2))
        probs = list(details.get("prob", np.ones(len(props), dtype=np.float32)))

        detections.append({"frame": fi, "centroids": cents, "probs": probs})
        seg_masks.append(labels)

        if progress_cb:
            progress_cb(int(fi / n * 100),
                        f"StarDist {fi+1}/{n} — {len(props)} particles")

    if progress_cb:
        total = sum(len(d["centroids"]) for d in detections)
        progress_cb(100, f"Done — {total} particles in {n} frames")

    print(f"[stardist] finished: {sum(len(d['centroids']) for d in detections)} total detections")
    return detections, seg_masks, seg_renders


# ─────────────────────────────────────────────────────────────────────────────
# Hungarian (optimal) tracker
# ─────────────────────────────────────────────────────────────────────────────

def run_tracking_hungarian(
    detections, px_um, dt_s,
    max_dist_px=25, max_missed=4, min_len=2,
    progress_cb=None,
):
    """
    Hungarian algorithm (linear_sum_assignment) tracker.

    At each frame we solve the optimal assignment between active track
    tail-positions and new detections, with a hard gating distance
    max_dist_px.  Unmatched tracks are kept alive for up to max_missed
    frames before being finalised.

    Returns
    -------
    all_tracks : list of dicts  {track_id, frames, rows, cols, probs}
    df_tracks  : pandas DataFrame (one row per detection per track)
    p50, p80   : nearest-neighbour distance percentiles (px)
    """
    import pandas as pd

    # ── internal state ────────────────────────────────────────────────────────
    # active[tid] = {"frames":[], "rows":[], "cols":[], "probs":[], "missed":int}
    active      = {}
    finished    = []
    tid_counter = [0]
    nn_dists    = []

    def _new_track(fi, row, col, prob):
        tid = tid_counter[0]; tid_counter[0] += 1
        active[tid] = {"frames": [fi], "rows": [row], "cols": [col],
                       "probs":  [prob], "missed": 0}

    n_frames = len(detections)

    for step, res in enumerate(detections):
        fi    = res["frame"]
        cents = np.array(res["centroids"])   # (N, 2)  row, col
        probs = list(res["probs"])

        if progress_cb:
            progress_cb(
                int(step / n_frames * 90) + 5,
                f"Hungarian tracking frame {step+1}/{n_frames}"
            )

        if len(active) == 0 or len(cents) == 0:
            # No active tracks or no detections → just seed / advance missed
            for k in range(len(cents)):
                _new_track(fi, float(cents[k, 0]), float(cents[k, 1]),
                           probs[k] if k < len(probs) else 0.8)
            # Increment missed for all active tracks
            for tid in list(active):
                active[tid]["missed"] += 1
                if active[tid]["missed"] > max_missed:
                    finished.append(active.pop(tid))
            continue

        act_ids = list(active.keys())
        act_pos = np.array([
            [active[t]["rows"][-1], active[t]["cols"][-1]] for t in act_ids
        ])  # (A, 2)

        # ── cost matrix (Euclidean distances) ────────────────────────────────
        diff     = act_pos[:, None, :] - cents[None, :, :]   # (A, D, 2)
        cost_mat = np.sqrt((diff ** 2).sum(-1))               # (A, D)

        # Gate: set cost above max to a large sentinel
        SENTINEL = max_dist_px * 10.0
        gated    = cost_mat.copy()
        gated[cost_mat > max_dist_px] = SENTINEL

        # ── Hungarian assignment ──────────────────────────────────────────────
        row_ind, col_ind = linear_sum_assignment(gated)

        assigned_act = set()
        assigned_det = set()

        for ai, di in zip(row_ind, col_ind):
            if gated[ai, di] >= SENTINEL:
                continue                    # gated out — treat as unmatched
            nn_dists.append(float(cost_mat[ai, di]))
            tid = act_ids[ai]
            active[tid]["frames"].append(fi)
            active[tid]["rows"].append(float(cents[di, 0]))
            active[tid]["cols"].append(float(cents[di, 1]))
            active[tid]["probs"].append(probs[di] if di < len(probs) else 0.8)
            active[tid]["missed"] = 0
            assigned_act.add(ai)
            assigned_det.add(di)

        # ── unmatched detections → new tracks ────────────────────────────────
        for di in range(len(cents)):
            if di not in assigned_det:
                _new_track(fi, float(cents[di, 0]), float(cents[di, 1]),
                           probs[di] if di < len(probs) else 0.8)

        # ── unmatched active tracks → increment missed ────────────────────────
        to_finish = []
        for ai, tid in enumerate(act_ids):
            if ai not in assigned_act:
                active[tid]["missed"] += 1
                if active[tid]["missed"] > max_missed:
                    to_finish.append(tid)
        for tid in to_finish:
            finished.append(active.pop(tid))

    # Flush remaining active tracks
    for tr in active.values():
        finished.append(tr)

    # Filter by minimum track length
    finished = [tr for tr in finished if len(tr["frames"]) >= min_len]

    # Re-index track IDs
    all_tracks = []
    for i, tr in enumerate(finished):
        tr["track_id"] = i
        all_tracks.append(tr)

    # ── Build DataFrame ───────────────────────────────────────────────────────
    records = []
    for tr in all_tracks:
        tid = tr["track_id"]
        for j in range(len(tr["frames"])):
            fi = tr["frames"][j]
            records.append({
                "track_id": tid,
                "frame":    fi,
                "time_s":   fi * dt_s,
                "row_px":   tr["rows"][j],
                "col_px":   tr["cols"][j],
                "row_um":   tr["rows"][j] * px_um,
                "col_um":   tr["cols"][j] * px_um,
                "prob":     tr["probs"][j],
            })

    df = pd.DataFrame(records) if records else pd.DataFrame(
        columns=["track_id","frame","time_s","row_px","col_px",
                 "row_um","col_um","prob"]
    )

    p50 = float(np.percentile(nn_dists, 50)) if nn_dists else 0.0
    p80 = float(np.percentile(nn_dists, 80)) if nn_dists else float(max_dist_px)

    if progress_cb:
        progress_cb(100, f"{len(all_tracks)} tracks found  |  P80 NN={p80:.1f}px")

    return all_tracks, df, p50, p80

def run_tracking_simple(detections, px_um, dt_s, max_dist_px, max_missed, min_len):
    """
    Greedy nearest-neighbour tracker. Returns (all_tracks_list, df_tracks, p50, p80).
    all_tracks_list: list of dicts {track_id, frames, rows, cols, probs}
    """
    import pandas as pd

    active = {}   # track_id -> {"frames":[], "rows":[], "cols":[], "probs":[], "missed":0}
    finished = []
    tid_counter = [0]

    def new_track(fi, row, col, prob):
        tid = tid_counter[0]; tid_counter[0] += 1
        active[tid] = {"frames":[fi], "rows":[row], "cols":[col],
                       "probs":[prob], "missed":0}

    nn_dists = []

    for res in detections:
        fi = res["frame"]
        cents = np.array(res["centroids"])  # shape (N,2) row,col
        probs = list(res["probs"])

        if len(active) == 0:
            for k in range(len(cents)):
                new_track(fi, cents[k,0], cents[k,1], probs[k] if probs else 0.8)
            continue

        # Build cost matrix between active track last positions and new detections
        act_ids = list(active.keys())
        act_pos = np.array([[active[t]["rows"][-1], active[t]["cols"][-1]] for t in act_ids])

        if len(cents) > 0:
            diff = act_pos[:, None, :] - cents[None, :, :]   # (A, D, 2)
            dist_mat = np.sqrt((diff**2).sum(-1))             # (A, D)

            # Greedy assignment
            assigned_det = set()
            assigned_act = set()
            for _ in range(min(len(act_ids), len(cents))):
                idx = np.argmin(dist_mat)
                ai, di = np.unravel_index(idx, dist_mat.shape)
                if dist_mat[ai, di] > max_dist_px:
                    break
                nn_dists.append(dist_mat[ai, di])
                tid = act_ids[ai]
                active[tid]["frames"].append(fi)
                active[tid]["rows"].append(cents[di, 0])
                active[tid]["cols"].append(cents[di, 1])
                active[tid]["probs"].append(probs[di] if di < len(probs) else 0.8)
                active[tid]["missed"] = 0
                assigned_det.add(di)
                assigned_act.add(ai)
                dist_mat[ai, :] = 1e9
                dist_mat[:, di] = 1e9

            # Unmatched detections → new tracks
            for di in range(len(cents)):
                if di not in assigned_det:
                    new_track(fi, cents[di, 0], cents[di, 1], probs[di] if di < len(probs) else 0.8)
        else:
            assigned_act = set()

        # Unmatched active → increment missed; finish if exceeded
        to_finish = []
        for ai, tid in enumerate(act_ids):
            if ai not in assigned_act:
                active[tid]["missed"] += 1
                if active[tid]["missed"] > max_missed:
                    to_finish.append(tid)
        for tid in to_finish:
            finished.append(active.pop(tid))

    # Flush remaining
    for tid, tr in active.items():
        finished.append(tr)

    # Filter by min length
    finished = [tr for tr in finished if len(tr["frames"]) >= min_len]

    # Re-index
    all_tracks = []
    for i, tr in enumerate(finished):
        tr["track_id"] = i
        all_tracks.append(tr)

    # Build DataFrame
    records = []
    for tr in all_tracks:
        tid = tr["track_id"]
        for j in range(len(tr["frames"])):
            fi = tr["frames"][j]
            records.append({
                "track_id": tid,
                "frame":    fi,
                "time_s":   fi * dt_s,
                "row_px":   tr["rows"][j],
                "col_px":   tr["cols"][j],
                "row_um":   tr["rows"][j] * px_um,
                "col_um":   tr["cols"][j] * px_um,
                "prob":     tr["probs"][j],
            })

    import pandas as pd
    df = pd.DataFrame(records) if records else pd.DataFrame(
        columns=["track_id","frame","time_s","row_px","col_px","row_um","col_um","prob"])

    p50 = float(np.percentile(nn_dists, 50)) if nn_dists else 0.0
    p80 = float(np.percentile(nn_dists, 80)) if nn_dists else max_dist_px

    return all_tracks, df, p50, p80


# ─────────────────────────────────────────────────────────────────────────────
# Pit analysis
# ─────────────────────────────────────────────────────────────────────────────

def assign_particles_to_pits(df_tracks, pit_list, px_um, raw_folder,
                               max_assign_um=300.0, win=7):
    import pandas as pd
    df_t = df_tracks.copy()
    pit_cols = np.array([p["col_um"] for p in pit_list])
    pit_rows = np.array([p["row_um"] for p in pit_list])

    dx = df_t["col_um"].values[:, None] - pit_cols[None, :]
    dy = df_t["row_um"].values[:, None] - pit_rows[None, :]
    dist_mat = np.sqrt(dx**2 + dy**2)
    ni = dist_mat.argmin(axis=1)

    df_t["pit_id"]    = [pit_list[i]["id"] for i in ni]
    df_t["r_um"]      = dist_mat.min(axis=1)
    df_t["theta_deg"] = np.degrees(np.arctan2(
        df_t["row_um"].values - pit_rows[ni],
        df_t["col_um"].values - pit_cols[ni]))
    df_t = df_t[df_t["r_um"] <= max_assign_um].copy()

    # Intensity
    orig_files = collect_images(Path(raw_folder)) if raw_folder else []
    frame_cache = {}
    for fi in sorted(df_t["frame"].unique()):
        idx = int(fi)
        if idx < len(orig_files):
            try: frame_cache[idx] = load_gray(orig_files[idx])
            except: pass

    intensities = []
    for _, row in df_t.iterrows():
        fi = int(row["frame"]); cx = int(round(row["col_px"])); cy = int(round(row["row_px"]))
        if fi not in frame_cache:
            intensities.append(float("nan")); continue
        img = frame_cache[fi]; h_i, w_i = img.shape
        patch = img[max(0,cy-win):min(h_i,cy+win+1), max(0,cx-win):min(w_i,cx+win+1)]
        intensities.append(float(patch.mean()) if patch.size > 0 else float("nan"))
    df_t["intensity_raw"] = intensities
    return df_t


# ─────────────────────────────────────────────────────────────────────────────
# Plot generation → base64 PNG
# ─────────────────────────────────────────────────────────────────────────────

def make_pit_plots(df_t, pit, last_frame_arr, dt_s):
    """
    Returns a dict with three separate base64 PNG images:
      { "r_b64": ..., "a_b64": ..., "i_b64": ... }
    one for distance r(t), one for angle α(t), one for intensity I(t).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pid    = pit["id"]
    df_pit = df_t[df_t["pit_id"] == pid].copy()
    tids   = df_pit["track_id"].unique()
    cmap   = plt.get_cmap("tab20")
    def col(tid): return cmap((int(tid) % 20) / 20)

    def gap_plot(ax, x, y, color, lw=1.5, alpha=0.85):
        x = np.array(x, dtype=float); y = np.array(y, dtype=float)
        if len(x) < 2: ax.plot(x, y, color=color, lw=lw, alpha=alpha); return
        gaps = np.where(np.diff(x) > dt_s * 1.5)[0] + 1
        ax.plot(np.insert(x, gaps, np.nan), np.insert(y, gaps, np.nan),
                color=color, lw=lw, alpha=alpha)

    def _style_ax(ax, title, xlabel, ylabel):
        ax.set_facecolor("#1a1a1a")
        for s in ax.spines.values(): s.set_color("#333")
        ax.tick_params(colors="#aaa")
        ax.xaxis.label.set_color("#aaa"); ax.yaxis.label.set_color("#aaa")
        ax.set_title(title, color="#e8e8e8", fontsize=11, pad=8)
        ax.set_xlabel(xlabel, fontsize=9); ax.set_ylabel(ylabel, fontsize=9)
        ax.grid(True, ls=":", alpha=0.3, color="#444")

    def _fig_to_b64(fig):
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=110, bbox_inches="tight", facecolor="#0f0f0f")
        plt.close(fig)
        return base64.b64encode(buf.getvalue()).decode()

    MP = 2

    # ── Distance r(t) ────────────────────────────────────────────────────────
    fig_r, ax_r = plt.subplots(figsize=(6, 4), facecolor="#0f0f0f")
    fig_r.subplots_adjust(left=0.12, right=0.96, top=0.88, bottom=0.14)
    for tid in tids:
        sub = df_pit[df_pit["track_id"] == tid].sort_values("time_s")
        if len(sub) < MP: continue
        c = col(tid)
        gap_plot(ax_r, sub["time_s"], sub["r_um"], c)
        ax_r.scatter(sub["time_s"], sub["r_um"], color=c, s=14, zorder=4, alpha=0.7)
    ax_r.set_ylim(bottom=0)
    _style_ax(ax_r, f"{pid} — r(t) distance", "Time [s]", "Distance r [µm]")
    r_b64 = _fig_to_b64(fig_r)

    # ── Angle α(t) ───────────────────────────────────────────────────────────
    fig_a, ax_a = plt.subplots(figsize=(6, 4), facecolor="#0f0f0f")
    fig_a.subplots_adjust(left=0.12, right=0.96, top=0.88, bottom=0.14)
    for tid in tids:
        sub = df_pit[df_pit["track_id"] == tid].sort_values("time_s")
        if len(sub) < MP: continue
        c = col(tid)
        gap_plot(ax_a, sub["time_s"], sub["theta_deg"], c)
        ax_a.scatter(sub["time_s"], sub["theta_deg"], color=c, s=14, zorder=4, alpha=0.7)
    ax_a.set_ylim(-185, 185); ax_a.set_yticks([-180, -90, 0, 90, 180])
    _style_ax(ax_a, f"{pid} — α(t) angle", "Time [s]", "Ejection angle α [°]")
    a_b64 = _fig_to_b64(fig_a)

    # ── Intensity I(t) ───────────────────────────────────────────────────────
    fig_i, ax_i = plt.subplots(figsize=(6, 4), facecolor="#0f0f0f")
    fig_i.subplots_adjust(left=0.12, right=0.96, top=0.88, bottom=0.14)
    any_i = False
    for tid in tids:
        sub = df_pit[df_pit["track_id"] == tid].sort_values("time_s")
        if len(sub) < MP: continue
        c = col(tid)
        vals = sub["intensity_raw"].values
        if not np.all(np.isnan(vals)):
            gap_plot(ax_i, sub["time_s"], vals, c)
            ax_i.scatter(sub["time_s"], vals, color=c, s=14, zorder=4, alpha=0.7)
            any_i = True
    if not any_i:
        ax_i.text(0.5, 0.5, "No intensity data", transform=ax_i.transAxes,
                  ha="center", va="center", color="#666", fontsize=11)
    _style_ax(ax_i, f"{pid} — I(t) intensity", "Time [s]", "Raw intensity [0–255]")
    i_b64 = _fig_to_b64(fig_i)

    return {"r_b64": r_b64, "a_b64": a_b64, "i_b64": i_b64}


def make_overlay_plot(df_t, pit_list, last_frame_arr):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 6), facecolor="#0f0f0f")
    ax.set_facecolor("#1a1a1a")
    if last_frame_arr is not None:
        ax.imshow(last_frame_arr, cmap="gray", vmin=0, vmax=255, alpha=0.85)

    cmap = plt.get_cmap("tab20")
    def col(tid): return cmap((int(tid) % 20) / 20)

    for pit in pit_list:
        cx, cy = pit["col_px"], pit["row_px"]
        r = pit.get("radius_px", 15)
        circle = plt.Circle((cx, cy), r, fill=False, color="#ff3b3b", lw=2, zorder=10)
        ax.add_patch(circle)
        ax.plot(cx, cy, "+", color="#ff3b3b", ms=12, mew=2, zorder=11)
        ax.text(cx + r + 2, cy - 4, pit["id"], color="#ff3b3b",
                fontsize=9, fontweight="bold", zorder=12)

    if df_t is not None:
        for tid in df_t["track_id"].unique():
            sub = df_t[df_t["track_id"] == tid].sort_values("time_s")
            if len(sub) < 2: continue
            c = col(tid)
            xs = sub["col_px"].values
            ys = sub["row_px"].values
            n = len(xs)
            # Fade each segment: dim at start → bright at end (shows direction)
            for i in range(n - 1):
                alpha = 0.15 + 0.85 * ((i + 1) / max(n - 1, 1))
                ax.plot([xs[i], xs[i+1]], [ys[i], ys[i+1]], "-", color=c, lw=1.3, alpha=alpha)
            # Arrowhead at final segment
            ax.annotate("", xy=(xs[-1], ys[-1]), xytext=(xs[-2], ys[-2]),
                        arrowprops=dict(arrowstyle="-|>", color=c, lw=1.0, mutation_scale=9),
                        zorder=6)
            # Start dot (dim) and end dot (bright)
            ax.scatter(xs[0],  ys[0],  color=c, s=10, zorder=5, alpha=0.35)
            ax.scatter(xs[-1], ys[-1], color=c, s=22, zorder=6, edgecolors="white", lw=0.4)

    ax.axis("off")
    ax.set_title("Particle trajectories overlay", color="#e8e8e8", fontsize=11, pad=6)
    for s in ax.spines.values(): s.set_color("#333")

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight", facecolor="#0f0f0f")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


# ─────────────────────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/static/<path:filename>")
def static_files(filename):
    return send_from_directory(BASE_DIR / "static", filename)

@app.route("/api/status")
def api_status():
    return jsonify({
        "scipy":     SCIPY_OK,
        "tifffile":  TIFFFILE_OK,
        "cv2":       CV2_OK,
        "skimage":   SKIMAGE_OK,
        "stardist":  STARDIST_OK,
        "hungarian": HUNGARIAN_OK,
        "stardist_model_loaded": STATE.get("stardist_model") is not None,
        "stardist_model_name":   STATE.get("stardist_model_name"),
        "crop_done": STATE.get("crop_folder") is not None,
        "prep_done": STATE.get("prep_folder") is not None,
    })

@app.route("/api/progress")
def api_progress():
    return jsonify(STATE["progress"])

@app.route("/api/set_raw_folder", methods=["POST"])
def api_set_raw_folder():
    data = request.json
    folder = Path(data.get("folder", ""))
    if not folder.exists():
        return jsonify({"error": f"Folder not found: {folder}"}), 400
    files = collect_images(folder)
    if not files:
        return jsonify({"error": "No image files found in folder"}), 400
    STATE["raw_folder"] = str(folder)
    # Preview last frame
    try:
        img = load_gray(files[-1])
        h, w = img.shape
        STATE["last_frame_b64"] = arr_to_b64(img)
        STATE["stats"]["raw_frames"] = len(files)
        STATE["stats"]["frame_shape"] = [h, w]
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    return jsonify({"ok": True, "n_frames": len(files),
                    "shape": list(img.shape), "preview": STATE["last_frame_b64"]})


@app.route("/api/crop_and_save", methods=["POST"])
def api_crop_and_save():
    data = request.json
    raw_str = STATE.get("raw_folder", "")
    if not raw_str:
        return jsonify({"error": "Set a raw dataset folder first (Step 0)."}), 400

    src = Path(raw_str)
    if not src.exists():
        return jsonify({"error": f"Raw folder no longer exists:\n  {src}"}), 400

    all_files = collect_images(src)
    if not all_files:
        return jsonify({"error": f"No images found in raw folder:\n  {src}"}), 400

    x0       = int(data.get("x0", 0))
    y0       = int(data.get("y0", 0))
    x1       = int(data.get("x1", 0))   # 0 or negative → use full width
    y1       = int(data.get("y1", 0))   # 0 or negative → use full height
    f0       = int(data.get("f0", 0))
    f1       = int(data.get("f1", 0))   # 0 → all frames
    out_name = data.get("out_name", "cropped") or "cropped"

    # Swap if user drew right-to-left or bottom-to-top
    if x1 > 0 and x0 > x1: x0, x1 = x1, x0
    if y1 > 0 and y0 > y1: y0, y1 = y1, y0

    out_folder = OUTPUT_DIR / out_name
    out_folder.mkdir(parents=True, exist_ok=True)

    # slice frame list
    sel = all_files[f0: f1 if f1 > 0 else None]
    if not sel:
        return jsonify({"error":
            f"Frame range f0={f0} → f1={f1} is empty "
            f"(total frames: {len(all_files)})."}), 400

    # ── reset progress synchronously ──────────────────────────────────────────
    STATE["progress"] = {"stage": "crop", "pct": 0, "msg": f"Cropping {len(sel)} frames…"}

    def _work():
        try:
            n = len(sel)
            saved = 0
            last_shape = None
            for i, fp in enumerate(sel):
                img = load_gray(fp)
                h, w = img.shape
                _x1 = x1 if x1 > 0 else w
                _y1 = y1 if y1 > 0 else h
                cropped = img[y0:_y1, x0:_x1]
                if cropped.size == 0:
                    raise ValueError(
                        f"Crop region produces empty image at frame {i}. "
                        f"Check x0={x0},y0={y0},x1={_x1},y1={_y1} vs image {w}×{h}."
                    )
                out_fp = out_folder / f"frame_{i:04d}.tif"
                lo, hi = cropped.min(), cropped.max()
                arr8   = ((cropped - lo) / (hi - lo) * 255).astype(np.uint8) \
                         if hi > lo else np.zeros_like(cropped, dtype=np.uint8)
                Image.fromarray(arr8).save(out_fp)
                saved      += 1
                last_shape  = cropped.shape
                STATE["progress"] = {"stage": "crop", "pct": int(i / n * 100),
                                     "msg": f"Cropping {i+1}/{n}"}

            STATE["crop_folder"] = str(out_folder)
            STATE["last_frame_b64"] = arr_to_b64(load_gray(out_folder / f"frame_{saved-1:04d}.tif"))
            h2, w2 = last_shape or (0, 0)
            STATE["stats"]["crop_frames"] = saved
            STATE["stats"]["crop_shape"]  = [h2, w2]
            STATE["progress"] = {"stage": "crop", "pct": 100,
                                  "msg": f"Saved {saved} cropped frames → {out_folder.name}/"}
        except Exception as e:
            STATE["progress"] = {"stage": "crop_error", "pct": 0, "msg": str(e)}
            print(traceback.format_exc())

    threading.Thread(target=_work, daemon=True).start()
    return jsonify({"ok": True, "out_folder": str(out_folder), "n_frames": len(sel)})


@app.route("/api/preprocess", methods=["POST"])
def api_preprocess():
    data = request.json

    # ── enforce pipeline order ────────────────────────────────────────────────
    crop_str = STATE.get("crop_folder")
    if not crop_str:
        return jsonify({"error":
            "No cropped folder in session.\n"
            "Complete Step 1 (Crop & Slice) first, then run preprocessing."}), 400

    crop_path = Path(crop_str)
    if not crop_path.exists():
        return jsonify({"error":
            f"Cropped folder no longer exists on disk:\n  {crop_path}\n"
            "Re-run Step 1 (Crop & Slice)."}), 400

    # Collect all image files from the crop folder (any extension, case-insensitive)
    src_files = collect_images(crop_path)
    if not src_files:
        return jsonify({"error":
            f"Cropped folder is empty — no images found in:\n  {crop_path}\n"
            "Re-run Step 1 (Crop & Slice)."}), 400

    out_name   = data.get("out_name", "preprocessed")
    out_folder = OUTPUT_DIR / out_name

    params = {
        "gaussian_sigma":        float(data.get("gaussian_sigma",        10)),
        "motion_gap":            int(  data.get("motion_gap",             1)),
        "threshold_value":       float(data.get("threshold_value",       1.0)),
        "morph_kernel_size":     int(  data.get("morph_kernel_size",      3)),
        "median_kernel_size":    int(  data.get("median_kernel_size",     3)),
        "bright_spot_threshold": float(data.get("bright_spot_threshold", 1.05)),
        "pct_low":               int(  data.get("pct_low",                1)),
        "pct_high":              int(  data.get("pct_high",               99)),
    }

    # ── reset progress BEFORE spawning thread so JS never sees stale pct=100 ─
    STATE["progress"] = {"stage": "preprocess", "pct": 0, "msg": "Starting…"}

    def _cb(pct, msg):
        STATE["progress"] = {"stage": "preprocess", "pct": pct, "msg": msg}

    def _work():
        try:
            saved = run_morphological_preprocessing(
                crop_path, out_folder, progress_cb=_cb,
                src_files=src_files,          # pass pre-collected file list
                **params
            )
            STATE["prep_folder"] = str(out_folder)
            if saved:
                STATE["last_frame_b64"] = arr_to_b64(load_gray(saved[-1]))
            STATE["stats"]["prep_frames"] = len(saved)
            STATE["progress"] = {
                "stage": "preprocess", "pct": 100,
                "msg":   f"Saved {len(saved)} preprocessed frames → {out_folder.name}/"
            }
        except Exception as e:
            STATE["progress"] = {"stage": "preprocess_error", "pct": 0, "msg": str(e)}
            print(traceback.format_exc())

    threading.Thread(target=_work, daemon=True).start()
    return jsonify({"ok": True, "out_folder": str(out_folder),
                    "n_src_files": len(src_files)})


@app.route("/api/detect", methods=["POST"])
def api_detect():
    data = request.json
    src_str = STATE.get("prep_folder")
    if not src_str:
        return jsonify({"error": "No preprocessed folder found. Run Step 2 (Preprocess) first."}), 400

    threshold = float(data.get("threshold", 0.7))
    min_area  = int(data.get("min_area", 3))
    max_area  = int(data.get("max_area", 500))
    t_method  = data.get("threshold_method", "otsu")
    min_circ  = float(data.get("min_circularity", 0.3))

    def _cb(pct, msg):
        STATE["progress"] = {"stage":"detect","pct":pct,"msg":msg}

    def _work():
        try:
            STATE["progress"] = {"stage":"detect","pct":0,"msg":"Detecting particles…"}
            dets = detect_particles_simple(Path(src_str), threshold, min_area, max_area,
                                           threshold_method=t_method,
                                           min_circularity=min_circ,
                                           progress_cb=_cb)
            STATE["detections"] = dets
            total = sum(len(d["centroids"]) for d in dets)
            STATE["stats"]["n_detections"] = total
            STATE["stats"]["n_frames_det"] = len(dets)
            STATE["progress"] = {"stage":"detect","pct":100,
                                  "msg":f"{total} detections in {len(dets)} frames"}
        except Exception as e:
            STATE["progress"] = {"stage":"detect_error","pct":0,"msg":str(e)}
            print(traceback.format_exc())

    threading.Thread(target=_work, daemon=True).start()
    return jsonify({"ok": True})


@app.route("/api/load_detections", methods=["POST"])
def api_load_detections():
    """Load a pre-existing detections.pkl file."""
    data = request.json
    pkl_path = Path(data.get("path",""))
    if not pkl_path.exists():
        return jsonify({"error": f"File not found: {pkl_path}"}), 400
    try:
        with open(pkl_path, "rb") as f:
            dets = pickle.load(f)
        STATE["detections"] = dets
        total = sum(len(d["centroids"]) for d in dets)
        STATE["stats"]["n_detections"] = total
        STATE["stats"]["n_frames_det"] = len(dets)
        return jsonify({"ok": True, "n_detections": total, "n_frames": len(dets)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/load_stardist_model", methods=["POST"])
def api_load_stardist_model():
    """
    Load a trained StarDist2D model from disk.
    Body: { "model_basedir": "/path/to/models", "model_name": "stardist_corrosion_particles" }
    The model folder structure expected by StarDist:
        <model_basedir>/<model_name>/config.json
        <model_basedir>/<model_name>/weights_best.h5   (or weights_last.h5)
    """
    if not STARDIST_OK:
        return jsonify({"error":
            "stardist not installed. Run: pip install stardist tensorflow csbdeep"}), 400

    data       = request.json
    basedir    = data.get("model_basedir", "").strip()
    model_name = data.get("model_name", "stardist_corrosion_particles").strip()

    if not basedir:
        return jsonify({"error": "model_basedir is required"}), 400

    model_path = Path(basedir) / model_name
    if not model_path.exists():
        return jsonify({"error": f"Model folder not found: {model_path}"}), 400
    if not (model_path / "config.json").exists():
        return jsonify({"error":
            f"No config.json found in {model_path}. Is this a valid StarDist model folder?"}), 400

    try:
        STATE["progress"] = {"stage": "load_model", "pct": 10, "msg": "Loading StarDist model…"}
        model = load_stardist_model(basedir, model_name)
        STATE["stardist_model"]      = model
        STATE["stardist_model_dir"]  = basedir
        STATE["stardist_model_name"] = model_name
        STATE["progress"] = {"stage": "load_model", "pct": 100,
                              "msg": f"Model '{model_name}' loaded"}
        return jsonify({
            "ok":         True,
            "model_name": model_name,
            "n_rays":     int(model.config.n_rays),
            "patch_size": list(model.config.train_patch_size),
        })
    except Exception as e:
        STATE["progress"] = {"stage": "load_model_error", "pct": 0, "msg": str(e)}
        return jsonify({"error": str(e)}), 500


@app.route("/api/detect_stardist", methods=["POST"])
def api_detect_stardist():
    """Step 3 — StarDist on the preprocessed frames (Step 2 output)."""
    model = STATE.get("stardist_model")
    if model is None:
        return jsonify({"error":
            "No model loaded. Enter model path and click LOAD MODEL first."}), 400

    prep_str = STATE.get("prep_folder")
    if not prep_str:
        return jsonify({"error":
            "Preprocessed folder not found in session.\n"
            "Complete Step 1 (Crop) then Step 2 (Preprocess) first."}), 400

    prep_folder = Path(prep_str)
    prep_files  = collect_images(prep_folder)
    if not prep_files:
        return jsonify({"error":
            f"Preprocessed folder is empty:\n  {prep_folder}\n"
            "Re-run Step 2 (Preprocess)."}), 400

    data = request.json or {}
    prob_thresh = float(data.get("prob_thresh", STATE.get("stardist_prob_thresh", 0.4)))
    nms_thresh  = float(data.get("nms_thresh",  STATE.get("stardist_nms_thresh",  0.3)))
    norm_pct_lo = float(data.get("norm_pct_lo", 1.0))
    norm_pct_hi = float(data.get("norm_pct_hi", 99.8))
    STATE["stardist_prob_thresh"] = prob_thresh
    STATE["stardist_nms_thresh"]  = nms_thresh

    masks_dir = OUTPUT_DIR / "seg_masks"

    # ── RESET PROGRESS before spawning thread — critical to avoid stale pct=100 ──
    STATE["progress"] = {"stage": "detect_stardist", "pct": 0,
                          "msg": f"Running StarDist on {len(prep_files)} frames…"}

    def _cb(pct, msg):
        STATE["progress"] = {"stage": "detect_stardist", "pct": pct, "msg": msg}

    def _work():
        try:
            dets, masks, renders = detect_particles_stardist(
                prep_folder, model,
                prob_thresh=prob_thresh, nms_thresh=nms_thresh,
                norm_pct_lo=norm_pct_lo, norm_pct_hi=norm_pct_hi,
                masks_out_dir=masks_dir,
                progress_cb=_cb,
            )
            STATE["detections"]       = dets
            STATE["seg_masks"]        = masks
            STATE["seg_renders"]      = renders
            STATE["seg_masks_folder"] = str(masks_dir)
            # Update main canvas with last rendered frame
            if renders:
                STATE["last_frame_b64"] = arr_to_b64(renders[-1])
            total = sum(len(d["centroids"]) for d in dets)
            STATE["stats"].update({
                "n_detections": total,
                "n_frames_det": len(dets),
                "detector":     "StarDist",
                "masks_folder": str(masks_dir),
            })
            STATE["progress"] = {
                "stage": "detect_stardist", "pct": 100,
                "msg":   f"★ {total} particles in {len(dets)} frames  |  masks → seg_masks/"
            }
        except Exception as e:
            STATE["progress"] = {"stage": "detect_stardist_error", "pct": 0, "msg": str(e)}
            print(traceback.format_exc())

    threading.Thread(target=_work, daemon=True).start()
    return jsonify({"ok": True, "n_prep_files": len(prep_files)})


@app.route("/api/get_segmentation_frame")
def api_get_segmentation_frame():
    """
    Serve a frame for the main viewer.

    Base image priority:
      1. seg_renders[fi]  — render_label() RGB (StarDist, proper coloured masks)
      2. preprocessed image + green centroid circles (classic fallback)

    If tracking done and show_tracks=1, overlays trajectories + pit markers.
    Query: frame=<int>, show_tracks=<0|1>
    """
    frame_idx   = int(request.args.get("frame", 0))
    show_tracks = int(request.args.get("show_tracks", 1))

    seg_renders = STATE.get("seg_renders")
    detections  = STATE.get("detections")
    df_tracks   = STATE.get("df_tracks")

    n_frames = (len(seg_renders) if seg_renders
                else len(detections) if detections
                else 0)
    if n_frames == 0:
        return jsonify({"error": "No detection results yet. Run Step 3 first."}), 400

    frame_idx = max(0, min(frame_idx, n_frames - 1))
    n_det     = 0

    # ── Base image ────────────────────────────────────────────────────────────
    if seg_renders and frame_idx < len(seg_renders):
        rgb = seg_renders[frame_idx].copy()
        if detections and frame_idx < len(detections):
            n_det = int(len(detections[frame_idx]["centroids"]))
        has_stardist = True
    else:
        # Classic fallback: preprocessed frame + centroid circles
        prep_str = STATE.get("prep_folder")
        if not prep_str:
            return jsonify({"error": "No preprocessed folder."}), 400
        files = collect_images(Path(prep_str))
        if not files or frame_idx >= len(files):
            return jsonify({"error": "Frame index out of range."}), 400
        img = load_gray(files[frame_idx])
        lo, hi = img.min(), img.max()
        g   = ((img - lo) / (hi - lo) * 255).astype(np.uint8) if hi > lo \
              else np.zeros_like(img, dtype=np.uint8)
        rgb = np.stack([g, g, g], axis=-1)
        if detections and frame_idx < len(detections):
            cents = np.array(detections[frame_idx]["centroids"])
            from PIL import ImageDraw
            pil = Image.fromarray(rgb); drw = ImageDraw.Draw(pil)
            for cy_c, cx_c in cents:
                rr = 5
                drw.ellipse([cx_c-rr, cy_c-rr, cx_c+rr, cy_c+rr],
                            outline=(0, 229, 160), width=2)
            rgb = np.array(pil); n_det = len(cents)
        has_stardist = False

    # ── Overlay trajectories + pit markers ───────────────────────────────────
    if show_tracks and df_tracks is not None and len(df_tracks) > 0:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        H, W = rgb.shape[:2]
        dpi  = 100
        fig, ax = plt.subplots(figsize=(W/dpi, H/dpi), dpi=dpi)
        fig.subplots_adjust(0, 0, 1, 1)
        ax.imshow(rgb); ax.set_xlim(0, W); ax.set_ylim(H, 0); ax.axis("off")
        cmap_t = plt.get_cmap("tab20")
        for tid in df_tracks["track_id"].unique():
            sub = df_tracks[df_tracks["track_id"] == tid].sort_values("time_s")
            if len(sub) < 2: continue
            c  = cmap_t((int(tid) % 20) / 20)
            xs, ys, nt = sub["col_px"].values, sub["row_px"].values, len(sub)
            for i in range(nt - 1):
                alpha = 0.2 + 0.8 * ((i+1) / max(nt-1, 1))
                ax.plot([xs[i], xs[i+1]], [ys[i], ys[i+1]], "-", color=c, lw=1.6, alpha=alpha)
            if nt >= 2:
                ax.annotate("", xy=(xs[-1], ys[-1]), xytext=(xs[-2], ys[-2]),
                            arrowprops=dict(arrowstyle="-|>", color=c, lw=1.0, mutation_scale=10),
                            zorder=7)
            ax.scatter(xs[0],  ys[0],  color=c, s=14, zorder=5, alpha=0.35)
            ax.scatter(xs[-1], ys[-1], color=c, s=32, zorder=6, edgecolors="white", lw=0.5)
        for pit in STATE.get("pit_list", []):
            cx, cy = pit["col_px"], pit["row_px"]
            r = pit.get("radius_px", 15)
            ax.add_patch(plt.Circle((cx, cy), r, fill=False, color="#ff3b3b", lw=2.5, zorder=10))
            ax.plot(cx, cy, "+", color="#ff3b3b", ms=14, mew=2.5, zorder=11)
            ax.text(cx+r+3, cy-5, pit["id"], color="#ff3b3b", fontsize=9,
                    fontweight="bold", zorder=12,
                    bbox=dict(boxstyle="round,pad=0.1", fc="black", alpha=0.5, lw=0))
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight", pad_inches=0)
        plt.close(fig)
        return jsonify({"image": base64.b64encode(buf.getvalue()).decode(),
                        "frame": frame_idx, "n_frames": n_frames,
                        "n_detections": n_det, "has_stardist": has_stardist,
                        "has_tracks": True})

    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="PNG")
    return jsonify({"image": base64.b64encode(buf.getvalue()).decode(),
                    "frame": frame_idx, "n_frames": n_frames,
                    "n_detections": n_det, "has_stardist": has_stardist,
                    "has_tracks": False})


@app.route("/api/track", methods=["POST"])
def api_track():
    if not STATE.get("detections"):
        return jsonify({"error": "Run detection first."}), 400

    data = request.json
    px_um      = float(data.get("px_um", STATE["px_um"]))
    dt_s       = float(data.get("dt_s", STATE["dt_s"]))
    max_dist   = float(data.get("max_dist", 25))
    max_miss   = int(data.get("max_missed", 4))
    min_len    = int(data.get("min_len", 2))
    use_hungarian = bool(data.get("use_hungarian", True))

    STATE["px_um"] = px_um; STATE["dt_s"] = dt_s

    def _cb(pct, msg):
        STATE["progress"] = {"stage": "track", "pct": pct, "msg": msg}

    def _work():
        try:
            STATE["progress"] = {"stage": "track", "pct": 5, "msg": "Starting tracker…"}

            if use_hungarian and HUNGARIAN_OK:
                all_tracks, df_tracks, p50, p80 = run_tracking_hungarian(
                    STATE["detections"], px_um, dt_s,
                    max_dist_px=max_dist, max_missed=max_miss, min_len=min_len,
                    progress_cb=_cb,
                )
                algo = "Hungarian"
            else:
                # Greedy fallback (works without scipy.optimize)
                all_tracks, df_tracks, p50, p80 = run_tracking_simple(
                    STATE["detections"], px_um, dt_s, max_dist, max_miss, min_len)
                algo = "Greedy NN"

            STATE["all_tracks"] = all_tracks
            STATE["df_tracks"]  = df_tracks
            n = len(all_tracks)
            STATE["stats"]["n_tracks"]  = n
            STATE["stats"]["p50_nn"]    = round(p50, 2)
            STATE["stats"]["p80_nn"]    = round(p80, 2)
            STATE["stats"]["track_algo"] = algo
            STATE["progress"] = {
                "stage": "track", "pct": 100,
                "msg": f"{n} tracks [{algo}]  |  P80 NN={p80:.1f}px"
            }
        except Exception as e:
            STATE["progress"] = {"stage": "track_error", "pct": 0, "msg": str(e)}
            print(traceback.format_exc())

    threading.Thread(target=_work, daemon=True).start()
    return jsonify({"ok": True})


@app.route("/api/get_frame_preview")
def api_get_frame_preview():
    b64 = STATE.get("last_frame_b64")
    if not b64:
        return jsonify({"error": "No frame loaded"}), 404
    return jsonify({"image": b64, "stats": STATE.get("stats", {})})


@app.route("/api/set_pits", methods=["POST"])
def api_set_pits():
    data = request.json
    pit_list = data.get("pits", [])
    px_um = STATE.get("px_um", 0.5)
    for p in pit_list:
        p["col_um"] = p["col_px"] * px_um
        p["row_um"] = p["row_px"] * px_um
        if "radius_px" not in p: p["radius_px"] = 15
        if "radius_um" not in p: p["radius_um"] = p["radius_px"] * px_um
    STATE["pit_list"] = pit_list
    return jsonify({"ok": True, "n_pits": len(pit_list)})


@app.route("/api/analyse", methods=["POST"])
def api_analyse():
    if STATE.get("df_tracks") is None:
        return jsonify({"error": "Run tracking first."}), 400
    if not STATE.get("pit_list"):
        return jsonify({"error": "Place at least one pit marker."}), 400

    data = request.json
    max_assign = float(data.get("max_assign_um", 300))

    def _work():
        try:
            STATE["progress"] = {"stage":"analyse","pct":20,"msg":"Assigning particles to pits…"}
            raw = STATE.get("crop_folder") or STATE.get("raw_folder")
            df_t = assign_particles_to_pits(
                STATE["df_tracks"], STATE["pit_list"], STATE["px_um"],
                raw, max_assign_um=max_assign)
            STATE["df_t"] = df_t
            STATE["progress"] = {"stage":"analyse","pct":70,"msg":"Building plots…"}

            # Build overlay
            lf = None
            raw_folder = STATE.get("crop_folder") or STATE.get("raw_folder")
            if raw_folder:
                files = collect_images(Path(raw_folder))
                if files:
                    try: lf = load_gray(files[-1])
                    except: pass
            STATE["overlay_b64"] = make_overlay_plot(df_t, STATE["pit_list"], lf)

            # Build per-pit plots
            pit_plots = {}
            for pit in STATE["pit_list"]:
                pit_plots[pit["id"]] = make_pit_plots(df_t, pit, lf, STATE["dt_s"])
            STATE["pit_plots"] = pit_plots

            # Stats per pit
            pit_stats = {}
            for pit in STATE["pit_list"]:
                pid = pit["id"]
                sub = df_t[df_t["pit_id"] == pid]
                pit_stats[pid] = {
                    "n_tracks": int(sub["track_id"].nunique()),
                    "n_obs":    len(sub),
                    "mean_r":   round(float(sub["r_um"].mean()), 2) if len(sub) else 0,
                    "max_r":    round(float(sub["r_um"].max()), 2) if len(sub) else 0,
                }
            STATE["pit_stats"] = pit_stats
            STATE["progress"] = {"stage":"analyse","pct":100,"msg":"Analysis complete!"}
        except Exception as e:
            STATE["progress"] = {"stage":"analyse_error","pct":0,"msg":str(e)}
            print(traceback.format_exc())

    threading.Thread(target=_work, daemon=True).start()
    return jsonify({"ok": True})


@app.route("/api/get_results")
def api_get_results():
    if STATE.get("df_t") is None:
        return jsonify({"error": "No results yet"}), 404

    return jsonify({
        "overlay_b64": STATE.get("overlay_b64"),
        "pit_plots":   STATE.get("pit_plots", {}),
        "pit_stats":   STATE.get("pit_stats", {}),
        "stats":       STATE.get("stats", {}),
        "pit_list":    STATE.get("pit_list", []),
    })


@app.route("/api/export_csv")
def api_export_csv():
    if STATE.get("df_t") is None:
        return jsonify({"error": "No results yet"}), 404
    buf = io.StringIO()
    STATE["df_t"].to_csv(buf, index=False)
    buf.seek(0)
    return send_file(io.BytesIO(buf.getvalue().encode()),
                     mimetype="text/csv",
                     as_attachment=True,
                     download_name="pit_analysis.csv")


@app.route("/api/export_detections_pkl")
def api_export_detections_pkl():
    if not STATE.get("detections"):
        return jsonify({"error": "No detections"}), 404
    buf = io.BytesIO()
    pickle.dump(STATE["detections"], buf)
    buf.seek(0)
    return send_file(buf, mimetype="application/octet-stream",
                     as_attachment=True, download_name="detections.pkl")


@app.route("/api/export_seg_masks")
def api_export_seg_masks():
    import zipfile
    folder = STATE.get("seg_masks_folder")
    if not folder or not Path(folder).exists():
        return jsonify({"error": "No masks saved. Run StarDist detection first."}), 404
    files = sorted(Path(folder).glob("mask_*"))
    if not files:
        return jsonify({"error": "No mask files found."}), 404
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for fp in files:
            zf.write(fp, fp.name)
    buf.seek(0)
    return send_file(buf, mimetype="application/zip",
                     as_attachment=True, download_name="seg_masks.zip")


@app.route("/api/get_detection_frame")
def api_get_detection_frame():
    frame_idx = int(request.args.get("frame", 0))
    src_str = STATE.get("prep_folder") or STATE.get("crop_folder") or STATE.get("raw_folder")
    if not src_str:
        return jsonify({"error": "No folder loaded"}), 400
    files = collect_images(Path(src_str))
    if not files:
        return jsonify({"error": "No frames found"}), 400
    frame_idx = max(0, min(frame_idx, len(files) - 1))

    img = load_gray(files[frame_idx])
    lo, hi = img.min(), img.max()
    img8 = ((img - lo) / (hi - lo) * 255).astype(np.uint8) if hi > lo else np.zeros_like(img, dtype=np.uint8)

    from PIL import ImageDraw
    pil_img = Image.fromarray(img8).convert("RGB")
    draw = ImageDraw.Draw(pil_img)

    detections = STATE.get("detections")
    n_det = 0
    if detections and frame_idx < len(detections):
        cents = np.array(detections[frame_idx]["centroids"])
        if len(cents) > 0:
            n_det = len(cents)
            for cy, cx in cents:
                r = 5
                draw.ellipse([cx-r, cy-r, cx+r, cy+r], outline=(0, 229, 160), width=2)
                draw.point((cx, cy), fill=(0, 229, 160))

    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return jsonify({
        "image":        base64.b64encode(buf.getvalue()).decode(),
        "frame":        frame_idx,
        "n_frames":     len(files),
        "n_detections": n_det,
        "filename":     files[frame_idx].name,
    })


@app.route("/api/export_results", methods=["POST"])
def api_export_results():
    """
    Save all visual results to a named folder inside outputs/:
      detections/   — every preprocessed frame with StarDist colour masks  (PNG)
      trajectories/ — trajectory overlay (PNG, one image)
      analysis/     — per-pit r(t), α(t), I(t) figures                     (PNG)
      pit_analysis.csv

    Body: { "folder_name": "my_experiment_01" }
    """
    data        = request.json or {}
    folder_name = (data.get("folder_name") or "export").strip().replace(" ", "_")
    if not folder_name:
        folder_name = "export"

    export_dir = OUTPUT_DIR / folder_name
    export_dir.mkdir(parents=True, exist_ok=True)

    saved = []

    # ── 1. Detection frames (StarDist render_label renders) ───────────────────
    seg_renders = STATE.get("seg_renders")
    if seg_renders:
        det_dir = export_dir / "detections"
        det_dir.mkdir(exist_ok=True)
        for i, rgb in enumerate(seg_renders):
            fp = det_dir / f"detection_{i:04d}.png"
            Image.fromarray(rgb).save(fp)
            saved.append(str(fp))

    # ── 2. Trajectory overlay ─────────────────────────────────────────────────
    overlay_b64 = STATE.get("overlay_b64")
    if overlay_b64:
        traj_dir = export_dir / "trajectories"
        traj_dir.mkdir(exist_ok=True)
        img_bytes = base64.b64decode(overlay_b64)
        fp = traj_dir / "trajectory_overlay.png"
        fp.write_bytes(img_bytes)
        saved.append(str(fp))

    # Also save a full trajectory overlay rendered at higher resolution
    df_tracks = STATE.get("df_tracks")
    seg_renders_list = STATE.get("seg_renders")
    if df_tracks is not None and len(df_tracks) > 0 and seg_renders_list:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        traj_dir = export_dir / "trajectories"
        traj_dir.mkdir(exist_ok=True)
        n_frames = len(seg_renders_list)
        for fi, rgb in enumerate(seg_renders_list):
            H, W = rgb.shape[:2]
            dpi = 150
            fig, ax = plt.subplots(figsize=(W/dpi, H/dpi), dpi=dpi)
            fig.subplots_adjust(0, 0, 1, 1)
            ax.imshow(rgb); ax.set_xlim(0, W); ax.set_ylim(H, 0); ax.axis("off")
            cmap_t = plt.get_cmap("tab20")
            # Tracks up to this frame
            df_here = df_tracks[df_tracks["frame"] <= fi]
            for tid in df_here["track_id"].unique():
                sub = df_here[df_here["track_id"] == tid].sort_values("time_s")
                if len(sub) < 2: continue
                c = cmap_t((int(tid) % 20) / 20)
                xs, ys = sub["col_px"].values, sub["row_px"].values
                ax.plot(xs, ys, "-", color=c, lw=1.4, alpha=0.85)
                ax.scatter(xs[-1], ys[-1], color=c, s=20, zorder=5, edgecolors="w", lw=0.3)
            for pit in STATE.get("pit_list", []):
                cx, cy, r = pit["col_px"], pit["row_px"], pit.get("radius_px", 15)
                ax.add_patch(plt.Circle((cx, cy), r, fill=False, color="#ff3b3b", lw=2, zorder=8))
                ax.plot(cx, cy, "+", color="#ff3b3b", ms=10, mew=2, zorder=9)
            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight", pad_inches=0)
            plt.close(fig)
            fp = traj_dir / f"traj_frame_{fi:04d}.png"
            fp.write_bytes(buf.getvalue())
            saved.append(str(fp))

    # ── 3. Analysis figures (per-pit plots, 3 images each) ───────────────────
    pit_plots = STATE.get("pit_plots")
    if pit_plots:
        ana_dir = export_dir / "analysis"
        ana_dir.mkdir(exist_ok=True)
        for pid, plots in pit_plots.items():
            safe_id = pid.replace("/", "_").replace("\\", "_")
            if isinstance(plots, dict):
                for key, b64 in plots.items():
                    label = {"r_b64": "distance", "a_b64": "angle", "i_b64": "intensity"}.get(key, key)
                    fp = ana_dir / f"{safe_id}_{label}.png"
                    fp.write_bytes(base64.b64decode(b64))
                    saved.append(str(fp))
            else:
                # Legacy fallback (single b64 string)
                fp = ana_dir / f"{safe_id}_analysis.png"
                fp.write_bytes(base64.b64decode(plots))
                saved.append(str(fp))

    # ── 4. CSV ────────────────────────────────────────────────────────────────
    if STATE.get("df_t") is not None:
        csv_fp = export_dir / "pit_analysis.csv"
        STATE["df_t"].to_csv(csv_fp, index=False)
        saved.append(str(csv_fp))

    return jsonify({
        "ok":         True,
        "folder":     str(export_dir),
        "n_saved":    len(saved),
        "subfolders": {
            "detections":   str(export_dir / "detections"),
            "trajectories": str(export_dir / "trajectories"),
            "analysis":     str(export_dir / "analysis"),
        }
    })


@app.route("/api/reset", methods=["POST"])
def api_reset():
    for k in ["raw_folder","crop_folder","prep_folder","detections",
              "df_tracks","all_tracks","pit_list","last_frame_b64",
              "df_t","overlay_b64","pit_plots","pit_stats",
              "stardist_model","stardist_model_dir","stardist_model_name",
              "seg_masks","seg_renders","seg_masks_folder"]:
        if k == "pit_list": STATE[k] = []
        else: STATE[k] = None
    STATE["stardist_prob_thresh"] = 0.4
    STATE["stardist_nms_thresh"]  = 0.3
    STATE["progress"] = {"stage":"idle","pct":0,"msg":""}
    STATE["stats"] = {}
    return jsonify({"ok": True})


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  Pit Analyzer Web App")
    print("  → http://127.0.0.1:5000")
    print("=" * 60)
    print(f"  scipy:     {'✓' if SCIPY_OK else '✗ missing'}")
    print(f"  tifffile:  {'✓' if TIFFFILE_OK else '✗ missing'}")
    print(f"  opencv:    {'✓' if CV2_OK else '✗ missing'}")
    print(f"  skimage:   {'✓' if SKIMAGE_OK else '✗ missing'}")
    print(f"  stardist:  {'✓' if STARDIST_OK else '✗  pip install stardist tensorflow csbdeep'}")
    print(f"  hungarian: {'✓' if HUNGARIAN_OK else '✗  pip install scipy'}")
    print()
    app.run(debug=False, host="127.0.0.1", port=5000)
