"""
analyze.py — MRI Quality Degradation vs Brain Tumor Size Measurement Accuracy

This script quantifies the effect of simulated MRI image quality degradation on
brain tumor size measurement accuracy using the BraTS 2021 Training Dataset.

Pipeline:
  1. discover_patients    — scan dataset directory for valid patient folders
  2. compute_gold_standard — measure tumor sizes from the original seg mask
  3. simulate_degradations — apply 9 degradation variants per patient
  4. measure_degraded      — re-measure tumor sizes from each degraded mask
  5. compute_errors        — compute MAE, % error, Dice, clinical danger flag
  6. save_results          — write results.csv and print grouped summary
  7. generate_plots        — produce 4 publication-quality PNG figures
"""

import os
import glob

import nibabel as nib
import numpy as np
import scipy.ndimage
import skimage
import matplotlib.pyplot as plt
import matplotlib.patches
import seaborn as sns
import pandas as pd

# ---------------------------------------------------------------------------
# Module-level path constants
# ---------------------------------------------------------------------------

DATASET_DIR = r"C:\Users\marwa\Pictures\Marwah (1)\Projects\Project 2\Data\BraTS2021_Training_Data"
OUTPUT_CSV  = r"C:\Users\marwa\Pictures\Marwah (1)\Projects\Project 2\results.csv"
OUTPUT_DIR  = r"C:\Users\marwa\Pictures\Marwah (1)\Projects\Project 2"
EXAMPLE_ID  = "BraTS2021_00506"


# ---------------------------------------------------------------------------
# Pipeline steps (stubs — implemented in subsequent tasks)
# ---------------------------------------------------------------------------

def discover_patients(dataset_dir: str, max_patients: int = 100) -> list:
    """
    Scans dataset_dir for subdirectories that contain both *_t1ce.nii.gz
    and *_seg.nii.gz files. Returns up to max_patients valid entries.

    Each entry dict:
        {
            "patient_id": str,   # e.g. "BraTS2021_00000"
            "t1ce_path":  str,   # absolute path to *_t1ce.nii.gz
            "seg_path":   str,   # absolute path to *_seg.nii.gz
        }
    """
    valid = []

    for entry in os.listdir(dataset_dir):
        if len(valid) >= max_patients:
            break

        subdir = os.path.join(dataset_dir, entry)
        if not os.path.isdir(subdir):
            continue

        t1ce_matches = glob.glob(os.path.join(subdir, "*_t1ce.nii.gz"))
        seg_matches  = glob.glob(os.path.join(subdir, "*_seg.nii.gz"))

        if not t1ce_matches or not seg_matches:
            continue

        patient_id = entry
        print(f"Processing patient {len(valid) + 1}/{max_patients}: {patient_id}")

        valid.append({
            "patient_id": patient_id,
            "t1ce_path":  t1ce_matches[0],
            "seg_path":   seg_matches[0],
        })

    return valid


def compute_gold_standard(
    seg_data: np.ndarray,
    t1ce_data: np.ndarray,
    spacing: tuple,
):
    """
    Computes gold-standard tumor measurements from the original seg mask.

    Returns a dict with volume/area measurements, binary_mask, and
    mid_slice_idx, or None if the whole-tumor volume is zero.
    """
    voxel_volume_mm3 = spacing[0] * spacing[1] * spacing[2]
    pixel_area_mm2 = spacing[0] * spacing[1]

    # Compute 4 volume measurements
    gold_whole_tumor_vol_mm3 = float(np.sum(seg_data > 0) * voxel_volume_mm3)
    gold_enhancing_tumor_vol_mm3 = float(np.sum(seg_data == 4) * voxel_volume_mm3)
    gold_necrotic_vol_mm3 = float(np.sum(seg_data == 1) * voxel_volume_mm3)
    gold_edema_vol_mm3 = float(np.sum(seg_data == 2) * voxel_volume_mm3)

    # Return None if whole-tumor volume is zero (patient skipped)
    if gold_whole_tumor_vol_mm3 == 0:
        return None

    # Middle axial slice index
    mid_slice_idx = seg_data.shape[2] // 2

    # Compute whole-tumor area on the middle axial slice
    gold_whole_tumor_area_mm2 = float(
        np.sum(seg_data[:, :, mid_slice_idx] > 0) * pixel_area_mm2
    )

    # Binary mask derived from seg
    binary_mask = (seg_data > 0).astype(np.uint8)

    return {
        "gold_whole_tumor_vol_mm3":     gold_whole_tumor_vol_mm3,
        "gold_enhancing_tumor_vol_mm3": gold_enhancing_tumor_vol_mm3,
        "gold_necrotic_vol_mm3":        gold_necrotic_vol_mm3,
        "gold_edema_vol_mm3":           gold_edema_vol_mm3,
        "gold_whole_tumor_area_mm2":    gold_whole_tumor_area_mm2,
        "binary_mask":                  binary_mask,
        "mid_slice_idx":                mid_slice_idx,
    }


def simulate_degradations(
    seg_data: np.ndarray,
    binary_mask: np.ndarray,
) -> list:
    """
    Applies 9 degradation variants (erosion, downsampling, motion blur ×
    3 levels each) and returns a list of 9 dicts.

    Each dict:
        {
            "degradation_type":  str,        # "erosion" | "downsampling" | "motion_blur"
            "degradation_level": int,        # 1, 2, or 3
            "degraded_binary":   np.ndarray, # uint8, same shape as binary_mask
            "degraded_labeled":  np.ndarray, # same dtype as seg_data, same shape
        }
    """
    results = []

    # --- Erosion: L1=1 iter, L2=2 iter, L3=3 iter ---
    for level, iterations in enumerate([1, 2, 3], start=1):
        eroded = scipy.ndimage.binary_erosion(binary_mask, iterations=iterations)
        degraded_binary = eroded.astype(np.uint8)
        degraded_labeled = np.where(degraded_binary, seg_data, 0).astype(seg_data.dtype)
        results.append({
            "degradation_type":  "erosion",
            "degradation_level": level,
            "degraded_binary":   degraded_binary,
            "degraded_labeled":  degraded_labeled,
        })

    # --- Downsampling: L1=0.5, L2=0.33, L3=0.25 ---
    for level, zoom_factor in enumerate([0.5, 0.33, 0.25], start=1):
        downsampled = scipy.ndimage.zoom(binary_mask.astype(float), zoom_factor, order=1)
        # Compute per-axis back-zoom factors
        zoom_back = tuple(
            o / d for o, d in zip(binary_mask.shape, downsampled.shape)
        )
        restored = scipy.ndimage.zoom(downsampled, zoom_back, order=1)
        degraded_binary = (restored > 0.5).astype(np.uint8)
        degraded_labeled = np.where(degraded_binary, seg_data, 0).astype(seg_data.dtype)
        results.append({
            "degradation_type":  "downsampling",
            "degradation_level": level,
            "degraded_binary":   degraded_binary,
            "degraded_labeled":  degraded_labeled,
        })

    # --- Motion Blur: L1=sigma 1, L2=sigma 2, L3=sigma 3 ---
    for level, sigma in enumerate([1, 2, 3], start=1):
        blurred = scipy.ndimage.gaussian_filter(binary_mask.astype(float), sigma=sigma)
        degraded_binary = (blurred > 0.5).astype(np.uint8)
        degraded_labeled = np.where(degraded_binary, seg_data, 0).astype(seg_data.dtype)
        results.append({
            "degradation_type":  "motion_blur",
            "degradation_level": level,
            "degraded_binary":   degraded_binary,
            "degraded_labeled":  degraded_labeled,
        })

    return results


def measure_degraded(
    degraded_labeled: np.ndarray,
    spacing: tuple,
    mid_slice_idx: int,
) -> dict:
    """
    Applies the same measurement formula as compute_gold_standard to a
    degraded labeled mask.

    Returns dict:
        {
            "degraded_whole_tumor_vol_mm3":    float,
            "degraded_enhancing_tumor_vol_mm3": float,
            "degraded_necrotic_vol_mm3":        float,
            "degraded_edema_vol_mm3":           float,
            "degraded_whole_tumor_area_mm2":    float,
        }
    """
    voxel_volume_mm3 = spacing[0] * spacing[1] * spacing[2]
    pixel_area_mm2 = spacing[0] * spacing[1]

    degraded_whole_tumor_vol_mm3 = float(np.sum(degraded_labeled > 0) * voxel_volume_mm3)
    degraded_enhancing_tumor_vol_mm3 = float(np.sum(degraded_labeled == 4) * voxel_volume_mm3)
    degraded_necrotic_vol_mm3 = float(np.sum(degraded_labeled == 1) * voxel_volume_mm3)
    degraded_edema_vol_mm3 = float(np.sum(degraded_labeled == 2) * voxel_volume_mm3)

    degraded_whole_tumor_area_mm2 = float(
        np.sum(degraded_labeled[:, :, mid_slice_idx] > 0) * pixel_area_mm2
    )

    return {
        "degraded_whole_tumor_vol_mm3":     degraded_whole_tumor_vol_mm3,
        "degraded_enhancing_tumor_vol_mm3": degraded_enhancing_tumor_vol_mm3,
        "degraded_necrotic_vol_mm3":        degraded_necrotic_vol_mm3,
        "degraded_edema_vol_mm3":           degraded_edema_vol_mm3,
        "degraded_whole_tumor_area_mm2":    degraded_whole_tumor_area_mm2,
    }


def compute_errors(
    gold: dict,
    degraded_meas: dict,
    original_binary: np.ndarray,
    degraded_binary: np.ndarray,
) -> dict:
    """
    Computes volume_MAE_mm3, volume_pct_error, area_MAE_mm2, dice_score,
    and clinically_dangerous for a single patient × degradation pair.
    """
    volume_MAE_mm3 = abs(
        degraded_meas["degraded_whole_tumor_vol_mm3"]
        - gold["gold_whole_tumor_vol_mm3"]
    )
    volume_pct_error = (volume_MAE_mm3 / gold["gold_whole_tumor_vol_mm3"]) * 100

    area_MAE_mm2 = abs(
        degraded_meas["degraded_whole_tumor_area_mm2"]
        - gold["gold_whole_tumor_area_mm2"]
    )

    dice_score = (
        2 * np.sum(original_binary & degraded_binary)
        / (np.sum(original_binary) + np.sum(degraded_binary) + 1e-8)
    )

    clinically_dangerous = volume_pct_error > 20

    return {
        "volume_MAE_mm3":       float(volume_MAE_mm3),
        "volume_pct_error":     float(volume_pct_error),
        "area_MAE_mm2":         float(area_MAE_mm2),
        "dice_score":           float(dice_score),
        "clinically_dangerous": bool(clinically_dangerous),
    }


def save_results(all_records: list, output_csv_path: str) -> pd.DataFrame:
    """
    Converts all_records to a DataFrame, saves to output_csv_path, prints
    a grouped summary, and returns the DataFrame.
    """
    # Define exact column order per Requirement 6.2
    columns = [
        "patient_id",
        "degradation_type",
        "degradation_level",
        "gold_whole_tumor_vol_mm3",
        "degraded_whole_tumor_vol_mm3",
        "volume_MAE_mm3",
        "volume_pct_error",
        "gold_area_mm2",
        "degraded_area_mm2",
        "area_MAE_mm2",
        "dice_score",
        "clinically_dangerous",
    ]

    # Convert records list to DataFrame with enforced column order
    df = pd.DataFrame(all_records, columns=columns)

    # Create output directory if it doesn't exist (Requirement 6.1)
    output_dir = os.path.dirname(output_csv_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Save to CSV without row index (Requirement 6.1)
    df.to_csv(output_csv_path, index=False)

    # Print grouped summary: mean ± std of volume_pct_error and dice_score
    # grouped by degradation_type and degradation_level (Requirement 6.3)
    summary = df.groupby(["degradation_type", "degradation_level"])[
        ["volume_pct_error", "dice_score"]
    ].agg(["mean", "std"])
    print("\n=== Grouped Summary (mean ± std) ===")
    print(summary.to_string())
    print()

    return df


def generate_plots(
    df: pd.DataFrame,
    example_patient: dict,
    output_dir: str,
) -> None:
    """
    Generates and saves plot1.png through plot4.png to output_dir.
    Uses seaborn style. All figures >= (12, 6).
    """
    import matplotlib
    matplotlib.use('Agg')

    # -----------------------------------------------------------------------
    # Task 8.1 — Set up style and label column
    # -----------------------------------------------------------------------
    sns.set_style("whitegrid")

    # Build display-name mapping for degradation types
    type_display_map = {
        "erosion":      "Noise",
        "downsampling": "Downsampling",
        "motion_blur":  "Motion Blur",
    }

    df = df.copy()
    df["label"] = df.apply(
        lambda row: f"{type_display_map.get(row['degradation_type'], row['degradation_type'])}-L{row['degradation_level']}",
        axis=1,
    )
    label_col = "label"

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # -----------------------------------------------------------------------
    # Task 8.2 — Plot 1: Line chart
    # -----------------------------------------------------------------------
    grouped_line = (
        df.groupby(["degradation_type", "degradation_level"])["volume_pct_error"]
        .mean()
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(12, 6))

    for deg_type, display_name in type_display_map.items():
        subset = grouped_line[grouped_line["degradation_type"] == deg_type].sort_values("degradation_level")
        ax.plot(
            subset["degradation_level"],
            subset["volume_pct_error"],
            marker="o",
            label=display_name,
        )

    ax.axhline(y=20, color="red", linestyle="--", label="Clinical Danger Threshold (RANO)")
    ax.set_xticks([1, 2, 3])
    ax.set_xlabel("Degradation Level")
    ax.set_ylabel("Mean Volume % Error (%)")
    ax.set_title("Tumor Volume Measurement Error vs. MRI Degradation Level")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "plot1.png"), dpi=150)
    plt.close(fig)

    # -----------------------------------------------------------------------
    # Task 8.3 — Plot 2: Box + strip plot
    # -----------------------------------------------------------------------
    # Build a consistent label order: Noise-L1..3, Downsampling-L1..3, Motion Blur-L1..3
    label_order = [
        f"{display}-L{lvl}"
        for display in ["Noise", "Downsampling", "Motion Blur"]
        for lvl in [1, 2, 3]
    ]
    # Keep only labels that actually appear in data
    label_order = [lbl for lbl in label_order if lbl in df[label_col].unique()]

    fig, ax = plt.subplots(figsize=(14, 7))
    sns.boxplot(
        data=df,
        x=label_col,
        y="volume_pct_error",
        order=label_order,
        ax=ax,
        hue=label_col,
        palette="Set2",
        legend=False,
    )
    sns.stripplot(
        data=df,
        x=label_col,
        y="volume_pct_error",
        order=label_order,
        ax=ax,
        alpha=0.3,
        color="black",
        jitter=True,
    )
    ax.axhline(y=20, color="red", linestyle="--", label="Clinical Danger Threshold (RANO)")
    ax.set_xlabel("Degradation Type and Level")
    ax.set_ylabel("Volume % Error (%)")
    ax.set_title("Distribution of Tumor Size Measurement Error by Degradation")
    ax.legend()
    plt.xticks(rotation=45, ha="right")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "plot2.png"), dpi=150)
    plt.close(fig)

    # -----------------------------------------------------------------------
    # Task 8.4 — Plot 3: 4-row segmentation overlay grid
    # -----------------------------------------------------------------------
    if example_patient is None:
        print("[WARNING] example_patient is None — skipping plot3.png")
    else:
        t1ce_data   = example_patient["t1ce_data"]
        seg_data    = example_patient["seg_data"]
        degradations = example_patient["degradations"]  # list of 9 dicts
        mid_idx     = example_patient["mid_slice_idx"]

        # Extract mid axial slices
        t1ce_slice = t1ce_data[:, :, mid_idx]

        # Original binary mask slice
        original_mask_slice = (seg_data[:, :, mid_idx] > 0).astype(np.uint8)

        # Locate the three L2 degradations
        def _find_degradation(deg_list, deg_type, deg_level):
            for d in deg_list:
                if d["degradation_type"] == deg_type and d["degradation_level"] == deg_level:
                    return d["degraded_binary"][:, :, mid_idx]
            return None

        noise_l2_slice       = _find_degradation(degradations, "erosion",      2)
        downsample_l2_slice  = _find_degradation(degradations, "downsampling",  2)
        motionblur_l2_slice  = _find_degradation(degradations, "motion_blur",   2)

        rows_data = [
            (original_mask_slice,      "green", "Row 0: Original Mask (Green)"),
            (noise_l2_slice,           "red",   "Row 1: Noise L2 Degraded Mask (Red)"),
            (downsample_l2_slice,      "red",   "Row 2: Downsampling L2 Degraded Mask (Red)"),
            (motionblur_l2_slice,      "red",   "Row 3: Motion Blur L2 Degraded Mask (Red)"),
        ]

        fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(8, 20))

        # Normalise T1ce for display
        t1ce_min = t1ce_slice.min()
        t1ce_max = t1ce_slice.max()
        t1ce_norm = (t1ce_slice - t1ce_min) / (t1ce_max - t1ce_min + 1e-8)

        color_rgba_map = {
            "green": (0.0, 1.0, 0.0),
            "red":   (1.0, 0.0, 0.0),
        }

        for row_idx, (mask_slice, color_name, row_label) in enumerate(rows_data):
            ax = axes[row_idx]
            ax.imshow(t1ce_norm.T, cmap="gray", origin="lower")

            if mask_slice is not None:
                r, g, b = color_rgba_map[color_name]
                h, w = mask_slice.shape
                rgba_overlay = np.zeros((h, w, 4), dtype=np.float32)
                rgba_overlay[..., 0] = r
                rgba_overlay[..., 1] = g
                rgba_overlay[..., 2] = b
                rgba_overlay[..., 3] = mask_slice.astype(np.float32) * 0.4
                ax.imshow(rgba_overlay.transpose(1, 0, 2), origin="lower")

            ax.set_title(row_label, fontsize=10)
            ax.axis("off")

        fig.suptitle(
            "Effect of MRI Degradation on Tumor Segmentation Boundary",
            fontsize=13,
            y=1.01,
        )
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, "plot3.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)

    # -----------------------------------------------------------------------
    # Task 8.5 — Plot 4: Bar chart
    # -----------------------------------------------------------------------
    pct_dangerous = (
        df.groupby(label_col)["clinically_dangerous"].mean() * 100
    ).reset_index()
    pct_dangerous.columns = [label_col, "pct_dangerous"]

    # Sort by label_order for consistency
    pct_dangerous[label_col] = pd.Categorical(
        pct_dangerous[label_col], categories=label_order, ordered=True
    )
    pct_dangerous = pct_dangerous.sort_values(label_col)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(
        pct_dangerous[label_col].astype(str),
        pct_dangerous["pct_dangerous"],
        color=sns.color_palette("Set2", len(pct_dangerous)),
        label="% Clinically Dangerous",
    )
    ax.set_xlabel("Degradation Type and Level")
    ax.set_ylabel("% Patients Exceeding 20% Volume Error")
    ax.set_title(
        "Percentage of Patients Exceeding 20% Volume Error Threshold (RANO Clinical Limit)"
    )
    ax.legend()
    plt.xticks(rotation=45, ha="right")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "plot4.png"), dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """
    Orchestrates the full pipeline end-to-end.
    """
    patients = discover_patients(DATASET_DIR)
    all_records: list = []
    example_patient = None

    for patient in patients:
        try:
            # Step 1-2: Load NIfTI images
            seg_img  = nib.load(patient["seg_path"])
            t1ce_img = nib.load(patient["t1ce_path"])

            # Step 3-5: Extract arrays and spacing
            seg_data  = seg_img.get_fdata().astype(np.int16)
            t1ce_data = t1ce_img.get_fdata()
            spacing   = seg_img.header.get_zooms()[:3]

            # Step 6: Compute gold standard; skip patient if None
            gold = compute_gold_standard(seg_data, t1ce_data, spacing)
            if gold is None:
                continue

            # Step 7: Simulate degradations
            degradations = simulate_degradations(seg_data, gold["binary_mask"])

            # Step 8: Capture example patient — prefer EXAMPLE_ID, fallback to first valid
            if example_patient is None or patient["patient_id"] == EXAMPLE_ID:
                example_patient = {
                    "patient_id":    patient["patient_id"],
                    "t1ce_data":     t1ce_data,
                    "seg_data":      seg_data,
                    "degradations":  degradations,
                    "mid_slice_idx": gold["mid_slice_idx"],
                }

            # Step 9: Measure each degradation and accumulate records
            for deg in degradations:
                meas = measure_degraded(
                    deg["degraded_labeled"], spacing, gold["mid_slice_idx"]
                )
                errors = compute_errors(
                    gold, meas, gold["binary_mask"], deg["degraded_binary"]
                )
                all_records.append({
                    "patient_id":                   patient["patient_id"],
                    "degradation_type":             deg["degradation_type"],
                    "degradation_level":            deg["degradation_level"],
                    "gold_whole_tumor_vol_mm3":     gold["gold_whole_tumor_vol_mm3"],
                    "degraded_whole_tumor_vol_mm3": meas["degraded_whole_tumor_vol_mm3"],
                    "volume_MAE_mm3":               errors["volume_MAE_mm3"],
                    "volume_pct_error":             errors["volume_pct_error"],
                    "gold_area_mm2":                gold["gold_whole_tumor_area_mm2"],
                    "degraded_area_mm2":            meas["degraded_whole_tumor_area_mm2"],
                    "area_MAE_mm2":                 errors["area_MAE_mm2"],
                    "dice_score":                   errors["dice_score"],
                    "clinically_dangerous":         errors["clinically_dangerous"],
                })
        except Exception as e:
            print(f"  [WARN] Skipping patient {patient['patient_id']}: {e}")
            continue

    df = save_results(all_records, OUTPUT_CSV)
    generate_plots(df, example_patient, OUTPUT_DIR)
    print("Done.")


if __name__ == "__main__":
    main()
