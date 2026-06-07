# Implementation Plan: Brain Tumor Size Estimator

## Overview

Implement a single Python script (`analyze.py`) in `C:\Users\marwa\Pictures\Marwah (1)\Projects\Project 2\` that processes the BraTS 2021 Training Dataset to study the effect of MRI quality degradation on tumor size measurement accuracy. The script is built step-by-step, one function per pipeline stage, wired together in `main()`.

---

## Tasks

- [x] 1. Create the script file and set up imports and constants
  - Create `analyze.py` at `C:\Users\marwa\Pictures\Marwah (1)\Projects\Project 2\analyze.py`
  - Add all required imports at the top: `os`, `glob`, `nibabel as nib`, `numpy as np`, `scipy.ndimage`, `skimage`, `matplotlib.pyplot as plt`, `matplotlib.patches`, `seaborn as sns`, `pandas as pd`
  - Define module-level path constants: `DATASET_DIR`, `OUTPUT_CSV`, `OUTPUT_DIR`, `EXAMPLE_ID = "BraTS2021_00506"`
  - Add the `if __name__ == "__main__": main()` guard at the bottom (stub `main()` for now)
  - _Requirements: 8.2, 8.3, 8.4_

- [x] 2. Implement `discover_patients`
  - [x] 2.1 Write `discover_patients(dataset_dir, max_patients=100) -> list[dict]`
    - Use `os.listdir` to enumerate subdirectories
    - For each subdir use `glob.glob` to find `*_t1ce.nii.gz` and `*_seg.nii.gz`
    - Skip subdir if either glob returns empty
    - Collect up to `max_patients` entries as dicts with keys `patient_id`, `t1ce_path`, `seg_path`
    - Print `"Processing patient X/100: BraTS2021_XXXXX"` for each valid patient collected
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5_

  - [ ]* 2.2 Write property tests for `discover_patients`
    - **Property 1: Patient filter accepts only complete folders** — generate synthetic folder lists with varying file presence; assert only complete folders are returned
    - **Property 2: Patient count never exceeds the maximum limit** — generate arbitrarily long valid folder lists; assert `len(result) <= max_patients`
    - **Validates: Requirements 1.2, 1.3, 1.4**

- [x] 3. Implement `compute_gold_standard`
  - [x] 3.1 Write `compute_gold_standard(seg_data, t1ce_data, spacing) -> dict | None`
    - Compute `voxel_volume_mm3 = spacing[0] * spacing[1] * spacing[2]`
    - Compute `pixel_area_mm2 = spacing[0] * spacing[1]`
    - Compute all 4 volume measurements (WT, ET, NCR, ED) and the middle-slice area
    - Derive `binary_mask = (seg_data > 0).astype(np.uint8)` and store `mid_slice_idx`
    - Return `None` if `gold_whole_tumor_vol_mm3 == 0`
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9_

  - [ ]* 3.2 Write property tests for measurement computations
    - **Property 3: Voxel volume and pixel area scaling** — generate random positive spacing tuples; assert volume scales as `k³` and area as `k²` when spacing is scaled by `k`
    - **Property 4: Volume measurement proportionality** — generate synthetic 3-D masks with known voxel counts; assert computed volume equals `count * voxel_volume_mm3`
    - **Property 5: Middle axial slice area** — generate synthetic 3-D masks; assert computed area equals 2-D non-zero pixel count times `pixel_area_mm2`
    - **Validates: Requirements 2.3, 2.4, 2.5, 2.6, 2.7, 2.8**

  - [ ]* 3.3 Write unit test: `compute_gold_standard` returns `None` for all-zero mask
    - _Requirements: 2.9_

- [x] 4. Implement `simulate_degradations`
  - [x] 4.1 Write `simulate_degradations(seg_data, binary_mask) -> list[dict]`
    - Implement Erosion L1–L3 using `scipy.ndimage.binary_erosion` with iterations=1,2,3
    - Implement Downsampling L1–L3 with zoom factors 0.5, 0.33, 0.25; zoom down with `order=1`, compute per-axis back-zoom factors, zoom back up, threshold at 0.5
    - Implement Motion Blur L1–L3 using `scipy.ndimage.gaussian_filter` with sigma=1,2,3 then threshold at 0.5
    - For each, compute `degraded_labeled = np.where(degraded_binary, seg_data, 0).astype(seg_data.dtype)`
    - Return exactly 9 dicts, each with keys `degradation_type`, `degradation_level`, `degraded_binary`, `degraded_labeled`
    - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9, 3.10, 3.11_

  - [ ]* 4.2 Write property tests for degradation simulation
    - **Property 6: Degradation monotonicity** — generate random binary masks; for each of the 9 degradation configs, assert `np.all((degraded_binary - original_binary) <= 0)`
    - **Property 7: Degraded labeled values are a subset of original seg values** — generate synthetic seg arrays; assert all non-zero values in `degraded_labeled` appear in original `seg_data` and zero locations are 0
    - **Validates: Requirements 3.2–3.11**

  - [ ]* 4.3 Write unit test: `simulate_degradations` returns exactly 9 entries
    - _Requirements: 3.2–3.10_

- [x] 5. Checkpoint — ensure steps 1–4 are working
  - Ensure all tests pass, ask the user if questions arise.

- [x] 6. Implement `measure_degraded` and `compute_errors`
  - [x] 6.1 Write `measure_degraded(degraded_labeled, spacing, mid_slice_idx) -> dict`
    - Apply the same formula as `compute_gold_standard` to a degraded labeled mask
    - Return `degraded_whole_tumor_vol_mm3`, `degraded_enhancing_tumor_vol_mm3`, `degraded_necrotic_vol_mm3`, `degraded_edema_vol_mm3`, `degraded_whole_tumor_area_mm2`
    - _Requirements: 4.1, 4.2_

  - [x] 6.2 Write `compute_errors(gold, degraded_meas, original_binary, degraded_binary) -> dict`
    - Compute `volume_MAE_mm3`, `volume_pct_error`, `area_MAE_mm2`, `dice_score`, `clinically_dangerous`
    - Use the exact formulas from the design document (including `+ 1e-8` in dice denominator)
    - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5_

  - [ ]* 6.3 Write property tests for error metrics
    - **Property 8: Error metric non-negativity and boundedness** — generate random gold/degraded measurement pairs; assert `volume_MAE >= 0`, `volume_pct_error >= 0`, `area_MAE >= 0`, `dice in [0,1]`, and `clinically_dangerous == (volume_pct_error > 20)`
    - **Property 9: Dice score is 1 for identical masks** — generate random non-empty binary masks; assert `dice(A, A) == 1.0` within 1e-6 tolerance
    - **Validates: Requirements 5.1, 5.2, 5.3, 5.4, 5.5**

  - [ ]* 6.4 Write unit tests for `compute_errors`
    - Test `clinically_dangerous = True` when `volume_pct_error = 20.1`
    - Test `clinically_dangerous = False` when `volume_pct_error = 19.9`
    - _Requirements: 5.5_

- [x] 7. Implement `save_results`
  - [x] 7.1 Write `save_results(all_records, output_csv_path) -> pd.DataFrame`
    - Convert `all_records` to a `pandas.DataFrame`
    - Ensure column order matches the spec: `patient_id`, `degradation_type`, `degradation_level`, `gold_whole_tumor_vol_mm3`, `degraded_whole_tumor_vol_mm3`, `volume_MAE_mm3`, `volume_pct_error`, `gold_area_mm2`, `degraded_area_mm2`, `area_MAE_mm2`, `dice_score`, `clinically_dangerous`
    - Create output directory with `os.makedirs(..., exist_ok=True)` if needed
    - Save with `df.to_csv(output_csv_path, index=False)`
    - Print grouped summary: `df.groupby(["degradation_type","degradation_level"])[["volume_pct_error","dice_score"]].agg(["mean","std"])`
    - Return the DataFrame
    - _Requirements: 6.1, 6.2, 6.3_

  - [ ]* 7.2 Write example-based test for CSV column schema
    - Build a small list of mock records and call `save_results`; read back CSV and assert column names match spec exactly
    - _Requirements: 6.2_

- [x] 8. Implement `generate_plots`
  - [x] 8.1 Write `generate_plots(df, example_patient, output_dir) -> None`
    - Set seaborn style at the start of the function: `sns.set_style("whitegrid")`
    - Add a `label` column to `df` combining type and level (e.g., `"Noise-L1"`, `"Downsampling-L2"`, `"Motion Blur-L3"`)

  - [x] 8.2 Implement Plot 1 (line chart) in `generate_plots`
    - Figure size `(12, 6)`; group by `degradation_type` + `degradation_level`, plot mean `volume_pct_error` per line
    - Add `axhline(y=20, color='red', linestyle='--', label='Clinical Danger Threshold (RANO)')`
    - X-axis: degradation level (1,2,3); legend; axis labels; title
    - Save to `os.path.join(output_dir, "plot1.png")`
    - _Requirements: 7.1, 7.5, 7.6_

  - [x] 8.3 Implement Plot 2 (box + strip plot) in `generate_plots`
    - Figure size `(14, 7)`; `sns.boxplot` + `sns.stripplot` (alpha=0.3) on `volume_pct_error` by label column
    - Add `axhline(y=20, color='red', linestyle='--')`
    - Save to `os.path.join(output_dir, "plot2.png")`
    - _Requirements: 7.2, 7.5, 7.6_

  - [x] 8.4 Implement Plot 3 (4-row segmentation overlay grid) in `generate_plots`
    - If `example_patient` is None, print a warning and skip this plot
    - Extract mid axial slice for t1ce and each of the 4 relevant masks (original + Noise L2 + Downsampling L2 + Motion Blur L2)
    - Create a `(4, 1)` subplot grid; for each row: `imshow(t1ce_slice, cmap='gray')` then overlay mask as RGBA image (green for original, red for degraded) with `alpha=0.4`
    - Label each row; overall title; figure size `(8, 20)` (tall layout)
    - Save to `os.path.join(output_dir, "plot3.png")`
    - _Requirements: 7.3, 7.5_

  - [x] 8.5 Implement Plot 4 (bar chart) in `generate_plots`
    - Compute `pct_dangerous` per label: `df.groupby(label_col)["clinically_dangerous"].mean() * 100`
    - Figure size `(12, 6)`; `plt.bar` or `sns.barplot`; axis labels; title
    - Save to `os.path.join(output_dir, "plot4.png")`
    - _Requirements: 7.4, 7.5, 7.6_

- [x] 9. Wire everything together in `main()`
  - [x] 9.1 Implement the full `main()` function
    - Call `discover_patients(DATASET_DIR)` → `patients`
    - Loop over patients with `try/except` wrapping all per-patient logic; print warning and `continue` on exception
    - Inside loop: load with `nibabel`, call `compute_gold_standard`, skip if `None`, call `simulate_degradations`, capture `example_patient` (prefer `EXAMPLE_ID`, fallback to first valid), loop over degradations calling `measure_degraded` + `compute_errors`, append to `all_records`
    - Call `save_results(all_records, OUTPUT_CSV)` → `df`
    - Call `generate_plots(df, example_patient, OUTPUT_DIR)`
    - Print `"Done."`
    - _Requirements: 1.1–1.5, 2.1–2.9, 3.1–3.11, 4.1–4.2, 5.1–5.5, 6.1–6.3, 7.1–7.6, 8.1–8.4_

  - [ ]* 9.2 Write example-based test: pipeline does not abort on a broken patient
    - Create a mock patient with an unreadable file path; patch `discover_patients` to return that patient first followed by a valid one; assert `main()` completes and `all_records` contains results for the valid patient
    - _Requirements: 8.1_

- [-] 10. Final checkpoint — ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

---

## Notes

- Tasks marked with `*` are optional and can be skipped for a faster MVP.
- All property tests use the `hypothesis` library (already in scope as a dev dependency) and run at least 100 iterations each.
- Each task references specific requirements for traceability.
- `analyze.py` should be the only new source file; no subpackages are needed.
- The `results.csv` and `plot*.png` files are output artifacts, not source files.
