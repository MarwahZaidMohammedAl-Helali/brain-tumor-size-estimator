# Requirements Document

## Introduction

This project is a research Python script that quantifies how MRI image quality degradation affects brain tumor size measurement accuracy. It uses the BraTS 2021 Training Dataset (~1,100 patients), applies three types of simulated degradation (erosion/noise, resolution downsampling, motion blur) at three severity levels each, and measures the resulting errors in tumor volume and area estimates. Results are saved to a CSV and four publication-quality plots. The clinical significance threshold follows the RANO (Response Assessment in Neuro-Oncology) criteria: a volume error exceeding 20% is flagged as clinically dangerous.

## Glossary

- **BraTS2021**: Brain Tumor Segmentation 2021 challenge dataset. Each patient folder contains multi-modal MRI scans and an expert segmentation mask.
- **t1ce**: T1 contrast-enhanced MRI modality used for visualization.
- **seg**: Segmentation mask NIfTI file encoding tumor sub-regions: 0 = background, 1 = necrotic core (NCR), 2 = edema (ED), 4 = enhancing tumor (ET).
- **NIfTI**: Neuroimaging Informatics Technology Initiative file format (`.nii.gz`). Loaded via `nibabel`.
- **Gold Standard**: Measurements derived directly from the original expert segmentation mask, treated as ground truth.
- **Degraded Mask**: A binary mask derived by applying a simulated image-quality degradation to the gold-standard binary mask, then re-labeling with the original seg values.
- **Voxel Spacing**: Physical dimensions (mm) of each voxel in x, y, z, extracted from the NIfTI header via `get_zooms()[:3]`.
- **Whole Tumor (WT)**: Union of all non-background label regions (seg > 0).
- **Enhancing Tumor (ET)**: seg == 4.
- **Necrotic Core (NCR)**: seg == 1.
- **Edema (ED)**: seg == 2.
- **Middle Axial Slice**: The 2D slice at index `shape[2] // 2` along the axial (z) axis.
- **Erosion**: Morphological binary erosion applied iteratively to simulate boundary noise.
- **Downsampling**: Spatial resolution reduction via zoom, simulating lower-resolution acquisition.
- **Motion Blur**: Gaussian blurring of the binary mask, simulating patient movement artifacts.
- **Dice Score**: Overlap metric between two binary masks: `2 * |A ∩ B| / (|A| + |B|)`.
- **Volume MAE**: Mean Absolute Error of whole-tumor volume between gold and degraded mask (mm³).
- **Area MAE**: Mean Absolute Error of the whole-tumor area on the middle axial slice (mm²).
- **RANO Threshold**: Response Assessment in Neuro-Oncology criterion; volume percentage error > 20% is clinically dangerous.
- **Script**: The single Python file implementing all steps.
- **Pipeline**: The end-to-end execution from patient discovery through result saving and plotting.

---

## Requirements

### Requirement 1: Patient Discovery

**User Story:** As a researcher, I want the Script to discover all valid patient folders in the dataset directory, so that I can process a representative sample without manually managing file paths.

#### Acceptance Criteria

1. WHEN the Script starts, THE Script SHALL scan the directory `C:\Users\marwa\Pictures\Marwah (1)\Projects\Project 2\Data\BraTS2021_Training_Data\` for all immediate subdirectories.
2. FOR each subdirectory found, THE Script SHALL check whether both a `*_t1ce.nii.gz` file and a `*_seg.nii.gz` file exist within that subdirectory.
3. IF either the `*_t1ce.nii.gz` or `*_seg.nii.gz` file is missing for a patient folder, THEN THE Script SHALL skip that folder and continue to the next.
4. THE Script SHALL collect only the first 100 subdirectories that pass the file-existence check.
5. WHEN processing each patient, THE Script SHALL print a progress message in the format `"Processing patient X/100: BraTS2021_XXXXX"` where X is the 1-based index.

---

### Requirement 2: Gold Standard Measurement

**User Story:** As a researcher, I want the Script to compute gold-standard tumor size measurements from the original segmentation masks, so that I have a baseline against which to compare degraded measurements.

#### Acceptance Criteria

1. WHEN loading patient data, THE Script SHALL load the segmentation mask using `nibabel` from the `*_seg.nii.gz` file.
2. WHEN loading patient data, THE Script SHALL load the T1ce image using `nibabel` from the `*_t1ce.nii.gz` file (for visualization only).
3. THE Script SHALL extract voxel spacing as `spacing = nib.load(path).header.get_zooms()[:3]` and compute `voxel_volume_mm3 = spacing[0] * spacing[1] * spacing[2]` and `pixel_area_mm2 = spacing[0] * spacing[1]`.
4. THE Script SHALL compute `gold_whole_tumor_vol` as the count of voxels where `seg > 0` multiplied by `voxel_volume_mm3`.
5. THE Script SHALL compute `gold_enhancing_tumor_vol` as the count of voxels where `seg == 4` multiplied by `voxel_volume_mm3`.
6. THE Script SHALL compute `gold_necrotic_vol` as the count of voxels where `seg == 1` multiplied by `voxel_volume_mm3`.
7. THE Script SHALL compute `gold_edema_vol` as the count of voxels where `seg == 2` multiplied by `voxel_volume_mm3`.
8. THE Script SHALL compute `gold_whole_tumor_area` as the count of pixels where `seg > 0` on the middle axial slice (index `seg.shape[2] // 2`) multiplied by `pixel_area_mm2`.
9. IF `gold_whole_tumor_vol == 0` for a patient, THEN THE Script SHALL skip that patient and continue to the next.

---

### Requirement 3: Degradation Simulation

**User Story:** As a researcher, I want the Script to simulate three types of MRI quality degradation at three severity levels each, so that I can study a range of real-world quality issues.

#### Acceptance Criteria

1. THE Script SHALL derive a binary mask `binary_mask = (seg_data > 0).astype(np.uint8)` from the gold-standard segmentation for each patient.
2. THE Script SHALL apply Erosion degradation at Level 1 using `scipy.ndimage.binary_erosion(binary_mask, iterations=1)`.
3. THE Script SHALL apply Erosion degradation at Level 2 using `scipy.ndimage.binary_erosion(binary_mask, iterations=2)`.
4. THE Script SHALL apply Erosion degradation at Level 3 using `scipy.ndimage.binary_erosion(binary_mask, iterations=3)`.
5. THE Script SHALL apply Downsampling degradation at Level 1 by zooming to factor 0.5, then zooming back to the original shape using per-axis factors, with threshold 0.5 applied to produce a binary mask.
6. THE Script SHALL apply Downsampling degradation at Level 2 using zoom factor 0.33.
7. THE Script SHALL apply Downsampling degradation at Level 3 using zoom factor 0.25.
8. THE Script SHALL apply Motion Blur degradation at Level 1 using `scipy.ndimage.gaussian_filter(binary_mask.astype(float), sigma=1) > 0.5`.
9. THE Script SHALL apply Motion Blur degradation at Level 2 using sigma=2.
10. THE Script SHALL apply Motion Blur degradation at Level 3 using sigma=3.
11. FOR each degraded binary mask produced, THE Script SHALL construct a labeled degraded mask as `degraded_labeled = np.where(degraded_binary_mask, seg_data, 0)`.

---

### Requirement 4: Re-measurement on Degraded Masks

**User Story:** As a researcher, I want the Script to re-measure tumor size from each degraded mask using the same method as the gold standard, so that measurements are directly comparable.

#### Acceptance Criteria

1. FOR each degraded labeled mask, THE Script SHALL compute `degraded_whole_tumor_vol`, `degraded_enhancing_tumor_vol`, `degraded_necrotic_vol`, and `degraded_edema_vol` using the same voxel-volume formula applied in Requirement 2.
2. FOR each degraded labeled mask, THE Script SHALL compute `degraded_whole_tumor_area` on the middle axial slice using the same pixel-area formula applied in Requirement 2.

---

### Requirement 5: Error Metric Computation

**User Story:** As a researcher, I want the Script to compute error and overlap metrics for each patient–degradation combination, so that I can quantify measurement inaccuracy.

#### Acceptance Criteria

1. FOR each patient × degradation type × degradation level combination, THE Script SHALL compute `volume_MAE_mm3 = abs(degraded_whole_tumor_vol - gold_whole_tumor_vol)`.
2. FOR each combination, THE Script SHALL compute `volume_pct_error = (volume_MAE_mm3 / gold_whole_tumor_vol) * 100`.
3. FOR each combination, THE Script SHALL compute `area_MAE_mm2 = abs(degraded_whole_tumor_area - gold_whole_tumor_area)`.
4. FOR each combination, THE Script SHALL compute `dice_score = 2 * np.sum(original_binary & degraded_binary) / (np.sum(original_binary) + np.sum(degraded_binary) + 1e-8)` where `original_binary` is the gold-standard binary mask and `degraded_binary` is the degraded binary mask.
5. FOR each combination, THE Script SHALL set `clinically_dangerous = True` if `volume_pct_error > 20`, otherwise `False`.

---

### Requirement 6: Results Persistence

**User Story:** As a researcher, I want the Script to save all computed metrics to a structured CSV file and print a summary, so that I can perform further statistical analysis.

#### Acceptance Criteria

1. THE Script SHALL save results to `C:\Users\marwa\Pictures\Marwah (1)\Projects\Project 2\results.csv`.
2. THE CSV SHALL contain exactly these columns: `patient_id`, `degradation_type`, `degradation_level`, `gold_whole_tumor_vol_mm3`, `degraded_whole_tumor_vol_mm3`, `volume_MAE_mm3`, `volume_pct_error`, `gold_area_mm2`, `degraded_area_mm2`, `area_MAE_mm2`, `dice_score`, `clinically_dangerous`.
3. THE Script SHALL print a summary table showing mean ± std of `volume_pct_error` and `dice_score` grouped by `degradation_type` and `degradation_level`.

---

### Requirement 7: Visualization

**User Story:** As a researcher, I want the Script to generate four publication-quality plots saved as PNG files, so that I can communicate findings visually.

#### Acceptance Criteria

1. THE Script SHALL save `plot1.png` to `C:\Users\marwa\Pictures\Marwah (1)\Projects\Project 2\` as a line chart with X=degradation level, Y=mean `volume_pct_error`, one line per degradation type, a red dashed reference line at y=20 labeled "Clinical Danger Threshold (RANO)", and title "Tumor Volume Measurement Error vs. MRI Degradation Level".
2. THE Script SHALL save `plot2.png` as a box plot with X=degradation type+level label (e.g. "Noise-L1"), Y=`volume_pct_error`, individual patient data points overlaid, a red dashed reference line at y=20, and title "Distribution of Tumor Size Measurement Error by Degradation".
3. THE Script SHALL save `plot3.png` as a 4-row image grid for one example patient (BraTS2021_00506 if valid, otherwise the first valid patient), where each row shows the middle axial slice of the T1ce image with a mask overlay: Row 1 = original mask in green (alpha=0.4), Row 2 = Noise L2 degraded mask in red (alpha=0.4), Row 3 = Downsampling L2 degraded mask in red (alpha=0.4), Row 4 = Motion Blur L2 degraded masks in red (alpha=0.4); rows are labeled and the overall title is "Effect of MRI Degradation on Tumor Segmentation Boundary".
4. THE Script SHALL save `plot4.png` as a bar chart with X=degradation type+level, Y=percentage of patients with `clinically_dangerous == True`, and title "Percentage of Patients Exceeding 20% Volume Error Threshold (RANO Clinical Limit)".
5. ALL plots SHALL use seaborn style and include axis labels, a title, and a legend where applicable.
6. ALL plots SHALL have a figure size of at least (12, 6).

---

### Requirement 8: Robustness and Error Handling

**User Story:** As a researcher, I want the Script to handle individual patient failures gracefully, so that a single corrupt or missing file does not abort the entire pipeline.

#### Acceptance Criteria

1. THE Script SHALL wrap each per-patient processing block in a `try/except` block and call `continue` on any exception, allowing the Pipeline to proceed to the next patient.
2. THE Script SHALL run entirely on CPU with no GPU dependency.
3. THE Script SHALL use only the following libraries: `os`, `glob`, `nibabel`, `numpy`, `scipy`, `scikit-image`, `matplotlib`, `seaborn`, `pandas`.
4. THE Script SHALL be a single Python file structured with one function per step and a `main()` function that calls all steps in order, guarded by `if __name__ == "__main__": main()`.
