# Design Document: MRI Quality Degradation vs Brain Tumor Size Measurement Accuracy

## Overview

This is a single-file Python research script (`analyze.py`) that quantifies the effect of simulated MRI image quality degradation on brain tumor size measurement accuracy. The script uses the BraTS 2021 Training Dataset. It processes up to 100 valid patients, applies 9 degradation variants (3 types × 3 levels), computes error metrics per patient–degradation pair, saves all results to a CSV, and generates 4 publication-quality plots. No machine learning or GPU is required; all operations are CPU-based pure Python/NumPy/SciPy.

The pipeline follows a strict sequential structure: discover patients → load data and compute gold standards → simulate degradations → re-measure degraded masks → compute error metrics → save CSV → generate plots.

---

## Architecture

```
analyze.py
│
├── discover_patients(dataset_dir, max_patients=100) → list[dict]
│
├── compute_gold_standard(seg_data, t1ce_data, spacing) → dict
│
├── simulate_degradations(seg_data, binary_mask) → list[dict]
│
├── measure_degraded(degraded_labeled, spacing) → dict
│
├── compute_errors(gold, degraded_meas, binary_mask, degraded_binary) → dict
│
├── save_results(all_records, output_csv_path)
│
├── generate_plots(df, example_patient_data, output_dir)
│
└── main()
```

All inter-step data is passed through plain Python dicts and lists of dicts, which are later converted to a `pandas.DataFrame` for aggregation and plotting. No global state is used.

---

## Components and Interfaces

### Step 1: `discover_patients`

```python
def discover_patients(
    dataset_dir: str,
    max_patients: int = 100
) -> list[dict]:
    """
    Scans dataset_dir for subdirectories that contain both *_t1ce.nii.gz
    and *_seg.nii.gz files. Returns up to max_patients valid entries.

    Each entry dict:
        {
            "patient_id": str,       # e.g. "BraTS2021_00000"
            "t1ce_path": str,        # absolute path to *_t1ce.nii.gz
            "seg_path":  str,        # absolute path to *_seg.nii.gz
        }

    Prints: "Processing patient X/100: BraTS2021_XXXXX" during iteration.
    """
```

**Implementation notes:**
- Use `os.listdir(dataset_dir)` to enumerate subdirectories.
- Use `glob.glob(os.path.join(subdir, "*_t1ce.nii.gz"))` and `glob.glob(..., "*_seg.nii.gz")` for file discovery.
- Collect entries until `len(valid) == max_patients`.

---

### Step 2: `compute_gold_standard`

```python
def compute_gold_standard(
    seg_data: np.ndarray,       # 3-D integer array from seg NIfTI
    t1ce_data: np.ndarray,      # 3-D float array from t1ce NIfTI (for viz)
    spacing: tuple[float, float, float]  # (sx, sy, sz) in mm
) -> dict:
    """
    Computes gold-standard tumor measurements.

    Returns dict:
        {
            "gold_whole_tumor_vol_mm3":    float,
            "gold_enhancing_tumor_vol_mm3": float,
            "gold_necrotic_vol_mm3":        float,
            "gold_edema_vol_mm3":           float,
            "gold_whole_tumor_area_mm2":    float,
            "binary_mask":                  np.ndarray (uint8, 3-D),
            "mid_slice_idx":                int,
        }

    Returns None if gold_whole_tumor_vol_mm3 == 0 (patient skipped).
    """
```

**Implementation notes:**
- `voxel_volume_mm3 = spacing[0] * spacing[1] * spacing[2]`
- `pixel_area_mm2 = spacing[0] * spacing[1]`
- `mid_slice_idx = seg_data.shape[2] // 2`
- `gold_whole_tumor_area_mm2 = np.sum(seg_data[:, :, mid_slice_idx] > 0) * pixel_area_mm2`
- Return `None` if `gold_whole_tumor_vol_mm3 == 0`.

---

### Step 3: `simulate_degradations`

```python
def simulate_degradations(
    seg_data: np.ndarray,
    binary_mask: np.ndarray   # uint8, same shape as seg_data
) -> list[dict]:
    """
    Applies 9 degradation variants and returns a list of dicts.

    Each dict:
        {
            "degradation_type":  str,        # "erosion" | "downsampling" | "motion_blur"
            "degradation_level": int,        # 1, 2, or 3
            "degraded_binary":   np.ndarray, # uint8, same shape as binary_mask
            "degraded_labeled":  np.ndarray, # int, same shape as seg_data
        }
    """
```

**Erosion** — uses `scipy.ndimage.binary_erosion`:
| Level | iterations |
|-------|-----------|
| 1     | 1         |
| 2     | 2         |
| 3     | 3         |

**Downsampling** — uses `scipy.ndimage.zoom`:
| Level | zoom_factor_down |
|-------|----------------|
| 1     | 0.50            |
| 2     | 0.33            |
| 3     | 0.25            |

```python
# Pseudocode for one downsampling level:
downsampled = scipy.ndimage.zoom(binary_mask.astype(float), zoom_factor, order=1)
zoom_back = tuple(o / d for o, d in zip(binary_mask.shape, downsampled.shape))
restored   = scipy.ndimage.zoom(downsampled, zoom_back, order=1)
degraded_binary = (restored > 0.5).astype(np.uint8)
```

**Motion Blur** — uses `scipy.ndimage.gaussian_filter`:
| Level | sigma |
|-------|-------|
| 1     | 1     |
| 2     | 2     |
| 3     | 3     |

```python
blurred = scipy.ndimage.gaussian_filter(binary_mask.astype(float), sigma=sigma)
degraded_binary = (blurred > 0.5).astype(np.uint8)
```

For all types: `degraded_labeled = np.where(degraded_binary, seg_data, 0).astype(seg_data.dtype)`

---

### Step 4: `measure_degraded`

```python
def measure_degraded(
    degraded_labeled: np.ndarray,
    spacing: tuple[float, float, float],
    mid_slice_idx: int
) -> dict:
    """
    Applies the same measurement formula as compute_gold_standard
    to a degraded labeled mask.

    Returns dict:
        {
            "degraded_whole_tumor_vol_mm3":    float,
            "degraded_enhancing_tumor_vol_mm3": float,
            "degraded_necrotic_vol_mm3":        float,
            "degraded_edema_vol_mm3":           float,
            "degraded_whole_tumor_area_mm2":    float,
        }
    """
```

---

### Step 5: `compute_errors`

```python
def compute_errors(
    gold: dict,
    degraded_meas: dict,
    original_binary: np.ndarray,   # uint8
    degraded_binary: np.ndarray    # uint8
) -> dict:
    """
    Computes comparison metrics between gold and degraded measurements.

    Returns dict:
        {
            "volume_MAE_mm3":       float,
            "volume_pct_error":     float,
            "area_MAE_mm2":         float,
            "dice_score":           float,   # in [0, 1]
            "clinically_dangerous": bool,
        }
    """
```

Formulas:
```python
volume_MAE_mm3   = abs(degraded_whole_tumor_vol - gold_whole_tumor_vol)
volume_pct_error = (volume_MAE_mm3 / gold_whole_tumor_vol) * 100
area_MAE_mm2     = abs(degraded_whole_tumor_area - gold_whole_tumor_area)
dice_score       = 2 * np.sum(original_binary & degraded_binary) \
                   / (np.sum(original_binary) + np.sum(degraded_binary) + 1e-8)
clinically_dangerous = volume_pct_error > 20
```

---

### Step 6: `save_results`

```python
def save_results(
    all_records: list[dict],
    output_csv_path: str
) -> pd.DataFrame:
    """
    Converts all_records to a DataFrame, saves it to output_csv_path,
    and prints a grouped summary (mean ± std of volume_pct_error and dice_score
    grouped by degradation_type and degradation_level).

    Returns the DataFrame for use in plot generation.
    """
```

CSV columns (in order):
`patient_id`, `degradation_type`, `degradation_level`, `gold_whole_tumor_vol_mm3`, `degraded_whole_tumor_vol_mm3`, `volume_MAE_mm3`, `volume_pct_error`, `gold_area_mm2`, `degraded_area_mm2`, `area_MAE_mm2`, `dice_score`, `clinically_dangerous`

---

### Step 7: `generate_plots`

```python
def generate_plots(
    df: pd.DataFrame,
    example_patient: dict,   # {"t1ce_data": ndarray, "seg_data": ndarray,
                             #  "degradations": list[dict], "mid_slice_idx": int}
    output_dir: str
) -> None:
    """
    Generates and saves plot1.png through plot4.png to output_dir.
    Uses seaborn style. All figures >= (12, 6).
    """
```

**Plot 1 — Line chart:**
```python
# Group df by degradation_type + degradation_level, compute mean volume_pct_error
# One line per degradation_type
# axhline at y=20, color='red', linestyle='--', label='Clinical Danger Threshold (RANO)'
```

**Plot 2 — Box + strip plot:**
```python
# Create label column: e.g. "Noise-L1", "Downsampling-L1", "Motion Blur-L1"
# seaborn boxplot + stripplot (alpha=0.3, jitter=True, dodge=True)
# axhline at y=20
```

**Plot 3 — 4-row image grid (1 patient):**
```python
# Row 0: T1ce + original mask overlay (green, alpha=0.4)
# Row 1: T1ce + Noise L2 degraded mask (red, alpha=0.4)
# Row 2: T1ce + Downsampling L2 degraded mask (red, alpha=0.4)
# Row 3: T1ce + Motion Blur L2 degraded mask (red, alpha=0.4)
# Use matplotlib imshow with cmap='gray' for T1ce, then overlay mask as colored alpha layer
```

**Plot 4 — Bar chart:**
```python
# Compute pct_dangerous = df.groupby(label_col)['clinically_dangerous'].mean() * 100
# seaborn barplot or plt.bar
```

---

### `main`

```python
def main() -> None:
    DATASET_DIR  = r"C:\Users\marwa\Pictures\Marwah (1)\Projects\Project 2\Data\BraTS2021_Training_Data"
    OUTPUT_CSV   = r"C:\Users\marwa\Pictures\Marwah (1)\Projects\Project 2\results.csv"
    OUTPUT_DIR   = r"C:\Users\marwa\Pictures\Marwah (1)\Projects\Project 2"
    EXAMPLE_ID   = "BraTS2021_00506"

    patients       = discover_patients(DATASET_DIR)
    all_records    = []
    example_patient = None

    for i, patient in enumerate(patients, 1):
        try:
            # Load
            seg_img  = nib.load(patient["seg_path"])
            t1ce_img = nib.load(patient["t1ce_path"])
            seg_data  = seg_img.get_fdata().astype(np.int16)
            t1ce_data = t1ce_img.get_fdata()
            spacing   = seg_img.header.get_zooms()[:3]

            # Gold standard
            gold = compute_gold_standard(seg_data, t1ce_data, spacing)
            if gold is None:
                continue

            # Degradations
            degradations = simulate_degradations(seg_data, gold["binary_mask"])

            # Capture example patient data
            if example_patient is None or patient["patient_id"] == EXAMPLE_ID:
                example_patient = {
                    "patient_id":   patient["patient_id"],
                    "t1ce_data":    t1ce_data,
                    "seg_data":     seg_data,
                    "degradations": degradations,
                    "mid_slice_idx": gold["mid_slice_idx"],
                }

            # Measure + compute errors
            for deg in degradations:
                meas = measure_degraded(
                    deg["degraded_labeled"], spacing, gold["mid_slice_idx"]
                )
                errors = compute_errors(
                    gold, meas, gold["binary_mask"], deg["degraded_binary"]
                )
                all_records.append({
                    "patient_id":               patient["patient_id"],
                    "degradation_type":         deg["degradation_type"],
                    "degradation_level":        deg["degradation_level"],
                    "gold_whole_tumor_vol_mm3": gold["gold_whole_tumor_vol_mm3"],
                    "degraded_whole_tumor_vol_mm3": meas["degraded_whole_tumor_vol_mm3"],
                    "volume_MAE_mm3":           errors["volume_MAE_mm3"],
                    "volume_pct_error":         errors["volume_pct_error"],
                    "gold_area_mm2":            gold["gold_whole_tumor_area_mm2"],
                    "degraded_area_mm2":        meas["degraded_whole_tumor_area_mm2"],
                    "area_MAE_mm2":             errors["area_MAE_mm2"],
                    "dice_score":               errors["dice_score"],
                    "clinically_dangerous":     errors["clinically_dangerous"],
                })
        except Exception as e:
            print(f"  [WARN] Skipping patient {patient['patient_id']}: {e}")
            continue

    df = save_results(all_records, OUTPUT_CSV)
    generate_plots(df, example_patient, OUTPUT_DIR)
    print("Done.")
```

---

## Data Models

### Patient Record (dict)
```
patient_id                  : str
degradation_type            : str    # "erosion" | "downsampling" | "motion_blur"
degradation_level           : int    # 1, 2, 3
gold_whole_tumor_vol_mm3    : float
degraded_whole_tumor_vol_mm3: float
volume_MAE_mm3              : float
volume_pct_error            : float  # >= 0
gold_area_mm2               : float
degraded_area_mm2           : float
area_MAE_mm2                : float
dice_score                  : float  # [0, 1]
clinically_dangerous        : bool
```

### Degradation Descriptor (dict)
```
degradation_type  : str
degradation_level : int
degraded_binary   : np.ndarray  # dtype uint8, shape = seg_data.shape
degraded_labeled  : np.ndarray  # dtype int16, shape = seg_data.shape
```

### Gold Standard Output (dict)
```
gold_whole_tumor_vol_mm3    : float
gold_enhancing_tumor_vol_mm3: float
gold_necrotic_vol_mm3       : float
gold_edema_vol_mm3          : float
gold_whole_tumor_area_mm2   : float
binary_mask                 : np.ndarray  # uint8
mid_slice_idx               : int
```

---

## Correctness Properties

*A property is a characteristic or behavior that should hold true across all valid executions of a system — essentially, a formal statement about what the system should do. Properties serve as the bridge between human-readable specifications and machine-verifiable correctness guarantees.*

**Prework reflection:** Several properties deal with pure arithmetic or mask transformations. After reflection:
- Volume measurement properties (2.4, 2.5-2.7) are combined into one since they follow the same formula pattern.
- Error metric properties (5.1 volume_MAE, 5.2 pct_error, 5.5 dangerous flag) are combined since they are all derived from the same computation chain.
- Erosion, downsampling, and motion blur share the invariant that the degraded mask is always a subset of the original mask (erosion strictly; downsampling and blur approximately via threshold). These are expressed as a shared monotonicity property.

---

### Property 1: Patient filter accepts only complete folders

*For any* list of folders where each folder may or may not contain both required files (`*_t1ce.nii.gz`, `*_seg.nii.gz`), the discover function SHALL return only folders where both files are present.

**Validates: Requirements 1.2, 1.3**

---

### Property 2: Patient count never exceeds the maximum limit

*For any* input list of valid patient folders of arbitrary length, the discover function SHALL return at most `max_patients` entries.

**Validates: Requirements 1.4**

---

### Property 3: Voxel volume and pixel area computation correctness

*For any* voxel spacing tuple `(sx, sy, sz)` with positive values, `voxel_volume_mm3 = sx * sy * sz` and `pixel_area_mm2 = sx * sy` SHALL be computed correctly such that scaling each dimension by a factor `k` scales the volume by `k³` and the area by `k²`.

**Validates: Requirements 2.3**

---

### Property 4: Volume measurement is proportional to labeled voxel count

*For any* synthetic 3-D mask array and voxel spacing, the whole-tumor volume SHALL equal the count of non-zero voxels multiplied by the voxel volume, and each sub-region volume SHALL equal the count of voxels matching that label multiplied by the voxel volume.

**Validates: Requirements 2.4, 2.5, 2.6, 2.7**

---

### Property 5: Middle axial slice area is consistent with 3-D mask

*For any* 3-D mask array and pixel spacing, the whole-tumor area on the middle slice (index `shape[2] // 2`) SHALL equal the count of non-zero pixels on that slice multiplied by `pixel_area_mm2`.

**Validates: Requirements 2.8**

---

### Property 6: Degradation never introduces new tumor voxels (monotonicity)

*For any* binary mask and any of the 9 degradation configurations (erosion L1–L3, downsampling L1–L3, motion blur L1–L3), the degraded binary mask SHALL contain no voxels set to 1 that were 0 in the original binary mask — i.e., `np.all((degraded_binary - original_binary) <= 0)`.

**Validates: Requirements 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9, 3.10**

---

### Property 7: Degraded labeled mask values are a subset of original seg values

*For any* original seg_data array and degraded binary mask, every non-zero value in `degraded_labeled` SHALL be a value that also exists in `seg_data`, and wherever `degraded_binary == 0`, `degraded_labeled` SHALL be 0.

**Validates: Requirements 3.11**

---

### Property 8: Error metrics satisfy non-negativity and boundedness invariants

*For any* gold measurement and degraded measurement pair:
- `volume_MAE_mm3 >= 0`
- `volume_pct_error >= 0`
- `area_MAE_mm2 >= 0`
- `dice_score` is in `[0.0, 1.0]`
- `clinically_dangerous == True` if and only if `volume_pct_error > 20`

**Validates: Requirements 5.1, 5.2, 5.3, 5.4, 5.5**

---

### Property 9: Dice score is 1 for identical masks

*For any* non-empty binary mask `A`, `dice(A, A) == 1.0` (within floating-point tolerance).

**Validates: Requirements 5.4**

---

## Error Handling

| Situation | Behavior |
|---|---|
| Missing `*_t1ce.nii.gz` or `*_seg.nii.gz` | Skip folder in `discover_patients` |
| `gold_whole_tumor_vol == 0` | Return `None` from `compute_gold_standard`; skip patient in `main` |
| Any exception during per-patient processing | Caught by `try/except` in `main`; print warning; `continue` |
| Output directory does not exist | `os.makedirs(output_dir, exist_ok=True)` before saving plots |
| CSV parent directory does not exist | `os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)` |
| `example_patient` is `None` when plotting | `generate_plots` skips `plot3.png` and logs a warning |

---

## Testing Strategy

This pipeline performs file I/O, numerical transforms, and visualization. Property-based testing is applied to the pure computational core (mask arithmetic, error metrics, degradation invariants). Example-based and integration tests cover I/O and end-to-end behavior.

### Property-Based Tests (using `hypothesis`)

Each property test runs a minimum of 100 iterations. The `hypothesis` library is used for input generation. Each test is tagged with its design property.

| Test | Property | Library |
|---|---|---|
| `test_patient_filter_completeness` | Property 1 | hypothesis |
| `test_patient_count_limit` | Property 2 | hypothesis |
| `test_voxel_volume_scaling` | Property 3 | hypothesis |
| `test_volume_measurement_proportionality` | Property 4 | hypothesis |
| `test_area_measurement_middle_slice` | Property 5 | hypothesis |
| `test_degradation_monotonicity` | Property 6 | hypothesis |
| `test_degraded_labeled_subset` | Property 7 | hypothesis |
| `test_error_metric_invariants` | Property 8 | hypothesis |
| `test_dice_identical_masks` | Property 9 | hypothesis |

### Example-Based / Integration Tests

- Verify that a patient with all-zero seg is skipped (Req 2.9)
- Verify CSV columns are exactly as specified (Req 6.2)
- Verify `try/except` in main does not abort on a deliberately broken patient folder (Req 8.1)

### Unit Tests

- `compute_gold_standard` returns `None` for zero mask
- `simulate_degradations` returns exactly 9 entries per patient
- `compute_errors` `clinically_dangerous` is `True` for pct_error = 20.1, `False` for 19.9
