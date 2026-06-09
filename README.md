# Brain Tumor Size Estimator

> **Research project:** How much does a bad-quality MRI scan affect the measurement of brain tumor size?

## What This Project Does

When doctors treat brain cancer, they track tumor size over time to see if the treatment is working. But what if the MRI scan is low quality, grainy, blurry, or shaky from patient movement? How much does that affect the measurement?

This project answers that question by:

1. Taking real brain tumor scans from the **BraTS 2021** dataset
2. Artificially making the scan quality worse in 3 different ways
3. Re-measuring the tumor size from the degraded scan
4. Comparing that to the original measurement to see how far off it is
5. Flagging any error over **20%** as clinically risky (based on the RANO medical guidelines)

## The Three Types of Scan Problems Tested

| Problem | What it simulates | Severity levels |
|---|---|---|
| **Grainy / Noisy scan** | High noise around tumor edges | Mild, Moderate, Severe |
| **Blurry / Low resolution** | Scan taken at lower quality | Mild, Moderate, Severe |
| **Patient movement (motion blur)** | Patient moved during the scan | Mild, Moderate, Severe |

## Project Structure

```
Project 2/
├── analyze.py          # Main research script, runs the full analysis pipeline
├── app.py              # Streamlit web dashboard, interactive UI
├── .gitignore          # Excludes large data files and generated outputs
├── README.md           # This file
└── .kiro/
    └── specs/
        └── brain-tumor-size-estimator/
            ├── requirements.md   # Full project requirements
            ├── design.md         # Technical design document
            └── tasks.md          # Implementation task list
```

> **Note:** The `Data/` folder (BraTS 2021 dataset) and generated output files (`results.csv`, `plot*.png`) are excluded from the repository because of their size.

## How to Run

### 1. Install dependencies

```bash
pip install nibabel numpy scipy scikit-image matplotlib seaborn pandas streamlit
```

### 2. Make sure the data is in place

Put the BraTS 2021 Training Data here:
```
C:\Users\marwa\Pictures\Marwah (1)\Projects\Project 2\Data\BraTS2021_Training_Data\
```

Each patient folder should contain:
```
BraTS2021_XXXXX\
    BraTS2021_XXXXX_t1ce.nii.gz
    BraTS2021_XXXXX_seg.nii.gz
```

### 3. Option A: Run the web dashboard (recommended)

```bash
py -m streamlit run "C:\Users\marwa\Pictures\Marwah (1)\Projects\Project 2\app.py"
```

This opens a browser interface where you can:
- Choose how many patients to analyze
- Pick which scan problems to test
- View results as charts and tables
- Compare original vs degraded scans side by side

### 4. Option B: Run the script directly

```bash
py "C:\Users\marwa\Pictures\Marwah (1)\Projects\Project 2\analyze.py"
```

This processes 100 patients and saves:
- `results.csv` with all measurements and errors
- `plot1.png` to `plot4.png` with publication-quality charts

## Output Files

| File | Description |
|---|---|
| `results.csv` | One row per patient and scan problem combination. Contains tumor size, error, overlap score, and danger flag. |
| `plot1.png` | Line chart showing how error grows as scan quality gets worse |
| `plot2.png` | Box plot showing the spread of errors across all patients |
| `plot3.png` | Side-by-side scan view of original vs degraded tumor boundaries |
| `plot4.png` | Bar chart showing the percentage of patients with dangerous measurement error per scan problem |

## CSV Columns Explained

| Column | What it means |
|---|---|
| `patient_id` | Patient folder name (e.g. BraTS2021_00001) |
| `degradation_type` | Type of scan problem (erosion / downsampling / motion_blur) |
| `degradation_level` | Severity: 1 = mild, 2 = moderate, 3 = severe |
| `gold_whole_tumor_vol_mm3` | Real tumor volume from the original clean scan (mm3) |
| `degraded_whole_tumor_vol_mm3` | Tumor volume measured after degradation (mm3) |
| `volume_MAE_mm3` | Absolute difference between the two volumes (mm3) |
| `volume_pct_error` | How far off the measurement is, as a percentage |
| `gold_area_mm2` | Real tumor area on the middle brain slice (mm2) |
| `degraded_area_mm2` | Tumor area measured after degradation (mm2) |
| `area_MAE_mm2` | Absolute difference in area (mm2) |
| `dice_score` | Overlap between original and degraded tumor outline (0 to 1, higher is better) |
| `clinically_dangerous` | True if error is above 20% (the RANO medical threshold) |

## Dataset

**BraTS 2021 Training Dataset**
- Approximately 1,251 patient folders
- Multi-modal MRI scans (T1, T1ce, T2, FLAIR) with expert segmentation masks
- Mask labels: `0` = background, `1` = necrotic core, `2` = edema, `4` = enhancing tumor

Dataset available at: [https://www.synapse.org/brats2021](https://www.synapse.org/brats2021)

## Technical Details

- **Language:** Python only
- **Libraries:** `nibabel`, `numpy`, `scipy`, `scikit-image`, `matplotlib`, `seaborn`, `pandas`, `streamlit`
- **No deep learning**, pure image processing and statistics
- **CPU only**, runs on a standard Windows laptop
- Processes up to 100 patients (configurable)
- 9 degradation variants per patient (3 types x 3 severity levels)

## Key Finding (RANO Threshold)

The **RANO criteria** (Response Assessment in Neuro-Oncology) define a **20% change in tumor volume** as clinically significant. Any measurement error above this threshold could lead a doctor to incorrectly classify a tumor as growing or shrinking, affecting treatment decisions.

This project systematically maps which types and severities of MRI degradation push measurements past that threshold.

## Repository

[github.com/MarwahZaidMohammedAl-Helali/brain-tumor-size-estimator](https://github.com/MarwahZaidMohammedAl-Helali/brain-tumor-size-estimator)
