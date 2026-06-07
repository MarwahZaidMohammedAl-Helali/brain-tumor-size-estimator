"""
app.py - Brain Tumor Size Estimator Dashboard
"""

import os
import sys
import importlib

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Brain Tumor Size Estimator",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_DIR = r"C:\Users\marwa\Pictures\Marwah (1)\Projects\Project 2"
DATASET_DIR = os.path.join(PROJECT_DIR, "Data", "BraTS2021_Training_Data")
OUTPUT_CSV  = os.path.join(PROJECT_DIR, "results.csv")
OUTPUT_DIR  = PROJECT_DIR

if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

# ---------------------------------------------------------------------------
# Load pipeline
# ---------------------------------------------------------------------------
@st.cache_resource
def load_pipeline():
    import analyze as az
    importlib.reload(az)
    return az

# ---------------------------------------------------------------------------
# Get list of valid patients
# ---------------------------------------------------------------------------
@st.cache_data
def list_valid_patients(dataset_dir):
    import glob as _glob
    valid = []
    if not os.path.isdir(dataset_dir):
        return valid
    for entry in sorted(os.listdir(dataset_dir)):
        subdir = os.path.join(dataset_dir, entry)
        if not os.path.isdir(subdir):
            continue
        if (_glob.glob(os.path.join(subdir, "*_t1ce.nii.gz")) and
                _glob.glob(os.path.join(subdir, "*_seg.nii.gz"))):
            valid.append(entry)
    return valid

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.image("https://img.icons8.com/emoji/96/brain-emoji.png", width=60)
st.sidebar.title("Settings")

st.sidebar.markdown("### How many patients?")
max_patients = st.sidebar.slider(
    "Number of patients to analyze",
    min_value=1, max_value=100, value=10, step=1,
    help="More patients = more accurate results but takes longer. Start with 10 to test."
)

st.sidebar.markdown("---")
st.sidebar.markdown("### What type of image problem to test?")
use_erosion      = st.sidebar.checkbox(
    "Grainy / Noisy scan",
    value=True,
    help="Simulates a scan with a lot of noise around the tumor edges."
)
use_downsampling = st.sidebar.checkbox(
    "Blurry / Low resolution scan",
    value=True,
    help="Simulates a scan taken at lower quality — less detail."
)
use_blur         = st.sidebar.checkbox(
    "Patient moved during scan",
    value=True,
    help="Simulates blur caused by the patient moving while being scanned."
)

st.sidebar.markdown("---")
st.sidebar.markdown("### Pick a patient to view up close")

all_patient_ids = list_valid_patients(DATASET_DIR)
if all_patient_ids:
    default_idx = all_patient_ids.index("BraTS2021_00506") if "BraTS2021_00506" in all_patient_ids else 0
    example_id = st.sidebar.selectbox(
        "Choose a patient for the scan view",
        options=all_patient_ids,
        index=default_idx,
        help="This patient's brain scan will be shown in the 'Scan View' tab."
    )
else:
    example_id = "BraTS2021_00506"
    st.sidebar.warning("Could not find the dataset folder.")

st.sidebar.markdown("---")
run_button = st.sidebar.button("Run Analysis", type="primary", use_container_width=True)

# ---------------------------------------------------------------------------
# Page title
# ---------------------------------------------------------------------------
st.title("🧠 Brain Tumor Size Estimator")
st.markdown(
    "This tool checks **how much a bad quality MRI scan affects the measurement of brain tumor size**. "
    "It compares the tumor size from a clean scan vs. a degraded one and tells you how far off the measurement is."
)
st.markdown("---")

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
if "results_df" not in st.session_state:
    st.session_state.results_df = None
if "run_complete" not in st.session_state:
    st.session_state.run_complete = False
if "example_patient" not in st.session_state:
    st.session_state.example_patient = None

# ---------------------------------------------------------------------------
# Run pipeline
# ---------------------------------------------------------------------------
def run_pipeline(max_pts, active_types, ex_id):
    az = load_pipeline()

    progress_bar = st.progress(0, text="Looking for patients...")
    status_box   = st.empty()

    patients = az.discover_patients(DATASET_DIR, max_patients=max_pts)
    if not patients:
        st.error("No patients found. Please check the dataset folder.")
        return None, None

    all_records     = []
    example_patient = None
    n = len(patients)

    for i, patient in enumerate(patients):
        progress_bar.progress(i / n, text=f"Analyzing patient {i+1} of {n}...")
        status_box.info(f"Working on: {patient['patient_id']}  ({i+1}/{n})")
        try:
            import nibabel as nib
            seg_img   = nib.load(patient["seg_path"])
            t1ce_img  = nib.load(patient["t1ce_path"])
            seg_data  = seg_img.get_fdata().astype(np.int16)
            t1ce_data = t1ce_img.get_fdata()
            spacing   = seg_img.header.get_zooms()[:3]

            gold = az.compute_gold_standard(seg_data, t1ce_data, spacing)
            if gold is None:
                continue

            degradations_all = az.simulate_degradations(seg_data, gold["binary_mask"])
            degradations = [
                d for d in degradations_all
                if (d["degradation_type"] == "erosion"      and use_erosion)
                or (d["degradation_type"] == "downsampling" and use_downsampling)
                or (d["degradation_type"] == "motion_blur"  and use_blur)
            ]

            if example_patient is None or patient["patient_id"] == ex_id:
                example_patient = {
                    "patient_id":    patient["patient_id"],
                    "t1ce_data":     t1ce_data,
                    "seg_data":      seg_data,
                    "degradations":  degradations_all,
                    "mid_slice_idx": gold["mid_slice_idx"],
                }

            for deg in degradations:
                meas   = az.measure_degraded(deg["degraded_labeled"], spacing, gold["mid_slice_idx"])
                errors = az.compute_errors(gold, meas, gold["binary_mask"], deg["degraded_binary"])
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
            status_box.warning(f"Skipped {patient['patient_id']}: {e}")
            continue

    progress_bar.progress(1.0, text="Done!")
    status_box.success(f"Finished! Analyzed {n} patients and got {len(all_records)} measurements.")

    columns = [
        "patient_id", "degradation_type", "degradation_level",
        "gold_whole_tumor_vol_mm3", "degraded_whole_tumor_vol_mm3",
        "volume_MAE_mm3", "volume_pct_error",
        "gold_area_mm2", "degraded_area_mm2", "area_MAE_mm2",
        "dice_score", "clinically_dangerous",
    ]
    df = pd.DataFrame(all_records, columns=columns)
    df.to_csv(OUTPUT_CSV, index=False)
    return df, example_patient


# Trigger run
if run_button:
    active_types = []
    if use_erosion:      active_types.append("erosion")
    if use_downsampling: active_types.append("downsampling")
    if use_blur:         active_types.append("motion_blur")

    if not active_types:
        st.sidebar.error("Please select at least one image problem type.")
    else:
        with st.spinner("Running analysis..."):
            df, ep = run_pipeline(max_patients, active_types, example_id)
        if df is not None and not df.empty:
            st.session_state.results_df = df
            st.session_state.example_patient = ep
            st.session_state.run_complete = True

# Auto-load previous results
if st.session_state.results_df is None and os.path.exists(OUTPUT_CSV):
    try:
        st.session_state.results_df = pd.read_csv(OUTPUT_CSV)
        st.session_state.run_complete = True
        st.info("Previous results loaded automatically. Press 'Run Analysis' to start fresh.")
    except Exception:
        pass

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
if st.session_state.run_complete and st.session_state.results_df is not None:
    df = st.session_state.results_df
    ep = st.session_state.example_patient

    tab_summary, tab_results, tab_plots, tab_scan = st.tabs([
        "Overview", "Full Data Table", "Charts", "Scan View"
    ])

    # friendly display names
    type_display = {
        "erosion":      "Grainy Scan",
        "downsampling": "Blurry Scan",
        "motion_blur":  "Motion Blur",
    }

    # -----------------------------------------------------------------------
    # TAB 1 — Overview
    # -----------------------------------------------------------------------
    with tab_summary:
        st.subheader("Quick Overview")
        st.markdown("Here is a summary of what was found across all patients and scan quality problems tested.")

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Patients Analyzed",     df["patient_id"].nunique())
        col2.metric("Total Measurements",    len(df))
        col3.metric(
            "Risky Measurements",
            f"{df['clinically_dangerous'].sum()} ({df['clinically_dangerous'].mean()*100:.1f}%)",
            help="Measurements where the error was large enough to potentially mislead a doctor (over 20% off)."
        )
        col4.metric(
            "Average Overlap Score",
            f"{df['dice_score'].mean():.3f}",
            help="How well the degraded tumor outline matches the original. 1.0 = perfect match, 0 = no match."
        )

        st.markdown("---")
        st.subheader("Average error and overlap, by scan problem and severity")
        st.caption(
            "Level 1 = mild problem, Level 2 = moderate, Level 3 = severe. "
            "Volume error % = how far off the tumor size measurement was."
        )

        summary_df = (
            df.groupby(["degradation_type", "degradation_level"])[["volume_pct_error", "dice_score"]]
            .agg(["mean", "std"])
            .round(3)
        )
        summary_df.index = summary_df.index.map(
            lambda x: (type_display.get(x[0], x[0]), f"Level {x[1]}")
        )
        summary_df.index.names = ["Scan Problem", "Severity"]
        summary_df.columns = ["Avg Size Error (%)", "Std Error (%)", "Avg Overlap Score", "Std Overlap"]
        st.dataframe(summary_df, use_container_width=True)

        st.markdown("---")
        st.subheader("How often did each scan problem cause a risky measurement?")
        st.caption("A measurement is considered risky if it is more than 20% off — this is the medical threshold used by doctors.")

        danger_df = (
            df.assign(label=df.apply(
                lambda r: f"{type_display.get(r['degradation_type'], r['degradation_type'])} - Level {r['degradation_level']}",
                axis=1
            ))
            .groupby("label")["clinically_dangerous"]
            .mean()
            .mul(100)
            .reset_index()
            .rename(columns={"label": "Scan Problem", "clinically_dangerous": "% of Patients at Risk"})
        )
        st.bar_chart(danger_df.set_index("Scan Problem"))

    # -----------------------------------------------------------------------
    # TAB 2 — Full Data Table
    # -----------------------------------------------------------------------
    with tab_results:
        st.subheader("Full Data Table")
        st.markdown("Every row is one patient + one scan problem combination. You can filter and download.")

        col_f1, col_f2, col_f3 = st.columns(3)
        with col_f1:
            type_filter = st.multiselect(
                "Filter by scan problem",
                options=df["degradation_type"].unique().tolist(),
                default=df["degradation_type"].unique().tolist(),
                format_func=lambda x: type_display.get(x, x),
            )
        with col_f2:
            level_filter = st.multiselect(
                "Filter by severity level",
                options=sorted(df["degradation_level"].unique().tolist()),
                default=sorted(df["degradation_level"].unique().tolist()),
                format_func=lambda x: f"Level {x}",
            )
        with col_f3:
            danger_filter = st.selectbox(
                "Show only risky measurements?",
                options=["Show all", "Risky only (>20% error)", "Safe only"],
                index=0,
            )

        filtered = df[
            df["degradation_type"].isin(type_filter) &
            df["degradation_level"].isin(level_filter)
        ].copy()
        if danger_filter == "Risky only (>20% error)":
            filtered = filtered[filtered["clinically_dangerous"] == True]
        elif danger_filter == "Safe only":
            filtered = filtered[filtered["clinically_dangerous"] == False]

        # Rename columns to friendly names for display
        display_cols = {
            "patient_id":                   "Patient",
            "degradation_type":             "Scan Problem",
            "degradation_level":            "Severity",
            "gold_whole_tumor_vol_mm3":     "Real Tumor Size (mm3)",
            "degraded_whole_tumor_vol_mm3": "Measured Size After Degradation (mm3)",
            "volume_MAE_mm3":               "Size Error (mm3)",
            "volume_pct_error":             "Size Error (%)",
            "gold_area_mm2":                "Real Tumor Area (mm2)",
            "degraded_area_mm2":            "Measured Area After Degradation (mm2)",
            "area_MAE_mm2":                 "Area Error (mm2)",
            "dice_score":                   "Overlap Score",
            "clinically_dangerous":         "Risky Measurement?",
        }
        display_df = filtered.rename(columns=display_cols)
        display_df["Scan Problem"] = display_df["Scan Problem"].map(type_display).fillna(display_df["Scan Problem"])
        display_df["Severity"] = display_df["Severity"].apply(lambda x: f"Level {x}")

        st.dataframe(
            display_df.style.format({
                "Real Tumor Size (mm3)":                  "{:.1f}",
                "Measured Size After Degradation (mm3)":  "{:.1f}",
                "Size Error (mm3)":                       "{:.1f}",
                "Size Error (%)":                         "{:.2f}",
                "Real Tumor Area (mm2)":                  "{:.1f}",
                "Measured Area After Degradation (mm2)":  "{:.1f}",
                "Area Error (mm2)":                       "{:.1f}",
                "Overlap Score":                          "{:.4f}",
            }),
            use_container_width=True,
            height=500,
        )

        csv_bytes = filtered.to_csv(index=False).encode()
        st.download_button(
            label="Download this table as CSV",
            data=csv_bytes,
            file_name="results_filtered.csv",
            mime="text/csv",
        )

    # -----------------------------------------------------------------------
    # TAB 3 — Charts
    # -----------------------------------------------------------------------
    with tab_plots:
        st.subheader("Charts")

        df_plot = df.copy()
        df_plot["label"] = df_plot.apply(
            lambda row: f"{type_display.get(row['degradation_type'], row['degradation_type'])} - L{row['degradation_level']}",
            axis=1,
        )
        label_order = [
            f"{d} - L{l}"
            for d in ["Grainy Scan", "Blurry Scan", "Motion Blur"]
            for l in [1, 2, 3]
            if f"{d} - L{l}" in df_plot["label"].unique()
        ]

        import seaborn as sns
        sns.set_style("whitegrid")

        # Chart 1
        st.markdown("#### How does size error grow as the scan gets worse?")
        st.caption(
            "Each line is a different type of scan problem. "
            "The red dashed line is the danger threshold — above it, the error is large enough to mislead a doctor."
        )
        grouped = (
            df_plot.groupby(["degradation_type", "degradation_level"])["volume_pct_error"]
            .mean().reset_index()
        )
        fig1, ax1 = plt.subplots(figsize=(10, 5))
        for deg_type, display_name in type_display.items():
            sub = grouped[grouped["degradation_type"] == deg_type].sort_values("degradation_level")
            if not sub.empty:
                ax1.plot(sub["degradation_level"], sub["volume_pct_error"], marker="o", label=display_name)
        ax1.axhline(y=20, color="red", linestyle="--", label="Danger Threshold (20% error)")
        ax1.set_xticks([1, 2, 3])
        ax1.set_xticklabels(["Level 1\n(Mild)", "Level 2\n(Moderate)", "Level 3\n(Severe)"])
        ax1.set_xlabel("Severity of Scan Problem")
        ax1.set_ylabel("Average Size Error (%)")
        ax1.set_title("How much does tumor size measurement error grow as scan quality gets worse?")
        ax1.legend()
        fig1.tight_layout()
        st.pyplot(fig1)
        plt.close(fig1)

        st.markdown("---")

        # Chart 2
        st.markdown("#### How spread out are the errors across all patients?")
        st.caption(
            "Each box shows the range of errors for that scan problem. "
            "Dots are individual patients. Above the red line = dangerous error."
        )
        fig2, ax2 = plt.subplots(figsize=(12, 6))
        sns.boxplot(data=df_plot, x="label", y="volume_pct_error",
                    order=label_order, ax=ax2, hue="label",
                    palette="Set2", legend=False)
        sns.stripplot(data=df_plot, x="label", y="volume_pct_error",
                      order=label_order, ax=ax2, alpha=0.3, color="black", jitter=True)
        ax2.axhline(y=20, color="red", linestyle="--", label="Danger Threshold (20% error)")
        ax2.set_xlabel("Scan Problem and Severity")
        ax2.set_ylabel("Size Error (%)")
        ax2.set_title("Spread of tumor size errors across all patients")
        ax2.legend()
        plt.xticks(rotation=30, ha="right")
        fig2.tight_layout()
        st.pyplot(fig2)
        plt.close(fig2)

        st.markdown("---")

        # Chart 3
        st.markdown("#### For each scan problem, what % of patients had a risky measurement?")
        st.caption("A risky measurement is one that is more than 20% off from the real tumor size.")
        pct_d = (
            df_plot.groupby("label")["clinically_dangerous"].mean() * 100
        ).reset_index()
        pct_d.columns = ["label", "pct_dangerous"]
        pct_d["label"] = pd.Categorical(pct_d["label"], categories=label_order, ordered=True)
        pct_d = pct_d.sort_values("label")

        fig4, ax4 = plt.subplots(figsize=(10, 5))
        ax4.bar(pct_d["label"].astype(str), pct_d["pct_dangerous"],
                color=sns.color_palette("Set2", len(pct_d)))
        ax4.set_xlabel("Scan Problem and Severity")
        ax4.set_ylabel("% of Patients with Risky Measurement")
        ax4.set_title("What % of patients had a measurement error over 20%?")
        plt.xticks(rotation=30, ha="right")
        fig4.tight_layout()
        st.pyplot(fig4)
        plt.close(fig4)

        st.markdown("---")
        st.markdown("#### Download charts")
        col_d1, col_d2, col_d3, col_d4 = st.columns(4)
        for col, fname, label in zip(
            [col_d1, col_d2, col_d3, col_d4],
            ["plot1.png", "plot2.png", "plot3.png", "plot4.png"],
            ["Line Chart", "Box Plot", "Scan Comparison", "Bar Chart"],
        ):
            fpath = os.path.join(OUTPUT_DIR, fname)
            if os.path.exists(fpath):
                with open(fpath, "rb") as f:
                    col.download_button(f"Download {label}", f.read(), file_name=fname, mime="image/png")
            else:
                col.caption(f"{label} not saved yet — run the full pipeline first.")

    # -----------------------------------------------------------------------
    # TAB 4 — Scan View
    # -----------------------------------------------------------------------
    with tab_scan:
        st.subheader("Side-by-side Scan View")
        st.markdown(
            "This shows the actual brain scan for one patient. "
            "The colored area is the tumor boundary — **green** is the original, **red** is what it looks like after the scan quality drops."
        )

        if ep is None:
            st.warning("No scan data available. Please run the analysis first.")
        else:
            st.markdown(f"**Showing patient:** `{ep['patient_id']}`")

            t1ce_data    = ep["t1ce_data"]
            seg_data     = ep["seg_data"]
            degradations = ep["degradations"]
            mid_idx      = ep["mid_slice_idx"]

            max_z     = t1ce_data.shape[2] - 1
            slice_idx = st.slider(
                "Move through the brain slices",
                min_value=0, max_value=max_z, value=mid_idx,
                help="Slide to scroll through the brain from bottom to top."
            )

            t1ce_slice = t1ce_data[:, :, slice_idx]
            t1ce_min, t1ce_max = t1ce_slice.min(), t1ce_slice.max()
            t1ce_norm  = (t1ce_slice - t1ce_min) / (t1ce_max - t1ce_min + 1e-8)

            def find_deg(dlist, dtype, dlevel):
                for d in dlist:
                    if d["degradation_type"] == dtype and d["degradation_level"] == dlevel:
                        return d["degraded_binary"][:, :, slice_idx]
                return None

            original_mask = (seg_data[:, :, slice_idx] > 0).astype(np.uint8)
            noise_l2      = find_deg(degradations, "erosion",      2)
            down_l2       = find_deg(degradations, "downsampling",  2)
            blur_l2       = find_deg(degradations, "motion_blur",   2)

            def make_overlay(t1ce_norm, mask, color):
                fig, ax = plt.subplots(figsize=(5, 5))
                ax.imshow(t1ce_norm.T, cmap="gray", origin="lower")
                if mask is not None:
                    r, g, b = {"green": (0., 1., 0.), "red": (1., 0., 0.)}[color]
                    h, w = mask.shape
                    rgba = np.zeros((h, w, 4), dtype=np.float32)
                    rgba[..., 0] = r
                    rgba[..., 1] = g
                    rgba[..., 2] = b
                    rgba[..., 3] = mask.astype(np.float32) * 0.4
                    ax.imshow(rgba.transpose(1, 0, 2), origin="lower")
                ax.axis("off")
                fig.tight_layout(pad=0)
                return fig

            col_a, col_b, col_c, col_d = st.columns(4)

            with col_a:
                st.caption("Original (clean scan)")
                fig = make_overlay(t1ce_norm, original_mask, "green")
                st.pyplot(fig); plt.close(fig)

            with col_b:
                st.caption("After: Grainy Scan (moderate)")
                fig = make_overlay(t1ce_norm, noise_l2, "red")
                st.pyplot(fig); plt.close(fig)

            with col_c:
                st.caption("After: Blurry Scan (moderate)")
                fig = make_overlay(t1ce_norm, down_l2, "red")
                st.pyplot(fig); plt.close(fig)

            with col_d:
                st.caption("After: Motion Blur (moderate)")
                fig = make_overlay(t1ce_norm, blur_l2, "red")
                st.pyplot(fig); plt.close(fig)

            st.markdown(
                "**Green** = original tumor boundary from the clean scan  "
                "&nbsp;&nbsp;  **Red** = boundary after scan quality drops"
            )

else:
    # Welcome screen
    st.info("Use the settings on the left, then click **Run Analysis** to get started.")

    st.markdown("### What does this tool do?")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("#### The Problem")
        st.markdown(
            "When doctors measure a brain tumor from an MRI scan, "
            "they need the measurement to be accurate. "
            "But what if the scan quality is poor — grainy, blurry, or shaky?"
        )

    with col2:
        st.markdown("#### What This Tool Tests")
        st.markdown(
            "It takes real brain tumor scans, makes the quality worse on purpose "
            "(3 ways, 3 levels each), re-measures the tumor, "
            "and checks how much the measurement changed."
        )

    with col3:
        st.markdown("#### What You Get")
        st.markdown(
            "Charts and tables showing which scan problems cause the biggest errors, "
            "and how often the error is large enough to matter clinically (over 20% off)."
        )

    st.markdown("---")
    st.markdown("### How to use it")
    st.markdown(
        "1. **Choose how many patients** to analyze (start small to test quickly)\n"
        "2. **Select which scan problems** to simulate\n"
        "3. **Pick a patient** to view their brain scan up close\n"
        "4. Click **Run Analysis** and wait for results\n"
        "5. Explore the **Overview**, **Charts**, and **Scan View** tabs"
    )
