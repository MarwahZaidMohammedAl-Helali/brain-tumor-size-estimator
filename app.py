"""
app.py — Streamlit Dashboard for MRI Quality Degradation Research
Wraps the analyze.py pipeline with an interactive web interface.
"""

import os
import io
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
PROJECT_DIR  = r"C:\Users\marwa\Pictures\Marwah (1)\Projects\Project 2"
DATASET_DIR  = os.path.join(PROJECT_DIR, "Data", "BraTS2021_Training_Data")
OUTPUT_CSV   = os.path.join(PROJECT_DIR, "results.csv")
OUTPUT_DIR   = PROJECT_DIR

# Add project dir to path so we can import analyze.py
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

# ---------------------------------------------------------------------------
# Import pipeline functions from analyze.py
# ---------------------------------------------------------------------------
@st.cache_resource
def load_pipeline():
    import analyze as az
    importlib.reload(az)
    return az

# ---------------------------------------------------------------------------
# Sidebar — configuration
# ---------------------------------------------------------------------------
st.sidebar.image("https://img.icons8.com/emoji/96/brain-emoji.png", width=60)
st.sidebar.title("🧠 Pipeline Settings")

max_patients = st.sidebar.slider(
    "Number of patients to process",
    min_value=1, max_value=100, value=10, step=1,
    help="Processing all 100 patients takes ~10–30 minutes on a CPU laptop."
)

st.sidebar.markdown("---")
st.sidebar.subheader("Degradation Types")
use_erosion     = st.sidebar.checkbox("Noise Erosion",   value=True)
use_downsampling = st.sidebar.checkbox("Downsampling",   value=True)
use_blur        = st.sidebar.checkbox("Motion Blur",     value=True)

st.sidebar.markdown("---")
st.sidebar.subheader("Example Patient")
example_id = st.sidebar.text_input(
    "Patient ID for Plot 3",
    value="BraTS2021_00506",
    help="Falls back to the first valid patient if not found."
)

st.sidebar.markdown("---")
run_button = st.sidebar.button("▶  Run Pipeline", type="primary", use_container_width=True)

# ---------------------------------------------------------------------------
# Title
# ---------------------------------------------------------------------------
st.title("🧠 Brain Tumor Size Estimator")
st.markdown(
    "**Research tool** — quantifies how MRI image quality degradation affects "
    "brain tumor size measurement accuracy using the BraTS 2021 dataset."
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
def run_pipeline(max_pts: int, active_types: list, ex_id: str):
    az = load_pipeline()

    progress_bar = st.progress(0, text="Discovering patients…")
    status_box   = st.empty()

    patients = az.discover_patients(DATASET_DIR, max_patients=max_pts)
    if not patients:
        st.error("No valid patients found. Check the dataset path.")
        return None, None

    all_records   = []
    example_patient = None
    n = len(patients)

    for i, patient in enumerate(patients):
        progress_bar.progress((i) / n, text=f"Processing {patient['patient_id']} ({i+1}/{n})…")
        status_box.info(f"⏳ Patient {i+1}/{n}: `{patient['patient_id']}`")
        try:
            import nibabel as nib
            seg_img  = nib.load(patient["seg_path"])
            t1ce_img = nib.load(patient["t1ce_path"])
            seg_data  = seg_img.get_fdata().astype(np.int16)
            t1ce_data = t1ce_img.get_fdata()
            spacing   = seg_img.header.get_zooms()[:3]

            gold = az.compute_gold_standard(seg_data, t1ce_data, spacing)
            if gold is None:
                continue

            degradations_all = az.simulate_degradations(seg_data, gold["binary_mask"])
            # Filter by selected types
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
                    "degradations":  degradations_all,   # always pass all 9 for Plot 3
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
            status_box.warning(f"⚠️ Skipped `{patient['patient_id']}`: {e}")
            continue

    progress_bar.progress(1.0, text="Done!")
    status_box.success(f"✅ Processed {n} patients → {len(all_records)} records.")

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
        st.sidebar.error("Select at least one degradation type.")
    else:
        with st.spinner("Running pipeline…"):
            df, ep = run_pipeline(max_patients, active_types, example_id)
        if df is not None and not df.empty:
            st.session_state.results_df = df
            st.session_state.example_patient = ep
            st.session_state.run_complete = True

# Load existing results if available and not yet run
if st.session_state.results_df is None and os.path.exists(OUTPUT_CSV):
    try:
        st.session_state.results_df = pd.read_csv(OUTPUT_CSV)
        st.session_state.run_complete = True
        st.info("📂 Loaded existing `results.csv`. Hit **Run Pipeline** to regenerate.")
    except Exception:
        pass

# ---------------------------------------------------------------------------
# Main content tabs
# ---------------------------------------------------------------------------
if st.session_state.run_complete and st.session_state.results_df is not None:
    df = st.session_state.results_df
    ep = st.session_state.example_patient

    tab_summary, tab_results, tab_plots, tab_plot3 = st.tabs([
        "📊 Summary", "📋 Results Table", "📈 Plots", "🖼️ Segmentation View"
    ])

    # -----------------------------------------------------------------------
    # TAB 1 — Summary
    # -----------------------------------------------------------------------
    with tab_summary:
        st.subheader("Pipeline Summary")

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Patients Processed", df["patient_id"].nunique())
        col2.metric("Total Records",       len(df))
        col3.metric("Clinically Dangerous",
                    f"{df['clinically_dangerous'].sum()} "
                    f"({df['clinically_dangerous'].mean()*100:.1f}%)")
        col4.metric("Mean Dice Score",     f"{df['dice_score'].mean():.3f}")

        st.markdown("---")
        st.subheader("Mean ± Std grouped by Degradation Type & Level")

        type_display = {"erosion": "Noise", "downsampling": "Downsampling", "motion_blur": "Motion Blur"}
        summary_df = (
            df.groupby(["degradation_type", "degradation_level"])[["volume_pct_error", "dice_score"]]
            .agg(["mean", "std"])
            .round(3)
        )
        summary_df.index = summary_df.index.map(
            lambda x: (type_display.get(x[0], x[0]), x[1])
        )
        summary_df.index.names = ["Degradation Type", "Level"]
        st.dataframe(summary_df, use_container_width=True)

        st.markdown("---")
        st.subheader("Clinical Danger Rate by Degradation")
        danger_df = (
            df.assign(label=df.apply(
                lambda r: f"{type_display.get(r['degradation_type'], r['degradation_type'])}-L{r['degradation_level']}",
                axis=1
            ))
            .groupby("label")["clinically_dangerous"]
            .mean()
            .mul(100)
            .reset_index()
            .rename(columns={"label": "Degradation", "clinically_dangerous": "% Dangerous"})
        )
        st.bar_chart(danger_df.set_index("Degradation"))

    # -----------------------------------------------------------------------
    # TAB 2 — Results Table
    # -----------------------------------------------------------------------
    with tab_results:
        st.subheader("Full Results Table")

        # Filters
        col_f1, col_f2, col_f3 = st.columns(3)
        with col_f1:
            type_filter = st.multiselect(
                "Degradation Type",
                options=df["degradation_type"].unique().tolist(),
                default=df["degradation_type"].unique().tolist(),
            )
        with col_f2:
            level_filter = st.multiselect(
                "Degradation Level",
                options=sorted(df["degradation_level"].unique().tolist()),
                default=sorted(df["degradation_level"].unique().tolist()),
            )
        with col_f3:
            danger_filter = st.selectbox(
                "Clinically Dangerous",
                options=["All", "Yes", "No"],
                index=0,
            )

        filtered = df[
            df["degradation_type"].isin(type_filter) &
            df["degradation_level"].isin(level_filter)
        ]
        if danger_filter == "Yes":
            filtered = filtered[filtered["clinically_dangerous"] == True]
        elif danger_filter == "No":
            filtered = filtered[filtered["clinically_dangerous"] == False]

        st.dataframe(
            filtered.style.format({
                "gold_whole_tumor_vol_mm3":     "{:.1f}",
                "degraded_whole_tumor_vol_mm3": "{:.1f}",
                "volume_MAE_mm3":               "{:.1f}",
                "volume_pct_error":             "{:.2f}",
                "gold_area_mm2":                "{:.1f}",
                "degraded_area_mm2":            "{:.1f}",
                "area_MAE_mm2":                 "{:.1f}",
                "dice_score":                   "{:.4f}",
            }).apply(
                lambda col: ["background-color: #ffcccc" if v else "" for v in df["clinically_dangerous"]]
                if col.name == "clinically_dangerous" else [""] * len(col),
                axis=0,
            ),
            use_container_width=True,
            height=500,
        )

        # Download button
        csv_bytes = filtered.to_csv(index=False).encode()
        st.download_button(
            label="⬇️ Download filtered CSV",
            data=csv_bytes,
            file_name="results_filtered.csv",
            mime="text/csv",
        )

    # -----------------------------------------------------------------------
    # TAB 3 — Plots
    # -----------------------------------------------------------------------
    with tab_plots:
        st.subheader("Research Plots")

        type_display_map = {
            "erosion":      "Noise",
            "downsampling": "Downsampling",
            "motion_blur":  "Motion Blur",
        }

        df_plot = df.copy()
        df_plot["label"] = df_plot.apply(
            lambda row: f"{type_display_map.get(row['degradation_type'], row['degradation_type'])}-L{row['degradation_level']}",
            axis=1,
        )
        label_order = [
            f"{d}-L{l}"
            for d in ["Noise", "Downsampling", "Motion Blur"]
            for l in [1, 2, 3]
            if f"{d}-L{l}" in df_plot["label"].unique()
        ]

        import seaborn as sns
        sns.set_style("whitegrid")

        # --- Plot 1: Line chart ---
        st.markdown("#### Plot 1 — Volume Error vs Degradation Level")
        grouped = (
            df_plot.groupby(["degradation_type", "degradation_level"])["volume_pct_error"]
            .mean().reset_index()
        )
        fig1, ax1 = plt.subplots(figsize=(10, 5))
        for deg_type, display_name in type_display_map.items():
            sub = grouped[grouped["degradation_type"] == deg_type].sort_values("degradation_level")
            if not sub.empty:
                ax1.plot(sub["degradation_level"], sub["volume_pct_error"], marker="o", label=display_name)
        ax1.axhline(y=20, color="red", linestyle="--", label="Clinical Danger Threshold (RANO)")
        ax1.set_xticks([1, 2, 3])
        ax1.set_xlabel("Degradation Level")
        ax1.set_ylabel("Mean Volume % Error (%)")
        ax1.set_title("Tumor Volume Measurement Error vs. MRI Degradation Level")
        ax1.legend()
        fig1.tight_layout()
        st.pyplot(fig1)
        plt.close(fig1)

        st.markdown("---")

        # --- Plot 2: Box + strip ---
        st.markdown("#### Plot 2 — Distribution of Volume Error by Degradation")
        fig2, ax2 = plt.subplots(figsize=(12, 6))
        sns.boxplot(data=df_plot, x="label", y="volume_pct_error",
                    order=label_order, ax=ax2, hue="label",
                    palette="Set2", legend=False)
        sns.stripplot(data=df_plot, x="label", y="volume_pct_error",
                      order=label_order, ax=ax2, alpha=0.3, color="black", jitter=True)
        ax2.axhline(y=20, color="red", linestyle="--", label="Clinical Danger Threshold (RANO)")
        ax2.set_xlabel("Degradation Type and Level")
        ax2.set_ylabel("Volume % Error (%)")
        ax2.set_title("Distribution of Tumor Size Measurement Error by Degradation")
        ax2.legend()
        plt.xticks(rotation=45, ha="right")
        fig2.tight_layout()
        st.pyplot(fig2)
        plt.close(fig2)

        st.markdown("---")

        # --- Plot 4: Bar chart ---
        st.markdown("#### Plot 4 — % Patients Exceeding 20% Volume Error (RANO Threshold)")
        pct_d = (
            df_plot.groupby("label")["clinically_dangerous"].mean() * 100
        ).reset_index()
        pct_d.columns = ["label", "pct_dangerous"]
        pct_d["label"] = pd.Categorical(pct_d["label"], categories=label_order, ordered=True)
        pct_d = pct_d.sort_values("label")

        fig4, ax4 = plt.subplots(figsize=(10, 5))
        ax4.bar(pct_d["label"].astype(str), pct_d["pct_dangerous"],
                color=sns.color_palette("Set2", len(pct_d)))
        ax4.set_xlabel("Degradation Type and Level")
        ax4.set_ylabel("% Patients Exceeding 20% Volume Error")
        ax4.set_title("Percentage of Patients Exceeding 20% Volume Error Threshold (RANO Clinical Limit)")
        plt.xticks(rotation=45, ha="right")
        fig4.tight_layout()
        st.pyplot(fig4)
        plt.close(fig4)

        # Download all saved PNGs if they exist
        st.markdown("---")
        st.markdown("#### Download Saved Plots")
        col_d1, col_d2, col_d3, col_d4 = st.columns(4)
        for col, fname in zip([col_d1, col_d2, col_d3, col_d4],
                               ["plot1.png", "plot2.png", "plot3.png", "plot4.png"]):
            fpath = os.path.join(OUTPUT_DIR, fname)
            if os.path.exists(fpath):
                with open(fpath, "rb") as f:
                    col.download_button(f"⬇️ {fname}", f.read(), file_name=fname, mime="image/png")
            else:
                col.caption(f"{fname} not saved yet")

    # -----------------------------------------------------------------------
    # TAB 4 — Segmentation View (Plot 3)
    # -----------------------------------------------------------------------
    with tab_plot3:
        st.subheader("Segmentation Boundary Comparison (Example Patient)")

        if ep is None:
            st.warning("Example patient data not available. Re-run the pipeline.")
        else:
            st.markdown(f"**Patient:** `{ep['patient_id']}`")

            t1ce_data    = ep["t1ce_data"]
            seg_data     = ep["seg_data"]
            degradations = ep["degradations"]
            mid_idx      = ep["mid_slice_idx"]

            # Slice selector
            max_z = t1ce_data.shape[2] - 1
            slice_idx = st.slider("Axial slice index", 0, max_z, mid_idx)

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
                """Return a figure with t1ce + colored mask overlay."""
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
                st.caption("Original Mask")
                fig = make_overlay(t1ce_norm, original_mask, "green")
                st.pyplot(fig); plt.close(fig)

            with col_b:
                st.caption("Noise L2 (Erosion)")
                fig = make_overlay(t1ce_norm, noise_l2, "red")
                st.pyplot(fig); plt.close(fig)

            with col_c:
                st.caption("Downsampling L2")
                fig = make_overlay(t1ce_norm, down_l2, "red")
                st.pyplot(fig); plt.close(fig)

            with col_d:
                st.caption("Motion Blur L2")
                fig = make_overlay(t1ce_norm, blur_l2, "red")
                st.pyplot(fig); plt.close(fig)

            st.markdown(
                "**Green** = original expert mask &nbsp;&nbsp; **Red** = degraded mask"
            )

else:
    # No results yet
    st.info(
        "👈 Configure the pipeline in the sidebar and click **▶ Run Pipeline** to get started.\n\n"
        "If you already ran `analyze.py` before, existing `results.csv` will be loaded automatically."
    )
    st.markdown("""
    ### What this tool does
    | Step | Description |
    |------|-------------|
    | 1 | Discovers up to 100 valid BraTS 2021 patient folders |
    | 2 | Computes gold-standard tumor volumes and areas from original seg masks |
    | 3 | Simulates 9 degradation variants (erosion, downsampling, motion blur × 3 levels) |
    | 4 | Re-measures tumor size from each degraded mask |
    | 5 | Computes volume MAE, % error, Dice score, and RANO clinical danger flag |
    | 6 | Saves results to `results.csv` |
    | 7 | Generates 4 publication-quality plots |
    """)
