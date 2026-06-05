from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def detect_column(df, candidates, name):
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(f"Could not find {name}. Tried: {candidates}. Available columns: {df.columns.tolist()}")


def normalize_feature_name(x):
    x = str(x).strip()
    feature_map = {
        "egemaps": "eGeMAPS",
        "whisper": "Whisper-small",
        "whisper_small": "Whisper-small",
        "xlsr": "XLSR-53",
        "xlsr_53": "XLSR-53",
        "whispa_small": "WhiSPA-Small",
        "whispa_tiny": "WhiSPA-Tiny",
        "egemaps_whisper": "eGeMAPS + Whisper",
        "egemaps_plus_whisper": "eGeMAPS + Whisper",
        "egemaps_xlsr": "eGeMAPS + XLSR-53",
        "egemaps_plus_xlsr": "eGeMAPS + XLSR-53",
        "whisper_xlsr": "Whisper + XLSR-53",
        "whisper_plus_xlsr": "Whisper + XLSR-53",
        "speech_all": "Speech fusion",
        "speech_fusion": "Speech fusion",
        "speech_all_plus_whisper": "Speech fusion + Whisper",
        "speech_fusion_plus_whisper": "Speech fusion + Whisper",
        "sbert": "SBERT transcript semantics",
        "sbert_transcript_semantics": "SBERT transcript semantics",
        "psycemb": "PsycEmb",
        "liwc": "LIWC",
        "psycemb_liwc": "PsycEmb + LIWC",
        "psycemb_plus_liwc": "PsycEmb + LIWC",
        "text_psych": "Text + psychology fusion",
        "text_psychology": "Text + psychology fusion",
        "text_plus_psychology": "Text + psychology fusion",
        "all_modalities": "Full multimodal fusion",
        "full_multimodal": "Full multimodal fusion",
        "all_modalities_plus_whisper": "Full multimodal fusion + Whisper",
        "full_multimodal_plus_whisper": "Full multimodal fusion + Whisper",
    }
    return feature_map.get(x, feature_map.get(x.lower(), x))


def normalize_model_name(x):
    x = str(x).strip()
    model_map = {
        "logreg": "Logistic regression",
        "logistic_regression": "Logistic regression",
        "linear_svm": "Linear SVM",
        "svm": "Linear SVM",
        "random_forest": "Random Forest",
        "rf": "Random Forest",
        "mlp_fc": "Shallow MLP",
        "shallow_mlp": "Shallow MLP",
        "deep_mlp": "Deep MLP",
        "tabular_resnet": "Tabular ResNet",
        "feature_gated_mlp": "Feature-Gated MLP",
        "cnn1d_pool": "1D-CNN with pooling",
        "cnn_1d_pool": "1D-CNN with pooling",
    }
    return model_map.get(x, model_map.get(x.lower(), x))


def group_feature(x):
    x = normalize_feature_name(x)

    speech = {
        "eGeMAPS", "Whisper-small", "WhiSPA-Small", "WhiSPA-Tiny", "XLSR-53",
        "eGeMAPS + Whisper", "eGeMAPS + XLSR-53", "Whisper + XLSR-53",
        "Speech fusion", "Speech fusion + Whisper",
    }
    text = {"SBERT transcript semantics"}
    psych = {"PsycEmb", "LIWC", "PsycEmb + LIWC"}
    text_psych = {"Text + psychology fusion"}
    full = {"Full multimodal fusion", "Full multimodal fusion + Whisper"}

    if x in speech:
        return "Speech"
    if x in text:
        return "Text"
    if x in psych:
        return "Psychology"
    if x in text_psych:
        return "Text + psychology"
    if x in full:
        return "Full multimodal"
    return "Other"


def main():
    parser = argparse.ArgumentParser(
        description="Make a feature-family heatmap from all8 modeling summary CSV."
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to summary_with_whisper_v2_all8.csv or summary_with_whisper_v2.csv",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save figure and CSV tables",
    )
    args = parser.parse_args()

    infile = Path(args.input)
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(infile)

    dataset_col = detect_column(df, ["dataset", "Dataset"], "dataset column")
    feature_col = detect_column(
        df,
        ["feature_display", "feature_set", "feature", "raw_feature_id", "feature_id", "feature_name", "condition", "setting", "combo"],
        "feature column",
    )
    score_col = detect_column(
        df,
        ["macro_f1_mean", "macro_f1", "mean_macro_f1", "macro_f1_avg"],
        "macro-F1 column",
    )

    model_col = None
    for c in ["classifier_display", "model_display", "model_name", "classifier", "model", "clf"]:
        if c in df.columns:
            model_col = c
            break

    df = df.copy()
    df["dataset_norm"] = df[dataset_col].astype(str).str.lower().str.strip()
    df["feature_display_norm"] = df[feature_col].apply(normalize_feature_name)
    df["feature_family"] = df["feature_display_norm"].apply(group_feature)
    if model_col is not None:
        df["classifier_display_norm"] = df[model_col].apply(normalize_model_name)
    else:
        df["classifier_display_norm"] = ""

    keep_families = ["Speech", "Text", "Psychology", "Text + psychology", "Full multimodal"]
    df = df[df["feature_family"].isin(keep_families)].copy()

    if df.empty:
        raise ValueError("No rows matched expected feature families. Check feature names in the input CSV.")

    # Best setting within each dataset x feature_family.
    idx = df.groupby(["dataset_norm", "feature_family"])[score_col].idxmax()
    best = df.loc[idx].copy()

    dataset_display = {
        "daic": "DAIC-WOZ",
        "edaic": "E-DAIC",
        "eatd": "EATD",
        "modma": "MODMA",
        "pdch": "PDCH",
    }
    best["dataset_display"] = best["dataset_norm"].map(dataset_display).fillna(best[dataset_col].astype(str))

    best["best_setting_text"] = (
        best["feature_display_norm"].astype(str)
        + " / "
        + best["classifier_display_norm"].astype(str)
        + " ("
        + best[score_col].astype(float).map(lambda x: f"{x:.3f}")
        + ")"
    )

    best_out = best[[
        "dataset_norm", "dataset_display", "feature_family", "feature_display_norm",
        "classifier_display_norm", score_col, "best_setting_text",
    ]].sort_values(["dataset_norm", "feature_family"])
    best_out.to_csv(outdir / "feature_family_best_macro_f1_long.csv", index=False)

    row_order = ["daic", "edaic", "eatd", "modma", "pdch"]
    col_order = ["Speech", "Text", "Psychology", "Text + psychology", "Full multimodal"]

    pivot = (
        best.pivot_table(
            index="dataset_norm",
            columns="feature_family",
            values=score_col,
            aggfunc="first",
        )
        .reindex(index=row_order, columns=col_order)
    )
    pivot.index = [dataset_display.get(x, x) for x in pivot.index]
    pivot.to_csv(outdir / "feature_family_best_macro_f1_matrix.csv")

    mat = pivot.values.astype(float)

    fig, ax = plt.subplots(figsize=(11.2, 4.6))
    im = ax.imshow(mat, aspect="auto")

    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=15, ha="right", fontsize=10)
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=10)

    finite_vals = mat[np.isfinite(mat)]
    threshold = (finite_vals.min() + finite_vals.max()) / 2.0 if finite_vals.size else 0.0

    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            val = mat[i, j]
            if np.isnan(val):
                txt = "--"
                text_color = "black"
            else:
                txt = f"{val:.3f}"
                text_color = "white" if val > threshold else "black"
            ax.text(j, i, txt, ha="center", va="center", fontsize=9, color=text_color)

    ax.set_xticks(np.arange(-0.5, len(pivot.columns), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(pivot.index), 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=1.2)
    ax.tick_params(which="minor", bottom=False, left=False)

    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("Best macro-F1", fontsize=10)
    ax.set_title("Best macro-F1 within each feature family", fontsize=12, pad=10)

    plt.tight_layout()

    png_path = outdir / "figure_feature_family_best_macro_f1_heatmap.png"
    pdf_path = outdir / "figure_feature_family_best_macro_f1_heatmap.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    print("Saved figure files:")
    print(png_path)
    print(pdf_path)
    print("\nSaved CSV files:")
    print(outdir / "feature_family_best_macro_f1_long.csv")
    print(outdir / "feature_family_best_macro_f1_matrix.csv")
    print("\nBest settings used for heatmap:")
    print(best_out.to_string(index=False))


if __name__ == "__main__":
    main()
