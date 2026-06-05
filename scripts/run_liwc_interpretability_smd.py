from pathlib import Path
import re
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("reviewer_robustness_outputs")
FEATURE_ROOT = ROOT / "modeling_ready_features_with_whisper_v2"
LABEL_FILE = ROOT / "reviewer_robustness_outputs" / "label_registry" / "canonical_labels.csv"
OUTDIR = ROOT / "liwc_interpretability_20260512"
OUTDIR.mkdir(parents=True, exist_ok=True)

DATASETS = ["daic", "edaic", "eatd", "modma", "pdch"]
DATASET_DISPLAY = {
    "daic": "DAIC-WOZ",
    "edaic": "E-DAIC",
    "eatd": "EATD",
    "modma": "MODMA",
    "pdch": "PDCH",
}

# Theory-guided LIWC domains.
# Aliases cover LIWC-22, LIWC-2015, and simplified Chinese LIWC-style column names.
DOMAIN_MAP = {
    "Self-focus": {
        "First-person singular": ["i", "me", "my", "mine", "ipron1"],
        "First-person plural": ["we", "us", "our", "ours"],
    },
    "Social orientation": {
        "Social": ["social", "socbehav", "socrefs"],
        "Affiliation": ["affiliation", "affil"],
        "Family": ["family"],
        "Friend": ["friend", "friends"],
        "Other pronouns": ["you", "shehe", "they", "youpl"],
    },
    "Negative affect": {
        "Negative emotion": ["negemo", "emo_neg", "negativeemotion", "negative_emotion"],
        "Sadness": ["sad", "sadness", "emo_sad"],
        "Anxiety": ["anx", "anxiety", "emo_anx"],
        "Anger": ["anger", "emo_anger"],
    },
    "Positive / reward orientation": {
        "Positive emotion": ["posemo", "emo_pos", "positiveemotion", "positive_emotion"],
        "Reward": ["reward"],
        "Leisure": ["leisure"],
        "Achievement": ["achieve", "achievement"],
    },
    "Cognitive processing": {
        "Cognitive process": ["cogproc", "cognition", "cognitiveprocess", "cognitive_process"],
        "Insight": ["insight"],
        "Causation": ["cause", "causation"],
        "Discrepancy": ["discrep", "discrepancy"],
        "Tentative": ["tentat", "tentative"],
        "Certainty": ["certain", "certainty"],
        "Negation": ["negate", "negation"],
        "Comparison": ["compare", "comparison"],
    },
    "Somatic / biological": {
        "Body": ["body"],
        "Health": ["health"],
        "Biological": ["bio", "biological", "biologicalprocess"],
        "Illness": ["illness", "sick"],
        "Sleep": ["sleep"],
        "Food / eating": ["food", "eat", "eating"],
    },
    "Temporal focus": {
        "Past focus": ["focuspast", "pastfocus", "past"],
        "Present focus": ["focuspresent", "presentfocus", "present"],
        "Future focus": ["focusfuture", "futurefocus", "future"],
        "Time": ["time"],
    },
    "Risk / death": {
        "Death": ["death", "die", "dying"],
        "Risk": ["risk", "danger"],
    },
}

NON_FEATURE_COLS = {
    "dataset", "participant_id", "id", "filename", "text", "patient_text",
    "segment", "label", "y", "score", "score_name", "label_source", "label_rule"
}

def norm_id(x):
    s = str(x).strip()
    if re.fullmatch(r"\d+\.0", s):
        s = s[:-2]
    return s.lower()

def norm_col(x):
    return re.sub(r"[^a-z0-9]", "", str(x).lower())

def find_col(columns, aliases):
    norm_to_orig = {norm_col(c): c for c in columns}
    for a in aliases:
        key = norm_col(a)
        if key in norm_to_orig:
            return norm_to_orig[key]
    return None

def smd(x_dep, x_ctl):
    x_dep = pd.to_numeric(x_dep, errors="coerce").dropna().astype(float)
    x_ctl = pd.to_numeric(x_ctl, errors="coerce").dropna().astype(float)
    n1, n0 = len(x_dep), len(x_ctl)
    if n1 < 2 or n0 < 2:
        return np.nan
    m1, m0 = x_dep.mean(), x_ctl.mean()
    s1, s0 = x_dep.std(ddof=1), x_ctl.std(ddof=1)
    pooled = np.sqrt(((n1 - 1) * s1**2 + (n0 - 1) * s0**2) / (n1 + n0 - 2))
    if pooled == 0 or np.isnan(pooled):
        return np.nan
    return (m1 - m0) / pooled

labels = pd.read_csv(LABEL_FILE)
labels["dataset"] = labels["dataset"].astype(str).str.lower().str.strip()
labels["participant_id_norm"] = labels["participant_id"].apply(norm_id)

all_rows = []
audit_rows = []
domain_category_rows = []

for dataset in DATASETS:
    fpath = FEATURE_ROOT / dataset / "liwc.csv"
    if not fpath.exists():
        raise FileNotFoundError(f"Missing LIWC file: {fpath}")

    liwc = pd.read_csv(fpath)
    liwc["dataset"] = liwc["dataset"].astype(str).str.lower().str.strip()
    liwc["participant_id_norm"] = liwc["participant_id"].apply(norm_id)

    lab = labels[labels["dataset"].eq(dataset)].copy()

    merged = liwc.merge(
        lab[["dataset", "participant_id_norm", "y", "score", "score_name", "label_rule"]],
        on=["dataset", "participant_id_norm"],
        how="inner"
    )

    audit_rows.append({
        "dataset": dataset,
        "dataset_display": DATASET_DISPLAY[dataset],
        "n_liwc_rows": len(liwc),
        "n_label_rows": len(lab),
        "n_merged_rows": len(merged),
        "n_depressed": int((merged["y"] == 1).sum()),
        "n_non_depressed": int((merged["y"] == 0).sum()),
        "label_rule": lab["label_rule"].iloc[0] if len(lab) else "",
    })

    if len(merged) == 0:
        continue

    cols = list(merged.columns)

    for domain, cat_map in DOMAIN_MAP.items():
        for category_display, aliases in cat_map.items():
            col = find_col(cols, aliases)
            if col is None:
                continue

            if norm_col(col) in {norm_col(c) for c in NON_FEATURE_COLS}:
                continue

            dep = merged.loc[merged["y"] == 1, col]
            ctl = merged.loc[merged["y"] == 0, col]
            val = smd(dep, ctl)

            if np.isnan(val):
                continue

            row = {
                "dataset": dataset,
                "dataset_display": DATASET_DISPLAY[dataset],
                "domain": domain,
                "category": category_display,
                "matched_column": col,
                "smd_depressed_minus_control": val,
                "mean_depressed": pd.to_numeric(dep, errors="coerce").mean(),
                "mean_non_depressed": pd.to_numeric(ctl, errors="coerce").mean(),
                "n_depressed": int(pd.to_numeric(dep, errors="coerce").notna().sum()),
                "n_non_depressed": int(pd.to_numeric(ctl, errors="coerce").notna().sum()),
            }
            all_rows.append(row)
            domain_category_rows.append({
                "dataset": DATASET_DISPLAY[dataset],
                "domain": domain,
                "category": category_display,
                "matched_column": col,
            })

cat_df = pd.DataFrame(all_rows)
audit_df = pd.DataFrame(audit_rows)

audit_path = OUTDIR / "liwc_interpretability_merge_audit.csv"
cat_path = OUTDIR / "liwc_category_smd_by_dataset.csv"
domain_path = OUTDIR / "liwc_domain_smd_by_dataset.csv"
domain_tex_path = OUTDIR / "liwc_domain_summary_table.tex"

audit_df.to_csv(audit_path, index=False)
cat_df.to_csv(cat_path, index=False)

if cat_df.empty:
    raise RuntimeError("No LIWC categories matched. Check LIWC column names.")

domain_df = (
    cat_df.groupby(["dataset", "dataset_display", "domain"], as_index=False)
    .agg(
        domain_smd_mean=("smd_depressed_minus_control", "mean"),
        domain_smd_median=("smd_depressed_minus_control", "median"),
        n_categories=("category", "nunique")
    )
)
domain_df.to_csv(domain_path, index=False)

# Domain summary table: which columns were actually used.
summary_rows = []
for domain, g in cat_df.groupby("domain"):
    used = (
        g[["category", "matched_column"]]
        .drop_duplicates()
        .sort_values(["category", "matched_column"])
    )
    cats = "; ".join([f"{r.category} ({r.matched_column})" for r in used.itertuples()])
    summary_rows.append({"Domain": domain, "LIWC categories used": cats})

summary = pd.DataFrame(summary_rows)
summary.to_latex(domain_tex_path, index=False, escape=True)

def save_heatmap(pivot, title, out_png, out_pdf):
    fig_w = max(8, 1.2 * len(pivot.columns) + 4)
    fig_h = max(4.5, 0.42 * len(pivot.index) + 1.8)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    data = pivot.to_numpy(dtype=float)
    finite = np.isfinite(data)
    vmax = np.nanmax(np.abs(data[finite])) if finite.any() else 1.0
    vmax = max(vmax, 0.30)

    im = ax.imshow(data, aspect="equal", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.set_aspect("equal")

    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=35, ha="right")
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index)

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if np.isfinite(data[i, j]):
                ax.text(j, i, f"{data[i, j]:.2f}", ha="center", va="center", fontsize=8)

    ax.set_title(title)
    cbar = fig.colorbar(im, ax=ax, shrink=0.82)
    cbar.set_label("SMD: depressed minus non-depressed")

    fig.tight_layout()
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)

# Domain heatmap.
domain_pivot = domain_df.pivot(index="domain", columns="dataset_display", values="domain_smd_mean")
domain_pivot = domain_pivot.reindex(index=list(DOMAIN_MAP.keys()))
domain_pivot = domain_pivot[[DATASET_DISPLAY[d] for d in DATASETS]]
save_heatmap(
    domain_pivot,
    "LIWC domain differences by dataset",
    OUTDIR / "figure_liwc_domain_smd_heatmap.png",
    OUTDIR / "figure_liwc_domain_smd_heatmap.pdf"
)

# Category heatmap: keep categories appearing in at least two datasets.
cat_df["category_label"] = cat_df["domain"] + " — " + cat_df["category"]
counts = cat_df.groupby("category_label")["dataset"].nunique()
keep = counts[counts >= 2].index.tolist()

cat_pivot = cat_df[cat_df["category_label"].isin(keep)].pivot_table(
    index="category_label",
    columns="dataset_display",
    values="smd_depressed_minus_control",
    aggfunc="mean"
)

# Order categories by domain order.
order = []
for domain, cat_map in DOMAIN_MAP.items():
    for cat in cat_map.keys():
        label = domain + " — " + cat
        if label in cat_pivot.index:
            order.append(label)

cat_pivot = cat_pivot.reindex(order)
cat_pivot = cat_pivot[[DATASET_DISPLAY[d] for d in DATASETS]]

save_heatmap(
    cat_pivot,
    "Theory-guided LIWC category differences by dataset",
    OUTDIR / "figure_liwc_category_smd_heatmap.png",
    OUTDIR / "figure_liwc_category_smd_heatmap.pdf"
)

print("\nSaved outputs to:", OUTDIR)
print("\nMerge audit:")
print(audit_df.to_string(index=False))

print("\nDomain-level SMD:")
print(domain_df.sort_values(["dataset", "domain"]).to_string(index=False))

print("\nTop category-level differences by absolute SMD:")
print(
    cat_df.assign(abs_smd=cat_df["smd_depressed_minus_control"].abs())
          .sort_values("abs_smd", ascending=False)
          .head(30)
          [["dataset_display", "domain", "category", "matched_column", "smd_depressed_minus_control"]]
          .to_string(index=False)
)
