import os
from pathlib import Path
import itertools
import numpy as np
import pandas as pd

ROOT = Path(os.environ.get("WHISPA_LIWC_ROOT", "/path/to/WhiSPA_LIWC"))

# Prefer stable rerun output if it exists; otherwise analyze the first completed v1 output.
CANDIDATES = [
    ROOT / "reviewer_robustness_outputs/liwc_substitution_v1_stable_20260509",
    ROOT / "reviewer_robustness_outputs/liwc_substitution_v1_fixed_20260509",
]
for c in CANDIDATES:
    if (c / "liwc_substitution_fold_results.csv").exists():
        OUTDIR = c
        break
else:
    raise FileNotFoundError("No LIWC substitution fold result file found.")

fold_path = OUTDIR / "liwc_substitution_fold_results.csv"
summary_path = OUTDIR / "liwc_substitution_summary.csv"

folds = pd.read_csv(fold_path)
summary = pd.read_csv(summary_path)

TABLE_DIR = OUTDIR / "tables"
TABLE_DIR.mkdir(parents=True, exist_ok=True)

METRICS = ["macro_f1", "recall_pos", "balanced_accuracy", "auroc"]

def bh_adjust(pvals):
    pvals = np.asarray(pvals, dtype=float)
    n = len(pvals)
    order = np.argsort(pvals)
    ranked = pvals[order]
    q = np.empty(n, dtype=float)
    prev = 1.0
    for i in range(n - 1, -1, -1):
        rank = i + 1
        val = ranked[i] * n / rank
        prev = min(prev, val)
        q[order[i]] = min(prev, 1.0)
    return q

def bootstrap_ci(diff, n_boot=10000, seed=42):
    diff = np.asarray(diff, dtype=float)
    rng = np.random.default_rng(seed)
    if len(diff) == 0:
        return np.nan, np.nan
    means = []
    for _ in range(n_boot):
        sample = rng.choice(diff, size=len(diff), replace=True)
        means.append(np.nanmean(sample))
    return float(np.nanpercentile(means, 2.5)), float(np.nanpercentile(means, 97.5))

def sign_flip_p(diff):
    diff = np.asarray(diff, dtype=float)
    diff = diff[~np.isnan(diff)]
    n = len(diff)
    if n == 0:
        return np.nan
    obs = abs(np.mean(diff))
    # Exact sign-flip is feasible for n=10 dataset-classifier pairs.
    if n <= 20:
        count = 0
        total = 0
        for signs in itertools.product([-1, 1], repeat=n):
            val = abs(np.mean(diff * np.asarray(signs)))
            count += int(val >= obs - 1e-12)
            total += 1
        return count / total
    rng = np.random.default_rng(123)
    vals = []
    for _ in range(20000):
        signs = rng.choice([-1, 1], size=n)
        vals.append(abs(np.mean(diff * signs)))
    return float(np.mean(np.asarray(vals) >= obs - 1e-12))

# Dataset-model means are the unit for broad paired tests.
means = (
    folds
    .groupby(["dataset", "model", "route", "liwc_variant"], as_index=False)[METRICS]
    .mean()
)

def get_condition(route, variant):
    sub = means[(means["route"] == route) & (means["liwc_variant"] == variant)].copy()
    return sub

comparison_specs = []

# Route-level base comparison
comparison_specs.append({
    "comparison_family": "route_base",
    "a_label": "WhiSPA base",
    "b_label": "Explicit base",
    "a_route": "integrated_whispa",
    "a_variant": "base",
    "b_route": "explicit_xlsr_sbert_psycemb",
    "b_variant": "base",
})

# Additive LIWC comparisons
for route, route_label in [
    ("integrated_whispa", "WhiSPA"),
    ("explicit_xlsr_sbert_psycemb", "Explicit"),
]:
    comparison_specs.append({
        "comparison_family": "additive_liwc",
        "a_label": f"{route_label}+intact LIWC",
        "b_label": f"{route_label} base",
        "a_route": route,
        "a_variant": "intact",
        "b_route": route,
        "b_variant": "base",
    })

# LIWC control comparisons
for route, route_label in [
    ("integrated_whispa", "WhiSPA"),
    ("explicit_xlsr_sbert_psycemb", "Explicit"),
]:
    for control in ["pca", "shuffled", "random"]:
        comparison_specs.append({
            "comparison_family": "liwc_control",
            "a_label": f"{route_label}+intact LIWC",
            "b_label": f"{route_label}+{control} LIWC",
            "a_route": route,
            "a_variant": "intact",
            "b_route": route,
            "b_variant": control,
        })

rows = []
pair_detail_rows = []

for spec in comparison_specs:
    a = get_condition(spec["a_route"], spec["a_variant"])
    b = get_condition(spec["b_route"], spec["b_variant"])
    merged = a.merge(
        b,
        on=["dataset", "model"],
        suffixes=("__a", "__b"),
    )

    if len(merged) == 0:
        continue

    for metric in METRICS:
        diff = merged[f"{metric}__a"] - merged[f"{metric}__b"]
        ci_low, ci_high = bootstrap_ci(diff.values)
        p = sign_flip_p(diff.values)

        rows.append({
            "comparison_family": spec["comparison_family"],
            "comparison": f"{spec['a_label']} minus {spec['b_label']}",
            "metric": metric,
            "n_pairs": len(diff),
            "mean_diff": float(np.nanmean(diff)),
            "median_diff": float(np.nanmedian(diff)),
            "ci_low": ci_low,
            "ci_high": ci_high,
            "p": p,
            "a_mean": float(np.nanmean(merged[f"{metric}__a"])),
            "b_mean": float(np.nanmean(merged[f"{metric}__b"])),
        })

        for _, r in merged.iterrows():
            pair_detail_rows.append({
                "comparison_family": spec["comparison_family"],
                "comparison": f"{spec['a_label']} minus {spec['b_label']}",
                "metric": metric,
                "dataset": r["dataset"],
                "model": r["model"],
                "a_value": r[f"{metric}__a"],
                "b_value": r[f"{metric}__b"],
                "diff": r[f"{metric}__a"] - r[f"{metric}__b"],
            })

tests = pd.DataFrame(rows)
tests["q"] = bh_adjust(tests["p"].values)

details = pd.DataFrame(pair_detail_rows)

tests_out = TABLE_DIR / "dataset_model_paired_tests.csv"
details_out = TABLE_DIR / "dataset_model_pair_details.csv"
tests.to_csv(tests_out, index=False)
details.to_csv(details_out, index=False)

# Compact manuscript-facing tables
main_tests = tests[
    (
        ((tests["comparison_family"] == "route_base") & (tests["metric"].isin(["macro_f1", "recall_pos", "auroc"])))
        | ((tests["comparison_family"] == "additive_liwc") & (tests["metric"].isin(["macro_f1", "recall_pos", "auroc"])))
        | ((tests["comparison_family"] == "liwc_control") & (tests["metric"] == "macro_f1"))
    )
].copy()
main_tests = main_tests.sort_values(["comparison_family", "comparison", "metric"])
main_out = TABLE_DIR / "main_liwc_substitution_tests.csv"
main_tests.to_csv(main_out, index=False)

# Route summary table averaged over models
summary_compact = summary[[
    "dataset", "route", "liwc_variant", "model",
    "n_total", "n_features", "macro_f1_mean", "recall_pos_mean",
    "balanced_accuracy_mean", "auroc_mean"
]].copy()
summary_compact_out = TABLE_DIR / "liwc_condition_summary_compact.csv"
summary_compact.to_csv(summary_compact_out, index=False)

print("Analyzing:", OUTDIR)
print("\nSaved:")
print(tests_out)
print(details_out)
print(main_out)
print(summary_compact_out)

print("\n=== Main LIWC substitution tests ===")
display_cols = [
    "comparison_family", "comparison", "metric", "n_pairs",
    "mean_diff", "ci_low", "ci_high", "p", "q", "a_mean", "b_mean"
]
print(main_tests[display_cols].to_string(index=False))

print("\n=== Macro-F1 condition summary ===")
print(
    summary_compact
    .sort_values(["dataset", "route", "model", "macro_f1_mean"], ascending=[True, True, True, False])
    [["dataset", "route", "liwc_variant", "model", "n_total", "n_features", "macro_f1_mean", "recall_pos_mean", "auroc_mean"]]
    .to_string(index=False)
)
