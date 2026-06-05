import os
from pathlib import Path
import itertools
import re
import numpy as np
import pandas as pd

ROOT = Path(os.environ.get("WHISPA_LIWC_ROOT", "/path/to/WhiSPA_LIWC"))
SEARCH_ROOT = ROOT / "reviewer_robustness_outputs"
OUTDIR = SEARCH_ROOT / "liwc_sensitivity_20260510"
OUTDIR.mkdir(parents=True, exist_ok=True)

DATASET_CANON = {
    "daic": "daic", "daic-woz": "daic", "daic_woz": "daic",
    "edaic": "edaic", "e-daic": "edaic",
    "eatd": "eatd", "modma": "modma", "pdch": "pdch",
}

LANGUAGE_GROUPS = {
    "English": ["daic", "edaic"],
    "Chinese": ["eatd", "modma", "pdch"],
}

def norm_name(x):
    return re.sub(r"[^a-z0-9]+", "_", str(x).strip().lower()).strip("_")

def norm_dataset(x):
    s = str(x).strip().lower()
    return DATASET_CANON.get(s, DATASET_CANON.get(s.replace("_", "-"), s))

def pick_col(cols, candidates):
    norm_cols = {norm_name(c): c for c in cols}
    for cand in candidates:
        key = norm_name(cand)
        if key in norm_cols:
            return norm_cols[key]
    return None

def sign_flip_p(diff):
    diff = np.asarray(diff, dtype=float)
    diff = diff[~np.isnan(diff)]
    n = len(diff)
    if n == 0:
        return np.nan
    obs = abs(np.mean(diff))
    count = 0
    total = 0
    for signs in itertools.product([-1, 1], repeat=n):
        val = abs(np.mean(diff * np.asarray(signs)))
        count += int(val >= obs - 1e-12)
        total += 1
    return count / total

def bootstrap_ci(diff, n_boot=10000, seed=42):
    diff = np.asarray(diff, dtype=float)
    diff = diff[~np.isnan(diff)]
    if len(diff) == 0:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    means = []
    for _ in range(n_boot):
        sample = rng.choice(diff, size=len(diff), replace=True)
        means.append(np.mean(sample))
    return np.percentile(means, 2.5), np.percentile(means, 97.5)

def bh_adjust(pvals):
    p = np.asarray(pvals, dtype=float)
    q = np.full_like(p, np.nan, dtype=float)
    mask = ~np.isnan(p)
    if mask.sum() == 0:
        return q
    idx = np.where(mask)[0]
    order = idx[np.argsort(p[idx])]
    m = len(order)
    prev = 1.0
    for rank_from_end, i in enumerate(order[::-1], start=1):
        rank = m - rank_from_end + 1
        val = p[i] * m / rank
        prev = min(prev, val)
        q[i] = min(prev, 1.0)
    return q

def file_candidate_score(path):
    try:
        df = pd.read_csv(path, nrows=300)
    except Exception:
        return -1, None

    cols = list(df.columns)
    comp = pick_col(cols, ["comparison", "contrast", "test", "comparison_label"])
    metric = pick_col(cols, ["metric", "score_metric"])
    dataset = pick_col(cols, ["dataset", "corpus"])
    model = pick_col(cols, ["model", "classifier", "clf"])
    diff = pick_col(cols, ["diff", "delta", "mean_diff", "score_diff"])
    a_value = pick_col(cols, ["a_value", "condition_a_value", "left_value", "value_a"])
    b_value = pick_col(cols, ["b_value", "condition_b_value", "right_value", "value_b"])

    score = 0
    if comp: score += 3
    if metric: score += 2
    if dataset: score += 2
    if model: score += 2
    if diff or (a_value and b_value): score += 3

    text = " ".join(map(str, df.head(300).astype(str).values.ravel())).lower()
    for term in ["liwc", "explicit", "whispa", "pca", "shuffled", "random", "macro", "recall", "auroc"]:
        if term in text:
            score += 1

    return score, {"path": path, "columns": cols, "comp": comp, "metric": metric,
                   "dataset": dataset, "model": model, "diff": diff,
                   "a_value": a_value, "b_value": b_value}

csvs = []
for p in SEARCH_ROOT.rglob("*.csv"):
    try:
        if p.stat().st_size < 80_000_000:
            csvs.append(p)
    except Exception:
        pass

scored = []
for p in csvs:
    score, info = file_candidate_score(p)
    if score >= 8:
        scored.append((score, info))

scored = sorted(scored, key=lambda x: x[0], reverse=True)

print("\n=== Candidate LIWC pair-detail files ===")
for score, info in scored[:20]:
    print(f"score={score:02d}", info["path"])
    print("  columns:", info["columns"][:12])

if not scored:
    raise SystemExit(
        "\nNo suitable pair-detail CSV found. Paste the candidate output from:\n"
        "find reviewer_robustness_outputs -type f -name '*.csv' | grep -Ei 'liwc|route|substitution|pair|detail|stable|control'\n"
    )

info = scored[0][1]
path = info["path"]
print("\nUSING:", path)

df = pd.read_csv(path)

comp_col = info["comp"]
metric_col = info["metric"]
dataset_col = info["dataset"]
model_col = info["model"]
diff_col = info["diff"]
a_col = info["a_value"]
b_col = info["b_value"]

if diff_col is None:
    df["__diff__"] = pd.to_numeric(df[a_col], errors="coerce") - pd.to_numeric(df[b_col], errors="coerce")
    diff_col = "__diff__"

work = df[[comp_col, metric_col, dataset_col, model_col, diff_col]].copy()
work.columns = ["comparison", "metric", "dataset", "model", "diff"]
work["dataset"] = work["dataset"].map(norm_dataset)
work["metric_norm"] = work["metric"].astype(str).str.lower()
work["comparison_norm"] = work["comparison"].astype(str).str.lower()
work["diff"] = pd.to_numeric(work["diff"], errors="coerce")

work = work[
    work["comparison_norm"].str.contains("liwc|whispa|explicit", na=False)
    & work["metric_norm"].str.contains("macro|recall|auroc|auc", na=False)
].copy()

pair = (
    work.groupby(["comparison", "metric", "dataset", "model"], as_index=False)
    .agg(diff=("diff", "mean"))
)

pair_out = OUTDIR / "liwc_pair_level_diffs_detected.csv"
pair.to_csv(pair_out, index=False)

print("\nSaved detected pair-level diffs:", pair_out)
print("Rows:", len(pair))
print(pair.head(20).to_string(index=False))

lodo_rows = []
for (comparison, metric), sub in pair.groupby(["comparison", "metric"]):
    datasets = sorted(sub["dataset"].dropna().unique())
    if len(datasets) < 3:
        continue

    diff = sub["diff"].dropna().values
    lo, hi = bootstrap_ci(diff)
    lodo_rows.append({
        "comparison": comparison, "metric": metric, "left_out_dataset": "NONE_FULL",
        "n_pairs": len(diff), "mean_diff": np.mean(diff), "median_diff": np.median(diff),
        "ci_low": lo, "ci_high": hi, "p": sign_flip_p(diff),
    })

    for left_out in datasets:
        rem = sub[sub["dataset"] != left_out]["diff"].dropna().values
        if len(rem) == 0:
            continue
        lo, hi = bootstrap_ci(rem)
        lodo_rows.append({
            "comparison": comparison, "metric": metric, "left_out_dataset": left_out,
            "n_pairs": len(rem), "mean_diff": np.mean(rem), "median_diff": np.median(rem),
            "ci_low": lo, "ci_high": hi, "p": sign_flip_p(rem),
        })

lodo = pd.DataFrame(lodo_rows)
if len(lodo):
    lodo["q_within_table"] = bh_adjust(lodo["p"].values)

lodo_out = OUTDIR / "liwc_leave_one_dataset_out_sensitivity.csv"
lodo.to_csv(lodo_out, index=False)

lang_rows = []
for (comparison, metric), sub in pair.groupby(["comparison", "metric"]):
    for lang, datasets in LANGUAGE_GROUPS.items():
        ss = sub[sub["dataset"].isin(datasets)]
        diff = ss["diff"].dropna().values
        if len(diff) == 0:
            continue
        lo, hi = bootstrap_ci(diff)
        lang_rows.append({
            "comparison": comparison, "metric": metric, "language_group": lang,
            "datasets": ",".join(datasets), "n_pairs": len(diff),
            "mean_diff": np.mean(diff), "median_diff": np.median(diff),
            "ci_low": lo, "ci_high": hi, "p_descriptive": sign_flip_p(diff),
        })

lang = pd.DataFrame(lang_rows)
lang_out = OUTDIR / "liwc_language_stratified_descriptive_summary.csv"
lang.to_csv(lang_out, index=False)

summary_path = OUTDIR / "liwc_sensitivity_summary.txt"

main_terms = ["explicit", "whispa", "random", "pca", "shuffled", "base"]

def compact(df):
    if df is None or len(df) == 0:
        return "EMPTY"
    keep = df.copy()
    keep = keep[
        keep["comparison"].astype(str).str.lower().apply(lambda s: any(t in s for t in main_terms))
    ]
    if "metric" in keep.columns:
        keep = keep[
            keep["metric"].astype(str).str.lower().str.contains("macro|recall|auroc|auc", na=False)
        ]
    return keep.to_string(index=False)

with open(summary_path, "w", encoding="utf-8") as f:
    f.write("SOURCE FILE\n")
    f.write(str(path) + "\n\n")
    f.write("PAIR-LEVEL DIFFS DETECTED\n")
    f.write(pair.to_string(index=False))
    f.write("\n\nLEAVE-ONE-DATASET-OUT SENSITIVITY\n")
    f.write(compact(lodo))
    f.write("\n\nLANGUAGE-STRATIFIED DESCRIPTIVE SUMMARY\n")
    f.write(compact(lang))
    f.write("\n")

print("\nSaved:")
print(pair_out)
print(lodo_out)
print(lang_out)
print(summary_path)

print("\n=== LEAVE-ONE-DATASET-OUT SENSITIVITY ===")
print(compact(lodo))

print("\n=== LANGUAGE-STRATIFIED DESCRIPTIVE SUMMARY ===")
print(compact(lang))
