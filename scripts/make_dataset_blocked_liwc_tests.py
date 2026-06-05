from pathlib import Path
from itertools import product
import numpy as np
import pandas as pd

ROOT = Path("reviewer_robustness_outputs/liwc_substitution_v1_stable_20260509")
INFILE = ROOT / "tables" / "dataset_model_pair_details.csv"
OUTDIR = ROOT / "tables"
OUTDIR.mkdir(parents=True, exist_ok=True)

OUT_CSV = OUTDIR / "dataset_blocked_liwc_macro_f1_tests.csv"
OUT_TEX = OUTDIR / "dataset_blocked_liwc_macro_f1_tests_table.tex"

df = pd.read_csv(INFILE)
df.columns = [c.strip().lower().replace("-", "_").replace(" ", "_") for c in df.columns]

# Keep Macro-F1 and the two main linear classifiers.
df = df[df["metric"].astype(str).str.lower().eq("macro_f1")].copy()
m = df["model"].astype(str).str.lower()
df = df[m.isin(["logreg", "linear_svm"])].copy()

# Average the two classifiers within each dataset.
blocked = (
    df.groupby(["comparison_family", "comparison", "metric", "dataset"], as_index=False)["diff"]
      .mean()
)

def signflip_p(values):
    values = np.asarray(values, dtype=float)
    obs = values.mean()
    stats = []
    for signs in product([-1, 1], repeat=len(values)):
        stats.append((values * np.asarray(signs)).mean())
    stats = np.asarray(stats)
    return float((np.abs(stats) >= abs(obs) - 1e-12).mean())

def bootstrap_ci(values, n_boot=20000, seed=13):
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        sample = rng.choice(values, size=len(values), replace=True)
        boots.append(sample.mean())
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(lo), float(hi)

rows = []
for (fam, comp, metric), g in blocked.groupby(["comparison_family", "comparison", "metric"]):
    vals = g.sort_values("dataset")["diff"].to_numpy()
    if len(vals) != 5:
        print("Warning: expected 5 datasets but got", len(vals), "for", comp)
    mean = vals.mean()
    lo, hi = bootstrap_ci(vals)
    p = signflip_p(vals)
    rows.append({
        "comparison_family": fam,
        "comparison": comp,
        "metric": metric,
        "n_datasets": len(vals),
        "mean_diff": mean,
        "ci_low": lo,
        "ci_high": hi,
        "p": p,
    })

out = pd.DataFrame(rows)

# BH q-values across the same dataset-blocked macro-F1 tests.
out = out.sort_values(["comparison_family", "comparison"]).reset_index(drop=True)
pvals = out["p"].to_numpy()
order = np.argsort(pvals)
q = np.empty_like(pvals, dtype=float)
m_tests = len(pvals)
prev = 1.0
for rank, idx in enumerate(order[::-1], start=1):
    # reversed rank: largest to smallest
    actual_rank = m_tests - rank + 1
    val = pvals[idx] * m_tests / actual_rank
    prev = min(prev, val)
    q[idx] = min(prev, 1.0)
out["q"] = q

# Order key rows similar to main table.
preferred = [
    "WhiSPA base minus Explicit base",
    "Explicit+LIWC minus explicit base",
    "WhiSPA+LIWC minus WhiSPA base",
    "Explicit+LIWC minus explicit+PCA LIWC",
    "Explicit+LIWC minus explicit+shuffled LIWC",
    "Explicit+LIWC minus explicit+random LIWC",
    "WhiSPA+LIWC minus WhiSPA+PCA LIWC",
    "WhiSPA+LIWC minus WhiSPA+shuffled LIWC",
    "WhiSPA+LIWC minus WhiSPA+random LIWC",
]
out["order"] = out["comparison"].apply(lambda x: preferred.index(x) if x in preferred else 999)
out = out.sort_values(["order", "comparison"]).drop(columns=["order"])

out.to_csv(OUT_CSV, index=False)

def fmt(x):
    return f"{x:.3f}"

tex = out.copy()
tex["mean_diff"] = tex["mean_diff"].apply(fmt)
tex["95% CI"] = "[" + tex["ci_low"].apply(fmt) + ", " + tex["ci_high"].apply(fmt) + "]"
tex["p"] = tex["p"].apply(fmt)
tex["q"] = tex["q"].apply(fmt)
tex = tex[["comparison", "n_datasets", "mean_diff", "95% CI", "p", "q"]]
tex.columns = ["Comparison", "$N$", "$\\Delta$", "95\\% CI", "$p$", "$q$"]

latex = tex.to_latex(
    index=False,
    escape=False,
    column_format="llrrrr",
    caption=(
        "Dataset-blocked macro-F1 sensitivity analysis. "
        "For each dataset, logistic regression and Linear SVM differences are first averaged, "
        "and paired tests are then computed over five dataset-level observations."
    ),
    label="tab:dataset-blocked-liwc-tests"
)

latex = latex.replace("\\begin{table}", "\\begin{table*}[t]")
latex = latex.replace("\\end{table}", "\\end{table*}")
latex = latex.replace("\\toprule", "\\hline")
latex = latex.replace("\\midrule", "\\hline")
latex = latex.replace("\\bottomrule", "\\hline")

OUT_TEX.write_text(latex)

print("Saved:", OUT_CSV)
print("Saved:", OUT_TEX)
print(out.to_string(index=False))
