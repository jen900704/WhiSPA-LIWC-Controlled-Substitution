#!/usr/bin/env python3
"""
Minimum detectable effect / sensitivity summary for LIWC substitution contrasts.

Important interpretation
------------------------
The exact sign-flip/sign test used in the paper is direction-based, so it does not
have a unique magnitude-based minimum detectable effect. This script therefore
reports two complementary quantities:

1. The smallest possible exact sign-flip p-value for n matched units.
2. A paired-t approximation to the minimum detectable absolute mean difference
   at 80% power, using the observed standard deviation of paired differences.

The paired-t MDE is a sensitivity analysis, not a replacement for the paper's
exact paired tests.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import brentq


def latest_pair_details(root: Path) -> Path:
    files = list(root.glob("liwc_substitution_v1_multidraw_*/tables/liwc_substitution_pair_details_multidraw.csv"))
    if not files:
        # fallback: sometimes stored one level higher
        files = list(root.glob("**/liwc_substitution_pair_details_multidraw.csv"))
    if not files:
        raise FileNotFoundError("Could not find liwc_substitution_pair_details_multidraw.csv under reviewer_robustness_outputs")
    return max(files, key=lambda p: p.stat().st_mtime)


def min_two_sided_sign_p(n: int) -> float:
    return min(1.0, 2.0 * (0.5 ** n))


def min_one_sided_sign_p(n: int) -> float:
    return 0.5 ** n


def paired_t_mde(sd: float, n: int, alpha: float = 0.05, power: float = 0.80) -> float:
    """Minimum absolute mean difference for a paired t-test approximation."""
    if n < 2 or not np.isfinite(sd) or sd <= 0:
        return np.nan
    df = n - 1
    tcrit = stats.t.ppf(1 - alpha / 2, df)

    def achieved_power(delta_mean: float) -> float:
        ncp = delta_mean / (sd / np.sqrt(n))
        # two-sided power under noncentral t
        return (1 - stats.nct.cdf(tcrit, df, ncp)) + stats.nct.cdf(-tcrit, df, ncp)

    lo, hi = 0.0, sd * 20
    while achieved_power(hi) < power and hi < sd * 1e6:
        hi *= 2
    try:
        return float(brentq(lambda d: achieved_power(d) - power, lo, hi))
    except Exception:
        # normal fallback
        zcrit = stats.norm.ppf(1 - alpha / 2)
        zpow = stats.norm.ppf(power)
        return float((zcrit + zpow) * sd / np.sqrt(n))


def summarize_diffs(df: pd.DataFrame, group_cols: list[str], diff_col: str = "diff") -> pd.DataFrame:
    rows = []
    for key, g in df.groupby(group_cols, dropna=False):
        diffs = g[diff_col].dropna().to_numpy(float)
        n = len(diffs)
        if n == 0:
            continue
        mean = float(np.mean(diffs))
        sd = float(np.std(diffs, ddof=1)) if n > 1 else np.nan
        se = float(sd / np.sqrt(n)) if n > 1 else np.nan
        ci_low, ci_high = (np.nan, np.nan)
        if n > 1 and np.isfinite(se):
            tcrit = stats.t.ppf(0.975, n - 1)
            ci_low, ci_high = mean - tcrit * se, mean + tcrit * se
        mde = paired_t_mde(sd, n)
        key_tuple = key if isinstance(key, tuple) else (key,)
        row = dict(zip(group_cols, key_tuple))
        row.update({
            "n": n,
            "observed_mean_diff": mean,
            "observed_sd_diff": sd,
            "observed_ci_low": ci_low,
            "observed_ci_high": ci_high,
            "paired_t_mde_80_power_alpha05": mde,
            "abs_observed_over_mde": abs(mean) / mde if np.isfinite(mde) and mde > 0 else np.nan,
            "min_two_sided_signflip_p": min_two_sided_sign_p(n),
            "min_one_sided_signflip_p": min_one_sided_sign_p(n),
            "n_positive": int(np.sum(diffs > 0)),
            "n_negative": int(np.sum(diffs < 0)),
            "n_zero": int(np.sum(diffs == 0)),
        })
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="reviewer_robustness_outputs")
    parser.add_argument("--pair_details", default=None)
    parser.add_argument("--metric", default="macro_f1")
    parser.add_argument("--out_dir", default=None)
    parser.add_argument("--tag", default=datetime.now().strftime("%Y%m%d"))
    args = parser.parse_args()

    root = Path(args.root)
    pair_path = Path(args.pair_details) if args.pair_details else latest_pair_details(root)
    if not pair_path.exists():
        raise FileNotFoundError(pair_path)
    out_dir = Path(args.out_dir) if args.out_dir else root / f"liwc_mde_{args.tag}"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(pair_path)
    required = {"comparison", "dataset", "model", "diff"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns {missing}. Columns are {df.columns.tolist()}")
    if "metric" in df.columns:
        df = df[df["metric"].astype(str).str.lower() == args.metric.lower()].copy()
    if df.empty:
        raise ValueError(f"No rows for metric={args.metric}")

    # Pair-level: each dataset-classifier pair is one unit.
    pair_summary = summarize_diffs(df, ["comparison"])
    pair_summary.insert(0, "analysis_unit", "dataset_classifier")

    # Dataset-blocked: average classifiers within dataset first.
    dblock = (df.groupby(["comparison", "dataset"], as_index=False)["diff"].mean())
    dblock_summary = summarize_diffs(dblock, ["comparison"])
    dblock_summary.insert(0, "analysis_unit", "dataset_blocked")

    combined = pd.concat([pair_summary, dblock_summary], ignore_index=True)

    # Sort important comparisons first if present.
    order = [
        "Explicit+intact LIWC minus Explicit base",
        "WhiSPA+intact LIWC minus WhiSPA base",
        "Explicit+intact LIWC minus Explicit+pca LIWC",
        "Explicit+intact LIWC minus Explicit+shuffled LIWC",
        "Explicit+intact LIWC minus Explicit+random LIWC",
        "WhiSPA+intact LIWC minus WhiSPA+pca LIWC",
        "WhiSPA+intact LIWC minus WhiSPA+shuffled LIWC",
        "WhiSPA+intact LIWC minus WhiSPA+random LIWC",
        "WhiSPA base minus Explicit base",
    ]
    combined["_order"] = combined["comparison"].apply(lambda x: order.index(x) if x in order else 999)
    combined = combined.sort_values(["analysis_unit", "_order", "comparison"]).drop(columns="_order")

    combined.to_csv(out_dir / "liwc_mde_summary.csv", index=False)
    dblock.to_csv(out_dir / "liwc_dataset_blocked_diffs.csv", index=False)

    # Compact LaTeX for selected key comparisons.
    key = combined[combined["comparison"].isin(order[:6])].copy()
    for c in ["observed_mean_diff", "observed_sd_diff", "paired_t_mde_80_power_alpha05", "abs_observed_over_mde", "min_two_sided_signflip_p"]:
        key[c] = key[c].map(lambda v: f"{v:.3f}" if np.isfinite(v) else "")
    key = key.rename(columns={
        "analysis_unit": "Unit",
        "comparison": "Comparison",
        "n": "N",
        "observed_mean_diff": "$\\Delta$",
        "observed_sd_diff": "SD",
        "paired_t_mde_80_power_alpha05": "MDE$_{80}$",
        "abs_observed_over_mde": "$|\\Delta|$/MDE",
        "min_two_sided_signflip_p": "min $p_{2s}$",
    })
    keep_cols = ["Unit", "Comparison", "N", "$\\Delta$", "SD", "MDE$_{80}$", "$|\\Delta|$/MDE", "min $p_{2s}$"]
    tex = key[keep_cols].to_latex(index=False, escape=False)
    (out_dir / "liwc_mde_summary_table.tex").write_text(tex)

    print(f"Using pair details: {pair_path}")
    print("\nSaved:")
    for name in ["liwc_mde_summary.csv", "liwc_dataset_blocked_diffs.csv", "liwc_mde_summary_table.tex"]:
        print(out_dir / name)
    print("\nSummary:")
    pd.set_option("display.max_colwidth", 80)
    print(combined[["analysis_unit", "comparison", "n", "observed_mean_diff", "observed_sd_diff", "paired_t_mde_80_power_alpha05", "abs_observed_over_mde", "min_two_sided_signflip_p"]].to_string(index=False))


if __name__ == "__main__":
    main()
