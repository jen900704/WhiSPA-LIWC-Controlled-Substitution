import os
from pathlib import Path
import argparse
import itertools
import numpy as np
import pandas as pd

ROOT = Path(os.environ.get("WHISPA_LIWC_ROOT", "/path/to/WhiSPA_LIWC"))

def exact_signflip_p(values, alternative="two-sided"):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return np.nan
    obs = values.mean()
    stats = []
    for signs in itertools.product([-1, 1], repeat=len(values)):
        stats.append((values * np.asarray(signs)).mean())
    stats = np.asarray(stats)

    if alternative == "greater":
        return float((stats >= obs - 1e-12).mean())
    if alternative == "less":
        return float((stats <= obs + 1e-12).mean())
    return float((np.abs(stats) >= abs(obs) - 1e-12).mean())

def bootstrap_ci(values, n_boot=20000, seed=13):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    vals = np.array([rng.choice(values, size=len(values), replace=True).mean() for _ in range(n_boot)])
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))

def hierarchical_cluster_bootstrap(df, n_boot=20000, seed=13):
    """
    Resample datasets as top-level clusters, then resample model rows within each sampled dataset.
    This uses more information than a pure dataset-blocked sign-flip test, but still treats dataset
    as the main clustering unit.
    """
    rng = np.random.default_rng(seed)
    datasets = sorted(df["dataset"].unique().tolist())
    boot_means = []
    for _ in range(n_boot):
        sampled_datasets = rng.choice(datasets, size=len(datasets), replace=True)
        vals = []
        for ds in sampled_datasets:
            sub = df[df["dataset"] == ds]
            idx = rng.integers(0, len(sub), size=len(sub))
            vals.extend(sub.iloc[idx]["diff"].astype(float).tolist())
        boot_means.append(np.mean(vals))
    boot_means = np.asarray(boot_means, dtype=float)
    return {
        "hier_boot_mean": float(boot_means.mean()),
        "hier_boot_ci_low": float(np.percentile(boot_means, 2.5)),
        "hier_boot_ci_high": float(np.percentile(boot_means, 97.5)),
        "hier_boot_pr_gt_0": float((boot_means > 0).mean()),
        "hier_boot_pr_lt_0": float((boot_means < 0).mean()),
    }

def bh(pvals):
    p = np.asarray(pvals, dtype=float)
    q = np.full(p.shape, np.nan)
    mask = np.isfinite(p)
    pv = p[mask]
    if len(pv) == 0:
        return q
    order = np.argsort(pv)
    ranked = pv[order]
    n = len(ranked)
    adj = ranked * n / np.arange(1, n + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    adj = np.clip(adj, 0, 1)
    out = np.empty_like(adj)
    out[order] = adj
    q[mask] = out
    return q

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--infile",
        default=str(ROOT / "reviewer_robustness_outputs/liwc_substitution_v1_multidraw_20260512_n30/tables/liwc_substitution_pair_details_multidraw.csv"),
    )
    parser.add_argument(
        "--outdir",
        default=str(ROOT / "reviewer_robustness_outputs/liwc_substitution_v1_multidraw_20260512_n30/tables"),
    )
    parser.add_argument("--metric", default="macro_f1")
    args = parser.parse_args()

    infile = Path(args.infile)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(infile)
    df = df[df["metric"] == args.metric].copy()

    rows = []
    for comp, sub in df.groupby("comparison", sort=False):
        vals_pair = sub["diff"].astype(float).values
        dataset_means = sub.groupby("dataset")["diff"].mean().reset_index()
        vals_ds = dataset_means["diff"].astype(float).values

        pair_ci = bootstrap_ci(vals_pair, seed=abs(hash(("pair", comp))) % 1000000)
        ds_ci = bootstrap_ci(vals_ds, seed=abs(hash(("dataset", comp))) % 1000000)
        hier = hierarchical_cluster_bootstrap(sub, seed=abs(hash(("hier", comp))) % 1000000)

        rows.append({
            "comparison": comp,
            "metric": args.metric,
            "n_pairs": len(vals_pair),
            "pair_mean": vals_pair.mean(),
            "pair_ci_low": pair_ci[0],
            "pair_ci_high": pair_ci[1],
            "pair_two_sided_p": exact_signflip_p(vals_pair, "two-sided"),
            "pair_one_sided_positive_p": exact_signflip_p(vals_pair, "greater"),
            "n_datasets": len(vals_ds),
            "dataset_blocked_mean": vals_ds.mean(),
            "dataset_ci_low": ds_ci[0],
            "dataset_ci_high": ds_ci[1],
            "dataset_two_sided_p": exact_signflip_p(vals_ds, "two-sided"),
            "dataset_one_sided_positive_p": exact_signflip_p(vals_ds, "greater"),
            "min_two_sided_p_possible_dataset_blocked": 2 / (2 ** len(vals_ds)),
            "min_one_sided_p_possible_dataset_blocked": 1 / (2 ** len(vals_ds)),
            **hier,
        })

    out = pd.DataFrame(rows)
    out["pair_q_two_sided_all"] = bh(out["pair_two_sided_p"].values)
    out["dataset_q_two_sided_all"] = bh(out["dataset_two_sided_p"].values)
    out["dataset_q_one_sided_all"] = bh(out["dataset_one_sided_positive_p"].values)

    out_csv = outdir / f"liwc_multidraw_{args.metric}_sensitivity_stats.csv"
    out.to_csv(out_csv, index=False)

    key = out[out["comparison"].str.contains(r"Explicit\+intact LIWC minus Explicit\+", regex=True)].copy()
    key_csv = outdir / f"liwc_multidraw_{args.metric}_key_control_sensitivity_stats.csv"
    key.to_csv(key_csv, index=False)

    print("Saved:", out_csv)
    print("Saved:", key_csv)

    print("\nKey explicit-route LIWC control sensitivity stats:")
    display_cols = [
        "comparison",
        "pair_mean", "pair_two_sided_p", "pair_q_two_sided_all",
        "dataset_blocked_mean", "dataset_two_sided_p", "dataset_one_sided_positive_p",
        "min_two_sided_p_possible_dataset_blocked",
        "hier_boot_ci_low", "hier_boot_ci_high", "hier_boot_pr_gt_0",
    ]
    print(key[display_cols].to_string(index=False))

    print("\nAll comparisons:")
    display_cols_all = [
        "comparison",
        "pair_mean", "pair_two_sided_p",
        "dataset_blocked_mean", "dataset_two_sided_p", "dataset_one_sided_positive_p",
        "hier_boot_ci_low", "hier_boot_ci_high", "hier_boot_pr_gt_0",
    ]
    print(out[display_cols_all].to_string(index=False))

if __name__ == "__main__":
    main()
