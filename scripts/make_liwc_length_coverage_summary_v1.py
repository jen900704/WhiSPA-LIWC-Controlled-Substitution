from pathlib import Path
import argparse
import numpy as np
import pandas as pd

DATASET_ORDER = ["daic", "edaic", "eatd", "modma", "pdch"]

def find_liwc_files(root: Path):
    chosen = {}
    for ds in DATASET_ORDER:
        candidates = []
        for p in root.rglob("liwc.csv"):
            parts = [x.lower() for x in p.parts]
            if ds in parts:  # exact path-part matching, so edaic is not mistaken for daic
                candidates.append(p)
        if candidates:
            chosen[ds] = sorted(candidates, key=lambda x: (len(x.parts), x.as_posix()))[0]
    return chosen

def pick_col(cols, candidates):
    lower = {c.lower(): c for c in cols}
    for cand in candidates:
        if cand.lower() in lower:
            return lower[cand.lower()]
    return None

def iqr(x):
    x = pd.to_numeric(x, errors="coerce").dropna()
    if len(x) == 0:
        return np.nan
    q1, q3 = np.percentile(x, [25, 75])
    return q3 - q1

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feature_root", default="reviewer_robustness_outputs/modeling_ready_features_with_whisper_v2")
    ap.add_argument("--tag", default="20260515_v2")
    args = ap.parse_args()

    root = Path(args.feature_root)
    outdir = Path("reviewer_robustness_outputs") / f"liwc_length_coverage_{args.tag}"
    outdir.mkdir(parents=True, exist_ok=True)

    files = find_liwc_files(root)
    if not files:
        raise SystemExit("No liwc.csv files found under feature_root.")

    rows = []
    for ds in DATASET_ORDER:
        p = files.get(ds)
        if p is None:
            print(f"[SKIP] {ds}: no liwc.csv found")
            continue
        df = pd.read_csv(p)
        wc_col = pick_col(df.columns, ["WC", "word_count", "words", "n_tokens", "num_tokens", "token_count"])
        cov_col = pick_col(df.columns, ["Dic", "dictionary_coverage", "liwc_coverage", "coverage", "dict_coverage"])

        wc = pd.to_numeric(df[wc_col], errors="coerce") if wc_col else pd.Series([np.nan] * len(df))
        cov = pd.to_numeric(df[cov_col], errors="coerce") if cov_col else pd.Series([np.nan] * len(df))
        if cov.notna().any() and cov.max() <= 1.5:
            cov = cov * 100

        rows.append({
            "dataset": ds, "n": len(df), "token_col": wc_col, "coverage_col": cov_col,
            "mean_tokens": wc.mean(), "median_tokens": wc.median(), "iqr_tokens": iqr(wc),
            "mean_liwc_coverage_pct": cov.mean(), "median_liwc_coverage_pct": cov.median(),
            "source_file": str(p)
        })

    out = pd.DataFrame(rows)
    out_path = outdir / "liwc_length_coverage_summary.csv"
    out.to_csv(out_path, index=False)
    print("\nSaved:")
    print(out_path)
    print("\nSummary:")
    print(out.to_string(index=False))

if __name__ == "__main__":
    main()
