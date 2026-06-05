import os
from pathlib import Path
import argparse
import json
import warnings
import zlib
from itertools import product

import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, recall_score, balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

warnings.filterwarnings("ignore")

ROOT = Path(os.environ.get("WHISPA_LIWC_ROOT", "/path/to/WhiSPA_LIWC"))

# This follows the original stable LIWC substitution script.
# Do not change to modeling_ready_features_with_whisper_v2 unless you intentionally
# want to redefine the controlled LIWC substitution experiment.
FEATURE_BASE = ROOT / "reviewer_robustness_outputs/modeling_ready_features"
LABEL_FILE = ROOT / "reviewer_robustness_outputs/label_registry/canonical_labels.csv"

DATASETS = ["daic", "edaic", "eatd", "modma", "pdch"]

MODELS = {
    "logreg": LogisticRegression(
        penalty="l2",
        solver="liblinear",
        class_weight="balanced",
        max_iter=5000,
        random_state=42,
    ),
    "linear_svm": LinearSVC(
        class_weight="balanced",
        random_state=42,
        max_iter=10000,
    ),
}

META_COLS = {"dataset", "participant_id"}


def stable_seed(*items):
    s = "|".join(map(str, items)).encode("utf-8")
    return 100000 + (zlib.crc32(s) % 900000)


def benjamini_hochberg(pvals):
    """Benjamini-Hochberg FDR correction without statsmodels."""
    p = np.asarray(pvals, dtype=float)
    q = np.full(p.shape, np.nan, dtype=float)

    mask = np.isfinite(p)
    pv = p[mask]
    if pv.size == 0:
        return q

    order = np.argsort(pv)
    ranked = pv[order]
    n = len(ranked)

    adjusted = ranked * n / np.arange(1, n + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0, 1)

    out = np.empty_like(adjusted)
    out[order] = adjusted
    q[mask] = out
    return q


def load_feature(dataset, feature):
    path = FEATURE_BASE / dataset / f"{feature}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing feature file: {path}")

    df = pd.read_csv(path, dtype={"participant_id": str})
    df["dataset"] = df["dataset"].astype(str)
    df["participant_id"] = df["participant_id"].astype(str)

    feature_cols = [c for c in df.columns if c not in META_COLS]
    X = df[feature_cols].apply(pd.to_numeric, errors="coerce")
    return df[["dataset", "participant_id"]].copy(), X


def merge_base_features(dataset, features):
    ids = None
    parts = []

    for feat in features:
        id_df, X = load_feature(dataset, feat)

        if ids is None:
            ids = id_df.copy()
        elif not ids["participant_id"].tolist() == id_df["participant_id"].tolist():
            raise ValueError(f"ID order mismatch for {dataset}/{feat}")

        parts.append(X.reset_index(drop=True))

    return ids, pd.concat(parts, axis=1)


def load_labels(dataset):
    lab = pd.read_csv(LABEL_FILE, dtype={"participant_id": str})
    lab = lab[lab["dataset"] == dataset].copy()
    return lab[["dataset", "participant_id", "y"]]


def align_X_y(dataset, X_ids, X):
    lab = load_labels(dataset)
    tmp = X_ids.copy()
    tmp["_row"] = np.arange(len(tmp))

    merged = lab.merge(tmp, on=["dataset", "participant_id"], how="inner")

    if len(merged) != len(lab):
        raise ValueError(
            f"Label-feature alignment failed for {dataset}: {len(merged)} vs {len(lab)}"
        )

    return (
        merged[["dataset", "participant_id"]],
        X.iloc[merged["_row"].values].reset_index(drop=True),
        merged["y"].astype(int).values,
    )


def make_liwc_control(control_type, X_train_liwc, X_test_liwc, rng):
    train = np.asarray(X_train_liwc, dtype=float)
    test = np.asarray(X_test_liwc, dtype=float)

    imp = SimpleImputer(strategy="median")
    train_filled = imp.fit_transform(train)
    test_filled = imp.transform(test)

    n_train, n_features = train_filled.shape
    n_test = test_filled.shape[0]

    if control_type == "intact":
        return train, test

    if control_type == "pca":
        k = min(n_features, max(1, n_train - 1))

        scaler = StandardScaler()
        Z_train = scaler.fit_transform(train_filled)
        Z_test = scaler.transform(test_filled)

        pca = PCA(n_components=k, random_state=0)
        T_train = pca.fit_transform(Z_train)
        T_test = pca.transform(Z_test)

        if k < n_features:
            T_train = np.pad(T_train, ((0, 0), (0, n_features - k)), mode="constant")
            T_test = np.pad(T_test, ((0, 0), (0, n_features - k)), mode="constant")

        return T_train, T_test

    if control_type == "shuffled":
        return (
            train_filled[rng.permutation(n_train)],
            train_filled[rng.integers(0, n_train, size=n_test)],
        )

    if control_type == "random":
        train_ctrl = np.zeros_like(train_filled)
        test_ctrl = np.zeros_like(test_filled)

        for j in range(n_features):
            col = train_filled[:, j]
            train_ctrl[:, j] = rng.choice(col, size=n_train, replace=True)
            test_ctrl[:, j] = rng.choice(col, size=n_test, replace=True)

        return train_ctrl, test_ctrl

    raise ValueError(control_type)


def fit_eval(clf, X_train, X_test, y_train, y_test):
    pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", clf),
        ]
    )

    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

    try:
        scores = pipe.decision_function(X_test)
        auroc = roc_auc_score(y_test, scores)
    except Exception:
        auroc = np.nan

    return {
        "macro_f1": f1_score(y_test, y_pred, average="macro", zero_division=0),
        "recall_pos": recall_score(y_test, y_pred, pos_label=1, zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y_test, y_pred),
        "auroc": auroc,
    }


def exact_signflip_p(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if len(values) == 0:
        return np.nan

    obs = values.mean()
    stats = []

    for signs in product([-1, 1], repeat=len(values)):
        stats.append((values * np.asarray(signs)).mean())

    stats = np.asarray(stats)
    return float((np.abs(stats) >= abs(obs) - 1e-12).mean())


def bootstrap_ci(values, n_boot=20000, seed=13):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if len(values) == 0:
        return np.nan, np.nan

    rng = np.random.default_rng(seed)
    boots = []

    for _ in range(n_boot):
        boots.append(rng.choice(values, size=len(values), replace=True).mean())

    return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def condition_name(route, variant):
    if variant == "base":
        return f"{route}__base"
    return f"{route}__{variant}_liwc"


def aggregate_draws(raw_df):
    """
    For base, intact, and PCA, there is one draw per fold.
    For shuffled and random, average over control draws within each fold.
    """
    group_cols = [
        "dataset",
        "route",
        "condition",
        "liwc_variant",
        "model",
        "fold",
        "n_total",
        "n_train",
        "n_test",
        "n_features",
        "n_liwc_features",
        "class_counts",
    ]

    agg = raw_df.groupby(group_cols, as_index=False).agg(
        macro_f1=("macro_f1", "mean"),
        recall_pos=("recall_pos", "mean"),
        balanced_accuracy=("balanced_accuracy", "mean"),
        auroc=("auroc", "mean"),
        n_control_draws=("control_draw", "nunique"),
        macro_f1_draw_sd=("macro_f1", "std"),
        recall_pos_draw_sd=("recall_pos", "std"),
        auroc_draw_sd=("auroc", "std"),
    )

    return agg


def summarize(avg_df):
    summary = (
        avg_df.groupby(["dataset", "route", "condition", "liwc_variant", "model"])
        .agg(
            n_folds=("fold", "count"),
            n_total=("n_total", "first"),
            n_features=("n_features", "first"),
            n_control_draws=("n_control_draws", "max"),
            macro_f1_mean=("macro_f1", "mean"),
            macro_f1_std=("macro_f1", "std"),
            recall_pos_mean=("recall_pos", "mean"),
            recall_pos_std=("recall_pos", "std"),
            balanced_accuracy_mean=("balanced_accuracy", "mean"),
            balanced_accuracy_std=("balanced_accuracy", "std"),
            auroc_mean=("auroc", "mean"),
            auroc_std=("auroc", "std"),
            macro_f1_draw_sd_mean=("macro_f1_draw_sd", "mean"),
            recall_pos_draw_sd_mean=("recall_pos_draw_sd", "mean"),
            auroc_draw_sd_mean=("auroc_draw_sd", "mean"),
        )
        .reset_index()
    )

    return summary


def make_test_table(summary_df, outdir):
    rows = []

    def get_mean(dataset, model, route, variant, metric):
        s = summary_df[
            (summary_df["dataset"] == dataset)
            & (summary_df["model"] == model)
            & (summary_df["route"] == route)
            & (summary_df["liwc_variant"] == variant)
        ]

        if len(s) != 1:
            raise ValueError(
                f"Expected one row for dataset={dataset}, model={model}, "
                f"route={route}, variant={variant}; got {len(s)}"
            )

        return float(s[f"{metric}_mean"].iloc[0])

    pairs = [(dataset, model) for dataset in DATASETS for model in MODELS.keys()]
    comparisons = []

    # Route base comparisons.
    for metric in ["macro_f1", "recall_pos", "auroc"]:
        comparisons.append(
            {
                "comparison_family": "route_base",
                "comparison": "WhiSPA base minus Explicit base",
                "route_a": "integrated_whispa",
                "variant_a": "base",
                "route_b": "explicit_xlsr_sbert_psycemb",
                "variant_b": "base",
                "metric": metric,
            }
        )

    # Additive LIWC comparisons.
    for route, display in [
        ("explicit_xlsr_sbert_psycemb", "Explicit"),
        ("integrated_whispa", "WhiSPA"),
    ]:
        for metric in ["macro_f1", "recall_pos", "auroc"]:
            comparisons.append(
                {
                    "comparison_family": "additive_liwc",
                    "comparison": f"{display}+intact LIWC minus {display} base",
                    "route_a": route,
                    "variant_a": "intact",
                    "route_b": route,
                    "variant_b": "base",
                    "metric": metric,
                }
            )

    # LIWC control comparisons.
    for route, display in [
        ("explicit_xlsr_sbert_psycemb", "Explicit"),
        ("integrated_whispa", "WhiSPA"),
    ]:
        for control in ["pca", "shuffled", "random"]:
            comparisons.append(
                {
                    "comparison_family": "liwc_control",
                    "comparison": f"{display}+intact LIWC minus {display}+{control} LIWC",
                    "route_a": route,
                    "variant_a": "intact",
                    "route_b": route,
                    "variant_b": control,
                    "metric": "macro_f1",
                }
            )

    detail_rows = []

    for comp in comparisons:
        diffs = []

        for dataset, model in pairs:
            a = get_mean(
                dataset,
                model,
                comp["route_a"],
                comp["variant_a"],
                comp["metric"],
            )
            b = get_mean(
                dataset,
                model,
                comp["route_b"],
                comp["variant_b"],
                comp["metric"],
            )

            d = a - b
            diffs.append(d)

            detail_rows.append(
                {
                    **comp,
                    "dataset": dataset,
                    "model": model,
                    "diff": d,
                }
            )

        diffs = np.asarray(diffs, dtype=float)
        lo, hi = bootstrap_ci(diffs, seed=stable_seed("ci", comp["comparison"], comp["metric"]))

        rows.append(
            {
                "comparison_family": comp["comparison_family"],
                "comparison": comp["comparison"],
                "metric": comp["metric"],
                "n_pairs": len(diffs),
                "mean_diff": diffs.mean(),
                "ci_low": lo,
                "ci_high": hi,
                "p": exact_signflip_p(diffs),
            }
        )

    test_df = pd.DataFrame(rows)
    test_df["q"] = benjamini_hochberg(test_df["p"].values)
    test_df["direction"] = np.where(test_df["q"] < 0.05, "corrected", "n.s.")

    detail_df = pd.DataFrame(detail_rows)

    # Dataset-blocked macro-F1 table.
    blocked_rows = []
    detail_macro = detail_df[detail_df["metric"] == "macro_f1"].copy()

    blocked = (
        detail_macro.groupby(
            ["comparison_family", "comparison", "metric", "dataset"],
            as_index=False,
        )["diff"]
        .mean()
    )

    for (fam, comp, metric), sub in blocked.groupby(
        ["comparison_family", "comparison", "metric"]
    ):
        vals = sub["diff"].values.astype(float)
        lo, hi = bootstrap_ci(vals, seed=stable_seed("blocked", comp, metric))

        blocked_rows.append(
            {
                "comparison_family": fam,
                "comparison": comp,
                "metric": metric,
                "n_datasets": len(vals),
                "mean_diff": vals.mean(),
                "ci_low": lo,
                "ci_high": hi,
                "p": exact_signflip_p(vals),
            }
        )

    blocked_df = pd.DataFrame(blocked_rows)
    blocked_df["q"] = benjamini_hochberg(blocked_df["p"].values)
    blocked_df["direction"] = np.where(blocked_df["q"] < 0.05, "corrected", "n.s.")

    outdir_tables = outdir / "tables"
    outdir_tables.mkdir(parents=True, exist_ok=True)

    test_df.to_csv(outdir_tables / "liwc_substitution_tests_multidraw.csv", index=False)
    detail_df.to_csv(outdir_tables / "liwc_substitution_pair_details_multidraw.csv", index=False)
    blocked_df.to_csv(
        outdir_tables / "dataset_blocked_liwc_macro_f1_tests_multidraw.csv",
        index=False,
    )

    tex_df = test_df.copy()
    for c in ["mean_diff", "ci_low", "ci_high", "p", "q"]:
        tex_df[c] = tex_df[c].map(lambda x: f"{x:.3f}" if pd.notna(x) else "NA")

    tex_df.to_latex(
        outdir_tables / "liwc_substitution_tests_multidraw.tex",
        index=False,
        escape=True,
    )

    tex_block = blocked_df.copy()
    for c in ["mean_diff", "ci_low", "ci_high", "p", "q"]:
        tex_block[c] = tex_block[c].map(lambda x: f"{x:.3f}" if pd.notna(x) else "NA")

    tex_block.to_latex(
        outdir_tables / "dataset_blocked_liwc_macro_f1_tests_multidraw.tex",
        index=False,
        escape=True,
    )

    return test_df, blocked_df


def run_experiment(n_draws, tag):
    outdir = ROOT / f"reviewer_robustness_outputs/liwc_substitution_v1_multidraw_20260512_{tag}"
    outdir.mkdir(parents=True, exist_ok=True)

    rows = []

    for dataset in DATASETS:
        wh_ids, X_wh = merge_base_features(dataset, ["whispa_small"])
        ex_ids, X_ex = merge_base_features(dataset, ["xlsr", "sbert", "psycemb"])
        liwc_ids, X_liwc = merge_base_features(dataset, ["liwc"])

        _, X_wh, y = align_X_y(dataset, wh_ids, X_wh)
        _, X_ex, y2 = align_X_y(dataset, ex_ids, X_ex)
        _, X_liwc, y3 = align_X_y(dataset, liwc_ids, X_liwc)

        if not np.array_equal(y, y2) or not np.array_equal(y, y3):
            raise ValueError(f"Y mismatch for {dataset}")

        class_counts = pd.Series(y).value_counts().sort_index().to_dict()
        n_splits = min(5, min(class_counts.values()))
        cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=10, random_state=42)

        X_wh_np = X_wh.values
        X_ex_np = X_ex.values
        X_liwc_np = X_liwc.values

        for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X_wh_np, y), start=1):
            y_train, y_test = y[train_idx], y[test_idx]
            L_train, L_test = X_liwc_np[train_idx], X_liwc_np[test_idx]

            for route_name, base_np in [
                ("integrated_whispa", X_wh_np),
                ("explicit_xlsr_sbert_psycemb", X_ex_np),
            ]:
                B_train, B_test = base_np[train_idx], base_np[test_idx]

                variant_draws = {
                    "base": [0],
                    "intact": [0],
                    "pca": [0],
                    "shuffled": list(range(n_draws)),
                    "random": list(range(n_draws)),
                }

                for variant, draws in variant_draws.items():
                    for draw in draws:
                        if variant == "base":
                            X_train, X_test = B_train, B_test
                        else:
                            rng = np.random.default_rng(
                                stable_seed(dataset, route_name, variant, fold_idx, draw)
                            )
                            C_train, C_test = make_liwc_control(variant, L_train, L_test, rng)
                            X_train = np.concatenate([B_train, C_train], axis=1)
                            X_test = np.concatenate([B_test, C_test], axis=1)

                        for model_name, clf in MODELS.items():
                            metrics = fit_eval(clf, X_train, X_test, y_train, y_test)

                            rows.append(
                                {
                                    "dataset": dataset,
                                    "route": route_name,
                                    "condition": condition_name(route_name, variant),
                                    "liwc_variant": variant,
                                    "control_draw": draw,
                                    "model": model_name,
                                    "fold": fold_idx,
                                    "n_total": len(y),
                                    "n_train": len(train_idx),
                                    "n_test": len(test_idx),
                                    "n_features": X_train.shape[1],
                                    "n_liwc_features": X_liwc_np.shape[1],
                                    "class_counts": json.dumps(
                                        {int(k): int(v) for k, v in class_counts.items()}
                                    ),
                                    **metrics,
                                }
                            )

            if fold_idx % 10 == 0:
                print(f"Finished {dataset} fold {fold_idx}", flush=True)

        print(f"Finished dataset {dataset}", flush=True)

    raw_df = pd.DataFrame(rows)
    raw_out = outdir / "liwc_substitution_fold_results_raw_multidraw.csv"
    raw_df.to_csv(raw_out, index=False)

    avg_df = aggregate_draws(raw_df)
    avg_out = outdir / "liwc_substitution_fold_results_averaged_controls.csv"
    avg_df.to_csv(avg_out, index=False)

    summary = summarize(avg_df)
    summary_out = outdir / "liwc_substitution_summary_multidraw.csv"
    summary.to_csv(summary_out, index=False)

    test_df, blocked_df = make_test_table(summary, outdir)

    print("\nSaved:", flush=True)
    print(raw_out, flush=True)
    print(avg_out, flush=True)
    print(summary_out, flush=True)
    print(outdir / "tables/liwc_substitution_tests_multidraw.csv", flush=True)
    print(outdir / "tables/dataset_blocked_liwc_macro_f1_tests_multidraw.csv", flush=True)

    print("\nMain multidraw tests:", flush=True)
    print(test_df.to_string(index=False), flush=True)

    print("\nDataset-blocked multidraw macro-F1 tests:", flush=True)
    print(blocked_df.to_string(index=False), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_draws", type=int, default=10)
    parser.add_argument("--tag", type=str, default=None)
    args = parser.parse_args()

    tag = args.tag or f"n{args.n_draws}"
    run_experiment(args.n_draws, tag)


if __name__ == "__main__":
    main()
