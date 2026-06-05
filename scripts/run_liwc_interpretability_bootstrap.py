from pathlib import Path
import re
import numpy as np
import pandas as pd

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

def smd_from_arrays(dep, ctl):
    dep = np.asarray(dep, dtype=float)
    ctl = np.asarray(ctl, dtype=float)
    dep = dep[np.isfinite(dep)]
    ctl = ctl[np.isfinite(ctl)]
    n1, n0 = len(dep), len(ctl)
    if n1 < 2 or n0 < 2:
        return np.nan
    s1, s0 = dep.std(ddof=1), ctl.std(ddof=1)
    pooled = np.sqrt(((n1 - 1) * s1**2 + (n0 - 1) * s0**2) / (n1 + n0 - 2))
    if pooled == 0 or not np.isfinite(pooled):
        return np.nan
    return (dep.mean() - ctl.mean()) / pooled

def bootstrap_smd(dep, ctl, n_boot=5000, seed=13):
    rng = np.random.default_rng(seed)
    dep = np.asarray(dep, dtype=float)
    ctl = np.asarray(ctl, dtype=float)
    dep = dep[np.isfinite(dep)]
    ctl = ctl[np.isfinite(ctl)]

    obs = smd_from_arrays(dep, ctl)
    if not np.isfinite(obs) or len(dep) < 2 or len(ctl) < 2:
        return obs, np.nan, np.nan

    vals = []
    for _ in range(n_boot):
        bd = rng.choice(dep, size=len(dep), replace=True)
        bc = rng.choice(ctl, size=len(ctl), replace=True)
        vals.append(smd_from_arrays(bd, bc))

    vals = np.asarray(vals, dtype=float)
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return obs, np.nan, np.nan
    return obs, float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))

def bootstrap_domain(df, matched_cols, n_boot=5000, seed=13):
    rng = np.random.default_rng(seed)
    dep_df = df[df["y"] == 1].copy()
    ctl_df = df[df["y"] == 0].copy()

    def domain_smd(d1, d0):
        vals = []
        for col in matched_cols:
            val = smd_from_arrays(d1[col].values, d0[col].values)
            if np.isfinite(val):
                vals.append(val)
        return np.nan if not vals else float(np.mean(vals))

    obs = domain_smd(dep_df, ctl_df)
    if len(dep_df) < 2 or len(ctl_df) < 2 or not np.isfinite(obs):
        return obs, np.nan, np.nan

    vals = []
    dep_idx = np.arange(len(dep_df))
    ctl_idx = np.arange(len(ctl_df))
    for _ in range(n_boot):
        bd = dep_df.iloc[rng.choice(dep_idx, size=len(dep_idx), replace=True)]
        bc = ctl_df.iloc[rng.choice(ctl_idx, size=len(ctl_idx), replace=True)]
        vals.append(domain_smd(bd, bc))

    vals = np.asarray(vals, dtype=float)
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return obs, np.nan, np.nan
    return obs, float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))

labels = pd.read_csv(LABEL_FILE)
labels["dataset"] = labels["dataset"].astype(str).str.lower().str.strip()
labels["participant_id_norm"] = labels["participant_id"].apply(norm_id)

category_rows = []
domain_rows = []
audit_rows = []

for ds in DATASETS:
    liwc_path = FEATURE_ROOT / ds / "liwc.csv"
    liwc = pd.read_csv(liwc_path)
    liwc["dataset"] = liwc["dataset"].astype(str).str.lower().str.strip()
    liwc["participant_id_norm"] = liwc["participant_id"].apply(norm_id)

    lab = labels[labels["dataset"].eq(ds)].copy()
    merged = liwc.merge(
        lab[["dataset", "participant_id_norm", "y", "score", "score_name", "label_rule"]],
        on=["dataset", "participant_id_norm"],
        how="inner",
    )

    audit_rows.append({
        "dataset": ds,
        "dataset_display": DATASET_DISPLAY[ds],
        "n_merged": len(merged),
        "n_depressed": int((merged["y"] == 1).sum()),
        "n_non_depressed": int((merged["y"] == 0).sum()),
        "label_rule": lab["label_rule"].iloc[0] if len(lab) else "",
    })

    for domain, cats in DOMAIN_MAP.items():
        matched_cols = []
        matched_names = []

        for cat_name, aliases in cats.items():
            col = find_col(merged.columns, aliases)
            if col is None:
                continue
            matched_cols.append(col)
            matched_names.append(cat_name)

            dep = pd.to_numeric(merged.loc[merged["y"] == 1, col], errors="coerce").values
            ctl = pd.to_numeric(merged.loc[merged["y"] == 0, col], errors="coerce").values
            obs, lo, hi = bootstrap_smd(dep, ctl, n_boot=5000, seed=1000 + hash((ds, domain, cat_name)) % 100000)

            category_rows.append({
                "dataset": ds,
                "dataset_display": DATASET_DISPLAY[ds],
                "domain": domain,
                "category": cat_name,
                "matched_column": col,
                "smd": obs,
                "ci_low": lo,
                "ci_high": hi,
                "n_depressed": int((merged["y"] == 1).sum()),
                "n_non_depressed": int((merged["y"] == 0).sum()),
            })

        if matched_cols:
            obs, lo, hi = bootstrap_domain(
                merged,
                matched_cols,
                n_boot=5000,
                seed=2000 + hash((ds, domain)) % 100000,
            )
            domain_rows.append({
                "dataset": ds,
                "dataset_display": DATASET_DISPLAY[ds],
                "domain": domain,
                "smd": obs,
                "ci_low": lo,
                "ci_high": hi,
                "n_categories": len(matched_cols),
                "categories": "; ".join(matched_names),
            })

audit = pd.DataFrame(audit_rows)
cat = pd.DataFrame(category_rows)
dom = pd.DataFrame(domain_rows)

audit.to_csv(OUTDIR / "liwc_bootstrap_merge_audit.csv", index=False)
cat.to_csv(OUTDIR / "liwc_category_smd_bootstrap_ci.csv", index=False)
dom.to_csv(OUTDIR / "liwc_domain_smd_bootstrap_ci.csv", index=False)

# Compact appendix table: domain-level SMD with CI.
dom_table = dom.copy()
dom_table["Dataset"] = dom_table["dataset_display"]
dom_table["Domain"] = dom_table["domain"]
dom_table["SMD [95% CI]"] = dom_table.apply(
    lambda r: f"{r['smd']:.2f} [{r['ci_low']:.2f}, {r['ci_high']:.2f}]"
    if np.isfinite(r["ci_low"]) else f"{r['smd']:.2f} [NA, NA]",
    axis=1,
)
dom_table = dom_table[["Dataset", "Domain", "SMD [95% CI]", "n_categories", "categories"]]
dom_table.to_latex(
    OUTDIR / "liwc_domain_smd_bootstrap_ci_table.tex",
    index=False,
    escape=True,
)

print("\nSaved to:", OUTDIR)
print("\nAudit:")
print(audit.to_string(index=False))

print("\nDomain bootstrap CI:")
print(dom_table.to_string(index=False))

print("\nLargest category-level effects by absolute SMD:")
print(
    cat.assign(abs_smd=cat["smd"].abs())
       .sort_values("abs_smd", ascending=False)
       .head(40)
       [["dataset_display", "domain", "category", "matched_column", "smd", "ci_low", "ci_high"]]
       .to_string(index=False)
)
