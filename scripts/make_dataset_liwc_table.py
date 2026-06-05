from pathlib import Path
import re
import pandas as pd

ROOT = Path("reviewer_robustness_outputs/liwc_substitution_v1_stable_20260509")
INFILE = ROOT / "tables" / "dataset_model_pair_details.csv"
OUTDIR = ROOT / "tables"
OUTDIR.mkdir(parents=True, exist_ok=True)

OUT_CSV = OUTDIR / "dataset_level_controlled_liwc_macro_f1.csv"
OUT_TEX = OUTDIR / "dataset_level_controlled_liwc_macro_f1_table.tex"

df = pd.read_csv(INFILE)
df.columns = [c.strip().lower().replace("-", "_").replace(" ", "_") for c in df.columns]

print("Loaded:", INFILE)
print("Shape:", df.shape)
print("Columns:", list(df.columns))

# Keep Macro-F1 only.
df = df[df["metric"].astype(str).str.lower().eq("macro_f1")].copy()

# Keep only the two main linear classifiers.
model_norm = df["model"].astype(str).str.lower()
df = df[
    model_norm.str.contains("logreg", regex=False)
    | model_norm.str.contains("logistic", regex=False)
    | model_norm.str.contains("linear_svm", regex=False)
    | model_norm.str.contains("linear svm", regex=False)
].copy()

def norm_text(x):
    x = str(x).lower()
    x = x.replace("_", " ")
    x = re.sub(r"\s+", " ", x)
    return x.strip()

def parse_route_and_target(comp):
    s = norm_text(comp)

    if "whispa" in s:
        route = "WhiSPA"
    elif "explicit" in s or "xlsr" in s or "sbert" in s or "psy" in s:
        route = "Explicit"
    else:
        route = "Unknown"

    if "random" in s:
        target = "LIWC $-$ random"
    elif "shuff" in s:
        target = "LIWC $-$ shuffled"
    elif "pca" in s:
        target = "LIWC $-$ PCA"
    elif "base" in s:
        target = "LIWC $-$ base"
    else:
        target = "Unknown"

    return route, target

parsed = df["comparison"].apply(parse_route_and_target)
df["Route"] = parsed.apply(lambda x: x[0])
df["Target"] = parsed.apply(lambda x: x[1])

df = df[df["Route"].isin(["Explicit", "WhiSPA"])].copy()
df = df[df["Target"].isin([
    "LIWC $-$ base",
    "LIWC $-$ PCA",
    "LIWC $-$ shuffled",
    "LIWC $-$ random",
])].copy()

def clean_dataset(x):
    s = str(x).lower()
    mapping = {
        "daic": "DAIC-WOZ",
        "edaic": "E-DAIC",
        "eatd": "EATD",
        "modma": "MODMA",
        "pdch": "PDCH",
    }
    return mapping.get(s, str(x))

def clean_model(x):
    s = str(x).lower()
    if "logreg" in s or "logistic" in s:
        return "Logistic reg."
    if "linear_svm" in s or "linear svm" in s:
        return "Linear SVM"
    return str(x)

df["Dataset"] = df["dataset"].apply(clean_dataset)
df["Classifier"] = df["model"].apply(clean_model)
df["Delta"] = pd.to_numeric(df["diff"], errors="coerce")

wide = (
    df.groupby(["Dataset", "Route", "Classifier", "Target"], as_index=False)["Delta"]
      .mean()
      .pivot_table(
          index=["Dataset", "Route", "Classifier"],
          columns="Target",
          values="Delta",
          aggfunc="mean"
      )
      .reset_index()
)

wanted_cols = [
    "Dataset",
    "Route",
    "Classifier",
    "LIWC $-$ base",
    "LIWC $-$ PCA",
    "LIWC $-$ shuffled",
    "LIWC $-$ random",
]
for c in wanted_cols:
    if c not in wide.columns:
        wide[c] = pd.NA

wide = wide[wanted_cols]

dataset_order = ["DAIC-WOZ", "E-DAIC", "EATD", "MODMA", "PDCH"]
route_order = ["Explicit", "WhiSPA"]
clf_order = ["Logistic reg.", "Linear SVM"]

wide["dataset_order"] = wide["Dataset"].apply(lambda x: dataset_order.index(x) if x in dataset_order else 999)
wide["route_order"] = wide["Route"].apply(lambda x: route_order.index(x) if x in route_order else 999)
wide["clf_order"] = wide["Classifier"].apply(lambda x: clf_order.index(x) if x in clf_order else 999)

wide = wide.sort_values(["dataset_order", "route_order", "clf_order"]).drop(
    columns=["dataset_order", "route_order", "clf_order"]
)

wide.to_csv(OUT_CSV, index=False)

def fmt(x):
    if pd.isna(x):
        return ""
    return f"{float(x):.3f}"

tex_df = wide.copy()
for c in ["LIWC $-$ base", "LIWC $-$ PCA", "LIWC $-$ shuffled", "LIWC $-$ random"]:
    tex_df[c] = tex_df[c].apply(fmt)

latex = tex_df.to_latex(
    index=False,
    escape=False,
    column_format="lllrrrr",
    caption=(
        "Dataset-level macro-F1 differences for controlled LIWC substitution. "
        "Positive values favor intact LIWC. Explicit denotes the XLSR+SBERT+PsycEmb route. "
        "WhiSPA denotes the integrated WhiSPA-Small route."
    ),
    label="tab:dataset-liwc-results"
)

latex = latex.replace("\\begin{table}", "\\begin{table*}[t]")
latex = latex.replace("\\end{table}", "\\end{table*}")
latex = latex.replace("\\toprule", "\\hline")
latex = latex.replace("\\midrule", "\\hline")
latex = latex.replace("\\bottomrule", "\\hline")

OUT_TEX.write_text(latex)

print("\nSaved CSV:", OUT_CSV)
print("Saved LaTeX:", OUT_TEX)
print("\nNumber of rows:", len(wide))
print("\nPreview:")
print(wide.to_string(index=False))
