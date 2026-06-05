# Scripts


This directory contains the scripts used for the reproducibility release.

## Main controlled-substitution pipeline

- `run_liwc_substitution_v1_multidraw.py`  
  Runs the controlled LIWC substitution experiments with intact, PCA, shuffled, and random LIWC variants. Shuffled and random controls use multiple draws per fold.

- `make_dataset_blocked_liwc_tests.py`  
  Computes the dataset-blocked primary contrasts reported in Table 3, including exact sign-flip p-values and Benjamini-Hochberg q-values over the eight prespecified primary contrasts.

- `run_liwc_sensitivity_from_existing_results.py`  
  Produces compact sensitivity checks for the primary controlled-substitution results.

## Interpretability analyses

- `run_liwc_interpretability_smd.py`  
  Computes theory-guided LIWC domain standardized mean differences and generates the LIWC domain heatmap.

- `run_liwc_interpretability_bootstrap.py`  
  Computes bootstrap confidence intervals for the LIWC domain and category-level descriptive analyses.

## Feature-family and dataset summaries

- `make_feature_family_heatmap.py`  
  Generates the feature-family upper-envelope macro-F1 heatmap.

- `make_dataset_liwc_table.py`  
  Generates dataset-level controlled LIWC substitution summaries.

- `make_liwc_length_coverage_summary_v1.py`  
  Generates participant-level transcript length and LIWC coverage summaries.

## Notes

The repository does not include raw datasets, transcripts, audio, proprietary LIWC resources, or restricted feature matrices. Paths must be configured locally.
