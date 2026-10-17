# F1 Podium Prediction (CSEN 903, Milestone 1)

For each driver in each race, predict whether they finish on the podium (top 3) using only information available after qualifying and before the race starts.

## Contents

| Path | What it is |
|---|---|
| `F1_Podium_Prediction.ipynb` | Run-all notebook: EDA, cleaning, data-engineering questions Q1-Q3, features, split, models, ablation, XAI, inference function |
| `report/F1_Podium_Report.pdf` | Analytical report (HTML source and figures alongside) |
| `dataset/archive/` | The 14 raw Kaggle/Ergast CSVs (never modified; cleaning is done on copies) |
| `DATASET.md` | Guide to the dataset's tables and joins |
| `docs/` | Project handouts |

## Running

Python 3.12 is recommended (TensorFlow 2.21 has no build for Python 3.14).

```bash
pip install -r requirements.txt
jupyter nbconvert --to notebook --execute F1_Podium_Prediction.ipynb
```

or open the notebook and use Run All (about 10 minutes on CPU). The notebook finds the data in `dataset/archive/` or, on Kaggle, under `/kaggle/input/`. The random seed is fixed, but the Keras neural-network numbers can move in the third decimal between machines.

## Method in brief

- **Split:** by time. Train 1950-2017, validation 2018-2021, test 2022-2024.
- **Models:** logistic regression (baseline), random forest, gradient boosting, shallow Keras network. Hyperparameters and thresholds are chosen on validation only.
- **Primary metric:** PR-AUC, because only about 1 row in 7 is a podium. F1, precision, recall and ROC-AUC are reported alongside it.
- **Explainability:** permutation importance and SHAP (global), LIME (local). These describe the model, not causes.
