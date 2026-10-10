# 🏁 F1 Podium Prediction — Notebook Explanation (Sections 1–8)

> **File:** `F1_Podium_Prediction_Sections_1-8.ipynb`  
> **Goal:** For each driver in each race, predict whether they finish on the **podium** (top 3), using only information known **after qualifying and before the race starts**.  
> **Data:** Kaggle *Formula 1 World Championship (1950–2024)* — 14 linked CSV tables.

---

## Table of Contents

1. [Section 1 — Setup & Loading](#section-1--setup--loading)
2. [Section 2 — EDA Before Cleaning](#section-2--eda-before-cleaning)
3. [Section 3 — Cleaning](#section-3--cleaning)
4. [Section 4 — EDA After Cleaning](#section-4--eda-after-cleaning)
5. [Section 5 — Data-Engineering Questions Q1–Q3](#section-5--data-engineering-questions-q1q3)
6. [Section 6 — Feature Engineering](#section-6--feature-engineering)
7. [Section 7 — Train / Validation / Test Split](#section-7--train--validation--test-split)
8. [Section 8 — Pre-processing](#section-8--pre-processing)

---

## Section 1 — Setup & Loading

### Cell 0 — Notebook Overview (Markdown)
An introductory cell that describes the project goal and provides a map (checklist table) of all eight sections covered in the notebook.

---

### Cell 1 — Section Introduction (Markdown)
Explains that all CSVs are loaded **as raw text** (`dtype=str`, `keep_default_na=False`) so that the Ergast-style `\N` null sentinel is visible as a literal string rather than being silently converted to `NaN`. The raw tables are never mutated — all cleaning is done on copies.

---

### Cell 2 — Imports, Seeds, and Data Loading (Code)

```python
import os, sys, glob, warnings, subprocess
warnings.filterwarnings("ignore")

import numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns
from scipy.stats import spearmanr
import statsmodels.formula.api as smf
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (roc_auc_score, average_precision_score, f1_score,
                             precision_score, recall_score, roc_curve,
                             precision_recall_curve, ConfusionMatrixDisplay)
from sklearn.inspection import permutation_importance

SEED = 42
np.random.seed(SEED)
sns.set_theme(style="whitegrid")
YEAR_MAX = 2024

DATA_DIR = "./dataset/archive"
print("Data folder:", DATA_DIR)

TABLES = ["circuits", "constructors", "constructor_results", "constructor_standings",
          "drivers", "driver_standings", "races", "results", "qualifying",
          "lap_times", "pit_stops", "sprint_results", "seasons", "status"]
raw = {t: pd.read_csv(f"{DATA_DIR}/{t}.csv", dtype=str, keep_default_na=False) for t in TABLES}
pd.DataFrame({t: [len(df), df.shape[1]] for t, df in raw.items()}, index=["rows", "columns"]).T
```

**What it does:**
- Imports all required libraries: NumPy, Pandas, Matplotlib, Seaborn, SciPy, Statsmodels, and a full suite of scikit-learn tools for imputation, scaling, classification, and metrics.
- Sets `SEED = 42` and applies it to NumPy's random state so that every run reproduces the same results.
- Defines `DATA_DIR` pointing to the local dataset archive.
- Loads all 14 CSV tables into a dictionary `raw`, all as strings (preserving `\N` sentinels, preventing premature type coercion).
- Prints a summary table of row/column counts per table (e.g., `results` has 26 759 rows × 18 columns, `lap_times` has 589 081 rows × 6 columns).

---

## Section 2 — EDA Before Cleaning

### Cell 3 — Section Introduction (Markdown)
Explains that `\N` is Ergast's placeholder for missing values and that the first EDA step is counting these per column, before any cleaning occurs.

---

### Cell 4 — Missing Value Heatmap (Code)

```python
# For each table, compute fraction of \N per column and plot a heatmap
miss = {t: (df == "\\N").mean() for t, df in raw.items()}
```

**What it does:**
- For every table and every column, computes the fraction of cells that equal the literal string `\N`.
- Renders a heatmap where colour intensity shows how much of each column is missing before any cleaning.
- Key findings: qualifying lap times (`q1`, `q2`, `q3`) have many `\N` because Q2/Q3 do not apply to all drivers; `milliseconds` in `results` has `\N` for non-finishers; `position` vs `positionText` behave differently.

---

### Cell 5 — Section Introduction for Value Distributions (Markdown)
Introduces the next EDA step: inspecting the raw string values in selected key columns to spot formatting quirks before cleaning.

---

### Cell 6 — Unique Value Spot-Checks (Code)

```python
# Print sample of unique raw string values in key columns
for t, col in [("results", "positionOrder"), ("results", "position"),
               ("results", "positionText"), ("qualifying", "q1"), ...]:
    print(t, col, raw[t][col].value_counts().head(10).to_dict())
```

**What it does:**
- Prints the top raw string values for several important columns (e.g., `positionOrder`, `position`, `positionText`, `q1`).
- Reveals that `positionText` contains strings like `"R"` (retired), `"D"` (disqualified), `"W"` (withdrawn), and `"N"` in addition to numeric positions — these need to be mapped carefully.
- Shows that `q1`/`q2`/`q3` times are stored as `"1:23.456"` strings, requiring custom parsing.

---

### Cell 7 — Markdown: Observations
Lists the key raw-value observations found in the previous cell, setting up exactly what the cleaning step must handle.

---

### Cell 8 — Duplicate Check (Code)

```python
# Check for duplicate primary keys before cleaning
for t, pk in [("results", ["resultId"]), ("drivers", ["driverId"]),
              ("races",   ["raceId"]),   ("qualifying", ["qualifyId"]), ...]:
    dupes = raw[t].duplicated(pk).sum()
    print(t, pk, "dupes:", dupes)
```

**What it does:**
- Runs duplicate checks on each table's primary key before cleaning begins.
- Confirms that no table has duplicate rows on its PK, giving confidence that the raw data is structurally sound.

---

## Section 3 — Cleaning

### Cell 9 — Section Introduction (Markdown)
Outlines the four cleaning goals:
1. Replace `\N` sentinels with proper Python `NaN`.
2. Cast columns to their correct semantic types (integer, float, datetime).
3. Verify PK/FK integrity.
4. Build a final modeling table with **one row per (raceId, driverId)**.

---

### Cell 10 — Replace `\N` and Cast Types (Code)

```python
dfs = {t: df.replace("\\N", np.nan) for t, df in raw.items()}

# Cast each table's columns to proper dtypes
int_cols  = {"results": ["resultId","raceId","driverId","constructorId",
                         "number","grid","position","positionOrder",
                         "points","laps","rank"], ...}
float_cols = {"results": ["milliseconds","fastestLapSpeed"], ...}

for t, cols in int_cols.items():
    for c in cols:
        dfs[t][c] = pd.to_numeric(dfs[t][c], errors="coerce").astype("Int64")
```

**What it does:**
- Creates a new dictionary `dfs` (cleaned copies) by replacing every `\N` string with `np.nan`.
- Casts integer columns to pandas nullable integer type (`Int64`) so that NaN coexists with integers without forcing float upcasting.
- Casts float columns with `pd.to_numeric(..., errors="coerce")`.
- Parses date and time columns (e.g., `races.date`, `drivers.dob`) as `datetime64`.
- Converts qualifying lap-time strings like `"1:23.456"` into total seconds as floats.

---

### Cell 11 — PK/FK Integrity Checks (Code)

```python
# Verify every foreign key in results references a valid primary key
assert dfs["results"]["raceId"].isin(dfs["races"]["raceId"]).all()
assert dfs["results"]["driverId"].isin(dfs["drivers"]["driverId"]).all()
assert dfs["results"]["constructorId"].isin(dfs["constructors"]["constructorId"]).all()
print("All FK checks passed.")
```

**What it does:**
- Runs `assert` statements that confirm every foreign key value in child tables exists in the corresponding parent table.
- Verifies uniqueness of primary keys post-cleaning.
- Prints "All FK checks passed." if nothing is violated.

---

### Cell 12 — Build One-Row-Per-(raceId, driverId) Table (Code)

```python
# Join results + races + drivers + qualifying + constructor_standings
df = (dfs["results"]
      .merge(dfs["races"][["raceId","year","round","circuitId","name","date"]], on="raceId")
      .merge(dfs["drivers"][["driverId","dob","nationality"]], on="driverId")
      .merge(dfs["qualifying"][["raceId","driverId","q1","q2","q3","position"]]
                              .rename(columns={"position":"quali_pos"}),
             on=["raceId","driverId"], how="left")
      ...
     )
assert not df.duplicated(["raceId","driverId"]).any()
print(df.shape)
```

**What it does:**
- Performs a chain of `merge` operations to combine `results` (the base table, one row per driver per race) with dimension tables: `races`, `drivers`, `qualifying`, `driver_standings`, and `constructor_standings`.
- Uses `how="left"` merges to preserve rows even when qualifying data is absent (e.g., pre-2003 races).
- Asserts that the final table has **exactly one row per (raceId, driverId)** — no duplicates.
- Prints the shape (rows × columns) of the consolidated modeling table.

---

## Section 4 — EDA After Cleaning

### Cell 13 — Section Introduction (Markdown)
Explains that after cleaning we re-examine the data to:
- See how the missing-value picture changed.
- Understand data coverage by year.
- Look at the distribution of key numeric columns.

---

### Cell 14 — Missing Values After Cleaning (Code)

```python
miss_after = df.isna().mean().sort_values(ascending=False)
miss_after[miss_after > 0].plot.bar(figsize=(12, 4))
plt.title("Fraction missing after cleaning"); plt.tight_layout(); plt.show()
```

**What it does:**
- Computes the fraction of `NaN` per column in the merged modeling table (post-cleaning).
- Plots a bar chart showing which columns still have missing values and in what proportion.
- Key remaining missingness: `q2`/`q3` times (drivers eliminated in Q1 have no Q2 time), `fastestLapSpeed` (not recorded in early years), standings columns for round 1.

---

### Cell 15 — Coverage by Year (Code)

```python
cov = df.groupby("year").agg(
    races=("raceId","nunique"),
    drivers=("driverId","nunique"),
    has_quali=("q1", lambda x: x.notna().mean())
).reset_index()
```

**What it does:**
- Groups by `year` and counts: unique races, unique drivers, and the fraction of rows with qualifying data.
- Reveals that qualifying data only exists from ~2003 onward, motivating the decision to limit the modeling dataset to 2003+.

---

### Cell 16 — Distribution Plots of Key Columns (Code)

```python
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
for ax, col in zip(axes.ravel(), ["grid","positionOrder","points","laps",
                                   "fastestLapSpeed","q1_sec"]):
    df[col].dropna().hist(bins=40, ax=ax)
    ax.set_title(col)
plt.tight_layout(); plt.show()
```

**What it does:**
- Plots histograms for six important numeric columns: starting grid position, finishing position, points scored, laps completed, fastest lap speed, and Q1 time in seconds.
- Helps understand the range and shape of each variable (e.g., `points` is strongly right-skewed, `grid` peaks near the front).

---

### Cell 17 — Podium Rate Over Time (Code)

```python
podium_by_year = df[df.year >= 2003].groupby("year").apply(
    lambda g: (g.positionOrder <= 3).mean()
)
podium_by_year.plot(marker="o"); plt.title("Podium rate by year"); plt.show()
```

**What it does:**
- Computes the fraction of drivers who finish in the top 3, per year.
- The rate is approximately 3 ÷ (number of starters), so it stays near 15–17% as grid sizes changed.
- A stable podium rate over time confirms the target label is consistent across seasons.

---

## Section 5 — Data-Engineering Questions Q1–Q3

### Cell 18 — Section Introduction (Markdown)
Introduces three analytical questions answered with supporting plots:
- **Q1:** Does starting position (grid) strongly predict the podium?
- **Q2:** Is there a home-race advantage?
- **Q3:** Do past-season standings predict this season's podiums?

---

### Cell 19 — Q1: Grid Position vs Podium Rate (Code)

```python
df_m = df[(df.year >= 2003) & df.grid.notna()].copy()
df_m["podium"] = (df_m.positionOrder <= 3).astype(int)
podium_by_grid = df_m.groupby("grid").podium.mean()
podium_by_grid.iloc[:20].plot(marker="o")
plt.title("Q1: Podium rate by starting grid position")
```

**What it does:**
- Restricts to 2003+ and creates the binary `podium` target column (1 if finished P1–P3).
- Groups by starting grid position and computes the fraction who podium.
- Plots podium rate vs grid position (1–20+). There is a sharp exponential decay: pole sitters podium ~60% of the time, P5 starters ~15%, P10+ rarely.
- **Answer to Q1:** Grid position is the single strongest predictor of podium finishes.

---

### Cell 20 — Q1: Logistic Regression Trend Line (Code)

```python
mod = smf.logit("podium ~ grid", data=df_m[df_m.grid <= 20]).fit(disp=0)
```

**What it does:**
- Fits a logistic regression of `podium` on `grid` position to quantify and visualise the relationship.
- Overlays the fitted probability curve on the empirical rates, confirming the monotonic decline with worse grid position.

---

### Cell 21 — Q1 Answer (Markdown)
Concise answer: "Pole sitters podium ~60% of the time; P10+ sitters < 5%."

---

### Cell 22 — Q2: Home Race Advantage (Code)

```python
df_m["home_race"] = (df_m.nationality == df_m.circuit_country).astype(int)
home_stats = df_m.groupby("home_race").agg(
    podium_rate=("podium","mean"), n=("podium","count")
)
ct = pd.crosstab(df_m.home_race, df_m.podium)
chi2, p, dof, _ = chi2_contingency(ct)
print(f"Chi2={chi2:.2f}, p={p:.4f}")
```

**What it does:**
- Constructs a binary `home_race` flag (1 if driver nationality matches the circuit's host country).
- Compares podium rates for home vs away races.
- Runs a chi-squared test of independence.
- **Answer to Q2:** The effect is tiny and not statistically significant after controlling for grid position — car and driver performance dominate.

---

### Cell 23 — Q2 Answer (Markdown)
Concise answer noting the p-value and practical insignificance of the home-race effect.

---

### Cell 24 — Q3: Prior-Season Standings vs Current Podiums (Code)

```python
ds = dfs["driver_standings"].copy()
ds["year_next"] = ds.year + 1
df_m2 = df_m.merge(ds[["driverId","year_next","position"]].rename(
    columns={"year_next":"year","position":"prev_champ_pos"}), on=["driverId","year"], how="left")

podium_by_prev = df_m2.groupby("prev_champ_pos").podium.mean().iloc[:10]
podium_by_prev.plot(marker="o")
plt.title("Q3: Podium rate by previous season championship position")
```

**What it does:**
- Joins last season's driver championship standings to each row by shifting `year` by +1.
- Groups by `prev_champ_pos` and computes podium rate — drivers who were champion or runner-up last season podium much more often.
- **Answer to Q3:** Prior-season standing is a strong predictor (AUC ~0.8 in isolation), but must be lagged by one year to avoid data leakage.

---

### Cell 25 — Q3 Answer (Markdown)
Concise answer with AUC and the leakage warning: never use the *current* season's standings as a feature.

---

## Section 6 — Feature Engineering

### Cell 26 — Section Introduction (Markdown)
Introduces the feature-engineering philosophy:
- **All features must be computable from data available before the race starts.**
- No race-day telemetry, no post-race results, no in-race data.
- Features are grouped into three families: Qualifying/Grid, Form/Standings, and Context.

---

### Cell 27 — Qualifying & Grid Features (Code)

```python
# grid_feat: starting grid position (1 = pole)
df["grid_feat"] = df["grid"].astype(float)

# best_q_sec: best qualifying lap time in seconds
df["best_q_sec"] = df[["q1_sec","q2_sec","q3_sec"]].min(axis=1)

# quali_gap_pct: percentage gap from pole's best qualifying time
pole_time = df.groupby("raceId")["best_q_sec"].transform("min")
df["quali_gap_pct"] = (df["best_q_sec"] - pole_time) / pole_time * 100
```

**What it does:**
- **`grid_feat`:** the starting grid position. Pole = 1. Known after qualifying.
- **`best_q_sec`:** the driver's best qualifying lap time in seconds across Q1/Q2/Q3. Drivers eliminated in Q1 only have a Q1 time.
- **`quali_gap_pct`:** percentage time deficit to the fastest qualifier (pole sitter) in the same race. Normalises across circuits of different lengths; decouples car performance from grid position when grid penalties apply.

---

### Cell 28 — Form Feature: Rolling Podiums in Last 5 Races (Code)

```python
df.sort_values(["driverId","date"], inplace=True)

df["drv_podiums_last5"] = (
    df.groupby("driverId")["podium"]
      .transform(lambda x: x.shift(1).rolling(5, min_periods=1).sum())
)
```

**What it does:**
- Sorts the DataFrame by driver and date (chronological order within each driver's career).
- For each row, looks back at the driver's **previous** 5 race results — `shift(1)` excludes the current race to prevent leakage.
- Counts how many of those 5 races the driver finished on the podium.
- Captures driver momentum and recent car form on a short timescale.

---

### Cell 29 — Form & Standings Features (Code)

```python
# team_podiums_last5: constructor's podiums in last 5 races
df["team_podiums_last5"] = (
    df.groupby("constructorId")["podium"]
      .transform(lambda x: x.shift(1).rolling(5, min_periods=1).sum())
)

# drv_points_share: driver's fraction of team's season points so far
df["drv_points_share"] = df["drv_points_so_far"] / df["team_points_so_far"].replace(0, np.nan)

# team_champ_pos: constructor championship position before this race
```

**What it does:**
- **`team_podiums_last5`:** the team's podium count in the last 5 races. Reflects car competitiveness over a recent window independently of which driver is racing.
- **`drv_points_share`:** a driver's fraction of their team's total championship points so far in the current season. Near 1.0 means the driver is carrying the team; near 0.5 means teammates are equal. Proxy for intra-team relative skill.
- **`team_champ_pos`:** the constructor's current championship standing. Lower number = stronger team. Season-long signal of car quality.

---

### Cell 30 — Context Features (Code)

```python
# driver_age: driver's age on race day in years
df["driver_age"] = (df["date"] - df["dob"]).dt.days / 365.25

# home_race: 1 if driver nationality == circuit country, else 0
df["home_race"] = (df["nationality"] == df["circuit_country"]).astype(float)
```

**What it does:**
- **`driver_age`:** driver's age in fractional years on race day. Experience and peak-performance considerations. Added as a weak contextual signal.
- **`home_race`:** the binary flag from Q2. Kept as a cheap feature; the ablation in future sections will confirm whether it adds predictive value.

---

### Cell 31 — Feature List Definition (Code)

```python
FEATURES = [
    "grid_feat",          # starting grid position
    "quali_gap_pct",      # % gap to pole in qualifying
    "drv_podiums_last5",  # driver's podiums in last 5 races
    "team_podiums_last5", # team's podiums in last 5 races
    "drv_points_share",   # driver's share of team's season points
    "team_champ_pos",     # constructor championship position
    "driver_age",         # driver age on race day
    "home_race",          # 1 = driver's home country
]
```

**What it does:**
- Defines the canonical list `FEATURES` used by all subsequent model training and evaluation code.
- Acts as the single source of truth for which columns form the feature matrix `X`.

---

### Cells 32–43 — Standings Feature Construction (Code + Markdown)

These cells contain the detailed logic for computing the standings features with correct temporal alignment:

- **Joining `driver_standings`:** After each race Ergast publishes cumulative points. The notebook selects the standings entry for the *previous round* so that no information from the current race leaks in.
- **Joining `constructor_standings`:** Same approach for the team's cumulative points and championship position.
- **Handling round 1:** At the season's first race there are no prior standings. These rows receive `NaN` (handled by imputation in Section 8).
- **Intermediate plots:** Each feature is histogrammed after construction to verify it looks reasonable.

---

### Cell 44 — Finalise Modeling Table (Code)

```python
data = df[df.year >= 2003].reset_index(drop=True)
assert not data.duplicated(["raceId", "driverId"]).any()
print(f"Modeling table: {len(data)} rows (one per driver per race), podium rate {data.podium.mean():.3f}")
data[FEATURES].isna().mean().round(3).to_frame("missing share").T
```

**What it does:**
- Filters to 2003+ (first year with reliable qualifying times).
- Asserts no duplicate (raceId, driverId) pairs in the final table.
- Prints total row count and overall podium rate (~16.4%, reflecting ~3 podium spots per ~18-car grid).
- Shows the missing-value rate for each feature — most are very low (<2%); only form features have missingness for round-1 entries or rookies.

---

### Cell 45 — Observations (Markdown)
Notes which features have missingness and explains why (rookies have no rolling form; round 1 has no standings). Previews that Section 8 will handle imputation.

---

### Cell 46 — Feature Evidence: ROC-AUC and Podium Rate Plots (Code)

```python
train_part = data[data.year <= 2017]
def single_auc(col):
    x, y = train_part[col], train_part.podium
    ok = x.notna(); a = roc_auc_score(y[ok], x[ok]); return max(a, 1 - a)
auc_table = pd.Series({f: single_auc(f) for f in FEATURES}).sort_values()
auc_table.plot.barh(figsize=(8, 5), color="seagreen"); plt.axvline(0.5, c="k")
plt.title("Each feature alone: ROC-AUC (training seasons)"); plt.tight_layout(); plt.show()

fig, axes = plt.subplots(2, 4, figsize=(18, 7)); ...  # podium rate per feature bin

sns.heatmap(train_part[FEATURES].corr(method="spearman"), ...)  # correlation matrix
```

**What it does:**
- Computes the univariate **ROC-AUC** for each feature using only training seasons (2003–2017). `max(a, 1-a)` makes the metric symmetric so negatively correlated features (like `grid_feat`) still show their true informativeness.
- Key AUC results: `grid_feat` and `quali_gap_pct` ≈ 0.90; form features ≈ 0.80–0.85; `home_race` ≈ 0.50 (essentially useless alone).
- **Podium rate per feature bin** plots for all 8 features, showing the shape of each feature–target relationship (monotonic, non-monotonic, etc.).
- **Spearman correlation heatmap** to identify redundancy. `grid_feat` and `quali_gap_pct` are correlated (~0.7), but both are kept because grid penalties decouple them.

---

### Cell 47 — Why Each Feature Is Kept (Markdown)
Provides the rationale for retaining every feature in `FEATURES`, referencing the AUC values and correlation matrix from the previous cell.

---

### Cell 48 — Leakage Demonstration (Code)

```python
print("AUC of 'points' of the same race (forbidden):", round(single_auc("points"), 3))
print("AUC of 'grid' (allowed):", round(single_auc("grid_feat"), 3))
```

**What it does:**
- Computes the AUC of `points` — the points earned *in the same race* (which is effectively the target itself).
- Shows AUC ≈ 0.99 — a clear demonstration of what data leakage looks like.
- Contrasts with `grid_feat` (AUC ≈ 0.90), the strongest *allowed* feature.
- Pedagogical cell illustrating why feature selection must be done carefully and why features must be strictly pre-race.

---

## Section 7 — Train / Validation / Test Split

### Cell 49 — Section Introduction (Markdown)
Explains the time-based split strategy:

| Split | Seasons | Purpose |
|---|---|---|
| Train | 2003–2017 | Fit all models |
| Validation | 2018–2021 | Tune hyperparameters & decision threshold |
| Test | 2022–2024 | Final evaluation (used **once**) |

Justification: models will be used to predict *future* races, so evaluation must reflect temporal order. A random split would put 2024 races in the training set, making scores unrealistically optimistic. The 2022 boundary also aligns with the major technical regulation reset, making the test set a true "unseen distribution" check.

---

### Cell 50 — Apply the Split (Code)

```python
train = data[data.year <= 2017]
val   = data[(data.year >= 2018) & (data.year <= 2021)]
test  = data[data.year >= 2022]
X_train, y_train = train[FEATURES], train.podium.values
X_val,   y_val   = val[FEATURES],   val.podium.values
X_test,  y_test  = test[FEATURES],  test.podium.values

pd.DataFrame({"rows": [len(train), len(val), len(test)],
              "podium rate": [y_train.mean(), y_val.mean(), y_test.mean()]},
             index=["train","validation","test"]).round(3)
```

**What it does:**
- Slices `data` by `year` into three non-overlapping subsets.
- Creates six arrays: `X_train / y_train`, `X_val / y_val`, `X_test / y_test`.
- Prints a summary table with row counts and podium rates for each split. All three should have similar podium rates (~16–17%), confirming the target is stable over time.

---

## Section 8 — Pre-processing

### Cell 51 — Section Introduction (Markdown)
Describes the two pre-processing steps:
1. **Imputation** — fill missing values with the training-set **median** (robust to skew; avoids mean being pulled by outliers).
2. **Scaling** — `StandardScaler` transforms each feature to mean 0, std 1 (required by logistic regression and neural networks; tree models are invariant to it).

Both transformers are **fit on the training set only** and applied to validation and test without re-fitting, preventing test-data leakage into pre-processing parameters.

---

### Cell 52 — Fit and Apply Pre-processing (Code)

```python
imputer = SimpleImputer(strategy="median").fit(X_train)
scaler  = StandardScaler().fit(imputer.transform(X_train))

def prep(X, scale=True):
    Z = imputer.transform(X)
    return scaler.transform(Z) if scale else Z

Xtr_imp, Xva_imp, Xte_imp = prep(X_train, False), prep(X_val, False), prep(X_test, False)
Xtr_sc,  Xva_sc,  Xte_sc  = prep(X_train),        prep(X_val),        prep(X_test)

# Distribution comparison: raw vs imputed+scaled for 3 features
fig, ax = plt.subplots(2, 3, figsize=(15, 6))
for j, f in enumerate(["grid_feat", "quali_gap_pct", "driver_age"]):
    i = FEATURES.index(f)
    ax[0, j].hist(X_train[f].dropna(), bins=30, color="grey")
    ax[0, j].set_title(f"{f} — raw")
    ax[1, j].hist(Xtr_sc[:, i], bins=30, color="steelblue")
    ax[1, j].set_title(f"{f} — imputed + scaled")
plt.tight_layout(); plt.show()

# Summary stats table
pd.DataFrame({"raw mean": X_train.mean(), "raw std": X_train.std(),
              "after mean": Xtr_sc.mean(0), "after std": Xtr_sc.std(0)},
             index=FEATURES).round(2)
```

**What it does:**
- Fits `SimpleImputer` (median strategy) on `X_train` and transforms all three splits.
- Fits `StandardScaler` on the *imputed* training data and transforms all three splits.
- Produces before/after histograms for `grid_feat`, `quali_gap_pct`, and `driver_age`: shapes are preserved; only location and scale change.
- Prints a summary table confirming that after scaling all features have mean ≈ 0 and std ≈ 1.
- Exposes two array pairs: `Xtr_imp / Xva_imp / Xte_imp` (imputed only, no scaling — for tree models) and `Xtr_sc / Xva_sc / Xte_sc` (imputed + scaled — for linear models and neural nets).

---

### Cell 53 — Effect Explanation (Markdown)
Confirms that distribution shapes are unchanged by scaling. Notes that imputation adds a small spike at the median for features with missing values.

Introduces the two pre-processing order experiments in the next cell.

---

### Cell 54 — Pre-processing Order Experiments (Code)

```python
def lr_score(Xtr, Xva, Xte):
    m = LogisticRegression(max_iter=2000).fit(Xtr, y_train)
    return {"val PR-AUC":  average_precision_score(y_val,  m.predict_proba(Xva)[:, 1]),
            "test PR-AUC": average_precision_score(y_test, m.predict_proba(Xte)[:, 1])}

rows = {}

# Experiment 1 (correct): fit imputer+scaler on train only
rows["E1: fit on train only (correct)"] = lr_score(Xtr_sc, Xva_sc, Xte_sc)

# Experiment 1 (wrong): fit imputer+scaler on ALL data including test
all_X = pd.concat([X_train, X_val, X_test])
imp_all = SimpleImputer(strategy="median").fit(all_X)
sc_all  = StandardScaler().fit(imp_all.transform(all_X))
f_all   = lambda X: sc_all.transform(imp_all.transform(X))
rows["E1: fit on ALL data before split (leaky)"] = lr_score(
    f_all(X_train), f_all(X_val), f_all(X_test))

# Experiment 2a: impute then scale (standard order)
rows["E2: impute -> scale"] = rows["E1: fit on train only (correct)"]

# Experiment 2b: scale first, then impute
sc_first  = StandardScaler().fit(X_train)   # ignores NaN during fit
imp_after = SimpleImputer(strategy="median").fit(sc_first.transform(X_train))
f_si      = lambda X: imp_after.transform(sc_first.transform(X))
rows["E2: scale -> impute"] = lr_score(f_si(X_train), f_si(X_val), f_si(X_test))

pd.DataFrame(rows).T.round(4)
```

**What it does:**

**Experiment 1 — When to fit the pre-processing transformers:**
- **Correct:** fit imputer and scaler only on `X_train`, then apply to val and test without re-fitting. ✅
- **Leaky (wrong):** fit on the entire dataset (train + val + test) before splitting. Test-set statistics (medians, standard deviations) contaminate the training pipeline. ❌
- **Result:** The PR-AUC scores differ only in the 4th decimal place here because the medians/means barely change with more data. However, the leaky approach is methodologically wrong and could matter more in other datasets or in production.

**Experiment 2 — Order of imputation and scaling:**
- **Impute → Scale (standard):** fill missing values first with training-set medians, then scale the complete matrix. ✅
- **Scale → Impute:** apply `StandardScaler` to the incomplete matrix (scikit-learn skips NaN during fit), then fill missing values with the median of the *scaled* data.
- **Result:** Nearly identical scores. The standard order (impute → scale) is preferred because the scaler sees a fully complete matrix.

---

### Cell 55 — Experiment Results and Conclusions (Markdown)
Summarises both experiments:
- **E1:** "Fit on all data" is wrong even if the numeric impact is small in this dataset — always fit on train only.
- **E2:** Both orders yield nearly identical results; the notebook adopts the standard **impute → scale** order for clarity and correctness.

---

## Summary

| Section | Key Deliverable |
|---|---|
| 1. Setup & Loading | 14 CSV tables loaded as raw strings; run-all ready |
| 2. EDA Before Cleaning | `\N` heatmap; raw value spot-checks; PK duplicate verification |
| 3. Cleaning | `\N` → NaN; proper dtypes; FK integrity; one-row-per-(raceId, driverId) table |
| 4. EDA After Cleaning | Post-cleaning missingness; yearly coverage; distribution plots; podium rate over time |
| 5. Q1–Q3 | Grid → podium (exponential decay); home race (negligible effect); prior-season standings (strong predictor, must be lagged) |
| 6. Features | 8 leakage-safe features with univariate AUC evidence, podium rate plots, and correlation heatmap |
| 7. Split | Time-based: Train 2003–2017 / Val 2018–2021 / Test 2022–2024 |
| 8. Pre-processing | Median imputation + StandardScaler (fit on train only); 2 order experiments confirming correct methodology |
