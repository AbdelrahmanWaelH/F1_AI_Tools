"""Fill Sections 12 and 13 of F1_Podium_Prediction.ipynb (run once, then delete this file)."""
import json, uuid
from pathlib import Path

NB = Path(__file__).with_name("F1_Podium_Prediction.ipynb")

SEC12_MD = '''## 12. Inference function

One function that takes a driver-race record as input and returns the podium probability and the podium/not-podium decision, using the chosen model and the threshold from validation.

Demonstrate it on:
1. a **complete** record (all features present), and
2. a **partially-missing** record (some features set to missing, to show the imputer handles them).

**Chosen model:** the tuned random forest `rf` from 9.2 (the model explained in Section 11), with its validation-F1 threshold `threshold_rf`. The function reuses the **already fitted** median `imputer` from Section 8; nothing is re-fitted, so inference sees exactly the same pipeline as training. The forest was trained on imputed, *unscaled* inputs, so the scaler is not applied.

The record can be a `dict`, a `pd.Series` or a one-row `pd.DataFrame`. Any feature that is absent or `None`/`NaN` is filled with the training median; extra columns (names, ids) are ignored. The function also reports which features were imputed, so the caller knows how much of the input was filled in rather than observed.'''

CELL_12_1 = '''# 12.1 Inference function: chosen model (random forest) + validation threshold + train-fitted imputer
def predict_podium(record, model=rf, threshold=threshold_rf):
    """Return the podium probability and decision for one driver-race record.

    record: dict, pd.Series or one-row pd.DataFrame with (some of) the FEATURES.
    Missing or NaN features are filled by the median imputer fitted on train only.
    """
    if isinstance(record, pd.DataFrame):
        if len(record) != 1:
            raise ValueError("pass exactly one driver-race record")
        record = record.iloc[0]
    record = pd.Series(record, dtype=object)
    # keep only model features, in training order; absent features become NaN
    row = pd.DataFrame([record.reindex(FEATURES)], columns=FEATURES).apply(pd.to_numeric, errors="coerce")
    imputed = [f for f in FEATURES if pd.isna(row.at[0, f])]
    X = imputer.transform(row)  # same transform as training; no re-fit, no scaling for the forest
    podium_class = int(np.flatnonzero(model.classes_ == 1)[0])
    p = float(model.predict_proba(X)[0, podium_class])
    return {"podium_probability": p,
            "threshold": float(threshold),
            "prediction": "Podium" if p >= threshold else "No podium",
            "imputed_features": imputed,
            "model_input": pd.Series(X[0], index=FEATURES)}

def show_prediction(out, title):
    print(title)
    print(f"  podium probability : {out['podium_probability']:.4f}")
    print(f"  threshold          : {out['threshold']:.4f}")
    print(f"  decision           : {out['prediction']}")
    print(f"  imputed features   : {out['imputed_features'] or 'none'}")'''

CELL_12_2 = '''# 12.2 Demo on a complete record: first test-set row with every feature present
complete_idx = test.index[test[FEATURES].notna().all(axis=1)][0]
complete_record = test.loc[complete_idx]
id_cols = [c for c in ["year", "race_name", "forename", "surname"] if c in test.columns]
display(complete_record[id_cols + FEATURES].to_frame("Complete record"))

out_complete = predict_podium(complete_record.to_dict())
show_prediction(out_complete, "Complete record")
print("  actual outcome     :", "Podium" if complete_record["podium"] == 1 else "No podium", "(comparison only)")

# sanity check: same probability as the batch pipeline used for evaluation
p_batch = rf.predict_proba(prep(test.loc[[complete_idx], FEATURES], False))[0, list(rf.classes_).index(1)]
print(f"  batch-pipeline prob: {p_batch:.4f}  (match: {np.isclose(p_batch, out_complete['podium_probability'])})")'''

CELL_12_3 = '''# 12.3 Demo on a partially-missing record: same driver-race, but qualifying data unknown,
# one form value set to None and driver_age left out of the input entirely
partial_record = {f: complete_record[f] for f in FEATURES if f != "driver_age"}
partial_record.update({"quali_pos": np.nan, "quali_gap_pct": np.nan, "drv_avg_finish_last5": None})

out_partial = predict_podium(partial_record)
show_prediction(out_partial, "Partially-missing record")

display(pd.DataFrame({"Given value": pd.Series(partial_record, dtype=object).reindex(FEATURES),
                      "Model input (after imputation)": out_partial["model_input"],
                      "Complete-record input": out_complete["model_input"]}))
print(f"Probability change vs complete record: "
      f"{out_partial['podium_probability'] - out_complete['podium_probability']:+.4f}")'''

MD_12_3 = '''**Reading the two demos.** The complete record goes through exactly the batch pipeline used for evaluation, which the check in 12.2 confirms by matching probabilities. In the partially-missing record, qualifying position, qualifying gap and recent average finish are replaced with their **training medians**, and `driver_age`, which is not passed at all, is imputed the same way. The function lists every imputed feature, so the caller can see how much of the input was filled in. The probability change shows how much the forest relied on the removed values. Median imputation pulls a record towards a "typical" driver, so a strong driver usually loses probability and a weak one gains some. A prediction with many imputed features should be trusted less.'''

SEC13_MD = '''## 13. Analytical report (PDF, written separately)

The report is a separate PDF deliverable. It should include:
- Answers to Q1, Q2 and Q3, each with its figure (Section 5)
- The selected features and why each is valid at the pre-race cutoff (Sections 6 and 10)
- The model comparison and metric choice (Section 9)
- The ablation and preprocessing-order results (Sections 8 and 10)
- The global and local explanations, with the causal caveat (Section 11)
- Known limitations, such as qualifying-coverage drift and era sensitivity

**Notes for the report**

1. **Problem and cutoff.** Each row is one driver in one race; the target is `podium` (finish in the top 3). Only information available after qualifying and before the start is used.
2. **Q1–Q3 (Section 5).** Q1: the podium rate falls steeply as grid position worsens, so grid is the dominant signal. Q2: there is no home advantage once grid is controlled for (odds ratio 0.92, 95 % CI 0.78–1.08). Q3: prior-season standings predict current podiums, but only *lagged* standings are leakage-safe.
3. **Features (Sections 6, 10).** There are six groups: grid, qualifying, driver form, team form, standings and context. Every rolling or standings feature is shifted to use **past races only**, and standings come from the previous round. The leakage demonstration in Section 6 shows what happens without this.
4. **Split and preprocessing (Sections 7, 8).** The split is by time: train 1950–2017, validation 2018–2021, test 2022–2024. The median imputer and scaler are fit on train only. The experiments show that leaky fitting and scale→impute ordering change PR-AUC only slightly here, but they are still methodologically wrong. The training-period experiment in 8.2 checks era sensitivity.
5. **Models and metric (Section 9).** The models are logistic regression (the baseline), a random forest, gradient boosting and a shallow neural network. Because the classes are imbalanced, **PR-AUC** is the main metric, with F1, precision, recall and ROC-AUC alongside it. Each model's threshold is chosen by maximising F1 on validation, and the test set is only scored. Use the 9.5 table and plot.
6. **Ablation (Section 10).** For the random forest, the full model scores PR-AUC 0.725 on validation and 0.709 on test. Removing grid causes the largest drop. The other groups overlap, so dropping one of them costs little, while keeping it alone shows its standalone signal.
7. **Explainability (Section 11).** In permutation importance and SHAP, grid dominates, followed by qualifying gap, points share, recent podium rate and championship position. The local LIME example (Bottas, Mexico 2019, a borderline case) is driven by a top-six start and recent podium form. All of these explain the **model**, not causes in the real race.
8. **Inference (Section 12).** A single `predict_podium` function applies the train-fitted imputer, the forest and the validation threshold. It handles missing inputs and reports which features were imputed.
9. **Limitations.** Qualifying-time coverage is poor in early seasons (`has_quali` flags this). The sport's eras differ (points systems, reliability, field size). Only one time-based split is used, and the test set is small (about 1,300 rows). Features are correlated, so importance values are not causal. Median imputation biases incomplete records towards an average driver.'''


def lines(s):
    parts = s.split("\n")
    return [p + "\n" for p in parts[:-1]] + [parts[-1]]


nb = json.loads(NB.read_text(encoding="utf-8"))
cells = {c.get("id"): c for c in nb["cells"]}
for cid, src in {"a4291cd2": SEC12_MD, "a3163888": CELL_12_1, "824593f0": CELL_12_2,
                 "9fdd2762": CELL_12_3, "81a3f862": SEC13_MD}.items():
    cells[cid]["source"] = lines(src)
    if cells[cid]["cell_type"] == "code":
        cells[cid]["outputs"], cells[cid]["execution_count"] = [], None
pos = next(i for i, c in enumerate(nb["cells"]) if c.get("id") == "9fdd2762")
if not any(c.get("id") == "b12c3d4e" for c in nb["cells"]):
    nb["cells"].insert(pos + 1, {"cell_type": "markdown", "id": "b12c3d4e", "metadata": {},
                                 "source": lines(MD_12_3)})
NB.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
print("Sections 12 and 13 written to", NB.name)
