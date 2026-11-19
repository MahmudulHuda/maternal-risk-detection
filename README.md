# Maternal Health Risk Classification

Classifies a pregnant patient's risk level (low, mid or high) from six routine
readings: age, systolic and diastolic blood pressure, blood sugar, body
temperature and heart rate.

The public dataset needs heavy cleaning before it can be trusted for
evaluation. More than half of it is exact duplicate rows, and some identical
readings carry different risk labels. On the raw file a random forest scores a
macro F1 of about 0.85; on the cleaned data the same model scores about 0.70,
and the selected model reaches 0.67 on a held-out test set.

## Dataset

[Maternal Health Risk](https://archive.ics.uci.edu/dataset/863/maternal+health+risk)
from the UCI Machine Learning Repository (Ahmed, 2020,
doi:10.24432/C5DP5D, CC BY 4.0). According to the repository page it was
collected from hospitals, community clinics and maternal health care centres in
rural Bangladesh through an IoT-based monitoring system. A copy is included in
`data/raw/`.

What `python -m maternal_risk.eda` reports about the file:

| | |
|---|---|
| Rows | 1,014 |
| Missing values | 0 |
| Exact duplicate rows | 562 |
| Rows with heart rate 7 bpm | 2 |
| Rows after removing duplicates and the 7 bpm record | 451 |
| Readings that appear with more than one label | 35 (71 rows) |
| Rows after removing those as well | 380 |

Class counts go from 406 low / 336 mid / 272 high in the raw file to
203 low / 76 mid / 101 high in the cleaned data.

The target is a risk category assigned in the source data. The dataset has no
diagnosis fields, so this project does not predict preeclampsia, preterm birth
or mortality.

## Method

**Cleaning.** Three steps, all in `maternal_risk/data.py`:

1. Rows with a heart rate under 30 bpm are dropped (the only ones are two
   copies of a 7 bpm record).
2. Exact duplicates are dropped. Whether they are repeat visits or copy errors
   is not documented; in both cases they are not independent samples.
3. Readings that appear with contradictory labels are dropped entirely, since
   nothing in the data says which label is correct.

Ages run from 10 to 70; these are unusual for pregnancy but not impossible, and
I had no ground truth to correct them, so they stay.

Step 3 deserves a caveat. It removes label noise from the test data as well as
the training data, so scores on the cleaned set describe performance on
readings whose label is unambiguous. Real patients with such ambiguous readings
exist, and the model would be less reliable for them. The comparison table
below shows how much this step moves the score.

**Splitting.** 20% of the cleaned data is held out as a test set (76 rows) and
used once, at the end. Splits are grouped by feature vector
(`StratifiedGroupKFold`) so that identical readings can never sit on both sides.
After cleaning every reading is unique, so this only matters for the
intermediate comparison where conflicting records are kept.

**Model comparison.** Models are compared on the training portion (304 rows)
with 5-fold cross-validation repeated 5 times, on identical folds. The
selection metric is macro F1, since the classes are imbalanced after cleaning
and plain accuracy rewards predicting "low". Recall on the high-risk class is
tracked alongside it, as a missed high-risk patient is the costly error.

Candidates: a majority-class baseline, logistic regression, random forest,
gradient boosting (scikit-learn's histogram version and XGBoost), each with and
without an imbalance treatment (class weights or SMOTE). SMOTE runs inside an
imbalanced-learn pipeline so that only training folds are oversampled.

No features are engineered. Trees do not need scaling; logistic regression gets
a `StandardScaler` inside its pipeline.

**Tuning.** Only the cross-validation winner is tuned, over a small grid that
controls tree depth and leaf size.

## Results

All numbers below come from running the code in this repository with seed 42.

### Effect of each cleaning step

Same default random forest, three protocols (`reports/leakage_check.csv`):

| Protocol | Rows | Macro F1 | Accuracy | High-risk recall |
|---|---|---|---|---|
| Raw file, random folds | 1,014 | 0.848 ± 0.027 | 0.843 | 0.897 |
| Deduplicated, grouped folds | 451 | 0.625 ± 0.043 | 0.703 | 0.786 |
| Deduplicated, conflicts removed (used here) | 380 | 0.700 ± 0.051 | 0.775 | 0.822 |

Most of the raw-file score comes from testing on rows the model has already
seen.

### Cross-validation on the training set

`reports/cv_results.csv`, mean ± standard deviation over 25 folds:

| Model | Macro F1 | Accuracy | High-risk recall |
|---|---|---|---|
| Hist. gradient boosting | 0.691 ± 0.056 | 0.760 | 0.778 |
| Hist. gradient boosting, class weights | 0.685 ± 0.050 | 0.743 | 0.772 |
| Random forest, class weights | 0.682 ± 0.069 | 0.737 | 0.795 |
| Random forest | 0.678 ± 0.066 | 0.762 | 0.820 |
| Logistic regression, class weights | 0.592 ± 0.044 | 0.634 | 0.655 |
| Logistic regression | 0.590 ± 0.055 | 0.718 | 0.690 |
| Majority-class baseline | 0.232 ± 0.002 | 0.533 | 0.000 |

The four tree-based models are within about one point of each other, far less
than the fold-to-fold spread, so the data does not support calling any of them
better than the others. Both clearly beat logistic regression.

**The XGBoost and SMOTE variants are not in this table.** They are implemented
(`maternal_risk/models.py`) but the machine these results were produced on did
not have `xgboost` or `imbalanced-learn` installed, so they were skipped.
Running `python -m maternal_risk.train` with the full `requirements.txt`
installed adds those rows and may change which model is selected.

Things that did not help: class weighting made no measurable difference, and
the tuning grid for gradient boosting selected the default settings.

### Held-out test set

Histogram gradient boosting, 76 rows (`reports/test_metrics.json`):

| | Precision | Recall | F1 | Rows |
|---|---|---|---|---|
| Low | 0.83 | 0.83 | 0.83 | 41 |
| Mid | 0.47 | 0.47 | 0.47 | 15 |
| High | 0.70 | 0.70 | 0.70 | 20 |

Macro F1 0.665, accuracy 0.724.

The mid class is the weak point: fewer than half of the mid-risk patients are
identified. Six of 20 high-risk patients are missed
(`reports/figures/confusion_matrix.png`).

## Limitations

- The test set has 76 rows, so each per-class figure rests on 15 to 41
  patients and would move noticeably with a different split.
- Scores are measured on readings with unambiguous labels (see the cleaning
  caveat above); on the deduplicated data with conflicts kept, macro F1 is
  about 0.63.
- How the risk labels were assigned is not described in enough detail to know
  whether they reflect outcomes or a clinician's judgement of the same readings.
- Single source, single country, no external validation. This is not a clinical
  tool and should not be used as one.

Possible next steps: run the XGBoost and SMOTE variants; try an ordinal model,
since the classes are ordered; move the decision threshold to trade precision
for high-risk recall; validate on a second dataset with richer features.

## Running it

Python 3.10 or newer. Everything runs on a CPU; the full training script took
about a minute and a half on a small cloud machine.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python -m pytest                     # 12 tests
python -m maternal_risk.eda          # dataset summary and figures
python -m maternal_risk.leakage_check
python -m maternal_risk.train        # comparison, tuning, test metrics, saved model
```

Scoring new readings needs a CSV with the six feature columns:

```bash
python -m maternal_risk.predict data/raw/maternal_health_risk.csv -o scored.csv
```

## Layout

```
config.yaml              paths, seed, split and CV settings
data/raw/                the UCI file
maternal_risk/
  data.py                loading, cleaning, grouped split
  models.py              candidate models and tuning grids
  evaluation.py          CV splits and scoring
  eda.py                 dataset summary and figures
  leakage_check.py       score under each cleaning protocol
  train.py               comparison, tuning, final test, model saving
  predict.py             CLI for scoring a CSV
tests/
reports/                 results from the last run
```
