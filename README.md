# Dead Product Prediction Dashboard

A Streamlit app that predicts which products in an ERP transaction dataset
are likely to become "dead" (i.e., stop selling) within a future time
window, using forward-looking labels and product-level behavioral features.

## What it does

- Loads a transactions CSV (Oid, Code, Label1, Quantity, Amount, VWAP,
  DiscountPercent, Date, Type, Oid-2)
- Builds one row per product with recency, frequency, volume, return-rate,
  rolling-window (30/60/90 day), interpurchase-gap, and trend features —
  computed strictly from data on or before a chosen cutoff date (no leakage)
- Labels each product **Dead (1)** if it has zero sales in the window
  `(cutoff, cutoff + horizon]`, otherwise **Active (0)**
- Trains and compares Logistic Regression, Random Forest, Gradient
  Boosting, and XGBoost (if installed), tuned with F-beta scoring
  (beta < 1, favoring precision — a false "dead" call is treated as
  costlier than a missed one)
- Lets you inspect confusion matrices, ROC/PR curves, and feature importance
- Scores currently-eligible products (recent sales activity) using their
  **full** transaction history and exports predictions as CSV

## Installation

```bash
pip install -r requirements.txt
```

`xgboost` and `imbalanced-learn` are optional — the app detects their
absence and simply skips XGBoost / SMOTE if not installed.

## Running

```bash
streamlit run dead_product_app.py
```

Then upload your transactions CSV in the sidebar.

## Sidebar configuration

| Section | Setting | Meaning |
|---|---|---|
| 2. Forward labeling rules | Feature cutoff date | Features use only data ≤ this date |
| | Forward window (days) | Product is "Dead" if zero sales occur in `(cutoff, cutoff+horizon]` |
| 3. Training | Test size | Train/test split fraction |
| | SMOTE | Optional oversampling of the minority (dead) class |
| | Precision weight (beta) | Lower beta → tuning favors precision over recall |
| | Decision threshold | Probability cutoff for calling a product "dead" |
| 4. Prediction eligibility | Minimum recent-activity date | A product must have ≥1 sale on/after this date to be scored on the Predict tab |

⚠️ If `cutoff + horizon` exceeds the last date in your data, the app warns
you — labels near the cutoff would be biased toward "Dead" because the
future window is only partially observed. Pick an earlier cutoff or a
shorter horizon.

## Tabs

1. **Overview** — raw data preview, row/product counts, transaction volume over time
2. **Labels** — engineered features + Active/Dead class balance
3. **Models** — train & compare classifiers, confusion matrix, ROC/PR curves
4. **Importance** — feature importance / coefficients per trained model
5. **Predict New Data** — scores currently-active products (full history features) and lets you download a CSV of predictions

## Notes on design decisions

- **No feature leakage**: features for a given cutoff date only ever see
  transactions up to that date; the label looks strictly forward.
- **Eligibility ≠ feature window**: which products get scored (recent
  activity) is decoupled from how their features are computed (always full
  lifetime history) — mirrors the same pattern used in the client-churn
  pipeline.
- **Precision-favoring tuning**: incorrectly flagging a still-selling
  product as dead is assumed more costly than missing a dead one, so
  F-beta (beta=0.5 default) is used instead of F1/accuracy during
  hyperparameter search.
