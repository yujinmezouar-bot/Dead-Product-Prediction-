"""
Dead Product Prediction — end-to-end pipeline + Streamlit dashboard
Run:      streamlit run dead_product_app.py
Install:  pip install streamlit pandas numpy scikit-learn imbalanced-learn xgboost plotly
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import timedelta
import plotly.express as px
import plotly.graph_objects as go

from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, precision_score,
                              recall_score, f1_score, fbeta_score, roc_auc_score,
                              average_precision_score, matthews_corrcoef, confusion_matrix,
                              roc_curve, precision_recall_curve, make_scorer)

try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

try:
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline as ImbPipeline
    IMB_AVAILABLE = True
except ImportError:
    IMB_AVAILABLE = False

st.set_page_config(page_title="Dead Product Prediction", layout="wide")

# =========================================================
# CONFIG - adjust to your real column names if different
# =========================================================
CFG = dict(
    date_col="Date", type_col="Type", sale_val=10, return_val=12,
    prod_id="Oid", prod_code="Code", prod_name="Label1",
    cust_id="Oid-2",
    qty="Quantity", amount="Amount", vwap="VWAP", disc="DiscountPercent",
)

# No product-level categorical dimensions are guaranteed in the schema
# (unlike the client dataset which had region/wilaya/segment). If you have
# a product category/family column, add its name here and it will be
# automatically one-hot encoded.
CAT_FEATS = []  # e.g. ["prod_family"] if you add such a column

# NOTE: unlike the client "recency_days" leakage problem, "days_since_last_sale"
# here is NOT part of the label definition (the label looks at a FUTURE window),
# so it is safe to keep it as a predictive feature.
NUM_FEATS = [
    "days_since_last_sale", "tenure_days", "sales_frequency",
    "total_sales_amount", "average_sale_amount",
    "total_quantity_sold", "average_quantity",
    "unique_customers", "average_discount", "average_price",
    "return_rate",
    "sales_amount_last_30", "sales_amount_last_60", "sales_amount_last_90",
    "sales_count_last_30", "sales_count_last_60", "sales_count_last_90",
    "avg_interpurchase_days", "sales_trend_ratio",
]

# =========================================================
# DATA LOADING
# =========================================================
@st.cache_data(show_spinner=False)
def load_data(file):
    df = pd.read_csv(file)
    df[CFG["date_col"]] = pd.to_datetime(df[CFG["date_col"]], errors="coerce")
    df = df.dropna(subset=[CFG["date_col"], CFG["prod_id"]])
    return df

def _safe_ratio(a, b):
    return a / b if b not in (0, None) and not pd.isna(b) else 0.0

# =========================================================
# PRODUCT-LEVEL FEATURE ENGINEERING
# =========================================================
def compute_product_features(df, cutoff_date, config=CFG, start_date=None):
    """
    One row per product, built ONLY from transactions where
    start_date < Date <= cutoff_date (strict "no future leakage" window).

    - start_date=None: uses the product's ENTIRE history up to cutoff_date.
      This is used both for training-time feature computation and for the
      final prediction run (features should reflect full product lifetime,
      not an arbitrarily truncated slice).
    """
    d = df[df[config["date_col"]] <= cutoff_date].copy()
    if start_date is not None:
        d = d[d[config["date_col"]] > start_date].copy()
    if d.empty:
        return pd.DataFrame()

    rows = []
    for pid, g in d.groupby(config["prod_id"]):
        g_sales = g[g[config["type_col"]] == config["sale_val"]]
        g_ret = g[g[config["type_col"]] == config["return_val"]]

        # --- Recency / tenure ---
        if g_sales.empty:
            last_sale = pd.NaT
            first_sale = pd.NaT
        else:
            last_sale = g_sales[config["date_col"]].max()
            first_sale = g_sales[config["date_col"]].min()

        days_since_last_sale = (cutoff_date - last_sale).days if pd.notna(last_sale) else np.nan
        tenure_days = (cutoff_date - first_sale).days if pd.notna(first_sale) else 0

        # --- Frequency / volume ---
        sales_frequency = g_sales[config["date_col"]].nunique()
        total_sales_amount = g_sales[config["amount"]].sum() if len(g_sales) else 0.0
        average_sale_amount = g_sales[config["amount"]].mean() if len(g_sales) else 0.0
        total_quantity_sold = g_sales[config["qty"]].sum() if len(g_sales) else 0.0
        average_quantity = g_sales[config["qty"]].mean() if len(g_sales) else 0.0
        unique_customers = g_sales[config["cust_id"]].nunique() if len(g_sales) else 0
        average_discount = g_sales[config["disc"]].mean() if len(g_sales) else 0.0
        average_price = g_sales[config["vwap"]].mean() if len(g_sales) else 0.0

        # --- Returns ---
        n_sales, n_returns = len(g_sales), len(g_ret)
        return_rate = _safe_ratio(n_returns, n_sales + n_returns)

        # --- Rolling windows (last 30/60/90 days before cutoff) ---
        last30 = g_sales[g_sales[config["date_col"]] > cutoff_date - timedelta(days=30)]
        last60 = g_sales[g_sales[config["date_col"]] > cutoff_date - timedelta(days=60)]
        last90 = g_sales[g_sales[config["date_col"]] > cutoff_date - timedelta(days=90)]

        # --- Interpurchase gap ---
        sale_dates = np.sort(g_sales[config["date_col"]].unique())
        if len(sale_dates) > 1:
            gaps = np.diff(sale_dates).astype("timedelta64[D]").astype(int)
            avg_interpurchase_days = gaps.mean()
        else:
            avg_interpurchase_days = tenure_days

        # --- Trend ratio: 2nd half of life vs 1st half, by sales amount ---
        if pd.notna(first_sale) and tenure_days > 0:
            mid = first_sale + (cutoff_date - first_sale) / 2
            first_half = g_sales[g_sales[config["date_col"]] <= mid][config["amount"]].sum()
            second_half = g_sales[g_sales[config["date_col"]] > mid][config["amount"]].sum()
            sales_trend_ratio = (second_half / first_half if first_half > 0
                                  else (1.0 if second_half == 0 else 2.0))
        else:
            sales_trend_ratio = 1.0

        rows.append({
            "product_id": pid,
            "product_code": g[config["prod_code"]].iloc[0],
            "product_name": g[config["prod_name"]].iloc[0],
            "days_since_last_sale": days_since_last_sale,
            "tenure_days": tenure_days,
            "sales_frequency": sales_frequency,
            "total_sales_amount": total_sales_amount,
            "average_sale_amount": average_sale_amount,
            "total_quantity_sold": total_quantity_sold,
            "average_quantity": average_quantity,
            "unique_customers": unique_customers,
            "average_discount": average_discount,
            "average_price": average_price,
            "return_rate": return_rate,
            "sales_amount_last_30": last30[config["amount"]].sum(),
            "sales_amount_last_60": last60[config["amount"]].sum(),
            "sales_amount_last_90": last90[config["amount"]].sum(),
            "sales_count_last_30": last30.shape[0],
            "sales_count_last_60": last60.shape[0],
            "sales_count_last_90": last90.shape[0],
            "avg_interpurchase_days": avg_interpurchase_days,
            "sales_trend_ratio": sales_trend_ratio,
        })

    feat_df = pd.DataFrame(rows)
    # Products with no sales at all before cutoff get a large "days since last
    # sale" fallback so they aren't dropped/NaN'd out of the model.
    if not feat_df.empty:
        max_gap = feat_df["days_since_last_sale"].max()
        fallback = max_gap + 1 if pd.notna(max_gap) else 9999
        feat_df["days_since_last_sale"] = feat_df["days_since_last_sale"].fillna(fallback)
    return feat_df

# =========================================================
# FORWARD LABELING (the key structural change vs. recency labeling)
# =========================================================
def add_forward_label(feat_df, df, cutoff_date, horizon_days, config=CFG):
    """
    Forward-looking label:
      - Features in feat_df are already computed using ONLY data <= cutoff_date.
      - Look at the window (cutoff_date, cutoff_date + horizon_days].
      - If a product has ZERO sales transactions in that future window -> Dead (1).
      - Otherwise -> Active (0).

    This requires that cutoff_date + horizon_days does not exceed the last
    available date in df, or the future window is only partially observed
    (which would bias products near the end of the data toward "Dead").
    """
    future_end = cutoff_date + timedelta(days=horizon_days)
    future_window = df[(df[config["date_col"]] > cutoff_date) &
                        (df[config["date_col"]] <= future_end) &
                        (df[config["type_col"]] == config["sale_val"])]
    products_alive_in_future = set(future_window[config["prod_id"]].unique())

    feat_df = feat_df.copy()
    feat_df["dead"] = (~feat_df["product_id"].isin(products_alive_in_future)).astype(int)
    return feat_df

def get_eligible_products(df, min_history_date, as_of, config=CFG):
    """
    Products eligible to be scored on the Predict tab: those with at least
    one SALE transaction between min_history_date and as_of. This defines
    WHO gets scored (currently-relevant products), independent of how their
    features are computed (which always uses full history up to as_of).
    """
    mask = ((df[config["date_col"]] >= min_history_date) &
            (df[config["date_col"]] <= as_of) &
            (df[config["type_col"]] == config["sale_val"]))
    return set(df.loc[mask, config["prod_id"]].unique())

# =========================================================
# MODELING (unchanged pipeline structure)
# =========================================================
def build_preprocessor():
    transformers = [("num", StandardScaler(), NUM_FEATS)]
    if CAT_FEATS:
        from sklearn.preprocessing import OneHotEncoder
        transformers.append(("cat", OneHotEncoder(handle_unknown="ignore"), CAT_FEATS))
    return ColumnTransformer(transformers)

def get_model_grid():
    models = {
        "LogisticRegression": (
            LogisticRegression(max_iter=2000, class_weight="balanced"),
            {"clf__C": [0.01, 0.1, 1, 10], "clf__penalty": ["l2"]}
        ),
        "RandomForest": (
            RandomForestClassifier(class_weight="balanced", random_state=42),
            {"clf__n_estimators": [200, 400, 600], "clf__max_depth": [4, 8, 12, None],
             "clf__min_samples_leaf": [1, 3, 5]}
        ),
        "GradientBoosting": (
            GradientBoostingClassifier(random_state=42),
            {"clf__n_estimators": [100, 200, 300], "clf__learning_rate": [0.01, 0.05, 0.1],
             "clf__max_depth": [2, 3, 4]}
        ),
    }
    if XGB_AVAILABLE:
        models["XGBoost"] = (
            XGBClassifier(eval_metric="logloss", random_state=42),
            {"clf__n_estimators": [200, 400], "clf__max_depth": [3, 5, 7],
             "clf__learning_rate": [0.01, 0.05, 0.1], "clf__scale_pos_weight": [1, 3, 5]}
        )
    return models

def _grid_size(grid):
    size = 1
    for v in grid.values():
        size *= len(v)
    return size

def train_and_tune(X_train, y_train, scoring_beta=0.5, n_iter=12, use_smote=False):
    """
    scoring_beta < 1 weights PRECISION more than recall during tuning.
    Business rule: flagging an active product as 'dead' (False Positive) means
    you might discontinue/de-stock something still selling -> costlier than
    missing a truly dead product (False Negative) -> favor precision.
    """
    fbeta_scorer = make_scorer(fbeta_score, beta=scoring_beta, zero_division=0)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = {}
    for name, (estimator, grid) in get_model_grid().items():
        preproc = build_preprocessor()
        if use_smote and IMB_AVAILABLE:
            pipe = ImbPipeline([("prep", preproc), ("smote", SMOTE(random_state=42)), ("clf", estimator)])
        else:
            pipe = Pipeline([("prep", preproc), ("clf", estimator)])
        search = RandomizedSearchCV(pipe, grid, n_iter=min(n_iter, _grid_size(grid)),
                                     scoring=fbeta_scorer, cv=cv, random_state=42, n_jobs=-1)
        search.fit(X_train, y_train)
        results[name] = search.best_estimator_
    return results

def evaluate_model(model, X_test, y_test, threshold=0.5):
    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, pred, labels=[0, 1]).ravel()
    return {
        "Accuracy": accuracy_score(y_test, pred),
        "Balanced Accuracy": balanced_accuracy_score(y_test, pred),
        "Precision (dead)": precision_score(y_test, pred, zero_division=0),
        "Recall (dead)": recall_score(y_test, pred, zero_division=0),
        "F1": f1_score(y_test, pred, zero_division=0),
        "F0.5 (precision-weighted)": fbeta_score(y_test, pred, beta=0.5, zero_division=0),
        "ROC-AUC": roc_auc_score(y_test, proba),
        "PR-AUC (avg precision)": average_precision_score(y_test, proba),
        "MCC": matthews_corrcoef(y_test, pred),
        "TP": tp, "FP": fp, "FN": fn, "TN": tn,
        "_proba": proba, "_pred": pred,
    }

def get_feature_names(preprocessor):
    names = list(NUM_FEATS)
    if CAT_FEATS:
        names += list(preprocessor.named_transformers_["cat"].get_feature_names_out(CAT_FEATS))
    return names

def get_feature_importance(model, feature_names):
    clf = model.named_steps["clf"]
    if hasattr(clf, "feature_importances_"):
        imp = clf.feature_importances_
    elif hasattr(clf, "coef_"):
        imp = np.abs(clf.coef_[0])
    else:
        return pd.DataFrame()
    return pd.DataFrame({"feature": feature_names, "importance": imp}).sort_values(
        "importance", ascending=False)

# =========================================================
# STREAMLIT APP
# =========================================================
def main():
    st.title("💀 Dead Product Prediction Dashboard")

    st.sidebar.header("1. Data")
    file = st.sidebar.file_uploader("Upload transactions CSV", type=["csv"])
    if file is None:
        st.info("Upload your transactions CSV to start (columns: Oid, Code, Label1, Quantity, "
                 "Amount, VWAP, DiscountPercent, Date, Type, Oid-2).")
        st.stop()

    df = load_data(file)
    max_date = df[CFG["date_col"]].max()

    st.sidebar.header("2. Forward labeling rules")
    cutoff_date = pd.Timestamp(st.sidebar.date_input(
        "Feature cutoff date (features use only data <= this date)",
        value=min(pd.Timestamp("2026-01-01"), max_date)))
    horizon_days = st.sidebar.number_input(
        "Forward window (days) to check for future sales", value=90, min_value=1,
        help="If a product has ZERO sales in (cutoff, cutoff + horizon], it is labeled Dead.")

    future_end = cutoff_date + timedelta(days=horizon_days)
    if future_end > max_date:
        st.sidebar.warning(
            f"⚠️ cutoff + horizon ({future_end.date()}) exceeds the last date in your data "
            f"({max_date.date()}). Labels near the cutoff will be biased toward 'Dead' because "
            f"the future window is only partially observed. Choose an earlier cutoff or shorter horizon."
        )

    st.sidebar.header("3. Training")
    test_size = st.sidebar.slider("Test size", 0.1, 0.4, 0.2)
    use_smote = st.sidebar.checkbox("Use SMOTE oversampling", value=False)
    beta = st.sidebar.slider("Precision weight (Fbeta) - lower = more precision-focused", 0.3, 1.5, 0.5)
    threshold = st.sidebar.slider("Decision threshold (probability of 'dead')", 0.05, 0.95, 0.5)

    st.sidebar.header("4. Prediction eligibility")
    predict_start = pd.Timestamp(st.sidebar.date_input(
        "Minimum recent-activity date for scoring",
        value=max_date - timedelta(days=365),
        help="A product is scored ONLY if it has at least one sale on or after this date "
             "(up to the latest date in the data). This decides WHICH products are scored — "
             "each eligible product's features are still computed from its COMPLETE history."))

    tabs = st.tabs(["📊 Overview", "🏷️ Labels", "🤖 Models", "⭐ Importance", "🔮 Predict New Data"])

    # ---- Overview ----
    with tabs[0]:
        st.subheader("Raw data")
        st.dataframe(df.head(20))
        c1, c2, c3 = st.columns(3)
        c1.metric("Rows", len(df))
        c2.metric("Products", df[CFG["prod_id"]].nunique())
        c3.metric("Date range", f"{df[CFG['date_col']].min().date()} → {max_date.date()}")
        st.plotly_chart(px.histogram(df, x=CFG["date_col"], title="Transactions over time"),
                         use_container_width=True)

    # ---- Labels / features for TRAINING ----
    # Features: only data <= cutoff_date. Label: sales in (cutoff, cutoff+horizon].
    feats = compute_product_features(df, cutoff_date)  # start_date=None -> full history to cutoff
    feats = add_forward_label(feats, df, cutoff_date, horizon_days)

    with tabs[1]:
        st.subheader(f"Product features as of {cutoff_date.date()} "
                      f"(labeled using sales through {future_end.date()})")
        st.dataframe(feats.head(20))
        dist = feats["dead"].value_counts().rename({0: "Active", 1: "Dead"})
        c1, c2 = st.columns(2)
        c1.plotly_chart(px.pie(values=dist.values, names=dist.index, title="Class balance"),
                         use_container_width=True)
        c2.metric("Dead rate", f"{feats['dead'].mean()*100:.1f}%")

    # ---- Train models ----
    with tabs[2]:
        st.subheader("Train & compare classification models")
        st.caption("Flagging a still-selling product as 'dead' (False Positive) is treated as "
                   "MORE costly than missing a truly dead product (False Negative), so tuning "
                   "favors precision via an F-beta score (beta < 1).")
        if st.button("🚀 Train models"):
            feature_cols = NUM_FEATS + CAT_FEATS
            X = feats[feature_cols]
            y = feats["dead"]
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, stratify=y, random_state=42)

            with st.spinner("Tuning models..."):
                models = train_and_tune(X_train, y_train, scoring_beta=beta, use_smote=use_smote)

            results = {name: evaluate_model(m, X_test, y_test, threshold) for name, m in models.items()}
            st.session_state["models"] = models
            st.session_state["results"] = results
            st.session_state["y_test"] = y_test

        if "results" in st.session_state:
            results = st.session_state["results"]
            comp = pd.DataFrame({name: {k: v for k, v in r.items() if not k.startswith("_")}
                                  for name, r in results.items()}).T
            sort_metric = st.selectbox("Sort by", comp.columns.tolist(),
                                        index=comp.columns.get_loc("F0.5 (precision-weighted)"))
            comp = comp.sort_values(sort_metric, ascending=False)
            st.dataframe(comp.style.background_gradient(cmap="Reds", subset=[sort_metric]))

            best_name = comp.index[0]
            st.success(f"Best model: **{best_name}**")
            st.session_state["best_model_name"] = best_name

            metric_cols = ["Precision (dead)", "Recall (dead)", "F1", "F0.5 (precision-weighted)",
                            "ROC-AUC", "PR-AUC (avg precision)"]
            fig = go.Figure()
            for name in comp.index:
                fig.add_trace(go.Bar(name=name, x=metric_cols, y=[results[name][m] for m in metric_cols]))
            fig.update_layout(barmode="group", title="Metric comparison")
            st.plotly_chart(fig, use_container_width=True)

            sel = st.selectbox("Inspect model", comp.index.tolist())
            r = results[sel]
            c1, c2 = st.columns(2)
            with c1:
                cm = np.array([[r["TN"], r["FP"]], [r["FN"], r["TP"]]])
                st.plotly_chart(px.imshow(cm, text_auto=True, x=["Pred Active", "Pred Dead"],
                                           y=["Actual Active", "Actual Dead"],
                                           title=f"Confusion matrix — {sel}"), use_container_width=True)
            with c2:
                fpr, tpr, _ = roc_curve(st.session_state["y_test"], r["_proba"])
                prec, rec, _ = precision_recall_curve(st.session_state["y_test"], r["_proba"])
                fig2 = go.Figure()
                fig2.add_trace(go.Scatter(x=fpr, y=tpr, name="ROC"))
                fig2.add_trace(go.Scatter(x=[0, 1], y=[0, 1], line=dict(dash="dash"), name="Random"))
                fig2.update_layout(title=f"ROC curve — {sel}", xaxis_title="FPR", yaxis_title="TPR")
                st.plotly_chart(fig2, use_container_width=True)
                fig3 = go.Figure()
                fig3.add_trace(go.Scatter(x=rec, y=prec, name="PR curve"))
                fig3.update_layout(title=f"Precision-Recall curve — {sel}",
                                    xaxis_title="Recall", yaxis_title="Precision")
                st.plotly_chart(fig3, use_container_width=True)

    # ---- Feature importance ----
    with tabs[3]:
        if "models" not in st.session_state:
            st.warning("Train models first (tab 'Models').")
        else:
            sel = st.selectbox("Model", list(st.session_state["models"].keys()), key="fi_sel")
            model = st.session_state["models"][sel]
            fnames = get_feature_names(model.named_steps["prep"])
            fi = get_feature_importance(model, fnames)
            if fi.empty:
                st.info(f"{sel} does not expose feature importances/coefficients directly.")
            else:
                st.plotly_chart(px.bar(fi.head(20), x="importance", y="feature", orientation="h",
                                        title=f"Top 20 important features — {sel}"
                                        ).update_yaxes(categoryorder="total ascending"),
                                 use_container_width=True)

    # ---- Predict: score all currently-active products using full history to date ----
    # Eligibility (WHO gets scored) is decoupled from feature computation (HOW features
    # are built), exactly as in the client version. A product qualifies if it had at
    # least one sale during [predict_start, as_of]; features are computed from its
    # ENTIRE history up to as_of, mirroring the training-time feature logic.
    with tabs[4]:
        st.subheader("Predict dead products — scored on latest available data")
        if "models" not in st.session_state or "best_model_name" not in st.session_state:
            st.warning("Train models first (tab 'Models').")
        else:
            as_of = max_date
            st.write(f"Eligible products: those with **at least one sale** between "
                     f"**{predict_start.date()}** and **{as_of.date()}**. "
                     f"For each eligible product, features are computed from its **complete "
                     f"sales history** up to {as_of.date()} (same feature logic as training).")

            model_names = list(st.session_state["models"].keys())
            model_name = st.selectbox("Model to use for prediction", model_names,
                                       index=model_names.index(st.session_state["best_model_name"]))
            model = st.session_state["models"][model_name]

            total_products = df[CFG["prod_id"]].nunique()

            # 1) WHO: products with recent sales activity
            eligible_ids = get_eligible_products(df, predict_start, as_of)

            # 2) HOW: features from FULL history up to as_of (start_date=None)
            full_feats = compute_product_features(df, as_of, start_date=None)

            if full_feats.empty:
                st.warning("No transactions found up to the latest date in the data.")
            else:
                new_feats = full_feats[full_feats["product_id"].isin(eligible_ids)].copy()

                if new_feats.empty:
                    st.warning("No products had sales in the selected activity window "
                               f"({predict_start.date()} → {as_of.date()}).")
                else:
                    feature_cols = NUM_FEATS + CAT_FEATS
                    X_new = new_feats[feature_cols]
                    proba = model.predict_proba(X_new)[:, 1]
                    new_feats["dead_probability"] = proba
                    new_feats["predicted_status"] = np.where(proba >= threshold, "Dead", "Active")

                    c1, c2, c3 = st.columns(3)
                    c1.metric("Eligible products (scored)", len(new_feats))
                    c2.metric("Predicted dead", int((new_feats["predicted_status"] == "Dead").sum()))
                    c3.metric("Predicted dead rate",
                              f"{(new_feats['predicted_status']=='Dead').mean()*100:.1f}%")

                    excluded_count = total_products - new_feats["product_id"].nunique()
                    if excluded_count > 0:
                        st.info(f"{excluded_count} product(s) had no sales between "
                                f"{predict_start.date()} and {as_of.date()} and were excluded "
                                f"from scoring.")

                    st.dataframe(new_feats.sort_values("dead_probability", ascending=False)[
                        ["product_id", "product_code", "product_name", "days_since_last_sale",
                         "sales_frequency", "total_sales_amount", "sales_trend_ratio",
                         "dead_probability", "predicted_status"]])

                    csv = new_feats.to_csv(index=False).encode("utf-8")
                    st.download_button("⬇️ Download predictions CSV", csv,
                                        "predicted_dead_products.csv", "text/csv")

if __name__ == "__main__":
    main()

    
