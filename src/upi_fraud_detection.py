# ============================================================
# UPI FRAUD DETECTION - V4.2
# ============================================================
# Leakage-aware, chronological, end-to-end fraud detection
#
# V4.2 fixes:
#   1. Proper HGB balanced sample weighting
#   2. Feature-ablation preprocessing bug
#   3. Full PR-curve threshold search
#   4. Data-driven medium/high risk thresholds
#   5. Separates 80%-recall threshold from risk tiers
#   6. External temporal test evaluation
#   7. Top-K fraud review analysis
#   8. Calibration diagnostics
#   9. Permutation importance
#  10. Model comparison using PR-AUC
#
# IMPORTANT:
#   - post_auth_risk_score is excluded (post-authorization leakage)
#   - raw transaction/customer/merchant IDs are excluded
#   - thresholds are selected ONLY on validation data
#   - external test is reserved for final evaluation
# ============================================================

import os
import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    RandomForestClassifier,
    HistGradientBoostingClassifier
)
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    brier_score_loss
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler
)
from sklearn.utils.class_weight import compute_sample_weight

warnings.filterwarnings("ignore")


# ============================================================
# 1. PATH CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

TRAIN_PATH = PROJECT_ROOT / "data" / "raw" / "transactions_train.csv"
TEST_PATH = PROJECT_ROOT / "data" / "raw" / "transactions_test.csv"

OUTPUT_DIR = PROJECT_ROOT / "outputs"
MODEL_DIR = OUTPUT_DIR / "models"
REPORT_DIR = OUTPUT_DIR / "reports"
PLOT_DIR = OUTPUT_DIR / "plots"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. CONFIGURATION
# ============================================================

RANDOM_STATE = 42

# Chronological validation split
VALIDATION_SIZE = 0.176

# Sensitivity operating point
TARGET_RECALL = 0.80

# Business risk targets
MEDIUM_PRECISION_TARGET = 0.20
HIGH_PRECISION_TARGET = 0.40

# Cost scenarios
COST_SCENARIOS = {
    "Balanced_FP1_FN1": (1, 1),
    "Fraud_Moderate_FP1_FN5": (1, 5),
    "Fraud_High_FP1_FN10": (1, 10),
    "Fraud_Very_High_FP1_FN20": (1, 20),
}


# ============================================================
# 3. LOAD DATA
# ============================================================

def load_data():

    print("\n" + "=" * 70)
    print("1. LOADING DATA")
    print("=" * 70)

    if not TRAIN_PATH.exists():
        raise FileNotFoundError(
            f"Training file not found:\n{TRAIN_PATH}"
        )

    if not TEST_PATH.exists():
        raise FileNotFoundError(
            f"Test file not found:\n{TEST_PATH}"
        )

    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)

    print(f"Train shape : {train_df.shape}")
    print(f"Test shape  : {test_df.shape}")

    return train_df, test_df


# ============================================================
# 4. DATA AUDIT
# ============================================================

def audit_dataset(df, name):

    print("\n" + "=" * 70)
    print(f"DATA AUDIT - {name}")
    print("=" * 70)

    print("\nMissing values:")
    print(df.isnull().sum().sum())

    print("\nDuplicate rows:")
    print(df.duplicated().sum())

    print("\nTarget distribution:")
    print(df["is_fraud"].value_counts())

    print("\nTarget percentage:")
    print(
        df["is_fraud"]
        .value_counts(normalize=True)
        .mul(100)
    )

    if "transaction_time" in df.columns:

        transaction_time = pd.to_datetime(
            df["transaction_time"],
            errors="coerce"
        )

        print("\nTransaction period:")
        print(f"Start: {transaction_time.min()}")
        print(f"End  : {transaction_time.max()}")


# ============================================================
# 5. FEATURE ENGINEERING
# ============================================================

def create_features(df):

    data = df.copy()

    # --------------------------------------------------------
    # Transaction time
    # --------------------------------------------------------

    data["transaction_time"] = pd.to_datetime(
        data["transaction_time"],
        errors="coerce"
    )

    data["transaction_hour"] = (
        data["transaction_time"].dt.hour
    )

    data["transaction_dayofweek"] = (
        data["transaction_time"].dt.dayofweek
    )

    data["transaction_day"] = (
        data["transaction_time"].dt.day
    )

    data["transaction_month"] = (
        data["transaction_time"].dt.month
    )

    data["is_weekend"] = (
        data["transaction_dayofweek"] >= 5
    ).astype(int)

    data["is_business_hours"] = (
        data["transaction_hour"].between(9, 18)
    ).astype(int)

    data["is_late_night"] = (
        (data["transaction_hour"] <= 5)
        |
        (data["transaction_hour"] >= 23)
    ).astype(int)

    # --------------------------------------------------------
    # Amount features
    # --------------------------------------------------------

    data["log_transaction_amount"] = np.log1p(
        data["transaction_amount"].clip(lower=0)
    )

    data["amount_to_monthly_spend"] = (
        data["transaction_amount"]
        /
        (data["avg_monthly_spend"] + 1.0)
    )

    data["amount_minus_monthly_spend"] = (
        data["transaction_amount"]
        -
        data["avg_monthly_spend"]
    )

    data["abs_amount_deviation"] = (
        data["amount_deviation_from_user_mean"]
        .abs()
    )

    # --------------------------------------------------------
    # Risk interactions
    # --------------------------------------------------------

    data["combined_merchant_ip_risk"] = (
        data["merchant_risk_score"]
        *
        data["ip_risk_score"]
    )

    data["max_risk_score"] = data[
        [
            "merchant_risk_score",
            "ip_risk_score"
        ]
    ].max(axis=1)

    data["mean_risk_score"] = data[
        [
            "merchant_risk_score",
            "ip_risk_score"
        ]
    ].mean(axis=1)

    # --------------------------------------------------------
    # Transaction velocity
    # --------------------------------------------------------

    data["txn_velocity_ratio"] = (
        data["txn_count_1h"]
        /
        (data["txn_count_24h"] + 1.0)
    )

    data["txn_count_difference"] = (
        data["txn_count_24h"]
        -
        data["txn_count_1h"]
    )

    data["international_velocity"] = (
        data["is_international"]
        *
        data["txn_velocity_ratio"]
    )

    # --------------------------------------------------------
    # Failed transactions
    # --------------------------------------------------------

    data["failed_txn_ratio"] = (
        data["failed_txn_count_24h"]
        /
        (data["txn_count_24h"] + 1.0)
    )

    # --------------------------------------------------------
    # International interactions
    # --------------------------------------------------------

    data["international_ip_risk"] = (
        data["is_international"]
        *
        data["ip_risk_score"]
    )

    data["international_merchant_risk"] = (
        data["is_international"]
        *
        data["merchant_risk_score"]
    )

    # --------------------------------------------------------
    # Geographic behavior
    # --------------------------------------------------------

    data["log_geo_distance"] = np.log1p(
        data["geo_distance_from_last_txn"].clip(lower=0)
    )

    # --------------------------------------------------------
    # Behavioral risk composite
    # --------------------------------------------------------

    data["behavioral_risk_composite"] = (
        0.30 * data["ip_risk_score"]
        +
        0.25 * data["merchant_risk_score"]
        +
        0.20 * data["failed_txn_ratio"]
        +
        0.15 * data["txn_velocity_ratio"]
        +
        0.10 * data["is_international"]
    )

    # --------------------------------------------------------
    # Explicitly remove leakage / identifiers
    # --------------------------------------------------------

    drop_columns = [
        "transaction_id",
        "customer_id",
        "merchant_id",
        "is_fraud",
        "post_auth_risk_score",
        "transaction_time"
    ]

    data.drop(
        columns=[
            column
            for column in drop_columns
            if column in data.columns
        ],
        inplace=True
    )

    return data


# ============================================================
# 6. PREPROCESSOR
# ============================================================

def build_preprocessor(X):

    categorical_columns = [
        column
        for column in [
            "payment_channel",
            "device_type"
        ]
        if column in X.columns
    ]

    numeric_columns = [
        column
        for column in X.columns
        if column not in categorical_columns
    ]

    numeric_pipeline = Pipeline(
        steps=[
            (
                "scaler",
                StandardScaler()
            )
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False
                )
            )
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                numeric_pipeline,
                numeric_columns
            ),
            (
                "cat",
                categorical_pipeline,
                categorical_columns
            )
        ],
        remainder="drop"
    )

    return (
        preprocessor,
        numeric_columns,
        categorical_columns
    )


# ============================================================
# 7. MODEL BUILDERS
# ============================================================

def build_model(
    model_type,
    X_reference
):

    preprocessor, _, _ = build_preprocessor(
        X_reference
    )

    if model_type == "LogisticRegression":

        estimator = LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            solver="liblinear",
            random_state=RANDOM_STATE
        )

    elif model_type == "RandomForest":

        estimator = RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            max_depth=14,
            min_samples_leaf=5,
            n_jobs=-1,
            random_state=RANDOM_STATE
        )

    elif model_type == "HistGradientBoosting":

        estimator = HistGradientBoostingClassifier(
            max_iter=300,
            learning_rate=0.08,
            max_leaf_nodes=31,
            min_samples_leaf=30,
            l2_regularization=1.0,
            random_state=RANDOM_STATE
        )

    else:
        raise ValueError(
            f"Unknown model type: {model_type}"
        )

    return Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor
            ),
            (
                "model",
                estimator
            )
        ]
    )


# ============================================================
# 8. MODEL FIT
# ============================================================

def fit_model(
    model_type,
    model,
    X_train,
    y_train
):

    print(
        f"\nTraining: {model_type}"
    )

    if model_type == "HistGradientBoosting":

        sample_weights = compute_sample_weight(
            class_weight="balanced",
            y=y_train
        )

        model.fit(
            X_train,
            y_train,
            model__sample_weight=sample_weights
        )

    else:

        model.fit(
            X_train,
            y_train
        )

    print("Training complete.")

    return model


# ============================================================
# 9. THRESHOLD METRICS
# ============================================================

def threshold_metrics(
    y_true,
    probabilities,
    threshold
):

    predictions = (
        probabilities >= threshold
    ).astype(int)

    return {
        "threshold": float(threshold),

        "precision": float(
            precision_score(
                y_true,
                predictions,
                zero_division=0
            )
        ),

        "recall": float(
            recall_score(
                y_true,
                predictions,
                zero_division=0
            )
        ),

        "f1": float(
            f1_score(
                y_true,
                predictions,
                zero_division=0
            )
        )
    }


# ============================================================
# 10. FULL PR-CURVE THRESHOLD TABLE
# ============================================================

def build_pr_threshold_table(
    y_true,
    probabilities
):

    precision, recall, thresholds = (
        precision_recall_curve(
            y_true,
            probabilities
        )
    )

    # precision_recall_curve returns
    # one extra precision/recall value
    # compared with thresholds.

    precision = precision[:-1]
    recall = recall[:-1]

    f1 = (
        2 * precision * recall
        /
        (
            precision
            +
            recall
            +
            1e-12
        )
    )

    table = pd.DataFrame({
        "threshold": thresholds,
        "precision": precision,
        "recall": recall,
        "f1": f1
    })

    table = table.replace(
        [np.inf, -np.inf],
        np.nan
    ).dropna()

    table = table.sort_values(
        "threshold"
    ).reset_index(drop=True)

    return table


# ============================================================
# 11. F1-OPTIMAL THRESHOLD
# ============================================================

def find_best_f1_threshold(
    y_true,
    probabilities
):

    table = build_pr_threshold_table(
        y_true,
        probabilities
    )

    if table.empty:
        return 0.5

    index = table["f1"].idxmax()

    return float(
        table.loc[index, "threshold"]
    )


# ============================================================
# 12. RECALL TARGET THRESHOLD
# ============================================================

def find_recall_threshold(
    y_true,
    probabilities,
    target_recall
):

    table = build_pr_threshold_table(
        y_true,
        probabilities
    )

    if table.empty:
        return 0.5

    candidates = table[
        table["recall"] >= target_recall
    ]

    if candidates.empty:

        # If target recall is impossible,
        # choose the threshold giving the
        # highest recall.
        index = table["recall"].idxmax()

        return float(
            table.loc[index, "threshold"]
        )

    # Highest threshold that still
    # achieves target recall.
    index = candidates["threshold"].idxmax()

    return float(
        table.loc[index, "threshold"]
    )


# ============================================================
# 13. PRECISION-CONSTRAINED THRESHOLD
# ============================================================

def find_best_recall_at_precision(
    threshold_table,
    precision_target
):

    candidates = threshold_table[
        threshold_table["precision"]
        >= precision_target
    ].copy()

    if candidates.empty:
        return None

    # Maximize recall.
    # If recall ties, maximize precision.
    candidates = candidates.sort_values(
        [
            "recall",
            "precision",
            "threshold"
        ],
        ascending=[
            False,
            False,
            True
        ]
    )

    return float(
        candidates.iloc[0]["threshold"]
    )


# ============================================================
# 14. BUSINESS RISK THRESHOLDS
# ============================================================

def select_risk_thresholds(
    threshold_table
):

    medium_threshold = (
        find_best_recall_at_precision(
            threshold_table,
            MEDIUM_PRECISION_TARGET
        )
    )

    high_threshold = (
        find_best_recall_at_precision(
            threshold_table,
            HIGH_PRECISION_TARGET
        )
    )

    # --------------------------------------------------------
    # Fallbacks only if requested precision level is impossible
    # --------------------------------------------------------

    if medium_threshold is None:

        medium_threshold = (
            find_best_f1_threshold_from_table(
                threshold_table
            )
        )

    if high_threshold is None:

        # If 40% precision is impossible,
        # use the highest available precision point.
        index = threshold_table[
            "precision"
        ].idxmax()

        high_threshold = float(
            threshold_table.loc[
                index,
                "threshold"
            ]
        )

    # --------------------------------------------------------
    # Make sure:
    #
    # medium threshold < high threshold
    # --------------------------------------------------------

    if high_threshold <= medium_threshold:

        higher_thresholds = threshold_table[
            threshold_table["threshold"]
            > medium_threshold
        ].copy()

        if not higher_thresholds.empty:

            high_threshold = float(
                higher_thresholds.sort_values(
                    [
                        "precision",
                        "recall"
                    ],
                    ascending=[
                        False,
                        False
                    ]
                ).iloc[0]["threshold"]
            )

        else:

            # Final fallback
            high_threshold = min(
                0.999999,
                medium_threshold + 1e-6
            )

    return (
        float(medium_threshold),
        float(high_threshold)
    )


def find_best_f1_threshold_from_table(
    threshold_table
):

    index = threshold_table[
        "f1"
    ].idxmax()

    return float(
        threshold_table.loc[
            index,
            "threshold"
        ]
    )


# ============================================================
# 15. COST-SENSITIVE THRESHOLD
# ============================================================

def find_cost_sensitive_threshold(
    y_true,
    probabilities,
    fp_cost,
    fn_cost
):

    # Use actual PR thresholds rather than
    # an arbitrary 0.001-0.999 grid.

    table = build_pr_threshold_table(
        y_true,
        probabilities
    )

    best_threshold = 0.5
    best_cost = np.inf

    for threshold in table[
        "threshold"
    ].values:

        predictions = (
            probabilities >= threshold
        ).astype(int)

        tn, fp, fn, tp = confusion_matrix(
            y_true,
            predictions,
            labels=[0, 1]
        ).ravel()

        total_cost = (
            fp * fp_cost
            +
            fn * fn_cost
        )

        if total_cost < best_cost:

            best_cost = total_cost
            best_threshold = float(
                threshold
            )

    return (
        best_threshold,
        float(best_cost)
    )


# ============================================================
# 16. COST ANALYSIS
# ============================================================

def cost_sensitive_analysis(
    y_true,
    probabilities
):

    print("\n" + "=" * 70)
    print("COST-SENSITIVE THRESHOLD ANALYSIS")
    print("=" * 70)

    results = []

    for scenario, (
        fp_cost,
        fn_cost
    ) in COST_SCENARIOS.items():

        threshold, cost = (
            find_cost_sensitive_threshold(
                y_true,
                probabilities,
                fp_cost,
                fn_cost
            )
        )

        metrics = threshold_metrics(
            y_true,
            probabilities,
            threshold
        )

        row = {
            "scenario": scenario,
            "fp_cost": fp_cost,
            "fn_cost": fn_cost,
            "threshold": threshold,
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"],
            "total_cost": cost
        }

        results.append(row)

        print(
            f"{scenario:30s} | "
            f"Threshold={threshold:.6f} | "
            f"Precision={metrics['precision']:.4f} | "
            f"Recall={metrics['recall']:.4f} | "
            f"F1={metrics['f1']:.4f}"
        )

    return pd.DataFrame(results)


# ============================================================
# 17. MODEL EVALUATION
# ============================================================

def evaluate_model(
    model,
    X,
    y,
    threshold,
    dataset_name,
    model_name
):

    probabilities = model.predict_proba(
        X
    )[:, 1]

    predictions = (
        probabilities >= threshold
    ).astype(int)

    precision = precision_score(
        y,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y,
        predictions,
        zero_division=0
    )

    roc_auc = roc_auc_score(
        y,
        probabilities
    )

    pr_auc = average_precision_score(
        y,
        probabilities
    )

    tn, fp, fn, tp = confusion_matrix(
        y,
        predictions,
        labels=[0, 1]
    ).ravel()

    print("\n" + "-" * 70)
    print(
        f"{model_name} - {dataset_name}"
    )
    print("-" * 70)

    print(
        f"Threshold : {threshold:.6f}"
    )

    print(
        f"Precision : {precision:.6f}"
    )

    print(
        f"Recall    : {recall:.6f}"
    )

    print(
        f"F1        : {f1:.6f}"
    )

    print(
        f"ROC-AUC   : {roc_auc:.6f}"
    )

    print(
        f"PR-AUC    : {pr_auc:.6f}"
    )

    print("\nConfusion Matrix:")

    print(
        np.array([
            [tn, fp],
            [fn, tp]
        ])
    )

    return {
        "model": model_name,
        "dataset": dataset_name,
        "threshold": float(threshold),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp)
    }


# ============================================================
# 18. MODEL COMPARISON
# ============================================================

def train_and_compare_models(
    X_train,
    y_train,
    X_val,
    y_val
):

    print("\n" + "=" * 70)
    print("MODEL COMPARISON")
    print("=" * 70)

    model_types = [
        "LogisticRegression",
        "RandomForest",
        "HistGradientBoosting"
    ]

    fitted_models = {}
    results = []

    display_names = {
        "LogisticRegression":
            "LogisticRegression_Balanced",

        "RandomForest":
            "RandomForest_Balanced",

        "HistGradientBoosting":
            "HistGradientBoosting_BalancedWeights"
    }

    for model_type in model_types:

        display_name = display_names[
            model_type
        ]

        model = build_model(
            model_type,
            X_train
        )

        model = fit_model(
            model_type,
            model,
            X_train,
            y_train
        )

        fitted_models[
            display_name
        ] = model

        val_probabilities = (
            model.predict_proba(
                X_val
            )[:, 1]
        )

        pr_auc = average_precision_score(
            y_val,
            val_probabilities
        )

        roc_auc = roc_auc_score(
            y_val,
            val_probabilities
        )

        threshold = find_recall_threshold(
            y_val,
            val_probabilities,
            TARGET_RECALL
        )

        metrics = threshold_metrics(
            y_val,
            val_probabilities,
            threshold
        )

        results.append({
            "model": display_name,
            "pr_auc": pr_auc,
            "roc_auc": roc_auc,
            "threshold": threshold,
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"]
        })

        print(
            f"\n{display_name}"
        )

        print(
            f"PR-AUC   : {pr_auc:.6f}"
        )

        print(
            f"ROC-AUC  : {roc_auc:.6f}"
        )

        print(
            f"Threshold: {threshold:.6f}"
        )

        print(
            f"Precision: {metrics['precision']:.6f}"
        )

        print(
            f"Recall   : {metrics['recall']:.6f}"
        )

        print(
            f"F1       : {metrics['f1']:.6f}"
        )

    comparison_df = (
        pd.DataFrame(results)
        .sort_values(
            "pr_auc",
            ascending=False
        )
        .reset_index(drop=True)
    )

    return (
        fitted_models,
        comparison_df
    )


# ============================================================
# 19. RISK TIERS
# ============================================================

def evaluate_risk_tiers(
    y_true,
    probabilities,
    medium_threshold,
    high_threshold,
    dataset_name
):

    print("\n" + "=" * 70)
    print(
        f"RISK TIER ANALYSIS - {dataset_name}"
    )
    print("=" * 70)

    risk_labels = np.select(
        [
            probabilities >= high_threshold,

            probabilities >= medium_threshold
        ],
        [
            "HIGH_RISK",

            "MEDIUM_RISK"
        ],
        default="LOW_RISK"
    )

    results = []

    for tier in [
        "LOW_RISK",
        "MEDIUM_RISK",
        "HIGH_RISK"
    ]:

        mask = (
            risk_labels == tier
        )

        transactions = int(
            mask.sum()
        )

        fraud_transactions = int(
            np.asarray(y_true)[mask].sum()
        )

        fraud_rate = (
            fraud_transactions
            /
            transactions
            if transactions > 0
            else 0.0
        )

        results.append({
            "dataset": dataset_name,
            "risk_tier": tier,
            "transactions": transactions,
            "fraud_transactions":
                fraud_transactions,
            "fraud_rate":
                fraud_rate
        })

        print(
            f"{tier:15s} | "
            f"Transactions={transactions:7d} | "
            f"Fraud={fraud_transactions:6d} | "
            f"Fraud Rate={fraud_rate:.4%}"
        )

    return pd.DataFrame(results)


# ============================================================
# 20. TOP-K ANALYSIS
# ============================================================

def top_k_analysis(
    y_true,
    probabilities,
    dataset_name
):

    print("\n" + "=" * 70)
    print(
        f"TOP-K FRAUD REVIEW ANALYSIS - "
        f"{dataset_name}"
    )
    print("=" * 70)

    y_array = np.asarray(
        y_true
    )

    order = np.argsort(
        probabilities
    )[::-1]

    sorted_labels = (
        y_array[order]
    )

    total_fraud = int(
        y_array.sum()
    )

    total_transactions = len(
        y_array
    )

    results = []

    for percentage in [
        0.01,
        0.02,
        0.05,
        0.10,
        0.20
    ]:

        k = max(
            1,
            int(
                np.ceil(
                    total_transactions
                    * percentage
                )
            )
        )

        reviewed = (
            sorted_labels[:k]
        )

        fraud_found = int(
            reviewed.sum()
        )

        precision = (
            fraud_found
            /
            k
        )

        recall = (
            fraud_found
            /
            total_fraud
            if total_fraud > 0
            else 0.0
        )

        results.append({
            "dataset": dataset_name,
            "top_k_percentage": percentage,
            "transactions_reviewed": k,
            "fraud_found": fraud_found,
            "precision": precision,
            "recall": recall
        })

        print(
            f"Top {percentage:.0%}: "
            f"Reviewed={k:6d} | "
            f"Fraud={fraud_found:6d} | "
            f"Precision={precision:.4f} | "
            f"Recall={recall:.4f}"
        )

    return pd.DataFrame(results)


# ============================================================
# 21. CALIBRATION
# ============================================================

def calibration_analysis(
    y_true,
    probabilities
):

    print("\n" + "=" * 70)
    print("CALIBRATION ANALYSIS")
    print("=" * 70)

    brier = brier_score_loss(
        y_true,
        probabilities
    )

    print(
        f"Brier Score: {brier:.6f}"
    )

    calibration_data = pd.DataFrame({
        "actual": np.asarray(y_true),
        "probability": probabilities
    })

    calibration_data["decile"] = pd.qcut(
        calibration_data[
            "probability"
        ],
        q=10,
        labels=False,
        duplicates="drop"
    )

    calibration = (
        calibration_data
        .groupby("decile")
        .agg(
            mean_predicted_probability=(
                "probability",
                "mean"
            ),
            actual_fraud_rate=(
                "actual",
                "mean"
            ),
            transactions=(
                "actual",
                "count"
            )
        )
        .reset_index()
    )

    print(
        "\nCalibration by probability decile:"
    )

    print(
        calibration.to_string(
            index=False
        )
    )

    return (
        float(brier),
        calibration
    )


# ============================================================
# 22. PERMUTATION IMPORTANCE
# ============================================================

def permutation_importance_analysis(
    model,
    X_val,
    y_val,
    model_name
):

    print("\n" + "=" * 70)
    print("PERMUTATION IMPORTANCE")
    print("=" * 70)

    print(
        "Calculating permutation importance..."
    )

    result = permutation_importance(
        model,
        X_val,
        y_val,
        scoring="average_precision",
        n_repeats=3,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    importance_df = pd.DataFrame({
        "feature": X_val.columns,
        "importance_mean":
            result.importances_mean,
        "importance_std":
            result.importances_std
    })

    importance_df = (
        importance_df
        .sort_values(
            "importance_mean",
            ascending=False
        )
        .reset_index(drop=True)
    )

    print("\nTop features:")

    print(
        importance_df.head(20).to_string(
            index=False
        )
    )

    # Plot
    top = (
        importance_df
        .head(15)
        .sort_values(
            "importance_mean"
        )
    )

    plt.figure(
        figsize=(10, 7)
    )

    plt.barh(
        top["feature"],
        top["importance_mean"]
    )

    plt.xlabel(
        "Mean decrease in PR-AUC"
    )

    plt.ylabel(
        "Feature"
    )

    plt.title(
        f"Permutation Importance - "
        f"{model_name}"
    )

    plt.tight_layout()

    plt.savefig(
        PLOT_DIR
        /
        "permutation_importance.png",
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()

    return importance_df


# ============================================================
# 23. FEATURE ABLATION
# ============================================================

def feature_ablation(
    model_type,
    X_train,
    y_train,
    X_val,
    y_val
):

    print("\n" + "=" * 70)
    print("FEATURE ABLATION")
    print("=" * 70)

    groups = {

        "all_features": [],

        "without_txn_count_24h": [
            "txn_count_24h"
        ],

        "without_failed_txn_count_24h": [
            "failed_txn_count_24h"
        ],

        "without_velocity_features": [
            "txn_count_1h",
            "txn_count_24h",
            "txn_velocity_ratio",
            "txn_count_difference",
            "international_velocity"
        ],

        "without_risk_features": [
            "merchant_risk_score",
            "ip_risk_score",
            "combined_merchant_ip_risk",
            "max_risk_score",
            "mean_risk_score",
            "international_ip_risk",
            "international_merchant_risk",
            "behavioral_risk_composite"
        ]
    }

    results = []

    for experiment_name, drop_columns in groups.items():

        print(
            f"\nAblation: {experiment_name}"
        )

        # ----------------------------------------------------
        # IMPORTANT FIX:
        #
        # Rebuild preprocessing using the reduced
        # feature set. We do NOT clone a pipeline whose
        # ColumnTransformer still expects removed columns.
        # ----------------------------------------------------

        train_columns = [
            column
            for column in X_train.columns
            if column not in drop_columns
        ]

        val_columns = [
            column
            for column in X_val.columns
            if column not in drop_columns
        ]

        X_train_subset = (
            X_train[train_columns]
        )

        X_val_subset = (
            X_val[val_columns]
        )

        model = build_model(
            model_type,
            X_train_subset
        )

        model = fit_model(
            model_type,
            model,
            X_train_subset,
            y_train
        )

        probabilities = (
            model.predict_proba(
                X_val_subset
            )[:, 1]
        )

        pr_auc = average_precision_score(
            y_val,
            probabilities
        )

        results.append({
            "experiment": experiment_name,
            "features_removed":
                ", ".join(drop_columns),
            "features_remaining":
                len(train_columns),
            "pr_auc": pr_auc
        })

        print(
            f"Features remaining: "
            f"{len(train_columns)}"
        )

        print(
            f"PR-AUC: {pr_auc:.6f}"
        )

    return (
        pd.DataFrame(results)
        .sort_values(
            "pr_auc",
            ascending=False
        )
        .reset_index(drop=True)
    )


# ============================================================
# 24. PLOTS
# ============================================================

def create_pr_curve(
    y_true,
    probabilities,
    model_name
):

    precision, recall, _ = (
        precision_recall_curve(
            y_true,
            probabilities
        )
    )

    plt.figure(
        figsize=(9, 6)
    )

    plt.plot(
        recall,
        precision,
        label=model_name
    )

    plt.xlabel(
        "Recall"
    )

    plt.ylabel(
        "Precision"
    )

    plt.title(
        "Precision-Recall Curve"
    )

    plt.legend()

    plt.grid(
        alpha=0.3
    )

    plt.tight_layout()

    plt.savefig(
        PLOT_DIR
        /
        "precision_recall_curve.png",
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()


def create_threshold_plot(
    threshold_table
):

    # To avoid an unreadable plot from
    # thousands of PR thresholds, plot
    # a quantile-based sample.

    if len(threshold_table) > 1000:

        indices = np.linspace(
            0,
            len(threshold_table) - 1,
            1000
        ).astype(int)

        plot_df = (
            threshold_table.iloc[
                indices
            ]
        )

    else:

        plot_df = threshold_table

    plt.figure(
        figsize=(10, 6)
    )

    plt.plot(
        plot_df["threshold"],
        plot_df["precision"],
        label="Precision"
    )

    plt.plot(
        plot_df["threshold"],
        plot_df["recall"],
        label="Recall"
    )

    plt.plot(
        plot_df["threshold"],
        plot_df["f1"],
        label="F1"
    )

    plt.xlabel(
        "Decision Threshold"
    )

    plt.ylabel(
        "Score"
    )

    plt.title(
        "Threshold vs Precision / Recall / F1"
    )

    plt.legend()

    plt.grid(
        alpha=0.3
    )

    plt.tight_layout()

    plt.savefig(
        PLOT_DIR
        /
        "threshold_analysis.png",
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()


# ============================================================
# 25. SAVE JSON
# ============================================================

def save_json(
    data,
    path
):

    def convert(value):

        if isinstance(
            value,
            np.integer
        ):
            return int(value)

        if isinstance(
            value,
            np.floating
        ):
            return float(value)

        if isinstance(
            value,
            np.ndarray
        ):
            return value.tolist()

        return value

    with open(
        path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            data,
            file,
            indent=4,
            default=convert
        )


# ============================================================
# 26. MAIN
# ============================================================

def main():

    print("\n")
    print("=" * 70)
    print("UPI FRAUD DETECTION PIPELINE V4.2")
    print("=" * 70)

    # ========================================================
    # LOAD
    # ========================================================

    train_df, test_df = load_data()

    audit_dataset(
        train_df,
        "TRAIN"
    )

    audit_dataset(
        test_df,
        "EXTERNAL TEST"
    )

    # ========================================================
    # CHRONOLOGICAL SPLIT
    # ========================================================

    print("\n" + "=" * 70)
    print("2. CHRONOLOGICAL SPLIT")
    print("=" * 70)

    train_df = (
        train_df
        .sort_values("transaction_time")
        .reset_index(drop=True)
    )

    test_df = (
        test_df
        .sort_values("transaction_time")
        .reset_index(drop=True)
    )

    split_index = int(
        len(train_df)
        *
        (1 - VALIDATION_SIZE)
    )

    model_train_df = (
        train_df.iloc[
            :split_index
        ].copy()
    )

    validation_df = (
        train_df.iloc[
            split_index:
        ].copy()
    )

    external_test_df = (
        test_df.copy()
    )

    print(
        f"Model train     : "
        f"{model_train_df.shape}"
    )

    print(
        f"Validation      : "
        f"{validation_df.shape}"
    )

    print(
        f"External test   : "
        f"{external_test_df.shape}"
    )

    print("\nFraud rates:")

    print(
        f"Model train : "
        f"{model_train_df['is_fraud'].mean():.4%}"
    )

    print(
        f"Validation  : "
        f"{validation_df['is_fraud'].mean():.4%}"
    )

    print(
        f"External    : "
        f"{external_test_df['is_fraud'].mean():.4%}"
    )

    # ========================================================
    # FEATURE ENGINEERING
    # ========================================================

    print("\n" + "=" * 70)
    print("3. FEATURE ENGINEERING")
    print("=" * 70)

    X_train = create_features(
        model_train_df
    )

    X_val = create_features(
        validation_df
    )

    X_test = create_features(
        external_test_df
    )

    y_train = (
        model_train_df["is_fraud"]
        .astype(int)
    )

    y_val = (
        validation_df["is_fraud"]
        .astype(int)
    )

    y_test = (
        external_test_df["is_fraud"]
        .astype(int)
    )

    # --------------------------------------------------------
    # Align validation/test columns with training columns
    # --------------------------------------------------------

    X_val = X_val.reindex(
        columns=X_train.columns
    )

    X_test = X_test.reindex(
        columns=X_train.columns
    )

    print(
        f"\nFinal feature count: "
        f"{X_train.shape[1]}"
    )

    print("\nFeatures:")

    for feature in X_train.columns:
        print(
            f" - {feature}"
        )

    _, numeric_columns, categorical_columns = (
        build_preprocessor(X_train)
    )

    print(
        f"\nNumeric features    : "
        f"{len(numeric_columns)}"
    )

    print(
        f"Categorical features: "
        f"{len(categorical_columns)}"
    )

    # ========================================================
    # MODEL COMPARISON
    # ========================================================

    (
        fitted_models,
        comparison_df
    ) = train_and_compare_models(
        X_train,
        y_train,
        X_val,
        y_val
    )

    print("\n" + "=" * 70)
    print("MODEL COMPARISON SUMMARY")
    print("=" * 70)

    print(
        comparison_df.to_string(
            index=False,
            float_format=lambda x:
                f"{x:.6f}"
        )
    )

    comparison_df.to_csv(
        REPORT_DIR
        /
        "model_comparison.csv",
        index=False
    )

    # ========================================================
    # BEST MODEL
    # ========================================================

    best_model_name = (
        comparison_df.iloc[0]["model"]
    )

    best_model = (
        fitted_models[
            best_model_name
        ]
    )

    print("\n" + "=" * 70)
    print("BEST MODEL")
    print("=" * 70)

    print(
        best_model_name
    )

    # Map display name back to model type
    model_type_map = {
        "LogisticRegression_Balanced":
            "LogisticRegression",

        "RandomForest_Balanced":
            "RandomForest",

        "HistGradientBoosting_BalancedWeights":
            "HistGradientBoosting"
    }

    best_model_type = (
        model_type_map[
            best_model_name
        ]
    )

    # ========================================================
    # PROBABILITIES
    # ========================================================

    val_probabilities = (
        best_model.predict_proba(
            X_val
        )[:, 1]
    )

    test_probabilities = (
        best_model.predict_proba(
            X_test
        )[:, 1]
    )

    # ========================================================
    # PR THRESHOLD TABLE
    # ========================================================

    print("\n" + "=" * 70)
    print("FULL PRECISION-RECALL THRESHOLD ANALYSIS")
    print("=" * 70)

    threshold_table = (
        build_pr_threshold_table(
            y_val,
            val_probabilities
        )
    )

    threshold_table.to_csv(
        REPORT_DIR
        /
        "full_pr_threshold_analysis.csv",
        index=False
    )

    # --------------------------------------------------------
    # 80% recall threshold
    # --------------------------------------------------------

    recall_threshold = (
        find_recall_threshold(
            y_val,
            val_probabilities,
            TARGET_RECALL
        )
    )

    # --------------------------------------------------------
    # F1 threshold
    # --------------------------------------------------------

    f1_threshold = (
        find_best_f1_threshold(
            y_val,
            val_probabilities
        )
    )

    # --------------------------------------------------------
    # F1 metrics
    # --------------------------------------------------------

    f1_validation_metrics = (
        threshold_metrics(
            y_val,
            val_probabilities,
            f1_threshold
        )
    )

    print(
        f"\n80% recall threshold : "
        f"{recall_threshold:.6f}"
    )

    print(
        f"F1-optimal threshold : "
        f"{f1_threshold:.6f}"
    )

    print(
        f"F1-optimal precision  : "
        f"{f1_validation_metrics['precision']:.6f}"
    )

    print(
        f"F1-optimal recall     : "
        f"{f1_validation_metrics['recall']:.6f}"
    )

    print(
        f"F1-optimal F1         : "
        f"{f1_validation_metrics['f1']:.6f}"
    )

    # ========================================================
    # BUSINESS RISK THRESHOLDS
    # ========================================================

    (
        medium_threshold,
        high_threshold
    ) = select_risk_thresholds(
        threshold_table
    )

    medium_validation_metrics = (
        threshold_metrics(
            y_val,
            val_probabilities,
            medium_threshold
        )
    )

    high_validation_metrics = (
        threshold_metrics(
            y_val,
            val_probabilities,
            high_threshold
        )
    )

    print("\n" + "=" * 70)
    print("DATA-DRIVEN BUSINESS RISK THRESHOLDS")
    print("=" * 70)

    print(
        f"Medium-risk threshold: "
        f"{medium_threshold:.6f}"
    )

    print(
        f"Medium precision     : "
        f"{medium_validation_metrics['precision']:.6f}"
    )

    print(
        f"Medium recall        : "
        f"{medium_validation_metrics['recall']:.6f}"
    )

    print(
        f"\nHigh-risk threshold  : "
        f"{high_threshold:.6f}"
    )

    print(
        f"High precision       : "
        f"{high_validation_metrics['precision']:.6f}"
    )

    print(
        f"High recall          : "
        f"{high_validation_metrics['recall']:.6f}"
    )

    print("\nRisk policy:")

    print(
        f"LOW    : probability < "
        f"{medium_threshold:.6f}"
    )

    print(
        f"MEDIUM : {medium_threshold:.6f} "
        f"<= probability < "
        f"{high_threshold:.6f}"
    )

    print(
        f"HIGH   : probability >= "
        f"{high_threshold:.6f}"
    )

    # ========================================================
    # PLOTS
    # ========================================================

    create_pr_curve(
        y_val,
        val_probabilities,
        best_model_name
    )

    create_threshold_plot(
        threshold_table
    )

    # ========================================================
    # COST-SENSITIVE ANALYSIS
    # ========================================================

    cost_df = (
        cost_sensitive_analysis(
            y_val,
            val_probabilities
        )
    )

    cost_df.to_csv(
        REPORT_DIR
        /
        "cost_sensitive_thresholds.csv",
        index=False
    )

    # ========================================================
    # VALIDATION EVALUATION
    # ========================================================

    validation_recall_metrics = (
        evaluate_model(
            best_model,
            X_val,
            y_val,
            recall_threshold,
            "Validation_80pct_Recall",
            best_model_name
        )
    )

    validation_f1_metrics_full = (
        evaluate_model(
            best_model,
            X_val,
            y_val,
            f1_threshold,
            "Validation_F1_Optimal",
            best_model_name
        )
    )

    # ========================================================
    # EXTERNAL TEST
    # ========================================================

    print("\n" + "=" * 70)
    print("EXTERNAL TEMPORAL TEST")
    print("=" * 70)

    external_recall_metrics = (
        evaluate_model(
            best_model,
            X_test,
            y_test,
            recall_threshold,
            "External_Test_80pct_Recall",
            best_model_name
        )
    )

    external_f1_metrics = (
        evaluate_model(
            best_model,
            X_test,
            y_test,
            f1_threshold,
            "External_Test_F1_Optimal",
            best_model_name
        )
    )

    external_medium_metrics = (
        evaluate_model(
            best_model,
            X_test,
            y_test,
            medium_threshold,
            "External_Test_Medium_Risk",
            best_model_name
        )
    )

    external_high_metrics = (
        evaluate_model(
            best_model,
            X_test,
            y_test,
            high_threshold,
            "External_Test_High_Risk",
            best_model_name
        )
    )

    # ========================================================
    # RISK TIERS
    # ========================================================

    validation_tiers = (
        evaluate_risk_tiers(
            y_val,
            val_probabilities,
            medium_threshold,
            high_threshold,
            "Validation"
        )
    )

    external_tiers = (
        evaluate_risk_tiers(
            y_test,
            test_probabilities,
            medium_threshold,
            high_threshold,
            "External_Test"
        )
    )

    risk_tiers_df = pd.concat(
        [
            validation_tiers,
            external_tiers
        ],
        ignore_index=True
    )

    risk_tiers_df.to_csv(
        REPORT_DIR
        /
        "risk_tiers.csv",
        index=False
    )

    # ========================================================
    # TOP-K
    # ========================================================

    validation_topk = (
        top_k_analysis(
            y_val,
            val_probabilities,
            "Validation"
        )
    )

    external_topk = (
        top_k_analysis(
            y_test,
            test_probabilities,
            "External_Test"
        )
    )

    topk_df = pd.concat(
        [
            validation_topk,
            external_topk
        ],
        ignore_index=True
    )

    topk_df.to_csv(
        REPORT_DIR
        /
        "top_k_analysis.csv",
        index=False
    )

    # ========================================================
    # CALIBRATION
    # ========================================================

    (
        brier_score,
        calibration_df
    ) = calibration_analysis(
        y_val,
        val_probabilities
    )

    calibration_df.to_csv(
        REPORT_DIR
        /
        "calibration_validation.csv",
        index=False
    )

    # ========================================================
    # PERMUTATION IMPORTANCE
    # ========================================================

    importance_df = (
        permutation_importance_analysis(
            best_model,
            X_val,
            y_val,
            best_model_name
        )
    )

    importance_df.to_csv(
        REPORT_DIR
        /
        "permutation_importance.csv",
        index=False
    )

    # ========================================================
    # FEATURE ABLATION
    # ========================================================

    ablation_df = (
        feature_ablation(
            best_model_type,
            X_train,
            y_train,
            X_val,
            y_val
        )
    )

    print("\n" + "=" * 70)
    print("ABLATION RESULTS")
    print("=" * 70)

    print(
        ablation_df.to_string(
            index=False,
            float_format=lambda x:
                f"{x:.6f}"
        )
    )

    ablation_df.to_csv(
        REPORT_DIR
        /
        "feature_ablation.csv",
        index=False
    )

    # ========================================================
    # SAVE MODEL
    # ========================================================

    model_path = (
        MODEL_DIR
        /
        "upi_fraud_model_v4_2.joblib"
    )

    joblib.dump(
        best_model,
        model_path
    )

    print("\n" + "=" * 70)
    print("MODEL SAVED")
    print("=" * 70)

    print(
        model_path
    )

    # ========================================================
    # FEATURE METADATA
    # ========================================================

    feature_metadata = {
        "feature_count":
            len(X_train.columns),

        "features":
            list(X_train.columns),

        "numeric_features":
            numeric_columns,

        "categorical_features":
            categorical_columns,

        "excluded_columns": [
            "transaction_id",
            "customer_id",
            "merchant_id",
            "is_fraud",
            "transaction_time",
            "post_auth_risk_score"
        ]
    }

    save_json(
        feature_metadata,
        REPORT_DIR
        /
        "feature_metadata.json"
    )

    # ========================================================
    # FINAL METRICS JSON
    # ========================================================

    final_metrics = {

        "pipeline_version":
            "V4.2",

        "model":
            best_model_name,

        "model_selection_metric":
            "validation_PR_AUC",

        "validation": {

            "pr_auc":
                validation_recall_metrics[
                    "pr_auc"
                ],

            "roc_auc":
                validation_recall_metrics[
                    "roc_auc"
                ],

            "threshold":
                recall_threshold,

            "precision":
                validation_recall_metrics[
                    "precision"
                ],

            "recall":
                validation_recall_metrics[
                    "recall"
                ],

            "f1":
                validation_recall_metrics[
                    "f1"
                ]
        },

        "validation_f1_optimal": {

            "threshold":
                f1_threshold,

            "precision":
                validation_f1_metrics_full[
                    "precision"
                ],

            "recall":
                validation_f1_metrics_full[
                    "recall"
                ],

            "f1":
                validation_f1_metrics_full[
                    "f1"
                ]
        },

        "external_test": {

            "threshold":
                recall_threshold,

            "precision":
                external_recall_metrics[
                    "precision"
                ],

            "recall":
                external_recall_metrics[
                    "recall"
                ],

            "f1":
                external_recall_metrics[
                    "f1"
                ],

            "roc_auc":
                external_recall_metrics[
                    "roc_auc"
                ],

            "pr_auc":
                external_recall_metrics[
                    "pr_auc"
                ],

            "confusion_matrix": [
                [
                    external_recall_metrics["tn"],
                    external_recall_metrics["fp"]
                ],
                [
                    external_recall_metrics["fn"],
                    external_recall_metrics["tp"]
                ]
            ]
        },

        "external_test_f1_optimal": {

            "threshold":
                f1_threshold,

            "precision":
                external_f1_metrics[
                    "precision"
                ],

            "recall":
                external_f1_metrics[
                    "recall"
                ],

            "f1":
                external_f1_metrics[
                    "f1"
                ]
        },

        "business_thresholds": {

            "medium":
                medium_threshold,

            "high":
                high_threshold
        },

        "external_medium_risk": {

            "precision":
                external_medium_metrics[
                    "precision"
                ],

            "recall":
                external_medium_metrics[
                    "recall"
                ],

            "f1":
                external_medium_metrics[
                    "f1"
                ]
        },

        "external_high_risk": {

            "precision":
                external_high_metrics[
                    "precision"
                ],

            "recall":
                external_high_metrics[
                    "recall"
                ],

            "f1":
                external_high_metrics[
                    "f1"
                ]
        },

        "calibration": {

            "validation_brier_score":
                brier_score
        },

        "target_recall":
            TARGET_RECALL,

        "medium_precision_target":
            MEDIUM_PRECISION_TARGET,

        "high_precision_target":
            HIGH_PRECISION_TARGET
    }

    save_json(
        final_metrics,
        REPORT_DIR
        /
        "final_metrics.json"
    )

    # ========================================================
    # README METRICS
    # ========================================================

    readme_metrics = f"""
# UPI Fraud Detection - V4.2

## Model

{best_model_name}

The model was selected using validation PR-AUC.

## Dataset

- Model training transactions: {len(X_train):,}
- Validation transactions: {len(X_val):,}
- External temporal test transactions: {len(X_test):,}

## Class Imbalance

- Validation fraud rate: {y_val.mean():.4%}
- External test fraud rate: {y_test.mean():.4%}

## Validation Performance

Using the threshold selected for approximately {TARGET_RECALL:.0%} recall:

- PR-AUC: {validation_recall_metrics["pr_auc"]:.4f}
- ROC-AUC: {validation_recall_metrics["roc_auc"]:.4f}
- Precision: {validation_recall_metrics["precision"]:.4f}
- Recall: {validation_recall_metrics["recall"]:.4f}
- F1: {validation_recall_metrics["f1"]:.4f}

## External Temporal Test

The threshold was selected using validation data and then frozen for the external test.

- PR-AUC: {external_recall_metrics["pr_auc"]:.4f}
- ROC-AUC: {external_recall_metrics["roc_auc"]:.4f}
- Precision: {external_recall_metrics["precision"]:.4f}
- Recall: {external_recall_metrics["recall"]:.4f}
- F1: {external_recall_metrics["f1"]:.4f}

## F1-Optimal Operating Point

Validation threshold:

{f1_threshold:.6f}

Validation:

- Precision: {validation_f1_metrics_full["precision"]:.4f}
- Recall: {validation_f1_metrics_full["recall"]:.4f}
- F1: {validation_f1_metrics_full["f1"]:.4f}

External:

- Precision: {external_f1_metrics["precision"]:.4f}
- Recall: {external_f1_metrics["recall"]:.4f}
- F1: {external_f1_metrics["f1"]:.4f}

## Business Risk Thresholds

Medium-risk threshold:

{medium_threshold:.6f}

High-risk threshold:

{high_threshold:.6f}

These thresholds were selected using validation data.

## Calibration

Validation Brier score:

{brier_score:.6f}

Raw model scores should not automatically be interpreted as perfectly calibrated probabilities.

## Leakage Prevention

The following fields were excluded:

- transaction_id
- customer_id
- merchant_id
- is_fraud
- transaction_time
- post_auth_risk_score

The post-authorization risk score was specifically excluded because it would not be available at the point where a real-time fraud decision is made.

## Evaluation Design

The dataset was split chronologically.

Earlier transactions were used for model training.

Later transactions from the training period were used for validation.

A completely later external test period was retained for final evaluation.

Decision thresholds were selected on validation data and not tuned using the external test set.

## Main Behavioral Signals

The feature engineering pipeline includes:

- transaction velocity
- failed transaction behavior
- transaction amount deviation
- transaction amount relative to monthly spending
- geographic distance
- merchant risk
- IP risk
- international transaction interactions
- time-of-day behavior
"""

    with open(
        REPORT_DIR
        /
        "README_metrics.md",
        "w",
        encoding="utf-8"
    ) as file:

        file.write(
            readme_metrics.strip()
        )

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)

    print(
        f"\nBest model: "
        f"{best_model_name}"
    )

    print(
        "\nExternal temporal test "
        "at validation-selected 80%-recall threshold:"
    )

    print(
        f"PR-AUC   : "
        f"{external_recall_metrics['pr_auc']:.6f}"
    )

    print(
        f"ROC-AUC  : "
        f"{external_recall_metrics['roc_auc']:.6f}"
    )

    print(
        f"Precision: "
        f"{external_recall_metrics['precision']:.6f}"
    )

    print(
        f"Recall   : "
        f"{external_recall_metrics['recall']:.6f}"
    )

    print(
        f"F1       : "
        f"{external_recall_metrics['f1']:.6f}"
    )

    print("\nConfusion matrix:")

    print(
        np.array([
            [
                external_recall_metrics["tn"],
                external_recall_metrics["fp"]
            ],
            [
                external_recall_metrics["fn"],
                external_recall_metrics["tp"]
            ]
        ])
    )

    print("\nOperating thresholds:")

    print(
        f"80% recall : "
        f"{recall_threshold:.6f}"
    )

    print(
        f"F1 optimal : "
        f"{f1_threshold:.6f}"
    )

    print(
        f"Medium risk: "
        f"{medium_threshold:.6f}"
    )

    print(
        f"High risk  : "
        f"{high_threshold:.6f}"
    )

    print("\nOutput files:")

    print(
        f"Model   : {model_path}"
    )

    print(
        f"Reports : {REPORT_DIR}"
    )

    print(
        f"Plots   : {PLOT_DIR}"
    )

    print("\n" + "=" * 70)
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 70)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()