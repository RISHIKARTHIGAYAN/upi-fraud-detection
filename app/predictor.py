import json
import math
from pathlib import Path

import joblib
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = PROJECT_ROOT / "outputs" / "models" / "upi_fraud_model_v4_2.joblib"
METRICS_PATH = PROJECT_ROOT / "outputs" / "reports" / "final_metrics.json"


# -------------------------------------------------------------------
# Required API input fields
# -------------------------------------------------------------------

REQUIRED_FIELDS = [
    "transaction_time",
    "account_age_days",
    "credit_score_band",
    "kyc_level",
    "avg_monthly_spend",
    "merchant_risk_score",
    "transaction_amount",
    "payment_channel",
    "device_type",
    "is_international",
    "ip_risk_score",
    "txn_count_1h",
    "txn_count_24h",
    "failed_txn_count_24h",
    "geo_distance_from_last_txn",
    "amount_deviation_from_user_mean",
]


NUMERIC_FIELDS = [
    "account_age_days",
    "credit_score_band",
    "kyc_level",
    "avg_monthly_spend",
    "merchant_risk_score",
    "transaction_amount",
    "is_international",
    "ip_risk_score",
    "txn_count_1h",
    "txn_count_24h",
    "failed_txn_count_24h",
    "geo_distance_from_last_txn",
    "amount_deviation_from_user_mean",
]


# -------------------------------------------------------------------
# Load trained model
# -------------------------------------------------------------------

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model file not found: {MODEL_PATH}"
    )

model = joblib.load(MODEL_PATH)


# -------------------------------------------------------------------
# Load thresholds selected during V4.2 validation
# -------------------------------------------------------------------

if METRICS_PATH.exists():
    with open(METRICS_PATH, "r", encoding="utf-8") as file:
        metrics = json.load(file)

    RECALL_THRESHOLD = float(
        metrics["validation"]["threshold"]
    )

    MEDIUM_THRESHOLD = float(
        metrics["business_thresholds"]["medium"]
    )

    HIGH_THRESHOLD = float(
        metrics["business_thresholds"]["high"]
    )

else:
    # Fallback values from the V4.2 run
    RECALL_THRESHOLD = 0.475376
    MEDIUM_THRESHOLD = 0.757239
    HIGH_THRESHOLD = 0.859161


# -------------------------------------------------------------------
# Validate API input
# -------------------------------------------------------------------

def validate_transaction(transaction):
    if not isinstance(transaction, dict):
        raise ValueError(
            "Request body must be a JSON object."
        )

    missing_fields = [
        field
        for field in REQUIRED_FIELDS
        if field not in transaction
    ]

    if missing_fields:
        raise KeyError(
            ", ".join(missing_fields)
        )

    # Validate numeric fields
    for field in NUMERIC_FIELDS:
        value = transaction[field]

        if isinstance(value, bool):
            raise ValueError(
                f"Field '{field}' must be numeric."
            )

        try:
            numeric_value = float(value)
        except (TypeError, ValueError):
            raise ValueError(
                f"Field '{field}' must be numeric."
            )

        if not math.isfinite(numeric_value):
            raise ValueError(
                f"Field '{field}' must be a finite number."
            )

    # Validate timestamp
    try:
        timestamp = pd.to_datetime(
            transaction["transaction_time"],
            errors="raise"
        )
    except Exception:
        raise ValueError(
            "Field 'transaction_time' must be a valid date/time."
        )

    if pd.isna(timestamp):
        raise ValueError(
            "Field 'transaction_time' must be a valid date/time."
        )

    # Validate categorical/string fields
    for field in ["payment_channel", "device_type"]:
        value = transaction[field]

        if not isinstance(value, str) or not value.strip():
            raise ValueError(
                f"Field '{field}' must be a non-empty string."
            )

    return True


# -------------------------------------------------------------------
# Feature engineering
# Must remain consistent with V4.2 training pipeline
# -------------------------------------------------------------------

def create_features(df):
    df = df.copy()

    timestamp = pd.to_datetime(df["transaction_time"])

    df["transaction_hour"] = timestamp.dt.hour
    df["transaction_dayofweek"] = timestamp.dt.dayofweek
    df["transaction_day"] = timestamp.dt.day
    df["transaction_month"] = timestamp.dt.month

    df["is_weekend"] = (
        df["transaction_dayofweek"] >= 5
    ).astype(int)

    df["is_business_hours"] = (
        (df["transaction_hour"] >= 9)
        & (df["transaction_hour"] <= 18)
    ).astype(int)

    df["is_late_night"] = (
        (df["transaction_hour"] <= 5)
        | (df["transaction_hour"] >= 23)
    ).astype(int)

    df["log_transaction_amount"] = (
        df["transaction_amount"]
        .clip(lower=0)
        .apply(math.log1p)
    )

    df["amount_to_monthly_spend"] = (
        df["transaction_amount"]
        / (df["avg_monthly_spend"] + 1.0)
    )

    df["amount_minus_monthly_spend"] = (
        df["transaction_amount"]
        - df["avg_monthly_spend"]
    )

    df["abs_amount_deviation"] = (
        df["amount_deviation_from_user_mean"].abs()
    )

    df["combined_merchant_ip_risk"] = (
        df["merchant_risk_score"]
        * df["ip_risk_score"]
    )

    df["max_risk_score"] = df[
        ["merchant_risk_score", "ip_risk_score"]
    ].max(axis=1)

    df["mean_risk_score"] = df[
        ["merchant_risk_score", "ip_risk_score"]
    ].mean(axis=1)

    df["txn_velocity_ratio"] = (
        df["txn_count_1h"]
        / (df["txn_count_24h"] + 1.0)
    )

    df["txn_count_difference"] = (
        df["txn_count_24h"]
        - df["txn_count_1h"]
    )

    df["international_velocity"] = (
        df["is_international"]
        * df["txn_velocity_ratio"]
    )

    df["failed_txn_ratio"] = (
        df["failed_txn_count_24h"]
        / (df["txn_count_24h"] + 1.0)
    )

    df["international_ip_risk"] = (
        df["is_international"]
        * df["ip_risk_score"]
    )

    df["international_merchant_risk"] = (
        df["is_international"]
        * df["merchant_risk_score"]
    )

    df["log_geo_distance"] = (
        df["geo_distance_from_last_txn"]
        .clip(lower=0)
        .apply(math.log1p)
    )

    df["behavioral_risk_composite"] = (
        0.30 * df["ip_risk_score"]
        + 0.25 * df["merchant_risk_score"]
        + 0.20 * df["failed_txn_ratio"]
        + 0.15 * df["txn_velocity_ratio"]
        + 0.10 * df["is_international"]
    )

    # Remove identifiers, target, post-authorization feature,
    # and raw timestamp.
    columns_to_drop = [
        "transaction_id",
        "customer_id",
        "merchant_id",
        "is_fraud",
        "post_auth_risk_score",
        "transaction_time",
    ]

    df = df.drop(
        columns=[
            col
            for col in columns_to_drop
            if col in df.columns
        ],
        errors="ignore",
    )

    return df


# -------------------------------------------------------------------
# Risk tier
# -------------------------------------------------------------------

def classify_risk(score):
    if score >= HIGH_THRESHOLD:
        return "HIGH"

    if score >= MEDIUM_THRESHOLD:
        return "MEDIUM"

    return "LOW"


# -------------------------------------------------------------------
# Operational decision
# -------------------------------------------------------------------

def get_operating_decision(score):
    if score >= HIGH_THRESHOLD:
        return "BLOCK_OR_MANUAL_REVIEW"

    if score >= MEDIUM_THRESHOLD:
        return "ENHANCED_REVIEW"

    if score >= RECALL_THRESHOLD:
        return "FLAG_FOR_REVIEW"

    return "ALLOW"


# -------------------------------------------------------------------
# Recommendation
# -------------------------------------------------------------------

def get_recommendation(score):
    return get_operating_decision(score)


# -------------------------------------------------------------------
# Prediction
# -------------------------------------------------------------------

def predict_transaction(transaction):
    validate_transaction(transaction)

    df = pd.DataFrame([transaction])

    features = create_features(df)

    fraud_score = float(
        model.predict_proba(features)[0, 1]
    )

    risk_tier = classify_risk(fraud_score)

    operating_decision = get_operating_decision(
        fraud_score
    )

    recommendation = get_recommendation(
        fraud_score
    )

    return {
        "fraud_score": round(fraud_score, 6),
        "risk_tier": risk_tier,
        "operating_decision": operating_decision,
        "recommendation": recommendation,
        "thresholds": {
            "80_percent_recall": round(
                RECALL_THRESHOLD, 6
            ),
            "medium_risk": round(
                MEDIUM_THRESHOLD, 6
            ),
            "high_risk": round(
                HIGH_THRESHOLD, 6
            ),
        },
    }