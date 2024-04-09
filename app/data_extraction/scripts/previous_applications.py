from functions import aggregate_and_join, encode_one_hot, replace_365243_with_nan, read_validate
from typing import List
import pandas as pd
import config


def get_data(path: str, table: str) -> pd.DataFrame:
    """
    Processes loan application data from previous applications and installment payments,
    performing feature engineering, encoding categorical variables, and aggregating data.

    Parameters:
    - path: Path to the directory containing the dataset files.

    Returns:
    - agg_prev: Aggregated DataFrame with engineered and aggregated features.
    """
    prev = read_validate(path, table, config.PREVIOUS_APPLICATIONS_DTYPES)
    pay = read_validate(path, "instalments_payments.csv", config.INSTALMENTS_PAYMENTS_DTYPES)
    ohe_columns = ["NAME_CONTRACT_STATUS", "NAME_CONTRACT_TYPE", "CHANNEL_TYPE",
                   "NAME_TYPE_SUITE", "NAME_YIELD_GROUP", "PRODUCT_COMBINATION",
                   "NAME_PRODUCT_TYPE", "NAME_CLIENT_TYPE"]
    prev, categorical_cols = encode_one_hot(prev, ohe_columns, nan_as_category=False)
    prev = feature_engineering(prev)
    active_df = active_loans_processing(prev, pay)
    invalid_cols = ["DAYS_FIRST_DRAWING", "DAYS_FIRST_DUE",
                    "DAYS_LAST_DUE_1ST_VERSION", "DAYS_LAST_DUE", "DAYS_TERMINATION"]
    prev = replace_365243_with_nan(prev, invalid_cols)
    agg_prev = aggregate_data(prev, active_df, pay, categorical_cols)
    return agg_prev


def feature_engineering(prev: pd.DataFrame) -> pd.DataFrame:
    """
    Performs feature engineering on the previous application DataFrame.

    Parameters:
    - prev: DataFrame containing previous application data.

    Returns:
    - prev: DataFrame with new engineered features.
    """
    prev["APPLICATION_CREDIT_DIFF"] = prev["AMT_APPLICATION"] - prev["AMT_CREDIT"]
    prev["APPLICATION_CREDIT_RATIO"] = prev["AMT_APPLICATION"] / prev["AMT_CREDIT"]
    prev["CREDIT_TO_ANNUITY_RATIO"] = prev["AMT_CREDIT"] / prev["AMT_ANNUITY"]
    prev["DOWN_PAYMENT_TO_CREDIT"] = prev["AMT_DOWN_PAYMENT"] / prev["AMT_CREDIT"]
    total_payment = prev["AMT_ANNUITY"] * prev["CNT_PAYMENT"]
    prev["SIMPLE_INTERESTS"] = (total_payment / prev["AMT_CREDIT"] - 1) / prev["CNT_PAYMENT"]
    return prev


def active_loans_processing(prev: pd.DataFrame, pay: pd.DataFrame) -> pd.DataFrame:
    """
    Processes active loans to calculate repayment ratios and remaining debt.

    Parameters:
    - param prev: DataFrame containing previous application data.
    - param pay: DataFrame containing installment payment data.

    Returns:
    - active_agg_df: DataFrame aggregated with active loan repayment data.
    """
    approved = prev[prev["NAME_CONTRACT_STATUS_Approved"] == 1]
    active_df = approved[approved["DAYS_LAST_DUE"] == 365243]
    active_pay = pay[pay["SK_ID_PREV"].isin(active_df["SK_ID_PREV"])]
    active_pay_agg = active_pay.groupby("SK_ID_PREV")[["AMT_INSTALMENT", "AMT_PAYMENT"]].sum().reset_index()
    active_pay_agg["INSTALMENT_PAYMENT_DIFF"] = active_pay_agg["AMT_INSTALMENT"] - active_pay_agg["AMT_PAYMENT"]
    active_df = active_df.merge(active_pay_agg, on="SK_ID_PREV", how="left")
    active_df["REMAINING_DEBT"] = active_df["AMT_CREDIT"] - active_df["AMT_PAYMENT"]
    active_df["REPAYMENT_RATIO"] = active_df["AMT_PAYMENT"] / active_df["AMT_CREDIT"]
    active_agg_df = aggregate_and_join(active_df, "PREV_ACTIVE_", config.PREVIOUS_ACTIVE_AGG)
    active_agg_df["TOTAL_REPAYMENT_RATIO"] = \
        active_agg_df["PREV_ACTIVE_AMT_PAYMENT_SUM"] / active_agg_df["PREV_ACTIVE_AMT_CREDIT_SUM"]

    return active_agg_df


def aggregate_data(prev: pd.DataFrame, active_agg_df: pd.DataFrame, pay: pd.DataFrame, categorical_cols: List[str]) -> pd.DataFrame:
    """
    Aggregates data from previous applications and payment history, integrating various metrics, including loan types
    and late payments, and merging with active aggregates.

    Parameters:
    - prev: DataFrame containing previous application data.
    - active_agg_df: DataFrame containing active loan aggregates.
    - pay: DataFrame containing payment history.
    - categorical_cols: List of categorical columns to aggregate.

    Returns:
    - agg_prev: DataFrame with aggregated previous application data.
    """
    prev["DAYS_LAST_DUE_DIFF"] = prev["DAYS_LAST_DUE_1ST_VERSION"] - prev["DAYS_LAST_DUE"]
    categorical_agg = {key: ["mean"] for key in categorical_cols}
    agg_prev = aggregate_and_join(prev, "PREV_", {**config.PREVIOUS_AGG, **categorical_agg})
    agg_prev = agg_prev.merge(active_agg_df, how="left", on="SK_ID_CURR")
    agg_prev = process_loan_types(prev, agg_prev)
    agg_prev = process_late_payments(pay, prev, agg_prev)
    return agg_prev


def process_loan_types(prev: pd.DataFrame, agg_prev: pd.DataFrame) -> pd.DataFrame:
    """
    Processes and aggregates data by loan type (Consumer loans, Cash loans).

    Parameters:
    - param prev: DataFrame containing previous application data.
    - param agg_prev: Aggregated DataFrame to merge processed loan type data.

    Returns:
    - agg_prev: Aggregated DataFrame with loan type data processed and added.
    """
    for loan_type in ["Consumer loans", "Cash loans"]:
        type_df = prev[prev[f"NAME_CONTRACT_TYPE_{loan_type}"] == 1]
        prefix = f'PREV_{loan_type.split(" ")[0]}_'
        agg_prev = aggregate_and_join(type_df, prefix, config.PREVIOUS_LOAN_TYPE_AGG, df_to_merge=agg_prev)
    return agg_prev


def process_late_payments(pay: pd.DataFrame, prev: pd.DataFrame, agg_prev: pd.DataFrame) -> pd.DataFrame:
    """
    Identifies late payments and aggregates related data for previous applications.

    Parameters:
    - param pay: DataFrame containing payment history.
    - param prev: DataFrame containing previous application data.
    - param agg_prev: Aggregated DataFrame to merge late payment data.

    Returns:
    - agg_prev: Aggregated DataFrame with late payment data processed and added.
    """
    pay["LATE_PAYMENT"] = (pay["DAYS_ENTRY_PAYMENT"] - pay["DAYS_INSTALMENT"]).apply(lambda x: 1 if x > 0 else 0)
    dpd_id = pay[pay["LATE_PAYMENT"] > 0]["SK_ID_PREV"].unique()
    agg_prev = aggregate_and_join(
        prev[prev["SK_ID_PREV"].isin(dpd_id)],
        "PREV_LATE_",
        config.PREVIOUS_LATE_PAYMENTS_AGG,
        df_to_merge=agg_prev
    )
    return agg_prev


