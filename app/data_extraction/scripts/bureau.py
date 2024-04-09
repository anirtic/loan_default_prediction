from functions import encode_one_hot, aggregate_and_join, read_validate
import pandas as pd
import numpy as np
import config


def get_data(path: str, table: str) -> pd.DataFrame:
    """
    Processes bureau data by performing several operations like encoding, merging, calculating specific features,
    and aggregating over different conditions and time frames.

    Parameters:
    - path (str): Path to the directory containing the bureau data file.

    Returns:
    - pd.DataFrame: The aggregated bureau data ready for further analysis or modeling.
    """
    bureau = read_validate(path, table, config.BUREAU_DTYPES)
    bureau = calculate_ratios(bureau)

    bureau, _ = encode_one_hot(bureau, nan_as_category=False)
    bureau = bureau.merge(bureau_balance_data(path, "bureau_balance.csv"), how="left", on="SK_ID_BUREAU")
    bureau["STATUS_LATE"] = bureau[[f"STATUS_{i}" for i in range(1, 6)]].sum(axis=1)
    bureau = process_loan_duration(bureau)
    agg_bureau = aggregate_and_join(bureau, "BUREAU_", config.BUREAU_AGG)

    agg_bureau = aggregate_and_join(df_to_agg=bureau[bureau["CREDIT_ACTIVE_Active"] == 1],
                                    prefix="BUREAU_ACTIVE_",
                                    aggregations=config.BUREAU_ACTIVE_AGG,
                                    df_to_merge=agg_bureau)

    agg_bureau = aggregate_and_join(df_to_agg=bureau[bureau["CREDIT_ACTIVE_Closed"] == 1],
                                    prefix="BUREAU_CLOSED_",
                                    aggregations=config.BUREAU_CLOSED_AGG,
                                    df_to_merge=agg_bureau)

    agg_bureau = process_credit_types(bureau, agg_bureau)

    recent_loan_max_overdue = (
        bureau.groupby("SK_ID_CURR")["AMT_CREDIT_MAX_OVERDUE"]
        .last()
        .reset_index()
        .rename(columns={"AMT_CREDIT_MAX_OVERDUE": "BUREAU_RECENT_LOAN_MAX_OVERDUE"})
    )
    agg_bureau = agg_bureau.merge(recent_loan_max_overdue, on="SK_ID_CURR", how="left")

    agg_bureau["BUREAU_DEBT_OVER_CREDIT"] = agg_bureau["BUREAU_AMT_CREDIT_SUM_DEBT_SUM"] / agg_bureau[
        "BUREAU_AMT_CREDIT_SUM_SUM"]
    agg_bureau["BUREAU_ACTIVE_DEBT_OVER_CREDIT"] = agg_bureau["BUREAU_ACTIVE_AMT_CREDIT_SUM_DEBT_SUM"] / agg_bureau[
        "BUREAU_ACTIVE_AMT_CREDIT_SUM_SUM"]
    return agg_bureau


def calculate_ratios(bureau: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates specific financial ratios from the bureau data for further analysis.

    Parameters:
    - df (pd.DataFrame): The input bureau data.

    Returns:
    - pd.DataFrame: The DataFrame with new ratio features.
    """
    bureau["LAST_CREDIT_TERM"] = -bureau["DAYS_CREDIT"] + bureau["DAYS_CREDIT_ENDDATE"]
    bureau["USED_CREDIT_PERC"] = (bureau["AMT_CREDIT_SUM"] / bureau["AMT_CREDIT_SUM_DEBT"].replace(0, np.nan)).fillna(
        value=0)
    bureau["AMT_CREDIT_SUM"].fillna(0, inplace=True)
    bureau["AMT_CREDIT_SUM_DEBT"].fillna(0, inplace=True)
    bureau["AVAILABLE_CREDIT"] = bureau["AMT_CREDIT_SUM"] - bureau["AMT_CREDIT_SUM_DEBT"]
    return bureau


def process_loan_duration(bureau: pd.DataFrame) -> pd.DataFrame:
    """
    Enriches each loan in the bureau DataFrame with aggregated financial behavior metrics based on the loan's duration.

    Parameters:
    - bureau (pd.DataFrame): The input DataFrame containing bureau data with a 'MONTHS_BALANCE_SIZE' column
                             representing the loan duration.

    Returns:
    - pd.DataFrame: The enriched DataFrame with additional features representing the average financial behavior
                    for loans of similar duration.
    """
    selected_features = [
        "AMT_CREDIT_MAX_OVERDUE", "AMT_CREDIT_SUM_OVERDUE", "AMT_CREDIT_SUM",
        "AMT_CREDIT_SUM_DEBT", "USED_CREDIT_PERC", "AVAILABLE_CREDIT",
        "STATUS_0", "STATUS_LATE"
    ]

    loan_duration_agg = bureau.groupby("MONTHS_BALANCE_SIZE")[selected_features].mean().reset_index()
    loan_duration_agg.rename(columns={feature: f"AVG_DURATION_{feature}" for feature in selected_features},
                             inplace=True)
    bureau = bureau.merge(loan_duration_agg, how="left", on="MONTHS_BALANCE_SIZE")
    return bureau


def process_credit_types(bureau: pd.DataFrame, agg_bureau: pd.DataFrame) -> pd.DataFrame:
    """
    Processes and aggregates the bureau DataFrame based on different credit types and merges the results into
    the aggregated bureau DataFrame (agg_bureau).

    This function iterates through a predefined list of credit types, filters the bureau DataFrame for each
    credit type, aggregates the filtered data based on configurations specified in `config.BUREAU_LOAN_TYPE_AGG`,
    and then joins the aggregated data with `agg_bureau` based on the SK_ID_CURR column.

    Parameters:
    - bureau (pd.DataFrame): The input bureau DataFrame containing credit history data.
    - agg_bureau (pd.DataFrame): The aggregated bureau DataFrame to which the aggregated data for each credit
      type will be joined.

    Returns:
    - pd.DataFrame: The updated aggregated bureau DataFrame with additional features for each credit type.
    """
    credit_types = ["Consumer credit", "Credit card", "Mortgage", "Car loan", "Microloan"]
    for credit_type in credit_types:
        agg_bureau = aggregate_and_join(
            bureau[bureau[f"CREDIT_TYPE_{credit_type}"] == 1],
            f"BUREAU_{credit_type.split(' ')[0].upper()}_",
            config.BUREAU_LOAN_TYPE_AGG,
            aggregate_by="SK_ID_CURR",
            df_to_merge=agg_bureau
        )
    return agg_bureau


def bureau_balance_data(path: str, table: str) -> pd.DataFrame:
    """
    Processes the bureau balance data by performing one-hot encoding on categorical variables and aggregating
    numerical features. The aggregated features are then merged based on 'SK_ID_BUREAU'.

    Parameters:
    - path (str): The directory path where the bureau balance CSV file is located.

    Returns:
    - pd.DataFrame: The processed and aggregated bureau balance data.
    """
    bureau_balance = read_validate(path, table, config.BUREAU_BALANCE_DTYPES)
    bureau_balance, encoded_cols = encode_one_hot(bureau_balance)
    bb_processed = bureau_balance.groupby("SK_ID_BUREAU")[encoded_cols].mean().reset_index()

    agg = {"MONTHS_BALANCE": ["min", "max", "mean", "size"]}
    bb_processed = aggregate_and_join(bureau_balance,  '', agg, "SK_ID_BUREAU", bb_processed)
    return bb_processed
