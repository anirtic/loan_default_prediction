from functions import aggregate_and_join, encode_one_hot, read_validate
import pandas as pd
import numpy as np
import config


def get_data(path: str, table: str) -> pd.DataFrame:
    """
    Processes credit card balance data to create aggregated features for each customer.

    - path: Path to the directory containing the credit card balance CSV file.
    - table: Table name
    """
    cc = read_validate(path, table, config.CREDIT_CARD_BALANCE_DTYPES)
    cc, _ = encode_one_hot(cc, nan_as_category=False)
    cc = create_new_features(cc)
    cc_agg = cc.groupby("SK_ID_CURR").agg(config.CREDIT_CARD_AGG)
    cc_agg.columns = ["CC_" + "_".join(col).upper() for col in cc_agg.columns]
    cc_agg.reset_index(inplace=True)
    cc_agg = last_month_balance(cc, cc_agg)
    return cc_agg


def create_new_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates new features in the DataFrame based on existing credit card balance information.

    Parameters:
    - df (pd.DataFrame): The input DataFrame containing credit card balance data.

    New Features:
    - LIMIT_USE: Ratio of AMT_BALANCE to AMT_CREDIT_LIMIT_ACTUAL.
    - PAYMENT_DIV_MIN: Ratio of AMT_PAYMENT_CURRENT to AMT_INST_MIN_REGULARITY.
    - LATE_PAYMENT: Binary flag indicating whether there is a late payment (SK_DPD > 0).
    - DRAWING_LIMIT_RATIO: Ratio of AMT_DRAWINGS_ATM_CURRENT to AMT_CREDIT_LIMIT_ACTUAL.

    Returns:
    - pd.DataFrame: The DataFrame with new features added.
    """
    df["LIMIT_USE"] = np.where(df["AMT_CREDIT_LIMIT_ACTUAL"] == 0, 0,
                               df["AMT_BALANCE"] / df["AMT_CREDIT_LIMIT_ACTUAL"])
    df["PAYMENT_DIV_MIN"] = np.where(df["AMT_INST_MIN_REGULARITY"] == 0, 0,
                                     df["AMT_PAYMENT_CURRENT"] / df["AMT_INST_MIN_REGULARITY"])
    df["LATE_PAYMENT"] = (df["SK_DPD"] > 0).astype(int)
    df["DRAWING_LIMIT_RATIO"] = np.where(df["AMT_CREDIT_LIMIT_ACTUAL"] == 0, 0,
                                         df["AMT_DRAWINGS_ATM_CURRENT"] / df["AMT_CREDIT_LIMIT_ACTUAL"])

    return df


def last_month_balance(df: pd.DataFrame, df_agg: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregates the latest month's balance information for each credit by SK_ID_PREV and merges it with an aggregated DataFrame.

    Parameters:
    - df (pd.DataFrame): The input DataFrame containing credit card balance data.
    - df_agg (pd.DataFrame): The aggregated DataFrame to which the new aggregated data will be merged.

    Returns:
    - pd.DataFrame: The updated aggregated DataFrame with the last month's balance information.
    """
    last_ids = df.groupby("SK_ID_PREV")["MONTHS_BALANCE"].idxmax()
    last_months_df = df.loc[last_ids]
    df_agg = aggregate_and_join(last_months_df, "CC_LAST_", {"AMT_BALANCE": ["mean", "max"]}, df_to_merge=df_agg)
    return df_agg


