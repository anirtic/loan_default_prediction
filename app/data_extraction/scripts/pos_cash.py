from functions import encode_one_hot, aggregate_and_join, read_validate
from pathlib import Path
import pandas as pd
import config


def get_data(path: str, table: str) -> pd.DataFrame:
    """
    Reads POS CASH balance data, encodes categorical variables, computes various features,
    and aggregates data on both POS CASH level and customer level.

    Paramaeters:
    - param path: Path to the data file directory.
    - table: Table name

    Returns:
    - pc_agg: DataFrame with aggregated POS CASH balance data and computed features.
    """
    pc = read_validate(path, table, config.POS_CASH_DTYPES)
    pc, categorical_cols = encode_one_hot(pc, nan_as_category=False)
    pc["FLAG_LATE_PYMNT"] = (pc["SK_DPD"] > 0).astype(int)

    categorical_agg = {cat: ["mean"] for cat in categorical_cols}
    pc_agg = aggregate_and_join(pc, "POS_", {**config.POS_CASH_AGG, **categorical_agg})

    pc_grouped = compute_features(pc)
    pc_agg = pd.merge(pc_agg, pc_grouped, on="SK_ID_CURR", how="left")

    mean_late_pymnts = aggregate_late_pymnts(pc)
    pc_agg = pd.merge(pc_agg, mean_late_pymnts, on="SK_ID_CURR", how="left")

    pc_agg = drop_features(pc_agg)

    return pc_agg


def compute_features(pc: pd.DataFrame) -> pd.DataFrame:
    """
    Computes additional features based on the POS CASH balance data.

    Parameters:
    - param pc: DataFrame containing the POS CASH balance data.

    Returns:
    - df_gp: DataFrame with computed features aggregated by previous loan ID and current customer ID.
    """
    sort_pc = pc.sort_values(by=["SK_ID_PREV", "MONTHS_BALANCE"])
    gp = sort_pc.groupby("SK_ID_PREV")

    df = pd.DataFrame({
        "SK_ID_CURR": gp["SK_ID_CURR"].first(),
        "MONTHS_BALANCE_MAX": gp["MONTHS_BALANCE"].max(),
        "POS_LOAN_COMPLETED_TERM": gp["NAME_CONTRACT_STATUS_Completed"].mean(),
        "POS_COMPLETED_BEFORE_TERM": gp["CNT_INSTALMENT"].first() - gp["CNT_INSTALMENT"].last()
    })

    df["POS_COMPLETED_BEFORE_TERM"] = (
        (df["POS_COMPLETED_BEFORE_TERM"] > 0) & (df["POS_LOAN_COMPLETED_TERM"] > 0)
    ).astype(int)

    df["POS_REMAINING_INSTALLMENTS"] = gp["CNT_INSTALMENT_FUTURE"].last()
    df["POS_REMAINING_INSTALLMENTS_RATIO"] = (
        gp["CNT_INSTALMENT_FUTURE"].last() / gp["CNT_INSTALMENT"].last()
    )

    df_gp = df.groupby("SK_ID_CURR").sum().drop(columns=["MONTHS_BALANCE_MAX"]).reset_index()
    return df_gp


def aggregate_late_pymnts(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregates the sum of late payments per previous loan and computes the mean of late payment flags for
    the last three applications per customer.

    Parameters:
    - param df: DataFrame containing the POS CASH balance data with late payment flags.

    Returns:
    - df_gp: DataFrame with mean late payment flags for the last three applications per customer.
    """
    temp = df[["SK_ID_PREV", "FLAG_LATE_PYMNT"]].groupby("SK_ID_PREV")["FLAG_LATE_PYMNT"].sum().reset_index().rename(
        columns={"FLAG_LATE_PYMNT": "FLAG_LATE_PYMNT_SUM"}
    )

    df = df.merge(temp, on=["SK_ID_PREV"], how="left")
    last_month_idx = df.groupby("SK_ID_PREV")["MONTHS_BALANCE"].idxmax()

    recent_apps = df.loc[last_month_idx].sort_values(by=['SK_ID_PREV', 'MONTHS_BALANCE'])
    df_gp = recent_apps.groupby("SK_ID_CURR").tail(3)

    df_gp = df_gp.groupby("SK_ID_CURR")["FLAG_LATE_PYMNT_SUM"].mean().reset_index()
    return df_gp


def drop_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drops specific features from the DataFrame.

    Parameters:
    - param df: DataFrame from which to drop features.

    Returns:
    - df: DataFrame with specified features dropped.
    """
    feature_names = [
        "POS_NAME_CONTRACT_STATUS_Canceled_MEAN",
        "POS_NAME_CONTRACT_STATUS_Amortized debt_MEAN",
        "POS_NAME_CONTRACT_STATUS_XNA_MEAN"
    ]
    df.drop(columns=feature_names, inplace=True)
    return df
