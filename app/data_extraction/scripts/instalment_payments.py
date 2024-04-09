from functions import aggregate_and_join, read_validate
import pandas as pd
import config


def get_data(path: str, table: str) -> pd.DataFrame:
    """
    Processes instalment payments data to create aggregated features for each customer.

    - param path: Path to the directory containing the credit card balance CSV file.
    - table: Table name
    - return: Aggregated DataFrame with new features based on credit card usage patterns.
    """
    ip = read_validate(path, table, config.INSTALMENTS_PAYMENTS_DTYPES)
    ip = calculate_payment_diff_ratios(ip)
    ip = handle_dpd_dbd(ip)
    ip = late_payment_metrics(ip)
    pay_agg = aggregate_and_join(ip, "INS_", config.INSTALLMENTS_AGG)
    return pay_agg


def calculate_payment_diff_ratios(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates payment difference, payment ratio, overpayment amount, and overpayment flag for each installment.

    - df: DataFrame containing installment payment information.
    - return: DataFrame with calculated payment metrics added.
    """
    temp = (df[["SK_ID_PREV", "NUM_INSTALMENT_NUMBER", "AMT_PAYMENT"]]
            .groupby(["SK_ID_PREV", "NUM_INSTALMENT_NUMBER"])["AMT_PAYMENT"]
            .sum()
            .reset_index()
            .rename(columns={"AMT_PAYMENT": "AMT_PAYMENT_GROUPED"})
            )
    df = df.merge(temp, on=["SK_ID_PREV", "NUM_INSTALMENT_NUMBER"], how="left")

    df["PAYMENT_DIFFERENCE"] = df["AMT_INSTALMENT"] - df["AMT_PAYMENT_GROUPED"]
    df["PAYMENT_RATIO"] = df["AMT_INSTALMENT"] / df["AMT_PAYMENT_GROUPED"]
    df["PAID_OVER_AMOUNT"] = df["AMT_PAYMENT"] - df["AMT_INSTALMENT"]
    df["PAID_OVER"] = (df["PAID_OVER_AMOUNT"] > 0).astype(int)
    return df


def handle_dpd_dbd(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates Days Past Due (DPD) and Days Before Due (DBD) for each installment.

    - param df: DataFrame with installment payment and due dates.
    - return: DataFrame with DPD and DBD calculated.
    """
    df["DPD"] = (df["DAYS_ENTRY_PAYMENT"] - df["DAYS_INSTALMENT"]).clip(lower=0)
    df["DBD"] = (df["DAYS_INSTALMENT"] - df["DAYS_ENTRY_PAYMENT"]).clip(lower=0)

    return df


def late_payment_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates various metrics related to late payments.

    - param df: DataFrame with payment information.
    - return: DataFrame with late payment metrics added.
    """
    df["LATE_PAYMENT"] = (df["DBD"] > 0).astype(int)
    df["INSTALMENT_PAYMENT_RATIO"] = df["AMT_PAYMENT"] / df["AMT_INSTALMENT"]
    df["LATE_PAYMENT_RATIO"] = (df["INSTALMENT_PAYMENT_RATIO"] * df["LATE_PAYMENT"]).clip(lower=0)
    df["SIGNIFICANT_LATE_PAYMENT"] = (df["LATE_PAYMENT_RATIO"] > 0.05).astype(int)

    df["DPD_7"] = (df["DPD"] > 7).astype(int)
    df["DPD_15"] = (df["DPD"] > 15).astype(int)
    return df


