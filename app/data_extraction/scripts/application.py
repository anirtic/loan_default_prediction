from functions import encode_label, replace_365243_with_nan, read_validate
import pandas as pd
import config
import numpy as np


def get_data(path: str, table: str, df: pd.DataFrame=None) -> pd.DataFrame:
    """
    Constructs a DataFrame by reading and processing loan application data from a specified path,
    or processes an existing DataFrame if provided.

    Parameters:
    - path (str): The path to the directory containing the CSV file. Ignored if df is not None.
    - table (str): The name of the table. Used for logging or other purposes. Ignored if df is not None.
    - df (pd.DataFrame, optional): An existing DataFrame to process. If None, data will be read from the specified path.

    Returns:
    - pd.DataFrame: The processed DataFrame.
    """

    if df is None:
        df = read_validate(path, table, config.APPLICATIONS_DTYPES)

    ap = (
        df.pipe(clean_data)
        .pipe(sum_docs_and_exsources)
        .pipe(categorize_age)
        .pipe(financial_ratios)
        .pipe(income_ratios)
        .pipe(time_ratios)
        .pipe(flag_missing_house_data)
    )
    ap, _ = encode_label(ap)
    return ap


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans loan application data by removing specific gender entries, replacing certain values with NaN,
    and treating days with no phone change as missing data.

    Parameters:
    - df (pd.DataFrame): The input DataFrame containing loan application data.

    Returns:
    - pd.DataFrame: The cleaned DataFrame.
    """
    df = df.copy()
    df = df[df["CODE_GENDER"] != "XNA"]
    df = replace_365243_with_nan(df, ["DAYS_EMPLOYED"])
    df["DAYS_LAST_PHONE_CHANGE"].replace(0, np.nan, inplace=True)
    return df


def sum_docs_and_exsources(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds features to the DataFrame by summing the document flags and external source values.

    Parameters:
    - df (pd.DataFrame): The input DataFrame containing loan application data.

    Returns:
    - pd.DataFrame: The DataFrame with added features for document count and external source sum.
    """
    doc_cols = df.columns[df.columns.str.contains("FLAG_DOC")]
    df["DOCUMENT_COUNT"] = df[doc_cols].sum(axis=1)
    ex_source_cols = df.columns[df.columns.str.contains("EXT_SOURCE")]
    df["EXT_SOURCE_SUM"] = df[ex_source_cols].sum(axis=1)
    return df


def categorize_age(df: pd.DataFrame) -> pd.DataFrame:
    """
    Categorizes applicants by age into bins.

    Parameters:
    - df (pd.DataFrame): The input DataFrame containing loan application data.

    Returns:
    - pd.DataFrame: The DataFrame with an added age category feature.
    """
    age_bins = np.array([25, 40, 60])
    df["AGE_YEARS"] = df["DAYS_BIRTH"] / -365
    df["AGE_RANGE"] = np.digitize(df["AGE_YEARS"], age_bins)
    return df


def financial_ratios(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates various financial ratios from the loan application data.

    Parameters:
    - df (pd.DataFrame): The input DataFrame.

    Returns:
    - pd.DataFrame: The DataFrame with added financial ratio columns.
    """
    df["CREDIT_TO_ANNUITY_RATIO"] = np.where(df["AMT_ANNUITY"] != 0, df["AMT_CREDIT"] / df["AMT_ANNUITY"], 0)
    df["CREDIT_TO_GOODS_RATIO"] = np.where(df["AMT_GOODS_PRICE"] != 0, df["AMT_CREDIT"] / df["AMT_GOODS_PRICE"], 0)
    return df


def income_ratios(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates income-related ratios from the loan application data.

    Parameters:
    - df (pd.DataFrame): The input DataFrame.

    Returns:
    - pd.DataFrame: The DataFrame with added income ratio columns.
    """
    df["ANNUITY_TO_INCOME_RATIO"] = np.where(df["AMT_INCOME_TOTAL"] != 0, df["AMT_ANNUITY"] / df["AMT_INCOME_TOTAL"], 0)
    df["CREDIT_TO_INCOME_RATIO"] = np.where(df["AMT_INCOME_TOTAL"] != 0, df["AMT_CREDIT"] / df["AMT_INCOME_TOTAL"], 0)
    df["INCOME_TO_EMPLOYED_RATIO"] = np.where(df["DAYS_EMPLOYED"] != 0,
                                              df["AMT_INCOME_TOTAL"] / (df["DAYS_EMPLOYED"] / 365.25), 0)
    df["INCOME_TO_BIRTH_RATIO"] = df["AMT_INCOME_TOTAL"] / (df["DAYS_BIRTH"] / 365.25)

    return df


def time_ratios(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates time-related ratios from the loan application data.

    Parameters:
    - df (pd.DataFrame): The input DataFrame.

    Returns:
    - pd.DataFrame: The DataFrame with added time ratio columns.
    """
    df["EMPLOYED_TO_BIRTH_RATIO"] = (df["DAYS_EMPLOYED"] / 365.25) / (df["DAYS_BIRTH"] / 365.25)
    df["ID_TO_BIRTH_RATIO"] = df["DAYS_ID_PUBLISH"] / (df["DAYS_BIRTH"] / 365.25)
    df["CAR_TO_BIRTH_RATIO"] = np.where(df["DAYS_BIRTH"] != 0, (df["OWN_CAR_AGE"] * 365.25) / df["DAYS_BIRTH"], 0)
    df["CAR_TO_EMPLOYED_RATIO"] = np.where(df["DAYS_EMPLOYED"] != 0,
                                           (df["OWN_CAR_AGE"] * 365.25) / df["DAYS_EMPLOYED"], 0)

    return df


def flag_missing_house_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Flags rows with missing house-related data in the loan application dataset.

    Parameters:
    - df (pd.DataFrame): The input DataFrame.

    Returns:
    - pd.DataFrame: The DataFrame with a new column indicating missing house data.
    """
    house_features = [
        "APARTMENTS_AVG", "BASEMENTAREA_AVG", "YEARS_BEGINEXPLUATATION_AVG", "YEARS_BUILD_AVG",
        "COMMONAREA_AVG", "ELEVATORS_AVG", "ENTRANCES_AVG", "FLOORSMAX_AVG", "FLOORSMIN_AVG",
        "LANDAREA_AVG", "LIVINGAPARTMENTS_AVG", "LIVINGAREA_AVG", "NONLIVINGAPARTMENTS_AVG",
        "NONLIVINGAREA_AVG", "APARTMENTS_MODE", "BASEMENTAREA_MODE", "YEARS_BEGINEXPLUATATION_MODE",
        "YEARS_BUILD_MODE", "COMMONAREA_MODE", "ELEVATORS_MODE", "ENTRANCES_MODE", "FLOORSMAX_MODE",
        "FLOORSMIN_MODE", "LANDAREA_MODE", "LIVINGAPARTMENTS_MODE", "LIVINGAREA_MODE",
        "NONLIVINGAPARTMENTS_MODE", "NONLIVINGAREA_MODE", "APARTMENTS_MEDI", "BASEMENTAREA_MEDI",
        "YEARS_BEGINEXPLUATATION_MEDI", "YEARS_BUILD_MEDI", "COMMONAREA_MEDI", "ELEVATORS_MEDI",
        "ENTRANCES_MEDI", "FLOORSMAX_MEDI", "FLOORSMIN_MEDI", "LANDAREA_MEDI", "LIVINGAPARTMENTS_MEDI",
        "LIVINGAREA_MEDI", "NONLIVINGAPARTMENTS_MEDI", "NONLIVINGAREA_MEDI", "FONDKAPREMONT_MODE",
        "HOUSETYPE_MODE", "TOTALAREA_MODE", "WALLSMATERIAL_MODE"
    ]

    missing_proportion = df[house_features].isnull().mean(axis=1)

    df["HOUSE_DATA_MISSING_FLAG"] = (missing_proportion >= 0.7).astype(int)
    return df
