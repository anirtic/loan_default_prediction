from pathlib import Path
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MinMaxScaler
from typing import Callable, Any, List, Tuple, Union, Optional
from google.cloud import storage
from io import BytesIO
from functions import load_or_run_rfecv
from sklearn.pipeline import Pipeline
from data_extraction.scripts import (
    bureau,
    pos_cash,
    instalment_payments,
    credit_card_balance,
    previous_applications,
    application,
)
import pandas as pd
import numpy as np
import logging
import config
import re
import joblib
import gcsfs

logger = logging.getLogger(__name__)


class DataExtractor(BaseEstimator, TransformerMixin):

    def __init__(self, table_name: str, process_function: Callable[[Path], pd.DataFrame], df: pd.DataFrame=None):
        """
        Initializes the DataExtractor.

        Parameters:
        - table_name (str): Name of the table to be processed or extracted.
        - process_function (Callable[[Path], pd.DataFrame]): A function that processes data and returns a DataFrame.
          This function should accept a Path object pointing to the data directory and return a pandas DataFrame.
        """
        self.table_name: str = table_name
        self.process_function: Callable[[Path], pd.DataFrame] = process_function
        self.data_directory = config.DATA_DIRECTORY
        self.fs = gcsfs.GCSFileSystem()
        self.df: pd.DataFrame = df if df is not None else None

    def fit(self, X: Any, y: Any = None):
        return self

    def transform(self, X: Any):
        """
        Transforms the data by either loading a pickled DataFrame or processing raw data into a DataFrame.

        Parameters:
        - X (Any): The input data. Not used in this transformer, but included for compatibility with the
          scikit-learn transformer interface.

        Returns:
        - df (pd.DataFrame): The processed or loaded DataFrame.
        """
        if self.df is None:
            pickle_path = f"{config.PICKLED_DATA_DIRECTORY}{self.table_name}.pkl"

            if self.fs.exists(pickle_path):
                logger.info(f"Loading {self.table_name} data from pickle in GCS.")
                with self.fs.open(pickle_path, 'rb') as f:
                    df = pd.read_pickle(f)
            else:
                logger.info(f"Processing {self.table_name} data and saving to GCS.")
                try:
                    df = self.process_function(self.data_directory, self.table_name)
                    with self.fs.open(pickle_path, 'wb') as f:
                        df.to_pickle(f)
                except Exception as e:
                    logger.error(f"Error processing {self.table_name} data: {e}")
                    raise
            return df
        else:
            df = self.process_function(self.data_directory, self.table_name, self.df)
            return df


class SequentialDataPipeline(BaseEstimator, TransformerMixin):
    def __init__(self, steps: List[Tuple[str, TransformerMixin]]):
        """
        Initializes the SequentialDataPipeline with a series of data processing and merging steps.

        Parameters:
        - steps (List[Tuple[str, TransformerMixin]]): A list of tuples where each tuple contains a name (str) and
          an instance of a transformer (TransformerMixin) that implements a `transform` method. The transformers
          are expected to return DataFrames that will be sequentially merged.
        """
        self.steps = steps

    def fit(self, X: Any, y: Any = None):
        return self

    def transform(self, X: Any):
        """
        Transforms the data by applying each step's transformer in sequence and merging their outputs.

        Parameters:
        - X (Any): The input data. Not used in this pipeline, but included for compatibility with the
          scikit-learn transformer interface.

        Returns:
        - merged_data (pd.DataFrame): The merged DataFrame produced by sequentially transforming and merging the
          output of each step's transformer.
        """
        merged_data: pd.DataFrame = None
        for name, data_extractor in self.steps:
            data = data_extractor.transform(None)
            if merged_data is None:
                merged_data = data_extractor.transform(X)
            else:
                merge_key = "SK_ID_CURR"
                merged_data = merged_data.merge(data, on=merge_key, how="left")

        return merged_data


class ContractTypeSelector(BaseEstimator, TransformerMixin):
    def __init__(self, contract_type_value: int = 0) -> None:
        """
        Initializes the ContractTypeSelector.

        Parameters:
        - contract_type_value (int, optional): The value of the 'NAME_CONTRACT_TYPE' to filter for. Defaults to 0.
        """
        self.contract_type_value = contract_type_value

    def fit(self, X: pd.DataFrame, y: None = None):
        return self

    def transform(self, X: pd.DataFrame):
        """
        Filters the DataFrame to include only rows where 'NAME_CONTRACT_TYPE' matches the specified contract type value.

        Parameters:
        - X (pd.DataFrame): The DataFrame to filter.

        Returns:
        - pd.DataFrame: A DataFrame filtered to include only the specified contract type.

        Raises:
        - ValueError: If the 'NAME_CONTRACT_TYPE' column is missing from the input DataFrame.
        """
        if 'NAME_CONTRACT_TYPE' not in X.columns:
            raise ValueError("Column 'NAME_CONTRACT_TYPE' is missing in the input DataFrame.")

        return X[X['NAME_CONTRACT_TYPE'] == self.contract_type_value]


class ColumnDropper(BaseEstimator, TransformerMixin):
    def __init__(self, col_names: Union[str, List[str]]) -> None:
        """
        Initializes the ColumnDropper.

        Parameters:
        - col_names (Union[str, List[str]]): The name or list of names of the columns to drop.
        """
        self.col_names = [col_names] if isinstance(col_names, str) else col_names

    def fit(self, X: pd.DataFrame, y: None = None):
        return self

    def transform(self, X: pd.DataFrame):
        """
        Transforms the DataFrame by dropping specified columns.

        Parameters:
        - X (pd.DataFrame): The DataFrame to transform.

        Returns:
        - pd.DataFrame: The DataFrame with specified columns dropped.

        Raises:
        - ValueError: If any of the specified columns to drop do not exist in the DataFrame.
        """
        missing_cols = [col for col in self.col_names if col not in X.columns]
        if missing_cols:
            raise ValueError(f"The following columns are not in the DataFrame: {', '.join(missing_cols)}")

        return X.drop(columns=self.col_names)


class RemoveInf(BaseEstimator, TransformerMixin):
    def fit(self, X: Union[pd.DataFrame, np.ndarray], y: None = None):
        return self

    def transform(self, X: Union[pd.DataFrame, np.ndarray]):
        """
        Transforms the input data by replacing infinite values with NaN.

        Parameters:
        - X (Union[pd.DataFrame, np.ndarray]): The data to transform.

        Returns:
        - Union[pd.DataFrame, np.ndarray]: The transformed data with infinite values replaced by NaN.
        """

        if isinstance(X, pd.DataFrame):
            X_transformed = X.copy()
            numeric_cols = X_transformed.select_dtypes(include=[np.number]).columns
            inf_mask = np.isinf(X_transformed[numeric_cols])
            X_transformed[numeric_cols] = X_transformed[numeric_cols].where(~inf_mask, np.nan)
        elif isinstance(X, np.ndarray):
            if X.dtype.kind in 'buifc':
                X_transformed = np.where(np.isinf(X), np.nan, X)
            else:
                raise TypeError("np.ndarray contains non-numeric data, cannot apply 'isinf'.")
        else:
            raise TypeError("Input must be a pandas DataFrame or a numpy ndarray.")

        return X_transformed


class ScaleMinMax(BaseEstimator, TransformerMixin):
    def __init__(self, scaler: Optional[str]=None) -> None:
        """
        Initializes the ScaleMinMax transformer, setting up the MinMaxScaler and specifying categorical columns.

        Parameters:
        - categorical_cols (Optional[List[str]], optional): List of column names to be treated as categorical.
          These columns will not be scaled. Defaults to None, indicating no categorical columns.
        - scaler (Optional[str]=None): Name of scaler, that needs to be loaded from config.SCALERS_PATH
        """
        if scaler:
            self.scaler = self.load_scaler(scaler)
        else:
            self.scaler = MinMaxScaler()
        self.categorical_cols = config.CATEGORICAL_COLS
        self.non_categorical_cols = None

    def fit(self, X: pd.DataFrame, y: None = None):
        """
        Fits the MinMaxScaler to the non-categorical columns of the DataFrame.

        Parameters:
        - X (pd.DataFrame): The input DataFrame to fit.
        - y (None, optional): Ignored. Exists for compatibility with scikit-learn's interface.

        Returns:
        - self (ScaleMinMax): The fitted transformer instance.
        """
        self.non_categorical_cols = [col for col in X.columns if col not in self.categorical_cols]
        if self.non_categorical_cols:
            self.scaler.fit(X[self.non_categorical_cols])
        return self

    def transform(self, X: pd.DataFrame):
        """
        Transforms the DataFrame, scaling non-categorical columns and leaving categorical columns unchanged.

        Parameters:
        - X (pd.DataFrame): The input DataFrame to transform.

        Returns:
        - X_transformed (pd.DataFrame): The transformed DataFrame with scaled non-categorical columns.
        """
        X_transformed = X.copy()
        if self.non_categorical_cols:
            scaled_features = self.scaler.transform(X_transformed[self.non_categorical_cols])
            X_transformed[self.non_categorical_cols] = scaled_features
        return X_transformed

    def save(self, name: str):
        """
        Saves the scaler to a specified file.

        Parameters:
        - filepath (str): The path to the file where the scaler will be saved.
        """
        storage_client = storage.Client()
        bucket = storage_client.bucket(config.BUCKET_NAME)
        blob = bucket.blob(f"{config.SCALER_FOLDER}{name}.pkl")

        byte_stream = BytesIO()
        joblib.dump(self.scaler, byte_stream)
        byte_stream.seek(0)

        blob.upload_from_file(byte_stream, content_type='application/octet-stream')
        print(f"Scaler saved to GCP bucket: {bucket}{blob}")


    def load_scaler(self, scaler: str):
        """
        Loads a scaler from a specified file and sets it as the scaler to be used.

        Parameters:
        - scaler (str): Scaler name

        Returns:
        - scaler: The loaded MinMaxScaler object.
        """

        storage_client = storage.Client()
        bucket = storage_client.bucket(config.BUCKET_NAME)
        blob = bucket.blob(f"{config.SCALER_FOLDER}{scaler}.pkl")

        byte_stream = BytesIO()
        blob.download_to_file(byte_stream)
        byte_stream.seek(0)

        scaler = joblib.load(byte_stream)
        return scaler


class ColumnRenamer(BaseEstimator, TransformerMixin):
    def fit(self, X: pd.DataFrame, y: None = None):
        return self

    def transform(self, X: pd.DataFrame):
        """
        Transforms the DataFrame by renaming its columns. Non-alphanumeric characters in column names
        are replaced with underscores.

        Parameters:
        - X (pd.DataFrame): The DataFrame to transform.

        Returns:
        - pd.DataFrame: The DataFrame with renamed columns.

        Raises:
        - TypeError: If the input is not a pandas DataFrame.
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame")

        X_renamed = X.rename(columns=lambda x: re.sub(r'\W+', '_', x))
        return X_renamed


class CategoricalColsConverter(BaseEstimator, TransformerMixin):
    def __init__(self, columns=config.CATEGORICAL_COLS):
        """
        Initialize the transformer with the columns to be converted to categorical.

        Parameters:
        - columns (list of str): The list of column names to convert to categorical. If None, all string columns will be converted.
        """
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        """
        Convert the specified columns of X to categorical.

        Parameters:
        - X (pd.DataFrame): The input DataFrame.

        Returns:
        - X_transformed (pd.DataFrame): The DataFrame with specified columns converted to categorical.
        """
        X_transformed = X.copy()
        if self.columns is None:
            self.columns = [col for col in X_transformed.columns if X_transformed[col].dtype == 'object']
        for col in self.columns:
            X_transformed[col] = X_transformed[col].astype('category')
        return X_transformed


class FeatureSelection(BaseEstimator, TransformerMixin):
    """
    Initializes the FeatureSelection transformer which applies RFECV for feature selection.

    Parameters:
    - name (str): Identifier for loading or saving the RFECV model.

    Returns:
    - X (pd.DataFrame): Transformed dataframe.
    """
    def __init__(self, name):
        self.name = name
        self.rfecv = load_or_run_rfecv(name)

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if self.name == "rfecv_early_pymnt":

            selected_feature_indices = self.rfecv.support_[1:] # Not picking up a target column indice, it was False.
        else:
            selected_feature_indices = self.rfecv.support_
        selected_columns = X.loc[:, selected_feature_indices].columns.values

        X = X[selected_columns]
        return X


class ModelLoader(BaseEstimator, TransformerMixin):
    def __init__(self, name):
        """
        Initializes the model loader with the path to the model.

        Parameters:
        - model_path: The path or URI to the trained model file.
        """
        self.name = name
        self.model = self.load_model(name)

    def load_model(self, name):
        """
        Loads the model from the specified path.

        Parameters:
        - model_path: The path or URI to the trained model file.

        Returns:
        - The loaded model.
        """
        storage_client = storage.Client()
        bucket = storage_client.bucket(config.BUCKET_NAME)
        blob = bucket.blob(f"{config.MODELS_FOLDER}{name}.pkl")

        byte_stream = BytesIO()
        blob.download_to_file(byte_stream)
        byte_stream.seek(0)

        model = joblib.load(byte_stream)
        return model

    def fit(self, X, y=None):
        return self

    def predict(self, X):
        """
        Make predictions with the loaded model.

        Parameters:
        - X: The data to make predictions on.

        Returns:
        - Predictions made by the model.
        """
        return self.model.predict(X)


class Print(BaseEstimator, TransformerMixin):
    """
    Class used to print out data from parent Pipeline class.
    """
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        """
        Parameters:
        - X: The data to print.

        Returns:
        - X: Passes same data to next pipeline.
        """
        print(X)
        return X


def get_preprocessing_pipeline(
    application_data: str = "application_train.csv",
    application_df: pd.DataFrame=None,
    bureau_data: str = "bureau.csv",
    previous_application_data: str = "previous_application.csv",
    pos_cash_data: str = "pos_cash.csv",
    instalments_payments_data: str = "instalments_payments.csv",
    credit_card_balance_data: str = "credit_card_balance.csv",
    contract_type: int = 0,
    drop_columns: List = ["SK_ID_CURR", "NAME_CONTRACT_TYPE"],
):
    data_steps = [
        ("application_data", DataExtractor(application_data, application.get_data, application_df)),
        ("bureau_data", DataExtractor(bureau_data, bureau.get_data)),
        (
            "prev_application_data",
            DataExtractor(previous_application_data, previous_applications.get_data),
        ),
        ("pos_cash_data", DataExtractor(pos_cash_data, pos_cash.get_data)),
        (
            "instalments_payments_data",
            DataExtractor(instalments_payments_data, instalment_payments.get_data),
        ),
        (
            "credit_card_balance_data",
            DataExtractor(credit_card_balance_data, credit_card_balance.get_data),
        ),
    ]

    preprocessing_pipeline = Pipeline(
        steps=[
            ("data_preparation", SequentialDataPipeline(data_steps)),
            ("select_loan_type", ContractTypeSelector(contract_type)),
            ("drop_id_loan_type", ColumnDropper(drop_columns)),
            ("remove_inf", RemoveInf()),
            ("conver_cat_cols", CategoricalColsConverter()),
        ]
    )

    return preprocessing_pipeline

