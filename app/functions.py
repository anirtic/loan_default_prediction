import pandas as pd
import numpy as np
import scipy.stats as stats
import itertools
import plotly.graph_objs as go
import pickle
import joblib
import os
from io import BytesIO
from google.cloud import storage
from scipy.stats import anderson, kstest, chi2_contingency
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.proportion import proportions_ztest
from typing import Tuple, List, Union, Any, Dict, Optional
from hyperopt import STATUS_OK, fmin, tpe, Trials
from sklearn.metrics import recall_score, f1_score, precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from numpy.random import RandomState
from hyperopt.early_stop import no_progress_loss
from sklearn.feature_selection import RFECV
from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedKFold

import config


def drop_rows_if(df: pd.DataFrame, perc: float) -> pd.DataFrame:
    """
    Drop rows from a DataFrame if the percentage of missing values in a row is greater than a threshold.

    Parameters:
        df (pd.DataFrame): The input DataFrame.
        perc (float): The threshold percentage for missing values in a row.

    Returns:
        pd.DataFrame: The DataFrame with rows dropped if they exceed the threshold.
    """
    df_na_counts = df.isna().sum(axis=1)
    df = df.drop(df_na_counts[df_na_counts > df.shape[1] * perc].index, axis=0)
    return df


def drop_columns_if(df: pd.DataFrame, perc: float) -> pd.DataFrame:
    """
    Drop columns from a DataFrame if the percentage of missing values in a column is greater than a threshold.

    Parameters:
        df (pd.DataFrame): The input DataFrame.
        perc (float): The threshold percentage for missing values in a column.

    Returns:
        pd.DataFrame: The DataFrame with columns dropped if they exceed the threshold.
    """
    df_na_counts = df.isna().sum(axis=0)
    df = df.drop(df_na_counts[df_na_counts > df.shape[0] * perc].index, axis=1)
    return df


def print_n_duplicates_missing(df: pd.DataFrame) -> None:
    """
    Print the number of duplicate rows and the total count of missing (null) values in a DataFrame.

    Parameters:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        None
    """
    duplicate_count = df.duplicated().sum()
    missing_count = df.isna().sum().sum()
    print(f"There are {duplicate_count} duplicates.\nNull values: {missing_count}")


def get_significance(p: float) -> str:
    """
    Returns a string indicating if H0 is rejected or not, comparing a given p-value to alpha (0.05).

    Args:
        p (float): The p-value from a statistical test.

    Returns:
        str: A string indicating whether to reject or fail to reject the null hypothesis.
    """
    if p <= 0.05:
        return "P value is below alpha 0.05 --> Reject H0."
    elif p > 0.05:
        return "P value is above alpha 0.05 --> Fail to reject H0"


def check_normality(data: np.ndarray) -> None:
    """
    Check the normality of a given dataset using the Anderson-Darling and Kolmogorov-Smirnov tests.

    Parameters:
        data (np.ndarray): The input dataset to be checked for normality.

    Returns:
        None
    """
    if data.size == 0 or np.isnan(data).all():
        print("Data is empty or contains only NaNs. Skipping normality checks.")
        return

    data_clean = data[~np.isnan(data)]

    if data_clean.size > 0:
        print(f"Skewness coef.: {stats.skew(data_clean):.2f}")
        anderson_test = stats.anderson(data_clean)
        print("Anderson-Darling Test:",
              "Not normally distributed" if anderson_test.statistic > anderson_test.critical_values[
                  2] else "Normally distributed")

        _, p = stats.kstest(data_clean, 'norm')
        print("Kolmogorov-Smirnov Test:", "Not normally distributed" if p < 0.05 else "Normally distributed", "\n")
    else:
        print("Data is empty after removing NaNs. Skipping normality checks.")


def marascuilo_procedure(contingency_table: pd.DataFrame, outcome_category: Union[int, str],
                         alpha: float = 0.05, adjust_method: str = 'bonferroni') -> pd.DataFrame:

    """
    Enhanced Marascuilo procedure with adjustable parameters for outcome category, significance level,
    and multiple comparison adjustments. It calculates pairwise differences in proportions, critical values,
    and adjusted p-values for each pair's comparison.

    Parameters:
    - contingency_table: pd.DataFrame, a contingency table with categories as rows.
    - outcome_category: int or str, the column name or index for the category of interest in the contingency table.
    - alpha: float, significance level for determining critical value.
    - adjust_method: str, method used for adjusting p-values for multiple comparisons ('bonferroni', 'holm', etc.).

    Returns:
    - DataFrame with comparison results, including categories compared, difference in proportions,
      critical value, raw p-values, adjusted p-values, and the proportions of each category.
    """
    total_observations = contingency_table.sum().sum()
    proportions = contingency_table[outcome_category] / contingency_table.sum(axis=1)

    critical_value = np.sqrt(stats.chi2.ppf(1 - alpha, df=len(contingency_table.index) - 1) / total_observations)

    comparisons = list(itertools.combinations(proportions.index, 2))
    p_values = []

    for category1, category2 in comparisons:
        prop1 = proportions.loc[category1]
        prop2 = proportions.loc[category2]
        diff = np.abs(prop1 - prop2)

        se = np.sqrt((prop1 * (1 - prop1) / contingency_table.sum(axis=1).loc[category1]) +
                     (prop2 * (1 - prop2) / contingency_table.sum(axis=1).loc[category2]))

        z_score = diff / se
        p_value = 2 * (1 - stats.norm.cdf(np.abs(z_score)))
        p_values.append(p_value)

    adjusted_p_values = multipletests(p_values, alpha=alpha, method=adjust_method)[1]

    summary_df = pd.DataFrame(comparisons, columns=['Category1', 'Category2'])
    summary_df['Difference'] = [np.abs(proportions.loc[cat1] - proportions.loc[cat2]) for cat1, cat2 in comparisons]
    summary_df['Proportion1'] = [proportions.loc[cat1] for cat1, cat2 in comparisons]
    summary_df['Proportion2'] = [proportions.loc[cat2] for cat1, cat2 in comparisons]
    summary_df['Critical Value'] = critical_value
    summary_df['P-Value'] = p_values
    summary_df['Adjusted P-Value'] = adjusted_p_values

    significant_diffs_df = summary_df[summary_df['Adjusted P-Value'] < alpha]

    return significant_diffs_df


def two_group_default_comparison(crosstab: pd.DataFrame, alpha: float = 0.05) -> None:
    """
    Compares default rates between two groups using a chi-square test and, if applicable, a proportions Z-test.
    Inputs:
    - crosstab: A pandas DataFrame containing the cross-tabulation of the two groups.
    - alpha: Significance level for statistical tests.

    Example usage:
    crosstab = pd.crosstab(df['Group'], df['Defaulted'])
    two_group_default_comparison(crosstab)
    """

    def perform_chi_square_test() -> float:
        """
        Performs the chi-square test on the given crosstab and prints the hypothesis and result.
        """
        print("Chi-square test for independence:")
        print(
            f"- Null Hypothesis (H0): No association between {crosstab.index.name} "
            "and the likelihood of defaulting on a loan. The proportions of default are equal across groups.")
        print(
            f"- Alternative Hypothesis (Ha): An association between {crosstab.index.name} "
            "and the likelihood of defaulting on a loan. The proportions of default differ between groups.")
        chi_stat, p, dof, expected = chi2_contingency(crosstab)

        print(
            f"\nChi-square Results: "
            f"\n- Statistic: {round(chi_stat, 2)} "
            f"\n- P-value: {round(p, 3)} "
            f"\n- Degrees of freedom: {dof}"
        )
        return p

    def calculate_proportions_and_ztest() -> None:
        """
        Calculates the proportions of default for each group and performs the proportions Z-test if applicable.
        """
        total_counts = crosstab.sum(axis=1)
        proportion_group1 = crosstab.iloc[0, 1] / total_counts[0]
        proportion_group2 = crosstab.iloc[1, 1] / total_counts[1]

        if not ((total_counts[0] * proportion_group1 > 10 and total_counts[0] * (1 - proportion_group1) > 10) and
                (total_counts[1] * proportion_group2 > 10 and total_counts[1] * (1 - proportion_group2) > 10)):
            print(
                "One or more groups do not meet the assumption criteria for the proportions Z-test "
                "(success and failure counts > 10)."
            )
            return

        print("\nProportions Analysis:")
        print(f"- Default rate for {crosstab.index.name} = {crosstab.index[0]}: {round(proportion_group1, 4)}")
        print(f"- Default rate for {crosstab.index.name} = {crosstab.index[1]}: {round(proportion_group2, 4)}")

        n_count = np.array([crosstab.iloc[1, 1], crosstab.iloc[0, 1]])
        n_obs = np.array([total_counts[1], total_counts[0]])

        stat, pval = proportions_ztest(n_count, n_obs)
        print("\nProportions Z-test:")
        print(f"- Z-statistic: {round(stat, 2)}\n- P-value: {round(pval, 3)}")
        if pval < 0.025:
            print("P-value is below 0.025 -> Reject Null Hypothesis.")
            more_likely_group = crosstab.index[0] if stat < 0 else crosstab.index[1]
            print(f"Group {more_likely_group} is more likely to default on a loan.")
        else:
            print("P-value is above 0.05 -> Fail to reject the Null Hypothesis.")
            print("The difference in proportions of default is not statistically significant between the groups.")

    p_value_chi_square = perform_chi_square_test()
    if p_value_chi_square <= alpha:
        print("P value is below alpha 0.05 --> Rejecting H0.")
        calculate_proportions_and_ztest()
    else:
        print("P value is above alpha 0.05 --> Fail to reject H0.")


def encode_one_hot(df: pd.DataFrame, columns_to_encode: List[str] = None,
                   nan_as_category: bool = True) -> Tuple[pd.DataFrame, List[str]]:
    """
    Applies one-hot encoding to specified categorical columns in a DataFrame.

    Parameters:
    - df (pd.DataFrame): The DataFrame containing the data to be encoded.
    - columns_to_encode (List[str], optional): A list of column names to be one-hot encoded. If None, all columns with
      object dtype will be automatically identified and encoded.
    - nan_as_category (bool): If True, NaN values will be treated as a separate category and included in the encoding.
      Defaults to True.

    Returns:
    - Tuple[pd.DataFrame, List[str]]: A tuple containing the DataFrame with one-hot encoded columns and a list of the
      newly created column names resulting from the encoding process.

    Raises:
    - ValueError: If any of the specified columns to encode does not exist in the DataFrame.
    """
    if columns_to_encode is not None:
        missing_cols = [col for col in columns_to_encode if col not in df.columns]
        if missing_cols:
            raise ValueError(f"The following columns to encode are not in the DataFrame: {', '.join(missing_cols)}")
    else:
        columns_to_encode = df.select_dtypes(include=['object']).columns.tolist()

    df_encoded = pd.get_dummies(df, columns=columns_to_encode, dummy_na=nan_as_category)
    encoded_columns = [col for col in df_encoded.columns if col not in df.columns or col in columns_to_encode]

    return df_encoded, encoded_columns


def encode_label(df: pd.DataFrame, categorical_columns: List[str] = None) -> Tuple[pd.DataFrame, List[str]]:
    """
    Encodes categorical columns in the DataFrame to numerical labels.

    Parameters:
    - df (pd.DataFrame): The DataFrame containing the data to be encoded.
    - categorical_columns (List[str], optional): A list of column names to be encoded. If None, all object dtype columns
      will be automatically identified and encoded.

    Returns:
    - Tuple[pd.DataFrame, List[str]]: A tuple containing the DataFrame with encoded categorical columns and the list
      of columns that were encoded.

    Raises:
    - ValueError: If any specified categorical column does not exist in the DataFrame.
    """
    if categorical_columns is not None:
        missing_cols = [col for col in categorical_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"The following categorical columns are not in the DataFrame: {', '.join(missing_cols)}")
    else:
        categorical_columns = df.select_dtypes(include='object').columns.tolist()

    df[categorical_columns] = df[categorical_columns].apply(lambda x: pd.factorize(x)[0])

    return df, categorical_columns


def aggregate_and_join(df_to_agg: pd.DataFrame, prefix: str, aggregations: dict,
                       aggregate_by: str = "SK_ID_CURR", df_to_merge: pd.DataFrame = None) -> pd.DataFrame:
    """
    Aggregates a DataFrame based on specified columns and aggregation functions,
    then optionally merges this aggregated data with another DataFrame.

    Parameters:
    - df_to_agg (pd.DataFrame): The DataFrame to aggregate.
    - prefix (str): A prefix to add to the new column names after aggregation.
    - aggregations (dict): A dictionary specifying the columns to aggregate and the functions to use for aggregation.
    - aggregate_by (str, optional): The column name to group by before aggregating. Defaults to "SK_ID_CURR".
    - df_to_merge (pd.DataFrame, optional): An optional DataFrame to merge with the aggregated data. Defaults to None.

    Returns:
    - pd.DataFrame: The aggregated DataFrame, optionally merged with `df_to_merge`.

    Raises:
    - ValueError: If `df_to_agg` does not contain the `aggregate_by` column.
    """
    if aggregate_by not in df_to_agg.columns:
        raise ValueError(f"The column to aggregate by ({aggregate_by}) does not exist in the DataFrame to aggregate.")

    agg_df = df_to_agg.groupby(aggregate_by).agg(aggregations)
    agg_df.columns = [f"{prefix}{col}_{func.upper()}" for col, func in agg_df.columns.to_flat_index()]
    if df_to_merge is not None:
        if aggregate_by not in df_to_merge.columns:
            raise ValueError(f"The column to aggregate by ({aggregate_by}) does not exist in the DataFrame to merge.")
        return df_to_merge.merge(agg_df.reset_index(), how="left", on=aggregate_by)
    else:
        return agg_df.reset_index()


def replace_365243_with_nan(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """
    Replaces the value 365243 with NaN in specified columns of a DataFrame.

    Parameters:
    - df (pd.DataFrame): The DataFrame in which the replacements are to be made.
    - columns (list): A list of column names (strings) where the replacement is to occur.

    Returns:
    - pd.DataFrame: The modified DataFrame with specified values replaced by NaN.
    """
    # Check for non-existent columns to avoid silent failure
    non_existent_cols = [col for col in columns if col not in df.columns]
    if non_existent_cols:
        raise ValueError(f"Columns not found in DataFrame: {', '.join(non_existent_cols)}")

    replace_values = {365243: np.nan}
    df[columns] = df[columns].replace(replace_values)
    return df


def read_validate(path: str, table: str, dtypes: dict) -> pd.DataFrame:
    """
    Reads a CSV/JSON file into a pandas DataFrame with specified data types, ensuring the file exists.

    Parameters:
    - path (str): The directory path where the CSV file is located.
    - table (str): The name of the table (file without the .csv extension).
    - dtypes (dict): A dictionary specifying the data types of the columns.

    Returns:
    - pd.DataFrame: The DataFrame created from the CSV file.

    Raises:
    - FileNotFoundError: If the CSV file does not exist at the specified path.
    """
    gcs_path = f"{path}{table}"
    _, file_extension = os.path.splitext(table)

    try:
        if file_extension == ".csv":
            df = pd.read_csv(gcs_path, dtype=dtypes)
        elif file_extension == ".json":
            df = pd.read_json(gcs_path, dtype=dtypes)
        else:
            raise ValueError(f"Unknown file extension: {file_extension}. Can only read .csv and .json")
    except Exception as e:
            raise FileNotFoundError(f"Failed to read {gcs_path}: {e}")

    return df


def train_test_sets(X: pd.DataFrame, label_col: str) -> tuple:
    """
    Split a DataFrame into training and testing sets for a binary classification problem.

    Parameters:
        X (pd.DataFrame): The input DataFrame containing features and labels.
        label_col: The column name of predictor column

    Returns:
        tuple: A tuple containing X_train, X_test, y_train, and y_test.
    """
    labels = X[label_col]
    predictors = X.drop(label_col, axis=1)
    X_train, X_test, y_train, y_test = train_test_split(
        predictors, labels, test_size=0.1, random_state=42
    )
    return X_train, X_test, y_train, y_test


def objective(
        params: Dict[str, Any],
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        scoring_type: str = "roc_auc",
        score_avg: str = "binary"
):
    """
    Calculates the loss based on the specified scoring type and average method.

    Parameters:
    - params (Dict[str, Any]): Hyperparameters for the model.
    - X_train (DataFrame): Training features.
    - y_train (Series): Training labels.
    - X_val (DataFrame): Validation features.
    - y_val (Series): Validation labels.
    - scoring_type (str): The metric to use for scoring the model performance ('f1'/'recall'/'weighted'/'rocauc').
    - score_avg (str): The method to average the scoring metric for multi-class/multi-label classification.

    Returns:
    - Dict[str, Any]: A dictionary containing the loss score and the status of the evaluation.
    """
    model_type = params["model"]["type"]
    model_params = params["model"]["params"]
    if model_params.get('n_estimators', 0) <= 0:
        print(f"Adjusting invalid n_estimators value: {model_params.get('n_estimators')} to 1")
        model_params['n_estimators'] = 1

    if model_type == "lightgbm":
        model = LGBMClassifier(**model_params, verbose=-1)
        model.fit(X_train, y_train, categorical_feature="auto")
    elif model_type == "xgboost":
        model = XGBClassifier(**model_params, enable_categorical=True)
        model.fit(X_train, y_train)



    if scoring_type == "f1":
        predictions = model.predict(X_val)
        f1 = f1_score(y_val, predictions, average=score_avg)
        score = 1 - f1
    elif scoring_type == "recall":
        predictions = model.predict(X_val)
        recall = recall_score(y_val, predictions, average=score_avg)
        score = 1 - recall
    elif scoring_type == "weighted":
        predictions = model.predict(X_val)
        precision = precision_score(y_val, predictions, average=score_avg)
        recall = recall_score(y_val, predictions, average=score_avg)
        score = 1 - ((recall*0.8) + (precision*0.2))
    elif scoring_type == "roc_auc":
        predictions_prob = model.predict_proba(X_val)
        predictions_prob = predictions_prob[:, 1]
        roc_auc = roc_auc_score(y_val, predictions_prob)
        score = 1 - roc_auc
    return {"loss": score, "status": STATUS_OK}


def run_hyperopt(
    name: str,
    objective_fn,
    search_space: Dict[str, Any],
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    X_val: pd.Series,
    y_val: pd.Series,
    max_evals: int = 120,
    seed: int = 42,
    n_early_stop: int = 10,
    scoring: str = "roc_auc",
    score_avg: str = "binary"
):
    """
    Executes hyperparameter optimization using Hyperopt and manages the storage of the trials and best parameters.

    Parameters:
    - name (str): The base name for saving/loading the trials and best parameters files.
    - objective_fn (Callable): The objective function to be minimized, which computes the metric to optimize.
    - search_space (Dict[str, Any]): The search space for hyperparameters.
    - X_train, y_train (DataFrame, Series): Training data and labels.
    - X_val, y_val (DataFrame, Series): Validation data and labels.
    - max_evals (int): The maximum number of evaluations for the optimizer.
    - seed (int): The random seed for reproducibility.
    - n_early_stop (int): The number of iterations with no improvement to stop the search early.
    - scoring (str): The scoring metric for the optimization. Default is "weighted".
    - score_avg (str): The averaging method for multi-class/multi-label classification.

    Returns:
    - Tuple[Dict[str, Any], Trials]: A tuple containing the dictionary of best parameters and the Trials object.
    """

    storage_client = storage.Client()
    bucket = storage_client.bucket(config.BUCKET_NAME)

    trials_blob = bucket.blob(f"{config.HYPERPARAMS_FOLDER}{name}_trials.pkl")
    fmin_blob = bucket.blob(f"{config.HYPERPARAMS_FOLDER}{name}_fmin.pkl")

    trials = Trials()
    best = None

    if trials_blob.exists() and fmin_blob.exists():
        print(f"Loading trials and best parameters from GCS.")
        trials_byte_stream = BytesIO()
        fmin_byte_stream = BytesIO()

        trials_blob.download_to_file(trials_byte_stream)
        fmin_blob.download_to_file(fmin_byte_stream)

        trials_byte_stream.seek(0)
        fmin_byte_stream.seek(0)

        trials = pickle.load(trials_byte_stream)
        best = pickle.load(fmin_byte_stream)
    else:
        print("Files not found in GCS. Running hyperparameter optimization...")
        best = fmin(
            fn=lambda params: objective_fn(params, X_train, y_train, X_val, y_val, scoring, score_avg),
            space=search_space,
            algo=tpe.suggest,
            max_evals=max_evals,
            trials=trials,
            rstate=np.random.default_rng(seed=seed),
            early_stop_fn=no_progress_loss(n_early_stop),
        )

        trials_byte_stream = BytesIO()
        fmin_byte_stream = BytesIO()

        pickle.dump(trials, trials_byte_stream)
        pickle.dump(best, fmin_byte_stream)

        trials_byte_stream.seek(0)
        fmin_byte_stream.seek(0)

        trials_blob.upload_from_file(trials_byte_stream, content_type='application/octet-stream')
        fmin_blob.upload_from_file(fmin_byte_stream, content_type='application/octet-stream')

    return best, trials


def unpack(x: Any) -> Union[Any, float]:
    """
    Returns the first element of an input if it exists, otherwise returns NaN.

    Parameters:
    - x (Any): Input which could be a list, tuple, or any other type that supports indexing.

    Returns:
    - Union[Any, float]: The first element of the input or NaN if the input is empty or does not support indexing.
    """
    if x:
        return x[0]
    return np.nan


def prepare_trials_dataframe(trials: List[Trials]) -> pd.DataFrame:
    """
    Prepares a DataFrame from a list of Trials objects. Extracts 'vals' from 'misc',
    applies unpack function to each series, and adds 'loss' and 'trial_number' columns.

    Parameters:
    - trials (List[Trials]): A list of Trials objects from hyperopt optimization.

    Returns:
    - DataFrame: A pandas DataFrame containing unpacked trial parameters, loss values, and trial numbers.
                      Additionally, maps model index to model name if 'model' column exists.
    """
    trials_df = pd.DataFrame([pd.Series(t["misc"]["vals"]).apply(unpack) for t in trials])
    trials_df["loss"] = [t["result"]["loss"] for t in trials]
    trials_df["trial_number"] = trials_df.index
    if 'model' in trials_df.columns:
        trials_df["model"] = np.where(trials_df["model"] == 0, "lightgbm", "xgboost")
    return trials_df


def add_hover_data(fig: go.Figure, df: pd.DataFrame, model_choice: str) -> go.Figure:
    """
    Adds hover data to a Plotly figure based on the model choice. It dynamically sets the hover template
    to display trial number, model, loss, and specific model parameters.

    Parameters:
    - fig (go.Figure): Plotly figure to which hover data will be added.
    - df (DataFrame): DataFrame containing model results and parameters.
    - model_choice (str): A string indicating the model choice, e.g., 'lightgbm' or 'xgboost'.

    Returns:
    - go.Figure: The updated Plotly figure with the new hover data.
    """
    params = {
        "lightgbm": ["lgbm_learning_rate", "lgbm_max_depth", "lgbm_n_estimators"],
        "xgboost": ["xgb_learning_rate", "xgb_max_depth", "xgb_n_estimators"],
    }
    filtered_df = df[df["model"] == model_choice]

    hover_template = (
        "Trial: %{x}<br>Model: " + model_choice + "<br>Loss: %{y:.3f}<extra></extra>"
    )
    for i, param in enumerate(params[model_choice]):
        hover_template += f"<br>{param}: %{{customdata[{i}]:.2f}}"

    fig.update_traces(
        customdata=filtered_df[params[model_choice]],
        hovertemplate=hover_template,
        selector={"name": model_choice},
    )
    return fig


def load_or_run_rfecv(name: str,
                      classifier: Optional = None,
                      X: Optional[pd.DataFrame] = None,
                      y: Optional[pd.Series] = None,
                      min_features_to_select: int = 100,
                      step: int = 3,
                      cv: int = 3,
                      scoring: str = "roc_auc") -> Optional[RFECV]:
    """
    Load RFECV results from a GCS bucket if exists; otherwise, calculate and save it.

    Parameters:
    - name: The name used for the pickle file.
    - classifier: A machine learning classifier supporting the `fit` method.
    - X (pd.DataFrame): Feature matrix.
    - y (pd.Series): Target variable.
    - min_features_to_select (int): Minimum number of features to select.
    - step (int): Number of features to remove at each iteration.
    - cv (int): Number of folds for cross-validation.
    - scoring (str): Metric for evaluating the performance of the model.

    Returns:
    - RFECV instance or None if parameters are missing.
    """
    storage_client = storage.Client()
    bucket = storage_client.bucket(config.BUCKET_NAME)
    blob = bucket.blob(f"{config.RFECV_FOLDER}{name}.pkl")

    if blob.exists():
        # print(f"Loading RFECV results from GCS: {config.RFECV_FOLDER}{name}.pkl")
        byte_stream = BytesIO()
        blob.download_to_file(byte_stream)
        byte_stream.seek(0)
        rfecv = joblib.load(byte_stream)
        return rfecv
    elif classifier and X is not None and y is not None:
        print(f"RFECV results not found in GCS. Running RFECV...")
        rfecv = RFECV(
            estimator=classifier,
            step=step,
            cv=StratifiedKFold(n_splits=cv),
            scoring=scoring,
            min_features_to_select=min_features_to_select,
            n_jobs=-1
        )
        rfecv.fit(X, y)

        # Save the RFECV results to GCS
        byte_stream = BytesIO()
        joblib.dump(rfecv, byte_stream)
        byte_stream.seek(0)
        blob.upload_from_file(byte_stream, content_type='application/octet-stream')
        print(f"RFECV results saved to GCS: {config.RFECV_FOLDER}{name}")
        print("Optimal number of features:", rfecv.n_features_)
        return rfecv
    else:
        print("Missing required parameters to run RFECV.")
        return None
