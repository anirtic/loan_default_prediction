import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import plotly.graph_objs as go
from matplotlib.patches import Patch
from typing import Union, Callable, Optional
from numpy import ndarray
from lightgbm import LGBMClassifier
from IPython.display import Image, display
from plotly.io import to_image
from sklearn.metrics import (
    precision_recall_curve,
    PrecisionRecallDisplay,
    ConfusionMatrixDisplay
)


def plot_displot(df_column: pd.Series) -> None:
    """
    Create a distribution plot (displot) for a DataFrame column.

    Parameters:
        df_column (pd.Series): The column from a DataFrame for which to create the distribution plot.

    Returns:
        None
    """
    plt.figure()
    sns.displot(df_column, kde=True, height=5, aspect=1.7)
    plt.ylabel("Count")
    plt.title(f"{df_column.name} distribution")
    plt.text(
        df_column.min(),
        plt.gca().get_ylim()[1] - 5,
        "Skew: {:.2f}".format(df_column.skew()),
        fontsize=12,
        horizontalalignment="left",
        verticalalignment="top",
    )
    custom_xticks = np.linspace(df_column.min(), df_column.max(), num=10)
    custom_xticks = np.round(custom_xticks).astype(int)
    plt.xticks(ticks=custom_xticks, labels=custom_xticks)

    plt.show()


def plot_categorical_distribution(df: pd.DataFrame, categorical_col: str, target_col: str) -> None:
    """
    Plots the distribution of a categorical column in relation to a target column.

    Parameters:
    - df (pd.DataFrame): The dataframe containing the data.
    - categorical_col (str): The name of the categorical column.
    - target_col (str): The name of the target column.

    Returns:
    - None
    """
    if not isinstance(df, pd.DataFrame):
        raise ValueError("df must be a pandas DataFrame")

    if categorical_col not in df.columns or target_col not in df.columns:
        raise ValueError("categorical_col or target_col not found in DataFrame columns")

    unique_categories = df[categorical_col].unique()
    palette = sns.color_palette("pastel", len(unique_categories))
    color_map = {category: color for category, color in zip(sorted(unique_categories), palette)}

    fig, axes = plt.subplots(1, 2, figsize=(15, 7))
    categorical_to_proportion_of_target(axes[0], df, categorical_col, target_col, color_map)
    plot_pie(axes[1], df[categorical_col], color_map)
    plt.tight_layout()
    plt.show()


def categorical_to_proportion_of_target(ax: plt.Axes, df: pd.DataFrame, categorical: str, target: str, color_map: dict) -> None:
    """
    Plots the proportion of the target column's value within each category of a categorical column.

    Parameters:
    - ax (plt.Axes): Matplotlib Axes object where the plot is drawn.
    - df (pd.DataFrame): The dataframe containing the data.
    - categorical (str): The name of the categorical column.
    - target (str): The name of the target column.
    - color_map (dict): A dictionary mapping categories to colors.

    Returns:
    - None
    """
    if not isinstance(df, pd.DataFrame):
        raise ValueError("df must be a pandas DataFrame")

    if categorical not in df.columns or target not in df.columns:
        raise ValueError("categorical or target not found in DataFrame columns")

    grouped = df.groupby([categorical, target]).size().unstack()
    percentage = (grouped[1] / (grouped[0] + grouped[1])) * 100
    percentage.sort_values(inplace=True)

    colors = [color_map[cat] for cat in percentage.index]

    sns.barplot(x=percentage.index, y=percentage.values, ax=ax, palette=colors)
    ax.set_title(f"Occurence of {target}=1 proportionally")
    ax.set_ylabel(f"{categorical} vs. % of {target}=1")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right')

    for p in ax.patches:
        ax.annotate(format(p.get_height(), '.1f') + '%',
                    (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center',
                    xytext=(0, 3),
                    textcoords='offset points')


def plot_pie(ax, df_column, color_map):
    """
    Plot an  pie chart using Matplotlib.

    Parameters:
        ax (plt.Axes): Matplotlib Axes object where the pie chart will be displayed.
        df_column (pd.Series): The pandas Series containing the data to be plotted as a pie chart.
        color_map:
    """
    cat_values = df_column.value_counts()
    colors = [color_map.get(cat, '#999999') for cat in cat_values.index]  # Default to a grey color if not found

    threshold = 0.05
    small_slices = cat_values[cat_values / cat_values.sum() < threshold]
    big_slices = cat_values[cat_values / cat_values.sum() >= threshold]
    combined_slices = big_slices.append(pd.Series(small_slices.sum(), index=['Other']))

    explode = [0.1 if cat in small_slices else 0 for cat in combined_slices.index]

    wedges, texts, autotexts = ax.pie(
        combined_slices.values,
        labels=combined_slices.index,
        colors=colors[:len(combined_slices)],
        autopct=make_autopct(combined_slices.values),
        startangle=140,
        explode=explode
    )

    for text, autotext in zip(texts, autotexts):
        if autotext.get_text() == '':
            text.set_color('grey')
            text.set_fontsize(8)

    ax.axis('equal')
    ax.set_title(f"{df_column.name} value distribution")

    plt.legend(wedges, combined_slices.index, title=df_column.name, loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))


def make_autopct(values: list) -> Callable[[float], str]:
    """
    Generates a function to be used as autopct in pie charts, displaying both
    percentage and absolute value. Formats large numbers using K for thousands,
    and M for millions.

    Parameters:
    - values (list): A list of the values used in the pie chart for calculating absolute values from percentages.

    Returns:
    - Callable[[float], str]: A function that takes a percentage (float) and returns a string representation
      of the percentage and its corresponding absolute value.
    """
    def my_autopct(pct: float) -> str:
        total = sum(values)
        val = int(round(pct * total / 100.0))
        if val >= 1000000:
            val_str = '{:.1f}M'.format(val / 1000000)
        elif val >= 1000:
            val_str = '{:.1f}K'.format(val / 1000)
        else:
            val_str = str(val)
        return '{p:.2f}%\n({v})'.format(p=pct, v=val_str) if pct > 5 else ''

    return my_autopct


def plot_num_vs_binary(df: pd.DataFrame, num_col: str, compared_col: str) -> None:
    """
    Plots the distribution of a numerical column compared across a binary categorical column
    in a DataFrame using a density plot.

    Parameters:
    - df (pd.DataFrame): The dataframe containing the data.
    - num_col (str): The name of the numerical column to be plotted.
    - compared_col (str): The name of the binary categorical column to compare against.

    Returns:
    - None
    """
    if not isinstance(df, pd.DataFrame):
        raise ValueError("df must be a pandas DataFrame")

    if num_col not in df.columns or compared_col not in df.columns:
        raise ValueError(f"One or both specified columns ({num_col}, {compared_col}) are not in the DataFrame")

    plt.figure()
    sns.displot(
        data=df,
        x=num_col,
        hue=compared_col,
        kind="kde",
        height=4.5,
        aspect=1.7,
        palette=["#3366FF", "#FF5733"],
    )
    plt.title(f"{num_col} vs {compared_col}")
    plt.show()


def get_cols_with_missing_values(df: pd.DataFrame, threshold_prc: float) -> None:
    """
    Plots a bar chart showing columns with missing values exceeding a specified threshold.

    Parameters:
        df (pd.DataFrame): The DataFrame to analyze.
        threshold_prc (float): The percentage threshold for missing values (e.g., 10.0 for 10%).

    Returns:
        None
    """
    nan_col = (df.isnull().sum() / len(df)) * 100
    nan_col_abv_threshold = nan_col[nan_col > threshold_prc].sort_values()
    if not nan_col_abv_threshold.empty:
        plt.figure(figsize=(20, 4))
        nan_col_abv_threshold.plot(kind="bar")

        plt.title(
            f"List of Columns & NA percentages where NA values are more than {threshold_prc}%"
        )
        plt.xlabel("Features")
        plt.ylabel("Percentage of Missing Values")
        plt.show()
    elif df.isna().any().any():
        total = df.isnull().sum().sort_values(ascending=False)
        nan_col = nan_col.sort_values(ascending=False)
        missing_below_threshold = pd.concat([total, round(nan_col, 2)], axis=1, keys=['Total', 'Percent'])
        print(f"There are no features with more than {threshold_prc}% missing values.")
        print("Missing values below threshold:")
        print(missing_below_threshold[missing_below_threshold["Total"] > 0])
    else:
        print("There are no missing values.")


def plot_dunn_results(data: Union[ndarray, pd.DataFrame], alpha: float) -> None:
    """
    Plot the results of Dunn's test as a heatmap.

    Parameters:
        data (Union[ndarray, pd.DataFrame]): A 2D array or DataFrame containing p-values from Dunn's test.
        alpha (float): The significance level for Dunn's test.

    Returns:
        None
    """
    up_triang = np.triu(np.ones_like(data, dtype=bool))

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        data,
        annot=True,
        mask=(data > alpha) | up_triang,
        cmap=sns.color_palette(["lightgreen"]),
        cbar=False,
        linewidths=0.5,
    )

    sns.heatmap(
        data,
        annot=False,
        cmap=sns.color_palette(["white"]),
        cbar=False,
        mask=~up_triang,
    )
    legend_elements = [
        Patch(
            facecolor="lightgreen",
            edgecolor="black",
            label="Statistically Significant Difference",
        )
    ]
    plt.legend(handles=legend_elements, loc="upper right")
    plt.title("Dunn's test results (p-values)")
    plt.show()


def plot_precision_recall_and_confusion_matrix(
        model: LGBMClassifier, X: pd.DataFrame, y: pd.Series
) -> None:
    """
    Plot the Precision-Recall Curve and Confusion Matrix for a given classification model.

    Parameters:
        model (XGBClassifier): The classification model.
        X (pd.DataFrame): The input features for evaluation.
        y (pd.Series): The true target labels.

    Returns:
        None
    """
    precision, recall, thresholds = precision_recall_curve(
        y, model.predict_proba(X)[:, 1]
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    pr_display = PrecisionRecallDisplay(precision=precision, recall=recall).plot(
        ax=axes[0]
    )
    axes[0].set_title("Precision-Recall Curve")

    cm = ConfusionMatrixDisplay.from_estimator(
        model, X, y, cmap=plt.cm.Blues, normalize="true", ax=axes[1]
    )
    cm.ax_.set_title("Confusion Matrix")


def show_fig(fig, render="interactive"):
    """
      Display a Plotly figure as a static image or as an interactive figure within a Jupyter Notebook.

      Parameters:
      - fig (go.Figure): The Plotly figure to display.
      - render (str, optional): The rendering type. Use 'image' for a static image or 'interactive' for an interactive figure. Defaults to 'image'.

      Returns:
      - None
      """
    if render == "image":
        image_bytes = to_image(fig, format="png")
        display(Image(image_bytes))
    elif render == "interactive":
        display(fig)


def create_contour_plot(
    trials_df: pd.DataFrame,
    x_param: str,
    y_param: str,
    model_type: Optional[str] = None,
    x_title: str = "X-axis",
    y_title: str = "Y-axis",
    plot_title: str = "Contour Plot",
):
    """
    Create a contour plot of hyperparameter optimization trials.

    Parameters:
    - trials_df (pd.DataFrame): A DataFrame containing the trials data.
    - x_param (str): The parameter to display on the x-axis.
    - y_param (str): The parameter to display on the y-axis.
    - model_type (Optional[str]): The model type to filter on. If None, use all data.
    - x_title (str): Title for the x-axis.
    - y_title (str): Title for the y-axis.
    - plot_title (str): Title for the plot.

    Returns:
    - A Plotly Figure object.
    """
    if model_type:
        filter_mask = trials_df["model"] == model_type
        filtered_df = trials_df.loc[filter_mask]
    else:
        filtered_df = trials_df

    fig = go.Figure(
        data=go.Contour(
            z=filtered_df["loss"],
            x=filtered_df[x_param],
            y=filtered_df[y_param],
            contours=dict(
                showlabels=True,
                labelfont=dict(size=12, color="white"),
            ),
            colorbar=dict(title="Loss", titleside="right"),
            hovertemplate=f"Loss: %{{z}}<br>{x_param}: %{{x}}<br>{y_param}: %{{y}}<extra></extra>",
        )
    )

    fig.update_layout(
        xaxis_title=x_title,
        yaxis_title=y_title,
        title={
            "text": plot_title,
            "xanchor": "center",
            "yanchor": "top",
            "x": 0.5,
        },
    )

    return fig