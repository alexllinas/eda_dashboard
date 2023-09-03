""" Helper functions for Streamlit app."""
import pandas as pd
import streamlit as st

def select_columns(numeric_cols: list[str],
                   categorical_cols: list[str],
                   chart_type: str,
                   prefix: str="") -> tuple[str, str | None, str | None]:
    """ Display UI elements for column selection based on chart type and return the selected columns.
    
    Args:
        numeric_cols (list[str]): List of numeric columns.
        categorical_cols (list[str]): List of categorical columns.
        chart_type (str): Type of chart (e.g., "Scatterplot", "Stacked Bar Plot").
        prefix (str): Optional prefix for UI label.

    Returns:
        tuple[str, str | None, str | None]: Selected x column, y column, and color column.
    """
    col1, col2, col3 = st.columns(3)
    if chart_type in ["Scatterplot", "Stacked Bar Plot"]:
        x_col = col1.selectbox(f"{prefix}X Variable:",
                               numeric_cols + categorical_cols, key=f"{prefix}_x_col_{chart_type}")
        y_col = col2.selectbox(f"{prefix}Y Variable:",
                               numeric_cols + categorical_cols, key=f"{prefix}_y_col_{chart_type}")
    else:
        x_col = col1.selectbox(f"{prefix}Variable:",
                               numeric_cols + categorical_cols if chart_type == "Histogram" else categorical_cols,
                               key=f"{prefix}_single_col_{chart_type}")
        y_col = None

    color_col = col3.selectbox(f"{prefix}Color Variable (optional):",
                               ["None"] + categorical_cols, key=f"{prefix}_color_col_{chart_type}")
    if color_col == "None":
        color_col = None
    
    return x_col, y_col, color_col


def filter_data(data: pd.DataFrame,
                column: str,
                col_type: str="numeric") -> pd.DataFrame:
    """Display UI elements for data filtering based on column type and return the filtered data.
    
    Args:
        data (pd.DataFrame): The data to filter.
        column (str): Column based on which data is to be filtered.
        col_type (str): Type of the column (default is "numeric", can also be "categorical").

    Returns:
        pd.DataFrame: Filtered data.
    """
    if col_type == "numeric":
        min_val, max_val = st.slider(f"Select Range for {column}:",
                                     float(data[column].min()),
                                     float(data[column].max()),
                                     (float(data[column].min()), float(data[column].max())),
                                     key=f"{column}_slider")
        return data[(data[column] >= min_val) & (data[column] <= max_val)]
    else:  # categorical
        selected_categories = st.multiselect("Select Values",
                                             data[column].unique(),
                                             default=data[column].unique())
        return data[data[column].isin(selected_categories)]
