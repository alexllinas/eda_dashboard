"""Module for exploratory data analysis."""
from __future__ import annotations
from typing import Any

import pandas as pd
import numpy as np
import xarray as xr
import scipy.stats as ss
import streamlit as st

@st.cache_data
def data_summary(data: pd.DataFrame | xr.Dataset) -> dict[str, Any]:
    """Generate a detailed summary of the dataset.
    
    Args:
        data (pd.DataFrame | xr.Dataset): The dataset, which can be either a pandas DataFrame or an xarray Dataset.
    
    Returns:
        dict: A dictionary containing the details of the summary.
    """
    if isinstance(data, pd.DataFrame):
        # Get dataset dimensions
        rows, cols = data.shape

        # Handle missing data
        missing_data = data.isnull().sum()
        total_missing = missing_data.sum()
        missing_data_filtered = {col: miss for col, miss in missing_data.items() if miss > 0} 

        # Data types
        data_types = data.dtypes.value_counts().to_dict()
        columns_by_type = {str(k): v.tolist() for k, v in data.columns.to_series()
                           .groupby(data.dtypes, sort=False).groups.items()}
        
        # Outliers detection
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        outliers_by_column = {}
        for col in numeric_cols:
            outliers = detect_outliers(data, col)
            if outliers.sum() > 0:
                outliers_by_column[col] = outliers.sum()

        summary = {
            'Rows': rows,
            'Columns': cols,
            'Total Missing Values': total_missing,
            'Missing Values by Column': missing_data_filtered,
            'Data Types Distribution': data_types,
            'Columns by Data Type': columns_by_type,
            'Outliers by Column': outliers_by_column,
        }

    elif isinstance(data, xr.Dataset):
        # Get number of coordinates, dimensions, variables, and attributes
        coords = len(data.coords)
        dimensions = len(data.dims)
        variables = len(data.data_vars)
        attributes = len(data.attrs)
        
        # Handle missing data for each variable
        missing_data = {var: int(data[var].isnull().sum().values) for var in data.data_vars}
        total_missing = sum(missing_data.values())
        missing_data_filtered = {var: miss for var, miss in missing_data.items() if miss > 0}

        # Get data types for each variable
        data_types = {var: str(data[var].dtype) for var in data.data_vars}
        # Outliers detection
        numeric_vars = [var for var, da in data.data_vars.items() if np.issubdtype(da.dtype, np.number)]
        outliers_by_var = {}
        for var in numeric_vars:
            outliers = detect_outliers(data, var)
            if outliers.sum().values > 0:  # Note the use of `.values` to get the underlying numpy value
                outliers_by_var[var] = int(outliers.sum().values)

        summary = {
            'Coordinates': coords,
            'Dimensions': dimensions,
            'Variables': variables,
            'Attributes': attributes,
            'Total Missing Values': total_missing,
            'Missing Values by Variable': missing_data_filtered,
            'Data Types by Variable': data_types,
            'Outliers by Variable': outliers_by_var,
        }
        
    return summary

@st.cache_data
def handle_missing_values(data: pd.DataFrame, method: str="Drop Rows") -> pd.DataFrame:
    """
    Function to handle missing values.
    Args:
        data (pd.DataFrame): DataFrame.
        method (str): Method to handle missing values ("drop" or "impute").
    Returns:
        pd.DataFrame: DataFrame without missing values.
    """
    if method == "Drop Rows":
        return data.dropna()
    elif method == "Drop Columns":
        return data.dropna(axis=1)
    elif method == "Impute with Mean/Median/Mode":
        for col in data.columns:
            if data[col].dtype == 'O':  # Object dtype, asume categorical or string
                data[col].fillna(data[col].mode()[0], inplace=True)
            else:
                data[col].fillna(data[col].median(), inplace=True)
        return data
    else:
        return data

@st.cache_data
def detect_outliers(data: pd.DataFrame | xr.Dataset, variable: str) -> pd.Series | xr.DataArray:
    """Detect outliers in a specific column or variable using the IQR method.
    
    Args:
        data (pd.DataFrame | xr.DataArray): DataFrame or Dataset.
        variable (str): Name of the column or variable.
    
    Returns:
        pd.Series | xr.DataArray: A boolean series or data array indicating whether each point is an outlier.
    """
    Q1 = data[variable].quantile(0.25)
    Q3 = data[variable].quantile(0.75)
    IQR = Q3 - Q1
    outliers = (data[variable] < (Q1 - 1.5 * IQR)) | (data[variable] > (Q3 + 1.5 * IQR))
    
    return outliers

@st.cache_data
def handle_outliers(data, column, method='remove', replace_value=None):
    """
    Handle outliers from a specific column.

    Args:
        data (pd.DataFrame): DataFrame.
        column (str): Column name.
        method (str): Method to handle outliers ('remove', 'replace', 'impute').
        replace_value (float, optional): Value to replace outliers if method is 'replace'.
    Returns:
        pd.DataFrame: DataFrame without outliers.
    """
    outliers = detect_outliers(data, column)
    
    if method == 'remove':
        return data[~outliers]
    
    elif method == 'replace' and replace_value is not None:
        data.loc[outliers, column] = replace_value
        return data
    
    elif method == 'impute':
        median_value = data[column].median()
        data.loc[outliers, column] = median_value
        return data

@st.cache_data
def change_dtype(data, column, new_dtype):
    """
    Change data type for a given column.
    Args:
        data (pd.DataFrame): DataFrame.
        col (str): Column name.
        new_dtype (str): New data type ("categorical", "numeric", etc.)
    Returns:
        pd.DataFrame: DataFrame with new column data type.
    """
    if new_dtype == "Numeric":
        if data[column].dtype == 'O':
            data[column] = data[column].str.replace(',', '.')  # Replace commas with dots
        data[column] = pd.to_numeric(data[column], errors='coerce')
    elif new_dtype == "Categorical":
        data[column] = data[column].astype('O')
    return data


def cramers_v(x: pd.Series, y: pd.Series) -> float:
    """
    Calculate Cramer's V statistic for categorical-categorical association.
    
    Args:
        x (pd.Series): First categorical variable.
        y (pd.Series): Second categorical variable.
    
    Returns:
        float: Cramer's V statistic.
    """
    # If one of the Series only has one unique value, return NaN
    if x.nunique() <= 1 or y.nunique() <= 1:
        return np.nan

    # Compute the confusion matrix
    confusion_matrix = pd.crosstab(x, y)
    
    # If the matrix is empty in one dimension, return NaN
    if confusion_matrix.shape[0] == 0 or confusion_matrix.shape[1] == 0:
        return np.nan

    # Compute the chi-squared statistic from the confusion matrix
    chi2 = ss.chi2_contingency(confusion_matrix)[0]
    # Get the total count from the matrix
    n = confusion_matrix.sum().sum()
    
    # If the matrix only has one entry, return NaN
    if n == 1:
        return np.nan

    # Calculate phi-squared (a measure of association)
    phi2 = chi2 / n
    # Get the matrix dimensions
    r, k = confusion_matrix.shape
    # Apply a bias correction to phi-squared
    phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))
    # Correct the row and column dimensions to account for bias
    rcorr = r - ((r-1)**2)/(n-1)
    kcorr = k - ((k-1)**2)/(n-1)

    # Determine the smaller of the corrected dimensions
    denominator = min((kcorr-1), (rcorr-1))
    # If denominator is zero, return NaN
    if denominator == 0:
        return np.nan

    return np.sqrt(phi2corr / denominator)


def calculate_cramers_v(data: pd.DataFrame) -> pd.DataFrame:
    """Calculate Cramer's V statistic for all pairs of categorical variables in a dataset.
    
    Args:
        data (pd.DataFrame): The dataset.
        
    Returns:
        pd.DataFrame: A matrix containing the Cramer's V statistic for all pairs of categorical variables.
    
    """
    cols = data.columns
    n = len(cols)
    cramers_matrix = pd.DataFrame(np.zeros((n, n)), columns=cols, index=cols)
    
    for i, col1 in enumerate(cols):
        for j, col2 in enumerate(cols):
            if i < j:  # just calculate the upper triangle
                cramers_value = cramers_v(data[col1], data[col2])
                cramers_matrix.loc[col1, col2] = cramers_value
                cramers_matrix.loc[col2, col1] = cramers_value  # use symmetry
            elif i == j:
                cramers_matrix.loc[col1, col2] = 1  # diagonal is always 1

    return cramers_matrix
