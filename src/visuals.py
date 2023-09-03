"""Module for visualizing data."""
import numpy as np
import pandas as pd
import xarray as xr

import plotly.express as px
import plotly.figure_factory as ff
import streamlit as st

from src.eda_tools import calculate_cramers_v

@st.cache_data
def plot_box(data: pd.DataFrame, column: str) -> px.box:
    """
    Create a box plot for th column.
    """
    fig = px.box(data, y=column)
    return fig

@st.cache_data
def plot_numeric(data: pd.DataFrame | xr.Dataset,
                 variable: str,
                 cat_varible: str | None=None) -> px.histogram:
    if isinstance(data, xr.Dataset):
        # Convert xarray DataArray to pandas DataFrame
        df = data[variable].to_dataframe().reset_index()
        col = df.columns[-1] # Assuming the last column is the variable of interest
    else:
        df = data
        col = variable

    fig = px.histogram(df,
                       x=col,
                       color=cat_varible,
                       nbins=30,
                       marginal="box",
                       title=f'Histogram for {col}')
    return fig


@st.cache_data
def plot_categorical(data: pd.DataFrame | xr.Dataset,
                     variable: str,
                     cat_variable: str | None=None) -> px.bar:
    if isinstance(data, xr.Dataset):
        # Convert xarray DataArray to pandas DataFrame
        df = data[variable].to_dataframe().reset_index()
        col = df.columns[-1] # Assuming the last column is the variable of interest
    else:
        df = data
        col = variable

    if cat_variable:
        # Group by both columns and then reset index
        aggregated_data = df.groupby([col, cat_variable]).size().reset_index(name='Count')
        fig = px.bar(aggregated_data,
                     x=col,
                     y='Count',
                     color=cat_variable,
                     title=f'Bar Plot for {col} by {cat_variable}')
    else:
        aggregated_data = df[col].value_counts().reset_index()
        aggregated_data.columns = [col, 'Count']
        fig = px.bar(aggregated_data,
                     x=col,
                     y='Count',
                     title=f'Bar Plot for {col}')

    return fig

@st.cache_data
def plot_scatter(data: pd.DataFrame,
                 x_col: str,
                 y_col: str,
                 color_col: str | None=None) -> px.scatter:
    fig = px.scatter(data,
                     x=x_col,
                     y=y_col,
                     color=color_col,
                     title=f'Scatterplot of {y_col} vs {x_col}')
    return fig

@st.cache_data
def plot_stacked_bar(data: pd.DataFrame,
                     x_col: str,
                     y_col: str,
                     category_col: str) -> px.bar:
    fig = px.bar(data,
                 x=x_col,
                 y=y_col,
                 color=category_col,
                 title=f'Stacked Bar of {y_col} by {x_col}')
    return fig

@st.cache_data
def plot_corr(data: pd.DataFrame,
              columns: str | None=None,
              method: str='pearson') -> px.imshow:
    """
    Generates a corrplot or heatmap correlationes using Plotly.
    
    Args:
    - data (pd.DataFrame): DataFrame.
    - columns (list, optional): Columns list to include in the corrplot. 
                                If None, all numeric or categorical columns are used.
                                    Default: None.
    - method (str, optional): Correlation method ('pearson', 'spearman', 'kendall', 'cramers_v').
    
    Returns:
    - plotly.graph_objs._figure.Figure: heatmap plotly figure.
    """
    if method == 'cramers_v':
        if columns is None:
            columns = data.select_dtypes(exclude=[np.number]).columns.tolist()
        corr_matrix = calculate_cramers_v(data[columns])
   
        fig = ff.create_annotated_heatmap(
            z=corr_matrix.values,
            x=list(corr_matrix.columns),
            y=list(corr_matrix.index),
            annotation_text=corr_matrix.round(2).values,
            showscale=True,
            colorscale='Viridis'
        )
        fig.update_layout(title='Categorical Correlation Plot', xaxis_title=None, yaxis_title=None)
        
        return fig
    
    if columns is None:
        columns = data.select_dtypes(include=[np.number]).columns.tolist()
        
    corr_matrix = data[columns].corr(method='pearson')
    
    fig = ff.create_annotated_heatmap(
        z=corr_matrix.values,
        x=list(corr_matrix.columns),
        y=list(corr_matrix.index),
        annotation_text=corr_matrix.round(2).values,
        showscale=True,
        colorscale='Viridis'
    )
    
    fig.update_layout(title='Correlation Plot', xaxis_title=None, yaxis_title=None)
    
    return fig


