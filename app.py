"""main app file for streamlit app"""""
import io
import contextlib

import numpy as np
import pandas as pd
import streamlit as st
import xarray as xr

from src.data_loader import load_data
import src.eda_tools as et
import src.visuals as visuals

st.set_page_config(layout="wide")

st.title("Automated EDA Dashboard")

uploaded_file = st.file_uploader("Choose a file (.csv, .h5)", type=['csv', 'h5'])
data = load_data(uploaded_file)

if data is not None:
    cols_to_drop = ['Unnamed: 0', 'index'] # drop common unnecessary columns
    if any(col in data.columns for col in cols_to_drop):
        data = data.drop(columns=[col for col in cols_to_drop if col in data.columns])

    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Data Summary", "Outliers", "Distributions",
                                            "Correlations", "Basic Visuals"])
    with tab1: # Data Summary
        # If it's a DataFrame
        if isinstance(data, pd.DataFrame):
            numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
            categorical_cols = data.select_dtypes(exclude=[np.number]).columns.tolist()
            
            # Empty Columns Handling
            empty_cols = data.columns[data.isnull().all()].tolist()
            if empty_cols:
                st.sidebar.warning(f"Empty columns detected: {', '.join(empty_cols)}.")
                remove_empty_cols = st.sidebar.checkbox("Do you want to remove them?", value=True)
                
                if remove_empty_cols:
                    data = data.drop(columns=empty_cols)
                    st.sidebar.success("Empty Columns Deleted.")
            
            # Missing Values Handling
            missing_values_cols = data.columns[data.isnull().any()].tolist()
            if missing_values_cols:
                missing_method = st.sidebar.selectbox("Handle Missing Values:",
                                                      ["None", "Drop Rows", "Drop Columns","Impute with Mean/Median/Mode"])
                if missing_method != "None":
                    data = et.handle_missing_values(data, method=missing_method)
            
            # Change Data Type
            columns_to_change_dtype = st.sidebar.multiselect("Select Columns to Change Data Type:",
                                                    data.columns.tolist(), default=[], key=0)

            if columns_to_change_dtype:
                new_dtype = st.sidebar.radio("Select New Data Type for Selected Columns:",
                                             ["Numeric", "Categorical"], key=1)
                for col in columns_to_change_dtype:
                    data = et.change_dtype(data, col, new_dtype)
                numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist() # update numeric cols
                categorical_cols = data.select_dtypes(exclude=[np.number]).columns.tolist() # update categorical cols
                
                columns_to_change_dtype2 = st.sidebar.multiselect("Select Columns to Change Data Type:",
                                                    data.columns.tolist(), default=[], key=2)
                if columns_to_change_dtype2:
                    new_dtype = st.sidebar.radio("Select New Data Type for Selected Columns:",
                                                 ["Numeric", "Categorical"], key=3)
                    for col in columns_to_change_dtype2:
                        data = et.change_dtype(data, col, new_dtype)
                        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist() # update numeric cols
                        categorical_cols = data.select_dtypes(exclude=[np.number]).columns.tolist() # update categorical cols
            
            # Outliers Handling:
            summary = et.data_summary(data)
            cols_with_outliers = [col for col, val in summary['Outliers by Column'].items()]
            if cols_with_outliers:                   
                outlier_handle_col = st.sidebar.selectbox("Select Column to Handle Outliers:",
                                                          ["None"] + cols_with_outliers)
                if outlier_handle_col != "None":
                    method = st.sidebar.radio("Select Method:", ("remove", "replace", "impute"))
                    if method == 'replace':
                        replace_val = st.sidebar.number_input("Enter the replace value:")
                        data = et.handle_outliers(data, outlier_handle_col, method, replace_val)
                    else:
                        data = et.handle_outliers(data, outlier_handle_col, method)
                    
                    summary = et.data_summary(data)
            
            # Display overview of data
            st.write("Loaded Data Overview:")
            st.write(data)
            for key, value in summary.items():
                st.write(f"**{key}:** {value}")
        
        # If xarray Dataset
        if isinstance(data, xr.Dataset):
            # Capture .info() output as string
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                data.info()
            info_str = buf.getvalue()
            
            st.text(info_str)
            
            # Display overview of data
            summary = et.data_summary(data)
            for key, value in summary.items():
                st.write(f"**{key}:** {value}")

    with tab2: # Outliers
        # Visualize Outliers:
        if cols_with_outliers:
            if st.button("Visualize Outliers"):
                st.success("Visualizing Outliers...")
                cols = st.columns(2)
                i = 0
                for col in cols_with_outliers:
                    with cols[i]:
                        fig = visuals.plot_box(data, col)
                        st.plotly_chart(fig)
                    i = 1 - i # toggle between columns
    
    with tab3: # Distributions
        columns = st.multiselect("Select Columns to Visualize Distributions:",
                                 data.columns.tolist(), default=data.columns.tolist())
        color_col_choice = st.selectbox("Select Color Variable (optional):",
                                        ["None"] + [col for col in categorical_cols if col not in columns])
        if not color_col_choice or color_col_choice == "None":
            color_col_choice = None
        cols = st.columns(2)
        i = 0
        if st.button("Visualize Distributions"):
            st.success("Visualizing Distributions...")  
            if isinstance(data, pd.DataFrame):
                for col in columns:
                    with cols[i]:
                        if pd.api.types.is_numeric_dtype(data[col]):
                            st.plotly_chart(visuals.plot_numeric(data, col, color_col_choice))
                        else:
                            st.plotly_chart(visuals.plot_categorical(data, col, color_col_choice))
                    i = 1 - i # toggle between columns
            elif isinstance(data, xr.Dataset):
                for var in data.data_vars:
                    with cols[i]:
                        if np.issubdtype(data[var].dtype, np.number):
                            st.plotly_chart(visuals.plot_numeric(data, var, color_col_choice))
                        else:
                            st.plotly_chart(visuals.plot_categorical(data, var, color_col_choice))
                    i = 1 - i # toggle between columns
                
    with tab4: # Correlations
        if isinstance(data, pd.DataFrame):
            corr_cols = st.multiselect("Select Columns to Visualize Correlations:",
                                       numeric_cols, default=numeric_cols)
            cat_vars = st.multiselect("Select Categorical Variables to Visualize Correlations:",
                                      categorical_cols, default=categorical_cols)
            if st.button("Visualize Correlations"):
                st.success("Visualizing Correlations...")
                if not corr_cols:
                    corr_cols = None
                fig = visuals.plot_corr(data, columns=corr_cols)
                st.plotly_chart(fig)
                
                if not cat_vars:
                    cat_vars = None
                fig = visuals.plot_corr(data, columns=cat_vars, method='cramers_v')
                st.plotly_chart(fig)

    with tab5: # Visuals
        # If it's a DataFrame
        if isinstance(data, pd.DataFrame):
            col1, col2 = st.columns(2)
            num_choice = col1.selectbox("Select numeric variable:", numeric_cols)
            color_col_choice = col2.selectbox("Select color variable (optional):", ["None"] + categorical_cols)
            min_val, max_val = col1.slider(f"Select Range for {num_choice}:",
                                           float(data[num_choice].min()), float(data[num_choice].max()),
                                           (float(data[num_choice].min()), float(data[num_choice].max())),
                                           key=f"Histogram_slider_{num_choice}")
            cat_choice = col2.selectbox("Select categoric variable:",
                                        [col for col in categorical_cols if col != color_col_choice])
            
            # Filter data based on user selection
            filtered_data = data[(data[num_choice] >= min_val) & (data[num_choice] <= max_val)]
            category_col_values = filtered_data[cat_choice].unique()
            selected_categories = col2.multiselect(f"Filter {cat_choice}",
                                                   ["ALL"] + list(category_col_values), default='ALL')
            if "ALL" in selected_categories:
                selected_categories = list(category_col_values)
            filtered_data = filtered_data[filtered_data[cat_choice].isin(selected_categories)]
            
            # Check if the button has been pressed before
            if 'button_pressed' not in st.session_state:
                st.session_state.button_pressed = False
            
            if col1.button("Visualize"):
                st.session_state.button_pressed = not st.session_state.button_pressed  # toggle the state

            if st.session_state.button_pressed:
                st.success("Visualizing...")
                col1, col2 = st.columns(2)

                with col1:
                    if not color_col_choice or color_col_choice == "None":
                        color_col_choice = None
                    fig = visuals.plot_numeric(filtered_data, num_choice, color_col_choice)
                    st.plotly_chart(fig)

                with col2:
                    fig = visuals.plot_categorical(filtered_data, cat_choice, color_col_choice)
                    st.plotly_chart(fig)

                with col1:
                    x_col_choice = num_choice
                    y_col_choice = st.selectbox("Select Scatterplot Y Variable:",
                                                [col for col in filtered_data.columns.to_list() if col != x_col_choice])
                    fig = visuals.plot_scatter(filtered_data, x_col_choice, y_col_choice, color_col_choice)
                    st.plotly_chart(fig)

                with col2:
                    x_col_choice = cat_choice
                    y_col_choice = num_choice
                    color_col_choice = color_col_choice if color_col_choice else st.selectbox("Stacked Bar Color Var:", categorical_cols)
                    fig = visuals.plot_stacked_bar(filtered_data, x_col_choice, y_col_choice, color_col_choice)
                    st.plotly_chart(fig)

    # TODO: Add support for xarray Dataset


