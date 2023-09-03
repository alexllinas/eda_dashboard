"""Module for loading data from a file."""
import os
import tempfile
import pandas as pd
import xarray as xr
import streamlit as st

@st.cache_data
def load_data(uploaded_file) -> pd.DataFrame | xr.Dataset:
    """Load data from a file.
    
    Args:
        uploaded_file (FileUploader): The file uploaded by the user.
        
    Returns:
        pd.DataFrame | xr.Dataset: The loaded data.
    
    """
    if uploaded_file:
        extension = os.path.splitext(uploaded_file.name)[1]
        
        if extension == '.csv':
            try:
                return pd.read_csv(uploaded_file, delimiter=',')
            except:
                return pd.read_csv(uploaded_file, delimiter=';')
        elif extension == '.h5':
            # Create temp file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".h5") as temp_file:
                temp_file.write(uploaded_file.getvalue())
                return xr.open_dataset(temp_file.name, engine='h5netcdf')
                # Optional: Delete temp file
                # os.remove(temp_file.name)
        else:
            return None
    return None

