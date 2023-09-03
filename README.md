# Automated EDA Dashboard

An interactive dashboard for automated Exploratory Data Analysis (EDA) using Streamlit. Supports data upload from CSV and HDF5 formats.

## Features

- Upload datasets in CSV or HDF5 formats up to 200MB.
- Get instant visualizations tailored to your data type: histograms, scatter plots, bar charts, and more.
- Customize visualizations by selecting specific variables.
- View statistics and correlations to quickly understand data distributions and relationships.

## Installation

1. Clone the repository:

```git clone https://github.com/alexllinas/eda_dashboard.git```

2. Navigate to the project directory:

```cd eda_dashboard```

3. Set up a virtual environment using Conda and activate it (optional but recommended):

```conda create --name eda_dashboard python=3.10```
```conda activate eda_dashboard```


4. Install the required packages:

```pip install -r requirements.txt```


## Usage

1. Activate the Conda environment (if using):

```conda activate eda_dashboard```

2. Run the Streamlit app:

```streamlit run app.py```

3. Open a browser and navigate to the provided local URL (usually `http://localhost:8501`).

4. Upload your dataset and explore the automated EDA visualizations!

## Contributing

If you'd like to contribute, please fork the repository and use a feature branch. Pull requests are warmly welcome.

## Licensing

The code in this project is licensed under MIT license.

## Contact

For any inquiries or feedback, please contact alexllinas@gmail.com(mailto:alexllinas@gmail.com).
