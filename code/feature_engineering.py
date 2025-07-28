# Script to perform feature engineering for Suspended Sediment Concentration (SSC) prediction in Gilgel Abay and Gumara watersheds.
# Generates predictors for model evaluation (Section 2.3) and correlation matrices.
# No normalization or outlier removal for tree-based models (Random Forest, Gradient Boosting, Quantile Random Forest).
# Author: Kindie B. Worku
# Date: 2025-07-18

# Importing necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os

# Setting plot style for publication quality
sns.set_style('white')
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 18

# Setting user and working directory
USER = 'worku'
BASE_DIR = Path(f"C:\\Users\\{USER}\\Documents\\sediment-yield-analysis")
try:
    os.chdir(BASE_DIR)
    print(f"Working directory set to: {os.getcwd()}")
except Exception as e:
    print(f"Error setting working directory: {e}")
    # Fallback to desktop directory if primary directory fails
    BASE_DIR = Path(f"C:\\Users\\{USER}\\Desktop\\sediment-yield-analysis")
    os.makedirs(BASE_DIR, exist_ok=True)
    os.chdir(BASE_DIR)
    print(f"Fallback working directory: {os.getcwd()}")

# Defining data paths and sample sizes for each watershed
data_paths = {
    'Gilgel Abay': {
        'intermittent': BASE_DIR / "data" / "Intermittent_data.xlsx",
        'n_samples': 251,
        'is_excel': True
    },
    'Gumara': {
        'intermittent': BASE_DIR / "data" / "Intermittent_data_gum.csv",
        'n_samples': 245,
        'is_excel': False
    }
}

# Verifying input file paths
print("\nChecking input file paths...")
for watershed, params in data_paths.items():
    path = params['intermittent']
    print(f"File {path} exists: {path.exists()}")

# Creating output directory
OUTPUT_DIR = BASE_DIR / "outputs"
try:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Output directory created/confirmed: {OUTPUT_DIR}")
except Exception as e:
    print(f"Error creating output directory: {e}")
    # Fallback to desktop output directory if primary directory fails
    OUTPUT_DIR = Path(f"C:\\Users\\{USER}\\Desktop\\sediment_outputs")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Using fallback output directory: {OUTPUT_DIR}")

# Function to perform feature engineering
def engineer_features(intermittent_path, watershed_name, n_samples, is_excel=False):
    """
    Load data, perform feature engineering with annual/cumulative rainfall, lags, and seasonal indices,
    and save processed data. No normalization or outlier removal for tree-based models.
    
    Args:
        intermittent_path (Path): Path to input data file (Excel or CSV).
        watershed_name (str): Name of the watershed ('Gilgel Abay' or 'Gumara').
        n_samples (int): Number of samples to process (251 for Gilgel Abay, 245 for Gumara).
        is_excel (bool): Flag to indicate if input file is Excel (True) or CSV (False).
    
    Returns:
        bool: True if successful, False otherwise.
    """
    print(f"\n=== Processing {watershed_name} ===")
    
    # Checking if input file exists
    if not intermittent_path.exists():
        print(f"Error: File not found at {intermittent_path}")
        return False
    
    # Loading data
    print(f"Loading data from {intermittent_path}...")
    try:
        if is_excel:
            df = pd.read_excel(intermittent_path, engine='openpyxl')
        else:
            df = pd.read_csv(intermittent_path)
        print(f"Successfully loaded {intermittent_path} with {len(df)} rows")
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        return False
    
    # Validating required columns
    required_cols = ['Date', 'Rainfall', 'Discharge', 'Temperature', 'ETo', 'SSC']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"{watershed_name} missing columns: {missing_cols}")
        return False
    print(f"Columns: {df.columns.tolist()}")
    
    # Converting columns to numeric
    numeric_cols = ['Rainfall', 'Discharge', 'Temperature', 'ETo', 'SSC']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Converting Date to datetime with multiple format attempts
    try:
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        for fmt in ['%d/%m/%Y', '%m/%d/%Y', '%Y-%m-%d', '%Y/%m/%d', '%d-%m-%Y', '%m-%d-%Y']:
            if df['Date'].isna().any():
                df['Date'] = pd.to_datetime(df['Date'], format=fmt, errors='coerce')
                if not df['Date'].isna().any():
                    print(f"Successfully parsed dates with format: {fmt}")
                    break
        if df['Date'].isna().any():
            print(f"Error: Unable to parse some dates in {watershed_name}")
            return False
        df['Year'] = df['Date'].dt.year
    except Exception as e:
        print(f"Error parsing Date column for {watershed_name}: {str(e)}")
        return False
    
    # Computing Annual Rainfall
    print(f"Computing Annual Rainfall for {watershed_name}...")
    annual_rainfall = df.groupby('Year')['Rainfall'].sum().reset_index()
    annual_rainfall.columns = ['Year', 'Annual_Rainfall']
    df = df.merge(annual_rainfall, on='Year', how='left')
    
    # Computing Cumulative Rainfall
    print(f"Computing Cumulative Rainfall for {watershed_name}...")
    df = df.sort_values('Date')
    df['Cumulative_Rainfall'] = df['Rainfall'].cumsum()
    
    # Performing feature engineering
    print(f"Performing feature engineering for {watershed_name}...")
    df['MA_Discharge_3'] = df['Discharge'].rolling(window=3, min_periods=1).mean().bfill()
    df['Lag_Discharge_1'] = df['Discharge'].shift(1).bfill()
    df['Lag_Discharge_3'] = df['Discharge'].shift(3).bfill()
    df['Julian_Day'] = df['Date'].dt.dayofyear
    df['Sin_Julian'] = np.sin(2 * np.pi * df['Julian_Day'] / 365.25)
    df['Cos_Julian'] = np.cos(2 * np.pi * df['Julian_Day'] / 365.25)
    
    # Checking for NaN values in features
    feature_cols = ['Discharge', 'MA_Discharge_3', 'Lag_Discharge_1', 'Lag_Discharge_3', 'Rainfall', 'ETo',
                    'Temperature', 'Annual_Rainfall', 'Cumulative_Rainfall', 'Sin_Julian', 'Cos_Julian']
    nan_counts = df[feature_cols + ['SSC']].isna().sum()
    if nan_counts.sum() > 0:
        print(f"Warning: NaN values found in {watershed_name}:\n{nan_counts}")
        return False
    
    # Limiting samples to specified number
    df = df.head(n_samples)
    print(f"Final sample count for {watershed_name}: {len(df)}")
    
    # Generating and saving correlation matrix
    print(f"Generating correlation matrix for {watershed_name}...")
    corr_matrix = df[feature_cols].corr()
    plt.figure(figsize=(12, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0)
    plt.title(f'Correlation Matrix - {watershed_name}', fontsize=20)
    plt.tight_layout()
    corr_output = OUTPUT_DIR / f"correlation_matrix_{watershed_name.lower().replace(' ', '_')}.png"
    try:
        plt.savefig(corr_output, dpi=600, transparent=True, bbox_inches='tight')
        print(f"Correlation matrix saved to {corr_output}")
    except Exception as e:
        print(f"Error saving correlation matrix for {watershed_name}: {e}")
    plt.close()
    
    # Saving processed data
    output_path = OUTPUT_DIR / f"{watershed_name.lower().replace(' ', '_')}_features.csv"
    try:
        df.to_csv(output_path, index=False)
        print(f"Processed data saved to {output_path}")
        return True
    except Exception as e:
        print(f"Error saving processed data for {watershed_name}: {str(e)}")
        return False

# Processing each watershed
for watershed_name, params in data_paths.items():
    success = engineer_features(
        params['intermittent'],
        watershed_name,
        params['n_samples'],
        params['is_excel']
    )
    if not success:
        print(f"Failed to process {watershed_name}")
