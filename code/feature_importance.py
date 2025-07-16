# Script to generate Figure 6: Feature importance analysis for SSC prediction in Gilgel Abay and Gumara watersheds
# using Quantile Random Forest (QRF). Computes importance scores for predictors (Discharge, MA_Discharge_3,
# Lag_Discharge, Lag_Discharge_3, Rainfall, ETo, Temperature, Annual_Rainfall, Cumulative_Rainfall) based on
# feature importance analysis (Section 3.2, Table 4). Produces a sorted feature importance CSV and bar plot
# for publication.
# Author: Kindie B. Worku
# Date: 2025-07-16

import pandas as pd
import numpy as np
from quantile_forest import RandomForestQuantileRegressor
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os

# Ensure matplotlib displays in Jupyter
%matplotlib inline
print("Matplotlib inline enabled")

# Set plot style for publication quality (Section 3.2)
sns.set_style('white')
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 18

# Set user and working directory
USER = 'worku'  
BASE_DIR = Path(f"C:\\Users\\{USER}\\Documents\\sediment-yield-analysis")
try:
    os.chdir(BASE_DIR)
    print(f"Working directory set to: {os.getcwd()}")
except Exception as e:
    print(f"Error setting working directory: {e}")
    BASE_DIR = Path(f"C:\\Users\\{USER}\\Desktop\\sediment-yield-analysis")
    os.makedirs(BASE_DIR, exist_ok=True)
    os.chdir(BASE_DIR)
    print(f"Fallback working directory: {os.getcwd()}")

# Define data paths and QRF parameters for each watershed
data_paths = {
    'Gilgel Abay': {
        'intermittent': BASE_DIR / "data" / "Intermittent_data.xlsx",
        'n_samples': 251,
        'qrf_params': {
            'n_estimators': 1000,
            'max_depth': 30,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'max_features': 'log2',
            'random_state': 42
        }
    },
    'Gumara': {
        'intermittent': BASE_DIR / "data" / "Intermittent_data_gum.csv",
        'n_samples': 245,
        'qrf_params': {
            'n_estimators': 1000,
            'max_depth': 30,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'max_features': 'log2',
            'random_state': 42
        }
    }
}

# Verify input file paths
print("\nChecking input file paths...")
for watershed, params in data_paths.items():
    path = params['intermittent']
    print(f"File {path} exists: {path.exists()}")

# Define and create output directory
OUTPUT_DIR = BASE_DIR / "outputs"
try:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Output directory created/confirmed: {OUTPUT_DIR}")
except Exception as e:
    print(f"Error creating output directory: {e}")
    OUTPUT_DIR = Path(f"C:\\Users\\{USER}\\Desktop\\sediment_outputs")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Using fallback output directory: {OUTPUT_DIR}")

def extract_feature_importance(intermittent_path, watershed_name, n_samples, qrf_params, is_excel=False):
    """
    Load data, perform feature engineering with annual and cumulative rainfall, train QRF model, and compute feature importance scores.
    Args:
        intermittent_path (Path): Path to input data file (Excel or CSV).
        watershed_name (str): Name of the watershed ('Gilgel Abay' or 'Gumara').
        n_samples (int): Number of samples to process (251 for Gilgel Abay, 245 for Gumara).
        qrf_params (dict): QRF hyperparameters aligned with model performance evaluation (Section 3.1).
        is_excel (bool): Flag to indicate if input file is Excel (True) or CSV (False).
    Returns:
        Dictionary of feature importance scores or None if processing fails.
    """
    print(f"\n=== Processing {watershed_name} ===")
    
    # Check if input file exists
    if not intermittent_path.exists():
        print(f"Error: File not found at {intermittent_path}")
        return None

    # Load data
    print(f"Loading data from {intermittent_path}...")
    try:
        if is_excel:
            df_inter = pd.read_excel(intermittent_path, engine='openpyxl')
        else:
            df_inter = pd.read_csv(intermittent_path)
        print(f"Successfully loaded {intermittent_path} with {len(df_inter)} rows")
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        return None

    # Validate required columns
    required_cols = ['Date', 'Rainfall', 'Discharge', 'Temperature', 'ETo', 'SSC']
    missing_cols = [col for col in required_cols if col not in df_inter.columns]
    if missing_cols:
        print(f"{watershed_name} missing columns: {missing_cols}")
        return None
    print(f"Columns: {df_inter.columns.tolist()}")

    # Display raw data sample
    print(f"Raw data sample for {watershed_name}:\n{df_inter.head(5)}")

    # Convert columns to numeric
    numeric_cols = ['Rainfall', 'Discharge', 'Temperature', 'ETo', 'SSC']
    for col in numeric_cols:
        df_inter[col] = pd.to_numeric(df_inter[col], errors='coerce')
    
    # Convert Date column to datetime
    try:
        df_inter['Date'] = pd.to_datetime(df_inter['Date'], errors='coerce')
        if df_inter['Date'].isna().any():
            print(f"Warning: Some dates could not be parsed in {watershed_name}. Trying alternative formats...")
            for fmt in ['%d/%m/%Y', '%m/%d/%Y', '%Y-%m-%d', '%Y/%m/%d', '%d-%m-%Y', '%m-%d-%Y']:
                df_inter['Date'] = pd.to_datetime(df_inter['Date'], format=fmt, errors='coerce')
                if not df_inter['Date'].isna().any():
                    print(f"Successfully parsed dates with format: {fmt}")
                    break
            if df_inter['Date'].isna().any():
                print(f"Error: Unable to parse some dates in {watershed_name}")
                print("Unparseable dates:\n", df_inter[df_inter['Date'].isna()][['Date']].head(10))
                return None
        print("Date column sample:\n", df_inter['Date'].head())
        df_inter['Year'] = df_inter['Date'].dt.year
    except Exception as e:
        print(f"Error parsing Date column for {watershed_name}: {str(e)}")
        return None

    # Compute Annual Rainfall
    print(f"Computing Annual Rainfall for {watershed_name}...")
    annual_rainfall = df_inter.groupby('Year')['Rainfall'].sum().reset_index()
    annual_rainfall.columns = ['Year', 'Annual_Rainfall']
    df_inter = df_inter.merge(annual_rainfall, on='Year', how='left')
    print(f"Annual Rainfall sample:\n{annual_rainfall.head()}")

    # Compute Cumulative Rainfall
    print(f"Computing Cumulative Rainfall for {watershed_name}...")
    df_inter = df_inter.sort_values('Date')  # Ensure sorted by date
    df_inter['Cumulative_Rainfall'] = df_inter['Rainfall'].cumsum()
    print(f"Cumulative Rainfall sample:\n{df_inter[['Date', 'Rainfall', 'Cumulative_Rainfall']].head()}")

    # Check for NaN values
    nan_counts = df_inter[numeric_cols + ['Annual_Rainfall', 'Cumulative_Rainfall']].isna().sum()
    if nan_counts.sum() > 0:
        print(f"Warning: NaN values found in {watershed_name}:\n{nan_counts}")
        return None
    print(f"No NaN values in {watershed_name} data")

    # Limit to specified number of samples
    if len(df_inter) < n_samples:
        print(f"Warning: {watershed_name} has only {len(df_inter)} samples, expected {n_samples}")
    df_inter = df_inter.head(n_samples)
    print(f"Final sample count for {watershed_name}: {len(df_inter)}")

    # Check if data is empty
    if df_inter.empty:
        print(f"{watershed_name} data empty after processing")
        return None

    # Feature engineering (Section 2.3)
    print(f"Performing feature engineering for {watershed_name}...")
    df_inter['MA_Discharge_3'] = df_inter['Discharge'].rolling(window=3, min_periods=1).mean().bfill()
    df_inter['Lag_Discharge_1'] = df_inter['Discharge'].shift(1).bfill()
    df_inter['Lag_Discharge_3'] = df_inter['Discharge'].shift(3).bfill()

    # Select predictors 
    predictors = ['Discharge', 'MA_Discharge_3', 'Lag_Discharge_1', 'Lag_Discharge_3', 'Rainfall', 'ETo', 'Temperature', 'Annual_Rainfall', 'Cumulative_Rainfall']
    X_inter = df_inter[predictors]
    y_inter = df_inter['SSC']
    print(f"Predictors for {watershed_name}: {predictors}")

    # Compute and display feature correlations
    print(f"\nFeature correlations for {watershed_name}:\n{X_inter.corr()}\n")
    plt.figure(figsize=(10, 6))
    sns.heatmap(X_inter.corr(), annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0)
    plt.title(f'Correlation Matrix - {watershed_name}', fontsize=20)
    plt.tight_layout()
    corr_output = OUTPUT_DIR / f"correlation_matrix_{watershed_name.lower().replace(' ', '_')}.png"
    try:
        plt.savefig(corr_output, dpi=600, transparent=True, bbox_inches='tight')
        print(f"Correlation matrix saved to {corr_output}")
    except Exception as e:
        print(f"Error saving correlation matrix for {watershed_name}: {e}")
    plt.show()

    # Scale features
    print(f"Scaling features for {watershed_name}...")
    scaler = StandardScaler()
    X_inter_scaled = scaler.fit_transform(X_inter)

    # Train QRF model
    print(f"Training QRF model for {watershed_name}...")
    try:
        qrf = RandomForestQuantileRegressor(**qrf_params)
        qrf.fit(X_inter_scaled, y_inter)
    except Exception as e:
        print(f"Error training QRF for {watershed_name}: {str(e)}")
        return None

    # Extract feature importance scores
    importances = qrf.feature_importances_
    importance_dict = dict(zip(predictors, importances))
    print(f"{watershed_name} Feature Importances:")
    for feature, importance in importance_dict.items():
        print(f"{feature}: {importance:.6f}")
    print(f"Sum of importances: {sum(importances):.6f}")

    return importance_dict

# Compute feature importances for both watersheds
feature_importances = {}
for watershed_name, params in data_paths.items():
    print(f"\nStarting processing for {watershed_name}")
    importances = extract_feature_importance(
        params['intermittent'],
        watershed_name,
        params['n_samples'],
        params['qrf_params'],
        is_excel=(watershed_name == 'Gilgel Abay')
    )
    if importances:
        feature_importances[watershed_name] = importances
    else:
        print(f"Failed to compute importances for {watershed_name}")

# Debug: Check feature importances
print(f"\nNumber of watersheds processed: {len(feature_importances)}")
print("Feature Importances Dictionary:", feature_importances)

# Generate combined feature importance table and plot
if len(feature_importances) >= 1:
    print("\nGenerating combined feature importance plot...")
    predictors = ['Discharge', 'MA_Discharge_3', 'Lag_Discharge_1', 'Lag_Discharge_3', 'Rainfall', 'ETo', 'Temperature', 'Annual_Rainfall', 'Cumulative_Rainfall']
    data = {
        'Feature': predictors,
        'Gilgel Abay': [feature_importances.get('Gilgel Abay', {}).get(f, 0) for f in predictors],
        'Gumara': [feature_importances.get('Gumara', {}).get(f, 0) for f in predictors]
    }
    df = pd.DataFrame(data)

    # Sort features by average importance
    df['Average_Importance'] = df[['Gilgel Abay', 'Gumara']].mean(axis=1)
    df = df.sort_values('Average_Importance', ascending=False).drop('Average_Importance', axis=1)
    sorted_predictors = df['Feature'].tolist()
    print(f"\nFeature Importance DataFrame (Sorted):\n{df}")

    # Save sorted feature importance table
    csv_output = OUTPUT_DIR / "feature_importances_qrf_fig6.csv"
    try:
        df.to_csv(csv_output, index=False)
        print(f"Feature importances saved to {csv_output}")
    except Exception as e:
        print(f"Error saving CSV: {e}")

    # Prepare data for bar plot
    df_melted = pd.melt(df, id_vars=['Feature'], value_vars=['Gilgel Abay', 'Gumara'],
                        var_name='Watershed', value_name='Importance')
    print(f"\nDebug: Melted DataFrame for Plotting:\n{df_melted}")

    # Generate bar plot for feature importance (Figure 6)
    plt.figure(figsize=(10, 6))  # Increased width for more predictors
    sns_plot = sns.barplot(data=df_melted, x='Feature', y='Importance', hue='Watershed',
                           order=sorted_predictors,
                           palette={'Gilgel Abay': '#1f77b4', 'Gumara': '#2ca02c'})

    # Add importance values on top of bars (only for non-zero heights)
    bar_width = 0.3
    for i, row in df_melted.iterrows():
        feature_idx = sorted_predictors.index(row['Feature'])
        watershed = row['Watershed']
        height = row['Importance']
        if height > 0:  # Only show labels for non-zero importances
            x = feature_idx - bar_width / 2 if watershed == 'Gilgel Abay' else feature_idx + bar_width / 2
            plt.text(x, height + 0.01, f'{height:.2f}', ha='center', va='bottom', fontsize=12, color='black')
    
    # Configure plot aesthetics
    plt.xlabel('Feature', fontsize=20)
    plt.ylabel('Feature Importance', fontsize=20)
    plt.xticks(rotation=45, ha='right', fontsize=16)
    plt.yticks(fontsize=16)
    plt.legend(title='Basin', fontsize=16, title_fontsize=16, loc='upper right')
    plt.ylim(0, max(df_melted['Importance']) + 0.1 if df_melted['Importance'].max() > 0 else 1.0)
    plt.tight_layout()

    # Save and display bar plot
    plot_output = OUTPUT_DIR / "Figure6_feature_importance_qrf.png"
    try:
        plt.savefig(plot_output, dpi=600, transparent=True, bbox_inches='tight')
        print(f"Figure 6 saved to {plot_output}")
    except Exception as e:
        print(f"Error saving Figure 6: {e}")
    plt.show()
else:
    print("\nNo feature importances calculated for any watershed.")
    for watershed, importances in feature_importances.items():
        print(f"{watershed}: {importances}")
