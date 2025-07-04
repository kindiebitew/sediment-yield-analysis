# Script for calculating and visualizing feature importance for SSC prediction (Figure 6)
# Purpose: Uses Quantile Random Forest (QRF) to compute feature importances for predictors of
# suspended sediment concentration (SSC) in Gilgel Abay and Gumara watersheds. Generates a
# bar plot comparing importances (Figure 6) and saves results to CSV.
# Author: Kindie B. Worku and co-authors
# Data: Intermittent SSC data from MoWE/ABAO (251 samples for Gilgel Abay, 245 for Gumara)
# Output: CSV with feature importances, bar plot (PNG) for Figure 6
# Reference: Paper Section 2.3 (feature engineering), Section 3.2 (predictor importance)

import pandas as pd
import numpy as np
from quantile_forest import RandomForestQuantileRegressor
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set plot style for publication quality (Times New Roman, font size 18)
# White background with seaborn for clean aesthetics
sns.set_style('white')
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 18

# Define data paths and QRF parameters
# Note: Paths are placeholders; actual data stored locally due to MoWE/ABAO restrictions
# n_samples based on available SSC data; ssc_clip limits outliers (paper Section 2.3)
# QRF parameters tuned via RandomizedSearchCV (paper Section 2.3)
data_paths = {
    'Gilgel Abay': {
        'intermittent': r"D:\Gilgel Abay\Sedigrapgh\Intermittent_data.xlsx",
        'n_samples': 251,  # Number of SSC samples
        'ssc_clip': 5.0,  # Upper bound for SSC clipping
        'qrf_params': {
            'n_estimators': 1000,  # Balance accuracy and computation
            'max_depth': 30,  # Capture non-linear dynamics
            'min_samples_split': 5,  # Prevent overfitting
            'min_samples_leaf': 2,  # Prevent overfitting
            'max_features': 'log2',  # Reduce feature correlation
            'random_state': 42  # Ensure reproducibility
        }
    },
    'Gumara': {
        'intermittent': r"D:\Gumara\Sedigrapgh\Intermittent_data_gum.csv",
        'n_samples': 245,  # Number of SSC samples
        'ssc_clip': 4.0,  # Upper bound for SSC clipping
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

def extract_feature_importance(intermittent_path, watershed_name, n_samples, ssc_clip, qrf_params, is_excel=False):
    """Extract feature importances from QRF model for SSC prediction.
    
    Args:
        intermittent_path (str): Path to intermittent SSC data (CSV or Excel).
        watershed_name (str): Name of watershed (Gilgel Abay or Gumara).
        n_samples (int): Number of samples to use.
        ssc_clip (float): Upper bound for SSC clipping.
        qrf_params (dict): QRF hyperparameters.
        is_excel (bool): Whether data is in Excel format.
    
    Returns:
        dict: Feature importances for predictors, or None if processing fails.
    """
    print(f"\nCalculating feature importance for {watershed_name}...")
    # Check file existence
    if not os.path.exists(intermittent_path):
        print(f"Error: File not found at {intermittent_path}")
        return None

    # Load data (Excel for Gilgel Abay, CSV for Gumara)
    try:
        if is_excel:
            df_inter = pd.read_excel(intermittent_path, engine='openpyxl')
        else:
            df_inter = pd.read_csv(intermittent_path)
        print(f"Successfully loaded {intermittent_path} with {len(df_inter)} rows")
    except Exception as e:
        print(f"Error loading data for {watershed_name}: {str(e)}")
        return None

    # Validate required columns (paper Section 2.2)
    required_cols = ['Rainfall', 'Discharge', 'Temperature', 'ETo', 'SSC']
    missing_cols = [col for col in required_cols if col not in df_inter.columns]
    if missing_cols:
        print(f"{watershed_name} missing columns: {missing_cols}")
        return None

    print(f"Raw data sample for {watershed_name}:\n{df_inter.head(10)}")

    # Ensure numeric columns
    numeric_cols = ['Rainfall', 'Discharge', 'Temperature', 'ETo', 'SSC']
    for col in numeric_cols:
        df_inter[col] = pd.to_numeric(df_inter[col], errors='coerce')
    
    # Check for NaNs before dropping
    nan_counts = df_inter[numeric_cols].isna().sum()
    print(f"NaN counts before dropping for {watershed_name}:\n{nan_counts}")
    
    # Drop rows with missing values
    df_inter = df_inter.dropna(subset=numeric_cols)
    print(f"Rows after dropping NaNs: {len(df_inter)}")

    # Ensure sufficient samples
    if len(df_inter) < n_samples:
        print(f"Warning: {watershed_name} has only {len(df_inter)} samples, expected {n_samples}")
    df_inter = df_inter.head(n_samples)
    print(f"Final sample count for {watershed_name}: {len(df_inter)}")

    if df_inter.empty:
        print(f"{watershed_name} data empty after cleaning.")
        return None

    # Clip SSC to realistic range (paper Section 2.3)
    df_inter['SSC'] = df_inter['SSC'].clip(lower=0.01, upper=ssc_clip)

    # Feature engineering (paper Section 2.3)
    # Log_Discharge linearizes flow-SSC relationship; lags and moving averages capture temporal dynamics
    df_inter['Log_Discharge'] = np.log1p(df_inter['Discharge'].clip(lower=0))
    df_inter['Lag_Discharge'] = df_inter['Discharge'].shift(1).bfill()
    df_inter['Lag_Discharge_2'] = df_inter['Discharge'].shift(2).bfill()
    df_inter['Lag_Discharge_3'] = df_inter['Discharge'].shift(3).bfill()
    df_inter['MA_Discharge_3'] = df_inter['Discharge'].rolling(window=3, min_periods=1).mean().bfill()

    # Select predictors based on physical relevance (paper Section 3.2)
    predictors = ['Rainfall', 'Temperature', 'ETo', 'Log_Discharge', 
                  'Lag_Discharge', 'Lag_Discharge_2', 'Lag_Discharge_3', 'MA_Discharge_3']
    X_inter = df_inter[predictors]
    y_inter = df_inter['SSC']

    # Visualize feature correlations
    print(f"\nFeature correlations for {watershed_name}:\n{X_inter.corr()}\n")
    plt.figure(figsize=(10, 6))
    sns.heatmap(X_inter.corr(), annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0)
    plt.title(f'Correlation Matrix - {watershed_name}')
    plt.tight_layout()
    plt.show()

    # Scale features using StandardScaler for consistency (paper Section 2.3)
    scaler = StandardScaler()
    X_inter_scaled = scaler.fit_transform(X_inter)

    # Train QRF model
    try:
        qrf = RandomForestQuantileRegressor(**qrf_params)
        qrf.fit(X_inter_scaled, y_inter)
    except Exception as e:
        print(f"Error training QRF for {watershed_name}: {str(e)}")
        return None

    # Extract feature importances
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
    importances = extract_feature_importance(
        params['intermittent'],
        watershed_name,
        params['n_samples'],
        params['ssc_clip'],
        params['qrf_params'],
        is_excel=(watershed_name == 'Gilgel Abay')
    )
    if importances:
        feature_importances[watershed_name] = importances
    else:
        print(f"Failed to compute importances for {watershed_name}")

print("\nDebug: Feature Importances Dictionary:")
print(feature_importances)

# Generate bar plot if both watersheds processed successfully
if len(feature_importances) == 2:
    predictors = ['Rainfall', 'Temperature', 'ETo', 'Log_Discharge', 
                  'Lag_Discharge', 'Lag_Discharge_2', 'Lag_Discharge_3', 'MA_Discharge_3']
    data = {
        'Feature': predictors,
        'Gilgel Abay': [feature_importances.get('Gilgel Abay', {}).get(f, 0) for f in predictors],
        'Gumara': [feature_importances.get('Gumara', {}).get(f, 0) for f in predictors]
    }
    df = pd.DataFrame(data)

    # Sort features by average importance across watersheds
    df['Average_Importance'] = df[['Gilgel Abay', 'Gumara']].mean(axis=1)
    df = df.sort_values('Average_Importance', ascending=False).drop('Average_Importance', axis=1)
    sorted_predictors = df['Feature'].tolist()

    print("\nDebug: Feature Importance DataFrame (Sorted):")
    print(df)

    # Save feature importances to CSV
    csv_output = r"D:\Gilgel Abay\Sedigrapgh\feature_importances_qrf_no_discharge_rainfall_sorted.csv"
    df.to_csv(csv_output, index=False)
    print(f"\nFeature importances saved to {csv_output}")

    # Prepare data for plotting
    df_melted = pd.melt(df, id_vars=['Feature'], value_vars=['Gilgel Abay', 'Gumara'], 
                        var_name='Watershed', value_name='Importance')

    print("\nDebug: Melted DataFrame for Plotting:")
    print(df_melted)

    # Create bar plot for Figure 6
    fig, ax = plt.subplots(figsize=(12, 6), facecolor='none')
    sns_plot = sns.barplot(data=df_melted, x='Feature', y='Importance', hue='Watershed', 
                           order=sorted_predictors, ax=ax, 
                           palette={'Gilgel Abay': '#1f77b4', 'Gumara': '#2ca02c'})
    
    # Add importance values on top of bars
    bar_width = 0.4
    for i, row in df_melted.iterrows():
        feature_idx = sorted_predictors.index(row['Feature'])
        watershed = row['Watershed']
        height = row['Importance']
        x = feature_idx - bar_width / 2 if watershed == 'Gilgel Abay' else feature_idx + bar_width / 2
        ax.text(x, height + 0.01, f'{height:.2f}', ha='center', va='bottom', fontsize=12, color='black')
    
    # Configure plot
    ax.set_xlabel('Feature', fontsize=18)
    ax.set_ylabel('Feature Importance', fontsize=18)
    ax.tick_params(axis='x', rotation=45, labelsize=16)
    ax.tick_params(axis='y', labelsize=16)
    ax.legend(title='Basin', fontsize=16, title_fontsize=16, loc='upper right')
    ax.set_ylim(0, max(df_melted['Importance']) + 0.1)
    plt.tight_layout()

    # Save plot as PNG
    plot_output = r"D:\Gilgel Abay\Sedigrapgh\feature_importances_qrf_no_discharge_rainfall_sorted.png"
    plt.savefig(plot_output, transparent=True, dpi=600, bbox_inches='tight')
    plt.show()
    print(f"Feature importance plot saved to {plot_output}")
else:
    print("Failed to calculate feature importances for both watersheds. Available data:")
    for watershed, importances in feature_importances.items():
        print(f"{watershed}: {importances}")