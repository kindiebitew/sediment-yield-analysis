#Script to generate Figure 6: Feature importance analysis for SSC prediction in Gilgel Abay and Gumara watersheds
#using Quantile Random Forest (QRF). Computes importance scores for predictors (Log_Discharge, MA_Discharge_3,
#Lag_Discharge, Lag_Discharge_3, Rainfall, ETo) based on feature importance analysis (Section 3.2, Table 4).
#Produces a sorted feature importance CSV and bar plot for publication.
#Author: Kindie B. Worku
#Date: 2025-07-07

import pandas as pd
import numpy as np
from quantile_forest import RandomForestQuantileRegressor
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set plot style for publication quality (Section 3.2)
sns.set_style('white')
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 18

# Define data paths and QRF parameters for each watershed
data_paths = {
    'Gilgel Abay': {
        'intermittent': Path(r"C:\Users\worku\Documents\sediment-yield-analysis\data\Intermittent_data.xlsx"),
        'n_samples': 251,
        'ssc_clip': 5.0,
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
        'intermittent': Path(r"C:\Users\worku\Documents\sediment-yield-analysis\data\Intermittent_data_gum.csv"),
        'n_samples': 245,
        'ssc_clip': 2.07,
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
    """
    Load data, perform feature engineering, train QRF model, and compute feature importance scores for the specified
    watershed.
    Args:
        intermittent_path (Path): Path to input data file (Excel or CSV).
        watershed_name (str): Name of the watershed ('Gilgel Abay' or 'Gumara').
        n_samples (int): Number of samples to process (251 for Gilgel Abay, 245 for Gumara).
        ssc_clip (float): Upper bound for clipping SSC (5.0 g/L for Gilgel Abay, 2.07 g/L for Gumara, Table 2).
        qrf_params (dict): QRF hyperparameters aligned with model performance evaluation (Section 3.1).
        is_excel (bool): Flag to indicate if input file is Excel (True) or CSV (False).
    Returns:
        Dictionary of feature importance scores or None if processing fails.
    """
    print(f"\nCalculating feature importance for {watershed_name}...")
    
    # Check if input file exists
    if not intermittent_path.exists():
        print(f"Error: File not found at {intermittent_path}")
        return None

    try:
        # Load data from specified path (Section 2.2)
        if is_excel:
            df_inter = pd.read_excel(intermittent_path, engine='openpyxl')
        else:
            df_inter = pd.read_csv(intermittent_path)
        print(f"Successfully loaded {intermittent_path} with {len(df_inter)} rows")
    except Exception as e:
        print(f"Error loading data for {watershed_name}: {str(e)}")
        return None

    # Validate required columns
    required_cols = ['Rainfall', 'Discharge', 'Temperature', 'ETo', 'SSC']
    missing_cols = [col for col in required_cols if col not in df_inter.columns]
    if missing_cols:
        print(f"{watershed_name} missing columns: {missing_cols}")
        return None

    # Display raw data sample for debugging
    print(f"Raw data sample for {watershed_name}:\n{df_inter.head(10)}")

    # Convert columns to numeric, coercing errors to NaN
    numeric_cols = ['Rainfall', 'Discharge', 'Temperature', 'ETo', 'SSC']
    for col in numeric_cols:
        df_inter[col] = pd.to_numeric(df_inter[col], errors='coerce')
    
    # Check for NaN values
    nan_counts = df_inter[numeric_cols].isna().sum()
    print(f"NaN counts before dropping for {watershed_name}:\n{nan_counts}")
    
    # Drop rows with NaN values in required columns
    df_inter = df_inter.dropna(subset=numeric_cols)
    print(f"Rows after dropping NaNs: {len(df_inter)}")

    # Warn if sample size is insufficient
    if len(df_inter) < n_samples:
        print(f"Warning: {watershed_name} has only {len(df_inter)} samples, expected {n_samples}")
    df_inter = df_inter.head(n_samples)
    print(f"Final sample count for {watershed_name}: {len(df_inter)}")

    # Check if data is empty after cleaning
    if df_inter.empty:
        print(f"{watershed_name} data empty after cleaning.")
        return None

    # Clip SSC to realistic ranges (Table 2)
    df_inter['SSC'] = df_inter['SSC'].clip(lower=0.01, upper=ssc_clip)

    # Feature engineering (Section 2.3)
    df_inter['Log_Discharge'] = np.log1p(df_inter['Discharge'].clip(lower=0))  # Log-transform discharge to reduce skewness
    df_inter['MA_Discharge_3'] = df_inter['Discharge'].rolling(window=3, min_periods=1).mean().bfill()  # 3-day moving average
    df_inter['Lag_Discharge'] = df_inter['Discharge'].shift(1).bfill()  # 1-day lag for temporal dependency
    df_inter['Lag_Discharge_3'] = df_inter['Discharge'].shift(3).bfill()  # 3-day lag for extended temporal dependency

    # Select predictors based on feature importance analysis (Section 3.2, Table 4)
    predictors = ['Log_Discharge', 'MA_Discharge_3', 'Lag_Discharge', 'Lag_Discharge_3', 'Rainfall', 'ETo']
    X_inter = df_inter[predictors]
    y_inter = df_inter['SSC']

    # Compute and display feature correlations
    print(f"\nFeature correlations for {watershed_name}:\n{X_inter.corr()}\n")
    plt.figure(figsize=(10, 6))
    sns.heatmap(X_inter.corr(), annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0)
    plt.title(f'Correlation Matrix - {watershed_name}', fontsize=20)
    plt.tight_layout()
    corr_output = OUTPUT_DIR / f"correlation_matrix_{watershed_name.lower().replace(' ', '_')}.png"
    plt.savefig(corr_output, dpi=300, transparent=True, bbox_inches='tight')
    plt.close()
    print(f"Correlation matrix saved to {corr_output}")

    # Scale features for QRF model
    scaler = StandardScaler()
    X_inter_scaled = scaler.fit_transform(X_inter)

    try:
        # Train QRF model with specified parameters
        qrf = RandomForestQuantileRegressor(**qrf_params)
        qrf.fit(X_inter_scaled, y_inter)
    except Exception as e:
        print(f"Error training QRF for {watershed_name}: {str(e)}")
        return None

    # Extract feature importance scores
    importances = qrf.feature_importances_
    importance_dict = dict(zip(predictors, importances))

    # Display feature importance scores
    print(f"{watershed_name} Feature Importances:")
    for feature, importance in importance_dict.items():
        print(f"{feature}: {importance:.6f}")
    print(f"Sum of importances: {sum(importances):.6f}")

    return importance_dict

# Define output directory
BASE_DIR = Path(r"C:\Users\acer\Documents\sediment-yield-analysis")
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

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

# Debug: Display feature importances dictionary
print("\nDebug: Feature Importances Dictionary:")
print(feature_importances)

# Generate combined feature importance table and plot
if len(feature_importances) == 2:
    predictors = ['Log_Discharge', 'MA_Discharge_3', 'Lag_Discharge', 'Lag_Discharge_3', 'Rainfall', 'ETo']
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

    # Save sorted feature importance table
    csv_output = OUTPUT_DIR / "feature_importances_qrf_fig6.csv"
    df.to_csv(csv_output, index=False)
    print(f"\nFeature Importance DataFrame (Sorted):\n{df}")
    print(f"Feature importances saved to {csv_output}")

    # Prepare data for bar plot
    df_melted = pd.melt(df, id_vars=['Feature'], value_vars=['Gilgel Abay', 'Gumara'],
                        var_name='Watershed', value_name='Importance')
    print(f"\nDebug: Melted DataFrame for Plotting:\n{df_melted}")

    # Generate bar plot for feature importance (Figure 6)
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
    
    # Configure plot aesthetics
    ax.set_xlabel('Feature', fontsize=18)
    ax.set_ylabel('Feature Importance', fontsize=18)
    ax.tick_params(axis='x', rotation=45, labelsize=16)
    ax.tick_params(axis='y', labelsize=16)
    ax.legend(title='Basin', fontsize=16, title_fontsize=16, loc='upper right')
    ax.set_ylim(0, max(df_melted['Importance']) + 0.1)
    plt.tight_layout()

    # Save bar plot
    plot_output = OUTPUT_DIR / "Figure6_feature_importance_qrf.png"
    plt.savefig(plot_output, dpi=300, transparent=True, bbox_inches='tight')
    plt.close()
    print(f"Figure 6 saved to {plot_output}")
else:
    print("Failed to calculate feature importances for both watersheds. Available data:")
    for watershed, importances in feature_importances.items():
        print(f"{watershed}: {importances}")
