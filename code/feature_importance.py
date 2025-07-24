# Script to compute QRF feature importance for SSC prediction in Gilgel Abay and Gumara watersheds (Section 3.2, Figure 6, Table 4).
# Runs after QRF is selected as the best model from model_selection.py.
# Uses best hyperparameters from model_selection.py, no scaling/outlier removal.
# Author: Kindie B. Worku
# Date: 2025-07-19

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from quantile_forest import RandomForestQuantileRegressor
from sklearn.model_selection import train_test_split
from pathlib import Path
import warnings

# Enable inline plotting for Jupyter Notebook
%matplotlib inline

# Suppress warnings
warnings.filterwarnings('ignore')

# Set plot style
sns.set_style('white')
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 14

# Directories
BASE_DIR = Path(r"C:\Users\worku\Documents\sediment-yield-analysis")
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

# Function to load best model parameters (unchanged)
def load_best_model_params(watershed_name, test_size=0.3):
    try:
        results_path = OUTPUT_DIR / f"model_performance_{watershed_name.lower().replace(' ', '_')}_{int(test_size*100)}split.csv"
        params_path = OUTPUT_DIR / f"best_params_{watershed_name.lower().replace(' ', '_')}_{int(test_size*100)}split.csv"
        
        if not results_path.exists() or not params_path.exists():
            print(f"Error: Required files not found for {watershed_name} ({results_path}, {params_path})")
            return None, None
        
        results_df = pd.read_csv(results_path)
        params_df = pd.read_csv(params_path)
        
        best_model = results_df.loc[results_df['Validation R²'].idxmax()]['Model']
        if best_model != 'QRF':
            print(f"QRF not selected for {watershed_name} (Best model: {best_model}). Skipping feature importance.")
            return best_model, None
        
        qrf_params = params_df[params_df['Model'] == 'QRF']['Parameters'].iloc[0]
        import ast
        qrf_params = ast.literal_eval(qrf_params)
        print(f"QRF selected for {watershed_name} with params: {qrf_params}")
        return best_model, qrf_params
    
    except Exception as e:
        print(f"Error loading best model for {watershed_name}: {str(e)}")
        return None, None

# Function to compute QRF feature importance (unchanged)
def compute_feature_importance(data_path, watershed_name, n_samples, qrf_params, test_size=0.3):
    if qrf_params is None:
        print(f"Skipping feature importance for {watershed_name}: QRF not selected.")
        return None
    
    try:
        df = pd.read_csv(data_path)
        required_columns = ['Rainfall', 'Discharge', 'Temperature', 'ETo', 'SSC', 'Date', 'Annual_Rainfall',
                           'Cumulative_Rainfall', 'Lag_Discharge_1', 'Lag_Discharge_3', 'Sin_Julian', 'Cos_Julian']
        if not all(col in df.columns for col in required_columns):
            raise ValueError(f"{watershed_name} data must contain: {', '.join(required_columns)}")
        
        df_filtered = df.dropna(subset=['SSC']).head(n_samples)
        print(f"Sample count for {watershed_name}: {len(df_filtered)}")
        if len(df_filtered) < 50:
            raise ValueError(f"Insufficient samples for {watershed_name}: {len(df_filtered)}")
        
        predictors = ['Discharge', 'MA_Discharge_3', 'Lag_Discharge_1', 'Lag_Discharge_3', 'Rainfall', 'ETo',
                      'Temperature', 'Annual_Rainfall', 'Cumulative_Rainfall', 'Sin_Julian', 'Cos_Julian']
        target = 'SSC'
        X = df_filtered[predictors]
        y = df_filtered[target]
        
        discharge_bins = pd.qcut(X['Discharge'], q=7, duplicates='drop')
        X_train, _, y_train, _ = train_test_split(
            X, y, test_size=test_size, stratify=discharge_bins, random_state=42
        )
        
        print(f"Training QRF for {watershed_name} with params: {qrf_params}")
        qrf = RandomForestQuantileRegressor(**qrf_params, random_state=42)
        qrf.fit(X_train, y_train)
        
        feature_importance = pd.DataFrame({
            'Feature': predictors,
            'Importance': qrf.feature_importances_
        }).sort_values(by='Importance', ascending=False)
        feature_importance.to_csv(OUTPUT_DIR / f"feature_importance_{watershed_name.lower().replace(' ', '_')}_70split.csv", index=False)
        print(f"Feature importance saved for {watershed_name}")
        
        return feature_importance
    
    except Exception as e:
        print(f"Error computing feature importance for {watershed_name}: {str(e)}")
        return None

# Process watersheds
data_paths = {
    'Gilgel Abay': {'data': OUTPUT_DIR / "gilgel_abay_features.csv", 'n_samples': 251},
    'Gumara': {'data': OUTPUT_DIR / "gumara_features.csv", 'n_samples': 245}
}
feature_importances = {}

for watershed_name, params in data_paths.items():
    best_model, qrf_params = load_best_model_params(watershed_name, test_size=0.3)
    if best_model == 'QRF':
        fi = compute_feature_importance(
            params['data'], watershed_name, params['n_samples'], qrf_params, test_size=0.3
        )
        if fi is not None:
            feature_importances[watershed_name] = fi
    else:
        print(f"Skipping {watershed_name}: Best model is {best_model}, not QRF.")

# Generate Figure 6 (Feature Importance) - Modified Section
if feature_importances:
    combined_feature_importance = pd.concat(
        [fi.assign(Watershed=w) for w, fi in feature_importances.items()],
        ignore_index=True
    )
    combined_feature_importance.to_csv(OUTPUT_DIR / "feature_importance_comparison_70split.csv", index=False)
    
    # Prepare data for plotting
    features = combined_feature_importance['Feature'].unique()
    n_features = len(features)
    bar_width = 0.35
    index = np.arange(n_features)
    
    # Create figure
    plt.figure(figsize=(10, 6))
    
    # Plot bars for each watershed with color and hatching
    gilgel_data = combined_feature_importance[combined_feature_importance['Watershed'] == 'Gilgel Abay']
    gumara_data = combined_feature_importance[combined_feature_importance['Watershed'] == 'Gumara']
    
    plt.bar(index - bar_width/2, gilgel_data['Importance'], bar_width, 
            color='#1f77b4', hatch='', label='Gilgel Abay')  # Solid for Gilgel Abay
    plt.bar(index + bar_width/2, gumara_data['Importance'], bar_width, 
            color='#2ca02c', hatch='///', label='Gumara')  # Hatched for Gumara
    
    # Add value labels on top of bars
    for i, (gilgel_val, gumara_val) in enumerate(zip(gilgel_data['Importance'], gumara_data['Importance'])):
        if gilgel_val > 0:
            plt.text(i - bar_width/2, gilgel_val + 0.01, f'{gilgel_val:.2f}', ha='center', va='bottom', fontsize=12)
        if gumara_val > 0:
            plt.text(i + bar_width/2, gumara_val + 0.01, f'{gumara_val:.2f}', ha='center', va='bottom', fontsize=12)
    
    plt.xlabel('Feature', fontsize=20)
    plt.ylabel('Feature Importance (%)', fontsize=20)
    plt.xticks(index, features, rotation=45, ha='right', fontsize=16)
    plt.yticks(fontsize=16)
    plt.legend(title='Watershed', fontsize=16, title_fontsize=16)
    plt.ylim(0, max(combined_feature_importance['Importance']) + 0.15)
    plt.tight_layout()
    
    # Save color version for online
    plt.savefig(OUTPUT_DIR / "Figure6_feature_importance_qrf_70split_color.png", dpi=600, transparent=True, bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / "Figure6_feature_importance_qrf_70split_color.svg", format='svg', transparent=True, bbox_inches='tight')
    
    # Save grayscale version for print
    plt.clf()  # Clear the figure
    plt.figure(figsize=(10, 6))
    plt.bar(index - bar_width/2, gilgel_data['Importance'], bar_width, 
            color='#000000', hatch='', label='Gilgel Abay')  # Black for Gilgel Abay
    plt.bar(index + bar_width/2, gumara_data['Importance'], bar_width, 
            color='#666666', hatch='///', label='Gumara')  # Gray for Gumara
    for i, (gilgel_val, gumara_val) in enumerate(zip(gilgel_data['Importance'], gumara_data['Importance'])):
        if gilgel_val > 0:
            plt.text(i - bar_width/2, gilgel_val + 0.01, f'{gilgel_val:.2f}', ha='center', va='bottom', fontsize=12)
        if gumara_val > 0:
            plt.text(i + bar_width/2, gumara_val + 0.01, f'{gumara_val:.2f}', ha='center', va='bottom', fontsize=12)
    plt.xlabel('Feature', fontsize=20)
    plt.ylabel('Feature Importance (%)', fontsize=20)
    plt.xticks(index, features, rotation=45, ha='right', fontsize=16)
    plt.yticks(fontsize=16)
    plt.legend(title='Watershed', fontsize=16, title_fontsize=16)
    plt.ylim(0, max(combined_feature_importance['Importance']) + 0.15)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "Figure6_feature_importance_qrf_70split_grayscale.png", dpi=600, transparent=True, bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / "Figure6_feature_importance_qrf_70split_grayscale.eps", format='eps', dpi=600, transparent=True, bbox_inches='tight')
    
    plt.show()
    plt.close()
else:
    print("No feature importance calculated: QRF not selected for any watershed.")
