# Script to generate Figure 4: Model performance evaluation for suspended sediment concentration (SSC) prediction.
# Compares Segmented Rating Curve (SRC), Gradient Boosting (GB), Random Forest (RF), and Quantile Random Forest (QRF)
# models for SSC prediction in Gilgel Abay and Gumara watersheds, producing scatter plots of observed vs. predicted SSC
# (Section 3.1 of the research paper). Uses engineered predictors (Discharge, MA_Discharge_3, Lag_Discharge_1,
# Lag_Discharge_3, Rainfall, ETo, Temperature, Annual_Rainfall, Cumulative_Rainfall) based on feature importance
# (Section 3.2). Outputs are saved in PNG and SVG formats for publication.
# Author: Kindie B. Worku
# Date: 2025-07-16

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from quantile_forest import RandomForestQuantileRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV, KFold
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import warnings

# Suppress warnings to ensure clean console output for debugging
warnings.filterwarnings('ignore')

# Define custom metric for model evaluation
def mean_absolute_percentage_error(y_true, y_pred):
    """
    Calculate Mean Absolute Percentage Error (MAPE), clipping y_true to avoid division by zero.
    Used to evaluate model performance alongside R², RMSE, and MAE (Section 3.1).
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    y_true = np.clip(y_true, 0.01, None)  # Avoid division by zero
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

# Define function to process and evaluate models for a given watershed
def process_watershed(data_path, watershed_name, n_samples, is_excel=False):
    """
    Process watershed data, perform feature engineering, evaluate SRC, GB, RF, and QRF models, and return predictions
    and metrics.
    Args:
        data_path (Path): Path to input data file (Excel or CSV).
        watershed_name (str): Name of the watershed ('Gilgel Abay' or 'Gumara').
        n_samples (int): Number of samples to process (251 for Gilgel Abay, 245 for Gumara).
        is_excel (bool): Flag to indicate if input file is Excel (True) or CSV (False).
    Returns:
        Tuple containing observed SSC, predictions, R² scores, and results DataFrame.
    """
    try:
        # Load data from specified path (Section 2.2)
        if is_excel:
            df = pd.read_excel(data_path)
        else:
            df = pd.read_csv(data_path)
        
        # Validate required columns to ensure data integrity
        required_columns = ['Rainfall', 'Discharge', 'Temperature', 'ETo', 'SSC']
        if not all(col in df.columns for col in required_columns):
            raise ValueError(f"{watershed_name} data must contain: {', '.join(required_columns)}")
        
        # Filter non-missing SSC values and limit to specified sample size (Section 2.3)
        df_filtered = df.dropna(subset=['SSC']).head(n_samples)
        print(f"Sample count for {watershed_name}: {len(df_filtered)}")
        
        # Compute Annual_Rainfall and Cumulative_Rainfall
        if 'Date' in df_filtered.columns:
            df_filtered['Date'] = pd.to_datetime(df_filtered['Date'], errors='coerce')
            if df_filtered['Date'].isna().any():
                raise ValueError(f"{watershed_name} data contains invalid or missing Date values")
            df_filtered['Annual_Rainfall'] = df_filtered.groupby(df_filtered['Date'].dt.year)['Rainfall'].transform('sum')
            df_filtered['Cumulative_Rainfall'] = df_filtered['Rainfall'].rolling(window=30, min_periods=1).sum().bfill()
        else:
            print(f"Warning: No Date column in {watershed_name}. Computing Annual_Rainfall and Cumulative_Rainfall without dates.")
            # Assume daily data, 365 days per year
            df_filtered['Year'] = (df_filtered.index // 365) + 1990  # Approximate years from 1990
            df_filtered['Annual_Rainfall'] = df_filtered.groupby('Year')['Rainfall'].transform('sum')
            df_filtered['Cumulative_Rainfall'] = df_filtered['Rainfall'].rolling(window=30, min_periods=1).sum().bfill()
        
        # Feature engineering to enhance model performance (Section 2.3)
        df_filtered['MA_Discharge_3'] = df_filtered['Discharge'].rolling(window=3, min_periods=1).mean()  # 3-day moving average
        df_filtered['Lag_Discharge_1'] = df_filtered['Discharge'].shift(1).bfill()  # 1-day lag for temporal dependency
        df_filtered['Lag_Discharge_3'] = df_filtered['Discharge'].shift(3).bfill()  # 3-day lag for extended temporal dependency
        
        # Select predictors based on feature importance analysis (Section 3.2, Table 4)
        predictors = ['Discharge', 'MA_Discharge_3', 'Lag_Discharge_1', 'Lag_Discharge_3', 'Rainfall', 'ETo', 'Temperature', 'Annual_Rainfall', 'Cumulative_Rainfall']
        target = 'SSC'
        X = df_filtered[predictors]
        y = df_filtered[target]
        
        # Scale features to standardize input for machine learning models
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=predictors, index=X.index)
        
        # Split data into training (80%) and testing (20%) sets (Section 2.3)
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
        X_train = X_train.reset_index(drop=True)
        X_test = X_test.reset_index(drop=True)
        y_train = y_train.reset_index(drop=True)
        y_test = y_test.reset_index(drop=True)
        
        # Reconstruct discharge for SRC model from scaled data
        X_train_raw = pd.DataFrame(scaler.inverse_transform(X_train), columns=predictors)
        X_test_raw = pd.DataFrame(scaler.inverse_transform(X_test), columns=predictors)
        
        # Initialize results storage for model metrics
        results = {'Model': [], 'R²': [], 'Validation R²': [], 'RMSE': [], 'MAE': [], 'MAPE': [], 'Watershed': []}
        
        # SRC: Segmented power-law model based on discharge threshold
        median_discharge = df_filtered['Discharge'].median()
        low_flow_train = X_train_raw['Discharge'] < median_discharge
        high_flow_train = X_train_raw['Discharge'] >= median_discharge
        
        src_model_low = None
        src_model_high = None
        if low_flow_train.any():
            X_src_low = np.log(X_train_raw.loc[low_flow_train, 'Discharge'].values + 1e-6).reshape(-1, 1)
            y_src_low = np.log(y_train[low_flow_train].values + 1e-6)
            src_model_low = LinearRegression().fit(X_src_low, y_src_low)
        
        if high_flow_train.any():
            X_src_high = np.log(X_train_raw.loc[high_flow_train, 'Discharge'].values + 1e-6).reshape(-1, 1)
            y_src_high = np.log(y_train[high_flow_train].values + 1e-6)
            src_model_high = LinearRegression().fit(X_src_high, y_src_high)
        
        # Generate SRC predictions for full dataset and test set
        low_flow_full = df_filtered['Discharge'] < median_discharge
        high_flow_full = df_filtered['Discharge'] >= median_discharge
        src_pred = np.zeros(len(y))
        if src_model_low and low_flow_full.any():
            X_src_low_full = np.log(df_filtered.loc[low_flow_full, 'Discharge'].values + 1e-6).reshape(-1, 1)
            src_pred[low_flow_full] = np.exp(src_model_low.predict(X_src_low_full))
        if src_model_high and high_flow_full.any():
            X_src_high_full = np.log(df_filtered.loc[high_flow_full, 'Discharge'].values + 1e-6).reshape(-1, 1)
            src_pred[high_flow_full] = np.exp(src_model_high.predict(X_src_high_full))
        
        low_flow_test = X_test_raw['Discharge'] < median_discharge
        high_flow_test = X_test_raw['Discharge'] >= median_discharge
        src_test_pred = np.zeros(len(y_test))
        if src_model_low and low_flow_test.any():
            X_src_low_test = np.log(X_test_raw.loc[low_flow_test, 'Discharge'].values + 1e-6).reshape(-1, 1)
            src_test_pred[low_flow_test] = np.exp(src_model_low.predict(X_src_low_test))
        if src_model_high and high_flow_test.any():
            X_src_high_test = np.log(X_test_raw.loc[high_flow_test, 'Discharge'].values + 1e-6).reshape(-1, 1)
            src_test_pred[high_flow_test] = np.exp(src_model_high.predict(X_src_high_test))
        
        # Compute SRC performance metrics
        src_r2 = r2_score(y, src_pred)
        src_val_r2 = r2_score(y_test, src_test_pred)
        src_rmse = np.sqrt(mean_squared_error(y, src_pred))
        src_mae = mean_absolute_error(y, src_pred)
        src_mape = mean_absolute_percentage_error(y, src_pred)
        
        results['Model'].append('SRC')
        results['R²'].append(src_r2)
        results['Validation R²'].append(src_val_r2)
        results['RMSE'].append(src_rmse)
        results['MAE'].append(src_mae)
        results['MAPE'].append(src_mape)
        results['Watershed'].append(watershed_name)
        print(f"SRC Performance ({watershed_name}): R²={src_r2:.3f}, Validation R²={src_val_r2:.3f}, "
              f"RMSE={src_rmse:.3f}, MAE={src_mae:.3f}, MAPE={src_mape:.3f}%")
        
        # GB: Gradient Boosting with hyperparameter tuning
        gb_param_grid = {
            'n_estimators': [500, 1000],
            'max_depth': [3, 5],
            'learning_rate': [0.01, 0.05],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [2, 5],
            'subsample': [0.7, 0.9]
        }
        gb = GradientBoostingRegressor(random_state=42, n_iter_no_change=10, validation_fraction=0.1)
        gb_search = RandomizedSearchCV(gb, gb_param_grid, n_iter=15, cv=KFold(n_splits=5, shuffle=True, random_state=42),
                                      scoring='r2', n_jobs=-1, random_state=42)
        gb_search.fit(X_train, y_train)
        best_gb = gb_search.best_estimator_
        
        gb_pred = best_gb.predict(X_scaled)
        gb_test_pred = best_gb.predict(X_test)
        gb_r2 = r2_score(y, gb_pred)
        gb_val_r2 = r2_score(y_test, gb_test_pred)
        gb_rmse = np.sqrt(mean_squared_error(y, gb_pred))
        gb_mae = mean_absolute_error(y, gb_pred)
        gb_mape = mean_absolute_percentage_error(y, gb_pred)
        
        results['Model'].append('GB')
        results['R²'].append(gb_r2)
        results['Validation R²'].append(gb_val_r2)
        results['RMSE'].append(gb_rmse)
        results['MAE'].append(gb_mae)
        results['MAPE'].append(gb_mape)
        results['Watershed'].append(watershed_name)
        print(f"GB Performance ({watershed_name}): R²={gb_r2:.3f}, Validation R²={gb_val_r2:.3f}, "
              f"RMSE={gb_rmse:.3f}, MAE={gb_mae:.3f}, MAPE={gb_mape:.3f}%")
        
        # RF: Random Forest with hyperparameter tuning
        rf_param_grid = {
            'n_estimators': [500, 1000],
            'max_depth': [10, 20],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [2, 5],
            'max_features': ['sqrt', 'log2']
        }
        rf = RandomForestRegressor(random_state=42)
        rf_search = RandomizedSearchCV(rf, rf_param_grid, n_iter=15, cv=KFold(n_splits=5, shuffle=True, random_state=42),
                                      scoring='r2', n_jobs=-1, random_state=42)
        rf_search.fit(X_train, y_train)
        best_rf = rf_search.best_estimator_
        
        rf_pred = best_rf.predict(X_scaled)
        rf_test_pred = best_rf.predict(X_test)
        rf_r2 = r2_score(y, rf_pred)
        rf_val_r2 = r2_score(y_test, rf_test_pred)
        rf_rmse = np.sqrt(mean_squared_error(y, rf_pred))
        rf_mae = mean_absolute_error(y, rf_pred)
        rf_mape = mean_absolute_percentage_error(y, rf_pred)
        
        results['Model'].append('RF')
        results['R²'].append(rf_r2)
        results['Validation R²'].append(rf_val_r2)
        results['RMSE'].append(rf_rmse)
        results['MAE'].append(rf_mae)
        results['MAPE'].append(rf_mape)
        results['Watershed'].append(watershed_name)
        print(f"RF Performance ({watershed_name}): R²={rf_r2:.3f}, Validation R²={rf_val_r2:.3f}, "
              f"RMSE={rf_rmse:.3f}, MAE={rf_mae:.3f}, MAPE={rf_mape:.3f}%")
        
        # QRF: Quantile Random Forest with hyperparameter tuning (selected model, Section 3.1)
        qrf_param_grid = {
            'n_estimators': [500, 1000, 2000],
            'max_depth': [10, 20, 30],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2, 3],
            'max_features': ['sqrt', 0.5, 0.8]
        }
        qrf = RandomForestQuantileRegressor(random_state=42)
        qrf_search = RandomizedSearchCV(qrf, qrf_param_grid, n_iter=20, cv=KFold(n_splits=5, shuffle=True, random_state=42),
                                       scoring='r2', n_jobs=-1, random_state=42)
        qrf_search.fit(X_train, y_train)
        best_qrf = qrf_search.best_estimator_
        
        # Predict median (quantile=0.5) to match mean predictions of other models
        quantiles = [0.5]
        qrf_pred = best_qrf.predict(X_scaled, quantiles=quantiles).flatten()
        qrf_test_pred = best_qrf.predict(X_test, quantiles=quantiles).flatten()
        qrf_r2 = r2_score(y, qrf_pred)
        qrf_val_r2 = r2_score(y_test, qrf_test_pred)
        qrf_rmse = np.sqrt(mean_squared_error(y, qrf_pred))
        qrf_mae = mean_absolute_error(y, qrf_pred)
        qrf_mape = mean_absolute_percentage_error(y, qrf_pred)
        
        results['Model'].append('QRF')
        results['R²'].append(qrf_r2)
        results['Validation R²'].append(qrf_val_r2)
        results['RMSE'].append(qrf_rmse)
        results['MAE'].append(qrf_mae)
        results['MAPE'].append(qrf_mape)
        results['Watershed'].append(watershed_name)
        print(f"QRF Performance ({watershed_name}): R²={qrf_r2:.3f}, Validation R²={qrf_val_r2:.3f}, "
              f"RMSE={qrf_rmse:.3f}, MAE={qrf_mae:.3f}, MAPE={qrf_mape:.3f}%")
        
        # Save results to CSV for transparency and reproducibility
        results_df = pd.DataFrame(results)
        results_df.to_csv(OUTPUT_DIR / f"model_performance_{watershed_name.lower().replace(' ', '_')}.csv", index=False)
        print(f"\nModel Performance Summary for {watershed_name}:\n{results_df}")
        
        return y, src_pred, src_r2, gb_pred, gb_r2, rf_pred, rf_r2, qrf_pred, qrf_r2, results_df
    
    except Exception as e:
        print(f"Error processing {watershed_name}: {str(e)}")
        return None

# Define base directories for input data and output figures
BASE_DIR = Path(r"C:\Users\worku\Documents\sediment-yield-analysis")
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

# Configure plot aesthetics for publication quality (Section 3.1)
sns.set_style('white')
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 14

# Process data for both watersheds
gilgel_path = DATA_DIR / "Intermittent_data.xlsx"
gumara_path = DATA_DIR / "Intermittent_data_gum.csv"
gilgel_data = process_watershed(gilgel_path, "Gilgel Abay", n_samples=251, is_excel=True)
gumara_data = process_watershed(gumara_path, "Gumara", n_samples=245)

# Check for processing errors
if gilgel_data is None or gumara_data is None:
    raise SystemExit("Processing failed for one or more watersheds.")

# Unpack results for plotting
y_gilgel, src_pred_gilgel, src_r2_gilgel, gb_pred_gilgel, gb_r2_gilgel, rf_pred_gilgel, rf_r2_gilgel, \
qrf_pred_gilgel, qrf_r2_gilgel, results_gilgel = gilgel_data

y_gumara, src_pred_gumara, src_r2_gumara, gb_pred_gumara, gb_r2_gumara, rf_pred_gumara, rf_r2_gumara, \
qrf_pred_gumara, qrf_r2_gumara, results_gumara = gumara_data

# Combine results for both watersheds and save
combined_results = pd.concat([results_gilgel, results_gumara], ignore_index=True)
csv_output = OUTPUT_DIR / "model_performance_comparison.csv"
combined_results.to_csv(csv_output, index=False)
print(f"\nCombined Model Performance Across Watersheds:\n{combined_results}")
print(f"Combined results saved to {csv_output}")

# Generate Figure 4: Scatter plots of observed vs. predicted SSC
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5), sharey=True)
fig.subplots_adjust(wspace=0.1)

# Plot for Gilgel Abay (subplot a)
ax1.scatter(y_gilgel, src_pred_gilgel, color='#1f77b4', marker='s', alpha=0.3, s=40, label=f'SRC (R² = {src_r2_gilgel:.2f})')
ax1.scatter(y_gilgel, gb_pred_gilgel, color='#800080', marker='^', alpha=0.3, s=40, label=f'GB (R² = {gb_r2_gilgel:.2f})')
ax1.scatter(y_gilgel, rf_pred_gilgel, color='#2ca02c', marker='d', alpha=0.3, s=40, label=f'RF (R² = {rf_r2_gilgel:.2f})')
ax1.scatter(y_gilgel, qrf_pred_gilgel, color='#ff7f0e', marker='o', alpha=0.6, s=50, edgecolor='black',
           label=f'QRF (R² = {qrf_r2_gilgel:.2f})')
max_val_gilgel = max(y_gilgel.max(), src_pred_gilgel.max(), gb_pred_gilgel.max(), rf_pred_gilgel.max(), qrf_pred_gilgel.max())
ax1.plot([0, max_val_gilgel], [0, max_val_gilgel], color='black', linestyle='--', label='1:1 Line')
ax1.set_xlabel('Observed SSC (g/L)', fontsize=18)
ax1.set_ylabel('Predicted SSC (g/L)', fontsize=18)
ax1.set_title('(a) Gilgel Abay', fontsize=20)
ax1.grid(False)
ax1.legend(loc='upper left', fontsize=14, frameon=True, edgecolor='lightgray', framealpha=0.8)
ax1.tick_params(axis='both', which='major', labelsize=16)

# Plot for Gumara (subplot b)
ax2.scatter(y_gumara, src_pred_gumara, color='#1f77b4', marker='s', alpha=0.3, s=40, label=f'SRC (R² = {src_r2_gumara:.2f})')
ax2.scatter(y_gumara, gb_pred_gumara, color='#800080', marker='^', alpha=0.3, s=40, label=f'GB (R² = {gb_r2_gumara:.2f})')
ax2.scatter(y_gumara, rf_pred_gumara, color='#2ca02c', marker='d', alpha=0.3, s=40, label=f'RF (R² = {rf_r2_gumara:.2f})')
ax2.scatter(y_gumara, qrf_pred_gumara, color='#ff7f0e', marker='o', alpha=0.6, s=50, edgecolor='black',
           label=f'QRF (R² = {qrf_r2_gumara:.2f})')
max_val_gumara = max(y_gumara.max(), src_pred_gumara.max(), gb_pred_gumara.max(), rf_pred_gumara.max(), qrf_pred_gumara.max())
ax2.plot([0, max_val_gumara], [0, max_val_gumara], color='black', linestyle='--', label='1:1 Line')
ax2.set_xlabel('Observed SSC (g/L)', fontsize=18)
ax2.set_title('(b) Gumara', fontsize=20)
ax2.grid(False)
ax2.legend(loc='upper left', fontsize=14, frameon=True, edgecolor='lightgray', framealpha=0.8)
ax2.tick_params(axis='both', which='major', labelsize=16)

# Save figure in PNG and SVG formats for publication
plt.tight_layout()
output_png = OUTPUT_DIR / "Figure4_observed_vs_predicted.png"
output_svg = OUTPUT_DIR / "Figure4_observed_vs_predicted.svg"
plt.savefig(output_png, transparent=True, dpi=600, bbox_inches='tight')
plt.savefig(output_svg, transparent=True, format='svg', bbox_inches='tight')
plt.close()
print(f"Figure 4 saved to {output_png} (PNG) and {output_svg} (SVG)")
