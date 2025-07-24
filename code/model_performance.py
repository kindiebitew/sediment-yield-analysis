# Script to evaluate model performance for SSC prediction in Gilgel Abay and Gumara watersheds.
# Uses uniform hyperparameter tuning for GB, RF, QRF, includes ETo, Sin_Julian, Cos_Julian, uses 70%/30% split.
# Processes all models for both watersheds, includes NaN checks, and fixes 1:1 line issue for Gilgel Abay.
# Author: Kindie B. Worku
# Date: 2025-07-19

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from quantile_forest import RandomForestQuantileRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV, KFold
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
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

# Custom metric
def mean_absolute_percentage_error(y_true, y_pred):
    """Calculate MAPE, clipping y_true to avoid division by zero."""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    y_true = np.clip(y_true, 0.01, None)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

# Composite score for reporting
def composite_score(r2, rmse, mae, mape, weights={'r2': 0.8, 'rmse': 0.1, 'mae': 0.05, 'mape': 0.05}):
    """Calculate a composite score for reporting."""
    r2_norm = r2
    rmse_norm = 1 / (1 + rmse)
    mae_norm = 1 / (1 + mae)
    mape_norm = 1 / (1 + mape / 100)
    score = (weights['r2'] * r2_norm + weights['rmse'] * rmse_norm + 
             weights['mae'] * mae_norm + weights['mape'] * mape_norm)
    return score

# Process and evaluate models
def process_watershed(data_path, watershed_name, n_samples, test_size=0.3, qrf_only=False):
    try:
        # Load data
        df = pd.read_csv(data_path)
        required_columns = ['Rainfall', 'Discharge', 'Temperature', 'ETo', 'SSC', 'Date', 'Annual_Rainfall',
                           'Cumulative_Rainfall', 'Lag_Discharge_1', 'Lag_Discharge_3', 'Sin_Julian', 'Cos_Julian']
        if not all(col in df.columns for col in required_columns):
            raise ValueError(f"{watershed_name} data must contain: {', '.join(required_columns)}")
        
        # Filter and limit samples
        df_filtered = df.dropna(subset=['SSC']).head(n_samples)

        print(f"Sample count for {watershed_name}: {len(df_filtered)}")
        if len(df_filtered) < 50:
            raise ValueError(f"Insufficient samples for {watershed_name}: {len(df_filtered)}")
        
        # Predictors and target
        predictors = ['Discharge', 'MA_Discharge_3', 'Lag_Discharge_1', 'Lag_Discharge_3', 'Rainfall',
                      'Temperature', 'Annual_Rainfall', 'Cumulative_Rainfall', 'ETo', 'Sin_Julian', 'Cos_Julian']
        target = 'SSC'
        X = df_filtered[predictors]
        y = df_filtered[target]
        
        # Split data
        discharge_bins = pd.qcut(X['Discharge'], q=7, duplicates='drop')
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, stratify=discharge_bins, random_state=42
        )
        print(f"Training set size ({watershed_name}): {len(X_train)} samples")
        print(f"Test set size ({watershed_name}): {len(X_test)} samples")
        
        # Results storage
        results = {'Model': [], 'R²': [], 'Validation R²': [], 'RMSE': [], 'MAE': [], 'MAPE': [], 'Composite Score': [], 'Watershed': []}
        best_params = {'Model': [], 'Parameters': [], 'Watershed': []}
        predictions = {'Model': [], 'Predictions': [], 'Test Predictions': [], 'Watershed': []}
        
        if not qrf_only:
            # SRC: SSC = a*Q^b
            X_src = np.log(X['Discharge'].values + 1e-8).reshape(-1, 1)
            y_src = np.log(y.values + 1e-8)
            X_test_src = np.log(X_test['Discharge'].values + 1e-8).reshape(-1, 1)
            src_model = LinearRegression().fit(X_src, y_src)
            src_pred = np.clip(np.exp(src_model.predict(X_src)), 0, 100)
            src_test_pred = np.clip(np.exp(src_model.predict(X_test_src)), 0, 100)
            

            src_r2 = r2_score(y, src_pred)
            src_val_r2 = r2_score(y_test, src_test_pred)
            src_rmse = np.sqrt(mean_squared_error(y_test, src_test_pred))
            src_mae = mean_absolute_error(y_test, src_test_pred)
            src_mape = mean_absolute_percentage_error(y_test, src_test_pred)
            src_score = composite_score(src_val_r2, src_rmse, src_mae, src_mape)
            
            results['Model'].append('SRC')
            results['R²'].append(src_r2)
            results['Validation R²'].append(src_val_r2)
            results['RMSE'].append(src_rmse)
            results['MAE'].append(src_mae)
            results['MAPE'].append(src_mape)
            results['Composite Score'].append(src_score)
            results['Watershed'].append(watershed_name)
            best_params['Model'].append('SRC')
            best_params['Parameters'].append({'a': np.exp(src_model.intercept_), 'b': src_model.coef_[0]})
            best_params['Watershed'].append(watershed_name)
            predictions['Model'].append('SRC')
            predictions['Predictions'].append(src_pred.tolist())
            predictions['Test Predictions'].append(src_test_pred.tolist())
            predictions['Watershed'].append(watershed_name)
            print(f"SRC Performance ({watershed_name}): R²={src_r2:.3f}, Validation R²={src_val_r2:.3f}, RMSE={src_rmse:.3f}, MAE={src_mae:.3f}, MAPE={src_mape:.3f}, Score={src_score:.3f}")
            
            # GB
            param_grid = {
                'n_estimators': [1000, 2000, 3000, 4000, 5000],
                'max_depth': [10, 20, 30, 40, 50],
                'min_samples_split': [2, 5],
                'min_samples_leaf': [1, 3],
                'max_features': [0.5, 0.7],
                'subsample': [0.8, 1.0],
                'learning_rate': [0.05, 0.1]
            }

            gb = GradientBoostingRegressor(random_state=42, n_iter_no_change=10, tol=0.01)
            gb_search = RandomizedSearchCV(gb, param_grid, n_iter=30, cv=KFold(n_splits=5, shuffle=True, random_state=42),
                                          scoring='r2', n_jobs=-1, random_state=42)
            gb_search.fit(X_train, y_train)
            best_gb = gb_search.best_estimator_
            
            gb_pred = np.clip(best_gb.predict(X), 0, None)
            gb_test_pred = np.clip(best_gb.predict(X_test), 0, None)
            gb_r2 = r2_score(y, gb_pred)
            gb_val_r2 = r2_score(y_test, gb_test_pred)
            gb_rmse = np.sqrt(mean_squared_error(y_test, gb_test_pred))
            gb_mae = mean_absolute_error(y_test, gb_test_pred)
            gb_mape = mean_absolute_percentage_error(y_test, gb_test_pred)
            gb_score = composite_score(gb_val_r2, gb_rmse, gb_mae, gb_mape)
            
            results['Model'].append('GB')
            results['R²'].append(gb_r2)
            results['Validation R²'].append(gb_val_r2)
            results['RMSE'].append(gb_rmse)
            results['MAE'].append(gb_mae)
            results['MAPE'].append(gb_mape)
            results['Composite Score'].append(gb_score)
            results['Watershed'].append(watershed_name)
            best_params['Model'].append('GB')
            best_params['Parameters'].append(gb_search.best_params_)
            best_params['Watershed'].append(watershed_name)
            predictions['Model'].append('GB')
            predictions['Predictions'].append(gb_pred.tolist())
            predictions['Test Predictions'].append(gb_test_pred.tolist())
            predictions['Watershed'].append(watershed_name)
            print(f"GB Performance ({watershed_name}): R²={gb_r2:.3f}, Validation R²={gb_val_r2:.3f}, RMSE={gb_rmse:.3f}, MAE={gb_mae:.3f}, MAPE={gb_mape:.3f}, Score={gb_score:.3f}")
            

            # RF
            rf_param_grid = {
                'n_estimators': [1000, 2000, 3000, 4000, 5000],
                'max_depth': [10, 20, 30, 40, 50],
                'min_samples_split': [2, 5],
                'min_samples_leaf': [1, 3],
                'max_features': [0.5, 0.7],
                'max_samples': [0.8, 1.0]
            }
            rf = RandomForestRegressor(random_state=42)
            rf_search = RandomizedSearchCV(rf, rf_param_grid, n_iter=30, cv=KFold(n_splits=5, shuffle=True, random_state=42),
                                          scoring='r2', n_jobs=-1, random_state=42)
            rf_search.fit(X_train, y_train)
            best_rf = rf_search.best_estimator_
            
            rf_pred = np.clip(best_rf.predict(X), 0, None)
            rf_test_pred = np.clip(best_rf.predict(X_test), 0, None)
            rf_r2 = r2_score(y, rf_pred)
            rf_val_r2 = r2_score(y_test, rf_test_pred)
            rf_rmse = np.sqrt(mean_squared_error(y_test, rf_test_pred))
            rf_mae = mean_absolute_error(y_test, rf_test_pred)
            rf_mape = mean_absolute_percentage_error(y_test, rf_test_pred)
            rf_score = composite_score(rf_val_r2, rf_rmse, rf_mae, rf_mape)
            

            results['Model'].append('RF')
            results['R²'].append(rf_r2)
            results['Validation R²'].append(rf_val_r2)
            results['RMSE'].append(rf_rmse)
            results['MAE'].append(rf_mae)
            results['MAPE'].append(rf_mape)
            results['Composite Score'].append(rf_score)
            results['Watershed'].append(watershed_name)
            best_params['Model'].append('RF')
            best_params['Parameters'].append(rf_search.best_params_)
            best_params['Watershed'].append(watershed_name)
            predictions['Model'].append('RF')
            predictions['Predictions'].append(rf_pred.tolist())
            predictions['Test Predictions'].append(rf_test_pred.tolist())
            predictions['Watershed'].append(watershed_name)
            print(f"RF Performance ({watershed_name}): R²={rf_r2:.3f}, Validation R²={rf_val_r2:.3f}, RMSE={rf_rmse:.3f}, MAE={rf_mae:.3f}, MAPE={rf_mape:.3f}, Score={rf_score:.3f}")
        
        # QRF
        qrf_param_grid = {
            'n_estimators': [1000, 2000, 3000, 4000, 5000],
            'max_depth': [10, 20, 30, 40, 50],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 3],
            'max_features': [0.5, 0.7],
            'max_samples': [0.8, 1.0]
        }

        qrf = RandomForestQuantileRegressor(random_state=42)
        qrf_search = RandomizedSearchCV(qrf, qrf_param_grid, n_iter=30, cv=KFold(n_splits=5, shuffle=True, random_state=42),
                                       scoring='r2', n_jobs=-1, random_state=42)
        qrf_search.fit(X_train, y_train)
        best_qrf = qrf_search.best_estimator_
        
        qrf_pred = np.clip(best_qrf.predict(X, quantiles=[0.5]).flatten(), 0, None)
        qrf_test_pred = np.clip(best_qrf.predict(X_test, quantiles=[0.5]).flatten(), 0, None)
        qrf_r2 = r2_score(y, qrf_pred)
        qrf_val_r2 = r2_score(y_test, qrf_test_pred)
        qrf_rmse = np.sqrt(mean_squared_error(y_test, qrf_test_pred))
        qrf_mae = mean_absolute_error(y_test, qrf_test_pred)
        qrf_mape = mean_absolute_percentage_error(y_test, qrf_test_pred)
        qrf_score = composite_score(qrf_val_r2, qrf_rmse, qrf_mae, qrf_mape)
        
        results['Model'].append('QRF')
        results['R²'].append(qrf_r2)
        results['Validation R²'].append(qrf_val_r2)
        results['RMSE'].append(qrf_rmse)
        results['MAE'].append(qrf_mae)
        results['MAPE'].append(qrf_mape)
        results['Composite Score'].append(qrf_score)
        results['Watershed'].append(watershed_name)
        best_params['Model'].append('QRF')
        best_params['Parameters'].append(qrf_search.best_params_)
        best_params['Watershed'].append(watershed_name)
        predictions['Model'].append('QRF')
        predictions['Predictions'].append(qrf_pred.tolist())
        predictions['Test Predictions'].append(qrf_test_pred.tolist())
        predictions['Watershed'].append(watershed_name)
        print(f"QRF Performance ({watershed_name}): R²={qrf_r2:.3f}, Validation R²={qrf_val_r2:.3f}, RMSE={qrf_rmse:.3f}, MAE={qrf_mae:.3f}, MAPE={qrf_mape:.3f}, Score={qrf_score:.3f}")
        

        # Select best model based on validation R² (exclude SRC)
        results_df = pd.DataFrame(results)
        non_src_results = results_df[results_df['Model'] != 'SRC']
        best_model = non_src_results.loc[non_src_results['Validation R²'].idxmax()]['Model']
        print(f"\nBest model for {watershed_name}: {best_model} (Validation R²={non_src_results['Validation R²'].max():.3f})")
        
        # Save results, parameters, and predictions
        results_df.to_csv(OUTPUT_DIR / f"model_performance_{watershed_name.lower().replace(' ', '_')}_{int(test_size*100)}split.csv", index=False)
        best_params_df = pd.DataFrame(best_params)
        best_params_df.to_csv(OUTPUT_DIR / f"best_params_{watershed_name.lower().replace(' ', '_')}_{int(test_size*100)}split.csv", index=False)
        predictions_df = pd.DataFrame(predictions)
        predictions_df.to_csv(OUTPUT_DIR / f"predictions_{watershed_name.lower().replace(' ', '_')}_{int(test_size*100)}split.csv", index=False)
        
        return y, src_pred if not qrf_only else None, src_r2 if not qrf_only else None, \
               gb_pred if not qrf_only else None, gb_r2 if not qrf_only else None, \
               rf_pred if not qrf_only else None, rf_r2 if not qrf_only else None, \
               qrf_pred, qrf_r2, results_df, best_params_df, best_qrf, best_model
    
    except Exception as e:
        print(f"Error processing {watershed_name}: {str(e)}")
        return None


# Directories
BASE_DIR = Path(r"C:\Users\worku\Documents\sediment-yield-analysis")
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

# Process watersheds (70%/30% split)
gilgel_data = process_watershed(OUTPUT_DIR / "gilgel_abay_features.csv", "Gilgel Abay", 251, test_size=0.3)
gumara_data = process_watershed(OUTPUT_DIR / "gumara_features.csv", "Gumara", 245, test_size=0.3, qrf_only=False)

if gilgel_data is None or gumara_data is None:
    raise SystemExit("Processing failed for one or more watersheds.")

# Unpack results
y_gilgel, src_pred_gilgel, src_r2_gilgel, gb_pred_gilgel, gb_r2_gilgel, rf_pred_gilgel, rf_r2_gilgel, \
qrf_pred_gilgel, qrf_r2_gilgel, results_gilgel, params_gilgel, best_qrf_gilgel, best_model_gilgel = gilgel_data

y_gumara, src_pred_gumara, src_r2_gumara, gb_pred_gumara, gb_r2_gumara, rf_pred_gumara, rf_r2_gumara, \
qrf_pred_gumara, qrf_r2_gumara, results_gumara, params_gumara, best_qrf_gumara, best_model_gumara = gumara_data

# Load existing Gumara results and predictions
try:
    existing_results = pd.read_csv(OUTPUT_DIR / "model_performance_comparison_70split.csv")
    existing_gumara = existing_results[existing_results['Watershed'] == 'Gumara']
    results_gumara = pd.concat([existing_gumara, results_gumara], ignore_index=True)
    results_gumara.to_csv(OUTPUT_DIR / "model_performance_gumara_70split.csv", index=False)
except FileNotFoundError:
    print("Warning: Existing results file not found, proceeding with current results")
    existing_gumara = pd.DataFrame()


# Load or initialize Gumara predictions
try:
    gumara_pred_df = pd.read_csv(OUTPUT_DIR / "predictions_gumara_70split.csv")
except FileNotFoundError:
    print("Warning: Gumara predictions file not found, initializing empty")
    gumara_pred_df = pd.DataFrame({'Model': [], 'Predictions': [], 'Test Predictions': [], 'Watershed': []})

# Combine results and parameters
combined_results = pd.concat([results_gilgel, results_gumara], ignore_index=True)
combined_results.to_csv(OUTPUT_DIR / "model_performance_comparison_70split.csv", index=False)
combined_params = pd.concat([params_gilgel, params_gumara], ignore_index=True)
combined_params.to_csv(OUTPUT_DIR / "best_params_comparison_70split.csv", index=False)

# Print selected models
print(f"\nSelected model for Gilgel Abay: {best_model_gilgel} (Validation R²={results_gilgel['Validation R²'].max():.3f})")
print(f"Selected model for Gumara: {best_model_gumara} (Validation R²={results_gumara['Validation R²'].max():.3f})")

# Debug maximum values for plotting (unchanged)
print("\nDebugging maximum values for Gilgel Abay:")
print(f"Observed SSC max: {y_gilgel.max():.3f}")
print(f"SRC pred max: {src_pred_gilgel.max():.3f}" if src_pred_gilgel is not None else "SRC pred: None")
print(f"GB pred max: {gb_pred_gilgel.max():.3f}" if gb_pred_gilgel is not None else "GB pred: None")
print(f"RF pred max: {rf_pred_gilgel.max():.3f}" if rf_pred_gilgel is not None else "RF pred: None")
print(f"QRF pred max: {qrf_pred_gilgel.max():.3f}" if qrf_pred_gilgel is not None else "QRF pred: None")
print("\nDebugging maximum values for Gumara:")
print(f"Observed SSC max: {y_gumara.max():.3f}")
print(f"SRC pred max: {src_pred_gumara.max():.3f}" if src_pred_gumara is not None else "SRC pred: None")
print(f"GB pred max: {gb_pred_gumara.max():.3f}" if gb_pred_gumara is not None else "GB pred: None")
print(f"RF pred max: {rf_pred_gumara.max():.3f}" if rf_pred_gumara is not None else "RF pred: None")
print(f"QRF pred max: {qrf_pred_gumara.max():.3f}" if qrf_pred_gumara is not None else "QRF pred: None")

# Generate Figure 4 - Modified Section
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5), sharey=False)
fig.subplots_adjust(wspace=0.1)

# Gilgel Abay - Color version
if src_pred_gilgel is not None and not np.any(np.isnan(src_pred_gilgel)):
    ax1.scatter(y_gilgel, src_pred_gilgel, color='#1f77b4', marker='s', alpha=0.3, s=40, 
                edgecolors='black', linewidths=0.5, label=f'SRC (R² = {src_r2_gilgel:.2f})')
else:
    print("Warning: SRC predictions for Gilgel Abay contain NaNs or are None")
if gb_pred_gilgel is not None and not np.any(np.isnan(gb_pred_gilgel)):
    ax1.scatter(y_gilgel, gb_pred_gilgel, color='#800080', marker='^', alpha=0.3, s=40, 
                edgecolors='black', linewidths=0.5, linestyle='--', label=f'GB (R² = {gb_r2_gilgel:.2f})')
else:
    print("Warning: GB predictions for Gilgel Abay contain NaNs or are None")
if rf_pred_gilgel is not None and not np.any(np.isnan(rf_pred_gilgel)):
    ax1.scatter(y_gilgel, rf_pred_gilgel, color='#2ca02c', marker='d', alpha=0.3, s=40, 
                edgecolors='black', linewidths=0.5, linestyle=':', label=f'RF (R² = {rf_r2_gilgel:.2f})')
else:
    print("Warning: RF predictions for Gilgel Abay contain NaNs or are None")
if qrf_pred_gilgel is not None and not np.any(np.isnan(qrf_pred_gilgel)):
    ax1.scatter(y_gilgel, qrf_pred_gilgel, color='#ff7f0e', marker='o', alpha=0.6, s=50, 
                edgecolors='black', linewidths=0.5, linestyle='-', label=f'QRF (R² = {qrf_r2_gilgel:.2f})')
else:
    print("Warning: QRF predictions for Gilgel Abay contain NaNs or are None")
valid_preds_gilgel = [p for p in [y_gilgel, src_pred_gilgel, gb_pred_gilgel, rf_pred_gilgel, qrf_pred_gilgel] if p is not None and not np.any(np.isnan(p))]
max_val_gilgel = np.ceil(max([p.max() for p in valid_preds_gilgel]) * 1.1) if valid_preds_gilgel else 10.0
print(f"Gilgel Abay plot max value: {max_val_gilgel:.3f}")
ax1.plot([0, max_val_gilgel], [0, max_val_gilgel], color='black', linestyle='--', label='1:1 Line')
ax1.set_xlim(0, max_val_gilgel)
ax1.set_ylim(0, max_val_gilgel)
ax1.set_xlabel('Observed SSC (g/L)', fontsize=18)
ax1.set_ylabel('Predicted SSC (g/L)', fontsize=18)
ax1.set_title('(a) Gilgel Abay', fontsize=20)
ax1.grid(False)
ax1.legend(loc='upper left', fontsize=14)
ax1.tick_params(axis='both', which='major', labelsize=16)
ax1.set_aspect('equal')

# Gumara - Color version
if src_pred_gumara is not None and not np.any(np.isnan(src_pred_gumara)):
    ax2.scatter(y_gumara, src_pred_gumara, color='#1f77b4', marker='s', alpha=0.3, s=40, 
                edgecolors='black', linewidths=0.5, label=f'SRC (R² = {src_r2_gumara:.2f})')
else:
    print("Warning: SRC predictions for Gumara contain NaNs or are None")
if gb_pred_gumara is not None and not np.any(np.isnan(gb_pred_gumara)):
    ax2.scatter(y_gumara, gb_pred_gumara, color='#800080', marker='^', alpha=0.3, s=40, 
                edgecolors='black', linewidths=0.5, linestyle='--', label=f'GB (R² = {gb_r2_gumara:.2f})')
else:
    print("Warning: GB predictions for Gumara contain NaNs or are None")
if rf_pred_gumara is not None and not np.any(np.isnan(rf_pred_gumara)):
    ax2.scatter(y_gumara, rf_pred_gumara, color='#2ca02c', marker='d', alpha=0.3, s=40, 
                edgecolors='black', linewidths=0.5, linestyle=':', label=f'RF (R² = {rf_r2_gumara:.2f})')
else:
    print("Warning: RF predictions for Gumara contain NaNs or are None")
if qrf_pred_gumara is not None and not np.any(np.isnan(qrf_pred_gumara)):
    ax2.scatter(y_gumara, qrf_pred_gumara, color='#ff7f0e', marker='o', alpha=0.6, s=50, 
                edgecolors='black', linewidths=0.5, linestyle='-', label=f'QRF (R² = {qrf_r2_gumara:.2f})')
else:
    print("Warning: QRF predictions for Gumara contain NaNs or are None")
valid_preds_gumara = [p for p in [y_gumara, src_pred_gumara, gb_pred_gumara, rf_pred_gumara, qrf_pred_gumara] if p is not None and not np.any(np.isnan(p))]
max_val_gumara = np.ceil(max([p.max() for p in valid_preds_gumara]) * 1.1) if valid_preds_gumara else 10.0
max_val_gumara = min(max_val_gumara, 10.0)
print(f"Gumara plot max value: {max_val_gumara:.3f}")
ax2.plot([0, max_val_gumara], [0, max_val_gumara], color='black', linestyle='--', label='1:1 Line')
ax2.set_xlim(0, max_val_gumara)
ax2.set_ylim(0, max_val_gumara)
ax2.set_xlabel('Observed SSC (g/L)', fontsize=18)
ax2.set_title('(b) Gumara', fontsize=20)
ax2.grid(False)
ax2.legend(loc='upper left', fontsize=14)
ax2.tick_params(axis='both', which='major', labelsize=16)
ax2.set_aspect('equal')

# Save color version for online
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "Figure4_observed_vs_predicted_70split_color.png", transparent=True, dpi=600, bbox_inches='tight')
plt.savefig(OUTPUT_DIR / "Figure4_observed_vs_predicted_70split_color.svg", transparent=True, format='svg', bbox_inches='tight')

# Grayscale version for print
plt.clf()
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5), sharey=False)
fig.subplots_adjust(wspace=0.1)

# Gilgel Abay - Grayscale
if src_pred_gilgel is not None and not np.any(np.isnan(src_pred_gilgel)):
    ax1.scatter(y_gilgel, src_pred_gilgel, color='#000000', marker='s', alpha=0.3, s=40, 
                edgecolors='black', linewidths=0.5, label=f'SRC (R² = {src_r2_gilgel:.2f})')
if gb_pred_gilgel is not None and not np.any(np.isnan(gb_pred_gilgel)):
    ax1.scatter(y_gilgel, gb_pred_gilgel, color='#333333', marker='^', alpha=0.3, s=40, 
                edgecolors='black', linewidths=0.5, linestyle='--', label=f'GB (R² = {gb_r2_gilgel:.2f})')
if rf_pred_gilgel is not None and not np.any(np.isnan(rf_pred_gilgel)):
    ax1.scatter(y_gilgel, rf_pred_gilgel, color='#666666', marker='d', alpha=0.3, s=40, 
                edgecolors='black', linewidths=0.5, linestyle=':', label=f'RF (R² = {rf_r2_gilgel:.2f})')
if qrf_pred_gilgel is not None and not np.any(np.isnan(qrf_pred_gilgel)):
    ax1.scatter(y_gilgel, qrf_pred_gilgel, color='#999999', marker='o', alpha=0.6, s=50, 
                edgecolors='black', linewidths=0.5, linestyle='-', label=f'QRF (R² = {qrf_r2_gilgel:.2f})')
ax1.plot([0, max_val_gilgel], [0, max_val_gilgel], color='black', linestyle='--', label='1:1 Line')
ax1.set_xlim(0, max_val_gilgel)
ax1.set_ylim(0, max_val_gilgel)
ax1.set_xlabel('Observed SSC (g/L)', fontsize=18)
ax1.set_ylabel('Predicted SSC (g/L)', fontsize=18)
ax1.set_title('(a) Gilgel Abay', fontsize=20)
ax1.grid(False)
ax1.legend(loc='upper left', fontsize=14)
ax1.tick_params(axis='both', which='major', labelsize=16)
ax1.set_aspect('equal')

# Gumara - Grayscale
if src_pred_gumara is not None and not np.any(np.isnan(src_pred_gumara)):
    ax2.scatter(y_gumara, src_pred_gumara, color='#000000', marker='s', alpha=0.3, s=40, 
                edgecolors='black', linewidths=0.5, label=f'SRC (R² = {src_r2_gumara:.2f})')
if gb_pred_gumara is not None and not np.any(np.isnan(gb_pred_gumara)):
    ax2.scatter(y_gumara, gb_pred_gumara, color='#333333', marker='^', alpha=0.3, s=40, 
                edgecolors='black', linewidths=0.5, linestyle='--', label=f'GB (R² = {gb_r2_gumara:.2f})')
if rf_pred_gumara is not None and not np.any(np.isnan(rf_pred_gumara)):
    ax2.scatter(y_gumara, rf_pred_gumara, color='#666666', marker='d', alpha=0.3, s=40, 
                edgecolors='black', linewidths=0.5, linestyle=':', label=f'RF (R² = {rf_r2_gumara:.2f})')
if qrf_pred_gumara is not None and not np.any(np.isnan(qrf_pred_gumara)):
    ax2.scatter(y_gumara, qrf_pred_gumara, color='#999999', marker='o', alpha=0.6, s=50, 
                edgecolors='black', linewidths=0.5, linestyle='-', label=f'QRF (R² = {qrf_r2_gumara:.2f})')
ax2.plot([0, max_val_gumara], [0, max_val_gumara], color='black', linestyle='--', label='1:1 Line')
ax2.set_xlim(0, max_val_gumara)
ax2.set_ylim(0, max_val_gumara)
ax2.set_xlabel('Observed SSC (g/L)', fontsize=18)
ax2.set_title('(b) Gumara', fontsize=20)
ax2.grid(False)
ax2.legend(loc='upper left', fontsize=14)
ax2.tick_params(axis='both', which='major', labelsize=16)
ax2.set_aspect('equal')

# Save grayscale version for print
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "Figure4_observed_vs_predicted_70split_grayscale.png", transparent=True, dpi=600, bbox_inches='tight')
plt.savefig(OUTPUT_DIR / "Figure4_observed_vs_predicted_70split_grayscale.eps", transparent=True, format='eps', dpi=600, bbox_inches='tight')
plt.show()
plt.close()
