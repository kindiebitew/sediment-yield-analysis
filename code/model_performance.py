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
import traceback

# Define MAPE metric
def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    y_true = np.clip(y_true, 0.01, None)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

# Compute sediment yields
def compute_sediment_yields(df, ssc_pred, watershed_name, date_col='Date'):
    try:
        if date_col not in df.columns:

            print(f"Warning: {date_col} column missing in {watershed_name} data. Adding dummy dates.")
            df['Date'] = pd.date_range(start='2020-01-01', periods=len(df), freq='D')
        
        df['Date'] = pd.to_datetime(df['Date'])
        df['Sediment_Load'] = ssc_pred * df['Discharge'] * 86400 / 1000  # g/L * m³/s * sec/day → tonnes/day
        df['Year'] = df['Date'].dt.year
        df['Month'] = df['Date'].dt.month
        
        # Define seasons (wet: Jun-Sep, dry: Oct-May)
        df['Season'] = df['Month'].apply(lambda x: 'Wet' if 6 <= x <= 9 else 'Dry')
        
        # Annual yields
        annual_yields = df.groupby('Year')['Sediment_Load'].sum().reset_index()
        annual_yields.columns = ['Year', f'Annual_Yield_tonnes_{watershed_name}']
        
        # Seasonal yields
        seasonal_yields = df.groupby(['Year', 'Season'])['Sediment_Load'].sum().unstack().reset_index()
        seasonal_yields.columns = ['Year', f'Dry_Yield_tonnes_{watershed_name}', f'Wet_Yield_tonnes_{watershed_name}']
        
        return annual_yields, seasonal_yields
    except Exception as e:
        print(f"Error computing sediment yields for {watershed_name}: {str(e)}")
        return None, None

# Process watershed data
def process_watershed(data_path, watershed_name, n_samples, is_excel=False):
    try:
        # Load data
        if is_excel:
            df = pd.read_excel(data_path)
        else:
            df = pd.read_csv(data_path)
        
        # Check required columns
        required_columns = ['Rainfall', 'Discharge', 'Temperature', 'ETo', 'SSC']
        if not all(col in df.columns for col in required_columns):
            raise ValueError(f"{watershed_name} data must contain: {', '.join(required_columns)}")
        
        # Filter non-missing SSC
        df_filtered = df.dropna(subset=['SSC']).head(n_samples)
        print(f"Sample count for {watershed_name}: {len(df_filtered)}")
        
        # Clip SSC
        df_filtered['SSC'] = df_filtered['SSC'].clip(lower=0.01, upper=5.0 if watershed_name == "Gilgel Abay" else 4.0)
        
        # Feature engineering
        df_filtered['Log_Discharge'] = np.log1p(df_filtered['Discharge'])
        df_filtered['MA_Discharge_3'] = df_filtered['Discharge'].rolling(window=3, min_periods=1).mean()
        df_filtered['Lag_Discharge'] = df_filtered['Discharge'].shift(1).bfill()
        df_filtered['Lag_Discharge_3'] = df_filtered['Discharge'].shift(3).bfill()
        df_filtered['Discharge_Rainfall'] = df_filtered['Discharge'] * df_filtered['Rainfall']
        
        # Select features
        predictors = ['Log_Discharge', 'MA_Discharge_3', 'Lag_Discharge', 'Lag_Discharge_3', 'Rainfall', 'Discharge_Rainfall']
        target = 'SSC'
        X = df_filtered[predictors]
        y = df_filtered[target]
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=predictors, index=X.index)
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
        X_train = X_train.reset_index(drop=True)
        X_test = X_test.reset_index(drop=True)
        y_train = y_train.reset_index(drop=True)
        y_test = y_test.reset_index(drop=True)
        
        # Reconstruct discharge for SRC
        X_train_raw = pd.DataFrame(scaler.inverse_transform(X_train), columns=predictors)
        X_train_raw['Discharge'] = np.expm1(X_train_raw['Log_Discharge'])
        X_test_raw = pd.DataFrame(scaler.inverse_transform(X_test), columns=predictors)
        X_test_raw['Discharge'] = np.expm1(X_test_raw['Log_Discharge'])
        
        # Results storage
        results = {'Model': [], 'R²': [], 'Validation R²': [], 'RMSE': [], 'MAE': [], 'MAPE': [], 'Watershed': []}
        
        # SRC: Segmented power-law
        median_discharge = df_filtered['Discharge'].median()
        low_flow_train = X_train_raw['Discharge'] < median_discharge
        high_flow_train = X_train_raw['Discharge'] >= median_discharge
        
        if low_flow_train.any():
            X_src_low = np.log(X_train_raw.loc[low_flow_train, 'Discharge'].values + 1e-6).reshape(-1, 1)
            y_src_low = np.log(y_train[low_flow_train].values + 1e-6)
            src_model_low = LinearRegression().fit(X_src_low, y_src_low)
        else:
            src_model_low = None
        
        if high_flow_train.any():
            X_src_high = np.log(X_train_raw.loc[high_flow_train, 'Discharge'].values + 1e-6).reshape(-1, 1)
            y_src_high = np.log(y_train[high_flow_train].values + 1e-6)
            src_model_high = LinearRegression().fit(X_src_high, y_src_high)
        else:
            src_model_high = None
        
        # SRC predictions
        low_flow_full = df_filtered['Discharge'] < median_discharge
        high_flow_full = df_filtered['Discharge'] >= median_discharge
        src_pred = np.zeros(len(y))
        if src_model_low and low_flow_full.any():
            X_src_low_full = np.log(df_filtered.loc[low_flow_full, 'Discharge'].values + 1e-6).reshape(-1, 1)
            src_pred[low_flow_full] = np.exp(src_model_low.predict(X_src_low_full))
        if src_model_high and high_flow_full.any():
            X_src_high_full = np.log(df_filtered.loc[high_flow_full, 'Discharge'].values + 1e-6).reshape(-1, 1)
            src_pred[high_flow_full] = np.exp(src_model_high.predict(X_src_high_full))
        src_pred = np.clip(src_pred, 0.01, 5.0 if watershed_name == "Gilgel Abay" else 4.0)
        
        low_flow_test = X_test_raw['Discharge'] < median_discharge
        high_flow_test = X_test_raw['Discharge'] >= median_discharge
        src_test_pred = np.zeros(len(y_test))
        if src_model_low and low_flow_test.any():
            X_src_low_test = np.log(X_test_raw.loc[low_flow_test, 'Discharge'].values + 1e-6).reshape(-1, 1)
            src_test_pred[low_flow_test] = np.exp(src_model_low.predict(X_src_low_test))
        if src_model_high and high_flow_test.any():
            X_src_high_test = np.log(X_test_raw.loc[high_flow_test, 'Discharge'].values + 1e-6).reshape(-1, 1)
            src_test_pred[high_flow_test] = np.exp(src_model_high.predict(X_src_high_test))
        src_test_pred = np.clip(src_test_pred, 0.01, 5.0 if watershed_name == "Gilgel Abay" else 4.0)
        
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
        print(f"SRC Performance ({watershed_name}): R²={src_r2:.3f}, Validation R²={src_val_r2:.3f}, RMSE={src_rmse:.3f}, MAE={src_mae:.3f}, MAPE={src_mape:.3f}%")
        
        # GB
        gb_param_grid = {
            'n_estimators': [500, 1000],
            'max_depth': [3, 5],
            'learning_rate': [0.01, 0.05],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [2, 5],
            'subsample': [0.7, 0.9]
        }
        gb = GradientBoostingRegressor(random_state=42, n_iter_no_change=10, validation_fraction=0.1)
        try:
            gb_search = RandomizedSearchCV(gb, gb_param_grid, n_iter=15, cv=KFold(n_splits=5, shuffle=True, random_state=42), scoring='r2', n_jobs=-1, random_state=42)
            gb_search.fit(X_train, y_train)
            best_gb = gb_search.best_estimator_
        except Exception as e:
            print(f"GB fitting failed for {watershed_name}: {str(e)}")
            return None
        
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
        print(f"GB Performance ({watershed_name}): R²={gb_r2:.3f}, Validation R²={gb_val_r2:.3f}, RMSE={gb_rmse:.3f}, MAE={gb_mae:.3f}, MAPE={gb_mape:.3f}%")
        
        # RF
        rf_param_grid = {
            'n_estimators': [500, 1000],
            'max_depth': [10, 20],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [2, 5],
            'max_features': ['sqrt', 'log2']
        }
        rf = RandomForestRegressor(random_state=42)
        try:
            rf_search = RandomizedSearchCV(rf, rf_param_grid, n_iter=15, cv=KFold(n_splits=5, shuffle=True, random_state=42), scoring='r2', n_jobs=-1, random_state=42)
            rf_search.fit(X_train, y_train)
            best_rf = rf_search.best_estimator_
        except Exception as e:
            print(f"RF fitting failed for {watershed_name}: {str(e)}")
            return None
        
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
        print(f"RF Performance ({watershed_name}): R²={rf_r2:.3f}, Validation R²={rf_val_r2:.3f}, RMSE={rf_rmse:.3f}, MAE={rf_mae:.3f}, MAPE={rf_mape:.3f}%")
        
        # QRF
        qrf_param_grid = {
            'n_estimators': [500, 1000, 2000],
            'max_depth': [10, 20, 40],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2, 3],
            'max_features': ['sqrt', 0.5, 0.8]
        }
        qrf = RandomForestQuantileRegressor(random_state=42)
        try:
            qrf_search = RandomizedSearchCV(qrf, qrf_param_grid, n_iter=20, cv=KFold(n_splits=5, shuffle=True, random_state=42), scoring='r2', n_jobs=-1, random_state=42)
            qrf_search.fit(X_train, y_train)
            best_qrf = qrf_search.best_estimator_
        except Exception as e:
            print(f"QRF fitting failed for {watershed_name}: {str(e)}")
            return None
        
        quantiles = [0.5]
        qrf_pred = best_qrf.predict(X_scaled, quantiles=quantiles).flatten()
        qrf_test_pred = best_qrf.predict(X_test, quantiles=quantiles).flatten()
        qrf_pred = np.clip(qrf_pred, 0.01, 5.0 if watershed_name == "Gilgel Abay" else 4.0)
        qrf_test_pred = np.clip(qrf_test_pred, 0.01, 5.0 if watershed_name == "Gilgel Abay" else 4.0)
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
        print(f"QRF Performance ({watershed_name}): R²={qrf_r2:.3f}, Validation R²={qrf_val_r2:.3f}, RMSE={qrf_rmse:.3f}, MAE={qrf_mae:.3f}, MAPE={qrf_mape:.3f}%")
        
        # Results table
        results_df = pd.DataFrame(results)
        print(f"\nModel Performance Summary for {watershed_name}:\n{results_df}")
        
        # Compute sediment yields with QRF
        annual_yields, seasonal_yields = compute_sediment_yields(df_filtered, qrf_pred, watershed_name)
        if annual_yields is not None:
            annual_yields.to_csv(f"D:\\Gilgel Abay\\Sedigrapgh\\qrf_annual_yields_{watershed_name.lower().replace(' ', '_')}_v14.csv", index=False)
            seasonal_yields.to_csv(f"D:\\Gilgel Abay\\Sedigrapgh\\qrf_seasonal_yields_{watershed_name.lower().replace(' ', '_')}_v14.csv", index=False)
            print(f"QRF annual yields saved to D:\\Gilgel Abay\\Sedigrapgh\\qrf_annual_yields_{watershed_name.lower().replace(' ', '_')}_v14.csv")
            print(f"QRF seasonal yields saved to D:\\Gilgel Abay\\Sedigrapgh\\qrf_seasonal_yields_{watershed_name.lower().replace(' ', '_')}_v14.csv")
        
        return y, src_pred, src_r2, gb_pred, gb_r2, rf_pred, rf_r2, qrf_pred, qrf_r2, results_df, annual_yields, seasonal_yields
    
    except Exception as e:
        print(f"Error processing {watershed_name}: {str(e)}")
        traceback.print_exc()
        return None

# Process watersheds
gilgel_path = r"D:\Gilgel Abay\Sedigrapgh\Intermittent_data.xlsx"
gumara_path = r"D:\Gumara\Sedigrapgh\Intermittent_data_gum.csv"
gilgel_data = process_watershed(gilgel_path, "Gilgel Abay", n_samples=251, is_excel=True)
gumara_data = process_watershed(gumara_path, "Gumara", n_samples=245)

if gilgel_data is None or gumara_data is None:
    raise SystemExit("Processing failed for one or more watersheds.")

# Unpack data
y_gilgel, src_pred_gilgel, src_r2_gilgel, gb_pred_gilgel, gb_r2_gilgel, rf_pred_gilgel, rf_r2_gilgel, \
qrf_pred_gilgel, qrf_r2_gilgel, results_gilgel, annual_yields_gilgel, seasonal_yields_gilgel = gilgel_data

y_gumara, src_pred_gumara, src_r2_gumara, gb_pred_gumara, gb_r2_gumara, rf_pred_gumara, rf_r2_gumara, \
qrf_pred_gumara, qrf_r2_gumara, results_gumara, annual_yields_gumara, seasonal_yields_gumara = gumara_data

# Combine results
combined_results = pd.concat([results_gilgel, results_gumara], ignore_index=True)
csv_output = r"D:\Gilgel Abay\Sedigrapgh\model_performance_comparison_v14.csv"
combined_results.to_csv(csv_output, index=False)
print(f"\nCombined Model Performance Across Watersheds:\n{combined_results}")
print(f"Combined results saved to {csv_output}")

# Plot style
sns.set_style('white')
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 14

# Combined scatterplot with subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5), facecolor='none', sharey=True)
fig.subplots_adjust(wspace=0.1)

# Gilgel Abay
ax1.scatter(y_gilgel, src_pred_gilgel, color='#1f77b4', marker='s', alpha=0.3, s=40, label=f'SRC (R² = {src_r2_gilgel:.2f})')
ax1.scatter(y_gilgel, gb_pred_gilgel, color='#800080', marker='^', alpha=0.3, s=40, label=f'GB (R² = {gb_r2_gilgel:.2f})')
ax1.scatter(y_gilgel, rf_pred_gilgel, color='#2ca02c', marker='d', alpha=0.3, s=40, label=f'RF (R² = {rf_r2_gilgel:.2f})')
ax1.scatter(y_gilgel, qrf_pred_gilgel, color='#ff7f0e', marker='o', alpha=0.6, s=50, edgecolor='black', 
            label=f'QRF (R² = {qrf_r2_gilgel:.2f})')
max_val_gilgel = max(y_gilgel.max(), src_pred_gilgel.max(), gb_pred_gilgel.max(), rf_pred_gilgel.max(), qrf_pred_gilgel.max())
ax1.plot([0, max_val_gilgel], [0, max_val_gilgel], color='black', linestyle='--', label='1:1 Line')
ax1.set_xlabel('Observed SSC (g/L)', fontsize=18)
ax1.set_ylabel('Predicted SSC (g/L)', fontsize=18)
ax1.set_title('Gilgel Abay', fontsize=20)
ax1.grid(False)
ax1.legend(loc='upper left', fontsize=14, frameon=True, edgecolor='lightgray', framealpha=0.8, 
           prop={'family': 'Times New Roman'}, bbox_to_anchor=(0.01, 0.99))
ax1.tick_params(axis='both', which='major', labelsize=16)
for spine in ax1.spines.values():
    spine.set_linewidth(1)
    spine.set_color('black')

# Gumara
ax2.scatter(y_gumara, src_pred_gumara, color='#1f77b4', marker='s', alpha=0.3, s=40, label=f'SRC (R² = {src_r2_gumara:.2f})')
ax2.scatter(y_gumara, gb_pred_gumara, color='#800080', marker='^', alpha=0.3, s=40, label=f'GB (R² = {gb_r2_gumara:.2f})')
ax2.scatter(y_gumara, rf_pred_gumara, color='#2ca02c', marker='d', alpha=0.3, s=40, label=f'RF (R² = {rf_r2_gumara:.2f})')
ax2.scatter(y_gumara, qrf_pred_gumara, color='#ff7f0e', marker='o', alpha=0.6, s=50, edgecolor='black', 
            label=f'QRF (R² = {qrf_r2_gumara:.2f})')
max_val_gumara = max(y_gumara.max(), src_pred_gumara.max(), gb_pred_gumara.max(), rf_pred_gumara.max(), qrf_pred_gumara.max())
ax2.plot([0, max_val_gumara], [0, max_val_gumara], color='black', linestyle='--', label='1:1 Line')
ax2.set_xlabel('Observed SSC (g/L)', fontsize=18)
ax2.set_title('Gumara', fontsize=20)
ax2.grid(False)
ax2.legend(loc='upper left', fontsize=14, frameon=True, edgecolor='lightgray', framealpha=0.8, 
           prop={'family': 'Times New Roman'}, bbox_to_anchor=(0.01, 0.99))
ax2.tick_params(axis='both', which='major', labelsize=16)
for spine in ax2.spines.values():
    spine.set_linewidth(1)
    spine.set_color('black')

plt.tight_layout()
combined_output_path = r"D:\Gilgel Abay\Sedigrapgh\observed_vs_predicted_combined_v14.png"
svg_output_path = combined_output_path.replace('.png', '.svg')
plt.savefig(combined_output_path, transparent=True, dpi=600, bbox_inches='tight')
plt.savefig(svg_output_path, transparent=True, format='svg', bbox_inches='tight')
plt.show()
print(f"Combined scatter plot saved to {combined_output_path} (PNG) and {svg_output_path} (SVG)")

# Validation R² bar chart
models = ['SRC', 'GB', 'RF', 'QRF']
gilgel_val_r2 = [results_gilgel['Validation R²'].iloc[i] for i in range(4)]
gumara_val_r2 = [results_gumara['Validation R²'].iloc[i] for i in range(4)]
x = np.arange(len(models))
width = 0.35
fig, ax = plt.subplots(figsize=(8, 6), facecolor='none')
ax.set_facecolor('none')
ax.bar(x - width/2, gilgel_val_r2, width, label='Gilgel Abay', color='#1f77b4')
ax.bar(x + width/2, gumara_val_r2, width, label='Gumara', color='#2ca02c')
ax.set_xlabel('Model', fontsize=16)
ax.set_ylabel('Validation R²', fontsize=16)
ax.set_title('Validation R² by Model and Watershed', fontsize=18)
ax.set_xticks(x)
ax.set_xticklabels(models, fontsize=14)
ax.set_ylim(0.75, 0.90)
ax.legend(loc='upper left', fontsize=12, frameon=True, edgecolor='lightgray', framealpha=0.8, 
         prop={'family': 'Times New Roman'})
ax.tick_params(axis='both', labelsize=14)
for spine in ax.spines.values():
    spine.set_linewidth(1)
    spine.set_color('black')
plt.tight_layout() bar_output_path = r"D:\Gilgel Abay\Sedigrapgh\validation_r2_bar_v14.png" svg_bar_output_path = bar_output_path.replace('.png', '.svg') plt.savefig(bar_output_path, transparent=True, dpi=600, bbox_inches='tight') plt.savefig(svg_bar_output_path, 
transparent=True, format='svg', bbox_inches='tight') plt.show() print(f"Validation R² bar chart saved to {bar_output_path} (PNG) and {svg_bar_output_path} (SVG)")
