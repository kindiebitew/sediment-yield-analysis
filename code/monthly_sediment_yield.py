# Script to generate Figure 8: Monthly sediment yield, rainfall, and discharge for Gilgel Abay and Gumara watersheds
# (Section 3.3). Predicts daily SSC (g/L) for 1990–2020 using Quantile Random Forest (QRF) trained on intermittent data,
# calculates daily sediment yield (t/ha/day), aggregates to monthly values, and produces a combination plot with a bar
# plot for monthly rainfall (mm, reversed axis) and line plots for discharge (m³/s) and sediment yield (t/ha/month),
# including uncertainty bands based on the interquartile range (IQR) of predictions.
# Author: Kindie B. Worku
# Date: 2025-07-16 

import pandas as pd
import numpy as np
from quantile_forest import RandomForestQuantileRegressor
from sklearn.preprocessing import RobustScaler
import matplotlib
matplotlib.use('TkAgg')  
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from matplotlib.ticker import MaxNLocator
import warnings
import os

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Set plot style for publication quality (Section 3.3)
sns.set_style('white')
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 16

print(f"Matplotlib version: {matplotlib.__version__}")  # Debug: Check matplotlib version
print(f"Matplotlib backend: {plt.get_backend()}")  # Debug: Check active backend

# Test plot to verify display (should show a simple line if display works)
plt.figure()
plt.plot([1, 2, 3], [4, 5, 6])
plt.title("Test Plot")
plt.show()  # Test display before main plot

# Define constants
WATERSHED_CONFIG = {
    'Gilgel Abay': {
        'intermittent': Path(r"C:\Users\worku\Documents\sediment-yield-analysis\data\Intermittent_data.xlsx"),
        'continuous': Path(r"C:\Users\worku\Documents\sediment-yield-analysis\data\continuous_data.csv"),
        'area_km2': 1664,
        'discharge_max': 800,
        'yield_max': 50,
        'rainfall_max': 1800,
        'output_dir': Path(r"C:\Users\worku\Documents\sediment-yield-analysis\outputs")
    },
    'Gumara': {
        'intermittent': Path(r"C:\Users\worku\Documents\sediment-yield-analysis\data\Intermittent_data_gum.csv"),
        'continuous': Path(r"C:\Users\worku\Documents\sediment-yield-analysis\data\continuous_data_gum.csv"),
        'area_km2': 1394,
        'discharge_max': 800,
        'yield_max': 60,
        'rainfall_max': 1800,
        'output_dir': Path(r"C:\Users\worku\Documents\sediment-yield-analysis\outputs")
    }
}

QRF_PARAMS = {
    'Gilgel Abay': {
        'n_estimators': 1000,
        'min_samples_split': 5,
        'min_samples_leaf': 2,
        'max_features': 'log2',
        'max_depth': 30,
        'random_state': 42
    },
    'Gumara': {
        'n_estimators': 1000,
        'min_samples_split': 5,
        'min_samples_leaf': 2,
        'max_features': 'log2',
        'max_depth': 30,
        'random_state': 42
    }
}

LOAD_FACTOR = 86.4  # Converts m³/s × g/L to t/day (86,400 s/day × 10⁻⁶ t/g)

def predict_ssc(intermittent_path, continuous_path, watershed_name, qrf_params, is_excel_inter=False):
    """
    Predict daily SSC (g/L) for 1990–2020 using QRF trained on intermittent data with uncertainty quantiles.
    Args:
        intermittent_path (Path): Path to intermittent data (Excel or CSV).
        continuous_path (Path): Path to continuous data (CSV).
        watershed_name (str): Name of the watershed ('Gilgel Abay' or 'Gumara').
        qrf_params (dict): QRF hyperparameters.
        is_excel_inter (bool): Flag to indicate if intermittent file is Excel (True) or CSV (False).
    Returns:
        DataFrame with continuous data and predicted SSC with quantiles, or None if processing fails.
    """
    print(f"\nPredicting SSC for {watershed_name}...")
    
    # Check file existence
    if not intermittent_path.exists():
        print(f"Intermittent file not found: {intermittent_path}")
        print(f"Available files in {intermittent_path.parent}: {os.listdir(intermittent_path.parent)}")
        return None
    if not continuous_path.exists():
        print(f"Continuous file not found: {continuous_path}")
        print(f"Available files in {continuous_path.parent}: {os.listdir(continuous_path.parent)}")
        return None
    
    # Load data
    try:
        if is_excel_inter:
            df_inter = pd.read_excel(intermittent_path, engine='openpyxl')
        else:
            df_inter = pd.read_csv(intermittent_path)
        
        df_cont = pd.read_csv(continuous_path)
        
        print(f"{watershed_name} Intermittent Data Shape: {df_inter.shape}")
        print(f"{watershed_name} Intermittent Columns: {list(df_inter.columns)}")
        print(f"{watershed_name} Continuous Data Shape: {df_cont.shape}")
        print(f"{watershed_name} Continuous Columns: {list(df_cont.columns)}")
    except Exception as e:
        print(f"Error loading data for {watershed_name}: {str(e)}")
        return None
    
    # Check if continuous data files are identical for both watersheds
    if watershed_name == 'Gumara':
        gilgel_cont_path = WATERSHED_CONFIG['Gilgel Abay']['continuous']
        if gilgel_cont_path == continuous_path:
            print(f"Warning: Identical continuous data files used for Gilgel Abay and Gumara ({continuous_path}). This may lead to identical results.")
        else:
            try:
                df_gilgel_cont = pd.read_csv(gilgel_cont_path)
                if df_cont.equals(df_gilgel_cont):
                    print(f"Warning: Continuous data for Gilgel Abay and Gumara are identical. This may lead to incorrect results.")
            except Exception:
                pass  # Skip comparison if file can't be read
    
    # Define expected column names and possible alternatives
    column_mapping = {
        'Rainfall': ['Rainfall', 'rainfall', 'Rain', 'rain'],
        'Discharge': ['Discharge', 'discharge', 'Flow', 'flow'],
        'Temperature': ['Temperature', 'temperature', 'Temp', 'temp'],
        'ETo': ['ETo', 'eto', 'ET0', 'Evapotranspiration', 'evapotranspiration'],
        'SSC': ['SSC', 'ssc', 'SuspendedSediment', 'suspended_sediment'],
        'Date': ['Date', 'date', 'Time', 'time', 'Timestamp', 'timestamp']
    }
    
    # Rename columns if necessary
    for df, df_name in [(df_inter, 'intermittent'), (df_cont, 'continuous')]:
        for expected_col, alternatives in column_mapping.items():
            found = False
            for alt in alternatives:
                if alt in df.columns:
                    df.rename(columns={alt: expected_col}, inplace=True)
                    found = True
                    break
            if not found and expected_col != 'Date' and not (df_name == 'continuous' and expected_col == 'SSC'):
                print(f"Warning: {watershed_name} {df_name} data missing column: {expected_col}. Available: {list(df.columns)}")
                return None
    
    # Validate required columns
    required_cols_inter = ['Rainfall', 'Discharge', 'Temperature', 'ETo', 'SSC']
    required_cols_cont = ['Rainfall', 'Discharge', 'Temperature', 'ETo']
    if not all(col in df_inter.columns for col in required_cols_inter):
        print(f"{watershed_name} intermittent data missing columns: {required_cols_inter}. Available: {list(df_inter.columns)}")
        return None
    if not all(col in df_cont.columns for col in required_cols_cont):
        print(f"{watershed_name} continuous data missing columns: {required_cols_cont}. Available: {list(df_cont.columns)}")
        return None
    
    # Convert dates to datetime or use index if Date is missing
    if 'Date' in df_inter.columns and 'Date' in df_cont.columns:
        df_inter['Date'] = pd.to_datetime(df_inter['Date'], errors='coerce')
        df_cont['Date'] = pd.to_datetime(df_cont['Date'], errors='coerce')
        df_inter = df_inter.dropna(subset=['Date'])
        df_cont = df_cont.dropna(subset=['Date'])
        if df_inter.empty or df_cont.empty:
            print(f"{watershed_name} data empty after date cleaning: Intermittent={len(df_inter)}, Continuous={len(df_cont)}")
            return None
    else:
        print(f"Warning: No Date column in {watershed_name} {df_inter.columns if 'Date' not in df_inter.columns else df_cont.columns}. Using index.")
        df_inter['Date'] = pd.date_range(start='1990-01-01', periods=len(df_inter), freq='D')
        df_cont['Date'] = pd.date_range(start='1990-01-01', periods=len(df_cont), freq='D')
    
    # Ensure numeric columns
    numeric_cols = ['Rainfall', 'Discharge', 'Temperature', 'ETo', 'SSC']
    for col in numeric_cols:
        if col in df_inter:
            df_inter[col] = pd.to_numeric(df_inter[col], errors='coerce')
    for col in numeric_cols[:-1]:
        df_cont[col] = pd.to_numeric(df_cont[col], errors='coerce')
    
    # Drop missing values
    df_inter = df_inter.dropna(subset=numeric_cols)
    df_cont = df_cont.dropna(subset=numeric_cols[:-1])
    print(f"{watershed_name} Intermittent Data after cleaning: {len(df_inter)} rows")
    print(f"{watershed_name} Continuous Data after cleaning: {len(df_cont)} rows")
    
    if df_inter.empty:
        print(f"{watershed_name} intermittent data empty after cleaning")
        return None
    
    # Feature engineering (Section 2.3)
    df_inter['MA_Discharge_3'] = df_inter['Discharge'].rolling(window=3, min_periods=1).mean().bfill()
    df_inter['Lag_Discharge'] = df_inter['Discharge'].shift(1).bfill()
    df_inter['Lag_Discharge_3'] = df_inter['Discharge'].shift(3).bfill()
    df_cont['MA_Discharge_3'] = df_cont['Discharge'].rolling(window=3, min_periods=1).mean().bfill()
    df_cont['Lag_Discharge'] = df_cont['Discharge'].shift(1).bfill()
    df_cont['Lag_Discharge_3'] = df_cont['Discharge'].shift(3).bfill()
    
    # Select predictors (Section 3.2, using Discharge instead of Log_Discharge)
    predictors = ['Discharge', 'MA_Discharge_3', 'Lag_Discharge', 'Lag_Discharge_3', 'Rainfall', 'ETo']
    print(f"{watershed_name} Predictors: {predictors}")
    X_inter = df_inter[predictors]
    y_inter = df_inter['SSC']
    X_cont = df_cont[predictors]
    
    # Scale features
    scaler = RobustScaler()
    X_inter_scaled = scaler.fit_transform(X_inter)
    X_cont_scaled = scaler.transform(X_cont)
    
    # Train QRF and predict SSC with quantiles for uncertainty
    try:
        qrf = RandomForestQuantileRegressor(**qrf_params)
        qrf.fit(X_inter_scaled, y_inter)
        ssc_pred = qrf.predict(X_cont_scaled, quantiles=[0.25, 0.5, 0.75])  # Added quantiles for IQR
    except Exception as e:
        print(f"Error training/predicting QRF for {watershed_name}: {str(e)}")
        return None
    
    # Debug: SSC prediction summary
    print(f"{watershed_name} SSC Prediction Summary (g/L):")
    for q, preds in zip([0.25, 0.5, 0.75], ssc_pred.T):
        print(f"Quantile {q}: {pd.Series(preds).describe()}")
    
    # Create output DataFrame with SSC quantiles
    df_cont['SSC_Q25'] = ssc_pred[:, 0]  # 0.25 quantile
    df_cont['SSC_Median'] = ssc_pred[:, 1]  # 0.5 quantile
    df_cont['SSC_Q75'] = ssc_pred[:, 2]  # 0.75 quantile
    return df_cont[['Date', 'Rainfall', 'Discharge', 'Temperature', 'ETo', 'SSC_Q25', 'SSC_Median', 'SSC_Q75']]

def calculate_sediment_yield(df, watershed_name, area_km2, output_dir):
    """
    Calculate daily sediment yield (t/ha/day) and uncertainty, then save to Excel.
    Args:
        df (DataFrame): DataFrame with Date, Rainfall, Discharge, SSC_Q25, SSC_Median, SSC_Q75.
        watershed_name (str): Name of the watershed.
        area_km2 (float): Watershed area in km².
        output_dir (Path): Directory for output files.
    Returns:
        DataFrame with daily sediment yield and uncertainty.
    """
    print(f"\nCalculating sediment yield for {watershed_name}...")
    
    # Create output directory
    output_dir.mkdir(exist_ok=True)
    
    # Filter for 1990–2020
    df = df[(df['Date'].dt.year >= 1990) & (df['Date'].dt.year <= 2020)]
    print(f"{watershed_name} Data after 1990–2020 filter: {len(df)} rows")
    
    # Calculate sediment yield (t/ha/day) and uncertainty
    df['Sediment_Yield_Median'] = df['Discharge'] * df['SSC_Median'] * LOAD_FACTOR / (area_km2 * 100)
    df['Sediment_Yield_Q25'] = df['Discharge'] * df['SSC_Q25'] * LOAD_FACTOR / (area_km2 * 100)
    df['Sediment_Yield_Q75'] = df['Discharge'] * df['SSC_Q75'] * LOAD_FACTOR / (area_km2 * 100)
    
    # Drop invalid data
    df = df.dropna(subset=['Date', 'Rainfall', 'Discharge', 'Sediment_Yield_Median', 'Sediment_Yield_Q25', 'Sediment_Yield_Q75'])
    print(f"{watershed_name} Data after dropping NaNs: {len(df)} rows")
    
    if df.empty:
        print(f"{watershed_name} data empty after cleaning")
        return None
    
    # Save to Excel
    output_path = output_dir / f"{watershed_name.replace(' ', '_')}_Daily_SSC_Sediment_Yield.xlsx"
    df.to_excel(output_path, index=False)
    print(f"Daily data saved to {output_path}")
    
    # Debug: Print sample and years
    print(f"\n{watershed_name} Daily Data Sample:")
    print(df[['Date', 'Rainfall', 'Discharge', 'Sediment_Yield_Median']].head())
    print(f"{watershed_name} Data Shape: {df.shape}")
    print(f"{watershed_name} Years Covered: {df['Date'].dt.year.nunique()} ({df['Date'].dt.year.min()}–{df['Date'].dt.year.max()})")
    
    return df

def process_monthly_data(df, watershed_name):
    """
    Process daily data to compute monthly aggregates with uncertainty.
    Args:
        df (DataFrame): DataFrame with daily data (Date, Rainfall, Discharge, Sediment_Yield_Median, Sediment_Yield_Q25, Sediment_Yield_Q75).
        watershed_name (str): Name of the watershed.
    Returns:
        DataFrame with monthly aggregates.
    """
    print(f"\nProcessing monthly data for {watershed_name}...")
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    
    # Filter for 1990–2020
    df = df[(df['Year'] >= 1990) & (df['Year'] <= 2020)]
    if df.empty:
        print(f"{watershed_name} data empty after year filtering (1990-2020).")
        return None
    
    monthly_data = df.groupby([df['Date'].dt.year, df['Date'].dt.month]).agg({
        'Discharge': 'mean',  # Mean monthly discharge (m³/s)
        'Rainfall': 'sum',    # Sum daily rainfall for monthly total (mm)
        'Sediment_Yield_Median': 'mean',  # Mean daily SY for monthly average (t/ha/day)
        'Sediment_Yield_Q25': 'mean',     # Mean daily Q25 SY
        'Sediment_Yield_Q75': 'mean'      # Mean daily Q75 SY
    })
    
    # Calculate days in each month
    monthly_data['Days_in_Month'] = df.groupby([df['Date'].dt.year, df['Date'].dt.month])['Date'].nunique()
    # Convert mean daily SY to monthly SY (t/ha/month)
    monthly_data['Monthly_Sediment_Yield_tons_ha'] = monthly_data['Sediment_Yield_Median'] * monthly_data['Days_in_Month']
    monthly_data['Monthly_Sediment_Yield_Q25'] = monthly_data['Sediment_Yield_Q25'] * monthly_data['Days_in_Month']
    monthly_data['Monthly_Sediment_Yield_Q75'] = monthly_data['Sediment_Yield_Q75'] * monthly_data['Days_in_Month']
    
    # Set index to first day of each month
    monthly_data.index = pd.to_datetime(monthly_data.index.map(lambda x: f'{x[0]}-{x[1]:02}-01'))
    
    # Debug: Print monthly summary
    print(f"{watershed_name} Monthly Data Summary:")
    print(monthly_data[['Monthly_Sediment_Yield_tons_ha', 'Discharge', 'Rainfall']].describe())
    print(f"{watershed_name} Sample Monthly Data with Uncertainty:")
    print(monthly_data[['Rainfall', 'Discharge', 'Monthly_Sediment_Yield_tons_ha', 'Monthly_Sediment_Yield_Q25', 'Monthly_Sediment_Yield_Q75']].head())
    print(f"{watershed_name} Months Covered: {len(monthly_data)}")
    
    return monthly_data

def create_figure8(monthly_data_dict, output_dir):
    """
    Generate Figure 8: Combination plot with bar plot for monthly rainfall (reversed axis) and line plots for discharge
    and sediment yield with uncertainty bands (Section 3.3), displayed in Jupyter notebook.
    Args:
        monthly_data_dict (dict): Dictionary with monthly data for each watershed.
        output_dir (Path): Directory for output plots.
    Returns:
        None
    """
    print("\nGenerating Figure 8...")
    watersheds = ['Gilgel Abay', 'Gumara']
    missing_watersheds = [ws for ws in watersheds if ws not in monthly_data_dict]
    if missing_watersheds:
        print(f"Error generating Figure 8: Missing data for watersheds: {missing_watersheds}")
        return
    
    plt.figure(figsize=(16, 6))  # Create new figure explicitly
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), sharey=False)
    
    all_lines = []
    all_labels = []
    
    for ax, watershed_name in zip([ax1, ax2], watersheds):
        monthly_data = monthly_data_dict[watershed_name]
        discharge_max = WATERSHED_CONFIG[watershed_name]['discharge_max']
        yield_max = WATERSHED_CONFIG[watershed_name]['yield_max']
        rainfall_max = WATERSHED_CONFIG[watershed_name]['rainfall_max']
        
        # Rainfall bars (right axis, inverted) with fixed range
        ax1_rain = ax.twinx()
        bars = ax1_rain.bar(monthly_data.index, monthly_data['Rainfall'], color='#003300', alpha=0.4, width=25, label='Rainfall (mm)')
        ax1_rain.set_ylim(0, rainfall_max)
        ax1_rain.invert_yaxis()  # Reverse axis
        ax1_rain.set_ylabel('Rainfall (mm)', color='#003300', fontsize=18)
        ax1_rain.yaxis.set_label_position('right')
        ax1_rain.yaxis.tick_right()
        ax1_rain.tick_params(axis='y', labelsize=14)
        
        # Discharge (left axis) with fixed range
        ax2_discharge = ax
        line_discharge = ax2_discharge.plot(monthly_data.index, monthly_data['Discharge'], color='#1f77b4', marker=None,
                                           linestyle='-', linewidth=1.5, label='Discharge (m³/s)')[0]
        ax2_discharge.set_ylim(0, discharge_max)
        ax2_discharge.set_ylabel('Discharge (m³/s)', color='#1f77b4', fontsize=18)
        ax2_discharge.yaxis.set_label_position('left')
        ax2_discharge.yaxis.tick_left()
        ax2_discharge.tick_params(axis='y', labelsize=14)
        
        # Sediment Yield (right axis, offset) with fixed range and uncertainty
        ax3_sediment = ax.twinx()
        ax3_sediment.spines['right'].set_position(('outward', 60))
        line_sediment = ax3_sediment.plot(monthly_data.index, monthly_data['Monthly_Sediment_Yield_tons_ha'],
                                         color='#d62728', marker=None, linestyle='-', linewidth=1.5,
                                         label='Sediment Yield (t/ha/month)')[0]
        ax3_sediment.fill_between(monthly_data.index, monthly_data['Monthly_Sediment_Yield_Q25'], monthly_data['Monthly_Sediment_Yield_Q75'],
                                 color='#d62728', alpha=0.2, label='Uncertainty (IQR)')
        ax3_sediment.set_ylim(0, yield_max)
        ax3_sediment.set_ylabel('Sediment Yield (t/ha/month)', color='#d62728', fontsize=18)
        ax3_sediment.tick_params(axis='y', labelsize=14)
        
        # Set x-axis ticks for years (every 5 years)
        years = range(1990, 2021, 5)
        year_ticks = [pd.to_datetime(f'{year}-01-01') for year in years]
        ax.set_xticks(year_ticks)
        ax.set_xticklabels([year.year for year in year_ticks], fontsize=14)
        ax.set_xlabel('Year', fontsize=18)
        
        ax.set_title(f"({'a' if watershed_name == 'Gilgel Abay' else 'b'}) {watershed_name}", fontsize=20)
        ax.grid(False)
        
        # Debug: Print axis limits
        print(f"{watershed_name} Plot Axis Limits:")
        print(f"  Rainfall: 0 to {rainfall_max:.2f} mm")
        print(f"  Discharge: 0 to {discharge_max:.2f} m³/s")
        print(f"  Sediment Yield: 0 to {yield_max:.2f} t/ha/month")
        print(f"{watershed_name} Months Plotted: {len(monthly_data)}")
        
        if watershed_name == 'Gilgel Abay':
            all_lines.extend([bars, line_discharge, line_sediment])
            all_labels.extend(['Rainfall (mm)', 'Discharge (m³/s)', 'Sediment Yield (t/ha/month)'])
    
    fig.legend(all_lines, all_labels, loc='lower center', bbox_to_anchor=(0.5, -0.04), ncol=3, fontsize=16)
    plt.tight_layout(pad=2.0)
    
    # Save plots
    output_png = output_dir / 'Figure8_Monthly_Sediment_Yield.png'
    output_svg = output_dir / 'Figure8_Monthly_Sediment_Yield.svg'
    plt.savefig(output_png, dpi=600, format='png', bbox_inches='tight')
    plt.savefig(output_svg, format='svg', bbox_inches='tight')
    print(f"Figure 8 saved to {output_png} (PNG) and {output_svg} (SVG)")
    plt.show()  # Display the plot in Jupyter notebook
    plt.close()  # Close figure after display

def main():
    """
    Main function to process data and generate Figure 8.
    """
    print("Starting script execution...")
    monthly_data_dict = {}
    for watershed_name, params in WATERSHED_CONFIG.items():
        print(f"\n=== Processing {watershed_name} ===")
        try:
            # Predict SSC
            df_cont = predict_ssc(
                params['intermittent'],
                params['continuous'],
                watershed_name,
                QRF_PARAMS[watershed_name],
                is_excel_inter=(watershed_name == 'Gilgel Abay')
            )
            if df_cont is None:
                print(f"Skipping {watershed_name} due to data loading issues")
                continue
            
            # Calculate daily sediment yield
            daily_data = calculate_sediment_yield(
                df_cont,
                watershed_name,
                params['area_km2'],
                params['output_dir']
            )
            if daily_data is None:
                print(f"Skipping {watershed_name} due to sediment yield calculation issues")
                continue
            
            # Process monthly data
            monthly_data = process_monthly_data(daily_data, watershed_name)
            if monthly_data is None:
                print(f"Skipping {watershed_name} due to monthly data processing issues")
                continue
            
            monthly_data_dict[watershed_name] = monthly_data
            
            # Save monthly data
            output_csv = params['output_dir'] / f"{watershed_name.replace(' ', '_')}_Monthly_Sediment_Yield.csv"
            monthly_data.to_csv(output_csv)
            print(f"Monthly data saved to {output_csv}")
        
        except Exception as e:
            print(f"Error processing {watershed_name}: {str(e)}")
            continue
    
    if monthly_data_dict:
        try:
            create_figure8(monthly_data_dict, WATERSHED_CONFIG['Gilgel Abay']['output_dir'])
        except Exception as e:
            print(f"Error generating Figure 8: {str(e)}")
    else:
        print("No data processed successfully. Figure 8 cannot be generated.")

if __name__ == "__main__":
    main()
