# Script to generate Figure 8: Monthly sediment yield, rainfall, and discharge for Gilgel Abay and Gumara watersheds
# (Section 3.3). Predicts daily SSC (g/L) for 1990–2020 using Quantile Random Forest (QRF) trained on intermittent data,
# calculates daily sediment yield (t/ha/day), aggregates to monthly values, and produces a combination plot with a bar
# plot for monthly rainfall (mm, reversed axis) and line plots for discharge (m³/s) and sediment yield (t/ha/month).
# Author: Kindie B. Worku
# Date: 2025-07-16

import pandas as pd
import numpy as np
from quantile_forest import RandomForestQuantileRegressor
from sklearn.preprocessing import RobustScaler
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from matplotlib.ticker import MaxNLocator
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Set plot style for publication quality (Section 3.3)
sns.set_style('white')
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 16

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
    Predict daily SSC (g/L) for 1990–2020 using QRF trained on intermittent data.
    Args:
        intermittent_path (Path): Path to intermittent data (Excel or CSV).
        continuous_path (Path): Path to continuous data (Excel or CSV).
        watershed_name (str): Name of the watershed ('Gilgel Abay' or 'Gumara').
        qrf_params (dict): QRF hyperparameters.
        is_excel_inter (bool): Flag to indicate if intermittent file is Excel (True) or CSV (False).
    Returns:
        DataFrame with continuous data and predicted SSC.
    """
    print(f"Predicting SSC for {watershed_name}...")
    
    # Check file existence
    if not intermittent_path.exists():
        raise FileNotFoundError(f"Intermittent file not found: {intermittent_path}")
    if not continuous_path.exists():
        raise FileNotFoundError(f"Continuous file not found: {continuous_path}")
    
    # Load data
    try:
        if is_excel_inter:
            df_inter = pd.read_excel(intermittent_path, engine='openpyxl')
        else:
            df_inter = pd.read_csv(intermittent_path)
        df_cont = pd.read_csv(continuous_path) if continuous_path.suffix == '.csv' else pd.read_excel(continuous_path, engine='openpyxl')
        print(f"{watershed_name} Intermittent Data Shape: {df_inter.shape}")
        print(f"{watershed_name} Continuous Data Shape: {df_cont.shape}")
    except Exception as e:
        raise ValueError(f"Error loading data for {watershed_name}: {str(e)}")
    
    # Validate columns
    required_cols_inter = ['Date', 'Rainfall', 'Discharge', 'Temperature', 'ETo', 'SSC']
    required_cols_cont = ['Date', 'Rainfall', 'Discharge', 'Temperature', 'ETo']
    if not all(col in df_inter.columns for col in required_cols_inter):
        raise ValueError(f"{watershed_name} intermittent data missing columns: {required_cols_inter}")
    if not all(col in df_cont.columns for col in required_cols_cont):
        raise ValueError(f"{watershed_name} continuous data missing columns: {required_cols_cont}")
    
    # Convert dates to datetime
    df_inter['Date'] = pd.to_datetime(df_inter['Date'], errors='coerce')
    df_cont['Date'] = pd.to_datetime(df_cont['Date'], errors='coerce')
    df_inter = df_inter.dropna(subset=['Date'])
    df_cont = df_cont.dropna(subset=['Date'])
    
    if df_inter.empty or df_cont.empty:
        raise ValueError(f"{watershed_name} data empty after date cleaning")
    
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
    
    if df_inter.empty:
        raise ValueError(f"{watershed_name} intermittent data empty after cleaning")
    
    # Feature engineering (Section 2.3)
    df_inter['MA_Discharge_3'] = df_inter['Discharge'].rolling(window=3, min_periods=1).mean().bfill()
    df_inter['Lag_Discharge'] = df_inter['Discharge'].shift(1).bfill()
    df_inter['Lag_Discharge_3'] = df_inter['Discharge'].shift(3).bfill()
    df_cont['MA_Discharge_3'] = df_cont['Discharge'].rolling(window=3, min_periods=1).mean().bfill()
    df_cont['Lag_Discharge'] = df_cont['Discharge'].shift(1).bfill()
    df_cont['Lag_Discharge_3'] = df_cont['Discharge'].shift(3).bfill()
    
    # Select predictors (Section 3.2)
    predictors = ['Discharge', 'MA_Discharge_3', 'Lag_Discharge', 'Lag_Discharge_3', 'Rainfall', 'ETo']
    X_inter = df_inter[predictors]
    y_inter = df_inter['SSC']
    X_cont = df_cont[predictors]
    
    # Scale features
    scaler = RobustScaler()
    X_inter_scaled = scaler.fit_transform(X_inter)
    X_cont_scaled = scaler.transform(X_cont)
    
    # Train QRF and predict SSC
    try:
        qrf = RandomForestQuantileRegressor(**qrf_params)
        qrf.fit(X_inter_scaled, y_inter)
        ssc_pred = qrf.predict(X_cont_scaled, quantiles=0.5)
    except Exception as e:
        raise ValueError(f"Error training/predicting QRF for {watershed_name}: {str(e)}")
    
    # Debug: SSC prediction summary
    print(f"{watershed_name} SSC Prediction Summary (g/L):")
    print(pd.Series(ssc_pred).describe())
    
    # Create output DataFrame
    df_cont['SSC'] = ssc_pred
    return df_cont[['Date', 'Rainfall', 'Discharge', 'Temperature', 'ETo', 'SSC']]

def calculate_sediment_yield(df, watershed_name, area_km2, output_dir):
    """
    Calculate daily sediment yield (t/ha/day) and save to Excel.
    Args:
        df (DataFrame): DataFrame with Date, Rainfall, Discharge, and SSC.
        watershed_name (str): Name of the watershed.
        area_km2 (float): Watershed area in km².
        output_dir (Path): Directory for output files.
    Returns:
        DataFrame with daily sediment yield.
    """
    print(f"Calculating sediment yield for {watershed_name}...")
    
    # Create output directory
    output_dir.mkdir(exist_ok=True)
    
    # Filter for 1990–2020
    df = df[(df['Date'].dt.year >= 1990) & (df['Date'].dt.year <= 2020)]
    
    # Calculate sediment yield (t/ha/day)
    df['Sediment_Yield'] = df['Discharge'] * df['SSC'] * LOAD_FACTOR / (area_km2 * 100)
    
    # Drop invalid data
    df = df.dropna(subset=['Date', 'Rainfall', 'Discharge', 'SSC', 'Sediment_Yield'])
    
    if df.empty:
        raise ValueError(f"{watershed_name} data empty after cleaning")
    
    # Save to Excel
    output_path = output_dir / f"{watershed_name.replace(' ', '_')}_Daily_SSC_Sediment_Yield.xlsx"
    df.to_excel(output_path, index=False)
    print(f"Daily data saved to {output_path}")
    
    # Debug: Print sample
    print(f"\n{watershed_name} Daily Data Sample:")
    print(df[['Date', 'Rainfall', 'Discharge', 'Sediment_Yield']].head())
    print(f"{watershed_name} Data Shape: {df.shape}")
    
    return df

def process_monthly_data(df, watershed_name):
    """
    Process daily data to compute monthly aggregates.
    Args:
        df (DataFrame): DataFrame with daily data (Date, Rainfall, Discharge, Sediment_Yield).
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
        raise ValueError(f"{watershed_name} data empty after year filtering (1990-2020).")
    
    monthly_data = df.groupby([df['Date'].dt.year, df['Date'].dt.month]).agg({
        'Discharge': 'mean',  # Mean monthly discharge (m³/s)
        'Rainfall': 'sum',    # Sum daily rainfall for monthly total (mm)
        'Sediment_Yield': 'mean'  # Mean daily SY for monthly average (t/ha/day)
    })
    
    # Calculate days in each month
    monthly_data['Days_in_Month'] = df.groupby([df['Date'].dt.year, df['Date'].dt.month])['Date'].nunique()
    # Convert mean daily SY (t/ha/day) to monthly SY (t/ha/month)
    monthly_data['Monthly_Sediment_Yield_tons_ha'] = monthly_data['Sediment_Yield'] * monthly_data['Days_in_Month']
    
    # Set index to first day of each month
    monthly_data.index = pd.to_datetime(monthly_data.index.map(lambda x: f'{x[0]}-{x[1]:02}-01'))
    
    print(f"{watershed_name} Monthly Data Summary:")
    print(monthly_data[['Monthly_Sediment_Yield_tons_ha', 'Discharge', 'Rainfall']].describe())
    print(f"{watershed_name} Sample Monthly Data:")
    print(monthly_data[['Rainfall', 'Discharge', 'Monthly_Sediment_Yield_tons_ha']].head())
    
    return monthly_data

def create_figure8(monthly_data_dict, output_dir):
    """
    Generate Figure 8: Combination plot with bar plot for monthly rainfall (reversed axis) and line plots for discharge
    and sediment yield (Section 3.3).
    Args:
        monthly_data_dict (dict): Dictionary with monthly data for each watershed.
        output_dir (Path): Directory for output plots.
    """
    print("\nGenerating Figure 8...")
    if not all(ws in monthly_data_dict for ws in ['Gilgel Abay', 'Gumara']):
        missing = [ws for ws in ['Gilgel Abay', 'Gumara'] if ws not in monthly_data_dict]
        raise ValueError(f"Missing data for watersheds: {missing}")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), sharey=False)
    fig.patch.set_facecolor('white')
    
    all_lines = []
    all_labels = []
    
    for ax, watershed_name in zip([ax1, ax2], ['Gilgel Abay', 'Gumara']):
        monthly_data = monthly_data_dict[watershed_name]
        discharge_max = WATERSHED_CONFIG[watershed_name]['discharge_max']
        yield_max = WATERSHED_CONFIG[watershed_name]['yield_max']
        rainfall_max = WATERSHED_CONFIG[watershed_name]['rainfall_max']
        
        ax.set_facecolor('white')
        
        # Rainfall bars (right axis, inverted)
        ax1_rain = ax.twinx()
        max_rainfall = max(monthly_data['Rainfall'].max() * 1.2, rainfall_max)
        reversed_rainfall = max_rainfall - monthly_data['Rainfall']
        bars = ax1_rain.bar(monthly_data.index, reversed_rainfall, color='#2ca02c', alpha=0.7, width=15, label='Rainfall (mm)')
        ax1_rain.set_ylim(max_rainfall * 1.5, 0)
        ax1_rain.set_ylabel('Rainfall (mm)', color='#2ca02c', fontsize=18)
        ax1_rain.yaxis.set_label_position('right')
        ax1_rain.yaxis.tick_right()
        ax1_rain.tick_params(axis='y', labelsize=14)
        
        # Discharge (left axis)
        ax2_discharge = ax
        line_discharge = ax2_discharge.plot(monthly_data.index, monthly_data['Discharge'], color='#1f77b4', marker=None,
                                           linestyle='-', linewidth=1.5, label='Discharge (m³/s)')[0]
        ax2_discharge.set_ylim(0, discharge_max)
        ax2_discharge.set_ylabel('Discharge (m³/s)', color='#1f77b4', fontsize=18)
        ax2_discharge.yaxis.set_label_position('left')
        ax2_discharge.yaxis.tick_left()
        ax2_discharge.tick_params(axis='y', labelsize=14)
        
        # Sediment Yield (right axis, offset)
        ax3_sediment = ax.twinx()
        ax3_sediment.spines['right'].set_position(('outward', 60))
        line_sediment = ax3_sediment.plot(monthly_data.index, monthly_data['Monthly_Sediment_Yield_tons_ha'],
                                          color='#d62728', marker=None, linestyle='-', linewidth=1.5,
                                          label='Sediment Yield (t/ha/month)')[0]
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
        print(f"  Rainfall: 0 to {max_rainfall:.2f} mm")
        print(f"  Discharge: 0 to {discharge_max:.2f} m³/s")
        print(f"  Sediment Yield: 0 to {yield_max:.2f} t/ha/month")
        
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
    plt.show()  # Display the plot in Jupyter notebook
    plt.close()

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
            
            # Calculate daily sediment yield
            daily_data = calculate_sediment_yield(
                df_cont,
                watershed_name,
                params['area_km2'],
                params['output_dir']
            )
            
            # Process monthly data
            monthly_data = process_monthly_data(daily_data, watershed_name)
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
