#Script to generate Figure 7: Annual sediment yield, rainfall, and discharge for Gilgel Abay and Gumara watersheds
#(Section 3.3). Predicts daily SSC (g/L) for 1990–2020 using Quantile Random Forest (QRF) trained on intermittent data,
#calculates daily sediment yield (t/ha/day), and aggregates annually (t/ha/yr). Produces a combination plot with a bar
#plot for annual rainfall (mm, reversed axis) and line plots for discharge (m³/s) and sediment yield (t/ha/yr).
#Outputs include Excel for daily data, CSV for annual data, and publication-quality PNG/SVG plots.
#Author: Kindie B. Worku
#Date: 2025-07-07

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
        'continuous': Path(r"C:\Users\worku\Documents\sediment-yield-analysis\data\continuous_data.xlsx"),
        'area_km2': 1664,
        'discharge_max': 180,
        'yield_max': 80,
        'output_dir': Path(r"C:\Users\worku\Documents\sediment-yield-analysis\outputs")
    },
    'Gumara': {
        'intermittent': Path(r"C:\Users\worku\Documents\sediment-yield-analysis\data\Intermittent_data_gum.csv"),
        'continuous': Path(r"C:\Users\worku\Documents\sediment-yield-analysis\data\continuous_data_gum.csv"),
        'area_km2': 1394,
        'discharge_max': 120,
        'yield_max': 80,
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
        DataFrame with continuous data and predicted SSC, or None if processing fails.
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
    df_inter['Log_Discharge'] = np.log1p(df_inter['Discharge'].clip(lower=0))
    df_inter['MA_Discharge_3'] = df_inter['Discharge'].rolling(window=3, min_periods=1).mean().bfill()
    df_inter['Lag_Discharge'] = df_inter['Discharge'].shift(1).bfill()
    df_inter['Lag_Discharge_3'] = df_inter['Discharge'].shift(3).bfill()
    df_cont['Log_Discharge'] = np.log1p(df_cont['Discharge'].clip(lower=0))
    df_cont['MA_Discharge_3'] = df_cont['Discharge'].rolling(window=3, min_periods=1).mean().bfill()
    df_cont['Lag_Discharge'] = df_cont['Discharge'].shift(1).bfill()
    df_cont['Lag_Discharge_3'] = df_cont['Discharge'].shift(3).bfill()
    
    # Select predictors (Section 3.2)
    predictors = ['Log_Discharge', 'MA_Discharge_3', 'Lag_Discharge', 'Lag_Discharge_3', 'Rainfall', 'ETo']
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
    print(df.head(10))
    print(f"{watershed_name} Data Shape: {df.shape}")
    
    return df

def process_annual_data(df, watershed_name):
    """
    Aggregate daily data to annual sediment yield, rainfall, and discharge.
    Args:
        df (DataFrame): DataFrame with daily data (Date, Rainfall, Discharge, Sediment_Yield).
        watershed_name (str): Name of the watershed.
    Returns:
        DataFrame with annual aggregates.
    """
    print(f"Processing annual data for {watershed_name}...")
    
    df['Year'] = df['Date'].dt.year
    yearly_data = df.groupby('Year').agg({
        'Discharge': 'mean',
        'Rainfall': 'sum',  # Sum daily rainfall for annual total
        'Sediment_Yield': 'sum'  # Sum daily SY for annual total (t/ha/yr)
    })
    
    yearly_data['Days_in_Year'] = df.groupby('Year')['Date'].nunique()
    yearly_data['Annual_Rainfall_mm'] = yearly_data['Rainfall']
    yearly_data['Annual_Sediment_Yield_tons_ha'] = yearly_data['Sediment_Yield']
    
    # Debug: Print annual summary
    print(f"{watershed_name} Yearly Data Summary:")
    print(yearly_data[['Annual_Sediment_Yield_tons_ha', 'Discharge', 'Annual_Rainfall_mm']].describe())
    print(f"{watershed_name} Annual Sediment Yield (t/ha/yr):")
    print(yearly_data['Annual_Sediment_Yield_tons_ha'])
    
    return yearly_data

def create_figure7(yearly_data_dict, output_dir):
    """
    Generate Figure 7: Combination plot with bar plot for annual rainfall (reversed axis) and line plots for discharge
    and sediment yield (Section 3.3).
    Args:
        yearly_data_dict (dict): Dictionary with annual data for each watershed.
        output_dir (Path): Directory for output plots.
    """
    print("Generating Figure 7...")
    
    if not all(ws in yearly_data_dict for ws in ['Gilgel Abay', 'Gumara']):
        missing = [ws for ws in ['Gilgel Abay', 'Gumara'] if ws not in yearly_data_dict]
        raise ValueError(f"Missing data for watersheds: {missing}")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), sharey=False)
    fig.patch.set_facecolor('white')
    
    all_lines = []
    all_labels = []
    
    for ax, watershed_name in zip([ax1, ax2], ['Gilgel Abay', 'Gumara']):
        yearly_data = yearly_data_dict[watershed_name]
        discharge_max = WATERSHED_CONFIG[watershed_name]['discharge_max']
        yield_max = WATERSHED_CONFIG[watershed_name]['yield_max']
        
        # Handle rainfall axis (reversed)
        max_rainfall = yearly_data['Annual_Rainfall_mm'].max() * 1.1 if yearly_data['Annual_Rainfall_mm'].max() > 0 else 1000
        reversed_rainfall = max_rainfall - yearly_data['Annual_Rainfall_mm']
        
        ax1_rain = ax
        ax1_rain.set_facecolor('white')
        bars = ax1_rain.bar(yearly_data.index, reversed_rainfall, color='#2ca02c', alpha=0.7, width=0.4, label='Rainfall (mm)')
        
        ax2_discharge = ax1_rain.twinx()
        ax3_sediment = ax1_rain.twinx()
        ax3_sediment.spines['right'].set_position(('outward', 60))
        
        line_discharge = ax2_discharge.plot(yearly_data.index, yearly_data['Discharge'], color='#1f77b4', marker='o',
                                           linestyle='-', linewidth=1.5, markersize=6, label='Discharge (m³/s)')[0]
        line_sediment = ax3_sediment.plot(yearly_data.index, yearly_data['Annual_Sediment_Yield_tons_ha'],
                                          color='#d62728', marker='s', linestyle='-', linewidth=1.5, markersize=6,
                                          label='Sediment Yield (t/ha/yr)')[0]
        
        ax1_rain.set_title(f"({'a' if watershed_name == 'Gilgel Abay' else 'b'}) {watershed_name}", fontsize=20)
        ax1_rain.set_xlabel('Year', fontsize=18)
        ax1_rain.set_ylabel('Rainfall (mm)', color='#2ca02c', fontsize=18)
        ax2_discharge.set_ylabel('Discharge (m³/s)', color='#1f77b4', fontsize=18)
        ax3_sediment.set_ylabel('Sediment Yield (t/ha/yr)', color='#d62728', fontsize=18)
        
        ax1_rain.yaxis.set_label_position('right')
        ax1_rain.yaxis.tick_right()
        ax2_discharge.yaxis.set_label_position('left')
        ax2_discharge.yaxis.tick_left()
        ax3_sediment.yaxis.set_label_position('right')
        ax3_sediment.yaxis.tick_right()
        
        ax1_rain.set_ylim(max_rainfall, 0)
        ax2_discharge.set_ylim(0, discharge_max)
        ax3_sediment.set_ylim(0, yield_max)
        
        ax1_rain.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax1_rain.grid(False)
        
        # Debug: Print axis limits
        print(f"{watershed_name} Plot Axis Limits:")
        print(f"  Rainfall: 0 to {max_rainfall:.2f} mm")
        print(f"  Discharge: 0 to {discharge_max:.2f} m³/s")
        print(f"  Sediment Yield: 0 to {yield_max:.2f} t/ha/yr")
        
        if watershed_name == 'Gilgel Abay':
            all_lines.extend([bars, line_discharge, line_sediment])
            all_labels.extend(['Rainfall (mm)', 'Discharge (m³/s)', 'Sediment Yield (t/ha/yr)'])
    
    fig.legend(all_lines, all_labels, loc='lower center', bbox_to_anchor=(0.5, -0.04), ncol=3, fontsize=16)
    plt.tight_layout(pad=2.0)
    
    # Save plots
    output_png = output_dir / 'Figure7_Annual_Sediment_Yield.png'
    output_svg = output_dir / 'Figure7_Annual_Sediment_Yield.svg'
    plt.savefig(output_png, dpi=600, format='png', bbox_inches='tight')
    plt.savefig(output_svg, format='svg', bbox_inches='tight')
    print(f"Figure 7 saved to {output_png} (PNG) and {output_svg} (SVG)")
    plt.close()

# Main processing loop
yearly_data_dict = {}
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
        
        # Process annual data
        yearly_data = process_annual_data(daily_data, watershed_name)
        yearly_data = yearly_data[yearly_data.index >= 1990]
        
        # Save annual data
        output_csv = params['output_dir'] / f"{watershed_name.replace(' ', '_')}_Yearly_Sediment_Yield.csv"
        yearly_data.to_csv(output_csv)
        print(f"Annual data saved to {output_csv}")
        
        yearly_data_dict[watershed_name] = yearly_data
    
    except Exception as e:
        print(f"Error processing {watershed_name}: {str(e)}")
        continue
# Generate Figure 7
try:
    create_figure7(yearly_data_dict, WATERSHED_CONFIG['Gilgel Abay']['output_dir'])
except ValueError as e:
    print(f"Error generating Figure 7: {str(e)}")
