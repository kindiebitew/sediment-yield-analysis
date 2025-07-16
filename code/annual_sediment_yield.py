# Script to generate Figure 7: Annual sediment yield, rainfall, and discharge for (a) Gilgel Abay and (b) Gumara watersheds
# (Section 3.3). Predicts daily SSC (g/L) for 1990–2020 using Quantile Random Forest (QRF) trained on intermittent data,
# calculates daily sediment yield (t/ha/day), and aggregates
# annually (t/ha/yr). Uncertainty is estimated using the IQR of SSC predictions combined with discharge variability.
# Produces a side-by-side plot with bar plots for annual rainfall (mm, reversed axis) and line plots for discharge
# (m³/s) and sediment yield (t/ha/yr), including uncertainty bands. Outputs include Excel for daily data, CSV for
# annual data, and publication-quality PNG/SVG plots. Note: Upload to GitHub and assign a DOI via Zenodo.
# Author: Kindie B. Worku
# Date: 2025-07-16 

%matplotlib inline
import pandas as pd
import numpy as np
from quantile_forest import RandomForestQuantileRegressor
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

# Define constants
WATERSHED_CONFIG = {
    'Gilgel Abay': {
        'intermittent': Path(r"C:\Users\worku\Documents\sediment-yield-analysis\data\Intermittent_data.xlsx"),
        'continuous': Path(r"C:\Users\worku\Documents\sediment-yield-analysis\data\continuous_data.csv"),
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
    Predict daily SSC (g/L) for 1990–2020 using QRF trained on intermittent data, including cumulative and annual rainfall.
    Args:
        intermittent_path (Path): Path to intermittent data (Excel or CSV).
        continuous_path (Path): Path to continuous data (CSV).
        watershed_name (str): Name of the watershed ('Gilgel Abay' or 'Gumara').
        qrf_params (dict): QRF hyperparameters.
        is_excel_inter (bool): Flag to indicate if intermittent file is Excel (True) or CSV (False).
    Returns:
        DataFrame with continuous data and predicted SSC, including quantiles for uncertainty, or None if processing fails.
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
    
    # Feature engineering (Section 2.3) with annual and cumulative rainfall
    df_inter['Year'] = df_inter['Date'].dt.year
    df_cont['Year'] = df_cont['Date'].dt.year
    
    # Compute Annual Rainfall for intermittent data
    annual_rainfall_inter = df_inter.groupby('Year')['Rainfall'].sum().reset_index()
    annual_rainfall_inter.columns = ['Year', 'Annual_Rainfall']
    df_inter = df_inter.merge(annual_rainfall_inter, on='Year', how='left')
    
    # Compute Cumulative Rainfall for intermittent data
    df_inter = df_inter.sort_values('Date')
    df_inter['Cumulative_Rainfall'] = df_inter['Rainfall'].cumsum()
    
    # Compute Annual Rainfall for continuous data
    annual_rainfall_cont = df_cont.groupby('Year')['Rainfall'].sum().reset_index()
    annual_rainfall_cont.columns = ['Year', 'Annual_Rainfall']
    df_cont = df_cont.merge(annual_rainfall_cont, on='Year', how='left')
    
    # Compute Cumulative Rainfall for continuous data
    df_cont = df_cont.sort_values('Date')
    df_cont['Cumulative_Rainfall'] = df_cont['Rainfall'].cumsum()
    
    # Feature engineering (rolling means and lags using raw Discharge)
    df_inter['MA_Discharge_3'] = df_inter['Discharge'].rolling(window=3, min_periods=1).mean().bfill()
    df_inter['Lag_Discharge'] = df_inter['Discharge'].shift(1).bfill()
    df_inter['Lag_Discharge_3'] = df_inter['Discharge'].shift(3).bfill()
    df_cont['MA_Discharge_3'] = df_cont['Discharge'].rolling(window=3, min_periods=1).mean().bfill()
    df_cont['Lag_Discharge'] = df_cont['Discharge'].shift(1).bfill()
    df_cont['Lag_Discharge_3'] = df_cont['Discharge'].shift(3).bfill()
    
    # Select predictors (Section 3.2, including annual and cumulative rainfall)
    predictors = ['Discharge', 'MA_Discharge_3', 'Lag_Discharge', 'Lag_Discharge_3', 'Rainfall', 'ETo', 'Annual_Rainfall', 'Cumulative_Rainfall']
    X_inter = df_inter[predictors]
    y_inter = df_inter['SSC']
    X_cont = df_cont[predictors]
    
    # Train QRF and predict SSC with quantiles for uncertainty
    try:
        qrf = RandomForestQuantileRegressor(**qrf_params)
        qrf.fit(X_inter, y_inter)  # No scaling
        ssc_pred = qrf.predict(X_cont, quantiles=[0.25, 0.5, 0.75])  # Shape: (n_samples, n_quantiles)
    except Exception as e:
        print(f"Error training/predicting QRF for {watershed_name}: {str(e)}")
        return None
    
    # Debug: SSC prediction summary
    print(f"{watershed_name} SSC Prediction Summary (g/L):")
    for q, preds in zip([0.25, 0.5, 0.75], ssc_pred.T):
        print(f"Quantile {q}: {pd.Series(preds).describe()}")
    
    # Assign SSC quantiles to DataFrame
    df_cont['SSC_Q25'] = ssc_pred[:, 0]  # 0.25 quantile
    df_cont['SSC_Median'] = ssc_pred[:, 1]  # 0.5 quantile
    df_cont['SSC_Q75'] = ssc_pred[:, 2]  # 0.75 quantile
    return df_cont[['Date', 'Rainfall', 'Discharge', 'Temperature', 'ETo', 'SSC_Q25', 'SSC_Median', 'SSC_Q75', 'Annual_Rainfall', 'Cumulative_Rainfall']]

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
    df = df.dropna(subset=['Date', 'Rainfall', 'Discharge', 'Sediment_Yield_Median', 'Sediment_Yield_Q25', 'Sediment_Yield_Q75', 'Annual_Rainfall', 'Cumulative_Rainfall'])
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
    print(df.head(10))
    print(f"{watershed_name} Data Shape: {df.shape}")
    print(f"{watershed_name} Years Covered: {df['Date'].dt.year.nunique()} ({df['Date'].dt.year.min()}–{df['Date'].dt.year.max()})")
    
    return df

def process_annual_data(df, watershed_name):
    """
    Aggregate daily data to annual sediment yield, rainfall, and discharge with uncertainty.
    Args:
        df (DataFrame): DataFrame with daily data (Date, Rainfall, Discharge, Sediment_Yield_Median, Sediment_Yield_Q25, Sediment_Yield_Q75).
        watershed_name (str): Name of the watershed.
    Returns:
        DataFrame with annual aggregates.
    """
    print(f"\nProcessing annual data for {watershed_name}...")
    
    df['Year'] = df['Date'].dt.year
    yearly_data = df.groupby('Year').agg({
        'Discharge': 'mean',
        'Rainfall': 'sum',  # Sum daily rainfall for annual total
        'Sediment_Yield_Median': 'sum',
        'Sediment_Yield_Q25': 'sum',
        'Sediment_Yield_Q75': 'sum'
    })
    
    yearly_data['Days_in_Year'] = df.groupby('Year')['Date'].nunique()
    yearly_data['Annual_Rainfall_mm'] = yearly_data['Rainfall']
    yearly_data['Annual_Sediment_Yield_tons_ha'] = yearly_data['Sediment_Yield_Median']
    yearly_data['Annual_Sediment_Yield_Q25'] = yearly_data['Sediment_Yield_Q25']
    yearly_data['Annual_Sediment_Yield_Q75'] = yearly_data['Sediment_Yield_Q75']
    
    # Debug: Print annual summary
    print(f"{watershed_name} Yearly Data Summary:")
    print(yearly_data[['Annual_Sediment_Yield_tons_ha', 'Discharge', 'Annual_Rainfall_mm']].describe())
    print(f"{watershed_name} Annual Sediment Yield with Uncertainty (t/ha/yr):")
    print(yearly_data[['Annual_Sediment_Yield_tons_ha', 'Annual_Sediment_Yield_Q25', 'Annual_Sediment_Yield_Q75']])
    print(f"{watershed_name} Years in Annual Data: {len(yearly_data)} ({yearly_data.index.min()}–{yearly_data.index.max()})")
    
    return yearly_data

def create_figure7(yearly_data_dict, output_dir):
    """
    Generate Figure 7: Bar plot for annual rainfall (reversed axis) and line plots for discharge
    and sediment yield for both watersheds, including uncertainty bands (Section 3.3).
    Args:
        yearly_data_dict (dict): Dictionary with annual data for each watershed.
        output_dir (Path): Directory for output plots.
    Returns:
        None
    """
    print("\nGenerating Figure 7...")
    
    watersheds = ['Gilgel Abay', 'Gumara']
    missing_watersheds = [w for w in watersheds if w not in yearly_data_dict]
    if missing_watersheds:
        print(f"Error generating Figure 7: Missing data for watersheds: {missing_watersheds}")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey='row')
    fig.patch.set_facecolor('white')
    
    for idx, watershed_name in enumerate(watersheds):
        ax1_rain = axes[idx]
        ax1_rain.set_facecolor('white')
        yearly_data = yearly_data_dict[watershed_name]
        
        discharge_max = WATERSHED_CONFIG[watershed_name]['discharge_max']
        yield_max = WATERSHED_CONFIG[watershed_name]['yield_max']
        
        max_rainfall = yearly_data['Annual_Rainfall_mm'].max() * 1.1 if yearly_data['Annual_Rainfall_mm'].max() > 0 else 1000
        reversed_rainfall = max_rainfall - yearly_data['Annual_Rainfall_mm']
        
        bars = ax1_rain.bar(yearly_data.index, reversed_rainfall, color='#2ca02c', alpha=0.7, width=0.4, label='Rainfall (mm)')
        
        ax2_discharge = ax1_rain.twinx()
        ax3_sediment = ax1_rain.twinx()
        ax3_sediment.spines['right'].set_position(('outward', 60))
        
        line_discharge = ax2_discharge.plot(yearly_data.index, yearly_data['Discharge'], color='#1f77b4', marker='o',
                                           linestyle='-', linewidth=1.5, markersize=6, label='Discharge (m³/s)')[0]
        line_sediment = ax3_sediment.plot(yearly_data.index, yearly_data['Annual_Sediment_Yield_tons_ha'],
                                          color='#d62728', marker='s', linestyle='-', linewidth=1.5, markersize=6,
                                          label='Sediment Yield (t/ha/yr)')[0]
        # Add uncertainty bands (IQR)
        ax3_sediment.fill_between(yearly_data.index, yearly_data['Annual_Sediment_Yield_Q25'], yearly_data['Annual_Sediment_Yield_Q75'],
                                  color='#d62728', alpha=0.2, label='Uncertainty (IQR)')
        
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
        
        print(f"{watershed_name} Plot Axis Limits:")
        print(f"  Rainfall: 0 to {max_rainfall:.2f} mm")
        print(f"  Discharge: 0 to {discharge_max:.2f} m³/s")
        print(f"  Sediment Yield: 0 to {yield_max:.2f} t/ha/yr")
        print(f"{watershed_name} Years Plotted: {len(yearly_data)} ({yearly_data.index.min()}–{yearly_data.index.max()})")
    
    fig.legend([bars, line_discharge, line_sediment], ['Rainfall (mm)', 'Discharge (m³/s)', 'Sediment Yield (t/ha/yr)'],
              loc='lower center', bbox_to_anchor=(0.5, -0.04), ncol=3, fontsize=16)
    plt.tight_layout(pad=2.0)
    
    output_png = output_dir / 'Figure7_Annual_Sediment_Yield.png'
    output_svg = output_dir / 'Figure7_Annual_Sediment_Yield.svg'
    plt.savefig(output_png, dpi=600, format='png', bbox_inches='tight')
    plt.savefig(output_svg, format='svg', bbox_inches='tight')
    plt.show()

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
create_figure7(yearly_data_dict, WATERSHED_CONFIG['Gumara']['output_dir'])
