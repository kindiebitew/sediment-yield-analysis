# Script for calculating monthly sediment yields in Gilgel Abay and Gumara watersheds
# Purpose: Reconstructs sedigraphs (1990–2020) using Quantile Random Forest (QRF) and estimates monthly sediment yields (tonnes/ha/month) for Figure 8 in the paper "Machine Learning-Based
# Sedigraph Reconstruction for Enhanced Sediment Yield Estimation in the Upper Blue Nile Basin."
# Author: Kindie B. Worku and co-authors
# Data: Intermittent SSC from MoWE/ABAO, continuous hydrological data from EMI
# Output: Monthly sediment yield CSV files and Figure 8 for each watershed

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from quantile_forest import RandomForestQuantileRegressor
from sklearn.preprocessing import RobustScaler
import os

# Set plot style for publication quality (Times New Roman, font size 18)
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 18

# Define watershed parameters and file paths
# Note: Paths are placeholders; actual data stored locally due to MoWE/ABAO restrictions
# Area (km² and ha) from watershed delineations; max values set for plot scaling based on data ranges
data_paths = {
    'Gilgel Abay': {
        'intermittent': r"D:\Gilgel Abay\Sedigrapgh\Intermittent_data.xlsx",
        'continuous': r"D:\Gilgel Abay\Sedigrapgh\continuous_data.csv",
        'output_csv': r"D:\Gilgel Abay\Sedigrapgh\Gilgel_Abay_Monthly_Data_ha_QRF.csv",
        'output_plot': r"D:\Gilgel Abay\Sedigrapgh\Gilgel_Abay_Monthly_Plot_ha_QRF.png",
        'area_km2': 1664,  # Watershed area for reference
        'area_ha': 1664 * 100,  # Convert km² to hectares for yield normalization
        'discharge_max': 800,  # Max discharge for plot scaling
        'yield_max': 25  # Max sediment yield (tonnes/ha/month) for plot scaling
    },
    'Gumara': {
        'intermittent': r"D:\Gumara\Sedigrapgh\Intermittent_data_gum.csv",
        'continuous': r"D:\Gumara\Sedigrapgh\continious_data_gum.csv",
        'output_csv': r"D:\Gumara\Sedigrapgh\Gumara_Monthly_Data_ha_QRF.csv",
        'output_plot': r"D:\Gumara\Sedigrapgh\Gumara_Monthly_Plot_ha_QRF.png",
        'area_km2': 1394,  # Watershed area for reference
        'area_ha': 1394 * 100,  # Convert km² to hectares for yield normalization
        'discharge_max': 700,  # Max discharge for plot scaling
        'yield_max': 25  # Max sediment yield (tonnes/ha/month) for plot scaling
    }
}

# QRF hyperparameters tuned via RandomizedSearchCV (paper Section 2.3)
# 1000 trees balance accuracy and computation; max_depth=30 captures non-linear dynamics
# min_samples_split/leaf and max_features='log2' prevent overfitting in sparse SSC data
qrf_params = {
    'Gilgel Abay': {
        'n_estimators': 1000,
        'min_samples_split': 5,
        'min_samples_leaf': 2,
        'max_features': 'log2',
        'max_depth': 30
    },
    'Gumara': {
        'n_estimators': 1000,
        'min_samples_split': 5,
        'min_samples_leaf': 2,
        'max_features': 'log2',
        'max_depth': 30
    }
}

def predict_ssc(intermittent_path, continuous_path, watershed_name, qrf_params):
    """Predict suspended sediment concentration (SSC) using QRF model.
    
    Args:
        intermittent_path (str): Path to intermittent SSC data (Excel/CSV).
        continuous_path (str): Path to continuous hydrological data (CSV).
        watershed_name (str): Name of watershed (Gilgel Abay or Gumara).
        qrf_params (dict): QRF hyperparameters.
    
    Returns:
        tuple: Dates and predicted SSC values for continuous data.
    """
    print(f"Predicting SSC for {watershed_name}...")
    
    # Load data, handling Excel or CSV formats
    if not os.path.exists(intermittent_path):
        raise FileNotFoundError(f"Intermittent data file not found: {intermittent_path}")
    
    try:
        if intermittent_path.endswith('.xlsx'):
            df_inter = pd.read_excel(intermittent_path, engine='openpyxl')
        else:
            df_inter = pd.read_csv(intermittent_path)
        df_cont = pd.read_csv(continuous_path)
    except Exception as e:
        raise ValueError(f"Error loading data for {watershed_name}: {str(e)}")
    
    # Validate required columns (see paper Section 2.2 for data description)
    required_cols_inter = ['Date', 'Rainfall', 'Discharge', 'Temperature', 'ETo', 'SSC']
    required_cols_cont = ['Date', 'Rainfall', 'Discharge', 'Temperature', 'ETo']
    missing_cols_inter = [col for col in required_cols_inter if col not in df_inter.columns]
    missing_cols_cont = [col for col in required_cols_cont if col not in df_cont.columns]
    if missing_cols_inter:
        raise ValueError(f"{watershed_name} intermittent data missing columns: {missing_cols_inter}")
    if missing_cols_cont:
        raise ValueError(f"{watershed_name} continuous data missing columns: {missing_cols_cont}")
    
    # Convert dates to datetime, dropping invalid entries
    df_inter['Date'] = pd.to_datetime(df_inter['Date'], errors='coerce')
    df_cont['Date'] = pd.to_datetime(df_cont['Date'], errors='coerce')
    
    initial_rows_inter = len(df_inter)
    df_inter = df_inter.dropna(subset=['Date'])
    if len(df_inter) < initial_rows_inter:
        print(f"{watershed_name} Dropped {initial_rows_inter - len(df_inter)} rows with invalid dates in intermittent data.")
    
    initial_rows_cont = len(df_cont)
    df_cont = df_cont.dropna(subset=['Date'])
    if len(df_cont) < initial_rows_cont:
        print(f"{watershed_name} Dropped {initial_rows_cont - len(df_cont)} rows with invalid dates in continuous data.")
    
    if df_inter.empty:
        raise ValueError(f"{watershed_name} intermittent data is empty after date parsing.")
    
    # Ensure numeric data types for predictors and SSC
    numeric_cols = ['Rainfall', 'Discharge', 'Temperature', 'ETo', 'SSC']
    for col in numeric_cols:
        df_inter[col] = pd.to_numeric(df_inter[col], errors='coerce')
    for col in numeric_cols[:-1]:
        df_cont[col] = pd.to_numeric(df_cont[col], errors='coerce')
    
    # Drop rows with missing numeric values
    df_inter = df_inter.dropna(subset=numeric_cols)
    df_cont = df_cont.dropna(subset=numeric_cols[:-1])
    
    print(f"{watershed_name} Intermittent Data Shape after Cleaning: {df_inter.shape}")
    if df_inter.empty:
        raise ValueError(f"{watershed_name} intermittent data is empty after cleaning.")
    
    # Add derived features (see paper Section 2.3 for predictor selection rationale)
    # Log_Discharge linearizes flow-SSC relationship; Discharge_Rainfall captures interaction
    # Lag_Discharge accounts for runoff delays
    df_inter['Log_Discharge'] = np.log1p(df_inter['Discharge'].clip(lower=0))
    df_inter['Discharge_Rainfall'] = df_inter['Discharge'] * df_inter['Rainfall']
    df_inter['Lag_Discharge'] = df_inter['Discharge'].shift(1).bfill()
    
    df_cont['Log_Discharge'] = np.log1p(df_cont['Discharge'].clip(lower=0))
    df_cont['Discharge_Rainfall'] = df_cont['Discharge'] * df_cont['Rainfall']
    df_cont['Lag_Discharge'] = df_cont['Discharge'].shift(1).bfill()
    
    # Select predictors based on physical relevance (paper Section 3.2)
    predictors = ['Log_Discharge', 'Rainfall', 'Temperature', 'ETo', 'Discharge_Rainfall', 'Lag_Discharge']
    X_inter = df_inter[predictors]
    y_inter = df_inter['SSC']
    X_cont = df_cont[predictors]
    
    if X_inter.shape[0] == 0:
        raise ValueError(f"{watershed_name} feature matrix X_inter is empty.")
    
    # Scale features using RobustScaler to handle outliers (paper Section 2.3)
    scaler = RobustScaler()
    X_inter_scaled = scaler.fit_transform(X_inter)
    X_cont_scaled = scaler.transform(X_cont)
    
    # Train QRF model with median quantile (0.5) for SSC prediction
    qrf = RandomForestQuantileRegressor(**qrf_params, random_state=42)
    qrf.fit(X_inter_scaled, y_inter)
    ssc_pred = qrf.predict(X_cont_scaled, quantiles=0.5)
    
    return df_cont['Date'], ssc_pred

def process_monthly_data(dates, discharge, rainfall, ssc_pred, area_ha, watershed_name):
    """Process daily data to compute monthly sediment yields.
    
    Args:
        dates (Series): Datetime series for continuous data.
        discharge (Series): Daily discharge (m³/s).
        rainfall (Series): Daily rainfall (mm).
        ssc_pred (array): Predicted SSC (g/L).
        area_ha (float): Watershed area (hectares).
        watershed_name (str): Name of watershed.
    
    Returns:
        DataFrame: Monthly metrics (sediment yield, discharge, rainfall).
    """
    print(f"Processing monthly data for {watershed_name}...")
    
    # Create daily DataFrame, ensuring numeric types
    df = pd.DataFrame({
        'Date': dates,
        'Discharge': pd.to_numeric(discharge, errors='coerce'),
        'Rainfall': pd.to_numeric(rainfall, errors='coerce'),
        'SSC_predicted': pd.to_numeric(ssc_pred, errors='coerce')
    })
    
    # Drop rows with missing values
    initial_rows = len(df)
    df = df.dropna(subset=['Date', 'Discharge', 'Rainfall', 'SSC_predicted'])
    if len(df) < initial_rows:
        print(f"{watershed_name} Dropped {initial_rows - len(df)} rows with invalid or missing data.")
    
    if df.empty:
        raise ValueError(f"{watershed_name} daily data is empty after cleaning.")
    
    # Extract year and month for grouping
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    
    # Filter for 1990–2020 (paper scope)
    df = df[(df['Year'] >= 1990) & (df['Year'] <= 2020)]
    
    # Calculate daily sediment load (tonnes/day) using Equation 3 (paper Section 2.4)
    # 0.0864 converts g/s to tonnes/day (seconds/day ÷ 10^6 g/tonne)
    df['Sediment_Load_tonnes_day'] = df['Discharge'] * df['SSC_predicted'] * 0.0864
    
    # Aggregate to monthly metrics
    monthly_data = df.groupby([df['Date'].dt.year, df['Date'].dt.month]).agg({
        'Discharge': 'mean',
        'Sediment_Load_tonnes_day': 'sum',
        'Rainfall': 'sum'
    })
    
    # Compute monthly sediment yield (tonnes/ha/month) by normalizing by area
    monthly_data['Monthly_Sediment_Yield_tons_ha'] = monthly_data['Sediment_Load_tonnes_day'] / area_ha
    monthly_data.index = pd.to_datetime(monthly_data.index.map(lambda x: f'{x[0]}-{x[1]:02}-01'))
    
    print(f"{watershed_name} Monthly Sediment Yield (tonnes/ha/month) Stats:")
    print(monthly_data['Monthly_Sediment_Yield_tons_ha'].describe())
    
    return monthly_data

def create_monthly_plot(monthly_data, watershed_name, output_plot, discharge_max, yield_max):
    """Create Figure 8: Monthly sediment yield, discharge, and rainfall plot.
    
    Args:
        monthly_data (DataFrame): Monthly metrics.
        watershed_name (str): Name of watershed.
        output_plot (str): Path to save plot.
        discharge_max (float): Max discharge for y-axis scaling.
        yield_max (float): Max sediment yield for y-axis scaling.
    """
    print(f"Generating monthly plot for {watershed_name}...")
    
    # Set up plot with three y-axes for discharge, rainfall, and sediment yield
    fig, ax1 = plt.subplots(figsize=(12, 5))
    fig.patch.set_facecolor('white')
    ax1.set_facecolor('white')
    
    # Plot discharge (left axis)
    ax1.plot(monthly_data.index, monthly_data['Discharge'], color='blue', marker='None', linestyle='-', 
             label='Discharge (m³/s)')
    ax1.set_ylim(0, discharge_max)
    ax1.set_ylabel('Discharge (m³/s)', color='blue', fontsize=18, fontname='Times New Roman', labelpad=12)
    ax1.set_xlabel('Date', fontsize=18, fontname='Times New Roman')
    ax1.tick_params(axis='x', rotation=45, labelsize=18)
    
    # Set x-axis ticks for every 5 years (1990–2020)
    years = range(1990, 2021, 5)
    year_ticks = [pd.to_datetime(f'{year}-01-01') for year in years]
    ax1.set_xticks(year_ticks)
    ax1.set_xticklabels([year.year for year in year_ticks], fontname='Times New Roman', fontsize=18)
    
    # Plot rainfall bars (right axis, inverted)
    ax2 = ax1.twinx()
    ax2.bar(monthly_data.index, monthly_data['Rainfall'], color='green', alpha=0.7, width=30, label='Rainfall (mm)')
    max_rainfall = monthly_data['Rainfall'].max() * 1.1
    ax2.set_ylim(max_rainfall * 3, 0)
    ax2.set_ylabel('Rainfall (mm)', color='green', fontsize=18, fontname='Times New Roman', labelpad=0)
    ax2.yaxis.set_label_position('right')
    ax2.yaxis.tick_right()
    ax2.tick_params(axis='y', which='both', left=False, labelsize=18)
    
    # Plot sediment yield (right axis, offset)
    ax3 = ax1.twinx()
    ax3.spines['right'].set_position(('outward', 60))
    ax3.plot(monthly_data.index, monthly_data['Monthly_Sediment_Yield_tons_ha'], color='red', marker='None', linestyle='-', 
             label='Sediment Yield (tonnes/ha)')
    ax3.set_ylim(0, yield_max)
    ax3.set_ylabel('Sediment Yield (tonnes/ha)', color='red', fontsize=18, fontname='Times New Roman', labelpad=1)
    ax3.tick_params(labelsize=18)
    
    # Add title
    plt.title(f'{watershed_name}', fontsize=20, fontname='Times New Roman')
    
    # Combine legends from all axes
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    lines3, labels3 = ax3.get_legend_handles_labels()
    plt.legend(lines1 + lines2 + lines3, labels1 + labels2 + labels3, loc='lower center', 
               bbox_to_anchor=(0.5, 0.5), ncol=3, prop={'family': 'Times New Roman', 'size': 16})
    
    # Remove gridlines
    ax1.grid(False)
    
    # Save and show plot
    plt.tight_layout()
    plt.savefig(output_plot, dpi=600, bbox_inches='tight')
    print(f"Figure saved to {output_plot}")
    plt.show()
    plt.close()

# Main processing loop for both watersheds
for watershed_name, params in data_paths.items():
    print(f"\n=== Processing {watershed_name} ===")
    
    try:
        # Load continuous data for merging
        df_cont = pd.read_csv(params['continuous'])
        
        # Predict SSC using QRF
        dates, ssc_pred = predict_ssc(
            params['intermittent'],
            params['continuous'],
            watershed_name,
            qrf_params[watershed_name]
        )
        
        # Merge predicted SSC with continuous data
        df_cont['Date'] = pd.to_datetime(df_cont['Date'], errors='coerce')
        df_temp = pd.DataFrame({'Date': dates, 'SSC_predicted': ssc_pred})
        df_merged = df_cont[['Date', 'Discharge', 'Rainfall']].merge(df_temp, on='Date', how='inner')
        
        if df_merged.empty:
            raise ValueError(f"{watershed_name} merged data is empty.")
        
        # Process monthly sediment yields
        monthly_data = process_monthly_data(
            df_merged['Date'],
            df_merged['Discharge'],
            df_merged['Rainfall'],
            df_merged['SSC_predicted'],
            params['area_ha'],
            watershed_name
        )
        
        print(f"\n{watershed_name} Monthly Metrics:")
        print(monthly_data[['Discharge', 'Rainfall', 'Monthly_Sediment_Yield_tons_ha']].round(2))
        
        # Save results to CSV
        monthly_data.to_csv(params['output_csv'])
        print(f"Data saved to {params['output_csv']}")
        
        # Generate Figure 8
        create_monthly_plot(
            monthly_data,
            watershed_name,
            params['output_plot'],
            params['discharge_max'],
            params['yield_max']
        )
        
    except Exception as e:
        print(f"Error processing {watershed_name}: {str(e)}")
        continue