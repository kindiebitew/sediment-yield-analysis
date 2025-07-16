# Script to generate Figure 9: Seasonal sediment yield for Gilgel Abay and Gumara watersheds (Section 3.4).
# Predicts daily SSC (g/L) for 1990–2020 using Quantile Random Forest (QRF) trained on intermittent data,
# calculates daily sediment yield (t/ha/day) with uncertainty quantiles,
# aggregates to seasonal means (Wet: June 1–Oct 31, Dry: Nov 1–May 31) in t/ha/yr, and produces side-by-side
# plots of mean sediment yield by Julian Day with IQR bands based on predicted quantiles.
# Outputs include Excel for daily data, CSV for seasonal data, and publication-quality PNG/SVG plots.
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

# Set plot style for publication quality (Section 3.4)
sns.set_style('white')
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 16

# Define constants
WATERSHED_CONFIG = {
    'Gilgel Abay': {
        'intermittent': Path(r"C:\Users\worku\Documents\sediment-yield-analysis\data\Intermittent_data.xlsx"),
        'continuous': Path(r"C:\Users\worku\Documents\sediment-yield-analysis\data\continuous_data.csv"),
        'area_km2': 1664,
        'yield_max': 80,
        'output_dir': Path(r"C:\Users\worku\Documents\sediment-yield-analysis\outputs"),
        'output_csv': Path(r"C:\Users\worku\Documents\sediment-yield-analysis\outputs\Gilgel_Abay_Seasonal_Sediment_Yield.csv")
    },
    'Gumara': {
        'intermittent': Path(r"C:\Users\worku\Documents\sediment-yield-analysis\data\Intermittent_data_gum.csv"),
        'continuous': Path(r"C:\Users\worku\Documents\sediment-yield-analysis\data\continuous_data_gum.csv"),
        'area_km2': 1394,
        'yield_max': 80,
        'output_dir': Path(r"C:\Users\worku\Documents\sediment-yield-analysis\outputs"),
        'output_csv': Path(r"C:\Users\worku\Documents\sediment-yield-analysis\outputs\Gumara_Seasonal_Sediment_Yield.csv")
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
        continuous_path (Path): Path to continuous data (Excel or CSV).
        watershed_name (str): Name of the watershed ('Gilgel Abay' or 'Gumara').
        qrf_params (dict): QRF hyperparameters.
        is_excel_inter (bool): Flag to indicate if intermittent file is Excel (True) or CSV (False).
    Returns:
        DataFrame with continuous data and predicted SSC with quantiles.
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
    
    # Feature engineering (rolling means and lags)
    df_inter['MA_Discharge_3'] = df_inter['Discharge'].rolling(window=3, min_periods=1).mean().bfill()
    df_inter['Lag_Discharge'] = df_inter['Discharge'].shift(1).bfill()
    df_inter['Lag_Discharge_3'] = df_inter['Discharge'].shift(3).bfill()
    df_cont['MA_Discharge_3'] = df_cont['Discharge'].rolling(window=3, min_periods=1).mean().bfill()
    df_cont['Lag_Discharge'] = df_cont['Discharge'].shift(1).bfill()
    df_cont['Lag_Discharge_3'] = df_cont['Discharge'].shift(3).bfill()
    
    # Select predictors (Section 3.2)
    predictors = ['Discharge', 'MA_Discharge_3', 'Lag_Discharge', 'Lag_Discharge_3', 'Rainfall', 'ETo', 'Annual_Rainfall', 'Cumulative_Rainfall']
    X_inter = df_inter[predictors]
    y_inter = df_inter['SSC']
    X_cont = df_cont[predictors]
    
    # Scale features
    scaler = RobustScaler()
    X_inter_scaled = scaler.fit_transform(X_inter)
    X_cont_scaled = scaler.transform(X_cont)
    
    # Train QRF and predict SSC with quantiles
    try:
        qrf = RandomForestQuantileRegressor(**qrf_params)
        qrf.fit(X_inter_scaled, y_inter)
        ssc_pred = qrf.predict(X_cont_scaled, quantiles=[0.25, 0.5, 0.75])  # Added quantiles for IQR
    except Exception as e:
        raise ValueError(f"Error training/predicting QRF for {watershed_name}: {str(e)}")
    
    # Debug: SSC prediction summary
    print(f"{watershed_name} SSC Prediction Summary (g/L):")
    for q, preds in zip([0.25, 0.5, 0.75], ssc_pred.T):
        print(f"Quantile {q}: {pd.Series(preds).describe()}")
    
    # Create output DataFrame with SSC quantiles
    df_cont['SSC_Q25'] = ssc_pred[:, 0]  # 0.25 quantile
    df_cont['SSC_Median'] = ssc_pred[:, 1]  # 0.5 quantile
    df_cont['SSC_Q75'] = ssc_pred[:, 2]  # 0.75 quantile
    return df_cont[['Date', 'Rainfall', 'Discharge', 'Temperature', 'ETo', 'SSC_Q25', 'SSC_Median', 'SSC_Q75', 'Annual_Rainfall', 'Cumulative_Rainfall']]

def calculate_sediment_yield(df, watershed_name, area_km2, output_dir):
    """
    Calculate daily sediment yield (t/ha/day) with uncertainty and save to Excel.
    Args:
        df (DataFrame): DataFrame with Date, Rainfall, Discharge, SSC_Q25, SSC_Median, SSC_Q75.
        watershed_name (str): Name of the watershed.
        area_km2 (float): Watershed area in km².
        output_dir (Path): Directory for output files.
    Returns:
        DataFrame with daily sediment yield and uncertainty.
    """
    print(f"Calculating sediment yield for {watershed_name}...")
    
    # Create output directory
    output_dir.mkdir(exist_ok=True)
    
    # Filter for 1990–2020
    df = df[(df['Date'].dt.year >= 1990) & (df['Date'].dt.year <= 2020)]
    
    # Calculate sediment yield (t/ha/day) and uncertainty
    df['Sediment_Yield_Median'] = df['Discharge'] * df['SSC_Median'] * LOAD_FACTOR / (area_km2 * 100)
    df['Sediment_Yield_Q25'] = df['Discharge'] * df['SSC_Q25'] * LOAD_FACTOR / (area_km2 * 100)
    df['Sediment_Yield_Q75'] = df['Discharge'] * df['SSC_Q75'] * LOAD_FACTOR / (area_km2 * 100)
    
    # Drop invalid data
    df = df.dropna(subset=['Date', 'Rainfall', 'Discharge', 'Sediment_Yield_Median', 'Sediment_Yield_Q25', 'Sediment_Yield_Q75', 'Annual_Rainfall', 'Cumulative_Rainfall'])
    
    if df.empty:
        raise ValueError(f"{watershed_name} data empty after cleaning")
    
    # Save to Excel
    output_path = output_dir / f"{watershed_name.replace(' ', '_')}_Daily_SSC_Sediment_Yield.xlsx"
    df.to_excel(output_path, index=False)
    print(f"Daily data saved to {output_path}")
    
    # Debug: Print sample
    print(f"\n{watershed_name} Daily Data Sample:")
    print(df[['Date', 'Rainfall', 'Discharge', 'Sediment_Yield_Median', 'Sediment_Yield_Q25', 'Sediment_Yield_Q75']].head())
    print(f"{watershed_name} Data Shape: {df.shape}")
    
    return df

def process_seasonal_data(df, watershed_name):
    """
    Process daily data to compute seasonal sediment yield in t/ha/yr with uncertainty.
    Args:
        df (DataFrame): DataFrame with daily data (Date, Rainfall, Discharge, Sediment_Yield_Median, Sediment_Yield_Q25, Sediment_Yield_Q75).
        watershed_name (str): Name of the watershed.
    Returns:
        DataFrame with seasonal data by Julian Day and uncertainty.
    """
    print(f"\nProcessing seasonal data for {watershed_name}...")
    
    # Filter for 1990–2020
    df = df[(df['Date'].dt.year >= 1990) & (df['Date'].dt.year <= 2020)]
    if df.empty:
        raise ValueError(f"{watershed_name} data empty after year filtering (1990-2020)")
    
    # Assign season
    df['Julian_Day'] = df['Date'].dt.dayofyear
    df['Season'] = df['Julian_Day'].apply(lambda x: 'Wet' if 152 <= x <= 304 else 'Dry')  # Wet: June 1–Oct 31, Dry: Nov 1–May 31
    
    # Convert daily yields to yearly (t/ha/yr) for median, Q25, and Q75
    df['Sediment_Yield_Median_tons_ha_yr'] = df['Sediment_Yield_Median'] * np.where(df['Season'] == 'Wet', 153, 212)
    df['Sediment_Yield_Q25_tons_ha_yr'] = df['Sediment_Yield_Q25'] * np.where(df['Season'] == 'Wet', 153, 212)
    df['Sediment_Yield_Q75_tons_ha_yr'] = df['Sediment_Yield_Q75'] * np.where(df['Season'] == 'Wet', 153, 212)
    
    # Aggregate by Julian Day and Season
    seasonal_data = df.groupby(['Julian_Day', 'Season'])[['Sediment_Yield_Median_tons_ha_yr', 'Sediment_Yield_Q25_tons_ha_yr', 'Sediment_Yield_Q75_tons_ha_yr']].mean().reset_index()
    
    # Calculate seasonal statistics (already included in quantile propagation)
    stats = df.groupby('Season')[['Sediment_Yield_Median_tons_ha_yr', 'Sediment_Yield_Q25_tons_ha_yr', 'Sediment_Yield_Q75_tons_ha_yr']].agg({
        'Sediment_Yield_Median_tons_ha_yr': 'mean',
        'Sediment_Yield_Q25_tons_ha_yr': 'mean',
        'Sediment_Yield_Q75_tons_ha_yr': 'mean'
    }).rename(columns={
        'Sediment_Yield_Median_tons_ha_yr': 'mean',
        'Sediment_Yield_Q25_tons_ha_yr': 'q25',
        'Sediment_Yield_Q75_tons_ha_yr': 'q75'
    }).reset_index()
    
    # Debug: Print seasonal statistics
    print(f"{watershed_name} Seasonal Yield Stats (t/ha/yr):")
    print(stats)
    wet_mean = stats[stats['Season'] == 'Wet']['mean'].iloc[0] if 'Wet' in stats['Season'].values else 0
    dry_mean = stats[stats['Season'] == 'Dry']['mean'].iloc[0] if 'Dry' in stats['Season'].values else 0
    if wet_mean + dry_mean > 0:
        print(f"{watershed_name} Wet Season Contribution: {wet_mean / (wet_mean + dry_mean) * 100:.1f}%")
    
    return seasonal_data, df, stats

def create_side_by_side_plot(data_dict, output_dir):
    """
    Generate Figure 9: Side-by-side seasonal sediment yield plot in t/ha/yr with IQR bands based on predicted quantiles.
    Args:
        data_dict (dict): Dictionary with seasonal data and stats for each watershed.
        output_dir (Path): Directory for output plots.
    """
    print("\nGenerating Figure 9...")
    
    # Calculate dynamic y-axis limit
    max_yield = max(data_dict[w]['seasonal_data']['Sediment_Yield_Median_tons_ha_yr'].max() for w in data_dict if w in data_dict)
    max_yield = max(max_yield * 1.1, 80)  # Ensure minimum 80 t/ha/yr
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), sharey=True)
    fig.patch.set_facecolor('white')
    
    for ax, watershed_name in zip([ax1, ax2], ['Gilgel Abay', 'Gumara']):
        if watershed_name in data_dict:
            seasonal_data = data_dict[watershed_name]['seasonal_data']
            stats = data_dict[watershed_name]['stats']
            
            # Plot Dry Season (split)
            dry_data = seasonal_data[seasonal_data['Season'] == 'Dry']
            dry_before = dry_data[dry_data['Julian_Day'] < 152]
            dry_after = dry_data[dry_data['Julian_Day'] > 304]
            ax.plot(dry_before['Julian_Day'], dry_before['Sediment_Yield_Median_tons_ha_yr'], color='#FF8C00', 
                    label='Dry Mean', linewidth=2)
            ax.plot(dry_after['Julian_Day'], dry_after['Sediment_Yield_Median_tons_ha_yr'], color='#FF8C00', 
                    linewidth=2)
            
            # Plot Wet Season
            wet_data = seasonal_data[seasonal_data['Season'] == 'Wet']
            ax.plot(wet_data['Julian_Day'], wet_data['Sediment_Yield_Median_tons_ha_yr'], color='#1f77b4', 
                    label='Wet Mean', linewidth=2)
            
            # Add IQR bands based on predicted quantiles
            ax.fill_between(wet_data['Julian_Day'], wet_data['Sediment_Yield_Q25_tons_ha_yr'], wet_data['Sediment_Yield_Q75_tons_ha_yr'],
                           color='#1f77b4', alpha=0.2, label='Wet IQR')
            ax.fill_between(dry_before['Julian_Day'], dry_before['Sediment_Yield_Q25_tons_ha_yr'], dry_before['Sediment_Yield_Q75_tons_ha_yr'],
                           color='#FF8C00', alpha=0.2, label='Dry IQR')
            ax.fill_between(dry_after['Julian_Day'], dry_after['Sediment_Yield_Q25_tons_ha_yr'], dry_after['Sediment_Yield_Q75_tons_ha_yr'],
                           color='#FF8C00', alpha=0.2)
            
            # Add statistical annotations
            if 'Wet' in stats['Season'].values:
                wet_stats = stats[stats['Season'] == 'Wet'].iloc[0]
                ax.text(0.02, 0.98, f"Wet: Mean={wet_stats['mean']:.2f}\nQ25={wet_stats['q25']:.2f}\nQ75={wet_stats['q75']:.2f}",
                        transform=ax.transAxes, fontsize=12, verticalalignment='top',
                        color='#1f77b4', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='#1f77b4'))
            if 'Dry' in stats['Season'].values:
                dry_stats = stats[stats['Season'] == 'Dry'].iloc[0]
                ax.text(0.98, 0.98, f"Dry: Mean={dry_stats['mean']:.2f}\nQ25={dry_stats['q25']:.2f}\nQ75={dry_stats['q75']:.2f}",
                        transform=ax.transAxes, fontsize=12, verticalalignment='top', horizontalalignment='right',
                        color='#FF8C00', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='#FF8C00'))
            
            # Add legend
            ax.legend(title='Season', fontsize=14, loc='center left',
                      bbox_to_anchor=(0.02, 0.5), frameon=True, edgecolor='black')
            
            # Set title and labels
            ax.set_title(f"({'a' if watershed_name == 'Gilgel Abay' else 'b'}) {watershed_name}", fontsize=20)
            ax.set_xlabel('Julian Day', fontsize=18)
            if ax == ax1:
                ax.set_ylabel('Mean Sediment Yield\n(t/ha/yr)', fontsize=18)
            
            # Configure axes
            ax.set_ylim(0, max_yield)
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            ax.tick_params(axis='both', labelsize=14)
            ax.grid(False)
            
            # Debug: Print axis limits
            print(f"{watershed_name} Plot Axis Limits: 0 to {max_yield:.2f} t/ha/yr")
        else:
            ax.text(0.5, 0.5, 'Data Unavailable', fontsize=16, ha='center', va='center')
            ax.set_title(f"({'a' if watershed_name == 'Gilgel Abay' else 'b'}) {watershed_name}", fontsize=20)
            ax.set_xlabel('Julian Day', fontsize=18)
            if ax == ax1:
                ax.set_ylabel('Mean Sediment Yield\n(t/ha/yr)', fontsize=18)
            ax.set_ylim(0, 80)
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            ax.tick_params(axis='both', labelsize=14)
            ax.grid(False)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save plots
    output_png = output_dir / 'Figure9_Seasonal_Sediment_Yield.png'
    output_svg = output_dir / 'Figure9_Seasonal_Sediment_Yield.svg'
    plt.savefig(output_png, dpi=600, format='png', bbox_inches='tight')
    plt.savefig(output_svg, format='svg', bbox_inches='tight')
    plt.show()  # Display the plot in Jupyter notebook
    plt.close()

def main():
    """
    Main function to process data and generate Figure 9.
    """
    print("Starting script execution...")
    data_dict = {}
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
            
            # Process seasonal data
            seasonal_data, df, stats = process_seasonal_data(daily_data, watershed_name)
            
            # Store data
            data_dict[watershed_name] = {
                'seasonal_data': seasonal_data,
                'df': df,
                'stats': stats
            }
            
            # Save seasonal data
            seasonal_data.to_csv(params['output_csv'], index=False)
            print(f"Seasonal data saved to {params['output_csv']}")
        
        except Exception as e:
            print(f"Error processing {watershed_name}: {str(e)}")
            continue
    
    if data_dict:
        try:
            create_side_by_side_plot(data_dict, WATERSHED_CONFIG['Gilgel Abay']['output_dir'])
        except Exception as e:
            print(f"Error generating Figure 9: {str(e)}")
    else:
        print("Error: No data processed successfully. Figure 9 cannot be generated.")

if __name__ == "__main__":
    main()
