# Script for calculating seasonal sediment yields in Gilgel Abay and Gumara watersheds
# Purpose: Estimates daily sediment yields (tonnes/ha/day) by season (wet: June 1–September 30, dry: otherwise)
# for Figure 9 in the paper "Machine Learning-Based Sedigraph Reconstruction for Enhanced Sediment Load
# Estimation in the Upper Blue Nile Basin." Uses Quantile Random Forest (QRF) to predict SSC.
# Author: Kindie B. Worku and co-authors
# Data: Intermittent SSC from MoWE/ABAO, continuous hydrological data from EMI
# Output: Seasonal sediment yield CSV files and Figure 9 (side-by-side plots with IQR bands)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from quantile_forest import RandomForestQuantileRegressor
from sklearn.preprocessing import RobustScaler
from matplotlib.ticker import MaxNLocator
import os

# Set plot style for publication quality (Times New Roman, font size 16)
# White background with seaborn for clean aesthetics
sns.set_style('white')
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 16
plt.rcParams['font.serif'] = ['Times New Roman']

# Define watershed parameters and file paths
# Note: Paths are placeholders; actual data stored locally due to MoWE/ABAO restrictions
# Area (km²) from watershed delineations; yield_max set for plot scaling based on data ranges
data_paths = {
    'Gilgel Abay': {
        'intermittent': r"D:\Gilgel Abay\Sedigrapgh\Intermittent_data.csv",
        'continuous': r"D:\Gilgel Abay\Sedigrapgh\continuous_data.csv",
        'area_km2': 1664,  # Watershed area for yield normalization
        'yield_max': 0.4  # Max sediment yield (tonnes/ha/day) for plot scaling
    },
    'Gumara': {
        'intermittent': r"D:\Gumara\Sedigrapgh\Intermittent_data_gum.csv",
        'continuous': r"D:\Gumara\Sedigrapgh\continious_data_gum.csv",
        'area_km2': 1394,  # Watershed area for yield normalization
        'yield_max': 0.4  # Max sediment yield (tonnes/ha/day) for plot scaling
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

# Output paths for combined plot and CSV
# Note: Outputs saved in Gumara directory for convenience; can be adjusted
output_dir = r"D:\Gumara\Sedigrapgh"
output_png = os.path.join(output_dir, "Seasonal_Sediment_Yield_QRF_Side_by_Side.png")
output_svg = os.path.join(output_dir, "Seasonal_Sediment_Yield_QRF_Side_by_Side.svg")
output_csv = os.path.join(output_dir, "Seasonal_Sediment_Yield_QRF_Combined.csv")

def predict_ssc(intermittent_path, continuous_path, watershed_name, qrf_params):
    """Predict suspended sediment concentration (SSC) using QRF model.
    
    Args:
        intermittent_path (str): Path to intermittent SSC data (CSV).
        continuous_path (str): Path to continuous hydrological data (CSV).
        watershed_name (str): Name of watershed (Gilgel Abay or Gumara).
        qrf_params (dict): QRF hyperparameters.
    
    Returns:
        tuple: DataFrame with continuous data (Date, Rainfall, Discharge), predicted SSC values.
    """
    print(f"Predicting SSC for {watershed_name}...")
    
    # Check file existence
    if not os.path.exists(intermittent_path):
        raise FileNotFoundError(f"Intermittent file not found: {intermittent_path}")
    if not os.path.exists(continuous_path):
        raise FileNotFoundError(f"Continuous file not found: {continuous_path}")
    
    # Load data
    try:
        df_inter = pd.read_csv(intermittent_path)
        df_cont = pd.read_csv(continuous_path)
    except Exception as e:
        raise ValueError(f"Error loading data for {watershed_name}: {str(e)}")
    
    # Validate columns (see paper Section 2.2 for data description)
    required_cols_inter = ['Date', 'Rainfall', 'Discharge', 'Temperature', 'ETo', 'SSC']
    required_cols_cont = ['Date', 'Rainfall', 'Discharge', 'Temperature', 'ETo']
    if not all(col in df_inter.columns for col in required_cols_inter):
        raise ValueError(f"{watershed_name} intermittent data missing columns")
    if not all(col in df_cont.columns for col in required_cols_cont):
        raise ValueError(f"{watershed_name} continuous data missing columns")
    
    # Convert dates to datetime, dropping invalid entries
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
    
    # Feature engineering (see paper Section 2.3 for predictor selection rationale)
    # Log_Discharge linearizes flow-SSC relationship; Discharge_Rainfall captures interaction
    df_inter['Log_Discharge'] = np.log1p(df_inter['Discharge'].clip(lower=0))
    df_inter['Discharge_Rainfall'] = df_inter['Discharge'] * df_inter['Rainfall']
    df_cont['Log_Discharge'] = np.log1p(df_cont['Discharge'].clip(lower=0))
    df_cont['Discharge_Rainfall'] = df_cont['Discharge'] * df_cont['Rainfall']
    
    # Select predictors based on physical relevance (paper Section 3.2)
    predictors = ['Log_Discharge', 'Rainfall', 'Temperature', 'ETo', 'Discharge_Rainfall']
    X_inter = df_inter[predictors]
    y_inter = df_inter['SSC']
    X_cont = df_cont[predictors]
    
    # Scale features using RobustScaler to handle outliers (paper Section 2.3)
    scaler = RobustScaler()
    X_inter_scaled = scaler.fit_transform(X_inter)
    X_cont_scaled = scaler.transform(X_cont)
    
    # Train QRF model with median quantile (0.5) for SSC prediction
    qrf = RandomForestQuantileRegressor(**qrf_params, random_state=42)
    qrf.fit(X_inter_scaled, y_inter)
    
    # Predict SSC
    ssc_pred = qrf.predict(X_cont_scaled, quantiles=0.5)
    
    print(f"{watershed_name} SSC Prediction Summary:")
    print(pd.Series(ssc_pred).describe())
    
    return df_cont[['Date', 'Rainfall', 'Discharge']], ssc_pred

def process_seasonal_data(dates, rainfall, discharge, ssc_pred, area_km2, watershed_name):
    """Process daily data to compute mean seasonal sediment yields (tonnes/ha/day).
    
    Args:
        dates (Series): Datetime series for continuous data.
        rainfall (Series): Daily rainfall (mm).
        discharge (Series): Daily discharge (m³/s).
        ssc_pred (array): Predicted SSC (g/L).
        area_km2 (float): Watershed area (km²).
        watershed_name (str): Name of watershed.
    
    Returns:
        tuple: Seasonal DataFrame (Julian Day, Season, Yield), full DataFrame with daily data.
    """
    print(f"Processing seasonal data for {watershed_name}...")
    
    # Create DataFrame, ensuring numeric types
    df = pd.DataFrame({
        'Date': dates,
        'Rainfall': pd.to_numeric(rainfall, errors='coerce'),
        'Discharge': pd.to_numeric(discharge, errors='coerce'),
        'SSC_predicted': pd.to_numeric(ssc_pred, errors='coerce')
    })
    
    # Drop invalid data
    df = df.dropna()
    if df.empty:
        raise ValueError(f"{watershed_name} data empty after cleaning")
    
    # Calculate sediment load (tonnes/day) using Equation 3 (paper Section 2.4)
    # 0.0864 converts g/s to tonnes/day (seconds/day ÷ 10^6 g/tonne)
    df['Sediment_Load_tonnes_day'] = df['Discharge'] * df['SSC_predicted'] * 0.0864
    
    # Calculate sediment yield (tonnes/ha/day) by normalizing by area (1 km² = 100 ha)
    df['Sediment_Yield_tons_ha_day'] = df['Sediment_Load_tonnes_day'] / (area_km2 * 100)
    
    # Add Julian Day for seasonal assignment
    df['Julian_Day'] = df['Date'].dt.dayofyear
    
    # Assign seasons based on Upper Blue Nile Basin monsoon (paper Section 2.4)
    # Wet: June 1 (day 152) to September 30 (day 273); Dry: otherwise
    wet_season_start = 152  # June 1
    wet_season_end = 273   # September 30
    df['Season'] = df['Julian_Day'].apply(lambda x: 'Wet' if wet_season_start <= x <= wet_season_end else 'Dry')
    
    # Debug seasonal assignment
    print(f"\n{watershed_name} Season Assignment (sample):")
    print(df[['Date', 'Julian_Day', 'Season', 'Sediment_Yield_tons_ha_day']].head(10))
    print(f"{watershed_name} Wet season count: {len(df[df['Season'] == 'Wet'])}")
    print(f"{watershed_name} Dry season count: {len(df[df['Season'] == 'Dry'])}")
    
    # Aggregate mean sediment yield by Julian Day and Season
    seasonal_data = df.groupby(['Julian_Day', 'Season'])['Sediment_Yield_tons_ha_day'].mean().reset_index()
    
    print(f"\n{watershed_name} Seasonal Yield Stats (tonnes/ha/day):")
    print(seasonal_data['Sediment_Yield_tons_ha_day'].describe())
    
    return seasonal_data, df

def create_side_by_side_plot(data_dict, output_png, output_svg, yield_max):
    """Create Figure 9: Side-by-side seasonal sediment yield plots with IQR bands.
    
    Args:
        data_dict (dict): Dictionary with seasonal data and daily DataFrame for each watershed.
        output_png (str): Path to save PNG plot.
        output_svg (str): Path to save SVG plot.
        yield_max (float): Max sediment yield for y-axis scaling.
    """
    print("Generating side-by-side seasonal plot...")
    
    # Set up side-by-side subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), sharey=True)
    fig.patch.set_facecolor('white')
    
    for ax, watershed_name in zip([ax1, ax2], ['Gilgel Abay', 'Gumara']):
        seasonal_data = data_dict[watershed_name]['seasonal_data']
        df = data_dict[watershed_name]['df']
        
        # Calculate seasonal statistics (mean, std, IQR) for IQR bands
        stats = df.groupby('Season')['Sediment_Yield_tons_ha_day'].agg([
            'mean', 'std',
            lambda x: np.percentile(x, 25),
            lambda x: np.percentile(x, 75)
        ]).rename(columns={
            '<lambda_0>': 'q25',
            '<lambda_1>': 'q75'
        }).reset_index()
        
        wet_stats = stats[stats['Season'] == 'Wet'].iloc[0] if 'Wet' in stats['Season'].values else None
        dry_stats = stats[stats['Season'] == 'Dry'].iloc[0] if 'Dry' in stats['Season'].values else None
        
        # Plot Dry Season (split before and after wet season)
        dry_data = seasonal_data[seasonal_data['Season'] == 'Dry']
        dry_before = dry_data[dry_data['Julian_Day'] < 152]
        dry_after = dry_data[dry_data['Julian_Day'] > 273]
        ax.plot(dry_before['Julian_Day'], dry_before['Sediment_Yield_tons_ha_day'], color='#FF8C00', 
                label='Dry Mean', linewidth=2)
        ax.plot(dry_after['Julian_Day'], dry_after['Sediment_Yield_tons_ha_day'], color='#FF8C00', 
                linewidth=2)
        
        # Plot Wet Season
        wet_data = seasonal_data[seasonal_data['Season'] == 'Wet']
        ax.plot(wet_data['Julian_Day'], wet_data['Sediment_Yield_tons_ha_day'], color='#1f77b4', 
                label='Wet Mean', linewidth=2)
        
        # Add IQR bands for uncertainty (paper Section 3.3)
        if wet_stats is not None:
            ax.fill_between(wet_data['Julian_Day'], wet_stats['q25'], wet_stats['q75'], 
                            color='#1f77b4', alpha=0.2, label='Wet IQR')
        if dry_stats is not None:
            ax.fill_between(dry_before['Julian_Day'], dry_stats['q25'], dry_stats['q75'], 
                            color='#FF8C00', alpha=0.2, label='Dry IQR')
            ax.fill_between(dry_after['Julian_Day'], dry_stats['q25'], dry_stats['q75'], 
                            color='#FF8C00', alpha=0.2)
        
        # Add statistical annotations (mean, SD) in plot
        if wet_stats is not None:
            ax.text(0.02, 0.98, f"Wet: Mean={wet_stats['mean']:.3f}\nSD={wet_stats['std']:.3f}", 
                    transform=ax.transAxes, fontsize=12, verticalalignment='top', 
                    color='#1f77b4', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='#1f77b4'))
        if dry_stats is not None:
            ax.text(0.98, 0.98, f"Dry: Mean={dry_stats['mean']:.3f}\nSD={dry_stats['std']:.3f}", 
                    transform=ax.transAxes, fontsize=12, verticalalignment='top', horizontalalignment='right', 
                    color='#FF8C00', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='#FF8C00'))
        
        # Add legend inside plot
        ax.legend(title='Season', fontsize=14, loc='center left', 
                  bbox_to_anchor=(0.02, 0.5), prop={'family': 'Times New Roman'},
                  frameon=True, edgecolor='black')
        
        # Set title and labels
        ax.set_title(f'{watershed_name}', fontsize=20, fontfamily='Times New Roman')
        ax.set_xlabel('Julian Day', fontsize=18, fontfamily='Times New Roman')
        if ax == ax1:
            ax.set_ylabel('Mean Sediment Yield\n(tonnes/ha/day)', fontsize=18, fontfamily='Times New Roman')
        
        # Configure axes
        ax.set_ylim(0, yield_max)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.tick_params(axis='both', labelsize=14)
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontfamily('Times New Roman')
        
        # Remove grid
        ax.grid(False)
        
        # Style spines for clarity
        for spine in ax.spines.values():
            spine.set_linewidth(1)
            spine.set_color('black')
    
    # Adjust layout to prevent overlap
    plt.tight_layout()
    
    # Save in PNG and SVG formats for journal submission
    plt.savefig(output_png, dpi=600, transparent=True, bbox_inches='tight')
    plt.savefig(output_svg, format='svg', transparent=True, bbox_inches='tight')
    print(f"Figure saved to {output_png} (PNG) and {output_svg} (SVG)")
    plt.show()
    plt.close()

# Main processing loop for both watersheds
data_dict = {}
for watershed_name, params in data_paths.items():
    print(f"\n=== Processing {watershed_name} ===")
    
    try:
        # Predict SSC using QRF
        df_cont, ssc_pred = predict_ssc(
            params['intermittent'],
            params['continuous'],
            watershed_name,
            qrf_params[watershed_name]
        )
        
        # Process seasonal sediment yields
        seasonal_data, df = process_seasonal_data(
            df_cont['Date'],
            df_cont['Rainfall'],
            df_cont['Discharge'],
            ssc_pred,
            params['area_km2'],
            watershed_name
        )
        
        # Store data for plotting
        data_dict[watershed_name] = {
            'seasonal_data': seasonal_data,
            'df': df,
            'yield_max': params['yield_max']
        }
        
        # Save individual seasonal data to CSV
        seasonal_data.to_csv(os.path.join(output_dir, f"{watershed_name}_Seasonal_Sediment_Yield_QRF.csv"), index=False)
        print(f"Data saved to '{os.path.join(output_dir, f'{watershed_name}_Seasonal_Sediment_Yield_QRF.csv')}'")
        
    except Exception as e:
        print(f"Error processing {watershed_name}: {str(e)}")
        continue

# Generate side-by-side plot for Figure 9
if len(data_dict) == 2:  # Ensure both watersheds processed
    create_side_by_side_plot(data_dict, output_png, output_svg, max(params['yield_max'] for params in data_paths.values()))
else:
    print("Error: Could not generate side-by-side plot due to missing data for one or both watersheds")