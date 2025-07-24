# Script to predict Suspended Sediment Concentration (SSC) and calculate seasonal sediment yield (t/ha/yr)
# for Gilgel Abay and Gumara watersheds, generating Figure 9 (Manuscript) as described in the study.
# Uses Quantile Random Forest (QRF) with parameters from model_selection.py to predict daily SSC (g/L).
# Processes data from 1990–2020, aggregates to seasonal yield, and creates side-by-side plots (color and grayscale).
# Requires input files: Intermittent_data.xlsx, continuous_data.csv (Gilgel Abay), Intermittent_data_gum.csv,
# continuous_data_gum.csv (Gumara), and best_params_comparison_70split.csv.
# Author: Kindie B. Worku
# Date: 2025-07-23

# Import required libraries
import pandas as pd
import numpy as np
from quantile_forest import RandomForestQuantileRegressor
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from matplotlib.ticker import MaxNLocator
import warnings
import ast
from uuid import uuid4

# Suppress warnings for cleaner console output
warnings.filterwarnings('ignore')

# Set plot style for publication quality
sns.set_style('white')
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 16

# Configuration dictionary for watershed-specific parameters
WATERSHED_CONFIG = {
    'Gilgel Abay': {
        'intermittent': Path(r"C:\Users\worku\Documents\sediment-yield-analysis\data\Intermittent_data.xlsx"),
        'continuous': Path(r"C:\Users\worku\Documents\sediment-yield-analysis\data\continuous_data.csv"),
        'area_km2': 1664,  # Watershed area in km² for yield calculations
        'yield_max': 80,   # Maximum sediment yield for plot scaling (t/ha/yr)
        'rainfall_max': 80,  # Maximum rainfall for plot scaling (mm/day)
        'output_dir': Path(r"C:\Users\worku\Documents\sediment-yield-analysis\outputs"),
        'output_csv': Path(r"C:\Users\worku\Documents\sediment-yield-analysis\outputs\Gilgel_Abay_Seasonal_Sediment_Yield.csv")
    },
    'Gumara': {
        'intermittent': Path(r"C:\Users\worku\Documents\sediment-yield-analysis\data\Intermittent_data_gum.csv"),
        'continuous': Path(r"C:\Users\worku\Documents\sediment-yield-analysis\data\continuous_data_gum.csv"),
        'area_km2': 1394,  # Watershed area in km² for yield calculations
        'yield_max': 80,   # Maximum sediment yield for plot scaling (t/ha/yr)
        'rainfall_max': 80,  # Maximum rainfall for plot scaling (mm/day)
        'output_dir': Path(r"C:\Users\worku\Documents\sediment-yield-analysis\outputs"),
        'output_csv': Path(r"C:\Users\worku\Documents\sediment-yield-analysis\outputs\Gumara_Seasonal_Sediment_Yield.csv")
    }
}

# Configuration settings
USE_LAG_RAINFALL = False  # Exclude Lag_Rainfall_7 and Lag_Rainfall_14 from predictors
LOAD_FACTOR = 86.4  # Conversion factor: m³/s × g/L to t/day (86,400 s/day × 10⁻⁶ t/g)

def add_seasonal_features(df):
    """
    Add sinusoidal seasonal features (Sin_Julian, Cos_Julian) based on Julian Day for temporal modeling.
    
    Args:
        df (pd.DataFrame): Input DataFrame with 'Date' column.
    
    Returns:
        pd.DataFrame: DataFrame with added Sin_Julian and Cos_Julian columns, or None if errors occur.
    """
    try:
        df = df.copy()
        # Convert Date column to datetime
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        if df['Date'].isna().any():
            print(f"Error: {df['Date'].isna().sum()} invalid dates found")
            return None
        # Calculate Julian Day and sinusoidal features
        df['Julian_Day'] = df['Date'].dt.dayofyear
        df['Sin_Julian'] = np.sin(2 * np.pi * df['Julian_Day'] / 365.25)
        df['Cos_Julian'] = np.cos(2 * np.pi * df['Julian_Day'] / 365.25)
        return df.drop(columns=['Julian_Day'])
    except Exception as e:
        print(f"Error in add_seasonal_features: {str(e)}")
        return None

def load_best_params(watershed_name, output_dir):
    """
    Load QRF hyperparameters from best_params_comparison_70split.csv for the specified watershed.
    
    Args:
        watershed_name (str): Name of the watershed ('Gilgel Abay' or 'Gumara').
        output_dir (Path): Directory containing the parameters CSV file.
    
    Returns:
        dict: QRF hyperparameters, or None if the file or parameters are not found.
    """
    try:
        params_path = output_dir / "best_params_comparison_70split.csv"
        # Check if parameters file exists
        if not params_path.exists():
            print(f"Error: {params_path} not found")
            return None
        params_df = pd.read_csv(params_path)
        # Filter for QRF parameters specific to the watershed
        qrf_params_row = params_df[(params_df['Model'] == 'QRF') & (params_df['Watershed'] == watershed_name)]
        if qrf_params_row.empty:
            print(f"Error: No QRF parameters for {watershed_name}")
            return None
        # Parse QRF parameters from string to dictionary
        qrf_params = ast.literal_eval(qrf_params_row['Parameters'].iloc[0])
        print(f"Loaded QRF params for {watershed_name}: {qrf_params}")
        return qrf_params
    except Exception as e:
        print(f"Error loading QRF params for {watershed_name}: {str(e)}")
        return None

def normalize_rainfall(df, watershed_name):
    """
    Normalize rainfall units to mm/day, handling different input units (cm, meters, or monthly aggregates).
    
    Args:
        df (pd.DataFrame): Input DataFrame with 'Rainfall' column.
        watershed_name (str): Name of the watershed for error reporting.
    
    Returns:
        pd.DataFrame: DataFrame with normalized 'Rainfall' column, or None if errors occur.
    """
    try:
        df = df.copy()
        rainfall_col = 'Rainfall'
        if rainfall_col not in df:
            print(f"Error: {watershed_name} missing Rainfall column")
            return None

        rainfall_max = df[rainfall_col].max()
        print(f"{watershed_name} Rainfall - Max: {rainfall_max:.2f}, Median: {df[rainfall_col].median():.2f}")
        
        # Convert units based on rainfall magnitude
        if rainfall_max > 100:  # Likely cm or monthly
            if rainfall_max > 1000:  # Monthly aggregate
                print(f"{watershed_name}: Assuming monthly rainfall, converting to daily")
                days_per_month = df.groupby(df['Date'].dt.to_period('M')).size()
                df[rainfall_col] = df.apply(lambda x: x[rainfall_col] / days_per_month.get(x['Date'].to_period('M'), 30), axis=1)
            else:  # cm
                print(f"{watershed_name}: Converting cm to mm")
                df[rainfall_col] = df[rainfall_col] * 10
        elif rainfall_max < 1:  # Meters
            print(f"{watershed_name}: Converting meters to mm")
            df[rainfall_col] = df[rainfall_col] * 1000
        
        # Clip rainfall values to reasonable range
        df[rainfall_col] = df[rainfall_col].clip(lower=0, upper=100)
        print(f"{watershed_name}: Rainfall clipped to 0–100 mm/day, new max: {df[rainfall_col].max():.2f}")
        return df
    except Exception as e:
        print(f"Error in normalize_rainfall for {watershed_name}: {str(e)}")
        return None

def predict_ssc(intermittent_path, continuous_path, watershed_name, qrf_params, is_excel_inter=False):
    """
    Predict daily SSC (g/L) for 1990–2020 using QRF and prepare data with engineered features.
    
    Args:
        intermittent_path (Path): Path to intermittent data (Excel for Gilgel Abay, CSV for Gumara).
        continuous_path (Path): Path to continuous data (CSV).
        watershed_name (str): Name of the watershed ('Gilgel Abay' or 'Gumara').
        qrf_params (dict): QRF hyperparameters.
        is_excel_inter (bool): True if intermittent data is in Excel format (default: False).
    
    Returns:
        pd.DataFrame: DataFrame with predicted SSC quantiles (Q25, Median, Q75) and features, or None if errors occur.
    """
    try:
        print(f"\nPredicting SSC for {watershed_name}...")
        # Check if input files exist
        if not intermittent_path.exists() or not continuous_path.exists():
            print(f"Error: File not found - Intermittent: {intermittent_path}, Continuous: {continuous_path}")
            return None

        # Load intermittent and continuous data
        df_inter = pd.read_excel(intermittent_path, engine='openpyxl') if is_excel_inter else pd.read_csv(intermittent_path)
        df_cont = pd.read_csv(continuous_path)
        print(f"{watershed_name} Intermittent Shape: {df_inter.shape}, Columns: {list(df_inter.columns)}")
        print(f"{watershed_name} Continuous Shape: {df_cont.shape}, Columns: {list(df_cont.columns)}")
        
        # Define column name mappings for flexibility
        column_mapping = {
            'Date': ['Date', 'date', 'Time', 'time', 'Timestamp', 'timestamp'],
            'Rainfall': ['Rainfall', 'rainfall', 'Rain', 'rain', 'Precipitation', 'precipitation', 'Rain_mm', 'rain_mm'],
            'Discharge': ['Discharge', 'discharge', 'Flow', 'flow'],
            'Temperature': ['Temperature', 'temperature', 'Temp', 'temp'],
            'ETo': ['ETo', 'eto', 'ET0', 'Evapotranspiration', 'evapotranspiration'],
            'SSC': ['SSC', 'ssc', 'SuspendedSediment', 'suspended_sediment']
        }
        
        # Rename columns to standard names
        for df, df_name in [(df_inter, 'intermittent'), (df_cont, 'continuous')]:
            for expected_col, alternatives in column_mapping.items():
                found = False
                for alt in alternatives:
                    for col in df.columns:
                        if col.lower() == alt.lower():
                            df.rename(columns={col: expected_col}, inplace=True)
                            found = True
                            break
                    if found:
                        break
                if not found and ((df_name == 'intermittent' and expected_col in ['Date', 'Rainfall', 'Discharge', 'Temperature', 'ETo', 'SSC']) or 
                                  (df_name == 'continuous' and expected_col in ['Date', 'Rainfall', 'Discharge', 'Temperature', 'ETo'])):
                    print(f"Error: {watershed_name} {df_name} missing column: {expected_col}")
                    return None
        
        # Convert Date columns to datetime and filter for 1990–2020
        df_inter['Date'] = pd.to_datetime(df_inter['Date'], errors='coerce')
        df_cont['Date'] = pd.to_datetime(df_cont['Date'], errors='coerce')
        df_inter = df_inter.dropna(subset=['Date', 'SSC'])
        df_cont = df_cont.dropna(subset=['Date'])
        
        df_inter = df_inter[(df_inter['Date'].dt.year >= 1990) & (df_inter['Date'].dt.year <= 2020)]
        df_cont = df_cont[(df_cont['Date'].dt.year >= 1990) & (df_cont['Date'].dt.year <= 2020)]
        
        # Check if data is empty after filtering
        if df_inter.empty or df_cont.empty:
            print(f"Error: {watershed_name} data empty after filtering - Intermittent: {len(df_inter)}, Continuous: {len(df_cont)}")
            return None
        
        # Normalize rainfall units
        df_inter = normalize_rainfall(df_inter, watershed_name)
        df_cont = normalize_rainfall(df_cont, watershed_name)
        if df_inter is None or df_cont is None:
            return None
        
        # Remove duplicate dates
        if df_inter['Date'].duplicated().any():
            print(f"Warning: {watershed_name} intermittent has {df_inter['Date'].duplicated().sum()} duplicates")
            df_inter = df_inter.drop_duplicates(subset='Date', keep='first')
        if df_cont['Date'].duplicated().any():
            print(f"Warning: {watershed_name} continuous has {df_cont['Date'].duplicated().sum()} duplicates")
            df_cont = df_cont.drop_duplicates(subset='Date', keep='first')
        
        # Convert columns to numeric and clip to non-negative
        numeric_cols = ['Rainfall', 'Discharge', 'Temperature', 'ETo', 'SSC']
        for col in numeric_cols:
            if col in df_inter:
                df_inter[col] = pd.to_numeric(df_inter[col], errors='coerce').clip(lower=0)
        for col in numeric_cols[:-1]:
            if col in df_cont:
                df_cont[col] = pd.to_numeric(df_cont[col], errors='coerce').clip(lower=0)
        
        # Add seasonal features
        df_inter = add_seasonal_features(df_inter)
        df_cont = add_seasonal_features(df_cont)
        if df_inter is None or df_cont is None:
            return None
        
        # Sort data by date
        df_inter = df_inter.sort_values('Date')
        df_cont = df_cont.sort_values('Date')
        
        # Add annual and cumulative rainfall features
        df_inter['Year'] = df_inter['Date'].dt.year
        df_cont['Year'] = df_cont['Date'].dt.year
        annual_rainfall_inter = df_inter.groupby('Year')['Rainfall'].sum().reset_index()
        annual_rainfall_inter.columns = ['Year', 'Annual_Rainfall']
        df_inter = df_inter.merge(annual_rainfall_inter, on='Year', how='left')
        annual_rainfall_cont = df_cont.groupby('Year')['Rainfall'].sum().reset_index()
        annual_rainfall_cont.columns = ['Year', 'Annual_Rainfall']
        df_cont = df_cont.merge(annual_rainfall_cont, on='Year', how='left')
        
        df_inter['Cumulative_Rainfall'] = df_inter.groupby('Year')['Rainfall'].cumsum()
        df_cont['Cumulative_Rainfall'] = df_cont.groupby('Year')['Rainfall'].cumsum()
        
        # Add moving average and lag features for Discharge
        df_inter['MA_Discharge_3'] = df_inter['Discharge'].rolling(window=3, min_periods=1).mean().bfill()
        df_inter['Lag_Discharge'] = df_inter['Discharge'].shift(1).bfill()
        df_inter['Lag_Discharge_3'] = df_inter['Discharge'].shift(3).bfill()
        df_cont['MA_Discharge_3'] = df_cont['Discharge'].rolling(window=3, min_periods=1).mean().bfill()
        df_cont['Lag_Discharge'] = df_cont['Discharge'].shift(1).bfill()
        df_cont['Lag_Discharge_3'] = df_cont['Discharge'].shift(3).bfill()
        
        # Define predictors for QRF model
        predictors = ['Discharge', 'MA_Discharge_3', 'Lag_Discharge', 'Lag_Discharge_3', 'Rainfall', 'ETo', 
                     'Temperature', 'Annual_Rainfall', 'Cumulative_Rainfall', 'Sin_Julian', 'Cos_Julian']
        
        # Handle missing or infinite values in predictors
        for df in [df_inter, df_cont]:
            for col in predictors:
                if col in df:
                    df[col] = df[col].fillna(0).replace([np.inf, -np.inf], 0)
        
        # Drop rows with missing predictors or SSC
        df_inter = df_inter.dropna(subset=predictors + ['SSC'])
        df_cont = df_cont.dropna(subset=predictors)
        if df_inter.empty or df_cont.empty:
            print(f"Error: {watershed_name} data empty after cleaning - Intermittent: {len(df_inter)}, Continuous: {len(df_cont)}")
            return None
        
        # Prepare training and prediction data
        X_inter = df_inter[predictors]
        y_inter = df_inter['SSC']
        X_cont = df_cont[predictors]
        
        # Train QRF model and predict SSC quantiles
        qrf = RandomForestQuantileRegressor(**qrf_params)
        qrf.fit(X_inter, y_inter)
        ssc_pred = qrf.predict(X_cont, quantiles=[0.25, 0.5, 0.75])
        
        # Print SSC prediction summary
        print(f"{watershed_name} SSC Prediction Summary (g/L):")
        for q, preds in zip([0.25, 0.5, 0.75], ssc_pred.T):
            print(f"Quantile {q}: {pd.Series(preds).describe()}")
        
        # Save feature importance
        feature_importance = pd.DataFrame({
            'Feature': predictors,
            'Importance': qrf.feature_importances_
        }).sort_values(by='Importance', ascending=False)
        print(f"{watershed_name} Feature Importance:")
        print(feature_importance)
        feature_importance.to_csv(WATERSHED_CONFIG[watershed_name]['output_dir'] / 
                                 f"feature_importance_seasonal_{watershed_name.lower().replace(' ', '_')}_70split.csv", index=False)
        
        # Add predicted SSC quantiles to continuous data
        df_cont['SSC_Q25'] = ssc_pred[:, 0]
        df_cont['SSC_Median'] = ssc_pred[:, 1]
        df_cont['SSC_Q75'] = ssc_pred[:, 2]
        return df_cont[['Date', 'Rainfall', 'Discharge', 'Temperature', 'ETo', 'SSC_Q25', 'SSC_Median', 'SSC_Q75', 
                        'Annual_Rainfall', 'Cumulative_Rainfall', 'Sin_Julian', 'Cos_Julian']]
    except Exception as e:
        print(f"Error in predict_ssc for {watershed_name}: {str(e)}")
        return None

def calculate_sediment_yield(df, watershed_name, area_km2, output_dir):
    """
    Calculate daily sediment yield (t/ha/day) with uncertainty using SSC predictions and discharge.
    
    Args:
        df (pd.DataFrame): DataFrame with SSC predictions and discharge data.
        watershed_name (str): Name of the watershed.
        area_km2 (float): Watershed area in km².
        output_dir (Path): Directory to save output Excel file.
    
    Returns:
        pd.DataFrame: DataFrame with calculated sediment yield (t/ha/day) for Q25, Median, Q75, or None if errors occur.
    """
    try:
        print(f"\nCalculating sediment yield for {watershed_name}...")
        output_dir.mkdir(exist_ok=True)
        
        # Check for required columns
        required_cols = ['Date', 'Rainfall', 'Discharge', 'SSC_Q25', 'SSC_Median', 'SSC_Q75']
        if not all(col in df for col in required_cols):
            print(f"Error: {watershed_name} missing required columns: {set(required_cols) - set(df.columns)}")
            return None
        
        # Calculate sediment yield (t/ha/day) using LOAD_FACTOR
        df['Sediment_Yield_Median'] = df['Discharge'] * df['SSC_Median'] * LOAD_FACTOR / (area_km2 * 100)
        df['Sediment_Yield_Q25'] = df['Discharge'] * df['SSC_Q25'] * LOAD_FACTOR / (area_km2 * 100)
        df['Sediment_Yield_Q75'] = df['Discharge'] * df['SSC_Q75'] * LOAD_FACTOR / (area_km2 * 100)
        
        # Clip sediment yield to avoid extreme values
        for col in ['Sediment_Yield_Median', 'Sediment_Yield_Q25', 'Sediment_Yield_Q75']:
            df[col] = df[col].clip(lower=0, upper=df[col].quantile(0.95))
        
        # Drop rows with missing required columns
        df = df.dropna(subset=required_cols + ['Sediment_Yield_Median', 'Sediment_Yield_Q25', 'Sediment_Yield_Q75'])
        if df.empty:
            print(f"Error: {watershed_name} data empty after cleaning")
            return None
        
        # Save daily data to Excel with unique filename
        output_path = output_dir / f"{watershed_name.replace(' ', '_')}_Daily_SSC_Sediment_Yield_{uuid4().hex[:8]}.xlsx"
        df.to_excel(output_path, index=False)
        print(f"Daily data saved to {output_path}")
        return df
    except Exception as e:
        print(f"Error in calculate_sediment_yield for {watershed_name}: {str(e)}")
        return None

def process_seasonal_data(df, watershed_name):
    """
    Process daily data to calculate seasonal sediment yield (t/ha/yr) for all 365 Julian Days.
    
    Args:
        df (pd.DataFrame): DataFrame with daily sediment yield data.
        watershed_name (str): Name of the watershed.
    
    Returns:
        tuple: (seasonal_data, df, stats) where seasonal_data is the aggregated seasonal DataFrame,
               df is the processed daily DataFrame, and stats contains seasonal statistics, or (None, None, None) if errors occur.
    """
    try:
        print(f"\nProcessing seasonal data for {watershed_name}...")
        df = df.copy()
        df['Year'] = df['Date'].dt.year
        
        # Filter data for 1990–2020
        df = df[(df['Year'] >= 1990) & (df['Year'] <= 2020)]
        if df.empty:
            print(f"Error: {watershed_name} data empty after year filtering")
            return None, None, None
        
        # Warn about data before 1990
        if (df['Year'] < 1990).any():
            print(f"Warning: {watershed_name} contains {sum(df['Year'] < 1990)} rows before 1990")
            df = df[df['Year'] >= 1990]
        
        # Remove duplicate dates
        if df['Date'].duplicated().any():
            print(f"Warning: {watershed_name} has {df['Date'].duplicated().sum()} duplicate dates")
            df = df.drop_duplicates(subset='Date', keep='first')
        
        # Assign seasons based on Julian Day (Wet: 152–304, Dry: else)
        df['Julian_Day'] = df['Date'].dt.dayofyear
        df['Season'] = df['Julian_Day'].apply(lambda x: 'Wet' if 152 <= x <= 304 else 'Dry')
        
        # Convert daily sediment yield to annual (t/ha/yr) based on season duration
        df['Sediment_Yield_Median_tons_ha_yr'] = df['Sediment_Yield_Median'] * np.where(df['Season'] == 'Wet', 153, 212)
        df['Sediment_Yield_Q25_tons_ha_yr'] = df['Sediment_Yield_Q25'] * np.where(df['Season'] == 'Wet', 153, 212)
        df['Sediment_Yield_Q75_tons_ha_yr'] = df['Sediment_Yield_Q75'] * np.where(df['Season'] == 'Wet', 153, 212)
        
        # Aggregate to seasonal data by Julian Day
        seasonal_data = df.groupby(['Julian_Day', 'Season'])[
            ['Sediment_Yield_Median_tons_ha_yr', 'Sediment_Yield_Q25_tons_ha_yr', 'Sediment_Yield_Q75_tons_ha_yr', 'Rainfall']
        ].mean().reset_index()
        
        if seasonal_data.empty:
            print(f"Error: {watershed_name} seasonal data empty after aggregation")
            return None, None, None
        
        # Ensure all 365 Julian Days are included
        all_julian_days = pd.DataFrame({
            'Julian_Day': range(1, 366),
            'Season': ['Dry' if (jd < 152 or jd > 304) else 'Wet' for jd in range(1, 366)]
        })
        seasonal_data = all_julian_days.merge(seasonal_data, on=['Julian_Day', 'Season'], how='left')
        seasonal_data.fillna({'Sediment_Yield_Median_tons_ha_yr': 0, 'Sediment_Yield_Q25_tons_ha_yr': 0, 
                             'Sediment_Yield_Q75_tons_ha_yr': 0, 'Rainfall': 0}, inplace=True)
        
        # Clip rainfall to configured maximum
        seasonal_data['Rainfall'] = seasonal_data['Rainfall'].clip(lower=0, upper=WATERSHED_CONFIG[watershed_name]['rainfall_max'])
        
        # Calculate seasonal statistics
        stats = df.groupby('Season')[['Sediment_Yield_Median_tons_ha_yr', 'Sediment_Yield_Q25_tons_ha_yr', 
                                     'Sediment_Yield_Q75_tons_ha_yr']].mean().reset_index()
        print(f"{watershed_name} Seasonal Yield Stats (t/ha/yr):")
        print(stats)
        wet_mean = stats[stats['Season'] == 'Wet']['Sediment_Yield_Median_tons_ha_yr'].iloc[0] if 'Wet' in stats['Season'].values else 0
        dry_mean = stats[stats['Season'] == 'Dry']['Sediment_Yield_Median_tons_ha_yr'].iloc[0] if 'Dry' in stats['Season'].values else 0
        if wet_mean + dry_mean > 0:
            print(f"{watershed_name} Wet Season Contribution: {wet_mean / (wet_mean + dry_mean) * 100:.1f}%")
        
        return seasonal_data, df, stats
    except Exception as e:
        print(f"Error in process_seasonal_data for {watershed_name}: {str(e)}")
        return None, None, None

def plot_watershed(ax, seasonal_data, stats, watershed_name, is_color=True):
    """
    Plot seasonal sediment yield and rainfall for one watershed on a given axis.
    
    Args:
        ax (matplotlib.axes.Axes): Axis to plot on.
        seasonal_data (pd.DataFrame): Aggregated seasonal data with sediment yield and rainfall.
        stats (pd.DataFrame): Seasonal statistics for text annotations.
        watershed_name (str): Name of the watershed.
        is_color (bool): True for color plot, False for grayscale (default: True).
    
    Returns:
        tuple: (lines, labels) for legend creation, or ([], []) if errors occur.
    """
    try:
        rainfall_max = WATERSHED_CONFIG[watershed_name]['rainfall_max']
        max_yield = max(80, seasonal_data['Sediment_Yield_Median_tons_ha_yr'].max() * 1.1)
        
        # Warn if Julian Days are incomplete
        if len(seasonal_data['Julian_Day'].unique()) != 365:
            print(f"Warning: {watershed_name} has {len(seasonal_data['Julian_Day'].unique())} unique Julian Days, expected 365")
        
        # Print rainfall data summary
        print(f"{watershed_name} Rainfall Data Summary (mm):")
        print(seasonal_data['Rainfall'].describe())
        if seasonal_data['Rainfall'].max() == 0:
            print(f"Warning: {watershed_name} Rainfall data is all zeros")
        
        # Define colors for plot elements
        colors = {
            'rainfall': '#2ca02c' if is_color else '#666666',
            'wet': '#1f77b4' if is_color else '#000000',
            'dry': '#FF8C00' if is_color else '#999999'
        }
        
        # Create twin axis for rainfall
        ax_rain = ax.twinx()
        bars = ax_rain.bar(seasonal_data['Julian_Day'], seasonal_data['Rainfall'], color=colors['rainfall'], 
                           alpha=0.7, width=4, hatch='///', label='Rainfall (mm)', zorder=1)
        
        # Split dry season data for plotting
        dry_data = seasonal_data[seasonal_data['Season'] == 'Dry']
        dry_before = dry_data[dry_data['Julian_Day'] < 152]
        dry_after = dry_data[dry_data['Julian_Day'] > 304]
        wet_data = seasonal_data[seasonal_data['Season'] == 'Wet']
        
        # Check for missing season data
        if wet_data.empty or dry_data.empty:
            print(f"Error: {watershed_name} missing Wet ({len(wet_data)}) or Dry ({len(dry_data)}) data")
            ax.text(0.5, 0.5, 'Data Unavailable', fontsize=14, ha='center', va='center')
            return [], []
        
        # Plot dry season sediment yield
        line_dry = ax.plot(dry_before['Julian_Day'], dry_before['Sediment_Yield_Median_tons_ha_yr'], 
                           color=colors['dry'], linestyle='--', label='Dry Mean', linewidth=2, zorder=2)[0]
        ax.plot(dry_after['Julian_Day'], dry_after['Sediment_Yield_Median_tons_ha_yr'], 
                color=colors['dry'], linestyle='--', linewidth=2, zorder=2)
        
        # Plot wet season sediment yield
        line_wet = ax.plot(wet_data['Julian_Day'], wet_data['Sediment_Yield_Median_tons_ha_yr'], 
                           color=colors['wet'], linestyle='-', label='Wet Mean', linewidth=2, zorder=2)[0]
        
        # Add uncertainty bands (IQR)
        ax.fill_between(wet_data['Julian_Day'], wet_data['Sediment_Yield_Q25_tons_ha_yr'], 
                        wet_data['Sediment_Yield_Q75_tons_ha_yr'], color=colors['wet'], alpha=0.2, 
                        hatch='\\', label='Wet IQR', zorder=1)
        ax.fill_between(dry_before['Julian_Day'], dry_before['Sediment_Yield_Q25_tons_ha_yr'], 
                        dry_before['Sediment_Yield_Q75_tons_ha_yr'], color=colors['dry'], alpha=0.2, 
                        hatch='//', label='Dry IQR', zorder=1)
        ax.fill_between(dry_after['Julian_Day'], dry_after['Sediment_Yield_Q25_tons_ha_yr'], 
                        dry_after['Sediment_Yield_Q75_tons_ha_yr'], color=colors['dry'], alpha=0.2, 
                        hatch='//', zorder=1)
        
        # Add seasonal statistics as text annotations
        if 'Wet' in stats['Season'].values:
            wet_stats = stats[stats['Season'] == 'Wet'].iloc[0]
            ax.text(0.02, 0.6, f"Wet: Mean={wet_stats['Sediment_Yield_Median_tons_ha_yr']:.2f}\n"
                               f"Q25={wet_stats['Sediment_Yield_Q25_tons_ha_yr']:.2f}\n"
                               f"Q75={wet_stats['Sediment_Yield_Q75_tons_ha_yr']:.2f}",
                    transform=ax.transAxes, fontsize=12, verticalalignment='top',
                    color=colors['wet'], bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor=colors['wet']))
        if 'Dry' in stats['Season'].values:
            dry_stats = stats[stats['Season'] == 'Dry'].iloc[0]
            ax.text(0.98, 0.6, f"Dry: Mean={dry_stats['Sediment_Yield_Median_tons_ha_yr']:.2f}\n"
                               f"Q25={dry_stats['Sediment_Yield_Q25_tons_ha_yr']:.2f}\n"
                               f"Q75={dry_stats['Sediment_Yield_Q75_tons_ha_yr']:.2f}",
                    transform=ax.transAxes, fontsize=12, verticalalignment='top', horizontalalignment='right',
                    color=colors['dry'], bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor=colors['dry']))
        
        # Set plot title and labels
        ax.set_title(f"({'a' if watershed_name == 'Gilgel Abay' else 'b'}) {watershed_name}", fontsize=20)
        ax.set_xlabel('Julian Day', fontsize=16)
        ax_rain.set_ylabel('Rainfall (mm)', color=colors['rainfall'], fontsize=16)
        ax.set_ylim(0, max_yield)
        ax_rain.set_ylim(rainfall_max, 0)
        ax_rain.set_yticks(np.arange(0, rainfall_max + 20, 20))
        ax_rain.tick_params(axis='y', colors=colors['rainfall'], labelsize=14)
        ax.set_xlim(1, 365)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.tick_params(axis='both', labelsize=14)
        ax.grid(True, axis='y', linestyle='--', alpha=0.3)
        
        # Print axis limits for debugging
        print(f"{watershed_name} Sediment Yield Axis Limits: 0 to {max_yield:.2f} t/ha/yr")
        print(f"{watershed_name} Rainfall Axis Limits: {rainfall_max} to 0 mm")
        
        return [line_wet, line_dry, bars], ['Wet Mean', 'Dry Mean', 'Rainfall (mm)']
    except Exception as e:
        print(f"Error plotting {watershed_name}: {str(e)}")
        ax.text(0.5, 0.5, 'Data Unavailable', fontsize=14, ha='center', va='center')
        return [], []

def create_side_by_side_plot(data_dict, output_dir):
    """
    Generate Figure 9: Side-by-side seasonal sediment yield plots for Gilgel Abay and Gumara.
    
    Args:
        data_dict (dict): Dictionary containing seasonal_data, df, and stats for each watershed.
        output_dir (Path): Directory to save output plot files.
    """
    try:
        print("\nGenerating Figure 9...")
        watersheds = ['Gilgel Abay', 'Gumara']
        # Check if data is available for both watersheds
        if not all(w in data_dict for w in watersheds):
            print(f"Error: Missing data for {set(watersheds) - set(data_dict.keys())}")
            return

        # Color plot
        plt.clf()
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
        fig.patch.set_facecolor('white')
        ax1.set_ylabel('Mean Sediment Yield\n(t/ha/yr)', fontsize=16)
        
        all_lines, all_labels = [], []
        # Plot each watershed
        for ax, watershed_name in zip([ax1, ax2], watersheds):
            lines, labels = plot_watershed(ax, data_dict[watershed_name]['seasonal_data'], 
                                          data_dict[watershed_name]['stats'], watershed_name, is_color=True)
            if watershed_name == 'Gilgel Abay':
                all_lines.extend(lines)
                all_labels.extend(labels)
        
        # Add shared legend
        fig.legend(all_lines, all_labels, loc='lower center', bbox_to_anchor=(0.5, -0.05), ncol=3, fontsize=14)
        plt.tight_layout(pad=2.0)
        
        # Save color plot
        output_png = output_dir / f'Figure9_Seasonal_Sediment_Yield_{uuid4().hex[:8]}_color.png'
        output_svg = output_dir / f'Figure9_Seasonal_Sediment_Yield_{uuid4().hex[:8]}_color.svg'
        plt.savefig(output_png, dpi=600, format='png', bbox_inches='tight')
        plt.savefig(output_svg, format='svg', bbox_inches='tight')
        print(f"Color plots saved to {output_png} (PNG) and {output_svg} (SVG)")
        
        # Grayscale plot
        plt.clf()
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
        fig.patch.set_facecolor('white')
        ax1.set_ylabel('Mean Sediment Yield\n(t/ha/yr)', fontsize=16)
        
        all_lines, all_labels = [], []
        for ax, watershed_name in zip([ax1, ax2], watersheds):
            lines, labels = plot_watershed(ax, data_dict[watershed_name]['seasonal_data'], 
                                          data_dict[watershed_name]['stats'], watershed_name, is_color=False)
            if watershed_name == 'Gilgel Abay':
                all_lines.extend(lines)
                all_labels.extend(labels)
        
        # Add shared legend
        fig.legend(all_lines, all_labels, loc='lower center', bbox_to_anchor=(0.5, -0.05), ncol=3, fontsize=14)
        plt.tight_layout(pad=2.0)
        
        # Save grayscale plot
        output_png = output_dir / f'Figure9_Seasonal_Sediment_Yield_{uuid4().hex[:8]}_grayscale.png'
        output_eps = output_dir / f'Figure9_Seasonal_Sediment_Yield_{uuid4().hex[:8]}_grayscale.eps'
        plt.savefig(output_png, dpi=600, format='png', bbox_inches='tight')
        plt.savefig(output_eps, format='eps', bbox_inches='tight')
        print(f"Grayscale plots saved to {output_png} (PNG) and {output_eps} (EPS)")
        
        plt.show()
        plt.close()
    except Exception as e:
        print(f"Error in create_side_by_side_plot: {str(e)}")

def main():
    """
    Main function to process data for both watersheds and generate Figure 9.
    """
    try:
        print("Starting script execution...")
        data_dict = {}
        # Process each watershed
        for watershed_name, params in WATERSHED_CONFIG.items():
            print(f"\n=== Processing {watershed_name} ===")
            
            # Load QRF parameters
            qrf_params = load_best_params(watershed_name, params['output_dir'])
            if qrf_params is None:
                continue
            
            # Predict SSC
            df_cont = predict_ssc(params['intermittent'], params['continuous'], watershed_name, 
                                  qrf_params, is_excel_inter=(watershed_name == 'Gilgel Abay'))
            if df_cont is None:
                continue
            
            # Calculate daily sediment yield
            daily_data = calculate_sediment_yield(df_cont, watershed_name, params['area_km2'], params['output_dir'])
            if daily_data is None:
                continue
            
            # Process seasonal data
            seasonal_data, df, stats = process_seasonal_data(daily_data, watershed_name)
            if seasonal_data is None:
                continue
            
            # Save seasonal data to Excel
            seasonal_data.to_excel(params['output_csv'], index=False)
            print(f"Seasonal data saved to {params['output_csv']}")
            
            data_dict[watershed_name] = {'seasonal_data': seasonal_data, 'df': df, 'stats': stats}
        
        # Generate Figure 9 if data is available
        if data_dict:
            create_side_by_side_plot(data_dict, WATERSHED_CONFIG['Gilgel Abay']['output_dir'])
        else:
            print("Error: No data processed successfully. Figure 9 cannot be generated.")
    except Exception as e:
        print(f"Error in main: {str(e)}")

if __name__ == "__main__":
    main()
