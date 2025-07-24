# Script to generate Figure 8: Monthly sediment yield, rainfall, and discharge for Gilgel Abay and Gumara watersheds
# (Section 3.3). Predicts daily SSC (g/L) for 1990–2020 using Quantile Random Forest (QRF) trained on intermittent data,
# calculates daily sediment yield (t/ha/day), aggregates to monthly values (t/ha/month), and produces a combination plot
# with bar plot for monthly rainfall (mm, reversed axis) and line plots for discharge (m³/s) and sediment yield (t/ha/month).
# Outputs daily Excel for WASA-SED, monthly CSV, feature importance, and PNG/SVG plots.
# Loads QRF parameters from best_params_comparison_70split.csv.
# Author: Kindie B. Worku
# Date: 2025-07-19


%matplotlib inline
import pandas as pd
import numpy as np
from quantile_forest import RandomForestQuantileRegressor
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from matplotlib.ticker import MaxNLocator
import warnings
import ast
import os
from uuid import uuid4
from matplotlib.dates import YearLocator, DateFormatter

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Set plot style for publication quality
sns.set_style('white')
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 18


# Configuration
USE_LAG_RAINFALL = False  # Excludes Lag_Rainfall_7 and Lag_Rainfall_14

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

LOAD_FACTOR = 86.4  # Converts m³/s × g/L to t/day (86,400 s/day × 10⁻⁶ t/g)

def add_seasonal_features(df):
    df = df.copy()
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df['Julian_Day'] = df['Date'].dt.dayofyear
    df['Sin_Julian'] = np.sin(2 * np.pi * df['Julian_Day'] / 365.25)
    df['Cos_Julian'] = np.cos(2 * np.pi * df['Julian_Day'] / 365.25)
    return df.drop(columns=['Julian_Day'])

def load_best_params(watershed_name, output_dir):
    params_path = output_dir / "best_params_comparison_70split.csv"
    if not params_path.exists():
        print(f"Error: {params_path} not found")
        return None
    params_df = pd.read_csv(params_path)
    qrf_params_row = params_df[(params_df['Model'] == 'QRF') & (params_df['Watershed'] == watershed_name)]
    if qrf_params_row.empty:
        print(f"Error: No QRF parameters for {watershed_name} in {params_path}")
        return None
    qrf_params = ast.literal_eval(qrf_params_row['Parameters'].iloc[0])
    print(f"Loaded QRF params for {watershed_name}: {qrf_params}")
    return qrf_params


def predict_ssc(intermittent_path, continuous_path, watershed_name, qrf_params, is_excel_inter=False):
    print(f"\nPredicting SSC for {watershed_name}...")
    
    if not intermittent_path.exists():
        print(f"Intermittent file not found: {intermittent_path}")
        return None
    if not continuous_path.exists():
        print(f"Continuous file not found: {continuous_path}")
        return None
    
    try:
        if is_excel_inter:
            df_inter = pd.read_excel(intermittent_path, engine='openpyxl')
        else:
            df_inter = pd.read_csv(intermittent_path)
        df_cont = pd.read_csv(continuous_path)
        print(f"{watershed_name} Intermittent Data Shape: {df_inter.shape}, Columns: {list(df_inter.columns)}")
        print(f"{watershed_name} Continuous Data Shape: {df_cont.shape}, Columns: {list(df_cont.columns)}")
    except Exception as e:
        print(f"Error loading data for {watershed_name}: {str(e)}")
        return None
  
  
    column_mapping = {
        'Date': ['Date', 'date', 'Time', 'time', 'Timestamp', 'timestamp'],
        'Rainfall': ['Rainfall', 'rainfall', 'Rain', 'rain'],
        'Discharge': ['Discharge', 'discharge', 'Flow', 'flow'],
        'Temperature': ['Temperature', 'temperature', 'Temp', 'temp'],
        'ETo': ['ETo', 'eto', 'ET0', 'Evapotranspiration', 'evapotranspiration'],
        'SSC': ['SSC', 'ssc', 'SuspendedSediment', 'suspended_sediment']
    }
    
    for df, df_name in [(df_inter, 'intermittent'), (df_cont, 'continuous')]:
        for expected_col, alternatives in column_mapping.items():
            found = False
            for alt in alternatives:
                if alt in df.columns:
                    df.rename(columns={alt: expected_col}, inplace=True)
                    found = True
                    break
            if not found:
                if (df_name == 'intermittent' and expected_col in ['Date', 'Rainfall', 'Discharge', 'Temperature', 'ETo', 'SSC']) or \
                   (df_name == 'continuous' and expected_col in ['Date', 'Rainfall', 'Discharge', 'Temperature', 'ETo']):
                    print(f"Error: {watershed_name} {df_name} data missing column: {expected_col}. Available: {list(df.columns)}")
                    return None
    

    df_inter['Date'] = pd.to_datetime(df_inter['Date'], errors='coerce')
    df_cont['Date'] = pd.to_datetime(df_cont['Date'], errors='coerce')
    df_inter = df_inter.dropna(subset=['Date', 'SSC'])
    df_cont = df_cont.dropna(subset=['Date'])
    
    # Strict year filtering to exclude 1989
    df_inter = df_inter[(df_inter['Date'].dt.year >= 1990) & (df_inter['Date'].dt.year <= 2020)]
    df_cont = df_cont[(df_cont['Date'].dt.year >= 1990) & (df_cont['Date'].dt.year <= 2020)]
    
    if df_inter.empty:
        print(f"{watershed_name} intermittent data empty after year filtering")
        return None
    if df_cont.empty:
        print(f"{watershed_name} continuous data empty after year filtering")
        return None
  
  
    # Stricter duplicate date removal with diagnostics
    if df_inter['Date'].duplicated().any():
        duplicates = df_inter[df_inter['Date'].duplicated(keep=False)][['Date', 'Discharge', 'Rainfall', 'SSC']].sort_values('Date')
        print(f"Warning: {watershed_name} intermittent data has {df_inter['Date'].duplicated().sum()} duplicate dates:")
        print(duplicates.head(10))
        df_inter = df_inter.drop_duplicates(subset='Date', keep='first')
    if df_cont['Date'].duplicated().any():
        duplicates = df_cont[df_cont['Date'].duplicated(keep=False)][['Date', 'Discharge', 'Rainfall']].sort_values('Date')
        print(f"Warning: {watershed_name} continuous data has {df_cont['Date'].duplicated().sum()} duplicate dates:")
        print(duplicates.head(10))
        df_cont = df_cont.drop_duplicates(subset='Date', keep='first')
    
    print(f"{watershed_name} Intermittent Date Range after deduplication: {df_inter['Date'].min()} to {df_inter['Date'].max()}")
    print(f"{watershed_name} Continuous Date Range after deduplication: {df_cont['Date'].min()} to {df_cont['Date'].max()}")
    
    numeric_cols = ['Rainfall', 'Discharge', 'Temperature', 'ETo', 'SSC']
    for col in numeric_cols:
        if col in df_inter:
            df_inter[col] = pd.to_numeric(df_inter[col], errors='coerce').clip(lower=0)
    for col in numeric_cols[:-1]:
        df_cont[col] = pd.to_numeric(df_cont[col], errors='coerce').clip(lower=0)
    
    df_inter = add_seasonal_features(df_inter)
    df_cont = add_seasonal_features(df_cont)
  
  
    df_inter = df_inter.sort_values('Date')
    df_cont = df_cont.sort_values('Date')
    
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
    
    df_inter['MA_Discharge_3'] = df_inter['Discharge'].rolling(window=3, min_periods=1).mean().bfill()
    df_inter['Lag_Discharge'] = df_inter['Discharge'].shift(1).bfill()
    df_inter['Lag_Discharge_3'] = df_inter['Discharge'].shift(3).bfill()
    df_cont['MA_Discharge_3'] = df_cont['Discharge'].rolling(window=3, min_periods=1).mean().bfill()
    df_cont['Lag_Discharge'] = df_cont['Discharge'].shift(1).bfill()
    df_cont['Lag_Discharge_3'] = df_cont['Discharge'].shift(3).bfill()
    
    predictors = ['Discharge', 'MA_Discharge_3', 'Lag_Discharge', 'Lag_Discharge_3', 'Rainfall', 'ETo', 
                 'Temperature', 'Annual_Rainfall', 'Cumulative_Rainfall', 'Sin_Julian', 'Cos_Julian']
    
    for df, df_name in [(df_inter, 'intermittent'), (df_cont, 'continuous')]:
        for col in predictors:
            if col in df and (df[col].isna().any() or np.isinf(df[col]).any()):
                df[col] = df[col].fillna(0).replace([np.inf, -np.inf], 0)
    

    df_inter = df_inter.dropna(subset=predictors + ['SSC'])
    df_cont = df_cont.dropna(subset=predictors)
    print(f"{watershed_name} Intermittent Data after cleaning: {len(df_inter)} rows")
    print(f"{watershed_name} Continuous Data after cleaning: {len(df_cont)} rows")
    
    if df_inter.empty:
        print(f"{watershed_name} intermittent data empty after cleaning")
        return None
    
    X_inter = df_inter[predictors]
    y_inter = df_inter['SSC']
    X_cont = df_cont[predictors]
    
    try:
        qrf = RandomForestQuantileRegressor(**qrf_params)
        qrf.fit(X_inter, y_inter)
        ssc_pred = qrf.predict(X_cont, quantiles=[0.05, 0.25, 0.5, 0.75, 0.95])
        print(f"{watershed_name} SSC Prediction Summary (g/L):")
        for q, preds in zip([0.05, 0.25, 0.5, 0.75, 0.95], ssc_pred.T):
            print(f"Quantile {q}: {pd.Series(preds).describe()}")

        
        feature_importance = pd.DataFrame({
            'Feature': predictors,
            'Importance': qrf.feature_importances_
        }).sort_values(by='Importance', ascending=False)
        print(f"{watershed_name} Feature Importance:")
        print(feature_importance)
        feature_importance.to_csv(WATERSHED_CONFIG[watershed_name]['output_dir'] / 
                                 f"feature_importance_monthly_{watershed_name.lower().replace(' ', '_')}_70split.csv", index=False)
    
    except Exception as e:
        print(f"Error training/predicting QRF for {watershed_name}: {str(e)}")
        return None
    
    df_cont['SSC'] = ssc_pred[:, 2]
    df_cont['SSC_Q05'] = ssc_pred[:, 0]
    df_cont['SSC_Q25'] = ssc_pred[:, 1]
    df_cont['SSC_Q75'] = ssc_pred[:, 3]
    df_cont['SSC_Q95'] = ssc_pred[:, 4]
    return df_cont[['Date', 'Rainfall', 'Discharge', 'Temperature', 'ETo', 'SSC', 'SSC_Q05', 'SSC_Q25', 'SSC_Q75', 'SSC_Q95', 
                    'Annual_Rainfall', 'Cumulative_Rainfall', 'Sin_Julian', 'Cos_Julian']]

def calculate_sediment_yield(df, watershed_name, area_km2, output_dir):
    print(f"\nCalculating sediment yield for {watershed_name}...")
    output_dir.mkdir(exist_ok=True)

   
    df['Sediment_Yield'] = df['Discharge'] * df['SSC'] * LOAD_FACTOR / (area_km2 * 100)
    df['Sediment_Yield_Q05'] = df['Discharge'] * df['SSC_Q05'] * LOAD_FACTOR / (area_km2 * 100)
    df['Sediment_Yield_Q25'] = df['Discharge'] * df['SSC_Q25'] * LOAD_FACTOR / (area_km2 * 100)
    df['Sediment_Yield_Q75'] = df['Discharge'] * df['SSC_Q75'] * LOAD_FACTOR / (area_km2 * 100)
    df['Sediment_Yield_Q95'] = df['Discharge'] * df['SSC_Q95'] * LOAD_FACTOR / (area_km2 * 100)
    
    # Clip negative values and cap outliers
    for col in ['Sediment_Yield', 'Sediment_Yield_Q05', 'Sediment_Yield_Q25', 'Sediment_Yield_Q75', 'Sediment_Yield_Q95']:
        df[col] = df[col].clip(lower=0, upper=df[col].quantile(0.95))
    
    df = df.dropna(subset=['Date', 'Rainfall', 'Discharge', 'SSC', 'Sediment_Yield', 'Sediment_Yield_Q05', 
                          'Sediment_Yield_Q25', 'Sediment_Yield_Q75', 'Sediment_Yield_Q95', 'Annual_Rainfall'])
    print(f"{watershed_name} Data after dropping NaNs: {len(df)} rows")
    
    if df.empty:
        print(f"{watershed_name} data empty after cleaning")
        return None
    
    output_path = output_dir / f"{watershed_name.replace(' ', '_')}_Daily_SSC_Sediment_Yield_{uuid4().hex[:8]}.xlsx"
    df.to_excel(output_path, index=False)
    print(f"Daily data saved to {output_path}")
    
    return df

def process_monthly_data(df, watershed_name):
    print(f"\nProcessing monthly data for {watershed_name}...")
 
   
    df = df.copy()
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    
    # Strict year 
    df = df[(df['Year'] >= 1990) & (df['Year'] <= 2020)]
    if df.empty:
        print(f"{watershed_name} data empty after year filtering")
        return None
    
    # Check for 1989 explicitly
    if (df['Year'] < 1990).any():
        print(f"Warning: {watershed_name} data contains {sum(df['Year'] < 1990)} rows before 1990")
        df = df[df['Year'] >= 1990]
    
    # Stricter duplicate date removal
    if df['Date'].duplicated().any():
        duplicates = df[df['Date'].duplicated(keep=False)][['Date', 'Discharge', 'Rainfall', 'SSC']].sort_values('Date')
        print(f"Warning: {watershed_name} has {df['Date'].duplicated().sum()} duplicate dates:")
        print(duplicates.head(10))
        df = df.drop_duplicates(subset='Date', keep='first')
    
    print(f"{watershed_name} Date Range: {df['Date'].min()} to {df['Date'].max()}")
    
    # Group by Year and Month
    monthly_data = df.groupby([df['Year'], df['Month']]).agg({
        'Discharge': 'mean',
        'Rainfall': 'sum',
        'SSC': 'mean',
        'Sediment_Yield': 'sum',
        'Date': lambda x: x.nunique()  # Count unique days
    }).reset_index()
    
   
 monthly_data = monthly_data.rename(columns={'Date': 'Days_in_Month'})
    
    # Filter out sparse months (<24 days)
    sparse_months = monthly_data[monthly_data['Days_in_Month'] < 24]
    if not sparse_months.empty:
        print(f"Warning: Dropping {len(sparse_months)} sparse months (<24 days) for {watershed_name}:")
        print(sparse_months[['Year', 'Month', 'Days_in_Month']])
        monthly_data = monthly_data[monthly_data['Days_in_Month'] >= 24]
    
    if monthly_data.empty:
        print(f"{watershed_name} monthly data empty after filtering sparse months")
        return None
    
    # Create complete monthly index
    monthly_data['Date'] = pd.to_datetime(monthly_data[['Year', 'Month']].assign(day=1))
    
    # Check for duplicate Year-Month
    if monthly_data[['Year', 'Month']].duplicated().any():
        duplicates = monthly_data[monthly_data[['Year', 'Month']].duplicated(keep=False)][['Year', 'Month', 'Discharge', 'Rainfall']]
        print(f"Warning: {watershed_name} has {monthly_data[['Year', 'Month']].duplicated().sum()} duplicate Year-Month combinations:")
        print(duplicates)
        monthly_data = monthly_data.drop_duplicates(subset=['Year', 'Month'], keep='first')
    
    # Reindex to ensure complete 1990–2020 monthly range
    expected_dates = pd.date_range(start='1990-01-01', end='2020-12-01', freq='MS')
    monthly_data = monthly_data.set_index('Date').reindex(expected_dates, method=None).reset_index()
    monthly_data = monthly_data.rename(columns={'index': 'Date'})
    monthly_data[['Discharge', 'Rainfall', 'SSC', 'Sediment_Yield']] = monthly_data[['Discharge', 'Rainfall', 'SSC', 'Sediment_Yield']].fillna(0)
    monthly_data['Days_in_Month'] = monthly_data['Days_in_Month'].fillna(0).astype(int)
    
    monthly_data = monthly_data.set_index('Date')
    monthly_data['Monthly_Sediment_Yield_tons_ha'] = monthly_data['Sediment_Yield']
  
  
    # Clip outliers using rainfall_max from WATERSHED_CONFIG
    rainfall_max = WATERSHED_CONFIG[watershed_name]['rainfall_max']
    monthly_data['Rainfall'] = monthly_data['Rainfall'].clip(lower=0, upper=rainfall_max)
    monthly_data['Discharge'] = monthly_data['Discharge'].clip(lower=0, upper=monthly_data['Discharge'].quantile(0.95) if monthly_data['Discharge'].max() > 0 else 1)
    monthly_data['Monthly_Sediment_Yield_tons_ha'] = monthly_data['Monthly_Sediment_Yield_tons_ha'].clip(lower=0, upper=monthly_data['Monthly_Sediment_Yield_tons_ha'].quantile(0.95) if monthly_data['Monthly_Sediment_Yield_tons_ha'].max() > 0 else 1)
    
    # Log diagnostics
    print(f"{watershed_name} Months in Monthly Data: {len(monthly_data)}")
    print(f"{watershed_name} Monthly Date Range: {monthly_data.index.min()} to {monthly_data.index.max()}")
    if monthly_data.index.duplicated().any():
        print(f"Error: {watershed_name} monthly data has {monthly_data.index.duplicated().sum()} duplicate indices")
        return None
    
    return monthly_data
# creating feagure 8 (Section 3.3 -manuscript)
def create_figure8(monthly_data_dict, output_dir):
    print("\nGenerating Figure 8...")
    
    watersheds = ['Gilgel Abay', 'Gumara']
    missing_watersheds = [w for w in watersheds if w not in monthly_data_dict]
    if missing_watersheds:
        print(f"Error generating Figure 8: Missing data for watersheds: {missing_watersheds}")
        return
    
    # Clear previous plots
    plt.clf()
    plt.cla()
    
    # Color version for online
    fig, axes = plt.subplots(1, 2, figsize=(16, 5), sharey=False)
    fig.patch.set_facecolor('white')
    
    for idx, watershed_name in enumerate(watersheds):
        ax1_rain = axes[idx]
        ax1_rain.set_facecolor('white')
        monthly_data = monthly_data_dict[watershed_name]
        
        # Validate index
        if monthly_data.index.isna().any():
            print(f"Error: {watershed_name} monthly data has NaN indices")
            return
        if monthly_data.index.duplicated().any():
            print(f"Error: {watershed_name} monthly data has {monthly_data.index.duplicated().sum()} duplicate indices")
            return
        
        # Ensure monthly dates
        expected_dates = pd.date_range(start='1990-01-01', end='2020-12-01', freq='MS')
        if not monthly_data.index.equals(expected_dates):
            print(f"Warning: {watershed_name} monthly data index does not match expected range")
            print(f"Expected: {len(expected_dates)} months, Got: {len(monthly_data)} months")
        
        # Check for 1989
        if (monthly_data.index < '1990-01-01').any():
            print(f"Error: {watershed_name} monthly data contains dates before 1990")
            monthly_data = monthly_data[monthly_data.index >= '1990-01-01']
        
        # Rainfall data diagnostics
        print(f"{watershed_name} Rainfall Data Summary (mm):")
        print(monthly_data['Rainfall'].describe())
        if monthly_data['Rainfall'].max() == 0:
            print(f"Warning: {watershed_name} Rainfall data is all zeros")
        
        discharge_max = WATERSHED_CONFIG[watershed_name]['discharge_max']
        yield_max = WATERSHED_CONFIG[watershed_name]['yield_max']
        rainfall_max = WATERSHED_CONFIG[watershed_name]['rainfall_max']
        
        monthly_data['Rainfall'] = monthly_data['Rainfall'].clip(lower=0)
        
        # Plot rainfall bars with hatching
        bars = ax1_rain.bar(monthly_data.index, monthly_data['Rainfall'], color='#2ca02c', alpha=0.4, width=50, 
                            hatch='///', label='Rainfall (mm)', zorder=1)
        
        ax2_discharge = ax1_rain.twinx()
        ax3_sediment = ax1_rain.twinx()
        ax3_sediment.spines['right'].set_position(('outward', 80))
        
        line_discharge = ax2_discharge.plot(monthly_data.index, monthly_data['Discharge'], color='#1f77b4', 
                                           linestyle='--', linewidth=2.5, label='Discharge (m³/s)', zorder=2)[0]  # Dashed line
        line_sediment = ax3_sediment.plot(monthly_data.index, monthly_data['Monthly_Sediment_Yield_tons_ha'],
                                          color='#d62728', linestyle='-', linewidth=2.5, 
                                          label='Sediment Yield (t/ha/month)', zorder=2)[0]  # Solid line
        
        ax1_rain.set_title(f"({'a' if watershed_name == 'Gilgel Abay' else 'b'}) {watershed_name}", fontsize=20)
        ax1_rain.set_xlabel('Year', fontsize=18)
        ax1_rain.set_ylabel('Rainfall (mm)', color='#2ca02c', fontsize=18)
        ax2_discharge.set_ylabel('Discharge (m³/s)', color='#1f77b4', fontsize=18)
        ax3_sediment.set_ylabel('Sediment Yield (t/ha/month)', color='#d62728', fontsize=18)
        
        ax1_rain.yaxis.set_label_position('right')
        ax1_rain.yaxis.tick_right()
        ax2_discharge.yaxis.set_label_position('left')
        ax2_discharge.yaxis.tick_left()
        ax3_sediment.yaxis.set_label_position('right')
        ax3_sediment.yaxis.tick_right()
        
        ax1_rain.set_ylim(rainfall_max, 0)
        ax2_discharge.set_ylim(0, discharge_max)
        ax3_sediment.set_ylim(0, yield_max)
        
        ax1_rain.set_yticks(np.arange(0, rainfall_max + 500, 500))
        ax1_rain.tick_params(axis='y', colors='#2ca02c', labelsize=16, pad=0, labelleft=False, labelright=True)
        ax2_discharge.tick_params(axis='y', colors='#1f77b4', labelsize=16, pad=0)
        ax3_sediment.tick_params(axis='y', colors='#d62728', labelsize=16, pad=0)
        
        ax1_rain.set_xlim(pd.to_datetime('1990-01-01'), pd.to_datetime('2021-01-01'))
        ax1_rain.xaxis.set_major_locator(YearLocator(base=5))
        ax1_rain.xaxis.set_major_formatter(DateFormatter('%Y'))
        ax1_rain.tick_params(axis='x', rotation=45, labelsize=14)
        ax1_rain.grid(True, axis='y', linestyle='--', alpha=0.3)
    
    fig.legend([bars, line_discharge, line_sediment], 
               ['Rainfall (mm)', 'Discharge (m³/s)', 'Sediment Yield (t/ha/month)'],
               loc='lower center', bbox_to_anchor=(0.5, -0.02), ncol=3, fontsize=16)
    plt.tight_layout(pad=3.0)
    
    output_png = output_dir / f'Figure8_Monthly_Sediment_Yield_{uuid4().hex[:8]}_color.png'
    output_svg = output_dir / f'Figure8_Monthly_Sediment_Yield_{uuid4().hex[:8]}_color.svg'
    plt.savefig(output_png, dpi=600, format='png', bbox_inches='tight')
    plt.savefig(output_svg, format='svg', bbox_inches='tight')
    print(f"Figure 8 color saved to {output_png} (PNG) and {output_svg} (SVG)")
    
    # Grayscale version for print
    plt.clf()
    fig, axes = plt.subplots(1, 2, figsize=(16, 5), sharey=False)
    fig.patch.set_facecolor('white')
    
    for idx, watershed_name in enumerate(watersheds):
        ax1_rain = axes[idx]
        ax1_rain.set_facecolor('white')
        monthly_data = monthly_data_dict[watershed_name]
        
        discharge_max = WATERSHED_CONFIG[watershed_name]['discharge_max']
        yield_max = WATERSHED_CONFIG[watershed_name]['yield_max']
        rainfall_max = WATERSHED_CONFIG[watershed_name]['rainfall_max']
        
        monthly_data['Rainfall'] = monthly_data['Rainfall'].clip(lower=0)
        
        # Plot rainfall bars with hatching
        bars = ax1_rain.bar(monthly_data.index, monthly_data['Rainfall'], color='#666666', alpha=0.4, width=50, 
                            hatch='///', label='Rainfall (mm)', zorder=1)
        
        ax2_discharge = ax1_rain.twinx()
        ax3_sediment = ax1_rain.twinx()
        ax3_sediment.spines['right'].set_position(('outward', 80))
        
        line_discharge = ax2_discharge.plot(monthly_data.index, monthly_data['Discharge'], color='#000000', 
                                           linestyle='--', linewidth=2.5, label='Discharge (m³/s)', zorder=2)[0]  # Black, dashed
        line_sediment = ax3_sediment.plot(monthly_data.index, monthly_data['Monthly_Sediment_Yield_tons_ha'],
                                          color='#999999', linestyle='-', linewidth=2.5, 
                                          label='Sediment Yield (t/ha/month)', zorder=2)[0]  # Gray, solid
        
        ax1_rain.set_title(f"({'a' if watershed_name == 'Gilgel Abay' else 'b'}) {watershed_name}", fontsize=20)
        ax1_rain.set_xlabel('Year', fontsize=18)
        ax1_rain.set_ylabel('Rainfall (mm)', color='#666666', fontsize=18)
        ax2_discharge.set_ylabel('Discharge (m³/s)', color='#000000', fontsize=18)
        ax3_sediment.set_ylabel('Sediment Yield (t/ha/month)', color='#999999', fontsize=18)
        
        ax1_rain.yaxis.set_label_position('right')
        ax1_rain.yaxis.tick_right()
        ax2_discharge.yaxis.set_label_position('left')
        ax2_discharge.yaxis.tick_left()
        ax3_sediment.yaxis.set_label_position('right')
        ax3_sediment.yaxis.tick_right()
        
        ax1_rain.set_ylim(rainfall_max, 0)
        ax2_discharge.set_ylim(0, discharge_max)
        ax3_sediment.set_ylim(0, yield_max)
        
        ax1_rain.set_yticks(np.arange(0, rainfall_max + 500, 500))
        ax1_rain.tick_params(axis='y', colors='#666666', labelsize=16, pad=0, labelleft=False, labelright=True)
        ax2_discharge.tick_params(axis='y', colors='#000000', labelsize=16, pad=0)
        ax3_sediment.tick_params(axis='y', colors='#999999', labelsize=16, pad=0)
        
        ax1_rain.set_xlim(pd.to_datetime('1990-01-01'), pd.to_datetime('2021-01-01'))
        ax1_rain.xaxis.set_major_locator(YearLocator(base=5))
        ax1_rain.xaxis.set_major_formatter(DateFormatter('%Y'))
        ax1_rain.tick_params(axis='x', rotation=45, labelsize=14)
        ax1_rain.grid(True, axis='y', linestyle='--', alpha=0.3)
    
    fig.legend([bars, line_discharge, line_sediment], 
               ['Rainfall (mm)', 'Discharge (m³/s)', 'Sediment Yield (t/ha/month)'],
               loc='lower center', bbox_to_anchor=(0.5, -0.02), ncol=3, fontsize=16)
    plt.tight_layout(pad=3.0)
    
    output_png = output_dir / f'Figure8_Monthly_Sediment_Yield_{uuid4().hex[:8]}_grayscale.png'
    output_eps = output_dir / f'Figure8_Monthly_Sediment_Yield_{uuid4().hex[:8]}_grayscale.eps'
    plt.savefig(output_png, dpi=600, format='png', bbox_inches='tight')
    plt.savefig(output_eps, format='eps', bbox_inches='tight')
    print(f"Figure 8 grayscale saved to {output_png} (PNG) and {output_eps} (EPS)")
    plt.show()
    plt.close()

def main():
    monthly_data_dict = {}
    for watershed_name, params in WATERSHED_CONFIG.items():
        print(f"\n=== Processing {watershed_name} ===")
        
        qrf_params = load_best_params(watershed_name, params['output_dir'])
        if qrf_params is None:
            print(f"Skipping {watershed_name} due to missing QRF parameters")
            continue
        
        df_cont = predict_ssc(
            params['intermittent'],
            params['continuous'],
            watershed_name,
            qrf_params,
            is_excel_inter=(watershed_name == 'Gilgel Abay')
        )
        if df_cont is None:
            print(f"Skipping {watershed_name} due to data loading issues")
            continue
        
        daily_data = calculate_sediment_yield(df_cont, watershed_name, params['area_km2'], params['output_dir'])
        if daily_data is None:
            print(f"Skipping {watershed_name} due to sediment yield calculation issues")
            continue
        
        monthly_data = process_monthly_data(daily_data, watershed_name)
        if monthly_data is None:
            print(f"Skipping {watershed_name} due to monthly data processing issues")
            continue
        
        output_csv = params['output_dir'] / f"{watershed_name.replace(' ', '_')}_Monthly_Sediment_Yield_{uuid4().hex[:8]}.xlsx"
        monthly_data.to_excel(output_csv)
        print(f"Monthly data saved to {output_csv}")
        
        monthly_data_dict[watershed_name] = monthly_data
    
    if monthly_data_dict:
        create_figure8(monthly_data_dict, WATERSHED_CONFIG['Gilgel Abay']['output_dir'])

if __name__ == "__main__":
    main()
