# Script to generate Figure 7: Annual sediment yield, rainfall, and discharge for Gilgel Abay and Gumara watersheds
# (Section 3.3). Trains Quantile Random Forest (QRF) on intermittent data (Date, Discharge, SSC, Rainfall, Temperature, ETo),
# predicts daily Suspended Sediment Concentration (SSC, g/L) for 1990–2020 on continuous data, calculates daily sediment yield
# (t/ha/day), and aggregates annually (t/ha/yr). Produces Figure 7 (annual rainfall, discharge, sediment yield with IQR uncertainty)
# and daily sedigraphs for key years. Outputs include Excel for daily SSC/sediment yield, CSV for annual data,
# feature importance, and PNG/SVG plots. Loads QRF parameters from best_params_comparison_70split.csv.
# Author: Kindie B. Worku
# Date: 2025-07-19

# Importing necessary libraries
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

# Suppressing warnings for cleaner output
warnings.filterwarnings('ignore')

# Setting plot style for publication quality
sns.set_style('white')
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 18

# Defining constants for watershed configurations
WATERSHED_CONFIG = {
    'Gilgel Abay': {
        'intermittent': Path(r"C:\Users\worku\Documents\sediment-yield-analysis\data\Intermittent_data.xlsx"),
        'continuous': Path(r"C:\Users\worku\Documents\sediment-yield-analysis\data\continuous_data.csv"),
        'area_km2': 1664,
        'discharge_max': 180,
        'yield_max': 90,
        'rainfall_max': 3500,
        'output_dir': Path(r"C:\Users\worku\Documents\sediment-yield-analysis\outputs")
    },
    'Gumara': {
        'intermittent': Path(r"C:\Users\worku\Documents\sediment-yield-analysis\data\Intermittent_data_gum.csv"),
        'continuous': Path(r"C:\Users\worku\Documents\sediment-yield-analysis\data\continuous_data_gum.csv"),
        'area_km2': 1394,
        'discharge_max': 180,
        'yield_max': 120,
        'rainfall_max': 3800,
        'output_dir': Path(r"C:\Users\worku\Documents\sediment-yield-analysis\outputs")
    }
}

# Defining load factor for unit conversion
LOAD_FACTOR = 86.4  # Converts m³/s × g/L to t/day (86,400 s/day × 10⁻⁶ t/g)

# Adding seasonal features to the dataframe
def add_seasonal_features(df):
    df = df.copy()
    # Converting Date column to datetime
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    # Calculating Julian day for seasonal analysis
    df['Julian_Day'] = df['Date'].dt.dayofyear
    # Adding sine and cosine transformations for periodicity
    df['Sin_Julian'] = np.sin(2 * np.pi * df['Julian_Day'] / 365.25)
    df['Cos_Julian'] = np.cos(2 * np.pi * df['Julian_Day'] / 365.25)
    return df.drop(columns=['Julian_Day'])

# Loading best QRF parameters from CSV
def load_best_params(watershed_name, output_dir):
    params_path = output_dir / "best_params_comparison_70split.csv"
    # Checking if parameters file exists
    if not params_path.exists():
        print(f"Error: {params_path} not found")
        return None
    params_df = pd.read_csv(params_path)
    # Filtering for QRF parameters specific to the watershed
    qrf_params_row = params_df[(params_df['Model'] == 'QRF') & (params_df['Watershed'] == watershed_name)]
    if qrf_params_row.empty:
        print(f"Error: No QRF parameters for {watershed_name} in {params_path}")
        return None
    # Parsing parameters from string to dictionary
    qrf_params = ast.literal_eval(qrf_params_row['Parameters'].iloc[0])
    print(f"Loaded QRF params for {watershed_name}: {qrf_params}")
    return qrf_params

# Predicting SSC using QRF model
def predict_ssc(intermittent_path, continuous_path, watershed_name, qrf_params, is_excel_inter=False):
    print(f"\nPredicting SSC for {watershed_name}...")
    # Checking if input files exist
    if not intermittent_path.exists():
        print(f"Intermittent file not found: {intermittent_path}")
        return None
    if not continuous_path.exists():
        print(f"Continuous file not found: {continuous_path}")
        return None

    # Loading intermittent and continuous data
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

    # Defining column name mappings for consistency
    column_mapping = {
        'Date': ['Date', 'date', 'Time', 'time', 'Timestamp', 'timestamp'],
        'Rainfall': ['Rainfall', 'rainfall', 'Rain', 'rain'],
        'Discharge': ['Discharge', 'discharge', 'Flow', 'flow'],
        'Temperature': ['Temperature', 'temperature', 'Temp', 'temp'],
        'ETo': ['ETo', 'eto', 'ET0', 'Evapotranspiration', 'evapotranspiration'],
        'SSC': ['SSC', 'ssc', 'SuspendedSediment', 'suspended_sediment']
    }

    # Renaming columns to standard names
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

    # Converting Date columns to datetime and cleaning data
    df_inter['Date'] = pd.to_datetime(df_inter['Date'], errors='coerce')
    df_cont['Date'] = pd.to_datetime(df_cont['Date'], errors='coerce')
    df_inter = df_inter.dropna(subset=['Date', 'SSC'])
    df_cont = df_cont[(df_cont['Date'].dt.year >= 1990) & (df_cont['Date'].dt.year <= 2020)]
    df_cont = df_cont.dropna(subset=['Date'])

    # Ensuring numeric columns and clipping negative values
    numeric_cols = ['Rainfall', 'Discharge', 'Temperature', 'ETo', 'SSC']
    for col in numeric_cols:
        if col in df_inter:
            df_inter[col] = pd.to_numeric(df_inter[col], errors='coerce').clip(lower=0)
    for col in numeric_cols[:-1]:
        df_cont[col] = pd.to_numeric(df_cont[col], errors='coerce').clip(lower=0)

    # Adding seasonal features
    df_inter = add_seasonal_features(df_inter)
    df_cont = add_seasonal_features(df_cont)

    # Sorting data by date
    df_inter = df_inter.sort_values('Date')
    df_cont = df_cont.sort_values('Date')

    # Adding year column and annual rainfall
    df_inter['Year'] = df_inter['Date'].dt.year
    df_cont['Year'] = df_cont['Date'].dt.year
    annual_rainfall_inter = df_inter.groupby('Year')['Rainfall'].sum().reset_index()
    annual_rainfall_inter.columns = ['Year', 'Annual_Rainfall']
    df_inter = df_inter.merge(annual_rainfall_inter, on='Year', how='left')
    annual_rainfall_cont = df_cont.groupby('Year')['Rainfall'].sum().reset_index()
    annual_rainfall_cont.columns = ['Year', 'Annual_Rainfall']
    df_cont = df_cont.merge(annual_rainfall_cont, on='Year', how='left')

    # Adding cumulative rainfall
    df_inter['Cumulative_Rainfall'] = df_inter.groupby('Year')['Rainfall'].cumsum()
    df_cont['Cumulative_Rainfall'] = df_cont.groupby('Year')['Rainfall'].cumsum()

    # Adding moving average and lagged discharge features
    df_inter['MA_Discharge_3'] = df_inter['Discharge'].rolling(window=3, min_periods=1).mean().bfill()
    df_inter['Lag_Discharge'] = df_inter['Discharge'].shift(1).bfill()
    df_inter['Lag_Discharge_3'] = df_inter['Discharge'].shift(3).bfill()
    df_cont['MA_Discharge_3'] = df_cont['Discharge'].rolling(window=3, min_periods=1).mean().bfill()
    df_cont['Lag_Discharge'] = df_cont['Discharge'].shift(1).bfill()
    df_cont['Lag_Discharge_3'] = df_cont['Discharge'].shift(3).bfill()

    # Defining predictors for QRF model
    predictors = ['Discharge', 'MA_Discharge_3', 'Lag_Discharge', 'Lag_Discharge_3', 'Rainfall', 'ETo',
                  'Temperature', 'Annual_Rainfall', 'Cumulative_Rainfall', 'Sin_Julian', 'Cos_Julian']

    # Handling missing or infinite values in predictors
    for df, df_name in [(df_inter, 'intermittent'), (df_cont, 'continuous')]:
        for col in predictors:
            if col in df and (df[col].isna().any() or np.isinf(df[col]).any()):
                df[col] = df[col].fillna(0).replace([np.inf, -np.inf], 0)

    # Final data cleaning
    df_inter = df_inter.dropna(subset=predictors + ['SSC'])
    df_cont = df_cont.dropna(subset=predictors)
    print(f"{watershed_name} Intermittent Data after cleaning: {len(df_inter)} rows")
    print(f"{watershed_name} Continuous Data after cleaning: {len(df_cont)} rows")

    if df_inter.empty:
        print(f"{watershed_name} intermittent data empty after cleaning")
        return None

    # Preparing data for QRF model
    X_inter = df_inter[predictors]
    y_inter = df_inter['SSC']
    X_cont = df_cont[predictors]

    # Training QRF model and predicting SSC
    try:
        qrf = RandomForestQuantileRegressor(**qrf_params)
        qrf.fit(X_inter, y_inter)
        ssc_pred = qrf.predict(X_cont, quantiles=[0.05, 0.25, 0.5, 0.75, 0.95])
        print(f"{watershed_name} SSC Prediction Summary (g/L):")
        for q, preds in zip([0.05, 0.25, 0.5, 0.75, 0.95], ssc_pred.T):
            print(f"Quantile {q}: {pd.Series(preds).describe()}")
        
        # Saving feature importance
        feature_importance = pd.DataFrame({
            'Feature': predictors,
            'Importance': qrf.feature_importances_
        }).sort_values(by='Importance', ascending=False)
        print(f"{watershed_name} Feature Importance:")
        print(feature_importance)
        feature_importance.to_csv(WATERSHED_CONFIG[watershed_name]['output_dir'] /
                                 f"feature_importance_annual_{watershed_name.lower().replace(' ', '_')}_70split.csv", index=False)
    except Exception as e:
        print(f"Error training/predicting QRF for {watershed_name}: {str(e)}")
        return None

    # Adding predictions to continuous dataframe
    df_cont['SSC'] = ssc_pred[:, 2]
    df_cont['SSC_Q05'] = ssc_pred[:, 0]
    df_cont['SSC_Q25'] = ssc_pred[:, 1]
    df_cont['SSC_Q75'] = ssc_pred[:, 3]
    df_cont['SSC_Q95'] = ssc_pred[:, 4]
    return df_cont[['Date', 'Rainfall', 'Discharge', 'Temperature', 'ETo', 'SSC', 'SSC_Q05', 'SSC_Q25', 'SSC_Q75', 'SSC_Q95',
                    'Annual_Rainfall', 'Cumulative_Rainfall', 'Sin_Julian', 'Cos_Julian']]

# Calculating sediment yield
def calculate_sediment_yield(df, watershed_name, area_km2, output_dir):
    print(f"\nCalculating sediment yield for {watershed_name}...")
    # Creating output directory if it doesn't exist
    output_dir.mkdir(exist_ok=True)

    # Calculating sediment yield for different quantiles
    df['Sediment_Yield'] = df['Discharge'] * df['SSC'] * LOAD_FACTOR / (area_km2 * 100)
    df['Sediment_Yield_Q05'] = df['Discharge'] * df['SSC_Q05'] * LOAD_FACTOR / (area_km2 * 100)
    df['Sediment_Yield_Q25'] = df['Discharge'] * df['SSC_Q25'] * LOAD_FACTOR / (area_km2 * 100)
    df['Sediment_Yield_Q75'] = df['Discharge'] * df['SSC_Q75'] * LOAD_FACTOR / (area_km2 * 100)
    df['Sediment_Yield_Q95'] = df['Discharge'] * df['SSC_Q95'] * LOAD_FACTOR / (area_km2 * 100)

    # Clipping negative sediment yield values
    for col in ['Sediment_Yield', 'Sediment_Yield_Q05', 'Sediment_Yield_Q25', 'Sediment_Yield_Q75', 'Sediment_Yield_Q95']:
        if (df[col] < 0).any():
            df[col] = df[col].clip(lower=0)

    # Dropping rows with missing critical values
    df = df.dropna(subset=['Date', 'Rainfall', 'Discharge', 'SSC', 'Sediment_Yield', 'Sediment_Yield_Q05',
                          'Sediment_Yield_Q25', 'Sediment_Yield_Q75', 'Sediment_Yield_Q95', 'Annual_Rainfall'])
    print(f"{watershed_name} Data after dropping NaNs: {len(df)} rows")

    if df.empty:
        print(f"{watershed_name} data empty after cleaning")
        return None

    # Saving daily data to Excel
    output_path = output_dir / f"{watershed_name.replace(' ', '_')}_Daily_SSC_Sediment_Yield_{uuid4().hex[:8]}.xlsx"
    df.to_excel(output_path, index=False)
    print(f"Daily data saved to {output_path}")

    return df

# Processing annual data aggregation
def process_annual_data(df, watershed_name):
    print(f"\nProcessing annual data for {watershed_name}...")
    df['Year'] = df['Date'].dt.year
    # Aggregating data by year
    yearly_data = df.groupby('Year').agg({
        'Discharge': 'mean',
        'Rainfall': 'sum',
        'SSC': 'mean',
        'SSC_Q05': 'mean',
        'SSC_Q25': 'mean',
        'SSC_Q75': 'mean',
        'SSC_Q95': 'mean',
        'Sediment_Yield': 'sum',
        'Sediment_Yield_Q05': 'sum',
        'Sediment_Yield_Q25': 'sum',
        'Sediment_Yield_Q75': 'sum',
        'Sediment_Yield_Q95': 'sum'
    })

    # Adding days in year and renaming columns
    yearly_data['Days_in_Year'] = df.groupby('Year')['Date'].nunique()
    yearly_data['Annual_Rainfall_mm'] = yearly_data['Rainfall']
    yearly_data['Annual_Sediment_Yield_tons_ha'] = yearly_data['Sediment_Yield']
    yearly_data['Annual_Sediment_Yield_Q05'] = yearly_data['Sediment_Yield_Q05']
    yearly_data['Annual_Sediment_Yield_Q25'] = yearly_data['Sediment_Yield_Q25']
    yearly_data['Annual_Sediment_Yield_Q75'] = yearly_data['Sediment_Yield_Q75']
    yearly_data['Annual_Sediment_Yield_Q95'] = yearly_data['Sediment_Yield_Q95']

    # Checking for sparse years
    sparse_years = yearly_data[yearly_data['Days_in_Year'] < 365 * 0.8].index
    if not sparse_years.empty:
        print(f"Warning: Sparse data in {watershed_name} for years {list(sparse_years)} (< 80% of days)")

    print(f"{watershed_name} Years in Annual Data: {len(yearly_data)} ({yearly_data.index.min()}–{yearly_data.index.max()})")
    return yearly_data

# Creating sedigraphs and Figure 7
def create_sedigraphs(daily_data_dict, yearly_data_dict, output_dir):
    print("\nGenerating Figure 7 (Annual Plots)...")
    watersheds = ['Gilgel Abay', 'Gumara']
    # Checking for missing watershed data
    missing_watersheds = [w for w in watersheds if w not in yearly_data_dict]
    if missing_watersheds:
        print(f"Error generating Figure 7: Missing data for watersheds: {missing_watersheds}")
        return

    # Creating color version of Figure 7
    fig, axes = plt.subplots(1, 2, figsize=(18, 6), sharey=False)
    fig.patch.set_facecolor('white')

    for idx, watershed_name in enumerate(watersheds):
        ax1_rain = axes[idx]
        ax1_rain.set_facecolor('white')
        yearly_data = yearly_data_dict[watershed_name]
        
        # Setting axis limits
        discharge_max = WATERSHED_CONFIG[watershed_name]['discharge_max']
        yield_max = WATERSHED_CONFIG[watershed_name]['yield_max']
        rainfall_max = WATERSHED_CONFIG[watershed_name]['rainfall_max']
        
        data_rainfall_max = yearly_data['Annual_Rainfall_mm'].max() * 1.1 if yearly_data['Annual_Rainfall_mm'].max() > 0 else 1000
        max_rainfall = min(data_rainfall_max, rainfall_max)
        max_rainfall = np.ceil(max_rainfall / 500) * 500
        yearly_data['Annual_Rainfall_mm'] = yearly_data['Annual_Rainfall_mm'].clip(lower=0)
        
        reversed_rainfall = max_rainfall - yearly_data['Annual_Rainfall_mm']
        
        # Plotting rainfall bars
        bars = ax1_rain.bar(yearly_data.index, reversed_rainfall, color='#2ca02c', alpha=0.7, width=0.4,
                            hatch='///', label='Rainfall (mm)')
        
        # Creating twin axes for discharge and sediment
        ax2_discharge = ax1_rain.twinx()
        ax3_sediment = ax1_rain.twinx()
        ax3_sediment.spines['right'].set_position(('outward', 60))
        
        # Plotting discharge and sediment lines
        line_discharge = ax2_discharge.plot(yearly_data.index, yearly_data['Discharge'], color='#1f77b4', marker='o',
                                          linestyle='--', linewidth=2, markersize=8,
                                          label='Discharge (m³/s)')[0]
        line_sediment = ax3_sediment.plot(yearly_data.index, yearly_data['Annual_Sediment_Yield_tons_ha'],
                                         color='#d62728', marker='s', linestyle='-', linewidth=2, markersize=8,
                                         label='Sediment Yield (t/ha/yr)')[0]
        ax3_sediment.fill_between(yearly_data.index, yearly_data['Annual_Sediment_Yield_Q25'],
                                 yearly_data['Annual_Sediment_Yield_Q75'], color='#d62728', alpha=0.2,
                                 hatch='\\', label='Uncertainty (IQR)')
        
        # Setting titles and labels
        ax1_rain.set_title(f"({'a' if watershed_name == 'Gilgel Abay' else 'b'}) {watershed_name}", fontsize=20)
        ax1_rain.set_xlabel('Year', fontsize=18)
        ax1_rain.set_ylabel('Rainfall (mm)', color='#2ca02c', fontsize=18)
        ax2_discharge.set_ylabel('Discharge (m³/s)', color='#1f77b4', fontsize=18)
        ax3_sediment.set_ylabel('Sediment Yield (t/ha/yr)', color='#d62728', fontsize=18)
        
        # Configuring axis positions and ticks
        ax1_rain.yaxis.set_label_position('right')
        ax1_rain.yaxis.tick_right()
        ax2_discharge.yaxis.set_label_position('left')
        ax2_discharge.yaxis.tick_left()
        ax3_sediment.yaxis.set_label_position('right')
        ax3_sediment.yaxis.tick_right()
        
        ax1_rain.set_ylim(max_rainfall, 0)
        ax2_discharge.set_ylim(0, discharge_max)
        ax3_sediment.set_ylim(0, yield_max)
        
        ax1_rain.set_yticks(np.arange(0, max_rainfall + 500, 500))
        ax1_rain.tick_params(axis='y', colors='#2ca02c', labelsize=16, pad=0, labelleft=False, labelright=True)
        ax2_discharge.tick_params(axis='y', colors='#1f77b4', labelsize=16, pad=0)
        ax3_sediment.tick_params(axis='y', colors='#d62728', labelsize=16, pad=0)
        ax1_rain.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax1_rain.grid(True, axis='y', linestyle='--', alpha=0.3)

    # Adding legend
    fig.legend([bars, line_discharge, line_sediment], ['Rainfall (mm)', 'Discharge (m³/s)', 'Sediment Yield (t/ha/yr)'],
              loc='lower center', bbox_to_anchor=(0.5, -0.02), ncol=3, fontsize=16)
    plt.tight_layout(pad=2.0)

    # Saving color plots
    output_png = output_dir / f'Figure7_Annual_Sediment_Yield_{uuid4().hex[:8]}_color.png'
    output_svg = output_dir / f'Figure7_Annual_Sediment_Yield_{uuid4().hex[:8]}_color.svg'
    plt.savefig(output_png, dpi=600, format='png', bbox_inches='tight')
    plt.savefig(output_svg, format='svg', bbox_inches='tight')

    # Creating grayscale version for print
    plt.clf()
    fig, axes = plt.subplots(1, 2, figsize=(18, 6), sharey=False)
    fig.patch.set_facecolor('white')

    for idx, watershed_name in enumerate(watersheds):
        ax1_rain = axes[idx]
        ax1_rain.set_facecolor('white')
        yearly_data = yearly_data_dict[watershed_name]
        
        discharge_max = WATERSHED_CONFIG[watershed_name]['discharge_max']
        yield_max = WATERSHED_CONFIG[watershed_name]['yield_max']
        rainfall_max = WATERSHED_CONFIG[watershed_name]['rainfall_max']
        
        data_rainfall_max = yearly_data['Annual_Rainfall_mm'].max() * 1.1 if yearly_data['Annual_Rainfall_mm'].max() > 0 else 1000
        max_rainfall = min(data_rainfall_max, rainfall_max)
        max_rainfall = np.ceil(max_rainfall / 500) * 500
        yearly_data['Annual_Rainfall_mm'] = yearly_data['Annual_Rainfall_mm'].clip(lower=0)
        
        reversed_rainfall = max_rainfall - yearly_data['Annual_Rainfall_mm']
        
        # Plotting grayscale bars and lines
        bars = ax1_rain.bar(yearly_data.index, reversed_rainfall, color='#666666', alpha=0.7, width=0.4,
                            hatch='///', label='Rainfall (mm)')
        
        ax2_discharge = ax1_rain.twinx()
        ax3_sediment = ax1_rain.twinx()
        ax3_sediment.spines['right'].set_position(('outward', 60))
        
        line_discharge = ax2_discharge.plot(yearly_data.index, yearly_data['Discharge'], color='#000000', marker='o',
                                          linestyle='--', linewidth=2, markersize=8,
                                          label='Discharge (m³/s)')[0]
        line_sediment = ax3_sediment.plot(yearly_data.index, yearly_data['Annual_Sediment_Yield_tons_ha'],
                                         color='#999999', marker='s', linestyle='-', linewidth=2, markersize=8,
                                         label='Sediment Yield (t/ha/yr)')[0]
        ax3_sediment.fill_between(yearly_data.index, yearly_data['Annual_Sediment_Yield_Q25'],
                                 yearly_data['Annual_Sediment_Yield_Q75'], color='#999999', alpha=0.2,
                                 hatch='\\', label='Uncertainty (IQR)')
        
        # Setting titles and labels for grayscale
        ax1_rain.set_title(f"({'a' if watershed_name == 'Gilgel Abay' else 'b'}) {watershed_name}", fontsize=20)
        ax1_rain.set_xlabel('Year', fontsize=18)
        ax1_rain.set_ylabel('Rainfall (mm)', color='#666666', fontsize=18)
        ax2_discharge.set_ylabel('Discharge (m³/s)', color='#000000', fontsize=18)
        ax3_sediment.set_ylabel('Sediment Yield (t/ha/yr)', color='#999999', fontsize=18)
        
        ax1_rain.yaxis.set_label_position('right')
        ax1_rain.yaxis.tick_right()
        ax2_discharge.yaxis.set_label_position('left')
        ax2_discharge.yaxis.tick_left()
        ax3_sediment.yaxis.set_label_position('right')
        ax3_sediment.yaxis.tick_right()
        
        ax1_rain.set_ylim(max_rainfall, 0)
        ax2_discharge.set_ylim(0, discharge_max)
        ax3_sediment.set_ylim(0, yield_max)
        
        ax1_rain.set_yticks(np.arange(0, max_rainfall + 500, 500))
        ax1_rain.tick_params(axis='y', colors='#666666', labelsize=16, pad=0, labelleft=False, labelright=True)
        ax2_discharge.tick_params(axis='y', colors='#000000', labelsize=16, pad=0)
        ax3_sediment.tick_params(axis='y', colors='#999999', labelsize=16, pad=0)
        ax1_rain.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax1_rain.grid(True, axis='y', linestyle='--', alpha=0.3)

    # Adding legend for grayscale
    fig.legend([bars, line_discharge, line_sediment], ['Rainfall (mm)', 'Discharge (m³/s)', 'Sediment Yield (t/ha/yr)'],
              loc='lower center', bbox_to_anchor=(0.5, -0.02), ncol=3, fontsize=16)
    plt.tight_layout(pad=2.0)

    # Saving grayscale plots
    output_png = output_dir / f'Figure7_Annual_Sediment_Yield_{uuid4().hex[:8]}_grayscale.png'
    output_eps = output_dir / f'Figure7_Annual_Sediment_Yield_{uuid4().hex[:8]}_grayscale.eps'
    plt.savefig(output_png, dpi=600, format='png', bbox_inches='tight')
    plt.savefig(output_eps, format='eps', bbox_inches='tight')
    plt.show()
    plt.close()

    # Generating daily sedigraphs for key years
    print("\nGenerating Daily Sedigraphs...")
    for watershed_name in watersheds:
        daily_data = daily_data_dict[watershed_name]
        yearly_data = yearly_data_dict[watershed_name]
        
        # Identifying wettest and driest years
        wettest_year = yearly_data['Annual_Rainfall_mm'].idxmax()
        driest_year = yearly_data['Annual_Rainfall_mm'].idxmin()
        key_years = [wettest_year, driest_year]
        print(f"{watershed_name} Key Years: Wettest={wettest_year}, Driest={driest_year}")
        
        for year in key_years:
            df_year = daily_data[daily_data['Date'].dt.year == year]
            if df_year.empty:
                print(f"No data for {watershed_name} in {year}")
                continue
            
            # Creating daily sedigraph
            fig, ax1 = plt.subplots(figsize=(10, 6))
            ax1.set_facecolor('white')
            
            ax1.plot(df_year['Date'], df_year['SSC'], color='#d62728', linestyle='-', linewidth=2,
                     label='SSC (g/L)')
            ax1.fill_between(df_year['Date'], df_year['SSC_Q25'], df_year['SSC_Q75'],
                             color='#d62728', alpha=0.2, hatch='\\', label='SSC IQR')
            ax1.set_xlabel('Date', fontsize=18)
            ax1.set_ylabel('SSC (g/L)', color='#d62728', fontsize=18)
            ax1.tick_params(axis='y', colors='#d62728', labelsize=16)
            
            ax2 = ax1.twinx()
            ax2.plot(df_year['Date'], df_year['Discharge'], color='#1f77b4', linestyle='--', linewidth=2,
                     label='Discharge (m³/s)')
            ax2.set_ylabel('Discharge (m³/s)', color='#1f77b4', fontsize=18)
            ax2.tick_params(axis='y', colors='#1f77b4', labelsize=16)
            
            ax1.set_title(f"{watershed_name} Daily Sedigraph ({year})", fontsize=20)
            ax1.grid(True, linestyle='--', alpha=0.3)
            
            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=14)
            
            plt.tight_layout()
            output_png = output_dir / f"{watershed_name.replace(' ', '_')}_Sedigraph_{year}_{uuid4().hex[:8]}.png"
            output_svg = output_dir / f"{watershed_name.replace(' ', '_')}_Sedigraph_{year}_{uuid4().hex[:8]}.svg"
            plt.savefig(output_png, dpi=600, format='png', bbox_inches='tight')
            plt.savefig(output_svg, format='svg', bbox_inches='tight')
            plt.show()
            plt.close()

# Main processing loop
daily_data_dict = {}
yearly_data_dict = {}
for watershed_name, params in WATERSHED_CONFIG.items():
    print(f"\n=== Processing {watershed_name} ===")
    # Loading QRF parameters
    qrf_params = load_best_params(watershed_name, params['output_dir'])
    if qrf_params is None:
        print(f"Skipping {watershed_name} due to missing QRF parameters")
        continue

    # Predicting SSC
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

    # Calculating sediment yield
    daily_data = calculate_sediment_yield(df_cont, watershed_name, params['area_km2'], params['output_dir'])
    if daily_data is None:
        print(f"Skipping {watershed_name} due to sediment yield calculation issues")
        continue

    # Processing annual data
    yearly_data = process_annual_data(daily_data, watershed_name)
    if yearly_data.empty:
        print(f"Skipping {watershed_name} due to empty annual data")
        continue

    # Saving annual data to CSV
    output_csv = params['output_dir'] / f"{watershed_name.replace(' ', '_')}_Yearly_Sediment_Yield_{uuid4().hex[:8]}.csv"
    yearly_data.to_csv(output_csv)
    print(f"Annual data saved to {output_csv}")

    daily_data_dict[watershed_name] = daily_data
    yearly_data_dict[watershed_name] = yearly_data

# Generating final plots
create_sedigraphs(daily_data_dict, yearly_data_dict, WATERSHED_CONFIG['Gumara']['output_dir'])
