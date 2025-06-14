import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from quantile_forest import RandomForestQuantileRegressor
from sklearn.preprocessing import RobustScaler
import os
from scipy.stats import pearsonr

# Set plot style: Times New Roman, font size 16
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 16

# Define watershed parameters and file paths
data_paths = {
    'Gilgel Abay': {
        'intermittent': r"D:\Gilgel Abay\Sedigrapgh\Intermittent_data.xlsx",
        'continuous': r"D:\Gilgel Abay\Sedigrapgh\continuous_data.csv",
        'output_csv': r"D:\Gilgel Abay\Sedigrapgh\Gilgel_Abay_Yearly_Data_ha_QRF.csv",
        'output_plot': r"D:\Gilgel Abay\Sedigrapgh\Gilgel_Abay_Annual_Discharge_Rainfall_Sediment_Yield_ha_QRF.png",
        'output_plot_fig9': r"D:\Gilgel Abay\Sedigrapgh\Gilgel_Abay_Figure9_Sediment_Yield_vs_Builtup.png",
        'area_km2': 1664,
        'discharge_max': 180,
        'yield_max': 40  # For tonnes/ha/year
    },
    'Gumara': {
        'intermittent': r"D:\Gumara\Sedigrapgh\Intermittent_data_gum.csv",
        'continuous': r"D:\Gumara\Sedigrapgh\continious_data_gum.csv",
        'output_csv': r"D:\Gumara\Sedigrapgh\Gumara_Yearly_Data_ha_QRF.csv",
        'output_plot': r"D:\Gumara\Sedigrapgh\Gumara_Annual_Discharge_Rainfall_Sediment_Yield_ha_QRF.png",
        'output_plot_fig9': r"D:\Gumara\Sedigrapgh\Gumara_Figure9_Sediment_Yield_vs_Builtup.png",
        'area_km2': 1394,
        'discharge_max': 120,
        'yield_max': 60  # For tonnes/ha/year
    }
}

# QRF hyperparameters
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

# Function to predict SSC using QRF
def predict_ssc(intermittent_path, continuous_path, watershed_name, qrf_params):
    print(f"Predicting SSC for {watershed_name}...")
    
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
    
    required_cols_inter = ['Date', 'Rainfall', 'Discharge', 'Temperature', 'ETo', 'SSC']
    required_cols_cont = ['Date', 'Rainfall', 'Discharge', 'Temperature', 'ETo']
    missing_cols_inter = [col for col in required_cols_inter if col not in df_inter.columns]
    missing_cols_cont = [col for col in required_cols_cont if col not in df_cont.columns]
    if missing_cols_inter:
        raise ValueError(f"{watershed_name} intermittent data missing columns: {missing_cols_inter}")
    if missing_cols_cont:
        raise ValueError(f"{watershed_name} continuous data missing columns: {missing_cols_cont}")
    
    print(f"{watershed_name} Intermittent Data Shape: {df_inter.shape}")
    print(f"{watershed_name} Intermittent Data Head:\n{df_inter.head()}")
    
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
    
    numeric_cols = ['Rainfall', 'Discharge', 'Temperature', 'ETo', 'SSC']
    for col in numeric_cols:
        df_inter[col] = pd.to_numeric(df_inter[col], errors='coerce')
    for col in numeric_cols[:-1]:
        df_cont[col] = pd.to_numeric(df_cont[col], errors='coerce')
    
    df_inter = df_inter.dropna(subset=numeric_cols)
    df_cont = df_cont.dropna(subset=numeric_cols[:-1])
    
    print(f"{watershed_name} Intermittent Data Shape after Cleaning: {df_inter.shape}")
    if df_inter.empty:
        raise ValueError(f"{watershed_name} intermittent data is empty after cleaning.")
    
    df_inter['Log_Discharge'] = np.log1p(df_inter['Discharge'].clip(lower=0))
    df_inter['Discharge_Rainfall'] = df_inter['Discharge'] * df_inter['Rainfall']
    df_inter['Lag_Discharge'] = df_inter['Discharge'].shift(1).bfill()
    
    df_cont['Log_Discharge'] = np.log1p(df_cont['Discharge'].clip(lower=0))
    df_cont['Discharge_Rainfall'] = df_cont['Discharge'] * df_cont['Rainfall']
    df_cont['Lag_Discharge'] = df_cont['Discharge'].shift(1).bfill()
    
    predictors = ['Log_Discharge', 'Rainfall', 'Temperature', 'ETo', 'Discharge_Rainfall', 'Lag_Discharge']
    X_inter = df_inter[predictors]
    y_inter = df_inter['SSC']
    X_cont = df_cont[predictors]
    
    print(f"{watershed_name} X_inter Shape: {X_inter.shape}")
    print(f"{watershed_name} y_inter Shape: {y_inter.shape}")
    print(f"{watershed_name} X_cont Shape: {X_cont.shape}")
    
    if X_inter.shape[0] == 0:
        raise ValueError(f"{watershed_name} feature matrix X_inter is empty.")
    
    scaler = RobustScaler()
    X_inter_scaled = scaler.fit_transform(X_inter)
    X_cont_scaled = scaler.transform(X_cont)
    
    qrf = RandomForestQuantileRegressor(**qrf_params, random_state=42)
    qrf.fit(X_inter_scaled, y_inter)
    ssc_pred = qrf.predict(X_cont_scaled, quantiles=0.5)
    
    print(f"{watershed_name} SSC Prediction Summary:")
    print(pd.Series(ssc_pred).describe())
    
    return df_cont['Date'], ssc_pred

# Function to process annual data
def process_annual_data(dates, discharge, rainfall, ssc_pred, area_km2, watershed_name):
    print(f"Processing annual data for {watershed_name}...")
    
    df = pd.DataFrame({
        'Date': dates,
        'Discharge': pd.to_numeric(discharge, errors='coerce'),
        'Rainfall': pd.to_numeric(rainfall, errors='coerce'),
        'SSC_predicted': pd.to_numeric(ssc_pred, errors='coerce')
    })
    
    initial_rows = len(df)
    df = df.dropna(subset=['Date', 'Discharge', 'Rainfall', 'SSC_predicted'])
    if len(df) < initial_rows:
        print(f"{watershed_name} Dropped {initial_rows - len(df)} rows with invalid or missing data.")
    
    if df.empty:
        raise ValueError(f"{watershed_name} daily data is empty after cleaning.")
    
    df['Year'] = df['Date'].dt.year
    
    df['Sediment_Load_tonnes_day'] = df['Discharge'] * df['SSC_predicted'] * 0.0864
    
    print(f"{watershed_name} Daily Data Summary:")
    print(df[['Discharge', 'Rainfall', 'SSC_predicted', 'Sediment_Load_tonnes_day']].describe())
    
    yearly_data = df.groupby('Year').agg({
        'Discharge': 'mean',
        'Sediment_Load_tonnes_day': 'sum',
        'Rainfall': 'mean'
    })
    
    yearly_data['Days_in_Year'] = df.groupby('Year')['Date'].nunique()
    
    yearly_data['Annual_Rainfall_mm'] = yearly_data['Rainfall'] * yearly_data['Days_in_Year']
    
    yearly_data['Annual_Sediment_Yield_tons_ha'] = yearly_data['Sediment_Load_tonnes_day'] / (area_km2 * 100)
    
    yearly_data = yearly_data.rename(columns={'Sediment_Load_tonnes_day': 'Annual_Sediment_Load_tons_year'})
    
    return yearly_data

# Function to create Figure 7 plot
def create_plot(yearly_data, watershed_name, output_plot, discharge_max, yield_max):
    print(f"Generating plot for {watershed_name}...")
    
    max_rainfall = yearly_data['Annual_Rainfall_mm'].max() * 1.1
    reversed_rainfall = max_rainfall - yearly_data['Annual_Rainfall_mm']
    
    fig, ax1 = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor('white')
    ax1.set_facecolor('white')
    
    ax1.bar(yearly_data.index, reversed_rainfall, color='green', alpha=0.7, width=0.4, label='Rainfall (mm)')
    
    ax2 = ax1.twinx()
    ax3 = ax1.twinx()
    ax3.spines['right'].set_position(('outward', 60))
    
    ax2.plot(yearly_data.index, yearly_data['Discharge'], color='blue', marker='o', linestyle='-', 
             label='Discharge (m³/s)')
    
    ax3.plot(yearly_data.index, yearly_data['Annual_Sediment_Yield_tons_ha'], color='red', 
             marker='s', linestyle='-', label='Sediment Yield (tonnes/ha/year)')
    
    plt.title(watershed_name, fontsize=20)
    ax1.set_xlabel('Year', fontsize=16)
    ax1.set_ylabel('Rainfall (mm)', color='green', fontsize=16)
    ax2.set_ylabel('Discharge (m³/s)', color='blue', fontsize=16)
    ax3.set_ylabel('Sediment Yield (tonnes/ha/year)', color='red', fontsize=16)
    
    ax1.yaxis.set_label_position('right')
    ax1.yaxis.tick_right()
    ax2.yaxis.set_label_position('left')
    ax2.yaxis.tick_left()
    ax3.yaxis.set_label_position('right')
    ax3.yaxis.tick_right()
    
    ax1.set_ylim(max_rainfall, 0)
    ax2.set_ylim(0, discharge_max)
    ax3.set_ylim(0, yield_max)
    
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
    
    ax1.grid(False)
    
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    lines3, labels3 = ax3.get_legend_handles_labels()
    plt.legend(lines1 + lines2 + lines3, labels1 + labels2 + labels3, loc='lower center', 
               bbox_to_anchor=(0.5, 0), ncol=3, fontsize=16)
    
    plt.tight_layout()
    plt.savefig(output_plot, dpi=300, bbox_inches='tight')
    print(f"Figure saved to {output_plot}")
    plt.show()
    plt.close()

# Function to create Figure 9: Scatter plot of sediment yield vs. built-up area
def create_figure9(yearly_data, watershed_name, output_plot):
    print(f"Generating Figure 9 for {watershed_name}...")
    
    # Filter data for 2000–2020
    yearly_data = yearly_data[(yearly_data.index >= 2000) & (yearly_data.index <= 2020)]
    if yearly_data.empty:
        print(f"No data available for {watershed_name} between 2000 and 2020.")
        return
    
    # Assume linear interpolation for built-up area (replace with actual data if available)
    years = yearly_data.index
    num_years = len(years)
    if watershed_name == 'Gilgel Abay':
        builtup_area = np.linspace(7.3, 76, num_years)  # 7.3 km² (2000) to 76 km² (2020)
    elif watershed_name == 'Gumara':
        builtup_area = np.linspace(4.6, 32, num_years)  # 4.6 km² (2000) to 32 km² (2020)
    
    sediment_yield = yearly_data['Annual_Sediment_Yield_tons_ha']
    
    # Compute actual correlation
    r, p_value = pearsonr(builtup_area, sediment_yield)
    print(f"{watershed_name} Actual Correlation (r): {r:.2f}, p-value: {p_value:.4f}")
    
    # Create scatter plot
    fig, ax = plt.subplots(figsize=(6, 5))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    
    ax.scatter(builtup_area, sediment_yield, color='blue' if watershed_name == 'Gilgel Abay' else 'red', alpha=0.5)
    
    # Add trend line
    z = np.polyfit(builtup_area, sediment_yield, 1)  # Linear fit
    p = np.poly1d(z)
    ax.plot(builtup_area, p(builtup_area), "--", color='gray', alpha=0.5)
    
    ax.set_title(f'({"a" if watershed_name == "Gilgel Abay" else "b"}) {watershed_name}', fontsize=16)
    ax.set_xlabel('Built-up Area (km²)', fontsize=12)
    ax.set_ylabel('Sediment Yield (tonnes/ha/yr)', fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Add correlation in legend
    ax.legend([f'r = {r:.2f} (p = {p_value:.4f})'], loc='best', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(output_plot, dpi=300, bbox_inches='tight')
    print(f"Figure 9 saved to {output_plot}")
    plt.show()
    plt.close()

# Main processing loop
for watershed_name, params in data_paths.items():
    print(f"\n=== Processing {watershed_name} ===")
    
    try:
        df_cont = pd.read_csv(params['continuous'])
        
        dates, ssc_pred = predict_ssc(
            params['intermittent'],
            params['continuous'],
            watershed_name,
            qrf_params[watershed_name]
        )
        
        df_cont['Date'] = pd.to_datetime(df_cont['Date'], errors='coerce')
        df_temp = pd.DataFrame({'Date': dates, 'SSC_predicted': ssc_pred})
        df_merged = df_cont[['Date', 'Discharge', 'Rainfall']].merge(df_temp, on='Date', how='inner')
        
        if df_merged.empty:
            raise ValueError(f"{watershed_name} merged data is empty.")
        
        yearly_data = process_annual_data(
            df_merged['Date'],
            df_merged['Discharge'],
            df_merged['Rainfall'],
            df_merged['SSC_predicted'],
            params['area_km2'],
            watershed_name
        )
        
        yearly_data = yearly_data[yearly_data.index >= 1990]
        
        print(f"\n{watershed_name} Annual Metrics:")
        print(yearly_data[['Annual_Sediment_Load_tons_year', 'Annual_Sediment_Yield_tons_ha', 
                           'Discharge', 'Annual_Rainfall_mm']].round(2))
        
        yearly_data.to_csv(params['output_csv'])
        print(f"Data saved to {params['output_csv']}")
        
        create_plot(
            yearly_data,
            watershed_name,
            params['output_plot'],
            params['discharge_max'],
            params['yield_max']
        )
        
        create_figure9(yearly_data, watershed_name, params['output_plot_fig9'])
        
    except Exception as e:
        print(f"Error processing {watershed_name}: {str(e)}")
        continue
