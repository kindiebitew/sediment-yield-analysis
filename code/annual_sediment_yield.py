# [Previous script content up to create_sedigraphs remains unchanged]

def create_sedigraphs(daily_data_dict, yearly_data_dict, output_dir):
    print("\nGenerating Figure 7 (Annual Plots)...")
    
    watersheds = ['Gilgel Abay', 'Gumara']
    missing_watersheds = [w for w in watersheds if w not in yearly_data_dict]
    if missing_watersheds:
        print(f"Error generating Figure 7: Missing data for watersheds: {missing_watersheds}")
        return
    
    # Color version for online
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
        
        bars = ax1_rain.bar(yearly_data.index, reversed_rainfall, color='#2ca02c', alpha=0.7, width=0.4, 
                            hatch='///', label='Rainfall (mm)')  # Added hatching
        
        ax2_discharge = ax1_rain.twinx()
        ax3_sediment = ax1_rain.twinx()
        ax3_sediment.spines['right'].set_position(('outward', 60))
        
        line_discharge = ax2_discharge.plot(yearly_data.index, yearly_data['Discharge'], color='#1f77b4', marker='o',
                                           linestyle='--', linewidth=2, markersize=8, 
                                           label='Discharge (m³/s)')[0]  # Dashed line
        line_sediment = ax3_sediment.plot(yearly_data.index, yearly_data['Annual_Sediment_Yield_tons_ha'],
                                          color='#d62728', marker='s', linestyle='-', linewidth=2, markersize=8,
                                          label='Sediment Yield (t/ha/yr)')[0]
        ax3_sediment.fill_between(yearly_data.index, yearly_data['Annual_Sediment_Yield_Q25'], 
                                  yearly_data['Annual_Sediment_Yield_Q75'], color='#d62728', alpha=0.2, 
                                  hatch='\\', label='Uncertainty (IQR)')  # Added hatching
        
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
        
        ax1_rain.set_yticks(np.arange(0, max_rainfall + 500, 500))
        ax1_rain.tick_params(axis='y', colors='#2ca02c', labelsize=16, pad=0, labelleft=False, labelright=True)
        ax2_discharge.tick_params(axis='y', colors='#1f77b4', labelsize=16, pad=0)
        ax3_sediment.tick_params(axis='y', colors='#d62728', labelsize=16, pad=0)
        ax1_rain.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax1_rain.grid(True, axis='y', linestyle='--', alpha=0.3)
    
    fig.legend([bars, line_discharge, line_sediment], ['Rainfall (mm)', 'Discharge (m³/s)', 'Sediment Yield (t/ha/yr)'],
              loc='lower center', bbox_to_anchor=(0.5, -0.02), ncol=3, fontsize=16)
    plt.tight_layout(pad=2.0)
    
    output_png = output_dir / f'Figure7_Annual_Sediment_Yield_{uuid4().hex[:8]}_color.png'
    output_svg = output_dir / f'Figure7_Annual_Sediment_Yield_{uuid4().hex[:8]}_color.svg'
    plt.savefig(output_png, dpi=600, format='png', bbox_inches='tight')
    plt.savefig(output_svg, format='svg', bbox_inches='tight')
    
    # Grayscale version for print
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
        
        bars = ax1_rain.bar(yearly_data.index, reversed_rainfall, color='#666666', alpha=0.7, width=0.4, 
                            hatch='///', label='Rainfall (mm)')  # Gray with hatching
        
        ax2_discharge = ax1_rain.twinx()
        ax3_sediment = ax1_rain.twinx()
        ax3_sediment.spines['right'].set_position(('outward', 60))
        
        line_discharge = ax2_discharge.plot(yearly_data.index, yearly_data['Discharge'], color='#000000', marker='o',
                                           linestyle='--', linewidth=2, markersize=8, 
                                           label='Discharge (m³/s)')[0]  # Black, dashed
        line_sediment = ax3_sediment.plot(yearly_data.index, yearly_data['Annual_Sediment_Yield_tons_ha'],
                                          color='#999999', marker='s', linestyle='-', linewidth=2, markersize=8,
                                          label='Sediment Yield (t/ha/yr)')[0]  # Gray, solid
        ax3_sediment.fill_between(yearly_data.index, yearly_data['Annual_Sediment_Yield_Q25'], 
                                  yearly_data['Annual_Sediment_Yield_Q75'], color='#999999', alpha=0.2, 
                                  hatch='\\', label='Uncertainty (IQR)')  # Gray with hatching
        
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
    
    fig.legend([bars, line_discharge, line_sediment], ['Rainfall (mm)', 'Discharge (m³/s)', 'Sediment Yield (t/ha/yr)'],
              loc='lower center', bbox_to_anchor=(0.5, -0.02), ncol=3, fontsize=16)
    plt.tight_layout(pad=2.0)
    
    output_png = output_dir / f'Figure7_Annual_Sediment_Yield_{uuid4().hex[:8]}_grayscale.png'
    output_eps = output_dir / f'Figure7_Annual_Sediment_Yield_{uuid4().hex[:8]}_grayscale.eps'
    plt.savefig(output_png, dpi=600, format='png', bbox_inches='tight')
    plt.savefig(output_eps, format='eps', bbox_inches='tight')
    plt.show()
    plt.close()
    
    # Daily Sedigraphs (unchanged, as they are supplementary)
    print("\nGenerating Daily Sedigraphs...")
    for watershed_name in watersheds:
        daily_data = daily_data_dict[watershed_name]
        yearly_data = yearly_data_dict[watershed_name]
        
        wettest_year = yearly_data['Annual_Rainfall_mm'].idxmax()
        driest_year = yearly_data['Annual_Rainfall_mm'].idxmin()
        key_years = [wettest_year, driest_year]
        print(f"{watershed_name} Key Years: Wettest={wettest_year}, Driest={driest_year}")
        
        for year in key_years:
            df_year = daily_data[daily_data['Date'].dt.year == year]
            if df_year.empty:
                print(f"No data for {watershed_name} in {year}")
                continue
            
            fig, ax1 = plt.subplots(figsize=(10, 6))
            ax1.set_facecolor('white')
            
            ax1.plot(df_year['Date'], df_year['SSC'], color='#d62728', linestyle='-', linewidth=2, 
                     label='SSC (g/L)')
            ax1.fill_between(df_year['Date'], df_year['SSC_Q25'], df_year['SSC_Q75'], 
                             color='#d62728', alpha=0.2, hatch='\\', label='SSC IQR')  # Added hatching
            ax1.set_xlabel('Date', fontsize=18)
            ax1.set_ylabel('SSC (g/L)', color='#d62728', fontsize=18)
            ax1.tick_params(axis='y', colors='#d62728', labelsize=16)
            
            ax2 = ax1.twinx()
            ax2.plot(df_year['Date'], df_year['Discharge'], color='#1f77b4', linestyle='--', linewidth=2, 
                     label='Discharge (m³/s)')  # Dashed line
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

# Main processing loop (unchanged)
daily_data_dict = {}
yearly_data_dict = {}
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
    
    yearly_data = process_annual_data(daily_data, watershed_name)
    if yearly_data.empty:
        print(f"Skipping {watershed_name} due to empty annual data")
        continue
    
    output_csv = params['output_dir'] / f"{watershed_name.replace(' ', '_')}_Yearly_Sediment_Yield_{uuid4().hex[:8]}.csv"
    yearly_data.to_csv(output_csv)
    print(f"Annual data saved to {output_csv}")
    
    daily_data_dict[watershed_name] = daily_data
    yearly_data_dict[watershed_name] = yearly_data

create_sedigraphs(daily_data_dict, yearly_data_dict, WATERSHED_CONFIG['Gumara']['output_dir'])
