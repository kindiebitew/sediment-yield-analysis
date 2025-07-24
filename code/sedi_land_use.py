# Script to generate Figure 10: relationship between annual sediment yield and land use components, for Gilgel Abay and Gumara.
# interpolates built up area and wetland for 2000–2020 and correlates with annual sediment yields, and creates color and grayscale plots.
# Author: Kindie B. Worku
# Date: 2025-07-23

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
from pathlib import Path
from uuid import uuid4

# Suppress warnings for cleaner output
import warnings
warnings.filterwarnings('ignore')

# Set plot style for publication quality
sns.set_style('white')
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 16

# Define constants
OUTPUT_DIR = Path(r"C:\Users\worku\Documents\sediment-yield-analysis\outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

# Land use data 
LAND_USE_DATA = {
    'Gilgel Abay': {
        'Built-up': {2000: 7.3, 2020: 76}
    },
    'Gumara': {
        'Built-up': {2000: 4.6, 2020: 32},
        'Wetland': {2000: 4.4, 2020: 1.6}
    }
}

# Annual sediment yield data 
SEDIMENT_YIELD_DATA = {
    'Year': list(range(2000, 2021)),
    'Gilgel Abay_Annual_Sediment_Yield_tons_ha': [
        31.552596427439862, 24.62289923076923, 18.375518296624804, 30.670470237461412, 21.930025908320253,
        18.498872080977605, 34.899693093578755, 36.65014088251937, 37.684674304421094, 24.54114316909143,
        29.663997045345994, 26.549246834594435, 41.74122530808266, 48.4967403193755, 33.73154759074239,
        22.27328498, 29.314642933298174, 34.92148572965988, 29.48432975803906, 37.69414720514334,
        34.74274151195411
    ],
    'Gumara_Annual_Sediment_Yield_tons_ha': [
        19.95023674394852, 19.784261831806653, 17.2675866324229, 25.157922232200516, 11.501502127588497,
        18.09672252661185, 27.08841530284911, 28.134947524766673, 31.73430666394086, 24.460093944295792,
        29.715316588385353, 26.88286398911461, 27.862815932846157, 30.95390050488425, 44.98343968110231,
        30.46335641516255, 34.96415861276442, 32.277686372356, 39.633567066575466, 48.47105784597058,
        60.16465814597529
    ]
}

def interpolate_land_use(years, land_use_dict):
    """Interpolate land use areas linearly between 2000 and 2020."""
    print("Interpolating land use data...")
    interpolated = {}
    for watershed, land_uses in land_use_dict.items():
        interpolated[watershed] = {}
        for land_type, data in land_uses.items():
            area_2000, area_2020 = data[2000], data[2020]
            areas = [area_2000 + (area_2020 - area_2000) * (year - 2000) / 20 for year in years]
            interpolated[watershed][land_type] = areas
            print(f"{watershed} {land_type}: {areas[:5]}... (first 5 years)")
    return interpolated

def create_figure_10():
    """Generate Figure 10: Scatter plots of sediment yield vs. land use with regression and correlations, in color and grayscale."""
    print("Generating Figure 10...")

    # Create DataFrame for sediment yield
    df_sediment = pd.DataFrame(SEDIMENT_YIELD_DATA)
    if len(df_sediment) != 21:
        print(f"Error: Expected 21 years of sediment yield data, got {len(df_sediment)}")
        return

    # Interpolate land use areas
    years = list(range(2000, 2021))
    interpolated_land_use = interpolate_land_use(years, LAND_USE_DATA)

    # Create DataFrames for plotting
    df_gilgel_builtup = pd.DataFrame({
        'Year': years,
        'Sediment_Yield': df_sediment['Gilgel Abay_Annual_Sediment_Yield_tons_ha'],
        'Built-up': interpolated_land_use['Gilgel Abay']['Built-up']
    })
    df_gumara_builtup = pd.DataFrame({
        'Year': years,
        'Sediment_Yield': df_sediment['Gumara_Annual_Sediment_Yield_tons_ha'],
        'Built-up': interpolated_land_use['Gumara']['Built-up']
    })
    df_gumara_wetland = pd.DataFrame({
        'Year': years,
        'Sediment_Yield': df_sediment['Gumara_Annual_Sediment_Yield_tons_ha'],
        'Wetland': interpolated_land_use['Gumara']['Wetland']
    })

    # Debug: Print data summaries
    print("Data summaries:")
    print("Gilgel Abay Built-up:")
    print(df_gilgel_builtup.describe())
    print("Gumara Built-up:")
    print(df_gumara_builtup.describe())
    print("Gumara Wetland:")
    print(df_gumara_wetland.describe())

    # Calculate Pearson correlations
    correlations = {}
    correlations['Gilgel Abay Built-up'] = pearsonr(df_gilgel_builtup['Built-up'], df_gilgel_builtup['Sediment_Yield'])
    correlations['Gumara Built-up'] = pearsonr(df_gumara_builtup['Built-up'], df_gumara_builtup['Sediment_Yield'])
    correlations['Gumara Wetland'] = pearsonr(df_gumara_wetland['Wetland'], df_gumara_wetland['Sediment_Yield'])

    print(f"Gilgel Abay Built-up: r = {correlations['Gilgel Abay Built-up'][0]:.2f}, p = {correlations['Gilgel Abay Built-up'][1]:.4f}")
    print(f"Gumara Built-up: r = {correlations['Gumara Built-up'][0]:.2f}, p = {correlations['Gumara Built-up'][1]:.4f}")
    print(f"Gumara Wetland: r = {correlations['Gumara Wetland'][0]:.2f}, p = {correlations['Gumara Wetland'][1]:.4f}")

    # Define colors and styles
    colors = {
        'Gilgel Abay Built-up': '#1f77b4',  # Dark blue
        'Gumara Built-up': '#ff7f0e',       # Dark orange
        'Gumara Wetland': '#2ca02c'         # Green
    }
    grayscale_colors = {
        'Gilgel Abay Built-up': '#000000',  # Black
        'Gumara Built-up': '#666666',       # Medium gray
        'Gumara Wetland': '#999999'         # Light gray
    }
    hatches = {
        'Gilgel Abay Built-up': '///',
        'Gumara Built-up': '\\',
        'Gumara Wetland': 'x'
    }

    # Color version for online submission
    plt.clf()
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    fig.patch.set_facecolor('white')
    #fig.suptitle('Sediment Yield vs. Land Use (2000–2020)', fontsize=20, y=1.05)

    # Panel (a): Gilgel Abay Built-up
    ax1.scatter(df_gilgel_builtup['Built-up'], df_gilgel_builtup['Sediment_Yield'],
                color=colors['Gilgel Abay Built-up'], s=50, zorder=2)
    z1 = np.polyfit(df_gilgel_builtup['Built-up'], df_gilgel_builtup['Sediment_Yield'], 1)
    p1 = np.poly1d(z1)
    ax1.plot(df_gilgel_builtup['Built-up'], p1(df_gilgel_builtup['Built-up']),
             color=colors['Gilgel Abay Built-up'], linestyle='--', zorder=1)
    ax1.text(0.05, 0.95, f'r = {correlations["Gilgel Abay Built-up"][0]:.2f}\np = {correlations["Gilgel Abay Built-up"][1]:.4f}',
             transform=ax1.transAxes, fontsize=12, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor=colors['Gilgel Abay Built-up']))
    ax1.set_title('(a) Gilgel Abay Built-up', fontsize=18)
    ax1.set_xlabel('Built-up Area (km²)', fontsize=16)
    ax1.set_ylabel('Annual Sediment Yield\n(t/ha/yr)', fontsize=16)
    ax1.grid(True, linestyle='--', alpha=0.3)
    ax1.tick_params(axis='both', labelsize=14)

    # Panel (b): Gumara Built-up
    ax2.scatter(df_gumara_builtup['Built-up'], df_gumara_builtup['Sediment_Yield'],
                color=colors['Gumara Built-up'], s=50, zorder=2)
    z2 = np.polyfit(df_gumara_builtup['Built-up'], df_gumara_builtup['Sediment_Yield'], 1)
    p2 = np.poly1d(z2)
    ax2.plot(df_gumara_builtup['Built-up'], p2(df_gumara_builtup['Built-up']),
             color=colors['Gumara Built-up'], linestyle='--', zorder=1)
    ax2.text(0.05, 0.95, f'r = {correlations["Gumara Built-up"][0]:.2f}\np = {correlations["Gumara Built-up"][1]:.4f}',
             transform=ax2.transAxes, fontsize=12, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor=colors['Gumara Built-up']))
    ax2.set_title('(b) Gumara Built-up', fontsize=18)
    ax2.set_xlabel('Built-up Area (km²)', fontsize=16)
    ax2.grid(True, linestyle='--', alpha=0.3)
    ax2.tick_params(axis='both', labelsize=14)

    # Panel (c): Gumara Wetland
    ax3.scatter(df_gumara_wetland['Wetland'], df_gumara_wetland['Sediment_Yield'],
                color=colors['Gumara Wetland'], s=50, zorder=2)
    z3 = np.polyfit(df_gumara_wetland['Wetland'], df_gumara_wetland['Sediment_Yield'], 1)
    p3 = np.poly1d(z3)
    ax3.plot(df_gumara_wetland['Wetland'], p3(df_gumara_wetland['Wetland']),
             color=colors['Gumara Wetland'], linestyle='--', zorder=1)
    ax3.text(0.05, 0.95, f'r = {correlations["Gumara Wetland"][0]:.2f}\np = {correlations["Gumara Wetland"][1]:.4f}',
             transform=ax3.transAxes, fontsize=12, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor=colors['Gumara Wetland']))
    ax3.set_title('(c) Gumara Wetland', fontsize=18)
    ax3.set_xlabel('Wetland Area (km²)', fontsize=16)
    ax3.grid(True, linestyle='--', alpha=0.3)
    ax3.tick_params(axis='both', labelsize=14)

    # Adjust layout
    plt.tight_layout(pad=2.0)

    # Save color plots
    output_png = OUTPUT_DIR / f'Figure10_Landuse_Sediment_Correlation_{uuid4().hex[:8]}_color.png'
    output_svg = OUTPUT_DIR / f'Figure10_Landuse_Sediment_Correlation_{uuid4().hex[:8]}_color.svg'
    plt.savefig(output_png, dpi=600, format='png', bbox_inches='tight')
    plt.savefig(output_svg, format='svg', bbox_inches='tight')
    print(f"Color plots saved to {output_png} (PNG) and {output_svg} (SVG)")

    # Grayscale version for print
    plt.clf()
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    fig.patch.set_facecolor('white')
    #fig.suptitle('Sediment Yield vs. Land Use (2000–2020)', fontsize=20, y=1.05)

    # Panel (a): Gilgel Abay Built-up
    scatter1 = ax1.scatter(df_gilgel_builtup['Built-up'], df_gilgel_builtup['Sediment_Yield'],
                          color=grayscale_colors['Gilgel Abay Built-up'], s=50, zorder=2)
    scatter1.set_hatch(hatches['Gilgel Abay Built-up'])
    ax1.plot(df_gilgel_builtup['Built-up'], p1(df_gilgel_builtup['Built-up']),
             color=grayscale_colors['Gilgel Abay Built-up'], linestyle='--', zorder=1)
    ax1.text(0.05, 0.95, f'r = {correlations["Gilgel Abay Built-up"][0]:.2f}\np = {correlations["Gilgel Abay Built-up"][1]:.4f}',
             transform=ax1.transAxes, fontsize=12, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor=grayscale_colors['Gilgel Abay Built-up']))
    ax1.set_title('(a) Gilgel Abay Built-up', fontsize=18)
    ax1.set_xlabel('Built-up Area (km²)', fontsize=16)
    ax1.set_ylabel('Annual Sediment Yield\n(t/ha/yr)', fontsize=16)
    ax1.grid(True, linestyle='--', alpha=0.3)
    ax1.tick_params(axis='both', labelsize=14)

    # Panel (b): Gumara Built-up
    scatter2 = ax2.scatter(df_gumara_builtup['Built-up'], df_gumara_builtup['Sediment_Yield'],
                          color=grayscale_colors['Gumara Built-up'], s=50, zorder=2)
    scatter2.set_hatch(hatches['Gumara Built-up'])
    ax2.plot(df_gumara_builtup['Built-up'], p2(df_gumara_builtup['Built-up']),
             color=grayscale_colors['Gumara Built-up'], linestyle='--', zorder=1)
    ax2.text(0.05, 0.95, f'r = {correlations["Gumara Built-up"][0]:.2f}\np = {correlations["Gumara Built-up"][1]:.4f}',
             transform=ax2.transAxes, fontsize=12, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor=grayscale_colors['Gumara Built-up']))
    ax2.set_title('(b) Gumara Built-up', fontsize=18)
    ax2.set_xlabel('Built-up Area (km²)', fontsize=16)
    ax2.grid(True, linestyle='--', alpha=0.3)
    ax2.tick_params(axis='both', labelsize=14)

    # Panel (c): Gumara Wetland
    scatter3 = ax3.scatter(df_gumara_wetland['Wetland'], df_gumara_wetland['Sediment_Yield'],
                          color=grayscale_colors['Gumara Wetland'], s=50, zorder=2)
    scatter3.set_hatch(hatches['Gumara Wetland'])
    ax3.plot(df_gumara_wetland['Wetland'], p3(df_gumara_wetland['Wetland']),
             color=grayscale_colors['Gumara Wetland'], linestyle='--', zorder=1)
    ax3.text(0.05, 0.95, f'r = {correlations["Gumara Wetland"][0]:.2f}\np = {correlations["Gumara Wetland"][1]:.4f}',
             transform=ax3.transAxes, fontsize=12, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor=grayscale_colors['Gumara Wetland']))
    ax3.set_title('(c) Gumara Wetland', fontsize=18)
    ax3.set_xlabel('Wetland Area (km²)', fontsize=16)
    ax3.grid(True, linestyle='--', alpha=0.3)
    ax3.tick_params(axis='both', labelsize=14)

    # Adjust layout
    plt.tight_layout(pad=2.0)

    # Save grayscale plots
    output_png = OUTPUT_DIR / f'Figure10_Landuse_Sediment_Correlation_{uuid4().hex[:8]}_grayscale.png'
    output_eps = OUTPUT_DIR / f'Figure10_Landuse_Sediment_Correlation_{uuid4().hex[:8]}_grayscale.eps'
    plt.savefig(output_png, dpi=600, format='png', bbox_inches='tight')
    plt.savefig(output_eps, format='eps', bbox_inches='tight')
    print(f"Grayscale plots saved to {output_png} (PNG) and {output_eps} (EPS)")

    plt.show()
    plt.close()

def main():
    """Main function to generate Figure 10."""
    print("Starting Figure 10 generation...")
    create_figure_10()

if __name__ == "__main__":
    main()