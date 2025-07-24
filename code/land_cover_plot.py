# Script to generate Figure 3: Grouped bar plot of land-use and land-cover (LULC) areas
# in the Gilgel Abay and Gumara watersheds for 2000 and 2020, derived from the Global Land Analysis
# and Discovery (GLAD) dataset. Bars show areas (km²) for each land cover type, with distinct colors
# for Gilgel Abay (2000, 2020) and Gumara (2000, 2020), and exact percentage change annotations
# above 2020 bars (-6%, -16%, +18%, +940%, 0%, -14%, +6%, +580%, -63%). Wetland is included only for Gumara.
# Outputs publication-quality PNG/SVG plots (600 DPI).
# Author: Kindie B. Worku
# Date: 2025-07-20

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from uuid import uuid4
import numpy as np

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

# LULC data from provided table
data = {
    'Watershed': ['Gilgel Abay', 'Gilgel Abay', 'Gilgel Abay', 'Gilgel Abay', 'Gumara', 'Gumara', 'Gumara', 'Gumara', 'Gumara'],
    'Land Cover': ['Cropland', 'Dense short vegetation', 'Tree cover', 'Built-up', 'Cropland', 'Dense short vegetation', 'Tree cover', 'Built-up', 'Wetland'],
    'Area_2000': [590, 631, 394, 7.3, 875, 304, 178, 4.6, 4.4],
    'Area_2020': [551, 527, 469, 76, 882, 261, 190, 32, 1.6],
    'Change (%)': [-6, -16, 18, 940, 0, -14, 6, 580, -63]
}
df = pd.DataFrame(data)

# Abbreviated labels for x-axis
df['Land Cover Abbrev'] = ['Cropland', 'Dense Veg.', 'Tree Cover', 'Built-up', 'Cropland', 'Dense Veg.', 'Tree Cover', 'Built-up', 'Wetland']

# Filter out Wetland for Gilgel Abay
df = df[~((df['Watershed'] == 'Gilgel Abay') & (df['Land Cover'] == 'Wetland'))]

def create_figure_3():
    """Generate Figure 3: Grouped bar plot of LULC areas for Gilgel Abay and Gumara (2000, 2020) in color and grayscale."""
    print("Generating Figure 3...")

    # Reshape data for grouped bar plot
    df_melted = pd.melt(
        df,
        id_vars=['Watershed', 'Land Cover', 'Land Cover Abbrev', 'Change (%)'],
        value_vars=['Area_2000', 'Area_2020'],
        var_name='Year',
        value_name='Area'
    )
    df_melted['Year'] = df_melted['Year'].replace({'Area_2000': '2000', 'Area_2020': '2020'})
    df_melted['Group'] = df_melted['Watershed'] + ' ' + df_melted['Year']

    # Debug: Print df_melted to check data
    print("df_melted contents:")
    print(df_melted[['Land Cover Abbrev', 'Group', 'Area', 'Change (%)']].to_string(index=False))

    # Define colors and hatches for bars
    colors = {
        'Gilgel Abay 2000': '#6baed6',  # Light blue
        'Gilgel Abay 2020': '#1f77b4',  # Dark blue
        'Gumara 2000': '#fdd0a2',       # Light orange
        'Gumara 2020': '#ff7f0e'        # Dark orange
    }
    hatches = {
        'Gilgel Abay 2000': '///',
        'Gilgel Abay 2020': '\\',
        'Gumara 2000': '///',
        'Gumara 2020': '\\'
    }

    # Create color version for online submission
    plt.clf()
    fig, ax = plt.subplots(figsize=(12, 5))
    fig.patch.set_facecolor('white')

    # Plot grouped bars (color)
    barplot = sns.barplot(
        data=df_melted,
        x='Land Cover Abbrev',
        y='Area',
        hue='Group',
        palette=colors,
        ax=ax,
        hue_order=['Gilgel Abay 2000', 'Gilgel Abay 2020', 'Gumara 2000', 'Gumara 2020']
    )

    # Create group order matching barplot order
    group_order = []
    for cover in df_melted['Land Cover Abbrev'].unique():
        for group in ['Gilgel Abay 2000', 'Gilgel Abay 2020', 'Gumara 2000', 'Gumara 2020']:
            if ((group.startswith('Gilgel Abay') and cover != 'Wetland') or
                (group.startswith('Gumara') and (cover in df_melted[df_melted['Group'] == group]['Land Cover Abbrev'].values))):
                group_order.append(group)

    # Apply hatching
    for i, bar in enumerate(barplot.patches):
        if i < len(group_order):
            bar.set_hatch(hatches[group_order[i]])

    # Debug: Print bar positions and heights
    print("Color bar positions:")
    for i, bar in enumerate(barplot.patches):
        print(f"Bar {i}: x={bar.get_x():.2f}, width={bar.get_width():.2f}, height={bar.get_height():.2f}, group={group_order[i] if i < len(group_order) else 'N/A'}")

    # Add percentage change annotations above 2020 bars
    bar_counter = 0
    expected_bars = {
        ('Cropland', 'Gilgel Abay 2020', 551.0): -6,
        ('Cropland', 'Gumara 2020', 882.0): 0,
        ('Dense Veg.', 'Gilgel Abay 2020', 527.0): -16,
        ('Dense Veg.', 'Gumara 2020', 261.0): -14,
        ('Tree Cover', 'Gilgel Abay 2020', 469.0): 18,
        ('Tree Cover', 'Gumara 2020', 190.0): 6,
        ('Built-up', 'Gilgel Abay 2020', 76.0): 940,
        ('Built-up', 'Gumara 2020', 32.0): 580,
        ('Wetland', 'Gumara 2020', 1.6): -63
    }
    for i, bar in enumerate(barplot.patches):
        height = bar.get_height()
        for _, row in df_melted[df_melted['Year'] == '2020'].iterrows():
            if abs(row['Area'] - height) < 0.01:  # Match height within tolerance
                cover = row['Land Cover Abbrev']
                group = row['Group']
                change = row['Change (%)']
                key = (cover, group, height)
                if key in expected_bars and expected_bars[key] == change:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        height + 100,  # Increased offset for tall bars
                        f'{change:+.0f}%',
                        ha='center',
                        va='bottom',
                        fontsize=12,
                        rotation=45,
                        color=colors[group]
                    )
                    bar_counter += 1
                    print(f"Annotated {cover} ({group}): {change:+.0f}% at bar index {i}, height={height:.2f}")
                    break

    print(f"Total annotated bars for 2020 (color): {bar_counter}")

    # Customize plot
    ax.set_xlabel('Land Cover Type', fontsize=16)
    ax.set_ylabel('Area (km²)', fontsize=16)
    ax.set_ylim(0, 1200)  # Adjusted for largest area
    ax.tick_params(axis='x', rotation=45, labelsize=14)
    ax.tick_params(axis='y', labelsize=14)
    ax.grid(True, axis='y', linestyle='--', alpha=0.3)

    # Create custom legend with matching hatches
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#6baed6', edgecolor='black', hatch='///', label='Gilgel Abay 2000'),
        Patch(facecolor='#1f77b4', edgecolor='black', hatch='\\', label='Gilgel Abay 2020'),
        Patch(facecolor='#fdd0a2', edgecolor='black', hatch='///', label='Gumara 2000'),
        Patch(facecolor='#ff7f0e', edgecolor='black', hatch='\\', label='Gumara 2020')
    ]
    ax.legend(handles=legend_elements, title='Watershed and Year', fontsize=14, title_fontsize=16, loc='upper right')

    # Save color plots
    output_png = OUTPUT_DIR / f'Figure3_LULC_Areas_{uuid4().hex[:8]}_color.png'
    output_svg = OUTPUT_DIR / f'Figure3_LULC_Areas_{uuid4().hex[:8]}_color.svg'
    plt.savefig(output_png, dpi=600, format='png', bbox_inches='tight')
    plt.savefig(output_svg, format='svg', bbox_inches='tight')
    print(f"Color plots saved to {output_png} (PNG) and {output_svg} (SVG)")

    # Define grayscale colors
    grayscale_colors = {
        'Gilgel Abay 2000': '#666666',  # Medium gray
        'Gilgel Abay 2020': '#000000',  # Black
        'Gumara 2000': '#999999',       # Light gray
        'Gumara 2020': '#333333'        # Dark gray
    }

    # Create grayscale version for print
    plt.clf()
    fig, ax = plt.subplots(figsize=(12, 5))
    fig.patch.set_facecolor('white')

    # Plot grouped bars (grayscale)
    barplot = sns.barplot(
        data=df_melted,
        x='Land Cover Abbrev',
        y='Area',
        hue='Group',
        palette=grayscale_colors,
        ax=ax,
        hue_order=['Gilgel Abay 2000', 'Gilgel Abay 2020', 'Gumara 2000', 'Gumara 2020']
    )

    # Apply hatching
    for i, bar in enumerate(barplot.patches):
        if i < len(group_order):
            bar.set_hatch(hatches[group_order[i]])

    # Debug: Print bar positions and heights
    print("Grayscale bar positions:")
    for i, bar in enumerate(barplot.patches):
        print(f"Bar {i}: x={bar.get_x():.2f}, width={bar.get_width():.2f}, height={bar.get_height():.2f}, group={group_order[i] if i < len(group_order) else 'N/A'}")

    # Add percentage change annotations above 2020 bars
    bar_counter = 0
    for i, bar in enumerate(barplot.patches):
        height = bar.get_height()
        for _, row in df_melted[df_melted['Year'] == '2020'].iterrows():
            if abs(row['Area'] - height) < 0.01:  # Match height within tolerance
                cover = row['Land Cover Abbrev']
                group = row['Group']
                change = row['Change (%)']
                key = (cover, group, height)
                if key in expected_bars and expected_bars[key] == change:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        height + 100,  # Increased offset for tall bars
                        f'{change:+.0f}%',
                        ha='center',
                        va='bottom',
                        fontsize=12,
                        rotation=45,
                        color=grayscale_colors[group]
                    )
                    bar_counter += 1
                    print(f"Annotated {cover} ({group}): {change:+.0f}% at bar index {i}, height={height:.2f}")
                    break

    print(f"Total annotated bars for 2020 (grayscale): {bar_counter}")

    # Customize plot
    ax.set_xlabel('Land Cover Type', fontsize=16)
    ax.set_ylabel('Area (km²)', fontsize=16)
    ax.set_ylim(0, 1200)  # Adjusted for largest area
    ax.tick_params(axis='x', rotation=45, labelsize=14)
    ax.tick_params(axis='y', labelsize=14)
    ax.grid(True, axis='y', linestyle='--', alpha=0.3)

    # Create custom legend with matching hatches for grayscale
    legend_elements = [
        Patch(facecolor='#666666', edgecolor='black', hatch='///', label='Gilgel Abay 2000'),
        Patch(facecolor='#000000', edgecolor='black', hatch='\\', label='Gilgel Abay 2020'),
        Patch(facecolor='#999999', edgecolor='black', hatch='///', label='Gumara 2000'),
        Patch(facecolor='#333333', edgecolor='black', hatch='\\', label='Gumara 2020')
    ]
    ax.legend(handles=legend_elements, title='Watershed and Year', fontsize=14, title_fontsize=16, loc='upper right')

    # Save grayscale plots
    output_png = OUTPUT_DIR / f'Figure3_LULC_Areas_{uuid4().hex[:8]}_grayscale.png'
    output_eps = OUTPUT_DIR / f'Figure3_LULC_Areas_{uuid4().hex[:8]}_grayscale.eps'
    plt.savefig(output_png, dpi=600, format='png', bbox_inches='tight')
    plt.savefig(output_eps, format='eps', bbox_inches='tight')
    print(f"Grayscale plots saved to {output_png} (PNG) and {output_eps} (EPS)")

    plt.show()
    plt.close()

def main():
    """Main function to generate Figure 3."""
    print("Starting Figure 3 generation...")
    create_figure_3()

if __name__ == "__main__":
    main()