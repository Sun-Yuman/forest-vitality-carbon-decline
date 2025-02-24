"""""""""""""""""""""""""""""""""""
This script processes defoliation data files and generates combined plots of frequency distribution and Kolmogorov-Smirnov (KS) statistics heatmap.

1. Defines the time periods for analysis.e.g.1990-1999, 2000-2009, 2010-2019, 2020-2022
2. Generates frequency distribution plots smoothed by Gaussian filters.
3. Computes KS statistics between different time periods and visualizes them in a heatmap.
4. Processes data files in different groups based on function group (Broadleaves, conifers and single species) and biogeographic_regions.
"""""""""""""""""""""""""""""""""""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.colors as mcolors
from scipy.ndimage import gaussian_filter1d
from scipy.stats import ks_2samp
import matplotlib.ticker as ticker
from matplotlib import rcParams
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.patches as patches

# Set the font to Arial
rcParams['font.family'] = 'Arial'
rcParams['font.size'] = 16  # Set font size

# Define directory path and other preset parameters
directory_path = 'C:/Def(tree)'
all_files = os.listdir(directory_path)
file_paths = [os.path.join(directory_path, file) for file in all_files if file.endswith('.txt') or file.endswith('.csv')]

# Define time periods
time_periods = [
    (1990, 1999),
    (2000, 2009),
    (2010, 2019),
    (2020, 2022)
]

# Colors corresponding to each time period
colors = ['#70AD47', '#5B9BD5', '#FFC000', '#C00000']
ks_colors = ['#70AD47', '#5B9BD5', '#FFC000', '#C00000']  # KS plot colors
start_color = '#bcbddc'  # KS plot start color
end_color = '#6a51a3'  # KS plot end color

def get_alpha_cmap(start_color, end_color, alpha=0.6):
    """Create a colormap with alpha transparency."""
    color_start_rgba = mcolors.to_rgba(start_color, alpha=alpha)
    color_end_rgba = mcolors.to_rgba(end_color, alpha=alpha)
    return mcolors.LinearSegmentedColormap.from_list("alpha_cmap", [color_start_rgba, color_end_rgba])

def create_combined_plot(data, time_periods, colors, directory_path, file_path, group_name):
    """Create a combined plot of frequency distribution and KS statistics heatmap."""
    fig, ax1 = plt.subplots(figsize=(4, 3))

    # Create main distribution plot
    bin_edges = [0] + list(np.arange(5, 100, 5)) + [100, 101]
    bin_edges1 = [0] + list(np.arange(20, 100, 20)) + [100]
    labels = bin_edges[:-1]
    max_freq_values = []

    for idx, (start, end) in enumerate(time_periods):
        subset_data = data[(data['survey_year'] >= start) & (data['survey_year'] <= end)].copy()
        subset_data['binned'] = pd.cut(subset_data['code_defoliation'], bins=bin_edges, labels=labels, right=False)
        count_data = subset_data['binned'].value_counts().sort_index().reindex(labels, fill_value=0)
        freq_data = (count_data / len(subset_data)) * 100
        smoothed_values = gaussian_filter1d(freq_data.values, sigma=3)
        ax1.plot(freq_data.index, smoothed_values, label=f"{start}-{end}", color=colors[idx], linewidth=1)
        smoothed_values = np.nan_to_num(smoothed_values, nan=0.0)
        max_freq_values.append(max(smoothed_values))

    # if max_freq_values:
    #     max_freq = max(max_freq_values)
    #     max_freq = int(np.ceil(max_freq / 2) * 2)
    # else:
    #     max_freq = 0

    ax1.set_ylim(0, 14.1)
    ax1.set_yticks(np.arange(0, 14, 2))
    ax1.set_xlim(0, 100)
    ax1.set_xticks(bin_edges1)
    ax1.set_ylabel('Frequency %', fontsize=14)
    ax1.set_xlabel('Defoliation %', fontsize=14)
    ax1.grid(False)

    # Create KS heatmap
    ax2 = inset_axes(ax1, width='43%', height='55%', loc='upper right')
    cmap = get_alpha_cmap(start_color, end_color, alpha=0.8)
    ks_statistics_matrix = np.zeros((len(time_periods), len(time_periods)))
    p_values_matrix = np.zeros((len(time_periods), len(time_periods)))
    annot_matrix = np.empty((len(time_periods), len(time_periods)), dtype='object')

    for i in range(len(time_periods)):
        for j in range(len(time_periods)):
            subset_data_i = data[(data['survey_year'] >= time_periods[i][0]) & (data['survey_year'] <= time_periods[i][1])]
            subset_data_j = data[(data['survey_year'] >= time_periods[j][0]) & (data['survey_year'] <= time_periods[j][1])]

            if not subset_data_i.empty and not subset_data_j.empty:
                ks_stat, p_val = ks_2samp(subset_data_i['code_defoliation'], subset_data_j['code_defoliation'])
                ks_statistics_matrix[i, j] = ks_stat
                p_values_matrix[i, j] = p_val

                if p_val < 0.001:
                    annot_matrix[i, j] = "***"
                elif p_val < 0.01:
                    annot_matrix[i, j] = "**"
                elif p_val < 0.05:
                    annot_matrix[i, j] = "*"
                else:
                    annot_matrix[i, j] = ""

    mask = np.tril(np.ones_like(ks_statistics_matrix))

    heatmap = sns.heatmap(ks_statistics_matrix, cmap=cmap, ax=ax2, linewidths=0, mask=mask,
                          annot=annot_matrix, fmt="", annot_kws={'size': 9}, cbar=False,
                          xticklabels=False, yticklabels=False)

    # Add time period labels and lines
    period_labels = [f"{str(start)}-{str(end)}" for start, end in time_periods]
    for i, label in enumerate(period_labels):
        rect = patches.Rectangle((i, i), 1, 1, color='white')
        ax2.add_patch(rect)
        line_color = ks_colors[i % len(ks_colors)]  # Choose color

        ax2.plot([i + 0.2, i + 0.8], [i + 0.2, i + 0.2], color=line_color, linewidth=2)  # Draw lines
        ax2.text(i + 0.4, i + 0.4, label, ha='center', va='top', color=line_color, fontsize=6)  # Add labels

    ax2.text(len(time_periods) - 1.5, 3.5, "Kolmogorov-Smirnov test", ha='right', va='bottom', fontname='Arial', fontsize=6)

    if np.any(ks_statistics_matrix):
        ks_nonzero_min = np.min(ks_statistics_matrix[np.nonzero(ks_statistics_matrix)])
    else:
        ks_nonzero_min = 0

    ks_max = np.max(ks_statistics_matrix)

    # Add color bar
    cbar_ax = fig.add_axes([0.58, 0.53, 0.02, 0.22])
    cbar = fig.colorbar(heatmap.collections[0], cax=cbar_ax)
    cbar.ax.yaxis.set_ticks_position('left')  # Move ticks to left
    cbar.ax.yaxis.set_label_position('left')
    ticks = [ks_nonzero_min, (ks_nonzero_min + ks_max) / 2, ks_max]
    cbar.set_ticks(ticks)
    cbar.ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))

    cbar.ax.tick_params(labelsize=6)
    cbar.outline.set_visible(False)

    # Save the image
    group_name_clean = ''.join(e for e in group_name if e.isalnum())
    output_filename = f"{os.path.basename(file_path).split('.')[0]}_{group_name_clean}_Freq.png"
    output_path = os.path.join(directory_path, output_filename)
    plt.savefig(output_path, dpi=300, format='png', bbox_inches='tight')
    plt.close(fig)

# Function to process data and save combined plots
def process_and_save_files_with_combined_plots(file_paths, time_periods, colors, directory_path):
    """Process data files and generate combined plots."""
    for file_path in file_paths:
        print(f"Processing: {file_path}")
        if file_path.endswith('.txt'):
            data = pd.read_csv(file_path, sep='\t', encoding='gbk', low_memory=False)
        else:
            data = pd.read_csv(file_path, encoding='gbk', low_memory=False)

        create_combined_plot(data, time_periods, colors, directory_path, file_path, 'Allsp')

        for group_field in ['grp_tree_species', 'English_names', 'biogeo_reg']:
            for group_value, group_data in data.groupby(group_field):
                create_combined_plot(group_data, time_periods, colors, directory_path, file_path, f"{group_value}")

        for biogeo_value, biogeo_group in data.groupby('biogeo_reg'):
            for grp_value, grp_data in biogeo_group.groupby('grp_tree_species'):
                group_name = f"{biogeo_value}_{grp_value}"
                create_combined_plot(grp_data,  time_periods, colors, directory_path, file_path, group_name)

            for eng_value, eng_data in biogeo_group.groupby('English_names'):
                group_name = f"{biogeo_value}_{eng_value}"
                create_combined_plot(eng_data, time_periods, colors, directory_path, file_path, group_name)

# Call the function to process data and save combined plots
process_and_save_files_with_combined_plots(file_paths, time_periods, colors, directory_path)

