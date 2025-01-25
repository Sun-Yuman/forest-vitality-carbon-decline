"""""""""""""""""""""""""""""""""""
This script processes defoliation data and generates combined plots with segmented trend analysis  and  heatmaps of biogeographic regions showing Defoliation, % deviation from 1990-2019 LTA.

1. Loads the defoliation data from a CSV file.
2. Defines functions for p-value to star conversion, polynomial fit addition, and trend plotting (Theil-Sen slope and Mann-Kendall trend).
3. Generates combined plots with segmented trend lines and  biogeographic regions-specific heatmaps.
4. Saves the resulting plots and baseline data to files.
5. Filters or processes data for these categories (e.g.,Conifers and broadleaves) to create separate plots.
6. Generates forest plots with trend analysis results (Theil-Sen slope and Mann-Kendall trend) over different periods and function group (e.g.different species).

Detailed explanation of the main functions:
- `p_value_to_stars(p)`: Converts a p-value to a star notation for significance levels.
- `add_poly_fit(ax, x, y, degree)`: Adds a polynomial fit line to a plot and calculates R-squared, standard deviation, and confidence intervals.
- `combined_trend_plot_segmented_without_mk(df, ax1, ax2, column_name)`: Plots segmented trend lines for different time periods.
- `combined_plot_with_ biogeographic regions_heatmap(df, columns_list, dpi)`: Generates combined plots with trend analysis and heatmaps for different columns.
"""""""""""""""""""""""""""""""""""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import theilslopes
from pymannkendall import original_test as mk
from matplotlib import rcParams
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score
import numpy as np
from matplotlib import ticker

# Set the font to Arial
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import theilslopes
from pymannkendall import original_test as mk
from matplotlib import rcParams
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score
import numpy as np
from matplotlib import ticker

# Set the font to Arial
rcParams['font.family'] = 'Arial'
rcParams['font.size'] = '16'

# Load data
file_path = "C:/Yuman/trend/alldefoliation.csv"
data = pd.read_csv(file_path, low_memory=False)

def p_value_to_stars(p):
    """Convert a p-value to star notation for significance levels."""
    if p < 0.001:
        return '***'
    elif p < 0.01:
        return '**'
    elif p < 0.05:
        return '*'
    else:
        return ''  # not significant

def add_poly_fit(ax, x, y, degree=3):
    """Add a polynomial fit line to a plot and calculate R-squared, standard deviation, and confidence intervals."""
    poly_model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
    poly_model.fit(x[:, np.newaxis], y)
    y_fit = poly_model.predict(x[:, np.newaxis])
    ax.plot(x, y_fit, label='Polynomial fit', color='#2F4F4F')

    r2 = r2_score(y, y_fit)
    std = np.std(y - y_fit)
    ax.text(0.3, 0.69, f'$R^2$ = {r2:.2f}\n Standard deviation = {std:.2f}', transform=ax.transAxes, fontsize=14, ha='right', va='bottom', color='#2F4F4F')

    residuals = y - y_fit
    s_res = np.sqrt(np.sum(residuals**2) / (len(y) - degree - 1))
    X_design = PolynomialFeatures(degree).fit_transform(x[:, np.newaxis])
    H_matrix = X_design @ np.linalg.inv(X_design.T @ X_design) @ X_design.T
    leverage = np.diag(H_matrix)
    y_err = s_res * np.sqrt(1 - leverage)
    ci = 1.96 * y_err  # 95% confidence interval
    ax.fill_between(x, y_fit - ci, y_fit + ci, color='lightgrey', alpha=0.8, label='95% Confidence Interval')

def combined_trend_plot_segmented_without_mk(df, ax1, ax2, column_name):
    """Plot segmented trend lines for different time periods."""
    periods = [(1990, 1999, '#70AD47'), (2000, 2009, '#5B9BD5'), (2010, 2019, '#FFC000')]

    full_x = np.array([])
    full_y = np.array([])

    for start, end, color in periods:
        subset = df[(df['survey_year'] >= start) & (df['survey_year'] <= end)]
        subset_grouped = subset.groupby('survey_year')[column_name].mean()

        if not subset_grouped.empty:
            x = subset_grouped.index.values
            y = subset_grouped.values
            full_x = np.concatenate((full_x, x))
            full_y = np.concatenate((full_y, y))

            slope, intercept, _, _ = theilslopes(y, x, 0.95)
            trend_result = mk(y)
            stars = p_value_to_stars(trend_result.p)
            trend_text = f"Slope: {slope:.3f}{stars}"
            ax2.text(start + 2, 1.8, trend_text, va='top', ha='left', fontsize=16, color=color, zorder=20)

    add_poly_fit(ax1, np.arange(1990, 2023), np.interp(np.arange(1990, 2023), full_x, full_y))

    ax1.axvline(x=2000, color='grey', linestyle='--')
    ax1.axvline(x=2010, color='grey', linestyle='--')
    ax1.axvline(x=2020, color='grey', linestyle='--')
    ax2.axvline(x=2000, color='grey', linestyle='--')
    ax2.axvline(x=2010, color='grey', linestyle='--')
    ax2.axvline(x=2020, color='grey', linestyle='--')

def combined_plot_with_bioregion_heatmap(df, columns_list, dpi=300):
    """Generate combined plots with trend analysis and heatmaps for different columns."""
    fig = plt.figure(figsize=(11, 12), dpi=dpi)
    total_height = 0.95
    gap = 0.012
    ax_height = 0.13 # Height for each subplot
    heatmap_height = 0.16
    ax3_gap = 0.001 # Control the distance between ax2 and ax3

    group_height = ax_height + heatmap_height + ax3_gap
    group_positions = [total_height - (i + 1) * (group_height + gap) for i in range(len(columns_list))]  # Control the position of each subplot group

    ax_positions = [[0.15, pos + heatmap_height + ax3_gap, 0.82, ax_height] for pos in group_positions]
    heatmap_positions = [[0.15, pos, 0.82, heatmap_height] for pos in group_positions]

    color_scheme = ["#053061","#2166ac", "#4393c3",  "#92c5de","#d1e5f0","#fddbc7","#f4a582","#d6604d","#b2182b","#67001f"]
    specific_regions = ["Alpine", "Atlantic", "Boreal", "Continental", "Mediterranean", "Pannonian"]

    baselines = []
    bio_baselines = []

    ax1_height = 0.13
    ax2_height = 0.01

    for idx, column_name in enumerate(columns_list):
        ax1 = fig.add_axes([ax_positions[idx][0], ax_positions[idx][1] + ax2_height, ax_positions[idx][2], ax1_height])
        ax2 = fig.add_axes([ax_positions[idx][0], ax_positions[idx][1], ax_positions[idx][2], ax2_height], sharex=ax1)

        combined_trend_plot_segmented_without_mk(df, ax1, ax2, column_name)

        df['survey_year'] = df['survey_year'].astype(int)
        filtered_data = df[(df['survey_year'] >= 1990) & (df['survey_year'] <= 2019)]
        overall_average = filtered_data[column_name].mean()
        yearly_averages = df.groupby('survey_year')[column_name].mean()
        deltas = yearly_averages - overall_average
        bar_colors = [color_scheme[-2] if delta >= 0 else color_scheme[1] for delta in deltas.values]

        ax1.bar(yearly_averages.index, deltas.values, bottom=overall_average, color=bar_colors, width=0.8, alpha=0.8)
        ax1.axhline(overall_average, color='black', linestyle='-', linewidth=0.5)
        ax1.set_xlim(1989.5, 2022.5)

        ax1.set_ylim(14, 30)
        ax1.set_yticks(range(15, 31, 5))  # Set Y-axis ticks

        ax2.bar(yearly_averages.index, deltas.values, bottom=overall_average, color=bar_colors, width=0.8, alpha=0.8)
        ax2.set_ylim(0, 1)
        ax2.set_yticks(range(0, 1, 1))
        ax1.set_ylabel("Defoliation %", labelpad=4, fontsize=15)
        ax1.yaxis.set_label_coords(-0.1, 0.5)  # Move the label to the left

        # Hide the bottom spine of the first subplot and the top spine of the second subplot
        ax1.spines['bottom'].set_visible(False)
        ax2.spines['top'].set_visible(False)
        ax2.set_xticks([])
        ax2.set_xticklabels([])

        # Add break lines
        d = .015
        offset1 = -0.01
        offset2 = -0.03
        right_offset = 0.005
        # Draw break lines on ax1
        kwargs = dict(transform=ax1.transAxes, color='k', clip_on=False, zorder=20)
        ax2.plot((-d+right_offset, +d), (offset1 - d, offset1 + d), **kwargs)
        ax2.plot((1 - d + right_offset, 1 + d), (offset1 - d, offset1 + d), **kwargs)
        ax2.plot((-d + right_offset, +d), (offset2 - d, offset2 + d), **kwargs)
        ax2.plot((1 - d + right_offset, 1 + d), (offset2 - d, offset2 + d), **kwargs)

        df['residuals'] = df[column_name] - overall_average
        residuals_by_bio_year = df.pivot_table(index='biogeo_reg', columns='survey_year', values='residuals', aggfunc='mean')
        residuals_by_bio_year = residuals_by_bio_year.loc[specific_regions]

        ax3 = fig.add_axes(heatmap_positions[idx])
        sns.heatmap(residuals_by_bio_year, cmap=sns.color_palette(color_scheme, as_cmap=True),
                    annot=True, fmt=".1f", center=0, cbar=False, linewidths=0, vmin=-15, vmax=15, annot_kws={"size": 9},
                    ax=ax3)
        ax3.tick_params(axis='x', labelsize=14)
        ax3.set_ylabel('')
        if idx < len(columns_list) - 1:
            ax3.set_xticklabels([]), ax3.set_xticks([])
        ax3.set_xlabel([])
        ax3.set_xlabel("Year", labelpad=12, fontsize=14)
        baselines.append({'Column': column_name, 'Baseline': overall_average})

        for region in specific_regions:
            dfbio = df[(df['survey_year'] >= 1990) & (df['survey_year'] <= 2019)]
            bio_average = dfbio[dfbio['biogeo_reg'] == region][column_name].mean()
            bio_baselines.append({'Column': column_name, 'Biogeo Reg': region, 'Baseline': bio_average})

    min_val = -15
    max_val = 15

    class CustomFormatter(ticker.Formatter):
        def __init__(self, min_val, max_val):
            self.min_val = min_val
            self.max_val = max_val

        def __call__(self, x, pos=None):
            if x <= self.min_val:
                return f'< {self.min_val}'
            elif x >= self.max_val:
                return f'> {self.max_val}'
            else:
                return f'{x:.0f}'

    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=len(labels), bbox_to_anchor=(0.645, 0.95), fontsize=14,
               columnspacing=1.3, frameon=True, fancybox=True, framealpha=0.5, facecolor='white', edgecolor='white')
    # Save legend(colorbar) as a separate image
    colorbar_fig = plt.figure(figsize=(6, 2))
    colorbar_ax = colorbar_fig.add_axes([0.05, 0.5, 0.9, 0.15])
    plt.colorbar(ax3.collections[0], cax=colorbar_ax, orientation='horizontal', format=CustomFormatter(min_val, max_val),
                 ticks=np.linspace(min_val, max_val, num=11))
    colorbar_file_path = "C:/Yuman/trend/colorbar_plot_horizontal.jpg"
    colorbar_fig.savefig(colorbar_file_path, format='jpg', dpi=600, bbox_inches='tight')
    plt.close(colorbar_fig)

    colorbar_fig = plt.figure(figsize=(2, 6))
    colorbar_ax = colorbar_fig.add_axes([0.05, 0.05, 0.15, 0.9])
    plt.colorbar(ax3.collections[0], cax=colorbar_ax, orientation='vertical', format=CustomFormatter(min_val, max_val),
                 ticks=np.linspace(min_val, max_val, num=11))
    colorbar_file_path = "C:/Yuman/trend/colorbar_plot_vertical.jpg"
    colorbar_fig.savefig(colorbar_file_path, format='jpg', dpi=600, bbox_inches='tight')
    plt.close(colorbar_fig)

    file_path = "C:/Yuman/trend/Trend_Poly_Heatmap.jpg"
    plt.savefig(file_path, format='jpg', dpi=600)
    plt.show()
    plt.close(fig)

    baseline_df = pd.DataFrame(baselines)
    bio_baseline_df = pd.DataFrame(bio_baselines)

    combined_baseline_df = baseline_df.merge(bio_baseline_df, on='Column', how='outer')
    combined_baseline_df.to_csv("C:/Yuman/trend/baseline_combine.csv", index=False)


combined_plot_with_bioregion_heatmap(data, ["All", "broadleaves", "conifers"])

def collect_trend_data(df, column_name, region=None):
    """Collect trend data for a specified column and region over different periods."""
    results = []
    colormaps = ['BuGn', 'Wistia', 'BuPu']
    new_colors = [plt.get_cmap(cmap)(0.8) for cmap in colormaps]
    periods = [(1990, 1999, new_colors[0]), (2000, 2009, new_colors[1]), (2010, 2019, new_colors[2]), (1990, 2022, 'grey')]

    for start, end, color in periods:
        if region:
            subset = df[(df['survey_year'] >= start) & (df['survey_year'] <= end) & (df['biogeo_reg'] == region)]
        else:
            subset = df[(df['survey_year'] >= start) & (df['survey_year'] <= end)]

        if not subset.empty:
            subset_grouped = subset.groupby('survey_year')[column_name]
            mean_values = subset_grouped.mean().dropna()
            x = mean_values.index.values
            y = mean_values.values
            total_points = subset_grouped.size().sum()

            valid_years = subset_grouped.count().loc[lambda x: x > 0].size

            if len(y) > 1:
                slope, intercept, lo_slope, up_slope = theilslopes(y, x, 0.95)
                trend_result = mk(y)
                stars = p_value_to_stars(trend_result.p)
                std_dev = np.std(y, ddof=1)
                std_error = std_dev / np.sqrt(len(y))
                ci_lower = slope - 1.96 * std_error
                ci_upper = slope + 1.96 * std_error
                results.append({
                    "Region": region if region else "Overall",
                    "Variable": column_name,
                    "Period": f"{start}-{end}",
                    "Slope": slope,
                    "Intercept": intercept,
                    "Lower CI Slope": ci_lower,
                    "Upper CI Slope": ci_upper,
                    "Trend": trend_result.trend,
                    "P-value": trend_result.p,
                    "Stars": stars,
                    "Standard Error": std_error,
                    "Standard Deviation": std_dev,
                    "Sample Size": valid_years,
                    "Total Data Points for Averaging": total_points
                })
        else:
            print(f"No data available for {column_name} in region {region} during {start}-{end}")
    return results

def combined_plot_with_heatmap(df, columns_list, dpi=300):
    """Generate combined plots with trend analysis and heatmaps for different columns."""
    regions = df['biogeo_reg'].unique().tolist() + [None]
    all_trend_data = []

    for region in regions:
        for column_name in columns_list:
            try:
                trend_data = collect_trend_data(df, column_name, region)
                all_trend_data.extend(trend_data)
            except Exception as e:
                print(f"Error processing {column_name} for region {region}: {e}")

    trend_df = pd.DataFrame(all_trend_data)
    trend_df.to_csv(f"C:/Yuman/trend/allperiod_mktrend_regions.csv", index=False)

# Generate the trend data
combined_plot_with_heatmap(data, ["All", "broadleaves", "conifers", "dombroadleaves","domconifers", "mixed", "Beech", "Oak", "Norway spruce", "Scots pine"])

# Load the trend data for plotting
file_path = "C:/Yuman/trend/allperiod_mktrend_regions.csv"
data = pd.read_csv(file_path, low_memory=False)

# Convert 'Trend' column to numerical values
trend_mapping = {
    'increasing': 1,
    'decreasing': -1,
    'stable': 0
}
data['Trend'] = data['Trend'].map(trend_mapping).fillna(0).astype(float)

# Ensure 'P-value' column is numeric
data['P-value'] = data['P-value'].astype(float)

# Map star levels to integers
star_mapping = {'***': 3, '**': 2, '*': 1, '': 0}
data['Stars Int'] = data['Stars'].map(star_mapping).fillna(0).astype(int)

# Rename variables
variable_mapping = {
    "All": "All species",
    "broadleaves": "Broadleaves",
    "conifers": "Conifers"
}
data['Variable'] = data['Variable'].replace(variable_mapping)

# Specify the order of variables
ordered_variables = ["All species", "Broadleaves", "Conifers", "Beech", "Oak", "Norway spruce", "Scots pine"][::-1]

data = data[data['Variable'].isin(ordered_variables)]
data['Variable'] = pd.Categorical(data['Variable'], categories=ordered_variables, ordered=True)
data.sort_values('Variable', inplace=True)

# Only keep specified regions
region_order = ["Overall"]
data = data[data['Region'].isin(region_order)]

data['Region'] = pd.Categorical(data['Region'], categories=region_order, ordered=True)
data.sort_values('Region', inplace=True)

# Define color mapping
palette = sns.color_palette("husl", len(ordered_variables))
color_mapping = dict(zip(ordered_variables, palette))

def plot_period(ax, period_data, xlabel):
    """Plot trend data for a specified period."""
    means = period_data['Slope']
    ci_lower = period_data['Lower CI Slope']
    ci_upper = period_data['Upper CI Slope']
    variables = period_data['Variable']
    stars_int = period_data['Stars Int']
    colors = [color_mapping[v] for v in variables]
    y_positions = [ordered_variables.index(v) for v in variables]

    for j, pos in enumerate(y_positions):
        ax.errorbar(means.iloc[j], pos, xerr=[[means.iloc[j] - ci_lower.iloc[j]], [ci_upper.iloc[j] - means.iloc[j]]],
                    fmt='o', color=colors[j], ecolor=colors[j], elinewidth=3, capsize=0, markersize=5)

    ax.set_xlim(-3, 3)
    ax.set_xticks(np.arange(-2, 3, 1))
    ax.set_xticklabels(np.arange(-2, 3, 1))
    ax.set_yticks(range(len(ordered_variables)))
    ax.set_yticklabels(ordered_variables)
    ax.axvline(x=0, color='grey', linestyle='--')
    ax.set_xlabel(xlabel)

    for j, pos in enumerate(y_positions):
        ax.annotate('*' * stars_int.iloc[j], (means.iloc[j], pos), textcoords="offset points", xytext=(5, 0),
                    ha='left')

# Plot the trend data for different periods
fig, axs = plt.subplots(1, 4, figsize=(6, 4), sharey=True)

periods = ['1990-2022', '1990-1999', '2000-2009', '2010-2019']
xlabels = ['Slope (1990-2022)', 'Slope (1990-1999)', 'Slope (2000-2009)', 'Slope (2010-2019)']

for ax, period, xlabel in zip(axs, periods, xlabels):
    period_data = data[data['Period'] == period]
    plot_period(ax, period_data, xlabel)

for ax in axs:
    ax.spines['top'].set_color('black')
    ax.spines['right'].set_color('black')
    ax.spines['bottom'].set_color('black')
    ax.spines['left'].set_color('black')

plt.tight_layout(pad=2.0, w_pad=0.5, h_pad=0.5)
plt.savefig('trend_region_periods_husl.png', dpi=300)
plt.show()