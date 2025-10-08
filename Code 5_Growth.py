"""""""""""""""""""""""""""""""""""
Summary of the script: data analysis of growth reduction and defoliation using linear regression, sensitivity analysis, and Growth reduction based on 1990-1999.

1. Load and clean data by removing rows with missing values for 'Growth' and 'Defoliation'.
2. Perform linear regression analysis between 'Growth' and 'Defoliation' and assess statistical significance.
3. Conduct group-based 5 fold cross-validation to calculate average R² values.
4. Plot scatter plots with regression lines and confidence intervals for 'overall' and 'BC (Broadleaves and conifers)' categorical data.
5. Conduct sensitivity analysis by adding random bias to 'Defoliation' values and evaluate impact(Group 5-fold cross-validiation).
6. Process cumulative growth data by splitting groups into 'biogeographic regions' and ''BC (Broadleaves and conifers)' across different time periods.
7. Create bar charts to depict relative growth across different regions and time periods.

"""""""""""""""""""""""""""""""""""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
# Set the font and size
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 14

# Load and clean the data

BASE_DIR = 'C:/Growth/'
file_path = os.path.join(BASE_DIR, 'Growth_defoliation(Metadata).csv')
# Replace with your file path
data = pd.read_csv(file_path)
filtered_data = data.dropna(subset=['Growth', 'Defoliation'])

# Create a color palette and assign it to each category
color_palette = sns.color_palette()
unique_categories = filtered_data['BC'].unique()
category_colors = {cat: color_palette[idx % len(color_palette)] for idx, cat in enumerate(unique_categories)}

# Linear regression analysis function
def linear_regression_details(data, x, y):
    X = sm.add_constant(data[x])
    model = sm.OLS(data[y], X).fit()
    return model

# Function to determine the significance asterisks based on p-value
def significance_asterisks(p_value):
    if p_value < 0.001:
        return '***'
    elif p_value < 0.01:
        return '**'
    elif p_value < 0.05:
        return '*'
    else:
        return ''

# Function to format the regression equation
def format_regression_equation(model):
    slope = model.params.iloc[1]
    intercept = model.params.iloc[0]
    intercept_sign = '+' if intercept >= 0 else '-'
    intercept_value = abs(intercept)
    return f'y = {slope:.2f}x {intercept_sign} {intercept_value:.2f}'

# Function to calculate cross-validated R²
def cross_validated_r2(data, x, y, groups):
    X = sm.add_constant(data[x])
    y_values = data[y]
    group_kfold = GroupKFold(n_splits=5)
    r2_scores = []

    for train_index, test_index in group_kfold.split(X, y_values, groups=groups):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y_values.iloc[train_index], y_values.iloc[test_index]
        model = sm.OLS(y_train, X_train).fit()
        y_pred = model.predict(X_test)
        r2_scores.append(r2_score(y_test, y_pred))

    return np.mean(r2_scores)

# Function to plot regression line with confidence interval and return label text
def plot_regression_line(g, data, x, y, label, color, groups, is_original=False):
    model = linear_regression_details(data, x, y)
    x_vals = np.linspace(min(data[x]), max(data[x]), 200)
    predictions = model.get_prediction(sm.add_constant(x_vals))
    prediction_summary = predictions.summary_frame(alpha=0.05)
    line_color = 'gray' if is_original else color
    g.ax_joint.plot(x_vals, prediction_summary["mean"], color=line_color)
    g.ax_joint.fill_between(x_vals, prediction_summary["mean_ci_lower"], prediction_summary["mean_ci_upper"], color=color, alpha=0.1)

    cv_r2 = cross_validated_r2(data, x, y, groups)
    regression_equation = format_regression_equation(model)
    significance = significance_asterisks(model.pvalues.iloc[1])
    label_text = f'{label}: {regression_equation}{significance}, CV R² = {cv_r2:.2f}'
    return label_text

# Function to create a plot with regression line and marginal density plots
def plot_jointplot_with_regression(data, x, y, title, color, filename=None):
    plt.figure(figsize=(10, 8))
    g = sns.JointGrid(x=x, y=y, data=data, xlim=(0, 100), ylim=(-100, 0))
    g = g.plot_joint(sns.scatterplot, color=color, alpha=0.6)

    model = linear_regression_details(data, x, y)
    x_vals = np.linspace(0, 100, 200)
    predictions = model.get_prediction(sm.add_constant(x_vals))
    prediction_summary = predictions.summary_frame(alpha=0.05)

    g.ax_joint.plot(x_vals, prediction_summary["mean"], color=color, alpha=0.8)
    g.ax_joint.fill_between(x_vals, prediction_summary["mean_ci_lower"], prediction_summary["mean_ci_upper"],
                            color=color, alpha=0.1)

    # Plot marginal density plots with fill
    g.plot_marginals(sns.kdeplot, color=color, fill=True, alpha=0.5)

    g.set_axis_labels("Defoliation %", "Growth Reduction %")
    rsquared = model.rsquared
    p_value = model.pvalues.iloc[1]
  #  correlation = np.sqrt(rsquared)
    significance = significance_asterisks(p_value)

    intercept = model.params.iloc[0]
    slope = model.params.iloc[1]
    intercept_sign = '+' if intercept >= 0 else '-'
    intercept_abs = abs(intercept)

    g.ax_joint.text(100, -0.50,
                    f'y = {slope:.2f}x {intercept_sign} {intercept_abs:.2f}{significance}\nR² = {rsquared:.2f}',
                    fontsize=14, ha='right', va='top')

    g.ax_joint.tick_params(axis='both', which='major', labelsize=14)
    plt.subplots_adjust(top=0.9)

    if filename:
        plt.savefig(filename)

# Function to create a chart with regression lines and marginal KDE plots
def plot_jointplot_with_sensitivity(data, x, y, bc_category, color, title, std_dev_x, bias, filename=None):
    plt.figure(figsize=(10, 8))
    g = sns.JointGrid(x=x, y=y, data=data, space=0, xlim=(0, 100), ylim=(-100, 0))

    # Plot the scatter plot
    sns.scatterplot(x=x, y=y, data=data, alpha=0.6, ax=g.ax_joint, color='gray')

    # Store information for the legend
    lines = []
    labels = []

    # Plot regression line for original data and add to legend
    groups = data['ID']  # Group by ID
    original_label_text = plot_regression_line(g, data, x, y, 'Original', color, groups, is_original=True)
    lines.extend(g.ax_joint.lines[-1:])
    labels.append(f'{original_label_text}')

    # Regression lines for sensitivity analysis
    for b, c in zip(bias, sns.color_palette("husl", len(bias))):
        biased_x = data[x] + np.random.uniform(-b, b, len(data))
        biased_x = np.clip(biased_x, 0, 100)  # Ensure values are within 0-100
        biased_data = pd.concat([biased_x, data[y]], axis=1)
        sensitivity_label_text = plot_regression_line(g, biased_data, x, y, f'{b}', c, groups)
        lines.extend(g.ax_joint.lines[-1:])
        labels.append(f' {sensitivity_label_text}')

    # Add marginal KDE plots
    sns.kdeplot(data[x], ax=g.ax_marg_x, color='gray', alpha=0.5, fill=True)
    sns.kdeplot(data[y], ax=g.ax_marg_y, color='gray', alpha=0.5, fill=True, vertical=True)

    # Final adjustments
    g.ax_joint.grid(False)
    g.set_axis_labels("Defoliation %", "Growth Reduction %")

    # Create legend
    g.ax_joint.legend(handles=lines, labels=labels, title=f'{bc_category}', loc='lower left', borderaxespad=0., frameon=False, fontsize='small')

    if filename:
        plt.savefig(filename, bbox_inches='tight', dpi=300)

# Create an empty DataFrame to store the results
results_df = pd.DataFrame(
    columns=['code_defoliation', 'Mean Growth', 'Min Growth', 'Max Growth', 'grp_tree_species', 'Intercept', 'Slope',
             'R_squared', 'RMSE', 'P_value', 'Correlation', 'n_points'])

# Analysis and plotting
categories = ['BC', 'English_names']
x = 'Defoliation'
y = 'Growth'

# Analysis of overall data
overall_model = linear_regression_details(filtered_data, x, y)
# Select specific Defoliation values
defoliation_values = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 99, 100]
overall_predictions = overall_model.get_prediction(sm.add_constant(defoliation_values)).summary_frame(alpha=0.05)
plot_jointplot_with_regression(filtered_data, x, y, 'Overall Data', color='#404040', filename='all.png')

# Add the overall data analysis results to results_df
for val, pred in zip(defoliation_values, overall_predictions.itertuples()):
    n_points = len(filtered_data[filtered_data[x] == val])
    row = pd.DataFrame([{
        'code_defoliation': val,
        'Mean Growth': pred.mean,
        'Min Growth': pred.mean_ci_lower,
        'Max Growth': pred.mean_ci_upper,
        'grp_tree_species': 'All Data',
        'Intercept': overall_model.params.iloc[0],
        'Slope': overall_model.params.iloc[1],
        'R_squared': overall_model.rsquared,
        'RMSE': np.sqrt(overall_model.mse_resid),
        'P_value': overall_model.pvalues.iloc[1],
        'Correlation': np.sqrt(overall_model.rsquared),
        'n_points': n_points
    }])
    results_df = pd.concat([results_df, row], ignore_index=True)

# Analysis for categorical data
for category in categories:
    unique_categories = filtered_data[category].unique()
    for cat in unique_categories:
        cat_data = filtered_data[filtered_data[category] == cat]
        if not cat_data.empty:
            cat_model = linear_regression_details(cat_data, x, y)
            cat_predictions = cat_model.get_prediction(sm.add_constant(defoliation_values)).summary_frame(alpha=0.05)
            color = category_colors.get(cat, 'k')  # Use .get to avoid KeyError, default to 'k' (black)
            plot_jointplot_with_regression(cat_data, x, y, f'{category}: {cat}', color=color,
                                           filename=f'1_Fit_{cat}.png')

            # Add the category data analysis results to results_df
            for val, pred in zip(defoliation_values, cat_predictions.itertuples()):
                n_points = len(cat_data[cat_data[x] == val])
                row = pd.DataFrame([{
                    'code_defoliation': val,
                    'Mean Growth': pred.mean,
                    'Min Growth': pred.mean_ci_lower,
                    'Max Growth': pred.mean_ci_upper,
                    'grp_tree_species': cat,
                    'Intercept': cat_model.params.iloc[0],
                    'Slope': cat_model.params.iloc[1],
                    'R_squared': cat_model.rsquared,
                    'RMSE': np.sqrt(cat_model.mse_resid),
                    'P_value': cat_model.pvalues.iloc[1],
                    'Correlation': np.sqrt(cat_model.rsquared),
                    'n_points': n_points
                }])
                results_df = pd.concat([results_df, row], ignore_index=True)

# Save the results to a CSV file
BASE_DIR = 'C:/Growth/'

results_csv_path = os.path.join(BASE_DIR, '1_fit_singlespecies.csv')

results_df.to_csv(results_csv_path, index=False)

print("Analysis and plotting completed, results saved to CSV.")

# Sensitivity analysis plots
std_dev_defoliation = filtered_data['Defoliation'].std()
bias = [5, 10, 15, 20]

# Generate plots for each BC category and for All species data
for idx, category in enumerate(['All species'] + list(filtered_data['BC'].unique())):
    if category == 'All species':
        category_data = filtered_data
    else:
        category_data = filtered_data[filtered_data['BC'] == category]
    color = color_palette[idx]
    plot_jointplot_with_sensitivity(category_data, 'Defoliation', 'Growth', category, color, f'Defoliation vs Growth Analysis for BC {category}', std_dev_defoliation, bias, filename=f'BC_{category}_sensitivity.png')

print("Sensitivity analysis plots completed.")

BASE_DIR = 'C:/Growth/'

results_csv_path = os.path.join(BASE_DIR, '1_fit_singlespecies.csv')

file_path_allsp = os.path.join(BASE_DIR, 'allsp(8)age.txt')
file_path_fit = os.path.join(BASE_DIR, '1_fit_singlespecies.csv')

data1 = pd.read_csv(file_path_allsp, sep='\t', encoding='gbk')
data6 = pd.read_csv(file_path_fit, header=0)
print(data6.columns)
print(data1.head())
if 'grp_tree_species' in data1.columns:
    unique_categories = data1['grp_tree_species'].unique()
    print("Unique categories in 'grp_tree_species':")
    for category in unique_categories:
        print(category)
else:
    print("'grp_tree_species' column not found in the data.")

data1 = pd.merge(data1, data6[['grp_tree_species', 'code_defoliation', 'Mean Growth', 'Min Growth', 'Max Growth']], on=['grp_tree_species', 'code_defoliation'], how='left')

missing_values = data1['Mean Growth'].isnull().sum()
if missing_values > 0:
    print(f"There are {missing_values} missing values in the 'Linear Fit of Concatenated Data' column of data1.")
else:
    print("There are no missing values in the 'Linear Fit of Concatenated Data' column of data1.")
print(f"Total number of rows in the dataframe after merging: {len(data1)}")

output_file_path = os.path.join(BASE_DIR, 'allspage(GroBC).txt')

data1.to_csv(output_file_path, sep='\t', encoding='gbk')

if 'grp_tree_species' in data1.columns:
    unique_grp_tree_species = data1['grp_tree_species'].unique()
    print("Unique values in 'grp_tree_species':")
    for name in unique_grp_tree_species:
        print(name)
else:
    print("'grp_tree_species' column not found in the data.")

if 'grp_tree_species' in data1.columns and 'Mean Growth' in data1.columns:
    # Filter data where Mean Growth is not null
    filtered_data = data1[data1['Mean Growth'].notnull()]
    unique_grp_tree_species_with_growth = filtered_data['grp_tree_species'].unique()

    print("Unique values in 'grp_tree_species' with a 'Mean Growth' value:")
    for name in unique_grp_tree_species_with_growth:
        print(name)

    print(f"\nTotal number of unique 'grp_tree_species' with 'Mean Growth': {len(unique_grp_tree_species_with_growth)}")
else:
    print("Required columns not found in the data.")


def create_combined_plot(data, time_periods, colors, bin_edges, labels, directory_path, group_name):
    fig, ax = plt.subplots(figsize=(3, 3.5))
    cumulative_results = []  # This will store the detailed results

    bar_values = []  # This will store the values for the bar plot

    for idx, (start, end) in enumerate(time_periods):
        subset_data = data[(data['survey_year'] >= start) & (data['survey_year'] <= end)]
        subset_data['binned'] = pd.cut(subset_data['code_defoliation'], bins=bin_edges, labels=labels, right=False)

        count_data = subset_data['binned'].value_counts().sort_index().reindex(labels, fill_value=0)
        freq_data = (count_data / len(subset_data)) * 100

        # Calculate cumulative growth
        aggregated_data = subset_data.groupby('binned').agg({
            'Adjusted Mean Growth': 'mean',
            'Adjusted Min Growth': 'mean',
            'Adjusted Max Growth': 'mean'
        }).reindex(labels).fillna(0)

        multiplied_mean = aggregated_data['Adjusted Mean Growth'] * freq_data
        cumulative_mean = multiplied_mean.cumsum() / 100
        multiplied_min = aggregated_data['Adjusted Min Growth'] * freq_data
        cumulative_min = multiplied_min.cumsum() / 100
        multiplied_max = aggregated_data['Adjusted Max Growth'] * freq_data
        cumulative_max = multiplied_max.cumsum() / 100

        # Store detailed results
        cumulative_results.append({
            'Group': group_name,
            'Time Period': f"{start}-{end}",
            'Mean Cumulative Value': cumulative_mean.iloc[-1],
            'Min Cumulative Value': cumulative_min.iloc[-1],
            'Max Cumulative Value': cumulative_max.iloc[-1]
        })

        ax.plot(freq_data.index, cumulative_mean, '--', color=colors[idx], linewidth=2)
        max_cumulative_value = cumulative_mean.max()
        max_cumulative_index = cumulative_mean.idxmax()
        vertical_offsets = [-8, -16, -24, -32]
        horizontal_offsets = [-24, -25, -26, -27]

        ax.annotate(f"{max_cumulative_value:.2f}",
                     xy=(max_cumulative_index, max_cumulative_value),
                     xytext=(
                         max_cumulative_index + horizontal_offsets[idx], max_cumulative_value + vertical_offsets[idx]),
                     arrowprops=dict(facecolor=colors[idx], edgecolor='none', shrink=0.05, width=0.5,
                                     headwidth=3),
                     color=colors[idx], fontweight='bold')

        ax.fill_between(freq_data.index, cumulative_min, cumulative_max, color=colors[idx], alpha=0.1)
        # Store mean value for the bar plot
        bar_values.append(cumulative_mean.iloc[-1])

    # Insertion of the bar plot
    inset_ax = inset_axes(ax, width="40%", height="30%", loc='lower right')
    inset_ax.bar(range(len(time_periods)), bar_values, color=colors)
    inset_ax.set_yticks(np.arange(0, 101, 25))
    for label in inset_ax.get_yticklabels():
        label.set_fontsize(6)  # Set font size
        label.set_family('Arial')
    inset_ax.set_xticks([])  # Remove x-axis ticks
    inset_ax.spines['top'].set_visible(False)  # Hide the top spine
    inset_ax.spines['right'].set_visible(False)  # Hide the right spine
    inset_ax.spines['bottom'].set_visible(False)  # Hide the bottom spine
    inset_ax.spines['left'].set_visible(False)  # Hide the left spine

    # Plot settings
    ax.set_ylabel("Cumulative relative growth %")
    ax.set_xlabel("Defoliation %")
    ax.set_xticks([0, 20, 40, 60, 80, 100])
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)

    # Save plot
    plt.tight_layout()
    # output_filename = f"{group_name}_BC.png"
    # plt.savefig(os.path.join(directory_path, output_filename), dpi=300, format='png')
    # plt.close()

    return cumulative_results

def process_and_save_file_with_combined_plot(file_path, time_periods, colors, directory_path):
    cumulative_data_all = []
    bin_edges = [0] + list(np.arange(5, 100, 5)) + [99, 100, 101]  # Left closed and right open interval
    labels = bin_edges[:-1]

    print(f"Processing: {file_path}")
    data = pd.read_csv(file_path, sep='\t', encoding='gbk', low_memory=False)

    data['Adjusted Mean Growth'] = data['Mean Growth'] + 100
    data['Adjusted Min Growth'] = data['Min Growth'] + 100
    data['Adjusted Max Growth'] = data['Max Growth'] + 100
    cumulative_data = create_combined_plot(data, time_periods, colors, bin_edges, labels, directory_path, 'All_Data')
    cumulative_data_all.extend(cumulative_data)

    # Process entire dataset and groupings
    for group_field in ['grp_tree_species', 'biogeo_reg']:
        for group_value, group_data in data.groupby(group_field):
            group_name = f"{group_value}"
            cumulative_data = create_combined_plot(group_data, time_periods, colors, bin_edges, labels, directory_path, group_name)
            cumulative_data_all.extend(cumulative_data)
    for biogeo_value, biogeo_group in data.groupby('biogeo_reg'):
        for grp_value, grp_data in biogeo_group.groupby('grp_tree_species'):
            group_name = f"{biogeo_value}_{grp_value}"
            cumulative_data = create_combined_plot(grp_data, time_periods, colors, bin_edges, labels,
                                                   directory_path, group_name)
            cumulative_data_all.extend(cumulative_data)

    # Save cumulative data to CSV
    cumulative_values_df = pd.DataFrame(cumulative_data_all)
    cumulative_values_df.to_csv(os.path.join(directory_path, 'cum_growth_BC.csv'), index=False)

# Set parameters and call the function
directory_path = 'C:/GrowthReduction/'
file_path = os.path.join(directory_path, 'allspage(GroBC).txt')

time_periods = [(1990, 1999), (2000, 2009), (2010, 2019), (2020, 2022)]
colors = ['#98DCED', '#A0BEF5', '#D88BF2', '#9D2BC2']

process_and_save_file_with_combined_plot(file_path, time_periods, colors, directory_path)

# Load the CSV file
BASE_DIR = 'C:/GrowthReduction'
file_path = os.path.join(BASE_DIR, 'cum_growth_BC.csv')
df = pd.read_csv(file_path)

# Specific regions list
specific_regions = ["Alpine", "Atlantic", "Boreal", "Continental", "Mediterranean", "Pannonian"]

# Function to split the group column
def split_group(group):
    parts = group.split('_')
    if len(parts) == 2:
        return parts[0], parts[1]
    else:
        return (group if group in specific_regions else "All"), (group if group in ["All_Data", "broadleaves", "conifers"] else "All")

# Apply the function to create new columns
df[['Region', 'BC']] = df['Group'].apply(lambda x: pd.Series(split_group(x)))

# Save the modified dataframe to a new CSV file
output_path = os.path.join(BASE_DIR, 'cum_growth_BC_drawm.csv')
df.to_csv(output_path, index=False)

print(f"Modified CSV saved to {output_path}")

plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 14

BASE_DIR = 'C:/GrowthReduction'
output_path = os.path.join(BASE_DIR, 'cum_growth_BC_drawm.csv')
df = pd.read_csv(file_path)
categories = df['BC'].unique()
years = df['Time Period'].unique()
colors = ['#98DCED', '#A0BEF5', '#D88BF2', '#9D2BC2']
colors1 = ['#86C2D1', '#8DA7D8', '#BE7AD5', '#8A26AB']
output_dir = 'C:/relative_growth_BC(right)'
os.makedirs(output_dir, exist_ok=True)

for category in categories:
    data = {}
    category_df = df[df['BC'] == category]
    regions = category_df['Region'].unique()

    for region in regions:
        region_data = []
        for year in years:
            subset = category_df[(category_df['Region'] == region) & (category_df['Time Period'] == year)]
            if not subset.empty:
                min_val = subset['Min Cumulative Value'].values[0]
                mean_val = subset['Mean Cumulative Value'].values[0]
                max_val = subset['Max Cumulative Value'].values[0]
                region_data.append([min_val, mean_val, max_val])
        data[region] = region_data

    all_regions_1990_1999 = [data[region][0] for region in regions if len(data[region]) > 0]
    all_min_mean = np.mean([v[0] for v in all_regions_1990_1999])
    all_mean_mean = np.mean([v[1] for v in all_regions_1990_1999])
    all_max_mean = np.mean([v[2] for v in all_regions_1990_1999])

    fig, ax = plt.subplots(figsize=(5, 8))

    bar_height = 0.15
    spacing = 0.06
    index = np.arange(len(regions))
    bar_positions = index * (len(years) * (bar_height + spacing) + spacing * 4)

    # Draw the gray area to represent the min-max value of the full region data from 1990 to 1999 and the white line to represent the mean
    ax.fill_betweenx(
        [-0.2, len(regions) + 2 * bar_height],
        all_min_mean,
        all_max_mean,
        color="#DCDCDC",
        alpha=0.5,
    )
    ax.plot(
        [all_mean_mean, all_mean_mean],
        [-0.2, len(regions) + 2 * bar_height],
        color="white",
        linewidth=2
    )
    for i, year in enumerate(years):
        for j, region in enumerate(regions):
            if region in data and i < len(data[region]):
                values = data[region][i]
                # Use the mean to draw the box, with width from minimum to maximum, increase transparency and darken the border
                ax.barh(
                    bar_positions[j] + i * (bar_height + spacing),
                    values[2] - values[0],
                    height=bar_height,
                    color=colors[i % len(colors)],
                    alpha=0.9,
                    edgecolor=colors1[i % len(colors)],
                    linewidth=2,
                    left=values[0],
                    label=year if j == 0 else "",
                )

                ax.plot(
                    [values[1], values[1]],
                    [bar_positions[j] + i * (bar_height + spacing) - bar_height / 2,
                     bar_positions[j] + i * (bar_height + spacing) + bar_height / 2],
                    color=colors1[i % len(colors)],
                    linewidth=2
                )

    ax.set_yticks(bar_positions + (bar_height * (len(years) - 1.5) + spacing * (len(years) - 1)) / 2)
    ax.set_ylim(-0.12, (len(regions) + 1.05) * (len(years) * (bar_height + spacing) + spacing))

    ax.set_xlabel("Relative growth, %")
    ax.set_xticks(np.arange(60, 101, 10))
    ax.set_xlim(60, 100)
    ax.set_title(f"{category}", fontsize=18)

    ax.set_yticks(bar_positions + len(years) * (bar_height + spacing) / 2)
    ax.set_yticklabels(regions, ha='left')
    ax.yaxis.tick_right()
    ax.yaxis.set_label_position('right')

    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.grid(False)
    plt.tight_layout()

    # save plot
    output_path = os.path.join(output_dir, f"relative_growth_{category}_right.png")
    plt.savefig(output_path, dpi=300)

    # show plot
    plt.show()
    plt.close()

