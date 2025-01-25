"""""""""""""""""""""""""""""""""""
This script performs several tasks related to the analysis of defoliation, net change, and carbon sink data.

1.It starts by loading and merging multiple CSV files, then processes and cleans the data.
2.It calculates means and deviations from baselines, filters the data for specific years.
3.The script also includes various plotting routines to visualize the relationships between defoliation, net change, and carbon sink data.
4.It generates scatter plots, line plots, and bar plots with linear regression and confidence intervals, and applies polynomial fit for better visualization(3-years average).
"""""""""""""""""""""""""""""""""""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

data1 = pd.read_csv('C:/Yuman/Carbon/plot_year.csv', encoding='gbk')
data6 = pd.read_csv('C:/Yuman/Carbon/Carbon.csv', header=0)

# Check columns of data1
print("Columns in data1:", data1.columns)

# Group by 'lib_country' and 'survey_year' and calculate the mean of 'mean_defoliation' and 'deviation_all'
data1_mean_defoliation = data1.groupby(['lib_country', 'survey_year'])['mean_defoliation'].mean().reset_index()
data1_deviation_all = data1.groupby(['lib_country', 'survey_year'])['deviation_all'].mean().reset_index()

# Merge the grouped data
data1 = pd.merge(data1_mean_defoliation, data1_deviation_all, on=['lib_country', 'survey_year'])

# Print shape and number of rows of data1 after grouping
print("Shape of data1 after grouping:", data1.shape)
print("Number of rows in data1 after grouping:", len(data1))

# Merge data1 with data6 on 'lib_country' and 'survey_year', keeping all columns from data6
data1 = pd.merge(data6, data1, on=['lib_country', 'survey_year'], how='left')

# Print the shape and number of rows after merging
print("Shape of data1 after merging:", data1.shape)
print("Number of rows in data1 after merging:", len(data1))

# Print the first few rows of the merged data to check the result
print(data1.head())

# Filter data for years 1990-2021
data1 = data1[(data1['survey_year'] >= 1990) & (data1['survey_year'] <= 2021)]

# Save the merged data to a new CSV file
data1.to_csv('C:/Yuman/Carbon/Plot_year_Carbon.csv', encoding='gbk', index=False)

# Print shape and number of rows of data1 after merging
print("Shape of data1 after merging:", data1.shape)
print("Number of rows in data1 after merging:", len(data1))

# Check for unique combinations of 'lib_country'
country_plot_combinations = data1.groupby(['lib_country']).ngroups
print(f"Unique combinations of lib_country: {country_plot_combinations}")

# Load the merged data
data_path = 'C:/Yuman/Carbon/Plot_year_Carbon.csv'
data = pd.read_csv(data_path, low_memory=False)

# Rename columns for consistency
data.rename(columns={'survey_year': 'Year', 'Net change': 'Net Change', 'Carbon sink': 'CO2 Removals'}, inplace=True)
print("Columns after renaming:", data.columns)

# Calculate the mean Net Change and sum of CO2 Removals for each year
data_avg_sum = data.groupby('Year').agg({'Net Change': 'mean', 'CO2 Removals': 'sum'}).reset_index()
data_avg_sum['CO2 Removals'] = data_avg_sum['CO2 Removals'] / 1000000
print(data_avg_sum)

# Plot Net Change and CO2 Removals
fig, ax1 = plt.subplots()

color = 'tab:orange'
ax1.set_xlabel('Year')
ax1.set_ylabel('Net Change in Carbon Stocks, tC ha$^{-1}$', color='black', fontsize=16)
ax1.plot(data_avg_sum['Year'], data_avg_sum['Net Change'], color=color, linewidth=2)
ax1.tick_params(axis='y', labelcolor='black', direction='out')
ax1.tick_params(axis='x', direction='out')
ax1.set_xlim(1989.5, 2021.5)
ax1.set_xticks([1990, 1995, 2000, 2005, 2010, 2015, 2020])
ax1.set_yticks([-1.0, -0.8, -0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8, 1.0])

ax1.annotate('Net Change (Living tree biomass)', xy=(0.02, 0.95), xycoords='axes fraction', fontsize=16, color='tab:orange', ha='left')

ax2 = ax1.twinx()
color = 'tab:blue'
ax2.set_ylabel('Total Carbon Balance, Mt CO$_2$', color='black', fontsize=16)
ax2.bar(data_avg_sum['Year'], data_avg_sum['CO2 Removals'], color=color)
ax2.tick_params(axis='y', labelcolor='black', direction='out')
ax1.axhline(y=0, color='black')

ax2.annotate('Total Carbon Balance (incl. Soil)', xy=(0.98, 0.02), xycoords='axes fraction', fontsize=16, color='tab:blue', ha='right')

ax2.set_ylim(-500, 500)
ax2.set_yticks([-500, -400, -300, -200, -100, 0, 100, 200, 300, 400, 500])
ax2.set_xlim(1989.5, 2021.5)

fig.tight_layout()

# Save the plot
output_path = 'C:/Yuman/Carbon/2_1Net_Change_CO2_Removals_Plot.png'
plt.savefig(output_path, dpi=300)

plt.show()
#-----------------------------------------------------
file_path = 'C:/Yuman/Carbon/Plot_year_Carbon.csv'
df = pd.read_csv(file_path)
print("Original DataFrame:")
print(df.head())

# Ensure numerical columns are properly typed, converting if necessary
numeric_columns = ['mean_defoliation', 'deviation_all', 'Grains', 'Losses', 'Net change', 'Carbon sink']
for col in numeric_columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')
print("\nDataFrame after converting to numeric types:")
print(df.head())

# Calculate yearly means for specified numerical fields
df_means = df.groupby('survey_year').agg({
    'mean_defoliation': 'mean',
    'deviation_all': 'mean',
    'Grains': 'mean',
    'Losses': 'mean',
    'Net change': 'mean',
    'Carbon sink': 'mean',
}).reset_index()
print("\nYearly means DataFrame:")
print(df_means.head())

df_meansq = df.groupby(['survey_year', 'lib_country']).agg({
    'Carbon sink': 'mean',
}).reset_index()
print("\nYearly means by country DataFrame:")
print(df_meansq.head())

# Filter data for years 1990-2019
df_filtered = df[(df['survey_year'] >= 1990) & (df['survey_year'] <= 2019)]
print("\nFiltered DataFrame (1990-2019):")
print(df_filtered.head())

# Calculate baseline for each country based on 1990-2019 average Carbon sink
baseline = df_filtered.groupby('lib_country')['Carbon sink'].mean().reset_index()
baseline.columns = ['lib_country', 'Carbon sink_baseline']
print("\nBaseline DataFrame:")
print(baseline.to_string())

# Merge baseline data into df_meansq
df_meansq = df_meansq.merge(baseline, on='lib_country', how='left')
print("\nYearly means by country with baseline DataFrame:")
print(df_meansq.head())

# Calculate Carbon sink deviation from baseline
df_meansq['Carbon sink_dev'] = ((df_meansq['Carbon sink'] - df_meansq['Carbon sink_baseline']) /
                                df_meansq['Carbon sink_baseline'] * (-100))
print("\nYearly means by country with Carbon sink deviation DataFrame:")
print(df_meansq.head())

# Calculate yearly mean deviation for all countries
yearly_dev_means = df_meansq.groupby('survey_year')['Carbon sink_dev'].mean().reset_index()
yearly_dev_means.columns = ['survey_year', 'Carbon sinkctrybase%']
print("\nYearly Carbon sink deviation means DataFrame:")
print(yearly_dev_means.to_string())

# Merge Carbon sink deviation means into df_means
df_means_extended = df_means.merge(yearly_dev_means, on='survey_year', how='left')
print("\nExtended yearly means DataFrame with Carbon sink deviation means:")
print(df_means_extended.head())

# Filter extended data for years 1990-2019
df_means_filtered = df_means_extended[(df_means_extended['survey_year'] >= 1990) & (df_means_extended['survey_year'] <= 2019)]

# Calculate means and sum for the period 1990-2019
mean_values = df_means_filtered.mean()
sum_value_carbon_sink = df_means_filtered['Carbon sink'].sum() / 1000000

# Store means and sum in a new DataFrame
mean_values_df = pd.DataFrame(mean_values).transpose()
mean_values_df['Cabon sink_sum_million'] = sum_value_carbon_sink
mean_values_df['survey_year'] = '1990-2019'

# Append means to df_means
df_means_extended = pd.concat([df_means_extended, mean_values_df], ignore_index=True)

# Calculate deviations from baseline for specified fields
for column in ['mean_defoliation', 'deviation_all']:
    baseline_value = mean_values[column]
    df_means_extended[f'{column}_base'] = df_means_extended[column] - baseline_value

for column in ['Grains', 'Losses', 'Net change', 'Carbon sink']:
    baseline_value = mean_values[column]
    df_means_extended[f'{column}_base%'] = (df_means_extended[column] - baseline_value) / abs(baseline_value) * 100

# Define output file path
means_output_file_path = r'C:\Yuman\Carbon\yearly19_base%.csv'

# Save extended data to a new CSV file
df_means_extended.to_csv(means_output_file_path, index=False)
print(f"\nYearly means with baseline data saved to {means_output_file_path}")

# Load CSV file for plotting
data = pd.read_csv(means_output_file_path)

# Extract necessary fields for plotting
data = data[['survey_year', 'Net change_base%', 'mean_defoliation_base', 'Net change']]
data.replace([np.inf, -np.inf], np.nan, inplace=True)
data.dropna(subset=['Net change_base%', 'mean_defoliation_base', 'Net change'], inplace=True)

# Define colors for different periods
colors = {
    (1990, 1999): '#70AD47',
    (2000, 2009): '#5B9BD5',
    (2010, 2019): '#FFC000',
    (2020, 2021): '#C00000'
}

# Add color column based on survey year
def get_color(year):
    for (start, end), color in colors.items():
        if start <= year <= end:
            return color
    return 'black'

data['color'] = data['survey_year'].apply(get_color)

# Determine Net change range
min_net_change = np.floor(data['Net change'].min() * 10) / 10
max_net_change = np.ceil(data['Net change'].max() * 10) / 10

# Define size ranges for scatter plot
size_ranges = [(0.4, 0.6), (0.6, 0.8), (0.8, 1.0)]
size_factors = {range_tuple: 100 * (idx + 1) for idx, range_tuple in enumerate(size_ranges)}

# Adjust size factors for clear distinction
size_factors[size_ranges[-1]] = 150
size_factors[size_ranges[-2]] = 80
size_factors[size_ranges[-3]] = 30

# Create scatter plot
plt.figure(figsize=(10, 8))
for period, color in colors.items():
    subset = data[(data['survey_year'] >= period[0]) & (data['survey_year'] <= period[1])]
    sizes = []
    for value in subset['Net change']:
        for range_tuple, factor in size_factors.items():
            if range_tuple[0] <= value < range_tuple[1]:
                sizes.append(factor)
                break
    plt.scatter(subset['mean_defoliation_base'], subset['Net change_base%'], c=color, s=sizes)

# Linear fit for 1990-2021
X = data['mean_defoliation_base'].values.reshape(-1, 1)
y = data['Net change_base%']
reg = LinearRegression().fit(X, y)
y_pred = reg.predict(X)

# Calculate confidence interval
X_with_const = sm.add_constant(X)
model = sm.OLS(y, X_with_const)
results = model.fit()
predictions = results.get_prediction(X_with_const)
frame = predictions.summary_frame(alpha=0.05)
conf_int_lower = frame['mean_ci_lower']
conf_int_upper = frame['mean_ci_upper']

# Plot regression line and confidence interval for 1990-2021
plt.plot(data['mean_defoliation_base'], y_pred, linestyle='-', color='gray', label='_nolegend_')
plt.fill_between(data['mean_defoliation_base'], conf_int_lower, conf_int_upper, color='gray', alpha=0.2)

# Linear fit for 2010-2021
data_2010_2021 = data[(data['survey_year'] >= 2010) & (data['survey_year'] <= 2021)]
X_2010_2021 = data_2010_2021['mean_defoliation_base'].values.reshape(-1, 1)
y_2010_2021 = data_2010_2021['Net change_base%']
reg_2010_2021 = LinearRegression().fit(X_2010_2021, y_2010_2021)
y_pred_2010_2021 = reg_2010_2021.predict(X_2010_2021)

# Ensure sorted data
sorted_idx_2010_2021 = np.argsort(data_2010_2021['mean_defoliation_base'])
sorted_x_2010_2021 = data_2010_2021['mean_defoliation_base'].values[sorted_idx_2010_2021]
sorted_y_pred_2010_2021 = y_pred_2010_2021[sorted_idx_2010_2021]

# Calculate confidence interval for 2010-2021
X_with_const_2010_2021 = sm.add_constant(X_2010_2021)
model_2010_2021 = sm.OLS(y_2010_2021, X_with_const_2010_2021)
results_2010_2021 = model_2010_2021.fit()
predictions_2010_2021 = results_2010_2021.get_prediction(X_with_const_2010_2021)
frame_2010_2021 = predictions_2010_2021.summary_frame(alpha=0.05)
conf_int_lower_2010_2021 = frame_2010_2021['mean_ci_lower']
conf_int_upper_2010_2021 = frame_2010_2021['mean_ci_upper']

# Ensure sorted confidence interval data
sorted_conf_int_lower_2010_2021 = conf_int_lower_2010_2021.values[sorted_idx_2010_2021]
sorted_conf_int_upper_2010_2021 = conf_int_upper_2010_2021.values[sorted_idx_2010_2021]

# Plot regression line and confidence interval for 2010-2021
plt.plot(sorted_x_2010_2021, sorted_y_pred_2010_2021, linestyle='-', color='#88419d', label='_nolegend_')
plt.fill_between(sorted_x_2010_2021, sorted_conf_int_lower_2010_2021, sorted_conf_int_upper_2010_2021, color='#88419d', alpha=0.2)

# Set axis ranges
plt.xlim(-5, 5)
plt.ylim(-50, 50)
plt.xticks(np.arange(-5, 5.00001, 1))
plt.yticks(np.arange(-50, 51, 10))

# Significance levels for 1990-2021
p_value = results.pvalues.iloc[1]
if p_value < 0.001:
    significance = '***'
elif p_value < 0.01:
    significance = '**'
elif p_value < 0.05:
    significance = '*'
else:
    significance = ''

# Display linear fit formula, R² value, and significance level for 1990-2021
coef = reg.coef_[0]
intercept = reg.intercept_
sign = '+' if intercept >= 0 else '-'
plt.text(0.45, 0.95, f'1990-2021: y = {coef:.2f}x {sign} {abs(intercept):.2f}{significance}  R² = {reg.score(X, y):.2f}', transform=plt.gca().transAxes, fontsize=14, verticalalignment='bottom')

# Significance levels for 2010-2021
p_value_2010_2021 = results_2010_2021.pvalues.iloc[1]
if p_value_2010_2021 < 0.001:
    significance_2010_2021 = '***'
elif p_value_2010_2021 < 0.01:
    significance_2010_2021 = '**'
elif p_value_2010_2021 < 0.05:
    significance_2010_2021 = '*'
else:
    significance_2010_2021 = ''

# Display linear fit formula, R² value, and significance level for 2010-2021
coef_2010_2021 = reg_2010_2021.coef_[0]
intercept_2010_2021 = reg_2010_2021.intercept_
sign_2010_2021 = '+' if intercept_2010_2021 >= 0 else '-'
plt.text(0.45, 0.90, f'2010-2021: y = {coef_2010_2021:.2f}x {sign_2010_2021} {abs(intercept_2010_2021):.2f}{significance_2010_2021}  R² = {reg_2010_2021.score(X_2010_2021, y_2010_2021):.2f}', transform=plt.gca().transAxes, fontsize=14, verticalalignment='bottom', color='#88419d')

# Add title and labels
plt.xlabel('Defoliation, % deviation from 1990-2019 LTA')
plt.ylabel('Net Change and Total Carbon Balance, \n% deviation from 1990-2019 LTA')

# Create inset legend in the lower left corner
ax_inset = inset_axes(plt.gca(), width="40%", height="30%", loc='lower left', borderpad=1)
ax_inset.axis('off')

# Add legend title
ax_inset.text(0.75, len(colors) + 1.2, 'Net Change, t ha$^{-1}$', horizontalalignment='left', fontsize=14)

# Add size ranges
for i, size_range in enumerate(size_ranges):
    x = i + 0.85
    ax_inset.text(x, len(colors) + 0.5, f'{size_range[0]}-{size_range[1]}', horizontalalignment='center', fontsize=16)

# Add legend content
for i, (period, color) in enumerate(colors.items()):
    y = len(colors) - i  # Arrange from top to bottom
    ax_inset.text(-1, y, f'{period[0]}-{period[1]}', verticalalignment='center', fontsize=16, color=color)
    for j, (size_range, factor) in enumerate(size_factors.items()):
        x = j + 0.85  # Arrange from left to right
        ax_inset.scatter(x, y, s=factor, color=color)

# Set legend range and labels
ax_inset.set_xlim(-1, len(size_factors) + 0.5)
ax_inset.set_ylim(0, len(colors) + 1)
ax_inset.set_xticks(np.arange(len(size_factors)) + 0.85)
ax_inset.set_xticklabels([])
ax_inset.set_yticks([])

# Save plot
output_path = 'C:/Yuman/Carbon/2_4net_change_and_defdev_combined_legend.png'
plt.savefig(output_path, bbox_inches='tight')

# Show plot
plt.show()
#-----------------------------------------
# Load CSV file for plotting
data = pd.read_csv(means_output_file_path)

# Extract necessary fields for plotting
data = data[['survey_year', 'Carbon sinkctrybase%', 'Net change_base%']]
data.replace([np.inf, -np.inf], np.nan, inplace=True)
data.dropna(subset=['Carbon sinkctrybase%', 'Net change_base%'], inplace=True)

# Add color column based on survey year
data['color'] = data['survey_year'].apply(get_color)

# Create scatter plot
plt.figure(figsize=(7, 6))
for period, color in colors.items():
    subset = data[(data['survey_year'] >= period[0]) & (data['survey_year'] <= period[1])]
    plt.scatter(subset['Net change_base%'], subset['Carbon sinkctrybase%'], c=color, label=f'{period[0]}-{period[1]}')

# Linear fit
X = data['Net change_base%'].values.reshape(-1, 1)
y = data['Carbon sinkctrybase%']
reg = LinearRegression().fit(X, y)
y_pred = reg.predict(X)

# Calculate confidence interval
X_with_const = sm.add_constant(X)
model = sm.OLS(y, X_with_const)
results = model.fit()
predictions = results.get_prediction(X_with_const)
frame = predictions.summary_frame(alpha=0.05)
lower = frame['mean_ci_lower']
upper = frame['mean_ci_upper']

# Ensure sorted data
sorted_idx = np.argsort(data['Net change_base%'])
sorted_x = data['Net change_base%'].values[sorted_idx]
sorted_lower = lower.values[sorted_idx]
sorted_upper = upper.values[sorted_idx]

# Plot regression line and confidence interval
plt.plot(sorted_x, y_pred[sorted_idx], linestyle='-', color='black')
plt.fill_between(sorted_x, sorted_lower, sorted_upper, color='gray', alpha=0.2)

# Significance levels
p_value = results.pvalues[1]
if p_value < 0.001:
    significance = '***'
elif p_value < 0.01:
    significance = '**'
elif p_value < 0.05:
    significance = '*'
else:
    significance = ''

# Display linear fit formula, R² value, and significance level
coef = reg.coef_[0]
intercept = reg.intercept_
r_squared = reg.score(X, y)
plt.text(0.05, 0.05, f'y = {coef:.2f}x + {intercept:.2f}{significance}\nR² = {r_squared:.2f}', transform=plt.gca().transAxes, fontsize=16, verticalalignment='bottom')

# Add title and labels
plt.xlabel('Net Change ha$^{-1}$, % deviation from 1990-2019')
plt.ylabel('Total Carbon Balance, % deviation from 1990-2019')

# Customize legend style
legend = plt.legend(frameon=False)
for handle in legend.legendHandles:
    handle.set_sizes([30])

# Remove grid lines
plt.grid(False)

# Save plot
output_path = 'C:/Yuman/Carbon/2_3net_change_and_total_sink.png'
plt.savefig(output_path, bbox_inches='tight')

# Show plot
plt.show()
#-----------------------------------
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import t
from sklearn.metrics import r2_score
from matplotlib import font_manager

# Set font properties
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 18

# Read CSV file
file_path = 'C:/YumanDG/9_Carbon/yearly19_baser%.csv'
data = pd.read_csv(file_path)

# Calculate 3-year moving averages
data['Net change_base%_MA'] = data['Net change_base%'].rolling(window=3).mean()
data['mean_defoliation_base_MA'] = data['mean_defoliation_base'].rolling(window=3).mean()
data['Carbon sinkctrybase%_MA'] = data['Carbon sinkctrybase%'].rolling(window=3).mean()

# Function to calculate confidence intervals for polynomial fit
def calculate_polyfit_confidence(x, y, degree=2, confidence=0.95):
    # Remove missing values
    mask = ~np.isnan(y)
    x = x[mask]
    y = y[mask]

    # Fit polynomial
    coeffs = np.polyfit(x, y, degree)
    p = np.poly1d(coeffs)

    # Predicted values
    y_fit = p(x)

    # Calculate residuals and Residual Sum of Squares (RSS)
    residuals = y - y_fit
    rss = np.sum(residuals ** 2)
    n = len(x)

    # Adjusted standard error
    se = np.sqrt(rss / (n - degree - 1))

    # Construct polynomial basis matrix
    X = np.vander(x, degree + 1)
    xtx_inv = np.linalg.inv(X.T @ X)  # Inverse of (X^T * X)

    # Confidence interval calculation
    t_value = t.ppf((1 + confidence) / 2, df=n - degree - 1)
    ci = t_value * se * np.sqrt(np.diag(X @ xtx_inv @ X.T))

    return y_fit, y_fit - ci, y_fit + ci, residuals, p

# Prepare data
data = data.dropna(subset=['survey_year', 'Net change_base%', 'mean_defoliation_base', 'Carbon sinkctrybase%'])
x = data['survey_year']

# Net change_base% y and moving average
y_net_change = data['Net change_base%']
y_net_change_ma = data['Net change_base%_MA'].dropna()
x_net_change_ma = x.iloc[y_net_change_ma.index]
net_change_fit, net_change_ci_lower, net_change_ci_upper, net_change_residuals, net_change_poly = calculate_polyfit_confidence(
    x_net_change_ma, y_net_change_ma)
net_change_r2 = r2_score(y_net_change_ma, net_change_fit)
net_change_std = np.std(net_change_residuals)

# Carbon sinkctrybase% y and moving average
y_carbon_sink = data['Carbon sinkctrybase%']
y_carbon_sink_ma = data['Carbon sinkctrybase%_MA'].dropna()
x_carbon_sink_ma = x.iloc[y_carbon_sink_ma.index]
carbon_sink_fit, carbon_sink_ci_lower, carbon_sink_ci_upper, carbon_sink_residuals, carbon_sink_poly = calculate_polyfit_confidence(
    x_carbon_sink_ma, y_carbon_sink_ma)
carbon_sink_r2 = r2_score(y_carbon_sink_ma, carbon_sink_fit)
carbon_sink_std = np.std(carbon_sink_residuals)

# mean_defoliation_base% y and moving average
y_defoliation = data['mean_defoliation_base']
y_defoliation_ma = data['mean_defoliation_base_MA'].dropna()
x_defoliation_ma = x.iloc[y_defoliation_ma.index]
defoliation_fit, defoliation_ci_lower, defoliation_ci_upper, defoliation_residuals, defoliation_poly = calculate_polyfit_confidence(
    x_defoliation_ma, y_defoliation_ma)
defoliation_r2 = r2_score(y_defoliation_ma, defoliation_fit)
defoliation_std = np.std(defoliation_residuals)

# Adjust figure
fig, ax1 = plt.subplots(figsize=(7, 6))

# Plot for Net change_base%
ax1.set_xlabel('Year')
ax1.set_ylabel('Net Change and Total Carbon Balance, \n% deviation from 1990-2019 LTA')
ax1.plot(x_net_change_ma, net_change_fit, color='#82d0ff', linewidth=3,
         label=f'Net change: $R^2$={net_change_r2:.2f}, Std={net_change_std:.2f}')
ax1.fill_between(x_net_change_ma, net_change_ci_lower, net_change_ci_upper, color='#82d0ff', alpha=0.2)
ax1.scatter(x, y_net_change, color='#82d0ff', s=30, marker='o', alpha=0.3, edgecolors='none')

# Plot for Carbon sinkctrybase%
ax1.plot(x_carbon_sink_ma, carbon_sink_fit, color='limegreen', linewidth=3,
         label=f'Carbon sink: $R^2$={carbon_sink_r2:.2f}, Std={carbon_sink_std:.2f}')
ax1.fill_between(x_carbon_sink_ma, carbon_sink_ci_lower, carbon_sink_ci_upper, color='limegreen', alpha=0.2)
ax1.scatter(x, y_carbon_sink, color='limegreen', s=30, marker='o', alpha=0.3, edgecolors='none')

# mean_defoliation_base% on secondary y-axis
ax2 = ax1.twinx()
ax2.set_ylabel('Defoliation, % deviation from 1990-2019 LTA', fontsize=18)
ax2.plot(x_defoliation_ma, defoliation_fit, color='#c00000', linewidth=3,
         label=f'Defoliation: $R^2$={defoliation_r2:.2f}, Std={defoliation_std:.2f}')
ax2.fill_between(x_defoliation_ma, defoliation_ci_lower, defoliation_ci_upper, color='#c00000', alpha=0.2)
ax2.scatter(x, y_defoliation, color='#c00000', s=30, marker='o', alpha=0.3, edgecolors='none')

# Add horizontal line at y=0
ax1.axhline(y=0, color='black', linestyle='--', linewidth=1)

# Adjust x-axis range and ticks
ax1.set_xlim([1989.5, 2021.5])
ax1.set_xticks([1990, 1995, 2000, 2005, 2010, 2015, 2020])
ax1.set_ylim([-50, 50])
ax1.set_yticks(range(-50, 51, 10))

# Adjust secondary y-axis range and ticks
ax2.set_ylim([-5, 5])
ax2.set_yticks(range(-5, 6, 1))

# Set tick direction and length
ax1.tick_params(direction='out', length=6)
ax2.tick_params(direction='out', length=6)

# Add legend with font size 16
fig.legend(loc='upper left', bbox_to_anchor=(0, 1), bbox_transform=ax1.transAxes, frameon=False, prop={'size': 16})

# Adjust layout
fig.tight_layout()

# Save the figure
plt.savefig('C:/YumanDG/9_Carbon/3_4carbon_defoliation_plot_polyfit_with_ci_ma.png', dpi=300)

# Show the figure
plt.show()


