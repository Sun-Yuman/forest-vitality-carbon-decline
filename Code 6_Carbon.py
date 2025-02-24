"""""""""""""""""""""""""""""""""""
This script performs several tasks related to the analysis of defoliation, net change, and Net CO2 emissions data.

The Carbon related data are available at https://unfccc.int/ghg-inventories-annex-i-parties/2023. For further inquiries, please contact the corresponding author.
1.It computes annual averages and plots a dual-axis chart of net change (line) and CO₂ removals (bars).
2.It performs linear regression on carbon emissions versus net change, displaying confidence and prediction intervals with regression statistics.
3.It creates scatter plots with regression lines for different periods (1990–2021 and 2010–2021), including annotated regression details. 
4.It calculates 3-year moving averages, applies polynomial fitting with confidence intervals, and plots the trends for net change, CO₂ emissions, and defoliation on dual y-axes.
"""""""""""""""""""""""""""""""""""

import pandas as pd
import matplotlib.pyplot as plt
import os

# Define base directory
BASE_DIR = 'C:/Carbon/'
data_filename = 'Plot_year_Carbon.csv'
output_filename = '1_Net_Change_CO2_Removals.jpg'

data_path = os.path.join(BASE_DIR, data_filename)
output_path = os.path.join(BASE_DIR, output_filename)

# Set font style and size
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 18

# Load data
data = pd.read_csv(data_path, low_memory=False)

# Rename columns to match the required names
data.rename(columns={'survey_year': 'Year', 'Net change': 'Net Change', 'Net CO2 emissions/kt': 'CO2 Removals'}, inplace=True)
print("Renamed columns:", data.columns)

# Compute annual average Net Change and total CO2 Removals (convert to Mt CO2)
data_avg_sum = data.groupby('Year').agg({'Net Change': 'mean', 'CO2 Removals': 'sum'}).reset_index()
data_avg_sum['CO2 Removals'] = data_avg_sum['CO2 Removals'] / 1000  # Convert kt CO2 to Mt CO2
print(data_avg_sum)

# Create figure with specified size
fig, ax1 = plt.subplots(figsize=(7, 6))

# Plot Net Change as a line plot
color = 'tab:orange'
ax1.set_xlabel('Year')
ax1.set_ylabel(r'Net Change in C Stocks, tC ha$^{-1}$', color='black', fontsize=16)
ax1.plot(data_avg_sum['Year'], data_avg_sum['Net Change'], color=color, linewidth=2)
ax1.tick_params(axis='y', labelcolor='black', direction='out')
ax1.tick_params(axis='x', direction='out')
ax1.set_xlim(1989.5, 2021.5)
ax1.set_xticks([1990, 1995, 2000, 2005, 2010, 2015, 2020])

# Adjust left y-axis range to center at 0
ax1.set_yticks([-1.0, -0.8, -0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8, 1.0])

# Annotate Net Change label inside the plot
ax1.annotate(r'NetC$_{\text{biomass}}$ (living forest biomass)',
             xy=(0.02, 0.95), xycoords='axes fraction',
             fontsize=16, color=color, ha='left')

# Create secondary y-axis for CO2 Removals
ax2 = ax1.twinx()
color = 'tab:blue'
ax2.set_ylabel(r'Total C Balance, Mt CO$_2$', color='black', fontsize=16)
ax2.bar(data_avg_sum['Year'], data_avg_sum['CO2 Removals'], color=color, alpha=0.6)
ax2.tick_params(axis='y', labelcolor='black', direction='out')
ax1.axhline(y=0, color='black', linewidth=1)

# Annotate CO2 Removals label inside the plot
ax2.annotate(r'NetC$_{\text{land}}$ (including soil, deadwood, and litter)',
             xy=(0.98, 0.02), xycoords='axes fraction',
             fontsize=16, color=color, ha='right')

# Adjust right y-axis range to center at 0
ax2.set_ylim(-500, 500)
ax2.set_yticks([-500, -400, -300, -200, -100, 0, 100, 200, 300, 400, 500])
ax2.set_xlim(1989.5, 2021.5)

# Ensure layout is compact
fig.tight_layout()

# Save figure
plt.savefig(output_path, dpi=300, bbox_inches='tight')

# Show plot
plt.show()
####################################Carbon###################################################
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm

# Set font family and font size
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 18

# Define base directory
BASE_DIR = 'C:/Carbon/'

# Load CSV file
file_path = BASE_DIR + 'yearly19_base%.csv'
data = pd.read_csv(file_path)

# Extract required columns
data = data[['survey_year', 'Net CO2 emissions/kt_ctrybase%', 'Net change_base%']]

# Check for invalid values in the data
data.replace([np.inf, -np.inf], np.nan, inplace=True)
data.dropna(subset=['Net CO2 emissions/kt_ctrybase%', 'Net change_base%'], inplace=True)

# Define colors for different survey year periods
colors = {
    (1990, 1999): '#70AD47',
    (2000, 2009): '#5B9BD5',
    (2010, 2019): '#FFC000',
    (2020, 2021): '#C00000'
}

# Add color column based on survey_year
def get_color(year):
    for (start, end), color in colors.items():
        if start <= year <= end:
            return color
    return 'black'

data['color'] = data['survey_year'].apply(get_color)

# Perform linear regression
X = data['Net change_base%'].values.reshape(-1, 1)
y = data['Net CO2 emissions/kt_ctrybase%']
reg = LinearRegression().fit(X, y)
y_pred = reg.predict(X)

# Calculate confidence intervals and prediction intervals for the observations
X_with_const = sm.add_constant(X)
model = sm.OLS(y, X_with_const)
results = model.fit()
predictions = results.get_prediction(X_with_const)
frame = predictions.summary_frame(alpha=0.05)  # 95% confidence and prediction intervals

# Confidence intervals
lower_conf = frame['mean_ci_lower']
upper_conf = frame['mean_ci_upper']

# Prediction intervals
lower_pred = frame['obs_ci_lower']
upper_pred = frame['obs_ci_upper']

# Ensure data is sorted for plotting
sorted_idx = np.argsort(data['Net change_base%'])
sorted_x = data['Net change_base%'].values[sorted_idx]
sorted_y_pred = y_pred[sorted_idx]
sorted_lower_conf = lower_conf.values[sorted_idx]
sorted_upper_conf = upper_conf.values[sorted_idx]
sorted_lower_pred = lower_pred.values[sorted_idx]
sorted_upper_pred = upper_pred.values[sorted_idx]

# Plot the confidence interval graph
plt.figure(figsize=(7, 6))
for period, color in colors.items():
    subset = data[(data['survey_year'] >= period[0]) & (data['survey_year'] <= period[1])]
    plt.scatter(subset['Net change_base%'], subset['Net CO2 emissions/kt_ctrybase%'], c=color, label=f'{period[0]}-{period[1]}', s=100)
plt.plot(sorted_x, sorted_y_pred, linestyle='-', color='black')
plt.fill_between(sorted_x, sorted_lower_conf, sorted_upper_conf, color='grey', alpha=0.2)
plt.xlim(-50, 50)
plt.ylim(-50, 50)
plt.xticks(np.arange(-50, 51, 10))
plt.yticks(np.arange(-50, 51, 10))
plt.xlabel('$\delta$LTA$_{NetC\\_biomass}$, %', fontsize=20)
plt.ylabel('$\delta$LTA$_{NetC\\_land}$, %', fontsize=20)
plt.legend(frameon=False)
plt.grid(False)

# Add equation and statistical information
coef = results.params[1]
intercept = results.params[0]
r_squared = reg.score(X, y)
p_value = results.pvalues[1]
if p_value < 0.001:
    significance = '***'
elif p_value < 0.01:
    significance = '**'
elif p_value < 0.05:
    significance = '*'
else:
    significance = ''
plt.text(-45, -40, f'y = {coef:.2f}x + {intercept:.2f} {significance}\nR² = {r_squared:.2f}', fontsize=18)

plt.savefig(BASE_DIR + '2_Net change and Net CO2 emissions confidence_interval.jpg', bbox_inches='tight')
plt.show()

# Plot the prediction interval graph
plt.figure(figsize=(7, 6))
for period, color in colors.items():
    subset = data[(data['survey_year'] >= period[0]) & (data['survey_year'] <= period[1])]
    plt.scatter(subset['Net change_base%'], subset['Net CO2 emissions/kt_ctrybase%'], c=color, label=f'{period[0]}-{period[1]}')
plt.plot(sorted_x, sorted_y_pred, linestyle='-', color='black', label="Regression Line")
plt.fill_between(sorted_x, sorted_lower_pred, sorted_upper_pred, color='gray', alpha=0.2, label="Prediction Interval")
plt.xlim(-50, 50)
plt.ylim(-50, 50)
plt.xticks(np.arange(-50, 51, 10))
plt.yticks(np.arange(-50, 51, 10))
plt.xlabel('$\delta$LTA$_{NetC\\_biomass}$, %', fontsize=20)
plt.ylabel('$\delta$LTA$_{NetC\\_land}$, %', fontsize=20)
plt.legend(frameon=False)
plt.grid(False)

# Add equation and statistical information
plt.text(-40, 40, f'y = {coef:.2f}x + {intercept:.2f} {significance}\nR² = {r_squared:.2f}', fontsize=14, bbox=dict(facecolor='white', alpha=0.5))

plt.savefig(BASE_DIR + 'prediction_interval.png', bbox_inches='tight')
plt.show()

# Print the 95% confidence intervals for model parameters
conf = results.conf_int(alpha=0.05)
print(f"Intercept 95% CI: [{conf.iloc[0, 0]:.2f}, {conf.iloc[0, 1]:.2f}]")
print(f"Coefficient 95% CI: [{conf.iloc[1, 0]:.2f}, {conf.iloc[1, 1]:.2f}]")

####################################Carbon###################################################
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# Set font family and font size
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 18

# Define base directory
BASE_DIR = 'C:/Carbon/'

# Load CSV file
file_path = BASE_DIR + 'yearly19_base%.csv'
data = pd.read_csv(file_path)

# Extract required columns
data = data[['survey_year', 'Net change_base%', 'mean_defoliation_base', 'Net change']]

# Check for invalid values in the data
data.replace([np.inf, -np.inf], np.nan, inplace=True)
data.dropna(subset=['Net change_base%', 'mean_defoliation_base', 'Net change'], inplace=True)

# Define colors for different survey year periods
colors = {
    (1990, 1999): '#70AD47',
    (2000, 2009): '#5B9BD5',
    (2010, 2019): '#FFC000',
    (2020, 2021): '#C00000'
}

# Add color column based on survey_year
def get_color(year):
    for (start, end), color in colors.items():
        if start <= year <= end:
            return color
    return 'black'

data['color'] = data['survey_year'].apply(get_color)

# Determine the actual range for Net change
min_net_change = np.floor(data['Net change'].min() * 10) / 10
max_net_change = np.ceil(data['Net change'].max() * 10) / 10

# Define size ranges for the circles
size_ranges = [(0.4, 0.6), (0.6, 0.8), (0.8, 1.0)]
size_factors = {range_tuple: 100 * (idx + 1) for idx, range_tuple in enumerate(size_ranges)}

# Adjust size factors to ensure clear distinction between different circle sizes
size_factors[size_ranges[-1]] = 150
size_factors[size_ranges[-2]] = 80
size_factors[size_ranges[-3]] = 30

# Create scatter plot (unsorted)
plt.figure(figsize=(7, 6))
for period, color in colors.items():
    subset = data[(data['survey_year'] >= period[0]) & (data['survey_year'] <= period[1])]
    sizes = []
    for value in subset['Net change']:
        for range_tuple, factor in size_factors.items():
            if range_tuple[0] <= value < range_tuple[1]:
                sizes.append(factor)
                break
    plt.scatter(subset['mean_defoliation_base'], subset['Net change_base%'], c=color, s=sizes)

# Add black dashed lines at x=0 and y=0
plt.axhline(y=0, color='black', linestyle='--', linewidth=1)
plt.axvline(x=0, color='black', linestyle='--', linewidth=1)

# Linear regression (1990-2021) without sorting
X = data['mean_defoliation_base'].values.reshape(-1, 1)
y = data['Net change_base%']
reg = LinearRegression().fit(X, y)
y_pred = reg.predict(X)

# Calculate confidence intervals
X_with_const = sm.add_constant(X)
model = sm.OLS(y, X_with_const)
results = model.fit()
predictions = results.get_prediction(X_with_const)
frame = predictions.summary_frame(alpha=0.05)
conf_int_lower = frame['mean_ci_lower']
conf_int_upper = frame['mean_ci_upper']

# Plot regression line and confidence interval (1990-2021) without sorting
plt.plot(data['mean_defoliation_base'], y_pred, linestyle='-', color='gray', label='_nolegend_')
plt.fill_between(data['mean_defoliation_base'], conf_int_lower, conf_int_upper, color='gray', alpha=0.2)

# Linear regression (2010-2021) without sorting
data_2010_2021 = data[(data['survey_year'] >= 2010) & (data['survey_year'] <= 2021)]
X_2010_2021 = data_2010_2021['mean_defoliation_base'].values.reshape(-1, 1)
y_2010_2021 = data_2010_2021['Net change_base%']
reg_2010_2021 = LinearRegression().fit(X_2010_2021, y_2010_2021)
y_pred_2010_2021 = reg_2010_2021.predict(X_2010_2021)

# Ensure data is sorted for 2010-2021 regression
sorted_idx_2010_2021 = np.argsort(data_2010_2021['mean_defoliation_base'])
sorted_x_2010_2021 = data_2010_2021['mean_defoliation_base'].values[sorted_idx_2010_2021]
sorted_y_pred_2010_2021 = y_pred_2010_2021[sorted_idx_2010_2021]

# Calculate confidence intervals (2010-2021)
X_with_const_2010_2021 = sm.add_constant(X_2010_2021)
model_2010_2021 = sm.OLS(y_2010_2021, X_with_const_2010_2021)
results_2010_2021 = model_2010_2021.fit()
predictions_2010_2021 = results_2010_2021.get_prediction(X_with_const_2010_2021)
frame_2010_2021 = predictions_2010_2021.summary_frame(alpha=0.05)
conf_int_lower_2010_2021 = frame_2010_2021['mean_ci_lower']
conf_int_upper_2010_2021 = frame_2010_2021['mean_ci_upper']

# Ensure confidence interval data is sorted for 2010-2021 regression
sorted_conf_int_lower_2010_2021 = conf_int_lower_2010_2021.values[sorted_idx_2010_2021]
sorted_conf_int_upper_2010_2021 = conf_int_upper_2010_2021.values[sorted_idx_2010_2021]

# Plot regression line and confidence interval (2010-2021) without sorting
plt.plot(sorted_x_2010_2021, sorted_y_pred_2010_2021, linestyle='-', color='#88419d', label='_nolegend_')
plt.fill_between(sorted_x_2010_2021, sorted_conf_int_lower_2010_2021, sorted_conf_int_upper_2010_2021, color='#88419d', alpha=0.2)

# Set axis ranges
plt.xlim(-5, 5)
plt.ylim(-50, 50)
plt.xticks(np.arange(-5, 5.00001, 1))
plt.yticks(np.arange(-50, 51, 10))

# Significance level for regression (1990-2021)
p_value = results.pvalues.iloc[1]
if p_value < 0.001:
    significance = '***'
elif p_value < 0.01:
    significance = '**'
elif p_value < 0.05:
    significance = '*'
else:
    significance = ''

# Display regression equation, R² value, and significance (1990-2021)
coef = reg.coef_[0]
intercept = reg.intercept_
sign = '+' if intercept >= 0 else '-'
plt.text(0.18, 0.90, f'1990-2021: y = {coef:.2f}x {sign} {abs(intercept):.2f}{significance}  R² = {reg.score(X, y):.2f}', transform=plt.gca().transAxes, fontsize=16, verticalalignment='bottom')

# Significance level for regression (2010-2021)
p_value_2010_2021 = results_2010_2021.pvalues.iloc[1]
if p_value_2010_2021 < 0.001:
    significance_2010_2021 = '***'
elif p_value_2010_2021 < 0.01:
    significance_2010_2021 = '**'
elif p_value_2010_2021 < 0.05:
    significance_2010_2021 = '*'
else:
    significance_2010_2021 = ''

# Display regression equation, R² value, and significance (2010-2021)
coef_2010_2021 = reg_2010_2021.coef_[0]
intercept_2010_2021 = reg_2010_2021.intercept_
sign_2010_2021 = '+' if intercept_2010_2021 >= 0 else '-'
plt.text(0.18, 0.85, f'2010-2021: y = {coef_2010_2021:.2f}x {sign_2010_2021} {abs(intercept_2010_2021):.2f}{significance_2010_2021}  R² = {reg_2010_2021.score(X_2010_2021, y_2010_2021):.2f}', transform=plt.gca().transAxes, fontsize=16, verticalalignment='bottom', color='#88419d')

# Add title and axis labels
plt.xlabel('$\delta$LTA$_{Def}$, %', fontsize=20)
plt.ylabel('$\delta$LTA$_{NetC\\_biomass}$, %', fontsize=20)

# Create an inset legend in a cross-tab format at the lower left of the plot
ax_inset = inset_axes(plt.gca(), width="43%", height="26%", loc='lower left', borderpad=0.8)
ax_inset.axis('off')

# Add legend title
ax_inset.text(0.75, len(colors) + 1.2, r'NetC$_{biomass}$, tC ha$^{-1}$',
              horizontalalignment='left', fontsize=12)

# Add size range labels
for i, size_range in enumerate(size_ranges):
    x = i + 0.85
    ax_inset.text(x, len(colors) + 0.5, f'{size_range[0]}-{size_range[1]}', horizontalalignment='center', fontsize=12)

# Add legend entries for each period
for i, (period, color) in enumerate(colors.items()):
    y = len(colors) - i  # Arrange from top to bottom
    ax_inset.text(-1.2, y, f'{period[0]}-{period[1]}', verticalalignment='center', fontsize=12, color=color)
    for j, (size_range, factor) in enumerate(size_factors.items()):
        x = j + 0.85  # Arrange from left to right
        ax_inset.scatter(x, y, s=factor, color=color)

# Set legend axis limits and remove ticks/labels
ax_inset.set_xlim(-1, len(size_factors) + 0.5)
ax_inset.set_ylim(0, len(colors) + 1)
ax_inset.set_xticks(np.arange(len(size_factors)) + 0.85)
ax_inset.set_xticklabels([])
ax_inset.set_yticks([])

# Save the plot to a file
output_path = BASE_DIR + '3_net_change_and_defdev.png'
plt.savefig(output_path, bbox_inches='tight')

# Display the plot
plt.show()

####################################Carbon###################################################
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import t
from sklearn.metrics import r2_score

# Set font properties
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 18

# Define base directory
BASE_DIR = 'C:/Carbon/'

# Read CSV file
file_path = BASE_DIR + 'yearly19_base%.csv'
data = pd.read_csv(file_path)

# Calculate 3-year moving averages
data['Net change_base%_MA'] = data['Net change_base%'].rolling(window=3).mean()
data['mean_defoliation_base_MA'] = data['mean_defoliation_base'].rolling(window=3).mean()
data['Net CO2 emissions/kt_ctrybase%_MA'] = data['Net CO2 emissions/kt_ctrybase%'].rolling(window=3).mean()

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

# Prepare data by dropping missing values
data = data.dropna(subset=['survey_year', 'Net change_base%', 'mean_defoliation_base', 'Net CO2 emissions/kt_ctrybase%'])
x = data['survey_year']

# For Net change_base%
y_net_change = data['Net change_base%']
y_net_change_ma = data['Net change_base%_MA'].dropna()
x_net_change_ma = x.iloc[y_net_change_ma.index]
net_change_fit, net_change_ci_lower, net_change_ci_upper, net_change_residuals, net_change_poly = calculate_polyfit_confidence(
    x_net_change_ma, y_net_change_ma)
net_change_r2 = r2_score(y_net_change_ma, net_change_fit)
net_change_std = np.std(net_change_residuals)

# For Net CO2 emissions/kt_ctrybase%
y_carbon_sink = data['Net CO2 emissions/kt_ctrybase%']
y_carbon_sink_ma = data['Net CO2 emissions/kt_ctrybase%_MA'].dropna()
x_carbon_sink_ma = x.iloc[y_carbon_sink_ma.index]
carbon_sink_fit, carbon_sink_ci_lower, carbon_sink_ci_upper, carbon_sink_residuals, carbon_sink_poly = calculate_polyfit_confidence(
    x_carbon_sink_ma, y_carbon_sink_ma)
carbon_sink_r2 = r2_score(y_carbon_sink_ma, carbon_sink_fit)
carbon_sink_std = np.std(carbon_sink_residuals)

# For mean_defoliation_base on moving average
y_defoliation = data['mean_defoliation_base']
y_defoliation_ma = data['mean_defoliation_base_MA'].dropna()
x_defoliation_ma = x.iloc[y_defoliation_ma.index]
defoliation_fit, defoliation_ci_lower, defoliation_ci_upper, defoliation_residuals, defoliation_poly = calculate_polyfit_confidence(
    x_defoliation_ma, y_defoliation_ma)
defoliation_r2 = r2_score(y_defoliation_ma, defoliation_fit)
defoliation_std = np.std(defoliation_residuals)

# Create figure and primary axis
fig, ax1 = plt.subplots(figsize=(7, 6))

# Plot for Net change_base%
ax1.set_xlabel('Year')
ax1.set_ylabel('$\delta$LTA$_{NetC\\_biomass}$ and $\delta$LTA$_{NetC\\_land}$, %')
ax1.plot(x_net_change_ma, net_change_fit, color='#82d0ff', linewidth=3,
         label=rf'$\delta$LTA$_{{NetC\\_biomass}}$: $R^2$={net_change_r2:.2f}, Std={net_change_std:.2f}')
ax1.fill_between(x_net_change_ma, net_change_ci_lower, net_change_ci_upper, color='#82d0ff', alpha=0.2)
ax1.scatter(x, y_net_change, color='#82d0ff', s=30, marker='o', alpha=0.3, edgecolors='none')

# Plot for Net CO2 emissions/kt_ctrybase%
ax1.plot(x_carbon_sink_ma, carbon_sink_fit, color='limegreen', linewidth=3,
         label=rf'$\delta$LTA$_{{NetC\\_land}}$: $R^2$={carbon_sink_r2:.2f}, Std={carbon_sink_std:.2f}')
ax1.fill_between(x_carbon_sink_ma, carbon_sink_ci_lower, carbon_sink_ci_upper, color='limegreen', alpha=0.2)
ax1.scatter(x, y_carbon_sink, color='limegreen', s=30, marker='o', alpha=0.3, edgecolors='none')

# Plot for mean_defoliation_base on secondary y-axis
ax2 = ax1.twinx()
ax2.set_ylabel('$\delta$LTA$_{Def}$, %', fontsize=18)
ax2.plot(x_defoliation_ma, defoliation_fit, color='#c00000', linewidth=3,
         label=rf'$\delta$LTA$_{{Def}}$: $R^2$={defoliation_r2:.2f}, Std={defoliation_std:.2f}')
ax2.fill_between(x_defoliation_ma, defoliation_ci_lower, defoliation_ci_upper, color='#c00000', alpha=0.2)
ax2.scatter(x, y_defoliation, color='#c00000', s=30, marker='o', alpha=0.3, edgecolors='none')

# Add horizontal line at y=0
ax1.axhline(y=0, color='black', linestyle='--', linewidth=1)

# Set x-axis range and ticks
ax1.set_xlim([1989.5, 2021.5])
ax1.set_xticks([1990, 1995, 2000, 2005, 2010, 2015, 2020])
ax1.set_ylim([-50, 50])
ax1.set_yticks(range(-50, 51, 10))

# Set secondary y-axis range and ticks
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
plt.savefig(BASE_DIR + '4_NetC02_def_Netbiomass_polyfi.jpg', dpi=300)

# Show the figure
plt.show()
