"""""""""""""""""""""""""""""""""""
This script performs several tasks related to the analysis of defoliation, Net Change, and Net CO2 emissions data.

The Carbon related data are available at https://unfccc.int/ghg-inventories-annex-i-parties/2024. 
1.It computes annual averages and plots a dual-axis chart of Net Change (line) and Net CO2 emissions (bars).
2.It performs linear regression on Net CO2 emissions (baseline) versus Net Change (baseline) , displaying confidence and prediction intervals with regression statistics.
3.It creates scatter plots with regression lines for different periods (1990–2022 and 2010–2022), including annotated regression details. 
4.It calculates 3-year moving averages, applies polynomial fitting with confidence intervals, and plots the trends for net change (baseline) and defoliation (baseline).
5.Performs random-effects meta-analysis of 1990–2022 and 2010–2022 country data and plots a forest plot.
"""""""""""""""""""""""""""""""""""
###################################### 1.It computes annual averages and plots a dual-axis chart of Net Change (line) and Net CO2 emissions (bars). ######################################
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
ax1.set_xlim(1989.5, 2022.5)
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
###################################### 2.It performs linear regression on Net CO2 emissions (baseline) versus Net Change(baseline), displaying confidence and prediction intervals with regression statistics. ######################################
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
    (1990, 1999): '#98DCED',
    (2000, 2009): '#A0BEF5',
    (2010, 2019): '#D88BF2',
    (2020, 2022): '#9D2BC2'
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
plt.xlabel('$\\delta$LTA$_{NetC\\_biomass}$, %', fontsize=20)
plt.ylabel('$\\delta$LTA$_{NetC\\_land}$, %', fontsize=20)
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
plt.xlabel('$\\delta$LTA$_{NetC\\_biomass}$, %', fontsize=20)
plt.ylabel('$\\delta$LTA$_{NetC\\_land}$, %', fontsize=20)
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

###################################### 3.It creates scatter plots with regression lines for different periods (1990–2022 and 2010–2022), including annotated regression details. ######################################
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
    (1990, 1999): '#98DCED',
    (2000, 2009): '#A0BEF5',
    (2010, 2019): '#D88BF2',
    (2020, 2022): '#9D2BC2'
}
# Add color column based on survey_year
def get_color(year):
    for (start, end), color in colors.items():
        if start <= year <= end:
            return color
    return 'black'

data['color'] = data['survey_year'].apply(get_color)

# Define size ranges for the circles
size_ranges = [(0.4, 0.6), (0.6, 0.8), (0.8, 1.0)]
size_values = [30, 80, 150]
fallback_size = 40
def map_size(v):
    for (lo, hi), s in zip(size_ranges, size_values):
        if lo <= v < hi:
            return s
    return fallback_size

# Create scatter plot (unsorted)
fig, ax = plt.subplots(figsize=(7, 6))

for period, color in colors.items():
    subset = data[(data['survey_year'] >= period[0]) & (data['survey_year'] <= period[1])]
    if not subset.empty:
        sizes = subset['Net change'].apply(map_size).values
        ax.scatter(subset['mean_defoliation_base'],
                   subset['Net change_base%'],
                   c=color, s=sizes, edgecolors='none')

ax.axhline(0, color='black', linestyle='--', linewidth=1)
ax.axvline(0, color='black', linestyle='--', linewidth=1)

X = data['mean_defoliation_base'].values.reshape(-1, 1)
y = data['Net change_base%'].values
order = np.argsort(X.ravel())
X_sorted = X[order]

reg_all = LinearRegression().fit(X, y)
yhat_all = reg_all.predict(X_sorted)
ols_all = sm.OLS(y, sm.add_constant(X)).fit()
pred_all = ols_all.get_prediction(sm.add_constant(X_sorted)).summary_frame(alpha=0.05)

ax.plot(X_sorted.ravel(), yhat_all, '-', color='gray', linewidth=1.5)
ax.fill_between(X_sorted.ravel(), pred_all['mean_ci_lower'], pred_all['mean_ci_upper'], color='gray', alpha=0.2)

p_all = ols_all.pvalues[1]
stars_all = '***' if p_all < 0.001 else '**' if p_all < 0.01 else '*' if p_all < 0.05 else ''
ax.text(0.18, 0.90,
        f'1990-2022: y = {reg_all.coef_[0]:.2f}x {reg_all.intercept_:+.2f}{stars_all}  R² = {reg_all.score(X, y):.2f}',
        transform=ax.transAxes, fontsize=16, va='bottom', color='black')


mask = (data['survey_year'] >= 2010) & (data['survey_year'] <= 2022)
d1021 = data.loc[mask]
if d1021.shape[0] >= 2:
    X2 = d1021['mean_defoliation_base'].values.reshape(-1, 1)
    y2 = d1021['Net change_base%'].values
    order2 = np.argsort(X2.ravel())
    X2_sorted = X2[order2]

    reg2 = LinearRegression().fit(X2, y2)
    yhat2 = reg2.predict(X2_sorted)
    ols2 = sm.OLS(y2, sm.add_constant(X2)).fit()
    pred2 = ols2.get_prediction(sm.add_constant(X2_sorted)).summary_frame(alpha=0.05)

    deep_pink = '#C7067F'
    ax.plot(X2_sorted.ravel(), yhat2, '-', color=deep_pink, linewidth=1.5)
    ax.fill_between(X2_sorted.ravel(), pred2['mean_ci_lower'], pred2['mean_ci_upper'], color=deep_pink, alpha=0.2)

    p2 = ols2.pvalues[1]
    stars2 = '***' if p2 < 0.001 else '**' if p2 < 0.01 else '*' if p2 < 0.05 else ''
    ax.text(0.18, 0.85,
            f'2010-2022: y = {reg2.coef_[0]:.2f}x {reg2.intercept_:+.2f}{stars2}  R² = {reg2.score(X2, y2):.2f}',
            transform=ax.transAxes, fontsize=16, va='bottom', color=deep_pink)

ax.set_xlim(-5, 5)
ax.set_ylim(-60, 60)
ax.set_xticks(np.arange(-5, 6, 1))
ax.set_yticks(np.arange(-60, 61, 10))
ax.set_xlabel(r'$\delta$LTA$_{Def}$, %', fontsize=20)
ax.set_ylabel(r'$\delta$LTA$_{NetC\_biomass}$, %', fontsize=20)

# Inset legend
ax_inset = inset_axes(ax, width="43%", height="26%", loc='lower left', borderpad=0.8)
ax_inset.axis('off')

ax_inset.text(0.75, len(colors) + 1.2, r'NetC$_{biomass}$, tC ha$^{-1}$', ha='left', fontsize=12)
for j, (rng, s) in enumerate(zip(size_ranges, size_values)):
    x = j + 0.85
    ax_inset.text(x, len(colors) + 0.5, f'{rng[0]}-{rng[1]}', ha='center', fontsize=12)

for i, (period, color) in enumerate(colors.items()):
    yrow = len(colors) - i
    ax_inset.text(-1.2, yrow, f'{period[0]}-{period[1]}', va='center', fontsize=12, color=color)
    for j, (rng, s) in enumerate(zip(size_ranges, size_values)):
        x = j + 0.85
        ax_inset.scatter(x, yrow, s=s, color=color, edgecolors='none')

ax_inset.set_xlim(-1, len(size_values) + 0.5)
ax_inset.set_ylim(0, len(colors) + 1)
ax_inset.set_xticks([]); ax_inset.set_yticks([])

# Remove legend
leg = ax.get_legend()
if leg is not None:
    leg.remove()


out_path = os.path.join(BASE_DIR, '3_net_change_and_defdev.png.pdf')
plt.savefig(out_path, bbox_inches='tight', dpi=300)
plt.close()
print(f"✅Save：{out_path}")

###################################### 4.It calculates 3-year moving averages, applies polynomial fitting with confidence intervals, and plots the trends for net change (baseline) and defoliation (baseline). ######################################
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import t
from sklearn.metrics import r2_score

# Set font
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
#data['Net CO2 emissions/kt_ctrybase%_MA'] = data['Net CO2 emissions/kt_ctrybase%'].rolling(window=3).mean()

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
#y_carbon_sink = data['Net CO2 emissions/kt_ctrybase%']
#y_carbon_sink_ma = data['Net CO2 emissions/kt_ctrybase%_MA'].dropna()
#x_carbon_sink_ma = x.iloc[y_carbon_sink_ma.index]
#carbon_sink_fit, carbon_sink_ci_lower, carbon_sink_ci_upper, carbon_sink_residuals, carbon_sink_poly = calculate_polyfit_confidence(
  #carbon_sink_r2 = r2_score(y_carbon_sink_ma, carbon_sink_fit)
#carbon_sink_std = np.std(carbon_sink_residuals)

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
ax1.set_ylabel('$\delta$LTA$_{NetC\\_biomass}$, %')
ax1.plot(x_net_change_ma, net_change_fit, color='#C7067F', linewidth=3,
         label=rf'$\delta$LTA$_{{NetC\_biomass}}$: $R^2$={net_change_r2:.2f}, Std={net_change_std:.2f}')
ax1.fill_between(x_net_change_ma, net_change_ci_lower, net_change_ci_upper, color='#C7067F', alpha=0.2)
ax1.scatter(x, y_net_change, color='#C7067F', s=30, marker='o', alpha=0.3, edgecolors='none')

# Plot for Net CO2 emissions/kt_ctrybase%
# ax1.plot(x_carbon_sink_ma, carbon_sink_fit, color='limegreen', linewidth=3,
#          label=rf'$\delta$LTA$_{{NetC\_land}}$: $R^2$={carbon_sink_r2:.2f}, Std={carbon_sink_std:.2f}')
# ax1.fill_between(x_carbon_sink_ma, carbon_sink_ci_lower, carbon_sink_ci_upper, color='limegreen', alpha=0.2)
# ax1.scatter(x, y_carbon_sink, color='limegreen', s=30, marker='o', alpha=0.3, edgecolors='none')

# Plot for mean_defoliation_base on secondary y-axis
ax2 = ax1.twinx()
ax2.set_ylabel('$\delta$LTA$_{Def}$, %', fontsize=18)
ax2.plot(x_defoliation_ma, defoliation_fit, color='Grey', linewidth=3,
         label=rf'$\delta$LTA$_{{Def}}$: $R^2$={defoliation_r2:.2f}, Std={defoliation_std:.2f}')
ax2.fill_between(x_defoliation_ma, defoliation_ci_lower, defoliation_ci_upper, color='Grey', alpha=0.2)
ax2.scatter(x, y_defoliation, color='Grey', s=30, marker='o', alpha=0.3, edgecolors='none')

# Add horizontal line at y=0
ax1.axhline(y=0, color='black', linestyle='--', linewidth=1)

# Set x-axis range and ticks
ax1.set_xlim([1989.5, 2022.5])
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
plt.savefig(BASE_DIR + '4_NetC02_def_Netbiomass_polyfit.jpg', dpi=300)

# Show the figure
plt.show()
###################################### 5.Performs random-effects meta-analysis of 1990–2022 and 2010–2022 country data and plots a forest plot. ######################################
import os, re, math, warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import statsmodels.api as sm

# Fonts
plt.rcParams["font.family"] = "Arial"
plt.rcParams["axes.unicode_minus"] = False

# Enforce Arial + normal weight; keep text editable in PDF ====
plt.rcParams["font.sans-serif"]   = ["Arial"]
plt.rcParams["font.weight"]       = "normal"
plt.rcParams["axes.titleweight"]  = "normal"
plt.rcParams["axes.labelweight"]  = "normal"
plt.rcParams["pdf.fonttype"]      = 42   # Embed TrueType; keep text editable in PDF
plt.rcParams["ps.fonttype"]       = 42
plt.rcParams["legend.title_fontsize"] = 20  # Coordinate with FS_LEGEND

# Robust chi-square SF: p(Q)
_HAVE_SCIPY = False
try:
    from scipy.stats import chi2 as _chi2
    _HAVE_SCIPY = True
except Exception:
    pass

_HAVE_MPMATH = False
try:
    import mpmath as _mp
    _HAVE_MPMATH = True
except Exception:
    pass

def _chi2_sf_wh(x, k):
    # Wilson–Hilferty normal approximation (right tail)
    if not (np.isfinite(x) and np.isfinite(k)) or k <= 0:
        return float('nan')
    if x < 0:
        return 1.0
    z = ((x / k) ** (1.0/3.0) - (1.0 - 2.0/(9.0*k))) / math.sqrt(2.0/(9.0*k))
    return 0.5 * math.erfc(z / math.sqrt(2.0))

def chi2_sf(x, k):
    # Survival function for chi-square: p(Q >= x | df = k)
    if not (np.isfinite(x) and np.isfinite(k)) or k <= 0:
        return float('nan')
    try:
        if _HAVE_SCIPY:
            return float(_chi2.sf(x, k))
        if _HAVE_MPMATH:
            return float(1.0 - (_mp.gammainc(k/2.0, 0, x/2.0) / _mp.gamma(k/2.0)))
    except Exception:
        pass
    return _chi2_sf_wh(x, k)

BASE_DIR   = r"C:\Carbon"
INPUT_FILE = os.path.join(BASE_DIR, r"yearcountry_base%.csv")
OUT_DIR    = os.path.join(BASE_DIR, "meta_outputs")
os.makedirs(OUT_DIR, exist_ok=True)

HAC_LAG = 1
ALPHA   = 0.05

# Output names
OUT_PDF = os.path.join(OUT_DIR, "forest_combined_two_periods.pdf")
CSV_EFFECTS = os.path.join(OUT_DIR, "supplement_effects_weights_RE.csv")
CSV_HETEROG = os.path.join(OUT_DIR, "supplement_heterogeneity_RE.csv")

# X-axis range and ticks
X_MIN, X_MAX, X_STEP = -80, 80, 20
X_TICKS = np.arange(X_MIN, X_MAX + 1, X_STEP)

# Vertical spacing (increased)
ROW_SPACING = 1.75       # was 1.35
DELTA_PAIR  = 0.28

# Font sizes
FS_AXIS     = 24
FS_AXIS_LBL = FS_AXIS + 1
FS_TICK     = 22
FS_HEADER   = 24
FS_STAR     = 19
FS_LEGEND   = 20

# Layout: forest area
FOREST_LEFT   = 0.16
FOREST_BOTTOM = 0.16
FOREST_WIDTH  = 0.56 * 0.75   # constrain to ~3/4 width
FOREST_HEIGHT = 0.76

# "Country" header position
HEADER_Y_ABOVE = 1.02
COUNTRY_HEADER_X = -0.23

# ===================== COLORS =====================
PALETTE = {
    "1990_pos": (0/255,   102/255, 204/255, 1.0),  # dark blue
    "1990_neg": (204/255,  51/255,  51/255, 1.0),  # dark red
    "2010_pos": (170/255, 200/255, 245/255, 1.0),  # light blue
    "2010_neg": (255/255, 175/255, 175/255, 1.0),  # light red
}
def color_by_sign_period(beta, period_key: str):
    # Choose color by sign and period
    return (PALETTE["1990_pos"] if beta >= 0 else PALETTE["1990_neg"]) if period_key=="1990" \
           else (PALETTE["2010_pos"] if beta >= 0 else PALETTE["2010_neg"])

# UTILS
def _norm_name(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(s).strip().lower())

def smart_find_column(df: pd.DataFrame, preferred_variants):
    # Fuzzy column matching by exact-lower, normalized tokens, and token containment
    lower_map = {c.lower(): c for c in df.columns}
    for name in preferred_variants:
        if name.lower() in lower_map: return lower_map[name.lower()]
    norm_map = {_norm_name(c): c for c in df.columns}
    for name in preferred_variants:
        key = _norm_name(name)
        if key in norm_map: return norm_map[key]
    for name in preferred_variants:
        toks = [t for t in re.split(r"[^a-z0-9]+", name.lower()) if t]
        for c in df.columns:
            if all(t in c.lower() for t in toks): return c
    return None

def detect_year_col(df: pd.DataFrame):
    # Try to detect a year-like column by name and plausible numeric range
    for c in df.columns:
        cl = c.lower()
        if ("year" in cl) or ("yr" in cl) or ("年份" in cl) or cl.endswith("年"):
            s = pd.to_numeric(df[c], errors="coerce")
            if s.dropna().between(1950,2035).any(): return c
    return None

def star_from_p(p):
    if pd.isna(p): return ""
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return ""

# Robust save
def safe_save(fig, path_pdf, dpi=300, bbox="tight"):
    path_pdf = os.path.normpath(path_pdf)
    os.makedirs(os.path.dirname(path_pdf), exist_ok=True)
    try:
        fig.savefig(path_pdf, bbox_inches=bbox)
    except Exception:
        # Fallback name if the first save fails
        alt = os.path.splitext(path_pdf)[0] + "_safe.pdf"
        try:
            fig.savefig(alt, bbox_inches=bbox)
        except Exception:
            pass

# EFFECTS & META
def fit_country_ols_hac(sub, x_col, y_col, hac_lag=1):
    # OLS with HAC (Newey-West) standard errors as default; fall back to plain OLS if needed
    X = sm.add_constant(sub[x_col].values)
    try:
        m = sm.OLS(sub[y_col].values, X).fit(cov_type="HAC", cov_kwds={"maxlags": hac_lag})
    except Exception:
        m = sm.OLS(sub[y_col].values, X).fit()
    beta = float(m.params[1]); se = float(m.bse[1])
    t = float(m.tvalues[1]) if hasattr(m, "tvalues") else (beta/se if se>0 else np.nan)
    p = float(m.pvalues[1]) if hasattr(m, "pvalues") else np.nan
    return beta, se, t, p, int(m.nobs)

def build_period_effects(df, country_col, x_col, y_col, year_col, y_min, y_max):
    dff = df.copy()
    # Filter by year if a year column exists
    if year_col is not None:
        years = pd.to_numeric(dff[year_col], errors="coerce")
        dff = dff.loc[(years >= y_min) & (years <= y_max)].copy()
    use_cols = [country_col, x_col, y_col] + ([year_col] if year_col else [])
    dff = dff[use_cols].copy()
    dff[x_col] = pd.to_numeric(dff[x_col], errors="coerce")
    dff[y_col] = pd.to_numeric(dff[y_col], errors="coerce")
    dff = dff.dropna(subset=[x_col, y_col])

    rows = []
    for g, sub in dff.groupby(country_col):
        sub = sub.dropna(subset=[x_col, y_col])
        if (len(sub) < 3) or (sub[x_col].nunique() < 2): continue
        beta, se, t, p, n = fit_country_ols_hac(sub, x_col, y_col, hac_lag=HAC_LAG)
        yr_min = yr_max = None
        if year_col:
            yrs = pd.to_numeric(sub[year_col], errors="coerce")
            if yrs.notna().any():
                yr_min, yr_max = int(np.nanmin(yrs.values)), int(np.nanmax(yrs.values))
        rows.append({"country":str(g),"beta":beta,"se":se,"t":t,"p":p,"n":n,"yr_min":yr_min,"yr_max":yr_max})

    eff = pd.DataFrame(rows).sort_values("country").reset_index(drop=True)
    if eff.empty: return eff, None, None
    eff["ci_low"]  = eff["beta"] - 1.96*eff["se"]
    eff["ci_high"] = eff["beta"] + 1.96*eff["se"]

    # Fixed-effect weights
    eff["w_FE"] = 1.0/(eff["se"]**2)
    sum_wFE = float(eff["w_FE"].sum())
    beta_FE = float((eff["w_FE"]*eff["beta"]).sum()/sum_wFE)
    se_FE   = math.sqrt(1.0/sum_wFE)
    ci_FE   = (beta_FE-1.96*se_FE, beta_FE+1.96*se_FE)

    # DerSimonian–Laird tau^2
    Q  = float(((eff["beta"]-beta_FE)**2 * eff["w_FE"]).sum())
    dfQ = len(eff)-1
    C  = float(sum_wFE - (eff["w_FE"]**2).sum()/sum_wFE)
    tau2 = float(max(0.0,(Q-dfQ)/C)) if C>0 else 0.0

    # Random-effects weights
    eff["w_RE"] = 1.0/(eff["se"]**2 + tau2)
    sum_wRE = float(eff["w_RE"].sum())
    beta_RE = float((eff["w_RE"]*eff["beta"]).sum()/sum_wRE)
    se_RE   = math.sqrt(1.0/sum_wRE)
    ci_RE   = (beta_RE-1.96*se_RE, beta_RE+1.96*se_RE)
    eff["w_RE_pct"] = 100.0 * eff["w_RE"]/sum_wRE

    # Heterogeneity (we export RE-only summary)
    I2 = float(max(0.0,(Q-dfQ)/Q))*100.0 if Q>0 else 0.0
    H  = math.sqrt(Q/dfQ) if dfQ>0 else np.nan
    Phi = lambda z: 0.5*(1.0 + math.erf(z/math.sqrt(2.0)))
    z_overall = beta_RE/se_RE if se_RE>0 else np.nan
    p_overall = 2.0*(1.0 - Phi(abs(z_overall))) if math.isfinite(z_overall) else np.nan
    pi_low  = beta_RE - 1.96*math.sqrt(tau2 + se_RE**2)
    pi_high = beta_RE + 1.96*math.sqrt(tau2 + se_RE**2)

    meta = dict(
        k=len(eff),
        beta_FE=beta_FE, se_FE=se_FE, ci_FE_low=ci_FE[0], ci_FE_high=ci_FE[1],
        beta_RE=beta_RE, se_RE=se_RE, ci_RE_low=ci_RE[0], ci_RE_high=ci_RE[1],
        Q=Q, df_Q=dfQ, I2=I2, tau2=tau2, H=H, z=z_overall, p=p_overall,
        PI_low=pi_low, PI_high=pi_high
    )

    appendix = eff.copy()
    appendix["stars"] = appendix["p"].apply(lambda p: "***" if p<0.001 else "**" if p<0.01 else "*" if p<0.05 else "")
    return eff, meta, appendix

# Combined forest plot (RE only; Overall at top; no right-side numbers / no heterogeneity text)
def draw_forest_combo_RE_only(eff_2010, meta_2010, eff_1990, meta_1990, order, out_pdf):
    # Align two periods and reindex by 'order'
    A2010 = eff_2010.set_index("country").reindex(order).dropna(subset=["beta","se"]).reset_index()
    B1990 = eff_1990.set_index("country").reindex(order).dropna(subset=["beta","se"]).reset_index()

    n_countries = len(order)
    # Reserve 1 extra row at the top for Overall (RE)
    y = np.arange(n_countries + 1, dtype=float) * ROW_SPACING
    y_overall = y[-1]  # top row
    def y_i(i): return y[i]  # i=0 is bottom-most

    dx = 0.02*(X_MAX - X_MIN)

    fig = plt.figure(figsize=(15.0, max(7.5, 0.70*(n_countries+1) + 2.8)))
    fig.patch.set_facecolor("white")
    ax = fig.add_axes([FOREST_LEFT, FOREST_BOTTOM, FOREST_WIDTH, FOREST_HEIGHT])

    # For each country: 1990 up, 2010 down; order is ascending by 2010 slope → low at bottom, high at top
    for i, country in enumerate(order):
        r1990 = B1990[B1990["country"] == country]
        r2010 = A2010[A2010["country"] == country]
        ybase = y_i(i)
        if len(r1990):
            r = r1990.iloc[0]
            col = color_by_sign_period(r["beta"], "1990")
            ax.plot([r["ci_low"], r["ci_high"]], [ybase+DELTA_PAIR, ybase+DELTA_PAIR], color=col, linewidth=2.6)
            sz = 140 + 260*(r["w_RE"]/B1990["w_RE"].max())
            ax.plot(r["beta"], ybase+DELTA_PAIR, marker="s", markersize=np.sqrt(sz), color=col)
            s = "***" if r["p"]<0.001 else "**" if r["p"]<0.01 else "*" if r["p"]<0.05 else ""
            if s:
                ax.text(r["beta"]+dx, ybase+DELTA_PAIR, s, va="center", ha="left", fontsize=FS_STAR, color="black")
        if len(r2010):
            r = r2010.iloc[0]
            col = color_by_sign_period(r["beta"], "2010")
            ax.plot([r["ci_low"], r["ci_high"]], [ybase-DELTA_PAIR, ybase-DELTA_PAIR], color=col, linewidth=2.6)
            sz = 140 + 260*(r["w_RE"]/A2010["w_RE"].max())
            ax.plot(r["beta"], ybase-DELTA_PAIR, marker="s", markersize=np.sqrt(sz), color=col)
            s = "***" if r["p"]<0.001 else "**" if r["p"]<0.01 else "*" if r["p"]<0.05 else ""
            if s:
                ax.text(r["beta"]+dx, ybase-DELTA_PAIR, s, va="center", ha="left", fontsize=FS_STAR, color="black")

    # Overall (RE) at the very top (1990 up, 2010 down)
    def draw_overall_pair_RE(y0, meta1990, meta2010):
        b90, lo90, hi90, p90 = meta1990["beta_RE"], meta1990["ci_RE_low"], meta1990["ci_RE_high"], meta1990["p"]
        b10, lo10, hi10, p10 = meta2010["beta_RE"], meta2010["ci_RE_low"], meta2010["ci_RE_high"], meta2010["p"]
        col90 = color_by_sign_period(b90, "1990")
        col10 = color_by_sign_period(b10, "2010")

        ax.plot([lo90, hi90], [y0+DELTA_PAIR, y0+DELTA_PAIR], color=col90, linewidth=2.8)
        ax.plot(b90, y0+DELTA_PAIR, marker="s", markersize=18, color=col90)
        s90 = "***" if (p90 is not None and p90<0.001) else "**" if (p90 is not None and p90<0.01) else "*" if (p90 is not None and p90<0.05) else ""
        if s90:
            ax.text(b90+dx, y0+DELTA_PAIR, s90, va="center", ha="left", fontsize=FS_STAR, color="black")

        ax.plot([lo10, hi10], [y0-DELTA_PAIR, y0-DELTA_PAIR], color=col10, linewidth=2.8)
        ax.plot(b10, y0-DELTA_PAIR, marker="s", markersize=18, color=col10)
        s10 = "***" if (p10 is not None and p10<0.001) else "**" if (p10 is not None and p10<0.01) else "*" if (p10 is not None and p10<0.05) else ""
        if s10:
            ax.text(b10+dx, y0-DELTA_PAIR, s10, va="center", ha="left", fontsize=FS_STAR, color="black")

    draw_overall_pair_RE(y_overall, meta_1990, meta_2010)

    # Axes and labels
    ax.axvline(0.0, color="grey", linestyle="--", linewidth=1)
    ax.set_xlim(X_MIN, X_MAX); ax.set_xticks(X_TICKS); ax.tick_params(axis='x', labelsize=FS_TICK)
    yticks = list(y[:-1]) + [y_overall]
    ylabels = order + ["Overall (RE)"]
    ax.set_ylim(-0.5*ROW_SPACING, (y_overall+0.5*ROW_SPACING))
    ax.set_yticks(yticks); ax.set_yticklabels(ylabels, fontsize=FS_TICK)
    ax.set_xlabel("Effect size (slope)", fontsize=FS_AXIS_LBL)

    # Country header
    ax.text(COUNTRY_HEADER_X, HEADER_Y_ABOVE, "Country", transform=ax.transAxes,
            fontsize=FS_HEADER, ha="left", va="top", clip_on=False)

    # Color legend (time): two blues on top, two reds below; upper-left; no background
    legend_handles = [
        Line2D([0], [0], marker="s", color=PALETTE["1990_pos"], markersize=13, linestyle="None", label="1990–2022 (+)"),
        Line2D([0], [0], marker="s", color=PALETTE["2010_pos"], markersize=13, linestyle="None", label="2010–2022 (+)"),
        Line2D([0], [0], marker="s", color=PALETTE["1990_neg"], markersize=13, linestyle="None", label="1990–2022 (–)"),
        Line2D([0], [0], marker="s", color=PALETTE["2010_neg"], markersize=13, linestyle="None", label="2010–2022 (–)"),
    ]
    leg_color = ax.legend(handles=legend_handles,  loc='upper left', bbox_to_anchor=(-0.05, 1), frameon=False, fontsize=20)
    ax.add_artist(leg_color)

    # ====== Weight-size legend: tertiles at 33%/66% on w_RE_pct → Low/Medium/High; match marker sizes with main plot; placed under time legend; no background; with title ======
    # (1) Collect the "display marker sizes" consistent with the main plot (each period normalized by its own max w_RE → markersize)
    w_norm_ms = []  # (w_pct, markersize)
    if len(A2010):
        w_norm_2010 = A2010["w_RE"] / A2010["w_RE"].max()
        ms_2010 = np.sqrt(140.0 + 260.0 * w_norm_2010.values)
        w_pct_2010 = A2010["w_RE_pct"].values
        w_norm_ms += list(zip(w_pct_2010, ms_2010))
    if len(B1990):
        w_norm_1990 = B1990["w_RE"] / B1990["w_RE"].max()
        ms_1990 = np.sqrt(140.0 + 260.0 * w_norm_1990.values)
        w_pct_1990 = B1990["w_RE_pct"].values
        w_norm_ms += list(zip(w_pct_1990, ms_1990))

    if not w_norm_ms:
        # Fallback if no weights detected
        wpct_all = np.array([10.0, 50.0, 90.0])
        ms_all   = np.array([np.sqrt(140.0 + 260.0 * 0.25),
                             np.sqrt(140.0 + 260.0 * 0.50),
                             np.sqrt(140.0 + 260.0 * 0.75)])
    else:
        wpct_all = np.array([p for p, _ in w_norm_ms if np.isfinite(p)])
        ms_all   = np.array([m for _, m in w_norm_ms if np.isfinite(m)])

    # (2) 33% / 66% tertiles
    q33, q66 = np.quantile(wpct_all, [1/3, 2/3])

    # (3) For each bin, take the median "display marker size" within the bin
    def bin_median_size(lo, hi, left_inclusive=True, right_inclusive=True):
        if left_inclusive and right_inclusive:
            sel = (wpct_all >= lo) & (wpct_all <= hi)
        elif left_inclusive and not right_inclusive:
            sel = (wpct_all >= lo) & (wpct_all <  hi)
        elif (not left_inclusive) and right_inclusive:
            sel = (wpct_all >  lo) & (wpct_all <= hi)
        else:
            sel = (wpct_all >  lo) & (wpct_all <  hi)
        vals = ms_all[sel]
        return float(np.median(vals)) if vals.size else float(np.median(ms_all))

    ms_low  = bin_median_size(-np.inf, q33, left_inclusive=True,  right_inclusive=True)    # ≤ q33
    ms_med  = bin_median_size(q33,     q66, left_inclusive=False, right_inclusive=True)    # (q33, q66]
    ms_high = bin_median_size(q66,     np.inf, left_inclusive=False, right_inclusive=True) # > q66

    size_handles = [
        Line2D([0], [0], marker="s", linestyle="None", color="gray",
               markersize=ms_low,  label=f"Low (≤ {q33:.1f}%)"),
        Line2D([0], [0], marker="s", linestyle="None", color="gray",
               markersize=ms_med,  label=f"Medium ({q33:.1f}–{q66:.1f}%)"),
        Line2D([0], [0], marker="s", linestyle="None", color="gray",
               markersize=ms_high, label=f"High (≥ {q66:.1f}%)"),
    ]

    leg_sizes = ax.legend(handles=size_handles,
                          loc="upper left",
                          bbox_to_anchor=(-0.04, 0.86),   # right under the time legend; tweak if needed
                          frameon=False, fontsize=18)
    # —— Title not bold (normal) ——
    leg_sizes.set_title("Weight (size)")
    leg_sizes.get_title().set_fontsize(FS_LEGEND)
    leg_sizes.get_title().set_fontweight("normal")
    ax.add_artist(leg_sizes)

    # Save the single PDF output
    safe_save(fig, out_pdf, dpi=300, bbox="tight")
    plt.close(fig)

# ===================== MAIN =====================
def main():
    # Read CSV with common encodings
    df = None
    for enc in ["utf-8", "utf-8-sig", "gbk", "latin1"]:
        try:
            df = pd.read_csv(INPUT_FILE, encoding=enc); break
        except Exception:
            continue
    if df is None or df.empty:
        raise RuntimeError("Failed to read CSV. Please check path/encoding/file.")

    # Column matching helpers
    def _norm(s): return re.sub(r"[^a-z0-9]+", "", str(s).strip().lower())
    def smart_col(df, names):
        m = {c.lower():c for c in df.columns}
        for n in names:
            if n.lower() in m: return m[n.lower()]
        mm = {_norm(c):c for c in df.columns}
        for n in names:
            if _norm(n) in mm: return mm[_norm(n)]
        for n in names:
            toks = [t for t in re.split(r"[^a-z0-9]+", n.lower()) if t]
            for c in df.columns:
                if all(t in c.lower() for t in toks): return c
        return None

    country_col = smart_col(df, ["lib_country", "country"])
    x_col       = smart_col(df, ["mean_defoliation_base", "mean defoliation base"])
    y_col       = smart_col(df, ["Net change_base%", "net change_base%", "net_change_base%"])
    year_col    = detect_year_col(df)
    if not all([country_col, x_col, y_col]):
        raise RuntimeError("Missing columns: lib_country / mean_defoliation_base / Net change_base%")
    print(f"[Columns] country: {country_col} | x: {x_col} | y: {y_col} | year: {year_col}")

    # Two periods
    eff_2010, meta_2010, appendix_2010 = build_period_effects(df, country_col, x_col, y_col, year_col, 2010, 2022)
    if eff_2010.empty or meta_2010 is None:
        raise RuntimeError("No usable countries for 2010–2022.")

    # Sorting: ascending by 2010–2022 slope
    eff_2010 = eff_2010.sort_values(by=["beta"], ascending=True).reset_index(drop=True)
    order = eff_2010["country"].tolist()
    print(f"[Sorting] 2010–2022 slope ascending; #countries: {len(order)}")

    eff_1990, meta_1990, appendix_1990 = build_period_effects(df, country_col, x_col, y_col, year_col, 1990, 2022)
    if eff_1990.empty or meta_1990 is None:
        raise RuntimeError("No usable countries for 1990–2022.")
    eff_1990 = eff_1990.set_index("country").reindex(order).dropna(subset=["beta","se"]).reset_index()

    # ——— Single output: combined forest plot (RE only; Overall at top; no right-side numbers / no heterogeneity text) ———
    draw_forest_combo_RE_only(eff_2010, meta_2010, eff_1990, meta_1990, order, OUT_PDF)
    print(f"✅ Saved figure: {OUT_PDF}")

    # ——— Supplementary CSV (effect sizes & weights; RE only; both periods by country) ———
    sup_rows = []
    A2010 = eff_2010.set_index("country")
    B1990 = eff_1990.set_index("country")
    for c in order:
        if c in B1990.index:
            r = B1990.loc[c]
            sup_rows.append(dict(country=c, period="1990–2022",
                                 beta=r["beta"], ci_low=r["ci_low"], ci_high=r["ci_high"],
                                 weight_RE_pct=r["w_RE_pct"]))
        if c in A2010.index:
            r = A2010.loc[c]
            sup_rows.append(dict(country=c, period="2010–2022",
                                 beta=r["beta"], ci_low=r["ci_low"], ci_high=r["ci_high"],
                                 weight_RE_pct=r["w_RE_pct"]))
    pd.DataFrame(sup_rows).to_csv(CSV_EFFECTS, index=False, encoding="utf-8-sig")
    print(f"✅ Saved supplementary effects/weights CSV: {CSV_EFFECTS}")

    # ——— Supplementary CSV (heterogeneity; RE only: Q, df, p(Q), I², τ², H, overall β_RE & CI, PI) ———
    pQ_1990 = chi2_sf(meta_1990["Q"], meta_1990["df_Q"])
    pQ_2010 = chi2_sf(meta_2010["Q"], meta_2010["df_Q"])
    hetero = pd.DataFrame([
        dict(period="1990–2022", k=meta_1990["k"], beta_RE=meta_1990["beta_RE"],
             ci_RE_low=meta_1990["ci_RE_low"], ci_RE_high=meta_1990["ci_RE_high"],
             PI_low=meta_1990["PI_low"], PI_high=meta_1990["PI_high"],
             Q=meta_1990["Q"], df=meta_1990["df_Q"], p_Q=pQ_1990,
             I2_percent=meta_1990["I2"], tau2=meta_1990["tau2"], H=meta_1990["H"]),
        dict(period="2010–2022", k=meta_2010["k"], beta_RE=meta_2010["beta_RE"],
             ci_RE_low=meta_2010["ci_RE_low"], ci_RE_high=meta_2010["ci_RE_high"],
             PI_low=meta_2010["PI_low"], PI_high=meta_2010["PI_high"],
             Q=meta_2010["Q"], df=meta_2010["df_Q"], p_Q=pQ_2010,
             I2_percent=meta_2010["I2"], tau2=meta_2010["tau2"], H=meta_2010["H"]),
    ])
    hetero.to_csv(CSV_HETEROG, index=False, encoding="utf-8-sig")
    print(f"✅ Saved heterogeneity (RE only) CSV: {CSV_HETEROG}")

    print("\n=== DONE ===")

if __name__ == "__main__":
    main()

