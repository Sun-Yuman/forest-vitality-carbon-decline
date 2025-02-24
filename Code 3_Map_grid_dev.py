"""""""""""""""""""""""""""""""""""
This script processes geographic and defoliation data (Defoliation, % deviation from 1990-2019 LTA.)to create and analyze a spatial grid (1 by 1).

1. Create a grid within specified longitude and latitude ranges for Europe.
   - Longitude range: -20 to 30
   - Latitude range: 25 to 70
   - Grid spacing: 1 degree
2. Read original shapefile of biogeoregions and CSV data of defoliation.
3. Adjust integer x and y values slightly to avoid issues in grid matching.
4. Convert data points to a GeoDataFrame and spatially join them with the grid.
5. Save the created grid and matched data with grid IDs to files.
6. Compute average and deviation for specified fields over different time periods:
   - Period A: 1990 to 1999
   - Period B: 2000 to 2009
   - Period C: 2010 to 2019
   - Period D: 2020 to 2022
   - Period E: 1990 to 2022 (entire period)
7. Create and save donut charts to visualize data percentages for various categories (percentage of plot).
8. Create and save horizontal stacked bar charts to visualize data deviations for various categories.
9. Clip data to biogeoregions and save the results.
10. Simplify geometries and process data in batches to create final shapefiles.
"""""""""""""""""""""""""""""""""""

import os
import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.geometry import Polygon, Point
import matplotlib.pyplot as plt

# Function to create a grid
def create_grid(lon_min, lon_max, lat_min, lat_max, spacing):
    cols = list(np.arange(lon_min, lon_max, spacing))
    rows = list(np.arange(lat_min, lat_max, spacing))
    polygons = []
    gridid = 0
    for lon in cols:
        for lat in rows:
            polygons.append({
                'geometry': Polygon([(lon, lat), (lon + spacing, lat),
                                     (lon + spacing, lat + spacing), (lon, lat + spacing)]),
                'gridid': gridid
            })
            gridid += 1
    grid = gpd.GeoDataFrame(polygons)
    return grid

# Reading original shapefile
BASE_DIR = 'C:/mapgriddev'


input_shp_path = os.path.join(BASE_DIR, 'biogeoreg.shp')
biogeoreg_gdf = gpd.read_file(input_shp_path)
biogeoreg_crs = biogeoreg_gdf.crs

# Defining grid parameters
lon_min, lon_max, lat_min, lat_max = -20, 30, 25, 70  # Europe range
spacing = 1

# Creating grid and converting coordinate system
grid_gdf = create_grid(lon_min, lon_max, lat_min, lat_max, spacing)
grid_gdf.set_crs(epsg=4326, inplace=True)
grid_gdf.to_crs(crs=biogeoreg_crs, inplace=True)

# Saving the grid with gridid to a Shapefile
grid_output_shp_path = os.path.join(BASE_DIR, 'grid_plot_gridid.shp')
grid_gdf.to_file(grid_output_shp_path)

# Reading and processing data
data_path = os.path.join(BASE_DIR, 'alldefoliation.csv')
data = pd.read_csv(data_path, low_memory=False)

# Adjusting integer x and y values slightly
data['x'] = data['x'].apply(lambda x: x if x % 1 != 0 else x + 0.00001)
data['y'] = data['y'].apply(lambda y: y if y % 1 != 0 else y + 0.00001)

# Converting data points to a GeoDataFrame
data['geometry'] = data.apply(lambda row: Point(row['x'], row['y']), axis=1)
data_gdf = gpd.GeoDataFrame(data, geometry='geometry')
data_gdf.set_crs(epsg=4326, inplace=True)
data_gdf.to_crs(crs=biogeoreg_crs, inplace=True)

# Matching data points with the grid
joined = gpd.sjoin(data_gdf, grid_gdf, how='left', op='within')

# Saving the data with gridid
output_path = os.path.join(BASE_DIR, 'alldefoliationgrid.csv')
joined.to_csv(output_path, index=False)

# Diagnostics
# Checking if original data points are within grid bounds
grid_bounds = grid_gdf.total_bounds
out_of_bounds = data_gdf[~data_gdf.within(Polygon([
    (grid_bounds[0], grid_bounds[1]),
    (grid_bounds[2], grid_bounds[1]),
    (grid_bounds[2], grid_bounds[3]),
    (grid_bounds[0], grid_bounds[3])
]))]

print(f"Number of points out of grid bounds: {len(out_of_bounds)}")

# Checking if there are rows with NaN gridid after sjoin
no_gridid = joined[joined['gridid'].isna()]
print(f"Number of points without gridid: {len(no_gridid)}")
if len(no_gridid) > 0:
    print(no_gridid[['x', 'y']].head())

# More diagnostics
print(joined.shape[1])  # Columns
print(len(joined))  # Rows
na_count = joined['biogeo_reg'].isna().sum()
print(na_count)
country_plot_combinations = joined.groupby(['code_country', 'code_plot']).ngroups
print(f"Unique combinations of code_country and code_plot: {country_plot_combinations}")

# Importing data for further analysis
data_path = os.path.join(BASE_DIR, 'alldefoliationgrid.csv')
data = pd.read_csv(data_path, low_memory=False)

# Defining time periods
time_periods = {
    'A': (1990, 1999),
    'B': (2000, 2009),
    'C': (2010, 2019),
    'D': (2020, 2022),
    'E': (1990, 2022)
}

# List of fields to process
fields = [
    'mean_defoliation', 'broadleaves', 'conifers', 'Beech', 'Norway spruce', 'Oak', 'Scots pine'
]

# Calculating baseline (1990-2019 average) for each field
baseline_dict = {}
baseline_data = data.query('1990 <= survey_year <= 2019')
for field in fields:
    baseline_dict[field] = baseline_data[field].mean()

# Initializing final_df to store results for each grid
final_df = pd.DataFrame(data['gridid'].unique(), columns=['gridid'])
all_period_data = pd.DataFrame()

# Calculating average and deviation for each time period and storing in all_period_data
for period, (start, end) in time_periods.items():
    for field in fields:
        period_data = data.query(f'{start} <= survey_year <= {end}')
        period_avg = period_data[['code_country', 'code_plot', 'survey_year', 'gridid', field]].copy()
        period_avg[f'{period}_{field}_avg'] = period_avg[field]
        period_avg[f'baseline_1990-2019_{field}'] = baseline_dict[field]
        period_avg[f'{period}_{field}_dev'] = period_avg[f'{period}_{field}_avg'] - period_avg[f'baseline_1990-2019_{field}']
        period_avg = period_avg.drop(columns=[field])
        if all_period_data.empty:
            all_period_data = period_avg
        else:
            all_period_data = pd.merge(all_period_data, period_avg, on=['code_country', 'code_plot', 'survey_year', 'gridid'], how='outer', suffixes=(None, f'_{period}'))

# Computing averages by grid and merging into final_df
for period, (start, end) in time_periods.items():
    for field in fields:
        grid_mean_avg = all_period_data.groupby('gridid')[f'{period}_{field}_avg'].mean().reset_index()
        grid_mean_dev = all_period_data.groupby('gridid')[f'{period}_{field}_dev'].mean().reset_index()
        final_df = final_df.merge(grid_mean_avg, on='gridid', how='left')
        final_df = final_df.merge(grid_mean_dev, on='gridid', how='left')

# Calculating counts and merging into final_df
for period, (start, end) in time_periods.items():
    period_data = data.query(f'{start} <= survey_year <= {end}')
    period_data_grouped = period_data.groupby('gridid')
    for field in fields:
        cnt1 = period_data_grouped.apply(lambda x: x.groupby(['code_country', 'code_plot']).ngroups).reset_index()
        cnt1.rename(columns={0: f'{period}_{field}_cnt1'}, inplace=True)
        final_df = final_df.merge(cnt1, on='gridid', how='left')

# Computing averages and deviations by plot
country_plot_avg_df = pd.DataFrame()
for period, (start, end) in time_periods.items():
    for field in fields:
        period_data = data.query(f'{start} <= survey_year <= {end}')
        period_avg = period_data.groupby(['code_country', 'code_plot'])[field].mean().reset_index()
        period_avg.rename(columns={field: f'{period}_{field}_avg'}, inplace=True)
        period_data[f'{field}_dev'] = period_data[field] - baseline_dict[field]
        period_dev = period_data.groupby(['code_country', 'code_plot'])[f'{field}_dev'].mean().reset_index()
        period_dev.rename(columns={f'{field}_dev': f'{period}_{field}_dev'}, inplace=True)
        if country_plot_avg_df.empty:
            country_plot_avg_df = period_avg.merge(period_dev, on=['code_country', 'code_plot'], how='outer')
        else:
            country_plot_avg_df = country_plot_avg_df.merge(period_avg, on=['code_country', 'code_plot'], how='outer')
            country_plot_avg_df = country_plot_avg_df.merge(period_dev, on=['code_country', 'code_plot'], how='outer')

# Exporting all period data
BASE_DIR = 'C:/mapgriddev'

all_period_output_path = os.path.join(BASE_DIR, '3_periods_merged_data_all.csv')
all_period_data.to_csv(all_period_output_path, index=False)

# Exporting final data
final_output_path = os.path.join(BASE_DIR, '2_5period_gridplotlevel.csv')
final_df.fillna(-9999).to_csv(output_path, index=False)

# Exporting country and plot average and deviation data
country_plot_output_path = os.path.join(BASE_DIR, '4_plot_avgdev.csv')
country_plot_avg_df.fillna(-9999).to_csv(country_plot_output_path, index=False)

# Reading data for further analysis
file_path = os.path.join(BASE_DIR, '4_plot_avgdev_data.csv')
df = pd.read_csv(file_path)

df_filtered = df[df != -9999]

# Print filtered data
print("Filtered data:")
print(df_filtered.head())

# Get columns containing "dev"
num_columns = [col for col in df_filtered.columns if 'dev' in col]

# Print columns containing "dev":
print("Columns containing 'dev':")
print(num_columns)

# Define bins
bins = [-np.inf, -12,-9,-6, -3, 0, 3,6,9, 12, np.inf]
labels = ["<-12", "-12 to -9","-9 to -6","-6 to -3", "-3 to 0", "0 to 3", "3 to 6", "6 to 9","9 to 12",">12"]

# Create DataFrames to store results
count_df = pd.DataFrame(index=labels)
percentage_df = pd.DataFrame(index=labels)

# Perform binning and statistics for each "dev" column
for col in num_columns:
    print(f"Processing column: {col}")

    # Create binned column
    df_filtered['bin'] = pd.cut(df_filtered[col], bins=bins, labels=labels, right=False)

    # Print binned data
    print(f"Binned data (first 5 rows):")
    print(df_filtered[['code_country', 'code_plot', col, 'bin']].head())

    # Count unique combinations in each bin
    count_series = df_filtered.groupby('bin').apply(
        lambda x: x[['code_country', 'code_plot']].drop_duplicates().shape[0])

    # Calculate percentages
    percentage_series = count_series / count_series.sum() * 100

    # Print statistics for current column
    print(f"Statistics for column {col}:")
    print(count_series)
    print(f"Percentage results for column {col}:")
    print(percentage_series)

    # Add results to DataFrames
    count_df[col] = count_series
    percentage_df[col] = percentage_series

# Save results to CSV files

count_output_file = os.path.join(BASE_DIR, '5_num_dev_bins_count.csv')
percentage_output_file = os.path.join(BASE_DIR, '5_num_dev_bins_percentage.csv')
count_df.to_csv(count_output_file)
percentage_df.to_csv(percentage_output_file)

print("Count data successfully saved to", count_output_file)
print("Percentage data successfully saved to", percentage_output_file)

plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 14

# Load data from CSV
file_path = os.path.join(BASE_DIR, '5_num_dev_bins_percentage.csv')

# Define colors and labels
colors = ['#4575b4', '#91bfdb', '#e0f3f8', '#fee090', '#fc8d59', '#d73027']
labels = ["<-12", "-12 to -9","-9 to -6","-6 to -3", "-3 to 0", "0 to 3", "3 to 6", "6 to 9","9 to 12",">12"]
time_periods = ["A", "B", "C", "D"]

# Function to create and save horizontal stacked bar charts
def save_horizontal_category_stacked_bar_limited_x_v3(period, df, labels, colors, save_path):
    fig, ax = plt.subplots(figsize=(6, 2))  # Adjusted figure size for better visibility

    # Extract data for the given period
    _deviation = df[f"{period}_mean_defoliation_dev"].tolist()
    broadleaves_deviation = df[f"{period}_broadleaves_dev"].tolist()
    conifers_deviation = df[f"{period}_conifers_dev"].tolist()

    # Adjust the bar positions to reduce spacing
    categories = ['A', 'B', 'C']
    bar_width = -0.16
    bar_positions = np.arange(len(categories)) * (1 + bar_width)

    # Function to display the values on the bars
    def display_values(lefts, deviations, category_position):
        accumulated = 0
        for i, deviation in enumerate(deviations):
            width = accumulated + deviation / 2
            accumulated += deviation

    # Plot total deviation and other categories
    for i, deviations in enumerate([_deviation, broadleaves_deviation, conifers_deviation]):
        lefts = 0
        for deviation, color, label in zip(deviations, colors, labels):
            ax.barh(bar_positions[i], deviation, left=lefts, color=color)
            lefts += deviation
        display_values([lefts - val for val in deviations], deviations, bar_positions[i])

    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_yticklabels([])
    ax.set_yticks(bar_positions)
    ax.set_yticklabels(categories)
    ax.set_xlim(0, 100)  # Set x-axis limits
    ax.tick_params(axis='x', pad=5)

    # Save the plot
    plt.tight_layout()
    plt.savefig(save_path, format='jpg')
    plt.close()

# Function to create and save donut charts
def save_donut_chart(data, colors, save_path):
    fig, ax = plt.subplots(figsize=(6, 6))
    wedges, texts = ax.pie(data, colors=colors, startangle=120, wedgeprops=dict(width=0.5))

    for i, wedge in enumerate(wedges):
        angle = (wedge.theta2 + wedge.theta1) / 2
        x = wedge.r * 0.85 * np.cos(np.radians(angle))
        y = wedge.r * 0.85 * np.sin(np.radians(angle))
        ax.text(x, y, f'{int(data[i])}%', ha='center', va='center', fontsize=22)  # 调整 fontsize

    ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    plt.tight_layout()
    plt.savefig(save_path, format='jpg')
    plt.close()

# Generate and save the horizontal stacked bar charts
saved_paths_v3 = []
for period in time_periods:
    filename = f"3_{period.replace('-', '_')}_BC.jpg"
    save_horizontal_category_stacked_bar_limited_x_v3(period, df, labels, colors, filename)
    saved_paths_v3.append(filename)

print("Horizontal stacked bar charts saved:", saved_paths_v3)

# Generate and save the donut charts
saved_paths_donut = []
for period in time_periods:
    _deviation = df[f"{period}_mean_defoliation_dev"].tolist()
    broadleaves_deviation = df[f"{period}_broadleaves_dev"].tolist()
    conifers_deviation = df[f"{period}_conifers_dev"].tolist()
    Beech_deviation = df[f"{period}_Beech_dev"].tolist()
    Oak_deviation = df[f"{period}_Oak_dev"].tolist()
    Norway_spruce_deviation = df[f"{period}_Norway spruce_dev"].tolist()
    Scots_pine_deviation = df[f"{period}_Scots pine_dev"].tolist()

    for category, data in zip(['total', 'broadleaves', 'conifers', 'Beech', 'Oak', 'Norway spruce', 'Scots pine'],
                              [_deviation, broadleaves_deviation, conifers_deviation, Beech_deviation, Oak_deviation,
                               Norway_spruce_deviation, Scots_pine_deviation]):
        filename = f"donut_{period}_{category}.jpg"
        save_donut_chart(data, colors, filename)
        saved_paths_donut.append(filename)

# Reading data for clipping
grid_output_shp_path = os.path.join(BASE_DIR, 'grid_plot_gridid.shp')
data_path = os.path.join(BASE_DIR, '2_5period_gridplotlevel.csv')
input_shp_path = os.path.join(BASE_DIR, 'biogeoreg.shp')
grid_gdf = gpd.read_file(grid_output_shp_path)
data = pd.read_csv(data_path, low_memory=False)
biogeoreg_gdf = gpd.read_file(input_shp_path)

# Convert biogeoreg_gdf to GCS_ETRS_1989
grid_gdf = grid_gdf.to_crs(epsg=4258)
biogeoreg_gdf = biogeoreg_gdf.to_crs(epsg=4258)

# Identify all prefixes
suffixes = ['avg', 'dev', 'cnt1']
prefixes = set(col.rsplit('_', 1)[0] for col in data.columns if any(col.endswith(f'_{suffix}') for suffix in suffixes))

# Function to simplify geometries
def simplify_geometry(gdf, tolerance=0.01):
    gdf['geometry'] = gdf['geometry'].simplify(tolerance, preserve_topology=True)
    return gdf

# Function for batch clipping
def batch_clip(gdf, clip_gdf, batch_size=1000):
    clipped = []
    for i in range(0, len(gdf), batch_size):
        batch = gdf.iloc[i:i+batch_size]
        clipped.append(gpd.clip(batch, clip_gdf))
    return gpd.GeoDataFrame(pd.concat(clipped))

# Processing each prefix
for prefix in prefixes:
    print(f"\nProcessing prefix: {prefix}")
    columns_to_select = [f'{prefix}_{s}' for s in suffixes if f'{prefix}_{s}' in data.columns]
    selected_data = data[['gridid'] + columns_to_select].copy()
    selected_data.replace(-9999, np.nan, inplace=True)
    selected_data.dropna(subset=columns_to_select, inplace=True)
    if selected_data.empty:
        continue
    merged_gdf = grid_gdf.merge(selected_data, on='gridid', how='left')
    if merged_gdf.empty:
        continue
    merged_gdf = merged_gdf.to_crs(epsg=4258)
    merged_gdf = merged_gdf.replace(-9999, np.nan).dropna()
    if merged_gdf.empty:
        continue
    merged_gdf = simplify_geometry(merged_gdf)
    clipped_gdf = batch_clip(merged_gdf, biogeoreg_gdf)
    if clipped_gdf.empty:
        continue
    output_folder = os.path.join(BASE_DIR, 'plotlevelout_allbase')
    output_path = os.path.join(BASE_DIR, f"{prefix}_plotall.shp")
    clipped_gdf.to_file(output_path)
    print(f"Output shapefile created for prefix {prefix} at {output_path}")

print("Shapefiles have been successfully created.")
