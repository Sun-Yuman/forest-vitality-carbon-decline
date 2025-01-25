"""""""""""""""""""""""""""""""""""
This script processes data loading and merging, followed by cleaning, further processing, and saving the cleaned datasets.

1. Load raw data and merge with additional datasets
2. Clean the data by removing rows(code_defoliation) with missing or invalid values
3: Exclude countries that have fewer than 30 unique 'survey_year' entries
4. Remove species ID changed across monitoring years and unknown species
5. Retain the values from the beginning to the first occurrence of code_defoliation with a value of 100
6. Remove the dataset by removing specific countries (Broadleaves in Norway and Sweden)
7. Identify and remove duplicate rows
8: Assign tree species codes to species in English names('English_names')
9. Assign plot age based on the code values
10. Remain specific columns from the dataset while discarding unnecessary ones
"""""""""""""""""""""""""""""""""""

import pandas as pd
import numpy as np
# # # # # # # # # # # # # # # # # # Step 1: Load raw data and merge with additional datasets # # # # # # # # # # # # # # # # # #
data1 = pd.read_csv('C:/Yuman/c1_tre.txt', sep=';', encoding='gbk')

# Print column names and dimensions
print(data1.columns)
print(data1.shape[1])  # Number of columns
print(len(data1))  # Number of rows

# Load additional datasets
data6 = pd.read_csv('C:/Yuman/biogeo_reg_l109_c2.csv', header=0)
data7 = pd.read_csv('C:/Yuman/y1_pl1_c2.csv', header=0)
data8 = pd.read_csv('C:/Yuman/y1_st1_c2.csv', header=0)
data9 = pd.read_csv('C:/Yuman/d_tree_spec_c2.csv', header=0)
data10 = pd.read_csv('C:/Yuman/d_country_c2.csv', header=0)

# Merge data1 with additional datasets
data1 = pd.merge(data1, data6[['code_country', 'code_plot', 'biogeo_reg']], on=['code_country', 'code_plot'], how='left')
print("Rows after merging with data6:", len(data1))
data1 = pd.merge(data1, data7[['code_country', 'code_plot', 'latitude', 'longitude', 'y', 'x', 'code_altitude']], on=['code_country', 'code_plot'], how='left')
print("Rows after merging with data7:", len(data1))
data1 = pd.merge(data1, data8[['code_country', 'code_plot', 'survey_year', 'code_mean_age']], on=['code_country', 'code_plot', 'survey_year'], how='left')
print("Rows after merging with data8:", len(data1))
data1 = pd.merge(data1, data9[['code_tree_species', 'grp_tree_species']], on=['code_tree_species'], how='left')
print("Rows after merging with data9:", len(data1))
data1 = pd.merge(data1, data10[['code_country', 'lib_country', 'code_iso']], on=['code_country'], how='left')
print("Rows after merging with data10:", len(data1))

# Save the merged dataset
data1.to_csv('C:/Yuman/c1_treconn.txt', sep='\t', encoding='gbk', index=False)
print(data1.shape[1])  # Number of columns
print(len(data1))  # Number of rows

# Check for missing values in 'biogeo_reg'
na_count = data1['biogeo_reg'].isna().sum()
print(na_count)

# Count unique combinations of 'code_country' and 'code_plot'
country_plot_combinations = data1.groupby(['code_country', 'code_plot']).ngroups
print(f"Unique combinations of code_country and code_plot: {country_plot_combinations}")

# Count unique combinations of 'code_country', 'code_plot', and 'tree_number'
country_plot_tree_combinations = data1.groupby(['code_country', 'code_plot', 'tree_number']).ngroups
print(f"Unique combinations of code_country, code_plot, and tree_number: {country_plot_tree_combinations}")



# # # # # # # # # # # # # # # # # # Step 2: Clean the data by removing rows(code_defoliation) with missing or invalid values # # # # # # # # # # # # # # # # # #
data = pd.read_csv('C:/Yuman/c1_treconn.txt', sep='\t', header=0, encoding='utf-8')
print(data.shape[1])  # Number of columns
print(len(data))  # Number of rows

# Remove rows where 'code_defoliation' is missing (NA or blank) and keep rows with 'survey_year' between 1990 and 2022
data.dropna(subset=['code_defoliation'], inplace=True)
data = data[data['code_defoliation'] != -1]
data = data[(data['survey_year'] >= 1990) & (data['survey_year'] <= 2022)]

# Save the cleaned data
data.to_csv('C:/Yuman/c1_tre_valid.txt', sep='\t', encoding='utf-8', index=False)

# Check for missing values in 'biogeo_reg'
na_count = data['biogeo_reg'].isna().sum()
print(na_count)

# Print dataset dimensions after cleaning
print(data.shape[1])  # Number of columns
print(len(data))  # Number of rows

# Count unique combinations of 'code_country' and 'code_plot'
country_plot_combinations = data.groupby(['code_country', 'code_plot']).ngroups
print(f"Unique combinations of code_country and code_plot: {country_plot_combinations}")

# Count unique combinations of 'code_country', 'code_plot', and 'tree_number'
country_plot_tree_combinations = data.groupby(['code_country', 'code_plot', 'tree_number']).ngroups
print(f"Unique combinations of code_country, code_plot, and tree_number: {country_plot_tree_combinations}")



# # # # # # # # # # # # # # # # # # Step 3: Exclude countries that have fewer than 30 unique 'survey_year' entries # # # # # # # # # # # # # # # # # #
data = pd.read_csv('C:/Yuman/c1_tre_valid.txt', sep='\t', header=0, encoding='utf-8')
grouped_data = data.groupby('code_country')['survey_year'].nunique()
countries_to_remove = grouped_data[grouped_data < 30].index.tolist()
data = data[~data['code_country'].isin(countries_to_remove)]

# Save the data after removing specific countries
data.to_csv('C:/Yuman/c1_tre_valid_19Ctry.txt', sep='\t', encoding='utf-8', index=False)
print(data['code_country'].unique())  #19

# Check for missing values in 'biogeo_reg'
na_count = data['biogeo_reg'].isna().sum()
print(na_count)

# Print dataset dimensions
print(data.shape[1])  # Number of columns
print(len(data))  # Number of rows

# Count unique combinations of 'code_country' and 'code_plot'
country_plot_combinations = data.groupby(['code_country', 'code_plot']).ngroups
print(f"Unique combinations of code_country and code_plot: {country_plot_combinations}")

# Count unique combinations of 'code_country', 'code_plot', and 'tree_number'
country_plot_tree_combinations = data.groupby(['code_country', 'code_plot', 'tree_number']).ngroups
print(f"Unique combinations of code_country, code_plot, and tree_number: {country_plot_tree_combinations}")



# # # # # # # # # # # # # # # # # # Step 4: Remove species ID changed across monitoring years and unknown species # # # # # # # # # # # # # # # # # #
data = pd.read_csv('C:/Yuman/c1_tre_valid_19Ctry.txt', sep='\t', encoding='gbk')
grouped = data.groupby(['code_country', 'code_plot', 'tree_number'])
species_counts = grouped['code_tree_species'].nunique().reset_index()

# Merge the counts back to the original data
merged_data = data.merge(species_counts, on=['code_country', 'code_plot', 'tree_number'], how='left', suffixes=('', '_count'))

# Split data based on 'code_tree_species_count' value
df1 = merged_data[merged_data['code_tree_species_count'] == 1].drop(columns=['code_tree_species_count'])
df2 = merged_data[merged_data['code_tree_species_count'] > 1].drop(columns=['code_tree_species_count'])

# Remove unknown tree species
delete_unk = df1[df1['code_tree_species'] != 0]
delete_unk.to_csv('C:/Yuman/allsp_sameDunksp.txt', sep='\t', encoding='gbk', index=False)

# Print dataset dimensions after removing unknown species
print(delete_unk.shape[1])  # Number of columns
print(len(delete_unk))  # Number of rows

# Check for missing values in 'biogeo_reg'
na_count = delete_unk['biogeo_reg'].isna().sum()
print(na_count)

# Count unique combinations of 'code_country' and 'code_plot'
country_plot_combinations = delete_unk.groupby(['code_country', 'code_plot']).ngroups
print(f"Unique combinations of code_country and code_plot: {country_plot_combinations}")

# Count unique combinations of 'code_country', 'code_plot', and 'tree_number'
country_plot_tree_combinations = delete_unk.groupby(['code_country', 'code_plot', 'tree_number']).ngroups
print(f"Unique combinations of code_country, code_plot and tree_number: {country_plot_tree_combinations}")

# # # # # # # # # # # # # # # # # # Step 5: Further split data based on the presence of code_defoliation 100 # # # # # # # # # # # # # # # # # #
# Retain the values from the beginning to the first occurrence of code_defoliation with a value of 100
data = pd.read_csv('C:/Yuman/allsp_sameDunksp.txt', sep='\t', encoding='gbk')

# Define processing function for each group
def process_group(group):
    group.sort_values(by='survey_year', inplace=True)
    if 100 in group['code_defoliation'].values:
        first_100_index = np.where(group['code_defoliation'].values == 100)[0][0]
        return group.iloc[:first_100_index + 1], group.iloc[first_100_index + 1:]
    else:
        return group, pd.DataFrame()

# Apply processing function and collect results
results = data.groupby(['code_country', 'code_plot', 'tree_number']).apply(lambda x: process_group(x))
df1_list = [result[0] for result in results if not result[0].empty]
df2_list = [result[1] for result in results if not result[1].empty]

# Combine results
df1 = pd.concat(df1_list)
df2 = pd.concat(df2_list)

# Save results
df1.to_csv('C:/Yuman/allspRR.txt', sep='\t', encoding='gbk', index=False)
#df2.to_csv('C:/Yuman/allspDD.csv', encoding='gbk', index=False)

# Print dataset dimensions after splitting
print(df1.shape[1])  # Number of columns
print(len(df1))  # Number of rows

# Check for missing values in 'biogeo_reg'
na_count = df1['biogeo_reg'].isna().sum()
print(na_count)

# Count unique combinations of 'code_country' and 'code_plot'
country_plot_combinations = df1.groupby(['code_country', 'code_plot']).ngroups
print(f"Unique combinations of code_country and code_plot: {country_plot_combinations}")

# Count unique combinations of 'code_country', 'code_plot', and 'tree_number'
country_plot_tree_combinations = df1.groupby(['code_country', 'code_plot', 'tree_number']).ngroups
print(f"Unique combinations of code_country, code_plot and tree_number: {country_plot_tree_combinations}")



# # # # # # # # # # # # # # # # # #  Step 6. Clean the dataset by removing specific countries (Broadleaves in Norway and Sweden) # # # # # # # # # # # # # # # # # #
data = pd.read_csv('C:/Yuman/allspRR.txt', sep='\t', encoding='gbk')

# Identify rows to delete
to_delete = data[(data['code_country'].isin([13, 55])) & (data['grp_tree_species'] == 'broadleaves')]

# Remove identified rows
data = data[~((data['code_country'].isin([13, 55])) & (data['grp_tree_species'] == 'broadleaves'))]

# Save the cleaned data
data.to_csv('C:/Yuman/allsp_clean.txt', sep='\t', encoding='gbk', index=False)
#to_delete.to_csv('C:/Yuman/allspD_clean.csv', encoding='gbk', index=False)

# Print dataset dimensions after cleaning
print("Number of columns:", data.shape[1])
print("Number of rows:", len(data))

# Check for missing values in 'biogeo_reg'
na_count = data['biogeo_reg'].isna().sum()
print("Number of missing values in 'biogeo_reg':", na_count)

# Count unique combinations of 'code_country' and 'code_plot'
country_plot_combinations = data.groupby(['code_country', 'code_plot']).ngroups
print(f"Unique combinations of code_country and code_plot: {country_plot_combinations}")

# Count unique combinations of 'code_country', 'code_plot', and 'tree_number'
country_plot_tree_combinations = data.groupby(['code_country', 'code_plot', 'tree_number']).ngroups
print(f"Unique combinations of code_country, code_plot, and tree_number: {country_plot_tree_combinations}")



# # # # # # # # # # # # # # # # # # Step 7: Identify and remove duplicate rows # # # # # # # # # # # # # # # # # #
data = pd.read_csv('C:/Yuman/allsp_clean.txt', sep='\t', low_memory=False)

# Identify duplicates
grouped_data1 = data.groupby(['code_country', 'code_plot', 'survey_year', 'tree_number']).size().reset_index(name='count')
duplicates1 = grouped_data1[grouped_data1['count'] > 1]

# Collect duplicate rows
duplicates_list = []
for _, row in duplicates1.iterrows():
    mask = (data['code_country'] == row['code_country']) & \
           (data['code_plot'] == row['code_plot']) & \
           (data['survey_year'] == row['survey_year']) & \
           (data['tree_number'] == row['tree_number'])
    duplicates_list.append(data[mask])

# Concatenate duplicates
if duplicates_list:
    duplicates_data = pd.concat(duplicates_list, ignore_index=True)
else:
    duplicates_data = pd.DataFrame()

# Save duplicates to a file
duplicates_data.to_csv('C:/Yuman/Dduplicate_row.csv', index=False)

# Remove duplicates and keep the first occurrence
unique_data = data.drop_duplicates(subset=['code_country', 'code_plot', 'survey_year', 'tree_number'], keep='first')

# Save the unique data
unique_data.to_csv('C:/Yuman/allsp.txt', sep='\t', index=False)

# Print dataset dimensions after removing duplicates
print(unique_data.shape[1])  # Number of columns
print(len(unique_data))  # Number of rows

# Check for missing values in 'biogeo_reg'
na_count = unique_data['biogeo_reg'].isna().sum()
print(na_count)

# Count unique combinations of 'code_country' and 'code_plot'
country_plot_combinations = unique_data.groupby(['code_country', 'code_plot']).ngroups
print(f"Unique combinations of code_country and code_plot: {country_plot_combinations}")

# Count unique combinations of 'code_country', 'code_plot', and 'tree_number'
country_plot_tree_combinations = unique_data.groupby(['code_country', 'code_plot', 'tree_number']).ngroups
print(f"Unique combinations of code_country, code_plot, and tree_number: {country_plot_tree_combinations}")

# # # # # # # # # # # # # # # # # # Step 8: Map tree species codes to English names # # # # # # # # # # # # # # # # # #
species_mapping = {
    20: 'Beech',
    41: 'Oak',
    48: 'Oak',
    49: 'Oak',
    51: 'Oak',
    98: 'Oak',
    100: 'Silver fir',
    118: 'Norway spruce',
    129: 'Black pine',
    130: 'Maritime pine',
    134: 'Scots pine',
    136: 'Douglas fir',
}

# Read the data
data1 = pd.read_csv('C:/Yuman/allsp.txt', sep='\t', encoding='utf-8')

# Assign the 'code_tree_species' to 'English_names'
data1['English_names'] = data1['code_tree_species'].map(species_mapping)

# Save the mapped data
data1.to_csv('C:/Yuman/allsp(8).txt', sep='\t', encoding='utf-8', index=False)

# Print dataset dimensions
print(data1.shape[1])  # Number of columns
print(len(data1))  # Number of rows
na_count = data1['biogeo_reg'].isna().sum()
print(na_count)
country_plot_combinations = data1.groupby(['code_country', 'code_plot']).ngroups
print(f"Unique combinations of code_country and code_number: {country_plot_combinations}")

# Count unique combinations of 'code_country', 'code_plot', and 'tree_number'
country_plot_tree_combinations = data1.groupby(['code_country', 'code_plot', 'tree_number']).ngroups
print(f"Unique combinations of code_country, code_plot and tree_number: {country_plot_tree_combinations}")

#  Calculate the number of NA entries in the 'English_names' column
count_non_empty = data1['English_names'].notnull().sum()
print(f"The number of non-empty entries in 'English_names' is: {count_non_empty}")



# # # # # # # # # # # # # # # # # # Step 9: Assign plot age based on the code values # # # # # # # # # # # # # # # # # #
def assign_plotage(value):
    if value == 1:
        return 10
    elif value == 2:
        return 30
    elif value == 3:
        return 50
    elif value == 4:
        return 70
    elif value == 5:
        return 90
    elif value == 6:
        return 110
    elif value == 7:
        return 130

# Assigning the age values based on code_mean_age columns
data1['plot_age'] = data1['code_mean_age'].apply(assign_plotage)

# Save the data
data1.to_csv('C:/Yuman/allsp(8)age.txt', sep='\t', index=False)

# Print dataset dimensions after processing
print(data1.shape[1])  # Number of columns
print(len(data1))  # Number of rows
na_count = data1['biogeo_reg'].isna().sum()
print(na_count)
country_plot_combinations = data1.groupby(['code_country', 'code_plot']).ngroups
print(f"Unique combinations of code_country and code_number: {country_plot_combinations}")

# Count unique combinations of 'code_country', 'code_plot', and 'tree_number'
country_plot_tree_combinations = data1.groupby(['code_country', 'code_plot', 'tree_number']).ngroups
print(f"Unique combinations of code_country, code_plot and tree_number: {country_plot_tree_combinations}")



# # # # # # # # # # # # # # # # # # Step 10: Remain specific columns from the dataset while discarding unnecessary ones # # # # # # # # # # # # # # # # # #
file_path = 'C:/Yuman/allsp(8)age.txt'
data = pd.read_csv(file_path, sep='\t')

# Define the list of columns to keep
columns_to_keep = [
    'code_country', 'code_plot', 'tree_number', 'survey_year','code_defoliation',
    'code_tree_species',  'latitude', 'longitude', 'y', 'x',
    'code_altitude', 'grp_tree_species', 'lib_country', 'code_iso','biogeo_reg', 'English_names', 'plot_age'
]

# Filter the DataFrame to keep only the specified columns
data_filtered = data[columns_to_keep]

# Print the names of the retained columns
print("Retained column names:")
print(data_filtered.columns.tolist())

# Print the total number of rows in the filtered DataFrame
print("\nTotal number of rows after filtering:")  #2688512
print(len(data_filtered))
# Save the processed data
data_filtered.to_csv('C:/Yuman/allsp(8)age.txt', sep='\t', index=False)

