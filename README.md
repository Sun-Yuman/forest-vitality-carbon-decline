# forest vitality-Europe
Code for the article 'Tree vitality decline': data preprocessing, analysis, and modeling scripts
README_code
"""""""""""""""""""""""""""""""""""
Code: 0_Data preprocessing.py

Python version: Python 3.11.5
Package: numpy 1.26.4, pandas 2.2.2

"""""""""""""""""""""""""""""""""""
 Summary: This script processes data loading and merging, followed by cleaning, further processing, and saving the cleaned datasets.

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
"""""""""""""""""""""""""""""""""""
1_Frequence(KS).py

Python version: Python 3.11.5
Package: numpy 1.26.4, pandas 2.2.2, matplotlib 3.8.4, seaborn 0.13.2, scipy 1.13.0

"""""""""""""""""""""""""""""""""""
 Summary: This script processes defoliation data files and generates combined plots of frequency distribution and Kolmogorov-Smirnov (KS) statistics heatmap during four time periods.

1. Defines the time periods for analysis.e.g.1990-1999, 2000-2009, 2010-2019, 2020-2022
2. Generates frequency distribution plots smoothed by Gaussian filters.
3. Computes KS statistics between different time periods and visualizes them in a heatmap.
4. Processes data files in different groups based on function group (Broadleaves, conifers and single species) and biogeographic_regions.
"""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""
2_Trend_Poly_Heatmap.py

Python version: Python 3.11.5
Package: numpy 1.26.4, pandas 2.2.2, matplotlib 3.8.4, seaborn 0.13.2, scipy 1.13.0, pymannkendall 1.4.3, sklearn 1.4.2

"""""""""""""""""""""""""""""""""""
 Summary:  This script processes defoliation data and generates combined plots with segmented trend analysis  and  heatmaps of biogeographic regions showing Defoliation, % deviation from 1990-2019 LTA.

1. Loads the defoliation data from a CSV file.
2. Defines functions for p-value to star conversion, polynomial fit addition, and trend plotting (Theil-Sen slope and Mann-Kendall trend).
3. Generates combined plots with segmented trend lines and  biogeographic regions-specific heatmaps.
4. Saves the resulting plots and baseline data to files.
5. Filters or processes data for these categories (e.g.,Conifers and broadleaves) to create separate plots.
6. Generates forest plots with trend analysis results (Theil-Sen slope and Mann-Kendall trend) over different periods and different function group.
"""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""
3_Map_grid_dev.py
Python version: Python 3.11.5
Package: numpy 1.26.4, pandas 2.2.2, matplotlib 3.8.4, geopandas 0.14.4, shapely 2.0.1

"""""""""""""""""""""""""""""""""""
This script processes geographic and defoliation data (Defoliation, % deviation from 1990-2019 LTA.) to create and analyze a spatial grid (1 by 1).

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
"""""""""""""""""""""""""""""""""""
4_RandomForest.py
Python version: Python 3.11.5
Package: numpy 1.26.4, pandas 2.2.2, matplotlib 3.8.4, scikit-learn 1.4.2, shap 0.45.0,scikit-optimize 0.10.2

"""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""
This script implements a Random Forest model for data preprocessing and optimization, incorporating spatial clustering, 
multicollinearity analysis, feature importance assessment, and generating explainable visual outputs like SHAP values and 
Partial Dependence Plots (PDPs). The primary focus is on predicting forest defoliation using nested cross-validation with the 
Random Forest model (Dev_all).

1. Data Loading and Preprocessing**:
   - Load data from a CSV file containing spatial, environmental, and biological variables.
   - Define target (`Dev_all`) and selected features for modeling.
   - Rename columns for better readability and usability.

2. Multicollinearity Analysis**:
   - Compute Variance Inflation Factors (VIF) for selected features.
   - Identify features with VIF > 5, which indicates potential multicollinearity issues.

3. Spatial Clustering**:
   - Apply KMeans clustering on longitude and latitude data to spatially group samples.
   - Visualize clustering results using 2D and 3D scatter plots.

4.Block Nested Cross-Validation with Hyperparameter Tuning**:
   - Use Bayesian Optimization (`BayesSearchCV`) to tune Random Forest hyperparameters.
   - Implement a custom scoring function to penalize overfitting by considering the difference between training and validation R² scores.
   - Conduct nested cross-validation to evaluate model performance using metrics such as R², RMSE, and MAE.
  
5. Model Training on Full Dataset and Interpretability**:
   - Train the final Random Forest model on the entire dataset using the best hyperparameters and selected features.
   - Generate and save feature importance plots to rank variable influence.
   - Create Partial Dependence Plots (PDPs) to visualize the relationships between features and the target variable.
   - Calculate SHAP values for detailed interpretability and generate summary and beeswarm plots.
  
7. Feature Importance and Selection**:
   - Calculate feature importance scores to prioritize variables.
   - Perform stepwise feature selection by iteratively removing less important variables and tracking performance.

8. Refined Model Training with Reduced Features**:
   - Retrain the Random Forest model on the reduced feature set with optimized hyperparameters.
   - Perform a final nested cross-validation to validate the optimized feature set.
  
9. Visual Outputs**:
   - Feature Importance: Save and visualize feature importance for model interpretability.
   - Partial Dependence Plots: Generate PDPs for the reduced feature set to understand the impact of each feature.
   - SHAP Analysis: Generate SHAP values, correlation metrics, and beeswarm plots to understand feature contributions.

10. Important output csv
    - Cross-validation results (`outer_cv_results.csv`, `inner_cv_results.csv`).
    - Selecte best hyperparameters(`best_params_overall.csv`).
    - Selected feature set and corresponding metrics (`selected_best_features.csv`).
    - Final model performance metrics (`average_reduced_features_results.csv`).
"""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""
5_Growth.py
Python version: Python 3.11.5
Package: numpy 1.26.4, pandas 2.2.2, matplotlib 3.8.4, seaborn 0.13.2, sklearn 1.4.2, Statsmodels: 0.14.2

"""""""""""""""""""""""""""""""""""
Summary: data analysis of growth reduction and defoliation using linear regression, cross-validation, sensitivity analysis, and Growth reduction based on 1990-1999.

1. Load and clean data by removing rows with missing values for 'Growth' and 'Defoliation'.
2. Perform linear regression analysis between 'Growth' and 'Defoliation' and assess statistical significance.
3. Conduct group-based 5 fold cross-validation to calculate average R² values.
4. Plot scatter plots with regression lines and confidence intervals for 'overall' and 'BC(Broadleaves and conifers)' categorical data.
5. Conduct sensitivity analysis by adding random bias to 'Defoliation' values and evaluate impact.
6. Process cumulative growth data by splitting groups into 'biogeographic regions' and ''BC(Broadleaves and conifers)' across different time periods.
7. Create bar charts to depict relative growth across different regions and time periods.
"""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""
6_Carbon.py
Python version: Python 3.11.5
Package: numpy 1.26.4, pandas 2.2.2, matplotlib 3.8.4, seaborn 0.13.2, sklearn 1.4.2, Statsmodels: 0.14.2

"""""""""""""""""""""""""""""""""""
This script performs several tasks related to the analysis of defoliation , net change,  and carbon sink data.

1.It starts by loading and merging multiple CSV files, then processes and cleans the data.
2.It calculates means and deviations from baselines, filters the data for specific years.
3.The script also includes various plotting routines to visualize the relationships between defoliation, net change, and carbon sink data.
4.It generates scatter plots, line plots, and bar plots with linear regression and confidence intervals, and applies polynomial fit for better visualization.
"""""""""""""""""""""""""""""""""""


