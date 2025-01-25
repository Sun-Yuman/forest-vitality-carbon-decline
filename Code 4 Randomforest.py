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

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from skopt import BayesSearchCV
from sklearn.cluster import KMeans
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score,make_scorer
from sklearn.inspection import PartialDependenceDisplay
from matplotlib.colors import LinearSegmentedColormap
import shap
import statsmodels.api as sm
from joblib import Parallel, delayed, Memory
from statsmodels.stats.outliers_influence import variance_inflation_factor
import warnings
warnings.filterwarnings('ignore')
from skopt.callbacks import EarlyStopper
# Load data

data_path = 'D:/Yuman/RF/RF_variables.csv'
data1 = pd.read_csv(data_path)
print("Data loaded from:", data_path)

# Define selected features
selected_features = [
    'dev_all', 'latitude', 'longitude', 'altitude', 'species_count', 'age',
    'Total', 'drought', 'biotic', 'abiotic',
    'SPEI24', 'SURF_ppb_O3', 'WDEP_SOX', 'NDEP', 'VPD9019', 'Temperature'
]

# Rename columns for convenience
feature_name_mapping = {
    'dev_all': 'Dev_all',
    'latitude': 'LAT',
    'longitude': 'LONG',
    'altitude': 'Elevation',
    'species_count': 'No.of species',
    'age': 'Age',
    'drought': 'Drought',
    'biotic': 'Biotic',
    'abiotic': 'Abiotic',
    'Total': 'Total',
    'SPEI24': 'SPEI24',
    'Temperature': 'Temp',
    'VPD9019': 'VPD anomaly',
    'SURF_ppb_O3': '[O₃]',
    'NDEP': 'N_dep',
    'WDEP_SOX': 'S_dep'
}

# Split data into features and target
data = data1[selected_features].rename(columns=feature_name_mapping)
y = data['Dev_all']
X = data.drop('Dev_all', axis=1)


def calculate_vif(X):
    # Add a constant column to calculate VIF
    X_with_constant = sm.add_constant(X)

    # Create a DataFrame to store the VIF results
    vif_data = pd.DataFrame()
    vif_data['Feature'] = X_with_constant.columns

    # Calculate VIF for each feature
    vif_data['VIF'] = [variance_inflation_factor(X_with_constant.values, i) for i in range(X_with_constant.shape[1])]

    return vif_data
vif_df = calculate_vif(X)

# Save VIF results to CSV
vif_df.to_csv('./vif_results.csv', index=False)
print("VIF results saved to vif_results.csv")

# Print features with VIF greater than 5
vif_greater_than_5 = vif_df[vif_df['VIF'] > 5]
print("\nFeatures with VIF greater than 5:")
print(vif_greater_than_5)
print("----------------------------------------------------------------Step1: variables xy---------------------------------------")


def kmeans_clustering_and_plotting(data, X, n_clusters=10):
    """
    Apply KMeans clustering to the spatial data (longitude and latitude) and
    plot the 2D and 3D results. Return the y_binned variable for further use.

    Parameters:
    - data: The input DataFrame with longitude, latitude, and survey_year columns.
    - X: The feature DataFrame (must include longitude and latitude).
    - n_clusters: The number of clusters for KMeans.

    Returns:
    - y_binned: The cluster labels obtained from KMeans.
    """
    # Apply KMeans clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=90)
    X['cluster'] = kmeans.fit_predict(data[['longitude', 'latitude']])

    # Assign cluster labels to y_binned
    y_binned = X['cluster']

    # Plot the 2D clustering result (longitude and latitude)
    fig, ax = plt.subplots(figsize=(7, 6))
    scatter = ax.scatter(data['longitude'], data['latitude'], c=X['cluster'], cmap='viridis')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    plt.colorbar(scatter, ax=ax, label='Cluster')
    plt.title('KMeans Clustering of Spatial Data (2D)')
    plt.savefig('./kmeans_clustering_2D.png')
    plt.show()
    print("KMeans clustering 2D plot saved to kmeans_clustering_2D.png.")

    # 3D plot (longitude, latitude, and survey_year)
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    scatter_3d = ax.scatter(data['longitude'], data['latitude'], data['survey_year'], c=X['cluster'], cmap='viridis')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_zlabel('Survey Year')
    plt.colorbar(scatter_3d, ax=ax, label='Cluster')
    plt.title('KMeans Clustering with Time (3D)')
    plt.savefig('./kmeans_clustering_3D.png')
    plt.show()
    print("KMeans clustering 3D plot saved to kmeans_clustering_3D.png.")

    return y_binned

y_binned = kmeans_clustering_and_plotting(data1, X)

if 'cluster' in X.columns:
    X= X.drop('cluster', axis=1)
    print("'cluster' column dropped.")
else:
    print("'cluster' column does not exist or has already been dropped.")
# Caching model training to avoid retraining the same model with the same parameters
memory = Memory(location='./cache_dir', verbose=0)

@memory.cache(ignore=['X_train', 'y_train'])
def fit_model(X_train, y_train, params):
    model = RandomForestRegressor(**params)
    model.fit(X_train, y_train)
    return model

# Custom scorer with penalty applied to inner cross-validation based on the difference between train and validation R² scores
def custom_scorer_with_logging(estimator, X, y, cv_splits=10):
    kf = KFold(n_splits=cv_splits, shuffle=True, random_state=90)

    # Parallelized K-Fold validation to speed up
    results = Parallel(n_jobs=-1)(delayed(evaluate_fold)(estimator, X, y, train_idx, val_idx)
                                  for train_idx, val_idx in kf.split(X))
    r2_train_scores, r2_val_scores = zip(*results)

    avg_r2_train = np.mean(r2_train_scores)
    avg_r2_val = np.mean(r2_val_scores)

    difference = avg_r2_train - avg_r2_val
    if difference > 0.35:
        penalty = abs(difference) * 8
    elif difference > 0.12:
        penalty = abs(difference) * 4
    else:
        penalty = 0

    penalized_r2 = avg_r2_val - penalty
    return penalized_r2


# Helper function to evaluate each fold during K-Fold cross-validation
def evaluate_fold(estimator, X, y, train_idx, val_idx):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    # Use cached model to avoid retraining
    trained_model = fit_model(X_train, y_train, estimator.get_params())

    # Predictions and R² scores
    y_train_pred = trained_model.predict(X_train)
    y_val_pred = trained_model.predict(X_val)

    r2_train = r2_score(y_train, y_train_pred)
    r2_val = r2_score(y_val, y_val_pred)

    return r2_train, r2_val


# Nested cross-validation with penalty applied only to inner cross-validation
def run_nested_cv_with_penalty(X, y, y_binned, random_forest_range, outer_cv, inner_cv, random_state=90, n_iter=20):
    random_forest_model = RandomForestRegressor()

    # Setup BayesSearchCV for hyperparameter tuning
    bayes_search = BayesSearchCV(
        estimator=random_forest_model,
        search_spaces=random_forest_range,
        n_iter=n_iter,
        cv=inner_cv,
        n_jobs=-1,  # Parallelized search
        n_points=20,
        verbose=1,
        random_state=random_state,
        scoring = lambda estimator, X_val, y_val: custom_scorer_with_logging(estimator, X_val, y_val)
        )

    # Outer cross-validation loop
    outer_cv_results = []
    best_params_list = []
    inner_cv_results_all = []

    for outer_fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(X, y_binned)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        # Perform inner cross-validation with BayesSearchCV
        bayes_search.fit(X_train, y_train)

        # Get best hyperparameters from inner cross-validation
        best_params = bayes_search.best_params_
        best_params_list.append(best_params)

        # Train the model with the best hyperparameters and evaluate on the test set
        best_model = RandomForestRegressor(**best_params, oob_score=True)
        best_model.fit(X_train, y_train)

        y_train_pred = best_model.predict(X_train)
        y_test_pred = best_model.predict(X_test)

        # Store results for the outer fold
        outer_results = {
            'Outer Fold': outer_fold_idx + 1,
            'Train R2': r2_score(y_train, y_train_pred),
            'Test R2': r2_score(y_test, y_test_pred),
            'Train RMSE': np.sqrt(mean_squared_error(y_train, y_train_pred)),
            'Test RMSE': np.sqrt(mean_squared_error(y_test, y_test_pred)),
            'Train MAE': mean_absolute_error(y_train, y_train_pred),
            'Test MAE': mean_absolute_error(y_test, y_test_pred),
            'OOB Score': best_model.oob_score_
        }
        outer_cv_results.append(outer_results)

        print(f"Outer Fold {outer_fold_idx + 1} results: {outer_results}")

        # Save inner cross-validation results (BayesSearchCV)
        inner_cv_results = pd.DataFrame(bayes_search.cv_results_)
        inner_cv_results['Outer Fold'] = outer_fold_idx + 1
        inner_cv_results['Param Set'] = range(1, len(inner_cv_results) + 1)
        inner_cv_results_all.append(inner_cv_results)

    return outer_cv_results, best_params_list, pd.concat(inner_cv_results_all)


# Function to select best hyperparameters across all outer folds
def select_best_params(params_df):
    selected_params = {}
    for column in params_df.columns:
        if column == 'bootstrap':
            selected_params[column] = params_df[column].mode()[0] == 1.0
        elif params_df[column].dtype == 'object':
            selected_params[column] = params_df[column].mode()[0]
        else:
            median_value = params_df[column].median()
            if column in ['n_estimators', 'max_depth', 'min_samples_split', 'min_samples_leaf']:
                selected_params[column] = int(median_value)
            else:
                selected_params[column] = median_value if median_value <= 1 else int(median_value)
    return selected_params


# Cross-validation setup
random_forest_range = {
    'n_estimators': (100, 1200),
    'max_features': ['sqrt', 'log2'],
    'max_depth': (3, 80),
    'min_samples_split': (15, 60),
    'min_samples_leaf': (15, 60),
    'bootstrap': [True]
}

outer_cv = KFold(n_splits=10, shuffle=True, random_state=90)
inner_cv = KFold(n_splits=10, shuffle=True, random_state=90)

# Run nested cross-validation
outer_cv_results, best_params_list, inner_cv_results_all = run_nested_cv_with_penalty(
    X, y, y_binned, random_forest_range, outer_cv, inner_cv
)

# Save outer CV results
outer_cv_results_df = pd.DataFrame(outer_cv_results)
outer_cv_results_df.to_csv('./outer_cv_results.csv', index=False)
print("Outer CV results saved to outer_cv_results.csv.")

# Save inner CV results
inner_cv_results_df = pd.DataFrame(inner_cv_results_all)
inner_cv_results_df.to_csv('./inner_cv_results.csv', index=False)
print("Inner CV results saved to inner_cv_results.csv.")

# Convert best hyperparameters list to DataFrame
best_params_df = pd.DataFrame(best_params_list)

# Select overall best hyperparameters
best_params_overall = select_best_params(best_params_df)
print(f"Final selected best hyperparameters: {best_params_overall}")

# Save final selected hyperparameters to CSV
best_params_overall_df = pd.DataFrame([best_params_overall])
best_params_overall_df.to_csv('./best_params_overall.csv', index=False)
print("Overall best hyperparameters saved to best_params_overall.csv.")

print("-------------------------------------------Step 3: Finding the best parameters for all variables using BayesSearchCV with nested cross-validation------------------------------------------------")

# Function to create and save partial dependence plots
model2 = RandomForestRegressor(**best_params_overall, oob_score=True)
model2.fit(X, y)
print("OOB Score:", model2.oob_score_)
# Evaluate the final model on the entire dataset
y_pred = model2.predict(X)
final_results = {
    'Train R2': r2_score(y, y_pred),
    'Train RMSE': np.sqrt(mean_squared_error(y, y_pred)),
    'Train MAE': mean_absolute_error(y, y_pred),
    'OOB Score': model2.oob_score_,
    'Test R2': None, # Placeholder for consistency
    'Test RMSE': None, # Placeholder for consistency
    'Test MAE': None # Placeholder for consistency
}

# Save final model performance to CSV
final_results_df = pd.DataFrame([final_results])
final_results_df.to_csv('./final_model_performance.csv', index=False)
print("Final model performance saved to final_model_performance.csv.")

# Feature importances
feature_importances = model2.feature_importances_
indices = np.argsort(feature_importances)[::-1]
feature_importance_df = pd.DataFrame({
    'Feature': X.columns[indices],
    'Importance': feature_importances[indices]
})
feature_importance_df.to_csv('./feature_importances.csv')
print("Feature importances saved to feature_importances.csv.")

# Plot feature importances
plt.figure(figsize=(12, 8))
plt.title("Feature Importances", fontsize=16)
colors = plt.cm.viridis(np.linspace(0, 1, len(indices)))
bar_plot = plt.bar(range(X.shape[1]), feature_importances[indices], color=colors)
plt.xticks(range(X.shape[1]), X.columns[indices], rotation=45, ha='right')
plt.xlabel('Features', fontsize=14)
plt.ylabel('Importance', fontsize=14)
for i, bar in enumerate(bar_plot):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f'{feature_importances[indices][i]:.3f}',
             ha='center', va='bottom')
plt.tight_layout()
plt.savefig('./Feature_Importances.png')
print("Feature importances plot saved.")

# Partial dependence plots
exclude_features = ['LAT', 'LONG', 'Elevation']

# Get the top 19 features, excluding specified ones
top_features = X.columns[np.argsort(feature_importances)[-16:]].tolist()
top_features.reverse()

# Get the list of features to plot, excluding specified ones
plot_features = [feature for feature in top_features if feature not in exclude_features]

# Number of features, columns, and rows
n_features = len(plot_features)
n_cols = 3
n_rows = (n_features + n_cols - 1) // n_cols

fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(11.8, 10.8), dpi=300)
axes = axes.ravel()

# Plot partial dependence plots
display = PartialDependenceDisplay.from_estimator(
    model2, X, plot_features, kind='average', grid_resolution=200, n_cols=n_cols, ax=axes[:n_features]
)

# Adjust plot layout and customize each subplot
plt.subplots_adjust(left=0.1, right=0.98, top=0.98, bottom=0.06, hspace=0.3, wspace=0.28)

# Set the unified y-label for the entire figure
fig.text(0.015, 0.5, r'Defoliation, LTA$_{\delta}$ (%)', fontsize=20, va='center', rotation='vertical')

for i, ax in enumerate(axes):
    if ax in axes[:n_features] and ax.has_data():
        x_data = X[ax.get_xlabel()]
        y_data = ax.get_lines()[0].get_ydata()
        quantile_25 = np.quantile(x_data, 0.25)
        quantile_75 = np.quantile(x_data, 0.75)

        print(f'25% quantile: {quantile_25}, 75% quantile: {quantile_75}')  # Print quantile check

        if quantile_25 == quantile_75:
            print(f'Warning: No range for quantiles on {ax.get_xlabel()}')

        x_min_current, x_max_current = ax.get_xlim()

        x_min_new = min(x_min_current, quantile_25)
        x_max_new = max(x_max_current, quantile_75)

        ax.set_xlim([x_min_new, x_max_new])
        y_min, y_max = min(y_data), max(y_data)
        y_range = y_max - y_min
        y_buffer = y_range * 0.05
        y_min_new = np.floor(y_min - y_buffer)
        y_max_new = np.ceil(y_max + y_buffer)
        real_y_range = y_min_new, y_max_new
        ax.add_patch(plt.Rectangle((quantile_25, real_y_range[0]),
                                   quantile_75 - quantile_25,
                                   real_y_range[1] - real_y_range[0],
                                   fill=True, color='Gray', alpha=0.3, linewidth=2, linestyle='--'))
        ax.set_ylim([y_min_new, y_max_new])
        ticks = np.linspace(y_min_new, y_max_new, 5)
        ax.set_yticks(ticks)
        ax.set_yticklabels([f'{tick:.2f}' for tick in ticks])
        ax.tick_params(axis='both', labelsize=15)
        ax.xaxis.set_major_locator(plt.MaxNLocator(5))
        ax.set_xlabel(ax.get_xlabel(), fontsize=17)

        # Remove all y-axis labels
        ax.set_ylabel('')

        for line in ax.get_lines():
            line.set_color('darkred')
            line.set_linewidth(2)
    else:
        ax.set_visible(False)

#plt.tight_layout(pad=1.0)
plt.savefig('./partial_dependence_plots_4x4.png')
plt.show()
print("Partial dependence plots saved as 'partial_dependence_plots_4x4.png'.")


# SHAP values
explainer = shap.TreeExplainer(model2)
shap_values = explainer.shap_values(X)

# Save SHAP values to CSV
shap_values_df = pd.DataFrame(shap_values, columns=X.columns)
shap_values_df.to_csv('./shap_values.csv', index=False)
print("SHAP values saved to shap_values.csv.")

# Calculate statistics for SHAP values
positive_counts = (shap_values_df > 0).sum(axis=0)
negative_counts = (shap_values_df < 0).sum(axis=0)
zero_counts = (shap_values_df == 0).sum(axis=0)
positive_means = shap_values_df[shap_values_df > 0].mean(axis=0)
negative_means = shap_values_df[shap_values_df < 0].mean(axis=0)

# Calculate total counts including zeros
total_counts = positive_counts + negative_counts + zero_counts

# Calculate percentages including zeros
positive_percentages = (positive_counts / total_counts * 100).fillna(0)
negative_percentages = (negative_counts / total_counts * 100).fillna(0)
zero_percentages = (zero_counts / total_counts * 100).fillna(0)

# Create DataFrame to save statistics
stats_df = pd.DataFrame({
    'Positive Count': positive_counts,
    'Negative Count': negative_counts,
    'Zero Count': zero_counts,
    'Positive Mean': positive_means,
    'Negative Mean': negative_means,
    'Positive Percentage': positive_percentages,
    'Negative Percentage': negative_percentages,
    'Zero Percentage': zero_percentages
})
stats_df.to_csv('./shap_negposzero_stats.csv')
print("Statistics for SHAP values saved to shap_negposzero_stats.csv.")

# Beeswarm plot with SHAP values
colors = ["#0D47A1", "#B71C1C"]
cmap = LinearSegmentedColormap.from_list("shap", colors)
shap.summary_plot(shap_values, X, plot_type="dot", color=cmap, color_bar_label="Feature Value", show=False)

plt.gcf().set_size_inches(10, 10)
plt.gcf().set_dpi(300)
plt.grid(True, linestyle='-', linewidth=0.5, color='gray', alpha=0.5)
ax = plt.gca()
ax.spines['top'].set_visible(True)
ax.spines['right'].set_visible(True)
ax.spines['left'].set_visible(True)
ax.spines['bottom'].set_visible(True)
ax.spines['left'].set_linewidth(0.5)
ax.spines['bottom'].set_linewidth(0.5)
ax.tick_params(axis='y', which='both', direction='out', length=6, width=0.5, labelsize=10)

features = X.columns
y_ticks = ax.get_yticks()
y_labels = [label.get_text() for label in ax.get_yticklabels()]
feature_y_position = dict(zip(y_labels, y_ticks))
x_max = ax.get_xlim()[1]
for feature in features:
    pos_percentage = positive_percentages[feature]
    neg_percentage = negative_percentages[feature]
    y_position = feature_y_position.get(feature, None)
    if y_position is not None:
        ax.text(x_max + 1, y_position, f'{pos_percentage:.0f}% (+)\n{neg_percentage:.0f}% (-)',
                verticalalignment='center', horizontalalignment='right',
                transform=ax.transData, fontsize=10, color='black', backgroundcolor='white')

# Extend the x-axis range to provide more space for text
x_min, x_max = ax.get_xlim()
ax.set_xlim(x_min, x_max + 1.2)

plt.savefig('./shap_beeswarm_plot_percentages.png', bbox_inches='tight')
print("SHAP beeswarm plot with percentages saved.")
plt.show()
plt.close()

# Bar plot for SHAP values
shap.summary_plot(shap_values, X, plot_type="bar", show=False)
plt.savefig('./shap_bar_plot.png')
print("SHAP bar plot saved.")
plt.show()

print("-------------------------------------------Step 4: Train the entire dataset with the best parameters (model2) and generate feature importance, SHAP values, and partial dependence plots-----------------------------------------------")

reduced_cv_results = []
reduced_features_results = []

best_val_r2 = -np.inf  # Track the highest validation R²
best_features_combination = None  # Track the best feature combination (highest validation R²)
best_results = None  # Store the best results (metrics)


# Step 1: Find the highest validation R² value
for i in range(len(indices), 0, -1):  # Start with all features and gradually reduce to 1 feature
    selected_features = X.columns[indices[:i]]  # Select top 'i' features based on importance
    X_reduced = X[selected_features]

    cv_results = []  # Reinitialize the result storage list for each iteration
    r2_train_scores = []
    r2_val_scores = []
    rmse_train_scores = []
    rmse_val_scores = []
    mae_train_scores = []
    mae_val_scores = []
    oob_scores = []
    # Perform cross-validation

    for train_idx, test_idx in outer_cv.split(X_reduced, y_binned):
        X_train, X_test = X_reduced.iloc[train_idx], X_reduced.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        # Use RandomForest model
        model = RandomForestRegressor(**best_params_overall, oob_score=True)
        model.fit(X_train, y_train)  # Train the model

        # Predict on training and validation sets
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        # Calculate metrics
        r2_train = r2_score(y_train, y_train_pred)
        r2_test = r2_score(y_test, y_test_pred)
        rmse_train = np.sqrt(mean_squared_error(y_train, y_train_pred))
        rmse_val = np.sqrt(mean_squared_error(y_test, y_test_pred))
        mae_train = mean_absolute_error(y_train, y_train_pred)
        mae_val = mean_absolute_error(y_test, y_test_pred)
        oob_score = model.oob_score_

        # 将得分追加到列表中，以便之后计算平均值
        r2_train_scores.append(r2_train)
        r2_val_scores.append(r2_test)
        rmse_train_scores.append(rmse_train)
        rmse_val_scores.append(rmse_val)
        mae_train_scores.append(mae_train)
        mae_val_scores.append(mae_val)
        oob_scores.append(oob_score)

        # Store the results for each fold
        results = {
            'Train R2': r2_train,
            'Test R2': r2_test,
            'Train RMSE': rmse_train,
            'Test RMSE': rmse_val,
            'Train MAE': mae_train,
            'Test MAE': mae_val,
            'OOB Score': oob_score,
            'Num Features': i,  # Record the number of features used
            'Features': ', '.join(selected_features)  # Record the selected feature combination
        }
        cv_results.append(results)

    # Calculate average metrics for the current feature combination
    avg_r2_train = np.mean(r2_train_scores)
    avg_r2_val = np.mean(r2_val_scores)
    avg_rmse_train = np.mean(rmse_train_scores)
    avg_rmse_val = np.mean(rmse_val_scores)
    avg_mae_train = np.mean(mae_train_scores)
    avg_mae_val = np.mean(mae_val_scores)
    avg_oob_score = np.mean(oob_scores)

    # Update the best validation R²
    if avg_r2_val > best_val_r2:
        best_val_r2 = avg_r2_val
        best_features_combination = selected_features  # Update best feature combination
        best_results = cv_results  # Store the best results

    # Store results for the current feature set
    reduced_cv_results.append(pd.DataFrame(cv_results))
    reduced_features_results.append({
        'Num Features': i,
        'Avg Train R2': avg_r2_train,
        'Avg Test R2': avg_r2_val,
        'Avg Train RMSE': avg_rmse_train,
        'Avg Test RMSE': avg_rmse_val,
        'Avg Train MAE': avg_mae_train,
        'Avg Test MAE': avg_mae_val,
        'Avg OOB Score': avg_oob_score,
        'Features': ', '.join(selected_features)
    })

# Step 2: Save the final selected feature combination
print(f"\nBest Validation R²: {best_val_r2}")
print(f"Best feature combination: {', '.join(best_features_combination)}")

# Save final selected feature combination
final_features = ', '.join(best_features_combination)

# Save final best features and performance metrics to a file
if best_results:
    best_results_df = pd.DataFrame(best_results)

    # Print and save results
    print("Final selected feature combination:")
    print(best_features_combination)

    best_results_df.to_csv('./selected_best_features.csv', index=False)
    print("Final best feature combination and performance metrics saved to 'selected_best_features.csv'.")

# Save cross-validation results for each feature combination
for i, df in enumerate(reduced_cv_results):
    df.to_csv(f'./reduced_features_cv_results_{len(indices) - i}.csv', index=False)
    print(f"Cross-validation results for {len(indices) - i} features saved.")

# Save average results for each feature set
if reduced_features_results:
    reduced_features_results_df = pd.DataFrame(reduced_features_results)

    # Ensure that the 'Features' column exists
    if 'Features' not in reduced_features_results_df.columns:
        raise ValueError("The 'Features' column is missing from the reduced_features_results_df")

    # Mark the best feature combination
    reduced_features_results_df['Best Combination'] = False
    best_reduced_result_index = reduced_features_results_df['Avg Test R2'].idxmax()  # Find the combination with the highest Test R²
    reduced_features_results_df.loc[best_reduced_result_index, 'Best Combination'] = True  # Mark the best combination as True
    # Save the data to a CSV file
    reduced_features_results_df.to_csv('./average_reduced_features_results.csv', index=False)
    print("Average reduced feature results saved to 'average_reduced_features_results.csv'.")
    # Print and check the best feature combination
    best_features = reduced_features_results_df.loc[best_reduced_result_index, 'Features']
    print("Best feature combination (as string):", best_features)

    # Convert the best feature combination from a string to a list, if needed
    if isinstance(best_features, str):
        best_features = best_features.split(', ')  # Split the string into a list

    # Print and check the final best features list
    print("Best features as list:", best_features)

    # Ensure the best_features are available in X_reduced by filtering out any missing ones
    best_features_filtered = [feature for feature in best_features if feature in X_reduced.columns]

    # Check if any features are missing
    missing_features = [feature for feature in best_features if feature not in X_reduced.columns]
    if missing_features:
        print("The following features are missing in X_reduced and will be ignored:", missing_features)

    # Use the filtered best features to create X_best
    X_best = X_reduced[best_features_filtered]  # Select the best features from X_reduced

    # Train the model using the best feature combination
    model_best = RandomForestRegressor(**best_params_overall, oob_score=True)
    model_best.fit(X_best, y)
    # Make predictions using the best feature set
    y_pred_best = model_best.predict(X_best)

    # Calculate residuals
    residuals = y - y_pred_best

    # Create a DataFrame to store actual values, predictions, and residuals
    results_df = pd.DataFrame({
        'Actual': y,
        'Predicted': y_pred_best,
        'Residual': residuals
    })

    # Save predictions to a CSV file
    results_df.to_csv('./best_feature_combination_predictions.csv', index=False)
    print("Predictions for the best feature combination saved to 'best_feature_combination_predictions.csv")

# Save final selected feature combination
print("------------------------------------------Step 5: Reduce one variable at a time based on importance to select the optimal variables----------------------------------------------")

#
# Fix: Use best_reduced_result_index to get the best result
best_reduced_result_index = reduced_features_results_df['Avg Test R2'].idxmax()  # Best feature combination index
best_reduced_result = reduced_features_results_df.iloc[best_reduced_result_index]  # Get the best result

# Get the optimal number of features from the best_reduced_result
num_features = int(best_reduced_result['Num Features'])

# Extract the corresponding feature names based on the optimal number of features
selected_features_vif = X.columns[indices[:num_features]]

# Create a subset of the data for VIF calculation using the selected features
X_selected_vif = data[selected_features_vif]

# Add a constant to the model (required for VIF calculation in statsmodels)
X_selected_vif = sm.add_constant(X_selected_vif)

# Calculate VIF for each selected feature
vif_selected_data = pd.DataFrame()
vif_selected_data['Feature'] = X_selected_vif.columns  # Feature names
vif_selected_data['VIF'] = [variance_inflation_factor(X_selected_vif.values, i) for i in range(X_selected_vif.shape[1])]  # VIF values

# Save VIF results to a CSV file
vif_selected_data_path = './vif_selected_features_results.csv'
vif_selected_data.to_csv(vif_selected_data_path, index=False)
print(f"VIF results saved to {vif_selected_data_path}.")

# Create the reduced dataset with the final selected features (after VIF check)
print("\nFeatures with VIF greater than 5:")
vif_greater_than_5 = vif_selected_data[vif_selected_data['VIF'] > 5]
print(vif_greater_than_5)

# Select the best features and create the reduced dataset
best_features = best_reduced_result['Features'].split(', ')
X_reduced = X[best_features]

# Save to CSV
X_reduced.to_csv('X_reduced_best_features.csv', index=False)
print("X_reduced saved to X_reduced_best_features.csv")
# Continue with the rest of your script for further steps
def clear_cache(memory_instance):
    memory_instance.clear()
    print("Cache has been cleared.")
clear_cache(memory)
print("------------------------------------------Step 5: Reduce one variable at a time based on importance to select the optimal variables----------------------------------------------")
# Initialize inner_cv_results_all_reduced as an empty list to store DataFrames from each fold
inner_cv_results_all_reduced = []


# Run the nested cross-validation with the reduced feature set
outer_cv_results_reduced, best_params_list_reduced, inner_cv_results_fold =  run_nested_cv_with_penalty(
    X_reduced, y, y_binned, random_forest_range, outer_cv, inner_cv
)

# Save the outer cross-validation results to a CSV file
outer_cv_results_reduced_df = pd.DataFrame(outer_cv_results_reduced)
outer_cv_results_reduced_df.to_csv('./outer_cv_results_best_features_reduced.csv', index=False)
print("Outer CV results saved to outer_cv_results_best_features_reduced.csv.")

# Save all inner cross-validation results to a CSV file
inner_cv_results_all_reduced_df = pd.DataFrame(inner_cv_results_fold)  # Concatenate results across all outer folds
inner_cv_results_all_reduced_df.to_csv('./inner_cv_results_best_features_reduced.csv', index=False)
print("Inner CV results saved to inner_cv_results_best_features_reduced.csv.")

# Save the best hyperparameters for each fold to a CSV file
best_params_reduced_df = pd.DataFrame(best_params_list_reduced)
best_params_reduced_df.to_csv('./best_hyperparameters_reduced_after_feature_selection.csv', index=False)
print("Best hyperparameters saved to best_hyperparameters_reduced_after_feature_selection.csv.")

# Compute the overall best hyperparameters from all the best parameters found in each fold
best_params_overall_reduced = select_best_params(best_params_reduced_df)

# Save the overall best hyperparameters to a CSV file
best_params_overall_reduced_df = pd.DataFrame([best_params_overall_reduced])  # Convert best hyperparameters to DataFrame
best_params_overall_reduced_df.to_csv('./best_params_overall_reduced.csv', index=False)  # Save as CSV
print("Overall best hyperparameters saved to best_params_overall_reduced.csv.")

print("------------------------------------------Step 6: Perform block nested validation on the best variables----------------------------------------------")
# Check if 'cluster' column exists before dropping it
print("Columns in X_reduced before dropping cluster:", X_reduced.columns)


# Continue with the model fitting using the reduced dataset
model3 = RandomForestRegressor(**best_params_overall_reduced, oob_score=True)
model3.fit(X_reduced, y)  # Ensure X_reduced is used without 'cluster'
print("OOB Score:", model3.oob_score_)

# Evaluate the final model on the entire dataset
y_pred = model3.predict(X_reduced)


final_results3 = {
    'Train R2': r2_score(y, y_pred),
    'Train RMSE': np.sqrt(mean_squared_error(y, y_pred)),
    'Train MAE': mean_absolute_error(y, y_pred),
    'OOB Score': model3.oob_score_,
    'Test R2': None, # Placeholder for consistency
    'Test RMSE': None, # Placeholder for consistency
    'Test MAE': None # Placeholder for consistency
}

# Save final model performance to CSV
final_results3_df = pd.DataFrame([final_results3])
final_results3_df.to_csv('./best_features_final_model_performance_reduced.csv', index=False)
print("Final model performance saved to best_features_final_model_performance_reduced.csv.")

# Feature importances
feature_importances = model3.feature_importances_
indices = np.argsort(feature_importances)[::-1]
feature_importance_df = pd.DataFrame({
    'Feature': X_reduced.columns[indices],
    'Importance': feature_importances[indices]
})
feature_importance_df.to_csv('./best_feature_importances_reduced.csv')
print("Feature importances saved to best_feature_importancess_reduced.csv.")

# Plot feature importances
plt.figure(figsize=(12, 8))
plt.title("Feature Importances", fontsize=16)
colors = plt.cm.viridis(np.linspace(0, 1, len(indices)))
bar_plot = plt.bar(range(X_reduced.shape[1]), feature_importances[indices], color=colors)
plt.xticks(range(X_reduced.shape[1]), X_reduced.columns[indices], rotation=45, ha='right')
plt.xlabel('Features', fontsize=14)
plt.ylabel('Importance', fontsize=14)
for i, bar in enumerate(bar_plot):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f'{feature_importances[indices][i]:.3f}',
             ha='center', va='bottom')
plt.tight_layout()
plt.savefig('./best_Feature_Importances_reduced.png')
print("Feature importances _reduced plot saved.")

# Partial dependence plots
#exclude_features = ['LAT', 'LONG', 'Elevation']

# Use all features from X_reduced.columns to generate PDP
plot_features = X_reduced.columns.tolist()  # Convert all column names to a list

# Print debug information to confirm the features being used
print("Features in X_reduced:", X_reduced.columns)
print("Features to plot (plot_features):", plot_features)

# Generate PDP based on all features from X_reduced
n_features = len(plot_features)  # Count the number of features

# Set the number of columns per row (same as in model2)
n_cols = 3

# Calculate the number of rows
n_rows = (n_features + n_cols - 1) // n_cols

# Set the width and height of each subplot (assuming the same size as model2)
subplot_width = 3# Width of each subplot
subplot_height = 2.2 # Height of each subplot

# Dynamically calculate the overall figure size
fig_width = subplot_width * n_cols
fig_height = subplot_height * n_rows

# Create the figure with dynamically calculated figsize
fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(fig_width, fig_height), dpi=300)
axes = axes.ravel()

# Plot the PDP with the same grid resolution as model2
display = PartialDependenceDisplay.from_estimator(
    model3, X_reduced, plot_features, kind='average', grid_resolution=200, n_cols=n_cols, ax=axes[:n_features]
)

# Adjust the layout
plt.subplots_adjust(left=0.12, right=0.95, top=0.95, bottom=0.15, hspace=0.35, wspace=0.35)

# Set a unified Y-axis label for the entire figure
fig.text(0.015, 0.5, r'Defoliation, LTA$_{\delta}$ (%)', fontsize=18, va='center', rotation='vertical')

# Customize each subplot
for i, ax in enumerate(axes):
    if ax in axes[:n_features] and ax.has_data():
        x_data = X_reduced[ax.get_xlabel()]
        y_data = ax.get_lines()[0].get_ydata()

        # Calculate the 25th and 75th percentiles
        quantile_25 = np.quantile(x_data, 0.25)
        quantile_75 = np.quantile(x_data, 0.75)

        print(f'25% quantile: {quantile_25}, 75% quantile: {quantile_75}')  # Print quantile check

        if quantile_25 == quantile_75:
            print(f'Warning: No range for quantiles on {ax.get_xlabel()}')

        # Set x and y axis limits and draw the quantile range
        x_min_current, x_max_current = ax.get_xlim()
        x_min_new = min(x_min_current, quantile_25)
        x_max_new = max(x_max_current, quantile_75)
        ax.set_xlim([x_min_new, x_max_new])

        y_min, y_max = min(y_data), max(y_data)
        y_range = y_max - y_min
        y_buffer = y_range * 0.05
        y_min_new = np.floor(y_min - y_buffer)
        y_max_new = np.ceil(y_max + y_buffer)
        real_y_range = y_min_new, y_max_new
        ax.add_patch(plt.Rectangle((quantile_25, real_y_range[0]),
                                   quantile_75 - quantile_25,
                                   real_y_range[1] - real_y_range[0],
                                   fill=True, color='Gray', alpha=0.3, linewidth=2, linestyle='--'))
        ax.set_ylim([y_min_new, y_max_new])

        # Set y-axis ticks and labels for consistency
        ticks = np.linspace(y_min_new, y_max_new, 5)
        ax.set_yticks(ticks)
        ax.set_yticklabels([f'{tick:.2f}' for tick in ticks])
        ax.tick_params(axis='both', labelsize=12)
        ax.xaxis.set_major_locator(plt.MaxNLocator(5))
        ax.set_xlabel(ax.get_xlabel(), fontsize=12)

        # Remove individual y-axis labels
        ax.set_ylabel('')

        # Customize line style and color
        for line in ax.get_lines():
            line.set_color('darkred')
            line.set_linewidth(2)
    else:
        # Hide the subplot if there is no data
        ax.set_visible(False)

# Save the PDP figure
plt.savefig('./partial_dependence_plots_reduced.png')
plt.show()

print("Partial dependence plots saved as 'partial_dependence_plots_reduced.png'.")

# SHAP values
explainer = shap.TreeExplainer(model3)
shap_values = explainer.shap_values(X_reduced)

# Save SHAP values to CSV
shap_values_df = pd.DataFrame(shap_values, columns=X_reduced.columns)
shap_values_df.to_csv('./shap_values_best_reduced.csv', index=False)
print("SHAP values saved to shap_values.csv.")

# Calculate correlations between SHAP values and target variable
correlations = shap_values_df.corrwith(pd.DataFrame(y).iloc[:, 0])
correlations_df = pd.DataFrame(correlations, columns=['Correlation'])
correlations_df.to_csv('./shap_correlations_with_target_best_reduced.csv', index=True, index_label='Feature')
print("Correlations saved to shap_correlations_with_target.csv.")

# Calculate statistics for SHAP values
positive_counts = (shap_values_df > 0).sum(axis=0)
negative_counts = (shap_values_df < 0).sum(axis=0)
zero_counts = (shap_values_df == 0).sum(axis=0)
positive_means = shap_values_df[shap_values_df > 0].mean(axis=0)
negative_means = shap_values_df[shap_values_df < 0].mean(axis=0)

# Calculate total counts including zeros
total_counts = positive_counts + negative_counts + zero_counts

# Calculate percentages including zeros
positive_percentages = (positive_counts / total_counts * 100).fillna(0)
negative_percentages = (negative_counts / total_counts * 100).fillna(0)
zero_percentages = (zero_counts / total_counts * 100).fillna(0)

# Create DataFrame to save statistics
stats_df = pd.DataFrame({
    'Positive Count': positive_counts,
    'Negative Count': negative_counts,
    'Zero Count': zero_counts,
    'Positive Mean': positive_means,
    'Negative Mean': negative_means,
    'Positive Percentage': positive_percentages,
    'Negative Percentage': negative_percentages,
    'Zero Percentage': zero_percentages
})
stats_df.to_csv('./shap_negposzero_stats_best_reduced.csv')
print("Statistics for SHAP values saved to shap_negposzero_stats_best_reduced.csv.")

# Beeswarm plot with SHAP values
colors = ["#0D47A1", "#B71C1C"]
cmap = LinearSegmentedColormap.from_list("shap", colors)
shap.summary_plot(shap_values, X_reduced, plot_type="dot", color=cmap, color_bar_label="Feature Value", show=False)

plt.gcf().set_size_inches(10, 10)
plt.gcf().set_dpi(300)
plt.grid(True, linestyle='-', linewidth=0.5, color='gray', alpha=0.5)
ax = plt.gca()
ax.spines['top'].set_visible(True)
ax.spines['right'].set_visible(True)
ax.spines['left'].set_visible(True)
ax.spines['bottom'].set_visible(True)
ax.spines['left'].set_linewidth(0.5)
ax.spines['bottom'].set_linewidth(0.5)
ax.tick_params(axis='y', which='both', direction='out', length=6, width=0.5, labelsize=10)

features = X_reduced.columns
y_ticks = ax.get_yticks()
y_labels = [label.get_text() for label in ax.get_yticklabels()]
feature_y_position = dict(zip(y_labels, y_ticks))
x_max = ax.get_xlim()[1]
for feature in features:
    pos_percentage = positive_percentages[feature]
    neg_percentage = negative_percentages[feature]
    y_position = feature_y_position.get(feature, None)
    if y_position is not None:
        ax.text(x_max + 1, y_position, f'{pos_percentage:.0f}% (+)\n{neg_percentage:.0f}% (-)',
                verticalalignment='center', horizontalalignment='right',
                transform=ax.transData, fontsize=10, color='black', backgroundcolor='white')

# Extend the x-axis range to provide more space for text
x_min, x_max = ax.get_xlim()
ax.set_xlim(x_min, x_max + 1.2)

plt.savefig('./shap_beeswarm_plot_percentages_reduced.png', bbox_inches='tight')
print("SHAP beeswarm plot with percentages saved_reduced.")
plt.show()
plt.close()

# Bar plot for SHAP values
shap.summary_plot(shap_values, X_reduced, plot_type="bar", show=False)
plt.savefig('./shap_bar_plot_reduced.png')
print("SHAP bar plot _reduced saved.")


# Train the random forest model using the best hyperparameters
final_model3 = RandomForestRegressor(**best_params_overall_reduced, oob_score=True)

# Get the best features from the reduced dataset
selected_features = best_reduced_result['Features'].split(', ')

# Fit the model using the selected features
final_model3.fit(X_reduced[selected_features], y)

# Generate predictions
predictions = final_model3.predict(X_reduced[selected_features])

# Check if 'Dev_all' or 'dev_all' exists in the data and calculate residuals
if 'Dev_all' in data1.columns:
    residuals = data1['Dev_all'] - predictions
elif 'dev_all' in data1.columns:
    residuals = data1['dev_all'] - predictions
else:
    raise KeyError("Could not find 'Dev_all' or 'dev_all' column in the data")

# Create a DataFrame to store predictions and residuals
results_df = pd.DataFrame({
    'Predicted_Dev_all': predictions,
    'Residuals': residuals
})

# Check the content of the DataFrame before saving (for debugging)
print(results_df.head())

# Define the output path for saving the results
output_path = './best_predictions_and_residuals_reduced.csv'

# Try to save the results to a CSV file, handle any errors if they occur
try:
    results_df.to_csv(output_path, index=False)
    print(f"Predictions and residuals successfully saved to {output_path}")
except Exception as e:
    print(f"An error occurred while saving the file: {e}")
clear_cache(memory)
print("-------------------------------------------Step 7: Train the best features entire dataset with the best parameters (model3) and generate feature importance, SHAP values, and partial dependence plots-----------------------------------------------")