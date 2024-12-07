import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, BaggingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR, NuSVR
from sklearn.neural_network import MLPRegressor
from sklearn.cross_decomposition import PLSRegression
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
import warnings
import shap
from sklearn.inspection import PartialDependenceDisplay
import os

# Suppress warnings
warnings.filterwarnings('ignore')

# File path using the user's desktop path
file_path = os.path.join(os.path.expanduser("~"), "Desktop", "Sr.xlsx")
df = pd.read_excel(file_path)

# Set font properties
font_path = 'C:/Windows/Fonts/msyh.ttc'  # Microsoft YaHei font path
prop = fm.FontProperties(fname=font_path)
plt.rcParams['font.sans-serif'] = [prop.get_name()]
plt.rcParams['axes.unicode_minus'] = False  # Handle negative signs

# Columns to exclude from normalization
columns_to_exclude = ['C(%)', 'H(%)', 'O(%)', 'N(%)', 'Ash(%)', 'H/C', 'O/C', 'N/C', '(O+N/C)', 'Pore volume(cm3/g)', 'Volume (L)',
                      'Loading (g)', 'Anion type']
columns_to_normalize = [col for col in df.columns if col not in columns_to_exclude]

# Normalize feature data
scaler = MinMaxScaler()
df_normalized = df.copy()
df_normalized[columns_to_normalize] = scaler.fit_transform(df[columns_to_normalize])

# Define target variable
target_column = 'qe（g/L）'

# Excluded features
excluded_features = ['Adsorption temperature(◦C)', 'Ion concentration(M)', 'DOM', 'Argon', 'Radius(pm)', 'Hydra radius(pm)',
                     'First ionic(IE_KJ/mol)', 'Stirring speed(r/min)', 'Anion type']

# Select features and target variable
features = df.drop(columns=[target_column] + excluded_features)
target = df[target_column]

# Apply log transformation to skewed features
for column in features.columns:
    if features[column].skew() > 1:
        features[f'{column}_log'] = np.log1p(features[column])
        features = features.drop(columns=[column])  # Drop original column

# Split dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Standardize the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define model list
models = [
    ('Decision Tree', DecisionTreeRegressor(random_state=42)),
    ('Random Forest', RandomForestRegressor(random_state=42)),
    ('Gradient Boosting', GradientBoostingRegressor(random_state=42)),
    ('Bagging', BaggingRegressor(random_state=42)),
    ('SVR', SVR()),
    ('NuSVR', NuSVR()),
    ('Neural Network', MLPRegressor(max_iter=5000, random_state=42)),
    ('PLS Regression', PLSRegression()),
    ('Gaussian Process', GaussianProcessRegressor(random_state=42)),
    ('XGBoost', XGBRegressor(random_state=42, verbosity=0)),
    ('LightGBM', LGBMRegressor(random_state=42, verbose=-1)),
    ('CatBoost', CatBoostRegressor(random_state=42, verbose=0)),
    ('Extra Trees', ExtraTreesRegressor(random_state=42)),
    ('KNN', KNeighborsRegressor()),
    ('Linear Regression', LinearRegression()),
    ('Ridge Regression', Ridge()),
    ('Lasso Regression', Lasso()),
    ('Elastic Net Regression', ElasticNet())
]

# Hyperparameter distributions for random search
param_distributions = {
    'Decision Tree': {
        'max_depth': [None, 3, 4, 5, 6],
        'min_samples_split': [2, 3, 4, 5],
        'min_samples_leaf': [1, 2, 3, 4],
        'max_features': ['sqrt', 'log2', None, 0.5]
    },
    # Add hyperparameters for other models similarly...
}

# Train and evaluate models
results = []
for name, model in models:
    print(f"Training {name}...")

    # Train the original model
    model.fit(X_train_scaled, y_train)
    y_train_pred = model.predict(X_train_scaled)
    y_test_pred = model.predict(X_test_scaled)

    # Evaluate performance on train and test sets
    for dataset_type, y_true, y_pred in [('Train', y_train, y_train_pred), ('Test', y_test, y_test_pred)]:
        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)

        # Save evaluation results
        results.append({
            'Model': name,
            'Type': 'Original',
            'Dataset': dataset_type,
            'MSE': mse,
            'MAE': mae,
            'R2': r2
        })

        # Print evaluation results
        print(f"{name} - Original Model Performance:")
        print(f"    {dataset_type} Set: MSE: {mse}, MAE: {mae}, R2: {r2}")
    print()

    # Randomized hyperparameter search
    if name in param_distributions:
        print(f"Optimizing {name}...")
        random_search = RandomizedSearchCV(model, param_distributions[name], n_iter=20, cv=10, random_state=42, n_jobs=-1)
        random_search.fit(X_train_scaled, y_train)

        best_model = random_search.best_estimator_

        # Predict with the optimized model
        y_train_pred_optimized = best_model.predict(X_train_scaled)
        y_test_pred_optimized = best_model.predict(X_test_scaled)

        for dataset_type, y_true, y_pred_optimized in [('Train', y_train, y_train_pred_optimized),
                                                       ('Test', y_test, y_test_pred_optimized)]:
            mse_optimized = mean_squared_error(y_true, y_pred_optimized)
            mae_optimized = mean_absolute_error(y_true, y_pred_optimized)
            r2_optimized = r2_score(y_true, y_pred_optimized)

            # Save optimized evaluation results
            results.append({
                'Model': name,
                'Type': 'Optimized',
                'Dataset': dataset_type,
                'MSE': mse_optimized,
                'MAE': mae_optimized,
                'R2': r2_optimized
            })

            # Print optimized evaluation results
            print(f"{name} - Optimized Model Performance:")
            print(f"    {dataset_type} Set: MSE: {mse_optimized}, MAE: {mae_optimized}, R2: {r2_optimized}")

        print(f"Best parameters for {name}: {random_search.best_params_}")
        print()

# Convert results to DataFrame
results_df = pd.DataFrame(results)

# Define color palette for scatter plot
cud_palette = ["#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7"]

plt.figure(figsize=(15, 8))
sns.scatterplot(x='MAE', y='R2', hue='Model', style='Type', data=results_df, s=100, palette=cud_palette)
plt.title('Comparison (Original vs Optimized)')
plt.xlabel('MAE')
plt.ylabel('R2')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# Performance comparison plot
def plot_performance_comparison(results_df, dataset_type):
    # Filter data by dataset type (Train or Test)
    df_filtered = results_df[results_df['Dataset'] == dataset_type]

    for metric in ['R2', 'MAE', 'MSE']:
        plt.figure(figsize=(15, 8))

        # Filter out models without optimized results
        original = df_filtered[df_filtered['Type'] == 'Original']
        optimized = df_filtered[df_filtered['Type'] == 'Optimized']

        # Ensure matching models for original and optimized
        common_models = set(original['Model']).intersection(set(optimized['Model']))
        original = original[original['Model'].isin(common_models)]
        optimized = optimized[optimized['Model'].isin(common_models)]

        sns.barplot(x='Model', y=metric, hue='Type', data=pd.concat([original, optimized]), palette=cud_palette)
        plt.title(f'{dataset_type} Performance Comparison ({metric})')
        plt.xlabel('Model')
        plt.ylabel(metric)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

# Plot performance comparison for Train and Test datasets
plot_performance_comparison(results_df, 'Train')
plot_performance_comparison(results_df, 'Test')
