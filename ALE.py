import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.font_manager as fm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
import lightgbm as lgb
import shap
import skexplain
import pickle
import scipy.io as scio
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet

# Set up the environment
sys.path.insert(0, os.path.dirname(os.getcwd()))

# Read data from Excel file
file_path = os.path.join(os.getcwd(), "data", "Sr.xlsx")
df = pd.read_excel(file_path, sheet_name='Sr(II)')

# Set font properties for plots
font_path = 'C:/Windows/Fonts/msyh.ttc'  # Path to Microsoft YaHei font
prop = fm.FontProperties(fname=font_path)
plt.rcParams['font.sans-serif'] = [prop.get_name()]
plt.rcParams['axes.unicode_minus'] = False  # Ensure minus signs are displayed correctly

# Exclude columns that don't need normalization
columns_to_exclude = ['C', 'H', 'O', 'N', 'Ash', 'H/C', 'O/C', 'N/C', '(O+N/C)', 'Pore volume', 'Volume (L)',
                      'loading (g)', 'Anion_type']
columns_to_normalize = [col for col in df.columns if col not in columns_to_exclude]

# Normalize feature data
scaler = MinMaxScaler()
df_normalized = df.copy()
df_normalized[columns_to_normalize] = scaler.fit_transform(df[columns_to_normalize])

# Define target column
target_column = 'qe（g/L）'

# Exclude irrelevant features
excluded_features = ['adsorption_temp', 'Ion Concentration (M)', 'DOM', 'Argon ', 'radius_pm', 'hydra_radius_pm',
                     'First_ionic_IE_KJ/mol', 'rpm', 'Anion_type']

# Select features and target variable
data_train_x = df.drop(columns=[target_column] + excluded_features)  # Features
data_train_y = df[target_column]  # Target

# Split into train, validation, and test sets
data_train_x, data_temp_x, data_train_y, data_temp_y = train_test_split(data_train_x, data_train_y, test_size=0.2, random_state=42)
data_vaild_x, data_test_x, data_vaild_y, data_test_y = train_test_split(data_temp_x, data_temp_y, test_size=0.5, random_state=42)

# Z-score standardization
X_mean, y_mean = data_train_x.mean(0), data_train_y.mean(0)
X_std, y_std = data_train_x.std(0), data_train_y.std(0)

data_train_x_nor = (data_train_x - X_mean) / X_std
data_vaild_x_nor = (data_vaild_x - X_mean) / X_std
data_test_x_nor = (data_test_x - X_mean) / X_std

data_train_y_nor = (data_train_y - y_mean) / y_std
data_vaild_y_nor = (data_vaild_y - y_mean) / y_std
data_test_y_nor = (data_test_y - y_mean) / y_std

# Inverse standardization
data_train_x_inv = data_train_x_nor * X_std + X_mean
data_vaild_x_inv = data_vaild_x_nor * X_std + X_mean
data_test_x_inv = data_test_x_nor * X_std + X_mean

data_train_y_inv = data_train_y_nor * y_std + y_mean
data_vaild_y_inv = data_vaild_y_nor * y_std + y_mean
data_test_y_inv = data_test_y_nor * y_std + y_mean

# Results
examples = data_train_x_inv  # Inverse standardized features
targets = data_train_y_inv.values  # Inverse standardized target variable

# Print shapes of the datasets
print('Inverse standardization results:')
print('Train feature shape:', examples.shape)
print('Train target shape:', targets.shape)

# Function to evaluate regression model
def evaluate_regress(y_pred, y_true):
    MAE = np.sum(np.abs(y_pred - y_true)) / len(y_true)
    MAPE = np.sum(np.abs((y_pred - y_true) / y_true)) / len(y_true)
    MSE = np.sum((y_pred - y_true) ** 2) / len(y_true)
    RMSE = np.sqrt(MSE)
    R2 = r2_score(y_true, y_pred)
    
    print(f'MAE: {MAE}, MAPE: {MAPE}, MSE: {MSE}, RMSE: {RMSE}, R²: {R2}')
    
    return MAE, MAPE, MSE, RMSE, R2

# Feature labels for ALE analysis
feature_label = [
    'Pyrolysis_temp', 'Heating rate (oC)', 'Pyrolysis_time (min)', 'C', 'H', 'O', 'N', 'Ash', 'H/C', 'O/C', 'N/C',
    '(O+N/C)', 'Surface area', 'Pore volume', 'Average pore size', 'Adsorption_time (min)', 'Ci', 'solution pH',
    'Volume (L)', 'Loading (g)', 'g/L', 'Cf'
]

# Load models and perform ALE analysis
estimators = skexplain.load_models()

# Create the explainer toolkit
explainer = skexplain.ExplainToolkit(estimators, X=examples, y=targets)

# ALE for 1D features
ale_1d_ds = explainer.ale(features='all', n_bootstrap=2, subsample=10000, n_jobs=1, n_bins=20)

# Plot ALE results for a feature
fig, ax = plt.subplots(dpi=150)
fig, ax = explainer.plot_ale(
    ale=ale_1d_ds,
    features='Surface area',
    display_units=plotting_config.display_units,
    display_feature_names=plotting_config.display_feature_names,
    ax=ax,
    line_kws={'line_colors': ['b', 'orange', 'k'], 'linewidth': 3.0, 'linestyle': 'dashed'},
    hist_color='blue'
)

# Function to run ALE for selected features
def run_ale():
    features = [('O+N/C', 'Surface area')]
    explainer = skexplain.ExplainToolkit(estimators, X=examples, y=targets)
    ale_2d_ds = explainer.ale(
        features=features,
        n_bootstrap=1,
        subsample=0.7,
        n_jobs=len(features) * len(estimators),
        n_bins=200
    )
    print(ale_2d_ds)
    fig, axes = explainer.plot_ale(
        ale=ale_2d_ds,
        display_units=plotting_config.display_units,
        display_feature_names=plotting_config.display_feature_names,
        kde_curve=False,
        scatter=False
    )
    fig.savefig('ale_plot.svg', format='svg')

# Run ALE analysis
if __name__ == '__main__':
    run_ale()
