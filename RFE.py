import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.feature_selection import mutual_info_regression, RFE
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.impute import SimpleImputer
from scipy.stats import pearsonr
import os

# Set font family for the plots
plt.rcParams['font.family'] = 'Arial Unicode MS'

# Define file path
file_path = os.path.join(os.getcwd(), "data", "Sr.xlsx")
df = pd.read_excel(file_path)

# Visualize missing values
plt.figure(figsize=(12, 6))
sns.heatmap(df.isnull(), cbar=False, yticklabels=False, cmap='viridis')
plt.title('Missing Values in Dataset')
plt.tight_layout()
plt.show()

# Columns to exclude from normalization
columns_to_exclude = ['C', 'H', 'O', 'N', 'Ash', 'H/C', 'O/C', 'N/C', '(O+N/C)', 'Pore volume', 'Volume (L)',
                      'loading (g)', 'Anion_type']
columns_to_normalize = [col for col in df.columns if col not in columns_to_exclude]

# Normalize the feature data
scaler = MinMaxScaler()
df_normalized = df.copy()
df_normalized[columns_to_normalize] = scaler.fit_transform(df[columns_to_normalize])

# Define features and target
features = df_normalized.drop(columns=['qe（g/L）'])
target = df['qe（g/L）']

# Feature engineering: handle missing values and scaling
imputer = SimpleImputer(strategy='mean')
features = pd.DataFrame(imputer.fit_transform(features), columns=features.columns)

scaler = StandardScaler()
features_scaled = pd.DataFrame(scaler.fit_transform(features), columns=features.columns)

# Apply log transformation to skewed features
for column in features.columns:
    if features[column].skew() > 1:
        features[f'{column}_log'] = np.log1p(features[column])

# Print log-transformed features
log_features = [f'{column}_log' for column in features.columns if features[column].skew() > 1]
print("Log-transformed features:", log_features)

# Feature selection using RFE with different models
def select_features(X, y, model, n_features=29):
    selector = RFE(model, n_features_to_select=n_features)
    selector = selector.fit(X, y)
    return X.columns[selector.support_].tolist()

lasso_features = select_features(features, target, Lasso(alpha=0.1))
rf_features = select_features(features, target, RandomForestRegressor(n_estimators=100))
linear_features = select_features(features, target, LinearRegression())

# Plot feature selection results
def plot_feature_selection(features, selected_features, title):
    plt.figure(figsize=(12, 6))
    plt.title(title, fontsize=16)
    plt.bar(range(len(features.columns)), [1 if f in selected_features else 0 for f in features.columns])
    plt.xticks(range(len(features.columns)), features.columns, rotation=90)
    plt.ylabel('Selected', fontsize=12)
    plt.xlabel('Features', fontsize=12)
    plt.tight_layout()
    plt.show()

plot_feature_selection(features, lasso_features, 'Lasso Feature Selection')
plot_feature_selection(features, rf_features, 'Random Forest Feature Selection')
plot_feature_selection(features, linear_features, 'Linear Regression Feature Selection')

# Calculate Pearson Correlation Coefficient (PCC) and Mutual Information (MI)
pcc = [pearsonr(features.iloc[:, i], target)[0] for i in range(features.shape[1])]
mi = mutual_info_regression(features, target)

# Store the results in a DataFrame
results = pd.DataFrame({
    'Feature': features.columns,
    'PCC': pcc,
    'MI': mi
})

# Plot Pearson Correlation Coefficient Heatmap
plt.figure(figsize=(20, 16))
mask = np.triu(np.ones_like(features.corr(), dtype=bool))
sns.heatmap(features.corr(), mask=mask, cmap='coolwarm', annot=True, fmt='.2f',
            square=True, linewidths=.5, cbar_kws={"shrink": .5}, annot_kws={"size": 8})
plt.title('Pearson Correlation Coefficient Heatmap', fontsize=20)
plt.xticks(rotation=90, fontsize=10)
plt.yticks(fontsize=10)
plt.tight_layout()
plt.show()

# Plot Mutual Information with Target Variable
plt.figure(figsize=(12, 6))
plt.title('Mutual Information with Target Variable', fontsize=16)
sns.barplot(x='Feature', y='MI', data=results.sort_values('MI', ascending=False))
plt.xticks(rotation=90)
plt.xlabel('Features', fontsize=12)
plt.ylabel('Mutual Information', fontsize=12)
plt.tight_layout()
plt.show()

# Plot Feature Importance
def plot_importance(features, importance, model_name):
    plt.figure(figsize=(12, 6))
    indices = np.argsort(importance)[::-1]
    plt.title(f'{model_name} Feature Importance', fontsize=16)
    plt.bar(range(len(importance)), importance[indices], align='center')
    plt.xticks(range(len(importance)), features.columns[indices], rotation=90)
    plt.ylabel('Importance', fontsize=12)
    plt.xlabel('Features', fontsize=12)
    plt.tight_layout()
    plt.show()

# Lasso Model Feature Importance
lasso_model = Lasso(alpha=0.1)
lasso_model.fit(features_scaled, target)
plot_importance(features_scaled, np.abs(lasso_model.coef_), 'Lasso')

# Random Forest Model Feature Importance
rf_model = RandomForestRegressor(n_estimators=100)
rf_model.fit(features_scaled, target)
plot_importance(features_scaled, rf_model.feature_importances_, 'Random Forest')

# Linear Regression Model Feature Importance
linear_model = LinearRegression()
linear_model.fit(features_scaled, target)
plot_importance(features_scaled, np.abs(linear_model.coef_), 'Linear Regression')
