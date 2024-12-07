import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.font_manager as fm
from matplotlib.colors import LinearSegmentedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBRegressor

# File path for the data
file_name = "Prediction_Biochar_Adsorption.xlsx"
file_path = os.path.join(os.getcwd(), "data", file_name)
df = pd.read_excel(file_path, sheet_name='Sr(II)')

# Set font properties for matplotlib
font_path = 'C:/Windows/Fonts/msyh.ttc'  # Adjust this path for your system
prop = fm.FontProperties(fname=font_path)
plt.rcParams['font.sans-serif'] = [prop.get_name()]
plt.rcParams['axes.unicode_minus'] = False

# Feature selection and scaling
excluded_columns = ['C(%)', 'H(%)', 'O(%)', 'N(%)', 'Ash(%)', 'H/C', 'O/C', 'N/C', '(O+N/C)', 'Pore volume(cm3/g)', 'Volume (L)', 'Loading (g)', 'Anion type']
target_column = 'qe（g/L）'
columns_to_normalize = [col for col in df.columns if col not in excluded_columns + [target_column]]

scaler = MinMaxScaler()
df_normalized = df.copy()
df_normalized[columns_to_normalize] = scaler.fit_transform(df[columns_to_normalize])

# Split the data into features and target
data_x = df.drop(columns=[target_column] + excluded_columns)
data_y = df[target_column]

# Split data into train, validation, and test sets
X_train, X_temp, y_train, y_temp = train_test_split(data_x, data_y, test_size=0.2, random_state=42)
X_valid, X_test, y_valid, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Standardize the data (Z-score normalization)
X_mean, X_std = X_train.mean(0), X_train.std(0)
y_mean, y_std = y_train.mean(0), y_train.std(0)

X_train_norm = (X_train - X_mean) / X_std
X_valid_norm = (X_valid - X_mean) / X_std
X_test_norm = (X_test - X_mean) / X_std

y_train_norm = (y_train - y_mean) / y_std
y_valid_norm = (y_valid - y_mean) / y_std
y_test_norm = (y_test - y_mean) / y_std

# Train the XGBoost model
model = XGBRegressor(verbosity=0, n_estimators=100, learning_rate=0.2, max_depth=7, min_child_weight=3)
model.fit(X_train_norm, y_train_norm)

# Predict using the trained model
y_pred_train_norm = model.predict(X_train_norm)
y_pred_test_norm = model.predict(X_test_norm)

# Denormalize the predictions
y_pred_train = y_pred_train_norm * y_std + y_mean
y_pred_test = y_pred_test_norm * y_std + y_mean

# Evaluate the model's performance
def evaluate_regression(y_pred, y_true):
    MAE = np.mean(np.abs(y_pred - y_true))
    MAPE = np.mean(np.abs((y_pred - y_true) / y_true))
    MSE = np.mean((y_pred - y_true) ** 2)
    RMSE = np.sqrt(MSE)
    R2 = r2_score(y_true, y_pred)
    
    return MAE, MAPE, MSE, RMSE, R2

# Get SHAP values
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_train_norm)

# Feature labels
feature_labels = [col for col in df.columns if col not in excluded_columns + [target_column]]

# Custom color map for SHAP plots
colors = ["#1d3557", "#f4a261"]
custom_cmap = LinearSegmentedColormap.from_list("green_to_purple", colors)

# Create SHAP summary plot
shap.summary_plot(shap_values, X_train_norm, feature_names=feature_labels, cmap=custom_cmap, show=False)

# Adjust font size and labels
plt.gcf().set_size_inches(10, 8)
arial_font = fm.FontProperties(fname="C:/Windows/Fonts/arial.ttf", size=18)  # Adjust font size
plt.xticks(fontproperties=arial_font)
plt.yticks(fontproperties=arial_font)
plt.xlabel("SHAP value", fontproperties=arial_font)
plt.ylabel("Features", fontproperties=arial_font)

# Display the plot
plt.tight_layout()
plt.show()

# Save the SHAP summary plot
plt.savefig('shap_summary_plot.png', dpi=300, bbox_inches='tight')

# Create SHAP interaction values and summary plot
shap_interaction_values = explainer.shap_interaction_values(X_train_norm)
shap.summary_plot(shap_interaction_values, X_train_norm)

# Feature importance based on SHAP values
feature_importance = np.abs(shap_values).mean(axis=0)
importance_df = pd.DataFrame({
    'Feature': feature_labels,
    'Importance': feature_importance
}).sort_values(by='Importance', ascending=False)

# Plot feature importance as a bar plot
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=importance_df)
plt.title('Feature Importance based on SHAP values')
plt.xlabel('Mean Absolute SHAP Value')
plt.ylabel('Feature')
plt.show()

# SHAP value distribution by feature categories
pyrolysis_features = ['Pyrolysis temperature(◦C)', 'Heating rate(◦C/min)', 'Pyrolysis time(min)']
biochar_features = ['C(%)', 'H(%)', 'O(%)', 'N(%)', 'Ash(%)', 'H/C', 'O/C', 'N/C', '(O+N/C)', 'Surface area(m2/g)', 'Pore volume(cm3/g)', 'Average pore size(nm)']
adsorption_features = ['Adsorption time(min)', 'Ci', 'Solution pH', 'Volume (L)', 'Loading (g)', 'Dosage(g/L)', 'Cf']

shap_values_abs = np.abs(shap_values).mean(axis=0)
pyrolysis_shap_sum = shap_values_abs[[feature_labels.index(f) for f in pyrolysis_features]].sum()
biochar_shap_sum = shap_values_abs[[feature_labels.index(f) for f in biochar_features]].sum()
adsorption_shap_sum = shap_values_abs[[feature_labels.index(f) for f in adsorption_features]].sum()

category_shap_values = [pyrolysis_shap_sum, biochar_shap_sum, adsorption_shap_sum]
category_labels = ['Pyrolysis conditions', 'Biochar properties', 'Adsorption conditions']

# Plot SHAP value distribution as a pie chart
plt.figure(figsize=(8, 8))
plt.pie(category_shap_values, labels=category_labels, autopct='%1.1f%%', colors=['#ff9999', '#66b3ff', '#99ff99'])
plt.title('SHAP Value Distribution across Feature Categories')
plt.show()
