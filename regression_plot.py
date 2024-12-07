import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.font_manager as fm
from pathlib import Path
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from xgboost import XGBRegressor

class AdsorptionModelAnalysis:
    def __init__(self, file_path):
        """
        Initialize the analysis with data loading and preprocessing.
        
        Args:
            file_path (str): Path to the Excel file containing adsorption data.
        """
        # Load data from specific worksheet
        self.df = pd.read_excel(file_path, sheet_name='Sr(II)')
        
        # Configure plot font
        self._configure_plot_font()
        
        # Define data preprocessing parameters
        self.columns_to_exclude = [
            'C(%)', 'H(%)', 'O(%)', 'N(%)', 'Ash(%)', 'H/C', 'O/C', 'N/C', 
            '(O+N/C)', 'Pore volume(cm3/g)', 'Volume (L)', 'Loading (g)', 'Anion type'
        ]
        self.excluded_features = [
            'Adsorption temperature(◦C)', 'Ion concentration(M)', 'DOM', 'Argon', 
            'Radius(pm)', 'Hydra radius(pm)', 'First ionic(IE_KJ/mol)', 
            'Stirring speed(r/min)', 'Anion type'
        ]
        self.target_column = 'qe（g/L）'
        
    def _configure_plot_font(self):
        """Configure matplotlib font settings."""
        font_path = 'C:/Windows/Fonts/msyh.ttc'
        prop = fm.FontProperties(fname=font_path)
        plt.rcParams['font.sans-serif'] = [prop.get_name()]
        plt.rcParams['axes.unicode_minus'] = False
    
    def preprocess_data(self):
        """
        Preprocess the data by normalizing features.
        
        Returns:
            tuple: Processed features (X) and target variable (y)
        """
        columns_to_normalize = [
            col for col in self.df.columns 
            if col not in self.columns_to_exclude
        ]
        
        # Normalize data
        scaler = MinMaxScaler()
        df_normalized = self.df.copy()
        df_normalized[columns_to_normalize] = scaler.fit_transform(
            self.df[columns_to_normalize]
        )
        
        # Prepare data for modeling
        data_x = self.df.drop(columns=[self.target_column] + self.excluded_features)
        data_y = self.df[self.target_column]
        
        return data_x, data_y
    
    def train_model(self, data_x, data_y):
        """
        Train XGBoost regressor with cross-validation.
        
        Args:
            data_x (pd.DataFrame): Input features
            data_y (pd.Series): Target variable
        
        Returns:
            tuple: Model predictions and true values
        """
        # Cross-validation setup
        kf = KFold(n_splits=10, shuffle=True, random_state=42)
        fold_indices = list(kf.split(data_x))
        random_fold = random.choice(fold_indices)
        train_idx, test_idx = random_fold
        
        data_train_x, data_test_x = data_x.iloc[train_idx], data_x.iloc[test_idx]
        data_train_y, data_test_y = data_y.iloc[train_idx], data_y.iloc[test_idx]
        
        # Standardize data
        scaler_x = StandardScaler()
        scaler_y = StandardScaler()
        
        data_train_x_nor = scaler_x.fit_transform(data_train_x)
        data_test_x_nor = scaler_x.transform(data_test_x)
        
        data_train_y_nor = scaler_y.fit_transform(
            data_train_y.values.reshape(-1, 1)
        ).ravel()
        data_test_y_nor = scaler_y.transform(
            data_test_y.values.reshape(-1, 1)
        ).ravel()
        
        # Model training
        model_params = {
            'verbosity': 0,
            'subsample': 1.0,
            'n_estimators': 100,
            'min_child_weight': 3,
            'max_depth': 7,
            'learning_rate': 0.2,
            'gamma': 0,
            'colsample_bytree': 1.0
        }
        
        model = XGBRegressor(**model_params)
        model.fit(data_train_x_nor, data_train_y_nor)
        
        # Predictions
        y_pred_test_nor = model.predict(data_test_x_nor)
        y_pred_train_nor = model.predict(data_train_x_nor)
        
        y_pred_test = scaler_y.inverse_transform(
            y_pred_test_nor.reshape(-1, 1)
        ).ravel()
        y_pred_train = scaler_y.inverse_transform(
            y_pred_train_nor.reshape(-1, 1)
        ).ravel()
        
        return y_pred_train, y_pred_test, data_train_y, data_test_y
    
    def plot_results(self, y_pred_train, y_pred_test, data_train_y, data_test_y):
        """
        Create visualization of model performance.
        
        Args:
            y_pred_train (np.ndarray): Training predictions
            y_pred_test (np.ndarray): Test predictions
            data_train_y (pd.Series): Training true values
            data_test_y (pd.Series): Test true values
        """
        data_train = pd.DataFrame({
            'Experimental Adsorption Capacity (mg/g)': data_train_y,
            'Predicted Adsorption Capacity (mg/g)': y_pred_train,
            'Data Set': 'Train'
        })
        data_test = pd.DataFrame({
            'Experimental Adsorption Capacity (mg/g)': data_test_y,
            'Predicted Adsorption Capacity (mg/g)': y_pred_test,
            'Data Set': 'Test'
        })
        data = pd.concat([data_train, data_test])
        
        palette = {'Train': '#F2A596', 'Test': '#64A7CD'}
        
        plt.figure(figsize=(12, 10), dpi=300)
        g = sns.JointGrid(
            data=data, 
            x="Experimental Adsorption Capacity (mg/g)", 
            y="Predicted Adsorption Capacity (mg/g)", 
            hue="Data Set", 
            height=10, 
            palette=palette
        )
        
        # Add more visualization details...
        
        
        plt.savefig("AdsorptionModelResults.pdf", format='pdf', bbox_inches='tight')
        plt.show()
    
    def run_analysis(self):
        """Main method to run the entire analysis pipeline."""
        data_x, data_y = self.preprocess_data()
        
        y_pred_train, y_pred_test, data_train_y, data_test_y = self.train_model(data_x, data_y)
        
        # Evaluation
        self.evaluate_regression(y_pred_test, data_test_y)
        
        # Visualization
        self.plot_results(y_pred_train, y_pred_test, data_train_y, data_test_y)
    
    @staticmethod
    def evaluate_regression(y_pred, y_true):
        """
        Calculate regression evaluation metrics.
        
        Args:
            y_pred (np.ndarray): Predicted values
            y_true (np.ndarray): True values
        
        Returns:
            dict: Evaluation metrics
        """
        mae = np.mean(np.abs(y_pred - y_true))
        mse = np.mean((y_pred - y_true) ** 2)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)
        
        print(f'MAE: {mae}, MSE: {mse}, RMSE: {rmse}, R2: {r2}')
        
        return {
            'MAE': mae, 
            'MSE': mse, 
            'RMSE': rmse, 
            'R2': r2
        }

def main():
    file_path = r"D:\桌面\Sr\预测生物炭对重金属吸附能力.xlsx"
    analysis = AdsorptionModelAnalysis(file_path)
    analysis.run_analysis()

if __name__ == "__main__":
    main()