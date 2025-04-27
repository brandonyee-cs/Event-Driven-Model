import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
import warnings

# Import shared models
from src.models import TimeSeriesRidge, XGBoostDecileModel

# Suppress SettingWithCopyWarning for cleaner output in feature engineering
warnings.filterwarnings('ignore', category=pd.errors.SettingWithCopyWarning)
# Suppress FutureWarnings from seaborn/matplotlib
warnings.simplefilter(action='ignore', category=FutureWarning)


class DataLoader:
    def __init__(self, fda_path, stock_paths, window_days=100):
        """
        Initialize DataLoader for FDA events.

        Parameters:
        fda_path (str): Path to the FDA approval event data CSV.
        stock_paths (list or str): List/single path to stock data CSV files.
        window_days (int): Number of days before/after event date.
        """
        self.fda_path = fda_path
        if isinstance(stock_paths, str): self.stock_paths = [stock_paths]
        elif isinstance(stock_paths, list): self.stock_paths = stock_paths
        else: raise TypeError("stock_paths must be a string or a list.")
        self.window_days = window_days

    def _load_single_stock_file(self, stock_path):
        """Load and process a single stock data file."""
        try:
            headers = pd.read_csv(stock_path, nrows=0).columns
            date_col = next((col for col in headers if col.lower() in ['date', 'trade_date', 'trading_date', 'tradedate', 'dt']), None)

            if date_col:
                stock_data = pd.read_csv(stock_path, parse_dates=[date_col])
                stock_data = stock_data.rename(columns={date_col: 'date'})
            else:
                stock_data = pd.read_csv(stock_path)
                print(f"Warning: No standard date column in {stock_path}. Attempting conversion.")

            ticker_col_map = {'sym_root': 'ticker', 'symbol': 'ticker'}
            stock_data.rename(columns=ticker_col_map, inplace=True)
            if 'ticker' not in stock_data.columns: raise ValueError(f"No 'ticker' column found in {stock_path}.")

            if 'date' not in stock_data.columns:
                 date_col_alt = next((col for col in stock_data.columns if col.lower() in ['trade_date', 'trading_date', 'tradedate', 'dt', 'time', 'timestamp']), None)
                 if date_col_alt: stock_data = stock_data.rename(columns={date_col_alt: 'date'})
                 else: raise ValueError(f"No 'date' column found in {stock_path}.")

            if not pd.api.types.is_datetime64_any_dtype(stock_data['date']):
                 try:
                     stock_data['date'] = pd.to_datetime(stock_data['date'], errors='coerce')
                     if stock_data['date'].isna().all(): raise ValueError("All dates failed parse.")
                 except Exception as e: raise ValueError(f"Date conversion failed in {stock_path}: {e}")

            required_cols = ['date', 'ticker', 'prc', 'ret'] # Essential for FDA features
            missing_cols = [col for col in required_cols if col not in stock_data.columns]
            if missing_cols: raise ValueError(f"Missing required columns in {stock_path}: {missing_cols}")

            numeric_cols = ['prc', 'ret', 'vol', 'openprc', 'askhi', 'bidlo', 'shrout']
            for col in numeric_cols:
                if col in stock_data.columns: stock_data[col] = pd.to_numeric(stock_data[col], errors='coerce')

            return stock_data
        except FileNotFoundError: raise ValueError(f"Stock file not found: {stock_path}")
        except Exception as e: raise ValueError(f"Error processing {stock_path}: {e}")

    def load_data(self):
        """Load FDA approval data and stock data, then merge them"""
        try:
            print(f"Loading FDA event data from: {self.fda_path}")
            fda_data = pd.read_csv(self.fda_path)
            if 'Approval Date' not in fda_data.columns: raise ValueError("Missing 'Approval Date' column.")
            fda_data['Approval Date'] = pd.to_datetime(fda_data['Approval Date'], errors='coerce')
            fda_data.dropna(subset=['Approval Date'], inplace=True)

            if 'ticker' not in fda_data.columns:
                if 'Ticker' in fda_data.columns:
                    fda_data['ticker'] = fda_data['Ticker'].astype(str).apply(lambda x: x.split(':')[-1].strip() if ':' in x else x.strip())
                else: raise ValueError("Missing 'ticker' or 'Ticker' column.")
            if 'Drug Name' not in fda_data.columns: fda_data['Drug Name'] = "N/A"
            fda_data['ticker'] = fda_data['ticker'].str.upper()

            approval_events = fda_data[['ticker', 'Approval Date', 'Drug Name']].drop_duplicates().reset_index(drop=True)
            print(f"Found {len(approval_events)} unique FDA approval events.")
            if len(approval_events) == 0: raise ValueError("No valid FDA events found.")
        except Exception as e: raise ValueError(f"Error loading FDA data: {e}")

        stock_data_list, failed_files = [], []
        for stock_path in self.stock_paths:
            try:
                print(f"Loading stock data from: {stock_path}...")
                stock_data = self._load_single_stock_file(stock_path)
                stock_data['ticker'] = stock_data['ticker'].str.upper()
                stock_data_list.append(stock_data)
                print(f"Loaded {stock_path} ({len(stock_data)} rows)")
            except ValueError as e: print(f"Warning: Failed load {stock_path}: {e}"); failed_files.append(stock_path)
            except Exception as e: print(f"Warning: Unexpected error {stock_path}: {e}"); failed_files.append(stock_path)

        if not stock_data_list: raise ValueError("No stock data loaded.")
        if failed_files: print(f"\nWarning: Skipped {len(failed_files)} stock file(s): {failed_files}\n")

        stock_data_combined = pd.concat(stock_data_list, ignore_index=True)
        print(f"Combined stock data rows before deduplication: {len(stock_data_combined)}")
        stock_data_combined = stock_data_combined.drop_duplicates(subset=['date', 'ticker'], keep='first')
        print(f"Combined stock data rows after deduplication: {len(stock_data_combined)}")
        if len(stock_data_combined) == 0: raise ValueError("Stock data empty after deduplication.")

        tickers_with_approvals = approval_events['ticker'].unique()
        stock_data_filtered = stock_data_combined[stock_data_combined['ticker'].isin(tickers_with_approvals)].copy()
        print(f"Filtered stock data for {len(tickers_with_approvals)} tickers. Rows: {len(stock_data_filtered)}")
        if len(stock_data_filtered) == 0: raise ValueError("No stock data for relevant tickers.")

        merged_data = pd.merge(stock_data_filtered, approval_events, on='ticker', how='inner')
        merged_data['days_to_approval'] = (merged_data['date'] - merged_data['Approval Date']).dt.days
        event_window_data = merged_data[(merged_data['days_to_approval'] >= -self.window_days) & (merged_data['days_to_approval'] <= self.window_days)].copy()
        if event_window_data.empty: raise ValueError("No stock data within event windows.")

        event_window_data['is_approval_date'] = (event_window_data['days_to_approval'] == 0).astype(int)
        event_window_data['event_id'] = event_window_data['ticker'] + "_" + event_window_data['Approval Date'].dt.strftime('%Y%m%d')
        print(f"Created windows for {event_window_data['event_id'].nunique()} FDA events.")
        combined_data = event_window_data.sort_values(by=['ticker', 'Approval Date', 'date']).reset_index(drop=True)
        print(f"Final FDA dataset shape: {combined_data.shape}")
        return combined_data


class FeatureEngineer:
    def __init__(self, prediction_window=5):
        self.windows = [5, 10, 20]
        self.prediction_window = prediction_window
        self.imputer = SimpleImputer(strategy='median')
        self.feature_names = [] # Store feature names created
        self.final_feature_names = [] # Store final names after imputation

    def create_target(self, df, price_col='prc'):
        """Create target variable for FDA analysis."""
        print(f"Creating target 'future_ret' (window: {self.prediction_window} days)...")
        df = df.copy()
        if 'event_id' not in df.columns: raise ValueError("'event_id' required.")
        if price_col not in df.columns: raise ValueError(f"Price column '{price_col}' not found.")
        df = df.sort_values(by=['event_id', 'date'])
        df['future_ret'] = df.groupby('event_id')[price_col].transform(
            lambda x: x.shift(-self.prediction_window) / x - 1 if x is not None and not x.empty and len(x) > self.prediction_window else np.nan
        )
        print(f"'future_ret' created. Non-null: {df['future_ret'].notna().sum()}")
        return df

    def calculate_features(self, df, price_col='prc', return_col='ret'):
        """Calculate features for FDA analysis."""
        print("Calculating FDA features...")
        df = df.copy()
        required = ['event_id', price_col, return_col, 'Approval Date', 'date']
        if not all(c in df.columns for c in required): raise ValueError(f"Missing required columns: {required}")
        df = df.sort_values(by=['event_id', 'date'])
        grouped = df.groupby('event_id')
        current_features = []

        for window in self.windows:
            col = f'momentum_{window}'; df[col] = grouped[price_col].transform(lambda x: x.pct_change(periods=window)); current_features.append(col)
        df['delta_momentum_5_10'] = df['momentum_5'] - df['momentum_10']; current_features.append('delta_momentum_5_10')
        df['delta_momentum_10_20'] = df['momentum_10'] - df['momentum_20']; current_features.append('delta_momentum_10_20')

        for window in self.windows:
            col = f'volatility_{window}'; min_p = max(2, min(window, 5)); df[col] = grouped[return_col].transform(lambda x: x.rolling(window, min_periods=min_p).std()); current_features.append(col)
        df['delta_volatility_5_10'] = df['volatility_5'] - df['volatility_10']; current_features.append('delta_volatility_5_10')
        df['delta_volatility_10_20'] = df['volatility_10'] - df['volatility_20']; current_features.append('delta_volatility_10_20')

        df['log_ret'] = grouped[price_col].transform(lambda x: np.log(x / x.shift(1))); current_features.append('log_ret')
        if 'days_to_approval' in df.columns: current_features.append('days_to_approval')
        for lag in range(1, 4): col = f'ret_lag_{lag}'; df[col] = grouped[return_col].shift(lag); current_features.append(col)

        pre_approval_rets = {}
        for event_id, group in grouped:
            approval_date = group['Approval Date'].iloc[0]
            pre_data = group[(group['date'] <= approval_date) & (group['date'] > approval_date - pd.Timedelta(days=30))]
            pre_approval_rets[event_id] = pre_data[return_col].sum() if not pre_data.empty else 0
        df['pre_approval_ret_30d'] = df['event_id'].map(pre_approval_rets); current_features.append('pre_approval_ret_30d')

        df['prev_day_volatility'] = grouped[return_col].shift(1).abs(); current_features.append('prev_day_volatility')
        df['prev_5d_vol_std'] = grouped[return_col].shift(1).rolling(window=5, min_periods=2).std(); current_features.append('prev_5d_vol_std')
        if 'is_approval_date' in df.columns: current_features.append('is_approval_date')

        # Add any other relevant features here

        self.feature_names = sorted(list(set(current_features)))
        print(f"Calculated {len(self.feature_names)} raw FDA features.")
        return df

    def get_features_target(self, df, fit_imputer=False):
        """Extract feature matrix X and target vector y, handling missing values."""
        print("Extracting FDA features (X) and target (y)...")
        df = df.copy()
        if not self.feature_names: raise RuntimeError("Run calculate_features first.")
        available_features = [col for col in self.feature_names if col in df.columns]
        if not available_features: raise ValueError("No calculated features found in DataFrame.")

        df_with_target = df.dropna(subset=['future_ret'])
        if len(df_with_target) == 0:
            print("Warning: No data remains after filtering for non-null target.")
            return pd.DataFrame(columns=available_features), pd.Series(dtype=float)

        X = df_with_target[available_features].copy()
        y = df_with_target['future_ret'].copy()
        print(f"Original X shape: {X.shape}. Non-null y count: {len(y)}")

        # Impute missing values
        if fit_imputer:
            print("Fitting imputer and transforming features...")
            X_imputed = self.imputer.fit_transform(X)
        else:
            if not hasattr(self.imputer, 'statistics_'): raise RuntimeError("Imputer not fitted.")
            print("Transforming features using pre-fitted imputer...")
            X_imputed = self.imputer.transform(X)

        X = pd.DataFrame(X_imputed, columns=available_features, index=X.index)
        if X.isna().any().any(): warnings.warn("NaNs remain AFTER imputation!")
        else: print("No NaNs remaining after imputation.")

        self.final_feature_names = X.columns.tolist()
        print(f"Final X shape: {X.shape}. Using {len(self.final_feature_names)} features.")
        return X, y


class Analysis:
    def __init__(self, data_loader, feature_engineer):
        """Analysis class for FDA data."""
        self.data_loader = data_loader
        self.feature_engineer = feature_engineer
        self.data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.train_indices = None
        self.test_indices = None
        self.models = {}
        self.results = {}

    def load_and_prepare_data(self):
        print("--- Loading FDA Data ---")
        self.data = self.data_loader.load_data()
        print("\n--- Creating Target Variable (FDA) ---")
        self.data = self.feature_engineer.create_target(self.data)
        print("\n--- Calculating Features (FDA) ---")
        self.data = self.feature_engineer.calculate_features(self.data)
        print("\n--- FDA Data Preparation Complete ---")
        return self.data

    def train_models(self, test_size=0.2, time_split_column='Approval Date'):
        """Split data and train models for FDA analysis."""
        if self.data is None: raise RuntimeError("Run load_and_prepare_data() first.")
        if time_split_column not in self.data.columns: raise ValueError(f"Time split column '{time_split_column}' not found.")
        if 'event_id' not in self.data.columns: raise ValueError("'event_id' required.")

        print(f"\n--- Splitting FDA Data (Train/Test based on {time_split_column}) ---")
        events = self.data[['event_id', time_split_column]].drop_duplicates().sort_values(time_split_column)
        split_index = int(len(events) * (1 - test_size))
        if split_index == 0 or split_index == len(events): raise ValueError("test_size results in empty train/test set.")
        split_date = events.iloc[split_index][time_split_column]
        print(f"Splitting {len(events)} unique events. Train before {split_date}.")
        train_event_ids = set(events.iloc[:split_index]['event_id'])
        test_event_ids = set(events.iloc[split_index:]['event_id'])
        train_mask = self.data['event_id'].isin(train_event_ids)
        test_mask = self.data['event_id'].isin(test_event_ids)
        self.train_indices = self.data.index[train_mask]
        self.test_indices = self.data.index[test_mask]
        print(f"Train rows: {len(self.train_indices)}, Test rows: {len(self.test_indices)}.")

        print("\nExtracting features/target for TRAIN set (fitting imputer)...")
        self.X_train, self.y_train = self.feature_engineer.get_features_target(self.data.loc[self.train_indices], fit_imputer=True)
        print("\nExtracting features/target for TEST set (transforming)...")
        self.X_test, self.y_test = self.feature_engineer.get_features_target(self.data.loc[self.test_indices], fit_imputer=False)
        print(f"\nTrain shapes: X={self.X_train.shape}, y={self.y_train.shape}")
        print(f"Test shapes: X={self.X_test.shape}, y={self.y_test.shape}")
        if len(self.X_train) == 0 or len(self.X_test) == 0: raise ValueError("Train or test set empty.")

        print("\n--- Training Models (FDA) ---")
        # 1. TimeSeriesRidge
        try:
             print("Training TimeSeriesRidge...")
             ts_ridge = TimeSeriesRidge(alpha=1.0, lambda2=0.1, feature_order=self.feature_engineer.final_feature_names)
             ts_ridge.fit(self.X_train, self.y_train)
             self.models['TimeSeriesRidge'] = ts_ridge
             print("TimeSeriesRidge complete.")
        except Exception as e: print(f"Error TimeSeriesRidge: {e}")
        # 2. XGBoostDecile
        try:
             print("\nTraining XGBoostDecile...")
             xgb_params = {'n_estimators': 150, 'max_depth': 5, 'learning_rate': 0.05, 'objective': 'reg:squarederror', 'subsample': 0.8, 'colsample_bytree': 0.8, 'random_state': 42, 'n_jobs': -1, 'early_stopping_rounds': 10}
             xgb_decile = XGBoostDecileModel(weight=0.7, momentum_feature='momentum_5', n_deciles=10, alpha=1.0, lambda_smooth=0.1, xgb_params=xgb_params, ts_ridge_feature_order=self.feature_engineer.final_feature_names)
             if 'momentum_5' not in self.X_train.columns: print("Warning: 'momentum_5' not found for XGBoostDecile.")
             xgb_decile.fit(self.X_train, self.y_train)
             self.models['XGBoostDecile'] = xgb_decile
             print("XGBoostDecile complete.")
        except Exception as e: print(f"Error XGBoostDecile: {e}")

        print("\n--- FDA Model Training Complete ---")
        return self.models

    def evaluate_models(self):
        """Evaluate trained models on the test set."""
        print("\n--- Evaluating FDA Models ---")
        if not self.models: print("No models trained."); return {}
        if self.X_test is None or self.y_test is None or len(self.X_test)==0: print("Test data unavailable or empty."); return {}

        self.results = {}
        for name, model in self.models.items():
            print(f"Evaluating {name}...")
            try:
                y_pred = model.predict(self.X_test)
                valid_mask = np.isfinite(self.y_test) & np.isfinite(y_pred)
                y_test_valid, y_pred_valid = self.y_test[valid_mask], y_pred[valid_mask]
                if len(y_test_valid) > 0:
                    mse = mean_squared_error(y_test_valid, y_pred_valid); r2 = r2_score(y_test_valid, y_pred_valid)
                else: mse, r2 = np.nan, np.nan
                self.results[name] = {'MSE': mse, 'RMSE': np.sqrt(mse), 'R2': r2, 'N': len(y_test_valid)}
                print(f"  {name}: MSE={mse:.6f}, RMSE={np.sqrt(mse):.6f}, R2={r2:.4f}, N={len(y_test_valid)}")
            except Exception as e: print(f"  Error evaluating {name}: {e}"); self.results[name] = {'Error': str(e)}
        print("\n--- FDA Evaluation Complete ---")
        return self.results

    def plot_feature_importance(self, model_name='TimeSeriesRidge', top_n=20):
        """Plot feature importance for a specified model."""
        print(f"\n--- Plotting FDA Feature Importance for {model_name} ---")
        if model_name not in self.models: print(f"Error: Model '{model_name}' not found."); return None
        model = self.models[model_name]; feature_names = getattr(self.feature_engineer, 'final_feature_names', None)
        if not feature_names: print("Error: Final feature names not found."); return None

        importances = None
        if isinstance(model, TimeSeriesRidge):
            if hasattr(model, 'coef_'): importances = np.abs(model.coef_)
        elif isinstance(model, XGBoostDecileModel):
             if hasattr(model, 'xgb_model') and hasattr(model.xgb_model, 'feature_importances_'):
                 try:
                     xgb_feat_names = model.xgb_model.get_booster().feature_names
                     if xgb_feat_names:
                         imp_dict = dict(zip(xgb_feat_names, model.xgb_model.feature_importances_))
                         importances = np.array([imp_dict.get(name, 0) for name in feature_names])
                     else: importances = model.xgb_model.feature_importances_
                 except Exception: importances = model.xgb_model.feature_importances_
        if importances is None: print(f"Could not get importance for {model_name}."); return None

        if len(importances) != len(feature_names):
             print(f"Warn: Importance len ({len(importances)}) != feature len ({len(feature_names)}). Aligning.");
             if len(importances) > len(feature_names): importances = importances[:len(feature_names)]
             elif len(importances) < len(feature_names): importances = np.pad(importances, (0, len(feature_names) - len(importances)))

        feat_imp_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances}).sort_values('Importance', ascending=False).head(top_n)
        plt.figure(figsize=(10, max(6, top_n // 2)))
        sns.barplot(x='Importance', y='Feature', data=feat_imp_df, palette='viridis')
        plt.title(f'Top {top_n} Features by Importance ({model_name} - FDA)')
        plt.xlabel('Importance Score'); plt.ylabel('Feature'); plt.tight_layout(); plt.show()
        return feat_imp_df

    def plot_predictions_for_event(self, event_id, model_name='TimeSeriesRidge'):
        """Plot actual vs. predicted returns for a specific FDA event."""
        print(f"\n--- Plotting FDA Predictions for Event: {event_id} ({model_name}) ---")
        if model_name not in self.models: print(f"Error: Model '{model_name}' not found."); return None
        if self.data is None: print("Error: Data not loaded."); return None
        model = self.models[model_name]
        event_data_full = self.data[self.data['event_id'] == event_id].sort_values('date').copy()
        if event_data_full.empty: print(f"Error: No data for event_id '{event_id}'."); return None

        ticker = event_data_full['ticker'].iloc[0]; approval_date = event_data_full['Approval Date'].iloc[0]
        if not hasattr(self.feature_engineer.imputer, 'statistics_'): print("Error: Imputer not fitted."); return None
        X_event, y_event_actual = self.feature_engineer.get_features_target(event_data_full, fit_imputer=False)
        if X_event.empty: print(f"Warn: No valid features for event {event_id}."); return None

        try: y_pred_event = model.predict(X_event)
        except Exception as e: print(f"Error predicting event {event_id}: {e}"); return None

        event_data_pred = event_data_full.loc[X_event.index].copy()
        event_data_pred['predicted_future_ret'] = y_pred_event
        # event_data_pred['actual_future_ret'] = y_event_actual # Optional

        plt.figure(figsize=(14, 6))
        plt.plot(event_data_full['date'], event_data_full['ret'], '.-', label='Actual Daily Return', alpha=0.6, color='blue')
        plt.scatter(event_data_pred['date'], event_data_pred['predicted_future_ret'],
                    color='red', label=f'Predicted {self.feature_engineer.prediction_window}-day Future Ret', s=15, zorder=5, alpha=0.8)
        plt.axvline(x=approval_date, color='g', linestyle='--', label='Approval Date')
        plt.title(f"{ticker} ({event_id}) - Daily Returns & Predicted Future Returns ({model_name} - FDA)")
        plt.ylabel("Return"); plt.xlabel("Date"); plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout(); plt.show()
        return event_data_pred

    def find_sample_event_ids(self, n=5):
        """Find sample FDA event identifiers."""
        if self.data is None or 'event_id' not in self.data.columns: return []
        unique_events = self.data['event_id'].unique()
        return list(unique_events[:min(n, len(unique_events))])

    def analyze_sharpe_ratio_dynamics(self, results_dir, file_prefix="fda", risk_free_rate=0.0, window=20, min_periods=10, pre_days=30, post_days=30):
        """Calculates, plots, and saves rolling Sharpe Ratio dynamics."""
        print(f"\n--- Analyzing Rolling Sharpe Ratio (Window={window}d) ---")
        if self.data is None or 'ret' not in self.data.columns or 'days_to_approval' not in self.data.columns:
            print("Error: Data/required columns missing."); return None

        df = self.data.copy().sort_values(by=['event_id', 'date'])
        df['rolling_mean_ret'] = df.groupby('event_id')['ret'].transform(lambda x: x.rolling(window=window, min_periods=min_periods).mean())
        df['rolling_std_ret'] = df.groupby('event_id')['ret'].transform(lambda x: x.rolling(window=window, min_periods=min_periods).std())
        epsilon = 1e-8; daily_risk_free = risk_free_rate / 252
        df['daily_sharpe'] = (df['rolling_mean_ret'] - daily_risk_free) / (df['rolling_std_ret'] + epsilon)
        df['annualized_sharpe'] = df['daily_sharpe'] * np.sqrt(252)
        aligned_sharpe = df.groupby('days_to_approval')['annualized_sharpe'].mean()
        aligned_sharpe_plot = aligned_sharpe.loc[-pre_days:post_days]

        # Plotting
        fig, ax = plt.subplots(figsize=(12, 6)) # Get figure and axes objects
        aligned_sharpe_plot.plot(kind='line', marker='.', linestyle='-', ax=ax)
        ax.axvline(0, color='red', linestyle='--', linewidth=1, label='Approval Day')
        ax.axhline(0, color='grey', linestyle=':', linewidth=1)
        ax.set_title(f'Average Annualized Rolling Sharpe Ratio Around FDA Approval (Window={window}d)')
        ax.set_xlabel('Days Relative to Approval'); ax.set_ylabel('Average Annualized Sharpe Ratio')
        ax.legend(); ax.grid(True, alpha=0.5)
        plt.tight_layout()

        # Save plot
        plot_filename = os.path.join(results_dir, f"{file_prefix}_sharpe_ratio_rolling_{window}d.png")
        try:
            plt.savefig(plot_filename); print(f"Saved Sharpe plot to: {plot_filename}")
        except Exception as e: print(f"Error saving Sharpe plot: {e}")
        plt.close(fig) # Close the plot figure

        # Save data
        csv_filename = os.path.join(results_dir, f"{file_prefix}_sharpe_ratio_rolling_{window}d_data.csv")
        try:
            aligned_sharpe_plot.to_csv(csv_filename, header=True)
            print(f"Saved Sharpe data to: {csv_filename}")
        except Exception as e: print(f"Error saving Sharpe data: {e}")

        print(f"Average Sharpe Ratio ({window}d rolling) in plot window: {aligned_sharpe_plot.mean():.4f}")
        return aligned_sharpe

    def analyze_volatility_spikes(self, results_dir, file_prefix="fda", window=5, min_periods=3, pre_days=30, post_days=30, baseline_window=(-60, -11), event_window=(-2, 2)):
        """Calculates, plots, and saves rolling volatility dynamics and compares event vs baseline."""
        print(f"\n--- Analyzing Rolling Volatility (Window={window}d) ---")
        if self.data is None or 'ret' not in self.data.columns or 'days_to_approval' not in self.data.columns:
            print("Error: Data/required columns missing."); return None

        df = self.data.copy().sort_values(by=['event_id', 'date'])
        df['rolling_vol'] = df.groupby('event_id')['ret'].transform(lambda x: x.rolling(window=window, min_periods=min_periods).std())
        df['annualized_vol'] = df['rolling_vol'] * np.sqrt(252) * 100 # In percent
        aligned_vol = df.groupby('days_to_approval')['annualized_vol'].mean()
        aligned_vol_plot = aligned_vol.loc[-pre_days:post_days]

        # --- Plotting & Saving Rolling Volatility ---
        fig1, ax1 = plt.subplots(figsize=(12, 6))
        aligned_vol_plot.plot(kind='line', marker='.', linestyle='-', ax=ax1)
        ax1.axvline(0, color='red', linestyle='--', linewidth=1, label='Approval Day')
        ax1.set_title(f'Average Annualized Rolling Volatility Around FDA Approval (Window={window}d)')
        ax1.set_xlabel('Days Relative to Approval'); ax1.set_ylabel('Avg. Annualized Volatility (%)')
        ax1.legend(); ax1.grid(True, alpha=0.5)
        plt.tight_layout()
        plot_filename_vol = os.path.join(results_dir, f"{file_prefix}_volatility_rolling_{window}d.png")
        try:
            plt.savefig(plot_filename_vol); print(f"Saved rolling vol plot to: {plot_filename_vol}")
        except Exception as e: print(f"Error saving rolling vol plot: {e}")
        plt.close(fig1)
        csv_filename_vol = os.path.join(results_dir, f"{file_prefix}_volatility_rolling_{window}d_data.csv")
        try:
            aligned_vol_plot.to_csv(csv_filename_vol, header=True)
            print(f"Saved rolling vol data to: {csv_filename_vol}")
        except Exception as e: print(f"Error saving rolling vol data: {e}")

        # --- Compare Event vs Baseline Volatility & Save Ratio Data/Plot ---
        vol_ratios = []
        for event_id, group in df.groupby('event_id'):
            baseline_data = group[(group['days_to_approval'] >= baseline_window[0]) & (group['days_to_approval'] <= baseline_window[1])]
            event_data = group[(group['days_to_approval'] >= event_window[0]) & (group['days_to_approval'] <= event_window[1])]
            if len(baseline_data) >= min_periods and len(event_data) >= min_periods:
                vol_baseline = baseline_data['ret'].std(); vol_event = event_data['ret'].std()
                if vol_baseline > 0: vol_ratios.append(vol_event / vol_baseline)

        if vol_ratios:
            avg_ratio = np.mean(vol_ratios); median_ratio = np.median(vol_ratios)
            print(f"\nVolatility Spike Analysis (Event: {event_window}, Baseline: {baseline_window}):")
            print(f"  Average Ratio: {avg_ratio:.4f}, Median Ratio: {median_ratio:.4f} ({len(vol_ratios)} events)")

            # Save ratio data
            vol_ratios_series = pd.Series(vol_ratios, name='volatility_ratio')
            csv_filename_ratio = os.path.join(results_dir, f"{file_prefix}_volatility_ratios.csv")
            try:
                vol_ratios_series.to_csv(csv_filename_ratio, index=False, header=True)
                print(f"Saved volatility ratios data to: {csv_filename_ratio}")
            except Exception as e: print(f"Error saving vol ratios data: {e}")

            # Plot histogram of ratios
            fig2, ax2 = plt.subplots(figsize=(8, 5))
            sns.histplot(vol_ratios, bins=30, kde=True, ax=ax2)
            ax2.axvline(1, color='grey', linestyle='--', label='Ratio = 1')
            ax2.axvline(avg_ratio, color='red', linestyle=':', label=f'Mean ({avg_ratio:.2f})')
            ax2.set_title('Distribution of Volatility Ratios (Event / Baseline)')
            ax2.set_xlabel('Volatility Ratio'); ax2.set_ylabel('Frequency')
            ax2.legend()
            plot_filename_hist = os.path.join(results_dir, f"{file_prefix}_volatility_ratio_hist.png")
            try:
                plt.savefig(plot_filename_hist); print(f"Saved vol ratio hist plot to: {plot_filename_hist}")
            except Exception as e: print(f"Error saving vol ratio hist plot: {e}")
            plt.close(fig2)
        else:
            print("\nCould not calculate volatility ratios (insufficient data).")

        return aligned_vol

if __name__ == "__main__":
    print("--- Running FDA Processor Example ---")
    # Define paths (replace with your actual paths)
    FDA_FILE = 'path/to/your/fda_approvals.csv' # Example path
    STOCK_FILES = [
        'path/to/stock_data_part1.csv', # Example path
        'path/to/stock_data_part2.csv'
    ]

    # --- Basic File Existence Check ---
    import os
    if not os.path.exists(FDA_FILE):
        print(f"Error: FDA file not found at {FDA_FILE}")
        print("Please update the FDA_FILE path.")
    elif not all(os.path.exists(f) for f in STOCK_FILES):
        missing_stock = [f for f in STOCK_FILES if not os.path.exists(f)]
        print(f"Error: Stock file(s) not found: {missing_stock}")
        print("Please update the STOCK_FILES list.")
    else:
        # Initialize components
        data_loader = DataLoader(fda_path=FDA_FILE, stock_paths=STOCK_FILES, window_days=60)
        feature_engineer = FeatureEngineer(prediction_window=5) # Predict 5 days ahead
        analyzer = Analysis(data_loader, feature_engineer)

        # Run analysis pipeline
        try:
            analyzer.load_and_prepare_data()
            analyzer.train_models(test_size=0.2)
            results = analyzer.evaluate_models()
            print("\n--- FDA Model Evaluation Results: ---")
            print(pd.DataFrame(results).T)

            # Plot feature importance
            if 'TimeSeriesRidge' in analyzer.models:
                analyzer.plot_feature_importance(model_name='TimeSeriesRidge', top_n=15)
            if 'XGBoostDecile' in analyzer.models:
                analyzer.plot_feature_importance(model_name='XGBoostDecile', top_n=15)

            # Plot predictions for a sample event
            sample_ids = analyzer.find_sample_event_ids(n=1)
            if sample_ids and 'XGBoostDecile' in analyzer.models:
                print(f"\nPlotting predictions for event: {sample_ids[0]}")
                analyzer.plot_predictions_for_event(event_id=sample_ids[0], model_name='XGBoostDecile')
            elif sample_ids and 'TimeSeriesRidge' in analyzer.models:
                 print(f"\nPlotting predictions for event: {sample_ids[0]}")
                 analyzer.plot_predictions_for_event(event_id=sample_ids[0], model_name='TimeSeriesRidge')


        except ValueError as ve: print(f"\nValueError during FDA analysis: {ve}")
        except RuntimeError as re: print(f"\nRuntimeError during FDA analysis: {re}")
        except FileNotFoundError as fnf: print(f"\nFileNotFoundError: {fnf}")
        except Exception as e:
            print(f"\nAn unexpected error occurred during FDA analysis: {e}")
            import traceback
            traceback.print_exc()