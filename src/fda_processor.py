# --- START OF FILE fda_processor.py ---

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
import warnings
import traceback
import os # Added import

# Import shared models
# Ensure src directory is discoverable or adjust path
try:
    from pandasmodels import TimeSeriesRidge, XGBoostDecileModel
except ImportError:
    # Fallback if src isn't directly importable (e.g., running script directly)
    try:
        from pandasmodels import TimeSeriesRidge, XGBoostDecileModel
    except ImportError:
        print("Error: Could not import models from 'src.models' or 'models'.")
        print("Ensure models.py is accessible.")
        # Depending on severity, you might want to exit:
        # import sys
        # sys.exit(1)


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
        stock_paths (list or str): List/single path to stock data PARQUET files.
        window_days (int): Number of days before/after event date.
        """
        self.fda_path = fda_path
        # Ensure stock_paths is always a list
        if isinstance(stock_paths, str): self.stock_paths = [stock_paths]
        elif isinstance(stock_paths, list): self.stock_paths = stock_paths
        else: raise TypeError("stock_paths must be a string or a list of Parquet file paths.")
        self.window_days = window_days

    def _load_single_stock_parquet(self, stock_path):
        """Load and process a single stock data PARQUET file."""
        try:
            print(f"  Reading Parquet file: {stock_path}")
            # Read the Parquet file
            # Specify columns if known and beneficial for memory
            # columns_to_read = ['date', 'ticker', 'prc', 'ret', 'vol', ...]
            # stock_data = pd.read_parquet(stock_path, columns=columns_to_read)
            stock_data = pd.read_parquet(stock_path)
            print(f"  Read {len(stock_data)} rows from {stock_path}")

            # --- Column Name Standardization (Case-Insensitive) ---
            # Store original columns
            original_columns = stock_data.columns.tolist()
            # Create a mapping from lowercase to original case
            col_map_lower = {col.lower(): col for col in original_columns}

            # Define standard names and potential variations (lowercase)
            standard_names = {
                'date': ['date', 'trade_date', 'trading_date', 'tradedate', 'dt'],
                'ticker': ['ticker', 'symbol', 'sym_root', 'tic'],
                'prc': ['prc', 'price', 'close', 'adj close', 'adj_prc'],
                'ret': ['ret', 'return', 'daily_ret'],
                # Add others if needed (vol, open, etc.)
                'vol': ['vol', 'volume'],
                'openprc': ['openprc', 'open'],
                'askhi': ['askhi', 'high', 'askhigh'],
                'bidlo': ['bidlo', 'low', 'bidlow'],
                'shrout': ['shrout', 'shares_outstanding']
            }

            rename_dict = {}
            found_std_names = {}

            for std_name, variations in standard_names.items():
                found = False
                for var in variations:
                    if var in col_map_lower:
                        original_case_col = col_map_lower[var]
                        if original_case_col != std_name:
                            rename_dict[original_case_col] = std_name
                        found_std_names[std_name] = True
                        found = True
                        break # Found the standard name for this variation
                if not found:
                    found_std_names[std_name] = False

            # Apply renaming
            if rename_dict:
                print(f"  Renaming columns: {rename_dict}")
                stock_data.rename(columns=rename_dict, inplace=True)

            # --- Data Type and Existence Checks ---
            # Check for essential columns AFTER potential renaming
            required_cols = ['date', 'ticker', 'prc', 'ret']
            missing_cols = [col for col in required_cols if col not in stock_data.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns after standardization in {stock_path}: {missing_cols}")

            # Ensure date column is datetime
            if not pd.api.types.is_datetime64_any_dtype(stock_data['date']):
                print(f"  Warning: 'date' column in {stock_path} is not datetime. Attempting conversion.")
                try:
                    stock_data['date'] = pd.to_datetime(stock_data['date'], errors='coerce')
                    if stock_data['date'].isna().all(): raise ValueError("All dates failed parse.")
                except Exception as e: raise ValueError(f"Date conversion failed in {stock_path}: {e}")
            # Ensure 'date' column has no NaT values after conversion/loading
            if stock_data['date'].isna().any():
                print(f"  Warning: Found {stock_data['date'].isna().sum()} null dates in {stock_path}. Dropping these rows.")
                stock_data.dropna(subset=['date'], inplace=True)


            # Ensure ticker is string
            if not pd.api.types.is_string_dtype(stock_data['ticker']):
                 stock_data['ticker'] = stock_data['ticker'].astype(str)

            # Ensure numeric columns are numeric, handle potential errors
            # Parquet usually preserves types, but conversion might still be needed if source was mixed
            numeric_cols_to_check = ['prc', 'ret', 'vol', 'openprc', 'askhi', 'bidlo', 'shrout']
            for col in numeric_cols_to_check:
                if col in stock_data.columns:
                    if not pd.api.types.is_numeric_dtype(stock_data[col]):
                        print(f"  Warning: Column '{col}' in {stock_path} is not numeric. Attempting conversion.")
                        stock_data[col] = pd.to_numeric(stock_data[col], errors='coerce')
                        if stock_data[col].isna().any():
                             print(f"  Warning: Coercion introduced NaNs in column '{col}'.")
                    # Check for infinite values
                    if np.isinf(stock_data[col]).any():
                        print(f"  Warning: Found infinite values in column '{col}'. Replacing with NaN.")
                        stock_data[col].replace([np.inf, -np.inf], np.nan, inplace=True)


            # Optional: Select only necessary columns to save memory
            final_cols = list(standard_names.keys()) # Get all standardized names we care about
            cols_present = [col for col in final_cols if col in stock_data.columns]
            stock_data = stock_data[cols_present]

            print(f"  Successfully processed {stock_path}. Final shape: {stock_data.shape}")
            return stock_data

        except FileNotFoundError:
            raise ValueError(f"Stock Parquet file not found: {stock_path}")
        except ImportError:
             raise ImportError("Reading Parquet files requires 'pyarrow'. Please install it: pip install pyarrow")
        except Exception as e:
            # Catch other potential errors during Parquet reading or processing
            raise ValueError(f"Error processing Parquet stock file {stock_path}: {e}")

    def load_data(self):
        """Load FDA approval CSV data and stock Parquet data, then merge them."""
        # --- Load FDA Event Data (CSV) ---
        try:
            print(f"Loading FDA event data from: {self.fda_path} (CSV)")
            fda_data = pd.read_csv(self.fda_path)
            # Basic validation and standardization for FDA data
            if 'Approval Date' not in fda_data.columns: raise ValueError("Missing 'Approval Date' column in FDA CSV.")
            fda_data['Approval Date'] = pd.to_datetime(fda_data['Approval Date'], errors='coerce')
            initial_rows = len(fda_data)
            fda_data.dropna(subset=['Approval Date'], inplace=True)
            if len(fda_data) < initial_rows: print(f"  Dropped {initial_rows - len(fda_data)} rows with invalid Approval Dates.")

            if 'ticker' not in fda_data.columns:
                if 'Ticker' in fda_data.columns:
                    fda_data['ticker'] = fda_data['Ticker'].astype(str).apply(lambda x: x.split(':')[-1].strip() if ':' in x else x.strip())
                else: raise ValueError("Missing 'ticker' or 'Ticker' column in FDA CSV.")
            if 'Drug Name' not in fda_data.columns:
                print("  Warning: 'Drug Name' column not found in FDA data. Using 'N/A'.")
                fda_data['Drug Name'] = "N/A"
            fda_data['ticker'] = fda_data['ticker'].astype(str).str.upper() # Standardize ticker

            approval_events = fda_data[['ticker', 'Approval Date', 'Drug Name']].drop_duplicates().reset_index(drop=True)
            print(f"Found {len(approval_events)} unique FDA approval events.")
            if len(approval_events) == 0: raise ValueError("No valid FDA events found after processing.")
        except FileNotFoundError:
             raise FileNotFoundError(f"FDA CSV file not found: {self.fda_path}")
        except Exception as e: raise ValueError(f"Error loading FDA data from {self.fda_path}: {e}")

        # --- Load Stock Data (Parquet) ---
        stock_data_list, failed_files = [], []
        print("\nLoading stock data from Parquet files...")
        for stock_path in self.stock_paths:
            try:
                # Use the new Parquet loading method
                stock_data = self._load_single_stock_parquet(stock_path)
                # Standardize ticker case AFTER loading
                stock_data['ticker'] = stock_data['ticker'].str.upper()
                stock_data_list.append(stock_data)
            except (ValueError, FileNotFoundError, ImportError) as e: # Catch specific errors from loading func
                print(f"Warning: Failed load/process {stock_path}: {e}")
                failed_files.append(stock_path)
            except Exception as e: # Catch unexpected errors
                print(f"Warning: Unexpected error loading {stock_path}: {e}")
                failed_files.append(stock_path)

        if not stock_data_list: raise ValueError("No stock Parquet data loaded successfully.")
        if failed_files: print(f"\nWarning: Skipped {len(failed_files)} stock file(s): {failed_files}\n")

        # --- Combine and Merge ---
        print("Combining loaded stock data...")
        stock_data_combined = pd.concat(stock_data_list, ignore_index=True)
        print(f"Combined stock data rows before deduplication: {len(stock_data_combined)}")
        # Ensure date and ticker types before deduplication
        stock_data_combined['date'] = pd.to_datetime(stock_data_combined['date'])
        stock_data_combined['ticker'] = stock_data_combined['ticker'].astype(str)
        stock_data_combined = stock_data_combined.drop_duplicates(subset=['date', 'ticker'], keep='first')
        print(f"Combined stock data rows after deduplication: {len(stock_data_combined)}")
        if len(stock_data_combined) == 0: raise ValueError("Stock data empty after deduplication.")

        print("\nMerging FDA events with stock data...")
        tickers_with_approvals = approval_events['ticker'].unique()
        stock_data_filtered = stock_data_combined[stock_data_combined['ticker'].isin(tickers_with_approvals)].copy()
        print(f"Filtered stock data for {len(tickers_with_approvals)} relevant tickers. Rows: {len(stock_data_filtered)}")
        if len(stock_data_filtered) == 0: raise ValueError("No stock data found for tickers listed in FDA events.")

        # Merge event info
        merged_data = pd.merge(stock_data_filtered, approval_events, on='ticker', how='inner')

        # Calculate relative days and filter window
        merged_data['days_to_approval'] = (merged_data['date'] - merged_data['Approval Date']).dt.days
        event_window_data = merged_data[
            (merged_data['days_to_approval'] >= -self.window_days) &
            (merged_data['days_to_approval'] <= self.window_days)
        ].copy()
        if event_window_data.empty: raise ValueError("No stock data found within the specified window for any FDA events.")

        # Add final identifiers
        event_window_data['is_approval_date'] = (event_window_data['days_to_approval'] == 0).astype(int)
        event_window_data['event_id'] = event_window_data['ticker'] + "_" + event_window_data['Approval Date'].dt.strftime('%Y%m%d')
        print(f"Created windows for {event_window_data['event_id'].nunique()} FDA events.")

        # Final sort and return
        combined_data = event_window_data.sort_values(by=['ticker', 'Approval Date', 'date']).reset_index(drop=True)
        print(f"Final FDA dataset shape: {combined_data.shape}")
        # Display memory usage
        mem_usage = combined_data.memory_usage(index=True, deep=True).sum()
        print(f"Final DataFrame memory usage: {mem_usage / 1024**2:.2f} MB")
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
        # Use .pct_change() for robustness against zero prices if shifting division used before
        df['future_price'] = df.groupby('event_id')[price_col].shift(-self.prediction_window)
        df['future_ret'] = (df['future_price'] / df[price_col]) - 1
        # Handle cases where price is zero or future price is missing
        df.loc[df[price_col] <= 0, 'future_ret'] = np.nan
        df.drop(columns=['future_price'], inplace=True)

        print(f"'future_ret' created. Non-null: {df['future_ret'].notna().sum()}")
        return df

    def calculate_features(self, df, price_col='prc', return_col='ret'):
        """Calculate features for FDA analysis."""
        print("Calculating FDA features...")
        df = df.copy()
        required = ['event_id', price_col, return_col, 'Approval Date', 'date', 'days_to_approval'] # days_to_approval needed
        missing = [col for col in required if col not in df.columns]
        if missing: raise ValueError(f"Missing required columns for feature calculation: {missing}")

        df = df.sort_values(by=['event_id', 'date'])
        grouped = df.groupby('event_id')
        current_features = []

        # Price momentum
        for window in self.windows:
            col = f'momentum_{window}'; df[col] = grouped[price_col].transform(lambda x: x.pct_change(periods=window)); current_features.append(col)
        df['delta_momentum_5_10'] = df['momentum_5'] - df['momentum_10']; current_features.append('delta_momentum_5_10')
        df['delta_momentum_10_20'] = df['momentum_10'] - df['momentum_20']; current_features.append('delta_momentum_10_20')

        # Return volatility
        for window in self.windows:
            col = f'volatility_{window}'; min_p = max(2, min(window, 5)); df[col] = grouped[return_col].transform(lambda x: x.rolling(window, min_periods=min_p).std()); current_features.append(col)
        df['delta_volatility_5_10'] = df['volatility_5'] - df['volatility_10']; current_features.append('delta_volatility_5_10')
        df['delta_volatility_10_20'] = df['volatility_10'] - df['volatility_20']; current_features.append('delta_volatility_10_20')

        # Log returns (handle potential zeros/negatives in price)
        df['log_ret'] = grouped[price_col].transform(lambda x: np.log(x / x.shift(1).replace(0, np.nan) )) # Avoid log(0) or log(neg)
        current_features.append('log_ret')

        # Days to approval (already present)
        current_features.append('days_to_approval')

        # Lagged returns
        for lag in range(1, 4): col = f'ret_lag_{lag}'; df[col] = grouped[return_col].shift(lag); current_features.append(col)

        # Pre-approval return
        pre_approval_rets = {}
        for event_id, group in grouped:
            approval_date = group['Approval Date'].iloc[0]
            pre_data = group[(group['date'] <= approval_date) & (group['date'] > approval_date - pd.Timedelta(days=30))]
            # Calculate compound return: (1+ret).prod() - 1 might be better than sum()
            pre_approval_rets[event_id] = (1 + pre_data[return_col].fillna(0)).prod() - 1 if not pre_data.empty else 0
        df['pre_approval_ret_30d'] = df['event_id'].map(pre_approval_rets); current_features.append('pre_approval_ret_30d')

        # Volatility features
        df['prev_day_volatility'] = grouped[return_col].shift(1).abs(); current_features.append('prev_day_volatility')
        df['prev_5d_vol_std'] = grouped[return_col].shift(1).rolling(window=5, min_periods=2).std(); current_features.append('prev_5d_vol_std')

        # Approval date flag (already present)
        if 'is_approval_date' in df.columns: current_features.append('is_approval_date')

        self.feature_names = sorted(list(set(current_features)))
        print(f"Calculated {len(self.feature_names)} raw FDA features.")
        # Replace infinities generated during calculations (e.g., pct_change with 0 denominator)
        df[self.feature_names] = df[self.feature_names].replace([np.inf, -np.inf], np.nan)
        return df

    def get_features_target(self, df, fit_imputer=False):
        """Extract feature matrix X and target vector y, handling missing values."""
        print("Extracting FDA features (X) and target (y)...")
        df = df.copy()
        if not self.feature_names: raise RuntimeError("Run calculate_features first.")
        available_features = [col for col in self.feature_names if col in df.columns]
        if not available_features: raise ValueError("No calculated features found in DataFrame.")

        # Replace any remaining infinities before dropping NaNs
        df[available_features] = df[available_features].replace([np.inf, -np.inf], np.nan)

        df_with_target = df.dropna(subset=['future_ret'])
        if len(df_with_target) == 0:
            print("Warning: No data remains after filtering for non-null target.")
            return pd.DataFrame(columns=available_features), pd.Series(dtype=float)

        X = df_with_target[available_features].copy()
        y = df_with_target['future_ret'].copy()
        print(f"Original X shape (before imputation): {X.shape}. Non-null y count: {len(y)}")
        initial_nan_count = X.isna().sum().sum()
        if initial_nan_count > 0:
             print(f"  Features contain {initial_nan_count} NaN values before imputation.")

        # Impute missing values
        if fit_imputer:
            print("Fitting imputer and transforming features...")
            X_imputed = self.imputer.fit_transform(X)
        else:
            if not hasattr(self.imputer, 'statistics_'): raise RuntimeError("Imputer not fitted.")
            print("Transforming features using pre-fitted imputer...")
            X_imputed = self.imputer.transform(X)

        X = pd.DataFrame(X_imputed, columns=available_features, index=X.index)

        final_nan_count = X.isna().sum().sum()
        if final_nan_count > 0:
            warnings.warn(f"NaNs ({final_nan_count}) remain AFTER imputation!")
        elif initial_nan_count > 0:
             print("No NaNs remaining after imputation.")
        else:
             print("No NaNs found before or after imputation.")


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
        print("--- Loading FDA Data (CSV) and Stock Data (Parquet) ---")
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
        self.X_train, self.y_train = self.feature_engineer.get_features_target(self.data.loc[self.train_indices].copy(), fit_imputer=True) # Pass copy
        print("\nExtracting features/target for TEST set (transforming)...")
        self.X_test, self.y_test = self.feature_engineer.get_features_target(self.data.loc[self.test_indices].copy(), fit_imputer=False) # Pass copy
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

    def plot_feature_importance(self, results_dir, file_prefix="fda", model_name='TimeSeriesRidge', top_n=20):
        """Plot feature importance and save the plot."""
        print(f"\n--- Plotting FDA Feature Importance for {model_name} ---")
        # ... (Plotting logic remains largely the same as previous version) ...
        if model_name not in self.models: print(f"Error: Model '{model_name}' not found."); return None
        model = self.models[model_name]; feature_names = getattr(self.feature_engineer, 'final_feature_names', None)
        if not feature_names: print("Error: Final feature names not found."); return None

        importances = None
        # (Get importances logic - same as before)
        if isinstance(model, TimeSeriesRidge):
            if hasattr(model, 'coef_'): importances = np.abs(model.coef_)
        elif isinstance(model, XGBoostDecileModel):
             if hasattr(model, 'xgb_model') and hasattr(model.xgb_model, 'feature_importances_'):
                 try:
                     xgb_feat_names = model.xgb_model.get_booster().feature_names
                     if xgb_feat_names: imp_dict = dict(zip(xgb_feat_names, model.xgb_model.feature_importances_)); importances = np.array([imp_dict.get(name, 0) for name in feature_names])
                     else: importances = model.xgb_model.feature_importances_
                 except Exception: importances = model.xgb_model.feature_importances_
        if importances is None: print(f"Could not get importance for {model_name}."); return None

        if len(importances) != len(feature_names):
             print(f"Warn: Importance len ({len(importances)}) != feature len ({len(feature_names)}). Aligning.");
             if len(importances) > len(feature_names): importances = importances[:len(feature_names)]
             elif len(importances) < len(feature_names): importances = np.pad(importances, (0, len(feature_names) - len(importances)))

        feat_imp_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances}).sort_values('Importance', ascending=False).head(top_n)

        # Plotting
        fig, ax = plt.subplots(figsize=(10, max(6, top_n // 2)))
        sns.barplot(x='Importance', y='Feature', data=feat_imp_df, palette='viridis', ax=ax)
        ax.set_title(f'Top {top_n} Features by Importance ({model_name} - FDA)')
        ax.set_xlabel('Importance Score'); ax.set_ylabel('Feature')
        plt.tight_layout()

        # Save plot
        plot_filename = os.path.join(results_dir, f"{file_prefix}_feat_importance_{model_name}.png")
        try:
            plt.savefig(plot_filename); print(f"Saved feature importance plot to: {plot_filename}")
        except Exception as e: print(f"Error saving feature importance plot: {e}")
        plt.close(fig)
        return feat_imp_df


    def plot_predictions_for_event(self, results_dir, event_id, file_prefix="fda", model_name='TimeSeriesRidge'):
        """Plot actual vs. predicted returns for a specific event and save the plot."""
        print(f"\n--- Plotting FDA Predictions for Event: {event_id} ({model_name}) ---")
        # ... (Prediction logic remains the same) ...
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

        # Plotting
        fig, ax = plt.subplots(figsize=(14, 6))
        ax.plot(event_data_full['date'], event_data_full['ret'], '.-', label='Actual Daily Return', alpha=0.6, color='blue')
        ax.scatter(event_data_pred['date'], event_data_pred['predicted_future_ret'],
                    color='red', label=f'Predicted {self.feature_engineer.prediction_window}-day Future Ret', s=15, zorder=5, alpha=0.8)
        ax.axvline(x=approval_date, color='g', linestyle='--', label='Approval Date')
        ax.set_title(f"{ticker} ({event_id}) - Daily Returns & Predicted Future Returns ({model_name} - FDA)")
        ax.set_ylabel("Return"); ax.set_xlabel("Date"); ax.legend(); ax.grid(True, alpha=0.3)
        plt.tight_layout()

        # Save plot
        # Sanitize event_id for filename if needed (e.g., replace special chars)
        safe_event_id = "".join(c if c.isalnum() else "_" for c in event_id)
        plot_filename = os.path.join(results_dir, f"{file_prefix}_pred_vs_actual_{safe_event_id}_{model_name}.png")
        try:
            plt.savefig(plot_filename); print(f"Saved prediction plot to: {plot_filename}")
        except Exception as e: print(f"Error saving prediction plot: {e}")
        plt.close(fig)

        return event_data_pred


    def find_sample_event_ids(self, n=5):
        """Find sample FDA event identifiers."""
        if self.data is None or 'event_id' not in self.data.columns: return []
        unique_events = self.data['event_id'].unique()
        return list(unique_events[:min(n, len(unique_events))])

    # --- Keep analyze_sharpe_ratio_dynamics and analyze_volatility_spikes ---
    # These methods were updated in the previous step to handle saving results
    # No further changes needed here for them.
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

    def calculate_rolling_sharpe_timeseries(self, results_dir: str, file_prefix: str = "fda",
                                  return_col: str = 'ret', 
                                  analysis_window: tuple = (-60, 60),
                                  sharpe_window: int = 5,
                                  annualize: bool = True, 
                                  risk_free_rate: float = 0.0):
        """
        Calculates a time series of rolling Sharpe ratios around FDA approvals.

        Parameters:
        results_dir (str): Directory to save results
        file_prefix (str): Prefix for saved files
        return_col (str): Column name containing returns
        analysis_window (tuple): Days relative to approval to analyze (start, end)
        sharpe_window (int): Size of window for rolling Sharpe calculation in days
        annualize (bool): Whether to annualize the Sharpe ratio
        risk_free_rate (float): Annualized risk-free rate for Sharpe calculation

        Returns:
        pd.DataFrame: DataFrame containing Sharpe ratio time series
        """
        print(f"\n--- Calculating Rolling Sharpe Ratio Time Series (Analysis Window: {analysis_window}, Sharpe Window: {sharpe_window}) ---")
        if self.data is None or return_col not in self.data.columns:
            print("Error: Data not loaded or missing return column.")
            return None

        # Filter data to include only days within the extended analysis window
        analysis_data = self.data[(self.data['days_to_approval'] >= analysis_window[0]) & 
                                 (self.data['days_to_approval'] <= analysis_window[1])].copy()

        if analysis_data.empty:
            print(f"Error: No data found within analysis window {analysis_window}.")
            return None

        # Calculate daily equivalent of risk-free rate if needed
        daily_rf = (1 + risk_free_rate) ** (1/252) - 1 if risk_free_rate > 0 else 0

        # Initialize results storage
        days_range = list(range(analysis_window[0], analysis_window[1] + 1))
        sharpe_results = []

        # Half window size for calculations
        half_window = sharpe_window // 2

        # For each day in the analysis window, calculate Sharpe ratio using surrounding days
        for center_day in days_range:
            window_start = center_day - half_window
            window_end = center_day + half_window

            # Filter data for current window
            window_data = analysis_data[(analysis_data['days_to_approval'] >= window_start) & 
                                       (analysis_data['days_to_approval'] <= window_end)]

            if len(window_data) < sharpe_window // 2:  # Not enough data in window
                sharpe_results.append({
                    'days_to_approval': center_day,
                    'mean_return': np.nan,
                    'std_dev': np.nan,
                    'sharpe_ratio': np.nan,
                    'num_observations': 0
                })
                continue
            
            # Calculate statistics for window
            mean_return = window_data[return_col].mean()
            std_dev = window_data[return_col].std()
            num_obs = len(window_data)

            # Calculate Sharpe ratio
            sharpe = np.nan
            if not np.isnan(mean_return) and not np.isnan(std_dev) and std_dev > 0:
                sharpe = (mean_return - daily_rf) / std_dev

                # Annualize if requested
                if annualize:
                    sharpe = sharpe * np.sqrt(252)

            sharpe_results.append({
                'days_to_approval': center_day,
                'mean_return': mean_return,
                'std_dev': std_dev,
                'sharpe_ratio': sharpe,
                'num_observations': num_obs
            })

        # Convert results to DataFrame
        results_df = pd.DataFrame(sharpe_results)

        # Create a second DataFrame with event counts for each day
        event_counts = analysis_data.groupby('days_to_approval')['event_id'].nunique().reset_index()
        event_counts.columns = ['days_to_approval', 'unique_events']

        # Join results with event counts
        results_with_counts = pd.merge(results_df, event_counts, on='days_to_approval', how='left')
        results_with_counts['unique_events'] = results_with_counts['unique_events'].fillna(0)

        # Plot the results
        try:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [3, 1]})

            # Sharpe ratio plot
            ax1.plot(results_with_counts['days_to_approval'], results_with_counts['sharpe_ratio'], 'b-', linewidth=2)
            ax1.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
            ax1.axvline(x=0, color='red', linestyle='--', linewidth=1.5, label='Approval Day')

            # Add vertical lines at key periods if using extended window
            if analysis_window[0] <= -30 and analysis_window[1] >= 30:
                ax1.axvline(x=-30, color='green', linestyle=':', linewidth=1, label='Month Before')
                ax1.axvline(x=30, color='purple', linestyle=':', linewidth=1, label='Month After')

            # Highlight the event window commonly used for volatility analysis
            event_start, event_end = -2, 2  # Standard event window
            if analysis_window[0] <= event_start and analysis_window[1] >= event_end:
                ax1.axvspan(event_start, event_end, alpha=0.2, color='yellow', label='Event Window (-2 to +2)')

            ax1.set_title(f'Rolling Sharpe Ratio Around FDA Approvals (Window Size: {sharpe_window} days)')
            ax1.set_xlabel('Days Relative to Approval')
            ax1.set_ylabel('Sharpe Ratio')
            ax1.legend(loc='best')
            ax1.grid(True, alpha=0.3)

            # Calculate and plot rolling average of Sharpe
            if len(results_with_counts) >= 10:  # Need enough data for meaningful average
                window_size = min(10, len(results_with_counts) // 5)
                results_with_counts['rolling_avg'] = results_with_counts['sharpe_ratio'].rolling(window=window_size, center=True).mean()
                ax1.plot(results_with_counts['days_to_approval'], results_with_counts['rolling_avg'], 'r-', 
                        linewidth=1.5, label=f'{window_size}-Day Rolling Avg')
                ax1.legend(loc='best')

            # Event count plot
            ax2.bar(results_with_counts['days_to_approval'], results_with_counts['unique_events'], color='lightblue')
            ax2.axvline(x=0, color='red', linestyle='--', linewidth=1.5)
            ax2.set_title('Number of Events per Day')
            ax2.set_xlabel('Days Relative to Approval')
            ax2.set_ylabel('Event Count')
            ax2.grid(True, alpha=0.3)

            plt.tight_layout()
            plot_filename = os.path.join(results_dir, f"{file_prefix}_rolling_sharpe_timeseries.png")
            plt.savefig(plot_filename)
            print(f"Saved rolling Sharpe time series plot to: {plot_filename}")
            plt.close(fig)

            # Also create a heatmap showing mean return and volatility
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))

            # Mean return heatmap-style plot
            mean_returns_df = results_with_counts[['days_to_approval', 'mean_return']].set_index('days_to_approval').T
            sns.heatmap(mean_returns_df, cmap='RdYlGn', center=0, ax=ax1, cbar_kws={'label': 'Mean Return'})
            ax1.set_title('Mean Return by Day Relative to Approval')
            ax1.set_ylabel('')

            # Volatility heatmap-style plot
            volatility_df = results_with_counts[['days_to_approval', 'std_dev']].set_index('days_to_approval').T
            sns.heatmap(volatility_df, cmap='YlOrRd', ax=ax2, cbar_kws={'label': 'Return Volatility'})
            ax2.set_title('Return Volatility by Day Relative to Approval')
            ax2.set_ylabel('')
            ax2.set_xlabel('Days Relative to Approval')

            plt.tight_layout()
            heatmap_filename = os.path.join(results_dir, f"{file_prefix}_return_volatility_heatmap.png")
            plt.savefig(heatmap_filename)
            print(f"Saved return and volatility heatmap to: {heatmap_filename}")
            plt.close(fig)

        except Exception as e:
            print(f"Error creating plots: {e}")
            traceback.print_exc()

        # Save results to CSV
        csv_filename = os.path.join(results_dir, f"{file_prefix}_rolling_sharpe_timeseries.csv")
        try:
            results_with_counts.to_csv(csv_filename, index=False)
            print(f"Saved rolling Sharpe time series data to: {csv_filename}")
        except Exception as e:
            print(f"Error saving time series data: {e}")

        return results_with_counts

    def calculate_sharpe_quantiles(self, results_dir: str, file_prefix: str = "fda",
                              return_col: str = 'ret', 
                              analysis_window: tuple = (-60, 60),
                              lookback_window: int = 10,
                              quantiles: list = [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95],
                              annualize: bool = True, 
                              risk_free_rate: float = 0.0):
        """
        Calculates and plots quantiles of Sharpe ratios across events for each day around FDA approvals.

        Parameters:
        results_dir (str): Directory to save results
        file_prefix (str): Prefix for saved files
        return_col (str): Column name containing returns
        analysis_window (tuple): Days relative to approval to analyze (start, end)
        lookback_window (int): Window size for calculating Sharpe ratios (# of days)
        quantiles (list): List of quantiles to calculate
        annualize (bool): Whether to annualize the Sharpe ratio
        risk_free_rate (float): Annualized risk-free rate for Sharpe calculation

        Returns:
        pd.DataFrame: DataFrame containing Sharpe ratio quantiles by day
        """
        print(f"\n--- Calculating Sharpe Ratio Quantiles (Analysis Window: {analysis_window}, Lookback: {lookback_window}) ---")

        if self.data is None or return_col not in self.data.columns:
            print("Error: Data not loaded or missing return column.")
            return None

        # Filter data to include only days within the extended analysis window
        # We'll need extra days before for the lookback window calculations
        extended_start = analysis_window[0] - lookback_window
        analysis_data = self.data[(self.data['days_to_approval'] >= extended_start) & 
                                 (self.data['days_to_approval'] <= analysis_window[1])].copy()

        if analysis_data.empty:
            print(f"Error: No data found within analysis window {analysis_window}.")
            return None

        # Calculate daily equivalent of risk-free rate if needed
        daily_rf = (1 + risk_free_rate) ** (1/252) - 1 if risk_free_rate > 0 else 0

        # Initialize results storage
        days_range = list(range(analysis_window[0], analysis_window[1] + 1))
        results_data = []

        # For each day in the analysis window
        print(f"Processing {len(days_range)} days...")
        for center_day in days_range:
            # Calculate the window for Sharpe calculation (lookback period)
            window_start = center_day - lookback_window
            window_end = center_day

            # Get data for all events in this window period
            window_data = analysis_data[(analysis_data['days_to_approval'] >= window_start) & 
                                       (analysis_data['days_to_approval'] <= window_end)]

            if window_data.empty:
                # Record empty result and continue if no data for this day
                empty_results = {"days_to_approval": center_day, "event_count": 0}
                for q in quantiles:
                    empty_results[f"sharpe_q{int(q*100)}"] = None
                results_data.append(empty_results)
                continue
            
            # Get all unique event IDs for this window
            event_ids = window_data['event_id'].unique()

            # Calculate Sharpe ratio for each event in this window
            event_sharpes = []
            valid_event_count = 0

            for event_id in event_ids:
                # Get data for this specific event in the window
                event_data = window_data[window_data['event_id'] == event_id]

                # Need sufficient data points for a meaningful calculation
                if len(event_data) < max(3, lookback_window // 3):
                    continue

                # Calculate mean return and std dev for this event
                mean_ret = event_data[return_col].mean()
                std_dev = event_data[return_col].std()

                # Calculate Sharpe ratio for this event
                if not np.isnan(std_dev) and std_dev > 0:
                    sharpe = (mean_ret - daily_rf) / std_dev

                    # Annualize if requested
                    if annualize:
                        sharpe = sharpe * np.sqrt(252)

                    event_sharpes.append(sharpe)
                    valid_event_count += 1

            # Calculate quantiles if we have enough events
            day_results = {"days_to_approval": center_day, "event_count": valid_event_count}

            if valid_event_count >= 5:  # Need a reasonable number of events for quantiles
                sharpe_array = np.array(event_sharpes)
                for q in quantiles:
                    quantile_value = np.nanquantile(sharpe_array, q)
                    day_results[f"sharpe_q{int(q*100)}"] = quantile_value
            else:
                # Not enough events for reliable quantiles
                for q in quantiles:
                    day_results[f"sharpe_q{int(q*100)}"] = None

            results_data.append(day_results)

        # Convert results to DataFrame
        results_df = pd.DataFrame(results_data)

        # Plot the results
        try:
            if 'event_count' in results_df.columns and not results_df.empty:
                results_df.set_index('days_to_approval', inplace=True)

                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [3, 1]})

                # Plot quantiles
                # Colors from light to dark
                colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(quantiles)))

                # Need to ensure all required columns exist
                expected_cols = [f"sharpe_q{int(q*100)}" for q in quantiles]
                missing_cols = set(expected_cols) - set(results_df.columns)
                if missing_cols:
                    print(f"Warning: Missing columns in results: {missing_cols}")
                    # Filter expected_cols to only include available columns
                    expected_cols = [col for col in expected_cols if col in results_df.columns]

                if expected_cols:  # If we have any valid columns
                    for i, q in enumerate(quantiles):
                        col_name = f"sharpe_q{int(q*100)}"
                        if col_name in results_df.columns:
                            ax1.plot(results_df.index, results_df[col_name], 
                                    color=colors[i], linewidth=1.5, 
                                    label=f"{int(q*100)}th percentile", alpha=0.7)

                # Highlight median more strongly
                median_col = "sharpe_q50"
                if median_col in results_df.columns:
                    ax1.plot(results_df.index, results_df[median_col], 
                           color='red', linewidth=2.5, 
                           label=f"Median (50th)", alpha=0.9)

                # Add reference lines
                ax1.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
                ax1.axvline(x=0, color='red', linestyle='--', linewidth=1.5, label='Approval Day')

                # Add vertical lines at key periods
                if analysis_window[0] <= -30 and analysis_window[1] >= 30:
                    ax1.axvline(x=-30, color='green', linestyle=':', linewidth=1, label='Month Before')
                    ax1.axvline(x=30, color='purple', linestyle=':', linewidth=1, label='Month After')

                # Highlight the event window commonly used for volatility analysis
                event_start, event_end = -2, 2  # Standard event window
                if analysis_window[0] <= event_start and analysis_window[1] >= event_end:
                    ax1.axvspan(event_start, event_end, alpha=0.1, color='yellow', label='Event Window (-2 to +2)')

                ax1.set_title(f'Sharpe Ratio Quantiles Around FDA Approvals (Lookback: {lookback_window} days)')
                ax1.set_xlabel('Days Relative to Approval')
                ax1.set_ylabel('Sharpe Ratio')
                ax1.legend(loc='best')
                ax1.grid(True, alpha=0.3)

                # Event count plot
                if 'event_count' in results_df.columns:
                    ax2.bar(results_df.index, results_df['event_count'], color='lightblue')
                    ax2.axvline(x=0, color='red', linestyle='--', linewidth=1.5)
                    ax2.set_title('Number of Events With Valid Sharpe Ratio')
                    ax2.set_xlabel('Days Relative to Approval')
                    ax2.set_ylabel('Event Count')
                    ax2.grid(True, alpha=0.3)

                plt.tight_layout()
                plot_filename = os.path.join(results_dir, f"{file_prefix}_sharpe_ratio_quantiles.png")
                plt.savefig(plot_filename)
                print(f"Saved Sharpe ratio quantiles plot to: {plot_filename}")
                plt.close(fig)

                # Create a heat map visualization of the quantiles
                try:
                    # Drop the event_count column and transpose
                    heatmap_data = results_df.drop(columns=['event_count']).T

                    # Rename index to more readable percentile names
                    new_index = [f"{int(q*100)}th" for q in quantiles]
                    heatmap_data.index = new_index

                    fig, ax = plt.subplots(figsize=(16, 8))

                    sns.heatmap(heatmap_data, cmap='RdYlGn', center=0, 
                               ax=ax, cbar_kws={'label': 'Sharpe Ratio'})

                    ax.set_title('Sharpe Ratio Quantiles Heatmap')
                    ax.set_xlabel('Days Relative to Approval')
                    ax.set_ylabel('Percentile')

                    plt.tight_layout()
                    heatmap_filename = os.path.join(results_dir, f"{file_prefix}_sharpe_quantiles_heatmap.png")
                    plt.savefig(heatmap_filename)
                    print(f"Saved Sharpe ratio quantiles heatmap to: {heatmap_filename}")
                    plt.close(fig)
                except Exception as e:
                    print(f"Error creating heatmap: {e}")
            else:
                print("Warning: No valid data for plotting")

        except Exception as e:
            print(f"Error creating plots: {e}")
            traceback.print_exc()

        # Save results to CSV
        csv_filename = os.path.join(results_dir, f"{file_prefix}_sharpe_ratio_quantiles.csv")
        try:
            results_df.to_csv(csv_filename)
            print(f"Saved Sharpe ratio quantiles data to: {csv_filename}")
        except Exception as e:
            print(f"Error saving quantiles data: {e}")

        return results_df