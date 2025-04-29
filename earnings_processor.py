import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report
from sklearn.impute import SimpleImputer
import xgboost as xgb
import warnings

# Import shared models
from models import TimeSeriesRidge, XGBoostDecileModel

# Suppress SettingWithCopyWarning for cleaner output
warnings.filterwarnings('ignore', category=pd.errors.SettingWithCopyWarning)
# Suppress FutureWarnings from seaborn/matplotlib
warnings.simplefilter(action='ignore', category=FutureWarning)

class DataLoader:
    def __init__(self, earnings_path, stock_paths, window_days=30):
        """
        Initialize DataLoader for Earnings events.

        Parameters:
        earnings_path (str): Path to the event data CSV (e.g., IBES Summary or Actuals).
                             *Must contain ticker and the actual announcement date*.
        stock_paths (list or str): List/single path to stock price/return data CSV files.
        window_days (int): Number of days before/after event date.
        """
        self.earnings_path = earnings_path
        if isinstance(stock_paths, str): self.stock_paths = [stock_paths]
        elif isinstance(stock_paths, list): self.stock_paths = stock_paths
        else: raise TypeError("stock_paths must be a string or a list.")
        self.window_days = window_days

    def _load_single_stock_file(self, stock_path):
        """Load and process a single stock data file. (Identical to previous version)"""
        try:
            headers = pd.read_csv(stock_path, nrows=0).columns
            date_col = next((col for col in headers if col.lower() in ['date', 'trade_date', 'trading_date', 'tradedate', 'dt']), None)
            if date_col:
                stock_data = pd.read_csv(stock_path, parse_dates=[date_col]).rename(columns={date_col: 'date'})
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

            required_cols = ['date', 'ticker', 'prc', 'ret', 'vol']
            missing_cols = [col for col in required_cols if col not in stock_data.columns]
            if missing_cols:
                 essential_pr = ['prc', 'ret']
                 if any(col in missing_cols for col in essential_pr):
                      raise ValueError(f"Missing essential columns in {stock_path}: {[c for c in essential_pr if c in missing_cols]}")
                 else: print(f"Warning: Missing optional columns in {stock_path}: {missing_cols}.")

            numeric_cols = ['prc', 'ret', 'vol', 'openprc', 'askhi', 'bidlo', 'shrout']
            for col in numeric_cols:
                if col in stock_data.columns: stock_data[col] = pd.to_numeric(stock_data[col], errors='coerce')
            return stock_data
        except FileNotFoundError: raise ValueError(f"Stock file not found: {stock_path}")
        except Exception as e: raise ValueError(f"Error processing {stock_path}: {e}")

    def load_data(self):
        """
        Load earnings event dates (from IBES summary/actuals) and stock data,
        then merge them. Focuses only on Ticker and Date from the event file.
        """
        # --- Load Earnings Event Dates ---
        try:
            print(f"Loading earnings event dates from: {self.earnings_path}")
            # Load only necessary columns if possible, otherwise load all and select
            # --> Determine necessary cols first
            event_df_peek = pd.read_csv(self.earnings_path, nrows=5) # Peek at header

            # Find Ticker Column
            ticker_col = next((c for c in ['TICKER', 'ticker', 'Ticker', 'symbol', 'tic'] if c in event_df_peek.columns), None)
            if not ticker_col: raise ValueError("Missing Ticker column (or recognized alternative) in event file.")

            # Find Announcement Date Column (MUST be actual announcement date)
            # Add ANNDATS based on OCR
            date_col = next((c for c in ['ANNDATS', 'Announcement Date', 'date', 'report_date', 'rdq'] if c in event_df_peek.columns), None)
            if not date_col: raise ValueError("Missing Announcement Date column (e.g., ANNDATS, Announcement Date, date) in event file.")

            print(f"Using columns '{ticker_col}' and '{date_col}' from event file.")

            # Load only these two columns for efficiency
            cols_to_load = [ticker_col, date_col]
            event_data_raw = pd.read_csv(self.earnings_path, usecols=cols_to_load)

            # Rename to standard names
            event_data_renamed = event_data_raw.rename(columns={
                ticker_col: 'ticker',
                date_col: 'Announcement Date'
            })

            # Convert date and handle errors
            event_data_renamed['Announcement Date'] = pd.to_datetime(event_data_renamed['Announcement Date'], errors='coerce')
            initial_event_count = len(event_data_renamed)
            event_data_renamed.dropna(subset=['Announcement Date'], inplace=True)
            if len(event_data_renamed) < initial_event_count:
                 print(f"Warning: Dropped {initial_event_count - len(event_data_renamed)} rows from event file due to invalid dates.")

            # Convert ticker to uppercase
            event_data_renamed['ticker'] = event_data_renamed['ticker'].astype(str).str.upper()

            # Get unique ticker-date pairs
            earnings_events = event_data_renamed[['ticker', 'Announcement Date']].drop_duplicates().reset_index(drop=True)

            print(f"Found {len(earnings_events)} unique earnings events (Ticker-Date pairs).")
            if len(earnings_events) == 0: raise ValueError("No valid earnings events found after processing.")

        except FileNotFoundError: raise ValueError(f"Earnings event file not found: {self.earnings_path}")
        except Exception as e: raise ValueError(f"Error processing earnings event file {self.earnings_path}: {e}")

        # --- Load Stock Data (from potentially multiple files) ---
        # (This section remains unchanged)
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

        # --- Merge Event Dates and Stock Data ---
        tickers_with_earnings = earnings_events['ticker'].unique()
        stock_data_filtered = stock_data_combined[stock_data_combined['ticker'].isin(tickers_with_earnings)].copy()
        print(f"Filtered stock data for {len(tickers_with_earnings)} tickers. Rows: {len(stock_data_filtered)}")
        if len(stock_data_filtered) == 0: raise ValueError("No stock data for relevant tickers.")

        # Merge based only on ticker (event date comes from earnings_events)
        # We need to be careful if a ticker has multiple events
        merged_data = pd.merge(
            stock_data_filtered,
            earnings_events, # Contains only ticker, Announcement Date
            on='ticker',
            how='inner'
        )

        # Calculate date difference relative to *each specific event date* for that ticker
        merged_data['days_to_announcement'] = (merged_data['date'] - merged_data['Announcement Date']).dt.days

        # Filter for the window around each specific event
        event_window_data = merged_data[
            (merged_data['days_to_announcement'] >= -self.window_days) &
            (merged_data['days_to_announcement'] <= self.window_days)
        ].copy()

        if event_window_data.empty: raise ValueError("No stock data found within the specified window for any earnings events.")

        # Add derived columns
        event_window_data['is_announcement_date'] = (event_window_data['days_to_announcement'] == 0).astype(int)
        event_window_data['event_id'] = event_window_data['ticker'] + "_" + event_window_data['Announcement Date'].dt.strftime('%Y%m%d')

        # **Crucially, remove columns from the event file we don't need downstream**
        # Keep only columns originating from stock data + the ones we just created/standardized
        stock_cols = list(stock_data_filtered.columns)
        derived_cols = ['Announcement Date', 'days_to_announcement', 'is_announcement_date', 'event_id']
        final_cols = [col for col in event_window_data.columns if col in stock_cols or col in derived_cols]
        event_window_data = event_window_data[final_cols]

        print(f"Created windows for {event_window_data['event_id'].nunique()} unique earnings events.")
        combined_data = event_window_data.sort_values(by=['ticker', 'Announcement Date', 'date']).reset_index(drop=True)
        print(f"Final Earnings dataset shape: {combined_data.shape}")
        # print(f"Final columns: {combined_data.columns.tolist()}") # Optional: check columns
        return combined_data


# --- FeatureEngineer Class ---
# (Modified to be robust to missing optional columns like Surprise, Time, Quarter, Sector, Industry)
class FeatureEngineer:
    def __init__(self, prediction_window=3):
        self.windows = [5, 10, 20]
        self.prediction_window = prediction_window
        self.imputer = SimpleImputer(strategy='median')
        self.feature_names = []
        self.final_feature_names = []
        self.categorical_features = [] # Store names of created categorical/dummy features
        self.sector_dummies_cols = None
        self.industry_dummies_cols = None
        self.top_industries = None

    def create_target(self, df, price_col='prc'):
        """Create target variable for Earnings analysis. (Unchanged)"""
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

    def calculate_features(self, df, price_col='prc', return_col='ret', volume_col='vol', fit_categorical=False):
        """Calculate features for Earnings analysis. Robust to missing optional columns."""
        print("Calculating Earnings features...")
        df = df.copy()
        required = ['event_id', price_col, return_col, 'Announcement Date', 'date'] # Base requirements
        if not all(c in df.columns for c in required): raise ValueError(f"Missing required columns: {required}")
        has_volume = volume_col in df.columns
        if not has_volume: print(f"Info: Volume column '{volume_col}' not found. Volume features skipped.")

        df = df.sort_values(by=['event_id', 'date'])
        grouped = df.groupby('event_id')
        current_features = []
        self.categorical_features = [] # Reset

        # --- Technical Features (Should always be calculable) ---
        for window in self.windows: col = f'momentum_{window}'; df[col] = grouped[price_col].transform(lambda x: x.pct_change(periods=window)); current_features.append(col)
        df['delta_momentum_5_10'] = df['momentum_5'] - df['momentum_10']; current_features.append('delta_momentum_5_10')
        df['delta_momentum_10_20'] = df['momentum_10'] - df['momentum_20']; current_features.append('delta_momentum_10_20')
        for window in self.windows: col = f'volatility_{window}'; min_p = max(2, min(window, 5)); df[col] = grouped[return_col].transform(lambda x: x.rolling(window, min_periods=min_p).std()); current_features.append(col)
        df['delta_volatility_5_10'] = df['volatility_5'] - df['volatility_10']; current_features.append('delta_volatility_5_10')
        df['delta_volatility_10_20'] = df['volatility_10'] - df['volatility_20']; current_features.append('delta_volatility_10_20')
        df['log_ret'] = grouped[price_col].transform(lambda x: np.log(x / x.shift(1))); current_features.append('log_ret')
        if 'days_to_announcement' in df.columns: current_features.append('days_to_announcement')
        for lag in range(1, 4): col = f'ret_lag_{lag}'; df[col] = grouped[return_col].shift(lag); current_features.append(col)

        # --- Volume Features (Conditional) ---
        if has_volume:
            df['norm_vol'] = grouped[volume_col].transform(lambda x: x / x.rolling(20, min_periods=5).mean()); current_features.append('norm_vol')
            for window in [5, 10]: col = f'vol_momentum_{window}'; df[col] = grouped[volume_col].transform(lambda x: x.pct_change(periods=window)); current_features.append(col)
        else: # Add placeholders if volume missing
             for col in ['norm_vol', 'vol_momentum_5', 'vol_momentum_10']: df[col] = np.nan; current_features.append(col)

        # --- Pre-announcement Return (Should always be calculable) ---
        pre_announce_rets = {}
        for event_id, group in grouped:
            announce_date = group['Announcement Date'].iloc[0]
            pre_data = group[(group['date'] < announce_date) & (group['date'] >= announce_date - pd.Timedelta(days=30))]
            pre_announce_rets[event_id] = pre_data[return_col].sum() if not pre_data.empty else 0
        df['pre_announce_ret_30d'] = df['event_id'].map(pre_announce_rets); current_features.append('pre_announce_ret_30d')

        # --- Earnings Surprise Features (Conditional) ---
        # Check for 'Surprise' column loaded *by the DataLoader*
        if 'Surprise' in df.columns:
            df['surprise_val'] = pd.to_numeric(df['Surprise'], errors='coerce').fillna(0)
            df['pos_surprise'] = (df['surprise_val'] > 0).astype(int); current_features.append('pos_surprise')
            df['neg_surprise'] = (df['surprise_val'] < 0).astype(int); current_features.append('neg_surprise')
            df['surprise_magnitude'] = df['surprise_val'].abs(); current_features.append('surprise_magnitude')
            current_features.append('surprise_val') # Keep the numeric value itself

            # Previous surprise calculation (remains the same logic, depends only on surprise_val)
            df['prev_surprise'] = df.groupby('ticker')['surprise_val'].shift(1).fillna(0)
            df['consecutive_beat'] = ((df['surprise_val'] > 0) & (df['prev_surprise'] > 0)).astype(int); current_features.append('consecutive_beat')
            df['consecutive_miss'] = ((df['surprise_val'] < 0) & (df['prev_surprise'] < 0)).astype(int); current_features.append('consecutive_miss')
            current_features.append('prev_surprise')
        else:
            print("Info: 'Surprise' column not loaded by DataLoader. Skipping surprise features.")
            # Add NaN placeholders if needed for consistent feature set downstream
            for col in ['surprise_val', 'pos_surprise', 'neg_surprise', 'surprise_magnitude', 'prev_surprise', 'consecutive_beat', 'consecutive_miss']:
                df[col] = np.nan
                current_features.append(col) # Still add name to list, will be imputed later

        # --- Announcement Time Features (Conditional) ---
        if 'Time' in df.columns:
             try:
                 time_parsed = pd.to_datetime(df['Time'], errors='coerce', format='%H:%M:%S').dt # Try specific
                 if time_parsed.isna().all(): time_parsed = pd.to_datetime(df['Time'], errors='coerce').dt # Try generic
                 df['announcement_hour'] = time_parsed.hour.fillna(-1).astype(int)
                 df['is_bmo'] = ((df['announcement_hour'] >= 0) & (df['announcement_hour'] < 9)).astype(int); current_features.append('is_bmo')
                 df['is_amc'] = ((df['announcement_hour'] >= 16) & (df['announcement_hour'] <= 23)).astype(int); current_features.append('is_amc')
                 df['is_market_hours'] = ((df['announcement_hour'] >= 9) & (df['announcement_hour'] < 16)).astype(int); current_features.append('is_market_hours')
                 current_features.append('announcement_hour')
             except Exception as e: print(f"Warning: Could not parse 'Time' column: {e}")
        else:
            print("Info: 'Time' column not loaded by DataLoader. Skipping time features.")
            for col in ['announcement_hour', 'is_bmo', 'is_amc', 'is_market_hours']: df[col] = np.nan; current_features.append(col)

        # --- Quarter Features (Conditional) ---
        if 'Quarter' in df.columns:
            try:
                df['quarter_num'] = df['Quarter'].astype(str).str.extract(r'Q(\d)', expand=False).astype(float)
                for i in range(1, 5): col = f'is_q{i}'; df[col] = (df['quarter_num'] == i).fillna(0).astype(int); current_features.append(col); self.categorical_features.append(col)
                current_features.append('quarter_num')
            except Exception as e: print(f"Warning: Could not extract quarter number: {e}")
        else:
            print("Info: 'Quarter' column not loaded by DataLoader. Skipping quarter features.")
            for col in ['quarter_num'] + [f'is_q{i}' for i in range(1,5)]: df[col] = np.nan; current_features.append(col)

        # --- Sector/Industry Features (Conditional) ---
        if 'Sector' in df.columns:
             df['Sector'] = df['Sector'].fillna('Unknown').astype(str)
             if fit_categorical:
                 self.sector_dummies_cols = pd.get_dummies(df['Sector'], prefix='sector', drop_first=True).columns.tolist(); print(f"Learned {len(self.sector_dummies_cols)} sector dummies.")
             if self.sector_dummies_cols:
                 sector_dummies_df = pd.get_dummies(df['Sector'], prefix='sector', drop_first=True)
                 for col in self.sector_dummies_cols: # Add missing/ensure order
                     if col not in sector_dummies_df: sector_dummies_df[col] = 0
                 df = pd.concat([df, sector_dummies_df[self.sector_dummies_cols]], axis=1); current_features.extend(self.sector_dummies_cols); self.categorical_features.extend(self.sector_dummies_cols)
             else: print("Info: Sector dummies not fitted or available. Skipping feature creation.") # Skip adding if not fitted
        else: print("Info: 'Sector' column not loaded by DataLoader. Skipping sector features.")

        if 'Industry' in df.columns:
             df['Industry'] = df['Industry'].fillna('Unknown').astype(str)
             if fit_categorical:
                 top_n = 20; self.top_industries = df['Industry'].value_counts().head(top_n).index.tolist()
                 df['Industry_Top'] = df['Industry'].apply(lambda x: x if x in self.top_industries else 'Other_Industry')
                 self.industry_dummies_cols = pd.get_dummies(df['Industry_Top'], prefix='industry', drop_first=True).columns.tolist(); print(f"Learned {len(self.industry_dummies_cols)} industry dummies.")
             if self.industry_dummies_cols:
                  df['Industry_Top'] = df['Industry'].apply(lambda x: x if x in self.top_industries else 'Other_Industry')
                  industry_dummies_df = pd.get_dummies(df['Industry_Top'], prefix='industry', drop_first=True)
                  for col in self.industry_dummies_cols: # Add missing/ensure order
                      if col not in industry_dummies_df: industry_dummies_df[col] = 0
                  df = pd.concat([df, industry_dummies_df[self.industry_dummies_cols]], axis=1); current_features.extend(self.industry_dummies_cols); self.categorical_features.extend(self.industry_dummies_cols)
             else: print("Info: Industry dummies not fitted or available. Skipping feature creation.") # Skip adding if not fitted
        else: print("Info: 'Industry' column not loaded by DataLoader. Skipping industry features.")


        self.feature_names = sorted(list(set(current_features)))
        print(f"Calculated/checked {len(self.feature_names)} raw Earnings features.")
        return df

    def get_features_target(self, df, fit_imputer=False):
        """Extract feature matrix X and target vector y, handling missing values. (Unchanged)"""
        print("Extracting Earnings features (X) and target (y)...")
        df = df.copy();
        if not self.feature_names: raise RuntimeError("Run calculate_features first.")
        available_features = [col for col in self.feature_names if col in df.columns]
        if not available_features: raise ValueError("No calculated features found in DataFrame.")
        df_with_target = df.dropna(subset=['future_ret'])
        if len(df_with_target) == 0: print("Warning: No data remains after filtering target."); return pd.DataFrame(columns=available_features), pd.Series(dtype=float)
        numeric_features = [f for f in available_features if f not in self.categorical_features]
        categorical_df = df_with_target[[f for f in available_features if f in self.categorical_features]].copy()
        X_numeric = df_with_target[numeric_features].copy(); y = df_with_target['future_ret'].copy()
        print(f"Original numeric X: {X_numeric.shape}. Categorical X: {categorical_df.shape}. Non-null y: {len(y)}")
        if fit_imputer: print("Fitting imputer..."); X_numeric_imputed = self.imputer.fit_transform(X_numeric)
        else:
            if not hasattr(self.imputer, 'statistics_'): raise RuntimeError("Imputer not fitted."); print("Transforming with imputer...")
            X_numeric_imputed = self.imputer.transform(X_numeric)
        X_numeric = pd.DataFrame(X_numeric_imputed, columns=numeric_features, index=X_numeric.index)
        X = pd.concat([X_numeric, categorical_df], axis=1); X = X[available_features] # Combine and reorder
        if X.isna().any().any(): warnings.warn("NaNs remain AFTER imputation!")
        else: print("No NaNs remaining after imputation.")
        self.final_feature_names = X.columns.tolist(); print(f"Final X shape: {X.shape}. Using {len(self.final_feature_names)} features.")
        return X, y

class SurpriseClassificationModel:
    """
    Model for classifying earnings surprises and predicting returns based on surprise.
    """
    def __init__(self, xgb_cls_params=None, xgb_reg_params=None):
        if xgb_cls_params is None: self.xgb_cls_params = {'n_estimators': 100, 'max_depth': 4, 'learning_rate': 0.05,'objective': 'binary:logistic', 'subsample': 0.8, 'colsample_bytree': 0.8,'random_state': 42, 'n_jobs': -1, 'eval_metric': 'logloss'}
        else: self.xgb_cls_params = xgb_cls_params
        if xgb_reg_params is None: self.xgb_reg_params = {'n_estimators': 100, 'max_depth': 5, 'learning_rate': 0.1,'objective': 'reg:squarederror', 'subsample': 0.8, 'colsample_bytree': 0.8,'random_state': 42, 'n_jobs': -1, 'eval_metric': 'rmse'}
        else: self.xgb_reg_params = xgb_reg_params
        self.surprise_pos_model = xgb.XGBClassifier(**self.xgb_cls_params)
        self.surprise_neg_model = xgb.XGBClassifier(**self.xgb_cls_params)
        self.return_model = xgb.XGBRegressor(**self.xgb_reg_params)
        self.feature_names_in_ = None
        self.return_feature_names_in_ = None

    def fit(self, X, y, surprise_data):
        """Fit surprise classifiers and return regressor."""
        print("Fitting SurpriseClassificationModel...")
        if not isinstance(X, pd.DataFrame): X = pd.DataFrame(X)
        self.feature_names_in_ = X.columns.tolist()
        if not all(X.index.equals(idx) for idx in [y.index, surprise_data.index]): raise ValueError("Indices must match.")

        pos_surprise = (surprise_data > 0).astype(int); neg_surprise = (surprise_data < 0).astype(int)
        print("  Fitting positive/negative surprise classifiers..."); self.surprise_pos_model.fit(X, pos_surprise); self.surprise_neg_model.fit(X, neg_surprise)
        X_with_surprise = X.copy()
        X_with_surprise['surprise_value_actual'] = surprise_data.fillna(0); X_with_surprise['pos_surprise_actual'] = pos_surprise
        X_with_surprise['neg_surprise_actual'] = neg_surprise; X_with_surprise['surprise_magnitude_actual'] = surprise_data.abs().fillna(0)
        self.return_feature_names_in_ = X_with_surprise.columns.tolist()
        print("  Fitting return regressor..."); self.return_model.fit(X_with_surprise, y)
        print("SurpriseClassificationModel fitting complete.")
        return self

    def predict(self, X):
        """Predict surprise probabilities/classes and returns."""
        if self.feature_names_in_ is None: raise RuntimeError("Model not fitted.")
        if not isinstance(X, pd.DataFrame): X = pd.DataFrame(X, columns=self.feature_names_in_)
        else: X = X[self.feature_names_in_]
        pos_prob = self.surprise_pos_model.predict_proba(X)[:, 1]; neg_prob = self.surprise_neg_model.predict_proba(X)[:, 1]
        pos_pred = (pos_prob > 0.5).astype(int); neg_pred = (neg_prob > 0.5).astype(int)
        X_for_ret = X.copy()
        X_for_ret['surprise_value_actual'] = pos_prob - neg_prob # Proxy value
        X_for_ret['pos_surprise_actual'] = pos_pred; X_for_ret['neg_surprise_actual'] = neg_pred # Use predicted class
        X_for_ret['surprise_magnitude_actual'] = np.abs(pos_prob - neg_prob) # Proxy magnitude
        X_for_ret = X_for_ret[self.return_feature_names_in_] # Ensure cols match training
        return_pred = self.return_model.predict(X_for_ret)
        return {'pos_surprise_prob': pos_prob, 'neg_surprise_prob': neg_prob, 'pos_surprise_pred': pos_pred, 'neg_surprise_pred': neg_pred, 'return_pred': return_pred}

    def evaluate(self, X, y, surprise_data):
        """Evaluate surprise classification and return prediction."""
        if not all(X.index.equals(idx) for idx in [y.index, surprise_data.index]): raise ValueError("Indices must match.")
        preds = self.predict(X)
        pos_actual = (surprise_data > 0).astype(int); neg_actual = (surprise_data < 0).astype(int)
        pos_acc = accuracy_score(pos_actual, preds['pos_surprise_pred']); neg_acc = accuracy_score(neg_actual, preds['neg_surprise_pred'])
        print("\n--- Surprise Classification Report (Positive) ---"); print(classification_report(pos_actual, preds['pos_surprise_pred'], zero_division=0))
        print("\n--- Surprise Classification Report (Negative) ---"); print(classification_report(neg_actual, preds['neg_surprise_pred'], zero_division=0))
        valid_mask = np.isfinite(y) & np.isfinite(preds['return_pred']); y_valid = y[valid_mask]; y_pred_valid = preds['return_pred'][valid_mask]
        if len(y_valid) == 0: ret_mse, ret_r2, ret_dir_acc = np.nan, np.nan, np.nan
        else: ret_mse = mean_squared_error(y_valid, y_pred_valid); ret_r2 = r2_score(y_valid, y_pred_valid); ret_dir_acc = np.mean(np.sign(y_pred_valid) == np.sign(y_valid))
        print("\n--- Return Prediction Evaluation ---")
        print(f"  MSE={ret_mse:.6f}, RMSE={np.sqrt(ret_mse):.6f}, R2={ret_r2:.4f}, DirAcc={ret_dir_acc:.4f}")
        return {'pos_surprise_accuracy': pos_acc, 'neg_surprise_accuracy': neg_acc, 'return_mse': ret_mse, 'return_rmse': np.sqrt(ret_mse), 'return_r2': ret_r2, 'return_direction_accuracy': ret_dir_acc}


class EarningsDriftModel:
    """Model post-earnings announcement drift (PEAD)."""
    def __init__(self, time_horizons=[1, 3, 5, 10, 20], model_params=None):
        self.time_horizons = sorted(time_horizons)
        if model_params is None: self.model_params = {'n_estimators': 100, 'max_depth': 4, 'learning_rate': 0.05,'objective': 'reg:squarederror', 'subsample': 0.8, 'colsample_bytree': 0.8,'random_state': 42, 'n_jobs': -1, 'eval_metric': 'rmse'}
        else: self.model_params = model_params
        self.models = {h: xgb.XGBRegressor(**self.model_params) for h in time_horizons}
        self.feature_names_in_ = None
        self.imputers = {} # Store one imputer per horizon if needed

    def _prepare_data_for_horizon(self, data, horizon, return_col='ret'):
        """Prepare features (announcement day) and target (cumulative return)."""
        required = ['event_id', 'date', 'days_to_announcement', return_col]
        if not all(c in data.columns for c in required): raise ValueError(f"Missing required columns: {required}")
        data = data.sort_values(by=['event_id', 'date'])
        announce_day_data = data[data['days_to_announcement'] == 0].set_index('event_id')

        # Calculate cumulative returns T+1 to T+horizon
        data['next_day_ret_factor'] = data.groupby('event_id')[return_col].transform(lambda x: (1 + x.shift(-1)).fillna(1))
        # Rolling product includes the return from T to T+1, T+1 to T+2, ..., T+h-1 to T+h
        data['cum_ret_factor'] = data.groupby('event_id')['next_day_ret_factor'].transform(lambda x: x.rolling(window=horizon, min_periods=1).apply(np.prod, raw=True))

        target_data = data[data['days_to_announcement'] == 0].set_index('event_id')
        target_data['target_cum_ret'] = target_data['cum_ret_factor'] - 1
        aligned_data = announce_day_data.join(target_data[['target_cum_ret']], how='inner').dropna(subset=['target_cum_ret'])

        if self.feature_names_in_ is None: # Infer features if first time
             self.feature_names_in_ = [col for col in aligned_data.columns if col not in ['date', 'Announcement Date', 'ticker', 'days_to_announcement', 'is_announcement_date', 'Quarter', 'Time', 'Sector', 'Industry', 'Industry_Top', 'future_ret', 'next_day_ret_factor', 'cum_ret_factor', 'target_cum_ret']] # Exclude metadata/targets
        missing = [f for f in self.feature_names_in_ if f not in aligned_data.columns]
        if missing: raise ValueError(f"PEAD features missing: {missing}")

        X = aligned_data[self.feature_names_in_].copy()
        y = aligned_data['target_cum_ret'].copy()
        return X, y

    def fit(self, data, feature_cols=None):
        """Fit separate models for each horizon."""
        print("Fitting EarningsDriftModel..."); self.feature_names_in_ = feature_cols
        if not self.feature_names_in_: # Infer if needed
             _, _ = self._prepare_data_for_horizon(data, self.time_horizons[0]) # Call to set self.feature_names_in_
             print(f"  Inferred {len(self.feature_names_in_)} PEAD features.")
        if not self.feature_names_in_: raise ValueError("Could not determine PEAD features.")

        for horizon in self.time_horizons:
            print(f"  Training PEAD model for {horizon}-day horizon...")
            try:
                X, y = self._prepare_data_for_horizon(data, horizon)
                if X.isna().any().any():
                    print(f"    Imputing NaNs for horizon {horizon}...")
                    imputer = SimpleImputer(strategy='median'); X_imputed = imputer.fit_transform(X)
                    X = pd.DataFrame(X_imputed, columns=X.columns, index=X.index); self.imputers[horizon] = imputer # Store imputer
                if len(X) == 0: print(f"    No data for horizon {horizon}."); continue
                self.models[horizon].fit(X, y); print(f"    PEAD model {horizon}d fitted ({len(X)} samples).")
            except Exception as e: print(f"    Error training PEAD {horizon}d: {e}")
        print("EarningsDriftModel fitting complete.")
        return self

    def predict(self, data):
        """Generate PEAD predictions using features from announcement day."""
        if self.feature_names_in_ is None: raise RuntimeError("PEAD model not fitted.")
        announce_days = data[data['days_to_announcement'] == 0].copy()
        if announce_days.empty: return pd.DataFrame()
        missing = [f for f in self.feature_names_in_ if f not in announce_days.columns]
        if missing: raise ValueError(f"PEAD prediction features missing: {missing}")
        X = announce_days[self.feature_names_in_].copy()

        # Impute using stored imputers or a new one if needed
        if X.isna().any().any():
             print("Imputing NaNs in PEAD prediction features...")
             # Simple approach: use a single imputer based on combined training data or median
             temp_imputer = SimpleImputer(strategy='median'); X_imputed = temp_imputer.fit_transform(X)
             X = pd.DataFrame(X_imputed, columns=X.columns, index=X.index)

        for horizon in self.time_horizons:
             pred_col = f'pred_drift_h{horizon}'
             if horizon in self.models:
                 try: announce_days[pred_col] = self.models[horizon].predict(X)
                 except Exception as e: print(f"Error predicting PEAD {horizon}d: {e}"); announce_days[pred_col] = np.nan
             else: announce_days[pred_col] = np.nan
        return announce_days

    def evaluate(self, data):
        """Evaluate PEAD models on test data."""
        print("\n--- Evaluating EarningsDriftModel ---"); results = {}
        if self.feature_names_in_ is None: print("PEAD model not fitted."); return results
        for horizon in self.time_horizons:
            print(f"  Evaluating PEAD {horizon}-day horizon...")
            try:
                X_test, y_test = self._prepare_data_for_horizon(data, horizon)
                if X_test.isna().any().any(): # Impute test data
                    if horizon in self.imputers: X_test_imputed = self.imputers[horizon].transform(X_test)
                    else: temp_imputer = SimpleImputer(strategy='median'); X_test_imputed = temp_imputer.fit_transform(X_test) # Fallback imputation
                    X_test = pd.DataFrame(X_test_imputed, columns=X_test.columns, index=X_test.index)
                if horizon not in self.models or len(X_test) == 0: print("    Model/Data unavailable."); continue

                y_pred = self.models[horizon].predict(X_test)
                valid_mask = np.isfinite(y_test) & np.isfinite(y_pred); y_test_v, y_pred_v = y_test[valid_mask], y_pred[valid_mask]
                if len(y_test_v) > 0: mse = mean_squared_error(y_test_v, y_pred_v); r2 = r2_score(y_test_v, y_pred_v); dir_acc = np.mean(np.sign(y_pred_v) == np.sign(y_test_v))
                else: mse, r2, dir_acc = np.nan, np.nan, np.nan
                results[horizon] = {'MSE': mse, 'RMSE': np.sqrt(mse), 'R2': r2, 'Direction Accuracy': dir_acc, 'N': len(y_test_v)}
                print(f"    PEAD {horizon}d: MSE={mse:.6f}, R2={r2:.4f}, DirAcc={dir_acc:.4f}, N={len(y_test_v)}")
            except Exception as e: print(f"    Error evaluating PEAD {horizon}d: {e}"); results[horizon] = {'Error': str(e)}
        return results


class Analysis:
    def __init__(self, data_loader, feature_engineer):
        """Analysis class for Earnings data."""
        self.data_loader = data_loader
        self.feature_engineer = feature_engineer
        self.data = None; self.X_train, self.X_test, self.y_train, self.y_test = [None]*4
        self.train_indices, self.test_indices = None, None
        self.models = {} # Standard models (Ridge, XGBDecile)
        self.surprise_model = None; self.pead_model = None
        self.results = {}; self.surprise_results = {}; self.pead_results = {}

    def load_and_prepare_data(self):
        """Load and prepare data for earnings analysis."""
        print("--- Loading Earnings Data ---"); self.data = self.data_loader.load_data()
        print("\n--- Creating Target Variable (Earnings) ---"); self.data = self.feature_engineer.create_target(self.data)
        print("\n--- Calculating Features (Earnings) ---"); self.data = self.feature_engineer.calculate_features(self.data, fit_categorical=False) # Fit categoricals on train split later
        print("\n--- Earnings Data Preparation Complete ---"); return self.data

    def train_models(self, test_size=0.2, time_split_column='Announcement Date'):
        """Split data and train all relevant models for earnings analysis."""
        if self.data is None: raise RuntimeError("Run load_and_prepare_data() first.")
        if time_split_column not in self.data.columns: raise ValueError(f"Time split column '{time_split_column}' not found.")
        if 'event_id' not in self.data.columns: raise ValueError("'event_id' required.")

        print(f"\n--- Splitting Earnings Data (Train/Test based on {time_split_column}) ---")
        events = self.data[['event_id', time_split_column]].drop_duplicates().sort_values(time_split_column)
        split_index = int(len(events) * (1 - test_size));
        if split_index == 0 or split_index == len(events): raise ValueError("test_size results in empty train/test set.")
        split_date = events.iloc[split_index][time_split_column]; print(f"Splitting {len(events)} unique events. Train before {split_date}.")
        train_event_ids = set(events.iloc[:split_index]['event_id']); test_event_ids = set(events.iloc[split_index:]['event_id'])
        train_mask = self.data['event_id'].isin(train_event_ids); test_mask = self.data['event_id'].isin(test_event_ids)
        self.train_indices = self.data.index[train_mask]; self.test_indices = self.data.index[test_mask]
        print(f"Train rows: {len(self.train_indices)}, Test rows: {len(self.test_indices)}.")

        print("\nFitting FeatureEngineer components (Categorical/Imputer) on Training Data...")
        train_data = self.feature_engineer.calculate_features(self.data.loc[self.train_indices].copy(), fit_categorical=True)
        self.X_train, self.y_train = self.feature_engineer.get_features_target(train_data, fit_imputer=True)

        print("\nApplying FeatureEngineer components to Test Data...")
        test_data = self.feature_engineer.calculate_features(self.data.loc[self.test_indices].copy(), fit_categorical=False)
        self.X_test, self.y_test = self.feature_engineer.get_features_target(test_data, fit_imputer=False)

        print(f"\nTrain shapes: X={self.X_train.shape}, y={self.y_train.shape}")
        print(f"Test shapes: X={self.X_test.shape}, y={self.y_test.shape}")
        if len(self.X_train) == 0 or len(self.X_test) == 0: raise ValueError("Train or test set empty.")

        print("\n--- Training Standard Models (Earnings) ---")
        # 1. TimeSeriesRidge
        try:
             print("Training TimeSeriesRidge..."); ts_ridge = TimeSeriesRidge(alpha=1.0, lambda2=0.1, feature_order=self.feature_engineer.final_feature_names)
             ts_ridge.fit(self.X_train, self.y_train); self.models['TimeSeriesRidge'] = ts_ridge; print("TimeSeriesRidge complete.")
        except Exception as e: print(f"Error TimeSeriesRidge: {e}")
        # 2. XGBoostDecile
        try:
             print("\nTraining XGBoostDecile..."); xgb_params = {'n_estimators': 150, 'max_depth': 5, 'learning_rate': 0.05, 'objective': 'reg:squarederror', 'subsample': 0.8, 'colsample_bytree': 0.8, 'random_state': 42, 'n_jobs': -1, 'early_stopping_rounds': 10}
             xgb_decile = XGBoostDecileModel(weight=0.7, momentum_feature='momentum_5', n_deciles=10, alpha=1.0, lambda_smooth=0.1, xgb_params=xgb_params, ts_ridge_feature_order=self.feature_engineer.final_feature_names)
             if 'momentum_5' not in self.X_train.columns: print("Warning: 'momentum_5' not found for XGBoostDecile.")
             xgb_decile.fit(self.X_train, self.y_train); self.models['XGBoostDecile'] = xgb_decile; print("XGBoostDecile complete.")
        except Exception as e: print(f"Error XGBoostDecile: {e}")

        # --- Train Surprise Classification Model ---
        surprise_col_name = 'surprise_val' # Standardized name from feature engineer
        if surprise_col_name in train_data.columns:
             print("\n--- Training Surprise Classification Model ---")
             try:
                 surprise_train = train_data.loc[self.X_train.index, surprise_col_name]
                 self.surprise_model = SurpriseClassificationModel()
                 self.surprise_model.fit(self.X_train, self.y_train, surprise_train)
                 print("SurpriseClassificationModel training complete.")
             except Exception as e: print(f"Error SurpriseClassificationModel: {e}"); self.surprise_model = None
        else: print("\nSurprise column not found. Skipping surprise model.")

        # --- Train Earnings Drift (PEAD) Model ---
        print("\n--- Training Earnings Drift (PEAD) Model ---")
        try:
             self.pead_model = EarningsDriftModel(time_horizons=[1, 3, 5, 10, 20])
             pead_feature_cols = self.feature_engineer.final_feature_names # Use same features for consistency
             self.pead_model.fit(train_data, feature_cols=pead_feature_cols) # Fit on processed train data
             print("EarningsDriftModel training complete.")
        except Exception as e: print(f"Error EarningsDriftModel: {e}"); self.pead_model = None

        print("\n--- All Earnings Model Training Complete ---"); return self.models

    def evaluate_models(self):
        """Evaluate all trained models on the test set."""
        print("\n--- Evaluating Earnings Models ---")
        if self.X_test is None or self.y_test is None or len(self.X_test)==0: print("Test data unavailable/empty."); return {}

        # Evaluate Standard Models
        print("\n--- Standard Model Evaluation ---"); self.results = {}
        for name, model in self.models.items():
            print(f"Evaluating {name}..."); 
            try:
                y_pred = model.predict(self.X_test); valid_mask = np.isfinite(self.y_test) & np.isfinite(y_pred)
                y_test_v, y_pred_v = self.y_test[valid_mask], y_pred[valid_mask]
                if len(y_test_v) > 0: mse = mean_squared_error(y_test_v, y_pred_v); r2 = r2_score(y_test_v, y_pred_v)
                else: mse, r2 = np.nan, np.nan
                self.results[name] = {'MSE': mse, 'RMSE': np.sqrt(mse), 'R2': r2, 'N': len(y_test_v)}
                print(f"  {name}: MSE={mse:.6f}, RMSE={np.sqrt(mse):.6f}, R2={r2:.4f}, N={len(y_test_v)}")
            except Exception as e: print(f"  Error evaluating {name}: {e}"); self.results[name] = {'Error': str(e)}

        # Evaluate Surprise Model
        surprise_col_name = 'surprise_val'
        if self.surprise_model:
             print("\n--- Surprise Model Evaluation ---")
             test_data_surprise = self.data.loc[self.X_test.index] # Original test data rows
             if surprise_col_name in test_data_surprise.columns:
                 surprise_test = test_data_surprise[surprise_col_name]
                 try: self.surprise_results = self.surprise_model.evaluate(self.X_test, self.y_test, surprise_test)
                 except Exception as e: print(f"  Error evaluating Surprise Model: {e}"); self.surprise_results = {'Error': str(e)}
             else: print("  Surprise column not found in test data. Cannot evaluate.")
        else: print("\nSurprise Model not trained. Skipping evaluation.")

        # Evaluate PEAD Model
        if self.pead_model:
             print("\n--- PEAD Model Evaluation ---"); 
             try:
                 test_data_full = self.data.loc[self.test_indices].copy() # Use original indices
                 self.pead_results = self.pead_model.evaluate(test_data_full)
             except Exception as e: print(f"  Error evaluating PEAD Model: {e}"); self.pead_results = {'Error': str(e)}
        else: print("\nPEAD Model not trained. Skipping evaluation.")

        print("\n--- Earnings Evaluation Complete ---")
        return {"standard": self.results, "surprise": self.surprise_results, "pead": self.pead_results}

    def plot_feature_importance(self, model_name='TimeSeriesRidge', top_n=20):
        """Plot feature importance for standard earnings models."""
        # This is identical to the FDA version, just change title context
        print(f"\n--- Plotting Earnings Feature Importance for {model_name} ---")
        # ... (rest of the plotting code is identical to fda_processor.py's version) ...
        # ... just ensure the title reflects "Earnings" analysis ...
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
                     if xgb_feat_names: imp_dict = dict(zip(xgb_feat_names, model.xgb_model.feature_importances_)); importances = np.array([imp_dict.get(name, 0) for name in feature_names])
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
        plt.title(f'Top {top_n} Features by Importance ({model_name} - Earnings)') # Modified Title
        plt.xlabel('Importance Score'); plt.ylabel('Feature'); plt.tight_layout(); plt.show()
        return feat_imp_df


    def analyze_earnings_surprise(self):
        """Analyze the impact of earnings surprises on actual returns."""
        # This method remains the same as in the original quarterlyearnings.py
        print("\n--- Analyzing Earnings Surprise Impact ---")
        surprise_col = 'surprise_val' # Standardized name
        if self.data is None or surprise_col not in self.data.columns: print(f"No surprise data ('{surprise_col}') available."); return None
        analysis_data = self.data[(self.data['days_to_announcement'] >= -2) & (self.data['days_to_announcement'] <= 5)].copy()
        bins = [-np.inf, -0.01, 0.01, np.inf]; labels = ['Negative Surprise', 'Near Zero Surprise', 'Positive Surprise']
        analysis_data['Surprise Group'] = pd.cut(analysis_data[surprise_col], bins=bins, labels=labels, right=False)
        avg_returns = analysis_data.groupby(['Surprise Group', 'days_to_announcement'])['ret'].mean().unstack(level=0)
        avg_cum_returns = (1 + avg_returns.fillna(0)).cumprod(axis=0) - 1
        fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
        avg_returns.plot(kind='bar', ax=axes[0], width=0.8); axes[0].set_title('Average Daily Returns by Surprise'); axes[0].set_ylabel('Avg Daily Return'); axes[0].axhline(0, c='grey', ls='--', lw=0.8); axes[0].legend(title='Group'); axes[0].grid(axis='y', ls=':', alpha=0.6)
        avg_cum_returns.plot(kind='line', marker='o', ax=axes[1]); axes[1].set_title('Average Cumulative Returns by Surprise'); axes[1].set_ylabel('Avg Cum Return'); axes[1].set_xlabel('Days Rel. to Announce'); axes[1].axhline(0, c='grey', ls='--', lw=0.8); axes[1].axvline(0, c='red', ls=':', lw=1, label='Announce Day'); axes[1].legend(title='Group'); axes[1].grid(True, ls=':', alpha=0.6)
        plt.tight_layout(); plt.show()
        return avg_cum_returns

    def plot_predictions_for_event(self, event_id, model_name='TimeSeriesRidge'):
        """Plot actual daily returns and model's predicted future returns for a specific event."""
        # This method remains the same as in the original quarterlyearnings.py
        print(f"\n--- Plotting Earnings Predictions for Event: {event_id} ({model_name}) ---")
        if model_name not in self.models: print(f"Error: Model '{model_name}' not found."); return None
        if self.data is None: print("Error: Data not loaded."); return None
        model = self.models[model_name]; event_data_full = self.data[self.data['event_id'] == event_id].sort_values('date').copy()
        if event_data_full.empty: print(f"Error: No data for event_id '{event_id}'."); return None
        ticker = event_data_full['ticker'].iloc[0]; announcement_date = event_data_full['Announcement Date'].iloc[0]
        if not hasattr(self.feature_engineer.imputer, 'statistics_'): print("Error: Imputer not fitted."); return None
        X_event, y_event_actual = self.feature_engineer.get_features_target(event_data_full, fit_imputer=False)
        if X_event.empty: print(f"Warn: No valid features for event {event_id}."); return None
        try: y_pred_event = model.predict(X_event)
        except Exception as e: print(f"Error predicting event {event_id}: {e}"); return None
        event_data_pred = event_data_full.loc[X_event.index].copy(); event_data_pred['predicted_future_ret'] = y_pred_event
        plt.figure(figsize=(14, 6)); plt.plot(event_data_full['date'], event_data_full['ret'], '.-', label='Actual Daily Return', alpha=0.6, color='blue')
        plt.scatter(event_data_pred['date'], event_data_pred['predicted_future_ret'], color='red', label=f'Predicted {self.feature_engineer.prediction_window}-day Future Ret', s=15, zorder=5, alpha=0.8)
        plt.axvline(x=announcement_date, color='g', linestyle='--', label='Announcement Date'); plt.title(f"{ticker} ({event_id}) - Daily Returns & Predicted Future Returns ({model_name} - Earnings)")
        plt.ylabel("Return"); plt.xlabel("Date"); plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout(); plt.show()
        return event_data_pred

    def find_sample_event_ids(self, n=5):
        """Find sample Earnings event identifiers."""
        # This method remains the same as in the original quarterlyearnings.py
        if self.data is None or 'event_id' not in self.data.columns: return []
        unique_events = self.data['event_id'].unique()
        return list(unique_events[:min(n, len(unique_events))])

    def plot_pead_predictions(self, n_events=3):
        """Plot PEAD predictions vs actual cumulative returns for sample events."""
        # This method remains the same as in the original quarterlyearnings.py
        print("\n--- Plotting PEAD Predictions vs Actual ---")
        if self.pead_model is None: print("PEAD Model not trained."); return
        if self.data is None: print("Data not available."); return
        sample_event_ids = self.find_sample_event_ids(n=n_events)
        if not sample_event_ids: print("No sample events found."); return
        test_data_full = self.data.loc[self.test_indices].copy() # Use test set data
        pead_predictions_df = self.pead_model.predict(test_data_full)
        if pead_predictions_df.empty: print("No PEAD predictions generated."); return
        for event_id in sample_event_ids:
            event_preds = pead_predictions_df[pead_predictions_df['event_id'] == event_id]
            if event_preds.empty: continue
            event_actual_data = test_data_full[test_data_full['event_id'] == event_id].sort_values('date')
            if event_actual_data.empty: continue
            ticker = event_actual_data['ticker'].iloc[0]; announce_date = event_actual_data['Announcement Date'].iloc[0]
            plt.figure(figsize=(12, 6))
            post_announce_actual = event_actual_data[event_actual_data['date'] >= announce_date].copy()
            post_announce_actual['actual_cum_ret'] = (1 + post_announce_actual['ret'].fillna(0)).cumprod() - 1
            plt.plot(post_announce_actual['date'], post_announce_actual['actual_cum_ret'], marker='.', linestyle='-', label='Actual Cum. Return', color='blue')
            pred_cols = sorted([col for col in event_preds.columns if col.startswith('pred_drift_h')], key=lambda x: int(x[len('pred_drift_h'):]))
            horizons = [int(col[len('pred_drift_h'):]) for col in pred_cols]
            pred_values = event_preds[pred_cols].iloc[0].values
            plot_dates = [announce_date + pd.Timedelta(days=h) for h in horizons]
            plt.scatter(plot_dates, pred_values, color='red', marker='x', s=50, label='Predicted Cum. Return (PEAD)', zorder=5)
            plt.title(f"PEAD Analysis: {ticker} ({event_id})"); plt.xlabel("Date"); plt.ylabel("Cumulative Return")
            plt.axvline(announce_date, color='grey', linestyle='--', label='Announcement Date'); plt.legend(); plt.grid(True, alpha=0.4); plt.tight_layout(); plt.show()

    def analyze_sharpe_ratio_dynamics(self, results_dir, file_prefix="earnings", risk_free_rate=0.0, window=20, min_periods=10, pre_days=30, post_days=30):
        """Calculates, plots, and saves rolling Sharpe Ratio dynamics."""
        print(f"\n--- Analyzing Rolling Sharpe Ratio (Window={window}d) ---")
        # Identical logic to FDA version, just different title/prefix/day_column
        if self.data is None or 'ret' not in self.data.columns or 'days_to_announcement' not in self.data.columns:
            print("Error: Data/required columns missing."); return None

        df = self.data.copy().sort_values(by=['event_id', 'date'])
        df['rolling_mean_ret'] = df.groupby('event_id')['ret'].transform(lambda x: x.rolling(window=window, min_periods=min_periods).mean())
        df['rolling_std_ret'] = df.groupby('event_id')['ret'].transform(lambda x: x.rolling(window=window, min_periods=min_periods).std())
        epsilon = 1e-8; daily_risk_free = risk_free_rate / 252
        df['daily_sharpe'] = (df['rolling_mean_ret'] - daily_risk_free) / (df['rolling_std_ret'] + epsilon)
        df['annualized_sharpe'] = df['daily_sharpe'] * np.sqrt(252)
        aligned_sharpe = df.groupby('days_to_announcement')['annualized_sharpe'].mean()
        aligned_sharpe_plot = aligned_sharpe.loc[-pre_days:post_days]

        # Plotting
        fig, ax = plt.subplots(figsize=(12, 6))
        aligned_sharpe_plot.plot(kind='line', marker='.', linestyle='-', ax=ax)
        ax.axvline(0, color='red', linestyle='--', linewidth=1, label='Announcement Day')
        ax.axhline(0, color='grey', linestyle=':', linewidth=1)
        ax.set_title(f'Average Annualized Rolling Sharpe Ratio Around Earnings Announcement (Window={window}d)') # Changed Title
        ax.set_xlabel('Days Relative to Announcement'); ax.set_ylabel('Average Annualized Sharpe Ratio')
        ax.legend(); ax.grid(True, alpha=0.5)
        plt.tight_layout()

        # Save plot
        plot_filename = os.path.join(results_dir, f"{file_prefix}_sharpe_ratio_rolling_{window}d.png")
        try:
            plt.savefig(plot_filename); print(f"Saved Sharpe plot to: {plot_filename}")
        except Exception as e: print(f"Error saving Sharpe plot: {e}")
        plt.close(fig)

        # Save data
        csv_filename = os.path.jin(results_dir, f"{file_prefix}_sharpe_ratio_rolling_{window}d_data.csv")
        try:
            aligned_sharpe_plot.to_csv(csv_filename, header=True)
            print(f"Saved Sharpe data to: {csv_filename}")
        except Exception as e: print(f"Error saving Sharpe data: {e}")

        print(f"Average Sharpe Ratio ({window}d rolling) in plot window: {aligned_sharpe_plot.mean():.4f}")
        return aligned_sharpe

    def analyze_volatility_spikes(self, results_dir, file_prefix="earnings", window=5, min_periods=3, pre_days=30, post_days=30, baseline_window=(-60, -11), event_window=(-2, 2)):
        """Calculates, plots, and saves rolling volatility dynamics and compares event vs baseline."""
        print(f"\n--- Analyzing Rolling Volatility (Window={window}d) ---")
        # Identical logic to FDA version, just different title/prefix/day_column
        if self.data is None or 'ret' not in self.data.columns or 'days_to_announcement' not in self.data.columns:
            print("Error: Data/required columns missing."); return None

        df = self.data.copy().sort_values(by=['event_id', 'date'])
        df['rolling_vol'] = df.groupby('event_id')['ret'].transform(lambda x: x.rolling(window=window, min_periods=min_periods).std())
        df['annualized_vol'] = df['rolling_vol'] * np.sqrt(252) * 100 # In percent
        aligned_vol = df.groupby('days_to_announcement')['annualized_vol'].mean()
        aligned_vol_plot = aligned_vol.loc[-pre_days:post_days]

        # --- Plotting & Saving Rolling Volatility ---
        fig1, ax1 = plt.subplots(figsize=(12, 6))
        aligned_vol_plot.plot(kind='line', marker='.', linestyle='-', ax=ax1)
        ax1.axvline(0, color='red', linestyle='--', linewidth=1, label='Announcement Day')
        ax1.set_title(f'Average Annualized Rolling Volatility Around Earnings Announcement (Window={window}d)') # Changed Title
        ax1.set_xlabel('Days Relative to Announcement'); ax1.set_ylabel('Avg. Annualized Volatility (%)')
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
            baseline_data = group[(group['days_to_announcement'] >= baseline_window[0]) & (group['days_to_announcement'] <= baseline_window[1])]
            event_data = group[(group['days_to_announcement'] >= event_window[0]) & (group['days_to_announcement'] <= event_window[1])]
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


# Example Usage:
if __name__ == "__main__":
    print("--- Running Earnings Processor Example ---")
    # Define paths (replace with your actual paths)
    EARNINGS_FILE = 'path/to/your/earnings_announcements.csv' # Example path
    STOCK_FILES = [
        'path/to/stock_data_part1.csv', # Example path
        'path/to/stock_data_part2.csv'
    ]

    # --- Basic File Existence Check ---
    import os
    if not os.path.exists(EARNINGS_FILE):
        print(f"Error: Earnings file not found at {EARNINGS_FILE}")
        print("Please update the EARNINGS_FILE path.")
    elif not all(os.path.exists(f) for f in STOCK_FILES):
        missing_stock = [f for f in STOCK_FILES if not os.path.exists(f)]
        print(f"Error: Stock file(s) not found: {missing_stock}")
        print("Please update the STOCK_FILES list.")
    else:
        # Initialize components
        data_loader = DataLoader(earnings_path=EARNINGS_FILE, stock_paths=STOCK_FILES, window_days=30)
        feature_engineer = FeatureEngineer(prediction_window=3) # Predict 3 days ahead
        analyzer = Analysis(data_loader, feature_engineer)

        # Run analysis pipeline
        try:
            analyzer.load_and_prepare_data()
            analyzer.train_models(test_size=0.2)
            results_dict = analyzer.evaluate_models()

            print("\n--- Earnings Evaluation Summary ---")
            print("\nStandard Models:")
            print(pd.DataFrame(results_dict.get('standard', {})).T)
            print("\nSurprise Model:")
            print(pd.Series(results_dict.get('surprise', {})))
            print("\nPEAD Model:")
            print(pd.DataFrame(results_dict.get('pead', {})).T)

            # Plot feature importance for a standard model
            if 'XGBoostDecile' in analyzer.models:
                 analyzer.plot_feature_importance(model_name='XGBoostDecile', top_n=15)
            elif 'TimeSeriesRidge' in analyzer.models:
                  analyzer.plot_feature_importance(model_name='TimeSeriesRidge', top_n=15)


            # Analyze surprise impact
            analyzer.analyze_earnings_surprise()

            # Plot predictions for a sample event (standard model)
            sample_ids = analyzer.find_sample_event_ids(n=1)
            if sample_ids:
                 model_to_plot = 'XGBoostDecile' if 'XGBoostDecile' in analyzer.models else 'TimeSeriesRidge'
                 if model_to_plot in analyzer.models:
                     print(f"\nPlotting standard predictions for event: {sample_ids[0]}")
                     analyzer.plot_predictions_for_event(event_id=sample_ids[0], model_name=model_to_plot)

            # Plot PEAD predictions
            analyzer.plot_pead_predictions(n_events=2)

        except ValueError as ve: print(f"\nValueError during Earnings analysis: {ve}")
        except RuntimeError as re: print(f"\nRuntimeError during Earnings analysis: {re}")
        except FileNotFoundError as fnf: print(f"\nFileNotFoundError: {fnf}")
        except Exception as e:
            print(f"\nAn unexpected error occurred during Earnings analysis: {e}")
            import traceback
            traceback.print_exc()
