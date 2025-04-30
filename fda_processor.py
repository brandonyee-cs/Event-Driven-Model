import polars as pl
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
import warnings
import os
from typing import List, Optional, Tuple, Dict, Any # For type hinting


# Import shared models (assuming models.py is accessible)
try:
    # models.py now handles Polars input but uses NumPy internally
    from models import TimeSeriesRidge, XGBoostDecileModel
except ImportError:
    print("Error: Could not import models from 'models'.")
    print("Ensure models.py is in the same directory or Python path.")
    import sys
    sys.exit(1)


# Suppress specific warnings if needed
warnings.filterwarnings('ignore', message='X does not have valid feature names, but SimpleImputer was fitted with feature names')
warnings.simplefilter(action='ignore', category=FutureWarning)


class DataLoader:
    def __init__(self, fda_path: str, stock_paths: List[str], window_days: int = 100):
        """
        Initialize DataLoader for FDA events using Polars.

        Parameters:
        fda_path (str): Path to the FDA approval event data CSV.
        stock_paths (list): List of paths to stock data PARQUET files.
        window_days (int): Number of days before/after event date.
        """
        self.fda_path = fda_path
        if isinstance(stock_paths, str): self.stock_paths = [stock_paths]
        elif isinstance(stock_paths, list): self.stock_paths = stock_paths
        else: raise TypeError("stock_paths must be a string or a list of Parquet file paths.")
        self.window_days = window_days

    def _load_single_stock_parquet(self, stock_path: str) -> Optional[pl.DataFrame]:
        """Load and process a single stock data PARQUET file using Polars."""
        try:
            print(f"  Reading Parquet file: {stock_path}")
            stock_data = pl.read_parquet(stock_path)
            print(f"  Read {len(stock_data)} rows from {stock_path}")

            # --- Column Name Standardization (Case-Insensitive) ---
            original_columns = stock_data.columns
            col_map_lower = {col.lower(): col for col in original_columns}

            standard_names = {
                'date': ['date', 'trade_date', 'trading_date', 'tradedate', 'dt'],
                'ticker': ['ticker', 'symbol', 'sym_root', 'tic'], # Removed permno as it's less common direct ticker
                'prc': ['prc', 'price', 'close', 'adj close', 'adj_prc'],
                'ret': ['ret', 'return', 'daily_ret'],
                'vol': ['vol', 'volume'],
                'openprc': ['openprc', 'open'],
                'askhi': ['askhi', 'high', 'askhigh'],
                'bidlo': ['bidlo', 'low', 'bidlow'],
                'shrout': ['shrout', 'shares_outstanding']
            }
            rename_dict = {}
            found_std_names = {}
            selected_cols = [] # Keep track of standardized columns found

            for std_name, variations in standard_names.items():
                found = False
                for var in variations:
                    if var in col_map_lower:
                        original_case_col = col_map_lower[var]
                        if original_case_col != std_name:
                            rename_dict[original_case_col] = std_name
                        found_std_names[std_name] = True
                        selected_cols.append(original_case_col) # Add original name to selection
                        found = True
                        break
                if not found:
                    found_std_names[std_name] = False

            # Select only the columns identified (using original names)
            stock_data = stock_data.select(selected_cols)
            # Apply renaming
            if rename_dict:
                stock_data = stock_data.rename(rename_dict)

            # --- Data Type and Existence Checks ---
            required_cols = ['date', 'ticker', 'prc', 'ret'] # Base requirements for FDA
            missing_cols = [col for col in required_cols if col not in stock_data.columns]
            if missing_cols:
                 raise ValueError(f"Missing required columns after standardization in {stock_path}: {missing_cols}")

            # Ensure date column is datetime
            if stock_data["date"].dtype != pl.Datetime:
                 print(f"  Warning: 'date' column in {stock_path} is {stock_data['date'].dtype}. Attempting conversion to Datetime.")
                 stock_data = stock_data.with_columns(
                     pl.col("date").str.to_datetime(strict=False, errors="null").alias("date")
                 )
            # Drop rows with null dates
            n_null_dates = stock_data.filter(pl.col("date").is_null()).height
            if n_null_dates > 0:
                print(f"  Warning: Found {n_null_dates} null dates in {stock_path}. Dropping these rows.")
                stock_data = stock_data.drop_nulls(subset=["date"])

            # Ensure ticker is string
            if stock_data["ticker"].dtype != pl.Utf8:
                stock_data = stock_data.with_columns(pl.col("ticker").cast(pl.Utf8))

            # Ensure numeric columns are numeric (Float64), handle inf
            numeric_cols_to_check = ['prc', 'ret', 'vol', 'openprc', 'askhi', 'bidlo', 'shrout']
            cast_expressions = []
            for col in numeric_cols_to_check:
                if col in stock_data.columns:
                    if stock_data[col].dtype not in [pl.Float32, pl.Float64]:
                         print(f"  Warning: Column '{col}' in {stock_path} is {stock_data[col].dtype}. Attempting conversion to Float64.")
                    cast_expressions.append(
                        pl.when(pl.col(col).cast(pl.Float64, strict=False).is_infinite())
                        .then(None) # Replace inf with null
                        .otherwise(pl.col(col).cast(pl.Float64, strict=False))
                        .alias(col)
                    )

            if cast_expressions:
                 stock_data = stock_data.with_columns(cast_expressions)

            # Select necessary columns based on standardized names found
            final_cols = list(standard_names.keys())
            cols_present = [col for col in final_cols if col in stock_data.columns]
            stock_data = stock_data.select(cols_present)

            print(f"  Successfully processed {stock_path}. Final shape: {stock_data.shape}")
            return stock_data

        except FileNotFoundError:
            print(f"Error: Stock Parquet file not found: {stock_path}")
            return None
        except Exception as e:
            print(f"Error processing Parquet stock file {stock_path}: {e}")
            import traceback
            traceback.print_exc()
            return None

    def load_data(self) -> Optional[pl.DataFrame]:
        """Load FDA approval CSV data and stock Parquet data using Polars, then merge them."""
        # --- Load FDA Event Data (CSV) ---
        try:
            print(f"Loading FDA event data from: {self.fda_path} (CSV)")
            # Define expected dtypes for faster parsing
            # Adjust 'Approval Date' format if known
            fda_data = pl.read_csv(self.fda_path) # Let Polars infer initially

            # Basic validation and standardization
            if 'Approval Date' not in fda_data.columns: raise ValueError("Missing 'Approval Date' column in FDA CSV.")
            # Use str.to_date or str.to_datetime based on format
            fda_data = fda_data.with_columns(
                 pl.col('Approval Date').str.to_datetime(strict=False, errors='null').alias('Approval Date')
                 # Add more formats if needed, e.g. pl.col('Approval Date').str.to_date('%Y-%m-%d', strict=False, errors='null')
            )
            initial_rows = fda_data.height
            fda_data = fda_data.drop_nulls(subset=['Approval Date'])
            if fda_data.height < initial_rows: print(f"  Dropped {initial_rows - fda_data.height} rows with invalid Approval Dates.")

            # Handle ticker column variations
            if 'ticker' not in fda_data.columns:
                if 'Ticker' in fda_data.columns:
                    # Extract ticker, handling potential prefixes like 'NASDAQ:AAPL'
                    fda_data = fda_data.with_columns(
                        pl.col('Ticker').cast(pl.Utf8)
                          .str.split(':') # Split by ':'
                          .list.last()     # Get the last element (ticker)
                          .str.strip_chars() # Remove leading/trailing whitespace
                          .alias('ticker')
                    )
                else: raise ValueError("Missing 'ticker' or 'Ticker' column in FDA CSV.")
            else:
                # Ensure ticker is Utf8 if it exists
                 fda_data = fda_data.with_columns(pl.col('ticker').cast(pl.Utf8))


            if 'Drug Name' not in fda_data.columns:
                print("  Warning: 'Drug Name' column not found in FDA data. Using 'N/A'.")
                fda_data = fda_data.with_columns(pl.lit("N/A").alias('Drug Name'))

            # Standardize ticker case and select/deduplicate
            approval_events = fda_data.with_columns(pl.col('ticker').str.to_uppercase()) \
                                       .select(['ticker', 'Approval Date', 'Drug Name']) \
                                       .unique(keep='first')

            print(f"Found {len(approval_events)} unique FDA approval events.")
            if approval_events.is_empty(): raise ValueError("No valid FDA events found after processing.")

        except FileNotFoundError:
             raise FileNotFoundError(f"FDA CSV file not found: {self.fda_path}")
        except Exception as e: raise ValueError(f"Error loading FDA data from {self.fda_path}: {e}")

        # --- Load Stock Data (Parquet) ---
        stock_data_list, failed_files = [], []
        print("\nLoading stock data from Parquet files...")
        for stock_path in self.stock_paths:
            processed_data = self._load_single_stock_parquet(stock_path)
            if processed_data is not None:
                stock_data_list.append(processed_data.with_columns(pl.col('ticker').str.to_uppercase()))
            else:
                failed_files.append(stock_path)

        if not stock_data_list: raise ValueError("No stock Parquet data loaded successfully.")
        if failed_files: print(f"\nWarning: Skipped {len(failed_files)} stock file(s): {failed_files}\n")

        # --- Combine and Merge ---
        print("Combining loaded stock data...")
        stock_data_combined = pl.concat(stock_data_list, how='vertical')
        print(f"Combined stock data rows before deduplication: {len(stock_data_combined)}")

        # Deduplicate based on date and ticker
        stock_data_combined = stock_data_combined.unique(subset=['date', 'ticker'], keep='first', maintain_order=False)
        print(f"Combined stock data rows after deduplication: {len(stock_data_combined)}")
        if stock_data_combined.is_empty(): raise ValueError("Stock data empty after deduplication.")

        print("\nMerging FDA events with stock data...")
        # Filter stock data for tickers present in FDA events
        tickers_with_approvals = approval_events.get_column('ticker').unique()
        stock_data_filtered = stock_data_combined.filter(pl.col('ticker').is_in(tickers_with_approvals))
        print(f"Filtered stock data for {tickers_with_approvals.len()} relevant tickers. Rows: {len(stock_data_filtered)}")
        if stock_data_filtered.is_empty(): raise ValueError("No stock data found for tickers listed in FDA events.")

        # Join FDA events with filtered stock data
        merged_data = stock_data_filtered.join(
            approval_events, on='ticker', how='inner'
        )

        # Calculate relative days and filter window
        event_window_data = merged_data.with_columns(
            (pl.col('date') - pl.col('Approval Date')).dt.total_days().cast(pl.Int32).alias('days_to_approval')
        ).filter(
            (pl.col('days_to_approval') >= -self.window_days) &
            (pl.col('days_to_approval') <= self.window_days)
        )

        if event_window_data.is_empty(): raise ValueError("No stock data found within the specified window for any FDA events.")

        # Add event identifier and approval flag
        event_window_data = event_window_data.with_columns([
            (pl.col('days_to_approval') == 0).cast(pl.Int8).alias('is_approval_date'),
            (pl.col("ticker") + "_" + pl.col("Approval Date").dt.strftime('%Y%m%d')).alias('event_id')
        ])

        # Select necessary columns
        final_cols = stock_data_filtered.columns + ['Approval Date', 'Drug Name', 'days_to_approval', 'is_approval_date', 'event_id']
        final_cols = sorted(list(set(final_cols)))
        final_cols = [c for c in final_cols if c in event_window_data.columns]
        event_window_data = event_window_data.select(final_cols)

        print(f"Created windows for {event_window_data['event_id'].n_unique()} FDA events.")
        combined_data = event_window_data.sort(['ticker', 'Approval Date', 'date'])

        print(f"Final FDA dataset shape: {combined_data.shape}")
        mem_usage_mb = combined_data.estimated_size("mb")
        print(f"Final DataFrame memory usage: {mem_usage_mb:.2f} MB")
        return combined_data


class FeatureEngineer:
    def __init__(self, prediction_window: int = 5):
        self.windows = [5, 10, 20]
        self.prediction_window = prediction_window
        self.imputer = SimpleImputer(strategy='median') # Operates on NumPy
        self.feature_names: List[str] = [] # Store raw feature names created
        self.final_feature_names: List[str] = [] # Store final names after imputation/selection
        self._imputer_fitted = False # Track if imputer has been fitted

    def create_target(self, df: pl.DataFrame, price_col: str = 'prc') -> pl.DataFrame:
        """Create target variable for FDA analysis using Polars."""
        print(f"Creating target 'future_ret' (window: {self.prediction_window} days)...")
        if 'event_id' not in df.columns: raise ValueError("'event_id' required.")
        if price_col not in df.columns: raise ValueError(f"Price column '{price_col}' not found.")

        df = df.sort(['event_id', 'date']) # Ensure sorted for shift

        df = df.with_columns(
            pl.col(price_col).shift(-self.prediction_window).over('event_id').alias('future_price')
        ).with_columns(
            # Calculate return, handle division by zero or zero price
            pl.when(pl.col(price_col).is_not_null() & (pl.col(price_col) != 0))
            .then((pl.col('future_price') / pl.col(price_col)) - 1)
            .otherwise(None)
            .alias('future_ret')
        ).drop('future_price')

        print(f"'future_ret' created. Non-null: {df.filter(pl.col('future_ret').is_not_null()).height}")
        return df

    def calculate_features(self, df: pl.DataFrame, price_col: str = 'prc', return_col: str = 'ret') -> pl.DataFrame:
        """Calculate features for FDA analysis using Polars."""
        print("Calculating FDA features (Polars)...")
        required = ['event_id', price_col, return_col, 'Approval Date', 'date', 'days_to_approval']
        missing = [col for col in required if col not in df.columns]
        if missing: raise ValueError(f"Missing required columns for feature calculation: {missing}")

        df = df.sort(['event_id', 'date']) # Ensure sorted for rolling/shift ops
        current_features = []
        feature_expressions = []

        # --- Price momentum ---
        for window in self.windows:
            col_name = f'momentum_{window}'
            shifted_price = pl.col(price_col).shift(window).over('event_id')
            feature_expressions.append(
                pl.when(shifted_price.is_not_null() & (shifted_price != 0))
                .then((pl.col(price_col) / shifted_price) - 1)
                .otherwise(None).alias(col_name)
            )
            current_features.append(col_name)

        # Need to add momentum cols first before calculating deltas
        df = df.with_columns(feature_expressions)
        feature_expressions = [] # Reset

        feature_expressions.extend([
            (pl.col('momentum_5') - pl.col('momentum_10')).alias('delta_momentum_5_10'),
            (pl.col('momentum_10') - pl.col('momentum_20')).alias('delta_momentum_10_20')
        ])
        current_features.extend(['delta_momentum_5_10', 'delta_momentum_10_20'])

        # --- Return volatility ---
        for window in self.windows:
            col_name = f'volatility_{window}'
            min_p = max(2, min(window, 5))
            # Polars rolling std over group
            feature_expressions.append(
                 pl.col(return_col).rolling_std(window_size=window, min_periods=min_p)
                   .over('event_id').alias(col_name)
            )
            current_features.append(col_name)

        # Need to add volatility cols first before calculating deltas
        df = df.with_columns(feature_expressions)
        feature_expressions = [] # Reset

        feature_expressions.extend([
             (pl.col('volatility_5') - pl.col('volatility_10')).alias('delta_volatility_5_10'),
             (pl.col('volatility_10') - pl.col('volatility_20')).alias('delta_volatility_10_20')
        ])
        current_features.extend(['delta_volatility_5_10', 'delta_volatility_10_20'])

        # --- Log returns ---
        shifted_price_log = pl.col(price_col).shift(1).over('event_id')
        feature_expressions.append(
             pl.when(shifted_price_log.is_not_null() & (shifted_price_log > 0) & pl.col(price_col).is_not_null() & (pl.col(price_col) > 0))
             .then(pl.ln(pl.col(price_col) / shifted_price_log))
             .otherwise(None).alias('log_ret')
        )
        current_features.append('log_ret')

        # Days to approval (already present)
        current_features.append('days_to_approval')

        # --- Lagged returns ---
        for lag in range(1, 4):
            col_name = f'ret_lag_{lag}'
            feature_expressions.append(pl.col(return_col).shift(lag).over('event_id').alias(col_name))
            current_features.append(col_name)


        # --- Pre-approval return (Groupby-Agg-Join approach) ---
        pre_approval_offset = pl.duration(days=-30)
        pre_approval_data = df.filter(
            (pl.col('date') <= pl.col('Approval Date')) & # Before or on approval date
            (pl.col('date') > (pl.col('Approval Date') + pre_approval_offset)) # Within 30 days before
        )
        # Compound return: product(1 + ret) - 1
        pre_approval_agg = pre_approval_data.group_by('event_id').agg(
             (pl.col(return_col).fill_null(0) + 1).product().alias("prod_ret_factor")
        ).with_columns(
            (pl.col("prod_ret_factor") - 1).alias('pre_approval_ret_30d')
        ).select(['event_id', 'pre_approval_ret_30d'])

        # Apply features calculated so far
        df = df.with_columns(feature_expressions)
        # Join pre-approval return
        df = df.join(pre_approval_agg, on='event_id', how='left')
        # Fill NaNs for events with no pre-approval data (set to 0)
        df = df.with_columns(pl.col('pre_approval_ret_30d').fill_null(0))
        current_features.append('pre_approval_ret_30d')

        feature_expressions = [] # Reset

        # --- More Volatility features ---
        feature_expressions.extend([
             pl.col(return_col).shift(1).over('event_id').abs().alias('prev_day_volatility'),
             pl.col(return_col).shift(1).over('event_id')
               .rolling_std(window_size=5, min_periods=2).over('event_id') # Rolling std on lagged returns
               .alias('prev_5d_vol_std')
        ])
        current_features.extend(['prev_day_volatility', 'prev_5d_vol_std'])

        # Approval date flag (already present)
        if 'is_approval_date' in df.columns: current_features.append('is_approval_date')

        # Apply final batch of features
        df = df.with_columns(feature_expressions)

        # --- Finalization ---
        self.feature_names = sorted(list(set(current_features)))
        print(f"Calculated {len(self.feature_names)} raw FDA features.")

        # Replace infinities generated during calculations
        feature_cols_in_df = [f for f in self.feature_names if f in df.columns]
        df = df.with_columns([
            pl.when(pl.col(c).is_infinite())
            .then(None)
            .otherwise(pl.col(c))
            .alias(c)
            for c in feature_cols_in_df if df[c].dtype in [pl.Float32, pl.Float64]
        ])

        # Select only necessary columns (original + features needed)
        # Example: Keep identifiers, date, returns, and calculated features
        final_cols_to_keep = ['event_id', 'ticker', 'date', 'Approval Date', 'ret', 'prc', 'days_to_approval', 'is_approval_date', 'future_ret'] + self.feature_names
        final_cols_to_keep = sorted(list(set(c for c in final_cols_to_keep if c in df.columns))) # Unique and existing

        return df.select(final_cols_to_keep)


    def get_features_target(self, df: pl.DataFrame, fit_imputer: bool = False) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Extract feature matrix X and target vector y as NumPy arrays, handling missing values.
        FDA version - simpler features, no categoricals handled here.

        Returns:
        Tuple[np.ndarray, np.ndarray, List[str]]: X_np, y_np, final_feature_names
        """
        print("Extracting FDA features (X) and target (y) as NumPy...")
        if not self.feature_names: raise RuntimeError("Run calculate_features first.")

        available_features = [f for f in self.feature_names if f in df.columns]
        if not available_features: raise ValueError("No calculated features found in DataFrame.")
        if 'future_ret' not in df.columns: raise ValueError("Target variable 'future_ret' not found.")

        # Replace infinities before dropping NaNs
        df = df.with_columns([
            pl.when(pl.col(c).is_infinite())
            .then(None)
            .otherwise(pl.col(c))
            .alias(c)
            for c in available_features if df[c].dtype in [pl.Float32, pl.Float64]
        ])

        # Drop rows where target is NaN
        df_with_target = df.drop_nulls(subset=['future_ret'])
        if df_with_target.is_empty():
            print("Warning: No data remains after filtering for non-null target.")
            return np.array([]).reshape(0, len(available_features)), np.array([]), available_features

        # Select features and target
        X_pl = df_with_target.select(available_features)
        y_pl = df_with_target.get_column('future_ret')

        print(f"Original X shape (Polars): {X_pl.shape}. Non-null y: {y_pl.len()}")

        # Convert features to NumPy for imputation/modeling
        try:
            X_np = X_pl.cast(pl.Float64).to_numpy() # Ensure float for imputer/models
        except Exception as e:
             raise ValueError(f"Failed to convert features to NumPy: {e}")

        initial_nan_count = np.isnan(X_np).sum()
        if initial_nan_count > 0: print(f"  Features contain {initial_nan_count} NaN values before imputation.")

        # Impute missing values using scikit-learn SimpleImputer on NumPy array
        if fit_imputer:
            print("Fitting imputer on NumPy data...")
            self.imputer.fit(X_np)
            self._imputer_fitted = True
            print("Transforming with imputer...")
            X_imputed_np = self.imputer.transform(X_np)
        else:
            if not self._imputer_fitted: raise RuntimeError("Imputer not fitted. Call with fit_imputer=True first.")
            print("Transforming NumPy data with pre-fitted imputer...")
            X_imputed_np = self.imputer.transform(X_np)

        final_nan_count = np.isnan(X_imputed_np).sum()
        if final_nan_count > 0:
            warnings.warn(f"NaNs ({final_nan_count}) remain AFTER imputation!")
        elif initial_nan_count > 0: print("No NaNs remaining after imputation.")
        else: print("No NaNs found before or after imputation.")

        # Convert target Polars Series to NumPy array
        y_np = y_pl.cast(pl.Float64).to_numpy()

        self.final_feature_names = available_features # Store the list of feature names used
        print(f"Final X NumPy shape: {X_imputed_np.shape}. y NumPy shape: {y_np.shape}. Using {len(self.final_feature_names)} features.")

        # Check final X_np for NaNs
        if np.isnan(X_imputed_np).any():
             warnings.warn("NaNs detected in the final imputed feature matrix X!")


        return X_imputed_np, y_np, self.final_feature_names


class Analysis:
    def __init__(self, data_loader: DataLoader, feature_engineer: FeatureEngineer):
        """Analysis class for FDA data using Polars."""
        self.data_loader = data_loader
        self.feature_engineer = feature_engineer
        self.data: Optional[pl.DataFrame] = None
        # Store NumPy arrays for ML models
        self.X_train_np: Optional[np.ndarray] = None
        self.X_test_np: Optional[np.ndarray] = None
        self.y_train_np: Optional[np.ndarray] = None
        self.y_test_np: Optional[np.ndarray] = None
        self.train_data: Optional[pl.DataFrame] = None # Store raw train split DF
        self.test_data: Optional[pl.DataFrame] = None # Store raw test split DF
        self.final_feature_names: Optional[List[str]] = None # Store final names after processing
        self.models: Dict[str, Any] = {}
        self.results: Dict[str, Dict] = {}

    def load_and_prepare_data(self):
        """Load and prepare data for FDA analysis using Polars."""
        print("--- Loading FDA Data (Polars) ---")
        self.data = self.data_loader.load_data()
        if self.data is None or self.data.is_empty():
            raise RuntimeError("Data loading failed.")
        print("\n--- Creating Target Variable (FDA - Polars) ---")
        self.data = self.feature_engineer.create_target(self.data)
        print("\n--- Calculating Features (FDA - Polars) ---")
        self.data = self.feature_engineer.calculate_features(self.data)
        print("\n--- FDA Data Preparation Complete ---")
        return self.data

    def train_models(self, test_size=0.2, time_split_column='Approval Date'):
        """Split data, process features/target, and train models using Polars/NumPy."""
        if self.data is None: raise RuntimeError("Run load_and_prepare_data() first.")
        if time_split_column not in self.data.columns: raise ValueError(f"Time split column '{time_split_column}' not found.")
        if 'event_id' not in self.data.columns: raise ValueError("'event_id' required.")
        # Assumes features/target already calculated by load_and_prepare_data

        print(f"\n--- Splitting FDA Data (Train/Test based on {time_split_column}) ---")
        # Find split date based on unique events
        events = self.data.select(['event_id', time_split_column]).unique().sort(time_split_column)
        split_index = int(events.height * (1 - test_size))
        if split_index == 0 or split_index == events.height: raise ValueError("test_size results in empty train/test set.")
        split_date = events.item(split_index, time_split_column)
        print(f"Splitting {events.height} unique events. Train before {split_date}.")

        # Filter main dataframe based on the split date
        train_mask = pl.col(time_split_column) < split_date
        test_mask = pl.col(time_split_column) >= split_date
        self.train_data = self.data.filter(train_mask) # Store raw train split
        self.test_data = self.data.filter(test_mask)   # Store raw test split
        print(f"Train rows (raw): {self.train_data.height}, Test rows (raw): {self.test_data.height}.")
        if self.train_data.is_empty() or self.test_data.is_empty():
            raise ValueError("Train or test split resulted in empty DataFrame.")

        print("\nExtracting features/target for TRAIN set (fitting imputer)...")
        # Fit imputer and get NumPy arrays for training
        self.X_train_np, self.y_train_np, _ = self.feature_engineer.get_features_target(self.train_data, fit_imputer=True)

        print("\nExtracting features/target for TEST set (transforming)...")
        # Transform test data using fitted imputer and get NumPy arrays
        self.X_test_np, self.y_test_np, self.final_feature_names = self.feature_engineer.get_features_target(self.test_data, fit_imputer=False)


        print(f"\nTrain shapes (NumPy): X={self.X_train_np.shape}, y={self.y_train_np.shape}")
        print(f"Test shapes (NumPy): X={self.X_test_np.shape}, y={self.y_test_np.shape}")
        if self.X_train_np.shape[0] == 0 or self.X_test_np.shape[0] == 0:
             raise ValueError("Train or test NumPy array empty after feature extraction.")

        print("\n--- Training Models (FDA) ---")
        # Models expect Polars DF input, but use NumPy internally.
        # We need to provide the imputed data as Polars DFs for consistency with model design.

        # Convert imputed NumPy arrays back to Polars DFs for model fitting
        try:
            X_train_pl_imputed = pl.DataFrame(self.X_train_np, schema=self.final_feature_names)
            y_train_pl = pl.Series("future_ret", self.y_train_np) # Target Series
        except Exception as e:
             raise RuntimeError(f"Could not convert imputed NumPy arrays back to Polars: {e}")


        # 1. TimeSeriesRidge (Expects Polars DF)
        try:
             print("Training TimeSeriesRidge...")
             ts_ridge = TimeSeriesRidge(alpha=1.0, lambda2=0.1, feature_order=self.final_feature_names)
             ts_ridge.fit(X_train_pl_imputed, y_train_pl)
             self.models['TimeSeriesRidge'] = ts_ridge
             print("TimeSeriesRidge complete.")
        except Exception as e: print(f"Error TimeSeriesRidge: {e}"); import traceback; traceback.print_exc()

        # 2. XGBoostDecile (Expects Polars DF)
        try:
             print("\nTraining XGBoostDecile...")
             xgb_params = {'n_estimators': 150, 'max_depth': 5, 'learning_rate': 0.05, 'objective': 'reg:squarederror', 'subsample': 0.8, 'colsample_bytree': 0.8, 'random_state': 42, 'n_jobs': -1, 'early_stopping_rounds': 10}

             if 'momentum_5' not in X_train_pl_imputed.columns: print("Warning: 'momentum_5' not found for XGBoostDecile.")

             xgb_decile = XGBoostDecileModel(weight=0.7, momentum_feature='momentum_5', n_deciles=10, alpha=1.0, lambda_smooth=0.1, xgb_params=xgb_params, ts_ridge_feature_order=self.final_feature_names)
             xgb_decile.fit(X_train_pl_imputed, y_train_pl) # Fit with Polars DF
             self.models['XGBoostDecile'] = xgb_decile
             print("XGBoostDecile complete.")
        except Exception as e: print(f"Error XGBoostDecile: {e}"); import traceback; traceback.print_exc()

        print("\n--- FDA Model Training Complete ---")
        return self.models

    def evaluate_models(self) -> Dict[str, Dict]:
        """Evaluate trained models on the test set using NumPy arrays."""
        print("\n--- Evaluating FDA Models ---")
        if not self.models: print("No models trained."); return {}
        if self.X_test_np is None or self.y_test_np is None or self.X_test_np.shape[0] == 0:
            print("Test data (NumPy) unavailable or empty."); return {}

        self.results = {}
        # Convert test NumPy features to Polars DF for model prediction input
        try:
            X_test_pl_imputed = pl.DataFrame(self.X_test_np, schema=self.final_feature_names)
        except Exception as e:
             print(f"Error converting test NumPy features to Polars DF: {e}. Cannot evaluate.")
             return {}


        for name, model in self.models.items():
            print(f"Evaluating {name}...")
            try:
                # Models predict using Polars DF, return NumPy array
                y_pred_np = model.predict(X_test_pl_imputed)

                # Use NumPy arrays for metrics
                valid_mask = np.isfinite(self.y_test_np) & np.isfinite(y_pred_np)
                y_test_valid, y_pred_valid = self.y_test_np[valid_mask], y_pred_np[valid_mask]

                if len(y_test_valid) > 0:
                    mse = mean_squared_error(y_test_valid, y_pred_valid)
                    r2 = r2_score(y_test_valid, y_pred_valid)
                else: mse, r2 = np.nan, np.nan
                self.results[name] = {'MSE': mse, 'RMSE': np.sqrt(mse), 'R2': r2, 'N': len(y_test_valid)}
                print(f"  {name}: MSE={mse:.6f}, RMSE={np.sqrt(mse):.6f}, R2={r2:.4f}, N={len(y_test_valid)}")
            except Exception as e:
                 print(f"  Error evaluating {name}: {e}")
                 self.results[name] = {'Error': str(e)}
                 import traceback; traceback.print_exc()

        print("\n--- FDA Evaluation Complete ---")
        return self.results


    def plot_feature_importance(self, results_dir: str, file_prefix: str = "fda", model_name: str = 'TimeSeriesRidge', top_n: int = 20):
        """Plot feature importance and save the plot. Uses model coefficients/importances."""
        print(f"\n--- Plotting FDA Feature Importance for {model_name} ---")
        if model_name not in self.models: print(f"Error: Model '{model_name}' not found."); return None
        model = self.models[model_name]
        # Use final_feature_names stored after processing test set
        feature_names = self.final_feature_names
        if not feature_names: print("Error: Final feature names not found (run training first)."); return None

        importances = None
        # Extract importances (model-specific)
        if isinstance(model, TimeSeriesRidge):
             if hasattr(model, 'coef_') and model.coef_ is not None:
                 if len(model.coef_) == len(feature_names):
                    importances = np.abs(model.coef_)
                 else: print(f"Warn: Ridge coef len ({len(model.coef_)}) != feature len ({len(feature_names)}). Cannot plot.")
        elif isinstance(model, XGBoostDecileModel):
             if hasattr(model, 'xgb_model') and hasattr(model.xgb_model, 'feature_importances_'):
                 xgb_importances = model.xgb_model.feature_importances_
                 if len(xgb_importances) == len(feature_names):
                     importances = xgb_importances
                 else:
                     # Fallback logic as in Earnings version
                     try:
                        booster = model.xgb_model.get_booster(); xgb_feat_names = booster.feature_names
                        if xgb_feat_names and len(xgb_feat_names) == len(xgb_importances):
                             imp_dict = dict(zip(xgb_feat_names, xgb_importances))
                             importances = np.array([imp_dict.get(name, 0) for name in feature_names])
                        else: raise ValueError("Mismatch")
                     except Exception: print(f"Warn: XGB importance len mismatch. Cannot plot reliably.")

        if importances is None: print(f"Could not get importance scores for {model_name}."); return None

        # Create Pandas DF for plotting
        feat_imp_df_pl = pl.DataFrame({'Feature': feature_names, 'Importance': importances}) \
                          .sort('Importance', descending=True) \
                          .head(top_n)
        feat_imp_df_pd = feat_imp_df_pl.to_pandas() # Convert top N to Pandas

        # Plotting
        fig, ax = plt.subplots(figsize=(10, max(6, top_n // 2)))
        sns.barplot(x='Importance', y='Feature', data=feat_imp_df_pd, palette='viridis', ax=ax)
        ax.set_title(f'Top {top_n} Features by Importance ({model_name} - FDA)')
        ax.set_xlabel('Importance Score'); ax.set_ylabel('Feature')
        plt.tight_layout()

        # Save plot
        plot_filename = os.path.join(results_dir, f"{file_prefix}_feat_importance_{model_name}.png")
        try: plt.savefig(plot_filename); print(f"Saved feature importance plot to: {plot_filename}")
        except Exception as e: print(f"Error saving feature importance plot: {e}")
        plt.close(fig)

        # Return Polars DF of importance
        return pl.DataFrame({'Feature': feature_names, 'Importance': importances}).sort('Importance', descending=True)


    def plot_predictions_for_event(self, results_dir: str, event_id: str, file_prefix: str = "fda", model_name: str = 'TimeSeriesRidge'):
        """Plot actual vs. predicted returns for a specific event using Polars data and save."""
        print(f"\n--- Plotting FDA Predictions for Event: {event_id} ({model_name}) ---")
        if model_name not in self.models: print(f"Error: Model '{model_name}' not found."); return None
        if self.data is None: print("Error: Data not loaded."); return None
        if not self.final_feature_names: print("Error: Features names not set (run training)."); return None
        model = self.models[model_name]

        # Find event data in the original dataset
        event_data_full = self.data.filter(pl.col('event_id') == event_id).sort('date')
        if event_data_full.is_empty(): print(f"Error: No data for event_id '{event_id}'."); return None

        ticker = event_data_full['ticker'][0]; approval_date = event_data_full['Approval Date'][0]

        # We need features for this event to predict
        # Extract features/target for this event using the FITTED imputer
        if not self.feature_engineer._imputer_fitted:
             print("Error: FeatureEngineer imputer not fitted (run training first)."); return None
        # Use the raw event data, get_features_target will handle imputation
        X_event_np, y_event_actual_np, event_features = self.feature_engineer.get_features_target(event_data_full, fit_imputer=False)

        if X_event_np.shape[0] == 0: print(f"Warn: No valid features/target rows for event {event_id}."); return None

        try:
            # Predict using Polars DF input
            X_event_pl = pl.DataFrame(X_event_np, schema=event_features)
            y_pred_event_np = model.predict(X_event_pl) # Returns NumPy
        except Exception as e: print(f"Error predicting event {event_id}: {e}"); return None

        # Align predictions back to the event dates
        event_data_pred_source = event_data_full.filter(pl.col('future_ret').is_not_null()) # Rows corresponding to predictions
        if event_data_pred_source.height != len(y_pred_event_np):
             print("Warn: Mismatch between prediction count and source data rows. Plot may be inaccurate.")
             # Pad or truncate prediction array? For now, proceed.
             min_len = min(event_data_pred_source.height, len(y_pred_event_np))
             event_data_pred_source = event_data_pred_source.head(min_len)
             y_pred_event_np = y_pred_event_np[:min_len]


        event_data_pred = event_data_pred_source.select(['date']).with_columns(
             pl.lit(y_pred_event_np).alias('predicted_future_ret')
        )

        # Convert to Pandas for plotting
        event_data_full_pd = event_data_full.select(['date', 'ret']).to_pandas()
        event_data_pred_pd = event_data_pred.to_pandas()
        approval_date_pd = pd.Timestamp(approval_date) # Requires pandas

        # Plotting
        fig, ax = plt.subplots(figsize=(14, 6))
        ax.plot(event_data_full_pd['date'], event_data_full_pd['ret'], '.-', label='Actual Daily Return', alpha=0.6, color='blue')
        ax.scatter(event_data_pred_pd['date'], event_data_pred_pd['predicted_future_ret'],
                    color='red', label=f'Predicted {self.feature_engineer.prediction_window}-day Future Ret', s=15, zorder=5, alpha=0.8)
        ax.axvline(x=approval_date_pd, color='g', linestyle='--', label='Approval Date')
        ax.set_title(f"{ticker} ({event_id}) - Daily Returns & Predicted Future Returns ({model_name} - FDA)")
        ax.set_ylabel("Return"); ax.set_xlabel("Date"); ax.legend(); ax.grid(True, alpha=0.3)
        plt.tight_layout()

        # Save plot
        safe_event_id = "".join(c if c.isalnum() else "_" for c in event_id)
        plot_filename = os.path.join(results_dir, f"{file_prefix}_pred_vs_actual_{safe_event_id}_{model_name}.png")
        try: plt.savefig(plot_filename); print(f"Saved prediction plot to: {plot_filename}")
        except Exception as e: print(f"Error saving prediction plot: {e}")
        plt.close(fig)

        return event_data_pred # Return Polars DF


    def find_sample_event_ids(self, n=5):
        """Find sample FDA event identifiers from Polars data."""
        if self.data is None or 'event_id' not in self.data.columns: return []
        unique_events = self.data['event_id'].unique().head(n)
        return unique_events.to_list()

    # --- Sharpe Ratio and Volatility Analysis using Polars ---

    def analyze_sharpe_ratio_dynamics(self, results_dir: str, file_prefix: str = "fda", risk_free_rate: float = 0.0, window: int = 20, min_periods: int = 10, pre_days: int = 30, post_days: int = 30):
        """Calculates, plots, and saves rolling Sharpe Ratio dynamics using Polars."""
        print(f"\n--- Analyzing Rolling Sharpe Ratio (Window={window} rows) using Polars ---")
        if self.data is None or 'ret' not in self.data.columns or 'days_to_approval' not in self.data.columns:
            print("Error: Data/required columns missing."); return None

        df = self.data.sort(['event_id', 'date']) # Ensure sorted

        # Calculate rolling mean and std deviation of returns within each event
        df = df.with_columns([
            pl.col('ret').rolling_mean(window_size=window, min_periods=min_periods).over('event_id').alias('rolling_mean_ret'),
            pl.col('ret').rolling_std(window_size=window, min_periods=min_periods).over('event_id').alias('rolling_std_ret')
        ])

        # Calculate daily and annualized Sharpe Ratio
        epsilon = 1e-8
        daily_risk_free = risk_free_rate / 252
        df = df.with_columns(
            ((pl.col('rolling_mean_ret') - daily_risk_free) / (pl.col('rolling_std_ret') + epsilon))
             .alias('daily_sharpe')
        )
        df = df.with_columns(
            (pl.col('daily_sharpe') * np.sqrt(252)).alias('annualized_sharpe')
        )

        # Align Sharpe Ratio by days relative to approval
        aligned_sharpe = df.group_by('days_to_approval').agg(
            pl.mean('annualized_sharpe').alias('avg_annualized_sharpe')
        ).filter(
            (pl.col('days_to_approval') >= -pre_days) &
            (pl.col('days_to_approval') <= post_days)
        ).sort('days_to_approval').drop_nulls()

        # --- Plotting & Saving ---
        if not aligned_sharpe.is_empty():
            aligned_sharpe_pd = aligned_sharpe.to_pandas().set_index('days_to_approval') # Convert for plot/save

            fig, ax = plt.subplots(figsize=(12, 6))
            aligned_sharpe_pd['avg_annualized_sharpe'].plot(kind='line', marker='.', linestyle='-', ax=ax)
            ax.axvline(0, color='red', linestyle='--', linewidth=1, label='Approval Day')
            ax.axhline(0, color='grey', linestyle=':', linewidth=1)
            ax.set_title(f'Average Annualized Rolling Sharpe Ratio Around FDA Approval (Window={window} rows)')
            ax.set_xlabel('Days Relative to Approval')
            ax.set_ylabel('Average Annualized Sharpe Ratio')
            ax.legend(); ax.grid(True, alpha=0.5)
            plt.tight_layout()

            plot_filename = os.path.join(results_dir, f"{file_prefix}_sharpe_ratio_rolling_{window}d.png")
            csv_filename = os.path.join(results_dir, f"{file_prefix}_sharpe_ratio_rolling_{window}d_data.csv")
            try: plt.savefig(plot_filename); print(f"Saved Sharpe plot to: {plot_filename}")
            except Exception as e: print(f"Error saving Sharpe plot: {e}")
            try: aligned_sharpe_pd.to_csv(csv_filename); print(f"Saved Sharpe data to: {csv_filename}") # Use Pandas to_csv
            except Exception as e: print(f"Error saving Sharpe data: {e}")
            plt.close(fig)

            print(f"Average Sharpe Ratio ({window}d rolling) in plot window: {aligned_sharpe_pd['avg_annualized_sharpe'].mean():.4f}")
        else:
            print("No data available for rolling Sharpe Ratio plot.")

        return aligned_sharpe # Return Polars DF


    def analyze_volatility_spikes(self, results_dir: str, file_prefix: str = "fda", window: int = 5, min_periods: int = 3, pre_days: int = 30, post_days: int = 30, baseline_window=(-60, -11), event_window=(-2, 2)):
        """Calculates, plots, and saves rolling volatility and event vs baseline comparison using Polars."""
        print(f"\n--- Analyzing Rolling Volatility (Window={window} rows) using Polars ---")
        if self.data is None or 'ret' not in self.data.columns or 'days_to_approval' not in self.data.columns:
            print("Error: Data/required columns missing."); return None

        df = self.data.sort(['event_id', 'date']) # Ensure sorted

        # Calculate rolling volatility (row-based window)
        df = df.with_columns(
            pl.col('ret').rolling_std(window_size=window, min_periods=min_periods)
              .over('event_id').alias('rolling_vol')
        )
        # Annualize volatility
        df = df.with_columns(
            (pl.col('rolling_vol') * np.sqrt(252) * 100).alias('annualized_vol') # In percent
        )

        # Align volatility by days relative to approval
        aligned_vol = df.group_by('days_to_approval').agg(
            pl.mean('annualized_vol').alias('avg_annualized_vol')
        ).filter(
            (pl.col('days_to_approval') >= -pre_days) &
            (pl.col('days_to_approval') <= post_days)
        ).sort('days_to_approval').drop_nulls()

        # --- Plotting & Saving Rolling Volatility ---
        if not aligned_vol.is_empty():
            aligned_vol_pd = aligned_vol.to_pandas().set_index('days_to_approval') # Convert for plot/save
            fig1, ax1 = plt.subplots(figsize=(12, 6))
            aligned_vol_pd['avg_annualized_vol'].plot(kind='line', marker='.', linestyle='-', ax=ax1)
            ax1.axvline(0, color='red', linestyle='--', lw=1, label='Approval Day')
            ax1.set_title(f'Average Rolling Volatility Around FDA Approval (Window={window} rows)')
            ax1.set_xlabel('Days Relative to Approval')
            ax1.set_ylabel('Avg. Annualized Volatility (%)')
            ax1.legend(); ax1.grid(True, alpha=0.5)
            plt.tight_layout()

            plot_filename_vol = os.path.join(results_dir, f"{file_prefix}_volatility_rolling_{window}d.png")
            csv_filename_vol = os.path.join(results_dir, f"{file_prefix}_volatility_rolling_{window}d_data.csv")
            try: plt.savefig(plot_filename_vol); print(f"Saved rolling vol plot to: {plot_filename_vol}")
            except Exception as e: print(f"Error saving plot: {e}")
            try: aligned_vol_pd.to_csv(csv_filename_vol); print(f"Saved rolling vol data to: {csv_filename_vol}")
            except Exception as e: print(f"Error saving data: {e}")
            plt.close(fig1)
        else:
            print("No data for rolling volatility plot.")

        # --- Compare Event vs Baseline Volatility ---
        vol_comp = df.group_by('event_id').agg([
            pl.std('ret').filter( # Baseline window std dev
                (pl.col('days_to_approval') >= baseline_window[0]) &
                (pl.col('days_to_approval') <= baseline_window[1])
            ).alias('vol_baseline'),
            pl.count('ret').filter( # Count for baseline
                 (pl.col('days_to_approval') >= baseline_window[0]) &
                 (pl.col('days_to_approval') <= baseline_window[1])
            ).alias('n_baseline'),
            pl.std('ret').filter( # Event window std dev
                (pl.col('days_to_approval') >= event_window[0]) &
                (pl.col('days_to_approval') <= event_window[1])
            ).alias('vol_event'),
             pl.count('ret').filter( # Count for event
                 (pl.col('days_to_approval') >= event_window[0]) &
                 (pl.col('days_to_approval') <= event_window[1])
            ).alias('n_event'),
        ]).filter( # Ensure enough data points and non-zero baseline vol
             (pl.col('n_baseline') >= min_periods) &
             (pl.col('n_event') >= min_periods) &
             (pl.col('vol_baseline').is_not_null()) &
             (pl.col('vol_baseline') > 1e-9) &
             (pl.col('vol_event').is_not_null())
        )

        if not vol_comp.is_empty():
            vol_ratios_df = vol_comp.with_columns(
                (pl.col('vol_event') / pl.col('vol_baseline')).alias('volatility_ratio')
            )

            avg_r = vol_ratios_df['volatility_ratio'].mean()
            med_r = vol_ratios_df['volatility_ratio'].median()
            num_valid_ratios = vol_ratios_df.height

            print(f"\nVolatility Spike (Event: {event_window}, Baseline: {baseline_window}): Avg Ratio={avg_r:.4f}, Median Ratio={med_r:.4f} ({num_valid_ratios} events)")

            # Save ratio data
            csv_filename_ratio = os.path.join(results_dir, f"{file_prefix}_volatility_ratios.csv")
            try:
                vol_ratios_df.select(['event_id', 'volatility_ratio']).write_csv(csv_filename_ratio)
                print(f"Saved vol ratios data to: {csv_filename_ratio}")
            except Exception as e: print(f"Error saving vol ratios: {e}")

            # Plot histogram of ratios (convert ratio column to Pandas)
            fig2, ax2 = plt.subplots(figsize=(8, 5))
            ratios_pd = vol_ratios_df['volatility_ratio'].to_pandas()
            sns.histplot(ratios_pd.dropna(), bins=30, kde=True, ax=ax2)
            ax2.axvline(1, c='grey', ls='--', label='Ratio = 1')
            ax2.axvline(avg_r, c='red', ls=':', label=f'Mean ({avg_r:.2f})')
            ax2.set_title('Distribution of Volatility Ratios (Event / Baseline)')
            ax2.set_xlabel('Volatility Ratio'); ax2.set_ylabel('Frequency')
            ax2.legend()
            plt.tight_layout()

            plot_filename_hist = os.path.join(results_dir, f"{file_prefix}_volatility_ratio_hist.png")
            try: plt.savefig(plot_filename_hist); print(f"Saved vol ratio hist plot: {plot_filename_hist}")
            except Exception as e: print(f"Error saving hist: {e}")
            plt.close(fig2)
        else:
            print("\nCould not calculate volatility ratios (insufficient valid data).")

        return aligned_vol # Return the aligned rolling vol dataframe
