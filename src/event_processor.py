import polars as pl
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns # Retained for potential use, though not directly in plots
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report
from sklearn.impute import SimpleImputer
import xgboost as xgb
import warnings
import os
import traceback  # Keep this import
import gc  # Keep this import
from typing import List, Optional, Tuple, Dict, Any
import datetime
import pandas as pd  # Required for plotting/CSV saving of some outputs

# Import shared models (assuming models.py is accessible)
try:
    # models.py now handles Polars input but uses NumPy internally
    from src.models import TimeSeriesRidge, XGBoostDecileModel, GARCHModel, GJRGARCHModel, ThreePhaseVolatilityModel
except ImportError:
    print("Error: Could not import models from 'models'.")
    print("Ensure models.py is in the same directory or Python path.")
    import sys
    sys.exit(1)

pl.Config.set_engine_affinity(engine="streaming")

# Suppress specific warnings if needed
warnings.filterwarnings('ignore', message='X does not have valid feature names, but SimpleImputer was fitted with feature names')
warnings.simplefilter(action='ignore', category=FutureWarning)


class EventDataLoader:
    def __init__(self, event_path: str, stock_paths: List[str], window_days: int = 30,
                 event_date_col: str = 'Event Date', ticker_col: str = 'ticker'):
        """
        Initialize DataLoader for events using Polars.

        Parameters:
        event_path (str): Path to the event data CSV. Must contain ticker and event date.
        stock_paths (list): List of paths to stock price/return data PARQUET files.
        window_days (int): Number of days before/after event date.
        event_date_col (str): Column name containing the event dates
        ticker_col (str): Column name containing ticker symbols
        """
        self.event_path = event_path
        if isinstance(stock_paths, str):
            self.stock_paths = [stock_paths]
        elif isinstance(stock_paths, list):
            self.stock_paths = stock_paths
        else:
            raise TypeError("stock_paths must be a string or a list of Parquet file paths.")

        self.window_days = window_days
        self.event_date_col = event_date_col
        self.ticker_col = ticker_col
        # Add chunk size parameter for event processing
        self.event_chunk_size = 89500  # Adjust based on memory capacity

    def _load_single_stock_parquet(self, stock_path: str) -> Optional[pl.DataFrame]:
        """Load and process a single stock data PARQUET file using Polars."""
        try:
            stock_data = pl.read_parquet(stock_path)

            original_columns = stock_data.columns
            col_map_lower = {col.lower(): col for col in original_columns}
            standard_names = {
                'date': ['date', 'trade_date', 'trading_date', 'tradedate', 'dt'],
                'ticker': ['ticker', 'symbol', 'sym_root', 'tic', 'permno'],
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
            selected_cols = []

            for std_name, variations in standard_names.items():
                found = False
                for var in variations:
                    if var in col_map_lower:
                        original_case_col = col_map_lower[var]
                        if original_case_col != std_name:
                            rename_dict[original_case_col] = std_name
                        found_std_names[std_name] = True
                        selected_cols.append(original_case_col)
                        found = True
                        break
                if not found:
                    found_std_names[std_name] = False

            # Check if essential columns were found before selecting
            essential_cols_found = all(found_std_names.get(c, False) for c in ['date', 'ticker', 'prc', 'ret'])
            if not essential_cols_found:
                 return None  # Skip this file if essentials are missing

            stock_data = stock_data.select(selected_cols)
            if rename_dict:
                stock_data = stock_data.rename(rename_dict)

            # --- Data Type and Existence Checks (on standardized names) ---
            # Note: Essential columns check already done implicitly above
            required_cols = ['date', 'ticker', 'prc', 'ret', 'vol']
            missing_cols = [col for col in required_cols if col not in stock_data.columns]
            if missing_cols:
                 essential_pr = ['prc', 'ret']  # Re-check just in case logic missed something
                 if any(col in missing_cols for col in essential_pr):
                      raise ValueError(f"Essential columns {missing_cols} missing in {stock_path} AFTER selection/rename.")
                 else:
                      pass  # Keep less verbose

            # --- Type Conversions ---
            if stock_data["date"].dtype != pl.Datetime:
                 stock_data = stock_data.with_columns(
                     pl.col("date").str.to_datetime(strict=False).alias("date")
                 )
            n_null_dates = stock_data.filter(pl.col("date").is_null()).height
            if n_null_dates > 0:
                stock_data = stock_data.drop_nulls(subset=["date"])

            if stock_data["ticker"].dtype != pl.Utf8:
                stock_data = stock_data.with_columns(pl.col("ticker").cast(pl.Utf8))

            numeric_cols_to_check = ['prc', 'ret', 'vol', 'openprc', 'askhi', 'bidlo', 'shrout']
            cast_expressions = []
            for col in numeric_cols_to_check:
                if col in stock_data.columns:
                    if stock_data[col].dtype not in [pl.Float32, pl.Float64]:
                         pass  # Keep less verbose
                    cast_expressions.append(
                        pl.when(pl.col(col).cast(pl.Float64, strict=False).is_infinite())
                        .then(None)
                        .otherwise(pl.col(col).cast(pl.Float64, strict=False))
                        .alias(col)
                    )
            if cast_expressions:
                 stock_data = stock_data.with_columns(cast_expressions)

            # --- Final Selection ---
            final_cols = list(standard_names.keys())
            cols_present = [col for col in final_cols if col in stock_data.columns]
            stock_data = stock_data.select(cols_present)
            return stock_data
        except FileNotFoundError:
             # print(f"Warning: Parquet file not found: {stock_path}")
             return None
        except Exception as e:
            # print(f"Warning: Error processing Parquet file {stock_path}: {e}")
            return None

    def load_data(self) -> Optional[pl.DataFrame]:
        """
        Load event dates (CSV) and stock data (PARQUET) using Polars,
        processing events in chunks to manage memory.
        """
        # --- Load Event Dates First ---
        try:
            # print(f"Loading event dates from: {self.event_path} (CSV)")
            event_df_peek = pl.read_csv_batched(self.event_path, batch_size=1).next_batches(1)[0]
            ticker_col = self.ticker_col
            date_col = self.event_date_col

            if ticker_col not in event_df_peek.columns:
                # Try to find ticker column if not found by name
                potential_ticker_cols = [c for c in event_df_peek.columns
                                        if 'ticker' in c.lower() or 'symbol' in c.lower()]
                if potential_ticker_cols:
                    ticker_col = potential_ticker_cols[0]
                    # print(f"Using '{ticker_col}' as ticker column")
                else:
                    raise ValueError(f"Ticker column '{self.ticker_col}' not found in event file.")

            if date_col not in event_df_peek.columns:
                # Try to find date column if not found by name
                potential_date_cols = [c for c in event_df_peek.columns
                                      if 'date' in c.lower() or 'day' in c.lower()]
                if potential_date_cols:
                    date_col = potential_date_cols[0]
                    # print(f"Using '{date_col}' as event date column")
                else:
                    raise ValueError(f"Event date column '{self.event_date_col}' not found.")

            # print(f"Using columns '{ticker_col}' (as ticker) and '{date_col}' (as Event Date) from event file.")
            event_data_raw = pl.read_csv(self.event_path, columns=[ticker_col, date_col], try_parse_dates=True)
            event_data_renamed = event_data_raw.rename({ticker_col: 'ticker', date_col: 'Event Date'})

            # --- Check/Correct Date Type ---
            if event_data_renamed['Event Date'].dtype == pl.Object or isinstance(event_data_renamed['Event Date'].dtype, pl.String):
                # print("    'Event Date' read as Object/String, attempting str.to_datetime...")
                event_data_processed = event_data_renamed.with_columns([
                     pl.col('Event Date').str.to_datetime(strict=False).cast(pl.Datetime),  # Explicit parse and cast
                     pl.col('ticker').cast(pl.Utf8).str.to_uppercase()
                 ])
            elif isinstance(event_data_renamed['Event Date'].dtype, (pl.Date, pl.Datetime)):
                 # print("    'Event Date' already parsed as Date/Datetime.")
                 event_data_processed = event_data_renamed.with_columns([
                     pl.col('Event Date').cast(pl.Datetime),  # Ensure it's Datetime specifically if needed
                     pl.col('ticker').cast(pl.Utf8).str.to_uppercase()
                 ])
            else:
                 raise TypeError(f"Unexpected dtype for 'Event Date': {event_data_renamed['Event Date'].dtype}")
            # --- End Corrected Date Handling ---

            event_data_processed = event_data_processed.drop_nulls(subset=['Event Date'])
            events = event_data_processed.unique(subset=['ticker', 'Event Date'], keep='first')
            n_total_events = events.height

            # print("\n--- Sample Parsed Events ---")
            # print(events.head(5))
            # print("-" * 35 + "\n")

            # print(f"Found {n_total_events} unique events (Ticker-Date pairs).")
            if events.is_empty(): raise ValueError("No valid events found.")

        except FileNotFoundError: raise FileNotFoundError(f"Event file not found: {self.event_path}")
        except Exception as e: raise ValueError(f"Error processing event file {self.event_path}: {e}")

        # --- Process Events in Chunks ---
        processed_chunks = []
        num_chunks = (n_total_events + self.event_chunk_size - 1) // self.event_chunk_size
        # print(f"Processing events in {num_chunks} chunk(s) of size {self.event_chunk_size}...")

        for i in range(num_chunks):
            start_idx = i * self.event_chunk_size
            end_idx = min((i + 1) * self.event_chunk_size, n_total_events)
            event_chunk = events.slice(start_idx, end_idx - start_idx)
            # print(f"--- Processing event chunk {i+1}/{num_chunks} ({event_chunk.height} events) ---")

            chunk_tickers = event_chunk['ticker'].unique()
            min_event_date = event_chunk['Event Date'].min()
            max_event_date = event_chunk['Event Date'].max()
            buffer = pl.duration(days=self.window_days + 1)  # Add buffer for safety
            required_min_date = min_event_date - buffer
            required_max_date = max_event_date + buffer

            # print(f"    Chunk Tickers: {chunk_tickers.len()} (Sample: {chunk_tickers[:5].to_list()})")
            # print(f"    Required Stock Date Range: {required_min_date} to {required_max_date}")

            stock_scans = []
            failed_stock_loads = 0
            # print("    Scanning and filtering stock Parquet files (lazily)...")
            for stock_path in self.stock_paths:
                try:
                    scan = pl.scan_parquet(stock_path)
                    # --- Standardization (Lazy) ---
                    original_columns = list(scan.collect_schema().names()) # Use collect_schema for safety
                    col_map_lower = {col.lower(): col for col in original_columns}
                    standard_names = {  # Keep consistent
                        'date': ['date', 'trade_date', 'trading_date', 'tradedate', 'dt'],
                        'ticker': ['ticker', 'symbol', 'sym_root', 'tic', 'permno'],
                        'prc': ['prc', 'price', 'close', 'adj close', 'adj_prc'],
                        'ret': ['ret', 'return', 'daily_ret'],
                        'vol': ['vol', 'volume'],
                        'openprc': ['openprc', 'open'], 'askhi': ['askhi', 'high', 'askhigh'],
                        'bidlo': ['bidlo', 'low', 'bidlow'], 'shrout': ['shrout', 'shares_outstanding']
                    }
                    rename_dict = {}
                    selected_orig_cols = []
                    for std_name, variations in standard_names.items():
                        for var in variations:
                            if var in col_map_lower:
                                original_case_col = col_map_lower[var]
                                if original_case_col != std_name:
                                    rename_dict[original_case_col] = std_name
                                selected_orig_cols.append(original_case_col)
                                break
                    if not any(sc in ['date','trade_date', 'trading_date', 'tradedate', 'dt'] for sc in col_map_lower) or \
                       not any(sc in ['ticker','symbol', 'sym_root', 'tic', 'permno'] for sc in col_map_lower):
                        # print(f"    Warning: Skipping {stock_path} as essential date or ticker column mapping not found.")
                        continue  # Skip file if essential keys aren't mapped

                    scan = scan.select(selected_orig_cols)
                    if rename_dict: scan = scan.rename(rename_dict)

                    # --- Lazy Type Conversion (Corrected) ---
                    current_schema = scan.collect_schema()
                    date_dtype = current_schema.get('date')
                    ticker_dtype = current_schema.get('ticker')

                    if date_dtype is None or ticker_dtype is None:
                         # print(f"    Warning: 'date' or 'ticker' column missing in {stock_path} schema after selection/rename. Skipping.")
                         continue

                    scan_expressions = []
                    if date_dtype == pl.Object or isinstance(date_dtype, pl.String): # String check for Polars
                        scan_expressions.append(pl.col("date").str.to_datetime(strict=False).cast(pl.Datetime))
                    elif date_dtype != pl.Datetime: # if not already datetime (e.g. Date)
                        scan_expressions.append(pl.col("date").cast(pl.Datetime))


                    if ticker_dtype != pl.Utf8:
                         scan_expressions.append(pl.col("ticker").cast(pl.Utf8).str.to_uppercase())
                    else: # Already Utf8, just ensure uppercase
                         scan_expressions.append(pl.col("ticker").str.to_uppercase())


                    if scan_expressions: # Only apply if there are changes
                        scan = scan.with_columns(scan_expressions)


                    # <<< --- Filtering Step --- >>>
                    filtered_scan = scan.filter(
                        pl.col('ticker').is_in(chunk_tickers) &
                        pl.col('date').is_between(required_min_date, required_max_date, closed='both')
                    )
                    stock_scans.append(filtered_scan)
                except Exception as e:
                    # print(f"    Warning: Failed to scan/prepare stock file {stock_path}: {e}")
                    failed_stock_loads += 1

            if not stock_scans:
                # print(f"    ERROR: No stock data could be scanned for chunk {i+1}. Skipping chunk.")
                continue

            # Concatenate scans
            combined_stock_scan = pl.concat(stock_scans, how='vertical_relaxed')

            # --- Collect Actual Stock Data ---
            # print(f"    Collecting filtered stock data for chunk {i+1}...")
            try:
                stock_data_chunk = combined_stock_scan.collect(streaming=True)
            except Exception as e:
                 # print(f"    ERROR collecting stock data for chunk {i+1}: {e}. Skipping chunk.")
                 # traceback.print_exc() # Keep for debugging if needed
                 continue

            # print(f"    Collected {stock_data_chunk.height} stock rows.")
            if stock_data_chunk.is_empty():
                 # print("    Collected stock data is empty. Skipping rest of chunk processing.")
                 continue  # Skip to next chunk if no stock data found

            # --- Standardize Types (Post-Collect) and Deduplicate ---
            cast_expressions_collect = []
            numeric_cols_to_check = ['prc', 'ret', 'vol', 'openprc', 'askhi', 'bidlo', 'shrout']
            for col_name in numeric_cols_to_check:
                 if col_name in stock_data_chunk.columns:
                      # Ensure column is float, handle potential infinities
                      cast_expressions_collect.append(
                           pl.when(pl.col(col_name).cast(pl.Float64, strict=False).is_infinite())
                           .then(None).otherwise(pl.col(col_name).cast(pl.Float64, strict=False))
                           .alias(col_name)
                      )
            if cast_expressions_collect:
                stock_data_chunk = stock_data_chunk.with_columns(cast_expressions_collect)

            # Ensure join keys are correct type before unique/join
            stock_data_chunk = stock_data_chunk.with_columns([
                pl.col('ticker').cast(pl.Utf8),
                pl.col('date').cast(pl.Datetime)
            ])
            stock_data_chunk = stock_data_chunk.unique(subset=['date', 'ticker'], keep='first', maintain_order=False)
            # print(f"    Deduplicated stock rows: {stock_data_chunk.height}")
            if stock_data_chunk.is_empty():
                 # print(f"    Warning: No stock data remained after deduplication for chunk {i+1}. Skipping chunk.")
                 continue

            # --- Merge event chunk with stock data chunk ---
            # print(f"    Merging events with stock data...")
            # Ensure join columns are same type
            event_chunk = event_chunk.with_columns([
                pl.col('ticker').cast(pl.Utf8),
                pl.col('Event Date').cast(pl.Datetime)
            ])
            merged_chunk = stock_data_chunk.join(
                event_chunk, on='ticker', how='inner'
            )
            # print(f"    Merged chunk rows: {merged_chunk.height}")
            if merged_chunk.is_empty():
                # print(f"    Warning: Merge resulted in empty data for chunk {i+1}. Check ticker matching.")
                continue  # Skip chunk

            # --- Calculate relative days and filter window FOR THE CHUNK ---
            processed_chunk = merged_chunk.with_columns(
                (pl.col('date') - pl.col('Event Date')).dt.total_days().cast(pl.Int32).alias('days_to_event')
            ).filter(
                (pl.col('days_to_event') >= -self.window_days) &
                (pl.col('days_to_event') <= self.window_days)
            )
            # print(f"    Rows after window filter ({self.window_days} days): {processed_chunk.height}")

            if processed_chunk.is_empty():
                # print(f"    Warning: No data found within event window for chunk {i+1}.")
                continue

            # --- Add final identifiers ---
            processed_chunk = processed_chunk.with_columns([
                (pl.col('days_to_event') == 0).cast(pl.Int8).alias('is_event_date'),
                (pl.col("ticker") + "_" + pl.col("Event Date").dt.strftime('%Y%m%d')).alias('event_id')
            ])

            # Select necessary columns
            stock_cols_present = stock_data_chunk.columns
            event_cols_present = event_chunk.columns # These were 'ticker' and 'Event Date'
            derived_cols_created = ['days_to_event', 'is_event_date', 'event_id']
            
            # Combine all potential columns, ensuring they exist in processed_chunk
            final_cols_set = set(stock_cols_present) | set(event_cols_present) | set(derived_cols_created)
            final_cols_to_select = [c for c in final_cols_set if c in processed_chunk.columns]


            processed_chunk = processed_chunk.select(final_cols_to_select)

            # print(f"    Processed chunk {i+1} FINAL shape: {processed_chunk.shape}")
            processed_chunks.append(processed_chunk)
            # print(f"--- Finished processing chunk {i+1} ---")

            del stock_data_chunk, merged_chunk, event_chunk, processed_chunk, combined_stock_scan, stock_scans
            gc.collect()

        # --- Final Concatenation ---
        if not processed_chunks:
            # print("Error: No data chunks were processed successfully.")
            return None  # Return None if no chunks succeeded

        # print("\nConcatenating processed chunks...")
        combined_data = pl.concat(processed_chunks, how='vertical').sort(['ticker', 'Event Date', 'date'])
        # print(f"Final dataset shape: {combined_data.shape}")
        # mem_usage_mb = combined_data.estimated_size("mb")
        # print(f"Final DataFrame memory usage: {mem_usage_mb:.2f} MB")

        if combined_data.is_empty():
             # print("Warning: Final combined data is empty after chunk processing.")
             return None

        return combined_data


# --- FeatureEngineer Class ---
class EventFeatureEngineer:
    def __init__(self, prediction_window: int = 3):
        self.windows = [5, 10, 20]  # Standard windows for technical indicators
        self.prediction_window = prediction_window
        self.imputer = SimpleImputer(strategy='median')  # Operates on NumPy
        self.feature_names: List[str] = []
        self.final_feature_names: List[str] = []  # After potential imputation/selection
        self.categorical_features: List[str] = []  # Store names of created categorical/dummy features
        self._imputer_fitted = False  # Track if imputer has been fitted

    def create_target(self, df: pl.DataFrame, price_col: str = 'prc') -> pl.DataFrame:
        """Create target variable using Polars."""
        # print(f"Creating target 'future_ret' (window: {self.prediction_window} days)...")
        if 'event_id' not in df.columns: raise ValueError("'event_id' required.")
        if price_col not in df.columns: raise ValueError(f"Price column '{price_col}' not found.")

        # Ensure sorted for correct shift
        df = df.sort(['event_id', 'date'])

        df = df.with_columns(
            pl.col(price_col).shift(-self.prediction_window).over('event_id').alias('future_price')
        ).with_columns(
            # Calculate return, handle division by zero or zero price
            pl.when(pl.col(price_col).is_not_null() & (pl.col(price_col) != 0))
            .then((pl.col('future_price') / pl.col(price_col)) - 1)
            .otherwise(None)  # Set to null if price is 0 or null
            .alias('future_ret')
        ).drop('future_price')

        # print(f"'future_ret' created. Non-null: {df.filter(pl.col('future_ret').is_not_null()).height}")
        return df

    def calculate_features(self, df: pl.DataFrame, price_col: str = 'prc', return_col: str = 'ret',
                           volume_col: str = 'vol', fit_categorical: bool = False) -> pl.DataFrame:
        """Calculate features for event analysis using Polars. Robust to missing optional columns."""
        # print("Calculating event features (Polars)...")
        required = ['event_id', price_col, return_col, 'Event Date', 'date', 'days_to_event']
        missing = [col for col in required if col not in df.columns]
        if missing: raise ValueError(f"Missing required columns for feature calculation: {missing}")

        has_volume = volume_col in df.columns
        # if not has_volume: print(f"Info: Volume column '{volume_col}' not found. Volume features skipped.")

        # Ensure sorted within event groups for rolling/shift operations
        df = df.sort(['event_id', 'date'])
        current_features: List[str] = []
        self.categorical_features = []  # Reset

        feature_expressions = []

        # --- Technical Features ---
        for window in self.windows:
            col_name = f'momentum_{window}'
            # Avoid division by zero in pct_change calculation
            shifted_price = pl.col(price_col).shift(window).over('event_id')
            feature_expressions.append(
                pl.when(shifted_price.is_not_null() & (shifted_price != 0))
                .then((pl.col(price_col) / shifted_price) - 1)
                .otherwise(None)
                .alias(col_name)
            )
            current_features.append(col_name)

        # Add momentum cols first before calculating deltas
        df = df.with_columns(feature_expressions)
        feature_expressions = []  # Reset

        feature_expressions.append((pl.col('momentum_5') - pl.col('momentum_10')).alias('delta_momentum_5_10'))
        current_features.append('delta_momentum_5_10')
        feature_expressions.append((pl.col('momentum_10') - pl.col('momentum_20')).alias('delta_momentum_10_20'))
        current_features.append('delta_momentum_10_20')

        for window in self.windows:
            col_name = f'volatility_{window}'
            min_p = max(2, min(window, 5))
            # Use rolling_std over event_id
            feature_expressions.append(pl.col(return_col).rolling_std(window_size=window, min_periods=min_p).over('event_id').alias(col_name))
            current_features.append(col_name)

        # Add volatility cols first before calculating deltas
        df = df.with_columns(feature_expressions)
        feature_expressions = []  # Reset

        feature_expressions.append((pl.col('volatility_5') - pl.col('volatility_10')).alias('delta_volatility_5_10'))
        current_features.append('delta_volatility_5_10')
        feature_expressions.append((pl.col('volatility_10') - pl.col('volatility_20')).alias('delta_volatility_10_20'))
        current_features.append('delta_volatility_10_20')

        # Log returns
        shifted_price_log = pl.col(price_col).shift(1).over('event_id')
        feature_expressions.append(
             pl.when(shifted_price_log.is_not_null() & (shifted_price_log > 0) & pl.col(price_col).is_not_null() & (pl.col(price_col) > 0) )
             .then(pl.Expr.log(pl.col(price_col) / shifted_price_log))
             .otherwise(None)
             .alias('log_ret')
        )
        current_features.append('log_ret')
        current_features.append('days_to_event')  # Already exists

        for lag in range(1, 4):
            col_name = f'ret_lag_{lag}'
            feature_expressions.append(pl.col(return_col).shift(lag).over('event_id').alias(col_name))
            current_features.append(col_name)

        # Apply features calculated so far
        df = df.with_columns(feature_expressions)
        feature_expressions = []  # Reset for next batch

        # --- Volume Features (Conditional) ---
        if has_volume:
            vol_mean_20d = pl.col(volume_col).rolling_mean(window_size=20, min_periods=5).over('event_id')
            df = df.with_columns(
                pl.when(vol_mean_20d.is_not_null() & (vol_mean_20d != 0))
                .then(pl.col(volume_col) / vol_mean_20d)
                .otherwise(None)
                .alias('norm_vol')
            )
            current_features.append('norm_vol')

            for window in [5, 10]:
                col_name = f'vol_momentum_{window}'
                shifted_vol = pl.col(volume_col).shift(window).over('event_id')
                df = df.with_columns(
                    pl.when(shifted_vol.is_not_null() & (shifted_vol != 0))
                    .then((pl.col(volume_col) / shifted_vol) - 1)
                    .otherwise(None)
                    .alias(col_name)
                )
                current_features.append(col_name)
        else:  # Add null columns if volume missing
             df = df.with_columns([pl.lit(None).cast(pl.Float64).alias(c) for c in ['norm_vol', 'vol_momentum_5', 'vol_momentum_10']])
             current_features.extend(['norm_vol', 'vol_momentum_5', 'vol_momentum_10'])

        # --- Pre-event Return (Requires group_by, agg, join) ---
        pre_event_start_offset = pl.duration(days=-30)
        pre_event_data = df.filter(
            (pl.col('date') < pl.col('Event Date')) &
            (pl.col('date') >= (pl.col('Event Date') + pre_event_start_offset))
        )
        pre_event_agg = pre_event_data.group_by('event_id').agg(
             (pl.col(return_col).fill_null(0) + 1).product().alias("prod_ret_factor")
        ).with_columns(
            (pl.col("prod_ret_factor") - 1).alias('pre_event_ret_30d')
        ).select(['event_id', 'pre_event_ret_30d'])

        df = df.join(pre_event_agg, on='event_id', how='left')
        df = df.with_columns(pl.col('pre_event_ret_30d').fill_null(0)) # Fill nulls after join
        current_features.append('pre_event_ret_30d')

        self.feature_names = sorted(list(set(current_features)))
        # print(f"Calculated {len(self.feature_names)} raw event features.")

        # Replace infinities generated during calculations
        feature_cols_in_df = [f for f in current_features if f in df.columns]
        df = df.with_columns([
            pl.when(pl.col(c).is_infinite())
            .then(None)
            .otherwise(pl.col(c))
            .alias(c)
            for c in feature_cols_in_df if df.select(pl.col(c)).dtypes[0] in [pl.Float32, pl.Float64]
        ])

        # Select final columns
        base_required = ['ticker', 'date', 'Event Date', 'ret', 'prc', 'days_to_event', 'event_id', 'future_ret']
        # Add any other input columns if they exist and aren't already features
        keep_cols = base_required + self.feature_names
        final_cols_to_keep = sorted(list(set(c for c in keep_cols if c in df.columns)))  # Unique and existing

        return df.select(final_cols_to_keep)


    def get_features_target(self, df: pl.DataFrame, fit_imputer: bool = False) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Extract feature matrix X and target vector y as NumPy arrays, handling missing values.
        Returns: Tuple[np.ndarray, np.ndarray, List[str]]: X_np, y_np, final_feature_names
        """
        # print("Extracting features (X) and target (y) as NumPy...")
        if not self.feature_names: raise RuntimeError("Run calculate_features first.")

        available_features = [f for f in self.feature_names if f in df.columns]
        if not available_features: raise ValueError("No calculated features found in DataFrame.")
        if 'future_ret' not in df.columns: raise ValueError("Target variable 'future_ret' not found in DataFrame.")

        # Replace infinities before dropping NaNs
        df = df.with_columns([
            pl.when(pl.col(c).is_infinite())
            .then(None)
            .otherwise(pl.col(c))
            .alias(c)
            for c in available_features if df.select(pl.col(c)).dtypes[0] in [pl.Float32, pl.Float64]
        ])

        df_with_target = df.drop_nulls(subset=['future_ret'])
        if df_with_target.is_empty():
            # print("Warning: No data remains after filtering for non-null target.")
            numeric_features = [f for f in available_features if f not in self.categorical_features]
            categorical_cols_present = [f for f in available_features if f in self.categorical_features]
            num_cols = len(numeric_features) + len(categorical_cols_present)
            return np.array([]).reshape(0, num_cols), np.array([]), numeric_features + categorical_cols_present

        # Separate numeric and categorical features
        numeric_features = [f for f in available_features if f not in self.categorical_features]
        categorical_cols_present = [f for f in available_features if f in self.categorical_features]
        categorical_df = df_with_target.select(categorical_cols_present)
        X_numeric_pl = df_with_target.select(numeric_features)
        y_pl = df_with_target.get_column('future_ret')

        # print(f"Original numeric X (Polars): {X_numeric_pl.shape}. Categorical X (Polars): {categorical_df.shape}. Non-null y: {y_pl.len()}")

        # Convert numeric features to NumPy
        try:
            X_numeric_np = X_numeric_pl.select(
                [pl.col(c).cast(pl.Float64, strict=False) for c in numeric_features]
            ).to_numpy()
        except Exception as e: raise ValueError(f"Failed to convert numeric features to NumPy: {e}")

        initial_nan_count = np.isnan(X_numeric_np).sum()
        # if initial_nan_count > 0: print(f"  Numeric features contain {initial_nan_count} NaN values before imputation.")

        # Impute missing values
        if fit_imputer:
            # print("Fitting imputer on numeric NumPy data...")
            if X_numeric_np.size > 0:
                self.imputer.fit(X_numeric_np)
                self._imputer_fitted = True
                # print("Transforming with imputer...")
                X_numeric_imputed_np = self.imputer.transform(X_numeric_np)
            else:
                 # print("Warning: Numeric feature array is empty. Skipping imputer fitting.")
                 X_numeric_imputed_np = X_numeric_np
                 self._imputer_fitted = True # Mark as fitted even if empty to prevent error
        else:
            if not self._imputer_fitted: raise RuntimeError("Imputer not fitted. Call with fit_imputer=True first.")
            # print("Transforming numeric NumPy data with pre-fitted imputer...")
            if X_numeric_np.size > 0:
                 X_numeric_imputed_np = self.imputer.transform(X_numeric_np)
            else:
                 X_numeric_imputed_np = X_numeric_np

        final_nan_count_numeric = np.isnan(X_numeric_imputed_np).sum()
        # if final_nan_count_numeric > 0: warnings.warn(f"NaNs ({final_nan_count_numeric}) remain in numeric features AFTER imputation!")
        # elif initial_nan_count > 0: print("No NaNs remaining in numeric features after imputation.")
        # else: print("No NaNs found in numeric features before or after imputation.")

        # Convert categorical features to NumPy
        if categorical_df.width > 0:
            try:
                X_categorical_np = categorical_df.select(
                    [pl.col(c).cast(pl.UInt8, strict=False) for c in categorical_cols_present]
                ).to_numpy()
            except Exception as e: raise ValueError(f"Failed to convert categorical features to NumPy: {e}")
        else:
             X_categorical_np = np.empty((X_numeric_imputed_np.shape[0], 0), dtype=np.uint8)
             # print("No categorical features found/used.")

        # if X_categorical_np.size > 0 and np.isnan(X_categorical_np).any(): warnings.warn("NaNs detected in categorical features after conversion!")

        # Combine NumPy arrays
        if X_numeric_imputed_np.size > 0 and X_categorical_np.size > 0:
            X_np = np.concatenate([X_numeric_imputed_np, X_categorical_np], axis=1)
            self.final_feature_names = numeric_features + categorical_cols_present
        elif X_numeric_imputed_np.size > 0:
            X_np = X_numeric_imputed_np
            self.final_feature_names = numeric_features
        elif X_categorical_np.size > 0:
             X_np = X_categorical_np
             self.final_feature_names = categorical_cols_present
        else:
             num_potential_cols = len(numeric_features) + len(categorical_cols_present)
             X_np = np.empty((0, num_potential_cols), dtype=np.float64)
             self.final_feature_names = numeric_features + categorical_cols_present


        y_np = y_pl.cast(pl.Float64).to_numpy()

        # print(f"Final X NumPy shape: {X_np.shape}. y NumPy shape: {y_np.shape}. Using {len(self.final_feature_names)} features.")
        final_nan_count_all = np.isnan(X_np).sum()
        # if final_nan_count_all > 0: warnings.warn(f"NaNs ({final_nan_count_all}) detected in the final combined feature matrix X!")

        return X_np, y_np, self.final_feature_names

class EventAnalysis:
    def __init__(self, data_loader: EventDataLoader, feature_engineer: EventFeatureEngineer):
        self.data_loader = data_loader
        self.feature_engineer = feature_engineer
        self.data: Optional[pl.DataFrame] = None
        self.models: Dict[str, Dict[str, Any]] = {}

    def load_and_prepare_data(self, run_feature_engineering: bool = False) -> Optional[pl.DataFrame]:
        try:
            # print("EventAnalysis: Loading data...")
            combined_data = self.data_loader.load_data()
            if combined_data is None or combined_data.is_empty():
                print("Error: Failed to load data or data is empty.")
                self.data = None
                return None
            # print(f"EventAnalysis: Data loaded. Shape: {combined_data.shape}")

            if run_feature_engineering:
                # print("EventAnalysis: Running feature engineering...")
                if 'prc' not in combined_data.columns:
                     print("Error: 'prc' column missing, cannot create target for feature engineering.")
                     self.data = combined_data # Store raw data at least
                     return combined_data

                combined_data = self.feature_engineer.create_target(combined_data)
                combined_data = self.feature_engineer.calculate_features(combined_data)
                # print(f"EventAnalysis: Feature engineering complete. Shape: {combined_data.shape}")
            
            self.data = combined_data
            return combined_data
        except Exception as e:
            print(f"Error loading and preparing data in EventAnalysis: {e}")
            traceback.print_exc()
            self.data = None # Ensure data is None on error
            return None

    def train_models(self, test_size: float = 0.2, time_split_column: str = "Event Date"):
        # print("Training models...")
        if self.data is None:
            print("Error: No data loaded. Call load_and_prepare_data first.")
            return
        if 'future_ret' not in self.data.columns:
            print("Error: 'future_ret' (target) column not found in data. Cannot train models.")
            return

        try:
            X, y, feature_names = self.feature_engineer.get_features_target(self.data, fit_imputer=True)
            if X.shape[0] == 0 or y.shape[0] == 0:
                print("Error: No valid features or target extracted for model training.")
                return
            # print(f"Training models on {X.shape[0]} samples with {X.shape[1]} features.")

            # train_indices_in_X = None # Not used directly, but part of split logic
            if time_split_column in self.data.columns:
                data_for_split_base = self.data.filter(pl.col('future_ret').is_not_null())
                dates_for_split = data_for_split_base.get_column(time_split_column).to_numpy().flatten()

                if len(dates_for_split) != X.shape[0]:
                     # print(f"Warning: Mismatch in date count ({len(dates_for_split)}) and X samples ({X.shape[0]}). Falling back to random split.")
                     from sklearn.model_selection import train_test_split
                     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
                     X_train_pl = pl.DataFrame(X_train, schema=feature_names)
                     y_train_pl = pl.Series("future_ret", y_train) # Polars Series needs a name
                else:
                    sorted_indices_relative_to_X = np.argsort(dates_for_split)
                    split_idx = int(len(sorted_indices_relative_to_X) * (1 - test_size))
                    train_indices_in_X_local = sorted_indices_relative_to_X[:split_idx] # Shadowing class member for local use
                    test_indices_in_X = sorted_indices_relative_to_X[split_idx:]
                    X_train, X_test = X[train_indices_in_X_local], X[test_indices_in_X]
                    y_train, y_test = y[train_indices_in_X_local], y[test_indices_in_X]
                    X_train_pl = data_for_split_base[train_indices_in_X_local].select(feature_names)
                    y_train_pl = data_for_split_base[train_indices_in_X_local].get_column('future_ret')
                    # print(f"Time-based split: Train {X_train.shape[0]}, Test {X_test.shape[0]}")
            else:
                from sklearn.model_selection import train_test_split
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
                X_train_pl = pl.DataFrame(X_train, schema=feature_names)
                y_train_pl = pl.Series("future_ret", y_train) # Polars Series needs a name
                # print(f"Random split: Train {X_train.shape[0]}, Test {X_test.shape[0]}")

            tsridge_model = TimeSeriesRidge(alpha=0.1, lambda2=0.5, feature_order=feature_names) 
            tsridge_model.fit(X_train_pl, y_train_pl)
            self.models['TimeSeriesRidge'] = {
                'model': tsridge_model, 'X_train': X_train, 'y_train': y_train, 
                'X_test': X_test, 'y_test': y_test, 'feature_names': feature_names
            }
            xgb_params_config = {
                'n_estimators': 100, 'max_depth': 5, 'learning_rate': 0.1,
                'objective': 'reg:squarederror', 'random_state': 42
            }
            xgb_model_instance = XGBoostDecileModel(weight=0.7, xgb_params=xgb_params_config, ts_ridge_feature_order=feature_names)
            xgb_model_instance.fit(X_train_pl, y_train_pl)
            self.models['XGBoostDecile'] = {
                'model': xgb_model_instance, 'X_train': X_train, 'y_train': y_train, 
                'X_test': X_test, 'y_test': y_test, 'feature_names': feature_names
            }
            # print(f"Successfully trained {len(self.models)} models.")
        except Exception as e:
            print(f"Error training models: {e}")
            traceback.print_exc()
            pass

    def evaluate_models(self) -> Dict[str, Any]:
        # print("Evaluating models...")
        if not self.models:
            print("Error: No models trained. Call train_models first.")
            return {}
        results = {}
        try:
            for model_name, model_info in self.models.items():
                model = model_info['model']
                X_test_np = model_info['X_test'] 
                y_test = model_info['y_test']
                feature_names = model_info['feature_names']
                
                if X_test_np.shape[0] == 0:
                    print(f"Warning: X_test is empty for model {model_name}. Skipping evaluation.")
                    results[model_name] = {'mse': np.nan, 'rmse': np.nan, 'r2': np.nan, 'y_pred': np.array([]), 'y_test': y_test}
                    continue

                X_test_pl = pl.DataFrame(X_test_np, schema=feature_names)
                y_pred = model.predict(X_test_pl) 

                mse = mean_squared_error(y_test, y_pred)
                rmse = np.sqrt(mse)
                r2 = r2_score(y_test, y_pred)
                results[model_name] = {'mse': mse, 'rmse': rmse, 'r2': r2, 'y_pred': y_pred, 'y_test': y_test}
                # print(f"Model: {model_name}, RMSE: {rmse:.4f}, R²: {r2:.4f}")
            return results
        except Exception as e:
            print(f"Error evaluating models: {e}")
            traceback.print_exc()
            return {}

    def plot_feature_importance(self, results_dir: str, file_prefix: str, model_name: str):
        # print(f"Plotting feature importance for {model_name}...")
        if model_name not in self.models:
            print(f"Error: Model '{model_name}' not found in trained models.")
            return
        try:
            model_info = self.models[model_name]
            model = model_info['model']
            feature_names = model_info['feature_names']
            if model_name == 'TimeSeriesRidge':
                if hasattr(model, 'coef_') and model.coef_ is not None:
                    importances = np.abs(model.coef_)
                    indices = np.argsort(importances)[::-1]
                else:
                    print(f"Error: TimeSeriesRidge model '{model_name}' not fitted or has no coefficients.")
                    return
            elif model_name == 'XGBoostDecile':
                if hasattr(model, 'xgb_model') and model.xgb_model is not None and hasattr(model.xgb_model, 'feature_importances_'):
                    xgb_model_actual = model.xgb_model
                    importances = xgb_model_actual.feature_importances_
                    indices = np.argsort(importances)[::-1]
                else:
                    print(f"Error: XGBoostDecile model '{model_name}' not fitted or has no feature importances.")
                    return
            else:
                print(f"Error: Feature importance not implemented for model type '{model_name}'.")
                return
            
            sorted_importances = importances[indices]
            sorted_features = [feature_names[i] for i in indices]
            top_n = min(15, len(sorted_features))

            fig, ax = plt.subplots(figsize=(10, 6))
            ax.barh(np.arange(top_n), sorted_importances[:top_n][::-1], align='center')
            ax.set_yticks(np.arange(top_n))
            ax.set_yticklabels(sorted_features[:top_n][::-1])
            ax.set_xlabel('Importance')
            ax.set_ylabel('Feature')
            ax.set_title(f'Feature Importance - {model_name}')
            plt.tight_layout()
            plot_filename = os.path.join(results_dir, f"{file_prefix}_feat_importance_{model_name}.png")
            try:
                plt.savefig(plot_filename, dpi=200)
                # print(f"Saved feature importance plot to: {plot_filename}")
            except Exception as e:
                print(f"Warning: Could not save plot image {plot_filename}: {e}")
            plt.close(fig)
        except Exception as e:
            print(f"Error plotting feature importance: {e}")
            traceback.print_exc()

    def find_sample_event_ids(self, n: int = 3) -> List[str]:
        # print(f"Finding {n} sample event IDs...")
        if self.data is None or 'event_id' not in self.data.columns:
            print("Error: No data or 'event_id' column loaded.")
            return []
        try:
            event_ids = self.data.get_column('event_id').unique().to_list()
            if not event_ids:
                print("No event IDs found in data.")
                return []
            if len(event_ids) <= n: return event_ids
            import random
            random.seed(42) # for reproducibility
            return random.sample(event_ids, n)
        except Exception as e:
            print(f"Error finding sample event IDs: {e}")
            return []

    def plot_predictions_for_event(self, results_dir: str, event_id: str, file_prefix: str, model_name: str):
        # print(f"Plotting predictions for event {event_id} with {model_name}...")
        if self.data is None:
            print("Error: No data loaded.")
            return
        if model_name not in self.models:
            print(f"Error: Model '{model_name}' not found.")
            return
        try:
            event_data_pl = self.data.filter(pl.col('event_id') == event_id).sort('days_to_event')
            if event_data_pl.is_empty():
                print(f"Error: No data for event ID '{event_id}'.")
                return

            if 'future_ret' not in event_data_pl.columns:
                print(f"Error: 'future_ret' missing for event ID '{event_id}'.")
                return

            model_info = self.models[model_name]
            feature_names_fitted = model_info['feature_names']
            
            # Ensure all features are present for prediction
            missing_features = [f for f in feature_names_fitted if f not in event_data_pl.columns]
            if missing_features:
                print(f"Error: Missing features {missing_features} for event ID '{event_id}'. Cannot predict.")
                return

            X_event_pl = event_data_pl.select(feature_names_fitted)
            y_event_actual = event_data_pl.get_column('future_ret') 

            if X_event_pl.is_empty():
                print(f"Error: No valid features for event ID '{event_id}'.")
                return

            model = model_info['model']
            y_pred = model.predict(X_event_pl) 

            days = event_data_pl.get_column('days_to_event').to_numpy()
            actual_returns_pct = y_event_actual.fill_null(np.nan).to_numpy() * 100 
            predicted_returns_pct = y_pred * 100

            fig, ax = plt.subplots(figsize=(12, 7))
            ax.plot(days, actual_returns_pct, marker='o', linestyle='-', color='blue', label='Actual Returns (%)')
            ax.plot(days, predicted_returns_pct, marker='x', linestyle='--', color='red', label='Predicted Returns (%)')
            ax.axvline(x=0, color='green', linestyle='--', label='Event Day')
            ax.text(0.1, ax.get_ylim()[1] * 0.9, 'Event Day', color='green', ha='left')
            
            ticker = event_data_pl.get_column('ticker').head(1).item(0,0) if 'ticker' in event_data_pl.columns and not event_data_pl.is_empty() else "N/A"
            event_date_val = event_data_pl.get_column('Event Date').head(1).item(0,0) if 'Event Date' in event_data_pl.columns and not event_data_pl.is_empty() else "N/A"
            
            event_date_str = "N/A"
            if isinstance(event_date_val, (datetime.datetime, datetime.date)):
                event_date_str = event_date_val.strftime('%Y-%m-%d')
            elif event_date_val != "N/A":
                event_date_str = str(event_date_val)
                
            ax.set_title(f'Return Predictions - {ticker} Event on {event_date_str} - {model_name}')
            ax.set_xlabel('Days Relative to Event'); ax.set_ylabel('Returns (%)')
            ax.legend(); ax.grid(True, linestyle=':', alpha=0.7); plt.tight_layout()
            plot_filename = os.path.join(results_dir, f"{file_prefix}_pred_{model_name}_{event_id}.png")
            try:
                plt.savefig(plot_filename, dpi=200)
                # print(f"Saved prediction plot to: {plot_filename}")
            except Exception as e:
                print(f"Warning: Could not save plot image {plot_filename}: {e}")
            plt.close(fig)
        except Exception as e:
            print(f"Error plotting predictions: {e}"); traceback.print_exc()
            
    def calculate_rolling_sharpe_timeseries(self,
        results_dir: str,
        file_prefix: str = "event",
        return_col: str = 'ret',
        analysis_window: Tuple[int, int] = (-60, 60),
        sharpe_window: int = 5,
        annualize: bool = True,
        risk_free_rate: float = 0.0 # Assumed to be annual rate, will be converted to daily
    ) -> Optional[pl.DataFrame]:
        # print(f"\n--- Calculating Rolling Sharpe Ratio Time Series ---")
        # print(f"Analysis Window: {analysis_window}, Sharpe Window: {sharpe_window} days, Annual RF: {risk_free_rate}")

        if self.data is None or return_col not in self.data.columns or 'days_to_event' not in self.data.columns:
            print("Error: Data not loaded or missing required columns (return_col, days_to_event).")
            return None

        if sharpe_window < 3:
            # print("Warning: Sharpe window too small (<3 days). Setting to 3.")
            sharpe_window = 3

        # Convert annual risk_free_rate to daily
        daily_rf = (1 + risk_free_rate)**(1/252) - 1 if risk_free_rate != 0 else 0.0
        # print(f"Daily risk-free rate used: {daily_rf:.6f}")
        
        extended_start = analysis_window[0] - sharpe_window # Need earlier data for rolling window
        analysis_data = self.data.filter(
            (pl.col('days_to_event') >= extended_start) &
            (pl.col('days_to_event') <= analysis_window[1])
        ).sort(['event_id', 'days_to_event']) # Sort for rolling operations

        if analysis_data.is_empty():
            print(f"Error: No data found within extended analysis window [{extended_start}, {analysis_window[1]}].")
            return None

        # Determine if returns are in percentage or decimal form
        sample_returns = analysis_data.select(pl.col(return_col)).sample(n=min(100, analysis_data.height), shuffle=True, seed=42)
        avg_abs_return = sample_returns.select(pl.all().mean().abs()).item(0,0) if sample_returns.height > 0 and sample_returns.width > 0 else 0.0
        returns_in_pct = avg_abs_return > 0.05 # Heuristic: if avg abs ret > 5%, assume %
        # print(f"Detected returns format for Sharpe: {'Percentage form' if returns_in_pct else 'Decimal form'} (avg abs: {avg_abs_return:.4f})")

        return_col_for_calc = return_col
        if returns_in_pct:
            # print("Converting percentage returns to decimal form for Sharpe calculation")
            analysis_data = analysis_data.with_columns((pl.col(return_col) / 100).alias('decimal_return'))
            return_col_for_calc = 'decimal_return'
        
        # Clip returns to mitigate outliers before calculating mean/std
        analysis_data = analysis_data.with_columns(
            pl.col(return_col_for_calc).clip_quantile(0.01, 0.99).over('event_id').alias('clipped_return')
        )

        # Calculate rolling mean and std for each event_id
        # The sharpe ratio is calculated for each day based on the *preceding* sharpe_window days.
        # So, for day `d`, we use returns from `d-sharpe_window` to `d-1`.
        
        # We need to group by event_id first, then calculate rolling stats, then align by days_to_event.
        # This is tricky with Polars' current rolling over groups for this specific alignment.
        # A more robust way for daily aggregation:
        
        sharpe_data_list = []
        for day_t in range(analysis_window[0], analysis_window[1] + 1):
            # Define the lookback window for this specific day_t
            window_start_day = day_t - sharpe_window
            window_end_day = day_t - 1 
            
            # Filter data for this lookback window across all events
            window_data_for_day_t = analysis_data.filter(
                (pl.col('days_to_event') >= window_start_day) & 
                (pl.col('days_to_event') <= window_end_day)
            )
            
            if window_data_for_day_t.is_empty():
                sharpe_data_list.append({'days_to_event': day_t, 'sharpe_ratio': None, 'event_count': 0})
                continue

            # Calculate mean return and std dev for each event within this window
            event_stats = window_data_for_day_t.group_by('event_id').agg([
                pl.mean('clipped_return').alias('mean_ret'),
                pl.std('clipped_return').alias('std_dev'),
                pl.count().alias('obs_in_window')
            ]).filter( # Ensure enough observations and non-zero std dev
                (pl.col('obs_in_window') >= max(2, sharpe_window // 2 + 1)) & (pl.col('std_dev') > 1e-9) # Min periods for std
            )

            if event_stats.is_empty():
                sharpe_data_list.append({'days_to_event': day_t, 'sharpe_ratio': None, 'event_count': 0})
                continue

            # Calculate Sharpe for each event
            event_sharpes = event_stats.with_columns(
                ((pl.col('mean_ret') - daily_rf) / pl.col('std_dev')).alias('daily_sharpe')
            )
            if annualize:
                event_sharpes = event_sharpes.with_columns(
                    (pl.col('daily_sharpe') * np.sqrt(252)).alias('annualized_sharpe_event')
                )
                avg_sharpe_for_day_t = event_sharpes['annualized_sharpe_event'].mean()
            else:
                avg_sharpe_for_day_t = event_sharpes['daily_sharpe'].mean()
            
            sharpe_data_list.append({
                'days_to_event': day_t, 
                'sharpe_ratio': avg_sharpe_for_day_t, 
                'event_count': event_sharpes.height
            })

        sharpe_df = pl.DataFrame(sharpe_data_list)
        
        # Fill missing days if any (should not happen with loop but good practice)
        all_days_df = pl.DataFrame({'days_to_event': range(analysis_window[0], analysis_window[1] + 1)})
        sharpe_df = all_days_df.join(sharpe_df, on='days_to_event', how='left').sort('days_to_event')

        # Interpolate missing Sharpe ratios (e.g., if too few events on a day)
        sharpe_df = sharpe_df.with_columns(pl.col('sharpe_ratio').interpolate())
        
        # Smooth the resulting time series
        smooth_window_sharpe = min(7, max(3, sharpe_window // 2)) # adaptive smoothing window
        sharpe_df = sharpe_df.with_columns(
            pl.col('sharpe_ratio').rolling_mean(window_size=smooth_window_sharpe, min_periods=1, center=True).alias('smooth_sharpe')
        )

        valid_days = sharpe_df.filter(pl.col('sharpe_ratio').is_not_null()).height
        if valid_days < (analysis_window[1] - analysis_window[0] +1) * 0.5 : # If less than half days have data
             print(f"Warning: Only {valid_days} days have valid Sharpe ratios out of {analysis_window[1] - analysis_window[0] + 1}.")
        
        # print("Rolling Sharpe Ratio Summary (Smoothed): Min: {:.2f}, Max: {:.2f}, Mean: {:.2f}, Median: {:.2f}".format(
        #       sharpe_df['smooth_sharpe'].min() or 0, sharpe_df['smooth_sharpe'].max() or 0,
        #       sharpe_df['smooth_sharpe'].mean() or 0, sharpe_df['smooth_sharpe'].median() or 0))
        
        try:
            os.makedirs(results_dir, exist_ok=True)
            results_pd = sharpe_df.to_pandas()
            fig, ax = plt.subplots(figsize=(12, 7))
            ax.plot(results_pd['days_to_event'], results_pd['sharpe_ratio'], color='blue', linewidth=1, alpha=0.3, label='Raw Avg. Sharpe Ratio')
            ax.plot(results_pd['days_to_event'], results_pd['smooth_sharpe'], color='red', linewidth=2, label=f'{smooth_window_sharpe}-Day Smoothed Avg. Sharpe')
            ax.axvline(x=0, color='red', linestyle='--', label='Event Day')
            
            y_data_smooth = results_pd['smooth_sharpe'].dropna()
            if not y_data_smooth.empty:
                y_min_plot, y_max_plot = y_data_smooth.min(), y_data_smooth.max()
                y_padding = 0.1 * (y_max_plot - y_min_plot) if (y_max_plot - y_min_plot) > 1e-6 else 0.1
                y_lim_final = [y_min_plot - y_padding, y_max_plot + y_padding]
            else:
                y_lim_final = [-1, 1]
            ax.set_ylim(np.clip(y_lim_final, -5, 5)) # Clip plot y-axis for readability

            ax.set_title(f'Rolling Sharpe Ratio vs. Days to Event ({sharpe_window}-Day Lookback)')
            ax.set_xlabel('Days Relative to Event'); ax.set_ylabel('Sharpe Ratio' + (' (Annualized)' if annualize else ''))
            ax.legend(); ax.grid(True, linestyle=':', alpha=0.7)
            plot_filename = os.path.join(results_dir, f"{file_prefix}_rolling_sharpe_timeseries.png")
            plt.savefig(plot_filename, dpi=200, bbox_inches='tight'); plt.close(fig)
            # print(f"Saved rolling Sharpe plot to: {plot_filename}")
            sharpe_df.write_csv(os.path.join(results_dir, f"{file_prefix}_rolling_sharpe_timeseries.csv"))
            # print(f"Saved rolling Sharpe data to: {os.path.join(results_dir, f'{file_prefix}_rolling_sharpe_timeseries.csv')}")
        except Exception as e:
            print(f"Error plotting/saving Sharpe: {e}"); traceback.print_exc()
        return sharpe_df

    def calculate_sharpe_quantiles(self, results_dir: str, file_prefix: str = "event",
                          return_col: str = 'ret',
                          analysis_window: Tuple[int, int] = (-60, 60),
                          lookback_window: int = 10, # Lookback for individual event Sharpe calc
                          quantiles: List[float] = [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95],
                          annualize: bool = True,
                          risk_free_rate: float = 0.0): # Annual RF rate
        # print(f"\n--- Calculating Sharpe Ratio Quantiles (Lookback: {lookback_window}) ---")
        if self.data is None or return_col not in self.data.columns:
            print("Error: Data/return_col missing for Sharpe quantiles.")
            return None
        
        daily_rf = (1 + risk_free_rate)**(1/252) - 1 if risk_free_rate != 0 else 0.0
        # print(f"Daily risk-free rate used for quantiles: {daily_rf:.6f}")

        extended_start = analysis_window[0] - lookback_window
        
        # Prepare data: filter, handle return format, clip
        temp_analysis_data = self.data.filter(
            (pl.col('days_to_event') >= extended_start) & 
            (pl.col('days_to_event') <= analysis_window[1])
        )
        if temp_analysis_data.is_empty():
            print(f"Error: No data in extended window [{extended_start}, {analysis_window[1]}] for Sharpe quantiles.")
            return None
            
        sample_returns = temp_analysis_data.select(pl.col(return_col)).sample(n=min(100, temp_analysis_data.height),shuffle=True, seed=42)
        avg_abs_return = sample_returns.select(pl.all().mean().abs()).item(0,0) if sample_returns.height > 0 and sample_returns.width > 0 else 0.0
        returns_in_pct = avg_abs_return > 0.05
        # print(f"Returns format for Sharpe quantiles: {'Percentage' if returns_in_pct else 'Decimal'} (avg abs: {avg_abs_return:.4f})")
        
        return_col_for_calc = return_col
        if returns_in_pct:
            # print("Converting returns to decimal for Sharpe quantile calc")
            analysis_data_sq = temp_analysis_data.with_columns((pl.col(return_col) / 100).alias('decimal_return'))
            return_col_for_calc = 'decimal_return'
        else:
            analysis_data_sq = temp_analysis_data.clone() # Use a clone if no conversion needed
            if return_col != 'decimal_return': # Ensure consistent naming if original is not 'decimal_return'
                 analysis_data_sq = analysis_data_sq.rename({return_col: 'decimal_return'})
                 return_col_for_calc = 'decimal_return'


        analysis_data_sq = analysis_data_sq.with_columns(
            pl.col(return_col_for_calc).clip_quantile(0.01, 0.99).over('event_id').alias('clipped_return')
        )
        
        days_range = list(range(analysis_window[0], analysis_window[1] + 1))
        sharpe_quantile_data_list = []
        # print(f"Processing {len(days_range)} days for Sharpe quantiles...")

        for day_t in days_range:
            # Define lookback window for this day_t (returns from day_t-lookback to day_t-1)
            window_start_day = day_t - lookback_window
            window_end_day = day_t - 1
            
            current_lookback_data = analysis_data_sq.filter(
                (pl.col('days_to_event') >= window_start_day) & 
                (pl.col('days_to_event') <= window_end_day)
            )
            
            if current_lookback_data.is_empty():
                empty_res_dict = {"days_to_event": day_t, "event_count": 0}
                for q in quantiles: empty_res_dict.update({f"sharpe_q{int(q*100)}": None})
                sharpe_quantile_data_list.append(empty_res_dict)
                continue

            # Calculate Sharpe for each event based on its returns in current_lookback_data
            sharpe_by_event_df = current_lookback_data.group_by('event_id').agg([
                pl.mean('clipped_return').alias('mean_ret'), 
                pl.std('clipped_return').alias('std_dev'), 
                pl.count().alias('n_obs')
            ]).filter(
                (pl.col('n_obs') >= max(3, lookback_window // 2 + 1)) & (pl.col('std_dev') > 1e-9) # Min obs for std dev
            )
            
            valid_events_count = sharpe_by_event_df.height
            if valid_events_count >= 5: # Need enough events to calculate quantiles of Sharpe ratios
                sharpe_by_event_df = sharpe_by_event_df.with_columns(
                    ((pl.col('mean_ret') - daily_rf) / pl.col('std_dev')).alias('sharpe_daily_event')
                )
                if annualize:
                    sharpe_by_event_df = sharpe_by_event_df.with_columns(
                        (pl.col('sharpe_daily_event') * np.sqrt(252)).alias('sharpe_event_final')
                    )
                else:
                    sharpe_by_event_df = sharpe_by_event_df.rename({'sharpe_daily_event':'sharpe_event_final'})

                # Clip extreme event Sharpe ratios before taking quantiles
                sharpe_by_event_df = sharpe_by_event_df.with_columns(
                    pl.col('sharpe_event_final').clip_quantile(0.05, 0.95).alias('sharpe_event_final_clipped')
                )

                quantile_values_dict = {
                    f"sharpe_q{int(q*100)}": sharpe_by_event_df.select(
                        pl.col('sharpe_event_final_clipped').quantile(q, interpolation='linear')
                    ).item() for q in quantiles
                }
                day_results_dict = {"days_to_event": day_t, "event_count": valid_events_count, **quantile_values_dict}
            else:
                day_results_dict = {"days_to_event": day_t, "event_count": valid_events_count}
                for q in quantiles: day_results_dict.update({f"sharpe_q{int(q*100)}": None})
            
            sharpe_quantile_data_list.append(day_results_dict)
            del current_lookback_data, sharpe_by_event_df; gc.collect()
            
        results_df = pl.DataFrame(sharpe_quantile_data_list)
        try:
            os.makedirs(results_dir, exist_ok=True)
            results_pd = results_df.to_pandas()
            fig, ax = plt.subplots(figsize=(12, 7))
            for q_val in quantiles:
                col_name_plot = f"sharpe_q{int(q_val*100)}"
                if col_name_plot in results_pd.columns:
                    ax.plot(results_pd['days_to_event'], results_pd[col_name_plot], 
                            linewidth=(2 if q_val == 0.5 else 1), 
                            linestyle=('solid' if q_val == 0.5 else 'dashed'), 
                            label=f'Q{int(q_val*100)}')
            ax.axvline(x=0, color='red', linestyle='--', label='Event Day')
            
            all_q_data_plot = pd.concat([results_pd[f"sharpe_q{int(q*100)}"] for q in quantiles if f"sharpe_q{int(q*100)}" in results_pd.columns]).dropna()
            if not all_q_data_plot.empty:
                y_min_plot, y_max_plot = all_q_data_plot.min(), all_q_data_plot.max()
                y_padding = 0.1 * (y_max_plot - y_min_plot) if (y_max_plot - y_min_plot) > 1e-6 else 0.1
                y_lim_final = [y_min_plot - y_padding, y_max_plot + y_padding]
            else:
                y_lim_final = [-1, 1]
            ax.set_ylim(np.clip(y_lim_final, -5, 5)) # Clip plot y-axis

            ax.set_title(f'Sharpe Ratio Quantiles Around Events (Lookback: {lookback_window} days)')
            ax.set_xlabel('Days Relative to Event'); ax.set_ylabel('Sharpe Ratio' + (' (Annualized)' if annualize else ''))
            ax.legend(loc='best'); ax.grid(True, linestyle=':', alpha=0.7)
            plot_filename = os.path.join(results_dir, f"{file_prefix}_sharpe_quantiles.png")
            plt.savefig(plot_filename, dpi=200, bbox_inches='tight'); plt.close(fig)
            # print(f"Saved Sharpe quantiles plot to: {plot_filename}")
            results_df.write_csv(os.path.join(results_dir, f"{file_prefix}_sharpe_quantiles.csv"))
            # print(f"Saved Sharpe quantiles data to: {os.path.join(results_dir, f'{file_prefix}_sharpe_quantiles.csv')}")
        except Exception as e:
            print(f"Error plotting/saving Sharpe quantiles: {e}"); traceback.print_exc()
        return results_df

    def analyze_volatility_spikes(self,
                                 results_dir: str,
                                 file_prefix: str = "event",
                                 return_col: str = 'ret', # Added return_col parameter
                                 window: int = 5,
                                 min_periods: int = 3,
                                 pre_days: int = 30, # Data range for analysis (relative to event)
                                 post_days: int = 30,
                                 baseline_window=(-60, -11), # For comparison
                                 event_window=(-2, 2)):     # For comparison
        # print(f"\n--- Analyzing Rolling Volatility (Window={window} days) ---")
        if self.data is None or return_col not in self.data.columns or 'days_to_event' not in self.data.columns:
            print("Error: Data/required columns missing for volatility spikes.")
            return None
        
        # Ensure data covers the full range needed for rolling calcs and baseline/event windows
        min_required_day = min(-pre_days, baseline_window[0]) - window 
        max_required_day = max(post_days, event_window[1])
        
        df_filtered = self.data.filter(
            (pl.col('days_to_event') >= min_required_day) & 
            (pl.col('days_to_event') <= max_required_day)
        ).sort(['event_id', 'date']) # Sort for rolling operations

        if df_filtered.is_empty():
            print("Error: Not enough data in the required range for volatility spike analysis.")
            return None

        # Clip returns before calculating volatility
        df_filtered = df_filtered.with_columns(
            pl.col(return_col).clip_quantile(0.01, 0.99).over('event_id').alias('clipped_ret_volspike')
        )
        
        # Calculate rolling volatility
        df_filtered = df_filtered.with_columns(
            pl.col('clipped_ret_volspike').rolling_std(window_size=window, min_periods=min_periods).over('event_id').alias('rolling_vol_raw')
        )
        # Annualize volatility (assuming returns are daily decimal)
        # If returns were in %, they should be converted before this or std dev adjusted.
        # For now, assume decimal returns for 'ret' or that it's handled upstream.
        df_filtered = df_filtered.with_columns(
            (pl.col('rolling_vol_raw') * np.sqrt(252) * 100).alias('annualized_vol_pct') # Convert to percentage
        )
        
        # Aggregate across events for plotting (only for the pre_days to post_days range)
        plot_range_data = df_filtered.filter(
            (pl.col('days_to_event') >= -pre_days) & (pl.col('days_to_event') <= post_days)
        )
        aligned_vol = plot_range_data.group_by('days_to_event').agg([
            pl.mean('annualized_vol_pct').alias('avg_annualized_vol'),
            pl.median('annualized_vol_pct').alias('median_annualized_vol'),
            pl.count().alias('event_count') # Number of events with data on this day
        ]).sort('days_to_event')

        # Calculate baseline and event window volatilities for comparison
        baseline_data = df_filtered.filter(
            (pl.col('days_to_event') >= baseline_window[0]) & (pl.col('days_to_event') <= baseline_window[1])
        )
        event_data = df_filtered.filter(
            (pl.col('days_to_event') >= event_window[0]) & (pl.col('days_to_event') <= event_window[1])
        )
        
        baseline_vol_val = baseline_data['annualized_vol_pct'].mean()
        event_vol_val = event_data['annualized_vol_pct'].mean()
        
        vol_change_pct_val = float('nan')
        if baseline_vol_val is not None and baseline_vol_val > 1e-6 and event_vol_val is not None:
            vol_change_pct_val = ((event_vol_val / baseline_vol_val) - 1) * 100
        
        # print(f"Baseline volatility ({baseline_window[0]} to {baseline_window[1]} days): {baseline_vol_val or 0:.2f}%")
        # print(f"Event window volatility ({event_window[0]} to {event_window[1]} days): {event_vol_val or 0:.2f}%")
        # print(f"Volatility change: {vol_change_pct_val:+.2f}%" if not pd.isna(vol_change_pct_val) else "Volatility change: N/A")
        
        if not aligned_vol.is_empty():
            try:
                os.makedirs(results_dir, exist_ok=True)
                aligned_vol_pd = aligned_vol.to_pandas()
                fig, ax = plt.subplots(figsize=(12, 7))
                ax.plot(aligned_vol_pd['days_to_event'], aligned_vol_pd['avg_annualized_vol'], color='blue', linewidth=2, label='Avg. Annualized Volatility')
                if baseline_vol_val is not None:
                    ax.axhline(y=baseline_vol_val, color='green', linestyle='--', label=f'Baseline Vol: {baseline_vol_val:.1f}%')
                ax.axvline(x=0, color='red', linestyle='--', label='Event Day')
                ax.axvspan(event_window[0], event_window[1], color='yellow', alpha=0.2, label=f'Event Window ({event_window[0]} to {event_window[1]})')
                ax.set_title(f'Average Rolling Volatility Around Event (Window={window} days)')
                ax.set_xlabel('Days Relative to Event'); ax.set_ylabel('Avg. Annualized Volatility (%)')
                ax.legend(); ax.grid(True, linestyle=':', alpha=0.7); ax.set_ylim(bottom=0)
                plot_filename = os.path.join(results_dir, f"{file_prefix}_volatility_rolling_{window}d.png")
                plt.savefig(plot_filename, dpi=200, bbox_inches='tight'); plt.close(fig)
                # print(f"Saved rolling volatility plot to: {plot_filename}")
                aligned_vol.write_csv(os.path.join(results_dir, f"{file_prefix}_volatility_rolling_{window}d_data.csv"))
                # print(f"Saved rolling volatility data to: {os.path.join(results_dir, f'{file_prefix}_volatility_rolling_{window}d_data.csv')}")
            except Exception as e:
                print(f"Error plotting/saving VolatilitySpikes: {e}"); traceback.print_exc()
        # else: print("No data for rolling volatility plot.")
        return aligned_vol

    def calculate_volatility_quantiles(self,
                                     results_dir: str,
                                     file_prefix: str = "event",
                                     return_col: str = 'ret',
                                     analysis_window: Tuple[int, int] = (-60, 60),
                                     lookback_window: int = 10, # Lookback for individual event vol calc
                                     quantiles: List[float] = [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]):
        # print(f"\n--- Calculating Volatility Quantiles (Lookback: {lookback_window}) ---")
        if self.data is None or return_col not in self.data.columns:
            print("Error: Data/return_col missing for volatility quantiles.")
            return None
        
        extended_start = analysis_window[0] - lookback_window
        analysis_data_filtered = self.data.filter(
            (pl.col('days_to_event') >= extended_start) & 
            (pl.col('days_to_event') <= analysis_window[1])
        )
        if analysis_data_filtered.is_empty():
            print(f"Error: No data in extended window [{extended_start}, {analysis_window[1]}] for vol quantiles.")
            return None

        # Clip returns before calculating volatility for quantiles
        analysis_data_filtered = analysis_data_filtered.with_columns(
            pl.col(return_col).clip_quantile(0.01, 0.99).over('event_id').alias('clipped_return_volquant')
        )
        
        days_range_list = list(range(analysis_window[0], analysis_window[1] + 1))
        vol_quantile_data_list = []
        # print(f"Processing {len(days_range_list)} days for volatility quantiles...")

        for day_val in days_range_list:
            # Define lookback window for this day_val (volatility of returns from day_val-lookback to day_val-1)
            window_start_val = day_val - lookback_window
            window_end_val = day_val - 1
            
            current_window_data = analysis_data_filtered.filter(
                (pl.col('days_to_event') >= window_start_val) & 
                (pl.col('days_to_event') <= window_end_val)
            )
            
            if current_window_data.is_empty():
                empty_res_dict = {"days_to_event": day_val, "event_count": 0}
                for q in quantiles: empty_res_dict.update({f"vol_q{int(q*100)}": None})
                vol_quantile_data_list.append(empty_res_dict)
                continue

            # Calculate volatility for each event based on its returns in current_window_data
            vol_by_event_df = current_window_data.group_by('event_id').agg([
                pl.std('clipped_return_volquant').alias('vol_raw_event'), 
                pl.count().alias('n_obs')
            ]).filter( # Ensure enough observations for std calculation
                (pl.col('n_obs') >= max(3, lookback_window // 2 + 1)) & pl.col('vol_raw_event').is_not_null() & (pl.col('vol_raw_event') > 1e-9)
            )
            
            num_valid_events = vol_by_event_df.height
            if num_valid_events >= 5: # Need enough events to calculate quantiles of volatilities
                # Annualize and convert to percentage
                vol_by_event_df = vol_by_event_df.with_columns(
                    (pl.col('vol_raw_event') * np.sqrt(252) * 100).alias('annualized_vol_pct_event')
                )
                quantile_values_dict = {
                    f"vol_q{int(q*100)}": vol_by_event_df.select(
                        pl.col('annualized_vol_pct_event').quantile(q, interpolation='linear')
                    ).item() for q in quantiles
                }
                day_results_dict = {"days_to_event": day_val, "event_count": num_valid_events, **quantile_values_dict}
            else:
                day_results_dict = {"days_to_event": day_val, "event_count": num_valid_events}
                for q in quantiles: day_results_dict.update({f"vol_q{int(q*100)}": None})
            
            vol_quantile_data_list.append(day_results_dict)
            del current_window_data, vol_by_event_df; gc.collect()
            
        results_vol_df = pl.DataFrame(vol_quantile_data_list)
        try:
            os.makedirs(results_dir, exist_ok=True)
            results_vol_pd = results_vol_df.to_pandas()
            fig, ax = plt.subplots(figsize=(12, 7))
            for q_item in quantiles:
                col_name_str = f"vol_q{int(q_item*100)}"
                if col_name_str in results_vol_pd.columns:
                    ax.plot(results_vol_pd['days_to_event'], results_vol_pd[col_name_str], 
                            linewidth=(2 if q_item == 0.5 else 1), 
                            linestyle=('solid' if q_item == 0.5 else 'dashed'), 
                            label=f'Q{int(q_item*100)}')
            ax.axvline(x=0, color='red', linestyle='--', label='Event Day')
            ax.set_title(f'Volatility Quantiles Around Events (Lookback: {lookback_window} days)')
            ax.set_xlabel('Days Relative to Event'); ax.set_ylabel('Annualized Volatility (%)')
            ax.legend(loc='best'); ax.grid(True, linestyle=':', alpha=0.7); ax.set_ylim(bottom=0)
            plot_filename = os.path.join(results_dir, f"{file_prefix}_volatility_quantiles.png")
            plt.savefig(plot_filename, dpi=200, bbox_inches='tight'); plt.close(fig)
            # print(f"Saved volatility quantiles plot to: {plot_filename}")
            results_vol_df.write_csv(os.path.join(results_dir, f"{file_prefix}_volatility_quantiles.csv"))
            # print(f"Saved volatility quantiles data to: {os.path.join(results_dir, f'{file_prefix}_volatility_quantiles.csv')}")
        except Exception as e:
            print(f"Error plotting/saving Volatility Quantiles: {e}"); traceback.print_exc()
        return results_vol_df

    def analyze_mean_returns(self,
                           results_dir: str,
                           file_prefix: str = "event",
                           return_col: str = 'ret',
                           window: int = 5, # Rolling window for mean return
                           min_periods: int = 3,
                           pre_days: int = 60, # Data range for analysis
                           post_days: int = 60,
                           baseline_window=(-60, -11), # For comparison
                           event_window=(-2, 2)):     # For comparison
        # print(f"\n--- Analyzing Rolling Mean Returns (Window={window} days) ---")
        if self.data is None or return_col not in self.data.columns or 'days_to_event' not in self.data.columns:
            print("Error: Data/required columns missing for mean returns analysis.")
            return None

        min_required_day = min(-pre_days, baseline_window[0]) - window
        max_required_day = max(post_days, event_window[1])

        df = self.data.filter(
            (pl.col('days_to_event') >= min_required_day) & 
            (pl.col('days_to_event') <= max_required_day)
        ).sort(['event_id', 'date'])

        if df.is_empty():
            print("Error: Not enough data in the required range for mean returns analysis.")
            return None

        # Clip returns before calculating mean
        df = df.with_columns(
            pl.col(return_col).clip_quantile(0.01,0.99).over('event_id').alias('clipped_return_meanret')
        )
        
        # Calculate rolling mean return
        df = df.with_columns(
            pl.col('clipped_return_meanret').rolling_mean(window_size=window,min_periods=min_periods).over('event_id').alias('rolling_mean_return_raw')
        )
        # Convert to percentage for display
        df = df.with_columns(
            (pl.col('rolling_mean_return_raw') * 100).alias('rolling_mean_return_pct')
        )
        
        plot_range_data = df.filter(
            (pl.col('days_to_event') >= -pre_days) & (pl.col('days_to_event') <= post_days)
        )
        aligned_means = plot_range_data.group_by('days_to_event').agg([
            pl.mean('rolling_mean_return_pct').alias('avg_rolling_mean_return'),
            pl.median('rolling_mean_return_pct').alias('median_rolling_mean_return'),
            pl.count().alias('event_count')
        ]).sort('days_to_event')
        
        baseline_data = df.filter(
            (pl.col('days_to_event') >= baseline_window[0]) & (pl.col('days_to_event') <= baseline_window[1])
        )
        event_data = df.filter(
            (pl.col('days_to_event') >= event_window[0]) & (pl.col('days_to_event') <= event_window[1])
        )
        
        baseline_ret_val = baseline_data['rolling_mean_return_pct'].mean()
        event_ret_val = event_data['rolling_mean_return_pct'].mean()
        
        ret_change_val = float('nan')
        if baseline_ret_val is not None and event_ret_val is not None:
            ret_change_val = (event_ret_val - baseline_ret_val) # Absolute change in percentage points
        
        # print(f"Baseline return ({baseline_window[0]} to {baseline_window[1]} days): {baseline_ret_val or 0:.2f}%")
        # print(f"Event window return ({event_window[0]} to {event_window[1]} days): {event_ret_val or 0:.2f}%")
        # print(f"Return change: {ret_change_val:+.2f}%" if not pd.isna(ret_change_val) else "Return change: N/A")
        
        if not aligned_means.is_empty():
            try:
                os.makedirs(results_dir, exist_ok=True)
                aligned_means_pd = aligned_means.to_pandas()
                fig, ax = plt.subplots(figsize=(12, 7))
                ax.plot(aligned_means_pd['days_to_event'], aligned_means_pd['avg_rolling_mean_return'], color='blue', linewidth=2, label='Avg. Rolling Mean Return')
                if baseline_ret_val is not None:
                    ax.axhline(y=baseline_ret_val, color='green', linestyle='--', label=f'Baseline Ret: {baseline_ret_val:.2f}%')
                ax.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
                ax.axvline(x=0, color='red', linestyle='--', label='Event Day')
                ax.axvspan(event_window[0], event_window[1], color='yellow', alpha=0.2, label=f'Event Window ({event_window[0]} to {event_window[1]})')
                ax.set_title(f'Average Rolling Mean Return Around Event (Window={window} days)')
                ax.set_xlabel('Days Relative to Event'); ax.set_ylabel('Avg. Rolling Mean Return (%)')
                ax.legend(); ax.grid(True, linestyle=':', alpha=0.7)
                ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f%%'))
                plot_filename = os.path.join(results_dir, f"{file_prefix}_mean_return_rolling_{window}d.png")
                plt.savefig(plot_filename, dpi=200, bbox_inches='tight'); plt.close(fig)
                # print(f"Saved rolling mean return plot to: {plot_filename}")
                aligned_means.write_csv(os.path.join(results_dir, f"{file_prefix}_mean_return_rolling_{window}d_data.csv"))
                # print(f"Saved rolling mean return data to: {os.path.join(results_dir, f'{file_prefix}_mean_return_rolling_{window}d_data.csv')}")
            except Exception as e:
                print(f"Error plotting/saving Mean Returns: {e}"); traceback.print_exc()
        # else: print("No data for rolling mean returns plot.")
        return aligned_means

    def calculate_mean_returns_quantiles(self,
                                       results_dir: str,
                                       file_prefix: str = "event",
                                       return_col: str = 'ret',
                                       analysis_window: Tuple[int, int] = (-60, 60),
                                       lookback_window: int = 10, # Lookback for individual event mean ret calc
                                       quantiles: List[float] = [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]):
        # print(f"\n--- Calculating Mean Return Quantiles (Lookback: {lookback_window}) ---")
        if self.data is None or return_col not in self.data.columns:
            print("Error: Data/return_col missing for mean return quantiles.")
            return None
        
        extended_start = analysis_window[0] - lookback_window
        analysis_data_filtered = self.data.filter(
            (pl.col('days_to_event') >= extended_start) & 
            (pl.col('days_to_event') <= analysis_window[1])
        )
        if analysis_data_filtered.is_empty():
            print(f"Error: No data in extended window [{extended_start}, {analysis_window[1]}] for mean ret quantiles.")
            return None
            
        # Clip returns before calculating mean for quantiles
        analysis_data_filtered = analysis_data_filtered.with_columns(
            pl.col(return_col).clip_quantile(0.01, 0.99).over('event_id').alias('clipped_return_meanretquant')
        )
        
        days_range_list = list(range(analysis_window[0], analysis_window[1] + 1))
        mean_ret_quantile_data_list = []
        # print(f"Processing {len(days_range_list)} days for mean return quantiles...")

        for day_val in days_range_list:
            # Define lookback window for this day_val (mean ret from day_val-lookback to day_val-1)
            window_start_val = day_val - lookback_window
            window_end_val = day_val - 1
            
            current_window_data = analysis_data_filtered.filter(
                (pl.col('days_to_event') >= window_start_val) & 
                (pl.col('days_to_event') <= window_end_val)
            )
            
            if current_window_data.is_empty():
                empty_res_dict = {"days_to_event": day_val, "event_count": 0}
                for q in quantiles: empty_res_dict.update({f"mean_ret_q{int(q*100)}": None})
                mean_ret_quantile_data_list.append(empty_res_dict)
                continue

            # Calculate mean return for each event based on its returns in current_window_data
            mean_ret_by_event_df = current_window_data.group_by('event_id').agg([
                pl.mean('clipped_return_meanretquant').alias('mean_ret_raw_event'), 
                pl.count().alias('n_obs')
            ]).filter( # Ensure enough observations
                (pl.col('n_obs') >= max(3, lookback_window // 2 + 1)) & pl.col('mean_ret_raw_event').is_not_null()
            )
            
            num_valid_events = mean_ret_by_event_df.height
            if num_valid_events >= 5: # Need enough events to calculate quantiles of mean returns
                # Convert to percentage
                mean_ret_by_event_df = mean_ret_by_event_df.with_columns(
                    (pl.col('mean_ret_raw_event') * 100).alias('mean_ret_pct_event')
                )
                quantile_values_dict = {
                    f"mean_ret_q{int(q*100)}": mean_ret_by_event_df.select(
                        pl.col('mean_ret_pct_event').quantile(q, interpolation='linear')
                    ).item() for q in quantiles
                }
                day_results_dict = {"days_to_event": day_val, "event_count": num_valid_events, **quantile_values_dict}
            else:
                day_results_dict = {"days_to_event": day_val, "event_count": num_valid_events}
                for q in quantiles: day_results_dict.update({f"mean_ret_q{int(q*100)}": None})
            
            mean_ret_quantile_data_list.append(day_results_dict)
            del current_window_data, mean_ret_by_event_df; gc.collect()
            
        results_mean_ret_df = pl.DataFrame(mean_ret_quantile_data_list)
        try:
            os.makedirs(results_dir, exist_ok=True)
            results_mean_ret_pd = results_mean_ret_df.to_pandas()
            fig, ax = plt.subplots(figsize=(12, 7))
            for q_item in quantiles:
                col_name_str = f"mean_ret_q{int(q_item*100)}"
                if col_name_str in results_mean_ret_pd.columns:
                    ax.plot(results_mean_ret_pd['days_to_event'], results_mean_ret_pd[col_name_str], 
                            linewidth=(2 if q_item == 0.5 else 1), 
                            linestyle=('solid' if q_item == 0.5 else 'dashed'), 
                            label=f'Q{int(q_item*100)}')
            ax.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
            ax.axvline(x=0, color='red', linestyle='--', label='Event Day')
            ax.set_title(f'Mean Return Quantiles Around Events (Lookback: {lookback_window} days)')
            ax.set_xlabel('Days Relative to Event'); ax.set_ylabel('Mean Return (%)')
            ax.legend(loc='best'); ax.grid(True, linestyle=':', alpha=0.7)
            ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f%%'))
            plot_filename = os.path.join(results_dir, f"{file_prefix}_mean_return_quantiles.png")
            plt.savefig(plot_filename, dpi=200, bbox_inches='tight'); plt.close(fig)
            # print(f"Saved mean return quantiles plot to: {plot_filename}")
            results_mean_ret_df.write_csv(os.path.join(results_dir, f"{file_prefix}_mean_return_quantiles.csv"))
            # print(f"Saved mean return quantiles data to: {os.path.join(results_dir, f'{file_prefix}_mean_return_quantiles.csv')}")
        except Exception as e:
            print(f"Error plotting/saving Mean Return Quantiles: {e}"); traceback.print_exc()
        return results_mean_ret_df

    def analyze_rvr(self,
                        results_dir: str,
                        file_prefix: str = "event",
                        return_col: str = 'ret',
                        analysis_window: tuple[int, int] = (-30, 30),
                        post_event_delta: int = 10, 
                        lookback_window: int = 5,
                        optimistic_bias: float = 0.01, # Decimal, simple additive bias
                        min_periods: int = 3,
                        variance_floor: float = 1e-6,
                        rvr_clip_threshold: float = 1e10,
                        adaptive_threshold: bool = True):
            # print(f"\n--- Analyzing Return-to-Variance Ratio (RVR) - Simple Bias Method ---")
            # print(f"Analysis Window: {analysis_window}, Post-Event Delta (phase def): {post_event_delta} days, Lookback: {lookback_window} days")
            rvr_daily_df = None
            if self.data is None or return_col not in self.data.columns or 'days_to_event' not in self.data.columns:
                print("Error: Data not loaded or missing required columns for RVR analysis.")
                return rvr_daily_df
            if post_event_delta <= 0 or post_event_delta > analysis_window[1]:
                print(f"Error: post_event_delta ({post_event_delta}) must be positive and within analysis window for RVR.")
                return rvr_daily_df

            phases_dict = {'pre_event': (analysis_window[0], -1), 
                           'post_event_rising': (0, post_event_delta), # Phase definition for this RVR method
                           'late_post_event': (post_event_delta + 1, analysis_window[1])}
            
            extended_start_day = analysis_window[0] - lookback_window
            analysis_data_rvr = self.data.filter(
                (pl.col('days_to_event') >= extended_start_day) & (pl.col('days_to_event') <= analysis_window[1])
            ).sort(['event_id', 'days_to_event'])
            
            if analysis_data_rvr.is_empty():
                print(f"Error: No data found within extended RVR analysis window [{extended_start_day}, {analysis_window[1]}].")
                return rvr_daily_df
            
            # Determine return format
            sample_returns_rvr = analysis_data_rvr.select(pl.col(return_col)).sample(n=min(100, analysis_data_rvr.height),shuffle=True, seed=42)
            avg_abs_return_rvr = sample_returns_rvr.select(pl.all().mean().abs()).item(0,0) if sample_returns_rvr.height > 0 and sample_returns_rvr.width > 0 else 0.0

            if adaptive_threshold:
                pct_05_rvr = sample_returns_rvr.select(pl.col(return_col).quantile(0.05)).item(0,0) if sample_returns_rvr.height > 0 and sample_returns_rvr.width > 0 else 0.0
                pct_95_rvr = sample_returns_rvr.select(pl.col(return_col).quantile(0.95)).item(0,0) if sample_returns_rvr.height > 0 and sample_returns_rvr.width > 0 else 0.0
                range_check_rvr = (pct_95_rvr - pct_05_rvr) > 0.1 if pd.notna(pct_05_rvr) and pd.notna(pct_95_rvr) else False
                std_dev_sample = sample_returns_rvr.select(pl.col(return_col).std()).item(0,0)
                std_check_rvr = (std_dev_sample or 0.0) > 0.02 # Ensure not None
                returns_in_pct_rvr = range_check_rvr or std_check_rvr or (avg_abs_return_rvr > 0.05)
                # print(f"Adaptive return format detection (RVR simple): {'Percentage form' if returns_in_pct_rvr else 'Decimal form'}")
            else:
                returns_in_pct_rvr = avg_abs_return_rvr > 0.05
                # print(f"Standard return format detection (RVR simple): {'Percentage form' if returns_in_pct_rvr else 'Decimal form'}")
            
            # Standardize return column to decimal and clip
            calc_return_col_name = 'decimal_return_rvr_simple'
            if returns_in_pct_rvr:
                # print("Converting percentage returns to decimal for RVR simple bias calculation")
                analysis_data_rvr = analysis_data_rvr.with_columns(
                    (pl.col(return_col) / 100).alias(calc_return_col_name)
                )
            else: # Returns are already decimal
                 analysis_data_rvr = analysis_data_rvr.with_columns(
                    pl.col(return_col).alias(calc_return_col_name) # Ensure consistent name
                )
            analysis_data_rvr = analysis_data_rvr.with_columns(
                pl.col(calc_return_col_name).clip_quantile(0.01,0.99).over('event_id').alias(calc_return_col_name) # Clip after naming
            )
            
            # Calculate rolling mean return and volatility
            analysis_data_rvr = analysis_data_rvr.with_columns([
                pl.col(calc_return_col_name).rolling_mean(window_size=lookback_window, min_periods=min_periods).over('event_id').alias('mean_return_rvr_simple'),
                pl.col(calc_return_col_name).rolling_std(window_size=lookback_window, min_periods=min_periods).over('event_id').alias('volatility_rvr_simple')
            ])
            
            # Apply simple optimistic bias during the 'post_event_rising' phase (0 to post_event_delta)
            # optimistic_bias is assumed to be in decimal form
            analysis_data_rvr = analysis_data_rvr.with_columns(
                pl.when( (pl.col('days_to_event') > 0) & (pl.col('days_to_event') <= post_event_delta) )
                .then(pl.col('mean_return_rvr_simple') + optimistic_bias)
                .otherwise(pl.col('mean_return_rvr_simple'))
                .alias('expected_return_rvr_simple')
            )
            
            # Calculate variance (volatility squared), ensuring a floor
            analysis_data_rvr = analysis_data_rvr.with_columns(
                pl.max_horizontal(pl.col('volatility_rvr_simple') ** 2, pl.lit(variance_floor)).alias('variance_rvr_simple')
            )
            
            # Calculate RVR, handling division by zero and clipping
            analysis_data_rvr = analysis_data_rvr.with_columns(
                pl.when(pl.col('variance_rvr_simple') > 1e-9)
                .then(pl.col('expected_return_rvr_simple') / pl.col('variance_rvr_simple'))
                .otherwise(None) # Or some other fill value like 0 or NaN
                .alias('raw_rvr_calc_simple')
            ).with_columns(
                pl.col('raw_rvr_calc_simple').clip(-rvr_clip_threshold, rvr_clip_threshold).alias('rvr_final_simple')
            )
            
            # Aggregate daily RVR across events
            rvr_daily_df = analysis_data_rvr.filter( # Filter for the actual analysis window for aggregation
                (pl.col('days_to_event') >= analysis_window[0]) & (pl.col('days_to_event') <= analysis_window[1])
            ).group_by('days_to_event').agg([
                pl.col('rvr_final_simple').mean().alias('avg_rvr'), 
                pl.col('rvr_final_simple').median().alias('median_rvr'),
                pl.col('expected_return_rvr_simple').mean().alias('avg_expected_return'), 
                pl.col('variance_rvr_simple').mean().alias('avg_variance'),
                pl.col('rvr_final_simple').is_not_null().sum().alias('event_count') # Count non-null RVRs
            ]).sort('days_to_event')
            
            # Summarize RVR by defined phases
            phase_summaries_list = []
            for phase_name_str, (start_day_val, end_day_val) in phases_dict.items():
                phase_data_df = analysis_data_rvr.filter( # Use original analysis_data_rvr for phase stats before daily aggregation
                    (pl.col('days_to_event') >= start_day_val) & (pl.col('days_to_event') <= end_day_val) &
                    (pl.col('days_to_event') >= analysis_window[0]) & (pl.col('days_to_event') <= analysis_window[1]) # Ensure within overall window
                )
                if not phase_data_df.is_empty() and 'rvr_final_simple' in phase_data_df.columns:
                    phase_rvr_stats = phase_data_df.select(
                        pl.mean('rvr_final_simple').alias('avg_rvr'),
                        pl.median('rvr_final_simple').alias('median_rvr'),
                        pl.mean('expected_return_rvr_simple').alias('avg_expected_return'),
                        pl.mean('variance_rvr_simple').alias('avg_variance'),
                        pl.sum(pl.col('rvr_final_simple').is_not_null()).alias('event_count')
                    ).row(0, named=True) # Get as dict

                    phase_stats_dict = {'phase': phase_name_str, 'start_day': start_day_val, 'end_day': end_day_val, **phase_rvr_stats}
                else: 
                    phase_stats_dict = {'phase': phase_name_str, 'start_day': start_day_val, 'end_day': end_day_val, 
                                        'avg_rvr': None, 'median_rvr': None, 
                                        'avg_expected_return': None, 'avg_variance': None, 'event_count': 0}
                phase_summaries_list.append(phase_stats_dict)
            phase_summary_df = pl.DataFrame(phase_summaries_list)
            
            # print("\nRVR by Phase (Simple Bias):")
            # for r in phase_summary_df.to_dicts():
            #     print(f"  Phase: {r['phase']} ({r['start_day']}-{r['end_day']}), Avg RVR: {r.get('avg_rvr',0):.4f}, Count: {r.get('event_count',0)}")

            try:
                os.makedirs(results_dir, exist_ok=True)
                rvr_pd_plot = rvr_daily_df.to_pandas()
                fig, ax = plt.subplots(figsize=(12, 7))
                ax.plot(rvr_pd_plot['days_to_event'], rvr_pd_plot['avg_rvr'], color='blue', linewidth=2, label='Avg RVR (Simple Bias)')
                ax.axvline(x=0, color='red', linestyle='--', label='Event Day')
                ax.axvline(x=post_event_delta, color='purple', linestyle=':', label=f'End Post-Event Rising (Day {post_event_delta})')
                # Highlight post_event_rising phase (0 to post_event_delta)
                ax.axvspan(phases_dict['post_event_rising'][0], phases_dict['post_event_rising'][1], color='yellow', alpha=0.2, label='Post-Event Rising Phase')
                
                valid_rvr_plot = rvr_pd_plot['avg_rvr'].dropna()
                if not valid_rvr_plot.empty:
                    y_abs_max = abs(valid_rvr_plot).max() * 1.1
                    y_lim_final_rvr = [-y_abs_max, y_abs_max]
                else:
                    y_lim_final_rvr = [-1,1]
                
                y_lim_final_rvr = [np.clip(y_lim_final_rvr[0], -200, 0), np.clip(y_lim_final_rvr[1], 0, 200)] 
                if y_lim_final_rvr[0] == 0 and y_lim_final_rvr[1] == 0 : y_lim_final_rvr = [-1,1] 
                ax.set_ylim(y_lim_final_rvr)

                ax.set_title(f'Return-to-Variance Ratio (Simple Bias) - Lookback: {lookback_window} days')
                ax.set_xlabel('Days Relative to Event'); ax.set_ylabel('Average RVR')
                ax.legend(loc='best'); ax.grid(True, linestyle=':', alpha=0.7)
                plot_filename = os.path.join(results_dir, f"{file_prefix}_rvr_timeseries_simple_bias.png")
                plt.savefig(plot_filename, dpi=200, bbox_inches='tight'); plt.close(fig)
                # print(f"Saved RVR (simple bias) time series plot to: {plot_filename}")
                if rvr_daily_df is not None: rvr_daily_df.write_csv(os.path.join(results_dir, f"{file_prefix}_rvr_daily_simple_bias.csv"))
                phase_summary_df.write_csv(os.path.join(results_dir, f"{file_prefix}_rvr_phase_summary_simple_bias.csv"))
            except Exception as e:
                print(f"Error creating RVR (simple bias) plot: {e}"); traceback.print_exc()
            
            del analysis_data_rvr, phase_summary_df; gc.collect()
            return rvr_daily_df

    def analyze_three_phase_volatility(self,
                                    results_dir: str,
                                    file_prefix: str = "event",
                                    return_col: str = 'ret',
                                    analysis_window: Tuple[int, int] = (-30, 30),
                                    garch_type: str = 'gjr',
                                    k1: float = 1.5,
                                    k2: float = 2.0,
                                    delta_t1: float = 5.0,
                                    delta_t2: float = 3.0,
                                    delta_t3: float = 10.0,
                                    delta: int = 5): # This delta is for ThreePhaseVolatilityModel definition
        # print(f"\n--- Analyzing Three-Phase Volatility (GARCH Type: {garch_type.upper()}) ---")
        if self.data is None or return_col not in self.data.columns:
            print("Error: Data not loaded or return column missing for 3-phase vol.")
            return None, None # Return two Nones as original

        # Extend window for GARCH fitting if necessary (needs sufficient history)
        extended_window_start = min(analysis_window[0], -60) 
        extended_window_end = max(analysis_window[1], 60) # Ensure some post-event data for decay if analysis_window is short
        
        proc_analysis_data = self.data.filter(
            (pl.col('days_to_event') >= extended_window_start) & 
            (pl.col('days_to_event') <= extended_window_end)
        ).sort(['event_id', 'days_to_event'])
        
        if proc_analysis_data.is_empty():
            print(f"Error: No data in extended window [{extended_window_start}, {extended_window_end}] for 3-phase vol.")
            return None, None

        event_ids = proc_analysis_data.get_column('event_id').unique().to_list()
        # print(f"Fitting GARCH models for {len(event_ids)} events for 3-phase volatility...")
        
        sample_event_ids = np.random.choice(event_ids, size=min(5, len(event_ids)), replace=False) if len(event_ids) > 0 else []
        all_events_results_list = []
        sample_events_data_list = []
        # analysis_days_np defines the common set of days for which aggregated volatility will be calculated.
        analysis_days_np = np.array(range(analysis_window[0], analysis_window[1]+1))

        for event_id_val in event_ids:
            event_data_current = proc_analysis_data.filter(pl.col('event_id') == event_id_val)
            # Ensure event_returns_current is a Polars Series for GARCH model
            event_returns_current_series = event_data_current.get_column(return_col)
            
            if event_returns_current_series.len() < 20: # Minimum data points for GARCH fit
                # print(f"Skipping event {event_id_val}: less than 20 data points ({event_returns_current_series.len()}).")
                continue
            
            try:
                garch_model_current: Union[GARCHModel, GJRGARCHModel]
                if garch_type.lower() == 'gjr':
                    garch_model_current = GJRGARCHModel()
                else:
                    garch_model_current = GARCHModel()
                
                garch_model_current.fit(event_returns_current_series) # Pass Polars Series
                
                # Create ThreePhaseVolatilityModel instance (Deterministic version)
                vol_model_current = ThreePhaseVolatilityModel(
                    baseline_model=garch_model_current, k1=k1, k2=k2, 
                    delta_t1=delta_t1, delta_t2=delta_t2, delta_t3=delta_t3, delta=delta
                )

                # Get baseline conditional volatility (sqrt(h_t)) from GARCH
                # This corresponds to the days where GARCH was fitted.
                garch_fitted_days = event_data_current.get_column('days_to_event').to_numpy()
                baseline_cond_vol_sqrt_h_t = np.sqrt(garch_model_current.variance_history)
                
                # Create a map from day_in_garch_fit to its baseline_cond_vol
                day_to_baseline_vol_map = dict(zip(garch_fitted_days, baseline_cond_vol_sqrt_h_t))
                
                # Determine fallback unconditional volatility if a day in analysis_days_np is not in garch_fitted_days
                bm = garch_model_current
                denom_uncond = 0.0
                if isinstance(bm, GJRGARCHModel): denom_uncond = (1 - bm.alpha - bm.beta - 0.5 * bm.gamma)
                else: denom_uncond = (1 - bm.alpha - bm.beta)
                
                fallback_uncond_vol_val = np.sqrt(1e-6) # Default small positive volatility
                if denom_uncond > 1e-7: 
                    fallback_uncond_vol_val = np.sqrt(max(bm.omega / denom_uncond, 1e-7))
                elif bm.variance_history is not None and len(bm.variance_history) > 0:
                     fallback_uncond_vol_val = np.sqrt(bm.variance_history[-1])

                # Align GARCH baseline volatility to the common analysis_days_np
                aligned_baseline_cond_vol_series = np.array(
                    [day_to_baseline_vol_map.get(day, fallback_uncond_vol_val) for day in analysis_days_np]
                )
                
                # Calculate three-phase volatility for the common analysis_days_np
                predicted_vol_series = vol_model_current.calculate_volatility_series(
                    analysis_days_np, # Days relative to event
                    aligned_baseline_cond_vol_series # Baseline sigma_e0 for these days
                )
                all_events_results_list.append({
                    'event_id': event_id_val, 
                    'days_to_event': analysis_days_np, # Common days
                    'predicted_volatility': predicted_vol_series # Vol for common days
                })

                if event_id_val in sample_event_ids:
                    # For sample plots, use the actual days of the event for GARCH vol
                    # and calculate three-phase vol for those specific days
                    
                    # GARCH vol already available as baseline_cond_vol_sqrt_h_t for garch_fitted_days
                    # Three-phase vol for the sample plot (using its own days and GARCH vol)
                    sample_three_phase_vol = vol_model_current.calculate_volatility_series(
                        garch_fitted_days, # Event's own days
                        baseline_cond_vol_sqrt_h_t # Event's own GARCH vol
                    )
                    sample_events_data_list.append({
                        'event_id': event_id_val, 
                        'garch_days': garch_fitted_days, 
                        'garch_vol': baseline_cond_vol_sqrt_h_t, # This is sigma_e0 for these days
                        'three_phase_days': garch_fitted_days, # Same days for direct comparison
                        'three_phase_vol': sample_three_phase_vol, 
                        'returns': event_returns_current_series.to_numpy()[:len(garch_fitted_days)], # Match length
                        'ticker': event_data_current.get_column('ticker').head(1).item(0,0)
                    })
            except Exception as e:
                # print(f"Error processing event {event_id_val} for 3-phase vol: {e}")
                # traceback.print_exc() # Optional: for more detailed debugging
                pass
            
        if not all_events_results_list:
            print("No valid 3-phase volatility results to analyze after GARCH fitting.")
            return None, None

        # Aggregate results across all events for each day in analysis_days_np
        aggregated_rows_list = []
        for day_idx, day_val_common in enumerate(analysis_days_np):
            day_vols_list = [res['predicted_volatility'][day_idx] for res in all_events_results_list if day_idx < len(res['predicted_volatility'])]
            
            if day_vols_list:
                aggregated_rows_list.append({
                    'days_to_event': day_val_common,
                    'avg_volatility': np.mean(day_vols_list),
                    'median_volatility': np.median(day_vols_list),
                    'std_volatility': np.std(day_vols_list),
                    'count': len(day_vols_list) # Number of events contributing to this day's average
                })
        
        volatility_df_agg = pl.DataFrame(aggregated_rows_list)
        if volatility_df_agg.is_empty():
            print("Aggregated 3-phase volatility data is empty.")
            return None, None
        volatility_df_agg = volatility_df_agg.sort('days_to_event')

        # Calculate stats per phase based on the aggregated daily volatilities
        phases_def = {'pre_event': (analysis_window[0], -1), 
                      'event_day': (0, 0), # Added event_day explicitly
                      'post_event_rising': (1, delta), 
                      'post_event_decay': (delta+1, analysis_window[1])}
        
        phase_stats_list_agg = []
        for phase_name, (start_day, end_day) in phases_def.items():
            phase_data_df_agg = volatility_df_agg.filter(
                (pl.col('days_to_event') >= start_day) & (pl.col('days_to_event') <= end_day)
            )
            if not phase_data_df_agg.is_empty():
                avg_vol_phase = phase_data_df_agg['avg_volatility'].mean()
                median_vol_phase = phase_data_df_agg['median_volatility'].mean() # Mean of medians
                avg_event_count_phase = phase_data_df_agg['count'].mean()
                phase_stats_list_agg.append({
                    'phase': phase_name, 'start_day': start_day, 'end_day': end_day,
                    'avg_volatility': avg_vol_phase, 
                    'median_volatility': median_vol_phase,
                    'avg_event_count': avg_event_count_phase
                })
            else: # Ensure all phases appear in output
                 phase_stats_list_agg.append({
                    'phase': phase_name, 'start_day': start_day, 'end_day': end_day,
                    'avg_volatility': None, 'median_volatility': None, 'avg_event_count': 0
                })

        phase_stats_df_agg = pl.DataFrame(phase_stats_list_agg)
        
        try:
            os.makedirs(results_dir, exist_ok=True)
            volatility_df_agg.write_csv(os.path.join(results_dir, f"{file_prefix}_three_phase_volatility_aggregated.csv"))
            phase_stats_df_agg.write_csv(os.path.join(results_dir, f"{file_prefix}_three_phase_volatility_phase_stats.csv"))
            # print(f"Saved 3-phase volatility aggregated analysis to {results_dir}")

            # Plotting aggregated results
            volatility_pd_plot = volatility_df_agg.to_pandas()
            fig, ax = plt.subplots(figsize=(12, 7))
            # Annualize for plotting if not already (model output is typically daily std dev)
            ax.plot(volatility_pd_plot['days_to_event'], volatility_pd_plot['avg_volatility'] * np.sqrt(252) * 100, 
                    color='blue', linewidth=2, label='Average Modeled Volatility (Annualized %)')
            ax.axvline(x=0, color='red', linestyle='--', label='Event Day')
            ax.axvline(x=delta, color='green', linestyle=':', label=f'End Rising Phase (Day {delta})')
            ax.axvspan(1, delta, color='yellow', alpha=0.2, label='Post-Event Rising Phase') # Rising is day 1 to delta
            ax.set_title(f'Three-Phase Volatility Around Events ({garch_type.upper()}-GARCH Based)')
            ax.set_xlabel('Days Relative to Event'); ax.set_ylabel('Volatility (Annualized Std Dev %)')
            ax.legend(loc='best'); ax.grid(True, linestyle=':', alpha=0.7); ax.set_ylim(bottom=0)
            plt.tight_layout()
            plt.savefig(os.path.join(results_dir, f"{file_prefix}_three_phase_volatility_aggregated.png"), dpi=200)
            plt.close(fig)

            # Plotting sample event volatilities
            for sample_data_item in sample_events_data_list:
                if (sample_data_item['garch_vol'] is not None and 
                    len(sample_data_item['garch_vol']) > 0 and 
                    not np.all(np.isnan(sample_data_item['garch_vol']))): 
                    
                    fig_s, ax_s = plt.subplots(figsize=(12, 7))
                    # Annualize GARCH and Three-Phase volatilities for plotting
                    ax_s.plot(sample_data_item['garch_days'], sample_data_item['garch_vol'] * np.sqrt(252) * 100, 
                              color='blue', linewidth=1, alpha=0.7, label='GARCH Volatility (Annualized %)') 
                    ax_s.plot(sample_data_item['three_phase_days'], sample_data_item['three_phase_vol'] * np.sqrt(252) * 100, 
                              color='red', linewidth=2, label='Three-Phase Model Volatility (Annualized %)') 
                    
                    ax_s.axvline(x=0, color='black', linestyle='--', label='Event Day')
                    ax_s.axvline(x=delta, color='green', linestyle=':', label=f'End Rising Phase (Day {delta})')
                    ax_s.set_title(f"Volatility for {sample_data_item['ticker']} (Event ID: {sample_data_item['event_id']})")
                    ax_s.set_xlabel('Days Relative to Event'); ax_s.set_ylabel('Annualized Volatility (%)')
                    ax_s.legend(loc='best'); ax_s.grid(True, linestyle=':', alpha=0.7); ax_s.set_ylim(bottom=0)
                    plt.tight_layout()
                    plt.savefig(os.path.join(results_dir, f"{file_prefix}_volatility_sample_{sample_data_item['event_id']}.png"), dpi=200)
                    plt.close(fig_s)
            # print(f"Saved 3-phase volatility plots to {results_dir}")
        except Exception as e:
            print(f"Error plotting/saving 3-phase volatility: {e}"); traceback.print_exc()
        
        return volatility_df_agg, phase_stats_df_agg

    # Optional method for event-specific kappa to introduce heterogeneity
    # def get_event_specific_kappa(self, event_id: str, base_kappa: float = 2.0) -> float:
    #     """
    #     Create event-specific kappa based on event characteristics (e.g., hash of ID).
    #     This adds natural heterogeneity without relying on random noise generation during the run.
    #     """
    #     # Use hash of event_id to create a deterministic but varied kappa for each event
    #     hash_value = hash(event_id) % 100  # Get a value between 0 and 99
    #     # Example: Scale kappa between 0.8x and 1.2x of the base_kappa
    #     # Adjust scaling factor range as needed.
    #     scaling_factor = 0.8 + (hash_value / 99.0) * 0.4 # Ensure division by 99.0 for float result
    #     return base_kappa * scaling_factor

    def calculate_endogenous_bias(self, day: int, sigma_e_t: float, sqrt_h_t: float,
                                  b0: float, kappa: float, k2: float, delta: int) -> float:
        """
        Calculate endogenous bias according to paper's Assumption 5:
        b_t = b_0 * (1 + κ * ( (σ_e(t)/√h_t) - 1) / (k_2 - 1) )
        This applies during the post-event rising phase (0 < day <= delta).
        Outside this phase, bias is b_0.
        
        Parameters:
        - day: days relative to event (t - t_event). Event day is 0.
        - sigma_e_t: total volatility at time t (from ThreePhaseVolatilityModel, daily std dev).
        - sqrt_h_t: baseline GARCH volatility (square root of h_t, daily std dev).
        - b0: baseline bias parameter (decimal form, e.g., 0.01 for 1%).
        - kappa: bias sensitivity to volatility changes.
        - k2: post-event volatility peak multiplier (from ThreePhaseVolatilityModel params).
        - delta: duration of the post-event rising phase (days 1 to delta).
        """
        if 0 < day <= delta:  # Post-event rising phase: t_event < t <= t_event + δ
            if sqrt_h_t < 1e-8: # Avoid division by zero if baseline GARCH vol is effectively zero
                sqrt_h_t = 1e-8 # Use a very small positive number
            
            # The term (k2 - 1) represents the peak increase factor over baseline due to the event model.
            # If k2 = 1, it means no peak increase, so the adjustment part should be zero.
            if abs(k2 - 1.0) < 1e-8: # k2 is essentially 1
                adjustment_term = 0.0
            else:
                volatility_ratio = sigma_e_t / sqrt_h_t
                # ( (σ_e(t)/√h_t) - 1) is the current modeled increase over baseline.
                # (k_2 - 1) is the max modeled increase over baseline at peak of rising phase.
                # The ratio of these gives a normalized measure of how far into the peak we are.
                adjustment_term = kappa * ( (volatility_ratio - 1.0) / (k2 - 1.0) )
            
            return b0 * (1.0 + adjustment_term)
        else:
            # Outside the post-event rising phase (pre-event, event day, or post-decay), bias is baseline b0.
            return b0

    def analyze_rvr_with_optimistic_bias(self, # Renamed from "optimistic" to "endogenous" in spirit
                                       results_dir: str,
                                       file_prefix: str = "event",
                                       return_col: str = 'ret',
                                       analysis_window: Tuple[int, int] = (-30, 30),
                                       garch_type: str = 'gjr',
                                       k1: float = 1.5,    # Param for ThreePhaseModel
                                       k2: float = 2.0,    # Param for ThreePhaseModel AND Endogenous Bias calc
                                       delta_t1: float = 5.0, # Param for ThreePhaseModel
                                       delta_t2: float = 3.0, # Param for ThreePhaseModel
                                       delta_t3: float = 10.0,# Param for ThreePhaseModel
                                       delta: int = 5,     # Param for ThreePhaseModel AND Endogenous Bias (duration of rising phase)
                                       b0: float = 0.01,   # Baseline bias (decimal) for Endogenous Bias calc
                                       kappa: float = 2.0, # Bias sensitivity (kappa) for Endogenous Bias calc
                                       risk_free_rate: float = 0.0, # Daily risk-free rate (decimal)
                                       tau_b: float = 0.001, # Transaction cost for buying (decimal)
                                       tau_s: float = 0.0005):# Transaction cost for selling (decimal)
        # print(f"\n--- Analyzing Return-to-Variance Ratio with Endogenous Bias & Costs ---")
        # print(f"Analysis Window: {analysis_window}, Rising Phase Duration (delta): {delta} days")
        # print(f"Bias Params: b0={b0}, kappa={kappa}, Model k2={k2}. Costs: RF={risk_free_rate*252*100:.2f}% pa, tau_b={tau_b*100:.3f}%, tau_s={tau_s*100:.3f}%")

        if self.data is None or return_col not in self.data.columns:
            print("Error: Data not loaded or return column missing for RVR Endogenous Bias.")
            return None, None 

        # Define phases for reporting H1 (RVR peaks in post-event rising)
        # Rising phase for H1 test is day 1 to delta (consistent with bias calculation)
        phases_h1 = {'pre_event': (analysis_window[0], -1), 
                  'event_day': (0, 0), # Explicitly separate event day if needed for other analysis
                  'post_event_rising': (1, delta), 
                  'post_event_decay': (delta + 1, analysis_window[1])}
        
        # Window for GARCH fitting and 3-phase model application needs to be broad enough
        extended_window_start = min(analysis_window[0], -60) 
        extended_window_end = max(analysis_window[1], 60)
        
        proc_analysis_data = self.data.filter(
            (pl.col('days_to_event') >= extended_window_start) & 
            (pl.col('days_to_event') <= extended_window_end)
        ).sort(['event_id', 'days_to_event'])
        
        if proc_analysis_data.is_empty():
            print(f"Error: No data found within extended window [{extended_window_start}, {extended_window_end}] for RVR Endogenous Bias.")
            return None, None

        # Determine if returns are in percentage or decimal and standardize
        sample_returns = proc_analysis_data.select(pl.col(return_col)).sample(n=min(100, proc_analysis_data.height), shuffle=True, seed=42)
        avg_abs_return = sample_returns.select(pl.all().mean().abs()).item(0,0) if sample_returns.height > 0 and sample_returns.width > 0 else 0.0
        returns_in_pct = avg_abs_return > 0.05 # Heuristic
        # print(f"Detected returns format for RVR Endog: {'Percentage form' if returns_in_pct else 'Decimal form'} (avg abs: {avg_abs_return:.4f})")
        
        return_col_decimal = 'decimal_return_rvr_endog' # Standardized name for decimal returns
        if returns_in_pct:
            # print("Converting percentage returns to decimal for RVR Endogenous Bias calculation")
            proc_analysis_data = proc_analysis_data.with_columns((pl.col(return_col) / 100).alias(return_col_decimal))
        else: # Returns are already decimal
            proc_analysis_data = proc_analysis_data.with_columns(pl.col(return_col).alias(return_col_decimal))
        # All parameters b0, kappa, risk_free_rate, tau_b, tau_s are assumed to be in decimal form.
        
        event_ids = proc_analysis_data.get_column('event_id').unique().to_list()
        # print(f"Processing {len(event_ids)} events for RVR Endogenous Bias analysis...")
        all_rvr_data_list_endog = []
        # analysis_days_np_rvr defines the common set of days for which RVR will be calculated and aggregated.
        analysis_days_np_rvr = np.array(range(analysis_window[0], analysis_window[1] + 1))

        for event_id_val_rvr in event_ids:
            event_data_current_rvr = proc_analysis_data.filter(pl.col('event_id') == event_id_val_rvr)
            # Use the standardized decimal return column for GARCH fitting
            event_returns_for_garch_fit_series = event_data_current_rvr.get_column(return_col_decimal)
            
            if event_returns_for_garch_fit_series.len() < 20: # Min data points for GARCH
                continue 
            
            try:
                garch_model_current_rvr: Union[GARCHModel, GJRGARCHModel]
                if garch_type.lower() == 'gjr':
                    garch_model_current_rvr = GJRGARCHModel()
                else:
                    garch_model_current_rvr = GARCHModel()
                
                garch_model_current_rvr.fit(event_returns_for_garch_fit_series) # Fit on decimal returns
                
                # Create ThreePhaseVolatilityModel instance (Deterministic version)
                # k2 and delta here are the model parameters for volatility shape
                vol_model_current_rvr = ThreePhaseVolatilityModel(
                    baseline_model=garch_model_current_rvr, k1=k1, k2=k2, 
                    delta_t1=delta_t1, delta_t2=delta_t2, delta_t3=delta_t3, delta=delta 
                )
                
                # --- Get GARCH baseline (sqrt_h_t) and Three-Phase vol (sigma_e_t) for analysis_days_np_rvr ---
                garch_fitted_days_rvr = event_data_current_rvr.get_column('days_to_event').to_numpy()
                baseline_cond_vol_sqrt_h_t_fitted = np.sqrt(garch_model_current_rvr.variance_history)
                day_to_baseline_vol_map_rvr = dict(zip(garch_fitted_days_rvr, baseline_cond_vol_sqrt_h_t_fitted))

                bm_rvr = garch_model_current_rvr
                denom_uncond_rvr = 0.0
                if isinstance(bm_rvr, GJRGARCHModel): denom_uncond_rvr = (1 - bm_rvr.alpha - bm_rvr.beta - 0.5 * bm_rvr.gamma)
                else: denom_uncond_rvr = (1 - bm_rvr.alpha - bm_rvr.beta)
                
                fallback_uncond_vol_rvr_val = np.sqrt(1e-6)
                if denom_uncond_rvr > 1e-7: 
                    fallback_uncond_vol_rvr_val = np.sqrt(max(bm_rvr.omega / denom_uncond_rvr, 1e-7))
                elif bm_rvr.variance_history is not None and len(bm_rvr.variance_history) > 0:
                     fallback_uncond_vol_rvr_val = np.sqrt(bm_rvr.variance_history[-1])
                
                # sqrt_h_t for each day in analysis_days_np_rvr
                aligned_baseline_cond_vol_for_analysis = np.array(
                    [day_to_baseline_vol_map_rvr.get(day, fallback_uncond_vol_rvr_val) for day in analysis_days_np_rvr]
                )
                # sigma_e_t (ThreePhaseModel vol) for each day in analysis_days_np_rvr
                volatility_series_sigma_e_t_analysis_days = vol_model_current_rvr.calculate_volatility_series(
                    analysis_days_np_rvr, 
                    aligned_baseline_cond_vol_for_analysis
                )

                # Calculate historical mean return for this event (using data within analysis_window)
                event_returns_in_analysis_window_series = event_data_current_rvr.filter(
                    (pl.col('days_to_event') >= analysis_window[0]) & (pl.col('days_to_event') <= analysis_window[1])
                ).get_column(return_col_decimal)
                
                mean_hist_return_event = event_returns_in_analysis_window_series.mean()
                if mean_hist_return_event is None: mean_hist_return_event = 0.0 # Handle all-null case for mean

                # Optional: Use event-specific kappa for heterogeneity
                # current_event_kappa = self.get_event_specific_kappa(event_id_val_rvr, kappa)
                current_event_kappa = kappa # Using global kappa for now

                # --- Calculate RVR components for each day in analysis_days_np_rvr ---
                for day_idx, day_val_rvr in enumerate(analysis_days_np_rvr):
                    sigma_e_t_current_day = volatility_series_sigma_e_t_analysis_days[day_idx]
                    sqrt_h_t_current_day = aligned_baseline_cond_vol_for_analysis[day_idx]
                    
                    # Calculate endogenous bias for the current day
                    # k2 and delta here are passed for the bias formula's logic
                    bias_t_current_day = self.calculate_endogenous_bias(
                        day=day_val_rvr, sigma_e_t=sigma_e_t_current_day, sqrt_h_t=sqrt_h_t_current_day,
                        b0=b0, kappa=current_event_kappa, k2=k2, delta=delta 
                    )
                    
                    # Determine transaction cost for the day
                    tau_i_current_day = tau_s if day_val_rvr > 0 else tau_b # Sell post-event, buy pre-event/event day
                    
                    # Calculate final expected return for RVR numerator
                    # I_t (other event-specific info) assumed 1.0 (or incorporated into mean_hist_return_event)
                    expected_return_final_current_day = mean_hist_return_event + bias_t_current_day - risk_free_rate - tau_i_current_day
                    
                    # Calculate RVR
                    numerator_rvr = expected_return_final_current_day
                    denominator_rvr = sigma_e_t_current_day**2 # Variance is sigma_e_t squared
                    
                    rvr_current_day = np.nan # Default to NaN
                    if denominator_rvr > 1e-10: # Avoid division by zero
                        rvr_current_day = numerator_rvr / denominator_rvr
                        rvr_current_day = np.clip(rvr_current_day, -1000, 1000) # Cap extreme RVR values
                                     
                    all_rvr_data_list_endog.append({
                         'event_id': event_id_val_rvr, 
                         'days_to_event': day_val_rvr,
                         'expected_return_biased': expected_return_final_current_day,
                         'three_phase_volatility': sigma_e_t_current_day, # This is sigma_e_t
                         'rvr': rvr_current_day
                     })
            except Exception as e:
                # print(f"Error processing event {event_id_val_rvr} for RVR Endogenous Bias: {e}")
                # traceback.print_exc() # For debugging
                pass
            
        if not all_rvr_data_list_endog:
            print("No valid RVR (Endogenous Bias) results to analyze after processing events.")
            return None, None

        combined_rvr_df_endog = pl.DataFrame(all_rvr_data_list_endog)
        agg_rvr_df_endog = combined_rvr_df_endog.group_by('days_to_event').agg([
            pl.mean('expected_return_biased').alias('mean_expected_return_biased'),
            pl.mean('three_phase_volatility').alias('mean_three_phase_volatility'), # Avg sigma_e_t
            pl.mean('rvr').alias('mean_rvr'),
            pl.median('rvr').alias('median_rvr'),
            pl.sum(pl.col('rvr').is_not_null()).alias('event_count_rvr') # Count non-null RVRs
        ]).sort('days_to_event')

        # Calculate H1 phase statistics from the aggregated daily RVRs
        phase_stats_rvr_list_endog = []
        for phase_name, (start_day, end_day) in phases_h1.items():
            phase_data_rvr_agg = agg_rvr_df_endog.filter(
                (pl.col('days_to_event') >= start_day) & (pl.col('days_to_event') <= end_day)
            )
            if not phase_data_rvr_agg.is_empty():
                # Take the mean of daily means for the phase summary
                avg_rvr_phase = phase_data_rvr_agg['mean_rvr'].mean()
                median_rvr_phase = phase_data_rvr_agg['median_rvr'].mean() # Mean of daily medians
                avg_vol_phase = phase_data_rvr_agg['mean_three_phase_volatility'].mean()
                avg_exp_ret_phase = phase_data_rvr_agg['mean_expected_return_biased'].mean()
                avg_count_phase = phase_data_rvr_agg['event_count_rvr'].mean()
                phase_stats_rvr_list_endog.append({
                    'phase': phase_name, 'start_day': start_day, 'end_day': end_day,
                    'avg_rvr': avg_rvr_phase, 'median_rvr': median_rvr_phase,
                    'avg_volatility': avg_vol_phase,
                    'avg_expected_return': avg_exp_ret_phase,
                    'avg_event_count': avg_count_phase
                })
            else:
                 phase_stats_rvr_list_endog.append({
                    'phase': phase_name, 'start_day': start_day, 'end_day': end_day,
                    'avg_rvr': None, 'median_rvr': None, 'avg_volatility': None,
                    'avg_expected_return': None, 'avg_event_count': 0
                })
        phase_stats_rvr_df_endog = pl.DataFrame(phase_stats_rvr_list_endog)
        
        # Test Hypothesis 1
        phases_for_h1_test = ['pre_event', 'post_event_rising', 'post_event_decay']
        h1_test_data_values = {}
        for row_dict in phase_stats_rvr_df_endog.filter(pl.col('phase').is_in(phases_for_h1_test)).to_dicts():
            h1_test_data_values[row_dict['phase']] = row_dict['avg_rvr']
        
        h1_result_bool = False
        if all(k in h1_test_data_values and pd.notna(h1_test_data_values[k]) for k in phases_for_h1_test):
            h1_result_bool = (h1_test_data_values['post_event_rising'] > h1_test_data_values['pre_event'] and 
                              h1_test_data_values['post_event_rising'] > h1_test_data_values['post_event_decay'])
        # print(f"\nHypothesis 1 (RVR peaks in post_event_rising phase, days 1-{delta}): {'SUPPORTED' if h1_result_bool else 'NOT SUPPORTED'}")

        try:
            os.makedirs(results_dir, exist_ok=True)
            agg_rvr_df_endog.write_csv(os.path.join(results_dir, f"{file_prefix}_rvr_endogenous_bias_timeseries.csv"))
            phase_stats_rvr_df_endog.write_csv(os.path.join(results_dir, f"{file_prefix}_rvr_endogenous_bias_phase_stats.csv"))
            
            h1_df_output_data = {
                'hypothesis': ['H1: RVR peaks during post-event rising phase'], 
                'result': [h1_result_bool],
                'pre_event_rvr': [h1_test_data_values.get('pre_event', None)],
                'post_rising_rvr': [h1_test_data_values.get('post_event_rising', None)],
                'post_decay_rvr': [h1_test_data_values.get('post_event_decay', None)]
            }
            h1_df_output = pl.DataFrame(h1_df_output_data)
            h1_df_output.write_csv(os.path.join(results_dir, f"{file_prefix}_hypothesis1_test.csv"))
            # print(f"Saved RVR (Endogenous Bias) analysis files to {results_dir}")
            
            # Plotting aggregated RVR
            rvr_pd_plot_endog = agg_rvr_df_endog.to_pandas()
            fig, ax = plt.subplots(figsize=(12, 7))
            ax.plot(rvr_pd_plot_endog['days_to_event'], rvr_pd_plot_endog['mean_rvr'], 
                    color='blue', linewidth=2, label='Mean RVR (Endogenous Bias)')
            ax.axvline(x=0, color='red', linestyle='--', label='Event Day') 
            ax.axvline(x=delta, color='green', linestyle=':', label=f'End Rising Phase (Day {delta})')
            # Highlight post_event_rising phase (day 1 to delta)
            ax.axvspan(phases_h1['post_event_rising'][0], phases_h1['post_event_rising'][1], 
                       color='yellow', alpha=0.2, label='Post-Event Rising Phase')
            ax.set_title(f'RVR with Endogenous Bias (b0={b0*100:.2f}%, kappa={kappa:.1f}, Model k2={k2}, Model delta={delta})')
            ax.set_xlabel('Days Relative to Event'); ax.set_ylabel('Return-to-Variance Ratio') 
            ax.legend(loc='best'); ax.grid(True, linestyle=':', alpha=0.7)
            plt.tight_layout()
            plt.savefig(os.path.join(results_dir, f"{file_prefix}_rvr_endogenous_bias_timeseries.png"), dpi=200)
            plt.close(fig)
            # print(f"Saved RVR (Endogenous Bias) plot to {results_dir}")
        except Exception as e:
            print(f"Error saving/plotting RVR (Endogenous Bias) analysis: {e}"); traceback.print_exc()
            
        return agg_rvr_df_endog, phase_stats_rvr_df_endog