import polars as pl
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report
from sklearn.impute import SimpleImputer
import xgboost as xgb
import warnings
import os
import traceback # Keep this import
import gc # Keep this import
from typing import List, Optional, Tuple, Dict, Any
import datetime
import pandas as pd # Required for plotting/CSV saving of some outputs

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
    def __init__(self, earnings_path: str, stock_paths: List[str], window_days: int = 30):
        """
        Initialize DataLoader for Earnings events using Polars.

        Parameters:
        earnings_path (str): Path to the event data CSV. Must contain ticker and announcement date.
        stock_paths (list): List of paths to stock price/return data PARQUET files.
        window_days (int): Number of days before/after event date.
        """
        self.earnings_path = earnings_path
        if isinstance(stock_paths, str): self.stock_paths = [stock_paths]
        elif isinstance(stock_paths, list): self.stock_paths = stock_paths
        else: raise TypeError("stock_paths must be a string or a list of Parquet file paths.")
        self.window_days = window_days
        # Add chunk size parameter for event processing
        self.event_chunk_size = 5000 # Adjust based on memory capacity

    def _load_single_stock_parquet(self, stock_path: str) -> Optional[pl.DataFrame]:
        """Load and process a single stock data PARQUET file using Polars."""
        # --- THIS METHOD IS UNCHANGED FROM THE PREVIOUS CORRECTION ---
        # --- It handles standardizing columns and types for one stock file ---
        # (Handles eager loading standardization - not used directly in the loop anymore)
        try:
            # print(f"  Reading Parquet file: {stock_path}") # Reduce verbosity inside chunk loop
            stock_data = pl.read_parquet(stock_path)
            # print(f"  Read {len(stock_data)} rows from {stock_path}") # Reduce verbosity

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
                 # print(f"Warning: Skipping {stock_path} as essential columns (date, ticker, prc, ret) not found after checking variations.")
                 return None # Skip this file if essentials are missing

            stock_data = stock_data.select(selected_cols)
            if rename_dict:
                stock_data = stock_data.rename(rename_dict)

            # --- Data Type and Existence Checks (on standardized names) ---
            # Note: Essential columns check already done implicitly above
            required_cols = ['date', 'ticker', 'prc', 'ret', 'vol']
            missing_cols = [col for col in required_cols if col not in stock_data.columns]
            if missing_cols:
                 essential_pr = ['prc', 'ret'] # Re-check just in case logic missed something
                 if any(col in missing_cols for col in essential_pr):
                      raise ValueError(f"Essential columns {missing_cols} missing in {stock_path} AFTER selection/rename.")
                 else:
                      # print(f"  Warning: Missing optional columns in {stock_path}: {missing_cols}. Some features might be skipped.")
                      pass # Keep less verbose

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
                         pass # Keep less verbose
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
             print(f"Warning: Parquet file not found: {stock_path}")
             return None
        except Exception as e:
            print(f"Warning: Error processing Parquet file {stock_path}: {e}")
            # traceback.print_exc() # Uncomment for full trace
            return None

    def load_data(self) -> Optional[pl.DataFrame]:
        """
        Load earnings event dates (CSV) and stock data (PARQUET) using Polars,
        processing events in chunks to manage memory. Includes debugging prints.
        Explicitly uses 'ANNDATS' as the announcement date column.
        """
        # --- Load ALL Unique Earnings Event Dates First ---
        try:
            print(f"Loading earnings event dates from: {self.earnings_path} (CSV)")
            event_df_peek = pl.read_csv_batched(self.earnings_path, batch_size=1).next_batches(1)[0]
            ticker_col = next((c for c in ['TICKER', 'ticker', 'Ticker', 'symbol', 'tic'] if c in event_df_peek.columns), None)
            if not ticker_col: raise ValueError("Missing Ticker column in event file.")
            date_col = 'ANNDATS'
            if date_col not in event_df_peek.columns: raise ValueError(f"Required announcement date column '{date_col}' not found.")

            print(f"Using columns '{ticker_col}' (as ticker) and '{date_col}' (as Announcement Date) from event file.")
            event_data_raw = pl.read_csv(self.earnings_path, columns=[ticker_col, date_col], try_parse_dates=True)
            event_data_renamed = event_data_raw.rename({ticker_col: 'ticker', date_col: 'Announcement Date'})

            # --- Check/Correct Date Type ---
            if event_data_renamed['Announcement Date'].dtype == pl.Object or isinstance(event_data_renamed['Announcement Date'].dtype, pl.String):
                print("    'Announcement Date' read as Object/String, attempting str.to_datetime...")
                event_data_processed = event_data_renamed.with_columns([
                     pl.col('Announcement Date').str.to_datetime(strict=False).cast(pl.Datetime), # Explicit parse and cast
                     pl.col('ticker').cast(pl.Utf8).str.to_uppercase()
                 ])
            elif isinstance(event_data_renamed['Announcement Date'].dtype, (pl.Date, pl.Datetime)):
                 print("    'Announcement Date' already parsed as Date/Datetime.")
                 event_data_processed = event_data_renamed.with_columns([
                     pl.col('Announcement Date').cast(pl.Datetime), # Ensure it's Datetime specifically if needed
                     pl.col('ticker').cast(pl.Utf8).str.to_uppercase()
                 ])
            else:
                 raise TypeError(f"Unexpected dtype for 'Announcement Date': {event_data_renamed['Announcement Date'].dtype}")
            # --- End Corrected Date Handling ---

            event_data_processed = event_data_processed.drop_nulls(subset=['Announcement Date'])
            earnings_events = event_data_processed.unique(subset=['ticker', 'Announcement Date'], keep='first')
            n_total_events = earnings_events.height

            print("\n--- Sample Parsed Earnings Events ---")
            print(earnings_events.head(5))
            print("-" * 35 + "\n")

            print(f"Found {n_total_events} unique earnings events (Ticker-Date pairs).")
            if earnings_events.is_empty(): raise ValueError("No valid earnings events found.")

        except FileNotFoundError: raise FileNotFoundError(f"Earnings event file not found: {self.earnings_path}")
        except Exception as e: raise ValueError(f"Error processing earnings event file {self.earnings_path}: {e}")

        # --- Process Events in Chunks ---
        processed_chunks = []
        num_chunks = (n_total_events + self.event_chunk_size - 1) // self.event_chunk_size
        print(f"Processing events in {num_chunks} chunk(s) of size {self.event_chunk_size}...")

        for i in range(num_chunks):
            start_idx = i * self.event_chunk_size
            end_idx = min((i + 1) * self.event_chunk_size, n_total_events)
            event_chunk = earnings_events.slice(start_idx, end_idx - start_idx)
            print(f"--- Processing event chunk {i+1}/{num_chunks} ({event_chunk.height} events) ---")

            chunk_tickers = event_chunk['ticker'].unique()
            min_ann_date = event_chunk['Announcement Date'].min()
            max_ann_date = event_chunk['Announcement Date'].max()
            buffer = pl.duration(days=self.window_days + 1) # Add buffer for safety
            required_min_date = min_ann_date - buffer
            required_max_date = max_ann_date + buffer

            print(f"    Chunk Tickers: {chunk_tickers.len()} (Sample: {chunk_tickers[:5].to_list()})")
            print(f"    Required Stock Date Range: {required_min_date} to {required_max_date}")

            stock_scans = []
            failed_stock_loads = 0
            print("    Scanning and filtering stock Parquet files (lazily)...")
            for stock_path in self.stock_paths:
                try:
                    scan = pl.scan_parquet(stock_path)
                    # --- Standardization (Lazy) ---
                    original_columns = list(scan.schema.keys())
                    col_map_lower = {col.lower(): col for col in original_columns}
                    standard_names = { # Keep consistent
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
                        print(f"    Warning: Skipping {stock_path} as essential date or ticker column mapping not found.")
                        continue # Skip file if essential keys aren't mapped

                    scan = scan.select(selected_orig_cols)
                    if rename_dict: scan = scan.rename(rename_dict)

                    # --- Lazy Type Conversion (Corrected) ---
                    date_dtype = scan.schema.get('date')
                    ticker_dtype = scan.schema.get('ticker')
                    if date_dtype is None or ticker_dtype is None:
                         print(f"    Warning: 'date' or 'ticker' column missing in {stock_path} schema after selection/rename. Skipping.")
                         continue

                    scan_expressions = []
                    if date_dtype == pl.Object or isinstance(date_dtype, pl.String):
                        scan_expressions.append(pl.col("date").str.to_datetime(strict=False).cast(pl.Datetime))
                    else:
                        scan_expressions.append(pl.col("date").cast(pl.Datetime))

                    if ticker_dtype != pl.Utf8:
                         scan_expressions.append(pl.col("ticker").cast(pl.Utf8).str.to_uppercase())
                    else:
                         scan_expressions.append(pl.col("ticker").str.to_uppercase())

                    scan = scan.with_columns(scan_expressions)

                    # <<< --- Filtering Step --- >>>
                    filtered_scan = scan.filter(
                        pl.col('ticker').is_in(chunk_tickers) &
                        pl.col('date').is_between(required_min_date, required_max_date, closed='both')
                    )
                    stock_scans.append(filtered_scan)
                except Exception as e:
                    print(f"    Warning: Failed to scan/prepare stock file {stock_path}: {e}")
                    failed_stock_loads += 1

            if not stock_scans:
                print(f"    ERROR: No stock data could be scanned for chunk {i+1}. Skipping chunk.")
                continue

            # Concatenate scans
            combined_stock_scan = pl.concat(stock_scans, how='vertical_relaxed')

            # --- Collect Actual Stock Data ---
            print(f"    Collecting filtered stock data for chunk {i+1}...")
            try:
                stock_data_chunk = combined_stock_scan.collect(streaming=True)
            except Exception as e:
                 print(f"    ERROR collecting stock data for chunk {i+1}: {e}. Skipping chunk.")
                 traceback.print_exc()
                 continue

            print(f"    Collected {stock_data_chunk.height} stock rows.")
            if stock_data_chunk.is_empty():
                 print("    Collected stock data is empty. Skipping rest of chunk processing.")
                 continue # Skip to next chunk if no stock data found

            # --- Standardize Types (Post-Collect) and Deduplicate ---
            cast_expressions_collect = []
            numeric_cols_to_check = ['prc', 'ret', 'vol', 'openprc', 'askhi', 'bidlo', 'shrout']
            for col in numeric_cols_to_check:
                 if col in stock_data_chunk.columns:
                      cast_expressions_collect.append(
                           pl.when(pl.col(col).cast(pl.Float64, strict=False).is_infinite())
                           .then(None).otherwise(pl.col(col).cast(pl.Float64, strict=False))
                           .alias(col)
                      )
            if cast_expressions_collect:
                stock_data_chunk = stock_data_chunk.with_columns(cast_expressions_collect)

            # Ensure join keys are correct type before unique/join
            stock_data_chunk = stock_data_chunk.with_columns([
                pl.col('ticker').cast(pl.Utf8),
                pl.col('date').cast(pl.Datetime)
            ])
            stock_data_chunk = stock_data_chunk.unique(subset=['date', 'ticker'], keep='first', maintain_order=False)
            print(f"    Deduplicated stock rows: {stock_data_chunk.height}")
            if stock_data_chunk.is_empty():
                 print(f"    Warning: No stock data remained after deduplication for chunk {i+1}. Skipping chunk.")
                 continue

            # --- Merge event chunk with stock data chunk ---
            print(f"    Merging events with stock data...")
            # Ensure join columns are same type
            event_chunk = event_chunk.with_columns([
                pl.col('ticker').cast(pl.Utf8),
                pl.col('Announcement Date').cast(pl.Datetime)
            ])
            merged_chunk = stock_data_chunk.join(
                event_chunk, on='ticker', how='inner'
            )
            print(f"    Merged chunk rows: {merged_chunk.height}")
            if merged_chunk.is_empty():
                print(f"    Warning: Merge resulted in empty data for chunk {i+1}. Check ticker matching.")
                continue # Skip chunk


            # --- Calculate relative days and filter window FOR THE CHUNK ---
            processed_chunk = merged_chunk.with_columns(
                (pl.col('date') - pl.col('Announcement Date')).dt.total_days().cast(pl.Int32).alias('days_to_announcement')
            ).filter(
                (pl.col('days_to_announcement') >= -self.window_days) &
                (pl.col('days_to_announcement') <= self.window_days)
            )
            print(f"    Rows after window filter ({self.window_days} days): {processed_chunk.height}")

            if processed_chunk.is_empty():
                print(f"    Warning: No data found within event window for chunk {i+1}.")
                continue

            # --- Add final identifiers ---
            processed_chunk = processed_chunk.with_columns([
                (pl.col('days_to_announcement') == 0).cast(pl.Int8).alias('is_announcement_date'),
                (pl.col("ticker") + "_" + pl.col("Announcement Date").dt.strftime('%Y%m%d')).alias('event_id')
            ])

            # Select necessary columns
            stock_cols = stock_data_chunk.columns
            event_cols = event_chunk.columns
            derived_cols = ['days_to_announcement', 'is_announcement_date', 'event_id']
            final_cols = list(set(stock_cols) | set(event_cols) | set(derived_cols))
            # Ensure columns exist in the final processed chunk before selecting
            final_cols = [c for c in final_cols if c in processed_chunk.columns]
            processed_chunk = processed_chunk.select(final_cols)

            print(f"    Processed chunk {i+1} FINAL shape: {processed_chunk.shape}")
            processed_chunks.append(processed_chunk)
            print(f"--- Finished processing chunk {i+1} ---")

            del stock_data_chunk, merged_chunk, event_chunk, processed_chunk, combined_stock_scan, stock_scans
            gc.collect()


        # --- Final Concatenation ---
        if not processed_chunks:
            print("Error: No data chunks were processed successfully.")
            return None # Return None if no chunks succeeded

        print("\nConcatenating processed chunks...")
        combined_data = pl.concat(processed_chunks, how='vertical').sort(['ticker', 'Announcement Date', 'date'])
        print(f"Final Earnings dataset shape: {combined_data.shape}")
        mem_usage_mb = combined_data.estimated_size("mb")
        print(f"Final DataFrame memory usage: {mem_usage_mb:.2f} MB")

        if combined_data.is_empty():
             print("Warning: Final combined data is empty after chunk processing.")
             return None

        return combined_data


# --- FeatureEngineer Class ---
class FeatureEngineer:
    def __init__(self, prediction_window: int = 3):
        self.windows = [5, 10, 20]
        self.prediction_window = prediction_window
        self.imputer = SimpleImputer(strategy='median') # Operates on NumPy
        self.feature_names: List[str] = []
        self.final_feature_names: List[str] = [] # After potential imputation/selection
        self.categorical_features: List[str] = [] # Store names of created categorical/dummy features
        self.sector_dummies_cols: Optional[List[str]] = None
        self.industry_dummies_cols: Optional[List[str]] = None
        self.top_industries: Optional[List[str]] = None
        self._imputer_fitted = False # Track if imputer has been fitted

    def create_target(self, df: pl.DataFrame, price_col: str = 'prc') -> pl.DataFrame:
        """Create target variable using Polars."""
        print(f"Creating target 'future_ret' (window: {self.prediction_window} days)...")
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
            .otherwise(None) # Set to null if price is 0 or null
            .alias('future_ret')
        ).drop('future_price')

        print(f"'future_ret' created. Non-null: {df.filter(pl.col('future_ret').is_not_null()).height}")
        return df

    def calculate_features(self, df: pl.DataFrame, price_col: str = 'prc', return_col: str = 'ret',
                           volume_col: str = 'vol', fit_categorical: bool = False) -> pl.DataFrame:
        """Calculate features for Earnings analysis using Polars. Robust to missing optional columns."""
        print("Calculating Earnings features (Polars)...")
        required = ['event_id', price_col, return_col, 'Announcement Date', 'date', 'days_to_announcement']
        missing = [col for col in required if col not in df.columns]
        if missing: raise ValueError(f"Missing required columns for feature calculation: {missing}")

        has_volume = volume_col in df.columns
        if not has_volume: print(f"Info: Volume column '{volume_col}' not found. Volume features skipped.")

        # Ensure sorted within event groups for rolling/shift operations
        df = df.sort(['event_id', 'date'])
        current_features: List[str] = []
        self.categorical_features = [] # Reset

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
        feature_expressions = [] # Reset

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
        feature_expressions = [] # Reset

        feature_expressions.append((pl.col('volatility_5') - pl.col('volatility_10')).alias('delta_volatility_5_10'))
        current_features.append('delta_volatility_5_10')
        feature_expressions.append((pl.col('volatility_10') - pl.col('volatility_20')).alias('delta_volatility_10_20'))
        current_features.append('delta_volatility_10_20')

        # Log returns
        shifted_price_log = pl.col(price_col).shift(1).over('event_id')
        feature_expressions.append(
             pl.when(shifted_price_log.is_not_null() & (shifted_price_log > 0) & pl.col(price_col).is_not_null() & (pl.col(price_col) > 0) )
             .then(pl.ln(pl.col(price_col) / shifted_price_log))
             .otherwise(None)
             .alias('log_ret')
        )
        current_features.append('log_ret')
        current_features.append('days_to_announcement') # Already exists

        for lag in range(1, 4):
            col_name = f'ret_lag_{lag}'
            feature_expressions.append(pl.col(return_col).shift(lag).over('event_id').alias(col_name))
            current_features.append(col_name)

        # Apply features calculated so far
        df = df.with_columns(feature_expressions)
        feature_expressions = [] # Reset for next batch

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
        else: # Add null columns if volume missing
             df = df.with_columns([pl.lit(None).cast(pl.Float64).alias(c) for c in ['norm_vol', 'vol_momentum_5', 'vol_momentum_10']])
             current_features.extend(['norm_vol', 'vol_momentum_5', 'vol_momentum_10'])


        # --- Pre-announcement Return (Requires group_by, agg, join) ---
        pre_announce_start_offset = pl.duration(days=-30)
        pre_announce_data = df.filter(
            (pl.col('date') < pl.col('Announcement Date')) &
            (pl.col('date') >= (pl.col('Announcement Date') + pre_announce_start_offset))
        )
        pre_announce_agg = pre_announce_data.group_by('event_id').agg(
             (pl.col(return_col).fill_null(0) + 1).product().alias("prod_ret_factor")
        ).with_columns(
            (pl.col("prod_ret_factor") - 1).alias('pre_announce_ret_30d')
        ).select(['event_id', 'pre_announce_ret_30d'])

        df = df.join(pre_announce_agg, on='event_id', how='left')
        df = df.with_columns(pl.col('pre_announce_ret_30d').fill_null(0))
        current_features.append('pre_announce_ret_30d')


        # --- Earnings Surprise Features (Conditional) ---
        surprise_cols_to_add = ['surprise_val', 'pos_surprise', 'neg_surprise', 'surprise_magnitude',
                                'prev_surprise', 'consecutive_beat', 'consecutive_miss']
        if 'Surprise' in df.columns:
            print("Info: 'Surprise' column found. Calculating surprise features.")
            df = df.with_columns(
                pl.col('Surprise').cast(pl.Float64, strict=False).fill_null(0).alias('surprise_val')
            )
            df = df.with_columns([
                (pl.col('surprise_val') > 0).cast(pl.Int8).alias('pos_surprise'),
                (pl.col('surprise_val') < 0).cast(pl.Int8).alias('neg_surprise'),
                pl.col('surprise_val').abs().alias('surprise_magnitude'),
                pl.col('surprise_val').shift(1).over('ticker').fill_null(0).alias('prev_surprise')
            ])
            df = df.with_columns([
                 ((pl.col('surprise_val') > 0) & (pl.col('prev_surprise') > 0)).cast(pl.Int8).alias('consecutive_beat'),
                 ((pl.col('surprise_val') < 0) & (pl.col('prev_surprise') < 0)).cast(pl.Int8).alias('consecutive_miss')
            ])
            current_features.extend(surprise_cols_to_add)
        else:
            print("Info: 'Surprise' column not present in data. Skipping surprise features.")
            df = df.with_columns([pl.lit(None).cast(pl.Float64).alias(c) for c in surprise_cols_to_add if c not in ['pos_surprise', 'neg_surprise', 'consecutive_beat', 'consecutive_miss']] +
                                 [pl.lit(None).cast(pl.Int8).alias(c) for c in ['pos_surprise', 'neg_surprise', 'consecutive_beat', 'consecutive_miss']] )
            current_features.extend(surprise_cols_to_add)


        # --- Announcement Time Features (Conditional) ---
        time_features = ['announcement_hour', 'is_bmo', 'is_amc', 'is_market_hours']
        if 'Time' in df.columns:
             print("Info: 'Time' column found. Calculating time features.")
             df = df.with_columns(
                 pl.col('Time').str.to_time(strict=False, format="%H:%M:%S").alias('time_parsed')
             )
             if df['time_parsed'].is_null().all():
                 print("  Warning: Failed to parse 'Time' column. Skipping time features.")
                 df = df.drop('time_parsed')
                 df = df.with_columns([pl.lit(None).cast(pl.Int8).alias(c) for c in time_features])
             else:
                 df = df.with_columns(
                     pl.col('time_parsed').dt.hour().fill_null(-1).cast(pl.Int8).alias('announcement_hour')
                 ).drop('time_parsed')
                 df = df.with_columns([
                      ((pl.col('announcement_hour') >= 0) & (pl.col('announcement_hour') < 9)).cast(pl.Int8).alias('is_bmo'),
                      ((pl.col('announcement_hour') >= 16) & (pl.col('announcement_hour') <= 23)).cast(pl.Int8).alias('is_amc'),
                      ((pl.col('announcement_hour') >= 9) & (pl.col('announcement_hour') < 16)).cast(pl.Int8).alias('is_market_hours')
                 ])
                 current_features.extend(time_features)
        else:
             print("Info: 'Time' column not present in data. Skipping time features.")
             df = df.with_columns([pl.lit(None).cast(pl.Int8).alias(c) for c in time_features])
             current_features.extend(time_features)

        # --- Quarter Features (Conditional) ---
        quarter_features = ['quarter_num'] + [f'is_q{i}' for i in range(1, 5)]
        if 'Quarter' in df.columns:
             print("Info: 'Quarter' column found. Calculating quarter features.")
             df = df.with_columns(
                 pl.col('Quarter').cast(pl.Utf8).str.extract(r"Q(\d)", 1)
                 .cast(pl.Int8, strict=False).alias('quarter_num')
             )
             quarter_expressions = []
             for i in range(1, 5):
                 col_name = f'is_q{i}'
                 quarter_expressions.append(
                     (pl.col('quarter_num') == i).fill_null(False).cast(pl.Int8).alias(col_name)
                 )
                 self.categorical_features.append(col_name)
             df = df.with_columns(quarter_expressions)
             current_features.extend(quarter_features)
        else:
            print("Info: 'Quarter' column not present in data. Skipping quarter features.")
            df = df.with_columns([pl.lit(None).cast(pl.Int8).alias(c) for c in quarter_features])
            current_features.extend(quarter_features)


        # --- Sector/Industry Features (Conditional - Requires Dummies) ---
        # Handle Sector
        if 'Sector' in df.columns:
            print("Info: 'Sector' column found. Processing sector features.")
            df = df.with_columns(pl.col('Sector').fill_null('Unknown').cast(pl.Utf8))
            if fit_categorical:
                sector_dummies_df = df.select('Sector').to_dummies(columns=['Sector'], drop_first=True)
                self.sector_dummies_cols = sector_dummies_df.columns
                print(f"Learned {len(self.sector_dummies_cols)} sector dummies.")
            if self.sector_dummies_cols:
                 current_sector_dummies = df.select('Sector').to_dummies(columns=['Sector'], drop_first=True)
                 missing_dummies = set(self.sector_dummies_cols) - set(current_sector_dummies.columns)
                 if missing_dummies:
                     current_sector_dummies = current_sector_dummies.with_columns(
                         [pl.lit(0).cast(pl.UInt8).alias(col) for col in missing_dummies]
                     )
                 if current_sector_dummies.width > 0:
                     df = pl.concat([df, current_sector_dummies.select(self.sector_dummies_cols)], how="horizontal")
                     current_features.extend(self.sector_dummies_cols)
                     self.categorical_features.extend(self.sector_dummies_cols)
                 else:
                      print("  Warning: No sector dummies generated (likely only one sector).")
            elif fit_categorical:
                print("  Warning: Could not learn sector dummies (likely only one sector present in training).")
        else: print("Info: 'Sector' column not present in data. Skipping sector features.")

        # Handle Industry
        if 'Industry' in df.columns:
            print("Info: 'Industry' column found. Processing industry features.")
            df = df.with_columns(pl.col('Industry').fill_null('Unknown').cast(pl.Utf8))
            if fit_categorical:
                top_n = 20
                industry_counts = df['Industry'].value_counts().sort(by="counts", descending=True)
                self.top_industries = industry_counts.head(top_n)['Industry'].to_list()
                df_temp = df.with_columns(
                    pl.when(pl.col('Industry').is_in(self.top_industries))
                    .then(pl.col('Industry'))
                    .otherwise(pl.lit('Other_Industry'))
                    .alias('Industry_Top')
                )
                industry_dummies_df = df_temp.select('Industry_Top').to_dummies(columns=['Industry_Top'], drop_first=True)
                self.industry_dummies_cols = industry_dummies_df.columns
                print(f"Learned {len(self.industry_dummies_cols)} industry dummies (Top {len(self.top_industries)} + Other).")

            if self.industry_dummies_cols and self.top_industries:
                df = df.with_columns(
                    pl.when(pl.col('Industry').is_in(self.top_industries))
                    .then(pl.col('Industry'))
                    .otherwise(pl.lit('Other_Industry'))
                    .alias('Industry_Top')
                )
                current_industry_dummies = df.select('Industry_Top').to_dummies(columns=['Industry_Top'], drop_first=True)
                missing_dummies = set(self.industry_dummies_cols) - set(current_industry_dummies.columns)
                if missing_dummies:
                     current_industry_dummies = current_industry_dummies.with_columns(
                         [pl.lit(0).cast(pl.UInt8).alias(col) for col in missing_dummies]
                     )
                if current_industry_dummies.width > 0:
                    df = pl.concat([df, current_industry_dummies.select(self.industry_dummies_cols)], how="horizontal")
                    current_features.extend(self.industry_dummies_cols)
                    self.categorical_features.extend(self.industry_dummies_cols)
                else:
                    print("  Warning: No industry dummies generated (likely only one industry).")
            elif fit_categorical:
                print("  Warning: Could not learn industry dummies (likely only one industry present in training).")
        else: print("Info: 'Industry' column not present in data. Skipping industry features.")

        # Replace any infinities generated during calculations
        feature_cols_in_df = [f for f in current_features if f in df.columns]
        df = df.with_columns([
            pl.when(pl.col(c).is_infinite())
            .then(None)
            .otherwise(pl.col(c))
            .alias(c)
            for c in feature_cols_in_df if df.select(pl.col(c)).dtypes[0] in [pl.Float32, pl.Float64]
        ])

        self.feature_names = sorted(list(set(current_features)))
        print(f"Calculated/checked {len(self.feature_names)} raw Earnings features.")

        # Select final columns
        base_required = ['ticker', 'date', 'Announcement Date', 'ret', 'prc', 'days_to_announcement', 'event_id', 'future_ret']
        # Add optional input columns if they exist and aren't already features
        optional_inputs = ['Sector', 'Industry', 'Time', 'Quarter', 'Surprise']
        keep_cols = base_required + [c for c in optional_inputs if c in df.columns] + self.feature_names
        final_cols_to_keep = sorted(list(set(c for c in keep_cols if c in df.columns))) # Unique and existing

        return df.select(final_cols_to_keep)


    def get_features_target(self, df: pl.DataFrame, fit_imputer: bool = False) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Extract feature matrix X and target vector y as NumPy arrays, handling missing values.
        Returns: Tuple[np.ndarray, np.ndarray, List[str]]: X_np, y_np, final_feature_names
        """
        print("Extracting Earnings features (X) and target (y) as NumPy...")
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
            print("Warning: No data remains after filtering for non-null target.")
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

        print(f"Original numeric X (Polars): {X_numeric_pl.shape}. Categorical X (Polars): {categorical_df.shape}. Non-null y: {y_pl.len()}")

        # Convert numeric features to NumPy
        try:
            X_numeric_np = X_numeric_pl.select(
                [pl.col(c).cast(pl.Float64, strict=False) for c in numeric_features]
            ).to_numpy()
        except Exception as e: raise ValueError(f"Failed to convert numeric features to NumPy: {e}")

        initial_nan_count = np.isnan(X_numeric_np).sum()
        if initial_nan_count > 0: print(f"  Numeric features contain {initial_nan_count} NaN values before imputation.")

        # Impute missing values
        if fit_imputer:
            print("Fitting imputer on numeric NumPy data...")
            if X_numeric_np.size > 0:
                self.imputer.fit(X_numeric_np)
                self._imputer_fitted = True
                print("Transforming with imputer...")
                X_numeric_imputed_np = self.imputer.transform(X_numeric_np)
            else:
                 print("Warning: Numeric feature array is empty. Skipping imputer fitting.")
                 X_numeric_imputed_np = X_numeric_np
                 self._imputer_fitted = True
        else:
            if not self._imputer_fitted: raise RuntimeError("Imputer not fitted. Call with fit_imputer=True first.")
            print("Transforming numeric NumPy data with pre-fitted imputer...")
            if X_numeric_np.size > 0:
                 X_numeric_imputed_np = self.imputer.transform(X_numeric_np)
            else:
                 X_numeric_imputed_np = X_numeric_np

        final_nan_count_numeric = np.isnan(X_numeric_imputed_np).sum()
        if final_nan_count_numeric > 0: warnings.warn(f"NaNs ({final_nan_count_numeric}) remain in numeric features AFTER imputation!")
        elif initial_nan_count > 0: print("No NaNs remaining in numeric features after imputation.")
        else: print("No NaNs found in numeric features before or after imputation.")

        # Convert categorical features to NumPy
        if categorical_df.width > 0:
            try:
                X_categorical_np = categorical_df.select(
                    [pl.col(c).cast(pl.UInt8, strict=False) for c in categorical_cols_present]
                ).to_numpy()
            except Exception as e: raise ValueError(f"Failed to convert categorical features to NumPy: {e}")
        else:
             X_categorical_np = np.empty((X_numeric_imputed_np.shape[0], 0), dtype=np.uint8)
             print("No categorical features found/used.")

        if X_categorical_np.size > 0 and np.isnan(X_categorical_np).any(): warnings.warn("NaNs detected in categorical features after conversion!")

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
             num_cols = len(numeric_features) + len(categorical_cols_present)
             X_np = np.empty((0, num_cols), dtype=np.float64)
             self.final_feature_names = numeric_features + categorical_cols_present

        y_np = y_pl.cast(pl.Float64).to_numpy()

        print(f"Final X NumPy shape: {X_np.shape}. y NumPy shape: {y_np.shape}. Using {len(self.final_feature_names)} features.")
        final_nan_count_all = np.isnan(X_np).sum()
        if final_nan_count_all > 0: warnings.warn(f"NaNs ({final_nan_count_all}) detected in the final combined feature matrix X!")

        return X_np, y_np, self.final_feature_names

# --- SurpriseClassificationModel Class ---
class SurpriseClassificationModel:
    """Model for classifying earnings surprises. Uses NumPy internally."""
    def __init__(self, xgb_cls_params=None, xgb_reg_params=None):
        if xgb_cls_params is None: self.xgb_cls_params = {'n_estimators': 100, 'max_depth': 4, 'learning_rate': 0.05,'objective': 'binary:logistic', 'subsample': 0.8, 'colsample_bytree': 0.8,'random_state': 42, 'n_jobs': -1, 'eval_metric': 'logloss', 'use_label_encoder': False}
        else: self.xgb_cls_params = {**xgb_cls_params, 'use_label_encoder': False}
        if xgb_reg_params is None: self.xgb_reg_params = {'n_estimators': 100, 'max_depth': 5, 'learning_rate': 0.1,'objective': 'reg:squarederror', 'subsample': 0.8, 'colsample_bytree': 0.8,'random_state': 42, 'n_jobs': -1, 'eval_metric': 'rmse', 'use_label_encoder': False}
        else: self.xgb_reg_params = {**xgb_reg_params, 'use_label_encoder': False}

        self.surprise_pos_model = xgb.XGBClassifier(**self.xgb_cls_params)
        self.surprise_neg_model = xgb.XGBClassifier(**self.xgb_cls_params)
        self.return_model = xgb.XGBRegressor(**self.xgb_reg_params)
        self.feature_names_in_ = None # Features for surprise classifiers
        self.return_feature_names_in_ = None # Features for return regressor

    def fit(self, X_np: np.ndarray, y_np: np.ndarray, surprise_np: np.ndarray, feature_names: List[str]):
        """Fit surprise classifiers and return regressor using NumPy arrays."""
        print("Fitting SurpriseClassificationModel...")
        if not all(isinstance(arr, np.ndarray) for arr in [X_np, y_np, surprise_np]):
            raise TypeError("Inputs X_np, y_np, surprise_np must be NumPy arrays.")
        if not X_np.shape[0] == y_np.shape[0] == surprise_np.shape[0]:
            raise ValueError("Inputs X_np, y_np, surprise_np must have the same number of rows.")

        self.feature_names_in_ = feature_names

        pos_surprise_target = (surprise_np > 0).astype(int)
        neg_surprise_target = (surprise_np < 0).astype(int)
        print("  Fitting positive/negative surprise classifiers...")
        if len(np.unique(pos_surprise_target)) > 1:
            self.surprise_pos_model.fit(X_np, pos_surprise_target)
        else:
            print("Warning: Positive surprise target is constant. Skipping positive classifier fit.")
        if len(np.unique(neg_surprise_target)) > 1:
            self.surprise_neg_model.fit(X_np, neg_surprise_target)
        else:
            print("Warning: Negative surprise target is constant. Skipping negative classifier fit.")

        # Prepare data for return regressor
        X_df_temp = pl.DataFrame(X_np, schema=self.feature_names_in_, strict=False)
        X_with_surprise_pl = X_df_temp.with_columns([
            pl.Series('surprise_value_actual', surprise_np), # Use Series constructor
            pl.Series('pos_surprise_actual', pos_surprise_target),
            pl.Series('neg_surprise_actual', neg_surprise_target),
            pl.Series('surprise_magnitude_actual', np.abs(surprise_np))
        ])
        self.return_feature_names_in_ = X_with_surprise_pl.columns
        X_with_surprise_np = X_with_surprise_pl.to_numpy()

        print("  Fitting return regressor...")
        self.return_model.fit(X_with_surprise_np, y_np)
        print("SurpriseClassificationModel fitting complete.")
        return self

    def predict(self, X_np: np.ndarray) -> Dict[str, np.ndarray]:
        """Predict surprise probabilities/classes and returns using NumPy input."""
        if self.feature_names_in_ is None or self.return_feature_names_in_ is None:
             raise RuntimeError("Model not fitted.")
        if not isinstance(X_np, np.ndarray): raise TypeError("Input X_np must be a NumPy array.")
        if X_np.ndim != 2 or X_np.shape[1] != len(self.feature_names_in_):
            raise ValueError(f"Input X_np has shape {X_np.shape}, but model expected (n_samples, {len(self.feature_names_in_)}).")

        pos_prob = np.zeros(X_np.shape[0]); neg_prob = np.zeros(X_np.shape[0])
        if hasattr(self.surprise_pos_model, 'classes_'):
             pos_prob = self.surprise_pos_model.predict_proba(X_np)[:, 1]
        if hasattr(self.surprise_neg_model, 'classes_'):
             neg_prob = self.surprise_neg_model.predict_proba(X_np)[:, 1]

        pos_pred = (pos_prob > 0.5).astype(int); neg_pred = (neg_prob > 0.5).astype(int)

        # Prepare data for return prediction
        X_df_temp = pl.DataFrame(X_np, schema=self.feature_names_in_, strict=False)
        X_for_ret_pl = X_df_temp.with_columns([
            pl.Series('surprise_value_actual', pos_prob - neg_prob),
            pl.Series('pos_surprise_actual', pos_pred),
            pl.Series('neg_surprise_actual', neg_pred),
            pl.Series('surprise_magnitude_actual', np.abs(pos_prob - neg_prob))
        ])

        missing_ret_cols = set(self.return_feature_names_in_) - set(X_for_ret_pl.columns)
        if missing_ret_cols: raise ValueError(f"Internal error: Missing columns required for return prediction: {missing_ret_cols}")
        X_for_ret_np = X_for_ret_pl.select(self.return_feature_names_in_).to_numpy()

        return_pred = self.return_model.predict(X_for_ret_np)
        return {'pos_surprise_prob': pos_prob, 'neg_surprise_prob': neg_prob,
                'pos_surprise_pred': pos_pred, 'neg_surprise_pred': neg_pred,
                'return_pred': return_pred}

    def evaluate(self, X_np: np.ndarray, y_np: np.ndarray, surprise_np: np.ndarray) -> Dict[str, float]:
        """Evaluate surprise classification and return prediction using NumPy inputs."""
        if not all(isinstance(arr, np.ndarray) for arr in [X_np, y_np, surprise_np]):
             raise TypeError("Inputs X_np, y_np, surprise_np must be NumPy arrays.")
        if not X_np.shape[0] == y_np.shape[0] == surprise_np.shape[0]:
            raise ValueError("Inputs X_np, y_np, surprise_np must have the same number of rows.")
        if X_np.shape[0] == 0:
            print("Warning: Empty input for SurpriseClassificationModel evaluation.")
            return {'pos_surprise_accuracy': np.nan, 'neg_surprise_accuracy': np.nan,
                    'return_mse': np.nan, 'return_rmse': np.nan,
                    'return_r2': np.nan, 'return_direction_accuracy': np.nan}

        preds = self.predict(X_np)
        pos_actual = (surprise_np > 0).astype(int); neg_actual = (surprise_np < 0).astype(int)
        pos_acc = np.nan; neg_acc = np.nan

        print("\n--- Surprise Classification Report (Positive) ---")
        if hasattr(self.surprise_pos_model, 'classes_'):
             print(classification_report(pos_actual, preds['pos_surprise_pred'], zero_division=0))
             pos_acc = accuracy_score(pos_actual, preds['pos_surprise_pred'])
        else: print("Positive classifier was not fitted (constant target?).")

        print("\n--- Surprise Classification Report (Negative) ---")
        if hasattr(self.surprise_neg_model, 'classes_'):
             print(classification_report(neg_actual, preds['neg_surprise_pred'], zero_division=0))
             neg_acc = accuracy_score(neg_actual, preds['neg_surprise_pred'])
        else: print("Negative classifier was not fitted (constant target?).")

        valid_mask = np.isfinite(y_np) & np.isfinite(preds['return_pred'])
        y_valid = y_np[valid_mask]; y_pred_valid = preds['return_pred'][valid_mask]
        ret_mse, ret_r2, ret_dir_acc = np.nan, np.nan, np.nan
        if len(y_valid) > 0:
             ret_mse = mean_squared_error(y_valid, y_pred_valid)
             ret_r2 = r2_score(y_valid, y_pred_valid)
             ret_dir_acc = np.mean(np.sign(y_pred_valid) == np.sign(y_valid))

        print("\n--- Return Prediction Evaluation ---")
        print(f"  MSE={ret_mse:.6f}, RMSE={np.sqrt(ret_mse):.6f}, R2={ret_r2:.4f}, DirAcc={ret_dir_acc:.4f}")

        return {'pos_surprise_accuracy': pos_acc, 'neg_surprise_accuracy': neg_acc,
                'return_mse': ret_mse, 'return_rmse': np.sqrt(ret_mse),
                'return_r2': ret_r2, 'return_direction_accuracy': ret_dir_acc}

# --- EarningsDriftModel Class ---
class EarningsDriftModel:
    """Model post-earnings announcement drift (PEAD). Uses NumPy internally."""
    def __init__(self, time_horizons: List[int] = [1, 3, 5, 10, 20], model_params: Optional[Dict] = None):
        self.time_horizons = sorted(time_horizons)
        if model_params is None:
            self.model_params = {
                'n_estimators': 100, 'max_depth': 4, 'learning_rate': 0.05,
                'objective': 'reg:squarederror', 'subsample': 0.8, 'colsample_bytree': 0.8,
                'random_state': 42, 'n_jobs': -1, 'eval_metric': 'rmse',
                'use_label_encoder': False
            }
        else: self.model_params = {**model_params, 'use_label_encoder': False}
        self.models = {h: xgb.XGBRegressor(**self.model_params) for h in time_horizons}
        self.feature_names_in_: Optional[List[str]] = None
        self.imputers: Dict[int, SimpleImputer] = {}

    def _prepare_data_for_horizon(self, data: pl.DataFrame, horizon: int, feature_cols: List[str], return_col: str = 'ret') -> Tuple[Optional[pl.DataFrame], Optional[pl.Series]]:
        """Prepare features (announcement day) and target (cumulative return) using Polars."""
        base_required = ['event_id', 'date', 'days_to_announcement', return_col, 'Announcement Date']
        required = base_required + feature_cols
        if not all(c in data.columns for c in required):
            missing = [c for c in required if c not in data.columns]
            print(f"Warning (PEAD Prep): Missing columns for horizon {horizon}: {missing}. Skipping.")
            return None, None

        data = data.sort(['event_id', 'date'])

        # Calculate cumulative returns factor
        data = data.with_columns(
            (pl.col(return_col).shift(-1).over("event_id").fill_null(0) + 1)
             .rolling_prod(window_size=horizon, min_periods=1)
             .over("event_id")
             .alias(f"cum_ret_factor_temp")
        )

        announce_day_data = data.filter(pl.col('days_to_announcement') == 0)
        if announce_day_data.is_empty():
             print(f"Warning (PEAD Prep): No announcement day data found for horizon {horizon}.")
             return None, None

        # Calculate target return and drop nulls
        final_data = announce_day_data.with_columns(
             (pl.col(f"cum_ret_factor_temp") - 1).alias(f"target_cum_ret_h{horizon}")
        ).drop_nulls(subset=[f"target_cum_ret_h{horizon}"])

        if final_data.is_empty():
            print(f"Warning (PEAD Prep): No valid target data after calculation for horizon {horizon}.")
            return None, None

        # Extract features and target
        X = final_data.select(feature_cols)
        y = final_data.get_column(f"target_cum_ret_h{horizon}")

        return X, y

    def fit(self, data: pl.DataFrame, feature_cols: List[str]):
        """Fit separate models for each horizon using NumPy conversion."""
        print("Fitting EarningsDriftModel...")
        if not feature_cols: raise ValueError("PEAD features cannot be empty.")
        self.feature_names_in_ = feature_cols

        for horizon in self.time_horizons:
            print(f"  Training PEAD model for {horizon}-day horizon...")
            try:
                X_pl, y_pl = self._prepare_data_for_horizon(data, horizon, self.feature_names_in_)

                if X_pl is None or y_pl is None or X_pl.is_empty():
                    print(f"    No data for horizon {horizon}. Skipping.")
                    continue

                # Convert to NumPy
                X_np = X_pl.select(
                     [pl.col(c).cast(pl.Float64, strict=False) for c in self.feature_names_in_]
                 ).to_numpy()
                y_np = y_pl.cast(pl.Float64).to_numpy()

                # Impute NaNs if necessary
                if np.isnan(X_np).any():
                    print(f"    Imputing NaNs for horizon {horizon}...")
                    imputer = SimpleImputer(strategy='median')
                    X_np = imputer.fit_transform(X_np)
                    self.imputers[horizon] = imputer

                if X_np.size > 0:
                    self.models[horizon].fit(X_np, y_np)
                    print(f"    PEAD model {horizon}d fitted ({len(X_np)} samples).")
                else:
                     print(f"    No samples remaining after preparation for horizon {horizon}. Skipping fit.")

            except Exception as e:
                print(f"    Error training PEAD {horizon}d: {e}")
                traceback.print_exc()
        print("EarningsDriftModel fitting complete.")
        return self

    def predict(self, data: pl.DataFrame) -> pl.DataFrame:
        """Generate PEAD predictions using features from announcement day. Returns Polars DF."""
        print("Generating PEAD predictions...")
        if self.feature_names_in_ is None: raise RuntimeError("PEAD model not fitted.")

        announce_days = data.filter(pl.col('days_to_announcement') == 0)
        if announce_days.is_empty():
            print("Warning: No announcement day data for PEAD prediction.")
            pred_cols_schema = {f'pred_drift_h{h}': pl.Float64 for h in self.time_horizons}
            announce_schema = announce_days.schema
            return pl.DataFrame(schema={**announce_schema, **pred_cols_schema})

        missing = [f for f in self.feature_names_in_ if f not in announce_days.columns]
        if missing: raise ValueError(f"PEAD prediction features missing: {missing}")

        X_pl = announce_days.select(self.feature_names_in_)

        try:
             X_np = X_pl.select(
                 [pl.col(c).cast(pl.Float64, strict=False) for c in self.feature_names_in_]
             ).to_numpy()
        except Exception as e:
             raise ValueError(f"Failed to convert PEAD prediction features to NumPy: {e}")

        pred_expressions = []
        for horizon in self.time_horizons:
             pred_col = f'pred_drift_h{horizon}'
             if horizon in self.models:
                 try:
                     X_np_h = X_np.copy()
                     if np.isnan(X_np_h).any():
                          if horizon in self.imputers:
                              X_np_h = self.imputers[horizon].transform(X_np_h)
                          else:
                              print(f"  Warning: No stored imputer for horizon {horizon}. Imputing PEAD predict with temporary median.")
                              temp_imputer = SimpleImputer(strategy='median')
                              X_np_h = temp_imputer.fit_transform(X_np_h)
                          if np.isnan(X_np_h).any():
                              print(f"  Warning: NaNs remain after imputation for horizon {horizon}. Predictions might be NaN.")

                     if X_np_h.size > 0:
                        predictions = self.models[horizon].predict(X_np_h)
                        pred_expressions.append(pl.Series(pred_col, predictions))
                     else:
                         pred_expressions.append(pl.lit(None).cast(pl.Float64).alias(pred_col))

                 except Exception as e:
                     print(f"Error predicting PEAD {horizon}d: {e}")
                     pred_expressions.append(pl.lit(None).cast(pl.Float64).alias(pred_col))
             else:
                 pred_expressions.append(pl.lit(None).cast(pl.Float64).alias(pred_col))

        predictions_df = announce_days.hstack(pred_expressions)
        return predictions_df


    def evaluate(self, data: pl.DataFrame) -> Dict[int, Dict[str, Any]]:
        """Evaluate PEAD models on test data using NumPy conversion."""
        print("\n--- Evaluating EarningsDriftModel ---")
        results: Dict[int, Dict[str, Any]] = {}
        if self.feature_names_in_ is None:
             print("PEAD model not fitted."); return results

        for horizon in self.time_horizons:
            print(f"  Evaluating PEAD {horizon}-day horizon...")
            try:
                X_test_pl, y_test_pl = self._prepare_data_for_horizon(data, horizon, self.feature_names_in_)

                if X_test_pl is None or y_test_pl is None or X_test_pl.is_empty():
                     print("    No test data available for this horizon.")
                     results[horizon] = {'MSE': np.nan, 'RMSE': np.nan, 'R2': np.nan, 'Direction Accuracy': np.nan, 'N': 0, 'Error': 'No data'}
                     continue

                X_test_np = X_test_pl.select(
                     [pl.col(c).cast(pl.Float64, strict=False) for c in self.feature_names_in_]
                 ).to_numpy()
                y_test_np = y_test_pl.cast(pl.Float64).to_numpy()

                if np.isnan(X_test_np).any():
                    if horizon in self.imputers:
                        X_test_np = self.imputers[horizon].transform(X_test_np)
                    else:
                        print(f"    Warning: No stored imputer for horizon {horizon} evaluation. Imputing with temporary median.")
                        temp_imputer = SimpleImputer(strategy='median')
                        X_test_np = temp_imputer.fit_transform(X_test_np)
                    if np.isnan(X_test_np).any():
                         print(f"  Warning: NaNs remain after imputation for horizon {horizon} evaluation.")

                if horizon not in self.models:
                    print("    Model not available.")
                    results[horizon] = {'MSE': np.nan, 'RMSE': np.nan, 'R2': np.nan, 'Direction Accuracy': np.nan, 'N': 0, 'Error': 'Model not trained'}
                    continue

                y_pred_np = np.array([]) # Default to empty
                if X_test_np.size > 0:
                    y_pred_np = self.models[horizon].predict(X_test_np)
                else:
                     print("    Test data empty after prep. Skipping prediction.")

                valid_mask = np.isfinite(y_test_np) & np.isfinite(y_pred_np)
                y_test_v, y_pred_v = y_test_np[valid_mask], y_pred_np[valid_mask]
                mse, r2, dir_acc = np.nan, np.nan, np.nan
                if len(y_test_v) > 0:
                    mse = mean_squared_error(y_test_v, y_pred_v)
                    r2 = r2_score(y_test_v, y_pred_v)
                    dir_acc = np.mean(np.sign(y_pred_v) == np.sign(y_test_v))

                results[horizon] = {'MSE': mse, 'RMSE': np.sqrt(mse), 'R2': r2, 'Direction Accuracy': dir_acc, 'N': len(y_test_v)}
                print(f"    PEAD {horizon}d: MSE={mse:.6f}, R2={r2:.4f}, DirAcc={dir_acc:.4f}, N={len(y_test_v)}")

            except Exception as e:
                print(f"    Error evaluating PEAD {horizon}d: {e}")
                results[horizon] = {'Error': str(e)}
                traceback.print_exc()
        return results

class Analysis:
    def __init__(self, data_loader: 'DataLoader', feature_engineer: 'FeatureEngineer'):
        """Analysis class for Earnings data using Polars."""
        self.data_loader = data_loader
        self.feature_engineer = feature_engineer
        self.data: Optional[pl.DataFrame] = None
        # Store NumPy arrays for ML models
        self.X_train_np: Optional[np.ndarray] = None
        self.X_test_np: Optional[np.ndarray] = None
        self.y_train_np: Optional[np.ndarray] = None
        self.y_test_np: Optional[np.ndarray] = None
        self.train_data: Optional[pl.DataFrame] = None # Store processed train split DF
        self.test_data: Optional[pl.DataFrame] = None # Store processed test split DF
        self.final_feature_names: Optional[List[str]] = None # Store final feature names after processing
        self.models: Dict[str, Any] = {} # Standard models (Ridge, XGBDecile)
        self.surprise_model: Optional['SurpriseClassificationModel'] = None
        self.pead_model: Optional['EarningsDriftModel'] = None
        self.results: Dict[str, Dict] = {} # Standard model results
        self.surprise_results: Dict = {}
        self.pead_results: Dict = {}

    def load_and_prepare_data(self, run_feature_engineering: bool = True):
        """Load and optionally prepare data for earnings analysis using Polars."""
        print("--- Loading Earnings Data (Polars) ---")
        self.data = self.data_loader.load_data() # Uses chunking internally now
        if self.data is None or self.data.is_empty():
            raise RuntimeError("Data loading failed or resulted in empty dataset.")

        if run_feature_engineering:
             if self.data is None or self.data.is_empty():
                  raise RuntimeError("Cannot run feature engineering on empty data.")
             print("\n--- Creating Target Variable (Earnings - Polars) ---")
             self.data = self.feature_engineer.create_target(self.data)
             print("\n--- Calculating Features (Earnings - Polars) ---")
             self.data = self.feature_engineer.calculate_features(self.data, fit_categorical=False)

        print("\n--- Earnings Data Loading/Preparation Complete ---")
        return self.data

    def train_models(self, test_size=0.2, time_split_column='Announcement Date'):
        """Split data, process features/target, and train all models using Polars/NumPy."""
        if self.data is None: raise RuntimeError("Run load_and_prepare_data() first.")
        if time_split_column not in self.data.columns: raise ValueError(f"Time split column '{time_split_column}' not found.")
        if 'event_id' not in self.data.columns: raise ValueError("'event_id' required.")
        if 'future_ret' not in self.data.columns:
             print("ML Prep: Target variable 'future_ret' not found. Creating...")
             self.data = self.feature_engineer.create_target(self.data)
        if not self.feature_engineer.feature_names:
             print("ML Prep: Features not calculated. Calculating...")
             self.data = self.feature_engineer.calculate_features(self.data, fit_categorical=False)

        print(f"\n--- Splitting Earnings Data (Train/Test based on {time_split_column}) ---")
        events = self.data.select(['event_id', time_split_column]).unique().sort(time_split_column)
        split_index = int(events.height * (1 - test_size))
        if split_index == 0 or split_index == events.height: raise ValueError("test_size results in empty train/test set.")
        split_date = events.item(split_index, time_split_column)
        print(f"Splitting {events.height} unique events. Train before {split_date}.")

        train_mask = pl.col(time_split_column) < split_date
        test_mask = pl.col(time_split_column) >= split_date
        train_data_raw = self.data.filter(train_mask)
        test_data_raw = self.data.filter(test_mask)
        print(f"Train rows (raw): {train_data_raw.height}, Test rows (raw): {test_data_raw.height}.")
        if train_data_raw.is_empty() or test_data_raw.is_empty():
            raise ValueError("Train or test split resulted in empty DataFrame.")

        print("\nFitting FeatureEngineer components (Categorical/Imputer) on Training Data...")
        self.train_data = self.feature_engineer.calculate_features(train_data_raw, fit_categorical=True)
        self.X_train_np, self.y_train_np, _ = self.feature_engineer.get_features_target(self.train_data, fit_imputer=True)

        print("\nApplying FeatureEngineer components to Test Data...")
        self.test_data = self.feature_engineer.calculate_features(test_data_raw, fit_categorical=False)
        self.X_test_np, self.y_test_np, self.final_feature_names = self.feature_engineer.get_features_target(self.test_data, fit_imputer=False)

        print(f"\nTrain shapes (NumPy): X={self.X_train_np.shape}, y={self.y_train_np.shape}")
        print(f"Test shapes (NumPy): X={self.X_test_np.shape}, y={self.y_test_np.shape}")
        if self.X_train_np.shape[0] == 0 or self.X_test_np.shape[0] == 0:
             raise ValueError("Train or test NumPy array empty after feature extraction.")
        if not self.final_feature_names:
            raise RuntimeError("Final feature names were not set during feature extraction.")

        print("\n--- Training Standard Models (Earnings) ---")
        try:
             if self.X_train_np.ndim == 2 and self.X_train_np.shape[1] == len(self.final_feature_names):
                 X_train_pl_imputed = pl.DataFrame(self.X_train_np, schema=self.final_feature_names, strict=False)
             else:
                  raise ValueError(f"Shape mismatch: X_train_np {self.X_train_np.shape} vs features {len(self.final_feature_names)}")
             y_train_pl = pl.Series("future_ret", self.y_train_np)
        except Exception as e:
             raise RuntimeError(f"Could not convert imputed NumPy training arrays to Polars: {e}")

        # 1. TimeSeriesRidge
        try:
             print("Training TimeSeriesRidge...")
             ts_ridge = TimeSeriesRidge(alpha=1.0, lambda2=0.1, feature_order=self.final_feature_names)
             ts_ridge.fit(X_train_pl_imputed, y_train_pl)
             self.models['TimeSeriesRidge'] = ts_ridge
             print("TimeSeriesRidge complete.")
        except Exception as e: print(f"Error TimeSeriesRidge: {e}"); traceback.print_exc()

        # 2. XGBoostDecile
        try:
             print("\nTraining XGBoostDecile...")
             xgb_params = {'n_estimators': 150, 'max_depth': 5, 'learning_rate': 0.05, 'objective': 'reg:squarederror', 'subsample': 0.8, 'colsample_bytree': 0.8, 'random_state': 42, 'n_jobs': -1, 'early_stopping_rounds': 10}
             if 'momentum_5' not in X_train_pl_imputed.columns: print("Warning: 'momentum_5' not found for XGBoostDecile.")
             xgb_decile = XGBoostDecileModel(weight=0.7, momentum_feature='momentum_5', n_deciles=10, alpha=1.0, lambda_smooth=0.1, xgb_params=xgb_params, ts_ridge_feature_order=self.final_feature_names)
             xgb_decile.fit(X_train_pl_imputed, y_train_pl)
             self.models['XGBoostDecile'] = xgb_decile
             print("XGBoostDecile complete.")
        except Exception as e: print(f"Error XGBoostDecile: {e}"); traceback.print_exc()

        # --- Train Surprise Classification Model ---
        surprise_col_name = 'surprise_val'
        if self.train_data is not None and surprise_col_name in self.train_data.columns and self.train_data.filter(pl.col(surprise_col_name).is_not_null()).height > 0:
             print("\n--- Training Surprise Classification Model ---")
             try:
                 train_data_for_surprise = self.train_data.filter(pl.col('future_ret').is_not_null())
                 if train_data_for_surprise.height == self.X_train_np.shape[0]:
                     surprise_train_np = train_data_for_surprise.get_column(surprise_col_name).cast(pl.Float64).fill_null(0).to_numpy()
                     self.surprise_model = SurpriseClassificationModel()
                     self.surprise_model.fit(self.X_train_np, self.y_train_np, surprise_train_np, self.final_feature_names)
                     print("SurpriseClassificationModel training complete.")
                 else:
                     print(f"Warning: Row mismatch between features ({self.X_train_np.shape[0]}) and filtered train data ({train_data_for_surprise.height}). Skipping surprise model.")
                     self.surprise_model = None
             except Exception as e: print(f"Error SurpriseClassificationModel: {e}"); self.surprise_model = None; traceback.print_exc()
        else: print(f"\n'{surprise_col_name}' column not found or all NaNs in processed train set. Skipping surprise model.")

        # --- Train Earnings Drift (PEAD) Model ---
        print("\n--- Training Earnings Drift (PEAD) Model ---")
        try:
             self.pead_model = EarningsDriftModel(time_horizons=[1, 3, 5, 10, 20])
             if self.train_data is not None:
                 self.pead_model.fit(self.train_data, feature_cols=self.final_feature_names)
                 print("EarningsDriftModel training complete.")
             else:
                 print("Error: Processed training data (self.train_data) is None. Skipping PEAD model.")
                 self.pead_model = None
        except Exception as e: print(f"Error EarningsDriftModel: {e}"); self.pead_model = None; traceback.print_exc()

        print("\n--- All Earnings Model Training Complete ---"); return self.models

    def evaluate_models(self) -> Dict[str, Dict]:
        """Evaluate all trained models on the test set using NumPy arrays."""
        print("\n--- Evaluating Earnings Models ---")
        if self.X_test_np is None or self.y_test_np is None or self.X_test_np.shape[0]==0:
             print("Test data (NumPy) unavailable/empty."); return {}
        if not self.final_feature_names:
             print("Final feature names not available. Cannot evaluate.")
             return {}

        print("\n--- Standard Model Evaluation ---")
        self.results = {}
        try:
            if self.X_test_np.ndim == 2 and self.X_test_np.shape[1] == len(self.final_feature_names):
                X_test_pl_imputed = pl.DataFrame(self.X_test_np, schema=self.final_feature_names, strict=False)
            elif self.X_test_np.shape[0] == 0:
                print("Warning: Test set is empty. Creating empty DataFrame for evaluation.")
                X_test_pl_imputed = pl.DataFrame(schema=self.final_feature_names)
            else:
                 raise ValueError(f"Shape mismatch: X_test_np {self.X_test_np.shape} vs features {len(self.final_feature_names)}")
        except Exception as e:
             print(f"Error converting imputed NumPy test features to Polars: {e}. Skipping standard model eval.")
             X_test_pl_imputed = None

        for name, model in self.models.items():
            print(f"Evaluating {name}...")
            if X_test_pl_imputed is None and isinstance(model, (TimeSeriesRidge, XGBoostDecileModel)):
                 print(f"  Skipping {name} because Polars DataFrame conversion failed.")
                 self.results[name] = {'Error': 'Polars DF conversion failed'}
                 continue
            if X_test_pl_imputed is not None and X_test_pl_imputed.is_empty():
                print(f"  Skipping {name} evaluation as test data is empty.")
                self.results[name] = {'MSE': np.nan, 'RMSE': np.nan, 'R2': np.nan, 'N': 0}
                continue

            try:
                y_pred_np = model.predict(X_test_pl_imputed)
                valid_mask = np.isfinite(self.y_test_np) & np.isfinite(y_pred_np)
                y_test_v, y_pred_v = self.y_test_np[valid_mask], y_pred_np[valid_mask]
                mse, r2 = np.nan, np.nan
                if len(y_test_v) > 0:
                    mse = mean_squared_error(y_test_v, y_pred_v)
                    r2 = r2_score(y_test_v, y_pred_v)
                self.results[name] = {'MSE': mse, 'RMSE': np.sqrt(mse), 'R2': r2, 'N': len(y_test_v)}
                print(f"  {name}: MSE={mse:.6f}, RMSE={np.sqrt(mse):.6f}, R2={r2:.4f}, N={len(y_test_v)}")
            except Exception as e:
                 print(f"  Error evaluating {name}: {e}")
                 self.results[name] = {'Error': str(e)}
                 traceback.print_exc()

        # Evaluate Surprise Model
        surprise_col_name = 'surprise_val'
        if self.surprise_model:
             print("\n--- Surprise Model Evaluation ---")
             if self.test_data is not None and surprise_col_name in self.test_data.columns and self.test_data.filter(pl.col(surprise_col_name).is_not_null()).height > 0:
                 test_data_for_surprise = self.test_data.filter(pl.col('future_ret').is_not_null())
                 if test_data_for_surprise.height == self.X_test_np.shape[0]:
                     surprise_test_np = test_data_for_surprise.get_column(surprise_col_name).cast(pl.Float64).fill_null(0).to_numpy()
                     try:
                         self.surprise_results = self.surprise_model.evaluate(self.X_test_np, self.y_test_np, surprise_test_np)
                     except Exception as e:
                         print(f"  Error evaluating Surprise Model: {e}"); self.surprise_results = {'Error': str(e)}; traceback.print_exc()
                 else:
                     print(f"  Warning: Row mismatch between test features ({self.X_test_np.shape[0]}) and filtered test data ({test_data_for_surprise.height}). Cannot evaluate surprise model.")
                     self.surprise_results = {'Error': 'Row mismatch'}
             else:
                 print(f"  '{surprise_col_name}' column not found or all NaNs in processed test data. Cannot evaluate.")
                 self.surprise_results = {'Error': f'{surprise_col_name} not found or all null'}
        else: print("\nSurprise Model not trained. Skipping evaluation.")

        # Evaluate PEAD Model
        if self.pead_model:
             print("\n--- PEAD Model Evaluation ---");
             try:
                 if self.test_data is not None:
                    self.pead_results = self.pead_model.evaluate(self.test_data)
                 else:
                     print("  Processed test data unavailable. Cannot evaluate PEAD model.")
                     self.pead_results = {'Error': 'Test data unavailable'}
             except Exception as e: print(f"  Error evaluating PEAD Model: {e}"); self.pead_results = {'Error': str(e)}; traceback.print_exc()
        else: print("\nPEAD Model not trained. Skipping evaluation.")

        print("\n--- Earnings Evaluation Complete ---")
        return {"standard": self.results, "surprise": self.surprise_results, "pead": self.pead_results}


    # --- Plotting and Analysis Methods ---
    def plot_feature_importance(self, results_dir: str, file_prefix: str = "earnings", model_name: str = 'TimeSeriesRidge', top_n: int = 20):
        """Plot feature importance and save the plot. Uses model coefficients/importances."""
        print(f"\n--- Plotting Earnings Feature Importance for {model_name} ---")
        if model_name not in self.models: print(f"Error: Model '{model_name}' not found."); return None
        model = self.models[model_name]
        feature_names = self.final_feature_names
        if not feature_names: print("Error: Final feature names not found (run training first)."); return None

        importances = None
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
                     try:
                        booster = model.xgb_model.get_booster(); xgb_feat_names = booster.feature_names
                        if xgb_feat_names and len(xgb_feat_names) == len(xgb_importances):
                             imp_dict = dict(zip(xgb_feat_names, xgb_importances))
                             importances = np.array([imp_dict.get(name, 0) for name in feature_names])
                        else: raise ValueError("Mismatch")
                     except Exception: print(f"Warn: XGB importance len mismatch. Cannot plot reliably.")

        if importances is None: print(f"Could not get importance scores for {model_name}."); return None

        feat_imp_df_pl = pl.DataFrame({'Feature': feature_names, 'Importance': importances}) \
                          .sort('Importance', descending=True) \
                          .head(top_n)
        feat_imp_df_pd = feat_imp_df_pl.to_pandas()

        fig, ax = plt.subplots(figsize=(10, max(6, top_n // 2)))
        sns.barplot(x='Importance', y='Feature', data=feat_imp_df_pd, palette='viridis', ax=ax)
        ax.set_title(f'Top {top_n} Features by Importance ({model_name} - Earnings)')
        ax.set_xlabel('Importance Score'); ax.set_ylabel('Feature')
        plt.tight_layout()
        plot_filename = os.path.join(results_dir, f"{file_prefix}_feat_importance_{model_name}.png")
        try: plt.savefig(plot_filename); print(f"Saved feature importance plot to: {plot_filename}")
        except Exception as e: print(f"Error saving feature importance plot: {e}")
        plt.close(fig)
        return pl.DataFrame({'Feature': feature_names, 'Importance': importances}).sort('Importance', descending=True)

    def analyze_earnings_surprise(self, results_dir: str, file_prefix: str = "earnings"):
        """Analyze the impact of earnings surprises on actual returns using Polars and plot."""
        print("\n--- Analyzing Earnings Surprise Impact (Polars) ---")
        surprise_col = 'surprise_val'
        if self.data is None or surprise_col not in self.data.columns or self.data.select(surprise_col).is_null().all():
            print(f"No valid earnings surprise data ('{surprise_col}') available."); return None

        analysis_data = self.data.filter(
            (pl.col('days_to_announcement') >= -2) & (pl.col('days_to_announcement') <= 5)
        )
        if analysis_data.is_empty(): print("No data in analysis window."); return None

        bins = [-np.inf, -0.01, 0.01, np.inf]
        labels = ['Negative Surprise', 'Near Zero Surprise', 'Positive Surprise']
        analysis_data = analysis_data.with_columns(
             pl.when(pl.col(surprise_col) < bins[1]).then(pl.lit(labels[0]))
             .when(pl.col(surprise_col) < bins[2]).then(pl.lit(labels[1]))
             .when(pl.col(surprise_col) >= bins[2]).then(pl.lit(labels[2]))
             .otherwise(None).alias('Surprise Group')
        ).drop_nulls(subset=['Surprise Group'])

        if analysis_data.is_empty(): print("No data after assigning surprise groups."); return None

        avg_returns = analysis_data.group_by(['Surprise Group', 'days_to_announcement']).agg(
            pl.mean('ret').alias('avg_ret')
        ).sort(['Surprise Group', 'days_to_announcement'])

        avg_returns_pivot = None
        try:
             avg_returns_pivot = avg_returns.pivot(index='days_to_announcement', columns='Surprise Group', values='avg_ret').sort('days_to_announcement')
        except Exception as e: print(f"Could not pivot average returns data: {e}. Skipping daily plot.")

        avg_cum_returns = avg_returns.sort(['Surprise Group', 'days_to_announcement']).with_columns(
            (pl.col('avg_ret').fill_null(0) + 1).cum_prod().over('Surprise Group').alias('cum_ret_factor')
        ).with_columns(
            (pl.col('cum_ret_factor') - 1).alias('avg_cum_ret')
        )

        avg_cum_returns_pivot = None; avg_cum_returns_pd = None
        try:
             avg_cum_returns_pivot = avg_cum_returns.pivot(index='days_to_announcement', columns='Surprise Group', values='avg_cum_ret').sort('days_to_announcement')
             avg_cum_returns_pd = avg_cum_returns_pivot.to_pandas().set_index('days_to_announcement')
        except Exception as e: print(f"Could not pivot cumulative returns data: {e}. Skipping cumulative plot.")

        fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
        if avg_returns_pivot is not None:
             try:
                avg_returns_pd = avg_returns_pivot.to_pandas().set_index('days_to_announcement')
                avg_returns_pd.plot(kind='bar', ax=axes[0], width=0.8)
                axes[0].set_title('Average Daily Returns by Surprise')
                axes[0].set_ylabel('Avg Daily Return'); axes[0].axhline(0, c='grey', ls='--', lw=0.8)
                axes[0].legend(title='Group'); axes[0].grid(axis='y', ls=':', alpha=0.6)
             except Exception as plot_err:
                 print(f"Error plotting daily surprise returns: {plot_err}")
                 axes[0].set_title('Average Daily Returns by Surprise (Plotting Error)')
        else: axes[0].set_title('Average Daily Returns by Surprise (Pivot Error)')

        if avg_cum_returns_pd is not None:
             try:
                avg_cum_returns_pd.plot(kind='line', marker='o', ax=axes[1])
                axes[1].set_title('Average Cumulative Returns by Surprise')
                axes[1].set_ylabel('Avg Cum Return'); axes[1].set_xlabel('Days Rel. to Announce')
                axes[1].axhline(0, c='grey', ls='--', lw=0.8); axes[1].axvline(0, c='red', ls=':', lw=1, label='Announce Day')
                axes[1].legend(title='Group'); axes[1].grid(True, ls=':', alpha=0.6)
             except Exception as plot_err:
                  print(f"Error plotting cumulative surprise returns: {plot_err}")
                  axes[1].set_title('Average Cumulative Returns by Surprise (Plotting Error)')
        else: axes[1].set_title('Average Cumulative Returns by Surprise (Pivot Error)')

        plt.tight_layout()
        plot_filename = os.path.join(results_dir, f"{file_prefix}_surprise_impact_returns.png")
        csv_filename = os.path.join(results_dir, f"{file_prefix}_surprise_impact_cum_returns_data.csv")
        try: plt.savefig(plot_filename); print(f"Saved surprise impact plot to: {plot_filename}")
        except Exception as e: print(f"Error saving surprise plot: {e}")
        if avg_cum_returns_pivot is not None:
            try: avg_cum_returns_pivot.write_csv(csv_filename); print(f"Saved surprise cum returns data to: {csv_filename}")
            except Exception as e: print(f"Error saving surprise data: {e}")
        plt.close(fig)
        return avg_cum_returns

    def plot_predictions_for_event(self, results_dir: str, event_id: str, file_prefix: str = "earnings", model_name: str = 'TimeSeriesRidge'):
        """Plot actual daily returns and model's predicted future returns using Polars data and save plot."""
        print(f"\n--- Plotting Earnings Predictions for Event: {event_id} ({model_name}) ---")
        if model_name not in self.models: print(f"Error: Model '{model_name}' not found."); return None
        if self.data is None: print("Error: Data not loaded."); return None
        model = self.models[model_name]

        event_data_full = self.data.filter(pl.col('event_id') == event_id).sort('date')
        if event_data_full.is_empty(): print(f"Error: No data for event_id '{event_id}'."); return None

        ticker = event_data_full['ticker'][0]; announcement_date = event_data_full['Announcement Date'][0]

        if not self.feature_engineer._imputer_fitted: print("Error: FeatureEngineer imputer not fitted (run training first)."); return None
        if not self.final_feature_names: print("Error: Final feature names not set (run training first)."); return None

        event_data_processed = self.feature_engineer.calculate_features(event_data_full, fit_categorical=False)
        X_event_np, y_event_actual_np, event_features = self.feature_engineer.get_features_target(event_data_processed, fit_imputer=False)

        if X_event_np.shape[0] == 0: print(f"Warn: No valid features/target rows for event {event_id}."); return None

        try:
            X_event_pl = pl.DataFrame(X_event_np, schema=event_features, strict=False)
            y_pred_event_np = model.predict(X_event_pl)
        except Exception as e: print(f"Error predicting event {event_id}: {e}"); return None

        event_data_pred_source = event_data_processed.filter(pl.col('future_ret').is_not_null())
        if event_data_pred_source.height != len(y_pred_event_np):
             print(f"Warn: Mismatch between prediction count ({len(y_pred_event_np)}) and source data rows ({event_data_pred_source.height}). Plot may be inaccurate.")
             min_len = min(event_data_pred_source.height, len(y_pred_event_np))
             event_data_pred_source = event_data_pred_source.head(min_len)
             y_pred_event_np = y_pred_event_np[:min_len]

        event_data_pred = event_data_pred_source.select(['date']).with_columns(pl.Series('predicted_future_ret', y_pred_event_np))

        event_data_full_pd = event_data_full.select(['date', 'ret']).to_pandas()
        event_data_pred_pd = event_data_pred.to_pandas()
        announcement_date_pd = pd.Timestamp(announcement_date)

        fig, ax = plt.subplots(figsize=(14, 6))
        ax.plot(event_data_full_pd['date'], event_data_full_pd['ret'], '.-', label='Actual Daily Return', alpha=0.6, color='blue')
        ax.scatter(event_data_pred_pd['date'], event_data_pred_pd['predicted_future_ret'], color='red', label=f'Predicted {self.feature_engineer.prediction_window}-day Future Ret', s=15, zorder=5, alpha=0.8)
        ax.axvline(x=announcement_date_pd, color='g', linestyle='--', label='Announcement Date')
        ax.set_title(f"{ticker} ({event_id}) - Daily Returns & Predicted Future Returns ({model_name} - Earnings)")
        ax.set_ylabel("Return"); ax.set_xlabel("Date"); ax.legend(); ax.grid(True, alpha=0.3)
        plt.tight_layout()

        safe_event_id = "".join(c if c.isalnum() else "_" for c in event_id)
        plot_filename = os.path.join(results_dir, f"{file_prefix}_pred_vs_actual_{safe_event_id}_{model_name}.png")
        try: plt.savefig(plot_filename); print(f"Saved prediction plot to: {plot_filename}")
        except Exception as e: print(f"Error saving prediction plot: {e}")
        plt.close(fig)
        return event_data_pred

    def find_sample_event_ids(self, n=5):
        """Find sample Earnings event identifiers from Polars data."""
        if self.data is None or 'event_id' not in self.data.columns: return []
        unique_events = self.data['event_id'].unique().head(n)
        return unique_events.to_list()

    def plot_pead_predictions(self, results_dir: str, file_prefix: str = "earnings", n_events: int = 3):
        """Plot PEAD predictions vs actual cumulative returns using Polars data."""
        print("\n--- Plotting PEAD Predictions vs Actual (Polars) ---")
        if self.pead_model is None: print("PEAD Model not trained."); return
        if self.test_data is None: print("Test data not available (run train_models first)."); return

        pead_predictions_df = self.pead_model.predict(self.test_data)
        if pead_predictions_df.is_empty(): print("No PEAD predictions generated for test data."); return

        available_event_ids = pead_predictions_df['event_id'].unique().to_list()
        if not available_event_ids: print("No event IDs found in PEAD predictions."); return

        sample_event_ids = available_event_ids[:min(n_events * 5, len(available_event_ids))]

        plotted_count = 0
        for event_id in sample_event_ids:
            if plotted_count >= n_events: break

            event_preds = pead_predictions_df.filter(pl.col('event_id') == event_id)
            if event_preds.is_empty(): continue

            event_actual_data = self.test_data.filter(pl.col('event_id') == event_id).sort('date')
            if event_actual_data.is_empty(): continue

            ticker = event_actual_data['ticker'][0]
            announce_date = event_actual_data['Announcement Date'][0]

            post_announce_actual = event_actual_data.filter(pl.col('date') >= announce_date)
            if post_announce_actual.is_empty(): print(f"No post-announcement data for {event_id}"); continue

            # Calculate cumulative product correctly within the filtered event data
            post_announce_actual = post_announce_actual.with_columns(
                 ((pl.col('ret').fill_null(0) + 1).cum_prod() - 1) # Cumprod over the filtered group
                 .alias('actual_cum_ret')
            )

            pred_cols = sorted([col for col in event_preds.columns if col.startswith('pred_drift_h')], key=lambda x: int(x[len('pred_drift_h'):]))
            horizons = [int(col[len('pred_drift_h'):]) for col in pred_cols]
            if event_preds.height == 0: continue
            pred_values = event_preds.select(pred_cols).row(0)

            plot_dates_pd = [pd.Timestamp(announce_date) + pd.Timedelta(days=h) for h in horizons]
            pred_values_pd = list(pred_values)
            post_announce_actual_pd = post_announce_actual.select(['date', 'actual_cum_ret']).to_pandas()
            announce_date_pd = pd.Timestamp(announce_date)

            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(post_announce_actual_pd['date'], post_announce_actual_pd['actual_cum_ret'], marker='.', linestyle='-', label='Actual Cum. Return', color='blue')
            ax.scatter(plot_dates_pd, pred_values_pd, color='red', marker='x', s=50, label='Predicted Cum. Return (PEAD)', zorder=5)
            ax.set_title(f"PEAD Analysis: {ticker} ({event_id})")
            ax.set_xlabel("Date"); ax.set_ylabel("Cumulative Return")
            ax.axvline(announce_date_pd, color='grey', linestyle='--', label='Announcement Date')
            ax.legend(); ax.grid(True, alpha=0.4)
            plt.tight_layout()

            safe_event_id = "".join(c if c.isalnum() else "_" for c in event_id)
            plot_filename = os.path.join(results_dir, f"{file_prefix}_pead_pred_{safe_event_id}.png")
            try: plt.savefig(plot_filename); print(f"Saved PEAD prediction plot to: {plot_filename}")
            except Exception as e: print(f"Error saving PEAD plot: {e}")
            plt.close(fig)
            plotted_count += 1

    def calculate_event_strategy_returns(self, holding_period: int = 20, entry_day: int = 0, return_col: str = 'ret') -> Optional[pl.DataFrame]:
        """
        Simulates a buy-and-hold strategy for each event using Polars and calculates returns,
        selecting exactly holding_period trading days post-entry.
        """
        print(f"  Calculating strategy returns (entry T{entry_day}, hold {holding_period} trading days)...")
        if self.data is None or return_col not in self.data.columns or 'days_to_announcement' not in self.data.columns:
            print("  Error: Data/required columns missing for strategy simulation.")
            return None

        df_sorted = self.data.sort(['event_id', 'date'])

        # Use row numbers for exact trading day count
        df_with_row_nr = df_sorted.with_columns(
            pl.int_range(0, pl.count()).over('event_id').alias('row_nr_in_event')
        )
        entry_point_rows = df_with_row_nr.filter(pl.col('days_to_announcement') == entry_day) \
                                        .select(['event_id', 'row_nr_in_event']) \
                                        .rename({'row_nr_in_event': 'entry_row_nr'})

        if entry_point_rows.is_empty():
             print(f"  Warning: No entry points found for entry_day = {entry_day}. Strategy returns will be empty.")
             return pl.DataFrame({'event_id': [], 'strategy_return': []}, schema={'event_id': pl.Utf8, 'strategy_return': pl.Float64})

        print(f"    Found {entry_point_rows.height} potential entry points.")
        df_with_entry_row = df_with_row_nr.join(entry_point_rows, on='event_id', how='left')

        # Filter for the 'holding_period' rows *after* the entry row
        holding_data = df_with_entry_row.filter(
            (pl.col('row_nr_in_event') > pl.col('entry_row_nr')) &
            (pl.col('row_nr_in_event') <= pl.col('entry_row_nr') + holding_period)
        )
        print(f"    Found {holding_data.height} rows within holding period windows (using row numbers).")
        if holding_data.is_empty():
            print("    Warning: No data rows found within any holding period window (using row numbers).")
            return pl.DataFrame({'event_id': [], 'strategy_return': []}, schema={'event_id': pl.Utf8, 'strategy_return': pl.Float64})

        # Calculate compound return, ensuring exactly holding_period days
        strategy_returns_agg = holding_data.group_by('event_id').agg(
            (pl.col(return_col).fill_null(0) + 1).product().alias("prod_ret_factor"),
            pl.count().alias("holding_days_count")
        ).filter(
            pl.col('holding_days_count') == holding_period # Crucial filter
        ).with_columns(
             (pl.col('prod_ret_factor') - 1).alias('strategy_return')
        ).select(['event_id', 'strategy_return'])

        print(f"    Found {strategy_returns_agg.height} events with exactly {holding_period} holding days after entry.")
        if strategy_returns_agg.is_empty():
             print(f"    Warning: No events had exactly {holding_period} trading days available after the entry point.")

        return strategy_returns_agg

    def analyze_event_sharpe_ratio(self, results_dir: str, file_prefix: str = "earnings", holding_period: int = 20, entry_day: int = 0, risk_free_rate: float = 0.0):
        """Calculates and analyzes the Sharpe Ratio of a simple event-based strategy using Polars."""
        print(f"\n--- Analyzing Event Strategy Sharpe Ratio (Entry: T{entry_day}, Hold: {holding_period} trading days) ---")
        strategy_returns_df = self.calculate_event_strategy_returns(holding_period=holding_period, entry_day=entry_day)

        if strategy_returns_df is None or strategy_returns_df.height < 2:
            num_returns = strategy_returns_df.height if strategy_returns_df is not None else 0
            print(f"Error: Insufficient valid returns ({num_returns}). Cannot calculate Sharpe.")
            return None
        if 'strategy_return' not in strategy_returns_df.columns:
            print("Error: 'strategy_return' column not found after calculation.")
            return None

        print(f"Using returns from {strategy_returns_df.height} events for Sharpe calculation.")

        stats = strategy_returns_df.select([
            pl.mean('strategy_return').alias('mean_ret'),
            pl.std('strategy_return').alias('std_ret')
        ]).row(0)
        mean_ret = stats[0] if stats[0] is not None else np.nan
        std_ret = stats[1] if stats[1] is not None else np.nan

        sharpe = np.nan
        if np.isnan(std_ret) or std_ret == 0:
             print("Warning: Standard deviation of returns is zero or NaN. Sharpe ratio is undefined.")
        else:
             period_rf = (1 + risk_free_rate)**(holding_period/252) - 1 if risk_free_rate != 0 else 0
             sharpe = (mean_ret - period_rf) / (std_ret + 1e-9)

        print(f"  Mean Return: {mean_ret:.4%}, Std Dev: {std_ret:.4%}, Period Sharpe: {sharpe:.4f}")

        csv_filename = os.path.join(results_dir, f"{file_prefix}_strategy_returns_h{holding_period}_e{entry_day}.csv")
        try:
            strategy_returns_df.write_csv(csv_filename)
            print(f"Saved strategy returns to: {csv_filename}")
        except Exception as e: print(f"Error saving returns data: {e}")

        fig, ax = plt.subplots(figsize=(10, 6))
        returns_pd = strategy_returns_df['strategy_return'].to_pandas()
        sns.histplot(returns_pd.dropna(), bins=30, kde=True, ax=ax)
        ax.axvline(mean_ret, color='red', linestyle='--', label=f'Mean ({mean_ret:.2%})')
        ax.set_title(f'Distribution of {holding_period}-Trading-Day Strategy Returns (Entry T{entry_day}) - Earnings')
        ax.set_xlabel(f'{holding_period}-Day Return'); ax.set_ylabel('Frequency'); ax.legend()
        plt.tight_layout()
        plot_filename = os.path.join(results_dir, f"{file_prefix}_strategy_returns_hist_h{holding_period}_e{entry_day}.png")
        try: plt.savefig(plot_filename); print(f"Saved returns histogram to: {plot_filename}")
        except Exception as e: print(f"Error saving histogram: {e}")
        plt.close(fig)

        return {'mean_return': mean_ret, 'std_dev_return': std_ret, 'period_sharpe': sharpe, 'num_events': strategy_returns_df.height}


    def analyze_volatility_spikes(self, results_dir: str, file_prefix: str = "earnings", window: int = 5, min_periods: int = 3, pre_days: int = 30, post_days: int = 30, baseline_window=(-60, -11), event_window=(-2, 2)):
        """Calculates, plots, and saves rolling volatility and event vs baseline comparison using Polars."""
        print(f"\n--- Analyzing Rolling Volatility (Window={window} rows) using Polars ---")
        if self.data is None or 'ret' not in self.data.columns or 'days_to_announcement' not in self.data.columns:
            print("Error: Data/required columns missing."); return None

        df = self.data.sort(['event_id', 'date'])

        df = df.with_columns(
            pl.col('ret').rolling_std(window_size=window, min_periods=min_periods)
              .over('event_id').alias('rolling_vol')
        )
        df = df.with_columns(
            (pl.col('rolling_vol') * np.sqrt(252) * 100).alias('annualized_vol')
        )

        aligned_vol = df.group_by('days_to_announcement').agg(
            pl.mean('annualized_vol').alias('avg_annualized_vol')
        ).filter(
            (pl.col('days_to_announcement') >= -pre_days) &
            (pl.col('days_to_announcement') <= post_days)
        ).sort('days_to_announcement').drop_nulls()

        # Plotting & Saving Rolling Volatility
        if not aligned_vol.is_empty():
            aligned_vol_pd = aligned_vol.to_pandas().set_index('days_to_announcement')
            fig1, ax1 = plt.subplots(figsize=(12, 6))
            aligned_vol_pd['avg_annualized_vol'].plot(kind='line', marker='.', linestyle='-', ax=ax1)
            ax1.axvline(0, color='red', linestyle='--', lw=1, label='Announcement Day')
            ax1.set_title(f'Average Rolling Volatility Around Earnings Announcement (Window={window} rows)')
            ax1.set_xlabel('Days Relative to Announcement'); ax1.set_ylabel('Avg. Annualized Volatility (%)')
            ax1.legend(); ax1.grid(True, alpha=0.5); plt.tight_layout()
            plot_filename_vol = os.path.join(results_dir, f"{file_prefix}_volatility_rolling_{window}d.png")
            csv_filename_vol = os.path.join(results_dir, f"{file_prefix}_volatility_rolling_{window}d_data.csv")
            try: plt.savefig(plot_filename_vol); print(f"Saved rolling vol plot to: {plot_filename_vol}")
            except Exception as e: print(f"Error saving plot: {e}")
            try: aligned_vol_pd.to_csv(csv_filename_vol); print(f"Saved rolling vol data to: {csv_filename_vol}")
            except Exception as e: print(f"Error saving data: {e}")
            plt.close(fig1)
        else:
            print("No data for rolling volatility plot.")

        # Compare Event vs Baseline Volatility
        print("    Calculating event vs baseline volatility...")
        baseline_filter = (
            (pl.col('days_to_announcement') >= baseline_window[0]) &
            (pl.col('days_to_announcement') <= baseline_window[1])
        )
        event_filter = (
            (pl.col('days_to_announcement') >= event_window[0]) &
            (pl.col('days_to_announcement') <= event_window[1])
        )

        try:
            vol_comp = df.group_by('event_id').agg([
                pl.col('ret').filter(baseline_filter).std().alias('vol_baseline'),
                pl.col('ret').filter(baseline_filter).count().alias('n_baseline'),
                pl.col('ret').filter(event_filter).std().alias('vol_event'),
                pl.col('ret').filter(event_filter).count().alias('n_event'),
            ]).filter(
                 (pl.col('n_baseline') >= min_periods) &
                 (pl.col('n_event') >= min_periods) &
                 (pl.col('vol_baseline').is_not_null()) &
                 (pl.col('vol_baseline') > 1e-9) &
                 (pl.col('vol_event').is_not_null())
            )
        except Exception as agg_err:
             print(f"    Error during volatility comparison aggregation: {agg_err}")
             traceback.print_exc()
             vol_comp = pl.DataFrame(schema={'event_id': pl.Utf8, 'vol_baseline': pl.Float64, 'n_baseline': pl.UInt32, 'vol_event': pl.Float64, 'n_event': pl.UInt32})

        if not vol_comp.is_empty():
            vol_ratios_df = vol_comp.with_columns(
                (pl.col('vol_event') / pl.col('vol_baseline')).alias('volatility_ratio')
            )
            avg_r = vol_ratios_df['volatility_ratio'].mean()
            med_r = vol_ratios_df['volatility_ratio'].median()
            num_valid_ratios = vol_ratios_df.height
            print(f"\nVolatility Spike (Event: {event_window}, Baseline: {baseline_window}): Avg Ratio={avg_r:.4f}, Median Ratio={med_r:.4f} ({num_valid_ratios} events)")

            csv_filename_ratio = os.path.join(results_dir, f"{file_prefix}_volatility_ratios.csv")
            try:
                vol_ratios_df.select(['event_id', 'volatility_ratio']).write_csv(csv_filename_ratio)
                print(f"Saved vol ratios data to: {csv_filename_ratio}")
            except Exception as e: print(f"Error saving vol ratios: {e}")

            fig2, ax2 = plt.subplots(figsize=(8, 5))
            ratios_pd = vol_ratios_df['volatility_ratio'].to_pandas()
            sns.histplot(ratios_pd.dropna(), bins=30, kde=True, ax=ax2)
            ax2.axvline(1, c='grey', ls='--', label='Ratio = 1')
            ax2.axvline(avg_r, c='red', ls=':', label=f'Mean ({avg_r:.2f})')
            ax2.set_title('Distribution of Volatility Ratios (Event / Baseline)'); ax2.set_xlabel('Volatility Ratio'); ax2.set_ylabel('Frequency'); ax2.legend()
            plt.tight_layout()
            plot_filename_hist = os.path.join(results_dir, f"{file_prefix}_volatility_ratio_hist.png")
            try: plt.savefig(plot_filename_hist); print(f"Saved vol ratio hist plot: {plot_filename_hist}")
            except Exception as e: print(f"Error saving hist: {e}")
            plt.close(fig2)
        else:
            print("\nCould not calculate volatility ratios (insufficient valid data or aggregation error).")

        return aligned_vol
    def plot_sharpe_ratio_time_series(self, results_dir: str, file_prefix: str = "earnings",
                                      entry_day: int = 0, holding_period: int = 20,
                                      time_grouping: str = 'year', risk_free_rate: float = 0.0):
        """
        Calculates and plots Sharpe ratio over time for an 'all buy' earnings strategy.
        
        Parameters:
        results_dir (str): Directory to save results
        file_prefix (str): Prefix for saved files
        entry_day (int): Day relative to announcement to enter positions
        holding_period (int): Holding period in trading days
        time_grouping (str): How to group time periods ('year', 'quarter', 'month')
        risk_free_rate (float): Annualized risk-free rate for Sharpe calculation
        
        Returns:
        pl.DataFrame: DataFrame containing Sharpe ratios for each time period
        """
        print(f"\n--- Analyzing Time Series of Sharpe Ratios (Entry: T{entry_day}, Hold: {holding_period} days) ---")
        if self.data is None:
            print("Error: Data not loaded.") 
            return None
            
        # Calculate strategy returns
        strategy_returns_df = self.calculate_event_strategy_returns(
            holding_period=holding_period, 
            entry_day=entry_day
        )
        
        if strategy_returns_df is None or strategy_returns_df.is_empty():
            print(f"Error: Not enough data for Entry T{entry_day}, Hold {holding_period}")
            return None
        
        # Join returns with event dates
        event_dates = self.data.filter(pl.col('days_to_announcement') == entry_day).select(['event_id', 'Announcement Date'])
        event_dates = event_dates.unique(subset=['event_id'], keep='first')
        
        returns_with_dates = strategy_returns_df.join(
            event_dates, 
            on='event_id', 
            how='inner'
        )
        
        if returns_with_dates.is_empty():
            print("Error: Failed to join strategy returns with announcement dates.")
            return None
        
        # Extract time period components
        returns_with_dates = returns_with_dates.with_columns([
            pl.col('Announcement Date').dt.year().alias('year'),
            pl.col('Announcement Date').dt.quarter().alias('quarter'),
            pl.col('Announcement Date').dt.month().alias('month')
        ])
        
        # Create time period label based on grouping
        if time_grouping == 'quarter':
            returns_with_dates = returns_with_dates.with_columns(
                (pl.col('year').cast(pl.Utf8) + "-Q" + pl.col('quarter').cast(pl.Utf8)).alias('time_period')
            )
        elif time_grouping == 'month':
            returns_with_dates = returns_with_dates.with_columns(
                (pl.col('year').cast(pl.Utf8) + "-" + pl.col('month').cast(pl.Utf8).str.zfill(2)).alias('time_period')
            )
        else:  # Default to year
            returns_with_dates = returns_with_dates.with_columns(
                pl.col('year').cast(pl.Utf8).alias('time_period')
            )
        
        # Calculate Sharpe ratio for each time period
        periods = []
        sharpes = []
        counts = []
        mean_returns = []
        std_returns = []
        
        for period in returns_with_dates['time_period'].unique().sort():
            period_returns = returns_with_dates.filter(pl.col('time_period') == period)
            if period_returns.height < 5:  # Skip periods with too few observations
                continue
                
            period_stats = period_returns.select([
                pl.mean('strategy_return').alias('mean_ret'),
                pl.std('strategy_return').alias('std_ret'),
                pl.count().alias('count')
            ]).row(0)
            
            mean_ret = period_stats[0] if period_stats[0] is not None else np.nan
            std_ret = period_stats[1] if period_stats[1] is not None else np.nan
            count = period_stats[2]
            
            # Annualize the Sharpe ratio
            sharpe = np.nan
            if not np.isnan(std_ret) and std_ret > 0:
                # Annualize returns and volatility
                ann_factor = 252 / holding_period
                ann_ret = (1 + mean_ret) ** ann_factor - 1
                ann_vol = std_ret * np.sqrt(ann_factor)
                sharpe = (ann_ret - risk_free_rate) / ann_vol
                
            periods.append(period)
            sharpes.append(sharpe)
            counts.append(count)
            mean_returns.append(mean_ret)
            std_returns.append(std_ret)
        
        # Create results DataFrame
        results_df = pl.DataFrame({
            'time_period': periods,
            'sharpe_ratio': sharpes,
            'mean_return': mean_returns,
            'std_dev': std_returns,
            'count': counts
        })
        
        # Plot time series
        try:
            results_pd = results_df.to_pandas()
            
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [3, 1]})
            
            # Sharpe ratio plot
            color = results_pd['sharpe_ratio'].map(lambda x: 'green' if x >= 0 else 'red')
            bars1 = ax1.bar(results_pd['time_period'], results_pd['sharpe_ratio'], color=color)
            ax1.set_title(f'Annualized Sharpe Ratio by {time_grouping.capitalize()} (Entry: T{entry_day}, Hold: {holding_period} days)')
            ax1.set_ylabel('Annualized Sharpe Ratio')
            ax1.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            
            # Add overall average line
            avg_sharpe = results_pd['sharpe_ratio'].mean()
            ax1.axhline(y=avg_sharpe, color='blue', linestyle='--', alpha=0.7, label=f'Average: {avg_sharpe:.2f}')
            ax1.legend()
            
            # Rotate x-axis labels if many periods
            if len(periods) > 8:
                plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')
            
            # Add value labels if fewer periods
            if len(periods) <= 15:
                for bar in bars1:
                    height = bar.get_height()
                    if not np.isnan(height):
                        ax1.annotate(f'{height:.2f}',
                                    xy=(bar.get_x() + bar.get_width() / 2, height),
                                    xytext=(0, 3 if height >= 0 else -10),
                                    textcoords="offset points",
                                    ha='center', va='bottom' if height >= 0 else 'top')
            
            # Sample count plot
            bars2 = ax2.bar(results_pd['time_period'], results_pd['count'], color='lightblue')
            ax2.set_title('Number of Events per Period')
            ax2.set_ylabel('Count')
            
            if len(periods) > 8:
                plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')
                
            plt.tight_layout()
            plot_filename = os.path.join(results_dir, f"{file_prefix}_sharpe_time_series_{time_grouping}.png")
            plt.savefig(plot_filename)
            print(f"Saved Sharpe ratio time series plot to: {plot_filename}")
            plt.close(fig)
            
            # Also create a rolling Sharpe ratio plot if there are enough periods
            if len(periods) >= 4:
                fig, ax = plt.subplots(figsize=(14, 6))
                
                # Calculate rolling averages
                roll_window = min(4, len(periods))
                results_pd['rolling_sharpe'] = results_pd['sharpe_ratio'].rolling(window=roll_window).mean()
                
                # Plot
                ax.plot(results_pd['time_period'], results_pd['sharpe_ratio'], 'o-', label='Period Sharpe')
                ax.plot(results_pd['time_period'], results_pd['rolling_sharpe'], 'r-', linewidth=2, 
                       label=f'{roll_window}-Period Rolling Avg')
                ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
                ax.axhline(y=avg_sharpe, color='blue', linestyle='--', alpha=0.7, label=f'Overall Avg: {avg_sharpe:.2f}')
                
                ax.set_title(f'Annualized Sharpe Ratio and Rolling Average (Entry: T{entry_day}, Hold: {holding_period} days)')
                ax.set_ylabel('Annualized Sharpe Ratio')
                ax.legend()
                
                if len(periods) > 8:
                    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
                    
                plt.tight_layout()
                roll_plot_filename = os.path.join(results_dir, f"{file_prefix}_sharpe_rolling_{time_grouping}.png")
                plt.savefig(roll_plot_filename)
                print(f"Saved rolling Sharpe ratio plot to: {roll_plot_filename}")
                plt.close(fig)
            
        except Exception as e:
            print(f"Error creating Sharpe ratio time series plots: {e}")
            traceback.print_exc()
        
        # Save results to CSV
        csv_filename = os.path.join(results_dir, f"{file_prefix}_sharpe_time_series_{time_grouping}.csv")
        try:
            results_df.write_csv(csv_filename)
            print(f"Saved Sharpe ratio time series data to: {csv_filename}")
        except Exception as e:
            print(f"Error saving Sharpe ratio data: {e}")
        
        return results_df
    
    def analyze_event_window_sharpe(self, results_dir: str, file_prefix: str = "earnings", 
                               event_window: Tuple[int, int] = (-2, 2), 
                               risk_free_rate: float = 0.0):
        """
        Calculates Sharpe ratio for a strategy that buys at the start of the event window
        and sells at the end of the event window for each earnings announcement.
        
        Parameters:
        results_dir (str): Directory to save results
        file_prefix (str): Prefix for saved files
        event_window (Tuple[int, int]): Days relative to announcement (start, end)
        risk_free_rate (float): Annualized risk-free rate for Sharpe calculation
        
        Returns:
        Dict: Dictionary containing Sharpe ratio and related statistics
        """
        print(f"\n--- Analyzing Event Window Sharpe Ratio (Window: {event_window}) ---")
        if self.data is None or 'ret' not in self.data.columns:
            print("Error: Data not loaded or missing return column.")
            return None
        
        # Filter data to include only days within the event window
        event_data = self.data.filter(
            (pl.col('days_to_announcement') >= event_window[0]) & 
            (pl.col('days_to_announcement') <= event_window[1])
        )
        
        if event_data.is_empty():
            print(f"Error: No data found within event window {event_window}.")
            return None
        
        # Calculate cumulative return for each event
        # First sort data by event_id and date
        event_data = event_data.sort(['event_id', 'date'])
        
        # Group by event_id and calculate cumulative return for each event
        event_returns = event_data.group_by('event_id').agg(
            # Calculate cumulative product of (1 + return)
            ((pl.col('ret').fill_null(0) + 1).product() - 1).alias('event_return'),
            # Count days to ensure we have complete window
            pl.count().alias('days_count'),
            # Get announcement date
            pl.first('Announcement Date').alias('announcement_date'),
            # Get ticker
            pl.first('ticker').alias('ticker')
        )
        
        # Filter to include only events with complete data (all days in window)
        window_size = event_window[1] - event_window[0] + 1
        complete_events = event_returns.filter(pl.col('days_count') == window_size)
        
        if complete_events.is_empty() or complete_events.height < 10:
            print(f"Warning: Insufficient events with complete {window_size}-day window data.")
            if not complete_events.is_empty():
                print(f"Found only {complete_events.height} complete events.")
            return None
        
        # Calculate statistics
        stats = complete_events.select([
            pl.mean('event_return').alias('mean_return'),
            pl.std('event_return').alias('std_dev'),
            pl.count().alias('event_count')
        ]).row(0)
        
        mean_return = stats[0] if stats[0] is not None else np.nan
        std_dev = stats[1] if stats[1] is not None else np.nan
        event_count = stats[2]
        
        # Calculate annualized figures
        trading_days_per_year = 252
        ann_factor = trading_days_per_year / window_size
        
        ann_return = (1 + mean_return) ** ann_factor - 1
        ann_vol = std_dev * np.sqrt(ann_factor)
        
        # Calculate Sharpe ratio
        sharpe_ratio = np.nan
        if not np.isnan(ann_vol) and ann_vol > 0:
            sharpe_ratio = (ann_return - risk_free_rate) / ann_vol
        
        # Print results
        print("\nEvent Window Strategy Results:")
        print(f"  Window: {event_window[0]} to {event_window[1]} days relative to announcement")
        print(f"  Number of events: {event_count}")
        print(f"  Average return per event: {mean_return:.4%}")
        print(f"  Standard deviation: {std_dev:.4%}")
        print(f"  Annualized return: {ann_return:.4%}")
        print(f"  Annualized volatility: {ann_vol:.4%}")
        print(f"  Annualized Sharpe ratio: {sharpe_ratio:.4f}")
        
        # Also calculate by year
        complete_events = complete_events.with_columns(
            pl.col('announcement_date').dt.year().alias('year')
        )
        
        yearly_stats = complete_events.group_by('year').agg([
            pl.mean('event_return').alias('mean_return'),
            pl.std('event_return').alias('std_dev'),
            pl.count().alias('event_count')
        ]).sort('year')
        
        # Calculate yearly Sharpe ratios
        yearly_stats = yearly_stats.with_columns([
            ((1 + pl.col('mean_return')) ** ann_factor - 1).alias('ann_return'),
            (pl.col('std_dev') * np.sqrt(ann_factor)).alias('ann_vol')
        ]).with_columns(
            ((pl.col('ann_return') - risk_free_rate) / pl.col('ann_vol')).alias('sharpe_ratio')
        )
        
        # Create plots
        try:
            # Distribution of returns
            fig, ax = plt.subplots(figsize=(10, 6))
            returns_pd = complete_events['event_return'].to_pandas()
            sns.histplot(returns_pd.dropna(), bins=30, kde=True, ax=ax)
            ax.axvline(mean_return, color='red', linestyle='--', label=f'Mean ({mean_return:.2%})')
            ax.set_title(f'Distribution of {window_size}-Day Event Window Returns')
            ax.set_xlabel('Return')
            ax.set_ylabel('Frequency')
            ax.legend()
            
            plt.tight_layout()
            hist_filename = os.path.join(results_dir, f"{file_prefix}_event_window_return_dist.png")
            plt.savefig(hist_filename)
            print(f"Saved return distribution plot to: {hist_filename}")
            plt.close(fig)
            
            # Yearly Sharpe ratios
            yearly_stats_pd = yearly_stats.to_pandas()
            
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [3, 1]})
            
            # Sharpe ratio by year
            bars = ax1.bar(yearly_stats_pd['year'].astype(str), yearly_stats_pd['sharpe_ratio'], color=yearly_stats_pd['sharpe_ratio'].map(lambda x: 'green' if x >= 0 else 'red'))
            ax1.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            ax1.axhline(y=sharpe_ratio, color='blue', linestyle='--', alpha=0.7, label=f'Overall: {sharpe_ratio:.2f}')
            ax1.set_title(f'Annualized Sharpe Ratio by Year (Event Window: {event_window[0]} to {event_window[1]})')
            ax1.set_ylabel('Sharpe Ratio')
            ax1.legend()
            
            # Add value labels
            for bar in bars:
                height = bar.get_height()
                if not np.isnan(height):
                    ax1.annotate(f'{height:.2f}',
                                xy=(bar.get_x() + bar.get_width() / 2, height),
                                xytext=(0, 3 if height >= 0 else -10),
                                textcoords="offset points",
                                ha='center', va='bottom' if height >= 0 else 'top')
            
            # Event count by year
            ax2.bar(yearly_stats_pd['year'].astype(str), yearly_stats_pd['event_count'], color='lightblue')
            ax2.set_title('Number of Events per Year')
            ax2.set_ylabel('Count')
            
            plt.tight_layout()
            year_filename = os.path.join(results_dir, f"{file_prefix}_event_window_sharpe_by_year.png")
            plt.savefig(year_filename)
            print(f"Saved yearly Sharpe ratio plot to: {year_filename}")
            plt.close(fig)
            
        except Exception as e:
            print(f"Error creating plots: {e}")
            traceback.print_exc()
        
        # Save results to CSV
        csv_filename = os.path.join(results_dir, f"{file_prefix}_event_window_yearly_stats.csv")
        try:
            yearly_stats.write_csv(csv_filename)
            print(f"Saved yearly statistics to: {csv_filename}")
        except Exception as e:
            print(f"Error saving yearly stats: {e}")
        
        # Also save individual event returns
        events_csv = os.path.join(results_dir, f"{file_prefix}_event_window_returns.csv")
        try:
            complete_events.write_csv(events_csv)
            print(f"Saved individual event returns to: {events_csv}")
        except Exception as e:
            print(f"Error saving event returns: {e}")
        
        return {
            'window': event_window,
            'mean_return': mean_return,
            'std_dev': std_dev,
            'ann_return': ann_return,
            'ann_vol': ann_vol,
            'sharpe_ratio': sharpe_ratio,
            'event_count': event_count
        }
    
    def calculate_average_sharpe(self, return_col: str = 'ret', event_window: Tuple[int, int] = (-2, 2), 
                           annualize: bool = True, risk_free_rate: float = 0.0) -> Dict[str, Any]:
        """
        Calculates the average Sharpe ratio across all earnings events.

        Parameters:
        return_col (str): Column name containing returns
        event_window (Tuple[int, int]): Days relative to announcement (start, end) to include
        annualize (bool): Whether to annualize the Sharpe ratio
        risk_free_rate (float): Annualized risk-free rate for Sharpe calculation

        Returns:
        Dict: Dictionary containing Sharpe ratio and related statistics
        """
        print(f"\n--- Calculating Average Sharpe Ratio (Window: {event_window}) ---")
        if self.data is None or return_col not in self.data.columns:
            print("Error: Data not loaded or missing return column.")
            return {'sharpe_ratio': np.nan, 'error': 'Data not available'}

        # Filter data to include only days within the event window
        event_data = self.data.filter(
            (pl.col('days_to_announcement') >= event_window[0]) & 
            (pl.col('days_to_announcement') <= event_window[1])
        )

        if event_data.is_empty():
            print(f"Error: No data found within event window {event_window}.")
            return {'sharpe_ratio': np.nan, 'error': 'No data in window'}

        # Calculate statistics directly on filtered returns
        stats = event_data.select([
            pl.mean(return_col).alias('mean_daily_return'),
            pl.std(return_col).alias('std_daily_return'),
            pl.count().alias('total_observations')
        ]).row(0)

        mean_daily_return = stats[0] if stats[0] is not None else np.nan
        std_daily_return = stats[1] if stats[1] is not None else np.nan
        total_observations = stats[2]

        # Calculate Sharpe ratio
        if np.isnan(mean_daily_return) or np.isnan(std_daily_return) or std_daily_return == 0:
            print("Warning: Cannot calculate Sharpe ratio (insufficient data or zero volatility).")
            return {
                'mean_return': mean_daily_return,
                'std_dev': std_daily_return,
                'observations': total_observations,
                'sharpe_ratio': np.nan,
                'error': 'Calculation failed'
            }

        # Calculate daily equivalent of risk-free rate if needed
        daily_rf = (1 + risk_free_rate) ** (1/252) - 1 if risk_free_rate > 0 else 0

        # Calculate daily Sharpe ratio
        daily_sharpe = (mean_daily_return - daily_rf) / std_daily_return

        # Annualize if requested
        if annualize:
            # Standard annualization formula: daily sharpe * sqrt(252)
            sharpe_ratio = daily_sharpe * np.sqrt(252)
            ann_return = (1 + mean_daily_return) ** 252 - 1
            ann_vol = std_daily_return * np.sqrt(252)
        else:
            sharpe_ratio = daily_sharpe
            ann_return = mean_daily_return
            ann_vol = std_daily_return

        # Collect event count information
        # Fix the error by using n_unique() instead of unique().height
        unique_events = event_data['event_id'].n_unique()

        # Print results
        period_desc = "Annualized" if annualize else "Daily"
        print(f"\nAverage {period_desc} Sharpe Ratio Results:")
        print(f"  Window: {event_window[0]} to {event_window[1]} days relative to announcement")
        print(f"  Unique events: {unique_events}")
        print(f"  Total observations: {total_observations}")
        print(f"  Mean return: {mean_daily_return:.6f} (daily)")
        print(f"  Standard deviation: {std_daily_return:.6f} (daily)")
        print(f"  {period_desc} return: {ann_return:.4%}")
        print(f"  {period_desc} volatility: {ann_vol:.4%}")
        print(f"  {period_desc} Sharpe ratio: {sharpe_ratio:.4f}")

        return {
            'window': event_window,
            'mean_daily_return': mean_daily_return,
            'std_daily_return': std_daily_return,
            'annualized_return': ann_return,
            'annualized_volatility': ann_vol,
            'sharpe_ratio': sharpe_ratio,
            'unique_events': unique_events,
            'total_observations': total_observations
        }
    
    def calculate_rolling_sharpe_timeseries(self, results_dir: str, file_prefix: str = "earnings",
                                      return_col: str = 'ret', 
                                      analysis_window: Tuple[int, int] = (-60, 60),
                                      sharpe_window: int = 5,
                                      annualize: bool = True, 
                                      risk_free_rate: float = 0.0):
        """
        Calculates a time series of rolling Sharpe ratios around earnings announcements.
        
        Parameters:
        results_dir (str): Directory to save results
        file_prefix (str): Prefix for saved files
        return_col (str): Column name containing returns
        analysis_window (Tuple[int, int]): Days relative to announcement to analyze (start, end)
        sharpe_window (int): Size of window for rolling Sharpe calculation in days
        annualize (bool): Whether to annualize the Sharpe ratio
        risk_free_rate (float): Annualized risk-free rate for Sharpe calculation
        
        Returns:
        pl.DataFrame: DataFrame containing Sharpe ratio time series
        """
        print(f"\n--- Calculating Rolling Sharpe Ratio Time Series (Analysis Window: {analysis_window}, Sharpe Window: {sharpe_window}) ---")
        if self.data is None or return_col not in self.data.columns:
            print("Error: Data not loaded or missing return column.")
            return None
        
        # Filter data to include only days within the extended analysis window
        analysis_data = self.data.filter(
            (pl.col('days_to_announcement') >= analysis_window[0]) & 
            (pl.col('days_to_announcement') <= analysis_window[1])
        )
        
        if analysis_data.is_empty():
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
            window_data = analysis_data.filter(
                (pl.col('days_to_announcement') >= window_start) & 
                (pl.col('days_to_announcement') <= window_end)
            )
            
            if window_data.height < sharpe_window // 2:  # Not enough data in window
                sharpe_results.append({
                    'days_to_announcement': center_day,
                    'mean_return': np.nan,
                    'std_dev': np.nan,
                    'sharpe_ratio': np.nan,
                    'num_observations': 0
                })
                continue
            
            # Calculate statistics for window
            stats = window_data.select([
                pl.mean(return_col).alias('mean_return'),
                pl.std(return_col).alias('std_dev'),
                pl.count().alias('num_observations')
            ]).row(0)
            
            mean_return = stats[0] if stats[0] is not None else np.nan
            std_dev = stats[1] if stats[1] is not None else np.nan
            num_obs = stats[2]
            
            # Calculate Sharpe ratio
            sharpe = np.nan
            if not np.isnan(mean_return) and not np.isnan(std_dev) and std_dev > 0:
                sharpe = (mean_return - daily_rf) / std_dev
                
                # Annualize if requested
                if annualize:
                    sharpe = sharpe * np.sqrt(252)
            
            sharpe_results.append({
                'days_to_announcement': center_day,
                'mean_return': mean_return,
                'std_dev': std_dev,
                'sharpe_ratio': sharpe,
                'num_observations': num_obs
            })
        
        # Convert results to DataFrame
        results_df = pl.DataFrame(sharpe_results)
        
        # Create a second DataFrame with event counts for each day
        event_counts = analysis_data.group_by('days_to_announcement').agg(
            pl.n_unique('event_id').alias('unique_events')
        ).sort('days_to_announcement')
        
        # Join results with event counts
        results_with_counts = results_df.join(
            event_counts, 
            on='days_to_announcement', 
            how='left'
        ).with_columns(
            pl.col('unique_events').fill_null(0)
        )
        
        # Plot the results
        try:
            results_pd = results_with_counts.to_pandas()
            
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [3, 1]})
            
            # Sharpe ratio plot
            ax1.plot(results_pd['days_to_announcement'], results_pd['sharpe_ratio'], 'b-', linewidth=2)
            ax1.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
            ax1.axvline(x=0, color='red', linestyle='--', linewidth=1.5, label='Announcement Day')
            
            # Add vertical lines at key periods if using extended window
            if analysis_window[0] <= -30 and analysis_window[1] >= 30:
                ax1.axvline(x=-30, color='green', linestyle=':', linewidth=1, label='Month Before')
                ax1.axvline(x=30, color='purple', linestyle=':', linewidth=1, label='Month After')
            
            # Highlight the event window commonly used for volatility analysis
            event_start, event_end = -2, 2  # Standard event window
            if analysis_window[0] <= event_start and analysis_window[1] >= event_end:
                ax1.axvspan(event_start, event_end, alpha=0.2, color='yellow', label='Event Window (-2 to +2)')
            
            ax1.set_title(f'Rolling Sharpe Ratio Around Earnings Announcements (Window Size: {sharpe_window} days)')
            ax1.set_xlabel('Days Relative to Announcement')
            ax1.set_ylabel('Sharpe Ratio')
            ax1.legend(loc='best')
            ax1.grid(True, alpha=0.3)
            
            # Calculate and plot rolling average of Sharpe
            if len(results_pd) >= 10:  # Need enough data for meaningful average
                window_size = min(10, len(results_pd) // 5)
                results_pd['rolling_avg'] = results_pd['sharpe_ratio'].rolling(window=window_size, center=True).mean()
                ax1.plot(results_pd['days_to_announcement'], results_pd['rolling_avg'], 'r-', 
                        linewidth=1.5, label=f'{window_size}-Day Rolling Avg')
                ax1.legend(loc='best')
            
            # Event count plot
            ax2.bar(results_pd['days_to_announcement'], results_pd['unique_events'], color='lightblue')
            ax2.axvline(x=0, color='red', linestyle='--', linewidth=1.5)
            ax2.set_title('Number of Events per Day')
            ax2.set_xlabel('Days Relative to Announcement')
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
            sns.heatmap(results_pd[['days_to_announcement', 'mean_return']].set_index('days_to_announcement').T, 
                       cmap='RdYlGn', center=0, ax=ax1, cbar_kws={'label': 'Mean Return'})
            ax1.set_title('Mean Return by Day Relative to Announcement')
            ax1.set_ylabel('')
            
            # Volatility heatmap-style plot
            sns.heatmap(results_pd[['days_to_announcement', 'std_dev']].set_index('days_to_announcement').T, 
                       cmap='YlOrRd', ax=ax2, cbar_kws={'label': 'Return Volatility'})
            ax2.set_title('Return Volatility by Day Relative to Announcement')
            ax2.set_ylabel('')
            ax2.set_xlabel('Days Relative to Announcement')
            
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
            results_with_counts.write_csv(csv_filename)
            print(f"Saved rolling Sharpe time series data to: {csv_filename}")
        except Exception as e:
            print(f"Error saving time series data: {e}")
        
        return results_with_counts