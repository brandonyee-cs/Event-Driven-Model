import polars as pl
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
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
    from models import TimeSeriesRidge, XGBoostDecileModel
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
        self.event_chunk_size = 5000  # Adjust based on memory capacity

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
             print(f"Warning: Parquet file not found: {stock_path}")
             return None
        except Exception as e:
            print(f"Warning: Error processing Parquet file {stock_path}: {e}")
            return None

    def load_data(self) -> Optional[pl.DataFrame]:
        """
        Load event dates (CSV) and stock data (PARQUET) using Polars,
        processing events in chunks to manage memory.
        """
        # --- Load Event Dates First ---
        try:
            print(f"Loading event dates from: {self.event_path} (CSV)")
            event_df_peek = pl.read_csv_batched(self.event_path, batch_size=1).next_batches(1)[0]
            ticker_col = self.ticker_col
            date_col = self.event_date_col
            
            if ticker_col not in event_df_peek.columns: 
                # Try to find ticker column if not found by name
                potential_ticker_cols = [c for c in event_df_peek.columns 
                                        if 'ticker' in c.lower() or 'symbol' in c.lower()]
                if potential_ticker_cols:
                    ticker_col = potential_ticker_cols[0]
                    print(f"Using '{ticker_col}' as ticker column")
                else:
                    raise ValueError(f"Ticker column '{self.ticker_col}' not found in event file.")
            
            if date_col not in event_df_peek.columns: 
                # Try to find date column if not found by name
                potential_date_cols = [c for c in event_df_peek.columns 
                                      if 'date' in c.lower() or 'day' in c.lower()]
                if potential_date_cols:
                    date_col = potential_date_cols[0]
                    print(f"Using '{date_col}' as event date column")
                else:
                    raise ValueError(f"Event date column '{self.event_date_col}' not found.")

            print(f"Using columns '{ticker_col}' (as ticker) and '{date_col}' (as Event Date) from event file.")
            event_data_raw = pl.read_csv(self.event_path, columns=[ticker_col, date_col], try_parse_dates=True)
            event_data_renamed = event_data_raw.rename({ticker_col: 'ticker', date_col: 'Event Date'})

            # --- Check/Correct Date Type ---
            if event_data_renamed['Event Date'].dtype == pl.Object or isinstance(event_data_renamed['Event Date'].dtype, pl.String):
                print("    'Event Date' read as Object/String, attempting str.to_datetime...")
                event_data_processed = event_data_renamed.with_columns([
                     pl.col('Event Date').str.to_datetime(strict=False).cast(pl.Datetime),  # Explicit parse and cast
                     pl.col('ticker').cast(pl.Utf8).str.to_uppercase()
                 ])
            elif isinstance(event_data_renamed['Event Date'].dtype, (pl.Date, pl.Datetime)):
                 print("    'Event Date' already parsed as Date/Datetime.")
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

            print("\n--- Sample Parsed Events ---")
            print(events.head(5))
            print("-" * 35 + "\n")

            print(f"Found {n_total_events} unique events (Ticker-Date pairs).")
            if events.is_empty(): raise ValueError("No valid events found.")

        except FileNotFoundError: raise FileNotFoundError(f"Event file not found: {self.event_path}")
        except Exception as e: raise ValueError(f"Error processing event file {self.event_path}: {e}")

        # --- Process Events in Chunks ---
        processed_chunks = []
        num_chunks = (n_total_events + self.event_chunk_size - 1) // self.event_chunk_size
        print(f"Processing events in {num_chunks} chunk(s) of size {self.event_chunk_size}...")

        for i in range(num_chunks):
            start_idx = i * self.event_chunk_size
            end_idx = min((i + 1) * self.event_chunk_size, n_total_events)
            event_chunk = events.slice(start_idx, end_idx - start_idx)
            print(f"--- Processing event chunk {i+1}/{num_chunks} ({event_chunk.height} events) ---")

            chunk_tickers = event_chunk['ticker'].unique()
            min_event_date = event_chunk['Event Date'].min()
            max_event_date = event_chunk['Event Date'].max()
            buffer = pl.duration(days=self.window_days + 1)  # Add buffer for safety
            required_min_date = min_event_date - buffer
            required_max_date = max_event_date + buffer

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
                        print(f"    Warning: Skipping {stock_path} as essential date or ticker column mapping not found.")
                        continue  # Skip file if essential keys aren't mapped

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
                 continue  # Skip to next chunk if no stock data found

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
                pl.col('Event Date').cast(pl.Datetime)
            ])
            merged_chunk = stock_data_chunk.join(
                event_chunk, on='ticker', how='inner'
            )
            print(f"    Merged chunk rows: {merged_chunk.height}")
            if merged_chunk.is_empty():
                print(f"    Warning: Merge resulted in empty data for chunk {i+1}. Check ticker matching.")
                continue  # Skip chunk

            # --- Calculate relative days and filter window FOR THE CHUNK ---
            processed_chunk = merged_chunk.with_columns(
                (pl.col('date') - pl.col('Event Date')).dt.total_days().cast(pl.Int32).alias('days_to_event')
            ).filter(
                (pl.col('days_to_event') >= -self.window_days) &
                (pl.col('days_to_event') <= self.window_days)
            )
            print(f"    Rows after window filter ({self.window_days} days): {processed_chunk.height}")

            if processed_chunk.is_empty():
                print(f"    Warning: No data found within event window for chunk {i+1}.")
                continue

            # --- Add final identifiers ---
            processed_chunk = processed_chunk.with_columns([
                (pl.col('days_to_event') == 0).cast(pl.Int8).alias('is_event_date'),
                (pl.col("ticker") + "_" + pl.col("Event Date").dt.strftime('%Y%m%d')).alias('event_id')
            ])

            # Select necessary columns
            stock_cols = stock_data_chunk.columns
            event_cols = event_chunk.columns
            derived_cols = ['days_to_event', 'is_event_date', 'event_id']
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
            return None  # Return None if no chunks succeeded

        print("\nConcatenating processed chunks...")
        combined_data = pl.concat(processed_chunks, how='vertical').sort(['ticker', 'Event Date', 'date'])
        print(f"Final dataset shape: {combined_data.shape}")
        mem_usage_mb = combined_data.estimated_size("mb")
        print(f"Final DataFrame memory usage: {mem_usage_mb:.2f} MB")

        if combined_data.is_empty():
             print("Warning: Final combined data is empty after chunk processing.")
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
            .otherwise(None)  # Set to null if price is 0 or null
            .alias('future_ret')
        ).drop('future_price')

        print(f"'future_ret' created. Non-null: {df.filter(pl.col('future_ret').is_not_null()).height}")
        return df

    def calculate_features(self, df: pl.DataFrame, price_col: str = 'prc', return_col: str = 'ret',
                           volume_col: str = 'vol', fit_categorical: bool = False) -> pl.DataFrame:
        """Calculate features for event analysis using Polars. Robust to missing optional columns."""
        print("Calculating event features (Polars)...")
        required = ['event_id', price_col, return_col, 'Event Date', 'date', 'days_to_event']
        missing = [col for col in required if col not in df.columns]
        if missing: raise ValueError(f"Missing required columns for feature calculation: {missing}")

        has_volume = volume_col in df.columns
        if not has_volume: print(f"Info: Volume column '{volume_col}' not found. Volume features skipped.")

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
             .then(pl.ln(pl.col(price_col) / shifted_price_log))
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
        df = df.with_columns(pl.col('pre_event_ret_30d').fill_null(0))
        current_features.append('pre_event_ret_30d')

        self.feature_names = sorted(list(set(current_features)))
        print(f"Calculated {len(self.feature_names)} raw event features.")

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
        print("Extracting features (X) and target (y) as NumPy...")
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


class EventAnalysis:
    def __init__(self, data_loader: EventDataLoader, feature_engineer: EventFeatureEngineer):
        """Analysis class for Event data using Polars."""
        self.data_loader = data_loader
        self.feature_engineer = feature_engineer
        self.data: Optional[pl.DataFrame] = None
        # Store NumPy arrays for ML models
        self.X_train_np: Optional[np.ndarray] = None
        self.X_test_np: Optional[np.ndarray] = None
        self.y_train_np: Optional[np.ndarray] = None
        self.y_test_np: Optional[np.ndarray] = None
        self.train_data: Optional[pl.DataFrame] = None  # Store processed train split DF
        self.test_data: Optional[pl.DataFrame] = None  # Store processed test split DF
        self.final_feature_names: Optional[List[str]] = None  # Store final feature names after processing
        self.models: Dict[str, Any] = {}  # Standard models (Ridge, XGBDecile)
        self.results: Dict[str, Dict] = {}  # Standard model results

    def load_and_prepare_data(self, run_feature_engineering: bool = True):
        """Load and optionally prepare data for event analysis using Polars."""
        print("--- Loading Event Data (Polars) ---")
        self.data = self.data_loader.load_data()  # Uses chunking internally now
        if self.data is None or self.data.is_empty():
            raise RuntimeError("Data loading failed or resulted in empty dataset.")

        if run_feature_engineering:
             if self.data is None or self.data.is_empty():
                  raise RuntimeError("Cannot run feature engineering on empty data.")
             print("\n--- Creating Target Variable (Polars) ---")
             self.data = self.feature_engineer.create_target(self.data)
             print("\n--- Calculating Features (Polars) ---")
             self.data = self.feature_engineer.calculate_features(self.data, fit_categorical=False)

        print("\n--- Event Data Loading/Preparation Complete ---")
        return self.data

    def train_models(self, test_size=0.2, time_split_column='Event Date'):
        """Split data, process features/target, and train models using Polars/NumPy."""
        if self.data is None: raise RuntimeError("Run load_and_prepare_data() first.")
        if time_split_column not in self.data.columns: raise ValueError(f"Time split column '{time_split_column}' not found.")
        if 'event_id' not in self.data.columns: raise ValueError("'event_id' required.")
        if 'future_ret' not in self.data.columns:
             print("ML Prep: Target variable 'future_ret' not found. Creating...")
             self.data = self.feature_engineer.create_target(self.data)
        if not self.feature_engineer.feature_names:
             print("ML Prep: Features not calculated. Calculating...")
             self.data = self.feature_engineer.calculate_features(self.data, fit_categorical=False)

        print(f"\n--- Splitting Event Data (Train/Test based on {time_split_column}) ---")
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

        print("\nFitting FeatureEngineer components (Imputer) on Training Data...")
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

        print("\n--- Training Models ---")
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

        print("\n--- All Model Training Complete ---")
        return self.models

    def evaluate_models(self) -> Dict[str, Dict]:
        """Evaluate all trained models on the test set using NumPy arrays."""
        print("\n--- Evaluating Models ---")
        if self.X_test_np is None or self.y_test_np is None or self.X_test_np.shape[0]==0:
             print("Test data (NumPy) unavailable/empty."); return {}
        if not self.final_feature_names:
             print("Final feature names not available. Cannot evaluate.")
             return {}

        print("\n--- Model Evaluation ---")
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

        print("\n--- Evaluation Complete ---")
        return self.results

    # --- Plotting and Analysis Methods ---
    def plot_feature_importance(self, results_dir: str, file_prefix: str = "event", model_name: str = 'TimeSeriesRidge', top_n: int = 20):
        """Plot feature importance and save the plot. Uses model coefficients/importances."""
        print(f"\n--- Plotting Feature Importance for {model_name} ---")
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
        ax.set_title(f'Top {top_n} Features by Importance ({model_name})')
        ax.set_xlabel('Importance Score'); ax.set_ylabel('Feature')
        plt.tight_layout()
        plot_filename = os.path.join(results_dir, f"{file_prefix}_feat_importance_{model_name}.png")
        try: plt.savefig(plot_filename); print(f"Saved feature importance plot to: {plot_filename}")
        except Exception as e: print(f"Error saving feature importance plot: {e}")
        plt.close(fig)
        return pl.DataFrame({'Feature': feature_names, 'Importance': importances}).sort('Importance', descending=True)

    def plot_predictions_for_event(self, results_dir: str, event_id: str, file_prefix: str = "event", model_name: str = 'TimeSeriesRidge'):
        """Plot actual daily returns and model's predicted future returns using Polars data and save plot."""
        print(f"\n--- Plotting Predictions for Event: {event_id} ({model_name}) ---")
        if model_name not in self.models: print(f"Error: Model '{model_name}' not found."); return None
        if self.data is None: print("Error: Data not loaded."); return None
        model = self.models[model_name]

        event_data_full = self.data.filter(pl.col('event_id') == event_id).sort('date')
        if event_data_full.is_empty(): print(f"Error: No data for event_id '{event_id}'."); return None

        ticker = event_data_full['ticker'][0]; event_date = event_data_full['Event Date'][0]

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
        event_date_pd = pd.Timestamp(event_date)

        fig, ax = plt.subplots(figsize=(14, 6))
        ax.plot(event_data_full_pd['date'], event_data_full_pd['ret'], '.-', label='Actual Daily Return', alpha=0.6, color='blue')
        ax.scatter(event_data_pred_pd['date'], event_data_pred_pd['predicted_future_ret'], color='red', label=f'Predicted {self.feature_engineer.prediction_window}-day Future Ret', s=15, zorder=5, alpha=0.8)
        ax.axvline(x=event_date_pd, color='g', linestyle='--', label='Event Date')
        ax.set_title(f"{ticker} ({event_id}) - Daily Returns & Predicted Future Returns ({model_name})")
        ax.set_ylabel("Return"); ax.set_xlabel("Date"); ax.legend(); ax.grid(True, alpha=0.3)
        plt.tight_layout()

        safe_event_id = "".join(c if c.isalnum() else "_" for c in event_id)
        plot_filename = os.path.join(results_dir, f"{file_prefix}_pred_vs_actual_{safe_event_id}_{model_name}.png")
        try: plt.savefig(plot_filename); print(f"Saved prediction plot to: {plot_filename}")
        except Exception as e: print(f"Error saving prediction plot: {e}")
        plt.close(fig)
        return event_data_pred

    def find_sample_event_ids(self, n=5):
        """Find sample event identifiers from Polars data."""
        if self.data is None or 'event_id' not in self.data.columns: return []
        unique_events = self.data['event_id'].unique().head(n)
        return unique_events.to_list()

    def calculate_event_strategy_returns(self, holding_period: int = 20, entry_day: int = 0, return_col: str = 'ret') -> Optional[pl.DataFrame]:
        """
        Simulates a buy-and-hold strategy for each event using Polars and calculates returns,
        selecting exactly holding_period trading days post-entry.
        """
        print(f"  Calculating strategy returns (entry T{entry_day}, hold {holding_period} trading days)...")
        if self.data is None or return_col not in self.data.columns or 'days_to_event' not in self.data.columns:
            print("  Error: Data/required columns missing for strategy simulation.")
            return None

        df_sorted = self.data.sort(['event_id', 'date'])

        # Use row numbers for exact trading day count
        df_with_row_nr = df_sorted.with_columns(
            pl.int_range(0, pl.count()).over('event_id').alias('row_nr_in_event')
        )
        entry_point_rows = df_with_row_nr.filter(pl.col('days_to_event') == entry_day) \
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
            pl.col('holding_days_count') == holding_period  # Crucial filter
        ).with_columns(
             (pl.col('prod_ret_factor') - 1).alias('strategy_return')
        ).select(['event_id', 'strategy_return'])

        print(f"    Found {strategy_returns_agg.height} events with exactly {holding_period} holding days after entry.")
        if strategy_returns_agg.is_empty():
             print(f"    Warning: No events had exactly {holding_period} trading days available after the entry point.")

        return strategy_returns_agg

    def analyze_volatility_spikes(self, results_dir: str, file_prefix: str = "event", window: int = 5, min_periods: int = 3, pre_days: int = 30, post_days: int = 30, baseline_window=(-60, -11), event_window=(-2, 2)):
        """Calculates, plots, and saves rolling volatility and event vs baseline comparison using Polars."""
        print(f"\n--- Analyzing Rolling Volatility (Window={window} rows) using Polars ---")
        if self.data is None or 'ret' not in self.data.columns or 'days_to_event' not in self.data.columns:
            print("Error: Data/required columns missing."); return None

        df = self.data.sort(['event_id', 'date'])

        df = df.with_columns(
            pl.col('ret').rolling_std(window_size=window, min_periods=min_periods)
              .over('event_id').alias('rolling_vol')
        )
        df = df.with_columns(
            (pl.col('rolling_vol') * np.sqrt(252) * 100).alias('annualized_vol')
        )

        aligned_vol = df.group_by('days_to_event').agg(
            pl.mean('annualized_vol').alias('avg_annualized_vol')
        ).filter(
            (pl.col('days_to_event') >= -pre_days) &
            (pl.col('days_to_event') <= post_days)
        ).sort('days_to_event').drop_nulls()

        # Plotting & Saving Rolling Volatility
        if not aligned_vol.is_empty():
            aligned_vol_pd = aligned_vol.to_pandas().set_index('days_to_event')
            fig1, ax1 = plt.subplots(figsize=(12, 6))
            aligned_vol_pd['avg_annualized_vol'].plot(kind='line', marker='.', linestyle='-', ax=ax1)
            ax1.axvline(0, color='red', linestyle='--', lw=1, label='Event Day')
            ax1.set_title(f'Average Rolling Volatility Around Event (Window={window} rows)')
            ax1.set_xlabel('Days Relative to Event'); ax1.set_ylabel('Avg. Annualized Volatility (%)')
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
            (pl.col('days_to_event') >= baseline_window[0]) &
            (pl.col('days_to_event') <= baseline_window[1])
        )
        event_filter = (
            (pl.col('days_to_event') >= event_window[0]) &
            (pl.col('days_to_event') <= event_window[1])
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
    
    def calculate_rolling_sharpe_timeseries(self, results_dir: str, file_prefix: str = "event",
                                      return_col: str = 'ret', 
                                      analysis_window: Tuple[int, int] = (-60, 60),
                                      sharpe_window: int = 5,
                                      annualize: bool = True, 
                                      risk_free_rate: float = 0.0):
        """
        Calculates a time series of rolling Sharpe ratios around events.
        
        Parameters:
        results_dir (str): Directory to save results
        file_prefix (str): Prefix for saved files
        return_col (str): Column name containing returns
        analysis_window (Tuple[int, int]): Days relative to event to analyze (start, end)
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
            (pl.col('days_to_event') >= analysis_window[0]) & 
            (pl.col('days_to_event') <= analysis_window[1])
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
                (pl.col('days_to_event') >= window_start) & 
                (pl.col('days_to_event') <= window_end)
            )
            
            if window_data.height < sharpe_window // 2:  # Not enough data in window
                sharpe_results.append({
                    'days_to_event': center_day,
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
                'days_to_event': center_day,
                'mean_return': mean_return,
                'std_dev': std_dev,
                'sharpe_ratio': sharpe,
                'num_observations': num_obs
            })
        
        # Convert results to DataFrame
        results_df = pl.DataFrame(sharpe_results)
        
        # Create a second DataFrame with event counts for each day
        event_counts = analysis_data.group_by('days_to_event').agg(
            pl.n_unique('event_id').alias('unique_events')
        ).sort('days_to_event')
        
        # FIX: Ensure both DataFrames have the same data type for the join key
        results_df = results_df.with_columns(
            pl.col('days_to_event').cast(pl.Int32)
        )
        
        event_counts = event_counts.with_columns(
            pl.col('days_to_event').cast(pl.Int32)
        )
        
        # Join results with event counts
        results_with_counts = results_df.join(
            event_counts, 
            on='days_to_event', 
            how='left'
        ).with_columns(
            pl.col('unique_events').fill_null(0)
        )
        
        # Plot the results
        try:
            results_pd = results_with_counts.to_pandas()
            
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [3, 1]})
            
            # Sharpe ratio plot
            ax1.plot(results_pd['days_to_event'], results_pd['sharpe_ratio'], 'b-', linewidth=2)
            ax1.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
            ax1.axvline(x=0, color='red', linestyle='--', linewidth=1.5, label='Event Day')
            
            # Add vertical lines at key periods if using extended window
            if analysis_window[0] <= -30 and analysis_window[1] >= 30:
                ax1.axvline(x=-30, color='green', linestyle=':', linewidth=1, label='Month Before')
                ax1.axvline(x=30, color='purple', linestyle=':', linewidth=1, label='Month After')
            
            # Highlight the event window commonly used for volatility analysis
            event_start, event_end = -2, 2  # Standard event window
            if analysis_window[0] <= event_start and analysis_window[1] >= event_end:
                ax1.axvspan(event_start, event_end, alpha=0.2, color='yellow', label='Event Window (-2 to +2)')
            
            ax1.set_title(f'Rolling Sharpe Ratio Around Events (Window Size: {sharpe_window} days)')
            ax1.set_xlabel('Days Relative to Event')
            ax1.set_ylabel('Sharpe Ratio')
            ax1.legend(loc='best')
            ax1.grid(True, alpha=0.3)
            
            # Calculate and plot rolling average of Sharpe
            if len(results_pd) >= 10:  # Need enough data for meaningful average
                window_size = min(10, len(results_pd) // 5)
                results_pd['rolling_avg'] = results_pd['sharpe_ratio'].rolling(window=window_size, center=True).mean()
                ax1.plot(results_pd['days_to_event'], results_pd['rolling_avg'], 'r-', 
                        linewidth=1.5, label=f'{window_size}-Day Rolling Avg')
                ax1.legend(loc='best')
            
            # Event count plot
            ax2.bar(results_pd['days_to_event'], results_pd['unique_events'], color='lightblue')
            ax2.axvline(x=0, color='red', linestyle='--', linewidth=1.5)
            ax2.set_title('Number of Events per Day')
            ax2.set_xlabel('Days Relative to Event')
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
            sns.heatmap(results_pd[['days_to_event', 'mean_return']].set_index('days_to_event').T, 
                       cmap='RdYlGn', center=0, ax=ax1, cbar_kws={'label': 'Mean Return'})
            ax1.set_title('Mean Return by Day Relative to Event')
            ax1.set_ylabel('')
            
            # Volatility heatmap-style plot
            sns.heatmap(results_pd[['days_to_event', 'std_dev']].set_index('days_to_event').T, 
                       cmap='YlOrRd', ax=ax2, cbar_kws={'label': 'Return Volatility'})
            ax2.set_title('Return Volatility by Day Relative to Event')
            ax2.set_ylabel('')
            ax2.set_xlabel('Days Relative to Event')
            
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
    
    def calculate_sharpe_quantiles(self, results_dir: str, file_prefix: str = "event",
                              return_col: str = 'ret', 
                              analysis_window: Tuple[int, int] = (-60, 60),
                              lookback_window: int = 10,
                              quantiles: List[float] = [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95],
                              annualize: bool = True, 
                              risk_free_rate: float = 0.0):
        """
        Calculates and plots quantiles of Sharpe ratios across events for each day around events.

        Parameters:
        results_dir (str): Directory to save results
        file_prefix (str): Prefix for saved files
        return_col (str): Column name containing returns
        analysis_window (Tuple[int, int]): Days relative to event to analyze (start, end)
        lookback_window (int): Window size for calculating Sharpe ratios (# of days)
        quantiles (List[float]): List of quantiles to calculate
        annualize (bool): Whether to annualize the Sharpe ratio
        risk_free_rate (float): Annualized risk-free rate for Sharpe calculation

        Returns:
        pl.DataFrame: DataFrame containing Sharpe ratio quantiles by day
        """
        print(f"\n--- Calculating Sharpe Ratio Quantiles (Analysis Window: {analysis_window}, Lookback: {lookback_window}) ---")

        if self.data is None or return_col not in self.data.columns:
            print("Error: Data not loaded or missing return column.")
            return None

        # Filter data to include only days within the extended analysis window
        # We'll need extra days before for the lookback window calculations
        extended_start = analysis_window[0] - lookback_window
        analysis_data = self.data.filter(
            (pl.col('days_to_event') >= extended_start) & 
            (pl.col('days_to_event') <= analysis_window[1])
        )

        if analysis_data.is_empty():
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
            window_data = analysis_data.filter(
                (pl.col('days_to_event') >= window_start) & 
                (pl.col('days_to_event') <= window_end)
            )

            if window_data.is_empty():
                # Record empty result and continue if no data for this day
                empty_results = {"days_to_event": center_day, "event_count": 0}
                for q in quantiles:
                    empty_results[f"sharpe_q{int(q*100)}"] = None
                results_data.append(empty_results)
                continue
            
            # Get all unique event IDs for this window
            event_ids = window_data.get_column('event_id').unique()

            # Calculate Sharpe ratio for each event in this window
            event_sharpes = []
            valid_event_count = 0

            for event_id in event_ids:
                # Get data for this specific event in the window
                event_data = window_data.filter(pl.col('event_id') == event_id)

                # Need sufficient data points for a meaningful calculation
                if event_data.height < max(3, lookback_window // 3):
                    continue

                # Calculate mean return and std dev for this event
                mean_ret = event_data.get_column(return_col).mean()
                std_dev = event_data.get_column(return_col).std()

                # Calculate Sharpe ratio for this event
                if std_dev is not None and std_dev > 0:
                    sharpe = (mean_ret - daily_rf) / std_dev

                    # Annualize if requested
                    if annualize:
                        sharpe = sharpe * np.sqrt(252)

                    event_sharpes.append(sharpe)
                    valid_event_count += 1

            # Calculate quantiles if we have enough events
            day_results = {"days_to_event": center_day, "event_count": valid_event_count}

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
        results_df = pl.DataFrame(results_data)

        # Plot the results
        try:
            if 'event_count' in results_df.columns and not results_df.is_empty():
                results_pd = results_df.to_pandas().set_index('days_to_event')

                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [3, 1]})

                # Plot quantiles
                # Colors from light to dark
                colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(quantiles)))

                # Need to ensure all required columns exist
                expected_cols = [f"sharpe_q{int(q*100)}" for q in quantiles]
                missing_cols = set(expected_cols) - set(results_pd.columns)
                if missing_cols:
                    print(f"Warning: Missing columns in results: {missing_cols}")
                    # Filter expected_cols to only include available columns
                    expected_cols = [col for col in expected_cols if col in results_pd.columns]

                if expected_cols:  # If we have any valid columns
                    for i, q in enumerate(quantiles):
                        if f"sharpe_q{int(q*100)}" in results_pd.columns:
                            ax1.plot(results_pd.index, results_pd[f"sharpe_q{int(q*100)}"], 
                                    color=colors[i], linewidth=1.5, 
                                    label=f"{int(q*100)}th percentile", alpha=0.7)

                # Highlight median more strongly
                median_col = "sharpe_q50"
                if median_col in results_pd.columns:
                    ax1.plot(results_pd.index, results_pd[median_col], 
                           color='red', linewidth=2.5, 
                           label=f"Median (50th)", alpha=0.9)

                # Add reference lines
                ax1.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
                ax1.axvline(x=0, color='red', linestyle='--', linewidth=1.5, label='Event Day')

                # Add vertical lines at key periods
                if analysis_window[0] <= -30 and analysis_window[1] >= 30:
                    ax1.axvline(x=-30, color='green', linestyle=':', linewidth=1, label='Month Before')
                    ax1.axvline(x=30, color='purple', linestyle=':', linewidth=1, label='Month After')

                # Highlight the event window commonly used for volatility analysis
                event_start, event_end = -2, 2  # Standard event window
                if analysis_window[0] <= event_start and analysis_window[1] >= event_end:
                    ax1.axvspan(event_start, event_end, alpha=0.1, color='yellow', label='Event Window (-2 to +2)')

                ax1.set_title(f'Sharpe Ratio Quantiles Around Events (Lookback: {lookback_window} days)')
                ax1.set_xlabel('Days Relative to Event')
                ax1.set_ylabel('Sharpe Ratio')
                ax1.legend(loc='best')
                ax1.grid(True, alpha=0.3)

                # Event count plot
                if 'event_count' in results_pd.columns:
                    ax2.bar(results_pd.index, results_pd['event_count'], color='lightblue')
                    ax2.axvline(x=0, color='red', linestyle='--', linewidth=1.5)
                    ax2.set_title('Number of Events With Valid Sharpe Ratio')
                    ax2.set_xlabel('Days Relative to Event')
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
                    heatmap_data = results_pd.drop(columns=['event_count']).T

                    # Rename index to more readable percentile names
                    new_index = [f"{int(q*100)}th" for q in quantiles]
                    heatmap_data.index = new_index

                    fig, ax = plt.subplots(figsize=(16, 8))

                    sns.heatmap(heatmap_data, cmap='RdYlGn', center=0, 
                               ax=ax, cbar_kws={'label': 'Sharpe Ratio'})

                    ax.set_title('Sharpe Ratio Quantiles Heatmap')
                    ax.set_xlabel('Days Relative to Event')
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
            results_df.write_csv(csv_filename)
            print(f"Saved Sharpe ratio quantiles data to: {csv_filename}")
        except Exception as e:
            print(f"Error saving quantiles data: {e}")

        return results_df