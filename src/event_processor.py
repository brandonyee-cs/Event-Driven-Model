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
                 self._imputer_fitted = True # Mark as fitted even if empty to prevent error
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
             # Handles case where both numeric and categorical are empty
             # (e.g. df_with_target was empty, or X_numeric_np was empty and no categorical features)
             num_potential_cols = len(numeric_features) + len(categorical_cols_present)
             X_np = np.empty((0, num_potential_cols), dtype=np.float64)
             self.final_feature_names = numeric_features + categorical_cols_present


        y_np = y_pl.cast(pl.Float64).to_numpy()

        print(f"Final X NumPy shape: {X_np.shape}. y NumPy shape: {y_np.shape}. Using {len(self.final_feature_names)} features.")
        final_nan_count_all = np.isnan(X_np).sum()
        if final_nan_count_all > 0: warnings.warn(f"NaNs ({final_nan_count_all}) detected in the final combined feature matrix X!")

        return X_np, y_np, self.final_feature_names

class EventAnalysis:
    def __init__(self, data_loader: EventDataLoader, feature_engineer: EventFeatureEngineer):
        """
        Initialize EventAnalysis with data loader and feature engineer.

        Parameters:
        data_loader (EventDataLoader): Instance to load event and stock data
        feature_engineer (EventFeatureEngineer): Instance to engineer features
        """
        self.data_loader = data_loader
        self.feature_engineer = feature_engineer
        self.data = None
        self.models = {}

    def load_and_prepare_data(self, run_feature_engineering: bool = False) -> Optional[pl.DataFrame]:
        """
        Loads and prepares data using the data loader and optionally applies feature engineering.

        Parameters:
        run_feature_engineering (bool): Whether to run feature engineering

        Returns:
        pl.DataFrame: Prepared data or None if loading fails
        """
        try:
            # Load event and stock data using the available load_data method
            combined_data = self.data_loader.load_data()

            if combined_data is None:
                print("Error: Failed to load data from data loader.")
                return None

            if run_feature_engineering:
                # Create target for predictive modeling
                combined_data = self.feature_engineer.create_target(combined_data)
                # Calculate features for analysis
                combined_data = self.feature_engineer.calculate_features(combined_data)

            self.data = combined_data
            return combined_data
        except Exception as e:
            print(f"Error loading and preparing data: {e}")
            traceback.print_exc()
            return None

    def train_models(self, test_size: float = 0.2, time_split_column: str = "Event Date"):
        """
        Train machine learning models on the prepared data.

        Parameters:
        test_size (float): Proportion of data to use for testing
        time_split_column (str): Column containing dates for time-based splitting
        """
        print("Training models...")
        if self.data is None:
            print("Error: No data loaded. Call load_and_prepare_data first.")
            return

        try:
            # Extract features and target
            X, y, feature_names = self.feature_engineer.get_features_target(self.data, fit_imputer=True)

            if X.shape[0] == 0 or y.shape[0] == 0:
                print("Error: No valid features or target extracted.")
                return

            print(f"Training models on {X.shape[0]} samples with {X.shape[1]} features.")

            # Create time-based split if time column provided
            if time_split_column in self.data.columns:
                # Sort by time and split
                # Ensure we only use rows that ended up in X and y (after drop_nulls on target)
                valid_indices_for_split = self.data.with_row_count().filter(pl.col('future_ret').is_not_null())['row_nr'].to_numpy()

                dates_for_split = self.data.filter(pl.col('future_ret').is_not_null()).select(time_split_column).to_numpy().flatten()

                if len(dates_for_split) != X.shape[0]:
                     print(f"Warning: Mismatch in date count ({len(dates_for_split)}) and X samples ({X.shape[0]}). Falling back to random split.")
                     from sklearn.model_selection import train_test_split
                     X_train, X_test, y_train, y_test = train_test_split(
                         X, y, test_size=test_size, random_state=42
                     )
                else:
                    sorted_indices_relative_to_X = np.argsort(dates_for_split)
                    split_idx = int(len(sorted_indices_relative_to_X) * (1 - test_size))
                    train_indices_in_X = sorted_indices_relative_to_X[:split_idx]
                    test_indices_in_X = sorted_indices_relative_to_X[split_idx:]

                    X_train, X_test = X[train_indices_in_X], X[test_indices_in_X]
                    y_train, y_test = y[train_indices_in_X], y[test_indices_in_X]
                    print(f"Time-based split: Train {X_train.shape[0]}, Test {X_test.shape[0]}")

            else:
                # Random split
                from sklearn.model_selection import train_test_split
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=42
                )
                print(f"Random split: Train {X_train.shape[0]}, Test {X_test.shape[0]}")


            # Initialize and train models
            # from src.models import TimeSeriesRidge, XGBoostDecileModel # Already imported at top

            # TimeSeriesRidge model
            tsridge_model = TimeSeriesRidge(alpha=0.1, lambda2=0.5)
            tsridge_model.fit(X_train, y_train)
            self.models['TimeSeriesRidge'] = {
                'model': tsridge_model,
                'X_train': X_train, 'y_train': y_train,
                'X_test': X_test, 'y_test': y_test,
                'feature_names': feature_names
            }

            # XGBoost model
            xgb_params = {
                'n_estimators': 100,
                'max_depth': 5,
                'learning_rate': 0.1,
                'objective': 'reg:squarederror',
                'random_state': 42
            }
            xgb_model = XGBoostDecileModel(weight=0.7, xgb_params=xgb_params)
            xgb_model.fit(X_train, y_train)
            self.models['XGBoostDecile'] = {
                'model': xgb_model,
                'X_train': X_train, 'y_train': y_train,
                'X_test': X_test, 'y_test': y_test,
                'feature_names': feature_names
            }

            print(f"Successfully trained {len(self.models)} models.")

        except Exception as e:
            print(f"Error training models: {e}")
            traceback.print_exc()

    def evaluate_models(self) -> Dict[str, Any]:
        """
        Evaluate trained machine learning models and return performance metrics.

        Returns:
        Dict[str, Any]: Dictionary of model evaluation results
        """
        print("Evaluating models...")
        if not self.models:
            print("Error: No models trained. Call train_models first.")
            return {}

        results = {}

        try:
            for model_name, model_info in self.models.items():
                model = model_info['model']
                X_test = model_info['X_test']
                y_test = model_info['y_test']

                # Make predictions
                y_pred = model.predict(X_test)

                # Calculate metrics
                from sklearn.metrics import mean_squared_error, r2_score
                mse = mean_squared_error(y_test, y_pred)
                rmse = np.sqrt(mse)
                r2 = r2_score(y_test, y_pred)

                # Store results
                results[model_name] = {
                    'mse': mse,
                    'rmse': rmse,
                    'r2': r2,
                    'y_pred': y_pred,
                    'y_test': y_test
                }

                print(f"Model: {model_name}, RMSE: {rmse:.4f}, R²: {r2:.4f}")

            return results

        except Exception as e:
            print(f"Error evaluating models: {e}")
            traceback.print_exc()
            return {}

    def plot_feature_importance(self, results_dir: str, file_prefix: str, model_name: str):
        """
        Plot feature importance for a specified model using Matplotlib.

        Parameters:
        results_dir (str): Directory to save the plot
        file_prefix (str): Prefix for the saved file name
        model_name (str): Name of the model to plot feature importance for
        """
        print(f"Plotting feature importance for {model_name}...")
        if model_name not in self.models:
            print(f"Error: Model '{model_name}' not found in trained models.")
            return

        try:
            model_info = self.models[model_name]
            model = model_info['model']
            feature_names = model_info['feature_names']

            if model_name == 'TimeSeriesRidge':
                importances = np.abs(model.coef_)
                indices = np.argsort(importances)[::-1]
            elif model_name == 'XGBoostDecile':
                xgb_model_actual = model.xgb_model # Access the underlying XGBoost model
                importances = xgb_model_actual.feature_importances_
                indices = np.argsort(importances)[::-1]
            else:
                print(f"Error: Feature importance not implemented for model type '{model_name}'.")
                return

            sorted_importances = importances[indices]
            sorted_features = [feature_names[i] for i in indices]

            top_n = min(15, len(sorted_features))

            fig, ax = plt.subplots(figsize=(10, 6)) # Adjusted for typical Matplotlib aspect
            ax.barh(
                np.arange(top_n),
                sorted_importances[:top_n][::-1], # Reverse for horizontal bar plot
                align='center'
            )
            ax.set_yticks(np.arange(top_n))
            ax.set_yticklabels(sorted_features[:top_n][::-1]) # Reverse for horizontal bar plot
            ax.set_xlabel('Importance')
            ax.set_ylabel('Feature')
            ax.set_title(f'Feature Importance - {model_name}')
            plt.tight_layout()

            # Save plot
            plot_filename = os.path.join(results_dir, f"{file_prefix}_feat_importance_{model_name}.png")
            try:
                plt.savefig(plot_filename, dpi=200)
                print(f"Saved feature importance plot to: {plot_filename}")
            except Exception as e:
                print(f"Warning: Could not save plot image {plot_filename}: {e}")
            plt.close(fig)

        except Exception as e:
            print(f"Error plotting feature importance: {e}")
            traceback.print_exc()

    def find_sample_event_ids(self, n: int = 3) -> List[str]:
        """
        Find sample event IDs for prediction visualization.

        Parameters:
        n (int): Number of sample events to return

        Returns:
        List[str]: List of event IDs
        """
        print(f"Finding {n} sample event IDs...")
        if self.data is None:
            print("Error: No data loaded.")
            return []

        try:
            # Get unique event IDs
            event_ids = self.data.get_column('event_id').unique().to_list()

            if len(event_ids) <= n:
                return event_ids

            # Sample n event IDs
            import random
            random.seed(42)  # For reproducibility
            sample_ids = random.sample(event_ids, n)

            return sample_ids

        except Exception as e:
            print(f"Error finding sample event IDs: {e}")
            return []

    def plot_predictions_for_event(self, results_dir: str, event_id: str, file_prefix: str, model_name: str):
        """
        Plot predictions for a specific event using a trained model with Matplotlib.

        Parameters:
        results_dir (str): Directory to save the plot
        event_id (str): Event ID to plot predictions for
        file_prefix (str): Prefix for the saved file name
        model_name (str): Name of the model to use for predictions
        """
        print(f"Plotting predictions for event {event_id} with {model_name}...")
        if self.data is None:
            print("Error: No data loaded.")
            return

        if model_name not in self.models:
            print(f"Error: Model '{model_name}' not found in trained models.")
            return

        try:
            # Filter data for the specified event
            event_data_pl = self.data.filter(pl.col('event_id') == event_id)

            if event_data_pl.is_empty():
                print(f"Error: No data found for event ID '{event_id}'.")
                return

            # Sort by days_to_event
            event_data_pl = event_data_pl.sort('days_to_event')

            # Extract features and target
            X, y, _ = self.feature_engineer.get_features_target(event_data_pl, fit_imputer=False) # Use pre-fitted imputer

            if X.shape[0] == 0:
                print(f"Error: No valid features extracted for event ID '{event_id}'.")
                return

            # Make predictions
            model = self.models[model_name]['model']
            y_pred = model.predict(X)

            days = event_data_pl.get_column('days_to_event').to_numpy()
            # Use 'y' which is future_ret from get_features_target, already handled NaNs
            actual_returns_pct = y * 100 # y is already future_ret, convert to percentage
            predicted_returns_pct = y_pred * 100 # Convert to percentage

            fig, ax = plt.subplots(figsize=(12, 7)) # Adjusted for typical Matplotlib aspect

            ax.plot(days, actual_returns_pct, marker='o', linestyle='-', color='blue', label='Actual Returns (%)')
            ax.plot(days, predicted_returns_pct, marker='x', linestyle='--', color='red', label='Predicted Returns (%)')

            ax.axvline(x=0, color='green', linestyle='--', label='Event Day')
            ax.text(0.1, ax.get_ylim()[1] * 0.9, 'Event Day', color='green', ha='left')


            ticker = event_data_pl.get_column('ticker').head(1).item()
            event_date_val = event_data_pl.get_column('Event Date').head(1).item()
            if isinstance(event_date_val, datetime.datetime):
                event_date_str = event_date_val.strftime('%Y-%m-%d')
            else:
                event_date_str = str(event_date_val)


            ax.set_title(f'Return Predictions - {ticker} Event on {event_date_str} - {model_name}')
            ax.set_xlabel('Days Relative to Event')
            ax.set_ylabel('Returns (%)')
            ax.legend()
            ax.grid(True, linestyle=':', alpha=0.7)
            plt.tight_layout()

            # Save plot
            plot_filename = os.path.join(results_dir, f"{file_prefix}_pred_{model_name}_{event_id}.png")
            try:
                plt.savefig(plot_filename, dpi=200)
                print(f"Saved prediction plot to: {plot_filename}")
            except Exception as e:
                print(f"Warning: Could not save plot image {plot_filename}: {e}")
            plt.close(fig)

        except Exception as e:
            print(f"Error plotting predictions: {e}")
            traceback.print_exc()

    def calculate_rolling_sharpe_timeseries(self,
        results_dir: str,
        file_prefix: str = "event",
        return_col: str = 'ret',
        analysis_window: Tuple[int, int] = (-60, 60),
        sharpe_window: int = 5,
        annualize: bool = True,
        risk_free_rate: float = 0.0
    ) -> Optional[pl.DataFrame]:
        """
        Calculates a time series of rolling Sharpe ratios around events.
        Uses Matplotlib for plotting.
        """
        print(f"\n--- Calculating Rolling Sharpe Ratio Time Series ---")
        print(f"Analysis Window: {analysis_window}, Sharpe Window: {sharpe_window} days")

        if self.data is None or return_col not in self.data.columns or 'days_to_event' not in self.data.columns:
            print("Error: Data not loaded or missing required columns (return_col, days_to_event).")
            return None

        if sharpe_window < 3:
            print("Warning: Sharpe window too small (<3 days). Setting to 3.")
            sharpe_window = 3

        daily_rf = (1 + risk_free_rate) ** (1/252) - 1 if annualize and risk_free_rate > 0 else 0
        extended_start = analysis_window[0] - sharpe_window
        analysis_data = self.data.filter(
            (pl.col('days_to_event') >= extended_start) &
            (pl.col('days_to_event') <= analysis_window[1])
        ).sort(['event_id', 'days_to_event'])

        if analysis_data.is_empty():
            print(f"Error: No data found within extended analysis window [{extended_start}, {analysis_window[1]}].")
            return None

        all_days = pl.DataFrame({'days_to_event': range(analysis_window[0], analysis_window[1] + 1)})
        sample_returns = analysis_data.select(pl.col(return_col)).sample(n=min(100, analysis_data.height))
        avg_abs_return = sample_returns.select(pl.all().mean().abs()).item() # Corrected mean for Polars

        returns_in_pct = avg_abs_return > 0.05
        print(f"Detected returns format: {'Percentage form' if returns_in_pct else 'Decimal form'} (avg abs: {avg_abs_return:.4f})")

        if returns_in_pct:
            print("Converting percentage returns to decimal form for Sharpe calculation")
            analysis_data = analysis_data.with_columns((pl.col(return_col) / 100).alias('decimal_return'))
            return_col_for_calc = 'decimal_return'
        else:
            return_col_for_calc = return_col

        analysis_data = analysis_data.with_columns(pl.col(return_col_for_calc).clip(-0.05, 0.05).alias('clipped_return'))

        sharpe_data = []
        for day in range(analysis_window[0], analysis_window[1] + 1):
            window_start_day = day - sharpe_window
            window_end_day = day - 1

            window_data_for_day = analysis_data.filter(
                (pl.col('days_to_event') >= window_start_day) &
                (pl.col('days_to_event') <= window_end_day)
            )

            if window_data_for_day.is_empty():
                sharpe_data.append({'days_to_event': day, 'mean_return': None, 'std_return': None, 'sharpe_ratio': None, 'event_count': 0, 'valid_returns': 0})
                continue

            event_sharpe = window_data_for_day.group_by('event_id').agg([
                pl.mean('clipped_return').alias('mean_return'),
                pl.std('clipped_return').alias('std_return'),
                pl.count().alias('window_count')
            ]).filter(
                (pl.col('window_count') >= max(3, sharpe_window // 2)) &
                (pl.col('std_return') > 1e-9) # Avoid division by zero or very small std
            )

            if event_sharpe.height > 0:
                event_sharpe = event_sharpe.with_columns(
                    ((pl.col('mean_return') - daily_rf) / pl.col('std_return') *
                     (np.sqrt(252) if annualize else 1)).alias('sharpe_ratio')
                )
                event_sharpe = event_sharpe.with_columns(pl.col('sharpe_ratio').clip(-5, 5).alias('sharpe_ratio'))

                day_stats = {
                    'days_to_event': day,
                    'mean_return': event_sharpe['mean_return'].mean(),
                    'std_return': event_sharpe['std_return'].mean(),
                    'sharpe_ratio': event_sharpe['sharpe_ratio'].mean(),
                    'median_sharpe': event_sharpe['sharpe_ratio'].median(),
                    'event_count': event_sharpe.height,
                    'valid_returns': event_sharpe['window_count'].sum()
                }
            else:
                day_stats = {'days_to_event': day, 'mean_return': None, 'std_return': None, 'sharpe_ratio': None, 'median_sharpe': None, 'event_count': 0, 'valid_returns': 0}
            sharpe_data.append(day_stats)

        sharpe_df = pl.DataFrame(sharpe_data)
        sharpe_df = all_days.join(sharpe_df, on='days_to_event', how='left').sort('days_to_event')
        sharpe_df = sharpe_df.with_columns([pl.col('sharpe_ratio').interpolate(), pl.col('mean_return').interpolate(), pl.col('std_return').interpolate()])

        smooth_window = min(7, sharpe_window)
        sharpe_df = sharpe_df.with_columns(
            pl.col('sharpe_ratio').rolling_mean(window_size=smooth_window, min_periods=smooth_window // 2, center=True).alias('smooth_sharpe')
        )

        valid_days = sharpe_df.filter(pl.col('sharpe_ratio').is_not_null()).height
        if valid_days < (analysis_window[1] - analysis_window[0]) / 2:
            print(f"Warning: Only {valid_days} days have valid Sharpe ratios. Results may be unreliable.")

        sharpe_stats_dict = {
            'Min Sharpe': sharpe_df['sharpe_ratio'].min(), 'Max Sharpe': sharpe_df['sharpe_ratio'].max(),
            'Mean Sharpe': sharpe_df['sharpe_ratio'].mean(), 'Median Sharpe': sharpe_df['sharpe_ratio'].median()
        }
        print("Sharpe Ratio Summary Statistics:")
        for stat, value in sharpe_stats_dict.items():
            print(f"  {stat}: {value:.4f}" if value is not None else f"  {stat}: N/A")

        # Plotting with Matplotlib
        try:
            results_pd = sharpe_df.to_pandas()
            fig, ax = plt.subplots(figsize=(12, 7))

            y_min_data = results_pd['sharpe_ratio'].dropna().min()
            y_max_data = results_pd['sharpe_ratio'].dropna().max()

            if pd.isna(y_min_data) or pd.isna(y_max_data):
                 y_range_plot = [-3, 3] # Default if no data
            elif -5 <= y_min_data <= 5 and -5 <= y_max_data <= 5:
                y_range_plot = [-3, 3]
            else:
                padding = 0.1 * (y_max_data - y_min_data) if y_max_data != y_min_data else 0.5
                y_range_plot = [y_min_data - padding, y_max_data + padding]
                if y_range_plot[0] > 0: y_range_plot[0] = 0
                if y_range_plot[1] < 0: y_range_plot[1] = 0


            ax.plot(results_pd['days_to_event'], results_pd['sharpe_ratio'], color='blue', linewidth=1, alpha=0.3, label='Raw Sharpe Ratio')
            ax.plot(results_pd['days_to_event'], results_pd['smooth_sharpe'], color='red', linewidth=2, label=f'{smooth_window}-Day Smoothed')

            ax.axvline(x=0, color='red', linestyle='--', label='Event Day')
            ax.text(0.1, y_range_plot[1] * 0.9, 'Event Day', color='red', ha='left')


            if analysis_window[0] <= -30 and analysis_window[1] >= 30:
                ax.axvline(x=-30, color='green', linestyle=':', label='Month Before')
                ax.text(-29.9, y_range_plot[1] * 0.85, 'Month Before', color='green', ha='left')
                ax.axvline(x=30, color='purple', linestyle=':', label='Month After')
                ax.text(30.1, y_range_plot[1] * 0.85, 'Month After', color='purple', ha='left')


            ax.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
            ax.axvspan(-2, 2, color='yellow', alpha=0.2, label='Event Window')

            ax.set_title(f'Rolling Sharpe Ratio vs. Days to Event ({sharpe_window}-Day Window)')
            ax.set_xlabel('Days Relative to Event')
            ax.set_ylabel('Sharpe Ratio (Annualized)' if annualize else 'Sharpe Ratio')
            ax.legend()
            ax.grid(True, linestyle=':', alpha=0.7)
            ax.set_ylim(y_range_plot)
            ax.xaxis.set_major_locator(mticker.MultipleLocator(10))
            ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))

            annotation_text = f"Mean: {sharpe_stats_dict.get('Mean Sharpe', 0.0):.2f}, Median: {sharpe_stats_dict.get('Median Sharpe', 0.0):.2f}"
            ax.text(0.05, 0.95, annotation_text, transform=ax.transAxes, ha="left", va="top",
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", alpha=0.8))

            plot_filename = os.path.join(results_dir, f"{file_prefix}_rolling_sharpe_timeseries.png")
            try:
                plt.savefig(plot_filename, dpi=200, bbox_inches='tight')
                print(f"Saved rolling Sharpe time series plot to: {plot_filename}")
            except Exception as e:
                print(f"Warning: Could not save plot image {plot_filename}: {e}")
            plt.close(fig)

        except Exception as e:
            print(f"Error creating plot: {e}")
            traceback.print_exc()

        csv_filename = os.path.join(results_dir, f"{file_prefix}_rolling_sharpe_timeseries.csv")
        try:
            sharpe_df.write_csv(csv_filename)
            print(f"Saved rolling Sharpe time series data to: {csv_filename}")
        except Exception as e:
            print(f"Error saving data: {e}")

        print(f"Completed Sharpe ratio calculation. Valid days: {valid_days}/{sharpe_df.height}")
        return sharpe_df

    def calculate_sharpe_quantiles(self, results_dir: str, file_prefix: str = "event",
                          return_col: str = 'ret',
                          analysis_window: Tuple[int, int] = (-60, 60),
                          lookback_window: int = 10,
                          quantiles: List[float] = [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95],
                          annualize: bool = True,
                          risk_free_rate: float = 0.0):
        """
        Calculates Sharpe ratio quantiles for each day relative to events.
        Uses Matplotlib for plotting.
        """
        print(f"\n--- Calculating Sharpe Ratio Quantiles (Analysis Window: {analysis_window}, Lookback: {lookback_window}) ---")

        if self.data is None or return_col not in self.data.columns:
            print("Error: Data not loaded or missing return column.")
            return None

        daily_rf = (1 + risk_free_rate) ** (1/252) - 1 if annualize and risk_free_rate > 0 else 0
        extended_start = analysis_window[0] - lookback_window

        temp_analysis_data = self.data.filter(
            (pl.col('days_to_event') >= extended_start) &
            (pl.col('days_to_event') <= analysis_window[1])
        )
        sample_returns = temp_analysis_data.select(pl.col(return_col)).sample(n=min(100, temp_analysis_data.height))
        avg_abs_return = sample_returns.select(pl.all().mean().abs()).item()

        returns_in_pct = avg_abs_return > 0.05
        print(f"Detected returns format: {'Percentage form' if returns_in_pct else 'Decimal form'} (avg abs: {avg_abs_return:.4f})")

        if returns_in_pct:
            print("Converting percentage returns to decimal form for Sharpe calculation")
            analysis_data = self.data.filter(
                (pl.col('days_to_event') >= extended_start) &
                (pl.col('days_to_event') <= analysis_window[1])
            ).with_columns((pl.col(return_col) / 100).alias('decimal_return'))
            return_col_for_calc = 'decimal_return'
        else:
            analysis_data = self.data.filter(
                (pl.col('days_to_event') >= extended_start) &
                (pl.col('days_to_event') <= analysis_window[1])
            )
            return_col_for_calc = return_col

        analysis_data = analysis_data.with_columns(pl.col(return_col_for_calc).clip(-0.05, 0.05).alias('clipped_return'))

        if analysis_data.is_empty():
            print(f"Error: No data found within analysis window {analysis_window}.")
            return None

        days_range = list(range(analysis_window[0], analysis_window[1] + 1))
        sharpe_data = []
        print(f"Processing {len(days_range)} days...")

        batch_size = 10
        for batch_start_idx in range(0, len(days_range), batch_size):
            batch_days = days_range[batch_start_idx : batch_start_idx + batch_size]
            print(f"Processing days {batch_days[0]} to {batch_days[-1]} ({len(batch_days)} days)...")
            batch_results = []
            for day in batch_days:
                window_start_day = day - lookback_window
                window_end_day = day - 1

                window_data_for_day = analysis_data.filter(
                    (pl.col('days_to_event') >= window_start_day) &
                    (pl.col('days_to_event') <= window_end_day)
                )

                if window_data_for_day.is_empty():
                    empty_res = {"days_to_event": day, "event_count": 0}
                    for q_val in quantiles: empty_res[f"sharpe_q{int(q_val*100)}"] = None
                    batch_results.append(empty_res)
                    continue

                sharpe_by_event_df = window_data_for_day.group_by('event_id').agg([
                    pl.mean('clipped_return').alias('mean_ret'),
                    pl.std('clipped_return').alias('std_dev'),
                    pl.count().alias('n_obs')
                ]).filter(
                    (pl.col('n_obs') >= max(3, lookback_window // 3)) &
                    (pl.col('std_dev') > 1e-9)
                )

                valid_events_count = sharpe_by_event_df.height
                if valid_events_count >= 5:
                    sharpe_by_event_df = sharpe_by_event_df.with_columns(
                        ((pl.col('mean_ret') - daily_rf) / pl.col('std_dev') *
                         (np.sqrt(252) if annualize else 1)).alias('sharpe')
                    )
                    sharpe_by_event_df = sharpe_by_event_df.with_columns(pl.col('sharpe').clip(-10, 10).alias('sharpe'))

                    q_values_dict = {}
                    for q_val in quantiles:
                        q_res = sharpe_by_event_df.select(pl.col('sharpe').quantile(q_val, interpolation='linear')).item()
                        q_values_dict[f"sharpe_q{int(q_val*100)}"] = q_res
                    day_res = {"days_to_event": day, "event_count": valid_events_count, **q_values_dict}
                else:
                    day_res = {"days_to_event": day, "event_count": valid_events_count}
                    for q_val in quantiles: day_res[f"sharpe_q{int(q_val*100)}"] = None
                batch_results.append(day_res)
            sharpe_data.extend(batch_results)
            del window_data_for_day, sharpe_by_event_df, batch_results
            gc.collect()

        results_df = pl.DataFrame(sharpe_data)

        min_val_plot, max_val_plot = float('inf'), float('-inf')
        for q_val in quantiles:
            col_name = f"sharpe_q{int(q_val*100)}"
            if col_name in results_df.columns:
                col_min_val = results_df[col_name].drop_nulls().min()
                col_max_val = results_df[col_name].drop_nulls().max()
                if col_min_val is not None and col_min_val < min_val_plot: min_val_plot = col_min_val
                if col_max_val is not None and col_max_val > max_val_plot: max_val_plot = col_max_val

        if min_val_plot == float('inf'): min_val_plot = -5 # Default if no data
        if max_val_plot == float('-inf'): max_val_plot = 5  # Default if no data

        if min_val_plot < -5 or max_val_plot > 5:
            print(f"Note: Wide range of Sharpe values detected: [{min_val_plot:.2f}, {max_val_plot:.2f}]")

        # Plotting with Matplotlib
        try:
            results_pd = results_df.to_pandas()
            fig, ax = plt.subplots(figsize=(12, 7))

            for q_val in quantiles:
                col_name_plot = f"sharpe_q{int(q_val*100)}"
                if col_name_plot in results_pd.columns:
                    lw = 2 if q_val == 0.5 else 1
                    ls = 'solid' if q_val == 0.5 else 'dashed'
                    ax.plot(results_pd['days_to_event'], results_pd[col_name_plot], linewidth=lw, linestyle=ls, label=f'Q{int(q_val*100)}')

            ax.axvline(x=0, color='red', linestyle='--', label='Event Day')
            y_lim_ax = ax.get_ylim() # Get current y_lim after plotting lines to place text
            ax.text(0.1, y_lim_ax[1] * 0.9, 'Event Day', color='red', ha='left')


            if analysis_window[0] <= -30 and analysis_window[1] >= 30:
                ax.axvline(x=-30, color='green', linestyle=':', label='Month Before')
                ax.text(-29.9, y_lim_ax[1] * 0.85, 'Month Before', color='green', ha='left')
                ax.axvline(x=30, color='purple', linestyle=':', label='Month After')
                ax.text(30.1, y_lim_ax[1] * 0.85, 'Month After', color='purple', ha='left')

            ax.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
            event_start_plot, event_end_plot = -2, 2
            if analysis_window[0] <= event_start_plot and analysis_window[1] >= event_end_plot:
                ax.axvspan(event_start_plot, event_end_plot, color='yellow', alpha=0.2, label='Event Window')

            if min_val_plot < -10 or max_val_plot > 10:
                padding_plot = 0.1 * (max_val_plot - min_val_plot) if max_val_plot != min_val_plot else 1.0
                y_range_final = [min_val_plot - padding_plot, max_val_plot + padding_plot]
            elif min_val_plot < -5 or max_val_plot > 5:
                y_range_final = [-10, 10] # Clamped from [-20,20] for typical Sharpe
            else:
                y_range_final = [-5, 5]
            if y_range_final[0] > 0: y_range_final[0] = 0
            if y_range_final[1] < 0: y_range_final[1] = 0


            ax.set_title(f'Sharpe Ratio Quantiles Around Events (Lookback: {lookback_window} days)')
            ax.set_xlabel('Days Relative to Event')
            ax.set_ylabel('Sharpe Ratio' + (' (Annualized)' if annualize else ''))
            ax.legend(loc='best')
            ax.grid(True, linestyle=':', alpha=0.7)
            ax.set_ylim(y_range_final)
            ax.xaxis.set_major_locator(mticker.MultipleLocator(20))
            ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))

            annotation_text_plot = f"Sharpe range: [{min_val_plot:.2f}, {max_val_plot:.2f}]"
            ax.text(0.05, 0.95, annotation_text_plot, transform=ax.transAxes, ha="left", va="top",
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", alpha=0.8))

            plot_filename = os.path.join(results_dir, f"{file_prefix}_sharpe_quantiles.png")
            try:
                plt.savefig(plot_filename, dpi=200, bbox_inches='tight')
                print(f"Saved Sharpe quantiles plot to: {plot_filename}")
            except Exception as e:
                print(f"Warning: Could not save plot image {plot_filename}: {e}")
            plt.close(fig)

        except Exception as e:
            print(f"Error creating plot: {e}")
            traceback.print_exc()

        csv_filename = os.path.join(results_dir, f"{file_prefix}_sharpe_quantiles.csv")
        try:
            results_df.write_csv(csv_filename)
            print(f"Saved Sharpe quantiles data to: {csv_filename}")
        except Exception as e:
            print(f"Error saving quantiles data: {e}")
        return results_df

    def analyze_volatility_spikes(self,
                                 results_dir: str,
                                 file_prefix: str = "event",
                                 window: int = 5,
                                 min_periods: int = 3,
                                 pre_days: int = 30,
                                 post_days: int = 30,
                                 baseline_window=(-60, -11),
                                 event_window=(-2, 2)):
        """
        Calculates and plots rolling volatility around events using Matplotlib.
        """
        print(f"\n--- Analyzing Rolling Volatility (Window={window} days) ---")

        if self.data is None or 'ret' not in self.data.columns or 'days_to_event' not in self.data.columns:
            print("Error: Data/required columns missing.")
            return None

        df = self.data.filter(
            (pl.col('days_to_event') >= -pre_days) &
            (pl.col('days_to_event') <= post_days)
        ).sort(['event_id', 'date'])

        df = df.with_columns(
            pl.col('ret').clip(-0.5, 0.5).rolling_std(
                window_size=window,
                min_periods=min_periods
            ).over('event_id').alias('rolling_vol')
        )
        df = df.with_columns((pl.col('rolling_vol') * np.sqrt(252) * 100).alias('annualized_vol'))

        aligned_vol = df.group_by('days_to_event').agg([
            pl.mean('annualized_vol').alias('avg_annualized_vol'),
            pl.median('annualized_vol').alias('median_annualized_vol'),
            pl.count().alias('event_count') # count on original col before it might become null
        ]).sort('days_to_event')

        baseline_data = df.filter(
            (pl.col('days_to_event') >= baseline_window[0]) &
            (pl.col('days_to_event') <= baseline_window[1])
        )
        event_data = df.filter(
            (pl.col('days_to_event') >= event_window[0]) &
            (pl.col('days_to_event') <= event_window[1])
        )

        baseline_vol_val = baseline_data['annualized_vol'].mean()
        event_vol_val = event_data['annualized_vol'].mean()
        vol_change_pct_val = ((event_vol_val / baseline_vol_val) - 1) * 100 if baseline_vol_val is not None and baseline_vol_val > 0 else float('nan')

        print(f"Baseline volatility ({baseline_window}): {baseline_vol_val or 0:.2f}%")
        print(f"Event window volatility ({event_window}): {event_vol_val or 0:.2f}%")
        print(f"Volatility change: {vol_change_pct_val:+.2f}%" if not pd.isna(vol_change_pct_val) else "Volatility change: N/A")


        # Plotting with Matplotlib
        if not aligned_vol.is_empty():
            aligned_vol_pd = aligned_vol.to_pandas()
            fig, ax = plt.subplots(figsize=(12, 7))

            ax.plot(aligned_vol_pd['days_to_event'], aligned_vol_pd['avg_annualized_vol'], color='blue', linewidth=2, label='Avg. Annualized Volatility')

            if baseline_vol_val is not None:
                ax.axhline(y=baseline_vol_val, color='green', linestyle='--', label=f'Baseline Vol: {baseline_vol_val:.1f}%')
            ax.axvline(x=0, color='red', linestyle='--', label='Event Day')
            ax.text(0.1, ax.get_ylim()[1]*0.95, 'Event Day', color='red', ha='left')


            ax.axvspan(event_window[0], event_window[1], color='yellow', alpha=0.2, label='Event Window')

            ax.set_title(f'Average Rolling Volatility Around Event (Window={window} days)')
            ax.set_xlabel('Days Relative to Event')
            ax.set_ylabel('Avg. Annualized Volatility (%)')
            ax.legend()
            ax.grid(True, linestyle=':', alpha=0.7)
            ax.set_ylim(bottom=0) # Volatility is non-negative
            ax.xaxis.set_major_locator(mticker.MultipleLocator(10))

            if event_vol_val is not None and not pd.isna(vol_change_pct_val):
                 ax.annotate(f"Event vol: {event_vol_val:.1f}%\nChange: {vol_change_pct_val:+.1f}%",
                            xy=(event_window[1], event_vol_val), xycoords='data',
                            xytext=(event_window[1] + 5, event_vol_val + (ax.get_ylim()[1]*0.05) ), textcoords='data',
                            arrowprops=dict(arrowstyle="->", connectionstyle="arc3"),
                            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))


            plot_filename = os.path.join(results_dir, f"{file_prefix}_volatility_rolling_{window}d.png")
            try:
                plt.savefig(plot_filename, dpi=200, bbox_inches='tight')
                print(f"Saved rolling volatility plot to: {plot_filename}")
            except Exception as e:
                print(f"Warning: Could not save plot image {plot_filename}: {e}")
            plt.close(fig)

            csv_filename = os.path.join(results_dir, f"{file_prefix}_volatility_rolling_{window}d_data.csv")
            try:
                aligned_vol.write_csv(csv_filename)
                print(f"Saved rolling volatility data to: {csv_filename}")
            except Exception as e:
                print(f"Error saving data: {e}")
        else:
            print("No data for rolling volatility plot.")
        return aligned_vol

    def calculate_volatility_quantiles(self,
                                     results_dir: str,
                                     file_prefix: str = "event",
                                     return_col: str = 'ret',
                                     analysis_window: Tuple[int, int] = (-60, 60),
                                     lookback_window: int = 10,
                                     quantiles: List[float] = [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]):
        """
        Calculates volatility quantiles for each day relative to events using Matplotlib.
        """
        print(f"\n--- Calculating Volatility Quantiles (Analysis Window: {analysis_window}, Lookback: {lookback_window}) ---")

        if self.data is None or return_col not in self.data.columns:
            print("Error: Data not loaded or missing return column.")
            return None

        extended_start = analysis_window[0] - lookback_window
        analysis_data_filtered = self.data.filter(
            (pl.col('days_to_event') >= extended_start) &
            (pl.col('days_to_event') <= analysis_window[1])
        ).with_columns(pl.col(return_col).clip(-0.5, 0.5).alias('clipped_return'))

        if analysis_data_filtered.is_empty():
            print(f"Error: No data found within analysis window {analysis_window}.")
            return None

        days_range_list = list(range(analysis_window[0], analysis_window[1] + 1))
        vol_quantile_data = []
        print(f"Processing {len(days_range_list)} days...")

        batch_size = 10
        for batch_start_idx in range(0, len(days_range_list), batch_size):
            current_batch_days = days_range_list[batch_start_idx : batch_start_idx + batch_size]
            print(f"Processing days {current_batch_days[0]} to {current_batch_days[-1]} ({len(current_batch_days)} days)...")
            batch_results_list = []
            for day_val in current_batch_days:
                window_start_val = day_val - lookback_window
                window_end_val = day_val - 1

                current_window_data = analysis_data_filtered.filter(
                    (pl.col('days_to_event') >= window_start_val) &
                    (pl.col('days_to_event') <= window_end_val)
                )

                if current_window_data.is_empty():
                    empty_res_dict = {"days_to_event": day_val, "event_count": 0}
                    for q_item in quantiles: empty_res_dict[f"vol_q{int(q_item*100)}"] = None
                    batch_results_list.append(empty_res_dict)
                    continue

                vol_by_event_df = current_window_data.group_by('event_id').agg([
                    pl.std('clipped_return').alias('vol'),
                    pl.count().alias('n_obs')
                ]).filter(
                    (pl.col('n_obs') >= max(3, lookback_window // 3)) &
                    pl.col('vol').is_not_null() & (pl.col('vol') > 1e-9) # Ensure std_dev is positive
                )

                num_valid_events = vol_by_event_df.height
                if num_valid_events >= 5:
                    vol_by_event_df = vol_by_event_df.with_columns(
                        (pl.col('vol') * np.sqrt(252) * 100).alias('annualized_vol')
                    )
                    quantile_values_dict = {}
                    for q_item in quantiles:
                        q_result = vol_by_event_df.select(pl.col('annualized_vol').quantile(q_item, interpolation='linear')).item()
                        quantile_values_dict[f"vol_q{int(q_item*100)}"] = q_result
                    day_results_dict = {"days_to_event": day_val, "event_count": num_valid_events, **quantile_values_dict}
                else:
                    day_results_dict = {"days_to_event": day_val, "event_count": num_valid_events}
                    for q_item in quantiles: day_results_dict[f"vol_q{int(q_item*100)}"] = None
                batch_results_list.append(day_results_dict)
            vol_quantile_data.extend(batch_results_list)
            del current_window_data, vol_by_event_df, batch_results_list
            gc.collect()

        results_vol_df = pl.DataFrame(vol_quantile_data)

        # Plotting with Matplotlib
        try:
            results_vol_pd = results_vol_df.to_pandas()
            fig, ax = plt.subplots(figsize=(12, 7))

            for q_item in quantiles:
                col_name_str = f"vol_q{int(q_item*100)}"
                if col_name_str in results_vol_pd.columns:
                    line_w = 2 if q_item == 0.5 else 1
                    line_s = 'solid' if q_item == 0.5 else 'dashed'
                    ax.plot(results_vol_pd['days_to_event'], results_vol_pd[col_name_str], linewidth=line_w, linestyle=line_s, label=f'Q{int(q_item*100)}')

            ax.axvline(x=0, color='red', linestyle='--', label='Event Day')
            y_curr_lims = ax.get_ylim()
            ax.text(0.1, y_curr_lims[1] * 0.9, 'Event Day', color='red', ha='left')


            if analysis_window[0] <= -30 and analysis_window[1] >= 30:
                ax.axvline(x=-30, color='green', linestyle=':', label='Month Before')
                ax.text(-29.9, y_curr_lims[1] * 0.85, 'Month Before', color='green', ha='left')
                ax.axvline(x=30, color='purple', linestyle=':', label='Month After')
                ax.text(30.1, y_curr_lims[1] * 0.85, 'Month After', color='purple', ha='left')


            event_window_start, event_window_end = -2, 2
            if analysis_window[0] <= event_window_start and analysis_window[1] >= event_window_end:
                ax.axvspan(event_window_start, event_window_end, color='yellow', alpha=0.2, label='Event Window')

            ax.set_title(f'Volatility Quantiles Around Events (Lookback: {lookback_window} days)')
            ax.set_xlabel('Days Relative to Event')
            ax.set_ylabel('Annualized Volatility (%)')
            ax.legend(loc='best')
            ax.grid(True, linestyle=':', alpha=0.7)
            ax.set_ylim(bottom=0) # Volatility is non-negative
            ax.xaxis.set_major_locator(mticker.MultipleLocator(20))

            plot_filename = os.path.join(results_dir, f"{file_prefix}_volatility_quantiles.png")
            try:
                plt.savefig(plot_filename, dpi=200, bbox_inches='tight')
                print(f"Saved volatility quantiles plot to: {plot_filename}")
            except Exception as e:
                print(f"Warning: Could not save plot image {plot_filename}: {e}")
            plt.close(fig)

        except Exception as e:
            print(f"Error creating plot: {e}")
            traceback.print_exc()

        csv_filename = os.path.join(results_dir, f"{file_prefix}_volatility_quantiles.csv")
        try:
            results_vol_df.write_csv(csv_filename)
            print(f"Saved volatility quantiles data to: {csv_filename}")
        except Exception as e:
            print(f"Error saving quantiles data: {e}")
        return results_vol_df

    def analyze_mean_returns(self,
                           results_dir: str,
                           file_prefix: str = "event",
                           return_col: str = 'ret',
                           window: int = 5,
                           min_periods: int = 3,
                           pre_days: int = 60,
                           post_days: int = 60,
                           baseline_window=(-60, -11),
                           event_window=(-2, 2)):
        """
        Calculates and plots rolling mean returns around events using Matplotlib.
        """
        print(f"\n--- Analyzing Rolling Mean Returns (Window={window} days) ---")

        if self.data is None or return_col not in self.data.columns or 'days_to_event' not in self.data.columns:
            print("Error: Data/required columns missing.")
            return None

        df = self.data.filter(
            (pl.col('days_to_event') >= -pre_days) &
            (pl.col('days_to_event') <= post_days)
        ).with_columns(
            pl.col(return_col).clip(-0.5, 0.5).alias('clipped_return')
        ).sort(['event_id', 'date'])

        df = df.with_columns(
            pl.col('clipped_return').rolling_mean(
                window_size=window,
                min_periods=min_periods
            ).over('event_id').alias('rolling_mean_return')
        )
        df = df.with_columns((pl.col('rolling_mean_return') * 100).alias('rolling_mean_return_pct'))

        aligned_means = df.group_by('days_to_event').agg([
            pl.mean('rolling_mean_return_pct').alias('avg_rolling_mean_return'),
            pl.median('rolling_mean_return_pct').alias('median_rolling_mean_return'),
            pl.count().alias('event_count')
        ]).sort('days_to_event')

        baseline_data = df.filter(
            (pl.col('days_to_event') >= baseline_window[0]) &
            (pl.col('days_to_event') <= baseline_window[1])
        )
        event_data = df.filter(
            (pl.col('days_to_event') >= event_window[0]) &
            (pl.col('days_to_event') <= event_window[1])
        )

        baseline_ret_val = baseline_data['rolling_mean_return_pct'].mean()
        event_ret_val = event_data['rolling_mean_return_pct'].mean()
        ret_change_val = (event_ret_val - baseline_ret_val) if baseline_ret_val is not None and event_ret_val is not None else float('nan')

        print(f"Baseline return ({baseline_window}): {baseline_ret_val or 0:.2f}%")
        print(f"Event window return ({event_window}): {event_ret_val or 0:.2f}%")
        print(f"Return change: {ret_change_val:+.2f}%" if not pd.isna(ret_change_val) else "Return change: N/A")

        # Plotting with Matplotlib
        if not aligned_means.is_empty():
            aligned_means_pd = aligned_means.to_pandas()
            fig, ax = plt.subplots(figsize=(12, 7))

            ax.plot(aligned_means_pd['days_to_event'], aligned_means_pd['avg_rolling_mean_return'], color='blue', linewidth=2, label='Avg. Rolling Mean Return')

            if baseline_ret_val is not None:
                ax.axhline(y=baseline_ret_val, color='green', linestyle='--', label=f'Baseline Ret: {baseline_ret_val:.2f}%')
            ax.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
            ax.axvline(x=0, color='red', linestyle='--', label='Event Day')
            ax.text(0.1, ax.get_ylim()[1]*0.95, 'Event Day', color='red', ha='left')

            ax.axvspan(event_window[0], event_window[1], color='yellow', alpha=0.2, label='Event Window')

            ax.set_title(f'Average Rolling Mean Return Around Event (Window={window} days)')
            ax.set_xlabel('Days Relative to Event')
            ax.set_ylabel('Avg. Rolling Mean Return (%)')
            ax.legend()
            ax.grid(True, linestyle=':', alpha=0.7)
            ax.xaxis.set_major_locator(mticker.MultipleLocator(10))

            if event_ret_val is not None and not pd.isna(ret_change_val):
                ax.annotate(f"Event: {event_ret_val:.2f}%\nChange: {ret_change_val:+.2f}%",
                            xy=(event_window[1], event_ret_val), xycoords='data',
                            xytext=(event_window[1] + 5, event_ret_val + (ax.get_ylim()[1]*0.05 if event_ret_val > 0 else ax.get_ylim()[0]*0.05) ), textcoords='data', # Adjust text position
                            arrowprops=dict(arrowstyle="->", connectionstyle="arc3"),
                            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))

            plot_filename = os.path.join(results_dir, f"{file_prefix}_mean_return_rolling_{window}d.png")
            try:
                plt.savefig(plot_filename, dpi=200, bbox_inches='tight')
                print(f"Saved rolling mean return plot to: {plot_filename}")
            except Exception as e:
                print(f"Warning: Could not save plot image {plot_filename}: {e}")
            plt.close(fig)

            csv_filename = os.path.join(results_dir, f"{file_prefix}_mean_return_rolling_{window}d_data.csv")
            try:
                aligned_means.write_csv(csv_filename)
                print(f"Saved rolling mean return data to: {csv_filename}")
            except Exception as e:
                print(f"Error saving data: {e}")
        else:
            print("No data for rolling mean returns plot.")
        return aligned_means

    def calculate_mean_returns_quantiles(self,
                                       results_dir: str,
                                       file_prefix: str = "event",
                                       return_col: str = 'ret',
                                       analysis_window: Tuple[int, int] = (-60, 60),
                                       lookback_window: int = 10,
                                       quantiles: List[float] = [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]):
        """
        Calculates mean return quantiles for each day relative to events using Matplotlib.
        """
        print(f"\n--- Calculating Mean Return Quantiles (Analysis Window: {analysis_window}, Lookback: {lookback_window}) ---")

        if self.data is None or return_col not in self.data.columns:
            print("Error: Data not loaded or missing return column.")
            return None

        extended_start = analysis_window[0] - lookback_window
        analysis_data_filtered = self.data.filter(
            (pl.col('days_to_event') >= extended_start) &
            (pl.col('days_to_event') <= analysis_window[1])
        ).with_columns(pl.col(return_col).clip(-0.5, 0.5).alias('clipped_return'))

        if analysis_data_filtered.is_empty():
            print(f"Error: No data found within analysis window {analysis_window}.")
            return None

        days_range_list = list(range(analysis_window[0], analysis_window[1] + 1))
        mean_ret_quantile_data = []
        print(f"Processing {len(days_range_list)} days...")

        batch_size = 10
        for batch_start_idx in range(0, len(days_range_list), batch_size):
            current_batch_days = days_range_list[batch_start_idx : batch_start_idx + batch_size]
            print(f"Processing days {current_batch_days[0]} to {current_batch_days[-1]} ({len(current_batch_days)} days)...")
            batch_results_list = []
            for day_val in current_batch_days:
                window_start_val = day_val - lookback_window
                window_end_val = day_val - 1 # Lookback up to the day before

                current_window_data = analysis_data_filtered.filter(
                    (pl.col('days_to_event') >= window_start_val) &
                    (pl.col('days_to_event') <= window_end_val)
                )

                if current_window_data.is_empty():
                    empty_res_dict = {"days_to_event": day_val, "event_count": 0}
                    for q_item in quantiles: empty_res_dict[f"mean_ret_q{int(q_item*100)}"] = None
                    batch_results_list.append(empty_res_dict)
                    continue

                mean_ret_by_event_df = current_window_data.group_by('event_id').agg([
                    pl.mean('clipped_return').alias('mean_ret_raw'),
                    pl.count().alias('n_obs')
                ]).filter(
                    (pl.col('n_obs') >= max(3, lookback_window // 3)) &
                    pl.col('mean_ret_raw').is_not_null()
                )

                num_valid_events = mean_ret_by_event_df.height
                if num_valid_events >= 5:
                    mean_ret_by_event_df = mean_ret_by_event_df.with_columns(
                        (pl.col('mean_ret_raw') * 100).alias('mean_ret_pct') # Convert to percentage
                    )
                    quantile_values_dict = {}
                    for q_item in quantiles:
                        q_result = mean_ret_by_event_df.select(pl.col('mean_ret_pct').quantile(q_item, interpolation='linear')).item()
                        quantile_values_dict[f"mean_ret_q{int(q_item*100)}"] = q_result
                    day_results_dict = {"days_to_event": day_val, "event_count": num_valid_events, **quantile_values_dict}
                else:
                    day_results_dict = {"days_to_event": day_val, "event_count": num_valid_events}
                    for q_item in quantiles: day_results_dict[f"mean_ret_q{int(q_item*100)}"] = None
                batch_results_list.append(day_results_dict)
            mean_ret_quantile_data.extend(batch_results_list)
            del current_window_data, mean_ret_by_event_df, batch_results_list
            gc.collect()

        results_mean_ret_df = pl.DataFrame(mean_ret_quantile_data)

        # Plotting with Matplotlib
        try:
            results_mean_ret_pd = results_mean_ret_df.to_pandas()
            fig, ax = plt.subplots(figsize=(12, 7))

            for q_item in quantiles:
                col_name_str = f"mean_ret_q{int(q_item*100)}"
                if col_name_str in results_mean_ret_pd.columns:
                    line_w = 2 if q_item == 0.5 else 1
                    line_s = 'solid' if q_item == 0.5 else 'dashed'
                    ax.plot(results_mean_ret_pd['days_to_event'], results_mean_ret_pd[col_name_str], linewidth=line_w, linestyle=line_s, label=f'Q{int(q_item*100)}')

            ax.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
            ax.axvline(x=0, color='red', linestyle='--', label='Event Day')
            y_curr_lims = ax.get_ylim()
            ax.text(0.1, y_curr_lims[1] * 0.9 if y_curr_lims[1] > 0 else y_curr_lims[0]*0.1 , 'Event Day', color='red', ha='left')


            if analysis_window[0] <= -30 and analysis_window[1] >= 30:
                ax.axvline(x=-30, color='green', linestyle=':', label='Month Before')
                ax.text(-29.9, y_curr_lims[1] * 0.85 if y_curr_lims[1] > 0 else y_curr_lims[0]*0.15, 'Month Before', color='green', ha='left')
                ax.axvline(x=30, color='purple', linestyle=':', label='Month After')
                ax.text(30.1, y_curr_lims[1] * 0.85 if y_curr_lims[1] > 0 else y_curr_lims[0]*0.15, 'Month After', color='purple', ha='left')


            event_window_start, event_window_end = -2, 2
            if analysis_window[0] <= event_window_start and analysis_window[1] >= event_window_end:
                ax.axvspan(event_window_start, event_window_end, color='yellow', alpha=0.2, label='Event Window')

            ax.set_title(f'Mean Return Quantiles Around Events (Lookback: {lookback_window} days)')
            ax.set_xlabel('Days Relative to Event')
            ax.set_ylabel('Mean Return (%)')
            ax.legend(loc='best')
            ax.grid(True, linestyle=':', alpha=0.7)
            ax.xaxis.set_major_locator(mticker.MultipleLocator(20))
            ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f%%')) # Percentage format

            plot_filename = os.path.join(results_dir, f"{file_prefix}_mean_return_quantiles.png")
            try:
                plt.savefig(plot_filename, dpi=200, bbox_inches='tight')
                print(f"Saved mean return quantiles plot to: {plot_filename}")
            except Exception as e:
                print(f"Warning: Could not save plot image {plot_filename}: {e}")
            plt.close(fig)

        except Exception as e:
            print(f"Error creating plot: {e}")
            traceback.print_exc()

        csv_filename = os.path.join(results_dir, f"{file_prefix}_mean_return_quantiles.csv")
        try:
            results_mean_ret_df.write_csv(csv_filename)
            print(f"Saved mean return quantiles data to: {csv_filename}")
        except Exception as e:
            print(f"Error saving quantiles data: {e}")
        return results_mean_ret_df


    def analyze_rvr(self,
                        results_dir: str,
                        file_prefix: str = "event",
                        return_col: str = 'ret',
                        analysis_window: tuple[int, int] = (-30, 30),
                        post_event_delta: int = 10,
                        lookback_window: int = 5,
                        optimistic_bias: float = 0.01,
                        min_periods: int = 3,
                        variance_floor: float = 1e-6,
                        rvr_clip_threshold: float = 1e10,
                        adaptive_threshold: bool = True):
            """
            Analyzes the Return-to-Variance Ratio (RVR) using Matplotlib.
            """
            print(f"\n--- Analyzing Return-to-Variance Ratio (RVR) ---")
            print(f"Analysis Window: {analysis_window}, Post-Event Delta: {post_event_delta} days, Lookback: {lookback_window} days")

            rvr_daily_df = None
            if self.data is None or return_col not in self.data.columns or 'days_to_event' not in self.data.columns:
                print("Error: Data not loaded or missing required columns.")
                return rvr_daily_df

            if post_event_delta <= 0 or post_event_delta > analysis_window[1]:
                print(f"Error: post_event_delta ({post_event_delta}) must be positive and within analysis window.")
                return rvr_daily_df

            phases_dict = {
                'pre_event': (analysis_window[0], -1),
                'post_event_rising': (0, post_event_delta),
                'late_post_event': (post_event_delta + 1, analysis_window[1])
            }

            extended_start_day = analysis_window[0] - lookback_window
            analysis_data_rvr = self.data.filter(
                (pl.col('days_to_event') >= extended_start_day) &
                (pl.col('days_to_event') <= analysis_window[1])
            ).with_columns(
                pl.col(return_col).clip(-0.5, 0.5).alias('clipped_return') # Clip first
            ).sort(['event_id', 'days_to_event'])


            if analysis_data_rvr.is_empty():
                print(f"Error: No data found within extended analysis window [{extended_start_day}, {analysis_window[1]}].")
                return rvr_daily_df

            sample_returns_rvr = analysis_data_rvr.select(pl.col('clipped_return')).sample(n=min(100, analysis_data_rvr.height)) # Use clipped for detection
            avg_abs_return_rvr = sample_returns_rvr.select(pl.all().mean().abs()).item()


            if adaptive_threshold:
                pct_05_rvr = sample_returns_rvr.select(pl.col('clipped_return').quantile(0.05)).item()
                pct_95_rvr = sample_returns_rvr.select(pl.col('clipped_return').quantile(0.95)).item()
                range_check_rvr = (pct_95_rvr - pct_05_rvr) > 0.1
                std_check_rvr = sample_returns_rvr.select(pl.col('clipped_return').std()).item() > 0.02
                returns_in_pct_rvr = range_check_rvr or std_check_rvr or (avg_abs_return_rvr > 0.05)
                print(f"Adaptive return format detection: {'Percentage form' if returns_in_pct_rvr else 'Decimal form'}")
                print(f"  Avg abs return: {avg_abs_return_rvr:.6f}, Range check: {range_check_rvr}, Std check: {std_check_rvr}")
            else:
                returns_in_pct_rvr = avg_abs_return_rvr > 0.05
                print(f"Standard return format detection: {'Percentage form' if returns_in_pct_rvr else 'Decimal form'} (avg abs: {avg_abs_return_rvr:.6f})")

            calc_return_col_rvr = 'decimal_return_rvr'
            if returns_in_pct_rvr:
                print("Converting percentage returns to decimal form for RVR calculation")
                analysis_data_rvr = analysis_data_rvr.with_columns(
                    (pl.col('clipped_return') / 100).alias(calc_return_col_rvr)
                )
            else:
                analysis_data_rvr = analysis_data_rvr.rename({'clipped_return': calc_return_col_rvr})


            analysis_data_rvr = analysis_data_rvr.with_columns([
                pl.col(calc_return_col_rvr).rolling_mean(
                    window_size=lookback_window, min_periods=min_periods
                ).over('event_id').alias('mean_return_rvr'),
                pl.col(calc_return_col_rvr).rolling_std(
                    window_size=lookback_window, min_periods=min_periods
                ).over('event_id').alias('volatility_rvr')
            ])

            adj_bias_rvr = optimistic_bias / 100 if returns_in_pct_rvr else optimistic_bias # Bias is in decimal form
            analysis_data_rvr = analysis_data_rvr.with_columns(
                pl.when(
                    (pl.col('days_to_event') >= phases_dict['post_event_rising'][0]) &
                    (pl.col('days_to_event') <= phases_dict['post_event_rising'][1])
                ).then(pl.col('mean_return_rvr') + adj_bias_rvr)
                .otherwise(pl.col('mean_return_rvr'))
                .alias('expected_return_rvr')
            )
            analysis_data_rvr = analysis_data_rvr.with_columns(
                pl.max_horizontal(pl.col('volatility_rvr') ** 2, pl.lit(variance_floor)).alias('variance_rvr')
            )
            analysis_data_rvr = analysis_data_rvr.with_columns(
                pl.when(pl.col('variance_rvr') > 1e-9) # Ensure variance is not effectively zero
                .then(pl.col('expected_return_rvr') / pl.col('variance_rvr'))
                .otherwise(None)
                .alias('raw_rvr_calc')
            )
            analysis_data_rvr = analysis_data_rvr.with_columns(
                pl.col('raw_rvr_calc').clip(-rvr_clip_threshold, rvr_clip_threshold).alias('rvr_final')
            )

            rvr_daily_df = analysis_data_rvr.group_by('days_to_event').agg([
                pl.col('rvr_final').mean().alias('avg_rvr'),
                pl.col('rvr_final').median().alias('median_rvr'),
                pl.col('raw_rvr_calc').mean().alias('avg_raw_rvr'),
                pl.col('expected_return_rvr').mean().alias('avg_expected_return'),
                pl.col('variance_rvr').mean().alias('avg_variance'),
                pl.col('rvr_final').count().alias('event_count') # count notnull rvr
            ]).sort('days_to_event')

            phase_summaries_list = []
            for phase_name_str, (start_day_val, end_day_val) in phases_dict.items():
                phase_data_df = analysis_data_rvr.filter(
                    (pl.col('days_to_event') >= start_day_val) & (pl.col('days_to_event') <= end_day_val)
                )
                if not phase_data_df.is_empty():
                    phase_stats_dict = {
                        'phase': phase_name_str, 'start_day': start_day_val, 'end_day': end_day_val,
                        'avg_rvr': phase_data_df['rvr_final'].mean(),
                        'median_rvr': phase_data_df['rvr_final'].median(),
                        'avg_raw_rvr': phase_data_df['raw_rvr_calc'].mean(),
                        'avg_expected_return': phase_data_df['expected_return_rvr'].mean(),
                        'avg_variance': phase_data_df['variance_rvr'].mean(),
                        'event_count': phase_data_df.filter(pl.col('rvr_final').is_not_null()).height
                    }
                    phase_summaries_list.append(phase_stats_dict)
                else: # Append Nones if phase is empty
                    phase_summaries_list.append({
                        'phase': phase_name_str, 'start_day': start_day_val, 'end_day': end_day_val,
                        'avg_rvr': None, 'median_rvr': None, 'avg_raw_rvr': None,
                        'avg_expected_return': None, 'avg_variance': None, 'event_count': 0
                    })


            phase_summary_df = pl.DataFrame(phase_summaries_list)
            print("\nRVR by Phase:")
            for row_dict in phase_summary_df.to_dicts():
                print(f"Phase: {row_dict['phase']} ({row_dict['start_day']} to {row_dict['end_day']} days)")
                print(f"  Avg RVR: {row_dict.get('avg_rvr', 0.0):.4f}")
                print(f"  Median RVR: {row_dict.get('median_rvr', 0.0):.4f}")
                print(f"  Avg Expected Return: {row_dict.get('avg_expected_return', 0.0):.6f}")
                print(f"  Avg Variance: {row_dict.get('avg_variance', 0.0):.6f}")
                print(f"  Events: {row_dict.get('event_count', 0)}")


            # Plotting with Matplotlib
            try:
                rvr_pd_plot = rvr_daily_df.to_pandas()
                fig, ax = plt.subplots(figsize=(12, 7))

                ax.plot(rvr_pd_plot['days_to_event'], rvr_pd_plot['avg_rvr'], color='blue', linewidth=2, label='Avg RVR')

                ax.axvline(x=0, color='red', linestyle='--', label='Event Day')
                ax.axvline(x=post_event_delta, color='purple', linestyle=':', label='End of Post-Event Rising')
                ax.axvspan(phases_dict['post_event_rising'][0], phases_dict['post_event_rising'][1], color='yellow', alpha=0.2, label='Post-Event Rising')

                y_lim_plot = ax.get_ylim()
                ax.text(0.1, y_lim_plot[1]*0.9, 'Event Day', color='red', ha='left')
                ax.text(post_event_delta + 0.1, y_lim_plot[1]*0.85, 'End Post-Event Rising', color='purple', ha='left')


                valid_rvr_plot = rvr_pd_plot['avg_rvr'].dropna()
                if not valid_rvr_plot.empty:
                    y_min_rvr, y_max_rvr = valid_rvr_plot.min(), valid_rvr_plot.max()
                    y_padding_rvr = 0.1 * (y_max_rvr - y_min_rvr) if y_max_rvr != y_min_rvr else 0.1
                    y_range_rvr = [y_min_rvr - y_padding_rvr, y_max_rvr + y_padding_rvr]
                else:
                    y_range_rvr = [-1, 1] # Default if no data

                ax.set_title(f'Return-to-Variance Ratio Around Events (Lookback: {lookback_window} days)')
                ax.set_xlabel('Days Relative to Event')
                ax.set_ylabel('Average RVR')
                ax.legend(loc='best')
                ax.grid(True, linestyle=':', alpha=0.7)
                ax.set_ylim(y_range_rvr)
                ax.xaxis.set_major_locator(mticker.MultipleLocator(10))
                ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))

                for row_dict_plot in phase_summary_df.to_dicts():
                    if row_dict_plot.get('avg_rvr') is not None:
                        x_pos_plot = (row_dict_plot['start_day'] + row_dict_plot['end_day']) / 2
                        y_pos_plot = row_dict_plot['avg_rvr']
                        # Adjust annotation position to avoid overlap
                        text_y_offset = (y_range_rvr[1] - y_range_rvr[0]) * 0.05
                        ax_offset = 20 if row_dict_plot['phase'] == 'pre_event' else -20
                        ay_offset = -30 if y_pos_plot > (y_range_rvr[0] + y_range_rvr[1])/2 else 30

                        ax.annotate(f"{row_dict_plot['phase']}: {y_pos_plot:.2f}",
                                    xy=(x_pos_plot, y_pos_plot), xycoords='data',
                                    xytext=(x_pos_plot + ax_offset/2 , y_pos_plot + ay_offset/10), textcoords='data', # Simplified offset
                                    arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"),
                                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.7))


                plot_filename = os.path.join(results_dir, f"{file_prefix}_rvr_timeseries.png")
                try:
                    plt.savefig(plot_filename, dpi=200, bbox_inches='tight')
                    print(f"Saved RVR time series plot to: {plot_filename}")
                except Exception as e:
                    print(f"Warning: Could not save plot image {plot_filename}: {e}")
                plt.close(fig)

            except Exception as e:
                print(f"Error creating RVR plot: {e}")
                traceback.print_exc()

            if rvr_daily_df is not None:
                csv_filename = os.path.join(results_dir, f"{file_prefix}_rvr_daily.csv")
                try:
                    rvr_daily_df.write_csv(csv_filename)
                    print(f"Saved daily RVR data to: {csv_filename}")
                except Exception as e:
                    print(f"Error saving daily RVR data: {e}")

            phase_csv_filename = os.path.join(results_dir, f"{file_prefix}_rvr_phase_summary.csv")
            try:
                phase_summary_df.write_csv(phase_csv_filename)
                print(f"Saved RVR phase summary to: {phase_csv_filename}")
            except Exception as e:
                print(f"Error saving phase summary: {e}")

            del analysis_data_rvr, phase_summary_df
            gc.collect()
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
                                    delta: int = 5):
        """
        Analyze event volatility using the three-phase volatility model with GARCH baseline.

        Parameters:
        -----------
        results_dir : str
            Directory to save results
        file_prefix : str
            Prefix for output files
        return_col : str
            Column containing returns
        analysis_window : Tuple[int, int]
            Window around event for analysis (days_to_event)
        garch_type : str
            Type of GARCH model ('garch' or 'gjr')
        k1 : float
            Pre-event volatility multiplier
        k2 : float
            Post-event volatility multiplier
        delta_t1 : float
            Pre-event volatility duration parameter
        delta_t2 : float
            Post-event rising phase rate parameter
        delta_t3 : float
            Post-event decay rate parameter
        delta : int
            Duration of post-event rising phase in days
        """
        print(f"\n--- Analyzing Three-Phase Volatility (GARCH Type: {garch_type}) ---")

        if self.data is None or return_col not in self.data.columns:
            print("Error: Data not loaded or return column missing.")
            return None

        # Prepare data - extend window for GARCH estimation
        extended_window = (min(analysis_window[0], -60), max(analysis_window[1], 60))
        analysis_data = self.data.filter(
            (pl.col('days_to_event') >= extended_window[0]) &
            (pl.col('days_to_event') <= extended_window[1])
        ).sort(['event_id', 'days_to_event'])

        # Check if we have enough data
        if analysis_data.is_empty():
            print(f"Error: No data found within extended window {extended_window}")
            return None

        # Group by event_id and fit GARCH models
        event_ids = analysis_data.select('event_id').unique().to_series().to_list()
        print(f"Fitting GARCH models for {len(event_ids)} events...")

        # Sample a few events for detailed analysis
        if len(event_ids) > 10:
            np.random.seed(42)
            sample_event_ids = np.random.choice(event_ids, size=5, replace=False)
        else:
            sample_event_ids = event_ids

        # Set up volatility models
        all_events_results = []
        sample_events_data = []

        # Process events
        for event_id in event_ids:
            event_data = analysis_data.filter(pl.col('event_id') == event_id)
            event_days = event_data.select('days_to_event').to_series().to_list()
            event_returns = event_data.select(return_col).to_series()

            # Skip if insufficient data
            if len(event_returns) < 20:  # Need minimum data for GARCH
                continue
            
            # Initialize and fit GARCH model
            try:
                if garch_type.lower() == 'gjr':
                    garch_model = GJRGARCHModel()
                else:
                    garch_model = GARCHModel()

                garch_model.fit(event_returns)

                # Create three-phase volatility model
                vol_model = ThreePhaseVolatilityModel(
                    baseline_model=garch_model,
                    k1=k1, k2=k2, 
                    delta_t1=delta_t1, delta_t2=delta_t2, delta_t3=delta_t3,
                    delta=delta
                )

                # Calculate volatility for analysis window
                analysis_days = np.array(range(analysis_window[0], analysis_window[1]+1))
                predicted_vol = vol_model.calculate_volatility_series(analysis_days)

                # Store results for this event
                event_results = {
                    'event_id': event_id,
                    'days_to_event': analysis_days,
                    'predicted_volatility': predicted_vol
                }
                all_events_results.append(event_results)

                # Save detailed data for sample events
                if event_id in sample_event_ids:
                    # Get GARCH volatility and three-phase volatility
                    garch_vol = np.sqrt(garch_model.variance_history)

                    # Map days in GARCH history to days_to_event
                    event_days_np = np.array(event_days)

                    # Collect data for this event
                    sample_data = {
                        'event_id': event_id,
                        'garch_days': event_days_np[:len(garch_vol)],
                        'garch_vol': garch_vol,
                        'three_phase_days': analysis_days,
                        'three_phase_vol': predicted_vol,
                        'returns': event_returns.to_numpy()[:len(garch_vol)],
                        'ticker': event_data.select('ticker').head(1).item()
                    }
                    sample_events_data.append(sample_data)

            except Exception as e:
                print(f"Error processing event {event_id}: {e}")
                continue
            
        # Aggregate results across events
        if not all_events_results:
            print("No valid results to analyze.")
            return None

        # Create a time series of average volatility
        aggregated_data = {}
        for day in range(analysis_window[0], analysis_window[1]+1):
            day_volatilities = []
            for event_result in all_events_results:
                day_idx = np.where(event_result['days_to_event'] == day)[0]
                if day_idx.size > 0:
                    day_volatilities.append(event_result['predicted_volatility'][day_idx[0]])

            if day_volatilities:
                aggregated_data[day] = {
                    'avg_volatility': np.mean(day_volatilities),
                    'median_volatility': np.median(day_volatilities),
                    'std_volatility': np.std(day_volatilities),
                    'count': len(day_volatilities)
                }

        # Convert to DataFrame
        volatility_df = pl.DataFrame({
            'days_to_event': list(aggregated_data.keys()),
            'avg_volatility': [data['avg_volatility'] for data in aggregated_data.values()],
            'median_volatility': [data['median_volatility'] for data in aggregated_data.values()],
            'std_volatility': [data['std_volatility'] for data in aggregated_data.values()],
            'event_count': [data['count'] for data in aggregated_data.values()]
        }).sort('days_to_event')

        # Define volatility phases
        phases = {
            'pre_event': (analysis_window[0], -1),
            'event_window': (0, 0),
            'post_event_rising': (1, delta),
            'post_event_decay': (delta+1, analysis_window[1])
        }

        # Calculate average volatility by phase
        phase_stats = []
        for phase_name, (start_day, end_day) in phases.items():
            phase_data = volatility_df.filter(
                (pl.col('days_to_event') >= start_day) &
                (pl.col('days_to_event') <= end_day)
            )

            if not phase_data.is_empty():
                avg_vol = phase_data['avg_volatility'].mean()
                median_vol = phase_data['median_volatility'].mean()
                phase_stats.append({
                    'phase': phase_name,
                    'start_day': start_day,
                    'end_day': end_day,
                    'avg_volatility': avg_vol,
                    'median_volatility': median_vol,
                    'avg_event_count': phase_data['event_count'].mean()
                })

        phase_stats_df = pl.DataFrame(phase_stats)

        # Save results
        try:
            volatility_df.write_csv(os.path.join(results_dir, f"{file_prefix}_three_phase_volatility.csv"))
            phase_stats_df.write_csv(os.path.join(results_dir, f"{file_prefix}_volatility_phase_stats.csv"))
            print(f"Saved volatility analysis to {results_dir}")
        except Exception as e:
            print(f"Error saving volatility analysis: {e}")

        # Plotting
        try:
            import matplotlib.pyplot as plt

            # Plot average volatility
            fig, ax = plt.subplots(figsize=(12, 7))
            volatility_pd = volatility_df.to_pandas()

            ax.plot(volatility_pd['days_to_event'], volatility_pd['avg_volatility'], 
                    color='blue', linewidth=2, label='Average Volatility')

            # Add phase markers
            ax.axvline(x=0, color='red', linestyle='--', label='Event Day')
            ax.axvline(x=delta, color='green', linestyle=':', label=f'End Rising Phase (t+{delta})')

            # Highlight phases
            ax.axvspan(0, delta, color='yellow', alpha=0.2, label='Post-Event Rising Phase')

            ax.set_title(f'Three-Phase Volatility Around Events ({garch_type.upper()}-GARCH Model)')
            ax.set_xlabel('Days Relative to Event')
            ax.set_ylabel('Volatility')
            ax.legend(loc='best')
            ax.grid(True, linestyle=':', alpha=0.7)

            # Add phase statistics annotations
            for phase in phase_stats_df.to_dicts():
                midpoint = (phase['start_day'] + phase['end_day']) / 2
                ax.annotate(
                    f"{phase['phase']}: {phase['avg_volatility']:.4f}",
                    xy=(midpoint, phase['avg_volatility']),
                    xytext=(midpoint, phase['avg_volatility'] * 1.1),
                    arrowprops=dict(arrowstyle="->", connectionstyle="arc3"),
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8)
                )

            plt.tight_layout()
            plt.savefig(os.path.join(results_dir, f"{file_prefix}_three_phase_volatility.png"), dpi=200)
            plt.close()

            # Plot sample events
            for sample_data in sample_events_data:
                fig, ax = plt.subplots(figsize=(12, 7))

                # Plot GARCH volatility
                ax.plot(sample_data['garch_days'], sample_data['garch_vol'], 
                        color='blue', linewidth=1, alpha=0.7, label='GARCH Volatility')

                # Plot three-phase volatility
                ax.plot(sample_data['three_phase_days'], sample_data['three_phase_vol'], 
                        color='red', linewidth=2, label='Three-Phase Model')

                ax.axvline(x=0, color='black', linestyle='--', label='Event Day')
                ax.axvline(x=delta, color='green', linestyle=':', label=f'End Rising Phase (t+{delta})')

                ax.set_title(f"Volatility for {sample_data['ticker']} (Event ID: {sample_data['event_id']})")
                ax.set_xlabel('Days Relative to Event')
                ax.set_ylabel('Volatility')
                ax.legend(loc='best')
                ax.grid(True, linestyle=':', alpha=0.7)

                plt.tight_layout()
                plt.savefig(os.path.join(results_dir, f"{file_prefix}_volatility_sample_{sample_data['event_id']}.png"), dpi=200)
                plt.close()

            print(f"Saved volatility plots to {results_dir}")

        except Exception as e:
            print(f"Error plotting volatility analysis: {e}")

        return volatility_df, phase_stats_df


    def analyze_rvr_with_optimistic_bias(self,
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
                                       delta: int = 5,
                                       optimistic_bias: float = 0.01,
                                       risk_free_rate: float = 0.0):
        """
        Analyze Return-to-Variance Ratio with optimistic bias for post-event rising phase.
        Tests Hypothesis 1 from the paper.

        Parameters:
        -----------
        results_dir : str
            Directory to save results
        file_prefix : str
            Prefix for output files
        return_col : str
            Column containing returns
        analysis_window : Tuple[int, int]
            Window around event for analysis (days_to_event)
        garch_type : str
            Type of GARCH model ('garch' or 'gjr')
        k1 : float
            Pre-event volatility multiplier
        k2 : float
            Post-event volatility multiplier
        delta_t1 : float
            Pre-event volatility duration parameter
        delta_t2 : float
            Post-event rising phase rate parameter
        delta_t3 : float
            Post-event decay rate parameter
        delta : int
            Duration of post-event rising phase in days
        optimistic_bias : float
            Bias parameter for post-event expected returns (decimal form)
        risk_free_rate : float
            Daily risk-free rate
        """
        print(f"\n--- Analyzing Return-to-Variance Ratio with Optimistic Bias ---")
        print(f"Analysis Window: {analysis_window}, Post-Event Rising Phase: 0 to {delta}")

        if self.data is None or return_col not in self.data.columns:
            print("Error: Data not loaded or return column missing.")
            return None

        # Define phases
        phases = {
            'pre_event': (analysis_window[0], -1),
            'event_day': (0, 0),
            'post_event_rising': (1, delta),
            'post_event_decay': (delta+1, analysis_window[1])
        }

        # Prepare data
        extended_window = (min(analysis_window[0], -60), max(analysis_window[1], 60))
        analysis_data = self.data.filter(
            (pl.col('days_to_event') >= extended_window[0]) &
            (pl.col('days_to_event') <= extended_window[1])
        ).sort(['event_id', 'days_to_event'])

        # Check if we have enough data
        if analysis_data.is_empty():
            print(f"Error: No data found within extended window {extended_window}")
            return None

        # Detect return format (percentage or decimal)
        sample_returns = analysis_data.select(pl.col(return_col)).sample(n=min(100, analysis_data.height))
        avg_abs_return = sample_returns.select(pl.all().mean().abs()).item()
        returns_in_pct = avg_abs_return > 0.05

        print(f"Detected returns format: {'Percentage form' if returns_in_pct else 'Decimal form'} (avg abs: {avg_abs_return:.4f})")

        # Convert returns if necessary and calculate expected returns with bias
        if returns_in_pct:
            print("Converting percentage returns to decimal form for RVR calculation")
            analysis_data = analysis_data.with_columns(
                (pl.col(return_col) / 100).alias('decimal_return')
            )
            return_col_for_calc = 'decimal_return'
            # Adjust bias to decimal form
            adj_bias = optimistic_bias / 100
        else:
            return_col_for_calc = return_col
            adj_bias = optimistic_bias

        # Add expected return with bias for post-event rising phase
        analysis_data = analysis_data.with_columns([
            pl.col(return_col_for_calc).alias('expected_return'),
            pl.lit(np.nan).cast(pl.Float64).alias('volatility'),
            pl.lit(np.nan).cast(pl.Float64).alias('rvr')
        ])

        # Group by event_id
        event_ids = analysis_data.select('event_id').unique().to_series().to_list()
        print(f"Processing {len(event_ids)} events...")

        # Process events to estimate volatility and calculate RVR
        all_rvr_data = []

        for event_id in event_ids:
            try:
                # Get data for this event
                event_data = analysis_data.filter(pl.col('event_id') == event_id)
                event_days = event_data.select('days_to_event').to_series().to_list()
                event_returns = event_data.select(return_col_for_calc).to_series()

                # Skip if insufficient data
                if len(event_returns) < 30:  # Need minimum data for GARCH
                    continue
                
                # Fit GARCH model
                if garch_type.lower() == 'gjr':
                    garch_model = GJRGARCHModel()
                else:
                    garch_model = GARCHModel()

                garch_model.fit(event_returns)

                # Create three-phase volatility model
                vol_model = ThreePhaseVolatilityModel(
                    baseline_model=garch_model,
                    k1=k1, k2=k2, 
                    delta_t1=delta_t1, delta_t2=delta_t2, delta_t3=delta_t3,
                    delta=delta
                )

                # Calculate expected returns
                mean_return = np.mean(event_returns.to_numpy())

                # Add optimistic bias for post-event rising phase
                expected_returns = np.array([
                    mean_return + adj_bias if phases['post_event_rising'][0] <= day <= phases['post_event_rising'][1] else mean_return
                    for day in event_days
                ])

                # Calculate volatility for each day
                volatility = [vol_model.calculate_volatility(day, 0, np.sqrt(garch_model.sigma2_t)) 
                              for day in event_days]

                # Calculate RVR
                rvr = expected_returns / (np.array(volatility)**2 + 1e-10)  # Add small value to avoid division by zero

                # Create result dataset
                event_rvr_data = pd.DataFrame({
                    'event_id': [event_id] * len(event_days),
                    'days_to_event': event_days,
                    'expected_return': expected_returns,
                    'volatility': volatility,
                    'rvr': rvr
                })

                all_rvr_data.append(event_rvr_data)

            except Exception as e:
                print(f"Error processing event {event_id}: {e}")
                continue
            
        # Combine all events
        if not all_rvr_data:
            print("No valid results to analyze.")
            return None

        combined_rvr = pd.concat(all_rvr_data, ignore_index=True)

        # Aggregate by days_to_event
        agg_rvr = combined_rvr.groupby('days_to_event').agg({
            'expected_return': ['mean', 'median'],
            'volatility': ['mean', 'median'],
            'rvr': ['mean', 'median', 'count']
        }).reset_index()

        # Flatten column names
        agg_rvr.columns = ['days_to_event', 'mean_expected_return', 'median_expected_return',
                           'mean_volatility', 'median_volatility', 'mean_rvr', 'median_rvr', 'count']

        # Convert to Polars DataFrame
        rvr_df = pl.from_pandas(agg_rvr)

        # Calculate average RVR by phase
        phase_stats = []
        for phase_name, (start_day, end_day) in phases.items():
            phase_data = rvr_df.filter(
                (pl.col('days_to_event') >= start_day) &
                (pl.col('days_to_event') <= end_day)
            )

            if not phase_data.is_empty():
                avg_rvr = phase_data['mean_rvr'].mean()
                median_rvr = phase_data['median_rvr'].mean()
                avg_vol = phase_data['mean_volatility'].mean()
                avg_ret = phase_data['mean_expected_return'].mean()

                phase_stats.append({
                    'phase': phase_name,
                    'start_day': start_day,
                    'end_day': end_day,
                    'avg_rvr': avg_rvr,
                    'median_rvr': median_rvr,
                    'avg_volatility': avg_vol,
                    'avg_expected_return': avg_ret,
                    'avg_event_count': phase_data['count'].mean()
                })

        phase_stats_df = pl.DataFrame(phase_stats)

        # Print phase RVR statistics
        print("\nRVR by Phase:")
        phases_to_highlight = ['pre_event', 'post_event_rising', 'post_event_decay']
        for row in phase_stats_df.filter(pl.col('phase').is_in(phases_to_highlight)).iter_rows(named=True):
            print(f"Phase: {row['phase']} ({row['start_day']} to {row['end_day']} days)")
            print(f"  Avg RVR: {row['avg_rvr']:.4f}, Median RVR: {row['median_rvr']:.4f}")
            print(f"  Avg Volatility: {row['avg_volatility']:.4f}, Avg Expected Return: {row['avg_expected_return']:.6f}")

        # Test Hypothesis 1 - Does RVR peak during post-event rising phase?
        pre_event_rvr = phase_stats_df.filter(pl.col('phase') == 'pre_event')['avg_rvr'].item()
        post_rising_rvr = phase_stats_df.filter(pl.col('phase') == 'post_event_rising')['avg_rvr'].item()
        post_decay_rvr = phase_stats_df.filter(pl.col('phase') == 'post_event_decay')['avg_rvr'].item()

        h1_result = post_rising_rvr > pre_event_rvr and post_rising_rvr > post_decay_rvr

        print("\nHypothesis 1 Test:")
        print(f"Does RVR peak during post-event rising phase? {'YES' if h1_result else 'NO'}")
        print(f"  Pre-event RVR: {pre_event_rvr:.4f}")
        print(f"  Post-event Rising RVR: {post_rising_rvr:.4f}")
        print(f"  Post-event Decay RVR: {post_decay_rvr:.4f}")

        # Save results
        try:
            rvr_df.write_csv(os.path.join(results_dir, f"{file_prefix}_rvr_timeseries.csv"))
            phase_stats_df.write_csv(os.path.join(results_dir, f"{file_prefix}_rvr_phase_stats.csv"))

            # Save hypothesis test results
            h1_df = pl.DataFrame({
                'hypothesis': ['H1: RVR peaks during post-event rising phase'],
                'result': [h1_result],
                'pre_event_rvr': [pre_event_rvr],
                'post_rising_rvr': [post_rising_rvr],
                'post_decay_rvr': [post_decay_rvr]
            })
            h1_df.write_csv(os.path.join(results_dir, f"{file_prefix}_hypothesis1_test.csv"))

            print(f"Saved RVR analysis to {results_dir}")
        except Exception as e:
            print(f"Error saving RVR analysis: {e}")

        # Plotting
        try:
            import matplotlib.pyplot as plt

            # Plot RVR time series
            fig, ax = plt.subplots(figsize=(12, 7))
            rvr_pd = rvr_df.to_pandas()

            ax.plot(rvr_pd['days_to_event'], rvr_pd['mean_rvr'], 
                    color='blue', linewidth=2, label='Mean RVR')

            # Add phase markers
            ax.axvline(x=0, color='red', linestyle='--', label='Event Day')
            ax.axvline(x=delta, color='green', linestyle=':', 
                       label=f'End Rising Phase (t+{delta})')

            # Highlight phases
            ax.axvspan(phases['post_event_rising'][0], phases['post_event_rising'][1], 
                       color='yellow', alpha=0.2, label='Post-Event Rising Phase')

            ax.set_title(f'Return-to-Variance Ratio with Optimistic Bias = {optimistic_bias*100 if not returns_in_pct else optimistic_bias}%')
            ax.set_xlabel('Days Relative to Event')
            ax.set_ylabel('Return-to-Variance Ratio')
            ax.legend(loc='best')
            ax.grid(True, linestyle=':', alpha=0.7)

            # Add phase statistics annotations
            for phase in phase_stats_df.to_dicts():
                midpoint = (phase['start_day'] + phase['end_day']) / 2
                ax.annotate(
                    f"{phase['phase']}: {phase['avg_rvr']:.4f}",
                    xy=(midpoint, phase['avg_rvr']),
                    xytext=(midpoint, phase['avg_rvr'] * 1.1),
                    arrowprops=dict(arrowstyle="->", connectionstyle="arc3"),
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8)
                )

            plt.tight_layout()
            plt.savefig(os.path.join(results_dir, f"{file_prefix}_rvr_timeseries.png"), dpi=200)
            plt.close()

            # Plot volatility and expected returns
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12), sharex=True)

            # Volatility
            ax1.plot(rvr_pd['days_to_event'], rvr_pd['mean_volatility'], 
                     color='red', linewidth=2, label='Mean Volatility')
            ax1.set_title('Volatility Around Events')
            ax1.set_ylabel('Volatility')
            ax1.legend(loc='best')
            ax1.grid(True, linestyle=':', alpha=0.7)

            # Expected returns
            ax2.plot(rvr_pd['days_to_event'], rvr_pd['mean_expected_return'], 
                     color='green', linewidth=2, label='Mean Expected Return')
            ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)

            # Add phase markers to both plots
            for ax in [ax1, ax2]:
                ax.axvline(x=0, color='red', linestyle='--', label='Event Day')
                ax.axvline(x=delta, color='green', linestyle=':', 
                          label=f'End Rising Phase (t+{delta})')
                ax.axvspan(phases['post_event_rising'][0], phases['post_event_rising'][1], 
                          color='yellow', alpha=0.2, label='Post-Event Rising Phase')

            ax2.set_title('Expected Returns Around Events')
            ax2.set_xlabel('Days Relative to Event')
            ax2.set_ylabel('Expected Return')
            ax2.legend(loc='best')
            ax2.grid(True, linestyle=':', alpha=0.7)

            plt.tight_layout()
            plt.savefig(os.path.join(results_dir, f"{file_prefix}_volatility_returns.png"), dpi=200)
            plt.close()

            # Plot RVR by phase - bar chart
            fig, ax = plt.subplots(figsize=(10, 6))

            phases_to_plot = phase_stats_df.filter(pl.col('phase').is_in(phases_to_highlight))
            phases_to_plot = phases_to_plot.with_columns(
                pl.col('phase').cast(pl.Categorical).cast(pl.Utf8)
            )

            phases_pd = phases_to_plot.to_pandas()

            # Define colors
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
            hatches = ['/', '\\', 'x']

            # Create bar chart
            bars = ax.bar(phases_pd['phase'], phases_pd['avg_rvr'], 
                          color=colors, alpha=0.7)

            # Add hatching
            for bar, hatch in zip(bars, hatches):
                bar.set_hatch(hatch)

            # Add value labels
            for i, v in enumerate(phases_pd['avg_rvr']):
                ax.text(i, v + 0.02, f"{v:.4f}", ha='center', fontweight='bold')

            # Highlight the highest bar
            max_idx = phases_pd['avg_rvr'].argmax()
            bars[max_idx].set_edgecolor('red')
            bars[max_idx].set_linewidth(2)

            ax.set_title('Return-to-Variance Ratio by Phase')
            ax.set_ylabel('Average RVR')
            ax.grid(True, axis='y', linestyle=':', alpha=0.7)

            # Add hypothesis test result
            result_text = "Hypothesis 1 SUPPORTED: RVR peaks during post-event rising phase" if h1_result else "Hypothesis 1 NOT SUPPORTED"
            ax.text(0.5, -0.1, result_text, ha='center', transform=ax.transAxes, 
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", alpha=0.8),
                    fontweight='bold', color='green' if h1_result else 'red')

            plt.tight_layout()
            plt.savefig(os.path.join(results_dir, f"{file_prefix}_rvr_by_phase.png"), dpi=200)
            plt.close()

            print(f"Saved RVR plots to {results_dir}")

        except Exception as e:
            print(f"Error plotting RVR analysis: {e}")

        return rvr_df, phase_stats_df