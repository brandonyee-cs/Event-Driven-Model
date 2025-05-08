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

# Import shared models
try:
    # Import all model classes from models.py
    from src.models import TimeSeriesRidge, XGBoostDecileModel, GARCH, EventAsset, TwoRiskFramework, MarketClearingModel
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
        # Initialize GARCH models dictionary to store models for each event
        self.garch_models = {}
        # Initialize EventAsset instances for each event
        self.event_assets = {}
        # Default parameters for GARCH model
        self.garch_params = {
            'omega': 0.00001,
            'alpha': 0.1,
            'beta': 0.8,
            'gamma': 0.05  # Default to GJR-GARCH
        }
        # Default parameters for three-phase volatility
        self.vol_params = {
            'k1': 1.5,   # Pre-event volatility multiplier
            'k2': 2.0,   # Post-event volatility multiplier
            'delta': 10, # Duration of post-event rising phase
            'dt1': 5,    # Pre-event rise duration parameter
            'dt2': 2,    # Post-event rise rate parameter
            'dt3': 10    # Post-event decay rate parameter
        }
        # Default parameters for EventAsset
        self.event_asset_params = {
            'baseline_mu': 0.001,
            'rf_rate': 0.0,
            'risk_aversion': 2.0,
            'corr_generic': 0.3,
            'sigma_generic': 0.01,
            'mu_generic': 0.0005,
            'transaction_cost_buy': 0.001,
            'transaction_cost_sell': 0.0005
        }
        # Default parameters for two-risk framework
        self.two_risk_params = {
            'directional_risk_vol': 0.05,
            'impact_uncertainty_vol': 0.02,
            'directional_risk_premium': 0.05,
            'impact_uncertainty_premium': 0.03
        }

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

    def fit_garch_models(self, return_col='ret', model_type='gjr'):
        """
        Fit GARCH models for each event in the data.
        
        Parameters:
        -----------
        return_col : str, optional
            Column name containing returns
        model_type : str, optional
            Type of GARCH model ('garch' or 'gjr')
            
        Returns:
        --------
        dict
            Dictionary of GARCH models for each event
        """
        print("\n--- Fitting GARCH models for each event ---")
        
        if self.data is None:
            print("Error: No data loaded. Call load_and_prepare_data first.")
            return {}
            
        if return_col not in self.data.columns:
            print(f"Error: Return column '{return_col}' not found in data.")
            return {}
            
        # Get unique event IDs
        event_ids = self.data.select('event_id').unique().to_series().to_list()
        print(f"Fitting GARCH models for {len(event_ids)} events...")
        
        for event_id in event_ids:
            # Filter data for this event
            event_data = self.data.filter(pl.col('event_id') == event_id)
            event_returns = event_data.select(pl.col(return_col)).to_series()
            
            # Ensure data is sorted by date
            event_data = event_data.sort('date')
            
            ## Get event date for this event
            event_day_df = event_data.filter(pl.col('days_to_event') == 0)
            if event_day_df.is_empty():
                print(f"Warning: No event day (days_to_event == 0) found for event {event_id}. Skipping.")
                continue  # Skip to the next event
            
            event_date = event_day_df.select('date').item()
            event_day_idx = event_day_df.with_row_count().select('row_nr').item()
            
            # Skip if event has insufficient data
            if len(event_returns) < 20:  # Minimum data needed for GARCH
                continue
                
            try:
                # Initialize and fit GARCH model
                garch_model = GARCH(**self.garch_params)
                garch_model.fit(returns=event_returns.to_numpy(), model_type=model_type)
                
                # Store the fitted model
                self.garch_models[event_id] = {
                    'model': garch_model,
                    'event_day_idx': event_day_idx,
                    'event_date': event_date
                }
                
                # Create EventAsset for this event
                event_asset = EventAsset(**self.event_asset_params)
                event_asset.garch_model = garch_model
                event_asset.garch_fitted = True
                
                # Store the event asset
                self.event_assets[event_id] = event_asset
                
            except Exception as e:
                print(f"Error fitting GARCH model for event {event_id}: {e}")
                
        print(f"Successfully fitted GARCH models for {len(self.garch_models)} events.")
        
        return self.garch_models

    def analyze_volatility_patterns(self, results_dir: str, file_prefix: str = "event", 
                                    return_col: str = 'ret'):
        """
        Analyze volatility patterns around events using fitted GARCH models.
        
        Parameters:
        -----------
        results_dir : str
            Directory to save results
        file_prefix : str, optional
            Prefix for output files
        return_col : str, optional
            Column name containing returns
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with volatility patterns
        """
        print("\n--- Analyzing Volatility Patterns with GARCH ---")
        
        if not self.garch_models:
            print("Fitting GARCH models first...")
            self.fit_garch_models(return_col=return_col)
            
        if not self.garch_models:
            print("Error: No GARCH models available.")
            return None
            
        # Create directory if it doesn't exist
        os.makedirs(results_dir, exist_ok=True)
            
        # Collect volatility patterns for each event
        volatility_patterns = []
        
        for event_id, garch_info in self.garch_models.items():
            garch_model = garch_info['model']
            event_day_idx = garch_info['event_day_idx']
            
            # Get volatility series
            volatility = np.sqrt(garch_model.h)
            
            # Get event window
            event_data = self.data.filter(pl.col('event_id') == event_id)
            days_to_event = event_data.select('days_to_event').to_series().to_list()
            
            # Match volatility to days_to_event
            # Ensure same length if needed
            if len(volatility) > len(days_to_event):
                volatility = volatility[-len(days_to_event):]
            elif len(volatility) < len(days_to_event):
                # Pad with the first value
                padding = np.full(len(days_to_event) - len(volatility), volatility[0])
                volatility = np.concatenate([padding, volatility])
                
            for day, vol in zip(days_to_event, volatility):
                volatility_patterns.append({
                    'event_id': event_id,
                    'days_to_event': day,
                    'volatility': vol,
                    'annualized_vol': vol * np.sqrt(252) * 100  # Convert to annual percentage
                })
                
        # Convert to DataFrame
        vol_df = pd.DataFrame(volatility_patterns)
        
        # Calculate average volatility for each day relative to event
        avg_vol = vol_df.groupby('days_to_event').agg(
            avg_vol=('volatility', 'mean'),
            avg_annualized_vol=('annualized_vol', 'mean'),
            median_vol=('volatility', 'median'),
            std_vol=('volatility', 'std'),
            count=('volatility', 'count')
        ).reset_index()
        
        # Save data
        csv_filename = os.path.join(results_dir, f"{file_prefix}_garch_volatility.csv")
        avg_vol.to_csv(csv_filename, index=False)
        print(f"Saved GARCH volatility data to: {csv_filename}")
        
        # Create plot
        fig, ax = plt.subplots(figsize=(12, 6))
        
        ax.plot(avg_vol['days_to_event'], avg_vol['avg_annualized_vol'], 'b-', linewidth=2, label='Avg. Annualized Volatility')
        
        # Add vertical line at event day
        ax.axvline(x=0, color='r', linestyle='--', label='Event Day')
        ax.text(0.1, ax.get_ylim()[1] * 0.9, 'Event Day', color='r', ha='left')
        
        # Mark the three phases from the paper
        pre_event_end = -1
        post_event_rise_end = self.vol_params['delta']
        
        ax.axvspan(-10, pre_event_end, color='yellow', alpha=0.1, label='Pre-Event Phase')
        ax.axvspan(0, post_event_rise_end, color='green', alpha=0.1, label='Post-Event Rising Phase')
        ax.axvspan(post_event_rise_end, 20, color='blue', alpha=0.1, label='Post-Event Decay Phase')
        
        ax.set_title('GARCH-Estimated Volatility Around Events')
        ax.set_xlabel('Days Relative to Event')
        ax.set_ylabel('Annualized Volatility (%)')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        # Save plot
        plot_filename = os.path.join(results_dir, f"{file_prefix}_garch_volatility.png")
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        print(f"Saved GARCH volatility plot to: {plot_filename}")
        plt.close(fig)
        
        return avg_vol

    def analyze_impact_uncertainty(self, results_dir: str, file_prefix: str = "event"):
        """
        Analyze impact uncertainty from GARCH models as defined in the paper.
        
        Parameters:
        -----------
        results_dir : str
            Directory to save results
        file_prefix : str, optional
            Prefix for output files
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with impact uncertainty data
        """
        print("\n--- Analyzing Impact Uncertainty ---")
        
        if not self.garch_models:
            print("Error: No GARCH models available. Call fit_garch_models first.")
            return None
            
        # Create directory if it doesn't exist
        os.makedirs(results_dir, exist_ok=True)
            
        # Collect impact uncertainty for each event
        impact_uncertainty_data = []
        
        for event_id, garch_info in self.garch_models.items():
            garch_model = garch_info['model']
            event_day_idx = garch_info['event_day_idx']
            
            # Calculate impact uncertainty
            try:
                impact_uncertainty = garch_model.impact_uncertainty()
                
                # Get event window
                event_data = self.data.filter(pl.col('event_id') == event_id)
                days_to_event = event_data.select('days_to_event').to_series().to_list()
                
                # Match impact uncertainty to days_to_event
                # Ensure same length if needed
                if len(impact_uncertainty) > len(days_to_event):
                    impact_uncertainty = impact_uncertainty[-len(days_to_event):]
                elif len(impact_uncertainty) < len(days_to_event):
                    # Pad with the first value
                    padding = np.full(len(days_to_event) - len(impact_uncertainty), impact_uncertainty[0])
                    impact_uncertainty = np.concatenate([padding, impact_uncertainty])
                    
                for day, uncertainty in zip(days_to_event, impact_uncertainty):
                    impact_uncertainty_data.append({
                        'event_id': event_id,
                        'days_to_event': day,
                        'impact_uncertainty': uncertainty
                    })
            except Exception as e:
                print(f"Error calculating impact uncertainty for event {event_id}: {e}")
                
        # Convert to DataFrame
        impact_df = pd.DataFrame(impact_uncertainty_data)
        
        # Calculate average impact uncertainty for each day relative to event
        avg_impact = impact_df.groupby('days_to_event').agg(
            avg_impact=('impact_uncertainty', 'mean'),
            median_impact=('impact_uncertainty', 'median'),
            std_impact=('impact_uncertainty', 'std'),
            count=('impact_uncertainty', 'count')
        ).reset_index()
        
        # Save data
        csv_filename = os.path.join(results_dir, f"{file_prefix}_impact_uncertainty.csv")
        avg_impact.to_csv(csv_filename, index=False)
        print(f"Saved impact uncertainty data to: {csv_filename}")
        
        # Create plot
        fig, ax = plt.subplots(figsize=(12, 6))
        
        ax.plot(avg_impact['days_to_event'], avg_impact['avg_impact'], 'b-', linewidth=2, label='Avg. Impact Uncertainty')
        
        # Add vertical line at event day
        ax.axvline(x=0, color='r', linestyle='--', label='Event Day')
        ax.text(0.1, ax.get_ylim()[1] * 0.9, 'Event Day', color='r', ha='left')
        
        # Highlight pre-event period where impact uncertainty matters most
        ax.axvspan(-10, 0, color='yellow', alpha=0.2, label='Pre-Event (Impact Uncertainty Phase)')
        
        ax.set_title('Impact Uncertainty Around Events')
        ax.set_xlabel('Days Relative to Event')
        ax.set_ylabel('Impact Uncertainty')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        # Save plot
        plot_filename = os.path.join(results_dir, f"{file_prefix}_impact_uncertainty.png")
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        print(f"Saved impact uncertainty plot to: {plot_filename}")
        plt.close(fig)
        
        return avg_impact

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
        Analyzes the Return-to-Variance Ratio (RVR) using GARCH volatility models.

        Parameters:
        -----------
        results_dir: str
            Directory to save results
        file_prefix: str, optional
            Prefix for output files
        return_col: str, optional
            Column name containing returns
        analysis_window: tuple(int, int), optional 
            Range of days to analyze relative to event
        post_event_delta: int, optional
            Number of days in post-event rising phase
        lookback_window: int, optional
            Window size for calculating rolling statistics
        optimistic_bias: float, optional
            Bias parameter for post-event rising phase
        min_periods: int, optional
            Minimum number of observations for calculations
        variance_floor: float, optional
            Minimum variance value to prevent division by zero
        rvr_clip_threshold: float, optional
            Maximum absolute value for RVR to handle outliers
        adaptive_threshold: bool, optional
            Whether to adapt thresholds based on data

        Returns:
        --------
        pd.DataFrame
            DataFrame with RVR data
        """
        print(f"\n--- Analyzing Return-to-Variance Ratio (RVR) with GARCH ---")
        print(f"Analysis Window: {analysis_window}, Post-Event Delta: {post_event_delta} days")

        if self.data is None:
            print("Error: No data loaded. Call load_and_prepare_data first.")
            return None

        if not self.garch_models:
            print("Fitting GARCH models first...")
            self.fit_garch_models(return_col=return_col)
            
        if not self.garch_models:
            print("Error: No GARCH models available.")
            return None

        # Create directory if it doesn't exist
        os.makedirs(results_dir, exist_ok=True)

        # Define the three phases based on the paper
        phases_dict = {
            'pre_event': (analysis_window[0], -1),
            'post_event_rising': (0, post_event_delta),
            'late_post_event': (post_event_delta + 1, analysis_window[1])
        }

        # Collect RVR data for all events
        rvr_data = []
        
        # Process each event
        for event_id, garch_info in self.garch_models.items():
            event_asset = self.event_assets[event_id]
            event_day_idx = garch_info['event_day_idx']
            
            # Filter data for this event and analysis window
            event_data = self.data.filter(
                (pl.col('event_id') == event_id) &
                (pl.col('days_to_event') >= analysis_window[0]) &
                (pl.col('days_to_event') <= analysis_window[1])
            ).sort('days_to_event')
            
            if event_data.is_empty():
                continue
                
            # Get returns and days to event
            returns = event_data.select(pl.col(return_col)).to_series().to_numpy()
            days_to_event = event_data.select('days_to_event').to_series().to_numpy()
            
            # Skip if insufficient data
            if len(returns) < lookback_window + 1:
                continue
                
            # Get GARCH-estimated volatility
            volatility = np.sqrt(garch_info['model'].h)
            
            # Ensure volatility matches length of returns
            if len(volatility) > len(returns):
                volatility = volatility[-len(returns):]
            elif len(volatility) < len(returns):
                # Pad with the first value
                padding = np.full(len(returns) - len(volatility), volatility[0])
                volatility = np.concatenate([padding, volatility])
                
            # Calculate rolling mean of returns
            rolling_mean = np.zeros_like(returns)
            for i in range(len(returns)):
                if i < lookback_window:
                    # Use available data for first points
                    rolling_mean[i] = np.mean(returns[:i+1])
                else:
                    rolling_mean[i] = np.mean(returns[i-lookback_window:i])
                    
            # Apply optimistic bias to post-event rising phase
            expected_return = np.copy(rolling_mean)
            post_event_mask = (days_to_event >= phases_dict['post_event_rising'][0]) & (days_to_event <= phases_dict['post_event_rising'][1])
            expected_return[post_event_mask] += optimistic_bias
            
            # Calculate variance with floor
            variance = np.maximum(volatility**2, variance_floor)
            
            # Calculate RVR
            rvr = expected_return / variance
            
            # Clip extreme values
            rvr = np.clip(rvr, -rvr_clip_threshold, rvr_clip_threshold)
            
            # Store results
            for i, (day, r) in enumerate(zip(days_to_event, rvr)):
                rvr_data.append({
                    'event_id': event_id,
                    'days_to_event': day,
                    'expected_return': expected_return[i],
                    'variance': variance[i],
                    'rvr': r
                })
                
        # Convert to DataFrame
        rvr_df = pd.DataFrame(rvr_data)
        
        # Calculate average RVR for each day relative to event
        avg_rvr = rvr_df.groupby('days_to_event').agg(
            avg_rvr=('rvr', 'mean'),
            median_rvr=('rvr', 'median'),
            avg_expected_return=('expected_return', 'mean'),
            avg_variance=('variance', 'mean'),
            count=('rvr', 'count')
        ).reset_index()
        
        # Calculate phase summaries
        phase_summaries = []
        for phase_name, (start_day, end_day) in phases_dict.items():
            phase_data = rvr_df[(rvr_df['days_to_event'] >= start_day) & (rvr_df['days_to_event'] <= end_day)]
            
            if not phase_data.empty:
                phase_stats = {
                    'phase': phase_name,
                    'start_day': start_day,
                    'end_day': end_day,
                    'avg_rvr': phase_data['rvr'].mean(),
                    'median_rvr': phase_data['rvr'].median(),
                    'avg_expected_return': phase_data['expected_return'].mean(),
                    'avg_variance': phase_data['variance'].mean(),
                    'event_count': phase_data['event_id'].nunique()
                }
                phase_summaries.append(phase_stats)
                
        phase_summary_df = pd.DataFrame(phase_summaries)
        
        # Print phase statistics
        print("\nRVR by Phase:")
        for _, row in phase_summary_df.iterrows():
            print(f"Phase: {row['phase']} ({row['start_day']} to {row['end_day']} days)")
            print(f"  Avg RVR: {row['avg_rvr']:.4f}")
            print(f"  Median RVR: {row['median_rvr']:.4f}")
            print(f"  Avg Expected Return: {row['avg_expected_return']:.6f}")
            print(f"  Avg Variance: {row['avg_variance']:.6f}")
            print(f"  Events: {row['event_count']}")
            
        # Save data
        csv_filename = os.path.join(results_dir, f"{file_prefix}_rvr_daily.csv")
        avg_rvr.to_csv(csv_filename, index=False)
        print(f"Saved daily RVR data to: {csv_filename}")
        
        phase_csv_filename = os.path.join(results_dir, f"{file_prefix}_rvr_phase_summary.csv")
        phase_summary_df.to_csv(phase_csv_filename, index=False)
        print(f"Saved RVR phase summary to: {phase_csv_filename}")
        
        # Create plot
        fig, ax = plt.subplots(figsize=(12, 6))
        
        ax.plot(avg_rvr['days_to_event'], avg_rvr['avg_rvr'], 'b-', linewidth=2, label='Avg RVR')
        
        # Add vertical line at event day
        ax.axvline(x=0, color='r', linestyle='--', label='Event Day')
        ax.axvline(x=post_event_delta, color='purple', linestyle=':', label='End of Post-Event Rising')
        
        # Highlight the phases
        ax.axvspan(phases_dict['pre_event'][0], phases_dict['pre_event'][1], color='blue', alpha=0.1, label='Pre-Event')
        ax.axvspan(phases_dict['post_event_rising'][0], phases_dict['post_event_rising'][1], color='yellow', alpha=0.2, label='Post-Event Rising')
        ax.axvspan(phases_dict['late_post_event'][0], phases_dict['late_post_event'][1], color='green', alpha=0.1, label='Late Post-Event')
        
        y_lim = ax.get_ylim()
        ax.text(0.1, y_lim[1]*0.9, 'Event Day', color='r', ha='left')
        ax.text(post_event_delta + 0.1, y_lim[1]*0.85, 'End Post-Event Rising', color='purple', ha='left')
        
        # Add annotations for phase averages
        for _, row in phase_summary_df.iterrows():
            x_pos = (row['start_day'] + row['end_day']) / 2
            y_pos = row['avg_rvr']
            ax.annotate(f"{row['phase']}: {y_pos:.2f}",
                        xy=(x_pos, y_pos), xycoords='data',
                        xytext=(x_pos, y_pos + 0.5), textcoords='data',
                        arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"),
                        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.7))
        
        ax.set_title('Return-to-Variance Ratio Around Events')
        ax.set_xlabel('Days Relative to Event')
        ax.set_ylabel('Average RVR')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        # Save plot
        plot_filename = os.path.join(results_dir, f"{file_prefix}_rvr_timeseries.png")
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        print(f"Saved RVR time series plot to: {plot_filename}")
        plt.close(fig)
        
        return avg_rvr

    def decompose_returns(self, results_dir: str, file_prefix: str = "event",
                         return_col: str = 'ret', pre_event_window: int = 10,
                         post_event_window: int = 10):
        """
        Decompose returns into directional news risk and impact uncertainty components.
        
        Parameters:
        -----------
        results_dir : str
            Directory to save results
        file_prefix : str, optional
            Prefix for output files
        return_col : str, optional
            Column name containing returns
        pre_event_window : int, optional
            Number of days before the event to consider
        post_event_window : int, optional
            Number of days after the event to consider
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with decomposed returns
        """
        print("\n--- Decomposing Returns into Risk Components ---")
        
        if not self.garch_models:
            print("Error: No GARCH models available. Call fit_garch_models first.")
            return None
            
        # Create directory if it doesn't exist
        os.makedirs(results_dir, exist_ok=True)
            
        # Initialize TwoRiskFramework
        two_risk = TwoRiskFramework(**self.two_risk_params)
        
        # Collect decomposed returns for each event
        decomposed_returns = []
        
        for event_id, garch_info in self.garch_models.items():
            garch_model = garch_info['model']
            event_day_idx = garch_info['event_day_idx']
            
            # Filter data for this event
            event_data = self.data.filter(pl.col('event_id') == event_id)
            returns = event_data.select(pl.col(return_col)).to_series().to_numpy()
            days_to_event = event_data.select('days_to_event').to_series().to_numpy()
            
            # Skip if insufficient data
            if len(returns) < pre_event_window + post_event_window + 1:
                continue
                
            try:
                # Estimate risk premia
                risk_premia = two_risk.estimate_risk_premia(
                    returns,
                    garch_model,
                    event_day_idx,
                    pre_event_window=pre_event_window,
                    post_event_window=post_event_window
                )
                
                # Decompose returns
                decomposition = two_risk.decompose_returns(
                    returns,
                    event_day_idx,
                    pre_event_window=pre_event_window,
                    post_event_window=post_event_window
                )
                
                # Store decomposed returns with days_to_event
                for i, day in enumerate(days_to_event):
                    if i < len(decomposition['directional_risk']):
                        decomposed_returns.append({
                            'event_id': event_id,
                            'days_to_event': day,
                            'return': returns[i],
                            'directional_risk': decomposition['directional_risk'][i],
                            'impact_uncertainty': decomposition['impact_uncertainty'][i],
                            'total_risk': decomposition['total_risk'][i]
                        })
                        
                print(f"Decomposed returns for event {event_id}")
                print(f"  Impact uncertainty premium: {risk_premia['impact_uncertainty_premium']:.6f}")
                print(f"  Directional risk premium: {risk_premia['directional_risk_premium']:.6f}")
                
            except Exception as e:
                print(f"Error decomposing returns for event {event_id}: {e}")
                
        # Convert to DataFrame
        decomp_df = pd.DataFrame(decomposed_returns)
        
        if decomp_df.empty:
            print("Error: No decomposed returns available.")
            return None
            
        # Calculate average values for each day relative to event
        avg_decomp = decomp_df.groupby('days_to_event').agg(
            avg_return=('return', 'mean'),
            avg_directional_risk=('directional_risk', 'mean'),
            avg_impact_uncertainty=('impact_uncertainty', 'mean'),
            avg_total_risk=('total_risk', 'mean'),
            count=('return', 'count')
        ).reset_index()
        
        # Save data
        csv_filename = os.path.join(results_dir, f"{file_prefix}_return_decomposition.csv")
        avg_decomp.to_csv(csv_filename, index=False)
        print(f"Saved return decomposition data to: {csv_filename}")
        
        # Create plot
        fig, ax = plt.subplots(figsize=(12, 6))
        
        ax.plot(avg_decomp['days_to_event'], avg_decomp['avg_return'], 'k-', linewidth=2, label='Actual Returns')
        ax.plot(avg_decomp['days_to_event'], avg_decomp['avg_directional_risk'], 'b-', linewidth=1.5, label='Directional News Risk')
        ax.plot(avg_decomp['days_to_event'], avg_decomp['avg_impact_uncertainty'], 'r-', linewidth=1.5, label='Impact Uncertainty')
        
        # Add vertical line at event day
        ax.axvline(x=0, color='g', linestyle='--', label='Event Day')
        ax.text(0.1, ax.get_ylim()[1] * 0.9, 'Event Day', color='g', ha='left')
        
        # Highlight the pre-event and post-event periods
        ax.axvspan(-pre_event_window, 0, color='red', alpha=0.1, label='Pre-Event (Impact Uncertainty)')
        ax.axvspan(0, post_event_window, color='blue', alpha=0.1, label='Post-Event (Directional Risk)')
        
        ax.set_title('Return Decomposition: Directional News Risk vs. Impact Uncertainty')
        ax.set_xlabel('Days Relative to Event')
        ax.set_ylabel('Return Component')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        # Save plot
        plot_filename = os.path.join(results_dir, f"{file_prefix}_return_decomposition.png")
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        print(f"Saved return decomposition plot to: {plot_filename}")
        plt.close(fig)
        
        return avg_decomp

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

    def analyze_volatility_spikes(self, results_dir: str, file_prefix: str = "event", 
                                 window: int = 5, pre_days: int = 60, post_days: int = 60,
                                 baseline_window: tuple = (-60, -11), event_window: tuple = (-2, 2)):
        """
        Analyze volatility spikes around events.

        Parameters:
        -----------
        results_dir : str
            Directory to save results
        file_prefix : str, optional
            Prefix for output files
        window : int, optional
            Window size for rolling volatility calculation
        pre_days : int, optional
            Number of days before the event to consider
        post_days : int, optional
            Number of days after the event to consider
        baseline_window : tuple, optional
            Window for baseline volatility calculation (days_to_event)
        event_window : tuple, optional
            Window for event volatility calculation (days_to_event)

        Returns:
        --------
        pd.DataFrame
            DataFrame with volatility spike data
        """
        print("\n--- Analyzing Volatility Spikes ---")

        if self.data is None:
            print("Error: No data loaded. Call load_and_prepare_data first.")
            return None

        # Create directory if it doesn't exist
        os.makedirs(results_dir, exist_ok=True)

        # Collect volatility data for each event
        volatility_data = []

        # Get unique event IDs
        event_ids = self.data.select('event_id').unique().to_series().to_list()

        for event_id in event_ids:
            # Filter data for this event
            event_data = self.data.filter(pl.col('event_id') == event_id)

            # Sort by days_to_event
            event_data = event_data.sort('days_to_event')

            # Get returns
            returns = event_data.select('ret').to_series().to_numpy()
            days_to_event = event_data.select('days_to_event').to_series().to_numpy()

            # Skip if insufficient data
            if len(returns) < window + 1:
                continue

            # Calculate rolling volatility (standard deviation)
            rolling_vol = np.zeros_like(returns)

            for i in range(len(returns)):
                if i < window:
                    # Use available data for first points
                    rolling_vol[i] = np.std(returns[:i+1], ddof=1) if i > 0 else 0
                else:
                    rolling_vol[i] = np.std(returns[i-window+1:i+1], ddof=1)

            # Annualize volatility (approximate by multiplying by sqrt(252))
            annualized_vol = rolling_vol * np.sqrt(252) * 100  # Convert to percentage

            # Store results with days_to_event
            for day, vol in zip(days_to_event, annualized_vol):
                volatility_data.append({
                    'event_id': event_id,
                    'days_to_event': day,
                    'rolling_volatility': vol
                })

        # Convert to DataFrame
        vol_df = pd.DataFrame(volatility_data)

        if vol_df.empty:
            print("Error: No volatility data collected.")
            return None

        # Calculate average volatility for each day relative to event
        avg_vol = vol_df.groupby('days_to_event').agg(
            avg_vol=('rolling_volatility', 'mean'),
            median_vol=('rolling_volatility', 'median'),
            std_vol=('rolling_volatility', 'std'),
            count=('rolling_volatility', 'count')
        ).reset_index()

        # Calculate baseline and event period volatilities
        baseline_mask = (avg_vol['days_to_event'] >= baseline_window[0]) & (avg_vol['days_to_event'] <= baseline_window[1])
        event_mask = (avg_vol['days_to_event'] >= event_window[0]) & (avg_vol['days_to_event'] <= event_window[1])

        baseline_vol = avg_vol.loc[baseline_mask, 'avg_vol'].mean()
        event_vol = avg_vol.loc[event_mask, 'avg_vol'].mean()

        # Calculate volatility spike ratio
        vol_spike_ratio = event_vol / baseline_vol if baseline_vol > 0 else np.nan

        print(f"\nVolatility Analysis Results:")
        print(f"  Baseline Volatility ({baseline_window[0]} to {baseline_window[1]} days): {baseline_vol:.2f}%")
        print(f"  Event Volatility ({event_window[0]} to {event_window[1]} days): {event_vol:.2f}%")
        print(f"  Volatility Spike Ratio: {vol_spike_ratio:.2f}")

        # Save data
        csv_filename = os.path.join(results_dir, f"{file_prefix}_volatility_spikes.csv")
        avg_vol.to_csv(csv_filename, index=False)
        print(f"Saved volatility spikes data to: {csv_filename}")

        # Create stats summary file
        stats_df = pd.DataFrame({
            'Metric': ['Baseline Volatility', 'Event Volatility', 'Volatility Spike Ratio'],
            'Value': [baseline_vol, event_vol, vol_spike_ratio],
            'Window': [f"{baseline_window[0]} to {baseline_window[1]}", 
                      f"{event_window[0]} to {event_window[1]}", 
                      "Event/Baseline"]
        })

        stats_filename = os.path.join(results_dir, f"{file_prefix}_volatility_stats.csv")
        stats_df.to_csv(stats_filename, index=False)
        print(f"Saved volatility statistics to: {stats_filename}")

        # Create plot
        fig, ax = plt.subplots(figsize=(12, 6))

        ax.plot(avg_vol['days_to_event'], avg_vol['avg_vol'], 'b-', linewidth=2, label='Avg. Volatility')

        # Add vertical line at event day
        ax.axvline(x=0, color='r', linestyle='--', label='Event Day')

        # Highlight baseline and event windows
        ax.axvspan(baseline_window[0], baseline_window[1], color='lightblue', alpha=0.3, label='Baseline Window')
        ax.axvspan(event_window[0], event_window[1], color='lightgreen', alpha=0.3, label='Event Window')

        # Add horizontal lines for baseline and event volatility
        ax.axhline(y=baseline_vol, color='blue', linestyle=':', label=f'Baseline Vol: {baseline_vol:.2f}%')
        ax.axhline(y=event_vol, color='green', linestyle=':', label=f'Event Vol: {event_vol:.2f}%')

        ax.set_title('Volatility Around Events')
        ax.set_xlabel('Days Relative to Event')
        ax.set_ylabel('Annualized Volatility (%)')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)

        # Add volatility spike ratio annotation
        ax.annotate(f'Volatility Spike Ratio: {vol_spike_ratio:.2f}',
                   xy=(0.05, 0.05), xycoords='axes fraction',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

        # Save plot
        plot_filename = os.path.join(results_dir, f"{file_prefix}_volatility_spikes.png")
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        print(f"Saved volatility spike plot to: {plot_filename}")
        plt.close(fig)

        return avg_vol

    def analyze_mean_returns(self, results_dir: str, file_prefix: str = "event", return_col: str = 'ret',
                            window: int = 5, pre_days: int = 60, post_days: int = 60,
                            baseline_window: tuple = (-60, -11), event_window: tuple = (-2, 2)):
        """
        Analyze mean returns around events.

        Parameters:
        -----------
        results_dir : str
            Directory to save results
        file_prefix : str, optional
            Prefix for output files
        return_col : str, optional
            Column name containing returns
        window : int, optional
            Window size for rolling mean calculation
        pre_days : int, optional
            Number of days before the event to consider
        post_days : int, optional
            Number of days after the event to consider
        baseline_window : tuple, optional
            Window for baseline returns calculation (days_to_event)
        event_window : tuple, optional
            Window for event returns calculation (days_to_event)

        Returns:
        --------
        pd.DataFrame
            DataFrame with mean returns data
        """
        print("\n--- Analyzing Mean Returns ---")

        if self.data is None:
            print("Error: No data loaded. Call load_and_prepare_data first.")
            return None

        # Create directory if it doesn't exist
        os.makedirs(results_dir, exist_ok=True)

        # Collect returns data for each event
        returns_data = []

        # Get unique event IDs
        event_ids = self.data.select('event_id').unique().to_series().to_list()

        for event_id in event_ids:
            # Filter data for this event
            event_data = self.data.filter(pl.col('event_id') == event_id)

            # Sort by days_to_event
            event_data = event_data.sort('days_to_event')

            # Get returns
            returns = event_data.select(pl.col(return_col)).to_series().to_numpy()
            days_to_event = event_data.select('days_to_event').to_series().to_numpy()

            # Skip if insufficient data
            if len(returns) < window + 1:
                continue

            # Calculate rolling mean returns
            rolling_mean = np.zeros_like(returns)

            for i in range(len(returns)):
                if i < window:
                    # Use available data for first points
                    rolling_mean[i] = np.mean(returns[:i+1]) if i > 0 else returns[0]
                else:
                    rolling_mean[i] = np.mean(returns[i-window+1:i+1])

            # Store results with days_to_event
            for day, ret, roll_mean in zip(days_to_event, returns, rolling_mean):
                returns_data.append({
                    'event_id': event_id,
                    'days_to_event': day,
                    'return': ret,
                    'rolling_mean': roll_mean
                })

        # Convert to DataFrame
        ret_df = pd.DataFrame(returns_data)

        if ret_df.empty:
            print("Error: No returns data collected.")
            return None

        # Calculate average returns for each day relative to event
        avg_ret = ret_df.groupby('days_to_event').agg(
            avg_return=('return', 'mean'),
            avg_rolling_mean=('rolling_mean', 'mean'),
            median_return=('return', 'median'),
            std_return=('return', 'std'),
            count=('return', 'count')
        ).reset_index()

        # Calculate baseline and event period returns
        baseline_mask = (avg_ret['days_to_event'] >= baseline_window[0]) & (avg_ret['days_to_event'] <= baseline_window[1])
        event_mask = (avg_ret['days_to_event'] >= event_window[0]) & (avg_ret['days_to_event'] <= event_window[1])

        baseline_ret = avg_ret.loc[baseline_mask, 'avg_return'].mean()
        event_ret = avg_ret.loc[event_mask, 'avg_return'].mean()

        print(f"\nMean Returns Analysis Results:")
        print(f"  Baseline Return ({baseline_window[0]} to {baseline_window[1]} days): {baseline_ret:.6f}")
        print(f"  Event Return ({event_window[0]} to {event_window[1]} days): {event_ret:.6f}")

        # Save data
        csv_filename = os.path.join(results_dir, f"{file_prefix}_mean_returns.csv")
        avg_ret.to_csv(csv_filename, index=False)
        print(f"Saved mean returns data to: {csv_filename}")

        # Create stats summary file
        stats_df = pd.DataFrame({
            'Metric': ['Baseline Return', 'Event Return'],
            'Value': [baseline_ret, event_ret],
            'Window': [f"{baseline_window[0]} to {baseline_window[1]}", 
                      f"{event_window[0]} to {event_window[1]}"]
        })

        stats_filename = os.path.join(results_dir, f"{file_prefix}_returns_stats.csv")
        stats_df.to_csv(stats_filename, index=False)
        print(f"Saved returns statistics to: {stats_filename}")

        # Create plot
        fig, ax = plt.subplots(figsize=(12, 6))

        ax.plot(avg_ret['days_to_event'], avg_ret['avg_return'] * 100, 'b-', linewidth=2, label='Avg. Return (%)')
        ax.plot(avg_ret['days_to_event'], avg_ret['avg_rolling_mean'] * 100, 'g-', linewidth=1.5, label=f'Rolling Mean ({window}-day window) (%)')

        # Add vertical line at event day
        ax.axvline(x=0, color='r', linestyle='--', label='Event Day')

        # Highlight baseline and event windows
        ax.axvspan(baseline_window[0], baseline_window[1], color='lightblue', alpha=0.3, label='Baseline Window')
        ax.axvspan(event_window[0], event_window[1], color='lightgreen', alpha=0.3, label='Event Window')

        ax.set_title('Returns Around Events')
        ax.set_xlabel('Days Relative to Event')
        ax.set_ylabel('Return (%)')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)

        # Save plot
        plot_filename = os.path.join(results_dir, f"{file_prefix}_mean_returns.png")
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        print(f"Saved mean returns plot to: {plot_filename}")
        plt.close(fig)

        return avg_ret

    def calculate_volatility_quantiles(self, results_dir: str, file_prefix: str = "event", 
                                       return_col: str = 'ret', analysis_window: tuple = (-30, 30),
                                       lookback_window: int = 10, quantiles: list = [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]):
        """
        Calculate volatility quantiles for each day relative to the event.

        Parameters:
        -----------
        results_dir : str
            Directory to save results
        file_prefix : str, optional
            Prefix for output files
        return_col : str, optional
            Column name containing returns
        analysis_window : tuple, optional
            Window for analysis (days_to_event)
        lookback_window : int, optional
            Window size for rolling volatility calculation
        quantiles : list, optional
            Quantiles to calculate

        Returns:
        --------
        pd.DataFrame
            DataFrame with volatility quantiles
        """
        print(f"\n--- Calculating Volatility Quantiles ---")

        if self.data is None:
            print("Error: No data loaded. Call load_and_prepare_data first.")
            return None

        # Create directory if it doesn't exist
        os.makedirs(results_dir, exist_ok=True)

        # Collect volatility data for each event and day
        volatility_data = []

        # Get unique event IDs
        event_ids = self.data.select('event_id').unique().to_series().to_list()

        for event_id in event_ids:
            # Filter data for this event and analysis window
            event_data = self.data.filter(
                (pl.col('event_id') == event_id) &
                (pl.col('days_to_event') >= analysis_window[0]) &
                (pl.col('days_to_event') <= analysis_window[1])
            ).sort('days_to_event')

            # Skip if insufficient data
            if event_data.height < lookback_window + 1:
                continue

            # Get returns and days to event
            returns = event_data.select(pl.col(return_col)).to_series().to_numpy()
            days_to_event = event_data.select('days_to_event').to_series().to_numpy()

            # Calculate rolling volatility for each day
            for i in range(len(returns)):
                if i < lookback_window:
                    # Skip if not enough data for lookback
                    continue

                # Calculate volatility using lookback window
                vol = np.std(returns[i-lookback_window:i], ddof=1)

                # Annualize volatility
                annualized_vol = vol * np.sqrt(252) * 100  # Convert to percentage

                volatility_data.append({
                    'event_id': event_id,
                    'days_to_event': days_to_event[i],
                    'volatility': annualized_vol
                })

        # Convert to DataFrame
        vol_df = pd.DataFrame(volatility_data)

        if vol_df.empty:
            print("Error: No volatility data collected.")
            return None

        # Calculate quantiles for each day relative to event
        quantile_data = []

        for day in range(analysis_window[0], analysis_window[1] + 1):
            day_data = vol_df[vol_df['days_to_event'] == day]

            if day_data.empty:
                continue

            day_quantiles = {
                'days_to_event': day,
                'count': len(day_data),
                'mean': day_data['volatility'].mean(),
                'median': day_data['volatility'].median()
            }

            # Calculate quantiles
            for q in quantiles:
                day_quantiles[f'q{int(q*100)}'] = day_data['volatility'].quantile(q)

            quantile_data.append(day_quantiles)

        # Convert to DataFrame
        quantile_df = pd.DataFrame(quantile_data)

        # Save data
        csv_filename = os.path.join(results_dir, f"{file_prefix}_volatility_quantiles.csv")
        quantile_df.to_csv(csv_filename, index=False)
        print(f"Saved volatility quantiles to: {csv_filename}")

        # Create plot
        fig, ax = plt.subplots(figsize=(12, 6))

        # Plot median and mean
        ax.plot(quantile_df['days_to_event'], quantile_df['median'], 'b-', linewidth=2, label='Median')
        ax.plot(quantile_df['days_to_event'], quantile_df['mean'], 'g--', linewidth=1.5, label='Mean')

        # Plot selected quantiles
        quantile_colors = {
            5: 'lightblue',
            25: 'skyblue',
            75: 'sandybrown',
            95: 'salmon'
        }

        for q in [5, 25, 75, 95]:
            if f'q{q}' in quantile_df.columns:
                ax.plot(quantile_df['days_to_event'], quantile_df[f'q{q}'], '-', 
                      color=quantile_colors.get(q, 'gray'), alpha=0.7, linewidth=1,
                      label=f'{q}th Percentile')

        # Add vertical line at event day
        ax.axvline(x=0, color='r', linestyle='--', label='Event Day')

        ax.set_title('Volatility Quantiles Around Events')
        ax.set_xlabel('Days Relative to Event')
        ax.set_ylabel('Annualized Volatility (%)')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)

        # Save plot
        plot_filename = os.path.join(results_dir, f"{file_prefix}_volatility_quantiles.png")
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        print(f"Saved volatility quantiles plot to: {plot_filename}")
        plt.close(fig)

        return quantile_df

    def calculate_mean_returns_quantiles(self, results_dir: str, file_prefix: str = "event", 
                                         return_col: str = 'ret', analysis_window: tuple = (-30, 30),
                                         lookback_window: int = 10, quantiles: list = [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]):
        """
        Calculate mean returns quantiles for each day relative to the event.

        Parameters:
        -----------
        results_dir : str
            Directory to save results
        file_prefix : str, optional
            Prefix for output files
        return_col : str, optional
            Column name containing returns
        analysis_window : tuple, optional
            Window for analysis (days_to_event)
        lookback_window : int, optional
            Window size for rolling mean calculation
        quantiles : list, optional
            Quantiles to calculate

        Returns:
        --------
        pd.DataFrame
            DataFrame with mean returns quantiles
        """
        print(f"\n--- Calculating Mean Returns Quantiles ---")

        if self.data is None:
            print("Error: No data loaded. Call load_and_prepare_data first.")
            return None

        # Create directory if it doesn't exist
        os.makedirs(results_dir, exist_ok=True)

        # Collect mean returns data for each event and day
        returns_data = []

        # Get unique event IDs
        event_ids = self.data.select('event_id').unique().to_series().to_list()

        for event_id in event_ids:
            # Filter data for this event and analysis window
            event_data = self.data.filter(
                (pl.col('event_id') == event_id) &
                (pl.col('days_to_event') >= analysis_window[0]) &
                (pl.col('days_to_event') <= analysis_window[1])
            ).sort('days_to_event')

            # Skip if insufficient data
            if event_data.height < lookback_window + 1:
                continue

            # Get returns and days to event
            returns = event_data.select(pl.col(return_col)).to_series().to_numpy()
            days_to_event = event_data.select('days_to_event').to_series().to_numpy()

            # Calculate rolling mean returns for each day
            for i in range(len(returns)):
                if i < lookback_window:
                    # Skip if not enough data for lookback
                    continue

                # Calculate mean return using lookback window
                mean_return = np.mean(returns[i-lookback_window:i])

                returns_data.append({
                    'event_id': event_id,
                    'days_to_event': days_to_event[i],
                    'mean_return': mean_return
                })

        # Convert to DataFrame
        ret_df = pd.DataFrame(returns_data)

        if ret_df.empty:
            print("Error: No returns data collected.")
            return None

        # Calculate quantiles for each day relative to event
        quantile_data = []

        for day in range(analysis_window[0], analysis_window[1] + 1):
            day_data = ret_df[ret_df['days_to_event'] == day]

            if day_data.empty:
                continue

            day_quantiles = {
                'days_to_event': day,
                'count': len(day_data),
                'mean': day_data['mean_return'].mean(),
                'median': day_data['mean_return'].median()
            }

            # Calculate quantiles
            for q in quantiles:
                day_quantiles[f'q{int(q*100)}'] = day_data['mean_return'].quantile(q)

            quantile_data.append(day_quantiles)

        # Convert to DataFrame
        quantile_df = pd.DataFrame(quantile_data)

        # Save data
        csv_filename = os.path.join(results_dir, f"{file_prefix}_returns_quantiles.csv")
        quantile_df.to_csv(csv_filename, index=False)
        print(f"Saved mean returns quantiles to: {csv_filename}")

        # Create plot
        fig, ax = plt.subplots(figsize=(12, 6))

        # Plot median and mean
        ax.plot(quantile_df['days_to_event'], quantile_df['median'] * 100, 'b-', linewidth=2, label='Median')
        ax.plot(quantile_df['days_to_event'], quantile_df['mean'] * 100, 'g--', linewidth=1.5, label='Mean')

        # Plot selected quantiles
        quantile_colors = {
            5: 'lightblue',
            25: 'skyblue',
            75: 'sandybrown',
            95: 'salmon'
        }

        for q in [5, 25, 75, 95]:
            if f'q{q}' in quantile_df.columns:
                ax.plot(quantile_df['days_to_event'], quantile_df[f'q{q}'] * 100, '-', 
                      color=quantile_colors.get(q, 'gray'), alpha=0.7, linewidth=1,
                      label=f'{q}th Percentile')

        # Add vertical line at event day
        ax.axvline(x=0, color='r', linestyle='--', label='Event Day')

        ax.set_title('Mean Returns Quantiles Around Events')
        ax.set_xlabel('Days Relative to Event')
        ax.set_ylabel('Mean Return (%)')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)

        # Save plot
        plot_filename = os.path.join(results_dir, f"{file_prefix}_returns_quantiles.png")
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        print(f"Saved mean returns quantiles plot to: {plot_filename}")
        plt.close(fig)

        return quantile_df

    def calculate_rolling_sharpe_timeseries(self, results_dir: str, file_prefix: str = "event", 
                                           return_col: str = 'ret', analysis_window: tuple = (-30, 30),
                                           sharpe_window: int = 5, annualize: bool = True):
        """
        Calculate rolling Sharpe ratio time series around events.

        Parameters:
        -----------
        results_dir : str
            Directory to save results
        file_prefix : str, optional
            Prefix for output files
        return_col : str, optional
            Column name containing returns
        analysis_window : tuple, optional
            Window for analysis (days_to_event)
        sharpe_window : int, optional
            Window size for rolling Sharpe calculation
        annualize : bool, optional
            Whether to annualize the Sharpe ratio

        Returns:
        --------
        pd.DataFrame
            DataFrame with rolling Sharpe ratio time series
        """
        print(f"\n--- Calculating Rolling Sharpe Ratio Time Series ---")

        if self.data is None:
            print("Error: No data loaded. Call load_and_prepare_data first.")
            return None

        # Create directory if it doesn't exist
        os.makedirs(results_dir, exist_ok=True)

        # Collect Sharpe ratio data for each event and day
        sharpe_data = []

        # Get unique event IDs
        event_ids = self.data.select('event_id').unique().to_series().to_list()

        for event_id in event_ids:
            # Filter data for this event and analysis window
            event_data = self.data.filter(
                (pl.col('event_id') == event_id) &
                (pl.col('days_to_event') >= analysis_window[0]) &
                (pl.col('days_to_event') <= analysis_window[1])
            ).sort('days_to_event')

            # Skip if insufficient data
            if event_data.height < sharpe_window + 1:
                continue

            # Get returns and days to event
            returns = event_data.select(pl.col(return_col)).to_series().to_numpy()
            days_to_event = event_data.select('days_to_event').to_series().to_numpy()

            # Calculate rolling Sharpe ratio for each day
            for i in range(len(returns)):
                if i < sharpe_window:
                    # Skip if not enough data for window
                    continue

                # Calculate mean and standard deviation of returns in the window
                window_returns = returns[i-sharpe_window:i]
                mean_return = np.mean(window_returns)
                std_return = np.std(window_returns, ddof=1)

                # Calculate Sharpe ratio, handle division by zero
                if std_return > 0:
                    sharpe = mean_return / std_return

                    # Annualize if requested
                    if annualize:
                        sharpe = sharpe * np.sqrt(252)
                else:
                    sharpe = np.nan

                sharpe_data.append({
                    'event_id': event_id,
                    'days_to_event': days_to_event[i],
                    'sharpe_ratio': sharpe
                    })

            # Convert to DataFrame
            sharpe_df = pd.DataFrame(sharpe_data)

        if sharpe_df.empty:
            print("Error: No Sharpe ratio data collected.")
            return None

        # Calculate average Sharpe ratio for each day relative to event
        avg_sharpe = sharpe_df.groupby('days_to_event').agg(
            avg_sharpe=('sharpe_ratio', 'mean'),
            median_sharpe=('sharpe_ratio', 'median'),
            std_sharpe=('sharpe_ratio', 'std'),
            count=('sharpe_ratio', 'count')
        ).reset_index()

        # Save data
        csv_filename = os.path.join(results_dir, f"{file_prefix}_rolling_sharpe.csv")
        avg_sharpe.to_csv(csv_filename, index=False)
        print(f"Saved rolling Sharpe ratio data to: {csv_filename}")

        # Create plot
        fig, ax = plt.subplots(figsize=(12, 6))

        ax.plot(avg_sharpe['days_to_event'], avg_sharpe['avg_sharpe'], 'b-', linewidth=2, label='Avg. Sharpe Ratio')
        ax.plot(avg_sharpe['days_to_event'], avg_sharpe['median_sharpe'], 'g--', linewidth=1.5, label='Median Sharpe Ratio')

        # Add vertical line at event day
        ax.axvline(x=0, color='r', linestyle='--', label='Event Day')

        # Add horizontal line at zero
        ax.axhline(y=0, color='gray', linestyle='-', alpha=0.5)

        ax.set_title(f'Rolling Sharpe Ratio Around Events ({sharpe_window}-day window)')
        ax.set_xlabel('Days Relative to Event')
        ax.set_ylabel('Sharpe Ratio')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)

        # Save plot
        plot_filename = os.path.join(results_dir, f"{file_prefix}_rolling_sharpe.png")
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        print(f"Saved rolling Sharpe ratio plot to: {plot_filename}")
        plt.close(fig)

        return avg_sharpe

    def calculate_sharpe_quantiles(self, results_dir: str, file_prefix: str = "event", 
                                  return_col: str = 'ret', analysis_window: tuple = (-30, 30),
                                  lookback_window: int = 10, quantiles: list = [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95],
                                  annualize: bool = True):
        """
        Calculate Sharpe ratio quantiles for each day relative to the event.

        Parameters:
        -----------
        results_dir : str
            Directory to save results
        file_prefix : str, optional
            Prefix for output files
        return_col : str, optional
            Column name containing returns
        analysis_window : tuple, optional
            Window for analysis (days_to_event)
        lookback_window : int, optional
            Window size for rolling Sharpe calculation
        quantiles : list, optional
            Quantiles to calculate
        annualize : bool, optional
            Whether to annualize the Sharpe ratio

        Returns:
        --------
        pd.DataFrame
            DataFrame with Sharpe ratio quantiles
        """
        print(f"\n--- Calculating Sharpe Ratio Quantiles ---")

        if self.data is None:
            print("Error: No data loaded. Call load_and_prepare_data first.")
            return None

        # Create directory if it doesn't exist
        os.makedirs(results_dir, exist_ok=True)

        # Collect Sharpe ratio data for each event and day
        sharpe_data = []

        # Get unique event IDs
        event_ids = self.data.select('event_id').unique().to_series().to_list()

        for event_id in event_ids:
            # Filter data for this event and analysis window
            event_data = self.data.filter(
                (pl.col('event_id') == event_id) &
                (pl.col('days_to_event') >= analysis_window[0]) &
                (pl.col('days_to_event') <= analysis_window[1])
            ).sort('days_to_event')

            # Skip if insufficient data
            if event_data.height < lookback_window + 1:
                continue

            # Get returns and days to event
            returns = event_data.select(pl.col(return_col)).to_series().to_numpy()
            days_to_event = event_data.select('days_to_event').to_series().to_numpy()

            # Calculate rolling Sharpe ratio for each day
            for i in range(len(returns)):
                if i < lookback_window:
                    # Skip if not enough data for lookback
                    continue

                # Calculate Sharpe ratio using lookback window
                window_returns = returns[i-lookback_window:i]
                mean_return = np.mean(window_returns)
                std_return = np.std(window_returns, ddof=1)

                # Calculate Sharpe ratio, handle division by zero
                if std_return > 0:
                    sharpe = mean_return / std_return

                    # Annualize if requested
                    if annualize:
                        sharpe = sharpe * np.sqrt(252)
                else:
                    sharpe = np.nan

                sharpe_data.append({
                    'event_id': event_id,
                    'days_to_event': days_to_event[i],
                    'sharpe_ratio': sharpe
                })

        # Convert to DataFrame
        sharpe_df = pd.DataFrame(sharpe_data)

        if sharpe_df.empty:
            print("Error: No Sharpe ratio data collected.")
            return None

        # Calculate quantiles for each day relative to event
        quantile_data = []

        for day in range(analysis_window[0], analysis_window[1] + 1):
            day_data = sharpe_df[sharpe_df['days_to_event'] == day]

            if day_data.empty:
                continue

            day_quantiles = {
                'days_to_event': day,
                'count': len(day_data),
                'mean': day_data['sharpe_ratio'].mean(),
                'median': day_data['sharpe_ratio'].median()
            }

            # Calculate quantiles
            for q in quantiles:
                day_quantiles[f'q{int(q*100)}'] = day_data['sharpe_ratio'].quantile(q)

            quantile_data.append(day_quantiles)

        # Convert to DataFrame
        quantile_df = pd.DataFrame(quantile_data)

        # Save data
        csv_filename = os.path.join(results_dir, f"{file_prefix}_sharpe_quantiles.csv")
        quantile_df.to_csv(csv_filename, index=False)
        print(f"Saved Sharpe ratio quantiles to: {csv_filename}")

        # Create plot
        fig, ax = plt.subplots(figsize=(12, 6))

        # Plot median and mean
        ax.plot(quantile_df['days_to_event'], quantile_df['median'], 'b-', linewidth=2, label='Median')
        ax.plot(quantile_df['days_to_event'], quantile_df['mean'], 'g--', linewidth=1.5, label='Mean')

        # Plot selected quantiles
        quantile_colors = {
            5: 'lightblue',
            25: 'skyblue',
            75: 'sandybrown',
            95: 'salmon'
        }

        for q in [5, 25, 75, 95]:
            if f'q{q}' in quantile_df.columns:
                ax.plot(quantile_df['days_to_event'], quantile_df[f'q{q}'], '-', 
                      color=quantile_colors.get(q, 'gray'), alpha=0.7, linewidth=1,
                      label=f'{q}th Percentile')

        # Add vertical line at event day
        ax.axvline(x=0, color='r', linestyle='--', label='Event Day')

        # Add horizontal line at zero
        ax.axhline(y=0, color='gray', linestyle='-', alpha=0.5)

        ax.set_title('Sharpe Ratio Quantiles Around Events')
        ax.set_xlabel('Days Relative to Event')
        ax.set_ylabel('Sharpe Ratio')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)

        # Save plot
        plot_filename = os.path.join(results_dir, f"{file_prefix}_sharpe_quantiles.png")
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        print(f"Saved Sharpe ratio quantiles plot to: {plot_filename}")
        plt.close(fig)

        return quantile_df