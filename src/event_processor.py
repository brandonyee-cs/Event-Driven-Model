import polars as pl
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
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
    from src.models import TimeSeriesRidge, XGBoostDecileModel
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
                dates = self.data.select(time_split_column).to_numpy().flatten()
                sorted_indices = np.argsort(dates)
                split_idx = int(len(sorted_indices) * (1 - test_size))
                train_indices = sorted_indices[:split_idx]
                test_indices = sorted_indices[split_idx:]
                
                X_train, X_test = X[train_indices], X[test_indices]
                y_train, y_test = y[train_indices], y[test_indices]
            else:
                # Random split
                from sklearn.model_selection import train_test_split
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=42
                )
            
            # Initialize and train models
            from src.models import TimeSeriesRidge, XGBoostDecileModel
            
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
        Plot feature importance for a specified model.
        
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
                # For Ridge models, feature importance is based on coefficient magnitude
                importances = np.abs(model.coef_)
                indices = np.argsort(importances)[::-1]
            elif model_name == 'XGBoostDecile':
                # For XGBoost models, use built-in feature importance
                xgb_model = model.xgb_model
                importances = xgb_model.feature_importances_
                indices = np.argsort(importances)[::-1]
            else:
                print(f"Error: Feature importance not implemented for model type '{model_name}'.")
                return
            
            # Create feature importance plot using Plotly
            import plotly.graph_objects as go
            
            sorted_importances = importances[indices]
            sorted_features = [feature_names[i] for i in indices]
            
            # Limit to top 15 features for readability
            top_n = min(15, len(sorted_features))
            
            fig = go.Figure(go.Bar(
                x=sorted_importances[:top_n][::-1],
                y=sorted_features[:top_n][::-1],
                orientation='h'
            ))
            
            fig.update_layout(
                title=f'Feature Importance - {model_name}',
                xaxis_title='Importance',
                yaxis_title='Feature',
                template='plotly_white',
                width=900,
                height=500
            )
            
            # Save plot
            plot_filename = os.path.join(results_dir, f"{file_prefix}_feat_importance_{model_name}.png")
            fig.write_image(plot_filename, format='png', scale=2)
            print(f"Saved feature importance plot to: {plot_filename}")
            
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
        Plot predictions for a specific event using a trained model.
        
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
            event_data = self.data.filter(pl.col('event_id') == event_id)
            
            if event_data.is_empty():
                print(f"Error: No data found for event ID '{event_id}'.")
                return
                
            # Sort by days_to_event
            event_data = event_data.sort('days_to_event')
            
            # Extract features and target
            X, y, _ = self.feature_engineer.get_features_target(event_data, fit_imputer=False)
            
            if X.shape[0] == 0:
                print(f"Error: No valid features extracted for event ID '{event_id}'.")
                return
                
            # Make predictions
            model = self.models[model_name]['model']
            y_pred = model.predict(X)
            
            # Create plot
            import plotly.graph_objects as go
            
            days = event_data.get_column('days_to_event').to_numpy()
            actual_returns = event_data.get_column('future_ret').to_numpy() * 100  # Convert to percentage
            
            fig = go.Figure()
            
            # Add actual returns
            fig.add_trace(go.Scatter(
                x=days,
                y=actual_returns,
                mode='lines+markers',
                name='Actual Returns (%)',
                line=dict(color='blue')
            ))
            
            # Add predicted returns
            fig.add_trace(go.Scatter(
                x=days,
                y=y_pred * 100,  # Convert to percentage
                mode='lines+markers',
                name='Predicted Returns (%)',
                line=dict(color='red')
            ))
            
            # Add vertical line at event day
            fig.add_vline(x=0, line=dict(color='green', dash='dash'), annotation_text='Event Day')
            
            ticker = event_data.get_column('ticker').head(1).item()
            event_date = event_data.get_column('Event Date').head(1).item()
            
            fig.update_layout(
                title=f'Return Predictions - {ticker} Event on {event_date} - {model_name}',
                xaxis_title='Days Relative to Event',
                yaxis_title='Returns (%)',
                showlegend=True,
                template='plotly_white',
                width=1000,
                height=600
            )
            
            # Save plot
            plot_filename = os.path.join(results_dir, f"{file_prefix}_pred_{model_name}_{event_id}.png")
            fig.write_image(plot_filename, format='png', scale=2)
            print(f"Saved prediction plot to: {plot_filename}")
            
        except Exception as e:
            print(f"Error plotting predictions: {e}")
            traceback.print_exc()

    def calculate_rolling_sharpe_timeseries(self, 
    results_dir: str, 
    file_prefix: str = "event",
    return_col: str = 'ret', 
    analysis_window: Tuple[int, int] = (-60, 60),
    sharpe_window: int = 10,
    annualize: bool = True, 
    risk_free_rate: float = 0.0) -> Optional[pl.DataFrame]:
        """
        Calculates a time series of rolling Sharpe ratios around events using Polars.

        Parameters:
        results_dir (str): Directory to save results
        file_prefix (str): Prefix for saved files
        return_col (str): Column name containing returns
        analysis_window (Tuple[int, int]): Days relative to event to analyze (start, end)
        sharpe_window (int): Size of window for rolling Sharpe calculation in days
        annualize (bool): Whether to annualize the Sharpe ratio
        risk_free_rate (float): Annualized risk-free rate for Sharpe calculation

        Returns:
        pl.DataFrame: DataFrame containing Sharpe ratio time series or None if calculation fails
        """
        print(f"\n--- Calculating Rolling Sharpe Ratio Time Series ---")
        print(f"Analysis Window: {analysis_window}, Sharpe Window: {sharpe_window} days")

        # Input validation
        if self.data is None or return_col not in self.data.columns:
            print("Error: Data not loaded or missing return column.")
            return None

        if sharpe_window < 5:
            print("Warning: Sharpe window too small (<5 days). Setting to 5.")
            sharpe_window = 5

        # Calculate daily risk-free rate
        daily_rf = (1 + risk_free_rate) ** (1/252) - 1 if annualize and risk_free_rate > 0 else 0

        # Filter data within analysis window
        analysis_data = self.data.filter(
            (pl.col('days_to_event') >= analysis_window[0]) & 
            (pl.col('days_to_event') <= analysis_window[1])
        ).sort(['event_id', 'days_to_event'])

        if analysis_data.is_empty():
            print(f"Error: No data found within analysis window {analysis_window}.")
            return None

        # Calculate rolling statistics per event
        min_periods = max(3, sharpe_window // 2)
        rolling_stats = analysis_data.group_by(['event_id', 'days_to_event']).agg(
            mean_ret=pl.col(return_col).mean(),
            std_ret=pl.col(return_col).std(),
            n_obs=pl.col(return_col).count()
        ).filter(
            pl.col('n_obs') >= min_periods
        ).with_columns(
            sharpe_ratio=pl.when(pl.col('std_ret') > 0)
                         .then((pl.col('mean_ret') - daily_rf) / pl.col('std_ret'))
                         .otherwise(None)
        )

        # Aggregate across events by day
        daily_stats = rolling_stats.group_by('days_to_event').agg(
            avg_sharpe=pl.col('sharpe_ratio').mean(),
            std_sharpe=pl.col('sharpe_ratio').std(),
            event_count=pl.col('event_id').count(),
            avg_return=pl.col('mean_ret').mean(),
            avg_std=pl.col('std_ret').mean()
        ).sort('days_to_event')

        # Apply annualization if requested
        if annualize:
            daily_stats = daily_stats.with_columns(
                avg_sharpe=pl.col('avg_sharpe') * np.sqrt(252),
                avg_return=pl.col('avg_return') * 252,
                avg_std=pl.col('avg_std') * np.sqrt(252)
            )

        # Smooth the Sharpe ratio for visualization
        smooth_window = min(7, sharpe_window)  # Use a smaller smoothing window
        daily_stats = daily_stats.with_columns(
            smooth_sharpe=pl.col('avg_sharpe').rolling_mean(
                window_size=smooth_window,
                min_periods=smooth_window // 2,
                center=True
            )
        )

        # Ensure we have enough data
        if daily_stats.height < 5:
            print("Error: Insufficient data points for reliable Sharpe ratio calculation.")
            return None

        # Convert to pandas for plotting
        results_pd = daily_stats.to_pandas()

        # Create Plotly figure
        try:
            fig = go.Figure()

            # Add raw Sharpe ratio
            fig.add_trace(go.Scatter(
                x=results_pd['days_to_event'],
                y=results_pd['avg_sharpe'],
                mode='lines',
                name='Raw Sharpe Ratio',
                line=dict(color='blue', width=1),
                opacity=0.3
            ))

            # Add smoothed Sharpe ratio
            fig.add_trace(go.Scatter(
                x=results_pd['days_to_event'],
                y=results_pd['smooth_sharpe'],
                mode='lines',
                name=f'{smooth_window}-Day Smoothed',
                line=dict(color='red', width=2)
            ))

            # Add annotations and formatting
            fig.add_vline(x=0, line=dict(color='red', dash='dash'), 
                         annotation_text='Event Day')

            # Add month markers if window is wide enough
            if analysis_window[0] <= -30 and analysis_window[1] >= 30:
                fig.add_vline(x=-30, line=dict(color='green', dash='dot'), 
                             annotation_text='Month Before')
                fig.add_vline(x=30, line=dict(color='purple', dash='dot'), 
                             annotation_text='Month After')

            # Add event window highlight
            fig.add_vrect(x0=-2, x1=2, fillcolor='yellow', opacity=0.2, 
                         line_width=0, annotation_text='Event Window')

            # Dynamic y-axis range
            y_values = results_pd['smooth_sharpe'].dropna()
            if not y_values.empty:
                y_min = min(y_values.min(), -0.5)
                y_max = max(y_values.max(), 0.5)
                y_range = max(abs(y_min), abs(y_max)) * 1.2
            else:
                y_range = 1.0

            fig.update_layout(
                title=f'Rolling Sharpe Ratio Around Events (Window: {sharpe_window} days)',
                xaxis_title='Days Relative to Event',
                yaxis_title='Sharpe Ratio (Annualized)' if annualize else 'Sharpe Ratio',
                showlegend=True,
                template='plotly_white',
                width=1000,
                height=600,
                yaxis=dict(
                    range=[-y_range, y_range],
                    zeroline=True,
                    zerolinewidth=1,
                    zerolinecolor='black',
                    gridcolor='lightgray'
                )
            )

            # Save plot
            plot_filename = os.path.join(results_dir, f"{file_prefix}_rolling_sharpe_timeseries.png")
            try:
                fig.write_image(plot_filename, format='png', scale=2)
                print(f"Saved rolling Sharpe time series plot to: {plot_filename}")
            except Exception as e:
                print(f"Warning: Could not save plot image: {e}")
                print("To fix this issue, install the kaleido package: pip install -U kaleido")
                html_filename = os.path.join(results_dir, f"{file_prefix}_rolling_sharpe_timeseries.html")
                fig.write_html(html_filename)
                print(f"Saved as HTML (fallback) to: {html_filename}")

        except Exception as e:
            print(f"Error creating plot: {e}")
            import traceback
            traceback.print_exc()

        # Save data to CSV
        csv_filename = os.path.join(results_dir, f"{file_prefix}_rolling_sharpe_timeseries.csv")
        try:
            daily_stats.write_csv(csv_filename)
            print(f"Saved rolling Sharpe time series data to: {csv_filename}")
        except Exception as e:
            print(f"Error saving data: {e}")

        print(f"Completed Sharpe ratio calculation. Data points: {daily_stats.height}")
        return daily_stats
        
    def calculate_sharpe_quantiles(self, results_dir: str, file_prefix: str = "event",
                          return_col: str = 'ret', 
                          analysis_window: Tuple[int, int] = (-60, 60),
                          lookback_window: int = 10,
                          quantiles: List[float] = [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95],
                          annualize: bool = True, 
                          risk_free_rate: float = 0.0):
        """
        Calculates Sharpe ratio quantiles using vectorized operations.
        """
        print(f"\n--- Calculating Sharpe Ratio Quantiles (Analysis Window: {analysis_window}, Lookback: {lookback_window}) ---")

        if self.data is None or return_col not in self.data.columns:
            print("Error: Data not loaded or missing return column.")
            return None

        # Pre-filter the data
        extended_start = analysis_window[0] - lookback_window
        analysis_data = self.data.filter(
            (pl.col('days_to_event') >= extended_start) & 
            (pl.col('days_to_event') <= analysis_window[1])
        )

        if analysis_data.is_empty():
            print(f"Error: No data found within analysis window {analysis_window}.")
            return None

        # Calculate daily risk-free rate
        daily_rf = (1 + risk_free_rate) ** (1/252) - 1 if risk_free_rate > 0 else 0

        days_range = list(range(analysis_window[0], analysis_window[1] + 1))
        sharpe_data = []
        print(f"Processing {len(days_range)} days with vectorized operations...")

        all_event_ids = analysis_data.get_column('event_id').unique()
        print(f"Processing {len(all_event_ids)} unique events...")

        batch_size = 10
        for batch_start in range(0, len(days_range), batch_size):
            batch_days = days_range[batch_start:batch_start + batch_size]
            print(f"Processing batch days {batch_days[0]} to {batch_days[-1]} ({len(batch_days)} days)...")

            batch_results = []
            for center_day in batch_days:
                window_start = center_day - lookback_window
                window_end = center_day

                window_data = analysis_data.filter(
                    (pl.col('days_to_event') >= window_start) & 
                    (pl.col('days_to_event') <= window_end)
                )

                if window_data.is_empty():
                    empty_results = {"days_to_event": center_day, "event_count": 0}
                    for q in quantiles:
                        empty_results[f"sharpe_q{int(q*100)}"] = None
                    batch_results.append(empty_results)
                    continue

                sharpe_by_event = window_data.group_by('event_id').agg([
                    pl.mean(return_col).alias('mean_ret'),
                    pl.std(return_col).alias('std_dev'),
                    pl.count().alias('n_obs')
                ]).filter(
                    (pl.col('n_obs') >= max(3, lookback_window // 3)) &
                    (pl.col('std_dev') > 0)
                )

                valid_events = sharpe_by_event.height

                if valid_events >= 5:
                    sharpe_by_event = sharpe_by_event.with_columns([
                        ((pl.col('mean_ret') - daily_rf) / pl.col('std_dev') * 
                         (np.sqrt(252) if annualize else 1)).alias('sharpe')
                    ])

                    q_values = {}
                    for q in quantiles:
                        q_value = sharpe_by_event.select(
                            pl.col('sharpe').quantile(q, interpolation='linear').alias(f"q{int(q*100)}")
                        ).item(0, 0)
                        q_values[f"sharpe_q{int(q*100)}"] = q_value

                    day_results = {"days_to_event": center_day, "event_count": valid_events, **q_values}
                else:
                    day_results = {"days_to_event": center_day, "event_count": valid_events}
                    for q in quantiles:
                        day_results[f"sharpe_q{int(q*100)}"] = None

                batch_results.append(day_results)

            sharpe_data.extend(batch_results)

            del window_data, sharpe_by_event, batch_results
            import gc
            gc.collect()

        results_df = pl.DataFrame(sharpe_data)

        # Plot quantiles using Plotly
        try:
            results_pd = results_df.to_pandas()
            fig = go.Figure()
            for q in quantiles:
                col = f"sharpe_q{int(q*100)}"
                if col in results_pd.columns:
                    fig.add_trace(go.Scatter(
                        x=results_pd['days_to_event'],
                        y=results_pd[col],
                        mode='lines',
                        name=f'Q{int(q*100)}',
                        line=dict(width=2 if q == 0.5 else 1, dash='dash' if q != 0.5 else 'solid')
                    ))
            
            fig.add_vline(x=0, line=dict(color='red', dash='dash'), annotation_text='Event Day')
            if analysis_window[0] <= -30 and analysis_window[1] >= 30:
                fig.add_vline(x=-30, line=dict(color='green', dash='dot'), annotation_text='Month Before')
                fig.add_vline(x=30, line=dict(color='purple', dash='dot'), annotation_text='Month After')
            event_start, event_end = -2, 2
            if analysis_window[0] <= event_start and analysis_window[1] >= event_end:
                fig.add_vrect(x0=event_start, x1=event_end, fillcolor='yellow', opacity=0.2, line_width=0, annotation_text='Event Window')
            
            fig.update_layout(
                title=f'Sharpe Ratio Quantiles Around Events (Lookback: {lookback_window} days)',
                xaxis_title='Days Relative to Event',
                yaxis_title='Sharpe Ratio',
                showlegend=True,
                template='plotly_white',
                width=1000,
                height=600
            )
            
            plot_filename = os.path.join(results_dir, f"{file_prefix}_sharpe_quantiles.png")
            fig.write_image(plot_filename, format='png', scale=2)
            print(f"Saved Sharpe quantiles plot to: {plot_filename}")
            
        except Exception as e:
            print(f"Error creating plots: {e}")
            import traceback
            traceback.print_exc()

        # Save results to CSV
        csv_filename = os.path.join(results_dir, f"{file_prefix}_sharpe_quantiles.csv")
        try:
            results_df.write_csv(csv_filename)
            print(f"Saved Sharpe quantiles data to: {csv_filename}")
        except Exception as e:
            print(f"Error saving quantiles data: {e}")

        return results_df

    def analyze_volatility_spikes(self, results_dir: str, file_prefix: str = "event", window: int = 5, min_periods: int = 3, pre_days: int = 30, post_days: int = 30, baseline_window=(-60, -11), event_window=(-2, 2)):
        """
        Calculates and plots rolling volatility using Polars and Plotly.
        """
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

        # Plotting Rolling Volatility with Plotly
        if not aligned_vol.is_empty():
            aligned_vol_pd = aligned_vol.to_pandas()
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=aligned_vol_pd['days_to_event'],
                y=aligned_vol_pd['avg_annualized_vol'],
                mode='lines+markers',
                name='Avg. Annualized Volatility',
                line=dict(color='blue')
            ))
            
            fig.add_vline(x=0, line=dict(color='red', dash='dash'), annotation_text='Event Day')
            fig.add_vrect(x0=event_window[0], x1=event_window[1], fillcolor='yellow', opacity=0.2, line_width=0, annotation_text='Event Window')
            
            fig.update_layout(
                title=f'Average Rolling Volatility Around Event (Window={window} days)',
                xaxis_title='Days Relative to Event',
                yaxis_title='Avg. Annualized Volatility (%)',
                showlegend=True,
                template='plotly_white',
                width=1000,
                height=600
            )
            
            plot_filename_vol = os.path.join(results_dir, f"{file_prefix}_volatility_rolling_{window}d.png")
            csv_filename_vol = os.path.join(results_dir, f"{file_prefix}_volatility_rolling_{window}d_data.csv")
            try:
                fig.write_image(plot_filename_vol, format='png', scale=2)
                print(f"Saved rolling volatility plot to: {plot_filename_vol}")
            except Exception as e:
                print(f"Error saving plot: {e}")
            try:
                aligned_vol_pd.to_csv(csv_filename_vol)
                print(f"Saved rolling volatility data to: {csv_filename_vol}")
            except Exception as e:
                print(f"Error saving data: {e}")
        else:
            print("No data for rolling volatility plot.")

        return aligned_vol

    def calculate_volatility_quantiles(self, results_dir: str, file_prefix: str = "event",
                                 return_col: str = 'ret', 
                                 analysis_window: Tuple[int, int] = (-60, 60),
                                 lookback_window: int = 10,
                                 quantiles: List[float] = [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]):
        """
        Calculates volatility quantiles using vectorized operations.
        """
        print(f"\n--- Calculating Volatility Quantiles (Analysis Window: {analysis_window}, Lookback: {lookback_window}) ---")

        if self.data is None or return_col not in self.data.columns:
            print("Error: Data not loaded or missing return column.")
            return None

        extended_start = analysis_window[0] - lookback_window
        analysis_data = self.data.filter(
            (pl.col('days_to_event') >= extended_start) & 
            (pl.col('days_to_event') <= analysis_window[1])
        )

        if analysis_data.is_empty():
            print(f"Error: No data found within analysis window {analysis_window}.")
            return None

        days_range = list(range(analysis_window[0], analysis_window[1] + 1))
        vol_data = []
        print(f"Processing {len(days_range)} days with vectorized operations...")

        all_event_ids = analysis_data.get_column('event_id').unique()
        print(f"Processing {len(all_event_ids)} unique events...")

        batch_size = 10
        for batch_start in range(0, len(days_range), batch_size):
            batch_days = days_range[batch_start:batch_start + batch_size]
            print(f"Processing batch days {batch_days[0]} to {batch_days[-1]} ({len(batch_days)} days)...")

            batch_results = []
            for center_day in batch_days:
                window_start = center_day - lookback_window
                window_end = center_day

                window_data = analysis_data.filter(
                    (pl.col('days_to_event') >= window_start) & 
                    (pl.col('days_to_event') <= window_end)
                )

                if window_data.is_empty():
                    empty_results = {"days_to_event": center_day, "event_count": 0}
                    for q in quantiles:
                        empty_results[f"vol_q{int(q*100)}"] = None
                    batch_results.append(empty_results)
                    continue

                vol_by_event = window_data.group_by('event_id').agg([
                    pl.std(return_col).alias('vol'),
                    pl.count().alias('n_obs')
                ]).filter(
                    (pl.col('n_obs') >= max(3, lookback_window // 3)) &
                    (pl.col('vol').is_not_null())
                )

                valid_events = vol_by_event.height

                if valid_events >= 5:
                    vol_by_event = vol_by_event.with_columns(
                        (pl.col('vol') * np.sqrt(252) * 100).alias('annualized_vol')
                    )

                    q_values = {}
                    for q in quantiles:
                        q_value = vol_by_event.select(
                            pl.col('annualized_vol').quantile(q, interpolation='linear').alias(f"q{int(q*100)}")
                        ).item(0, 0)
                        q_values[f"vol_q{int(q*100)}"] = q_value

                    day_results = {"days_to_event": center_day, "event_count": valid_events, **q_values}
                else:
                    day_results = {"days_to_event": center_day, "event_count": valid_events}
                    for q in quantiles:
                        day_results[f"vol_q{int(q*100)}"] = None

                batch_results.append(day_results)

            vol_data.extend(batch_results)

            del window_data, vol_by_event, batch_results
            import gc
            gc.collect()

        results_df = pl.DataFrame(vol_data)

        # Plot quantiles using Plotly
        try:
            results_pd = results_df.to_pandas()
            fig = go.Figure()
            for q in quantiles:
                col = f"vol_q{int(q*100)}"
                if col in results_pd.columns:
                    fig.add_trace(go.Scatter(
                        x=results_pd['days_to_event'],
                        y=results_pd[col],
                        mode='lines',
                        name=f'Q{int(q*100)}',
                        line=dict(width=2 if q == 0.5 else 1, dash='dash' if q != 0.5 else 'solid')
                    ))

            fig.add_vline(x=0, line=dict(color='red', dash='dash'), annotation_text='Event Day')
            if analysis_window[0] <= -30 and analysis_window[1] >= 30:
                fig.add_vline(x=-30, line=dict(color='green', dash='dot'), annotation_text='Month Before')
                fig.add_vline(x=30, line=dict(color='purple', dash='dot'), annotation_text='Month After')
            event_start, event_end = -2, 2
            if analysis_window[0] <= event_start and analysis_window[1] >= event_end:
                fig.add_vrect(x0=event_start, x1=event_end, fillcolor='yellow', opacity=0.2, line_width=0, annotation_text='Event Window')

            fig.update_layout(
                title=f'Volatility Quantiles Around Events (Lookback: {lookback_window} days)',
                xaxis_title='Days Relative to Event',
                yaxis_title='Annualized Volatility (%)',
                showlegend=True,
                template='plotly_white',
                width=1000,
                height=600
            )

            try:
                # Try to save with kaleido
                plot_filename = os.path.join(results_dir, f"{file_prefix}_volatility_quantiles.png")
                fig.write_image(plot_filename, format='png', scale=2)
                print(f"Saved volatility quantiles plot to: {plot_filename}")
            except Exception as e:
                # Fall back to saving without image if kaleido is not available
                print(f"Warning: Could not save plot image: {e}")
                print("To fix this issue, install the kaleido package: pip install -U kaleido")
                # Save as HTML as fallback
                html_filename = os.path.join(results_dir, f"{file_prefix}_volatility_quantiles.html")
                fig.write_html(html_filename)
                print(f"Saved volatility quantiles as HTML (fallback) to: {html_filename}")

        except Exception as e:
            print(f"Error creating plots: {e}")
            import traceback
            traceback.print_exc()

        # Save results to CSV (this will work even if plotting fails)
        csv_filename = os.path.join(results_dir, f"{file_prefix}_volatility_quantiles.csv")
        try:
            results_df.write_csv(csv_filename)
            print(f"Saved volatility quantiles data to: {csv_filename}")
        except Exception as e:
            print(f"Error saving quantiles data: {e}")

        return results_df

    def analyze_mean_returns(self, results_dir: str, file_prefix: str = "event", 
                            return_col: str = 'ret', window: int = 5, 
                            min_periods: int = 3, pre_days: int = 60, 
                            post_days: int = 60, baseline_window=(-60, -11), 
                            event_window=(-2, 2)):
        """
        Calculates and plots rolling mean returns using Polars and Plotly.
        """
        print(f"\n--- Analyzing Rolling Mean Returns (Window={window} rows) using Polars ---")
        if self.data is None or return_col not in self.data.columns or 'days_to_event' not in self.data.columns:
            print("Error: Data/required columns missing."); return None

        df = self.data.sort(['event_id', 'date'])

        df = df.with_columns(
            pl.col(return_col).rolling_mean(window_size=window, min_periods=min_periods)
              .over('event_id').alias('rolling_mean_return')
        )
        df = df.with_columns(
            (pl.col('rolling_mean_return') * 100).alias('rolling_mean_return_pct')
        )

        aligned_means = df.group_by('days_to_event').agg(
            pl.mean('rolling_mean_return_pct').alias('avg_rolling_mean_return')
        ).filter(
            (pl.col('days_to_event') >= -pre_days) &
            (pl.col('days_to_event') <= post_days)
        ).sort('days_to_event').drop_nulls()

        # Plotting Rolling Mean Returns with Plotly
        if not aligned_means.is_empty():
            aligned_means_pd = aligned_means.to_pandas()
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=aligned_means_pd['days_to_event'],
                y=aligned_means_pd['avg_rolling_mean_return'],
                mode='lines+markers',
                name='Avg. Rolling Mean Return',
                line=dict(color='blue')
            ))

            fig.add_vline(x=0, line=dict(color='red', dash='dash'), annotation_text='Event Day')
            fig.add_hline(y=0, line=dict(color='gray'), opacity=0.3)
            fig.add_vrect(x0=event_window[0], x1=event_window[1], fillcolor='yellow', opacity=0.2, line_width=0, annotation_text='Event Window')

            fig.update_layout(
                title=f'Average Rolling Mean Return Around Event (Window={window} days)',
                xaxis_title='Days Relative to Event',
                yaxis_title='Avg. Rolling Mean Return (%)',
                showlegend=True,
                template='plotly_white',
                width=1000,
                height=600
            )

            plot_filename_mean = os.path.join(results_dir, f"{file_prefix}_mean_return_rolling_{window}d.png")
            csv_filename_mean = os.path.join(results_dir, f"{file_prefix}_mean_return_rolling_{window}d_data.csv")
            try:
                fig.write_image(plot_filename_mean, format='png', scale=2)
                print(f"Saved rolling mean return plot to: {plot_filename_mean}")
            except Exception as e:
                print(f"Warning: Could not save plot image: {e}")
                print("To fix this issue, install the kaleido package: pip install -U kaleido")
                # Save as HTML as fallback
                html_filename = os.path.join(results_dir, f"{file_prefix}_mean_return_rolling_{window}d.html")
                fig.write_html(html_filename)
                print(f"Saved mean return plot as HTML (fallback) to: {html_filename}")

            try:
                aligned_means_pd.to_csv(csv_filename_mean)
                print(f"Saved rolling mean return data to: {csv_filename_mean}")
            except Exception as e:
                print(f"Error saving data: {e}")
        else:
            print("No data for rolling mean returns plot.")

        return aligned_means

    def calculate_mean_returns_quantiles(self, results_dir: str, file_prefix: str = "event",
                                   return_col: str = 'ret', 
                                   analysis_window: Tuple[int, int] = (-60, 60),
                                   lookback_window: int = 10,
                                   quantiles: List[float] = [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]):
        """
        Calculates mean return quantiles using vectorized operations.
        """
        print(f"\n--- Calculating Mean Return Quantiles (Analysis Window: {analysis_window}, Lookback: {lookback_window}) ---")

        if self.data is None or return_col not in self.data.columns:
            print("Error: Data not loaded or missing return column.")
            return None

        extended_start = analysis_window[0] - lookback_window
        analysis_data = self.data.filter(
            (pl.col('days_to_event') >= extended_start) & 
            (pl.col('days_to_event') <= analysis_window[1])
        )

        if analysis_data.is_empty():
            print(f"Error: No data found within analysis window {analysis_window}.")
            return None

        days_range = list(range(analysis_window[0], analysis_window[1] + 1))
        mean_data = []
        print(f"Processing {len(days_range)} days with vectorized operations...")

        all_event_ids = analysis_data.get_column('event_id').unique()
        print(f"Processing {len(all_event_ids)} unique events...")

        batch_size = 10
        for batch_start in range(0, len(days_range), batch_size):
            batch_days = days_range[batch_start:batch_start + batch_size]
            print(f"Processing batch days {batch_days[0]} to {batch_days[-1]} ({len(batch_days)} days)...")

            batch_results = []
            for center_day in batch_days:
                window_start = center_day - lookback_window
                window_end = center_day

                window_data = analysis_data.filter(
                    (pl.col('days_to_event') >= window_start) & 
                    (pl.col('days_to_event') <= window_end)
                )

                if window_data.is_empty():
                    empty_results = {"days_to_event": center_day, "event_count": 0}
                    for q in quantiles:
                        empty_results[f"mean_q{int(q*100)}"] = None
                    batch_results.append(empty_results)
                    continue

                mean_by_event = window_data.group_by('event_id').agg([
                    pl.mean(return_col).alias('mean_ret'),
                    pl.count().alias('n_obs')
                ]).filter(
                    (pl.col('n_obs') >= max(3, lookback_window // 3)) &
                    (pl.col('mean_ret').is_not_null())
                )

                valid_events = mean_by_event.height

                if valid_events >= 5:
                    mean_by_event = mean_by_event.with_columns(
                        (pl.col('mean_ret') * 100).alias('mean_ret_pct')
                    )

                    q_values = {}
                    for q in quantiles:
                        q_value = mean_by_event.select(
                            pl.col('mean_ret_pct').quantile(q, interpolation='linear').alias(f"q{int(q*100)}")
                        ).item(0, 0)
                        q_values[f"mean_q{int(q*100)}"] = q_value

                    day_results = {"days_to_event": center_day, "event_count": valid_events, **q_values}
                else:
                    day_results = {"days_to_event": center_day, "event_count": valid_events}
                    for q in quantiles:
                        day_results[f"mean_q{int(q*100)}"] = None

                batch_results.append(day_results)

            mean_data.extend(batch_results)

            del window_data, mean_by_event, batch_results
            import gc
            gc.collect()

        results_df = pl.DataFrame(mean_data)

        # Plot quantiles using Plotly
        try:
            results_pd = results_df.to_pandas()
            fig = go.Figure()
            for q in quantiles:
                col = f"mean_q{int(q*100)}"
                if col in results_pd.columns:
                    fig.add_trace(go.Scatter(
                        x=results_pd['days_to_event'],
                        y=results_pd[col],
                        mode='lines',
                        name=f'Q{int(q*100)}',
                        line=dict(width=2 if q == 0.5 else 1, dash='dash' if q != 0.5 else 'solid')
                    ))

            fig.add_vline(x=0, line=dict(color='red', dash='dash'), annotation_text='Event Day')
            if analysis_window[0] <= -30 and analysis_window[1] >= 30:
                fig.add_vline(x=-30, line=dict(color='green', dash='dot'), annotation_text='Month Before')
                fig.add_vline(x=30, line=dict(color='purple', dash='dot'), annotation_text='Month After')
            event_start, event_end = -2, 2
            if analysis_window[0] <= event_start and analysis_window[1] >= event_end:
                fig.add_vrect(x0=event_start, x1=event_end, fillcolor='yellow', opacity=0.2, line_width=0, annotation_text='Event Window')

            fig.update_layout(
                title=f'Mean Return Quantiles Around Events (Lookback: {lookback_window} days)',
                xaxis_title='Days Relative to Event',
                yaxis_title='Mean Return (%)',
                showlegend=True,
                template='plotly_white',
                width=1000,
                height=600
            )

            try:
                # Try to save with kaleido
                plot_filename = os.path.join(results_dir, f"{file_prefix}_mean_return_quantiles.png")
                fig.write_image(plot_filename, format='png', scale=2)
                print(f"Saved mean return quantiles plot to: {plot_filename}")
            except Exception as e:
                # Fall back to saving without image if kaleido is not available
                print(f"Warning: Could not save plot image: {e}")
                print("To fix this issue, install the kaleido package: pip install -U kaleido")
                # Save as HTML as fallback
                html_filename = os.path.join(results_dir, f"{file_prefix}_mean_return_quantiles.html")
                fig.write_html(html_filename)
                print(f"Saved mean return quantiles as HTML (fallback) to: {html_filename}")

        except Exception as e:
            print(f"Error creating plots: {e}")
            import traceback
            traceback.print_exc()

        # Save results to CSV (this will work even if plotting fails)
        csv_filename = os.path.join(results_dir, f"{file_prefix}_mean_return_quantiles.csv")
        try:
            results_df.write_csv(csv_filename)
            print(f"Saved mean return quantiles data to: {csv_filename}")
        except Exception as e:
            print(f"Error saving quantiles data: {e}")

        return results_df