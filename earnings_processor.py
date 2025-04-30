import polars as pl
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report
from sklearn.impute import SimpleImputer
import xgboost as xgb
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
    # Keep __init__ and _load_single_stock_parquet as they were in the previous correction
    # _load_single_stock_parquet is used within the chunked load_data now.
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
        # --- THIS METHOD IS UNCHANGED ---
        # (Same as previous corrected version)
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

            stock_data = stock_data.select(selected_cols)
            if rename_dict:
                stock_data = stock_data.rename(rename_dict)

            required_cols = ['date', 'ticker', 'prc', 'ret', 'vol']
            missing_cols = [col for col in required_cols if col not in stock_data.columns]
            if missing_cols:
                 essential_pr = ['prc', 'ret']
                 if any(col in missing_cols for col in essential_pr):
                      raise ValueError(f"Missing essential columns after standardization in {stock_path}: {[c for c in essential_pr if c in missing_cols]}")
                 else:
                      # print(f"  Warning: Missing optional columns in {stock_path}: {missing_cols}. Some features might be skipped.")
                      pass # Keep less verbose

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

            final_cols = list(standard_names.keys())
            cols_present = [col for col in final_cols if col in stock_data.columns]
            stock_data = stock_data.select(cols_present)
            return stock_data
        except FileNotFoundError: return None
        except Exception: return None


    def load_data(self) -> Optional[pl.DataFrame]:
        """
        Load earnings event dates (CSV) and stock data (PARQUET) using Polars,
        processing events in chunks to manage memory. Includes debugging prints.
        """
        # --- Load ALL Unique Earnings Event Dates First ---
        try:
            print(f"Loading earnings event dates from: {self.earnings_path} (CSV)")
            # ... (finding ticker_col and date_col='ANNDATS' remains the same) ...
            event_df_peek = pl.read_csv_batched(self.earnings_path, batch_size=1).next_batches(1)[0]
            ticker_col = next((c for c in ['TICKER', 'ticker', 'Ticker', 'symbol', 'tic'] if c in event_df_peek.columns), None)
            if not ticker_col: raise ValueError("Missing Ticker column in event file.")
            date_col = 'ANNDATS'
            if date_col not in event_df_peek.columns: raise ValueError(f"Required announcement date column '{date_col}' not found.")

            print(f"Using columns '{ticker_col}' (as ticker) and '{date_col}' (as Announcement Date) from event file.")
            event_data_raw = pl.read_csv(self.earnings_path, columns=[ticker_col, date_col], try_parse_dates=True) # Try Polars parsing
            event_data_renamed = event_data_raw.rename({ticker_col: 'ticker', date_col: 'Announcement Date'})

            # Try converting Announcement Date (handle potential non-standard formats)
            event_data_processed = event_data_renamed.with_columns([
                pl.col('Announcement Date').str.to_datetime(strict=False), # Use default formats
                # Add more specific formats if needed: .str.to_datetime(format="%Y%m%d", strict=False) etc.
                pl.col('ticker').cast(pl.Utf8).str.to_uppercase()
            ]).drop_nulls(subset=['Announcement Date'])

            earnings_events = event_data_processed.unique(subset=['ticker', 'Announcement Date'], keep='first')
            n_total_events = earnings_events.height

            # *** DEBUG: Print sample event data ***
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

            # Determine required tickers and date range for this chunk
            chunk_tickers = event_chunk['ticker'].unique()
            min_ann_date = event_chunk['Announcement Date'].min()
            max_ann_date = event_chunk['Announcement Date'].max()
            buffer = pl.duration(days=self.window_days + 1) # Add buffer for safety
            required_min_date = min_ann_date - buffer
            required_max_date = max_ann_date + buffer

            # *** DEBUG: Print chunk info ***
            print(f"    Chunk Tickers: {chunk_tickers.len()} (Sample: {chunk_tickers[:5].to_list()})")
            print(f"    Chunk Ann Date Range: {min_ann_date} to {max_ann_date}")
            print(f"    Required Stock Date Range: {required_min_date} to {required_max_date}")

            stock_scans = []
            failed_stock_loads = 0
            # --- Scan/Filter Stock Data (Lazy) ---
            print("    Scanning and filtering stock Parquet files (lazily)...")
            for stock_path in self.stock_paths:
                try:
                    scan = pl.scan_parquet(stock_path)
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

                    scan = scan.select(selected_orig_cols)
                    if rename_dict: scan = scan.rename(rename_dict)

                    scan = scan.with_columns([
                        pl.col("date").str.to_datetime(strict=False).cast(pl.Datetime), # Ensure correct type
                        pl.col("ticker").cast(pl.Utf8).str.to_uppercase()
                    ])

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

            # *** DEBUG: Estimate filtered size before collecting ***
            try:
                # schema = combined_stock_scan.schema # Fetch schema first
                # estimated_rows = combined_stock_scan.select(pl.count()).collect().item() # Estimate count
                # print(f"    Estimated stock rows to collect: {estimated_rows}")
                # Note: Estimating rows can sometimes be slow or inaccurate depending on source
                pass
            except Exception as est_e:
                 print(f"    Warning: Could not estimate rows before collect: {est_e}")

            # --- Collect Actual Stock Data ---
            print(f"    Collecting filtered stock data for chunk {i+1}...")
            try:
                stock_data_chunk = combined_stock_scan.collect(streaming=True)
            except Exception as e:
                 print(f"    ERROR collecting stock data for chunk {i+1}: {e}. Skipping chunk.")
                 traceback.print_exc() # Print full traceback for collection errors
                 continue

            print(f"    Collected {stock_data_chunk.height} stock rows.")
            # *** DEBUG: Print sample collected stock data ***
            if not stock_data_chunk.is_empty():
                 print("    Sample collected stock data:")
                 print(stock_data_chunk.head(3))
                 # Check date range collected
                 min_coll_date = stock_data_chunk['date'].min()
                 max_coll_date = stock_data_chunk['date'].max()
                 print(f"    Collected Stock Date Range: {min_coll_date} to {max_coll_date}")
            else:
                 print("    Collected stock data is empty. Skipping rest of chunk processing.")
                 continue # Skip to next chunk if no stock data found

            # --- Standardize Types and Deduplicate ---
            # (Moved type standardization earlier, just deduplicate now)
            stock_data_chunk = stock_data_chunk.unique(subset=['date', 'ticker'], keep='first', maintain_order=False)
            print(f"    Deduplicated stock rows: {stock_data_chunk.height}")
            if stock_data_chunk.is_empty():
                 print(f"    Warning: No stock data remained after deduplication for chunk {i+1}. Skipping chunk.")
                 continue

            # --- Merge event chunk with stock data chunk ---
            print(f"    Merging events with stock data...")
            # Ensure join columns are same type
            event_chunk = event_chunk.with_columns(pl.col('ticker').cast(pl.Utf8))
            stock_data_chunk = stock_data_chunk.with_columns(pl.col('ticker').cast(pl.Utf8))
            merged_chunk = stock_data_chunk.join(
                event_chunk, on='ticker', how='inner'
            )
            print(f"    Merged chunk rows: {merged_chunk.height}")
            if merged_chunk.is_empty():
                print(f"    Warning: Merge resulted in empty data for chunk {i+1}. Check ticker matching.")
                # *** DEBUG: Check if tickers existed in stock_data_chunk ***
                # stock_tickers_in_chunk = stock_data_chunk['ticker'].unique()
                # event_tickers_in_chunk = event_chunk['ticker'].unique()
                # common_tickers = stock_tickers_in_chunk.filter(pl.col('ticker').is_in(event_tickers_in_chunk))
                # print(f"    Common tickers found: {common_tickers.len()}")
                # print(f"    Sample stock tickers in collected data: {stock_tickers_in_chunk[:5].to_list()}")
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
            final_cols = [c for c in final_cols if c in processed_chunk.columns]
            processed_chunk = processed_chunk.select(final_cols)

            print(f"    Processed chunk {i+1} FINAL shape: {processed_chunk.shape}")
            processed_chunks.append(processed_chunk)
            print(f"--- Finished processing chunk {i+1} ---")

            # Optional: Clean up memory explicitly
            del stock_data_chunk, merged_chunk, event_chunk, processed_chunk, combined_stock_scan, stock_scans
            gc.collect()


        # --- Final Concatenation ---
        if not processed_chunks:
            print("Error: No data chunks were processed successfully.") # This error will now be raised below
            return None # Return None if no chunks succeeded

        print("\nConcatenating processed chunks...")
        combined_data = pl.concat(processed_chunks, how='vertical').sort(['ticker', 'Announcement Date', 'date'])
        print(f"Final Earnings dataset shape: {combined_data.shape}")
        mem_usage_mb = combined_data.estimated_size("mb")
        print(f"Final DataFrame memory usage: {mem_usage_mb:.2f} MB")

        if combined_data.is_empty():
             print("Warning: Final combined data is empty after chunk processing.")
             # This condition should ideally be caught by the 'if not processed_chunks' earlier
             return None

        return combined_data

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

        # Add all calculated features so far
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
             df = df.with_columns([pl.lit(None).alias(c) for c in ['norm_vol', 'vol_momentum_5', 'vol_momentum_10']])
             current_features.extend(['norm_vol', 'vol_momentum_5', 'vol_momentum_10'])


        # --- Pre-announcement Return (Requires group_by, agg, join) ---
        pre_announce_start_offset = pl.duration(days=-30)
        pre_announce_data = df.filter(
            (pl.col('date') < pl.col('Announcement Date')) &
            (pl.col('date') >= (pl.col('Announcement Date') + pre_announce_start_offset))
        )
        # Calculate compound return: (1+ret).product() - 1
        # Polars product needs care with nulls. Fill null returns with 0 for product.
        pre_announce_agg = pre_announce_data.group_by('event_id').agg(
             # Compound return: product(1 + ret) - 1
             (pl.col(return_col).fill_null(0) + 1).product().alias("prod_ret_factor")
        ).with_columns(
            (pl.col("prod_ret_factor") - 1).alias('pre_announce_ret_30d')
        ).select(['event_id', 'pre_announce_ret_30d'])

        # Join back to the main dataframe
        df = df.join(pre_announce_agg, on='event_id', how='left')
        # Fill NaNs for events with no pre-announce data (set to 0)
        df = df.with_columns(pl.col('pre_announce_ret_30d').fill_null(0))
        current_features.append('pre_announce_ret_30d')


        # --- Earnings Surprise Features (Conditional) ---
        # Assumes 'Surprise' might be present from DataLoader (though current loader doesn't add it)
        # Adapt this if surprise data comes from a different source or is joined earlier
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
                pl.col('surprise_val').shift(1).over('ticker').fill_null(0).alias('prev_surprise') # Shift over ticker
            ])
            df = df.with_columns([
                 ((pl.col('surprise_val') > 0) & (pl.col('prev_surprise') > 0)).cast(pl.Int8).alias('consecutive_beat'),
                 ((pl.col('surprise_val') < 0) & (pl.col('prev_surprise') < 0)).cast(pl.Int8).alias('consecutive_miss')
            ])
            current_features.extend(surprise_cols_to_add)
        else:
            print("Info: 'Surprise' column not present in data. Skipping surprise features.")
            df = df.with_columns([pl.lit(None).cast(pl.Float64).alias(c) for c in surprise_cols_to_add if c != 'pos_surprise' and c != 'neg_surprise' and c!= 'consecutive_beat' and c!= 'consecutive_miss'] +
                                 [pl.lit(None).cast(pl.Int8).alias(c) for c in ['pos_surprise', 'neg_surprise', 'consecutive_beat', 'consecutive_miss']] )
            current_features.extend(surprise_cols_to_add)


        # --- Announcement Time Features (Conditional) ---
        time_features = ['announcement_hour', 'is_bmo', 'is_amc', 'is_market_hours']
        if 'Time' in df.columns:
             print("Info: 'Time' column found. Calculating time features.")
             df = df.with_columns(
                 pl.col('Time').str.to_time(strict=False, format="%H:%M:%S", errors='null').alias('time_parsed')
                 # Add more formats if needed: e.g., .str.to_time(format="%H:%M")
             )
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
                 pl.col('Quarter').cast(pl.Utf8).str.extract(r"Q(\d)", 1) # Extract first group
                 .cast(pl.Int8, strict=False).alias('quarter_num') # Cast to Int8, errors become null
             )
             quarter_expressions = []
             for i in range(1, 5):
                 col_name = f'is_q{i}'
                 quarter_expressions.append(
                     (pl.col('quarter_num') == i).fill_null(False).cast(pl.Int8).alias(col_name)
                 )
                 self.categorical_features.append(col_name) # Mark as categorical
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
                 # Create dummies on the current df (train or test)
                 current_sector_dummies = df.select('Sector').to_dummies(columns=['Sector'], drop_first=True)
                 # Add missing dummy columns found during fit (with value 0)
                 missing_dummies = set(self.sector_dummies_cols) - set(current_sector_dummies.columns)
                 if missing_dummies:
                     current_sector_dummies = current_sector_dummies.with_columns(
                         [pl.lit(0).cast(pl.UInt8).alias(col) for col in missing_dummies]
                     )
                 # Select only the fitted columns in the correct order and add to main df
                 df = pl.concat([df, current_sector_dummies.select(self.sector_dummies_cols)], how="horizontal")
                 current_features.extend(self.sector_dummies_cols)
                 self.categorical_features.extend(self.sector_dummies_cols)
            else: print("Info: Sector dummies not fitted/available.")
        else: print("Info: 'Sector' column not present in data. Skipping sector features.")

        # Handle Industry
        if 'Industry' in df.columns:
            print("Info: 'Industry' column found. Processing industry features.")
            df = df.with_columns(pl.col('Industry').fill_null('Unknown').cast(pl.Utf8))
            if fit_categorical:
                top_n = 20
                self.top_industries = df['Industry'].value_counts().sort(by="counts", descending=True).head(top_n)['Industry'].to_list()
                df_temp = df.with_columns(
                    pl.when(pl.col('Industry').is_in(self.top_industries))
                    .then(pl.col('Industry'))
                    .otherwise(pl.lit('Other_Industry'))
                    .alias('Industry_Top')
                )
                industry_dummies_df = df_temp.select('Industry_Top').to_dummies(columns=['Industry_Top'], drop_first=True)
                self.industry_dummies_cols = industry_dummies_df.columns
                print(f"Learned {len(self.industry_dummies_cols)} industry dummies (Top 20 + Other).")

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
                df = pl.concat([df, current_industry_dummies.select(self.industry_dummies_cols)], how="horizontal")
                current_features.extend(self.industry_dummies_cols)
                self.categorical_features.extend(self.industry_dummies_cols)
            else: print("Info: Industry dummies not fitted/available.")
        else: print("Info: 'Industry' column not present in data. Skipping industry features.")

        # Replace any infinities generated during calculations (should be less likely now)
        # Select only feature columns and apply replacement
        feature_cols_in_df = [f for f in current_features if f in df.columns]
        df = df.with_columns([
            pl.when(pl.col(c).is_infinite())
            .then(None)
            .otherwise(pl.col(c))
            .alias(c)
            for c in feature_cols_in_df if df[c].dtype in [pl.Float32, pl.Float64] # Only for float cols
        ])

        self.feature_names = sorted(list(set(current_features)))
        print(f"Calculated/checked {len(self.feature_names)} raw Earnings features.")
        # Return only relevant columns (original + features) to avoid carrying intermediate ones
        final_cols_to_keep = [c for c in df.columns if c in required or c in self.feature_names or c == 'event_id' or c == 'future_ret']
        # Add back key identifiers if dropped somehow: 'ticker', 'date', 'Announcement Date'
        for key_col in ['ticker', 'date', 'Announcement Date']:
             if key_col not in final_cols_to_keep and key_col in df.columns:
                 final_cols_to_keep.append(key_col)
        # Ensure unique columns
        final_cols_to_keep = sorted(list(set(final_cols_to_keep)))
        # Ensure columns actually exist
        final_cols_to_keep = [c for c in final_cols_to_keep if c in df.columns]

        return df.select(final_cols_to_keep)


    def get_features_target(self, df: pl.DataFrame, fit_imputer: bool = False) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Extract feature matrix X and target vector y as NumPy arrays, handling missing values.

        Returns:
        Tuple[np.ndarray, np.ndarray, List[str]]: X_np, y_np, final_feature_names
        """
        print("Extracting Earnings features (X) and target (y) as NumPy...")
        if not self.feature_names: raise RuntimeError("Run calculate_features first.")

        available_features = [f for f in self.feature_names if f in df.columns]
        if not available_features: raise ValueError("No calculated features found in DataFrame.")

        # Ensure 'future_ret' exists
        if 'future_ret' not in df.columns:
             raise ValueError("Target variable 'future_ret' not found in DataFrame.")

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

        # Separate numeric and categorical features (including dummies)
        numeric_features = [f for f in available_features if f not in self.categorical_features]
        categorical_df = df_with_target.select([f for f in available_features if f in self.categorical_features]) # Keep as Polars for now
        X_numeric_pl = df_with_target.select(numeric_features)
        y_pl = df_with_target.get_column('future_ret')

        print(f"Original numeric X (Polars): {X_numeric_pl.shape}. Categorical X (Polars): {categorical_df.shape}. Non-null y: {y_pl.len()}")

        # Convert numeric features to NumPy for imputation
        try:
            X_numeric_np = X_numeric_pl.cast(pl.Float64).to_numpy() # Cast to float for imputer
        except Exception as e:
             raise ValueError(f"Failed to convert numeric features to NumPy: {e}")

        initial_nan_count = np.isnan(X_numeric_np).sum()
        if initial_nan_count > 0: print(f"  Numeric features contain {initial_nan_count} NaN values before imputation.")

        # Impute missing values using scikit-learn SimpleImputer on NumPy array
        if fit_imputer:
            print("Fitting imputer on numeric NumPy data...")
            self.imputer.fit(X_numeric_np)
            self._imputer_fitted = True
            print("Transforming with imputer...")
            X_numeric_imputed_np = self.imputer.transform(X_numeric_np)
        else:
            if not self._imputer_fitted: raise RuntimeError("Imputer not fitted. Call with fit_imputer=True first.")
            print("Transforming numeric NumPy data with pre-fitted imputer...")
            X_numeric_imputed_np = self.imputer.transform(X_numeric_np)

        final_nan_count_numeric = np.isnan(X_numeric_imputed_np).sum()
        if final_nan_count_numeric > 0:
            warnings.warn(f"NaNs ({final_nan_count_numeric}) remain in numeric features AFTER imputation!")
        elif initial_nan_count > 0: print("No NaNs remaining in numeric features after imputation.")
        else: print("No NaNs found in numeric features before or after imputation.")

        # Convert categorical features (dummies) to NumPy
        try:
            X_categorical_np = categorical_df.cast(pl.UInt8).to_numpy() # Assume dummies are UInt8
        except Exception as e:
             # Check if categorical_df is empty
             if categorical_df.is_empty() and X_numeric_imputed_np.shape[0] > 0 :
                 # Handle case with only numeric features
                 X_categorical_np = np.empty((X_numeric_imputed_np.shape[0], 0), dtype=np.uint8)
                 print("No categorical features found/used.")
             elif categorical_df.is_empty() and X_numeric_imputed_np.shape[0] == 0:
                 # Handle case where everything is empty
                 X_categorical_np = np.empty((0, 0), dtype=np.uint8)

             else:
                raise ValueError(f"Failed to convert categorical features to NumPy: {e}")

        # Check for NaNs in categorical (shouldn't happen with dummies)
        if np.isnan(X_categorical_np).any():
            warnings.warn("NaNs detected in categorical features after conversion!")

        # Combine numeric and categorical NumPy arrays
        X_np = np.concatenate([X_numeric_imputed_np, X_categorical_np], axis=1)

        # Convert target Polars Series to NumPy array
        y_np = y_pl.cast(pl.Float64).to_numpy() # Ensure float target

        # Define final feature names based on combined array order
        self.final_feature_names = numeric_features + categorical_df.columns
        print(f"Final X NumPy shape: {X_np.shape}. y NumPy shape: {y_np.shape}. Using {len(self.final_feature_names)} features.")

        # Check final X_np for NaNs
        final_nan_count_all = np.isnan(X_np).sum()
        if final_nan_count_all > 0:
             warnings.warn(f"NaNs ({final_nan_count_all}) detected in the final combined feature matrix X!")


        return X_np, y_np, self.final_feature_names


# --- SurpriseClassificationModel / EarningsDriftModel ---
# These models primarily use XGBoost, which works with NumPy.
# Modify fit/predict/evaluate to accept Polars DF, extract necessary cols,
# convert to NumPy, and pass to XGBoost.
class SurpriseClassificationModel:
    """Model for classifying earnings surprises. Uses NumPy internally."""
    def __init__(self, xgb_cls_params=None, xgb_reg_params=None):
        # ... (Parameter setup same as before) ...
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

        # Fit classifiers on X_np
        pos_surprise_target = (surprise_np > 0).astype(int)
        neg_surprise_target = (surprise_np < 0).astype(int)
        print("  Fitting positive/negative surprise classifiers...")
        self.surprise_pos_model.fit(X_np, pos_surprise_target)
        self.surprise_neg_model.fit(X_np, neg_surprise_target)

        # Prepare data for return regressor (add actual surprise features)
        # Create a temporary DF might be easiest, then convert back
        X_df_temp = pl.DataFrame(X_np, schema=self.feature_names_in_)
        X_with_surprise_pl = X_df_temp.with_columns([
            pl.lit(surprise_np).alias('surprise_value_actual'),
            pl.lit(pos_surprise_target).alias('pos_surprise_actual'),
            pl.lit(neg_surprise_target).alias('neg_surprise_actual'),
            pl.lit(np.abs(surprise_np)).alias('surprise_magnitude_actual')
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
        if X_np.shape[1] != len(self.feature_names_in_):
            raise ValueError(f"Input X_np has {X_np.shape[1]} features, but model expected {len(self.feature_names_in_)}.")

        # Predict surprise probs/classes
        pos_prob = self.surprise_pos_model.predict_proba(X_np)[:, 1]
        neg_prob = self.surprise_neg_model.predict_proba(X_np)[:, 1]
        pos_pred = (pos_prob > 0.5).astype(int)
        neg_pred = (neg_prob > 0.5).astype(int)

        # Prepare data for return prediction (use *predicted* surprise info)
        # Need to construct the feature set expected by the return model
        X_df_temp = pl.DataFrame(X_np, schema=self.feature_names_in_) # Base features
        X_for_ret_pl = X_df_temp.with_columns([
            pl.lit(pos_prob - neg_prob).alias('surprise_value_actual'), # Use expected surprise
            pl.lit(pos_pred).alias('pos_surprise_actual'),
            pl.lit(neg_pred).alias('neg_surprise_actual'),
            pl.lit(np.abs(pos_prob - neg_prob)).alias('surprise_magnitude_actual')
        ])

        # Ensure columns match training for return model
        missing_ret_cols = set(self.return_feature_names_in_) - set(X_for_ret_pl.columns)
        if missing_ret_cols: raise ValueError(f"Internal error: Missing columns required for return prediction: {missing_ret_cols}")
        # Select and order columns, then convert to NumPy
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

        preds = self.predict(X_np)
        pos_actual = (surprise_np > 0).astype(int)
        neg_actual = (surprise_np < 0).astype(int)

        print("\n--- Surprise Classification Report (Positive) ---")
        print(classification_report(pos_actual, preds['pos_surprise_pred'], zero_division=0))
        print("\n--- Surprise Classification Report (Negative) ---")
        print(classification_report(neg_actual, preds['neg_surprise_pred'], zero_division=0))

        pos_acc = accuracy_score(pos_actual, preds['pos_surprise_pred'])
        neg_acc = accuracy_score(neg_actual, preds['neg_surprise_pred'])

        # Evaluate return prediction
        valid_mask = np.isfinite(y_np) & np.isfinite(preds['return_pred'])
        y_valid = y_np[valid_mask]
        y_pred_valid = preds['return_pred'][valid_mask]

        if len(y_valid) == 0:
             ret_mse, ret_r2, ret_dir_acc = np.nan, np.nan, np.nan
        else:
             ret_mse = mean_squared_error(y_valid, y_pred_valid)
             ret_r2 = r2_score(y_valid, y_pred_valid)
             # Directional accuracy
             ret_dir_acc = np.mean(np.sign(y_pred_valid) == np.sign(y_valid)) # Handle sign(0) if needed

        print("\n--- Return Prediction Evaluation ---")
        print(f"  MSE={ret_mse:.6f}, RMSE={np.sqrt(ret_mse):.6f}, R2={ret_r2:.4f}, DirAcc={ret_dir_acc:.4f}")

        return {'pos_surprise_accuracy': pos_acc, 'neg_surprise_accuracy': neg_acc,
                'return_mse': ret_mse, 'return_rmse': np.sqrt(ret_mse),
                'return_r2': ret_r2, 'return_direction_accuracy': ret_dir_acc}


class EarningsDriftModel:
    """Model post-earnings announcement drift (PEAD). Uses NumPy internally."""
    def __init__(self, time_horizons: List[int] = [1, 3, 5, 10, 20], model_params: Optional[Dict] = None):
        self.time_horizons = sorted(time_horizons)
        if model_params is None:
            self.model_params = {
                'n_estimators': 100, 'max_depth': 4, 'learning_rate': 0.05,
                'objective': 'reg:squarederror', 'subsample': 0.8, 'colsample_bytree': 0.8,
                'random_state': 42, 'n_jobs': -1, 'eval_metric': 'rmse',
                'use_label_encoder': False # For newer XGBoost versions
            }
        else: self.model_params = {**model_params, 'use_label_encoder': False}
        self.models = {h: xgb.XGBRegressor(**self.model_params) for h in time_horizons}
        self.feature_names_in_: Optional[List[str]] = None
        self.imputers: Dict[int, SimpleImputer] = {} # Store one imputer per horizon if needed


    def _prepare_data_for_horizon(self, data: pl.DataFrame, horizon: int, feature_cols: List[str], return_col: str = 'ret') -> Tuple[Optional[pl.DataFrame], Optional[pl.Series]]:
        """Prepare features (announcement day) and target (cumulative return) using Polars."""
        required = ['event_id', 'date', 'days_to_announcement', return_col] + feature_cols
        if not all(c in data.columns for c in required):
            missing = [c for c in required if c not in data.columns]
            # raise ValueError(f"PEAD Prep: Missing required columns: {missing}")
            print(f"Warning (PEAD Prep): Missing columns for horizon {horizon}: {missing}. Skipping.")
            return None, None

        data = data.sort(['event_id', 'date'])

        # Calculate cumulative returns T+1 to T+horizon using rolling product
        # Shift returns by 1 day forward to align: ret(t+1) used for day t
        # Fill null returns with 0 for product calculation
        data = data.with_columns(
            (pl.col(return_col).shift(-1).over("event_id").fill_null(0) + 1)
             .rolling_prod(window_size=horizon, min_periods=1) # Product of (1+ret)
             .over("event_id")
             .alias(f"cum_ret_factor_h{horizon}")
        )

        # Get features from announcement day (T=0)
        # Get target cumulative return calculated above, also at T=0
        announce_day_data = data.filter(pl.col('days_to_announcement') == 0)
        if announce_day_data.is_empty():
             print(f"Warning (PEAD Prep): No announcement day data found for horizon {horizon}.")
             return None, None

        # Select features and calculate target return
        final_data = announce_day_data.with_columns(
             (pl.col(f"cum_ret_factor_h{horizon}") - 1).alias(f"target_cum_ret_h{horizon}")
        ).drop_nulls(subset=[f"target_cum_ret_h{horizon}"]) # Drop if target couldn't be calculated

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
                X_np = X_pl.cast(pl.Float64).to_numpy()
                y_np = y_pl.cast(pl.Float64).to_numpy()

                # Impute NaNs if necessary
                if np.isnan(X_np).any():
                    print(f"    Imputing NaNs for horizon {horizon}...")
                    imputer = SimpleImputer(strategy='median')
                    X_np = imputer.fit_transform(X_np)
                    self.imputers[horizon] = imputer # Store imputer

                self.models[horizon].fit(X_np, y_np)
                print(f"    PEAD model {horizon}d fitted ({len(X_np)} samples).")
            except Exception as e:
                print(f"    Error training PEAD {horizon}d: {e}")
                traceback.print_exc()
        print("EarningsDriftModel fitting complete.")
        return self

    def predict(self, data: pl.DataFrame) -> pl.DataFrame:
        """Generate PEAD predictions using features from announcement day. Returns Polars DF."""
        print("Generating PEAD predictions...")
        if self.feature_names_in_ is None: raise RuntimeError("PEAD model not fitted.")

        # Filter for announcement days and select features
        announce_days = data.filter(pl.col('days_to_announcement') == 0)
        if announce_days.is_empty():
            print("Warning: No announcement day data for PEAD prediction.")
            # Return empty DF with expected prediction columns
            pred_cols = [f'pred_drift_h{h}' for h in self.time_horizons]
            return pl.DataFrame({col: [] for col in announce_days.columns + pred_cols})


        missing = [f for f in self.feature_names_in_ if f not in announce_days.columns]
        if missing: raise ValueError(f"PEAD prediction features missing: {missing}")

        X_pl = announce_days.select(self.feature_names_in_)

        # Convert to NumPy
        try:
            X_np = X_pl.cast(pl.Float64).to_numpy()
        except Exception as e:
             raise ValueError(f"Failed to convert PEAD prediction features to NumPy: {e}")


        # Impute using stored imputers if necessary
        if np.isnan(X_np).any():
             print("Imputing NaNs in PEAD prediction features...")
             # Try to use horizon-specific imputers if available, else use a generic one
             # This is complex because imputation depends on the horizon model being predicted
             # Simplification: Use a single imputer fitted on the first horizon's data or median
             # Or impute per prediction call inside the loop
             # Let's impute inside the loop using the specific imputer if present
             pass # Imputation handled below

        pred_expressions = []
        base_cols = announce_days.columns # Columns to keep from original data

        for horizon in self.time_horizons:
             pred_col = f'pred_drift_h{horizon}'
             if horizon in self.models:
                 try:
                     X_np_h = X_np.copy() # Work on a copy for imputation
                     if np.isnan(X_np_h).any():
                          if horizon in self.imputers:
                              # print(f"  Imputing with stored imputer for horizon {horizon}")
                              X_np_h = self.imputers[horizon].transform(X_np_h)
                          else:
                              # Fallback: impute with median for this specific call
                              print(f"  Warning: No stored imputer for horizon {horizon}. Imputing with temporary median.")
                              temp_imputer = SimpleImputer(strategy='median')
                              X_np_h = temp_imputer.fit_transform(X_np_h)

                     # Predict using the horizon-specific model
                     predictions = self.models[horizon].predict(X_np_h)
                     pred_expressions.append(pl.lit(predictions).alias(pred_col)) # Add predictions as a new column literal

                 except Exception as e:
                     print(f"Error predicting PEAD {horizon}d: {e}")
                     pred_expressions.append(pl.lit(None).cast(pl.Float64).alias(pred_col)) # Add null column on error
             else:
                 pred_expressions.append(pl.lit(None).cast(pl.Float64).alias(pred_col)) # Add null column if model missing

        # Add prediction columns to the announcement days data
        predictions_df = announce_days.with_columns(pred_expressions)

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

                # Convert to NumPy
                X_test_np = X_test_pl.cast(pl.Float64).to_numpy()
                y_test_np = y_test_pl.cast(pl.Float64).to_numpy()

                # Impute test data using stored imputer
                if np.isnan(X_test_np).any():
                    if horizon in self.imputers:
                        X_test_np = self.imputers[horizon].transform(X_test_np)
                    else:
                        print(f"    Warning: No stored imputer for horizon {horizon} evaluation. Imputing with temporary median.")
                        temp_imputer = SimpleImputer(strategy='median')
                        X_test_np = temp_imputer.fit_transform(X_test_np) # Fit AND transform on test data (suboptimal but fallback)


                if horizon not in self.models:
                    print("    Model not available.")
                    results[horizon] = {'MSE': np.nan, 'RMSE': np.nan, 'R2': np.nan, 'Direction Accuracy': np.nan, 'N': 0, 'Error': 'Model not trained'}
                    continue

                # Predict
                y_pred_np = self.models[horizon].predict(X_test_np)

                # Calculate metrics, handling NaNs
                valid_mask = np.isfinite(y_test_np) & np.isfinite(y_pred_np)
                y_test_v, y_pred_v = y_test_np[valid_mask], y_pred_np[valid_mask]

                if len(y_test_v) > 0:
                    mse = mean_squared_error(y_test_v, y_pred_v)
                    r2 = r2_score(y_test_v, y_pred_v)
                    dir_acc = np.mean(np.sign(y_pred_v) == np.sign(y_test_v)) # Handle sign(0)
                else:
                    mse, r2, dir_acc = np.nan, np.nan, np.nan

                results[horizon] = {'MSE': mse, 'RMSE': np.sqrt(mse), 'R2': r2, 'Direction Accuracy': dir_acc, 'N': len(y_test_v)}
                print(f"    PEAD {horizon}d: MSE={mse:.6f}, R2={r2:.4f}, DirAcc={dir_acc:.4f}, N={len(y_test_v)}")

            except Exception as e:
                print(f"    Error evaluating PEAD {horizon}d: {e}")
                results[horizon] = {'Error': str(e)}
        return results


class Analysis:
    def __init__(self, data_loader: DataLoader, feature_engineer: FeatureEngineer):
        """Analysis class for Earnings data using Polars."""
        self.data_loader = data_loader
        self.feature_engineer = feature_engineer
        self.data: Optional[pl.DataFrame] = None
        # Store NumPy arrays for ML models
        self.X_train_np: Optional[np.ndarray] = None
        self.X_test_np: Optional[np.ndarray] = None
        self.y_train_np: Optional[np.ndarray] = None
        self.y_test_np: Optional[np.ndarray] = None
        self.train_indices = None # Not used with Polars filtering approach
        self.test_indices = None
        self.train_data: Optional[pl.DataFrame] = None # Store processed train split DF
        self.test_data: Optional[pl.DataFrame] = None # Store processed test split DF
        self.models: Dict[str, Any] = {} # Standard models (Ridge, XGBDecile)
        self.surprise_model: Optional[SurpriseClassificationModel] = None
        self.pead_model: Optional[EarningsDriftModel] = None
        self.results: Dict[str, Dict] = {} # Standard model results
        self.surprise_results: Dict = {}
        self.pead_results: Dict = {}

    def load_and_prepare_data(self, run_feature_engineering: bool = True):
        """Load and optionally prepare data for earnings analysis using Polars."""
        print("--- Loading Earnings Data (Polars) ---")
        self.data = self.data_loader.load_data() # Uses chunking internally now
        if self.data is None or self.data.is_empty():
            # Raise error here if loading genuinely failed (returned None or empty)
            raise RuntimeError("Data loading failed or resulted in empty dataset.")

        if run_feature_engineering:
             # Check if data is still valid before proceeding
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
        # Ensure features/target exist if ML analysis is intended
        if 'future_ret' not in self.data.columns:
             print("ML Prep: Target variable 'future_ret' not found. Creating...")
             self.data = self.feature_engineer.create_target(self.data)
        if not self.feature_engineer.feature_names: # Check if features were calculated
             print("ML Prep: Features not calculated. Calculating...")
             self.data = self.feature_engineer.calculate_features(self.data, fit_categorical=False)

        print(f"\n--- Splitting Earnings Data (Train/Test based on {time_split_column}) ---")
        # Find split date based on unique events
        events = self.data.select(['event_id', time_split_column]).unique().sort(time_split_column)
        split_index = int(events.height * (1 - test_size))
        if split_index == 0 or split_index == events.height: raise ValueError("test_size results in empty train/test set.")
        split_date = events.item(split_index, time_split_column) # Get the date value at the split index
        print(f"Splitting {events.height} unique events. Train before {split_date}.")

        # Filter main dataframe based on the split date
        train_mask = pl.col(time_split_column) < split_date
        test_mask = pl.col(time_split_column) >= split_date
        train_data_raw = self.data.filter(train_mask)
        test_data_raw = self.data.filter(test_mask)
        print(f"Train rows (raw): {train_data_raw.height}, Test rows (raw): {test_data_raw.height}.")
        if train_data_raw.is_empty() or test_data_raw.is_empty():
            raise ValueError("Train or test split resulted in empty DataFrame.")

        print("\nFitting FeatureEngineer components (Categorical/Imputer) on Training Data...")
        # Calculate features on TRAIN split, fitting categoricals
        self.train_data = self.feature_engineer.calculate_features(train_data_raw, fit_categorical=True)
        # Extract NumPy arrays for training, fitting the imputer
        self.X_train_np, self.y_train_np, _ = self.feature_engineer.get_features_target(self.train_data, fit_imputer=True)

        print("\nApplying FeatureEngineer components to Test Data...")
        # Calculate features on TEST split, NOT fitting categoricals
        self.test_data = self.feature_engineer.calculate_features(test_data_raw, fit_categorical=False)
        # Ensure test data has the same columns as train data expected by features/imputer
        # This is implicitly handled by get_features_target using self.final_feature_names
        # Extract NumPy arrays for testing, using the FITTED imputer
        self.X_test_np, self.y_test_np, self.final_feature_names = self.feature_engineer.get_features_target(self.test_data, fit_imputer=False)

        print(f"\nTrain shapes (NumPy): X={self.X_train_np.shape}, y={self.y_train_np.shape}")
        print(f"Test shapes (NumPy): X={self.X_test_np.shape}, y={self.y_test_np.shape}")
        if self.X_train_np.shape[0] == 0 or self.X_test_np.shape[0] == 0:
             raise ValueError("Train or test NumPy array empty after feature extraction.")

        print("\n--- Training Standard Models (Earnings) ---")
        # 1. TimeSeriesRidge (Expects Polars DF input, uses NumPy internally)
        try:
             print("Training TimeSeriesRidge...")
             # Pass Polars DataFrame corresponding to X_train_np
             # Need to reconstruct the train DF slice used in get_features_target
             # Simpler: Modify TimeSeriesRidge to accept NumPy and feature names directly
             # Let's keep TimeSeriesRidge accepting Polars for consistency with its design
             # We need the Polars DF *before* imputation/NumPy conversion but *after* categorical fitting
             # This is self.train_data, but we need to align it with the rows kept in get_features_target

             # Alternative: Pass NumPy arrays directly to models if they support it.
             # TimeSeriesRidge does NOT currently support NumPy input directly.

             # Let's pass the relevant slice of self.train_data
             # Need indices from get_features_target... but Polars doesn't use index easily.
             # Re-filter self.train_data based on non-null target:
             train_data_for_ridge = self.train_data.filter(pl.col('future_ret').is_not_null())
             # Select only final features
             X_train_ridge_pl = train_data_for_ridge.select(self.feature_engineer.final_feature_names)
             y_train_ridge_pl = train_data_for_ridge.get_column('future_ret')

             if X_train_ridge_pl.height == self.X_train_np.shape[0]: # Check alignment
                 ts_ridge = TimeSeriesRidge(alpha=1.0, lambda2=0.1, feature_order=self.feature_engineer.final_feature_names)
                 # Fit with the Polars DataFrame (imputation happens inside Ridge if needed, but shouldn't be required now)
                 # Note: TimeSeriesRidge currently doesn't impute. Assumes clean input.
                 # Imputation MUST happen before Ridge. Let's pass NumPy.
                 # ***MODIFYING TimeSeriesRidge TO ACCEPT NUMPY***
                 # Ok, let's NOT modify TimeSeriesRidge, we'll fit it *before* imputation maybe?
                 # No, imputation is essential. Let's modify Ridge to take NumPy.

                 # ***Modification Plan B:***
                 # Fit TimeSeriesRidge using the IMPUTED NumPy array and feature names
                 ts_ridge = TimeSeriesRidge(alpha=1.0, lambda2=0.1) # No feature order needed if passing NumPy
                 # We need a way for Ridge to know the feature names if feature_order is None
                 # Let's stick to passing Polars to TimeSeriesRidge for now, but use the IMPUTED data
                 # Convert imputed NumPy back to Polars DF for Ridge fit
                 X_train_ridge_pl_imputed = pl.DataFrame(self.X_train_np, schema=self.feature_engineer.final_feature_names)
                 y_train_ridge_pl = pl.Series("", self.y_train_np) # Convert y back too

                 ts_ridge.fit(X_train_ridge_pl_imputed, y_train_ridge_pl)
                 self.models['TimeSeriesRidge'] = ts_ridge
                 print("TimeSeriesRidge complete.")

             else:
                 print("Warning: Row mismatch between imputed NumPy array and Polars DF for Ridge. Skipping Ridge.")

        except Exception as e: print(f"Error TimeSeriesRidge: {e}"); traceback.print_exc()

        # 2. XGBoostDecile (Expects Polars DF input, uses NumPy internally)
        try:
             print("\nTraining XGBoostDecile...")
             xgb_params = {'n_estimators': 150, 'max_depth': 5, 'learning_rate': 0.05, 'objective': 'reg:squarederror', 'subsample': 0.8, 'colsample_bytree': 0.8, 'random_state': 42, 'n_jobs': -1, 'early_stopping_rounds': 10}

             # We need the Polars DF corresponding to X_train_np for fitting
             # Use the same imputed Polars DF created for Ridge
             if 'ts_ridge' in self.models: # Check if the DF was created successfully
                X_train_xgb_pl = X_train_ridge_pl_imputed
                y_train_xgb_pl = y_train_ridge_pl
             else: # Recreate if Ridge failed
                 X_train_xgb_pl = pl.DataFrame(self.X_train_np, schema=self.feature_engineer.final_feature_names)
                 y_train_xgb_pl = pl.Series("", self.y_train_np)

             if 'momentum_5' not in X_train_xgb_pl.columns: print("Warning: 'momentum_5' not found for XGBoostDecile.")

             xgb_decile = XGBoostDecileModel(weight=0.7, momentum_feature='momentum_5', n_deciles=10, alpha=1.0, lambda_smooth=0.1, xgb_params=xgb_params, ts_ridge_feature_order=self.feature_engineer.final_feature_names)
             xgb_decile.fit(X_train_xgb_pl, y_train_xgb_pl) # Fit with Polars DF
             self.models['XGBoostDecile'] = xgb_decile
             print("XGBoostDecile complete.")
        except Exception as e: print(f"Error XGBoostDecile: {e}"); traceback.print_exc()


        # --- Train Surprise Classification Model ---
        surprise_col_name = 'surprise_val' # Standardized name
        # Check if surprise column exists in the *processed* train data
        if surprise_col_name in self.train_data.columns and self.train_data.filter(pl.col(surprise_col_name).is_not_null()).height > 0:
             print("\n--- Training Surprise Classification Model ---")
             try:
                 # Align surprise data with X_train_np (order should be preserved)
                 # Get surprise values from the train_data slice corresponding to X_train_np/y_train_np
                 train_data_for_surprise = self.train_data.filter(pl.col('future_ret').is_not_null()) # Align rows
                 surprise_train_np = train_data_for_surprise.get_column(surprise_col_name).cast(pl.Float64).fill_null(0).to_numpy() # To NumPy

                 if len(surprise_train_np) == self.X_train_np.shape[0]:
                     self.surprise_model = SurpriseClassificationModel()
                     self.surprise_model.fit(self.X_train_np, self.y_train_np, surprise_train_np, self.final_feature_names)
                     print("SurpriseClassificationModel training complete.")
                 else:
                      print("Warning: Mismatch rows between features and surprise data. Skipping surprise model.")

             except Exception as e: print(f"Error SurpriseClassificationModel: {e}"); self.surprise_model = None; traceback.print_exc()
        else: print(f"\n'{surprise_col_name}' column not found or all NaNs in processed train set. Skipping surprise model.")


        # --- Train Earnings Drift (PEAD) Model ---
        print("\n--- Training Earnings Drift (PEAD) Model ---")
        try:
             self.pead_model = EarningsDriftModel(time_horizons=[1, 3, 5, 10, 20])
             # Fit using the *processed* training data Polars DataFrame
             # PEAD model handles internal NumPy conversion and feature prep
             self.pead_model.fit(self.train_data, feature_cols=self.final_feature_names)
             print("EarningsDriftModel training complete.")
        except Exception as e: print(f"Error EarningsDriftModel: {e}"); self.pead_model = None; traceback.print_exc()

        print("\n--- All Earnings Model Training Complete ---")
        return self.models

    def evaluate_models(self) -> Dict[str, Dict]:
        """Evaluate all trained models on the test set using NumPy arrays."""
        print("\n--- Evaluating Earnings Models ---")
        if self.X_test_np is None or self.y_test_np is None or self.X_test_np.shape[0]==0:
             print("Test data (NumPy) unavailable/empty."); return {}

        # Evaluate Standard Models
        print("\n--- Standard Model Evaluation ---")
        self.results = {}
        for name, model in self.models.items():
            print(f"Evaluating {name}...")
            try:
                # Predict using appropriate input type (Polars DF for Ridge/XGBDecile)
                # Reconstruct test Polars DF from NumPy array
                X_test_pl = pl.DataFrame(self.X_test_np, schema=self.final_feature_names)

                # Predict returns NumPy array
                y_pred_np = model.predict(X_test_pl) # Models now return NumPy

                # Use NumPy for metrics
                valid_mask = np.isfinite(self.y_test_np) & np.isfinite(y_pred_np)
                y_test_v, y_pred_v = self.y_test_np[valid_mask], y_pred_np[valid_mask]
                if len(y_test_v) > 0:
                    mse = mean_squared_error(y_test_v, y_pred_v)
                    r2 = r2_score(y_test_v, y_pred_v)
                else: mse, r2 = np.nan, np.nan
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
             # Need surprise values for the test set, aligned with X_test_np
             if self.test_data is not None and surprise_col_name in self.test_data.columns and self.test_data.filter(pl.col(surprise_col_name).is_not_null()).height > 0:
                 # Align surprise data with X_test_np
                 test_data_for_surprise = self.test_data.filter(pl.col('future_ret').is_not_null()) # Align rows
                 surprise_test_np = test_data_for_surprise.get_column(surprise_col_name).cast(pl.Float64).fill_null(0).to_numpy()

                 if len(surprise_test_np) == self.X_test_np.shape[0]:
                     try:
                         self.surprise_results = self.surprise_model.evaluate(self.X_test_np, self.y_test_np, surprise_test_np)
                     except Exception as e:
                         print(f"  Error evaluating Surprise Model: {e}"); self.surprise_results = {'Error': str(e)}; traceback.print_exc()
                 else:
                      print("  Warning: Mismatch rows between test features and surprise data. Cannot evaluate surprise model.")
             else: print(f"  '{surprise_col_name}' column not found or all NaNs in processed test data. Cannot evaluate.")
        else: print("\nSurprise Model not trained. Skipping evaluation.")

        # Evaluate PEAD Model
        if self.pead_model:
             print("\n--- PEAD Model Evaluation ---")
             try:
                 # Evaluate using the *processed* test data Polars DataFrame
                 if self.test_data is not None:
                    self.pead_results = self.pead_model.evaluate(self.test_data)
                 else:
                     print("  Processed test data unavailable. Cannot evaluate PEAD model.")
             except Exception as e: print(f"  Error evaluating PEAD Model: {e}"); self.pead_results = {'Error': str(e)}; traceback.print_exc()
        else: print("\nPEAD Model not trained. Skipping evaluation.")

        print("\n--- Earnings Evaluation Complete ---")
        return {"standard": self.results, "surprise": self.surprise_results, "pead": self.pead_results}


    def plot_feature_importance(self, results_dir: str, file_prefix: str = "earnings", model_name: str = 'TimeSeriesRidge', top_n: int = 20):
        """Plot feature importance and save the plot. Uses model coefficients/importances."""
        print(f"\n--- Plotting Earnings Feature Importance for {model_name} ---")
        if model_name not in self.models: print(f"Error: Model '{model_name}' not found."); return None
        model = self.models[model_name]
        # Use final_feature_names stored after get_features_target
        feature_names = self.feature_engineer.final_feature_names
        if not feature_names: print("Error: Final feature names not found (run training first)."); return None

        importances = None
        if isinstance(model, TimeSeriesRidge):
             # Coef_ should correspond to final_feature_names
             if hasattr(model, 'coef_') and model.coef_ is not None:
                 if len(model.coef_) == len(feature_names):
                    importances = np.abs(model.coef_)
                 else: print(f"Warn: Ridge coef len ({len(model.coef_)}) != feature len ({len(feature_names)}). Cannot plot.")
        elif isinstance(model, XGBoostDecileModel):
             # Use importances from the core XGBoost model
             if hasattr(model, 'xgb_model') and hasattr(model.xgb_model, 'feature_importances_'):
                 xgb_importances = model.xgb_model.feature_importances_
                 # Assume xgb_model was trained on features in order of feature_names
                 if len(xgb_importances) == len(feature_names):
                     importances = xgb_importances
                 else:
                     # Fallback: Try to get names from booster if available
                     try:
                        booster = model.xgb_model.get_booster()
                        xgb_feat_names = booster.feature_names
                        if xgb_feat_names and len(xgb_feat_names) == len(xgb_importances):
                             imp_dict = dict(zip(xgb_feat_names, xgb_importances))
                             importances = np.array([imp_dict.get(name, 0) for name in feature_names]) # Align with final names
                        else: raise ValueError("Mismatch or missing booster names")
                     except Exception:
                         print(f"Warn: XGB importance len ({len(xgb_importances)}) != feature len ({len(feature_names)}). Cannot plot reliably.")

        if importances is None: print(f"Could not get importance scores for {model_name}."); return None

        # Create Pandas DF for plotting with Seaborn
        feat_imp_df = pl.DataFrame({'Feature': feature_names, 'Importance': importances}) \
                        .sort('Importance', descending=True) \
                        .head(top_n) \
                        .to_pandas() # Convert top N to Pandas

        fig, ax = plt.subplots(figsize=(10, max(6, top_n // 2)))
        sns.barplot(x='Importance', y='Feature', data=feat_imp_df, palette='viridis', ax=ax)
        ax.set_title(f'Top {top_n} Features by Importance ({model_name} - Earnings)')
        ax.set_xlabel('Importance Score'); ax.set_ylabel('Feature')
        plt.tight_layout()

        plot_filename = os.path.join(results_dir, f"{file_prefix}_feat_importance_{model_name}.png")
        try: plt.savefig(plot_filename); print(f"Saved feature importance plot to: {plot_filename}")
        except Exception as e: print(f"Error saving feature importance plot: {e}")
        plt.close(fig)
        # Return Polars DF of importance
        return pl.DataFrame({'Feature': feature_names, 'Importance': importances}).sort('Importance', descending=True)


    def analyze_earnings_surprise(self, results_dir: str, file_prefix: str = "earnings"):
        """Analyze the impact of earnings surprises on actual returns using Polars and plot."""
        print("\n--- Analyzing Earnings Surprise Impact (Polars) ---")
        surprise_col = 'surprise_val' # Standardized name
        if self.data is None or surprise_col not in self.data.columns or self.data[surprise_col].is_null().all():
            print(f"No valid earnings surprise data ('{surprise_col}') available."); return None

        # Use a window around the announcement
        analysis_data = self.data.filter(
            (pl.col('days_to_announcement') >= -2) & (pl.col('days_to_announcement') <= 5)
        )
        if analysis_data.is_empty(): print("No data in analysis window."); return None

        # Define bins and labels
        bins = [-np.inf, -0.01, 0.01, np.inf]
        labels = ['Negative Surprise', 'Near Zero Surprise', 'Positive Surprise']

        # Assign surprise groups using when/then
        analysis_data = analysis_data.with_columns(
             pl.when(pl.col(surprise_col) < bins[1]).then(pl.lit(labels[0]))
             .when(pl.col(surprise_col) < bins[2]).then(pl.lit(labels[1]))
             .when(pl.col(surprise_col) >= bins[2]).then(pl.lit(labels[2]))
             .otherwise(None) # Handle potential edge cases/NaNs
             .alias('Surprise Group')
        ).drop_nulls(subset=['Surprise Group']) # Remove rows where group couldn't be assigned

        if analysis_data.is_empty(): print("No data after assigning surprise groups."); return None

        # Calculate average daily returns per group and day
        avg_returns = analysis_data.group_by(['Surprise Group', 'days_to_announcement']).agg(
            pl.mean('ret').alias('avg_ret')
        ).sort(['Surprise Group', 'days_to_announcement'])

        # Pivot for plotting daily returns
        try:
             avg_returns_pivot = avg_returns.pivot(
                 index='days_to_announcement', columns='Surprise Group', values='avg_ret'
             ).sort('days_to_announcement')
        except Exception as e:
             print(f"Could not pivot average returns data: {e}. Skipping daily plot.")
             avg_returns_pivot = None


        # Calculate cumulative returns
        # Ensure data is sorted correctly for cumulative product
        avg_returns_sorted = avg_returns.sort(['Surprise Group', 'days_to_announcement'])
        avg_cum_returns = avg_returns_sorted.with_columns(
            # Calculate cumulative product of (1 + avg_ret) within each group
            (pl.col('avg_ret').fill_null(0) + 1).cum_prod().over('Surprise Group').alias('cum_ret_factor')
        ).with_columns(
            (pl.col('cum_ret_factor') - 1).alias('avg_cum_ret')
        )

        # Pivot for plotting cumulative returns
        try:
             avg_cum_returns_pivot = avg_cum_returns.pivot(
                 index='days_to_announcement', columns='Surprise Group', values='avg_cum_ret'
             ).sort('days_to_announcement')
             # Convert to Pandas for plotting
             avg_cum_returns_pd = avg_cum_returns_pivot.to_pandas().set_index('days_to_announcement')
        except Exception as e:
             print(f"Could not pivot cumulative returns data: {e}. Skipping cumulative plot.")
             avg_cum_returns_pivot = None
             avg_cum_returns_pd = None

        # Plotting
        fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

        if avg_returns_pivot is not None:
             avg_returns_pd = avg_returns_pivot.to_pandas().set_index('days_to_announcement')
             avg_returns_pd.plot(kind='bar', ax=axes[0], width=0.8)
             axes[0].set_title('Average Daily Returns by Surprise')
             axes[0].set_ylabel('Avg Daily Return')
             axes[0].axhline(0, c='grey', ls='--', lw=0.8)
             axes[0].legend(title='Group')
             axes[0].grid(axis='y', ls=':', alpha=0.6)
        else:
            axes[0].set_title('Average Daily Returns by Surprise (Plotting Error)')

        if avg_cum_returns_pd is not None:
             avg_cum_returns_pd.plot(kind='line', marker='o', ax=axes[1])
             axes[1].set_title('Average Cumulative Returns by Surprise')
             axes[1].set_ylabel('Avg Cum Return')
             axes[1].set_xlabel('Days Rel. to Announce')
             axes[1].axhline(0, c='grey', ls='--', lw=0.8)
             axes[1].axvline(0, c='red', ls=':', lw=1, label='Announce Day')
             axes[1].legend(title='Group')
             axes[1].grid(True, ls=':', alpha=0.6)
        else:
            axes[1].set_title('Average Cumulative Returns by Surprise (Plotting Error)')

        plt.tight_layout()

        plot_filename = os.path.join(results_dir, f"{file_prefix}_surprise_impact_returns.png")
        csv_filename = os.path.join(results_dir, f"{file_prefix}_surprise_impact_cum_returns_data.csv")
        try: plt.savefig(plot_filename); print(f"Saved surprise impact plot to: {plot_filename}")
        except Exception as e: print(f"Error saving surprise plot: {e}")
        if avg_cum_returns_pivot is not None:
            try: avg_cum_returns_pivot.write_csv(csv_filename); print(f"Saved surprise cum returns data to: {csv_filename}")
            except Exception as e: print(f"Error saving surprise data: {e}")
        plt.close(fig)

        return avg_cum_returns # Return Polars DF


    def plot_predictions_for_event(self, results_dir: str, event_id: str, file_prefix: str = "earnings", model_name: str = 'TimeSeriesRidge'):
        """Plot actual daily returns and model's predicted future returns using Polars data and save plot."""
        print(f"\n--- Plotting Earnings Predictions for Event: {event_id} ({model_name}) ---")
        if model_name not in self.models: print(f"Error: Model '{model_name}' not found."); return None
        if self.data is None: print("Error: Data not loaded."); return None
        model = self.models[model_name]

        # Find event data in the original dataset
        event_data_full = self.data.filter(pl.col('event_id') == event_id).sort('date')
        if event_data_full.is_empty(): print(f"Error: No data for event_id '{event_id}'."); return None

        ticker = event_data_full['ticker'][0]; announcement_date = event_data_full['Announcement Date'][0]

        # We need features for this event to predict
        # Process only the specific event data to get features
        # Important: Use the *fitted* imputer and categorical encodings
        if not self.feature_engineer._imputer_fitted:
             print("Error: FeatureEngineer imputer not fitted (run training first)."); return None

        # Calculate features for the single event, not fitting categoricals/imputer
        event_data_processed = self.feature_engineer.calculate_features(event_data_full, fit_categorical=False)
        # Extract features/target, using the fitted imputer
        X_event_np, y_event_actual_np, event_features = self.feature_engineer.get_features_target(event_data_processed, fit_imputer=False)

        if X_event_np.shape[0] == 0: print(f"Warn: No valid features/target rows for event {event_id}."); return None

        try:
            # Predict using the appropriate input format (Polars DF for Ridge/XGBDecile)
            X_event_pl = pl.DataFrame(X_event_np, schema=event_features)
            y_pred_event_np = model.predict(X_event_pl) # Returns NumPy array
        except Exception as e:
            print(f"Error predicting event {event_id}: {e}"); return None

        # Align predictions back to the event dates
        # Get the dates corresponding to X_event_np rows
        event_data_pred_source = event_data_processed.filter(pl.col('future_ret').is_not_null()) # Same filter as get_features_target
        if event_data_pred_source.height != len(y_pred_event_np):
             print("Warn: Mismatch between prediction count and source data rows. Plot may be inaccurate."); # return None?

        # Create DF with predictions and dates
        event_data_pred = event_data_pred_source.select(['date']).with_columns(
             pl.lit(y_pred_event_np).alias('predicted_future_ret')
        )

        # Convert to Pandas for plotting
        event_data_full_pd = event_data_full.select(['date', 'ret']).to_pandas()
        event_data_pred_pd = event_data_pred.to_pandas()
        announcement_date_pd = announcement_date # Already datetime? Check type
        if not isinstance(announcement_date_pd, (pd.Timestamp, np.datetime64, datetime.datetime)):
             try: announcement_date_pd = pd.to_datetime(announcement_date)
             except: print("Warn: Could not convert announcement date for plotting.")


        # Plotting
        fig, ax = plt.subplots(figsize=(14, 6))
        ax.plot(event_data_full_pd['date'], event_data_full_pd['ret'], '.-', label='Actual Daily Return', alpha=0.6, color='blue')
        ax.scatter(event_data_pred_pd['date'], event_data_pred_pd['predicted_future_ret'], color='red', label=f'Predicted {self.feature_engineer.prediction_window}-day Future Ret', s=15, zorder=5, alpha=0.8)
        if 'announcement_date_pd' in locals(): ax.axvline(x=announcement_date_pd, color='g', linestyle='--', label='Announcement Date')
        ax.set_title(f"{ticker} ({event_id}) - Daily Returns & Predicted Future Returns ({model_name} - Earnings)")
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
        """Find sample Earnings event identifiers from Polars data."""
        if self.data is None or 'event_id' not in self.data.columns: return []
        unique_events = self.data['event_id'].unique().head(n)
        return unique_events.to_list()

    def plot_pead_predictions(self, results_dir: str, file_prefix: str = "earnings", n_events: int = 3):
        """Plot PEAD predictions vs actual cumulative returns using Polars data."""
        print("\n--- Plotting PEAD Predictions vs Actual (Polars) ---")
        if self.pead_model is None: print("PEAD Model not trained."); return
        if self.test_data is None: print("Test data not available (run train_models first)."); return

        # Generate predictions on the test set
        pead_predictions_df = self.pead_model.predict(self.test_data) # Returns Polars DF
        if pead_predictions_df.is_empty(): print("No PEAD predictions generated for test data."); return

        # Find sample event IDs from the test set that have predictions
        available_event_ids = pead_predictions_df['event_id'].unique().to_list()
        if not available_event_ids: print("No event IDs found in PEAD predictions."); return

        sample_event_ids = available_event_ids[:min(n_events * 5, len(available_event_ids))] # Get samples

        plotted_count = 0
        for event_id in sample_event_ids:
            if plotted_count >= n_events: break

            # Get predictions for this event
            event_preds = pead_predictions_df.filter(pl.col('event_id') == event_id)
            if event_preds.is_empty(): continue

            # Get actual data for this event from the test set (need returns post-announcement)
            # Use self.test_data as it contains the necessary columns
            event_actual_data = self.test_data.filter(pl.col('event_id') == event_id).sort('date')
            if event_actual_data.is_empty(): continue

            ticker = event_actual_data['ticker'][0]
            announce_date = event_actual_data['Announcement Date'][0]

            # Calculate actual cumulative return post-announcement
            post_announce_actual = event_actual_data.filter(pl.col('date') >= announce_date)
            if post_announce_actual.is_empty(): print(f"No post-announcement data for {event_id}"); continue

            post_announce_actual = post_announce_actual.with_columns(
                 (pl.col('ret').fill_null(0) + 1).cum_prod().alias('cum_ret_factor') - 1
            ).rename({'cum_ret_factor': 'actual_cum_ret'})

            # Prepare data for plotting (predictions are for specific horizons)
            pred_cols = sorted([col for col in event_preds.columns if col.startswith('pred_drift_h')], key=lambda x: int(x[len('pred_drift_h'):]))
            horizons = [int(col[len('pred_drift_h'):]) for col in pred_cols]
            pred_values = event_preds.select(pred_cols).row(0) # Get predictions as a tuple

            # Convert to Pandas for plotting
            plot_dates_pd = [pd.Timestamp(announce_date) + pd.Timedelta(days=h) for h in horizons] # Requires pandas
            pred_values_pd = list(pred_values)
            post_announce_actual_pd = post_announce_actual.select(['date', 'actual_cum_ret']).to_pandas()
            announce_date_pd = pd.Timestamp(announce_date)

            # Plotting
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
        """Simulates a buy-and-hold strategy for each event using Polars and calculates returns."""
        if self.data is None or return_col not in self.data.columns or 'days_to_announcement' not in self.data.columns:
            print("Error: Data/required columns missing for strategy simulation.")
            return None

        df_sorted = self.data.sort(['event_id', 'date'])

        # Find entry date for each event
        entry_points = df_sorted.filter(pl.col('days_to_announcement') == entry_day) \
                                .select(['event_id', 'date']) \
                                .rename({'date': 'entry_date'})

        if entry_points.is_empty():
             print(f"No entry points found for entry_day = {entry_day}")
             return pl.DataFrame({'event_id': [], 'strategy_return': []})

        # Join entry date back to main data
        df_with_entry = df_sorted.join(entry_points, on='event_id', how='left')

        # Filter for holding period rows
        holding_data = df_with_entry.filter(
             (pl.col('date') > pl.col('entry_date')) & # Must be strictly after entry
             (pl.col('date') <= pl.col('entry_date') + pl.duration(days=holding_period)) # Within holding period end date
             # Note: Polars duration might not be exact number of trading days.
             # For exact trading days, use row number shift after filtering entry day.
        )

        # Calculate compound return over the holding period per event
        # Group by event, calculate product(1+ret)-1
        # Ensure enough rows exist for the holding period
        strategy_returns_agg = holding_data.group_by('event_id').agg(
            (pl.col(return_col).fill_null(0) + 1).product().alias("prod_ret_factor"),
            pl.count().alias("holding_days") # Count actual days in window
        ).filter(
            pl.col('holding_days') >= holding_period # Ensure enough data points
        ).with_columns(
             (pl.col('prod_ret_factor') - 1).alias('strategy_return')
        ).select(['event_id', 'strategy_return'])


        return strategy_returns_agg

    def analyze_event_sharpe_ratio(self, results_dir: str, file_prefix: str = "earnings", holding_period: int = 20, entry_day: int = 0, risk_free_rate: float = 0.0):
        """Calculates and analyzes the Sharpe Ratio of a simple event-based strategy using Polars."""
        print(f"\n--- Analyzing Event Strategy Sharpe Ratio (Entry: T{entry_day}, Hold: {holding_period}d) ---")
        strategy_returns_df = self.calculate_event_strategy_returns(holding_period=holding_period, entry_day=entry_day)

        if strategy_returns_df is None or strategy_returns_df.is_empty() or strategy_returns_df.height < 2:
            print(f"Error: Insufficient valid returns ({strategy_returns_df.height if strategy_returns_df is not None else 0}). Cannot calculate Sharpe.")
            return None

        print(f"Calculated returns for {strategy_returns_df.height} events.")

        # Calculate stats using Polars expressions
        stats = strategy_returns_df.select([
            pl.mean('strategy_return').alias('mean_ret'),
            pl.std('strategy_return').alias('std_ret')
        ]).row(0) # Get stats as a tuple (mean, std)

        mean_ret = stats[0]
        std_ret = stats[1]

        # Adjust risk-free rate for the period (approximate)
        # Assumes 252 trading days per year
        period_rf = (1 + risk_free_rate)**(holding_period/252) - 1 if risk_free_rate != 0 else 0
        sharpe = (mean_ret - period_rf) / (std_ret + 1e-9) # Add epsilon for stability

        print(f"  Mean Return: {mean_ret:.4%}, Std Dev: {std_ret:.4%}, Period Sharpe: {sharpe:.4f}")

        # Save returns data
        csv_filename = os.path.join(results_dir, f"{file_prefix}_strategy_returns_h{holding_period}_e{entry_day}.csv")
        try:
            strategy_returns_df.write_csv(csv_filename)
            print(f"Saved strategy returns to: {csv_filename}")
        except Exception as e: print(f"Error saving returns data: {e}")

        # Plot histogram (convert returns column to Pandas Series)
        fig, ax = plt.subplots(figsize=(10, 6))
        returns_pd = strategy_returns_df['strategy_return'].to_pandas() # Convert to Pandas for plotting
        sns.histplot(returns_pd, bins=30, kde=True, ax=ax)
        ax.axvline(mean_ret, color='red', linestyle='--', label=f'Mean ({mean_ret:.2%})')
        ax.set_title(f'Distribution of {holding_period}-Day Strategy Returns (Entry T{entry_day}) - Earnings')
        ax.set_xlabel(f'{holding_period}-Day Return')
        ax.set_ylabel('Frequency')
        ax.legend()
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

        df = self.data.sort(['event_id', 'date']) # Ensure sorted for rolling ops

        # Calculate rolling volatility (row-based window)
        df = df.with_columns(
            pl.col('ret').rolling_std(window_size=window, min_periods=min_periods)
              .over('event_id') # Apply rolling within each event group
              .alias('rolling_vol')
        )
        # Annualize volatility
        df = df.with_columns(
            (pl.col('rolling_vol') * np.sqrt(252) * 100).alias('annualized_vol') # In percent
        )

        # Align volatility by days relative to announcement
        aligned_vol = df.group_by('days_to_announcement').agg(
            pl.mean('annualized_vol').alias('avg_annualized_vol')
        ).filter(
            (pl.col('days_to_announcement') >= -pre_days) &
            (pl.col('days_to_announcement') <= post_days)
        ).sort('days_to_announcement').drop_nulls() # Drop days with no avg vol

        # --- Plotting & Saving Rolling Volatility ---
        if not aligned_vol.is_empty():
            aligned_vol_pd = aligned_vol.to_pandas().set_index('days_to_announcement') # Convert for plot/save
            fig1, ax1 = plt.subplots(figsize=(12, 6))
            aligned_vol_pd['avg_annualized_vol'].plot(kind='line', marker='.', linestyle='-', ax=ax1)
            ax1.axvline(0, color='red', linestyle='--', lw=1, label='Announcement Day')
            ax1.set_title(f'Average Rolling Volatility Around Earnings Announcement (Window={window} rows)')
            ax1.set_xlabel('Days Relative to Announcement')
            ax1.set_ylabel('Avg. Annualized Volatility (%)')
            ax1.legend(); ax1.grid(True, alpha=0.5)
            plt.tight_layout()

            plot_filename_vol = os.path.join(results_dir, f"{file_prefix}_volatility_rolling_{window}d.png")
            csv_filename_vol = os.path.join(results_dir, f"{file_prefix}_volatility_rolling_{window}d_data.csv")
            try: plt.savefig(plot_filename_vol); print(f"Saved rolling vol plot to: {plot_filename_vol}")
            except Exception as e: print(f"Error saving plot: {e}")
            try: aligned_vol_pd.to_csv(csv_filename_vol); print(f"Saved rolling vol data to: {csv_filename_vol}") # Use Pandas to_csv
            except Exception as e: print(f"Error saving data: {e}")
            plt.close(fig1)
        else:
            print("No data for rolling volatility plot.")


        # --- Compare Event vs Baseline Volatility ---
        # Calculate std dev for baseline and event windows per event
        vol_comp = df.group_by('event_id').agg([
            pl.std('ret').filter( # Baseline window std dev
                (pl.col('days_to_announcement') >= baseline_window[0]) &
                (pl.col('days_to_announcement') <= baseline_window[1])
            ).alias('vol_baseline'),
            pl.count('ret').filter( # Count for baseline
                 (pl.col('days_to_announcement') >= baseline_window[0]) &
                 (pl.col('days_to_announcement') <= baseline_window[1])
            ).alias('n_baseline'),
            pl.std('ret').filter( # Event window std dev
                (pl.col('days_to_announcement') >= event_window[0]) &
                (pl.col('days_to_announcement') <= event_window[1])
            ).alias('vol_event'),
             pl.count('ret').filter( # Count for event
                 (pl.col('days_to_announcement') >= event_window[0]) &
                 (pl.col('days_to_announcement') <= event_window[1])
            ).alias('n_event'),
        ]).filter( # Ensure enough data points and non-zero baseline vol
             (pl.col('n_baseline') >= min_periods) &
             (pl.col('n_event') >= min_periods) &
             (pl.col('vol_baseline').is_not_null()) &
             (pl.col('vol_baseline') > 1e-9) & # Avoid division by zero/tiny
             (pl.col('vol_event').is_not_null())
        )

        # Calculate ratio
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
            ax2.set_xlabel('Volatility Ratio')
            ax2.set_ylabel('Frequency')
            ax2.legend()
            plt.tight_layout()

            plot_filename_hist = os.path.join(results_dir, f"{file_prefix}_volatility_ratio_hist.png")
            try: plt.savefig(plot_filename_hist); print(f"Saved vol ratio hist plot: {plot_filename_hist}")
            except Exception as e: print(f"Error saving hist: {e}")
            plt.close(fig2)
        else:
            print("\nCould not calculate volatility ratios (insufficient valid data).")

        return aligned_vol # Return the aligned rolling vol dataframe
