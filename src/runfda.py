import pandas as pd
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import traceback

# --- Configuration ---
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

try:
    from fda_processor import DataLoader, FeatureEngineer, Analysis
    print("Successfully imported FDA processor classes.")
except ImportError as e:
    print(f"Error importing from fda_processor: {e}"); sys.exit(1)

# --- File Paths and Parameters ---
# <<< --- UPDATE THESE PATHS --- >>>
FDA_FILE = '/home/d87016661/fda_ticker_list_2000_to_2024.csv'
STOCK_FILES = [
    '/home/d87016661/crsp_dsf-2000-2001.parquet',   
    '/home/d87016661/crsp_dsf-2002-2003.parquet',
    '/home/d87016661/crsp_dsf-2004-2005.parquet',
    '/home/d87016661/crsp_dsf-2006-2007.parquet',
    '/home/d87016661/crsp_dsf-2008-2009.parquet',
    '/home/d87016661/crsp_dsf-2010-2011.parquet',
    '/home/d87016661/crsp_dsf-2016-2017.parquet',
    '/home/d87016661/crsp_dsf-2018-2019.parquet',
    '/home/d87016661/crsp_dsf-2020-2021.parquet',
    '/home/d87016661/crsp_dsf-2022-2023.parquet',
    '/home/d87016661/crsp_dsf-2024-2025.parquet'
]
# <<< -------------------------- >>>

# --- Analysis Parameters ---
RESULTS_DIR = "results_earnings_polars" # Directory to save results
FILE_PREFIX = "earnings"       # Prefix for saved files
WINDOW_DAYS = 30               # Days around event for data loading

# Event Strategy Parameters
STRATEGY_HOLDING_PERIOD = 20 # Days to hold after entry
STRATEGY_ENTRY_DAY = 0      # Day relative to event to enter (0=Announcement Day)

# Volatility Analysis Parameters
VOL_ROLLING_WINDOW = 5     # Days for rolling Volatility calculation (row-based)
VOL_BASELINE_WINDOW = (-60, 60) # Days relative to event for baseline vol calc
VOL_EVENT_WINDOW = (-2, 2)       # Days relative to event for event vol calc
VOL_PRE_DAYS = 60         # Look back 60 days before announcement
VOL_POST_DAYS = 60        # Look ahead 60 days after announcement

# Sharpe Ratio Time Series Parameters
SHARPE_TIME_GROUPING = 'quarter'  # 'year', 'quarter', or 'month'

# Rolling Sharpe Time Series Parameters
ROLLING_SHARPE_WINDOW = 5  # Window size for rolling Sharpe calculation (in days)
ROLLING_ANALYSIS_WINDOW = (-60, 60)  # Days relative to announcement to analyze


def run_analysis():
    """Runs the FDA Sharpe Ratio and Volatility analysis pipeline."""
    print("--- Starting FDA Sharpe Ratio & Volatility Analysis ---")

    # --- Path Validation & Results Dir Creation ---
    if not os.path.exists(FDA_FILE): print(f"\n*** Error: FDA file not found: {FDA_FILE} ***"); return
    missing_stock = [f for f in STOCK_FILES if not os.path.exists(f)]
    if missing_stock: print(f"\n*** Error: Stock file(s) not found: {missing_stock} ***"); return
    print("File paths validated.")
    try:
        os.makedirs(RESULTS_DIR, exist_ok=True)
        print(f"Results will be saved to: {os.path.abspath(RESULTS_DIR)}")
    except OSError as oe:
        print(f"\n*** Error creating results directory '{RESULTS_DIR}': {oe} ***"); return

    try:
        # --- Initialize Components ---
        print("\nInitializing components...")
        data_loader = DataLoader(fda_path=FDA_FILE, stock_paths=STOCK_FILES, window_days=WINDOW_DAYS)
        feature_engineer = FeatureEngineer(prediction_window=5)
        analyzer = Analysis(data_loader, feature_engineer)
        print("Components initialized.")

        # --- Load Data ---
        print("\nLoading and preparing basic data...")
        analyzer.data = analyzer.data_loader.load_data()
        print(f"Data loaded successfully. Shape: {analyzer.data.shape}")
        if analyzer.data is None or analyzer.data.empty: print("\n*** Error: Data loading failed. ***"); return

        # --- Run Volatility Spike Analysis ---
        analyzer.analyze_volatility_spikes(
            results_dir=RESULTS_DIR,
            file_prefix=FILE_PREFIX,
            window=VOL_ROLLING_WINDOW,
            pre_days=VOL_PRE_DAYS,             # Updated to 60 days
            post_days=VOL_POST_DAYS,           # Updated to 60 days
            baseline_window=VOL_BASELINE_WINDOW,
            event_window=VOL_EVENT_WINDOW
        )
        # --- Run Rolling Sharpe Time Series Analysis ---
        analyzer.calculate_rolling_sharpe_timeseries(
            results_dir=RESULTS_DIR,
            file_prefix=FILE_PREFIX,
            return_col='ret',
            analysis_window=ROLLING_ANALYSIS_WINDOW,
            sharpe_window=ROLLING_SHARPE_WINDOW,
            annualize=True,
            # risk_free_rate=0.0  # Optional
        )

        print(f"\n--- FDA Sharpe Ratio & Volatility Analysis Finished (Results in '{RESULTS_DIR}') ---")

    except ValueError as ve: print(f"\n*** ValueError: {ve} ***"); traceback.print_exc()
    except RuntimeError as re: print(f"\n*** RuntimeError: {re} ***"); traceback.print_exc()
    except FileNotFoundError as fnf: print(f"\n*** FileNotFoundError: {fnf} ***")
    except Exception as e: print(f"\n*** An unexpected error occurred: {e} ***"); traceback.print_exc()


if __name__ == "__main__":
    run_analysis()
