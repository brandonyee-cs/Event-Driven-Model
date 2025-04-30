import polars as pl
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
    # Assuming models.py is also in the same directory or path
    # The processor files now use Polars
    from polars_pre_chunk.fda_processor import DataLoader, FeatureEngineer, Analysis
    print("Successfully imported FDA processor classes (Polars version).")
except ImportError as e:
    print(f"Error importing from fda_processor: {e}")
    print("Ensure 'fda_processor.py', 'models.py' are in the same directory or Python path.")
    print("Ensure Polars is installed: pip install polars pyarrow")
    sys.exit(1)

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
RESULTS_DIR = "results_fda_polars" # Directory to save results
FILE_PREFIX = "fda"         # Prefix for saved files
WINDOW_DAYS = 60            # Days around event for loading data
SHARPE_ROLLING_WINDOW = 20  # Rows for rolling Sharpe calculation
VOL_ROLLING_WINDOW = 5      # Rows for rolling Volatility calculation
VOL_BASELINE_WINDOW = (-60, -11) # Days relative to event for baseline vol calc
VOL_EVENT_WINDOW = (-2, 2)       # Days relative to event for event vol calc


def run_analysis():
    """Runs the FDA Sharpe Ratio and Volatility analysis pipeline using Polars."""
    print("--- Starting FDA Sharpe Ratio & Volatility Analysis (Polars Version) ---")

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
        print("\nInitializing components (Polars)...")
        data_loader = DataLoader(fda_path=FDA_FILE, stock_paths=STOCK_FILES, window_days=WINDOW_DAYS)
        # Prediction window is not strictly needed if only running Sharpe/Vol analysis
        feature_engineer = FeatureEngineer(prediction_window=5)
        analyzer = Analysis(data_loader, feature_engineer)
        print("Components initialized.")

        # --- Load Data ---
        # No need for full feature engineering for these analyses
        print("\nLoading FDA event and stock data (Polars)...")
        # Load data without running feature engineering yet
        analyzer.data = analyzer.data_loader.load_data()
        print(f"Data loaded successfully. Shape: {analyzer.data.shape}")
        if analyzer.data is None or analyzer.data.is_empty():
             print("\n*** Error: Data loading failed. ***"); return

        # --- Run Sharpe Ratio Analysis ---
        analyzer.analyze_sharpe_ratio_dynamics(
            results_dir=RESULTS_DIR,
            file_prefix=FILE_PREFIX,
            window=SHARPE_ROLLING_WINDOW, # Row-based window
            pre_days=WINDOW_DAYS,
            post_days=WINDOW_DAYS
            # risk_free_rate=0.0 # Optional, default is 0
        )

        # --- Run Volatility Spike Analysis ---
        analyzer.analyze_volatility_spikes(
            results_dir=RESULTS_DIR,
            file_prefix=FILE_PREFIX,
            window=VOL_ROLLING_WINDOW,    # Row-based window
            pre_days=WINDOW_DAYS,
            post_days=WINDOW_DAYS,
            baseline_window=VOL_BASELINE_WINDOW,
            event_window=VOL_EVENT_WINDOW
        )

        print(f"\n--- FDA Sharpe Ratio & Volatility Analysis Finished (Results in '{RESULTS_DIR}') ---")

    except ValueError as ve: print(f"\n*** ValueError: {ve} ***"); traceback.print_exc()
    except RuntimeError as re: print(f"\n*** RuntimeError: {re} ***"); traceback.print_exc()
    except FileNotFoundError as fnf: print(f"\n*** FileNotFoundError: {fnf} ***")
    # Catch Polars-specific errors if necessary
    except pl.exceptions.PolarsError as pe: print(f"\n*** PolarsError: {pe} ***"); traceback.print_exc()
    except Exception as e: print(f"\n*** An unexpected error occurred: {e} ***"); traceback.print_exc()


if __name__ == "__main__":
    run_analysis()