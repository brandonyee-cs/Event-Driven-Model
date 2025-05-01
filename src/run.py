import pandas as pd
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import traceback
import polars as pl

# --- Configuration ---
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path: sys.path.append(current_dir)
try: from fda_processor import DataLoader, FeatureEngineer, Analysis; print("Successfully imported FDA processor classes.")
except ImportError as e: print(f"Error importing from fda_processor: {e}"); sys.exit(1)
try: from earnings_processor import DataLoader, FeatureEngineer, Analysis; print("Successfully imported Earnings processor classes (Polars version).")
except ImportError as e: print(f"Error importing from earnings_processor: {e}"); print("Ensure 'earnings_processor.py', 'models.py' are in the same directory or Python path."); print("Ensure Polars is installed: pip install polars pyarrow"); sys.exit(1)

# --- File Paths and Parameters ---
# <<< --- UPDATE THESE PATHS --- >>>
FDA_FILE = '/home/d87016661/fda_ticker_list_2000_to_2024.csv'
EARNINGS_EVENT_FILE = '/home/d87016661/detail_history_actuals.csv'
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
FDA_RESULTS_DIR = "results_fda" # Directory to save results
FDA_FILE_PREFIX = "fda"       # Prefix for saved files
QE_RESULTS_DIR = "results_earnings" # Directory to save results
QE_FILE_PREFIX = "earnings"       # Prefix for saved files
WINDOW_DAYS = 60
# Volatility Analysis Parameters
VOL_ROLLING_WINDOW = 5     # Days for rolling Volatility calculation (row-based)
VOL_BASELINE_WINDOW = (-60, 60) # Days relative to event for baseline vol calc
VOL_EVENT_WINDOW = (-2, 2)       # Days relative to event for event vol calc
VOL_PRE_DAYS = 60         # Look back 60 days before announcement
VOL_POST_DAYS = 60        # Look ahead 60 days after announcement
# Rolling Sharpe Time Series Parameters
ROLLING_SHARPE_WINDOW = 5  # Window size for rolling Sharpe calculation (in days)
ROLLING_ANALYSIS_WINDOW = (-60, 60)  # Days relative to announcement to analyze
# Sharpe Ratio Quantile Analysis Parameters
SHARPE_QUANTILE_LOOKBACK = 10  # Days to look back for calculating Sharpe ratios
SHARPE_QUANTILE_WINDOW = (-60, 60)  # Days relative to announcement for analysis
SHARPE_QUANTILES = [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]  # Quantiles to calculate
# ML Parameters
RUN_ML_ANALYSIS = False # Not yet implemented
ML_PREDICTION_WINDOW = 3    # Target for ML models (e.g., predict 3-day return)
ML_TEST_SPLIT_SIZE = 0.2


def fda_run_analysis():
    """Runs the FDA Sharpe Ratio and Volatility analysis pipeline."""
    print("--- Starting FDA Sharpe Ratio & Volatility Analysis ---")

    # --- Path Validation & Results Dir Creation ---
    if not os.path.exists(FDA_FILE): print(f"\n*** Error: FDA file not found: {FDA_FILE} ***"); return
    missing_stock = [f for f in STOCK_FILES if not os.path.exists(f)]
    if missing_stock: print(f"\n*** Error: Stock file(s) not found: {missing_stock} ***"); return
    print("File paths validated.")
    try:
        os.makedirs(FDA_RESULTS_DIR, exist_ok=True)
        print(f"Results will be saved to: {os.path.abspath(FDA_RESULTS_DIR)}")
    except OSError as oe:
        print(f"\n*** Error creating results directory '{FDA_RESULTS_DIR}': {oe} ***"); return

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
            results_dir=FDA_RESULTS_DIR,
            file_prefix=FDA_FILE_PREFIX,
            window=VOL_ROLLING_WINDOW,
            pre_days=VOL_PRE_DAYS,             # Updated to 60 days
            post_days=VOL_POST_DAYS,           # Updated to 60 days
            baseline_window=VOL_BASELINE_WINDOW,
            event_window=VOL_EVENT_WINDOW
        )
        # --- Run Rolling Sharpe Time Series Analysis ---
        analyzer.calculate_rolling_sharpe_timeseries(
            results_dir=FDA_RESULTS_DIR,
            file_prefix=FDA_FILE_PREFIX,
            return_col='ret',
            analysis_window=ROLLING_ANALYSIS_WINDOW,
            sharpe_window=ROLLING_SHARPE_WINDOW,
            annualize=True,
            # risk_free_rate=0.0  # Optional
        )
        # --- Run Sharpe Ratio Quantile Analysis ---
        analyzer.calculate_sharpe_quantiles(
            results_dir=FDA_RESULTS_DIR,
            file_prefix=FDA_FILE_PREFIX,
            return_col='ret',
            analysis_window=SHARPE_QUANTILE_WINDOW,
            lookback_window=SHARPE_QUANTILE_LOOKBACK,
            quantiles=SHARPE_QUANTILES,
            annualize=True,
            # risk_free_rate=0.0  # Optional
        )

        print(f"\n--- FDA Sharpe Ratio & Volatility Analysis Finished (Results in '{FDA_RESULTS_DIR}') ---")

    except ValueError as ve: print(f"\n*** ValueError: {ve} ***"); traceback.print_exc()
    except RuntimeError as re: print(f"\n*** RuntimeError: {re} ***"); traceback.print_exc()
    except FileNotFoundError as fnf: print(f"\n*** FileNotFoundError: {fnf} ***")
    except Exception as e: print(f"\n*** An unexpected error occurred: {e} ***"); traceback.print_exc()

def qe_run_analysis():
    """Runs the Earnings Sharpe Ratio and Volatility analysis pipeline using Polars."""
    print("--- Starting Earnings Event Analysis (Polars Version) ---")

    # --- Path Validation & Results Dir Creation ---
    if not os.path.exists(EARNINGS_EVENT_FILE): print(f"\n*** Error: Earnings event file not found: {EARNINGS_EVENT_FILE} ***"); return
    missing_stock = [f for f in STOCK_FILES if not os.path.exists(f)]
    if missing_stock: print(f"\n*** Error: Stock file(s) not found: {missing_stock} ***"); return
    print("File paths validated.")
    try:
        os.makedirs(QE_RESULTS_DIR, exist_ok=True)
        print(f"Results will be saved to: {os.path.abspath(QE_RESULTS_DIR)}")
    except OSError as oe: print(f"\n*** Error creating results directory '{QE_RESULTS_DIR}': {oe} ***"); return

    try:
        # --- Initialize Components ---
        print("\nInitializing components (Polars)...")
        feature_engineer_window = ML_PREDICTION_WINDOW if RUN_ML_ANALYSIS else 3
        data_loader = DataLoader(earnings_path=EARNINGS_EVENT_FILE, stock_paths=STOCK_FILES, window_days=WINDOW_DAYS)
        feature_engineer = FeatureEngineer(prediction_window=feature_engineer_window)
        analyzer = Analysis(data_loader, feature_engineer)
        print("Components initialized.")

        # --- Load Data ---
        print("\nLoading event dates and stock data (Polars)...")
        # Use the modified loader which only extracts ticker/date from event file
        analyzer.data = analyzer.load_and_prepare_data(run_feature_engineering=False) # Don't run FE yet
        print(f"Data loaded successfully. Shape: {analyzer.data.shape}")
        if analyzer.data is None or analyzer.data.is_empty(): print("\n*** Error: Data loading failed. ***"); return

        # --- Run Volatility Spike Analysis ---
        analyzer.analyze_volatility_spikes(
            results_dir=QE_RESULTS_DIR,
            file_prefix=QE_FILE_PREFIX,
            window=VOL_ROLLING_WINDOW,
            pre_days=VOL_PRE_DAYS,             # Updated to 60 days
            post_days=VOL_POST_DAYS,           # Updated to 60 days
            baseline_window=VOL_BASELINE_WINDOW,
            event_window=VOL_EVENT_WINDOW
        )

        # --- Run Rolling Sharpe Time Series Analysis ---
        analyzer.calculate_rolling_sharpe_timeseries(
            results_dir=QE_RESULTS_DIR,
            file_prefix=QE_FILE_PREFIX,
            return_col='ret',
            analysis_window=ROLLING_ANALYSIS_WINDOW,
            sharpe_window=ROLLING_SHARPE_WINDOW,
            annualize=True,
            # risk_free_rate=0.0  # Optional
        )
        # --- Run Sharpe Ratio Quantile Analysis ---
        analyzer.calculate_sharpe_quantiles(
            results_dir=QE_RESULTS_DIR,
            file_prefix=QE_FILE_PREFIX,
            return_col='ret',
            analysis_window=SHARPE_QUANTILE_WINDOW,
            lookback_window=SHARPE_QUANTILE_LOOKBACK,
            quantiles=SHARPE_QUANTILES,
            annualize=True,
            # risk_free_rate=0.0  # Optional
        )

        print(f"\n--- Earnings Analysis Finished (Results in '{QE_RESULTS_DIR}') ---")

    except ValueError as ve: print(f"\n*** ValueError: {ve} ***"); traceback.print_exc()
    except RuntimeError as re: print(f"\n*** RuntimeError: {re} ***"); traceback.print_exc()
    except FileNotFoundError as fnf: print(f"\n*** FileNotFoundError: {fnf} ***")
    # Catch Polars-specific errors if necessary
    except pl.exceptions.PolarsError as pe: print(f"\n*** PolarsError: {pe} ***"); traceback.print_exc()
    except Exception as e: print(f"\n*** An unexpected error occurred: {e} ***"); traceback.print_exc()


if __name__ == "__main__":
    fda_run_analysis()
    qe_run_analysis()