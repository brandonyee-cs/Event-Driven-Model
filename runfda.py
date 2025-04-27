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
    from src.fda_processor import DataLoader, FeatureEngineer, Analysis
    print("Successfully imported FDA processor classes.")
except ImportError as e:
    print(f"Error importing from fda_processor: {e}"); sys.exit(1)

# --- File Paths and Parameters ---
# <<< --- UPDATE THESE PATHS --- >>>
FDA_FILE = 'path/to/your/fda_approvals.csv'
STOCK_FILES = [
    'path/to/stock_data_part1.csv',
    'path/to/stock_data_part2.csv'
]
# <<< -------------------------- >>>

# --- Analysis Parameters ---
RESULTS_DIR = "results_fda" # Directory to save results
FILE_PREFIX = "fda"         # Prefix for saved files
WINDOW_DAYS = 60
SHARPE_ROLLING_WINDOW = 20
VOL_ROLLING_WINDOW = 5
VOL_BASELINE_WINDOW = (-60, -11)
VOL_EVENT_WINDOW = (-2, 2)


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

        # --- Run Sharpe Ratio Analysis ---
        analyzer.analyze_sharpe_ratio_dynamics(
            results_dir=RESULTS_DIR,
            file_prefix=FILE_PREFIX,
            window=SHARPE_ROLLING_WINDOW,
            pre_days=WINDOW_DAYS,
            post_days=WINDOW_DAYS
        )

        # --- Run Volatility Spike Analysis ---
        analyzer.analyze_volatility_spikes(
            results_dir=RESULTS_DIR,
            file_prefix=FILE_PREFIX,
            window=VOL_ROLLING_WINDOW,
            pre_days=WINDOW_DAYS,
            post_days=WINDOW_DAYS,
            baseline_window=VOL_BASELINE_WINDOW,
            event_window=VOL_EVENT_WINDOW
        )

        print(f"\n--- FDA Sharpe Ratio & Volatility Analysis Finished (Results in '{RESULTS_DIR}') ---")

    except ValueError as ve: print(f"\n*** ValueError: {ve} ***"); traceback.print_exc()
    except RuntimeError as re: print(f"\n*** RuntimeError: {re} ***"); traceback.print_exc()
    except FileNotFoundError as fnf: print(f"\n*** FileNotFoundError: {fnf} ***")
    except Exception as e: print(f"\n*** An unexpected error occurred: {e} ***"); traceback.print_exc()


if __name__ == "__main__":
    run_analysis()