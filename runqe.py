import polars as pl
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import traceback

# --- Configuration ---
# Ensure the directory containing the processor files is in the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# Import custom classes AFTER potentially modifying sys.path
try:
    # Assuming models.py is also in the same directory or path
    # The processor files now use Polars
    from earnings_processor import DataLoader, FeatureEngineer, Analysis
    print("Successfully imported Earnings processor classes (Polars version).")
except ImportError as e:
    print(f"Error importing from earnings_processor: {e}")
    print("Ensure 'earnings_processor.py', 'models.py' are in the same directory or Python path.")
    print("Ensure Polars is installed: pip install polars pyarrow")
    sys.exit(1) # Exit if imports fail


# --- File Paths and Parameters ---
# <<< --- UPDATE THESE PATHS --- >>>
# This file should contain Ticker and ANNDATS (or similar actual announcement date)
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
RESULTS_DIR = "results_earnings_polars" # Directory to save results
FILE_PREFIX = "earnings"       # Prefix for saved files
WINDOW_DAYS = 30               # Days around event for data loading

# Event Strategy Parameters
STRATEGY_HOLDING_PERIOD = 20 # Days to hold after entry
STRATEGY_ENTRY_DAY = 0      # Day relative to event to enter (0=Announcement Day)

# Volatility Analysis Parameters
VOL_ROLLING_WINDOW = 5     # Days for rolling Volatility calculation (row-based)
VOL_BASELINE_WINDOW = (-60, -11) # Days relative to event for baseline vol calc
VOL_EVENT_WINDOW = (-2, 2)       # Days relative to event for event vol calc

# Sharpe Ratio Comparison Parameters
SHARPE_ENTRY_DAYS = [-1, 0, 1, 2]  # Days relative to announcement to enter
SHARPE_HOLDING_PERIODS = [5, 10, 20, 30]  # Holding periods to compare

# --- Optional ML Parameters ---
# Set to True to run the ML prediction parts (slower)
RUN_ML_ANALYSIS = False # Keep False initially for faster testing
ML_PREDICTION_WINDOW = 3    # Target for ML models (e.g., predict 3-day return)
ML_TEST_SPLIT_SIZE = 0.2

def run_analysis():
    """Runs the Earnings Sharpe Ratio and Volatility analysis pipeline using Polars."""
    print("--- Starting Earnings Event Analysis (Polars Version) ---")

    # --- Path Validation & Results Dir Creation ---
    if not os.path.exists(EARNINGS_EVENT_FILE): print(f"\n*** Error: Earnings event file not found: {EARNINGS_EVENT_FILE} ***"); return
    missing_stock = [f for f in STOCK_FILES if not os.path.exists(f)]
    if missing_stock: print(f"\n*** Error: Stock file(s) not found: {missing_stock} ***"); return
    print("File paths validated.")
    try:
        os.makedirs(RESULTS_DIR, exist_ok=True)
        print(f"Results will be saved to: {os.path.abspath(RESULTS_DIR)}")
    except OSError as oe: print(f"\n*** Error creating results directory '{RESULTS_DIR}': {oe} ***"); return

    try:
        # --- Initialize Components ---
        print("\nInitializing components (Polars)...")
        # FeatureEngineer window doesn't matter if not running ML, but needs a value
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

        # --- Run Event Strategy Sharpe Ratio Analysis ---
        analyzer.analyze_event_sharpe_ratio(
            results_dir=RESULTS_DIR,
            file_prefix=FILE_PREFIX,
            holding_period=STRATEGY_HOLDING_PERIOD,
            entry_day=STRATEGY_ENTRY_DAY,
            # risk_free_rate=0.0 # Optional
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
        # --- Run Sharpe Ratio Comparison Analysis ---
        analyzer.plot_sharpe_ratio_comparison(
            results_dir=RESULTS_DIR,
            file_prefix=FILE_PREFIX,
            entry_days=SHARPE_ENTRY_DAYS,
            holding_periods=SHARPE_HOLDING_PERIODS,
            # risk_free_rate=0.0  # Optional
        )

        # --- Optional: Run ML Prediction Analysis ---
        if RUN_ML_ANALYSIS:
            print("\n--- Running Optional ML Analysis (Polars) ---")
            # Reload/prepare data including features/target needed for ML
            print("   Re-preparing data with features and target for ML...")
            # analyzer.load_and_prepare_data(run_feature_engineering=True) # This will calculate features/target

            # 1. Create target and calculate features (needs to happen before train/eval)
            print("   Creating target variable for ML...")
            analyzer.data = analyzer.feature_engineer.create_target(analyzer.data)
            print("   Calculating features for ML...")
            # Run feature calculation again, explicitly setting fit_categorical for train later
            # Note: calculate_features now returns the df, doesn't modify in place implicitly
            analyzer.data = analyzer.feature_engineer.calculate_features(analyzer.data, fit_categorical=False)


            # 2. Train models (Analysis.train_models handles splitting and feature processing)
            print("   Training ML models...")
            analyzer.train_models(test_size=ML_TEST_SPLIT_SIZE) # This now handles feature fitting on train split

            # 3. Evaluate models
            print("   Evaluating ML models...")
            results_dict = analyzer.evaluate_models() # Uses stored X_test_np, y_test_np

            print("\n--- ML Evaluation Summary ---")
            # Display results (convert Polars Series/DF to Pandas for display if needed, or print directly)
            if 'standard' in results_dict and results_dict['standard']:
                 print("\nStandard Models:")
                 # Convert dict of dicts to Polars DF for printing
                 print(pl.DataFrame(results_dict['standard']).transpose(include_header=True, column_names=list(results_dict['standard'].keys())))

            if 'surprise' in results_dict and results_dict['surprise']:
                 print("\nSurprise Model:")
                 # Convert dict to Polars Series/DF for printing
                 print(pl.DataFrame(results_dict.get('surprise', {}))) # Simple dict display

            if 'pead' in results_dict and results_dict['pead']:
                 print("\nPEAD Model:")
                 # Convert dict of dicts to Polars DF for printing
                 print(pl.DataFrame(results_dict.get('pead', {})).transpose(include_header=True, column_names=list(results_dict['pead'].keys())))


            # 4. Plot ML-related results (optional)
            # Ensure plotting functions handle Polars input or convert internally
            if analyzer.models:
                model_to_plot_imp = 'XGBoostDecile' if 'XGBoostDecile' in analyzer.models else 'TimeSeriesRidge'
                if model_to_plot_imp in analyzer.models:
                    analyzer.plot_feature_importance(results_dir=RESULTS_DIR, file_prefix=FILE_PREFIX, model_name=model_to_plot_imp, top_n=15)

            if analyzer.surprise_model: # Check if surprise model exists
                 # This plotting function needs access to data, potentially test split
                 analyzer.analyze_earnings_surprise(results_dir=RESULTS_DIR, file_prefix=FILE_PREFIX) # Plot surprise impact

            if analyzer.pead_model: # Check if pead model exists
                 analyzer.plot_pead_predictions(results_dir=RESULTS_DIR, file_prefix=FILE_PREFIX, n_events=2) # Plot PEAD

            print("--- Optional ML Analysis Complete ---")
        # ---------------------------------------------

        print(f"\n--- Earnings Analysis Finished (Results in '{RESULTS_DIR}') ---")

    except ValueError as ve: print(f"\n*** ValueError: {ve} ***"); traceback.print_exc()
    except RuntimeError as re: print(f"\n*** RuntimeError: {re} ***"); traceback.print_exc()
    except FileNotFoundError as fnf: print(f"\n*** FileNotFoundError: {fnf} ***")
    # Catch Polars-specific errors if necessary
    except pl.exceptions.PolarsError as pe: print(f"\n*** PolarsError: {pe} ***"); traceback.print_exc()
    except Exception as e: print(f"\n*** An unexpected error occurred: {e} ***"); traceback.print_exc()


if __name__ == "__main__":
    run_analysis()
