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
try: 
    from event_processor import EventDataLoader, EventFeatureEngineer, EventAnalysis
    print("Successfully imported Event processor classes.")
except ImportError as e: 
    print(f"Error importing from event_processor: {e}")
    print("Ensure 'event_processor.py' and 'models.py' are in the same directory or Python path.")
    print("Ensure Polars is installed: pip install polars pyarrow")
    sys.exit(1)

pl.Config.set_engine_affinity(engine="streaming")

# --- Hardcoded Analysis Parameters from config.yaml ---
# Shared stock files for both analyses
STOCK_FILES = [
    "/home/d87016661/crsp_dsf-2000-2001.parquet",
    "/home/d87016661/crsp_dsf-2002-2003.parquet",
    "/home/d87016661/crsp_dsf-2004-2005.parquet",
    "/home/d87016661/crsp_dsf-2006-2007.parquet",
    "/home/d87016661/crsp_dsf-2008-2009.parquet",
    "/home/d87016661/crsp_dsf-2010-2011.parquet",
    "/home/d87016661/crsp_dsf-2016-2017.parquet",
    "/home/d87016661/crsp_dsf-2018-2019.parquet",
    "/home/d87016661/crsp_dsf-2020-2021.parquet",
    "/home/d87016661/crsp_dsf-2022-2023.parquet",
    "/home/d87016661/crsp_dsf-2024-2025.parquet"
]

# Earnings event specific parameters
EARNINGS_EVENT_FILE = "/home/d87016661/detail_history_actuals.csv"
EARNINGS_RESULTS_DIR = "results/results_earnings/"
EARNINGS_FILE_PREFIX = "earnings"
EARNINGS_EVENT_DATE_COL = "ANNDATS"
EARNINGS_TICKER_COL = "ticker"

# Comparison directory
COMPARISON_DIR = "results_comparison"

# Shared analysis parameters
WINDOW_DAYS = 60
    
# Volatility analysis
VOL_WINDOW = 5
VOL_PRE_DAYS = 60
VOL_POST_DAYS = 60
VOL_BASELINE_START = -60
VOL_BASELINE_END = -11
VOL_EVENT_START = -2
VOL_EVENT_END = 2
    
# Sharpe ratio analysis
SHARPE_WINDOW = 5
SHARPE_ANALYSIS_START = -60
SHARPE_ANALYSIS_END = 60
SHARPE_LOOKBACK = 10
    
# Machine learning parameters
ML_WINDOW = 3
RUN_ML = True
ML_TEST_SPLIT = 0.2

def run_earnings_analysis():
    """
    Runs the earnings event analysis pipeline using parameters from config.yaml.
    """
    print("\n=== Starting Earnings Announcement Event Analysis ===")

    # --- Path Validation & Results Dir Creation ---
    if not os.path.exists(EARNINGS_EVENT_FILE): 
        print(f"\n*** Error: Earnings event file not found: {EARNINGS_EVENT_FILE} ***")
        return False
    
    missing_stock = [f for f in STOCK_FILES if not os.path.exists(f)]
    if missing_stock: 
        print(f"\n*** Error: Stock file(s) not found: {missing_stock} ***")
        return False
    
    print("Earnings file paths validated.")
    try:
        os.makedirs(EARNINGS_RESULTS_DIR, exist_ok=True)
        print(f"Earnings results will be saved to: {os.path.abspath(EARNINGS_RESULTS_DIR)}")
    except OSError as oe:
        print(f"\n*** Error creating earnings results directory '{EARNINGS_RESULTS_DIR}': {oe} ***")
        return False

    try:
        # --- Initialize Components ---
        print("\nInitializing earnings components...")
        data_loader = EventDataLoader(
            event_path=EARNINGS_EVENT_FILE, 
            stock_paths=STOCK_FILES, 
            window_days=WINDOW_DAYS,
            event_date_col=EARNINGS_EVENT_DATE_COL,
            ticker_col=EARNINGS_TICKER_COL
        )
        feature_engineer = EventFeatureEngineer(prediction_window=ML_WINDOW)
        analyzer = EventAnalysis(data_loader, feature_engineer)
        print("Earnings components initialized.")

        # --- Load Data ---
        print("\nLoading and preparing earnings data...")
        analyzer.data = analyzer.load_and_prepare_data(run_feature_engineering=RUN_ML)
        if analyzer.data is None or analyzer.data.is_empty(): 
            print("\n*** Error: Earnings data loading failed. ***")
            return False
            
        print(f"Earnings data loaded successfully. Shape: {analyzer.data.shape}")
        
        # Setup analysis parameters
        baseline_window = (VOL_BASELINE_START, VOL_BASELINE_END)
        event_window = (VOL_EVENT_START, VOL_EVENT_END)
        analysis_window = (SHARPE_ANALYSIS_START, SHARPE_ANALYSIS_END)
        
        # --- Run Volatility Spike Analysis ---
        analyzer.analyze_volatility_spikes(
            results_dir=EARNINGS_RESULTS_DIR,
            file_prefix=EARNINGS_FILE_PREFIX,
            window=VOL_WINDOW,
            pre_days=VOL_PRE_DAYS,
            post_days=VOL_POST_DAYS,
            baseline_window=baseline_window,
            event_window=event_window
        )
        
        # --- Run Rolling Sharpe Time Series Analysis ---
        analyzer.calculate_rolling_sharpe_timeseries(
            results_dir=EARNINGS_RESULTS_DIR,
            file_prefix=EARNINGS_FILE_PREFIX,
            return_col='ret',
            analysis_window=analysis_window,
            sharpe_window=SHARPE_WINDOW,
            annualize=True,
        )
        
        # --- Run Sharpe Ratio Quantile Analysis ---
        analyzer.calculate_sharpe_quantiles(
            results_dir=EARNINGS_RESULTS_DIR,
            file_prefix=EARNINGS_FILE_PREFIX,
            return_col='ret',
            analysis_window=analysis_window,
            lookback_window=SHARPE_LOOKBACK,
            quantiles=[0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95],
            annualize=True,
        )
        
        # --- Run ML Analysis if requested ---
        if RUN_ML:
            print("\n--- Running Earnings ML Analysis ---")
            # Train models
            analyzer.train_models(test_size=ML_TEST_SPLIT, time_split_column=EARNINGS_EVENT_DATE_COL)
            
            # Evaluate models
            results = analyzer.evaluate_models()
            
            # Plot feature importance
            for model_name in analyzer.models.keys():
                analyzer.plot_feature_importance(
                    results_dir=EARNINGS_RESULTS_DIR,
                    file_prefix=EARNINGS_FILE_PREFIX,
                    model_name=model_name
                )
            
            # Plot predictions for sample events
            sample_events = analyzer.find_sample_event_ids(n=3)
            for event_id in sample_events:
                for model_name in analyzer.models.keys():
                    analyzer.plot_predictions_for_event(
                        results_dir=EARNINGS_RESULTS_DIR,
                        event_id=event_id,
                        file_prefix=EARNINGS_FILE_PREFIX,
                        model_name=model_name
                    )
        
        print(f"\n--- Earnings Event Analysis Finished (Results in '{EARNINGS_RESULTS_DIR}') ---")
        return True

    except ValueError as ve: 
        print(f"\n*** Earnings ValueError: {ve} ***")
        traceback.print_exc()
    except RuntimeError as re: 
        print(f"\n*** Earnings RuntimeError: {re} ***")
        traceback.print_exc()
    except FileNotFoundError as fnf: 
        print(f"\n*** Earnings FileNotFoundError: {fnf} ***")
    except pl.exceptions.PolarsError as pe: 
        print(f"\n*** Earnings PolarsError: {pe} ***")
        traceback.print_exc()
    except Exception as e: 
        print(f"\n*** An unexpected error occurred in earnings analysis: {e} ***")
        traceback.print_exc()
    
    return False

if __name__ == "__main__":
    run_earnings_analysis()