import os
import sys
import polars as pl
import numpy as np
import traceback
from typing import Dict, Any

# --- Configuration ---
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path: sys.path.append(current_dir)

# Import from existing modules
try:
    from src.event_processor import EventDataLoader, EventFeatureEngineer, EventAnalysis
    from src.models import TimeSeriesRidge, XGBoostDecileModel
    from src.rtv_analysis import ReturnToVarianceAnalysis
    print("Successfully imported all required modules.")
except ImportError as e:
    print(f"Error importing required modules: {e}")
    print("Ensure all required files are in the correct path.")
    sys.exit(1)

pl.Config.set_engine_affinity(engine="streaming")

def run_rtv_analysis(event_file: str,
                    stock_files: list,
                    results_dir: str,
                    file_prefix: str,
                    event_date_col: str,
                    ticker_col: str,
                    window_days: int = 60,
                    return_col: str = "ret",
                    rolling_window: int = 5,
                    delta_days: int = 10) -> Dict[str, Any]:
    """
    Run the return-to-variance analysis to test Hypothesis 1.
    
    Parameters:
    event_file (str): Path to event data file (CSV)
    stock_files (list): List of paths to stock data files (Parquet)
    results_dir (str): Directory to save results
    file_prefix (str): Prefix for saved files
    event_date_col (str): Column name containing event dates
    ticker_col (str): Column name containing ticker symbols
    window_days (int): Days before/after event to include
    return_col (str): Column name containing returns
    rolling_window (int): Window size for rolling calculations
    delta_days (int): Parameter δ from the paper (post-event rising phase duration)
    
    Returns:
    Dict[str, Any]: Results of the hypothesis test
    """
    try:
        # Initialize data loader
        data_loader = EventDataLoader(
            event_path=event_file,
            stock_paths=stock_files,
            window_days=window_days,
            event_date_col=event_date_col,
            ticker_col=ticker_col
        )
        
        # Initialize event analysis
        feature_engineer = EventFeatureEngineer(prediction_window=3)  # Default value, not used for RTV
        event_analysis = EventAnalysis(data_loader, feature_engineer)
        
        # Load data
        print("Loading and preparing data...")
        event_analysis.data = event_analysis.load_and_prepare_data(run_feature_engineering=False)
        
        if event_analysis.data is None or event_analysis.data.is_empty():
            print("Error: Failed to load data.")
            return {"hypothesis_supported": False, "error": "Failed to load data"}
        
        print(f"Data loaded successfully. Shape: {event_analysis.data.shape}")
        
        # Initialize return-to-variance analysis
        rtv_analysis = ReturnToVarianceAnalysis(event_analysis)
        
        # Run hypothesis test
        print(f"\nRunning hypothesis test with delta = {delta_days} days...")
        results = rtv_analysis.run_hypothesis_1_test(
            return_col=return_col,
            rolling_window=rolling_window,
            delta_days=delta_days,
            results_dir=results_dir,
            file_prefix=file_prefix
        )
        
        print("\n=== Hypothesis 1 Test Completed ===")
        print(f"Results saved to: {os.path.abspath(results_dir)}")
        print(f"Hypothesis Supported: {results['hypothesis_supported']}")
        
        return results
        
    except Exception as e:
        print(f"Error running hypothesis test: {e}")
        traceback.print_exc()
        return {"hypothesis_supported": False, "error": str(e)}


def main():
    """Main function to run the hypothesis test."""
    
    # --- Configuration ---
    # FDA event specific parameters
    FDA_EVENT_FILE = "/home/d87016661/fda_ticker_list_2000_to_2024.csv"
    FDA_STOCK_FILES = [
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
    FDA_RESULTS_DIR = "results/rtv_analysis/fda"
    FDA_FILE_PREFIX = "fda"
    FDA_EVENT_DATE_COL = "Approval Date"
    FDA_TICKER_COL = "ticker"
    
    # Earnings event specific parameters
    EARNINGS_EVENT_FILE = "/home/d87016661/detail_history_actuals.csv"
    EARNINGS_STOCK_FILES = FDA_STOCK_FILES  # Using the same stock files
    EARNINGS_RESULTS_DIR = "results/rtv_analysis/earnings"
    EARNINGS_FILE_PREFIX = "earnings"
    EARNINGS_EVENT_DATE_COL = "ANNDATS"
    EARNINGS_TICKER_COL = "ticker"
    
    # Analysis parameters
    WINDOW_DAYS = 60
    RETURN_COL = "ret"
    ROLLING_WINDOW = 5
    DELTA_DAYS = 10  # δ parameter from the paper
    
    # Create results directories
    os.makedirs(FDA_RESULTS_DIR, exist_ok=True)
    os.makedirs(EARNINGS_RESULTS_DIR, exist_ok=True)
    
    # Run FDA analysis
    print("\n=== Starting FDA Approval Event RTV Analysis ===")
    fda_results = run_rtv_analysis(
        event_file=FDA_EVENT_FILE,
        stock_files=FDA_STOCK_FILES,
        results_dir=FDA_RESULTS_DIR,
        file_prefix=FDA_FILE_PREFIX,
        event_date_col=FDA_EVENT_DATE_COL,
        ticker_col=FDA_TICKER_COL,
        window_days=WINDOW_DAYS,
        return_col=RETURN_COL,
        rolling_window=ROLLING_WINDOW,
        delta_days=DELTA_DAYS
    )
    
    # Run Earnings analysis
    print("\n=== Starting Earnings Announcement Event RTV Analysis ===")
    earnings_results = run_rtv_analysis(
        event_file=EARNINGS_EVENT_FILE,
        stock_files=EARNINGS_STOCK_FILES,
        results_dir=EARNINGS_RESULTS_DIR,
        file_prefix=EARNINGS_FILE_PREFIX,
        event_date_col=EARNINGS_EVENT_DATE_COL,
        ticker_col=EARNINGS_TICKER_COL,
        window_days=WINDOW_DAYS,
        return_col=RETURN_COL,
        rolling_window=ROLLING_WINDOW,
        delta_days=DELTA_DAYS
    )

if __name__ == "__main__":
    main()