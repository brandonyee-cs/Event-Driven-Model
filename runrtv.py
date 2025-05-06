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
                    delta_days: int = 10,
                    use_actual_values: bool = False) -> Dict[str, Any]:
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
    use_actual_values (bool): Whether to use actual values instead of rolling windows
    
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
        
        # Create appropriate results directory based on method
        method_dir = os.path.join(results_dir, "actual" if use_actual_values else "rolling")
        os.makedirs(method_dir, exist_ok=True)
        
        # Run hypothesis test with appropriate method
        if use_actual_values:
            print(f"\nRunning actual-value hypothesis test with delta = {delta_days} days...")
            results = rtv_analysis.run_actual_hypothesis_1_test(
                return_col=return_col,
                delta_days=delta_days,
                results_dir=method_dir,
                file_prefix=file_prefix
            )
        else:
            print(f"\nRunning rolling-window hypothesis test with delta = {delta_days} days...")
            results = rtv_analysis.run_hypothesis_1_test(
                return_col=return_col,
                rolling_window=rolling_window,
                delta_days=delta_days,
                results_dir=method_dir,
                file_prefix=file_prefix
            )
        
        method_name = "Actual Values" if use_actual_values else "Rolling Window"
        print(f"\n=== Hypothesis 1 Test ({method_name}) Completed ===")
        print(f"Results saved to: {os.path.abspath(method_dir)}")
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
    
    # Run FDA analysis with rolling windows
    print("\n=== Starting FDA Approval Event RTV Analysis (Rolling Window) ===")
    fda_rolling_results = run_rtv_analysis(
        event_file=FDA_EVENT_FILE,
        stock_files=FDA_STOCK_FILES,
        results_dir=FDA_RESULTS_DIR,
        file_prefix=FDA_FILE_PREFIX,
        event_date_col=FDA_EVENT_DATE_COL,
        ticker_col=FDA_TICKER_COL,
        window_days=WINDOW_DAYS,
        return_col=RETURN_COL,
        rolling_window=ROLLING_WINDOW,
        delta_days=DELTA_DAYS,
        use_actual_values=False
    )
    
    # Run FDA analysis with actual values
    print("\n=== Starting FDA Approval Event RTV Analysis (Actual Values) ===")
    fda_actual_results = run_rtv_analysis(
        event_file=FDA_EVENT_FILE,
        stock_files=FDA_STOCK_FILES,
        results_dir=FDA_RESULTS_DIR,
        file_prefix=FDA_FILE_PREFIX,
        event_date_col=FDA_EVENT_DATE_COL,
        ticker_col=FDA_TICKER_COL,
        window_days=WINDOW_DAYS,
        return_col=RETURN_COL,
        rolling_window=ROLLING_WINDOW,
        delta_days=DELTA_DAYS,
        use_actual_values=True
    )
    
    # Run Earnings analysis with rolling windows
    print("\n=== Starting Earnings Announcement Event RTV Analysis (Rolling Window) ===")
    earnings_rolling_results = run_rtv_analysis(
        event_file=EARNINGS_EVENT_FILE,
        stock_files=EARNINGS_STOCK_FILES,
        results_dir=EARNINGS_RESULTS_DIR,
        file_prefix=EARNINGS_FILE_PREFIX,
        event_date_col=EARNINGS_EVENT_DATE_COL,
        ticker_col=EARNINGS_TICKER_COL,
        window_days=WINDOW_DAYS,
        return_col=RETURN_COL,
        rolling_window=ROLLING_WINDOW,
        delta_days=DELTA_DAYS,
        use_actual_values=False
    )
    
    # Run Earnings analysis with actual values
    print("\n=== Starting Earnings Announcement Event RTV Analysis (Actual Values) ===")
    earnings_actual_results = run_rtv_analysis(
        event_file=EARNINGS_EVENT_FILE,
        stock_files=EARNINGS_STOCK_FILES,
        results_dir=EARNINGS_RESULTS_DIR,
        file_prefix=EARNINGS_FILE_PREFIX,
        event_date_col=EARNINGS_EVENT_DATE_COL,
        ticker_col=EARNINGS_TICKER_COL,
        window_days=WINDOW_DAYS,
        return_col=RETURN_COL,
        rolling_window=ROLLING_WINDOW,
        delta_days=DELTA_DAYS,
        use_actual_values=True
    )
    
    # Create comparison summary
    comparison_dir = "results/rtv_analysis/comparison"
    os.makedirs(comparison_dir, exist_ok=True)
    
    try:
        with open(os.path.join(comparison_dir, "rtv_method_comparison.txt"), "w") as f:
            f.write("===== Hypothesis 1 Test Comparison: Rolling Window vs Actual Values =====\n\n")
            f.write(f"Parameter δ (post-event rising phase duration): {DELTA_DAYS} days\n\n")
            
            f.write("--- FDA Approval Events ---\n")
            f.write(f"Rolling Window Method: Hypothesis Supported = {fda_rolling_results.get('hypothesis_supported', False)}\n")
            f.write(f"Actual Values Method: Hypothesis Supported = {fda_actual_results.get('hypothesis_supported', False)}\n\n")
            
            f.write("--- Earnings Announcement Events ---\n")
            f.write(f"Rolling Window Method: Hypothesis Supported = {earnings_rolling_results.get('hypothesis_supported', False)}\n")
            f.write(f"Actual Values Method: Hypothesis Supported = {earnings_actual_results.get('hypothesis_supported', False)}\n\n")
            
            f.write("=== Conclusion ===\n")
            f.write("This comparison assesses whether Hypothesis 1 (return-to-variance ratio peaks during post-event\n")
            f.write("rising phase) holds when using both methodologies: rolling windows and actual values.\n")
            
            # Determine overall conclusion based on results
            rolling_supported = fda_rolling_results.get('hypothesis_supported', False) or earnings_rolling_results.get('hypothesis_supported', False)
            actual_supported = fda_actual_results.get('hypothesis_supported', False) or earnings_actual_results.get('hypothesis_supported', False)
            
            if rolling_supported and actual_supported:
                f.write("\nBoth methodologies support Hypothesis 1, indicating robust evidence that the\n")
                f.write("return-to-variance ratio indeed peaks during the post-event rising phase.\n")
            elif rolling_supported:
                f.write("\nOnly the rolling window methodology supports Hypothesis 1, suggesting that the pattern\n")
                f.write("may be more apparent when using smoothed metrics than when using raw event-phase values.\n")
            elif actual_supported:
                f.write("\nOnly the actual values methodology supports Hypothesis 1, indicating that the pattern\n")
                f.write("exists in the raw data but may be diluted by the smoothing effect of rolling windows.\n")
            else:
                f.write("\nNeither methodology fully supports Hypothesis 1, suggesting that the hypothesized\n")
                f.write("return-to-variance pattern may not be consistent across the analyzed events.\n")
        
        print(f"\nComparison summary saved to: {os.path.join(comparison_dir, 'rtv_method_comparison.txt')}")
    except Exception as e:
        print(f"Error creating comparison summary: {e}")

if __name__ == "__main__":
    main()