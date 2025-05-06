import os
import sys
import polars as pl
import numpy as np
import traceback
from typing import Dict, Any
from datetime import datetime

# --- Configuration ---
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path: sys.path.append(current_dir)

# Import from existing modules
try:
    from src.event_processor import EventDataLoader, EventFeatureEngineer, EventAnalysis
    from src.models import TimeSeriesRidge, XGBoostDecileModel
    from src.vix_analysis import RefinedVIXAnalysis
    print("Successfully imported all required modules.")
except ImportError as e:
    print(f"Error importing required modules: {e}")
    print("Ensure all required files are in the correct path.")
    sys.exit(1)

pl.Config.set_engine_affinity(engine="streaming")

def load_and_prepare_vix_data(vix_file_path: str) -> pl.DataFrame:
    """
    Load VIX data from CSV file and prepare it for merging with stock data.
    
    Parameters:
    vix_file_path (str): Path to VIX CSV file
    
    Returns:
    pl.DataFrame: Prepared VIX data
    """
    print(f"Loading VIX data from: {vix_file_path}")
    
    try:
        # Load VIX data
        vix_data = pl.read_csv(vix_file_path)
        
        # Explore the schema to see available columns
        print(f"VIX data columns: {vix_data.columns}")
        
        # Detect date column and VIX value column
        date_columns = [col for col in vix_data.columns if 'date' in col.lower()]
        vix_value_columns = [col for col in vix_data.columns if 'vix' in col.lower() or 'close' in col.lower()]
        
        if not date_columns:
            raise ValueError("No date column found in VIX data.")
        if not vix_value_columns:
            raise ValueError("No VIX value column found in VIX data.")
        
        date_col = date_columns[0]
        vix_val_col = vix_value_columns[0]
        
        print(f"Using '{date_col}' as date column and '{vix_val_col}' as VIX value column.")
        
        # Standardize column names
        vix_data = vix_data.rename({date_col: 'date', vix_val_col: 'vix_value'})
        
        # Ensure date is in datetime format
        if vix_data['date'].dtype != pl.Datetime:
            vix_data = vix_data.with_columns(
                pl.col('date').str.to_datetime(strict=False).alias('date')
            )
        
        # Convert VIX value to float
        vix_data = vix_data.with_columns(
            pl.col('vix_value').cast(pl.Float64).alias('vix_value')
        )
        
        # Select only necessary columns
        vix_data = vix_data.select(['date', 'vix_value'])
        
        # Check for missing values
        null_count = vix_data.filter(pl.col('vix_value').is_null()).height
        if null_count > 0:
            print(f"Warning: {null_count} rows with missing VIX values detected.")
            
            # Fill or drop nulls based on ratio
            if null_count / vix_data.height < 0.05:  # If less than 5% are nulls
                # Fill with forward fill, then backward fill
                vix_data = vix_data.with_columns(
                    pl.col('vix_value').fill_null(strategy='forward').alias('vix_value')
                )
                vix_data = vix_data.with_columns(
                    pl.col('vix_value').fill_null(strategy='backward').alias('vix_value')
                )
                print("Missing values filled with forward/backward fill.")
            else:
                # If too many nulls, drop them
                vix_data = vix_data.drop_nulls()
                print("Rows with missing values dropped.")
        
        print(f"VIX data loaded successfully. Shape: {vix_data.shape}")
        
        return vix_data
        
    except Exception as e:
        print(f"Error loading VIX data: {e}")
        traceback.print_exc()
        return None

def merge_stock_data_with_vix(stock_data: pl.DataFrame, vix_data: pl.DataFrame) -> pl.DataFrame:
    """
    Merge stock data with VIX data based on date.
    
    Parameters:
    stock_data (pl.DataFrame): Stock data
    vix_data (pl.DataFrame): VIX data
    
    Returns:
    pl.DataFrame: Merged data
    """
    print("Merging stock data with VIX data...")
    
    try:
        # Ensure date columns are in the same format
        if stock_data['date'].dtype != vix_data['date'].dtype:
            stock_data = stock_data.with_columns(
                pl.col('date').cast(pl.Datetime).alias('date')
            )
            vix_data = vix_data.with_columns(
                pl.col('date').cast(pl.Datetime).alias('date')
            )
        
        # Join stock data with VIX data
        merged_data = stock_data.join(
            vix_data.select(['date', 'vix_value']),
            on='date',
            how='left'
        )
        
        # Check for missing VIX values after merge
        null_count = merged_data.filter(pl.col('vix_value').is_null()).height
        if null_count > 0:
            print(f"Warning: {null_count}/{merged_data.height} rows with missing VIX values after merge.")
            
            # Fill or drop nulls based on ratio
            if null_count / merged_data.height < 0.10:  # If less than 10% are nulls
                # Fill with forward fill, then backward fill by event_id
                merged_data = merged_data.with_columns(
                    pl.col('vix_value').fill_null(strategy='forward').over('event_id').alias('vix_value')
                )
                merged_data = merged_data.with_columns(
                    pl.col('vix_value').fill_null(strategy='backward').over('event_id').alias('vix_value')
                )
                print("Missing VIX values filled within each event.")
            else:
                print("Warning: High percentage of missing VIX values.")
        
        # Rename to standard column name expected by analysis code
        merged_data = merged_data.rename({'vix_value': 'vix'})
        
        print(f"Data merged successfully. Shape: {merged_data.shape}")
        
        return merged_data
        
    except Exception as e:
        print(f"Error merging data: {e}")
        traceback.print_exc()
        return stock_data  # Return original data if merge fails

def run_refined_vix_analysis(event_file: str,
                    stock_files: list,
                    vix_file: str,
                    results_dir: str,
                    file_prefix: str,
                    event_date_col: str,
                    ticker_col: str,
                    return_col: str = "ret",
                    window_days: int = 60,
                    delta_days: int = 10) -> Dict[str, Any]:
    """
    Run the refined VIX analysis to test the revised Hypothesis 2.
    
    Parameters:
    event_file (str): Path to event data file (CSV)
    stock_files (list): List of paths to stock data files (Parquet)
    vix_file (str): Path to VIX data file (CSV)
    results_dir (str): Directory to save results
    file_prefix (str): Prefix for saved files
    event_date_col (str): Column name containing event dates
    ticker_col (str): Column name containing ticker symbols
    return_col (str): Column name containing returns
    window_days (int): Days before/after event to include
    delta_days (int): Parameter δ from the paper (post-event rising phase duration)
    
    Returns:
    Dict[str, Any]: Results of the hypothesis test
    """
    try:
        # Load VIX data first
        vix_data = load_and_prepare_vix_data(vix_file)
        if vix_data is None:
            print("Error: Failed to load VIX data.")
            return {"hypothesis_supported": False, "error": "Failed to load VIX data"}
        
        # Initialize data loader
        data_loader = EventDataLoader(
            event_path=event_file,
            stock_paths=stock_files,
            window_days=window_days,
            event_date_col=event_date_col,
            ticker_col=ticker_col
        )
        
        # Initialize event analysis
        feature_engineer = EventFeatureEngineer(prediction_window=3)  # Default value, not used for VIX analysis
        event_analysis = EventAnalysis(data_loader, feature_engineer)
        
        # Load data
        print("Loading and preparing data...")
        event_analysis.data = event_analysis.load_and_prepare_data(run_feature_engineering=False)
        
        if event_analysis.data is None or event_analysis.data.is_empty():
            print("Error: Failed to load data.")
            return {"hypothesis_supported": False, "error": "Failed to load data"}
        
        print(f"Data loaded successfully. Shape: {event_analysis.data.shape}")
        
        # Merge stock data with VIX data
        event_analysis.data = merge_stock_data_with_vix(event_analysis.data, vix_data)
        
        # Check if VIX column exists after merge
        if 'vix' not in event_analysis.data.columns:
            print("Error: VIX column not available after merge.")
            return {"hypothesis_supported": False, "error": "VIX column not available"}
        
        # Initialize RefinedVIXAnalysis
        refined_vix_analysis = RefinedVIXAnalysis(event_analysis)
        
        # Run refined hypothesis test
        print(f"\nRunning Refined Hypothesis 2 test with delta = {delta_days} days...")
        results = refined_vix_analysis.run_refined_hypothesis_2_test(
            vix_col='vix',  # Use standardized column name
            return_col=return_col,
            delta_days=delta_days,
            results_dir=results_dir,
            file_prefix=file_prefix
        )
        
        print("\n=== Refined Hypothesis 2 Test Completed ===")
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
    # VIX data file
    VIX_FILE = "/home/d87016661/VIX_History.csv"
    
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
    FDA_RESULTS_DIR = "results/refined_vix_analysis/fda"
    FDA_FILE_PREFIX = "fda"
    FDA_EVENT_DATE_COL = "Approval Date"
    FDA_TICKER_COL = "ticker"
    
    # Earnings event specific parameters
    EARNINGS_EVENT_FILE = "/home/d87016661/detail_history_actuals.csv"
    EARNINGS_STOCK_FILES = FDA_STOCK_FILES  # Using the same stock files
    EARNINGS_RESULTS_DIR = "results/refined_vix_analysis/earnings"
    EARNINGS_FILE_PREFIX = "earnings"
    EARNINGS_EVENT_DATE_COL = "ANNDATS"
    EARNINGS_TICKER_COL = "ticker"
    
    # Analysis parameters
    WINDOW_DAYS = 60
    RETURN_COL = "ret"
    DELTA_DAYS = 10  # δ parameter from the paper
    
    # Create results directories
    os.makedirs(FDA_RESULTS_DIR, exist_ok=True)
    os.makedirs(EARNINGS_RESULTS_DIR, exist_ok=True)
    
    # Run FDA analysis
    print("\n=== Starting FDA Approval Event VIX Analysis (Refined) ===")
    fda_results = run_refined_vix_analysis(
        event_file=FDA_EVENT_FILE,
        stock_files=FDA_STOCK_FILES,
        vix_file=VIX_FILE,
        results_dir=FDA_RESULTS_DIR,
        file_prefix=FDA_FILE_PREFIX,
        event_date_col=FDA_EVENT_DATE_COL,
        ticker_col=FDA_TICKER_COL,
        return_col=RETURN_COL,
        window_days=WINDOW_DAYS,
        delta_days=DELTA_DAYS
    )
    
    # Run Earnings analysis
    print("\n=== Starting Earnings Announcement Event VIX Analysis (Refined) ===")
    earnings_results = run_refined_vix_analysis(
        event_file=EARNINGS_EVENT_FILE,
        stock_files=EARNINGS_STOCK_FILES,
        vix_file=VIX_FILE,
        results_dir=EARNINGS_RESULTS_DIR,
        file_prefix=EARNINGS_FILE_PREFIX,
        event_date_col=EARNINGS_EVENT_DATE_COL,
        ticker_col=EARNINGS_TICKER_COL,
        return_col=RETURN_COL,
        window_days=WINDOW_DAYS,
        delta_days=DELTA_DAYS
    )
    
    # Compare and summarize results
    print("\n=== Refined Hypothesis 2 Test Results Summary ===")
    print(f"FDA Approval Events: Hypothesis Supported = {fda_results.get('hypothesis_supported', False)}")
    print(f"Earnings Announcement Events: Hypothesis Supported = {earnings_results.get('hypothesis_supported', False)}")

if __name__ == "__main__":
    main()