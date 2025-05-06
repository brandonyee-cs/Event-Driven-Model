import pandas as pd
import numpy as np
import os
import sys
import plotly.graph_objects as go
import plotly.express as px
import traceback
import polars as pl

# --- Configuration ---
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path: sys.path.append(current_dir)
try: 
    from src.event_processor import EventDataLoader, EventFeatureEngineer, EventAnalysis
    print("Successfully imported Event processor classes.")
except ImportError as e: 
    print(f"Error importing from event_processor: {e}")
    print("Ensure 'event_processor.py' and 'models.py' are in the same directory or Python path.")
    print("Ensure Polars and Plotly are installed: pip install polars pyarrow plotly")
    sys.exit(1)

pl.Config.set_engine_affinity(engine="streaming")

# --- Hardcoded Analysis Parameters ---
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

# FDA event specific parameters
FDA_EVENT_FILE = "/home/d87016661/fda_ticker_list_2000_to_2024.csv"
FDA_RESULTS_DIR = "results/rvr_improved/results_fda/"
FDA_FILE_PREFIX = "fda"
FDA_EVENT_DATE_COL = "Approval Date"
FDA_TICKER_COL = "ticker"

# Earnings event specific parameters
EARNINGS_EVENT_FILE = "/home/d87016661/detail_history_actuals.csv"
EARNINGS_RESULTS_DIR = "results/rvr_improved/results_earnings/"
EARNINGS_FILE_PREFIX = "earnings"
EARNINGS_EVENT_DATE_COL = "ANNDATS"
EARNINGS_TICKER_COL = "ticker"

# Shared analysis parameters
WINDOW_DAYS = 60

# RVR analysis parameters
RVR_ANALYSIS_WINDOW = (-30, 30)
RVR_POST_EVENT_DELTA = 10
RVR_LOOKBACK_WINDOW = 5
RVR_OPTIMISTIC_BIAS = 0.01
RVR_MIN_PERIODS = 3
RVR_VARIANCE_FLOOR = 1e-6  # Added minimum variance to prevent division issues
RVR_CLIP_THRESHOLD = 1e5   # Added clipping threshold for extreme RVR values

def run_fda_analysis():
    """
    Runs the FDA event RVR analysis pipeline using parameters from config.
    """
    print("\n=== Starting FDA Approval Event RVR Analysis with Improved Method ===")

    # --- Path Validation & Results Dir Creation ---
    if not os.path.exists(FDA_EVENT_FILE): 
        print(f"\n*** Error: FDA event file not found: {FDA_EVENT_FILE} ***")
        return False
    
    missing_stock = [f for f in STOCK_FILES if not os.path.exists(f)]
    if missing_stock: 
        print(f"\n*** Error: Stock file(s) not found: {missing_stock} ***")
        return False
    
    print("FDA file paths validated.")
    try:
        os.makedirs(FDA_RESULTS_DIR, exist_ok=True)
        print(f"FDA results will be saved to: {os.path.abspath(FDA_RESULTS_DIR)}")
    except OSError as oe:
        print(f"\n*** Error creating FDA results directory '{FDA_RESULTS_DIR}': {oe} ***")
        return False

    try:
        # --- Initialize Components ---
        print("\nInitializing FDA components...")
        data_loader = EventDataLoader(
            event_path=FDA_EVENT_FILE, 
            stock_paths=STOCK_FILES, 
            window_days=WINDOW_DAYS,
            event_date_col=FDA_EVENT_DATE_COL,
            ticker_col=FDA_TICKER_COL
        )
        feature_engineer = EventFeatureEngineer(prediction_window=3)
        analyzer = EventAnalysis(data_loader, feature_engineer)
        print("FDA components initialized.")

        # --- Load Data ---
        print("\nLoading and preparing FDA data...")
        analyzer.data = analyzer.load_and_prepare_data(run_feature_engineering=False)
        if analyzer.data is None or analyzer.data.is_empty(): 
            print("\n*** Error: FDA data loading failed. ***")
            return False
            
        print(f"FDA data loaded successfully. Shape: {analyzer.data.shape}")
        
        # --- Run RVR Analysis with improved parameters ---
        analyzer.analyze_rvr(
            results_dir=FDA_RESULTS_DIR,
            file_prefix=FDA_FILE_PREFIX,
            return_col='ret',
            analysis_window=RVR_ANALYSIS_WINDOW,
            post_event_delta=RVR_POST_EVENT_DELTA,
            lookback_window=RVR_LOOKBACK_WINDOW,
            optimistic_bias=RVR_OPTIMISTIC_BIAS,
            min_periods=RVR_MIN_PERIODS,
            variance_floor=RVR_VARIANCE_FLOOR,
            rvr_clip_threshold=RVR_CLIP_THRESHOLD,
            adaptive_threshold=True  # Use adaptive thresholding
        )
        
        print(f"\n--- FDA Event RVR Analysis Finished (Results in '{FDA_RESULTS_DIR}') ---")
        return True

    except ValueError as ve: 
        print(f"\n*** FDA ValueError: {ve} ***")
        traceback.print_exc()
    except RuntimeError as re: 
        print(f"\n*** FDA RuntimeError: {re} ***")
        traceback.print_exc()
    except FileNotFoundError as fnf: 
        print(f"\n*** FDA FileNotFoundError: {fnf} ***")
    except pl.exceptions.PolarsError as pe: 
        print(f"\n*** FDA PolarsError: {pe} ***")
        traceback.print_exc()
    except Exception as e: 
        print(f"\n*** An unexpected error occurred in FDA analysis: {e} ***")
        traceback.print_exc()
    
    return False

def run_earnings_analysis():
    """
    Runs the earnings event RVR analysis pipeline using parameters from config.
    """
    print("\n=== Starting Earnings Announcement Event RVR Analysis with Improved Method ===")

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
        feature_engineer = EventFeatureEngineer(prediction_window=3)
        analyzer = EventAnalysis(data_loader, feature_engineer)
        print("Earnings components initialized.")

        # --- Load Data ---
        print("\nLoading and preparing earnings data...")
        analyzer.data = analyzer.load_and_prepare_data(run_feature_engineering=False)
        if analyzer.data is None or analyzer.data.is_empty(): 
            print("\n*** Error: Earnings data loading failed. ***")
            return False
            
        print(f"Earnings data loaded successfully. Shape: {analyzer.data.shape}")
        
        # --- Run RVR Analysis with improved parameters ---
        analyzer.analyze_rvr(
            results_dir=EARNINGS_RESULTS_DIR,
            file_prefix=EARNINGS_FILE_PREFIX,
            return_col='ret',
            analysis_window=RVR_ANALYSIS_WINDOW,
            post_event_delta=RVR_POST_EVENT_DELTA,
            lookback_window=RVR_LOOKBACK_WINDOW,
            optimistic_bias=RVR_OPTIMISTIC_BIAS,
            min_periods=RVR_MIN_PERIODS,
            variance_floor=RVR_VARIANCE_FLOOR,
            rvr_clip_threshold=RVR_CLIP_THRESHOLD,
            adaptive_threshold=True  # Use adaptive thresholding
        )
        
        print(f"\n--- Earnings Event RVR Analysis Finished (Results in '{EARNINGS_RESULTS_DIR}') ---")
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

def compare_results():
    """
    Creates a comparison visualization between the original and improved RVR methods
    for both FDA and earnings events.
    """
    print("\n=== Comparing Original vs. Improved RVR Results ===")
    
    try:
        # Define file paths
        original_fda_file = "results/obs/results_fda/fda_rvr_daily.csv"
        improved_fda_file = f"{FDA_RESULTS_DIR}/fda_rvr_daily.csv"
        original_earnings_file = "results/obs/results_earnings/earnings_rvr_daily.csv"
        improved_earnings_file = f"{EARNINGS_RESULTS_DIR}/earnings_rvr_daily.csv"
        
        # Check if all files exist
        missing_files = []
        for filepath in [original_fda_file, improved_fda_file, original_earnings_file, improved_earnings_file]:
            if not os.path.exists(filepath):
                missing_files.append(filepath)
        
        if missing_files:
            print(f"Error: The following files are missing: {missing_files}")
            print("Run both original and improved analyses first.")
            return False
        
        # Create results directory for comparison
        comparison_dir = "results/rvr_comparison/"
        os.makedirs(comparison_dir, exist_ok=True)
        
        # Load all data
        try:
            original_fda = pd.read_csv(original_fda_file)
            improved_fda = pd.read_csv(improved_fda_file)
            original_earnings = pd.read_csv(original_earnings_file)
            improved_earnings = pd.read_csv(improved_earnings_file)
            
            print(f"Successfully loaded all RVR data files.")
        except Exception as e:
            print(f"Error loading CSV files: {e}")
            return False
        
        # FDA Comparison
        fig_fda = go.Figure()
        
        fig_fda.add_trace(go.Scatter(
            x=original_fda['days_to_event'],
            y=original_fda['avg_rvr'],
            mode='lines',
            name='Original Method',
            line=dict(color='red', width=2)
        ))
        
        fig_fda.add_trace(go.Scatter(
            x=improved_fda['days_to_event'],
            y=improved_fda['avg_rvr'],
            mode='lines',
            name='Improved Method',
            line=dict(color='blue', width=2)
        ))
        
        fig_fda.add_vline(x=0, line=dict(color='green', dash='dash'), annotation_text='Event Day')
        
        fig_fda.update_layout(
            title='FDA Events: Original vs. Improved RVR Methods',
            xaxis_title='Days Relative to Event',
            yaxis_title='Average RVR',
            template='plotly_white',
            width=1000,
            height=600
        )
        
        # Save FDA comparison
        fda_comparison_file = os.path.join(comparison_dir, "fda_rvr_comparison.png")
        fig_fda.write_image(fda_comparison_file, format='png', scale=2)
        print(f"Saved FDA RVR comparison to: {fda_comparison_file}")
        
        # Earnings Comparison
        fig_earnings = go.Figure()
        
        fig_earnings.add_trace(go.Scatter(
            x=original_earnings['days_to_event'],
            y=original_earnings['avg_rvr'],
            mode='lines',
            name='Original Method',
            line=dict(color='red', width=2)
        ))
        
        fig_earnings.add_trace(go.Scatter(
            x=improved_earnings['days_to_event'],
            y=improved_earnings['avg_rvr'],
            mode='lines',
            name='Improved Method',
            line=dict(color='blue', width=2)
        ))
        
        fig_earnings.add_vline(x=0, line=dict(color='green', dash='dash'), annotation_text='Event Day')
        
        fig_earnings.update_layout(
            title='Earnings Events: Original vs. Improved RVR Methods',
            xaxis_title='Days Relative to Event',
            yaxis_title='Average RVR',
            template='plotly_white',
            width=1000,
            height=600
        )
        
        # Save Earnings comparison
        earnings_comparison_file = os.path.join(comparison_dir, "earnings_rvr_comparison.png")
        fig_earnings.write_image(earnings_comparison_file, format='png', scale=2)
        print(f"Saved Earnings RVR comparison to: {earnings_comparison_file}")
        
        # Create summary table
        phase_summary = {
            'Event Type': ['FDA (Original)', 'FDA (Improved)', 'Earnings (Original)', 'Earnings (Improved)'],
            'Pre-Event Avg RVR': [0, 0, 0, 0],
            'Post-Event Rising Avg RVR': [0, 0, 0, 0],
            'Late Post-Event Avg RVR': [0, 0, 0, 0]
        }
        
        # Load phase summary data
        try:
            original_fda_phase = pd.read_csv("results/obs/results_fda/fda_rvr_phase_summary.csv")
            improved_fda_phase = pd.read_csv(f"{FDA_RESULTS_DIR}/fda_rvr_phase_summary.csv")
            original_earnings_phase = pd.read_csv("results/obs/results_earnings/earnings_rvr_phase_summary.csv")
            improved_earnings_phase = pd.read_csv(f"{EARNINGS_RESULTS_DIR}/earnings_rvr_phase_summary.csv")
            
            # Extract values
            for i, df in enumerate([original_fda_phase, improved_fda_phase, original_earnings_phase, improved_earnings_phase]):
                for j, phase in enumerate(['pre_event', 'post_event_rising', 'late_post_event']):
                    row = df[df['phase'] == phase]
                    if not row.empty:
                        col_name = list(phase_summary.keys())[j+1]
                        phase_summary[col_name][i] = row['avg_rvr'].values[0]
            
            # Create summary DataFrame
            summary_df = pd.DataFrame(phase_summary)
            
            # Save summary table
            summary_file = os.path.join(comparison_dir, "rvr_phase_comparison.csv")
            summary_df.to_csv(summary_file, index=False)
            print(f"Saved phase summary comparison to: {summary_file}")
            
            # Print summary
            print("\nRVR Phase Comparison Summary:")
            print(summary_df.to_string(index=False))
            
            return True
        
        except Exception as e:
            print(f"Error creating phase summary: {e}")
            traceback.print_exc()
            return False
        
    except Exception as e:
        print(f"Error comparing results: {e}")
        traceback.print_exc()
        return False

def main():
    # Run FDA RVR analysis with improved method
    fda_success = run_fda_analysis()
    
    # Run earnings RVR analysis with improved method
    earnings_success = run_earnings_analysis()
    
    # Compare results if both analyses succeeded
    if fda_success and earnings_success:
        compare_success = compare_results()
        if compare_success:
            print("\n=== All analyses and comparisons completed successfully ===")
        else:
            print("\n=== Analyses completed, but comparison failed ===")
    elif fda_success:
        print("\n=== Only FDA analysis completed successfully ===")
    elif earnings_success:
        print("\n=== Only earnings analysis completed successfully ===")
    else:
        print("\n=== Both analyses failed ===")

if __name__ == "__main__":
    main()