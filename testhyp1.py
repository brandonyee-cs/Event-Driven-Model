# runhypothesis1.py

import pandas as pd
import numpy as np
import os
import sys
import traceback
import polars as pl
from typing import List, Tuple

# --- Configuration ---
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path: sys.path.append(current_dir)
try: 
    from src.event_processor import EventDataLoader, EventFeatureEngineer, EventAnalysis
    from src.models import GARCHModel, GJRGARCHModel, ThreePhaseVolatilityModel
    print("Successfully imported Event processor classes and models.")
except ImportError as e: 
    print(f"Error importing modules: {e}")
    print("Ensure 'event_processor.py' and 'models.py' are in the same directory or Python path.")
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
FDA_RESULTS_DIR = "results/hypothesis1/results_fda/"
FDA_FILE_PREFIX = "fda"
FDA_EVENT_DATE_COL = "Approval Date"
FDA_TICKER_COL = "ticker"

# Earnings event specific parameters
EARNINGS_EVENT_FILE = "/home/d87016661/detail_history_actuals.csv"
EARNINGS_RESULTS_DIR = "results/hypothesis1/results_earnings/"
EARNINGS_FILE_PREFIX = "earnings"
EARNINGS_EVENT_DATE_COL = "ANNDATS"
EARNINGS_TICKER_COL = "ticker"

# Shared analysis parameters
WINDOW_DAYS = 60
ANALYSIS_WINDOW = (-30, 30)

# Volatility model parameters
GARCH_TYPE = 'gjr'  # 'garch' or 'gjr'
K1 = 1.5  # Pre-event volatility multiplier
K2 = 2.0  # Post-event volatility multiplier
DELTA_T1 = 5.0  # Pre-event volatility duration parameter
DELTA_T2 = 3.0  # Post-event rising phase rate parameter
DELTA_T3 = 10.0  # Post-event decay rate parameter
DELTA = 5  # Duration of post-event rising phase in days

# RVR parameters
OPTIMISTIC_BIAS = 0.01  # Bias parameter for post-event expected returns (as decimal)
RISK_FREE_RATE = 0.0  # Daily risk-free rate

def run_fda_analysis():
    """
    Runs the FDA event analysis to test Hypothesis 1.
    """
    print("\n=== Starting FDA Approval Event Analysis for Hypothesis 1 ===")

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
        
        # --- Run Three-Phase Volatility Analysis ---
        analyzer.analyze_three_phase_volatility(
            results_dir=FDA_RESULTS_DIR,
            file_prefix=FDA_FILE_PREFIX,
            return_col='ret',
            analysis_window=ANALYSIS_WINDOW,
            garch_type=GARCH_TYPE,
            k1=K1, k2=K2,
            delta_t1=DELTA_T1, delta_t2=DELTA_T2, delta_t3=DELTA_T3,
            delta=DELTA
        )
        
        # --- Run RVR Analysis with Optimistic Bias (Hypothesis 1) ---
        analyzer.analyze_rvr_with_optimistic_bias(
            results_dir=FDA_RESULTS_DIR,
            file_prefix=FDA_FILE_PREFIX,
            return_col='ret',
            analysis_window=ANALYSIS_WINDOW,
            garch_type=GARCH_TYPE,
            k1=K1, k2=K2,
            delta_t1=DELTA_T1, delta_t2=DELTA_T2, delta_t3=DELTA_T3,
            delta=DELTA,
            optimistic_bias=OPTIMISTIC_BIAS,
            risk_free_rate=RISK_FREE_RATE
        )
        
        print(f"\n--- FDA Event Analysis for Hypothesis 1 Finished (Results in '{FDA_RESULTS_DIR}') ---")
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
    Runs the earnings event analysis to test Hypothesis 1.
    """
    print("\n=== Starting Earnings Announcement Event Analysis for Hypothesis 1 ===")

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
        
        # --- Run Three-Phase Volatility Analysis ---
        analyzer.analyze_three_phase_volatility(
            results_dir=EARNINGS_RESULTS_DIR,
            file_prefix=EARNINGS_FILE_PREFIX,
            return_col='ret',
            analysis_window=ANALYSIS_WINDOW,
            garch_type=GARCH_TYPE,
            k1=K1, k2=K2,
            delta_t1=DELTA_T1, delta_t2=DELTA_T2, delta_t3=DELTA_T3,
            delta=DELTA
        )
        
        # --- Run RVR Analysis with Optimistic Bias (Hypothesis 1) ---
        analyzer.analyze_rvr_with_optimistic_bias(
            results_dir=EARNINGS_RESULTS_DIR,
            file_prefix=EARNINGS_FILE_PREFIX,
            return_col='ret',
            analysis_window=ANALYSIS_WINDOW,
            garch_type=GARCH_TYPE,
            k1=K1, k2=K2,
            delta_t1=DELTA_T1, delta_t2=DELTA_T2, delta_t3=DELTA_T3,
            delta=DELTA,
            optimistic_bias=OPTIMISTIC_BIAS,
            risk_free_rate=RISK_FREE_RATE
        )
        
        print(f"\n--- Earnings Event Analysis for Hypothesis 1 Finished (Results in '{EARNINGS_RESULTS_DIR}') ---")
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
    Compares the hypothesis test results between FDA and earnings events.
    """
    print("\n=== Comparing FDA and Earnings Results for Hypothesis 1 ===")
    
    # Define file paths
    fda_rvr_file = os.path.join(FDA_RESULTS_DIR, f"{FDA_FILE_PREFIX}_rvr_phase_stats.csv")
    earnings_rvr_file = os.path.join(EARNINGS_RESULTS_DIR, f"{EARNINGS_FILE_PREFIX}_rvr_phase_stats.csv")
    fda_h1_file = os.path.join(FDA_RESULTS_DIR, f"{FDA_FILE_PREFIX}_hypothesis1_test.csv")
    earnings_h1_file = os.path.join(EARNINGS_RESULTS_DIR, f"{EARNINGS_FILE_PREFIX}_hypothesis1_test.csv")
    
    # Check if files exist
    missing_files = []
    for file_path in [fda_rvr_file, earnings_rvr_file, fda_h1_file, earnings_h1_file]:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        print(f"Error: The following files are missing: {missing_files}")
        return False
    
    # Create comparison directory
    comparison_dir = "results/hypothesis1/comparison/"
    os.makedirs(comparison_dir, exist_ok=True)
    
    try:
        # Load RVR phase stats
        fda_rvr = pl.read_csv(fda_rvr_file)
        earnings_rvr = pl.read_csv(earnings_rvr_file)
        
        # Load hypothesis test results
        fda_h1 = pl.read_csv(fda_h1_file)
        earnings_h1 = pl.read_csv(earnings_h1_file)
        
        print("Successfully loaded result files.")
        
        # Create comparison table
        comparison_data = {
            'Event Type': ['FDA Approvals', 'Earnings Announcements'],
            'Hypothesis 1 Supported': [
                fda_h1['result'].item(),
                earnings_h1['result'].item()
            ]
        }
        
        # Add RVR by phase
        phases = ['pre_event', 'post_event_rising', 'post_event_decay']
        
        for phase in phases:
            fda_phase = fda_rvr.filter(pl.col('phase') == phase)
            earnings_phase = earnings_rvr.filter(pl.col('phase') == phase)
            
            if not fda_phase.is_empty() and not earnings_phase.is_empty():
                comparison_data[f'{phase}_rvr_fda'] = [fda_phase['avg_rvr'].item(), None]
                comparison_data[f'{phase}_rvr_earnings'] = [None, earnings_phase['avg_rvr'].item()]
        
        comparison_df = pl.DataFrame(comparison_data)
        
        # Save comparison table
        comparison_df.write_csv(os.path.join(comparison_dir, "hypothesis1_comparison.csv"))
        
        # Print comparison results
        print("\nHypothesis 1 Comparison Results:")
        print(f"FDA Approvals: H1 {'Supported' if fda_h1['result'].item() else 'Not Supported'}")
        print(f"Earnings Announcements: H1 {'Supported' if earnings_h1['result'].item() else 'Not Supported'}")
        
        # Create phases comparison for plotting
        phases_data = []
        
        for phase in phases:
            fda_phase = fda_rvr.filter(pl.col('phase') == phase)
            earnings_phase = earnings_rvr.filter(pl.col('phase') == phase)
            
            if not fda_phase.is_empty() and not earnings_phase.is_empty():
                phases_data.append({
                    'phase': phase,
                    'fda_rvr': fda_phase['avg_rvr'].item(),
                    'earnings_rvr': earnings_phase['avg_rvr'].item()
                })
        
        phases_df = pl.DataFrame(phases_data)
        phases_df.write_csv(os.path.join(comparison_dir, "phases_comparison.csv"))
        
        # Generate comparison plot
        try:
            import matplotlib.pyplot as plt
            
            # Bar chart comparing FDA and Earnings RVR by phase
            phases_pd = phases_df.to_pandas()
            
            fig, ax = plt.subplots(figsize=(10, 6))
            
            x = np.arange(len(phases))
            width = 0.35
            
            rects1 = ax.bar(x - width/2, phases_pd['fda_rvr'], width, label='FDA Approvals', color='blue', alpha=0.7)
            rects2 = ax.bar(x + width/2, phases_pd['earnings_rvr'], width, label='Earnings Announcements', color='red', alpha=0.7)
            
            ax.set_title('Return-to-Variance Ratio by Phase: FDA vs Earnings')
            ax.set_xticks(x)
            ax.set_xticklabels([phase.replace('_', ' ').title() for phase in phases])
            ax.set_ylabel('Average RVR')
            ax.legend()
            
            # Add value labels
            for i, rect in enumerate(rects1):
                height = rect.get_height()
                ax.annotate(f"{height:.4f}",
                            xy=(rect.get_x() + rect.get_width()/2, height),
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom')
            
            for i, rect in enumerate(rects2):
                height = rect.get_height()
                ax.annotate(f"{height:.4f}",
                            xy=(rect.get_x() + rect.get_width()/2, height),
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom')
            
            # Highlight the post-event rising phase
            ax.axvspan(0.5, 1.5, color='yellow', alpha=0.2, label='Key Phase for H1')
            
            # Add hypothesis test results
            result_text = (f"FDA: H1 {'SUPPORTED' if fda_h1['result'].item() else 'NOT SUPPORTED'}\n"
                           f"Earnings: H1 {'SUPPORTED' if earnings_h1['result'].item() else 'NOT SUPPORTED'}")
            ax.text(0.5, -0.15, result_text, ha='center', transform=ax.transAxes, 
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", alpha=0.8),
                    fontweight='bold')
            
            plt.tight_layout()
            plt.savefig(os.path.join(comparison_dir, "hypothesis1_comparison.png"), dpi=200)
            plt.close()
            
            print(f"Saved comparison plot to: {os.path.join(comparison_dir, 'hypothesis1_comparison.png')}")
        
        except Exception as e:
            print(f"Error creating comparison plot: {e}")
            traceback.print_exc()
        
        return True
    
    except Exception as e:
        print(f"Error comparing results: {e}")
        traceback.print_exc()
        return False

def main():
    # Run FDA analysis
    fda_success = run_fda_analysis()
    
    # Run earnings analysis
    earnings_success = run_earnings_analysis()
    
    # Compare results if both analyses succeeded
    #if fda_success and earnings_success:
    #    compare_success = compare_results()
    #    if compare_success:
    #        print("\n=== All analyses and comparisons completed successfully ===")
    #    else:
    #        print("\n=== Analyses completed, but comparison failed ===")
    #elif fda_success:
    #    print("\n=== Only FDA analysis completed successfully ===")
    #elif earnings_success:
    #    print("\n=== Only earnings analysis completed successfully ===")
    #else:
    #    print("\n=== Both analyses failed ===")

if __name__ == "__main__":
    main()