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
WINDOW_DAYS = 60 # For EventDataLoader
ANALYSIS_WINDOW = (-30, 30) # For EventAnalysis methods

# Volatility model parameters (from paper Assumption 4 and for H1)
GARCH_TYPE = 'gjr'  # 'garch' or 'gjr' as baseline
K1 = 1.5  # Pre-event volatility multiplier
K2 = 2.0  # Post-event volatility multiplier
DELTA_T1 = 5.0  # Pre-event volatility duration parameter
DELTA_T2 = 3.0  # Post-event rising phase rate parameter
DELTA_T3 = 10.0  # Post-event decay rate parameter
DELTA = 5  # Duration of post-event rising phase in days (t_event+1 to t_event+delta)

# RVR parameters (from paper for H1)
OPTIMISTIC_BIAS = 0.01  # Bias parameter for post-event expected returns (decimal form)
RISK_FREE_RATE = 0.0  # Daily risk-free rate (decimal form, e.g., 0.02/252 for 2% annual)

def run_fda_analysis():
    """
    Runs the FDA event analysis to test Hypothesis 1.
    """
    print("\n=== Starting FDA Approval Event Analysis for Hypothesis 1 ===")

    if not os.path.exists(FDA_EVENT_FILE): 
        print(f"\n*** Error: FDA event file not found: {FDA_EVENT_FILE} ***"); return False
    missing_stock = [f for f in STOCK_FILES if not os.path.exists(f)]
    if missing_stock: 
        print(f"\n*** Error: Stock file(s) not found: {missing_stock} ***"); return False
    
    # print("FDA file paths validated.") # Less verbose
    try:
        os.makedirs(FDA_RESULTS_DIR, exist_ok=True)
        # print(f"FDA results will be saved to: {os.path.abspath(FDA_RESULTS_DIR)}") # Less verbose
    except OSError as oe:
        print(f"\n*** Error creating FDA results directory '{FDA_RESULTS_DIR}': {oe} ***"); return False

    try:
        # print("\nInitializing FDA components...") # Less verbose
        data_loader = EventDataLoader(
            event_path=FDA_EVENT_FILE, stock_paths=STOCK_FILES, window_days=WINDOW_DAYS,
            event_date_col=FDA_EVENT_DATE_COL, ticker_col=FDA_TICKER_COL
        )
        feature_engineer = EventFeatureEngineer(prediction_window=3) # Window not critical for H1
        analyzer = EventAnalysis(data_loader, feature_engineer)
        # print("FDA components initialized.") # Less verbose

        # print("\nLoading and preparing FDA data...") # Less verbose
        analyzer.data = analyzer.load_and_prepare_data(run_feature_engineering=False) # H1 doesn't need ML features
        if analyzer.data is None or analyzer.data.is_empty(): 
            print("\n*** Error: FDA data loading failed. ***"); return False
        # print(f"FDA data loaded successfully. Shape: {analyzer.data.shape}") # Less verbose
        
        # print("\nRunning Three-Phase Volatility Analysis for FDA events...") # Less verbose
        analyzer.analyze_three_phase_volatility(
            results_dir=FDA_RESULTS_DIR, file_prefix=FDA_FILE_PREFIX, return_col='ret',
            analysis_window=ANALYSIS_WINDOW, garch_type=GARCH_TYPE,
            k1=K1, k2=K2, delta_t1=DELTA_T1, delta_t2=DELTA_T2, delta_t3=DELTA_T3, delta=DELTA
        )
        
        # print("\nRunning RVR Analysis with Optimistic Bias (Hypothesis 1) for FDA events...") # Less verbose
        analyzer.analyze_rvr_with_optimistic_bias(
            results_dir=FDA_RESULTS_DIR, file_prefix=FDA_FILE_PREFIX, return_col='ret',
            analysis_window=ANALYSIS_WINDOW, garch_type=GARCH_TYPE,
            k1=K1, k2=K2, delta_t1=DELTA_T1, delta_t2=DELTA_T2, delta_t3=DELTA_T3, delta=DELTA,
            optimistic_bias=OPTIMISTIC_BIAS, risk_free_rate=RISK_FREE_RATE
        )
        
        print(f"\n--- FDA Event Analysis for Hypothesis 1 Finished (Results in '{FDA_RESULTS_DIR}') ---")
        return True

    except Exception as e: 
        print(f"\n*** An unexpected error occurred in FDA analysis: {e} ***")
        traceback.print_exc()
    return False

def run_earnings_analysis():
    """
    Runs the earnings event analysis to test Hypothesis 1.
    """
    print("\n=== Starting Earnings Announcement Event Analysis for Hypothesis 1 ===")

    if not os.path.exists(EARNINGS_EVENT_FILE): 
        print(f"\n*** Error: Earnings event file not found: {EARNINGS_EVENT_FILE} ***"); return False
    missing_stock = [f for f in STOCK_FILES if not os.path.exists(f)]
    if missing_stock: 
        print(f"\n*** Error: Stock file(s) not found: {missing_stock} ***"); return False
    
    # print("Earnings file paths validated.") # Less verbose
    try:
        os.makedirs(EARNINGS_RESULTS_DIR, exist_ok=True)
        # print(f"Earnings results will be saved to: {os.path.abspath(EARNINGS_RESULTS_DIR)}") # Less verbose
    except OSError as oe:
        print(f"\n*** Error creating earnings results directory '{EARNINGS_RESULTS_DIR}': {oe} ***"); return False

    try:
        # print("\nInitializing earnings components...") # Less verbose
        data_loader = EventDataLoader(
            event_path=EARNINGS_EVENT_FILE, stock_paths=STOCK_FILES, window_days=WINDOW_DAYS,
            event_date_col=EARNINGS_EVENT_DATE_COL, ticker_col=EARNINGS_TICKER_COL
        )
        feature_engineer = EventFeatureEngineer(prediction_window=3)
        analyzer = EventAnalysis(data_loader, feature_engineer)
        # print("Earnings components initialized.") # Less verbose

        # print("\nLoading and preparing earnings data...") # Less verbose
        analyzer.data = analyzer.load_and_prepare_data(run_feature_engineering=False)
        if analyzer.data is None or analyzer.data.is_empty(): 
            print("\n*** Error: Earnings data loading failed. ***"); return False
        # print(f"Earnings data loaded successfully. Shape: {analyzer.data.shape}") # Less verbose
        
        # print("\nRunning Three-Phase Volatility Analysis for Earnings events...") # Less verbose
        analyzer.analyze_three_phase_volatility(
            results_dir=EARNINGS_RESULTS_DIR, file_prefix=EARNINGS_FILE_PREFIX, return_col='ret',
            analysis_window=ANALYSIS_WINDOW, garch_type=GARCH_TYPE,
            k1=K1, k2=K2, delta_t1=DELTA_T1, delta_t2=DELTA_T2, delta_t3=DELTA_T3, delta=DELTA
        )
        
        # print("\nRunning RVR Analysis with Optimistic Bias (Hypothesis 1) for Earnings events...") # Less verbose
        analyzer.analyze_rvr_with_optimistic_bias(
            results_dir=EARNINGS_RESULTS_DIR, file_prefix=EARNINGS_FILE_PREFIX, return_col='ret',
            analysis_window=ANALYSIS_WINDOW, garch_type=GARCH_TYPE,
            k1=K1, k2=K2, delta_t1=DELTA_T1, delta_t2=DELTA_T2, delta_t3=DELTA_T3, delta=DELTA,
            optimistic_bias=OPTIMISTIC_BIAS, risk_free_rate=RISK_FREE_RATE
        )
        
        print(f"\n--- Earnings Event Analysis for Hypothesis 1 Finished (Results in '{EARNINGS_RESULTS_DIR}') ---")
        return True

    except Exception as e: 
        print(f"\n*** An unexpected error occurred in earnings analysis: {e} ***")
        traceback.print_exc()
    return False

def compare_results():
    """
    Compares the hypothesis test results between FDA and earnings events for Hypothesis 1.
    """
    print("\n=== Comparing FDA and Earnings Results for Hypothesis 1 ===")
    
    comparison_dir = "results/hypothesis1/comparison/"
    os.makedirs(comparison_dir, exist_ok=True)
    
    fda_h1_file = os.path.join(FDA_RESULTS_DIR, f"{FDA_FILE_PREFIX}_hypothesis1_test.csv")
    earnings_h1_file = os.path.join(EARNINGS_RESULTS_DIR, f"{EARNINGS_FILE_PREFIX}_hypothesis1_test.csv")
    
    missing_files = [fp for fp in [fda_h1_file, earnings_h1_file] if not os.path.exists(fp)]
    if missing_files:
        print(f"Error: The following H1 result files are missing: {missing_files}"); return False
    
    try:
        fda_h1_df = pl.read_csv(fda_h1_file)
        earnings_h1_df = pl.read_csv(earnings_h1_file)
        
        # print("Successfully loaded H1 result files.") # Less verbose
        
        comparison_summary = pl.DataFrame({
            "Event Type": ["FDA Approvals", "Earnings Announcements"],
            "Hypothesis 1 Supported": [fda_h1_df["result"][0], earnings_h1_df["result"][0]],
            "Pre-Event RVR": [fda_h1_df["pre_event_rvr"][0], earnings_h1_df["pre_event_rvr"][0]],
            "Post-Rising RVR": [fda_h1_df["post_rising_rvr"][0], earnings_h1_df["post_rising_rvr"][0]],
            "Post-Decay RVR": [fda_h1_df["post_decay_rvr"][0], earnings_h1_df["post_decay_rvr"][0]],
        })
        
        comparison_summary.write_csv(os.path.join(comparison_dir, "hypothesis1_overall_comparison.csv"))
        print("\nHypothesis 1 Comparison Summary:")
        print(comparison_summary)

        # Create plot comparing RVR phases for both event types
        phases_plot_data = []
        for event_type, df in [("FDA Approvals", fda_h1_df), ("Earnings Announcements", earnings_h1_df)]:
            phases_plot_data.append({'event_type': event_type, 'phase': 'Pre-Event', 'rvr': df['pre_event_rvr'][0]})
            phases_plot_data.append({'event_type': event_type, 'phase': 'Post-Event Rising', 'rvr': df['post_rising_rvr'][0]})
            phases_plot_data.append({'event_type': event_type, 'phase': 'Post-Event Decay', 'rvr': df['post_decay_rvr'][0]})
        
        phases_plot_df = pl.from_dicts(phases_plot_data)
        
        try:
            import matplotlib.pyplot as plt
            phases_pd = phases_plot_df.to_pandas()
            
            fig, ax = plt.subplots(figsize=(12, 7)) # Increased figure size slightly
            
            event_types_unique = phases_pd['event_type'].unique()
            phases_order = ['Pre-Event', 'Post-Event Rising', 'Post-Event Decay']
            num_phases = len(phases_order)
            x_indices = np.arange(num_phases) # x locations for the groups
            bar_width = 0.35 # width of the bars
            
            # Filter data for each event type and plot
            data_fda = phases_pd[phases_pd['event_type'] == event_types_unique[0]].set_index('phase').reindex(phases_order)['rvr']
            data_earnings = phases_pd[phases_pd['event_type'] == event_types_unique[1]].set_index('phase').reindex(phases_order)['rvr']

            rects1 = ax.bar(x_indices - bar_width/2, data_fda.fillna(0), bar_width, label=event_types_unique[0], color='deepskyblue', alpha=0.85, edgecolor='black')
            rects2 = ax.bar(x_indices + bar_width/2, data_earnings.fillna(0), bar_width, label=event_types_unique[1], color='salmon', alpha=0.85, edgecolor='black')

            ax.set_ylabel('Average RVR', fontsize=12)
            ax.set_title('Hypothesis 1: RVR by Phase and Event Type', fontsize=14, fontweight='bold')
            ax.set_xticks(x_indices)
            ax.set_xticklabels(phases_order, fontsize=11)
            ax.legend(fontsize=10, loc='upper right')
            ax.grid(True, axis='y', linestyle=':', alpha=0.6)
            ax.axhline(0, color='black', linewidth=0.5) # Add horizontal line at y=0

            def autolabel_bars(rects):
                for rect in rects:
                    height = rect.get_height()
                    ax.annotate(f'{height:.3f}' if pd.notnull(height) else 'N/A',
                                xy=(rect.get_x() + rect.get_width() / 2, height),
                                xytext=(0, 3), textcoords="offset points",
                                ha='center', va='bottom', fontsize=9)
            autolabel_bars(rects1); autolabel_bars(rects2)
            
            # Add text about hypothesis support
            fda_supported_text = "SUPPORTED" if fda_h1_df["result"][0] else "NOT SUPPORTED"
            earn_supported_text = "SUPPORTED" if earnings_h1_df["result"][0] else "NOT SUPPORTED"
            support_text = f"H1 Support: FDA {fda_supported_text} | Earnings {earn_supported_text}"
            fig.text(0.5, 0.01, support_text, ha='center', va='bottom', fontsize=10, 
                     bbox=dict(boxstyle='round,pad=0.3', fc='lightyellow', alpha=0.7))

            plt.tight_layout(rect=[0, 0.05, 1, 0.95]) # Adjust layout to make space for text
            plt.savefig(os.path.join(comparison_dir, "hypothesis1_rvr_phase_comparison.png"), dpi=200)
            plt.close(fig)
            print(f"Saved RVR phase comparison plot to: {os.path.join(comparison_dir, 'hypothesis1_rvr_phase_comparison.png')}")

        except Exception as e:
            print(f"Error creating H1 comparison plot: {e}"); traceback.print_exc()
        return True

    except Exception as e:
        print(f"Error comparing H1 results: {e}"); traceback.print_exc()
        return False

def main():
    fda_success = run_fda_analysis()
    earnings_success = run_earnings_analysis()
    
    if fda_success and earnings_success:
        compare_success = compare_results()
        if compare_success: print("\n=== All H1 analyses and comparisons completed successfully ===")
        else: print("\n=== H1 Analyses completed, but comparison failed ===")
    elif fda_success: print("\n=== Only FDA H1 analysis completed successfully ===")
    elif earnings_success: print("\n=== Only Earnings H1 analysis completed successfully ===")
    else: print("\n=== Both H1 analyses failed ===")

if __name__ == "__main__":
    main()