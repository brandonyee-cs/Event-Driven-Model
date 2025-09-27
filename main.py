#!/usr/bin/env python3
"""
Dynamic Asset Pricing Model for High-Uncertainty Events
Main Entry Point - Direct Execution Script

Based on original_model.py structure, this script runs the complete analysis pipeline
using the VM file paths and generates hypothesis testing results.
"""

import pandas as pd
import numpy as np
import os
import sys
import traceback
from datetime import datetime
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# --- Configuration ---
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path: 
    sys.path.append(current_dir)

try:
    from src.event_processor import EventDataLoader, EventFeatureEngineer, EventAnalysis
    from src.models import GARCHModel, GJRGARCHModel, ThreePhaseVolatilityModel
    print("Successfully imported Event processor classes and models.")
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Ensure 'event_processor.py' and 'models.py' are in the src directory.")
    sys.exit(1)

# --- VM File Paths (from original_model.py) ---
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

FDA_EVENT_FILE = "/home/d87016661/fda_ticker_list_2000_to_2024.csv"
EARNINGS_EVENT_FILE = "/home/d87016661/detail_history_actuals.csv"

# --- Analysis Parameters ---
FDA_RESULTS_DIR = "results/hypothesis1/results_fda/"
FDA_FILE_PREFIX = "fda"
FDA_EVENT_DATE_COL = "Approval Date"
FDA_TICKER_COL = "ticker"

EARNINGS_RESULTS_DIR = "results/hypothesis1/results_earnings/"
EARNINGS_FILE_PREFIX = "earnings"
EARNINGS_EVENT_DATE_COL = "ANNDATS"
EARNINGS_TICKER_COL = "ticker"

WINDOW_DAYS = 30
ANALYSIS_WINDOW = (-15, 15)
GARCH_TYPE = 'gjr'
K1 = 1.5
K2 = 2.0
DELTA_T1 = 5.0
DELTA_T2 = 3.0
DELTA_T3 = 10.0
DELTA = 5
OPTIMISTIC_BIAS = 0.01
RISK_FREE_RATE = 0.0


def generate_h1_summary_report(results_dir: str, file_prefix: str):
    """Generate summary report for Hypothesis 1 analysis."""
    print(f"\n--- Generating Hypothesis 1 Summary Report for {file_prefix.upper()} ---")
    h1_test_file = os.path.join(results_dir, f"{file_prefix}_hypothesis1_test.csv")
    
    if not os.path.exists(h1_test_file):
        print(f"  Warning: H1 test result file not found: {h1_test_file}")
        return

    try:
        import polars as pl
        h1_df = pl.read_csv(h1_test_file)
        if h1_df.is_empty():
            print(f"  Warning: H1 test result file is empty: {h1_test_file}")
            return

        h1_supported = h1_df['result'][0]
        details = (f"Pre-RVR: {h1_df['pre_event_rvr'][0]:.3f}, "
                   f"Rising-RVR: {h1_df['post_rising_rvr'][0]:.3f}, "
                   f"Decay-RVR: {h1_df['post_decay_rvr'][0]:.3f}")

        summary_df = pl.DataFrame({
            'hypothesis': [h1_df['hypothesis'][0]],
            'result': [h1_supported],
            'details': [details]
        })
        summary_df.write_csv(os.path.join(results_dir, f"{file_prefix}_h1_overall_summary.csv"))
        print(f"  Hypothesis 1 for {file_prefix.upper()}: {'SUPPORTED' if h1_supported else 'NOT SUPPORTED'}")
        print(f"  Details: {details}")

        # Generate visualization
        try:
            import matplotlib.pyplot as plt
            phases_data = [
                {'phase': 'Pre-Event', 'rvr': h1_df['pre_event_rvr'][0]},
                {'phase': 'Post-Event Rising', 'rvr': h1_df['post_rising_rvr'][0]},
                {'phase': 'Post-Event Decay', 'rvr': h1_df['post_decay_rvr'][0]}
            ]
            phases_pd = pd.DataFrame(phases_data)
            
            fig, ax = plt.subplots(figsize=(8, 5))
            bars = ax.bar(phases_pd['phase'], phases_pd['rvr'], 
                          color=['lightsteelblue', 'salmon', 'lightgreen'], 
                          alpha=0.8, edgecolor='black')
            
            ax.set_ylabel('Average RVR')
            ax.set_title(f'Hypothesis 1: RVR by Phase - {file_prefix.upper()} Events', fontsize=12)
            ax.grid(True, axis='y', linestyle=':', alpha=0.6)
            ax.axhline(0, color='black', linewidth=0.5)

            for bar in bars:
                height = bar.get_height()
                if pd.notna(height):
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                            f'{height:.3f}', ha='center', va='bottom' if height >= 0 else 'top')
            
            status_text = "SUPPORTED" if h1_supported else "NOT SUPPORTED"
            ax.text(0.5, -0.15, f"Overall H1 Status: {status_text}", 
                    ha='center', transform=ax.transAxes,
                    bbox=dict(boxstyle='round,pad=0.3', fc='lightyellow', alpha=0.8))

            plt.tight_layout(rect=[0, 0.05, 1, 0.95])
            plt.savefig(os.path.join(results_dir, f"{file_prefix}_h1_rvr_phase_plot.png"), dpi=150)
            plt.close(fig)

        except Exception as e:
            print(f"  Error generating H1 plot for {file_prefix}: {e}")

    except Exception as e:
        print(f"  Error generating H1 summary for {file_prefix}: {e}")


def run_fda_analysis():
    """Run FDA approval event analysis."""
    print("\n=== Starting FDA Approval Event Analysis for Hypothesis 1 ===")
    
    # Check file existence
    if not os.path.exists(FDA_EVENT_FILE):
        print(f"\n*** Error: FDA event file not found: {FDA_EVENT_FILE} ***")
        return False
    
    missing_stock = [f for f in STOCK_FILES if not os.path.exists(f)]
    if missing_stock:
        print(f"\n*** Error: Stock file(s) not found: {missing_stock} ***")
        return False
    
    try:
        os.makedirs(FDA_RESULTS_DIR, exist_ok=True)
    except OSError as oe:
        print(f"\n*** Error creating FDA results directory '{FDA_RESULTS_DIR}': {oe} ***")
        return False

    try:
        data_loader = EventDataLoader(
            event_path=FDA_EVENT_FILE, 
            stock_paths=STOCK_FILES, 
            window_days=WINDOW_DAYS,
            event_date_col=FDA_EVENT_DATE_COL, 
            ticker_col=FDA_TICKER_COL
        )
        
        analyzer = EventAnalysis(data_loader, EventFeatureEngineer())
        analyzer.data = analyzer.load_and_prepare_data(run_feature_engineering=False)
        
        if analyzer.data is None or analyzer.data.is_empty():
            print("\n*** Error: FDA data loading failed. ***")
            return False
        
        print(f"FDA data loaded. Shape: {analyzer.data.shape}")
        
        # Run three-phase volatility analysis
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
        
        # Run RVR analysis with optimistic bias
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
        
        generate_h1_summary_report(FDA_RESULTS_DIR, FDA_FILE_PREFIX)
        print(f"\n--- FDA Event Analysis for Hypothesis 1 Finished (Results in '{FDA_RESULTS_DIR}') ---")
        return True
        
    except Exception as e:
        print(f"\n*** An unexpected error occurred in FDA analysis: {e} ***")
        traceback.print_exc()
        return False


def run_earnings_analysis():
    """Run earnings announcement event analysis."""
    print("\n=== Starting Earnings Announcement Event Analysis for Hypothesis 1 ===")
    
    # Check file existence
    if not os.path.exists(EARNINGS_EVENT_FILE):
        print(f"\n*** Error: Earnings event file not found: {EARNINGS_EVENT_FILE} ***")
        return False
    
    missing_stock = [f for f in STOCK_FILES if not os.path.exists(f)]
    if missing_stock:
        print(f"\n*** Error: Stock file(s) not found: {missing_stock} ***")
        return False
    
    try:
        os.makedirs(EARNINGS_RESULTS_DIR, exist_ok=True)
    except OSError as oe:
        print(f"\n*** Error creating earnings results directory '{EARNINGS_RESULTS_DIR}': {oe} ***")
        return False

    try:
        data_loader = EventDataLoader(
            event_path=EARNINGS_EVENT_FILE, 
            stock_paths=STOCK_FILES, 
            window_days=WINDOW_DAYS,
            event_date_col=EARNINGS_EVENT_DATE_COL, 
            ticker_col=EARNINGS_TICKER_COL
        )
        
        analyzer = EventAnalysis(data_loader, EventFeatureEngineer())
        analyzer.data = analyzer.load_and_prepare_data(run_feature_engineering=False)
        
        if analyzer.data is None or analyzer.data.is_empty():
            print("\n*** Error: Earnings data loading failed. ***")
            return False
        
        print(f"Earnings data loaded. Shape: {analyzer.data.shape}")
        
        # Run three-phase volatility analysis
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
        
        # Run RVR analysis with optimistic bias
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
        
        generate_h1_summary_report(EARNINGS_RESULTS_DIR, EARNINGS_FILE_PREFIX)
        print(f"\n--- Earnings Event Analysis for Hypothesis 1 Finished (Results in '{EARNINGS_RESULTS_DIR}') ---")
        return True
        
    except Exception as e:
        print(f"\n*** An unexpected error occurred in earnings analysis: {e} ***")
        traceback.print_exc()
        return False


def compare_results():
    """Compare FDA and Earnings results for Hypothesis 1."""
    print("\n=== Comparing FDA and Earnings Results for Hypothesis 1 ===")
    
    comparison_dir = "results/hypothesis1/comparison/"
    os.makedirs(comparison_dir, exist_ok=True)
    
    # File paths for comparison
    fda_h1_summary_file = os.path.join(FDA_RESULTS_DIR, f"{FDA_FILE_PREFIX}_h1_overall_summary.csv")
    earnings_h1_summary_file = os.path.join(EARNINGS_RESULTS_DIR, f"{EARNINGS_FILE_PREFIX}_h1_overall_summary.csv")
    fda_h1_test_file = os.path.join(FDA_RESULTS_DIR, f"{FDA_FILE_PREFIX}_hypothesis1_test.csv")
    earnings_h1_test_file = os.path.join(EARNINGS_RESULTS_DIR, f"{EARNINGS_FILE_PREFIX}_hypothesis1_test.csv")

    missing_files = [fp for fp in [fda_h1_summary_file, earnings_h1_summary_file, 
                                   fda_h1_test_file, earnings_h1_test_file] 
                     if not os.path.exists(fp)]
    if missing_files:
        print(f"Error: The following H1 result files are missing for comparison: {missing_files}")
        return False
    
    try:
        import polars as pl
        
        fda_h1_summary = pl.read_csv(fda_h1_summary_file)
        earnings_h1_summary = pl.read_csv(earnings_h1_summary_file)
        fda_h1_test = pl.read_csv(fda_h1_test_file)
        earnings_h1_test = pl.read_csv(earnings_h1_test_file)
        
        # Create comparison data
        comparison_data = {
            'Event Type': ['FDA Approvals', 'Earnings Announcements'],
            'Hypothesis 1 Supported': [
                fda_h1_summary['result'][0],
                earnings_h1_summary['result'][0]
            ],
            'Pre-Event RVR': [fda_h1_test["pre_event_rvr"][0], earnings_h1_test["pre_event_rvr"][0]],
            'Post-Rising RVR': [fda_h1_test["post_rising_rvr"][0], earnings_h1_test["post_rising_rvr"][0]],
            'Post-Decay RVR': [fda_h1_test["post_decay_rvr"][0], earnings_h1_test["post_decay_rvr"][0]],
            'Details': [fda_h1_summary['details'][0], earnings_h1_summary['details'][0]]
        }
        
        comparison_df = pl.DataFrame(comparison_data)
        comparison_df.write_csv(os.path.join(comparison_dir, "hypothesis1_overall_comparison.csv"))
        
        print("\nHypothesis 1 Comparison Summary:")
        print(comparison_df)
        
        # Generate comparison visualization
        try:
            import matplotlib.pyplot as plt
            
            phases_plot_data = []
            for event_type, df_test in [("FDA Approvals", fda_h1_test), ("Earnings Announcements", earnings_h1_test)]:
                phases_plot_data.append({'event_type': event_type, 'phase': 'Pre-Event', 'rvr': df_test['pre_event_rvr'][0]})
                phases_plot_data.append({'event_type': event_type, 'phase': 'Post-Event Rising', 'rvr': df_test['post_rising_rvr'][0]})
                phases_plot_data.append({'event_type': event_type, 'phase': 'Post-Event Decay', 'rvr': df_test['post_decay_rvr'][0]})
            
            phases_plot_df = pl.from_dicts(phases_plot_data)
            phases_pd = phases_plot_df.to_pandas()
            
            fig, ax = plt.subplots(figsize=(12, 7))
            
            event_types_unique = phases_pd['event_type'].unique()
            phases_order = ['Pre-Event', 'Post-Event Rising', 'Post-Event Decay']
            num_phases = len(phases_order)
            x_indices = np.arange(num_phases)
            bar_width = 0.35
            
            data_fda = phases_pd[phases_pd['event_type'] == event_types_unique[0]].set_index('phase').reindex(phases_order)['rvr']
            data_earnings = phases_pd[phases_pd['event_type'] == event_types_unique[1]].set_index('phase').reindex(phases_order)['rvr']

            rects1 = ax.bar(x_indices - bar_width/2, data_fda.fillna(0), bar_width, 
                           label=event_types_unique[0], color='deepskyblue', alpha=0.85, edgecolor='black')
            rects2 = ax.bar(x_indices + bar_width/2, data_earnings.fillna(0), bar_width, 
                           label=event_types_unique[1], color='salmon', alpha=0.85, edgecolor='black')

            ax.set_ylabel('Average RVR', fontsize=12)
            ax.set_title(f'Hypothesis 1: RVR by Phase and Event Type (Window: {ANALYSIS_WINDOW[0]} to {ANALYSIS_WINDOW[1]} days)', 
                        fontsize=14, fontweight='bold')
            ax.set_xticks(x_indices)
            ax.set_xticklabels(phases_order, fontsize=11)
            ax.legend(fontsize=10, loc='upper right')
            ax.grid(True, axis='y', linestyle=':', alpha=0.6)
            ax.axhline(0, color='black', linewidth=0.5)

            # Add value labels on bars
            def autolabel_bars(rects_group):
                for rect_item in rects_group:
                    height = rect_item.get_height()
                    ax.annotate(f'{height:.3f}' if pd.notnull(height) else 'N/A',
                                xy=(rect_item.get_x() + rect_item.get_width() / 2, height),
                                xytext=(0, 3), textcoords="offset points",
                                ha='center', va='bottom', fontsize=9)
            
            autolabel_bars(rects1)
            autolabel_bars(rects2)
            
            # Add support status
            fda_supported_text = "SUPPORTED" if fda_h1_summary["result"][0] else "NOT SUPPORTED"
            earn_supported_text = "SUPPORTED" if earnings_h1_summary["result"][0] else "NOT SUPPORTED"
            support_text = f"H1 Support: FDA {fda_supported_text} | Earnings {earn_supported_text}"
            fig.text(0.5, 0.01, support_text, ha='center', va='bottom', fontsize=10, 
                     bbox=dict(boxstyle='round,pad=0.3', fc='lightyellow', alpha=0.7))

            plt.tight_layout(rect=[0, 0.05, 1, 0.95])
            plt.savefig(os.path.join(comparison_dir, "hypothesis1_rvr_phase_comparison.png"), dpi=200)
            plt.close(fig)
            print(f"Saved RVR phase comparison plot to: {os.path.join(comparison_dir, 'hypothesis1_rvr_phase_comparison.png')}")

        except Exception as e:
            print(f"Error creating H1 comparison plot: {e}")
            traceback.print_exc()
        
        return True

    except Exception as e:
        print(f"Error comparing H1 results: {e}")
        traceback.print_exc()
        return False


def main():
    """Main execution function."""
    print("=" * 80)
    print("DYNAMIC ASSET PRICING MODEL FOR HIGH-UNCERTAINTY EVENTS")
    print("=" * 80)
    print(f"Starting analysis at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Analysis window: {ANALYSIS_WINDOW[0]} to {ANALYSIS_WINDOW[1]} days")
    print(f"GARCH type: {GARCH_TYPE}")
    print("=" * 80)
    
    # Run FDA analysis
    fda_success = run_fda_analysis()
    
    # Run Earnings analysis
    earnings_success = run_earnings_analysis()
    
    # Compare results if both succeeded
    if fda_success and earnings_success:
        compare_success = compare_results()
        if compare_success:
            print("\n=== All H1 analyses and comparisons completed successfully ===")
        else:
            print("\n=== H1 Analyses completed, but comparison failed ===")
    elif fda_success:
        print("\n=== Only FDA H1 analysis completed successfully ===")
    elif earnings_success:
        print("\n=== Only Earnings H1 analysis completed successfully ===")
    else:
        print("\n=== Both H1 analyses failed ===")
    
    print(f"\nAnalysis completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()