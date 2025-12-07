import pandas as pd
import numpy as np
import os
import sys
import traceback
import polars as pl
from typing import List, Tuple

# --- Configuration ---
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path: 
    sys.path.append(current_dir)

try: 
    from src.event_processor import EventDataLoader, EventFeatureEngineer, EventAnalysis
    from src.models import GARCHModel, GJRGARCHModel, ThreePhaseVolatilityModel
    from src.statistical_tests import PhaseComparisonTests
    from src.cross_sectional_analysis import CrossSectionalAnalyzer
    from src.regression_analysis import EventRegressionAnalyzer
    print("Successfully imported all modules.")
except ImportError as e: 
    print(f"Error importing modules: {e}")
    print("Ensure all files are in src/ directory or Python path.")
    sys.exit(1)

pl.Config.set_engine_affinity(engine="streaming")

# --- Hardcoded Analysis Parameters ---
STOCK_FILES = [
    "/home/d87016661/crsp_dsf-2000-2001.parquet",
    "/home/d87016661/crsp_dsf-2002-2003.parquet",
    "/home/d87016661/crsp_dsf-2004-2005.parquet",
    "/home/d87016661/crsp_dsf-2006-2007.parquet",
    "/home/d87016661/crsp_dsf-2008-2009.parquet",
    "/home/d87016661/crsp_dsf-2010-2011.parquet",
    "/home/d87016661/crsp_dsf-2012-2013.parquet",
    "/home/d87016661/crsp_dsf-2014-2015.parquet",
    "/home/d87016661/crsp_dsf-2016-2017.parquet",
    "/home/d87016661/crsp_dsf-2018-2019.parquet",
    "/home/d87016661/crsp_dsf-2020-2021.parquet",
    "/home/d87016661/crsp_dsf-2022-2023.parquet",
    "/home/d87016661/crsp_dsf-2024-2025.parquet"
]

FDA_EVENT_FILE = "/home/d87016661/fda_ticker_list_2000_to_2024.csv"
FDA_RESULTS_DIR = "results/hypothesis1/results_fda/"
FDA_FILE_PREFIX = "fda"
FDA_EVENT_DATE_COL = "Approval Date"
FDA_TICKER_COL = "ticker"

EARNINGS_EVENT_FILE = "/home/d87016661/detail_history_actuals.csv"
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

# Testing mode - set to True to run on small sample first
TEST_MODE = False
TEST_SAMPLE_SIZE = 1000  # Number of events for testing


def generate_h1_summary_report(results_dir: str, file_prefix: str):
    """Generate summary CSV and plot for Hypothesis 1"""
    print(f"\n--- Generating Hypothesis 1 Summary Report for {file_prefix.upper()} ---")
    h1_test_file = os.path.join(results_dir, f"{file_prefix}_hypothesis1_test.csv")
    
    if not os.path.exists(h1_test_file):
        print(f"  Warning: H1 test result file not found: {h1_test_file}")
        summary_df = pl.DataFrame({
            'hypothesis': ["H1: RVR peaks during post-event rising phase"],
            'result': [False],
            'details': ["Test file not found"]
        })
        summary_df.write_csv(os.path.join(results_dir, f"{file_prefix}_h1_overall_summary.csv"))
        print(f"  Hypothesis 1 for {file_prefix.upper()}: NOT SUPPORTED (result file missing).")
        return

    try:
        h1_df = pl.read_csv(h1_test_file)
        if h1_df.is_empty():
            print(f"  Warning: H1 test result file is empty: {h1_test_file}")
            summary_df = pl.DataFrame({
                'hypothesis': ["H1: RVR peaks during post-event rising phase"],
                'result': [False],
                'details': ["Result file empty"]
            })
            summary_df.write_csv(os.path.join(results_dir, f"{file_prefix}_h1_overall_summary.csv"))
            print(f"  Hypothesis 1 for {file_prefix.upper()}: NOT SUPPORTED (result file empty).")
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

        # Generate plot
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
                            f'{height:.3f}', ha='center', 
                            va='bottom' if height >= 0 else 'top')
            
            status_text = "SUPPORTED" if h1_supported else "NOT SUPPORTED"
            ax.text(0.5, -0.15, f"Overall H1 Status: {status_text}", 
                    ha='center', transform=ax.transAxes,
                    bbox=dict(boxstyle='round,pad=0.3', fc='lightyellow', alpha=0.8))

            phase_windows = {
                'Pre-Event': f'({ANALYSIS_WINDOW[0]} to -1)',
                'Post-Event Rising': '(0 to 5)',
                'Post-Event Decay': '(6 to 15)'
            }
            phase_labels = [f"{phase}\n{phase_windows[phase]}" for phase in phases_pd['phase']]
            ax.set_xticklabels(phase_labels, fontsize=10)

            plt.tight_layout(rect=[0, 0.05, 1, 0.95])
            plt.savefig(os.path.join(results_dir, f"{file_prefix}_h1_rvr_phase_plot.png"), dpi=150)
            plt.close(fig)

        except Exception as e:
            print(f"  Error generating H1 plot for {file_prefix}: {e}")

    except Exception as e:
        print(f"  Error generating H1 summary for {file_prefix}: {e}")


def run_comprehensive_analysis(analyzer, results_dir: str, file_prefix: str):
    """
    Run complete statistical analysis suite
    
    Includes:
    1. Statistical significance tests
    2. Cross-sectional analysis (size, industry, liquidity)
    3. Temporal stability
    4. Panel regression
    """
    
    phases = {
        'pre_event': (ANALYSIS_WINDOW[0], -1),
        'post_event_rising': (0, DELTA),
        'post_event_decay': (DELTA+1, ANALYSIS_WINDOW[1])
    }
    
    # ============================================================
    # 1. STATISTICAL SIGNIFICANCE TESTS
    # ============================================================
    print("\n" + "="*70)
    print("STEP 1: Statistical Significance Tests")
    print("="*70)
    
    try:
        tester = PhaseComparisonTests(analyzer.data, phases)
        
        # T-tests
        print("  Running Welch's t-tests...")
        pre_vs_rising = tester.welch_t_test('pre_event', 'post_event_rising', 'rvr')
        rising_vs_decay = tester.welch_t_test('post_event_rising', 'post_event_decay', 'rvr')
        
        print(f"    Pre vs Rising: t={pre_vs_rising['t_statistic']:.3f}, p={pre_vs_rising['p_value']:.6f}")
        print(f"    Rising vs Decay: t={rising_vs_decay['t_statistic']:.3f}, p={rising_vs_decay['p_value']:.6f}")
        
        # Bootstrap CI
        print("  Computing bootstrap confidence intervals (10,000 iterations)...")
        boot_ci = tester.bootstrap_phase_difference('pre_event', 'post_event_rising', 'rvr')
        print(f"    Bootstrap CI: [{boot_ci['ci_lower']:.3f}, {boot_ci['ci_upper']:.3f}]")
        
        # Mann-Whitney U
        print("  Running Mann-Whitney U test...")
        mw_test = tester.mann_whitney_u_test('pre_event', 'post_event_rising', 'rvr')
        print(f"    Mann-Whitney U: statistic={mw_test['u_statistic']:.1f}, p={mw_test['p_value']:.6f}")
        
        # Save results
        test_results = pl.DataFrame([
            {
                'test': 'Welch t-test: Pre vs Rising',
                't_stat': pre_vs_rising['t_statistic'],
                'p_value': pre_vs_rising['p_value'],
                'mean_diff': pre_vs_rising['mean_diff'],
                'ci_95_lower': pre_vs_rising['ci_lower'],
                'ci_95_upper': pre_vs_rising['ci_upper']
            },
            {
                'test': 'Welch t-test: Rising vs Decay',
                't_stat': rising_vs_decay['t_statistic'],
                'p_value': rising_vs_decay['p_value'],
                'mean_diff': rising_vs_decay['mean_diff'],
                'ci_95_lower': rising_vs_decay['ci_lower'],
                'ci_95_upper': rising_vs_decay['ci_upper']
            },
            {
                'test': 'Bootstrap CI: Pre vs Rising',
                't_stat': None,
                'p_value': None,
                'mean_diff': boot_ci['mean_diff'],
                'ci_95_lower': boot_ci['ci_lower'],
                'ci_95_upper': boot_ci['ci_upper']
            },
            {
                'test': 'Mann-Whitney U: Pre vs Rising',
                't_stat': mw_test['u_statistic'],
                'p_value': mw_test['p_value'],
                'mean_diff': mw_test['median_diff'],
                'ci_95_lower': None,
                'ci_95_upper': None
            }
        ])
        
        test_results.write_csv(os.path.join(results_dir, f"{file_prefix}_statistical_tests.csv"))
        print(f"  ✓ Saved: {file_prefix}_statistical_tests.csv")
        
    except Exception as e:
        print(f"  ✗ Error in statistical tests: {e}")
        traceback.print_exc()
    
    # ============================================================
    # 2. CROSS-SECTIONAL ANALYSIS
    # ============================================================
    print("\n" + "="*70)
    print("STEP 2: Cross-Sectional Analysis")
    print("="*70)
    
    try:
        cross_sec = CrossSectionalAnalyzer(analyzer.data, STOCK_FILES)
        
        # Size quintiles
        print("\n  [2.1] Assigning size quintiles from market cap...")
        analyzer.data = cross_sec.assign_size_quintiles()
        cross_sec.data = analyzer.data  # Update cross_sec reference
        
        print("  Analyzing RVR by size quintile...")
        size_results = cross_sec.analyze_rvr_by_characteristic(
            'size_quintile', phases, results_dir, file_prefix
        )
        print(f"  ✓ Saved: {file_prefix}_rvr_by_size_quintile.csv and .png")
        
        # Industry
        print("\n  [2.2] Assigning industry classification from SIC codes...")
        analyzer.data = cross_sec.assign_industry_classification()
        cross_sec.data = analyzer.data
        
        print("  Analyzing RVR by industry...")
        industry_results = cross_sec.analyze_rvr_by_characteristic(
            'industry', phases, results_dir, file_prefix
        )
        print(f"  ✓ Saved: {file_prefix}_rvr_by_industry.csv and .png")
        
        # Liquidity
        print("\n  [2.3] Assigning liquidity quintiles from trading volume...")
        analyzer.data = cross_sec.assign_liquidity_quintiles()
        cross_sec.data = analyzer.data
        
        print("  Analyzing RVR by liquidity quintile...")
        liquidity_results = cross_sec.analyze_rvr_by_characteristic(
            'liquidity_quintile', phases, results_dir, file_prefix
        )
        print(f"  ✓ Saved: {file_prefix}_rvr_by_liquidity_quintile.csv and .png")
        
    except Exception as e:
        print(f"  ✗ Error in cross-sectional analysis: {e}")
        traceback.print_exc()
    
    # ============================================================
    # 3. TEMPORAL STABILITY
    # ============================================================
    print("\n" + "="*70)
    print("STEP 3: Temporal Stability Analysis")
    print("="*70)
    
    try:
        periods = [
            ('2000-2007', '2000-01-01', '2007-12-31'),
            ('2008-2015', '2008-01-01', '2015-12-31'),
            ('2016-2024', '2016-01-01', '2024-12-31')
        ]
        
        print("  Testing pattern persistence across time periods...")
        temporal_results = cross_sec.analyze_temporal_stability(
            periods, phases, results_dir, file_prefix
        )
        print(f"  ✓ Saved: {file_prefix}_temporal_stability.csv and .png")
        
        # Print amplification ratios by period
        print("\n  Amplification Ratios by Period:")
        for period_name in ['2000-2007', '2008-2015', '2016-2024']:
            period_data = temporal_results.filter(pl.col('period') == period_name)
            rising_data = period_data.filter(pl.col('phase') == 'post_event_rising')
            if rising_data.height > 0 and rising_data['amplification'][0] is not None:
                print(f"    {period_name}: {rising_data['amplification'][0]:.2f}x")
        
    except Exception as e:
        print(f"  ✗ Error in temporal stability analysis: {e}")
        traceback.print_exc()
    
    # ============================================================
    # 4. PANEL REGRESSION
    # ============================================================
    print("\n" + "="*70)
    print("STEP 4: Panel Regression Analysis")
    print("="*70)
    
    try:
        reg_analyzer = EventRegressionAnalyzer(analyzer.data)
        
        # Basic specification
        print("  [4.1] Running basic specification (no controls)...")
        basic_reg = reg_analyzer.run_panel_regression(
            dependent_var='rvr',
            control_vars=None,
            cluster_vars=['ticker'],
            firm_fe=False,
            time_fe=False
        )
        print(f"    Post-Rising β: {basic_reg['coef_rising']:.4f}")
        print(f"    t-statistic: {basic_reg['t_rising']:.3f}")
        print(f"    p-value: {basic_reg['p_rising']:.6f}")
        print(f"    N: {basic_reg['n_obs']:,}")
        
        # With size control
        print("\n  [4.2] Running specification with size control...")
        size_reg = reg_analyzer.run_panel_regression(
            dependent_var='rvr',
            control_vars=['size_quintile'],
            cluster_vars=['ticker'],
            firm_fe=False,
            time_fe=True
        )
        print(f"    Post-Rising β: {size_reg['coef_rising']:.4f}")
        print(f"    t-statistic: {size_reg['t_rising']:.3f}")
        print(f"    p-value: {size_reg['p_rising']:.6f}")
        
        # Full specification
        print("\n  [4.3] Running full specification (all controls + FE)...")
        full_reg = reg_analyzer.run_panel_regression(
            dependent_var='rvr',
            control_vars=['size_quintile', 'industry', 'liquidity_quintile'],
            cluster_vars=['ticker'],
            firm_fe=True,
            time_fe=True
        )
        print(f"    Post-Rising β: {full_reg['coef_rising']:.4f}")
        print(f"    t-statistic: {full_reg['t_rising']:.3f}")
        print(f"    p-value: {full_reg['p_rising']:.6f}")
        print(f"    R²: {full_reg['r_squared']:.4f}")
        
        # Save regression results
        with open(os.path.join(results_dir, f"{file_prefix}_regression_results.txt"), 'w') as f:
            f.write("="*70 + "\n")
            f.write("PANEL REGRESSION RESULTS\n")
            f.write("="*70 + "\n\n")
            
            f.write("SPECIFICATION 1: Basic (No Controls)\n")
            f.write("-"*70 + "\n")
            f.write(str(basic_reg['summary']))
            f.write("\n\n")
            
            f.write("SPECIFICATION 2: Size Control + Time FE\n")
            f.write("-"*70 + "\n")
            f.write(str(size_reg['summary']))
            f.write("\n\n")
            
            f.write("SPECIFICATION 3: Full Controls + Firm FE + Time FE\n")
            f.write("-"*70 + "\n")
            f.write(str(full_reg['summary']))
        
        print(f"  ✓ Saved: {file_prefix}_regression_results.txt")
        
        # Create summary table
        reg_summary = pl.DataFrame([
            {
                'specification': 'Basic',
                'controls': 'None',
                'firm_fe': 'No',
                'time_fe': 'No',
                'coef_rising': basic_reg['coef_rising'],
                't_stat': basic_reg['t_rising'],
                'p_value': basic_reg['p_rising'],
                'n_obs': basic_reg['n_obs']
            },
            {
                'specification': 'Size Control',
                'controls': 'Size Quintile',
                'firm_fe': 'No',
                'time_fe': 'Yes',
                'coef_rising': size_reg['coef_rising'],
                't_stat': size_reg['t_rising'],
                'p_value': size_reg['p_rising'],
                'n_obs': size_reg['n_obs']
            },
            {
                'specification': 'Full',
                'controls': 'Size + Industry + Liquidity',
                'firm_fe': 'Yes',
                'time_fe': 'Yes',
                'coef_rising': full_reg['coef_rising'],
                't_stat': full_reg['t_rising'],
                'p_value': full_reg['p_rising'],
                'n_obs': full_reg['n_obs']
            }
        ])
        
        reg_summary.write_csv(os.path.join(results_dir, f"{file_prefix}_regression_summary.csv"))
        print(f"  ✓ Saved: {file_prefix}_regression_summary.csv")
        
    except Exception as e:
        print(f"  ✗ Error in regression analysis: {e}")
        traceback.print_exc()
    
    print("\n" + "="*70)
    print("COMPREHENSIVE ANALYSIS COMPLETE")
    print("="*70)


def run_fda_analysis():
    """Run complete FDA analysis pipeline"""
    print("\n" + "="*70)
    print("FDA APPROVAL EVENT ANALYSIS")
    print("="*70)
    
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
        print(f"\n*** Error creating results directory: {oe} ***")
        return False

    try:
        # Load data
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
        
        print(f"\nFDA data loaded: {analyzer.data.shape}")
        print(f"Unique events: {analyzer.data['event_id'].n_unique():,}")
        
        # Test mode - sample events
        if TEST_MODE:
            print(f"\n*** TEST MODE: Sampling {TEST_SAMPLE_SIZE} events ***")
            unique_events = analyzer.data['event_id'].unique()[:TEST_SAMPLE_SIZE]
            analyzer.data = analyzer.data.filter(pl.col('event_id').is_in(unique_events))
            print(f"Test sample: {analyzer.data.shape}, {analyzer.data['event_id'].n_unique():,} events")
        
        # Core RVR analysis
        print("\nRunning three-phase volatility analysis...")
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
        
        print("\nRunning RVR analysis with optimistic bias...")
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
        
        # Run comprehensive analysis suite
        run_comprehensive_analysis(analyzer, FDA_RESULTS_DIR, FDA_FILE_PREFIX)
        
        # Generate H1 summary
        generate_h1_summary_report(FDA_RESULTS_DIR, FDA_FILE_PREFIX)
        
        print(f"\n✓ FDA analysis complete. Results in: {FDA_RESULTS_DIR}")
        return True
        
    except Exception as e: 
        print(f"\n*** Error in FDA analysis: {e} ***")
        traceback.print_exc()
        return False


def run_earnings_analysis():
    """Run complete earnings analysis pipeline"""
    print("\n" + "="*70)
    print("EARNINGS ANNOUNCEMENT EVENT ANALYSIS")
    print("="*70)
    
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
        print(f"\n*** Error creating results directory: {oe} ***")
        return False

    try:
        # Load data
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
        
        print(f"\nEarnings data loaded: {analyzer.data.shape}")
        print(f"Unique events: {analyzer.data['event_id'].n_unique():,}")
        
        # Test mode - sample events
        if TEST_MODE:
            print(f"\n*** TEST MODE: Sampling {TEST_SAMPLE_SIZE} events ***")
            unique_events = analyzer.data['event_id'].unique()[:TEST_SAMPLE_SIZE]
            analyzer.data = analyzer.data.filter(pl.col('event_id').is_in(unique_events))
            print(f"Test sample: {analyzer.data.shape}, {analyzer.data['event_id'].n_unique():,} events")
        
        # Core RVR analysis
        print("\nRunning three-phase volatility analysis...")
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
        
        print("\nRunning RVR analysis with optimistic bias...")
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
        
        # Run comprehensive analysis suite
        run_comprehensive_analysis(analyzer, EARNINGS_RESULTS_DIR, EARNINGS_FILE_PREFIX)
        
        # Generate H1 summary
        generate_h1_summary_report(EARNINGS_RESULTS_DIR, EARNINGS_FILE_PREFIX)
        
        print(f"\n✓ Earnings analysis complete. Results in: {EARNINGS_RESULTS_DIR}")
        return True
        
    except Exception as e: 
        print(f"\n*** Error in earnings analysis: {e} ***")
        traceback.print_exc()
        return False


def compare_results():
    """Compare FDA and Earnings results"""
    print("\n" + "="*70)
    print("COMPARING FDA AND EARNINGS RESULTS")
    print("="*70)
    
    comparison_dir = "results/hypothesis1/comparison/"
    os.makedirs(comparison_dir, exist_ok=True)
    
    fda_h1_summary_file = os.path.join(FDA_RESULTS_DIR, f"{FDA_FILE_PREFIX}_h1_overall_summary.csv")
    earnings_h1_summary_file = os.path.join(EARNINGS_RESULTS_DIR, f"{EARNINGS_FILE_PREFIX}_h1_overall_summary.csv")
    fda_h1_test_file = os.path.join(FDA_RESULTS_DIR, f"{FDA_FILE_PREFIX}_hypothesis1_test.csv")
    earnings_h1_test_file = os.path.join(EARNINGS_RESULTS_DIR, f"{EARNINGS_FILE_PREFIX}_hypothesis1_test.csv")
    
    missing_files = [fp for fp in [fda_h1_summary_file, earnings_h1_summary_file, 
                                   fda_h1_test_file, earnings_h1_test_file] 
                    if not os.path.exists(fp)]
    if missing_files:
        print(f"Error: Missing files: {missing_files}")
        return False
    
    try:
        fda_h1_summary = pl.read_csv(fda_h1_summary_file)
        earnings_h1_summary = pl.read_csv(earnings_h1_summary_file)
        fda_h1_test = pl.read_csv(fda_h1_test_file)
        earnings_h1_test = pl.read_csv(earnings_h1_test_file)
        
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
        
        print("\nComparison Summary:")
        print(comparison_df)
        
        # Create comparison plot
        import matplotlib.pyplot as plt
        
        phases_plot_data = []
        for event_type, df_test in [("FDA Approvals", fda_h1_test), 
                                    ("Earnings Announcements", earnings_h1_test)]:
            phases_plot_data.append({'event_type': event_type, 'phase': 'Pre-Event', 
                                    'rvr': df_test['pre_event_rvr'][0]})
            phases_plot_data.append({'event_type': event_type, 'phase': 'Post-Event Rising', 
                                    'rvr': df_test['post_rising_rvr'][0]})
            phases_plot_data.append({'event_type': event_type, 'phase': 'Post-Event Decay', 
                                    'rvr': df_test['post_decay_rvr'][0]})
        
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
        ax.set_title(f'Hypothesis 1: RVR by Phase and Event Type', fontsize=14, fontweight='bold')
        ax.set_xticks(x_indices)
        ax.set_xticklabels(phases_order, fontsize=11)
        ax.legend(fontsize=10, loc='upper right')
        ax.grid(True, axis='y', linestyle=':', alpha=0.6)
        ax.axhline(0, color='black', linewidth=0.5)
        
        def autolabel_bars(rects_group):
            for rect_item in rects_group:
                height = rect_item.get_height()
                ax.annotate(f'{height:.3f}' if pd.notnull(height) else 'N/A',
                           xy=(rect_item.get_x() + rect_item.get_width() / 2, height),
                           xytext=(0, 3), textcoords="offset points",
                           ha='center', va='bottom', fontsize=9)
        
        autolabel_bars(rects1)
        autolabel_bars(rects2)
        
        fda_supported_text = "SUPPORTED" if fda_h1_summary["result"][0] else "NOT SUPPORTED"
        earn_supported_text = "SUPPORTED" if earnings_h1_summary["result"][0] else "NOT SUPPORTED"
        support_text = f"H1 Support: FDA {fda_supported_text} | Earnings {earn_supported_text}"
        fig.text(0.5, 0.01, support_text, ha='center', va='bottom', fontsize=10,
                bbox=dict(boxstyle='round,pad=0.3', fc='lightyellow', alpha=0.7))
        
        plt.tight_layout(rect=[0, 0.05, 1, 0.95])
        plt.savefig(os.path.join(comparison_dir, "hypothesis1_rvr_phase_comparison.png"), dpi=200)
        plt.close(fig)
        
        print(f"\n✓ Comparison complete. Results in: {comparison_dir}")
        return True
        
    except Exception as e:
        print(f"Error comparing results: {e}")
        traceback.print_exc()
        return False


def main():
    """Main execution function"""
    print("\n" + "="*70)
    print("EVENT-DRIVEN ASSET PRICING ANALYSIS")
    print("Comprehensive Statistical Analysis Suite")
    print("="*70)
    print(f"\nTest Mode: {'ENABLED' if TEST_MODE else 'DISABLED'}")
    if TEST_MODE:
        print(f"Sample Size: {TEST_SAMPLE_SIZE} events per dataset")
    print("\n")
    
    fda_success = run_fda_analysis()
    earnings_success = run_earnings_analysis()
    
    if fda_success and earnings_success:
        compare_success = compare_results()
        if compare_success:
            print("\n" + "="*70)
            print("ALL ANALYSES COMPLETED SUCCESSFULLY")
            print("="*70)
        else:
            print("\n" + "="*70)
            print("ANALYSES COMPLETED - COMPARISON FAILED")
            print("="*70)
    elif fda_success:
        print("\n" + "="*70)
        print("ONLY FDA ANALYSIS COMPLETED")
        print("="*70)
    elif earnings_success:
        print("\n" + "="*70)
        print("ONLY EARNINGS ANALYSIS COMPLETED")
        print("="*70)
    else:
        print("\n" + "="*70)
        print("BOTH ANALYSES FAILED")
        print("="*70)


if __name__ == "__main__":
    main()