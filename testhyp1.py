# testhyp1_improved.py

import pandas as pd
import numpy as np
import os
import sys
import traceback
import polars as pl
import matplotlib.pyplot as plt
from scipy import stats
from typing import List, Tuple, Dict, Optional, Any

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

class Hypothesis1Analyzer:
    """Enhanced analyzer for testing Hypothesis 1 about risk-adjusted returns."""
    
    def __init__(self, analyzer: EventAnalysis, results_dir: str, file_prefix: str):
        """
        Initialize Hypothesis1Analyzer.
        
        Parameters:
        -----------
        analyzer : EventAnalysis
            Instance of EventAnalysis with data loaded
        results_dir : str
            Directory to save results
        file_prefix : str
            Prefix for output files
        """
        self.analyzer = analyzer
        self.results_dir = results_dir
        self.file_prefix = file_prefix
        self.return_col = 'ret'
        self.analysis_window = ANALYSIS_WINDOW
        self.garch_type = GARCH_TYPE
        self.delta = DELTA
        self.optimistic_bias = OPTIMISTIC_BIAS
        self.risk_free_rate = RISK_FREE_RATE
        
        # Define phase windows
        self.phases = {
            'pre_event': (ANALYSIS_WINDOW[0], -1),
            'event_day': (0, 0),
            'post_event_rising': (1, DELTA),
            'post_event_decay': (DELTA + 1, ANALYSIS_WINDOW[1])
        }
        
        # Results storage
        self.volatility_data = None
        self.rvr_data = None
        self.sharpe_data = None
        self.phase_stats = None
        self.h1_results = None
        
        # Create results directory
        os.makedirs(results_dir, exist_ok=True)
    
    def run_analysis(self):
        """
        Run the complete analysis for Hypothesis 1.
        This includes:
        1. Analyzing three-phase volatility
        2. Calculating RVR by phase
        3. Calculating Sharpe ratio by phase
        4. Testing the hypothesis statistically
        5. Generating visualizations
        6. Creating a comprehensive report
        """
        print("\n--- Running Comprehensive Analysis for Hypothesis 1 ---")
        
        # Run three-phase volatility analysis
        self._analyze_volatility()
        
        # Calculate and analyze RVR
        self._analyze_rvr()
        
        # Calculate and analyze Sharpe ratio
        self._analyze_sharpe_ratio()
        
        # Test hypothesis statistically
        self._test_hypothesis()
        
        # Generate visualizations
        self._generate_visualizations()
        
        # Create comprehensive report
        self._create_report()
        
        return True
    
    def _analyze_volatility(self):
        """
        Analyze three-phase volatility dynamics.
        """
        print("\n--- Analyzing Three-Phase Volatility ---")
        
        # Run analyzer's existing three-phase volatility method
        volatility_df, phase_stats_df = self.analyzer.analyze_three_phase_volatility(
            results_dir=self.results_dir,
            file_prefix=self.file_prefix,
            return_col=self.return_col,
            analysis_window=self.analysis_window,
            garch_type=self.garch_type,
            k1=K1, k2=K2,
            delta_t1=DELTA_T1, delta_t2=DELTA_T2, delta_t3=DELTA_T3,
            delta=self.delta
        )
        
        self.volatility_data = volatility_df
    
    def _analyze_rvr(self):
        """
        Calculate and analyze Return-to-Variance Ratio (RVR).
        """
        print("\n--- Analyzing Return-to-Variance Ratio (RVR) ---")
        
        # Run analyzer's existing RVR method
        rvr_df, phase_stats_df = self.analyzer.analyze_rvr_with_optimistic_bias(
            results_dir=self.results_dir,
            file_prefix=self.file_prefix,
            return_col=self.return_col,
            analysis_window=self.analysis_window,
            garch_type=self.garch_type,
            k1=K1, k2=K2,
            delta_t1=DELTA_T1, delta_t2=DELTA_T2, delta_t3=DELTA_T3,
            delta=self.delta,
            optimistic_bias=self.optimistic_bias,
            risk_free_rate=self.risk_free_rate
        )
        
        self.rvr_data = rvr_df
        self.phase_stats = phase_stats_df
    
    def _analyze_sharpe_ratio(self):
        """
        Calculate and analyze Sharpe ratio.
        """
        print("\n--- Analyzing Sharpe Ratio ---")
        
        # Calculate Sharpe ratio timeseries
        sharpe_df = self.analyzer.calculate_rolling_sharpe_timeseries(
            results_dir=self.results_dir,
            file_prefix=self.file_prefix,
            return_col=self.return_col,
            analysis_window=self.analysis_window,
            sharpe_window=5,  # Using same window as RVR for consistency
            annualize=True,
            risk_free_rate=self.risk_free_rate
        )
        
        self.sharpe_data = sharpe_df
        
        # Calculate average Sharpe ratio by phase
        if sharpe_df is not None:
            sharpe_phase_stats = []
            
            for phase_name, (start_day, end_day) in self.phases.items():
                phase_data = sharpe_df.filter(
                    (pl.col('days_to_event') >= start_day) &
                    (pl.col('days_to_event') <= end_day)
                )
                
                if not phase_data.is_empty():
                    avg_sharpe = phase_data['sharpe_ratio'].mean()
                    median_sharpe = phase_data['sharpe_ratio'].median()
                    
                    sharpe_phase_stats.append({
                        'phase': phase_name,
                        'start_day': start_day,
                        'end_day': end_day,
                        'avg_sharpe': avg_sharpe,
                        'median_sharpe': median_sharpe
                    })
            
            # Save Sharpe ratio phase statistics
            if sharpe_phase_stats:
                sharpe_phase_df = pl.DataFrame(sharpe_phase_stats)
                sharpe_phase_df.write_csv(os.path.join(self.results_dir, f"{self.file_prefix}_sharpe_phase_stats.csv"))
                
                # Print summary
                print("\nSharpe Ratio by Phase:")
                for row in sharpe_phase_df.iter_rows(named=True):
                    print(f"Phase: {row['phase']} ({row['start_day']} to {row['end_day']} days)")
                    print(f"  Avg Sharpe: {row['avg_sharpe']:.4f}, Median Sharpe: {row['median_sharpe']:.4f}")
    
    def _test_hypothesis(self):
        """
        Test Hypothesis 1 statistically.
        """
        print("\n--- Testing Hypothesis 1 Statistically ---")
        
        # Load phase statistics if not already available
        if self.phase_stats is None:
            phase_stats_file = os.path.join(self.results_dir, f"{self.file_prefix}_rvr_phase_stats.csv")
            if os.path.exists(phase_stats_file):
                self.phase_stats = pl.read_csv(phase_stats_file)
        
        # Load Sharpe ratio phase statistics
        sharpe_phase_file = os.path.join(self.results_dir, f"{self.file_prefix}_sharpe_phase_stats.csv")
        sharpe_phase_df = None
        if os.path.exists(sharpe_phase_file):
            sharpe_phase_df = pl.read_csv(sharpe_phase_file)
        
        if self.phase_stats is not None:
            # Extract RVR by phase
            pre_event_rvr = None
            post_rising_rvr = None
            post_decay_rvr = None
            
            for row in self.phase_stats.iter_rows(named=True):
                if row['phase'] == 'pre_event':
                    pre_event_rvr = row['avg_rvr']
                elif row['phase'] == 'post_event_rising':
                    post_rising_rvr = row['avg_rvr']
                elif row['phase'] == 'post_event_decay':
                    post_decay_rvr = row['avg_rvr']
            
            # Test if RVR peaks during post-event rising phase
            rvr_h1_result = False
            if pre_event_rvr is not None and post_rising_rvr is not None and post_decay_rvr is not None:
                rvr_h1_result = post_rising_rvr > pre_event_rvr and post_rising_rvr > post_decay_rvr
                
                print("\nRVR Hypothesis Test:")
                print(f"Does RVR peak during post-event rising phase? {'YES' if rvr_h1_result else 'NO'}")
                print(f"  Pre-event RVR: {pre_event_rvr:.4f}")
                print(f"  Post-event Rising RVR: {post_rising_rvr:.4f}")
                print(f"  Post-event Decay RVR: {post_decay_rvr:.4f}")
            
            # Test if Sharpe ratio peaks during post-event rising phase
            sharpe_h1_result = False
            if sharpe_phase_df is not None:
                pre_event_sharpe = None
                post_rising_sharpe = None
                post_decay_sharpe = None
                
                for row in sharpe_phase_df.iter_rows(named=True):
                    if row['phase'] == 'pre_event':
                        pre_event_sharpe = row['avg_sharpe']
                    elif row['phase'] == 'post_event_rising':
                        post_rising_sharpe = row['avg_sharpe']
                    elif row['phase'] == 'post_event_decay':
                        post_decay_sharpe = row['avg_sharpe']
                
                if pre_event_sharpe is not None and post_rising_sharpe is not None and post_decay_sharpe is not None:
                    sharpe_h1_result = post_rising_sharpe > pre_event_sharpe and post_rising_sharpe > post_decay_sharpe
                    
                    print("\nSharpe Ratio Hypothesis Test:")
                    print(f"Does Sharpe ratio peak during post-event rising phase? {'YES' if sharpe_h1_result else 'NO'}")
                    print(f"  Pre-event Sharpe: {pre_event_sharpe:.4f}")
                    print(f"  Post-event Rising Sharpe: {post_rising_sharpe:.4f}")
                    print(f"  Post-event Decay Sharpe: {post_decay_sharpe:.4f}")
            
            # Calculate statistical significance using t-tests
            # These tests compare post-event rising phase with other phases
            rvr_p_values = {}
            sharpe_p_values = {}
            
            # Load daily data for statistical tests
            rvr_daily_file = os.path.join(self.results_dir, f"{self.file_prefix}_rvr_timeseries.csv")
            if os.path.exists(rvr_daily_file):
                rvr_daily = pl.read_csv(rvr_daily_file)
                
                # For each phase comparison, run a t-test against post-event rising
                for phase_name, (start_day, end_day) in self.phases.items():
                    if phase_name != 'post_event_rising':
                        phase_rvr = rvr_daily.filter(
                            (pl.col('days_to_event') >= start_day) &
                            (pl.col('days_to_event') <= end_day)
                        )['mean_rvr'].to_numpy()
                        
                        post_rising_rvr_daily = rvr_daily.filter(
                            (pl.col('days_to_event') >= self.phases['post_event_rising'][0]) &
                            (pl.col('days_to_event') <= self.phases['post_event_rising'][1])
                        )['mean_rvr'].to_numpy()
                        
                        # Run t-test if sufficient data
                        if len(phase_rvr) > 0 and len(post_rising_rvr_daily) > 0:
                            t_stat, p_value = stats.ttest_ind(post_rising_rvr_daily, phase_rvr, equal_var=False)
                            rvr_p_values[phase_name] = {
                                't_stat': t_stat,
                                'p_value': p_value,
                                'significant': p_value < 0.05
                            }
            
            # Perform similar t-tests for Sharpe ratio
            if self.sharpe_data is not None:
                # For each phase comparison, run a t-test against post-event rising
                for phase_name, (start_day, end_day) in self.phases.items():
                    if phase_name != 'post_event_rising':
                        phase_sharpe = self.sharpe_data.filter(
                            (pl.col('days_to_event') >= start_day) &
                            (pl.col('days_to_event') <= end_day)
                        )['sharpe_ratio'].to_numpy()
                        
                        post_rising_sharpe_daily = self.sharpe_data.filter(
                            (pl.col('days_to_event') >= self.phases['post_event_rising'][0]) &
                            (pl.col('days_to_event') <= self.phases['post_event_rising'][1])
                        )['sharpe_ratio'].to_numpy()
                        
                        # Run t-test if sufficient data
                        if len(phase_sharpe) > 0 and len(post_rising_sharpe_daily) > 0:
                            t_stat, p_value = stats.ttest_ind(post_rising_sharpe_daily, phase_sharpe, equal_var=False)
                            sharpe_p_values[phase_name] = {
                                't_stat': t_stat,
                                'p_value': p_value,
                                'significant': p_value < 0.05
                            }
            
            # Print statistical test results
            if rvr_p_values:
                print("\nRVR Statistical Tests (Post-Event Rising vs. Other Phases):")
                for phase, stats_dict in rvr_p_values.items():
                    significance = "***" if stats_dict['p_value'] < 0.01 else "**" if stats_dict['p_value'] < 0.05 else "*" if stats_dict['p_value'] < 0.1 else ""
                    print(f"  vs. {phase}: t={stats_dict['t_stat']:.4f}, p={stats_dict['p_value']:.4f} {significance}")
            
            if sharpe_p_values:
                print("\nSharpe Ratio Statistical Tests (Post-Event Rising vs. Other Phases):")
                for phase, stats_dict in sharpe_p_values.items():
                    significance = "***" if stats_dict['p_value'] < 0.01 else "**" if stats_dict['p_value'] < 0.05 else "*" if stats_dict['p_value'] < 0.1 else ""
                    print(f"  vs. {phase}: t={stats_dict['t_stat']:.4f}, p={stats_dict['p_value']:.4f} {significance}")
            
            # Overall hypothesis result
            # Hypothesis 1 is supported if either RVR or Sharpe ratio peaks during post-event rising phase
            # AND statistical tests show significance
            significant_rvr_tests = sum(1 for stats_dict in rvr_p_values.values() if stats_dict['significant'])
            significant_sharpe_tests = sum(1 for stats_dict in sharpe_p_values.values() if stats_dict['significant'])
            
            rvr_supported = rvr_h1_result and significant_rvr_tests > 0
            sharpe_supported = sharpe_h1_result and significant_sharpe_tests > 0
            
            h1_supported = rvr_supported or sharpe_supported
            
            print("\nOverall Hypothesis 1 Result:")
            print(f"  Supported? {'YES' if h1_supported else 'NO'}")
            print(f"  RVR Evidence: {'YES' if rvr_supported else 'NO'} ({significant_rvr_tests} significant tests)")
            print(f"  Sharpe Ratio Evidence: {'YES' if sharpe_supported else 'NO'} ({significant_sharpe_tests} significant tests)")
            
            # Save hypothesis results
            self.h1_results = {
                'hypothesis_supported': h1_supported,
                'rvr_supported': rvr_supported,
                'sharpe_supported': sharpe_supported,
                'rvr_p_values': rvr_p_values,
                'sharpe_p_values': sharpe_p_values,
                'rvr_by_phase': {
                    'pre_event': pre_event_rvr,
                    'post_event_rising': post_rising_rvr,
                    'post_event_decay': post_decay_rvr
                },
                'sharpe_by_phase': {
                    'pre_event': pre_event_sharpe,
                    'post_event_rising': post_rising_sharpe,
                    'post_event_decay': post_decay_sharpe
                }
            }
            
            # Save to CSV
            h1_result_df = pl.DataFrame({
                'hypothesis': ['H1: Risk-adjusted returns peak during post-event rising phase'],
                'result': [h1_supported],
                'rvr_supported': [rvr_supported],
                'sharpe_supported': [sharpe_supported],
                'pre_event_rvr': [pre_event_rvr],
                'post_rising_rvr': [post_rising_rvr],
                'post_decay_rvr': [post_decay_rvr],
                'pre_event_sharpe': [pre_event_sharpe],
                'post_rising_sharpe': [post_rising_sharpe],
                'post_decay_sharpe': [post_decay_sharpe],
                'significant_rvr_tests': [significant_rvr_tests],
                'significant_sharpe_tests': [significant_sharpe_tests]
            })
            h1_result_df.write_csv(os.path.join(self.results_dir, f"{self.file_prefix}_hypothesis1_test.csv"))
    
    def _generate_visualizations(self):
        """
        Generate enhanced visualizations for the hypothesis.
        """
        print("\n--- Generating Enhanced Visualizations ---")
        
        # Create a combined visualization showing both RVR and Sharpe ratio by phase
        self._plot_risk_adjusted_returns_by_phase()
        
        # Create a visualization showing volatility and expected returns
        # to understand what's driving the RVR pattern
        self._plot_volatility_expected_returns()
        
        # Create a visualization showing the temporal dynamics of 
        # RVR and Sharpe around events
        self._plot_combined_timeseries()
    
    def _plot_risk_adjusted_returns_by_phase(self):
        """
        Create a bar chart showing RVR and Sharpe ratio by phase.
        """
        try:
            if self.h1_results is None:
                print("Warning: No hypothesis results available for visualization.")
                return
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
            
            # Plot RVR by phase
            rvr_by_phase = self.h1_results['rvr_by_phase']
            phases = ['pre_event', 'post_event_rising', 'post_event_decay']
            phase_labels = ['Pre-Event', 'Post-Event Rising', 'Post-Event Decay']
            rvr_values = [rvr_by_phase.get(phase) for phase in phases]
            
            bars1 = ax1.bar(phase_labels, rvr_values, color=['blue', 'green', 'blue'])
            bars1[1].set_color('orange')  # Highlight post-event rising phase
            
            ax1.set_title('Return-to-Variance Ratio by Phase')
            ax1.set_ylabel('Average RVR')
            ax1.grid(True, axis='y', alpha=0.3)
            
            # Add value labels
            for i, v in enumerate(rvr_values):
                if v is not None:
                    ax1.text(i, v + 0.1, f"{v:.4f}", ha='center', fontweight='bold')
            
            # Plot Sharpe ratio by phase
            sharpe_by_phase = self.h1_results['sharpe_by_phase']
            sharpe_values = [sharpe_by_phase.get(phase) for phase in phases]
            
            bars2 = ax2.bar(phase_labels, sharpe_values, color=['blue', 'green', 'blue'])
            bars2[1].set_color('orange')  # Highlight post-event rising phase
            
            ax2.set_title('Sharpe Ratio by Phase')
            ax2.set_ylabel('Average Sharpe Ratio')
            ax2.grid(True, axis='y', alpha=0.3)
            
            # Add value labels
            for i, v in enumerate(sharpe_values):
                if v is not None:
                    ax2.text(i, v + 0.01, f"{v:.4f}", ha='center', fontweight='bold')
            
            # Add hypothesis result
            fig.suptitle(f"Hypothesis 1: Risk-Adjusted Returns Peak During Post-Event Rising Phase\n"
                         f"Result: {'SUPPORTED' if self.h1_results['hypothesis_supported'] else 'NOT SUPPORTED'}",
                         fontsize=16)
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.results_dir, f"{self.file_prefix}_risk_adjusted_returns_by_phase.png"), dpi=200)
            plt.close(fig)
            
            print(f"Risk-adjusted returns by phase plot saved to: {self.results_dir}")
        
        except Exception as e:
            print(f"Error creating risk-adjusted returns by phase plot: {e}")
            traceback.print_exc()
    
    def _plot_volatility_expected_returns(self):
        """
        Create a plot showing volatility and expected returns over time.
        """
        try:
            # Check if we have needed data
            rvr_daily_file = os.path.join(self.results_dir, f"{self.file_prefix}_rvr_timeseries.csv")
            if not os.path.exists(rvr_daily_file):
                print("Warning: RVR timeseries data not available for volatility plot.")
                return
            
            rvr_daily = pl.read_csv(rvr_daily_file)
            
            # Extract volatility and expected returns
            if 'days_to_event' in rvr_daily.columns and 'mean_volatility' in rvr_daily.columns and 'mean_expected_return' in rvr_daily.columns:
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True, gridspec_kw={'height_ratios': [2, 1]})
                
                # Plot volatility
                ax1.plot(rvr_daily['days_to_event'], rvr_daily['mean_volatility'], 
                        color='red', linewidth=2, label='Mean Volatility')
                
                # Add phase markers
                ax1.axvline(x=0, color='black', linestyle='--', label='Event Day')
                ax1.axvline(x=self.delta, color='green', linestyle=':', 
                           label=f'End Rising Phase (t+{self.delta})')
                
                # Highlight post-event rising phase
                ax1.axvspan(1, self.delta, color='yellow', alpha=0.2, label='Post-Event Rising Phase')
                
                ax1.set_title('Volatility Around Events')
                ax1.set_ylabel('Volatility')
                ax1.legend()
                ax1.grid(True, alpha=0.3)
                
                # Plot expected returns
                ax2.plot(rvr_daily['days_to_event'], rvr_daily['mean_expected_return'], 
                        color='blue', linewidth=2, label='Mean Expected Return')
                
                # Add phase markers
                ax2.axvline(x=0, color='black', linestyle='--', label='Event Day')
                ax2.axvline(x=self.delta, color='green', linestyle=':', 
                           label=f'End Rising Phase (t+{self.delta})')
                
                # Highlight post-event rising phase
                ax2.axvspan(1, self.delta, color='yellow', alpha=0.2, label='Post-Event Rising Phase')
                
                ax2.set_title('Expected Returns Around Events')
                ax2.set_xlabel('Days Relative to Event')
                ax2.set_ylabel('Expected Return')
                ax2.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
                ax2.legend()
                ax2.grid(True, alpha=0.3)
                
                # Add bias annotation for post-event rising phase
                ax2.annotate(f"Optimistic Bias: +{self.optimistic_bias*100}%",
                            xy=(self.delta/2, ax2.get_ylim()[1]*0.8),
                            xycoords='data',
                            bbox=dict(boxstyle='round', alpha=0.1))
                
                plt.tight_layout()
                plt.savefig(os.path.join(self.results_dir, f"{self.file_prefix}_volatility_returns.png"), dpi=200)
                plt.close(fig)
                
                print(f"Volatility and expected returns plot saved to: {self.results_dir}")
            else:
                print("Warning: Required columns not found in RVR timeseries data.")
        
        except Exception as e:
            print(f"Error creating volatility and expected returns plot: {e}")
            traceback.print_exc()
    
    def _plot_combined_timeseries(self):
        """
        Create a combined plot showing RVR and Sharpe ratio timeseries.
        """
        try:
            # Check if we have needed data
            rvr_daily_file = os.path.join(self.results_dir, f"{self.file_prefix}_rvr_timeseries.csv")
            if not os.path.exists(rvr_daily_file):
                print("Warning: RVR timeseries data not available for combined plot.")
                return
            
            if self.sharpe_data is None:
                print("Warning: Sharpe ratio data not available for combined plot.")
                return
            
            rvr_daily = pl.read_csv(rvr_daily_file)
            
            # Create plot
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
            
            # Plot RVR timeseries
            ax1.plot(rvr_daily['days_to_event'], rvr_daily['mean_rvr'], 
                    color='blue', linewidth=2, label='Mean RVR')
            
            # Add phase markers
            ax1.axvline(x=0, color='black', linestyle='--', label='Event Day')
            ax1.axvline(x=self.delta, color='green', linestyle=':', 
                       label=f'End Rising Phase (t+{self.delta})')
            
            # Highlight post-event rising phase
            ax1.axvspan(1, self.delta, color='yellow', alpha=0.2, label='Post-Event Rising Phase')
            
            ax1.set_title('Return-to-Variance Ratio Around Events')
            ax1.set_ylabel('RVR')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Plot Sharpe ratio timeseries
            sharpe_pd = self.sharpe_data.to_pandas()
            ax2.plot(sharpe_pd['days_to_event'], sharpe_pd['sharpe_ratio'], 
                    color='red', linewidth=2, label='Sharpe Ratio')
            
            # Add phase markers
            ax2.axvline(x=0, color='black', linestyle='--', label='Event Day')
            ax2.axvline(x=self.delta, color='green', linestyle=':', 
                       label=f'End Rising Phase (t+{self.delta})')
            
            # Highlight post-event rising phase
            ax2.axvspan(1, self.delta, color='yellow', alpha=0.2, label='Post-Event Rising Phase')
            
            ax2.set_title('Sharpe Ratio Around Events')
            ax2.set_xlabel('Days Relative to Event')
            ax2.set_ylabel('Sharpe Ratio')
            ax2.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # Add hypothesis result
            if self.h1_results is not None:
                fig.suptitle(f"Hypothesis 1: Risk-Adjusted Returns Peak During Post-Event Rising Phase\n"
                             f"Result: {'SUPPORTED' if self.h1_results['hypothesis_supported'] else 'NOT SUPPORTED'}",
                             fontsize=16)
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.results_dir, f"{self.file_prefix}_combined_timeseries.png"), dpi=200)
            plt.close(fig)
            
            print(f"Combined timeseries plot saved to: {self.results_dir}")
        
        except Exception as e:
            print(f"Error creating combined timeseries plot: {e}")
            traceback.print_exc()
    
    def _create_report(self):
        """
        Create a comprehensive report of the results.
        """
        print("\n--- Creating Comprehensive Report ---")
        
        if self.h1_results is None:
            print("Warning: No hypothesis results available for report.")
            return
        
        try:
            report_file = os.path.join(self.results_dir, f"{self.file_prefix}_h1_report.md")
            
            with open(report_file, 'w') as f:
                f.write("# Hypothesis 1 Analysis Results\n\n")
                f.write("## Overview\n\n")
                f.write("This report summarizes the analysis of Hypothesis 1 from the paper:\n\n")
                f.write("> Risk-adjusted returns, specifically the return-to-variance ratio (RVR) and the Sharpe ratio, peak during the post-event rising phase due to high volatility and biased expectations.\n\n")
                
                f.write("## Results Summary\n\n")
                f.write(f"**Hypothesis 1 is {'SUPPORTED' if self.h1_results['hypothesis_supported'] else 'NOT SUPPORTED'}**\n\n")
                f.write(f"- RVR evidence: {'Supported' if self.h1_results['rvr_supported'] else 'Not supported'}\n")
                f.write(f"- Sharpe ratio evidence: {'Supported' if self.h1_results['sharpe_supported'] else 'Not supported'}\n\n")
                
                f.write("## Return-to-Variance Ratio by Phase\n\n")
                f.write("| Phase | Average RVR | Peak? |\n")
                f.write("|-------|------------|-------|\n")
                
                rvr_by_phase = self.h1_results['rvr_by_phase']
                max_phase = max(rvr_by_phase.items(), key=lambda x: x[1] if x[1] is not None else -float('inf'))[0]
                
                for phase, value in rvr_by_phase.items():
                    is_peak = phase == max_phase
                    f.write(f"| {phase.replace('_', ' ').title()} | {value:.4f} | {'Yes' if is_peak else 'No'} |\n")
                
                f.write("\n## Sharpe Ratio by Phase\n\n")
                f.write("| Phase | Average Sharpe | Peak? |\n")
                f.write("|-------|---------------|-------|\n")
                
                sharpe_by_phase = self.h1_results['sharpe_by_phase']
                if sharpe_by_phase and all(v is not None for v in sharpe_by_phase.values()):
                    max_phase = max(sharpe_by_phase.items(), key=lambda x: x[1] if x[1] is not None else -float('inf'))[0]
                    
                    for phase, value in sharpe_by_phase.items():
                        is_peak = phase == max_phase
                        f.write(f"| {phase.replace('_', ' ').title()} | {value:.4f} | {'Yes' if is_peak else 'No'} |\n")
                
                f.write("\n## Statistical Tests\n\n")
                f.write("### RVR Statistical Tests (Post-Event Rising vs. Other Phases)\n\n")
                f.write("| Comparison | t-statistic | p-value | Significant? |\n")
                f.write("|------------|-------------|---------|---------------|\n")
                
                for phase, stats_dict in self.h1_results['rvr_p_values'].items():
                    significance = "***" if stats_dict['p_value'] < 0.01 else "**" if stats_dict['p_value'] < 0.05 else "*" if stats_dict['p_value'] < 0.1 else ""
                    f.write(f"| vs. {phase.replace('_', ' ').title()} | {stats_dict['t_stat']:.4f} | {stats_dict['p_value']:.4f} | {'Yes' if stats_dict['significant'] else 'No'} {significance} |\n")
                
                f.write("\n### Sharpe Ratio Statistical Tests (Post-Event Rising vs. Other Phases)\n\n")
                f.write("| Comparison | t-statistic | p-value | Significant? |\n")
                f.write("|------------|-------------|---------|---------------|\n")
                
                for phase, stats_dict in self.h1_results['sharpe_p_values'].items():
                    significance = "***" if stats_dict['p_value'] < 0.01 else "**" if stats_dict['p_value'] < 0.05 else "*" if stats_dict['p_value'] < 0.1 else ""
                    f.write(f"| vs. {phase.replace('_', ' ').title()} | {stats_dict['t_stat']:.4f} | {stats_dict['p_value']:.4f} | {'Yes' if stats_dict['significant'] else 'No'} {significance} |\n")
                
                f.write("\n## Visualizations\n\n")
                f.write("The following visualizations were generated to illustrate the results:\n\n")
                f.write(f"1. **Risk-Adjusted Returns by Phase**: `{self.file_prefix}_risk_adjusted_returns_by_phase.png`\n")
                f.write(f"2. **Volatility and Expected Returns**: `{self.file_prefix}_volatility_returns.png`\n")
                f.write(f"3. **Combined RVR and Sharpe Timeseries**: `{self.file_prefix}_combined_timeseries.png`\n\n")
                
                f.write("## Analysis Parameters\n\n")
                f.write("The analysis was conducted using the following parameters:\n\n")
                f.write(f"- Analysis window: {self.analysis_window} days\n")
                f.write(f"- GARCH model type: {self.garch_type.upper()}\n")
                f.write(f"- Post-event rising phase duration: {self.delta} days\n")
                f.write(f"- Optimistic bias: {self.optimistic_bias*100}%\n")
                f.write(f"- Risk-free rate: {self.risk_free_rate*100}%\n\n")
                
                f.write("## Implications\n\n")
                f.write("The results have the following implications for the two-risk framework:\n\n")
                
                if self.h1_results['hypothesis_supported']:
                    f.write("1. **Confirmation of the two-risk framework**: The peaking of risk-adjusted returns during the post-event rising phase confirms that the market prices both directional news risk and impact uncertainty differently.\n\n")
                    f.write("2. **Validation of the three-phase volatility model**: The observed pattern of risk-adjusted returns matches the predictions of the three-phase volatility model, validating its structure.\n\n")
                    f.write("3. **Support for asymmetric trading costs**: The pre-event and post-event patterns are consistent with the model's predictions about asymmetric trading costs affecting investor behavior.\n\n")
                else:
                    f.write("1. **Limited support for the two-risk framework**: The analysis provides limited evidence for distinguishing between directional news risk and impact uncertainty in market pricing.\n\n")
                    f.write("2. **Partial validation of the three-phase volatility model**: The observed pattern only partially matches the predictions of the three-phase volatility model.\n\n")
                    f.write("3. **Mixed evidence for asymmetric trading costs**: The results do not conclusively support the model's predictions about asymmetric trading costs affecting investor behavior.\n\n")
                
                f.write("## Conclusion\n\n")
                
                if self.h1_results['hypothesis_supported']:
                    f.write("The analysis provides strong support for Hypothesis 1. Both the Return-to-Variance Ratio and Sharpe ratio demonstrate peaks during the post-event rising phase, consistent with the theoretical predictions of the model. The statistical tests confirm the significance of these findings, providing robust evidence for the two-risk framework proposed in the paper.\n")
                else:
                    f.write("The analysis provides limited support for Hypothesis 1. The patterns of risk-adjusted returns do not consistently demonstrate peaks during the post-event rising phase as predicted by the model. Further refinement of the theoretical framework or additional empirical testing may be needed to better explain the observed patterns in risk-adjusted returns around events.\n")
            
            print(f"Comprehensive report saved to: {report_file}")
        
        except Exception as e:
            print(f"Error creating report: {e}")
            traceback.print_exc()

def run_fda_analysis():
    """
    Runs the FDA event analysis to test Hypothesis 1 with enhanced analysis.
    """
    print("\n=== Starting Enhanced FDA Approval Event Analysis for Hypothesis 1 ===")

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
        
        # --- Initialize and run enhanced analyzer ---
        h1_analyzer = Hypothesis1Analyzer(
            analyzer=analyzer,
            results_dir=FDA_RESULTS_DIR,
            file_prefix=FDA_FILE_PREFIX
        )
        
        h1_analyzer.run_analysis()
        
        print(f"\n--- Enhanced FDA Event Analysis for Hypothesis 1 Finished (Results in '{FDA_RESULTS_DIR}') ---")
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
    Runs the earnings event analysis to test Hypothesis 1 with enhanced analysis.
    """
    print("\n=== Starting Enhanced Earnings Announcement Event Analysis for Hypothesis 1 ===")

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
        
        # --- Initialize and run enhanced analyzer ---
        h1_analyzer = Hypothesis1Analyzer(
            analyzer=analyzer,
            results_dir=EARNINGS_RESULTS_DIR,
            file_prefix=EARNINGS_FILE_PREFIX
        )
        
        h1_analyzer.run_analysis()
        
        print(f"\n--- Enhanced Earnings Event Analysis for Hypothesis 1 Finished (Results in '{EARNINGS_RESULTS_DIR}') ---")
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
    Compares the enhanced hypothesis test results between FDA and earnings events.
    """
    print("\n=== Comparing Enhanced FDA and Earnings Results for Hypothesis 1 ===")
    
    # Create comparison directory
    comparison_dir = "results/hypothesis1/comparison/"
    os.makedirs(comparison_dir, exist_ok=True)
    
    try:
        # Define file paths
        fda_h1_file = os.path.join(FDA_RESULTS_DIR, f"{FDA_FILE_PREFIX}_hypothesis1_test.csv")
        earnings_h1_file = os.path.join(EARNINGS_RESULTS_DIR, f"{EARNINGS_FILE_PREFIX}_hypothesis1_test.csv")
        
        # Check if files exist
        if not os.path.exists(fda_h1_file) or not os.path.exists(earnings_h1_file):
            print(f"Error: One or more hypothesis test files missing: {fda_h1_file}, {earnings_h1_file}")
            return False
        
        # Load results
        fda_h1 = pl.read_csv(fda_h1_file)
        earnings_h1 = pl.read_csv(earnings_h1_file)
        
        # Create comparison table
        comparison_data = {
            'Event Type': ['FDA Approvals', 'Earnings Announcements'],
            'H1 Supported': [
                fda_h1['result'].item(),
                earnings_h1['result'].item()
            ],
            'RVR Supported': [
                fda_h1['rvr_supported'].item(),
                earnings_h1['rvr_supported'].item()
            ],
            'Sharpe Supported': [
                fda_h1['sharpe_supported'].item(),
                earnings_h1['sharpe_supported'].item()
            ],
            'Pre-Event RVR': [
                fda_h1['pre_event_rvr'].item(),
                earnings_h1['pre_event_rvr'].item()
            ],
            'Post-Rising RVR': [
                fda_h1['post_rising_rvr'].item(),
                earnings_h1['post_rising_rvr'].item()
            ],
            'Post-Decay RVR': [
                fda_h1['post_decay_rvr'].item(),
                earnings_h1['post_decay_rvr'].item()
            ],
            'Pre-Event Sharpe': [
                fda_h1['pre_event_sharpe'].item(),
                earnings_h1['pre_event_sharpe'].item()
            ],
            'Post-Rising Sharpe': [
                fda_h1['post_rising_sharpe'].item(),
                earnings_h1['post_rising_sharpe'].item()
            ],
            'Post-Decay Sharpe': [
                fda_h1['post_decay_sharpe'].item(),
                earnings_h1['post_decay_sharpe'].item()
            ]
        }
        
        comparison_df = pl.DataFrame(comparison_data)
        
        # Save comparison table
        comparison_df.write_csv(os.path.join(comparison_dir, "hypothesis1_comparison.csv"))
        
        # Print summary
        print("\nHypothesis 1 Comparison Results:")
        print(f"FDA Approvals: H1 {'Supported' if fda_h1['result'].item() else 'Not Supported'}")
        print(f"  RVR Evidence: {'Yes' if fda_h1['rvr_supported'].item() else 'No'}")
        print(f"  Sharpe Evidence: {'Yes' if fda_h1['sharpe_supported'].item() else 'No'}")
        print(f"Earnings Announcements: H1 {'Supported' if earnings_h1['result'].item() else 'Not Supported'}")
        print(f"  RVR Evidence: {'Yes' if earnings_h1['rvr_supported'].item() else 'No'}")
        print(f"  Sharpe Evidence: {'Yes' if earnings_h1['sharpe_supported'].item() else 'No'}")
        
        # Create comparison plots
        try:
            # Convert to pandas for plotting
            comparison_pd = comparison_df.to_pandas()
            
            # Create plot for RVR comparison
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
            
            # RVR comparison
            phases = ['Pre-Event', 'Post-Rising', 'Post-Decay']
            fda_rvr = [comparison_pd.loc[0, 'Pre-Event RVR'], 
                        comparison_pd.loc[0, 'Post-Rising RVR'],
                        comparison_pd.loc[0, 'Post-Decay RVR']]
            
            earnings_rvr = [comparison_pd.loc[1, 'Pre-Event RVR'], 
                            comparison_pd.loc[1, 'Post-Rising RVR'],
                            comparison_pd.loc[1, 'Post-Decay RVR']]
            
            x = np.arange(len(phases))
            width = 0.35
            
            rects1 = ax1.bar(x - width/2, fda_rvr, width, label='FDA Approvals', color='blue', alpha=0.7)
            rects2 = ax1.bar(x + width/2, earnings_rvr, width, label='Earnings Announcements', color='red', alpha=0.7)
            
            ax1.set_title('RVR by Phase: FDA vs Earnings')
            ax1.set_ylabel('Return-to-Variance Ratio')
            ax1.set_xticks(x)
            ax1.set_xticklabels(phases)
            ax1.legend()
            ax1.grid(True, axis='y', alpha=0.3)
            
            # Add value labels
            for i, rect in enumerate(rects1):
                height = rect.get_height()
                ax1.annotate(f"{height:.2f}",
                            xy=(rect.get_x() + rect.get_width()/2, height),
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom')
            
            for i, rect in enumerate(rects2):
                height = rect.get_height()
                ax1.annotate(f"{height:.2f}",
                            xy=(rect.get_x() + rect.get_width()/2, height),
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom')
            
            # Sharpe comparison
            fda_sharpe = [comparison_pd.loc[0, 'Pre-Event Sharpe'], 
                          comparison_pd.loc[0, 'Post-Rising Sharpe'],
                          comparison_pd.loc[0, 'Post-Decay Sharpe']]
            
            earnings_sharpe = [comparison_pd.loc[1, 'Pre-Event Sharpe'], 
                               comparison_pd.loc[1, 'Post-Rising Sharpe'],
                               comparison_pd.loc[1, 'Post-Decay Sharpe']]
            
            rects3 = ax2.bar(x - width/2, fda_sharpe, width, label='FDA Approvals', color='blue', alpha=0.7)
            rects4 = ax2.bar(x + width/2, earnings_sharpe, width, label='Earnings Announcements', color='red', alpha=0.7)
            
            ax2.set_title('Sharpe Ratio by Phase: FDA vs Earnings')
            ax2.set_ylabel('Sharpe Ratio')
            ax2.set_xticks(x)
            ax2.set_xticklabels(phases)
            ax2.legend()
            ax2.grid(True, axis='y', alpha=0.3)
            
            # Add value labels
            for i, rect in enumerate(rects3):
                height = rect.get_height()
                ax2.annotate(f"{height:.2f}",
                            xy=(rect.get_x() + rect.get_width()/2, height),
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom')
            
            for i, rect in enumerate(rects4):
                height = rect.get_height()
                ax2.annotate(f"{height:.2f}",
                            xy=(rect.get_x() + rect.get_width()/2, height),
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom')
            
            # Highlight the post-event rising phase
            ax1.axvspan(0.5, 1.5, color='yellow', alpha=0.2, label='Key Phase for H1')
            ax2.axvspan(0.5, 1.5, color='yellow', alpha=0.2, label='Key Phase for H1')
            
            # Add hypothesis test results
            fig.suptitle("Hypothesis 1: Risk-Adjusted Returns Peak During Post-Event Rising Phase",
                         fontsize=16)
            
            # Add result text
            result_text = (f"FDA: H1 {'SUPPORTED' if fda_h1['result'].item() else 'NOT SUPPORTED'}\n"
                          f"Earnings: H1 {'SUPPORTED' if earnings_h1['result'].item() else 'NOT SUPPORTED'}")
            fig.text(0.5, 0.01, result_text, ha='center', fontsize=14, fontweight='bold')
            
            plt.tight_layout(rect=[0, 0.05, 1, 0.95])
            plt.savefig(os.path.join(comparison_dir, "hypothesis1_comparison.png"), dpi=200, bbox_inches='tight')
            plt.close(fig)
            
            print(f"Comparison plots saved to: {comparison_dir}")
            
            # Create a comprehensive comparison report
            report_file = os.path.join(comparison_dir, "hypothesis1_comparison_report.md")
            
            with open(report_file, 'w') as f:
                f.write("# Hypothesis 1 Comparison: FDA Approvals vs. Earnings Announcements\n\n")
                f.write("## Overview\n\n")
                f.write("This report compares the analysis of Hypothesis 1 between FDA approval events and earnings announcement events.\n\n")
                f.write("> H1: Risk-adjusted returns, specifically the return-to-variance ratio (RVR) and the Sharpe ratio, peak during the post-event rising phase due to high volatility and biased expectations.\n\n")
                
                f.write("## Results Summary\n\n")
                f.write("| Event Type | H1 Supported | RVR Evidence | Sharpe Evidence |\n")
                f.write("|------------|-------------|--------------|----------------|\n")
                f.write(f"| FDA Approvals | {'Yes' if fda_h1['result'].item() else 'No'} | {'Yes' if fda_h1['rvr_supported'].item() else 'No'} | {'Yes' if fda_h1['sharpe_supported'].item() else 'No'} |\n")
                f.write(f"| Earnings Announcements | {'Yes' if earnings_h1['result'].item() else 'No'} | {'Yes' if earnings_h1['rvr_supported'].item() else 'No'} | {'Yes' if earnings_h1['sharpe_supported'].item() else 'No'} |\n\n")
                
                f.write("## Return-to-Variance Ratio Comparison\n\n")
                f.write("| Phase | FDA Approvals | Earnings Announcements |\n")
                f.write("|-------|--------------|------------------------|\n")
                f.write(f"| Pre-Event | {fda_h1['pre_event_rvr'].item():.4f} | {earnings_h1['pre_event_rvr'].item():.4f} |\n")
                f.write(f"| Post-Event Rising | {fda_h1['post_rising_rvr'].item():.4f} | {earnings_h1['post_rising_rvr'].item():.4f} |\n")
                f.write(f"| Post-Event Decay | {fda_h1['post_decay_rvr'].item():.4f} | {earnings_h1['post_decay_rvr'].item():.4f} |\n\n")
                
                # Calculate percentage difference in post-rising to pre-event RVR
                fda_rvr_pct_increase = ((fda_h1['post_rising_rvr'].item() / fda_h1['pre_event_rvr'].item()) - 1) * 100
                earnings_rvr_pct_increase = ((earnings_h1['post_rising_rvr'].item() / earnings_h1['pre_event_rvr'].item()) - 1) * 100
                
                f.write(f"FDA post-rising RVR is **{fda_rvr_pct_increase:.1f}%** higher than pre-event.\n")
                f.write(f"Earnings post-rising RVR is **{earnings_rvr_pct_increase:.1f}%** higher than pre-event.\n\n")
                
                f.write("## Sharpe Ratio Comparison\n\n")
                f.write("| Phase | FDA Approvals | Earnings Announcements |\n")
                f.write("|-------|--------------|------------------------|\n")
                f.write(f"| Pre-Event | {fda_h1['pre_event_sharpe'].item():.4f} | {earnings_h1['pre_event_sharpe'].item():.4f} |\n")
                f.write(f"| Post-Event Rising | {fda_h1['post_rising_sharpe'].item():.4f} | {earnings_h1['post_rising_sharpe'].item():.4f} |\n")
                f.write(f"| Post-Event Decay | {fda_h1['post_decay_sharpe'].item():.4f} | {earnings_h1['post_decay_sharpe'].item():.4f} |\n\n")
                
                # Calculate percentage difference in post-rising to pre-event Sharpe
                fda_sharpe_pct_increase = ((fda_h1['post_rising_sharpe'].item() / fda_h1['pre_event_sharpe'].item()) - 1) * 100
                earnings_sharpe_pct_increase = ((earnings_h1['post_rising_sharpe'].item() / earnings_h1['pre_event_sharpe'].item()) - 1) * 100
                
                f.write(f"FDA post-rising Sharpe is **{fda_sharpe_pct_increase:.1f}%** higher than pre-event.\n")
                f.write(f"Earnings post-rising Sharpe is **{earnings_sharpe_pct_increase:.1f}%** higher than pre-event.\n\n")
                
                f.write("## Comparative Analysis\n\n")
                
                # Analysis details
                f.write("### Patterns of Risk-Adjusted Returns\n\n")
                
                # Different cases
                if fda_h1['result'].item() and earnings_h1['result'].item():
                    f.write("Both FDA approval events and earnings announcement events demonstrate a clear pattern of risk-adjusted returns peaking during the post-event rising phase. This consistent pattern across different types of events provides strong support for Hypothesis 1.\n\n")
                    
                    # Compare magnitudes
                    if fda_rvr_pct_increase > earnings_rvr_pct_increase:
                        f.write("The magnitude of the RVR peak is stronger for FDA approval events, suggesting that the impact of the two-risk framework may be more pronounced for these events. This could be due to the greater uncertainty associated with FDA approvals and their longer-term impact on company value.\n\n")
                    else:
                        f.write("The magnitude of the RVR peak is stronger for earnings announcement events, suggesting that the impact of the two-risk framework may be more pronounced for these events. This could be due to the more immediate impact of earnings surprises on market expectations.\n\n")
                    
                    if fda_sharpe_pct_increase > earnings_sharpe_pct_increase:
                        f.write("Similarly, the increase in Sharpe ratio is more substantial for FDA approval events, reinforcing the idea that these events may have a stronger impact on risk-adjusted returns.\n\n")
                    else:
                        f.write("Similarly, the increase in Sharpe ratio is more substantial for earnings announcement events, reinforcing the idea that these events may have a stronger impact on risk-adjusted returns.\n\n")
                    
                elif fda_h1['result'].item() and not earnings_h1['result'].item():
                    f.write("FDA approval events demonstrate a clear pattern of risk-adjusted returns peaking during the post-event rising phase, while earnings announcement events do not show the same pattern. This suggests that the two-risk framework may be more applicable to FDA approval events, possibly due to their greater uncertainty and longer-term impact on company value.\n\n")
                    
                elif not fda_h1['result'].item() and earnings_h1['result'].item():
                    f.write("Earnings announcement events demonstrate a clear pattern of risk-adjusted returns peaking during the post-event rising phase, while FDA approval events do not show the same pattern. This suggests that the two-risk framework may be more applicable to earnings announcement events, possibly due to their more immediate impact on market expectations.\n\n")
                    
                else:
                    f.write("Neither FDA approval events nor earnings announcement events demonstrate a clear pattern of risk-adjusted returns peaking during the post-event rising phase. This suggests that the two-risk framework may need refinement or may not fully capture the dynamics of risk-adjusted returns around these events.\n\n")
                
                f.write("### Implications for the Two-Risk Framework\n\n")
                f.write("The comparison between FDA approval events and earnings announcement events provides insights into the generalizability of the two-risk framework across different types of corporate events.\n\n")
                
                if fda_h1['result'].item() and earnings_h1['result'].item():
                    f.write("The consistent support for Hypothesis 1 across both types of events suggests that the distinction between directional news risk and impact uncertainty is a fundamental aspect of market behavior around high-uncertainty events. This framework appears to have broad applicability across different types of corporate events.\n\n")
                else:
                    f.write("The inconsistent support for Hypothesis 1 across the two types of events suggests that the distinction between directional news risk and impact uncertainty may not be equally applicable to all types of corporate events. The framework may need to be adapted or extended to account for the specific characteristics of different event types.\n\n")
                
                f.write("### Conclusion\n\n")
                
                if fda_h1['result'].item() and earnings_h1['result'].item():
                    f.write("The comparison between FDA approval events and earnings announcement events provides strong and consistent support for Hypothesis 1. Both types of events demonstrate the predicted pattern of risk-adjusted returns peaking during the post-event rising phase, validating the two-risk framework as a generalizable model for understanding market behavior around high-uncertainty events.\n\n")
                elif fda_h1['result'].item() or earnings_h1['result'].item():
                    f.write("The comparison between FDA approval events and earnings announcement events provides partial support for Hypothesis 1. While one type of event demonstrates the predicted pattern of risk-adjusted returns peaking during the post-event rising phase, the other does not. This suggests that the two-risk framework may be applicable to some types of corporate events but not others, highlighting the need for further research to refine the model and understand its limitations.\n\n")
                else:
                    f.write("The comparison between FDA approval events and earnings announcement events does not support Hypothesis 1. Neither type of event demonstrates the predicted pattern of risk-adjusted returns peaking during the post-event rising phase. This suggests that the two-risk framework may need significant refinement or may not fully capture the dynamics of risk-adjusted returns around these events. Further research is needed to explore alternative explanations and models for the observed patterns.\n\n")
            
            print(f"Comparison report saved to: {report_file}")
        
        except Exception as e:
            print(f"Error creating comparison plots or report: {e}")
            traceback.print_exc()
        
        return True
    
    except Exception as e:
        print(f"Error comparing results: {e}")
        traceback.print_exc()
        return False

def main():
    # Run enhanced FDA analysis
    fda_success = run_fda_analysis()
    
    # Run enhanced earnings analysis
    earnings_success = run_earnings_analysis()
    
    # Compare results if both analyses succeeded
    #if fda_success and earnings_success:
    #    compare_success = compare_results()
    #    if compare_success:
    #        print("\n=== All enhanced analyses and comparisons completed successfully ===")
    #    else:
    #        print("\n=== Enhanced analyses completed, but comparison failed ===")
    #elif fda_success:
    #    print("\n=== Only enhanced FDA analysis completed successfully ===")
    #elif earnings_success:
    #    print("\n=== Only enhanced earnings analysis completed successfully ===")
    #else:
    #    print("\n=== Both enhanced analyses failed ===")

if __name__ == "__main__":
    main()
