"""
Test Hypothesis 1: Risk-Adjusted Returns Peak During Post-Event Rising Phase

Tests the hypothesis that Return-to-Variance Ratio (RVR) and Sharpe ratios 
peak during the post-event rising phase (t_event < t ≤ t_event + δ) due to 
high volatility and biased expectations.

Based on Proposition 3 from "Modeling Equilibrium Asset Pricing Around Events 
with Heterogeneous Beliefs, Dynamic Volatility, and a Two-Risk Uncertainty Framework"
"""

import pandas as pd
import numpy as np
import os
import sys
import polars as pl
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from typing import List, Tuple, Dict, Optional, Any
from datetime import datetime, timedelta

# Add src directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)
src_dir = os.path.join(current_dir, 'src')
if src_dir not in sys.path:
    sys.path.append(src_dir)

try:
    from src.event_processor import EventProcessor, VolatilityParameters
    from src.models import RiskMetrics, UnifiedVolatilityModel
    from src.config import Config
    print("Successfully imported required modules.")
except ImportError as e:
    print(f"Error importing modules: {e}")
    print(f"Python Path: {sys.path}")
    sys.exit(1)

# Configure plotting style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Configure Polars display
pl.Config.set_tbl_rows(20)
pl.Config.set_tbl_cols(12)
pl.Config.set_fmt_str_lengths(100)

# Data file paths
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

# FDA event parameters
FDA_EVENT_FILE = "/home/d87016661/fda_ticker_list_2000_to_2024.csv"
FDA_RESULTS_DIR = "results/hypothesis1/fda/"
FDA_EVENT_DATE_COL = "Approval Date"
FDA_TICKER_COL = "ticker"

# Earnings event parameters
EARNINGS_EVENT_FILE = "/home/d87016661/detail_history_actuals.csv"
EARNINGS_RESULTS_DIR = "results/hypothesis1/earnings/"
EARNINGS_EVENT_DATE_COL = "ANNDATS"
EARNINGS_TICKER_COL = "ticker"

class Hypothesis1Tester:
    """
    Class to test Hypothesis 1 from the paper:
    Risk-adjusted returns (RVR and Sharpe) peak during post-event rising phase.
    """
    
    def __init__(self, config: Config, event_type: str, results_dir: str):
        """
        Initialize tester with configuration and event type.
        
        Args:
            config: Configuration object
            event_type: 'fda' or 'earnings'
            results_dir: Directory to save results
        """
        self.config = config
        self.event_type = event_type
        self.results_dir = results_dir
        self.event_processor = EventProcessor(config)
        
        # Create results directory
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Define event phases based on paper
        self.phases = {
            'pre_event_early': (-config.event_window_pre, -6),
            'pre_event_late': (-5, 0),
            'post_event_rising': (1, config.event_delta),
            'post_event_decay': (config.event_delta + 1, config.event_window_post)
        }
        
        # Storage for results
        self.processed_data = None
        self.phase_metrics = None
        self.hypothesis_results = None
    
    def load_stock_data(self, file_paths: List[str]) -> Optional[pl.DataFrame]:
        """Load and combine stock price data from multiple files."""
        print(f"Loading stock data from {len(file_paths)} files...")
        
        dfs = []
        for i, path in enumerate(file_paths):
            if i % 2 == 0:
                print(f"  Loading file {i+1}/{len(file_paths)}...")
            
            try:
                df_raw = pl.read_parquet(path) # Read raw dataframe
                
                # Identify source column names
                source_ticker_col = None
                source_date_col = None
                source_price_col = None
                source_returns_col = None

                # Create a mapping of standardized uppercase names to actual column names found
                col_map_upper = {col.upper(): col for col in df_raw.columns}

                # For ticker: Prioritize PERMNO, then TICKER, then SYMBOL
                if 'PERMNO' in col_map_upper:
                    source_ticker_col = col_map_upper['PERMNO']
                elif 'TICKER' in col_map_upper:
                    source_ticker_col = col_map_upper['TICKER']
                elif 'SYMBOL' in col_map_upper:
                    source_ticker_col = col_map_upper['SYMBOL']
                
                # For date:
                if 'DATE' in col_map_upper:
                    source_date_col = col_map_upper['DATE']
                
                # For price: Prioritize PRC, then PRICE
                if 'PRC' in col_map_upper:
                    source_price_col = col_map_upper['PRC']
                elif 'PRICE' in col_map_upper:
                    source_price_col = col_map_upper['PRICE']

                # For returns: Prioritize RET, then RETURNS, then RETURN
                if 'RET' in col_map_upper:
                    source_returns_col = col_map_upper['RET']
                elif 'RETURNS' in col_map_upper:
                    source_returns_col = col_map_upper['RETURNS']
                elif 'RETURN' in col_map_upper: # Some files might use singular
                    source_returns_col = col_map_upper['RETURN']

                # Check if all required source columns were found
                missing_cols = []
                if not source_ticker_col: missing_cols.append("ticker (PERMNO/TICKER/SYMBOL)")
                if not source_date_col: missing_cols.append("date (DATE)")
                if not source_price_col: missing_cols.append("price (PRC/PRICE)")
                if not source_returns_col: missing_cols.append("returns (RET/RETURNS/RETURN)")

                if missing_cols:
                    print(f"  Warning: Could not find all required source columns in {path}. Missing: {', '.join(missing_cols)}.")
                    print(f"    Available columns (original): {df_raw.columns}")
                    continue

                # Select and rename in one go using expressions
                # This ensures no intermediate duplicate names are created.
                df = df_raw.select([
                    pl.col(source_ticker_col).alias('ticker'),
                    pl.col(source_date_col).alias('date'),
                    pl.col(source_price_col).alias('price'),
                    pl.col(source_returns_col).alias('returns')
                ])
                
                # Now cast
                df = df.with_columns([
                    pl.col('ticker').cast(pl.Utf8),
                    pl.col('date').cast(pl.Date),
                    pl.col('price').cast(pl.Float64).abs(),  # Handle negative prices
                    pl.col('returns').cast(pl.Float64)
                ])
                dfs.append(df)
                    
            except Exception as e:
                # More specific error for loading vs processing
                print(f"  Warning: Error processing file {path}: {e}")
                continue
        
        if not dfs:
            print("Error: No stock data loaded!")
            return None
        
        # Combine all dataframes
        combined_df = pl.concat(dfs)
        
        # Remove duplicates and invalid data
        combined_df = combined_df.filter(
            pl.col('price').is_not_null() & 
            pl.col('returns').is_not_null() &
            (pl.col('price') > 0)
        ).unique(subset=['ticker', 'date'])
        
        print(f"Loaded {combined_df.height:,} stock observations")
        return combined_df
    
    def load_event_data(self, file_path: str, date_col: str, 
                       ticker_col: str) -> Optional[pl.DataFrame]:
        """Load event data from CSV file."""
        print(f"Loading {self.event_type} event data...")
        
        try:
            # Read CSV with proper date parsing
            df = pl.read_csv(file_path, try_parse_dates=True, infer_schema_length=10000)
            
            # Find columns (case-insensitive)
            actual_date_col = None
            actual_ticker_col = None
            
            for col in df.columns:
                if date_col.lower() in col.lower() and actual_date_col is None:
                    actual_date_col = col
                if ticker_col.lower() in col.lower() and actual_ticker_col is None:
                    actual_ticker_col = col
            
            if not actual_date_col or not actual_ticker_col:
                print(f"Error: Could not find required columns {date_col}, {ticker_col}")
                print(f"Available columns: {df.columns}")
                return None
            
            # Rename and process
            df = df.rename({
                actual_date_col: 'event_date',
                actual_ticker_col: 'symbol'
            })
            
            # Add event type
            df = df.with_columns([
                pl.col('event_date').cast(pl.Date),
                pl.col('symbol').cast(pl.Utf8).str.strip_chars(),
                pl.lit(self.event_type).alias('event_type')
            ])
            
            # Remove invalid events
            df = df.filter(
                pl.col('event_date').is_not_null() &
                pl.col('symbol').is_not_null() &
                (pl.col('symbol').str.len_chars() > 0)
            ).unique(subset=['symbol', 'event_date'])
            
            print(f"Loaded {df.height:,} {self.event_type} events")
            return df
            
        except Exception as e:
            print(f"Error loading event data: {e}")
            return None
    
    def run_analysis(self, stock_df: pl.DataFrame, event_df: pl.DataFrame) -> bool:
        """
        Run the complete Hypothesis 1 analysis.
        
        Returns:
            True if analysis completed successfully
        """
        print(f"\n{'='*60}")
        print(f"Running Hypothesis 1 Analysis for {self.event_type.upper()} Events")
        print(f"{'='*60}")
        
        # Process events using EventProcessor
        print("\nProcessing events with unified volatility model...")
        self.processed_data = self.event_processor.process_events(
            price_data=stock_df.to_pandas(), # EventProcessor expects pandas
            event_data=event_df.to_pandas()  # EventProcessor expects pandas
        )
        
        if self.processed_data is None or self.processed_data.empty:
            print("Error: No events successfully processed")
            return False
        
        print(f"Processed {len(self.processed_data):,} event-day observations")
        
        # Calculate phase-level metrics
        self._calculate_phase_metrics()
        
        # Test hypothesis
        self._test_hypothesis()
        
        # Generate visualizations
        self._create_visualizations()
        
        # Save results
        self._save_results()
        
        return True
    
    def _calculate_phase_metrics(self):
        """Calculate average metrics by event phase."""
        print("\nCalculating phase-level metrics...")
        
        # Add phase labels to data
        def get_phase(days_to_event):
            for phase_name, (start, end) in self.phases.items():
                if start <= days_to_event <= end:
                    return phase_name
            return 'other'
        
        self.processed_data['phase'] = self.processed_data['days_to_event'].apply(get_phase)
        
        # Calculate metrics by phase
        phase_results = []
        
        for phase_name, (start, end) in self.phases.items():
            phase_data = self.processed_data[
                (self.processed_data['days_to_event'] >= start) &
                (self.processed_data['days_to_event'] <= end)
            ]
            
            if len(phase_data) > 0:
                # Calculate average metrics
                metrics = {
                    'phase': phase_name,
                    'n_obs': len(phase_data),
                    'avg_return': phase_data['expected_return'].mean(),
                    'median_return': phase_data['expected_return'].median(),
                    'avg_volatility': phase_data['unified_volatility'].mean(),
                    'avg_rvr': phase_data['rvr'].mean(),
                    'median_rvr': phase_data['rvr'].median(),
                    'avg_sharpe': phase_data['sharpe_ratio'].mean(),
                    'median_sharpe': phase_data['sharpe_ratio'].median(),
                    'avg_bias': phase_data['bias_parameter'].mean(),
                    'avg_phi': phase_data['phi_adjustment'].mean()
                }
                
                # Add confidence intervals using bootstrap
                n_bootstrap = 1000
                rvr_samples = []
                sharpe_samples = []
                
                for _ in range(n_bootstrap):
                    sample = phase_data.sample(n=len(phase_data), replace=True)
                    rvr_samples.append(sample['rvr'].mean())
                    sharpe_samples.append(sample['sharpe_ratio'].mean())
                
                metrics['rvr_ci_lower'] = np.percentile(rvr_samples, 2.5)
                metrics['rvr_ci_upper'] = np.percentile(rvr_samples, 97.5)
                metrics['sharpe_ci_lower'] = np.percentile(sharpe_samples, 2.5)
                metrics['sharpe_ci_upper'] = np.percentile(sharpe_samples, 97.5)
                
                phase_results.append(metrics)
        
        self.phase_metrics = pd.DataFrame(phase_results)
        
        # Display results
        print("\nPhase Metrics Summary:")
        print(self.phase_metrics[['phase', 'n_obs', 'avg_rvr', 'avg_sharpe']].to_string(index=False))
    
    def _test_hypothesis(self):
        """
        Test Hypothesis 1: RVR and Sharpe peak during post-event rising phase.
        """
        print("\nTesting Hypothesis 1...")
        
        results = {}
        if self.phase_metrics is None or self.phase_metrics.empty:
            print("Warning: Phase metrics not calculated. Skipping hypothesis test.")
            self.hypothesis_results = results
            return

        # Get metrics for each phase
        metrics_dict = {
            row['phase']: row for _, row in self.phase_metrics.iterrows()
        }
        
        # Test for RVR peak
        if 'post_event_rising' in metrics_dict:
            rising_rvr = metrics_dict['post_event_rising']['avg_rvr']
            
            # Check if rising phase has highest RVR
            rvr_peak = True
            rvr_comparisons = {}
            
            for phase in ['pre_event_early', 'pre_event_late', 'post_event_decay']:
                if phase in metrics_dict:
                    other_rvr = metrics_dict[phase]['avg_rvr']
                    rvr_comparisons[phase] = other_rvr < rising_rvr
                    if other_rvr >= rising_rvr:
                        rvr_peak = False
            
            # Statistical tests
            rvr_stats = self._perform_statistical_tests('rvr', 'post_event_rising')
            
            results['rvr'] = {
                'peak_in_rising_phase': rvr_peak,
                'rising_phase_value': rising_rvr,
                'comparisons': rvr_comparisons,
                'statistical_tests': rvr_stats
            }
        
        # Test for Sharpe ratio peak
        if 'post_event_rising' in metrics_dict:
            rising_sharpe = metrics_dict['post_event_rising']['avg_sharpe']
            
            # Check if rising phase has highest Sharpe
            sharpe_peak = True
            sharpe_comparisons = {}
            
            for phase in ['pre_event_early', 'pre_event_late', 'post_event_decay']:
                if phase in metrics_dict:
                    other_sharpe = metrics_dict[phase]['avg_sharpe']
                    sharpe_comparisons[phase] = other_sharpe < rising_sharpe
                    if other_sharpe >= rising_sharpe:
                        sharpe_peak = False
            
            # Statistical tests
            sharpe_stats = self._perform_statistical_tests('sharpe_ratio', 'post_event_rising')
            
            results['sharpe'] = {
                'peak_in_rising_phase': sharpe_peak,
                'rising_phase_value': rising_sharpe,
                'comparisons': sharpe_comparisons,
                'statistical_tests': sharpe_stats
            }
        
        self.hypothesis_results = results
        
        # Print summary
        print("\nHypothesis 1 Test Results:")
        print(f"  RVR peaks in post-event rising phase: {results.get('rvr', {}).get('peak_in_rising_phase', 'N/A')}")
        print(f"  Sharpe peaks in post-event rising phase: {results.get('sharpe', {}).get('peak_in_rising_phase', 'N/A')}")
    
    def _perform_statistical_tests(self, metric: str, test_phase: str) -> Dict[str, Any]:
        """Perform statistical tests comparing phases."""
        if self.processed_data is None or self.processed_data.empty:
            return {}
            
        test_data = self.processed_data[self.processed_data['phase'] == test_phase][metric].dropna().values
        results = {}
        
        for compare_phase in ['pre_event_early', 'pre_event_late', 'post_event_decay']:
            compare_data = self.processed_data[self.processed_data['phase'] == compare_phase][metric].dropna().values
            
            if len(test_data) > 1 and len(compare_data) > 1:
                # T-test (one-sided, testing if rising > other)
                t_stat, p_value = stats.ttest_ind(test_data, compare_data, 
                                                 equal_var=False, alternative='greater')
                
                # Mann-Whitney U test (non-parametric alternative)
                u_stat, u_pvalue = stats.mannwhitneyu(test_data, compare_data, 
                                                      alternative='greater')
                
                results[compare_phase] = {
                    't_statistic': t_stat,
                    't_pvalue': p_value,
                    'u_statistic': u_stat,
                    'u_pvalue': u_pvalue,
                    'significant_at_5pct': p_value < 0.05
                }
        
        return results
    
    def _create_visualizations(self):
        """Create plots for Hypothesis 1 analysis."""
        print("\nGenerating visualizations...")
        if self.processed_data is None or self.processed_data.empty:
            print("Warning: No processed data to visualize.")
            return
        if self.phase_metrics is None or self.phase_metrics.empty:
            print("Warning: No phase metrics to visualize.")
            return

        # 1. Time series plot of RVR and Sharpe around events
        self._plot_time_series()
        
        # 2. Phase comparison bar plots
        self._plot_phase_comparison()
        
        # 3. Distribution plots by phase
        self._plot_distributions()
        
        # 4. Volatility decomposition plot
        self._plot_volatility_decomposition()
    
    def _plot_time_series(self):
        """Plot average RVR and Sharpe ratio around events."""
        # Calculate daily averages
        daily_avg = self.processed_data.groupby('days_to_event').agg({
            'rvr': ['mean', 'std', 'count'],
            'sharpe_ratio': ['mean', 'std', 'count'],
            'unified_volatility': 'mean',
            'phi_adjustment': 'mean'
        }).reset_index()
        
        # Flatten column names
        daily_avg.columns = ['_'.join(col).strip('_') for col in daily_avg.columns.values]
        
        # Create figure
        fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
        
        # Plot RVR
        ax = axes[0]
        ax.plot(daily_avg['days_to_event'], daily_avg['rvr_mean'], 
                color='blue', linewidth=2, label='RVR')
        
        # Add confidence bands
        sem = daily_avg['rvr_std'] / np.sqrt(daily_avg['rvr_count'])
        ax.fill_between(daily_avg['days_to_event'],
                       daily_avg['rvr_mean'] - 1.96 * sem,
                       daily_avg['rvr_mean'] + 1.96 * sem,
                       alpha=0.3, color='blue')
        
        # Mark phases
        rising_phase_label_added = False
        for phase, (start, end) in self.phases.items():
            if 'rising' in phase:
                if not rising_phase_label_added:
                    ax.axvspan(start, end, alpha=0.2, color='red', label='Post-event rising')
                    rising_phase_label_added = True
                else:
                    ax.axvspan(start, end, alpha=0.2, color='red')

        ax.axvline(0, color='black', linestyle='--', label='Event day')
        ax.set_ylabel('Return-to-Variance Ratio')
        ax.set_title(f'{self.event_type.upper()}: Average RVR Around Events')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot Sharpe ratio
        ax = axes[1]
        ax.plot(daily_avg['days_to_event'], daily_avg['sharpe_ratio_mean'],
                color='green', linewidth=2, label='Sharpe Ratio')
        
        sem = daily_avg['sharpe_ratio_std'] / np.sqrt(daily_avg['sharpe_ratio_count'])
        ax.fill_between(daily_avg['days_to_event'],
                       daily_avg['sharpe_ratio_mean'] - 1.96 * sem,
                       daily_avg['sharpe_ratio_mean'] + 1.96 * sem,
                       alpha=0.3, color='green')
        
        for phase, (start, end) in self.phases.items():
            if 'rising' in phase:
                ax.axvspan(start, end, alpha=0.2, color='red')
        
        ax.axvline(0, color='black', linestyle='--')
        ax.set_ylabel('Sharpe Ratio')
        ax.set_title(f'{self.event_type.upper()}: Average Sharpe Ratio Around Events')
        ax.grid(True, alpha=0.3)
        
        # Plot volatility adjustment (phi)
        ax = axes[2]
        ax.plot(daily_avg['days_to_event'], daily_avg['phi_adjustment_mean'],
                color='purple', linewidth=2, label='Phi adjustment')
        
        for phase, (start, end) in self.phases.items():
            if 'rising' in phase:
                ax.axvspan(start, end, alpha=0.2, color='red')
        
        ax.axvline(0, color='black', linestyle='--')
        ax.set_xlabel('Days to Event')
        ax.set_ylabel('Volatility Adjustment (φ)')
        ax.set_title('Event-Specific Volatility Adjustment')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, f'{self.event_type}_h1_timeseries.png'), 
                    dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_phase_comparison(self):
        """Create bar plots comparing metrics across phases."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Prepare data
        phases = self.phase_metrics['phase'].values
        x_pos = np.arange(len(phases))
        
        # RVR comparison
        ax = axes[0]
        rvr_means = self.phase_metrics['avg_rvr'].values
        rvr_errors = [
            self.phase_metrics['avg_rvr'].values - self.phase_metrics['rvr_ci_lower'].values,
            self.phase_metrics['rvr_ci_upper'].values - self.phase_metrics['avg_rvr'].values
        ]
        
        bars = ax.bar(x_pos, rvr_means, yerr=rvr_errors, capsize=5,
                      color=['lightblue' if 'rising' not in p else 'salmon' for p in phases])
        
        ax.set_xticks(x_pos)
        ax.set_xticklabels([p.replace('_', ' ').title() for p in phases], rotation=45, ha='right')
        ax.set_ylabel('Average RVR')
        ax.set_title('Return-to-Variance Ratio by Phase')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add significance markers
        if self.hypothesis_results and 'rvr' in self.hypothesis_results:
            stats_results = self.hypothesis_results['rvr'].get('statistical_tests', {})
            # rising_idx = list(phases).index('post_event_rising') # Not needed if comparing to rising
            
            for i, phase_name in enumerate(phases):
                if phase_name != 'post_event_rising' and phase_name in stats_results and \
                   stats_results[phase_name].get('significant_at_5pct'):
                    # This marks if post_event_rising is significantly greater than phase_name
                    ax.annotate('*', xy=(i, rvr_means[i]), xytext=(i, rvr_means[i] + 0.001 * np.sign(rvr_means[i]) if rvr_means[i] != 0 else 0.001),
                               ha='center', fontsize=16, color='red')
        
        # Sharpe comparison
        ax = axes[1]
        sharpe_means = self.phase_metrics['avg_sharpe'].values
        sharpe_errors = [
            self.phase_metrics['avg_sharpe'].values - self.phase_metrics['sharpe_ci_lower'].values,
            self.phase_metrics['sharpe_ci_upper'].values - self.phase_metrics['avg_sharpe'].values
        ]
        
        bars = ax.bar(x_pos, sharpe_means, yerr=sharpe_errors, capsize=5,
                      color=['lightgreen' if 'rising' not in p else 'salmon' for p in phases])
        
        ax.set_xticks(x_pos)
        ax.set_xticklabels([p.replace('_', ' ').title() for p in phases], rotation=45, ha='right')
        ax.set_ylabel('Average Sharpe Ratio')
        ax.set_title('Sharpe Ratio by Phase')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add significance markers
        if self.hypothesis_results and 'sharpe' in self.hypothesis_results:
            stats_results = self.hypothesis_results['sharpe'].get('statistical_tests', {})
            
            for i, phase_name in enumerate(phases):
                if phase_name != 'post_event_rising' and phase_name in stats_results and \
                   stats_results[phase_name].get('significant_at_5pct'):
                    ax.annotate('*', xy=(i, sharpe_means[i]), xytext=(i, sharpe_means[i] + 0.01 * np.sign(sharpe_means[i]) if sharpe_means[i] != 0 else 0.01),
                               ha='center', fontsize=16, color='red')
        
        # Add overall result as title
        if self.hypothesis_results:
            rvr_supported = self.hypothesis_results.get('rvr', {}).get('peak_in_rising_phase', False)
            sharpe_supported = self.hypothesis_results.get('sharpe', {}).get('peak_in_rising_phase', False)
            
            support_text = f"Hypothesis 1 Support - RVR: {'YES' if rvr_supported else 'NO'}, " \
                          f"Sharpe: {'YES' if sharpe_supported else 'NO'}"
            fig.suptitle(f'{self.event_type.upper()}: {support_text}', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, f'{self.event_type}_h1_phase_comparison.png'),
                    dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_distributions(self):
        """Plot distributions of RVR and Sharpe by phase."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # RVR distributions
        ax = axes[0, 0]
        for phase in self.phases.keys():
            phase_data = self.processed_data[self.processed_data['phase'] == phase]['rvr'].dropna()
            if len(phase_data) > 0:
                # Remove extreme outliers for better visualization
                q_low, q_high = phase_data.quantile([0.01, 0.99])
                phase_data_clean = phase_data[(phase_data >= q_low) & (phase_data <= q_high)]
                
                if len(phase_data_clean) > 0:
                    ax.hist(phase_data_clean, bins=50, alpha=0.5, 
                           label=phase.replace('_', ' ').title(), density=True)
        
        ax.set_xlabel('RVR')
        ax.set_ylabel('Density')
        ax.set_title('Distribution of Return-to-Variance Ratio by Phase')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # RVR box plot
        ax = axes[0, 1]
        rvr_data = []
        labels = []
        
        for phase in self.phases.keys():
            phase_data = self.processed_data[self.processed_data['phase'] == phase]['rvr'].dropna()
            if len(phase_data) > 0:
                # Remove extreme outliers
                q_low, q_high = phase_data.quantile([0.01, 0.99])
                phase_data_clean = phase_data[(phase_data >= q_low) & (phase_data <= q_high)]
                if len(phase_data_clean) > 0:
                    rvr_data.append(phase_data_clean)
                    labels.append(phase.replace('_', ' ').title())
        
        if rvr_data:
            bp = ax.boxplot(rvr_data, labels=labels, patch_artist=True)
            
            # Color the post-event rising box differently
            for i, patch in enumerate(bp['boxes']):
                if 'Rising' in labels[i]:
                    patch.set_facecolor('salmon')
                else:
                    patch.set_facecolor('lightblue')
        
        ax.set_ylabel('RVR')
        ax.set_title('RVR Distribution by Phase (1st-99th percentile)')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Sharpe distributions
        ax = axes[1, 0]
        for phase in self.phases.keys():
            phase_data = self.processed_data[self.processed_data['phase'] == phase]['sharpe_ratio'].dropna()
            if len(phase_data) > 0:
                # Remove extreme outliers
                q_low, q_high = phase_data.quantile([0.01, 0.99])
                phase_data_clean = phase_data[(phase_data >= q_low) & (phase_data <= q_high)]
                
                if len(phase_data_clean) > 0:
                    ax.hist(phase_data_clean, bins=50, alpha=0.5,
                           label=phase.replace('_', ' ').title(), density=True)
        
        ax.set_xlabel('Sharpe Ratio')
        ax.set_ylabel('Density')
        ax.set_title('Distribution of Sharpe Ratio by Phase')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Sharpe box plot
        ax = axes[1, 1]
        sharpe_data = []
        # labels are already defined from RVR box plot
        
        for phase in self.phases.keys():
            phase_data = self.processed_data[self.processed_data['phase'] == phase]['sharpe_ratio'].dropna()
            if len(phase_data) > 0:
                # Remove extreme outliers
                q_low, q_high = phase_data.quantile([0.01, 0.99])
                phase_data_clean = phase_data[(phase_data >= q_low) & (phase_data <= q_high)]
                if len(phase_data_clean) > 0:
                    sharpe_data.append(phase_data_clean)
        
        if sharpe_data:
            bp = ax.boxplot(sharpe_data, labels=labels, patch_artist=True)
            
            # Color the post-event rising box differently
            for i, patch in enumerate(bp['boxes']):
                if 'Rising' in labels[i]:
                    patch.set_facecolor('salmon')
                else:
                    patch.set_facecolor('lightgreen')
        
        ax.set_ylabel('Sharpe Ratio')
        ax.set_title('Sharpe Ratio Distribution by Phase (1st-99th percentile)')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, f'{self.event_type}_h1_distributions.png'),
                    dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_volatility_decomposition(self):
        """Plot the decomposition of unified volatility."""
        # Calculate daily averages
        daily_avg = self.processed_data.groupby('days_to_event').agg({
            'baseline_volatility': 'mean',
            'unified_volatility': 'mean',
            'phi_adjustment': 'mean'
        }).reset_index()
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot baseline and unified volatility
        ax.plot(daily_avg['days_to_event'], daily_avg['baseline_volatility'],
                color='blue', linewidth=2, label='Baseline (GJR-GARCH)', linestyle='--')
        ax.plot(daily_avg['days_to_event'], daily_avg['unified_volatility'],
                color='red', linewidth=2, label='Unified Volatility')
        
        # Fill between to show the impact of phi adjustment
        ax.fill_between(daily_avg['days_to_event'],
                       daily_avg['baseline_volatility'],
                       daily_avg['unified_volatility'],
                       alpha=0.3, color='orange', label='Event adjustment (φ)')
        
        # Mark phases
        rising_phase_label_added = False
        for phase, (start, end) in self.phases.items():
            if 'rising' in phase:
                if not rising_phase_label_added:
                    ax.axvspan(start, end, alpha=0.2, color='red', label='Post-event rising')
                    rising_phase_label_added = True
                else:
                    ax.axvspan(start, end, alpha=0.2, color='red')
        
        ax.axvline(0, color='black', linestyle='--', label='Event day')
        ax.set_xlabel('Days to Event')
        ax.set_ylabel('Volatility')
        ax.set_title(f'{self.event_type.upper()}: Unified Volatility Decomposition')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, f'{self.event_type}_h1_volatility_decomposition.png'),
                    dpi=300, bbox_inches='tight')
        plt.close()
    
    def _save_results(self):
        """Save analysis results to files."""
        print("\nSaving results...")
        
        if self.processed_data is None or self.processed_data.empty:
            print("Warning: No processed data to save.")
            return

        # Save processed data
        self.processed_data.to_csv(
            os.path.join(self.results_dir, f'{self.event_type}_h1_processed_data.csv'),
            index=False
        )
        
        if self.phase_metrics is not None and not self.phase_metrics.empty:
            # Save phase metrics
            self.phase_metrics.to_csv(
                os.path.join(self.results_dir, f'{self.event_type}_h1_phase_metrics.csv'),
                index=False
            )
        
        if self.hypothesis_results is not None:
            # Save hypothesis test results
            results_summary = pd.DataFrame([{
                'event_type': self.event_type,
                'rvr_peak_supported': self.hypothesis_results.get('rvr', {}).get('peak_in_rising_phase', None),
                'rvr_rising_value': self.hypothesis_results.get('rvr', {}).get('rising_phase_value', None),
                'sharpe_peak_supported': self.hypothesis_results.get('sharpe', {}).get('peak_in_rising_phase', None),
                'sharpe_rising_value': self.hypothesis_results.get('sharpe', {}).get('rising_phase_value', None),
                'n_total_obs': len(self.processed_data),
                'n_events': self.processed_data['event_date'].nunique(),
                'n_symbols': self.processed_data['symbol'].nunique()
            }])
            
            results_summary.to_csv(
                os.path.join(self.results_dir, f'{self.event_type}_h1_test_results.csv'),
                index=False
            )
        
        print(f"Results saved to {self.results_dir}")


def run_hypothesis1_analysis(event_type: str, event_file: str, date_col: str, 
                           ticker_col: str, stock_files: List[str]) -> bool:
    """
    Run Hypothesis 1 analysis for a specific event type.
    
    Returns:
        True if analysis completed successfully
    """
    # Set up configuration
    config = Config()
    
    # Set up results directory
    if event_type == 'fda':
        results_dir = FDA_RESULTS_DIR
    else:
        results_dir = EARNINGS_RESULTS_DIR
    
    # Initialize tester
    tester = Hypothesis1Tester(config, event_type, results_dir)
    
    # Load data
    stock_df = tester.load_stock_data(stock_files)
    if stock_df is None or stock_df.is_empty():
        print(f"Failed to load stock data for {event_type}.")
        return False
    
    event_df = tester.load_event_data(event_file, date_col, ticker_col)
    if event_df is None or event_df.is_empty():
        print(f"Failed to load event data for {event_type}.")
        return False
    
    # Run analysis
    return tester.run_analysis(stock_df, event_df)


def compare_results():
    """Compare results between FDA and earnings events."""
    print("\n" + "="*60)
    print("Comparing FDA and Earnings Results for Hypothesis 1")
    print("="*60)
    
    fda_results_path = os.path.join(FDA_RESULTS_DIR, 'fda_h1_test_results.csv')
    earnings_results_path = os.path.join(EARNINGS_RESULTS_DIR, 'earnings_h1_test_results.csv')

    if not os.path.exists(fda_results_path):
        print(f"FDA results file not found: {fda_results_path}")
        return
    if not os.path.exists(earnings_results_path):
        print(f"Earnings results file not found: {earnings_results_path}")
        return

    # Load results
    fda_results = pd.read_csv(fda_results_path)
    earnings_results = pd.read_csv(earnings_results_path)
    
    if fda_results.empty or earnings_results.empty:
        print("One or both result files are empty. Cannot compare.")
        return

    # Create comparison figure
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Support comparison
    ax = axes[0]
    support_data = pd.DataFrame({
        'FDA': [fda_results['rvr_peak_supported'].iloc[0],
                fda_results['sharpe_peak_supported'].iloc[0]],
        'Earnings': [earnings_results['rvr_peak_supported'].iloc[0],
                     earnings_results['sharpe_peak_supported'].iloc[0]]
    }, index=['RVR Peak', 'Sharpe Peak'])
    
    support_data.plot(kind='bar', ax=ax, color=['lightblue', 'lightgreen'])
    ax.set_ylabel('Hypothesis Supported (1=Yes, 0=No)')
    ax.set_title('Hypothesis 1 Support Comparison')
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['No', 'Yes'])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Value comparison
    ax = axes[1]
    value_data = pd.DataFrame({
        'FDA': [fda_results['rvr_rising_value'].iloc[0],
                fda_results['sharpe_rising_value'].iloc[0]],
        'Earnings': [earnings_results['rvr_rising_value'].iloc[0],
                     earnings_results['sharpe_rising_value'].iloc[0]]
    }, index=['RVR (Rising)', 'Sharpe (Rising)'])
    
    value_data.plot(kind='bar', ax=ax, color=['salmon', 'peachpuff'])
    ax.set_ylabel('Average Value in Rising Phase')
    ax.set_title('Risk-Adjusted Returns in Post-Event Rising Phase')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    # Save comparison
    comparison_dir = 'results/hypothesis1/comparison/'
    os.makedirs(comparison_dir, exist_ok=True)
    plt.savefig(os.path.join(comparison_dir, 'h1_comparison.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # Print summary
    print("\nHypothesis 1 Results Summary:")
    print(f"  FDA - RVR Peak Supported: {fda_results['rvr_peak_supported'].iloc[0]}")
    print(f"  FDA - Sharpe Peak Supported: {fda_results['sharpe_peak_supported'].iloc[0]}")
    print(f"  Earnings - RVR Peak Supported: {earnings_results['rvr_peak_supported'].iloc[0]}")
    print(f"  Earnings - Sharpe Peak Supported: {earnings_results['sharpe_peak_supported'].iloc[0]}")


def main():
    """Main function to run all analyses."""
    print("Starting Hypothesis 1 Analysis")
    print("="*60)
    
    # Run FDA analysis
    fda_success = run_hypothesis1_analysis(
        event_type='fda',
        event_file=FDA_EVENT_FILE,
        date_col=FDA_EVENT_DATE_COL,
        ticker_col=FDA_TICKER_COL,
        stock_files=STOCK_FILES
    )
    
    # Run Earnings analysis
    earnings_success = run_hypothesis1_analysis(
        event_type='earnings',
        event_file=EARNINGS_EVENT_FILE,
        date_col=EARNINGS_EVENT_DATE_COL,
        ticker_col=EARNINGS_TICKER_COL,
        stock_files=STOCK_FILES
    )
    
    # Compare results if both succeeded
    if fda_success and earnings_success:
        compare_results()
    else:
        print("\nComparison skipped as one or both analyses did not complete successfully.")
    
    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()
