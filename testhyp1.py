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
                elif 'SYMBOL' in col_map_upper: # Ensure this matches load_event_data if it uses SYMBOL
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
                df = df_raw.select([
                    pl.col(source_ticker_col).alias('ticker'), # Keep as 'ticker' for now, rename to 'symbol' before EventProcessor
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
                print(f"  Warning: Error processing file {path}: {e}")
                continue
        
        if not dfs:
            print("Error: No stock data loaded!")
            return None
        
        combined_df = pl.concat(dfs)
        
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
            df = pl.read_csv(file_path, try_parse_dates=True, infer_schema_length=10000)
            
            actual_date_col = None
            actual_ticker_col = None # This will become 'symbol'
            
            # Case-insensitive search for columns
            df_cols_lower = {c.lower(): c for c in df.columns}

            if date_col.lower() in df_cols_lower:
                actual_date_col = df_cols_lower[date_col.lower()]
            
            # For ticker, we will map it to 'symbol' as expected by EventProcessor later if needed
            # Here, we map to 'symbol' directly as EventProcessor expects 'symbol'
            if ticker_col.lower() in df_cols_lower:
                actual_ticker_col = df_cols_lower[ticker_col.lower()]
            
            if not actual_date_col or not actual_ticker_col:
                print(f"Error: Could not find required columns '{date_col}' or '{ticker_col}' in event file.")
                print(f"Available columns: {df.columns}")
                return None
            
            df = df.rename({
                actual_date_col: 'event_date',
                actual_ticker_col: 'symbol' # Rename to 'symbol' directly
            })
            
            df = df.with_columns([
                pl.col('event_date').cast(pl.Date),
                pl.col('symbol').cast(pl.Utf8).str.strip_chars(),
                pl.lit(self.event_type).alias('event_type')
            ])
            
            df = df.filter(
                pl.col('event_date').is_not_null() &
                pl.col('symbol').is_not_null() &
                (pl.col('symbol').str.len_chars() > 0)
            ).unique(subset=['symbol', 'event_date'])
            
            print(f"Loaded {df.height:,} {self.event_type} events")
            return df
            
        except Exception as e:
            print(f"Error loading event data from {file_path}: {e}")
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

        stock_pd_df = stock_df.to_pandas()
        event_pd_df = event_df.to_pandas()

        # Ensure the stock data has 'symbol' column as expected by EventProcessor
        if 'ticker' in stock_pd_df.columns and 'symbol' not in stock_pd_df.columns:
            stock_pd_df = stock_pd_df.rename(columns={'ticker': 'symbol'})
        elif 'ticker' not in stock_pd_df.columns and 'symbol' not in stock_pd_df.columns:
            print("Critical Error: Neither 'ticker' nor 'symbol' found in stock_pd_df after to_pandas().")
            return False

        # Event data should already have 'symbol' from load_event_data
        if 'symbol' not in event_pd_df.columns:
            print("Critical Error: 'symbol' column missing in event_pd_df.")
            # Attempt to rename if 'ticker' exists, though load_event_data should handle this
            if 'ticker' in event_pd_df.columns:
                event_pd_df = event_pd_df.rename(columns={'ticker':'symbol'})
            else:
                return False
        
        # Process events using EventProcessor
        print("\nProcessing events with unified volatility model...")
        self.processed_data = self.event_processor.process_events(
            price_data=stock_pd_df, 
            event_data=event_pd_df
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
                
                # Handle cases where RVR or Sharpe might be all NaN due to zero volatility
                rvr_valid = phase_data['rvr'].dropna()
                sharpe_valid = phase_data['sharpe_ratio'].dropna()

                if len(rvr_valid) > 0 and len(sharpe_valid) >0 :
                    for _ in range(n_bootstrap):
                        sample = phase_data.sample(n=len(phase_data), replace=True)
                        rvr_samples.append(sample['rvr'].dropna().mean()) # dropna inside loop too
                        sharpe_samples.append(sample['sharpe_ratio'].dropna().mean())
                    
                    if rvr_samples: # Check if list is not empty
                        metrics['rvr_ci_lower'] = np.percentile(rvr_samples, 2.5)
                        metrics['rvr_ci_upper'] = np.percentile(rvr_samples, 97.5)
                    else:
                        metrics['rvr_ci_lower'] = np.nan
                        metrics['rvr_ci_upper'] = np.nan

                    if sharpe_samples:
                        metrics['sharpe_ci_lower'] = np.percentile(sharpe_samples, 2.5)
                        metrics['sharpe_ci_upper'] = np.percentile(sharpe_samples, 97.5)
                    else:
                        metrics['sharpe_ci_lower'] = np.nan
                        metrics['sharpe_ci_upper'] = np.nan
                else:
                    metrics['rvr_ci_lower'] = np.nan
                    metrics['rvr_ci_upper'] = np.nan
                    metrics['sharpe_ci_lower'] = np.nan
                    metrics['sharpe_ci_upper'] = np.nan

                phase_results.append(metrics)
        
        self.phase_metrics = pd.DataFrame(phase_results)
        
        # Display results
        print("\nPhase Metrics Summary:")
        if not self.phase_metrics.empty:
            print(self.phase_metrics[['phase', 'n_obs', 'avg_rvr', 'avg_sharpe']].to_string(index=False))
        else:
            print("No phase metrics calculated.")
    
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
        if 'post_event_rising' in metrics_dict and pd.notna(metrics_dict['post_event_rising']['avg_rvr']):
            rising_rvr = metrics_dict['post_event_rising']['avg_rvr']
            
            rvr_peak = True
            rvr_comparisons = {}
            
            for phase in ['pre_event_early', 'pre_event_late', 'post_event_decay']:
                if phase in metrics_dict and pd.notna(metrics_dict[phase]['avg_rvr']):
                    other_rvr = metrics_dict[phase]['avg_rvr']
                    rvr_comparisons[phase] = other_rvr < rising_rvr
                    if other_rvr >= rising_rvr:
                        rvr_peak = False
            
            rvr_stats = self._perform_statistical_tests('rvr', 'post_event_rising')
            
            results['rvr'] = {
                'peak_in_rising_phase': rvr_peak,
                'rising_phase_value': rising_rvr,
                'comparisons': rvr_comparisons,
                'statistical_tests': rvr_stats
            }
        else:
            results['rvr'] = {'peak_in_rising_phase': False, 'rising_phase_value': np.nan}

        # Test for Sharpe ratio peak
        if 'post_event_rising' in metrics_dict and pd.notna(metrics_dict['post_event_rising']['avg_sharpe']):
            rising_sharpe = metrics_dict['post_event_rising']['avg_sharpe']
            
            sharpe_peak = True
            sharpe_comparisons = {}
            
            for phase in ['pre_event_early', 'pre_event_late', 'post_event_decay']:
                if phase in metrics_dict and pd.notna(metrics_dict[phase]['avg_sharpe']):
                    other_sharpe = metrics_dict[phase]['avg_sharpe']
                    sharpe_comparisons[phase] = other_sharpe < rising_sharpe
                    if other_sharpe >= rising_sharpe:
                        sharpe_peak = False
            
            sharpe_stats = self._perform_statistical_tests('sharpe_ratio', 'post_event_rising')
            
            results['sharpe'] = {
                'peak_in_rising_phase': sharpe_peak,
                'rising_phase_value': rising_sharpe,
                'comparisons': sharpe_comparisons,
                'statistical_tests': sharpe_stats
            }
        else:
            results['sharpe'] = {'peak_in_rising_phase': False, 'rising_phase_value': np.nan}

        self.hypothesis_results = results
        
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
                try:
                    t_stat, p_value = stats.ttest_ind(test_data, compare_data, 
                                                     equal_var=False, alternative='greater', nan_policy='omit')
                    u_stat, u_pvalue = stats.mannwhitneyu(test_data, compare_data, 
                                                          alternative='greater', nan_policy='omit')
                    results[compare_phase] = {
                        't_statistic': t_stat,
                        't_pvalue': p_value,
                        'u_statistic': u_stat,
                        'u_pvalue': u_pvalue,
                        'significant_at_5pct': p_value < 0.05
                    }
                except Exception as e:
                    print(f"Statistical test error for {metric}, {test_phase} vs {compare_phase}: {e}")
                    results[compare_phase] = {'significant_at_5pct': False, 't_pvalue': np.nan, 'u_pvalue': np.nan}
            else:
                results[compare_phase] = {'significant_at_5pct': False, 't_pvalue': np.nan, 'u_pvalue': np.nan}

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

        self._plot_time_series()
        self._plot_phase_comparison()
        self._plot_distributions()
        self._plot_volatility_decomposition()
    
    def _plot_time_series(self):
        """Plot average RVR and Sharpe ratio around events."""
        daily_avg = self.processed_data.groupby('days_to_event').agg(
            rvr_mean=('rvr', 'mean'),
            rvr_std=('rvr', 'std'),
            rvr_count=('rvr', 'count'),
            sharpe_ratio_mean=('sharpe_ratio', 'mean'),
            sharpe_ratio_std=('sharpe_ratio', 'std'),
            sharpe_ratio_count=('sharpe_ratio', 'count'),
            phi_adjustment_mean=('phi_adjustment', 'mean')
        ).reset_index()
        
        fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
        
        # Plot RVR
        ax = axes[0]
        ax.plot(daily_avg['days_to_event'], daily_avg['rvr_mean'], color='blue', linewidth=2, label='RVR')
        sem_rvr = daily_avg['rvr_std'] / np.sqrt(daily_avg['rvr_count'])
        ax.fill_between(daily_avg['days_to_event'],
                       daily_avg['rvr_mean'] - 1.96 * sem_rvr.fillna(0),
                       daily_avg['rvr_mean'] + 1.96 * sem_rvr.fillna(0),
                       alpha=0.3, color='blue')
        
        rising_phase_label_added = False
        for phase, (start, end) in self.phases.items():
            if 'rising' in phase:
                label = 'Post-event rising' if not rising_phase_label_added else None
                ax.axvspan(start, end, alpha=0.2, color='red', label=label)
                if label: rising_phase_label_added = True
        ax.axvline(0, color='black', linestyle='--', label='Event day')
        ax.set_ylabel('Return-to-Variance Ratio')
        ax.set_title(f'{self.event_type.upper()}: Average RVR Around Events')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot Sharpe ratio
        ax = axes[1]
        ax.plot(daily_avg['days_to_event'], daily_avg['sharpe_ratio_mean'], color='green', linewidth=2, label='Sharpe Ratio')
        sem_sharpe = daily_avg['sharpe_ratio_std'] / np.sqrt(daily_avg['sharpe_ratio_count'])
        ax.fill_between(daily_avg['days_to_event'],
                       daily_avg['sharpe_ratio_mean'] - 1.96 * sem_sharpe.fillna(0),
                       daily_avg['sharpe_ratio_mean'] + 1.96 * sem_sharpe.fillna(0),
                       alpha=0.3, color='green')
        for phase, (start, end) in self.phases.items():
            if 'rising' in phase: ax.axvspan(start, end, alpha=0.2, color='red')
        ax.axvline(0, color='black', linestyle='--')
        ax.set_ylabel('Sharpe Ratio')
        ax.set_title(f'{self.event_type.upper()}: Average Sharpe Ratio Around Events')
        ax.grid(True, alpha=0.3)
        
        # Plot phi adjustment
        ax = axes[2]
        ax.plot(daily_avg['days_to_event'], daily_avg['phi_adjustment_mean'], color='purple', linewidth=2, label='Phi adjustment')
        for phase, (start, end) in self.phases.items():
            if 'rising' in phase: ax.axvspan(start, end, alpha=0.2, color='red')
        ax.axvline(0, color='black', linestyle='--')
        ax.set_xlabel('Days to Event')
        ax.set_ylabel('Volatility Adjustment (φ)')
        ax.set_title('Event-Specific Volatility Adjustment (φ)')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, f'{self.event_type}_h1_timeseries.png'), dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_phase_comparison(self):
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        phases_order = ['pre_event_early', 'pre_event_late', 'post_event_rising', 'post_event_decay']
        
        # Ensure phase_metrics is sorted according to phases_order
        self.phase_metrics['phase'] = pd.Categorical(self.phase_metrics['phase'], categories=phases_order, ordered=True)
        plot_metrics = self.phase_metrics.sort_values('phase').reset_index(drop=True)

        x_pos = np.arange(len(plot_metrics))
        
        # RVR
        ax = axes[0]
        rvr_means = plot_metrics['avg_rvr'].fillna(0).values
        rvr_ci_lower = plot_metrics['rvr_ci_lower'].fillna(rvr_means).values
        rvr_ci_upper = plot_metrics['rvr_ci_upper'].fillna(rvr_means).values
        rvr_errors = [rvr_means - rvr_ci_lower, rvr_ci_upper - rvr_means]
        
        ax.bar(x_pos, rvr_means, yerr=rvr_errors, capsize=5, color=['lightblue' if 'rising' not in p else 'salmon' for p in plot_metrics['phase']])
        ax.set_xticks(x_pos)
        ax.set_xticklabels([p.replace('_', ' ').title() for p in plot_metrics['phase']], rotation=45, ha='right')
        ax.set_ylabel('Average RVR')
        ax.set_title('Return-to-Variance Ratio by Phase')
        ax.grid(True, alpha=0.3, axis='y')

        if self.hypothesis_results and 'rvr' in self.hypothesis_results and self.hypothesis_results['rvr'].get('statistical_tests'):
            stats_rvr = self.hypothesis_results['rvr']['statistical_tests']
            for i, phase_name in enumerate(plot_metrics['phase']):
                if phase_name != 'post_event_rising' and phase_name in stats_rvr and stats_rvr[phase_name].get('significant_at_5pct'):
                    y_val = rvr_means[i]
                    offset = (rvr_ci_upper[i] - y_val) * 0.1 if pd.notna(rvr_ci_upper[i]) else abs(y_val * 0.05) + 0.001
                    ax.annotate('*', xy=(i, rvr_ci_upper[i] if pd.notna(rvr_ci_upper[i]) else y_val), xytext=(i, (rvr_ci_upper[i] if pd.notna(rvr_ci_upper[i]) else y_val) + offset), ha='center', fontsize=16, color='red')

        # Sharpe
        ax = axes[1]
        sharpe_means = plot_metrics['avg_sharpe'].fillna(0).values
        sharpe_ci_lower = plot_metrics['sharpe_ci_lower'].fillna(sharpe_means).values
        sharpe_ci_upper = plot_metrics['sharpe_ci_upper'].fillna(sharpe_means).values
        sharpe_errors = [sharpe_means - sharpe_ci_lower, sharpe_ci_upper - sharpe_means]

        ax.bar(x_pos, sharpe_means, yerr=sharpe_errors, capsize=5, color=['lightgreen' if 'rising' not in p else 'salmon' for p in plot_metrics['phase']])
        ax.set_xticks(x_pos)
        ax.set_xticklabels([p.replace('_', ' ').title() for p in plot_metrics['phase']], rotation=45, ha='right')
        ax.set_ylabel('Average Sharpe Ratio')
        ax.set_title('Sharpe Ratio by Phase')
        ax.grid(True, alpha=0.3, axis='y')

        if self.hypothesis_results and 'sharpe' in self.hypothesis_results and self.hypothesis_results['sharpe'].get('statistical_tests'):
            stats_sharpe = self.hypothesis_results['sharpe']['statistical_tests']
            for i, phase_name in enumerate(plot_metrics['phase']):
                 if phase_name != 'post_event_rising' and phase_name in stats_sharpe and stats_sharpe[phase_name].get('significant_at_5pct'):
                    y_val = sharpe_means[i]
                    offset = (sharpe_ci_upper[i] - y_val) * 0.1 if pd.notna(sharpe_ci_upper[i]) else abs(y_val * 0.05) + 0.01
                    ax.annotate('*', xy=(i, sharpe_ci_upper[i] if pd.notna(sharpe_ci_upper[i]) else y_val), xytext=(i, (sharpe_ci_upper[i] if pd.notna(sharpe_ci_upper[i]) else y_val) + offset), ha='center', fontsize=16, color='red')

        if self.hypothesis_results:
            rvr_supported = self.hypothesis_results.get('rvr', {}).get('peak_in_rising_phase', False)
            sharpe_supported = self.hypothesis_results.get('sharpe', {}).get('peak_in_rising_phase', False)
            support_text = f"Hypothesis 1 Support - RVR: {'YES' if rvr_supported else 'NO'}, Sharpe: {'YES' if sharpe_supported else 'NO'}"
            fig.suptitle(f'{self.event_type.upper()}: {support_text}', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, f'{self.event_type}_h1_phase_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_distributions(self):
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        phases_order = ['pre_event_early', 'pre_event_late', 'post_event_rising', 'post_event_decay']

        # RVR distributions
        ax = axes[0, 0]
        for phase in phases_order:
            phase_data = self.processed_data[self.processed_data['phase'] == phase]['rvr'].dropna()
            if len(phase_data) > 1: # Need more than 1 point for quantile
                q_low, q_high = phase_data.quantile([0.01, 0.99])
                phase_data_clean = phase_data[(phase_data >= q_low) & (phase_data <= q_high)]
                if len(phase_data_clean) > 0:
                    ax.hist(phase_data_clean, bins=50, alpha=0.5, label=phase.replace('_', ' ').title(), density=True)
        ax.set_xlabel('RVR')
        ax.set_ylabel('Density')
        ax.set_title('Distribution of RVR by Phase')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # RVR box plot
        ax = axes[0, 1]
        rvr_plot_data = []
        labels = []
        for phase in phases_order:
            phase_data = self.processed_data[self.processed_data['phase'] == phase]['rvr'].dropna()
            if len(phase_data) > 1:
                q_low, q_high = phase_data.quantile([0.01, 0.99])
                phase_data_clean = phase_data[(phase_data >= q_low) & (phase_data <= q_high)]
                if len(phase_data_clean) > 0:
                    rvr_plot_data.append(phase_data_clean)
                    labels.append(phase.replace('_', ' ').title())
        if rvr_plot_data:
            bp = ax.boxplot(rvr_plot_data, labels=labels, patch_artist=True, showfliers=False) # hide outliers as we clipped
            for i, patch in enumerate(bp['boxes']):
                patch.set_facecolor('salmon' if 'Rising' in labels[i] else 'lightblue')
        ax.set_ylabel('RVR')
        ax.set_title('RVR Distribution by Phase (1st-99th percentile)')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3, axis='y')

        # Sharpe distributions
        ax = axes[1, 0]
        for phase in phases_order:
            phase_data = self.processed_data[self.processed_data['phase'] == phase]['sharpe_ratio'].dropna()
            if len(phase_data) > 1:
                q_low, q_high = phase_data.quantile([0.01, 0.99])
                phase_data_clean = phase_data[(phase_data >= q_low) & (phase_data <= q_high)]
                if len(phase_data_clean) > 0:
                    ax.hist(phase_data_clean, bins=50, alpha=0.5, label=phase.replace('_', ' ').title(), density=True)
        ax.set_xlabel('Sharpe Ratio')
        ax.set_ylabel('Density')
        ax.set_title('Distribution of Sharpe Ratio by Phase')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Sharpe box plot
        ax = axes[1, 1]
        sharpe_plot_data = []
        # labels already defined
        for phase in phases_order:
            phase_data = self.processed_data[self.processed_data['phase'] == phase]['sharpe_ratio'].dropna()
            if len(phase_data) > 1:
                q_low, q_high = phase_data.quantile([0.01, 0.99])
                phase_data_clean = phase_data[(phase_data >= q_low) & (phase_data <= q_high)]
                if len(phase_data_clean) > 0:
                    sharpe_plot_data.append(phase_data_clean)
        if sharpe_plot_data:
            bp = ax.boxplot(sharpe_plot_data, labels=labels, patch_artist=True, showfliers=False)
            for i, patch in enumerate(bp['boxes']):
                patch.set_facecolor('salmon' if 'Rising' in labels[i] else 'lightgreen')
        ax.set_ylabel('Sharpe Ratio')
        ax.set_title('Sharpe Ratio Distribution by Phase (1st-99th percentile)')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, f'{self.event_type}_h1_distributions.png'), dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_volatility_decomposition(self):
        daily_avg = self.processed_data.groupby('days_to_event').agg(
            baseline_volatility_mean=('baseline_volatility', 'mean'),
            unified_volatility_mean=('unified_volatility', 'mean')
        ).reset_index()
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        ax.plot(daily_avg['days_to_event'], daily_avg['baseline_volatility_mean'], color='blue', linewidth=2, label='Baseline (GJR-GARCH)', linestyle='--')
        ax.plot(daily_avg['days_to_event'], daily_avg['unified_volatility_mean'], color='red', linewidth=2, label='Unified Volatility')
        
        ax.fill_between(daily_avg['days_to_event'],
                       daily_avg['baseline_volatility_mean'],
                       daily_avg['unified_volatility_mean'],
                       where=daily_avg['unified_volatility_mean'] >= daily_avg['baseline_volatility_mean'],
                       alpha=0.3, color='orange', interpolate=True, label='Event adjustment (φ > 0)')
        ax.fill_between(daily_avg['days_to_event'],
                       daily_avg['baseline_volatility_mean'],
                       daily_avg['unified_volatility_mean'],
                       where=daily_avg['unified_volatility_mean'] < daily_avg['baseline_volatility_mean'],
                       alpha=0.3, color='cyan', interpolate=True, label='Event adjustment (φ < 0)')

        rising_phase_label_added = False
        for phase, (start, end) in self.phases.items():
            if 'rising' in phase:
                label = 'Post-event rising' if not rising_phase_label_added else None
                ax.axvspan(start, end, alpha=0.2, color='red', label=label)
                if label: rising_phase_label_added = True
        ax.axvline(0, color='black', linestyle='--', label='Event day')
        ax.set_xlabel('Days to Event')
        ax.set_ylabel('Volatility')
        ax.set_title(f'{self.event_type.upper()}: Unified Volatility Decomposition')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, f'{self.event_type}_h1_volatility_decomposition.png'), dpi=300, bbox_inches='tight')
        plt.close()

    def _save_results(self):
        print("\nSaving results...")
        if self.processed_data is None or self.processed_data.empty:
            print("Warning: No processed data to save.")
            return

        self.processed_data.to_csv(os.path.join(self.results_dir, f'{self.event_type}_h1_processed_data.csv'), index=False)
        
        if self.phase_metrics is not None and not self.phase_metrics.empty:
            self.phase_metrics.to_csv(os.path.join(self.results_dir, f'{self.event_type}_h1_phase_metrics.csv'), index=False)
        
        if self.hypothesis_results is not None:
            n_events = self.processed_data['event_date'].nunique() if 'event_date' in self.processed_data else 0
            n_symbols = self.processed_data['symbol'].nunique() if 'symbol' in self.processed_data else 0 # EventProcessor output uses 'symbol'

            results_summary = pd.DataFrame([{
                'event_type': self.event_type,
                'rvr_peak_supported': self.hypothesis_results.get('rvr', {}).get('peak_in_rising_phase', None),
                'rvr_rising_value': self.hypothesis_results.get('rvr', {}).get('rising_phase_value', None),
                'sharpe_peak_supported': self.hypothesis_results.get('sharpe', {}).get('peak_in_rising_phase', None),
                'sharpe_rising_value': self.hypothesis_results.get('sharpe', {}).get('rising_phase_value', None),
                'n_total_obs': len(self.processed_data),
                'n_events': n_events,
                'n_symbols': n_symbols
            }])
            results_summary.to_csv(os.path.join(self.results_dir, f'{self.event_type}_h1_test_results.csv'), index=False)
        
        print(f"Results saved to {self.results_dir}")

def run_hypothesis1_analysis(event_type: str, event_file: str, date_col: str, 
                           ticker_col: str, stock_files: List[str]) -> bool:
    config = Config()
    results_dir = FDA_RESULTS_DIR if event_type == 'fda' else EARNINGS_RESULTS_DIR
    tester = Hypothesis1Tester(config, event_type, results_dir)
    
    stock_df = tester.load_stock_data(stock_files)
    if stock_df is None or stock_df.is_empty():
        print(f"Failed to load stock data for {event_type}.")
        return False
    
    event_df = tester.load_event_data(event_file, date_col, ticker_col)
    if event_df is None or event_df.is_empty():
        print(f"Failed to load event data for {event_type}.")
        return False
    
    return tester.run_analysis(stock_df, event_df)

def compare_results():
    print("\n" + "="*60)
    print("Comparing FDA and Earnings Results for Hypothesis 1")
    print("="*60)
    
    fda_results_path = os.path.join(FDA_RESULTS_DIR, 'fda_h1_test_results.csv')
    earnings_results_path = os.path.join(EARNINGS_RESULTS_DIR, 'earnings_h1_test_results.csv')

    if not (os.path.exists(fda_results_path) and os.path.exists(earnings_results_path)):
        print("One or both result files not found. Cannot compare.")
        return

    fda_results = pd.read_csv(fda_results_path)
    earnings_results = pd.read_csv(earnings_results_path)
    
    if fda_results.empty or earnings_results.empty:
        print("One or both result files are empty. Cannot compare.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Support comparison
    ax = axes[0]
    support_data = pd.DataFrame({
        'FDA': [fda_results['rvr_peak_supported'].iloc[0], fda_results['sharpe_peak_supported'].iloc[0]],
        'Earnings': [earnings_results['rvr_peak_supported'].iloc[0], earnings_results['sharpe_peak_supported'].iloc[0]]
    }, index=['RVR Peak', 'Sharpe Peak'])
    
    support_data.astype(float).plot(kind='bar', ax=ax, color=['lightblue', 'lightgreen']) # Cast to float for plotting 0/1
    ax.set_ylabel('Hypothesis Supported (1=Yes, 0=No)')
    ax.set_title('Hypothesis 1 Support Comparison')
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['No', 'Yes'])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Value comparison
    ax = axes[1]
    value_data = pd.DataFrame({
        'FDA': [fda_results['rvr_rising_value'].iloc[0], fda_results['sharpe_rising_value'].iloc[0]],
        'Earnings': [earnings_results['rvr_rising_value'].iloc[0], earnings_results['sharpe_rising_value'].iloc[0]]
    }, index=['RVR (Rising)', 'Sharpe (Rising)'])
    
    value_data.plot(kind='bar', ax=ax, color=['salmon', 'peachpuff'])
    ax.set_ylabel('Average Value in Rising Phase')
    ax.set_title('Risk-Adjusted Returns in Post-Event Rising Phase')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    comparison_dir = 'results/hypothesis1/comparison/'
    os.makedirs(comparison_dir, exist_ok=True)
    plt.savefig(os.path.join(comparison_dir, 'h1_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print("\nHypothesis 1 Results Summary:")
    print(f"  FDA - RVR Peak Supported: {fda_results['rvr_peak_supported'].iloc[0]}")
    print(f"  FDA - Sharpe Peak Supported: {fda_results['sharpe_peak_supported'].iloc[0]}")
    print(f"  Earnings - RVR Peak Supported: {earnings_results['rvr_peak_supported'].iloc[0]}")
    print(f"  Earnings - Sharpe Peak Supported: {earnings_results['sharpe_peak_supported'].iloc[0]}")

def main():
    print("Starting Hypothesis 1 Analysis")
    print("="*60)
    
    fda_success = run_hypothesis1_analysis(
        event_type='fda', event_file=FDA_EVENT_FILE, date_col=FDA_EVENT_DATE_COL,
        ticker_col=FDA_TICKER_COL, stock_files=STOCK_FILES
    )
    
    earnings_success = run_hypothesis1_analysis(
        event_type='earnings', event_file=EARNINGS_EVENT_FILE, date_col=EARNINGS_EVENT_DATE_COL,
        ticker_col=EARNINGS_TICKER_COL, stock_files=STOCK_FILES
    )
    
    if fda_success and earnings_success:
        compare_results()
    else:
        print("\nComparison skipped as one or both analyses did not complete successfully.")
    
    print("\nAnalysis complete!")

if __name__ == "__main__":
    main()
