# testhyp1.py
# Aligned with the paper: "Modeling Equilibrium Asset Pricing Around Events..."

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

# --- Hardcoded Analysis Parameters (aligned with paper) ---
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

# Shared analysis parameters from the paper
WINDOW_DAYS = 60  # General window for data loading
ANALYSIS_WINDOW = (-30, 30) # Analysis window relative to event date (t_event=0)

# Volatility model parameters (mapping to paper's notation)
GARCH_TYPE = 'gjr'      # Baseline GARCH model for h_t
K1 = 1.5                # k1: Pre-event volatility multiplier
K2 = 2.0                # k2: Post-event volatility multiplier (k2 > k1)
DELTA_T1 = 5.0          # Δt1: Pre-event volatility duration parameter
DELTA_T2 = 3.0          # Δt2: Post-event rising phase rate parameter
DELTA_T3 = 10.0         # Δt3: Post-event decay rate parameter
DELTA = 5               # δ: Duration of post-event rising phase in days

# Expectation formation parameters (mapping to paper's notation)
B0_BIAS = 0.01          # b0: Baseline bias (as decimal, e.g., 1% daily)
KAPPA_BIAS = 1.0        # κ: Multiplier for heightened optimism during post-event rising phase
RISK_FREE_RATE = 0.0    # Daily risk-free rate (r_t)

class Hypothesis1Analyzer:
    """
    Analyzer for testing Hypothesis 1:
    Risk-adjusted returns (RVR and Sharpe ratio) peak during the post-event rising phase.
    """
    def __init__(self, analyzer: EventAnalysis, results_dir: str, file_prefix: str):
        self.analyzer = analyzer
        self.results_dir = results_dir
        self.file_prefix = file_prefix
        self.return_col = 'ret' # Assumes 'ret' column in EventAnalysis.data
        self.analysis_window = ANALYSIS_WINDOW
        self.garch_type = GARCH_TYPE

        # Parameters for unified volatility and biased expectations model
        self.k1, self.k2 = K1, K2
        self.delta_t1, self.delta_t2, self.delta_t3 = DELTA_T1, DELTA_T2, DELTA_T3
        self.delta = DELTA
        self.b0_bias = B0_BIAS
        self.kappa_bias = KAPPA_BIAS
        self.risk_free_rate = RISK_FREE_RATE

        # Define phase windows based on paper's δ (DELTA)
        # t_event is day 0
        # Pre-event: t <= t_event (includes event day for phi1)
        # Post-event rising: t_event < t <= t_event + δ
        # Post-event decay: t > t_event + δ
        self.phases = {
            'pre_event': (ANALYSIS_WINDOW[0], -1), # Practical pre-event window
            'event_day': (0, 0),                   # Event day itself
            'post_event_rising': (1, DELTA),       # Paper: (t_event, t_event + δ]
            'post_event_decay': (DELTA + 1, ANALYSIS_WINDOW[1]) # Paper: (t_event + δ, end_window]
        }

        # Results storage
        self.volatility_model_data = None # Stores output from analyze_three_phase_volatility
        self.risk_adjusted_returns_data = None # Stores output from analyze_rvr_with_optimistic_bias (contains model-based RVR, mu_hat, sigma_e)
        self.sharpe_by_phase_data = None # Stores Sharpe calculated from risk_adjusted_returns_data
        self.rvr_by_phase_data = None    # Stores RVR by phase from risk_adjusted_returns_data
        self.h1_results = None

        os.makedirs(results_dir, exist_ok=True)

    def run_analysis(self):
        print("\n--- Running Comprehensive Analysis for Hypothesis 1 (Paper Version) ---")
        # 1. Illustrative Three-Phase Volatility (using unconditional GARCH variance as base for phi)
        self._analyze_illustrative_volatility_shape()
        # 2. Calculate RVR and Sharpe using unified volatility and dynamic bias
        self._calculate_model_based_risk_adjusted_returns()
        # 3. Test hypothesis statistically using model-based RVR and Sharpe
        self._test_hypothesis()
        # 4. Generate visualizations based on model outputs
        self._generate_visualizations()
        # 5. Create a comprehensive report
        self._create_report()
        return True

    def _analyze_illustrative_volatility_shape(self):
        """
        Analyze and plot the general shape of three-phase volatility.
        This uses the existing EventAnalysis method which typically bases phi functions on
        unconditional GARCH variance for simplicity in illustrating the average shape.
        """
        print("\n--- Analyzing Illustrative Three-Phase Volatility Shape ---")
        # This method is for illustrating the average shape of volatility
        # It might use unconditional GARCH variance as sigma_e0 for the phi functions
        volatility_df, _ = self.analyzer.analyze_three_phase_volatility(
            results_dir=self.results_dir,
            file_prefix=self.file_prefix + "_illustrative", # Differentiate output
            return_col=self.return_col,
            analysis_window=self.analysis_window,
            garch_type=self.garch_type,
            k1=self.k1, k2=self.k2,
            delta_t1=self.delta_t1, delta_t2=self.delta_t2, delta_t3=self.delta_t3,
            delta=self.delta
        )
        self.volatility_model_data = volatility_df # Store for potential reference

    def _calculate_model_based_risk_adjusted_returns(self):
        """
        Calculate RVR using the paper's unified volatility and dynamic bias model.
        Then, calculate Sharpe Ratio from these model-consistent metrics.
        Assumes EventAnalysis.analyze_rvr_with_optimistic_bias is updated per paper.
        """
        print("\n--- Calculating Model-Based RVR and Sharpe Ratio ---")

        # Calculate RVR using the model. This method is assumed to be updated in EventAnalysis
        # to use event-specific GJR-GARCH h_t, unified volatility sigma_e(t),
        # and the dynamic bias b_t.
        # It should return a DataFrame with 'days_to_event', 'mean_rvr',
        # 'mean_expected_return' (this is mu_hat_e_t), 'mean_volatility' (this is sigma_e_t)
        rvr_daily_df, rvr_phase_stats_df = self.analyzer.analyze_rvr_with_optimistic_bias(
            results_dir=self.results_dir,
            file_prefix=self.file_prefix, # Main RVR results
            return_col=self.return_col,
            analysis_window=self.analysis_window,
            garch_type=self.garch_type, # Pass GJR GARCH
            k1=self.k1, k2=self.k2,
            delta_t1=self.delta_t1, delta_t2=self.delta_t2, delta_t3=self.delta_t3,
            delta=self.delta,
            # Pass parameters for dynamic bias: b0_bias (as optimistic_bias) and kappa_bias
            optimistic_bias=self.b0_bias, # Renamed in call if method signature changes
            kappa_bias=self.kappa_bias,   # New parameter for EventAnalysis method
            risk_free_rate=self.risk_free_rate
        )

        self.risk_adjusted_returns_data = rvr_daily_df # This df contains model-based mu_hat and sigma_e
        self.rvr_by_phase_data = rvr_phase_stats_df # RVR phase stats from the method

        # Now, calculate Sharpe Ratio using the model-consistent mu_hat and sigma_e from rvr_daily_df
        if rvr_daily_df is not None and 'mean_expected_return' in rvr_daily_df.columns and 'mean_volatility' in rvr_daily_df.columns:
            # Ensure daily_rf is calculated correctly
            daily_rf = (1 + self.risk_free_rate)**(1/252) - 1 if self.risk_free_rate > 0 else 0.0

            # Calculate Sharpe Ratio: (mu_hat_e_t - r_f) / sigma_e_t
            # Annualize if sigma_e_t is daily standard deviation: multiply by sqrt(252)
            # The 'mean_volatility' from analyze_rvr should be sigma_e(t)
            # Assume analyze_rvr_with_optimistic_bias returns daily sigma_e(t)
            sharpe_df = rvr_daily_df.with_columns(
                ((pl.col('mean_expected_return') - daily_rf) / pl.col('mean_volatility') * np.sqrt(252)
                ).alias('sharpe_ratio')
            ).select(['days_to_event', 'sharpe_ratio', 'mean_expected_return', 'mean_volatility'])
            
            # Store this derived Sharpe data
            self.risk_adjusted_returns_data = self.risk_adjusted_returns_data.join(
                sharpe_df.select(['days_to_event', 'sharpe_ratio']),
                on='days_to_event',
                how='left'
            )
            
            # Calculate average Sharpe ratio by phase
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
                        'phase': phase_name, 'start_day': start_day, 'end_day': end_day,
                        'avg_sharpe': avg_sharpe, 'median_sharpe': median_sharpe
                    })
            
            if sharpe_phase_stats:
                self.sharpe_by_phase_data = pl.DataFrame(sharpe_phase_stats)
                self.sharpe_by_phase_data.write_csv(os.path.join(self.results_dir, f"{self.file_prefix}_sharpe_phase_stats.csv"))
                print("\nModel-Based Sharpe Ratio by Phase:")
                for row in self.sharpe_by_phase_data.iter_rows(named=True):
                    print(f"  Phase: {row['phase']}, Avg Sharpe: {row.get('avg_sharpe', float('nan')):.4f}")
        else:
            print("Warning: Could not calculate model-based Sharpe ratio as RVR data is missing or incomplete.")

    def _test_hypothesis(self):
        print("\n--- Testing Hypothesis 1 Statistically (Model-Based RVR & Sharpe) ---")
        if self.rvr_by_phase_data is None or self.sharpe_by_phase_data is None:
            print("Error: RVR or Sharpe phase data not available for hypothesis testing.")
            return

        # Extract RVR by phase
        pre_event_rvr = self.rvr_by_phase_data.filter(pl.col('phase') == 'pre_event')['avg_rvr'].item()
        post_rising_rvr = self.rvr_by_phase_data.filter(pl.col('phase') == 'post_event_rising')['avg_rvr'].item()
        post_decay_rvr = self.rvr_by_phase_data.filter(pl.col('phase') == 'post_event_decay')['avg_rvr'].item()
        
        rvr_h1_condition = (post_rising_rvr is not None and pre_event_rvr is not None and post_decay_rvr is not None and
                            post_rising_rvr > pre_event_rvr and post_rising_rvr > post_decay_rvr)
        print("\nRVR Hypothesis Test (Peak during post-event rising):")
        print(f"  Condition Met? {'YES' if rvr_h1_condition else 'NO'}")
        print(f"  Pre-event RVR: {pre_event_rvr:.4f}, Post-rising RVR: {post_rising_rvr:.4f}, Post-decay RVR: {post_decay_rvr:.4f}")

        # Extract Sharpe by phase
        pre_event_sharpe = self.sharpe_by_phase_data.filter(pl.col('phase') == 'pre_event')['avg_sharpe'].item()
        post_rising_sharpe = self.sharpe_by_phase_data.filter(pl.col('phase') == 'post_event_rising')['avg_sharpe'].item()
        post_decay_sharpe = self.sharpe_by_phase_data.filter(pl.col('phase') == 'post_event_decay')['avg_sharpe'].item()

        sharpe_h1_condition = (post_rising_sharpe is not None and pre_event_sharpe is not None and post_decay_sharpe is not None and
                               post_rising_sharpe > pre_event_sharpe and post_rising_sharpe > post_decay_sharpe)
        print("\nSharpe Ratio Hypothesis Test (Peak during post-event rising):")
        print(f"  Condition Met? {'YES' if sharpe_h1_condition else 'NO'}")
        print(f"  Pre-event Sharpe: {pre_event_sharpe:.4f}, Post-rising Sharpe: {post_rising_sharpe:.4f}, Post-decay Sharpe: {post_decay_sharpe:.4f}")

        # Statistical significance (t-tests on daily model-based data)
        # Assumes self.risk_adjusted_returns_data contains daily 'avg_rvr' and 'sharpe_ratio'
        rvr_p_values, sharpe_p_values = {}, {}
        if self.risk_adjusted_returns_data is not None:
            daily_data = self.risk_adjusted_returns_data
            
            post_rising_phase_def = self.phases['post_event_rising']
            daily_post_rising_rvr = daily_data.filter(
                (pl.col('days_to_event') >= post_rising_phase_def[0]) & (pl.col('days_to_event') <= post_rising_phase_def[1])
            )['avg_rvr'].drop_nulls().to_numpy()
            daily_post_rising_sharpe = daily_data.filter(
                (pl.col('days_to_event') >= post_rising_phase_def[0]) & (pl.col('days_to_event') <= post_rising_phase_def[1])
            )['sharpe_ratio'].drop_nulls().to_numpy()

            for phase_name, (start_day, end_day) in self.phases.items():
                if phase_name == 'post_event_rising': continue

                daily_phase_rvr = daily_data.filter(
                    (pl.col('days_to_event') >= start_day) & (pl.col('days_to_event') <= end_day)
                )['avg_rvr'].drop_nulls().to_numpy()
                daily_phase_sharpe = daily_data.filter(
                    (pl.col('days_to_event') >= start_day) & (pl.col('days_to_event') <= end_day)
                )['sharpe_ratio'].drop_nulls().to_numpy()

                if len(daily_post_rising_rvr) > 1 and len(daily_phase_rvr) > 1:
                    t_stat, p_val = stats.ttest_ind(daily_post_rising_rvr, daily_phase_rvr, equal_var=False, nan_policy='omit')
                    rvr_p_values[phase_name] = {'t_stat': t_stat, 'p_value': p_val, 'significant': p_val < 0.05 and t_stat > 0}
                if len(daily_post_rising_sharpe) > 1 and len(daily_phase_sharpe) > 1:
                    t_stat, p_val = stats.ttest_ind(daily_post_rising_sharpe, daily_phase_sharpe, equal_var=False, nan_policy='omit')
                    sharpe_p_values[phase_name] = {'t_stat': t_stat, 'p_value': p_val, 'significant': p_val < 0.05 and t_stat > 0}
            
            if rvr_p_values: print("\nRVR T-Tests (Post-Event Rising vs. Other Phases):")
            for phase, stats_dict in rvr_p_values.items(): print(f"  vs. {phase}: t={stats_dict['t_stat']:.2f}, p={stats_dict['p_value']:.3f}")
            if sharpe_p_values: print("\nSharpe T-Tests (Post-Event Rising vs. Other Phases):")
            for phase, stats_dict in sharpe_p_values.items(): print(f"  vs. {phase}: t={stats_dict['t_stat']:.2f}, p={stats_dict['p_value']:.3f}")

        significant_rvr_superiority = sum(1 for p_val_dict in rvr_p_values.values() if p_val_dict['significant'])
        significant_sharpe_superiority = sum(1 for p_val_dict in sharpe_p_values.values() if p_val_dict['significant'])
        
        # H1 supported if conditions met and statistically significant improvements
        rvr_supported_final = rvr_h1_condition and significant_rvr_superiority > 0
        sharpe_supported_final = sharpe_h1_condition and significant_sharpe_superiority > 0
        h1_supported_final = rvr_supported_final or sharpe_supported_final

        print("\nOverall Hypothesis 1 Result (Model-Based):")
        print(f"  Supported? {'YES' if h1_supported_final else 'NO'}")
        print(f"  RVR Evidence: {'YES' if rvr_supported_final else 'NO'} ({significant_rvr_superiority} sig. tests vs other phases)")
        print(f"  Sharpe Ratio Evidence: {'YES' if sharpe_supported_final else 'NO'} ({significant_sharpe_superiority} sig. tests vs other phases)")

        self.h1_results = {
            'hypothesis_supported': h1_supported_final,
            'rvr_supported': rvr_supported_final, 'sharpe_supported': sharpe_supported_final,
            'rvr_p_values': rvr_p_values, 'sharpe_p_values': sharpe_p_values,
            'rvr_by_phase': {'pre_event': pre_event_rvr, 'post_event_rising': post_rising_rvr, 'post_event_decay': post_decay_rvr},
            'sharpe_by_phase': {'pre_event': pre_event_sharpe, 'post_event_rising': post_rising_sharpe, 'post_event_decay': post_decay_sharpe}
        }
        h1_result_df = pl.DataFrame({
            'hypothesis': ['H1: Risk-adjusted returns peak during post-event rising phase'],
            'result': [h1_supported_final],
            'rvr_supported': [rvr_supported_final], 'sharpe_supported': [sharpe_supported_final],
            'pre_event_rvr': [pre_event_rvr], 'post_rising_rvr': [post_rising_rvr], 'post_decay_rvr': [post_decay_rvr],
            'pre_event_sharpe': [pre_event_sharpe], 'post_rising_sharpe': [post_rising_sharpe], 'post_decay_sharpe': [post_decay_sharpe],
            'significant_rvr_tests': [significant_rvr_superiority], 'significant_sharpe_tests': [significant_sharpe_superiority]
        })
        h1_result_df.write_csv(os.path.join(self.results_dir, f"{self.file_prefix}_hypothesis1_test.csv"))

    def _generate_visualizations(self):
        print("\n--- Generating Visualizations for Hypothesis 1 ---")
        if self.h1_results is None or self.risk_adjusted_returns_data is None:
            print("Warning: No hypothesis or daily data available for visualization.")
            return

        # Plot 1: Risk-Adjusted Returns (RVR & Sharpe) by Phase (Bar Chart)
        try:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
            phases_plot = ['pre_event', 'post_event_rising', 'post_event_decay']
            phase_labels = ['Pre-Event', 'Post-Event Rising', 'Post-Event Decay']
            
            rvr_values = [self.h1_results['rvr_by_phase'].get(p) for p in phases_plot]
            ax1.bar(phase_labels, rvr_values, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
            ax1.set_title('Model-Based RVR by Phase')
            ax1.set_ylabel('Average RVR')
            for i, v in enumerate(rvr_values): ax1.text(i, (v if v is not None else 0) * 1.05, f"{v:.3f}" if v is not None else "N/A", ha='center')

            sharpe_values = [self.h1_results['sharpe_by_phase'].get(p) for p in phases_plot]
            ax2.bar(phase_labels, sharpe_values, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
            ax2.set_title('Model-Based Sharpe Ratio by Phase')
            ax2.set_ylabel('Average Sharpe Ratio (Annualized)')
            for i, v in enumerate(sharpe_values): ax2.text(i, (v if v is not None else 0) * 1.05, f"{v:.3f}" if v is not None else "N/A", ha='center')
            
            fig.suptitle(f"H1: Risk-Adjusted Returns Peak Post-Event - Result: {'SUPPORTED' if self.h1_results['hypothesis_supported'] else 'NOT SUPPORTED'}", fontsize=14)
            plt.tight_layout(rect=[0, 0, 1, 0.96])
            plt.savefig(os.path.join(self.results_dir, f"{self.file_prefix}_H1_risk_adjusted_by_phase.png"), dpi=200)
            plt.close(fig)
        except Exception as e: print(f"Error in bar chart plot: {e}")

        # Plot 2: Timeseries of Model-Based RVR and Sharpe
        try:
            daily_data_pd = self.risk_adjusted_returns_data.to_pandas()
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

            ax1.plot(daily_data_pd['days_to_event'], daily_data_pd['avg_rvr'], color='blue', label='Mean RVR (Model-Based)')
            ax1.set_title('Model-Based RVR Around Events')
            ax1.set_ylabel('RVR')
            ax1.axvline(0, color='k', ls='--', lw=1); ax1.axvline(self.delta, color='grey', ls=':', lw=1, label=f'End Rising (day {self.delta})')
            ax1.legend()

            ax2.plot(daily_data_pd['days_to_event'], daily_data_pd['sharpe_ratio'], color='red', label='Mean Sharpe Ratio (Model-Based, Annualized)')
            ax2.set_title('Model-Based Sharpe Ratio Around Events')
            ax2.set_ylabel('Sharpe Ratio')
            ax2.axvline(0, color='k', ls='--', lw=1); ax2.axvline(self.delta, color='grey', ls=':', lw=1)
            ax2.legend()
            
            plt.xlabel('Days Relative to Event')
            plt.tight_layout()
            plt.savefig(os.path.join(self.results_dir, f"{self.file_prefix}_H1_risk_adjusted_timeseries.png"), dpi=200)
            plt.close(fig)
        except Exception as e: print(f"Error in timeseries plot: {e}")

        # Plot 3: Underlying mu_hat_e_t and sigma_e_t from model
        try:
            daily_data_pd = self.risk_adjusted_returns_data.to_pandas()
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

            ax1.plot(daily_data_pd['days_to_event'], daily_data_pd['mean_expected_return'] * 100, color='green', label=r'Mean $\hat{\mu}_{e,t}$ (Model-Based, % Daily)')
            ax1.set_title(r'Model-Based Expected Return ($\hat{\mu}_{e,t}$) Around Events')
            ax1.set_ylabel('Expected Return (%)')
            ax1.axhline(0, color='grey', ls='-', lw=0.5)
            ax1.axvline(0, color='k', ls='--', lw=1); ax1.axvline(self.delta, color='grey', ls=':', lw=1, label=f'End Rising (day {self.delta})')
            ax1.legend()

            ax2.plot(daily_data_pd['days_to_event'], daily_data_pd['mean_volatility'] * 100, color='purple', label=r'Mean $\sigma_e(t)$ (Unified Volatility, % Daily)')
            ax2.set_title(r'Model-Based Unified Volatility ($\sigma_e(t)$) Around Events')
            ax2.set_ylabel('Volatility (%)')
            ax2.axvline(0, color='k', ls='--', lw=1); ax2.axvline(self.delta, color='grey', ls=':', lw=1)
            ax2.legend()
            
            plt.xlabel('Days Relative to Event')
            plt.tight_layout()
            plt.savefig(os.path.join(self.results_dir, f"{self.file_prefix}_H1_mu_sigma_timeseries.png"), dpi=200)
            plt.close(fig)
        except Exception as e: print(f"Error in mu-sigma plot: {e}")

        print(f"Visualizations for Hypothesis 1 saved to: {self.results_dir}")

    def _create_report(self):
        print("\n--- Creating Comprehensive Report for Hypothesis 1 ---")
        if self.h1_results is None:
            print("Warning: No hypothesis results available for report.")
            return

        report_file = os.path.join(self.results_dir, f"{self.file_prefix}_H1_report.md")
        with open(report_file, 'w') as f:
            f.write(f"# Hypothesis 1 Analysis Report: {self.file_prefix.upper()}\n\n")
            f.write("## Hypothesis Statement (from paper)\n")
            f.write("> Risk-adjusted returns, specifically the return-to-variance ratio (RVR) and the Sharpe ratio, peak during the post-event rising phase ($t_{event} < t \\leq t_{event} + \\delta$). This peak is attributed to the combined effect of high unified volatility and endogenously increased expected returns through bias evolution, causing these ratios to exceed their levels in pre-event and late post-event phases.\n\n")

            f.write("## Overall Result\n")
            f.write(f"**Hypothesis 1 is {'SUPPORTED' if self.h1_results['hypothesis_supported'] else 'NOT SUPPORTED'}** based on the model-derived metrics.\n")
            f.write(f"- RVR Evidence: {'Supported' if self.h1_results['rvr_supported'] else 'Not supported'}\n")
            f.write(f"- Sharpe Ratio Evidence: {'Supported' if self.h1_results['sharpe_supported'] else 'Not supported'}\n\n")

            f.write("## Model-Based RVR by Phase\n")
            f.write("| Phase               | Avg RVR |\n")
            f.write("|---------------------|---------|\n")
            for phase in ['pre_event', 'post_event_rising', 'post_event_decay']:
                rvr = self.h1_results['rvr_by_phase'].get(phase)
                f.write(f"| {phase.replace('_', ' ').title():<19} | {rvr:.4f} |\n" if rvr is not None else f"| {phase.replace('_', ' ').title():<19} | N/A     |\n")
            
            f.write("\n## Model-Based Sharpe Ratio by Phase (Annualized)\n")
            f.write("| Phase               | Avg Sharpe |\n")
            f.write("|---------------------|------------|\n")
            for phase in ['pre_event', 'post_event_rising', 'post_event_decay']:
                sharpe = self.h1_results['sharpe_by_phase'].get(phase)
                f.write(f"| {phase.replace('_', ' ').title():<19} | {sharpe:.4f}   |\n" if sharpe is not None else f"| {phase.replace('_', ' ').title():<19} | N/A        |\n")

            f.write("\n## Statistical Tests (T-tests of Post-Event Rising vs. Other Phases)\n")
            if self.h1_results['rvr_p_values']:
                f.write("### RVR Tests:\n")
                for phase, stats_dict in self.h1_results['rvr_p_values'].items():
                    f.write(f"- vs. {phase}: t={stats_dict['t_stat']:.2f}, p={stats_dict['p_value']:.3f} ({'Significant improvement' if stats_dict['significant'] else 'Not significant'})\n")
            if self.h1_results['sharpe_p_values']:
                f.write("### Sharpe Ratio Tests:\n")
                for phase, stats_dict in self.h1_results['sharpe_p_values'].items():
                    f.write(f"- vs. {phase}: t={stats_dict['t_stat']:.2f}, p={stats_dict['p_value']:.3f} ({'Significant improvement' if stats_dict['significant'] else 'Not significant'})\n")
            
            f.write("\n## Key Parameters Used\n")
            f.write(f"- GARCH Model for $h_t$: {self.garch_type.upper()}\n")
            f.write(f"- Unified Volatility ($\sigma_e(t) = \sqrt{h_t} \cdot \Phi(t)$) Parameters:\n")
            f.write(f"  - $k_1 = {self.k1}$, $k_2 = {self.k2}$\n")
            f.write(f"  - $\Delta t_1 = {self.delta_t1}$, $\Delta t_2 = {self.delta_t2}$, $\Delta t_3 = {self.delta_t3}$\n")
            f.write(f"  - $\delta = {self.delta}$ (duration of post-event rising phase)\n")
            f.write(f"- Biased Expectations ($\hat{{\mu}}_{{e,t}} = \mu_{{e,t}} + b_0 \cdot \Omega_t$) Parameters:\n")
            f.write(f"  - $b_0 = {self.b0_bias}$ (baseline daily bias)\n")
            f.write(f"  - $\kappa = {self.kappa_bias}$ (optimism multiplier for bias evolution)\n")
            f.write(f"- Risk-free rate: {self.risk_free_rate}\n")

            f.write("\n## Conclusion\n")
            if self.h1_results['hypothesis_supported']:
                f.write("The analysis provides support for Hypothesis 1. Model-derived risk-adjusted returns (RVR and/or Sharpe Ratio) demonstrate a peak during the post-event rising phase, consistent with the theoretical model predictions of unified volatility and endogenous bias evolution.\n")
            else:
                f.write("The analysis does not provide clear support for Hypothesis 1 using the model-derived metrics. The predicted peak in risk-adjusted returns during the post-event rising phase was not consistently observed or statistically significant across both RVR and Sharpe Ratio.\n")
        
        print(f"Hypothesis 1 report saved to: {report_file}")


def run_fda_analysis():
    print("\n=== Starting FDA Approval Event Analysis for Hypothesis 1 (Paper Version) ===")
    # Path Validation & Results Dir Creation (omitted for brevity, same as original)
    os.makedirs(FDA_RESULTS_DIR, exist_ok=True)
    print(f"FDA results will be saved to: {os.path.abspath(FDA_RESULTS_DIR)}")

    try:
        data_loader = EventDataLoader(
            event_path=FDA_EVENT_FILE, stock_paths=STOCK_FILES, window_days=WINDOW_DAYS,
            event_date_col=FDA_EVENT_DATE_COL, ticker_col=FDA_TICKER_COL
        )
        # Feature engineering is not strictly needed for H1 if EventAnalysis methods handle it
        feature_engineer = EventFeatureEngineer(prediction_window=3) # Dummy prediction window
        analyzer = EventAnalysis(data_loader, feature_engineer)
        
        print("\nLoading and preparing FDA data (minimal features for H1)...")
        analyzer.data = analyzer.load_and_prepare_data(run_feature_engineering=False) # H1 primarily needs 'ret'
        if analyzer.data is None or analyzer.data.is_empty():
            print("\n*** Error: FDA data loading failed. ***")
            return False
        print(f"FDA data loaded. Shape: {analyzer.data.shape}")

        h1_analyzer = Hypothesis1Analyzer(analyzer, FDA_RESULTS_DIR, FDA_FILE_PREFIX)
        h1_analyzer.run_analysis()
        
        print(f"\n--- FDA Event Analysis for Hypothesis 1 Finished (Results in '{FDA_RESULTS_DIR}') ---")
        return True
    except Exception as e:
        print(f"\n*** An unexpected error occurred in FDA H1 analysis: {e} ***")
        traceback.print_exc()
    return False

def run_earnings_analysis():
    print("\n=== Starting Earnings Announcement Event Analysis for Hypothesis 1 (Paper Version) ===")
    os.makedirs(EARNINGS_RESULTS_DIR, exist_ok=True)
    print(f"Earnings results will be saved to: {os.path.abspath(EARNINGS_RESULTS_DIR)}")

    try:
        data_loader = EventDataLoader(
            event_path=EARNINGS_EVENT_FILE, stock_paths=STOCK_FILES, window_days=WINDOW_DAYS,
            event_date_col=EARNINGS_EVENT_DATE_COL, ticker_col=EARNINGS_TICKER_COL
        )
        feature_engineer = EventFeatureEngineer(prediction_window=3)
        analyzer = EventAnalysis(data_loader, feature_engineer)

        print("\nLoading and preparing Earnings data (minimal features for H1)...")
        analyzer.data = analyzer.load_and_prepare_data(run_feature_engineering=False)
        if analyzer.data is None or analyzer.data.is_empty():
            print("\n*** Error: Earnings data loading failed. ***")
            return False
        print(f"Earnings data loaded. Shape: {analyzer.data.shape}")

        h1_analyzer = Hypothesis1Analyzer(analyzer, EARNINGS_RESULTS_DIR, EARNINGS_FILE_PREFIX)
        h1_analyzer.run_analysis()

        print(f"\n--- Earnings Event Analysis for Hypothesis 1 Finished (Results in '{EARNINGS_RESULTS_DIR}') ---")
        return True
    except Exception as e:
        print(f"\n*** An unexpected error occurred in Earnings H1 analysis: {e} ***")
        traceback.print_exc()
    return False

def compare_results():
    """
    Compares the Hypothesis 1 test results between FDA and earnings events.
    (Adapted from the original testhyp1.py, assuming similar output CSVs are generated)
    """
    print("\n=== Comparing FDA and Earnings Results for Hypothesis 1 (Paper Version) ===")
    comparison_dir = "results/hypothesis1/comparison/"
    os.makedirs(comparison_dir, exist_ok=True)

    try:
        fda_h1_file = os.path.join(FDA_RESULTS_DIR, f"{FDA_FILE_PREFIX}_hypothesis1_test.csv")
        earnings_h1_file = os.path.join(EARNINGS_RESULTS_DIR, f"{EARNINGS_FILE_PREFIX}_hypothesis1_test.csv")

        if not os.path.exists(fda_h1_file) or not os.path.exists(earnings_h1_file):
            print(f"Error: One or more H1 test files missing. FDA: {fda_h1_file}, Earnings: {earnings_h1_file}")
            return False

        fda_h1_df = pl.read_csv(fda_h1_file)
        earnings_h1_df = pl.read_csv(earnings_h1_file)

        comparison_data = {
            'Event Type': ['FDA Approvals', 'Earnings Announcements'],
            'H1 Supported': [fda_h1_df['result'].item(), earnings_h1_df['result'].item()],
            'RVR Supported': [fda_h1_df['rvr_supported'].item(), earnings_h1_df['rvr_supported'].item()],
            'Sharpe Supported': [fda_h1_df['sharpe_supported'].item(), earnings_h1_df['sharpe_supported'].item()],
            'Post-Rising RVR': [fda_h1_df['post_rising_rvr'].item(), earnings_h1_df['post_rising_rvr'].item()],
            'Post-Rising Sharpe': [fda_h1_df['post_rising_sharpe'].item(), earnings_h1_df['post_rising_sharpe'].item()],
        }
        comparison_df = pl.DataFrame(comparison_data)
        comparison_df.write_csv(os.path.join(comparison_dir, "H1_comparison_summary.csv"))

        print("\nHypothesis 1 Comparison Summary:")
        print(comparison_df)

        # Plotting (Simplified version)
        fig, ax = plt.subplots(1, 2, figsize=(15, 6))
        metrics = ['RVR Supported', 'Sharpe Supported']
        bar_width = 0.35
        index = np.arange(len(metrics))

        fda_scores = [fda_h1_df[m.lower().replace(" ", "_")].item() for m in metrics]
        earnings_scores = [earnings_h1_df[m.lower().replace(" ", "_")].item() for m in metrics]

        ax[0].bar(index - bar_width/2, [1 if s else 0 for s in fda_scores], bar_width, label='FDA Supported', color='skyblue')
        ax[0].bar(index + bar_width/2, [1 if s else 0 for s in earnings_scores], bar_width, label='Earnings Supported', color='lightcoral')
        ax[0].set_ylabel('Supported (1=Yes, 0=No)')
        ax[0].set_xticks(index)
        ax[0].set_xticklabels(metrics)
        ax[0].set_title('H1 Metric Support')
        ax[0].legend()

        ax[1].bar(['FDA', 'Earnings'], 
                  [fda_h1_df['result'].item(), earnings_h1_df['result'].item()], 
                  color=['skyblue', 'lightcoral'])
        ax[1].set_ylabel('Overall H1 Supported (1=Yes, 0=No)')
        ax[1].set_title('Overall Hypothesis 1 Support')
        
        plt.tight_layout()
        plt.savefig(os.path.join(comparison_dir, "H1_comparison_plot.png"), dpi=200)
        plt.close(fig)
        print(f"Comparison plot saved to {comparison_dir}")
        return True

    except Exception as e:
        print(f"Error comparing H1 results: {e}")
        traceback.print_exc()
        return False

def main():
    fda_success = run_fda_analysis()
    earnings_success = run_earnings_analysis()

    if fda_success and earnings_success:
        compare_results()
    elif fda_success:
        print("\n=== FDA H1 analysis completed. Earnings analysis failed or was skipped. ===")
    elif earnings_success:
        print("\n=== Earnings H1 analysis completed. FDA analysis failed or was skipped. ===")
    else:
        print("\n=== Both FDA and Earnings H1 analyses failed. ===")

if __name__ == "__main__":
    main()