# testhyp1.py
# Script to test Hypothesis 1:
# Risk-adjusted returns (RVR and Sharpe ratio) peak during the post-event rising phase.

import pandas as pd
import numpy as np
import os
import sys
import traceback
import polars as pl
import matplotlib.pyplot as plt
from scipy import stats
from typing import List, Tuple, Dict, Optional, Any
from datetime import timedelta

# --- Configuration ---
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path: sys.path.append(current_dir)

# Adjust src_path to point to the directory containing 'event_processor.py' and 'models.py'
# Assuming 'src' is a subdirectory of the current script's directory
project_root = os.path.dirname(current_dir) # Goes up one level from the script's dir
src_path = os.path.join(project_root, 'src')
if src_path not in sys.path:
    sys.path.append(src_path)


try:
    # Assuming the original EventDataLoader is what you want for data loading
    # from the prompt's event_processor.py.
    # If you have a new one in your src/event_processor.py, adjust this.
    # For now, I'll assume the original structure for data loading.
    # Let's call the original `event_processor_legacy.py` if needed.
    # For now, let's assume testhyp1.py is in the same dir as event_processor.py and models.py
    # So, change the import to be relative to the script location if they are siblings
    # OR if they are in a 'src' folder one level up.
    # Given your setup, it's likely they are in a 'src' folder.

    # This import assumes EventDataLoader is in the *original* event_processor.py
    # And UnifiedVolatilityModel, RiskMetrics are in the *new* models.py
    # For simplicity, let's assume the original EventDataLoader logic is needed.
    # If EventProcessor from the new event_processor.py is to be used, the data loading part needs to change.

    # Using models from the *new* models.py
    from models import UnifiedVolatilityModel, RiskMetrics, InvestorParams, MarketParams, PortfolioOptimizer
    # For data loading, we will use the structure from the *original* testhyp1_improved.py
    # which implies an EventDataLoader class. Let's assume this is in the *new* event_processor.py
    # or that you want to re-implement that part here.
    # Given the request to "scrap what is there", I'll define a simplified data loading
    # and processing sequence here, calling your new models.

    print("Successfully imported models from models.py.")
except ImportError as e:
    print(f"Error importing modules: {e}")
    print(f"Python Path: {sys.path}")
    print("Ensure 'models.py' and 'event_processor.py' (if used for data loading) are accessible.")
    sys.exit(1)

pl.Config.set_tbl_rows(20)
pl.Config.set_tbl_cols(10)
pl.Config.set_fmt_str_lengths(80)

# --- Hardcoded Analysis Parameters ---
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
FDA_EVENT_DATE_COL = "Approval Date" # Match this to your CSV
FDA_TICKER_COL = "ticker"       # Match this to your CSV

# Earnings event specific parameters
EARNINGS_EVENT_FILE = "/home/d87016661/detail_history_actuals.csv"
EARNINGS_RESULTS_DIR = "results/hypothesis1/results_earnings/"
EARNINGS_FILE_PREFIX = "earnings"
EARNINGS_EVENT_DATE_COL = "ANNDATS" # Match this to your CSV
EARNINGS_TICKER_COL = "ticker"    # Match this to your CSV

# Shared analysis parameters
ANALYSIS_WINDOW_DAYS = 30 # Days before and after event for analysis
EVENT_DAY_ZERO = 0
POST_EVENT_RISING_DURATION_DELTA = 5 # `delta` from the paper

# Unified Volatility Model Parameters (example, adjust as needed)
# These should ideally be estimated or taken from literature/calibration
VOL_PARAMS_DEFAULT = {
    'omega': 1e-6, 'alpha': 0.08, 'beta': 0.9, 'gamma': 0.04, # GJR-GARCH
    'k1': 1.3, 'k2': 1.5, 'delta': POST_EVENT_RISING_DURATION_DELTA, # Event-specific
    'delta_t1': 5.0, 'delta_t2': 3.0, 'delta_t3': 10.0
}
OPTIMISTIC_BIAS_BASELINE = 0.001 # b0 in the paper
OPTIMISTIC_BIAS_KAPPA = 0.5    # kappa in the paper
RISK_FREE_RATE_DAILY = 0.045 / 252 # Example daily risk-free rate

class Hypothesis1Framework:
    def __init__(self, results_dir: str, file_prefix: str, vol_params_override: Optional[Dict] = None):
        self.results_dir = results_dir
        self.file_prefix = file_prefix
        os.makedirs(self.results_dir, exist_ok=True)

        vol_params_dict = VOL_PARAMS_DEFAULT.copy()
        if vol_params_override:
            vol_params_dict.update(vol_params_override)
        self.vol_model = UnifiedVolatilityModel(**vol_params_dict)
        self.risk_metrics = RiskMetrics()

        self.phases = {
            'pre_event': (-ANALYSIS_WINDOW_DAYS, -1),
            'event_day': (EVENT_DAY_ZERO, EVENT_DAY_ZERO),
            'post_event_rising': (EVENT_DAY_ZERO + 1, EVENT_DAY_ZERO + POST_EVENT_RISING_DURATION_DELTA),
            'post_event_decay': (EVENT_DAY_ZERO + POST_EVENT_RISING_DURATION_DELTA + 1, ANALYSIS_WINDOW_DAYS)
        }
        self.h1_results_summary = None


    def _load_and_prepare_data(self, event_file_path: str, stock_file_paths: List[str],
                               event_date_col: str, ticker_col: str) -> Optional[pl.DataFrame]:
        """
        Loads event data and stock data, merges them, and calculates returns.
        Simplified version for this script.
        """
        try:
            print(f"Loading event data from: {event_file_path}")
            # Use Polars for reading CSV
            event_df = pl.read_csv(event_file_path, try_parse_dates=True)
            event_df = event_df.rename({event_date_col: "event_date", ticker_col: "ticker"})
            event_df = event_df.with_columns([
                pl.col("event_date").cast(pl.Date), # Ensure Date type
                pl.col("ticker").cast(pl.Utf8)
            ])
            event_df = event_df.drop_nulls(subset=["event_date", "ticker"]).unique(subset=["ticker", "event_date"])
            if event_df.is_empty():
                print("No valid events found.")
                return None
            print(f"Loaded {event_df.height} unique events.")

            print(f"Loading stock data from {len(stock_file_paths)} files...")
            stock_df_list = []
            for f_path in stock_file_paths:
                try:
                    # Assuming parquet files have 'date', 'ticker', 'PRC', 'RET'
                    # Adjust column names as per your actual CRSP data structure
                    df = pl.read_parquet(f_path)
                    # Standardize common column name variations
                    rename_map = {}
                    if 'PERMNO' in df.columns and 'ticker' not in df.columns : rename_map['PERMNO'] = 'ticker' # Example
                    if 'date' not in df.columns and 'DATE' in df.columns: rename_map['DATE'] = 'date'
                    if 'PRC' not in df.columns and 'prc' in df.columns: rename_map['prc'] = 'PRC'
                    if 'RET' not in df.columns and 'ret' in df.columns: rename_map['ret'] = 'RET'
                    if rename_map:
                        df = df.rename(rename_map)

                    required_cols = ['date', 'ticker', 'PRC', 'RET']
                    if not all(col in df.columns for col in required_cols):
                        print(f"Skipping {f_path}, missing one of {required_cols}. Found: {df.columns}")
                        continue

                    df = df.select(required_cols)
                    df = df.with_columns([
                        pl.col("date").cast(pl.Date),
                        pl.col("ticker").cast(pl.Utf8), # Ensure ticker is string for join
                        pl.col("PRC").cast(pl.Float64),
                        pl.col("RET").cast(pl.Float64)
                    ])
                    stock_df_list.append(df)
                except Exception as e:
                    print(f"Error loading stock file {f_path}: {e}")
            if not stock_df_list:
                print("No stock data loaded.")
                return None

            stock_df = pl.concat(stock_df_list).drop_nulls().unique(subset=["ticker", "date"])
            print(f"Loaded and combined {stock_df.height} stock records.")

            # Merge event and stock data
            # Polars join expects date types to match exactly. Event date is Date, stock date is Date.
            merged_df = event_df.join(stock_df, on="ticker", how="inner")

            # Calculate days_to_event and filter window
            window_delta_pre = timedelta(days=ANALYSIS_WINDOW_DAYS + 60) # Larger window for GARCH
            window_delta_post = timedelta(days=ANALYSIS_WINDOW_DAYS + 5)

            merged_df = merged_df.with_columns(
                (pl.col("date") - pl.col("event_date")).dt.total_days().cast(pl.Int32).alias("days_to_event")
            )
            merged_df = merged_df.filter(
                (pl.col("days_to_event") >= -(ANALYSIS_WINDOW_DAYS + 50)) & # Window for GARCH estimation
                (pl.col("days_to_event") <= ANALYSIS_WINDOW_DAYS + 5)      # Window for analysis
            )
            merged_df = merged_df.sort(["ticker", "event_date", "date"])
            merged_df = merged_df.with_columns(
                pl.col("ticker").cast(str) + "_" + pl.col("event_date").dt.strftime("%Y%m%d")
            ).rename({"literal": "event_id"})


            # Fill missing RET if PRC is available (simplistic, real data might need more care)
            merged_df = merged_df.with_columns(
                pl.when(pl.col("RET").is_null() & pl.col("PRC").is_not_null() & pl.col("PRC").shift(1).over("event_id").is_not_null())
                .then( (pl.col("PRC") / pl.col("PRC").shift(1).over("event_id")) - 1)
                .otherwise(pl.col("RET"))
                .alias("RET")
            )
            merged_df = merged_df.drop_nulls(subset=["RET"]) # Drop rows where return cannot be computed

            if merged_df.is_empty():
                print("No data after merging and filtering.")
                return None

            print(f"Prepared data shape: {merged_df.shape}")
            return merged_df

        except Exception as e:
            print(f"Error in _load_and_prepare_data: {e}")
            traceback.print_exc()
            return None

    def _calculate_event_metrics(self, data: pl.DataFrame) -> pl.DataFrame:
        """
        For each event, calculate baseline volatility (from GJR-GARCH on pre-event data),
        then unified volatility, expected returns with bias, RVR, and Sharpe.
        """
        all_event_metrics = []

        for event_id_val, group_df_pl in data.group_by("event_id"):
            group_df = group_df_pl.to_pandas() # Easier iteration for GARCH-like logic
            group_df = group_df.sort_values("days_to_event")

            # Ensure enough data for baseline vol estimation
            pre_event_returns_for_garch = group_df[group_df['days_to_event'] < EVENT_DAY_ZERO]['RET'].dropna().to_numpy()
            if len(pre_event_returns_for_garch) < 30: # Min obs for GARCH
                # print(f"Skipping event {event_id_val} due to insufficient pre-event data for GARCH.")
                continue

            try:
                # Estimate baseline volatility (sigma_hat_t in paper) using GJR-GARCH from UnifiedVolatilityModel
                # This needs historical returns for the specific stock up to each point.
                # For simplicity, let's use a rolling GARCH or a single GARCH fit on pre-event window.
                # The paper's UnifiedVolatilityModel takes GJR-GARCH params as input.
                # Here, we'll simulate its output using rolling std as a proxy for h_t.
                # A full GJR-GARCH estimation per event-day is computationally intensive.
                # Let's use the provided UnifiedVolatilityModel's phi functions with a simplified baseline.
                
                # Simplified baseline_vol_series: rolling std of past returns
                # Or, estimate one set of GJR-GARCH params per stock and use that.
                # For now, let's use a fixed average pre-event volatility as baseline_vol_scalar
                # This aligns with how baseline_volatility is used in ThreePhaseVolatilityModel in existing event_processor
                
                baseline_vol_scalar = np.std(pre_event_returns_for_garch)
                if baseline_vol_scalar < 1e-6: # Avoid division by zero
                    baseline_vol_scalar = 1e-6


                group_df['baseline_vol_metric'] = baseline_vol_scalar # Fixed for the event
                
                group_df['unified_vol_metric'] = group_df.apply(
                    lambda row: self.vol_model.unified_volatility(
                        t=row['days_to_event'],
                        baseline_vol=baseline_vol_scalar, # Use the estimated scalar
                        t_event=EVENT_DAY_ZERO
                    ), axis=1
                )

                group_df['bias_metric'] = group_df.apply(
                    lambda row: self.vol_model.bias_parameter(
                        t=row['days_to_event'],
                        baseline_vol=baseline_vol_scalar,
                        b0=OPTIMISTIC_BIAS_BASELINE,
                        kappa=OPTIMISTIC_BIAS_KAPPA,
                        t_event=EVENT_DAY_ZERO
                    ), axis=1
                )
                group_df['expected_ret_metric'] = group_df['RET'] + group_df['bias_metric'] # Daily return + bias

                # Calculate RVR and Sharpe using RiskMetrics
                group_df['rvr_metric'] = group_df.apply(
                    lambda row: self.risk_metrics.return_to_variance_ratio(
                        expected_return=row['expected_ret_metric'],
                        risk_free_rate=RISK_FREE_RATE_DAILY,
                        volatility=row['unified_vol_metric']
                    ), axis=1
                )
                group_df['sharpe_metric'] = group_df.apply(
                    lambda row: self.risk_metrics.sharpe_ratio(
                        expected_return=row['expected_ret_metric'],
                        risk_free_rate=RISK_FREE_RATE_DAILY,
                        volatility=row['unified_vol_metric']
                    ), axis=1
                )
                all_event_metrics.append(pl.from_pandas(group_df))
            except Exception as e:
                print(f"Error processing event {event_id_val}: {e}")
                # traceback.print_exc() # Optional: for detailed debugging
                continue
        
        if not all_event_metrics:
            print("No event metrics could be calculated.")
            return pl.DataFrame()
            
        return pl.concat(all_event_metrics)

    def analyze_hypothesis1(self, data_with_metrics: pl.DataFrame):
        """
        Test Hypothesis 1 by comparing RVR and Sharpe across phases.
        """
        if data_with_metrics.is_empty() or 'rvr_metric' not in data_with_metrics.columns:
            print("No data or metrics to analyze for Hypothesis 1.")
            self.h1_results_summary = pl.DataFrame({
                "metric": ["RVR_peak", "Sharpe_peak"],
                "supported": [False, False],
                "pre_event_mean": [None, None],
                "post_rising_mean": [None, None],
                "post_decay_mean": [None, None],
                "p_value_rising_vs_pre": [None, None],
                "p_value_rising_vs_decay": [None, None]
            })
            return

        phase_metrics_list = []
        for phase_name, (start_day, end_day) in self.phases.items():
            phase_data = data_with_metrics.filter(
                (pl.col('days_to_event') >= start_day) & (pl.col('days_to_event') <= end_day)
            )
            if not phase_data.is_empty():
                phase_metrics_list.append({
                    'phase': phase_name,
                    'avg_rvr': phase_data['rvr_metric'].mean(),
                    'median_rvr': phase_data['rvr_metric'].median(),
                    'avg_sharpe': phase_data['sharpe_metric'].mean(),
                    'median_sharpe': phase_data['sharpe_metric'].median(),
                    'rvr_values': phase_data['rvr_metric'].drop_nulls().to_list(), # For t-tests
                    'sharpe_values': phase_data['sharpe_metric'].drop_nulls().to_list() # For t-tests
                })
        
        phase_summary_df = pl.DataFrame(phase_metrics_list)
        print("\n--- Risk-Adjusted Metrics by Phase ---")
        print(phase_summary_df.select(['phase', 'avg_rvr', 'avg_sharpe']))

        # Save phase summary
        phase_summary_df.drop(['rvr_values', 'sharpe_values']).write_csv(
            os.path.join(self.results_dir, f"{self.file_prefix}_h1_phase_summary.csv")
        )

        # Hypothesis 1 Test
        results = []
        for metric_name, values_col, avg_col in [
            ("RVR", "rvr_values", "avg_rvr"),
            ("Sharpe", "sharpe_values", "avg_sharpe")
        ]:
            try:
                pre_event_data = phase_summary_df.filter(pl.col('phase') == 'pre_event')
                post_rising_data = phase_summary_df.filter(pl.col('phase') == 'post_event_rising')
                post_decay_data = phase_summary_df.filter(pl.col('phase') == 'post_event_decay')

                if pre_event_data.is_empty() or post_rising_data.is_empty() or post_decay_data.is_empty():
                    print(f"Warning: Missing data for one or more phases for {metric_name}. Skipping test.")
                    results.append({
                        "metric": f"{metric_name}_peak", "supported": False,
                        "pre_event_mean": None, "post_rising_mean": None, "post_decay_mean": None,
                        "p_value_rising_vs_pre": None, "p_value_rising_vs_decay": None
                    })
                    continue

                pre_event_mean = pre_event_data[avg_col][0]
                post_rising_mean = post_rising_data[avg_col][0]
                post_decay_mean = post_decay_data[avg_col][0]

                hypothesis_supported = (post_rising_mean > pre_event_mean) and \
                                       (post_rising_mean > post_decay_mean)

                # Statistical Significance (t-test)
                pre_values = pre_event_data[values_col][0]
                rising_values = post_rising_data[values_col][0]
                decay_values = post_decay_data[values_col][0]

                p_val_vs_pre, p_val_vs_decay = None, None
                if len(rising_values) > 1 and len(pre_values) > 1:
                    # One-sided t-test: post_rising > pre_event
                    t_stat_pre, p_val_pre = stats.ttest_ind(rising_values, pre_values, equal_var=False, alternative='greater')
                    p_val_vs_pre = p_val_pre
                if len(rising_values) > 1 and len(decay_values) > 1:
                     # One-sided t-test: post_rising > post_decay
                    t_stat_decay, p_val_decay = stats.ttest_ind(rising_values, decay_values, equal_var=False, alternative='greater')
                    p_val_vs_decay = p_val_decay
                
                # More robust check for support: mean difference AND statistical significance (e.g., p < 0.1)
                statistically_supported = hypothesis_supported
                if p_val_vs_pre is not None and p_val_vs_pre >= 0.1: # If p-value is high, not significant
                    statistically_supported = False
                if p_val_vs_decay is not None and p_val_vs_decay >= 0.1:
                    statistically_supported = False


                print(f"\nHypothesis 1 for {metric_name}:")
                print(f"  Pre-event Mean: {pre_event_mean:.4f}")
                print(f"  Post-event Rising Mean: {post_rising_mean:.4f} (p-vs-pre: {p_val_vs_pre:.4f if p_val_vs_pre is not None else 'N/A'})")
                print(f"  Post-event Decay Mean: {post_decay_mean:.4f} (p-vs-decay: {p_val_vs_decay:.4f if p_val_vs_decay is not None else 'N/A'})")
                print(f"  Peak during post-event rising (mean comparison)? {'YES' if hypothesis_supported else 'NO'}")
                print(f"  Peak statistically supported (p < 0.1)? {'YES' if statistically_supported else 'NO'}")


                results.append({
                    "metric": f"{metric_name}_peak",
                    "supported": statistically_supported, # Use statistically_supported
                    "pre_event_mean": pre_event_mean,
                    "post_rising_mean": post_rising_mean,
                    "post_decay_mean": post_decay_mean,
                    "p_value_rising_vs_pre": p_val_vs_pre,
                    "p_value_rising_vs_decay": p_val_vs_decay
                })

            except Exception as e:
                print(f"Error testing Hypothesis 1 for {metric_name}: {e}")
                results.append({
                    "metric": f"{metric_name}_peak", "supported": False,
                    "pre_event_mean": None, "post_rising_mean": None, "post_decay_mean": None,
                    "p_value_rising_vs_pre": None, "p_value_rising_vs_decay": None
                })

        self.h1_results_summary = pl.DataFrame(results)
        print("\n--- Hypothesis 1 Summary ---")
        print(self.h1_results_summary)
        self.h1_results_summary.write_csv(
            os.path.join(self.results_dir, f"{self.file_prefix}_h1_results.csv")
        )
        self._generate_visualizations(data_with_metrics.to_pandas(), phase_summary_df.to_pandas())

    def _generate_visualizations(self, data_pd: pd.DataFrame, phase_summary_pd: pd.DataFrame):
        """Generate plots for RVR and Sharpe Ratio dynamics."""
        if data_pd.empty:
            print("No data to visualize.")
            return

        # Plot 1: Time series of RVR and Sharpe
        fig, axs = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

        daily_avg = data_pd.groupby('days_to_event')[['rvr_metric', 'sharpe_metric']].mean().reset_index()

        axs[0].plot(daily_avg['days_to_event'], daily_avg['rvr_metric'], label='Average RVR', color='blue')
        axs[0].set_ylabel('RVR')
        axs[0].set_title(f'{self.file_prefix.upper()}: Average RVR Around Event')
        axs[0].grid(True, linestyle=':')

        axs[1].plot(daily_avg['days_to_event'], daily_avg['sharpe_metric'], label='Average Sharpe Ratio', color='red')
        axs[1].set_ylabel('Sharpe Ratio')
        axs[1].set_title(f'{self.file_prefix.upper()}: Average Sharpe Ratio Around Event')
        axs[1].set_xlabel('Days Relative to Event')
        axs[1].grid(True, linestyle=':')

        for ax in axs:
            ax.axvline(EVENT_DAY_ZERO, color='black', linestyle='--', label='Event Day')
            ax.axvline(EVENT_DAY_ZERO + POST_EVENT_RISING_DURATION_DELTA, color='grey', linestyle=':', label='End of Rising Phase')
            ax.legend()

        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, f"{self.file_prefix}_h1_timeseries_plot.png"), dpi=200)
        plt.close(fig)
        print(f"Saved time series plot to {self.results_dir}")

        # Plot 2: Bar chart of RVR and Sharpe by phase
        fig, axs = plt.subplots(1, 2, figsize=(16, 6))
        
        relevant_phases = phase_summary_pd[phase_summary_pd['phase'].isin(['pre_event', 'post_event_rising', 'post_event_decay'])]

        axs[0].bar(relevant_phases['phase'], relevant_phases['avg_rvr'], color=['skyblue', 'salmon', 'lightgreen'])
        axs[0].set_ylabel('Average RVR')
        axs[0].set_title(f'{self.file_prefix.upper()}: Average RVR by Phase')

        axs[1].bar(relevant_phases['phase'], relevant_phases['avg_sharpe'], color=['skyblue', 'salmon', 'lightgreen'])
        axs[1].set_ylabel('Average Sharpe Ratio')
        axs[1].set_title(f'{self.file_prefix.upper()}: Average Sharpe by Phase')
        
        if self.h1_results_summary is not None:
            rvr_supported = self.h1_results_summary.filter(pl.col("metric")=="RVR_peak")["supported"][0]
            sharpe_supported = self.h1_results_summary.filter(pl.col("metric")=="Sharpe_peak")["supported"][0]
            fig.suptitle(f"H1 Support - RVR: {'YES' if rvr_supported else 'NO'}, Sharpe: {'YES' if sharpe_supported else 'NO'}",
                         fontsize=14, fontweight='bold')


        plt.tight_layout(rect=[0, 0, 1, 0.96]) # Adjust layout to make space for suptitle
        plt.savefig(os.path.join(self.results_dir, f"{self.file_prefix}_h1_phase_barplot.png"), dpi=200)
        plt.close(fig)
        print(f"Saved phase bar plot to {self.results_dir}")


    def run_full_analysis(self, event_file_path: str, stock_file_paths: List[str],
                          event_date_col: str, ticker_col: str):
        print(f"\n--- Starting Hypothesis 1 Analysis for {self.file_prefix.upper()} ---")
        data = self._load_and_prepare_data(event_file_path, stock_file_paths, event_date_col, ticker_col)
        if data is None or data.is_empty():
            print(f"Failed to load or prepare data for {self.file_prefix}. Aborting H1 analysis.")
            return False

        print("Calculating event metrics (volatility, RVR, Sharpe)...")
        data_with_metrics = self._calculate_event_metrics(data)
        if data_with_metrics.is_empty():
            print(f"Failed to calculate metrics for {self.file_prefix}. Aborting H1 analysis.")
            return False

        print("Analyzing metrics for Hypothesis 1...")
        self.analyze_hypothesis1(data_with_metrics)
        
        data_with_metrics.write_csv(os.path.join(self.results_dir, f"{self.file_prefix}_full_data_with_metrics.csv"))
        print(f"Saved full data with metrics to {self.results_dir}")
        print(f"--- Hypothesis 1 Analysis for {self.file_prefix.upper()} Complete ---")
        return True


def run_specific_analysis(event_type: str):
    if event_type == "FDA":
        analyzer = Hypothesis1Framework(results_dir=FDA_RESULTS_DIR, file_prefix=FDA_FILE_PREFIX)
        success = analyzer.run_full_analysis(
            event_file_path=FDA_EVENT_FILE,
            stock_file_paths=STOCK_FILES,
            event_date_col=FDA_EVENT_DATE_COL,
            ticker_col=FDA_TICKER_COL
        )
    elif event_type == "Earnings":
        analyzer = Hypothesis1Framework(results_dir=EARNINGS_RESULTS_DIR, file_prefix=EARNINGS_FILE_PREFIX)
        success = analyzer.run_full_analysis(
            event_file_path=EARNINGS_EVENT_FILE,
            stock_file_paths=STOCK_FILES,
            event_date_col=EARNINGS_EVENT_DATE_COL,
            ticker_col=EARNINGS_TICKER_COL
        )
    else:
        print(f"Unknown event type: {event_type}")
        return False
    return success

def compare_results():
    print("\n=== Comparing FDA and Earnings Results for Hypothesis 1 ===")
    comparison_dir = "results/hypothesis1/comparison/"
    os.makedirs(comparison_dir, exist_ok=True)

    fda_h1_file = os.path.join(FDA_RESULTS_DIR, f"{FDA_FILE_PREFIX}_h1_results.csv")
    earnings_h1_file = os.path.join(EARNINGS_RESULTS_DIR, f"{EARNINGS_FILE_PREFIX}_h1_results.csv")

    if not os.path.exists(fda_h1_file) or not os.path.exists(earnings_h1_file):
        print("One or both H1 result files are missing. Cannot compare.")
        return

    fda_results = pl.read_csv(fda_h1_file)
    earnings_results = pl.read_csv(earnings_h1_file)

    comparison_data = []
    metrics = ["RVR_peak", "Sharpe_peak"]
    for metric in metrics:
        fda_metric_row = fda_results.filter(pl.col("metric") == metric)
        earnings_metric_row = earnings_results.filter(pl.col("metric") == metric)

        comparison_data.append({
            "metric": metric,
            "fda_supported": fda_metric_row["supported"][0] if not fda_metric_row.is_empty() else None,
            "fda_post_rising_mean": fda_metric_row["post_rising_mean"][0] if not fda_metric_row.is_empty() else None,
            "earnings_supported": earnings_metric_row["supported"][0] if not earnings_metric_row.is_empty() else None,
            "earnings_post_rising_mean": earnings_metric_row["post_rising_mean"][0] if not earnings_metric_row.is_empty() else None,
        })
    
    comparison_df = pl.DataFrame(comparison_data)
    print("\n--- H1 Comparison Summary ---")
    print(comparison_df)
    comparison_df.write_csv(os.path.join(comparison_dir, "h1_comparison_summary.csv"))

    # Visualization for comparison
    fig, axs = plt.subplots(1, 2, figsize=(15, 6))
    bar_width = 0.35
    index = np.arange(len(metrics))

    fda_means = [comparison_df.filter(pl.col("metric") == m)["fda_post_rising_mean"][0] for m in metrics]
    earnings_means = [comparison_df.filter(pl.col("metric") == m)["earnings_post_rising_mean"][0] for m in metrics]
    
    # Filter out None values before plotting, or handle them (e.g., plot as 0 or skip)
    fda_means_plot = [x if x is not None else 0 for x in fda_means]
    earnings_means_plot = [x if x is not None else 0 for x in earnings_means]


    axs[0].bar(index - bar_width/2, fda_means_plot, bar_width, label='FDA', color='blue')
    axs[0].bar(index + bar_width/2, earnings_means_plot, bar_width, label='Earnings', color='orange')
    axs[0].set_ylabel('Mean Value in Post-Event Rising Phase')
    axs[0].set_xticks(index)
    axs[0].set_xticklabels([m.split('_')[0] for m in metrics]) # RVR, Sharpe
    axs[0].set_title('Comparison of Post-Event Rising Means')
    axs[0].legend()

    fda_supported_plot = [1 if comparison_df.filter(pl.col("metric") == m)["fda_supported"][0] else 0 for m in metrics]
    earnings_supported_plot = [1 if comparison_df.filter(pl.col("metric") == m)["earnings_supported"][0] else 0 for m in metrics]

    axs[1].bar(index - bar_width/2, fda_supported_plot, bar_width, label='FDA Supported', color='lightblue')
    axs[1].bar(index + bar_width/2, earnings_supported_plot, bar_width, label='Earnings Supported', color='peachpuff')
    axs[1].set_ylabel('Hypothesis Supported (1=Yes, 0=No)')
    axs[1].set_xticks(index)
    axs[1].set_xticklabels([m.split('_')[0] for m in metrics])
    axs[1].set_yticks([0, 1])
    axs[1].set_yticklabels(['No', 'Yes'])
    axs[1].set_title('H1 Support Comparison')
    axs[1].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(comparison_dir, "h1_comparison_plot.png"), dpi=200)
    plt.close(fig)
    print(f"Saved H1 comparison plot to {comparison_dir}")


def main():
    # Run analyses for each event type
    fda_success = run_specific_analysis("FDA")
    earnings_success = run_specific_analysis("Earnings")

    # Compare results if both analyses were successful
    if fda_success and earnings_success:
        compare_results()
    else:
        print("One or both analyses failed, skipping comparison.")

if __name__ == "__main__":
    main()