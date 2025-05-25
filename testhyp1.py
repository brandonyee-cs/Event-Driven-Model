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
# Ensure 'src' and the directory containing 'config.py' are in the Python path
# Assuming 'testhyp1.py' is in the main project directory,
# and 'src' and 'config.py' are also in the main project directory or a 'src' subdirectory.
# If config.py is in the same dir as testhyp1.py:
if current_dir not in sys.path:
    sys.path.append(current_dir)
# If src is a subdir:
src_dir = os.path.join(current_dir, 'src')
if src_dir not in sys.path:
    sys.path.append(src_dir)


try:
    from src.event_processor import EventProcessor, VolatilityParameters
    from src.models import RiskMetrics # UnifiedVolatilityModel is used internally by EventProcessor
    from src.config import Config # Import the Config class
    print("Successfully imported EventProcessor, models, and Config.")
except ImportError as e:
    print(f"Error importing modules: {e}")
    print(f"Python Path: {sys.path}")
    print("Ensure 'event_processor.py', 'models.py', and 'config.py' are accessible.")
    sys.exit(1)

pl.Config.set_tbl_rows(10)
pl.Config.set_tbl_cols(12)
pl.Config.set_fmt_str_lengths(70)

# --- Hardcoded Analysis Parameters (could also be moved to Config) ---
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

# Get relevant parameters from Config
app_config = Config()
ANALYSIS_WINDOW_DAYS = app_config.event_window_post # Using post window for symmetric analysis range
EVENT_DAY_ZERO = 0
POST_EVENT_RISING_DURATION_DELTA = app_config.event_delta
RISK_FREE_RATE_DAILY = app_config.risk_free_rate_daily

def load_stock_data(stock_file_paths: List[str]) -> Optional[pl.DataFrame]:
    """Loads and combines stock data from multiple parquet files."""
    stock_df_list = []
    print(f"Loading stock data from {len(stock_file_paths)} files...")
    for f_path in stock_file_paths:
        try:
            df = pl.read_parquet(f_path)
            # Standardize column names (case-insensitive check then rename)
            # CRSP specific: PERMNO, date, PRC, RET. Assume 'ticker' is desired.
            cols = {col.upper(): col for col in df.columns}
            rename_map = {}
            if 'PERMNO' in cols and 'TICKER' not in cols: rename_map[cols['PERMNO']] = 'ticker'
            elif 'TICKER' in cols: rename_map[cols['TICKER']] = 'ticker'

            if 'DATE' in cols: rename_map[cols['DATE']] = 'date'
            if 'PRC' in cols: rename_map[cols['PRC']] = 'price' # 'price' is used by EventProcessor
            if 'RET' in cols: rename_map[cols['RET']] = 'returns'# 'returns' is used by EventProcessor

            df = df.rename(rename_map)
            required_cols = ['date', 'ticker', 'price', 'returns']
            if not all(col in df.columns for col in required_cols):
                # print(f"Skipping {f_path}, missing one of {required_cols}. Found: {df.columns}")
                continue

            df = df.select(required_cols).with_columns([
                pl.col("date").cast(pl.Date),
                pl.col("ticker").cast(pl.Utf8),
                pl.col("price").cast(pl.Float64),
                pl.col("returns").cast(pl.Float64)
            ])
            stock_df_list.append(df)
        except Exception as e:
            print(f"Error loading stock file {f_path}: {e}")
            # traceback.print_exc()
    if not stock_df_list:
        print("No stock data loaded.")
        return None
    stock_df = pl.concat(stock_df_list).drop_nulls(subset=['date', 'ticker', 'price', 'returns'])
    stock_df = stock_df.unique(subset=["ticker", "date"], keep="first")
    print(f"Loaded and combined {stock_df.height} unique stock records.")
    return stock_df

def load_event_data(event_file_path: str, event_date_col: str, ticker_col: str) -> Optional[pl.DataFrame]:
    """Loads event data from a CSV file."""
    try:
        print(f"Loading event data from: {event_file_path}")
        event_df = pl.read_csv(event_file_path, try_parse_dates=True, infer_schema_length=10000)
        
        # Robust column renaming
        actual_event_date_col = event_date_col
        if event_date_col not in event_df.columns:
            potential_cols = [c for c in event_df.columns if event_date_col.lower() in c.lower()]
            if potential_cols: actual_event_date_col = potential_cols[0]
            else: raise ValueError(f"Event date column '{event_date_col}' not found.")
        
        actual_ticker_col = ticker_col
        if ticker_col not in event_df.columns:
            potential_cols = [c for c in event_df.columns if ticker_col.lower() in c.lower()]
            if potential_cols: actual_ticker_col = potential_cols[0]
            else: raise ValueError(f"Ticker column '{ticker_col}' not found.")

        event_df = event_df.rename({actual_event_date_col: "event_date", actual_ticker_col: "symbol"}) # EventProcessor expects 'symbol'
        
        event_df = event_df.with_columns([
            pl.col("event_date").cast(pl.Date),
            pl.col("symbol").cast(pl.Utf8).str.strip_chars() # Clean ticker
        ])
        event_df = event_df.drop_nulls(subset=["event_date", "symbol"])
        event_df = event_df.unique(subset=["symbol", "event_date"], keep="first")
        if event_df.is_empty():
            print("No valid events found.")
            return None
        print(f"Loaded {event_df.height} unique events.")
        return event_df
    except Exception as e:
        print(f"Error loading event data from {event_file_path}: {e}")
        traceback.print_exc()
        return None


class Hypothesis1Tester:
    def __init__(self, config: Config, results_dir: str, file_prefix: str):
        self.config = config
        self.event_processor = EventProcessor(config)
        self.risk_metrics = RiskMetrics() # From models.py
        self.results_dir = results_dir
        self.file_prefix = file_prefix
        os.makedirs(self.results_dir, exist_ok=True)

        self.phases = {
            'pre_event': (-self.config.event_window_pre, -1), # Use config
            'event_day': (0, 0),
            'post_event_rising': (1, self.config.event_delta),
            'post_event_decay': (self.config.event_delta + 1, self.config.event_window_post)
        }
        self.h1_results_summary = None

    def run_analysis(self, stock_data_df: pl.DataFrame, event_data_df: pl.DataFrame):
        print(f"\n--- Running H1 Analysis for {self.file_prefix.upper()} ---")
        if stock_data_df is None or stock_data_df.is_empty() or \
           event_data_df is None or event_data_df.is_empty():
            print("Stock or event data is missing. Aborting.")
            return False

        # EventProcessor.process_events returns a pandas DataFrame
        print("Processing events with EventProcessor...")
        processed_events_pd = self.event_processor.process_events(
            price_data=stock_data_df.to_pandas(), # EventProcessor expects pandas
            event_data=event_data_df.to_pandas()
        )

        if processed_events_pd is None or processed_events_pd.empty:
            print("No events processed by EventProcessor. Aborting.")
            return False
        
        processed_events_pl = pl.from_pandas(processed_events_pd)
        print(f"EventProcessor returned {processed_events_pl.shape[0]} records.")
        
        # Ensure necessary columns from EventProcessor are present
        # Expected columns from EventProcessor: 'symbol', 'date', 'price', 'returns',
        # 'event_date', 'event_type', 'days_to_event', 'baseline_volatility',
        # 'unified_volatility', 'bias_parameter', 'expected_return',
        # 'rvr', 'sharpe_ratio', 'phase', 'impact_uncertainty', 'volatility_innovation'
        
        required_metric_cols = ['days_to_event', 'rvr', 'sharpe_ratio', 'phase']
        if not all(col in processed_events_pl.columns for col in required_metric_cols):
            print(f"Missing one or more required columns from EventProcessor output: {required_metric_cols}")
            print(f"Available columns: {processed_events_pl.columns}")
            return False
            
        self.analyze_hypothesis1_from_processed(processed_events_pl)
        
        processed_events_pl.write_csv(os.path.join(self.results_dir, f"{self.file_prefix}_h1_processed_event_data.csv"))
        print(f"Saved processed event data for H1 to {self.results_dir}")
        return True

    def analyze_hypothesis1_from_processed(self, processed_data_pl: pl.DataFrame):
        """
        Test Hypothesis 1 using data already processed by EventProcessor.
        """
        phase_metrics_list = []
        for phase_name, (start_day, end_day) in self.phases.items():
            phase_data = processed_data_pl.filter(
                (pl.col('days_to_event') >= start_day) & (pl.col('days_to_event') <= end_day)
            )
            if not phase_data.is_empty():
                phase_metrics_list.append({
                    'phase': phase_name,
                    'avg_rvr': phase_data['rvr'].mean(),
                    'median_rvr': phase_data['rvr'].median(),
                    'avg_sharpe': phase_data['sharpe_ratio'].mean(),
                    'median_sharpe': phase_data['sharpe_ratio'].median(),
                    'rvr_values': phase_data['rvr'].drop_nulls().to_list(),
                    'sharpe_values': phase_data['sharpe_ratio'].drop_nulls().to_list()
                })
        
        phase_summary_df = pl.DataFrame(phase_metrics_list)
        print("\n--- Risk-Adjusted Metrics by Phase (from EventProcessor) ---")
        print(phase_summary_df.select(['phase', 'avg_rvr', 'avg_sharpe']))

        phase_summary_df.drop(['rvr_values', 'sharpe_values']).write_csv(
            os.path.join(self.results_dir, f"{self.file_prefix}_h1_phase_summary.csv")
        )

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
                    print(f"Warning: Missing data for H1 test of {metric_name}.")
                    results.append({
                        "metric": f"{metric_name}_peak", "supported": False,
                        "pre_event_mean": None, "post_rising_mean": None, "post_decay_mean": None,
                        "p_value_rising_vs_pre": None, "p_value_rising_vs_decay": None
                    })
                    continue
                
                pre_event_mean = pre_event_data[avg_col][0]
                post_rising_mean = post_rising_data[avg_col][0]
                post_decay_mean = post_decay_data[avg_col][0]

                # Check if means are not None before comparison
                if pre_event_mean is None or post_rising_mean is None or post_decay_mean is None:
                     hypothesis_supported_mean = False
                else:
                    hypothesis_supported_mean = (post_rising_mean > pre_event_mean) and \
                                                (post_rising_mean > post_decay_mean)
                
                pre_values = pre_event_data[values_col][0]
                rising_values = post_rising_data[values_col][0]
                decay_values = post_decay_data[values_col][0]

                p_val_vs_pre, p_val_vs_decay = None, None
                if len(rising_values) > 1 and len(pre_values) > 1:
                    t_stat_pre, p_val_pre = stats.ttest_ind(rising_values, pre_values, equal_var=False, alternative='greater', nan_policy='omit')
                    p_val_vs_pre = p_val_pre
                if len(rising_values) > 1 and len(decay_values) > 1:
                    t_stat_decay, p_val_decay = stats.ttest_ind(rising_values, decay_values, equal_var=False, alternative='greater', nan_policy='omit')
                    p_val_vs_decay = p_val_decay
                
                statistically_supported = hypothesis_supported_mean
                if p_val_vs_pre is not None and p_val_vs_pre >= self.config.alpha_significance:
                    statistically_supported = False
                if p_val_vs_decay is not None and p_val_vs_decay >= self.config.alpha_significance:
                    statistically_supported = False

                print(f"\nHypothesis 1 for {metric_name}:")
                print(f"  Pre-event Mean: {pre_event_mean if pre_event_mean is not None else 'N/A':.4f}")
                print(f"  Post-event Rising Mean: {post_rising_mean if post_rising_mean is not None else 'N/A':.4f} (p-vs-pre: {p_val_vs_pre:.4f if p_val_vs_pre is not None else 'N/A'})")
                print(f"  Post-event Decay Mean: {post_decay_mean if post_decay_mean is not None else 'N/A':.4f} (p-vs-decay: {p_val_vs_decay:.4f if p_val_vs_decay is not None else 'N/A'})")
                print(f"  Peak during post-event rising (mean comparison)? {'YES' if hypothesis_supported_mean else 'NO'}")
                print(f"  Peak statistically supported (p < {self.config.alpha_significance})? {'YES' if statistically_supported else 'NO'}")

                results.append({
                    "metric": f"{metric_name}_peak", "supported": statistically_supported,
                    "pre_event_mean": pre_event_mean, "post_rising_mean": post_rising_mean,
                    "post_decay_mean": post_decay_mean,
                    "p_value_rising_vs_pre": p_val_vs_pre, "p_value_rising_vs_decay": p_val_vs_decay
                })
            except Exception as e:
                print(f"Error testing H1 for {metric_name}: {e}")
                # traceback.print_exc()
                results.append({
                    "metric": f"{metric_name}_peak", "supported": False,
                    "pre_event_mean": None, "post_rising_mean": None, "post_decay_mean": None,
                    "p_value_rising_vs_pre": None, "p_value_rising_vs_decay": None})

        self.h1_results_summary = pl.DataFrame(results)
        print("\n--- Hypothesis 1 Summary ---")
        print(self.h1_results_summary)
        self.h1_results_summary.write_csv(
            os.path.join(self.results_dir, f"{self.file_prefix}_h1_results.csv")
        )
        self._generate_visualizations(processed_data_pl.to_pandas(), phase_summary_df.to_pandas())


    def _generate_visualizations(self, data_pd: pd.DataFrame, phase_summary_pd: pd.DataFrame):
        """Generate plots for RVR and Sharpe Ratio dynamics."""
        if data_pd.empty:
            print("No data to visualize.")
            return

        fig, axs = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
        daily_avg = data_pd.groupby('days_to_event')[['rvr', 'sharpe_ratio']].mean().reset_index()

        axs[0].plot(daily_avg['days_to_event'], daily_avg['rvr'], label='Average RVR', color='blue')
        axs[0].set_ylabel('RVR')
        axs[0].set_title(f'{self.file_prefix.upper()}: Average RVR Around Event')
        axs[0].grid(True, linestyle=':')

        axs[1].plot(daily_avg['days_to_event'], daily_avg['sharpe_ratio'], label='Average Sharpe Ratio', color='red')
        axs[1].set_ylabel('Sharpe Ratio')
        axs[1].set_title(f'{self.file_prefix.upper()}: Average Sharpe Ratio Around Event')
        axs[1].set_xlabel('Days Relative to Event')
        axs[1].grid(True, linestyle=':')

        for ax in axs:
            ax.axvline(EVENT_DAY_ZERO, color='black', linestyle='--', label='Event Day')
            ax.axvline(EVENT_DAY_ZERO + self.config.event_delta, color='grey', linestyle=':', label='End of Rising Phase')
            ax.legend()

        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, f"{self.file_prefix}_h1_timeseries_plot.png"), dpi=200)
        plt.close(fig)

        fig, axs = plt.subplots(1, 2, figsize=(16, 6))
        relevant_phases = phase_summary_pd[phase_summary_pd['phase'].isin(self.phases.keys())]
        
        axs[0].bar(relevant_phases['phase'], relevant_phases['avg_rvr'], color=['skyblue', 'lightcoral', 'salmon', 'lightgreen'])
        axs[0].set_ylabel('Average RVR')
        axs[0].set_title(f'{self.file_prefix.upper()}: Average RVR by Phase')
        axs[0].tick_params(axis='x', rotation=45)


        axs[1].bar(relevant_phases['phase'], relevant_phases['avg_sharpe'], color=['skyblue', 'lightcoral', 'salmon', 'lightgreen'])
        axs[1].set_ylabel('Average Sharpe Ratio')
        axs[1].set_title(f'{self.file_prefix.upper()}: Average Sharpe by Phase')
        axs[1].tick_params(axis='x', rotation=45)
        
        if self.h1_results_summary is not None:
            rvr_supported_row = self.h1_results_summary.filter(pl.col("metric")=="RVR_peak")
            sharpe_supported_row = self.h1_results_summary.filter(pl.col("metric")=="Sharpe_peak")
            rvr_supported = rvr_supported_row["supported"][0] if not rvr_supported_row.is_empty() else False
            sharpe_supported = sharpe_supported_row["supported"][0] if not sharpe_supported_row.is_empty() else False
            fig.suptitle(f"H1 Support - RVR: {'YES' if rvr_supported else 'NO'}, Sharpe: {'YES' if sharpe_supported else 'NO'}",
                         fontsize=14, fontweight='bold')

        plt.tight_layout(rect=[0, 0.05, 1, 0.95])
        plt.savefig(os.path.join(self.results_dir, f"{self.file_prefix}_h1_phase_barplot.png"), dpi=200)
        plt.close(fig)
        print(f"Saved H1 plots to {self.results_dir}")


def run_event_type_analysis(event_type_name: str, event_file: str, event_date_col: str, ticker_col: str, results_dir: str, file_prefix: str, global_stock_df: pl.DataFrame, app_config: Config):
    print(f"\n>>>> Starting H1 analysis for {event_type_name.upper()} EVENTS <<<<")
    event_data_df = load_event_data(event_file, event_date_col, ticker_col)
    if event_data_df is None or event_data_df.is_empty():
        print(f"Could not load event data for {event_type_name}. Skipping.")
        return False

    tester = Hypothesis1Tester(config=app_config, results_dir=results_dir, file_prefix=file_prefix)
    success = tester.run_analysis(global_stock_df, event_data_df)
    print(f">>>> H1 analysis for {event_type_name.upper()} EVENTS {'SUCCESSFUL' if success else 'FAILED'} <<<<")
    return success

def compare_h1_results():
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
    
    # Ensure 'metric' column exists before proceeding
    if 'metric' not in fda_results.columns or 'metric' not in earnings_results.columns:
        print("Column 'metric' missing in one of the results files. Cannot compare.")
        return

    comparison_data = []
    metrics_to_compare = ["RVR_peak", "Sharpe_peak"]
    for metric_val in metrics_to_compare:
        fda_metric_row = fda_results.filter(pl.col("metric") == metric_val)
        earnings_metric_row = earnings_results.filter(pl.col("metric") == metric_val)

        comparison_data.append({
            "metric": metric_val,
            "fda_supported": fda_metric_row["supported"][0] if not fda_metric_row.is_empty() and "supported" in fda_metric_row.columns else None,
            "fda_post_rising_mean": fda_metric_row["post_rising_mean"][0] if not fda_metric_row.is_empty() and "post_rising_mean" in fda_metric_row.columns else None,
            "earnings_supported": earnings_metric_row["supported"][0] if not earnings_metric_row.is_empty() and "supported" in earnings_metric_row.columns else None,
            "earnings_post_rising_mean": earnings_metric_row["post_rising_mean"][0] if not earnings_metric_row.is_empty() and "post_rising_mean" in earnings_metric_row.columns else None,
        })
    
    comparison_df = pl.DataFrame(comparison_data)
    print("\n--- H1 Comparison Summary ---")
    print(comparison_df)
    comparison_df.write_csv(os.path.join(comparison_dir, "h1_comparison_summary.csv"))

    fig, axs = plt.subplots(1, 2, figsize=(15, 6))
    bar_width = 0.35
    index = np.arange(len(metrics_to_compare))

    fda_means = [row["fda_post_rising_mean"] for row in comparison_df.to_dicts() if row["fda_post_rising_mean"] is not None]
    earnings_means = [row["earnings_post_rising_mean"] for row in comparison_df.to_dicts() if row["earnings_post_rising_mean"] is not None]
    
    if len(fda_means) != len(metrics_to_compare) or len(earnings_means) != len(metrics_to_compare):
         print("Warning: Mismatch in mean data length for plotting, likely due to None values. Plot may be incomplete.")
         # Pad with zeros or handle appropriately if you want to plot partial data
         fda_means = [x if x is not None else 0 for x in [row["fda_post_rising_mean"] for row in comparison_df.to_dicts()]]
         earnings_means = [x if x is not None else 0 for x in [row["earnings_post_rising_mean"] for row in comparison_df.to_dicts()]]


    axs[0].bar(index - bar_width/2, fda_means, bar_width, label='FDA', color='blue')
    axs[0].bar(index + bar_width/2, earnings_means, bar_width, label='Earnings', color='orange')
    axs[0].set_ylabel('Mean Value in Post-Event Rising Phase')
    axs[0].set_xticks(index)
    axs[0].set_xticklabels([m.split('_')[0] for m in metrics_to_compare])
    axs[0].set_title('Comparison of Post-Event Rising Means')
    axs[0].legend()

    fda_supported = [1 if row["fda_supported"] else 0 for row in comparison_df.to_dicts()]
    earnings_supported = [1 if row["earnings_supported"] else 0 for row in comparison_df.to_dicts()]

    axs[1].bar(index - bar_width/2, fda_supported, bar_width, label='FDA Supported', color='lightblue')
    axs[1].bar(index + bar_width/2, earnings_supported, bar_width, label='Earnings Supported', color='peachpuff')
    axs[1].set_ylabel('Hypothesis Supported (1=Yes, 0=No)')
    axs[1].set_xticks(index)
    axs[1].set_xticklabels([m.split('_')[0] for m in metrics_to_compare])
    axs[1].set_yticks([0, 1])
    axs[1].set_yticklabels(['No', 'Yes'])
    axs[1].set_title('H1 Support Comparison')
    axs[1].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(comparison_dir, "h1_comparison_plot.png"), dpi=200)
    plt.close(fig)
    print(f"Saved H1 comparison plot to {comparison_dir}")


def main():
    app_config = Config() # Initialize configuration

    # Load stock data once
    global_stock_df = load_stock_data(STOCK_FILES)
    if global_stock_df is None or global_stock_df.is_empty():
        print("Failed to load global stock data. Exiting.")
        return

    # Run FDA analysis
    fda_success = run_event_type_analysis(
        "FDA", FDA_EVENT_FILE, FDA_EVENT_DATE_COL, FDA_TICKER_COL,
        FDA_RESULTS_DIR, FDA_FILE_PREFIX, global_stock_df, app_config
    )

    # Run Earnings analysis
    earnings_success = run_event_type_analysis(
        "Earnings", EARNINGS_EVENT_FILE, EARNINGS_EVENT_DATE_COL, EARNINGS_TICKER_COL,
        EARNINGS_RESULTS_DIR, EARNINGS_FILE_PREFIX, global_stock_df, app_config
    )

    if fda_success and earnings_success:
        compare_h1_results()
    else:
        print("One or both H1 analyses failed, skipping comparison.")

if __name__ == "__main__":
    main()