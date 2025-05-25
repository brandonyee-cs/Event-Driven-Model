# testhyp2.py
# Script to test Hypothesis 2:
# GARCH-estimated conditional volatility innovations serve as an effective proxy for impact uncertainty.
#   2.1: Pre-event volatility innovations predict subsequent returns.
#   2.2: Post-event volatility persistence extends elevated expected returns.
#   2.3: Asymmetric volatility response (GJR-GARCH gamma) correlates with asymmetric price adjustment.

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
if current_dir not in sys.path:
    sys.path.append(current_dir)
src_dir = os.path.join(current_dir, 'src')
if src_dir not in sys.path:
    sys.path.append(src_dir)


try:
    from src.event_processor import EventProcessor, VolatilityParameters # Uses EventProcessor
    # UnifiedVolatilityModel is used internally by EventProcessor
    from src.config import Config
    print("Successfully imported EventProcessor, models, and Config.")
except ImportError as e:
    print(f"Error importing modules: {e}")
    print(f"Python Path: {sys.path}")
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
FDA_RESULTS_DIR = "results/hypothesis2/results_fda/"
FDA_FILE_PREFIX = "fda"
FDA_EVENT_DATE_COL = "Approval Date"
FDA_TICKER_COL = "ticker"

# Earnings event specific parameters
EARNINGS_EVENT_FILE = "/home/d87016661/detail_history_actuals.csv"
EARNINGS_RESULTS_DIR = "results/hypothesis2/results_earnings/"
EARNINGS_FILE_PREFIX = "earnings"
EARNINGS_EVENT_DATE_COL = "ANNDATS"
EARNINGS_TICKER_COL = "ticker"

# Get relevant parameters from Config
app_config = Config()
PREDICTION_HORIZONS = app_config.prediction_horizons
VOL_PERSISTENCE_WINDOW_DAYS = app_config.vol_persistence_window_days
PRE_EVENT_INNOV_WINDOW = (-10, -1) # Window for averaging pre-event innovations

# Data loading functions (re-used from testhyp1.py, ensure consistency)
def load_stock_data(stock_file_paths: List[str]) -> Optional[pl.DataFrame]:
    stock_df_list = []
    print(f"Loading stock data from {len(stock_file_paths)} files...")
    for f_path in stock_file_paths:
        try:
            df = pl.read_parquet(f_path)
            cols = {col.upper(): col for col in df.columns}
            rename_map = {}
            if 'PERMNO' in cols and 'TICKER' not in cols: rename_map[cols['PERMNO']] = 'ticker'
            elif 'TICKER' in cols: rename_map[cols['TICKER']] = 'ticker'
            if 'DATE' in cols: rename_map[cols['DATE']] = 'date'
            if 'PRC' in cols: rename_map[cols['PRC']] = 'price'
            if 'RET' in cols: rename_map[cols['RET']] = 'returns'
            df = df.rename(rename_map)
            required_cols = ['date', 'ticker', 'price', 'returns']
            if not all(col in df.columns for col in required_cols): continue
            df = df.select(required_cols).with_columns([
                pl.col("date").cast(pl.Date), pl.col("ticker").cast(pl.Utf8),
                pl.col("price").cast(pl.Float64), pl.col("returns").cast(pl.Float64)
            ])
            stock_df_list.append(df)
        except Exception as e: print(f"Error loading stock file {f_path}: {e}")
    if not stock_df_list: return None
    stock_df = pl.concat(stock_df_list).drop_nulls(subset=['date', 'ticker', 'price', 'returns'])
    stock_df = stock_df.unique(subset=["ticker", "date"], keep="first")
    print(f"Loaded and combined {stock_df.height} unique stock records.")
    return stock_df

def load_event_data(event_file_path: str, event_date_col: str, ticker_col: str) -> Optional[pl.DataFrame]:
    try:
        print(f"Loading event data from: {event_file_path}")
        event_df = pl.read_csv(event_file_path, try_parse_dates=True, infer_schema_length=10000)
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
        event_df = event_df.rename({actual_event_date_col: "event_date", actual_ticker_col: "symbol"})
        event_df = event_df.with_columns([
            pl.col("event_date").cast(pl.Date),
            pl.col("symbol").cast(pl.Utf8).str.strip_chars()
        ]).drop_nulls(subset=["event_date", "symbol"]).unique(subset=["symbol", "event_date"], keep="first")
        if event_df.is_empty(): return None
        print(f"Loaded {event_df.height} unique events.")
        return event_df
    except Exception as e:
        print(f"Error loading event data from {event_file_path}: {e}")
        return None

class Hypothesis2Tester:
    def __init__(self, config: Config, results_dir: str, file_prefix: str):
        self.config = config
        self.event_processor = EventProcessor(config) # EventProcessor will handle vol estimation
        self.results_dir = results_dir
        self.file_prefix = file_prefix
        os.makedirs(self.results_dir, exist_ok=True)
        self.h2_results_summary = {} # To store DataFrames for H2.1, H2.2, H2.3

    def run_analysis(self, stock_data_df: pl.DataFrame, event_data_df: pl.DataFrame):
        print(f"\n--- Running H2 Analysis for {self.file_prefix.upper()} ---")
        if stock_data_df is None or stock_data_df.is_empty() or \
           event_data_df is None or event_data_df.is_empty():
            print("Stock or event data is missing. Aborting H2.")
            return False

        print("Processing events with EventProcessor for H2...")
        processed_events_pd = self.event_processor.process_events(
            price_data=stock_data_df.to_pandas(),
            event_data=event_data_df.to_pandas()
        )

        if processed_events_pd is None or processed_events_pd.empty:
            print("No events processed by EventProcessor for H2. Aborting.")
            return False
        
        processed_events_pl = pl.from_pandas(processed_events_pd)
        print(f"EventProcessor (H2) returned {processed_events_pl.shape[0]} records.")

        # Ensure necessary columns are present. EventProcessor should add:
        # 'volatility_innovation' (as impact_uncertainty or similar), 'baseline_volatility' (as h_t)
        # and it should have estimated GJR-GARCH params per stock (stored in EventProcessor.volatility_params)
        required_cols_h2 = ['symbol', 'event_date', 'days_to_event', 'returns',
                              'baseline_volatility', 'volatility_innovation']
        if not all(col in processed_events_pl.columns for col in required_cols_h2):
            print(f"Missing one or more required columns from EventProcessor output for H2: {required_cols_h2}")
            print(f"Available columns: {processed_events_pl.columns}")
            # Attempt to derive 'volatility_innovation' if 'impact_uncertainty' exists and 'baseline_volatility' exists
            if 'impact_uncertainty' in processed_events_pl.columns and 'baseline_volatility' in processed_events_pl.columns:
                # The paper's "ImpactUncertainty_t = h_t - E_{t-1}[h_t]"
                # The EventProcessor's "impact_uncertainty = unified_volatility - baseline_volatility"
                # These are different. H2 needs GARCH innovations.
                # The current `EventProcessor._add_impact_uncertainty` calculates `unified - baseline`.
                # We need GARCH_ht - GARCH_pred_ht.
                # This requires a change in EventProcessor or re-calculation here.
                # For now, assuming 'volatility_innovation' is correctly GARCH-based.
                # If not, EventProcessor needs adjustment.
                print("Assuming 'volatility_innovation' is the GARCH-based innovation.")
            else:
                return False

        # Add unique event_id for grouping
        processed_events_pl = processed_events_pl.with_columns(
            (pl.col("symbol").cast(str) + "_" + pl.col("event_date").dt.strftime("%Y%m%d")).alias("event_id")
        )
        
        self._test_h2_1(processed_events_pl)
        self._test_h2_2(processed_events_pl)
        self._test_h2_3(processed_events_pl) # Needs access to GJR-gamma from EventProcessor
        
        processed_events_pl.write_csv(os.path.join(self.results_dir, f"{self.file_prefix}_h2_processed_event_data.csv"))
        print(f"Saved processed event data for H2 to {self.results_dir}")
        self._generate_h2_visualizations(processed_events_pl)
        return True

    def _test_h2_1(self, data: pl.DataFrame):
        """H2.1: Pre-event volatility innovations predict subsequent returns."""
        print("\n--- Testing H2.1: Volatility Innovations Predict Returns ---")
        
        avg_pre_innov = data.filter(
            (pl.col('days_to_event') >= PRE_EVENT_INNOV_WINDOW[0]) &
            (pl.col('days_to_event') <= PRE_EVENT_INNOV_WINDOW[1])
        ).group_by("event_id").agg(
            pl.mean("volatility_innovation").alias("avg_pre_event_vol_innov")
        ).drop_nulls()

        h2_1_results_list = []
        for k in self.config.prediction_horizons:
            data_with_future_ret = data.with_columns(
                pl.col("returns").shift(-k).over("event_id").alias(f"future_ret_{k}d")
            )
            
            regression_df = data_with_future_ret.filter(pl.col("days_to_event") == 0).join(
                avg_pre_innov, on="event_id", how="inner"
            ).select(["event_id", f"future_ret_{k}d", "avg_pre_event_vol_innov"]).drop_nulls()

            if regression_df.height > 10:
                X = regression_df["avg_pre_event_vol_innov"].to_numpy()
                y = regression_df[f"future_ret_{k}d"].to_numpy()
                
                if np.var(X) < 1e-10:
                    slope, intercept, r_value, p_value, std_err = 0, np.mean(y) if len(y)>0 else 0, 0, 1.0, 0
                else:
                    slope, intercept, r_value, p_value, std_err = stats.linregress(X, y)
                
                supported = p_value < self.config.alpha_significance # Assuming positive or negative relation can be support
                h2_1_results_list.append({
                    "horizon": k, "slope": slope, "intercept": intercept, "r_squared": r_value**2,
                    "p_value": p_value, "supported": supported, "n_obs": len(regression_df)
                })
                print(f"  H2.1 (Ret {k}d): Slope={slope:.4f}, R2={r_value**2:.3f}, p={p_value:.3f}, Supported={supported}, N={len(regression_df)}")
            else:
                 h2_1_results_list.append({"horizon": k, "slope": None, "intercept":None, "r_squared": None, "p_value": None, "supported": False, "n_obs": len(regression_df)})
        
        self.h2_results_summary['H2_1'] = pl.DataFrame(h2_1_results_list)
        if not self.h2_results_summary['H2_1'].is_empty():
            self.h2_results_summary['H2_1'].write_csv(os.path.join(self.results_dir, f"{self.file_prefix}_h2_1_results.csv"))

    def _test_h2_2(self, data: pl.DataFrame):
        """H2.2: Post-event volatility persistence extends elevated expected returns."""
        print("\n--- Testing H2.2: Volatility Persistence Extends Returns ---")
        
        # Use 'baseline_volatility' as h_t from GJR-GARCH
        event_vol_metrics = data.group_by("event_id").agg(
            pl.mean("baseline_volatility").filter(
                (pl.col("days_to_event") >= -self.config.lookback_days) & # Pre-event window for baseline
                (pl.col("days_to_event") <= -1)
            ).alias("avg_pre_event_h"),
            pl.mean("baseline_volatility").filter(
                (pl.col("days_to_event") >= 1) &
                (pl.col("days_to_event") <= self.config.vol_persistence_window_days)
            ).alias("avg_post_event_h"),
            pl.mean("expected_return").filter( # 'expected_return' from EventProcessor includes bias
                (pl.col("days_to_event") >= 1) &
                (pl.col("days_to_event") <= self.config.vol_persistence_window_days)
            ).alias("avg_post_event_exp_return")
        ).drop_nulls()

        if event_vol_metrics.is_empty():
            print("  H2.2: No events with valid pre/post volatility for persistence.")
            self.h2_results_summary['H2_2'] = pl.DataFrame()
            return

        event_vol_metrics = event_vol_metrics.with_columns(
            (pl.col("avg_post_event_h") / pl.col("avg_pre_event_h").clip_min(1e-9)).alias("vol_persistence_ratio")
        ).drop_nulls()
        
        if event_vol_metrics.height > 10:
            X = event_vol_metrics["vol_persistence_ratio"].to_numpy()
            y = event_vol_metrics["avg_post_event_exp_return"].to_numpy()

            if np.var(X) < 1e-10:
                slope, intercept, r_value, p_value, std_err = 0, np.mean(y) if len(y)>0 else 0, 0, 1.0, 0
            else:
                slope, intercept, r_value, p_value, std_err = stats.linregress(X, y)
            
            supported = p_value < self.config.alpha_significance and slope > 0 # Expect positive relation
            result = {"slope": slope, "intercept": intercept, "r_squared": r_value**2, "p_value": p_value, "supported": supported, "n_obs": len(event_vol_metrics)}
            print(f"  H2.2: Slope={slope:.4f}, R2={r_value**2:.3f}, p={p_value:.3f}, Supported={supported}, N={len(event_vol_metrics)}")
        else:
            result = {"slope": None, "intercept":None, "r_squared": None, "p_value": None, "supported": False, "n_obs": len(event_vol_metrics)}
        
        self.h2_results_summary['H2_2'] = pl.DataFrame([result])
        if not self.h2_results_summary['H2_2'].is_empty():
            self.h2_results_summary['H2_2'].write_csv(os.path.join(self.results_dir, f"{self.file_prefix}_h2_2_results.csv"))

    def _test_h2_3(self, data: pl.DataFrame):
        """H2.3: Asymmetric volatility response (GJR-GARCH gamma) correlates with asymmetric price adjustment."""
        print("\n--- Testing H2.3: Asymmetric Volatility Response ---")
        # EventProcessor stores estimated GJR-GARCH params in self.event_processor.volatility_params[symbol]
        # We need to extract gamma for each symbol and then average it.
        
        all_gammas = []
        for symbol_key, vol_params_instance in self.event_processor.volatility_params.items():
            if isinstance(vol_params_instance, VolatilityParameters):
                 all_gammas.append(vol_params_instance.gamma)

        if not all_gammas:
            print("  H2.3: No GJR-GARCH gamma parameters found from EventProcessor.")
            self.h2_results_summary['H2_3'] = pl.DataFrame()
            return
            
        gammas_np = np.array(all_gammas)
        gammas_np = gammas_np[~np.isnan(gammas_np)] # Remove NaNs if any

        if len(gammas_np) > 1:
            avg_gamma = np.mean(gammas_np)
            t_stat, p_value_one_sided = stats.ttest_1samp(gammas_np, 0, alternative='greater') # Test if gamma > 0
            supported = p_value_one_sided < self.config.alpha_significance and avg_gamma > 0
            
            print(f"  H2.3: Average GJR-GARCH Gamma = {avg_gamma:.4f} (from {len(gammas_np)} stocks)")
            print(f"  T-test (gamma > 0): t-stat={t_stat:.3f}, p-value={p_value_one_sided:.3f}, Supported={supported}")
            result = {"avg_gamma": avg_gamma, "p_value_gamma_gt_0": p_value_one_sided, "supported": supported, "n_stocks": len(gammas_np)}
        else:
            print("  H2.3: Insufficient gamma values for statistical test.")
            result = {"avg_gamma": None, "p_value_gamma_gt_0": None, "supported": False, "n_stocks": len(gammas_np)}

        self.h2_results_summary['H2_3'] = pl.DataFrame([result])
        if not self.h2_results_summary['H2_3'].is_empty():
            self.h2_results_summary['H2_3'].write_csv(os.path.join(self.results_dir, f"{self.file_prefix}_h2_3_results.csv"))
            
    def _generate_h2_visualizations(self, data_pl: pl.DataFrame):
        print(f"Generating H2 visualizations for {self.file_prefix}...")
        # Plot for H2.1 (if significant)
        if 'H2_1' in self.h2_results_summary and not self.h2_results_summary['H2_1'].is_empty():
            h2_1_res_df = self.h2_results_summary['H2_1']
            supported_h2_1 = h2_1_res_df.filter(pl.col("supported") == True)
            if not supported_h2_1.is_empty():
                # Reconstruct data for plot
                avg_pre_innov_pl = data_pl.filter(
                    (pl.col('days_to_event') >= PRE_EVENT_INNOV_WINDOW[0]) &
                    (pl.col('days_to_event') <= PRE_EVENT_INNOV_WINDOW[1])
                ).group_by("event_id").agg(
                    pl.mean("volatility_innovation").alias("avg_pre_event_vol_innov")
                ).drop_nulls()

                for row_dict in supported_h2_1.to_dicts():
                    k = row_dict['horizon']
                    slope = row_dict['slope']
                    intercept = row_dict['intercept']
                    
                    data_with_future_ret_pl = data_pl.with_columns(
                        pl.col("returns").shift(-k).over("event_id").alias(f"future_ret_{k}d")
                    )
                    plot_df_pl = data_with_future_ret_pl.filter(pl.col("days_to_event") == 0).join(
                        avg_pre_innov_pl, on="event_id", how="inner"
                    ).select(["event_id", f"future_ret_{k}d", "avg_pre_event_vol_innov"]).drop_nulls()
                    
                    if not plot_df_pl.is_empty() and slope is not None and intercept is not None:
                        plot_df_pd = plot_df_pl.to_pandas()
                        plt.figure(figsize=(8,6))
                        plt.scatter(plot_df_pd["avg_pre_event_vol_innov"], plot_df_pd[f"future_ret_{k}d"], alpha=0.5, label="Data points")
                        x_vals_plot = np.array(plt.xlim())
                        y_vals_plot = intercept + slope * x_vals_plot
                        plt.plot(x_vals_plot, y_vals_plot, color='red', label=f"Fit: y={slope:.3f}x+{intercept:.3f}\np={row_dict['p_value']:.3f}")
                        plt.xlabel("Avg Pre-Event Volatility Innovation")
                        plt.ylabel(f"{k}-day Future Return")
                        plt.title(f"H2.1 ({self.file_prefix.upper()}): Vol Innov. vs {k}d Future Return (Supported)")
                        plt.legend(); plt.grid(True, linestyle=':');
                        plt.savefig(os.path.join(self.results_dir, f"{self.file_prefix}_h2_1_plot_horizon{k}.png"), dpi=150)
                        plt.close()
        print(f"H2 Visualizations saved to {self.results_dir}")


def run_event_type_analysis_h2(event_type_name: str, event_file: str, event_date_col: str, ticker_col: str, results_dir: str, file_prefix: str, global_stock_df: pl.DataFrame, app_config: Config):
    print(f"\n>>>> Starting H2 analysis for {event_type_name.upper()} EVENTS <<<<")
    event_data_df = load_event_data(event_file, event_date_col, ticker_col)
    if event_data_df is None or event_data_df.is_empty():
        print(f"Could not load event data for {event_type_name} (H2). Skipping.")
        return False

    tester = Hypothesis2Tester(config=app_config, results_dir=results_dir, file_prefix=file_prefix)
    success = tester.run_analysis(global_stock_df, event_data_df)
    print(f">>>> H2 analysis for {event_type_name.upper()} EVENTS {'SUCCESSFUL' if success else 'FAILED'} <<<<")
    return success

def compare_h2_results():
    print("\n=== Comparing FDA and Earnings Results for Hypothesis 2 ===")
    comparison_dir = "results/hypothesis2/comparison/"
    os.makedirs(comparison_dir, exist_ok=True)

    hypotheses_sub_keys = ["H2_1", "H2_2", "H2_3"]
    overall_comparison_data = []

    for h_key in hypotheses_sub_keys:
        fda_file = os.path.join(FDA_RESULTS_DIR, f"{FDA_FILE_PREFIX}_{h_key.lower()}_results.csv")
        earn_file = os.path.join(EARNINGS_RESULTS_DIR, f"{EARNINGS_FILE_PREFIX}_{h_key.lower()}_results.csv")

        fda_supported_overall = False
        earn_supported_overall = False
        fda_details, earn_details = "N/A", "N/A"

        if os.path.exists(fda_file):
            fda_df = pl.read_csv(fda_file)
            if not fda_df.is_empty() and "supported" in fda_df.columns:
                fda_supported_overall = fda_df["supported"].any() if h_key == "H2_1" else fda_df["supported"][0]
                if h_key == "H2_1": fda_details = f"{fda_df.filter(pl.col('supported')==True).height}/{fda_df.height} horiz. supp."
                elif h_key == "H2_2" and "p_value" in fda_df.columns: fda_details = f"p={fda_df['p_value'][0]:.3f}"
                elif h_key == "H2_3" and "avg_gamma" in fda_df.columns: fda_details = f"gamma={fda_df['avg_gamma'][0]:.3f}"


        if os.path.exists(earn_file):
            earn_df = pl.read_csv(earn_file)
            if not earn_df.is_empty() and "supported" in earn_df.columns:
                earn_supported_overall = earn_df["supported"].any() if h_key == "H2_1" else earn_df["supported"][0]
                if h_key == "H2_1": earn_details = f"{earn_df.filter(pl.col('supported')==True).height}/{earn_df.height} horiz. supp."
                elif h_key == "H2_2" and "p_value" in earn_df.columns: earn_details = f"p={earn_df['p_value'][0]:.3f}"
                elif h_key == "H2_3" and "avg_gamma" in earn_df.columns: earn_details = f"gamma={earn_df['avg_gamma'][0]:.3f}"
        
        overall_comparison_data.append({
            "Hypothesis": h_key,
            "FDA_Supported": fda_supported_overall, "FDA_Details": fda_details,
            "Earnings_Supported": earn_supported_overall, "Earnings_Details": earn_details
        })

    summary_df = pl.DataFrame(overall_comparison_data)
    print("\n--- H2 Comparison Summary ---")
    print(summary_df)
    summary_df.write_csv(os.path.join(comparison_dir, "h2_comparison_summary.csv"))
    
    n_hypotheses = len(summary_df)
    index = np.arange(n_hypotheses)
    bar_width = 0.35
    fda_support_plot = summary_df["FDA_Supported"].cast(pl.Int8).to_list()
    earn_support_plot = summary_df["Earnings_Supported"].cast(pl.Int8).to_list()
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(index - bar_width/2, fda_support_plot, bar_width, label='FDA Supported', color='blue')
    ax.bar(index + bar_width/2, earn_support_plot, bar_width, label='Earnings Supported', color='orange')
    ax.set_xlabel('Sub-Hypothesis'); ax.set_ylabel('Supported (1=Yes, 0=No)')
    ax.set_title('Hypothesis 2 Support Comparison: FDA vs Earnings')
    ax.set_xticks(index); ax.set_xticklabels(summary_df["Hypothesis"].to_list())
    ax.set_yticks([0, 1]); ax.set_yticklabels(['No', 'Yes']); ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(comparison_dir, "h2_comparison_plot.png"), dpi=200)
    plt.close(fig)
    print(f"Saved H2 comparison plot to {comparison_dir}")


def main():
    app_config = Config()
    global_stock_df = load_stock_data(STOCK_FILES)
    if global_stock_df is None or global_stock_df.is_empty():
        print("Failed to load global stock data. Exiting.")
        return

    fda_success = run_event_type_analysis_h2(
        "FDA", FDA_EVENT_FILE, FDA_EVENT_DATE_COL, FDA_TICKER_COL,
        FDA_RESULTS_DIR, FDA_FILE_PREFIX, global_stock_df, app_config
    )
    earnings_success = run_event_type_analysis_h2(
        "Earnings", EARNINGS_EVENT_FILE, EARNINGS_EVENT_DATE_COL, EARNINGS_TICKER_COL,
        EARNINGS_RESULTS_DIR, EARNINGS_FILE_PREFIX, global_stock_df, app_config
    )

    if fda_success and earnings_success:
        compare_h2_results()
    else:
        print("One or both H2 analyses failed, skipping comparison.")

if __name__ == "__main__":
    main()