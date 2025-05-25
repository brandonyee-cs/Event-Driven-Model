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
if current_dir not in sys.path: sys.path.append(current_dir)

project_root = os.path.dirname(current_dir)
src_path = os.path.join(project_root, 'src')
if src_path not in sys.path:
    sys.path.append(src_path)

try:
    from models import UnifiedVolatilityModel # Using GJR-GARCH component
    # Data loading similar to testhyp1.py
    print("Successfully imported models from models.py.")
except ImportError as e:
    print(f"Error importing modules: {e}")
    print(f"Python Path: {sys.path}")
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

# Shared analysis parameters
ANALYSIS_WINDOW_DAYS = 30 # Days before and after event for analysis
GARCH_ESTIMATION_WINDOW_DAYS = 60 # Days before event for GARCH fitting
PREDICTION_HORIZONS = [1, 5, 10] # Days for return prediction
VOL_PERSISTENCE_WINDOW_DAYS = 10 # Days post-event to measure persistence

# Unified Volatility Model Parameters (example, adjust as needed)
VOL_PARAMS_DEFAULT = {
    'omega': 1e-6, 'alpha': 0.08, 'beta': 0.9, 'gamma': 0.04, # GJR-GARCH
    'k1': 1.3, 'k2': 1.5, 'delta': 5,
    'delta_t1': 5.0, 'delta_t2': 3.0, 'delta_t3': 10.0
}

class Hypothesis2Framework:
    def __init__(self, results_dir: str, file_prefix: str, vol_params_override: Optional[Dict] = None):
        self.results_dir = results_dir
        self.file_prefix = file_prefix
        os.makedirs(self.results_dir, exist_ok=True)

        vol_params_dict = VOL_PARAMS_DEFAULT.copy()
        if vol_params_override:
            vol_params_dict.update(vol_params_override)
        # We primarily need the GJR-GARCH component for H2
        self.vol_model_template = UnifiedVolatilityModel(**vol_params_dict) # Template to get params
        self.h2_results_summary = {}


    def _load_and_prepare_data(self, event_file_path: str, stock_file_paths: List[str],
                               event_date_col: str, ticker_col: str) -> Optional[pl.DataFrame]:
        # Same data loading as in Hypothesis1Framework
        try:
            print(f"Loading event data from: {event_file_path}")
            event_df = pl.read_csv(event_file_path, try_parse_dates=True)
            event_df = event_df.rename({event_date_col: "event_date", ticker_col: "ticker"})
            event_df = event_df.with_columns([
                pl.col("event_date").cast(pl.Date),
                pl.col("ticker").cast(pl.Utf8)
            ])
            event_df = event_df.drop_nulls(subset=["event_date", "ticker"]).unique(subset=["ticker", "event_date"])
            if event_df.is_empty(): return None
            print(f"Loaded {event_df.height} unique events.")

            stock_df_list = []
            for f_path in stock_file_paths:
                try:
                    df = pl.read_parquet(f_path)
                    rename_map = {}
                    if 'PERMNO' in df.columns and 'ticker' not in df.columns : rename_map['PERMNO'] = 'ticker' # Example
                    if 'date' not in df.columns and 'DATE' in df.columns: rename_map['DATE'] = 'date'
                    if 'PRC' not in df.columns and 'prc' in df.columns: rename_map['prc'] = 'PRC'
                    if 'RET' not in df.columns and 'ret' in df.columns: rename_map['ret'] = 'RET'
                    if rename_map: df = df.rename(rename_map)

                    required_cols = ['date', 'ticker', 'PRC', 'RET']
                    if not all(col in df.columns for col in required_cols): continue
                    df = df.select(required_cols).with_columns([
                        pl.col("date").cast(pl.Date), pl.col("ticker").cast(pl.Utf8),
                        pl.col("PRC").cast(pl.Float64), pl.col("RET").cast(pl.Float64)
                    ])
                    stock_df_list.append(df)
                except Exception as e: print(f"Error loading stock file {f_path}: {e}")
            if not stock_df_list: return None
            stock_df = pl.concat(stock_df_list).drop_nulls().unique(subset=["ticker", "date"])
            print(f"Loaded and combined {stock_df.height} stock records.")

            merged_df = event_df.join(stock_df, on="ticker", how="inner")
            merged_df = merged_df.with_columns(
                (pl.col("date") - pl.col("event_date")).dt.total_days().cast(pl.Int32).alias("days_to_event")
            )
            # Window needs to be large enough for GARCH estimation + analysis
            merged_df = merged_df.filter(
                (pl.col("days_to_event") >= -(GARCH_ESTIMATION_WINDOW_DAYS + ANALYSIS_WINDOW_DAYS)) &
                (pl.col("days_to_event") <= ANALYSIS_WINDOW_DAYS)
            )
            merged_df = merged_df.sort(["ticker", "event_date", "date"])
            merged_df = merged_df.with_columns(
                pl.col("ticker").cast(str) + "_" + pl.col("event_date").dt.strftime("%Y%m%d")
            ).rename({"literal": "event_id"})
            merged_df = merged_df.with_columns(
                pl.when(pl.col("RET").is_null() & pl.col("PRC").is_not_null() & pl.col("PRC").shift(1).over("event_id").is_not_null())
                .then( (pl.col("PRC") / pl.col("PRC").shift(1).over("event_id")) - 1)
                .otherwise(pl.col("RET"))
                .alias("RET")
            ).drop_nulls(subset=["RET"])
            if merged_df.is_empty(): return None
            print(f"Prepared data shape for H2: {merged_df.shape}")
            return merged_df
        except Exception as e:
            print(f"Error in _load_and_prepare_data (H2): {e}")
            traceback.print_exc()
            return None

    def _calculate_garch_vol_and_innovations(self, data: pl.DataFrame) -> pl.DataFrame:
        """
        Estimate GJR-GARCH for each event's pre-event period.
        Calculate baseline GJR-GARCH volatility (h_t) and volatility innovations.
        Volatility Innovation = Realized Vol (or Event-Adjusted Vol) - GARCH Predicted Vol
        """
        results_list = []
        for event_id_val, group_df_pl in data.group_by("event_id"):
            group_df = group_df_pl.to_pandas().sort_values("days_to_event").reset_index(drop=True)
            
            # GARCH estimation period: e.g., -60 to -1 days_to_event
            garch_estimation_data = group_df[
                (group_df['days_to_event'] < 0) &
                (group_df['days_to_event'] >= -GARCH_ESTIMATION_WINDOW_DAYS)
            ]['RET'].dropna().to_numpy()

            if len(garch_estimation_data) < 30: # Min obs for GARCH
                continue

            try:
                # Use the GJR-GARCH part of UnifiedVolatilityModel
                # Here, baseline_volatility method estimates h_t
                # This method in `models.py` needs to be callable with just returns.
                # Let's assume it is, or we use a direct GJR-GARCH fit here.
                # For simplicity with the current `models.py` structure,
                # we'll use `arch` package directly for GJR-GARCH as in new event_processor.

                from arch import arch_model # Local import for this part
                
                # Ensure returns are in percentage for arch model if it expects that
                model_fit = arch_model(garch_estimation_data * 100, vol='GARCH', p=1, o=1, q=1, dist='normal').fit(disp='off')
                
                # Get GJR-GARCH parameters for this event's pre-period
                event_gjr_params = {
                    'omega': model_fit.params['omega'],
                    'alpha': model_fit.params['alpha[1]'],
                    'beta': model_fit.params['beta[1]'],
                    'gamma': model_fit.params.get('gamma[1]', 0.0) # Handle if GARCH was fit
                }
                
                # Create a temporary UVM instance with these event-specific GJR params
                # but use the global event-adjustment params (k1,k2 etc.)
                temp_uvm_params = {**self.vol_model_template.__dict__, **event_gjr_params}
                temp_uvm = UnifiedVolatilityModel(**{k: v for k, v in temp_uvm_params.items() if k in UnifiedVolatilityModel.__init__.__code__.co_varnames})


                # Now, calculate h_t (baseline GJR-GARCH vol) for the *entire* event window
                # using the fitted parameters on the *full* return series for the event.
                full_event_returns = group_df['RET'].to_numpy()
                
                # The `baseline_volatility` method in `models.py` calculates sqrt(h_t)
                h_sqrt_series = temp_uvm.baseline_volatility(full_event_returns) # This should return sqrt of variance
                h_series = h_sqrt_series**2 # This is h_t (variance)

                # Calculate GARCH predicted volatility for t based on t-1
                # h_pred_t = omega + alpha*eps_{t-1}^2 + beta*h_{t-1} + gamma*I*eps_{t-1}^2
                h_pred_series = np.zeros_like(h_series)
                h_pred_series[0] = h_series[0] # Or some initial value
                for t_idx in range(1, len(full_event_returns)):
                    eps_prev_sq = full_event_returns[t_idx-1]**2 # Simplified eps^2 (actual ret^2)
                    indicator_prev = 1 if full_event_returns[t_idx-1] < 0 else 0
                    h_pred_series[t_idx] = (
                        event_gjr_params['omega'] +
                        event_gjr_params['alpha'] * eps_prev_sq +
                        event_gjr_params['beta'] * h_series[t_idx-1] + # Use actual h_{t-1}
                        event_gjr_params['gamma'] * indicator_prev * eps_prev_sq
                    )
                
                # Volatility Innovation: Realized (actual h_t) - Predicted (GARCH h_t based on t-1)
                # Using h_series as "realized" conditional variance
                vol_innovations = h_series - h_pred_series
                
                group_df['garch_h_t'] = h_series
                group_df['garch_vol_innov'] = vol_innovations
                group_df['gjr_gamma'] = event_gjr_params['gamma'] # Store gamma for H2.3

                results_list.append(pl.from_pandas(group_df))

            except Exception as e:
                # print(f"Error in GARCH for event {event_id_val}: {e}") # Can be noisy
                continue
        
        if not results_list: return pl.DataFrame()
        return pl.concat(results_list)

    def _test_h2_1_innovations_predict_returns(self, data: pl.DataFrame):
        """
        Test H2.1: Pre-event volatility innovations predict subsequent returns.
        Regression: FutureReturn_k = a + b * AvgPreEventVolInnovation + e
        """
        print("\n--- Testing H2.1: Volatility Innovations Predict Returns ---")
        # Filter for pre-event period to get innovations
        # e.g., days_to_event from -10 to -1 for "pre-event innovations"
        pre_event_innov_window = (-10, -1)
        
        # Aggregate average pre-event innovation per event_id
        avg_pre_innov = data.filter(
            (pl.col('days_to_event') >= pre_event_innov_window[0]) &
            (pl.col('days_to_event') <= pre_event_innov_window[1])
        ).group_by("event_id").agg(
            pl.mean("garch_vol_innov").alias("avg_pre_event_vol_innov")
        )

        h2_1_results = []
        for k in PREDICTION_HORIZONS:
            # Get future returns starting from event day (days_to_event = 0)
            # Shift returns by -k to get future returns
            data_with_future_ret = data.with_columns(
                pl.col("RET").shift(-k).over("event_id").alias(f"future_ret_{k}d")
            )
            
            # Join future returns at event day with pre-event innovations
            regression_df = data_with_future_ret.filter(pl.col("days_to_event") == 0).join(
                avg_pre_innov, on="event_id", how="inner"
            ).select(["event_id", f"future_ret_{k}d", "avg_pre_event_vol_innov"]).drop_nulls()

            if regression_df.height > 10: # Min samples for regression
                X = regression_df["avg_pre_event_vol_innov"].to_numpy().reshape(-1, 1)
                y = regression_df[f"future_ret_{k}d"].to_numpy()
                
                # Check for near-zero variance in X
                if np.var(X) < 1e-10:
                    print(f"  H2.1 (Future Ret {k}d): Insufficient variance in predictor. Slope=0, p=1.")
                    slope, intercept, r_value, p_value, std_err = 0, np.mean(y), 0, 1.0, 0
                else:
                    slope, intercept, r_value, p_value, std_err = stats.linregress(X.flatten(), y)
                
                supported = p_value < 0.05 and slope > 0 # Assuming positive relation
                h2_1_results.append({
                    "horizon": k, "slope": slope, "r_squared": r_value**2,
                    "p_value": p_value, "supported": supported, "n_obs": len(regression_df)
                })
                print(f"  H2.1 (Future Ret {k}d): Slope={slope:.4f}, R2={r_value**2:.3f}, p={p_value:.3f}, Supported={supported}, N={len(regression_df)}")
            else:
                print(f"  H2.1 (Future Ret {k}d): Insufficient data (N={len(regression_df)})")
                h2_1_results.append({
                    "horizon": k, "slope": None, "r_squared": None,
                    "p_value": None, "supported": False, "n_obs": len(regression_df)
                })
        
        self.h2_results_summary['H2_1'] = pl.DataFrame(h2_1_results)
        self.h2_results_summary['H2_1'].write_csv(os.path.join(self.results_dir, f"{self.file_prefix}_h2_1_results.csv"))

    def _test_h2_2_vol_persistence_extends_returns(self, data: pl.DataFrame):
        """
        Test H2.2: Post-event volatility persistence extends elevated expected returns.
        Regression: AvgPostEventReturn = a + b * VolPersistenceRatio + e
        VolPersistenceRatio = AvgPostEventVol / AvgPreEventVol
        """
        print("\n--- Testing H2.2: Volatility Persistence Extends Returns ---")
        pre_event_vol_window = (-GARCH_ESTIMATION_WINDOW_DAYS, -1) # Match GARCH est. window
        post_event_vol_window = (1, VOL_PERSISTENCE_WINDOW_DAYS)
        post_event_ret_window = (1, VOL_PERSISTENCE_WINDOW_DAYS)

        # Calculate avg pre-event and post-event GARCH h_t (variance)
        event_vol_metrics = data.group_by("event_id").agg(
            pl.mean("garch_h_t").filter(
                (pl.col("days_to_event") >= pre_event_vol_window[0]) &
                (pl.col("days_to_event") <= pre_event_vol_window[1])
            ).alias("avg_pre_event_h_t"),
            pl.mean("garch_h_t").filter(
                (pl.col("days_to_event") >= post_event_vol_window[0]) &
                (pl.col("days_to_event") <= post_event_vol_window[1])
            ).alias("avg_post_event_h_t"),
             pl.mean("RET").filter( # Use actual returns for "elevated expected returns"
                (pl.col("days_to_event") >= post_event_ret_window[0]) &
                (pl.col("days_to_event") <= post_event_ret_window[1])
            ).alias("avg_post_event_return")
        ).drop_nulls()

        if event_vol_metrics.is_empty():
             print("  H2.2: No events with valid pre/post volatility metrics.")
             self.h2_results_summary['H2_2'] = pl.DataFrame({"supported": [False], "n_obs": [0]})
             return

        event_vol_metrics = event_vol_metrics.with_columns(
            (pl.col("avg_post_event_h_t") / pl.col("avg_pre_event_h_t").clip_min(1e-8) # Avoid div by zero
            ).alias("vol_persistence_ratio")
        ).drop_nulls()
        
        if event_vol_metrics.height > 10:
            X = event_vol_metrics["vol_persistence_ratio"].to_numpy().reshape(-1, 1)
            y = event_vol_metrics["avg_post_event_return"].to_numpy()

            if np.var(X) < 1e-10:
                print(f"  H2.2: Insufficient variance in predictor. Slope=0, p=1.")
                slope, intercept, r_value, p_value, std_err = 0, np.mean(y), 0, 1.0, 0
            else:
                slope, intercept, r_value, p_value, std_err = stats.linregress(X.flatten(), y)
            
            supported = p_value < 0.05 and slope > 0 # Assuming positive relation
            result = {
                "slope": slope, "r_squared": r_value**2, "p_value": p_value,
                "supported": supported, "n_obs": len(event_vol_metrics)
            }
            print(f"  H2.2: Slope={slope:.4f}, R2={r_value**2:.3f}, p={p_value:.3f}, Supported={supported}, N={len(event_vol_metrics)}")
        else:
            print(f"  H2.2: Insufficient data (N={len(event_vol_metrics)})")
            result = {"slope": None, "r_squared": None, "p_value": None, "supported": False, "n_obs": len(event_vol_metrics)}

        self.h2_results_summary['H2_2'] = pl.DataFrame([result])
        self.h2_results_summary['H2_2'].write_csv(os.path.join(self.results_dir, f"{self.file_prefix}_h2_2_results.csv"))


    def _test_h2_3_asymmetric_vol_response(self, data: pl.DataFrame):
        """
        Test H2.3: Asymmetric volatility response (GJR-GARCH gamma) correlates with asymmetric price adjustment.
        Compare returns on days with high gamma vs low gamma, conditional on negative news.
        Or, simpler: check if avg gamma is significantly positive.
        """
        print("\n--- Testing H2.3: Asymmetric Volatility Response (GJR-GARCH Gamma) ---")
        if 'gjr_gamma' not in data.columns:
            print("  H2.3: 'gjr_gamma' column not found. Skipping test.")
            self.h2_results_summary['H2_3'] = pl.DataFrame({"supported": [False], "avg_gamma": [None], "p_value_gamma_gt_0": [None]})
            return

        # Get unique gamma per event_id (as it's estimated once per event's pre-period)
        event_gammas = data.select(["event_id", "gjr_gamma"]).unique(subset="event_id")["gjr_gamma"].drop_nulls().to_numpy()

        if len(event_gammas) > 1:
            avg_gamma = np.mean(event_gammas)
            # Test if gamma is significantly greater than 0
            t_stat, p_value = stats.ttest_1samp(event_gammas, 0, alternative='greater')
            supported = p_value < 0.05 and avg_gamma > 0
            print(f"  H2.3: Average GJR-GARCH Gamma = {avg_gamma:.4f}")
            print(f"  T-test (gamma > 0): t-stat={t_stat:.3f}, p-value={p_value:.3f}, Supported={supported}, N_events={len(event_gammas)}")
            result = {"avg_gamma": avg_gamma, "p_value_gamma_gt_0": p_value, "supported": supported, "n_events": len(event_gammas)}
        else:
            print("  H2.3: Insufficient events with GJR-GARCH gamma values.")
            result = {"avg_gamma": None, "p_value_gamma_gt_0": None, "supported": False, "n_events": len(event_gammas)}

        self.h2_results_summary['H2_3'] = pl.DataFrame([result])
        self.h2_results_summary['H2_3'].write_csv(os.path.join(self.results_dir, f"{self.file_prefix}_h2_3_results.csv"))

    def _generate_h2_visualizations(self, data: pl.DataFrame):
        # Placeholder for H2 visualizations (e.g., scatter plot for H2.1)
        # Plot for H2.1 if any horizon was significant
        if 'H2_1' in self.h2_results_summary and self.h2_results_summary['H2_1'] is not None:
            h2_1_res = self.h2_results_summary['H2_1']
            for row_dict in h2_1_res.filter(pl.col("supported") == True).to_dicts():
                k = row_dict['horizon']
                slope = row_dict['slope']
                intercept = slope * data["avg_pre_event_vol_innov"].mean() if slope is not None and not data.filter(pl.col("avg_pre_event_vol_innov").is_not_null()).is_empty() else 0


                # Re-create data for plotting this specific significant result
                pre_event_innov_window = (-10, -1)
                avg_pre_innov = data.filter(
                    (pl.col('days_to_event') >= pre_event_innov_window[0]) &
                    (pl.col('days_to_event') <= pre_event_innov_window[1])
                ).group_by("event_id").agg(
                    pl.mean("garch_vol_innov").alias("avg_pre_event_vol_innov")
                )
                data_with_future_ret = data.with_columns(
                    pl.col("RET").shift(-k).over("event_id").alias(f"future_ret_{k}d")
                )
                plot_df = data_with_future_ret.filter(pl.col("days_to_event") == 0).join(
                    avg_pre_innov, on="event_id", how="inner"
                ).select(["event_id", f"future_ret_{k}d", "avg_pre_event_vol_innov"]).drop_nulls().to_pandas()

                if not plot_df.empty and slope is not None:
                    plt.figure(figsize=(8,6))
                    plt.scatter(plot_df["avg_pre_event_vol_innov"], plot_df[f"future_ret_{k}d"], alpha=0.5, label="Data points")
                    x_vals = np.array(plt.xlim())
                    y_vals = intercept + slope * x_vals
                    plt.plot(x_vals, y_vals, color='red', label=f"Fit: y={slope:.3f}x+{intercept:.3f}\np={row_dict['p_value']:.3f}")
                    plt.xlabel("Avg Pre-Event Volatility Innovation")
                    plt.ylabel(f"{k}-day Future Return")
                    plt.title(f"H2.1: {self.file_prefix.upper()} - Vol Innovation vs {k}d Future Return")
                    plt.legend()
                    plt.grid(True, linestyle=':')
                    plt.savefig(os.path.join(self.results_dir, f"{self.file_prefix}_h2_1_plot_horizon{k}.png"), dpi=150)
                    plt.close()


    def run_full_analysis(self, event_file_path: str, stock_file_paths: List[str],
                          event_date_col: str, ticker_col: str):
        print(f"\n--- Starting Hypothesis 2 Analysis for {self.file_prefix.upper()} ---")
        data = self._load_and_prepare_data(event_file_path, stock_file_paths, event_date_col, ticker_col)
        if data is None or data.is_empty():
            print(f"Failed to load or prepare data for {self.file_prefix}. Aborting H2 analysis.")
            return False

        print("Calculating GARCH volatility and innovations...")
        data_with_garch = self._calculate_garch_vol_and_innovations(data)
        if data_with_garch.is_empty():
            print(f"Failed to calculate GARCH metrics for {self.file_prefix}. Aborting H2 analysis.")
            return False
        
        data_with_garch.write_csv(os.path.join(self.results_dir, f"{self.file_prefix}_h2_data_with_garch.csv"))

        self._test_h2_1_innovations_predict_returns(data_with_garch)
        self._test_h2_2_vol_persistence_extends_returns(data_with_garch)
        self._test_h2_3_asymmetric_vol_response(data_with_garch)
        
        self._generate_h2_visualizations(data_with_garch) # Pass data with necessary columns

        print(f"--- Hypothesis 2 Analysis for {self.file_prefix.upper()} Complete ---")
        return True


def run_specific_analysis(event_type: str):
    if event_type == "FDA":
        analyzer = Hypothesis2Framework(results_dir=FDA_RESULTS_DIR, file_prefix=FDA_FILE_PREFIX)
        success = analyzer.run_full_analysis(
            event_file_path=FDA_EVENT_FILE, stock_file_paths=STOCK_FILES,
            event_date_col=FDA_EVENT_DATE_COL, ticker_col=FDA_TICKER_COL
        )
    elif event_type == "Earnings":
        analyzer = Hypothesis2Framework(results_dir=EARNINGS_RESULTS_DIR, file_prefix=EARNINGS_FILE_PREFIX)
        success = analyzer.run_full_analysis(
            event_file_path=EARNINGS_EVENT_FILE, stock_file_paths=STOCK_FILES,
            event_date_col=EARNINGS_EVENT_DATE_COL, ticker_col=EARNINGS_TICKER_COL
        )
    else:
        print(f"Unknown event type: {event_type}")
        return False
    return success

def compare_results():
    print("\n=== Comparing FDA and Earnings Results for Hypothesis 2 ===")
    comparison_dir = "results/hypothesis2/comparison/"
    os.makedirs(comparison_dir, exist_ok=True)

    results_files = {
        "H2_1": ("h2_1_results.csv", ["horizon"]),
        "H2_2": ("h2_2_results.csv", []),
        "H2_3": ("h2_3_results.csv", [])
    }
    
    overall_comparison_data = []

    for h_key, (fname, id_vars) in results_files.items():
        fda_file = os.path.join(FDA_RESULTS_DIR, f"{FDA_FILE_PREFIX}_{fname}")
        earn_file = os.path.join(EARNINGS_RESULTS_DIR, f"{EARNINGS_FILE_PREFIX}_{fname}")

        if not os.path.exists(fda_file) or not os.path.exists(earn_file):
            print(f"Skipping comparison for {h_key}: one or both files missing.")
            continue

        fda_df = pl.read_csv(fda_file)
        earn_df = pl.read_csv(earn_file)

        # For H2.1, we might have multiple rows (horizons). For others, usually one.
        # For simplicity, we'll report if *any* test for H2.1 was supported.
        if h_key == "H2_1":
            fda_supported_any = fda_df['supported'].any()
            earn_supported_any = earn_df['supported'].any()
            fda_details = f"{fda_df.filter(pl.col('supported')==True).height}/{fda_df.height} horizons supported"
            earn_details = f"{earn_df.filter(pl.col('supported')==True).height}/{earn_df.height} horizons supported"
        else: # H2.2, H2.3 should have one row
            fda_supported_any = fda_df['supported'][0] if not fda_df.is_empty() else False
            earn_supported_any = earn_df['supported'][0] if not earn_df.is_empty() else False
            fda_details = f"p={fda_df['p_value'][0]:.3f}" if 'p_value' in fda_df.columns and not fda_df.is_empty() else f"gamma={fda_df['avg_gamma'][0]:.3f}" if 'avg_gamma' in fda_df.columns and not fda_df.is_empty() else ""
            earn_details = f"p={earn_df['p_value'][0]:.3f}" if 'p_value' in earn_df.columns and not earn_df.is_empty() else f"gamma={earn_df['avg_gamma'][0]:.3f}" if 'avg_gamma' in earn_df.columns and not earn_df.is_empty() else ""
        
        overall_comparison_data.append({
            "Hypothesis": h_key,
            "FDA_Supported": fda_supported_any,
            "FDA_Details": fda_details,
            "Earnings_Supported": earn_supported_any,
            "Earnings_Details": earn_details
        })

    if not overall_comparison_data:
        print("No results to compare.")
        return

    summary_df = pl.DataFrame(overall_comparison_data)
    print("\n--- H2 Comparison Summary ---")
    print(summary_df)
    summary_df.write_csv(os.path.join(comparison_dir, "h2_comparison_summary.csv"))

    # Simple plot for overall support
    n_hypotheses = len(summary_df)
    index = np.arange(n_hypotheses)
    bar_width = 0.35

    fda_support_plot = summary_df["FDA_Supported"].cast(pl.Int8).to_list()
    earn_support_plot = summary_df["Earnings_Supported"].cast(pl.Int8).to_list()

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(index - bar_width/2, fda_support_plot, bar_width, label='FDA Supported', color='blue')
    ax.bar(index + bar_width/2, earn_support_plot, bar_width, label='Earnings Supported', color='orange')

    ax.set_xlabel('Sub-Hypothesis')
    ax.set_ylabel('Supported (1=Yes, 0=No)')
    ax.set_title('Hypothesis 2 Support Comparison: FDA vs Earnings')
    ax.set_xticks(index)
    ax.set_xticklabels(summary_df["Hypothesis"].to_list())
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['No', 'Yes'])
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(comparison_dir, "h2_comparison_plot.png"), dpi=200)
    plt.close(fig)
    print(f"Saved H2 comparison plot to {comparison_dir}")


def main():
    fda_success = run_specific_analysis("FDA")
    earnings_success = run_specific_analysis("Earnings")

    if fda_success and earnings_success:
        compare_results()
    else:
        print("One or both H2 analyses failed, skipping comparison.")

if __name__ == "__main__":
    main()