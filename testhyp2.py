# testhyp2.py
# Aligned with the paper: "Modeling Equilibrium Asset Pricing Around Events..."

import pandas as pd
import numpy as np
import os
import sys
import traceback
import polars as pl
from typing import List, Tuple, Dict, Any
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

# --- Configuration ---
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path: sys.path.append(current_dir)
try:
    from src.event_processor import EventDataLoader, EventFeatureEngineer, EventAnalysis
    from src.models import GARCHModel, GJRGARCHModel # ThreePhaseVolatilityModel not directly used by H2 tests
    print("Successfully imported Event processor classes and models.")
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Ensure 'event_processor.py' and 'models.py' are in the same directory or Python path.")
    sys.exit(1)

pl.Config.set_engine_affinity(engine="streaming")

# --- Hardcoded Analysis Parameters (aligned with paper) ---
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
WINDOW_DAYS = 60 # General window for data loading
ANALYSIS_WINDOW = (-30, 30) # Analysis window for features around event
PRE_EVENT_WINDOW_H2_1 = (-15, -1)  # Window for pre-event volatility innovations (H2.1)
POST_EVENT_WINDOW_H2_2 = (1, 15)   # Window for post-event volatility persistence & returns (H2.2)

# GARCH model parameters (initial guesses for fitting)
# Paper uses GJR-GARCH for h_t, so it's the primary model. GARCH can be for comparison.
MODEL_TYPES = ['gjr', 'garch'] # Test GJR (primary) and GARCH
GARCH_PARAMS = {
    'garch': {'omega': 0.00001, 'alpha': 0.05, 'beta': 0.90}, # Standard GARCH(1,1)
    'gjr': {'omega': 0.00001, 'alpha': 0.03, 'beta': 0.90, 'gamma': 0.04} # GJR-GARCH(1,1)
}
PREDICTION_WINDOWS = [1, 3, 5]  # Days ahead to predict returns for H2.1

class Hypothesis2Analyzer:
    """
    Analyzer for testing Hypothesis 2:
    GARCH-estimated conditional volatility innovations serve as an effective proxy for impact uncertainty.
    Sub-hypotheses:
    H2.1: Pre-event volatility innovations predict subsequent returns.
    H2.2: Post-event volatility persistence extends the period of elevated expected returns.
    H2.3: Asymmetric volatility response (GJR-GARCH) correlates with asymmetric price adjustment.
    """
    def __init__(self, analyzer: EventAnalysis, results_dir: str, file_prefix: str):
        self.analyzer = analyzer
        self.results_dir = results_dir
        self.file_prefix = file_prefix
        self.return_col = 'ret'
        self.analysis_window = ANALYSIS_WINDOW # General features window
        self.pre_event_window_h2_1 = PRE_EVENT_WINDOW_H2_1
        self.post_event_window_h2_2 = POST_EVENT_WINDOW_H2_2
        self.prediction_windows = PREDICTION_WINDOWS
        self.model_types = MODEL_TYPES
        self.garch_params = GARCH_PARAMS

        self.event_garch_fits = {} # Store fitted models and innovations per event
        self.prediction_results_h2_1 = {}
        self.persistence_results_h2_2 = {}
        self.asymmetry_results_h2_3 = {}

        os.makedirs(results_dir, exist_ok=True)

    def _fit_garch_for_events(self):
        """Fit GARCH models to each event's return series."""
        print("\n--- Fitting GARCH/GJR-GARCH models for each event ---")
        if self.analyzer.data is None:
            print("Error: No data loaded for GARCH fitting.")
            return False

        # Extend window slightly for GARCH estimation stability if needed, but paper uses h_t
        # The main analysis window is fine for extracting returns for GARCH.
        garch_estimation_window = (min(self.analysis_window[0], -60), max(self.analysis_window[1], 60)) # Window for GARCH estimation
        
        # Ensure data is sorted for consistent processing
        analysis_data = self.analyzer.data.filter(
            (pl.col('days_to_event') >= garch_estimation_window[0]) &
            (pl.col('days_to_event') <= garch_estimation_window[1])
        ).sort(['event_id', 'days_to_event'])

        if analysis_data.is_empty():
            print(f"Error: No data found within GARCH estimation window {garch_estimation_window}")
            return False

        event_ids = analysis_data.select('event_id').unique().to_series().to_list()
        print(f"Processing {len(event_ids)} events for GARCH fitting...")

        # Limit samples for dev/testing speed if necessary
        # sample_event_ids = np.random.choice(event_ids, size=min(100, len(event_ids)), replace=False) if len(event_ids) > 100 else event_ids
        sample_event_ids = event_ids # Process all for production

        processed_count = 0
        for event_id in sample_event_ids:
            event_data_series = analysis_data.filter(pl.col('event_id') == event_id)
            event_returns = event_data_series.select(self.return_col).to_series()
            
            if len(event_returns.drop_nulls()) < 30: # Min data for GARCH
                continue

            self.event_garch_fits[event_id] = {
                'ticker': event_data_series.select('ticker').head(1).item(),
                'event_date_polars': event_data_series.select('Event Date').head(1).item(), # Keep Polars datetime
                'days_to_event': event_data_series.select('days_to_event').to_series().to_list(),
                'returns': event_returns.to_numpy(), # Full returns series for this event
                'fitted_models': {},
                'innovations': {}, # h_t - E[h_t]
                'cond_volatility': {}, # sqrt(h_t)
                'persistence_metric': {} # For H2.2
            }

            for model_type in self.model_types:
                try:
                    params = self.garch_params[model_type]
                    if model_type == 'gjr':
                        model = GJRGARCHModel(**params)
                    else:
                        model = GARCHModel(**params)
                    
                    model.fit(event_returns.drop_nulls()) # Fit on non-null returns

                    self.event_garch_fits[event_id]['fitted_models'][model_type] = model
                    self.event_garch_fits[event_id]['innovations'][model_type] = model.volatility_innovations()
                    self.event_garch_fits[event_id]['cond_volatility'][model_type] = model.conditional_volatility()
                    
                    # Calculate persistence for H2.2 (e.g., beta or alpha+beta for GARCH, or post/pre vol ratio)
                    # Paper suggests "post-event volatility persistence extends elevated returns"
                    # Let's use ratio of post-event avg volatility to pre-event avg volatility
                    days_np = np.array(self.event_garch_fits[event_id]['days_to_event'])
                    vol_series = self.event_garch_fits[event_id]['cond_volatility'][model_type]
                    
                    # Align vol_series with days_np (GARCH output might be shorter due to drop_nulls or lags)
                    # This alignment is tricky if GARCH model.fit() doesn't return original indexing.
                    # For simplicity, assume GARCH output aligns with the start of the non-null return series.
                    # A more robust way would be to ensure GARCH models preserve original indexing or return aligned series.
                    # Current GARCHModel class implies variance_history aligns with input `returns_centered`.
                    
                    # We need to map innovations and cond_volatility back to original days_to_event indices.
                    # The GARCH models are fit on `event_returns.drop_nulls()`.
                    # The `volatility_innovations` has length T-1 of fitted data.
                    # `conditional_volatility` has length T of fitted data.
                    
                    # This part needs careful handling of indices if GARCH output isn't directly aligned
                    # with the full 'days_to_event' series that includes NaNs.
                    # For now, we'll assume the GARCH outputs correspond to the returns it was fit on.
                    # The selection of pre/post windows will be on the `days_to_event` that correspond to these outputs.
                    
                    # Placeholder for persistence metric for H2.2
                    # This will be calculated more carefully in _test_h2_2_volatility_persistence
                    
                except Exception as e:
                    # print(f" Error fitting {model_type} for {event_id}: {e}")
                    self.event_garch_fits[event_id]['fitted_models'][model_type] = None
            processed_count +=1
            if processed_count % 50 == 0: print(f"  Fitted GARCH for {processed_count}/{len(sample_event_ids)} events...")
        
        print(f"Finished GARCH fitting. Successful fits for {len(self.event_garch_fits)} events.")
        return bool(self.event_garch_fits)

    def _test_h2_1_innovations_predict_returns(self):
        print("\n--- H2.1: Testing Pre-event Volatility Innovations' Predictive Power ---")
        if not self.event_garch_fits:
            print("Error: GARCH models not fitted. Cannot test H2.1.")
            return

        results_h2_1 = {}
        for model_type in self.model_types:
            results_h2_1[model_type] = {}
            for pred_window in self.prediction_windows:
                X_innovations, y_future_returns = [], []
                
                for event_id, data in self.event_garch_fits.items():
                    if data['fitted_models'].get(model_type) is None: continue

                    # Innovations: h_t - E[h_t]. Length is T-1 of fitted GARCH returns.
                    # GARCH models are fit on non-NaN returns.
                    # We need to align these innovations and original days_to_event.
                    
                    # Get the part of the original event data that has non-null returns
                    event_full_data = self.analyzer.data.filter(pl.col('event_id') == event_id).sort('days_to_event')
                    event_nn_returns_data = event_full_data.filter(pl.col(self.return_col).is_not_null())
                    
                    # GARCH innovations and cond_volatility align with event_nn_returns_data
                    innovations_series = data['innovations'].get(model_type) # length T_nn - 1
                    
                    # We need innovations for days in PRE_EVENT_WINDOW_H2_1
                    # And future returns starting from day 0
                    
                    # 1. Get average pre-event innovation
                    # Innovations are for t=1 to T_nn. So innovations_series[k] corresponds to info at day k, predicting for day k+1 of nn_returns
                    # days_for_innov_calc = event_nn_returns_data['days_to_event'].to_list() # These are the days GARCH was fit on.
                    
                    # Let's use days_to_event from the fitted data (event_nn_returns_data) for indexing innovations
                    nn_days_to_event = event_nn_returns_data.select('days_to_event').to_series()

                    # Pre-event innovations: innovations for days in PRE_EVENT_WINDOW_H2_1
                    # Note: innovations_series has length T_nn-1. It corresponds to nn_days_to_event[1:].
                    # So, innovations_series[i] is the innovation for nn_days_to_event[i+1]
                    
                    avg_pre_event_innovation = np.nan
                    if innovations_series is not None and len(innovations_series) > 0:
                        pre_event_innovs_for_avg = []
                        # Iterate through the non-null days for which we have innovations
                        for i in range(len(innovations_series)):
                            day_val = nn_days_to_event[i+1] # Day corresponding to innovations_series[i]
                            if self.pre_event_window_h2_1[0] <= day_val <= self.pre_event_window_h2_1[1]:
                                pre_event_innovs_for_avg.append(innovations_series[i])
                        if pre_event_innovs_for_avg:
                            avg_pre_event_innovation = np.mean(pre_event_innovs_for_avg)

                    if np.isnan(avg_pre_event_innovation): continue
                    
                    # 2. Get future return (from day 0 over `pred_window` days)
                    # Use the original full event data for future returns
                    day_0_data = event_full_data.filter(pl.col('days_to_event') == 0)
                    if day_0_data.is_empty(): continue
                    
                    day_0_index_in_full = event_full_data.with_row_count().filter(pl.col('days_to_event') == 0)['row_nr'].item()
                    
                    actual_future_return = np.nan
                    if day_0_index_in_full + pred_window < event_full_data.height:
                        # Sum of log returns or compound return
                        # Using simple sum of arithmetic returns for now, as in H1
                        future_period_returns = event_full_data.slice(day_0_index_in_full + 1, pred_window).select(self.return_col).to_series().drop_nulls()
                        if len(future_period_returns) == pred_window : # Ensure full window of returns
                           actual_future_return = future_period_returns.sum() # Or (1+future_period_returns).product() - 1

                    if np.isnan(actual_future_return): continue

                    X_innovations.append(avg_pre_event_innovation)
                    y_future_returns.append(actual_future_return)

                if len(X_innovations) > 10:
                    X_arr = np.array(X_innovations)
                    y_arr = np.array(y_future_returns)
                    
                    if np.var(X_arr) < 1e-10: # Check for effectively zero variance
                        print(f"  Warning: Insufficient variance in X_innovations for {model_type}, pred_window {pred_window}.")
                        slope, intercept, r_value, p_value, std_err = 0, np.mean(y_arr), 0, 1.0, 0
                    else:
                        slope, intercept, r_value, p_value, std_err = stats.linregress(X_arr, y_arr)
                    
                    results_h2_1[model_type][pred_window] = {
                        'slope': slope, 'intercept': intercept, 'r_squared': r_value**2,
                        'p_value': p_value, 'std_err': std_err, 'n_samples': len(X_arr)
                    }
                    print(f"  {model_type}, {pred_window}-day ret: slope={slope:.4f}, R²={r_value**2:.4f}, p={p_value:.3f}, N={len(X_arr)}")
                else:
                    results_h2_1[model_type][pred_window] = None
                    print(f"  {model_type}, {pred_window}-day ret: Insufficient data (N={len(X_innovations)})")

        self.prediction_results_h2_1 = results_h2_1
        self._save_and_plot_h2_1_results()


    def _save_and_plot_h2_1_results(self):
        summary_rows = []
        supported_count = 0
        for model_type, preds in self.prediction_results_h2_1.items():
            for window, res in preds.items():
                if res:
                    is_supported = res['p_value'] < 0.1 # Arbitrary significance for "supported"
                    if is_supported: supported_count +=1
                    summary_rows.append({
                        'model_type': model_type, 'prediction_window': window,
                        'slope': res['slope'], 'r_squared': res['r_squared'],
                        'p_value': res['p_value'], 'n_samples': res['n_samples'],
                        'supported_by_p_0.1': is_supported
                    })
        if summary_rows:
            summary_df = pl.DataFrame(summary_rows)
            summary_df.write_csv(os.path.join(self.results_dir, f"{self.file_prefix}_H2_1_prediction_summary.csv"))
            print(f"\nH2.1 Summary: {supported_count}/{len(summary_rows)} tests supported (p<0.1).")
            # Plotting logic can be added here if needed, similar to original testhyp2.py
        else:
            print("\nNo results for H2.1 to save/plot.")


    def _test_h2_2_volatility_persistence(self):
        print("\n--- H2.2: Testing Post-event Volatility Persistence and Expected Returns ---")
        if not self.event_garch_fits:
            print("Error: GARCH models not fitted. Cannot test H2.2.")
            return

        results_h2_2 = {}
        for model_type in self.model_types:
            persistence_metrics, post_event_avg_returns = [], []
            
            for event_id, data in self.event_garch_fits.items():
                if data['fitted_models'].get(model_type) is None: continue

                event_full_data = self.analyzer.data.filter(pl.col('event_id') == event_id).sort('days_to_event')
                event_nn_returns_data = event_full_data.filter(pl.col(self.return_col).is_not_null())
                nn_days_to_event = event_nn_returns_data.select('days_to_event').to_series()
                
                cond_vol_series = data['cond_volatility'].get(model_type) # Aligns with nn_days_to_event

                if cond_vol_series is None or len(cond_vol_series) == 0: continue

                # Persistence: ratio of post-event avg cond. volatility to pre-event avg cond. volatility
                pre_vols, post_vols = [], []
                for i in range(len(cond_vol_series)):
                    day_val = nn_days_to_event[i]
                    if self.pre_event_window_h2_1[0] <= day_val <= self.pre_event_window_h2_1[1]: # Use H2.1 pre-window for consistency
                        pre_vols.append(cond_vol_series[i])
                    if self.post_event_window_h2_2[0] <= day_val <= self.post_event_window_h2_2[1]:
                        post_vols.append(cond_vol_series[i])
                
                if not pre_vols or not post_vols: continue
                avg_pre_vol = np.mean(pre_vols)
                avg_post_vol = np.mean(post_vols)

                if avg_pre_vol < 1e-9: continue # Avoid division by zero
                persistence_metric = avg_post_vol / avg_pre_vol
                
                # Average actual return in the post-event window
                post_event_actual_returns = event_full_data.filter(
                    (pl.col('days_to_event') >= self.post_event_window_h2_2[0]) &
                    (pl.col('days_to_event') <= self.post_event_window_h2_2[1])
                ).select(self.return_col).to_series().drop_nulls()

                if post_event_actual_returns.is_empty(): continue
                avg_post_event_return = post_event_actual_returns.mean()

                persistence_metrics.append(persistence_metric)
                post_event_avg_returns.append(avg_post_event_return)

            if len(persistence_metrics) > 10:
                X_arr = np.array(persistence_metrics)
                y_arr = np.array(post_event_avg_returns)
                slope, intercept, r_value, p_value, std_err = stats.linregress(X_arr, y_arr)
                results_h2_2[model_type] = {
                    'slope': slope, 'intercept': intercept, 'r_squared': r_value**2,
                    'p_value': p_value, 'std_err': std_err, 'n_samples': len(X_arr)
                }
                print(f"  {model_type}: slope={slope:.4f}, R²={r_value**2:.4f}, p={p_value:.3f}, N={len(X_arr)}")
            else:
                results_h2_2[model_type] = None
                print(f"  {model_type}: Insufficient data (N={len(persistence_metrics)})")

        self.persistence_results_h2_2 = results_h2_2
        self._save_and_plot_h2_2_results()

    def _save_and_plot_h2_2_results(self):
        summary_rows = []
        supported_count = 0
        for model_type, res in self.persistence_results_h2_2.items():
            if res:
                is_supported = res['p_value'] < 0.1 and res['slope'] > 0 # Positive relationship
                if is_supported: supported_count +=1
                summary_rows.append({
                    'model_type': model_type,
                    'slope': res['slope'], 'r_squared': res['r_squared'],
                    'p_value': res['p_value'], 'n_samples': res['n_samples'],
                    'supported_by_p_0.1_and_positive_slope': is_supported
                })
        if summary_rows:
            summary_df = pl.DataFrame(summary_rows)
            summary_df.write_csv(os.path.join(self.results_dir, f"{self.file_prefix}_H2_2_persistence_summary.csv"))
            print(f"\nH2.2 Summary: {supported_count}/{len(summary_rows)} tests supported (p<0.1 & slope>0).")
        else:
            print("\nNo results for H2.2 to save/plot.")


    def _test_h2_3_asymmetric_volatility_response(self):
        print("\n--- H2.3: Testing Asymmetric Volatility Response (GJR-GARCH) ---")
        if 'gjr' not in self.model_types or 'garch' not in self.model_types:
            print("Error: Both GJR and GARCH models needed for H2.3.")
            return
        if not self.event_garch_fits:
            print("Error: GARCH models not fitted. Cannot test H2.3.")
            return

        # H2.3: Asymmetric volatility response (GJR-GARCH gamma) correlates with asymmetric price adjustment.
        # We test if GJR-GARCH (which has gamma) explains volatility better after negative returns
        # compared to standard GARCH.
        # We look at the difference in conditional volatility: GJR_vol - GARCH_vol
        # This difference should be larger (more positive) following negative shocks if GJR captures asymmetry.
        
        vol_diffs_after_neg_ret = []
        vol_diffs_after_pos_ret = []

        for event_id, data in self.event_garch_fits.items():
            if data['fitted_models'].get('gjr') is None or data['fitted_models'].get('garch') is None:
                continue

            # Align GJR and GARCH outputs with common set of days_to_event (from non-null returns)
            event_full_data = self.analyzer.data.filter(pl.col('event_id') == event_id).sort('days_to_event')
            event_nn_returns_data = event_full_data.filter(pl.col(self.return_col).is_not_null())
            nn_days_to_event = event_nn_returns_data.select('days_to_event').to_series()
            nn_returns = event_nn_returns_data.select(self.return_col).to_series()

            gjr_vol = data['cond_volatility'].get('gjr') # sqrt(h_t) from GJR
            garch_vol = data['cond_volatility'].get('garch') # sqrt(h_t) from GARCH
            
            if gjr_vol is None or garch_vol is None or len(gjr_vol) != len(garch_vol): continue
            if len(gjr_vol) != len(nn_days_to_event): continue # Ensure alignment

            # Compare vol_t based on return_{t-1}
            for i in range(1, len(nn_returns)): # Start from second obs as we need return_{t-1}
                prev_return = nn_returns[i-1]
                vol_diff = gjr_vol[i] - garch_vol[i] # Volatility at time t

                if prev_return < 0:
                    vol_diffs_after_neg_ret.append(vol_diff)
                else: # prev_return >= 0
                    vol_diffs_after_pos_ret.append(vol_diff)
        
        if len(vol_diffs_after_neg_ret) > 10 and len(vol_diffs_after_pos_ret) > 10:
            mean_diff_neg = np.mean(vol_diffs_after_neg_ret)
            mean_diff_pos = np.mean(vol_diffs_after_pos_ret)
            
            # T-test: is mean_diff_neg significantly greater than mean_diff_pos?
            t_stat, p_value_ttest = stats.ttest_ind(vol_diffs_after_neg_ret, vol_diffs_after_pos_ret, equal_var=False, alternative='greater')
            
            is_supported = p_value_ttest < 0.05 # One-sided test, GJR vol > GARCH vol more after neg shocks
            self.asymmetry_results_h2_3 = {
                'mean_vol_diff_after_neg_ret': mean_diff_neg,
                'mean_vol_diff_after_pos_ret': mean_diff_pos,
                't_stat': t_stat, 'p_value': p_value_ttest,
                'n_neg_shocks': len(vol_diffs_after_neg_ret),
                'n_pos_shocks': len(vol_diffs_after_pos_ret),
                'supported': is_supported
            }
            print(f"  Mean (GJR_vol - GARCH_vol) after neg shocks: {mean_diff_neg:.6f}")
            print(f"  Mean (GJR_vol - GARCH_vol) after pos shocks: {mean_diff_pos:.6f}")
            print(f"  T-test (one-sided, H_alt: diff_neg > diff_pos): t={t_stat:.2f}, p={p_value_ttest:.3f}")
            print(f"  H2.3 Supported (p<0.05): {'YES' if is_supported else 'NO'}")
            
            res_df = pl.DataFrame([self.asymmetry_results_h2_3])
            res_df.write_csv(os.path.join(self.results_dir, f"{self.file_prefix}_H2_3_asymmetry_summary.csv"))
        else:
            print("  Insufficient data for H2.3 t-test.")
            self.asymmetry_results_h2_3 = None
            
    def generate_summary_report(self):
        print("\n--- Generating Hypothesis 2 Summary Report ---")
        h2_1_supported_overall = any(
            preds.get(w, {}).get('p_value', 1.0) < 0.1
            for mt, preds in self.prediction_results_h2_1.items()
            for w in self.prediction_windows
        )
        
        h2_2_supported_overall = any(
            res.get('p_value', 1.0) < 0.1 and res.get('slope', -1) > 0
            for mt, res_dict in self.persistence_results_h2_2.items()
            for res_key, res in (res_dict.items() if isinstance(res_dict, dict) else [(None, res_dict)]) if res # handle if res_dict is None or contains None
        )
        
        h2_3_supported_overall = self.asymmetry_results_h2_3.get('supported', False) if self.asymmetry_results_h2_3 else False

        report_data = {
            'Sub-Hypothesis': [
                "H2.1: Pre-event volatility innovations predict subsequent returns",
                "H2.2: Post-event volatility persistence extends elevated expected returns",
                "H2.3: Asymmetric volatility response correlates with asymmetric price adjustment"
            ],
            'Supported': [h2_1_supported_overall, h2_2_supported_overall, h2_3_supported_overall],
            'Details': [
                f"{sum(1 for mt, preds in self.prediction_results_h2_1.items() for w_res in preds.values() if w_res and w_res['p_value'] < 0.1)} significant tests (p<0.1)",
                f"{sum(1 for mt, res_dict in self.persistence_results_h2_2.items() for res_key, res in (res_dict.items() if isinstance(res_dict,dict) else [(None,res_dict)]) if res and res['p_value'] < 0.1 and res['slope'] > 0)} significant tests (p<0.1, slope>0)",
                f"p-value={self.asymmetry_results_h2_3.get('p_value',1.0):.3f}" if self.asymmetry_results_h2_3 else "N/A"
            ]
        }
        summary_df = pl.DataFrame(report_data)
        
        overall_h2_supported = all(report_data['Supported'])
        
        report_file = os.path.join(self.results_dir, f"{self.file_prefix}_H2_summary_report.md")
        with open(report_file, 'w') as f:
            f.write(f"# Hypothesis 2 Analysis Report: {self.file_prefix.upper()}\n\n")
            f.write("## Hypothesis Statement (from paper)\n")
            f.write("> GARCH-estimated conditional volatility innovations serve as an effective proxy for impact uncertainty.\n\n")
            f.write("## Overall Result\n")
            f.write(f"**Hypothesis 2 is {'SUPPORTED' if overall_h2_supported else 'PARTIALLY SUPPORTED' if any(report_data['Supported']) else 'NOT SUPPORTED'}**.\n\n")
            f.write(summary_df.to_pandas().to_markdown(index=False) + "\n") # Use pandas for markdown

        print(f"Hypothesis 2 report saved to: {report_file}")
        summary_df.write_csv(os.path.join(self.results_dir, f"{self.file_prefix}_H2_sub_hypotheses.csv"))


    def run_full_h2_analysis(self):
        if self._fit_garch_for_events():
            self._test_h2_1_innovations_predict_returns()
            self._test_h2_2_volatility_persistence()
            self._test_h2_3_asymmetric_volatility_response()
            self.generate_summary_report()
            return True
        return False

def run_fda_analysis():
    print("\n=== Starting FDA Approval Event Analysis for Hypothesis 2 (Paper Version) ===")
    os.makedirs(FDA_RESULTS_DIR, exist_ok=True)
    print(f"FDA results will be saved to: {os.path.abspath(FDA_RESULTS_DIR)}")
    try:
        data_loader = EventDataLoader(
            event_path=FDA_EVENT_FILE, stock_paths=STOCK_FILES, window_days=WINDOW_DAYS,
            event_date_col=FDA_EVENT_DATE_COL, ticker_col=FDA_TICKER_COL
        )
        feature_engineer = EventFeatureEngineer(prediction_window=max(PREDICTION_WINDOWS)) # Not heavily used by H2
        analyzer = EventAnalysis(data_loader, feature_engineer)
        
        print("\nLoading and preparing FDA data (minimal features for H2)...")
        analyzer.data = analyzer.load_and_prepare_data(run_feature_engineering=False) # H2 relies on returns for GARCH
        if analyzer.data is None or analyzer.data.is_empty():
            print("\n*** Error: FDA data loading failed. ***")
            return False
        print(f"FDA data loaded. Shape: {analyzer.data.shape}")

        h2_analyzer = Hypothesis2Analyzer(analyzer, FDA_RESULTS_DIR, FDA_FILE_PREFIX)
        success = h2_analyzer.run_full_h2_analysis()
        
        print(f"\n--- FDA Event Analysis for Hypothesis 2 {'Finished' if success else 'Failed'} (Results in '{FDA_RESULTS_DIR}') ---")
        return success
    except Exception as e:
        print(f"\n*** An unexpected error occurred in FDA H2 analysis: {e} ***")
        traceback.print_exc()
    return False

def run_earnings_analysis():
    print("\n=== Starting Earnings Announcement Event Analysis for Hypothesis 2 (Paper Version) ===")
    os.makedirs(EARNINGS_RESULTS_DIR, exist_ok=True)
    print(f"Earnings results will be saved to: {os.path.abspath(EARNINGS_RESULTS_DIR)}")
    try:
        data_loader = EventDataLoader(
            event_path=EARNINGS_EVENT_FILE, stock_paths=STOCK_FILES, window_days=WINDOW_DAYS,
            event_date_col=EARNINGS_EVENT_DATE_COL, ticker_col=EARNINGS_TICKER_COL
        )
        feature_engineer = EventFeatureEngineer(prediction_window=max(PREDICTION_WINDOWS))
        analyzer = EventAnalysis(data_loader, feature_engineer)

        print("\nLoading and preparing Earnings data (minimal features for H2)...")
        analyzer.data = analyzer.load_and_prepare_data(run_feature_engineering=False)
        if analyzer.data is None or analyzer.data.is_empty():
            print("\n*** Error: Earnings data loading failed. ***")
            return False
        print(f"Earnings data loaded. Shape: {analyzer.data.shape}")

        h2_analyzer = Hypothesis2Analyzer(analyzer, EARNINGS_RESULTS_DIR, EARNINGS_FILE_PREFIX)
        success = h2_analyzer.run_full_h2_analysis()

        print(f"\n--- Earnings Event Analysis for Hypothesis 2 {'Finished' if success else 'Failed'} (Results in '{EARNINGS_RESULTS_DIR}') ---")
        return success
    except Exception as e:
        print(f"\n*** An unexpected error occurred in Earnings H2 analysis: {e} ***")
        traceback.print_exc()
    return False

def compare_results():
    """Compares Hypothesis 2 results between FDA and earnings events."""
    print("\n=== Comparing FDA and Earnings Results for Hypothesis 2 ===")
    comparison_dir = "results/hypothesis2/comparison/"
    os.makedirs(comparison_dir, exist_ok=True)

    try:
        fda_h2_subs_file = os.path.join(FDA_RESULTS_DIR, f"{FDA_FILE_PREFIX}_H2_sub_hypotheses.csv")
        earn_h2_subs_file = os.path.join(EARNINGS_RESULTS_DIR, f"{EARNINGS_FILE_PREFIX}_H2_sub_hypotheses.csv")

        if not os.path.exists(fda_h2_subs_file) or not os.path.exists(earn_h2_subs_file):
            print(f"Error: H2 sub-hypothesis summary files missing.")
            return False

        fda_subs = pl.read_csv(fda_h2_subs_file)
        earn_subs = pl.read_csv(earn_h2_subs_file)

        # Align by sub-hypothesis name for comparison
        fda_subs = fda_subs.rename({"Sub-Hypothesis": "sub_hypothesis", "Supported": "fda_supported", "Details": "fda_details"})
        earn_subs = earn_subs.rename({"Sub-Hypothesis": "sub_hypothesis", "Supported": "earn_supported", "Details": "earn_details"})
        
        comp_df = fda_subs.join(earn_subs, on="sub_hypothesis", how="outer")
        comp_df = comp_df.select(['sub_hypothesis', 'fda_supported', 'earn_supported', 'fda_details', 'earn_details'])
        
        comp_df.write_csv(os.path.join(comparison_dir, "H2_comparison_sub_hypotheses.csv"))
        print("\nHypothesis 2 Sub-Hypothesis Comparison:")
        print(comp_df)

        # Plotting comparison (simplified)
        fig, ax = plt.subplots(figsize=(10, len(comp_df) * 0.5 + 2)) # Dynamic height
        bar_height = 0.35
        y_pos = np.arange(len(comp_df))

        ax.barh(y_pos + bar_height/2, comp_df['fda_supported'].cast(pl.Int8).to_numpy(), bar_height, label='FDA Supported', color='skyblue')
        ax.barh(y_pos - bar_height/2, comp_df['earn_supported'].cast(pl.Int8).to_numpy(), bar_height, label='Earnings Supported', color='lightcoral')
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(comp_df['sub_hypothesis'].to_list())
        ax.set_xlabel('Supported (1=Yes, 0=No)')
        ax.set_title('Hypothesis 2 Support Comparison')
        ax.legend()
        plt.gca().invert_yaxis() # Display H2.1 at top
        plt.tight_layout()
        plt.savefig(os.path.join(comparison_dir, "H2_comparison_plot.png"), dpi=200)
        plt.close(fig)
        print(f"H2 Comparison plot saved to {comparison_dir}")
        return True

    except Exception as e:
        print(f"Error comparing H2 results: {e}")
        traceback.print_exc()
    return False


def main():
    fda_success = run_fda_analysis()
    earnings_success = run_earnings_analysis()

    if fda_success and earnings_success:
        compare_results()
    elif fda_success:
        print("\n=== FDA H2 analysis completed. Earnings analysis failed or was skipped. ===")
    elif earnings_success:
        print("\n=== Earnings H2 analysis completed. FDA analysis failed or was skipped. ===")
    else:
        print("\n=== Both FDA and Earnings H2 analyses failed. ===")

if __name__ == "__main__":
    main()