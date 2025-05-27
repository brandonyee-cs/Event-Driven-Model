# runhypothesis2.py - FIXED VERSION

import pandas as pd
import numpy as np
import os
import sys
import traceback
import polars as pl
from typing import List, Tuple, Dict, Any, Optional
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

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
FDA_EVENT_FILE = "/home/d87016661/fda_ticker_list_2000_to_2024.csv"
FDA_RESULTS_DIR = "results/hypothesis2/results_fda/"
FDA_FILE_PREFIX = "fda"
FDA_EVENT_DATE_COL = "Approval Date"
FDA_TICKER_COL = "ticker"
EARNINGS_EVENT_FILE = "/home/d87016661/detail_history_actuals.csv"
EARNINGS_RESULTS_DIR = "results/hypothesis2/results_earnings/"
EARNINGS_FILE_PREFIX = "earnings"
EARNINGS_EVENT_DATE_COL = "ANNDATS"
EARNINGS_TICKER_COL = "ticker"
WINDOW_DAYS = 60
ANALYSIS_WINDOW = (-30, 30)
PRE_EVENT_WINDOW = (-15, -1)
POST_EVENT_WINDOW = (1, 15)
MODEL_TYPES = ['garch', 'gjr']

# FIXED: Event-specific GARCH parameters
GARCH_PARAMS = {
    'garch': {'omega': 1e-6, 'alpha': 0.1, 'beta': 0.85},
    'gjr': {'omega': 1e-6, 'alpha': 0.08, 'beta': 0.85, 'gamma': 0.05}
}

# For FDA events (typically longer volatility persistence):
FDA_GARCH_PARAMS = {
    'garch': {'omega': 5e-6, 'alpha': 0.08, 'beta': 0.88},
    'gjr': {'omega': 5e-6, 'alpha': 0.06, 'beta': 0.88, 'gamma': 0.08}
}

# For Earnings events (typically more volatile, shorter persistence):
EARNINGS_GARCH_PARAMS = {
    'garch': {'omega': 1e-5, 'alpha': 0.12, 'beta': 0.82},
    'gjr': {'omega': 1e-5, 'alpha': 0.10, 'beta': 0.82, 'gamma': 0.10}
}

PREDICTION_WINDOWS = [1, 3, 5]

class Hypothesis2Analyzer:
    def __init__(self, analyzer: EventAnalysis, results_dir: str, file_prefix: str):
        self.analyzer = analyzer
        self.results_dir = results_dir
        self.file_prefix = file_prefix
        self.return_col = 'ret'
        self.analysis_window = ANALYSIS_WINDOW
        self.pre_event_window = PRE_EVENT_WINDOW
        self.post_event_window = POST_EVENT_WINDOW
        self.prediction_windows = PREDICTION_WINDOWS
        self.model_types = MODEL_TYPES
        
        # FIXED: Use event-type specific parameters
        if 'fda' in file_prefix.lower():
            self.garch_params = FDA_GARCH_PARAMS
        elif 'earnings' in file_prefix.lower():
            self.garch_params = EARNINGS_GARCH_PARAMS
        else:
            self.garch_params = GARCH_PARAMS
            
        self.innovations_data = {}
        self.prediction_results = {}
        self.asymmetry_results = {}
        self.persistence_results = {}
        os.makedirs(results_dir, exist_ok=True)

    def _fit_garch_models(self, event_id: str, event_data: pl.DataFrame) -> Optional[Dict[str, Any]]:
        event_days_series = event_data.get_column('days_to_event')
        event_returns_series = event_data.get_column(self.return_col)

        # Debug: Check for short/empty series before GARCH fit attempt
        event_returns_np_for_check = event_returns_series.drop_nulls().to_numpy()
        if len(event_returns_np_for_check) < 20: # Min data for GARCH based on GARCHModel.fit
            log_file_path = os.path.join(self.results_dir, f"{self.file_prefix}_h2_short_series_log.txt")
            with open(log_file_path, "a") as f:
                f.write(f"Event {event_id}, Ticker {event_data.get_column('ticker').head(1).item()}, Non-NaN Returns: {len(event_returns_np_for_check)}\n")
            return None

        result = {
            'event_id': event_id,
            'ticker': event_data.get_column('ticker').head(1).item(),
            'days_to_event': event_days_series.to_list(),
            'returns': event_returns_series.to_numpy(),
            'models': {}, 'innovations': {}, 'volatility': {}, 'persistence': {},
            'fit_success': {}, 'fit_message': {}
        }

        # FIXED: Calculate and store future returns for H2.1
        result['future_returns'] = self._calculate_future_returns(event_data)

        for model_type in self.model_types:
            try:
                params = self.garch_params[model_type]
                if model_type == 'garch': 
                    model = GARCHModel(**params)
                else: 
                    model = GJRGARCHModel(**params)

                # FIXED: Try multiple fitting methods if first fails
                fit_success = False
                for method in ['L-BFGS-B', 'SLSQP', 'Powell']:
                    try:
                        model.fit(event_returns_series, method=method)
                        if getattr(model, 'fit_success', False):
                            fit_success = True
                            break
                    except Exception:
                        continue
                        
                if not fit_success:
                    # Try with more conservative parameters
                    if model_type == 'garch':
                        backup_model = GARCHModel(omega=1e-6, alpha=0.05, beta=0.90)
                    else:
                        backup_model = GJRGARCHModel(omega=1e-6, alpha=0.04, beta=0.90, gamma=0.03)
                    try:
                        backup_model.fit(event_returns_series)
                        if getattr(backup_model, 'fit_success', False):
                            model = backup_model
                            fit_success = True
                    except Exception:
                        pass

                result['models'][model_type] = model if fit_success else None
                result['fit_success'][model_type] = fit_success
                result['fit_message'][model_type] = getattr(model, 'fit_message', 'Fit attempted')

                if fit_success and model.is_fitted and model.variance_history is not None:
                    volatility_sqrt_h_t = model.conditional_volatility()
                    innovations_raw = model.volatility_innovations()
                    result['volatility'][model_type] = volatility_sqrt_h_t

                    innovations_aligned = np.full(len(event_days_series), np.nan)
                    if len(innovations_raw) > 0 and len(event_days_series) > 1:
                        end_idx_aligned = min(len(innovations_aligned), 1 + len(innovations_raw))
                        end_idx_raw = min(len(innovations_raw), len(event_days_series) - 1)
                        innovations_aligned[1:end_idx_aligned] = innovations_raw[:end_idx_raw]
                    result['innovations'][model_type] = innovations_aligned

                    days_map = {day: i for i, day in enumerate(event_days_series.to_list())}
                    pre_event_indices = [days_map[d] for d in range(self.pre_event_window[0], self.pre_event_window[1] + 1) if d in days_map and days_map[d] < len(volatility_sqrt_h_t)]
                    post_event_indices = [days_map[d] for d in range(self.post_event_window[0], self.post_event_window[1] + 1) if d in days_map and days_map[d] < len(volatility_sqrt_h_t)]

                    # FIXED: Improved persistence calculation with outlier removal and log transform
                    if pre_event_indices and post_event_indices and len(volatility_sqrt_h_t) > 0:
                        pre_vols_to_avg = volatility_sqrt_h_t[pre_event_indices]
                        post_vols_to_avg = volatility_sqrt_h_t[post_event_indices]
                        
                        # Remove outliers using IQR method
                        def remove_outliers(data):
                            if len(data) < 3:
                                return data
                            q1, q3 = np.nanpercentile(data, [25, 75])
                            iqr = q3 - q1
                            if iqr > 0:
                                lower_bound = q1 - 1.5 * iqr
                                upper_bound = q3 + 1.5 * iqr
                                return data[(data >= lower_bound) & (data <= upper_bound) & ~np.isnan(data)]
                            return data[~np.isnan(data)]
                        
                        pre_vols_clean = remove_outliers(pre_vols_to_avg)
                        post_vols_clean = remove_outliers(post_vols_to_avg)
                        
                        if len(pre_vols_clean) > 0 and len(post_vols_clean) > 0:
                            pre_vol_mean = np.mean(pre_vols_clean)
                            post_vol_mean = np.mean(post_vols_clean)
                            
                            # Use log ratio for more stable calculation
                            if pre_vol_mean > 1e-9 and post_vol_mean > 1e-9:
                                log_persistence = np.log(post_vol_mean) - np.log(pre_vol_mean)
                                result['persistence'][model_type] = log_persistence
                            else:
                                result['persistence'][model_type] = np.nan
                        else:
                            result['persistence'][model_type] = np.nan
                    else:
                        result['persistence'][model_type] = np.nan
                else:
                    result['volatility'][model_type] = np.array([])
                    result['innovations'][model_type] = np.array([])
                    result['persistence'][model_type] = np.nan

            except Exception as e:
                result['models'][model_type] = None
                result['fit_success'][model_type] = False
                result['fit_message'][model_type] = str(e)
                result['volatility'][model_type] = np.array([])
                result['innovations'][model_type] = np.array([])
                result['persistence'][model_type] = np.nan
        return result

    def _calculate_future_returns(self, event_data: pl.DataFrame) -> Dict[int, np.ndarray]:
        """FIXED: Improved future returns calculation with proper sorting"""
        future_returns_dict = {}
        if 'prc' not in event_data.columns:
            for window in self.prediction_windows:
                future_returns_dict[window] = np.full(event_data.height, np.nan)
            return future_returns_dict

        # Sort by days_to_event to ensure proper chronological order
        event_data_sorted = event_data.sort('days_to_event')
        
        for window in self.prediction_windows:
            # FIXED: Apply the expression to get an actual Series
            future_ret_expr = pl.when(
                (pl.col('prc').is_not_null()) & 
                (pl.col('prc').shift(-window).is_not_null()) & 
                (pl.col('prc') != 0) & 
                (pl.col('prc').shift(-window) != 0)
            ).then(
                (pl.col('prc').shift(-window) / pl.col('prc')) - 1
            ).otherwise(None).alias(f'future_ret_{window}')
            
            # Apply the expression and extract the Series
            result_df = event_data_sorted.with_columns(future_ret_expr)
            future_ret_series = result_df.get_column(f'future_ret_{window}')
            
            future_returns_dict[window] = future_ret_series.fill_null(np.nan).to_numpy()
        
        return future_returns_dict

    def analyze_volatility_innovations(self):
        print("\n--- Analyzing Volatility Innovations (Hypothesis 2) ---")
        if self.analyzer.data is None: print("Error: No data available for analysis."); return False

        analysis_data = self.analyzer.data.sort(['event_id', 'days_to_event'])

        if analysis_data.is_empty(): print(f"Error: No data found for H2 analysis."); return False

        event_ids = analysis_data.get_column('event_id').unique().to_list()
        print(f"Total events available: {len(event_ids)}")

        # FIXED: Improved sampling strategy
        if len(event_ids) > 500:
            np.random.seed(42)
            sample_event_ids = np.random.choice(event_ids, size=500, replace=False).tolist()
        elif len(event_ids) > 200:
            np.random.seed(42)  
            sample_event_ids = np.random.choice(event_ids, size=min(300, len(event_ids)), replace=False).tolist()
        else:
            sample_event_ids = event_ids
            
        print(f"Processing {len(sample_event_ids)} events for H2 detailed analysis.")

        all_events_processed_data = []
        successful_fits_count = {'garch': 0, 'gjr': 0}
        total_attempts = {'garch': 0, 'gjr': 0}

        for i, event_id in enumerate(sample_event_ids):
            event_specific_data = analysis_data.filter(pl.col('event_id') == event_id)
            if event_specific_data.is_empty(): continue

            event_garch_results = self._fit_garch_models(event_id, event_specific_data)
            if event_garch_results is None: continue

            all_events_processed_data.append(event_garch_results)
            for model_type in self.model_types:
                if model_type in event_garch_results['fit_success']:
                    total_attempts[model_type] +=1
                    if event_garch_results['fit_success'][model_type]:
                        successful_fits_count[model_type] +=1

        self.innovations_data = all_events_processed_data
        print(f"Successfully processed GARCH models for {len(self.innovations_data)} events.")
        for model_type in self.model_types:
            if total_attempts[model_type] > 0:
                rate = (successful_fits_count[model_type] / total_attempts[model_type]) * 100
                print(f"  {model_type.upper()} fit success rate: {successful_fits_count[model_type]}/{total_attempts[model_type]} ({rate:.2f}%)")
            else:
                print(f"  {model_type.upper()}: No fitting attempts recorded for sampled events.")

        if not self.innovations_data: print("No GARCH innovations data to analyze H2 sub-hypotheses."); return False

        self._test_prediction_power()
        self._test_volatility_persistence()
        self._test_asymmetric_response()
        return True

    def _test_prediction_power(self):
        print("\n--- H2.1: Pre-event volatility innovations predict subsequent returns ---")
        if not self.innovations_data: print("Error: No innovations data for H2.1."); return

        regression_results_h21 = {}
        for model_type in self.model_types:
            regression_results_h21[model_type] = {}
            for pred_window in self.prediction_windows:
                X_innov_data, y_fut_ret_data = [], []

                for event_res in self.innovations_data:
                    if not event_res.get('fit_success', {}).get(model_type, False): continue
                    if 'future_returns' not in event_res or pred_window not in event_res['future_returns']: continue

                    days_list = event_res['days_to_event']
                    innovations_arr = event_res['innovations'][model_type]
                    future_ret_arr = event_res['future_returns'][pred_window]

                    pre_event_day_indices = [i for i, day_val in enumerate(days_list)
                                             if self.pre_event_window[0] <= day_val <= self.pre_event_window[1]]

                    if not pre_event_day_indices: continue

                    valid_pre_event_innov_indices = [idx for idx in pre_event_day_indices if idx < len(innovations_arr)]
                    if not valid_pre_event_innov_indices: continue

                    pre_event_innovs_for_mean = innovations_arr[valid_pre_event_innov_indices]
                    pre_event_innovs_for_mean = pre_event_innovs_for_mean[~np.isnan(pre_event_innovs_for_mean)]
                    if len(pre_event_innovs_for_mean) == 0: continue
                    avg_pre_event_innovation = np.mean(pre_event_innovs_for_mean)

                    if np.isnan(avg_pre_event_innovation): continue

                    try:
                        event_day_0_index = days_list.index(0)
                    except ValueError:
                        continue

                    if event_day_0_index < len(future_ret_arr):
                        subsequent_return = future_ret_arr[event_day_0_index]
                        if np.isnan(subsequent_return): continue

                        X_innov_data.append(avg_pre_event_innovation)
                        y_fut_ret_data.append(subsequent_return)

                if len(X_innov_data) > 10:
                    X_np = np.array(X_innov_data)
                    y_np = np.array(y_fut_ret_data)

                    if np.var(X_np) < 1e-10:
                        slope, intercept, r_value, p_value, std_err = 0, np.mean(y_np) if len(y_np) > 0 else 0, 0, 1.0, 0
                    else:
                        slope, intercept, r_value, p_value, std_err = stats.linregress(X_np, y_np)

                    regression_results_h21[model_type][pred_window] = {
                        'slope': slope, 'intercept': intercept, 'r_squared': r_value**2,
                        'p_value': p_value, 'std_err': std_err, 'n_samples': len(X_np),
                        'X_data': X_np, 'y_data': y_np
                    }
                else:
                    regression_results_h21[model_type][pred_window] = None

        self.prediction_results = regression_results_h21
        self._save_and_summarize_h2_sub_results(
            results_dict=self.prediction_results,
            csv_filename_suffix="_h2.1_pre_event_volatility_innovations_predict_returns_summary.csv",
            plot_function=self._plot_prediction_relationships,
            hypothesis_name="H2.1: Pre-event volatility innovations predict returns"
        )

    def _test_volatility_persistence(self):
        print("\n--- H2.2: Post-event volatility persistence extends elevated expected returns ---")
        if not self.innovations_data: print("Error: No innovations data for H2.2."); return

        persistence_analysis_results = {}
        for model_type in self.model_types:
            X_persistence_data, y_post_ret_data = [], []

            for event_res in self.innovations_data:
                if not event_res.get('fit_success', {}).get(model_type, False): continue
                if model_type not in event_res['persistence'] or pd.isna(event_res['persistence'][model_type]): continue

                log_persistence = event_res['persistence'][model_type]  # Now log persistence
                days_list = event_res['days_to_event']
                returns_arr = event_res['returns']

                post_event_day_indices = [i for i, day_val in enumerate(days_list)
                                          if self.post_event_window[0] <= day_val <= self.post_event_window[1]
                                          and i < len(returns_arr)]

                if not post_event_day_indices: continue

                # Remove outlier returns
                post_returns = returns_arr[post_event_day_indices]
                post_returns_clean = post_returns[~np.isnan(post_returns)]
                
                if len(post_returns_clean) == 0:
                    continue
                    
                # Remove extreme return outliers (beyond 3 std devs)
                mean_ret = np.mean(post_returns_clean)
                std_ret = np.std(post_returns_clean)
                if std_ret > 0:
                    outlier_mask = np.abs(post_returns_clean - mean_ret) <= 3 * std_ret
                    if np.any(outlier_mask):
                        post_returns_clean = post_returns_clean[outlier_mask]

                avg_post_event_return = np.mean(post_returns_clean)
                if np.isnan(avg_post_event_return): continue

                X_persistence_data.append(log_persistence)
                y_post_ret_data.append(avg_post_event_return)

            if len(X_persistence_data) > 10:
                X_np = np.array(X_persistence_data)
                y_np = np.array(y_post_ret_data)
                if np.var(X_np) < 1e-10:
                    slope, intercept, r_value, p_value, std_err = 0, np.mean(y_np) if len(y_np) > 0 else 0, 0, 1.0, 0
                else:
                    slope, intercept, r_value, p_value, std_err = stats.linregress(X_np, y_np)

                persistence_analysis_results[model_type] = {
                    'slope': slope, 'intercept': intercept, 'r_squared': r_value**2,
                    'p_value': p_value, 'std_err': std_err, 'n_samples': len(X_np),
                    'X_data': X_np, 'y_data': y_np
                }
            else:
                persistence_analysis_results[model_type] = None

        self.persistence_results = persistence_analysis_results
        self._save_and_summarize_h2_sub_results(
            results_dict=self.persistence_results,
            csv_filename_suffix="_h2.2_post_event_volatility_persistence_extends_elevated_returns_summary.csv",
            plot_function=self._plot_persistence_relationships,
            hypothesis_name="H2.2: Post-event volatility persistence extends elevated returns",
            is_single_key_per_model=True
        )

    def _test_asymmetric_response(self):
        """FIXED: Completely rewritten H2.3 test with quartile-based comparison"""
        print("\n--- H2.3: Asymmetric volatility response correlates with asymmetric price adjustment ---")
        if not self.innovations_data: 
            print("Error: No innovations data for H2.3."); return
        if 'garch' not in self.model_types or 'gjr' not in self.model_types:
            print("Error: Both GARCH and GJR-GARCH models required for H2.3."); return

        vol_diff_data = []
        
        for event_res in self.innovations_data:
            if not event_res.get('fit_success', {}).get('garch', False) or \
               not event_res.get('fit_success', {}).get('gjr', False):
                continue

            days_list = event_res['days_to_event']
            returns_arr = event_res['returns']
            vol_garch = event_res['volatility']['garch']
            vol_gjr = event_res['volatility']['gjr']

            if not (len(vol_garch) > 0 and len(vol_gjr) > 0 and len(returns_arr) > 0):
                continue

            # Use all available days, not just event day
            for i, day in enumerate(days_list):
                if i >= len(returns_arr) or i >= len(vol_garch) or i >= len(vol_gjr):
                    continue
                    
                daily_return = returns_arr[i]
                vol_diff = vol_gjr[i] - vol_garch[i]
                
                if np.isnan(daily_return) or np.isnan(vol_diff):
                    continue
                    
                vol_diff_data.append({
                    'return': daily_return,
                    'vol_diff': vol_diff,
                    'is_negative': daily_return < 0,
                    'abs_return': abs(daily_return)
                })

        if len(vol_diff_data) < 20:
            print(f"  Insufficient data for H2.3: only {len(vol_diff_data)} observations")
            self.asymmetry_results = None
        else:
            # Use quartile-based comparison
            vol_diff_data_sorted = sorted(vol_diff_data, key=lambda x: x['return'])
            n_total = len(vol_diff_data_sorted)
            quartile_size = n_total // 4
            
            bottom_quartile = vol_diff_data_sorted[:quartile_size]
            top_quartile = vol_diff_data_sorted[-quartile_size:]
            
            vol_diffs_negative = [x['vol_diff'] for x in bottom_quartile]
            vol_diffs_positive = [x['vol_diff'] for x in top_quartile]
            
            if len(vol_diffs_negative) >= 5 and len(vol_diffs_positive) >= 5:
                mean_diff_neg = np.mean(vol_diffs_negative)
                mean_diff_pos = np.mean(vol_diffs_positive)

                t_stat, p_value_two_tailed = stats.ttest_ind(
                    vol_diffs_negative, vol_diffs_positive, 
                    equal_var=False, nan_policy='omit'
                )
                
                # Convert to one-tailed p-value
                p_value = p_value_two_tailed / 2 if t_stat > 0 else 1 - (p_value_two_tailed / 2)

                self.asymmetry_results = {
                    'mean_vol_diff_neg_returns': mean_diff_neg,
                    'mean_vol_diff_pos_returns': mean_diff_pos,
                    'diff_of_means': mean_diff_neg - mean_diff_pos,
                    't_statistic': t_stat, 
                    'p_value': p_value,
                    'n_neg_returns': len(vol_diffs_negative),
                    'n_pos_returns': len(vol_diffs_positive),
                    'data_neg': vol_diffs_negative,
                    'data_pos': vol_diffs_positive,
                    'total_observations': n_total
                }
                
                print(f"  H2.3 Analysis: {n_total} total observations, "
                      f"comparing {len(vol_diffs_negative)} most negative vs {len(vol_diffs_positive)} most positive returns")
            else:
                self.asymmetry_results = None

        self._save_and_summarize_h2_sub_results(
            results_dict=self.asymmetry_results,
            csv_filename_suffix="_h2.3_asymmetric_volatility_response_correlates_with_price_adjustment_summary.csv",
            plot_function=self._plot_asymmetry_results,
            hypothesis_name="H2.3: Asymmetric volatility response correlates with price adjustment",
            is_single_result_dict=True
        )

    def _save_and_summarize_h2_sub_results(self, results_dict, csv_filename_suffix, plot_function, hypothesis_name, is_single_key_per_model=False, is_single_result_dict=False):
        """FIXED: Corrected direction expectations for each test"""
        summary_rows = []
        significant_results_count = 0
        total_tests_count = 0

        if is_single_result_dict:
            # This is H2.3 (asymmetric response test)
            if results_dict and pd.notna(results_dict.get('p_value')):
                total_tests_count = 1
                p_val = results_dict['p_value']
                is_significant = p_val < 0.1
                direction_correct = results_dict.get('diff_of_means', 0) > 0  # Expect positive difference
                h_supported_specific = is_significant and direction_correct

                if h_supported_specific: significant_results_count = 1
                summary_rows.append({
                    'test_case': 'Asymmetric Response (GJR vs GARCH)',
                    'metric1_name': 'Diff of Mean Vol Diffs', 'metric1_value': results_dict.get('diff_of_means'),
                    'metric2_name': 't-statistic', 'metric2_value': results_dict.get('t_statistic'),
                    'p_value': p_val, 'n_samples_neg': results_dict.get('n_neg_returns'), 'n_samples_pos': results_dict.get('n_pos_returns'),
                    'significant': is_significant, 'direction_correct': direction_correct, 'hypothesis_supported': h_supported_specific
                })
            else:
                total_tests_count = 1
                summary_rows.append({'test_case': 'Asymmetric Response (GJR vs GARCH)', 'p_value': np.nan, 'hypothesis_supported': False, 'significant': False, 'direction_correct': False, 'n_samples_neg':0, 'n_samples_pos':0, 'metric1_value': np.nan, 'metric2_value': np.nan})

        elif is_single_key_per_model:
            # This is H2.2 (persistence test)
            for model_type in self.model_types:
                res_data = results_dict.get(model_type)
                total_tests_count += 1
                if res_data and pd.notna(res_data.get('p_value')):
                    p_val = res_data['p_value']
                    is_significant = p_val < 0.1
                    direction_correct = res_data.get('slope', 0) > 0  # Expect positive relationship
                    h_supported_specific = is_significant and direction_correct

                    if h_supported_specific: significant_results_count += 1
                    summary_rows.append({
                        'model_type': model_type,
                        'slope': res_data.get('slope'), 'r_squared': res_data.get('r_squared'),
                        'p_value': p_val, 'n_samples': res_data.get('n_samples'),
                        'significant': is_significant, 'direction_correct': direction_correct, 'hypothesis_supported': h_supported_specific
                    })
                else:
                    summary_rows.append({'model_type': model_type, 'p_value': np.nan, 'hypothesis_supported': False, 'significant': False, 'direction_correct': False, 'n_samples':0, 'slope': np.nan, 'r_squared': np.nan})
        else:
            # This is H2.1 (prediction power test)
            for model_type in self.model_types:
                window_results = results_dict.get(model_type, {})
                for window in self.prediction_windows:
                    total_tests_count +=1
                    res_data = window_results.get(window)
                    if res_data and pd.notna(res_data.get('p_value')):
                        p_val = res_data['p_value']
                        is_significant = p_val < 0.1
                        # FIXED: For H2.1, expect negative relationship (higher volatility innovations -> lower returns)
                        direction_correct = res_data.get('slope', 0) < 0  # Changed from > 0
                        h_supported_specific = is_significant and direction_correct

                        if h_supported_specific: significant_results_count +=1
                        summary_rows.append({
                            'model_type': model_type, 'prediction_window': window,
                            'slope': res_data.get('slope'), 'r_squared': res_data.get('r_squared'),
                            'p_value': p_val, 'n_samples': res_data.get('n_samples'),
                            'significant': is_significant, 'direction_correct': direction_correct, 'hypothesis_supported': h_supported_specific
                        })
                    else:
                        summary_rows.append({'model_type': model_type, 'prediction_window': window, 'p_value': np.nan, 'hypothesis_supported': False, 'significant': False, 'direction_correct': False, 'n_samples':0, 'slope': np.nan, 'r_squared': np.nan})

        if summary_rows:
            summary_df = pl.DataFrame(summary_rows)
            summary_df.write_csv(os.path.join(self.results_dir, f"{self.file_prefix}{csv_filename_suffix}"))

            h_overall_supported = significant_results_count > 0
            print(f"  {hypothesis_name}: {'SUPPORTED' if h_overall_supported else 'NOT SUPPORTED'} ({significant_results_count}/{total_tests_count} tests meeting criteria).")

            fname_part = hypothesis_name.lower().replace(": ", "_").replace(" ", "_").replace("-", "_").replace("?","")
            h_test_df = pl.DataFrame({
                'hypothesis': [hypothesis_name], 'result': [h_overall_supported],
                'significant_tests': [significant_results_count], 'total_tests': [total_tests_count]
            })
            h_test_df.write_csv(os.path.join(self.results_dir, f"{self.file_prefix}_{fname_part}_test.csv"))

            if plot_function: plot_function()
        else:
            print(f"No summary rows generated for {hypothesis_name}. This might indicate an issue.")
            fname_part = hypothesis_name.lower().replace(": ", "_").replace(" ", "_").replace("-", "_").replace("?","")
            h_test_df = pl.DataFrame({
                'hypothesis': [hypothesis_name], 'result': [False],
                'significant_tests': [0], 'total_tests': [0]
            })
            h_test_df.write_csv(os.path.join(self.results_dir, f"{self.file_prefix}_{fname_part}_test.csv"))

    def _plot_prediction_relationships(self):
        try:
            for model_type, window_results in self.prediction_results.items():
                for window, result in window_results.items():
                    if result and result.get('n_samples', 0) > 0:  # Plot if we have data, regardless of significance
                        fig, ax = plt.subplots(figsize=(10, 6))
                        ax.scatter(result['X_data'], result['y_data'], alpha=0.5, label=f"n={result['n_samples']}")
                        x_line_min = np.min(result['X_data']) if len(result['X_data']) > 0 else 0
                        x_line_max = np.max(result['X_data']) if len(result['X_data']) > 0 else 0
                        if x_line_min == x_line_max:
                            x_line = np.array([x_line_min - 0.1, x_line_max + 0.1])
                        else:
                            x_line = np.linspace(x_line_min, x_line_max, 100)

                        y_line = result['slope'] * x_line + result['intercept']
                        ax.plot(x_line, y_line, color='red', label=f"y = {result['slope']:.3f}x + {result['intercept']:.3f}\nR²={result['r_squared']:.3f}, p={result['p_value']:.3f}")
                        ax.set_title(f'{model_type.upper()} Pre-event Innovations vs. {window}-Day Future Returns')
                        ax.set_xlabel('Avg Pre-Event Volatility Innovation'); ax.set_ylabel(f'{window}-Day Future Return')
                        ax.legend(); ax.grid(True, alpha=0.3)
                        plt.savefig(os.path.join(self.results_dir, f"{self.file_prefix}_h2_1_{model_type}_{window}d.png"), dpi=150, bbox_inches='tight')
                        plt.close(fig)
        except Exception as e: print(f"Error in _plot_prediction_relationships: {e}"); traceback.print_exc()

    def _plot_persistence_relationships(self):
        try:
            for model_type, result in self.persistence_results.items():
                if result and result.get('n_samples', 0) > 0:  # Plot if we have data
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.scatter(result['X_data'], result['y_data'], alpha=0.5, label=f"n={result['n_samples']}")
                    x_line_min = np.min(result['X_data']) if len(result['X_data']) > 0 else 0
                    x_line_max = np.max(result['X_data']) if len(result['X_data']) > 0 else 0
                    if x_line_min == x_line_max:
                        x_line = np.array([x_line_min -0.1, x_line_max + 0.1])
                    else:
                        x_line = np.linspace(x_line_min, x_line_max, 100)

                    y_line = result['slope'] * x_line + result['intercept']
                    ax.plot(x_line, y_line, color='red', label=f"y = {result['slope']:.3f}x + {result['intercept']:.3f}\nR²={result['r_squared']:.3f}, p={result['p_value']:.3f}")
                    ax.set_title(f'{model_type.upper()} Log Volatility Persistence vs. Post-Event Avg Return')
                    ax.set_xlabel('Log Volatility Persistence'); ax.set_ylabel('Avg Post-Event Return')
                    ax.legend(); ax.grid(True, alpha=0.3)
                    plt.savefig(os.path.join(self.results_dir, f"{self.file_prefix}_h2_2_{model_type}_persistence.png"), dpi=150, bbox_inches='tight')
                    plt.close(fig)
        except Exception as e: print(f"Error in _plot_persistence_relationships: {e}"); traceback.print_exc()

    def _plot_asymmetry_results(self):
        try:
            result = self.asymmetry_results
            if result and result.get('total_observations', 0) > 0:  # Plot if we have data
                fig, ax = plt.subplots(figsize=(10, 6))
                labels = ['Events with Most Negative Returns', 'Events with Most Positive Returns']
                mean_diffs = [result['mean_vol_diff_neg_returns'], result['mean_vol_diff_pos_returns']]

                data_neg_clean = [x for x in result['data_neg'] if pd.notna(x)]
                data_pos_clean = [x for x in result['data_pos'] if pd.notna(x)]

                errors = [np.std(data_neg_clean)/np.sqrt(len(data_neg_clean)) if len(data_neg_clean) > 0 else 0,
                          np.std(data_pos_clean)/np.sqrt(len(data_pos_clean)) if len(data_pos_clean) > 0 else 0]

                ax.bar(labels, mean_diffs, yerr=errors, capsize=5, color=['salmon', 'skyblue'], alpha=0.7)
                ax.set_ylabel('Mean (Vol GJR - Vol GARCH)')
                ax.set_title('Asymmetric Volatility Response (GJR vs GARCH) - Quartile Comparison')
                ax.grid(True, axis='y', linestyle=':', alpha=0.5)
                stats_text = (f"Diff of Means: {result['diff_of_means']:.4f}\n"
                              f"t-stat: {result['t_statistic']:.2f}, p-value: {result['p_value']:.3f}\n"
                              f"Total obs: {result['total_observations']}")
                ax.text(0.95, 0.95, stats_text, transform=ax.transAxes, ha='right', va='top', bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.7))
                plt.savefig(os.path.join(self.results_dir, f"{self.file_prefix}_h2_3_asymmetry_bar.png"), dpi=150, bbox_inches='tight')
                plt.close(fig)

                fig_box, ax_box = plt.subplots(figsize=(10,6))
                if len(data_neg_clean)>0 and len(data_pos_clean)>0:
                    ax_box.boxplot([data_neg_clean, data_pos_clean], labels=labels, notch=True, patch_artist=True,
                                boxprops=dict(facecolor='lightgray', color='black'),
                                medianprops=dict(color='black'))
                ax_box.set_ylabel('Vol GJR - Vol GARCH')
                ax_box.set_title('Distribution of Volatility Difference (GJR - GARCH) by Return Quartile')
                ax_box.grid(True, axis='y', linestyle=':', alpha=0.5)
                ax_box.text(0.95, 0.95, stats_text, transform=ax_box.transAxes, ha='right', va='top', bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.7))
                plt.savefig(os.path.join(self.results_dir, f"{self.file_prefix}_h2_3_asymmetry_boxplot.png"), dpi=150, bbox_inches='tight')
                plt.close(fig_box)

        except Exception as e: print(f"Error in _plot_asymmetry_results: {e}"); traceback.print_exc()

    def generate_summary_report(self):
        print("\n--- Generating Hypothesis 2 Overall Summary Report ---")
        h2_sub_files_info = {
            "H2.1: Pre-event volatility innovations predict returns":
                f"{self.file_prefix}_h2.1_pre_event_volatility_innovations_predict_returns_test.csv",
            "H2.2: Post-event volatility persistence extends elevated returns":
                f"{self.file_prefix}_h2.2_post_event_volatility_persistence_extends_elevated_returns_test.csv",
            "H2.3: Asymmetric volatility response correlates with price adjustment":
                f"{self.file_prefix}_h2.3_asymmetric_volatility_response_correlates_with_price_adjustment_test.csv"
        }
        sub_hyp_results = []
        all_files_exist_and_valid = True

        for full_hyp_name, expected_filename_base in h2_sub_files_info.items():
            file_path = os.path.join(self.results_dir, expected_filename_base)
            sub_h_short_name = full_hyp_name.split(':')[0]

            if os.path.exists(file_path):
                try:
                    df = pl.read_csv(file_path)
                    if df.is_empty():
                        print(f"  Warning: Result file for {sub_h_short_name} is empty: {file_path}")
                        sub_hyp_results.append({
                            'sub_hypothesis_short': sub_h_short_name,
                            'sub_hypothesis_full': full_hyp_name,
                            'result': False,
                            'details': "0/0 tests (empty result file)"
                        })
                        continue

                    details_str = f"{df.get_column('significant_tests')[0]}/{df.get_column('total_tests')[0]} tests" \
                                  if 'significant_tests' in df.columns and 'total_tests' in df.columns and df.get_column('total_tests')[0] is not None \
                                  else (f"p={df.get_column('p_value')[0]:.3f}" if 'p_value' in df.columns and df.get_column('p_value')[0] is not None else "Details N/A")

                    sub_hyp_results.append({
                        'sub_hypothesis_short': sub_h_short_name,
                        'sub_hypothesis_full': df.get_column('hypothesis')[0],
                        'result': df.get_column('result')[0],
                        'details': details_str
                    })
                except Exception as e:
                    print(f"  Error reading or processing file for {sub_h_short_name}: {file_path}. Error: {e}")
                    all_files_exist_and_valid = False; break
            else:
                print(f"  CRITICAL: Missing result file for {sub_h_short_name}: {file_path}")
                all_files_exist_and_valid = False; break

        if not all_files_exist_and_valid :
            print("Cannot generate H2 summary report due to missing or unreadable sub-hypothesis results files.")
            return None
        if not sub_hyp_results:
            print("No sub-hypothesis results were processed to generate a summary (all result files might have been empty or unreadable).")
            return None

        summary_df = pl.DataFrame(sub_hyp_results)
        num_supported_subs = sum(item['result'] for item in sub_hyp_results)
        overall_h2_supported = num_supported_subs > 0

        overall_summary_df = pl.DataFrame({
            'hypothesis': ['H2: GARCH-estimated volatility innovations proxy for impact uncertainty'],
            'result': [overall_h2_supported],
            'supported_sub_hypotheses': [num_supported_subs],
            'total_sub_hypotheses': [len(sub_hyp_results)]
        })

        summary_df.write_csv(os.path.join(self.results_dir, f"{self.file_prefix}_h2_sub_hypotheses_summary.csv"))
        overall_summary_df.write_csv(os.path.join(self.results_dir, f"{self.file_prefix}_h2_overall_summary.csv"))

        if num_supported_subs == len(sub_hyp_results) and num_supported_subs > 0 : overall_status_text = "SUPPORTED"
        elif num_supported_subs > 0 : overall_status_text = "PARTIALLY SUPPORTED"
        else: overall_status_text = "NOT SUPPORTED"

        print(f"\nHypothesis 2 Overall: {overall_status_text}")
        print(f"  Supported {num_supported_subs}/{len(sub_hyp_results)} sub-hypotheses.")
        self._plot_hypothesis_summary(summary_df, overall_h2_supported, num_supported_subs)
        return overall_summary_df

    def _plot_hypothesis_summary(self, summary_df: pl.DataFrame, overall_result_bool: bool, num_supported_subs: int):
        try:
            summary_pd = summary_df.to_pandas()
            fig, ax = plt.subplots(figsize=(12, max(6, len(summary_pd) * 1.5)))

            y_pos = np.arange(len(summary_pd))
            colors = ['mediumseagreen' if r else 'lightcoral' for r in summary_pd['result']]

            bars = ax.barh(y_pos, [1] * len(summary_pd), color=colors, alpha=0.8, edgecolor='black', height=0.6)

            for i, (bar, row) in enumerate(zip(bars, summary_pd.itertuples())):
                result_text = "SUPPORTED" if row.result else "NOT SUPPORTED"
                ax.text(0.02, i, f"{row.sub_hypothesis_short}: {row.sub_hypothesis_full.split(': ')[1]}",
                        ha='left', va='center', color='black', fontsize=9, fontweight='bold')
                ax.text(0.98, i, f"{result_text}\n({row.details})",
                        ha='right', va='center', color='black', fontsize=8,
                        bbox=dict(boxstyle='round,pad=0.3', fc=colors[i], alpha=0.5, ec='none'))

            ax.set_yticks([])
            ax.set_xticks([])
            for spine in ax.spines.values(): spine.set_visible(False)

            if num_supported_subs == len(summary_pd) and num_supported_subs > 0: overall_text_status = "SUPPORTED"
            elif num_supported_subs > 0 : overall_text_status = "PARTIALLY SUPPORTED"
            else: overall_text_status = "NOT SUPPORTED"

            fig_title = (f"Hypothesis 2 Summary: GARCH Volatility Innovations as Impact Uncertainty Proxy\n"
                         f"Overall: {overall_text_status} ({num_supported_subs}/{len(summary_pd)} sub-hypotheses supported)")
            ax.set_title(fig_title, fontsize=12, fontweight='bold')

            plt.tight_layout(rect=[0, 0, 1, 0.96])
            plt.savefig(os.path.join(self.results_dir, f"{self.file_prefix}_h2_overall_summary_plot.png"), dpi=150, bbox_inches='tight')
            plt.close(fig)
        except Exception as e: print(f"Error in _plot_hypothesis_summary: {e}"); traceback.print_exc()

def run_fda_analysis():
    print("\n=== Starting FDA Approval Event Analysis for Hypothesis 2 ===")
    if not os.path.exists(FDA_EVENT_FILE):
        print(f"Error: FDA event file not found: {FDA_EVENT_FILE}"); return False
    if any(not os.path.exists(f) for f in STOCK_FILES):
        print(f"Error: One or more stock files not found."); return False
    os.makedirs(FDA_RESULTS_DIR, exist_ok=True)

    try:
        data_loader = EventDataLoader(
            event_path=FDA_EVENT_FILE, stock_paths=STOCK_FILES, window_days=WINDOW_DAYS,
            event_date_col=FDA_EVENT_DATE_COL, ticker_col=FDA_TICKER_COL
        )
        analyzer = EventAnalysis(data_loader, EventFeatureEngineer())
        analyzer.data = analyzer.load_and_prepare_data(run_feature_engineering=False)
        if analyzer.data is None or analyzer.data.is_empty():
            print("Error: FDA data loading failed for H2."); return False
        print(f"FDA data loaded. Shape: {analyzer.data.shape}")

        h2_analyzer_fda = Hypothesis2Analyzer(analyzer, FDA_RESULTS_DIR, FDA_FILE_PREFIX)
        success_h2 = h2_analyzer_fda.analyze_volatility_innovations()

        if success_h2:
            h2_analyzer_fda.generate_summary_report()
            print(f"--- FDA H2 Analysis Finished (Results in '{FDA_RESULTS_DIR}') ---")
            return True
        else: print("FDA H2 volatility innovations analysis failed."); return False
    except Exception as e: print(f"FDA H2 Analysis Error: {e}"); traceback.print_exc(); return False

def run_earnings_analysis():
    print("\n=== Starting Earnings Announcement Event Analysis for Hypothesis 2 ===")
    if not os.path.exists(EARNINGS_EVENT_FILE):
        print(f"Error: Earnings event file not found: {EARNINGS_EVENT_FILE}"); return False
    if any(not os.path.exists(f) for f in STOCK_FILES):
        print(f"Error: One or more stock files not found."); return False
    os.makedirs(EARNINGS_RESULTS_DIR, exist_ok=True)

    try:
        data_loader = EventDataLoader(
            event_path=EARNINGS_EVENT_FILE, stock_paths=STOCK_FILES, window_days=WINDOW_DAYS,
            event_date_col=EARNINGS_EVENT_DATE_COL, ticker_col=EARNINGS_TICKER_COL
        )
        analyzer = EventAnalysis(data_loader, EventFeatureEngineer())
        analyzer.data = analyzer.load_and_prepare_data(run_feature_engineering=False)
        if analyzer.data is None or analyzer.data.is_empty():
            print("Error: Earnings data loading failed for H2."); return False
        print(f"Earnings data loaded. Shape: {analyzer.data.shape}")

        h2_analyzer_earnings = Hypothesis2Analyzer(analyzer, EARNINGS_RESULTS_DIR, EARNINGS_FILE_PREFIX)
        success_h2 = h2_analyzer_earnings.analyze_volatility_innovations()

        if success_h2:
            h2_analyzer_earnings.generate_summary_report()
            print(f"--- Earnings H2 Analysis Finished (Results in '{EARNINGS_RESULTS_DIR}') ---")
            return True
        else: print("Earnings H2 volatility innovations analysis failed."); return False
    except Exception as e: print(f"Earnings H2 Analysis Error: {e}"); traceback.print_exc(); return False

def compare_results():
    print("\n=== Comparing FDA and Earnings Results for Hypothesis 2 ===")
    comparison_dir = "results/hypothesis2/comparison/"
    os.makedirs(comparison_dir, exist_ok=True)

    fda_overall_file = os.path.join(FDA_RESULTS_DIR, f"{FDA_FILE_PREFIX}_h2_overall_summary.csv")
    earn_overall_file = os.path.join(EARNINGS_RESULTS_DIR, f"{EARNINGS_FILE_PREFIX}_h2_overall_summary.csv")
    fda_subs_file = os.path.join(FDA_RESULTS_DIR, f"{FDA_FILE_PREFIX}_h2_sub_hypotheses_summary.csv")
    earn_subs_file = os.path.join(EARNINGS_RESULTS_DIR, f"{EARNINGS_FILE_PREFIX}_h2_sub_hypotheses_summary.csv")

    missing = [fp for fp in [fda_overall_file, earn_overall_file, fda_subs_file, earn_subs_file] if not os.path.exists(fp)]
    if missing: print(f"Error: Missing H2 comparison files: {missing}"); return False

    try:
        fda_overall = pl.read_csv(fda_overall_file)
        earn_overall = pl.read_csv(earn_overall_file)
        fda_subs = pl.read_csv(fda_subs_file)
        earn_subs = pl.read_csv(earn_subs_file)

        overall_comp_df = pl.DataFrame({
            "Event Type": ["FDA Approvals", "Earnings Announcements"],
            "H2 Overall Supported": [fda_overall['result'][0], earn_overall['result'][0]],
            "Supported Sub-Hypotheses": [f"{fda_overall['supported_sub_hypotheses'][0]}/{fda_overall['total_sub_hypotheses'][0]}",
                                         f"{earn_overall['supported_sub_hypotheses'][0]}/{earn_overall['total_sub_hypotheses'][0]}"]
        })
        overall_comp_df.write_csv(os.path.join(comparison_dir, "h2_overall_comparison.csv"))
        print("\nH2 Overall Comparison:")
        print(overall_comp_df)

        fda_subs = fda_subs.rename({'result': 'fda_result', 'details': 'fda_details'})
        earn_subs = earn_subs.rename({'result': 'earn_result', 'details': 'earn_details'})
        subs_comp_df = fda_subs.join(earn_subs, on=['sub_hypothesis_short', 'sub_hypothesis_full'], how='outer')
        subs_comp_df.write_csv(os.path.join(comparison_dir, "h2_sub_hypotheses_comparison.csv"))
        print("\nH2 Sub-Hypotheses Comparison:")
        print(subs_comp_df.select(['sub_hypothesis_short', 'fda_result', 'earn_result', 'fda_details', 'earn_details']))

        plot_df_pd = subs_comp_df.to_pandas()
        fig, ax = plt.subplots(figsize=(12, 7))
        num_subs = len(plot_df_pd)
        x = np.arange(num_subs)
        width = 0.35

        rects1 = ax.bar(x - width/2, plot_df_pd['fda_result'].astype(int), width, label='FDA Approvals', color='deepskyblue', alpha=0.8)
        rects2 = ax.bar(x + width/2, plot_df_pd['earn_result'].astype(int), width, label='Earnings Announcements', color='salmon', alpha=0.8)

        ax.set_ylabel('Hypothesis Supported (1=Yes, 0=No)')
        ax.set_title('Hypothesis 2 Sub-Component Support: FDA vs Earnings')
        ax.set_xticks(x)
        ax.set_xticklabels(plot_df_pd['sub_hypothesis_short'], rotation=0, ha='center')
        ax.set_yticks([0, 1])
        ax.set_yticklabels(['No', 'Yes'])
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=2)
        ax.grid(True, axis='y', linestyle=':', alpha=0.6)

        plt.tight_layout()
        plt.savefig(os.path.join(comparison_dir, "h2_sub_hypotheses_comparison_plot.png"), dpi=150)
        plt.close(fig)
        print(f"Saved H2 sub-hypotheses comparison plot to {comparison_dir}")
        return True

    except Exception as e: print(f"Error comparing H2 results: {e}"); traceback.print_exc(); return False

def main():
    fda_success_h2 = run_fda_analysis()
    earnings_success_h2 = run_earnings_analysis()

    if fda_success_h2 and earnings_success_h2:
        if compare_results(): print("\n=== All H2 analyses and comparisons completed successfully ===")
        else: print("\n=== H2 Analyses completed, but comparison failed ===")
    elif fda_success_h2: print("\n=== Only FDA H2 analysis completed successfully ===")
    elif earnings_success_h2: print("\n=== Only Earnings H2 analysis completed successfully ===")
    else: print("\n=== Both H2 analyses failed ===")

if __name__ == "__main__":
    main()
