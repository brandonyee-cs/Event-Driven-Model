# runhypothesis2.py

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
    from src.models import GARCHModel, GJRGARCHModel, ThreePhaseVolatilityModel # ThreePhase not used by H2 directly
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
WINDOW_DAYS = 60 # For EventDataLoader
ANALYSIS_WINDOW = (-30, 30) # General analysis window for H2 (days_to_event)
PRE_EVENT_WINDOW = (-15, -1)  # Window for pre-event volatility innovations (H2.1)
POST_EVENT_WINDOW = (1, 15)   # Window for post-event volatility persistence & returns (H2.2)

# GARCH model parameters for H2
MODEL_TYPES = ['garch', 'gjr']  # Test both types for asymmetric response (H2.3)
GARCH_PARAMS = { # Initial parameters for GARCH fitting
    'garch': {'omega': 1e-6, 'alpha': 0.1, 'beta': 0.85},
    'gjr': {'omega': 1e-6, 'alpha': 0.08, 'beta': 0.85, 'gamma': 0.05}
}

# Define prediction windows for return forecasting (H2.1)
PREDICTION_WINDOWS = [1, 3, 5]  # Days ahead to predict returns

class Hypothesis2Analyzer:
    def __init__(self, analyzer: EventAnalysis, results_dir: str, file_prefix: str):
        self.analyzer = analyzer
        self.results_dir = results_dir
        self.file_prefix = file_prefix
        self.return_col = 'ret' # Column name for returns in EventAnalysis.data
        self.analysis_window = ANALYSIS_WINDOW
        self.pre_event_window = PRE_EVENT_WINDOW
        self.post_event_window = POST_EVENT_WINDOW
        self.prediction_windows = PREDICTION_WINDOWS
        self.model_types = MODEL_TYPES
        self.garch_params = GARCH_PARAMS
        
        self.innovations_data = {} # Stores results from _fit_garch_models by event_id
        self.prediction_results = {} # Stores H2.1 regression results
        self.asymmetry_results = {} # Stores H2.3 t-test results
        self.persistence_results = {} # Stores H2.2 regression results
        
        os.makedirs(results_dir, exist_ok=True)
    
    def _fit_garch_models(self, event_id: str, event_data: pl.DataFrame) -> Optional[Dict[str, Any]]:
        # event_data is for a single event_id, sorted by days_to_event
        event_days_series = event_data.get_column('days_to_event')
        event_returns_series = event_data.get_column(self.return_col)
        
        if len(event_returns_series) < 30: # Min data for GARCH
            # print(f"Skipping event {event_id}: Insufficient data ({len(event_returns_series)} points).")
            return None
        
        result = {
            'event_id': event_id,
            'ticker': event_data.get_column('ticker').head(1).item(),
            'days_to_event': event_days_series.to_list(), # Full window days for this event
            'returns': event_returns_series.to_numpy(), # Full window returns
            'models': {}, 'innovations': {}, 'volatility': {}, 'persistence': {}
        }
        
        for model_type in self.model_types:
            try:
                params = self.garch_params[model_type]
                if model_type == 'garch': model = GARCHModel(**params)
                else: model = GJRGARCHModel(**params)
                
                model.fit(event_returns_series) 
                
                volatility_sqrt_h_t = model.conditional_volatility()
                innovations_raw = model.volatility_innovations() 
                
                result['models'][model_type] = model
                result['volatility'][model_type] = volatility_sqrt_h_t 
                
                # Align innovations: innovations_raw is length T-1, for days_to_event[1:]
                # We want innovations_aligned to be length T, with NaN for the first day.
                innovations_aligned = np.full(len(event_days_series), np.nan)
                if len(innovations_raw) > 0 and len(event_days_series) > 1:
                    innovations_aligned[1:len(innovations_raw)+1] = innovations_raw[:len(innovations_aligned)-1]
                result['innovations'][model_type] = innovations_aligned

                days_map = {day: i for i, day in enumerate(event_days_series.to_list())}
                
                pre_event_indices = [days_map[d] for d in range(self.pre_event_window[0], self.pre_event_window[1] + 1) if d in days_map and days_map[d] < len(volatility_sqrt_h_t)]
                post_event_indices = [days_map[d] for d in range(self.post_event_window[0], self.post_event_window[1] + 1) if d in days_map and days_map[d] < len(volatility_sqrt_h_t)]

                if pre_event_indices and post_event_indices and len(volatility_sqrt_h_t) > 0:
                    pre_vol_mean = np.nanmean(volatility_sqrt_h_t[pre_event_indices]) if pre_event_indices else np.nan
                    post_vol_mean = np.nanmean(volatility_sqrt_h_t[post_event_indices]) if post_event_indices else np.nan
                    result['persistence'][model_type] = post_vol_mean / pre_vol_mean if pd.notna(pre_vol_mean) and pd.notna(post_vol_mean) and pre_vol_mean > 1e-9 else np.nan
                else:
                    result['persistence'][model_type] = np.nan
                
            except Exception as e:
                # print(f"Error fitting {model_type} model for event {event_id}: {e}")
                # traceback.print_exc() 
                result['models'][model_type] = None
                result['volatility'][model_type] = np.array([])
                result['innovations'][model_type] = np.array([]) # Ensure it's an array
                result['persistence'][model_type] = np.nan
        return result
    
    def _calculate_future_returns(self, event_data: pl.DataFrame) -> Dict[int, np.ndarray]:
        future_returns_dict = {}
        if 'prc' not in event_data.columns:
            # print(f"Warning: 'prc' column not in event_data. Cannot calculate future returns accurately.")
            for window in self.prediction_windows:
                future_returns_dict[window] = np.full(event_data.height, np.nan)
            return future_returns_dict

        # Ensure sorted by date for correct shift calculations
        # The input event_data should already be sorted by 'days_to_event' which implies date order within an event
        for window in self.prediction_windows:
            # Calculate P_{t+k} / P_t - 1
            # The shift(-window) gets P_{t+k} if data is sorted ascending by date.
            future_ret_series = (
                event_data.get_column('prc').shift(-window) / event_data.get_column('prc')
            ) - 1
            future_returns_dict[window] = future_ret_series.fill_null(np.nan).to_numpy()
        return future_returns_dict
    
    def analyze_volatility_innovations(self):
        print("\n--- Analyzing Volatility Innovations (Hypothesis 2) ---")
        if self.analyzer.data is None: print("Error: No data available for analysis."); return False
        
        # Use the data loaded by EventDataLoader, which respects WINDOW_DAYS (e.g. -60 to +60)
        # This ensures GARCH has enough history.
        analysis_data = self.analyzer.data.sort(['event_id', 'days_to_event']) 
        
        if analysis_data.is_empty(): print(f"Error: No data found for H2 analysis."); return False
        
        event_ids = analysis_data.get_column('event_id').unique().to_list()
        print(f"Total events available: {len(event_ids)}")
        
        # Sample for computational efficiency as per original logic
        if len(event_ids) > 100:
            np.random.seed(42) # For reproducibility
            sample_event_ids = np.random.choice(event_ids, size=100, replace=False).tolist()
            print(f"Sampling {len(sample_event_ids)} events for detailed H2 analysis.")
        else:
            sample_event_ids = event_ids
        
        all_events_processed_data = []
        for i, event_id in enumerate(sample_event_ids):
            # if (i+1) % 10 == 0 : print(f"  Processing event {i+1}/{len(sample_event_ids)}: {event_id}")
            event_specific_data = analysis_data.filter(pl.col('event_id') == event_id)
            if event_specific_data.is_empty(): continue

            event_garch_results = self._fit_garch_models(event_id, event_specific_data)
            if event_garch_results is None: continue
            
            future_returns_for_event = self._calculate_future_returns(event_specific_data)
            event_garch_results['future_returns'] = future_returns_for_event
            all_events_processed_data.append(event_garch_results)
        
        self.innovations_data = all_events_processed_data 
        print(f"Successfully analyzed GARCH models for {len(self.innovations_data)} events.")
        
        if not self.innovations_data: print("No GARCH innovations data to analyze H2 sub-hypotheses."); return False

        self._test_prediction_power() # H2.1
        self._test_volatility_persistence() # H2.2
        self._test_asymmetric_response() # H2.3
        return True
    
    def _test_prediction_power(self): # H2.1
        print("\n--- H2.1: Pre-event volatility innovations predict subsequent returns ---")
        if not self.innovations_data: print("Error: No innovations data for H2.1."); return
        
        regression_results_h21 = {}
        for model_type in self.model_types:
            regression_results_h21[model_type] = {}
            for pred_window in self.prediction_windows:
                # print(f"  Testing {model_type} innovations for {pred_window}-day future returns...")
                X_innov_data, y_fut_ret_data = [], []
                
                for event_res in self.innovations_data:
                    if model_type not in event_res['models'] or event_res['models'][model_type] is None: continue
                    if 'future_returns' not in event_res or pred_window not in event_res['future_returns']: continue

                    days_list = event_res['days_to_event']
                    innovations_arr = event_res['innovations'][model_type] # Aligned with days_list (NaN for day 0)
                    future_ret_arr = event_res['future_returns'][pred_window] # Aligned with days_list

                    # Find indices for PRE_EVENT_WINDOW in days_list
                    pre_event_day_indices = [i for i, day_val in enumerate(days_list) 
                                             if self.pre_event_window[0] <= day_val <= self.pre_event_window[1]]
                    
                    if not pre_event_day_indices: continue

                    # Average pre-event innovations (using aligned innovations_arr)
                    avg_pre_event_innovation = np.nanmean(innovations_arr[pre_event_day_indices])
                    if np.isnan(avg_pre_event_innovation): continue

                    # Find index for event day (day 0) to get subsequent return
                    try:
                        event_day_0_index = days_list.index(0)
                    except ValueError:
                        continue # Event day 0 not in this event's data range
                    
                    # Subsequent return is future_ret_arr at event_day_0_index
                    if event_day_0_index < len(future_ret_arr):
                        subsequent_return = future_ret_arr[event_day_0_index]
                        if np.isnan(subsequent_return): continue
                        
                        X_innov_data.append(avg_pre_event_innovation)
                        y_fut_ret_data.append(subsequent_return)
                    
                if len(X_innov_data) > 10:
                    X_np = np.array(X_innov_data)
                    y_np = np.array(y_fut_ret_data)
                    
                    # Check for variance in X
                    if np.var(X_np) < 1e-10:
                        # print(f"    Warning: Insufficient variance in X data for {model_type}, {pred_window}-day returns.")
                        slope, intercept, r_value, p_value, std_err = 0, np.mean(y_np), 0, 1.0, 0
                    else:
                        slope, intercept, r_value, p_value, std_err = stats.linregress(X_np, y_np)
                    
                    regression_results_h21[model_type][pred_window] = {
                        'slope': slope, 'intercept': intercept, 'r_squared': r_value**2,
                        'p_value': p_value, 'std_err': std_err, 'n_samples': len(X_np),
                        'X_data': X_np, 'y_data': y_np # For plotting
                    }
                    # print(f"    {model_type}, {pred_window}-day: slope={slope:.4f}, R²={r_value**2:.4f}, p={p_value:.4f}, n={len(X_np)}")
                else:
                    # print(f"    Insufficient data for {model_type}, {pred_window}-day regression (n={len(X_innov_data)}).")
                    regression_results_h21[model_type][pred_window] = None
        
        self.prediction_results = regression_results_h21
        self._save_and_summarize_h2_sub_results(
            results_dict=self.prediction_results,
            csv_filename_suffix="_h2_1_prediction_summary.csv",
            plot_function=self._plot_prediction_relationships,
            hypothesis_name="H2.1: Pre-event volatility innovations predict returns"
        )

    def _test_volatility_persistence(self): # H2.2
        print("\n--- H2.2: Post-event volatility persistence extends elevated expected returns ---")
        if not self.innovations_data: print("Error: No innovations data for H2.2."); return

        persistence_analysis_results = {}
        for model_type in self.model_types:
            # print(f"  Testing {model_type} volatility persistence effect...")
            X_persistence_data, y_post_ret_data = [], []

            for event_res in self.innovations_data:
                if model_type not in event_res['persistence'] or pd.isna(event_res['persistence'][model_type]): continue
                
                persistence_ratio = event_res['persistence'][model_type]
                days_list = event_res['days_to_event']
                returns_arr = event_res['returns'] # Original returns for the event

                # Average return in POST_EVENT_WINDOW
                post_event_day_indices = [i for i, day_val in enumerate(days_list)
                                          if self.post_event_window[0] <= day_val <= self.post_event_window[1]
                                          and i < len(returns_arr)] # Ensure index is valid
                
                if not post_event_day_indices: continue
                
                avg_post_event_return = np.nanmean(returns_arr[post_event_day_indices])
                if np.isnan(avg_post_event_return): continue

                X_persistence_data.append(persistence_ratio)
                y_post_ret_data.append(avg_post_event_return)

            if len(X_persistence_data) > 10:
                X_np = np.array(X_persistence_data)
                y_np = np.array(y_post_ret_data)
                if np.var(X_np) < 1e-10:
                    # print(f"    Warning: Insufficient variance in X persistence data for {model_type}.")
                    slope, intercept, r_value, p_value, std_err = 0, np.mean(y_np), 0, 1.0, 0
                else:
                    slope, intercept, r_value, p_value, std_err = stats.linregress(X_np, y_np)
                
                persistence_analysis_results[model_type] = {
                    'slope': slope, 'intercept': intercept, 'r_squared': r_value**2,
                    'p_value': p_value, 'std_err': std_err, 'n_samples': len(X_np),
                    'X_data': X_np, 'y_data': y_np # For plotting
                }
                # print(f"    {model_type}: slope={slope:.4f}, R²={r_value**2:.4f}, p={p_value:.4f}, n={len(X_np)}")
            else:
                # print(f"    Insufficient data for {model_type} persistence regression (n={len(X_persistence_data)}).")
                persistence_analysis_results[model_type] = None
        
        self.persistence_results = persistence_analysis_results
        self._save_and_summarize_h2_sub_results(
            results_dict=self.persistence_results, # This will be dict by model_type, not model_type & window
            csv_filename_suffix="_h2_2_persistence_summary.csv",
            plot_function=self._plot_persistence_relationships,
            hypothesis_name="H2.2: Post-event volatility persistence extends elevated returns",
            is_single_key_per_model=True # Special handling for summary
        )

    def _test_asymmetric_response(self): # H2.3
        print("\n--- H2.3: Asymmetric volatility response correlates with asymmetric price adjustment ---")
        if not self.innovations_data: print("Error: No innovations data for H2.3."); return
        if 'garch' not in self.model_types or 'gjr' not in self.model_types:
            print("Error: Both GARCH and GJR-GARCH models required for H2.3."); return

        # H2.3: Test if GJR model's gamma (asymmetry param) is significant OR
        # Test if GJR-GARCH provides better fit for negative vs positive shock days,
        # or if vol innovations differ. Paper: "Asymmetric volatility response ... correlates with asymmetric price adjustment"
        # This implies looking at the GJR model's gamma or comparing GJR vs GARCH on neg/pos return days.
        
        # Let's test if the GJR model's gamma parameter is significantly different from zero, averaged across events.
        # This requires access to fitted GJR model parameters.
        # Alternatively, compare vol levels on event day for GJR vs GARCH based on prior day's return sign.

        # Approach: Compare GJR vol vs GARCH vol on event day (d=0), conditional on event day return (proxy for news sign).
        # If event day return is negative, does GJR predict higher vol than GARCH, compared to positive return days?
        
        diff_gjr_garch_vol_neg_ret = []
        diff_gjr_garch_vol_pos_ret = []

        for event_res in self.innovations_data:
            if event_res['models'].get('garch') is None or event_res['models'].get('gjr') is None: continue
            
            days_list = event_res['days_to_event']
            returns_arr = event_res['returns']
            vol_garch = event_res['volatility']['garch']
            vol_gjr = event_res['volatility']['gjr']

            try: event_day_0_index = days_list.index(0)
            except ValueError: continue # Event day 0 not found

            if event_day_0_index >= len(returns_arr) or \
               event_day_0_index >= len(vol_garch) or \
               event_day_0_index >= len(vol_gjr):
                continue # Data alignment issue

            event_day_return = returns_arr[event_day_0_index]
            vol_diff_on_event_day = vol_gjr[event_day_0_index] - vol_garch[event_day_0_index]

            if np.isnan(event_day_return) or np.isnan(vol_diff_on_event_day): continue

            if event_day_return < 0:
                diff_gjr_garch_vol_neg_ret.append(vol_diff_on_event_day)
            elif event_day_return > 0: # Exclude zero return events
                diff_gjr_garch_vol_pos_ret.append(vol_diff_on_event_day)
        
        if len(diff_gjr_garch_vol_neg_ret) > 5 and len(diff_gjr_garch_vol_pos_ret) > 5:
            mean_diff_neg = np.mean(diff_gjr_garch_vol_neg_ret)
            mean_diff_pos = np.mean(diff_gjr_garch_vol_pos_ret)
            
            # T-test for difference in means of vol_diff
            t_stat, p_value = stats.ttest_ind(diff_gjr_garch_vol_neg_ret, diff_gjr_garch_vol_pos_ret, 
                                              equal_var=False, nan_policy='omit')
            
            self.asymmetry_results = {
                'mean_vol_diff_neg_returns': mean_diff_neg,
                'mean_vol_diff_pos_returns': mean_diff_pos,
                'diff_of_means': mean_diff_neg - mean_diff_pos,
                't_statistic': t_stat, 'p_value': p_value,
                'n_neg_returns': len(diff_gjr_garch_vol_neg_ret),
                'n_pos_returns': len(diff_gjr_garch_vol_pos_ret),
                'data_neg': diff_gjr_garch_vol_neg_ret, # For plotting
                'data_pos': diff_gjr_garch_vol_pos_ret  # For plotting
            }
            # print(f"  Asymmetry test: Mean GJR-GARCH vol diff (Neg Ret): {mean_diff_neg:.6f} (n={len(diff_gjr_garch_vol_neg_ret)})")
            # print(f"  Mean GJR-GARCH vol diff (Pos Ret): {mean_diff_pos:.6f} (n={len(diff_gjr_garch_vol_pos_ret)})")
            # print(f"  Diff of means: {mean_diff_neg - mean_diff_pos:.6f}, t-stat={t_stat:.4f}, p-value={p_value:.4f}")
        else:
            # print("  Insufficient data for asymmetry t-test.")
            self.asymmetry_results = None

        self._save_and_summarize_h2_sub_results(
            results_dict=self.asymmetry_results, # This is a single dict, not by model_type
            csv_filename_suffix="_h2_3_asymmetry_summary.csv",
            plot_function=self._plot_asymmetry_results,
            hypothesis_name="H2.3: Asymmetric volatility response correlates with price adjustment",
            is_single_result_dict=True # Special handling for summary
        )

    def _save_and_summarize_h2_sub_results(self, results_dict, csv_filename_suffix, plot_function, hypothesis_name, is_single_key_per_model=False, is_single_result_dict=False):
        summary_rows = []
        significant_results_count = 0
        total_tests_count = 0

        if is_single_result_dict: # For H2.3 asymmetry test
            if results_dict and pd.notna(results_dict.get('p_value')):
                total_tests_count = 1
                p_val = results_dict['p_value']
                # H2.3 support: p < 0.1 AND GJR vol diff is greater for neg returns than pos returns
                # (i.e., results_dict['diff_of_means'] > 0, assuming vol_diff = vol_gjr - vol_garch)
                is_significant = p_val < 0.1
                # Directional check: diff_of_means > 0 implies GJR shows more vol for neg returns than for pos returns, relative to GARCH
                direction_correct = results_dict.get('diff_of_means', 0) > 0 
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
                summary_rows.append({'test_case': 'Asymmetric Response (GJR vs GARCH)', 'p_value': np.nan, 'hypothesis_supported': False})
        
        elif is_single_key_per_model: # For H2.2 persistence test
            for model_type, res_data in results_dict.items():
                if res_data and pd.notna(res_data.get('p_value')):
                    total_tests_count += 1
                    p_val = res_data['p_value']
                    # H2.2 support: p < 0.1 AND slope > 0 (higher persistence -> higher post-event returns)
                    is_significant = p_val < 0.1
                    direction_correct = res_data.get('slope', 0) > 0
                    h_supported_specific = is_significant and direction_correct

                    if h_supported_specific: significant_results_count += 1
                    summary_rows.append({
                        'model_type': model_type,
                        'slope': res_data.get('slope'), 'r_squared': res_data.get('r_squared'),
                        'p_value': p_val, 'n_samples': res_data.get('n_samples'),
                        'significant': is_significant, 'direction_correct': direction_correct, 'hypothesis_supported': h_supported_specific
                    })
                else:
                    summary_rows.append({'model_type': model_type, 'p_value': np.nan, 'hypothesis_supported': False})
        else: # For H2.1 prediction test
            for model_type, window_results in results_dict.items():
                for window, res_data in window_results.items():
                    if res_data and pd.notna(res_data.get('p_value')):
                        total_tests_count +=1
                        p_val = res_data['p_value']
                        # H2.1 support: p < 0.1 (direction of slope can vary)
                        is_significant = p_val < 0.1 
                        # For H2.1, the paper doesn't strictly define direction, just "predict". So significance is key.
                        # However, typically expect positive relationship: higher uncertainty -> higher expected future return.
                        direction_correct = res_data.get('slope', 0) > 0 # Let's assume positive prediction for "support"
                        h_supported_specific = is_significant # and direction_correct (optional based on interpretation)

                        if h_supported_specific: significant_results_count +=1
                        summary_rows.append({
                            'model_type': model_type, 'prediction_window': window,
                            'slope': res_data.get('slope'), 'r_squared': res_data.get('r_squared'),
                            'p_value': p_val, 'n_samples': res_data.get('n_samples'),
                            'significant': is_significant, 'direction_correct': direction_correct, 'hypothesis_supported': h_supported_specific
                        })
                    else:
                        summary_rows.append({'model_type': model_type, 'prediction_window': window, 'p_value': np.nan, 'hypothesis_supported': False})

        if summary_rows:
            summary_df = pl.DataFrame(summary_rows)
            summary_df.write_csv(os.path.join(self.results_dir, f"{self.file_prefix}{csv_filename_suffix}"))
            
            # Check overall hypothesis support for this sub-part
            h_overall_supported = significant_results_count > 0
            print(f"  {hypothesis_name}: {'SUPPORTED' if h_overall_supported else 'NOT SUPPORTED'} ({significant_results_count}/{total_tests_count} significant tests meeting criteria).")
            
            # Save specific hypothesis test result
            h_test_df = pl.DataFrame({
                'hypothesis': [hypothesis_name], 'result': [h_overall_supported],
                'significant_tests': [significant_results_count], 'total_tests': [total_tests_count]
            })
            h_test_df.write_csv(os.path.join(self.results_dir, f"{self.file_prefix}_{hypothesis_name.split(':')[0].lower().replace(' ', '_')}_test.csv"))

            if plot_function: plot_function() # Call plotting function
        else:
            print(f"No valid results to save or summarize for {hypothesis_name}.")


    def _plot_prediction_relationships(self): # For H2.1
        # print("  Generating H2.1 plots for significant relationships...")
        try:
            for model_type, window_results in self.prediction_results.items():
                for window, result in window_results.items():
                    if result and result.get('significant') and result.get('direction_correct', True): # Plot if significant (and optionally direction_correct)
                        fig, ax = plt.subplots(figsize=(10, 6))
                        ax.scatter(result['X_data'], result['y_data'], alpha=0.5, label=f"n={result['n_samples']}")
                        x_line = np.linspace(np.min(result['X_data']), np.max(result['X_data']), 100)
                        y_line = result['slope'] * x_line + result['intercept']
                        ax.plot(x_line, y_line, color='red', label=f"y = {result['slope']:.3f}x + {result['intercept']:.3f}\nR²={result['r_squared']:.3f}, p={result['p_value']:.3f}")
                        ax.set_title(f'{model_type.upper()} Pre-event Innovations vs. {window}-Day Future Returns')
                        ax.set_xlabel('Avg Pre-Event Volatility Innovation'); ax.set_ylabel(f'{window}-Day Future Return')
                        ax.legend(); ax.grid(True, alpha=0.3)
                        plt.savefig(os.path.join(self.results_dir, f"{self.file_prefix}_h2_1_{model_type}_{window}d.png"), dpi=150, bbox_inches='tight')
                        plt.close(fig)
        except Exception as e: print(f"Error in _plot_prediction_relationships: {e}"); traceback.print_exc()

    def _plot_persistence_relationships(self): # For H2.2
        # print("  Generating H2.2 plots for significant relationships...")
        try:
            for model_type, result in self.persistence_results.items():
                if result and result.get('significant') and result.get('direction_correct', True):
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.scatter(result['X_data'], result['y_data'], alpha=0.5, label=f"n={result['n_samples']}")
                    x_line = np.linspace(np.min(result['X_data']), np.max(result['X_data']), 100)
                    y_line = result['slope'] * x_line + result['intercept']
                    ax.plot(x_line, y_line, color='red', label=f"y = {result['slope']:.3f}x + {result['intercept']:.3f}\nR²={result['r_squared']:.3f}, p={result['p_value']:.3f}")
                    ax.set_title(f'{model_type.upper()} Volatility Persistence vs. Post-Event Avg Return')
                    ax.set_xlabel('Volatility Persistence (Post/Pre Ratio)'); ax.set_ylabel('Avg Post-Event Return')
                    ax.legend(); ax.grid(True, alpha=0.3)
                    plt.savefig(os.path.join(self.results_dir, f"{self.file_prefix}_h2_2_{model_type}_persistence.png"), dpi=150, bbox_inches='tight')
                    plt.close(fig)
        except Exception as e: print(f"Error in _plot_persistence_relationships: {e}"); traceback.print_exc()

    def _plot_asymmetry_results(self): # For H2.3
        # print("  Generating H2.3 plots...")
        try:
            result = self.asymmetry_results
            if result and result.get('significant') and result.get('direction_correct', True) :
                fig, ax = plt.subplots(figsize=(10, 6))
                labels = ['Events with Negative Returns', 'Events with Positive Returns']
                mean_diffs = [result['mean_vol_diff_neg_returns'], result['mean_vol_diff_pos_returns']]
                errors = [np.std(result['data_neg'])/np.sqrt(result['n_neg_returns']) if result['n_neg_returns'] > 0 else 0, 
                          np.std(result['data_pos'])/np.sqrt(result['n_pos_returns']) if result['n_pos_returns'] > 0 else 0]
                
                ax.bar(labels, mean_diffs, yerr=errors, capsize=5, color=['salmon', 'skyblue'], alpha=0.7)
                ax.set_ylabel('Mean (Vol GJR - Vol GARCH) on Event Day')
                ax.set_title('Asymmetric Volatility Response (GJR vs GARCH)')
                ax.grid(True, axis='y', linestyle=':', alpha=0.5)
                stats_text = (f"Diff of Means: {result['diff_of_means']:.4f}\n"
                              f"t-stat: {result['t_statistic']:.2f}, p-value: {result['p_value']:.3f}")
                ax.text(0.95, 0.95, stats_text, transform=ax.transAxes, ha='right', va='top', bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.7))
                plt.savefig(os.path.join(self.results_dir, f"{self.file_prefix}_h2_3_asymmetry_bar.png"), dpi=150, bbox_inches='tight')
                plt.close(fig)

                # Boxplot
                fig_box, ax_box = plt.subplots(figsize=(10,6))
                ax_box.boxplot([result['data_neg'], result['data_pos']], labels=labels, notch=True, patch_artist=True,
                               boxprops=dict(facecolor='lightgray', color='black'),
                               medianprops=dict(color='black'))
                ax_box.set_ylabel('Vol GJR - Vol GARCH on Event Day')
                ax_box.set_title('Distribution of Volatility Difference (GJR - GARCH) by Event Return Sign')
                ax_box.grid(True, axis='y', linestyle=':', alpha=0.5)
                ax_box.text(0.95, 0.95, stats_text, transform=ax_box.transAxes, ha='right', va='top', bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.7))
                plt.savefig(os.path.join(self.results_dir, f"{self.file_prefix}_h2_3_asymmetry_boxplot.png"), dpi=150, bbox_inches='tight')
                plt.close(fig_box)

        except Exception as e: print(f"Error in _plot_asymmetry_results: {e}"); traceback.print_exc()
    
    def generate_summary_report(self):
        print("\n--- Generating Hypothesis 2 Overall Summary Report ---")
        h2_sub_files_info = {
            "H2.1": "_h2.1_pre-event_volatility_innovations_predict_returns_test.csv",
            "H2.2": "_h2.2_post-event_volatility_persistence_extends_elevated_returns_test.csv",
            "H2.3": "_h2.3_asymmetric_volatility_response_correlates_with_price_adjustment_test.csv"
        }
        sub_hyp_results = []
        all_files_exist = True
        for sub_h_name, file_suffix in h2_sub_files_info.items():
            file_path = os.path.join(self.results_dir, f"{self.file_prefix}{file_suffix}")
            if os.path.exists(file_path):
                df = pl.read_csv(file_path)
                sub_hyp_results.append({
                    'sub_hypothesis_short': sub_h_name,
                    'sub_hypothesis_full': df['hypothesis'][0],
                    'result': df['result'][0],
                    'details': f"{df['significant_tests'][0]}/{df['total_tests'][0]} significant tests" if 'significant_tests' in df.columns else f"p={df.get('p_value', [np.nan])[0]:.3f}"
                })
            else:
                print(f"  Warning: Missing result file for {sub_h_name}: {file_path}")
                all_files_exist = False; break
        
        if not all_files_exist or not sub_hyp_results:
            print("Cannot generate H2 summary report due to missing sub-hypothesis results.")
            return None

        summary_df = pl.DataFrame(sub_hyp_results)
        overall_h2_supported = all(item['result'] for item in sub_hyp_results) # True if all sub-hypotheses are true
        num_supported_subs = sum(item['result'] for item in sub_hyp_results)
        
        overall_summary_df = pl.DataFrame({
            'hypothesis': ['H2: GARCH-estimated volatility innovations proxy for impact uncertainty'],
            'result': [overall_h2_supported],
            'supported_sub_hypotheses': [num_supported_subs],
            'total_sub_hypotheses': [len(sub_hyp_results)]
        })
        
        summary_df.write_csv(os.path.join(self.results_dir, f"{self.file_prefix}_h2_sub_hypotheses_summary.csv"))
        overall_summary_df.write_csv(os.path.join(self.results_dir, f"{self.file_prefix}_h2_overall_summary.csv"))
        
        print(f"\nHypothesis 2 Overall: {'SUPPORTED' if overall_h2_supported else ('PARTIALLY SUPPORTED' if num_supported_subs > 0 else 'NOT SUPPORTED')}")
        print(f"  Supported {num_supported_subs}/{len(sub_hyp_results)} sub-hypotheses.")
        self._plot_hypothesis_summary(summary_df, overall_h2_supported, num_supported_subs)
        return overall_summary_df

    def _plot_hypothesis_summary(self, summary_df: pl.DataFrame, overall_result: bool, num_supported_subs: int):
        # print("  Generating H2 overall summary plot...")
        try:
            summary_pd = summary_df.to_pandas()
            fig, ax = plt.subplots(figsize=(12, max(6, len(summary_pd) * 1.5))) # Adjust height
            
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
            
            overall_text_status = "SUPPORTED" if overall_result else ("PARTIALLY SUPPORTED" if num_supported_subs > 0 else "NOT SUPPORTED")
            fig_title = (f"Hypothesis 2 Summary: GARCH Volatility Innovations as Impact Uncertainty Proxy\n"
                         f"Overall: {overall_text_status} ({num_supported_subs}/{len(summary_pd)} sub-hypotheses supported)")
            ax.set_title(fig_title, fontsize=12, fontweight='bold')
            
            plt.tight_layout(rect=[0, 0, 1, 0.96]) # Adjust for title
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
    print(f"FDA results will be saved to: {os.path.abspath(FDA_RESULTS_DIR)}")

    try:
        data_loader = EventDataLoader(
            event_path=FDA_EVENT_FILE, stock_paths=STOCK_FILES, window_days=WINDOW_DAYS,
            event_date_col=FDA_EVENT_DATE_COL, ticker_col=FDA_TICKER_COL
        )
        analyzer = EventAnalysis(data_loader, EventFeatureEngineer()) # FE not critical for H2
        print("Loading FDA data for H2...")
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
    print(f"Earnings results will be saved to: {os.path.abspath(EARNINGS_RESULTS_DIR)}")

    try:
        data_loader = EventDataLoader(
            event_path=EARNINGS_EVENT_FILE, stock_paths=STOCK_FILES, window_days=WINDOW_DAYS,
            event_date_col=EARNINGS_EVENT_DATE_COL, ticker_col=EARNINGS_TICKER_COL
        )
        analyzer = EventAnalysis(data_loader, EventFeatureEngineer()) # FE not critical for H2
        print("Loading Earnings data for H2...")
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
        print("\nH2 Overall Comparison:\n", overall_comp_df)

        # Merge sub-hypothesis results for easier comparison
        fda_subs = fda_subs.rename({'result': 'fda_result', 'details': 'fda_details'})
        earn_subs = earn_subs.rename({'result': 'earn_result', 'details': 'earn_details'})
        subs_comp_df = fda_subs.join(earn_subs, on=['sub_hypothesis_short', 'sub_hypothesis_full'], how='outer')
        subs_comp_df.write_csv(os.path.join(comparison_dir, "h2_sub_hypotheses_comparison.csv"))
        print("\nH2 Sub-Hypotheses Comparison:\n", subs_comp_df.select(['sub_hypothesis_short', 'fda_result', 'earn_result', 'fda_details', 'earn_details']))

        # Plotting
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
