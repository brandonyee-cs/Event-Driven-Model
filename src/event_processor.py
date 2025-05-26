"""
Event Processor Module - Unified Volatility Model Implementation

Implements the unified volatility model from "Modeling Equilibrium Asset Pricing 
Around Events with Heterogeneous Beliefs, Dynamic Volatility, and a Two-Risk 
Uncertainty Framework" by Brandon Yee.

Key components:
- GJR-GARCH(1,1) baseline volatility estimation
- Three-phase event-specific volatility adjustments (phi functions)
- Impact uncertainty measurement
- Bias parameter evolution
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime, timedelta
import warnings
from scipy import stats
from arch import arch_model
from dataclasses import dataclass, field 
from .config import Config

warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=UserWarning) 

@dataclass
class VolatilityParameters:
    """
    Parameters for the unified volatility model.
    Stores both GJR-GARCH parameters and event-specific adjustments.
    """
    omega: float
    alpha: float
    beta: float
    gamma: float
    k1: float = 1.3
    k2: float = 1.5
    delta_t1: float = 5.0
    delta_t2: float = 3.0
    delta_t3: float = 10.0
    delta: int = 5
    conditional_volatility_series: Optional[pd.Series] = field(default=None, repr=False)
    log_likelihood: Optional[float] = None

class EventProcessor:
    """
    Main class for processing event-driven asset pricing data.
    Implements the unified volatility model: sigma_e(t) = sqrt(h_t) * (1 + phi(t))
    """
    
    def __init__(self, config: Config):
        self.config = config
        self.config.validate_parameters()
        self.volatility_params: Dict[str, VolatilityParameters] = {}
        
    def _estimate_gjr_garch(self, symbol: str, symbol_price_data_full: pd.DataFrame) -> bool:
        # print(f"  Attempting GARCH for {symbol}: input data rows {len(symbol_price_data_full)}") # DEBUG
        if 'returns' not in symbol_price_data_full.columns:
            # print(f"    GARCH {symbol}: 'returns' column missing in input. Columns: {symbol_price_data_full.columns}") # DEBUG
            return False
        
        # Ensure returns are numeric, coerce errors, then dropna
        # This handles cases where 'returns' might be object type or contain non-numeric strings
        numeric_returns = pd.to_numeric(symbol_price_data_full['returns'], errors='coerce')
        returns_decimal = numeric_returns.dropna()
        
        if returns_decimal.empty and not numeric_returns.empty: # All were NaN after coerce or originally all NaN
            # print(f"    GARCH {symbol}: All 'returns' are NaN or became NaN after to_numeric.") # DEBUG
            return False

        # print(f"    GARCH {symbol}: Found {len(returns_decimal)} non-NaN numeric returns. Required: {self.config.lookback_days}") # DEBUG

        if len(returns_decimal) < self.config.lookback_days:
            # if len(returns_decimal) > 0 : # only print if there was *some* data
                 # print(f"    GARCH {symbol}: Not enough non-NaN numeric returns ({len(returns_decimal)}). Skipping GARCH.") # DEBUG
            return False 
        
        scaled_returns = returns_decimal * 100 

        try:
            model = arch_model(scaled_returns, vol='GARCH', p=1, q=1, o=1, dist='normal', rescale=False)
            res = model.fit(disp='off', show_warning=False, update_freq=0, options={'maxiter': 200}) 
            
            omega, alpha, beta, gamma = res.params.get('omega', self.config.gjr_omega), \
                                        res.params.get('alpha[1]', self.config.gjr_alpha), \
                                        res.params.get('beta[1]', self.config.gjr_beta), \
                                        res.params.get('gamma[1]', self.config.gjr_gamma)
            loglik = res.loglikelihood if hasattr(res, 'loglikelihood') else None

            if alpha + beta + gamma/2 >= 0.9999: 
                # print(f"    GARCH {symbol}: Non-stationary. Using defaults.") # DEBUG
                omega, alpha, beta, gamma = self.config.gjr_omega, self.config.gjr_alpha, \
                                            self.config.gjr_beta, self.config.gjr_gamma
                loglik = None 
        except Exception as e:
            # print(f"    GARCH {symbol}: Estimation failed: {e}. Using defaults.") # DEBUG
            omega, alpha, beta, gamma = self.config.gjr_omega, self.config.gjr_alpha, \
                                        self.config.gjr_beta, self.config.gjr_gamma
            loglik = None
        
        params_obj = VolatilityParameters(
            omega=omega, alpha=alpha, beta=beta, gamma=gamma,
            k1=self.config.event_k1, k2=self.config.event_k2, delta=self.config.event_delta,
            delta_t1=self.config.event_delta_t1, delta_t2=self.config.event_delta_t2, 
            delta_t3=self.config.event_delta_t3, log_likelihood=loglik
        )
        self.volatility_params[symbol] = params_obj
        # print(f"    GARCH {symbol}: Successfully stored params.") # DEBUG
        return True

    def _get_baseline_volatility(self, symbol: str, event_window_price_data: pd.DataFrame) -> pd.Series:
        params = self.volatility_params.get(symbol)
        if params is None: 
            std_dev = event_window_price_data['returns'].std()
            return pd.Series(std_dev if pd.notna(std_dev) and std_dev > 0 else 0.01, index=event_window_price_data.index)

        returns = event_window_price_data['returns'].fillna(0).values 
        n_obs = len(returns)
        if n_obs == 0: return pd.Series(dtype=float)

        h_t = np.zeros(n_obs) 
        uncond_var_denom = (1 - params.alpha - params.beta - params.gamma/2)
        if uncond_var_denom > 1e-7: 
            h_t[0] = params.omega / uncond_var_denom
        else: 
            lookback_for_empirical_var = min(10, n_obs) if n_obs > 0 else 0
            if lookback_for_empirical_var > 0 :
                initial_var = np.var(returns[:lookback_for_empirical_var])
                h_t[0] = initial_var if initial_var > 1e-7 else (params.omega / 1e-7) 
            else: 
                 h_t[0] = params.omega / 1e-7 
        if h_t[0] <= 1e-9: h_t[0] = 1e-9 

        for t_idx in range(1, n_obs):
            ret_prev_sq = returns[t_idx-1]**2 
            indicator = 1 if returns[t_idx-1] < 0 else 0
            h_t[t_idx] = (params.omega + params.alpha * ret_prev_sq + 
                         params.beta * h_t[t_idx-1] + params.gamma * indicator * ret_prev_sq)
            if h_t[t_idx] <= 1e-9: h_t[t_idx] = 1e-9 
        
        baseline_vol_series = pd.Series(np.sqrt(h_t), index=event_window_price_data.index)
        return baseline_vol_series.fillna(method='bfill').fillna(method='ffill').fillna(0.01)

    def _calculate_phi_function(self, t_rel: int, t_event_day_zero: int, params: VolatilityParameters) -> float:
        if t_rel <= t_event_day_zero:
            phi = (params.k1 - 1) * np.exp(-((t_rel - t_event_day_zero)**2) / (2 * params.delta_t1**2))
        elif t_event_day_zero < t_rel <= t_event_day_zero + params.delta:
            phi = (params.k2 - 1) * (1 - np.exp(-(t_rel - t_event_day_zero) / params.delta_t2))
        else:
            phi = (params.k2 - 1) * np.exp(-(t_rel - t_event_day_zero - params.delta) / params.delta_t3)
        return phi
    
    def _calculate_unified_volatility(self, t_rel: int, t_event_day_zero: int, 
                                    baseline_vol_td: float, params: VolatilityParameters) -> float:
        phi = self._calculate_phi_function(t_rel, t_event_day_zero, params)
        safe_baseline_vol = max(baseline_vol_td, 1e-8) 
        sigma_e = safe_baseline_vol * (1 + phi)
        return sigma_e
    
    def _calculate_bias_parameter(self, t_rel: int, t_event_day_zero: int, 
                                sigma_e: float, baseline_vol_td: float,
                                investor_type: str = 'average') -> float: 
        b_0 = self.config.bias_baseline_b0
        kappa = self.config.bias_kappa_sensitivity
        
        if t_event_day_zero < t_rel <= t_event_day_zero + self.config.event_delta: 
            safe_baseline_vol = max(baseline_vol_td, 1e-8) 
            psi_t = sigma_e / safe_baseline_vol
            k2_minus_1 = self.config.event_k2 - 1
            if abs(k2_minus_1) < 1e-8: k2_minus_1 = 1e-8 
            bias = b_0 * (1 + kappa * (psi_t - 1) / k2_minus_1)
        else:
            bias = b_0 
        return bias

    def _calculate_impact_uncertainty(self, symbol_params: VolatilityParameters, 
                                     event_window_returns_decimal: pd.Series, 
                                     event_window_baseline_variance: pd.Series) -> pd.Series:
        n_obs = len(event_window_returns_decimal)
        if n_obs == 0: return pd.Series(dtype=float)

        E_h_t = np.zeros(n_obs) 
        actual_h_t = event_window_baseline_variance.values
        if n_obs > 0: E_h_t[0] = actual_h_t[0] 

        for t_idx in range(1, n_obs):
            ret_prev_sq = event_window_returns_decimal.iloc[t_idx-1]**2
            indicator = 1 if event_window_returns_decimal.iloc[t_idx-1] < 0 else 0
            h_prev = actual_h_t[t_idx-1] 
            E_h_t[t_idx] = (symbol_params.omega + symbol_params.alpha * ret_prev_sq +
                           symbol_params.beta * h_prev + symbol_params.gamma * indicator * ret_prev_sq)
            if E_h_t[t_idx] <= 1e-9: E_h_t[t_idx] = 1e-9
        
        vol_innovation = np.sqrt(actual_h_t) - np.sqrt(E_h_t)
        return pd.Series(vol_innovation, index=event_window_returns_decimal.index).fillna(0)

    def _process_single_event(self, symbol_prices_full_history: pd.DataFrame, 
                            event_row: pd.Series, symbol: str) -> Optional[pd.DataFrame]:
        event_date = pd.to_datetime(event_row['event_date'])
        params = self.volatility_params.get(symbol) 
        if params is None: 
            # print(f"    DEBUG _process_single_event: No GARCH params for {symbol} @ {event_date}.") # DEBUG
            return None

        local_garch_lookback = 60 
        analysis_start_date = event_date - timedelta(days=self.config.event_window_pre)
        garch_calc_start_date = analysis_start_date - timedelta(days=local_garch_lookback)
        analysis_end_date = event_date + timedelta(days=self.config.event_window_post)

        event_garch_window_df = symbol_prices_full_history[
            (symbol_prices_full_history['date'] >= garch_calc_start_date) &
            (symbol_prices_full_history['date'] <= analysis_end_date)
        ].copy()

        if len(event_garch_window_df) < self.config.min_required_days : 
            # print(f"    DEBUG _process_single_event: Insufficient data for event GARCH window {symbol} @ {event_date} ({len(event_garch_window_df)}).") # DEBUG
            return None
            
        baseline_vols_for_garch_window = self._get_baseline_volatility(symbol, event_garch_window_df)
        
        event_garch_window_df['days_to_event'] = (event_garch_window_df['date'] - event_date).dt.days
        analysis_mask = (event_garch_window_df['days_to_event'] >= -self.config.event_window_pre) & \
                       (event_garch_window_df['days_to_event'] <= self.config.event_window_post)
        event_data_final = event_garch_window_df[analysis_mask].copy()

        if event_data_final.empty: 
            # print(f"    DEBUG _process_single_event: No data in final analysis window for {symbol} @ {event_date}.") # DEBUG
            return None

        event_data_final['baseline_volatility'] = baseline_vols_for_garch_window.reindex(event_data_final.index).fillna(method='bfill').fillna(method='ffill').fillna(0.01)
        baseline_variance_final_window = event_data_final['baseline_volatility']**2
        event_data_final['volatility_innovation'] = self._calculate_impact_uncertainty(
            params, event_data_final['returns'], baseline_variance_final_window
        )
        
        unified_vols, biases_list, phi_values = [], [], []
        t_event_day_zero = 0 
        
        for _, row_data in event_data_final.iterrows():
            t_rel = row_data['days_to_event']
            baseline_vol_td = row_data['baseline_volatility']
            phi = self._calculate_phi_function(t_rel, t_event_day_zero, params)
            phi_values.append(phi)
            unified_vol = self._calculate_unified_volatility(t_rel, t_event_day_zero, baseline_vol_td, params)
            unified_vols.append(unified_vol)
            bias = self._calculate_bias_parameter(t_rel, t_event_day_zero, unified_vol, baseline_vol_td)
            biases_list.append(bias)
        
        event_data_final['phi_adjustment'] = phi_values
        event_data_final['unified_volatility'] = unified_vols
        event_data_final['bias_parameter'] = biases_list
        event_data_final['expected_return'] = event_data_final['returns'].fillna(0) + event_data_final['bias_parameter']
        event_data_final['event_date_orig'] = event_date # Keep original event_date for grouping/identification
        event_data_final['event_type'] = event_row.get('event_type', 'unknown')
        # 'symbol' column should already exist from symbol_prices_full_history
        
        event_data_final['garch_alpha'] = params.alpha
        event_data_final['garch_beta'] = params.beta
        event_data_final['garch_gamma'] = params.gamma
        event_data_final = event_data_final.rename(columns={'event_date_orig':'event_date'})
        return event_data_final

    def process_events(self, price_data: pd.DataFrame, event_data: pd.DataFrame) -> pd.DataFrame:
        results_list = []
        required_price_cols = ['symbol', 'date', 'price', 'returns']
        required_event_cols = ['symbol', 'event_date']

        if not all(col in price_data.columns for col in required_price_cols):
            raise ValueError(f"Price data must contain columns: {required_price_cols}, found {price_data.columns}")
        if not all(col in event_data.columns for col in required_event_cols):
            raise ValueError(f"Event data must contain columns: {required_event_cols}, found {event_data.columns}")

        price_data['date'] = pd.to_datetime(price_data['date'])
        event_data['event_date'] = pd.to_datetime(event_data['event_date'])

        unique_symbols = event_data['symbol'].unique()
        print(f"Processing {len(unique_symbols)} symbols with events...")
        
        symbols_with_garch_params_count = 0
        for i, symbol_val in enumerate(unique_symbols):
            if i > 0 and i % 100 == 0: 
                print(f"  Estimated GARCH for {i}/{len(unique_symbols)} symbols. Successful so far: {symbols_with_garch_params_count}")
            
            symbol_prices_full = price_data[price_data['symbol'] == symbol_val].copy()
            
            if symbol_prices_full.empty:
                # print(f"  DEBUG process_events: No price data for symbol {symbol_val} in GARCH loop.") # DEBUG
                continue
            if 'returns' not in symbol_prices_full.columns:
                # print(f"  DEBUG process_events: 'returns' column missing for symbol {symbol_val} in GARCH loop.") # DEBUG
                continue
                
            symbol_prices_full = symbol_prices_full.sort_values('date').reset_index(drop=True)
            # It's crucial that _winsorize_returns handles potential non-numeric data in 'returns' gracefully
            symbol_prices_full = self._winsorize_returns(symbol_prices_full) 
            
            if self._estimate_gjr_garch(symbol_val, symbol_prices_full):
                symbols_with_garch_params_count +=1
            # else: # DEBUG
                # print(f"  DEBUG process_events: GARCH estimation FAILED for symbol {symbol_val}.") # DEBUG

        print(f"Finished GARCH estimations. GARCH parameters stored for {symbols_with_garch_params_count}/{len(unique_symbols)} symbols.")
        if symbols_with_garch_params_count == 0 and len(unique_symbols) > 0:
            print("WARNING: No symbols had successful GARCH parameter estimation. Event processing will likely yield no results.")
            # Depending on how critical GARCH is, you might return early:
            # return pd.DataFrame() 

        processed_event_count = 0
        for i, symbol_val in enumerate(unique_symbols):
            if i > 0 and i % 100 == 0 :
                print(f"  Processing events for symbol {i+1}/{len(unique_symbols)}. Events processed so far: {processed_event_count}")
            if symbol_val not in self.volatility_params: 
                # print(f"  DEBUG process_events: Skipping events for {symbol_val}, GARCH params not found in dict.") # DEBUG
                continue

            symbol_event_rows = event_data[event_data['symbol'] == symbol_val]
            symbol_prices_full_hist = price_data[price_data['symbol'] == symbol_val].copy() # Fresh copy for each symbol
            if symbol_prices_full_hist.empty: 
                # print(f"  DEBUG process_events: No price data for {symbol_val} in event processing loop (second phase).") # DEBUG
                continue
            
            symbol_prices_full_hist = symbol_prices_full_hist.sort_values('date').reset_index(drop=True)
            symbol_prices_full_hist = self._winsorize_returns(symbol_prices_full_hist) # Winsorize again for safety

            for _, event_row_data in symbol_event_rows.iterrows():
                single_event_df = self._process_single_event(
                    symbol_prices_full_hist, event_row_data, symbol_val
                )
                if single_event_df is not None and not single_event_df.empty:
                    results_list.append(single_event_df)
                    processed_event_count +=1
        
        if not results_list:
            print("Warning: No events successfully processed after iterating all symbols and events.")
            return pd.DataFrame()
            
        final_results_df = pd.concat(results_list, ignore_index=True)
        
        if final_results_df.empty:
             print("Warning: Concatenated results DataFrame is empty (though results_list was not). This is unexpected.")
             return pd.DataFrame()

        # The nunique might be on a column that was renamed/dropped if events were processed
        # Ensure 'event_date' is the final column name after _process_single_event
        if 'event_date' in final_results_df.columns:
            print(f"Successfully processed data for {final_results_df['event_date'].nunique()} unique event dates across symbols.")
        else:
            print("Successfully processed data, but 'event_date' column not found for summary stats.")
        print(f"Total event-day observations generated: {len(final_results_df)}")
        return final_results_df

    def _winsorize_returns(self, df: pd.DataFrame) -> pd.DataFrame:
        if 'returns' in df.columns and not df['returns'].empty:
            # Make a copy to avoid SettingWithCopyWarning if df is a slice
            df_copy = df.copy()
            df_copy['returns'] = pd.to_numeric(df_copy['returns'], errors='coerce')
            
            # Operate on the subset of rows where 'returns' is not NaN after coercion
            not_na_mask = df_copy['returns'].notna()
            if not_na_mask.any():
                df_clean_returns = df_copy.loc[not_na_mask, 'returns']
                
                if len(df_clean_returns) > 1 : 
                    lower_b = df_clean_returns.quantile(self.config.return_winsorization_lower)
                    upper_b = df_clean_returns.quantile(self.config.return_winsorization_upper)
                    lower_b = max(lower_b, -self.config.max_daily_return)
                    upper_b = min(upper_b, self.config.max_daily_return)
                    df_copy.loc[not_na_mask, 'returns'] = df_clean_returns.clip(lower=lower_b, upper=upper_b)
            return df_copy
        return df
