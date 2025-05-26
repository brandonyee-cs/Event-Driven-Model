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
from dataclasses import dataclass, field # Added field for default_factory
from .config import Config

warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=UserWarning) # For arch_model show_warning=False

@dataclass
class VolatilityParameters:
    """
    Parameters for the unified volatility model.
    Stores both GJR-GARCH parameters and event-specific adjustments.
    """
    # GJR-GARCH(1,1) parameters
    omega: float
    alpha: float
    beta: float
    gamma: float
    
    # Event-specific parameters (from paper Section 3.1)
    k1: float = 1.3
    k2: float = 1.5
    delta_t1: float = 5.0
    delta_t2: float = 3.0
    delta_t3: float = 10.0
    delta: int = 5
    
    # Store for full series conditional volatility (decimal, pd.Series with date index)
    conditional_volatility_series: Optional[pd.Series] = field(default=None, repr=False)
    log_likelihood: Optional[float] = None
    # Standardized residuals are not directly stored here anymore to avoid alignment issues.
    # They can be re-derived if needed or passed differently.

class EventProcessor:
    """
    Main class for processing event-driven asset pricing data.
    Implements the unified volatility model: sigma_e(t) = sqrt(h_t) * (1 + phi(t))
    """
    
    def __init__(self, config: Config):
        """Initialize with configuration object"""
        self.config = config
        self.config.validate_parameters()
        self.volatility_params: Dict[str, VolatilityParameters] = {}
        # self.impact_uncertainty_cache: Dict[str, pd.DataFrame] = {} # Not currently used
        
    def _estimate_gjr_garch(self, symbol: str, symbol_price_data_full: pd.DataFrame) -> bool:
        """
        Estimate GJR-GARCH(1,1) parameters for baseline volatility using full history for a symbol.
        Stores parameters. The actual conditional volatilities for event windows are calculated on-the-fly.
        symbol_price_data_full should be sorted by date and already winsorized if applicable.
        """
        returns_decimal = symbol_price_data_full['returns'].dropna()
        
        if len(returns_decimal) < self.config.lookback_days:
            # print(f"  Skipping GARCH for {symbol}: not enough returns ({len(returns_decimal)} < {self.config.lookback_days})")
            return False
        
        scaled_returns = returns_decimal * 100 

        try:
            model = arch_model(scaled_returns, vol='GARCH', p=1, q=1, o=1, dist='normal', rescale=False)
            res = model.fit(disp='off', show_warning=False, update_freq=0)
            
            omega, alpha, beta, gamma = res.params.get('omega', self.config.gjr_omega), \
                                        res.params.get('alpha[1]', self.config.gjr_alpha), \
                                        res.params.get('beta[1]', self.config.gjr_beta), \
                                        res.params.get('gamma[1]', self.config.gjr_gamma)
            
            loglik = res.loglikelihood if hasattr(res, 'loglikelihood') else None

            if alpha + beta + gamma/2 >= 0.9999: # Check stationarity
                # print(f"  GARCH for {symbol} non-stationary. Using defaults.")
                omega, alpha, beta, gamma = self.config.gjr_omega, self.config.gjr_alpha, \
                                            self.config.gjr_beta, self.config.gjr_gamma
                loglik = None # Loglikelihood from non-stationary model is not reliable
        except Exception as e:
            # print(f"  GARCH estimation failed for {symbol}: {e}. Using defaults.")
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
        return True

    def _get_baseline_volatility(self, symbol: str, event_window_price_data: pd.DataFrame) -> pd.Series:
        """
        Calculate baseline GJR-GARCH volatility for the specific event window using pre-estimated parameters.
        event_window_price_data is ALREADY filtered for the event's local GARCH lookback + analysis window.
        Returns decimal volatility.
        """
        params = self.volatility_params.get(symbol)
        # This should always find params because process_events calls _estimate_gjr_garch first.
        # If it's None, it means _estimate_gjr_garch failed to store even default params, which is an issue.
        if params is None: 
            # Fallback if somehow params are missing (should ideally not happen if _estimate_gjr_garch ran)
            # print(f"  CRITICAL: No GARCH params for {symbol} in _get_baseline_volatility. Using default std.")
            std_dev = event_window_price_data['returns'].std()
            return pd.Series(std_dev if pd.notna(std_dev) and std_dev > 0 else 0.01, index=event_window_price_data.index)

        returns = event_window_price_data['returns'].fillna(0).values # Decimal returns
        n_obs = len(returns)
        if n_obs == 0:
            return pd.Series(dtype=float) # Return empty series if no returns

        h_t = np.zeros(n_obs) # Variance series
        
        uncond_var_denom = (1 - params.alpha - params.beta - params.gamma/2)
        if uncond_var_denom > 1e-7: # Check for stationarity and avoid division by zero
            h_t[0] = params.omega / uncond_var_denom
        else: # Non-stationary or near non-stationary, use empirical variance of a few initial returns
            lookback_for_empirical_var = min(10, n_obs) if n_obs > 0 else 0
            if lookback_for_empirical_var > 0 :
                initial_var = np.var(returns[:lookback_for_empirical_var])
                h_t[0] = initial_var if initial_var > 1e-7 else (params.omega / 1e-7) # Fallback if empirical is zero/too small
            else: # no returns to calculate empirical var
                 h_t[0] = params.omega / 1e-7 # default small denominator

        if h_t[0] <= 1e-9: h_t[0] = 1e-9 # Ensure positive initial variance, floor at a very small number

        for t_idx in range(1, n_obs):
            ret_prev_sq = returns[t_idx-1]**2 
            indicator = 1 if returns[t_idx-1] < 0 else 0
            
            h_t[t_idx] = (params.omega + 
                         params.alpha * ret_prev_sq + 
                         params.beta * h_t[t_idx-1] + 
                         params.gamma * indicator * ret_prev_sq)
            
            if h_t[t_idx] <= 1e-9: h_t[t_idx] = 1e-9 # Floor variance at a small positive value
        
        baseline_vol_series = pd.Series(np.sqrt(h_t), index=event_window_price_data.index)
        # Fill NaNs that might occur from sqrt(negative) if floor wasn't perfect, or other issues.
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
        # Ensure baseline_vol_td is positive for multiplication
        safe_baseline_vol = max(baseline_vol_td, 1e-8) # Use a small floor
        sigma_e = safe_baseline_vol * (1 + phi)
        return sigma_e
    
    def _calculate_bias_parameter(self, t_rel: int, t_event_day_zero: int, 
                                sigma_e: float, baseline_vol_td: float,
                                investor_type: str = 'average') -> float: # investor_type not used from config here
        # Using config directly for b0 and kappa as per original structure
        b_0 = self.config.bias_baseline_b0
        kappa = self.config.bias_kappa_sensitivity
        
        if t_event_day_zero < t_rel <= t_event_day_zero + self.config.event_delta: # post-event rising
            safe_baseline_vol = max(baseline_vol_td, 1e-8) # Avoid division by zero
            psi_t = sigma_e / safe_baseline_vol
            
            # Ensure (k2-1) is not zero
            k2_minus_1 = self.config.event_k2 - 1
            if abs(k2_minus_1) < 1e-8: k2_minus_1 = 1e-8 # Avoid division by zero

            bias = b_0 * (1 + kappa * (psi_t - 1) / k2_minus_1)
        else:
            bias = b_0 # Baseline bias outside the rising phase
        return bias

    def _calculate_impact_uncertainty(self, symbol_params: VolatilityParameters, 
                                     event_window_returns_decimal: pd.Series, 
                                     event_window_baseline_variance: pd.Series) -> pd.Series:
        """
        Calculate impact uncertainty (volatility innovations) for the event window.
        ImpactUncertainty_t = sqrt(h_t) - sqrt(E_{t-1}[h_t]) (in decimal terms)
        event_window_returns_decimal: decimal returns for the event window
        event_window_baseline_variance: h_t (variance) for the event window
        """
        n_obs = len(event_window_returns_decimal)
        if n_obs == 0:
            return pd.Series(dtype=float)

        # E_h_t: Expected variance for current period, conditional on t-1 info
        E_h_t = np.zeros(n_obs) 
        actual_h_t = event_window_baseline_variance.values

        # Initialize E_h_t[0]
        # This is E[h_0 | info at -1]. If actual_h_t[0] is h_0, then E_h_t[0] is tricky.
        # For simplicity, assume E_h_t[0] is the same as actual_h_t[0] (no surprise for the first day of window)
        # Or use unconditional variance if available and appropriate.
        if n_obs > 0:
             E_h_t[0] = actual_h_t[0] 

        for t_idx in range(1, n_obs):
            ret_prev_sq = event_window_returns_decimal.iloc[t_idx-1]**2
            indicator = 1 if event_window_returns_decimal.iloc[t_idx-1] < 0 else 0
            h_prev = actual_h_t[t_idx-1] # Actual variance from t-1

            E_h_t[t_idx] = (symbol_params.omega +
                           symbol_params.alpha * ret_prev_sq +
                           symbol_params.beta * h_prev +
                           symbol_params.gamma * indicator * ret_prev_sq)
            if E_h_t[t_idx] <= 1e-9: E_h_t[t_idx] = 1e-9
        
        # Volatility innovation = sqrt(actual_h_t) - sqrt(expected_h_t)
        vol_innovation = np.sqrt(actual_h_t) - np.sqrt(E_h_t)
        
        return pd.Series(vol_innovation, index=event_window_returns_decimal.index).fillna(0)

    def _process_single_event(self, symbol_prices_full_history: pd.DataFrame, 
                            event_row: pd.Series, symbol: str) -> Optional[pd.DataFrame]:
        event_date = pd.to_datetime(event_row['event_date'])
        params = self.volatility_params.get(symbol) # These are GJR-GARCH parameters
        if params is None: 
            # print(f"  Skipping event for {symbol} @ {event_date}: No GARCH parameters.")
            return None

        # Define window for local GARCH baseline calculation and analysis
        local_garch_lookback = 60 
        analysis_start_date = event_date - timedelta(days=self.config.event_window_pre)
        garch_calc_start_date = analysis_start_date - timedelta(days=local_garch_lookback)
        analysis_end_date = event_date + timedelta(days=self.config.event_window_post)

        # This data is used to calculate baseline volatility for the analysis window
        event_garch_window_df = symbol_prices_full_history[
            (symbol_prices_full_history['date'] >= garch_calc_start_date) &
            (symbol_prices_full_history['date'] <= analysis_end_date)
        ].copy()

        if len(event_garch_window_df) < self.config.min_required_days : # Check if enough data for reliable GARCH calc over the window
            # print(f"  Skipping event for {symbol} @ {event_date}: Insufficient data for event GARCH window ({len(event_garch_window_df)}).")
            return None
            
        # Calculate baseline volatilities for this specific window
        baseline_vols_for_garch_window = self._get_baseline_volatility(symbol, event_garch_window_df)
        
        # Filter down to the actual analysis window [-pre_window, +post_window]
        event_garch_window_df['days_to_event'] = (event_garch_window_df['date'] - event_date).dt.days
        analysis_mask = (event_garch_window_df['days_to_event'] >= -self.config.event_window_pre) & \
                       (event_garch_window_df['days_to_event'] <= self.config.event_window_post)
        event_data_final = event_garch_window_df[analysis_mask].copy()

        if event_data_final.empty:
            # print(f"  Skipping event for {symbol} @ {event_date}: No data in final analysis window.")
            return None

        # Assign baseline volatility (already calculated for the garch_window, now just select the analysis part)
        event_data_final['baseline_volatility'] = baseline_vols_for_garch_window.reindex(event_data_final.index).fillna(method='bfill').fillna(method='ffill').fillna(0.01)
        
        # Calculate impact uncertainty
        # Need baseline variance (h_t) for this window
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
        
        event_data_final['event_date_orig'] = event_date 
        event_data_final['event_type'] = event_row.get('event_type', 'unknown')
        # 'symbol' column should already exist from symbol_prices_full_history
        
        event_data_final['garch_alpha'] = params.alpha
        event_data_final['garch_beta'] = params.beta
        event_data_final['garch_gamma'] = params.gamma
        
        event_data_final = event_data_final.rename(columns={'event_date_orig':'event_date'})
        return event_data_final

    def process_events(self, price_data: pd.DataFrame, event_data: pd.DataFrame) -> pd.DataFrame:
        """
        Main method to process events.
        1. Estimates GARCH parameters for each symbol.
        2. For each event, calculates event-specific metrics using these parameters.
        """
        results_list = []
        required_price_cols = ['symbol', 'date', 'price', 'returns']
        required_event_cols = ['symbol', 'event_date']

        if not all(col in price_data.columns for col in required_price_cols):
            raise ValueError(f"Price data must contain columns: {required_price_cols}, found {price_data.columns}")
        if not all(col in event_data.columns for col in required_event_cols):
            raise ValueError(f"Event data must contain columns: {required_event_cols}, found {event_data.columns}")

        # Ensure date columns are datetime
        price_data['date'] = pd.to_datetime(price_data['date'])
        event_data['event_date'] = pd.to_datetime(event_data['event_date'])

        unique_symbols = event_data['symbol'].unique()
        print(f"Processing {len(unique_symbols)} symbols with events...")
        
        symbols_with_garch_params_count = 0
        # --- Step 1: Estimate GARCH parameters for all symbols ---
        for i, symbol_val in enumerate(unique_symbols):
            if i % 100 == 0 and i > 0:
                print(f"  Estimated GARCH for {i}/{len(unique_symbols)} symbols. Successful so far: {symbols_with_garch_params_count}")
            
            symbol_prices_full = price_data[price_data['symbol'] == symbol_val].copy()
            if symbol_prices_full.empty: continue

            symbol_prices_full = symbol_prices_full.sort_values('date').reset_index(drop=True)
            symbol_prices_full = self._winsorize_returns(symbol_prices_full)
            
            if self._estimate_gjr_garch(symbol_val, symbol_prices_full):
                symbols_with_garch_params_count +=1
        print(f"Finished GARCH estimations. GARCH parameters stored for {symbols_with_garch_params_count}/{len(unique_symbols)} symbols.")

        # --- Step 2: Process each event using the stored GARCH parameters ---
        processed_event_count = 0
        for i, symbol_val in enumerate(unique_symbols):
            if i % 100 == 0 and i > 0 :
                print(f"  Processing events for symbol {i+1}/{len(unique_symbols)}. Events processed: {processed_event_count}")
                
            if symbol_val not in self.volatility_params:
                continue # Skip if GARCH params (even defaults) weren't stored

            symbol_event_rows = event_data[event_data['symbol'] == symbol_val]
            symbol_prices_full_hist = price_data[price_data['symbol'] == symbol_val].copy()
            if symbol_prices_full_hist.empty: continue
            
            symbol_prices_full_hist = symbol_prices_full_hist.sort_values('date').reset_index(drop=True)
            symbol_prices_full_hist = self._winsorize_returns(symbol_prices_full_hist) # Ensure returns are winsorized

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
        
        # These are specific to Hypothesis1Tester, so should be done there.
        # final_results_df = self._add_risk_adjusted_metrics(final_results_df)
        # final_results_df = self._add_phase_indicators(final_results_df) 
        
        print(f"Successfully processed data for {final_results_df['event_date'].nunique()} unique event dates across symbols.")
        print(f"Total event-day observations generated: {len(final_results_df)}")
        return final_results_df

    def _winsorize_returns(self, df: pd.DataFrame) -> pd.DataFrame:
        if 'returns' in df.columns and not df['returns'].empty:
            # Ensure returns is float before quantile
            df['returns'] = pd.to_numeric(df['returns'], errors='coerce')
            df_clean_returns = df.dropna(subset=['returns'])

            if not df_clean_returns.empty:
                lower_b = df_clean_returns['returns'].quantile(self.config.return_winsorization_lower)
                upper_b = df_clean_returns['returns'].quantile(self.config.return_winsorization_upper)
                
                lower_b = max(lower_b, -self.config.max_daily_return)
                upper_b = min(upper_b, self.config.max_daily_return)
                
                df['returns'] = df['returns'].clip(lower=lower_b, upper=upper_b)
        return df

    # Methods for Hypothesis testing (like _add_risk_adjusted_metrics, _add_phase_indicators)
    # are better placed in the specific HypothesisXTestet.py files or a separate metrics module.
    # If they were here, they would look like:
    # def _add_risk_adjusted_metrics(self, df: pd.DataFrame) -> pd.DataFrame: ...
    # def _add_phase_indicators(self, df: pd.DataFrame) -> pd.DataFrame: ...
