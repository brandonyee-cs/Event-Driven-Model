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
from dataclasses import dataclass
from .config import Config

warnings.filterwarnings('ignore', category=RuntimeWarning)

@dataclass
class VolatilityParameters:
    """
    Parameters for the unified volatility model.
    Stores both GJR-GARCH parameters and event-specific adjustments.
    """
    # GJR-GARCH(1,1) parameters
    omega: float  # Long-run variance component
    alpha: float  # ARCH effect
    beta: float   # GARCH persistence
    gamma: float  # Asymmetry parameter (leverage effect)
    
    # Event-specific parameters (from paper Section 3.1)
    k1: float = 1.3      # Pre-event peak multiplier
    k2: float = 1.5      # Post-event peak multiplier (k2 > k1)
    delta_t1: float = 5.0  # Pre-event duration parameter
    delta_t2: float = 3.0  # Post-event rise rate
    delta_t3: float = 10.0 # Post-event decay rate
    delta: int = 5       # Post-event rising phase duration (days)
    
    # Additional stored values
    conditional_variance: Optional[np.ndarray] = None
    standardized_residuals: Optional[np.ndarray] = None
    log_likelihood: Optional[float] = None

class EventProcessor:
    """
    Main class for processing event-driven asset pricing data.
    Implements the unified volatility model: sigma_e(t) = sqrt(h_t) * (1 + phi(t))
    """
    
    def __init__(self, config: Config):
        """Initialize with configuration object"""
        self.config = config
        self.config.validate_parameters()  # Ensure parameters are valid
        
        # Store estimated parameters by symbol
        self.volatility_params: Dict[str, VolatilityParameters] = {}
        self.impact_uncertainty_cache: Dict[str, pd.DataFrame] = {}
        
    def process_events(self, price_data: pd.DataFrame, event_data: pd.DataFrame) -> pd.DataFrame:
        """
        Process events with unified volatility model.
        
        Args:
            price_data: DataFrame with columns [symbol, date, price, returns]
            event_data: DataFrame with columns [symbol, event_date, event_type]
            
        Returns:
            DataFrame with processed event data including all theoretical measures
        """
        results = []
        
        # Validate input data
        required_price_cols = ['symbol', 'date', 'price', 'returns']
        required_event_cols = ['symbol', 'event_date']
        
        if not all(col in price_data.columns for col in required_price_cols):
            raise ValueError(f"Price data must contain columns: {required_price_cols}")
        if not all(col in event_data.columns for col in required_event_cols):
            raise ValueError(f"Event data must contain columns: {required_event_cols}")
        
        # Process each symbol
        unique_symbols = event_data['symbol'].unique()
        print(f"Processing {len(unique_symbols)} symbols with events...")
        
        for i, symbol in enumerate(unique_symbols):
            if i % 100 == 0:
                print(f"  Processing symbol {i+1}/{len(unique_symbols)}")
                
            symbol_events = event_data[event_data['symbol'] == symbol]
            symbol_prices = price_data[price_data['symbol'] == symbol].copy()
            
            if len(symbol_prices) < self.config.lookback_days + self.config.min_required_days:
                continue
                
            # Sort prices by date
            symbol_prices = symbol_prices.sort_values('date').reset_index(drop=True)
            
            # Winsorize returns to handle outliers
            symbol_prices = self._winsorize_returns(symbol_prices)
            
            # Estimate GJR-GARCH parameters for baseline volatility
            garch_success = self._estimate_gjr_garch(symbol, symbol_prices)
            if not garch_success:
                continue
            
            # Process each event for this symbol
            for _, event in symbol_events.iterrows():
                event_results = self._process_single_event(
                    symbol_prices, event, symbol
                )
                if event_results is not None:
                    results.append(event_results)
        
        if not results:
            print("Warning: No events successfully processed")
            return pd.DataFrame()
            
        # Combine results
        results_df = pd.concat(results, ignore_index=True)
        
        # Add risk-adjusted metrics
        results_df = self._add_risk_adjusted_metrics(results_df)
        
        # Add market phase indicators
        results_df = self._add_phase_indicators(results_df)
        
        print(f"Successfully processed {len(results_df)} event-day observations")
        
        return results_df
    
    def _estimate_gjr_garch(self, symbol: str, price_data: pd.DataFrame) -> bool:
        """
        Estimate GJR-GARCH(1,1) parameters for baseline volatility.
        
        Implements: h_t = omega + alpha * epsilon_{t-1}^2 + beta * h_{t-1} 
                          + gamma * I_{t-1} * epsilon_{t-1}^2
                          
        where I_{t-1} = 1 if epsilon_{t-1} < 0 (leverage effect)
        """
        returns = price_data['returns'].dropna().values * 100  # Convert to percentage
        
        if len(returns) < self.config.lookback_days:
            return False
        
        try:
            # Fit GJR-GARCH(1,1) model
            model = arch_model(returns, vol='GARCH', p=1, q=1, o=1, dist='normal')
            res = model.fit(disp='off', show_warning=False)
            
            # Extract and validate parameters
            omega = res.params.get('omega', self.config.gjr_omega)
            alpha = res.params.get('alpha[1]', self.config.gjr_alpha)
            beta = res.params.get('beta[1]', self.config.gjr_beta)
            gamma = res.params.get('gamma[1]', self.config.gjr_gamma)
            
            # Check stationarity
            if alpha + beta + gamma/2 >= 0.9999:
                # Use default parameters if estimation violates stationarity
                omega = self.config.gjr_omega
                alpha = self.config.gjr_alpha
                beta = self.config.gjr_beta
                gamma = self.config.gjr_gamma
            
            # Create parameter object
            params = VolatilityParameters(
                omega=omega,
                alpha=alpha,
                beta=beta,
                gamma=gamma,
                k1=self.config.event_k1,
                k2=self.config.event_k2,
                delta=self.config.event_delta,
                delta_t1=self.config.event_delta_t1,
                delta_t2=self.config.event_delta_t2,
                delta_t3=self.config.event_delta_t3
            )
            
            # Store conditional variance and standardized residuals
            params.conditional_variance = res.conditional_volatility ** 2
            params.standardized_residuals = res.std_resid
            params.log_likelihood = res.loglikelihood
            
            self.volatility_params[symbol] = params
            return True
            
        except Exception as e:
            # Fallback to default parameters
            params = VolatilityParameters(
                omega=self.config.gjr_omega,
                alpha=self.config.gjr_alpha,
                beta=self.config.gjr_beta,
                gamma=self.config.gjr_gamma,
                k1=self.config.event_k1,
                k2=self.config.event_k2,
                delta=self.config.event_delta,
                delta_t1=self.config.event_delta_t1,
                delta_t2=self.config.event_delta_t2,
                delta_t3=self.config.event_delta_t3
            )
            self.volatility_params[symbol] = params
            return True
    
    def _calculate_phi_function(self, t: int, t_event: int, params: VolatilityParameters) -> float:
        """
        Calculate phi adjustment factor based on event phase.
        
        Three phases from the paper:
        1. Pre-event (t <= t_event): phi_1(t) with Gaussian rise
        2. Post-event rising (t_event < t <= t_event + delta): phi_2(t) with exponential rise
        3. Post-event decay (t > t_event + delta): phi_3(t) with exponential decay
        """
        if t <= t_event:
            # Pre-event phase: phi_1(t) = (k1-1) * exp(-((t-t_event)^2)/(2*delta_t1^2))
            phi = (params.k1 - 1) * np.exp(-((t - t_event)**2) / (2 * params.delta_t1**2))
        elif t_event < t <= t_event + params.delta:
            # Post-event rising: phi_2(t) = (k2-1) * (1 - exp(-(t-t_event)/delta_t2))
            phi = (params.k2 - 1) * (1 - np.exp(-(t - t_event) / params.delta_t2))
        else:
            # Post-event decay: phi_3(t) = (k2-1) * exp(-(t-t_event-delta)/delta_t3)
            phi = (params.k2 - 1) * np.exp(-(t - t_event - params.delta) / params.delta_t3)
        
        return phi
    
    def _calculate_unified_volatility(self, t: int, t_event: int, 
                                    baseline_vol: float, params: VolatilityParameters) -> float:
        """
        Calculate unified volatility: sigma_e(t) = sqrt(h_t) * (1 + phi(t))
        
        This is the core of the unified volatility model, combining:
        - Baseline volatility from GJR-GARCH (sqrt(h_t))
        - Event-specific adjustment (1 + phi(t))
        """
        phi = self._calculate_phi_function(t, t_event, params)
        sigma_e = baseline_vol * (1 + phi)
        return sigma_e
    
    def _calculate_bias_parameter(self, t: int, t_event: int, 
                                sigma_e: float, baseline_vol: float,
                                investor_type: str = 'average') -> float:
        """
        Calculate bias parameter b_t from Assumption 5.
        
        b_t = b_0 * (1 + kappa * (sigma_e(t)/sqrt(h_t) - 1)/(k_2 - 1))
        
        Only applies during post-event rising phase when optimism is heightened.
        """
        # Get investor-specific parameters
        if investor_type == 'informed':
            b_0 = 0.001
            kappa = 0.3
        elif investor_type == 'uninformed':
            b_0 = 0.003
            kappa = 0.5
        else:  # average
            b_0 = self.config.bias_baseline_b0
            kappa = self.config.bias_kappa_sensitivity
        
        if t_event < t <= t_event + self.config.event_delta:
            # Calculate Psi_t = sigma_e(t) / sqrt(h_t)
            psi_t = sigma_e / baseline_vol if baseline_vol > 0 else 1
            
            # Apply bias adjustment
            bias = b_0 * (1 + kappa * (psi_t - 1) / (self.config.event_k2 - 1))
        else:
            bias = b_0
            
        return bias
    
    def _calculate_impact_uncertainty(self, params: VolatilityParameters,
                                    returns: pd.Series, dates: pd.Series) -> pd.Series:
        """
        Calculate impact uncertainty as GARCH volatility innovations.
        
        From the paper: ImpactUncertainty_t = h_t - E_{t-1}[h_t]
        
        This captures unexpected changes in conditional variance.
        """
        if params.conditional_variance is None:
            return pd.Series(0, index=dates.index)
        
        # Calculate one-step-ahead variance forecasts
        h_t = params.conditional_variance[-len(returns):]
        
        # Initialize expected variance
        E_h_t = np.zeros_like(h_t)
        E_h_t[0] = params.omega / (1 - params.alpha - params.beta - params.gamma/2)
        
        # Calculate expected variance for each period
        for i in range(1, len(h_t)):
            eps_sq = (returns.iloc[i-1] * 100) ** 2 / h_t[i-1] if h_t[i-1] > 0 else 0
            indicator = 1 if returns.iloc[i-1] < 0 else 0
            
            E_h_t[i] = (params.omega + 
                       params.alpha * eps_sq * h_t[i-1] +
                       params.beta * h_t[i-1] +
                       params.gamma * indicator * eps_sq * h_t[i-1])
        
        # Volatility innovation = actual - expected
        vol_innovation = (np.sqrt(h_t) - np.sqrt(E_h_t)) / 100  # Convert back to decimal
        
        return pd.Series(vol_innovation, index=dates.index)
    
    def _process_single_event(self, price_data: pd.DataFrame, event: pd.Series, 
                            symbol: str) -> Optional[pd.DataFrame]:
        """Process a single event with unified volatility model"""
        event_date = pd.to_datetime(event['event_date'])
        
        # Get event window data
        start_date = event_date - timedelta(days=self.config.event_window_pre + self.config.lookback_days)
        end_date = event_date + timedelta(days=self.config.event_window_post)
        
        # Filter data for extended window (includes lookback for GARCH)
        mask = (price_data['date'] >= start_date) & (price_data['date'] <= end_date)
        window_data = price_data[mask].copy()
        
        if len(window_data) < self.config.min_required_days:
            return None
        
        # Get volatility parameters
        params = self.volatility_params.get(symbol)
        if params is None:
            return None
        
        # Calculate days relative to event
        window_data['days_to_event'] = (window_data['date'] - event_date).dt.days
        
        # Filter to analysis window only
        analysis_mask = (window_data['days_to_event'] >= -self.config.event_window_pre) & \
                       (window_data['days_to_event'] <= self.config.event_window_post)
        event_data = window_data[analysis_mask].copy()
        
        # Calculate baseline volatility from GARCH model
        baseline_vols = self._get_baseline_volatility(symbol, window_data, event_data.index)
        
        # Calculate impact uncertainty (volatility innovations)
        impact_uncertainty = self._calculate_impact_uncertainty(
            params, window_data['returns'], window_data['date']
        )
        impact_uncertainty_event = impact_uncertainty[event_data.index]
        
        # Calculate unified volatility and bias for each day
        unified_vols = []
        biases = []
        phi_values = []
        
        t_event = 0  # Event occurs at t=0
        
        for idx, row in event_data.iterrows():
            t = row['days_to_event']
            
            # Get baseline volatility
            baseline_vol = baseline_vols[idx]
            
            # Calculate phi adjustment
            phi = self._calculate_phi_function(t, t_event, params)
            phi_values.append(phi)
            
            # Calculate unified volatility
            unified_vol = self._calculate_unified_volatility(t, t_event, baseline_vol, params)
            unified_vols.append(unified_vol)
            
            # Calculate bias parameter
            bias = self._calculate_bias_parameter(t, t_event, unified_vol, baseline_vol)
            biases.append(bias)
        
        # Store calculated values
        event_data['baseline_volatility'] = list(baseline_vols)
        event_data['phi_adjustment'] = phi_values
        event_data['unified_volatility'] = unified_vols
        event_data['bias_parameter'] = biases
        event_data['volatility_innovation'] = list(impact_uncertainty_event)
        
        # Calculate expected returns with bias
        event_data['expected_return'] = event_data['returns'] + event_data['bias_parameter']
        
        # Add event information
        event_data['event_date'] = event_date
        event_data['event_type'] = event.get('event_type', 'unknown')
        event_data['symbol'] = symbol
        
        # Add GARCH parameters for analysis
        event_data['garch_alpha'] = params.alpha
        event_data['garch_beta'] = params.beta
        event_data['garch_gamma'] = params.gamma
        
        return event_data
    
    def _get_baseline_volatility(self, symbol: str, window_data: pd.DataFrame,
                                event_indices: pd.Index) -> pd.Series:
        """
        Get baseline volatility from estimated GARCH model.
        Returns volatility (not variance) in decimal form.
        """
        params = self.volatility_params.get(symbol)
        if params is None:
            return pd.Series(window_data['returns'].std(), index=event_indices)
        
        # Calculate conditional variance for the window
        returns = window_data['returns'].values
        h_t = np.zeros(len(returns))
        
        # Initialize with unconditional variance
        h_t[0] = params.omega / (1 - params.alpha - params.beta - params.gamma/2)
        
        # Forward recursion for GARCH
        for t in range(1, len(returns)):
            eps_prev = returns[t-1] / np.sqrt(h_t[t-1]) if h_t[t-1] > 0 else 0
            indicator = 1 if returns[t-1] < 0 else 0
            
            h_t[t] = (params.omega + 
                     params.alpha * (eps_prev**2) * h_t[t-1] +
                     params.beta * h_t[t-1] +
                     params.gamma * indicator * (eps_prev**2) * h_t[t-1])
        
        # Convert to volatility and select event window indices
        baseline_vol_series = pd.Series(np.sqrt(h_t), index=window_data.index)
        
        return baseline_vol_series[event_indices]
    
    def _add_risk_adjusted_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add risk-adjusted return metrics (RVR and Sharpe ratio).
        These are key for testing Hypothesis 1.
        """
        # Calculate excess returns
        df['excess_return'] = df['expected_return'] - self.config.risk_free_rate_daily
        
        # Return-to-variance ratio (RVR) = excess return / variance
        df['rvr'] = df['excess_return'] / (df['unified_volatility'] ** 2)
        
        # Sharpe ratio = excess return / volatility
        df['sharpe_ratio'] = df['excess_return'] / df['unified_volatility']
        
        # Also calculate metrics using baseline volatility for comparison
        df['rvr_baseline'] = df['excess_return'] / (df['baseline_volatility'] ** 2)
        df['sharpe_baseline'] = df['excess_return'] / df['baseline_volatility']
        
        # Calculate the enhancement from event adjustments
        df['rvr_enhancement'] = df['rvr'] / df['rvr_baseline'] - 1
        df['sharpe_enhancement'] = df['sharpe_ratio'] / df['sharpe_baseline'] - 1
        
        return df
    
    def _add_phase_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add phase identification for analysis.
        Critical for testing hypotheses about different event phases.
        """
        def identify_phase(days_to_event: int) -> str:
            if days_to_event < -5:
                return 'pre_event_early'
            elif -5 <= days_to_event <= 0:
                return 'pre_event_late'
            elif 0 < days_to_event <= self.config.event_delta:
                return 'post_event_rising'
            else:
                return 'post_event_decay'
        
        df['phase'] = df['days_to_event'].apply(identify_phase)
        
        # Add binary indicators
        df['is_pre_event'] = df['days_to_event'] <= 0
        df['is_post_rising'] = (df['days_to_event'] > 0) & (df['days_to_event'] <= self.config.event_delta)
        df['is_post_decay'] = df['days_to_event'] > self.config.event_delta
        
        return df
    
    def _winsorize_returns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Winsorize returns to handle outliers.
        More statistically sound than hard clipping.
        """
        if 'returns' in df.columns:
            # Calculate percentiles
            lower_bound = df['returns'].quantile(self.config.return_winsorization_lower)
            upper_bound = df['returns'].quantile(self.config.return_winsorization_upper)
            
            # Additional check for extreme values
            lower_bound = max(lower_bound, -self.config.max_daily_return)
            upper_bound = min(upper_bound, self.config.max_daily_return)
            
            # Winsorize
            df['returns_original'] = df['returns']
            df['returns'] = df['returns'].clip(lower=lower_bound, upper=upper_bound)
            df['returns_winsorized'] = df['returns']
        
        return df
    
    def get_volatility_summary(self, symbol: str) -> Dict[str, float]:
        """
        Get summary statistics for a symbol's volatility parameters.
        Useful for model diagnostics.
        """
        params = self.volatility_params.get(symbol)
        if params is None:
            return {}
        
        # Calculate unconditional variance
        uncond_var = params.omega / (1 - params.alpha - params.beta - params.gamma/2)
        
        return {
            'omega': params.omega,
            'alpha': params.alpha,
            'beta': params.beta,
            'gamma': params.gamma,
            'persistence': params.alpha + params.beta + params.gamma/2,
            'unconditional_volatility': np.sqrt(uncond_var),
            'log_likelihood': params.log_likelihood
        }
