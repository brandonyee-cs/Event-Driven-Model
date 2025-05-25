"""
Event Processor Module - Unified Volatility Model Implementation

This module implements the unified volatility model from the paper that combines
GJR-GARCH baseline volatility with event-specific adjustments through phi functions.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime, timedelta
import warnings
from scipy import stats
from arch import arch_model
from scipy.optimize import minimize
from scipy.special import ndtr
from dataclasses import dataclass
from config import Config

warnings.filterwarnings('ignore', category=RuntimeWarning)

@dataclass
class VolatilityParameters:
    """Parameters for the unified volatility model"""
    # GJR-GARCH parameters
    omega: float
    alpha: float
    beta: float
    gamma: float  # Asymmetry parameter
    
    # Event-specific parameters
    k1: float = 1.3  # Pre-event peak multiplier
    k2: float = 1.5  # Post-event peak multiplier
    delta_t1: float = 5.0  # Pre-event duration
    delta_t2: float = 3.0  # Post-event rise rate
    delta_t3: float = 10.0  # Post-event decay rate
    delta: int = 5  # Post-event rising phase duration (days)

class EventProcessor:
    """
    Main class for processing event-driven asset pricing data.
    Implements the unified volatility model from the paper.
    """
    
    def __init__(self, config: Config):
        """Initialize with configuration object"""
        self.config = config
        self.lookback_days = config.lookback_days
        self.event_window_pre = config.event_window_pre
        self.event_window_post = config.event_window_post
        self.min_required_days = config.min_required_days
        self.volatility_params = {}  # Store estimated parameters by symbol
        
    def process_events(self, price_data: pd.DataFrame, event_data: pd.DataFrame) -> pd.DataFrame:
        """
        Process events with unified volatility model.
        
        Args:
            price_data: DataFrame with columns [symbol, date, price, returns]
            event_data: DataFrame with columns [symbol, event_date, event_type]
            
        Returns:
            DataFrame with processed event data including unified volatility measures
        """
        results = []
        
        # Group events by symbol
        for symbol in event_data['symbol'].unique():
            symbol_events = event_data[event_data['symbol'] == symbol]
            symbol_prices = price_data[price_data['symbol'] == symbol].copy()
            
            if len(symbol_prices) < self.min_required_days:
                continue
                
            # Sort prices by date
            symbol_prices = symbol_prices.sort_values('date')
            
            # Estimate GJR-GARCH parameters for baseline volatility
            self._estimate_gjr_garch(symbol, symbol_prices)
            
            # Process each event
            for _, event in symbol_events.iterrows():
                event_results = self._process_single_event(
                    symbol_prices, event, symbol
                )
                if event_results is not None:
                    results.append(event_results)
        
        if not results:
            return pd.DataFrame()
            
        # Combine results
        results_df = pd.concat(results, ignore_index=True)
        
        # Add risk-adjusted metrics
        results_df = self._add_risk_adjusted_metrics(results_df)
        
        # Add impact uncertainty measures
        results_df = self._add_impact_uncertainty(results_df)
        
        return results_df
    
    def _estimate_gjr_garch(self, symbol: str, price_data: pd.DataFrame) -> None:
        """
        Estimate GJR-GARCH(1,1) parameters for baseline volatility.
        
        This implements the baseline volatility process from the paper:
        h_t = omega + alpha * epsilon_{t-1}^2 + beta * h_{t-1} + gamma * I_{t-1} * epsilon_{t-1}^2
        """
        returns = price_data['returns'].dropna().values * 100  # Convert to percentage
        
        try:
            # Fit GJR-GARCH(1,1) model
            model = arch_model(returns, vol='GARCH', p=1, q=1, o=1, dist='normal')
            res = model.fit(disp='off')
            
            # Extract parameters
            params = VolatilityParameters(
                omega=res.params['omega'],
                alpha=res.params['alpha[1]'],
                beta=res.params['beta[1]'],
                gamma=res.params['gamma[1]'] if 'gamma[1]' in res.params else 0.0
            )
            
            # Store conditional variance series
            params.conditional_variance = res.conditional_volatility ** 2
            params.returns_series = returns
            
            self.volatility_params[symbol] = params
            
        except Exception as e:
            # Fallback to simple GARCH if GJR fails
            try:
                model = arch_model(returns, vol='GARCH', p=1, q=1, dist='normal')
                res = model.fit(disp='off')
                
                params = VolatilityParameters(
                    omega=res.params['omega'],
                    alpha=res.params['alpha[1]'],
                    beta=res.params['beta[1]'],
                    gamma=0.0  # No asymmetry
                )
                
                params.conditional_variance = res.conditional_volatility ** 2
                params.returns_series = returns
                
                self.volatility_params[symbol] = params
                
            except:
                # Ultimate fallback to simple volatility
                params = VolatilityParameters(
                    omega=np.var(returns),
                    alpha=0.1,
                    beta=0.85,
                    gamma=0.0
                )
                self.volatility_params[symbol] = params
    
    def _calculate_unified_volatility(self, t: int, t_event: int, 
                                    baseline_vol: float, params: VolatilityParameters) -> float:
        """
        Calculate unified volatility sigma_e(t) = sqrt(h_t) * Phi(t)
        
        This implements the three-phase volatility process from the paper.
        """
        # Phi(t) function based on event phase
        if t <= t_event:
            # Pre-event phase (phi_1)
            phi = (params.k1 - 1) * np.exp(-((t - t_event)**2) / (2 * params.delta_t1**2))
        elif t_event < t <= t_event + params.delta:
            # Post-event rising phase (phi_2)
            phi = (params.k2 - 1) * (1 - np.exp(-(t - t_event) / params.delta_t2))
        else:
            # Post-event decay phase (phi_3)
            phi = (params.k2 - 1) * np.exp(-(t - t_event - params.delta) / params.delta_t3)
        
        # Unified volatility
        sigma_e = baseline_vol * (1 + phi)
        
        return sigma_e
    
    def _calculate_bias_parameter(self, t: int, t_event: int, 
                                params: VolatilityParameters, baseline_vol: float) -> float:
        """
        Calculate bias parameter b_t from the paper.
        
        b_t = b_0 * (1 + kappa * (sigma_e(t)/sqrt(h_t) - 1)/(k_2 - 1))
        for post-event rising phase only.
        """
        b_0 = 0.001  # Baseline bias (small positive value)
        kappa = 0.5  # Sensitivity to volatility changes
        
        if t_event < t <= t_event + params.delta:
            # Calculate Phi(t) for the current phase
            phi = (params.k2 - 1) * (1 - np.exp(-(t - t_event) / params.delta_t2))
            psi_t = 1 + phi  # sigma_e(t) / sqrt(h_t)
            
            bias = b_0 * (1 + kappa * (psi_t - 1) / (params.k2 - 1))
        else:
            bias = b_0
            
        return bias
    
    def _process_single_event(self, price_data: pd.DataFrame, event: pd.Series, 
                            symbol: str) -> Optional[pd.DataFrame]:
        """Process a single event with unified volatility model"""
        event_date = pd.to_datetime(event['event_date'])
        
        # Get event window data
        start_date = event_date - timedelta(days=self.event_window_pre)
        end_date = event_date + timedelta(days=self.event_window_post)
        
        # Filter data for event window
        mask = (price_data['date'] >= start_date) & (price_data['date'] <= end_date)
        event_data = price_data[mask].copy()
        
        if len(event_data) < self.min_required_days:
            return None
        
        # Get volatility parameters
        params = self.volatility_params.get(symbol)
        if params is None:
            return None
        
        # Calculate days relative to event
        event_data['days_to_event'] = (event_data['date'] - event_date).dt.days
        
        # Calculate baseline volatility from GARCH model
        baseline_vols = self._get_baseline_volatility(symbol, event_data)
        
        # Calculate unified volatility for each day
        unified_vols = []
        biases = []
        
        for idx, row in event_data.iterrows():
            t = row['days_to_event']
            t_event = 0  # Event occurs at t=0
            
            # Get baseline volatility
            baseline_vol = baseline_vols[idx] if idx < len(baseline_vols) else np.mean(baseline_vols)
            
            # Calculate unified volatility
            unified_vol = self._calculate_unified_volatility(t, t_event, baseline_vol, params)
            unified_vols.append(unified_vol)
            
            # Calculate bias parameter
            bias = self._calculate_bias_parameter(t, t_event, params, baseline_vol)
            biases.append(bias)
        
        event_data['baseline_volatility'] = baseline_vols[:len(event_data)]
        event_data['unified_volatility'] = unified_vols
        event_data['bias_parameter'] = biases
        
        # Calculate expected returns with bias
        event_data['expected_return'] = event_data['returns'] + event_data['bias_parameter']
        
        # Add event information
        event_data['event_date'] = event_date
        event_data['event_type'] = event.get('event_type', 'unknown')
        event_data['symbol'] = symbol
        
        # Calculate impact uncertainty (volatility innovations)
        event_data['impact_uncertainty'] = event_data['unified_volatility'] - event_data['baseline_volatility']
        
        return event_data
    
    def _get_baseline_volatility(self, symbol: str, event_data: pd.DataFrame) -> np.array:
        """Get baseline volatility from estimated GARCH model"""
        params = self.volatility_params.get(symbol)
        if params is None:
            return np.sqrt(event_data['returns'].var()) * np.ones(len(event_data))
        
        # Use stored conditional variance if available
        if hasattr(params, 'conditional_variance'):
            # Match dates and get corresponding volatilities
            baseline_vols = []
            for _, row in event_data.iterrows():
                # Simple approximation: use average conditional volatility
                baseline_vol = np.sqrt(np.mean(params.conditional_variance)) / 100
                baseline_vols.append(baseline_vol)
            return np.array(baseline_vols)
        else:
            # Fallback to simple calculation
            return np.sqrt(params.omega / (1 - params.alpha - params.beta)) * np.ones(len(event_data))
    
    def _add_risk_adjusted_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add risk-adjusted return metrics (RVR and Sharpe ratio)"""
        # Use 3-month Treasury rate as risk-free rate (annualized)
        risk_free_rate_annual = 0.045  # 4.5% annual
        risk_free_rate_daily = risk_free_rate_annual / 252
        
        # Calculate excess returns
        df['excess_return'] = df['expected_return'] - risk_free_rate_daily
        
        # Return-to-variance ratio (RVR)
        df['rvr'] = df['excess_return'] / (df['unified_volatility'] ** 2)
        
        # Sharpe ratio
        df['sharpe_ratio'] = df['excess_return'] / df['unified_volatility']
        
        # Phase identification for analysis
        df['phase'] = df.apply(lambda row: self._identify_phase(row['days_to_event']), axis=1)
        
        return df
    
    def _identify_phase(self, days_to_event: int) -> str:
        """Identify which phase of the event cycle we're in"""
        if days_to_event < -5:
            return 'pre_event_early'
        elif -5 <= days_to_event <= 0:
            return 'pre_event_late'
        elif 0 < days_to_event <= 5:
            return 'post_event_rising'
        else:
            return 'post_event_decay'
    
    def _add_impact_uncertainty(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add impact uncertainty measures based on GARCH innovations"""
        # Group by symbol and event
        for (symbol, event_date), group in df.groupby(['symbol', 'event_date']):
            # Calculate volatility innovations (unexpected changes)
            group_sorted = group.sort_values('date')
            
            # Simple approximation of volatility innovation
            vols = group_sorted['unified_volatility'].values
            vol_innovations = np.diff(vols, prepend=vols[0])
            
            # Store innovations
            df.loc[group.index, 'volatility_innovation'] = vol_innovations
        
        return df
    
    def winsorize_returns(self, df: pd.DataFrame, lower_pct: float = 0.01, 
                         upper_pct: float = 0.99) -> pd.DataFrame:
        """
        Winsorize returns instead of hard clipping.
        This is more statistically sound than hard clipping.
        """
        if 'returns' in df.columns:
            # Calculate percentiles
            lower_bound = df['returns'].quantile(lower_pct)
            upper_bound = df['returns'].quantile(upper_pct)
            
            # Winsorize
            df['returns_winsorized'] = df['returns'].clip(lower=lower_bound, upper=upper_bound)
            
            # Keep original returns for comparison
            df['returns_original'] = df['returns']
            
            # Use winsorized returns for analysis
            df['returns'] = df['returns_winsorized']
        
        return df
