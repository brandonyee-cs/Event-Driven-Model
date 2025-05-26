"""
Theoretical Models Module - Unified Asset Pricing Framework

Implements the theoretical models from "Modeling Equilibrium Asset Pricing 
Around Events with Heterogeneous Beliefs, Dynamic Volatility, and a Two-Risk 
Uncertainty Framework" by Brandon Yee.

Key components:
- Multi-period portfolio optimization with transaction costs
- Equilibrium price dynamics with heterogeneous investors  
- Unified volatility dynamics (GJR-GARCH + event adjustments)
- Risk metrics for hypothesis testing
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, List, Union
from dataclasses import dataclass
from scipy.optimize import minimize_scalar, brentq, minimize
from scipy.stats import norm
import warnings

warnings.filterwarnings('ignore')

@dataclass
class InvestorParams:
    """Parameters for different investor types (Section 3.1, Assumption 6)"""
    risk_aversion: float  # gamma
    bias_baseline: float  # b_0
    bias_sensitivity: float  # kappa
    transaction_cost_buy: float  # tau_b
    transaction_cost_sell: float  # tau_s
    liquidity_constraint: float = 0.0  # lambda for liquidity traders
    information_quality: float = 1.0  # precision of information
    wealth: float = 1000000.0  # investor wealth

@dataclass
class MarketParams:
    """Market-wide parameters (Section 3.1)"""
    risk_free_rate: float  # r_t (daily)
    generic_return: float  # mu_g (daily expected return)
    generic_volatility: float  # sigma_g (daily volatility)
    correlation: float  # rho between event and generic assets
    event_asset_supply: float  # S_e (fixed supply)
    
class UnifiedVolatilityModel:
    """
    Implements the unified volatility model from Section 3.1, Assumption 4.
    Combines GJR-GARCH baseline with event-specific adjustments.
    
    sigma_e(t) = sqrt(h_t) * (1 + phi(t))
    """
    
    def __init__(self, omega: float, alpha: float, beta: float, gamma: float,
                 k1: float = 1.3, k2: float = 1.5, delta: int = 5,
                 delta_t1: float = 5.0, delta_t2: float = 3.0, delta_t3: float = 10.0):
        """Initialize volatility model parameters"""
        # GJR-GARCH parameters
        self.omega = omega
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma  # Asymmetry parameter
        
        # Event-specific parameters
        self.k1 = k1  # Pre-event peak
        self.k2 = k2  # Post-event peak
        self.delta = delta  # Post-event rising duration
        self.delta_t1 = delta_t1  # Pre-event duration
        self.delta_t2 = delta_t2  # Post-event rise rate
        self.delta_t3 = delta_t3  # Post-event decay rate
        
        # Validate parameters
        self._validate_parameters()
    
    def _validate_parameters(self):
        """Check parameter constraints"""
        # Stationarity condition
        persistence = self.alpha + self.beta + self.gamma/2
        if persistence >= 1:
            raise ValueError(f"GJR-GARCH parameters violate stationarity: {persistence} >= 1")
        
        # Event parameters
        if self.k1 <= 1.0 or self.k2 <= 1.0:
            raise ValueError("k1 and k2 must be > 1")
        if self.k2 <= self.k1:
            raise ValueError("k2 must be > k1 (post-event peak exceeds pre-event)")
    
    def baseline_volatility_path(self, returns: np.array, h0: Optional[float] = None) -> np.array:
        """
        Calculate baseline volatility path using GJR-GARCH(1,1).
        
        h_t = omega + alpha * eps_{t-1}^2 + beta * h_{t-1} + gamma * I_{t-1} * eps_{t-1}^2
        """
        T = len(returns)
        h = np.zeros(T)
        
        # Initialize with unconditional variance or provided value
        if h0 is None:
            h[0] = self.omega / (1 - self.alpha - self.beta - self.gamma/2)
        else:
            h[0] = h0
        
        for t in range(1, T):
            eps_prev = returns[t-1] / np.sqrt(h[t-1]) if h[t-1] > 0 else 0
            indicator = 1 if returns[t-1] < 0 else 0
            
            h[t] = (self.omega + 
                   self.alpha * (eps_prev**2) * h[t-1] +
                   self.beta * h[t-1] +
                   self.gamma * indicator * (eps_prev**2) * h[t-1])
        
        return np.sqrt(h)  # Return volatility, not variance
    
    def phi_function(self, t: Union[int, np.ndarray], t_event: int = 0) -> Union[float, np.ndarray]:
        """
        Calculate phi adjustment factor based on event phase (Section 3.1).
        
        phi_1(t) for t <= t_event: Pre-event rise
        phi_2(t) for t_event < t <= t_event + delta: Post-event rising
        phi_3(t) for t > t_event + delta: Post-event decay
        """
        # Handle both scalar and array inputs
        is_scalar = np.isscalar(t)
        t_array = np.atleast_1d(t)
        phi = np.zeros_like(t_array, dtype=float)
        
        # Pre-event phase
        pre_mask = t_array <= t_event
        if np.any(pre_mask):
            phi[pre_mask] = (self.k1 - 1) * np.exp(
                -((t_array[pre_mask] - t_event)**2) / (2 * self.delta_t1**2)
            )
        
        # Post-event rising phase
        rising_mask = (t_array > t_event) & (t_array <= t_event + self.delta)
        if np.any(rising_mask):
            phi[rising_mask] = (self.k2 - 1) * (
                1 - np.exp(-(t_array[rising_mask] - t_event) / self.delta_t2)
            )
        
        # Post-event decay phase
        decay_mask = t_array > t_event + self.delta
        if np.any(decay_mask):
            phi[decay_mask] = (self.k2 - 1) * np.exp(
                -(t_array[decay_mask] - t_event - self.delta) / self.delta_t3
            )
        
        return phi[0] if is_scalar else phi
    
    def unified_volatility(self, t: Union[int, np.ndarray], baseline_vol: Union[float, np.ndarray], 
                          t_event: int = 0) -> Union[float, np.ndarray]:
        """
        Calculate unified volatility: sigma_e(t) = sqrt(h_t) * (1 + phi(t))
        """
        phi = self.phi_function(t, t_event)
        return baseline_vol * (1 + phi)
    
    def bias_parameter(self, t: Union[int, np.ndarray], baseline_vol: Union[float, np.ndarray], 
                      b0: float, kappa: float, t_event: int = 0) -> Union[float, np.ndarray]:
        """
        Calculate time-varying bias parameter (Section 3.1, Assumption 5).
        b_t = b_0 * (1 + kappa * (Psi_t - 1)/(k_2 - 1)) for post-event rising phase
        """
        is_scalar = np.isscalar(t)
        t_array = np.atleast_1d(t)
        bias = np.full_like(t_array, b0, dtype=float)
        
        # Only adjust during post-event rising phase
        rising_mask = (t_array > t_event) & (t_array <= t_event + self.delta)
        if np.any(rising_mask):
            phi = self.phi_function(t_array[rising_mask], t_event)
            psi_t = 1 + phi  # sigma_e(t) / sqrt(h_t)
            bias[rising_mask] = b0 * (1 + kappa * (psi_t - 1) / (self.k2 - 1))
        
        return bias[0] if is_scalar else bias

class PortfolioOptimizer:
    """
    Implements multi-period portfolio optimization with transaction costs (Section 3.2).
    Solves the mean-variance optimization problem with asymmetric transaction costs.
    """
    
    def __init__(self, investor_params: InvestorParams, market_params: MarketParams,
                 volatility_model: UnifiedVolatilityModel):
        """Initialize optimizer with investor and market parameters"""
        self.investor = investor_params
        self.market = market_params
        self.vol_model = volatility_model
    
    def optimal_weights(self, expected_return_e: float, volatility_e: float,
                       previous_weight_e: float, t: int = 0, t_event: int = 0) -> Tuple[float, float]:
        """
        Calculate optimal portfolio weights for event and generic assets.
        
        Implements equations (14)-(15) from Section 3.2.1 with transaction costs.
        
        Returns:
            Tuple of (weight_event, weight_generic)
        """
        gamma = self.investor.risk_aversion
        rho = self.market.correlation
        sigma_g = self.market.generic_volatility
        sigma_e = volatility_e
        wealth = self.investor.wealth
        
        # Check for liquidity constraints (pre-event)
        is_pre_event = t <= t_event
        
        # Define objective function (negative utility for minimization)
        def utility(w_e):
            # Determine transaction cost
            if w_e > previous_weight_e:
                tau = self.investor.transaction_cost_buy
                # Apply liquidity constraint for liquidity traders pre-event
                if is_pre_event and self.investor.liquidity_constraint > 0:
                    max_increase = (1 - self.investor.liquidity_constraint) * (1 - previous_weight_e)
                    if w_e > previous_weight_e + max_increase:
                        return -np.inf  # Constraint violated
            elif w_e < previous_weight_e:
                tau = self.investor.transaction_cost_sell
            else:
                tau = 0
            
            # Transaction cost adjustment
            tc_adjustment = tau * abs(w_e - previous_weight_e) * wealth
            
            # Calculate optimal weight for generic asset given w_e
            # From first-order conditions (Section 3.2.1)
            denominator = gamma * (sigma_g**2 - rho * sigma_e * sigma_g)
            if abs(denominator) < 1e-10:
                return -np.inf
            
            numerator_g = self.market.generic_return - self.market.risk_free_rate
            cross_term = (rho * sigma_e * sigma_g * (expected_return_e - self.market.risk_free_rate)) / \
                        (sigma_g**2 * (sigma_e**2 - rho * sigma_e * sigma_g))
            
            w_g = numerator_g / denominator - cross_term * w_e
            
            # Risk-free weight
            w_rf = 1 - w_e - w_g
            
            # Portfolio return (including transaction costs)
            exp_return = (w_e * expected_return_e + 
                         w_g * self.market.generic_return + 
                         w_rf * self.market.risk_free_rate - 
                         tc_adjustment / wealth)
            
            # Portfolio variance
            variance = (w_e**2 * sigma_e**2 + 
                       w_g**2 * sigma_g**2 + 
                       2 * w_e * w_g * rho * sigma_e * sigma_g)
            
            # Mean-variance utility
            utility_val = exp_return - (gamma / 2) * variance
            
            return utility_val
        
        # Find optimal weight for event asset
        # Use bounded optimization to ensure weights are reasonable
        bounds = (0, 1)
        
        # Try multiple starting points to avoid local optima
        candidates = []
        for start in [previous_weight_e, 0.1, 0.5]:
            try:
                result = minimize_scalar(lambda w: -utility(w), bounds=bounds, 
                                       method='bounded', options={'xatol': 1e-6})
                if result.success:
                    candidates.append((result.x, utility(result.x)))
            except:
                continue
        
        if not candidates:
            # Fallback to no change
            return previous_weight_e, self._calculate_generic_weight(
                previous_weight_e, expected_return_e, volatility_e
            )
        
        # Select best candidate
        w_e_optimal = max(candidates, key=lambda x: x[1])[0]
        w_g_optimal = self._calculate_generic_weight(w_e_optimal, expected_return_e, volatility_e)
        
        return w_e_optimal, w_g_optimal
    
    def _calculate_generic_weight(self, w_e: float, expected_return_e: float, 
                                 volatility_e: float) -> float:
        """Calculate optimal generic asset weight given event asset weight"""
        gamma = self.investor.risk_aversion
        rho = self.market.correlation
        sigma_g = self.market.generic_volatility
        sigma_e = volatility_e
        
        denominator = gamma * (sigma_g**2 - rho * sigma_e * sigma_g)
        if abs(denominator) < 1e-10:
            return 0
        
        numerator_g = self.market.generic_return - self.market.risk_free_rate
        cross_term = (rho * sigma_e * sigma_g * (expected_return_e - self.market.risk_free_rate)) / \
                    (sigma_g**2 * (sigma_e**2 - rho * sigma_e * sigma_g))
        
        w_g = numerator_g / denominator - cross_term * w_e
        
        # Ensure non-negative
        return max(0, min(1 - w_e, w_g))
    
    def no_trade_region(self, previous_weight_e: float, expected_return_e: float,
                       volatility_e: float) -> Tuple[float, float]:
        """
        Calculate the no-trade region boundaries.
        
        Returns:
            Tuple of (lower_bound, upper_bound) for event asset weight
        """
        gamma = self.investor.risk_aversion
        tau_b = self.investor.transaction_cost_buy
        tau_s = self.investor.transaction_cost_sell
        
        # Simplified calculation based on transaction cost impact
        tc_impact = (tau_b + tau_s) * self.investor.wealth / (gamma * volatility_e**2)
        
        # Adjust for expected return differential
        ret_adjustment = abs(expected_return_e - self.market.risk_free_rate) / volatility_e**2
        width = tc_impact * (1 + ret_adjustment)
        
        lower_bound = max(0, previous_weight_e - width/2)
        upper_bound = min(1, previous_weight_e + width/2)
        
        return lower_bound, upper_bound

class EquilibriumModel:
    """
    Implements market equilibrium with heterogeneous investors (Section 3.3).
    Solves for equilibrium prices where aggregate demand equals fixed supply.
    """
    
    def __init__(self, market_params: MarketParams, volatility_model: UnifiedVolatilityModel,
                 n_informed: float = 0.3, n_uninformed: float = 0.5, n_liquidity: float = 0.2):
        """Initialize equilibrium model"""
        self.market = market_params
        self.vol_model = volatility_model
        
        # Investor proportions
        self.n_informed = n_informed
        self.n_uninformed = n_uninformed
        self.n_liquidity = n_liquidity
        
        # Define investor types based on paper
        self.informed_investor = InvestorParams(
            risk_aversion=2.0,
            bias_baseline=0.001,
            bias_sensitivity=0.3,
            transaction_cost_buy=0.002,
            transaction_cost_sell=0.001,
            liquidity_constraint=0.0,
            information_quality=0.9
        )
        
        self.uninformed_investor = InvestorParams(
            risk_aversion=3.0,
            bias_baseline=0.003,
            bias_sensitivity=0.5,
            transaction_cost_buy=0.003,
            transaction_cost_sell=0.002,
            liquidity_constraint=0.0,
            information_quality=0.5
        )
        
        self.liquidity_trader = InvestorParams(
            risk_aversion=2.5,
            bias_baseline=0.0,
            bias_sensitivity=0.0,
            transaction_cost_buy=0.002,
            transaction_cost_sell=0.001,
            liquidity_constraint=0.3,  # 30% purchase reduction pre-event
            information_quality=0.0
        )
    
    def aggregate_demand(self, mu_e: float, t: int, baseline_vol: float,
                        information: float, previous_weights: Dict[str, float],
                        t_event: int = 0) -> float:
        """
        Calculate aggregate demand for the event asset at given expected return.
        Implements equation (20) from Section 3.3.
        """
        # Calculate unified volatility
        vol_e = self.vol_model.unified_volatility(t, baseline_vol, t_event)
        
        total_demand = 0
        total_wealth = 0
        
        # Informed investor demand
        bias_informed = self.vol_model.bias_parameter(
            t, baseline_vol, self.informed_investor.bias_baseline,
            self.informed_investor.bias_sensitivity, t_event
        )
        exp_return_informed = mu_e + bias_informed * information * self.informed_investor.information_quality
        
        optimizer_informed = PortfolioOptimizer(
            self.informed_investor, self.market, self.vol_model
        )
        w_e_informed, _ = optimizer_informed.optimal_weights(
            exp_return_informed, vol_e, previous_weights.get('informed', 0.1), t, t_event
        )
        
        informed_wealth = self.n_informed * self.informed_investor.wealth
        demand_informed = informed_wealth * w_e_informed
        
        # Uninformed investor demand
        bias_uninformed = self.vol_model.bias_parameter(
            t, baseline_vol, self.uninformed_investor.bias_baseline,
            self.uninformed_investor.bias_sensitivity, t_event
        )
        # Uninformed have noisier information
        noisy_info = information * self.uninformed_investor.information_quality + \
                    np.random.normal(0, 0.1 * abs(information))
        exp_return_uninformed = mu_e + bias_uninformed * noisy_info
        
        optimizer_uninformed = PortfolioOptimizer(
            self.uninformed_investor, self.market, self.vol_model
        )
        w_e_uninformed, _ = optimizer_uninformed.optimal_weights(
            exp_return_uninformed, vol_e, previous_weights.get('uninformed', 0.1), t, t_event
        )
        
        uninformed_wealth = self.n_uninformed * self.uninformed_investor.wealth
        demand_uninformed = uninformed_wealth * w_e_uninformed
        
        # Liquidity trader demand (no information-based bias)
        optimizer_liquidity = PortfolioOptimizer(
            self.liquidity_trader, self.market, self.vol_model
        )
        w_e_liquidity, _ = optimizer_liquidity.optimal_weights(
            mu_e, vol_e, previous_weights.get('liquidity', 0.1), t, t_event
        )
        
        liquidity_wealth = self.n_liquidity * self.liquidity_trader.wealth
        demand_liquidity = liquidity_wealth * w_e_liquidity
        
        # Total demand
        total_demand = demand_informed + demand_uninformed + demand_liquidity
        total_wealth = informed_wealth + uninformed_wealth + liquidity_wealth
        
        return total_demand / total_wealth  # Return as fraction of total wealth
    
    def find_equilibrium_return(self, t: int, baseline_vol: float, information: float,
                               previous_weights: Dict[str, float], t_event: int = 0) -> float:
        """
        Find equilibrium expected return where demand equals supply.
        Implements market clearing condition from Section 3.3.
        """
        supply = self.market.event_asset_supply
        
        # Define excess demand function
        def excess_demand(mu_e):
            demand = self.aggregate_demand(
                mu_e, t, baseline_vol, information, previous_weights, t_event
            )
            return demand - supply
        
        # Find equilibrium using bisection
        try:
            # Search for reasonable bounds
            mu_low = -0.1  # -10% daily return
            mu_high = 0.1  # +10% daily return
            
            # Check if bounds bracket the solution
            ed_low = excess_demand(mu_low)
            ed_high = excess_demand(mu_high)
            
            if ed_low * ed_high > 0:
                # Bounds don't bracket, expand search
                if ed_low > 0:  # Demand too high even at low return
                    mu_low = -0.5
                else:  # Demand too low even at high return
                    mu_high = 0.5
            
            # Find equilibrium
            mu_e_eq = brentq(excess_demand, mu_low, mu_high, xtol=1e-6)
            
            return mu_e_eq
            
        except Exception as e:
            # Fallback to risk-free rate plus small premium
            return self.market.risk_free_rate + 0.001
    
    def simulate_equilibrium_path(self, T_pre: int = 30, T_post: int = 30, 
                                 baseline_vol: float = 0.02, information: float = 1.0) -> pd.DataFrame:
        """
        Simulate equilibrium price and return path around an event.
        """
        t_event = 0
        times = range(-T_pre, T_post + 1)
        
        results = []
        weights = {'informed': 0.1, 'uninformed': 0.1, 'liquidity': 0.1}
        
        # Initial price normalization
        P0 = 100.0
        prices = [P0]
        
        for i, t in enumerate(times):
            # Find equilibrium return
            mu_e = self.find_equilibrium_return(
                t, baseline_vol, information, weights, t_event
            )
            
            # Calculate unified volatility
            vol_e = self.vol_model.unified_volatility(t, baseline_vol, t_event)
            
            # Update price based on return
            if i > 0:
                P_new = prices[-1] * (1 + mu_e)
                prices.append(P_new)
            
            # Update weights based on new allocation
            # Simplified: gradual mean reversion
            for investor_type in weights:
                weights[investor_type] = 0.9 * weights[investor_type] + 0.1 * 0.1
            
            # Store results
            results.append({
                'time': t,
                'days_to_event': t,
                'equilibrium_return': mu_e,
                'price': prices[-1] if i > 0 else P0,
                'unified_volatility': vol_e,
                'baseline_volatility': baseline_vol,
                'phi_adjustment': self.vol_model.phi_function(t, t_event),
                'phase': self._identify_phase(t, t_event)
            })
        
        return pd.DataFrame(results)
    
    def _identify_phase(self, t: int, t_event: int = 0) -> str:
        """Identify market phase"""
        if t < t_event - 5:
            return 'pre_event_early'
        elif t_event - 5 <= t <= t_event:
            return 'pre_event_late'
        elif t_event < t <= t_event + self.vol_model.delta:
            return 'post_event_rising'
        else:
            return 'post_event_decay'

class RiskMetrics:
    """
    Calculate risk-adjusted return metrics for hypothesis testing.
    Implements metrics from Section 4 of the paper.
    """
    
    @staticmethod
    def return_to_variance_ratio(expected_return: float, risk_free_rate: float,
                                volatility: float, transaction_cost: float = 0) -> float:
        """
        Calculate Return-to-Variance Ratio (RVR).
        RVR = (E[R] - r_f - tau) / sigma^2
        
        Key metric for Hypothesis 1.
        """
        excess_return = expected_return - risk_free_rate - transaction_cost
        if volatility > 0:
            return excess_return / (volatility ** 2)
        else:
            return 0
    
    @staticmethod
    def sharpe_ratio(expected_return: float, risk_free_rate: float,
                    volatility: float, transaction_cost: float = 0) -> float:
        """
        Calculate Sharpe Ratio.
        SR = (E[R] - r_f - tau) / sigma
        """
        excess_return = expected_return - risk_free_rate - transaction_cost
        if volatility > 0:
            return excess_return / volatility
        else:
            return 0
    
    @staticmethod
    def calculate_phase_metrics(data: pd.DataFrame, risk_free_rate: float = 0.00018) -> pd.DataFrame:
        """
        Calculate average risk-adjusted metrics by event phase.
        Used for testing Hypothesis 1 about RVR/Sharpe peaks.
        """
        phases = ['pre_event_early', 'pre_event_late', 'post_event_rising', 'post_event_decay']
        results = []
        
        for phase in phases:
            phase_data = data[data['phase'] == phase] if 'phase' in data.columns else data
            
            if len(phase_data) > 0:
                # Use expected_return if available, otherwise use returns
                return_col = 'expected_return' if 'expected_return' in phase_data.columns else 'returns'
                vol_col = 'unified_volatility' if 'unified_volatility' in phase_data.columns else 'volatility'
                
                avg_return = phase_data[return_col].mean()
                avg_volatility = phase_data[vol_col].mean()
                
                # Calculate metrics
                rvr = RiskMetrics.return_to_variance_ratio(
                    avg_return, risk_free_rate, avg_volatility
                )
                
                sharpe = RiskMetrics.sharpe_ratio(
                    avg_return, risk_free_rate, avg_volatility
                )
                
                # Additional statistics
                results.append({
                    'phase': phase,
                    'avg_return': avg_return,
                    'avg_volatility': avg_volatility,
                    'return_volatility': phase_data[return_col].std(),
                    'rvr': rvr,
                    'sharpe_ratio': sharpe,
                    'n_obs': len(phase_data),
                    'median_return': phase_data[return_col].median(),
                    'return_skewness': phase_data[return_col].skew(),
                    'return_kurtosis': phase_data[return_col].kurtosis()
                })
        
        return pd.DataFrame(results)
    
    @staticmethod
    def test_rvr_peak_hypothesis(phase_metrics: pd.DataFrame, 
                                significance_level: float = 0.05) -> Dict[str, bool]:
        """
        Test Hypothesis 1: RVR and Sharpe ratios peak during post-event rising phase.
        
        Returns:
            Dictionary with test results for RVR and Sharpe ratio
        """
        results = {}
        
        # Get metrics for each phase
        pre_early = phase_metrics[phase_metrics['phase'] == 'pre_event_early']
        pre_late = phase_metrics[phase_metrics['phase'] == 'pre_event_late']
        post_rising = phase_metrics[phase_metrics['phase'] == 'post_event_rising']
        post_decay = phase_metrics[phase_metrics['phase'] == 'post_event_decay']
        
        # Test for RVR
        if len(post_rising) > 0:
            rvr_rising = post_rising['rvr'].iloc[0]
            
            # Check if RVR in rising phase exceeds other phases
            rvr_peak = True
            for phase_data in [pre_early, pre_late, post_decay]:
                if len(phase_data) > 0 and phase_data['rvr'].iloc[0] >= rvr_rising:
                    rvr_peak = False
                    break
            
            results['rvr_peak_supported'] = rvr_peak
            results['rvr_post_rising'] = rvr_rising
        else:
            results['rvr_peak_supported'] = False
            results['rvr_post_rising'] = None
        
        # Test for Sharpe ratio
        if len(post_rising) > 0:
            sharpe_rising = post_rising['sharpe_ratio'].iloc[0]
            
            # Check if Sharpe in rising phase exceeds other phases
            sharpe_peak = True
            for phase_data in [pre_early, pre_late, post_decay]:
                if len(phase_data) > 0 and phase_data['sharpe_ratio'].iloc[0] >= sharpe_rising:
                    sharpe_peak = False
                    break
            
            results['sharpe_peak_supported'] = sharpe_rising
            results['sharpe_post_rising'] = sharpe_rising
        else:
            results['sharpe_peak_supported'] = False
            results['sharpe_post_rising'] = None
        
        return results
