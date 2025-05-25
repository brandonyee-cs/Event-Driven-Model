"""
Theoretical Models Module - Unified Asset Pricing Framework

This module implements the theoretical models from the paper, including:
- Multi-period portfolio optimization with transaction costs
- Equilibrium price dynamics with heterogeneous investors
- Unified volatility dynamics (GJR-GARCH + event adjustments)
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass
from scipy.optimize import minimize_scalar, brentq
from scipy.stats import norm
import warnings

warnings.filterwarnings('ignore')

@dataclass
class InvestorParams:
    """Parameters for different investor types"""
    risk_aversion: float  # gamma
    bias_baseline: float  # b_0
    bias_sensitivity: float  # kappa
    transaction_cost_buy: float  # tau_b
    transaction_cost_sell: float  # tau_s
    liquidity_constraint: float  # lambda for liquidity traders
    information_quality: float  # precision of information

@dataclass
class MarketParams:
    """Market-wide parameters"""
    risk_free_rate: float  # r_t
    generic_return: float  # mu_g
    generic_volatility: float  # sigma_g
    correlation: float  # rho
    event_asset_supply: float  # S_e
    
class UnifiedVolatilityModel:
    """
    Implements the unified volatility model from the paper.
    Combines GJR-GARCH baseline with event-specific adjustments.
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
        
        # Check stationarity condition
        if self.alpha + self.beta + self.gamma/2 >= 1:
            warnings.warn("GJR-GARCH parameters violate stationarity condition")
    
    def baseline_volatility(self, returns: np.array) -> np.array:
        """
        Calculate baseline volatility using GJR-GARCH(1,1).
        h_t = omega + alpha * eps_{t-1}^2 + beta * h_{t-1} + gamma * I_{t-1} * eps_{t-1}^2
        """
        T = len(returns)
        h = np.zeros(T)
        
        # Initialize with unconditional variance
        h[0] = self.omega / (1 - self.alpha - self.beta - self.gamma/2)
        
        for t in range(1, T):
            eps_prev = returns[t-1] / np.sqrt(h[t-1])
            indicator = 1 if eps_prev < 0 else 0
            
            h[t] = (self.omega + 
                   self.alpha * eps_prev**2 * h[t-1] +
                   self.beta * h[t-1] +
                   self.gamma * indicator * eps_prev**2 * h[t-1])
        
        return np.sqrt(h)
    
    def phi_function(self, t: int, t_event: int = 0) -> float:
        """
        Calculate phi adjustment factor based on event phase.
        """
        if t <= t_event:
            # Pre-event phase (phi_1)
            phi = (self.k1 - 1) * np.exp(-((t - t_event)**2) / (2 * self.delta_t1**2))
        elif t_event < t <= t_event + self.delta:
            # Post-event rising phase (phi_2)
            phi = (self.k2 - 1) * (1 - np.exp(-(t - t_event) / self.delta_t2))
        else:
            # Post-event decay phase (phi_3)
            phi = (self.k2 - 1) * np.exp(-(t - t_event - self.delta) / self.delta_t3)
        
        return phi
    
    def unified_volatility(self, t: int, baseline_vol: float, t_event: int = 0) -> float:
        """
        Calculate unified volatility: sigma_e(t) = sqrt(h_t) * (1 + phi(t))
        """
        phi = self.phi_function(t, t_event)
        return baseline_vol * (1 + phi)
    
    def bias_parameter(self, t: int, baseline_vol: float, b0: float, kappa: float, 
                      t_event: int = 0) -> float:
        """
        Calculate time-varying bias parameter.
        b_t = b_0 * (1 + kappa * (Psi_t - 1)/(k_2 - 1)) for post-event rising phase
        """
        if t_event < t <= t_event + self.delta:
            phi = self.phi_function(t, t_event)
            psi_t = 1 + phi  # sigma_e(t) / sqrt(h_t)
            return b0 * (1 + kappa * (psi_t - 1) / (self.k2 - 1))
        else:
            return b0

class PortfolioOptimizer:
    """
    Implements multi-period portfolio optimization with transaction costs.
    """
    
    def __init__(self, investor_params: InvestorParams, market_params: MarketParams,
                 volatility_model: UnifiedVolatilityModel):
        """Initialize optimizer with investor and market parameters"""
        self.investor = investor_params
        self.market = market_params
        self.vol_model = volatility_model
    
    def optimal_weights(self, expected_return_e: float, volatility_e: float,
                       previous_weight_e: float, wealth: float) -> Tuple[float, float]:
        """
        Calculate optimal portfolio weights for event and generic assets.
        
        Returns:
            Tuple of (weight_event, weight_generic)
        """
        gamma = self.investor.risk_aversion
        rho = self.market.correlation
        sigma_g = self.market.generic_volatility
        
        # Determine transaction cost based on trading direction
        def objective(w_e):
            if w_e > previous_weight_e:
                tau = self.investor.transaction_cost_buy
            elif w_e < previous_weight_e:
                tau = self.investor.transaction_cost_sell
            else:
                tau = 0
            
            # Account for transaction costs in expected return
            adj_return_e = expected_return_e - tau * abs(w_e - previous_weight_e) * wealth
            
            # Calculate optimal weight for generic asset given w_e
            numerator_g = self.market.generic_return - self.market.risk_free_rate
            denominator_g = gamma * (sigma_g**2 - rho * volatility_e * sigma_g)
            
            term1_g = numerator_g / denominator_g
            term2_g = (rho * volatility_e * sigma_g * (adj_return_e - self.market.risk_free_rate)) / (sigma_g**2 * (volatility_e**2 - rho * volatility_e * sigma_g))
            
            w_g = term1_g - term2_g
            
            # Calculate portfolio expected return and variance
            w_rf = 1 - w_e - w_g
            
            exp_return = (w_e * adj_return_e + 
                         w_g * self.market.generic_return + 
                         w_rf * self.market.risk_free_rate)
            
            variance = (w_e**2 * volatility_e**2 + 
                       w_g**2 * sigma_g**2 + 
                       2 * w_e * w_g * rho * volatility_e * sigma_g)
            
            # Mean-variance utility
            utility = exp_return - (gamma / 2) * variance
            
            return -utility  # Negative for minimization
        
        # Optimize weight for event asset
        result = minimize_scalar(objective, bounds=(0, 1), method='bounded')
        w_e_optimal = result.x
        
        # Calculate corresponding generic asset weight
        if w_e_optimal > previous_weight_e:
            tau = self.investor.transaction_cost_buy
        elif w_e_optimal < previous_weight_e:
            tau = self.investor.transaction_cost_sell
        else:
            tau = 0
        
        adj_return_e = expected_return_e - tau * abs(w_e_optimal - previous_weight_e) * wealth
        
        numerator_g = self.market.generic_return - self.market.risk_free_rate
        denominator_g = gamma * (sigma_g**2 - rho * volatility_e * sigma_g)
        
        term1_g = numerator_g / denominator_g
        term2_g = (rho * volatility_e * sigma_g * (adj_return_e - self.market.risk_free_rate)) / (sigma_g**2 * (volatility_e**2 - rho * volatility_e * sigma_g))
        
        w_g_optimal = term1_g - term2_g
        
        return w_e_optimal, w_g_optimal
    
    def no_trade_region(self, previous_weight_e: float, volatility_e: float,
                       wealth: float) -> Tuple[float, float]:
        """
        Calculate the no-trade region boundaries.
        
        Returns:
            Tuple of (lower_bound, upper_bound) for event asset weight
        """
        gamma = self.investor.risk_aversion
        tau_b = self.investor.transaction_cost_buy
        tau_s = self.investor.transaction_cost_sell
        
        # Width of no-trade region
        width = (tau_b + tau_s) * wealth / (gamma * volatility_e**2)
        
        lower_bound = max(0, previous_weight_e - width/2)
        upper_bound = min(1, previous_weight_e + width/2)
        
        return lower_bound, upper_bound

class EquilibriumModel:
    """
    Implements market equilibrium with heterogeneous investors.
    """
    
    def __init__(self, market_params: MarketParams, volatility_model: UnifiedVolatilityModel):
        """Initialize equilibrium model"""
        self.market = market_params
        self.vol_model = volatility_model
        
        # Define investor types
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
        
        # Investor proportions
        self.n_informed = 0.3
        self.n_uninformed = 0.5
        self.n_liquidity = 0.2
    
    def aggregate_demand(self, price: float, t: int, baseline_vol: float,
                        information: float, previous_weights: Dict[str, float],
                        wealth: float, t_event: int = 0) -> float:
        """
        Calculate aggregate demand for the event asset at given price.
        """
        # Calculate unified volatility
        vol_e = self.vol_model.unified_volatility(t, baseline_vol, t_event)
        
        # Calculate expected return from price
        # Assuming next period price follows martingale plus drift
        expected_price_next = price * (1 + self.market.risk_free_rate)
        expected_return = (expected_price_next - price) / price
        
        total_demand = 0
        
        # Informed investor demand
        bias_informed = self.vol_model.bias_parameter(
            t, baseline_vol, self.informed_investor.bias_baseline,
            self.informed_investor.bias_sensitivity, t_event
        )
        exp_return_informed = expected_return + bias_informed * information
        
        optimizer_informed = PortfolioOptimizer(
            self.informed_investor, self.market, self.vol_model
        )
        w_e_informed, _ = optimizer_informed.optimal_weights(
            exp_return_informed, vol_e, previous_weights['informed'], wealth
        )
        
        demand_informed = self.n_informed * wealth * w_e_informed
        
        # Uninformed investor demand
        bias_uninformed = self.vol_model.bias_parameter(
            t, baseline_vol, self.uninformed_investor.bias_baseline,
            self.uninformed_investor.bias_sensitivity, t_event
        )
        # Uninformed have noisier information
        noisy_info = information + np.random.normal(0, 0.5)
        exp_return_uninformed = expected_return + bias_uninformed * noisy_info
        
        optimizer_uninformed = PortfolioOptimizer(
            self.uninformed_investor, self.market, self.vol_model
        )
        w_e_uninformed, _ = optimizer_uninformed.optimal_weights(
            exp_return_uninformed, vol_e, previous_weights['uninformed'], wealth
        )
        
        demand_uninformed = self.n_uninformed * wealth * w_e_uninformed
        
        # Liquidity trader demand (with constraints)
        optimizer_liquidity = PortfolioOptimizer(
            self.liquidity_trader, self.market, self.vol_model
        )
        w_e_liquidity, _ = optimizer_liquidity.optimal_weights(
            expected_return, vol_e, previous_weights['liquidity'], wealth
        )
        
        # Apply liquidity constraints pre-event
        if t < t_event and w_e_liquidity > previous_weights['liquidity']:
            # Reduce purchases by liquidity_constraint factor
            max_purchase = previous_weights['liquidity'] + (w_e_liquidity - previous_weights['liquidity']) * (1 - self.liquidity_trader.liquidity_constraint)
            w_e_liquidity = min(w_e_liquidity, max_purchase)
        
        demand_liquidity = self.n_liquidity * wealth * w_e_liquidity
        
        total_demand = demand_informed + demand_uninformed + demand_liquidity
        
        return total_demand
    
    def find_equilibrium_price(self, t: int, baseline_vol: float, information: float,
                             previous_weights: Dict[str, float], wealth: float,
                             t_event: int = 0) -> float:
        """
        Find equilibrium price where demand equals supply.
        """
        supply = self.market.event_asset_supply
        
        # Define excess demand function
        def excess_demand(price):
            if price <= 0:
                return float('inf')
            demand = self.aggregate_demand(
                price, t, baseline_vol, information, previous_weights, wealth, t_event
            )
            return demand - supply
        
        # Find equilibrium using bisection
        try:
            # Initial price bounds
            price_low = 0.1
            price_high = 1000.0
            
            # Ensure bounds bracket the solution
            while excess_demand(price_low) < 0:
                price_low /= 2
            while excess_demand(price_high) > 0:
                price_high *= 2
            
            # Find equilibrium
            eq_price = brentq(excess_demand, price_low, price_high)
            
            return eq_price
            
        except:
            # Fallback to simple approximation
            return 100.0  # Default price
    
    def simulate_price_path(self, T: int, t_event: int, baseline_vol: float,
                          information: float = 1.0, wealth: float = 1000000.0) -> pd.DataFrame:
        """
        Simulate equilibrium price path around an event.
        """
        prices = []
        weights = {
            'informed': 0.1,
            'uninformed': 0.1,
            'liquidity': 0.1
        }
        
        for t in range(-T, T+1):
            # Find equilibrium price
            price = self.find_equilibrium_price(
                t, baseline_vol, information, weights, wealth, t_event
            )
            
            # Update weights based on new price
            # (simplified - in reality would track actual trades)
            weights['informed'] *= 0.95
            weights['uninformed'] *= 0.95
            weights['liquidity'] *= 0.95
            
            # Store results
            prices.append({
                'time': t,
                'price': price,
                'volatility': self.vol_model.unified_volatility(t, baseline_vol, t_event),
                'phase': self._identify_phase(t, t_event)
            })
        
        return pd.DataFrame(prices)
    
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
    Calculate risk-adjusted return metrics.
    """
    
    @staticmethod
    def return_to_variance_ratio(expected_return: float, risk_free_rate: float,
                                volatility: float, transaction_cost: float = 0) -> float:
        """
        Calculate Return-to-Variance Ratio (RVR).
        RVR = (E[R] - r_f - tau) / sigma^2
        """
        excess_return = expected_return - risk_free_rate - transaction_cost
        return excess_return / (volatility ** 2)
    
    @staticmethod
    def sharpe_ratio(expected_return: float, risk_free_rate: float,
                    volatility: float, transaction_cost: float = 0) -> float:
        """
        Calculate Sharpe Ratio.
        SR = (E[R] - r_f - tau) / sigma
        """
        excess_return = expected_return - risk_free_rate - transaction_cost
        return excess_return / volatility
    
    @staticmethod
    def analyze_phase_metrics(returns: pd.DataFrame, risk_free_rate: float = 0.00018) -> pd.DataFrame:
        """
        Analyze risk-adjusted metrics by event phase.
        """
        results = []
        
        for phase in ['pre_event_early', 'pre_event_late', 'post_event_rising', 'post_event_decay']:
            phase_data = returns[returns['phase'] == phase]
            
            if len(phase_data) > 0:
                avg_return = phase_data['expected_return'].mean()
                avg_volatility = phase_data['unified_volatility'].mean()
                
                rvr = RiskMetrics.return_to_variance_ratio(
                    avg_return, risk_free_rate, avg_volatility
                )
                
                sharpe = RiskMetrics.sharpe_ratio(
                    avg_return, risk_free_rate, avg_volatility
                )
                
                results.append({
                    'phase': phase,
                    'avg_return': avg_return,
                    'avg_volatility': avg_volatility,
                    'rvr': rvr,
                    'sharpe_ratio': sharpe,
                    'n_obs': len(phase_data)
                })
        
        return pd.DataFrame(results)