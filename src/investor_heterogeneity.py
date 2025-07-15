# src/investor_heterogeneity.py
"""
Heterogeneous Investor Framework
Implements the three investor types from the theoretical model:
- Informed investors
- Uninformed investors  
- Liquidity traders
"""

import numpy as np
import polars as pl
from typing import Dict, List, Optional, Tuple, Union
from abc import ABC, abstractmethod
from dataclasses import dataclass
import warnings
from scipy.optimize import minimize
from scipy import stats

@dataclass
class InvestorParameters:
    """Parameters for different investor types"""
    gamma_T: float  # Terminal wealth risk aversion
    gamma_V: float  # Real-time variance aversion
    b0: float       # Baseline bias parameter
    kappa: float    # Bias sensitivity to volatility
    tau_b: float    # Transaction cost for purchases
    tau_s: float    # Transaction cost for sales
    information_quality: float  # Quality of information access (0-1)
    constraint_factor: float = 1.0  # Position constraint factor

class BaseInvestor(ABC):
    """Abstract base class for investor types"""
    
    def __init__(self, params: InvestorParameters, investor_id: str):
        self.params = params
        self.investor_id = investor_id
        self.portfolio_history = []
        self.performance_metrics = {}
        
    @abstractmethod
    def form_expectations(self, 
                         data: pl.DataFrame, 
                         volatility: np.ndarray,
                         information: Dict) -> np.ndarray:
        """Form expectations about future returns"""
        pass
    
    @abstractmethod
    def optimize_portfolio(self, 
                          expected_returns: np.ndarray,
                          volatility: np.ndarray,
                          current_weights: np.ndarray) -> np.ndarray:
        """Optimize portfolio weights"""
        pass
    
    def calculate_transaction_costs(self, 
                                  weight_changes: np.ndarray,
                                  wealth: float) -> float:
        """Calculate transaction costs for portfolio changes"""
        costs = 0.0
        for dw in weight_changes:
            if dw > 0:  # Purchase
                costs += self.params.tau_b * abs(dw) * wealth
            elif dw < 0:  # Sale
                costs += self.params.tau_s * abs(dw) * wealth
        return costs
    
    def update_performance(self, returns: float, transaction_costs: float):
        """Update performance tracking"""
        net_return = returns - transaction_costs
        self.performance_metrics.setdefault('net_returns', []).append(net_return)
        self.performance_metrics.setdefault('transaction_costs', []).append(transaction_costs)

class InformedInvestor(BaseInvestor):
    """
    Informed investors with accurate information and lower bias
    """
    
    def __init__(self, params: Optional[InvestorParameters] = None):
        if params is None:
            params = InvestorParameters(
                gamma_T=2.0,      # Moderate risk aversion
                gamma_V=0.5,      # Lower real-time variance aversion
                b0=0.005,         # Low baseline bias (0.5%)
                kappa=0.8,        # Moderate bias sensitivity
                tau_b=0.002,      # 20 bps purchase cost
                tau_s=0.001,      # 10 bps sale cost
                information_quality=0.9  # High information quality
            )
        super().__init__(params, "informed")
        
    def form_expectations(self, 
                         data: pl.DataFrame, 
                         volatility: np.ndarray,
                         information: Dict) -> np.ndarray:
        """
        Form expectations with high-quality information and low bias
        """
        # Get historical returns for baseline
        returns = data.get_column('ret').to_numpy()
        days_to_event = data.get_column('days_to_event').to_numpy()
        
        # Informed investors have access to better information
        information_signal = information.get('information_signal', np.zeros_like(returns))
        
        # Form expectations with bias adjustment from paper (Equation 7-8)
        expected_returns = np.zeros_like(returns)
        
        for i, day in enumerate(days_to_event):
            # Base expectation from historical mean
            hist_mean = np.mean(returns[max(0, i-20):i+1]) if i > 0 else 0
            
            # Information advantage
            info_adjustment = self.params.information_quality * information_signal[i]
            
            # Bias component (lower for informed investors)
            bias_factor = self._calculate_bias_factor(day, volatility[i] if i < len(volatility) else 0)
            bias_adjustment = self.params.b0 * bias_factor
            
            expected_returns[i] = hist_mean + info_adjustment + bias_adjustment
            
        return expected_returns
    
    def _calculate_bias_factor(self, days_to_event: int, current_volatility: float) -> float:
        """Calculate bias factor based on event timing and volatility"""
        # Post-event rising phase bias (days 0-5)
        if 0 <= days_to_event <= 5:
            time_factor = np.exp(-days_to_event / 3.0)  # Decay over time
            vol_factor = self.params.kappa * (current_volatility - 1.0) if current_volatility > 1.0 else 0
            return time_factor * (1 + vol_factor)
        else:
            return 0.1  # Minimal bias outside rising phase
    
    def optimize_portfolio(self, 
                          expected_returns: np.ndarray,
                          volatility: np.ndarray,
                          current_weights: np.ndarray) -> np.ndarray:
        """
        Optimize portfolio using mean-variance framework with real-time constraints
        """
        T = len(expected_returns)
        optimal_weights = np.zeros((T, 3))  # [event_asset, generic_asset, risk_free]
        
        for t in range(T):
            # Current period optimization
            mu_e = expected_returns[t]
            sigma_e = volatility[t] if t < len(volatility) else np.std(expected_returns[:t+1])
            
            # Generic asset parameters (assumed constant)
            mu_g = 0.0001  # Small positive expected return
            sigma_g = 0.02  # 2% daily volatility
            rho = 0.3      # Correlation with event asset
            r_f = 0.0      # Risk-free rate
            
            # Calculate optimal weights using analytical solution from paper
            gamma_total = self.params.gamma_T + self.params.gamma_V
            
            # Event asset weight (Equation 14)
            numerator_e = mu_e - r_f
            denominator_e = gamma_total * (sigma_e**2 - rho * sigma_e * sigma_g)
            w_e_star = numerator_e / denominator_e if abs(denominator_e) > 1e-8 else 0
            
            # Generic asset weight (Equation 15)  
            numerator_g = mu_g - r_f
            denominator_g = gamma_total * (sigma_g**2 - rho * sigma_e * sigma_g)
            cross_term = (rho * sigma_e * sigma_g * (mu_e - r_f)) / (sigma_e**2 * (sigma_g**2 - rho * sigma_e * sigma_g))
            w_g_star = numerator_g / denominator_g - cross_term if abs(denominator_g) > 1e-8 else 0
            
            # Risk-free asset weight
            w_f_star = 1 - w_e_star - w_g_star
            
            # Apply constraints and transaction cost considerations
            if t > 0:
                # Consider transaction costs in optimization
                prev_weights = optimal_weights[t-1]
                weight_changes = np.array([w_e_star, w_g_star, w_f_star]) - prev_weights
                
                # Adjust for transaction costs (simplified)
                tc_penalty = np.sum(np.abs(weight_changes) * 
                                   [self.params.tau_b if dw > 0 else self.params.tau_s for dw in weight_changes])
                
                # Reduce position changes if transaction costs are high
                if tc_penalty > 0.01:  # 1% threshold
                    weight_changes *= 0.5
                    w_e_star = prev_weights[0] + weight_changes[0]
                    w_g_star = prev_weights[1] + weight_changes[1]
                    w_f_star = 1 - w_e_star - w_g_star
            
            # Store optimal weights
            optimal_weights[t] = [w_e_star, w_g_star, w_f_star]
        
        return optimal_weights

class UninformedInvestor(BaseInvestor):
    """
    Uninformed investors with limited information and higher bias
    """
    
    def __init__(self, params: Optional[InvestorParameters] = None):
        if params is None:
            params = InvestorParameters(
                gamma_T=3.0,      # Higher risk aversion
                gamma_V=1.0,      # Higher real-time variance aversion
                b0=0.02,          # Higher baseline bias (2%)
                kappa=1.5,        # Higher bias sensitivity
                tau_b=0.003,      # 30 bps purchase cost
                tau_s=0.0015,     # 15 bps sale cost
                information_quality=0.3  # Low information quality
            )
        super().__init__(params, "uninformed")
        
    def form_expectations(self, 
                         data: pl.DataFrame, 
                         volatility: np.ndarray,
                         information: Dict) -> np.ndarray:
        """
        Form expectations with limited information and higher bias
        """
        returns = data.get_column('ret').to_numpy()
        days_to_event = data.get_column('days_to_event').to_numpy()
        
        # Uninformed investors get noisy information signals
        information_signal = information.get('information_signal', np.zeros_like(returns))
        noise = np.random.normal(0, 0.01, len(information_signal))  # Add noise
        noisy_signal = self.params.information_quality * (information_signal + noise)
        
        expected_returns = np.zeros_like(returns)
        
        for i, day in enumerate(days_to_event):
            # Base expectation (uses shorter history due to limited analysis capability)
            hist_mean = np.mean(returns[max(0, i-10):i+1]) if i > 0 else 0
            
            # Limited information advantage
            info_adjustment = noisy_signal[i]
            
            # Higher bias component
            bias_factor = self._calculate_bias_factor(day, volatility[i] if i < len(volatility) else 0)
            bias_adjustment = self.params.b0 * bias_factor
            
            expected_returns[i] = hist_mean + info_adjustment + bias_adjustment
            
        return expected_returns
    
    def _calculate_bias_factor(self, days_to_event: int, current_volatility: float) -> float:
        """Calculate bias factor with higher sensitivity for uninformed investors"""
        # Stronger bias in post-event rising phase
        if 0 <= days_to_event <= 5:
            time_factor = np.exp(-days_to_event / 2.0)  # Slower decay
            vol_factor = self.params.kappa * (current_volatility - 1.0) if current_volatility > 1.0 else 0
            return time_factor * (1 + vol_factor) * 1.5  # 50% amplification
        else:
            return 0.3  # Higher baseline bias
    
    def optimize_portfolio(self, 
                          expected_returns: np.ndarray,
                          volatility: np.ndarray,
                          current_weights: np.ndarray) -> np.ndarray:
        """
        Portfolio optimization with higher risk aversion and simplified approach
        """
        T = len(expected_returns)
        optimal_weights = np.zeros((T, 3))
        
        for t in range(T):
            mu_e = expected_returns[t]
            sigma_e = volatility[t] if t < len(volatility) else np.std(expected_returns[:t+1])
            
            # More conservative parameters
            mu_g = 0.00005  # Lower expected return assumption
            sigma_g = 0.02
            rho = 0.3
            r_f = 0.0
            
            # Higher risk aversion
            gamma_total = self.params.gamma_T + self.params.gamma_V
            
            # Simplified optimization (less sophisticated than informed investors)
            if sigma_e > 0:
                # Simple Sharpe ratio based allocation
                sharpe_e = (mu_e - r_f) / sigma_e
                sharpe_g = (mu_g - r_f) / sigma_g
                
                # Conservative position sizing
                w_e_star = min(0.3, max(-0.1, sharpe_e / gamma_total))  # Cap at 30%
                w_g_star = min(0.4, max(0, sharpe_g / gamma_total))     # Cap at 40%
                w_f_star = 1 - w_e_star - w_g_star
            else:
                # Default conservative allocation
                w_e_star = 0.1
                w_g_star = 0.2
                w_f_star = 0.7
            
            optimal_weights[t] = [w_e_star, w_g_star, w_f_star]
        
        return optimal_weights

class LiquidityTrader(BaseInvestor):
    """
    Liquidity traders with non-information based trading motives
    """
    
    def __init__(self, params: Optional[InvestorParameters] = None):
        if params is None:
            params = InvestorParameters(
                gamma_T=1.5,      # Lower risk aversion (more willing to trade)
                gamma_V=0.3,      # Lower real-time variance aversion
                b0=0.0,           # No information-based bias
                kappa=0.0,        # No bias sensitivity
                tau_b=0.001,      # Lower transaction costs (better execution)
                tau_s=0.0005,     # Lower sale costs
                information_quality=0.0,  # No information advantage
                constraint_factor=0.8     # 20% tighter constraints
            )
        super().__init__(params, "liquidity")
        self.liquidity_needs = {}
        
    def set_liquidity_needs(self, 
                           funding_schedule: Dict[int, float],
                           emergency_factor: float = 1.2):
        """
        Set exogenous liquidity needs
        funding_schedule: {days_to_event: funding_need}
        """
        self.liquidity_needs = funding_schedule
        self.emergency_factor = emergency_factor
        
    def form_expectations(self, 
                         data: pl.DataFrame, 
                         volatility: np.ndarray,
                         information: Dict) -> np.ndarray:
        """
        Form simple expectations without information advantage
        """
        returns = data.get_column('ret').to_numpy()
        
        # Simple moving average expectation
        expected_returns = np.zeros_like(returns)
        for i in range(len(returns)):
            if i > 5:
                expected_returns[i] = np.mean(returns[i-5:i])
            else:
                expected_returns[i] = 0
                
        return expected_returns
    
    def optimize_portfolio(self, 
                          expected_returns: np.ndarray,
                          volatility: np.ndarray,
                          current_weights: np.ndarray) -> np.ndarray:
        """
        Portfolio optimization driven by liquidity needs rather than return expectations
        """
        T = len(expected_returns)
        optimal_weights = np.zeros((T, 3))
        days_to_event = np.arange(-T//2, T//2 + T%2)  # Approximate event timing
        
        for t in range(T):
            day = days_to_event[t] if t < len(days_to_event) else 0
            
            # Check for liquidity needs
            liquidity_need = self.liquidity_needs.get(day, 0.0)
            
            # Base allocation without liquidity constraints
            mu_e = expected_returns[t]
            sigma_e = volatility[t] if t < len(volatility) else 0.02
            
            if sigma_e > 0:
                # Simple risk-adjusted allocation
                risk_capacity = 1.0 / (1.0 + self.params.gamma_T * sigma_e**2)
                w_e_base = 0.2 * risk_capacity  # Conservative base allocation
                w_g_base = 0.3 * risk_capacity
            else:
                w_e_base = 0.2
                w_g_base = 0.3
            
            # Adjust for liquidity needs
            if liquidity_need > 0:
                # Need to raise cash - reduce risky positions
                reduction_factor = min(1.0, liquidity_need * self.emergency_factor)
                w_e_star = w_e_base * (1 - reduction_factor)
                w_g_star = w_g_base * (1 - reduction_factor)
                w_f_star = 1 - w_e_star - w_g_star
            elif liquidity_need < 0:
                # Excess cash - can increase positions
                increase_factor = min(0.5, abs(liquidity_need))
                w_e_star = min(0.4, w_e_base * (1 + increase_factor))
                w_g_star = min(0.5, w_g_base * (1 + increase_factor))
                w_f_star = 1 - w_e_star - w_g_star
            else:
                # No special liquidity needs
                w_e_star = w_e_base
                w_g_star = w_g_base
                w_f_star = 1 - w_e_star - w_g_star
            
            # Apply constraint factor
            w_e_star *= self.params.constraint_factor
            w_g_star *= self.params.constraint_factor
            w_f_star = 1 - w_e_star - w_g_star
            
            optimal_weights[t] = [w_e_star, w_g_star, w_f_star]
        
        return optimal_weights

class HeterogeneousInvestorMarket:
    """
    Market with heterogeneous investors
    """
    
    def __init__(self, 
                 investor_proportions: Optional[Dict[str, float]] = None,
                 total_wealth: float = 1.0):
        
        # Default proportions if not specified
        if investor_proportions is None:
            investor_proportions = {
                'informed': 0.2,      # 20% informed
                'uninformed': 0.6,    # 60% uninformed  
                'liquidity': 0.2      # 20% liquidity traders
            }
            
        self.proportions = investor_proportions
        self.total_wealth = total_wealth
        
        # Initialize investor types
        self.investors = {
            'informed': InformedInvestor(),
            'uninformed': UninformedInvestor(), 
            'liquidity': LiquidityTrader()
        }
        
        # Market state
        self.market_prices = []
        self.aggregate_demand = []
        self.market_clearing_prices = []
        
    def simulate_market(self, 
                       data: pl.DataFrame,
                       volatility: np.ndarray,
                       information_signals: Dict,
                       asset_supply: float = 1.0) -> Dict:
        """
        Simulate market with heterogeneous investors
        """
        T = len(data)
        
        # Get investor expectations and demands
        investor_demands = {}
        investor_weights = {}
        
        for investor_type, investor in self.investors.items():
            if investor_type in self.proportions:
                # Form expectations
                expectations = investor.form_expectations(data, volatility, information_signals)
                
                # Optimize portfolio  
                current_weights = np.array([0.1, 0.2, 0.7])  # Initial allocation
                optimal_weights = investor.optimize_portfolio(expectations, volatility, current_weights)
                
                investor_weights[investor_type] = optimal_weights
                
                # Calculate aggregate demand for event asset
                wealth_share = self.proportions[investor_type] * self.total_wealth
                demand = optimal_weights[:, 0] * wealth_share  # Event asset demand
                investor_demands[investor_type] = demand
        
        # Calculate market clearing
        aggregate_demand = np.zeros(T)
        for investor_type, demand in investor_demands.items():
            aggregate_demand += demand
            
        # Simple price adjustment mechanism
        market_clearing_prices = np.zeros(T)
        excess_demand = aggregate_demand - asset_supply
        
        # Price adjustment based on excess demand
        for t in range(T):
            if t == 0:
                market_clearing_prices[t] = 1.0  # Normalized initial price
            else:
                # Price adjusts to clear market
                price_adjustment = 0.1 * excess_demand[t]  # Price elasticity
                market_clearing_prices[t] = market_clearing_prices[t-1] * (1 + price_adjustment)
        
        return {
            'investor_weights': investor_weights,
            'investor_demands': investor_demands,
            'aggregate_demand': aggregate_demand,
            'market_clearing_prices': market_clearing_prices,
            'excess_demand': excess_demand
        }
    
    def analyze_market_dynamics(self, simulation_results: Dict) -> Dict:
        """
        Analyze market dynamics and investor behavior
        """
        analysis = {}
        
        # Analyze investor behavior differences
        weights = simulation_results['investor_weights']
        
        for investor_type in weights.keys():
            investor_weights = weights[investor_type]
            analysis[investor_type] = {
                'avg_event_weight': np.mean(investor_weights[:, 0]),
                'avg_generic_weight': np.mean(investor_weights[:, 1]),
                'avg_rf_weight': np.mean(investor_weights[:, 2]),
                'weight_volatility': np.std(investor_weights[:, 0]),
                'max_event_weight': np.max(investor_weights[:, 0]),
                'min_event_weight': np.min(investor_weights[:, 0])
            }
        
        # Market-level analysis
        analysis['market'] = {
            'price_volatility': np.std(simulation_results['market_clearing_prices']),
            'avg_excess_demand': np.mean(simulation_results['excess_demand']),
            'demand_volatility': np.std(simulation_results['aggregate_demand'])
        }
        
        return analysis
    
    def plot_market_simulation(self, 
                              simulation_results: Dict,
                              data: pl.DataFrame,
                              results_dir: str, 
                              file_prefix: str):
        """
        Plot market simulation results
        """
        try:
            import matplotlib.pyplot as plt
            import os
            
            days_to_event = data.get_column('days_to_event').to_numpy()
            
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            
            # Plot 1: Investor portfolio weights
            weights = simulation_results['investor_weights']
            for investor_type, investor_weights in weights.items():
                axes[0, 0].plot(days_to_event, investor_weights[:, 0], 
                               label=f'{investor_type.title()} Investor', linewidth=2)
            
            axes[0, 0].axvline(x=0, color='black', linestyle='--', alpha=0.5)
            axes[0, 0].set_xlabel('Days to Event')
            axes[0, 0].set_ylabel('Event Asset Weight')
            axes[0, 0].set_title('Portfolio Allocation by Investor Type')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
            
            # Plot 2: Market clearing prices
            axes[0, 1].plot(days_to_event, simulation_results['market_clearing_prices'], 
                           color='red', linewidth=2, label='Market Price')
            axes[0, 1].axvline(x=0, color='black', linestyle='--', alpha=0.5)
            axes[0, 1].set_xlabel('Days to Event')
            axes[0, 1].set_ylabel('Normalized Price')
            axes[0, 1].set_title('Market Clearing Prices')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
            
            # Plot 3: Aggregate demand vs supply
            axes[1, 0].plot(days_to_event, simulation_results['aggregate_demand'], 
                           label='Aggregate Demand', color='blue', linewidth=2)
            axes[1, 0].axhline(y=1.0, color='red', linestyle='--', label='Asset Supply')
            axes[1, 0].axvline(x=0, color='black', linestyle='--', alpha=0.5)
            axes[1, 0].set_xlabel('Days to Event')
            axes[1, 0].set_ylabel('Quantity')
            axes[1, 0].set_title('Aggregate Demand vs Supply')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
            
            # Plot 4: Excess demand
            axes[1, 1].plot(days_to_event, simulation_results['excess_demand'], 
                           color='orange', linewidth=2)
            axes[1, 1].axhline(y=0, color='black', linestyle='-', alpha=0.5)
            axes[1, 1].axvline(x=0, color='black', linestyle='--', alpha=0.5)
            axes[1, 1].set_xlabel('Days to Event')
            axes[1, 1].set_ylabel('Excess Demand')
            axes[1, 1].set_title('Market Excess Demand')
            axes[1, 1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plot_filename = os.path.join(results_dir, f"{file_prefix}_heterogeneous_market.png")
            plt.savefig(plot_filename, dpi=200, bbox_inches='tight')
            plt.close()
            
            print(f"Market simulation plot saved to: {plot_filename}")
            
        except Exception as e:
            warnings.warn(f"Could not create market simulation plot: {e}")