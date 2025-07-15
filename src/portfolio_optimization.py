# src/portfolio_optimization.py
"""
Enhanced Portfolio Optimization Framework
Implements the continuous-time mean-variance optimization from the theoretical model
with real-time variance penalties and transaction costs
"""

import numpy as np
import polars as pl
from typing import Dict, List, Optional, Tuple, Callable, Union
from scipy.optimize import minimize, differential_evolution
from scipy.integrate import solve_ivp
import warnings
from dataclasses import dataclass
from abc import ABC, abstractmethod

@dataclass
class OptimizationParameters:
    """Parameters for portfolio optimization"""
    gamma_T: float = 2.0        # Terminal wealth risk aversion
    gamma_V: float = 0.5        # Real-time variance aversion  
    risk_free_rate: float = 0.0 # Risk-free rate
    correlation_eg: float = 0.3 # Correlation between event and generic assets
    sigma_g: float = 0.02       # Generic asset volatility
    mu_g: float = 0.0001        # Generic asset expected return
    transaction_cost_penalty: float = 1000.0  # Penalty for transaction costs
    
@dataclass
class AssetParameters:
    """Parameters for individual assets"""
    expected_return: np.ndarray
    volatility: np.ndarray
    transaction_costs: Dict[str, float]  # {'buy': rate, 'sell': rate}
    
class PortfolioOptimizer(ABC):
    """Abstract base class for portfolio optimizers"""
    
    @abstractmethod
    def optimize(self, 
                asset_params: Dict[str, AssetParameters],
                optimization_params: OptimizationParameters,
                initial_weights: np.ndarray,
                constraints: Optional[Dict] = None) -> Dict:
        pass

class ContinuousTimeOptimizer(PortfolioOptimizer):
    """
    Continuous-time portfolio optimizer implementing the Hamilton-Jacobi-Bellman approach
    """
    
    def __init__(self, time_steps: int = 100):
        self.time_steps = time_steps
        self.optimization_history = []
        
    def optimize(self,
                asset_params: Dict[str, AssetParameters], 
                optimization_params: OptimizationParameters,
                initial_weights: np.ndarray,
                constraints: Optional[Dict] = None) -> Dict:
        """
        Solve the continuous-time portfolio optimization problem
        """
        # Extract asset parameters
        event_asset = asset_params['event_asset']
        generic_asset = asset_params.get('generic_asset')
        
        T = len(event_asset.expected_return)
        
        # Set up time grid
        t_grid = np.linspace(0, 1, T)  # Normalized time [0,1]
        
        # Initialize optimization results
        optimal_weights = np.zeros((T, 3))  # [event, generic, risk_free]
        value_function = np.zeros(T)
        
        # Solve backwards in time (dynamic programming)
        for i in range(T-1, -1, -1):
            t = t_grid[i]
            
            # Current asset parameters
            mu_e = event_asset.expected_return[i]
            sigma_e = event_asset.volatility[i]
            mu_g = optimization_params.mu_g
            sigma_g = optimization_params.sigma_g
            rho = optimization_params.correlation_eg
            r_f = optimization_params.risk_free_rate
            
            # Previous weights for transaction cost calculation
            prev_weights = optimal_weights[i+1] if i < T-1 else initial_weights
            
            # Solve single-period optimization
            result = self._solve_single_period(
                mu_e, sigma_e, mu_g, sigma_g, rho, r_f,
                optimization_params, event_asset.transaction_costs,
                prev_weights, constraints
            )
            
            optimal_weights[i] = result['weights']
            value_function[i] = result['value']
        
        # Calculate portfolio metrics
        portfolio_returns = self._calculate_portfolio_returns(
            optimal_weights, asset_params, optimization_params
        )
        
        transaction_costs = self._calculate_transaction_costs(
            optimal_weights, asset_params
        )
        
        return {
            'optimal_weights': optimal_weights,
            'value_function': value_function,
            'portfolio_returns': portfolio_returns,
            'transaction_costs': transaction_costs,
            'optimization_params': optimization_params
        }
    
    def _solve_single_period(self,
                           mu_e: float, sigma_e: float,
                           mu_g: float, sigma_g: float, 
                           rho: float, r_f: float,
                           params: OptimizationParameters,
                           tc_rates: Dict[str, float],
                           prev_weights: np.ndarray,
                           constraints: Optional[Dict] = None) -> Dict:
        """
        Solve single-period optimization problem
        """
        gamma_total = params.gamma_T + params.gamma_V
        
        # Analytical solution when possible (no transaction costs or constraints)
        if not constraints and tc_rates.get('buy', 0) == 0 and tc_rates.get('sell', 0) == 0:
            # Closed-form solution from Equations 14-15 in paper
            
            # Variance-covariance matrix components
            var_matrix = np.array([
                [sigma_e**2, rho * sigma_e * sigma_g],
                [rho * sigma_e * sigma_g, sigma_g**2]
            ])
            
            # Expected excess returns
            excess_returns = np.array([mu_e - r_f, mu_g - r_f])
            
            # Optimal risky weights
            if np.linalg.det(var_matrix) > 1e-8:
                risky_weights = np.linalg.solve(var_matrix, excess_returns) / gamma_total
                w_e_star = risky_weights[0]
                w_g_star = risky_weights[1]
            else:
                # Fallback for singular matrix
                w_e_star = (mu_e - r_f) / (gamma_total * sigma_e**2) if sigma_e > 0 else 0
                w_g_star = (mu_g - r_f) / (gamma_total * sigma_g**2) if sigma_g > 0 else 0
            
            w_f_star = 1 - w_e_star - w_g_star
            optimal_weights = np.array([w_e_star, w_g_star, w_f_star])
            
        else:
            # Numerical optimization with transaction costs and constraints
            optimal_weights = self._numerical_optimization(
                mu_e, sigma_e, mu_g, sigma_g, rho, r_f,
                params, tc_rates, prev_weights, constraints
            )
        
        # Calculate value function
        portfolio_return = (optimal_weights[0] * mu_e + 
                          optimal_weights[1] * mu_g + 
                          optimal_weights[2] * r_f)
        
        portfolio_variance = (optimal_weights[0]**2 * sigma_e**2 + 
                            optimal_weights[1]**2 * sigma_g**2 + 
                            2 * optimal_weights[0] * optimal_weights[1] * rho * sigma_e * sigma_g)
        
        # Transaction costs
        weight_changes = optimal_weights - prev_weights
        tc_cost = self._calculate_tc_cost(weight_changes, tc_rates)
        
        # Utility value
        value = portfolio_return - 0.5 * gamma_total * portfolio_variance - tc_cost
        
        return {
            'weights': optimal_weights,
            'value': value,
            'portfolio_return': portfolio_return,
            'portfolio_variance': portfolio_variance,
            'transaction_cost': tc_cost
        }
    
    def _numerical_optimization(self,
                              mu_e: float, sigma_e: float,
                              mu_g: float, sigma_g: float,
                              rho: float, r_f: float,
                              params: OptimizationParameters,
                              tc_rates: Dict[str, float],
                              prev_weights: np.ndarray,
                              constraints: Optional[Dict] = None) -> np.ndarray:
        """
        Numerical optimization for complex cases
        """
        def objective(weights):
            w_e, w_g = weights
            w_f = 1 - w_e - w_g
            
            # Portfolio return
            port_return = w_e * mu_e + w_g * mu_g + w_f * r_f
            
            # Portfolio variance
            port_var = (w_e**2 * sigma_e**2 + w_g**2 * sigma_g**2 + 
                       2 * w_e * w_g * rho * sigma_e * sigma_g)
            
            # Transaction costs
            weight_changes = np.array([w_e, w_g, w_f]) - prev_weights
            tc_cost = self._calculate_tc_cost(weight_changes, tc_rates)
            
            # Utility (negative for minimization)
            gamma_total = params.gamma_T + params.gamma_V
            utility = port_return - 0.5 * gamma_total * port_var - tc_cost
            
            return -utility
        
        # Default constraints
        bounds = [(-0.5, 1.5), (-0.5, 1.5)]  # Allow some leverage/short selling
        
        # Add custom constraints if provided
        constraint_list = []
        if constraints:
            if 'max_event_weight' in constraints:
                constraint_list.append({
                    'type': 'ineq',
                    'fun': lambda w: constraints['max_event_weight'] - w[0]
                })
            if 'min_event_weight' in constraints:
                constraint_list.append({
                    'type': 'ineq', 
                    'fun': lambda w: w[0] - constraints['min_event_weight']
                })
        
        # Initial guess (previous weights projected to 2D)
        x0 = prev_weights[:2]
        
        try:
            # Use multiple optimization methods for robustness
            result1 = minimize(objective, x0, method='SLSQP', bounds=bounds, 
                             constraints=constraint_list)
            
            # Try differential evolution for global optimization
            result2 = differential_evolution(objective, bounds, seed=42, maxiter=100)
            
            # Choose best result
            if result1.success and (not result2.success or result1.fun < result2.fun):
                optimal_2d = result1.x
            else:
                optimal_2d = result2.x
                
            # Construct full weight vector
            w_e_opt, w_g_opt = optimal_2d
            w_f_opt = 1 - w_e_opt - w_g_opt
            
            return np.array([w_e_opt, w_g_opt, w_f_opt])
            
        except Exception as e:
            warnings.warn(f"Optimization failed: {e}. Using analytical approximation.")
            # Fallback to analytical solution
            gamma_total = params.gamma_T + params.gamma_V
            w_e_star = (mu_e - r_f) / (gamma_total * sigma_e**2) if sigma_e > 0 else 0
            w_g_star = (mu_g - r_f) / (gamma_total * sigma_g**2) if sigma_g > 0 else 0
            w_f_star = 1 - w_e_star - w_g_star
            return np.array([w_e_star, w_g_star, w_f_star])
    
    def _calculate_tc_cost(self, weight_changes: np.ndarray, tc_rates: Dict[str, float]) -> float:
        """Calculate transaction costs"""
        total_cost = 0.0
        buy_rate = tc_rates.get('buy', 0.001)
        sell_rate = tc_rates.get('sell', 0.0005)
        
        for dw in weight_changes:
            if dw > 0:  # Buying
                total_cost += buy_rate * dw
            elif dw < 0:  # Selling
                total_cost += sell_rate * abs(dw)
                
        return total_cost
    
    def _calculate_portfolio_returns(self,
                                   weights: np.ndarray,
                                   asset_params: Dict[str, AssetParameters],
                                   opt_params: OptimizationParameters) -> np.ndarray:
        """Calculate portfolio returns over time"""
        T = weights.shape[0]
        portfolio_returns = np.zeros(T)
        
        event_asset = asset_params['event_asset']
        
        for t in range(T):
            w_e, w_g, w_f = weights[t]
            
            # Asset returns
            r_e = event_asset.expected_return[t]
            r_g = opt_params.mu_g
            r_f = opt_params.risk_free_rate
            
            portfolio_returns[t] = w_e * r_e + w_g * r_g + w_f * r_f
            
        return portfolio_returns
    
    def _calculate_transaction_costs(self,
                                   weights: np.ndarray,
                                   asset_params: Dict[str, AssetParameters]) -> np.ndarray:
        """Calculate transaction costs over time"""
        T = weights.shape[0]
        transaction_costs = np.zeros(T)
        
        event_tc = asset_params['event_asset'].transaction_costs
        
        for t in range(1, T):  # Start from t=1 since t=0 has no previous weights
            weight_changes = weights[t] - weights[t-1]
            transaction_costs[t] = self._calculate_tc_cost(weight_changes, event_tc)
            
        return transaction_costs

class RealTimeOptimizer(PortfolioOptimizer):
    """
    Real-time portfolio optimizer with emphasis on variance control
    """
    
    def __init__(self, variance_monitoring_frequency: int = 1):
        self.monitoring_freq = variance_monitoring_frequency
        self.variance_history = []
        
    def optimize(self,
                asset_params: Dict[str, AssetParameters],
                optimization_params: OptimizationParameters, 
                initial_weights: np.ndarray,
                constraints: Optional[Dict] = None) -> Dict:
        """
        Optimize with real-time variance monitoring
        """
        event_asset = asset_params['event_asset']
        T = len(event_asset.expected_return)
        
        optimal_weights = np.zeros((T, 3))
        realized_variance = np.zeros(T)
        variance_breaches = []
        
        # Variance threshold (dynamic based on event phases)
        base_var_threshold = 0.01  # 1% daily variance threshold
        
        for t in range(T):
            # Calculate expected variance for current period
            mu_e = event_asset.expected_return[t]
            sigma_e = event_asset.volatility[t]
            
            # Determine if we're in high-variance event period
            # (This could be enhanced with the two-risk framework)
            if abs(t - T//2) <= 2:  # Around event day
                var_threshold = base_var_threshold * 2.0  # Higher tolerance
            else:
                var_threshold = base_var_threshold
            
            # Single period optimization with variance constraint
            weights_t = self._optimize_with_variance_constraint(
                mu_e, sigma_e, optimization_params, var_threshold,
                initial_weights if t == 0 else optimal_weights[t-1],
                event_asset.transaction_costs
            )
            
            optimal_weights[t] = weights_t
            
            # Calculate realized variance
            w_e, w_g, w_f = weights_t
            realized_var = (w_e**2 * sigma_e**2 + 
                          w_g**2 * optimization_params.sigma_g**2 + 
                          2 * w_e * w_g * optimization_params.correlation_eg * 
                          sigma_e * optimization_params.sigma_g)
            
            realized_variance[t] = realized_var
            
            # Check for variance breaches
            if realized_var > var_threshold:
                variance_breaches.append({
                    'time': t,
                    'realized_variance': realized_var,
                    'threshold': var_threshold,
                    'breach_ratio': realized_var / var_threshold
                })
        
        # Calculate performance metrics
        portfolio_returns = self._calculate_portfolio_returns_rt(
            optimal_weights, asset_params, optimization_params
        )
        
        sharpe_ratio = self._calculate_sharpe_ratio(portfolio_returns, optimization_params)
        
        return {
            'optimal_weights': optimal_weights,
            'realized_variance': realized_variance,
            'variance_breaches': variance_breaches,
            'portfolio_returns': portfolio_returns,
            'sharpe_ratio': sharpe_ratio,
            'optimization_params': optimization_params
        }
    
    def _optimize_with_variance_constraint(self,
                                         mu_e: float, sigma_e: float,
                                         params: OptimizationParameters,
                                         var_threshold: float,
                                         prev_weights: np.ndarray,
                                         tc_rates: Dict[str, float]) -> np.ndarray:
        """
        Optimize with explicit variance constraint
        """
        def objective(weights):
            w_e, w_g = weights
            w_f = 1 - w_e - w_g
            
            # Portfolio expected return
            port_return = (w_e * mu_e + 
                          w_g * params.mu_g + 
                          w_f * params.risk_free_rate)
            
            # Transaction costs
            weight_changes = np.array([w_e, w_g, w_f]) - prev_weights
            tc_cost = sum(tc_rates.get('buy', 0.001) * max(0, dw) + 
                         tc_rates.get('sell', 0.0005) * max(0, -dw) 
                         for dw in weight_changes)
            
            # Maximize return minus transaction costs
            return -(port_return - tc_cost)
        
        def variance_constraint(weights):
            w_e, w_g = weights
            
            # Portfolio variance
            port_var = (w_e**2 * sigma_e**2 + 
                       w_g**2 * params.sigma_g**2 + 
                       2 * w_e * w_g * params.correlation_eg * 
                       sigma_e * params.sigma_g)
            
            # Constraint: variance <= threshold
            return var_threshold - port_var
        
        # Constraints
        constraints = [
            {'type': 'ineq', 'fun': variance_constraint},
            {'type': 'eq', 'fun': lambda w: 1 - w[0] - w[1] - (1 - w[0] - w[1])}  # Weights sum to 1
        ]
        
        bounds = [(-0.5, 1.5), (-0.5, 1.5)]
        x0 = prev_weights[:2]
        
        try:
            result = minimize(objective, x0, method='SLSQP', 
                            bounds=bounds, constraints=constraints)
            
            if result.success:
                w_e_opt, w_g_opt = result.x
                w_f_opt = 1 - w_e_opt - w_g_opt
                return np.array([w_e_opt, w_g_opt, w_f_opt])
            else:
                # Fallback: scale down positions to meet variance constraint
                return self._variance_scaling_fallback(prev_weights, sigma_e, params, var_threshold)
                
        except Exception as e:
            warnings.warn(f"Variance-constrained optimization failed: {e}")
            return self._variance_scaling_fallback(prev_weights, sigma_e, params, var_threshold)
    
    def _variance_scaling_fallback(self,
                                 prev_weights: np.ndarray,
                                 sigma_e: float,
                                 params: OptimizationParameters,
                                 var_threshold: float) -> np.ndarray:
        """
        Fallback method: scale down risky positions to meet variance constraint
        """
        w_e_prev, w_g_prev, w_f_prev = prev_weights
        
        # Calculate current variance
        current_var = (w_e_prev**2 * sigma_e**2 + 
                      w_g_prev**2 * params.sigma_g**2 + 
                      2 * w_e_prev * w_g_prev * params.correlation_eg * 
                      sigma_e * params.sigma_g)
        
        if current_var <= var_threshold:
            return prev_weights  # Already meets constraint
        
        # Scale down risky positions
        scale_factor = np.sqrt(var_threshold / current_var) * 0.95  # 5% buffer
        
        w_e_new = w_e_prev * scale_factor
        w_g_new = w_g_prev * scale_factor
        w_f_new = 1 - w_e_new - w_g_new
        
        return np.array([w_e_new, w_g_new, w_f_new])
    
    def _calculate_portfolio_returns_rt(self,
                                      weights: np.ndarray,
                                      asset_params: Dict[str, AssetParameters],
                                      opt_params: OptimizationParameters) -> np.ndarray:
        """Calculate portfolio returns for real-time optimizer"""
        return self._calculate_portfolio_returns_helper(weights, asset_params, opt_params)
    
    def _calculate_portfolio_returns_helper(self,
                                          weights: np.ndarray,
                                          asset_params: Dict[str, AssetParameters],
                                          opt_params: OptimizationParameters) -> np.ndarray:
        """Helper method for calculating portfolio returns"""
        T = weights.shape[0]
        portfolio_returns = np.zeros(T)
        
        event_asset = asset_params['event_asset']
        
        for t in range(T):
            w_e, w_g, w_f = weights[t]
            
            r_e = event_asset.expected_return[t]
            r_g = opt_params.mu_g
            r_f = opt_params.risk_free_rate
            
            portfolio_returns[t] = w_e * r_e + w_g * r_g + w_f * r_f
            
        return portfolio_returns
    
    def _calculate_sharpe_ratio(self,
                              returns: np.ndarray,
                              params: OptimizationParameters) -> float:
        """Calculate portfolio Sharpe ratio"""
        excess_returns = returns - params.risk_free_rate
        
        if len(excess_returns) > 1 and np.std(excess_returns) > 0:
            return np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)  # Annualized
        else:
            return 0.0

class PortfolioOptimizationFramework:
    """
    Main framework integrating different optimization approaches
    """
    
    def __init__(self, 
                 optimizer_type: str = 'continuous_time',
                 **optimizer_kwargs):
        
        if optimizer_type == 'continuous_time':
            self.optimizer = ContinuousTimeOptimizer(**optimizer_kwargs)
        elif optimizer_type == 'real_time':
            self.optimizer = RealTimeOptimizer(**optimizer_kwargs)
        else:
            raise ValueError(f"Unknown optimizer type: {optimizer_type}")
            
        self.optimization_results = {}
        
    def run_optimization(self,
                        data: pl.DataFrame,
                        expected_returns: np.ndarray,
                        volatility: np.ndarray,
                        optimization_params: Optional[OptimizationParameters] = None,
                        constraints: Optional[Dict] = None) -> Dict:
        """
        Run portfolio optimization on event data
        """
        if optimization_params is None:
            optimization_params = OptimizationParameters()
            
        # Prepare asset parameters
        asset_params = {
            'event_asset': AssetParameters(
                expected_return=expected_returns,
                volatility=volatility,
                transaction_costs={'buy': 0.002, 'sell': 0.001}
            )
        }
        
        # Initial portfolio weights
        initial_weights = np.array([0.1, 0.2, 0.7])  # Conservative start
        
        # Run optimization
        results = self.optimizer.optimize(
            asset_params=asset_params,
            optimization_params=optimization_params,
            initial_weights=initial_weights,
            constraints=constraints
        )
        
        # Store results
        self.optimization_results = results
        
        # Add data context
        results['data'] = data
        results['input_expected_returns'] = expected_returns
        results['input_volatility'] = volatility
        
        return results
    
    def analyze_optimization_results(self) -> Dict:
        """
        Analyze optimization results and provide insights
        """
        if not self.optimization_results:
            raise RuntimeError("No optimization results available. Run optimization first.")
            
        results = self.optimization_results
        weights = results['optimal_weights']
        
        analysis = {}
        
        # Weight statistics
        analysis['weight_statistics'] = {
            'event_asset': {
                'mean': np.mean(weights[:, 0]),
                'std': np.std(weights[:, 0]),
                'min': np.min(weights[:, 0]),
                'max': np.max(weights[:, 0])
            },
            'generic_asset': {
                'mean': np.mean(weights[:, 1]),
                'std': np.std(weights[:, 1]),
                'min': np.min(weights[:, 1]),
                'max': np.max(weights[:, 1])
            },
            'risk_free_asset': {
                'mean': np.mean(weights[:, 2]),
                'std': np.std(weights[:, 2]),
                'min': np.min(weights[:, 2]),
                'max': np.max(weights[:, 2])
            }
        }
        
        # Portfolio performance
        if 'portfolio_returns' in results:
            portfolio_returns = results['portfolio_returns']
            analysis['performance'] = {
                'total_return': np.sum(portfolio_returns),
                'average_return': np.mean(portfolio_returns),
                'volatility': np.std(portfolio_returns),
                'sharpe_ratio': results.get('sharpe_ratio', 0.0),
                'max_drawdown': self._calculate_max_drawdown(portfolio_returns)
            }
        
        # Transaction cost analysis
        if 'transaction_costs' in results:
            tc = results['transaction_costs']
            analysis['transaction_costs'] = {
                'total_costs': np.sum(tc),
                'average_costs': np.mean(tc),
                'max_costs': np.max(tc),
                'cost_as_pct_return': np.sum(tc) / max(abs(np.sum(portfolio_returns)), 1e-8) * 100
            }
        
        # Phase analysis
        if 'data' in results:
            analysis['phase_analysis'] = self._analyze_by_phase(results)
            
        return analysis
    
    def _calculate_max_drawdown(self, returns: np.ndarray) -> float:
        """Calculate maximum drawdown"""
        cumulative = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        return np.min(drawdown)
    
    def _analyze_by_phase(self, results: Dict) -> Dict:
        """Analyze results by event phases"""
        data = results['data']
        weights = results['optimal_weights']
        
        if 'days_to_event' not in data.columns:
            return {}
            
        days_to_event = data.get_column('days_to_event').to_numpy()
        
        phases = {
            'pre_event': (-15, -1),
            'event_window': (-2, 2),
            'post_event_rising': (1, 5),
            'post_event_decay': (6, 15)
        }
        
        phase_analysis = {}
        
        for phase_name, (start_day, end_day) in phases.items():
            mask = (days_to_event >= start_day) & (days_to_event <= end_day)
            
            if np.any(mask):
                phase_weights = weights[mask]
                phase_analysis[phase_name] = {
                    'avg_event_weight': np.mean(phase_weights[:, 0]),
                    'avg_generic_weight': np.mean(phase_weights[:, 1]),
                    'avg_rf_weight': np.mean(phase_weights[:, 2]),
                    'weight_volatility': np.std(phase_weights[:, 0]),
                    'sample_size': np.sum(mask)
                }
        
        return phase_analysis
    
    def plot_optimization_results(self, results_dir: str, file_prefix: str):
        """
        Plot optimization results
        """
        if not self.optimization_results:
            raise RuntimeError("No optimization results available.")
            
        try:
            import matplotlib.pyplot as plt
            import os
            
            results = self.optimization_results
            weights = results['optimal_weights']
            
            if 'data' in results:
                days_to_event = results['data'].get_column('days_to_event').to_numpy()
            else:
                days_to_event = np.arange(len(weights))
            
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            
            # Plot 1: Portfolio weights over time
            axes[0, 0].plot(days_to_event, weights[:, 0], label='Event Asset', linewidth=2, color='blue')
            axes[0, 0].plot(days_to_event, weights[:, 1], label='Generic Asset', linewidth=2, color='green')
            axes[0, 0].plot(days_to_event, weights[:, 2], label='Risk-Free Asset', linewidth=2, color='red')
            axes[0, 0].axvline(x=0, color='black', linestyle='--', alpha=0.5, label='Event Day')
            axes[0, 0].set_xlabel('Days to Event')
            axes[0, 0].set_ylabel('Portfolio Weight')
            axes[0, 0].set_title('Optimal Portfolio Weights')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
            
            # Plot 2: Portfolio returns
            if 'portfolio_returns' in results:
                portfolio_returns = results['portfolio_returns']
                cumulative_returns = np.cumprod(1 + portfolio_returns)
                
                axes[0, 1].plot(days_to_event, cumulative_returns, linewidth=2, color='purple')
                axes[0, 1].axvline(x=0, color='black', linestyle='--', alpha=0.5)
                axes[0, 1].set_xlabel('Days to Event')
                axes[0, 1].set_ylabel('Cumulative Return')
                axes[0, 1].set_title('Portfolio Performance')
                axes[0, 1].grid(True, alpha=0.3)
            
            # Plot 3: Transaction costs
            if 'transaction_costs' in results:
                tc = results['transaction_costs']
                axes[1, 0].plot(days_to_event, tc, linewidth=2, color='orange')
                axes[1, 0].axvline(x=0, color='black', linestyle='--', alpha=0.5)
                axes[1, 0].set_xlabel('Days to Event')
                axes[1, 0].set_ylabel('Transaction Costs')
                axes[1, 0].set_title('Transaction Costs Over Time')
                axes[1, 0].grid(True, alpha=0.3)
            
            # Plot 4: Risk metrics
            if 'realized_variance' in results:
                variance = results['realized_variance']
                axes[1, 1].plot(days_to_event, np.sqrt(variance * 252), linewidth=2, color='red')
                axes[1, 1].axvline(x=0, color='black', linestyle='--', alpha=0.5)
                axes[1, 1].set_xlabel('Days to Event')
                axes[1, 1].set_ylabel('Annualized Volatility')
                axes[1, 1].set_title('Portfolio Risk Over Time')
                axes[1, 1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plot_filename = os.path.join(results_dir, f"{file_prefix}_portfolio_optimization.png")
            plt.savefig(plot_filename, dpi=200, bbox_inches='tight')
            plt.close()
            
            print(f"Portfolio optimization plot saved to: {plot_filename}")
            
        except Exception as e:
            warnings.warn(f"Could not create optimization plot: {e}")