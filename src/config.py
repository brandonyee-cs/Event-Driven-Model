# config.py
# Central configuration for the event-driven asset pricing model
# Based on "Modeling Equilibrium Asset Pricing Around Events with Heterogeneous Beliefs, 
# Dynamic Volatility, and a Two-Risk Uncertainty Framework"

from dataclasses import dataclass
from typing import Tuple

@dataclass
class Config:
    """
    Configuration parameters for the unified volatility model and event study analysis.
    Parameters align with the theoretical framework from the paper.
    """
    
    # --- Data Loading and Window Parameters ---
    lookback_days: int = 60  # Days before event for baseline GARCH estimation
    event_window_pre: int = 30  # Pre-event analysis window (days)
    event_window_post: int = 30  # Post-event analysis window (days)
    min_required_days: int = 20  # Minimum data points required for analysis
    
    # --- GJR-GARCH(1,1) Baseline Volatility Parameters ---
    # h_t = omega + alpha * epsilon_{t-1}^2 + beta * h_{t-1} + gamma * I_{t-1} * epsilon_{t-1}^2
    gjr_omega: float = 1e-6  # Long-run average variance component
    gjr_alpha: float = 0.08  # ARCH parameter (impact of past shocks)
    gjr_beta: float = 0.90   # GARCH parameter (volatility persistence)
    gjr_gamma: float = 0.04  # Asymmetry parameter (leverage effect)
    
    # --- Unified Volatility Model Event-Specific Parameters ---
    # sigma_e(t) = sqrt(h_t) * (1 + phi(t))
    # Three-phase volatility adjustments:
    
    # Phase 1: Pre-event (phi_1)
    event_k1: float = 1.3  # Pre-event volatility peak multiplier
    event_delta_t1: float = 5.0  # Pre-event rise duration parameter
    
    # Phase 2: Post-event rising (phi_2)
    event_k2: float = 1.5  # Post-event volatility peak multiplier (k2 > k1)
    event_delta_t2: float = 3.0  # Post-event rise rate parameter
    event_delta: int = 5  # Duration of post-event rising phase (trading days)
    
    # Phase 3: Post-event decay (phi_3)
    event_delta_t3: float = 10.0  # Post-event decay rate parameter
    
    # --- Bias Parameters for Expected Return Formation ---
    # b_t = b_0 * (1 + kappa * (sigma_e(t)/sqrt(h_t) - 1)/(k_2 - 1))
    bias_baseline_b0: float = 0.001  # Baseline bias parameter
    bias_kappa_sensitivity: float = 0.5  # Sensitivity to volatility changes
    
    # --- Market Parameters ---
    risk_free_rate_annual: float = 0.045  # Annual risk-free rate (4.5%)
    risk_free_rate_daily: float = 0.045 / 252  # Daily risk-free rate
    
    # --- Investor Heterogeneity Parameters ---
    # Proportions of investor types
    n_informed: float = 0.3  # Proportion of informed investors
    n_uninformed: float = 0.5  # Proportion of uninformed investors
    n_liquidity: float = 0.2  # Proportion of liquidity traders
    
    # --- Transaction Cost Parameters ---
    # Asymmetric costs: tau_b > tau_s
    transaction_cost_buy_base: float = 0.002  # Base buying cost
    transaction_cost_sell_base: float = 0.001  # Base selling cost
    transaction_cost_multiplier_stress: float = 1.5  # Multiplier during high volatility
    
    # --- Hypothesis Testing Parameters ---
    # H1: RVR and Sharpe ratio peak testing
    rvr_peak_test_phases: Tuple[str, ...] = ('pre_event', 'post_event_rising', 'post_event_decay')
    
    # H2.1: Prediction horizons for volatility innovation analysis
    prediction_horizons: Tuple[int, ...] = (1, 5, 10)  # Days ahead
    
    # H2.2: Post-event volatility persistence window
    vol_persistence_window_days: int = 10
    
    # H2.3: Asymmetric response threshold
    gamma_significance_threshold: float = 0.0  # Test if gamma > 0
    
    # Statistical significance level
    alpha_significance: float = 0.05
    
    # --- Data Quality Parameters ---
    return_winsorization_lower: float = 0.01  # Lower percentile for winsorization
    return_winsorization_upper: float = 0.99  # Upper percentile for winsorization
    max_daily_return: float = 1.0  # Maximum plausible daily return (100%)
    
    def get_volatility_parameters_dict(self) -> dict:
        """
        Returns a dictionary of parameters for VolatilityParameters dataclass.
        Used by EventProcessor for unified volatility calculations.
        """
        return {
            "omega": self.gjr_omega,
            "alpha": self.gjr_alpha,
            "beta": self.gjr_beta,
            "gamma": self.gjr_gamma,
            "k1": self.event_k1,
            "k2": self.event_k2,
            "delta_t1": self.event_delta_t1,
            "delta_t2": self.event_delta_t2,
            "delta_t3": self.event_delta_t3,
            "delta": self.event_delta,
        }
    
    def get_investor_parameters(self, investor_type: str) -> dict:
        """
        Returns parameters for different investor types based on the paper.
        
        Args:
            investor_type: One of 'informed', 'uninformed', or 'liquidity'
            
        Returns:
            Dictionary with investor-specific parameters
        """
        if investor_type == 'informed':
            return {
                'risk_aversion': 2.0,
                'bias_baseline': 0.001,
                'bias_sensitivity': 0.3,
                'information_quality': 0.9,
                'transaction_cost_buy': self.transaction_cost_buy_base,
                'transaction_cost_sell': self.transaction_cost_sell_base
            }
        elif investor_type == 'uninformed':
            return {
                'risk_aversion': 3.0,
                'bias_baseline': 0.003,
                'bias_sensitivity': 0.5,
                'information_quality': 0.5,
                'transaction_cost_buy': self.transaction_cost_buy_base * 1.5,
                'transaction_cost_sell': self.transaction_cost_sell_base * 1.5
            }
        elif investor_type == 'liquidity':
            return {
                'risk_aversion': 2.5,
                'bias_baseline': 0.0,
                'bias_sensitivity': 0.0,
                'information_quality': 0.0,
                'liquidity_constraint': 0.3,  # 30% purchase reduction pre-event
                'transaction_cost_buy': self.transaction_cost_buy_base,
                'transaction_cost_sell': self.transaction_cost_sell_base
            }
        else:
            raise ValueError(f"Unknown investor type: {investor_type}")
    
    def validate_parameters(self) -> bool:
        """
        Validates that all parameters satisfy theoretical constraints.
        
        Returns:
            True if all parameters are valid
        """
        # Check GARCH stationarity condition
        if self.gjr_alpha + self.gjr_beta + self.gjr_gamma/2 >= 1: raise ValueError("GJR-GARCH parameters violate stationarity condition")
        
        # Check volatility multipliers
        if self.event_k1 <= 1.0 or self.event_k2 <= 1.0: raise ValueError("Volatility multipliers k1 and k2 must be > 1")
        
        if self.event_k2 <= self.event_k1: raise ValueError("Post-event peak k2 must exceed pre-event peak k1")
        
        # Check investor proportions
        if abs(self.n_informed + self.n_uninformed + self.n_liquidity - 1.0) > 1e-6: raise ValueError("Investor proportions must sum to 1")
        
        # Check transaction costs
        if self.transaction_cost_buy_base <= self.transaction_cost_sell_base: raise ValueError("Buy costs must exceed sell costs (tau_b > tau_s)")
        
        return True
