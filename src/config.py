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
    # lookback_days for GARCH estimation if done per stock (e.g. 252)
    # For applying fixed params, this is less relevant for h_initial calculation.
    garch_burn_in_period: int = 50 # Days to let GARCH recursion stabilize if h_initial is estimated
    
    event_window_pre: int = 30  # Pre-event analysis window (days) from t_event
    event_window_post: int = 30  # Post-event analysis window (days) from t_event
    
    # --- GJR-GARCH(1,1) Baseline Volatility Parameters (from paper or to be estimated) ---
    # h_t = omega + alpha * epsilon_{t-1}^2 + beta * h_{t-1} + gamma * I_{t-1} * epsilon_{t-1}^2
    gjr_omega: float = 1e-6  # Long-run average variance component
    gjr_alpha: float = 0.08  # ARCH parameter (impact of past shocks)
    gjr_beta: float = 0.89   # GARCH parameter (volatility persistence) -  FIXED FROM 0.90
    gjr_gamma: float = 0.04  # Asymmetry parameter (leverage effect)
    
    # --- Unified Volatility Model Event-Specific Parameters (phi functions) ---
    # k1, k2, delta_t1, delta_t2, delta_t3, delta are from paper's eq. 4,5,6
    event_k1: float = 1.3  # Pre-event volatility peak multiplier (phi_1)
    event_delta_t1: float = 5.0  # Pre-event rise duration parameter (phi_1)
    
    event_k2: float = 1.5  # Post-event volatility peak multiplier (phi_2, phi_3) (k2 > k1 generally)
    event_delta_t2: float = 3.0  # Post-event rise rate parameter (phi_2)
    event_delta: int = 5  # Duration of post-event rising phase (phi_2 ends, phi_3 starts) (trading days)
    event_delta_t3: float = 10.0  # Post-event decay rate parameter (phi_3)
    
    # --- Bias Parameters for Expected Return Formation (Assumption 5) ---
    # b_t = b_0 * (1 + kappa * (sigma_e(t)/sqrt(h_t) - 1)/(k_2 - 1))
    bias_baseline_b0: float = 0.001  # Baseline bias parameter b_0
    bias_kappa_sensitivity: float = 0.5  # Sensitivity kappa to vol changes in post-event rising
    
    # --- Market Parameters (Assumed for RVR/Sharpe, not explicitly modeled in equilibrium here) ---
    risk_free_rate_annual: float = 0.03  # Annual risk-free rate
    risk_free_rate_daily: float = (1 + 0.03)**(1/252) - 1 # Daily risk-free rate
    
    # --- Hypothesis Testing Parameters ---
    # H2.1: Prediction horizons for volatility innovation analysis
    prediction_horizons_h2: Tuple[int, ...] = (1, 5, 10)  # Days ahead
    
    # H2.2: Post-event volatility persistence window
    vol_persistence_window_days_h2: int = 10
    
    # Statistical significance level
    alpha_significance: float = 0.05
    
    # --- Data Quality / Processing Parameters ---
    min_obs_for_event_window: int = 20 # Min obs required within the +/- event_window_pre/post
    min_returns_for_garch_init: int = 30 # Min returns to estimate initial variance for GARCH

    def __post_init__(self):
        """Validate parameters after initialization."""
        if not (self.gjr_alpha + self.gjr_beta + self.gjr_gamma / 2 < 1.0):
            raise ValueError("GJR-GARCH parameters violate stationarity condition.")
        if self.event_k1 <= 1.0 or self.event_k2 <= 1.0:
            raise ValueError("Volatility multipliers k1 and k2 must be > 1.")
        if self.event_k2 <= self.event_k1 and self.event_k2 != 0 : # Allow k2=0 if no post-event effect is desired
            print(f"Warning: Config has event_k2 ({self.event_k2}) <= event_k1 ({self.event_k1}). Typically k2 > k1.")
        if self.event_delta <= 0:
            raise ValueError("event_delta (duration of post-event rising phase) must be positive.")
