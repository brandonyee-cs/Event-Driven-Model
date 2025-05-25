# config.py
# Central configuration for the event study analysis

from dataclasses import dataclass

@dataclass
class Config:
    # Data loading and window parameters
    lookback_days: int = 60  # Days before event for GARCH estimation or baseline
    event_window_pre: int = 30  # Days before event for analysis
    event_window_post: int = 30  # Days after event for analysis
    min_required_days: int = 20 # Minimum data points in window for an event to be processed

    # GJR-GARCH parameters (can be overridden if estimated per stock)
    gjr_omega: float = 1e-6
    gjr_alpha: float = 0.08
    gjr_beta: float = 0.90
    gjr_gamma: float = 0.04 # Asymmetry

    # Unified Volatility Model - Event-specific parameters
    # (phi functions adjusting the GJR-GARCH baseline)
    event_k1: float = 1.3  # Pre-event volatility peak multiplier (phi1)
    event_k2: float = 1.5  # Post-event volatility peak multiplier (phi2, phi3)
    event_delta_t1: float = 5.0  # Duration/decay for pre-event rise (phi1)
    event_delta_t2: float = 3.0  # Rise rate for post-event phase (phi2)
    event_delta_t3: float = 10.0 # Decay rate for post-event phase (phi3)
    event_delta: int = 5  # Duration of post-event rising phase before decay starts

    # Bias parameters for expected return adjustment (Hypothesis 1)
    # b_t = b_0 * (1 + kappa * (Psi_t - 1)/(k_2 - 1)) for post-event rising phase
    bias_baseline_b0: float = 0.001 # Baseline optimism/pessimism
    bias_kappa_sensitivity: float = 0.5 # Sensitivity of bias to event volatility phase

    # Risk-free rate (daily)
    risk_free_rate_daily: float = 0.045 / 252 # Example: 4.5% annualized

    # Prediction horizons for Hypothesis 2.1 (days)
    prediction_horizons: tuple = (1, 5, 10)

    # Volatility persistence window for Hypothesis 2.2 (days post-event)
    vol_persistence_window_days: int = 10

    # Significance level for statistical tests
    alpha_significance: float = 0.05

    def get_volatility_parameters_dict(self) -> dict:
        """Returns a dictionary of parameters for VolatilityParameters dataclass"""
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