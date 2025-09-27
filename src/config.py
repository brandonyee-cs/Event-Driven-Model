"""
Configuration management system for the Dynamic Asset Pricing Model.
Handles model parameters, data sources, and environment-specific settings.
"""

import os
import yaml
import json
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Union, Any
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


@dataclass
class VolatilityConfig:
    """Configuration for three-phase volatility model parameters."""
    k1: float = 1.5  # Pre-event volatility scaling
    k2: float = 2.0  # Post-event volatility scaling
    delta_t1: float = 5.0  # Pre-event duration
    delta_t2: float = 3.0  # Rising phase duration
    delta_t3: float = 10.0  # Decay phase duration
    delta: float = 5.0  # General delta parameter


@dataclass
class OptimizationConfig:
    """Configuration for portfolio optimization parameters."""
    gamma_T: float = 2.0  # Terminal wealth risk aversion
    gamma_V: float = 1.0  # Real-time variance aversion
    risk_free_rate: float = 0.0  # Risk-free rate
    optimistic_bias: float = 0.01  # Optimistic bias parameter


@dataclass
class TransactionCostConfig:
    """Configuration for transaction cost parameters."""
    tau_b: float = 0.001  # Purchase transaction cost
    tau_s: float = 0.0005  # Sale transaction cost (tau_s <= tau_b)


@dataclass
class DataConfig:
    """Configuration for data sources and processing."""
    # Development environment settings
    use_mock_data: bool = True
    mock_events: int = 1000
    mock_assets: int = 50
    
    # Production data paths
    stock_files: List[str] = None
    fda_events: str = None
    earnings_events: str = None
    
    # Analysis parameters
    window_days: int = 30
    analysis_window: List[int] = None  # [-15, 15]
    
    def __post_init__(self):
        if self.analysis_window is None:
            self.analysis_window = [-15, 15]
        if self.stock_files is None:
            self.stock_files = []


@dataclass
class MLConfig:
    """Configuration for machine learning parameters."""
    xgb_params: Dict[str, Any] = None
    ensemble_weight: float = 0.5
    n_deciles: int = 10
    alpha: float = 0.1
    lambda_smooth: float = 0.1
    
    def __post_init__(self):
        if self.xgb_params is None:
            self.xgb_params = {
                'n_estimators': 100,
                'max_depth': 5,
                'learning_rate': 0.1,
                'objective': 'reg:squarederror',
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': 42,
                'n_jobs': -1
            }


@dataclass
class SimulationConfig:
    """Configuration for Monte Carlo simulation parameters."""
    n_simulations: int = 1000
    n_steps: int = 252  # Trading days in a year
    parallel_processing: bool = True
    random_seed: int = 42


@dataclass
class ModelConfig:
    """Main configuration class containing all model parameters."""
    volatility: VolatilityConfig = None
    optimization: OptimizationConfig = None
    transaction_costs: TransactionCostConfig = None
    data: DataConfig = None
    ml: MLConfig = None
    simulation: SimulationConfig = None
    
    def __post_init__(self):
        if self.volatility is None:
            self.volatility = VolatilityConfig()
        if self.optimization is None:
            self.optimization = OptimizationConfig()
        if self.transaction_costs is None:
            self.transaction_costs = TransactionCostConfig()
        if self.data is None:
            self.data = DataConfig()
        if self.ml is None:
            self.ml = MLConfig()
        if self.simulation is None:
            self.simulation = SimulationConfig()


class ConfigManager:
    """Manages configuration loading, validation, and environment-specific settings."""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or self._get_default_config_path()
        self.config = ModelConfig()
        self._load_config()
    
    def _get_default_config_path(self) -> str:
        """Get default configuration file path."""
        return os.path.join(os.path.dirname(__file__), '..', 'config', 'model_config.yaml')
    
    def _load_config(self) -> None:
        """Load configuration from file if it exists, otherwise use defaults."""
        try:
            if os.path.exists(self.config_path):
                logger.info(f"Loading configuration from {self.config_path}")
                with open(self.config_path, 'r') as f:
                    if self.config_path.endswith('.yaml') or self.config_path.endswith('.yml'):
                        config_dict = yaml.safe_load(f)
                    else:
                        config_dict = json.load(f)
                
                self._update_config_from_dict(config_dict)
                logger.info("Configuration loaded successfully")
            else:
                logger.info(f"Configuration file not found at {self.config_path}, using defaults")
                self._create_default_config_file()
        except Exception as e:
            logger.warning(f"Error loading configuration: {e}. Using defaults.")
    
    def _update_config_from_dict(self, config_dict: Dict[str, Any]) -> None:
        """Update configuration from dictionary."""
        if 'volatility' in config_dict:
            self.config.volatility = VolatilityConfig(**config_dict['volatility'])
        
        if 'optimization' in config_dict:
            self.config.optimization = OptimizationConfig(**config_dict['optimization'])
        
        if 'transaction_costs' in config_dict:
            self.config.transaction_costs = TransactionCostConfig(**config_dict['transaction_costs'])
        
        if 'data' in config_dict:
            self.config.data = DataConfig(**config_dict['data'])
        
        if 'ml' in config_dict:
            self.config.ml = MLConfig(**config_dict['ml'])
        
        if 'simulation' in config_dict:
            self.config.simulation = SimulationConfig(**config_dict['simulation'])
    
    def _create_default_config_file(self) -> None:
        """Create default configuration file."""
        try:
            config_dir = os.path.dirname(self.config_path)
            os.makedirs(config_dir, exist_ok=True)
            
            config_dict = {
                'volatility': asdict(self.config.volatility),
                'optimization': asdict(self.config.optimization),
                'transaction_costs': asdict(self.config.transaction_costs),
                'data': asdict(self.config.data),
                'ml': asdict(self.config.ml),
                'simulation': asdict(self.config.simulation)
            }
            
            with open(self.config_path, 'w') as f:
                yaml.dump(config_dict, f, default_flow_style=False, indent=2)
            
            logger.info(f"Created default configuration file at {self.config_path}")
        except Exception as e:
            logger.warning(f"Could not create default configuration file: {e}")
    
    def get_config(self) -> ModelConfig:
        """Get the current configuration."""
        return self.config
    
    def update_config(self, **kwargs) -> None:
        """Update configuration parameters."""
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
            else:
                logger.warning(f"Unknown configuration parameter: {key}")
    
    def save_config(self, path: Optional[str] = None) -> None:
        """Save current configuration to file."""
        save_path = path or self.config_path
        try:
            config_dict = {
                'volatility': asdict(self.config.volatility),
                'optimization': asdict(self.config.optimization),
                'transaction_costs': asdict(self.config.transaction_costs),
                'data': asdict(self.config.data),
                'ml': asdict(self.config.ml),
                'simulation': asdict(self.config.simulation)
            }
            
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            with open(save_path, 'w') as f:
                if save_path.endswith('.yaml') or save_path.endswith('.yml'):
                    yaml.dump(config_dict, f, default_flow_style=False, indent=2)
                else:
                    json.dump(config_dict, f, indent=2)
            
            logger.info(f"Configuration saved to {save_path}")
        except Exception as e:
            logger.error(f"Error saving configuration: {e}")
    
    def is_development_mode(self) -> bool:
        """Check if running in development mode (using mock data)."""
        return self.config.data.use_mock_data
    
    def validate_config(self) -> List[str]:
        """Validate configuration and return list of issues."""
        issues = []
        
        # Validate volatility parameters
        vol = self.config.volatility
        if vol.k1 <= 0:
            issues.append("Volatility k1 must be positive")
        if vol.k2 <= vol.k1:
            issues.append("Volatility k2 must be greater than k1")
        if any(param <= 0 for param in [vol.delta_t1, vol.delta_t2, vol.delta_t3]):
            issues.append("All delta_t parameters must be positive")
        
        # Validate optimization parameters
        opt = self.config.optimization
        if opt.gamma_T <= 0 or opt.gamma_V <= 0:
            issues.append("Risk aversion parameters must be positive")
        
        # Validate transaction costs
        tc = self.config.transaction_costs
        if tc.tau_s > tc.tau_b:
            issues.append("Sale transaction cost (tau_s) must be <= purchase cost (tau_b)")
        if tc.tau_b < 0 or tc.tau_s < 0:
            issues.append("Transaction costs must be non-negative")
        
        # Validate data configuration
        data = self.config.data
        if not data.use_mock_data:
            if not data.stock_files:
                issues.append("Stock files must be specified when not using mock data")
            if not data.fda_events and not data.earnings_events:
                issues.append("At least one event file must be specified when not using mock data")
        
        if data.window_days <= 0:
            issues.append("Window days must be positive")
        
        return issues


# Global configuration manager instance
_config_manager = None

def get_config_manager(config_path: Optional[str] = None) -> ConfigManager:
    """Get the global configuration manager instance."""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager(config_path)
    return _config_manager

def get_config() -> ModelConfig:
    """Get the current model configuration."""
    return get_config_manager().get_config()