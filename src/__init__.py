"""
Dynamic Asset Pricing Model Package
Provides infrastructure for continuous-time asset pricing analysis around high-uncertainty events.
"""

from .config import get_config, get_config_manager, ModelConfig
from .logging_config import setup_logging, get_logger
from .enhanced_data_loader import create_data_loader, load_event_data, validate_data_files
from .models import (
    # Core volatility models
    GARCHModel, GJRGARCHModel, ThreePhaseVolatilityModel, EnhancedGJRGARCHModel,
    # Risk and investor models
    MultiRiskFramework, InvestorModel, InformedInvestor, UninformedInvestor, LiquidityTrader,
    # Portfolio optimization
    PortfolioOptimizer, HJBSolver, TransactionCostModel, UtilityOptimizer,
    # Monte Carlo and simulation
    MonteCarloEngine, MonteCarloStatistics,
    # Machine learning components
    MLParameterEstimator, FeatureEngineer, RegimeIdentifier, ValidationFramework,
    # Performance and testing
    PerformanceMetrics, HypothesisTester,
    # Additional models
    JumpProcessModel, TimeSeriesRidge, XGBoostDecileModel
)

# Package version
__version__ = "0.1.0"

# Initialize logging on import
setup_logging()

# Get logger for package initialization
logger = get_logger(__name__)

def initialize_model_infrastructure(config_path=None, log_level="INFO"):
    """
    Initialize the model infrastructure with configuration and logging.
    
    Args:
        config_path: Optional path to configuration file
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
    
    Returns:
        tuple: (config_manager, logger)
    """
    # Setup logging
    logging_manager = setup_logging(log_level=log_level)
    logger = get_logger(__name__)
    
    # Load configuration
    config_manager = get_config_manager(config_path)
    config = config_manager.get_config()
    
    # Validate configuration
    issues = config_manager.validate_config()
    if issues:
        logger.warning(f"Configuration validation issues: {issues}")
    
    # Validate data files if not using mock data
    if not config.data.use_mock_data:
        data_issues = validate_data_files(config)
        if data_issues:
            logger.warning(f"Data file validation issues: {data_issues}")
            logger.info("Consider setting use_mock_data: true for development")
    
    logger.info("Model infrastructure initialized successfully")
    logger.info(f"Configuration: {config_manager.config_path}")
    logger.info(f"Data source: {'Mock data' if config.data.use_mock_data else 'Production data'}")
    
    return config_manager, logger

# Auto-initialize with defaults
try:
    _config_manager, _logger = initialize_model_infrastructure()
    logger.info("Package initialized with default configuration")
except Exception as e:
    logger = get_logger(__name__)
    logger.error(f"Failed to initialize package: {e}")

__all__ = [
    # Configuration and infrastructure
    'get_config',
    'get_config_manager', 
    'ModelConfig',
    'setup_logging',
    'get_logger',
    'create_data_loader',
    'load_event_data',
    'validate_data_files',
    'initialize_model_infrastructure',
    # Core volatility models
    'GARCHModel', 'GJRGARCHModel', 'ThreePhaseVolatilityModel', 'EnhancedGJRGARCHModel',
    # Risk and investor models
    'MultiRiskFramework', 'InvestorModel', 'InformedInvestor', 'UninformedInvestor', 'LiquidityTrader',
    # Portfolio optimization
    'PortfolioOptimizer', 'HJBSolver', 'TransactionCostModel', 'UtilityOptimizer',
    # Monte Carlo and simulation
    'MonteCarloEngine', 'MonteCarloStatistics',
    # Machine learning components
    'MLParameterEstimator', 'FeatureEngineer', 'RegimeIdentifier', 'ValidationFramework',
    # Performance and testing
    'PerformanceMetrics', 'HypothesisTester',
    # Additional models
    'JumpProcessModel', 'TimeSeriesRidge', 'XGBoostDecileModel'
]