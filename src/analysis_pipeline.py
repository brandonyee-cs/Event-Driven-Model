"""
Analysis Pipeline for Dynamic Asset Pricing Model.
Implements hardcoded parameter analysis with FDA and earnings specific result organization.
"""

import os
import sys
import numpy as np
import polars as pl
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import logging
from pathlib import Path

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.enhanced_data_loader import EnhancedEventDataLoader, create_result_directories, save_analysis_results
from src.models import GJRGARCHModel, ThreePhaseVolatilityModel, MultiRiskFramework
from src.config import ModelConfig

logger = logging.getLogger(__name__)


class AnalysisPipeline:
    """
    Main analysis pipeline with hardcoded parameters for FDA and earnings analysis.
    Implements the specific parameter configuration required by task 11.1.
    """
    
    def __init__(self, results_base_dir: str = "analysis_results"):
        """
        Initialize analysis pipeline with hardcoded parameters.
        
        Args:
            results_base_dir: Base directory for saving analysis results
        """
        # Hardcoded parameters as specified in requirements
        self.WINDOW_DAYS = 30
        self.ANALYSIS_WINDOW = (-15, 15)
        
        # GJR-GARCH parameters
        self.K1 = 1.5
        self.K2 = 2.0
        self.DELTA_T1 = 5.0
        self.DELTA_T2 = 3.0
        self.DELTA_T3 = 10.0
        self.DELTA = 5
        
        # Additional parameters
        self.OPTIMISTIC_BIAS = 0.01
        self.RISK_FREE_RATE = 0.0
        
        self.results_base_dir = results_base_dir
        self.config = self._create_analysis_config()
        
        logger.info("Analysis pipeline initialized with hardcoded parameters")
        logger.info(f"Window days: {self.WINDOW_DAYS}, Analysis window: {self.ANALYSIS_WINDOW}")
        logger.info(f"GJR-GARCH params: K1={self.K1}, K2={self.K2}, DELTA_T1={self.DELTA_T1}")
    
    def _create_analysis_config(self) -> ModelConfig:
        """Create model configuration with hardcoded parameters."""
        config = ModelConfig()
        
        # Data configuration
        config.data.window_days = self.WINDOW_DAYS
        config.data.analysis_window = list(self.ANALYSIS_WINDOW)
        
        # Volatility model configuration
        config.volatility.k1 = self.K1
        config.volatility.k2 = self.K2
        config.volatility.delta_t1 = self.DELTA_T1
        config.volatility.delta_t2 = self.DELTA_T2
        config.volatility.delta_t3 = self.DELTA_T3
        config.volatility.delta = self.DELTA
        
        # Optimization configuration
        config.optimization.optimistic_bias = self.OPTIMISTIC_BIAS
        config.optimization.risk_free_rate = self.RISK_FREE_RATE
        
        return config
    
    def setup_result_directories(self) -> Dict[str, str]:
        """
        Create result directories for FDA and earnings analysis.
        
        Returns:
            Dictionary mapping directory names to paths
        """
        directories = create_result_directories(self.results_base_dir)
        
        # Ensure we have the expected directory names for FDA and earnings analysis
        expected_dirs = {
            'fda_results': directories.get('fda', directories.get('base')),
            'earnings_results': directories.get('earnings', directories.get('base')),
            'combined_results': directories.get('combined', directories.get('base'))
        }
        
        # Add any additional directories that were created
        for key, value in directories.items():
            if key not in ['fda', 'earnings', 'combined', 'base']:
                expected_dirs[key] = value
        
        logger.info(f"Created result directories: {list(expected_dirs.keys())}")
        return expected_dirs
    
    def load_and_prepare_data(self, 
                            stock_files: Optional[List[str]] = None,
                            fda_events_file: Optional[str] = None,
                            earnings_events_file: Optional[str] = None,
                            use_mock_data: bool = True) -> Optional[pl.DataFrame]:
        """
        Load and prepare data for analysis.
        
        Args:
            stock_files: List of CRSP parquet files
            fda_events_file: FDA events CSV file
            earnings_events_file: Earnings events CSV file  
            use_mock_data: Whether to use mock data for development
            
        Returns:
            Prepared data DataFrame or None if loading fails
        """
        # Update configuration based on parameters
        self.config.data.use_mock_data = use_mock_data
        if not use_mock_data:
            self.config.data.stock_files = stock_files or []
            self.config.data.fda_events = fda_events_file
            self.config.data.earnings_events = earnings_events_file
        
        # Load data using enhanced data loader
        loader = EnhancedEventDataLoader(self.config)
        data = loader.load_data()
        
        if data is not None:
            logger.info(f"Loaded data: {len(data):,} records, {data['ticker'].n_unique()} tickers")
            logger.info(f"Date range: {data['date'].min()} to {data['date'].max()}")
            
            # Add analysis-specific columns
            data = self._add_analysis_columns(data)
            
        return data
    
    def _add_analysis_columns(self, data: pl.DataFrame) -> pl.DataFrame:
        """Add columns needed for analysis."""
        return data.with_columns([
            # Event phase classification
            pl.when(pl.col('days_to_event') < -self.DELTA_T1)
            .then(pl.lit('pre_event'))
            .when((pl.col('days_to_event') >= -self.DELTA_T1) & (pl.col('days_to_event') < 0))
            .then(pl.lit('pre_event_close'))
            .when((pl.col('days_to_event') >= 0) & (pl.col('days_to_event') <= self.DELTA_T2))
            .then(pl.lit('rising_post_event'))
            .when(pl.col('days_to_event') > self.DELTA_T2)
            .then(pl.lit('decay_post_event'))
            .otherwise(pl.lit('other'))
            .alias('event_phase'),
            
            # Analysis window indicator
            pl.when(
                (pl.col('days_to_event') >= self.ANALYSIS_WINDOW[0]) & 
                (pl.col('days_to_event') <= self.ANALYSIS_WINDOW[1])
            )
            .then(pl.lit(True))
            .otherwise(pl.lit(False))
            .alias('in_analysis_window')
        ])
    
    def run_volatility_analysis(self, data: pl.DataFrame) -> Dict[str, Any]:
        """
        Run three-phase volatility analysis with hardcoded parameters.
        
        Args:
            data: Input data DataFrame
            
        Returns:
            Dictionary containing volatility analysis results
        """
        logger.info("Running three-phase volatility analysis")
        
        results = {
            'model_parameters': {
                'k1': self.K1,
                'k2': self.K2,
                'delta_t1': self.DELTA_T1,
                'delta_t2': self.DELTA_T2,
                'delta_t3': self.DELTA_T3,
                'delta': self.DELTA
            },
            'phase_statistics': {},
            'volatility_forecasts': {}
        }
        
        # Filter to analysis window
        analysis_data = data.filter(pl.col('in_analysis_window'))
        
        # Group by event and calculate phase-specific statistics
        phase_stats = analysis_data.group_by(['event_id', 'event_phase']).agg([
            pl.col('ret').std().alias('volatility'),
            pl.col('ret').mean().alias('mean_return'),
            pl.col('ret').count().alias('n_observations'),
            pl.col('prc').mean().alias('avg_price')
        ])
        
        results['phase_statistics'] = phase_stats
        
        # Run GJR-GARCH model for each event
        garch_results = []
        unique_events = analysis_data['event_id'].unique().to_list()
        
        for event_id in unique_events[:10]:  # Limit for demonstration
            event_data = analysis_data.filter(pl.col('event_id') == event_id)
            if len(event_data) >= 20:  # Minimum observations for GARCH
                try:
                    returns = event_data.sort('date')['ret'].to_numpy()
                    
                    # Initialize GJR-GARCH with hardcoded parameters
                    garch_model = GJRGARCHModel(
                        omega=1e-6,
                        alpha=0.08,
                        beta=0.85,
                        gamma=0.05
                    )
                    
                    # Fit model
                    garch_model.fit(returns)
                    
                    if garch_model.fit_success:
                        # Generate forecasts
                        forecasts = garch_model.predict(n_steps=5)
                        
                        garch_results.append({
                            'event_id': event_id,
                            'omega': garch_model.omega,
                            'alpha': garch_model.alpha,
                            'beta': garch_model.beta,
                            'gamma': garch_model.gamma,
                            'forecasts': forecasts.tolist(),
                            'fit_success': True
                        })
                    else:
                        garch_results.append({
                            'event_id': event_id,
                            'fit_success': False,
                            'error_message': garch_model.fit_message
                        })
                        
                except Exception as e:
                    logger.warning(f"GARCH fitting failed for event {event_id}: {e}")
                    garch_results.append({
                        'event_id': event_id,
                        'fit_success': False,
                        'error_message': str(e)
                    })
        
        results['garch_results'] = garch_results
        
        logger.info(f"Volatility analysis completed for {len(garch_results)} events")
        return results
    
    def run_multi_risk_analysis(self, data: pl.DataFrame) -> Dict[str, Any]:
        """
        Run multi-risk framework analysis.
        
        Args:
            data: Input data DataFrame
            
        Returns:
            Dictionary containing multi-risk analysis results
        """
        logger.info("Running multi-risk framework analysis")
        
        results = {
            'risk_decomposition': {},
            'directional_news_effects': {},
            'impact_uncertainty_measures': {}
        }
        
        # Filter to analysis window
        analysis_data = data.filter(pl.col('in_analysis_window'))
        
        # Calculate risk decomposition by event phase
        risk_stats = analysis_data.group_by('event_phase').agg([
            pl.col('ret').var().alias('total_variance'),
            pl.col('ret').std().alias('total_volatility'),
            pl.col('ret').skew().alias('skewness'),
            pl.col('ret').kurtosis().alias('kurtosis'),
            pl.col('ret').count().alias('n_observations')
        ])
        
        results['risk_decomposition'] = risk_stats
        
        # Analyze directional news effects (event day returns)
        event_day_returns = analysis_data.filter(pl.col('days_to_event') == 0)
        if len(event_day_returns) > 0:
            directional_stats = event_day_returns.select([
                pl.col('ret').mean().alias('mean_event_return'),
                pl.col('ret').std().alias('event_volatility'),
                pl.col('ret').min().alias('min_event_return'),
                pl.col('ret').max().alias('max_event_return'),
                (pl.col('ret') > 0).sum().alias('positive_events'),
                (pl.col('ret') < 0).sum().alias('negative_events')
            ])
            
            results['directional_news_effects'] = directional_stats
        
        # Calculate impact uncertainty measures
        uncertainty_measures = analysis_data.group_by('days_to_event').agg([
            pl.col('ret').std().alias('daily_volatility'),
            pl.col('ret').var().alias('daily_variance'),
            pl.col('ret').abs().mean().alias('mean_absolute_return')
        ]).sort('days_to_event')
        
        results['impact_uncertainty_measures'] = uncertainty_measures
        
        logger.info("Multi-risk analysis completed")
        return results
    
    def calculate_performance_metrics(self, data: pl.DataFrame) -> Dict[str, Any]:
        """
        Calculate performance metrics with hardcoded parameters.
        
        Args:
            data: Input data DataFrame
            
        Returns:
            Dictionary containing performance metrics
        """
        logger.info("Calculating performance metrics")
        
        results = {
            'sharpe_ratios': {},
            'return_variance_ratios': {},
            'risk_adjusted_metrics': {}
        }
        
        # Filter to analysis window
        analysis_data = data.filter(pl.col('in_analysis_window'))
        
        # Calculate daily Sharpe ratios
        daily_metrics = analysis_data.group_by('days_to_event').agg([
            pl.col('ret').mean().alias('mean_return'),
            pl.col('ret').std().alias('volatility'),
            pl.col('ret').count().alias('n_observations')
        ]).with_columns([
            # Sharpe ratio (using risk-free rate = 0)
            (pl.col('mean_return') / pl.col('volatility')).alias('sharpe_ratio'),
            # Return-to-variance ratio
            (pl.col('mean_return') / pl.col('volatility').pow(2)).alias('return_variance_ratio')
        ]).sort('days_to_event')
        
        results['daily_metrics'] = daily_metrics
        
        # Calculate phase-specific metrics
        phase_metrics = analysis_data.group_by('event_phase').agg([
            pl.col('ret').mean().alias('mean_return'),
            pl.col('ret').std().alias('volatility'),
            pl.col('ret').count().alias('n_observations')
        ]).with_columns([
            (pl.col('mean_return') / pl.col('volatility')).alias('sharpe_ratio'),
            (pl.col('mean_return') / pl.col('volatility').pow(2)).alias('return_variance_ratio'),
            # Risk-adjusted return with optimistic bias
            (pl.col('mean_return') + self.OPTIMISTIC_BIAS).alias('bias_adjusted_return')
        ])
        
        results['phase_metrics'] = phase_metrics
        
        # Calculate event-specific metrics
        event_metrics = analysis_data.group_by('event_id').agg([
            pl.col('ret').mean().alias('mean_return'),
            pl.col('ret').std().alias('volatility'),
            pl.col('ret').sum().alias('cumulative_return'),
            pl.col('ret').count().alias('n_observations')
        ]).with_columns([
            (pl.col('mean_return') / pl.col('volatility')).alias('sharpe_ratio'),
            (pl.col('cumulative_return') / pl.col('volatility').pow(2)).alias('cumulative_rvr')
        ])
        
        results['event_metrics'] = event_metrics
        
        logger.info("Performance metrics calculation completed")
        return results
    
    def save_analysis_results(self, 
                            data: pl.DataFrame,
                            volatility_results: Dict[str, Any],
                            risk_results: Dict[str, Any],
                            performance_results: Dict[str, Any],
                            directories: Dict[str, str]) -> Dict[str, str]:
        """
        Save analysis results to FDA and earnings specific directories.
        
        Args:
            data: Original data DataFrame
            volatility_results: Volatility analysis results
            risk_results: Multi-risk analysis results
            performance_results: Performance metrics results
            directories: Result directories
            
        Returns:
            Dictionary mapping result types to saved file paths
        """
        logger.info("Saving analysis results")
        
        saved_files = {}
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Determine event types in data
        event_types = []
        if 'Event Type' in data.columns:
            unique_types = data['Event Type'].unique().to_list()
            event_types = [t.lower() for t in unique_types if t is not None]
        
        # Save results for each event type
        for event_type in ['fda', 'earnings']:
            if event_type in [t.lower() for t in event_types] or len(event_types) == 0:
                
                # Filter data for this event type if applicable
                if len(event_types) > 0:
                    event_data = data.filter(
                        pl.col('Event Type').str.to_lowercase() == event_type
                    )
                else:
                    event_data = data  # Use all data if no event type specified
                
                if len(event_data) > 0:
                    # Save main analysis data
                    data_file = save_analysis_results(
                        event_data, event_type, self.results_base_dir, 'parquet'
                    )
                    saved_files[f'{event_type}_data'] = data_file
                    
                    # Save volatility results
                    if 'phase_statistics' in volatility_results:
                        vol_file = os.path.join(
                            directories.get(f'{event_type}_results', directories['combined_results']),
                            f'{event_type}_volatility_analysis_{timestamp}.parquet'
                        )
                        volatility_results['phase_statistics'].write_parquet(vol_file)
                        saved_files[f'{event_type}_volatility'] = vol_file
                    
                    # Save performance metrics
                    if 'daily_metrics' in performance_results:
                        perf_file = os.path.join(
                            directories.get(f'{event_type}_results', directories['combined_results']),
                            f'{event_type}_performance_metrics_{timestamp}.parquet'
                        )
                        performance_results['daily_metrics'].write_parquet(perf_file)
                        saved_files[f'{event_type}_performance'] = perf_file
                    
                    # Save risk analysis
                    if 'risk_decomposition' in risk_results:
                        risk_file = os.path.join(
                            directories.get(f'{event_type}_results', directories['combined_results']),
                            f'{event_type}_risk_analysis_{timestamp}.parquet'
                        )
                        risk_results['risk_decomposition'].write_parquet(risk_file)
                        saved_files[f'{event_type}_risk'] = risk_file
        
        # Save combined summary
        summary_data = {
            'analysis_timestamp': timestamp,
            'parameters': {
                'window_days': self.WINDOW_DAYS,
                'analysis_window': self.ANALYSIS_WINDOW,
                'k1': self.K1,
                'k2': self.K2,
                'delta_t1': self.DELTA_T1,
                'delta_t2': self.DELTA_T2,
                'delta_t3': self.DELTA_T3,
                'delta': self.DELTA,
                'optimistic_bias': self.OPTIMISTIC_BIAS,
                'risk_free_rate': self.RISK_FREE_RATE
            },
            'data_summary': {
                'total_records': len(data),
                'unique_events': data['event_id'].n_unique(),
                'unique_tickers': data['ticker'].n_unique(),
                'date_range': [str(data['date'].min()), str(data['date'].max())]
            }
        }
        
        summary_file = os.path.join(
            directories['combined_results'],
            f'analysis_summary_{timestamp}.json'
        )
        
        import json
        with open(summary_file, 'w') as f:
            json.dump(summary_data, f, indent=2, default=str)
        
        saved_files['summary'] = summary_file
        
        logger.info(f"Analysis results saved: {len(saved_files)} files")
        return saved_files
    
    def run_full_analysis(self,
                         stock_files: Optional[List[str]] = None,
                         fda_events_file: Optional[str] = None,
                         earnings_events_file: Optional[str] = None,
                         use_mock_data: bool = True) -> Dict[str, Any]:
        """
        Run complete analysis pipeline with hardcoded parameters.
        
        Args:
            stock_files: List of CRSP parquet files
            fda_events_file: FDA events CSV file
            earnings_events_file: Earnings events CSV file
            use_mock_data: Whether to use mock data for development
            
        Returns:
            Dictionary containing all analysis results and saved file paths
        """
        logger.info("Starting full analysis pipeline")
        logger.info(f"Hardcoded parameters: WINDOW_DAYS={self.WINDOW_DAYS}, K1={self.K1}, K2={self.K2}")
        
        # Step 1: Setup result directories
        directories = self.setup_result_directories()
        
        # Step 2: Load and prepare data
        data = self.load_and_prepare_data(
            stock_files=stock_files,
            fda_events_file=fda_events_file,
            earnings_events_file=earnings_events_file,
            use_mock_data=use_mock_data
        )
        
        if data is None:
            logger.error("Failed to load data, aborting analysis")
            return {'error': 'Data loading failed'}
        
        # Step 3: Run volatility analysis
        volatility_results = self.run_volatility_analysis(data)
        
        # Step 4: Run multi-risk analysis
        risk_results = self.run_multi_risk_analysis(data)
        
        # Step 5: Calculate performance metrics
        performance_results = self.calculate_performance_metrics(data)
        
        # Step 6: Save all results
        saved_files = self.save_analysis_results(
            data, volatility_results, risk_results, performance_results, directories
        )
        
        # Compile final results
        final_results = {
            'status': 'completed',
            'parameters': {
                'window_days': self.WINDOW_DAYS,
                'analysis_window': self.ANALYSIS_WINDOW,
                'gjr_garch': {
                    'k1': self.K1,
                    'k2': self.K2,
                    'delta_t1': self.DELTA_T1,
                    'delta_t2': self.DELTA_T2,
                    'delta_t3': self.DELTA_T3,
                    'delta': self.DELTA
                },
                'optimization': {
                    'optimistic_bias': self.OPTIMISTIC_BIAS,
                    'risk_free_rate': self.RISK_FREE_RATE
                }
            },
            'data_summary': {
                'total_records': len(data),
                'unique_events': data['event_id'].n_unique(),
                'unique_tickers': data['ticker'].n_unique(),
                'analysis_window_records': len(data.filter(pl.col('in_analysis_window')))
            },
            'results': {
                'volatility_analysis': volatility_results,
                'risk_analysis': risk_results,
                'performance_metrics': performance_results
            },
            'saved_files': saved_files,
            'directories': directories
        }
        
        logger.info("Full analysis pipeline completed successfully")
        return final_results


def main():
    """Example usage of the analysis pipeline."""
    import tempfile
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("=" * 70)
    print("DYNAMIC ASSET PRICING MODEL - ANALYSIS PIPELINE")
    print("=" * 70)
    print()
    
    # Create temporary directory for results
    with tempfile.TemporaryDirectory() as temp_dir:
        results_dir = os.path.join(temp_dir, 'analysis_results')
        
        # Initialize pipeline
        pipeline = AnalysisPipeline(results_base_dir=results_dir)
        
        print("Pipeline Configuration:")
        print(f"  - Window Days: {pipeline.WINDOW_DAYS}")
        print(f"  - Analysis Window: {pipeline.ANALYSIS_WINDOW}")
        print(f"  - GJR-GARCH K1: {pipeline.K1}, K2: {pipeline.K2}")
        print(f"  - Delta T1: {pipeline.DELTA_T1}, T2: {pipeline.DELTA_T2}, T3: {pipeline.DELTA_T3}")
        print(f"  - Optimistic Bias: {pipeline.OPTIMISTIC_BIAS}")
        print(f"  - Risk-Free Rate: {pipeline.RISK_FREE_RATE}")
        print()
        
        # Run analysis with mock data
        print("Running analysis with mock data...")
        results = pipeline.run_full_analysis(use_mock_data=True)
        
        if results.get('status') == 'completed':
            print("✓ Analysis completed successfully!")
            print()
            print("Results Summary:")
            print(f"  - Total Records: {results['data_summary']['total_records']:,}")
            print(f"  - Unique Events: {results['data_summary']['unique_events']}")
            print(f"  - Unique Tickers: {results['data_summary']['unique_tickers']}")
            print(f"  - Analysis Window Records: {results['data_summary']['analysis_window_records']:,}")
            print()
            print("Saved Files:")
            for file_type, file_path in results['saved_files'].items():
                print(f"  - {file_type}: {os.path.basename(file_path)}")
            print()
            print("Result Directories:")
            for dir_name, dir_path in results['directories'].items():
                print(f"  - {dir_name}: {dir_path}")
        else:
            print("✗ Analysis failed")
            if 'error' in results:
                print(f"Error: {results['error']}")


if __name__ == "__main__":
    main()