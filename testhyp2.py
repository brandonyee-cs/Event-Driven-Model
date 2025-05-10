# runhypothesis2.py

import pandas as pd
import numpy as np
import os
import sys
import traceback
import polars as pl
from typing import List, Tuple, Dict, Any
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

# --- Configuration ---
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path: sys.path.append(current_dir)
try: 
    from src.event_processor import EventDataLoader, EventFeatureEngineer, EventAnalysis
    from src.models import GARCHModel, GJRGARCHModel, ThreePhaseVolatilityModel
    print("Successfully imported Event processor classes and models.")
except ImportError as e: 
    print(f"Error importing modules: {e}")
    print("Ensure 'event_processor.py' and 'models.py' are in the same directory or Python path.")
    sys.exit(1)

pl.Config.set_engine_affinity(engine="streaming")

# --- Hardcoded Analysis Parameters ---
# Shared stock files for both analyses
STOCK_FILES = [
    "/home/d87016661/crsp_dsf-2000-2001.parquet",
    "/home/d87016661/crsp_dsf-2002-2003.parquet",
    "/home/d87016661/crsp_dsf-2004-2005.parquet",
    "/home/d87016661/crsp_dsf-2006-2007.parquet",
    "/home/d87016661/crsp_dsf-2008-2009.parquet",
    "/home/d87016661/crsp_dsf-2010-2011.parquet",
    "/home/d87016661/crsp_dsf-2016-2017.parquet",
    "/home/d87016661/crsp_dsf-2018-2019.parquet",
    "/home/d87016661/crsp_dsf-2020-2021.parquet",
    "/home/d87016661/crsp_dsf-2022-2023.parquet",
    "/home/d87016661/crsp_dsf-2024-2025.parquet"
]

# FDA event specific parameters
FDA_EVENT_FILE = "/home/d87016661/fda_ticker_list_2000_to_2024.csv"
FDA_RESULTS_DIR = "results/hypothesis2/results_fda/"
FDA_FILE_PREFIX = "fda"
FDA_EVENT_DATE_COL = "Approval Date"
FDA_TICKER_COL = "ticker"

# Earnings event specific parameters
EARNINGS_EVENT_FILE = "/home/d87016661/detail_history_actuals.csv"
EARNINGS_RESULTS_DIR = "results/hypothesis2/results_earnings/"
EARNINGS_FILE_PREFIX = "earnings"
EARNINGS_EVENT_DATE_COL = "ANNDATS"
EARNINGS_TICKER_COL = "ticker"

# Shared analysis parameters
WINDOW_DAYS = 60
ANALYSIS_WINDOW = (-30, 30)
PRE_EVENT_WINDOW = (-15, -1)  # Window for pre-event volatility innovations
POST_EVENT_WINDOW = (1, 15)   # Window for post-event volatility persistence

# GARCH model parameters
MODEL_TYPES = ['garch', 'gjr']  # Test both types for asymmetric response
GARCH_PARAMS = {
    'garch': {'omega': 0.00001, 'alpha': 0.05, 'beta': 0.90},
    'gjr': {'omega': 0.00001, 'alpha': 0.03, 'beta': 0.90, 'gamma': 0.04}
}

# Define prediction windows for return forecasting
PREDICTION_WINDOWS = [1, 3, 5]  # Days ahead to predict returns

class Hypothesis2Analyzer:
    """Class for testing Hypothesis 2 about volatility innovations."""
    
    def __init__(self, analyzer: EventAnalysis, results_dir: str, file_prefix: str):
        """
        Initialize Hypothesis2Analyzer.
        
        Parameters:
        -----------
        analyzer : EventAnalysis
            Instance of EventAnalysis with data loaded
        results_dir : str
            Directory to save results
        file_prefix : str
            Prefix for output files
        """
        self.analyzer = analyzer
        self.results_dir = results_dir
        self.file_prefix = file_prefix
        self.return_col = 'ret'
        self.analysis_window = ANALYSIS_WINDOW
        self.pre_event_window = PRE_EVENT_WINDOW
        self.post_event_window = POST_EVENT_WINDOW
        self.prediction_windows = PREDICTION_WINDOWS
        self.model_types = MODEL_TYPES
        self.garch_params = GARCH_PARAMS
        
        # Results storage
        self.innovations_data = {}
        self.prediction_results = {}
        self.asymmetry_results = {}
        self.persistence_results = {}
        
        # Create results directory
        os.makedirs(results_dir, exist_ok=True)
    
    def _fit_garch_models(self, event_id: str, event_data: pl.DataFrame) -> Dict[str, Any]:
        """
        Fit GARCH models to event data and extract volatility innovations.
        
        Parameters:
        -----------
        event_id : str
            Identifier for the event
        event_data : pl.DataFrame
            Data for a single event
            
        Returns:
        --------
        Dict[str, Any]
            Dictionary of fitted models and extracted features
        """
        event_days = event_data.select('days_to_event').to_series().to_list()
        event_returns = event_data.select(self.return_col).to_series()
        
        # Skip if insufficient data
        if len(event_returns) < 30:
            return None
        
        # Initialize result dictionary
        result = {
            'event_id': event_id,
            'ticker': event_data.select('ticker').head(1).item(),
            'days_to_event': event_days,
            'returns': event_returns.to_numpy(),
            'models': {},
            'innovations': {},
            'volatility': {},
            'persistence': {}
        }
        
        # Fit both GARCH and GJR-GARCH models
        for model_type in self.model_types:
            try:
                # Initialize model with parameters
                if model_type == 'garch':
                    model = GARCHModel(**self.garch_params['garch'])
                else:  # 'gjr'
                    model = GJRGARCHModel(**self.garch_params['gjr'])
                
                # Fit model to returns
                model.fit(event_returns)
                
                # Extract volatility series
                volatility = np.sqrt(model.variance_history)
                
                # Calculate volatility innovations (difference between realized and expected volatility)
                innovations = model.volatility_innovations()
                
                # Store model and features
                result['models'][model_type] = model
                result['volatility'][model_type] = volatility
                result['innovations'][model_type] = innovations
                
                # Calculate post-event volatility persistence
                # (Ratio of post-event to pre-event average volatility)
                pre_event_indices = [i for i, day in enumerate(event_days) 
                                      if self.pre_event_window[0] <= day <= self.pre_event_window[1]]
                post_event_indices = [i for i, day in enumerate(event_days) 
                                       if self.post_event_window[0] <= day <= self.post_event_window[1]]
                
                if pre_event_indices and post_event_indices:
                    pre_vol = np.mean(volatility[pre_event_indices])
                    post_vol = np.mean(volatility[post_event_indices])
                    if pre_vol > 0:
                        persistence = post_vol / pre_vol
                    else:
                        persistence = np.nan
                    result['persistence'][model_type] = persistence
                else:
                    result['persistence'][model_type] = np.nan
                
            except Exception as e:
                print(f"Error fitting {model_type} model for event {event_id}: {e}")
                result['models'][model_type] = None
                result['volatility'][model_type] = np.array([])
                result['innovations'][model_type] = np.array([])
                result['persistence'][model_type] = np.nan
        
        return result
    
    def _calculate_future_returns(self, event_data: pl.DataFrame) -> Dict[int, np.ndarray]:
        """
        Calculate future returns for various prediction windows.
        
        Parameters:
        -----------
        event_data : pl.DataFrame
            Data for a single event
            
        Returns:
        --------
        Dict[int, np.ndarray]
            Dictionary mapping prediction window to future returns
        """
        future_returns = {}
        
        for window in self.prediction_windows:
            # Use with_columns to calculate future returns
            with_future = event_data.with_columns(
                pl.col(self.return_col).shift(-window).over('event_id').alias(f'future_ret_{window}')
            )
            
            # Extract the column
            future_ret = with_future.select(f'future_ret_{window}').to_series().to_numpy()
            future_returns[window] = future_ret
        
        return future_returns
    
    def analyze_volatility_innovations(self):
        """
        Analyze volatility innovations and test their predictive power for returns.
        """
        print("\n--- Analyzing Volatility Innovations (Hypothesis 2) ---")
        
        if self.analyzer.data is None:
            print("Error: No data available for analysis.")
            return False
        
        # Extend window for GARCH estimation
        extended_window = (min(self.analysis_window[0], -60), max(self.analysis_window[1], 60))
        analysis_data = self.analyzer.data.filter(
            (pl.col('days_to_event') >= extended_window[0]) &
            (pl.col('days_to_event') <= extended_window[1])
        ).sort(['event_id', 'days_to_event'])
        
        if analysis_data.is_empty():
            print(f"Error: No data found within extended window {extended_window}")
            return False
        
        # Group by event_id and fit GARCH models
        event_ids = analysis_data.select('event_id').unique().to_series().to_list()
        print(f"Processing {len(event_ids)} events...")
        
        # Sample a subset of events for detailed analysis (limit to 100 for computational efficiency)
        if len(event_ids) > 100:
            np.random.seed(42)
            sample_event_ids = np.random.choice(event_ids, size=100, replace=False)
        else:
            sample_event_ids = event_ids
        
        # Store volatility innovations and future returns by event
        all_events_data = []
        
        for event_id in sample_event_ids:
            try:
                # Filter data for this event
                event_data = analysis_data.filter(pl.col('event_id') == event_id)
                
                # Fit GARCH models and extract volatility innovations
                event_result = self._fit_garch_models(event_id, event_data)
                
                if event_result is None:
                    continue
                
                # Calculate future returns for prediction windows
                future_returns = self._calculate_future_returns(event_data)
                event_result['future_returns'] = future_returns
                
                all_events_data.append(event_result)
                
            except Exception as e:
                print(f"Error processing event {event_id}: {e}")
                continue
        
        # Save volatility innovations data
        self.innovations_data = all_events_data
        
        print(f"Successfully analyzed {len(all_events_data)} events.")
        
        # Test Hypothesis 2.1: Pre-event volatility innovations predict subsequent returns
        self._test_prediction_power()
        
        # Test Hypothesis 2.2: Post-event volatility persistence extends elevated expected returns
        self._test_volatility_persistence()
        
        # Test Hypothesis 2.3: Asymmetric volatility response correlates with asymmetric price adjustment
        self._test_asymmetric_response()
        
        return True
    
    def _test_prediction_power(self):
        """
        Test the predictive power of pre-event volatility innovations for subsequent returns.
        """
        print("\n--- Testing Predictive Power of Volatility Innovations ---")
        
        if not self.innovations_data:
            print("Error: No volatility data available.")
            return
        
        regression_results = {}
        
        for model_type in self.model_types:
            regression_results[model_type] = {}
            
            for window in self.prediction_windows:
                print(f"Testing {model_type} innovations to predict {window}-day future returns...")
                
                # Prepare data for regression
                X_data = []  # Pre-event volatility innovations (predictor)
                y_data = []  # Future returns (target)
                
                for event_data in self.innovations_data:
                    # Skip if model or data missing
                    if (model_type not in event_data['models'] or 
                        event_data['models'][model_type] is None or
                        'future_returns' not in event_data or
                        window not in event_data['future_returns']):
                        continue
                    
                    days = event_data['days_to_event']
                    innovations = event_data['innovations'][model_type]
                    future_ret = event_data['future_returns'][window]
                    
                    # Find indices for pre-event window
                    pre_event_indices = [i for i, day in enumerate(days) 
                                         if self.pre_event_window[0] <= day <= self.pre_event_window[1]]
                    
                    # Find index for event (day 0)
                    event_index = days.index(0) if 0 in days else None
                    
                    if pre_event_indices and event_index and event_index < len(future_ret):
                        # Calculate average pre-event innovation
                        pre_event_innovation = np.mean(innovations[pre_event_indices])
                        
                        # Get future return from event day
                        event_future_return = future_ret[event_index]
                        
                        # Add to regression data
                        if not np.isnan(pre_event_innovation) and not np.isnan(event_future_return):
                            X_data.append(pre_event_innovation)
                            y_data.append(event_future_return)
                
                # Run regression
                if len(X_data) > 10:  # Ensure sufficient data
                    X = np.array(X_data).reshape(-1, 1)
                    y = np.array(y_data)
                    
                    # Add this new check here
                    if np.var(X) < 1e-10:  # Check for effectively zero variance
                        print(f"  Warning: Insufficient variance in X data for {model_type} innovations and {window}-day returns.")
                        print(f"  X variance: {np.var(X):.10f}, X range: [{np.min(X):.6f}, {np.max(X):.6f}]")
                        regression_results[model_type][window] = {
                            'slope': 0,
                            'intercept': np.mean(y) if len(y) > 0 else 0,
                            'r_squared': 0,
                            'p_value': 1.0,
                            'std_err': 0,
                            'n_samples': len(X),
                            'error': 'Insufficient X variance for regression'
                        }
                    else:
                        # Simple OLS regression - existing code
                        slope, intercept, r_value, p_value, std_err = stats.linregress(X.flatten(), y)
                        
                        # Store results - existing code
                        regression_results[model_type][window] = {
                            'slope': slope,
                            'intercept': intercept,
                            'r_squared': r_value**2,
                            'p_value': p_value,
                            'std_err': std_err,
                            'n_samples': len(X),
                            'X_data': X,
                            'y_data': y
                        }
                        
                        print(f"  Regression results: slope={slope:.4f}, R²={r_value**2:.4f}, p={p_value:.4f}, n={len(X)}")
                else:
                    # Keep existing code for insufficient data case
                    print(f"  Insufficient data for regression (n={len(X_data)})")
                    regression_results[model_type][window] = None
        
        # Save regression results
        self.prediction_results = regression_results
        
        # Create summary
        summary_rows = []
        significant_results = 0
        
        for model_type in self.model_types:
            for window in self.prediction_windows:
                if regression_results[model_type][window] is not None:
                    result = regression_results[model_type][window]
                    significance = "***" if result['p_value'] < 0.01 else "**" if result['p_value'] < 0.05 else "*" if result['p_value'] < 0.1 else ""
                    
                    if significance:
                        significant_results += 1
                    
                    summary_rows.append({
                        'model_type': model_type,
                        'prediction_window': window,
                        'slope': result['slope'],
                        'r_squared': result['r_squared'],
                        'p_value': result['p_value'],
                        'n_samples': result['n_samples'],
                        'significant': significance != "",
                        'significance': significance
                    })
        
        # Convert to DataFrame
        if summary_rows:
            summary_df = pl.DataFrame(summary_rows)
            
            # Save results
            summary_df.write_csv(os.path.join(self.results_dir, f"{self.file_prefix}_prediction_results.csv"))
            
            # Check if Hypothesis 2.1 is supported
            h2_1_supported = significant_results > 0
            print(f"\nHypothesis 2.1 (Innovations predict returns): {'SUPPORTED' if h2_1_supported else 'NOT SUPPORTED'}")
            print(f"Found {significant_results} significant relationships out of {len(summary_rows)} tests.")
            
            # Save in a format for later comparison
            h2_1_result_df = pl.DataFrame({
                'hypothesis': ['H2.1: Pre-event volatility innovations predict returns'],
                'result': [h2_1_supported],
                'significant_tests': [significant_results],
                'total_tests': [len(summary_rows)]
            })
            h2_1_result_df.write_csv(os.path.join(self.results_dir, f"{self.file_prefix}_hypothesis2_1_test.csv"))
            
            # Generate plots for significant relationships
            self._plot_prediction_relationships()
        else:
            print("No valid regression results to save.")
    
    def _plot_prediction_relationships(self):
        """
        Plot regression relationships between volatility innovations and future returns.
        """
        try:
            for model_type in self.model_types:
                for window in self.prediction_windows:
                    if (model_type in self.prediction_results and 
                        window in self.prediction_results[model_type] and 
                        self.prediction_results[model_type][window] is not None):
                        
                        result = self.prediction_results[model_type][window]
                        
                        # Skip if not significant
                        if result['p_value'] >= 0.1:
                            continue
                        
                        # Create plot
                        fig, ax = plt.subplots(figsize=(10, 6))
                        
                        # Scatter plot
                        ax.scatter(result['X_data'], result['y_data'], alpha=0.6, 
                                   label=f'n={result["n_samples"]}')
                        
                        # Regression line
                        x_line = np.linspace(min(result['X_data']), max(result['X_data']), 100)
                        y_line = result['slope'] * x_line + result['intercept']
                        ax.plot(x_line, y_line, color='red', 
                                label=f'y = {result["slope"]:.4f}x + {result["intercept"]:.4f}')
                        
                        # Add statistics
                        significance = "***" if result['p_value'] < 0.01 else "**" if result['p_value'] < 0.05 else "*" if result['p_value'] < 0.1 else ""
                        stats_text = (f"R² = {result['r_squared']:.4f}\n"
                                      f"p-value = {result['p_value']:.4f} {significance}\n"
                                      f"Slope = {result['slope']:.4f} ± {result['std_err']:.4f}")
                        
                        ax.text(0.05, 0.95, stats_text, transform=ax.transAxes,
                                verticalalignment='top', bbox=dict(boxstyle='round', alpha=0.5))
                        
                        ax.set_title(f'{model_type.upper()}-GARCH Volatility Innovations vs. {window}-Day Future Returns')
                        ax.set_xlabel('Pre-Event Volatility Innovations (Avg)')
                        ax.set_ylabel(f'{window}-Day Future Returns')
                        ax.legend()
                        ax.grid(True, alpha=0.3)
                        
                        # Save plot
                        plot_path = os.path.join(self.results_dir, 
                                                f"{self.file_prefix}_prediction_{model_type}_{window}day.png")
                        plt.savefig(plot_path, dpi=200, bbox_inches='tight')
                        plt.close(fig)
        
        except Exception as e:
            print(f"Error creating prediction plots: {e}")
            traceback.print_exc()
    
    def _test_volatility_persistence(self):
        """
        Test if post-event volatility persistence extends elevated expected returns.
        """
        print("\n--- Testing Volatility Persistence Impact on Returns ---")
        
        if not self.innovations_data:
            print("Error: No volatility data available.")
            return
        
        # Prepare data for regression
        persistence_results = {}
        
        for model_type in self.model_types:
            persistence_data = []
            
            for event_data in self.innovations_data:
                # Skip if model or data missing
                if (model_type not in event_data['models'] or 
                    event_data['models'][model_type] is None or
                    'persistence' not in event_data or
                    model_type not in event_data['persistence']):
                    continue
                
                days = event_data['days_to_event']
                returns = event_data['returns']
                persistence = event_data['persistence'][model_type]
                
                # Find indices for post-event window
                post_event_indices = [i for i, day in enumerate(days) 
                                     if self.post_event_window[0] <= day <= self.post_event_window[1]]
                
                if post_event_indices and not np.isnan(persistence):
                    # Calculate average post-event return
                    post_event_return = np.mean(returns[post_event_indices])
                    
                    # Skip if NaN
                    if not np.isnan(post_event_return):
                        persistence_data.append({
                            'event_id': event_data['event_id'],
                            'persistence': persistence,
                            'post_event_return': post_event_return
                        })
            
            # Run regression if enough data
            if len(persistence_data) > 10:
                persistence_df = pl.DataFrame(persistence_data)
                
                # Extract arrays for regression
                X = persistence_df.select('persistence').to_numpy()
                y = persistence_df.select('post_event_return').to_numpy()
                
                # OLS Regression
                slope, intercept, r_value, p_value, std_err = stats.linregress(X.flatten(), y.flatten())
                
                # Store results
                persistence_results[model_type] = {
                    'slope': slope,
                    'intercept': intercept,
                    'r_squared': r_value**2,
                    'p_value': p_value,
                    'std_err': std_err,
                    'n_samples': len(persistence_data),
                    'data': persistence_df
                }
                
                print(f"  {model_type.upper()}-GARCH: slope={slope:.4f}, R²={r_value**2:.4f}, p={p_value:.4f}, n={len(persistence_data)}")
            else:
                print(f"  Insufficient data for {model_type.upper()}-GARCH persistence analysis (n={len(persistence_data)})")
                persistence_results[model_type] = None
        
        # Save persistence results
        self.persistence_results = persistence_results
        
        # Create summary
        summary_rows = []
        significant_results = 0
        
        for model_type, result in persistence_results.items():
            if result is not None:
                significance = "***" if result['p_value'] < 0.01 else "**" if result['p_value'] < 0.05 else "*" if result['p_value'] < 0.1 else ""
                
                if significance:
                    significant_results += 1
                
                summary_rows.append({
                    'model_type': model_type,
                    'slope': result['slope'],
                    'r_squared': result['r_squared'],
                    'p_value': result['p_value'],
                    'n_samples': result['n_samples'],
                    'significant': significance != "",
                    'significance': significance
                })
        
        # Convert to DataFrame
        if summary_rows:
            summary_df = pl.DataFrame(summary_rows)
            
            # Save results
            summary_df.write_csv(os.path.join(self.results_dir, f"{self.file_prefix}_persistence_results.csv"))
            
            # Check if Hypothesis 2.2 is supported
            h2_2_supported = significant_results > 0
            print(f"\nHypothesis 2.2 (Volatility persistence extends elevated returns): {'SUPPORTED' if h2_2_supported else 'NOT SUPPORTED'}")
            print(f"Found {significant_results} significant relationships out of {len(summary_rows)} tests.")
            
            # Save in a format for later comparison
            h2_2_result_df = pl.DataFrame({
                'hypothesis': ['H2.2: Post-event volatility persistence extends elevated returns'],
                'result': [h2_2_supported],
                'significant_tests': [significant_results],
                'total_tests': [len(summary_rows)]
            })
            h2_2_result_df.write_csv(os.path.join(self.results_dir, f"{self.file_prefix}_hypothesis2_2_test.csv"))
            
            # Generate plots
            self._plot_persistence_relationships()
        else:
            print("No valid persistence results to save.")
    
    def _plot_persistence_relationships(self):
        """
        Plot relationships between volatility persistence and post-event returns.
        """
        try:
            for model_type, result in self.persistence_results.items():
                if result is not None:
                    # Create plot
                    fig, ax = plt.subplots(figsize=(10, 6))
                    
                    # Convert to pandas for scatter plot
                    data_pd = result['data'].to_pandas()
                    
                    # Scatter plot
                    ax.scatter(data_pd['persistence'], data_pd['post_event_return'], alpha=0.6, 
                               label=f'n={result["n_samples"]}')
                    
                    # Regression line
                    x_range = np.linspace(data_pd['persistence'].min(), data_pd['persistence'].max(), 100)
                    ax.plot(x_range, result['slope'] * x_range + result['intercept'], color='red',
                            label=f'y = {result["slope"]:.4f}x + {result["intercept"]:.4f}')
                    
                    # Add statistics
                    significance = "***" if result['p_value'] < 0.01 else "**" if result['p_value'] < 0.05 else "*" if result['p_value'] < 0.1 else ""
                    stats_text = (f"R² = {result['r_squared']:.4f}\n"
                                  f"p-value = {result['p_value']:.4f} {significance}\n"
                                  f"Slope = {result['slope']:.4f} ± {result['std_err']:.4f}")
                    
                    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes,
                            verticalalignment='top', bbox=dict(boxstyle='round', alpha=0.5))
                    
                    ax.set_title(f'{model_type.upper()}-GARCH Volatility Persistence vs. Post-Event Returns')
                    ax.set_xlabel('Volatility Persistence (Post/Pre Ratio)')
                    ax.set_ylabel('Average Post-Event Return')
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    
                    # Save plot
                    plot_path = os.path.join(self.results_dir, 
                                            f"{self.file_prefix}_persistence_{model_type}.png")
                    plt.savefig(plot_path, dpi=200, bbox_inches='tight')
                    plt.close(fig)
        
        except Exception as e:
            print(f"Error creating persistence plots: {e}")
            traceback.print_exc()
    
    def _test_asymmetric_response(self):
        """
        Test if asymmetric volatility response (GJR-GARCH) correlates with asymmetric price adjustment.
        """
        print("\n--- Testing Asymmetric Volatility Response ---")
        
        if not self.innovations_data:
            print("Error: No volatility data available.")
            return
        
        # Check if we have both GARCH and GJR-GARCH results
        if 'garch' not in self.model_types or 'gjr' not in self.model_types:
            print("Error: Both GARCH and GJR-GARCH models required for asymmetry test.")
            return
        
        # Test advantage of GJR-GARCH over GARCH for negative vs. positive returns
        asymmetry_data = []
        
        for event_data in self.innovations_data:
            # Skip if either model is missing
            if ('garch' not in event_data['models'] or event_data['models']['garch'] is None or
                'gjr' not in event_data['models'] or event_data['models']['gjr'] is None):
                continue
            
            days = event_data['days_to_event']
            returns = event_data['returns']
            
            # Find index for day before event
            pre_day_index = next((i for i, day in enumerate(days) if day == -1), None)
            # Find index for event day
            event_day_index = next((i for i, day in enumerate(days) if day == 0), None)
            
            if pre_day_index is not None and event_day_index is not None:
                # Get pre-event to event return
                event_return = returns[event_day_index]
                
                # Skip if NaN
                if np.isnan(event_return):
                    continue
                
                # Get volatilities from both models
                garch_vol = event_data['volatility']['garch'][event_day_index] if len(event_data['volatility']['garch']) > event_day_index else np.nan
                gjr_vol = event_data['volatility']['gjr'][event_day_index] if len(event_data['volatility']['gjr']) > event_day_index else np.nan
                
                if not np.isnan(garch_vol) and not np.isnan(gjr_vol):
                    # Calculate relative improvement of GJR-GARCH over GARCH
                    # For volatility prediction
                    vol_difference = gjr_vol - garch_vol
                    
                    asymmetry_data.append({
                        'event_id': event_data['event_id'],
                        'event_return': event_return,
                        'garch_vol': garch_vol,
                        'gjr_vol': gjr_vol,
                        'vol_difference': vol_difference,
                        'negative_return': event_return < 0
                    })
        
        # Create DataFrame
        if asymmetry_data:
            asymmetry_df = pl.DataFrame(asymmetry_data)
            
            # Split by negative/positive returns
            negative_df = asymmetry_df.filter(pl.col('negative_return') == True)
            positive_df = asymmetry_df.filter(pl.col('negative_return') == False)
            
            # Calculate average differences
            if not negative_df.is_empty() and not positive_df.is_empty():
                avg_diff_negative = negative_df.select(pl.col('vol_difference').mean()).item()
                avg_diff_positive = positive_df.select(pl.col('vol_difference').mean()).item()
                
                # Calculate t-test for difference
                if len(negative_df) > 5 and len(positive_df) > 5:
                    neg_diff = negative_df.select('vol_difference').to_numpy().flatten()
                    pos_diff = positive_df.select('vol_difference').to_numpy().flatten()
                    
                    t_stat, p_value = stats.ttest_ind(neg_diff, pos_diff, equal_var=False)
                    
                    # Store results
                    self.asymmetry_results = {
                        'avg_diff_negative': avg_diff_negative,
                        'avg_diff_positive': avg_diff_positive,
                        'diff_of_diffs': avg_diff_negative - avg_diff_positive,
                        't_stat': t_stat,
                        'p_value': p_value,
                        'n_negative': len(negative_df),
                        'n_positive': len(positive_df),
                        'negative_data': negative_df,
                        'positive_data': positive_df
                    }
                    
                    print(f"  Average GJR-GARCH vs. GARCH difference for negative returns: {avg_diff_negative:.6f}")
                    print(f"  Average GJR-GARCH vs. GARCH difference for positive returns: {avg_diff_positive:.6f}")
                    print(f"  Difference of differences: {avg_diff_negative - avg_diff_positive:.6f}")
                    print(f"  t-statistic: {t_stat:.4f}, p-value: {p_value:.4f}")
                    
                    # Check if hypothesis is supported
                    asymmetry_supported = p_value < 0.1 and avg_diff_negative > avg_diff_positive
                    print(f"\nHypothesis 2.3 (Asymmetric volatility response): {'SUPPORTED' if asymmetry_supported else 'NOT SUPPORTED'}")
                    
                    # Save results
                    asymmetry_result_df = pl.DataFrame([{
                        'avg_diff_negative': avg_diff_negative,
                        'avg_diff_positive': avg_diff_positive,
                        'diff_of_diffs': avg_diff_negative - avg_diff_positive,
                        't_stat': t_stat,
                        'p_value': p_value,
                        'n_negative': len(negative_df),
                        'n_positive': len(positive_df),
                        'significant': p_value < 0.1,
                        'hypothesis_supported': asymmetry_supported
                    }])
                    
                    asymmetry_result_df.write_csv(os.path.join(self.results_dir,
                                                             f"{self.file_prefix}_asymmetry_results.csv"))
                    
                    # Save in a format for later comparison
                    h2_3_result_df = pl.DataFrame({
                        'hypothesis': ['H2.3: Asymmetric volatility response correlates with price adjustment'],
                        'result': [asymmetry_supported],
                        'p_value': [p_value],
                        'diff_of_diffs': [avg_diff_negative - avg_diff_positive]
                    })
                    h2_3_result_df.write_csv(os.path.join(self.results_dir,
                                                         f"{self.file_prefix}_hypothesis2_3_test.csv"))
                    
                    # Generate plot
                    self._plot_asymmetry_results()
                else:
                    print(f"  Insufficient data for asymmetry test (negative: {len(negative_df)}, positive: {len(positive_df)})")
            else:
                print("  Insufficient data for asymmetry analysis (missing negative or positive returns)")
        else:
            print("No valid asymmetry data to analyze.")
    
    def _plot_asymmetry_results(self):
        """
        Plot asymmetric volatility response results.
        """
        try:
            result = self.asymmetry_results
            
            # Bar plot comparing negative and positive returns
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Data
            labels = ['Negative Returns', 'Positive Returns']
            values = [result['avg_diff_negative'], result['avg_diff_positive']]
            
            # Bar chart
            bars = ax.bar(labels, values, color=['red', 'green'], alpha=0.7)
            
            # Add value labels
            for i, v in enumerate(values):
                ax.text(i, v + 0.0001 if v > 0 else v - 0.0001, 
                        f"{v:.6f}", ha='center', va='bottom' if v > 0 else 'top')
            
            # Add statistical significance
            significance = "***" if result['p_value'] < 0.01 else "**" if result['p_value'] < 0.05 else "*" if result['p_value'] < 0.1 else ""
            
            ax.set_title(f'GJR-GARCH vs. GARCH Volatility Difference')
            ax.set_ylabel('Average Volatility Difference')
            ax.grid(True, axis='y', alpha=0.3)
            
            # Add t-test result
            stats_text = (f"Difference of differences: {result['diff_of_diffs']:.6f}\n"
                          f"t-statistic: {result['t_stat']:.4f}\n"
                          f"p-value: {result['p_value']:.4f} {significance}\n"
                          f"n_negative: {result['n_negative']}, n_positive: {result['n_positive']}")
            
            ax.text(0.5, 0.01, stats_text, ha='center', transform=ax.transAxes,
                    bbox=dict(boxstyle='round', alpha=0.5))
            
            # Save plot
            plot_path = os.path.join(self.results_dir, f"{self.file_prefix}_asymmetry_results.png")
            plt.savefig(plot_path, dpi=200, bbox_inches='tight')
            plt.close(fig)
            
            # Create boxplot comparing distributions
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Convert to pandas for boxplot
            neg_data = result['negative_data'].select('vol_difference').to_numpy().flatten()
            pos_data = result['positive_data'].select('vol_difference').to_numpy().flatten()
            
            # Boxplot
            boxplot = ax.boxplot([neg_data, pos_data], labels=labels,
                                patch_artist=True, notch=True)
            
            # Customize boxplot colors
            boxplot['boxes'][0].set(facecolor='red', alpha=0.6)
            boxplot['boxes'][1].set(facecolor='green', alpha=0.6)
            
            ax.set_title('Distribution of GJR-GARCH vs. GARCH Volatility Differences')
            ax.set_ylabel('Volatility Difference')
            ax.grid(True, axis='y', alpha=0.3)
            
            # Add t-test result
            ax.text(0.5, 0.01, stats_text, ha='center', transform=ax.transAxes,
                    bbox=dict(boxstyle='round', alpha=0.5))
            
            # Save plot
            plot_path = os.path.join(self.results_dir, f"{self.file_prefix}_asymmetry_boxplot.png")
            plt.savefig(plot_path, dpi=200, bbox_inches='tight')
            plt.close(fig)
        
        except Exception as e:
            print(f"Error creating asymmetry plots: {e}")
            traceback.print_exc()
    
    def generate_summary_report(self):
        """
        Generate a summary report of all Hypothesis 2 tests.
        """
        print("\n--- Generating Hypothesis 2 Summary Report ---")
        
        # Create summary of all sub-hypotheses
        h2_1_file = os.path.join(self.results_dir, f"{self.file_prefix}_hypothesis2_1_test.csv")
        h2_2_file = os.path.join(self.results_dir, f"{self.file_prefix}_hypothesis2_2_test.csv")
        h2_3_file = os.path.join(self.results_dir, f"{self.file_prefix}_hypothesis2_3_test.csv")
        
        try:
            # Check if all files exist
            if all(os.path.exists(f) for f in [h2_1_file, h2_2_file, h2_3_file]):
                h2_1 = pl.read_csv(h2_1_file)
                h2_2 = pl.read_csv(h2_2_file)
                h2_3 = pl.read_csv(h2_3_file)
                
                # Create summary
                summary_data = {
                    'sub_hypothesis': [
                        'H2.1: Pre-event volatility innovations predict returns',
                        'H2.2: Post-event volatility persistence extends elevated returns',
                        'H2.3: Asymmetric volatility response correlates with price adjustment'
                    ],
                    'result': [
                        h2_1['result'].item(),
                        h2_2['result'].item(),
                        h2_3['result'].item()
                    ],
                    'details': [
                        f"{h2_1['significant_tests'].item()}/{h2_1['total_tests'].item()} significant tests",
                        f"{h2_2['significant_tests'].item()}/{h2_2['total_tests'].item()} significant tests",
                        f"p={h2_3['p_value'].item():.4f}, diff={h2_3['diff_of_diffs'].item():.6f}"
                    ]
                }
                
                summary_df = pl.DataFrame(summary_data)
                
                # Overall hypothesis result
                overall_result = all(summary_df['result'])
                overall_summary = pl.DataFrame({
                    'hypothesis': ['H2: GARCH-estimated volatility innovations proxy for impact uncertainty'],
                    'result': [overall_result],
                    'supported_sub_hypotheses': [sum(summary_df['result'])],
                    'total_sub_hypotheses': [len(summary_df)]
                })
                
                # Save summary
                summary_df.write_csv(os.path.join(self.results_dir, f"{self.file_prefix}_h2_sub_hypotheses.csv"))
                overall_summary.write_csv(os.path.join(self.results_dir, f"{self.file_prefix}_hypothesis2_overall.csv"))
                
                print(f"\nHypothesis 2 Overall: {'SUPPORTED' if overall_result else 'PARTIALLY SUPPORTED'}")
                print(f"Supported {sum(summary_df['result'])}/{len(summary_df)} sub-hypotheses")
                
                # Generate summary plot
                self._plot_hypothesis_summary(summary_df, overall_result)
                
                return overall_summary
            else:
                print("Missing results files for one or more sub-hypotheses.")
                return None
        
        except Exception as e:
            print(f"Error generating summary report: {e}")
            traceback.print_exc()
            return None
    
    def _plot_hypothesis_summary(self, summary_df: pl.DataFrame, overall_result: bool):
        """
        Create a summary plot of Hypothesis 2 results.
        
        Parameters:
        -----------
        summary_df : pl.DataFrame
            DataFrame with sub-hypothesis results
        overall_result : bool
            Whether the overall hypothesis is supported
        """
        try:
            # Convert to pandas for plotting
            summary_pd = summary_df.to_pandas()
            
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # Bar chart of sub-hypotheses
            y_pos = np.arange(len(summary_pd))
            colors = ['green' if r else 'red' for r in summary_pd['result']]
            
            ax.barh(y_pos, [1] * len(summary_pd), color=colors, alpha=0.6)
            
            # Add hypothesis labels
            for i, (hypothesis, result) in enumerate(zip(summary_pd['sub_hypothesis'], summary_pd['result'])):
                result_text = "SUPPORTED" if result else "NOT SUPPORTED"
                ax.text(0.5, i, f"{hypothesis}: {result_text}", 
                        ha='center', va='center', color='black', fontweight='bold')
            
            # Remove axes ticks and spines
            ax.set_yticks([])
            ax.set_xticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            
            # Set title
            overall_text = "SUPPORTED" if overall_result else "PARTIALLY SUPPORTED"
            ax.set_title(f"Hypothesis 2: GARCH-estimated volatility innovations as impact uncertainty proxy\nOverall: {overall_text}")
            
            # Add details
            for i, detail in enumerate(summary_pd['details']):
                ax.text(0.95, i, detail, ha='right', va='center', 
                        bbox=dict(boxstyle='round', alpha=0.1))
            
            # Save plot
            plot_path = os.path.join(self.results_dir, f"{self.file_prefix}_hypothesis2_summary.png")
            plt.savefig(plot_path, dpi=200, bbox_inches='tight')
            plt.close(fig)
        
        except Exception as e:
            print(f"Error creating hypothesis summary plot: {e}")
            traceback.print_exc()

def run_fda_analysis():
    """
    Runs the FDA event analysis to test Hypothesis 2.
    """
    print("\n=== Starting FDA Approval Event Analysis for Hypothesis 2 ===")

    # --- Path Validation & Results Dir Creation ---
    if not os.path.exists(FDA_EVENT_FILE): 
        print(f"\n*** Error: FDA event file not found: {FDA_EVENT_FILE} ***")
        return False
    
    missing_stock = [f for f in STOCK_FILES if not os.path.exists(f)]
    if missing_stock: 
        print(f"\n*** Error: Stock file(s) not found: {missing_stock} ***")
        return False
    
    print("FDA file paths validated.")
    try:
        os.makedirs(FDA_RESULTS_DIR, exist_ok=True)
        print(f"FDA results will be saved to: {os.path.abspath(FDA_RESULTS_DIR)}")
    except OSError as oe:
        print(f"\n*** Error creating FDA results directory '{FDA_RESULTS_DIR}': {oe} ***")
        return False

    try:
        # --- Initialize Components ---
        print("\nInitializing FDA components...")
        data_loader = EventDataLoader(
            event_path=FDA_EVENT_FILE, 
            stock_paths=STOCK_FILES, 
            window_days=WINDOW_DAYS,
            event_date_col=FDA_EVENT_DATE_COL,
            ticker_col=FDA_TICKER_COL
        )
        feature_engineer = EventFeatureEngineer(prediction_window=3)
        analyzer = EventAnalysis(data_loader, feature_engineer)
        print("FDA components initialized.")

        # --- Load Data ---
        print("\nLoading and preparing FDA data...")
        analyzer.data = analyzer.load_and_prepare_data(run_feature_engineering=False)
        if analyzer.data is None or analyzer.data.is_empty(): 
            print("\n*** Error: FDA data loading failed. ***")
            return False
            
        print(f"FDA data loaded successfully. Shape: {analyzer.data.shape}")
        
        # --- Initialize Hypothesis 2 Analyzer ---
        h2_analyzer = Hypothesis2Analyzer(
            analyzer=analyzer,
            results_dir=FDA_RESULTS_DIR,
            file_prefix=FDA_FILE_PREFIX
        )
        
        # --- Run Volatility Innovations Analysis ---
        success = h2_analyzer.analyze_volatility_innovations()
        
        if success:
            # --- Generate Summary Report ---
            h2_analyzer.generate_summary_report()
            print(f"\n--- FDA Event Analysis for Hypothesis 2 Finished (Results in '{FDA_RESULTS_DIR}') ---")
            return True
        else:
            print("\n*** Error: Volatility innovations analysis failed. ***")
            return False

    except ValueError as ve: 
        print(f"\n*** FDA ValueError: {ve} ***")
        traceback.print_exc()
    except RuntimeError as re: 
        print(f"\n*** FDA RuntimeError: {re} ***")
        traceback.print_exc()
    except FileNotFoundError as fnf: 
        print(f"\n*** FDA FileNotFoundError: {fnf} ***")
    except pl.exceptions.PolarsError as pe: 
        print(f"\n*** FDA PolarsError: {pe} ***")
        traceback.print_exc()
    except Exception as e: 
        print(f"\n*** An unexpected error occurred in FDA analysis: {e} ***")
        traceback.print_exc()
    
    return False

def run_earnings_analysis():
    """
    Runs the earnings event analysis to test Hypothesis 2.
    """
    print("\n=== Starting Earnings Announcement Event Analysis for Hypothesis 2 ===")

    # --- Path Validation & Results Dir Creation ---
    if not os.path.exists(EARNINGS_EVENT_FILE): 
        print(f"\n*** Error: Earnings event file not found: {EARNINGS_EVENT_FILE} ***")
        return False
    
    missing_stock = [f for f in STOCK_FILES if not os.path.exists(f)]
    if missing_stock: 
        print(f"\n*** Error: Stock file(s) not found: {missing_stock} ***")
        return False
    
    print("Earnings file paths validated.")
    try:
        os.makedirs(EARNINGS_RESULTS_DIR, exist_ok=True)
        print(f"Earnings results will be saved to: {os.path.abspath(EARNINGS_RESULTS_DIR)}")
    except OSError as oe:
        print(f"\n*** Error creating earnings results directory '{EARNINGS_RESULTS_DIR}': {oe} ***")
        return False

    try:
        # --- Initialize Components ---
        print("\nInitializing earnings components...")
        data_loader = EventDataLoader(
            event_path=EARNINGS_EVENT_FILE, 
            stock_paths=STOCK_FILES, 
            window_days=WINDOW_DAYS,
            event_date_col=EARNINGS_EVENT_DATE_COL,
            ticker_col=EARNINGS_TICKER_COL
        )
        feature_engineer = EventFeatureEngineer(prediction_window=3)
        analyzer = EventAnalysis(data_loader, feature_engineer)
        print("Earnings components initialized.")

        # --- Load Data ---
        print("\nLoading and preparing earnings data...")
        analyzer.data = analyzer.load_and_prepare_data(run_feature_engineering=False)
        if analyzer.data is None or analyzer.data.is_empty(): 
            print("\n*** Error: Earnings data loading failed. ***")
            return False
            
        print(f"Earnings data loaded successfully. Shape: {analyzer.data.shape}")
        
        # --- Initialize Hypothesis 2 Analyzer ---
        h2_analyzer = Hypothesis2Analyzer(
            analyzer=analyzer,
            results_dir=EARNINGS_RESULTS_DIR,
            file_prefix=EARNINGS_FILE_PREFIX
        )
        
        # --- Run Volatility Innovations Analysis ---
        success = h2_analyzer.analyze_volatility_innovations()
        
        if success:
            # --- Generate Summary Report ---
            h2_analyzer.generate_summary_report()
            print(f"\n--- Earnings Event Analysis for Hypothesis 2 Finished (Results in '{EARNINGS_RESULTS_DIR}') ---")
            return True
        else:
            print("\n*** Error: Volatility innovations analysis failed. ***")
            return False

    except ValueError as ve: 
        print(f"\n*** Earnings ValueError: {ve} ***")
        traceback.print_exc()
    except RuntimeError as re: 
        print(f"\n*** Earnings RuntimeError: {re} ***")
        traceback.print_exc()
    except FileNotFoundError as fnf: 
        print(f"\n*** Earnings FileNotFoundError: {fnf} ***")
    except pl.exceptions.PolarsError as pe: 
        print(f"\n*** Earnings PolarsError: {pe} ***")
        traceback.print_exc()
    except Exception as e: 
        print(f"\n*** An unexpected error occurred in earnings analysis: {e} ***")
        traceback.print_exc()
    
    return False

def compare_results():
    """
    Compares the hypothesis test results between FDA and earnings events.
    """
    print("\n=== Comparing FDA and Earnings Results for Hypothesis 2 ===")
    
    # Create comparison directory
    comparison_dir = "results/hypothesis2/comparison/"
    os.makedirs(comparison_dir, exist_ok=True)
    
    try:
        # Define files to compare
        fda_overall = os.path.join(FDA_RESULTS_DIR, f"{FDA_FILE_PREFIX}_hypothesis2_overall.csv")
        earnings_overall = os.path.join(EARNINGS_RESULTS_DIR, f"{EARNINGS_FILE_PREFIX}_hypothesis2_overall.csv")
        
        fda_subs = os.path.join(FDA_RESULTS_DIR, f"{FDA_FILE_PREFIX}_h2_sub_hypotheses.csv")
        earnings_subs = os.path.join(EARNINGS_RESULTS_DIR, f"{EARNINGS_FILE_PREFIX}_h2_sub_hypotheses.csv")
        
        # Check if files exist
        missing_files = []
        for file_path in [fda_overall, earnings_overall, fda_subs, earnings_subs]:
            if not os.path.exists(file_path):
                missing_files.append(file_path)
        
        if missing_files:
            print(f"Error: The following files are missing: {missing_files}")
            return False
        
        # Load results
        fda_overall_df = pl.read_csv(fda_overall)
        earnings_overall_df = pl.read_csv(earnings_overall)
        
        fda_subs_df = pl.read_csv(fda_subs)
        earnings_subs_df = pl.read_csv(earnings_subs)
        
        # Create comparison table
        overall_comparison = pl.DataFrame({
            'hypothesis': ['Hypothesis 2: Volatility innovations as impact uncertainty proxy'],
            'fda_result': [fda_overall_df['result'].item()],
            'earnings_result': [earnings_overall_df['result'].item()],
            'fda_supported_sub': [fda_overall_df['supported_sub_hypotheses'].item()],
            'fda_total_sub': [fda_overall_df['total_sub_hypotheses'].item()],
            'earnings_supported_sub': [earnings_overall_df['supported_sub_hypotheses'].item()],
            'earnings_total_sub': [earnings_overall_df['total_sub_hypotheses'].item()]
        })
        
        # Save comparison
        overall_comparison.write_csv(os.path.join(comparison_dir, "hypothesis2_overall_comparison.csv"))
        
        # Compare sub-hypotheses
        sub_comparison_rows = []
        
        for i, sub_hypothesis in enumerate(fda_subs_df['sub_hypothesis']):
            fda_result = fda_subs_df['result'][i]
            earnings_result = earnings_subs_df['result'][i]
            
            sub_comparison_rows.append({
                'sub_hypothesis': sub_hypothesis,
                'fda_result': fda_result,
                'earnings_result': earnings_result,
                'both_supported': fda_result and earnings_result
            })
        
        sub_comparison_df = pl.DataFrame(sub_comparison_rows)
        sub_comparison_df.write_csv(os.path.join(comparison_dir, "hypothesis2_sub_comparison.csv"))
        
        # Print comparison
        print("\nHypothesis 2 Comparison Results:")
        print(f"FDA Overall: {fda_overall_df['supported_sub_hypotheses'].item()}/{fda_overall_df['total_sub_hypotheses'].item()} sub-hypotheses supported")
        print(f"Earnings Overall: {earnings_overall_df['supported_sub_hypotheses'].item()}/{earnings_overall_df['total_sub_hypotheses'].item()} sub-hypotheses supported")
        
        # Sub-hypotheses comparison
        for i, row in enumerate(sub_comparison_rows):
            sub_hyp = row['sub_hypothesis']
            fda_res = "SUPPORTED" if row['fda_result'] else "NOT SUPPORTED"
            earn_res = "SUPPORTED" if row['earnings_result'] else "NOT SUPPORTED"
            
            print(f"  {sub_hyp}:")
            print(f"    FDA: {fda_res}")
            print(f"    Earnings: {earn_res}")
        
        # Create comparison plot
        try:
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Convert to pandas for plotting
            sub_comparison_pd = sub_comparison_df.to_pandas()
            
            # Set up grid
            n_subs = len(sub_comparison_pd)
            y_positions = np.arange(n_subs)
            x_positions = np.array([0, 1])
            width = 0.4
            
            # Plot grid
            for i, sub in enumerate(sub_comparison_pd['sub_hypothesis']):
                ax.plot([0, 2], [i, i], 'k--', alpha=0.2)
            
            # Plot FDA results
            for i, result in enumerate(sub_comparison_pd['fda_result']):
                color = 'green' if result else 'red'
                ax.plot(0.5, i, 'o', color=color, markersize=15, alpha=0.7)
            
            # Plot Earnings results
            for i, result in enumerate(sub_comparison_pd['earnings_result']):
                color = 'green' if result else 'red'
                ax.plot(1.5, i, 'o', color=color, markersize=15, alpha=0.7)
            
            # Add labels
            ax.set_yticks(y_positions)
            ax.set_yticklabels([sub.split(': ')[1] for sub in sub_comparison_pd['sub_hypothesis']])
            
            ax.set_xticks([0.5, 1.5])
            ax.set_xticklabels(['FDA Approvals', 'Earnings Announcements'])
            
            ax.set_title('Hypothesis 2 Results Comparison: FDA vs Earnings')
            
            # Add legend
            ax.plot([], [], 'o', color='green', label='Supported')
            ax.plot([], [], 'o', color='red', label='Not Supported')
            ax.legend()
            
            # Add overall results
            overall_text = (f"Overall:\n"
                           f"FDA: {fda_overall_df['supported_sub_hypotheses'].item()}/{fda_overall_df['total_sub_hypotheses'].item()} supported\n"
                           f"Earnings: {earnings_overall_df['supported_sub_hypotheses'].item()}/{earnings_overall_df['total_sub_hypotheses'].item()} supported")
            
            ax.text(1.0, -0.5, overall_text, ha='center', 
                    bbox=dict(boxstyle='round', alpha=0.1))
            
            # Adjust layout
            plt.tight_layout()
            
            # Save plot
            plt.savefig(os.path.join(comparison_dir, "hypothesis2_comparison.png"), dpi=200, bbox_inches='tight')
            plt.close()
            
            print(f"Saved comparison plot to: {os.path.join(comparison_dir, 'hypothesis2_comparison.png')}")
        
        except Exception as e:
            print(f"Error creating comparison plot: {e}")
            traceback.print_exc()
        
        return True
    
    except Exception as e:
        print(f"Error comparing results: {e}")
        traceback.print_exc()
        return False

def main():
    # Run FDA analysis
    fda_success = run_fda_analysis()
    
    # Run earnings analysis
    earnings_success = run_earnings_analysis()
    
    # Compare results if both analyses succeeded
    #if fda_success and earnings_success:
    #    compare_success = compare_results()
    #    if compare_success:
    #        print("\n=== All analyses and comparisons completed successfully ===")
    #    else:
    #        print("\n=== Analyses completed, but comparison failed ===")
    #elif fda_success:
    #    print("\n=== Only FDA analysis completed successfully ===")
    #elif earnings_success:
    #    print("\n=== Only earnings analysis completed successfully ===")
    #else:
    #    print("\n=== Both analyses failed ===")

if __name__ == "__main__":
    main()