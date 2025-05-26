"""
Test Hypothesis 2: GARCH Volatility Innovations as Impact Uncertainty Proxy

Tests three sub-hypotheses:
- H2.1: Pre-event volatility innovations predict subsequent returns
- H2.2: Post-event volatility persistence extends elevated expected returns
- H2.3: Asymmetric volatility response (gamma) correlates with price adjustment

Based on "Modeling Equilibrium Asset Pricing Around Events with Heterogeneous 
Beliefs, Dynamic Volatility, and a Two-Risk Uncertainty Framework" by Brandon Yee
"""

import pandas as pd
import numpy as np
import os
import sys
import polars as pl
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from typing import List, Tuple, Dict, Optional, Any
from datetime import datetime, timedelta

# Add src directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)
src_dir = os.path.join(current_dir, 'src')
if src_dir not in sys.path:
    sys.path.append(src_dir)

try:
    from src.event_processor import EventProcessor, VolatilityParameters
    from src.models import UnifiedVolatilityModel
    from src.config import Config
    print("Successfully imported required modules.")
except ImportError as e:
    print(f"Error importing modules: {e}")
    print(f"Python Path: {sys.path}")
    sys.exit(1)

# Configure plotting style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Configure Polars display
pl.Config.set_tbl_rows(20)
pl.Config.set_tbl_cols(12)
pl.Config.set_fmt_str_lengths(100)

# Data file paths
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

# FDA event parameters
FDA_EVENT_FILE = "/home/d87016661/fda_ticker_list_2000_to_2024.csv"
FDA_RESULTS_DIR = "results/hypothesis2/fda/"
FDA_EVENT_DATE_COL = "Approval Date"
FDA_TICKER_COL = "ticker"

# Earnings event parameters
EARNINGS_EVENT_FILE = "/home/d87016661/detail_history_actuals.csv"
EARNINGS_RESULTS_DIR = "results/hypothesis2/earnings/"
EARNINGS_EVENT_DATE_COL = "ANNDATS"
EARNINGS_TICKER_COL = "ticker"

class Hypothesis2Tester:
    """
    Class to test Hypothesis 2 from the paper:
    GARCH-estimated conditional volatility innovations serve as effective proxy for impact uncertainty.
    """
    
    def __init__(self, config: Config, event_type: str, results_dir: str):
        """
        Initialize tester with configuration and event type.
        
        Args:
            config: Configuration object
            event_type: 'fda' or 'earnings'
            results_dir: Directory to save results
        """
        self.config = config
        self.event_type = event_type
        self.results_dir = results_dir
        self.event_processor = EventProcessor(config)
        
        # Create results directory
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Storage for results
        self.processed_data = None
        self.h2_1_results = None
        self.h2_2_results = None
        self.h2_3_results = None
        
        # Pre-event window for averaging innovations
        self.pre_event_innovation_window = (-10, -1)
    
    def load_stock_data(self, file_paths: List[str]) -> Optional[pl.DataFrame]:
        """Load and combine stock price data from multiple files."""
        print(f"Loading stock data from {len(file_paths)} files...")
        
        dfs = []
        for i, path in enumerate(file_paths):
            if i % 2 == 0:
                print(f"  Loading file {i+1}/{len(file_paths)}...")
            
            try:
                df = pl.read_parquet(path)
                
                # Standardize column names
                rename_map = {}
                for col in df.columns:
                    col_upper = col.upper()
                    if col_upper == 'PERMNO':
                        rename_map[col] = 'ticker'
                    elif col_upper == 'DATE':
                        rename_map[col] = 'date'
                    elif col_upper == 'PRC':
                        rename_map[col] = 'price'
                    elif col_upper == 'RET':
                        rename_map[col] = 'returns'
                
                df = df.rename(rename_map)
                
                # Select and cast required columns
                required_cols = ['ticker', 'date', 'price', 'returns']
                if all(col in df.columns for col in required_cols):
                    df = df.select(required_cols).with_columns([
                        pl.col('ticker').cast(pl.Utf8),
                        pl.col('date').cast(pl.Date),
                        pl.col('price').cast(pl.Float64).abs(),
                        pl.col('returns').cast(pl.Float64)
                    ])
                    dfs.append(df)
                    
            except Exception as e:
                print(f"  Warning: Could not load {path}: {e}")
                continue
        
        if not dfs:
            print("Error: No stock data loaded!")
            return None
        
        # Combine all dataframes
        combined_df = pl.concat(dfs)
        
        # Remove duplicates and invalid data
        combined_df = combined_df.filter(
            pl.col('price').is_not_null() & 
            pl.col('returns').is_not_null() &
            (pl.col('price') > 0)
        ).unique(subset=['ticker', 'date'])
        
        print(f"Loaded {combined_df.height:,} stock observations")
        return combined_df
    
    def load_event_data(self, file_path: str, date_col: str, 
                       ticker_col: str) -> Optional[pl.DataFrame]:
        """Load event data from CSV file."""
        print(f"Loading {self.event_type} event data...")
        
        try:
            # Read CSV with proper date parsing
            df = pl.read_csv(file_path, try_parse_dates=True, infer_schema_length=10000)
            
            # Find columns (case-insensitive)
            actual_date_col = None
            actual_ticker_col = None
            
            for col in df.columns:
                if date_col.lower() in col.lower() and actual_date_col is None:
                    actual_date_col = col
                if ticker_col.lower() in col.lower() and actual_ticker_col is None:
                    actual_ticker_col = col
            
            if not actual_date_col or not actual_ticker_col:
                print(f"Error: Could not find required columns {date_col}, {ticker_col}")
                return None
            
            # Rename and process
            df = df.rename({
                actual_date_col: 'event_date',
                actual_ticker_col: 'symbol'
            })
            
            # Add event type
            df = df.with_columns([
                pl.col('event_date').cast(pl.Date),
                pl.col('symbol').cast(pl.Utf8).str.strip_chars(),
                pl.lit(self.event_type).alias('event_type')
            ])
            
            # Remove invalid events
            df = df.filter(
                pl.col('event_date').is_not_null() &
                pl.col('symbol').is_not_null() &
                (pl.col('symbol').str.len_chars() > 0)
            ).unique(subset=['symbol', 'event_date'])
            
            print(f"Loaded {df.height:,} {self.event_type} events")
            return df
            
        except Exception as e:
            print(f"Error loading event data: {e}")
            return None
    
    def run_analysis(self, stock_df: pl.DataFrame, event_df: pl.DataFrame) -> bool:
        """
        Run the complete Hypothesis 2 analysis.
        
        Returns:
            True if analysis completed successfully
        """
        print(f"\n{'='*60}")
        print(f"Running Hypothesis 2 Analysis for {self.event_type.upper()} Events")
        print(f"{'='*60}")
        
        # Process events using EventProcessor
        print("\nProcessing events with unified volatility model...")
        self.processed_data = self.event_processor.process_events(
            price_data=stock_df.to_pandas(),
            event_data=event_df.to_pandas()
        )
        
        if self.processed_data is None or self.processed_data.empty:
            print("Error: No events successfully processed")
            return False
        
        print(f"Processed {len(self.processed_data):,} event-day observations")
        
        # Add event identifier for grouping
        self.processed_data['event_id'] = (
            self.processed_data['symbol'] + '_' + 
            self.processed_data['event_date'].dt.strftime('%Y%m%d')
        )
        
        # Test sub-hypotheses
        self._test_h2_1()  # Pre-event innovations predict returns
        self._test_h2_2()  # Post-event persistence extends returns
        self._test_h2_3()  # Asymmetric response (gamma)
        
        # Generate visualizations
        self._create_visualizations()
        
        # Save results
        self._save_results()
        
        return True
    
    def _test_h2_1(self):
        """
        Test H2.1: Pre-event volatility innovations predict subsequent returns.
        
        Uses GARCH volatility innovations as proxy for impact uncertainty.
        """
        print("\n" + "-"*50)
        print("Testing H2.1: Pre-event volatility innovations predict returns")
        print("-"*50)
        
        # Calculate average pre-event volatility innovations for each event
        pre_event_data = self.processed_data[
            (self.processed_data['days_to_event'] >= self.pre_event_innovation_window[0]) &
            (self.processed_data['days_to_event'] <= self.pre_event_innovation_window[1])
        ].copy()
        
        # Group by event and calculate mean innovation
        event_innovations = pre_event_data.groupby('event_id').agg({
            'volatility_innovation': 'mean',
            'symbol': 'first'
        }).reset_index()
        
        # Get event-day data for each event
        event_day_data = self.processed_data[
            self.processed_data['days_to_event'] == 0
        ].copy()
        
        # Merge with innovations
        analysis_data = event_day_data.merge(
            event_innovations[['event_id', 'volatility_innovation']],
            on='event_id',
            suffixes=('', '_pre_avg')
        )
        
        # Test for different prediction horizons
        results = []
        
        for horizon in self.config.prediction_horizons:
            print(f"\n  Testing {horizon}-day ahead prediction...")
            
            # Get future returns
            future_returns = []
            
            for event_id in analysis_data['event_id'].unique():
                # Get post-event data for this event
                post_data = self.processed_data[
                    (self.processed_data['event_id'] == event_id) &
                    (self.processed_data['days_to_event'] > 0) &
                    (self.processed_data['days_to_event'] <= horizon)
                ]
                
                if len(post_data) > 0:
                    # Calculate cumulative return over horizon
                    cum_return = (1 + post_data['returns']).prod() - 1
                    future_returns.append({
                        'event_id': event_id,
                        f'return_{horizon}d': cum_return
                    })
            
            if not future_returns:
                continue
                
            future_returns_df = pd.DataFrame(future_returns)
            
            # Merge with analysis data
            regression_data = analysis_data.merge(
                future_returns_df,
                on='event_id',
                how='inner'
            )
            
            if len(regression_data) < 10:
                print(f"    Insufficient data for {horizon}-day horizon")
                continue
            
            # Run regression: future_return = alpha + beta * vol_innovation + epsilon
            X = regression_data['volatility_innovation_pre_avg'].values.reshape(-1, 1)
            y = regression_data[f'return_{horizon}d'].values
            
            # Remove outliers (top/bottom 1%)
            mask = (
                (X.flatten() > np.percentile(X, 1)) & 
                (X.flatten() < np.percentile(X, 99)) &
                (y > np.percentile(y, 1)) & 
                (y < np.percentile(y, 99))
            )
            X_clean = X[mask]
            y_clean = y[mask]
            
            if len(X_clean) < 10:
                continue
            
            # Fit regression
            reg = LinearRegression()
            reg.fit(X_clean, y_clean)
            
            # Calculate statistics
            y_pred = reg.predict(X_clean)
            r2 = r2_score(y_clean, y_pred)
            
            # T-test for significance of slope
            n = len(X_clean)
            residuals = y_clean - y_pred
            se_residuals = np.sqrt(np.sum(residuals**2) / (n - 2))
            x_var = np.sum((X_clean - X_clean.mean())**2)
            se_beta = se_residuals / np.sqrt(x_var) if x_var > 0 else np.inf
            t_stat = reg.coef_[0] / se_beta if se_beta != np.inf else 0
            p_value = 2 * (1 - stats.t.cdf(abs(t_stat), n - 2))
            
            # Store results
            result = {
                'horizon': horizon,
                'n_obs': len(X_clean),
                'beta': reg.coef_[0],
                'alpha': reg.intercept_,
                'r_squared': r2,
                't_statistic': t_stat,
                'p_value': p_value,
                'significant_5pct': p_value < 0.05,
                'mean_innovation': X_clean.mean(),
                'std_innovation': X_clean.std()
            }
            results.append(result)
            
            print(f"    Beta: {result['beta']:.4f}, R²: {result['r_squared']:.3f}, "
                  f"p-value: {result['p_value']:.3f} {'*' if result['significant_5pct'] else ''}")
        
        self.h2_1_results = pd.DataFrame(results)
        
        # Overall support assessment
        if len(self.h2_1_results) > 0:
            supported = any(self.h2_1_results['significant_5pct'])
            print(f"\nH2.1 Overall Support: {'YES' if supported else 'NO'}")
            print(f"Significant predictions: {sum(self.h2_1_results['significant_5pct'])}/{len(self.h2_1_results)}")
    
    def _test_h2_2(self):
        """
        Test H2.2: Post-event volatility persistence extends elevated expected returns.
        
        Tests whether higher post-event volatility persistence leads to higher returns.
        """
        print("\n" + "-"*50)
        print("Testing H2.2: Post-event volatility persistence extends returns")
        print("-"*50)
        
        # Calculate volatility persistence metrics for each event
        persistence_metrics = []
        
        for event_id in self.processed_data['event_id'].unique():
            event_data = self.processed_data[self.processed_data['event_id'] == event_id]
            
            # Pre-event average volatility (baseline)
            pre_vol = event_data[event_data['days_to_event'] < 0]['baseline_volatility'].mean()
            
            # Post-event volatility in persistence window
            post_data = event_data[
                (event_data['days_to_event'] > 0) &
                (event_data['days_to_event'] <= self.config.vol_persistence_window_days)
            ]
            
            if len(post_data) > 0 and pre_vol > 0:
                post_vol = post_data['baseline_volatility'].mean()
                post_returns = post_data['expected_return'].mean()
                
                # Volatility persistence ratio
                persistence_ratio = post_vol / pre_vol
                
                persistence_metrics.append({
                    'event_id': event_id,
                    'pre_event_vol': pre_vol,
                    'post_event_vol': post_vol,
                    'persistence_ratio': persistence_ratio,
                    'avg_post_returns': post_returns,
                    'n_post_days': len(post_data)
                })
        
        if not persistence_metrics:
            print("  Insufficient data for H2.2 test")
            self.h2_2_results = pd.DataFrame()
            return
        
        persistence_df = pd.DataFrame(persistence_metrics)
        
        # Remove extreme outliers
        persistence_df = persistence_df[
            (persistence_df['persistence_ratio'] > persistence_df['persistence_ratio'].quantile(0.01)) &
            (persistence_df['persistence_ratio'] < persistence_df['persistence_ratio'].quantile(0.99))
        ]
        
        if len(persistence_df) < 10:
            print("  Insufficient data after outlier removal")
            self.h2_2_results = pd.DataFrame()
            return
        
        # Test relationship: returns = f(persistence_ratio)
        X = persistence_df['persistence_ratio'].values.reshape(-1, 1)
        y = persistence_df['avg_post_returns'].values
        
        # Fit regression
        reg = LinearRegression()
        reg.fit(X, y)
        
        # Calculate statistics
        y_pred = reg.predict(X)
        r2 = r2_score(y, y_pred)
        
        # T-test for significance
        n = len(X)
        residuals = y - y_pred
        se_residuals = np.sqrt(np.sum(residuals**2) / (n - 2))
        x_var = np.sum((X - X.mean())**2)
        se_beta = se_residuals / np.sqrt(x_var) if x_var > 0 else np.inf
        t_stat = reg.coef_[0] / se_beta if se_beta != np.inf else 0
        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), n - 2))
        
        # Test if positive relationship (one-sided test)
        p_value_one_sided = p_value / 2 if reg.coef_[0] > 0 else 1 - p_value / 2
        
        self.h2_2_results = pd.DataFrame([{
            'n_events': len(persistence_df),
            'beta': reg.coef_[0],
            'alpha': reg.intercept_,
            'r_squared': r2,
            't_statistic': t_stat,
            'p_value': p_value,
            'p_value_one_sided': p_value_one_sided,
            'positive_and_significant': (reg.coef_[0] > 0) and (p_value_one_sided < 0.05),
            'mean_persistence_ratio': X.mean(),
            'std_persistence_ratio': X.std()
        }])
        
        print(f"\n  Regression Results:")
        print(f"    Beta: {reg.coef_[0]:.4f} (higher persistence → {'higher' if reg.coef_[0] > 0 else 'lower'} returns)")
        print(f"    R²: {r2:.3f}")
        print(f"    p-value (one-sided): {p_value_one_sided:.3f} {'*' if p_value_one_sided < 0.05 else ''}")
        print(f"\n  H2.2 Supported: {'YES' if self.h2_2_results['positive_and_significant'].iloc[0] else 'NO'}")
    
    def _test_h2_3(self):
        """
        Test H2.3: Asymmetric volatility response (gamma) correlates with asymmetric price adjustment.
        
        Tests whether GJR-GARCH gamma parameter is significantly positive.
        """
        print("\n" + "-"*50)
        print("Testing H2.3: Asymmetric volatility response")
        print("-"*50)
        
        # Collect gamma parameters from all estimated models
        gamma_values = []
        
        for symbol, params in self.event_processor.volatility_params.items():
            if isinstance(params, VolatilityParameters) and params.gamma is not None:
                gamma_values.append({
                    'symbol': symbol,
                    'gamma': params.gamma,
                    'alpha': params.alpha,
                    'beta': params.beta,
                    'persistence': params.alpha + params.beta + params.gamma/2
                })
        
        if not gamma_values:
            print("  No gamma parameters available")
            self.h2_3_results = pd.DataFrame()
            return
        
        gamma_df = pd.DataFrame(gamma_values)
        
        # Remove extreme outliers
        gamma_df = gamma_df[
            (gamma_df['gamma'] > gamma_df['gamma'].quantile(0.01)) &
            (gamma_df['gamma'] < gamma_df['gamma'].quantile(0.99))
        ]
        
        # Test if average gamma is significantly positive
        gammas = gamma_df['gamma'].values
        
        # One-sample t-test: H0: gamma = 0, H1: gamma > 0
        t_stat, p_value_two_sided = stats.ttest_1samp(gammas, 0)
        p_value_one_sided = p_value_two_sided / 2 if t_stat > 0 else 1 - p_value_two_sided / 2
        
        # Calculate confidence interval
        mean_gamma = gammas.mean()
        se_gamma = gammas.std() / np.sqrt(len(gammas))
        ci_lower = mean_gamma - 1.96 * se_gamma
        ci_upper = mean_gamma + 1.96 * se_gamma
        
        # Calculate proportion of stocks with positive gamma
        prop_positive = (gammas > 0).mean()
        
        # Binomial test: is proportion > 0.5?
        n_positive = sum(gammas > 0)
        n_total = len(gammas)
        binom_p_value = stats.binom_test(n_positive, n_total, p=0.5, alternative='greater')
        
        self.h2_3_results = pd.DataFrame([{
            'n_stocks': len(gammas),
            'mean_gamma': mean_gamma,
            'std_gamma': gammas.std(),
            'median_gamma': np.median(gammas),
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            't_statistic': t_stat,
            'p_value_gamma_positive': p_value_one_sided,
            'gamma_positive_significant': p_value_one_sided < 0.05,
            'prop_positive_gamma': prop_positive,
            'binom_p_value': binom_p_value,
            'majority_positive_significant': binom_p_value < 0.05
        }])
        
        print(f"\n  Asymmetry Parameter (γ) Analysis:")
        print(f"    Mean γ: {mean_gamma:.4f} (95% CI: [{ci_lower:.4f}, {ci_upper:.4f}])")
        print(f"    Proportion positive: {prop_positive:.1%}")
        print(f"    T-test (γ > 0): t={t_stat:.3f}, p={p_value_one_sided:.3f} {'*' if p_value_one_sided < 0.05 else ''}")
        print(f"    Binomial test: p={binom_p_value:.3f} {'*' if binom_p_value < 0.05 else ''}")
        print(f"\n  H2.3 Supported: {'YES' if self.h2_3_results['gamma_positive_significant'].iloc[0] else 'NO'}")
    
    def _create_visualizations(self):
        """Create visualizations for Hypothesis 2 results."""
        print("\nGenerating visualizations...")
        
        # H2.1: Volatility innovations vs future returns
        if len(self.h2_1_results) > 0:
            self._plot_h2_1()
        
        # H2.2: Volatility persistence vs returns
        if len(self.h2_2_results) > 0:
            self._plot_h2_2()
        
        # H2.3: Gamma distribution
        if len(self.h2_3_results) > 0:
            self._plot_h2_3()
        
        # Overall summary plot
        self._plot_summary()
    
    def _plot_h2_1(self):
        """Plot volatility innovations vs future returns for different horizons."""
        # Get the most significant horizon
        best_horizon = self.h2_1_results.loc[self.h2_1_results['r_squared'].idxmax()]
        horizon = int(best_horizon['horizon'])
        
        # Recreate the data for plotting
        pre_event_data = self.processed_data[
            (self.processed_data['days_to_event'] >= self.pre_event_innovation_window[0]) &
            (self.processed_data['days_to_event'] <= self.pre_event_innovation_window[1])
        ]
        
        event_innovations = pre_event_data.groupby('event_id')['volatility_innovation'].mean().reset_index()
        
        # Get future returns for best horizon
        future_returns = []
        for event_id in event_innovations['event_id']:
            post_data = self.processed_data[
                (self.processed_data['event_id'] == event_id) &
                (self.processed_data['days_to_event'] > 0) &
                (self.processed_data['days_to_event'] <= horizon)
            ]
            if len(post_data) > 0:
                cum_return = (1 + post_data['returns']).prod() - 1
                future_returns.append({
                    'event_id': event_id,
                    'future_return': cum_return
                })
        
        plot_data = event_innovations.merge(pd.DataFrame(future_returns), on='event_id')
        
        # Remove outliers for plotting
        plot_data = plot_data[
            (plot_data['volatility_innovation'] > plot_data['volatility_innovation'].quantile(0.01)) &
            (plot_data['volatility_innovation'] < plot_data['volatility_innovation'].quantile(0.99)) &
            (plot_data['future_return'] > plot_data['future_return'].quantile(0.01)) &
            (plot_data['future_return'] < plot_data['future_return'].quantile(0.99))
        ]
        
        # Create scatter plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.scatter(plot_data['volatility_innovation'], plot_data['future_return'],
                  alpha=0.5, s=50, color='blue')
        
        # Add regression line
        X = plot_data['volatility_innovation'].values.reshape(-1, 1)
        y = plot_data['future_return'].values
        reg = LinearRegression().fit(X, y)
        x_line = np.linspace(X.min(), X.max(), 100)
        y_line = reg.predict(x_line.reshape(-1, 1))
        
        ax.plot(x_line, y_line, color='red', linewidth=2,
                label=f'β={best_horizon["beta"]:.3f}, R²={best_horizon["r_squared"]:.3f}')
        
        ax.set_xlabel('Pre-Event Volatility Innovation (Average)')
        ax.set_ylabel(f'{horizon}-Day Future Return')
        ax.set_title(f'{self.event_type.upper()}: H2.1 - Volatility Innovations Predict Returns')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, f'{self.event_type}_h2_1_scatter.png'),
                    dpi=300, bbox_inches='tight')
        plt.close()
        
        # Plot results for all horizons
        fig, ax = plt.subplots(figsize=(10, 6))
        
        horizons = self.h2_1_results['horizon'].values
        betas = self.h2_1_results['beta'].values
        p_values = self.h2_1_results['p_value'].values
        
        # Create bar plot
        bars = ax.bar(horizons, betas, color=['green' if p < 0.05 else 'gray' for p in p_values])
        
        # Add significance stars
        for i, (h, b, p) in enumerate(zip(horizons, betas, p_values)):
            if p < 0.05:
                ax.text(h, b + 0.001 * np.sign(b), '*', ha='center', fontsize=16, color='red')
        
        ax.set_xlabel('Prediction Horizon (days)')
        ax.set_ylabel('Beta Coefficient')
        ax.set_title(f'{self.event_type.upper()}: Impact of Volatility Innovations on Future Returns')
        ax.grid(True, alpha=0.3, axis='y')
        ax.axhline(0, color='black', linestyle='-', linewidth=0.5)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, f'{self.event_type}_h2_1_horizons.png'),
                    dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_h2_2(self):
        """Plot volatility persistence vs returns relationship."""
        # Recreate the analysis data
        persistence_data = []
        
        for event_id in self.processed_data['event_id'].unique():
            event_data = self.processed_data[self.processed_data['event_id'] == event_id]
            
            pre_vol = event_data[event_data['days_to_event'] < 0]['baseline_volatility'].mean()
            post_data = event_data[
                (event_data['days_to_event'] > 0) &
                (event_data['days_to_event'] <= self.config.vol_persistence_window_days)
            ]
            
            if len(post_data) > 0 and pre_vol > 0:
                post_vol = post_data['baseline_volatility'].mean()
                post_returns = post_data['expected_return'].mean()
                persistence_ratio = post_vol / pre_vol
                
                persistence_data.append({
                    'persistence_ratio': persistence_ratio,
                    'avg_returns': post_returns
                })
        
        plot_df = pd.DataFrame(persistence_data)
        
        # Remove outliers
        plot_df = plot_df[
            (plot_df['persistence_ratio'] > plot_df['persistence_ratio'].quantile(0.01)) &
            (plot_df['persistence_ratio'] < plot_df['persistence_ratio'].quantile(0.99))
        ]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Scatter plot
        ax.scatter(plot_df['persistence_ratio'], plot_df['avg_returns'],
                  alpha=0.5, s=50, color='purple')
        
        # Add regression line
        X = plot_df['persistence_ratio'].values.reshape(-1, 1)
        y = plot_df['avg_returns'].values
        reg = LinearRegression().fit(X, y)
        x_line = np.linspace(X.min(), X.max(), 100)
        y_line = reg.predict(x_line.reshape(-1, 1))
        
        result = self.h2_2_results.iloc[0]
        ax.plot(x_line, y_line, color='red', linewidth=2,
                label=f'β={result["beta"]:.3f}, R²={result["r_squared"]:.3f}')
        
        # Add vertical line at ratio = 1
        ax.axvline(1, color='gray', linestyle='--', alpha=0.5, label='No persistence change')
        
        ax.set_xlabel('Volatility Persistence Ratio (Post/Pre)')
        ax.set_ylabel('Average Post-Event Expected Return')
        ax.set_title(f'{self.event_type.upper()}: H2.2 - Volatility Persistence Extends Returns')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, f'{self.event_type}_h2_2_persistence.png'),
                    dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_h2_3(self):
        """Plot distribution of gamma parameters."""
        # Collect gamma values
        gamma_values = []
        for symbol, params in self.event_processor.volatility_params.items():
            if isinstance(params, VolatilityParameters) and params.gamma is not None:
                gamma_values.append(params.gamma)
        
        gamma_values = np.array(gamma_values)
        
        # Remove outliers
        gamma_clean = gamma_values[
            (gamma_values > np.percentile(gamma_values, 1)) &
            (gamma_values < np.percentile(gamma_values, 99))
        ]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Histogram
        ax1.hist(gamma_clean, bins=50, density=True, alpha=0.7, color='orange', edgecolor='black')
        ax1.axvline(0, color='black', linestyle='--', linewidth=2, label='γ = 0')
        ax1.axvline(gamma_clean.mean(), color='red', linestyle='-', linewidth=2,
                   label=f'Mean γ = {gamma_clean.mean():.3f}')
        
        # Add normal distribution overlay
        x = np.linspace(gamma_clean.min(), gamma_clean.max(), 100)
        ax1.plot(x, stats.norm.pdf(x, gamma_clean.mean(), gamma_clean.std()),
                'b-', linewidth=2, label='Normal fit')
        
        ax1.set_xlabel('Gamma (γ) Parameter')
        ax1.set_ylabel('Density')
        ax1.set_title('Distribution of Asymmetry Parameters')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Q-Q plot
        stats.probplot(gamma_clean, dist="norm", plot=ax2)
        ax2.set_title('Q-Q Plot of Gamma Parameters')
        ax2.grid(True, alpha=0.3)
        
        # Add test results as text
        result = self.h2_3_results.iloc[0]
        text = f"Mean γ: {result['mean_gamma']:.4f}\n"
        text += f"t-stat: {result['t_statistic']:.3f}\n"
        text += f"p-value: {result['p_value_gamma_positive']:.3f}\n"
        text += f"Significant: {'Yes' if result['gamma_positive_significant'] else 'No'}"
        
        ax1.text(0.05, 0.95, text, transform=ax1.transAxes, 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                verticalalignment='top', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, f'{self.event_type}_h2_3_gamma.png'),
                    dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_summary(self):
        """Create summary plot of all H2 test results."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Prepare summary data
        tests = ['H2.1\nVolatility Innovations\nPredict Returns',
                 'H2.2\nPersistence Extends\nReturns',
                 'H2.3\nAsymmetric Response\n(γ > 0)']
        
        supported = []
        p_values = []
        
        # H2.1 - use best horizon
        if len(self.h2_1_results) > 0:
            best_h21 = self.h2_1_results.loc[self.h2_1_results['r_squared'].idxmax()]
            supported.append(best_h21['significant_5pct'])
            p_values.append(best_h21['p_value'])
        else:
            supported.append(False)
            p_values.append(1.0)
        
        # H2.2
        if len(self.h2_2_results) > 0:
            supported.append(self.h2_2_results['positive_and_significant'].iloc[0])
            p_values.append(self.h2_2_results['p_value_one_sided'].iloc[0])
        else:
            supported.append(False)
            p_values.append(1.0)
        
        # H2.3
        if len(self.h2_3_results) > 0:
            supported.append(self.h2_3_results['gamma_positive_significant'].iloc[0])
            p_values.append(self.h2_3_results['p_value_gamma_positive'].iloc[0])
        else:
            supported.append(False)
            p_values.append(1.0)
        
        # Create bar plot
        x_pos = np.arange(len(tests))
        colors = ['green' if s else 'red' for s in supported]
        
        bars = ax.bar(x_pos, [1 if s else 0 for s in supported], color=colors, alpha=0.7)
        
        # Add p-values as text
        for i, (bar, p) in enumerate(zip(bars, p_values)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                   f'p={p:.3f}', ha='center', va='bottom', fontsize=12)
        
        ax.set_ylim(0, 1.2)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(tests)
        ax.set_ylabel('Hypothesis Supported')
        ax.set_title(f'{self.event_type.upper()}: Hypothesis 2 Test Results Summary')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add overall assessment
        n_supported = sum(supported)
        overall_text = f"Overall: {n_supported}/3 sub-hypotheses supported"
        ax.text(0.5, 0.5, overall_text, transform=ax.transAxes,
               fontsize=16, ha='center', va='center',
               bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, f'{self.event_type}_h2_summary.png'),
                    dpi=300, bbox_inches='tight')
        plt.close()
    
    def _save_results(self):
        """Save analysis results to files."""
        print("\nSaving results...")
        
        # Save processed data (sample to reduce size)
        sample_size = min(10000, len(self.processed_data))
        self.processed_data.sample(n=sample_size).to_csv(
            os.path.join(self.results_dir, f'{self.event_type}_h2_processed_sample.csv'),
            index=False
        )
        
        # Save test results
        if len(self.h2_1_results) > 0:
            self.h2_1_results.to_csv(
                os.path.join(self.results_dir, f'{self.event_type}_h2_1_results.csv'),
                index=False
            )
        
        if len(self.h2_2_results) > 0:
            self.h2_2_results.to_csv(
                os.path.join(self.results_dir, f'{self.event_type}_h2_2_results.csv'),
                index=False
            )
        
        if len(self.h2_3_results) > 0:
            self.h2_3_results.to_csv(
                os.path.join(self.results_dir, f'{self.event_type}_h2_3_results.csv'),
                index=False
            )
        
        # Create overall summary
        summary = {
            'event_type': self.event_type,
            'n_events': self.processed_data['event_id'].nunique(),
            'n_symbols': self.processed_data['symbol'].nunique(),
            'n_obs': len(self.processed_data),
            'h2_1_supported': any(self.h2_1_results['significant_5pct']) if len(self.h2_1_results) > 0 else False,
            'h2_2_supported': self.h2_2_results['positive_and_significant'].iloc[0] if len(self.h2_2_results) > 0 else False,
            'h2_3_supported': self.h2_3_results['gamma_positive_significant'].iloc[0] if len(self.h2_3_results) > 0 else False
        }
        
        pd.DataFrame([summary]).to_csv(
            os.path.join(self.results_dir, f'{self.event_type}_h2_summary.csv'),
            index=False
        )
        
        print(f"Results saved to {self.results_dir}")


def run_hypothesis2_analysis(event_type: str, event_file: str, date_col: str, 
                           ticker_col: str, stock_files: List[str]) -> bool:
    """
    Run Hypothesis 2 analysis for a specific event type.
    
    Returns:
        True if analysis completed successfully
    """
    # Set up configuration
    config = Config()
    
    # Set up results directory
    if event_type == 'fda':
        results_dir = FDA_RESULTS_DIR
    else:
        results_dir = EARNINGS_RESULTS_DER
    
    # Initialize tester
    tester = Hypothesis2Tester(config, event_type, results_dir)
    
    # Load data
    stock_df = tester.load_stock_data(stock_files)
    if stock_df is None:
        return False
    
    event_df = tester.load_event_data(event_file, date_col, ticker_col)
    if event_df is None:
        return False
    
    # Run analysis
    return tester.run_analysis(stock_df, event_df)


def compare_results():
    """Compare results between FDA and earnings events."""
    print("\n" + "="*60)
    print("Comparing FDA and Earnings Results for Hypothesis 2")
    print("="*60)
    
    # Load summaries
    fda_summary = pd.read_csv(os.path.join(FDA_RESULTS_DIR, 'fda_h2_summary.csv'))
    earnings_summary = pd.read_csv(os.path.join(EARNINGS_RESULTS_DIR, 'earnings_h2_summary.csv'))
    
    # Create comparison plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    tests = ['H2.1\nInnovations→Returns', 'H2.2\nPersistence→Returns', 'H2.3\nAsymmetry (γ>0)']
    x_pos = np.arange(len(tests))
    width = 0.35
    
    fda_support = [
        fda_summary['h2_1_supported'].iloc[0],
        fda_summary['h2_2_supported'].iloc[0],
        fda_summary['h2_3_supported'].iloc[0]
    ]
    
    earnings_support = [
        earnings_summary['h2_1_supported'].iloc[0],
        earnings_summary['h2_2_supported'].iloc[0],
        earnings_summary['h2_3_supported'].iloc[0]
    ]
    
    # Create bars
    ax.bar(x_pos - width/2, [1 if s else 0 for s in fda_support], 
           width, label='FDA', color='lightblue', alpha=0.8)
    ax.bar(x_pos + width/2, [1 if s else 0 for s in earnings_support], 
           width, label='Earnings', color='lightgreen', alpha=0.8)
    
    ax.set_ylabel('Hypothesis Supported')
    ax.set_title('Hypothesis 2 Support: FDA vs Earnings Events')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(tests)
    ax.legend()
    ax.set_ylim(0, 1.2)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    # Save comparison
    comparison_dir = 'results/hypothesis2/comparison/'
    os.makedirs(comparison_dir, exist_ok=True)
    plt.savefig(os.path.join(comparison_dir, 'h2_comparison.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # Print summary
    print("\nHypothesis 2 Comparison Summary:")
    print(f"  H2.1 (Innovations predict returns):")
    print(f"    FDA: {'Supported' if fda_support[0] else 'Not Supported'}")
    print(f"    Earnings: {'Supported' if earnings_support[0] else 'Not Supported'}")
    print(f"  H2.2 (Persistence extends returns):")
    print(f"    FDA: {'Supported' if fda_support[1] else 'Not Supported'}")
    print(f"    Earnings: {'Supported' if earnings_support[1] else 'Not Supported'}")
    print(f"  H2.3 (Asymmetric response):")
    print(f"    FDA: {'Supported' if fda_support[2] else 'Not Supported'}")
    print(f"    Earnings: {'Supported' if earnings_support[2] else 'Not Supported'}")


def main():
    """Main function to run all analyses."""
    print("Starting Hypothesis 2 Analysis")
    print("="*60)
    
    # Run FDA analysis
    fda_success = run_hypothesis2_analysis(
        event_type='fda',
        event_file=FDA_EVENT_FILE,
        date_col=FDA_EVENT_DATE_COL,
        ticker_col=FDA_TICKER_COL,
        stock_files=STOCK_FILES
    )
    
    # Run Earnings analysis
    earnings_success = run_hypothesis2_analysis(
        event_type='earnings',
        event_file=EARNINGS_EVENT_FILE,
        date_col=EARNINGS_EVENT_DATE_COL,
        ticker_col=EARNINGS_TICKER_COL,
        stock_files=STOCK_FILES
    )
    
    # Compare results if both succeeded
    if fda_success and earnings_success:
        compare_results()
    
    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()
