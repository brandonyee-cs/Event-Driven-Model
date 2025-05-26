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
    from src.models import UnifiedVolatilityModel # Not directly used here but good for context
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
        self.h2_1_results = pd.DataFrame() # Initialize as empty DataFrame
        self.h2_2_results = pd.DataFrame() # Initialize as empty DataFrame
        self.h2_3_results = pd.DataFrame() # Initialize as empty DataFrame
        
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
                df_raw = pl.read_parquet(path) # Read raw dataframe
                
                # Identify source column names
                source_ticker_col = None
                source_date_col = None
                source_price_col = None
                source_returns_col = None

                # Create a mapping of standardized uppercase names to actual column names found
                col_map_upper = {col.upper(): col for col in df_raw.columns}

                # For ticker: Prioritize PERMNO, then TICKER, then SYMBOL
                if 'PERMNO' in col_map_upper:
                    source_ticker_col = col_map_upper['PERMNO']
                elif 'TICKER' in col_map_upper:
                    source_ticker_col = col_map_upper['TICKER']
                elif 'SYMBOL' in col_map_upper:
                    source_ticker_col = col_map_upper['SYMBOL']
                
                # For date:
                if 'DATE' in col_map_upper:
                    source_date_col = col_map_upper['DATE']
                
                # For price: Prioritize PRC, then PRICE
                if 'PRC' in col_map_upper:
                    source_price_col = col_map_upper['PRC']
                elif 'PRICE' in col_map_upper:
                    source_price_col = col_map_upper['PRICE']

                # For returns: Prioritize RET, then RETURNS, then RETURN
                if 'RET' in col_map_upper:
                    source_returns_col = col_map_upper['RET']
                elif 'RETURNS' in col_map_upper:
                    source_returns_col = col_map_upper['RETURNS']
                elif 'RETURN' in col_map_upper: # Some files might use singular
                    source_returns_col = col_map_upper['RETURN']

                # Check if all required source columns were found
                missing_cols = []
                if not source_ticker_col: missing_cols.append("ticker (PERMNO/TICKER/SYMBOL)")
                if not source_date_col: missing_cols.append("date (DATE)")
                if not source_price_col: missing_cols.append("price (PRC/PRICE)")
                if not source_returns_col: missing_cols.append("returns (RET/RETURNS/RETURN)")

                if missing_cols:
                    print(f"  Warning: Could not find all required source columns in {path}. Missing: {', '.join(missing_cols)}.")
                    print(f"    Available columns (original): {df_raw.columns}")
                    continue

                # Select and rename in one go using expressions
                # This ensures no intermediate duplicate names are created.
                df = df_raw.select([
                    pl.col(source_ticker_col).alias('ticker'),
                    pl.col(source_date_col).alias('date'),
                    pl.col(source_price_col).alias('price'),
                    pl.col(source_returns_col).alias('returns')
                ])
                
                # Now cast
                df = df.with_columns([
                    pl.col('ticker').cast(pl.Utf8),
                    pl.col('date').cast(pl.Date),
                    pl.col('price').cast(pl.Float64).abs(),
                    pl.col('returns').cast(pl.Float64)
                ])
                dfs.append(df)
                    
            except Exception as e:
                # More specific error for loading vs processing
                print(f"  Warning: Error processing file {path}: {e}")
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
            price_data=stock_df.to_pandas(), # EventProcessor expects pandas
            event_data=event_df.to_pandas()  # EventProcessor expects pandas
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
            'symbol': 'first' # Keep symbol for potential later use
        }).reset_index()
        
        # Get event-day data for each event (or first day of event window)
        # We need a single point per event to attach the pre-event average innovation to.
        # And then link it to future returns.
        
        # Consider using data at t=0 or t=1 for linking pre-event features to post-event returns
        # Let's pick t=1 as the start of the post-event period for returns.
        # The 'volatility_innovation_pre_avg' is calculated *before* t=0.
        
        # Merge pre-event innovations with data points from which future returns will be calculated
        # We need one row per event_id that has the 'volatility_innovation_pre_avg'
        # For calculating future returns, we will iterate unique event_ids from this merged set.

        analysis_base = self.processed_data[
            self.processed_data['days_to_event'] == 1 # Day after event
        ][['event_id']].drop_duplicates()

        analysis_data = analysis_base.merge(
            event_innovations[['event_id', 'volatility_innovation']],
            on='event_id',
            how='inner' # Only events where we have pre-event innovations and a t=1 point
        ).rename(columns={'volatility_innovation': 'volatility_innovation_pre_avg'})


        if analysis_data.empty:
            print("  No events with pre-event innovations and t=1 data. Skipping H2.1.")
            self.h2_1_results = pd.DataFrame()
            return

        results = []
        
        for horizon in self.config.prediction_horizons:
            print(f"\n  Testing {horizon}-day ahead prediction...")
            
            future_returns_list = []
            
            for event_id_val in analysis_data['event_id'].unique():
                # Get post-event data for this event, starting from t=1 up to t=horizon
                post_data = self.processed_data[
                    (self.processed_data['event_id'] == event_id_val) &
                    (self.processed_data['days_to_event'] > 0) & # t=1, 2, ..., horizon
                    (self.processed_data['days_to_event'] <= horizon)
                ]
                
                if not post_data.empty and 'returns' in post_data.columns:
                    # Calculate cumulative return over horizon
                    # Ensure 'returns' is numeric and handle NaNs
                    valid_returns = post_data['returns'].dropna()
                    if not valid_returns.empty:
                        cum_return = (1 + valid_returns).prod() - 1
                        future_returns_list.append({
                            'event_id': event_id_val,
                            f'return_{horizon}d': cum_return
                        })
            
            if not future_returns_list:
                print(f"    No future returns data for {horizon}-day horizon.")
                continue
                
            future_returns_df = pd.DataFrame(future_returns_list)
            
            # Merge with analysis data
            regression_data = analysis_data.merge(
                future_returns_df,
                on='event_id',
                how='inner'
            )
            
            if len(regression_data) < 10 or \
               'volatility_innovation_pre_avg' not in regression_data.columns or \
               f'return_{horizon}d' not in regression_data.columns:
                print(f"    Insufficient data for {horizon}-day horizon after merge (N={len(regression_data)}).")
                continue
            
            # Drop NaNs that might result from merge or original data
            regression_data = regression_data[['volatility_innovation_pre_avg', f'return_{horizon}d']].dropna()
            if len(regression_data) < 10:
                print(f"    Insufficient data for {horizon}-day horizon after NaN drop (N={len(regression_data)}).")
                continue

            # Run regression: future_return = alpha + beta * vol_innovation + epsilon
            X = regression_data['volatility_innovation_pre_avg'].values.reshape(-1, 1)
            y = regression_data[f'return_{horizon}d'].values
            
            # Remove outliers (top/bottom 1%) - apply carefully
            x_q01, x_q99 = np.percentile(X, [1, 99])
            y_q01, y_q99 = np.percentile(y, [1, 99])

            mask = (X.flatten() > x_q01) & (X.flatten() < x_q99) & \
                   (y > y_q01) & (y < y_q99)
            
            X_clean = X[mask]
            y_clean = y[mask]
            
            if len(X_clean) < 10:
                print(f"    Insufficient data for {horizon}-day horizon after outlier removal (N={len(X_clean)}).")
                continue
            
            # Fit regression
            reg = LinearRegression()
            reg.fit(X_clean, y_clean)
            
            # Calculate statistics
            y_pred = reg.predict(X_clean)
            r2 = r2_score(y_clean, y_pred)
            
            # T-test for significance of slope
            n_reg = len(X_clean)
            if n_reg <= 2: # Degrees of freedom check
                print(f"    Skipping t-test for {horizon}-day horizon due to too few observations after cleaning (N={n_reg}).")
                continue

            residuals = y_clean - y_pred
            # Ensure residuals is 1D array for sum of squares
            residuals = residuals.flatten() if residuals.ndim > 1 else residuals
            
            se_residuals_sq = np.sum(residuals**2) / (n_reg - 2) # Variance of residuals
            if se_residuals_sq < 0: se_residuals_sq = 0 # Avoid sqrt domain error
            se_residuals = np.sqrt(se_residuals_sq)

            x_var = np.sum((X_clean.flatten() - X_clean.flatten().mean())**2)
            se_beta = se_residuals / np.sqrt(x_var) if x_var > 1e-9 else np.inf # Avoid division by zero
            
            t_stat_val = reg.coef_[0] / se_beta if se_beta != np.inf and se_beta > 1e-9 else 0
            p_value_val = 2 * (1 - stats.t.cdf(abs(t_stat_val), n_reg - 2)) if n_reg > 2 else 1.0
            
            # Store results
            result = {
                'horizon': horizon,
                'n_obs': n_reg,
                'beta': reg.coef_[0],
                'alpha': reg.intercept_,
                'r_squared': r2,
                't_statistic': t_stat_val,
                'p_value': p_value_val,
                'significant_5pct': p_value_val < 0.05,
                'mean_innovation': X_clean.mean(),
                'std_innovation': X_clean.std()
            }
            results.append(result)
            
            print(f"    Beta: {result['beta']:.4f}, R²: {result['r_squared']:.3f}, "
                  f"p-value: {result['p_value']:.3f} {'*' if result['significant_5pct'] else ''}")
        
        self.h2_1_results = pd.DataFrame(results)
        
        # Overall support assessment
        if not self.h2_1_results.empty:
            supported = any(self.h2_1_results['significant_5pct'])
            print(f"\nH2.1 Overall Support: {'YES' if supported else 'NO'}")
            print(f"Significant predictions: {sum(self.h2_1_results['significant_5pct'])}/{len(self.h2_1_results)}")
        else:
            print("\nH2.1 Overall Support: NO (no results generated)")

    def _test_h2_2(self):
        """
        Test H2.2: Post-event volatility persistence extends elevated expected returns.
        
        Tests whether higher post-event volatility persistence leads to higher returns.
        """
        print("\n" + "-"*50)
        print("Testing H2.2: Post-event volatility persistence extends returns")
        print("-"*50)
        
        persistence_metrics = []
        
        for event_id_val in self.processed_data['event_id'].unique():
            event_data = self.processed_data[self.processed_data['event_id'] == event_id_val]
            
            # Pre-event average volatility (baseline)
            pre_vol_series = event_data[event_data['days_to_event'] < 0]['baseline_volatility'].dropna()
            if pre_vol_series.empty: continue
            pre_vol = pre_vol_series.mean()
            
            # Post-event volatility in persistence window
            post_data = event_data[
                (event_data['days_to_event'] > 0) &
                (event_data['days_to_event'] <= self.config.vol_persistence_window_days)
            ]
            
            post_vol_series = post_data['baseline_volatility'].dropna()
            post_returns_series = post_data['expected_return'].dropna()

            if not post_vol_series.empty and not post_returns_series.empty and pre_vol > 1e-9: # Avoid division by zero
                post_vol = post_vol_series.mean()
                post_returns_avg = post_returns_series.mean()
                
                # Volatility persistence ratio
                persistence_ratio = post_vol / pre_vol
                
                persistence_metrics.append({
                    'event_id': event_id_val,
                    'pre_event_vol': pre_vol,
                    'post_event_vol': post_vol,
                    'persistence_ratio': persistence_ratio,
                    'avg_post_returns': post_returns_avg,
                    'n_post_days': len(post_data)
                })
        
        if not persistence_metrics:
            print("  Insufficient data for H2.2 test (no persistence metrics calculated).")
            self.h2_2_results = pd.DataFrame()
            return
        
        persistence_df = pd.DataFrame(persistence_metrics)
        
        # Remove extreme outliers and NaNs/Infs that might have slipped through
        persistence_df = persistence_df.replace([np.inf, -np.inf], np.nan).dropna()
        if persistence_df.empty:
            print("  Insufficient data for H2.2 test after NaN/Inf removal.")
            self.h2_2_results = pd.DataFrame()
            return

        q_low_ratio, q_high_ratio = persistence_df['persistence_ratio'].quantile([0.01, 0.99])
        q_low_ret, q_high_ret = persistence_df['avg_post_returns'].quantile([0.01, 0.99])

        persistence_df = persistence_df[
            (persistence_df['persistence_ratio'] >= q_low_ratio) &
            (persistence_df['persistence_ratio'] <= q_high_ratio) &
            (persistence_df['avg_post_returns'] >= q_low_ret) &
            (persistence_df['avg_post_returns'] <= q_high_ret)
        ]
        
        if len(persistence_df) < 10:
            print(f"  Insufficient data after outlier removal (N={len(persistence_df)}).")
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
        n_reg = len(X)
        if n_reg <= 2:
            print(f"    Skipping t-test for H2.2 due to too few observations (N={n_reg}).")
            self.h2_2_results = pd.DataFrame()
            return
            
        residuals = y - y_pred
        residuals = residuals.flatten() if residuals.ndim > 1 else residuals

        se_residuals_sq = np.sum(residuals**2) / (n_reg - 2)
        if se_residuals_sq < 0: se_residuals_sq = 0
        se_residuals = np.sqrt(se_residuals_sq)

        x_var = np.sum((X.flatten() - X.flatten().mean())**2)
        se_beta = se_residuals / np.sqrt(x_var) if x_var > 1e-9 else np.inf
        
        t_stat_val = reg.coef_[0] / se_beta if se_beta != np.inf and se_beta > 1e-9 else 0
        p_value_two_sided = 2 * (1 - stats.t.cdf(abs(t_stat_val), n_reg - 2)) if n_reg > 2 else 1.0
        
        # Test if positive relationship (one-sided test)
        p_value_one_sided = p_value_two_sided / 2 if reg.coef_[0] > 0 else 1 - p_value_two_sided / 2
        
        self.h2_2_results = pd.DataFrame([{
            'n_events': n_reg,
            'beta': reg.coef_[0],
            'alpha': reg.intercept_,
            'r_squared': r2,
            't_statistic': t_stat_val,
            'p_value_two_sided': p_value_two_sided,
            'p_value_one_sided': p_value_one_sided,
            'positive_and_significant': (reg.coef_[0] > 0) and (p_value_one_sided < 0.05),
            'mean_persistence_ratio': X.mean(),
            'std_persistence_ratio': X.std()
        }])
        
        print(f"\n  Regression Results:")
        print(f"    Beta: {reg.coef_[0]:.4f} (higher persistence → {'higher' if reg.coef_[0] > 0 else 'lower'} returns)")
        print(f"    R²: {r2:.3f}")
        print(f"    p-value (one-sided): {p_value_one_sided:.3f} {'*' if self.h2_2_results['positive_and_significant'].iloc[0] else ''}")
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
        gamma_values_list = []
        
        # Assuming event_processor.volatility_params is populated correctly by EventProcessor
        # with {'symbol': VolatilityParameters_instance}
        for symbol, params_instance in self.event_processor.volatility_params.items():
            if isinstance(params_instance, VolatilityParameters) and hasattr(params_instance, 'gamma') and params_instance.gamma is not None:
                gamma_values_list.append({
                    'symbol': symbol,
                    'gamma': params_instance.gamma,
                    'alpha': params_instance.alpha if hasattr(params_instance, 'alpha') else np.nan,
                    'beta': params_instance.beta if hasattr(params_instance, 'beta') else np.nan
                })
        
        if not gamma_values_list:
            print("  No gamma parameters available from EventProcessor.")
            self.h2_3_results = pd.DataFrame()
            return
        
        gamma_df = pd.DataFrame(gamma_values_list).dropna(subset=['gamma'])
        if gamma_df.empty:
            print("  No valid gamma parameters after filtering.")
            self.h2_3_results = pd.DataFrame()
            return
            
        # Remove extreme outliers
        q_low_gamma, q_high_gamma = gamma_df['gamma'].quantile([0.01, 0.99])
        gamma_df = gamma_df[
            (gamma_df['gamma'] >= q_low_gamma) &
            (gamma_df['gamma'] <= q_high_gamma)
        ]
        
        if gamma_df.empty:
            print("  No gamma parameters after outlier removal.")
            self.h2_3_results = pd.DataFrame()
            return

        gammas = gamma_df['gamma'].values
        if len(gammas) < 2: # Need at least 2 samples for t-test
            print(f"  Insufficient gamma values for t-test (N={len(gammas)}).")
            self.h2_3_results = pd.DataFrame()
            return

        # One-sample t-test: H0: gamma = 0, H1: gamma > 0
        t_stat_val, p_value_two_sided = stats.ttest_1samp(gammas, 0, nan_policy='omit')
        p_value_one_sided = p_value_two_sided / 2 if t_stat_val > 0 else 1 - p_value_two_sided / 2
        
        # Calculate confidence interval
        mean_gamma = np.nanmean(gammas)
        se_gamma = np.nanstd(gammas) / np.sqrt(np.sum(~np.isnan(gammas))) if np.sum(~np.isnan(gammas)) > 0 else np.nan
        
        ci_lower, ci_upper = np.nan, np.nan
        if not np.isnan(se_gamma) and np.sum(~np.isnan(gammas)) > 1:
             ci_lower = mean_gamma - stats.t.ppf(0.975, np.sum(~np.isnan(gammas))-1) * se_gamma
             ci_upper = mean_gamma + stats.t.ppf(0.975, np.sum(~np.isnan(gammas))-1) * se_gamma

        # Calculate proportion of stocks with positive gamma
        prop_positive = np.nanmean(gammas > 0) if len(gammas) > 0 else np.nan
        
        # Binomial test: is proportion > 0.5?
        n_positive = np.sum(gammas > 0)
        n_total = len(gammas)
        binom_p_value = np.nan
        if n_total > 0:
            binom_p_value = stats.binomtest(n_positive, n_total, p=0.5, alternative='greater').pvalue

        
        self.h2_3_results = pd.DataFrame([{
            'n_stocks': n_total,
            'mean_gamma': mean_gamma,
            'std_gamma': np.nanstd(gammas),
            'median_gamma': np.nanmedian(gammas),
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            't_statistic': t_stat_val,
            'p_value_gamma_positive': p_value_one_sided,
            'gamma_positive_significant': p_value_one_sided < 0.05,
            'prop_positive_gamma': prop_positive,
            'binom_p_value': binom_p_value,
            'majority_positive_significant': binom_p_value < 0.05 if not np.isnan(binom_p_value) else False
        }])
        
        print(f"\n  Asymmetry Parameter (γ) Analysis:")
        print(f"    Mean γ: {mean_gamma:.4f} (95% CI: [{ci_lower:.4f}, {ci_upper:.4f}])")
        print(f"    Proportion positive: {prop_positive:.1%}" if not np.isnan(prop_positive) else "N/A")
        print(f"    T-test (γ > 0): t={t_stat_val:.3f}, p={p_value_one_sided:.3f} {'*' if p_value_one_sided < 0.05 else ''}")
        print(f"    Binomial test: p={binom_p_value:.3f} {'*' if not np.isnan(binom_p_value) and binom_p_value < 0.05 else ''}")
        
        h2_3_supported = self.h2_3_results['gamma_positive_significant'].iloc[0] if not self.h2_3_results.empty else False
        print(f"\n  H2.3 Supported: {'YES' if h2_3_supported else 'NO'}")

    def _create_visualizations(self):
        """Create visualizations for Hypothesis 2 results."""
        print("\nGenerating visualizations...")
        
        if self.processed_data is None or self.processed_data.empty:
            print("Warning: No processed data available for H2 visualizations.")
            return

        # H2.1: Volatility innovations vs future returns
        if not self.h2_1_results.empty:
            self._plot_h2_1()
        else:
            print("Skipping H2.1 visualization: no results.")
        
        # H2.2: Volatility persistence vs returns
        if not self.h2_2_results.empty:
            self._plot_h2_2()
        else:
            print("Skipping H2.2 visualization: no results.")

        # H2.3: Gamma distribution
        if not self.h2_3_results.empty:
            self._plot_h2_3()
        else:
            print("Skipping H2.3 visualization: no results.")
        
        # Overall summary plot
        self._plot_summary()
    
    def _plot_h2_1(self):
        """Plot volatility innovations vs future returns for different horizons."""
        if 'r_squared' not in self.h2_1_results.columns or self.h2_1_results['r_squared'].isnull().all():
            print("Cannot plot H2.1 scatter: R-squared data missing or all NaN.")
            # Plot horizons if possible
            if 'horizon' in self.h2_1_results.columns and 'beta' in self.h2_1_results.columns:
                 self._plot_h2_1_horizons_only()
            return

        # Get the most significant horizon or best R-squared
        if self.h2_1_results['r_squared'].notna().any():
            best_horizon_row = self.h2_1_results.loc[self.h2_1_results['r_squared'].idxmax()]
        else: # Fallback if R-squared is all NaN
            best_horizon_row = self.h2_1_results.iloc[0]

        horizon = int(best_horizon_row['horizon'])
        
        # Recreate the data for plotting - consistent with _test_h2_1
        pre_event_data = self.processed_data[
            (self.processed_data['days_to_event'] >= self.pre_event_innovation_window[0]) &
            (self.processed_data['days_to_event'] <= self.pre_event_innovation_window[1])
        ]
        event_innovations = pre_event_data.groupby('event_id')['volatility_innovation'].mean().reset_index()
        event_innovations = event_innovations.rename(columns={'volatility_innovation': 'volatility_innovation_pre_avg'})

        future_returns_list = []
        for event_id_val in event_innovations['event_id']:
            post_data = self.processed_data[
                (self.processed_data['event_id'] == event_id_val) &
                (self.processed_data['days_to_event'] > 0) &
                (self.processed_data['days_to_event'] <= horizon)
            ]
            if not post_data.empty and 'returns' in post_data.columns:
                valid_returns = post_data['returns'].dropna()
                if not valid_returns.empty:
                    cum_return = (1 + valid_returns).prod() - 1
                    future_returns_list.append({
                        'event_id': event_id_val,
                        f'future_return_{horizon}d': cum_return
                    })
        
        if not future_returns_list:
            print(f"No future returns for H2.1 scatter plot (horizon {horizon}). Plotting horizons only.")
            self._plot_h2_1_horizons_only()
            return

        future_returns_df = pd.DataFrame(future_returns_list)
        plot_data = event_innovations.merge(future_returns_df, on='event_id', how='inner')
        plot_data = plot_data[['volatility_innovation_pre_avg', f'future_return_{horizon}d']].dropna()

        if plot_data.empty or len(plot_data) < 2:
            print(f"Not enough data points for H2.1 scatter plot (horizon {horizon}) after merge/NaN drop. Plotting horizons only.")
            self._plot_h2_1_horizons_only()
            return

        # Remove outliers for plotting
        x_q01, x_q99 = plot_data['volatility_innovation_pre_avg'].quantile([0.01, 0.99])
        y_q01, y_q99 = plot_data[f'future_return_{horizon}d'].quantile([0.01, 0.99])

        plot_data_clean = plot_data[
            (plot_data['volatility_innovation_pre_avg'] >= x_q01) &
            (plot_data['volatility_innovation_pre_avg'] <= x_q99) &
            (plot_data[f'future_return_{horizon}d'] >= y_q01) &
            (plot_data[f'future_return_{horizon}d'] <= y_q99)
        ]
        
        if plot_data_clean.empty or len(plot_data_clean) < 2:
            print(f"Not enough data points for H2.1 scatter plot (horizon {horizon}) after outlier removal. Plotting horizons only.")
            self._plot_h2_1_horizons_only()
            return

        # Create scatter plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.scatter(plot_data_clean['volatility_innovation_pre_avg'], plot_data_clean[f'future_return_{horizon}d'],
                  alpha=0.5, s=50, color='blue')
        
        # Add regression line
        X_plot = plot_data_clean['volatility_innovation_pre_avg'].values.reshape(-1, 1)
        y_plot = plot_data_clean[f'future_return_{horizon}d'].values
        reg = LinearRegression().fit(X_plot, y_plot)
        x_line = np.linspace(X_plot.min(), X_plot.max(), 100)
        y_line = reg.predict(x_line.reshape(-1, 1))
        
        r2_plot = r2_score(y_plot, reg.predict(X_plot)) # Recalculate R2 for plotted data

        ax.plot(x_line, y_line, color='red', linewidth=2,
                label=f'β={reg.coef_[0]:.3f}, R²={r2_plot:.3f}')
        
        ax.set_xlabel('Pre-Event Volatility Innovation (Average)')
        ax.set_ylabel(f'{horizon}-Day Future Return')
        ax.set_title(f'{self.event_type.upper()}: H2.1 - Vol Innovations vs Returns (Horizon {horizon}d)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, f'{self.event_type}_h2_1_scatter.png'),
                    dpi=300, bbox_inches='tight')
        plt.close()
        
        self._plot_h2_1_horizons_only() # Always plot the horizons bar chart

    def _plot_h2_1_horizons_only(self):
        """Plot beta coefficients for all horizons for H2.1."""
        if self.h2_1_results.empty or 'horizon' not in self.h2_1_results.columns:
            print("No H2.1 horizon data to plot.")
            return

        fig, ax = plt.subplots(figsize=(10, 6))
        
        horizons = self.h2_1_results['horizon'].values
        betas = self.h2_1_results['beta'].values
        p_values = self.h2_1_results['p_value'].values
        
        # Create bar plot
        bars = ax.bar(horizons.astype(str), betas, color=['green' if p < 0.05 else 'gray' for p in p_values])
        
        # Add significance stars
        for i, (h, b, p) in enumerate(zip(horizons, betas, p_values)):
            if not (np.isnan(b) or np.isnan(p)): # Check for NaN before plotting text
                if p < 0.05:
                    y_pos = b + 0.001 * np.sign(b) if b!=0 else 0.001
                    ax.text(str(h), y_pos, '*', ha='center', va='bottom' if b >= 0 else 'top', fontsize=16, color='red')
        
        ax.set_xlabel('Prediction Horizon (days)')
        ax.set_ylabel('Beta Coefficient')
        ax.set_title(f'{self.event_type.upper()}: H2.1 - Impact of Vol Innovations on Future Returns')
        ax.grid(True, alpha=0.3, axis='y')
        ax.axhline(0, color='black', linestyle='-', linewidth=0.5)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, f'{self.event_type}_h2_1_horizons.png'),
                    dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_h2_2(self):
        """Plot volatility persistence vs returns relationship."""
        # Recreate the analysis data - consistent with _test_h2_2
        persistence_data_list = []
        for event_id_val in self.processed_data['event_id'].unique():
            event_data = self.processed_data[self.processed_data['event_id'] == event_id_val]
            pre_vol_series = event_data[event_data['days_to_event'] < 0]['baseline_volatility'].dropna()
            if pre_vol_series.empty: continue
            pre_vol = pre_vol_series.mean()
            
            post_data = event_data[
                (event_data['days_to_event'] > 0) &
                (event_data['days_to_event'] <= self.config.vol_persistence_window_days)
            ]
            post_vol_series = post_data['baseline_volatility'].dropna()
            post_returns_series = post_data['expected_return'].dropna()

            if not post_vol_series.empty and not post_returns_series.empty and pre_vol > 1e-9:
                post_vol = post_vol_series.mean()
                post_returns_avg = post_returns_series.mean()
                persistence_ratio = post_vol / pre_vol
                persistence_data_list.append({
                    'persistence_ratio': persistence_ratio,
                    'avg_post_returns': post_returns_avg # Use avg_post_returns to match test
                })
        
        if not persistence_data_list:
            print("No data for H2.2 persistence plot.")
            return

        plot_df = pd.DataFrame(persistence_data_list).replace([np.inf, -np.inf], np.nan).dropna()
        if plot_df.empty or len(plot_df) < 2:
            print("Not enough data for H2.2 persistence plot after NaN/Inf removal.")
            return

        # Remove outliers
        q_low_ratio, q_high_ratio = plot_df['persistence_ratio'].quantile([0.01, 0.99])
        q_low_ret, q_high_ret = plot_df['avg_post_returns'].quantile([0.01, 0.99])

        plot_df_clean = plot_df[
            (plot_df['persistence_ratio'] >= q_low_ratio) &
            (plot_df['persistence_ratio'] <= q_high_ratio) &
            (plot_df['avg_post_returns'] >= q_low_ret) &
            (plot_df['avg_post_returns'] <= q_high_ret)
        ]
        
        if plot_df_clean.empty or len(plot_df_clean) < 2:
            print("Not enough data points for H2.2 persistence plot after outlier removal.")
            return

        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Scatter plot
        ax.scatter(plot_df_clean['persistence_ratio'], plot_df_clean['avg_post_returns'],
                  alpha=0.5, s=50, color='purple')
        
        # Add regression line
        X_plot = plot_df_clean['persistence_ratio'].values.reshape(-1, 1)
        y_plot = plot_df_clean['avg_post_returns'].values
        reg = LinearRegression().fit(X_plot, y_plot)
        x_line = np.linspace(X_plot.min(), X_plot.max(), 100)
        y_line = reg.predict(x_line.reshape(-1, 1))
        
        r2_plot = r2_score(y_plot, reg.predict(X_plot)) # Recalculate R2 for plotted data
        beta_plot = reg.coef_[0]

        ax.plot(x_line, y_line, color='red', linewidth=2,
                label=f'β={beta_plot:.3f}, R²={r2_plot:.3f}')
        
        # Add vertical line at ratio = 1
        ax.axvline(1, color='gray', linestyle='--', alpha=0.5, label='No persistence change')
        
        ax.set_xlabel('Volatility Persistence Ratio (Post/Pre)')
        ax.set_ylabel(f'Avg Post-Event Expected Return ({self.config.vol_persistence_window_days}-day window)')
        ax.set_title(f'{self.event_type.upper()}: H2.2 - Volatility Persistence Extends Returns')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, f'{self.event_type}_h2_2_persistence.png'),
                    dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_h2_3(self):
        """Plot distribution of gamma parameters."""
        gamma_values_list = []
        for symbol, params_instance in self.event_processor.volatility_params.items():
            if isinstance(params_instance, VolatilityParameters) and hasattr(params_instance, 'gamma') and params_instance.gamma is not None:
                gamma_values_list.append(params_instance.gamma)
        
        if not gamma_values_list:
            print("No gamma values for H2.3 plot.")
            return

        gamma_values = np.array(gamma_values_list)
        gamma_values = gamma_values[~np.isnan(gamma_values)] # Remove NaNs

        if len(gamma_values) < 2:
            print("Not enough gamma values for H2.3 plot after NaN removal.")
            return
            
        # Remove outliers
        q_low_gamma, q_high_gamma = np.percentile(gamma_values, [1, 99])
        gamma_clean = gamma_values[
            (gamma_values >= q_low_gamma) &
            (gamma_values <= q_high_gamma)
        ]
        
        if len(gamma_clean) < 2:
            print("Not enough gamma values for H2.3 plot after outlier removal.")
            return

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Histogram
        ax1.hist(gamma_clean, bins=50, density=True, alpha=0.7, color='orange', edgecolor='black')
        ax1.axvline(0, color='black', linestyle='--', linewidth=2, label='γ = 0')
        ax1.axvline(gamma_clean.mean(), color='red', linestyle='-', linewidth=2,
                   label=f'Mean γ = {gamma_clean.mean():.3f}')
        
        # Add normal distribution overlay
        x_norm = np.linspace(gamma_clean.min(), gamma_clean.max(), 100)
        try:
            ax1.plot(x_norm, stats.norm.pdf(x_norm, gamma_clean.mean(), gamma_clean.std()),
                    'b-', linewidth=2, label='Normal fit')
        except Exception as e:
            print(f"Could not plot normal fit for H2.3: {e}")

        ax1.set_xlabel('Gamma (γ) Parameter')
        ax1.set_ylabel('Density')
        ax1.set_title('Distribution of Asymmetry Parameters (γ)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Q-Q plot
        stats.probplot(gamma_clean, dist="norm", plot=ax2)
        ax2.set_title('Q-Q Plot of Gamma Parameters (γ)')
        ax2.grid(True, alpha=0.3)
        
        # Add test results as text
        if not self.h2_3_results.empty:
            result = self.h2_3_results.iloc[0]
            text = f"Mean γ: {result['mean_gamma']:.4f}\n"
            text += f"t-stat (γ>0): {result['t_statistic']:.3f}\n"
            text += f"p-value (γ>0): {result['p_value_gamma_positive']:.3f}\n"
            text += f"Significant (γ>0): {'Yes' if result['gamma_positive_significant'] else 'No'}"
            
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
        
        tests = ['H2.1\nVol Innovations\nPredict Returns',
                 'H2.2\nPersistence Extends\nReturns',
                 'H2.3\nAsymmetric Response\n(γ > 0)']
        
        supported_flags = []
        p_values_for_plot = []
        
        # H2.1
        if not self.h2_1_results.empty and 'significant_5pct' in self.h2_1_results.columns:
            h2_1_support = any(self.h2_1_results['significant_5pct'])
            supported_flags.append(h2_1_support)
            # Use p-value of the best R-squared or most significant horizon if available
            if h2_1_support and 'p_value' in self.h2_1_results.columns:
                 best_h21 = self.h2_1_results[self.h2_1_results['significant_5pct']].sort_values('p_value').iloc[0]
                 p_values_for_plot.append(best_h21['p_value'])
            elif 'p_value' in self.h2_1_results.columns and not self.h2_1_results['p_value'].isnull().all():
                 p_values_for_plot.append(self.h2_1_results['p_value'].min()) # Smallest p-value
            else:
                 p_values_for_plot.append(1.0)
        else:
            supported_flags.append(False)
            p_values_for_plot.append(1.0)
        
        # H2.2
        if not self.h2_2_results.empty and 'positive_and_significant' in self.h2_2_results.columns:
            supported_flags.append(self.h2_2_results['positive_and_significant'].iloc[0])
            p_values_for_plot.append(self.h2_2_results['p_value_one_sided'].iloc[0])
        else:
            supported_flags.append(False)
            p_values_for_plot.append(1.0)
        
        # H2.3
        if not self.h2_3_results.empty and 'gamma_positive_significant' in self.h2_3_results.columns:
            supported_flags.append(self.h2_3_results['gamma_positive_significant'].iloc[0])
            p_values_for_plot.append(self.h2_3_results['p_value_gamma_positive'].iloc[0])
        else:
            supported_flags.append(False)
            p_values_for_plot.append(1.0)
        
        x_pos = np.arange(len(tests))
        colors = ['green' if s else 'red' for s in supported_flags]
        
        bars = ax.bar(x_pos, [1 if s else 0 for s in supported_flags], color=colors, alpha=0.7)
        
        for i, (bar, p_val) in enumerate(zip(bars, p_values_for_plot)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                   f'p={p_val:.3f}' if not np.isnan(p_val) else 'p=N/A', 
                   ha='center', va='bottom', fontsize=12)
        
        ax.set_ylim(0, 1.2)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(tests)
        ax.set_yticks([0, 1])
        ax.set_yticklabels(['Not Supported', 'Supported'])
        ax.set_title(f'{self.event_type.upper()}: Hypothesis 2 Test Results Summary')
        ax.grid(True, alpha=0.3, axis='y')
        
        n_supported = sum(supported_flags)
        overall_text = f"Overall: {n_supported}/{len(tests)} sub-hypotheses supported"
        ax.text(0.5, 0.5, overall_text, transform=ax.transAxes,
               fontsize=14, ha='center', va='center',
               bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, f'{self.event_type}_h2_summary_plot.png'), # Renamed to avoid conflict with CSV
                    dpi=300, bbox_inches='tight')
        plt.close()
    
    def _save_results(self):
        """Save analysis results to files."""
        print("\nSaving results...")
        
        # Save processed data (sample to reduce size)
        if self.processed_data is not None and not self.processed_data.empty:
            sample_size = min(10000, len(self.processed_data))
            self.processed_data.sample(n=sample_size, shuffle=True, seed=42).to_csv( # Added shuffle and seed
                os.path.join(self.results_dir, f'{self.event_type}_h2_processed_sample.csv'),
                index=False
            )
        
        # Save test results
        if not self.h2_1_results.empty:
            self.h2_1_results.to_csv(
                os.path.join(self.results_dir, f'{self.event_type}_h2_1_results.csv'),
                index=False
            )
        
        if not self.h2_2_results.empty:
            self.h2_2_results.to_csv(
                os.path.join(self.results_dir, f'{self.event_type}_h2_2_results.csv'),
                index=False
            )
        
        if not self.h2_3_results.empty:
            self.h2_3_results.to_csv(
                os.path.join(self.results_dir, f'{self.event_type}_h2_3_results.csv'),
                index=False
            )
        
        # Create overall summary
        summary = {
            'event_type': self.event_type,
            'n_events': self.processed_data['event_id'].nunique() if self.processed_data is not None else 0,
            'n_symbols': self.processed_data['symbol'].nunique() if self.processed_data is not None else 0,
            'n_obs': len(self.processed_data) if self.processed_data is not None else 0,
            'h2_1_supported': any(self.h2_1_results['significant_5pct']) if not self.h2_1_results.empty and 'significant_5pct' in self.h2_1_results else False,
            'h2_2_supported': self.h2_2_results['positive_and_significant'].iloc[0] if not self.h2_2_results.empty and 'positive_and_significant' in self.h2_2_results else False,
            'h2_3_supported': self.h2_3_results['gamma_positive_significant'].iloc[0] if not self.h2_3_results.empty and 'gamma_positive_significant' in self.h2_3_results else False
        }
        
        pd.DataFrame([summary]).to_csv(
            os.path.join(self.results_dir, f'{self.event_type}_h2_overall_summary.csv'), # Renamed to avoid conflict
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
    else: # earnings
        results_dir = EARNINGS_RESULTS_DIR # Corrected variable name
    
    # Initialize tester
    tester = Hypothesis2Tester(config, event_type, results_dir)
    
    # Load data
    stock_df = tester.load_stock_data(stock_files)
    if stock_df is None or stock_df.is_empty():
        print(f"Failed to load stock data for H2 {event_type}.")
        return False
    
    event_df = tester.load_event_data(event_file, date_col, ticker_col)
    if event_df is None or event_df.is_empty():
        print(f"Failed to load event data for H2 {event_type}.")
        return False
    
    # Run analysis
    return tester.run_analysis(stock_df, event_df)


def compare_results():
    """Compare results between FDA and earnings events."""
    print("\n" + "="*60)
    print("Comparing FDA and Earnings Results for Hypothesis 2")
    print("="*60)
    
    fda_summary_path = os.path.join(FDA_RESULTS_DIR, 'fda_h2_overall_summary.csv')
    earnings_summary_path = os.path.join(EARNINGS_RESULTS_DIR, 'earnings_h2_overall_summary.csv')

    if not os.path.exists(fda_summary_path):
        print(f"FDA H2 summary file not found: {fda_summary_path}")
        return
    if not os.path.exists(earnings_summary_path):
        print(f"Earnings H2 summary file not found: {earnings_summary_path}")
        return

    # Load summaries
    fda_summary = pd.read_csv(fda_summary_path)
    earnings_summary = pd.read_csv(earnings_summary_path)
    
    if fda_summary.empty or earnings_summary.empty:
        print("One or both H2 summary files are empty. Cannot compare.")
        return

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
    
    ax.set_ylabel('Hypothesis Supported (1=Yes, 0=No)')
    ax.set_title('Hypothesis 2 Support: FDA vs Earnings Events')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(tests)
    ax.set_yticks([0,1])
    ax.set_yticklabels(['No', 'Yes'])
    ax.legend()
    ax.set_ylim(-0.1, 1.2) # Adjusted ylim slightly for better visual
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    # Save comparison
    comparison_dir = 'results/hypothesis2/comparison/'
    os.makedirs(comparison_dir, exist_ok=True)
    plt.savefig(os.path.join(comparison_dir, 'h2_comparison_plot.png'), 
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
    else:
        print("\nH2 Comparison skipped as one or both H2 analyses did not complete successfully.")

    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()
