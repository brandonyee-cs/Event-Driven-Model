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
        
        os.makedirs(self.results_dir, exist_ok=True)
        
        self.processed_data = None
        self.h2_1_results = pd.DataFrame() 
        self.h2_2_results = pd.DataFrame() 
        self.h2_3_results = pd.DataFrame() 
        
        self.pre_event_innovation_window = (-10, -1)
    
    def load_stock_data(self, file_paths: List[str]) -> Optional[pl.DataFrame]:
        """Load and combine stock price data from multiple files."""
        print(f"Loading stock data from {len(file_paths)} files...")
        
        dfs = []
        for i, path in enumerate(file_paths):
            if i % 2 == 0:
                print(f"  Loading file {i+1}/{len(file_paths)}...")
            
            try:
                df_raw = pl.read_parquet(path)
                
                source_ticker_col = None
                source_date_col = None
                source_price_col = None
                source_returns_col = None
                col_map_upper = {col.upper(): col for col in df_raw.columns}

                if 'PERMNO' in col_map_upper: source_ticker_col = col_map_upper['PERMNO']
                elif 'TICKER' in col_map_upper: source_ticker_col = col_map_upper['TICKER']
                elif 'SYMBOL' in col_map_upper: source_ticker_col = col_map_upper['SYMBOL']
                if 'DATE' in col_map_upper: source_date_col = col_map_upper['DATE']
                if 'PRC' in col_map_upper: source_price_col = col_map_upper['PRC']
                elif 'PRICE' in col_map_upper: source_price_col = col_map_upper['PRICE']
                if 'RET' in col_map_upper: source_returns_col = col_map_upper['RET']
                elif 'RETURNS' in col_map_upper: source_returns_col = col_map_upper['RETURNS']
                elif 'RETURN' in col_map_upper: source_returns_col = col_map_upper['RETURN']

                missing_cols = []
                if not source_ticker_col: missing_cols.append("ticker (PERMNO/TICKER/SYMBOL)")
                if not source_date_col: missing_cols.append("date (DATE)")
                if not source_price_col: missing_cols.append("price (PRC/PRICE)")
                if not source_returns_col: missing_cols.append("returns (RET/RETURNS/RETURN)")

                if missing_cols:
                    print(f"  Warning: Could not find all required source columns in {path}. Missing: {', '.join(missing_cols)}.")
                    print(f"    Available columns (original): {df_raw.columns}")
                    continue

                df = df_raw.select([
                    pl.col(source_ticker_col).alias('ticker'),
                    pl.col(source_date_col).alias('date'),
                    pl.col(source_price_col).alias('price'),
                    pl.col(source_returns_col).alias('returns')
                ])
                
                df = df.with_columns([
                    pl.col('ticker').cast(pl.Utf8),
                    pl.col('date').cast(pl.Date),
                    pl.col('price').cast(pl.Float64).abs(),
                    pl.col('returns').cast(pl.Float64)
                ])
                dfs.append(df)
            except Exception as e:
                print(f"  Warning: Error processing file {path}: {e}")
                continue
        
        if not dfs:
            print("Error: No stock data loaded!")
            return None
        
        combined_df = pl.concat(dfs)
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
            df = pl.read_csv(file_path, try_parse_dates=True, infer_schema_length=10000)
            actual_date_col = None
            actual_ticker_col = None
            df_cols_lower = {c.lower(): c for c in df.columns}

            if date_col.lower() in df_cols_lower: actual_date_col = df_cols_lower[date_col.lower()]
            if ticker_col.lower() in df_cols_lower: actual_ticker_col = df_cols_lower[ticker_col.lower()]
            
            if not actual_date_col or not actual_ticker_col:
                print(f"Error: Could not find required columns '{date_col}' or '{ticker_col}' in event file.")
                print(f"Available columns: {df.columns}")
                return None
            
            df = df.rename({
                actual_date_col: 'event_date',
                actual_ticker_col: 'symbol' # Rename to 'symbol'
            })
            
            df = df.with_columns([
                pl.col('event_date').cast(pl.Date),
                pl.col('symbol').cast(pl.Utf8).str.strip_chars(),
                pl.lit(self.event_type).alias('event_type')
            ])
            
            df = df.filter(
                pl.col('event_date').is_not_null() &
                pl.col('symbol').is_not_null() &
                (pl.col('symbol').str.len_chars() > 0)
            ).unique(subset=['symbol', 'event_date'])
            
            print(f"Loaded {df.height:,} {self.event_type} events")
            return df
        except Exception as e:
            print(f"Error loading event data from {file_path}: {e}")
            return None
    
    def run_analysis(self, stock_df: pl.DataFrame, event_df: pl.DataFrame) -> bool:
        """
        Run the complete Hypothesis 2 analysis.
        """
        print(f"\n{'='*60}")
        print(f"Running Hypothesis 2 Analysis for {self.event_type.upper()} Events")
        print(f"{'='*60}")

        stock_pd_df = stock_df.to_pandas()
        event_pd_df = event_df.to_pandas()

        if 'ticker' in stock_pd_df.columns and 'symbol' not in stock_pd_df.columns:
            stock_pd_df = stock_pd_df.rename(columns={'ticker': 'symbol'})
        elif 'ticker' not in stock_pd_df.columns and 'symbol' not in stock_pd_df.columns:
            print("Critical Error: Neither 'ticker' nor 'symbol' found in stock_pd_df for H2.")
            return False
        
        if 'symbol' not in event_pd_df.columns:
            print("Critical Error: 'symbol' column missing in event_pd_df for H2.")
            if 'ticker' in event_pd_df.columns: event_pd_df = event_pd_df.rename(columns={'ticker':'symbol'})
            else: return False
        
        print("\nProcessing events with unified volatility model...")
        self.processed_data = self.event_processor.process_events(
            price_data=stock_pd_df,
            event_data=event_pd_df
        )
        
        if self.processed_data is None or self.processed_data.empty:
            print("Error: No events successfully processed for H2")
            return False
        
        print(f"Processed {len(self.processed_data):,} event-day observations for H2")
        
        self.processed_data['event_id'] = (
            self.processed_data['symbol'] + '_' + 
            self.processed_data['event_date'].dt.strftime('%Y%m%d')
        )
        
        self._test_h2_1()
        self._test_h2_2()
        self._test_h2_3()
        self._create_visualizations()
        self._save_results()
        
        return True
    
    def _test_h2_1(self):
        """
        Test H2.1: Pre-event volatility innovations predict subsequent returns.
        """
        print("\n" + "-"*50)
        print("Testing H2.1: Pre-event volatility innovations predict returns")
        print("-"*50)
        
        pre_event_data = self.processed_data[
            (self.processed_data['days_to_event'] >= self.pre_event_innovation_window[0]) &
            (self.processed_data['days_to_event'] <= self.pre_event_innovation_window[1])
        ].copy()
        
        event_innovations = pre_event_data.groupby('event_id').agg(
            volatility_innovation_pre_avg=('volatility_innovation', 'mean')
        ).reset_index()
        
        if event_innovations.empty:
            print("  No pre-event innovations calculated. Skipping H2.1.")
            self.h2_1_results = pd.DataFrame()
            return

        analysis_base = self.processed_data[
            self.processed_data['days_to_event'] == 1 # Day after event for starting future returns
        ][['event_id']].drop_duplicates()

        # Merge pre-event innovations. analysis_data now has one row per event_id with its pre-event innovation.
        analysis_data_for_regression = analysis_base.merge(event_innovations, on='event_id', how='inner')

        if analysis_data_for_regression.empty:
            print("  No events to analyze for H2.1 after merging innovations. Skipping.")
            self.h2_1_results = pd.DataFrame()
            return

        results = []
        for horizon in self.config.prediction_horizons:
            print(f"\n  Testing {horizon}-day ahead prediction...")
            
            future_returns_list = []
            for event_id_val in analysis_data_for_regression['event_id'].unique():
                post_data = self.processed_data[
                    (self.processed_data['event_id'] == event_id_val) &
                    (self.processed_data['days_to_event'] > 0) & 
                    (self.processed_data['days_to_event'] <= horizon)
                ]
                
                if not post_data.empty and 'returns' in post_data.columns:
                    valid_returns = post_data['returns'].dropna()
                    if not valid_returns.empty:
                        cum_return = (1 + valid_returns).prod() - 1
                        future_returns_list.append({'event_id': event_id_val, f'return_{horizon}d': cum_return})
            
            if not future_returns_list:
                print(f"    No future returns data for {horizon}-day horizon.")
                continue
                
            future_returns_df = pd.DataFrame(future_returns_list)
            regression_data = analysis_data_for_regression.merge(future_returns_df, on='event_id', how='inner')
            
            if len(regression_data) < 10 or \
               'volatility_innovation_pre_avg' not in regression_data.columns or \
               f'return_{horizon}d' not in regression_data.columns:
                print(f"    Insufficient data for {horizon}-day horizon after merge (N={len(regression_data)}).")
                continue
            
            regression_data = regression_data[['volatility_innovation_pre_avg', f'return_{horizon}d']].dropna()
            if len(regression_data) < 10:
                print(f"    Insufficient data for {horizon}-day horizon after NaN drop (N={len(regression_data)}).")
                continue

            X = regression_data['volatility_innovation_pre_avg'].values.reshape(-1, 1)
            y = regression_data[f'return_{horizon}d'].values
            
            x_q01, x_q99 = np.percentile(X, [1, 99]) if len(X) > 0 else (np.nan, np.nan)
            y_q01, y_q99 = np.percentile(y, [1, 99]) if len(y) > 0 else (np.nan, np.nan)

            if np.isnan(x_q01) or np.isnan(y_q01): # Not enough data for percentiles
                 X_clean, y_clean = X, y
            else:
                mask = (X.flatten() >= x_q01) & (X.flatten() <= x_q99) & \
                       (y >= y_q01) & (y <= y_q99)
                X_clean, y_clean = X[mask], y[mask]
            
            if len(X_clean) < 10:
                print(f"    Insufficient data for {horizon}-day horizon after outlier removal (N={len(X_clean)}).")
                continue
            
            reg = LinearRegression().fit(X_clean, y_clean)
            y_pred = reg.predict(X_clean)
            r2 = r2_score(y_clean, y_pred)
            n_reg = len(X_clean)

            if n_reg <= 2: continue

            residuals = (y_clean - y_pred).flatten()
            se_residuals_sq = np.sum(residuals**2) / (n_reg - 2)
            se_residuals = np.sqrt(se_residuals_sq) if se_residuals_sq >=0 else np.nan
            x_var = np.sum((X_clean.flatten() - X_clean.flatten().mean())**2)
            se_beta = se_residuals / np.sqrt(x_var) if x_var > 1e-9 and pd.notna(se_residuals) else np.inf
            
            t_stat_val = reg.coef_[0] / se_beta if se_beta != np.inf and se_beta > 1e-9 else 0
            p_value_val = 2 * (1 - stats.t.cdf(abs(t_stat_val), n_reg - 2)) if n_reg > 2 else 1.0
            
            results.append({
                'horizon': horizon, 'n_obs': n_reg, 'beta': reg.coef_[0], 'alpha': reg.intercept_,
                'r_squared': r2, 't_statistic': t_stat_val, 'p_value': p_value_val,
                'significant_5pct': p_value_val < 0.05,
                'mean_innovation': X_clean.mean(), 'std_innovation': X_clean.std()
            })
            print(f"    Beta: {reg.coef_[0]:.4f}, R²: {r2:.3f}, p-value: {p_value_val:.3f} {'*' if p_value_val < 0.05 else ''}")
        
        self.h2_1_results = pd.DataFrame(results)
        if not self.h2_1_results.empty:
            supported = any(self.h2_1_results['significant_5pct'])
            print(f"\nH2.1 Overall Support: {'YES' if supported else 'NO'}")
            print(f"Significant predictions: {sum(self.h2_1_results['significant_5pct'])}/{len(self.h2_1_results)}")
        else:
            print("\nH2.1 Overall Support: NO (no results generated)")

    def _test_h2_2(self):
        print("\n" + "-"*50 + "\nTesting H2.2: Post-event volatility persistence extends returns\n" + "-"*50)
        persistence_metrics = []
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
                post_vol, post_returns_avg = post_vol_series.mean(), post_returns_series.mean()
                persistence_metrics.append({
                    'event_id': event_id_val, 'pre_event_vol': pre_vol, 'post_event_vol': post_vol,
                    'persistence_ratio': post_vol / pre_vol, 'avg_post_returns': post_returns_avg,
                    'n_post_days': len(post_data)
                })
        
        if not persistence_metrics:
            print("  Insufficient data for H2.2 test.")
            self.h2_2_results = pd.DataFrame(); return
        
        persistence_df = pd.DataFrame(persistence_metrics).replace([np.inf, -np.inf], np.nan).dropna()
        if persistence_df.empty:
            print("  Insufficient data for H2.2 test after NaN/Inf removal."); self.h2_2_results = pd.DataFrame(); return

        q_low_r, q_high_r = persistence_df['persistence_ratio'].quantile([0.01, 0.99])
        q_low_ret, q_high_ret = persistence_df['avg_post_returns'].quantile([0.01, 0.99])
        persistence_df = persistence_df[
            (persistence_df['persistence_ratio'] >= q_low_r) & (persistence_df['persistence_ratio'] <= q_high_r) &
            (persistence_df['avg_post_returns'] >= q_low_ret) & (persistence_df['avg_post_returns'] <= q_high_ret)
        ]
        
        if len(persistence_df) < 10:
            print(f"  Insufficient data after outlier removal (N={len(persistence_df)})."); self.h2_2_results = pd.DataFrame(); return
        
        X, y = persistence_df['persistence_ratio'].values.reshape(-1, 1), persistence_df['avg_post_returns'].values
        reg = LinearRegression().fit(X, y)
        y_pred, r2 = reg.predict(X), r2_score(y, y_pred)
        n_reg = len(X)

        if n_reg <= 2: self.h2_2_results = pd.DataFrame(); return
            
        residuals = (y - y_pred).flatten()
        se_residuals_sq = np.sum(residuals**2) / (n_reg - 2)
        se_residuals = np.sqrt(se_residuals_sq) if se_residuals_sq >= 0 else np.nan
        x_var = np.sum((X.flatten() - X.flatten().mean())**2)
        se_beta = se_residuals / np.sqrt(x_var) if x_var > 1e-9 and pd.notna(se_residuals) else np.inf
        
        t_stat_val = reg.coef_[0] / se_beta if se_beta != np.inf and se_beta > 1e-9 else 0
        p_val_two_sided = 2 * (1 - stats.t.cdf(abs(t_stat_val), n_reg - 2)) if n_reg > 2 else 1.0
        p_val_one_sided = p_val_two_sided / 2 if reg.coef_[0] > 0 else 1 - p_val_two_sided / 2
        
        self.h2_2_results = pd.DataFrame([{'n_events': n_reg, 'beta': reg.coef_[0], 'alpha': reg.intercept_, 'r_squared': r2,
            't_statistic': t_stat_val, 'p_value_two_sided': p_val_two_sided, 'p_value_one_sided': p_val_one_sided,
            'positive_and_significant': (reg.coef_[0] > 0) and (p_val_one_sided < 0.05),
            'mean_persistence_ratio': X.mean(), 'std_persistence_ratio': X.std()}])
        
        print(f"\n  Regression Results:\n    Beta: {reg.coef_[0]:.4f}\n    R²: {r2:.3f}\n    p-value (one-sided): {p_val_one_sided:.3f} {'*' if self.h2_2_results['positive_and_significant'].iloc[0] else ''}")
        print(f"\n  H2.2 Supported: {'YES' if self.h2_2_results['positive_and_significant'].iloc[0] else 'NO'}")

    def _test_h2_3(self):
        print("\n" + "-"*50 + "\nTesting H2.3: Asymmetric volatility response\n" + "-"*50)
        gamma_list = []
        for symbol, params_instance in self.event_processor.volatility_params.items():
            if isinstance(params_instance, VolatilityParameters) and hasattr(params_instance, 'gamma') and params_instance.gamma is not None:
                gamma_list.append({'symbol': symbol, 'gamma': params_instance.gamma})
        
        if not gamma_list: print("  No gamma parameters available."); self.h2_3_results = pd.DataFrame(); return
        gamma_df = pd.DataFrame(gamma_list).dropna(subset=['gamma'])
        if gamma_df.empty: print("  No valid gamma parameters after filtering."); self.h2_3_results = pd.DataFrame(); return
            
        q_low_g, q_high_g = gamma_df['gamma'].quantile([0.01, 0.99])
        gamma_df = gamma_df[(gamma_df['gamma'] >= q_low_g) & (gamma_df['gamma'] <= q_high_g)]
        if gamma_df.empty: print("  No gamma parameters after outlier removal."); self.h2_3_results = pd.DataFrame(); return

        gammas = gamma_df['gamma'].values
        if len(gammas) < 2: print(f"  Insufficient gamma values for t-test (N={len(gammas)})."); self.h2_3_results = pd.DataFrame(); return

        t_stat_val, p_val_two_sided = stats.ttest_1samp(gammas, 0, nan_policy='omit')
        p_val_one_sided = p_val_two_sided / 2 if t_stat_val > 0 else 1 - p_val_two_sided / 2
        
        mean_g, se_g = np.nanmean(gammas), np.nanstd(gammas) / np.sqrt(np.sum(~np.isnan(gammas))) if np.sum(~np.isnan(gammas)) > 0 else np.nan
        ci_l, ci_u = (mean_g - stats.t.ppf(0.975, np.sum(~np.isnan(gammas))-1) * se_g, mean_g + stats.t.ppf(0.975, np.sum(~np.isnan(gammas))-1) * se_g) if not np.isnan(se_g) and np.sum(~np.isnan(gammas)) > 1 else (np.nan, np.nan)
        prop_pos = np.nanmean(gammas > 0) if len(gammas) > 0 else np.nan
        n_pos, n_tot = np.sum(gammas > 0), len(gammas)
        binom_p = stats.binomtest(n_pos, n_tot, p=0.5, alternative='greater').pvalue if n_tot > 0 else np.nan
        
        self.h2_3_results = pd.DataFrame([{'n_stocks': n_tot, 'mean_gamma': mean_g, 'std_gamma': np.nanstd(gammas),
            'median_gamma': np.nanmedian(gammas), 'ci_lower': ci_l, 'ci_upper': ci_u, 't_statistic': t_stat_val,
            'p_value_gamma_positive': p_val_one_sided, 'gamma_positive_significant': p_val_one_sided < 0.05,
            'prop_positive_gamma': prop_pos, 'binom_p_value': binom_p,
            'majority_positive_significant': binom_p < 0.05 if pd.notna(binom_p) else False}])
        
        print(f"\n  Asymmetry Parameter (γ) Analysis:\n    Mean γ: {mean_g:.4f} (95% CI: [{ci_l:.4f}, {ci_u:.4f}])")
        print(f"    Proportion positive: {prop_pos:.1%}" if pd.notna(prop_pos) else "N/A")
        print(f"    T-test (γ > 0): t={t_stat_val:.3f}, p={p_val_one_sided:.3f} {'*' if p_val_one_sided < 0.05 else ''}")
        print(f"    Binomial test: p={binom_p:.3f} {'*' if pd.notna(binom_p) and binom_p < 0.05 else ''}")
        h2_3_supp = self.h2_3_results['gamma_positive_significant'].iloc[0] if not self.h2_3_results.empty else False
        print(f"\n  H2.3 Supported: {'YES' if h2_3_supp else 'NO'}")

    def _create_visualizations(self):
        print("\nGenerating visualizations...")
        if self.processed_data is None or self.processed_data.empty:
            print("Warning: No processed data for H2 visualizations."); return

        if not self.h2_1_results.empty: self._plot_h2_1()
        else: print("Skipping H2.1 visualization: no results.")
        
        if not self.h2_2_results.empty: self._plot_h2_2()
        else: print("Skipping H2.2 visualization: no results.")

        if not self.h2_3_results.empty: self._plot_h2_3()
        else: print("Skipping H2.3 visualization: no results.")
        
        self._plot_summary() # Always try to plot summary
    
    def _plot_h2_1(self):
        if 'r_squared' not in self.h2_1_results.columns or self.h2_1_results['r_squared'].isnull().all():
            print("Cannot plot H2.1 scatter: R-squared data missing/NaN."); self._plot_h2_1_horizons_only(); return

        best_row = self.h2_1_results.loc[self.h2_1_results['r_squared'].idxmax()] if self.h2_1_results['r_squared'].notna().any() else self.h2_1_results.iloc[0]
        horizon = int(best_row['horizon'])
        
        pre_event_data = self.processed_data[(self.processed_data['days_to_event'] >= self.pre_event_innovation_window[0]) & (self.processed_data['days_to_event'] <= self.pre_event_innovation_window[1])]
        event_innovations = pre_event_data.groupby('event_id')['volatility_innovation'].mean().reset_index().rename(columns={'volatility_innovation': 'volatility_innovation_pre_avg'})

        future_returns_list = []
        for eid in event_innovations['event_id']:
            post_d = self.processed_data[(self.processed_data['event_id'] == eid) & (self.processed_data['days_to_event'] > 0) & (self.processed_data['days_to_event'] <= horizon)]
            if not post_d.empty and 'returns' in post_d.columns:
                valid_r = post_d['returns'].dropna()
                if not valid_r.empty: future_returns_list.append({'event_id': eid, f'future_return_{horizon}d': (1 + valid_r).prod() - 1})
        
        if not future_returns_list: print(f"No future returns for H2.1 scatter (h={horizon})."); self._plot_h2_1_horizons_only(); return
        plot_data = event_innovations.merge(pd.DataFrame(future_returns_list), on='event_id', how='inner')[['volatility_innovation_pre_avg', f'future_return_{horizon}d']].dropna()
        if plot_data.empty or len(plot_data) < 2: print(f"Not enough data for H2.1 scatter (h={horizon}) after merge/NaN."); self._plot_h2_1_horizons_only(); return

        x_q01, x_q99 = plot_data['volatility_innovation_pre_avg'].quantile([0.01, 0.99])
        y_q01, y_q99 = plot_data[f'future_return_{horizon}d'].quantile([0.01, 0.99])
        plot_clean = plot_data[(plot_data['volatility_innovation_pre_avg'] >= x_q01) & (plot_data['volatility_innovation_pre_avg'] <= x_q99) & (plot_data[f'future_return_{horizon}d'] >= y_q01) & (plot_data[f'future_return_{horizon}d'] <= y_q99)]
        if plot_clean.empty or len(plot_clean) < 2: print(f"Not enough data for H2.1 scatter (h={horizon}) after outliers."); self._plot_h2_1_horizons_only(); return

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(plot_clean['volatility_innovation_pre_avg'], plot_clean[f'future_return_{horizon}d'], alpha=0.5, s=50, color='blue')
        X_plot, y_plot = plot_clean['volatility_innovation_pre_avg'].values.reshape(-1, 1), plot_clean[f'future_return_{horizon}d'].values
        reg = LinearRegression().fit(X_plot, y_plot)
        x_line, y_line = np.linspace(X_plot.min(), X_plot.max(), 100), reg.predict(np.linspace(X_plot.min(), X_plot.max(), 100).reshape(-1, 1))
        r2_plot = r2_score(y_plot, reg.predict(X_plot))
        ax.plot(x_line, y_line, color='red', linewidth=2, label=f'β={reg.coef_[0]:.3f}, R²={r2_plot:.3f}')
        ax.set_xlabel('Pre-Event Volatility Innovation (Average)'); ax.set_ylabel(f'{horizon}-Day Future Return')
        ax.set_title(f'{self.event_type.upper()}: H2.1 - Vol Innovations vs Returns (Horizon {horizon}d)'); ax.legend(); ax.grid(True, alpha=0.3)
        plt.tight_layout(); plt.savefig(os.path.join(self.results_dir, f'{self.event_type}_h2_1_scatter.png'), dpi=300, bbox_inches='tight'); plt.close()
        self._plot_h2_1_horizons_only()

    def _plot_h2_1_horizons_only(self):
        if self.h2_1_results.empty or 'horizon' not in self.h2_1_results.columns: print("No H2.1 horizon data."); return
        fig, ax = plt.subplots(figsize=(10, 6))
        horizons, betas, p_values = self.h2_1_results['horizon'].values, self.h2_1_results['beta'].values, self.h2_1_results['p_value'].values
        ax.bar(horizons.astype(str), betas, color=['green' if p < 0.05 else 'gray' for p in p_values])
        for h, b, p in zip(horizons, betas, p_values):
            if not (np.isnan(b) or np.isnan(p)) and p < 0.05:
                ax.text(str(h), b + (0.001 * np.sign(b) if b!=0 else 0.001), '*', ha='center', va='bottom' if b >= 0 else 'top', fontsize=16, color='red')
        ax.set_xlabel('Prediction Horizon (days)'); ax.set_ylabel('Beta Coefficient'); ax.set_title(f'{self.event_type.upper()}: H2.1 - Impact of Vol Innovations on Future Returns')
        ax.grid(True, alpha=0.3, axis='y'); ax.axhline(0, color='black', linestyle='-', linewidth=0.5); plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, f'{self.event_type}_h2_1_horizons.png'), dpi=300, bbox_inches='tight'); plt.close()

    def _plot_h2_2(self):
        persistence_list = []
        for eid in self.processed_data['event_id'].unique():
            event_d = self.processed_data[self.processed_data['event_id'] == eid]
            pre_vol_s = event_d[event_d['days_to_event'] < 0]['baseline_volatility'].dropna()
            if pre_vol_s.empty: continue
            pre_v = pre_vol_s.mean()
            post_d = event_d[(event_d['days_to_event'] > 0) & (event_d['days_to_event'] <= self.config.vol_persistence_window_days)]
            post_vol_s, post_ret_s = post_d['baseline_volatility'].dropna(), post_d['expected_return'].dropna()
            if not post_vol_s.empty and not post_ret_s.empty and pre_v > 1e-9:
                persistence_list.append({'persistence_ratio': post_vol_s.mean() / pre_v, 'avg_post_returns': post_ret_s.mean()})
        
        if not persistence_list: print("No data for H2.2 plot."); return
        plot_df = pd.DataFrame(persistence_list).replace([np.inf, -np.inf], np.nan).dropna()
        if plot_df.empty or len(plot_df) < 2: print("Not enough data for H2.2 plot after NaN/Inf."); return

        q_low_r, q_high_r = plot_df['persistence_ratio'].quantile([0.01, 0.99])
        q_low_ret, q_high_ret = plot_df['avg_post_returns'].quantile([0.01, 0.99])
        plot_clean = plot_df[(plot_df['persistence_ratio'] >= q_low_r) & (plot_df['persistence_ratio'] <= q_high_r) & (plot_df['avg_post_returns'] >= q_low_ret) & (plot_df['avg_post_returns'] <= q_high_ret)]
        if plot_clean.empty or len(plot_clean) < 2: print("Not enough data for H2.2 plot after outliers."); return

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(plot_clean['persistence_ratio'], plot_clean['avg_post_returns'], alpha=0.5, s=50, color='purple')
        X_plot, y_plot = plot_clean['persistence_ratio'].values.reshape(-1, 1), plot_clean['avg_post_returns'].values
        reg = LinearRegression().fit(X_plot, y_plot)
        x_line, y_line = np.linspace(X_plot.min(), X_plot.max(), 100), reg.predict(np.linspace(X_plot.min(), X_plot.max(), 100).reshape(-1,1))
        r2_plot, beta_plot = r2_score(y_plot, reg.predict(X_plot)), reg.coef_[0]
        ax.plot(x_line, y_line, color='red', linewidth=2, label=f'β={beta_plot:.3f}, R²={r2_plot:.3f}')
        ax.axvline(1, color='gray', linestyle='--', alpha=0.5, label='No persistence change')
        ax.set_xlabel('Volatility Persistence Ratio (Post/Pre)'); ax.set_ylabel(f'Avg Post-Event Expected Return ({self.config.vol_persistence_window_days}-day)')
        ax.set_title(f'{self.event_type.upper()}: H2.2 - Volatility Persistence Extends Returns'); ax.legend(); ax.grid(True, alpha=0.3)
        plt.tight_layout(); plt.savefig(os.path.join(self.results_dir, f'{self.event_type}_h2_2_persistence.png'), dpi=300, bbox_inches='tight'); plt.close()

    def _plot_h2_3(self):
        gamma_vals = [p.gamma for s, p in self.event_processor.volatility_params.items() if isinstance(p, VolatilityParameters) and hasattr(p, 'gamma') and p.gamma is not None]
        if not gamma_vals: print("No gamma values for H2.3 plot."); return
        gamma_arr = np.array(gamma_vals)[~np.isnan(gamma_vals)]
        if len(gamma_arr) < 2: print("Not enough gamma values for H2.3 plot after NaN removal."); return
            
        q_low_g, q_high_g = np.percentile(gamma_arr, [1, 99])
        gamma_clean = gamma_arr[(gamma_arr >= q_low_g) & (gamma_arr <= q_high_g)]
        if len(gamma_clean) < 2: print("Not enough gamma values for H2.3 plot after outliers."); return

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        ax1.hist(gamma_clean, bins=50, density=True, alpha=0.7, color='orange', edgecolor='black')
        ax1.axvline(0, color='black', linestyle='--', linewidth=2, label='γ = 0')
        ax1.axvline(gamma_clean.mean(), color='red', linestyle='-', linewidth=2, label=f'Mean γ = {gamma_clean.mean():.3f}')
        try: ax1.plot(np.linspace(gamma_clean.min(), gamma_clean.max(), 100), stats.norm.pdf(np.linspace(gamma_clean.min(), gamma_clean.max(), 100), gamma_clean.mean(), gamma_clean.std()), 'b-', linewidth=2, label='Normal fit')
        except Exception as e: print(f"Could not plot normal fit for H2.3: {e}")
        ax1.set_xlabel('Gamma (γ) Parameter'); ax1.set_ylabel('Density'); ax1.set_title('Distribution of Asymmetry Parameters (γ)'); ax1.legend(); ax1.grid(True, alpha=0.3)
        stats.probplot(gamma_clean, dist="norm", plot=ax2); ax2.set_title('Q-Q Plot of Gamma Parameters (γ)'); ax2.grid(True, alpha=0.3)
        
        if not self.h2_3_results.empty:
            res = self.h2_3_results.iloc[0]
            text = f"Mean γ: {res['mean_gamma']:.4f}\nt-stat (γ>0): {res['t_statistic']:.3f}\np-value (γ>0): {res['p_value_gamma_positive']:.3f}\nSignificant (γ>0): {'Yes' if res['gamma_positive_significant'] else 'No'}"
            ax1.text(0.05, 0.95, text, transform=ax1.transAxes, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8), verticalalignment='top', fontsize=10)
        plt.tight_layout(); plt.savefig(os.path.join(self.results_dir, f'{self.event_type}_h2_3_gamma.png'), dpi=300, bbox_inches='tight'); plt.close()

    def _plot_summary(self):
        fig, ax = plt.subplots(figsize=(10, 6))
        tests = ['H2.1\nVol Innovations\nPredict Returns', 'H2.2\nPersistence Extends\nReturns', 'H2.3\nAsymmetric Response\n(γ > 0)']
        flags, p_vals = [], []

        if not self.h2_1_results.empty and 'significant_5pct' in self.h2_1_results.columns:
            supp = any(self.h2_1_results['significant_5pct'])
            flags.append(supp)
            p_vals.append(self.h2_1_results[self.h2_1_results['significant_5pct']]['p_value'].min() if supp and 'p_value' in self.h2_1_results.columns else (self.h2_1_results['p_value'].min() if 'p_value' in self.h2_1_results.columns and not self.h2_1_results['p_value'].isnull().all() else 1.0))
        else: flags.append(False); p_vals.append(1.0)
        
        if not self.h2_2_results.empty and 'positive_and_significant' in self.h2_2_results.columns:
            flags.append(self.h2_2_results['positive_and_significant'].iloc[0]); p_vals.append(self.h2_2_results['p_value_one_sided'].iloc[0])
        else: flags.append(False); p_vals.append(1.0)
        
        if not self.h2_3_results.empty and 'gamma_positive_significant' in self.h2_3_results.columns:
            flags.append(self.h2_3_results['gamma_positive_significant'].iloc[0]); p_vals.append(self.h2_3_results['p_value_gamma_positive'].iloc[0])
        else: flags.append(False); p_vals.append(1.0)
        
        x_pos = np.arange(len(tests))
        bars = ax.bar(x_pos, [1 if s else 0 for s in flags], color=['green' if s else 'red' for s in flags], alpha=0.7)
        for i, (bar, p) in enumerate(zip(bars, p_vals)):
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.05, f'p={p:.3f}' if pd.notna(p) else 'p=N/A', ha='center', va='bottom', fontsize=12)
        
        ax.set_ylim(0, 1.2); ax.set_xticks(x_pos); ax.set_xticklabels(tests); ax.set_yticks([0, 1]); ax.set_yticklabels(['Not Supported', 'Supported'])
        ax.set_title(f'{self.event_type.upper()}: Hypothesis 2 Test Results Summary'); ax.grid(True, alpha=0.3, axis='y')
        ax.text(0.5, 0.5, f"Overall: {sum(flags)}/{len(tests)} supported", transform=ax.transAxes, fontsize=14, ha='center', va='center', bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
        plt.tight_layout(); plt.savefig(os.path.join(self.results_dir, f'{self.event_type}_h2_summary_plot.png'), dpi=300, bbox_inches='tight'); plt.close()

    def _save_results(self):
        print("\nSaving results...")
        if self.processed_data is not None and not self.processed_data.empty:
            self.processed_data.sample(n=min(10000, len(self.processed_data)), shuffle=True, random_state=42).to_csv(os.path.join(self.results_dir, f'{self.event_type}_h2_processed_sample.csv'), index=False)
        
        if not self.h2_1_results.empty: self.h2_1_results.to_csv(os.path.join(self.results_dir, f'{self.event_type}_h2_1_results.csv'), index=False)
        if not self.h2_2_results.empty: self.h2_2_results.to_csv(os.path.join(self.results_dir, f'{self.event_type}_h2_2_results.csv'), index=False)
        if not self.h2_3_results.empty: self.h2_3_results.to_csv(os.path.join(self.results_dir, f'{self.event_type}_h2_3_results.csv'), index=False)
        
        summary = {'event_type': self.event_type,
                   'n_events': self.processed_data['event_id'].nunique() if self.processed_data is not None and 'event_id' in self.processed_data else 0,
                   'n_symbols': self.processed_data['symbol'].nunique() if self.processed_data is not None and 'symbol' in self.processed_data else 0,
                   'n_obs': len(self.processed_data) if self.processed_data is not None else 0,
                   'h2_1_supported': any(self.h2_1_results['significant_5pct']) if not self.h2_1_results.empty and 'significant_5pct' in self.h2_1_results else False,
                   'h2_2_supported': self.h2_2_results['positive_and_significant'].iloc[0] if not self.h2_2_results.empty and 'positive_and_significant' in self.h2_2_results else False,
                   'h2_3_supported': self.h2_3_results['gamma_positive_significant'].iloc[0] if not self.h2_3_results.empty and 'gamma_positive_significant' in self.h2_3_results else False}
        pd.DataFrame([summary]).to_csv(os.path.join(self.results_dir, f'{self.event_type}_h2_overall_summary.csv'), index=False)
        print(f"Results saved to {self.results_dir}")

def run_hypothesis2_analysis(event_type: str, event_file: str, date_col: str, 
                           ticker_col: str, stock_files: List[str]) -> bool:
    config = Config()
    results_dir = FDA_RESULTS_DIR if event_type == 'fda' else EARNINGS_RESULTS_DIR
    tester = Hypothesis2Tester(config, event_type, results_dir)
    
    stock_df = tester.load_stock_data(stock_files)
    if stock_df is None or stock_df.is_empty():
        print(f"Failed to load stock data for H2 {event_type}."); return False
    
    event_df = tester.load_event_data(event_file, date_col, ticker_col)
    if event_df is None or event_df.is_empty():
        print(f"Failed to load event data for H2 {event_type}."); return False
    
    return tester.run_analysis(stock_df, event_df)

def compare_results():
    print("\n" + "="*60 + "\nComparing FDA and Earnings Results for Hypothesis 2\n" + "="*60)
    fda_summary_path = os.path.join(FDA_RESULTS_DIR, 'fda_h2_overall_summary.csv')
    earnings_summary_path = os.path.join(EARNINGS_RESULTS_DIR, 'earnings_h2_overall_summary.csv')

    if not (os.path.exists(fda_summary_path) and os.path.exists(earnings_summary_path)):
        print("One or both H2 summary files not found. Cannot compare."); return

    fda_summary = pd.read_csv(fda_summary_path)
    earnings_summary = pd.read_csv(earnings_summary_path)
    if fda_summary.empty or earnings_summary.empty:
        print("One or both H2 summary files are empty. Cannot compare."); return

    fig, ax = plt.subplots(figsize=(10, 6))
    tests = ['H2.1\nInnovations→Returns', 'H2.2\nPersistence→Returns', 'H2.3\nAsymmetry (γ>0)']
    x_pos, width = np.arange(len(tests)), 0.35
    fda_supp = [fda_summary['h2_1_supported'].iloc[0], fda_summary['h2_2_supported'].iloc[0], fda_summary['h2_3_supported'].iloc[0]]
    earn_supp = [earnings_summary['h2_1_supported'].iloc[0], earnings_summary['h2_2_supported'].iloc[0], earnings_summary['h2_3_supported'].iloc[0]]
    
    ax.bar(x_pos - width/2, [1 if s else 0 for s in fda_supp], width, label='FDA', color='lightblue', alpha=0.8)
    ax.bar(x_pos + width/2, [1 if s else 0 for s in earn_supp], width, label='Earnings', color='lightgreen', alpha=0.8)
    
    ax.set_ylabel('Hypothesis Supported (1=Yes, 0=No)'); ax.set_title('Hypothesis 2 Support: FDA vs Earnings Events')
    ax.set_xticks(x_pos); ax.set_xticklabels(tests); ax.set_yticks([0,1]); ax.set_yticklabels(['No', 'Yes'])
    ax.legend(); ax.set_ylim(-0.1, 1.2); ax.grid(True, alpha=0.3, axis='y'); plt.tight_layout()
    
    comparison_dir = 'results/hypothesis2/comparison/'
    os.makedirs(comparison_dir, exist_ok=True)
    plt.savefig(os.path.join(comparison_dir, 'h2_comparison_plot.png'), dpi=300, bbox_inches='tight'); plt.close()
    
    print("\nHypothesis 2 Comparison Summary:")
    for i, test_name in enumerate(['H2.1', 'H2.2', 'H2.3']):
        print(f"  {test_name}: FDA: {'Supported' if fda_supp[i] else 'Not Supported'}, Earnings: {'Supported' if earn_supp[i] else 'Not Supported'}")

def main():
    print("Starting Hypothesis 2 Analysis\n" + "="*60)
    fda_success = run_hypothesis2_analysis(
        event_type='fda', event_file=FDA_EVENT_FILE, date_col=FDA_EVENT_DATE_COL,
        ticker_col=FDA_TICKER_COL, stock_files=STOCK_FILES
    )
    earnings_success = run_hypothesis2_analysis(
        event_type='earnings', event_file=EARNINGS_EVENT_FILE, date_col=EARNINGS_EVENT_DATE_COL,
        ticker_col=EARNINGS_TICKER_COL, stock_files=STOCK_FILES
    )
    if fda_success and earnings_success: compare_results()
    else: print("\nH2 Comparison skipped as one or both H2 analyses did not complete successfully.")
    print("\nAnalysis complete!")

if __name__ == "__main__":
    main()
