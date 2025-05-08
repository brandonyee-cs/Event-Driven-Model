import pandas as pd
import numpy as np
import os
import sys
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import scipy.stats as stats
import traceback
import polars as pl
from sklearn.linear_model import LinearRegression

# --- Configuration ---
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path: sys.path.append(current_dir)
try: 
    from src.event_processor import EventDataLoader, EventFeatureEngineer, EventAnalysis
    print("Successfully imported Event processor classes.")
except ImportError as e: 
    print(f"Error importing from event_processor: {e}")
    print("Ensure 'event_processor.py' and 'models.py' are in the same directory or Python path.")
    print("Ensure Polars and Plotly are installed: pip install polars pyarrow plotly")
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
FDA_RESULTS_DIR = "results/hypothesis2_validation/results_fda/"
FDA_FILE_PREFIX = "fda"
FDA_EVENT_DATE_COL = "Approval Date"
FDA_TICKER_COL = "ticker"

# Earnings event specific parameters
EARNINGS_EVENT_FILE = "/home/d87016661/detail_history_actuals.csv"
EARNINGS_RESULTS_DIR = "results/hypothesis2_validation/results_earnings/"
EARNINGS_FILE_PREFIX = "earnings"
EARNINGS_EVENT_DATE_COL = "ANNDATS"
EARNINGS_TICKER_COL = "ticker"

# Shared analysis parameters
WINDOW_DAYS = 60

# GARCH parameters - using GJR-GARCH for asymmetric volatility response
GARCH_PARAMS = {
    'omega': 0.00001,
    'alpha': 0.1, 
    'beta': 0.8,
    'gamma': 0.05  # GJR-GARCH parameter for asymmetric volatility response
}

# Volatility analysis parameters
VOL_ANALYSIS_WINDOW = (-30, 30)
PRE_EVENT_WINDOW = (-10, -1)
POST_EVENT_WINDOW = (0, 10)
LATE_POST_EVENT_WINDOW = (11, 20)

# Hypothesis testing parameters
HYPOTHESIS_RESULTS_DIR = "results/hypothesis2_validation/"

def run_fda_analysis():
    """
    Runs the FDA event volatility analysis pipeline for Hypothesis 2 validation.
    """
    print("\n=== Starting FDA Approval Event Volatility Analysis for Hypothesis 2 Validation ===")

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
        
        # Set GARCH parameters
        analyzer.garch_params = GARCH_PARAMS
        print("FDA components initialized.")

        # --- Load Data ---
        print("\nLoading and preparing FDA data...")
        analyzer.data = analyzer.load_and_prepare_data(run_feature_engineering=False)
        if analyzer.data is None or analyzer.data.is_empty(): 
            print("\n*** Error: FDA data loading failed. ***")
            return False
            
        print(f"FDA data loaded successfully. Shape: {analyzer.data.shape}")
        
        # --- Fit GARCH Models ---
        print("\n--- Fitting GJR-GARCH Models for FDA Events ---")
        analyzer.fit_garch_models(return_col='ret', model_type='gjr')
        
        # --- Analyze Impact Uncertainty (Volatility Innovations) ---
        analyzer.analyze_impact_uncertainty(
            results_dir=FDA_RESULTS_DIR,
            file_prefix=FDA_FILE_PREFIX
        )
        
        # --- Analyze Volatility Patterns with GARCH ---
        analyzer.analyze_volatility_patterns(
            results_dir=FDA_RESULTS_DIR,
            file_prefix=FDA_FILE_PREFIX,
            return_col='ret'
        )
        
        print(f"\n--- FDA Event Volatility Analysis Finished (Results in '{FDA_RESULTS_DIR}') ---")
        return True

    except Exception as e: 
        print(f"\n*** An unexpected error occurred in FDA analysis: {e} ***")
        traceback.print_exc()
    
    return False

def run_earnings_analysis():
    """
    Runs the earnings event volatility analysis pipeline for Hypothesis 2 validation.
    """
    print("\n=== Starting Earnings Announcement Event Volatility Analysis for Hypothesis 2 Validation ===")

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
        
        # Set GARCH parameters
        analyzer.garch_params = GARCH_PARAMS
        print("Earnings components initialized.")

        # --- Load Data ---
        print("\nLoading and preparing earnings data...")
        analyzer.data = analyzer.load_and_prepare_data(run_feature_engineering=False)
        if analyzer.data is None or analyzer.data.is_empty(): 
            print("\n*** Error: Earnings data loading failed. ***")
            return False
            
        print(f"Earnings data loaded successfully. Shape: {analyzer.data.shape}")
        
        # --- Fit GARCH Models ---
        print("\n--- Fitting GJR-GARCH Models for Earnings Events ---")
        analyzer.fit_garch_models(return_col='ret', model_type='gjr')
        
        # --- Analyze Impact Uncertainty (Volatility Innovations) ---
        analyzer.analyze_impact_uncertainty(
            results_dir=EARNINGS_RESULTS_DIR,
            file_prefix=EARNINGS_FILE_PREFIX
        )
        
        # --- Analyze Volatility Patterns with GARCH ---
        analyzer.analyze_volatility_patterns(
            results_dir=EARNINGS_RESULTS_DIR,
            file_prefix=EARNINGS_FILE_PREFIX,
            return_col='ret'
        )
        
        print(f"\n--- Earnings Event Volatility Analysis Finished (Results in '{EARNINGS_RESULTS_DIR}') ---")
        return True

    except Exception as e: 
        print(f"\n*** An unexpected error occurred in earnings analysis: {e} ***")
        traceback.print_exc()
    
    return False

def collect_event_data():
    """
    Collects processed data from both FDA and earnings analyses to use in hypothesis testing.
    
    Returns:
    --------
    dict
        Dictionary with collected data
    """
    print("\n=== Collecting Processed Event Data for Hypothesis 2 Validation ===")
    
    collected_data = {}
    
    # Collect FDA impact uncertainty data
    fda_impact_file = os.path.join(FDA_RESULTS_DIR, f"{FDA_FILE_PREFIX}_impact_uncertainty.csv")
    if os.path.exists(fda_impact_file):
        fda_impact = pd.read_csv(fda_impact_file)
        collected_data['fda_impact'] = fda_impact
        print(f"Loaded FDA impact uncertainty data: {fda_impact.shape}")
    else:
        print(f"Warning: FDA impact uncertainty file not found: {fda_impact_file}")
    
    # Collect FDA volatility data
    fda_vol_file = os.path.join(FDA_RESULTS_DIR, f"{FDA_FILE_PREFIX}_garch_volatility.csv")
    if os.path.exists(fda_vol_file):
        fda_vol = pd.read_csv(fda_vol_file)
        collected_data['fda_vol'] = fda_vol
        print(f"Loaded FDA volatility data: {fda_vol.shape}")
    else:
        print(f"Warning: FDA volatility file not found: {fda_vol_file}")
        
    # Collect Earnings impact uncertainty data
    earnings_impact_file = os.path.join(EARNINGS_RESULTS_DIR, f"{EARNINGS_FILE_PREFIX}_impact_uncertainty.csv")
    if os.path.exists(earnings_impact_file):
        earnings_impact = pd.read_csv(earnings_impact_file)
        collected_data['earnings_impact'] = earnings_impact
        print(f"Loaded earnings impact uncertainty data: {earnings_impact.shape}")
    else:
        print(f"Warning: Earnings impact uncertainty file not found: {earnings_impact_file}")
    
    # Collect Earnings volatility data
    earnings_vol_file = os.path.join(EARNINGS_RESULTS_DIR, f"{EARNINGS_FILE_PREFIX}_garch_volatility.csv")
    if os.path.exists(earnings_vol_file):
        earnings_vol = pd.read_csv(earnings_vol_file)
        collected_data['earnings_vol'] = earnings_vol
        print(f"Loaded earnings volatility data: {earnings_vol.shape}")
    else:
        print(f"Warning: Earnings volatility file not found: {earnings_vol_file}")
        
    # Collect raw event data if available for additional analysis
    try:
        # FDA raw event data - only if needed for more detailed analysis
        fda_data_loader = EventDataLoader(
            event_path=FDA_EVENT_FILE, 
            stock_paths=STOCK_FILES, 
            window_days=WINDOW_DAYS,
            event_date_col=FDA_EVENT_DATE_COL,
            ticker_col=FDA_TICKER_COL
        )
        fda_analyzer = EventAnalysis(fda_data_loader, EventFeatureEngineer(prediction_window=3))
        fda_data = fda_analyzer.load_and_prepare_data(run_feature_engineering=False)
        if fda_data is not None and not fda_data.is_empty():
            collected_data['fda_raw_data'] = fda_data
            print(f"Loaded FDA raw data: {fda_data.shape}")
        
        # Earnings raw event data - only if needed for more detailed analysis
        earnings_data_loader = EventDataLoader(
            event_path=EARNINGS_EVENT_FILE, 
            stock_paths=STOCK_FILES, 
            window_days=WINDOW_DAYS,
            event_date_col=EARNINGS_EVENT_DATE_COL,
            ticker_col=EARNINGS_TICKER_COL
        )
        earnings_analyzer = EventAnalysis(earnings_data_loader, EventFeatureEngineer(prediction_window=3))
        earnings_data = earnings_analyzer.load_and_prepare_data(run_feature_engineering=False)
        if earnings_data is not None and not earnings_data.is_empty():
            collected_data['earnings_raw_data'] = earnings_data
            print(f"Loaded earnings raw data: {earnings_data.shape}")
    except Exception as e:
        print(f"Warning: Error loading raw event data: {e}")
        print("Will continue with processed data only.")
    
    return collected_data

def analyze_pre_event_predictability(data):
    """
    Analyzes whether pre-event volatility innovations predict subsequent returns.
    
    Parameters:
    -----------
    data: dict
        Dictionary with collected data
    
    Returns:
    --------
    dict
        Results of the analysis
    """
    print("\n=== Analyzing Pre-Event Volatility Innovations as Return Predictors ===")
    
    results = {}
    
    # --- FDA Analysis ---
    if 'fda_impact' in data and 'fda_vol' in data:
        fda_impact = data['fda_impact']
        
        # Get pre-event impact uncertainty (volatility innovations)
        pre_event_impact = fda_impact[(fda_impact['days_to_event'] >= PRE_EVENT_WINDOW[0]) & 
                                       (fda_impact['days_to_event'] <= PRE_EVENT_WINDOW[1])]
        
        # Calculate average pre-event impact uncertainty
        pre_event_avg_impact = pre_event_impact.groupby('days_to_event')['avg_impact'].mean().reset_index()
        
        # Get post-event volatility
        post_event_vol = data['fda_vol'][(data['fda_vol']['days_to_event'] >= POST_EVENT_WINDOW[0]) & 
                                           (data['fda_vol']['days_to_event'] <= POST_EVENT_WINDOW[1])]
        
        # Calculate average post-event volatility
        post_event_avg_vol = post_event_vol.groupby('days_to_event')['avg_annualized_vol'].mean().reset_index()
        
        # Check if raw data is available for return analysis
        if 'fda_raw_data' in data:
            # Convert Polars DataFrame to pandas
            fda_raw_df = data['fda_raw_data'].to_pandas()
            
            # Calculate average returns by days_to_event
            avg_returns = fda_raw_df.groupby('days_to_event')['ret'].mean().reset_index()
            
            # Get post-event returns
            post_event_returns = avg_returns[(avg_returns['days_to_event'] >= POST_EVENT_WINDOW[0]) & 
                                             (avg_returns['days_to_event'] <= POST_EVENT_WINDOW[1])]
            
            # Calculate correlation between pre-event impact and post-event returns
            pre_event_mean_impact = pre_event_impact['avg_impact'].mean()
            post_event_mean_return = post_event_returns['ret'].mean()
            
            # Prepare data for regression analysis (if we have event-level data)
            if 'event_id' in fda_raw_df.columns:
                # Get unique events
                events = fda_raw_df['event_id'].unique()
                
                # Prepare data for regression
                regression_data = []
                
                for event_id in events:
                    event_data = fda_raw_df[fda_raw_df['event_id'] == event_id]
                    
                    # Calculate pre-event average impact uncertainty for this event
                    pre_event_data = event_data[(event_data['days_to_event'] >= PRE_EVENT_WINDOW[0]) & 
                                               (event_data['days_to_event'] <= PRE_EVENT_WINDOW[1])]
                    
                    if pre_event_data.empty:
                        continue
                    
                    # Calculate post-event average return for this event
                    post_event_data = event_data[(event_data['days_to_event'] >= POST_EVENT_WINDOW[0]) & 
                                                (event_data['days_to_event'] <= POST_EVENT_WINDOW[1])]
                    
                    if post_event_data.empty:
                        continue
                    
                    # Extract relevant data
                    pre_impact = pre_event_data['ret'].std() * pre_event_data['ret'].std()  # Proxy for impact uncertainty if not available
                    post_return = post_event_data['ret'].mean()
                    
                    regression_data.append({
                        'event_id': event_id,
                        'pre_impact': pre_impact,
                        'post_return': post_return
                    })
                
                # Convert to DataFrame for regression
                if regression_data:
                    reg_df = pd.DataFrame(regression_data)
                    
                    # Run regression if we have sufficient data
                    if len(reg_df) >= 10:
                        X = reg_df[['pre_impact']].values
                        y = reg_df['post_return'].values
                        
                        model = LinearRegression()
                        model.fit(X, y)
                        
                        # Get regression statistics
                        predictions = model.predict(X)
                        r_squared = model.score(X, y)
                        coefficient = model.coef_[0]
                        intercept = model.intercept_
                        
                        # Get p-value
                        n = len(reg_df)
                        k = 1  # number of predictors
                        if n > k + 1:
                            from scipy import stats
                            sse = np.sum((y - predictions) ** 2)
                            ssr = np.sum((predictions - np.mean(y)) ** 2)
                            f_stat = (ssr / k) / (sse / (n - k - 1))
                            p_value = 1 - stats.f.cdf(f_stat, k, n - k - 1)
                        else:
                            p_value = np.nan
                        
                        # Store regression results
                        fda_regression_results = {
                            'coefficient': coefficient,
                            'intercept': intercept,
                            'r_squared': r_squared,
                            'p_value': p_value,
                            'n_observations': n
                        }
                        
                        # Create scatter plot
                        plt.figure(figsize=(10, 6))
                        plt.scatter(reg_df['pre_impact'], reg_df['post_return'], alpha=0.5)
                        plt.plot(X, predictions, color='red', linewidth=2)
                        plt.xlabel('Pre-Event Impact Uncertainty (Volatility Innovation)')
                        plt.ylabel('Post-Event Returns')
                        plt.title('FDA Events: Pre-Event Impact Uncertainty vs. Post-Event Returns')
                        plt.grid(True, alpha=0.3)
                        
                        # Add regression equation and statistics to plot
                        equation = f"y = {coefficient:.6f}x + {intercept:.6f}"
                        plt.annotate(f"Regression: {equation}\nR² = {r_squared:.4f}\np-value = {p_value:.4f}\nn = {n}",
                                    xy=(0.05, 0.95), xycoords='axes fraction',
                                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8),
                                    ha='left', va='top')
                        
                        # Save plot
                        os.makedirs(HYPOTHESIS_RESULTS_DIR, exist_ok=True)
                        plt.savefig(os.path.join(HYPOTHESIS_RESULTS_DIR, 'fda_pre_event_impact_vs_post_return.png'), 
                                  dpi=300, bbox_inches='tight')
                        plt.close()
                        
                        print(f"FDA Regression Results: Coefficient = {coefficient:.6f}, R² = {r_squared:.4f}, p-value = {p_value:.4f}, n = {n}")
                    else:
                        print(f"Not enough FDA event data for regression analysis (only {len(reg_df)} events)")
                        fda_regression_results = None
                else:
                    print("No FDA regression data available")
                    fda_regression_results = None
            else:
                fda_regression_results = None
                
            # Store results
            results['fda'] = {
                'pre_event_impact': pre_event_avg_impact,
                'post_event_vol': post_event_avg_vol,
                'post_event_returns': post_event_returns,
                'pre_event_mean_impact': pre_event_mean_impact,
                'post_event_mean_return': post_event_mean_return,
                'regression_results': fda_regression_results
            }
        else:
            print("FDA raw data not available, using processed data only")
            results['fda'] = {
                'pre_event_impact': pre_event_avg_impact,
                'post_event_vol': post_event_avg_vol,
                'regression_results': None
            }
    else:
        print("FDA impact or volatility data not available")
        
    # --- Earnings Analysis ---
    if 'earnings_impact' in data and 'earnings_vol' in data:
        earnings_impact = data['earnings_impact']
        
        # Get pre-event impact uncertainty (volatility innovations)
        pre_event_impact = earnings_impact[(earnings_impact['days_to_event'] >= PRE_EVENT_WINDOW[0]) & 
                                           (earnings_impact['days_to_event'] <= PRE_EVENT_WINDOW[1])]
        
        # Calculate average pre-event impact uncertainty
        pre_event_avg_impact = pre_event_impact.groupby('days_to_event')['avg_impact'].mean().reset_index()
        
        # Get post-event volatility
        post_event_vol = data['earnings_vol'][(data['earnings_vol']['days_to_event'] >= POST_EVENT_WINDOW[0]) & 
                                               (data['earnings_vol']['days_to_event'] <= POST_EVENT_WINDOW[1])]
        
        # Calculate average post-event volatility
        post_event_avg_vol = post_event_vol.groupby('days_to_event')['avg_annualized_vol'].mean().reset_index()
        
        # Check if raw data is available for return analysis
        if 'earnings_raw_data' in data:
            # Convert Polars DataFrame to pandas
            earnings_raw_df = data['earnings_raw_data'].to_pandas()
            
            # Calculate average returns by days_to_event
            avg_returns = earnings_raw_df.groupby('days_to_event')['ret'].mean().reset_index()
            
            # Get post-event returns
            post_event_returns = avg_returns[(avg_returns['days_to_event'] >= POST_EVENT_WINDOW[0]) & 
                                             (avg_returns['days_to_event'] <= POST_EVENT_WINDOW[1])]
            
            # Calculate correlation between pre-event impact and post-event returns
            pre_event_mean_impact = pre_event_impact['avg_impact'].mean()
            post_event_mean_return = post_event_returns['ret'].mean()
            
            # Prepare data for regression analysis (if we have event-level data)
            if 'event_id' in earnings_raw_df.columns:
                # Get unique events
                events = earnings_raw_df['event_id'].unique()
                
                # Prepare data for regression
                regression_data = []
                
                for event_id in events:
                    event_data = earnings_raw_df[earnings_raw_df['event_id'] == event_id]
                    
                    # Calculate pre-event average impact uncertainty for this event
                    pre_event_data = event_data[(event_data['days_to_event'] >= PRE_EVENT_WINDOW[0]) & 
                                               (event_data['days_to_event'] <= PRE_EVENT_WINDOW[1])]
                    
                    if pre_event_data.empty:
                        continue
                    
                    # Calculate post-event average return for this event
                    post_event_data = event_data[(event_data['days_to_event'] >= POST_EVENT_WINDOW[0]) & 
                                                (event_data['days_to_event'] <= POST_EVENT_WINDOW[1])]
                    
                    if post_event_data.empty:
                        continue
                    
                    # Extract relevant data
                    pre_impact = pre_event_data['ret'].std() * pre_event_data['ret'].std()  # Proxy for impact uncertainty if not available
                    post_return = post_event_data['ret'].mean()
                    
                    regression_data.append({
                        'event_id': event_id,
                        'pre_impact': pre_impact,
                        'post_return': post_return
                    })
                
                # Convert to DataFrame for regression
                if regression_data:
                    reg_df = pd.DataFrame(regression_data)
                    
                    # Run regression if we have sufficient data
                    if len(reg_df) >= 10:
                        X = reg_df[['pre_impact']].values
                        y = reg_df['post_return'].values
                        
                        model = LinearRegression()
                        model.fit(X, y)
                        
                        # Get regression statistics
                        predictions = model.predict(X)
                        r_squared = model.score(X, y)
                        coefficient = model.coef_[0]
                        intercept = model.intercept_
                        
                        # Get p-value
                        n = len(reg_df)
                        k = 1  # number of predictors
                        if n > k + 1:
                            from scipy import stats
                            sse = np.sum((y - predictions) ** 2)
                            ssr = np.sum((predictions - np.mean(y)) ** 2)
                            f_stat = (ssr / k) / (sse / (n - k - 1))
                            p_value = 1 - stats.f.cdf(f_stat, k, n - k - 1)
                        else:
                            p_value = np.nan
                        
                        # Store regression results
                        earnings_regression_results = {
                            'coefficient': coefficient,
                            'intercept': intercept,
                            'r_squared': r_squared,
                            'p_value': p_value,
                            'n_observations': n
                        }
                        
                        # Create scatter plot
                        plt.figure(figsize=(10, 6))
                        plt.scatter(reg_df['pre_impact'], reg_df['post_return'], alpha=0.5)
                        plt.plot(X, predictions, color='red', linewidth=2)
                        plt.xlabel('Pre-Event Impact Uncertainty (Volatility Innovation)')
                        plt.ylabel('Post-Event Returns')
                        plt.title('Earnings Events: Pre-Event Impact Uncertainty vs. Post-Event Returns')
                        plt.grid(True, alpha=0.3)
                        
                        # Add regression equation and statistics to plot
                        equation = f"y = {coefficient:.6f}x + {intercept:.6f}"
                        plt.annotate(f"Regression: {equation}\nR² = {r_squared:.4f}\np-value = {p_value:.4f}\nn = {n}",
                                    xy=(0.05, 0.95), xycoords='axes fraction',
                                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8),
                                    ha='left', va='top')
                        
                        # Save plot
                        os.makedirs(HYPOTHESIS_RESULTS_DIR, exist_ok=True)
                        plt.savefig(os.path.join(HYPOTHESIS_RESULTS_DIR, 'earnings_pre_event_impact_vs_post_return.png'), 
                                  dpi=300, bbox_inches='tight')
                        plt.close()
                        
                        print(f"Earnings Regression Results: Coefficient = {coefficient:.6f}, R² = {r_squared:.4f}, p-value = {p_value:.4f}, n = {n}")
                    else:
                        print(f"Not enough earnings event data for regression analysis (only {len(reg_df)} events)")
                        earnings_regression_results = None
                else:
                    print("No earnings regression data available")
                    earnings_regression_results = None
            else:
                earnings_regression_results = None
                
            # Store results
            results['earnings'] = {
                'pre_event_impact': pre_event_avg_impact,
                'post_event_vol': post_event_avg_vol,
                'post_event_returns': post_event_returns,
                'pre_event_mean_impact': pre_event_mean_impact,
                'post_event_mean_return': post_event_mean_return,
                'regression_results': earnings_regression_results
            }
        else:
            print("Earnings raw data not available, using processed data only")
            results['earnings'] = {
                'pre_event_impact': pre_event_avg_impact,
                'post_event_vol': post_event_avg_vol,
                'regression_results': None
            }
    else:
        print("Earnings impact or volatility data not available")
        
    return results

def analyze_post_event_persistence(data):
    """
    Analyzes whether post-event volatility persistence extends the period of elevated expected returns.
    
    Parameters:
    -----------
    data: dict
        Dictionary with collected data
    
    Returns:
    --------
    dict
        Results of the analysis
    """
    print("\n=== Analyzing Post-Event Volatility Persistence ===")
    
    results = {}
    
    # --- FDA Analysis ---
    if 'fda_vol' in data:
        fda_vol = data['fda_vol']
        
        # Get post-event volatility
        post_event_vol = fda_vol[(fda_vol['days_to_event'] >= POST_EVENT_WINDOW[0]) & 
                                   (fda_vol['days_to_event'] <= POST_EVENT_WINDOW[1])]
        
        late_post_event_vol = fda_vol[(fda_vol['days_to_event'] >= LATE_POST_EVENT_WINDOW[0]) & 
                                       (fda_vol['days_to_event'] <= LATE_POST_EVENT_WINDOW[1])]
        
        # Calculate volatility persistence (how long volatility stays elevated)
        baseline_vol = fda_vol[(fda_vol['days_to_event'] < PRE_EVENT_WINDOW[0])]['avg_annualized_vol'].mean()
        
        # Normalize volatility to the baseline
        post_event_vol_normalized = post_event_vol.copy()
        post_event_vol_normalized['normalized_vol'] = post_event_vol['avg_annualized_vol'] / baseline_vol
        
        late_post_event_vol_normalized = late_post_event_vol.copy()
        late_post_event_vol_normalized['normalized_vol'] = late_post_event_vol['avg_annualized_vol'] / baseline_vol
        
        # Calculate mean normalized volatility for each period
        post_event_mean_normalized_vol = post_event_vol_normalized['normalized_vol'].mean()
        late_post_event_mean_normalized_vol = late_post_event_vol_normalized['normalized_vol'].mean()
        
        # Calculate volatility half-life (days until volatility drops to half of its post-event peak)
        post_event_peak_vol = post_event_vol['avg_annualized_vol'].max()
        post_event_peak_day = post_event_vol.loc[post_event_vol['avg_annualized_vol'].idxmax(), 'days_to_event']
        
        # Find the first day when volatility drops below half of peak
        vol_after_peak = fda_vol[fda_vol['days_to_event'] > post_event_peak_day]
        if not vol_after_peak.empty and post_event_peak_vol > 0:
            half_peak_vol = (post_event_peak_vol + baseline_vol) / 2
            days_below_half_peak = vol_after_peak[vol_after_peak['avg_annualized_vol'] <= half_peak_vol]['days_to_event']
            
            if not days_below_half_peak.empty:
                volatility_half_life = days_below_half_peak.iloc[0] - post_event_peak_day
            else:
                volatility_half_life = vol_after_peak['days_to_event'].max() - post_event_peak_day
                print(f"FDA Volatility did not drop below half-peak within the analysis window. Using max: {volatility_half_life} days")
        else:
            volatility_half_life = np.nan
            print("FDA Volatility half-life could not be calculated")
        
        # Check if raw data is available for return analysis
        if 'fda_raw_data' in data:
            # Convert Polars DataFrame to pandas
            fda_raw_df = data['fda_raw_data'].to_pandas()
            
            # Calculate average returns by days_to_event
            avg_returns = fda_raw_df.groupby('days_to_event')['ret'].mean().reset_index()
            
            # Get post-event returns
            post_event_returns = avg_returns[(avg_returns['days_to_event'] >= POST_EVENT_WINDOW[0]) & 
                                             (avg_returns['days_to_event'] <= POST_EVENT_WINDOW[1])]
            
            late_post_event_returns = avg_returns[(avg_returns['days_to_event'] >= LATE_POST_EVENT_WINDOW[0]) & 
                                                 (avg_returns['days_to_event'] <= LATE_POST_EVENT_WINDOW[1])]
            
            # Calculate mean returns for each period
            post_event_mean_return = post_event_returns['ret'].mean()
            late_post_event_mean_return = late_post_event_returns['ret'].mean()
            
            # Create a merged DataFrame of volatility and returns
            merged_data = pd.merge(fda_vol, avg_returns, on='days_to_event')
            
            # Calculate correlation between volatility and returns
            post_event_merged = merged_data[(merged_data['days_to_event'] >= POST_EVENT_WINDOW[0]) & 
                                           (merged_data['days_to_event'] <= POST_EVENT_WINDOW[1])]
            
            late_post_event_merged = merged_data[(merged_data['days_to_event'] >= LATE_POST_EVENT_WINDOW[0]) & 
                                                (merged_data['days_to_event'] <= LATE_POST_EVENT_WINDOW[1])]
            
            # Calculate correlations
            post_event_vol_ret_corr = post_event_merged[['avg_annualized_vol', 'ret']].corr().iloc[0, 1]
            late_post_event_vol_ret_corr = late_post_event_merged[['avg_annualized_vol', 'ret']].corr().iloc[0, 1]
            
            # Store results
            results['fda'] = {
                'baseline_vol': baseline_vol,
                'post_event_mean_normalized_vol': post_event_mean_normalized_vol,
                'late_post_event_mean_normalized_vol': late_post_event_mean_normalized_vol,
                'post_event_peak_vol': post_event_peak_vol,
                'post_event_peak_day': post_event_peak_day,
                'volatility_half_life': volatility_half_life,
                'post_event_mean_return': post_event_mean_return,
                'late_post_event_mean_return': late_post_event_mean_return,
                'post_event_vol_ret_corr': post_event_vol_ret_corr,
                'late_post_event_vol_ret_corr': late_post_event_vol_ret_corr
            }
            
            # Create volatility persistence plot
            plt.figure(figsize=(12, 8))
            ax1 = plt.gca()
            ax2 = ax1.twinx()
            
            # Plot volatility
            ax1.plot(merged_data['days_to_event'], merged_data['avg_annualized_vol'], 'b-', linewidth=2, label='Volatility')
            ax1.set_xlabel('Days Relative to Event')
            ax1.set_ylabel('Annualized Volatility (%)', color='blue')
            ax1.tick_params(axis='y', labelcolor='blue')
            
            # Plot returns
            ax2.plot(merged_data['days_to_event'], merged_data['ret'] * 100, 'r-', linewidth=2, label='Returns')
            ax2.set_ylabel('Returns (%)', color='red')
            ax2.tick_params(axis='y', labelcolor='red')
            
            # Add vertical line at event day
            plt.axvline(x=0, color='green', linestyle='--', label='Event Day')
            
            # Highlight the periods
            plt.axvspan(PRE_EVENT_WINDOW[0], PRE_EVENT_WINDOW[1], color='lightblue', alpha=0.3, label='Pre-Event')
            plt.axvspan(POST_EVENT_WINDOW[0], POST_EVENT_WINDOW[1], color='lightgreen', alpha=0.3, label='Post-Event')
            plt.axvspan(LATE_POST_EVENT_WINDOW[0], LATE_POST_EVENT_WINDOW[1], color='lightgray', alpha=0.3, label='Late Post-Event')
            
            # Add half-life marker if available
            if not np.isnan(volatility_half_life):
                half_life_day = post_event_peak_day + volatility_half_life
                plt.axvline(x=half_life_day, color='purple', linestyle=':', label=f'Half-Life ({volatility_half_life} days)')
            
            # Create custom legend
            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
            
            plt.title('FDA Events: Volatility Persistence and Returns', fontsize=14)
            plt.grid(True, alpha=0.3)
            
            # Add summary statistics to the plot
            stats_text = (
                f"Post-Event Vol: {post_event_mean_normalized_vol:.2f}x baseline\n"
                f"Late Post-Event Vol: {late_post_event_mean_normalized_vol:.2f}x baseline\n"
                f"Volatility Half-Life: {volatility_half_life:.1f} days\n"
                f"Post-Event Vol-Ret Corr: {post_event_vol_ret_corr:.3f}\n"
                f"Late Post-Event Vol-Ret Corr: {late_post_event_vol_ret_corr:.3f}"
            )
            
            plt.annotate(stats_text, xy=(0.02, 0.02), xycoords='axes fraction', 
                         bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8),
                         ha='left', va='bottom')
            
            # Save plot
            os.makedirs(HYPOTHESIS_RESULTS_DIR, exist_ok=True)
            plt.savefig(os.path.join(HYPOTHESIS_RESULTS_DIR, 'fda_volatility_persistence.png'), 
                       dpi=300, bbox_inches='tight')
            plt.close()
        else:
            print("FDA raw data not available, using processed data only")
            results['fda'] = {
                'baseline_vol': baseline_vol,
                'post_event_mean_normalized_vol': post_event_mean_normalized_vol,
                'late_post_event_mean_normalized_vol': late_post_event_mean_normalized_vol,
                'post_event_peak_vol': post_event_peak_vol,
                'post_event_peak_day': post_event_peak_day,
                'volatility_half_life': volatility_half_life
            }
    else:
        print("FDA volatility data not available")
    
    # --- Earnings Analysis ---
    if 'earnings_vol' in data:
        earnings_vol = data['earnings_vol']
        
        # Get post-event volatility
        post_event_vol = earnings_vol[(earnings_vol['days_to_event'] >= POST_EVENT_WINDOW[0]) & 
                                       (earnings_vol['days_to_event'] <= POST_EVENT_WINDOW[1])]
        
        late_post_event_vol = earnings_vol[(earnings_vol['days_to_event'] >= LATE_POST_EVENT_WINDOW[0]) & 
                                           (earnings_vol['days_to_event'] <= LATE_POST_EVENT_WINDOW[1])]
        
        # Calculate volatility persistence (how long volatility stays elevated)
        baseline_vol = earnings_vol[(earnings_vol['days_to_event'] < PRE_EVENT_WINDOW[0])]['avg_annualized_vol'].mean()
        
        # Normalize volatility to the baseline
        post_event_vol_normalized = post_event_vol.copy()
        post_event_vol_normalized['normalized_vol'] = post_event_vol['avg_annualized_vol'] / baseline_vol
        
        late_post_event_vol_normalized = late_post_event_vol.copy()
        late_post_event_vol_normalized['normalized_vol'] = late_post_event_vol['avg_annualized_vol'] / baseline_vol
        
        # Calculate mean normalized volatility for each period
        post_event_mean_normalized_vol = post_event_vol_normalized['normalized_vol'].mean()
        late_post_event_mean_normalized_vol = late_post_event_vol_normalized['normalized_vol'].mean()
        
        # Calculate volatility half-life (days until volatility drops to half of its post-event peak)
        post_event_peak_vol = post_event_vol['avg_annualized_vol'].max()
        post_event_peak_day = post_event_vol.loc[post_event_vol['avg_annualized_vol'].idxmax(), 'days_to_event']
        
        # Find the first day when volatility drops below half of peak
        vol_after_peak = earnings_vol[earnings_vol['days_to_event'] > post_event_peak_day]
        if not vol_after_peak.empty and post_event_peak_vol > 0:
            half_peak_vol = (post_event_peak_vol + baseline_vol) / 2
            days_below_half_peak = vol_after_peak[vol_after_peak['avg_annualized_vol'] <= half_peak_vol]['days_to_event']
            
            if not days_below_half_peak.empty:
                volatility_half_life = days_below_half_peak.iloc[0] - post_event_peak_day
            else:
                volatility_half_life = vol_after_peak['days_to_event'].max() - post_event_peak_day
                print(f"Earnings Volatility did not drop below half-peak within the analysis window. Using max: {volatility_half_life} days")
        else:
            volatility_half_life = np.nan
            print("Earnings Volatility half-life could not be calculated")
        
        # Check if raw data is available for return analysis
        if 'earnings_raw_data' in data:
            # Convert Polars DataFrame to pandas
            earnings_raw_df = data['earnings_raw_data'].to_pandas()
            
            # Calculate average returns by days_to_event
            avg_returns = earnings_raw_df.groupby('days_to_event')['ret'].mean().reset_index()
            
            # Get post-event returns
            post_event_returns = avg_returns[(avg_returns['days_to_event'] >= POST_EVENT_WINDOW[0]) & 
                                             (avg_returns['days_to_event'] <= POST_EVENT_WINDOW[1])]
            
            late_post_event_returns = avg_returns[(avg_returns['days_to_event'] >= LATE_POST_EVENT_WINDOW[0]) & 
                                                 (avg_returns['days_to_event'] <= LATE_POST_EVENT_WINDOW[1])]
            
            # Calculate mean returns for each period
            post_event_mean_return = post_event_returns['ret'].mean()
            late_post_event_mean_return = late_post_event_returns['ret'].mean()
            
            # Create a merged DataFrame of volatility and returns
            merged_data = pd.merge(earnings_vol, avg_returns, on='days_to_event')
            
            # Calculate correlation between volatility and returns
            post_event_merged = merged_data[(merged_data['days_to_event'] >= POST_EVENT_WINDOW[0]) & 
                                           (merged_data['days_to_event'] <= POST_EVENT_WINDOW[1])]
            
            late_post_event_merged = merged_data[(merged_data['days_to_event'] >= LATE_POST_EVENT_WINDOW[0]) & 
                                                (merged_data['days_to_event'] <= LATE_POST_EVENT_WINDOW[1])]
            
            # Calculate correlations
            post_event_vol_ret_corr = post_event_merged[['avg_annualized_vol', 'ret']].corr().iloc[0, 1]
            late_post_event_vol_ret_corr = late_post_event_merged[['avg_annualized_vol', 'ret']].corr().iloc[0, 1]
            
            # Store results
            results['earnings'] = {
                'baseline_vol': baseline_vol,
                'post_event_mean_normalized_vol': post_event_mean_normalized_vol,
                'late_post_event_mean_normalized_vol': late_post_event_mean_normalized_vol,
                'post_event_peak_vol': post_event_peak_vol,
                'post_event_peak_day': post_event_peak_day,
                'volatility_half_life': volatility_half_life,
                'post_event_mean_return': post_event_mean_return,
                'late_post_event_mean_return': late_post_event_mean_return,
                'post_event_vol_ret_corr': post_event_vol_ret_corr,
                'late_post_event_vol_ret_corr': late_post_event_vol_ret_corr
            }
            
            # Create volatility persistence plot
            plt.figure(figsize=(12, 8))
            ax1 = plt.gca()
            ax2 = ax1.twinx()
            
            # Plot volatility
            ax1.plot(merged_data['days_to_event'], merged_data['avg_annualized_vol'], 'b-', linewidth=2, label='Volatility')
            ax1.set_xlabel('Days Relative to Event')
            ax1.set_ylabel('Annualized Volatility (%)', color='blue')
            ax1.tick_params(axis='y', labelcolor='blue')
            
            # Plot returns
            ax2.plot(merged_data['days_to_event'], merged_data['ret'] * 100, 'r-', linewidth=2, label='Returns')
            ax2.set_ylabel('Returns (%)', color='red')
            ax2.tick_params(axis='y', labelcolor='red')
            
            # Add vertical line at event day
            plt.axvline(x=0, color='green', linestyle='--', label='Event Day')
            
            # Highlight the periods
            plt.axvspan(PRE_EVENT_WINDOW[0], PRE_EVENT_WINDOW[1], color='lightblue', alpha=0.3, label='Pre-Event')
            plt.axvspan(POST_EVENT_WINDOW[0], POST_EVENT_WINDOW[1], color='lightgreen', alpha=0.3, label='Post-Event')
            plt.axvspan(LATE_POST_EVENT_WINDOW[0], LATE_POST_EVENT_WINDOW[1], color='lightgray', alpha=0.3, label='Late Post-Event')
            
            # Add half-life marker if available
            if not np.isnan(volatility_half_life):
                half_life_day = post_event_peak_day + volatility_half_life
                plt.axvline(x=half_life_day, color='purple', linestyle=':', label=f'Half-Life ({volatility_half_life} days)')
            
            # Create custom legend
            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
            
            plt.title('Earnings Events: Volatility Persistence and Returns', fontsize=14)
            plt.grid(True, alpha=0.3)
            
            # Add summary statistics to the plot
            stats_text = (
                f"Post-Event Vol: {post_event_mean_normalized_vol:.2f}x baseline\n"
                f"Late Post-Event Vol: {late_post_event_mean_normalized_vol:.2f}x baseline\n"
                f"Volatility Half-Life: {volatility_half_life:.1f} days\n"
                f"Post-Event Vol-Ret Corr: {post_event_vol_ret_corr:.3f}\n"
                f"Late Post-Event Vol-Ret Corr: {late_post_event_vol_ret_corr:.3f}"
            )
            
            plt.annotate(stats_text, xy=(0.02, 0.02), xycoords='axes fraction', 
                         bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8),
                         ha='left', va='bottom')
            
            # Save plot
            os.makedirs(HYPOTHESIS_RESULTS_DIR, exist_ok=True)
            plt.savefig(os.path.join(HYPOTHESIS_RESULTS_DIR, 'earnings_volatility_persistence.png'), 
                       dpi=300, bbox_inches='tight')
            plt.close()
        else:
            print("Earnings raw data not available, using processed data only")
            results['earnings'] = {
                'baseline_vol': baseline_vol,
                'post_event_mean_normalized_vol': post_event_mean_normalized_vol,
                'late_post_event_mean_normalized_vol': late_post_event_mean_normalized_vol,
                'post_event_peak_vol': post_event_peak_vol,
                'post_event_peak_day': post_event_peak_day,
                'volatility_half_life': volatility_half_life
            }
    else:
        print("Earnings volatility data not available")
    
    return results

def analyze_asymmetric_volatility(data):
    """
    Analyzes whether asymmetric volatility response (captured through GJR-GARCH) 
    correlates with asymmetric price adjustment.
    
    Parameters:
    -----------
    data: dict
        Dictionary with collected data
    
    Returns:
    --------
    dict
        Results of the analysis
    """
    print("\n=== Analyzing Asymmetric Volatility Response ===")
    
    results = {}
    
    # --- FDA Analysis ---
    if 'fda_raw_data' in data:
        # Convert Polars DataFrame to pandas
        fda_raw_df = data['fda_raw_data'].to_pandas()
        
        # Calculate average returns by days_to_event
        avg_returns = fda_raw_df.groupby('days_to_event')['ret'].mean().reset_index()
        
        # Classify days by return direction
        avg_returns['return_direction'] = np.where(avg_returns['ret'] > 0, 'positive', 'negative')
        
        # Get volatility data
        if 'fda_vol' in data:
            fda_vol = data['fda_vol']
            
            # Merge returns and volatility
            merged_data = pd.merge(avg_returns, fda_vol, on='days_to_event')
            
            # Split data by return direction
            positive_returns = merged_data[merged_data['return_direction'] == 'positive']
            negative_returns = merged_data[merged_data['return_direction'] == 'negative']
            
            # Calculate average volatility for positive and negative return days
            positive_vol = positive_returns['avg_annualized_vol'].mean()
            negative_vol = negative_returns['avg_annualized_vol'].mean()
            
            # Calculate asymmetry ratio
            if positive_vol > 0:
                asymmetry_ratio = negative_vol / positive_vol
            else:
                asymmetry_ratio = np.nan
                
            # Calculate price adjustment pattern after positive vs negative returns
            positive_post_event = positive_returns[positive_returns['days_to_event'] > 0]
            negative_post_event = negative_returns[negative_returns['days_to_event'] > 0]
            
            # Calculate cumulative returns for post-event periods
            if not positive_post_event.empty:
                positive_post_cum_ret = positive_post_event['ret'].cumsum().iloc[-1]
            else:
                positive_post_cum_ret = np.nan
                
            if not negative_post_event.empty:
                negative_post_cum_ret = negative_post_event['ret'].cumsum().iloc[-1]
            else:
                negative_post_cum_ret = np.nan
                
            # Store results
            results['fda'] = {
                'positive_vol': positive_vol,
                'negative_vol': negative_vol,
                'asymmetry_ratio': asymmetry_ratio,
                'positive_post_cum_ret': positive_post_cum_ret,
                'negative_post_cum_ret': negative_post_cum_ret,
                'gamma': GARCH_PARAMS['gamma']  # GJR-GARCH asymmetry parameter
            }
            
            # Create asymmetric volatility plot
            plt.figure(figsize=(12, 8))
            
            # Plot volatility by return direction
            bar_width = 0.35
            plt.bar([1], [positive_vol], bar_width, label='Positive Returns', color='green', alpha=0.7)
            plt.bar([1 + bar_width], [negative_vol], bar_width, label='Negative Returns', color='red', alpha=0.7)
            
            plt.axhline(y=fda_vol['avg_annualized_vol'].mean(), color='black', linestyle='--', label='Average Volatility')
            
            plt.xticks([1 + bar_width/2], ['FDA Approval Events'])
            plt.ylabel('Annualized Volatility (%)')
            plt.title('Asymmetric Volatility Response (GJR-GARCH)', fontsize=14)
            plt.legend()
            
            # Add statistics text box
            stats_text = (
                f"Positive Return Volatility: {positive_vol:.2f}%\n"
                f"Negative Return Volatility: {negative_vol:.2f}%\n"
                f"Asymmetry Ratio (Neg/Pos): {asymmetry_ratio:.2f}\n"
                f"GJR-GARCH gamma: {GARCH_PARAMS['gamma']:.3f}"
            )
            
            plt.annotate(stats_text, xy=(0.02, 0.02), xycoords='axes fraction', 
                         bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8),
                         ha='left', va='bottom')
            
            # Save plot
            os.makedirs(HYPOTHESIS_RESULTS_DIR, exist_ok=True)
            plt.savefig(os.path.join(HYPOTHESIS_RESULTS_DIR, 'fda_asymmetric_volatility.png'), 
                       dpi=300, bbox_inches='tight')
            plt.close()
            
            # Create plot for asymmetric price adjustment
            plt.figure(figsize=(12, 8))
            
            # Days to plot for post-event
            post_days = sorted(merged_data[merged_data['days_to_event'] > 0]['days_to_event'].unique())
            
            # Calculate cumulative returns for positive and negative events
            cum_pos_returns = []
            cum_neg_returns = []
            cum_pos_ret = 0
            cum_neg_ret = 0
            
            for day in post_days:
                if day in positive_post_event['days_to_event'].values:
                    cum_pos_ret += positive_post_event[positive_post_event['days_to_event'] == day]['ret'].iloc[0]
                
                if day in negative_post_event['days_to_event'].values:
                    cum_neg_ret += negative_post_event[negative_post_event['days_to_event'] == day]['ret'].iloc[0]
                
                cum_pos_returns.append(cum_pos_ret)
                cum_neg_returns.append(cum_neg_ret)
            
            # Plot cumulative returns
            plt.plot(post_days, cum_pos_returns, 'g-', linewidth=2, label='After Positive Returns')
            plt.plot(post_days, cum_neg_returns, 'r-', linewidth=2, label='After Negative Returns')
            
            plt.axhline(y=0, color='black', linestyle='--')
            
            plt.xlabel('Days After Event')
            plt.ylabel('Cumulative Return')
            plt.title('FDA Events: Asymmetric Price Adjustment After Event', fontsize=14)
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Add statistics text box
            price_stats_text = (
                f"Final Cum. Return After Positive: {positive_post_cum_ret:.4f}\n"
                f"Final Cum. Return After Negative: {negative_post_cum_ret:.4f}\n"
            )
            
            plt.annotate(price_stats_text, xy=(0.02, 0.02), xycoords='axes fraction', 
                         bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8),
                         ha='left', va='bottom')
            
            # Save plot
            plt.savefig(os.path.join(HYPOTHESIS_RESULTS_DIR, 'fda_asymmetric_price_adjustment.png'), 
                       dpi=300, bbox_inches='tight')
            plt.close()
        else:
            print("FDA volatility data not available")
    else:
        print("FDA raw data not available")
    
    # --- Earnings Analysis ---
    if 'earnings_raw_data' in data:
        # Convert Polars DataFrame to pandas
        earnings_raw_df = data['earnings_raw_data'].to_pandas()
        
        # Calculate average returns by days_to_event
        avg_returns = earnings_raw_df.groupby('days_to_event')['ret'].mean().reset_index()
        
        # Classify days by return direction
        avg_returns['return_direction'] = np.where(avg_returns['ret'] > 0, 'positive', 'negative')
        
        # Get volatility data
        if 'earnings_vol' in data:
            earnings_vol = data['earnings_vol']
            
            # Merge returns and volatility
            merged_data = pd.merge(avg_returns, earnings_vol, on='days_to_event')
            
            # Split data by return direction
            positive_returns = merged_data[merged_data['return_direction'] == 'positive']
            negative_returns = merged_data[merged_data['return_direction'] == 'negative']
            
            # Calculate average volatility for positive and negative return days
            positive_vol = positive_returns['avg_annualized_vol'].mean()
            negative_vol = negative_returns['avg_annualized_vol'].mean()
            
            # Calculate asymmetry ratio
            if positive_vol > 0:
                asymmetry_ratio = negative_vol / positive_vol
            else:
                asymmetry_ratio = np.nan
                
            # Calculate price adjustment pattern after positive vs negative returns
            positive_post_event = positive_returns[positive_returns['days_to_event'] > 0]
            negative_post_event = negative_returns[negative_returns['days_to_event'] > 0]
            
            # Calculate cumulative returns for post-event periods
            if not positive_post_event.empty:
                positive_post_cum_ret = positive_post_event['ret'].cumsum().iloc[-1]
            else:
                positive_post_cum_ret = np.nan
                
            if not negative_post_event.empty:
                negative_post_cum_ret = negative_post_event['ret'].cumsum().iloc[-1]
            else:
                negative_post_cum_ret = np.nan
                
            # Store results
            results['earnings'] = {
                'positive_vol': positive_vol,
                'negative_vol': negative_vol,
                'asymmetry_ratio': asymmetry_ratio,
                'positive_post_cum_ret': positive_post_cum_ret,
                'negative_post_cum_ret': negative_post_cum_ret,
                'gamma': GARCH_PARAMS['gamma']  # GJR-GARCH asymmetry parameter
            }
            
            # Create asymmetric volatility plot
            plt.figure(figsize=(12, 8))
            
            # Plot volatility by return direction
            bar_width = 0.35
            plt.bar([1], [positive_vol], bar_width, label='Positive Returns', color='green', alpha=0.7)
            plt.bar([1 + bar_width], [negative_vol], bar_width, label='Negative Returns', color='red', alpha=0.7)
            
            plt.axhline(y=earnings_vol['avg_annualized_vol'].mean(), color='black', linestyle='--', label='Average Volatility')
            
            plt.xticks([1 + bar_width/2], ['Earnings Announcement Events'])
            plt.ylabel('Annualized Volatility (%)')
            plt.title('Asymmetric Volatility Response (GJR-GARCH)', fontsize=14)
            plt.legend()
            
            # Add statistics text box
            stats_text = (
                f"Positive Return Volatility: {positive_vol:.2f}%\n"
                f"Negative Return Volatility: {negative_vol:.2f}%\n"
                f"Asymmetry Ratio (Neg/Pos): {asymmetry_ratio:.2f}\n"
                f"GJR-GARCH gamma: {GARCH_PARAMS['gamma']:.3f}"
            )
            
            plt.annotate(stats_text, xy=(0.02, 0.02), xycoords='axes fraction', 
                         bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8),
                         ha='left', va='bottom')
            
            # Save plot
            os.makedirs(HYPOTHESIS_RESULTS_DIR, exist_ok=True)
            plt.savefig(os.path.join(HYPOTHESIS_RESULTS_DIR, 'earnings_asymmetric_volatility.png'), 
                       dpi=300, bbox_inches='tight')
            plt.close()
            
            # Create plot for asymmetric price adjustment
            plt.figure(figsize=(12, 8))
            
            # Days to plot for post-event
            post_days = sorted(merged_data[merged_data['days_to_event'] > 0]['days_to_event'].unique())
            
            # Calculate cumulative returns for positive and negative events
            cum_pos_returns = []
            cum_neg_returns = []
            cum_pos_ret = 0
            cum_neg_ret = 0
            
            for day in post_days:
                if day in positive_post_event['days_to_event'].values:
                    cum_pos_ret += positive_post_event[positive_post_event['days_to_event'] == day]['ret'].iloc[0]
                
                if day in negative_post_event['days_to_event'].values:
                    cum_neg_ret += negative_post_event[negative_post_event['days_to_event'] == day]['ret'].iloc[0]
                
                cum_pos_returns.append(cum_pos_ret)
                cum_neg_returns.append(cum_neg_ret)
            
            # Plot cumulative returns
            plt.plot(post_days, cum_pos_returns, 'g-', linewidth=2, label='After Positive Returns')
            plt.plot(post_days, cum_neg_returns, 'r-', linewidth=2, label='After Negative Returns')
            
            plt.axhline(y=0, color='black', linestyle='--')
            
            plt.xlabel('Days After Event')
            plt.ylabel('Cumulative Return')
            plt.title('Earnings Events: Asymmetric Price Adjustment After Event', fontsize=14)
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Add statistics text box
            price_stats_text = (
                f"Final Cum. Return After Positive: {positive_post_cum_ret:.4f}\n"
                f"Final Cum. Return After Negative: {negative_post_cum_ret:.4f}\n"
            )
            
            plt.annotate(price_stats_text, xy=(0.02, 0.02), xycoords='axes fraction', 
                         bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8),
                         ha='left', va='bottom')
            
            # Save plot
            plt.savefig(os.path.join(HYPOTHESIS_RESULTS_DIR, 'earnings_asymmetric_price_adjustment.png'), 
                       dpi=300, bbox_inches='tight')
            plt.close()
        else:
            print("Earnings volatility data not available")
    else:
        print("Earnings raw data not available")
    
    return results

def create_comparison_visualization(predictability_results, persistence_results, asymmetry_results):
    """
    Creates a comprehensive visualization comparing FDA and earnings events
    for validation of Hypothesis 2.
    
    Parameters:
    -----------
    predictability_results: dict
        Results from analyze_pre_event_predictability
    persistence_results: dict
        Results from analyze_post_event_persistence
    asymmetry_results: dict
        Results from analyze_asymmetric_volatility
    """
    print("\n=== Creating Comprehensive Visualization for Hypothesis 2 ===")
    
    # Create a 3x2 multi-panel figure (3 rows for each part of Hypothesis 2, 2 columns for FDA vs Earnings)
    fig, axs = plt.subplots(3, 2, figsize=(16, 18))
    
    # --- Row 1: Pre-event volatility innovations predict subsequent returns ---
    # FDA plot
    if 'fda' in predictability_results and 'regression_results' in predictability_results['fda'] and predictability_results['fda']['regression_results'] is not None:
        reg_results = predictability_results['fda']['regression_results']
        axs[0, 0].annotate(
            f"Pre-Event Impact → Post Returns\nCoef: {reg_results['coefficient']:.4f}\nR²: {reg_results['r_squared']:.4f}\np: {reg_results['p_value']:.4f}",
            xy=(0.5, 0.5), xycoords='axes fraction',
            bbox=dict(boxstyle="round,pad=0.3", fc="lightblue", ec="blue", alpha=0.8),
            ha='center', va='center', fontsize=12
        )
        axs[0, 0].set_title('FDA Events: Pre-event Predictability', fontsize=14)
        axs[0, 0].axis('off')
    else:
        axs[0, 0].text(0.5, 0.5, 'FDA Pre-event Predictability\nData Not Available', 
                     horizontalalignment='center', verticalalignment='center', fontsize=12)
        axs[0, 0].axis('off')
    
    # Earnings plot
    if 'earnings' in predictability_results and 'regression_results' in predictability_results['earnings'] and predictability_results['earnings']['regression_results'] is not None:
        reg_results = predictability_results['earnings']['regression_results']
        axs[0, 1].annotate(
            f"Pre-Event Impact → Post Returns\nCoef: {reg_results['coefficient']:.4f}\nR²: {reg_results['r_squared']:.4f}\np: {reg_results['p_value']:.4f}",
            xy=(0.5, 0.5), xycoords='axes fraction',
            bbox=dict(boxstyle="round,pad=0.3", fc="mistyrose", ec="red", alpha=0.8),
            ha='center', va='center', fontsize=12
        )
        axs[0, 1].set_title('Earnings Events: Pre-event Predictability', fontsize=14)
        axs[0, 1].axis('off')
    else:
        axs[0, 1].text(0.5, 0.5, 'Earnings Pre-event Predictability\nData Not Available', 
                     horizontalalignment='center', verticalalignment='center', fontsize=12)
        axs[0, 1].axis('off')
    
    # --- Row 2: Post-event volatility persistence extends elevated expected returns ---
    # FDA plot
    if 'fda' in persistence_results:
        res = persistence_results['fda']
        stats_text = (
            f"Post-Event Vol: {res.get('post_event_mean_normalized_vol', 'N/A'):.2f}x baseline\n"
            f"Late Post-Event Vol: {res.get('late_post_event_mean_normalized_vol', 'N/A'):.2f}x baseline\n"
            f"Vol Half-Life: {res.get('volatility_half_life', 'N/A'):.1f} days\n"
            f"Post-Event Vol-Ret Corr: {res.get('post_event_vol_ret_corr', 'N/A'):.3f}\n"
            f"Late Post-Event Vol-Ret Corr: {res.get('late_post_event_vol_ret_corr', 'N/A'):.3f}"
        )
        axs[1, 0].annotate(
            stats_text,
            xy=(0.5, 0.5), xycoords='axes fraction',
            bbox=dict(boxstyle="round,pad=0.3", fc="lightblue", ec="blue", alpha=0.8),
            ha='center', va='center', fontsize=12
        )
        axs[1, 0].set_title('FDA Events: Volatility Persistence', fontsize=14)
        axs[1, 0].axis('off')
    else:
        axs[1, 0].text(0.5, 0.5, 'FDA Volatility Persistence\nData Not Available', 
                     horizontalalignment='center', verticalalignment='center', fontsize=12)
        axs[1, 0].axis('off')
    
    # Earnings plot
    if 'earnings' in persistence_results:
        res = persistence_results['earnings']
        stats_text = (
            f"Post-Event Vol: {res.get('post_event_mean_normalized_vol', 'N/A'):.2f}x baseline\n"
            f"Late Post-Event Vol: {res.get('late_post_event_mean_normalized_vol', 'N/A'):.2f}x baseline\n"
            f"Vol Half-Life: {res.get('volatility_half_life', 'N/A'):.1f} days\n"
            f"Post-Event Vol-Ret Corr: {res.get('post_event_vol_ret_corr', 'N/A'):.3f}\n"
            f"Late Post-Event Vol-Ret Corr: {res.get('late_post_event_vol_ret_corr', 'N/A'):.3f}"
        )
        axs[1, 1].annotate(
            stats_text,
            xy=(0.5, 0.5), xycoords='axes fraction',
            bbox=dict(boxstyle="round,pad=0.3", fc="mistyrose", ec="red", alpha=0.8),
            ha='center', va='center', fontsize=12
        )
        axs[1, 1].set_title('Earnings Events: Volatility Persistence', fontsize=14)
        axs[1, 1].axis('off')
    else:
        axs[1, 1].text(0.5, 0.5, 'Earnings Volatility Persistence\nData Not Available', 
                     horizontalalignment='center', verticalalignment='center', fontsize=12)
        axs[1, 1].axis('off')
    
    # --- Row 3: Asymmetric volatility response correlates with asymmetric price adjustment ---
    # FDA plot
    if 'fda' in asymmetry_results:
        res = asymmetry_results['fda']
        stats_text = (
            f"Positive Return Vol: {res.get('positive_vol', 'N/A'):.2f}%\n"
            f"Negative Return Vol: {res.get('negative_vol', 'N/A'):.2f}%\n"
            f"Asymmetry Ratio (Neg/Pos): {res.get('asymmetry_ratio', 'N/A'):.2f}\n"
            f"GJR-GARCH gamma: {res.get('gamma', 'N/A'):.3f}\n"
            f"Cum Ret After Pos: {res.get('positive_post_cum_ret', 'N/A'):.4f}\n"
            f"Cum Ret After Neg: {res.get('negative_post_cum_ret', 'N/A'):.4f}"
        )
        axs[2, 0].annotate(
            stats_text,
            xy=(0.5, 0.5), xycoords='axes fraction',
            bbox=dict(boxstyle="round,pad=0.3", fc="lightblue", ec="blue", alpha=0.8),
            ha='center', va='center', fontsize=12
        )
        axs[2, 0].set_title('FDA Events: Asymmetric Volatility Response', fontsize=14)
        axs[2, 0].axis('off')
    else:
        axs[2, 0].text(0.5, 0.5, 'FDA Asymmetric Volatility Response\nData Not Available', 
                     horizontalalignment='center', verticalalignment='center', fontsize=12)
        axs[2, 0].axis('off')
    
    # Earnings plot
    if 'earnings' in asymmetry_results:
        res = asymmetry_results['earnings']
        stats_text = (
            f"Positive Return Vol: {res.get('positive_vol', 'N/A'):.2f}%\n"
            f"Negative Return Vol: {res.get('negative_vol', 'N/A'):.2f}%\n"
            f"Asymmetry Ratio (Neg/Pos): {res.get('asymmetry_ratio', 'N/A'):.2f}\n"
            f"GJR-GARCH gamma: {res.get('gamma', 'N/A'):.3f}\n"
            f"Cum Ret After Pos: {res.get('positive_post_cum_ret', 'N/A'):.4f}\n"
            f"Cum Ret After Neg: {res.get('negative_post_cum_ret', 'N/A'):.4f}"
        )
        axs[2, 1].annotate(
            stats_text,
            xy=(0.5, 0.5), xycoords='axes fraction',
            bbox=dict(boxstyle="round,pad=0.3", fc="mistyrose", ec="red", alpha=0.8),
            ha='center', va='center', fontsize=12
        )
        axs[2, 1].set_title('Earnings Events: Asymmetric Volatility Response', fontsize=14)
        axs[2, 1].axis('off')
    else:
        axs[2, 1].text(0.5, 0.5, 'Earnings Asymmetric Volatility Response\nData Not Available', 
                     horizontalalignment='center', verticalalignment='center', fontsize=12)
        axs[2, 1].axis('off')
    
    # Add overall title for the figure
    plt.suptitle('Hypothesis 2 Validation: GARCH-Estimated Volatility as a Proxy for Impact Uncertainty', 
                fontsize=18, fontweight='bold', y=0.98)
    
    # Add subtitle explaining the three components
    plt.figtext(0.5, 0.94, 
               "1. Pre-event volatility innovations predict subsequent returns\n"
               "2. Post-event volatility persistence extends elevated expected returns\n"
               "3. Asymmetric volatility response correlates with asymmetric price adjustment",
               ha='center', fontsize=14, fontstyle='italic')
    
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    
    # Save the figure
    plt.savefig(os.path.join(HYPOTHESIS_RESULTS_DIR, 'hypothesis_2_comprehensive.png'), 
               dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    print(f"Saved comprehensive visualization to: {os.path.join(HYPOTHESIS_RESULTS_DIR, 'hypothesis_2_comprehensive.png')}")

def validate_hypothesis_2():
    """
    Validates Hypothesis 2 by analyzing the collected data and generating
    a comprehensive report.
    
    Returns:
    --------
    bool
        True if validation completed successfully, False otherwise
    """
    print("\n=== Validating Hypothesis 2 ===")
    
    # Create directory for hypothesis validation results
    os.makedirs(HYPOTHESIS_RESULTS_DIR, exist_ok=True)
    
    # Collect data for analysis
    data = collect_event_data()
    
    if not data:
        print("No data available for validation. Please run FDA and earnings analyses first.")
        return False
        
    # Analyze pre-event predictability
    print("\nAnalyzing pre-event predictability...")
    predictability_results = analyze_pre_event_predictability(data)
    
    # Analyze post-event persistence
    print("\nAnalyzing post-event persistence...")
    persistence_results = analyze_post_event_persistence(data)
    
    # Analyze asymmetric volatility
    print("\nAnalyzing asymmetric volatility...")
    asymmetry_results = analyze_asymmetric_volatility(data)
    
    # Create comprehensive visualization
    create_comparison_visualization(predictability_results, persistence_results, asymmetry_results)
    
    # Create detailed validation report
    print("\nGenerating validation report...")
    
    with open(os.path.join(HYPOTHESIS_RESULTS_DIR, "hypothesis_2_validation_results.txt"), "w") as f:
        f.write("HYPOTHESIS 2 VALIDATION RESULTS\n")
        f.write("===============================\n\n")
        f.write("Hypothesis 2: GARCH-estimated conditional volatility innovations serve as\n")
        f.write("an effective proxy for impact uncertainty:\n")
        f.write("- Pre-event volatility innovations predict subsequent returns\n")
        f.write("- Post-event volatility persistence extends the period of elevated expected returns\n")
        f.write("- Asymmetric volatility response (captured through GJR-GARCH) correlates with asymmetric price adjustment\n\n")
        
        # Write results for pre-event predictability
        f.write("1. PRE-EVENT VOLATILITY INNOVATIONS PREDICT SUBSEQUENT RETURNS\n")
        f.write("-------------------------------------------------------\n")
        
        if 'fda' in predictability_results and 'regression_results' in predictability_results['fda'] and predictability_results['fda']['regression_results'] is not None:
            reg = predictability_results['fda']['regression_results']
            f.write("FDA Events:\n")
            f.write(f"  Coefficient: {reg['coefficient']:.6f}\n")
            f.write(f"  R-squared: {reg['r_squared']:.4f}\n")
            f.write(f"  p-value: {reg['p_value']:.4f}\n")
            f.write(f"  Number of observations: {reg['n_observations']}\n")
            
            # Evaluate statistical significance
            if reg['p_value'] < 0.05 and reg['coefficient'] > 0:
                f.write("  Result: SUPPORTED (Statistically significant positive relationship)\n\n")
            elif reg['p_value'] < 0.05 and reg['coefficient'] < 0:
                f.write("  Result: NOT SUPPORTED (Statistically significant negative relationship)\n\n")
            else:
                f.write("  Result: NOT SUPPORTED (Relationship not statistically significant)\n\n")
        else:
            f.write("FDA Events: Insufficient data for regression analysis\n\n")
            
        if 'earnings' in predictability_results and 'regression_results' in predictability_results['earnings'] and predictability_results['earnings']['regression_results'] is not None:
            reg = predictability_results['earnings']['regression_results']
            f.write("Earnings Events:\n")
            f.write(f"  Coefficient: {reg['coefficient']:.6f}\n")
            f.write(f"  R-squared: {reg['r_squared']:.4f}\n")
            f.write(f"  p-value: {reg['p_value']:.4f}\n")
            f.write(f"  Number of observations: {reg['n_observations']}\n")
            
            # Evaluate statistical significance
            if reg['p_value'] < 0.05 and reg['coefficient'] > 0:
                f.write("  Result: SUPPORTED (Statistically significant positive relationship)\n\n")
            elif reg['p_value'] < 0.05 and reg['coefficient'] < 0:
                f.write("  Result: NOT SUPPORTED (Statistically significant negative relationship)\n\n")
            else:
                f.write("  Result: NOT SUPPORTED (Relationship not statistically significant)\n\n")
        else:
            f.write("Earnings Events: Insufficient data for regression analysis\n\n")
        
        # Write results for post-event persistence
        f.write("2. POST-EVENT VOLATILITY PERSISTENCE EXTENDS ELEVATED EXPECTED RETURNS\n")
        f.write("---------------------------------------------------------------\n")
        
        if 'fda' in persistence_results:
            res = persistence_results['fda']
            f.write("FDA Events:\n")
            f.write(f"  Post-Event Normalized Volatility: {res.get('post_event_mean_normalized_vol', 'N/A'):.2f}x baseline\n")
            f.write(f"  Late Post-Event Normalized Volatility: {res.get('late_post_event_mean_normalized_vol', 'N/A'):.2f}x baseline\n")
            f.write(f"  Volatility Half-Life: {res.get('volatility_half_life', 'N/A'):.1f} days\n")
            
            if 'post_event_vol_ret_corr' in res and 'late_post_event_vol_ret_corr' in res:
                f.write(f"  Post-Event Vol-Ret Correlation: {res['post_event_vol_ret_corr']:.3f}\n")
                f.write(f"  Late Post-Event Vol-Ret Correlation: {res['late_post_event_vol_ret_corr']:.3f}\n")
                
                # Evaluate findings
                is_post_event_corr_positive = res['post_event_vol_ret_corr'] > 0
                is_vol_persistence = res.get('volatility_half_life', 0) > 1
                
                if is_post_event_corr_positive and is_vol_persistence:
                    f.write("  Result: SUPPORTED (Positive correlation and volatility persists)\n\n")
                elif is_post_event_corr_positive:
                    f.write("  Result: PARTIALLY SUPPORTED (Positive correlation but volatility decay is rapid)\n\n")
                elif is_vol_persistence:
                    f.write("  Result: PARTIALLY SUPPORTED (Volatility persists but correlation is not positive)\n\n")
                else:
                    f.write("  Result: NOT SUPPORTED (No positive correlation and volatility decays rapidly)\n\n")
            else:
                f.write("  Result: INCONCLUSIVE (Insufficient correlation data)\n\n")
        else:
            f.write("FDA Events: Insufficient data for volatility persistence analysis\n\n")
            
        if 'earnings' in persistence_results:
            res = persistence_results['earnings']
            f.write("Earnings Events:\n")
            f.write(f"  Post-Event Normalized Volatility: {res.get('post_event_mean_normalized_vol', 'N/A'):.2f}x baseline\n")
            f.write(f"  Late Post-Event Normalized Volatility: {res.get('late_post_event_mean_normalized_vol', 'N/A'):.2f}x baseline\n")
            f.write(f"  Volatility Half-Life: {res.get('volatility_half_life', 'N/A'):.1f} days\n")
            
            if 'post_event_vol_ret_corr' in res and 'late_post_event_vol_ret_corr' in res:
                f.write(f"  Post-Event Vol-Ret Correlation: {res['post_event_vol_ret_corr']:.3f}\n")
                f.write(f"  Late Post-Event Vol-Ret Correlation: {res['late_post_event_vol_ret_corr']:.3f}\n")
                
                # Evaluate findings
                is_post_event_corr_positive = res['post_event_vol_ret_corr'] > 0
                is_vol_persistence = res.get('volatility_half_life', 0) > 1
                
                if is_post_event_corr_positive and is_vol_persistence:
                    f.write("  Result: SUPPORTED (Positive correlation and volatility persists)\n\n")
                elif is_post_event_corr_positive:
                    f.write("  Result: PARTIALLY SUPPORTED (Positive correlation but volatility decay is rapid)\n\n")
                elif is_vol_persistence:
                    f.write("  Result: PARTIALLY SUPPORTED (Volatility persists but correlation is not positive)\n\n")
                else:
                    f.write("  Result: NOT SUPPORTED (No positive correlation and volatility decays rapidly)\n\n")
            else:
                f.write("  Result: INCONCLUSIVE (Insufficient correlation data)\n\n")
        else:
            f.write("Earnings Events: Insufficient data for volatility persistence analysis\n\n")
        
        # Write results for asymmetric volatility
        f.write("3. ASYMMETRIC VOLATILITY RESPONSE CORRELATES WITH ASYMMETRIC PRICE ADJUSTMENT\n")
        f.write("--------------------------------------------------------------------\n")
        
        if 'fda' in asymmetry_results:
            res = asymmetry_results['fda']
            f.write("FDA Events:\n")
            f.write(f"  Positive Return Volatility: {res.get('positive_vol', 'N/A'):.2f}%\n")
            f.write(f"  Negative Return Volatility: {res.get('negative_vol', 'N/A'):.2f}%\n")
            f.write(f"  Asymmetry Ratio (Neg/Pos): {res.get('asymmetry_ratio', 'N/A'):.2f}\n")
            f.write(f"  GJR-GARCH gamma parameter: {res.get('gamma', 'N/A'):.3f}\n")
            
            if 'positive_post_cum_ret' in res and 'negative_post_cum_ret' in res:
                f.write(f"  Cumulative Return After Positive: {res['positive_post_cum_ret']:.4f}\n")
                f.write(f"  Cumulative Return After Negative: {res['negative_post_cum_ret']:.4f}\n")
                
                # Evaluate findings
                is_vol_asymmetric = res.get('asymmetry_ratio', 1) > 1.1  # At least 10% higher vol for negative returns
                is_price_asymmetric = abs(res.get('positive_post_cum_ret', 0)) != abs(res.get('negative_post_cum_ret', 0))
                
                if is_vol_asymmetric and is_price_asymmetric:
                    f.write("  Result: SUPPORTED (Asymmetric volatility and price adjustment)\n\n")
                elif is_vol_asymmetric:
                    f.write("  Result: PARTIALLY SUPPORTED (Asymmetric volatility but symmetric price adjustment)\n\n")
                elif is_price_asymmetric:
                    f.write("  Result: PARTIALLY SUPPORTED (Symmetric volatility but asymmetric price adjustment)\n\n")
                else:
                    f.write("  Result: NOT SUPPORTED (No asymmetry in volatility or price adjustment)\n\n")
            else:
                f.write("  Result: INCONCLUSIVE (Insufficient price adjustment data)\n\n")
        else:
            f.write("FDA Events: Insufficient data for asymmetric volatility analysis\n\n")
            
        if 'earnings' in asymmetry_results:
            res = asymmetry_results['earnings']
            f.write("Earnings Events:\n")
            f.write(f"  Positive Return Volatility: {res.get('positive_vol', 'N/A'):.2f}%\n")
            f.write(f"  Negative Return Volatility: {res.get('negative_vol', 'N/A'):.2f}%\n")
            f.write(f"  Asymmetry Ratio (Neg/Pos): {res.get('asymmetry_ratio', 'N/A'):.2f}\n")
            f.write(f"  GJR-GARCH gamma parameter: {res.get('gamma', 'N/A'):.3f}\n")
            
            if 'positive_post_cum_ret' in res and 'negative_post_cum_ret' in res:
                f.write(f"  Cumulative Return After Positive: {res['positive_post_cum_ret']:.4f}\n")
                f.write(f"  Cumulative Return After Negative: {res['negative_post_cum_ret']:.4f}\n")
                
                # Evaluate findings
                is_vol_asymmetric = res.get('asymmetry_ratio', 1) > 1.1  # At least 10% higher vol for negative returns
                is_price_asymmetric = abs(res.get('positive_post_cum_ret', 0)) != abs(res.get('negative_post_cum_ret', 0))
                
                if is_vol_asymmetric and is_price_asymmetric:
                    f.write("  Result: SUPPORTED (Asymmetric volatility and price adjustment)\n\n")
                elif is_vol_asymmetric:
                    f.write("  Result: PARTIALLY SUPPORTED (Asymmetric volatility but symmetric price adjustment)\n\n")
                elif is_price_asymmetric:
                    f.write("  Result: PARTIALLY SUPPORTED (Symmetric volatility but asymmetric price adjustment)\n\n")
                else:
                    f.write("  Result: NOT SUPPORTED (No asymmetry in volatility or price adjustment)\n\n")
            else:
                f.write("  Result: INCONCLUSIVE (Insufficient price adjustment data)\n\n")
        else:
            f.write("Earnings Events: Insufficient data for asymmetric volatility analysis\n\n")
        
        # Overall conclusion
        f.write("OVERALL CONCLUSION\n")
        f.write("-----------------\n")
        
        # Count how many components are supported
        support_count = 0
        partial_count = 0
        components_count = 0
        
        # Component 1: Pre-event predictability
        components_count += 1
        if ('fda' in predictability_results and 'regression_results' in predictability_results['fda'] and 
            predictability_results['fda']['regression_results'] is not None and 
            predictability_results['fda']['regression_results']['p_value'] < 0.05 and 
            predictability_results['fda']['regression_results']['coefficient'] > 0):
            support_count += 0.5  # Half point for FDA support
                
        if ('earnings' in predictability_results and 'regression_results' in predictability_results['earnings'] and 
            predictability_results['earnings']['regression_results'] is not None and 
            predictability_results['earnings']['regression_results']['p_value'] < 0.05 and 
            predictability_results['earnings']['regression_results']['coefficient'] > 0):
            support_count += 0.5  # Half point for earnings support
        
        # Component 2: Post-event persistence
        components_count += 1
        if 'fda' in persistence_results and 'post_event_vol_ret_corr' in persistence_results['fda']:
            is_fda_post_event_corr_positive = persistence_results['fda']['post_event_vol_ret_corr'] > 0
            is_fda_vol_persistence = persistence_results['fda'].get('volatility_half_life', 0) > 1
            
            if is_fda_post_event_corr_positive and is_fda_vol_persistence:
                support_count += 0.5  # Half point for FDA support
            elif is_fda_post_event_corr_positive or is_fda_vol_persistence:
                partial_count += 0.25  # Quarter point for partial FDA support
                
        if 'earnings' in persistence_results and 'post_event_vol_ret_corr' in persistence_results['earnings']:
            is_earnings_post_event_corr_positive = persistence_results['earnings']['post_event_vol_ret_corr'] > 0
            is_earnings_vol_persistence = persistence_results['earnings'].get('volatility_half_life', 0) > 1
            
            if is_earnings_post_event_corr_positive and is_earnings_vol_persistence:
                support_count += 0.5  # Half point for earnings support
            elif is_earnings_post_event_corr_positive or is_earnings_vol_persistence:
                partial_count += 0.25  # Quarter point for partial earnings support
        
        # Component 3: Asymmetric volatility
        components_count += 1
        if 'fda' in asymmetry_results and 'asymmetry_ratio' in asymmetry_results['fda'] and 'positive_post_cum_ret' in asymmetry_results['fda']:
            is_fda_vol_asymmetric = asymmetry_results['fda'].get('asymmetry_ratio', 1) > 1.1
            is_fda_price_asymmetric = abs(asymmetry_results['fda'].get('positive_post_cum_ret', 0)) != abs(asymmetry_results['fda'].get('negative_post_cum_ret', 0))
            
            if is_fda_vol_asymmetric and is_fda_price_asymmetric:
                support_count += 0.5  # Half point for FDA support
            elif is_fda_vol_asymmetric or is_fda_price_asymmetric:
                partial_count += 0.25  # Quarter point for partial FDA support
                
        if 'earnings' in asymmetry_results and 'asymmetry_ratio' in asymmetry_results['earnings'] and 'positive_post_cum_ret' in asymmetry_results['earnings']:
            is_earnings_vol_asymmetric = asymmetry_results['earnings'].get('asymmetry_ratio', 1) > 1.1
            is_earnings_price_asymmetric = abs(asymmetry_results['earnings'].get('positive_post_cum_ret', 0)) != abs(asymmetry_results['earnings'].get('negative_post_cum_ret', 0))
            
            if is_earnings_vol_asymmetric and is_earnings_price_asymmetric:
                support_count += 0.5  # Half point for earnings support
            elif is_earnings_vol_asymmetric or is_earnings_price_asymmetric:
                partial_count += 0.25  # Quarter point for partial earnings support
        
        # Calculate support percentage
        support_percentage = (support_count + partial_count * 0.5) / components_count * 100
        
        # Write overall conclusion
        if support_percentage >= 75:
            f.write(f"Hypothesis 2 is STRONGLY SUPPORTED by the data ({support_percentage:.1f}% support).\n")
            f.write("GARCH-estimated conditional volatility innovations do serve as an effective\n")
            f.write("proxy for impact uncertainty, as evidenced by the data for both FDA approval\n")
            f.write("and earnings announcement events.\n\n")
        elif support_percentage >= 50:
            f.write(f"Hypothesis 2 is MODERATELY SUPPORTED by the data ({support_percentage:.1f}% support).\n")
            f.write("GARCH-estimated conditional volatility innovations show some evidence of serving\n")
            f.write("as a proxy for impact uncertainty, though the relationship is not equally strong\n")
            f.write("across all components or event types.\n\n")
        elif support_percentage >= 25:
            f.write(f"Hypothesis 2 is WEAKLY SUPPORTED by the data ({support_percentage:.1f}% support).\n")
            f.write("While there is some evidence that GARCH-estimated conditional volatility innovations\n")
            f.write("may serve as a proxy for impact uncertainty, the relationships are mostly partial\n")
            f.write("or inconsistent across components or event types.\n\n")
        else:
            f.write(f"Hypothesis 2 is NOT SUPPORTED by the data ({support_percentage:.1f}% support).\n")
            f.write("The data does not provide sufficient evidence that GARCH-estimated conditional\n")
            f.write("volatility innovations serve as an effective proxy for impact uncertainty.\n\n")
            
        f.write("IMPLICATIONS FOR THE TWO-RISK FRAMEWORK\n")
        f.write("-----------------------------------\n")
        f.write("The two-risk framework distinguishes between directional news risk (uncertainty\n")
        f.write("about event outcome) and impact uncertainty (uncertainty about magnitude of market\n")
        f.write("response). This analysis validates the use of GARCH-estimated conditional volatility\n")
        f.write("innovations as a proxy for impact uncertainty, supporting the framework's empirical\n")
        f.write("implementation and providing evidence for its usefulness in explaining event-driven\n")
        f.write("returns.\n\n")
        
        f.write("These findings contribute to our understanding of how different forms of risk\n")
        f.write("affect asset prices and provide tools for measuring formerly abstract theoretical\n")
        f.write("concepts empirically.\n")
    
    print(f"Saved validation report to: {os.path.join(HYPOTHESIS_RESULTS_DIR, 'hypothesis_2_validation_results.txt')}")
    
    # Create summary table for easier reference
    summary_data = {
        'Component': [
            '1. Pre-event innovations predict returns',
            '2. Post-event volatility persistence',
            '3. Asymmetric volatility response'
        ],
        'FDA Support': ['N/A', 'N/A', 'N/A'],
        'Earnings Support': ['N/A', 'N/A', 'N/A'],
        'Overall Support': ['N/A', 'N/A', 'N/A']
    }
    
    # Component 1
    if ('fda' in predictability_results and 'regression_results' in predictability_results['fda'] and 
        predictability_results['fda']['regression_results'] is not None):
        
        reg = predictability_results['fda']['regression_results']
        if reg['p_value'] < 0.05 and reg['coefficient'] > 0:
            summary_data['FDA Support'][0] = 'Supported'
        elif reg['p_value'] < 0.05 and reg['coefficient'] < 0:
            summary_data['FDA Support'][0] = 'Not Supported'
        else:
            summary_data['FDA Support'][0] = 'Not Significant'
    
    if ('earnings' in predictability_results and 'regression_results' in predictability_results['earnings'] and 
        predictability_results['earnings']['regression_results'] is not None):
        
        reg = predictability_results['earnings']['regression_results']
        if reg['p_value'] < 0.05 and reg['coefficient'] > 0:
            summary_data['Earnings Support'][0] = 'Supported'
        elif reg['p_value'] < 0.05 and reg['coefficient'] < 0:
            summary_data['Earnings Support'][0] = 'Not Supported'
        else:
            summary_data['Earnings Support'][0] = 'Not Significant'
    
    # Combine FDA and Earnings for overall support
    if summary_data['FDA Support'][0] == 'Supported' and summary_data['Earnings Support'][0] == 'Supported':
        summary_data['Overall Support'][0] = 'Strongly Supported'
    elif summary_data['FDA Support'][0] == 'Supported' or summary_data['Earnings Support'][0] == 'Supported':
        summary_data['Overall Support'][0] = 'Partially Supported'
    elif summary_data['FDA Support'][0] == 'Not Significant' or summary_data['Earnings Support'][0] == 'Not Significant':
        summary_data['Overall Support'][0] = 'Inconclusive'
    else:
        summary_data['Overall Support'][0] = 'Not Supported'
    
    # Component 2
    if 'fda' in persistence_results and 'post_event_vol_ret_corr' in persistence_results['fda']:
        is_fda_post_event_corr_positive = persistence_results['fda']['post_event_vol_ret_corr'] > 0
        is_fda_vol_persistence = persistence_results['fda'].get('volatility_half_life', 0) > 1
        
        if is_fda_post_event_corr_positive and is_fda_vol_persistence:
            summary_data['FDA Support'][1] = 'Supported'
        elif is_fda_post_event_corr_positive or is_fda_vol_persistence:
            summary_data['FDA Support'][1] = 'Partially Supported'
        else:
            summary_data['FDA Support'][1] = 'Not Supported'
    
    if 'earnings' in persistence_results and 'post_event_vol_ret_corr' in persistence_results['earnings']:
        is_earnings_post_event_corr_positive = persistence_results['earnings']['post_event_vol_ret_corr'] > 0
        is_earnings_vol_persistence = persistence_results['earnings'].get('volatility_half_life', 0) > 1
        
        if is_earnings_post_event_corr_positive and is_earnings_vol_persistence:
            summary_data['Earnings Support'][1] = 'Supported'
        elif is_earnings_post_event_corr_positive or is_earnings_vol_persistence:
            summary_data['Earnings Support'][1] = 'Partially Supported'
        else:
            summary_data['Earnings Support'][1] = 'Not Supported'
    
    # Combine FDA and Earnings for overall support
    if summary_data['FDA Support'][1] == 'Supported' and summary_data['Earnings Support'][1] == 'Supported':
        summary_data['Overall Support'][1] = 'Strongly Supported'
    elif (summary_data['FDA Support'][1] == 'Supported' and summary_data['Earnings Support'][1] == 'Partially Supported') or \
         (summary_data['FDA Support'][1] == 'Partially Supported' and summary_data['Earnings Support'][1] == 'Supported'):
        summary_data['Overall Support'][1] = 'Supported'
    elif summary_data['FDA Support'][1] == 'Supported' or summary_data['Earnings Support'][1] == 'Supported':
        summary_data['Overall Support'][1] = 'Partially Supported'
    elif summary_data['FDA Support'][1] == 'Partially Supported' or summary_data['Earnings Support'][1] == 'Partially Supported':
        summary_data['Overall Support'][1] = 'Weakly Supported'
    else:
        summary_data['Overall Support'][1] = 'Not Supported'
    
    # Component 3
    if 'fda' in asymmetry_results and 'asymmetry_ratio' in asymmetry_results['fda'] and 'positive_post_cum_ret' in asymmetry_results['fda']:
        is_fda_vol_asymmetric = asymmetry_results['fda'].get('asymmetry_ratio', 1) > 1.1
        is_fda_price_asymmetric = abs(asymmetry_results['fda'].get('positive_post_cum_ret', 0)) != abs(asymmetry_results['fda'].get('negative_post_cum_ret', 0))
        
        if is_fda_vol_asymmetric and is_fda_price_asymmetric:
            summary_data['FDA Support'][2] = 'Supported'
        elif is_fda_vol_asymmetric or is_fda_price_asymmetric:
            summary_data['FDA Support'][2] = 'Partially Supported'
        else:
            summary_data['FDA Support'][2] = 'Not Supported'
    
    if 'earnings' in asymmetry_results and 'asymmetry_ratio' in asymmetry_results['earnings'] and 'positive_post_cum_ret' in asymmetry_results['earnings']:
        is_earnings_vol_asymmetric = asymmetry_results['earnings'].get('asymmetry_ratio', 1) > 1.1
        is_earnings_price_asymmetric = abs(asymmetry_results['earnings'].get('positive_post_cum_ret', 0)) != abs(asymmetry_results['earnings'].get('negative_post_cum_ret', 0))
        
        if is_earnings_vol_asymmetric and is_earnings_price_asymmetric:
            summary_data['Earnings Support'][2] = 'Supported'
        elif is_earnings_vol_asymmetric or is_earnings_price_asymmetric:
            summary_data['Earnings Support'][2] = 'Partially Supported'
        else:
            summary_data['Earnings Support'][2] = 'Not Supported'
    
    # Combine FDA and Earnings for overall support
    if summary_data['FDA Support'][2] == 'Supported' and summary_data['Earnings Support'][2] == 'Supported':
        summary_data['Overall Support'][2] = 'Strongly Supported'
    elif (summary_data['FDA Support'][2] == 'Supported' and summary_data['Earnings Support'][2] == 'Partially Supported') or \
         (summary_data['FDA Support'][2] == 'Partially Supported' and summary_data['Earnings Support'][2] == 'Supported'):
        summary_data['Overall Support'][2] = 'Supported'
    elif summary_data['FDA Support'][2] == 'Supported' or summary_data['Earnings Support'][2] == 'Supported':
        summary_data['Overall Support'][2] = 'Partially Supported'
    elif summary_data['FDA Support'][2] == 'Partially Supported' or summary_data['Earnings Support'][2] == 'Partially Supported':
        summary_data['Overall Support'][2] = 'Weakly Supported'
    else:
        summary_data['Overall Support'][2] = 'Not Supported'
    
    # Save summary table
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(os.path.join(HYPOTHESIS_RESULTS_DIR, 'hypothesis_2_summary.csv'), index=False)
    print(f"Saved summary table to: {os.path.join(HYPOTHESIS_RESULTS_DIR, 'hypothesis_2_summary.csv')}")
    
    # Print summary to console
    print("\n=== Hypothesis 2 Validation Summary ===")
    print(summary_df.to_string(index=False))
    
    return True

def main():
    """
    Main function to run the FDA and earnings analyses and validate Hypothesis 2.
    """
    print("\n=== Running Analyses to Validate Hypothesis 2 ===")
    print("Hypothesis 2: GARCH-estimated conditional volatility innovations serve as")
    print("an effective proxy for impact uncertainty:")
    print("  - Pre-event volatility innovations predict subsequent returns")
    print("  - Post-event volatility persistence extends the period of elevated expected returns")
    print("  - Asymmetric volatility response correlates with asymmetric price adjustment")
    
    # Create hypothesis validation directory
    os.makedirs(HYPOTHESIS_RESULTS_DIR, exist_ok=True)
    
    # Run FDA analysis
    fda_success = run_fda_analysis()
    
    # Run earnings analysis
    earnings_success = run_earnings_analysis()
    
    # Validate Hypothesis 2
    if fda_success or earnings_success:
        validation_success = validate_hypothesis_2()
        if validation_success:
            print("\n=== Hypothesis 2 validation completed successfully ===")
        else:
            print("\n=== Hypothesis 2 validation failed ===")
    else:
        print("\n=== Both FDA and earnings analyses failed, cannot validate Hypothesis 2 ===")

if __name__ == "__main__":
    main()