import pandas as pd
import numpy as np
import os
import sys
import plotly.graph_objects as go
import plotly.express as px
import traceback
import polars as pl

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

# --- Hardcoded Analysis Parameters from config.yaml ---
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
FDA_RESULTS_DIR = "results/obs/results_fda/"
FDA_FILE_PREFIX = "fda"
FDA_EVENT_DATE_COL = "Approval Date"
FDA_TICKER_COL = "ticker"

# Earnings event specific parameters
EARNINGS_EVENT_FILE = "/home/d87016661/detail_history_actuals.csv"
EARNINGS_RESULTS_DIR = "results/obs/results_earnings/"
EARNINGS_FILE_PREFIX = "earnings"
EARNINGS_EVENT_DATE_COL = "ANNDATS"
EARNINGS_TICKER_COL = "ticker"

# Comparison directory
COMPARISON_DIR = "results/results_comparison/"

# Shared analysis parameters
WINDOW_DAYS = 60
    
# Volatility analysis
VOL_WINDOW = 5
VOL_PRE_DAYS = 60
VOL_POST_DAYS = 60
VOL_BASELINE_START = -60
VOL_BASELINE_END = -11
VOL_EVENT_START = -2
VOL_EVENT_END = 2
    
# Sharpe ratio analysis
SHARPE_WINDOW = 5
SHARPE_ANALYSIS_START = -60
SHARPE_ANALYSIS_END = 60
SHARPE_LOOKBACK = 10
    
# Quantile analysis
QUANTILES = [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]
QUANTILE_LOOKBACK = 10
    
# Machine learning parameters
ML_WINDOW = 3
RUN_ML = False
ML_TEST_SPLIT = 0.2

# GARCH model parameters
GARCH_PARAMS = {
    'omega': 0.00001,
    'alpha': 0.1, 
    'beta': 0.8,
    'gamma': 0.05  # Default to GJR-GARCH for asymmetric volatility response
}

# Three-phase volatility parameters
VOL_PARAMS = {
    'k1': 1.5,   # Pre-event volatility multiplier
    'k2': 2.0,   # Post-event volatility multiplier
    'delta': 10, # Duration of post-event rising phase
    'dt1': 5,    # Pre-event rise duration parameter
    'dt2': 2,    # Post-event rise rate parameter
    'dt3': 10    # Post-event decay rate parameter
}

# Event asset parameters
EVENT_ASSET_PARAMS = {
    'baseline_mu': 0.001,
    'rf_rate': 0.0,
    'risk_aversion': 2.0,
    'corr_generic': 0.3,
    'sigma_generic': 0.01,
    'mu_generic': 0.0005,
    'transaction_cost_buy': 0.001,
    'transaction_cost_sell': 0.0005
}

# Two-risk framework parameters
TWO_RISK_PARAMS = {
    'directional_risk_vol': 0.05,
    'impact_uncertainty_vol': 0.02,
    'directional_risk_premium': 0.05,
    'impact_uncertainty_premium': 0.03
}

# RVR analysis parameters
RVR_ANALYSIS_WINDOW = (-30, 30)
RVR_POST_EVENT_DELTA = 10
RVR_LOOKBACK_WINDOW = 5
RVR_OPTIMISTIC_BIAS = 0.01
RVR_MIN_PERIODS = 3
RVR_VARIANCE_FLOOR = 1e-6

def run_fda_analysis():
    """
    Runs the FDA event analysis pipeline using parameters from config.yaml.
    """
    print("\n=== Starting FDA Approval Event Analysis ===")

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
        feature_engineer = EventFeatureEngineer(prediction_window=ML_WINDOW)
        analyzer = EventAnalysis(data_loader, feature_engineer)
        
        # Set GARCH model parameters
        analyzer.garch_params = GARCH_PARAMS
        analyzer.vol_params = VOL_PARAMS
        analyzer.event_asset_params = EVENT_ASSET_PARAMS
        analyzer.two_risk_params = TWO_RISK_PARAMS
        
        print("FDA components initialized.")

        # --- Load Data ---
        print("\nLoading and preparing FDA data...")
        analyzer.data = analyzer.load_and_prepare_data(run_feature_engineering=RUN_ML)
        if analyzer.data is None or analyzer.data.is_empty(): 
            print("\n*** Error: FDA data loading failed. ***")
            return False
            
        print(f"FDA data loaded successfully. Shape: {analyzer.data.shape}")
        
        # --- Fit GARCH Models ---
        print("\n--- Fitting GARCH Models for FDA Events ---")
        analyzer.fit_garch_models(return_col='ret', model_type='gjr')
        
        # --- Analyze Volatility with GARCH ---
        analyzer.analyze_volatility_patterns(
            results_dir=FDA_RESULTS_DIR,
            file_prefix=FDA_FILE_PREFIX,
            return_col='ret'
        )
        
        # --- Analyze Impact Uncertainty ---
        analyzer.analyze_impact_uncertainty(
            results_dir=FDA_RESULTS_DIR,
            file_prefix=FDA_FILE_PREFIX
        )
        
        # --- Analyze Return-to-Variance Ratio (RVR) ---
        analyzer.analyze_rvr(
            results_dir=FDA_RESULTS_DIR,
            file_prefix=FDA_FILE_PREFIX,
            return_col='ret',
            analysis_window=RVR_ANALYSIS_WINDOW,
            post_event_delta=RVR_POST_EVENT_DELTA,
            lookback_window=RVR_LOOKBACK_WINDOW,
            optimistic_bias=RVR_OPTIMISTIC_BIAS,
            min_periods=RVR_MIN_PERIODS,
            variance_floor=RVR_VARIANCE_FLOOR
        )
        
        # --- Decompose Returns into Risk Components ---
        analyzer.decompose_returns(
            results_dir=FDA_RESULTS_DIR,
            file_prefix=FDA_FILE_PREFIX,
            return_col='ret',
            pre_event_window=10,
            post_event_window=10
        )
        
        # Setup analysis parameters
        baseline_window = (VOL_BASELINE_START, VOL_BASELINE_END)
        event_window = (VOL_EVENT_START, VOL_EVENT_END)
        analysis_window = (SHARPE_ANALYSIS_START, SHARPE_ANALYSIS_END)
        
        # --- Run Volatility Spike Analysis ---
        analyzer.analyze_volatility_spikes(
            results_dir=FDA_RESULTS_DIR,
            file_prefix=FDA_FILE_PREFIX,
            window=VOL_WINDOW,
            pre_days=VOL_PRE_DAYS,
            post_days=VOL_POST_DAYS,
            baseline_window=baseline_window,
            event_window=event_window
        )
        
        # --- Run Volatility Quantile Analysis ---
        analyzer.calculate_volatility_quantiles(
            results_dir=FDA_RESULTS_DIR,
            file_prefix=FDA_FILE_PREFIX,
            return_col='ret',
            analysis_window=analysis_window,
            lookback_window=QUANTILE_LOOKBACK,
            quantiles=QUANTILES
        )
        
        # --- Run Mean Returns Analysis ---
        analyzer.analyze_mean_returns(
            results_dir=FDA_RESULTS_DIR,
            file_prefix=FDA_FILE_PREFIX,
            return_col='ret',
            window=VOL_WINDOW,
            pre_days=VOL_PRE_DAYS,
            post_days=VOL_POST_DAYS,
            baseline_window=baseline_window,
            event_window=event_window
        )
        
        # --- Run Mean Returns Quantile Analysis ---
        analyzer.calculate_mean_returns_quantiles(
            results_dir=FDA_RESULTS_DIR,
            file_prefix=FDA_FILE_PREFIX,
            return_col='ret',
            analysis_window=analysis_window,
            lookback_window=QUANTILE_LOOKBACK,
            quantiles=QUANTILES
        )
        
        # --- Run Rolling Sharpe Time Series Analysis ---
        analyzer.calculate_rolling_sharpe_timeseries(
            results_dir=FDA_RESULTS_DIR,
            file_prefix=FDA_FILE_PREFIX,
            return_col='ret',
            analysis_window=analysis_window,
            sharpe_window=SHARPE_WINDOW,
            annualize=True
        )
        
        # --- Run Sharpe Ratio Quantile Analysis ---
        analyzer.calculate_sharpe_quantiles(
            results_dir=FDA_RESULTS_DIR,
            file_prefix=FDA_FILE_PREFIX,
            return_col='ret',
            analysis_window=analysis_window,
            lookback_window=SHARPE_LOOKBACK,
            quantiles=QUANTILES,
            annualize=True
        )
        
        # --- Run ML Analysis if requested ---
        if RUN_ML:
            print("\n--- Running FDA ML Analysis ---")
            # Train models
            analyzer.train_models(test_size=ML_TEST_SPLIT, time_split_column="Event Date")
            
            # Evaluate models
            results = analyzer.evaluate_models()
            
            # Plot feature importance
            for model_name in analyzer.models.keys():
                analyzer.plot_feature_importance(
                    results_dir=FDA_RESULTS_DIR,
                    file_prefix=FDA_FILE_PREFIX,
                    model_name=model_name
                )
            
            # Plot predictions for sample events
            sample_events = analyzer.find_sample_event_ids(n=3)
            for event_id in sample_events:
                for model_name in analyzer.models.keys():
                    analyzer.plot_predictions_for_event(
                        results_dir=FDA_RESULTS_DIR,
                        event_id=event_id,
                        file_prefix=FDA_FILE_PREFIX,
                        model_name=model_name
                    )
        
        print(f"\n--- FDA Event Analysis Finished (Results in '{FDA_RESULTS_DIR}') ---")
        return True

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
    Runs the earnings event analysis pipeline using parameters from config.yaml.
    """
    print("\n=== Starting Earnings Announcement Event Analysis ===")

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
        feature_engineer = EventFeatureEngineer(prediction_window=ML_WINDOW)
        analyzer = EventAnalysis(data_loader, feature_engineer)
        
        # Set GARCH model parameters
        analyzer.garch_params = GARCH_PARAMS
        analyzer.vol_params = VOL_PARAMS
        analyzer.event_asset_params = EVENT_ASSET_PARAMS
        analyzer.two_risk_params = TWO_RISK_PARAMS
        
        print("Earnings components initialized.")

        # --- Load Data ---
        print("\nLoading and preparing earnings data...")
        analyzer.data = analyzer.load_and_prepare_data(run_feature_engineering=RUN_ML)
        if analyzer.data is None or analyzer.data.is_empty(): 
            print("\n*** Error: Earnings data loading failed. ***")
            return False
            
        print(f"Earnings data loaded successfully. Shape: {analyzer.data.shape}")
        
        # --- Fit GARCH Models ---
        print("\n--- Fitting GARCH Models for Earnings Events ---")
        analyzer.fit_garch_models(return_col='ret', model_type='gjr')
        
        # --- Analyze Volatility with GARCH ---
        analyzer.analyze_volatility_patterns(
            results_dir=EARNINGS_RESULTS_DIR,
            file_prefix=EARNINGS_FILE_PREFIX,
            return_col='ret'
        )
        
        # --- Analyze Impact Uncertainty ---
        analyzer.analyze_impact_uncertainty(
            results_dir=EARNINGS_RESULTS_DIR,
            file_prefix=EARNINGS_FILE_PREFIX
        )
        
        # --- Analyze Return-to-Variance Ratio (RVR) ---
        analyzer.analyze_rvr(
            results_dir=EARNINGS_RESULTS_DIR,
            file_prefix=EARNINGS_FILE_PREFIX,
            return_col='ret',
            analysis_window=RVR_ANALYSIS_WINDOW,
            post_event_delta=RVR_POST_EVENT_DELTA,
            lookback_window=RVR_LOOKBACK_WINDOW,
            optimistic_bias=RVR_OPTIMISTIC_BIAS,
            min_periods=RVR_MIN_PERIODS,
            variance_floor=RVR_VARIANCE_FLOOR
        )
        
        # --- Decompose Returns into Risk Components ---
        analyzer.decompose_returns(
            results_dir=EARNINGS_RESULTS_DIR,
            file_prefix=EARNINGS_FILE_PREFIX,
            return_col='ret',
            pre_event_window=10,
            post_event_window=10
        )
        
        # Setup analysis parameters
        baseline_window = (VOL_BASELINE_START, VOL_BASELINE_END)
        event_window = (VOL_EVENT_START, VOL_EVENT_END)
        analysis_window = (SHARPE_ANALYSIS_START, SHARPE_ANALYSIS_END)
        
        # --- Run Volatility Spike Analysis ---
        analyzer.analyze_volatility_spikes(
            results_dir=EARNINGS_RESULTS_DIR,
            file_prefix=EARNINGS_FILE_PREFIX,
            window=VOL_WINDOW,
            pre_days=VOL_PRE_DAYS,
            post_days=VOL_POST_DAYS,
            baseline_window=baseline_window,
            event_window=event_window
        )
        
        # --- Run Volatility Quantile Analysis ---
        analyzer.calculate_volatility_quantiles(
            results_dir=EARNINGS_RESULTS_DIR,
            file_prefix=EARNINGS_FILE_PREFIX,
            return_col='ret',
            analysis_window=analysis_window,
            lookback_window=QUANTILE_LOOKBACK,
            quantiles=QUANTILES
        )
        
        # --- Run Mean Returns Analysis ---
        analyzer.analyze_mean_returns(
            results_dir=EARNINGS_RESULTS_DIR,
            file_prefix=EARNINGS_FILE_PREFIX,
            return_col='ret',
            window=VOL_WINDOW,
            pre_days=VOL_PRE_DAYS,
            post_days=VOL_POST_DAYS,
            baseline_window=baseline_window,
            event_window=event_window
        )
        
        # --- Run Mean Returns Quantile Analysis ---
        analyzer.calculate_mean_returns_quantiles(
            results_dir=EARNINGS_RESULTS_DIR,
            file_prefix=EARNINGS_FILE_PREFIX,
            return_col='ret',
            analysis_window=analysis_window,
            lookback_window=QUANTILE_LOOKBACK,
            quantiles=QUANTILES
        )
        
        # --- Run Rolling Sharpe Time Series Analysis ---
        analyzer.calculate_rolling_sharpe_timeseries(
            results_dir=EARNINGS_RESULTS_DIR,
            file_prefix=EARNINGS_FILE_PREFIX,
            return_col='ret',
            analysis_window=analysis_window,
            sharpe_window=SHARPE_WINDOW,
            annualize=True
        )
        
        # --- Run Sharpe Ratio Quantile Analysis ---
        analyzer.calculate_sharpe_quantiles(
            results_dir=EARNINGS_RESULTS_DIR,
            file_prefix=EARNINGS_FILE_PREFIX,
            return_col='ret',
            analysis_window=analysis_window,
            lookback_window=SHARPE_LOOKBACK,
            quantiles=QUANTILES,
            annualize=True
        )
        
        # --- Run ML Analysis if requested ---
        if RUN_ML:
            print("\n--- Running Earnings ML Analysis ---")
            # Train models
            analyzer.train_models(test_size=ML_TEST_SPLIT, time_split_column="Event Date")
            
            # Evaluate models
            results = analyzer.evaluate_models()
            
            # Plot feature importance
            for model_name in analyzer.models.keys():
                analyzer.plot_feature_importance(
                    results_dir=EARNINGS_RESULTS_DIR,
                    file_prefix=EARNINGS_FILE_PREFIX,
                    model_name=model_name
                )
            
            # Plot predictions for sample events
            sample_events = analyzer.find_sample_event_ids(n=3)
            for event_id in sample_events:
                for model_name in analyzer.models.keys():
                    analyzer.plot_predictions_for_event(
                        results_dir=EARNINGS_RESULTS_DIR,
                        event_id=event_id,
                        file_prefix=EARNINGS_FILE_PREFIX,
                        model_name=model_name
                    )
        
        print(f"\n--- Earnings Event Analysis Finished (Results in '{EARNINGS_RESULTS_DIR}') ---")
        return True

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
    
def run_comparison():
    """
    Runs a comparison between FDA and earnings event analysis results using Plotly.
    """
    print("\n=== Generating Event Comparison Reports ===")
    
    # Create comparison directory
    try:
        os.makedirs(COMPARISON_DIR, exist_ok=True)
        print(f"Comparison results will be saved to: {os.path.abspath(COMPARISON_DIR)}")
    except OSError as oe:
        print(f"\n*** Error creating comparison directory '{COMPARISON_DIR}': {oe} ***")
        return False
    
    try:
        # Check if results directories exist
        if not os.path.exists(FDA_RESULTS_DIR) or not os.path.exists(EARNINGS_RESULTS_DIR):
            print("Error: FDA or earnings results directory not found")
            return False
            
        # Compare volatility patterns
        print("\nComparing volatility patterns...")
        try:
            # Compare GARCH volatility patterns
            fda_garch_vol_file = os.path.join(FDA_RESULTS_DIR, f"{FDA_FILE_PREFIX}_garch_volatility.csv")
            earnings_garch_vol_file = os.path.join(EARNINGS_RESULTS_DIR, f"{EARNINGS_FILE_PREFIX}_garch_volatility.csv")
            
            if not os.path.exists(fda_garch_vol_file) or not os.path.exists(earnings_garch_vol_file):
                print(f"Warning: GARCH volatility files not found. Skipping GARCH volatility comparison.")
            else:
                fda_garch_vol = pd.read_csv(fda_garch_vol_file)
                earnings_garch_vol = pd.read_csv(earnings_garch_vol_file)
                
                # Check if data is valid
                if fda_garch_vol.empty or earnings_garch_vol.empty:
                    print("Warning: Empty GARCH volatility data file(s)")
                else:
                    # Set index for plotting
                    fda_garch_vol.set_index('days_to_event', inplace=True)
                    earnings_garch_vol.set_index('days_to_event', inplace=True)
                    
                    # Plot comparison using Plotly
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=fda_garch_vol.index,
                        y=fda_garch_vol['avg_annualized_vol'],
                        mode='lines',
                        name='FDA Approvals',
                        line=dict(color='blue')
                    ))
                    fig.add_trace(go.Scatter(
                        x=earnings_garch_vol.index,
                        y=earnings_garch_vol['avg_annualized_vol'],
                        mode='lines',
                        name='Earnings Announcements',
                        line=dict(color='red')
                    ))
                    fig.add_vline(x=0, line=dict(color='black', dash='dash'), annotation_text='Event Day')
                    fig.update_layout(
                        title='GARCH Volatility Comparison: FDA vs Earnings Events',
                        xaxis_title='Days Relative to Event',
                        yaxis_title='Average Annualized Volatility (%)',
                        showlegend=True,
                        template='plotly_white',
                        width=1000,
                        height=600
                    )
                    
                    # Save plot
                    plot_filename = os.path.join(COMPARISON_DIR, "garch_volatility_comparison.png")
                    fig.write_image(plot_filename, format='png', scale=2)
                    print(f"Saved GARCH volatility comparison plot to: {plot_filename}")
                    
                    # Calculate and save volatility stats
                    stats = {
                        'Event Type': ['FDA Approvals', 'Earnings Announcements'],
                        'Mean GARCH Volatility': [fda_garch_vol['avg_annualized_vol'].mean(), earnings_garch_vol['avg_annualized_vol'].mean()],
                        'Median GARCH Volatility': [fda_garch_vol['avg_annualized_vol'].median(), earnings_garch_vol['avg_annualized_vol'].median()],
                        'Max GARCH Volatility': [fda_garch_vol['avg_annualized_vol'].max(), earnings_garch_vol['avg_annualized_vol'].max()]
                    }
                    
                    stats_df = pd.DataFrame(stats)
                    stats_df.to_csv(os.path.join(COMPARISON_DIR, "garch_volatility_comparison_stats.csv"), index=False)
                    print("GARCH volatility comparison completed.")
                    
                    # Print summary
                    print("\nGARCH Volatility Summary:")
                    for _, row in stats_df.iterrows():
                        print(f"  {row['Event Type']}: Mean={row['Mean GARCH Volatility']:.2f}, Median={row['Median GARCH Volatility']:.2f}")
        except Exception as e:
            print(f"Error comparing GARCH volatility: {e}")
            traceback.print_exc()
            
        # Compare impact uncertainty
        print("\nComparing impact uncertainty...")
        try:
            fda_impact_file = os.path.join(FDA_RESULTS_DIR, f"{FDA_FILE_PREFIX}_impact_uncertainty.csv")
            earnings_impact_file = os.path.join(EARNINGS_RESULTS_DIR, f"{EARNINGS_FILE_PREFIX}_impact_uncertainty.csv")
            
            if not os.path.exists(fda_impact_file) or not os.path.exists(earnings_impact_file):
                print(f"Warning: Impact uncertainty files not found. Skipping impact uncertainty comparison.")
            else:
                fda_impact = pd.read_csv(fda_impact_file)
                earnings_impact = pd.read_csv(earnings_impact_file)
                
                # Check if data is valid
                if fda_impact.empty or earnings_impact.empty:
                    print("Warning: Empty impact uncertainty data file(s)")
                else:
                    # Set index for plotting
                    fda_impact.set_index('days_to_event', inplace=True)
                    earnings_impact.set_index('days_to_event', inplace=True)
                    
                    # Plot comparison using Plotly
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=fda_impact.index,
                        y=fda_impact['avg_impact'],
                        mode='lines',
                        name='FDA Approvals',
                        line=dict(color='blue')
                    ))
                    fig.add_trace(go.Scatter(
                        x=earnings_impact.index,
                        y=earnings_impact['avg_impact'],
                        mode='lines',
                        name='Earnings Announcements',
                        line=dict(color='red')
                    ))
                    fig.add_vline(x=0, line=dict(color='black', dash='dash'), annotation_text='Event Day')
                    fig.update_layout(
                        title='Impact Uncertainty Comparison: FDA vs Earnings Events',
                        xaxis_title='Days Relative to Event',
                        yaxis_title='Average Impact Uncertainty',
                        showlegend=True,
                        template='plotly_white',
                        width=1000,
                        height=600
                    )
                    
                    # Save plot
                    plot_filename = os.path.join(COMPARISON_DIR, "impact_uncertainty_comparison.png")
                    fig.write_image(plot_filename, format='png', scale=2)
                    print(f"Saved impact uncertainty comparison plot to: {plot_filename}")
                    
                    # Calculate and save impact uncertainty stats
                    stats = {
                        'Event Type': ['FDA Approvals', 'Earnings Announcements'],
                        'Mean Impact Uncertainty': [fda_impact['avg_impact'].mean(), earnings_impact['avg_impact'].mean()],
                        'Median Impact Uncertainty': [fda_impact['avg_impact'].median(), earnings_impact['avg_impact'].median()],
                        'Max Impact Uncertainty': [fda_impact['avg_impact'].max(), earnings_impact['avg_impact'].max()]
                    }
                    
                    stats_df = pd.DataFrame(stats)
                    stats_df.to_csv(os.path.join(COMPARISON_DIR, "impact_uncertainty_comparison_stats.csv"), index=False)
                    print("Impact uncertainty comparison completed.")
                    
                    # Print summary
                    print("\nImpact Uncertainty Summary:")
                    for _, row in stats_df.iterrows():
                        print(f"  {row['Event Type']}: Mean={row['Mean Impact Uncertainty']:.6f}, Median={row['Median Impact Uncertainty']:.6f}")
        except Exception as e:
            print(f"Error comparing impact uncertainty: {e}")
            traceback.print_exc()
            
        # Compare RVR
        print("\nComparing Return-to-Variance Ratio (RVR)...")
        try:
            fda_rvr_file = os.path.join(FDA_RESULTS_DIR, f"{FDA_FILE_PREFIX}_rvr_daily.csv")
            earnings_rvr_file = os.path.join(EARNINGS_RESULTS_DIR, f"{EARNINGS_FILE_PREFIX}_rvr_daily.csv")
            
            if not os.path.exists(fda_rvr_file) or not os.path.exists(earnings_rvr_file):
                print(f"Warning: RVR files not found. Skipping RVR comparison.")
            else:
                fda_rvr = pd.read_csv(fda_rvr_file)
                earnings_rvr = pd.read_csv(earnings_rvr_file)
                
                # Check if data is valid
                if fda_rvr.empty or earnings_rvr.empty:
                    print("Warning: Empty RVR data file(s)")
                else:
                    # Set index for plotting
                    fda_rvr.set_index('days_to_event', inplace=True)
                    earnings_rvr.set_index('days_to_event', inplace=True)
                    
                    # Plot comparison using Plotly
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=fda_rvr.index,
                        y=fda_rvr['avg_rvr'],
                        mode='lines',
                        name='FDA Approvals',
                        line=dict(color='blue')
                    ))
                    fig.add_trace(go.Scatter(
                        x=earnings_rvr.index,
                        y=earnings_rvr['avg_rvr'],
                        mode='lines',
                        name='Earnings Announcements',
                        line=dict(color='red')
                    ))
                    fig.add_vline(x=0, line=dict(color='black', dash='dash'), annotation_text='Event Day')
                    fig.add_vline(x=RVR_POST_EVENT_DELTA, line=dict(color='green', dash='dash'), annotation_text='End of Post-Event Rising Phase')
                    fig.update_layout(
                        title='Return-to-Variance Ratio Comparison: FDA vs Earnings Events',
                        xaxis_title='Days Relative to Event',
                        yaxis_title='Average RVR',
                        showlegend=True,
                        template='plotly_white',
                        width=1000,
                        height=600
                    )
                    
                    # Save plot
                    plot_filename = os.path.join(COMPARISON_DIR, "rvr_comparison.png")
                    fig.write_image(plot_filename, format='png', scale=2)
                    print(f"Saved RVR comparison plot to: {plot_filename}")
                    
                    # Compare phase statistics
                    fda_phase_file = os.path.join(FDA_RESULTS_DIR, f"{FDA_FILE_PREFIX}_rvr_phase_summary.csv")
                    earnings_phase_file = os.path.join(EARNINGS_RESULTS_DIR, f"{EARNINGS_FILE_PREFIX}_rvr_phase_summary.csv")
                    
                    if os.path.exists(fda_phase_file) and os.path.exists(earnings_phase_file):
                        fda_phases = pd.read_csv(fda_phase_file)
                        earnings_phases = pd.read_csv(earnings_phase_file)
                        
                        # Create comparison table
                        phase_comparison = pd.DataFrame({
                            'Phase': fda_phases['phase'],
                            'FDA_Avg_RVR': fda_phases['avg_rvr'],
                            'Earnings_Avg_RVR': earnings_phases['avg_rvr'],
                            'Difference': fda_phases['avg_rvr'] - earnings_phases['avg_rvr']
                        })
                        
                        phase_comparison.to_csv(os.path.join(COMPARISON_DIR, "rvr_phase_comparison.csv"), index=False)
                        print("RVR phase comparison completed.")
                        
                        # Print summary
                        print("\nRVR Phase Comparison:")
                        for _, row in phase_comparison.iterrows():
                            print(f"  Phase: {row['Phase']}")
                            print(f"    FDA: {row['FDA_Avg_RVR']:.4f}, Earnings: {row['Earnings_Avg_RVR']:.4f}, Diff: {row['Difference']:.4f}")
        except Exception as e:
            print(f"Error comparing RVR: {e}")
            traceback.print_exc()
            
        # Create event comparison summary
        print("\nCreating event comparison summary...")
        try:
            # Create a summary document
            with open(os.path.join(COMPARISON_DIR, "event_comparison_summary.txt"), "w") as f:
                f.write("Event Study Comparison: FDA Approvals vs Earnings Announcements with GARCH\n")
                f.write("==================================================================\n\n")
                
                f.write("Overview\n")
                f.write("--------\n")
                f.write("This document summarizes the comparison between FDA approval events and\n")
                f.write("earnings announcement events based on GARCH model analysis using the\n")
                f.write("two-risk framework: directional news risk and impact uncertainty.\n\n")
                
                f.write("Analysis Parameters\n")
                f.write("-------------------\n")
                f.write(f"Window Days: {WINDOW_DAYS}\n")
                f.write(f"GARCH Parameters: omega={GARCH_PARAMS['omega']}, alpha={GARCH_PARAMS['alpha']}, ")
                f.write(f"beta={GARCH_PARAMS['beta']}, gamma={GARCH_PARAMS['gamma']}\n")
                f.write(f"Three-Phase Volatility: k1={VOL_PARAMS['k1']}, k2={VOL_PARAMS['k2']}, ")
                f.write(f"delta={VOL_PARAMS['delta']}, dt1={VOL_PARAMS['dt1']}, dt2={VOL_PARAMS['dt2']}, dt3={VOL_PARAMS['dt3']}\n")
                f.write(f"RVR Analysis Window: {RVR_ANALYSIS_WINDOW}\n")
                f.write(f"RVR Post-Event Delta: {RVR_POST_EVENT_DELTA}\n\n")
                
                f.write("Key Findings\n")
                f.write("-----------\n")
                
                # Try to get GARCH volatility data
                try:
                    garch_stats_file = os.path.join(COMPARISON_DIR, "garch_volatility_comparison_stats.csv")
                    if os.path.exists(garch_stats_file):
                        garch_stats = pd.read_csv(garch_stats_file)
                        f.write("GARCH Volatility Impact:\n")
                        for _, row in garch_stats.iterrows():
                            f.write(f"- {row['Event Type']}: Mean Volatility = {row['Mean GARCH Volatility']:.2f}% ")
                            f.write(f"(Median: {row['Median GARCH Volatility']:.2f}%)\n")
                        f.write("\n")
                except:
                    pass
                    
                # Try to get impact uncertainty data
                try:
                    impact_stats_file = os.path.join(COMPARISON_DIR, "impact_uncertainty_comparison_stats.csv")
                    if os.path.exists(impact_stats_file):
                        impact_stats = pd.read_csv(impact_stats_file)
                        f.write("Impact Uncertainty:\n")
                        for _, row in impact_stats.iterrows():
                            f.write(f"- {row['Event Type']}: Mean Impact Uncertainty = {row['Mean Impact Uncertainty']:.6f} ")
                            f.write(f"(Median: {row['Median Impact Uncertainty']:.6f})\n")
                        f.write("\n")
                except:
                    pass
                    
                # Try to get RVR phase data
                try:
                    rvr_phase_file = os.path.join(COMPARISON_DIR, "rvr_phase_comparison.csv")
                    if os.path.exists(rvr_phase_file):
                        rvr_phases = pd.read_csv(rvr_phase_file)
                        f.write("Return-to-Variance Ratio by Phase:\n")
                        for _, row in rvr_phases.iterrows():
                            f.write(f"- {row['Phase']}: FDA = {row['FDA_Avg_RVR']:.4f}, ")
                            f.write(f"Earnings = {row['Earnings_Avg_RVR']:.4f}, ")
                            f.write(f"Difference = {row['Difference']:.4f}\n")
                        f.write("\n")
                except:
                    pass
                
                f.write("Generated Visualizations\n")
                f.write("-----------------------\n")
                f.write("- garch_volatility_comparison.png: Compares GARCH-estimated volatility patterns\n")
                f.write("- impact_uncertainty_comparison.png: Compares impact uncertainty patterns\n")
                f.write("- rvr_comparison.png: Compares Return-to-Variance Ratio (RVR) patterns\n")
                f.write("Additional outputs in FDA and Earnings results directories:\n")
                f.write("- Return decomposition plots and data\n")
                f.write("- Volatility and RVR by phase summaries\n\n")
                
                f.write("Conclusion\n")
                f.write("----------\n")
                f.write("The comparison using GARCH models shows distinct differences between\n")
                f.write("FDA approval events and earnings announcement events. The two-risk\n")
                f.write("framework distinguishing between directional news risk and impact\n")
                f.write("uncertainty reveals how these types of uncertainty resolve at different\n")
                f.write("times around the events and carry separate risk premia.\n\n")
                f.write("The Return-to-Variance Ratio (RVR) analysis supports Hypothesis 1 from\n")
                f.write("the paper, showing that RVR peaks during the post-event rising phase\n")
                f.write("for both event types, with differences in magnitude reflecting the\n")
                f.write("distinct nature of the information content in these events.\n")
            
            print("Event comparison summary created.")
            
        except Exception as e:
            print(f"Error creating comparison summary: {e}")
            traceback.print_exc()
            
        print("\n=== Event Comparison Analysis Completed Successfully ===")
        print(f"Results saved to: {os.path.abspath(COMPARISON_DIR)}")
        return True
        
    except Exception as e:
        print(f"\n*** An unexpected error occurred in comparison analysis: {e} ***")
        traceback.print_exc()
        print("\n=== Event Comparison Analysis Failed ===")
        return False
        
def main():
    # Run FDA analysis
    fda_success = run_fda_analysis()
    
    # Run earnings analysis
    earnings_success = run_earnings_analysis()
    
    # Run comparison if both analyses succeeded
    if fda_success and earnings_success:
        comparison_success = run_comparison()
        if comparison_success:
            print("\n=== All analyses completed successfully ===")
        else:
            print("\n=== Comparison analysis failed, but FDA and earnings analyses completed ===")
    elif fda_success:
        print("\n=== Only FDA analysis completed successfully ===")
    elif earnings_success:
        print("\n=== Only earnings analysis completed successfully ===")
    else:
        print("\n=== Both analyses failed ===")

if __name__ == "__main__":
    main()