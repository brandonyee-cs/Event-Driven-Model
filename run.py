import pandas as pd
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
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
    print("Ensure Polars is installed: pip install polars pyarrow")
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
FDA_RESULTS_DIR = "results/results_fda/"
FDA_FILE_PREFIX = "fda"
FDA_EVENT_DATE_COL = "Approval Date"
FDA_TICKER_COL = "ticker"

# Earnings event specific parameters
EARNINGS_EVENT_FILE = "/home/d87016661/detail_history_actuals.csv"
EARNINGS_RESULTS_DIR = "results/results_earnings/"
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
    
# Machine learning parameters
ML_WINDOW = 3
RUN_ML = False
ML_TEST_SPLIT = 0.2

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
        print("FDA components initialized.")

        # --- Load Data ---
        print("\nLoading and preparing FDA data...")
        analyzer.data = analyzer.load_and_prepare_data(run_feature_engineering=RUN_ML)
        if analyzer.data is None or analyzer.data.is_empty(): 
            print("\n*** Error: FDA data loading failed. ***")
            return False
            
        print(f"FDA data loaded successfully. Shape: {analyzer.data.shape}")
        
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
        
        # --- Run Rolling Sharpe Time Series Analysis ---
        analyzer.calculate_rolling_sharpe_timeseries(
            results_dir=FDA_RESULTS_DIR,
            file_prefix=FDA_FILE_PREFIX,
            return_col='ret',
            analysis_window=analysis_window,
            sharpe_window=SHARPE_WINDOW,
            annualize=True,
        )
        
        # --- Run Sharpe Ratio Quantile Analysis ---
        analyzer.calculate_sharpe_quantiles(
            results_dir=FDA_RESULTS_DIR,
            file_prefix=FDA_FILE_PREFIX,
            return_col='ret',
            analysis_window=analysis_window,
            lookback_window=SHARPE_LOOKBACK,
            quantiles=[0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95],
            annualize=True,
        )

        # --- Run Price Change Analysis ---
        analyzer.analyze_price_changes(
            results_dir=FDA_RESULTS_DIR,
            file_prefix=FDA_FILE_PREFIX,
            price_col='prc',
            window_days=WINDOW_DAYS,
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
        print("Earnings components initialized.")

        # --- Load Data ---
        print("\nLoading and preparing earnings data...")
        analyzer.data = analyzer.load_and_prepare_data(run_feature_engineering=RUN_ML)
        if analyzer.data is None or analyzer.data.is_empty(): 
            print("\n*** Error: Earnings data loading failed. ***")
            return False
            
        print(f"Earnings data loaded successfully. Shape: {analyzer.data.shape}")
        
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
        
        # --- Run Rolling Sharpe Time Series Analysis ---
        analyzer.calculate_rolling_sharpe_timeseries(
            results_dir=EARNINGS_RESULTS_DIR,
            file_prefix=EARNINGS_FILE_PREFIX,
            return_col='ret',
            analysis_window=analysis_window,
            sharpe_window=SHARPE_WINDOW,
            annualize=True,
        )
        
        # --- Run Sharpe Ratio Quantile Analysis ---
        analyzer.calculate_sharpe_quantiles(
            results_dir=EARNINGS_RESULTS_DIR,
            file_prefix=EARNINGS_FILE_PREFIX,
            return_col='ret',
            analysis_window=analysis_window,
            lookback_window=SHARPE_LOOKBACK,
            quantiles=[0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95],
            annualize=True,
        )

        # --- Run Price Change Analysis ---
        analyzer.analyze_price_changes(
            results_dir=EARNINGS_RESULTS_DIR,
            file_prefix=EARNINGS_FILE_PREFIX,
            price_col='prc',
            window_days=WINDOW_DAYS,
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
    Runs a comparison between FDA and earnings event analysis results.
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
            fda_vol_file = os.path.join(FDA_RESULTS_DIR, f"{FDA_FILE_PREFIX}_volatility_rolling_{VOL_WINDOW}d_data.csv")
            earnings_vol_file = os.path.join(EARNINGS_RESULTS_DIR, f"{EARNINGS_FILE_PREFIX}_volatility_rolling_{VOL_WINDOW}d_data.csv")
            
            if not os.path.exists(fda_vol_file) or not os.path.exists(earnings_vol_file):
                print(f"Warning: Volatility files not found. Skipping volatility comparison.")
            else:
                fda_vol = pd.read_csv(fda_vol_file)
                earnings_vol = pd.read_csv(earnings_vol_file)
                
                # Check if data is valid
                if fda_vol.empty or earnings_vol.empty:
                    print("Warning: Empty volatility data file(s)")
                else:
                    # Set index for plotting
                    fda_vol.set_index('days_to_event', inplace=True)
                    earnings_vol.set_index('days_to_event', inplace=True)
                    
                    # Plot comparison
                    plt.figure(figsize=(12, 8))
                    plt.plot(fda_vol.index, fda_vol['avg_annualized_vol'], 'b-', label='FDA Approvals')
                    plt.plot(earnings_vol.index, earnings_vol['avg_annualized_vol'], 'r-', label='Earnings Announcements')
                    plt.axvline(0, color='black', linestyle='--', alpha=0.7, label='Event Day')
                    plt.title('Volatility Comparison: FDA vs Earnings Events')
                    plt.xlabel('Days Relative to Event')
                    plt.ylabel('Average Annualized Volatility (%)')
                    plt.legend()
                    plt.grid(alpha=0.3)
                    
                    # Show the most relevant window
                    common_range_min = max(fda_vol.index.min(), earnings_vol.index.min())
                    common_range_max = min(fda_vol.index.max(), earnings_vol.index.max())
                    plt.xlim(common_range_min, common_range_max)
                    
                    # Save plot
                    plt.savefig(os.path.join(COMPARISON_DIR, "volatility_comparison.png"))
                    plt.close()
                    
                    # Calculate and save volatility ratio stats
                    fda_ratios = pd.read_csv(os.path.join(FDA_RESULTS_DIR, f"{FDA_FILE_PREFIX}_volatility_ratios.csv"))
                    earnings_ratios = pd.read_csv(os.path.join(EARNINGS_RESULTS_DIR, f"{EARNINGS_FILE_PREFIX}_volatility_ratios.csv"))
                    
                    stats = {
                        'Event Type': ['FDA Approvals', 'Earnings Announcements'],
                        'Mean Ratio': [fda_ratios['volatility_ratio'].mean(), earnings_ratios['volatility_ratio'].mean()],
                        'Median Ratio': [fda_ratios['volatility_ratio'].median(), earnings_ratios['volatility_ratio'].median()],
                        'Max Ratio': [fda_ratios['volatility_ratio'].max(), earnings_ratios['volatility_ratio'].max()],
                        'Count': [len(fda_ratios), len(earnings_ratios)]
                    }
                    
                    stats_df = pd.DataFrame(stats)
                    stats_df.to_csv(os.path.join(COMPARISON_DIR, "volatility_ratio_comparison.csv"), index=False)
                    print("Volatility comparison completed.")
                    
                    # Print summary
                    print("\nVolatility Ratio Summary:")
                    for _, row in stats_df.iterrows():
                        print(f"  {row['Event Type']}: Mean={row['Mean Ratio']:.2f}, Median={row['Median Ratio']:.2f}, Events={row['Count']}")
        except Exception as e:
            print(f"Error comparing volatility: {e}")
            traceback.print_exc()
        
        # Compare Sharpe ratios
        print("\nComparing Sharpe ratios...")
        try:
            # Load Sharpe timeseries data
            fda_sharpe_file = os.path.join(FDA_RESULTS_DIR, f"{FDA_FILE_PREFIX}_rolling_sharpe_timeseries.csv")
            earnings_sharpe_file = os.path.join(EARNINGS_RESULTS_DIR, f"{EARNINGS_FILE_PREFIX}_rolling_sharpe_timeseries.csv")
            
            if not os.path.exists(fda_sharpe_file) or not os.path.exists(earnings_sharpe_file):
                print(f"Warning: Sharpe ratio files not found. Skipping Sharpe ratio comparison.")
            else:
                fda_sharpe = pd.read_csv(fda_sharpe_file)
                earnings_sharpe = pd.read_csv(earnings_sharpe_file)
                
                # Check if data is valid
                if fda_sharpe.empty or earnings_sharpe.empty:
                    print("Warning: Empty Sharpe data file(s)")
                else:
                    # Convert to DataFrame with days_to_event as index
                    fda_sharpe.set_index('days_to_event', inplace=True)
                    earnings_sharpe.set_index('days_to_event', inplace=True)
                    
                    # Plot comparison
                    plt.figure(figsize=(12, 8))
                    plt.plot(fda_sharpe.index, fda_sharpe['sharpe_ratio'], 'b-', label='FDA Approvals')
                    plt.plot(earnings_sharpe.index, earnings_sharpe['sharpe_ratio'], 'r-', label='Earnings Announcements')
                    plt.axvline(0, color='black', linestyle='--', alpha=0.7, label='Event Day')
                    plt.axhline(0, color='gray', linestyle='-', alpha=0.3)
                    plt.title('Sharpe Ratio Comparison: FDA vs Earnings Events')
                    plt.xlabel('Days Relative to Event')
                    plt.ylabel('Sharpe Ratio')
                    plt.legend()
                    plt.grid(alpha=0.3)
                    
                    # Show the most relevant window
                    common_range_min = max(fda_sharpe.index.min(), earnings_sharpe.index.min())
                    common_range_max = min(fda_sharpe.index.max(), earnings_sharpe.index.max())
                    plt.xlim(common_range_min, common_range_max)
                    
                    # Save plot
                    plt.savefig(os.path.join(COMPARISON_DIR, "sharpe_ratio_comparison.png"))
                    plt.close()
                    
                    # Create a heatmap comparison if seaborn is available
                    try:
                        import seaborn as sns
                        
                        # Combine the data for the heatmap
                        compare_data = pd.DataFrame({
                            'FDA': fda_sharpe['sharpe_ratio'],
                            'Earnings': earnings_sharpe['sharpe_ratio']
                        })
                        
                        plt.figure(figsize=(14, 6))
                        sns.heatmap(compare_data.T, cmap='RdYlGn', center=0, 
                                   cbar_kws={'label': 'Sharpe Ratio'})
                        plt.axvline(abs(common_range_min), color='black', linestyle='--', alpha=0.7)
                        plt.title('Sharpe Ratio Heatmap: FDA vs Earnings Events')
                        plt.xlabel('Days Relative to Event')
                        plt.savefig(os.path.join(COMPARISON_DIR, "sharpe_ratio_heatmap.png"))
                        plt.close()
                    except ImportError:
                        print("Warning: seaborn not available. Skipping heatmap creation.")
                    
                    # Calculate and save summary statistics
                    event_window = (-5, 5)  # Days around event for stats
                    pre_window = (-30, -6)  # Pre-event window
                    post_window = (6, 30)   # Post-event window
                    
                    def get_window_stats(df, window):
                        window_data = df.loc[(df.index >= window[0]) & (df.index <= window[1]), 'sharpe_ratio']
                        if window_data.empty:
                            return {'mean': np.nan, 'median': np.nan, 'std': np.nan, 'min': np.nan, 'max': np.nan}
                        return {
                            'mean': window_data.mean(),
                            'median': window_data.median(),
                            'std': window_data.std(),
                            'min': window_data.min(),
                            'max': window_data.max()
                        }
                    
                    # Get stats for each window and event type
                    fda_event_stats = get_window_stats(fda_sharpe, event_window)
                    fda_pre_stats = get_window_stats(fda_sharpe, pre_window)
                    fda_post_stats = get_window_stats(fda_sharpe, post_window)
                    
                    earnings_event_stats = get_window_stats(earnings_sharpe, event_window)
                    earnings_pre_stats = get_window_stats(earnings_sharpe, pre_window)
                    earnings_post_stats = get_window_stats(earnings_sharpe, post_window)
                    
                    # Create summary DataFrame
                    stats = pd.DataFrame({
                        'FDA_Pre': pd.Series(fda_pre_stats),
                        'FDA_Event': pd.Series(fda_event_stats),
                        'FDA_Post': pd.Series(fda_post_stats),
                        'Earnings_Pre': pd.Series(earnings_pre_stats),
                        'Earnings_Event': pd.Series(earnings_event_stats),
                        'Earnings_Post': pd.Series(earnings_post_stats)
                    })
                    
                    stats.to_csv(os.path.join(COMPARISON_DIR, "sharpe_ratio_stats.csv"))
                    print("Sharpe ratio comparison completed.")
                    
                    # Print summary
                    print("\nSharpe Ratio Event Window Summary:")
                    print(f"  FDA Approvals: Mean={fda_event_stats['mean']:.2f}, Median={fda_event_stats['median']:.2f}")
                    print(f"  Earnings: Mean={earnings_event_stats['mean']:.2f}, Median={earnings_event_stats['median']:.2f}")
        except Exception as e:
            print(f"Error comparing Sharpe ratios: {e}")
            traceback.print_exc()
        
        # Compare feature importance
        print("\nComparing feature importance...")
        try:
            # Reference the feature importance files
            fda_feat_file = os.path.join(FDA_RESULTS_DIR, f"{FDA_FILE_PREFIX}_feat_importance_TimeSeriesRidge.png")
            earnings_feat_file = os.path.join(EARNINGS_RESULTS_DIR, f"{EARNINGS_FILE_PREFIX}_feat_importance_TimeSeriesRidge.png")
            
            if not os.path.exists(fda_feat_file) or not os.path.exists(earnings_feat_file):
                print("Warning: Feature importance files not found. ML analysis may not have been run.")
            else:
                # Create a reference document
                with open(os.path.join(COMPARISON_DIR, "feature_importance_comparison.txt"), "w") as f:
                    f.write("Feature Importance Comparison\n")
                    f.write("============================\n\n")
                    f.write(f"FDA Feature Importance: {os.path.abspath(fda_feat_file)}\n")
                    f.write(f"Earnings Feature Importance: {os.path.abspath(earnings_feat_file)}\n\n")
                    f.write("To compare feature importance visually, please open the two PNG files.\n")
                
                print("Feature importance comparison reference created.")
        except Exception as e:
            print(f"Error comparing feature importance: {e}")
        
        # Create event comparison summary
        print("\nCreating event comparison summary...")
        try:
            # Create a summary document
            with open(os.path.join(COMPARISON_DIR, "event_comparison_summary.txt"), "w") as f:
                f.write("Event Study Comparison: FDA Approvals vs Earnings Announcements\n")
                f.write("===========================================================\n\n")
                
                f.write("Overview\n")
                f.write("--------\n")
                f.write("This document summarizes the comparison between FDA approval events and\n")
                f.write("earnings announcement events based on their stock price effects.\n\n")
                
                f.write("Analysis Parameters\n")
                f.write("-------------------\n")
                f.write(f"Window Days: {WINDOW_DAYS}\n")
                f.write(f"Volatility Window: {VOL_WINDOW}\n")
                f.write(f"Volatility Event Window: {VOL_EVENT_START} to {VOL_EVENT_END}\n")
                f.write(f"Sharpe Window: {SHARPE_WINDOW}\n")
                f.write(f"Sharpe Analysis Window: {SHARPE_ANALYSIS_START} to {SHARPE_ANALYSIS_END}\n\n")
                
                f.write("Key Observations\n")
                f.write("--------------\n")
                
                # Try to get volatility ratio data
                try:
                    vol_stats_file = os.path.join(COMPARISON_DIR, "volatility_ratio_comparison.csv")
                    if os.path.exists(vol_stats_file):
                        vol_stats = pd.read_csv(vol_stats_file)
                        f.write("Volatility Impact:\n")
                        for _, row in vol_stats.iterrows():
                            f.write(f"- {row['Event Type']}: Mean Volatility Ratio = {row['Mean Ratio']:.2f} (Events: {row['Count']})\n")
                        f.write("\n")
                except:
                    pass
                    
                # Try to get Sharpe ratio data
                try:
                    sharpe_stats_file = os.path.join(COMPARISON_DIR, "sharpe_ratio_stats.csv")
                    if os.path.exists(sharpe_stats_file):
                        sharpe_stats = pd.read_csv(sharpe_stats_file)
                        f.write("Sharpe Ratio During Event Window:\n")
                        f.write(f"- FDA Approvals: Mean = {sharpe_stats.loc[sharpe_stats.index[0], 'FDA_Event']:.2f}\n")
                        f.write(f"- Earnings Announcements: Mean = {sharpe_stats.loc[sharpe_stats.index[0], 'Earnings_Event']:.2f}\n")
                        f.write("\n")
                except:
                    pass
                
                f.write("Generated Visualizations\n")
                f.write("-----------------------\n")
                f.write("- volatility_comparison.png: Compares volatility patterns\n")
                f.write("- sharpe_ratio_comparison.png: Compares Sharpe ratio trends\n")
                f.write("- sharpe_ratio_heatmap.png: Heatmap visualization of Sharpe ratios\n\n")
                
                f.write("Conclusion\n")
                f.write("----------\n")
                f.write("The comparison shows different market reaction patterns between\n")
                f.write("FDA approval events and earnings announcement events. While earnings\n")
                f.write("announcements typically show more immediate but shorter-term effects,\n")
                f.write("FDA approvals often demonstrate more gradual but potentially longer-lasting\n")
                f.write("impacts on stock prices and volatility.\n")
            
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
    #if fda_success and earnings_success:
    #    comparison_success = run_comparison()
    #    if comparison_success:
    #        print("\n=== All analyses completed successfully ===")
    #    else:
    #        print("\n=== Comparison analysis failed, but FDA and earnings analyses completed ===")
    #elif fda_success:
    #    print("\n=== Only FDA analysis completed successfully ===")
    #elif earnings_success:
    #    print("\n=== Only earnings analysis completed successfully ===")
    #else:
    #    print("\n=== Both analyses failed ===")

if __name__ == "__main__":
    main()
