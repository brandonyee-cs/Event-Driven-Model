import pandas as pd
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import traceback
import polars as pl
from argparse import ArgumentParser

# --- Configuration ---
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path: sys.path.append(current_dir)
try: 
    from event_processor import EventDataLoader, EventFeatureEngineer, EventAnalysis
    print("Successfully imported Event processor classes.")
except ImportError as e: 
    print(f"Error importing from event_processor: {e}")
    print("Ensure 'event_processor.py' and 'models.py' are in the same directory or Python path.")
    print("Ensure Polars is installed: pip install polars pyarrow")
    sys.exit(1)

pl.Config.set_engine_affinity(engine="streaming")

def parse_arguments():
    """Parse command line arguments"""
    parser = ArgumentParser(description='Run event study analysis')
    parser.add_argument('--event-file', type=str, required=True, 
                       help='Path to the event data CSV file')
    parser.add_argument('--stock-files', type=str, nargs='+', required=True,
                       help='Paths to the stock data PARQUET files')
    parser.add_argument('--results-dir', type=str, default='results_event',
                       help='Directory to save results')
    parser.add_argument('--file-prefix', type=str, default='event',
                       help='Prefix for saved files')
    parser.add_argument('--window-days', type=int, default=60,
                       help='Number of days before/after event to analyze')
    parser.add_argument('--event-date-col', type=str, default='Event Date',
                       help='Column name in event file containing event dates')
    parser.add_argument('--ticker-col', type=str, default='ticker',
                       help='Column name in event file containing ticker symbols')
    parser.add_argument('--ml-window', type=int, default=3,
                       help='Target prediction window for ML models (days)')
    parser.add_argument('--test-split', type=float, default=0.2,
                       help='Proportion of data to use for test set')
    parser.add_argument('--run-ml', action='store_true',
                       help='Run machine learning analysis')
    
    # Analysis parameters
    parser.add_argument('--vol-window', type=int, default=5,
                       help='Window size for rolling volatility calculation')
    parser.add_argument('--vol-pre-days', type=int, default=60,
                       help='Days before event for volatility analysis')
    parser.add_argument('--vol-post-days', type=int, default=60,
                       help='Days after event for volatility analysis')
    parser.add_argument('--vol-baseline-start', type=int, default=-60,
                       help='Start of baseline window for volatility comparison')
    parser.add_argument('--vol-baseline-end', type=int, default=-11,
                       help='End of baseline window for volatility comparison')
    parser.add_argument('--vol-event-start', type=int, default=-2,
                       help='Start of event window for volatility comparison')
    parser.add_argument('--vol-event-end', type=int, default=2,
                       help='End of event window for volatility comparison')
    
    # Sharpe analysis parameters
    parser.add_argument('--sharpe-window', type=int, default=5,
                       help='Window size for rolling Sharpe calculation')
    parser.add_argument('--sharpe-analysis-start', type=int, default=-60,
                       help='Start of window for Sharpe analysis')
    parser.add_argument('--sharpe-analysis-end', type=int, default=60,
                       help='End of window for Sharpe analysis')
    parser.add_argument('--sharpe-lookback', type=int, default=10,
                       help='Lookback period for Sharpe quantile calculation')
    
    return parser.parse_args()

def run_event_analysis(args):
    """
    Runs the event analysis pipeline using the specified parameters.
    """
    print("--- Starting Event Analysis ---")

    # --- Path Validation & Results Dir Creation ---
    if not os.path.exists(args.event_file): 
        print(f"\n*** Error: Event file not found: {args.event_file} ***")
        return
    
    missing_stock = [f for f in args.stock_files if not os.path.exists(f)]
    if missing_stock: 
        print(f"\n*** Error: Stock file(s) not found: {missing_stock} ***")
        return
    
    print("File paths validated.")
    try:
        os.makedirs(args.results_dir, exist_ok=True)
        print(f"Results will be saved to: {os.path.abspath(args.results_dir)}")
    except OSError as oe:
        print(f"\n*** Error creating results directory '{args.results_dir}': {oe} ***")
        return

    try:
        # --- Initialize Components ---
        print("\nInitializing components...")
        data_loader = EventDataLoader(
            event_path=args.event_file, 
            stock_paths=args.stock_files, 
            window_days=args.window_days,
            event_date_col=args.event_date_col,
            ticker_col=args.ticker_col
        )
        feature_engineer = EventFeatureEngineer(prediction_window=args.ml_window)
        analyzer = EventAnalysis(data_loader, feature_engineer)
        print("Components initialized.")

        # --- Load Data ---
        print("\nLoading and preparing data...")
        analyzer.data = analyzer.load_and_prepare_data(run_feature_engineering=(args.run_ml))
        if analyzer.data is None or analyzer.data.is_empty(): 
            print("\n*** Error: Data loading failed. ***")
            return
            
        print(f"Data loaded successfully. Shape: {analyzer.data.shape}")
        
        # Setup analysis parameters
        baseline_window = (args.vol_baseline_start, args.vol_baseline_end)
        event_window = (args.vol_event_start, args.vol_event_end)
        analysis_window = (args.sharpe_analysis_start, args.sharpe_analysis_end)
        
        # --- Run Volatility Spike Analysis ---
        analyzer.analyze_volatility_spikes(
            results_dir=args.results_dir,
            file_prefix=args.file_prefix,
            window=args.vol_window,
            pre_days=args.vol_pre_days,
            post_days=args.vol_post_days,
            baseline_window=baseline_window,
            event_window=event_window
        )
        
        # --- Run Rolling Sharpe Time Series Analysis ---
        analyzer.calculate_rolling_sharpe_timeseries(
            results_dir=args.results_dir,
            file_prefix=args.file_prefix,
            return_col='ret',
            analysis_window=analysis_window,
            sharpe_window=args.sharpe_window,
            annualize=True,
        )
        
        # --- Run Sharpe Ratio Quantile Analysis ---
        analyzer.calculate_sharpe_quantiles(
            results_dir=args.results_dir,
            file_prefix=args.file_prefix,
            return_col='ret',
            analysis_window=analysis_window,
            lookback_window=args.sharpe_lookback,
            quantiles=[0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95],
            annualize=True,
        )
        
        # --- Run ML Analysis if requested ---
        if args.run_ml:
            print("\n--- Running Machine Learning Analysis ---")
            # Train models
            analyzer.train_models(test_size=args.test_split, time_split_column=args.event_date_col)
            
            # Evaluate models
            results = analyzer.evaluate_models()
            
            # Plot feature importance
            for model_name in analyzer.models.keys():
                analyzer.plot_feature_importance(
                    results_dir=args.results_dir,
                    file_prefix=args.file_prefix,
                    model_name=model_name
                )
            
            # Plot predictions for sample events
            sample_events = analyzer.find_sample_event_ids(n=3)
            for event_id in sample_events:
                for model_name in analyzer.models.keys():
                    analyzer.plot_predictions_for_event(
                        results_dir=args.results_dir,
                        event_id=event_id,
                        file_prefix=args.file_prefix,
                        model_name=model_name
                    )
        
        print(f"\n--- Event Analysis Finished (Results in '{args.results_dir}') ---")

    except ValueError as ve: 
        print(f"\n*** ValueError: {ve} ***")
        traceback.print_exc()
    except RuntimeError as re: 
        print(f"\n*** RuntimeError: {re} ***")
        traceback.print_exc()
    except FileNotFoundError as fnf: 
        print(f"\n*** FileNotFoundError: {fnf} ***")
    except pl.exceptions.PolarsError as pe: 
        print(f"\n*** PolarsError: {pe} ***")
        traceback.print_exc()
    except Exception as e: 
        print(f"\n*** An unexpected error occurred: {e} ***")
        traceback.print_exc()

def main():
    args = parse_arguments()
    run_event_analysis(args)

if __name__ == "__main__":
    main()