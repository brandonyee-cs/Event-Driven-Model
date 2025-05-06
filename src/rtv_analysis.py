import os
import polars as pl
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
import pandas as pd
from typing import List, Optional, Tuple, Dict, Any

# Import from existing modules
try:
    from src.event_processor import EventDataLoader, EventAnalysis
    from src.models import TimeSeriesRidge, XGBoostDecileModel
except ImportError:
    print("Error: Could not import required modules.")
    print("Make sure src/event_processor.py and src/models.py are in the correct path.")
    import sys
    sys.exit(1)

pl.Config.set_engine_affinity(engine="streaming")

class ReturnToVarianceAnalysis:
    """Extends event analysis to calculate return-to-variance ratios for testing Hypothesis 1."""
    
    def __init__(self, event_analysis: EventAnalysis):
        """
        Initialize with an existing EventAnalysis instance.
        
        Parameters:
        event_analysis (EventAnalysis): An initialized and data-loaded EventAnalysis instance
        """
        self.event_analysis = event_analysis
        self.data = event_analysis.data
        
    def calculate_rtv_ratio(self, 
                           return_col: str = 'ret',
                           rolling_window: int = 5,
                           min_periods: int = 3,
                           pre_days: int = 60,
                           post_days: int = 60,
                           delta_days: int = 10) -> pl.DataFrame:
        """
        Calculate return-to-variance ratio across the three event phases.
        
        Parameters:
        return_col (str): Column name containing daily returns
        rolling_window (int): Window size for rolling calculations
        min_periods (int): Minimum observations in window
        pre_days (int): Days before event to analyze
        post_days (int): Days after event to analyze
        delta_days (int): Parameter δ from the paper defining the post-event rising phase
        
        Returns:
        pl.DataFrame: DataFrame containing return-to-variance ratios by phase
        """
        print(f"\n--- Calculating Return-to-Variance Ratios (Window={rolling_window} days) ---")
        
        if self.data is None or return_col not in self.data.columns:
            print("Error: Data not loaded or missing required columns.")
            return None
            
        # Filter data to analysis period
        df = self.data.filter(
            (pl.col('days_to_event') >= -pre_days) & 
            (pl.col('days_to_event') <= post_days)
        ).sort(['event_id', 'days_to_event'])
        
        # Clip extreme returns to prevent outliers from skewing results
        df = df.with_columns(
            pl.col(return_col).clip(-0.25, 0.25).alias('clipped_return')
        )
        
        # Calculate rolling means and variances by event
        df = df.with_columns([
            pl.col('clipped_return').rolling_mean(
                window_size=rolling_window, 
                min_periods=min_periods
            ).over('event_id').alias('rolling_mean'),
            
            pl.col('clipped_return').rolling_var(
                window_size=rolling_window, 
                min_periods=min_periods
            ).over('event_id').alias('rolling_var')
        ])
        
        # Calculate return-to-variance ratio (avoiding division by zero)
        df = df.with_columns(
            pl.when(pl.col('rolling_var') > 0)
            .then(pl.col('rolling_mean') / pl.col('rolling_var'))
            .otherwise(None)
            .alias('rtv_ratio')
        )
        
        # Identify event phases
        df = df.with_columns([
            # Pre-event phase: t ≤ tevent
            (pl.col('days_to_event') <= 0).alias('is_pre_event'),
            
            # Post-event rising phase: tevent < t ≤ tevent + δ
            ((pl.col('days_to_event') > 0) & 
             (pl.col('days_to_event') <= delta_days)).alias('is_post_event_rising'),
            
            # Post-event decay phase: t > tevent + δ
            (pl.col('days_to_event') > delta_days).alias('is_post_event_decay')
        ])
        
        return df
    
    def analyze_rtv_by_phase(self, 
                            rtv_data: pl.DataFrame, 
                            delta_days: int = 10) -> Dict[str, Any]:
        """
        Analyze return-to-variance ratios across the three event phases.
        
        Parameters:
        rtv_data (pl.DataFrame): DataFrame with return-to-variance ratios
        delta_days (int): Parameter δ defining the post-event rising phase
        
        Returns:
        Dict[str, Any]: Dictionary with phase-by-phase statistics
        """
        print("\n--- Analyzing Return-to-Variance Ratios by Phase ---")
        
        if rtv_data is None or rtv_data.is_empty():
            print("Error: No data available for analysis.")
            return None
        
        # Calculate statistics by phase
        pre_event_stats = rtv_data.filter(pl.col('is_pre_event')).select(
            pl.mean('rtv_ratio').alias('mean_rtv'),
            pl.median('rtv_ratio').alias('median_rtv'),
            pl.std('rtv_ratio').alias('std_rtv'),
            pl.count('rtv_ratio').alias('count')
        )
        
        post_rising_stats = rtv_data.filter(pl.col('is_post_event_rising')).select(
            pl.mean('rtv_ratio').alias('mean_rtv'),
            pl.median('rtv_ratio').alias('median_rtv'),
            pl.std('rtv_ratio').alias('std_rtv'),
            pl.count('rtv_ratio').alias('count')
        )
        
        post_decay_stats = rtv_data.filter(pl.col('is_post_event_decay')).select(
            pl.mean('rtv_ratio').alias('mean_rtv'),
            pl.median('rtv_ratio').alias('median_rtv'),
            pl.std('rtv_ratio').alias('std_rtv'),
            pl.count('rtv_ratio').alias('count')
        )
        
        # Check if hypothesis is supported
        is_supported = False
        if (not pre_event_stats.is_empty() and not post_rising_stats.is_empty() and 
            not post_decay_stats.is_empty()):
            pre_mean = pre_event_stats.select('mean_rtv').item(0, 0)
            post_rising_mean = post_rising_stats.select('mean_rtv').item(0, 0)
            post_decay_mean = post_decay_stats.select('mean_rtv').item(0, 0)
            
            is_supported = (post_rising_mean > pre_mean) and (post_rising_mean > post_decay_mean)
        
        results = {
            'pre_event': pre_event_stats.to_dict(as_series=False) if not pre_event_stats.is_empty() else None,
            'post_event_rising': post_rising_stats.to_dict(as_series=False) if not post_rising_stats.is_empty() else None,
            'post_event_decay': post_decay_stats.to_dict(as_series=False) if not post_decay_stats.is_empty() else None,
            'hypothesis_supported': is_supported,
            'delta_days': delta_days
        }
        
        # Print summary
        print("\nReturn-to-Variance Ratio by Phase:")
        print(f"Pre-Event Phase: Mean={pre_mean:.4f}")
        print(f"Post-Event Rising Phase (1 to {delta_days} days): Mean={post_rising_mean:.4f}")
        print(f"Post-Event Decay Phase (>{delta_days} days): Mean={post_decay_mean:.4f}")
        print(f"Hypothesis 1 Supported: {is_supported}")
        
        return results
    
    def analyze_rtv_timeseries(self, 
                              rtv_data: pl.DataFrame, 
                              results_dir: str, 
                              file_prefix: str = "event",
                              delta_days: int = 10) -> None:
        """
        Create and save time series plot of return-to-variance ratios.
        
        Parameters:
        rtv_data (pl.DataFrame): DataFrame with return-to-variance ratios
        results_dir (str): Directory to save the plot
        file_prefix (str): Prefix for saved files
        delta_days (int): Parameter δ defining the post-event rising phase
        """
        print("\n--- Creating Return-to-Variance Time Series Plot ---")
        
        if rtv_data is None or rtv_data.is_empty():
            print("Error: No data available for plotting.")
            return
        
        # Aggregate RTV ratios by days to event
        agg_rtv = rtv_data.group_by('days_to_event').agg([
            pl.mean('rtv_ratio').alias('mean_rtv'),
            pl.median('rtv_ratio').alias('median_rtv'),
            pl.std('rtv_ratio').alias('std_rtv'),
            pl.count('rtv_ratio').alias('count')
        ]).sort('days_to_event')
        
        # Apply smoothing
        smooth_window = 5
        agg_rtv = agg_rtv.with_columns([
            pl.col('mean_rtv').rolling_mean(
                window_size=smooth_window, 
                min_periods=smooth_window // 2,
                center=True
            ).alias('smooth_mean_rtv')
        ])
        
        # Convert to pandas for plotting with Plotly
        agg_rtv_pd = agg_rtv.to_pandas()
        
        # Create plot
        fig = go.Figure()
        
        # Add RTV ratio line
        fig.add_trace(go.Scatter(
            x=agg_rtv_pd['days_to_event'],
            y=agg_rtv_pd['mean_rtv'],
            mode='lines',
            name='Raw Return-to-Variance Ratio',
            line=dict(color='blue', width=1),
            opacity=0.3
        ))
        
        # Add smoothed RTV ratio line
        fig.add_trace(go.Scatter(
            x=agg_rtv_pd['days_to_event'],
            y=agg_rtv_pd['smooth_mean_rtv'],
            mode='lines',
            name=f'{smooth_window}-Day Smoothed',
            line=dict(color='red', width=2)
        ))
        
        # Add event day line
        fig.add_vline(x=0, line=dict(color='black', dash='dash'), annotation_text='Event Day')
        
        # Highlight post-event rising phase
        fig.add_vrect(
            x0=1, 
            x1=delta_days, 
            fillcolor='yellow', 
            opacity=0.2, 
            line_width=0,
            annotation_text='Post-Event Rising Phase'
        )
        
        # Set layout
        fig.update_layout(
            title='Return-to-Variance Ratio Around Events',
            xaxis_title='Days Relative to Event',
            yaxis_title='Return-to-Variance Ratio',
            showlegend=True,
            template='plotly_white',
            width=1000,
            height=600,
            xaxis=dict(
                tickmode='linear',
                tick0=-60,
                dtick=10,
                gridcolor='lightgray'
            ),
            yaxis=dict(
                gridcolor='lightgray'
            )
        )
        
        # Save plot
        os.makedirs(results_dir, exist_ok=True)
        plot_filename = os.path.join(results_dir, f"{file_prefix}_rtv_ratio_timeseries.png")
        
        try:
            fig.write_image(plot_filename, format='png', scale=2)
            print(f"Saved RTV time series plot to: {plot_filename}")
        except Exception as e:
            print(f"Warning: Could not save plot image: {e}")
            html_filename = os.path.join(results_dir, f"{file_prefix}_rtv_ratio_timeseries.html")
            fig.write_html(html_filename)
            print(f"Saved as HTML (fallback) to: {html_filename}")
        
        # Save data to CSV
        csv_filename = os.path.join(results_dir, f"{file_prefix}_rtv_ratio_data.csv")
        try:
            agg_rtv.write_csv(csv_filename)
            print(f"Saved RTV data to: {csv_filename}")
        except Exception as e:
            print(f"Error saving data: {e}")
    
    def run_hypothesis_1_test(self,
                             return_col: str = 'ret',
                             rolling_window: int = 5,
                             delta_days: int = 10,
                             results_dir: str = "results/hypothesis_1",
                             file_prefix: str = "event") -> Dict[str, Any]:
        """
        Run complete test of Hypothesis 1.
        
        Parameters:
        return_col (str): Column name containing daily returns
        rolling_window (int): Window size for rolling calculations
        delta_days (int): Parameter δ defining the post-event rising phase
        results_dir (str): Directory to save results
        file_prefix (str): Prefix for saved files
        
        Returns:
        Dict[str, Any]: Results of the hypothesis test
        """
        print("\n=== Running Test of Hypothesis 1 ===")
        print(f"Hypothesis 1: The return-to-variance ratio peaks during the post-event rising phase")
        print(f"Using delta = {delta_days} days for post-event rising phase")
        
        # Calculate return-to-variance ratios
        rtv_data = self.calculate_rtv_ratio(
            return_col=return_col,
            rolling_window=rolling_window,
            delta_days=delta_days
        )
        
        if rtv_data is None or rtv_data.is_empty():
            print("Error: Failed to calculate return-to-variance ratios.")
            return {'hypothesis_supported': False, 'error': 'No data available'}
        
        # Analyze by phase
        phase_results = self.analyze_rtv_by_phase(rtv_data, delta_days)
        
        # Create time series plot
        os.makedirs(results_dir, exist_ok=True)
        self.analyze_rtv_timeseries(rtv_data, results_dir, file_prefix, delta_days)
        
        # Save detailed results to CSV
        csv_filename = os.path.join(results_dir, f"{file_prefix}_rtv_phase_results.csv")
        try:
            # Create a DataFrame with the phase results
            results_df = pl.DataFrame({
                'phase': ['Pre-Event', 'Post-Event Rising', 'Post-Event Decay'],
                'mean_rtv': [
                    phase_results['pre_event']['mean_rtv'][0] if phase_results['pre_event'] else None,
                    phase_results['post_event_rising']['mean_rtv'][0] if phase_results['post_event_rising'] else None,
                    phase_results['post_event_decay']['mean_rtv'][0] if phase_results['post_event_decay'] else None
                ],
                'median_rtv': [
                    phase_results['pre_event']['median_rtv'][0] if phase_results['pre_event'] else None,
                    phase_results['post_event_rising']['median_rtv'][0] if phase_results['post_event_rising'] else None,
                    phase_results['post_event_decay']['median_rtv'][0] if phase_results['post_event_decay'] else None
                ],
                'std_rtv': [
                    phase_results['pre_event']['std_rtv'][0] if phase_results['pre_event'] else None,
                    phase_results['post_event_rising']['std_rtv'][0] if phase_results['post_event_rising'] else None,
                    phase_results['post_event_decay']['std_rtv'][0] if phase_results['post_event_decay'] else None
                ],
                'count': [
                    phase_results['pre_event']['count'][0] if phase_results['pre_event'] else None,
                    phase_results['post_event_rising']['count'][0] if phase_results['post_event_rising'] else None,
                    phase_results['post_event_decay']['count'][0] if phase_results['post_event_decay'] else None
                ]
            })
            
            results_df.write_csv(csv_filename)
            print(f"Saved phase results to: {csv_filename}")
        except Exception as e:
            print(f"Error saving results: {e}")
        
        # Create and save summary report
        report_filename = os.path.join(results_dir, f"{file_prefix}_hypothesis_1_report.txt")
        try:
            with open(report_filename, 'w') as f:
                f.write("===== Hypothesis 1 Test Report =====\n\n")
                f.write("Hypothesis 1: The return-to-variance ratio peaks during the post-event rising phase\n")
                f.write(f"Delta (post-event rising phase duration): {delta_days} days\n\n")
                
                f.write("Return-to-Variance Ratio by Phase:\n")
                
                if phase_results['pre_event']:
                    f.write(f"Pre-Event Phase: Mean={phase_results['pre_event']['mean_rtv'][0]:.4f}, ")
                    f.write(f"Median={phase_results['pre_event']['median_rtv'][0]:.4f}, ")
                    f.write(f"N={phase_results['pre_event']['count'][0]}\n")
                
                if phase_results['post_event_rising']:
                    f.write(f"Post-Event Rising Phase (1 to {delta_days} days): Mean={phase_results['post_event_rising']['mean_rtv'][0]:.4f}, ")
                    f.write(f"Median={phase_results['post_event_rising']['median_rtv'][0]:.4f}, ")
                    f.write(f"N={phase_results['post_event_rising']['count'][0]}\n")
                
                if phase_results['post_event_decay']:
                    f.write(f"Post-Event Decay Phase (>{delta_days} days): Mean={phase_results['post_event_decay']['mean_rtv'][0]:.4f}, ")
                    f.write(f"Median={phase_results['post_event_decay']['median_rtv'][0]:.4f}, ")
                    f.write(f"N={phase_results['post_event_decay']['count'][0]}\n")
                
                f.write(f"\nHypothesis 1 Supported: {phase_results['hypothesis_supported']}\n")
                
                if phase_results['hypothesis_supported']:
                    f.write("\nConclusion: The results support Hypothesis 1. The return-to-variance ratio peaks during\n")
                    f.write(f"the post-event rising phase (1 to {delta_days} days after the event), exceeding both the\n")
                    f.write("pre-event and post-event decay phases.\n")
                else:
                    f.write("\nConclusion: The results do not support Hypothesis 1. The return-to-variance ratio\n")
                    f.write(f"does not peak during the post-event rising phase (1 to {delta_days} days after the event).\n")
            
            print(f"Saved hypothesis report to: {report_filename}")
        except Exception as e:
            print(f"Error saving report: {e}")
        
        return phase_results