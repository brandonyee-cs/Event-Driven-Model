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

    def calculate_actual_rtv_ratio(self, 
                                  return_col: str = 'ret',
                                  pre_days: int = 60,
                                  post_days: int = 60,
                                  delta_days: int = 10) -> pl.DataFrame:
        """
        Calculate return-to-variance ratio using actual values across event phases.

        Parameters:
        return_col (str): Column name containing daily returns
        pre_days (int): Days before event to analyze
        post_days (int): Days after event to analyze
        delta_days (int): Parameter δ from the paper defining the post-event rising phase

        Returns:
        pl.DataFrame: DataFrame containing return-to-variance ratios by phase
        """
        print(f"\n--- Calculating Actual Return-to-Variance Ratios ---")

        if self.data is None or return_col not in self.data.columns:
            print("Error: Data not loaded or missing required columns.")
            return None

        # Filter data to analysis period
        df = self.data.filter(
            (pl.col('days_to_event') >= -pre_days) & 
            (pl.col('days_to_event') <= post_days)
        ).sort(['event_id', 'days_to_event'])

        # Clip extreme returns to prevent outliers
        df = df.with_columns(
            pl.col(return_col).clip(-0.25, 0.25).alias('clipped_return')
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

        # Group by event and phase, then calculate phase-specific returns and variance
        phase_stats = df.group_by(['event_id', 'is_pre_event', 'is_post_event_rising', 'is_post_event_decay']).agg([
            pl.mean('clipped_return').alias('mean_return'),
            pl.var('clipped_return').alias('return_variance'),
            pl.count().alias('n_observations')
        ])

        # Calculate return-to-variance ratio
        phase_stats = phase_stats.with_columns(
            pl.when(pl.col('return_variance') > 0)
            .then(pl.col('mean_return') / pl.col('return_variance'))
            .otherwise(None)
            .alias('rtv_ratio')
        )

        # Filter for sufficient observations
        phase_stats = phase_stats.filter(pl.col('n_observations') >= 5)

        return phase_stats

    def analyze_actual_rtv_by_phase(self, rtv_data: pl.DataFrame, delta_days: int = 10) -> Dict[str, Any]:
        """
        Analyze actual return-to-variance ratios across the three event phases.

        Parameters:
        rtv_data (pl.DataFrame): DataFrame with return-to-variance ratios by phase
        delta_days (int): Parameter δ defining the post-event rising phase

        Returns:
        Dict[str, Any]: Dictionary with phase-by-phase statistics
        """
        print("\n--- Analyzing Actual Return-to-Variance Ratios by Phase ---")

        if rtv_data is None or rtv_data.is_empty():
            print("Error: No data available for analysis.")
            return None

        # Calculate statistics for each phase
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

        # Check if hypothesis is supported (same logic as before)
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
        print("\nActual Return-to-Variance Ratio by Phase:")
        print(f"Pre-Event Phase: Mean={pre_mean:.4f}")
        print(f"Post-Event Rising Phase (1 to {delta_days} days): Mean={post_rising_mean:.4f}")
        print(f"Post-Event Decay Phase (>{delta_days} days): Mean={post_decay_mean:.4f}")
        print(f"Hypothesis 1 Supported: {is_supported}")

        return results

    def run_actual_hypothesis_1_test(self,
                                   return_col: str = 'ret',
                                   delta_days: int = 10,
                                   results_dir: str = "results/hypothesis_1_actual",
                                   file_prefix: str = "event") -> Dict[str, Any]:
        """
        Run complete test of Hypothesis 1 using actual values.

        Parameters:
        return_col (str): Column name containing daily returns
        delta_days (int): Parameter δ defining the post-event rising phase
        results_dir (str): Directory to save results
        file_prefix (str): Prefix for saved files

        Returns:
        Dict[str, Any]: Results of the hypothesis test
        """
        print("\n=== Running Test of Hypothesis 1 (Actual Values) ===")
        print(f"Hypothesis 1: The return-to-variance ratio peaks during the post-event rising phase")
        print(f"Using delta = {delta_days} days for post-event rising phase")

        # Calculate actual return-to-variance ratios
        rtv_data = self.calculate_actual_rtv_ratio(
            return_col=return_col,
            delta_days=delta_days
        )

        if rtv_data is None or rtv_data.is_empty():
            print("Error: Failed to calculate actual return-to-variance ratios.")
            return {'hypothesis_supported': False, 'error': 'No data available'}

        # Analyze by phase
        phase_results = self.analyze_actual_rtv_by_phase(rtv_data, delta_days)

        # Create and save visualization and report similar to original implementation
        os.makedirs(results_dir, exist_ok=True)

        # Save detailed results to CSV
        csv_filename = os.path.join(results_dir, f"{file_prefix}_actual_rtv_phase_results.csv")
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

        # Create bar chart of actual RTV by phase
        try:
            phases = ['Pre-Event', 'Post-Event Rising', 'Post-Event Decay']
            rtv_values = [
                phase_results['pre_event']['mean_rtv'][0] if phase_results['pre_event'] else 0,
                phase_results['post_event_rising']['mean_rtv'][0] if phase_results['post_event_rising'] else 0,
                phase_results['post_event_decay']['mean_rtv'][0] if phase_results['post_event_decay'] else 0
            ]

            fig = go.Figure(data=[
                go.Bar(x=phases, y=rtv_values, marker_color=['blue', 'green', 'orange'])
            ])

            fig.update_layout(
                title='Actual Return-to-Variance Ratio by Event Phase',
                xaxis_title='Event Phase',
                yaxis_title='Mean Return-to-Variance Ratio',
                template='plotly_white',
                width=800,
                height=500
            )

            plot_filename = os.path.join(results_dir, f"{file_prefix}_actual_rtv_by_phase.png")
            fig.write_image(plot_filename, format='png', scale=2)
            print(f"Saved actual RTV bar chart to: {plot_filename}")
        except Exception as e:
            print(f"Error creating bar chart: {e}")

        return phase_results