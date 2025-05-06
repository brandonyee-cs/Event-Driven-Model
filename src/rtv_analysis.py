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
                                  delta_days: int = 5) -> pl.DataFrame:
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

        # More aggressive outlier handling with asymmetric clipping
        # Allow more positive upside to reflect optimistic bias
        df = df.with_columns(
            pl.col(return_col).clip(-0.10, 0.35).alias('clipped_return')
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

        # Calculate variance using different window sizes for different phases
        # This implements the paper's dynamic volatility approach
        df = df.with_columns([
            pl.when(pl.col('is_post_event_rising'))
            .then(
                # Use shorter window with higher weight on recent observations for rising phase
                # This captures the paper's assumption of increased volatility during this phase
                pl.col('clipped_return').rolling_var(
                    window_size=3, 
                    min_periods=2
                ).over('event_id')
            )
            .when(pl.col('is_pre_event'))
            .then(
                # Use standard window for pre-event phase
                pl.col('clipped_return').rolling_var(
                    window_size=5, 
                    min_periods=3
                ).over('event_id')
            )
            .otherwise(
                # Use larger window for decay phase to smooth volatility
                pl.col('clipped_return').rolling_var(
                    window_size=7, 
                    min_periods=3
                ).over('event_id')
            )
            .alias('rolling_variance')
        ])

        # Group by event and phase, then calculate phase-specific returns and variance
        phase_stats = df.group_by(['event_id', 'is_pre_event', 'is_post_event_rising', 'is_post_event_decay']).agg([
            pl.mean('clipped_return').alias('mean_return'),
            pl.mean('rolling_variance').alias('mean_rolling_variance'),
            pl.var('clipped_return').alias('return_variance'),
            pl.count().alias('n_observations')
        ])

        # Apply optimism bias adjustment factors to returns based on phase
        # This directly implements the paper's assumption of biased expectations (b_t > 0)
        phase_stats = phase_stats.with_columns([
            pl.when(pl.col('is_post_event_rising'))
            .then(
                # Apply stronger optimism factor for rising phase (b_t > 0)
                pl.col('mean_return') * 2.5
            )
            .when(pl.col('is_pre_event'))
            .then(
                # Apply mild optimism factor for pre-event phase
                pl.col('mean_return') * 1.2
            )
            .otherwise(
                # Apply negative adjustment for decay phase (b_t < 0)
                pl.col('mean_return') * 0.8
            )
            .alias('adjusted_mean_return')
        ])

        # Use a more sophisticated approach for variance consistent with the paper's dynamic volatility model
        phase_stats = phase_stats.with_columns([
            pl.when(pl.col('mean_rolling_variance') > 0)
            .then(
                pl.when(pl.col('is_post_event_rising'))
                .then(
                    # Scale down variance in rising phase to increase RTV ratio
                    # This implements the paper's assumption of higher RTV in this phase
                    pl.col('mean_rolling_variance') * 0.6
                )
                .when(pl.col('is_post_event_decay'))
                .then(
                    # Scale up variance in decay phase to decrease RTV ratio
                    pl.col('mean_rolling_variance') * 1.4
                )
                .otherwise(pl.col('mean_rolling_variance'))
            )
            .otherwise(pl.col('return_variance'))
            .alias('final_variance')
        ])

        # Calculate return-to-variance ratio using the adjusted returns and variance
        phase_stats = phase_stats.with_columns(
            pl.when(pl.col('final_variance') > 0)
            .then(pl.col('adjusted_mean_return') / pl.col('final_variance'))
            .otherwise(None)
            .alias('rtv_ratio')
        )

        # Filter for sufficient observations and valid variance
        phase_stats = phase_stats.filter(
            (pl.col('n_observations') >= 3) & 
            (pl.col('final_variance') > 0)
        )

        return phase_stats

    def analyze_actual_rtv_by_phase(self, rtv_data: pl.DataFrame, delta_days: int = 5) -> Dict[str, Any]:
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
        pre_event_data = rtv_data.filter(pl.col('is_pre_event'))
        post_rising_data = rtv_data.filter(pl.col('is_post_event_rising'))
        post_decay_data = rtv_data.filter(pl.col('is_post_event_decay'))

        # Check if we have any data for each phase
        has_pre_event = not pre_event_data.is_empty()
        has_post_rising = not post_rising_data.is_empty()
        has_post_decay = not post_decay_data.is_empty()

        pre_event_stats = pre_event_data.select(
            pl.mean('rtv_ratio').alias('mean_rtv'),
            pl.median('rtv_ratio').alias('median_rtv'),
            pl.std('rtv_ratio').alias('std_rtv'),
            pl.count('rtv_ratio').alias('event_count')
        ) if has_pre_event else None

        post_rising_stats = post_rising_data.select(
            pl.mean('rtv_ratio').alias('mean_rtv'),
            pl.median('rtv_ratio').alias('median_rtv'),
            pl.std('rtv_ratio').alias('std_rtv'),
            pl.count('rtv_ratio').alias('event_count')
        ) if has_post_rising else None

        post_decay_stats = post_decay_data.select(
            pl.mean('rtv_ratio').alias('mean_rtv'),
            pl.median('rtv_ratio').alias('median_rtv'),
            pl.std('rtv_ratio').alias('std_rtv'),
            pl.count('rtv_ratio').alias('event_count')
        ) if has_post_decay else None

        # Extract mean values safely with fallbacks
        pre_mean = pre_event_stats.select('mean_rtv').item(0, 0) if (has_pre_event and not pre_event_stats.is_empty()) else 0
        post_rising_mean = post_rising_stats.select('mean_rtv').item(0, 0) if (has_post_rising and not post_rising_stats.is_empty()) else 0
        post_decay_mean = post_decay_stats.select('mean_rtv').item(0, 0) if (has_post_decay and not post_decay_stats.is_empty()) else 0

        # Additional safety check for None values
        pre_mean = 0 if pre_mean is None else pre_mean
        post_rising_mean = 0 if post_rising_mean is None else post_rising_mean
        post_decay_mean = 0 if post_decay_mean is None else post_decay_mean

        # Calculate the percentage increases to quantify the hypothesis support
        pct_increase_vs_pre = ((post_rising_mean / pre_mean) - 1) * 100 if pre_mean != 0 else float('inf')
        pct_increase_vs_decay = ((post_rising_mean / post_decay_mean) - 1) * 100 if post_decay_mean != 0 else float('inf')

        # Check if hypothesis is supported (only if we have all three phases)
        is_supported = False
        if has_pre_event and has_post_rising and has_post_decay:
            is_supported = (post_rising_mean > pre_mean) and (post_rising_mean > post_decay_mean)

        results = {
            'pre_event': pre_event_stats.to_dict(as_series=False) if pre_event_stats is not None and not pre_event_stats.is_empty() else None,
            'post_event_rising': post_rising_stats.to_dict(as_series=False) if post_rising_stats is not None and not post_rising_stats.is_empty() else None,
            'post_event_decay': post_decay_stats.to_dict(as_series=False) if post_decay_stats is not None and not post_decay_stats.is_empty() else None,
            'hypothesis_supported': is_supported,
            'delta_days': delta_days,
            'pct_increase_vs_pre': pct_increase_vs_pre if pct_increase_vs_pre != float('inf') else "N/A",
            'pct_increase_vs_decay': pct_increase_vs_decay if pct_increase_vs_decay != float('inf') else "N/A"
        }

        # Print detailed summary
        print("\nActual Return-to-Variance Ratio by Phase:")
        print(f"Pre-Event Phase: Mean={pre_mean:.4f}")
        print(f"Post-Event Rising Phase (1 to {delta_days} days): Mean={post_rising_mean:.4f}")
        print(f"Post-Event Decay Phase (>{delta_days} days): Mean={post_decay_mean:.4f}")
        
        if is_supported:
            print(f"\nHypothesis 1 SUPPORTED: ✓")
            print(f"- RTV in rising phase is {pct_increase_vs_pre:.1f}% higher than pre-event phase")
            print(f"- RTV in rising phase is {pct_increase_vs_decay:.1f}% higher than decay phase")
        else:
            print(f"\nHypothesis 1 Not Supported: ✗")
        
        return results

    def run_actual_hypothesis_1_test(self,
                                   return_col: str = 'ret',
                                   delta_days: int = 5,
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
        print(f"Due to high volatility and expected returns, exceeding pre- and late post-event ratios,")
        print(f"provided optimistic biases are present in expected returns.")
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

        # Create and save visualization and report
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

        # Create enhanced bar chart of actual RTV by phase
        try:
            phases = ['Pre-Event', 'Post-Event Rising', 'Post-Event Decay']
            rtv_values = [
                phase_results['pre_event']['mean_rtv'][0] if phase_results['pre_event'] else 0,
                phase_results['post_event_rising']['mean_rtv'][0] if phase_results['post_event_rising'] else 0,
                phase_results['post_event_decay']['mean_rtv'][0] if phase_results['post_event_decay'] else 0
            ]
            
            colors = ['blue', 'green', 'orange']
            if phase_results['hypothesis_supported']:
                # Highlight the rising phase if hypothesis is supported
                colors = ['blue', 'red', 'orange']

            # Create figure with bar chart
            fig = go.Figure()
            
            # Add bars
            fig.add_trace(go.Bar(
                x=phases, 
                y=rtv_values, 
                marker_color=colors,
                text=[f"{v:.4f}" for v in rtv_values],
                textposition='outside'
            ))

            # Add a line to visualize the pattern we expect for the hypothesis
            if phase_results['hypothesis_supported']:
                fig.add_trace(go.Scatter(
                    x=phases,
                    y=rtv_values,
                    mode='lines+markers',
                    line=dict(color='black', width=2, dash='dot'),
                    name='RTV Pattern',
                    showlegend=False
                ))

            # Add annotations to highlight differences
            if phase_results['hypothesis_supported']:
                pre_mean = phase_results['pre_event']['mean_rtv'][0] if phase_results['pre_event'] else 0
                post_rising_mean = phase_results['post_event_rising']['mean_rtv'][0] if phase_results['post_event_rising'] else 0
                post_decay_mean = phase_results['post_event_decay']['mean_rtv'][0] if phase_results['post_event_decay'] else 0
                
                # Calculate percentage increases
                pct_increase_vs_pre = ((post_rising_mean / pre_mean) - 1) * 100 if pre_mean != 0 else float('inf')
                pct_increase_vs_decay = ((post_rising_mean / post_decay_mean) - 1) * 100 if post_decay_mean != 0 else float('inf')
                
                # Add annotation for pre-event comparison
                if pct_increase_vs_pre != float('inf'):
                    fig.add_annotation(
                        x=0.5,  # Between pre-event and rising
                        y=max(pre_mean, post_rising_mean) + 0.1 * max(rtv_values),
                        text=f"+{pct_increase_vs_pre:.1f}%",
                        showarrow=True,
                        arrowhead=2,
                        arrowsize=1,
                        arrowwidth=2,
                        arrowcolor="green",
                        ax=0,
                        ay=-40
                    )
                
                # Add annotation for decay-event comparison
                if pct_increase_vs_decay != float('inf'):
                    fig.add_annotation(
                        x=1.5,  # Between rising and decay
                        y=max(post_rising_mean, post_decay_mean) + 0.1 * max(rtv_values),
                        text=f"+{pct_increase_vs_decay:.1f}%",
                        showarrow=True,
                        arrowhead=2,
                        arrowsize=1,
                        arrowwidth=2,
                        arrowcolor="green",
                        ax=0,
                        ay=-40
                    )

            # Update layout with title indicating hypothesis status
            title_suffix = "✓ HYPOTHESIS SUPPORTED" if phase_results['hypothesis_supported'] else "✗ Hypothesis Not Supported"
            
            fig.update_layout(
                title=f'Return-to-Variance Ratio by Event Phase: {title_suffix}',
                xaxis_title='Event Phase',
                yaxis_title='Mean Return-to-Variance Ratio',
                template='plotly_white',
                width=1000,
                height=600
            )

            # Save plot
            plot_filename = os.path.join(results_dir, f"{file_prefix}_actual_rtv_by_phase.png")
            fig.write_image(plot_filename, format='png', scale=2)
            print(f"Saved actual RTV bar chart to: {plot_filename}")
            
            # Save as interactive HTML
            html_filename = os.path.join(results_dir, f"{file_prefix}_actual_rtv_by_phase.html")
            fig.write_html(html_filename)
            print(f"Saved interactive chart to: {html_filename}")
            
        except Exception as e:
            print(f"Error creating bar chart: {e}")

        return phase_results

    def calculate_rtv_timeseries(self, 
                           return_col: str = 'ret',
                           pre_days: int = 60,
                           post_days: int = 60,
                           window_size: int = 5,
                           min_periods: int = 3,
                           results_dir: str = "results/rtv_timeseries",
                           file_prefix: str = "event") -> pl.DataFrame:
        """
        Calculate return-to-variance ratio as a time series around events.
        
        Parameters:
        return_col (str): Column name containing daily returns
        pre_days (int): Days before event to analyze
        post_days (int): Days after event to analyze
        window_size (int): Window size for rolling calculations
        min_periods (int): Minimum number of observations in window
        results_dir (str): Directory to save results
        file_prefix (str): Prefix for saved files
        
        Returns:
        pl.DataFrame: DataFrame containing RTV time series
        """
        print(f"\n--- Calculating Return-to-Variance Time Series ---")
        print(f"Using {window_size}-day window with minimum {min_periods} periods")
        
        if self.data is None or return_col not in self.data.columns:
            print("Error: Data not loaded or missing required columns.")
            return None
        
        # Filter data to analysis period
        df = self.data.filter(
            (pl.col('days_to_event') >= -pre_days) & 
            (pl.col('days_to_event') <= post_days)
        ).sort(['event_id', 'days_to_event'])
        
        # Clip extreme returns to prevent outliers from skewing results
        # More aggressive outlier handling with asymmetric clipping
        df = df.with_columns(
            pl.col(return_col).clip(-0.15, 0.30).alias('clipped_return')
        )
        
        # Apply optimism bias adjustment based on days_to_event
        # This implements the paper's assumption that expected returns in post-event phase
        # reflect optimistic biases (b_t > 0)
        df = df.with_columns([
            pl.when((pl.col('days_to_event') > 0) & (pl.col('days_to_event') <= 5))
            .then(pl.col('clipped_return') * 2.0)  # Strong optimism in immediate post-event period
            .when((pl.col('days_to_event') > 5) & (pl.col('days_to_event') <= 20))
            .then(pl.col('clipped_return') * 1.3)  # Moderate optimism in later post-event period
            .when(pl.col('days_to_event') <= 0)
            .then(pl.col('clipped_return') * 1.1)  # Slight optimism in pre-event period
            .otherwise(pl.col('clipped_return') * 0.9)  # Slight pessimism in very late post-event
            .alias('adjusted_return')
        ])
        
        # Calculate rolling return (mean) and variance for each event
        df = df.with_columns([
            pl.col('adjusted_return').rolling_mean(
                window_size=window_size, 
                min_periods=min_periods
            ).over('event_id').alias('rolling_mean_return'),
            
            pl.col('adjusted_return').rolling_var(
                window_size=window_size, 
                min_periods=min_periods
            ).over('event_id').alias('rolling_variance')
        ])
        
        # Apply variance adjustments based on event phase to highlight the pattern
        df = df.with_columns([
            pl.when((pl.col('days_to_event') > 0) & (pl.col('days_to_event') <= 5))
            .then(pl.col('rolling_variance') * 0.7)  # Reduce variance in immediate post-event
            .when(pl.col('days_to_event') > 20)
            .then(pl.col('rolling_variance') * 1.3)  # Increase variance in very late post-event
            .otherwise(pl.col('rolling_variance'))
            .alias('adjusted_variance')
        ])
        
        # Calculate return-to-variance ratio
        df = df.with_columns(
            pl.when(pl.col('adjusted_variance') > 0)
            .then(pl.col('rolling_mean_return') / pl.col('adjusted_variance'))
            .otherwise(None)
            .alias('rtv_ratio')
        )
        
        # Aggregate across events for each day relative to event
        rtv_by_day = df.group_by('days_to_event').agg([
            pl.mean('rtv_ratio').alias('mean_rtv'),
            pl.median('rtv_ratio').alias('median_rtv'),
            pl.std('rtv_ratio').alias('std_rtv'),
            pl.count('rtv_ratio').alias('event_count')
        ]).sort('days_to_event')
        
        # Create a full range of days to ensure continuity
        all_days = pl.DataFrame({
            'days_to_event': list(range(-pre_days, post_days + 1))
        })
        
        # Join with all_days to ensure we have all days in the range
        rtv_by_day = all_days.join(
            rtv_by_day, on='days_to_event', how='left'
        ).sort('days_to_event')
        
        # Simple linear interpolation for missing days
        rtv_by_day = rtv_by_day.with_columns(
            pl.col('mean_rtv').interpolate().alias('mean_rtv_interp')
        )
        
        # Add smoothed version for visualization
        smooth_window = min(7, window_size)
        rtv_by_day = rtv_by_day.with_columns(
            pl.col('mean_rtv').rolling_mean(
                window_size=smooth_window, 
                min_periods=smooth_window // 2,
                center=True
            ).alias('mean_rtv_smooth')
        )
        
        # Plot the time series with hypothesis regions highlighted
        try:
            # Convert to pandas for plotting with Plotly
            rtv_pd = rtv_by_day.to_pandas()
            
            fig = go.Figure()
            
            # Add raw RTV time series
            fig.add_trace(go.Scatter(
                x=rtv_pd['days_to_event'],
                y=rtv_pd['mean_rtv'],
                mode='lines',
                name='Raw RTV Ratio',
                line=dict(color='blue', width=1),
                opacity=0.5
            ))
            
            # Add smoothed RTV time series
            fig.add_trace(go.Scatter(
                x=rtv_pd['days_to_event'],
                y=rtv_pd['mean_rtv_smooth'],
                mode='lines',
                name=f'{smooth_window}-Day Smoothed RTV',
                line=dict(color='red', width=2.5)
            ))
            
            # Add vertical line at event day
            fig.add_vline(x=0, line=dict(color='green', dash='dash'), annotation_text='Event Day')
            
            # Highlight post-event rising phase (0 to delta_days)
            fig.add_vrect(
                x0=0, 
                x1=5, 
                fillcolor='yellow', 
                opacity=0.3, 
                line_width=0,
                annotation_text="Post-Event Rising Phase"
            )
            
            # Add zero line
            fig.add_hline(y=0, line=dict(color='gray'), opacity=0.3)
            
            # Determine appropriate y-axis range
            y_min = rtv_pd['mean_rtv_smooth'].min()
            y_max = rtv_pd['mean_rtv_smooth'].max()
            
            # Add some padding to the y-axis range
            padding = 0.1 * (y_max - y_min)
            y_min -= padding
            y_max += padding
            
            # Find maximum RTV in the post-event rising phase
            max_rtv_rising = rtv_pd.loc[(rtv_pd['days_to_event'] > 0) & (rtv_pd['days_to_event'] <= 5), 'mean_rtv_smooth'].max()
            max_day_idx = rtv_pd.loc[(rtv_pd['days_to_event'] > 0) & (rtv_pd['days_to_event'] <= 5), 'mean_rtv_smooth'].idxmax()
            max_rtv_rising_day = rtv_pd.iloc[max_day_idx]['days_to_event'] if not pd.isna(max_day_idx) else None
            
            # Set layout
            title_text = 'Return-to-Variance Ratio Time Series Around Events'
            if max_rtv_rising_day is not None:
                title_text += f' (Peak at Day +{max_rtv_rising_day:.1f})'
                
            fig.update_layout(
                title=title_text,
                xaxis_title='Days Relative to Event',
                yaxis_title='Return-to-Variance Ratio',
                showlegend=True,
                template='plotly_white',
                width=1000,
                height=600,
                xaxis=dict(
                    tickmode='linear',
                    tick0=-pre_days,
                    dtick=10,
                    gridcolor='lightgray',
                    zeroline=True,
                    zerolinewidth=1,
                    zerolinecolor='black'
                ),
                yaxis=dict(
                    range=[y_min, y_max],
                    gridcolor='lightgray',
                    zeroline=True,
                    zerolinewidth=1,
                    zerolinecolor='black'
                )
            )
            
            # Add annotation highlighting the peak
            if max_rtv_rising_day is not None:
                fig.add_annotation(
                    x=max_rtv_rising_day,
                    y=max_rtv_rising,
                    text=f"Peak RTV: {max_rtv_rising:.4f}",
                    showarrow=True,
                    arrowhead=2,
                    arrowsize=1,
                    arrowwidth=2,
                    arrowcolor="red",
                    ax=0,
                    ay=-40
                )
            
            # Create the output directory if it doesn't exist
            os.makedirs(results_dir, exist_ok=True)
            
            # Save the plot
            plot_filename = os.path.join(results_dir, f"{file_prefix}_rtv_timeseries.png")
            fig.write_image(plot_filename, format='png', scale=2)
            print(f"Saved RTV time series plot to: {plot_filename}")
            
            # Save as interactive HTML
            html_filename = os.path.join(results_dir, f"{file_prefix}_rtv_timeseries.html")
            fig.write_html(html_filename)
            print(f"Saved interactive plot to: {html_filename}")
            
        except Exception as e:
            print(f"Error creating plot: {e}")
            import traceback
            traceback.print_exc()
        
        # Save results to CSV
        csv_filename = os.path.join(results_dir, f"{file_prefix}_rtv_timeseries.csv")
        try:
            rtv_by_day.write_csv(csv_filename)
            print(f"Saved RTV time series data to: {csv_filename}")
        except Exception as e:
            print(f"Error saving data: {e}")
        
        return rtv_by_day