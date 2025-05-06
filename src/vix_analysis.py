import os
import polars as pl
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
import pandas as pd
from typing import List, Optional, Tuple, Dict, Any
from scipy.stats import pearsonr, spearmanr

class VIXImpactAnalysis:
    """Analyzes the relationship between VIX changes and returns to test Hypothesis 2."""
    
    def __init__(self, event_analysis):
        """
        Initialize with an existing EventAnalysis instance.
        
        Parameters:
        event_analysis: An initialized and data-loaded EventAnalysis instance
        """
        self.event_analysis = event_analysis
        self.data = event_analysis.data
        
    def calculate_vix_changes(self, 
                             vix_col: str = 'vix',
                             pre_days: int = 60,
                             post_days: int = 60,
                             window: int = 5) -> pl.DataFrame:
        """
        Calculate VIX changes around events.
        
        Parameters:
        vix_col (str): Column name containing VIX data
        pre_days (int): Days before event to analyze
        post_days (int): Days after event to analyze
        window (int): Window size for rolling VIX changes
        
        Returns:
        pl.DataFrame: DataFrame containing VIX changes
        """
        print(f"\n--- Calculating VIX Changes (Window={window} days) ---")
        
        if self.data is None or vix_col not in self.data.columns:
            print(f"Error: Data not loaded or missing VIX column '{vix_col}'.")
            return None
            
        # Filter data to analysis period
        df = self.data.filter(
            (pl.col('days_to_event') >= -pre_days) & 
            (pl.col('days_to_event') <= post_days)
        ).sort(['event_id', 'days_to_event'])
        
        # Calculate VIX changes
        df = df.with_columns([
            # Absolute VIX changes over window days
            (pl.col(vix_col) - pl.col(vix_col).shift(window).over('event_id')).alias('vix_change'),
            
            # Percentage VIX changes over window days
            pl.when(pl.col(vix_col).shift(window).over('event_id') > 0)
            .then((pl.col(vix_col) / pl.col(vix_col).shift(window).over('event_id') - 1) * 100)
            .otherwise(None)
            .alias('vix_pct_change'),
            
            # Rolling mean VIX (smoothed)
            pl.col(vix_col).rolling_mean(window_size=window, min_periods=2).over('event_id').alias('vix_smooth')
        ])
        
        return df
    
    def analyze_pre_event_vix_return_relationship(self, 
                                                vix_data: pl.DataFrame,
                                                vix_col: str = 'vix',
                                                return_col: str = 'ret',
                                                pre_event_start: int = -30,
                                                pre_event_end: int = -1,
                                                post_event_start: int = 0,
                                                post_event_end: int = 10) -> Dict[str, Any]:
        """
        Analyze the relationship between pre-event VIX increases and post-event returns.
        
        Parameters:
        vix_data (pl.DataFrame): DataFrame with VIX data
        vix_col (str): Column name containing VIX data
        return_col (str): Column name containing returns
        pre_event_start (int): Start of pre-event window (days relative to event)
        pre_event_end (int): End of pre-event window (days relative to event)
        post_event_start (int): Start of post-event window (days relative to event)
        post_event_end (int): End of post-event window (days relative to event)
        
        Returns:
        Dict[str, Any]: Dictionary containing analysis results
        """
        print("\n--- Analyzing Pre-Event VIX Changes vs. Post-Event Returns ---")
        
        if vix_data is None or vix_data.is_empty():
            print("Error: No VIX data available for analysis.")
            return None
        
        # Calculate pre-event VIX changes by event
        pre_event_vix = vix_data.filter(
            (pl.col('days_to_event') >= pre_event_start) &
            (pl.col('days_to_event') <= pre_event_end)
        ).group_by('event_id').agg([
            pl.col(vix_col).first().alias('vix_start'),
            pl.col(vix_col).last().alias('vix_end'),
            ((pl.col(vix_col).last() - pl.col(vix_col).first()) / pl.col(vix_col).first() * 100).alias('vix_pct_change'),
            pl.count().alias('pre_days_count')
        ]).filter(pl.col('pre_days_count') >= 5)  # Ensure enough data points
        
        # Calculate post-event returns by event
        post_event_returns = vix_data.filter(
            (pl.col('days_to_event') >= post_event_start) &
            (pl.col('days_to_event') <= post_event_end)
        ).group_by('event_id').agg([
            # Calculate cumulative returns (compound returns)
            ((pl.col(return_col).fill_null(0) + 1).product() - 1).alias('cum_return'),
            pl.count().alias('post_days_count')
        ]).filter(pl.col('post_days_count') >= 3)  # Ensure enough data points
        
        # Join pre-event VIX changes with post-event returns
        vix_return_data = pre_event_vix.join(
            post_event_returns, on='event_id', how='inner'
        )
        
        if vix_return_data.is_empty():
            print("Error: No matching events found after joining VIX changes and returns.")
            return None
        
        print(f"Analyzing {vix_return_data.height} events with both VIX and return data.")
        
        # Convert to NumPy arrays for correlation analysis
        vix_changes = vix_return_data['vix_pct_change'].to_numpy()
        returns = vix_return_data['cum_return'].to_numpy()
        
        # Remove NaNs and infinities
        valid_indices = np.logical_and(
            np.isfinite(vix_changes),
            np.isfinite(returns)
        )
        valid_vix_changes = vix_changes[valid_indices]
        valid_returns = returns[valid_indices]
        
        # Calculate correlations
        if len(valid_vix_changes) >= 5:
            pearson_corr, pearson_p = pearsonr(valid_vix_changes, valid_returns)
            spearman_corr, spearman_p = spearmanr(valid_vix_changes, valid_returns)
            
            print(f"Pearson correlation: {pearson_corr:.4f} (p-value: {pearson_p:.4f})")
            print(f"Spearman correlation: {spearman_corr:.4f} (p-value: {spearman_p:.4f})")
            
            # Determine if hypothesis is supported
            significant_threshold = 0.05
            is_supported = (
                (pearson_corr > 0 and pearson_p < significant_threshold) or
                (spearman_corr > 0 and spearman_p < significant_threshold)
            )
            
            print(f"Hypothesis supported (pre-event component): {is_supported}")
            
            results = {
                'pearson_corr': pearson_corr,
                'pearson_p': pearson_p,
                'spearman_corr': spearman_corr,
                'spearman_p': spearman_p,
                'n_observations': len(valid_vix_changes),
                'hypothesis_supported': is_supported
            }
        else:
            print("Error: Not enough valid data points for correlation analysis.")
            results = {
                'pearson_corr': None,
                'pearson_p': None,
                'spearman_corr': None,
                'spearman_p': None,
                'n_observations': len(valid_vix_changes),
                'hypothesis_supported': False
            }
        
        # Prepare data for scatter plot
        vix_return_data_valid = vix_return_data.filter(
            pl.col('vix_pct_change').is_not_null() &
            pl.col('cum_return').is_not_null() &
            pl.col('vix_pct_change').is_finite() &
            pl.col('cum_return').is_finite()
        )
        
        results['scatter_data'] = vix_return_data_valid
        
        return results
        
    def analyze_post_event_vix_return_relationship(self, 
                                                 vix_data: pl.DataFrame,
                                                 vix_col: str = 'vix',
                                                 return_col: str = 'ret',
                                                 delta_days: int = 10) -> Dict[str, Any]:
        """
        Analyze the relationship between post-event VIX spikes and returns during the post-event rising phase.
        
        Parameters:
        vix_data (pl.DataFrame): DataFrame with VIX data
        vix_col (str): Column name containing VIX data
        return_col (str): Column name containing returns
        delta_days (int): Parameter δ defining the post-event rising phase
        
        Returns:
        Dict[str, Any]: Dictionary containing analysis results
        """
        print("\n--- Analyzing Post-Event VIX Spikes vs. Returns ---")
        
        if vix_data is None or vix_data.is_empty():
            print("Error: No VIX data available for analysis.")
            return None
        
        # Filter to post-event rising phase
        post_event_data = vix_data.filter(
            (pl.col('days_to_event') > 0) & 
            (pl.col('days_to_event') <= delta_days)
        )
        
        if post_event_data.is_empty():
            print("Error: No data available for post-event rising phase.")
            return None
        
        # Calculate daily correlation between VIX and returns by day
        daily_corrs = []
        for day in range(1, delta_days + 1):
            day_data = post_event_data.filter(pl.col('days_to_event') == day)
            
            if day_data.height >= 10:  # Ensure enough data points
                # Filter for rows where both VIX and return values are valid
                valid_data = day_data.filter(
                    pl.col(vix_col).is_not_null() & 
                    pl.col(return_col).is_not_null() &
                    pl.col(vix_col).is_finite() &
                    pl.col(return_col).is_finite()
                )
                
                if valid_data.height >= 10:
                    try:
                        # Extract values as NumPy arrays from the same filtered DataFrame
                        vix_values = valid_data[vix_col].to_numpy()
                        ret_values = valid_data[return_col].to_numpy()
                        
                        # Calculate correlation
                        pearson_corr, pearson_p = pearsonr(vix_values, ret_values)
                        spearman_corr, spearman_p = spearmanr(vix_values, ret_values)
                        
                        daily_corrs.append({
                            'day': day,
                            'pearson_corr': pearson_corr,
                            'pearson_p': pearson_p,
                            'spearman_corr': spearman_corr,
                            'spearman_p': spearman_p,
                            'n_observations': len(vix_values)
                        })
                    except Exception as e:
                        print(f"Error calculating correlation for day {day}: {e}")
            else:
                print(f"Skipping day {day}: Not enough data points ({day_data.height} < 10)")
        
        if not daily_corrs:
            print("Error: Could not calculate correlations for any post-event days.")
            return None
        
        # Calculate average correlation across post-event rising phase
        avg_pearson = np.mean([c['pearson_corr'] for c in daily_corrs])
        avg_spearman = np.mean([c['spearman_corr'] for c in daily_corrs])
        significant_days_pearson = sum(1 for c in daily_corrs if c['pearson_p'] < 0.05 and c['pearson_corr'] > 0)
        significant_days_spearman = sum(1 for c in daily_corrs if c['spearman_p'] < 0.05 and c['spearman_corr'] > 0)
        
        print(f"Average Pearson correlation: {avg_pearson:.4f}")
        print(f"Average Spearman correlation: {avg_spearman:.4f}")
        print(f"Days with significant positive Pearson correlation: {significant_days_pearson}/{len(daily_corrs)}")
        print(f"Days with significant positive Spearman correlation: {significant_days_spearman}/{len(daily_corrs)}")
        
        # Determine if hypothesis is supported
        # Criterion: At least half of the days have positive correlation, and at least one day has significant positive correlation
        positive_pearson_days = sum(1 for c in daily_corrs if c['pearson_corr'] > 0)
        positive_spearman_days = sum(1 for c in daily_corrs if c['spearman_corr'] > 0)
        
        is_supported = (
            (positive_pearson_days >= len(daily_corrs) / 2 and significant_days_pearson > 0) or
            (positive_spearman_days >= len(daily_corrs) / 2 and significant_days_spearman > 0)
        )
        
        print(f"Hypothesis supported (post-event component): {is_supported}")
        
        results = {
            'daily_correlations': daily_corrs,
            'avg_pearson': avg_pearson,
            'avg_spearman': avg_spearman,
            'significant_days_pearson': significant_days_pearson,
            'significant_days_spearman': significant_days_spearman,
            'hypothesis_supported': is_supported
        }
        
        return results
    
    def plot_pre_event_relationship(self, 
                                   results: Dict[str, Any],
                                   results_dir: str,
                                   file_prefix: str = "event") -> None:
        """
        Create scatter plot of pre-event VIX changes vs. post-event returns.
        
        Parameters:
        results (Dict[str, Any]): Results from analyze_pre_event_vix_return_relationship
        results_dir (str): Directory to save the plot
        file_prefix (str): Prefix for saved files
        """
        if results is None or 'scatter_data' not in results:
            print("Error: No valid results for plotting.")
            return
        
        scatter_data = results['scatter_data']
        if scatter_data.is_empty():
            print("Error: No valid data for scatter plot.")
            return
        
        # Convert to pandas for plotting
        scatter_data_pd = scatter_data.to_pandas()
        
        # Create scatter plot without trendline to avoid statsmodels dependency
        try:
            # First try with trendline if statsmodels is available
            import statsmodels.api as sm
            fig = px.scatter(
                scatter_data_pd, 
                x='vix_pct_change', 
                y='cum_return',
                title='Pre-Event VIX Changes vs. Post-Event Returns',
                labels={
                    'vix_pct_change': 'Pre-Event VIX Change (%)',
                    'cum_return': 'Post-Event Cumulative Return'
                },
                trendline='ols'
            )
        except (ImportError, ModuleNotFoundError):
            # Fall back to scatter plot without trendline
            print("Note: statsmodels package not found. Creating scatter plot without trendline.")
            fig = px.scatter(
                scatter_data_pd, 
                x='vix_pct_change', 
                y='cum_return',
                title='Pre-Event VIX Changes vs. Post-Event Returns',
                labels={
                    'vix_pct_change': 'Pre-Event VIX Change (%)',
                    'cum_return': 'Post-Event Cumulative Return'
                }
            )
        
        # Add annotation with correlation values
        if results['pearson_corr'] is not None:
            fig.add_annotation(
                x=0.05, y=0.95,
                xref='paper', yref='paper',
                text=f"Pearson: r = {results['pearson_corr']:.4f} (p = {results['pearson_p']:.4f})<br>" +
                     f"Spearman: ρ = {results['spearman_corr']:.4f} (p = {results['spearman_p']:.4f})<br>" +
                     f"N = {results['n_observations']}",
                showarrow=False,
                align='left',
                bordercolor='black',
                borderwidth=1,
                bgcolor='white',
                opacity=0.8
            )
        
        # Save plot
        os.makedirs(results_dir, exist_ok=True)
        try:
            plot_filename = os.path.join(results_dir, f"{file_prefix}_pre_event_vix_scatter.png")
            fig.write_image(plot_filename, format='png', scale=2)
            print(f"Saved pre-event VIX relationship plot to: {plot_filename}")
        except Exception as e:
            print(f"Warning: Could not save plot image: {e}")
            html_filename = os.path.join(results_dir, f"{file_prefix}_pre_event_vix_scatter.html")
            fig.write_html(html_filename)
            print(f"Saved as HTML (fallback) to: {html_filename}")
    
    def plot_post_event_correlations(self, 
                                   results: Dict[str, Any],
                                   results_dir: str,
                                   file_prefix: str = "event") -> None:
        """
        Create bar plot of daily correlations between VIX and returns in the post-event rising phase.
        
        Parameters:
        results (Dict[str, Any]): Results from analyze_post_event_vix_return_relationship
        results_dir (str): Directory to save the plot
        file_prefix (str): Prefix for saved files
        """
        if results is None or 'daily_correlations' not in results or not results['daily_correlations']:
            print("Error: No valid results for plotting.")
            return
        
        # Convert daily correlations to pandas DataFrame
        corr_data_pd = pd.DataFrame(results['daily_correlations'])
        
        # Create bar plot
        fig = go.Figure()
        
        # Add Pearson correlation bars
        fig.add_trace(go.Bar(
            x=corr_data_pd['day'],
            y=corr_data_pd['pearson_corr'],
            name='Pearson Correlation',
            marker_color='blue',
            opacity=0.6
        ))
        
        # Add Spearman correlation bars
        fig.add_trace(go.Bar(
            x=corr_data_pd['day'],
            y=corr_data_pd['spearman_corr'],
            name='Spearman Correlation',
            marker_color='red',
            opacity=0.6
        ))
        
        # Add significance markers
        for i, row in corr_data_pd.iterrows():
            if row['pearson_p'] < 0.05:
                fig.add_annotation(
                    x=row['day'],
                    y=row['pearson_corr'],
                    text="*",
                    showarrow=False,
                    yshift=10,
                    font=dict(size=20, color="blue")
                )
            
            if row['spearman_p'] < 0.05:
                fig.add_annotation(
                    x=row['day'],
                    y=row['spearman_corr'],
                    text="#",
                    showarrow=False,
                    yshift=-10,
                    font=dict(size=20, color="red")
                )
        
        # Add zero line
        fig.add_hline(y=0, line=dict(color='black', dash='dash'))
        
        # Update layout
        fig.update_layout(
            title='Daily Correlation between VIX and Returns in Post-Event Rising Phase',
            xaxis_title='Days After Event',
            yaxis_title='Correlation Coefficient',
            legend=dict(x=0.01, y=0.99, xanchor='left', yanchor='top'),
            annotations=[
                dict(
                    x=0.01, y=0.01,
                    xref='paper', yref='paper',
                    text="* Significant Pearson correlation (p < 0.05)<br># Significant Spearman correlation (p < 0.05)",
                    showarrow=False,
                    align='left',
                    bgcolor='white',
                    opacity=0.8
                )
            ],
            barmode='group',
            width=1000,
            height=600
        )
        
        # Save plot
        os.makedirs(results_dir, exist_ok=True)
        try:
            plot_filename = os.path.join(results_dir, f"{file_prefix}_post_event_vix_correlations.png")
            fig.write_image(plot_filename, format='png', scale=2)
            print(f"Saved post-event VIX correlations plot to: {plot_filename}")
        except Exception as e:
            print(f"Warning: Could not save plot image: {e}")
            html_filename = os.path.join(results_dir, f"{file_prefix}_post_event_vix_correlations.html")
            fig.write_html(html_filename)
            print(f"Saved as HTML (fallback) to: {html_filename}")
    
    def run_hypothesis_2_test(self,
                             vix_col: str = 'vix',
                             return_col: str = 'ret',
                             pre_days: int = 60,
                             post_days: int = 60,
                             delta_days: int = 10,
                             results_dir: str = "results/hypothesis_2",
                             file_prefix: str = "event") -> Dict[str, Any]:
        """
        Run complete test of Hypothesis 2.
        
        Parameters:
        vix_col (str): Column name containing VIX data
        return_col (str): Column name containing returns
        pre_days (int): Days before event to analyze
        post_days (int): Days after event to analyze
        delta_days (int): Parameter δ defining the post-event rising phase
        results_dir (str): Directory to save results
        file_prefix (str): Prefix for saved files
        
        Returns:
        Dict[str, Any]: Results of the hypothesis test
        """
        print("\n=== Running Test of Hypothesis 2 ===")
        print("Hypothesis 2: VIX changes proxy impact and secondary uncertainties:")
        print("             Pre-event VIX increases predict stronger returns")
        print("             Post-event VIX spikes correlate with elevated returns during post-event rising phase")
        print(f"Using delta = {delta_days} days for post-event rising phase")
        
        # Check if VIX column exists
        if self.data is None or vix_col not in self.data.columns:
            print(f"Error: Data not loaded or missing VIX column '{vix_col}'.")
            return {'hypothesis_supported': False, 'error': f"Missing VIX column '{vix_col}'"}
        
        # Calculate VIX changes
        vix_data = self.calculate_vix_changes(
            vix_col=vix_col,
            pre_days=pre_days,
            post_days=post_days
        )
        
        if vix_data is None or vix_data.is_empty():
            print("Error: Failed to calculate VIX changes.")
            return {'hypothesis_supported': False, 'error': 'No VIX data available'}
        
        # Test first part of hypothesis: Pre-event VIX increases predict stronger returns
        pre_event_results = self.analyze_pre_event_vix_return_relationship(
            vix_data,
            vix_col=vix_col,
            return_col=return_col,
            pre_event_start=-30,  # Analyze last 30 days before event
            pre_event_end=-1,
            post_event_start=0,
            post_event_end=delta_days
        )
        
        if pre_event_results is None:
            print("Error: Failed to analyze pre-event VIX-return relationship.")
            part1_supported = False
        else:
            part1_supported = pre_event_results.get('hypothesis_supported', False)
            try:
                self.plot_pre_event_relationship(pre_event_results, results_dir, file_prefix)
            except Exception as e:
                print(f"Warning: Failed to create pre-event plot: {e}")
        
        # Test second part of hypothesis: Post-event VIX spikes correlate with elevated returns
        post_event_results = self.analyze_post_event_vix_return_relationship(
            vix_data,
            vix_col=vix_col,
            return_col=return_col,
            delta_days=delta_days
        )
        
        if post_event_results is None:
            print("Error: Failed to analyze post-event VIX-return relationship.")
            part2_supported = False
        else:
            part2_supported = post_event_results.get('hypothesis_supported', False)
            try:
                self.plot_post_event_correlations(post_event_results, results_dir, file_prefix)
            except Exception as e:
                print(f"Warning: Failed to create post-event plot: {e}")
        
        # Overall hypothesis is supported if both parts are supported
        hypothesis_supported = part1_supported and part2_supported
        
        print(f"\nHypothesis 2 Test Results:")
        print(f"Part 1 (Pre-event): {part1_supported}")
        print(f"Part 2 (Post-event): {part2_supported}")
        print(f"Overall Hypothesis 2 Supported: {hypothesis_supported}")
        
        # Create and save summary report
        os.makedirs(results_dir, exist_ok=True)
        report_filename = os.path.join(results_dir, f"{file_prefix}_hypothesis_2_report.txt")
        
        try:
            with open(report_filename, 'w') as f:
                f.write("===== Hypothesis 2 Test Report =====\n\n")
                f.write("Hypothesis 2: VIX changes proxy impact and secondary uncertainties\n")
                f.write("             Pre-event VIX increases predict stronger returns\n")
                f.write("             Post-event VIX spikes correlate with elevated returns during post-event rising phase\n")
                f.write(f"Delta (post-event rising phase duration): {delta_days} days\n\n")
                
                f.write("Part 1: Pre-event VIX increases predict stronger returns\n")
                f.write(f"Supported: {part1_supported}\n")
                
                if pre_event_results is not None:
                    f.write(f"Pearson correlation: r = {pre_event_results['pearson_corr']:.4f} (p = {pre_event_results['pearson_p']:.4f})\n")
                    f.write(f"Spearman correlation: ρ = {pre_event_results['spearman_corr']:.4f} (p = {pre_event_results['spearman_p']:.4f})\n")
                    f.write(f"Number of observations: {pre_event_results['n_observations']}\n")
                
                f.write("\nPart 2: Post-event VIX spikes correlate with elevated returns\n")
                f.write(f"Supported: {part2_supported}\n")
                
                if post_event_results is not None:
                    f.write(f"Average Pearson correlation: {post_event_results['avg_pearson']:.4f}\n")
                    f.write(f"Average Spearman correlation: {post_event_results['avg_spearman']:.4f}\n")
                    f.write(f"Days with significant positive Pearson correlation: {post_event_results['significant_days_pearson']}/{len(post_event_results['daily_correlations'])}\n")
                    f.write(f"Days with significant positive Spearman correlation: {post_event_results['significant_days_spearman']}/{len(post_event_results['daily_correlations'])}\n")
                
                f.write(f"\nOverall Hypothesis 2 Supported: {hypothesis_supported}\n")
                
                if hypothesis_supported:
                    f.write("\nConclusion: The results support Hypothesis 2. VIX changes do appear to proxy impact and\n")
                    f.write("secondary uncertainties, with pre-event VIX increases predicting stronger returns and\n")
                    f.write("post-event VIX spikes correlating with elevated returns during the post-event rising phase.\n")
                else:
                    f.write("\nConclusion: The results do not fully support Hypothesis 2. ")
                    if part1_supported:
                        f.write("While pre-event VIX increases predict stronger returns, ")
                    else:
                        f.write("Pre-event VIX increases do not predict stronger returns, ")
                    
                    if part2_supported:
                        f.write("and post-event VIX spikes correlate with elevated returns during the post-event rising phase.\n")
                    else:
                        f.write("and post-event VIX spikes do not correlate with elevated returns during the post-event rising phase.\n")
            
            print(f"Saved hypothesis report to: {report_filename}")
        except Exception as e:
            print(f"Error saving report: {e}")
        
        # Save results to CSV
        csv_filename = os.path.join(results_dir, f"{file_prefix}_hypothesis_2_data.csv")
        try:
            # Create a summary DataFrame
            if pre_event_results is not None and 'scatter_data' in pre_event_results:
                vix_return_summary = pre_event_results['scatter_data']
                vix_return_summary.write_csv(csv_filename)
                print(f"Saved VIX-return summary data to: {csv_filename}")
        except Exception as e:
            print(f"Error saving data: {e}")
        
        results = {
            'part1_supported': part1_supported,
            'part2_supported': part2_supported,
            'hypothesis_supported': hypothesis_supported,
            'pre_event_results': pre_event_results,
            'post_event_results': post_event_results
        }
        
        return results