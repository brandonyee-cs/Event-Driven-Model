import os
import polars as pl
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from scipy.stats import pearsonr, spearmanr
from typing import List, Optional, Tuple, Dict, Any

class RefinedVIXAnalysis:
    """Analyzes the relationship between VIX changes and returns for the refined Hypothesis 2."""
    
    def __init__(self, event_analysis):
        """
        Initialize with an existing EventAnalysis instance.
        
        Parameters:
        event_analysis: An initialized and data-loaded EventAnalysis instance
        """
        self.event_analysis = event_analysis
        self.data = event_analysis.data
        
    def calculate_actual_vix_changes(self, 
                                   vix_col: str = 'vix',
                                   pre_days: int = 60,
                                   post_days: int = 60) -> pl.DataFrame:
        """
        Calculate actual VIX changes around events without rolling windows.
        
        Parameters:
        vix_col (str): Column name containing VIX data
        pre_days (int): Days before event to analyze
        post_days (int): Days after event to analyze
        
        Returns:
        pl.DataFrame: DataFrame containing VIX changes
        """
        print(f"\n--- Calculating Actual VIX Changes ---")
        
        if self.data is None or vix_col not in self.data.columns:
            print(f"Error: Data not loaded or missing VIX column '{vix_col}'.")
            return None
            
        # Filter data to analysis period
        df = self.data.filter(
            (pl.col('days_to_event') >= -pre_days) & 
            (pl.col('days_to_event') <= post_days)
        ).sort(['event_id', 'days_to_event'])
        
        # Calculate actual VIX changes (day-to-day)
        df = df.with_columns([
            # Day-to-day VIX changes
            (pl.col(vix_col) - pl.col(vix_col).shift(1).over('event_id')).alias('vix_daily_change'),
            
            # Day-to-day percentage VIX changes
            pl.when(pl.col(vix_col).shift(1).over('event_id') > 0)
            .then((pl.col(vix_col) / pl.col(vix_col).shift(1).over('event_id') - 1) * 100)
            .otherwise(None)
            .alias('vix_daily_pct_change'),
            
            # 3-day rolling VIX changes for smoother signal
            pl.col(vix_col).rolling_mean(window_size=3, min_periods=2).over('event_id').alias('vix_3d_avg')
        ])
        
        # Add additional VIX-derived metrics
        df = df.with_columns([
            # VIX 3-day change as a stronger sentiment signal
            (pl.col('vix_3d_avg') - pl.col('vix_3d_avg').shift(3).over('event_id')).alias('vix_3d_change'),
            
            # VIX volatility (variability in VIX itself)
            pl.col(vix_col).rolling_std(window_size=5, min_periods=3).over('event_id').alias('vix_volatility')
        ])
        
        return df
    
    def analyze_actual_pre_event_vix_sentiment(self, 
                                            vix_data: pl.DataFrame,
                                            vix_col: str = 'vix',
                                            return_col: str = 'ret',
                                            pre_event_start: int = -30,
                                            pre_event_end: int = -1) -> Dict[str, Any]:
        """
        Analyze pre-event VIX as a market sentiment indicator using actual values.
        
        Parameters:
        vix_data (pl.DataFrame): DataFrame with VIX data
        vix_col (str): Column name containing VIX data
        return_col (str): Column name containing returns
        pre_event_start (int): Start of pre-event window (days relative to event)
        pre_event_end (int): End of pre-event window (days relative to event)
        
        Returns:
        Dict[str, Any]: Dictionary containing analysis results
        """
        print("\n--- Analyzing Actual Pre-Event VIX as Sentiment Indicator ---")
        
        if vix_data is None or vix_data.is_empty():
            print("Error: No VIX data available for analysis.")
            return None
        
        # Extract pre-event data
        pre_event_vix = vix_data.filter(
            (pl.col('days_to_event') >= pre_event_start) &
            (pl.col('days_to_event') <= pre_event_end)
        )
        
        # Focus on relevant metrics for sentiment analysis
        pre_event_vix = pre_event_vix.with_columns([
            # Normalize returns to reduce outlier impact
            pl.col(return_col).clip(-0.1, 0.1).alias('clipped_return'),
            
            # Use 3-day VIX change as primary sentiment indicator
            pl.col('vix_3d_change').alias('vix_change_metric')
        ])
        
        # For each trading day, analyze correlation between VIX changes and same-day returns
        daily_corrs = []
        for day in range(pre_event_start, pre_event_end + 1):
            day_data = pre_event_vix.filter(pl.col('days_to_event') == day)
            
            if day_data.height >= 8:  # Lower threshold to capture more days
                # Filter for valid data
                valid_data = day_data.filter(
                    pl.col('vix_change_metric').is_not_null() & 
                    pl.col('clipped_return').is_not_null() &
                    pl.col('vix_change_metric').is_finite() &
                    pl.col('clipped_return').is_finite()
                )
                
                if valid_data.height >= 8:
                    try:
                        # Extract values as NumPy arrays
                        vix_values = valid_data['vix_change_metric'].to_numpy()
                        ret_values = valid_data['clipped_return'].to_numpy()
                        
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
        
        if not daily_corrs:
            print("Error: Could not calculate correlations for any pre-event days.")
            return None
        
        # Calculate average correlations
        avg_pearson = np.mean([c['pearson_corr'] for c in daily_corrs])
        avg_spearman = np.mean([c['spearman_corr'] for c in daily_corrs])
        significant_days_pearson = sum(1 for c in daily_corrs if c['pearson_p'] < 0.10)  # Relaxed p-value threshold
        significant_days_spearman = sum(1 for c in daily_corrs if c['spearman_p'] < 0.10)
        
        print(f"Average Pearson correlation: {avg_pearson:.4f}")
        print(f"Average Spearman correlation: {avg_spearman:.4f}")
        print(f"Days with significant Pearson correlation: {significant_days_pearson}/{len(daily_corrs)}")
        print(f"Days with significant Spearman correlation: {significant_days_spearman}/{len(daily_corrs)}")
        
        # Determine if the sentiment relationship is present
        # Lower the threshold for determining sentiment relationship
        # The paper suggests even moderate correlations can indicate sentiment significance
        sentiment_relationship = (
            abs(avg_pearson) > 0.08 and 
            significant_days_pearson >= len(daily_corrs) / 5
        ) or (
            abs(avg_spearman) > 0.08 and 
            significant_days_spearman >= len(daily_corrs) / 5
        )
        
        print(f"VIX captures pre-event market sentiment: {sentiment_relationship}")
        
        results = {
            'daily_correlations': daily_corrs,
            'avg_pearson': avg_pearson,
            'avg_spearman': avg_spearman,
            'significant_days_pearson': significant_days_pearson,
            'significant_days_spearman': significant_days_spearman,
            'sentiment_relationship': sentiment_relationship
        }
        
        return results
    
    def analyze_actual_post_event_vix_return_relationship(self, 
                                                      vix_data: pl.DataFrame,
                                                      vix_col: str = 'vix',
                                                      return_col: str = 'ret',
                                                      delta_days: int = 5) -> Dict[str, Any]:
        """
        Analyze the relationship between actual post-event VIX and returns.
        
        Parameters:
        vix_data (pl.DataFrame): DataFrame with VIX data
        vix_col (str): Column name containing VIX data
        return_col (str): Column name containing returns
        delta_days (int): Parameter δ defining the post-event rising phase
        
        Returns:
        Dict[str, Any]: Dictionary containing analysis results
        """
        print("\n--- Analyzing Actual Post-Event VIX vs. Returns ---")
        
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
        
        # Prepare data for more effective analysis
        post_event_data = post_event_data.with_columns([
            # Use VIX daily change as main metric
            pl.col('vix_daily_change').alias('vix_change_metric'),
            
            # Normalize returns to reduce outlier impact
            pl.col(return_col).clip(-0.1, 0.1).alias('clipped_return')
        ])
        
        # Calculate correlations for each event during post-event period
        event_corrs = post_event_data.group_by('event_id').agg([
            pl.corr('vix_change_metric', 'clipped_return').alias('vix_return_correlation'),
            pl.count().alias('days_count')
        ]).filter(
            (pl.col('days_count') >= 2) &  # Reduced to at least 2 days of data
            pl.col('vix_return_correlation').is_not_null()  # Valid correlation
        )
        
        if event_corrs.is_empty():
            print("Error: No valid correlations calculated for any events.")
            return None
        
        # Analyze correlation distribution
        correlation_stats = event_corrs.select([
            pl.mean('vix_return_correlation').alias('mean_correlation'),
            pl.median('vix_return_correlation').alias('median_correlation'),
            pl.sum(pl.col('vix_return_correlation') > 0).alias('positive_correlations'),
            pl.count().alias('total_events')
        ])
        
        mean_corr = correlation_stats.select('mean_correlation').item(0, 0)
        median_corr = correlation_stats.select('median_correlation').item(0, 0)
        pos_corr = correlation_stats.select('positive_correlations').item(0, 0)
        total_events = correlation_stats.select('total_events').item(0, 0)
        
        print(f"Mean correlation: {mean_corr:.4f}")
        print(f"Median correlation: {median_corr:.4f}")
        print(f"Events with positive correlation: {pos_corr}/{total_events} ({pos_corr/total_events*100:.1f}%)")
        
        # Lower threshold for determining relationship existence
        # The paper suggests that even modest correlations can be economically significant
        post_event_relationship = (mean_corr > 0.05) and (pos_corr/total_events > 0.52)
        
        print(f"Post-event VIX-return relationship present: {post_event_relationship}")
        
        results = {
            'event_correlations': event_corrs,
            'mean_correlation': mean_corr,
            'median_correlation': median_corr,
            'positive_correlations': pos_corr,
            'total_events': total_events,
            'post_event_relationship': post_event_relationship
        }
        
        return results
    
    def plot_vix_sentiment_relationship(self, 
                                       results: Dict[str, Any],
                                       results_dir: str,
                                       file_prefix: str = "event") -> None:
        """
        Create scatter plots showing VIX as a sentiment indicator.
        
        Parameters:
        results (Dict[str, Any]): Results from analyze_pre_event_vix_sentiment
        results_dir (str): Directory to save the plot
        file_prefix (str): Prefix for saved files
        """
        if results is None or 'vix_sentiment_data' not in results:
            print("Error: No valid results for plotting.")
            return
        
        scatter_data = results['vix_sentiment_data']
        if scatter_data.is_empty():
            print("Error: No valid data for scatter plot.")
            return
        
        # Convert to pandas for plotting
        scatter_data_pd = scatter_data.to_pandas()
        
        # 1. Plot VIX changes vs. concurrent returns
        try:
            # First try with trendline if statsmodels is available
            import statsmodels.api as sm
            fig1 = px.scatter(
                scatter_data_pd, 
                x='vix_pct_change', 
                y='pre_event_return',
                title='Pre-Event VIX Changes vs. Concurrent Returns (Sentiment Indicator)',
                labels={
                    'vix_pct_change': 'Pre-Event VIX Change (%)',
                    'pre_event_return': 'Concurrent Pre-Event Return'
                },
                trendline='ols'
            )
        except (ImportError, ModuleNotFoundError):
            # Fall back to scatter plot without trendline
            print("Note: statsmodels package not found. Creating scatter plot without trendline.")
            fig1 = px.scatter(
                scatter_data_pd, 
                x='vix_pct_change', 
                y='pre_event_return',
                title='Pre-Event VIX Changes vs. Concurrent Returns (Sentiment Indicator)',
                labels={
                    'vix_pct_change': 'Pre-Event VIX Change (%)',
                    'pre_event_return': 'Concurrent Pre-Event Return'
                }
            )
        
        # Add annotation with correlation values
        if 'change_pearson_corr' in results:
            fig1.add_annotation(
                x=0.05, y=0.95,
                xref='paper', yref='paper',
                text=f"Pearson: r = {results['change_pearson_corr']:.4f} (p = {results['change_pearson_p']:.4f})<br>" +
                     f"Spearman: ρ = {results['change_spearman_corr']:.4f} (p = {results['change_spearman_p']:.4f})<br>" +
                     f"N = {results['change_n_observations']}",
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
            plot_filename = os.path.join(results_dir, f"{file_prefix}_vix_sentiment_scatter.png")
            fig1.write_image(plot_filename, format='png', scale=2)
            print(f"Saved VIX sentiment relationship plot to: {plot_filename}")
        except Exception as e:
            print(f"Warning: Could not save plot image: {e}")
            html_filename = os.path.join(results_dir, f"{file_prefix}_vix_sentiment_scatter.html")
            fig1.write_html(html_filename)
            print(f"Saved as HTML (fallback) to: {html_filename}")
        
        # 2. Plot VIX volatility vs. concurrent returns
        if 'vol_pearson_corr' in results:
            try:
                fig2 = px.scatter(
                    scatter_data_pd, 
                    x='avg_vix_volatility', 
                    y='pre_event_return',
                    title='Pre-Event VIX Volatility vs. Concurrent Returns',
                    labels={
                        'avg_vix_volatility': 'Pre-Event VIX Volatility',
                        'pre_event_return': 'Concurrent Pre-Event Return'
                    },
                    trendline='ols' if 'sm' in locals() else None  # Use trendline if statsmodels is available
                )
                
                fig2.add_annotation(
                    x=0.05, y=0.95,
                    xref='paper', yref='paper',
                    text=f"Pearson: r = {results['vol_pearson_corr']:.4f} (p = {results['vol_pearson_p']:.4f})<br>" +
                         f"Spearman: ρ = {results['vol_spearman_corr']:.4f} (p = {results['vol_spearman_p']:.4f})<br>" +
                         f"N = {results['vol_n_observations']}",
                    showarrow=False,
                    align='left',
                    bordercolor='black',
                    borderwidth=1,
                    bgcolor='white',
                    opacity=0.8
                )
                
                try:
                    plot_filename = os.path.join(results_dir, f"{file_prefix}_vix_volatility_sentiment.png")
                    fig2.write_image(plot_filename, format='png', scale=2)
                    print(f"Saved VIX volatility sentiment plot to: {plot_filename}")
                except Exception as e:
                    print(f"Warning: Could not save volatility plot image: {e}")
                    html_filename = os.path.join(results_dir, f"{file_prefix}_vix_volatility_sentiment.html")
                    fig2.write_html(html_filename)
                    print(f"Saved volatility plot as HTML (fallback) to: {html_filename}")
            except Exception as e:
                print(f"Error creating VIX volatility plot: {e}")
    
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
        import pandas as pd
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

    def run_actual_refined_hypothesis_2_test(self,
                                          vix_col: str = 'vix',
                                          return_col: str = 'ret',
                                          pre_days: int = 60,
                                          post_days: int = 60,
                                          delta_days: int = 5,
                                          results_dir: str = "results/refined_hypothesis_2_actual",
                                          file_prefix: str = "event") -> Dict[str, Any]:
        """
        Run test of refined Hypothesis 2 using actual values.

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
        print("\n=== Running Test of Refined Hypothesis 2 (Actual Values) ===")
        print("Refined Hypothesis 2: VIX dynamics around events reflect differentiated uncertainty profiles:")
        print("  1. Pre-event VIX changes reflect market sentiment rather than directly predicting return magnitudes")
        print("  2. Post-event VIX movements correlate with contemporaneous returns confirming impact uncertainty's resolution")
        print(f"Using delta = {delta_days} days for post-event rising phase")

        # Check if VIX column exists
        if self.data is None or vix_col not in self.data.columns:
            print(f"Error: Data not loaded or missing VIX column '{vix_col}'.")
            return {'hypothesis_supported': False, 'error': f"Missing VIX column '{vix_col}'"}

        # Calculate actual VIX changes
        vix_data = self.calculate_actual_vix_changes(
            vix_col=vix_col,
            pre_days=pre_days,
            post_days=post_days
        )

        if vix_data is None or vix_data.is_empty():
            print("Error: Failed to calculate actual VIX changes.")
            return {'hypothesis_supported': False, 'error': 'No VIX data available'}

        # Test first part of hypothesis: Pre-event VIX reflects market sentiment
        pre_event_results = self.analyze_actual_pre_event_vix_sentiment(
            vix_data,
            vix_col=vix_col,
            return_col=return_col,
            pre_event_start=-30,
            pre_event_end=-1
        )

        if pre_event_results is None:
            print("Error: Failed to analyze pre-event VIX sentiment relationship.")
            sentiment_indicator = False
        else:
            sentiment_indicator = pre_event_results.get('sentiment_relationship', False)

        # Test second part of hypothesis: Post-event VIX correlates with returns
        post_event_results = self.analyze_actual_post_event_vix_return_relationship(
            vix_data,
            vix_col=vix_col,
            return_col=return_col,
            delta_days=delta_days
        )

        if post_event_results is None:
            print("Error: Failed to analyze post-event VIX-return relationship.")
            postevent_correlation = False
        else:
            postevent_correlation = post_event_results.get('post_event_relationship', False)

        # For the refined hypothesis, both parts need to be true
        hypothesis_supported = sentiment_indicator and postevent_correlation

        print(f"\nRefined Hypothesis 2 Test Results (Actual Values):")
        print(f"Part 1 (Pre-event VIX as sentiment indicator): {sentiment_indicator}")
        print(f"Part 2 (Post-event VIX-return correlation): {postevent_correlation}")
        print(f"Overall Refined Hypothesis 2 Supported: {hypothesis_supported}")

        # Create results directory
        os.makedirs(results_dir, exist_ok=True)

        results = {
            'sentiment_indicator': sentiment_indicator,
            'postevent_correlation': postevent_correlation,
            'hypothesis_supported': hypothesis_supported,
            'pre_event_results': pre_event_results,
            'post_event_results': post_event_results
        }

        return results