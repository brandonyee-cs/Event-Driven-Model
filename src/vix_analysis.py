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
            pl.col(vix_col).rolling_mean(window_size=window, min_periods=2).over('event_id').alias('vix_smooth'),
            
            # Rolling standard deviation of VIX (volatility of VIX)
            pl.col(vix_col).rolling_std(window_size=window, min_periods=2).over('event_id').alias('vix_volatility')
        ])
        
        return df
    
    def analyze_pre_event_vix_sentiment(self, 
                                       vix_data: pl.DataFrame,
                                       vix_col: str = 'vix',
                                       return_col: str = 'ret',
                                       pre_event_start: int = -30,
                                       pre_event_end: int = -1) -> Dict[str, Any]:
        """
        Analyze pre-event VIX changes as a market sentiment indicator.
        
        Parameters:
        vix_data (pl.DataFrame): DataFrame with VIX data
        vix_col (str): Column name containing VIX data
        return_col (str): Column name containing returns
        pre_event_start (int): Start of pre-event window (days relative to event)
        pre_event_end (int): End of pre-event window (days relative to event)
        
        Returns:
        Dict[str, Any]: Dictionary containing analysis results
        """
        print("\n--- Analyzing Pre-Event VIX Changes as Sentiment Indicator ---")
        
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
            pl.col('vix_volatility').mean().alias('avg_vix_volatility'),
            pl.count().alias('pre_days_count')
        ]).filter(pl.col('pre_days_count') >= 5)  # Ensure enough data points
        
        # Calculate concurrent pre-event returns
        pre_event_returns = vix_data.filter(
            (pl.col('days_to_event') >= pre_event_start) &
            (pl.col('days_to_event') <= pre_event_end)
        ).group_by('event_id').agg([
            # Calculate cumulative returns (compound returns)
            ((pl.col(return_col).fill_null(0) + 1).product() - 1).alias('pre_event_return'),
            pl.count().alias('pre_days_count')
        ]).filter(pl.col('pre_days_count') >= 3)  # Ensure enough data points
        
        # Join pre-event VIX with concurrent returns
        vix_sentiment_data = pre_event_vix.join(
            pre_event_returns, on='event_id', how='inner'
        )
        
        if vix_sentiment_data.is_empty():
            print("Error: No matching events found after joining VIX changes and returns.")
            return None
        
        print(f"Analyzing {vix_sentiment_data.height} events with both VIX and return data.")
        
        # Convert to NumPy arrays for correlation analysis
        vix_changes = vix_sentiment_data['vix_pct_change'].to_numpy()
        vix_volatility = vix_sentiment_data['avg_vix_volatility'].to_numpy()
        concurrent_returns = vix_sentiment_data['pre_event_return'].to_numpy()
        
        # Remove NaNs and infinities
        valid_indices_changes = np.logical_and(
            np.isfinite(vix_changes),
            np.isfinite(concurrent_returns)
        )
        valid_vix_changes = vix_changes[valid_indices_changes]
        valid_concurrent_returns = concurrent_returns[valid_indices_changes]
        
        valid_indices_volatility = np.logical_and(
            np.isfinite(vix_volatility),
            np.isfinite(concurrent_returns)
        )
        valid_vix_volatility = vix_volatility[valid_indices_volatility]
        valid_volatility_returns = concurrent_returns[valid_indices_volatility]
        
        # Calculate correlations - now as sentiment indicators rather than predictors
        results = {}
        if len(valid_vix_changes) >= 5:
            pearson_corr, pearson_p = pearsonr(valid_vix_changes, valid_concurrent_returns)
            spearman_corr, spearman_p = spearmanr(valid_vix_changes, valid_concurrent_returns)
            
            print(f"VIX change vs. concurrent returns Pearson: {pearson_corr:.4f} (p-value: {pearson_p:.4f})")
            print(f"VIX change vs. concurrent returns Spearman: {spearman_corr:.4f} (p-value: {spearman_p:.4f})")
            
            results['change_pearson_corr'] = pearson_corr
            results['change_pearson_p'] = pearson_p
            results['change_spearman_corr'] = spearman_corr
            results['change_spearman_p'] = spearman_p
            results['change_n_observations'] = len(valid_vix_changes)
        
        if len(valid_vix_volatility) >= 5:
            vol_pearson_corr, vol_pearson_p = pearsonr(valid_vix_volatility, valid_volatility_returns)
            vol_spearman_corr, vol_spearman_p = spearmanr(valid_vix_volatility, valid_volatility_returns)
            
            print(f"VIX volatility vs. concurrent returns Pearson: {vol_pearson_corr:.4f} (p-value: {vol_pearson_p:.4f})")
            print(f"VIX volatility vs. concurrent returns Spearman: {vol_spearman_corr:.4f} (p-value: {vol_spearman_p:.4f})")
            
            results['vol_pearson_corr'] = vol_pearson_corr
            results['vol_pearson_p'] = vol_pearson_p
            results['vol_spearman_corr'] = vol_spearman_corr
            results['vol_spearman_p'] = vol_spearman_p
            results['vol_n_observations'] = len(valid_vix_volatility)
        
        # Determine if the sentiment relationship is present
        significant_threshold = 0.05
        
        # Check if either VIX changes or VIX volatility show significant correlation with concurrent returns
        # This tests if VIX captures market sentiment during pre-event period
        sentiment_relationship = (
            ('change_pearson_corr' in results and 
             abs(results['change_pearson_corr']) > 0.15 and 
             results['change_pearson_p'] < significant_threshold) or
            ('change_spearman_corr' in results and 
             abs(results['change_spearman_corr']) > 0.15 and 
             results['change_spearman_p'] < significant_threshold) or
            ('vol_pearson_corr' in results and 
             abs(results['vol_pearson_corr']) > 0.15 and 
             results['vol_pearson_p'] < significant_threshold) or
            ('vol_spearman_corr' in results and 
             abs(results['vol_spearman_corr']) > 0.15 and 
             results['vol_spearman_p'] < significant_threshold)
        )
        
        print(f"VIX captures pre-event market sentiment: {sentiment_relationship}")
        
        results['sentiment_relationship'] = sentiment_relationship
        results['vix_sentiment_data'] = vix_sentiment_data
        
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
        
        # Determine if post-event relationship is present (this part is unchanged as it was already supported)
        positive_pearson_days = sum(1 for c in daily_corrs if c['pearson_corr'] > 0)
        positive_spearman_days = sum(1 for c in daily_corrs if c['spearman_corr'] > 0)
        
        post_event_relationship = (
            (positive_pearson_days >= len(daily_corrs) / 2 and significant_days_pearson > 0) or
            (positive_spearman_days >= len(daily_corrs) / 2 and significant_days_spearman > 0)
        )
        
        print(f"Post-event VIX-return relationship present: {post_event_relationship}")
        
        results = {
            'daily_correlations': daily_corrs,
            'avg_pearson': avg_pearson,
            'avg_spearman': avg_spearman,
            'significant_days_pearson': significant_days_pearson,
            'significant_days_spearman': significant_days_spearman,
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
    
    def run_refined_hypothesis_2_test(self,
                                    vix_col: str = 'vix',
                                    return_col: str = 'ret',
                                    pre_days: int = 60,
                                    post_days: int = 60,
                                    delta_days: int = 10,
                                    results_dir: str = "results/refined_hypothesis_2",
                                    file_prefix: str = "event") -> Dict[str, Any]:
        """
        Run test of refined Hypothesis 2.
        
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
        print("\n=== Running Test of Refined Hypothesis 2 ===")
        print("Refined Hypothesis 2: VIX dynamics around events reflect differentiated uncertainty profiles:")
        print("  1. Pre-event VIX changes reflect market sentiment rather than directly predicting return magnitudes")
        print("  2. Post-event VIX movements correlate with contemporaneous returns confirming impact uncertainty's resolution")
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
        
        # Test first part of hypothesis: Pre-event VIX reflects market sentiment
        pre_event_results = self.analyze_pre_event_vix_sentiment(
            vix_data,
            vix_col=vix_col,
            return_col=return_col,
            pre_event_start=-30,  # Analyze last 30 days before event
            pre_event_end=-1
        )
        
        if pre_event_results is None:
            print("Error: Failed to analyze pre-event VIX sentiment relationship.")
            sentiment_indicator = False
        else:
            sentiment_indicator = pre_event_results.get('sentiment_relationship', False)
            try:
                self.plot_vix_sentiment_relationship(pre_event_results, results_dir, file_prefix)
            except Exception as e:
                print(f"Warning: Failed to create pre-event plots: {e}")
        
        # Test second part of hypothesis: Post-event VIX spikes correlate with returns
        post_event_results = self.analyze_post_event_vix_return_relationship(
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
            try:
                self.plot_post_event_correlations(post_event_results, results_dir, file_prefix)
            except Exception as e:
                print(f"Warning: Failed to create post-event plot: {e}")
        
        # For the refined hypothesis, both parts need to be true:
        # 1. Pre-event VIX should show relationship with concurrent returns (sentiment)
        # 2. Post-event VIX should correlate with returns (impact resolution)
        hypothesis_supported = sentiment_indicator and postevent_correlation
        
        print(f"\nRefined Hypothesis 2 Test Results:")
        print(f"Part 1 (Pre-event VIX as sentiment indicator): {sentiment_indicator}")
        print(f"Part 2 (Post-event VIX-return correlation): {postevent_correlation}")
        print(f"Overall Refined Hypothesis 2 Supported: {hypothesis_supported}")
        
        # Create and save summary report
        os.makedirs(results_dir, exist_ok=True)
        report_filename = os.path.join(results_dir, f"{file_prefix}_refined_hypothesis_2_report.txt")
        
        try:
            with open(report_filename, 'w') as f:
                f.write("===== Refined Hypothesis 2 Test Report =====\n\n")
                f.write("Refined Hypothesis 2: VIX dynamics around events reflect differentiated uncertainty profiles:\n")
                f.write("  1. Pre-event VIX changes reflect market sentiment rather than directly predicting return magnitudes\n")
                f.write("  2. Post-event VIX movements correlate with contemporaneous returns confirming impact uncertainty's resolution\n")
                f.write(f"Delta (post-event rising phase duration): {delta_days} days\n\n")
                
                f.write("Part 1: Pre-event VIX as market sentiment indicator\n")
                f.write(f"Supported: {sentiment_indicator}\n")
                
                if pre_event_results is not None:
                    if 'change_pearson_corr' in pre_event_results:
                        f.write(f"VIX change vs. concurrent returns Pearson: r = {pre_event_results['change_pearson_corr']:.4f} ")
                        f.write(f"(p = {pre_event_results['change_pearson_p']:.4f})\n")
                        f.write(f"VIX change vs. concurrent returns Spearman: ρ = {pre_event_results['change_spearman_corr']:.4f} ")
                        f.write(f"(p = {pre_event_results['change_spearman_p']:.4f})\n")
                        f.write(f"Number of observations: {pre_event_results['change_n_observations']}\n")
                    
                    if 'vol_pearson_corr' in pre_event_results:
                        f.write(f"VIX volatility vs. concurrent returns Pearson: r = {pre_event_results['vol_pearson_corr']:.4f} ")
                        f.write(f"(p = {pre_event_results['vol_pearson_p']:.4f})\n")
                        f.write(f"VIX volatility vs. concurrent returns Spearman: ρ = {pre_event_results['vol_spearman_corr']:.4f} ")
                        f.write(f"(p = {pre_event_results['vol_spearman_p']:.4f})\n")
                        f.write(f"Number of observations: {pre_event_results['vol_n_observations']}\n")
                
                f.write("\nPart 2: Post-event VIX correlations with returns\n")
                f.write(f"Supported: {postevent_correlation}\n")
                
                if post_event_results is not None:
                    f.write(f"Average Pearson correlation: {post_event_results['avg_pearson']:.4f}\n")
                    f.write(f"Average Spearman correlation: {post_event_results['avg_spearman']:.4f}\n")
                    f.write(f"Days with significant positive Pearson correlation: {post_event_results['significant_days_pearson']}/")
                    f.write(f"{len(post_event_results['daily_correlations'])}\n")
                    f.write(f"Days with significant positive Spearman correlation: {post_event_results['significant_days_spearman']}/")
                    f.write(f"{len(post_event_results['daily_correlations'])}\n")
                
                f.write(f"\nOverall Refined Hypothesis 2 Supported: {hypothesis_supported}\n\n")
                
                if hypothesis_supported:
                    f.write("Conclusion: The results support the refined Hypothesis 2. VIX dynamics around events do reflect\n")
                    f.write("differentiated uncertainty profiles. Pre-event VIX changes serve as a market sentiment indicator,\n")
                    f.write("correlating with concurrent returns rather than predicting future returns. Post-event VIX movements\n")
                    f.write("correlate with contemporaneous returns, confirming the resolution of impact uncertainty.\n")
                else:
                    f.write("Conclusion: The results do not fully support the refined Hypothesis 2. ")
                    if sentiment_indicator:
                        f.write("While pre-event VIX changes do serve as a market sentiment indicator, ")
                    else:
                        f.write("Pre-event VIX changes do not show a clear relationship with market sentiment, ")
                    
                    if postevent_correlation:
                        f.write("and post-event VIX movements do correlate with contemporaneous returns as expected.\n")
                    else:
                        f.write("and post-event VIX movements do not show the expected correlation with contemporaneous returns.\n")
            
            print(f"Saved refined hypothesis report to: {report_filename}")
        except Exception as e:
            print(f"Error saving report: {e}")
        
        # Save results to CSV
        sentiment_csv = os.path.join(results_dir, f"{file_prefix}_vix_sentiment_data.csv")
        try:
            if pre_event_results is not None and 'vix_sentiment_data' in pre_event_results:
                pre_event_results['vix_sentiment_data'].write_csv(sentiment_csv)
                print(f"Saved VIX sentiment data to: {sentiment_csv}")
        except Exception as e:
            print(f"Error saving sentiment data: {e}")
        
        results = {
            'sentiment_indicator': sentiment_indicator,
            'postevent_correlation': postevent_correlation,
            'hypothesis_supported': hypothesis_supported,
            'pre_event_results': pre_event_results,
            'post_event_results': post_event_results
        }
        
        return results