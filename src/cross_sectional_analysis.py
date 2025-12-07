import polars as pl
import numpy as np
import warnings
from typing import List, Dict, Tuple
import matplotlib.pyplot as plt
import os

class CrossSectionalAnalyzer:
    """Analyze heterogeneity across firm characteristics"""
    
    def __init__(self, data: pl.DataFrame, stock_data_paths: List[str]):
        self.data = data
        self.stock_paths = stock_data_paths
        
    def load_market_cap_data(self) -> pl.DataFrame:
        """
        Load market cap from CRSP
        
        Market Cap = |price| * shares_outstanding / 1000
        (Result in millions)
        """
        market_cap_data = []
        
        for stock_path in self.stock_paths:
            try:
                df = pl.scan_parquet(stock_path)
                
                # Get column names (case-insensitive)
                schema = df.collect_schema()
                cols_lower = {col.lower(): col for col in schema.names()}
                
                # Map to standard names
                select_cols = []
                rename_map = {}
                
                for std_name in ['ticker', 'date', 'prc', 'shrout']:
                    if std_name in cols_lower:
                        orig_name = cols_lower[std_name]
                        select_cols.append(orig_name)
                        if orig_name != std_name:
                            rename_map[orig_name] = std_name
                
                df = df.select(select_cols)
                if rename_map:
                    df = df.rename(rename_map)
                
                # Calculate market cap
                df = df.with_columns([
                    pl.col('prc').abs().alias('prc_abs'),
                    pl.col('shrout').alias('shrout_thousands')
                ]).with_columns(
                    (pl.col('prc_abs') * pl.col('shrout_thousands') / 1000)
                    .alias('market_cap_millions')
                ).select(['ticker', 'date', 'market_cap_millions'])
                
                # Filter valid
                df = df.filter(
                    pl.col('market_cap_millions').is_not_null() &
                    (pl.col('market_cap_millions') > 0)
                )
                
                market_cap_data.append(df.collect(streaming=True))
                
            except Exception as e:
                warnings.warn(f"Error loading {stock_path}: {e}")
                continue
        
        if not market_cap_data:
            raise ValueError("No market cap data loaded")
        
        return pl.concat(market_cap_data, how='vertical')
    
    def assign_size_quintiles(self) -> pl.DataFrame:
        """Assign market cap quintiles by year"""
        
        print("Loading market cap data...")
        market_cap = self.load_market_cap_data()
        
        # Cast date column to match Event Date datetime type before join
        market_cap = market_cap.with_columns(
            pl.col('date').cast(pl.Datetime('us'))
        )
        
        # Join with events
        data_with_cap = self.data.join(
            market_cap,
            left_on=['ticker', 'Event Date'],
            right_on=['ticker', 'date'],
            how='left'
        )
        
        # Add year
        data_with_cap = data_with_cap.with_columns(
            pl.col('Event Date').dt.year().alias('event_year')
        )
        
        # Assign quintiles within year
        data_with_cap = data_with_cap.with_columns(
            pl.col('market_cap_millions')
            .qcut(5, labels=['Q1_Small', 'Q2', 'Q3', 'Q4', 'Q5_Large'],
                  allow_duplicates=True)
            .over('event_year')
            .alias('size_quintile')
        )
        
        # Handle missing
        data_with_cap = data_with_cap.with_columns(
            pl.when(pl.col('market_cap_millions').is_null())
            .then(pl.lit('Unknown'))
            .otherwise(pl.col('size_quintile'))
            .alias('size_quintile')
        )
        
        # Print distribution
        dist = data_with_cap.group_by('size_quintile').agg(
            pl.col('event_id').n_unique().alias('n_events')
        ).sort('size_quintile')
        print("\nSize quintile distribution:")
        print(dist)
        
        return data_with_cap
    
    def assign_industry_classification(self) -> pl.DataFrame:
        """Classify by SIC code"""
        
        sic_data = []
        
        for stock_path in self.stock_paths:
            try:
                df = pl.scan_parquet(stock_path)
                schema = df.collect_schema()
                cols_lower = {col.lower(): col for col in schema.names()}
                
                select_cols = []
                rename_map = {}
                for std_name in ['ticker', 'date', 'siccd']:
                    if std_name in cols_lower:
                        orig = cols_lower[std_name]
                        select_cols.append(orig)
                        if orig != std_name:
                            rename_map[orig] = std_name
                
                df = df.select(select_cols)
                if rename_map:
                    df = df.rename(rename_map)
                
                sic_data.append(df.collect(streaming=True))
                
            except Exception as e:
                warnings.warn(f"Error loading SIC from {stock_path}: {e}")
                continue
        
        if not sic_data:
            warnings.warn("No SIC data loaded")
            return self.data
        
        sic_df = pl.concat(sic_data, how='vertical')
        
        # Cast date column to match Event Date datetime type before join
        sic_df = sic_df.with_columns(
            pl.col('date').cast(pl.Datetime('us'))
        )
        
        # Join
        data_with_sic = self.data.join(
            sic_df,
            left_on=['ticker', 'Event Date'],
            right_on=['ticker', 'date'],
            how='left'
        )
        
        # Extract 2-digit SIC
        data_with_sic = data_with_sic.with_columns(
            (pl.col('siccd').cast(pl.Int64) // 100).alias('sic_2digit')
        )
        
        # Map to industry
        industry_map = {
            (0, 9): 'Agriculture',
            (10, 14): 'Mining',
            (15, 17): 'Construction',
            (20, 39): 'Manufacturing',
            (40, 49): 'Transportation',
            (50, 51): 'Wholesale',
            (52, 59): 'Retail',
            (60, 67): 'Finance',
            (70, 89): 'Services',
            (91, 99): 'Public Admin'
        }
        
        def map_sic(sic_2digit):
            if sic_2digit is None:
                return 'Unknown'
            for (low, high), name in industry_map.items():
                if low <= sic_2digit <= high:
                    return name
            return 'Other'
        
        data_with_sic = data_with_sic.with_columns(
            pl.col('sic_2digit').map_elements(map_sic, return_dtype=pl.Utf8).alias('industry')
        )
        
        return data_with_sic
    
    def assign_liquidity_quintiles(self) -> pl.DataFrame:
        """Assign based on average pre-event volume"""
        
        # Calculate average volume in days -20 to -1
        pre_vol = self.data.filter(
            pl.col('days_to_event').is_between(-20, -1)
        ).group_by('event_id').agg(
            pl.mean('vol').alias('avg_pre_volume')
        )
        
        # Join
        data_with_liq = self.data.join(pre_vol, on='event_id', how='left')
        
        # Add year
        data_with_liq = data_with_liq.with_columns(
            pl.col('Event Date').dt.year().alias('event_year')
        )
        
        # Quintiles by year
        data_with_liq = data_with_liq.with_columns(
            pl.col('avg_pre_volume')
            .qcut(5, labels=['Q1_Illiquid', 'Q2', 'Q3', 'Q4', 'Q5_Liquid'],
                  allow_duplicates=True)
            .over('event_year')
            .alias('liquidity_quintile')
        )
        
        # Handle missing
        data_with_liq = data_with_liq.with_columns(
            pl.when(pl.col('avg_pre_volume').is_null())
            .then(pl.lit('Unknown'))
            .otherwise(pl.col('liquidity_quintile'))
            .alias('liquidity_quintile')
        )
        
        return data_with_liq
    
    def analyze_rvr_by_characteristic(self, 
                                     characteristic_col: str,
                                     phases: Dict[str, Tuple[int, int]],
                                     results_dir: str,
                                     file_prefix: str) -> pl.DataFrame:
        """Compute RVR by characteristic and phase"""
        
        results = []
        
        for char_value in self.data[characteristic_col].unique().sort():
            subset = self.data.filter(pl.col(characteristic_col) == char_value)
            
            for phase_name, (start, end) in phases.items():
                phase_data = subset.filter(
                    (pl.col('days_to_event') >= start) &
                    (pl.col('days_to_event') <= end)
                )
                
                results.append({
                    'characteristic': characteristic_col,
                    'value': char_value,
                    'phase': phase_name,
                    'avg_rvr': phase_data['rvr'].mean(),
                    'median_rvr': phase_data['rvr'].median(),
                    'std_rvr': phase_data['rvr'].std(),
                    'n_events': phase_data['event_id'].n_unique()
                })
        
        results_df = pl.DataFrame(results)
        results_df.write_csv(
            os.path.join(results_dir, f"{file_prefix}_rvr_by_{characteristic_col}.csv")
        )
        
        # Create plot
        self._plot_cross_sectional(results_df, characteristic_col, phases,
                                   results_dir, file_prefix)
        
        return results_df
    
    def _plot_cross_sectional(self, results_df: pl.DataFrame, 
                             characteristic: str, phases: Dict,
                             results_dir: str, file_prefix: str):
        """Create bar plot comparing phases across characteristic values"""
        
        results_pd = results_df.to_pandas()
        
        fig, ax = plt.subplots(figsize=(12, 7))
        
        # Get unique values and phases
        char_values = results_pd['value'].unique()
        phase_names = list(phases.keys())
        
        x = np.arange(len(char_values))
        width = 0.25
        
        for i, phase in enumerate(phase_names):
            phase_data = results_pd[results_pd['phase'] == phase]
            phase_data = phase_data.set_index('value').reindex(char_values)
            
            offset = (i - 1) * width
            ax.bar(x + offset, phase_data['avg_rvr'], width, 
                  label=phase, alpha=0.8)
        
        ax.set_xlabel(characteristic.replace('_', ' ').title())
        ax.set_ylabel('Average RVR')
        ax.set_title(f'RVR by {characteristic.replace("_", " ").title()} and Phase')
        ax.set_xticks(x)
        ax.set_xticklabels(char_values, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        
        plt.tight_layout()
        plt.savefig(
            os.path.join(results_dir, f"{file_prefix}_rvr_by_{characteristic}.png"),
            dpi=200
        )
        plt.close()
    
    def analyze_temporal_stability(self,
                                   periods: List[Tuple[str, str, str]],
                                   phases: Dict[str, Tuple[int, int]],
                                   results_dir: str,
                                   file_prefix: str) -> pl.DataFrame:
        """Test if pattern persists across time periods"""
        
        results = []
        
        for period_name, start_date, end_date in periods:
            period_data = self.data.filter(
                (pl.col('Event Date') >= pl.lit(start_date).str.to_datetime()) &
                (pl.col('Event Date') <= pl.lit(end_date).str.to_datetime())
            )
            
            for phase_name, (day_start, day_end) in phases.items():
                phase_period_data = period_data.filter(
                    (pl.col('days_to_event') >= day_start) &
                    (pl.col('days_to_event') <= day_end)
                )
                
                results.append({
                    'period': period_name,
                    'phase': phase_name,
                    'avg_rvr': phase_period_data['rvr'].mean(),
                    'median_rvr': phase_period_data['rvr'].median(),
                    'n_events': phase_period_data['event_id'].n_unique()
                })
        
        results_df = pl.DataFrame(results)
        
        # Calculate amplification ratios
        results_df = results_df.sort(['period', 'phase'])
        
        amplifications = []
        for period in results_df['period'].unique():
            period_data = results_df.filter(pl.col('period') == period)
            
            pre_rvr = period_data.filter(pl.col('phase') == 'pre_event')['avg_rvr'][0]
            rising_rvr = period_data.filter(pl.col('phase') == 'post_event_rising')['avg_rvr'][0]
            
            if pre_rvr is not None and pre_rvr != 0:
                amp = rising_rvr / pre_rvr
            else:
                amp = None
            
            amplifications.extend([None, amp, None])  # One per phase
        
        results_df = results_df.with_columns(
            pl.Series('amplification', amplifications)
        )
        
        results_df.write_csv(
            os.path.join(results_dir, f"{file_prefix}_temporal_stability.csv")
        )
        
        # Plot
        self._plot_temporal_stability(results_df, results_dir, file_prefix)
        
        return results_df
    
    def _plot_temporal_stability(self, results_df: pl.DataFrame,
                                results_dir: str, file_prefix: str):
        """Plot RVR across time periods"""
        
        results_pd = results_df.to_pandas()
        
        fig, ax = plt.subplots(figsize=(12, 7))
        
        periods = results_pd['period'].unique()
        phases = results_pd['phase'].unique()
        
        x = np.arange(len(periods))
        width = 0.25
        
        for i, phase in enumerate(phases):
            phase_data = results_pd[results_pd['phase'] == phase]
            phase_data = phase_data.set_index('period').reindex(periods)
            
            offset = (i - 1) * width
            ax.bar(x + offset, phase_data['avg_rvr'], width,
                  label=phase, alpha=0.8)
        
        ax.set_xlabel('Time Period')
        ax.set_ylabel('Average RVR')
        ax.set_title('Temporal Stability of RVR Pattern')
        ax.set_xticks(x)
        ax.set_xticklabels(periods)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        
        plt.tight_layout()
        plt.savefig(
            os.path.join(results_dir, f"{file_prefix}_temporal_stability.png"),
            dpi=200
        )
        plt.close()