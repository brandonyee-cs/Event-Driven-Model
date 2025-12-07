import polars as pl
import numpy as np
from scipy.stats import ttest_ind, mannwhitneyu, ttest_rel
from typing import Dict, Tuple

class PhaseComparisonTests:
    """Statistical tests for comparing metrics across event phases"""
    
    def __init__(self, data: pl.DataFrame, phases: Dict[str, Tuple[int, int]]):
        """
        Parameters:
        - data: DataFrame with 'days_to_event', 'event_id', 'rvr'
        - phases: {'phase_name': (start_day, end_day)}
        """
        self.data = data
        self.phases = phases
        
    def welch_t_test(self, phase1: str, phase2: str, metric_col: str) -> Dict:
        """
        Welch's t-test (unequal variances) between two phases
        
        Returns: t_statistic, p_value, df, mean_diff, ci_lower, ci_upper
        """
        # Extract data for each phase
        phase1_start, phase1_end = self.phases[phase1]
        phase2_start, phase2_end = self.phases[phase2]
        
        phase1_data = self.data.filter(
            (pl.col('days_to_event') >= phase1_start) &
            (pl.col('days_to_event') <= phase1_end) &
            pl.col(metric_col).is_not_null()
        )[metric_col].to_numpy()
        
        phase2_data = self.data.filter(
            (pl.col('days_to_event') >= phase2_start) &
            (pl.col('days_to_event') <= phase2_end) &
            pl.col(metric_col).is_not_null()
        )[metric_col].to_numpy()
        
        # Perform Welch's t-test
        t_stat, p_value = ttest_ind(phase2_data, phase1_data, equal_var=False)
        
        # Calculate means and difference
        mean1 = np.mean(phase1_data)
        mean2 = np.mean(phase2_data)
        mean_diff = mean2 - mean1
        
        # Calculate 95% confidence interval for difference
        se_diff = np.sqrt(np.var(phase1_data)/len(phase1_data) + 
                         np.var(phase2_data)/len(phase2_data))
        ci_lower = mean_diff - 1.96 * se_diff
        ci_upper = mean_diff + 1.96 * se_diff
        
        return {
            't_statistic': t_stat,
            'p_value': p_value,
            'mean_diff': mean_diff,
            'mean_phase1': mean1,
            'mean_phase2': mean2,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'n_phase1': len(phase1_data),
            'n_phase2': len(phase2_data)
        }
    
    def bootstrap_phase_difference(self, phase1: str, phase2: str, 
                                   metric_col: str, n_bootstrap: int = 10000,
                                   ci_level: float = 0.95) -> Dict:
        """
        Bootstrap confidence interval for difference in means
        
        Samples events with replacement and recalculates mean difference
        """
        # Get event-level means for each phase
        phase1_start, phase1_end = self.phases[phase1]
        phase2_start, phase2_end = self.phases[phase2]
        
        # Calculate mean per event for each phase
        phase1_means = self.data.filter(
            (pl.col('days_to_event') >= phase1_start) &
            (pl.col('days_to_event') <= phase1_end)
        ).group_by('event_id').agg(
            pl.mean(metric_col).alias('phase1_mean')
        )
        
        phase2_means = self.data.filter(
            (pl.col('days_to_event') >= phase2_start) &
            (pl.col('days_to_event') <= phase2_end)
        ).group_by('event_id').agg(
            pl.mean(metric_col).alias('phase2_mean')
        )
        
        # Join to get paired observations
        paired = phase1_means.join(phase2_means, on='event_id', how='inner')
        
        phase1_np = paired['phase1_mean'].to_numpy()
        phase2_np = paired['phase2_mean'].to_numpy()
        
        n_events = len(phase1_np)
        
        # Bootstrap
        bootstrap_diffs = []
        np.random.seed(42)
        
        for _ in range(n_bootstrap):
            # Sample events with replacement
            indices = np.random.choice(n_events, size=n_events, replace=True)
            
            boot_phase1 = phase1_np[indices]
            boot_phase2 = phase2_np[indices]
            
            boot_diff = np.mean(boot_phase2) - np.mean(boot_phase1)
            bootstrap_diffs.append(boot_diff)
        
        bootstrap_diffs = np.array(bootstrap_diffs)
        
        # Calculate confidence interval
        alpha = 1 - ci_level
        lower_percentile = (alpha/2) * 100
        upper_percentile = (1 - alpha/2) * 100
        
        return {
            'mean_diff': np.mean(bootstrap_diffs),
            'ci_lower': np.percentile(bootstrap_diffs, lower_percentile),
            'ci_upper': np.percentile(bootstrap_diffs, upper_percentile),
            'std_error': np.std(bootstrap_diffs),
            'n_bootstrap': n_bootstrap,
            'n_events': n_events
        }
    
    def mann_whitney_u_test(self, phase1: str, phase2: str, metric_col: str) -> Dict:
        """
        Mann-Whitney U test (non-parametric)
        Tests if distributions differ (doesn't assume normality)
        """
        phase1_start, phase1_end = self.phases[phase1]
        phase2_start, phase2_end = self.phases[phase2]
        
        phase1_data = self.data.filter(
            (pl.col('days_to_event') >= phase1_start) &
            (pl.col('days_to_event') <= phase1_end) &
            pl.col(metric_col).is_not_null()
        )[metric_col].to_numpy()
        
        phase2_data = self.data.filter(
            (pl.col('days_to_event') >= phase2_start) &
            (pl.col('days_to_event') <= phase2_end) &
            pl.col(metric_col).is_not_null()
        )[metric_col].to_numpy()
        
        statistic, p_value = mannwhitneyu(phase2_data, phase1_data, 
                                         alternative='two-sided')
        
        return {
            'u_statistic': statistic,
            'p_value': p_value,
            'median_phase1': np.median(phase1_data),
            'median_phase2': np.median(phase2_data),
            'median_diff': np.median(phase2_data) - np.median(phase1_data)
        }