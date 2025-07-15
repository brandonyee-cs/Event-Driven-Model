# src/statistical_testing.py
"""
Enhanced Statistical Testing Framework
Implements comprehensive hypothesis testing for the theoretical model
with proper econometric methods, bootstrap confidence intervals, and robustness checks
"""

import numpy as np
import polars as pl
from typing import Dict, List, Optional, Tuple, Union, Any
from scipy import stats
from scipy.stats import ttest_ind, ttest_rel, levene, jarque_bera, normaltest
from sklearn.utils import resample
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import r2_score
import warnings
from dataclasses import dataclass
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
import seaborn as sns
import os

@dataclass
class HypothesisTestResult:
    """Container for hypothesis test results"""
    hypothesis_name: str
    test_statistic: float
    p_value: float
    critical_value: Optional[float] = None
    confidence_interval: Optional[Tuple[float, float]] = None
    effect_size: Optional[float] = None
    power: Optional[float] = None
    is_significant: bool = False
    alpha: float = 0.05
    method: str = "unknown"
    sample_size: int = 0
    degrees_freedom: Optional[int] = None
    additional_info: Dict[str, Any] = None

    def __post_init__(self):
        if self.additional_info is None:
            self.additional_info = {}
        self.is_significant = self.p_value < self.alpha if self.p_value is not None else False

class StatisticalTest(ABC):
    """Abstract base class for statistical tests"""
    
    @abstractmethod
    def run_test(self, data: pl.DataFrame, **kwargs) -> HypothesisTestResult:
        pass
    
    @abstractmethod
    def get_description(self) -> str:
        pass

class RVRPeakTest(StatisticalTest):
    """
    Test Hypothesis 1: RVR peaks during post-event rising phase
    """
    
    def __init__(self, 
                 rising_phase: Tuple[int, int] = (0, 5),
                 pre_phase: Tuple[int, int] = (-15, -1),
                 decay_phase: Tuple[int, int] = (6, 15),
                 alpha: float = 0.05):
        self.rising_phase = rising_phase
        self.pre_phase = pre_phase
        self.decay_phase = decay_phase
        self.alpha = alpha
    
    def run_test(self, data: pl.DataFrame, rvr_column: str = 'rvr', **kwargs) -> HypothesisTestResult:
        """
        Test if RVR peaks in rising phase vs pre-event and decay phases
        """
        if rvr_column not in data.columns or 'days_to_event' not in data.columns:
            raise ValueError(f"Required columns missing: {rvr_column}, days_to_event")
        
        # Extract RVR values for each phase
        pre_rvr = self._extract_phase_data(data, self.pre_phase, rvr_column)
        rising_rvr = self._extract_phase_data(data, self.rising_phase, rvr_column)
        decay_rvr = self._extract_phase_data(data, self.decay_phase, rvr_column)
        
        # Calculate phase means
        pre_mean = np.nanmean(pre_rvr) if len(pre_rvr) > 0 else 0
        rising_mean = np.nanmean(rising_rvr) if len(rising_rvr) > 0 else 0
        decay_mean = np.nanmean(decay_rvr) if len(decay_rvr) > 0 else 0
        
        # Test: rising > pre AND rising > decay
        test1_stat, test1_p = self._compare_phases(rising_rvr, pre_rvr, "Rising vs Pre")
        test2_stat, test2_p = self._compare_phases(rising_rvr, decay_rvr, "Rising vs Decay")
        
        # Combined test using Bonferroni correction
        combined_p = min(1.0, min(test1_p, test2_p) * 2)  # Bonferroni correction
        
        # Effect size (Cohen's d)
        effect_size_1 = self._cohens_d(rising_rvr, pre_rvr)
        effect_size_2 = self._cohens_d(rising_rvr, decay_rvr)
        avg_effect_size = (effect_size_1 + effect_size_2) / 2
        
        # Bootstrap confidence interval for rising phase mean
        ci_lower, ci_upper = self._bootstrap_ci(rising_rvr, statistic=np.nanmean)
        
        return HypothesisTestResult(
            hypothesis_name="H1: RVR peaks during post-event rising phase",
            test_statistic=min(test1_stat, test2_stat),
            p_value=combined_p,
            confidence_interval=(ci_lower, ci_upper),
            effect_size=avg_effect_size,
            alpha=self.alpha,
            method="Two-sample t-tests with Bonferroni correction",
            sample_size=len(rising_rvr) + len(pre_rvr) + len(decay_rvr),
            additional_info={
                'pre_event_mean': pre_mean,
                'rising_phase_mean': rising_mean,
                'decay_phase_mean': decay_mean,
                'rising_vs_pre_p': test1_p,
                'rising_vs_decay_p': test2_p,
                'effect_size_vs_pre': effect_size_1,
                'effect_size_vs_decay': effect_size_2,
                'phase_sample_sizes': {
                    'pre': len(pre_rvr),
                    'rising': len(rising_rvr),
                    'decay': len(decay_rvr)
                }
            }
        )
    
    def _extract_phase_data(self, data: pl.DataFrame, phase: Tuple[int, int], column: str) -> np.ndarray:
        """Extract data for a specific phase"""
        phase_data = data.filter(
            (pl.col('days_to_event') >= phase[0]) & 
            (pl.col('days_to_event') <= phase[1])
        ).get_column(column).to_numpy()
        
        return phase_data[~np.isnan(phase_data)]
    
    def _compare_phases(self, phase1: np.ndarray, phase2: np.ndarray, comparison_name: str) -> Tuple[float, float]:
        """Compare two phases using appropriate statistical test"""
        if len(phase1) == 0 or len(phase2) == 0:
            return 0.0, 1.0
        
        # Check for equal variances
        if len(phase1) > 2 and len(phase2) > 2:
            _, levene_p = levene(phase1, phase2)
            equal_var = levene_p > 0.05
        else:
            equal_var = True
        
        # Perform t-test
        if len(phase1) > 1 and len(phase2) > 1:
            stat, p_value = ttest_ind(phase1, phase2, equal_var=equal_var, alternative='greater')
            return stat, p_value
        else:
            return 0.0, 1.0
    
    def _cohens_d(self, group1: np.ndarray, group2: np.ndarray) -> float:
        """Calculate Cohen's d effect size"""
        if len(group1) == 0 or len(group2) == 0:
            return 0.0
        
        n1, n2 = len(group1), len(group2)
        var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
        
        # Pooled standard deviation
        pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
        
        if pooled_std == 0:
            return 0.0
        
        return (np.mean(group1) - np.mean(group2)) / pooled_std
    
    def _bootstrap_ci(self, data: np.ndarray, statistic=np.mean, n_bootstrap: int = 1000, confidence: float = 0.95) -> Tuple[float, float]:
        """Calculate bootstrap confidence interval"""
        if len(data) == 0:
            return 0.0, 0.0
        
        bootstrap_stats = []
        for _ in range(n_bootstrap):
            sample = resample(data, n_samples=len(data), random_state=None)
            bootstrap_stats.append(statistic(sample))
        
        alpha = 1 - confidence
        lower_percentile = (alpha/2) * 100
        upper_percentile = (1 - alpha/2) * 100
        
        ci_lower = np.percentile(bootstrap_stats, lower_percentile)
        ci_upper = np.percentile(bootstrap_stats, upper_percentile)
        
        return ci_lower, ci_upper
    
    def get_description(self) -> str:
        return "Tests whether RVR peaks during post-event rising phase compared to pre-event and decay phases"

class VolatilityInnovationTest(StatisticalTest):
    """
    Test Hypothesis 2: GARCH volatility innovations predict returns
    """
    
    def __init__(self, 
                 prediction_horizons: List[int] = [1, 3, 5],
                 alpha: float = 0.05):
        self.prediction_horizons = prediction_horizons
        self.alpha = alpha
    
    def run_test(self, data: pl.DataFrame, **kwargs) -> HypothesisTestResult:
        """
        Test predictive power of volatility innovations
        """
        required_cols = ['volatility_innovation', 'ret', 'days_to_event']
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Sort data properly
        data_sorted = data.sort(['event_id', 'days_to_event'])
        
        results = {}
        combined_r2 = []
        combined_p_values = []
        
        for horizon in self.prediction_horizons:
            # Create forward returns
            data_with_forward = data_sorted.with_columns([
                pl.col('ret').shift(-horizon).over('event_id').alias(f'forward_ret_{horizon}'),
                pl.col('volatility_innovation').alias('vol_innovation')
            ])
            
            # Extract valid observations
            valid_data = data_with_forward.filter(
                pl.col('vol_innovation').is_not_null() & 
                pl.col(f'forward_ret_{horizon}').is_not_null()
            )
            
            if valid_data.height < 20:
                continue
            
            X = valid_data.get_column('vol_innovation').to_numpy().reshape(-1, 1)
            y = valid_data.get_column(f'forward_ret_{horizon}').to_numpy()
            
            # Run regression
            model = LinearRegression()
            model.fit(X, y)
            y_pred = model.predict(X)
            
            # Calculate statistics
            r2 = r2_score(y, y_pred)
            n = len(y)
            
            # F-test for regression significance
            mse_residual = np.mean((y - y_pred)**2)
            mse_total = np.var(y)
            f_stat = (r2 / (1 - r2)) * (n - 2) if r2 < 1 else np.inf
            p_value = 1 - stats.f.cdf(f_stat, 1, n - 2) if np.isfinite(f_stat) else 0.0
            
            results[f'horizon_{horizon}'] = {
                'r2': r2,
                'f_statistic': f_stat,
                'p_value': p_value,
                'coefficient': model.coef_[0],
                'sample_size': n
            }
            
            combined_r2.append(r2)
            combined_p_values.append(p_value)
        
        # Overall test result
        avg_r2 = np.mean(combined_r2) if combined_r2 else 0
        min_p_value = min(combined_p_values) if combined_p_values else 1.0
        
        # Bonferroni correction
        adjusted_p = min(1.0, min_p_value * len(combined_p_values)) if combined_p_values else 1.0
        
        return HypothesisTestResult(
            hypothesis_name="H2: Volatility innovations predict returns",
            test_statistic=max([results[k]['f_statistic'] for k in results.keys()]) if results else 0,
            p_value=adjusted_p,
            effect_size=avg_r2,
            alpha=self.alpha,
            method="Linear regression with F-test",
            sample_size=sum([results[k]['sample_size'] for k in results.keys()]) if results else 0,
            additional_info={
                'horizon_results': results,
                'average_r2': avg_r2,
                'best_horizon': max(results.keys(), key=lambda k: results[k]['r2']) if results else None
            }
        )
    
    def get_description(self) -> str:
        return "Tests whether GARCH volatility innovations have predictive power for future returns"

class AsymmetricBiasTest(StatisticalTest):
    """
    Test Hypothesis 3: Asymmetric bias effects (positive vs negative events)
    """
    
    def __init__(self, alpha: float = 0.05):
        self.alpha = alpha
    
    def run_test(self, data: pl.DataFrame, **kwargs) -> HypothesisTestResult:
        """
        Test for asymmetric bias effects
        """
        required_cols = ['ret', 'days_to_event', 'event_id']
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Classify events as positive or negative based on event-day return
        event_day_data = data.filter(pl.col('days_to_event') == 0)
        
        if event_day_data.height == 0:
            return HypothesisTestResult(
                hypothesis_name="H3: Asymmetric bias effects",
                test_statistic=0.0,
                p_value=1.0,
                alpha=self.alpha,
                method="Event classification",
                sample_size=0,
                additional_info={'error': 'No event day data found'}
            )
        
        # Classify events
        positive_events = event_day_data.filter(pl.col('ret') > 0).get_column('event_id')
        negative_events = event_day_data.filter(pl.col('ret') < 0).get_column('event_id')
        
        # Calculate post-event bias for each type
        post_event_window = (1, 5)
        
        positive_post_returns = self._calculate_post_event_bias(data, positive_events, post_event_window)
        negative_post_returns = self._calculate_post_event_bias(data, negative_events, post_event_window)
        
        # Test for difference in bias
        if len(positive_post_returns) > 1 and len(negative_post_returns) > 1:
            stat, p_value = ttest_ind(positive_post_returns, negative_post_returns)
            effect_size = self._cohens_d(positive_post_returns, negative_post_returns)
        else:
            stat, p_value = 0.0, 1.0
            effect_size = 0.0
        
        return HypothesisTestResult(
            hypothesis_name="H3: Asymmetric bias effects",
            test_statistic=stat,
            p_value=p_value,
            effect_size=effect_size,
            alpha=self.alpha,
            method="Two-sample t-test",
            sample_size=len(positive_post_returns) + len(negative_post_returns),
            additional_info={
                'positive_events_count': len(positive_events),
                'negative_events_count': len(negative_events),
                'positive_post_bias_mean': np.mean(positive_post_returns) if len(positive_post_returns) > 0 else 0,
                'negative_post_bias_mean': np.mean(negative_post_returns) if len(negative_post_returns) > 0 else 0
            }
        )
    
    def _calculate_post_event_bias(self, data: pl.DataFrame, event_ids: pl.Series, window: Tuple[int, int]) -> np.ndarray:
        """Calculate post-event bias for specific events"""
        if event_ids.is_empty():
            return np.array([])
        
        post_event_data = data.filter(
            pl.col('event_id').is_in(event_ids) &
            (pl.col('days_to_event') >= window[0]) &
            (pl.col('days_to_event') <= window[1])
        )
        
        if post_event_data.is_empty():
            return np.array([])
        
        # Calculate average return for each event in post-event window
        event_post_returns = post_event_data.group_by('event_id').agg(
            pl.mean('ret').alias('avg_post_return')
        ).get_column('avg_post_return').to_numpy()
        
        return event_post_returns[~np.isnan(event_post_returns)]
    
    def _cohens_d(self, group1: np.ndarray, group2: np.ndarray) -> float:
        """Calculate Cohen's d effect size"""
        if len(group1) == 0 or len(group2) == 0:
            return 0.0
        
        n1, n2 = len(group1), len(group2)
        var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
        
        pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
        
        if pooled_std == 0:
            return 0.0
        
        return (np.mean(group1) - np.mean(group2)) / pooled_std
    
    def get_description(self) -> str:
        return "Tests for asymmetric bias effects between positive and negative events"

class RobustnessTest:
    """
    Robustness testing framework
    """
    
    def __init__(self):
        self.subsample_ratios = [0.5, 0.7, 0.9]
        self.bootstrap_iterations = 500
        
    def run_robustness_checks(self, 
                            data: pl.DataFrame,
                            primary_test: StatisticalTest,
                            **test_kwargs) -> Dict[str, Any]:
        """
        Run robustness checks on a primary test
        """
        robustness_results = {}
        
        # 1. Subsample robustness
        robustness_results['subsample'] = self._subsample_robustness(data, primary_test, **test_kwargs)
        
        # 2. Bootstrap robustness
        robustness_results['bootstrap'] = self._bootstrap_robustness(data, primary_test, **test_kwargs)
        
        # 3. Outlier robustness
        robustness_results['outlier'] = self._outlier_robustness(data, primary_test, **test_kwargs)
        
        # 4. Time period robustness
        if 'Event Date' in data.columns:
            robustness_results['temporal'] = self._temporal_robustness(data, primary_test, **test_kwargs)
        
        return robustness_results
    
    def _subsample_robustness(self, data: pl.DataFrame, test: StatisticalTest, **kwargs) -> Dict:
        """Test robustness across different subsamples"""
        results = []
        
        for ratio in self.subsample_ratios:
            for i in range(10):  # 10 random subsamples per ratio
                subsample = data.sample(fraction=ratio, seed=i*42)
                try:
                    result = test.run_test(subsample, **kwargs)
                    results.append({
                        'ratio': ratio,
                        'iteration': i,
                        'p_value': result.p_value,
                        'effect_size': result.effect_size,
                        'is_significant': result.is_significant
                    })
                except Exception as e:
                    continue
        
        if results:
            results_df = pl.DataFrame(results)
            return {
                'mean_p_value': results_df.get_column('p_value').mean(),
                'std_p_value': results_df.get_column('p_value').std(),
                'mean_effect_size': results_df.get_column('effect_size').mean(),
                'significance_rate': results_df.get_column('is_significant').mean(),
                'sample_count': len(results)
            }
        else:
            return {'error': 'No valid subsample results'}
    
    def _bootstrap_robustness(self, data: pl.DataFrame, test: StatisticalTest, **kwargs) -> Dict:
        """Test robustness using bootstrap resampling"""
        bootstrap_results = []
        
        for i in range(self.bootstrap_iterations):
            try:
                bootstrap_sample = data.sample(fraction=1.0, with_replacement=True, seed=i)
                result = test.run_test(bootstrap_sample, **kwargs)
                bootstrap_results.append({
                    'p_value': result.p_value,
                    'effect_size': result.effect_size,
                    'is_significant': result.is_significant
                })
            except Exception:
                continue
        
        if bootstrap_results:
            results_df = pl.DataFrame(bootstrap_results)
            p_values = results_df.get_column('p_value').to_numpy()
            effect_sizes = results_df.get_column('effect_size').to_numpy()
            
            return {
                'mean_p_value': np.mean(p_values),
                'p_value_ci': (np.percentile(p_values, 2.5), np.percentile(p_values, 97.5)),
                'mean_effect_size': np.mean(effect_sizes),
                'effect_size_ci': (np.percentile(effect_sizes, 2.5), np.percentile(effect_sizes, 97.5)),
                'significance_rate': results_df.get_column('is_significant').mean(),
                'bootstrap_count': len(bootstrap_results)
            }
        else:
            return {'error': 'No valid bootstrap results'}
    
    def _outlier_robustness(self, data: pl.DataFrame, test: StatisticalTest, **kwargs) -> Dict:
        """Test robustness to outliers by winsorizing"""
        winsorize_levels = [0.01, 0.05, 0.1]  # 1%, 5%, 10% winsorization
        results = []
        
        numeric_columns = [col for col in data.columns if data[col].dtype in [pl.Float32, pl.Float64, pl.Int32, pl.Int64]]
        
        for level in winsorize_levels:
            try:
                # Winsorize numeric columns
                winsorized_data = data.clone()
                for col in numeric_columns:
                    lower_bound = data.select(pl.col(col).quantile(level)).item()
                    upper_bound = data.select(pl.col(col).quantile(1 - level)).item()
                    
                    winsorized_data = winsorized_data.with_columns(
                        pl.col(col).clip(lower_bound, upper_bound)
                    )
                
                result = test.run_test(winsorized_data, **kwargs)
                results.append({
                    'winsorize_level': level,
                    'p_value': result.p_value,
                    'effect_size': result.effect_size,
                    'is_significant': result.is_significant
                })
            except Exception:
                continue
        
        if results:
            return {
                'results_by_level': results,
                'consistent_significance': all(r['is_significant'] for r in results) or all(not r['is_significant'] for r in results)
            }
        else:
            return {'error': 'No valid winsorization results'}
    
    def _temporal_robustness(self, data: pl.DataFrame, test: StatisticalTest, **kwargs) -> Dict:
        """Test robustness across different time periods"""
        try:
            # Split data into time periods
            data_with_year = data.with_columns(
                pl.col('Event Date').dt.year().alias('year')
            )
            
            years = sorted(data_with_year.get_column('year').unique().to_list())
            
            if len(years) < 3:
                return {'error': 'Insufficient time periods for temporal robustness'}
            
            # Split into early, middle, late periods
            n_years = len(years)
            early_years = years[:n_years//3]
            middle_years = years[n_years//3:2*n_years//3]
            late_years = years[2*n_years//3:]
            
            periods = {
                'early': early_years,
                'middle': middle_years,
                'late': late_years
            }
            
            results = {}
            for period_name, period_years in periods.items():
                period_data = data_with_year.filter(pl.col('year').is_in(period_years))
                if period_data.height > 50:  # Minimum sample size
                    try:
                        result = test.run_test(period_data, **kwargs)
                        results[period_name] = {
                            'years': period_years,
                            'p_value': result.p_value,
                            'effect_size': result.effect_size,
                            'is_significant': result.is_significant,
                            'sample_size': period_data.height
                        }
                    except Exception:
                        continue
            
            return {
                'period_results': results,
                'consistent_across_periods': len(set(r['is_significant'] for r in results.values())) <= 1
            }
            
        except Exception as e:
            return {'error': f'Temporal robustness failed: {str(e)}'}

class ComprehensiveTestSuite:
    """
    Comprehensive test suite for the enhanced event study model
    """
    
    def __init__(self, alpha: float = 0.05):
        self.alpha = alpha
        self.tests = {
            'rvr_peak': RVRPeakTest(alpha=alpha),
            'volatility_innovation': VolatilityInnovationTest(alpha=alpha),
            'asymmetric_bias': AsymmetricBiasTest(alpha=alpha)
        }
        self.robustness_tester = RobustnessTest()
        
    def run_all_tests(self, 
                     data: pl.DataFrame,
                     run_robustness: bool = True,
                     **test_kwargs) -> Dict[str, Any]:
        """
        Run all statistical tests
        """
        print("Running comprehensive statistical test suite...")
        
        results = {}
        
        # Run primary tests
        for test_name, test in self.tests.items():
            print(f"  Running {test_name} test...")
            try:
                result = test.run_test(data, **test_kwargs)
                results[test_name] = result
                
                # Run robustness checks if requested
                if run_robustness:
                    print(f"    Running robustness checks for {test_name}...")
                    robustness = self.robustness_tester.run_robustness_checks(
                        data, test, **test_kwargs
                    )
                    results[f'{test_name}_robustness'] = robustness
                    
            except Exception as e:
                warnings.warn(f"Test {test_name} failed: {e}")
                results[test_name] = HypothesisTestResult(
                    hypothesis_name=f"{test_name} (failed)",
                    test_statistic=0.0,
                    p_value=1.0,
                    alpha=self.alpha,
                    method="failed",
                    sample_size=0,
                    additional_info={'error': str(e)}
                )
        
        # Generate summary
        results['summary'] = self._generate_test_summary(results)
        
        print("Statistical testing complete.")
        return results
    
    def _generate_test_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary of test results"""
        summary = {
            'total_tests': 0,
            'significant_tests': 0,
            'test_results': {},
            'overall_conclusion': 'inconclusive'
        }
        
        primary_tests = [k for k in results.keys() if not k.endswith('_robustness') and k != 'summary']
        
        for test_name in primary_tests:
            result = results[test_name]
            if isinstance(result, HypothesisTestResult):
                summary['total_tests'] += 1
                summary['test_results'][test_name] = {
                    'significant': result.is_significant,
                    'p_value': result.p_value,
                    'effect_size': result.effect_size
                }
                
                if result.is_significant:
                    summary['significant_tests'] += 1
        
        # Overall conclusion
        if summary['total_tests'] > 0:
            significance_rate = summary['significant_tests'] / summary['total_tests']
            if significance_rate >= 0.67:  # 2/3 of tests significant
                summary['overall_conclusion'] = 'strong_support'
            elif significance_rate >= 0.33:  # 1/3 of tests significant
                summary['overall_conclusion'] = 'moderate_support'
            else:
                summary['overall_conclusion'] = 'limited_support'
        
        return summary
    
    def plot_test_results(self, 
                         results: Dict[str, Any],
                         results_dir: str,
                         file_prefix: str):
        """
        Create visualization of test results
        """
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            
            # Plot 1: P-values
            primary_tests = [k for k in results.keys() if not k.endswith('_robustness') and k != 'summary']
            test_names = []
            p_values = []
            significance = []
            
            for test_name in primary_tests:
                result = results[test_name]
                if isinstance(result, HypothesisTestResult):
                    test_names.append(test_name.replace('_', ' ').title())
                    p_values.append(result.p_value)
                    significance.append(result.is_significant)
            
            if test_names:
                colors = ['red' if sig else 'blue' for sig in significance]
                bars = axes[0, 0].bar(test_names, p_values, color=colors, alpha=0.7)
                axes[0, 0].axhline(y=self.alpha, color='black', linestyle='--', label=f'α = {self.alpha}')
                axes[0, 0].set_ylabel('P-value')
                axes[0, 0].set_title('Hypothesis Test P-values')
                axes[0, 0].legend()
                axes[0, 0].tick_params(axis='x', rotation=45)
                
                # Add p-value labels
                for bar, p_val in zip(bars, p_values):
                    height = bar.get_height()
                    axes[0, 0].text(bar.get_x() + bar.get_width()/2., height + 0.001,
                                   f'{p_val:.3f}', ha='center', va='bottom')
            
            # Plot 2: Effect sizes
            effect_sizes = []
            for test_name in primary_tests:
                result = results[test_name]
                if isinstance(result, HypothesisTestResult) and result.effect_size is not None:
                    effect_sizes.append(result.effect_size)
                else:
                    effect_sizes.append(0)
            
            if test_names and effect_sizes:
                axes[0, 1].bar(test_names, effect_sizes, alpha=0.7, color='green')
                axes[0, 1].set_ylabel('Effect Size')
                axes[0, 1].set_title('Effect Sizes')
                axes[0, 1].tick_params(axis='x', rotation=45)
            
            # Plot 3: Robustness summary
            robustness_data = []
            for test_name in primary_tests:
                rob_key = f'{test_name}_robustness'
                if rob_key in results:
                    rob_result = results[rob_key]
                    if 'bootstrap' in rob_result and 'significance_rate' in rob_result['bootstrap']:
                        robustness_data.append(rob_result['bootstrap']['significance_rate'])
                    else:
                        robustness_data.append(0)
                else:
                    robustness_data.append(0)
            
            if test_names and robustness_data:
                axes[1, 0].bar(test_names, robustness_data, alpha=0.7, color='orange')
                axes[1, 0].set_ylabel('Bootstrap Significance Rate')
                axes[1, 0].set_title('Robustness: Bootstrap Significance Rates')
                axes[1, 0].tick_params(axis='x', rotation=45)
                axes[1, 0].set_ylim(0, 1)
            
            # Plot 4: Summary statistics
            if 'summary' in results:
                summary = results['summary']
                total_tests = summary.get('total_tests', 0)
                significant_tests = summary.get('significant_tests', 0)
                
                categories = ['Total Tests', 'Significant Tests', 'Non-significant']
                values = [total_tests, significant_tests, total_tests - significant_tests]
                colors = ['gray', 'red', 'blue']
                
                axes[1, 1].pie(values, labels=categories, colors=colors, autopct='%1.0f', startangle=90)
                axes[1, 1].set_title(f'Test Summary\n({summary.get("overall_conclusion", "").replace("_", " ").title()})')
            
            plt.tight_layout()
            plot_filename = os.path.join(results_dir, f"{file_prefix}_statistical_tests.png")
            plt.savefig(plot_filename, dpi=200, bbox_inches='tight')
            plt.close()
            
            print(f"Statistical test results plot saved to: {plot_filename}")
            
        except Exception as e:
            warnings.warn(f"Could not create statistical test plot: {e}")
    
    def save_test_results(self,
                         results: Dict[str, Any],
                         results_dir: str,
                         file_prefix: str):
        """
        Save test results to CSV files
        """
        try:
            # Create summary table
            summary_data = []
            primary_tests = [k for k in results.keys() if not k.endswith('_robustness') and k != 'summary']
            
            for test_name in primary_tests:
                result = results[test_name]
                if isinstance(result, HypothesisTestResult):
                    summary_data.append({
                        'test_name': test_name,
                        'hypothesis': result.hypothesis_name,
                        'test_statistic': result.test_statistic,
                        'p_value': result.p_value,
                        'is_significant': result.is_significant,
                        'effect_size': result.effect_size,
                        'method': result.method,
                        'sample_size': result.sample_size,
                        'alpha': result.alpha
                    })
            
            if summary_data:
                summary_df = pl.DataFrame(summary_data)
                summary_df.write_csv(os.path.join(results_dir, f"{file_prefix}_test_summary.csv"))
            
            # Save detailed results for each test
            for test_name in primary_tests:
                result = results[test_name]
                if isinstance(result, HypothesisTestResult) and result.additional_info:
                    detail_data = [{'metric': k, 'value': str(v)} for k, v in result.additional_info.items()]
                    if detail_data:
                        detail_df = pl.DataFrame(detail_data)
                        detail_df.write_csv(os.path.join(results_dir, f"{file_prefix}_{test_name}_details.csv"))
            
            print(f"Statistical test results saved to: {results_dir}")
            
        except Exception as e:
            warnings.warn(f"Could not save statistical test results: {e}")