# src/two_risk_framework.py
"""
Two-Risk Framework Implementation
Separates directional news risk (ε) from impact uncertainty (η) as described in the theoretical model
"""

import numpy as np
import polars as pl
from typing import Tuple, Optional, Union, Dict, List
from abc import ABC, abstractmethod
import warnings
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Import compatibility fixes
try:
    from src.polars_compatibility_fixes import safe_clip_quantile, safe_apply, safe_interpolate
except ImportError:
    # Fallback implementations
    def safe_clip_quantile(expr, lower, upper):
        return expr.clip(expr.quantile(lower), expr.quantile(upper))
    
    def safe_apply(expr, func, return_dtype=None):
        try:
            return expr.map_elements(func, return_dtype=return_dtype) if return_dtype else expr.map_elements(func)
        except AttributeError:
            return expr.apply(func)
    
    def safe_interpolate(expr, method="linear"):
        try:
            return expr.interpolate()
        except AttributeError:
            return expr.forward_fill().backward_fill()

class RiskComponent(ABC):
    """Abstract base class for risk components"""
    
    @abstractmethod
    def fit(self, data: pl.DataFrame) -> 'RiskComponent':
        pass
    
    @abstractmethod
    def extract(self, data: pl.DataFrame) -> np.ndarray:
        pass

class DirectionalNewsRisk(RiskComponent):
    """
    Directional News Risk (ε_t): Uncertainty about event outcome direction
    Models the binary/categorical nature of news (positive/negative/neutral)
    """
    
    def __init__(self, jump_threshold: float = 2.0, window_size: int = 5):
        self.jump_threshold = jump_threshold
        self.window_size = window_size
        self.baseline_volatility = None
        self.jump_probabilities = {}
        self.is_fitted = False
        
    def fit(self, data: pl.DataFrame) -> 'DirectionalNewsRisk':
        """
        Fit the directional news risk model to identify jump patterns
        """
        if 'ret' not in data.columns or 'days_to_event' not in data.columns:
            raise ValueError("Data must contain 'ret' and 'days_to_event' columns")
        
        # Calculate rolling volatility to identify jumps
        data_sorted = data.sort(['event_id', 'days_to_event'])
        data_with_vol = data_sorted.with_columns([
            pl.col('ret').rolling_std(window_size=self.window_size, min_periods=2)
            .over('event_id').alias('rolling_vol'),
            pl.col('ret').rolling_mean(window_size=self.window_size, min_periods=2)
            .over('event_id').alias('rolling_mean')
        ])
        
        # Identify jumps as returns exceeding threshold * rolling volatility
        data_with_jumps = data_with_vol.with_columns([
            pl.when(pl.col('rolling_vol') > 0)
            .then(pl.col('ret').abs() / pl.col('rolling_vol'))
            .otherwise(0)
            .alias('normalized_return'),
            
            pl.when(pl.col('rolling_vol') > 0)
            .then(pl.col('ret') / pl.col('rolling_vol') > self.jump_threshold)
            .otherwise(False)
            .alias('is_jump_positive'),
            
            pl.when(pl.col('rolling_vol') > 0)
            .then(pl.col('ret') / pl.col('rolling_vol') < -self.jump_threshold)
            .otherwise(False)
            .alias('is_jump_negative')
        ])
        
        # Calculate baseline volatility and jump probabilities by event phase
        self.baseline_volatility = data_with_jumps.select(pl.col('rolling_vol').mean()).item()
        
        # Calculate jump probabilities for different event phases
        phases = {
            'pre_event': (-15, -1),
            'event_day': (0, 0), 
            'post_event': (1, 15)
        }
        
        for phase_name, (start_day, end_day) in phases.items():
            phase_data = data_with_jumps.filter(
                (pl.col('days_to_event') >= start_day) & 
                (pl.col('days_to_event') <= end_day)
            )
            
            if not phase_data.is_empty():
                self.jump_probabilities[phase_name] = {
                    'positive': phase_data.select(pl.col('is_jump_positive').mean()).item(),
                    'negative': phase_data.select(pl.col('is_jump_negative').mean()).item(),
                    'total': phase_data.select(
                        (pl.col('is_jump_positive') | pl.col('is_jump_negative')).mean()
                    ).item()
                }
        
        self.is_fitted = True
        return self
    
    def extract(self, data: pl.DataFrame) -> np.ndarray:
        """
        Extract directional news risk component from returns
        Returns array of directional risk measures
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before extraction")
            
        data_sorted = data.sort(['event_id', 'days_to_event'])
        data_with_vol = data_sorted.with_columns([
            pl.col('ret').rolling_std(window_size=self.window_size, min_periods=2)
            .over('event_id').alias('rolling_vol')
        ])
        
        # Calculate directional risk as standardized jump intensity
        directional_risk = data_with_vol.with_columns([
            pl.when(pl.col('rolling_vol') > 0)
            .then(pl.col('ret').abs() / pl.col('rolling_vol'))
            .otherwise(0)
            .alias('directional_risk_raw')
        ]).with_columns([
            # Normalize by baseline volatility and apply sigmoid transformation
            safe_apply(
                pl.col('directional_risk_raw'),
                lambda x: 2 / (1 + np.exp(-x / self.baseline_volatility)) - 1 if self.baseline_volatility > 0 else 0,
                return_dtype=pl.Float64
            ).alias('directional_risk')
        ])
        
        return directional_risk.get_column('directional_risk').to_numpy()

class ImpactUncertainty(RiskComponent):
    """
    Impact Uncertainty (η_t): Uncertainty about market reaction magnitude
    Models the unpredictable intensity of market response to news
    """
    
    def __init__(self, lookback_window: int = 20, alpha: float = 0.05):
        self.lookback_window = lookback_window
        self.alpha = alpha  # Significance level for uncertainty detection
        self.baseline_variance = None
        self.impact_models = {}
        self.is_fitted = False
        
    def fit(self, data: pl.DataFrame) -> 'ImpactUncertainty':
        """
        Fit impact uncertainty model using volatility innovations
        """
        if 'ret' not in data.columns or 'days_to_event' not in data.columns:
            raise ValueError("Data must contain 'ret' and 'days_to_event' columns")
        
        # Calculate realized volatility and expected volatility
        data_sorted = data.sort(['event_id', 'days_to_event'])
        data_with_vol = data_sorted.with_columns([
            pl.col('ret').rolling_var(window_size=self.lookback_window, min_periods=5)
            .over('event_id').alias('realized_variance'),
            pl.col('ret').rolling_var(window_size=self.lookback_window, min_periods=5)
            .shift(1).over('event_id').alias('expected_variance')
        ])
        
        # Calculate volatility innovations (η_t proxy)
        data_with_innovations = data_with_vol.with_columns([
            (pl.col('realized_variance') - pl.col('expected_variance'))
            .alias('volatility_innovation'),
            pl.col('realized_variance').alias('h_t'),
            pl.col('expected_variance').alias('h_t_expected')
        ])
        
        # Fit models for different event phases
        phases = {
            'pre_event': (-15, -1),
            'event_window': (-2, 2),
            'post_event': (3, 15)
        }
        
        for phase_name, (start_day, end_day) in phases.items():
            phase_data = data_with_innovations.filter(
                (pl.col('days_to_event') >= start_day) & 
                (pl.col('days_to_event') <= end_day) &
                pl.col('volatility_innovation').is_not_null()
            )
            
            if not phase_data.is_empty():
                innovations = phase_data.get_column('volatility_innovation').to_numpy()
                innovations_clean = innovations[~np.isnan(innovations)]
                
                if len(innovations_clean) > 10:
                    # Fit distribution parameters for impact uncertainty
                    self.impact_models[phase_name] = {
                        'mean': np.mean(innovations_clean),
                        'std': np.std(innovations_clean),
                        'skewness': stats.skew(innovations_clean),
                        'kurtosis': stats.kurtosis(innovations_clean),
                        'var_ratio': np.var(innovations_clean) / max(np.mean(innovations_clean)**2, 1e-8)
                    }
        
        # Calculate baseline variance
        valid_variances = data_with_innovations.filter(
            pl.col('realized_variance').is_not_null()
        ).get_column('realized_variance').to_numpy()
        self.baseline_variance = np.median(valid_variances[~np.isnan(valid_variances)])
        
        self.is_fitted = True
        return self
    
    def extract(self, data: pl.DataFrame) -> np.ndarray:
        """
        Extract impact uncertainty component
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before extraction")
            
        data_sorted = data.sort(['event_id', 'days_to_event'])
        data_with_vol = data_sorted.with_columns([
            pl.col('ret').rolling_var(window_size=self.lookback_window, min_periods=5)
            .over('event_id').alias('realized_variance'),
            pl.col('ret').rolling_var(window_size=self.lookback_window, min_periods=5)
            .shift(1).over('event_id').alias('expected_variance')
        ])
        
        # Calculate impact uncertainty using formula from paper
        impact_uncertainty = data_with_vol.with_columns([
            # h(t) - E_{t^-}[h(t)] from Assumption 7
            (pl.col('realized_variance') - pl.col('expected_variance'))
            .alias('raw_impact_uncertainty')
        ]).with_columns([
            # Normalize and apply transformation
            safe_apply(
                pl.col('raw_impact_uncertainty'),
                lambda x: x / self.baseline_variance if self.baseline_variance > 0 else 0,
                return_dtype=pl.Float64
            ).alias('impact_uncertainty')
        ])
        
        return impact_uncertainty.get_column('impact_uncertainty').to_numpy()

class TwoRiskFramework:
    """
    Integrated Two-Risk Framework combining directional news risk and impact uncertainty
    """
    
    def __init__(self, 
                 directional_params: Optional[Dict] = None,
                 impact_params: Optional[Dict] = None):
        
        # Initialize components with default or custom parameters
        dir_params = directional_params or {}
        impact_params = impact_params or {}
        
        self.directional_risk = DirectionalNewsRisk(**dir_params)
        self.impact_uncertainty = ImpactUncertainty(**impact_params)
        
        self.risk_correlation = None
        self.decomposition_quality = {}
        self.is_fitted = False
        
    def fit(self, data: pl.DataFrame) -> 'TwoRiskFramework':
        """
        Fit the two-risk framework to data
        """
        print("Fitting Two-Risk Framework...")
        
        # Fit individual components
        self.directional_risk.fit(data)
        self.impact_uncertainty.fit(data)
        
        # Analyze risk decomposition quality
        self._analyze_decomposition_quality(data)
        
        self.is_fitted = True
        print("Two-Risk Framework fitted successfully.")
        return self
    
    def extract_risks(self, data: pl.DataFrame) -> Dict[str, np.ndarray]:
        """
        Extract both risk components from data
        """
        if not self.is_fitted:
            raise RuntimeError("Framework must be fitted before risk extraction")
        
        directional = self.directional_risk.extract(data)
        impact = self.impact_uncertainty.extract(data)
        
        # Calculate total risk decomposition
        total_risk_proxy = np.abs(data.get_column('ret').to_numpy())
        
        return {
            'directional_news_risk': directional,
            'impact_uncertainty': impact,
            'total_risk': total_risk_proxy,
            'days_to_event': data.get_column('days_to_event').to_numpy()
        }
    
    def _analyze_decomposition_quality(self, data: pl.DataFrame):
        """
        Analyze the quality of risk decomposition
        """
        try:
            risks = self.extract_risks(data)
            
            # Calculate correlation between risk components
            valid_mask = (~np.isnan(risks['directional_news_risk']) & 
                         ~np.isnan(risks['impact_uncertainty']))
            
            if np.sum(valid_mask) > 10:
                self.risk_correlation = np.corrcoef(
                    risks['directional_news_risk'][valid_mask],
                    risks['impact_uncertainty'][valid_mask]
                )[0, 1]
                
                # Calculate explained variance
                from sklearn.linear_model import LinearRegression
                
                X = np.column_stack([
                    risks['directional_news_risk'][valid_mask],
                    risks['impact_uncertainty'][valid_mask]
                ])
                y = risks['total_risk'][valid_mask]
                
                reg = LinearRegression().fit(X, y)
                explained_var = reg.score(X, y)
                
                self.decomposition_quality = {
                    'risk_correlation': self.risk_correlation,
                    'explained_variance': explained_var,
                    'directional_contribution': np.abs(reg.coef_[0]),
                    'impact_contribution': np.abs(reg.coef_[1])
                }
                
        except Exception as e:
            warnings.warn(f"Could not analyze decomposition quality: {e}")
            self.decomposition_quality = {}
    
    def get_phase_analysis(self, data: pl.DataFrame) -> Dict:
        """
        Analyze risk components by event phases
        """
        if not self.is_fitted:
            raise RuntimeError("Framework must be fitted first")
        
        risks = self.extract_risks(data)
        
        phases = {
            'pre_event': (-15, -1),
            'event_window': (-2, 2), 
            'post_event_rising': (1, 5),
            'post_event_decay': (6, 15)
        }
        
        phase_analysis = {}
        
        for phase_name, (start_day, end_day) in phases.items():
            mask = ((risks['days_to_event'] >= start_day) & 
                   (risks['days_to_event'] <= end_day))
            
            if np.sum(mask) > 0:
                phase_analysis[phase_name] = {
                    'directional_mean': np.nanmean(risks['directional_news_risk'][mask]),
                    'directional_std': np.nanstd(risks['directional_news_risk'][mask]),
                    'impact_mean': np.nanmean(risks['impact_uncertainty'][mask]),
                    'impact_std': np.nanstd(risks['impact_uncertainty'][mask]),
                    'total_risk_mean': np.nanmean(risks['total_risk'][mask]),
                    'sample_size': np.sum(mask)
                }
        
        return phase_analysis
    
    def plot_risk_decomposition(self, data: pl.DataFrame, results_dir: str, file_prefix: str):
        """
        Create visualization of risk decomposition
        """
        if not self.is_fitted:
            raise RuntimeError("Framework must be fitted first")
        
        try:
            import matplotlib.pyplot as plt
            import os
            
            risks = self.extract_risks(data)
            
            # Create risk decomposition plot
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            
            # Plot 1: Risk components over time
            axes[0, 0].plot(risks['days_to_event'], risks['directional_news_risk'], 
                          alpha=0.6, label='Directional News Risk (ε)', color='blue')
            axes[0, 0].plot(risks['days_to_event'], risks['impact_uncertainty'], 
                          alpha=0.6, label='Impact Uncertainty (η)', color='red')
            axes[0, 0].axvline(x=0, color='black', linestyle='--', alpha=0.5, label='Event Day')
            axes[0, 0].set_xlabel('Days to Event')
            axes[0, 0].set_ylabel('Risk Component')
            axes[0, 0].set_title('Two-Risk Components Over Event Window')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
            
            # Plot 2: Risk correlation scatter
            valid_mask = (~np.isnan(risks['directional_news_risk']) & 
                         ~np.isnan(risks['impact_uncertainty']))
            if np.sum(valid_mask) > 10:
                axes[0, 1].scatter(risks['directional_news_risk'][valid_mask], 
                                 risks['impact_uncertainty'][valid_mask], 
                                 alpha=0.5, s=20)
                axes[0, 1].set_xlabel('Directional News Risk (ε)')
                axes[0, 1].set_ylabel('Impact Uncertainty (η)')
                axes[0, 1].set_title(f'Risk Components Correlation\n(ρ = {self.risk_correlation:.3f})')
                axes[0, 1].grid(True, alpha=0.3)
            
            # Plot 3: Phase analysis
            phase_analysis = self.get_phase_analysis(data)
            phases = list(phase_analysis.keys())
            dir_means = [phase_analysis[p]['directional_mean'] for p in phases]
            impact_means = [phase_analysis[p]['impact_mean'] for p in phases]
            
            x_pos = np.arange(len(phases))
            width = 0.35
            
            axes[1, 0].bar(x_pos - width/2, dir_means, width, 
                          label='Directional Risk', alpha=0.8, color='blue')
            axes[1, 0].bar(x_pos + width/2, impact_means, width, 
                          label='Impact Uncertainty', alpha=0.8, color='red')
            axes[1, 0].set_xlabel('Event Phase')
            axes[1, 0].set_ylabel('Average Risk Level')
            axes[1, 0].set_title('Risk Components by Event Phase')
            axes[1, 0].set_xticks(x_pos)
            axes[1, 0].set_xticklabels(phases, rotation=45)
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
            
            # Plot 4: Decomposition quality metrics
            if self.decomposition_quality:
                metrics = ['Risk Correlation', 'Explained Variance', 
                          'Directional Contribution', 'Impact Contribution']
                values = [self.decomposition_quality.get('risk_correlation', 0),
                         self.decomposition_quality.get('explained_variance', 0),
                         self.decomposition_quality.get('directional_contribution', 0),
                         self.decomposition_quality.get('impact_contribution', 0)]
                
                axes[1, 1].bar(metrics, values, alpha=0.8, color=['blue', 'green', 'orange', 'red'])
                axes[1, 1].set_ylabel('Value')
                axes[1, 1].set_title('Risk Decomposition Quality Metrics')
                axes[1, 1].tick_params(axis='x', rotation=45)
                axes[1, 1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plot_filename = os.path.join(results_dir, f"{file_prefix}_two_risk_decomposition.png")
            plt.savefig(plot_filename, dpi=200, bbox_inches='tight')
            plt.close()
            
            print(f"Risk decomposition plot saved to: {plot_filename}")
            
        except Exception as e:
            warnings.warn(f"Could not create risk decomposition plot: {e}")