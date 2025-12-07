import polars as pl
import pandas as pd
import statsmodels.formula.api as smf
from typing import List, Dict

class EventRegressionAnalyzer:
    """Panel regression analysis"""
    
    def __init__(self, data: pl.DataFrame):
        self.data = data
        
    def prepare_regression_data(self, control_vars: List[str] = None) -> pd.DataFrame:
        """Prepare data with phase dummies"""
        
        # Create phase dummies
        reg_data = self.data.with_columns([
            pl.when(pl.col('days_to_event').is_between(0, 5))
            .then(1).otherwise(0).alias('post_rising'),
            
            pl.when(pl.col('days_to_event').is_between(6, 15))
            .then(1).otherwise(0).alias('post_decay')
        ])
        
        # Select columns
        cols = ['rvr', 'post_rising', 'post_decay', 'ticker', 'Event Date']
        if control_vars:
            cols.extend(control_vars)
        
        # Convert to pandas
        return reg_data.select(cols).to_pandas()
    
    def run_panel_regression(self,
                            dependent_var: str = 'rvr',
                            control_vars: List[str] = None,
                            cluster_vars: List[str] = ['ticker'],
                            firm_fe: bool = False,
                            time_fe: bool = False) -> Dict:
        """
        Run OLS with clustered standard errors
        
        Model: RVR = β0 + β1*Post_Rising + β2*Post_Decay + Controls + FE + ε
        """
        
        data = self.prepare_regression_data(control_vars)
        
        # Build formula
        formula_parts = [dependent_var, '~', 'post_rising + post_decay']
        
        if control_vars:
            for ctrl in control_vars:
                if ctrl in ['size_quintile', 'industry', 'liquidity_quintile']:
                    formula_parts.append(f' + C({ctrl})')
                else:
                    formula_parts.append(f' + {ctrl}')
        
        if firm_fe:
            formula_parts.append(' + C(ticker)')
        
        if time_fe:
            data['year_month'] = pd.to_datetime(data['Event Date']).dt.strftime('%Y-%m')
            formula_parts.append(' + C(year_month)')
        
        formula = ''.join(formula_parts)
        
        # Drop rows with NaN in dependent variable before regression
        data = data.dropna(subset=[dependent_var])
        
        # Run regression
        model = smf.ols(formula, data=data).fit()
        
        # Cluster standard errors
        if cluster_vars:
            # Get the indices used in the regression (after dropping NaN)
            used_indices = model.model.data.row_labels
            cluster_groups = data.loc[used_indices, cluster_vars].apply(
                lambda x: '_'.join(x.astype(str)), axis=1
            )
            model = model.get_robustcov_results(
                cov_type='cluster',
                groups=cluster_groups
            )
        
        # Get parameter names for indexing
        param_names = model.model.exog_names
        rising_idx = param_names.index('post_rising') if 'post_rising' in param_names else None
        
        return {
            'model': model,
            'summary': model.summary(),
            'coef_rising': model.params[rising_idx] if rising_idx is not None else None,
            'se_rising': model.bse[rising_idx] if rising_idx is not None else None,
            't_rising': model.tvalues[rising_idx] if rising_idx is not None else None,
            'p_rising': model.pvalues[rising_idx] if rising_idx is not None else None,
            'n_obs': int(model.nobs),
            'r_squared': model.rsquared
        }