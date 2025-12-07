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
            data['year_month'] = pd.to_datetime(data['Event Date']).dt.to_period('M')
            formula_parts.append(' + C(year_month)')
        
        formula = ''.join(formula_parts)
        
        # Run regression
        model = smf.ols(formula, data=data).fit()
        
        # Cluster standard errors
        if cluster_vars:
            cluster_groups = data[cluster_vars].apply(
                lambda x: '_'.join(x.astype(str)), axis=1
            )
            model = model.get_robustcov_results(
                cov_type='cluster',
                groups=cluster_groups
            )
        
        return {
            'model': model,
            'summary': model.summary(),
            'coef_rising': model.params.get('post_rising'),
            'se_rising': model.bse.get('post_rising'),
            't_rising': model.tvalues.get('post_rising'),
            'p_rising': model.pvalues.get('post_rising'),
            'n_obs': int(model.nobs),
            'r_squared': model.rsquared
        }