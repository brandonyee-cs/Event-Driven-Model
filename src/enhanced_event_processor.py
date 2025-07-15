# src/enhanced_event_processor.py
"""
Enhanced Event Processor that integrates the theoretical model components
Builds upon the existing event_processor.py with:
- Two-risk framework integration
- Heterogeneous investor modeling  
- Enhanced portfolio optimization
- Market equilibrium simulation
"""

import polars as pl
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import warnings
from typing import Dict, List, Optional, Tuple, Any
import gc

# Import existing components
try:
    from src.event_processor import EventDataLoader, EventFeatureEngineer, EventAnalysis
    from src.models import GARCHModel, GJRGARCHModel, ThreePhaseVolatilityModel
except ImportError:
    # Fallback for direct execution
    import sys
    sys.path.append('.')
    from event_processor import EventDataLoader, EventFeatureEngineer, EventAnalysis
    from models import GARCHModel, GJRGARCHModel, ThreePhaseVolatilityModel

# Import new enhanced components
try:
    from src.two_risk_framework import TwoRiskFramework, DirectionalNewsRisk, ImpactUncertainty
    from src.investor_heterogeneity import (HeterogeneousInvestorMarket, InformedInvestor, 
                                          UninformedInvestor, LiquidityTrader, InvestorParameters)
    from src.portfolio_optimization import (PortfolioOptimizationFramework, OptimizationParameters,
                                          AssetParameters)
except ImportError:
    warnings.warn("Enhanced components not available. Some features will be limited.")
    TwoRiskFramework = None
    HeterogeneousInvestorMarket = None
    PortfolioOptimizationFramework = None

class EnhancedEventAnalysis(EventAnalysis):
    """
    Enhanced Event Analysis incorporating the theoretical model components
    """
    
    def __init__(self, 
                 data_loader: EventDataLoader, 
                 feature_engineer: EventFeatureEngineer,
                 enable_two_risk: bool = True,
                 enable_heterogeneous_investors: bool = True,
                 enable_portfolio_optimization: bool = True):
        
        super().__init__(data_loader, feature_engineer)
        
        # Enhanced components
        self.enable_two_risk = enable_two_risk and TwoRiskFramework is not None
        self.enable_heterogeneous_investors = enable_heterogeneous_investors and HeterogeneousInvestorMarket is not None
        self.enable_portfolio_optimization = enable_portfolio_optimization and PortfolioOptimizationFramework is not None
        
        # Initialize enhanced frameworks
        if self.enable_two_risk:
            self.two_risk_framework = TwoRiskFramework()
        else:
            self.two_risk_framework = None
            
        if self.enable_heterogeneous_investors:
            self.investor_market = HeterogeneousInvestorMarket()
        else:
            self.investor_market = None
            
        if self.enable_portfolio_optimization:
            self.portfolio_optimizer = PortfolioOptimizationFramework()
        else:
            self.portfolio_optimizer = None
            
        # Storage for enhanced results
        self.risk_decomposition_results = {}
        self.market_simulation_results = {}
        self.portfolio_optimization_results = {}
        
    def run_comprehensive_analysis(self,
                                 results_dir: str,
                                 file_prefix: str = "enhanced_event",
                                 analysis_window: Tuple[int, int] = (-15, 15),
                                 garch_type: str = 'gjr',
                                 optimistic_bias: float = 0.01,
                                 risk_free_rate: float = 0.0) -> Dict[str, Any]:
        """
        Run comprehensive analysis combining all theoretical model components
        """
        print(f"\n=== Running Comprehensive Enhanced Analysis ===")
        
        # Ensure data is loaded
        if self.data is None:
            print("Loading and preparing data...")
            self.data = self.load_and_prepare_data(run_feature_engineering=False)
            if self.data is None:
                print("Error: Failed to load data")
                return {}
        
        results = {}
        
        # 1. Traditional Analysis (baseline)
        print("1. Running traditional three-phase volatility analysis...")
        traditional_results = self._run_traditional_analysis(
            results_dir, file_prefix, analysis_window, garch_type, optimistic_bias, risk_free_rate
        )
        results['traditional'] = traditional_results
        
        # 2. Two-Risk Framework Analysis
        if self.enable_two_risk:
            print("2. Running two-risk framework analysis...")
            risk_results = self._run_two_risk_analysis(results_dir, file_prefix)
            results['two_risk'] = risk_results
        else:
            print("2. Two-risk framework disabled")
            results['two_risk'] = {}
        
        # 3. Heterogeneous Investor Analysis
        if self.enable_heterogeneous_investors:
            print("3. Running heterogeneous investor market simulation...")
            investor_results = self._run_investor_analysis(results_dir, file_prefix, analysis_window)
            results['heterogeneous_investors'] = investor_results
        else:
            print("3. Heterogeneous investor analysis disabled")
            results['heterogeneous_investors'] = {}
        
        # 4. Enhanced Portfolio Optimization
        if self.enable_portfolio_optimization:
            print("4. Running enhanced portfolio optimization...")
            portfolio_results = self._run_portfolio_optimization(
                results_dir, file_prefix, analysis_window, optimistic_bias, risk_free_rate
            )
            results['portfolio_optimization'] = portfolio_results
        else:
            print("4. Portfolio optimization disabled")
            results['portfolio_optimization'] = {}
        
        # 5. Integrated Analysis
        print("5. Running integrated model analysis...")
        integrated_results = self._run_integrated_analysis(results_dir, file_prefix, results)
        results['integrated'] = integrated_results
        
        # 6. Generate Comprehensive Report
        print("6. Generating comprehensive report...")
        self._generate_comprehensive_report(results_dir, file_prefix, results)
        
        print(f"=== Comprehensive Analysis Complete ===")
        print(f"Results saved to: {results_dir}")
        
        return results
    
    def _run_traditional_analysis(self,
                                results_dir: str,
                                file_prefix: str,
                                analysis_window: Tuple[int, int],
                                garch_type: str,
                                optimistic_bias: float,
                                risk_free_rate: float) -> Dict:
        """Run traditional analysis using existing methods"""
        
        traditional_results = {}
        
        try:
            # Three-phase volatility analysis
            vol_results = self.analyze_three_phase_volatility(
                results_dir=results_dir,
                file_prefix=f"{file_prefix}_traditional",
                analysis_window=analysis_window,
                garch_type=garch_type
            )
            traditional_results['volatility'] = vol_results
            
            # RVR analysis with optimistic bias
            rvr_results = self.analyze_rvr_with_optimistic_bias(
                results_dir=results_dir,
                file_prefix=f"{file_prefix}_traditional",
                analysis_window=analysis_window,
                garch_type=garch_type,
                optimistic_bias=optimistic_bias,
                risk_free_rate=risk_free_rate
            )
            traditional_results['rvr'] = rvr_results
            
            # Sharpe ratio analysis
            sharpe_results = self.calculate_rolling_sharpe_timeseries(
                results_dir=results_dir,
                file_prefix=f"{file_prefix}_traditional",
                analysis_window=analysis_window
            )
            traditional_results['sharpe'] = sharpe_results
            
        except Exception as e:
            warnings.warn(f"Traditional analysis failed: {e}")
            traditional_results = {'error': str(e)}
        
        return traditional_results
    
    def _run_two_risk_analysis(self, results_dir: str, file_prefix: str) -> Dict:
        """Run two-risk framework analysis"""
        
        if not self.enable_two_risk or self.two_risk_framework is None:
            return {'error': 'Two-risk framework not available'}
        
        try:
            # Fit the two-risk framework
            self.two_risk_framework.fit(self.data)
            
            # Extract risk components
            risk_components = self.two_risk_framework.extract_risks(self.data)
            
            # Analyze risk decomposition by phases
            phase_analysis = self.two_risk_framework.get_phase_analysis(self.data)
            
            # Create visualizations
            self.two_risk_framework.plot_risk_decomposition(
                self.data, results_dir, f"{file_prefix}_two_risk"
            )
            
            # Save risk decomposition data
            risk_df = pl.DataFrame({
                'days_to_event': risk_components['days_to_event'],
                'directional_news_risk': risk_components['directional_news_risk'],
                'impact_uncertainty': risk_components['impact_uncertainty'],
                'total_risk': risk_components['total_risk']
            })
            risk_df.write_csv(os.path.join(results_dir, f"{file_prefix}_risk_decomposition.csv"))
            
            # Save phase analysis
            phase_df = pl.DataFrame([
                {'phase': phase, **stats} for phase, stats in phase_analysis.items()
            ])
            phase_df.write_csv(os.path.join(results_dir, f"{file_prefix}_risk_phase_analysis.csv"))
            
            return {
                'risk_components': risk_components,
                'phase_analysis': phase_analysis,
                'decomposition_quality': self.two_risk_framework.decomposition_quality
            }
            
        except Exception as e:
            warnings.warn(f"Two-risk analysis failed: {e}")
            return {'error': str(e)}
    
    def _run_investor_analysis(self, 
                             results_dir: str, 
                             file_prefix: str,
                             analysis_window: Tuple[int, int]) -> Dict:
        """Run heterogeneous investor market simulation"""
        
        if not self.enable_heterogeneous_investors or self.investor_market is None:
            return {'error': 'Heterogeneous investor framework not available'}
        
        try:
            # Prepare volatility data
            event_ids = self.data.get_column('event_id').unique().to_list()
            
            # Sample a subset for computational efficiency
            sample_size = min(100, len(event_ids))
            np.random.seed(42)
            sample_event_ids = np.random.choice(event_ids, size=sample_size, replace=False)
            
            sample_data = self.data.filter(pl.col('event_id').is_in(sample_event_ids))
            
            # Estimate volatility using GARCH
            volatility_estimates = self._estimate_volatility_for_simulation(sample_data)
            
            # Create information signals (simplified)
            information_signals = self._create_information_signals(sample_data)
            
            # Run market simulation
            simulation_results = self.investor_market.simulate_market(
                data=sample_data,
                volatility=volatility_estimates,
                information_signals=information_signals
            )
            
            # Analyze market dynamics
            market_analysis = self.investor_market.analyze_market_dynamics(simulation_results)
            
            # Create visualizations
            self.investor_market.plot_market_simulation(
                simulation_results, sample_data, results_dir, f"{file_prefix}_investor_market"
            )
            
            # Save results
            market_df = pl.DataFrame({
                'days_to_event': sample_data.get_column('days_to_event').to_numpy()[:len(simulation_results['market_clearing_prices'])],
                'market_prices': simulation_results['market_clearing_prices'],
                'aggregate_demand': simulation_results['aggregate_demand'],
                'excess_demand': simulation_results['excess_demand']
            })
            market_df.write_csv(os.path.join(results_dir, f"{file_prefix}_market_simulation.csv"))
            
            return {
                'simulation_results': simulation_results,
                'market_analysis': market_analysis,
                'sample_data': sample_data
            }
            
        except Exception as e:
            warnings.warn(f"Investor analysis failed: {e}")
            return {'error': str(e)}
    
    def _run_portfolio_optimization(self,
                                  results_dir: str,
                                  file_prefix: str,
                                  analysis_window: Tuple[int, int],
                                  optimistic_bias: float,
                                  risk_free_rate: float) -> Dict:
        """Run enhanced portfolio optimization"""
        
        if not self.enable_portfolio_optimization or self.portfolio_optimizer is None:
            return {'error': 'Portfolio optimization framework not available'}
        
        try:
            # Prepare data for optimization
            filtered_data = self.data.filter(
                (pl.col('days_to_event') >= analysis_window[0]) &
                (pl.col('days_to_event') <= analysis_window[1])
            ).sort(['event_id', 'days_to_event'])
            
            # Sample for computational efficiency
            event_ids = filtered_data.get_column('event_id').unique().to_list()
            sample_size = min(50, len(event_ids))
            np.random.seed(42)
            sample_event_ids = np.random.choice(event_ids, size=sample_size, replace=False)
            
            sample_data = filtered_data.filter(pl.col('event_id').is_in(sample_event_ids))
            
            # Create expected returns with optimistic bias
            expected_returns = self._create_biased_expected_returns(sample_data, optimistic_bias)
            
            # Estimate volatility
            volatility = self._estimate_volatility_for_optimization(sample_data)
            
            # Set optimization parameters
            opt_params = OptimizationParameters(
                gamma_T=2.0,
                gamma_V=0.5,
                risk_free_rate=risk_free_rate,
                correlation_eg=0.3,
                sigma_g=0.02,
                mu_g=0.0001
            )
            
            # Run optimization
            optimization_results = self.portfolio_optimizer.run_optimization(
                data=sample_data,
                expected_returns=expected_returns,
                volatility=volatility,
                optimization_params=opt_params
            )
            
            # Analyze results
            analysis = self.portfolio_optimizer.analyze_optimization_results()
            
            # Create visualizations
            self.portfolio_optimizer.plot_optimization_results(
                results_dir, f"{file_prefix}_portfolio_opt"
            )
            
            # Save results
            weights_df = pl.DataFrame({
                'days_to_event': sample_data.get_column('days_to_event').to_numpy()[:len(optimization_results['optimal_weights'])],
                'event_asset_weight': optimization_results['optimal_weights'][:, 0],
                'generic_asset_weight': optimization_results['optimal_weights'][:, 1],
                'risk_free_weight': optimization_results['optimal_weights'][:, 2]
            })
            weights_df.write_csv(os.path.join(results_dir, f"{file_prefix}_optimal_weights.csv"))
            
            return {
                'optimization_results': optimization_results,
                'analysis': analysis,
                'sample_data': sample_data
            }
            
        except Exception as e:
            warnings.warn(f"Portfolio optimization failed: {e}")
            return {'error': str(e)}
    
    def _run_integrated_analysis(self, 
                               results_dir: str, 
                               file_prefix: str,
                               all_results: Dict) -> Dict:
        """Run integrated analysis combining all components"""
        
        integrated_results = {}
        
        try:
            # Compare traditional vs enhanced RVR
            if ('traditional' in all_results and 'two_risk' in all_results and 
                all_results['traditional'].get('rvr') is not None and 
                all_results['two_risk'].get('risk_components') is not None):
                
                rvr_comparison = self._compare_rvr_approaches(all_results)
                integrated_results['rvr_comparison'] = rvr_comparison
            
            # Analyze investor behavior impact on market dynamics
            if ('heterogeneous_investors' in all_results and 
                all_results['heterogeneous_investors'].get('simulation_results') is not None):
                
                behavior_impact = self._analyze_investor_behavior_impact(all_results)
                integrated_results['behavior_impact'] = behavior_impact
            
            # Portfolio optimization validation
            if ('portfolio_optimization' in all_results and
                all_results['portfolio_optimization'].get('optimization_results') is not None):
                
                optimization_validation = self._validate_optimization_results(all_results)
                integrated_results['optimization_validation'] = optimization_validation
            
            # Model validation summary
            validation_summary = self._create_model_validation_summary(all_results)
            integrated_results['validation_summary'] = validation_summary
            
            # Save integrated results
            if integrated_results:
                integrated_df = pl.DataFrame([
                    {'metric': k, 'value': str(v)} for k, v in validation_summary.items()
                ])
                integrated_df.write_csv(os.path.join(results_dir, f"{file_prefix}_integrated_analysis.csv"))
            
        except Exception as e:
            warnings.warn(f"Integrated analysis failed: {e}")
            integrated_results = {'error': str(e)}
        
        return integrated_results
    
    def _generate_comprehensive_report(self, 
                                     results_dir: str, 
                                     file_prefix: str,
                                     all_results: Dict):
        """Generate comprehensive analysis report"""
        
        try:
            report_lines = []
            report_lines.append("# Enhanced Event Analysis Report")
            report_lines.append(f"Generated for: {file_prefix}")
            report_lines.append("")
            
            # Executive Summary
            report_lines.append("## Executive Summary")
            if 'traditional' in all_results and all_results['traditional'].get('rvr'):
                report_lines.append("- Traditional three-phase volatility model implemented")
            if 'two_risk' in all_results and not all_results['two_risk'].get('error'):
                report_lines.append("- Two-risk framework successfully decomposed directional news risk and impact uncertainty")
            if 'heterogeneous_investors' in all_results and not all_results['heterogeneous_investors'].get('error'):
                report_lines.append("- Heterogeneous investor market simulation completed")
            if 'portfolio_optimization' in all_results and not all_results['portfolio_optimization'].get('error'):
                report_lines.append("- Enhanced portfolio optimization with real-time variance penalties executed")
            report_lines.append("")
            
            # Detailed Results
            for section_name, section_results in all_results.items():
                if section_results and not section_results.get('error'):
                    report_lines.append(f"## {section_name.replace('_', ' ').title()}")
                    
                    if section_name == 'two_risk' and 'decomposition_quality' in section_results:
                        quality = section_results['decomposition_quality']
                        if quality:
                            report_lines.append(f"- Risk correlation: {quality.get('risk_correlation', 'N/A'):.3f}")
                            report_lines.append(f"- Explained variance: {quality.get('explained_variance', 'N/A'):.3f}")
                    
                    if section_name == 'heterogeneous_investors' and 'market_analysis' in section_results:
                        market = section_results['market_analysis'].get('market', {})
                        if market:
                            report_lines.append(f"- Price volatility: {market.get('price_volatility', 'N/A'):.4f}")
                            report_lines.append(f"- Average excess demand: {market.get('avg_excess_demand', 'N/A'):.4f}")
                    
                    report_lines.append("")
            
            # Save report
            report_filename = os.path.join(results_dir, f"{file_prefix}_comprehensive_report.md")
            with open(report_filename, 'w') as f:
                f.write('\n'.join(report_lines))
            
            print(f"Comprehensive report saved to: {report_filename}")
            
        except Exception as e:
            warnings.warn(f"Could not generate comprehensive report: {e}")
    
    # Helper methods for analysis components
    
    def _estimate_volatility_for_simulation(self, data: pl.DataFrame) -> np.ndarray:
        """Estimate volatility for market simulation"""
        returns = data.get_column('ret').to_numpy()
        returns_clean = returns[~np.isnan(returns)]
        
        if len(returns_clean) > 20:
            # Simple rolling volatility
            window = min(20, len(returns_clean) // 2)
            volatility = np.array([
                np.std(returns_clean[max(0, i-window):i+1]) 
                for i in range(len(returns_clean))
            ])
            # Extend to match original length
            volatility_full = np.full(len(returns), np.mean(volatility))
            valid_mask = ~np.isnan(returns)
            volatility_full[valid_mask] = volatility
            return volatility_full
        else:
            return np.full(len(returns), 0.02)  # Default 2% volatility
    
    def _create_information_signals(self, data: pl.DataFrame) -> Dict:
        """Create information signals for investor simulation"""
        returns = data.get_column('ret').to_numpy()
        days_to_event = data.get_column('days_to_event').to_numpy()
        
        # Simple information signal based on lagged returns and event timing
        info_signal = np.zeros_like(returns)
        
        for i in range(1, len(returns)):
            # Information builds up approaching the event
            time_factor = max(0, 1 - abs(days_to_event[i]) / 15)
            momentum_factor = returns[i-1] if not np.isnan(returns[i-1]) else 0
            info_signal[i] = time_factor * momentum_factor * 0.5
        
        return {'information_signal': info_signal}
    
    def _estimate_volatility_for_optimization(self, data: pl.DataFrame) -> np.ndarray:
        """Estimate volatility for portfolio optimization"""
        # Use GARCH model for better volatility estimates
        try:
            returns = data.get_column('ret').to_numpy()
            returns_clean = returns[~np.isnan(returns)]
            
            if len(returns_clean) > 30:
                garch_model = GJRGARCHModel()
                garch_model.fit(returns_clean)
                conditional_vol = garch_model.conditional_volatility()
                
                # Map back to original data length
                volatility_full = np.full(len(returns), np.mean(conditional_vol))
                valid_mask = ~np.isnan(returns)
                if len(conditional_vol) == np.sum(valid_mask):
                    volatility_full[valid_mask] = conditional_vol
                
                return volatility_full
            else:
                return self._estimate_volatility_for_simulation(data)
                
        except Exception as e:
            warnings.warn(f"GARCH volatility estimation failed: {e}")
            return self._estimate_volatility_for_simulation(data)
    
    def _create_biased_expected_returns(self, data: pl.DataFrame, bias: float) -> np.ndarray:
        """Create expected returns with optimistic bias"""
        returns = data.get_column('ret').to_numpy()
        days_to_event = data.get_column('days_to_event').to_numpy()
        
        expected_returns = np.zeros_like(returns)
        
        for i in range(len(returns)):
            # Historical mean
            hist_window = 20
            start_idx = max(0, i - hist_window)
            hist_returns = returns[start_idx:i] if i > 0 else returns[:1]
            hist_mean = np.nanmean(hist_returns)
            
            # Bias factor (stronger during post-event rising phase)
            if 0 <= days_to_event[i] <= 5:
                bias_factor = np.exp(-days_to_event[i] / 3.0)
                bias_adjustment = bias * bias_factor
            else:
                bias_adjustment = 0
            
            expected_returns[i] = hist_mean + bias_adjustment
        
        return expected_returns
    
    def _compare_rvr_approaches(self, all_results: Dict) -> Dict:
        """Compare traditional vs enhanced RVR approaches"""
        comparison = {}
        
        try:
            traditional_rvr = all_results['traditional']['rvr']
            risk_components = all_results['two_risk']['risk_components']
            
            # Extract RVR values from traditional approach
            if isinstance(traditional_rvr, tuple) and len(traditional_rvr) > 0:
                trad_rvr_df = traditional_rvr[0] if hasattr(traditional_rvr[0], 'get_column') else None
                if trad_rvr_df is not None and 'mean_rvr' in trad_rvr_df.columns:
                    trad_rvr_values = trad_rvr_df.get_column('mean_rvr').to_numpy()
                    comparison['traditional_rvr_mean'] = np.nanmean(trad_rvr_values)
                    comparison['traditional_rvr_std'] = np.nanstd(trad_rvr_values)
            
            # Calculate enhanced RVR using risk components
            if 'directional_news_risk' in risk_components and 'impact_uncertainty' in risk_components:
                directional_risk = risk_components['directional_news_risk']
                impact_uncertainty = risk_components['impact_uncertainty']
                
                # Simple enhanced RVR calculation
                valid_mask = ~(np.isnan(directional_risk) | np.isnan(impact_uncertainty))
                if np.sum(valid_mask) > 0:
                    enhanced_rvr = directional_risk[valid_mask] / (impact_uncertainty[valid_mask]**2 + 1e-6)
                    comparison['enhanced_rvr_mean'] = np.nanmean(enhanced_rvr)
                    comparison['enhanced_rvr_std'] = np.nanstd(enhanced_rvr)
                    comparison['correlation'] = np.corrcoef(directional_risk[valid_mask], impact_uncertainty[valid_mask])[0,1]
            
        except Exception as e:
            comparison['error'] = str(e)
        
        return comparison
    
    def _analyze_investor_behavior_impact(self, all_results: Dict) -> Dict:
        """Analyze impact of investor behavior on market dynamics"""
        impact_analysis = {}
        
        try:
            investor_results = all_results['heterogeneous_investors']
            market_analysis = investor_results.get('market_analysis', {})
            
            # Analyze difference in investor behavior
            for investor_type in ['informed', 'uninformed', 'liquidity']:
                if investor_type in market_analysis:
                    investor_stats = market_analysis[investor_type]
                    impact_analysis[f'{investor_type}_avg_weight'] = investor_stats.get('avg_event_weight', 0)
                    impact_analysis[f'{investor_type}_weight_volatility'] = investor_stats.get('weight_volatility', 0)
            
            # Market-level impact
            market_stats = market_analysis.get('market', {})
            impact_analysis['market_price_volatility'] = market_stats.get('price_volatility', 0)
            impact_analysis['market_demand_volatility'] = market_stats.get('demand_volatility', 0)
            
        except Exception as e:
            impact_analysis['error'] = str(e)
        
        return impact_analysis
    
    def _validate_optimization_results(self, all_results: Dict) -> Dict:
        """Validate portfolio optimization results"""
        validation = {}
        
        try:
            opt_results = all_results['portfolio_optimization']
            analysis = opt_results.get('analysis', {})
            
            # Weight validation
            weight_stats = analysis.get('weight_statistics', {})
            if weight_stats:
                event_weight = weight_stats.get('event_asset', {})
                validation['avg_event_weight'] = event_weight.get('mean', 0)
                validation['event_weight_range'] = [event_weight.get('min', 0), event_weight.get('max', 0)]
            
            # Performance validation
            performance = analysis.get('performance', {})
            if performance:
                validation['sharpe_ratio'] = performance.get('sharpe_ratio', 0)
                validation['total_return'] = performance.get('total_return', 0)
                validation['max_drawdown'] = performance.get('max_drawdown', 0)
            
            # Transaction cost analysis
            tc_analysis = analysis.get('transaction_costs', {})
            if tc_analysis:
                validation['total_transaction_costs'] = tc_analysis.get('total_costs', 0)
                validation['tc_as_pct_return'] = tc_analysis.get('cost_as_pct_return', 0)
            
        except Exception as e:
            validation['error'] = str(e)
        
        return validation
    
    def _create_model_validation_summary(self, all_results: Dict) -> Dict:
        """Create overall model validation summary"""
        summary = {}
        
        # Component availability
        summary['traditional_model'] = 'available' if 'traditional' in all_results else 'unavailable'
        summary['two_risk_framework'] = 'available' if 'two_risk' in all_results and not all_results['two_risk'].get('error') else 'unavailable'
        summary['heterogeneous_investors'] = 'available' if 'heterogeneous_investors' in all_results and not all_results['heterogeneous_investors'].get('error') else 'unavailable'
        summary['portfolio_optimization'] = 'available' if 'portfolio_optimization' in all_results and not all_results['portfolio_optimization'].get('error') else 'unavailable'
        
        # Model enhancement assessment
        enhancements = []
        if summary['two_risk_framework'] == 'available':
            enhancements.append('Risk decomposition')
        if summary['heterogeneous_investors'] == 'available':
            enhancements.append('Investor heterogeneity')
        if summary['portfolio_optimization'] == 'available':
            enhancements.append('Enhanced optimization')
        
        summary['implemented_enhancements'] = enhancements
        summary['enhancement_count'] = len(enhancements)
        
        return summary

# Convenience function for running enhanced analysis
def run_enhanced_event_analysis(event_file: str,
                               stock_files: List[str],
                               results_dir: str,
                               file_prefix: str,
                               event_date_col: str = 'Event Date',
                               ticker_col: str = 'ticker',
                               window_days: int = 30,
                               analysis_window: Tuple[int, int] = (-15, 15),
                               **kwargs) -> Dict[str, Any]:
    """
    Convenience function to run complete enhanced event analysis
    """
    
    # Ensure results directory exists
    os.makedirs(results_dir, exist_ok=True)
    
    # Initialize components
    data_loader = EventDataLoader(
        event_path=event_file,
        stock_paths=stock_files,
        window_days=window_days,
        event_date_col=event_date_col,
        ticker_col=ticker_col
    )
    
    feature_engineer = EventFeatureEngineer()
    
    # Create enhanced analyzer
    analyzer = EnhancedEventAnalysis(
        data_loader=data_loader,
        feature_engineer=feature_engineer,
        enable_two_risk=kwargs.get('enable_two_risk', True),
        enable_heterogeneous_investors=kwargs.get('enable_heterogeneous_investors', True),
        enable_portfolio_optimization=kwargs.get('enable_portfolio_optimization', True)
    )
    
    # Run comprehensive analysis
    results = analyzer.run_comprehensive_analysis(
        results_dir=results_dir,
        file_prefix=file_prefix,
        analysis_window=analysis_window,
        garch_type=kwargs.get('garch_type', 'gjr'),
        optimistic_bias=kwargs.get('optimistic_bias', 0.01),
        risk_free_rate=kwargs.get('risk_free_rate', 0.0)
    )
    
    return results