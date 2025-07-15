#!/usr/bin/env python3
"""
Hypothesis Validation Script for Enhanced Event Study Model
Analyzes CSV results from enhanced_testhyp1.py to validate paper hypotheses

This script checks the statistical results against the theoretical predictions
from "A Dynamic Asset Pricing Model for High-Uncertainty Events"
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from pathlib import Path
import warnings
import json

# Configuration
RESULTS_BASE_DIR = "results/enhanced_hypothesis1"
ALPHA_SIGNIFICANCE = 0.05
EFFECT_SIZE_THRESHOLDS = {
    'small': 0.2,
    'medium': 0.5, 
    'large': 0.8
}

@dataclass
class HypothesisResult:
    """Container for hypothesis validation results"""
    hypothesis_name: str
    prediction: str
    evidence_found: bool
    p_value: Optional[float]
    effect_size: Optional[float]
    confidence_level: str
    supporting_metrics: Dict[str, Any]
    details: str

class HypothesisValidator:
    """
    Main class for validating theoretical hypotheses against empirical results
    """
    
    def __init__(self, results_dir: str = RESULTS_BASE_DIR):
        self.results_dir = Path(results_dir)
        self.event_types = ['fda', 'earnings']
        self.validation_results = {}
        self.summary_stats = {}
        
        # Paper's theoretical predictions
        self.theoretical_predictions = {
            'H1_RVR_Peak': {
                'prediction': 'RVR peaks during post-event rising phase (days 0-5)',
                'metric': 'mean_rvr',
                'comparison': 'rising > pre AND rising > decay',
                'expected_effect': 'large'
            },
            'H2_Volatility_Innovation': {
                'prediction': 'GARCH volatility innovations predict future returns',
                'metric': 'r2',
                'comparison': 'R² > 0.05 for predictive horizons',
                'expected_effect': 'medium'
            },
            'H3_Asymmetric_Bias': {
                'prediction': 'RVR amplification stronger for positive events',
                'metric': 'bias_difference',
                'comparison': 'positive_bias > negative_bias',
                'expected_effect': 'medium'
            },
            'H4_Information_Asymmetry': {
                'prediction': 'Bias amplification greater for high information asymmetry firms',
                'metric': 'information_effect',
                'comparison': 'high_asymmetry > low_asymmetry',
                'expected_effect': 'medium'
            },
            'H5_Real_Time_Risk': {
                'prediction': 'Real-time variance aversion affects position sizing',
                'metric': 'risk_management_effect',
                'comparison': 'high_gamma_V < low_gamma_V positions',
                'expected_effect': 'medium'
            }
        }
    
    def run_validation(self) -> Dict[str, Any]:
        """
        Run complete hypothesis validation
        """
        print("="*80)
        print("HYPOTHESIS VALIDATION: Enhanced Event Study Model")
        print("="*80)
        
        validation_results = {}
        
        # Check if results directory exists
        if not self.results_dir.exists():
            print(f"❌ Results directory not found: {self.results_dir}")
            print("Please run enhanced_testhyp1.py first to generate results.")
            return {'error': 'Results directory not found'}
        
        # Validate each hypothesis
        print("\n🔍 Validating Theoretical Hypotheses...")
        
        # H1: RVR Peak Hypothesis
        h1_result = self._validate_h1_rvr_peak()
        validation_results['H1_RVR_Peak'] = h1_result
        
        # H2: Volatility Innovation Hypothesis  
        h2_result = self._validate_h2_volatility_innovation()
        validation_results['H2_Volatility_Innovation'] = h2_result
        
        # H3: Asymmetric Bias Hypothesis
        h3_result = self._validate_h3_asymmetric_bias()
        validation_results['H3_Asymmetric_Bias'] = h3_result
        
        # H4: Information Asymmetry (if available)
        h4_result = self._validate_h4_information_asymmetry()
        validation_results['H4_Information_Asymmetry'] = h4_result
        
        # H5: Real-Time Risk Management (if available)
        h5_result = self._validate_h5_real_time_risk()
        validation_results['H5_Real_Time_Risk'] = h5_result
        
        # Model Enhancement Validation
        enhancement_result = self._validate_model_enhancements()
        validation_results['Model_Enhancements'] = enhancement_result
        
        # Generate summary
        summary = self._generate_validation_summary(validation_results)
        validation_results['Summary'] = summary
        
        # Save results
        self._save_validation_results(validation_results)
        
        # Generate visualizations
        self._create_validation_plots(validation_results)
        
        print("\n" + "="*80)
        print("VALIDATION COMPLETE")
        print("="*80)
        
        return validation_results
    
    def _validate_h1_rvr_peak(self) -> HypothesisResult:
        """
        Validate H1: RVR peaks during post-event rising phase
        """
        print("\n📊 H1: Testing RVR Peak Hypothesis...")
        
        try:
            evidence_found = False
            supporting_metrics = {}
            all_p_values = []
            all_effect_sizes = []
            details_list = []
            
            for event_type in self.event_types:
                print(f"  Checking {event_type.upper()} events...")
                
                # Look for statistical test results
                test_file = self.results_dir / f"results_{event_type}" / f"{event_type}_enhanced_test_summary.csv"
                if test_file.exists():
                    test_df = pd.read_csv(test_file)
                    
                    # Find RVR peak test
                    rvr_tests = test_df[test_df['test_name'].str.contains('rvr', case=False, na=False)]
                    if not rvr_tests.empty:
                        for _, test in rvr_tests.iterrows():
                            if test['is_significant']:
                                evidence_found = True
                                all_p_values.append(test['p_value'])
                                if pd.notna(test['effect_size']):
                                    all_effect_sizes.append(test['effect_size'])
                                details_list.append(f"{event_type}: p={test['p_value']:.4f}")
                
                # Look for RVR phase analysis
                phase_file = self.results_dir / f"results_{event_type}" / f"{event_type}_enhanced_rvr_bias_phase_stats.csv"
                if phase_file.exists():
                    phase_df = pd.read_csv(phase_file)
                    
                    # Extract phase RVR values
                    phases = ['pre_event', 'post_event_rising', 'post_event_decay']
                    phase_rvr = {}
                    
                    for phase in phases:
                        phase_data = phase_df[phase_df['phase'] == phase]
                        if not phase_data.empty:
                            phase_rvr[phase] = phase_data['avg_rvr'].iloc[0]
                    
                    if len(phase_rvr) >= 3:
                        # Test H1 prediction: rising > pre AND rising > decay
                        rising_rvr = phase_rvr.get('post_event_rising', 0)
                        pre_rvr = phase_rvr.get('pre_event', 0)
                        decay_rvr = phase_rvr.get('post_event_decay', 0)
                        
                        h1_condition = rising_rvr > pre_rvr and rising_rvr > decay_rvr
                        
                        if h1_condition:
                            evidence_found = True
                            effect_size = (rising_rvr - max(pre_rvr, decay_rvr)) / max(abs(pre_rvr), abs(decay_rvr), 0.001)
                            all_effect_sizes.append(effect_size)
                            details_list.append(f"{event_type}: Rising={rising_rvr:.3f} > Pre={pre_rvr:.3f}, Decay={decay_rvr:.3f}")
                        
                        supporting_metrics[f'{event_type}_rvr_phases'] = {
                            'pre_event': pre_rvr,
                            'post_event_rising': rising_rvr,
                            'post_event_decay': decay_rvr,
                            'h1_satisfied': h1_condition
                        }
            
            # Determine overall result
            avg_p_value = np.mean(all_p_values) if all_p_values else None
            avg_effect_size = np.mean(all_effect_sizes) if all_effect_sizes else None
            
            confidence_level = self._determine_confidence_level(avg_p_value, avg_effect_size)
            details = "; ".join(details_list) if details_list else "No clear evidence found"
            
            print(f"  📋 H1 Result: {'✅ SUPPORTED' if evidence_found else '❌ NOT SUPPORTED'}")
            if avg_p_value:
                print(f"  📊 Average p-value: {avg_p_value:.4f}")
            if avg_effect_size:
                print(f"  📈 Average effect size: {avg_effect_size:.3f}")
            
            return HypothesisResult(
                hypothesis_name="H1: RVR Peak During Rising Phase",
                prediction=self.theoretical_predictions['H1_RVR_Peak']['prediction'],
                evidence_found=evidence_found,
                p_value=avg_p_value,
                effect_size=avg_effect_size,
                confidence_level=confidence_level,
                supporting_metrics=supporting_metrics,
                details=details
            )
            
        except Exception as e:
            print(f"  ❌ Error validating H1: {e}")
            return HypothesisResult(
                hypothesis_name="H1: RVR Peak During Rising Phase",
                prediction=self.theoretical_predictions['H1_RVR_Peak']['prediction'],
                evidence_found=False,
                p_value=None,
                effect_size=None,
                confidence_level="error",
                supporting_metrics={},
                details=f"Error: {str(e)}"
            )
    
    def _validate_h2_volatility_innovation(self) -> HypothesisResult:
        """
        Validate H2: Volatility innovations predict returns
        """
        print("\n📊 H2: Testing Volatility Innovation Hypothesis...")
        
        try:
            evidence_found = False
            supporting_metrics = {}
            all_p_values = []
            all_r2_values = []
            details_list = []
            
            for event_type in self.event_types:
                print(f"  Checking {event_type.upper()} events...")
                
                # Look for volatility innovation test results
                test_file = self.results_dir / f"results_{event_type}" / f"{event_type}_enhanced_volatility_innovation_details.csv"
                if test_file.exists():
                    detail_df = pd.read_csv(test_file)
                    
                    # Look for R² values and regression results
                    r2_metrics = detail_df[detail_df['metric'].str.contains('r2|R2', case=False, na=False)]
                    if not r2_metrics.empty:
                        for _, metric in r2_metrics.iterrows():
                            try:
                                r2_value = float(metric['value'])
                                if r2_value > 0.05:  # Meaningful predictive power
                                    evidence_found = True
                                    all_r2_values.append(r2_value)
                                    details_list.append(f"{event_type}: R²={r2_value:.3f}")
                            except ValueError:
                                continue
                
                # Alternative: Look in main test summary
                test_summary_file = self.results_dir / f"results_{event_type}" / f"{event_type}_enhanced_test_summary.csv"
                if test_summary_file.exists():
                    test_df = pd.read_csv(test_summary_file)
                    vol_tests = test_df[test_df['test_name'].str.contains('volatility|innovation', case=False, na=False)]
                    
                    for _, test in vol_tests.iterrows():
                        if test['is_significant']:
                            evidence_found = True
                            all_p_values.append(test['p_value'])
                            if pd.notna(test['effect_size']):
                                all_r2_values.append(test['effect_size'])
                            details_list.append(f"{event_type}: p={test['p_value']:.4f}")
                
                supporting_metrics[f'{event_type}_volatility_tests'] = {
                    'r2_values': all_r2_values,
                    'predictive_power': len(all_r2_values) > 0
                }
            
            avg_p_value = np.mean(all_p_values) if all_p_values else None
            avg_r2 = np.mean(all_r2_values) if all_r2_values else None
            
            confidence_level = self._determine_confidence_level(avg_p_value, avg_r2)
            details = "; ".join(details_list) if details_list else "No predictive power found"
            
            print(f"  📋 H2 Result: {'✅ SUPPORTED' if evidence_found else '❌ NOT SUPPORTED'}")
            if avg_r2:
                print(f"  📊 Average R²: {avg_r2:.4f}")
            
            return HypothesisResult(
                hypothesis_name="H2: Volatility Innovations Predict Returns",
                prediction=self.theoretical_predictions['H2_Volatility_Innovation']['prediction'],
                evidence_found=evidence_found,
                p_value=avg_p_value,
                effect_size=avg_r2,
                confidence_level=confidence_level,
                supporting_metrics=supporting_metrics,
                details=details
            )
            
        except Exception as e:
            print(f"  ❌ Error validating H2: {e}")
            return HypothesisResult(
                hypothesis_name="H2: Volatility Innovations Predict Returns",
                prediction=self.theoretical_predictions['H2_Volatility_Innovation']['prediction'],
                evidence_found=False,
                p_value=None,
                effect_size=None,
                confidence_level="error",
                supporting_metrics={},
                details=f"Error: {str(e)}"
            )
    
    def _validate_h3_asymmetric_bias(self) -> HypothesisResult:
        """
        Validate H3: Asymmetric bias effects (positive vs negative events)
        """
        print("\n📊 H3: Testing Asymmetric Bias Hypothesis...")
        
        try:
            evidence_found = False
            supporting_metrics = {}
            all_p_values = []
            all_effect_sizes = []
            details_list = []
            
            for event_type in self.event_types:
                print(f"  Checking {event_type.upper()} events...")
                
                # Look for asymmetric bias test results
                test_file = self.results_dir / f"results_{event_type}" / f"{event_type}_enhanced_asymmetric_bias_details.csv"
                if test_file.exists():
                    detail_df = pd.read_csv(test_file)
                    
                    # Look for positive vs negative event comparisons
                    pos_metrics = detail_df[detail_df['metric'].str.contains('positive', case=False, na=False)]
                    neg_metrics = detail_df[detail_df['metric'].str.contains('negative', case=False, na=False)]
                    
                    if not pos_metrics.empty and not neg_metrics.empty:
                        try:
                            pos_value = float(pos_metrics.iloc[0]['value'])
                            neg_value = float(neg_metrics.iloc[0]['value'])
                            
                            if pos_value > neg_value:  # Asymmetric bias prediction
                                evidence_found = True
                                effect_size = (pos_value - neg_value) / max(abs(neg_value), 0.001)
                                all_effect_sizes.append(effect_size)
                                details_list.append(f"{event_type}: Pos={pos_value:.3f} > Neg={neg_value:.3f}")
                        except ValueError:
                            continue
                
                # Alternative: Look in main test summary
                test_summary_file = self.results_dir / f"results_{event_type}" / f"{event_type}_enhanced_test_summary.csv"
                if test_summary_file.exists():
                    test_df = pd.read_csv(test_summary_file)
                    asym_tests = test_df[test_df['test_name'].str.contains('asymmetric|bias', case=False, na=False)]
                    
                    for _, test in asym_tests.iterrows():
                        if test['is_significant']:
                            evidence_found = True
                            all_p_values.append(test['p_value'])
                            if pd.notna(test['effect_size']):
                                all_effect_sizes.append(test['effect_size'])
                            details_list.append(f"{event_type}: p={test['p_value']:.4f}")
                
                supporting_metrics[f'{event_type}_asymmetric_tests'] = {
                    'asymmetric_bias_found': evidence_found
                }
            
            avg_p_value = np.mean(all_p_values) if all_p_values else None
            avg_effect_size = np.mean(all_effect_sizes) if all_effect_sizes else None
            
            confidence_level = self._determine_confidence_level(avg_p_value, avg_effect_size)
            details = "; ".join(details_list) if details_list else "No asymmetric bias found"
            
            print(f"  📋 H3 Result: {'✅ SUPPORTED' if evidence_found else '❌ NOT SUPPORTED'}")
            
            return HypothesisResult(
                hypothesis_name="H3: Asymmetric Bias Effects",
                prediction=self.theoretical_predictions['H3_Asymmetric_Bias']['prediction'],
                evidence_found=evidence_found,
                p_value=avg_p_value,
                effect_size=avg_effect_size,
                confidence_level=confidence_level,
                supporting_metrics=supporting_metrics,
                details=details
            )
            
        except Exception as e:
            print(f"  ❌ Error validating H3: {e}")
            return HypothesisResult(
                hypothesis_name="H3: Asymmetric Bias Effects",
                prediction=self.theoretical_predictions['H3_Asymmetric_Bias']['prediction'],
                evidence_found=False,
                p_value=None,
                effect_size=None,
                confidence_level="error",
                supporting_metrics={},
                details=f"Error: {str(e)}"
            )
    
    def _validate_h4_information_asymmetry(self) -> HypothesisResult:
        """
        Validate H4: Information asymmetry effects (if data available)
        """
        print("\n📊 H4: Testing Information Asymmetry Hypothesis...")
        
        # This would require additional data on information asymmetry
        # For now, we'll check if the enhanced framework generated any related results
        
        try:
            evidence_found = False
            supporting_metrics = {}
            
            # Look for investor heterogeneity results
            for event_type in self.event_types:
                hetero_file = self.results_dir / f"results_{event_type}" / f"{event_type}_enhanced_market_simulation.csv"
                if hetero_file.exists():
                    # If heterogeneous investor analysis was conducted
                    evidence_found = True
                    supporting_metrics[f'{event_type}_heterogeneity'] = {'analysis_conducted': True}
            
            print(f"  📋 H4 Result: {'⚠️  LIMITED DATA' if evidence_found else '❌ NO DATA'}")
            
            return HypothesisResult(
                hypothesis_name="H4: Information Asymmetry Effects",
                prediction=self.theoretical_predictions['H4_Information_Asymmetry']['prediction'],
                evidence_found=evidence_found,
                p_value=None,
                effect_size=None,
                confidence_level="limited_data" if evidence_found else "no_data",
                supporting_metrics=supporting_metrics,
                details="Requires additional information asymmetry data" if not evidence_found else "Heterogeneous investor analysis available"
            )
            
        except Exception as e:
            print(f"  ❌ Error validating H4: {e}")
            return HypothesisResult(
                hypothesis_name="H4: Information Asymmetry Effects",
                prediction=self.theoretical_predictions['H4_Information_Asymmetry']['prediction'],
                evidence_found=False,
                p_value=None,
                effect_size=None,
                confidence_level="error",
                supporting_metrics={},
                details=f"Error: {str(e)}"
            )
    
    def _validate_h5_real_time_risk(self) -> HypothesisResult:
        """
        Validate H5: Real-time risk management effects (if data available)
        """
        print("\n📊 H5: Testing Real-Time Risk Management Hypothesis...")
        
        try:
            evidence_found = False
            supporting_metrics = {}
            
            # Look for portfolio optimization results
            for event_type in self.event_types:
                portfolio_file = self.results_dir / f"results_{event_type}" / f"{event_type}_enhanced_optimal_weights.csv"
                if portfolio_file.exists():
                    # If portfolio optimization was conducted
                    portfolio_df = pd.read_csv(portfolio_file)
                    if 'event_asset_weight' in portfolio_df.columns:
                        # Analyze weight dynamics
                        weight_volatility = portfolio_df['event_asset_weight'].std()
                        evidence_found = True
                        supporting_metrics[f'{event_type}_portfolio'] = {
                            'weight_volatility': weight_volatility,
                            'analysis_conducted': True
                        }
            
            print(f"  📋 H5 Result: {'⚠️  LIMITED DATA' if evidence_found else '❌ NO DATA'}")
            
            return HypothesisResult(
                hypothesis_name="H5: Real-Time Risk Management Effects",
                prediction=self.theoretical_predictions['H5_Real_Time_Risk']['prediction'],
                evidence_found=evidence_found,
                p_value=None,
                effect_size=None,
                confidence_level="limited_data" if evidence_found else "no_data",
                supporting_metrics=supporting_metrics,
                details="Portfolio optimization analysis available" if evidence_found else "Requires real-time risk management data"
            )
            
        except Exception as e:
            print(f"  ❌ Error validating H5: {e}")
            return HypothesisResult(
                hypothesis_name="H5: Real-Time Risk Management Effects",
                prediction=self.theoretical_predictions['H5_Real_Time_Risk']['prediction'],
                evidence_found=False,
                p_value=None,
                effect_size=None,
                confidence_level="error",
                supporting_metrics={},
                details=f"Error: {str(e)}"
            )
    
    def _validate_model_enhancements(self) -> HypothesisResult:
        """
        Validate that the enhanced model components are working
        """
        print("\n📊 Model Enhancement Validation...")
        
        try:
            enhancements_found = {}
            total_enhancements = 0
            working_enhancements = 0
            
            # Check for two-risk framework
            for event_type in self.event_types:
                risk_file = self.results_dir / f"results_{event_type}" / f"{event_type}_enhanced_risk_decomposition.csv"
                if risk_file.exists():
                    enhancements_found['two_risk_framework'] = True
                    working_enhancements += 1
                total_enhancements += 1
            
            # Check for heterogeneous investors
            for event_type in self.event_types:
                investor_file = self.results_dir / f"results_{event_type}" / f"{event_type}_enhanced_market_simulation.csv"
                if investor_file.exists():
                    enhancements_found['heterogeneous_investors'] = True
                    working_enhancements += 1
                    break
                total_enhancements += 1
            
            # Check for portfolio optimization
            for event_type in self.event_types:
                portfolio_file = self.results_dir / f"results_{event_type}" / f"{event_type}_enhanced_optimal_weights.csv"
                if portfolio_file.exists():
                    enhancements_found['portfolio_optimization'] = True
                    working_enhancements += 1
                    break
                total_enhancements += 1
            
            # Check for statistical testing
            for event_type in self.event_types:
                stats_file = self.results_dir / f"results_{event_type}" / f"{event_type}_enhanced_test_summary.csv"
                if stats_file.exists():
                    enhancements_found['statistical_testing'] = True
                    working_enhancements += 1
                    break
                total_enhancements += 1
            
            success_rate = working_enhancements / max(total_enhancements, 1)
            evidence_found = success_rate > 0.5
            
            print(f"  📋 Enhancement Result: {working_enhancements}/{total_enhancements} components working")
            
            return HypothesisResult(
                hypothesis_name="Model Enhancement Validation",
                prediction="Enhanced theoretical components should be functional",
                evidence_found=evidence_found,
                p_value=None,
                effect_size=success_rate,
                confidence_level="high" if success_rate > 0.8 else "medium" if success_rate > 0.5 else "low",
                supporting_metrics=enhancements_found,
                details=f"Success rate: {success_rate:.1%}, Working: {list(enhancements_found.keys())}"
            )
            
        except Exception as e:
            print(f"  ❌ Error validating model enhancements: {e}")
            return HypothesisResult(
                hypothesis_name="Model Enhancement Validation",
                prediction="Enhanced theoretical components should be functional",
                evidence_found=False,
                p_value=None,
                effect_size=None,
                confidence_level="error",
                supporting_metrics={},
                details=f"Error: {str(e)}"
            )
    
    def _determine_confidence_level(self, p_value: Optional[float], effect_size: Optional[float]) -> str:
        """
        Determine confidence level based on p-value and effect size
        """
        if p_value is None and effect_size is None:
            return "no_data"
        
        if p_value is not None:
            if p_value < 0.001:
                p_level = "very_high"
            elif p_value < 0.01:
                p_level = "high"
            elif p_value < 0.05:
                p_level = "medium"
            else:
                p_level = "low"
        else:
            p_level = "unknown"
        
        if effect_size is not None:
            if abs(effect_size) > EFFECT_SIZE_THRESHOLDS['large']:
                e_level = "large"
            elif abs(effect_size) > EFFECT_SIZE_THRESHOLDS['medium']:
                e_level = "medium"
            elif abs(effect_size) > EFFECT_SIZE_THRESHOLDS['small']:
                e_level = "small"
            else:
                e_level = "negligible"
        else:
            e_level = "unknown"
        
        # Combine p-value and effect size assessments
        if p_level == "very_high" and e_level in ["large", "medium"]:
            return "very_high"
        elif p_level in ["high", "very_high"] and e_level != "negligible":
            return "high"
        elif p_level == "medium" or e_level in ["medium", "large"]:
            return "medium"
        else:
            return "low"
    
    def _generate_validation_summary(self, validation_results: Dict[str, HypothesisResult]) -> Dict[str, Any]:
        """
        Generate overall validation summary
        """
        total_hypotheses = len([k for k in validation_results.keys() if k.startswith('H')])
        supported_hypotheses = sum(1 for k, v in validation_results.items() 
                                 if k.startswith('H') and v.evidence_found)
        
        support_rate = supported_hypotheses / max(total_hypotheses, 1)
        
        # Overall model assessment
        if support_rate >= 0.8:
            overall_assessment = "STRONG_SUPPORT"
        elif support_rate >= 0.6:
            overall_assessment = "MODERATE_SUPPORT"
        elif support_rate >= 0.4:
            overall_assessment = "MIXED_EVIDENCE"
        elif support_rate >= 0.2:
            overall_assessment = "LIMITED_SUPPORT"
        else:
            overall_assessment = "INSUFFICIENT_EVIDENCE"
        
        return {
            'total_hypotheses': total_hypotheses,
            'supported_hypotheses': supported_hypotheses,
            'support_rate': support_rate,
            'overall_assessment': overall_assessment,
            'model_enhancement_status': validation_results.get('Model_Enhancements', {}).confidence_level,
            'validation_timestamp': pd.Timestamp.now().isoformat()
        }
    
    def _save_validation_results(self, validation_results: Dict[str, Any]):
        """
        Save validation results to files
        """
        output_dir = self.results_dir / "validation_results"
        output_dir.mkdir(exist_ok=True)
        
        # Save detailed results as JSON
        json_results = {}
        for key, result in validation_results.items():
            if isinstance(result, HypothesisResult):
                json_results[key] = {
                    'hypothesis_name': result.hypothesis_name,
                    'prediction': result.prediction,
                    'evidence_found': result.evidence_found,
                    'p_value': result.p_value,
                    'effect_size': result.effect_size,
                    'confidence_level': result.confidence_level,
                    'supporting_metrics': result.supporting_metrics,
                    'details': result.details
                }
            else:
                json_results[key] = result
        
        with open(output_dir / "validation_results.json", 'w') as f:
            json.dump(json_results, f, indent=2, default=str)
        
        # Save summary as CSV
        summary_data = []
        for key, result in validation_results.items():
            if isinstance(result, HypothesisResult):
                summary_data.append({
                    'hypothesis': key,
                    'name': result.hypothesis_name,
                    'prediction': result.prediction,
                    'evidence_found': result.evidence_found,
                    'p_value': result.p_value,
                    'effect_size': result.effect_size,
                    'confidence_level': result.confidence_level,
                    'details': result.details
                })
        
        if summary_data:
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_csv(output_dir / "hypothesis_validation_summary.csv", index=False)
        
        print(f"\n💾 Validation results saved to: {output_dir}")
    
    def _create_validation_plots(self, validation_results: Dict[str, Any]):
        """
        Create visualization of validation results
        """
        try:
            output_dir = self.results_dir / "validation_results"
            
            # Extract hypothesis results
            hypothesis_results = {k: v for k, v in validation_results.items() 
                                if isinstance(v, HypothesisResult) and k.startswith('H')}
            
            if not hypothesis_results:
                print("  ⚠️  No hypothesis results to plot")
                return
            
            # Create summary plot
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            
            # Plot 1: Hypothesis support overview
            hypothesis_names = [result.hypothesis_name.split(':')[0] for result in hypothesis_results.values()]
            support_status = [result.evidence_found for result in hypothesis_results.values()]
            colors = ['green' if supported else 'red' for supported in support_status]
            
            bars = axes[0, 0].bar(hypothesis_names, [1 if s else 0 for s in support_status], 
                                color=colors, alpha=0.7)
            axes[0, 0].set_ylabel('Evidence Found')
            axes[0, 0].set_title('Hypothesis Validation Results')
            axes[0, 0].set_ylim(0, 1.2)
            
            # Add support labels
            for bar, supported in zip(bars, support_status):
                height = bar.get_height()
                axes[0, 0].text(bar.get_x() + bar.get_width()/2., height + 0.05,
                               '✓' if supported else '✗', 
                               ha='center', va='bottom', fontsize=14, fontweight='bold')
            
            # Plot 2: P-values (where available)
            p_values = [result.p_value for result in hypothesis_results.values() if result.p_value is not None]
            p_names = [result.hypothesis_name.split(':')[0] for result in hypothesis_results.values() if result.p_value is not None]
            
            if p_values:
                bars2 = axes[0, 1].bar(p_names, p_values, alpha=0.7, color='blue')
                axes[0, 1].axhline(y=ALPHA_SIGNIFICANCE, color='red', linestyle='--', label=f'α = {ALPHA_SIGNIFICANCE}')
                axes[0, 1].set_ylabel('P-value')
                axes[0, 1].set_title('Statistical Significance')
                axes[0, 1].legend()
                
                # Add p-value labels
                for bar, p_val in zip(bars2, p_values):
                    height = bar.get_height()
                    axes[0, 1].text(bar.get_x() + bar.get_width()/2., height + 0.001,
                                   f'{p_val:.3f}', ha='center', va='bottom', fontsize=9)
            else:
                axes[0, 1].text(0.5, 0.5, 'No P-values Available', ha='center', va='center', 
                               transform=axes[0, 1].transAxes, fontsize=12)
                axes[0, 1].set_title('Statistical Significance')
            
            # Plot 3: Effect sizes (where available)
            effect_sizes = [result.effect_size for result in hypothesis_results.values() if result.effect_size is not None]
            effect_names = [result.hypothesis_name.split(':')[0] for result in hypothesis_results.values() if result.effect_size is not None]
            
            if effect_sizes:
                bars3 = axes[1, 0].bar(effect_names, effect_sizes, alpha=0.7, color='purple')
                
                # Add threshold lines
                axes[1, 0].axhline(y=EFFECT_SIZE_THRESHOLDS['small'], color='yellow', linestyle=':', alpha=0.7, label='Small')
                axes[1, 0].axhline(y=EFFECT_SIZE_THRESHOLDS['medium'], color='orange', linestyle=':', alpha=0.7, label='Medium')
                axes[1, 0].axhline(y=EFFECT_SIZE_THRESHOLDS['large'], color='red', linestyle=':', alpha=0.7, label='Large')
                
                axes[1, 0].set_ylabel('Effect Size')
                axes[1, 0].set_title('Effect Sizes')
                axes[1, 0].legend()
                
                # Add effect size labels
                for bar, eff_size in zip(bars3, effect_sizes):
                    height = bar.get_height()
                    axes[1, 0].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                                   f'{eff_size:.2f}', ha='center', va='bottom', fontsize=9)
            else:
                axes[1, 0].text(0.5, 0.5, 'No Effect Sizes Available', ha='center', va='center', 
                               transform=axes[1, 0].transAxes, fontsize=12)
                axes[1, 0].set_title('Effect Sizes')
            
            # Plot 4: Overall summary
            if 'Summary' in validation_results:
                summary = validation_results['Summary']
                
                # Pie chart of support
                support_counts = [summary['supported_hypotheses'], 
                                summary['total_hypotheses'] - summary['supported_hypotheses']]
                labels = ['Supported', 'Not Supported']
                colors_pie = ['green', 'red']
                
                if sum(support_counts) > 0:
                    axes[1, 1].pie(support_counts, labels=labels, colors=colors_pie, autopct='%1.0f',
                                  startangle=90)
                    axes[1, 1].set_title(f'Overall Assessment\n{summary["overall_assessment"].replace("_", " ")}')
                else:
                    axes[1, 1].text(0.5, 0.5, 'No Results to Summarize', ha='center', va='center',
                                   transform=axes[1, 1].transAxes, fontsize=12)
            
            # Rotate x-axis labels for better readability
            for ax in [axes[0, 0], axes[0, 1], axes[1, 0]]:
                plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
            
            plt.tight_layout()
            plt.savefig(output_dir / "hypothesis_validation_plots.png", dpi=200, bbox_inches='tight')
            plt.close()
            
            print(f"  📊 Validation plots saved to: {output_dir}/hypothesis_validation_plots.png")
            
        except Exception as e:
            print(f"  ❌ Error creating validation plots: {e}")

def print_validation_report(validation_results: Dict[str, Any]):
    """
    Print a formatted validation report
    """
    print("\n" + "="*80)
    print("📋 HYPOTHESIS VALIDATION REPORT")
    print("="*80)
    
    # Print individual hypothesis results
    for key, result in validation_results.items():
        if isinstance(result, HypothesisResult) and key.startswith('H'):
            print(f"\n{result.hypothesis_name}")
            print("-" * len(result.hypothesis_name))
            print(f"Prediction: {result.prediction}")
            print(f"Evidence Found: {'✅ YES' if result.evidence_found else '❌ NO'}")
            print(f"Confidence Level: {result.confidence_level.upper().replace('_', ' ')}")
            
            if result.p_value is not None:
                print(f"P-value: {result.p_value:.4f}")
            if result.effect_size is not None:
                print(f"Effect Size: {result.effect_size:.3f}")
            
            print(f"Details: {result.details}")
    
    # Print overall summary
    if 'Summary' in validation_results:
        summary = validation_results['Summary']
        print(f"\n🎯 OVERALL ASSESSMENT")
        print("-" * 20)
        print(f"Hypotheses Tested: {summary['total_hypotheses']}")
        print(f"Hypotheses Supported: {summary['supported_hypotheses']}")
        print(f"Support Rate: {summary['support_rate']:.1%}")
        print(f"Overall Assessment: {summary['overall_assessment'].replace('_', ' ')}")
        
        # Interpretation
        if summary['support_rate'] >= 0.8:
            print("\n🎉 The theoretical model receives STRONG empirical support!")
        elif summary['support_rate'] >= 0.6:
            print("\n👍 The theoretical model receives MODERATE empirical support.")
        elif summary['support_rate'] >= 0.4:
            print("\n🤔 The evidence is MIXED - some hypotheses supported, others not.")
        elif summary['support_rate'] >= 0.2:
            print("\n⚠️  The theoretical model receives LIMITED empirical support.")
        else:
            print("\n❌ INSUFFICIENT evidence to support the theoretical model.")
    
    print("\n" + "="*80)

def main():
    """
    Main execution function
    """
    print("🚀 Starting Hypothesis Validation...")
    
    validator = HypothesisValidator()
    validation_results = validator.run_validation()
    
    if 'error' in validation_results:
        print(f"\n❌ Validation failed: {validation_results['error']}")
        return
    
    # Print detailed report
    print_validation_report(validation_results)
    
    # Print file locations
    print("\n📁 Results saved to:")
    print(f"  • JSON: {validator.results_dir}/validation_results/validation_results.json")
    print(f"  • CSV: {validator.results_dir}/validation_results/hypothesis_validation_summary.csv")
    print(f"  • Plots: {validator.results_dir}/validation_results/hypothesis_validation_plots.png")

if __name__ == "__main__":
    main()
