#!/usr/bin/env python3
"""
Test Enhanced Framework with Compatibility Fixes
Quick test to verify the enhanced framework works with the applied fixes
"""

import os
import sys

# Add current directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

def test_imports():
    """Test that all enhanced components can be imported"""
    print("Testing enhanced framework imports...")
    
    try:
        # Test compatibility fixes
        from src.polars_compatibility_fixes import safe_clip_quantile, safe_apply, safe_interpolate
        print("✓ Polars compatibility fixes imported successfully")
        
        # Test two-risk framework
        from src.two_risk_framework import TwoRiskFramework, DirectionalNewsRisk, ImpactUncertainty
        print("✓ Two-risk framework imported successfully")
        
        # Test investor heterogeneity
        from src.investor_heterogeneity import HeterogeneousInvestorMarket
        print("✓ Heterogeneous investor framework imported successfully")
        
        # Test portfolio optimization
        from src.portfolio_optimization import PortfolioOptimizationFramework
        print("✓ Portfolio optimization framework imported successfully")
        
        # Test statistical testing
        from src.statistical_testing import ComprehensiveTestSuite
        print("✓ Statistical testing framework imported successfully")
        
        # Test enhanced event processor
        from src.enhanced_event_processor import EnhancedEventAnalysis
        print("✓ Enhanced event processor imported successfully")
        
        return True
        
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False

def test_basic_functionality():
    """Test basic functionality of key components"""
    print("\nTesting basic functionality...")
    
    try:
        import polars as pl
        import numpy as np
        
        # Create sample data
        np.random.seed(42)
        n_days = 31
        days_to_event = list(range(-15, 16))
        
        sample_data = pl.DataFrame({
            'days_to_event': days_to_event,
            'ret': np.random.normal(0, 0.02, n_days),
            'event_id': ['test_event'] * n_days,
            'Event Date': [pl.datetime(2024, 1, 15)] * n_days
        })
        
        print("✓ Sample data created successfully")
        
        # Test two-risk framework
        from src.two_risk_framework import TwoRiskFramework
        
        risk_framework = TwoRiskFramework()
        risk_framework.fit(sample_data)
        risk_components = risk_framework.extract_risks(sample_data)
        
        print("✓ Two-risk framework basic test passed")
        print(f"  - Directional risk shape: {np.array(risk_components['directional_news_risk']).shape}")
        print(f"  - Impact uncertainty shape: {np.array(risk_components['impact_uncertainty']).shape}")
        
        # Test heterogeneous investors (simplified)
        from src.investor_heterogeneity import HeterogeneousInvestorMarket
        
        investor_market = HeterogeneousInvestorMarket()
        
        print("✓ Heterogeneous investor framework basic test passed")
        
        # Test statistical testing (basic)
        from src.statistical_testing import RVRPeakTest
        
        # Add RVR column for testing
        test_data = sample_data.with_columns([
            pl.lit(np.random.normal(0, 1, n_days)).alias('rvr')
        ])
        
        rvr_test = RVRPeakTest()
        test_result = rvr_test.run_test(test_data)
        
        print("✓ Statistical testing basic test passed")
        print(f"  - Test result: {test_result.hypothesis_name}")
        print(f"  - P-value: {test_result.p_value:.4f}")
        
        return True
        
    except Exception as e:
        print(f"✗ Functionality test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_polars_compatibility():
    """Test that Polars compatibility fixes work"""
    print("\nTesting Polars compatibility fixes...")
    
    try:
        import polars as pl
        import numpy as np
        from src.polars_compatibility_fixes import safe_clip_quantile, safe_apply, safe_interpolate
        
        # Create test data
        test_df = pl.DataFrame({
            'values': [1.0, 2.0, np.nan, 4.0, 5.0, 100.0, -50.0, 3.0],
            'group': ['A', 'A', 'A', 'A', 'B', 'B', 'B', 'B']
        })
        
        # Test safe_clip_quantile
        result1 = test_df.with_columns([
            safe_clip_quantile(pl.col('values'), 0.1, 0.9).alias('clipped')
        ])
        print("✓ safe_clip_quantile works")
        
        # Test safe_apply
        result2 = test_df.with_columns([
            safe_apply(pl.col('values'), lambda x: x * 2 if not np.isnan(x) else 0, pl.Float64).alias('doubled')
        ])
        print("✓ safe_apply works")
        
        # Test safe_interpolate
        result3 = test_df.with_columns([
            safe_interpolate(pl.col('values')).alias('interpolated')
        ])
        print("✓ safe_interpolate works")
        
        return True
        
    except Exception as e:
        print(f"✗ Polars compatibility test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("="*60)
    print("ENHANCED FRAMEWORK COMPATIBILITY TEST")
    print("="*60)
    
    success_count = 0
    total_tests = 3
    
    # Test 1: Imports
    if test_imports():
        success_count += 1
    
    # Test 2: Polars compatibility
    if test_polars_compatibility():
        success_count += 1
    
    # Test 3: Basic functionality
    if test_basic_functionality():
        success_count += 1
    
    print(f"\n{'='*60}")
    print(f"TEST RESULTS: {success_count}/{total_tests} tests passed")
    
    if success_count == total_tests:
        print("✓ All tests passed! Enhanced framework is ready.")
        print("\nThe compatibility fixes should resolve the Polars version issues.")
        print("You can now run the enhanced hypothesis testing with:")
        print("  python enhanced_testhyp1.py")
    else:
        print("✗ Some tests failed. Check the error messages above.")
    
    print("="*60)

if __name__ == "__main__":
    main()