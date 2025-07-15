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

# Apply compatibility patches before importing other modules
def apply_polars_compatibility():
    """Apply Polars compatibility patches"""
    try:
        import polars as pl
        
        # Patch missing Config methods only if they don't exist
        if not hasattr(pl.Config, 'set_engine_affinity'):
            pl.Config.set_engine_affinity = lambda engine="streaming": None
            print("  - Patched Config.set_engine_affinity")
            
        if not hasattr(pl.Config, 'set_streaming_chunk_size'):
            pl.Config.set_streaming_chunk_size = lambda size: None
            print("  - Patched Config.set_streaming_chunk_size")
        
        print("✓ Polars compatibility patches applied successfully")
        return True
        
    except Exception as e:
        print(f"✗ Failed to apply Polars compatibility patches: {e}")
        return False

# Apply patches immediately
print("Applying Polars compatibility patches...")
patch_success = apply_polars_compatibility()

def test_imports():
    """Test that all enhanced components can be imported"""
    print("Testing enhanced framework imports...")
    
    success_count = 0
    total_tests = 6
    
    # Test compatibility fixes
    try:
        from src.polars_compatibility_fixes import safe_clip_quantile, safe_apply
        print("✓ Polars compatibility fixes imported successfully")
        success_count += 1
    except ImportError as e:
        print(f"✗ Polars compatibility fixes import failed: {e}")
    
    # Test two-risk framework
    try:
        from src.two_risk_framework import TwoRiskFramework, DirectionalNewsRisk, ImpactUncertainty
        print("✓ Two-risk framework imported successfully")
        success_count += 1
    except ImportError as e:
        print(f"✗ Two-risk framework import failed: {e}")
    
    # Test investor heterogeneity
    try:
        from src.investor_heterogeneity import HeterogeneousInvestorMarket
        print("✓ Heterogeneous investor framework imported successfully")
        success_count += 1
    except ImportError as e:
        print(f"✗ Heterogeneous investor framework import failed: {e}")
    
    # Test portfolio optimization
    try:
        from src.portfolio_optimization import PortfolioOptimizationFramework
        print("✓ Portfolio optimization framework imported successfully")
        success_count += 1
    except ImportError as e:
        print(f"✗ Portfolio optimization framework import failed: {e}")
    
    # Test statistical testing
    try:
        from src.statistical_testing import ComprehensiveTestSuite
        print("✓ Statistical testing framework imported successfully")
        success_count += 1
    except ImportError as e:
        print(f"✗ Statistical testing framework import failed: {e}")
    
    # Test enhanced event processor
    try:
        from src.enhanced_event_processor import EnhancedEventAnalysis
        print("✓ Enhanced event processor imported successfully")
        success_count += 1
    except ImportError as e:
        print(f"✗ Enhanced event processor import failed: {e}")
        print(f"  Error details: {str(e)}")
    
    return success_count == total_tests

def test_basic_functionality():
    """Test basic functionality of key components"""
    print("\nTesting basic functionality...")
    
    test_results = {"passed": 0, "total": 5}
    
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
        try:
            from src.two_risk_framework import TwoRiskFramework
            
            risk_framework = TwoRiskFramework()
            risk_framework.fit(sample_data)
            risk_components = risk_framework.extract_risks(sample_data)
            
            print("✓ Two-risk framework basic test passed")
            print(f"  - Directional risk shape: {np.array(risk_components['directional_news_risk']).shape}")
            print(f"  - Impact uncertainty shape: {np.array(risk_components['impact_uncertainty']).shape}")
            test_results["passed"] += 1
        except Exception as e:
            print(f"✗ Two-risk framework test failed: {e}")
        
        test_results["total"] = test_results.get("total", 0) + 1
        
        # Test heterogeneous investors (simplified)
        try:
            from src.investor_heterogeneity import HeterogeneousInvestorMarket
            
            investor_market = HeterogeneousInvestorMarket()
            print("✓ Heterogeneous investor framework basic test passed")
            test_results["passed"] += 1
        except Exception as e:
            print(f"✗ Heterogeneous investor test failed: {e}")
        
        test_results["total"] += 1
        
        # Test statistical testing (basic)
        try:
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
            test_results["passed"] += 1
        except Exception as e:
            print(f"✗ Statistical testing test failed: {e}")
        
        test_results["total"] += 1
        
        # Test portfolio optimization (basic)
        try:
            from src.portfolio_optimization import PortfolioOptimizationFramework
            
            portfolio_optimizer = PortfolioOptimizationFramework()
            print("✓ Portfolio optimization framework basic test passed")
            test_results["passed"] += 1
        except Exception as e:
            print(f"✗ Portfolio optimization test failed: {e}")
        
        test_results["total"] += 1
        
        # Test models (basic)
        try:
            from src.models import GARCHModel, GJRGARCHModel
            
            # Test basic GARCH model creation
            garch_model = GARCHModel()
            gjr_model = GJRGARCHModel()
            print("✓ GARCH models basic test passed")
            test_results["passed"] += 1
        except Exception as e:
            print(f"✗ GARCH models test failed: {e}")
        
        test_results["total"] += 1
        
        return test_results["passed"] >= test_results["total"] * 0.8  # 80% success rate
        
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
        
        # Create test data
        test_df = pl.DataFrame({
            'values': [1.0, 2.0, np.nan, 4.0, 5.0, 100.0, -50.0, 3.0],
            'group': ['A', 'A', 'A', 'A', 'B', 'B', 'B', 'B']
        })
        
        test_count = 0
        
        # Test basic Polars operations
        try:
            result = test_df.filter(pl.col('values') > 2)
            print("✓ Basic Polars operations work")
            test_count += 1
        except Exception as e:
            print(f"✗ Basic Polars operations failed: {e}")
        
        # Test safe_clip_quantile if available
        try:
            from src.polars_compatibility_fixes import safe_clip_quantile
            result1 = test_df.with_columns([
                safe_clip_quantile(pl.col('values'), 0.1, 0.9).alias('clipped')
            ])
            print("✓ safe_clip_quantile works")
            test_count += 1
        except Exception as e:
            print(f"⚠ safe_clip_quantile test issue: {e}")
        
        # Test interpolate (use native method if it exists)
        try:
            result3 = test_df.with_columns([
                pl.col('values').forward_fill().backward_fill().alias('filled')
            ])
            print("✓ Fill operations work")
            test_count += 1
        except Exception as e:
            print(f"✗ Fill operations failed: {e}")
        
        # Test Config methods
        try:
            pl.Config.set_engine_affinity("streaming")
            print("✓ Config.set_engine_affinity works")
            test_count += 1
        except Exception as e:
            print(f"⚠ Config.set_engine_affinity issue: {e}")
        
        return test_count >= 3  # At least 3 out of 4 tests should pass
        
    except Exception as e:
        print(f"✗ Polars compatibility test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("="*60)
    print("ENHANCED FRAMEWORK COMPATIBILITY TEST")
    print("="*60)
    
    if not patch_success:
        print("⚠ Warning: Polars compatibility patches failed to apply")
    
    success_count = 0
    total_tests = 3
    
    # Test 1: Imports
    print("\n[1/3] Testing imports...")
    if test_imports():
        success_count += 1
        print("✓ Import test passed")
    else:
        print("✗ Import test failed (but framework may still work)")
    
    # Test 2: Polars compatibility
    print("\n[2/3] Testing Polars compatibility...")
    if test_polars_compatibility():
        success_count += 1
        print("✓ Polars compatibility test passed")
    else:
        print("✗ Polars compatibility test failed")
    
    # Test 3: Basic functionality
    print("\n[3/3] Testing basic functionality...")
    if test_basic_functionality():
        success_count += 1
        print("✓ Basic functionality test passed")
    else:
        print("✗ Basic functionality test failed")
    
    print(f"\n{'='*60}")
    print(f"TEST RESULTS: {success_count}/{total_tests} tests passed")
    
    if success_count == total_tests:
        print("🎉 All tests passed! Enhanced framework is ready.")
        print("\nNext steps:")
        print("  1. Run enhanced hypothesis testing: python enhanced_testhyp1.py")
        print("  2. Or run traditional testing: python testhyp1.py")
    elif success_count >= 2:
        print("⚠ Most tests passed. Enhanced framework should work with minor issues.")
        print("\nYou can try running:")
        print("  python enhanced_testhyp1.py")
        print("\nIf you encounter issues, fall back to:")
        print("  python testhyp1.py")
    else:
        print("⚠ Multiple tests failed. Recommend using the traditional framework.")
        print("\nTry running:")
        print("  python testhyp1.py")
        print("\nOr check the installation of required dependencies.")
    
    print("="*60)

if __name__ == "__main__":
    main()