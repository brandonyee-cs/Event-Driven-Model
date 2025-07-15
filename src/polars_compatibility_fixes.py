# src/polars_compatibility_fixes.py
"""
Polars Compatibility Fixes
Addresses version-specific method names and syntax differences
"""

import polars as pl
import numpy as np
from typing import Union, Optional

def safe_clip_quantile(expr: pl.Expr, lower: float, upper: float) -> pl.Expr:
    """
    Safe implementation of clip_quantile that works across Polars versions
    """
    try:
        # Try the newer method first
        return expr.clip(expr.quantile(lower), expr.quantile(upper))
    except AttributeError:
        try:
            # Try older method
            return expr.clip_min(expr.quantile(lower)).clip_max(expr.quantile(upper))
        except AttributeError:
            # Fallback to manual implementation
            return pl.when(expr < expr.quantile(lower)).then(expr.quantile(lower))\
                    .when(expr > expr.quantile(upper)).then(expr.quantile(upper))\
                    .otherwise(expr)

def safe_apply(expr: pl.Expr, func, return_dtype: Optional[pl.DataType] = None) -> pl.Expr:
    """
    Safe implementation of apply that works across Polars versions
    """
    try:
        # Try the newer method
        if return_dtype:
            return expr.map_elements(func, return_dtype=return_dtype)
        else:
            return expr.map_elements(func)
    except AttributeError:
        try:
            # Try older method
            return expr.apply(func)
        except AttributeError:
            # Fallback - this should work in most versions
            return expr.map(func)

def safe_interpolate(expr: pl.Expr, method: str = "linear") -> pl.Expr:
    """
    Safe implementation of interpolate that works across Polars versions
    Note: This should only be called if interpolate doesn't exist
    """
    # Try forward fill then backward fill as fallback
    try:
        return expr.forward_fill().backward_fill()
    except AttributeError:
        # If even forward_fill doesn't exist, return the original expression
        return expr

def safe_rolling_mean(expr: pl.Expr, window_size: int, min_periods: Optional[int] = None, center: bool = False) -> pl.Expr:
    """
    Safe implementation of rolling_mean with consistent parameters
    """
    try:
        if min_periods is not None and center:
            return expr.rolling_mean(window_size=window_size, min_periods=min_periods, center=center)
        elif min_periods is not None:
            return expr.rolling_mean(window_size=window_size, min_periods=min_periods)
        else:
            return expr.rolling_mean(window_size=window_size)
    except TypeError:
        # Fallback for older versions that don't support all parameters
        return expr.rolling_mean(window_size)

def safe_rolling_std(expr: pl.Expr, window_size: int, min_periods: Optional[int] = None) -> pl.Expr:
    """
    Safe implementation of rolling_std with consistent parameters
    """
    try:
        if min_periods is not None:
            return expr.rolling_std(window_size=window_size, min_periods=min_periods)
        else:
            return expr.rolling_std(window_size=window_size)
    except TypeError:
        return expr.rolling_std(window_size)

def safe_rolling_var(expr: pl.Expr, window_size: int, min_periods: Optional[int] = None) -> pl.Expr:
    """
    Safe implementation of rolling_var with consistent parameters
    """
    try:
        if min_periods is not None:
            return expr.rolling_var(window_size=window_size, min_periods=min_periods)
        else:
            return expr.rolling_var(window_size=window_size)
    except TypeError:
        return expr.rolling_var(window_size)

def safe_set_engine_affinity(engine: str = "streaming"):
    """
    Safe implementation of set_engine_affinity that works across Polars versions
    """
    try:
        # Try the newer method
        pl.Config.set_engine_affinity(engine=engine)
    except AttributeError:
        # Older versions don't have this method - just pass silently
        pass
    except Exception:
        # Any other error - just pass silently
        pass

def safe_config_set_streaming_chunk_size(size: int):
    """
    Safe configuration for streaming chunk size
    """
    try:
        pl.Config.set_streaming_chunk_size(size)
    except (AttributeError, Exception):
        # Not available in this version - pass silently
        pass

def patch_polars_methods():
    """
    Apply compatibility patches to Polars expressions and Config
    """
    # Only patch Config methods if they don't exist
    if not hasattr(pl.Config, 'set_engine_affinity'):
        pl.Config.set_engine_affinity = safe_set_engine_affinity
        
    if not hasattr(pl.Config, 'set_streaming_chunk_size'):
        pl.Config.set_streaming_chunk_size = safe_config_set_streaming_chunk_size
    
    # Only patch expression methods if they don't exist
    if not hasattr(pl.Expr, 'clip_quantile'):
        pl.Expr.clip_quantile = lambda self, lower, upper: safe_clip_quantile(self, lower, upper)
    
    # Only patch apply methods if neither exists
    if not hasattr(pl.Expr, 'apply') and not hasattr(pl.Expr, 'map_elements'):
        pl.Expr.apply = lambda self, func: safe_apply(self, func)
    
    # Don't patch interpolate - it usually exists and works fine
    # If it doesn't exist, users can call safe_interpolate directly

# Configure Polars safely
safe_set_engine_affinity("streaming")
safe_config_set_streaming_chunk_size(1000000)

# Apply conservative patches on import
patch_polars_methods()