"""
Feature engineering module for computing rolling statistics and feature vectors.
"""

import pandas as pd
import numpy as np
from typing import Tuple


def compute_rolling_volatility(
    returns: pd.DataFrame,
    window: int = 60
) -> pd.DataFrame:
    """
    Compute rolling volatility (standard deviation) of returns.
    
    Parameters:
    -----------
    returns : pd.DataFrame
        DataFrame with daily log returns
    window : int
        Rolling window size in days (default: 60)
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with rolling volatility for each asset
    """
    # Compute rolling standard deviation
    rolling_vol = returns.rolling(window=window, min_periods=window).std()
    
    return rolling_vol


def compute_rolling_mean(
    returns: pd.DataFrame,
    window: int = 20
) -> pd.DataFrame:
    """
    Compute rolling mean (momentum) of returns.
    
    Parameters:
    -----------
    returns : pd.DataFrame
        DataFrame with daily log returns
    window : int
        Rolling window size in days (default: 20)
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with rolling mean for each asset
    """
    # Compute rolling mean
    rolling_mean = returns.rolling(window=window, min_periods=window).mean()
    
    return rolling_mean


def construct_feature_vector(
    returns: pd.DataFrame,
    vol_window: int = 60,
    mom_window: int = 20
) -> pd.DataFrame:
    """
    Construct the feature vector x_t for each day as specified in the paper.
    
    The feature vector consists of:
    - Daily log returns (lagged by 1 day for causality)
    - Rolling volatility (lagged by 1 day)
    - Rolling mean/momentum (lagged by 1 day)
    
    For N assets, this creates a 3N-dimensional feature vector.
    
    Parameters:
    -----------
    returns : pd.DataFrame
        DataFrame with daily log returns
    vol_window : int
        Window size for volatility calculation (default: 60)
    mom_window : int
        Window size for momentum calculation (default: 20)
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with feature vectors (3N columns) for each date
    """
    # Compute rolling statistics
    rolling_vol = compute_rolling_volatility(returns, window=vol_window)
    rolling_mom = compute_rolling_mean(returns, window=mom_window)
    
    # Lag all features by 1 day to ensure strict causality
    # x_t uses data up to t-1
    lagged_returns = returns.shift(1)
    lagged_vol = rolling_vol.shift(1)
    lagged_mom = rolling_mom.shift(1)
    
    # Concatenate features: [returns; volatility; momentum]
    feature_list = []
    
    # Add lagged returns
    for col in lagged_returns.columns:
        feature_list.append(lagged_returns[col].rename(f'{col}_return'))
    
    # Add lagged volatility
    for col in lagged_vol.columns:
        feature_list.append(lagged_vol[col].rename(f'{col}_vol'))
    
    # Add lagged momentum
    for col in lagged_mom.columns:
        feature_list.append(lagged_mom[col].rename(f'{col}_mom'))
    
    # Combine all features
    features = pd.concat(feature_list, axis=1)
    
    # Drop rows with NaN values (initial period where rolling windows are not complete)
    features = features.dropna()
    
    return features


def get_feature_dimension(n_assets: int) -> int:
    """
    Calculate the dimension of the feature vector.
    
    Parameters:
    -----------
    n_assets : int
        Number of assets
        
    Returns:
    --------
    int
        Feature dimension (3 * n_assets)
    """
    return 3 * n_assets


def validate_features(features: pd.DataFrame) -> bool:
    """
    Validate that features are properly constructed and contain no invalid values.
    
    Parameters:
    -----------
    features : pd.DataFrame
        Feature DataFrame to validate
        
    Returns:
    --------
    bool
        True if features are valid, raises ValueError otherwise
    """
    # Check for NaN values
    if features.isnull().any().any():
        raise ValueError("Features contain NaN values")
    
    # Check for infinite values
    if np.isinf(features.values).any():
        raise ValueError("Features contain infinite values")
    
    # Check that we have data
    if len(features) == 0:
        raise ValueError("Features DataFrame is empty")
    
    return True
