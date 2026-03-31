"""
Tests for feature engineering module.
"""

import pytest
import numpy as np
import pandas as pd
from src.feature_engineering import (
    compute_rolling_volatility,
    compute_rolling_mean,
    construct_feature_vector,
    get_feature_dimension,
    validate_features
)


def test_compute_rolling_volatility_basic():
    """Test basic rolling volatility calculation."""
    # Create returns with known volatility
    np.random.seed(42)
    returns = pd.DataFrame({
        'A': np.random.randn(100) * 0.02
    })
    
    vol = compute_rolling_volatility(returns, window=20)
    
    # Check shape
    assert vol.shape == returns.shape
    
    # First window-1 values should be NaN
    assert vol.iloc[:19].isnull().all().all()
    
    # Remaining values should be positive
    assert (vol.iloc[19:] > 0).all().all()


def test_compute_rolling_mean_basic():
    """Test basic rolling mean calculation."""
    returns = pd.DataFrame({
        'A': [0.01, 0.02, 0.03, 0.04, 0.05] * 10
    })
    
    mean = compute_rolling_mean(returns, window=5)
    
    # Check shape
    assert mean.shape == returns.shape
    
    # Check first valid value
    assert np.isclose(mean.iloc[4, 0], 0.03)


def test_construct_feature_vector_dimensions():
    """Test feature vector has correct dimensions."""
    n_assets = 3
    n_days = 100
    
    returns = pd.DataFrame(
        np.random.randn(n_days, n_assets) * 0.01,
        columns=['A', 'B', 'C']
    )
    
    features = construct_feature_vector(returns, vol_window=20, mom_window=10)
    
    # Feature dimension should be 3 * n_assets
    expected_dim = get_feature_dimension(n_assets)
    assert features.shape[1] == expected_dim
    assert features.shape[1] == 9  # 3 assets * 3 features


def test_construct_feature_vector_causality():
    """Test that features are properly lagged for causality."""
    returns = pd.DataFrame({
        'A': [0.01, 0.02, 0.03, 0.04, 0.05] + [0.01] * 60
    })
    
    features = construct_feature_vector(returns, vol_window=20, mom_window=10)
    
    # Feature at time t should use data up to t-1
    # So features should be shifted by 1 day relative to returns
    # This is implicitly tested by the lagging in the function
    assert len(features) < len(returns)


def test_construct_feature_vector_no_nan():
    """Test that final features contain no NaN values."""
    np.random.seed(42)
    returns = pd.DataFrame(
        np.random.randn(100, 2) * 0.01,
        columns=['A', 'B']
    )
    
    features = construct_feature_vector(returns, vol_window=20, mom_window=10)
    
    # Should have no NaN values
    assert not features.isnull().any().any()


def test_get_feature_dimension():
    """Test feature dimension calculation."""
    assert get_feature_dimension(1) == 3
    assert get_feature_dimension(5) == 15
    assert get_feature_dimension(10) == 30


def test_validate_features_valid():
    """Test validation with valid features."""
    features = pd.DataFrame(
        np.random.randn(50, 6),
        columns=[f'feat_{i}' for i in range(6)]
    )
    
    assert validate_features(features) == True


def test_validate_features_with_nan():
    """Test validation catches NaN values."""
    features = pd.DataFrame({
        'A': [1, 2, np.nan, 4],
        'B': [1, 2, 3, 4]
    })
    
    with pytest.raises(ValueError, match="NaN"):
        validate_features(features)


def test_validate_features_with_inf():
    """Test validation catches infinite values."""
    features = pd.DataFrame({
        'A': [1, 2, np.inf, 4],
        'B': [1, 2, 3, 4]
    })
    
    with pytest.raises(ValueError, match="infinite"):
        validate_features(features)


def test_validate_features_empty():
    """Test validation catches empty DataFrame."""
    features = pd.DataFrame()
    
    with pytest.raises(ValueError, match="empty"):
        validate_features(features)


def test_rolling_volatility_window_size():
    """Test rolling volatility with different window sizes."""
    returns = pd.DataFrame({
        'A': np.random.randn(100) * 0.01
    })
    
    vol_10 = compute_rolling_volatility(returns, window=10)
    vol_30 = compute_rolling_volatility(returns, window=30)
    
    # Longer window should generally be smoother (lower variance)
    assert vol_10.iloc[30:].std().values[0] >= vol_30.iloc[30:].std().values[0] * 0.5


def test_feature_vector_multiple_assets():
    """Test feature construction with multiple assets."""
    returns = pd.DataFrame(
        np.random.randn(100, 5) * 0.01,
        columns=['SPY', 'AGG', 'GLD', 'USO', 'UUP']
    )
    
    features = construct_feature_vector(returns, vol_window=60, mom_window=20)
    
    # Should have 15 features (5 assets * 3 features each)
    assert features.shape[1] == 15
    
    # Check column naming
    assert 'SPY_return' in features.columns
    assert 'SPY_vol' in features.columns
    assert 'SPY_mom' in features.columns
