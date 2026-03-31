"""
Tests for performance metrics module.
"""

import pytest
import numpy as np
import pandas as pd
from src.metrics import (
    compute_sharpe_ratio,
    compute_sortino_ratio,
    compute_max_drawdown,
    compute_annualized_return,
    compute_annualized_volatility,
    compute_calmar_ratio,
    compute_all_metrics
)


def test_compute_sharpe_ratio_positive():
    """Test Sharpe ratio with positive returns."""
    # Constant positive returns
    returns = np.ones(252) * 0.001
    
    sharpe = compute_sharpe_ratio(returns, annualization_factor=252)
    
    # Should be positive and large (no volatility in constant returns)
    assert sharpe > 0


def test_compute_sharpe_ratio_zero_returns():
    """Test Sharpe ratio with zero returns."""
    returns = np.zeros(100)
    
    sharpe = compute_sharpe_ratio(returns)
    
    assert sharpe == 0.0


def test_compute_sharpe_ratio_negative():
    """Test Sharpe ratio with negative returns."""
    returns = np.ones(252) * -0.001
    
    sharpe = compute_sharpe_ratio(returns)
    
    assert sharpe < 0


def test_compute_sortino_ratio_no_downside():
    """Test Sortino ratio with no downside."""
    returns = np.ones(252) * 0.001
    
    sortino = compute_sortino_ratio(returns)
    
    # Should be very high (infinite) with no downside
    assert sortino > 0


def test_compute_sortino_ratio_with_downside():
    """Test Sortino ratio with downside volatility."""
    returns = np.array([0.01, -0.01, 0.01, -0.01] * 63)
    
    sortino = compute_sortino_ratio(returns)
    
    # Should be finite
    assert np.isfinite(sortino)


def test_compute_max_drawdown_no_drawdown():
    """Test max drawdown with monotonically increasing returns."""
    returns = np.ones(100) * 0.01
    
    max_dd = compute_max_drawdown(returns)
    
    # Should be zero or very close
    assert max_dd >= -1e-10


def test_compute_max_drawdown_with_loss():
    """Test max drawdown with losses."""
    # Returns that go up then down
    returns = np.array([0.1] * 10 + [-0.05] * 10)
    
    max_dd = compute_max_drawdown(returns)
    
    # Should be negative
    assert max_dd < 0


def test_compute_max_drawdown_complete_loss():
    """Test max drawdown with severe losses."""
    returns = np.array([0.0, -0.5, -0.5])
    
    max_dd = compute_max_drawdown(returns)
    
    # Should be close to -0.75 (1 - 0.25)
    assert max_dd < -0.5


def test_compute_annualized_return():
    """Test annualized return calculation."""
    # 1% daily return for 252 days
    returns = np.ones(252) * 0.01
    
    ann_return = compute_annualized_return(returns, annualization_factor=252)
    
    # Should be approximately 0.01 * 252 = 2.52
    assert np.isclose(ann_return, 2.52, atol=0.01)


def test_compute_annualized_volatility():
    """Test annualized volatility calculation."""
    np.random.seed(42)
    returns = np.random.randn(252) * 0.01
    
    ann_vol = compute_annualized_volatility(returns, annualization_factor=252)
    
    # Should be approximately 0.01 * sqrt(252) ≈ 0.159
    assert ann_vol > 0.1
    assert ann_vol < 0.2


def test_compute_calmar_ratio():
    """Test Calmar ratio calculation."""
    # Positive returns with some drawdown
    returns = np.array([0.01] * 100 + [-0.005] * 20 + [0.01] * 100)
    
    calmar = compute_calmar_ratio(returns)
    
    # Should be positive
    assert calmar > 0


def test_compute_all_metrics():
    """Test computation of all metrics together."""
    np.random.seed(42)
    returns = np.random.randn(252) * 0.01 + 0.0005
    
    metrics = compute_all_metrics(returns)
    
    # Check all expected keys present
    assert 'annualized_return' in metrics
    assert 'annualized_volatility' in metrics
    assert 'sharpe_ratio' in metrics
    assert 'sortino_ratio' in metrics
    assert 'max_drawdown' in metrics
    assert 'calmar_ratio' in metrics
    assert 'total_return' in metrics
    assert 'n_periods' in metrics
    
    # Check values are reasonable
    assert np.isfinite(metrics['sharpe_ratio'])
    assert metrics['n_periods'] == 252


def test_sharpe_ratio_properties():
    """Test mathematical properties of Sharpe ratio."""
    np.random.seed(42)
    returns1 = np.random.randn(252) * 0.01 + 0.001
    returns2 = np.random.randn(252) * 0.02 + 0.001
    
    sharpe1 = compute_sharpe_ratio(returns1)
    sharpe2 = compute_sharpe_ratio(returns2)
    
    # Higher volatility with same mean should give lower Sharpe
    assert sharpe1 > sharpe2


def test_max_drawdown_properties():
    """Test max drawdown is always non-positive."""
    np.random.seed(42)
    
    for _ in range(10):
        returns = np.random.randn(100) * 0.01
        max_dd = compute_max_drawdown(returns)
        
        assert max_dd <= 0


def test_metrics_with_empty_returns():
    """Test metrics with empty returns array."""
    returns = np.array([])
    
    sharpe = compute_sharpe_ratio(returns)
    sortino = compute_sortino_ratio(returns)
    max_dd = compute_max_drawdown(returns)
    
    assert sharpe == 0.0
    assert sortino == 0.0
    assert max_dd == 0.0


def test_annualization_factor():
    """Test different annualization factors."""
    returns = np.ones(12) * 0.01
    
    # Monthly data
    ann_return_monthly = compute_annualized_return(returns, annualization_factor=12)
    
    # Should be 0.01 * 12 = 0.12
    assert np.isclose(ann_return_monthly, 0.12)
