"""
Tests for portfolio optimization module.
"""

import pytest
import numpy as np
from src.optimizer import (
    optimize_portfolio,
    compute_turnover,
    adjust_weights_for_returns,
    compute_effective_positions,
    compute_portfolio_return,
    validate_weights
)


def test_optimize_portfolio_basic():
    """Test basic portfolio optimization."""
    n_assets = 3
    expected_returns = np.array([0.001, 0.002, 0.0015])
    cov_matrix = np.eye(n_assets) * 0.0001
    previous_weights = np.ones(n_assets) / n_assets
    
    weights = optimize_portfolio(
        expected_returns,
        cov_matrix,
        previous_weights,
        risk_aversion=1.0,
        turnover_penalty=0.1
    )
    
    # Check constraints
    assert np.isclose(weights.sum(), 1.0)
    assert np.all(weights >= 0)
    assert np.all(weights <= 0.5)


def test_optimize_portfolio_high_turnover_penalty():
    """Test that high turnover penalty keeps weights close to previous."""
    n_assets = 3
    expected_returns = np.array([0.01, 0.001, 0.001])
    cov_matrix = np.eye(n_assets) * 0.0001
    previous_weights = np.array([0.2, 0.4, 0.4])
    
    # Very high turnover penalty
    weights = optimize_portfolio(
        expected_returns,
        cov_matrix,
        previous_weights,
        risk_aversion=1.0,
        turnover_penalty=10.0
    )
    
    # Weights should be close to previous
    assert np.allclose(weights, previous_weights, atol=0.1)


def test_optimize_portfolio_constraints():
    """Test portfolio optimization respects constraints."""
    n_assets = 5
    expected_returns = np.random.randn(n_assets) * 0.01
    cov_matrix = np.eye(n_assets) * 0.0001
    previous_weights = np.ones(n_assets) / n_assets
    
    max_weight = 0.3
    
    weights = optimize_portfolio(
        expected_returns,
        cov_matrix,
        previous_weights,
        max_weight=max_weight
    )
    
    # Check max weight constraint
    assert np.all(weights <= max_weight + 1e-6)
    
    # Check sum to 1
    assert np.isclose(weights.sum(), 1.0)
    
    # Check non-negative
    assert np.all(weights >= -1e-6)


def test_compute_turnover_no_change():
    """Test turnover is zero when weights don't change."""
    weights = np.array([0.2, 0.3, 0.5])
    
    turnover = compute_turnover(weights, weights)
    
    assert np.isclose(turnover, 0.0)


def test_compute_turnover_complete_rebalance():
    """Test turnover with complete rebalancing."""
    weights_old = np.array([1.0, 0.0, 0.0])
    weights_new = np.array([0.0, 0.0, 1.0])
    
    turnover = compute_turnover(weights_new, weights_old)
    
    # Complete rebalance should give turnover of 1.0
    assert np.isclose(turnover, 1.0)


def test_compute_turnover_partial():
    """Test turnover with partial rebalancing."""
    weights_old = np.array([0.5, 0.5])
    weights_new = np.array([0.6, 0.4])
    
    turnover = compute_turnover(weights_new, weights_old)
    
    # Turnover should be 0.5 * (0.1 + 0.1) = 0.1
    assert np.isclose(turnover, 0.1)


def test_adjust_weights_for_returns_positive():
    """Test weight adjustment with positive returns."""
    weights = np.array([0.5, 0.5])
    returns = np.array([0.1, 0.0])  # Log returns
    
    adjusted = adjust_weights_for_returns(weights, returns)
    
    # First asset should have higher weight after positive return
    assert adjusted[0] > weights[0]
    assert adjusted[1] < weights[1]
    
    # Should still sum to 1
    assert np.isclose(adjusted.sum(), 1.0)


def test_adjust_weights_for_returns_zero():
    """Test weight adjustment with zero returns."""
    weights = np.array([0.3, 0.4, 0.3])
    returns = np.array([0.0, 0.0, 0.0])
    
    adjusted = adjust_weights_for_returns(weights, returns)
    
    # Weights should remain unchanged
    assert np.allclose(adjusted, weights)


def test_compute_effective_positions():
    """Test effective position counting."""
    # Three significant positions
    weights = np.array([0.4, 0.3, 0.2, 0.05, 0.05])
    
    eff_pos = compute_effective_positions(weights, threshold=0.01)
    
    assert eff_pos == 5
    
    # With higher threshold
    eff_pos = compute_effective_positions(weights, threshold=0.1)
    
    assert eff_pos == 3


def test_compute_portfolio_return():
    """Test portfolio return calculation."""
    weights = np.array([0.5, 0.5])
    returns = np.array([0.02, 0.01])
    
    port_return = compute_portfolio_return(weights, returns)
    
    expected = 0.5 * 0.02 + 0.5 * 0.01
    assert np.isclose(port_return, expected)


def test_compute_portfolio_return_single_asset():
    """Test portfolio return with single asset."""
    weights = np.array([1.0])
    returns = np.array([0.05])
    
    port_return = compute_portfolio_return(weights, returns)
    
    assert np.isclose(port_return, 0.05)


def test_validate_weights_valid():
    """Test weight validation with valid weights."""
    weights = np.array([0.2, 0.3, 0.5])
    
    assert validate_weights(weights) == True


def test_validate_weights_not_sum_to_one():
    """Test weight validation catches incorrect sum."""
    weights = np.array([0.2, 0.3, 0.4])
    
    assert validate_weights(weights) == False


def test_validate_weights_negative():
    """Test weight validation catches negative weights."""
    weights = np.array([0.5, -0.1, 0.6])
    
    assert validate_weights(weights) == False


def test_validate_weights_nan():
    """Test weight validation catches NaN."""
    weights = np.array([0.5, np.nan, 0.5])
    
    assert validate_weights(weights) == False


def test_validate_weights_inf():
    """Test weight validation catches infinite values."""
    weights = np.array([0.5, np.inf, 0.5])
    
    assert validate_weights(weights) == False


def test_optimize_portfolio_equal_returns():
    """Test optimization with equal expected returns."""
    n_assets = 4
    expected_returns = np.ones(n_assets) * 0.01
    cov_matrix = np.eye(n_assets) * 0.0001
    previous_weights = np.ones(n_assets) / n_assets
    
    weights = optimize_portfolio(
        expected_returns,
        cov_matrix,
        previous_weights,
        turnover_penalty=0.5
    )
    
    # With equal returns and high turnover penalty, should stay near equal weight
    assert np.allclose(weights, previous_weights, atol=0.1)
