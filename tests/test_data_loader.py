"""
Tests for data loading and preprocessing module.
"""

import pytest
import numpy as np
import pandas as pd
from src.data_loader import (
    compute_log_returns,
    align_trading_days,
    get_default_tickers
)


def test_compute_log_returns_basic():
    """Test basic log returns calculation."""
    # Create simple price series
    prices = pd.DataFrame({
        'A': [100, 110, 105],
        'B': [50, 55, 52.5]
    })
    
    returns = compute_log_returns(prices)
    
    # Check shape
    assert returns.shape == (2, 2)
    
    # Check values approximately
    expected_A_1 = np.log(110/100)
    assert np.isclose(returns.iloc[0, 0], expected_A_1)
    
    # Check no NaN in result
    assert not returns.isnull().any().any()


def test_compute_log_returns_single_asset():
    """Test log returns with single asset."""
    prices = pd.DataFrame({
        'A': [100, 105, 110, 108]
    })
    
    returns = compute_log_returns(prices)
    
    assert returns.shape == (3, 1)
    assert not returns.isnull().any().any()


def test_compute_log_returns_properties():
    """Test mathematical properties of log returns."""
    prices = pd.DataFrame({
        'A': [100, 110, 121],
        'B': [50, 60, 72]
    })
    
    returns = compute_log_returns(prices)
    
    # Log returns should be additive
    total_return_A = returns['A'].sum()
    expected_total = np.log(121/100)
    assert np.isclose(total_return_A, expected_total)


def test_align_trading_days_no_missing():
    """Test alignment when no missing data."""
    dates = pd.date_range('2020-01-01', periods=5)
    prices = pd.DataFrame({
        'A': [100, 101, 102, 103, 104],
        'B': [50, 51, 52, 53, 54]
    }, index=dates)
    
    aligned = align_trading_days(prices)
    
    assert len(aligned) == 5
    assert aligned.shape == prices.shape


def test_align_trading_days_with_missing():
    """Test alignment with missing data."""
    dates = pd.date_range('2020-01-01', periods=5)
    prices = pd.DataFrame({
        'A': [100, 101, np.nan, 103, 104],
        'B': [50, 51, 52, 53, 54]
    }, index=dates)
    
    aligned = align_trading_days(prices)
    
    # Should drop row with NaN
    assert len(aligned) == 4
    assert not aligned.isnull().any().any()


def test_get_default_tickers():
    """Test default ticker configuration."""
    tickers = get_default_tickers()
    
    # Check all required asset classes present
    assert 'SPX' in tickers
    assert 'BOND' in tickers
    assert 'GOLD' in tickers
    assert 'OIL' in tickers
    assert 'USD' in tickers
    
    # Check correct tickers
    assert tickers['SPX'] == 'SPY'
    assert tickers['BOND'] == 'AGG'
    assert tickers['GOLD'] == 'GLD'
    assert tickers['OIL'] == 'USO'
    assert tickers['USD'] == 'UUP'


def test_log_returns_extreme_values():
    """Test log returns with extreme price changes."""
    prices = pd.DataFrame({
        'A': [100, 200, 50, 150]
    })
    
    returns = compute_log_returns(prices)
    
    # Check finite values
    assert np.all(np.isfinite(returns.values))
    
    # Check first return (100 -> 200, should be log(2))
    assert np.isclose(returns.iloc[0, 0], np.log(2))


def test_log_returns_empty_dataframe():
    """Test log returns with empty DataFrame."""
    prices = pd.DataFrame()
    
    returns = compute_log_returns(prices)
    
    assert len(returns) == 0
