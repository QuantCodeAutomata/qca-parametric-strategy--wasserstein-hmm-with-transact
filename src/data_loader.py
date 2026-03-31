"""
Data loading and preprocessing module for asset allocation strategy.
"""

import pandas as pd
import numpy as np
import yfinance as yf
from typing import Dict, List, Tuple
from datetime import datetime


def load_asset_data(
    tickers: List[str],
    start_date: str,
    end_date: str,
    cache_path: str = None
) -> pd.DataFrame:
    """
    Load daily adjusted close prices for specified tickers from Yahoo Finance.
    
    Parameters:
    -----------
    tickers : List[str]
        List of ticker symbols to download
    start_date : str
        Start date in 'YYYY-MM-DD' format
    end_date : str
        End date in 'YYYY-MM-DD' format
    cache_path : str, optional
        Path to cache the downloaded data
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with dates as index and tickers as columns containing adjusted close prices
    """
    # Download data from Yahoo Finance
    data = yf.download(tickers, start=start_date, end=end_date, progress=False, auto_adjust=True)
    
    # Extract close prices (auto_adjust=True gives adjusted prices)
    if len(tickers) == 1:
        if 'Close' in data.columns:
            prices = data['Close'].to_frame()
            prices.columns = tickers
        else:
            prices = data.to_frame()
            prices.columns = tickers
    else:
        if 'Close' in data.columns.get_level_values(0):
            prices = data['Close']
        else:
            # Fallback: try to get the data directly
            prices = pd.DataFrame()
            for ticker in tickers:
                ticker_data = yf.download(ticker, start=start_date, end=end_date, progress=False, auto_adjust=True)
                prices[ticker] = ticker_data['Close']
    
    # Handle missing values by forward filling
    prices = prices.ffill()
    
    # Drop any remaining NaN values at the beginning
    prices = prices.dropna()
    
    # Cache if path provided
    if cache_path:
        prices.to_csv(cache_path)
    
    return prices


def align_trading_days(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Align all assets to common trading days (intersection of all trading days).
    
    Parameters:
    -----------
    prices : pd.DataFrame
        DataFrame with dates as index and tickers as columns
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with only dates where all assets have data
    """
    # Drop any rows with NaN values to ensure all assets trade on same days
    aligned_prices = prices.dropna()
    
    return aligned_prices


def compute_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Compute daily log returns from price series.
    
    Parameters:
    -----------
    prices : pd.DataFrame
        DataFrame with dates as index and tickers as columns containing prices
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with daily log returns
    """
    # Compute log returns: r_t = log(P_t) - log(P_{t-1})
    log_returns = np.log(prices).diff()
    
    # Drop the first row which will be NaN
    log_returns = log_returns.iloc[1:]
    
    return log_returns


def prepare_data(
    tickers: List[str],
    start_date: str,
    end_date: str,
    cache_path: str = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Complete data preparation pipeline: load, align, and compute returns.
    
    Parameters:
    -----------
    tickers : List[str]
        List of ticker symbols to download
    start_date : str
        Start date in 'YYYY-MM-DD' format
    end_date : str
        End date in 'YYYY-MM-DD' format
    cache_path : str, optional
        Path to cache the downloaded data
        
    Returns:
    --------
    Tuple[pd.DataFrame, pd.DataFrame]
        Tuple of (prices, returns) DataFrames
    """
    # Load data
    prices = load_asset_data(tickers, start_date, end_date, cache_path)
    
    # Align trading days
    prices = align_trading_days(prices)
    
    # Compute log returns
    returns = compute_log_returns(prices)
    
    return prices, returns


def get_default_tickers() -> Dict[str, str]:
    """
    Get the default asset class proxies as specified in the paper.
    
    Returns:
    --------
    Dict[str, str]
        Dictionary mapping asset class names to ticker symbols
    """
    return {
        'SPX': 'SPY',    # Equities
        'BOND': 'AGG',   # Fixed Income
        'GOLD': 'GLD',   # Commodities - Gold
        'OIL': 'USO',    # Commodities - Oil
        'USD': 'UUP'     # FX
    }
