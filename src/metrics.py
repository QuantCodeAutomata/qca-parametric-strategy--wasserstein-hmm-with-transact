"""
Performance metrics and analysis module.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple


def compute_sharpe_ratio(
    returns: np.ndarray,
    risk_free_rate: float = 0.0,
    annualization_factor: int = 252
) -> float:
    """
    Compute annualized Sharpe ratio.
    
    Sharpe = (mean_return - rf) / std_return * sqrt(annualization_factor)
    
    Parameters:
    -----------
    returns : np.ndarray
        Array of returns
    risk_free_rate : float
        Risk-free rate (annualized)
    annualization_factor : int
        Number of periods per year (252 for daily)
        
    Returns:
    --------
    float
        Annualized Sharpe ratio
    """
    if len(returns) == 0:
        return 0.0
    
    excess_returns = returns - risk_free_rate / annualization_factor
    
    if np.std(excess_returns) == 0:
        return 0.0
    
    sharpe = np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(annualization_factor)
    
    return sharpe


def compute_sortino_ratio(
    returns: np.ndarray,
    risk_free_rate: float = 0.0,
    annualization_factor: int = 252
) -> float:
    """
    Compute annualized Sortino ratio.
    
    Sortino = (mean_return - rf) / downside_std * sqrt(annualization_factor)
    
    Parameters:
    -----------
    returns : np.ndarray
        Array of returns
    risk_free_rate : float
        Risk-free rate (annualized)
    annualization_factor : int
        Number of periods per year
        
    Returns:
    --------
    float
        Annualized Sortino ratio
    """
    if len(returns) == 0:
        return 0.0
    
    excess_returns = returns - risk_free_rate / annualization_factor
    
    # Downside deviation (only negative returns)
    downside_returns = excess_returns[excess_returns < 0]
    
    if len(downside_returns) == 0:
        return np.inf if np.mean(excess_returns) > 0 else 0.0
    
    downside_std = np.std(downside_returns)
    
    if downside_std == 0:
        return 0.0
    
    sortino = np.mean(excess_returns) / downside_std * np.sqrt(annualization_factor)
    
    return sortino


def compute_max_drawdown(returns: np.ndarray) -> float:
    """
    Compute maximum drawdown from returns series.
    
    Parameters:
    -----------
    returns : np.ndarray
        Array of returns
        
    Returns:
    --------
    float
        Maximum drawdown (negative value)
    """
    if len(returns) == 0:
        return 0.0
    
    # Compute cumulative returns
    cumulative = np.exp(np.cumsum(returns))
    
    # Compute running maximum
    running_max = np.maximum.accumulate(cumulative)
    
    # Compute drawdown
    drawdown = (cumulative - running_max) / running_max
    
    # Maximum drawdown
    max_dd = np.min(drawdown)
    
    return max_dd


def compute_annualized_return(
    returns: np.ndarray,
    annualization_factor: int = 252
) -> float:
    """
    Compute annualized return.
    
    Parameters:
    -----------
    returns : np.ndarray
        Array of log returns
    annualization_factor : int
        Number of periods per year
        
    Returns:
    --------
    float
        Annualized return
    """
    if len(returns) == 0:
        return 0.0
    
    # Total log return
    total_log_return = np.sum(returns)
    
    # Annualize
    n_periods = len(returns)
    annualized = total_log_return * annualization_factor / n_periods
    
    return annualized


def compute_annualized_volatility(
    returns: np.ndarray,
    annualization_factor: int = 252
) -> float:
    """
    Compute annualized volatility.
    
    Parameters:
    -----------
    returns : np.ndarray
        Array of returns
    annualization_factor : int
        Number of periods per year
        
    Returns:
    --------
    float
        Annualized volatility
    """
    if len(returns) == 0:
        return 0.0
    
    return np.std(returns) * np.sqrt(annualization_factor)


def compute_calmar_ratio(
    returns: np.ndarray,
    annualization_factor: int = 252
) -> float:
    """
    Compute Calmar ratio (annualized return / max drawdown).
    
    Parameters:
    -----------
    returns : np.ndarray
        Array of returns
    annualization_factor : int
        Number of periods per year
        
    Returns:
    --------
    float
        Calmar ratio
    """
    ann_return = compute_annualized_return(returns, annualization_factor)
    max_dd = compute_max_drawdown(returns)
    
    if max_dd == 0:
        return np.inf if ann_return > 0 else 0.0
    
    return ann_return / abs(max_dd)


def compute_all_metrics(
    returns: np.ndarray,
    risk_free_rate: float = 0.0,
    annualization_factor: int = 252
) -> Dict[str, float]:
    """
    Compute all performance metrics.
    
    Parameters:
    -----------
    returns : np.ndarray
        Array of returns
    risk_free_rate : float
        Risk-free rate (annualized)
    annualization_factor : int
        Number of periods per year
        
    Returns:
    --------
    Dict[str, float]
        Dictionary of all metrics
    """
    metrics = {
        'annualized_return': compute_annualized_return(returns, annualization_factor),
        'annualized_volatility': compute_annualized_volatility(returns, annualization_factor),
        'sharpe_ratio': compute_sharpe_ratio(returns, risk_free_rate, annualization_factor),
        'sortino_ratio': compute_sortino_ratio(returns, risk_free_rate, annualization_factor),
        'max_drawdown': compute_max_drawdown(returns),
        'calmar_ratio': compute_calmar_ratio(returns, annualization_factor),
        'total_return': np.exp(np.sum(returns)) - 1,
        'n_periods': len(returns)
    }
    
    return metrics


def compute_turnover_statistics(turnovers: np.ndarray) -> Dict[str, float]:
    """
    Compute turnover statistics.
    
    Parameters:
    -----------
    turnovers : np.ndarray
        Array of daily turnovers
        
    Returns:
    --------
    Dict[str, float]
        Dictionary of turnover statistics
    """
    stats = {
        'mean_turnover': np.mean(turnovers),
        'median_turnover': np.median(turnovers),
        'std_turnover': np.std(turnovers),
        'max_turnover': np.max(turnovers),
        'min_turnover': np.min(turnovers),
        'total_turnover': np.sum(turnovers)
    }
    
    return stats


def compute_weight_statistics(weights_df: pd.DataFrame) -> Dict[str, float]:
    """
    Compute portfolio weight statistics.
    
    Parameters:
    -----------
    weights_df : pd.DataFrame
        DataFrame with portfolio weights over time
        
    Returns:
    --------
    Dict[str, float]
        Dictionary of weight statistics
    """
    stats = {
        'mean_concentration': np.mean(np.max(weights_df.values, axis=1)),
        'mean_effective_positions': np.mean(np.sum(weights_df.values >= 0.01, axis=1)),
        'weight_stability': np.mean(np.std(weights_df.values, axis=0))
    }
    
    return stats


def compute_regime_statistics(regimes: np.ndarray) -> Dict[str, float]:
    """
    Compute regime statistics.
    
    Parameters:
    -----------
    regimes : np.ndarray
        Array of dominant regime indices over time
        
    Returns:
    --------
    Dict[str, float]
        Dictionary of regime statistics
    """
    unique_regimes, counts = np.unique(regimes, return_counts=True)
    
    stats = {
        'n_unique_regimes': len(unique_regimes),
        'most_common_regime': unique_regimes[np.argmax(counts)],
        'regime_persistence': np.mean(regimes[1:] == regimes[:-1])
    }
    
    return stats


def print_performance_summary(
    metrics: Dict[str, float],
    turnover_stats: Dict[str, float],
    weight_stats: Dict[str, float],
    regime_stats: Dict[str, float] = None
):
    """
    Print formatted performance summary.
    
    Parameters:
    -----------
    metrics : Dict[str, float]
        Performance metrics
    turnover_stats : Dict[str, float]
        Turnover statistics
    weight_stats : Dict[str, float]
        Weight statistics
    regime_stats : Dict[str, float], optional
        Regime statistics
    """
    print("\n" + "="*60)
    print("PERFORMANCE SUMMARY")
    print("="*60)
    
    print("\nReturn Metrics:")
    print(f"  Annualized Return:     {metrics['annualized_return']*100:>8.2f}%")
    print(f"  Total Return:          {metrics['total_return']*100:>8.2f}%")
    print(f"  Annualized Volatility: {metrics['annualized_volatility']*100:>8.2f}%")
    
    print("\nRisk-Adjusted Metrics:")
    print(f"  Sharpe Ratio:          {metrics['sharpe_ratio']:>8.2f}")
    print(f"  Sortino Ratio:         {metrics['sortino_ratio']:>8.2f}")
    print(f"  Calmar Ratio:          {metrics['calmar_ratio']:>8.2f}")
    print(f"  Max Drawdown:          {metrics['max_drawdown']*100:>8.2f}%")
    
    print("\nTurnover Statistics:")
    print(f"  Mean Daily Turnover:   {turnover_stats['mean_turnover']*100:>8.4f}%")
    print(f"  Median Turnover:       {turnover_stats['median_turnover']*100:>8.4f}%")
    print(f"  Max Turnover:          {turnover_stats['max_turnover']*100:>8.2f}%")
    print(f"  Total Turnover:        {turnover_stats['total_turnover']:>8.2f}")
    
    print("\nPortfolio Characteristics:")
    print(f"  Mean Concentration:    {weight_stats['mean_concentration']*100:>8.2f}%")
    print(f"  Mean Effective Pos:    {weight_stats['mean_effective_positions']:>8.2f}")
    print(f"  Weight Stability:      {weight_stats['weight_stability']:>8.4f}")
    
    if regime_stats:
        print("\nRegime Statistics:")
        print(f"  Unique Regimes:        {regime_stats['n_unique_regimes']:>8.0f}")
        print(f"  Most Common Regime:    {regime_stats['most_common_regime']:>8.0f}")
        print(f"  Regime Persistence:    {regime_stats['regime_persistence']*100:>8.2f}%")
    
    print("\n" + "="*60)
