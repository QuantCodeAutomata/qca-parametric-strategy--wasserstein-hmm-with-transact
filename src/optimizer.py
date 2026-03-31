"""
Transaction-cost-aware Mean-Variance Optimization module.
"""

import numpy as np
import cvxpy as cp
from typing import Optional


def optimize_portfolio(
    expected_returns: np.ndarray,
    covariance_matrix: np.ndarray,
    previous_weights: np.ndarray,
    risk_aversion: float = 1.0,
    turnover_penalty: float = 0.5,
    max_weight: float = 0.5,
    min_weight: float = 0.0
) -> np.ndarray:
    """
    Solve transaction-cost-aware Mean-Variance Optimization problem.
    
    The optimization problem is:
    
    max_w: mu' * w - gamma * w' * Sigma * w - tau * ||w - w_prev||_1
    s.t.:  sum(w) = 1
           w >= 0
           w <= w_max
    
    Parameters:
    -----------
    expected_returns : np.ndarray
        Expected returns vector (n_assets,)
    covariance_matrix : np.ndarray
        Covariance matrix (n_assets, n_assets)
    previous_weights : np.ndarray
        Previous portfolio weights (n_assets,)
    risk_aversion : float
        Risk aversion parameter (gamma)
    turnover_penalty : float
        Turnover penalty parameter (tau)
    max_weight : float
        Maximum weight per asset
    min_weight : float
        Minimum weight per asset (default: 0 for long-only)
        
    Returns:
    --------
    np.ndarray
        Optimal portfolio weights (n_assets,)
    """
    n_assets = len(expected_returns)
    
    # Define optimization variable
    w = cp.Variable(n_assets)
    
    # Add small regularization to covariance for numerical stability
    cov_reg = covariance_matrix + np.eye(n_assets) * 1e-6
    
    # Objective function components
    # Return term: mu' * w
    return_term = expected_returns @ w
    
    # Risk term: gamma * w' * Sigma * w
    risk_term = risk_aversion * cp.quad_form(w, cov_reg)
    
    # Turnover term: tau * ||w - w_prev||_1
    turnover_term = turnover_penalty * cp.norm1(w - previous_weights)
    
    # Total objective (maximize)
    objective = cp.Maximize(return_term - risk_term - turnover_term)
    
    # Constraints
    constraints = [
        cp.sum(w) == 1,           # Fully invested
        w >= min_weight,          # Long-only (or minimum weight)
        w <= max_weight           # Maximum weight per asset
    ]
    
    # Solve optimization problem
    problem = cp.Problem(objective, constraints)
    
    try:
        problem.solve(solver=cp.ECOS)
        
        if problem.status not in ['optimal', 'optimal_inaccurate']:
            # If optimization fails, return previous weights
            return previous_weights
        
        optimal_weights = w.value
        
        # Ensure weights are valid
        if optimal_weights is None or np.any(np.isnan(optimal_weights)):
            return previous_weights
        
        # Normalize to ensure sum = 1 (numerical precision)
        optimal_weights = optimal_weights / np.sum(optimal_weights)
        
        # Clip to valid range
        optimal_weights = np.clip(optimal_weights, min_weight, max_weight)
        
        # Renormalize after clipping
        optimal_weights = optimal_weights / np.sum(optimal_weights)
        
        return optimal_weights
        
    except Exception as e:
        # If solver fails, return previous weights
        return previous_weights


def compute_turnover(
    current_weights: np.ndarray,
    previous_weights_post_return: np.ndarray
) -> float:
    """
    Compute portfolio turnover.
    
    Turnover is defined as:
    TO_t = 0.5 * ||w_t - w_{t-1,post}||_1
    
    where w_{t-1,post} is the previous day's weight vector adjusted for price changes.
    
    Parameters:
    -----------
    current_weights : np.ndarray
        Current portfolio weights
    previous_weights_post_return : np.ndarray
        Previous weights adjusted for returns
        
    Returns:
    --------
    float
        Turnover value
    """
    turnover = 0.5 * np.sum(np.abs(current_weights - previous_weights_post_return))
    return turnover


def adjust_weights_for_returns(
    weights: np.ndarray,
    returns: np.ndarray
) -> np.ndarray:
    """
    Adjust portfolio weights for asset returns (drift).
    
    After returns are realized, weights drift:
    w_{t,post} = w_t * (1 + r_t) / sum(w_t * (1 + r_t))
    
    For log returns, we use: (1 + r) ≈ exp(r)
    
    Parameters:
    -----------
    weights : np.ndarray
        Portfolio weights before returns
    returns : np.ndarray
        Realized log returns
        
    Returns:
    --------
    np.ndarray
        Adjusted weights after returns
    """
    # Convert log returns to simple returns
    simple_returns = np.exp(returns) - 1
    
    # Adjust weights
    weights_post = weights * (1 + simple_returns)
    
    # Normalize
    weights_post = weights_post / np.sum(weights_post)
    
    return weights_post


def compute_effective_positions(weights: np.ndarray, threshold: float = 0.01) -> int:
    """
    Compute the effective number of positions in the portfolio.
    
    Counts assets with weight above threshold.
    
    Parameters:
    -----------
    weights : np.ndarray
        Portfolio weights
    threshold : float
        Minimum weight to count as a position
        
    Returns:
    --------
    int
        Number of effective positions
    """
    return np.sum(weights >= threshold)


def compute_portfolio_return(
    weights: np.ndarray,
    returns: np.ndarray
) -> float:
    """
    Compute portfolio log return.
    
    r_p,t = w_t' * r_t
    
    Parameters:
    -----------
    weights : np.ndarray
        Portfolio weights
    returns : np.ndarray
        Asset log returns
        
    Returns:
    --------
    float
        Portfolio log return
    """
    return np.dot(weights, returns)


def validate_weights(weights: np.ndarray, tolerance: float = 1e-4) -> bool:
    """
    Validate that portfolio weights satisfy constraints.
    
    Parameters:
    -----------
    weights : np.ndarray
        Portfolio weights to validate
    tolerance : float
        Numerical tolerance for constraint checks
        
    Returns:
    --------
    bool
        True if weights are valid
    """
    # Check sum to 1
    if not np.isclose(np.sum(weights), 1.0, atol=tolerance):
        return False
    
    # Check non-negative
    if np.any(weights < -tolerance):
        return False
    
    # Check no NaN or inf
    if np.any(np.isnan(weights)) or np.any(np.isinf(weights)):
        return False
    
    return True
