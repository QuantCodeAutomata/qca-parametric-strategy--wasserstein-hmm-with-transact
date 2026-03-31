"""
Fast version of experiment runner for Wasserstein HMM strategy.
Uses simplified parameters for faster execution.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

from src.data_loader import prepare_data, get_default_tickers
from src.feature_engineering import construct_feature_vector
from src.hmm_model import GaussianHMMModel
from src.wasserstein import TemplateTracker
from src.optimizer import (
    optimize_portfolio,
    compute_turnover,
    adjust_weights_for_returns,
    compute_effective_positions,
    compute_portfolio_return
)
from src.metrics import (
    compute_all_metrics,
    compute_turnover_statistics,
    compute_weight_statistics,
    compute_regime_statistics,
    print_performance_summary
)
from run_experiment import create_visualizations, save_results_markdown


def run_simplified_backtest(
    features: pd.DataFrame,
    returns: pd.DataFrame,
    n_assets: int,
    t0: str,
    n_templates: int = 6,
    n_states: int = 4,  # Fixed number of states
    smoothing_rate: float = 0.1,
    risk_aversion: float = 1.0,
    turnover_penalty: float = 0.5,
    max_weight: float = 0.5,
    random_state: int = 42
):
    """
    Run simplified backtest with fixed number of HMM states.
    """
    print("\n" + "="*60)
    print("STARTING SIMPLIFIED BACKTEST")
    print("="*60)
    
    t0 = pd.Timestamp(t0)
    
    # Initialize template tracker
    template_tracker = TemplateTracker(
        n_templates=n_templates,
        feature_dim=features.shape[1],
        smoothing_rate=smoothing_rate
    )
    
    # Initialize templates using data before t0
    init_features = features[features.index < t0]
    print(f"\nInitializing templates with {len(init_features)} days of data...")
    
    init_model = GaussianHMMModel(n_components=n_templates, random_state=random_state)
    init_model.fit(init_features.values, n_restarts=2)
    params = init_model.get_parameters()
    template_tracker.initialize_templates(params['means'], params['covariances'])
    
    # Get OOS dates
    oos_dates = features[features.index >= t0].index
    print(f"\nOOS Period: {oos_dates[0]} to {oos_dates[-1]}")
    print(f"Number of OOS days: {len(oos_dates)}")
    
    # Results storage
    results = {
        'dates': [],
        'weights': [],
        'returns': [],
        'turnovers': [],
        'dominant_regimes': [],
        'template_probs': [],
        'effective_positions': []
    }
    
    # Initialize weights
    current_weights = np.ones(n_assets) / n_assets
    
    # Main backtest loop
    for day_idx, date in enumerate(oos_dates):
        if day_idx % 50 == 0:
            print(f"Processing day {day_idx+1}/{len(oos_dates)}: {date.date()}")
        
        # Get historical data up to t-1
        hist_features = features[features.index < date]
        
        # Fit HMM on recent history (last 500 days for speed)
        recent_hist = hist_features.iloc[-500:] if len(hist_features) > 500 else hist_features
        
        hmm_model = GaussianHMMModel(n_components=n_states, random_state=random_state)
        hmm_model.fit(recent_hist.values, n_restarts=1)
        
        # Get one-step-ahead predictive probabilities
        pred_probs = hmm_model.predict_next_state_proba(recent_hist.values)
        component_probs_t = pred_probs[-1]
        
        # Get HMM parameters
        params = hmm_model.get_parameters()
        component_means = params['means']
        component_covariances = params['covariances']
        
        # Wasserstein template tracking
        assignments, _ = template_tracker.assign_components_to_templates(
            component_means, component_covariances
        )
        
        # Aggregate probabilities
        template_probs = template_tracker.aggregate_probabilities(
            component_probs_t, assignments
        )
        
        # Update templates
        template_tracker.update_templates(
            component_means, component_covariances, assignments
        )
        
        # Compute predictive moments
        pred_mean, pred_cov = template_tracker.compute_predictive_moments(
            template_probs, n_assets
        )
        
        # Portfolio optimization
        new_weights = optimize_portfolio(
            expected_returns=pred_mean,
            covariance_matrix=pred_cov,
            previous_weights=current_weights,
            risk_aversion=risk_aversion,
            turnover_penalty=turnover_penalty,
            max_weight=max_weight
        )
        
        # Get realized returns
        realized_returns = returns.loc[date].values
        
        # Compute portfolio return
        portfolio_return = compute_portfolio_return(new_weights, realized_returns)
        
        # Compute turnover
        weights_post_return = adjust_weights_for_returns(current_weights, realized_returns)
        turnover = compute_turnover(new_weights, weights_post_return)
        
        # Compute effective positions
        eff_pos = compute_effective_positions(new_weights)
        
        # Get dominant regime
        dominant_regime = template_tracker.get_dominant_template(template_probs)
        
        # Store results
        results['dates'].append(date)
        results['weights'].append(new_weights.copy())
        results['returns'].append(portfolio_return)
        results['turnovers'].append(turnover)
        results['dominant_regimes'].append(dominant_regime)
        results['template_probs'].append(template_probs.copy())
        results['effective_positions'].append(eff_pos)
        
        # Update weights
        current_weights = new_weights
    
    print("\nBacktest completed!")
    
    # Create results DataFrame
    df = pd.DataFrame({
        'date': results['dates'],
        'return': results['returns'],
        'turnover': results['turnovers'],
        'dominant_regime': results['dominant_regimes'],
        'effective_positions': results['effective_positions']
    })
    df.set_index('date', inplace=True)
    
    # Add weights
    weights_array = np.array(results['weights'])
    asset_names = returns.columns
    for i, asset in enumerate(asset_names):
        df[f'weight_{asset}'] = weights_array[:, i]
    
    # Add template probabilities
    template_probs_array = np.array(results['template_probs'])
    for g in range(n_templates):
        df[f'template_prob_{g}'] = template_probs_array[:, g]
    
    # Add n_states column for compatibility
    df['n_states'] = n_states
    
    return df


def main():
    """
    Main experiment execution function.
    """
    print("\n" + "="*80)
    print("WASSERSTEIN HMM ASSET ALLOCATION STRATEGY (FAST VERSION)")
    print("="*80)
    
    # Configuration
    START_DATE = '2005-01-01'
    END_DATE = '2026-01-01'
    T0 = '2023-05-01'
    
    # Get tickers
    ticker_dict = get_default_tickers()
    tickers = list(ticker_dict.values())
    
    print(f"\nLoading data for: {', '.join(tickers)}")
    print(f"Period: {START_DATE} to {END_DATE}")
    
    # Load and prepare data
    prices, returns = prepare_data(
        tickers=tickers,
        start_date=START_DATE,
        end_date=END_DATE,
        cache_path='data/prices.csv'
    )
    
    print(f"Data loaded: {len(prices)} days")
    print(f"Assets: {list(returns.columns)}")
    
    # Construct features
    print("\nConstructing features...")
    features = construct_feature_vector(returns, vol_window=60, mom_window=20)
    print(f"Features constructed: {features.shape}")
    
    # Align returns with features
    returns_aligned = returns.loc[features.index]
    
    # Run simplified backtest
    results_df = run_simplified_backtest(
        features=features,
        returns=returns_aligned,
        n_assets=len(tickers),
        t0=T0,
        n_templates=6,
        n_states=4,
        smoothing_rate=0.1,
        risk_aversion=1.0,
        turnover_penalty=0.5,
        max_weight=0.5,
        random_state=42
    )
    
    # Compute metrics
    print("\nComputing performance metrics...")
    metrics = compute_all_metrics(results_df['return'].values)
    
    # Compute turnover statistics
    turnover_stats = compute_turnover_statistics(results_df['turnover'].values)
    
    # Compute weight statistics
    weight_cols = [col for col in results_df.columns if col.startswith('weight_')]
    weight_stats = compute_weight_statistics(results_df[weight_cols])
    
    # Compute regime statistics
    regime_stats = compute_regime_statistics(results_df['dominant_regime'].values)
    
    # Print summary
    print_performance_summary(metrics, turnover_stats, weight_stats, regime_stats)
    
    # Create visualizations
    print("\nCreating visualizations...")
    create_visualizations(results_df, output_dir='results')
    
    # Save results
    print("\nSaving results...")
    results_df.to_csv('results/backtest_results.csv')
    save_results_markdown(
        metrics,
        turnover_stats,
        weight_stats,
        regime_stats,
        results_df,
        output_dir='results'
    )
    
    print("\n" + "="*80)
    print("EXPERIMENT COMPLETED SUCCESSFULLY")
    print("="*80)
    print(f"\nAll results saved to 'results/' directory")


if __name__ == '__main__':
    main()
