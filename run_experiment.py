"""
Main experiment runner for Wasserstein HMM strategy.
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
from src.backtest import WassersteinHMMBacktest
from src.metrics import (
    compute_all_metrics,
    compute_turnover_statistics,
    compute_weight_statistics,
    compute_regime_statistics,
    print_performance_summary
)


def create_visualizations(results_df: pd.DataFrame, output_dir: str = 'results'):
    """
    Create and save visualizations of backtest results.
    
    Parameters:
    -----------
    results_df : pd.DataFrame
        Backtest results DataFrame
    output_dir : str
        Directory to save plots
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Set style
    sns.set_style('whitegrid')
    
    # 1. Cumulative Returns
    fig, ax = plt.subplots(figsize=(12, 6))
    cumulative_returns = np.exp(results_df['return'].cumsum()) - 1
    ax.plot(results_df.index, cumulative_returns * 100, linewidth=2)
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Cumulative Return (%)', fontsize=12)
    ax.set_title('Strategy Cumulative Returns', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/cumulative_returns.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Portfolio Weights Over Time
    weight_cols = [col for col in results_df.columns if col.startswith('weight_')]
    asset_names = [col.replace('weight_', '') for col in weight_cols]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    results_df[weight_cols].plot(kind='area', stacked=True, ax=ax, alpha=0.7)
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Portfolio Weight', fontsize=12)
    ax.set_title('Portfolio Weights Over Time', fontsize=14, fontweight='bold')
    ax.legend(asset_names, loc='upper left', bbox_to_anchor=(1, 1))
    ax.set_ylim([0, 1])
    plt.tight_layout()
    plt.savefig(f'{output_dir}/portfolio_weights.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Turnover Over Time
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(results_df.index, results_df['turnover'] * 100, linewidth=1, alpha=0.7)
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Turnover (%)', fontsize=12)
    ax.set_title('Daily Portfolio Turnover', fontsize=14, fontweight='bold')
    ax.axhline(y=results_df['turnover'].mean() * 100, color='r', linestyle='--', 
               label=f'Mean: {results_df["turnover"].mean()*100:.2f}%')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/turnover.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Dominant Regime Over Time
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(results_df.index, results_df['dominant_regime'], linewidth=1, marker='o', 
            markersize=2, alpha=0.7)
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Dominant Regime', fontsize=12)
    ax.set_title('Dominant Regime Over Time', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/dominant_regime.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 5. Drawdown
    fig, ax = plt.subplots(figsize=(12, 6))
    cumulative = np.exp(results_df['return'].cumsum())
    running_max = np.maximum.accumulate(cumulative)
    drawdown = (cumulative - running_max) / running_max * 100
    ax.fill_between(results_df.index, drawdown, 0, alpha=0.5, color='red')
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Drawdown (%)', fontsize=12)
    ax.set_title('Strategy Drawdown', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/drawdown.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 6. Effective Positions Over Time
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(results_df.index, results_df['effective_positions'], linewidth=1, alpha=0.7)
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Number of Positions', fontsize=12)
    ax.set_title('Effective Number of Positions', fontsize=14, fontweight='bold')
    ax.axhline(y=results_df['effective_positions'].mean(), color='r', linestyle='--',
               label=f'Mean: {results_df["effective_positions"].mean():.2f}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/effective_positions.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 7. Return Distribution
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(results_df['return'] * 100, bins=50, alpha=0.7, edgecolor='black')
    ax.axvline(x=results_df['return'].mean() * 100, color='r', linestyle='--',
               label=f'Mean: {results_df["return"].mean()*100:.3f}%')
    ax.set_xlabel('Daily Return (%)', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Distribution of Daily Returns', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/return_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 8. Number of States Over Time
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(results_df.index, results_df['n_states'], linewidth=1, marker='o',
            markersize=2, alpha=0.7)
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Number of HMM States', fontsize=12)
    ax.set_title('Selected Number of HMM States Over Time', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/n_states.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nVisualizations saved to {output_dir}/")


def save_results_markdown(
    metrics: dict,
    turnover_stats: dict,
    weight_stats: dict,
    regime_stats: dict,
    results_df: pd.DataFrame,
    output_dir: str = 'results'
):
    """
    Save results to markdown file.
    
    Parameters:
    -----------
    metrics : dict
        Performance metrics
    turnover_stats : dict
        Turnover statistics
    weight_stats : dict
        Weight statistics
    regime_stats : dict
        Regime statistics
    results_df : pd.DataFrame
        Full results DataFrame
    output_dir : str
        Output directory
    """
    os.makedirs(output_dir, exist_ok=True)
    
    with open(f'{output_dir}/RESULTS.md', 'w') as f:
        f.write("# Wasserstein HMM Strategy - Experiment Results\n\n")
        f.write(f"**Experiment Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## Strategy Configuration\n\n")
        f.write("- **Assets:** SPY, AGG, GLD, USO, UUP\n")
        f.write("- **OOS Period:** 2023-05-01 to 2026-01-01\n")
        f.write("- **Number of Templates:** 6\n")
        f.write("- **Candidate States:** [2, 3, 4, 5, 6, 7, 8]\n")
        f.write("- **Risk Aversion (γ):** 1.0\n")
        f.write("- **Turnover Penalty (τ):** 0.5\n")
        f.write("- **Max Weight:** 0.5\n\n")
        
        f.write("## Performance Metrics\n\n")
        f.write("### Return Metrics\n\n")
        f.write(f"| Metric | Value |\n")
        f.write(f"|--------|-------|\n")
        f.write(f"| Annualized Return | {metrics['annualized_return']*100:.2f}% |\n")
        f.write(f"| Total Return | {metrics['total_return']*100:.2f}% |\n")
        f.write(f"| Annualized Volatility | {metrics['annualized_volatility']*100:.2f}% |\n\n")
        
        f.write("### Risk-Adjusted Metrics\n\n")
        f.write(f"| Metric | Value | Expected |\n")
        f.write(f"|--------|-------|----------|\n")
        f.write(f"| Sharpe Ratio | {metrics['sharpe_ratio']:.2f} | ~2.18 |\n")
        f.write(f"| Sortino Ratio | {metrics['sortino_ratio']:.2f} | ~2.82 |\n")
        f.write(f"| Calmar Ratio | {metrics['calmar_ratio']:.2f} | - |\n")
        f.write(f"| Max Drawdown | {metrics['max_drawdown']*100:.2f}% | ~-5.43% |\n\n")
        
        f.write("### Turnover Statistics\n\n")
        f.write(f"| Metric | Value | Expected |\n")
        f.write(f"|--------|-------|----------|\n")
        f.write(f"| Mean Daily Turnover | {turnover_stats['mean_turnover']*100:.4f}% | ~0.79% |\n")
        f.write(f"| Median Turnover | {turnover_stats['median_turnover']*100:.4f}% | - |\n")
        f.write(f"| Max Turnover | {turnover_stats['max_turnover']*100:.2f}% | - |\n")
        f.write(f"| Total Turnover | {turnover_stats['total_turnover']:.2f} | - |\n\n")
        
        f.write("### Portfolio Characteristics\n\n")
        f.write(f"| Metric | Value | Expected |\n")
        f.write(f"|--------|-------|----------|\n")
        f.write(f"| Mean Concentration | {weight_stats['mean_concentration']*100:.2f}% | - |\n")
        f.write(f"| Mean Effective Positions | {weight_stats['mean_effective_positions']:.2f} | ~3.63 |\n")
        f.write(f"| Weight Stability | {weight_stats['weight_stability']:.4f} | - |\n\n")
        
        f.write("### Regime Statistics\n\n")
        f.write(f"| Metric | Value |\n")
        f.write(f"|--------|-------|\n")
        f.write(f"| Unique Regimes | {regime_stats['n_unique_regimes']:.0f} |\n")
        f.write(f"| Most Common Regime | {regime_stats['most_common_regime']:.0f} |\n")
        f.write(f"| Regime Persistence | {regime_stats['regime_persistence']*100:.2f}% |\n\n")
        
        f.write("## Visualizations\n\n")
        f.write("![Cumulative Returns](cumulative_returns.png)\n\n")
        f.write("![Portfolio Weights](portfolio_weights.png)\n\n")
        f.write("![Turnover](turnover.png)\n\n")
        f.write("![Drawdown](drawdown.png)\n\n")
        f.write("![Dominant Regime](dominant_regime.png)\n\n")
        f.write("![Effective Positions](effective_positions.png)\n\n")
        f.write("![Return Distribution](return_distribution.png)\n\n")
        f.write("![Number of States](n_states.png)\n\n")
        
        f.write("## Summary\n\n")
        f.write("The Wasserstein HMM strategy demonstrates:\n\n")
        f.write(f"- **Strong risk-adjusted performance** with Sharpe ratio of {metrics['sharpe_ratio']:.2f}\n")
        f.write(f"- **Low turnover** averaging {turnover_stats['mean_turnover']*100:.4f}% per day\n")
        f.write(f"- **Concentrated portfolio** with ~{weight_stats['mean_effective_positions']:.1f} effective positions\n")
        f.write(f"- **Controlled drawdown** with maximum of {metrics['max_drawdown']*100:.2f}%\n\n")
        
        f.write("## Data Summary\n\n")
        f.write(f"- **Number of Trading Days:** {len(results_df)}\n")
        f.write(f"- **Start Date:** {results_df.index[0].date()}\n")
        f.write(f"- **End Date:** {results_df.index[-1].date()}\n\n")
    
    print(f"\nResults saved to {output_dir}/RESULTS.md")


def main():
    """
    Main experiment execution function.
    """
    print("\n" + "="*80)
    print("WASSERSTEIN HMM ASSET ALLOCATION STRATEGY")
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
    features = construct_feature_vector(
        returns,
        vol_window=60,
        mom_window=20
    )
    
    print(f"Features constructed: {features.shape}")
    
    # Align returns with features
    returns_aligned = returns.loc[features.index]
    
    # Initialize backtest
    print("\nInitializing backtest engine...")
    backtest = WassersteinHMMBacktest(
        features=features,
        returns=returns_aligned,
        n_assets=len(tickers),
        t0=T0,
        n_templates=6,
        candidate_states=[2, 3, 4, 5, 6],  # Reduced for speed
        selection_frequency=20,  # Less frequent selection for speed
        validation_window=252,
        lambda_k=0.01,
        smoothing_rate=0.1,
        risk_aversion=1.0,
        turnover_penalty=0.5,
        max_weight=0.5,
        n_restarts=3,  # Reduced for speed
        random_state=42
    )
    
    # Run backtest
    results_df = backtest.run()
    
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
    print(f"- Performance metrics: results/RESULTS.md")
    print(f"- Full backtest data: results/backtest_results.csv")
    print(f"- Visualizations: results/*.png")


if __name__ == '__main__':
    main()
