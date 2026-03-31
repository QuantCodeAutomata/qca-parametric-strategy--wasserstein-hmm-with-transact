# Parametric Strategy: Wasserstein HMM with Transaction-Cost-Aware MVO

This repository implements a parametric asset allocation strategy using a Gaussian Hidden Markov Model (HMM) for regime inference, combined with Wasserstein distance-based template tracking and transaction-cost-aware Mean-Variance Optimization (MVO).

## Overview

The strategy evaluates a parametric asset allocation approach with the following key features:

- **Gaussian Hidden Markov Model (HMM)** for regime inference
- **Predictive model-order selection** to determine the optimal number of states
- **Wasserstein distance-based template tracking** to ensure regime stability over time
- **Long-only, transaction-cost-aware Mean-Variance Optimization (MVO)** for daily portfolio weights

## Research Objectives

1. Implement the complete Wasserstein HMM asset allocation strategy
2. Generate daily time series of portfolio weights and returns for the out-of-sample period
3. Calculate performance metrics (Sharpe Ratio, Sortino Ratio, Max Drawdown)
4. Compute diagnostic metrics (daily turnover, effective number of positions, allocation stability)
5. Validate the strategy's superior risk-adjusted performance and stability

## Data Requirements

The experiment uses daily adjusted close prices for 5 asset class proxies from Yahoo Finance (2005-01-01 to 2026-01-01):

- **SPX (Equities)**: SPY
- **BOND (Fixed Income)**: AGG
- **GOLD (Commodities)**: GLD
- **OIL (Commodities)**: USO
- **USD (FX)**: UUP

## Methodology

The experiment is conducted as a daily backtest over an out-of-sample period starting from 2023-05-01:

1. **Data Preprocessing & Feature Engineering**: Compute daily log returns, 60-day rolling volatility, and 20-day rolling mean
2. **Initialization**: Set up 6 persistent templates using initial calibration window
3. **Daily Backtesting Loop**:
   - Model-order selection (weekly)
   - HMM fitting and prediction
   - Wasserstein template tracking
   - Template and moment updating
   - Portfolio optimization with transaction costs
   - Performance calculation and recording
4. **Post-Backtest Analysis**: Compute cumulative returns and performance metrics

## Key Parameters

- **Time Period**: 2005-01-01 to 2026-01-01, OOS start: 2023-05-01
- **Feature Windows**: 60 days (volatility), 20 days (mean)
- **HMM States**: K ∈ [2, 3, 4, 5, 6, 7, 8]
- **Templates**: G = 6, smoothing rate η = 0.1
- **MVO**: Risk aversion γ = 1.0, turnover penalty τ = 0.5, max weight = 0.5

## Expected Outcomes

- **Sharpe Ratio**: ~2.18 (annualized)
- **Sortino Ratio**: ~2.82 (annualized)
- **Maximum Drawdown**: ~-5.43%
- **Average Daily Turnover**: ~0.79%

## Repository Structure

```
.
├── src/
│   ├── __init__.py
│   ├── data_loader.py          # Data loading and preprocessing
│   ├── feature_engineering.py  # Feature computation
│   ├── hmm_model.py            # HMM fitting and model selection
│   ├── wasserstein.py          # Wasserstein distance and template tracking
│   ├── optimizer.py            # Transaction-cost-aware MVO
│   ├── backtest.py             # Main backtesting engine
│   └── metrics.py              # Performance metrics and analysis
├── tests/
│   ├── __init__.py
│   ├── test_data_loader.py
│   ├── test_feature_engineering.py
│   ├── test_hmm_model.py
│   ├── test_wasserstein.py
│   ├── test_optimizer.py
│   └── test_metrics.py
├── results/
│   └── RESULTS.md              # Experiment results and metrics
├── data/                       # Downloaded data cache
├── requirements.txt
├── run_experiment.py           # Main experiment runner
└── README.md
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

Run the full experiment:

```bash
python run_experiment.py
```

Run tests:

```bash
pytest tests/ -v
```

## Results

All results, including performance metrics and visualizations, are saved in the `results/` directory. See `results/RESULTS.md` for detailed outcomes.

## License

MIT License
