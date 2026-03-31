"""
Main backtesting engine for Wasserstein HMM strategy.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, List
from datetime import datetime

from src.hmm_model import GaussianHMMModel, select_optimal_states
from src.wasserstein import TemplateTracker
from src.optimizer import (
    optimize_portfolio,
    compute_turnover,
    adjust_weights_for_returns,
    compute_effective_positions,
    compute_portfolio_return
)


class WassersteinHMMBacktest:
    """
    Main backtesting engine for Wasserstein HMM asset allocation strategy.
    """
    
    def __init__(
        self,
        features: pd.DataFrame,
        returns: pd.DataFrame,
        n_assets: int,
        t0: str,
        n_templates: int = 6,
        candidate_states: List[int] = [2, 3, 4, 5, 6, 7, 8],
        selection_frequency: int = 5,
        validation_window: int = 252,
        lambda_k: float = 0.01,
        smoothing_rate: float = 0.1,
        risk_aversion: float = 1.0,
        turnover_penalty: float = 0.5,
        max_weight: float = 0.5,
        n_restarts: int = 10,
        random_state: int = 42
    ):
        """
        Initialize backtesting engine.
        
        Parameters:
        -----------
        features : pd.DataFrame
            Feature matrix with dates as index
        returns : pd.DataFrame
            Returns matrix with dates as index
        n_assets : int
            Number of assets
        t0 : str
            Out-of-sample start date
        n_templates : int
            Number of persistent templates
        candidate_states : List[int]
            Candidate number of HMM states
        selection_frequency : int
            Frequency of model-order selection (in days)
        validation_window : int
            Size of validation window for model selection
        lambda_k : float
            Complexity penalty coefficient
        smoothing_rate : float
            Template smoothing rate
        risk_aversion : float
            Risk aversion parameter
        turnover_penalty : float
            Turnover penalty parameter
        max_weight : float
            Maximum weight per asset
        n_restarts : int
            Number of HMM random restarts
        random_state : int
            Random seed
        """
        self.features = features
        self.returns = returns
        self.n_assets = n_assets
        self.t0 = pd.Timestamp(t0)
        self.n_templates = n_templates
        self.candidate_states = candidate_states
        self.selection_frequency = selection_frequency
        self.validation_window = validation_window
        self.lambda_k = lambda_k
        self.smoothing_rate = smoothing_rate
        self.risk_aversion = risk_aversion
        self.turnover_penalty = turnover_penalty
        self.max_weight = max_weight
        self.n_restarts = n_restarts
        self.random_state = random_state
        
        # Initialize template tracker
        self.template_tracker = TemplateTracker(
            n_templates=n_templates,
            feature_dim=features.shape[1],
            smoothing_rate=smoothing_rate
        )
        
        # Results storage
        self.results = {
            'dates': [],
            'weights': [],
            'returns': [],
            'turnovers': [],
            'dominant_regimes': [],
            'template_probs': [],
            'n_states': [],
            'effective_positions': []
        }
        
        self.current_K = None
        self.days_since_selection = 0
        
    def initialize_templates(self):
        """
        Initialize templates using data before t0.
        """
        # Get data before t0
        init_features = self.features[self.features.index < self.t0]
        
        if len(init_features) < self.validation_window:
            raise ValueError("Insufficient data for initialization")
        
        # Fit initial HMM with n_templates states
        init_model = GaussianHMMModel(
            n_components=self.n_templates,
            random_state=self.random_state
        )
        
        init_model.fit(init_features.values, n_restarts=self.n_restarts)
        
        # Extract parameters
        params = init_model.get_parameters()
        
        # Initialize templates
        self.template_tracker.initialize_templates(
            params['means'],
            params['covariances']
        )
        
        print(f"Templates initialized with {len(init_features)} days of data")
    
    def should_select_model_order(self, day_idx: int) -> bool:
        """
        Determine if model-order selection should be performed.
        
        Parameters:
        -----------
        day_idx : int
            Current day index in backtest
            
        Returns:
        --------
        bool
            True if selection should be performed
        """
        # Perform selection on first day and then every selection_frequency days
        if day_idx == 0:
            return True
        
        self.days_since_selection += 1
        
        if self.days_since_selection >= self.selection_frequency:
            self.days_since_selection = 0
            return True
        
        return False
    
    def run(self) -> pd.DataFrame:
        """
        Run the complete backtest.
        
        Returns:
        --------
        pd.DataFrame
            DataFrame with backtest results
        """
        print("\n" + "="*60)
        print("STARTING BACKTEST")
        print("="*60)
        
        # Initialize templates
        self.initialize_templates()
        
        # Get OOS dates
        oos_dates = self.features[self.features.index >= self.t0].index
        
        if len(oos_dates) == 0:
            raise ValueError(f"No data available after t0={self.t0}")
        
        print(f"\nOOS Period: {oos_dates[0]} to {oos_dates[-1]}")
        print(f"Number of OOS days: {len(oos_dates)}")
        
        # Initialize weights as equal-weight
        current_weights = np.ones(self.n_assets) / self.n_assets
        
        # Main backtest loop
        for day_idx, date in enumerate(oos_dates):
            if day_idx % 50 == 0:
                print(f"Processing day {day_idx+1}/{len(oos_dates)}: {date.date()}")
            
            # Get historical data up to t-1
            hist_features = self.features[self.features.index < date]
            
            # Model-order selection (if needed)
            if self.should_select_model_order(day_idx):
                # Define validation set (last validation_window days)
                if len(hist_features) >= self.validation_window:
                    val_start_idx = len(hist_features) - self.validation_window
                    X_train = hist_features.iloc[:val_start_idx].values
                    X_val = hist_features.iloc[val_start_idx:].values
                    
                    # Select optimal K
                    optimal_K, scores = select_optimal_states(
                        X_train,
                        X_val,
                        self.candidate_states,
                        lambda_k=self.lambda_k,
                        n_restarts=self.n_restarts,
                        random_state=self.random_state
                    )
                    
                    self.current_K = optimal_K
                    print(f"  Selected K={optimal_K} (scores: {scores})")
                else:
                    # Use default if insufficient data
                    self.current_K = self.n_templates
            
            # Fit HMM on full history
            hmm_model = GaussianHMMModel(
                n_components=self.current_K,
                random_state=self.random_state
            )
            
            hmm_model.fit(hist_features.values, n_restarts=self.n_restarts)
            
            # Get one-step-ahead predictive probabilities
            # We need to predict for date t using data up to t-1
            # The last observation in hist_features is t-1
            pred_probs = hmm_model.predict_next_state_proba(hist_features.values)
            component_probs_t = pred_probs[-1]  # Probabilities for time t
            
            # Get HMM parameters
            params = hmm_model.get_parameters()
            component_means = params['means']
            component_covariances = params['covariances']
            
            # Wasserstein template tracking
            assignments, distances = self.template_tracker.assign_components_to_templates(
                component_means,
                component_covariances
            )
            
            # Aggregate probabilities by template
            template_probs = self.template_tracker.aggregate_probabilities(
                component_probs_t,
                assignments
            )
            
            # Update templates
            self.template_tracker.update_templates(
                component_means,
                component_covariances,
                assignments
            )
            
            # Compute predictive moments for returns
            pred_mean, pred_cov = self.template_tracker.compute_predictive_moments(
                template_probs,
                self.n_assets
            )
            
            # Portfolio optimization
            new_weights = optimize_portfolio(
                expected_returns=pred_mean,
                covariance_matrix=pred_cov,
                previous_weights=current_weights,
                risk_aversion=self.risk_aversion,
                turnover_penalty=self.turnover_penalty,
                max_weight=self.max_weight
            )
            
            # Get realized returns for date t
            realized_returns = self.returns.loc[date].values
            
            # Compute portfolio return
            portfolio_return = compute_portfolio_return(new_weights, realized_returns)
            
            # Adjust previous weights for returns (for turnover calculation)
            weights_post_return = adjust_weights_for_returns(current_weights, realized_returns)
            
            # Compute turnover
            turnover = compute_turnover(new_weights, weights_post_return)
            
            # Compute effective positions
            eff_pos = compute_effective_positions(new_weights)
            
            # Get dominant regime
            dominant_regime = self.template_tracker.get_dominant_template(template_probs)
            
            # Store results
            self.results['dates'].append(date)
            self.results['weights'].append(new_weights.copy())
            self.results['returns'].append(portfolio_return)
            self.results['turnovers'].append(turnover)
            self.results['dominant_regimes'].append(dominant_regime)
            self.results['template_probs'].append(template_probs.copy())
            self.results['n_states'].append(self.current_K)
            self.results['effective_positions'].append(eff_pos)
            
            # Update current weights
            current_weights = new_weights
        
        print("\nBacktest completed!")
        
        # Convert results to DataFrame
        results_df = self._create_results_dataframe()
        
        return results_df
    
    def _create_results_dataframe(self) -> pd.DataFrame:
        """
        Create DataFrame from results dictionary.
        
        Returns:
        --------
        pd.DataFrame
            Results DataFrame
        """
        # Create main results DataFrame
        df = pd.DataFrame({
            'date': self.results['dates'],
            'return': self.results['returns'],
            'turnover': self.results['turnovers'],
            'dominant_regime': self.results['dominant_regimes'],
            'n_states': self.results['n_states'],
            'effective_positions': self.results['effective_positions']
        })
        
        df.set_index('date', inplace=True)
        
        # Add weights as separate columns
        weights_array = np.array(self.results['weights'])
        asset_names = self.returns.columns
        
        for i, asset in enumerate(asset_names):
            df[f'weight_{asset}'] = weights_array[:, i]
        
        # Add template probabilities
        template_probs_array = np.array(self.results['template_probs'])
        
        for g in range(self.n_templates):
            df[f'template_prob_{g}'] = template_probs_array[:, g]
        
        return df
