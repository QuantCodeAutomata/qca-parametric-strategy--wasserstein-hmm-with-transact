"""
Hidden Markov Model fitting and model-order selection module.
"""

import numpy as np
import pandas as pd
from hmmlearn import hmm
from typing import Tuple, List, Dict
from sklearn.preprocessing import StandardScaler


class GaussianHMMModel:
    """
    Gaussian Hidden Markov Model wrapper for regime detection.
    """
    
    def __init__(
        self,
        n_components: int,
        n_iter: int = 100,
        tol: float = 1e-4,
        random_state: int = 42,
        covariance_type: str = 'full',
        reg_covar: float = 1e-6
    ):
        """
        Initialize Gaussian HMM.
        
        Parameters:
        -----------
        n_components : int
            Number of hidden states
        n_iter : int
            Maximum number of EM iterations
        tol : float
            Convergence tolerance
        random_state : int
            Random seed for reproducibility
        covariance_type : str
            Type of covariance matrix ('full', 'diag', 'spherical', 'tied')
        reg_covar : float
            Regularization added to diagonal of covariance matrices
        """
        self.n_components = n_components
        self.n_iter = n_iter
        self.tol = tol
        self.random_state = random_state
        self.covariance_type = covariance_type
        self.reg_covar = reg_covar
        self.model = None
        self.scaler = StandardScaler()
        
    def fit(self, X: np.ndarray, n_restarts: int = 10) -> 'GaussianHMMModel':
        """
        Fit HMM using EM algorithm with multiple random restarts.
        
        Parameters:
        -----------
        X : np.ndarray
            Feature matrix of shape (n_samples, n_features)
        n_restarts : int
            Number of random restarts to find best model
            
        Returns:
        --------
        self
        """
        # Standardize features for numerical stability
        X_scaled = self.scaler.fit_transform(X)
        
        best_model = None
        best_score = -np.inf
        
        # Try multiple random initializations
        for i in range(n_restarts):
            model = hmm.GaussianHMM(
                n_components=self.n_components,
                covariance_type=self.covariance_type,
                n_iter=self.n_iter,
                tol=self.tol,
                random_state=self.random_state + i,
                params='stmc',
                init_params='stmc'
            )
            
            # Add regularization to covariance
            model.covars_prior = self.reg_covar
            
            try:
                model.fit(X_scaled)
                score = model.score(X_scaled)
                
                if score > best_score:
                    best_score = score
                    best_model = model
            except Exception as e:
                # Skip failed fits
                continue
        
        if best_model is None:
            raise ValueError(f"Failed to fit HMM with {self.n_components} components")
        
        self.model = best_model
        return self
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Compute posterior state probabilities for each observation.
        
        Parameters:
        -----------
        X : np.ndarray
            Feature matrix of shape (n_samples, n_features)
            
        Returns:
        --------
        np.ndarray
            Posterior probabilities of shape (n_samples, n_components)
        """
        if self.model is None:
            raise ValueError("Model must be fitted before prediction")
        
        X_scaled = self.scaler.transform(X)
        posteriors = self.model.predict_proba(X_scaled)
        
        return posteriors
    
    def predict_next_state_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Compute one-step-ahead predictive state probabilities.
        
        For time t, compute p(z_t | F_{t-1}) using data up to t-1.
        
        Parameters:
        -----------
        X : np.ndarray
            Feature matrix of shape (n_samples, n_features)
            
        Returns:
        --------
        np.ndarray
            Predictive probabilities of shape (n_samples, n_components)
        """
        if self.model is None:
            raise ValueError("Model must be fitted before prediction")
        
        X_scaled = self.scaler.transform(X)
        
        # Get posterior probabilities for all time steps
        posteriors = self.model.predict_proba(X_scaled)
        
        # Compute one-step-ahead predictions
        # p(z_t | F_{t-1}) = sum_k p(z_{t-1}=k | F_{t-1}) * p(z_t | z_{t-1}=k)
        n_samples = len(X_scaled)
        predictive_probs = np.zeros((n_samples, self.n_components))
        
        # For first observation, use stationary distribution
        predictive_probs[0] = self.model.startprob_
        
        # For subsequent observations
        for t in range(1, n_samples):
            # Use posterior from t-1 and transition matrix
            predictive_probs[t] = posteriors[t-1] @ self.model.transmat_
        
        return predictive_probs
    
    def get_parameters(self) -> Dict[str, np.ndarray]:
        """
        Extract model parameters (means and covariances).
        
        Returns:
        --------
        Dict[str, np.ndarray]
            Dictionary containing 'means' and 'covariances'
        """
        if self.model is None:
            raise ValueError("Model must be fitted before extracting parameters")
        
        # Transform means back to original scale
        means_scaled = self.model.means_
        means = self.scaler.inverse_transform(means_scaled)
        
        # Transform covariances back to original scale
        scale = self.scaler.scale_
        covariances = np.zeros_like(self.model.covars_)
        
        for k in range(self.n_components):
            if self.covariance_type == 'full':
                # Scale covariance matrix
                scale_matrix = np.outer(scale, scale)
                covariances[k] = self.model.covars_[k] * scale_matrix
            else:
                covariances[k] = self.model.covars_[k] * (scale ** 2)
        
        return {
            'means': means,
            'covariances': covariances
        }
    
    def score(self, X: np.ndarray) -> float:
        """
        Compute log-likelihood of observations.
        
        Parameters:
        -----------
        X : np.ndarray
            Feature matrix
            
        Returns:
        --------
        float
            Log-likelihood
        """
        if self.model is None:
            raise ValueError("Model must be fitted before scoring")
        
        X_scaled = self.scaler.transform(X)
        return self.model.score(X_scaled)


def compute_predictive_log_likelihood(
    model: GaussianHMMModel,
    X_val: np.ndarray
) -> float:
    """
    Compute one-step-ahead predictive log-likelihood on validation set.
    
    Parameters:
    -----------
    model : GaussianHMMModel
        Fitted HMM model
    X_val : np.ndarray
        Validation feature matrix
        
    Returns:
    --------
    float
        Predictive log-likelihood
    """
    # Get predictive probabilities
    pred_probs = model.predict_next_state_proba(X_val)
    
    # Get model parameters
    params = model.get_parameters()
    means = params['means']
    covariances = params['covariances']
    
    # Compute log-likelihood for each observation
    log_likelihoods = []
    
    for t in range(len(X_val)):
        # Compute likelihood under each component
        component_lls = []
        
        for k in range(model.n_components):
            # Multivariate normal log-likelihood
            diff = X_val[t] - means[k]
            cov = covariances[k]
            
            # Add regularization for numerical stability
            cov_reg = cov + np.eye(len(cov)) * 1e-6
            
            try:
                # Log-likelihood of observation under component k
                sign, logdet = np.linalg.slogdet(cov_reg)
                if sign <= 0:
                    component_lls.append(-np.inf)
                else:
                    inv_cov = np.linalg.inv(cov_reg)
                    ll = -0.5 * (len(X_val[t]) * np.log(2 * np.pi) + logdet + diff @ inv_cov @ diff)
                    component_lls.append(ll)
            except:
                component_lls.append(-np.inf)
        
        component_lls = np.array(component_lls)
        
        # Weight by predictive probabilities
        # log p(x_t | F_{t-1}) = log sum_k p(z_t=k | F_{t-1}) * p(x_t | z_t=k)
        max_ll = np.max(component_lls)
        weighted_ll = max_ll + np.log(np.sum(pred_probs[t] * np.exp(component_lls - max_ll)))
        
        log_likelihoods.append(weighted_ll)
    
    # Return average predictive log-likelihood
    return np.mean(log_likelihoods)


def compute_complexity_penalty(
    n_components: int,
    feature_dim: int,
    lambda_k: float = 0.01
) -> float:
    """
    Compute AIC-like complexity penalty for model selection.
    
    Parameters:
    -----------
    n_components : int
        Number of HMM states
    feature_dim : int
        Dimension of feature vector
    lambda_k : float
        Penalty coefficient
        
    Returns:
    --------
    float
        Complexity penalty
    """
    # Number of parameters: K * (d + d*(d+1)/2)
    # d mean parameters + d*(d+1)/2 covariance parameters per state
    n_params = n_components * (feature_dim + feature_dim * (feature_dim + 1) / 2)
    
    penalty = lambda_k * n_params
    
    return penalty


def select_optimal_states(
    X_train: np.ndarray,
    X_val: np.ndarray,
    candidate_states: List[int],
    lambda_k: float = 0.01,
    n_restarts: int = 10,
    random_state: int = 42
) -> Tuple[int, Dict[int, float]]:
    """
    Select optimal number of HMM states using predictive log-likelihood.
    
    Parameters:
    -----------
    X_train : np.ndarray
        Training feature matrix
    X_val : np.ndarray
        Validation feature matrix
    candidate_states : List[int]
        List of candidate number of states to evaluate
    lambda_k : float
        Complexity penalty coefficient
    n_restarts : int
        Number of random restarts for HMM fitting
    random_state : int
        Random seed
        
    Returns:
    --------
    Tuple[int, Dict[int, float]]
        Optimal number of states and dictionary of scores for each candidate
    """
    feature_dim = X_train.shape[1]
    scores = {}
    
    for K in candidate_states:
        # Fit HMM with K states
        model = GaussianHMMModel(
            n_components=K,
            random_state=random_state
        )
        
        try:
            model.fit(X_train, n_restarts=n_restarts)
            
            # Compute predictive log-likelihood on validation set
            pred_ll = compute_predictive_log_likelihood(model, X_val)
            
            # Compute complexity penalty
            penalty = compute_complexity_penalty(K, feature_dim, lambda_k)
            
            # Adjusted score
            score = pred_ll - penalty
            scores[K] = score
            
        except Exception as e:
            # If fitting fails, assign very low score
            scores[K] = -np.inf
    
    # Select K with highest score
    optimal_K = max(scores, key=scores.get)
    
    return optimal_K, scores
