"""
Tests for HMM model and model selection module.
"""

import pytest
import numpy as np
from src.hmm_model import (
    GaussianHMMModel,
    compute_complexity_penalty,
    select_optimal_states
)


def test_gaussian_hmm_initialization():
    """Test HMM model initialization."""
    model = GaussianHMMModel(n_components=3, random_state=42)
    
    assert model.n_components == 3
    assert model.random_state == 42
    assert model.model is None


def test_gaussian_hmm_fit():
    """Test HMM model fitting."""
    np.random.seed(42)
    X = np.random.randn(100, 5)
    
    model = GaussianHMMModel(n_components=2, random_state=42)
    model.fit(X, n_restarts=2)
    
    assert model.model is not None


def test_gaussian_hmm_predict_proba():
    """Test posterior probability prediction."""
    np.random.seed(42)
    X = np.random.randn(100, 5)
    
    model = GaussianHMMModel(n_components=3, random_state=42)
    model.fit(X, n_restarts=2)
    
    probs = model.predict_proba(X)
    
    # Check shape
    assert probs.shape == (100, 3)
    
    # Check probabilities sum to 1
    assert np.allclose(probs.sum(axis=1), 1.0)
    
    # Check probabilities are non-negative
    assert np.all(probs >= 0)


def test_gaussian_hmm_predict_next_state_proba():
    """Test one-step-ahead predictive probabilities."""
    np.random.seed(42)
    X = np.random.randn(100, 5)
    
    model = GaussianHMMModel(n_components=2, random_state=42)
    model.fit(X, n_restarts=2)
    
    pred_probs = model.predict_next_state_proba(X)
    
    # Check shape
    assert pred_probs.shape == (100, 2)
    
    # Check probabilities sum to 1
    assert np.allclose(pred_probs.sum(axis=1), 1.0)


def test_gaussian_hmm_get_parameters():
    """Test parameter extraction."""
    np.random.seed(42)
    X = np.random.randn(100, 5)
    
    model = GaussianHMMModel(n_components=2, random_state=42)
    model.fit(X, n_restarts=2)
    
    params = model.get_parameters()
    
    # Check keys
    assert 'means' in params
    assert 'covariances' in params
    
    # Check shapes
    assert params['means'].shape == (2, 5)
    assert params['covariances'].shape == (2, 5, 5)


def test_gaussian_hmm_score():
    """Test log-likelihood scoring."""
    np.random.seed(42)
    X = np.random.randn(100, 5)
    
    model = GaussianHMMModel(n_components=2, random_state=42)
    model.fit(X, n_restarts=2)
    
    score = model.score(X)
    
    # Score should be finite
    assert np.isfinite(score)


def test_compute_complexity_penalty():
    """Test complexity penalty calculation."""
    # For K=2, d=5: penalty = 0.01 * 2 * (5 + 5*6/2) = 0.01 * 2 * 20 = 0.4
    penalty = compute_complexity_penalty(n_components=2, feature_dim=5, lambda_k=0.01)
    
    expected = 0.01 * 2 * (5 + 5 * 6 / 2)
    assert np.isclose(penalty, expected)


def test_compute_complexity_penalty_increases_with_K():
    """Test that penalty increases with number of components."""
    penalty_2 = compute_complexity_penalty(2, 5, 0.01)
    penalty_5 = compute_complexity_penalty(5, 5, 0.01)
    
    assert penalty_5 > penalty_2


def test_select_optimal_states():
    """Test model order selection."""
    np.random.seed(42)
    X_train = np.random.randn(200, 5)
    X_val = np.random.randn(50, 5)
    
    candidate_states = [2, 3]
    
    optimal_K, scores = select_optimal_states(
        X_train,
        X_val,
        candidate_states,
        lambda_k=0.01,
        n_restarts=2,
        random_state=42
    )
    
    # Check optimal K is in candidates
    assert optimal_K in candidate_states
    
    # Check scores computed for all candidates
    assert len(scores) == len(candidate_states)
    
    # Check optimal K has highest score
    assert scores[optimal_K] == max(scores.values())


def test_hmm_different_n_components():
    """Test HMM with different numbers of components."""
    np.random.seed(42)
    X = np.random.randn(100, 5)
    
    for K in [2, 3, 4]:
        model = GaussianHMMModel(n_components=K, random_state=42)
        model.fit(X, n_restarts=2)
        
        params = model.get_parameters()
        assert params['means'].shape[0] == K
        assert params['covariances'].shape[0] == K


def test_hmm_reproducibility():
    """Test that HMM fitting is reproducible with same random seed."""
    np.random.seed(42)
    X = np.random.randn(100, 5)
    
    model1 = GaussianHMMModel(n_components=2, random_state=42)
    model1.fit(X, n_restarts=1)
    score1 = model1.score(X)
    
    model2 = GaussianHMMModel(n_components=2, random_state=42)
    model2.fit(X, n_restarts=1)
    score2 = model2.score(X)
    
    # Scores should be similar (not necessarily identical due to EM)
    assert np.abs(score1 - score2) < 1.0


def test_hmm_with_insufficient_data():
    """Test HMM behavior with very little data."""
    np.random.seed(42)
    X = np.random.randn(10, 5)
    
    model = GaussianHMMModel(n_components=2, random_state=42)
    
    # Should still fit (though may not be reliable)
    model.fit(X, n_restarts=1)
    assert model.model is not None
