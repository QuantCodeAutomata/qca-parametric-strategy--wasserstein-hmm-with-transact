"""
Tests for Wasserstein distance and template tracking module.
"""

import pytest
import numpy as np
from src.wasserstein import (
    wasserstein_distance_gaussian,
    TemplateTracker
)


def test_wasserstein_distance_identical():
    """Test Wasserstein distance between identical distributions is zero."""
    mu = np.array([0.0, 0.0])
    cov = np.eye(2)
    
    dist = wasserstein_distance_gaussian(mu, cov, mu, cov)
    
    assert np.isclose(dist, 0.0, atol=1e-6)


def test_wasserstein_distance_different_means():
    """Test Wasserstein distance with different means."""
    mu1 = np.array([0.0, 0.0])
    mu2 = np.array([1.0, 0.0])
    cov = np.eye(2)
    
    dist = wasserstein_distance_gaussian(mu1, cov, mu2, cov)
    
    # Distance should be approximately 1.0 (Euclidean distance of means)
    assert dist > 0.9
    assert dist < 1.1


def test_wasserstein_distance_symmetric():
    """Test Wasserstein distance is symmetric."""
    mu1 = np.array([0.0, 0.0])
    mu2 = np.array([1.0, 1.0])
    cov1 = np.eye(2)
    cov2 = np.eye(2) * 2
    
    dist1 = wasserstein_distance_gaussian(mu1, cov1, mu2, cov2)
    dist2 = wasserstein_distance_gaussian(mu2, cov2, mu1, cov1)
    
    assert np.isclose(dist1, dist2)


def test_wasserstein_distance_positive():
    """Test Wasserstein distance is always non-negative."""
    np.random.seed(42)
    
    for _ in range(10):
        mu1 = np.random.randn(3)
        mu2 = np.random.randn(3)
        
        # Generate random positive definite covariances
        A1 = np.random.randn(3, 3)
        cov1 = A1 @ A1.T + np.eye(3) * 0.1
        
        A2 = np.random.randn(3, 3)
        cov2 = A2 @ A2.T + np.eye(3) * 0.1
        
        dist = wasserstein_distance_gaussian(mu1, cov1, mu2, cov2)
        
        assert dist >= 0


def test_template_tracker_initialization():
    """Test template tracker initialization."""
    n_templates = 6
    feature_dim = 15
    
    tracker = TemplateTracker(
        n_templates=n_templates,
        feature_dim=feature_dim,
        smoothing_rate=0.1
    )
    
    assert tracker.n_templates == n_templates
    assert tracker.feature_dim == feature_dim
    assert tracker.template_means is None


def test_template_tracker_initialize_templates():
    """Test template initialization."""
    n_templates = 3
    feature_dim = 6
    
    tracker = TemplateTracker(n_templates, feature_dim)
    
    means = np.random.randn(n_templates, feature_dim)
    covs = np.array([np.eye(feature_dim) for _ in range(n_templates)])
    
    tracker.initialize_templates(means, covs)
    
    assert tracker.template_means.shape == (n_templates, feature_dim)
    assert tracker.template_covariances.shape == (n_templates, feature_dim, feature_dim)


def test_template_tracker_assign_components():
    """Test component assignment to templates."""
    n_templates = 3
    n_components = 4
    feature_dim = 6
    
    tracker = TemplateTracker(n_templates, feature_dim)
    
    # Initialize templates
    template_means = np.random.randn(n_templates, feature_dim)
    template_covs = np.array([np.eye(feature_dim) for _ in range(n_templates)])
    tracker.initialize_templates(template_means, template_covs)
    
    # Create components
    component_means = np.random.randn(n_components, feature_dim)
    component_covs = np.array([np.eye(feature_dim) for _ in range(n_components)])
    
    assignments, distances = tracker.assign_components_to_templates(
        component_means,
        component_covs
    )
    
    # Check assignments
    assert assignments.shape == (n_components,)
    assert np.all(assignments >= 0)
    assert np.all(assignments < n_templates)
    
    # Check distances
    assert distances.shape == (n_components, n_templates)
    assert np.all(distances >= 0)


def test_template_tracker_aggregate_probabilities():
    """Test probability aggregation."""
    n_templates = 3
    n_components = 5
    feature_dim = 6
    
    tracker = TemplateTracker(n_templates, feature_dim)
    
    # Create component probabilities
    component_probs = np.array([0.1, 0.2, 0.3, 0.25, 0.15])
    
    # Create assignments
    assignments = np.array([0, 0, 1, 1, 2])
    
    template_probs = tracker.aggregate_probabilities(component_probs, assignments)
    
    # Check shape
    assert template_probs.shape == (n_templates,)
    
    # Check sum to 1
    assert np.isclose(template_probs.sum(), 1.0)
    
    # Check specific values
    assert np.isclose(template_probs[0], 0.1 + 0.2)
    assert np.isclose(template_probs[1], 0.3 + 0.25)
    assert np.isclose(template_probs[2], 0.15)


def test_template_tracker_update_templates():
    """Test template parameter updating."""
    n_templates = 2
    n_components = 3
    feature_dim = 4
    
    tracker = TemplateTracker(n_templates, feature_dim, smoothing_rate=0.5)
    
    # Initialize templates
    initial_means = np.zeros((n_templates, feature_dim))
    initial_covs = np.array([np.eye(feature_dim) for _ in range(n_templates)])
    tracker.initialize_templates(initial_means, initial_covs)
    
    # Create new components
    component_means = np.ones((n_components, feature_dim))
    component_covs = np.array([np.eye(feature_dim) * 2 for _ in range(n_components)])
    
    # Assign all to first template
    assignments = np.array([0, 0, 0])
    
    # Update
    tracker.update_templates(component_means, component_covs, assignments)
    
    # First template should have moved toward 1.0
    assert np.all(tracker.template_means[0] > 0)
    assert np.all(tracker.template_means[0] < 1.0)  # Due to smoothing


def test_template_tracker_compute_predictive_moments():
    """Test predictive moment computation."""
    n_templates = 2
    n_assets = 3
    feature_dim = 9  # 3 assets * 3 features
    
    tracker = TemplateTracker(n_templates, feature_dim)
    
    # Initialize templates
    means = np.random.randn(n_templates, feature_dim)
    covs = np.array([np.eye(feature_dim) for _ in range(n_templates)])
    tracker.initialize_templates(means, covs)
    
    # Template probabilities
    template_probs = np.array([0.6, 0.4])
    
    pred_mean, pred_cov = tracker.compute_predictive_moments(template_probs, n_assets)
    
    # Check shapes
    assert pred_mean.shape == (n_assets,)
    assert pred_cov.shape == (n_assets, n_assets)
    
    # Check covariance is symmetric
    assert np.allclose(pred_cov, pred_cov.T)


def test_template_tracker_get_dominant_template():
    """Test dominant template identification."""
    tracker = TemplateTracker(n_templates=4, feature_dim=6)
    
    template_probs = np.array([0.1, 0.5, 0.2, 0.2])
    
    dominant = tracker.get_dominant_template(template_probs)
    
    assert dominant == 1  # Index of maximum probability
