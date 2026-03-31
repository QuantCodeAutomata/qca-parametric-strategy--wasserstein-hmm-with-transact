"""
Wasserstein distance computation and template tracking module.
"""

import numpy as np
from typing import List, Dict, Tuple
from scipy.linalg import sqrtm


def wasserstein_distance_gaussian(
    mu1: np.ndarray,
    cov1: np.ndarray,
    mu2: np.ndarray,
    cov2: np.ndarray
) -> float:
    """
    Compute 2-Wasserstein distance between two Gaussian distributions.
    
    For two Gaussian distributions N(mu1, cov1) and N(mu2, cov2), the
    2-Wasserstein distance is:
    
    W_2^2 = ||mu1 - mu2||^2 + Tr(cov1 + cov2 - 2 * sqrt(cov1^{1/2} * cov2 * cov1^{1/2}))
    
    Parameters:
    -----------
    mu1 : np.ndarray
        Mean vector of first distribution
    cov1 : np.ndarray
        Covariance matrix of first distribution
    mu2 : np.ndarray
        Mean vector of second distribution
    cov2 : np.ndarray
        Covariance matrix of second distribution
        
    Returns:
    --------
    float
        2-Wasserstein distance
    """
    # Mean difference term
    mean_diff = np.linalg.norm(mu1 - mu2) ** 2
    
    # Covariance term
    # Add small regularization for numerical stability
    reg = 1e-6 * np.eye(len(cov1))
    cov1_reg = cov1 + reg
    cov2_reg = cov2 + reg
    
    # Compute sqrt(cov1)
    cov1_sqrt = sqrtm(cov1_reg)
    
    # Ensure real-valued (sqrtm can return complex with tiny imaginary parts)
    if np.iscomplexobj(cov1_sqrt):
        cov1_sqrt = np.real(cov1_sqrt)
    
    # Compute sqrt(cov1^{1/2} * cov2 * cov1^{1/2})
    middle_term = cov1_sqrt @ cov2_reg @ cov1_sqrt
    middle_sqrt = sqrtm(middle_term)
    
    if np.iscomplexobj(middle_sqrt):
        middle_sqrt = np.real(middle_sqrt)
    
    # Trace term
    trace_term = np.trace(cov1_reg) + np.trace(cov2_reg) - 2 * np.trace(middle_sqrt)
    
    # Total Wasserstein distance
    # Ensure non-negative before sqrt (numerical precision)
    total = max(0, mean_diff + trace_term)
    w_dist = np.sqrt(total)
    
    return w_dist


class TemplateTracker:
    """
    Manages persistent templates and tracks HMM components using Wasserstein distance.
    """
    
    def __init__(
        self,
        n_templates: int,
        feature_dim: int,
        smoothing_rate: float = 0.1
    ):
        """
        Initialize template tracker.
        
        Parameters:
        -----------
        n_templates : int
            Number of persistent templates (G)
        feature_dim : int
            Dimension of feature vectors
        smoothing_rate : float
            Exponential smoothing rate (eta) for template updates
        """
        self.n_templates = n_templates
        self.feature_dim = feature_dim
        self.smoothing_rate = smoothing_rate
        
        # Initialize templates (will be set during initialization phase)
        self.template_means = None
        self.template_covariances = None
        
    def initialize_templates(
        self,
        initial_means: np.ndarray,
        initial_covariances: np.ndarray
    ):
        """
        Initialize templates from initial calibration period.
        
        Parameters:
        -----------
        initial_means : np.ndarray
            Initial mean vectors of shape (n_templates, feature_dim)
        initial_covariances : np.ndarray
            Initial covariance matrices of shape (n_templates, feature_dim, feature_dim)
        """
        assert initial_means.shape[0] == self.n_templates
        assert initial_covariances.shape[0] == self.n_templates
        
        self.template_means = initial_means.copy()
        self.template_covariances = initial_covariances.copy()
    
    def assign_components_to_templates(
        self,
        component_means: np.ndarray,
        component_covariances: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Assign HMM components to closest templates using Wasserstein distance.
        
        Parameters:
        -----------
        component_means : np.ndarray
            Component mean vectors of shape (n_components, feature_dim)
        component_covariances : np.ndarray
            Component covariance matrices of shape (n_components, feature_dim, feature_dim)
            
        Returns:
        --------
        Tuple[np.ndarray, np.ndarray]
            Assignment array (n_components,) and distance matrix (n_components, n_templates)
        """
        n_components = component_means.shape[0]
        
        # Compute Wasserstein distance from each component to each template
        distance_matrix = np.zeros((n_components, self.n_templates))
        
        for k in range(n_components):
            for g in range(self.n_templates):
                distance_matrix[k, g] = wasserstein_distance_gaussian(
                    component_means[k],
                    component_covariances[k],
                    self.template_means[g],
                    self.template_covariances[g]
                )
        
        # Assign each component to closest template
        assignments = np.argmin(distance_matrix, axis=1)
        
        return assignments, distance_matrix
    
    def aggregate_probabilities(
        self,
        component_probs: np.ndarray,
        assignments: np.ndarray
    ) -> np.ndarray:
        """
        Aggregate component probabilities by template assignment.
        
        Parameters:
        -----------
        component_probs : np.ndarray
            Component probabilities of shape (n_components,)
        assignments : np.ndarray
            Template assignments for each component of shape (n_components,)
            
        Returns:
        --------
        np.ndarray
            Template probabilities of shape (n_templates,)
        """
        template_probs = np.zeros(self.n_templates)
        
        for k, g in enumerate(assignments):
            template_probs[g] += component_probs[k]
        
        return template_probs
    
    def update_templates(
        self,
        component_means: np.ndarray,
        component_covariances: np.ndarray,
        assignments: np.ndarray
    ):
        """
        Update template parameters using exponential moving average.
        
        Parameters:
        -----------
        component_means : np.ndarray
            Component mean vectors of shape (n_components, feature_dim)
        component_covariances : np.ndarray
            Component covariance matrices of shape (n_components, feature_dim, feature_dim)
        assignments : np.ndarray
            Template assignments for each component
        """
        # For each template, compute weighted average of assigned components
        for g in range(self.n_templates):
            # Find components assigned to this template
            assigned_components = np.where(assignments == g)[0]
            
            if len(assigned_components) > 0:
                # Compute average of assigned components
                avg_mean = np.mean(component_means[assigned_components], axis=0)
                avg_cov = np.mean(component_covariances[assigned_components], axis=0)
                
                # Update template with exponential smoothing
                # template_new = (1 - eta) * template_old + eta * avg_new
                self.template_means[g] = (
                    (1 - self.smoothing_rate) * self.template_means[g] +
                    self.smoothing_rate * avg_mean
                )
                
                self.template_covariances[g] = (
                    (1 - self.smoothing_rate) * self.template_covariances[g] +
                    self.smoothing_rate * avg_cov
                )
    
    def compute_predictive_moments(
        self,
        template_probs: np.ndarray,
        n_assets: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute predictive mean and covariance for returns using template probabilities.
        
        The feature vector contains [returns; volatility; momentum].
        We extract only the returns portion for portfolio optimization.
        
        Parameters:
        -----------
        template_probs : np.ndarray
            Template probabilities of shape (n_templates,)
        n_assets : int
            Number of assets
            
        Returns:
        --------
        Tuple[np.ndarray, np.ndarray]
            Predictive mean (n_assets,) and covariance (n_assets, n_assets) for returns
        """
        # Extract return portion from feature vectors (first n_assets dimensions)
        return_means = self.template_means[:, :n_assets]
        return_covariances = self.template_covariances[:, :n_assets, :n_assets]
        
        # Compute weighted average
        predictive_mean = np.sum(
            template_probs[:, np.newaxis] * return_means,
            axis=0
        )
        
        predictive_cov = np.sum(
            template_probs[:, np.newaxis, np.newaxis] * return_covariances,
            axis=0
        )
        
        return predictive_mean, predictive_cov
    
    def get_dominant_template(self, template_probs: np.ndarray) -> int:
        """
        Get the index of the dominant (highest probability) template.
        
        Parameters:
        -----------
        template_probs : np.ndarray
            Template probabilities
            
        Returns:
        --------
        int
            Index of dominant template
        """
        return np.argmax(template_probs)
