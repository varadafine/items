"""
Uncertainty Estimation Module

Quantifies prediction uncertainty for risk-aware decision making

ETHICAL DESIGN:
- Transparent uncertainty quantification
- Enables cautious behavior under uncertainty
- Supports explainable AI
"""

import numpy as np

class UncertaintyEstimator:
    """Estimate prediction uncertainty using multiple methods"""
    
    def __init__(self):
        self.uncertainty_history = []
    
    def compute(self, probabilities):
        """
        Compute uncertainty from prediction probabilities
        
        Methods:
        1. Shannon entropy
        2. Confidence margin
        3. Variance-based
        
        Returns:
            uncertainty_score (0-1, higher = more uncertain)
        """
        probs = np.array(probabilities)
        
        # Method 1: Shannon entropy
        entropy = self._compute_entropy(probs)
        
        # Method 2: Confidence margin (difference between top 2 predictions)
        sorted_probs = np.sort(probs)[::-1]
        margin = sorted_probs[0] - sorted_probs[1] if len(sorted_probs) > 1 else 1.0
        margin_uncertainty = 1.0 - margin
        
        # Method 3: Variance
        variance = np.var(probs)
        variance_uncertainty = variance * 3  # Scale to [0, 1] range
        
        # Combined uncertainty
        uncertainty = (
            0.5 * entropy +
            0.3 * margin_uncertainty +
            0.2 * variance_uncertainty
        )
        
        uncertainty = np.clip(uncertainty, 0, 1)
        
        # Log history
        self.uncertainty_history.append(uncertainty)
        if len(self.uncertainty_history) > 100:
            self.uncertainty_history.pop(0)
        
        return float(uncertainty)
    
    def _compute_entropy(self, probabilities):
        """
        Compute normalized Shannon entropy
        
        Entropy = -sum(p * log(p))
        Normalized to [0, 1]
        """
        probs = probabilities[probabilities > 1e-10]  # Avoid log(0)
        
        if len(probs) == 0:
            return 1.0  # Maximum uncertainty
        
        entropy = -np.sum(probs * np.log2(probs))
        
        # Normalize by maximum possible entropy
        max_entropy = np.log2(len(probabilities))
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
        
        return normalized_entropy
    
    def compute_mc_dropout_uncertainty(self, predictions_list):
        """
        Compute uncertainty from Monte Carlo Dropout samples
        
        Args:
            predictions_list: List of prediction arrays from multiple forward passes
        
        Returns:
            uncertainty_score
        """
        predictions = np.array(predictions_list)
        
        # Variance across predictions
        variance = np.var(predictions, axis=0).mean()
        
        # Entropy of mean prediction
        mean_pred = predictions.mean(axis=0)
        entropy = self._compute_entropy(mean_pred)
        
        # Combined uncertainty
        uncertainty = 0.6 * variance + 0.4 * entropy
        
        return float(np.clip(uncertainty, 0, 1))
    
    def get_uncertainty_trend(self):
        """Get recent uncertainty trend"""
        if len(self.uncertainty_history) < 5:
            return 0.0
        
        recent = self.uncertainty_history[-10:]
        trend = np.polyfit(range(len(recent)), recent, 1)[0]
        return float(trend)
    
    def is_high_uncertainty(self, threshold=0.6):
        """Check if current uncertainty is high"""
        if not self.uncertainty_history:
            return False
        
        return self.uncertainty_history[-1] > threshold

class BayesianUncertainty:
    """
    Bayesian approach to uncertainty estimation
    
    ETHICAL NOTE:
    - Provides probabilistic confidence bounds
    - Supports cautious decision-making
    """
    
    def __init__(self, prior_alpha=1.0):
        self.prior_alpha = prior_alpha
        self.observations = []
    
    def update(self, prediction, outcome):
        """Update belief based on observation"""
        self.observations.append({
            'prediction': prediction,
            'outcome': outcome,
            'correct': prediction == outcome
        })
    
    def get_credible_interval(self, confidence=0.95):
        """Compute Bayesian credible interval for accuracy"""
        if not self.observations:
            return (0.0, 1.0)
        
        successes = sum(1 for obs in self.observations if obs['correct'])
        trials = len(self.observations)
        
        # Beta distribution parameters
        alpha = self.prior_alpha + successes
        beta = self.prior_alpha + (trials - successes)
        
        # Approximate credible interval
        mean = alpha / (alpha + beta)
        variance = (alpha * beta) / ((alpha + beta)**2 * (alpha + beta + 1))
        std = np.sqrt(variance)
        
        # 95% credible interval (approximate)
        z = 1.96  # For 95% confidence
        lower = max(0, mean - z * std)
        upper = min(1, mean + z * std)
        
        return (lower, upper)
    
    def get_epistemic_uncertainty(self):
        """Get model uncertainty (epistemic)"""
        if len(self.observations) < 2:
            return 1.0
        
        # Uncertainty decreases with more observations
        n = len(self.observations)
        uncertainty = 1.0 / np.sqrt(n)
        
        return min(uncertainty, 1.0)
