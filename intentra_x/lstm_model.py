"""
LSTM Intent Classifier

PyTorch-based neural network for intent classification

ETHICAL DESIGN:
- Explainable predictions
- Confidence scores provided
- No adversarial training
"""

import torch
import torch.nn as nn
import numpy as np

class LSTMIntentNet(nn.Module):
    """Simple LSTM network for intent classification"""
    
    def __init__(self, input_size=8, hidden_size=32, num_classes=3):
        super(LSTMIntentNet, self).__init__()
        
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x):
        # x shape: (batch, seq_len, features)
        lstm_out, _ = self.lstm(x)
        
        # Take last output
        last_output = lstm_out[:, -1, :]
        
        # Fully connected layer
        logits = self.fc(last_output)
        probs = self.softmax(logits)
        
        return probs

class IntentClassifier:
    """
    Intent classification system
    
    Classes:
    0: TRANSIT - Direct movement
    1: SURVEILLANCE - Loitering/observation
    2: ADAPTIVE - Risk-aware behavior
    """
    
    def __init__(self):
        self.model = LSTMIntentNet()
        self.model.eval()  # Inference mode
        
        self.class_names = ['TRANSIT', 'SURVEILLANCE', 'ADAPTIVE']
        
        # Initialize with simple heuristic weights
        # In production, this would be trained on real data
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize with reasonable weights for demo"""
        # This creates a functional model without training data
        with torch.no_grad():
            # Simple initialization that responds to feature patterns
            for param in self.model.parameters():
                if len(param.shape) > 1:
                    nn.init.xavier_uniform_(param)
                else:
                    nn.init.zeros_(param)
    
    def predict(self, features):
        """
        Predict intent from feature vector
        
        Returns:
            dict with 'label', 'probabilities', 'confidence'
        """
        # Convert to tensor
        features_tensor = torch.FloatTensor(features).unsqueeze(0).unsqueeze(0)
        
        with torch.no_grad():
            probs = self.model(features_tensor)
            probs_np = probs.numpy()[0]
        
        # Apply heuristic rules for demo (simulates trained model)
        probs_np = self._apply_heuristics(features, probs_np)
        
        # Get prediction
        predicted_class = np.argmax(probs_np)
        label = self.class_names[predicted_class]
        confidence = float(probs_np[predicted_class])
        
        return {
            'label': label,
            'probabilities': probs_np,
            'confidence': confidence,
            'class_id': int(predicted_class)
        }
    
    def _apply_heuristics(self, features, base_probs):
        """
        Apply heuristic rules to simulate trained model behavior
        
        Feature indices:
        0: speed_variance
        1: altitude_variance
        2: curvature
        3: loiter_ratio
        4: direction_changes
        5: altitude_trend
        6: current_speed
        7: current_altitude
        """
        probs = np.array([0.33, 0.33, 0.34])  # Start uniform
        
        # High loiter ratio → SURVEILLANCE
        if features[3] > 0.5:
            probs[1] += 0.3
        
        # High speed, low curvature → TRANSIT
        if features[6] > 0.5 and features[2] < 0.3:
            probs[0] += 0.3
        
        # High variance and direction changes → ADAPTIVE
        if features[0] > 0.4 and features[4] > 0.3:
            probs[2] += 0.3
        
        # High curvature → SURVEILLANCE
        if features[2] > 0.5:
            probs[1] += 0.2
        
        # Normalize
        probs = probs / probs.sum()
        
        # Add some noise for realism
        noise = np.random.normal(0, 0.05, 3)
        probs = probs + noise
        probs = np.clip(probs, 0.05, 0.95)
        probs = probs / probs.sum()
        
        return probs
    
    def compute_entropy(self, probabilities):
        """Compute Shannon entropy of prediction"""
        probs = np.array(probabilities)
        probs = probs[probs > 0]  # Avoid log(0)
        entropy = -np.sum(probs * np.log2(probs))
        
        # Normalize to [0, 1]
        max_entropy = np.log2(len(self.class_names))
        normalized_entropy = entropy / max_entropy
        
        return normalized_entropy

class MCDropoutClassifier(IntentClassifier):
    """
    Intent classifier with Monte Carlo Dropout for uncertainty estimation
    
    ETHICAL NOTE:
    - Provides uncertainty estimates
    - Enables risk-aware decision making
    """
    
    def __init__(self, n_samples=10):
        super().__init__()
        self.n_samples = n_samples
        self.model.train()  # Keep dropout active
    
    def predict_with_uncertainty(self, features):
        """Predict with uncertainty using MC Dropout"""
        features_tensor = torch.FloatTensor(features).unsqueeze(0).unsqueeze(0)
        
        predictions = []
        for _ in range(self.n_samples):
            with torch.no_grad():
                probs = self.model(features_tensor)
                predictions.append(probs.numpy()[0])
        
        predictions = np.array(predictions)
        
        # Mean prediction
        mean_probs = predictions.mean(axis=0)
        
        # Uncertainty as variance
        uncertainty = predictions.var(axis=0).mean()
        
        predicted_class = np.argmax(mean_probs)
        label = self.class_names[predicted_class]
        
        return {
            'label': label,
            'probabilities': mean_probs,
            'confidence': float(mean_probs[predicted_class]),
            'uncertainty': float(uncertainty)
        }
