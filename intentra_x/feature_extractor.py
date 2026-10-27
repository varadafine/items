"""
Feature Extractor for Behavioral Analysis

Extracts time-series features from drone telemetry
for intent classification
"""

import numpy as np

class FeatureExtractor:
    """Extract behavioral features from trajectory and telemetry"""
    
    def __init__(self, sequence_length=20):
        self.sequence_length = sequence_length
    
    def extract(self, trajectory, telemetry):
        """
        Extract feature vector from trajectory
        
        Features:
        1. Speed variance
        2. Altitude variance
        3. Path curvature
        4. Loiter duration (low speed time)
        5. Direction changes
        6. Altitude trend
        """
        if len(trajectory) < 3:
            return self._default_features()
        
        # Convert to numpy array
        traj = np.array(trajectory[-self.sequence_length:])
        
        # Speed calculation
        speeds = []
        for i in range(1, len(traj)):
            delta = traj[i] - traj[i-1]
            speed = np.linalg.norm(delta)
            speeds.append(speed)
        
        speeds = np.array(speeds) if speeds else np.array([0])
        
        # Feature 1: Speed variance
        speed_variance = np.var(speeds)
        
        # Feature 2: Altitude variance
        altitudes = traj[:, 2]
        altitude_variance = np.var(altitudes)
        
        # Feature 3: Path curvature (angle changes)
        curvature = self._compute_curvature(traj)
        
        # Feature 4: Loiter duration (proportion of low-speed movement)
        loiter_ratio = np.sum(speeds < 1.0) / len(speeds) if len(speeds) > 0 else 0
        
        # Feature 5: Direction changes
        direction_changes = self._compute_direction_changes(traj)
        
        # Feature 6: Altitude trend
        altitude_trend = self._compute_trend(altitudes)
        
        # Feature 7: Current speed
        current_speed = telemetry.get('speed', 0)
        
        # Feature 8: Current altitude
        current_altitude = telemetry.get('altitude', 0)
        
        features = np.array([
            speed_variance,
            altitude_variance,
            curvature,
            loiter_ratio,
            direction_changes,
            altitude_trend,
            current_speed,
            current_altitude
        ])
        
        # Normalize features
        features = self._normalize(features)
        
        return features
    
    def _compute_curvature(self, trajectory):
        """Compute average path curvature"""
        if len(trajectory) < 3:
            return 0.0
        
        angles = []
        for i in range(1, len(trajectory) - 1):
            v1 = trajectory[i] - trajectory[i-1]
            v2 = trajectory[i+1] - trajectory[i]
            
            v1_norm = np.linalg.norm(v1)
            v2_norm = np.linalg.norm(v2)
            
            if v1_norm > 0 and v2_norm > 0:
                cos_angle = np.dot(v1, v2) / (v1_norm * v2_norm)
                cos_angle = np.clip(cos_angle, -1, 1)
                angle = np.arccos(cos_angle)
                angles.append(angle)
        
        return np.mean(angles) if angles else 0.0
    
    def _compute_direction_changes(self, trajectory):
        """Count significant direction changes"""
        if len(trajectory) < 3:
            return 0.0
        
        changes = 0
        for i in range(1, len(trajectory) - 1):
            v1 = trajectory[i] - trajectory[i-1]
            v2 = trajectory[i+1] - trajectory[i]
            
            v1_norm = np.linalg.norm(v1)
            v2_norm = np.linalg.norm(v2)
            
            if v1_norm > 0.5 and v2_norm > 0.5:
                cos_angle = np.dot(v1, v2) / (v1_norm * v2_norm)
                angle = np.arccos(np.clip(cos_angle, -1, 1))
                
                if angle > np.pi / 4:  # 45 degrees
                    changes += 1
        
        return changes / len(trajectory)
    
    def _compute_trend(self, values):
        """Compute linear trend (positive = increasing)"""
        if len(values) < 2:
            return 0.0
        
        x = np.arange(len(values))
        coeffs = np.polyfit(x, values, 1)
        return coeffs[0]  # Slope
    
    def _normalize(self, features):
        """Simple normalization"""
        # Clip extreme values
        features = np.clip(features, -100, 100)
        
        # Scale to reasonable range
        features = features / (np.abs(features).max() + 1e-6)
        
        return features
    
    def _default_features(self):
        """Return default feature vector"""
        return np.zeros(8)
