"""
Risk Assessment Engine

Computes exposure and detection risk using spatial reasoning

ETHICAL FRAMEWORK:
- Risk assessment for AWARENESS, not evasion
- Transparent risk factors
- Explainable decisions
- Safety-oriented design
"""

import numpy as np
from environment import EnvironmentModel

class RiskEngine:
    """Compute multi-factor risk assessment"""
    
    def __init__(self):
        self.environment = EnvironmentModel()
        self.risk_history = []
    
    def compute_risk(self, position, trajectory, altitude):
        """
        Compute comprehensive risk score
        
        Risk factors:
        1. Camera detection probability
        2. Altitude exposure
        3. Loiter duration in area
        4. Proximity to sensitive zones
        5. Trajectory predictability
        
        Returns:
            dict with risk_score, breakdown, and explanation
        """
        # Factor 1: Camera detection probability
        detection_prob = self.environment.camera.compute_detection_probability(
            position, altitude
        )
        
        # Factor 2: Altitude exposure (higher = more exposed)
        altitude_risk = min(altitude / 50.0, 1.0)
        
        # Factor 3: Loiter duration
        loiter_risk = self._compute_loiter_risk(trajectory)
        
        # Factor 4: Zone proximity
        zone_proximity = self.environment.get_zone_proximity(position)
        proximity_risk = zone_proximity['building'] * 0.7 + zone_proximity['tree'] * 0.3
        
        # Factor 5: Trajectory predictability
        predictability_risk = self._compute_predictability(trajectory)
        
        # Weighted combination
        risk_score = (
            0.35 * detection_prob +
            0.20 * altitude_risk +
            0.20 * loiter_risk +
            0.15 * proximity_risk +
            0.10 * predictability_risk
        )
        
        risk_score = np.clip(risk_score, 0, 1)
        
        # Build breakdown
        breakdown = {
            'detection_probability': float(detection_prob),
            'altitude_risk': float(altitude_risk),
            'loiter_risk': float(loiter_risk),
            'proximity_risk': float(proximity_risk),
            'predictability_risk': float(predictability_risk)
        }
        
        # Generate explanation
        explanation = self._generate_explanation(breakdown, risk_score)
        
        # Log history
        self.risk_history.append(risk_score)
        if len(self.risk_history) > 100:
            self.risk_history.pop(0)
        
        return {
            'risk_score': float(risk_score),
            'risk_breakdown': breakdown,
            'explanation': explanation
        }
    
    def _compute_loiter_risk(self, trajectory):
        """Compute risk from loitering behavior"""
        if len(trajectory) < 5:
            return 0.0
        
        # Check if drone is staying in small area
        recent_traj = np.array(trajectory[-10:])
        centroid = recent_traj.mean(axis=0)
        
        distances = [np.linalg.norm(p - centroid) for p in recent_traj]
        avg_distance = np.mean(distances)
        
        # Small average distance = loitering
        loiter_risk = max(0, 1 - avg_distance / 20.0)
        
        return loiter_risk
    
    def _compute_predictability(self, trajectory):
        """Compute trajectory predictability"""
        if len(trajectory) < 5:
            return 0.0
        
        # Linear trajectory = predictable = higher risk
        traj = np.array(trajectory[-10:])
        
        # Fit line and compute deviation
        if len(traj) > 2:
            # Compute straightness
            start = traj[0]
            end = traj[-1]
            line_length = np.linalg.norm(end - start)
            
            if line_length < 1:
                return 0.5  # Stationary
            
            # Sum of deviations from straight line
            total_deviation = 0
            for point in traj[1:-1]:
                # Distance from point to line
                deviation = self._point_to_line_distance(point, start, end)
                total_deviation += deviation
            
            avg_deviation = total_deviation / len(traj)
            
            # Low deviation = predictable = higher risk
            predictability = max(0, 1 - avg_deviation / 10.0)
            return predictability
        
        return 0.5
    
    def _point_to_line_distance(self, point, line_start, line_end):
        """Compute perpendicular distance from point to line"""
        line_vec = line_end - line_start
        point_vec = point - line_start
        
        line_len = np.linalg.norm(line_vec)
        if line_len < 1e-6:
            return np.linalg.norm(point_vec)
        
        line_unitvec = line_vec / line_len
        projection = np.dot(point_vec, line_unitvec)
        
        if projection < 0:
            return np.linalg.norm(point_vec)
        elif projection > line_len:
            return np.linalg.norm(point - line_end)
        else:
            closest_point = line_start + projection * line_unitvec
            return np.linalg.norm(point - closest_point)
    
    def _generate_explanation(self, breakdown, total_risk):
        """Generate human-readable risk explanation"""
        explanation = []
        
        # Overall assessment
        if total_risk > 0.7:
            explanation.append("⚠️ HIGH RISK: Significant exposure detected")
        elif total_risk > 0.4:
            explanation.append("⚡ MODERATE RISK: Elevated exposure level")
        else:
            explanation.append("✓ LOW RISK: Minimal exposure")
        
        # Specific factors
        if breakdown['detection_probability'] > 0.6:
            explanation.append("• High camera detection probability")
        
        if breakdown['altitude_risk'] > 0.6:
            explanation.append("• Elevated altitude increases visibility")
        
        if breakdown['loiter_risk'] > 0.5:
            explanation.append("• Loitering behavior detected in area")
        
        if breakdown['proximity_risk'] > 0.5:
            explanation.append("• Close proximity to monitored zone")
        
        if breakdown['predictability_risk'] > 0.6:
            explanation.append("• Trajectory pattern is highly predictable")
        
        # Recommendations
        if total_risk > 0.6:
            explanation.append("→ Recommend: Adaptive behavior mode")
        
        return explanation
    
    def get_risk_trend(self):
        """Get recent risk trend"""
        if len(self.risk_history) < 5:
            return 0.0
        
        recent = self.risk_history[-10:]
        trend = np.polyfit(range(len(recent)), recent, 1)[0]
        return float(trend)
