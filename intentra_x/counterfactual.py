"""
Counterfactual Risk Prediction

Simulates "what-if" scenarios for risk projection

ETHICAL DESIGN:
- Predictive risk awareness
- Enables proactive safety measures
- Transparent future state modeling
"""

import numpy as np
from environment import EnvironmentModel

class CounterfactualEngine:
    """Predict future risk based on current trajectory"""
    
    def __init__(self, prediction_horizon=5.0):
        self.prediction_horizon = prediction_horizon  # seconds
        self.environment = EnvironmentModel()
    
    def predict_future_risk(self, current_position, current_velocity, current_state):
        """
        Predict risk if current behavior continues
        
        Args:
            current_position: [x, y, z]
            current_velocity: [vx, vy, vz]
            current_state: DroneState enum
        
        Returns:
            predicted_risk_score (0-1)
        """
        # Simulate future position
        future_position = self._simulate_future_position(
            current_position,
            current_velocity,
            current_state
        )
        
        # Compute risk at future position
        future_altitude = -future_position[2]  # NED to altitude
        
        detection_prob = self.environment.camera.compute_detection_probability(
            future_position,
            future_altitude
        )
        
        # Factor in trajectory predictability
        velocity_magnitude = np.linalg.norm(current_velocity)
        predictability = min(velocity_magnitude / 10.0, 1.0)
        
        # Future risk estimate
        future_risk = (
            0.6 * detection_prob +
            0.3 * predictability +
            0.1 * (future_altitude / 50.0)
        )
        
        future_risk = np.clip(future_risk, 0, 1)
        
        return float(future_risk)
    
    def _simulate_future_position(self, position, velocity, state):
        """Simulate position after prediction horizon"""
        pos = np.array(position)
        vel = np.array(velocity)
        
        # Simple linear extrapolation
        # In production, would use more sophisticated motion model
        future_pos = pos + vel * self.prediction_horizon
        
        return future_pos
    
    def generate_alternative_scenarios(self, current_position, current_velocity):
        """
        Generate multiple counterfactual scenarios
        
        Returns:
            dict of scenario_name: predicted_risk
        """
        scenarios = {}
        
        # Convert to numpy array for operations
        vel = np.array(current_velocity)
        
        # Scenario 1: Continue current trajectory
        scenarios['continue'] = self.predict_future_risk(
            current_position,
            current_velocity,
            None
        )
        
        # Scenario 2: Increase altitude
        alt_velocity = vel.copy()
        alt_velocity[2] -= 2  # Move up (NED coords)
        scenarios['climb'] = self.predict_future_risk(
            current_position,
            alt_velocity,
            None
        )
        
        # Scenario 3: Decrease speed
        slow_velocity = vel * 0.5
        scenarios['slow_down'] = self.predict_future_risk(
            current_position,
            slow_velocity,
            None
        )
        
        # Scenario 4: Change direction
        rotated_velocity = self._rotate_velocity(vel, np.pi/2)
        scenarios['turn_90deg'] = self.predict_future_risk(
            current_position,
            rotated_velocity,
            None
        )
        
        return scenarios
    
    def _rotate_velocity(self, velocity, angle):
        """Rotate velocity vector in XY plane"""
        vel = np.array(velocity)
        
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)
        
        rotated = vel.copy()
        rotated[0] = vel[0] * cos_a - vel[1] * sin_a
        rotated[1] = vel[0] * sin_a + vel[1] * cos_a
        
        return rotated
    
    def recommend_action(self, scenarios):
        """
        Recommend best action based on scenario analysis
        
        ETHICAL NOTE:
        - Recommends safest option
        - Transparent decision logic
        """
        # Find scenario with lowest risk
        best_scenario = min(scenarios.items(), key=lambda x: x[1])
        
        recommendations = {
            'continue': "Maintain current trajectory",
            'climb': "Increase altitude for better safety margin",
            'slow_down': "Reduce speed to decrease exposure",
            'turn_90deg': "Change direction to avoid high-risk zone"
        }
        
        return {
            'recommended_action': best_scenario[0],
            'expected_risk': best_scenario[1],
            'description': recommendations.get(best_scenario[0], "Unknown action")
        }

class RiskProjection:
    """Project risk over time horizon"""
    
    def __init__(self):
        self.time_steps = 10
    
    def project_risk_timeline(self, current_position, current_velocity, risk_engine):
        """
        Project risk over next N time steps
        
        Returns:
            list of (time, risk) tuples
        """
        timeline = []
        
        pos = np.array(current_position)
        vel = np.array(current_velocity)
        
        for t in range(self.time_steps):
            # Simulate position at time t
            future_pos = pos + vel * t * 0.5  # 0.5 second steps
            future_alt = -future_pos[2]
            
            # Compute risk (simplified)
            risk = risk_engine.compute_risk(
                future_pos.tolist(),
                [future_pos.tolist()],
                future_alt
            )
            
            timeline.append((t * 0.5, risk['risk_score']))
        
        return timeline
