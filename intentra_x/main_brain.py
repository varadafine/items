"""
INTENTRA-X Main Brain
Cognitive Airspace Intelligence & Risk-Aware Autonomous Drone System

ETHICAL NOTICE:
- This system is for SAFETY and AWARENESS only
- No stealth, evasion, or invisibility logic
- Decision-support tool for risk assessment
- Explainable AI with full transparency
- Designed for research and competition use
"""

import airsim
import numpy as np
import time
import json
from datetime import datetime
from state_machine import StateMachine, DroneState
from feature_extractor import FeatureExtractor
from lstm_model import IntentClassifier
from risk_engine import RiskEngine
from uncertainty import UncertaintyEstimator
from counterfactual import CounterfactualEngine
from telemetry_logger import TelemetryLogger

class IntentRAXBrain:
    """Main control system for autonomous drone with cognitive intelligence"""
    
    def __init__(self, simulation_mode=True):
        self.simulation_mode = simulation_mode
        self.client = None
        self.state_machine = StateMachine()
        self.feature_extractor = FeatureExtractor()
        self.intent_model = IntentClassifier()
        self.risk_engine = RiskEngine()
        self.uncertainty_estimator = UncertaintyEstimator()
        self.counterfactual = CounterfactualEngine()
        self.logger = TelemetryLogger()
        
        # Trajectory history
        self.trajectory = []
        self.max_trajectory_length = 100
        
        # Behavior parameters
        self.transit_speed = 5.0
        self.surveillance_radius = 15.0
        self.surveillance_altitude = -20.0
        self.risk_threshold = 0.7
        
        if simulation_mode:
            self.connect_airsim()
    
    def connect_airsim(self):
        """Connect to AirSim simulator"""
        try:
            self.client = airsim.MultirotorClient()
            self.client.confirmConnection()
            self.client.enableApiControl(True)
            self.client.armDisarm(True)
            print("✓ Connected to AirSim")
        except Exception as e:
            print(f"✗ AirSim connection failed: {e}")
            self.simulation_mode = False
    
    def get_telemetry(self):
        """Extract current drone telemetry"""
        if not self.client:
            return None
        
        state = self.client.getMultirotorState()
        pos = state.kinematics_estimated.position
        vel = state.kinematics_estimated.linear_velocity
        orientation = state.kinematics_estimated.orientation
        
        # Convert quaternion to yaw
        yaw = airsim.to_eularian_angles(orientation)[2]
        
        telemetry = {
            'timestamp': time.time(),
            'position': [pos.x_val, pos.y_val, pos.z_val],
            'velocity': [vel.x_val, vel.y_val, vel.z_val],
            'altitude': -pos.z_val,  # AirSim uses NED coordinates
            'yaw': np.degrees(yaw),
            'speed': np.linalg.norm([vel.x_val, vel.y_val, vel.z_val])
        }
        
        return telemetry
    
    def execute_transit(self):
        """Execute transit behavior - straight line movement"""
        if not self.client:
            return
        
        # Move forward at constant speed
        self.client.moveByVelocityAsync(
            self.transit_speed, 0, 0,
            duration=1,
            drivetrain=airsim.DrivetrainType.ForwardOnly
        )
    
    def execute_surveillance(self, center_x=0, center_y=0):
        """Execute surveillance behavior - circular loitering pattern"""
        if not self.client:
            return
        
        # Circular path around point of interest
        t = time.time()
        angle = (t % 20) / 20 * 2 * np.pi  # 20 second orbit
        
        target_x = center_x + self.surveillance_radius * np.cos(angle)
        target_y = center_y + self.surveillance_radius * np.sin(angle)
        
        self.client.moveToPositionAsync(
            target_x, target_y, self.surveillance_altitude,
            velocity=3,
            timeout_sec=1
        )
    
    def execute_adaptive(self, risk_breakdown):
        """Execute adaptive behavior - risk mitigation"""
        if not self.client:
            return
        
        # ETHICAL NOTE: This is risk AWARENESS, not evasion
        # System adapts to maintain safety margins
        
        current_pos = self.get_telemetry()['position']
        
        # If too close to high-risk zone, increase altitude
        if risk_breakdown.get('proximity_risk', 0) > 0.5:
            target_z = current_pos[2] - 5  # Go higher (NED coords)
            self.client.moveToPositionAsync(
                current_pos[0], current_pos[1], target_z,
                velocity=2,
                timeout_sec=1
            )
        else:
            # Slow, deliberate movement
            self.client.moveByVelocityAsync(2, 0, 0, duration=1)
    
    def update_trajectory(self, position):
        """Maintain trajectory history"""
        self.trajectory.append(position)
        if len(self.trajectory) > self.max_trajectory_length:
            self.trajectory.pop(0)
    
    def run_cycle(self):
        """Execute one intelligence cycle"""
        # Get current telemetry
        telemetry = self.get_telemetry()
        if not telemetry:
            return None
        
        self.update_trajectory(telemetry['position'])
        
        # Extract behavioral features
        features = self.feature_extractor.extract(self.trajectory, telemetry)
        
        # Classify intent using LSTM
        intent_result = self.intent_model.predict(features)
        intent_label = intent_result['label']
        intent_probs = intent_result['probabilities']
        
        # Estimate uncertainty
        uncertainty = self.uncertainty_estimator.compute(intent_probs)
        
        # Compute risk
        risk_result = self.risk_engine.compute_risk(
            telemetry['position'],
            self.trajectory,
            telemetry['altitude']
        )
        
        # Predict counterfactual risk
        cf_risk = self.counterfactual.predict_future_risk(
            telemetry['position'],
            telemetry['velocity'],
            self.state_machine.current_state
        )
        
        # State machine decision
        self.state_machine.update(
            risk_result['risk_score'],
            uncertainty,
            self.risk_threshold
        )
        
        # Execute behavior based on state
        if self.state_machine.current_state == DroneState.TRANSIT:
            self.execute_transit()
        elif self.state_machine.current_state == DroneState.SURVEILLANCE:
            self.execute_surveillance()
        elif self.state_machine.current_state == DroneState.ADAPTIVE:
            self.execute_adaptive(risk_result['risk_breakdown'])
        
        # Compile output
        output = {
            'timestamp': telemetry['timestamp'],
            'position': telemetry['position'],
            'velocity': telemetry['velocity'],
            'altitude': telemetry['altitude'],
            'speed': telemetry['speed'],
            'trajectory': self.trajectory[-20:],  # Last 20 points
            'intent': intent_label,
            'intent_confidence': float(np.max(intent_probs)),
            'intent_probabilities': intent_probs.tolist(),
            'uncertainty': float(uncertainty),
            'risk_score': risk_result['risk_score'],
            'risk_breakdown': risk_result['risk_breakdown'],
            'risk_explanation': risk_result['explanation'],
            'state': self.state_machine.current_state.name,
            'counterfactual_risk': cf_risk,
            'state_history': self.state_machine.get_history()
        }
        
        # Log and save
        self.logger.log(output)
        self.save_live_output(output)
        
        return output
    
    def save_live_output(self, data):
        """Save current state to JSON for dashboard"""
        with open('intentra_x/live_output.json', 'w') as f:
            json.dump(data, f, indent=2)
    
    def takeoff(self):
        """Takeoff sequence"""
        if self.client:
            print("Taking off...")
            self.client.takeoffAsync().join()
            self.client.moveToZAsync(-10, 2).join()
            print("✓ Airborne")
    
    def land(self):
        """Landing sequence"""
        if self.client:
            print("Landing...")
            self.client.landAsync().join()
            self.client.armDisarm(False)
            self.client.enableApiControl(False)
            print("✓ Landed")

def main():
    """Main execution loop"""
    brain = IntentRAXBrain(simulation_mode=True)
    
    if brain.simulation_mode:
        brain.takeoff()
        
        try:
            print("\n🧠 INTENTRA-X Brain Active")
            print("Press Ctrl+C to stop\n")
            
            while True:
                output = brain.run_cycle()
                if output:
                    print(f"State: {output['state']} | "
                          f"Intent: {output['intent']} ({output['intent_confidence']:.2f}) | "
                          f"Risk: {output['risk_score']:.2f} | "
                          f"Uncertainty: {output['uncertainty']:.2f}")
                
                time.sleep(0.5)
        
        except KeyboardInterrupt:
            print("\n\nShutting down...")
            brain.land()
    else:
        print("Run in demo mode - check dashboard with sample data")

if __name__ == "__main__":
    main()
