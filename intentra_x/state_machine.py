"""
State Machine for Drone Behavior Control

ETHICAL DESIGN:
- Transparent state transitions
- Risk-aware decision making
- Safety-first logic
"""

from enum import Enum
from datetime import datetime

class DroneState(Enum):
    """Drone operational states"""
    TRANSIT = "TRANSIT"
    SURVEILLANCE = "SURVEILLANCE"
    ADAPTIVE = "ADAPTIVE"

class StateMachine:
    """Manages drone behavioral state transitions"""
    
    def __init__(self):
        self.current_state = DroneState.TRANSIT
        self.state_history = []
        self.transition_count = 0
    
    def update(self, risk_score, uncertainty, risk_threshold):
        """
        Update state based on risk and uncertainty
        
        Transition logic:
        - High risk OR high uncertainty → ADAPTIVE
        - Medium risk + low uncertainty → SURVEILLANCE
        - Low risk → TRANSIT
        """
        previous_state = self.current_state
        
        # Decision logic
        if risk_score > risk_threshold or uncertainty > 0.6:
            self.current_state = DroneState.ADAPTIVE
        elif risk_score > 0.4 and uncertainty < 0.4:
            self.current_state = DroneState.SURVEILLANCE
        else:
            self.current_state = DroneState.TRANSIT
        
        # Log transition
        if previous_state != self.current_state:
            self.log_transition(previous_state, self.current_state, risk_score, uncertainty)
    
    def log_transition(self, from_state, to_state, risk, uncertainty):
        """Record state transition"""
        self.transition_count += 1
        transition = {
            'timestamp': datetime.now().isoformat(),
            'from': from_state.name,
            'to': to_state.name,
            'risk': float(risk),
            'uncertainty': float(uncertainty),
            'transition_id': self.transition_count
        }
        self.state_history.append(transition)
        
        # Keep last 50 transitions
        if len(self.state_history) > 50:
            self.state_history.pop(0)
        
        print(f"⚡ STATE TRANSITION: {from_state.name} → {to_state.name} "
              f"(Risk: {risk:.2f}, Uncertainty: {uncertainty:.2f})")
    
    def get_history(self):
        """Return recent state transition history"""
        return self.state_history[-10:]  # Last 10 transitions
