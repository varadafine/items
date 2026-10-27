"""
INTENTRA-X: Cognitive Airspace Intelligence & Risk-Aware Autonomous Drone System

A competition-grade autonomous drone intelligence platform combining:
- AirSim simulation
- PyTorch LSTM intent classification
- Multi-factor risk assessment
- Uncertainty quantification
- Counterfactual reasoning
- Mission-grade Streamlit dashboard

ETHICAL FRAMEWORK:
- Designed for SAFETY and AWARENESS
- No stealth or evasion capabilities
- Transparent, explainable AI
- Decision-support tool
- Full audit trail

Author: Competition Team
Version: 1.0.0
"""

__version__ = "1.0.0"
__author__ = "INTENTRA-X Team"

# Core modules are imported lazily so package-level tools such as the joint-risk
# CLI do not require optional AirSim/Streamlit dependencies merely to start.
from importlib import import_module


_LAZY_IMPORTS = {
    'IntentRAXBrain': ('.main_brain', 'IntentRAXBrain'),
    'StateMachine': ('.state_machine', 'StateMachine'),
    'DroneState': ('.state_machine', 'DroneState'),
    'EnvironmentModel': ('.environment', 'EnvironmentModel'),
    'CameraModel': ('.environment', 'CameraModel'),
    'FeatureExtractor': ('.feature_extractor', 'FeatureExtractor'),
    'IntentClassifier': ('.lstm_model', 'IntentClassifier'),
    'RiskEngine': ('.risk_engine', 'RiskEngine'),
    'UncertaintyEstimator': ('.uncertainty', 'UncertaintyEstimator'),
    'CounterfactualEngine': ('.counterfactual', 'CounterfactualEngine'),
    'HeatmapEngine': ('.heatmap_engine', 'HeatmapEngine'),
    'TelemetryLogger': ('.telemetry_logger', 'TelemetryLogger'),
}


def __getattr__(name):
    if name not in _LAZY_IMPORTS:
        raise AttributeError(name)
    module_name, attribute_name = _LAZY_IMPORTS[name]
    value = getattr(import_module(module_name, __name__), attribute_name)
    globals()[name] = value
    return value

__all__ = [
    'IntentRAXBrain',
    'StateMachine',
    'DroneState',
    'EnvironmentModel',
    'CameraModel',
    'FeatureExtractor',
    'IntentClassifier',
    'RiskEngine',
    'UncertaintyEstimator',
    'CounterfactualEngine',
    'HeatmapEngine',
    'TelemetryLogger'
]
