"""
INTENTRA-X System Test Suite

Tests all major components without AirSim
"""

import numpy as np
import sys

def test_feature_extraction():
    """Test feature extractor"""
    print("Testing Feature Extraction...", end=" ")
    
    from feature_extractor import FeatureExtractor
    
    extractor = FeatureExtractor()
    
    # Create test trajectory
    trajectory = [[i, i*0.5, -10] for i in range(20)]
    telemetry = {'speed': 5.0, 'altitude': 10}
    
    features = extractor.extract(trajectory, telemetry)
    
    assert len(features) == 8, "Feature vector should have 8 dimensions"
    assert not np.any(np.isnan(features)), "Features should not contain NaN"
    
    print("✅ PASSED")
    return True

def test_lstm_model():
    """Test LSTM intent classifier"""
    print("Testing LSTM Model...", end=" ")
    
    from lstm_model import IntentClassifier
    
    classifier = IntentClassifier()
    
    # Test prediction
    features = np.random.randn(8)
    result = classifier.predict(features)
    
    assert 'label' in result, "Result should contain label"
    assert 'probabilities' in result, "Result should contain probabilities"
    assert 'confidence' in result, "Result should contain confidence"
    assert result['label'] in ['TRANSIT', 'SURVEILLANCE', 'ADAPTIVE'], "Invalid label"
    assert 0 <= result['confidence'] <= 1, "Confidence should be in [0, 1]"
    
    print("✅ PASSED")
    return True

def test_risk_engine():
    """Test risk assessment"""
    print("Testing Risk Engine...", end=" ")
    
    from risk_engine import RiskEngine
    
    engine = RiskEngine()
    
    # Test risk computation
    position = [10, 10, -20]
    trajectory = [[10, 10, -20]] * 10
    altitude = 20
    
    result = engine.compute_risk(position, trajectory, altitude)
    
    assert 'risk_score' in result, "Result should contain risk_score"
    assert 'risk_breakdown' in result, "Result should contain breakdown"
    assert 'explanation' in result, "Result should contain explanation"
    assert 0 <= result['risk_score'] <= 1, "Risk score should be in [0, 1]"
    
    print("✅ PASSED")
    return True

def test_uncertainty():
    """Test uncertainty estimation"""
    print("Testing Uncertainty Estimator...", end=" ")
    
    from uncertainty import UncertaintyEstimator
    
    estimator = UncertaintyEstimator()
    
    # Test with different probability distributions
    probs_certain = np.array([0.9, 0.05, 0.05])
    probs_uncertain = np.array([0.33, 0.33, 0.34])
    
    unc_certain = estimator.compute(probs_certain)
    unc_uncertain = estimator.compute(probs_uncertain)
    
    assert 0 <= unc_certain <= 1, "Uncertainty should be in [0, 1]"
    assert 0 <= unc_uncertain <= 1, "Uncertainty should be in [0, 1]"
    assert unc_uncertain > unc_certain, "Uniform distribution should have higher uncertainty"
    
    print("✅ PASSED")
    return True

def test_counterfactual():
    """Test counterfactual engine"""
    print("Testing Counterfactual Engine...", end=" ")
    
    from counterfactual import CounterfactualEngine
    
    engine = CounterfactualEngine()
    
    # Test future risk prediction
    position = [0, 0, -10]
    velocity = [5, 0, 0]
    
    future_risk = engine.predict_future_risk(position, velocity, None)
    
    assert 0 <= future_risk <= 1, "Future risk should be in [0, 1]"
    
    # Test scenario generation
    scenarios = engine.generate_alternative_scenarios(position, velocity)
    
    assert 'continue' in scenarios, "Should have 'continue' scenario"
    assert 'climb' in scenarios, "Should have 'climb' scenario"
    assert len(scenarios) >= 3, "Should have multiple scenarios"
    
    print("✅ PASSED")
    return True

def test_heatmap():
    """Test heatmap generation"""
    print("Testing Heatmap Engine...", end=" ")
    
    from heatmap_engine import HeatmapEngine
    
    engine = HeatmapEngine(grid_size=20, map_range=50)  # Smaller for speed
    
    # Test heatmap generation
    heatmap = engine.generate_heatmap(altitude=20)
    
    assert 'x' in heatmap, "Heatmap should contain x"
    assert 'y' in heatmap, "Heatmap should contain y"
    assert 'z' in heatmap, "Heatmap should contain z (risk values)"
    
    # Test zone identification
    safe_zones = engine.get_safe_zones(altitude=20, risk_threshold=0.3)
    high_risk_zones = engine.get_high_risk_zones(altitude=20, risk_threshold=0.7)
    
    assert isinstance(safe_zones, list), "Safe zones should be a list"
    assert isinstance(high_risk_zones, list), "High risk zones should be a list"
    
    print("✅ PASSED")
    return True

def test_state_machine():
    """Test state machine"""
    print("Testing State Machine...", end=" ")
    
    from state_machine import StateMachine, DroneState
    
    sm = StateMachine()
    
    assert sm.current_state == DroneState.TRANSIT, "Should start in TRANSIT"
    
    # Test transition to ADAPTIVE (high risk)
    sm.update(risk_score=0.8, uncertainty=0.2, risk_threshold=0.7)
    assert sm.current_state == DroneState.ADAPTIVE, "Should transition to ADAPTIVE on high risk"
    
    # Test transition to SURVEILLANCE (medium risk)
    sm.update(risk_score=0.5, uncertainty=0.2, risk_threshold=0.7)
    assert sm.current_state == DroneState.SURVEILLANCE, "Should transition to SURVEILLANCE"
    
    # Test transition to TRANSIT (low risk)
    sm.update(risk_score=0.2, uncertainty=0.2, risk_threshold=0.7)
    assert sm.current_state == DroneState.TRANSIT, "Should transition to TRANSIT on low risk"
    
    print("✅ PASSED")
    return True

def test_environment():
    """Test environment model"""
    print("Testing Environment Model...", end=" ")
    
    from environment import EnvironmentModel
    
    env = EnvironmentModel()
    
    # Test camera detection
    position = [10, 10, -20]
    altitude = 20
    
    detection_prob = env.camera.compute_detection_probability(position, altitude)
    
    assert 0 <= detection_prob <= 1, "Detection probability should be in [0, 1]"
    
    # Test zone proximity
    proximity = env.get_zone_proximity(position)
    
    assert 'tree' in proximity, "Should have tree proximity"
    assert 'building' in proximity, "Should have building proximity"
    
    print("✅ PASSED")
    return True

def test_integration():
    """Test integrated system flow"""
    print("Testing Integrated System...", end=" ")
    
    from feature_extractor import FeatureExtractor
    from lstm_model import IntentClassifier
    from risk_engine import RiskEngine
    from uncertainty import UncertaintyEstimator
    from counterfactual import CounterfactualEngine
    from state_machine import StateMachine
    
    # Create components
    extractor = FeatureExtractor()
    classifier = IntentClassifier()
    risk_engine = RiskEngine()
    uncertainty_est = UncertaintyEstimator()
    cf_engine = CounterfactualEngine()
    state_machine = StateMachine()
    
    # Simulate one cycle
    trajectory = [[i, 0, -10] for i in range(20)]
    telemetry = {'speed': 5.0, 'altitude': 10}
    position = [20, 0, -10]
    velocity = [5, 0, 0]
    
    # Extract features
    features = extractor.extract(trajectory, telemetry)
    
    # Classify intent
    intent_result = classifier.predict(features)
    
    # Estimate uncertainty
    uncertainty = uncertainty_est.compute(intent_result['probabilities'])
    
    # Compute risk
    risk_result = risk_engine.compute_risk(position, trajectory, telemetry['altitude'])
    
    # Predict counterfactual
    cf_risk = cf_engine.predict_future_risk(position, velocity, None)
    
    # Update state machine
    state_machine.update(risk_result['risk_score'], uncertainty, 0.7)
    
    # Verify all outputs
    assert intent_result['label'] in ['TRANSIT', 'SURVEILLANCE', 'ADAPTIVE']
    assert 0 <= uncertainty <= 1
    assert 0 <= risk_result['risk_score'] <= 1
    assert 0 <= cf_risk <= 1
    
    print("✅ PASSED")
    return True

def main():
    """Run all tests"""
    print("=" * 60)
    print("INTENTRA-X System Test Suite")
    print("=" * 60)
    print()
    
    tests = [
        test_feature_extraction,
        test_lstm_model,
        test_risk_engine,
        test_uncertainty,
        test_counterfactual,
        test_heatmap,
        test_state_machine,
        test_environment,
        test_integration
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ FAILED: {e}")
            failed += 1
    
    print()
    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)
    
    if failed == 0:
        print("\n🎉 All tests passed! System is ready.")
        return 0
    else:
        print(f"\n⚠️  {failed} test(s) failed. Please check the errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
