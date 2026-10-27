# INTENTRA-X
## Cognitive Airspace Intelligence & Risk-Aware Autonomous Drone System

![Status](https://img.shields.io/badge/status-MVP-green)
![Python](https://img.shields.io/badge/python-3.9--3.12-blue)
![License](https://img.shields.io/badge/license-Competition-orange)

---

## 🎯 Overview

INTENTRA-X is a competition-grade autonomous drone intelligence platform that combines:

- **AirSim Simulation** - Realistic drone physics and control
- **PyTorch LSTM** - Intent classification from behavioral features
- **Multi-Factor Risk Assessment** - Spatial reasoning and exposure analysis
- **Uncertainty Quantification** - Confidence-aware decision making
- **Counterfactual Reasoning** - Predictive risk projection
- **Mission-Grade Dashboard** - Professional UAV ground control interface

---

## 🛡️ Ethical Framework

**CRITICAL: This system is designed for SAFETY and AWARENESS**

- ✅ Risk assessment for decision support
- ✅ Transparent, explainable AI
- ✅ Full audit trail of decisions
- ✅ Educational and research purposes
- ❌ NO stealth or evasion capabilities
- ❌ NO adversarial training
- ❌ NO surveillance evasion logic

All code includes ethical design comments and transparency measures.

---

## 🏗️ Architecture

```
intentra_x/
│
├── main_brain.py          # Core control system + AirSim interface
├── dashboard.py           # Streamlit mission control dashboard
├── state_machine.py       # Behavioral state management
├── environment.py         # Spatial environment model
├── feature_extractor.py   # Time-series feature extraction
├── lstm_model.py          # PyTorch intent classifier
├── risk_engine.py         # Multi-factor risk assessment
├── uncertainty.py         # Prediction uncertainty estimation
├── counterfactual.py      # Future risk projection
├── heatmap_engine.py      # Spatial risk visualization
├── telemetry_logger.py    # Flight data recording
├── live_output.json       # Real-time data exchange
└── data/
    └── sample_flight.csv  # Demo data
```

---

## 🚀 Quick Start

### Prerequisites

1. **Python 3.9–3.12**
2. **AirSim** (for live mode)
   - Download from: https://github.com/microsoft/AirSim
   - Run in Unreal Engine environment

### Installation

```bash
# Install dependencies
pip install -r intentra_x/requirements.txt

# Verify installation
python -c "import torch; import streamlit; import plotly; print('Core dependencies installed')"
```

### Running the System

#### Option 1: Live Mode (with AirSim)

**Terminal 1 - Start AirSim:**
```bash
# Launch AirSim in Unreal Engine
# Use Multirotor mode
```

**Terminal 2 - Start Brain:**
```bash
cd intentra_x
python main_brain.py
```

**Terminal 3 - Start Dashboard:**
```bash
cd intentra_x
streamlit run dashboard.py
```

#### Option 2: Demo Mode (no AirSim required)

```bash
cd intentra_x
streamlit run dashboard.py
# Select "Demo" mode in sidebar
```

---

## 🎮 System Components

### 1. Main Brain (`main_brain.py`)

**Core intelligence loop:**
- Connects to AirSim MultirotorClient
- Executes behavioral scripts (Transit, Surveillance, Adaptive)
- Collects telemetry (position, velocity, altitude)
- Runs feature extraction → LSTM → risk assessment
- Makes state transition decisions
- Outputs to `live_output.json`

**Behaviors:**
- **TRANSIT**: Direct movement, constant speed
- **SURVEILLANCE**: Circular loitering pattern
- **ADAPTIVE**: Risk-aware mitigation behavior

### 2. Dashboard (`dashboard.py`)

**Mission-grade interface:**
- Real-time telemetry gauges (altitude, speed)
- 2D tactical map with risk heatmap
- Trajectory visualization
- Intent classification display
- Risk breakdown and explanation
- Counterfactual risk projection
- State transition log
- Time-series analysis

### 3. LSTM Intent Classifier (`lstm_model.py`)

**Neural network architecture:**
- Input: 8-dimensional feature vector
- 1-layer LSTM (32 hidden units)
- Output: 3-class softmax (Transit/Surveillance/Adaptive)
- Includes entropy-based uncertainty

**Features used:**
1. Speed variance
2. Altitude variance
3. Path curvature
4. Loiter ratio
5. Direction changes
6. Altitude trend
7. Current speed
8. Current altitude

### 4. Risk Engine (`risk_engine.py`)

**Multi-factor risk assessment:**
- Camera detection probability (FOV geometry)
- Altitude exposure factor
- Loiter duration risk
- Zone proximity (building/tree)
- Trajectory predictability

**Output:**
- Risk score (0-1)
- Factor breakdown
- Human-readable explanation

### 5. Uncertainty Estimator (`uncertainty.py`)

**Methods:**
- Shannon entropy
- Confidence margin
- Variance-based
- Optional Monte Carlo Dropout

### 6. Counterfactual Engine (`counterfactual.py`)

**Predictive reasoning:**
- Simulates future position (5s horizon)
- Computes projected risk
- Generates alternative scenarios
- Recommends safest action

### 7. Heatmap Engine (`heatmap_engine.py`)

**Spatial visualization:**
- 2D risk grid (50x50)
- Camera FOV overlay
- Environment zones
- Safe/high-risk zone identification

---

## 📊 Dashboard Guide

### Layout

**TOP BAR:**
- Mission ID and timestamp
- System status indicator
- Current behavioral state

**LEFT COLUMN:**
- Altitude gauge (0-50m)
- Speed gauge (0-15 m/s)
- Intent classification + confidence
- Uncertainty meter

**CENTER:**
- 2D tactical map
- Risk heatmap overlay
- Drone trajectory trail
- Current position marker
- Camera and zone boundaries

**RIGHT COLUMN:**
- Risk score meter
- Risk factor breakdown
- Counterfactual projection
- State transition log

**BOTTOM:**
- Risk vs time graph
- Intent confidence vs time
- Altitude vs time

**EXPLANATION PANEL:**
- Human-readable risk factors
- Recommendations

---

## 🧪 Testing

### Test Intent Classification

```python
from feature_extractor import FeatureExtractor
from lstm_model import IntentClassifier

extractor = FeatureExtractor()
classifier = IntentClassifier()

# Simulate trajectory
trajectory = [[i, 0, -10] for i in range(20)]
telemetry = {'speed': 5.0, 'altitude': 10}

features = extractor.extract(trajectory, telemetry)
result = classifier.predict(features)

print(f"Intent: {result['label']}")
print(f"Confidence: {result['confidence']:.2f}")
```

### Test Risk Assessment

```python
from risk_engine import RiskEngine

engine = RiskEngine()

position = [10, 10, -20]
trajectory = [[10, 10, -20]] * 10
altitude = 20

risk = engine.compute_risk(position, trajectory, altitude)

print(f"Risk Score: {risk['risk_score']:.2f}")
print("Explanation:")
for exp in risk['explanation']:
    print(f"  {exp}")
```

---

## 📈 Performance

**System Specifications:**
- Update rate: 2 Hz (0.5s cycle time)
- Trajectory history: 100 points
- Feature extraction: <10ms
- LSTM inference: <5ms
- Risk computation: <15ms
- Total latency: ~30ms

**Dashboard:**
- Refresh rate: 1 Hz (configurable)
- Heatmap resolution: 50x50 grid
- Map range: ±100m

---

## 🔧 Configuration

### Adjust Risk Threshold

Edit `main_brain.py`:
```python
self.risk_threshold = 0.7  # Default: 0.7
```

### Modify Behaviors

Edit behavior parameters in `main_brain.py`:
```python
self.transit_speed = 5.0              # m/s
self.surveillance_radius = 15.0       # meters
self.surveillance_altitude = -20.0    # NED coords
```

### Camera Configuration

Edit `environment.py`:
```python
self.position = np.array([10, 10, 15])  # Camera position
self.fov_degrees = 90                    # Field of view
self.max_range = 100                     # Detection range
```

---

## 📝 Data Logging

All flights are automatically logged to `intentra_x/logs/`:

- `session_YYYYMMDD_HHMMSS.json` - Full telemetry
- `session_YYYYMMDD_HHMMSS.csv` - Tabular data

**CSV Columns:**
- timestamp, pos_x, pos_y, pos_z
- vel_x, vel_y, vel_z
- altitude, speed
- intent, intent_confidence
- uncertainty, risk_score
- state, counterfactual_risk

---

## 🎓 Educational Use

This system demonstrates:

1. **Autonomous Systems**: State machines, behavior scripting
2. **Machine Learning**: LSTM networks, feature engineering
3. **Uncertainty Quantification**: Bayesian reasoning, entropy
4. **Risk Assessment**: Multi-factor analysis, spatial reasoning
5. **Counterfactual Reasoning**: Predictive modeling
6. **Human-AI Interaction**: Explainable AI, visualization

---

## 🏆 Competition Features

**Technical Excellence:**
- ✅ Complete end-to-end system
- ✅ Real-time operation
- ✅ Professional-grade UI
- ✅ Modular architecture
- ✅ Comprehensive logging

**Innovation:**
- ✅ Counterfactual risk projection
- ✅ Uncertainty-aware decisions
- ✅ Spatial risk heatmaps
- ✅ Multi-factor risk engine

**Ethics & Safety:**
- ✅ Transparent decision-making
- ✅ Explainable AI
- ✅ Safety-first design
- ✅ Full audit trail

---

## 🐛 Troubleshooting

### AirSim Connection Failed

```bash
# Check AirSim is running
# Verify settings.json has Multirotor mode
# Try: python -c "import airsim; client = airsim.MultirotorClient(); client.confirmConnection()"
```

### Dashboard Not Loading

```bash
# Check port 8501 is available
# Try: streamlit run dashboard.py --server.port 8502
```

### Import Errors

```bash
# Reinstall dependencies
pip install -r intentra_x/requirements.txt --upgrade
```

---

## 📚 References

- **AirSim**: https://github.com/microsoft/AirSim
- **PyTorch**: https://pytorch.org/
- **Streamlit**: https://streamlit.io/
- **Plotly**: https://plotly.com/python/

---

## 📄 License

Competition Use - Educational and Research Purposes

---

## 👥 Team

INTENTRA-X Development Team
Competition 2026

---

## 🚨 Disclaimer

This system is a research prototype for educational and competition purposes. It demonstrates AI safety concepts and risk-aware decision making. Not intended for operational deployment without extensive testing and validation.

**ETHICAL USE ONLY**

---

**Built with ❤️ for autonomous systems research**
