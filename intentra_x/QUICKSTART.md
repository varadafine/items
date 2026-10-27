# INTENTRA-X Quick Start Guide

## 🚀 Get Running in 3 Minutes

### Step 1: Install Dependencies (1 minute)

```bash
pip install -r requirements.txt
```

Or use requirements file:
```bash
cd intentra_x
pip install -r requirements.txt
```

### Step 2: Test System (30 seconds)

```bash
cd intentra_x
python test_system.py
```

You should see:
```
✅ Testing Feature Extraction... PASSED
✅ Testing LSTM Model... PASSED
✅ Testing Risk Engine... PASSED
...
🎉 All tests passed! System is ready.
```

### Step 3: Run Demo (30 seconds)

```bash
python run_demo.py
```

Or manually:
```bash
streamlit run dashboard.py
```

Then in the sidebar, select **"Demo"** mode.

---

## 🎮 Demo Mode Features

The demo mode runs WITHOUT AirSim and shows:

- ✅ Live dashboard with synthetic data
- ✅ Animated drone trajectory
- ✅ Risk heatmap visualization
- ✅ Intent classification
- ✅ Risk assessment
- ✅ Counterfactual predictions
- ✅ All UI components

**Perfect for:**
- Testing the system
- Demonstrating capabilities
- Understanding the interface
- Competition presentations

---

## 🛸 Live Mode (with AirSim)

### Prerequisites

1. **Download AirSim**
   - Windows: https://github.com/microsoft/AirSim/releases
   - Or build from source

2. **Configure AirSim**
   - Create `settings.json` in `Documents/AirSim/`
   - Use Multirotor mode

Example `settings.json`:
```json
{
  "SettingsVersion": 1.2,
  "SimMode": "Multirotor",
  "ClockSpeed": 1
}
```

### Running Live Mode

**Terminal 1 - AirSim:**
```bash
# Launch AirSim executable
# Or run in Unreal Engine
```

**Terminal 2 - Brain:**
```bash
cd intentra_x
python main_brain.py
```

You should see:
```
✓ Connected to AirSim
Taking off...
✓ Airborne
🧠 INTENTRA-X Brain Active
```

**Terminal 3 - Dashboard:**
```bash
cd intentra_x
streamlit run dashboard.py
```

In sidebar, select **"Live"** mode.

---

## 📊 Dashboard Controls

### Sidebar Options

- **Mode**: Live / Demo
- **Auto-refresh**: Enable real-time updates
- **Refresh rate**: 0.5 - 5.0 seconds

### What You'll See

**Left Panel:**
- Altitude gauge (0-50m)
- Speed gauge (0-15 m/s)
- Intent classification
- Uncertainty meter

**Center Map:**
- Drone position (yellow diamond)
- Trajectory trail (cyan line)
- Risk heatmap (red = high risk)
- Camera position (red square)
- Environment zones

**Right Panel:**
- Risk score meter
- Risk factor breakdown
- Counterfactual projection
- State transition log

**Bottom Graphs:**
- Risk over time
- Intent confidence over time
- Altitude over time

---

## 🧪 Testing Individual Components

### Test Feature Extraction
```python
from feature_extractor import FeatureExtractor

extractor = FeatureExtractor()
trajectory = [[i, 0, -10] for i in range(20)]
telemetry = {'speed': 5.0, 'altitude': 10}

features = extractor.extract(trajectory, telemetry)
print(f"Features: {features}")
```

### Test Intent Classification
```python
from lstm_model import IntentClassifier
import numpy as np

classifier = IntentClassifier()
features = np.random.randn(8)

result = classifier.predict(features)
print(f"Intent: {result['label']}")
print(f"Confidence: {result['confidence']:.2%}")
```

### Test Risk Assessment
```python
from risk_engine import RiskEngine

engine = RiskEngine()
position = [10, 10, -20]
trajectory = [[10, 10, -20]] * 10

risk = engine.compute_risk(position, trajectory, 20)
print(f"Risk: {risk['risk_score']:.2%}")
print("Explanation:")
for exp in risk['explanation']:
    print(f"  {exp}")
```

---

## 🎯 Competition Demo Script

### 1. Introduction (30 seconds)
"INTENTRA-X is a cognitive airspace intelligence system that combines autonomous drone control with AI-powered risk assessment."

### 2. Show Dashboard (1 minute)
- Point out real-time telemetry
- Highlight risk heatmap
- Show intent classification
- Explain state transitions

### 3. Demonstrate Intelligence (2 minutes)
- Show drone in TRANSIT state
- Watch it transition to SURVEILLANCE
- Point out risk increase
- Show ADAPTIVE behavior activation
- Highlight counterfactual predictions

### 4. Explain Ethics (1 minute)
"The system is designed for SAFETY and AWARENESS, not evasion. All decisions are transparent and explainable."

### 5. Show Technical Depth (1 minute)
- LSTM neural network
- Multi-factor risk engine
- Uncertainty quantification
- Counterfactual reasoning

---

## 🐛 Troubleshooting

### "ModuleNotFoundError"
```bash
pip install -r requirements.txt
```

### "AirSim connection failed"
- Check AirSim is running
- Verify Multirotor mode in settings.json
- Try restarting AirSim

### Dashboard won't load
```bash
# Try different port
streamlit run dashboard.py --server.port 8502
```

### "No live data" message
- This is normal without AirSim
- Switch to Demo mode in sidebar
- Or start main_brain.py first

---

## 📈 Performance Tips

### For Smooth Demo
- Use Demo mode (no AirSim overhead)
- Set refresh rate to 1.0s
- Close other browser tabs

### For Live Mode
- Reduce refresh rate if laggy
- Check AirSim performance
- Monitor CPU usage

---

## 🎓 Learning Path

1. **Start with Demo Mode** - Understand the interface
2. **Run Tests** - See how components work
3. **Read Code** - Start with `main_brain.py`
4. **Modify Parameters** - Experiment with risk thresholds
5. **Try Live Mode** - Connect to AirSim
6. **Analyze Logs** - Review recorded flights

---

## 📚 Key Files to Understand

1. `main_brain.py` - Core control loop
2. `dashboard.py` - UI and visualization
3. `lstm_model.py` - Intent classification
4. `risk_engine.py` - Risk assessment logic
5. `state_machine.py` - Behavioral states

---

## 🏆 Competition Checklist

- ✅ System runs without errors
- ✅ Dashboard displays correctly
- ✅ All gauges show data
- ✅ Map renders with heatmap
- ✅ Intent classification works
- ✅ Risk assessment functional
- ✅ State transitions occur
- ✅ Explanations are clear
- ✅ Demo mode works standalone
- ✅ Code is well-commented

---

## 💡 Pro Tips

1. **Use Demo Mode for Presentations** - More reliable than live AirSim
2. **Adjust Risk Threshold** - Change behavior sensitivity
3. **Monitor State Transitions** - Shows intelligence in action
4. **Explain Counterfactuals** - Unique feature, highlight it
5. **Emphasize Ethics** - Built-in transparency and safety

---

## 🎬 Ready to Go!

You now have a fully functional autonomous drone intelligence system.

**Next Steps:**
- Explore the code
- Customize behaviors
- Add new features
- Win the competition! 🏆

---

**Questions? Check README.md for detailed documentation.**
