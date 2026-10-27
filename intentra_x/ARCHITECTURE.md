# INTENTRA-X System Architecture

## 🏗️ High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         INTENTRA-X SYSTEM                        │
└─────────────────────────────────────────────────────────────────┘

┌──────────────┐         ┌──────────────┐         ┌──────────────┐
│   AirSim     │◄────────┤  Main Brain  ├────────►│  Dashboard   │
│  Simulator   │         │   (Control)  │         │  (Streamlit) │
└──────────────┘         └──────┬───────┘         └──────────────┘
                                │
                    ┌───────────┼───────────┐
                    │           │           │
            ┌───────▼────┐ ┌───▼────┐ ┌───▼────────┐
            │  Feature   │ │  LSTM  │ │    Risk    │
            │ Extractor  │ │ Model  │ │   Engine   │
            └────────────┘ └────────┘ └────────────┘
                    │           │           │
                    └───────────┼───────────┘
                                │
                    ┌───────────▼───────────┐
                    │   State Machine       │
                    │  (Decision Logic)     │
                    └───────────────────────┘
```

---

## 🔄 Data Flow

### 1. Perception → Intelligence → Action Loop

```
┌─────────────────────────────────────────────────────────────┐
│                    MAIN CONTROL LOOP                         │
│                    (2 Hz / 0.5s cycle)                       │
└─────────────────────────────────────────────────────────────┘

    ┌──────────────────────────────────────────────┐
    │  1. GET TELEMETRY                            │
    │     - Position (x, y, z)                     │
    │     - Velocity (vx, vy, vz)                  │
    │     - Altitude, Yaw, Speed                   │
    └──────────────┬───────────────────────────────┘
                   │
    ┌──────────────▼───────────────────────────────┐
    │  2. EXTRACT FEATURES                         │
    │     - Speed variance                         │
    │     - Altitude variance                      │
    │     - Path curvature                         │
    │     - Loiter ratio                           │
    │     - Direction changes                      │
    │     - Altitude trend                         │
    │     → 8D feature vector                      │
    └──────────────┬───────────────────────────────┘
                   │
    ┌──────────────▼───────────────────────────────┐
    │  3. CLASSIFY INTENT (LSTM)                   │
    │     Input: 8D features                       │
    │     Output: [P(Transit), P(Surv), P(Adapt)]  │
    │     → Intent label + confidence              │
    └──────────────┬───────────────────────────────┘
                   │
    ┌──────────────▼───────────────────────────────┐
    │  4. ESTIMATE UNCERTAINTY                     │
    │     - Shannon entropy                        │
    │     - Confidence margin                      │
    │     - Variance                               │
    │     → Uncertainty score (0-1)                │
    └──────────────┬───────────────────────────────┘
                   │
    ┌──────────────▼───────────────────────────────┐
    │  5. COMPUTE RISK                             │
    │     - Camera detection probability           │
    │     - Altitude exposure                      │
    │     - Loiter duration                        │
    │     - Zone proximity                         │
    │     - Trajectory predictability              │
    │     → Risk score (0-1) + breakdown           │
    └──────────────┬───────────────────────────────┘
                   │
    ┌──────────────▼───────────────────────────────┐
    │  6. PREDICT COUNTERFACTUAL                   │
    │     - Simulate future position (5s)          │
    │     - Compute projected risk                 │
    │     → Future risk estimate                   │
    └──────────────┬───────────────────────────────┘
                   │
    ┌──────────────▼───────────────────────────────┐
    │  7. STATE MACHINE DECISION                   │
    │     IF risk > threshold OR uncertainty high: │
    │        → ADAPTIVE                            │
    │     ELIF risk > 0.4 AND uncertainty low:     │
    │        → SURVEILLANCE                        │
    │     ELSE:                                    │
    │        → TRANSIT                             │
    └──────────────┬───────────────────────────────┘
                   │
    ┌──────────────▼───────────────────────────────┐
    │  8. EXECUTE BEHAVIOR                         │
    │     TRANSIT: Straight movement               │
    │     SURVEILLANCE: Circular loiter            │
    │     ADAPTIVE: Risk mitigation                │
    └──────────────┬───────────────────────────────┘
                   │
    ┌──────────────▼───────────────────────────────┐
    │  9. LOG & OUTPUT                             │
    │     - Save to live_output.json               │
    │     - Log to CSV/JSON files                  │
    │     - Update dashboard                       │
    └──────────────────────────────────────────────┘
                   │
                   └──────► REPEAT
```

---

## 🧠 Intelligence Components

### Feature Extractor

```
Input: Trajectory (list of positions) + Current telemetry
       │
       ├─► Speed variance ────────────┐
       ├─► Altitude variance ─────────┤
       ├─► Path curvature ────────────┤
       ├─► Loiter ratio ──────────────┤──► 8D Feature Vector
       ├─► Direction changes ─────────┤
       ├─► Altitude trend ────────────┤
       ├─► Current speed ─────────────┤
       └─► Current altitude ───────────┘
```

### LSTM Intent Classifier

```
Architecture:
┌─────────────────────────────────────┐
│  Input Layer (8 features)           │
└──────────────┬──────────────────────┘
               │
┌──────────────▼──────────────────────┐
│  LSTM Layer (32 hidden units)       │
│  - Processes sequential patterns    │
│  - Captures temporal dependencies   │
└──────────────┬──────────────────────┘
               │
┌──────────────▼──────────────────────┐
│  Fully Connected Layer              │
│  - Maps to 3 classes                │
└──────────────┬──────────────────────┘
               │
┌──────────────▼──────────────────────┐
│  Softmax Layer                      │
│  - Outputs probabilities            │
│  - [P(Transit), P(Surv), P(Adapt)]  │
└─────────────────────────────────────┘

Output: Intent label + confidence + probabilities
```

### Risk Engine

```
Multi-Factor Risk Assessment:

┌─────────────────────────────────────────────────┐
│  Factor 1: Camera Detection (35% weight)        │
│  - Distance to camera                           │
│  - Within FOV cone                              │
│  - Altitude factor                              │
│  - Occlusion (trees)                            │
└─────────────────┬───────────────────────────────┘
                  │
┌─────────────────▼───────────────────────────────┐
│  Factor 2: Altitude Exposure (20% weight)       │
│  - Higher altitude = more visible               │
│  - Normalized to [0, 1]                         │
└─────────────────┬───────────────────────────────┘
                  │
┌─────────────────▼───────────────────────────────┐
│  Factor 3: Loiter Duration (20% weight)         │
│  - Time spent in small area                     │
│  - Centroid-based calculation                   │
└─────────────────┬───────────────────────────────┘
                  │
┌─────────────────▼───────────────────────────────┐
│  Factor 4: Zone Proximity (15% weight)          │
│  - Distance to building                         │
│  - Distance to tree zone                        │
└─────────────────┬───────────────────────────────┘
                  │
┌─────────────────▼───────────────────────────────┐
│  Factor 5: Predictability (10% weight)          │
│  - Trajectory linearity                         │
│  - Pattern regularity                           │
└─────────────────┬───────────────────────────────┘
                  │
┌─────────────────▼───────────────────────────────┐
│  Weighted Combination → Risk Score (0-1)        │
│  + Breakdown dictionary                         │
│  + Human-readable explanation                   │
└─────────────────────────────────────────────────┘
```

### Uncertainty Estimator

```
Input: Probability distribution [p1, p2, p3]
       │
       ├─► Shannon Entropy ────────────┐
       │    H = -Σ(p * log(p))         │
       │    Normalized to [0, 1]       │
       │                               │
       ├─► Confidence Margin ──────────┤──► Combined
       │    margin = p_max - p_2nd     │    Uncertainty
       │    uncertainty = 1 - margin   │    Score (0-1)
       │                               │
       └─► Variance ───────────────────┘
            var = Σ((p - mean)²)
            Scaled to [0, 1]

High entropy + Low margin + High variance = High uncertainty
```

### Counterfactual Engine

```
Current State:
  - Position: [x, y, z]
  - Velocity: [vx, vy, vz]
  - Behavior: TRANSIT/SURVEILLANCE/ADAPTIVE
       │
       ├─► Simulate Future (5s horizon)
       │    future_pos = pos + vel * 5
       │
       ├─► Compute Future Risk
       │    - Detection probability at future_pos
       │    - Trajectory predictability
       │    - Altitude factor
       │
       └─► Generate Alternative Scenarios
            ├─► Continue current path
            ├─► Climb (increase altitude)
            ├─► Slow down (reduce speed)
            └─► Turn 90° (change direction)
                 │
                 └─► Recommend safest option
```

---

## 🎮 State Machine

```
┌─────────────────────────────────────────────────────────┐
│                   STATE TRANSITIONS                      │
└─────────────────────────────────────────────────────────┘

                    ┌──────────────┐
                    │   TRANSIT    │
                    │ (Low Risk)   │
                    └──────┬───────┘
                           │
        ┌──────────────────┼──────────────────┐
        │                  │                  │
   Risk > 0.4         Risk < 0.4         Risk > 0.7
   Unc < 0.4          Unc < 0.6          OR Unc > 0.6
        │                  │                  │
        ▼                  ▼                  ▼
┌───────────────┐  ┌──────────────┐  ┌──────────────┐
│ SURVEILLANCE  │  │   TRANSIT    │  │   ADAPTIVE   │
│ (Medium Risk) │  │  (Low Risk)  │  │  (High Risk) │
└───────┬───────┘  └──────────────┘  └──────┬───────┘
        │                                    │
        └────────────────┬───────────────────┘
                         │
                    Risk changes
                         │
                         ▼
                  Re-evaluate state

Transition Logic:
- IF risk > threshold OR uncertainty > 0.6:
    → ADAPTIVE (risk mitigation)
- ELIF risk > 0.4 AND uncertainty < 0.4:
    → SURVEILLANCE (observation mode)
- ELSE:
    → TRANSIT (normal movement)
```

---

## 🗺️ Environment Model

```
┌─────────────────────────────────────────────────────────┐
│                  ENVIRONMENT LAYOUT                      │
└─────────────────────────────────────────────────────────┘

        Y
        ▲
        │
   50   │     [Tree Zone]
        │    (20, 30) R=10
        │        🌳
        │
    0   ├────────[Building]────────► X
        │      (-30 to 30)
        │      (-30 to 30)
        │         🏢
        │      📷 Camera
  -50   │     (10, 10, 15)
        │
        └────────────────────────────►
       -50      0        50

Camera Model:
- Position: (10, 10, 15)
- FOV: 90 degrees
- Range: 100 meters
- Direction: Northeast (1, 1, 0)

Detection Probability:
  P(detect) = 0.4 * distance_factor
            + 0.3 * fov_factor
            + 0.2 * altitude_factor
            + 0.1 * (1 - occlusion_factor)

Zones:
- Tree Zone: Provides 60% occlusion
- Building Zone: High-risk proximity
- Open Zone: Low occlusion, high exposure
```

---

## 📊 Dashboard Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    STREAMLIT DASHBOARD                   │
└─────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────┐
│  HEADER: Mission ID | Status | Current State            │
└──────────────────────────────────────────────────────────┘

┌─────────────┬──────────────────────────┬────────────────┐
│   LEFT      │        CENTER            │     RIGHT      │
│  COLUMN     │        COLUMN            │    COLUMN      │
├─────────────┼──────────────────────────┼────────────────┤
│             │                          │                │
│ Altitude    │   2D Tactical Map        │  Risk Meter    │
│  Gauge      │   - Heatmap overlay      │   (0-100%)     │
│             │   - Trajectory trail     │                │
│ Speed       │   - Drone position       │  Risk          │
│  Gauge      │   - Camera location      │  Breakdown     │
│             │   - Zone boundaries      │                │
│ Intent      │                          │  Counter-      │
│ Display     │   Plotly Interactive     │  factual       │
│ + Conf.     │   - Pan/Zoom             │  Projection    │
│             │   - Hover info           │                │
│ Uncertainty │                          │  State         │
│  Meter      │                          │  Transition    │
│             │                          │  Log           │
└─────────────┴──────────────────────────┴────────────────┘

┌──────────────────────────────────────────────────────────┐
│  EXPLANATION PANEL: Risk factors + recommendations       │
└──────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────┐
│  TIME SERIES GRAPHS (3 columns)                          │
│  - Risk vs Time                                          │
│  - Intent Confidence vs Time                             │
│  - Altitude vs Time                                      │
└──────────────────────────────────────────────────────────┘

Data Flow:
  main_brain.py → live_output.json → dashboard.py
                   (0.5s cycle)      (1s refresh)
```

---

## 🔐 Ethical Design Principles

```
┌─────────────────────────────────────────────────────────┐
│              ETHICAL FRAMEWORK INTEGRATION               │
└─────────────────────────────────────────────────────────┘

1. TRANSPARENCY
   ├─► All decisions logged
   ├─► Risk factors explained
   ├─► Confidence scores provided
   └─► Full audit trail

2. EXPLAINABILITY
   ├─► Human-readable explanations
   ├─► Visual risk breakdown
   ├─► State transition reasons
   └─► Counterfactual scenarios

3. SAFETY-FIRST
   ├─► Risk-aware behavior
   ├─► Uncertainty consideration
   ├─► Adaptive responses
   └─► No adversarial logic

4. ACCOUNTABILITY
   ├─► Complete telemetry logging
   ├─► Decision history
   ├─► Performance metrics
   └─► Reproducible results

5. AWARENESS NOT EVASION
   ├─► Risk assessment (not avoidance)
   ├─► Spatial understanding
   ├─► Probabilistic reasoning
   └─► Decision support tool
```

---

## 📈 Performance Characteristics

```
Component Performance:

Feature Extraction:     < 10ms
LSTM Inference:         < 5ms
Risk Computation:       < 15ms
Uncertainty Estimate:   < 2ms
Counterfactual:         < 5ms
State Machine:          < 1ms
Total Cycle Time:       ~30ms

System Throughput:      2 Hz (0.5s cycle)
Dashboard Refresh:      1 Hz (configurable)
Trajectory History:     100 points
Heatmap Resolution:     50x50 grid
Map Coverage:           ±100m

Memory Usage:
- Base system:          ~200 MB
- With PyTorch:         ~500 MB
- Dashboard:            ~300 MB
- Total:                ~1 GB
```

---

## 🔄 Module Dependencies

```
main_brain.py
├── airsim (AirSim client)
├── state_machine.py
├── feature_extractor.py
├── lstm_model.py (PyTorch)
├── risk_engine.py
│   └── environment.py
├── uncertainty.py
├── counterfactual.py
│   └── environment.py
└── telemetry_logger.py

dashboard.py
├── streamlit
├── plotly
├── heatmap_engine.py
│   └── environment.py
└── live_output.json (data source)

All modules are independent and testable
```

---

This architecture enables:
- ✅ Real-time operation
- ✅ Modular design
- ✅ Explainable AI
- ✅ Ethical transparency
- ✅ Competition-grade quality
