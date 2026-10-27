"""
INTENTRA-X Military-Grade Tactical Command Interface
Full-Spectrum UAV Mission Control Console

ETHICAL DESIGN:
- Full transparency of AI decisions
- Explainable risk assessment
- Military-grade operational visualization
"""

import streamlit as st
import plotly.graph_objects as go
import json
import numpy as np
from datetime import datetime, timedelta
import time

from heatmap_engine import HeatmapEngine
from environment import EnvironmentModel

# Full-screen military-grade configuration
st.set_page_config(
    page_title="INTENTRA-X TACTICAL COMMAND",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Military-grade tactical CSS with all enhancements
st.markdown("""
<style>
    /* Remove Streamlit defaults */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .stDeployButton {display: none;}
    section[data-testid="stSidebar"] {display: none;}
    
    /* Full-screen military grid background */
    .main {
        background-color: #0A0E14;
        background-image: 
            linear-gradient(rgba(0, 255, 136, 0.03) 1px, transparent 1px),
            linear-gradient(90deg, rgba(0, 255, 136, 0.03) 1px, transparent 1px);
        background-size: 30px 30px;
        padding: 0 !important;
        margin: 0 !important;
        overflow-x: hidden;
    }
    
    .block-container {
        padding: 0.5rem 1rem !important;
        margin: 0 !important;
        max-width: 100% !important;
    }
    
    /* Fix Streamlit column gaps */
    [data-testid="column"] {
        padding: 0 0.5rem !important;
    }
    
    div[data-testid="stVerticalBlock"] > div {
        gap: 0.5rem;
    }
    
    /* Typography - Military technical */
    @import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@300;400;600;700&family=Share+Tech+Mono&display=swap');
    
    * {
        font-family: 'Rajdhani', sans-serif;
        color: #00FF88;
    }
    
    /* Top command bar */
    .command-bar {
        position: fixed;
        top: 0;
        left: 0;
        right: 0;
        height: 60px;
        background: linear-gradient(180deg, rgba(10, 14, 20, 0.98) 0%, rgba(10, 14, 20, 0.95) 100%);
        border-bottom: 2px solid rgba(0, 255, 136, 0.3);
        backdrop-filter: blur(10px);
        z-index: 999999;
        display: flex;
        align-items: center;
        padding: 0 2%;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.5);
    }
    
    .command-section {
        flex: 1;
        display: flex;
        align-items: center;
        gap: 1.5rem;
    }
    
    .command-section.left { justify-content: flex-start; }
    .command-section.center { justify-content: center; }
    .command-section.right { justify-content: flex-end; }
    
    /* Alert banner */
    .alert-banner {
        position: fixed;
        top: 60px;
        left: 0;
        right: 0;
        height: 40px;
        background: linear-gradient(90deg, rgba(255, 68, 102, 0.9) 0%, rgba(255, 136, 0, 0.9) 100%);
        border-bottom: 2px solid #FF4466;
        z-index: 999998;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 0.9rem;
        font-weight: 700;
        letter-spacing: 2px;
        animation: pulse-alert 2s infinite;
        box-shadow: 0 4px 15px rgba(255, 68, 102, 0.4);
    }
    
    @keyframes pulse-alert {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.85; }
    }
    
    /* Floating tactical panels */
    .tactical-grid {
        position: fixed;
        top: 100px;
        left: 1%;
        right: 1%;
        bottom: 1%;
        display: grid;
        grid-template-columns: 23% 54% 23%;
        grid-template-rows: auto 1fr;
        gap: 1%;
        z-index: 100;
    }
    
    /* Panel styling with corner brackets */
    .tactical-panel {
        background: linear-gradient(135deg, rgba(10, 20, 30, 0.85) 0%, rgba(5, 15, 25, 0.85) 100%);
        border: 1px solid rgba(0, 255, 136, 0.3);
        border-radius: 3px;
        padding: 1rem;
        backdrop-filter: blur(15px);
        position: relative;
        overflow-y: auto;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.4);
    }
    
    /* Corner brackets effect */
    .tactical-panel::before,
    .tactical-panel::after {
        content: '';
        position: absolute;
        width: 15px;
        height: 15px;
        border: 2px solid rgba(0, 255, 136, 0.5);
    }
    
    .tactical-panel::before {
        top: 5px;
        left: 5px;
        border-right: none;
        border-bottom: none;
    }
    
    .tactical-panel::after {
        bottom: 5px;
        right: 5px;
        border-left: none;
        border-top: none;
    }
    
    .panel-raised {
        background: linear-gradient(135deg, rgba(15, 25, 35, 0.9) 0%, rgba(10, 20, 30, 0.9) 100%);
        border-color: rgba(0, 255, 136, 0.4);
        box-shadow: 0 0 30px rgba(0, 255, 136, 0.15);
    }
    
    /* Typography styles */
    h1, h2, h3, h4 {
        font-family: 'Rajdhani', sans-serif;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 3px;
        color: #00FF88;
        margin: 0 0 0.8rem 0;
    }
    
    h1 { font-size: 1.3rem; }
    h2 { font-size: 1.1rem; }
    h3 { font-size: 0.85rem; letter-spacing: 2px; }
    h4 { font-size: 0.75rem; letter-spacing: 1.5px; }
    
    .numeric-huge {
        font-family: 'Share Tech Mono', monospace;
        font-size: 4.5rem;
        font-weight: 700;
        letter-spacing: 5px;
        text-align: center;
        line-height: 1;
    }
    
    .numeric-large {
        font-family: 'Share Tech Mono', monospace;
        font-size: 2.8rem;
        font-weight: 600;
        letter-spacing: 4px;
    }
    
    .label-military {
        font-family: 'Rajdhani', sans-serif;
        font-size: 0.7rem;
        font-weight: 400;
        color: rgba(0, 255, 136, 0.6);
        text-transform: uppercase;
        letter-spacing: 2px;
    }
    
    .micro-data {
        font-family: 'Share Tech Mono', monospace;
        font-size: 0.65rem;
        color: rgba(255, 255, 255, 0.5);
        letter-spacing: 1px;
    }
    
    /* Status indicators */
    .status-led {
        display: inline-block;
        width: 10px;
        height: 10px;
        border-radius: 50%;
        margin-right: 8px;
        animation: pulse-led 2s infinite;
    }
    
    @keyframes pulse-led {
        0%, 100% { box-shadow: 0 0 8px currentColor; }
        50% { box-shadow: 0 0 15px currentColor; }
    }
    
    .led-active { background: #00FF88; color: #00FF88; }
    .led-warning { background: #FFB800; color: #FFB800; }
    .led-critical { background: #FF4466; color: #FF4466; }
    
    /* System health bars */
    .health-bar-container {
        background: rgba(0, 0, 0, 0.4);
        border: 1px solid rgba(0, 255, 136, 0.2);
        border-radius: 2px;
        height: 18px;
        margin-bottom: 0.5rem;
        overflow: hidden;
        position: relative;
    }
    
    .health-bar {
        height: 100%;
        transition: width 0.5s ease;
        position: relative;
        overflow: hidden;
    }
    
    .health-bar::after {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.2), transparent);
        animation: shimmer 2s infinite;
    }
    
    @keyframes shimmer {
        0% { transform: translateX(-100%); }
        100% { transform: translateX(100%); }
    }
    
    /* Vertical threat indicators */
    .threat-indicator {
        width: 100%;
        height: 120px;
        background: rgba(0, 0, 0, 0.4);
        border: 1px solid rgba(0, 255, 136, 0.2);
        border-radius: 2px;
        position: relative;
        overflow: hidden;
        margin-bottom: 0.8rem;
    }
    
    .threat-fill {
        position: absolute;
        bottom: 0;
        left: 0;
        right: 0;
        transition: height 0.5s ease;
        background: linear-gradient(180deg, transparent 0%, currentColor 100%);
    }
    
    .threat-value {
        position: absolute;
        top: 5px;
        left: 0;
        right: 0;
        text-align: center;
        font-family: 'Share Tech Mono', monospace;
        font-size: 1.1rem;
        font-weight: 700;
        z-index: 10;
    }
    
    /* Sparkline container */
    .sparkline {
        width: 100%;
        height: 25px;
        margin-top: 3px;
    }
    
    /* Timeline with timestamps */
    .timeline-enhanced {
        background: rgba(5, 15, 25, 0.7);
        border: 1px solid rgba(0, 255, 136, 0.2);
        border-radius: 2px;
        padding: 0.6rem;
        margin-top: 1rem;
    }
    
    .timeline-bar-enhanced {
        height: 35px;
        display: flex;
        border-radius: 2px;
        overflow: hidden;
        border: 1px solid rgba(0, 255, 136, 0.2);
        position: relative;
    }
    
    .timeline-segment-enhanced {
        height: 100%;
        border-right: 1px solid rgba(10, 14, 20, 0.5);
        position: relative;
        cursor: pointer;
        transition: all 0.3s ease;
    }
    
    .timeline-segment-enhanced:hover {
        filter: brightness(1.3);
        transform: scaleY(1.1);
    }
    
    .timeline-segment-enhanced.active {
        animation: pulse-segment 1.5s infinite;
        box-shadow: inset 0 0 10px rgba(255, 255, 255, 0.3);
    }
    
    @keyframes pulse-segment {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.7; }
    }
    
    /* Circular progress indicator */
    .circular-progress {
        position: relative;
        width: 140px;
        height: 140px;
        margin: 1rem auto;
    }
    
    .progress-ring {
        transform: rotate(-90deg);
    }
    
    .progress-ring-circle {
        transition: stroke-dashoffset 0.5s ease;
    }
    
    /* Log entries with icons */
    .log-container-enhanced {
        max-height: 220px;
        overflow-y: auto;
        background: rgba(0, 0, 0, 0.5);
        border: 1px solid rgba(0, 255, 136, 0.15);
        border-radius: 2px;
        padding: 0.5rem;
    }
    
    .log-entry-enhanced {
        font-family: 'Share Tech Mono', monospace;
        font-size: 0.7rem;
        padding: 0.4rem;
        margin-bottom: 0.3rem;
        background: rgba(0, 255, 136, 0.05);
        border-left: 3px solid;
        border-radius: 2px;
        display: flex;
        align-items: center;
        gap: 0.5rem;
        transition: all 0.2s ease;
    }
    
    .log-entry-enhanced:hover {
        background: rgba(0, 255, 136, 0.1);
        transform: translateX(3px);
    }
    
    .log-entry-enhanced.info { border-left-color: #00E5FF; }
    .log-entry-enhanced.warning { border-left-color: #FFB800; }
    .log-entry-enhanced.critical { border-left-color: #FF4466; }
    
    /* Trend indicators */
    .trend-indicator {
        display: inline-block;
        font-size: 1.2rem;
        margin-left: 0.5rem;
        animation: trend-pulse 1.5s infinite;
    }
    
    .trend-up { color: #FF4466; }
    .trend-down { color: #00FF88; }
    .trend-stable { color: #FFB800; }
    
    @keyframes trend-pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.6; }
    }
    
    /* Glow effects */
    .glow-green { 
        text-shadow: 0 0 15px rgba(0, 255, 136, 0.7);
        color: #00FF88;
    }
    .glow-amber { 
        text-shadow: 0 0 15px rgba(255, 184, 0, 0.7);
        color: #FFB800;
    }
    .glow-red { 
        text-shadow: 0 0 15px rgba(255, 68, 102, 0.7);
        color: #FF4466;
    }
    .glow-cyan { 
        text-shadow: 0 0 15px rgba(0, 229, 255, 0.7);
        color: #00E5FF;
    }
    
    /* Critical pulse */
    @keyframes pulse-critical {
        0%, 100% { 
            box-shadow: 0 0 20px rgba(255, 68, 102, 0.5);
            border-color: rgba(255, 68, 102, 0.6);
        }
        50% { 
            box-shadow: 0 0 40px rgba(255, 68, 102, 0.8);
            border-color: rgba(255, 68, 102, 0.9);
        }
    }
    
    .critical-pulse {
        animation: pulse-critical 2s infinite;
    }
    
    /* Scan line effect */
    @keyframes scan-line {
        0% { top: 0%; }
        100% { top: 100%; }
    }
    
    .scan-line-effect::after {
        content: '';
        position: absolute;
        left: 0;
        right: 0;
        height: 2px;
        background: linear-gradient(90deg, transparent, rgba(0, 255, 136, 0.5), transparent);
        animation: scan-line 3s linear infinite;
        pointer-events: none;
    }
    
    /* Scrollbar */
    ::-webkit-scrollbar { width: 6px; height: 6px; }
    ::-webkit-scrollbar-track { background: rgba(0, 0, 0, 0.3); }
    ::-webkit-scrollbar-thumb { 
        background: rgba(0, 255, 136, 0.4); 
        border-radius: 3px; 
    }
    ::-webkit-scrollbar-thumb:hover { 
        background: rgba(0, 255, 136, 0.6); 
    }
    
    /* Tooltip */
    .tooltip {
        position: relative;
        display: inline-block;
    }
    
    .tooltip .tooltiptext {
        visibility: hidden;
        background-color: rgba(10, 20, 30, 0.95);
        color: #00FF88;
        text-align: center;
        border-radius: 3px;
        padding: 5px 10px;
        position: absolute;
        z-index: 3000;
        bottom: 125%;
        left: 50%;
        transform: translateX(-50%);
        opacity: 0;
        transition: opacity 0.3s;
        border: 1px solid rgba(0, 255, 136, 0.3);
        font-size: 0.7rem;
        white-space: nowrap;
    }
    
    .tooltip:hover .tooltiptext {
        visibility: visible;
        opacity: 1;
    }
</style>
""", unsafe_allow_html=True)


class MilitaryTacticalHUD:
    """Military-grade tactical command interface"""
    
    def __init__(self):
        self.heatmap_engine = HeatmapEngine()
        self.environment = EnvironmentModel()
        self.mission_start_time = datetime.now()
        
        # Initialize session state for historical data
        if 'risk_history' not in st.session_state:
            st.session_state.risk_history = []
        if 'altitude_history' not in st.session_state:
            st.session_state.altitude_history = []
        if 'speed_history' not in st.session_state:
            st.session_state.speed_history = []
    
    def load_live_data(self):
        """Load live telemetry"""
        try:
            with open('intentra_x/live_output.json', 'r') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return None
    
    def load_sample_data(self):
        """Load enhanced sample data with all military-grade metrics"""
        t = time.time()
        angle = (t % 30) / 30 * 2 * np.pi
        
        # Simulate GPS coordinates (example: somewhere over a test range)
        base_lat, base_lon = 35.0522, -106.6056  # Example coordinates
        lat_offset = 15 * np.cos(angle) / 111000  # Convert meters to degrees
        lon_offset = 15 * np.sin(angle) / (111000 * np.cos(np.radians(base_lat)))
        
        return {
            'timestamp': t,
            'position': [15 * np.cos(angle), 15 * np.sin(angle), -20],
            'velocity': [-15 * np.sin(angle), 15 * np.cos(angle), 0],
            'altitude': 20 + np.sin(t * 0.5) * 2,  # Slight variation
            'speed': 5.0 + np.sin(t * 0.3) * 0.5,
            'heading': np.degrees(angle) % 360,
            'gps_lat': base_lat + lat_offset,
            'gps_lon': base_lon + lon_offset,
            'gps_accuracy': 2.5,
            'battery_level': max(65, 100 - (t % 300) / 3),  # Simulated drain
            'signal_strength': 85 + np.sin(t * 0.2) * 10,
            'datalink_quality': 92 + np.sin(t * 0.15) * 5,
            'datalink_latency': 45 + np.sin(t * 0.25) * 15,
            'trajectory': [[15 * np.cos(angle - i*0.1), 
                           15 * np.sin(angle - i*0.1), -20] 
                          for i in range(20)],
            'intent': 'SURVEILLANCE',
            'intent_confidence': 0.85,
            'intent_probabilities': [0.05, 0.85, 0.10],
            'uncertainty': 0.25,
            'risk_score': 0.45 + np.sin(t * 0.1) * 0.1,
            'risk_breakdown': {
                'detection_probability': 0.5,
                'altitude_risk': 0.4,
                'loiter_risk': 0.6,
                'proximity_risk': 0.3,
                'predictability_risk': 0.4
            },
            'risk_explanation': [
                "Elevated exposure level detected",
                "Loitering behavior in surveillance zone",
                "Moderate detection probability from ground sensors"
            ],
            'state': 'SURVEILLANCE',
            'counterfactual_risk': 0.52,
            'state_history': [
                {'from': 'TRANSIT', 'to': 'SURVEILLANCE', 'risk': 0.42, 
                 'timestamp': (datetime.now() - timedelta(seconds=120)).strftime('%H:%M:%S')},
                {'from': 'SURVEILLANCE', 'to': 'ADAPTIVE', 'risk': 0.68, 
                 'timestamp': (datetime.now() - timedelta(seconds=60)).strftime('%H:%M:%S')},
            ]
        }
    
    def update_history(self, data):
        """Update historical data for sparklines"""
        st.session_state.risk_history.append(data.get('risk_score', 0))
        st.session_state.altitude_history.append(data.get('altitude', 0))
        st.session_state.speed_history.append(data.get('speed', 0))
        
        # Keep last 60 data points
        if len(st.session_state.risk_history) > 60:
            st.session_state.risk_history.pop(0)
        if len(st.session_state.altitude_history) > 60:
            st.session_state.altitude_history.pop(0)
        if len(st.session_state.speed_history) > 60:
            st.session_state.speed_history.pop(0)
    
    def get_mission_elapsed_time(self):
        """Get formatted mission elapsed time"""
        elapsed = datetime.now() - self.mission_start_time
        hours = int(elapsed.total_seconds() // 3600)
        minutes = int((elapsed.total_seconds() % 3600) // 60)
        seconds = int(elapsed.total_seconds() % 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    
    def get_trend_indicator(self, current, history):
        """Calculate trend indicator"""
        if len(history) < 2:
            return "→", "stable"
        
        recent_avg = np.mean(history[-5:]) if len(history) >= 5 else history[-1]
        older_avg = np.mean(history[-10:-5]) if len(history) >= 10 else history[0]
        
        diff = recent_avg - older_avg
        if abs(diff) < 0.05:
            return "→", "stable"
        elif diff > 0:
            return "↑", "up"
        else:
            return "↓", "down"
    
    def render_command_bar(self, data):
        """Render top command bar with all system metrics"""
        status = "OPERATIONAL" if data else "NO SIGNAL"
        led_class = "led-active" if data else "led-critical"
        
        mission_time = self.get_mission_elapsed_time()
        
        if data:
            gps_lat = data.get('gps_lat', 0)
            gps_lon = data.get('gps_lon', 0)
            heading = data.get('heading', 0)
            battery = data.get('battery_level', 0)
            signal = data.get('signal_strength', 0)
            datalink = data.get('datalink_quality', 0)
            latency = data.get('datalink_latency', 0)
            state = data.get('state', 'UNKNOWN')
            risk = data.get('risk_score', 0)
        else:
            gps_lat = gps_lon = heading = battery = signal = datalink = latency = 0
            state = 'UNKNOWN'
            risk = 0
        
        state_colors = {
            'TRANSIT': '#00FF88',
            'SURVEILLANCE': '#FFB800',
            'ADAPTIVE': '#FF4466'
        }
        state_color = state_colors.get(state, '#00E5FF')
        
        # Battery color
        batt_color = '#00FF88' if battery > 50 else '#FFB800' if battery > 20 else '#FF4466'
        
        # Signal color
        sig_color = '#00FF88' if signal > 70 else '#FFB800' if signal > 40 else '#FF4466'
        
        # Use st.html for better rendering
        command_bar_html = f"""
<div class='command-bar'>
    <div class='command-section left'>
        <div>
            <span style='font-size: 1.3rem; font-weight: 600; letter-spacing: 3px; color: #00FF88;'>INTENTRA-X</span>
            <span class='label-military' style='margin-left: 1rem;'>TACTICAL COMMAND</span>
        </div>
        <div style='border-left: 1px solid rgba(0, 255, 136, 0.3); padding-left: 1rem;'>
            <div class='label-military'>MISSION TIME</div>
            <div style='font-family: Share Tech Mono; font-size: 1rem; font-weight: 600;'>{mission_time}</div>
        </div>
        <div style='border-left: 1px solid rgba(0, 255, 136, 0.3); padding-left: 1rem;'>
            <div class='label-military'>COORDINATES</div>
            <div style='font-family: Share Tech Mono; font-size: 0.8rem;'>{gps_lat:.6f}N {abs(gps_lon):.6f}W</div>
        </div>
        <div style='border-left: 1px solid rgba(0, 255, 136, 0.3); padding-left: 1rem;'>
            <div class='label-military'>HEADING</div>
            <div style='font-family: Share Tech Mono; font-size: 1rem; font-weight: 600;'>{heading:.0f}°</div>
        </div>
    </div>
    
    <div class='command-section center'>
        <div style='text-align: center;'>
            <span class='status-led {led_class}'></span>
            <span style='font-size: 1.1rem; font-weight: 700; letter-spacing: 2px;'>{status}</span>
        </div>
    </div>
    
    <div class='command-section right'>
        <div style='text-align: right;'>
            <div class='label-military'>BATTERY</div>
            <div style='font-family: Share Tech Mono; font-size: 1rem; font-weight: 600; color: {batt_color};'>{battery:.0f}%</div>
        </div>
        <div style='border-left: 1px solid rgba(0, 255, 136, 0.3); padding-left: 1rem; text-align: right;'>
            <div class='label-military'>SIGNAL</div>
            <div style='font-family: Share Tech Mono; font-size: 1rem; font-weight: 600; color: {sig_color};'>{signal:.0f}%</div>
        </div>
        <div style='border-left: 1px solid rgba(0, 255, 136, 0.3); padding-left: 1rem; text-align: right;'>
            <div class='label-military'>DATALINK</div>
            <div style='font-family: Share Tech Mono; font-size: 0.8rem;'>{datalink:.0f}% | {latency:.0f}ms</div>
        </div>
        <div style='border-left: 1px solid rgba(0, 255, 136, 0.3); padding-left: 1rem; text-align: right;'>
            <div style='color: {state_color}; font-size: 1.1rem; font-weight: 700; letter-spacing: 2px;'>{state}</div>
            <div class='label-military'>RISK: {risk:.0%}</div>
        </div>
    </div>
</div>
"""
        st.markdown(command_bar_html, unsafe_allow_html=True)
    
    def render_alert_banner(self, data):
        """Render critical alert banner if needed"""
        risk_score = data.get('risk_score', 0) if data else 0
        
        if risk_score > 0.7:
            alert_html = """
            <div class='alert-banner'>
                ⚠ CRITICAL ALERT: HIGH RISK EXPOSURE DETECTED - RECOMMEND IMMEDIATE EVASIVE ACTION ⚠
            </div>
            """
            st.markdown(alert_html, unsafe_allow_html=True)
    
    def create_sparkline(self, data, color="#00FF88", key_suffix=""):
        """Create small sparkline chart"""
        if len(data) < 2:
            return None
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            y=data,
            mode='lines',
            line=dict(color=color, width=1.5),
            fill='tozeroy',
            fillcolor=f'rgba{tuple(list(int(color[i:i+2], 16) for i in (1, 3, 5)) + [0.2])}',
            hoverinfo='skip'
        ))
        
        fig.update_layout(
            height=25,
            margin=dict(l=0, r=0, t=0, b=0),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            showlegend=False
        )
        
        return fig
    
    def create_gauge_with_trend(self, value, max_value, title, unit="", color="#00FF88", 
                                history=None, key_suffix=""):
        """Create gauge with trend indicator"""
        fig = go.Figure()
        
        # Background arc
        fig.add_trace(go.Scatterpolar(
            r=[0.75, 0.75],
            theta=[0, 270],
            mode='lines',
            line=dict(color='rgba(0, 255, 136, 0.1)', width=2),
            showlegend=False,
            hoverinfo='skip'
        ))
        
        # Value arc
        value_angle = (value / max_value) * 270
        fig.add_trace(go.Scatterpolar(
            r=[0.75, 0.75],
            theta=[0, value_angle],
            mode='lines',
            line=dict(color=color, width=3),
            showlegend=False,
            hoverinfo='skip'
        ))
        
        # Tick marks
        for angle in range(0, 271, 45):
            fig.add_trace(go.Scatterpolar(
                r=[0.7, 0.75],
                theta=[angle, angle],
                mode='lines',
                line=dict(color='rgba(0, 255, 136, 0.3)', width=1),
                showlegend=False,
                hoverinfo='skip'
            ))
        
        # Min/Max markers
        fig.add_annotation(
            x=0.15, y=0.5,
            text=f"0",
            showarrow=False,
            font=dict(size=8, color='rgba(0, 255, 136, 0.4)'),
            xref="paper", yref="paper"
        )
        fig.add_annotation(
            x=0.85, y=0.5,
            text=f"{max_value}",
            showarrow=False,
            font=dict(size=8, color='rgba(0, 255, 136, 0.4)'),
            xref="paper", yref="paper"
        )
        
        # Center value
        trend_symbol, trend_class = self.get_trend_indicator(value, history) if history else ("", "")
        fig.add_annotation(
            x=0.5, y=0.5,
            text=f"<b>{value:.1f}</b>{unit}",
            showarrow=False,
            font=dict(size=22, color=color, family='Share Tech Mono'),
            xref="paper", yref="paper"
        )
        
        # Title
        fig.add_annotation(
            x=0.5, y=0.15,
            text=title,
            showarrow=False,
            font=dict(size=8, color='rgba(0, 255, 136, 0.6)', family='Rajdhani'),
            xref="paper", yref="paper"
        )
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(visible=False, range=[0, 1]),
                angularaxis=dict(visible=False, rotation=135, direction='clockwise')
            ),
            showlegend=False,
            height=150,
            margin=dict(l=5, r=5, t=5, b=5),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        
        return fig, trend_symbol, trend_class
    
    def render_telemetry_panel(self, data):
        """Render enhanced telemetry section with trends"""
        st.markdown("<div class='tactical-panel'><h3>◈ TELEMETRY & SENSORS</h3>", 
                   unsafe_allow_html=True)
        
        # Altitude
        altitude = data.get('altitude', 0)
        fig_alt, trend_alt, trend_class_alt = self.create_gauge_with_trend(
            altitude, 50, "ALTITUDE", "m", "#00E5FF", 
            st.session_state.altitude_history, "alt"
        )
        st.plotly_chart(fig_alt, use_container_width=True, key="alt_gauge", 
                       config={'displayModeBar': False})
        
        trend_color = {'up': '#FF4466', 'down': '#00FF88', 'stable': '#FFB800'}
        st.markdown(f"<div style='text-align: center;'>"
                   f"<span class='micro-data'>RANGE: 0-50m</span>"
                   f"<span class='trend-indicator trend-{trend_class_alt}' "
                   f"style='color: {trend_color.get(trend_class_alt, '#FFB800')};'>{trend_alt}</span>"
                   f"</div>", unsafe_allow_html=True)
        
        # Sparkline for altitude
        if len(st.session_state.altitude_history) > 1:
            fig_spark_alt = self.create_sparkline(st.session_state.altitude_history, "#00E5FF", "alt")
            if fig_spark_alt:
                st.plotly_chart(fig_spark_alt, use_container_width=True, key="spark_alt",
                               config={'displayModeBar': False})
        
        st.markdown("<div style='margin: 1rem 0; border-top: 1px solid rgba(0, 255, 136, 0.2);'></div>",
                   unsafe_allow_html=True)
        
        # Velocity
        speed = data.get('speed', 0)
        fig_speed, trend_speed, trend_class_speed = self.create_gauge_with_trend(
            speed, 15, "VELOCITY", " m/s", "#00FF88",
            st.session_state.speed_history, "speed"
        )
        st.plotly_chart(fig_speed, use_container_width=True, key="speed_gauge",
                       config={'displayModeBar': False})
        
        st.markdown(f"<div style='text-align: center;'>"
                   f"<span class='micro-data'>MAX: 15 m/s</span>"
                   f"<span class='trend-indicator trend-{trend_class_speed}' "
                   f"style='color: {trend_color.get(trend_class_speed, '#FFB800')};'>{trend_speed}</span>"
                   f"</div>", unsafe_allow_html=True)
        
        # Sparkline for speed
        if len(st.session_state.speed_history) > 1:
            fig_spark_speed = self.create_sparkline(st.session_state.speed_history, "#00FF88", "speed")
            if fig_spark_speed:
                st.plotly_chart(fig_spark_speed, use_container_width=True, key="spark_speed",
                               config={'displayModeBar': False})
        
        st.markdown("<div style='margin: 1rem 0; border-top: 1px solid rgba(0, 255, 136, 0.2);'></div>",
                   unsafe_allow_html=True)
        
        # GPS Accuracy
        gps_acc = data.get('gps_accuracy', 0)
        st.markdown(f"<h4>GPS ACCURACY</h4>", unsafe_allow_html=True)
        st.markdown(f"<div style='text-align: center; font-family: Share Tech Mono; "
                   f"font-size: 1.5rem; font-weight: 600; color: #00E5FF;'>{gps_acc:.1f}m</div>",
                   unsafe_allow_html=True)
        st.markdown(f"<div class='micro-data' style='text-align: center;'>HDOP: EXCELLENT</div>",
                   unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    def render_system_health_panel(self, data):
        """Render system health indicators"""
        st.markdown("<div class='tactical-panel'><h3>◈ SYSTEM HEALTH</h3>", 
                   unsafe_allow_html=True)
        
        battery = data.get('battery_level', 0)
        signal = data.get('signal_strength', 0)
        datalink = data.get('datalink_quality', 0)
        
        # Battery
        batt_color = '#00FF88' if battery > 50 else '#FFB800' if battery > 20 else '#FF4466'
        st.markdown(f"<div class='label-military'>BATTERY LEVEL</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='health-bar-container'>"
                   f"<div class='health-bar' style='width: {battery}%; background: {batt_color};'></div>"
                   f"</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='micro-data'>{battery:.0f}% | EST. {int(battery * 0.3)} MIN REMAINING</div>",
                   unsafe_allow_html=True)
        
        # Signal Strength
        sig_color = '#00FF88' if signal > 70 else '#FFB800' if signal > 40 else '#FF4466'
        st.markdown(f"<div class='label-military' style='margin-top: 0.8rem;'>GPS SIGNAL</div>", 
                   unsafe_allow_html=True)
        st.markdown(f"<div class='health-bar-container'>"
                   f"<div class='health-bar' style='width: {signal}%; background: {sig_color};'></div>"
                   f"</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='micro-data'>{signal:.0f}% | {int(signal/10)} SATELLITES</div>",
                   unsafe_allow_html=True)
        
        # Datalink Quality
        link_color = '#00FF88' if datalink > 80 else '#FFB800' if datalink > 50 else '#FF4466'
        st.markdown(f"<div class='label-military' style='margin-top: 0.8rem;'>DATALINK QUALITY</div>",
                   unsafe_allow_html=True)
        st.markdown(f"<div class='health-bar-container'>"
                   f"<div class='health-bar' style='width: {datalink}%; background: {link_color};'></div>"
                   f"</div>", unsafe_allow_html=True)
        latency = data.get('datalink_latency', 0)
        st.markdown(f"<div class='micro-data'>{datalink:.0f}% | LATENCY: {latency:.0f}ms</div>",
                   unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    def render_intent_classification_panel(self, data):
        """Render intent classification with confidence"""
        st.markdown("<div class='tactical-panel'><h3>◈ INTENT CLASSIFICATION</h3>",
                   unsafe_allow_html=True)
        
        intent = data.get('intent', 'UNKNOWN')
        confidence = data.get('intent_confidence', 0)
        uncertainty = data.get('uncertainty', 0)
        
        intent_colors = {
            'TRANSIT': '#00FF88',
            'SURVEILLANCE': '#FFB800',
            'ADAPTIVE': '#FF4466'
        }
        color = intent_colors.get(intent, '#00E5FF')
        
        st.markdown(f"<div style='text-align: center; color: {color}; "
                   f"font-size: 1.8rem; font-weight: 700; letter-spacing: 3px; "
                   f"margin: 1rem 0; text-shadow: 0 0 15px {color};'>{intent}</div>",
                   unsafe_allow_html=True)
        
        # Confidence bar
        st.markdown(f"<div class='label-military'>MODEL CONFIDENCE</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='health-bar-container'>"
                   f"<div class='health-bar' style='width: {confidence*100}%; background: {color};'></div>"
                   f"</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='micro-data'>{confidence:.0%}</div>", unsafe_allow_html=True)
        
        # Uncertainty
        unc_color = '#FF4466' if uncertainty > 0.6 else '#FFB800' if uncertainty > 0.3 else '#00FF88'
        st.markdown(f"<div class='label-military' style='margin-top: 0.8rem;'>UNCERTAINTY METRIC</div>",
                   unsafe_allow_html=True)
        st.markdown(f"<div class='health-bar-container'>"
                   f"<div class='health-bar' style='width: {uncertainty*100}%; background: {unc_color};'></div>"
                   f"</div>", unsafe_allow_html=True)
        
        status = "STABLE" if uncertainty < 0.3 else "FLUCTUATING" if uncertainty < 0.6 else "HIGH"
        st.markdown(f"<div class='micro-data'>{uncertainty:.0%} | STATUS: {status}</div>",
                   unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    def render_situational_map(self, data):
        """Render enhanced tactical situational map"""
        st.markdown("<div class='tactical-panel panel-raised scan-line-effect'>"
                   "<h3>◈ TACTICAL SITUATIONAL DISPLAY</h3>",
                   unsafe_allow_html=True)
        
        altitude = data.get('altitude', 20)
        heatmap_data = self.heatmap_engine.generate_heatmap(altitude)
        
        fig = go.Figure()
        
        # Risk heatmap with graduated opacity
        fig.add_trace(go.Contour(
            x=heatmap_data['x'][0],
            y=heatmap_data['y'][:, 0],
            z=heatmap_data['z'],
            colorscale=[
                [0, 'rgba(0, 255, 136, 0.15)'],
                [0.3, 'rgba(255, 255, 0, 0.2)'],
                [0.6, 'rgba(255, 136, 0, 0.25)'],
                [1, 'rgba(255, 68, 102, 0.3)']
            ],
            opacity=0.6,
            showscale=True,
            colorbar=dict(
                title=dict(text="THREAT", font=dict(size=8, family='Rajdhani')),
                x=1.01,
                len=0.5,
                thickness=8,
                tickfont=dict(size=7, family='Share Tech Mono'),
                bgcolor='rgba(0, 0, 0, 0.7)',
                bordercolor='rgba(0, 255, 136, 0.3)',
                borderwidth=1
            ),
            contours=dict(start=0, end=1, size=0.1, showlines=True, coloring='heatmap'),
            line=dict(width=0.5, color='rgba(0, 255, 136, 0.2)'),
            hovertemplate='Threat Level: %{z:.2f}<extra></extra>'
        ))
        
        # Tactical grid overlay
        for x in range(-100, 101, 20):
            fig.add_trace(go.Scatter(
                x=[x, x], y=[-100, 100],
                mode='lines',
                line=dict(color='rgba(0, 255, 136, 0.08)', width=0.5),
                showlegend=False,
                hoverinfo='skip'
            ))
        for y in range(-100, 101, 20):
            fig.add_trace(go.Scatter(
                x=[-100, 100], y=[y, y],
                mode='lines',
                line=dict(color='rgba(0, 255, 136, 0.08)', width=0.5),
                showlegend=False,
                hoverinfo='skip'
            ))
        
        # Radar rings with distance markers
        drone_pos = data.get('position', [0, 0, 0])
        for radius in [30, 60, 90]:
            theta = np.linspace(0, 2*np.pi, 100)
            x_ring = drone_pos[0] + radius * np.cos(theta)
            y_ring = drone_pos[1] + radius * np.sin(theta)
            fig.add_trace(go.Scatter(
                x=x_ring, y=y_ring,
                mode='lines',
                line=dict(color='rgba(0, 255, 136, 0.15)', width=1, dash='dot'),
                showlegend=False,
                hoverinfo='skip'
            ))
            # Distance label
            fig.add_annotation(
                x=drone_pos[0] + radius,
                y=drone_pos[1],
                text=f"{radius}m",
                showarrow=False,
                font=dict(size=8, color='rgba(0, 255, 136, 0.5)'),
                bgcolor='rgba(0, 0, 0, 0.5)'
            )
        
        # Crosshairs
        fig.add_trace(go.Scatter(
            x=[-100, 100], y=[drone_pos[1], drone_pos[1]],
            mode='lines',
            line=dict(color='rgba(0, 255, 136, 0.1)', width=1),
            showlegend=False,
            hoverinfo='skip'
        ))
        fig.add_trace(go.Scatter(
            x=[drone_pos[0], drone_pos[0]], y=[-100, 100],
            mode='lines',
            line=dict(color='rgba(0, 255, 136, 0.1)', width=1),
            showlegend=False,
            hoverinfo='skip'
        ))
        
        # Environment zones with wireframe
        zones = self.heatmap_engine.get_zone_boundaries()
        for zone in zones:
            points = np.array(zone['points'])
            fig.add_trace(go.Scatter(
                x=points[:, 0], y=points[:, 1],
                mode='lines',
                line=dict(color=zone['color'], width=1.5, dash='dot'),
                name=zone['name'],
                fill='toself',
                fillcolor=zone['color'].replace(')', ', 0.1)').replace('rgb', 'rgba'),
                hoverinfo='name'
            ))
        
        # Camera FOV cone
        cam_pos = self.environment.camera.position
        fov_angle = np.radians(self.environment.camera.fov_degrees / 2)
        fov_range = self.environment.camera.max_range
        angles = np.linspace(-fov_angle, fov_angle, 30)
        fov_x = [cam_pos[0]] + [cam_pos[0] + fov_range * np.sin(a) for a in angles] + [cam_pos[0]]
        fov_y = [cam_pos[1]] + [cam_pos[1] + fov_range * np.cos(a) for a in angles] + [cam_pos[1]]
        fig.add_trace(go.Scatter(
            x=fov_x, y=fov_y,
            mode='lines',
            line=dict(color='rgba(255, 68, 102, 0.5)', width=1.5),
            fill='toself',
            fillcolor='rgba(255, 68, 102, 0.15)',
            name='CAMERA FOV',
            hoverinfo='name'
        ))
        
        # Trajectory with time markers
        trajectory = data.get('trajectory', [])
        if trajectory:
            traj_array = np.array(trajectory)
            fig.add_trace(go.Scatter(
                x=traj_array[:, 0], y=traj_array[:, 1],
                mode='lines+markers',
                line=dict(color='#00E5FF', width=2, dash='dot'),
                marker=dict(size=4, color='#00E5FF', opacity=0.6),
                name='TRAJECTORY',
                hoverinfo='skip'
            ))
            
            # Add time markers every 5 points
            for i in range(0, len(traj_array), 5):
                fig.add_annotation(
                    x=traj_array[i, 0],
                    y=traj_array[i, 1],
                    text=f"T-{i}s",
                    showarrow=False,
                    font=dict(size=7, color='rgba(0, 229, 255, 0.6)'),
                    bgcolor='rgba(0, 0, 0, 0.6)'
                )
        
        # Drone marker with heading indicator
        pos = data.get('position', [0, 0, 0])
        heading = data.get('heading', 0)
        heading_rad = np.radians(heading)
        heading_length = 10
        heading_x = pos[0] + heading_length * np.sin(heading_rad)
        heading_y = pos[1] + heading_length * np.cos(heading_rad)
        
        # Heading line
        fig.add_trace(go.Scatter(
            x=[pos[0], heading_x], y=[pos[1], heading_y],
            mode='lines',
            line=dict(color='#FFB800', width=3),
            showlegend=False,
            hoverinfo='skip'
        ))
        
        # Drone marker
        fig.add_trace(go.Scatter(
            x=[pos[0]], y=[pos[1]],
            mode='markers+text',
            marker=dict(size=20, color='#FFB800', symbol='triangle-up',
                       line=dict(color='#FFFFFF', width=2)),
            text=['◈'],
            textfont=dict(size=16, color='#FFB800'),
            name='DRONE',
            hovertemplate=f'POS: ({pos[0]:.1f}, {pos[1]:.1f})<br>ALT: {altitude:.1f}m<br>HDG: {heading:.0f}°<extra></extra>'
        ))
        
        # Camera marker
        fig.add_trace(go.Scatter(
            x=[cam_pos[0]], y=[cam_pos[1]],
            mode='markers+text',
            marker=dict(size=14, color='#FF4466', symbol='square',
                       line=dict(color='#FFFFFF', width=1.5)),
            text=['◆'],
            textfont=dict(size=12, color='#FF4466'),
            name='CAMERA',
            hoverinfo='name'
        ))
        
        # Compass rose
        compass_x = 85
        compass_y = 85
        compass_size = 10
        
        # N, E, S, W markers
        directions = [
            ('N', 0, compass_x, compass_y + compass_size),
            ('E', 90, compass_x + compass_size, compass_y),
            ('S', 180, compass_x, compass_y - compass_size),
            ('W', 270, compass_x - compass_size, compass_y)
        ]
        
        for label, angle, x, y in directions:
            fig.add_annotation(
                x=x, y=y,
                text=label,
                showarrow=False,
                font=dict(size=10, color='rgba(0, 255, 136, 0.7)', family='Rajdhani', weight='bold'),
                bgcolor='rgba(0, 0, 0, 0.7)',
                bordercolor='rgba(0, 255, 136, 0.5)',
                borderwidth=1
            )
        
        # Compass circle
        compass_theta = np.linspace(0, 2*np.pi, 100)
        compass_x_circle = compass_x + compass_size * 0.7 * np.cos(compass_theta)
        compass_y_circle = compass_y + compass_size * 0.7 * np.sin(compass_theta)
        fig.add_trace(go.Scatter(
            x=compass_x_circle, y=compass_y_circle,
            mode='lines',
            line=dict(color='rgba(0, 255, 136, 0.5)', width=1),
            showlegend=False,
            hoverinfo='skip'
        ))
        
        # Scale indicator
        fig.add_annotation(
            x=-85, y=-90,
            text="SCALE: 1:1000",
            showarrow=False,
            font=dict(size=8, color='rgba(0, 255, 136, 0.6)', family='Rajdhani'),
            bgcolor='rgba(0, 0, 0, 0.7)',
            bordercolor='rgba(0, 255, 136, 0.3)',
            borderwidth=1
        )
        
        # Layout
        fig.update_layout(
            height=520,
            plot_bgcolor='rgba(0, 0, 0, 0.7)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='rgba(0, 255, 136, 0.6)', size=8, family='Rajdhani'),
            xaxis=dict(
                title=dict(text="X POSITION (m)", font=dict(size=8)),
                gridcolor='rgba(0, 255, 136, 0.05)',
                gridwidth=1,
                range=[-100, 100],
                zeroline=True,
                zerolinecolor='rgba(0, 255, 136, 0.2)',
                zerolinewidth=1.5,
                tickfont=dict(size=7)
            ),
            yaxis=dict(
                title=dict(text="Y POSITION (m)", font=dict(size=8)),
                gridcolor='rgba(0, 255, 136, 0.05)',
                gridwidth=1,
                range=[-100, 100],
                scaleanchor="x",
                scaleratio=1,
                zeroline=True,
                zerolinecolor='rgba(0, 255, 136, 0.2)',
                zerolinewidth=1.5,
                tickfont=dict(size=7)
            ),
            showlegend=True,
            legend=dict(
                x=0.01, y=0.01,
                bgcolor='rgba(10, 20, 30, 0.9)',
                bordercolor='rgba(0, 255, 136, 0.3)',
                borderwidth=1,
                font=dict(size=7)
            ),
            margin=dict(l=40, r=40, t=5, b=40)
        )
        
        st.plotly_chart(fig, use_container_width=True, key="tactical_map",
                       config={'displayModeBar': False})
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    def render_behavioral_timeline(self, data):
        """Render enhanced behavioral timeline with timestamps"""
        st.markdown("<div class='timeline-enhanced'><h3>◈ BEHAVIORAL STATE TIMELINE</h3>",
                   unsafe_allow_html=True)
        
        state = data.get('state', 'TRANSIT')
        state_colors = {
            'TRANSIT': 'rgba(0, 255, 136, 0.7)',
            'SURVEILLANCE': 'rgba(255, 184, 0, 0.7)',
            'ADAPTIVE': 'rgba(255, 68, 102, 0.7)'
        }
        
        # Create timeline with hover tooltips
        timeline_html = "<div class='timeline-bar-enhanced'>"
        for i in range(15):
            color = state_colors.get(state, 'rgba(0, 229, 255, 0.5)')
            active_class = "active" if i == 14 else ""
            tooltip_time = f"T-{(14-i)*5}s"
            
            timeline_html += f"""
            <div class='timeline-segment-enhanced {active_class}' 
                 style='background: {color}; flex: 1;'
                 title='{state} | {tooltip_time}'>
            </div>
            """
        timeline_html += "</div>"
        
        st.markdown(timeline_html, unsafe_allow_html=True)
        
        # Current state indicator
        state_color_solid = state_colors.get(state, '#00E5FF').replace('rgba', 'rgb').replace(', 0.7)', ')')
        st.markdown(f"<div style='margin-top: 0.5rem; text-align: center;'>"
                   f"<span class='label-military'>CURRENT STATE:</span> "
                   f"<span style='color: {state_color_solid}; font-weight: 700; font-size: 1rem;'>{state}</span>"
                   f"</div>", unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    def render_threat_indicators(self, data):
        """Render vertical threat indicators with sparklines"""
        st.markdown("<div class='tactical-panel'><h3>◈ THREAT ASSESSMENT</h3>",
                   unsafe_allow_html=True)
        
        breakdown = data.get('risk_breakdown', {})
        factor_labels = {
            'detection_probability': 'DETECTION',
            'altitude_risk': 'ALTITUDE',
            'loiter_risk': 'LOITER',
            'proximity_risk': 'PROXIMITY',
            'predictability_risk': 'TRAJECTORY'
        }
        
        for idx, (factor, value) in enumerate(breakdown.items()):
            label = factor_labels.get(factor, factor.upper())
            
            # Color coding
            if value > 0.7:
                color = '#FF4466'
                level = 'CRITICAL'
            elif value > 0.4:
                color = '#FFB800'
                level = 'ELEVATED'
            else:
                color = '#00FF88'
                level = 'NOMINAL'
            
            st.markdown(f"<div class='label-military'>{label}</div>", unsafe_allow_html=True)
            
            # Vertical threat indicator
            st.markdown(f"""
            <div class='threat-indicator'>
                <div class='threat-value' style='color: {color};'>{value:.0%}</div>
                <div class='threat-fill' style='height: {value*100}%; color: {color};'></div>
            </div>
            """, unsafe_allow_html=True)
            
            # Sparkline (simulated trend)
            if len(st.session_state.risk_history) > 1:
                # Create factor-specific history (simulated)
                factor_history = [v * (0.8 + np.random.random() * 0.4) for v in st.session_state.risk_history[-20:]]
                fig_spark = self.create_sparkline(factor_history, color, f"threat_{idx}")
                if fig_spark:
                    st.plotly_chart(fig_spark, use_container_width=True, 
                                   key=f"spark_threat_{idx}",
                                   config={'displayModeBar': False})
            
            st.markdown(f"<div class='micro-data' style='text-align: center;'>{level}</div>",
                       unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    def render_risk_index_panel(self, data):
        """Render main risk index with circular progress"""
        risk_score = data.get('risk_score', 0)
        
        # Determine threat level
        if risk_score > 0.7:
            threat_level = "CRITICAL"
            threat_color = "#FF4466"
            panel_class = "critical-pulse"
            glow_class = "glow-red"
        elif risk_score > 0.4:
            threat_level = "ELEVATED"
            threat_color = "#FFB800"
            panel_class = ""
            glow_class = "glow-amber"
        else:
            threat_level = "NOMINAL"
            threat_color = "#00FF88"
            panel_class = ""
            glow_class = "glow-green"
        
        st.markdown(f"<div class='tactical-panel {panel_class}'><h3>◈ RISK INDEX</h3>",
                   unsafe_allow_html=True)
        
        # Circular progress indicator
        radius = 60
        circumference = 2 * np.pi * radius
        offset = circumference - (risk_score * circumference)
        
        circular_progress_html = f"""
        <div class='circular-progress'>
            <svg width="140" height="140" style="transform: rotate(-90deg);">
                <circle cx="70" cy="70" r="{radius}" 
                        stroke="rgba(0, 255, 136, 0.1)" 
                        stroke-width="8" 
                        fill="none"/>
                <circle cx="70" cy="70" r="{radius}" 
                        stroke="{threat_color}" 
                        stroke-width="8" 
                        fill="none"
                        stroke-dasharray="{circumference}"
                        stroke-dashoffset="{offset}"
                        style="transition: stroke-dashoffset 0.5s ease;
                               filter: drop-shadow(0 0 8px {threat_color});"/>
            </svg>
            <div style="position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%);
                        text-align: center;">
                <div class='numeric-huge {glow_class}' style='color: {threat_color}; font-size: 3rem;'>
                    {risk_score:.0%}
                </div>
            </div>
        </div>
        """
        st.markdown(circular_progress_html, unsafe_allow_html=True)
        
        st.markdown(f"<div style='text-align: center; font-size: 1.2rem; font-weight: 700; "
                   f"color: {threat_color}; letter-spacing: 2px; margin-top: 0.5rem;'>"
                   f"{threat_level}</div>",
                   unsafe_allow_html=True)
        
        # Trend indicator
        trend_symbol, trend_class = self.get_trend_indicator(risk_score, st.session_state.risk_history)
        trend_colors = {'up': '#FF4466', 'down': '#00FF88', 'stable': '#FFB800'}
        st.markdown(f"<div style='text-align: center; margin-top: 0.5rem;'>"
                   f"<span class='label-military'>TREND:</span> "
                   f"<span class='trend-indicator trend-{trend_class}' "
                   f"style='color: {trend_colors.get(trend_class, '#FFB800')}; font-size: 1.5rem;'>"
                   f"{trend_symbol}</span></div>",
                   unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    def render_projected_risk_panel(self, data):
        """Render projected risk with enhanced visualization"""
        st.markdown("<div class='tactical-panel'><h3>◈ PROJECTED RISK (T+5s)</h3>",
                   unsafe_allow_html=True)
        
        cf_risk = data.get('counterfactual_risk', 0)
        risk_score = data.get('risk_score', 0)
        delta = cf_risk - risk_score
        delta_pct = (delta / risk_score * 100) if risk_score > 0 else 0
        
        delta_color = "#FF4466" if delta > 0 else "#00FF88"
        delta_symbol = "↑" if delta > 0 else "↓"
        
        # Projected risk display
        st.markdown(f"<div style='text-align: center;'>"
                   f"<div class='numeric-large' style='color: {delta_color}; margin: 1rem 0;'>"
                   f"{cf_risk:.0%}</div>"
                   f"<div style='font-size: 1.1rem; color: {delta_color}; font-weight: 600;'>"
                   f"{delta_symbol} {abs(delta):.1%} ({abs(delta_pct):.1f}%)</div>"
                   f"</div>", unsafe_allow_html=True)
        
        # Time horizon selector (visual only)
        st.markdown(f"<div style='text-align: center; margin-top: 1rem;'>"
                   f"<span class='label-military'>TIME HORIZON:</span><br>"
                   f"<span style='color: #00FF88; font-weight: 600;'>T+5S</span> | "
                   f"<span style='color: rgba(255, 255, 255, 0.3);'>T+10S</span> | "
                   f"<span style='color: rgba(255, 255, 255, 0.3);'>T+30S</span>"
                   f"</div>", unsafe_allow_html=True)
        
        # Recommendation
        if cf_risk > 0.7:
            recommendation = "IMMEDIATE EVASIVE ACTION RECOMMENDED"
            rec_color = "#FF4466"
        elif cf_risk > 0.4:
            recommendation = "MONITOR SITUATION CLOSELY"
            rec_color = "#FFB800"
        else:
            recommendation = "CONTINUE CURRENT MISSION"
            rec_color = "#00FF88"
        
        st.markdown(f"<div style='margin-top: 1rem; padding: 0.5rem; "
                   f"background: rgba(0, 0, 0, 0.4); border-left: 3px solid {rec_color}; "
                   f"border-radius: 2px;'>"
                   f"<span class='label-military'>RECOMMENDATION:</span><br>"
                   f"<span style='color: {rec_color}; font-size: 0.8rem; font-weight: 600;'>"
                   f"{recommendation}</span></div>",
                   unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    def render_state_transition_log(self, data):
        """Render enhanced state transition log"""
        st.markdown("<div class='tactical-panel'><h3>◈ STATE TRANSITION LOG</h3>",
                   unsafe_allow_html=True)
        
        state_history = data.get('state_history', [])
        
        st.markdown("<div class='log-container-enhanced'>", unsafe_allow_html=True)
        
        if state_history:
            for transition in state_history[-8:]:
                from_state = transition.get('from', '?')
                to_state = transition.get('to', '?')
                trans_risk = transition.get('risk', 0)
                timestamp = transition.get('timestamp', '00:00:00')
                
                # Determine severity
                if trans_risk > 0.7:
                    severity = 'critical'
                    icon = '⚠'
                elif trans_risk > 0.4:
                    severity = 'warning'
                    icon = '⚡'
                else:
                    severity = 'info'
                    icon = '●'
                
                st.markdown(f"""
                <div class='log-entry-enhanced {severity}'>
                    <span style='font-size: 1rem;'>{icon}</span>
                    <span style='flex: 1;'>
                        <span style='color: #00E5FF;'>{from_state}</span> → 
                        <span style='color: #FFB800;'>{to_state}</span>
                    </span>
                    <span style='color: rgba(255, 255, 255, 0.5);'>{timestamp}</span>
                    <span style='color: rgba(255, 255, 255, 0.7);'>R:{trans_risk:.2f}</span>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown("<div class='label-military' style='text-align: center; padding: 1rem;'>"
                       "NO TRANSITIONS RECORDED</div>",
                       unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Log controls
        st.markdown(f"<div style='margin-top: 0.5rem; text-align: center;'>"
                   f"<span class='label-military'>FILTERS:</span> "
                   f"<span style='color: #00FF88; font-size: 0.7rem; cursor: pointer;'>ALL</span> | "
                   f"<span style='color: rgba(255, 255, 255, 0.3); font-size: 0.7rem;'>CRITICAL</span> | "
                   f"<span style='color: rgba(255, 255, 255, 0.3); font-size: 0.7rem;'>WARNING</span>"
                   f"</div>", unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    def render_threat_analysis_panel(self, data):
        """Render detailed threat analysis"""
        st.markdown("<div class='tactical-panel'><h3>◈ THREAT ANALYSIS</h3>",
                   unsafe_allow_html=True)
        
        explanations = data.get('risk_explanation', [])
        
        if explanations:
            for idx, exp in enumerate(explanations):
                # Determine severity from content
                if 'critical' in exp.lower() or 'high' in exp.lower():
                    severity = 'critical'
                    icon = '⚠'
                    color = '#FF4466'
                elif 'moderate' in exp.lower() or 'elevated' in exp.lower():
                    severity = 'warning'
                    icon = '⚡'
                    color = '#FFB800'
                else:
                    severity = 'info'
                    icon = '●'
                    color = '#00E5FF'
                
                st.markdown(f"""
                <div class='log-entry-enhanced {severity}' style='margin-bottom: 0.5rem;'>
                    <span style='font-size: 1rem; color: {color};'>{icon}</span>
                    <span style='flex: 1; line-height: 1.4; color: rgba(255, 255, 255, 0.8);'>
                        {exp}
                    </span>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown("<div class='label-military' style='text-align: center; padding: 1rem;'>"
                       "NO THREATS DETECTED</div>",
                       unsafe_allow_html=True)
        
        # Threat summary
        risk_score = data.get('risk_score', 0)
        if risk_score > 0.7:
            summary = "MULTIPLE HIGH-PRIORITY THREATS DETECTED"
            sum_color = "#FF4466"
        elif risk_score > 0.4:
            summary = "MODERATE THREAT LEVEL - MAINTAIN VIGILANCE"
            sum_color = "#FFB800"
        else:
            summary = "THREAT ENVIRONMENT: NOMINAL"
            sum_color = "#00FF88"
        
        st.markdown(f"<div style='margin-top: 1rem; padding: 0.6rem; "
                   f"background: rgba(0, 0, 0, 0.5); border: 1px solid {sum_color}; "
                   f"border-radius: 2px; text-align: center;'>"
                   f"<span class='label-military'>ASSESSMENT:</span><br>"
                   f"<span style='color: {sum_color}; font-size: 0.8rem; font-weight: 600;'>"
                   f"{summary}</span></div>",
                   unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    def render_quick_actions_panel(self, data):
        """Render quick action buttons"""
        st.markdown("<div class='tactical-panel'><h3>◈ QUICK ACTIONS</h3>",
                   unsafe_allow_html=True)
        
        # Action buttons (visual only - no backend integration)
        actions = [
            ("RTH", "Return to Home", "#00FF88"),
            ("HOLD", "Hold Position", "#FFB800"),
            ("EVADE", "Evasive Maneuver", "#FF4466"),
            ("LAND", "Emergency Land", "#FF4466")
        ]
        
        for action_code, action_name, action_color in actions:
            st.markdown(f"""
            <div style='margin-bottom: 0.6rem; padding: 0.6rem; 
                        background: rgba(0, 0, 0, 0.4); 
                        border: 1px solid {action_color}; 
                        border-radius: 2px; 
                        text-align: center; 
                        cursor: pointer;
                        transition: all 0.2s ease;'
                 onmouseover='this.style.background="rgba(0, 255, 136, 0.1)"; this.style.transform="translateX(3px)";'
                 onmouseout='this.style.background="rgba(0, 0, 0, 0.4)"; this.style.transform="translateX(0)";'>
                <span style='color: {action_color}; font-weight: 700; font-size: 0.9rem; 
                             letter-spacing: 2px;'>{action_code}</span><br>
                <span class='label-military' style='font-size: 0.65rem;'>{action_name}</span>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)
    

def main():
    """Main military-grade tactical HUD application"""
    
    hud = MilitaryTacticalHUD()
    
    # Load data
    data = hud.load_live_data()
    if not data:
        data = hud.load_sample_data()
    
    # Update historical data
    hud.update_history(data)
    
    # Render alert banner if critical (before command bar)
    hud.render_alert_banner(data)
    
    # Render command bar (after CSS is loaded)
    hud.render_command_bar(data)
    
    # Create main content area with proper spacing for command bar and alert banner
    risk_score = data.get('risk_score', 0) if data else 0
    top_spacing = 110 if risk_score > 0.7 else 70
    st.markdown(f"<div style='height: {top_spacing}px;'></div>", unsafe_allow_html=True)
    
    # Create three-column layout using Streamlit columns
    col_left, col_center, col_right = st.columns([1, 2.3, 1], gap="small")
    
    with col_left:
        # Left panel - Telemetry and System Health
        hud.render_telemetry_panel(data)
        hud.render_system_health_panel(data)
        hud.render_intent_classification_panel(data)
    
    with col_center:
        # Center panel - Situational Map and Timeline
        hud.render_situational_map(data)
        hud.render_behavioral_timeline(data)
    
    with col_right:
        # Right panel - Risk and Threat Assessment
        hud.render_risk_index_panel(data)
        hud.render_threat_indicators(data)
        hud.render_projected_risk_panel(data)
        hud.render_state_transition_log(data)
        hud.render_threat_analysis_panel(data)
        hud.render_quick_actions_panel(data)
    
    # DISABLED AUTO-REFRESH - Remove the lines below to enable
    # if st.session_state.get('auto_refresh', False):
    #     time.sleep(1.5)
    #     st.rerun()

if __name__ == "__main__":
    main()
