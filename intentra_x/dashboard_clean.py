"""
INTENTRA-X Clean Dashboard - NO CUSTOM HTML
Pure Streamlit components only
"""

import streamlit as st
import plotly.graph_objects as go
import json
import numpy as np
from datetime import datetime, timedelta
import time

from heatmap_engine import HeatmapEngine
from environment import EnvironmentModel

st.set_page_config(
    page_title="INTENTRA-X TACTICAL COMMAND",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Simple dark theme CSS
st.markdown("""
<style>
    .main {
        background-color: #0A0E14;
    }
    .stMetric {
        background-color: rgba(10, 20, 30, 0.8);
        padding: 10px;
        border-radius: 5px;
        border: 1px solid rgba(0, 255, 136, 0.3);
    }
</style>
""", unsafe_allow_html=True)


class CleanDashboard:
    def __init__(self):
        self.heatmap_engine = HeatmapEngine()
        self.environment = EnvironmentModel()
        self.mission_start_time = datetime.now()
        
        if 'risk_history' not in st.session_state:
            st.session_state.risk_history = []
        if 'altitude_history' not in st.session_state:
            st.session_state.altitude_history = []
        if 'speed_history' not in st.session_state:
            st.session_state.speed_history = []
    
    def load_live_data(self):
        try:
            with open('intentra_x/live_output.json', 'r') as f:
                return json.load(f)
        except:
            return None
    
    def load_sample_data(self):
        t = time.time()
        angle = (t % 30) / 30 * 2 * np.pi
        base_lat, base_lon = 35.0522, -106.6056
        lat_offset = 15 * np.cos(angle) / 111000
        lon_offset = 15 * np.sin(angle) / (111000 * np.cos(np.radians(base_lat)))
        
        return {
            'timestamp': t,
            'position': [15 * np.cos(angle), 15 * np.sin(angle), -20],
            'velocity': [-15 * np.sin(angle), 15 * np.cos(angle), 0],
            'altitude': 20 + np.sin(t * 0.5) * 2,
            'speed': 5.0 + np.sin(t * 0.3) * 0.5,
            'heading': np.degrees(angle) % 360,
            'gps_lat': base_lat + lat_offset,
            'gps_lon': base_lon + lon_offset,
            'battery_level': max(65, 100 - (t % 300) / 3),
            'signal_strength': 85 + np.sin(t * 0.2) * 10,
            'datalink_quality': 92 + np.sin(t * 0.15) * 5,
            'datalink_latency': 45 + np.sin(t * 0.25) * 15,
            'trajectory': [[15 * np.cos(angle - i*0.1), 
                           15 * np.sin(angle - i*0.1), -20] 
                          for i in range(20)],
            'intent': 'SURVEILLANCE',
            'intent_confidence': 0.85,
            'uncertainty': 0.25,
            'risk_score': 0.45 + np.sin(t * 0.1) * 0.1,
            'risk_breakdown': {
                'detection_probability': 0.5,
                'altitude_risk': 0.4,
                'loiter_risk': 0.6,
                'proximity_risk': 0.3,
                'predictability_risk': 0.4
            },
            'state': 'SURVEILLANCE',
            'counterfactual_risk': 0.52,
        }
    
    def update_history(self, data):
        st.session_state.risk_history.append(data.get('risk_score', 0))
        st.session_state.altitude_history.append(data.get('altitude', 0))
        st.session_state.speed_history.append(data.get('speed', 0))
        
        if len(st.session_state.risk_history) > 60:
            st.session_state.risk_history.pop(0)
        if len(st.session_state.altitude_history) > 60:
            st.session_state.altitude_history.pop(0)
        if len(st.session_state.speed_history) > 60:
            st.session_state.speed_history.pop(0)
    
    def get_mission_elapsed_time(self):
        elapsed = datetime.now() - self.mission_start_time
        hours = int(elapsed.total_seconds() // 3600)
        minutes = int((elapsed.total_seconds() % 3600) // 60)
        seconds = int(elapsed.total_seconds() % 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    
    def render_command_bar(self, data):
        st.title("🎯 INTENTRA-X TACTICAL COMMAND")
        
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
        
        cols = st.columns(8)
        
        with cols[0]:
            st.metric("MISSION TIME", mission_time)
        
        with cols[1]:
            st.metric("GPS LAT", f"{gps_lat:.4f}N")
        
        with cols[2]:
            st.metric("GPS LON", f"{abs(gps_lon):.4f}W")
        
        with cols[3]:
            st.metric("HEADING", f"{heading:.0f}°")
        
        with cols[4]:
            st.metric("BATTERY", f"{battery:.0f}%")
        
        with cols[5]:
            st.metric("SIGNAL", f"{signal:.0f}%")
        
        with cols[6]:
            st.metric("DATALINK", f"{datalink:.0f}%")
        
        with cols[7]:
            st.metric("STATE", state)
            st.caption(f"Risk: {risk:.0%}")
        
        st.divider()
    
    def render_tactical_map(self, data):
        st.subheader("📍 TACTICAL SITUATIONAL DISPLAY")
        
        altitude = data.get('altitude', 20)
        heatmap_data = self.heatmap_engine.generate_heatmap(altitude)
        
        fig = go.Figure()
        
        # Risk heatmap
        fig.add_trace(go.Contour(
            x=heatmap_data['x'][0],
            y=heatmap_data['y'][:, 0],
            z=heatmap_data['z'],
            colorscale=[
                [0, 'rgba(0, 255, 136, 0.3)'],
                [0.3, 'rgba(255, 255, 0, 0.4)'],
                [0.6, 'rgba(255, 136, 0, 0.5)'],
                [1, 'rgba(255, 68, 102, 0.6)']
            ],
            showscale=True,
            colorbar=dict(title="Threat Level", x=1.02),
            hovertemplate='Threat: %{z:.2f}<extra></extra>'
        ))
        
        # Trajectory trail
        trajectory = data.get('trajectory', [])
        if trajectory:
            traj_array = np.array(trajectory)
            fig.add_trace(go.Scatter(
                x=traj_array[:, 0], y=traj_array[:, 1],
                mode='lines',
                line=dict(color='cyan', width=2, dash='dot'),
                name='Trajectory',
                hoverinfo='skip'
            ))
        
        # Drone position with heading
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
            line=dict(color='yellow', width=3),
            showlegend=False,
            hoverinfo='skip'
        ))
        
        # Drone marker
        fig.add_trace(go.Scatter(
            x=[pos[0]], y=[pos[1]],
            mode='markers+text',
            marker=dict(size=20, color='yellow', symbol='triangle-up', 
                       line=dict(color='white', width=2)),
            text=['◈'],
            textfont=dict(size=16, color='yellow'),
            name='Drone',
            hovertemplate=f'Position: ({pos[0]:.1f}, {pos[1]:.1f})<br>Altitude: {altitude:.1f}m<br>Heading: {heading:.0f}°<extra></extra>'
        ))
        
        # Camera position
        cam_pos = self.environment.camera.position
        fig.add_trace(go.Scatter(
            x=[cam_pos[0]], y=[cam_pos[1]],
            mode='markers+text',
            marker=dict(size=14, color='red', symbol='square'),
            text=['CAM'],
            name='Camera',
            hoverinfo='name'
        ))
        
        fig.update_layout(
            height=500,
            plot_bgcolor='#0A0E14',
            paper_bgcolor='#0A0E14',
            font=dict(color='#00FF88'),
            xaxis=dict(
                title="X Position (m)", 
                range=[-100, 100],
                gridcolor='rgba(0, 255, 136, 0.1)',
                zeroline=True,
                zerolinecolor='rgba(0, 255, 136, 0.3)'
            ),
            yaxis=dict(
                title="Y Position (m)", 
                range=[-100, 100], 
                scaleanchor="x",
                gridcolor='rgba(0, 255, 136, 0.1)',
                zeroline=True,
                zerolinecolor='rgba(0, 255, 136, 0.3)'
            ),
            showlegend=True,
            legend=dict(
                x=0.01, y=0.01,
                bgcolor='rgba(10, 20, 30, 0.8)',
                bordercolor='rgba(0, 255, 136, 0.3)',
                borderwidth=1
            )
        )
        
        st.plotly_chart(fig, use_container_width=True, key="tactical_map")


def main():
    dashboard = CleanDashboard()
    
    # Load data
    data = dashboard.load_live_data()
    if not data:
        data = dashboard.load_sample_data()
    
    dashboard.update_history(data)
    
    # Render command bar
    dashboard.render_command_bar(data)
    
    # Alert if high risk
    risk_score = data.get('risk_score', 0)
    if risk_score > 0.7:
        st.error("⚠️ CRITICAL ALERT: HIGH RISK EXPOSURE DETECTED")
    
    # Three columns
    col_left, col_center, col_right = st.columns([1, 2, 1])
    
    with col_left:
        st.subheader("📊 TELEMETRY")
        st.metric("Altitude", f"{data.get('altitude', 0):.1f} m")
        st.metric("Speed", f"{data.get('speed', 0):.1f} m/s")
        st.metric("Intent", data.get('intent', 'UNKNOWN'))
        st.metric("Confidence", f"{data.get('intent_confidence', 0):.0%}")
    
    with col_center:
        dashboard.render_tactical_map(data)
    
    with col_right:
        st.subheader("⚠️ RISK ASSESSMENT")
        st.metric("Current Risk", f"{risk_score:.0%}")
        st.metric("Projected Risk", f"{data.get('counterfactual_risk', 0):.0%}")
        
        st.subheader("📈 RISK BREAKDOWN")
        breakdown = data.get('risk_breakdown', {})
        for factor, value in breakdown.items():
            st.metric(factor.replace('_', ' ').title(), f"{value:.0%}")


if __name__ == "__main__":
    main()
