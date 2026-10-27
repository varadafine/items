import re

# Read the file
with open('intentra_x/dashboard.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Find and replace the render_command_bar function
# Pattern to match from 'def render_command_bar' to the next 'def'
pattern = r'(    def render_command_bar\(self, data\):.*?)(    def render_alert_banner)'

replacement = r'''    def render_command_bar(self, data):
        """Render top command bar using Streamlit native components"""
        status = "OPERATIONAL" if data else "NO SIGNAL"
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
        
        # Use Streamlit columns instead of HTML
        st.markdown("---")
        cols = st.columns([2, 1, 1, 1, 1, 1, 1, 1])
        
        with cols[0]:
            st.markdown(f"### 🎯 INTENTRA-X")
            st.caption("TACTICAL COMMAND")
        
        with cols[1]:
            st.metric("MISSION TIME", mission_time)
        
        with cols[2]:
            st.metric("COORDINATES", f"{gps_lat:.4f}N")
            st.caption(f"{abs(gps_lon):.4f}W")
        
        with cols[3]:
            st.metric("HEADING", f"{heading:.0f}°")
        
        with cols[4]:
            batt_emoji = "🔋" if battery > 50 else "⚠️"
            st.metric("BATTERY", f"{battery:.0f}%", delta=None)
            st.caption(batt_emoji)
        
        with cols[5]:
            sig_emoji = "📡" if signal > 70 else "⚠️"
            st.metric("SIGNAL", f"{signal:.0f}%")
            st.caption(sig_emoji)
        
        with cols[6]:
            st.metric("DATALINK", f"{datalink:.0f}%")
            st.caption(f"{latency:.0f}ms")
        
        with cols[7]:
            st.metric("STATE", state)
            st.caption(f"Risk: {risk:.0%}")
        
        st.markdown("---")
    
\2'''

new_content = re.sub(pattern, replacement, content, flags=re.DOTALL)

# Write back
with open('intentra_x/dashboard.py', 'w', encoding='utf-8') as f:
    f.write(new_content)

print('Fixed render_command_bar function!')
print('The HTML tooltip issue should now be resolved.')
