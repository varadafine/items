"""MINIMAL TEST - Debug Streamlit HTML Rendering"""
import streamlit as st

st.set_page_config(page_title="TEST", layout="wide")

# Test 1: Simple HTML
st.markdown("""
<div style='background: red; padding: 20px; color: white;'>
<h1>TEST 1: If you see this in red background, HTML is working</h1>
</div>
""", unsafe_allow_html=True)

# Test 2: Complex HTML like your command bar
st.markdown("""
<div style='position: fixed; top: 0; left: 0; right: 0; height: 60px; background: rgba(10, 14, 20, 0.98); border-bottom: 2px solid rgba(0, 255, 136, 0.3); z-index: 999999; display: flex; align-items: center; padding: 0 2%;'>
<span style='font-size: 1.3rem; color: #00FF88;'>COMMAND BAR TEST</span>
</div>
""", unsafe_allow_html=True)

st.markdown("<div style='height: 80px;'></div>", unsafe_allow_html=True)

st.write("# Regular Streamlit Text")
st.write("If you see the red box above and the green command bar at the very top, HTML is working correctly.")
st.write("If you see raw HTML code instead, there's a Streamlit configuration issue.")

# Test 3: Check Streamlit version
st.write(f"**Streamlit Version:** {st.__version__}")
