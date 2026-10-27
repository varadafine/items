"""
Test HTML rendering in Streamlit
"""
import streamlit as st

st.set_page_config(page_title="HTML Rendering Test", layout="wide")

st.title("HTML Rendering Test")

# Test 1: Simple HTML
st.markdown("## Test 1: Simple HTML with unsafe_allow_html=True")
st.markdown("""
<div style='background-color: red; color: white; padding: 20px; border-radius: 5px;'>
    <h2>RED BOX TEST</h2>
    <p>If you see this as a RED BOX with white text, HTML is rendering correctly!</p>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# Test 2: Command bar style HTML
st.markdown("## Test 2: Command Bar Style")
st.markdown("""
<style>
    .test-command-bar {
        background: linear-gradient(180deg, rgba(10, 14, 20, 0.98) 0%, rgba(10, 14, 20, 0.95) 100%);
        border: 2px solid #00FF88;
        padding: 20px;
        color: #00FF88;
        font-size: 1.2rem;
        text-align: center;
        border-radius: 5px;
    }
</style>
<div class='test-command-bar'>
    GREEN COMMAND BAR TEST - If you see this in a green bordered box, CSS is working!
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# Test 3: Check Streamlit version
st.markdown("## Test 3: System Info")
st.write(f"Streamlit version: {st.__version__}")

# Test 4: Alternative rendering method
st.markdown("## Test 4: Using st.html (if available)")
try:
    st.html("""
    <div style='background-color: blue; color: white; padding: 20px;'>
        BLUE BOX TEST using st.html()
    </div>
    """)
    st.success("st.html() is available and working!")
except AttributeError:
    st.warning("st.html() is not available in your Streamlit version")

st.markdown("---")
st.markdown("### What do you see?")
st.markdown("""
- ✅ **CORRECT**: Colored boxes with text inside
- ❌ **WRONG**: Raw HTML code displayed as text
- ❌ **WRONG**: HTML code in a tooltip/popup
""")
