"""
Direct runner for INTENTRA-X dashboard that bypasses Streamlit config issues
"""
import sys
import os

# Set environment variables before importing streamlit
os.environ['STREAMLIT_BROWSER_GATHER_USAGE_STATS'] = 'false'
os.environ['STREAMLIT_SERVER_PORT'] = '8501'
os.environ['STREAMLIT_SERVER_HEADLESS'] = 'true'

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import and run streamlit
from streamlit.web import cli as stcli

if __name__ == '__main__':
    sys.argv = ["streamlit", "run", "dashboard.py", 
                "--server.port=8501",
                "--browser.gatherUsageStats=false",
                "--server.headless=true"]
    sys.exit(stcli.main())
