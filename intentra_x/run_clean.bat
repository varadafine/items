@echo off
echo Starting INTENTRA-X Dashboard with clean config...
set STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
cd /d "%~dp0"
python -m streamlit run dashboard.py --server.port 8501 2>nul
pause
