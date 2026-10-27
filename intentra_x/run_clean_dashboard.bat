@echo off
echo Starting INTENTRA-X Clean Dashboard...
echo.
echo NOTE: This version uses only Streamlit components (no HTML tooltips)
echo Dashboard will open at http://localhost:8501
echo Press Ctrl+C to stop
echo.
python -c "import sys; sys.path.insert(0, '.'); exec(open('run_dashboard_patched.py').read().replace('dashboard.py', 'dashboard_clean.py'))"
