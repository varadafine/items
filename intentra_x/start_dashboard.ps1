# PowerShell script to run INTENTRA-X dashboard
Write-Host "Starting INTENTRA-X Military-Grade Tactical Dashboard..." -ForegroundColor Green
Write-Host ""

# Set environment to bypass config issues
$env:STREAMLIT_BROWSER_GATHER_USAGE_STATS = "false"

# Change to script directory
Set-Location $PSScriptRoot

# Run streamlit
Write-Host "Dashboard will open at http://localhost:8501" -ForegroundColor Cyan
Write-Host "Press Ctrl+C to stop the server" -ForegroundColor Yellow
Write-Host ""

python -m streamlit run dashboard.py --server.port 8501 --browser.gatherUsageStats false 2>&1 | Where-Object { $_ -notmatch "Error parsing config toml" -and $_ -notmatch "UnicodeDecodeError" }
