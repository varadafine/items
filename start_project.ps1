param(
    [ValidateSet('demo', 'test')]
    [string]$Mode = 'demo',
    [int]$Port = 8501
)

$projectRoot = $PSScriptRoot
$pythonExe = Join-Path $projectRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $pythonExe)) {
    throw "Project environment missing: $pythonExe"
}

# The project environment inherits packages from Anaconda. On Windows its
# native extensions need Anaconda's DLL directory on PATH.
$venvConfig = Join-Path $projectRoot '.venv\pyvenv.cfg'
$homeLine = Get-Content -LiteralPath $venvConfig | Where-Object { $_ -match '^home\s*=' } | Select-Object -First 1
if (-not $homeLine) {
    throw "Cannot find Anaconda home in $venvConfig"
}
$anacondaRoot = ($homeLine -split '=', 2)[1].Trim()
$dllDirectory = Join-Path $anacondaRoot 'Library\bin'
if (-not (Test-Path -LiteralPath $dllDirectory)) {
    throw "Anaconda DLL directory missing: $dllDirectory"
}
$env:PATH = "$dllDirectory;$anacondaRoot\Scripts;$anacondaRoot;$env:PATH"
$env:MPLCONFIGDIR = Join-Path $projectRoot '.cache\matplotlib'
$env:NUMBA_CACHE_DIR = Join-Path $projectRoot '.cache\numba'
$env:PYTHONIOENCODING = 'utf-8'
$env:STREAMLIT_BROWSER_GATHER_USAGE_STATS = 'false'
New-Item -ItemType Directory -Force -Path $env:MPLCONFIGDIR, $env:NUMBA_CACHE_DIR | Out-Null

Push-Location $projectRoot
try {
    switch ($Mode) {
        'demo' {
            Push-Location (Join-Path $projectRoot 'intentra_x')
            try { & $pythonExe -m streamlit run 'dashboard.py' --server.port $Port --browser.gatherUsageStats false }
            finally { Pop-Location }
        }
        'test' {
            Push-Location (Join-Path $projectRoot 'intentra_x')
            try { & $pythonExe 'test_system.py' } finally { Pop-Location }
            if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
            & $pythonExe -m unittest intentra_x.test_air_risk_12 intentra_x.test_joint_risk_integration
        }
    }
    exit $LASTEXITCODE
}
finally {
    Pop-Location
}
