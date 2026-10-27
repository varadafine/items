# Windows setup (this machine)

The project uses `F:\Anaconda` (Python 3.9.13) and a local `.venv` that inherits
its installed packages. `start_project.ps1` adds Anaconda's `Library\bin` to
`PATH`, which is required for Python's SSL and native packages after the system
reset. The virtual environment is excluded from Git.

Verified package versions: NumPy 1.24.4, PyTorch 2.8.0+cpu, Streamlit 1.50.0,
Plotly 5.24.1, pandas 2.3.3, SciPy 1.9.1, scikit-learn 1.0.2,
Numba 0.60.0, rasterio 1.4.3, tifffile 2024.8.30, Casex 1.0.5,
Matplotlib 3.5.2, and imageio 2.19.3. Git 2.52.0 is at `D:\Git`.

Run from this repository in PowerShell:

```powershell
powershell -ExecutionPolicy Bypass -File .\start_project.ps1 test
powershell -ExecutionPolicy Bypass -File .\start_project.ps1 demo
```

The dashboard is available at `http://localhost:8501`. Select **Demo** in the sidebar.
To use another port, append `-Port 8502` to the demo command.
For an offline joint-risk sample, run:

```powershell
& '.\.venv\Scripts\python.exe' -m intentra_x.run_joint_demo --output-dir '.cache\joint_demo'
```

That command needs `F:\Anaconda\Library\bin` on `PATH` as in the setup block
below. The sample generates JSON, NumPy arrays and a ground-risk image.

If `.venv` is deleted, recreate it with:

```powershell
$env:PATH = 'F:\Anaconda\Library\bin;F:\Anaconda\Scripts;F:\Anaconda;' + $env:PATH
& 'F:\Anaconda\python.exe' -m venv --system-site-packages .venv
& '.\.venv\Scripts\python.exe' -m pip install -r intentra_x\requirements.txt
```

`F:\桌面重要文件\Cosys-AirSim-main` is Cosys-AirSim 3.3.0 source, including the
`Blocks.uproject` scene. Its Python package is named `cosysairsim`, while this
project's `main_brain.py` currently imports the older `airsim` package. Live
flight mode requires a separate client integration and a running scene; the
commands above cover the standalone demo and offline analysis.
