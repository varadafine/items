"""
INTENTRA-X Demo Runner

Quick start script for demonstration mode
No AirSim required - generates synthetic data
"""

import subprocess
import sys
import time
import os

def check_dependencies():
    """Check if required packages are installed"""
    required = ['streamlit', 'plotly', 'numpy', 'torch', 'pandas']
    missing = []
    
    for package in required:
        try:
            __import__(package)
        except ImportError:
            missing.append(package)
    
    if missing:
        print(f"❌ Missing packages: {', '.join(missing)}")
        print(f"\nInstall with: pip install {' '.join(missing)}")
        return False
    
    print("✅ All dependencies installed")
    return True

def main():
    """Run demo mode"""
    print("=" * 60)
    print("INTENTRA-X - Cognitive Airspace Intelligence System")
    print("Demo Mode (No AirSim Required)")
    print("=" * 60)
    print()
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    print("\n🚀 Starting dashboard in demo mode...")
    print("📊 Dashboard will open in your browser")
    print("⚙️  Select 'Demo' mode in the sidebar")
    print("\n💡 Press Ctrl+C to stop\n")
    
    time.sleep(2)
    
    # Change to intentra_x directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    # Run streamlit
    try:
        subprocess.run([
            sys.executable, '-m', 'streamlit', 'run', 'dashboard.py',
            '--server.headless', 'false'
        ])
    except KeyboardInterrupt:
        print("\n\n✋ Shutting down...")
        print("✅ Demo stopped")

if __name__ == "__main__":
    main()
