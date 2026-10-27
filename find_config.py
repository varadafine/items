import os
import glob

# Check common locations
locations = [
    os.path.join(os.path.expanduser('~'), '.streamlit', 'config.toml'),
    os.path.join(os.getcwd(), '.streamlit', 'config.toml'),
    'C:\\Users\\guruc\\.streamlit\\config.toml',
]

print("Checking for Streamlit config files:")
for loc in locations:
    if os.path.exists(loc):
        print(f"FOUND: {loc}")
        try:
            with open(loc, 'rb') as f:
                first_bytes = f.read(10)
                print(f"  First bytes: {first_bytes}")
        except Exception as e:
            print(f"  Error reading: {e}")
    else:
        print(f"NOT FOUND: {loc}")

# Search in AppData
appdata = os.path.join(os.path.expanduser('~'), 'AppData')
print(f"\nSearching in {appdata}...")
for root, dirs, files in os.walk(appdata):
    if 'config.toml' in files and 'streamlit' in root.lower():
        config_path = os.path.join(root, 'config.toml')
        print(f"FOUND: {config_path}")
        try:
            with open(config_path, 'rb') as f:
                first_bytes = f.read(10)
                print(f"  First bytes: {first_bytes}")
        except Exception as e:
            print(f"  Error: {e}")
