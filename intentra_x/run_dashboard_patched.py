"""
Patched runner for INTENTRA-X dashboard that handles config errors
"""
import sys
import os

# Create credentials file to skip email prompt
cred_dir = os.path.join(os.path.expanduser('~'), '.streamlit')
cred_file = os.path.join(cred_dir, 'credentials.toml')
os.makedirs(cred_dir, exist_ok=True)
if not os.path.exists(cred_file):
    with open(cred_file, 'w') as f:
        f.write('[general]\nemail = ""\n')

# Monkey-patch the config loader before importing streamlit
def patch_streamlit_config():
    import streamlit.config as config_module
    original_get_config = config_module.get_config_options
    
    def patched_get_config(*args, **kwargs):
        try:
            return original_get_config(*args, **kwargs)
        except (UnicodeDecodeError, Exception) as e:
            print(f"Warning: Skipping corrupted config file: {e}")
            # Return with minimal config
            return original_get_config(force_reparse=False, options_from_flags=kwargs.get('options_from_flags', {}))
    
    config_module.get_config_options = patched_get_config

# Set environment before import
os.environ['STREAMLIT_BROWSER_GATHER_USAGE_STATS'] = 'false'

# Import and patch
try:
    import streamlit
    patch_streamlit_config()
except:
    pass

# Run streamlit
from streamlit.web import cli as stcli

if __name__ == '__main__':
    sys.argv = ["streamlit", "run", "dashboard.py"]
    sys.exit(stcli.main())
