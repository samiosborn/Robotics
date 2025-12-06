# src/utils/load_config.py
import yaml

# Load Config
def load_config(path):
  with open(path, "r") as f:
    return yaml.safe_load(f)
