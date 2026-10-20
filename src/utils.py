import os
import json

def load_json_file(filepath, default_data):
    if os.path.exists(filepath):
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return default_data
    return default_data

def save_json_file(filepath, data):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def calculate_ev(prob, odds):
    """Oblicza Oczekiwaną Wartość (Expected Value - EV %)."""
    if not odds or odds <= 1.0:
        return 0.0
    return (prob * odds - 1) * 100
  
