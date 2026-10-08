"""
Streamlit Entrypoint Alias for AURA-AQI Bayesian Dashboard
"""
import runpy
from pathlib import Path

if __name__ == "__main__":
    main_path = Path(__file__).parent / "main.py"
    runpy.run_path(str(main_path), run_name="__main__")
