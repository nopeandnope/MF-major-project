"""
Save model state for easy sharing
"""
import torch
import json
from pathlib import Path

def save_project_state():
    """Save all necessary files for project sharing"""
    
    # Create distribution folder
    dist_dir = Path("distribution")
    dist_dir.mkdir(exist_ok=True)
    
    # Save configuration
    config = {
        "model": "microsoft/phi-2",
        "trained": True,
        "version": "1.0.0",
        "requirements": [
            "torch",
            "transformers",
            "fastapi",
            "uvicorn",
            "pydantic",
            "numpy",
            "pandas"
        ]
    }
    
    with open(dist_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    print("Project state saved to distribution/")
    print("To share: Compress the entire MF_Advisor_Project folder")
    print("Recipients need to:")
    print("1. Extract the folder")
    print("2. Run: pip install -r requirements.txt")
    print("3. Run: python run_server.py")

if __name__ == "__main__":
    save_project_state()