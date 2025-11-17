"""
Check training status
"""
from pathlib import Path
import json
from datetime import datetime

def check_status():
    # Check different model directories
    models = {
        "fine_tuned": Path("models/fine_tuned"),
        "enhanced_model": Path("models/enhanced_model")
    }
    
    print("=" * 60)
    print("TRAINING STATUS CHECK")
    print("=" * 60)
    
    for name, path in models.items():
        print(f"\n{name}:")
        print("-" * 20)
        
        if not path.exists():
            print("  ❌ Model directory not found")
            continue
        
        print("  ✅ Model directory exists")
        
        # Check for key files
        files_to_check = [
            "adapter_config.json",
            "adapter_model.safetensors",
            "adapter_model.bin",
            "training_info.json",
            "training_complete.json"
        ]
        
        for file in files_to_check:
            if (path / file).exists():
                print(f"  ✅ {file} found")
                
                # Read training info if exists
                if "training" in file and file.endswith(".json"):
                    with open(path / file) as f:
                        info = json.load(f)
                        if "completed" in info:
                            print(f"     Completed: {info['completed']}")
                        if "epochs" in info:
                            print(f"     Epochs: {info['epochs']}")
                        if "final_loss" in info:
                            print(f"     Final loss: {info['final_loss']}")
    
    # Check logs
    print("\nLog Files:")
    print("-" * 20)
    log_dir = Path("logs")
    if log_dir.exists():
        for log_file in log_dir.glob("*.log"):
            size = log_file.stat().st_size / 1024  # KB
            modified = datetime.fromtimestamp(log_file.stat().st_mtime)
            print(f"  {log_file.name}: {size:.1f} KB, Modified: {modified:%Y-%m-%d %H:%M}")
            
            # Check last line
            try:
                with open(log_file, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    if lines:
                        last_line = lines[-1].strip()
                        if "complete" in last_line.lower():
                            print(f"    ✅ Training completed!")
                        elif "error" in last_line.lower() or "failed" in last_line.lower():
                            print(f"    ❌ Training failed!")
                        else:
                            print(f"    ⏳ Status unclear - check log")
            except:
                pass

if __name__ == "__main__":
    check_status()