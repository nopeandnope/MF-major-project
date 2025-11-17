"""
Run the MF Advisor web server
"""
import uvicorn
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

if __name__ == "__main__":
    print("="*60)
    print("Starting MF Advisor Server...")
    print("="*60)
    print("Server running at: http://localhost:8000")
    print("Press CTRL+C to stop")
    print("="*60)
    
    # Run without reload to avoid the warning
    uvicorn.run("api.app:app", host="0.0.0.0", port=8000, reload=False)