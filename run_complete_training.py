"""
Complete pipeline to collect data and train model
"""
import asyncio
import sys
from pathlib import Path
from loguru import logger

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from api.models import init_db
from src.historical_data_collector import run_historical_collection
from src.generate_gemma_training_data import TrainingDataGenerator
from src.train_gemma_gpu import main as train_model

async def run_complete_pipeline():
    """Run the complete training pipeline"""
    
    print("="*60)
    print("MUTUAL FUND ADVISOR - COMPLETE TRAINING PIPELINE")
    print("="*60)
    
    # Step 1: Initialize Database
    print("\n[1/4] Initializing database...")
    init_db()
    print("[OK] Database initialized")

    # Step 2: Collect Historical Data
    print("\n[2/4] Collecting 5-10 years of historical data...")
    print("This may take 30-60 minutes...")
    await run_historical_collection()
    print("[OK] Historical data collected")

    # Step 3: Generate Training Data
    print("\n[3/4] Generating training data...")
    generator = TrainingDataGenerator()
    generator.generate_quality_training_data()
    print("[OK] Training data generated")

    # Step 4: Train Model on GPU
    print("\n[4/4] Training Gemma model on GPU...")
    print("This may take 2-4 hours depending on GPU...")
    train_model()
    print("[OK] Model training completed")
    
    print("\n" + "="*60)
    print("TRAINING PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*60)
    print("\nYour model is ready for deployment!")
    print("Start the API server with: python api/simple_app.py")

if __name__ == "__main__":
    asyncio.run(run_complete_pipeline())