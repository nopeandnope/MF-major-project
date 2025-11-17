"""Configuration management"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

class Settings:
    # Paths
    BASE_DIR = Path(__file__).parent.parent
    DATA_DIR = BASE_DIR / "data"
    RAW_DATA_DIR = DATA_DIR / "raw"
    PROCESSED_DATA_DIR = DATA_DIR / "processed"
    TRAINING_DATA_DIR = DATA_DIR / "training"
    MODELS_DIR = BASE_DIR / "models"
    LOGS_DIR = BASE_DIR / "logs"
    
    # Database
    MYSQL_HOST = os.getenv("MYSQL_HOST", "127.0.0.1")
    MYSQL_PORT = int(os.getenv("MYSQL_PORT", "3306"))
    MYSQL_USER = os.getenv("MYSQL_USER", "mf_user")
    MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD", "secure_password_123")
    MYSQL_DATABASE = os.getenv("MYSQL_DATABASE", "mf_advisor")
    
    # API
    API_HOST = os.getenv("APP_HOST", "127.0.0.1")
    API_PORT = int(os.getenv("APP_PORT", "8000"))
    DEBUG = os.getenv("APP_DEBUG", "True").lower() == "true"
    
    # Model
    MODEL_NAME = os.getenv("MODEL_NAME", "google/gemma-2b")
    USE_GPU = os.getenv("USE_GPU", "False").lower() == "true"
    
    # API Keys
    NEWS_API_KEY = os.getenv("NEWS_API_KEY")

settings = Settings()

# Create directories
for dir_path in [
    settings.RAW_DATA_DIR,
    settings.PROCESSED_DATA_DIR,
    settings.TRAINING_DATA_DIR,
    settings.MODELS_DIR,
    settings.LOGS_DIR
]:
    dir_path.mkdir(parents=True, exist_ok=True)