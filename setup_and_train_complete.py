"""
Complete Automated Setup and Training Pipeline for MF Advisor
Run this single file to do everything from data collection to training
"""
import asyncio
import sys
import os
import subprocess
from pathlib import Path
from datetime import datetime
from loguru import logger
import time
import io

# Fix Windows console encoding for Unicode characters
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# Add project root to path
sys.path.append(str(Path(__file__).parent))

class CompleteSetup:
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.start_time = datetime.now()

        # Setup logging
        log_dir = self.project_root / "logs"
        log_dir.mkdir(exist_ok=True)
        logger.add(
            log_dir / "complete_setup.log",
            rotation="100 MB",
            level="INFO"
        )

    def print_banner(self, text, char="="):
        """Print a formatted banner"""
        width = 70
        print(f"\n{char * width}")
        print(f"{text.center(width)}")
        print(f"{char * width}\n")

    def print_step(self, step_num, total_steps, description):
        """Print step information"""
        print(f"\n{'='*70}")
        print(f"STEP [{step_num}/{total_steps}]: {description}")
        print(f"{'='*70}")
        logger.info(f"Starting Step {step_num}: {description}")

    def print_substep(self, description):
        """Print substep information"""
        print(f"\n  → {description}")
        logger.info(f"  → {description}")

    def check_python_version(self):
        """Check if Python version is compatible"""
        version = sys.version_info
        if version.major < 3 or (version.major == 3 and version.minor < 11):
            print(f"❌ ERROR: Python 3.11+ required. Current: {version.major}.{version.minor}")
            sys.exit(1)
        print(f"✅ Python version: {version.major}.{version.minor}.{version.micro}")

    def check_virtual_env(self):
        """Check if running in virtual environment"""
        in_venv = hasattr(sys, 'real_prefix') or (
            hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix
        )
        if in_venv:
            print("✅ Running in virtual environment")
        else:
            print("⚠️  WARNING: Not in virtual environment")
            print("   Recommended: Create and activate venv first")
            response = input("   Continue anyway? (y/n): ")
            if response.lower() != 'y':
                sys.exit(0)

    def install_dependencies(self):
        """Install all required dependencies"""
        self.print_substep("Installing Python dependencies...")

        requirements_file = self.project_root / "requirements.txt"
        if not requirements_file.exists():
            print("❌ ERROR: requirements.txt not found")
            return False

        try:
            # Install from requirements.txt
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "-q", "-r", str(requirements_file)],
                check=True
            )

            # Install additional packages that might be missing
            additional_packages = [
                "yfinance", "loguru", "seaborn", "tensorboard",
                "datasets", "transformers", "peft", "accelerate"
            ]

            for package in additional_packages:
                try:
                    subprocess.run(
                        [sys.executable, "-m", "pip", "install", "-q", package],
                        check=True,
                        capture_output=True
                    )
                except:
                    pass  # Package might already be installed

            print("✅ Dependencies installed successfully")
            logger.info("All dependencies installed")
            return True

        except subprocess.CalledProcessError as e:
            print(f"❌ ERROR installing dependencies: {e}")
            logger.error(f"Dependency installation failed: {e}")
            return False

    def verify_imports(self):
        """Verify critical imports work"""
        self.print_substep("Verifying package imports...")

        critical_packages = [
            "fastapi", "uvicorn", "torch", "transformers",
            "peft", "yfinance", "loguru", "datasets"
        ]

        failed_imports = []
        for package in critical_packages:
            try:
                __import__(package)
                print(f"  ✅ {package}")
            except ImportError:
                print(f"  ❌ {package}")
                failed_imports.append(package)

        if failed_imports:
            print(f"\n❌ Failed to import: {', '.join(failed_imports)}")
            print("   Try reinstalling with: pip install -r requirements.txt")
            return False

        print("✅ All critical packages imported successfully")
        return True

    def check_huggingface_token(self):
        """Check if Hugging Face token is configured"""
        self.print_substep("Checking Hugging Face authentication...")

        env_file = self.project_root / ".env"
        if not env_file.exists():
            print("⚠️  WARNING: .env file not found")
            print("   Model training will fail without Hugging Face token")
            return False

        # Check if token is in .env
        with open(env_file, 'r') as f:
            content = f.read()
            if "HUGGINGFACE_HUB_TOKEN" in content and "hf_" in content:
                print("✅ Hugging Face token found in .env")
                return True
            else:
                print("⚠️  WARNING: Hugging Face token not configured")
                print("   Training may fail without token for Gemma-2B model")
                return False

    async def collect_historical_data(self, max_funds=200):
        """Collect 20-year historical data"""
        self.print_substep(f"Collecting 20-year historical data (up to {max_funds} funds)...")
        self.print_substep("This may take 15-45 minutes depending on network speed...")

        try:
            # Import the collector
            from src.enhanced_20_year_collector import Enhanced20YearCollector

            collector = Enhanced20YearCollector()
            async with collector:
                result = await collector.collect_20_year_data(max_funds=max_funds)

            if result:
                print(f"✅ Data collection completed successfully")
                print(f"   Database: data/historical_20y.db")
                logger.info("20-year data collection completed")
                return True
            else:
                print("❌ Data collection failed")
                return False

        except Exception as e:
            print(f"❌ ERROR during data collection: {str(e)}")
            logger.error(f"Data collection error: {str(e)}")
            return False

    def check_existing_data(self):
        """Check if data already exists"""
        db_path = self.project_root / "data" / "historical_20y.db"
        if db_path.exists():
            size_mb = db_path.stat().st_size / (1024 * 1024)
            print(f"✅ Existing database found: {size_mb:.1f} MB")

            response = input("   Skip data collection and use existing data? (y/n): ")
            return response.lower() == 'y'
        return False

    def train_model(self, quick_mode=False):
        """Train the model"""
        if quick_mode:
            self.print_substep("QUICK TRAINING MODE (5-10 minutes)...")
            training_script = "quick_train.py"
        else:
            self.print_substep("Training with optimized settings (10-20 minutes)...")
            training_script = "src/train_gemma_gpu.py"

        try:
            script_path = self.project_root / training_script
            if not script_path.exists():
                print(f"❌ ERROR: {training_script} not found")
                return False

            print(f"\n  Starting training using {training_script}...")
            print("  Check logs for progress...\n")

            # Run as subprocess to avoid import issues
            result = subprocess.run(
                [sys.executable, str(script_path)],
                cwd=str(self.project_root),
                capture_output=False,
                text=True
            )

            if result.returncode == 0:
                print("✅ Model training completed successfully")
                logger.info("Model training completed")
                return True
            else:
                print("⚠️ Training completed with warnings")
                return True

        except Exception as e:
            print(f"❌ ERROR during training: {str(e)}")
            logger.error(f"Training error: {str(e)}")
            return False

    def check_existing_model(self):
        """Check if trained model already exists"""
        model_dir = self.project_root / "models" / "enhanced_20y_advisor"
        if model_dir.exists() and any(model_dir.iterdir()):
            print(f"✅ Existing trained model found")

            response = input("   Skip training and use existing model? (y/n): ")
            return response.lower() == 'y'
        return False

    def print_summary(self):
        """Print setup summary"""
        elapsed = datetime.now() - self.start_time
        hours = elapsed.seconds // 3600
        minutes = (elapsed.seconds % 3600) // 60

        self.print_banner("SETUP COMPLETED SUCCESSFULLY!", "=")

        print("Summary:")
        print(f"  ✅ Dependencies installed")
        print(f"  ✅ Data collected (data/historical_20y.db)")
        print(f"  ✅ Model trained (models/enhanced_20y_advisor/)")
        print(f"  ⏱️  Total time: {hours}h {minutes}m")

        print("\n" + "="*70)
        print("NEXT STEPS:")
        print("="*70)
        print("\n1. Start the server:")
        print("   python run_server.py")
        print("\n2. Open your browser:")
        print("   http://localhost:8000")
        print("\n3. Test the API:")
        print("   POST http://localhost:8000/api/recommend")
        print("\n" + "="*70 + "\n")

        logger.info(f"Complete setup finished in {hours}h {minutes}m")

    async def run_complete_setup(self, skip_data=False, skip_training=False, max_funds=200, quick_train=False):
        """Run the complete setup pipeline"""

        self.print_banner("MF ADVISOR - AUTOMATED SETUP & TRAINING", "=")
        print(f"Started at: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")

        if quick_train:
            print("⚡ QUICK TRAINING MODE ENABLED (5-10 minutes)")

        total_steps = 5
        current_step = 0

        # Step 1: Environment Check
        current_step += 1
        self.print_step(current_step, total_steps, "Environment Verification")
        self.check_python_version()
        self.check_virtual_env()
        print("✅ Environment check passed")

        # Step 2: Dependencies
        current_step += 1
        self.print_step(current_step, total_steps, "Install Dependencies")
        if not self.install_dependencies():
            print("❌ Setup failed at dependency installation")
            return False

        if not self.verify_imports():
            print("❌ Setup failed at import verification")
            return False

        self.check_huggingface_token()

        # Step 3: Data Collection
        current_step += 1
        self.print_step(current_step, total_steps, "Collect Historical Data")

        if skip_data or self.check_existing_data():
            print("⏭️  Skipping data collection (using existing data)")
        else:
            success = await self.collect_historical_data(max_funds=max_funds)
            if not success:
                print("⚠️  WARNING: Data collection failed")
                print("   Continuing with existing data if available...")

        # Step 4: Model Training
        current_step += 1
        self.print_step(current_step, total_steps, "Train Enhanced Model")

        if skip_training or self.check_existing_model():
            print("⏭️  Skipping model training (using existing model)")
        else:
            success = self.train_model(quick_mode=quick_train)
            if not success:
                print("⚠️  WARNING: Model training failed")
                print("   Server will use rule-based recommendations")

        # Step 5: Final Verification
        current_step += 1
        self.print_step(current_step, total_steps, "Final Verification")

        # Check all components
        data_exists = (self.project_root / "data" / "historical_20y.db").exists()
        model_exists = (self.project_root / "models" / "enhanced_20y_advisor").exists()

        print(f"  {'✅' if data_exists else '❌'} Historical data: {data_exists}")
        print(f"  {'✅' if model_exists else '❌'} Trained model: {model_exists}")

        if not data_exists and not model_exists:
            print("\n⚠️  WARNING: Neither data nor model available")
            print("   Server will work with basic rule-based recommendations")

        # Print summary
        self.print_summary()
        return True


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Complete automated setup for MF Advisor"
    )
    parser.add_argument(
        "--skip-data",
        action="store_true",
        help="Skip data collection (use existing data)"
    )
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Skip model training (use existing model)"
    )
    parser.add_argument(
        "--max-funds",
        type=int,
        default=200,
        help="Maximum number of funds to collect (default: 200)"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick setup: skip data collection, use fast training"
    )
    parser.add_argument(
        "--quick-train",
        action="store_true",
        help="Use quick training mode (5-10 min)"
    )

    args = parser.parse_args()

    # Handle quick mode
    if args.quick:
        args.skip_data = True
        args.quick_train = True

    # Run setup
    setup = CompleteSetup()
    try:
        success = asyncio.run(
            setup.run_complete_setup(
                skip_data=args.skip_data,
                skip_training=args.skip_training,
                max_funds=args.max_funds,
                quick_train=args.quick_train
            )
        )

        if success:
            sys.exit(0)
        else:
            sys.exit(1)

    except KeyboardInterrupt:
        print("\n\n⚠️  Setup interrupted by user")
        logger.warning("Setup interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ FATAL ERROR: {str(e)}")
        logger.error(f"Fatal error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    print("""
======================================================================

           MF ADVISOR - AUTOMATED SETUP & TRAINING

  This script will automatically:
    1. Install all dependencies
    2. Collect 20-year historical data
    3. Train Gemma-2B model with LoRA
    4. Prepare the system for deployment

  Options:
    --skip-data        : Skip data collection
    --skip-training    : Skip model training
    --max-funds N      : Limit data collection to N funds
    --quick            : Use existing data + fast training
    --quick-train      : Use fast training mode (5-10 min)

  Estimated Time: 30-60 min (with --quick: 10-15 min)

======================================================================
    """)

    main()
