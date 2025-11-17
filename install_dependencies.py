"""
Dependency Installation Script
Ensures all required packages are installed for the MF Advisor project
"""
import subprocess
import sys
import importlib

def install_package(package):
    """Install a package using pip"""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        return True
    except subprocess.CalledProcessError:
        return False

def check_import(package_name, import_name=None):
    """Check if a package can be imported"""
    try:
        if import_name is None:
            import_name = package_name
        importlib.import_module(import_name)
        return True
    except ImportError:
        return False

def main():
    """Main installation and verification function"""
    print("=" * 60)
    print("MF ADVISOR - DEPENDENCY INSTALLATION")
    print("=" * 60)

    # Required packages for enhanced features
    packages = [
        ("yfinance", "yfinance"),
        ("loguru", "loguru"),
        ("seaborn", "seaborn"),
        ("tensorboard", "tensorboard"),
        ("datasets", "datasets"),
        ("scikit-learn", "sklearn"),
        ("matplotlib", "matplotlib"),
        ("pandas", "pandas"),
        ("numpy", "numpy"),
        ("fastapi", "fastapi"),
        ("uvicorn", "uvicorn"),
        ("torch", "torch"),
        ("transformers", "transformers"),
        ("peft", "peft")
    ]

    print("Checking and installing packages...")
    print("-" * 60)

    missing_packages = []
    installed_packages = []

    for package, import_name in packages:
        print(f"Checking {package}...", end=" ")

        if check_import(import_name):
            print("OK - Already installed")
            installed_packages.append(package)
        else:
            print("MISSING - Installing...")
            if install_package(package):
                if check_import(import_name):
                    print(f"  SUCCESS - {package} installed")
                    installed_packages.append(package)
                else:
                    print(f"  FAILED - {package} installation failed")
                    missing_packages.append(package)
            else:
                print(f"  ERROR - Failed to install {package}")
                missing_packages.append(package)

    print("-" * 60)
    print("INSTALLATION SUMMARY")
    print("-" * 60)
    print(f"SUCCESS - Verified: {len(installed_packages)} packages")

    if missing_packages:
        print(f"FAILED - Missing packages: {len(missing_packages)}")
        for pkg in missing_packages:
            print(f"  - {pkg}")
        print("\nTry running: python -m pip install " + " ".join(missing_packages))
    else:
        print("SUCCESS - All packages are ready!")
        print("\nYou can now run:")
        print("  - python src/enhanced_20_year_collector.py")
        print("  - python train_enhanced_model.py")
        print("  - python run_server.py")

    print("=" * 60)

    # Test specific imports for the project
    print("Testing project-specific imports...")
    try:
        import yfinance as yf
        from loguru import logger
        print("SUCCESS - yfinance and loguru working")

        # Test the enhanced collector import
        import sys
        sys.path.append('.')
        from src.enhanced_20_year_collector import Enhanced20YearCollector
        print("SUCCESS - Enhanced 20-year collector ready")

        print("SUCCESS - ALL SYSTEMS READY!")

    except ImportError as e:
        print(f"ERROR - Import test failed: {e}")
        print("Please run the installation commands above")

if __name__ == "__main__":
    main()