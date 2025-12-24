#!/usr/bin/env python3
"""
Setup script for the Scikit-Learn Network Security Tutorial
"""

import subprocess
import sys
import os


def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required.")
        print(f"   Current version: {sys.version}")
        return False
    print(f"✅ Python version: {sys.version.split()[0]}")
    return True


def install_requirements():
    """Install required packages"""
    print("\n📦 Installing required packages...")
    try:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-r", "requirements.txt"]
        )
        print("✅ All packages installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install packages: {e}")
        return False


def verify_installation():
    """Verify that all packages are installed correctly"""
    print("\n🔍 Verifying installation...")

    packages = [
        ("sklearn", "scikit-learn"),
        ("numpy", "NumPy"),
        ("pandas", "Pandas"),
        ("matplotlib", "Matplotlib"),
        ("seaborn", "Seaborn"),
    ]

    all_good = True
    for package, name in packages:
        try:
            __import__(package)
            print(f"   ✅ {name}")
        except ImportError:
            print(f"   ❌ {name} - Not installed")
            all_good = False

    return all_good


def run_quick_test():
    """Run a quick test to make sure everything works"""
    print("\n🧪 Running quick test...")
    try:
        from data_generator import NetworkDataGenerator

        generator = NetworkDataGenerator()

        # Test feature extraction
        test_ports = [22, 80, 443]
        features = generator.extract_features(test_ports)

        if len(features) == 10:
            print("✅ Data generator test passed!")
            return True
        else:
            print("❌ Data generator test failed!")
            return False

    except ImportError as e:
        print(f"❌ Quick test failed (ImportError): {e}")
        return False
    except AttributeError as e:
        print(f"❌ Quick test failed (AttributeError): {e}")
        return False
    except Exception as e:
        print(f"❌ Quick test failed (Other error): {e}")
        return False


def main():
    """Main setup function"""
    print("🎓 SCIKIT-LEARN NETWORK SECURITY TUTORIAL SETUP")
    print("=" * 60)

    # Check Python version
    if not check_python_version():
        return False

    # Install requirements
    if not install_requirements():
        return False

    # Verify installation
    if not verify_installation():
        print("\n❌ Some packages failed to install. Please check the errors above.")
        return False

    # Run quick test
    if not run_quick_test():
        print("\n❌ Quick test failed. Please check for errors.")
        return False

    print("\n🎉 SETUP COMPLETE!")
    print("=" * 60)
    print("🚀 You're ready to start the tutorial!")
    print("\n📚 To run the tutorial:")
    print("   Basic tutorial:      python email_spam_ml.py")
    print("   Interactive mode:    python email_spam_ml.py --mode interactive")
    print("   With visualizations: python email_spam_ml.py --mode advanced --visualize")

    return True


if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)
