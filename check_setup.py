#!/usr/bin/env python3
"""
Setup verification script for Trading Bot

This script checks if your environment is properly configured
and all required dependencies are installed.
"""

import sys
import os
import importlib
import platform

def check_python_version():
    """Check if Python version is compatible"""
    print(f"Python version: {platform.python_version()}")
    if sys.version_info < (3, 6):
        print("❌ Python 3.6 or higher is required")
        return False
    print("✅ Python version is compatible")
    return True

def check_required_packages():
    """Check if required packages are installed"""
    required_packages = [
        "pandas",
        "numpy",
        "scikit-learn",
        "matplotlib",
        "pillow",
        "backtrader",
    ]
    
    optional_packages = [
        "yfinance",
        "fyers_apiv3",
    ]
    
    all_required_installed = True
    print("\nChecking required packages:")
    for package in required_packages:
        try:
            importlib.import_module(package)
            print(f"✅ {package} is installed")
        except ImportError:
            print(f"❌ {package} is NOT installed")
            all_required_installed = False
    
    print("\nChecking optional packages:")
    for package in optional_packages:
        try:
            importlib.import_module(package)
            print(f"✅ {package} is installed")
        except ImportError:
            print(f"⚠️ {package} is NOT installed (optional)")
    
    return all_required_installed

def check_config_file():
    """Check if config file exists and is properly set up"""
    print("\nChecking configuration:")
    
    if not os.path.exists("config_template.py"):
        print("❌ config_template.py is missing")
        return False
    
    print("✅ config_template.py exists")
    
    if not os.path.exists("config.py"):
        print("⚠️ config.py is missing - you need to create it from the template")
        print("    Copy config_template.py to config.py and add your API credentials")
        return False
    
    try:
        # Try to import without exposing the actual values
        import config
        has_fyers = hasattr(config, "FYERS_CLIENT_ID") and hasattr(config, "FYERS_SECRET_KEY")
        
        if has_fyers:
            print("✅ config.py has Fyers API credentials")
        else:
            print("⚠️ config.py exists but may be missing some credentials")
        
        return True
    except Exception as e:
        print(f"❌ Error in config.py: {e}")
        return False

def check_gitignore():
    """Check if .gitignore is properly set up"""
    print("\nChecking .gitignore:")
    
    if not os.path.exists(".gitignore"):
        print("❌ .gitignore file is missing")
        return False
    
    with open(".gitignore", "r") as f:
        content = f.read()
    
    critical_patterns = ["config.py", "*.csv", "__pycache__/", "*.pyc"]
    all_patterns_present = True
    
    for pattern in critical_patterns:
        if pattern in content:
            print(f"✅ .gitignore contains {pattern}")
        else:
            print(f"❌ .gitignore is missing {pattern}")
            all_patterns_present = False
    
    return all_patterns_present

def main():
    """Run all checks and provide summary"""
    print("=" * 60)
    print("TRADING BOT SETUP VERIFICATION")
    print("=" * 60)
    
    python_ok = check_python_version()
    packages_ok = check_required_packages()
    config_ok = check_config_file()
    gitignore_ok = check_gitignore()
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    if python_ok and packages_ok and config_ok and gitignore_ok:
        print("✅ Your environment is properly set up!")
    else:
        print("⚠️ There are some issues with your setup.")
        print("   Please address the items marked with ❌ above.")
    
    print("\nReady for GitHub?")
    if gitignore_ok and config_ok:
        print("✅ Your project appears to be ready for GitHub!")
        print("   The .gitignore file is properly configured to exclude sensitive files.")
    else:
        print("❌ Your project is NOT ready for GitHub yet.")
        print("   Please fix the .gitignore and config issues before pushing to GitHub.")
    
    print("\nFor more information, see the README.md and FYERS_AUTHORIZATION_GUIDE.md files.")
    print("=" * 60)

if __name__ == "__main__":
    main()