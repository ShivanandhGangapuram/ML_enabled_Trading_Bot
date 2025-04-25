# Installation Guide

This guide will walk you through the process of installing and setting up the ML-enabled Trading Bot on your system.

## System Requirements

Before you begin, make sure your system meets the following requirements:

- **Operating System**: Windows, macOS, or Linux
- **Python**: Version 3.6 or higher
- **RAM**: 8GB minimum (16GB recommended for larger datasets)
- **Processor**: Intel i5 or equivalent (or better)
- **Disk Space**: At least 1GB of free space

## Step 1: Clone the Repository

```bash
git clone https://github.com/ShivanandhGangapuram/ML_enabled_Trading_Bot.git
cd ML_enabled_Trading_Bot
```

## Step 2: Set Up a Virtual Environment (Recommended)

It's recommended to use a virtual environment to avoid conflicts with other Python packages.

### For Windows:

```bash
python -m venv venv
venv\Scripts\activate
```

### For macOS/Linux:

```bash
python3 -m venv venv
source venv/bin/activate
```

## Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

This will install all the required packages, including:
- backtrader
- pandas
- numpy
- scikit-learn
- matplotlib
- pillow
- yfinance (for Yahoo Finance data)
- fyers-apiv3 (for Fyers API data)

## Step 4: Configure API Credentials

If you plan to use the Fyers API for data, you'll need to set up your API credentials:

1. Copy the template configuration file:
   ```bash
   cp config_template.py config.py
   ```

2. Edit `config.py` with your actual API credentials:
   ```python
   FYERS_CLIENT_ID = "YOUR_CLIENT_ID_HERE"
   FYERS_SECRET_KEY = "YOUR_SECRET_KEY_HERE"
   FYERS_REDIRECT_URL = "YOUR_REDIRECT_URL_HERE"
   FYERS_USERNAME = "YOUR_USERNAME_HERE"
   ```

## Step 5: Verify Installation

Run the setup verification script to ensure everything is installed correctly:

```bash
python check_setup.py
```

If all checks pass, you're ready to use the trading bot!

## Step 6: Run the Trading Bot GUI

To start the trading bot with the graphical user interface:

```bash
python run_trading_bot.py
```

## Troubleshooting

If you encounter any issues during installation, check the following:

- Make sure you have the correct Python version installed
- Ensure all dependencies are installed correctly
- Check that your API credentials are set up properly (if using Fyers API)
- Verify that you have sufficient disk space and RAM

For more detailed troubleshooting, see the [Common Issues](Common-Issues) page.