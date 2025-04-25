# Live Paper Trading Guide

This guide explains how to use the Live Paper Trading feature of the Trading Bot.

## Overview

The Live Paper Trading feature allows you to simulate real trading without risking actual money. It connects to the Fyers API to get real-time market data and executes trades in a simulated environment.

The system offers two modes:
1. **Automated (Machine-Driven)**: The bot automatically executes trades based on the selected strategy.
2. **User-Selected Strategy**: You choose the strategy and can manually execute trades while the bot provides signals.

## Getting Started

### Prerequisites
- A Fyers API account with API credentials
- The Trading Bot application installed and configured

### Setting Up Fyers API
1. Create an account on [Fyers](https://fyers.in/)
2. Apply for API access in your Fyers account
3. Create an App in the Fyers API dashboard
4. Note your Client ID, Secret Key, and set your Redirect URI

### Configuration
1. Open the Trading Bot application
2. Navigate to the "Live Trading" tab
3. Enter your Fyers API credentials when prompted during the first run

## Using Live Paper Trading

### Step 1: Select Trading Mode
- **Automated (Machine-Driven)**: The bot will automatically execute trades based on the selected strategy.
- **User-Selected Strategy**: You'll receive trading signals but must manually execute trades.

### Step 2: Configure Trading Parameters
- **Stock Symbol**: Select the stock you want to trade (e.g., NSE:RELIANCE-EQ)
- **Strategy**: Choose from available strategies (SMA Crossover, RSI, MACD, etc.)
- **Strategy Parameters**: Customize the indicators based on your preferences
- **Initial Capital**: Set your starting capital for paper trading
- **Commission**: Set the commission rate to simulate real trading costs

### Step 3: Start Trading
1. Click "Start Live Paper Trading"
2. Authenticate with Fyers API when prompted
3. The bot will fetch historical data to initialize the strategy
4. Trading will begin based on your selected mode

### Step 4: Monitor Performance
- The trading log shows all actions and signals
- The portfolio status displays current position, value, and profit/loss
- In User mode, use the Buy and Sell buttons to execute trades manually

### Step 5: End Trading Session
1. Click "Stop Trading" when you're done
2. The final performance summary will be displayed
3. Trading state is saved for future reference

## Trading Modes Explained

### Automated (Machine-Driven) Mode
In this mode, the bot:
- Continuously monitors market data
- Applies the selected strategy to identify trading signals
- Automatically executes buy and sell orders based on signals
- Manages position sizing and risk
- Tracks performance metrics

### User-Selected Strategy Mode
In this mode, you:
- Select and customize a trading strategy
- Receive buy and sell signals from the strategy
- Decide whether to act on those signals
- Manually execute trades using the Buy and Sell buttons
- Can specify the quantity for each trade or use available funds

## Available Strategies

1. **SMA Crossover**: Trades based on when a fast Simple Moving Average crosses a slow Simple Moving Average.
2. **RSI**: Uses the Relative Strength Index to identify overbought and oversold conditions.
3. **MACD**: Uses the Moving Average Convergence Divergence indicator for trend following.
4. **Bollinger Bands**: Trades based on price movements relative to volatility bands.
5. **Dual MA**: Uses two different types of moving averages (SMA/EMA) for crossover signals.

## Customizing Strategies

Each strategy has parameters you can adjust:

### SMA Crossover
- Fast SMA Period: Length of the faster moving average
- Slow SMA Period: Length of the slower moving average

### RSI
- RSI Period: Length of the RSI calculation period
- Oversold Level: RSI level considered oversold (buy signal)
- Overbought Level: RSI level considered overbought (sell signal)

### MACD
- Fast Period: Length of the fast EMA
- Slow Period: Length of the slow EMA
- Signal Period: Length of the signal line EMA

### Bollinger Bands
- Period: Length of the moving average
- Std Dev Factor: Number of standard deviations for the bands

### Dual MA
- MA1 Period: Length of the first moving average
- MA2 Period: Length of the second moving average
- MA1 Type: Type of the first moving average (SMA or EMA)
- MA2 Type: Type of the second moving average (SMA or EMA)

## Tips for Effective Paper Trading

1. **Start with a realistic amount**: Use an initial capital amount that matches what you would actually trade with.
2. **Include commissions**: Set realistic commission rates to account for trading costs.
3. **Try different strategies**: Test multiple strategies to find what works best for different market conditions.
4. **Compare modes**: Try both automated and user-selected modes to see which suits your trading style.
5. **Keep notes**: Document what works and what doesn't for future reference.
6. **Be patient**: Let strategies run long enough to see meaningful results.
7. **Avoid overtrading**: Don't make excessive manual trades in user mode.

## Troubleshooting

### Authentication Issues
- Ensure your Fyers API credentials are correct
- Check that your Redirect URI matches exactly what's in your Fyers API app settings
- Make sure your Fyers account has API access enabled

### Data Issues
- If no data appears, check your internet connection
- Verify the stock symbol is correct and actively trading
- Ensure your Fyers API session hasn't expired

### Trading Issues
- If orders aren't executing in automated mode, check the trading log for errors
- Verify you have sufficient simulated capital for the trades
- Check that your strategy parameters are reasonable

## Next Steps

After successful paper trading:
1. Analyze your trading performance
2. Refine your strategies based on results
3. Consider implementing additional risk management rules
4. When confident, consider transitioning to real trading (requires separate setup)