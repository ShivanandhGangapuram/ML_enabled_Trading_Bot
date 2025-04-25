# ML-enabled Trading Bot Examples

This document provides practical examples of how to use the ML-enabled Trading Bot for various trading scenarios. These examples demonstrate the flexibility and power of the system.

## Basic Examples

### Example 1: Simple Moving Average Crossover Strategy

This example shows how to backtest a traditional moving average crossover strategy:

```python
from strategies import SMACrossoverStrategy
from backtest import run_backtest
import pandas as pd

# Load historical data
data = pd.read_csv('NSE_RELIANCE-EQ_D.csv', index_col='datetime', parse_dates=True)

# Define strategy parameters
params = {
    'short_window': 20,
    'long_window': 50
}

# Run backtest
results = run_backtest(data, SMACrossoverStrategy, params)

# Print results
print(f"Total Return: {results['total_return']:.2f}%")
print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
print(f"Max Drawdown: {results['max_drawdown']:.2f}%")
```

### Example 2: Machine Learning Strategy

This example demonstrates how to train and backtest a machine learning-based strategy:

```python
from ml_strategy import MLStrategy, train_model
from backtest_ml import run_ml_backtest
import pandas as pd

# Load historical data
data = pd.read_csv('NSE_RELIANCE-EQ_D.csv', index_col='datetime', parse_dates=True)

# Train the ML model
model, features = train_model(data, test_size=0.2)

# Define strategy parameters
params = {
    'prediction_threshold': 0.6,  # Probability threshold for buy signals
    'stop_loss': 0.05,            # 5% stop loss
    'take_profit': 0.1            # 10% take profit
}

# Run backtest with the trained model
results = run_ml_backtest(data, model, features, params)

# Print results
print(f"Total Return: {results['total_return']:.2f}%")
print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
print(f"Max Drawdown: {results['max_drawdown']:.2f}%")
```

## Intermediate Examples

### Example 3: Comparing Multiple Strategies

This example shows how to compare the performance of different strategies:

```python
from compare_strategies import compare_strategies
import pandas as pd

# Load historical data
data = pd.read_csv('NSE_RELIANCE-EQ_D.csv', index_col='datetime', parse_dates=True)

# Define strategies to compare
strategies = [
    {
        'name': 'SMA Crossover (20,50)',
        'type': 'traditional',
        'params': {'short_window': 20, 'long_window': 50}
    },
    {
        'name': 'SMA Crossover (10,30)',
        'type': 'traditional',
        'params': {'short_window': 10, 'long_window': 30}
    },
    {
        'name': 'ML Strategy',
        'type': 'ml',
        'params': {'prediction_threshold': 0.6}
    }
]

# Run comparison
results = compare_strategies(data, strategies)

# Results are saved as 'strategy_comparison.png'
print("Comparison completed and saved to 'strategy_comparison.png'")
```

### Example 4: Custom Feature Engineering

This example demonstrates how to add custom features to the ML model:

```python
from ml_strategy import add_features, train_model_with_custom_features
import pandas as pd
import numpy as np

# Load historical data
data = pd.read_csv('NSE_RELIANCE-EQ_D.csv', index_col='datetime', parse_dates=True)

# Define custom feature function
def add_custom_features(df):
    # Add price momentum features
    df['momentum_5d'] = df['close'].pct_change(5)
    df['momentum_10d'] = df['close'].pct_change(10)
    
    # Add volatility features
    df['volatility_5d'] = df['close'].rolling(5).std() / df['close']
    df['volatility_10d'] = df['close'].rolling(10).std() / df['close']
    
    # Add custom indicator
    df['custom_indicator'] = (df['close'] - df['close'].rolling(20).mean()) / df['close'].rolling(20).std()
    
    return df

# Add standard features and custom features
data_with_features = add_features(data)
data_with_features = add_custom_features(data_with_features)

# Train model with custom features
model, features = train_model_with_custom_features(data_with_features, test_size=0.2)

print(f"Model trained with {len(features)} features")
print(f"Top 5 important features: {features[:5]}")
```

## Advanced Examples

### Example 5: Multi-Asset Portfolio Strategy

This example shows how to implement a portfolio strategy across multiple assets:

```python
from backtest_multi import run_portfolio_backtest
import pandas as pd
import os

# Load multiple asset data
assets = ['RELIANCE', 'TCS', 'HDFCBANK', 'INFY', 'ITC']
data_dict = {}

for asset in assets:
    file_path = f'NSE_{asset}-EQ_D.csv'
    if os.path.exists(file_path):
        data_dict[asset] = pd.read_csv(file_path, index_col='datetime', parse_dates=True)

# Define portfolio strategy parameters
portfolio_params = {
    'allocation_method': 'equal_weight',  # Equal weight allocation
    'rebalance_frequency': 20,            # Rebalance every 20 trading days
    'max_position_size': 0.2,             # Maximum 20% in any single asset
    'strategy_type': 'ml',                # Use ML strategy for signals
    'prediction_threshold': 0.6           # Probability threshold for buy signals
}

# Run portfolio backtest
portfolio_results = run_portfolio_backtest(data_dict, portfolio_params)

# Print portfolio results
print(f"Portfolio Return: {portfolio_results['total_return']:.2f}%")
print(f"Portfolio Sharpe: {portfolio_results['sharpe_ratio']:.2f}")
print(f"Portfolio Max Drawdown: {portfolio_results['max_drawdown']:.2f}%")
```

### Example 6: Live Paper Trading Simulation

This example demonstrates how to set up a live paper trading simulation:

```python
from live_paper_trading import PaperTradingSimulation
import pandas as pd
import datetime

# Load historical data for simulation
data = pd.read_csv('NSE_RELIANCE-EQ_D.csv', index_col='datetime', parse_dates=True)

# Define simulation parameters
sim_params = {
    'initial_capital': 100000,
    'strategy_type': 'ml',
    'prediction_threshold': 0.65,
    'stop_loss': 0.03,
    'take_profit': 0.06,
    'max_position_size': 0.2,
    'commission': 0.0005  # 0.05% commission per trade
}

# Create simulation
simulation = PaperTradingSimulation(data, sim_params)

# Run simulation for a specific period
start_date = datetime.datetime(2023, 1, 1)
end_date = datetime.datetime(2023, 3, 31)
simulation_results = simulation.run(start_date, end_date)

# Print simulation results
print(f"Simulation Period: {start_date.date()} to {end_date.date()}")
print(f"Final Portfolio Value: ${simulation_results['final_value']:.2f}")
print(f"Total Return: {simulation_results['total_return']:.2f}%")
print(f"Number of Trades: {simulation_results['num_trades']}")
print(f"Win Rate: {simulation_results['win_rate']:.2f}%")
```

## Visualization Examples

### Example 7: Advanced Performance Visualization

This example shows how to create advanced visualizations of strategy performance:

```python
from ml_strategy import MLStrategy, train_model
from backtest_ml import run_ml_backtest
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load historical data
data = pd.read_csv('NSE_RELIANCE-EQ_D.csv', index_col='datetime', parse_dates=True)

# Train the ML model and run backtest
model, features = train_model(data, test_size=0.2)
results = run_ml_backtest(data, model, features, {'prediction_threshold': 0.6})

# Create advanced visualization
plt.figure(figsize=(15, 10))

# Plot 1: Equity curve
plt.subplot(2, 2, 1)
plt.plot(results['equity_curve'])
plt.title('Equity Curve')
plt.grid(True)

# Plot 2: Drawdown
plt.subplot(2, 2, 2)
plt.fill_between(results['drawdown'].index, results['drawdown'].values, 0, alpha=0.3, color='red')
plt.title('Drawdown')
plt.grid(True)

# Plot 3: Monthly returns heatmap
monthly_returns = results['returns'].resample('M').sum().unstack().fillna(0)
plt.subplot(2, 2, 3)
sns.heatmap(monthly_returns, cmap='RdYlGn', annot=True, fmt='.2%')
plt.title('Monthly Returns')

# Plot 4: Trade distribution
plt.subplot(2, 2, 4)
plt.hist(results['trade_returns'], bins=20, alpha=0.7)
plt.axvline(0, color='r', linestyle='--')
plt.title('Trade Return Distribution')
plt.grid(True)

plt.tight_layout()
plt.savefig('advanced_performance_visualization.png', dpi=300)
plt.close()

print("Advanced visualization saved to 'advanced_performance_visualization.png'")
```

These examples demonstrate various ways to use the ML-enabled Trading Bot for different trading scenarios and analysis tasks. Feel free to adapt them to your specific needs or use them as a starting point for your own experiments.