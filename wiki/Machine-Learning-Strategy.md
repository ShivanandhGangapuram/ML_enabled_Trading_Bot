# Machine Learning Strategy

This page explains the machine learning approach used in the trading bot, how it works, and how you can customize it.

## Overview

The ML-enabled Trading Bot uses a supervised machine learning approach to predict price movements and generate trading signals. The core of this approach is a Random Forest classifier that is trained on historical price data with various technical indicators as features.

## How It Works

### 1. Feature Engineering

The system creates over 30 technical indicators and features from the raw price data, including:

- Moving averages (simple, exponential, weighted)
- Oscillators (RSI, MACD, Stochastic)
- Volatility indicators (Bollinger Bands, ATR)
- Volume indicators
- Price patterns
- Trend indicators

These features are calculated using various lookback periods to capture different time horizons.

### 2. Target Variable

The target variable is a binary classification:
- 1 (Buy/Long): If the price will increase by a certain percentage within a specific time period
- 0 (Sell/Short): If the price will decrease or remain relatively unchanged

### 3. Model Training

The Random Forest classifier is trained on historical data with the following process:

1. Data is split into training and testing sets (typically 80/20)
2. Features are normalized/standardized
3. The model is trained on the training set
4. Hyperparameters are optimized using cross-validation
5. The model is evaluated on the testing set

### 4. Signal Generation

Once trained, the model generates trading signals as follows:

1. For each new data point, features are calculated
2. The model predicts the probability of a price increase
3. If the probability exceeds a threshold (default: 0.6), a buy signal is generated
4. If the probability falls below a threshold, a sell signal is generated

### 5. Risk Management

The ML strategy includes risk management components:
- Position sizing based on prediction confidence
- Stop-loss and take-profit levels
- Maximum drawdown limits

## Performance Considerations

The ML strategy is optimized for standard hardware (i5 processor, 8GB RAM):

- The Random Forest uses a limited number of estimators (100 by default)
- Feature calculation is optimized for memory efficiency
- Prediction is fast enough for daily trading decisions

## Customization Options

You can customize various aspects of the ML strategy:

### Model Parameters

In `ml_strategy.py`, you can modify:

```python
# Number of trees in the forest
n_estimators = 100

# Maximum depth of trees
max_depth = 10

# Minimum samples required to split a node
min_samples_split = 5

# Minimum samples required at a leaf node
min_samples_leaf = 2
```

### Feature Selection

You can add or remove features:

```python
def add_features(df):
    # Add your custom features here
    df['custom_feature'] = ...
    return df
```

### Prediction Threshold

Adjust the threshold for generating buy signals:

```python
# Higher threshold = more conservative (fewer trades)
# Lower threshold = more aggressive (more trades)
prediction_threshold = 0.6
```

## Advanced Customization

For advanced users, you can:

1. Replace the Random Forest with another algorithm (e.g., XGBoost, Neural Network)
2. Implement ensemble methods combining multiple models
3. Add feature selection algorithms to identify the most predictive features
4. Incorporate alternative data sources (e.g., sentiment analysis)

## Example Usage

```python
from ml_strategy import train_model, predict
import pandas as pd

# Load data
data = pd.read_csv('stock_data.csv')

# Train model
model, features = train_model(data)

# Make predictions
predictions = predict(model, data, features)

# Generate signals
signals = generate_signals(predictions, threshold=0.65)
```

For more examples, see the [Examples](EXAMPLES.md) file.