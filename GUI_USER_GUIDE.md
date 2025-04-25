# Trading Bot GUI User Guide

This guide will help you navigate and use the Trading Bot GUI effectively.

## Getting Started

1. Launch the application by running:
   ```
   python run_trading_bot.py
   ```

2. The application will open with five tabs: Data, Train Model, Backtest, Compare, and Results.

## Data Tab

This tab allows you to download or load historical price data.

### Features:

- **Data Source Selection**: Choose between Yahoo Finance, Fyers API, or a local file.
- **Stock Symbol**: Enter the stock symbol (e.g., "RELIANCE.NS" for Reliance Industries on NSE).
- **Date Range**: Specify the start and end dates for the data.
- **Download Data**: Click to download data from the selected source.
- **Browse Local File**: Select a local CSV file containing historical price data.
- **View Data**: Display the loaded data in the preview area.

### How to Use:

1. Select your preferred data source.
2. Enter the stock symbol and date range.
3. Click "Download Data" or "Browse Local File" to load data.
4. The data preview will show basic information and statistics about the loaded data.

## Train Model Tab

This tab allows you to configure and train the machine learning model.

### Features:

- **Prediction Target**: Choose how many days ahead to predict (1, 3, or 5 days).
- **Test Size**: Set the proportion of data to use for testing (e.g., 0.2 = 20%).
- **Estimators**: Set the number of trees in the Random Forest model.
- **Max Depth**: Set the maximum depth of each tree in the Random Forest.
- **Train Model**: Start the training process.

### How to Use:

1. Configure the model parameters according to your preferences.
2. Click "Train Model" to start the training process.
3. The training log will display progress and results.
4. Once training is complete, the model will be saved as "ml_model.pkl".

## Backtest Tab

This tab allows you to run backtests with different trading strategies.

### Features:

- **Strategy Selection**: Choose between Machine Learning and SMA Crossover strategies.
- **Strategy Parameters**:
  - For ML: Set the prediction threshold (probability above which to buy).
  - For SMA: Set the fast and slow moving average periods.
- **Common Parameters**: Set initial cash and commission percentage.
- **Run Backtest**: Start the backtest process.

### How to Use:

1. Select the strategy you want to test.
2. Configure the strategy-specific parameters.
3. Set the common parameters (initial cash and commission).
4. Click "Run Backtest" to start the backtest.
5. The backtest log will display progress and results.

## Compare Tab

This tab allows you to compare the performance of different strategies.

### Features:

- **Compare Strategies**: Run backtests for both ML and SMA strategies and compare results.
- **Comparison Results**: Display performance metrics and charts for both strategies.

### How to Use:

1. Click "Compare Strategies" to start the comparison.
2. The comparison log will display progress.
3. Once complete, a chart will show the performance comparison between strategies.

## Results Tab

This tab allows you to view and analyze backtest results.

### Features:

- **Results Selection**: Choose which results to view (ML, SMA, or Comparison).
- **Results Display**: Show charts and metrics for the selected results.

### How to Use:

1. Select the type of results you want to view.
2. The results will be displayed in the main area.
3. You can switch between different result types to compare them.

## Tips for Optimal Performance

1. **Data Size**: For an i5 processor with 8GB RAM, limit your data to 2-3 years for optimal performance.
2. **Model Complexity**: The default model parameters are optimized for your system. Increasing them may slow down training.
3. **Threading**: The GUI uses threading to keep the interface responsive during long operations.
4. **Memory Usage**: Close other memory-intensive applications when running backtests or training models.

## Troubleshooting

1. **GUI Not Responding**: If the GUI becomes unresponsive during a long operation, wait for the operation to complete.
2. **Data Loading Issues**: Make sure your CSV file has the correct format (datetime, open, high, low, close, adjclose, volume).
3. **Model Training Errors**: Check the training log for specific error messages.
4. **Backtest Errors**: Ensure you have trained a model before running an ML backtest.

## Next Steps

After you've become familiar with the GUI, you might want to:

1. **Customize Features**: Edit ml_strategy.py to add or modify technical indicators.
2. **Try Different Models**: Modify the code to use different machine learning algorithms.
3. **Add New Strategies**: Create new strategy classes in separate files.
4. **Optimize Parameters**: Use the comparison feature to find the best parameters for your strategies.