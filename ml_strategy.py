import backtrader as bt
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import datetime
import pickle
import os

# --- Feature Engineering Functions ---
def add_features(df):
    """Add technical indicators and features to the dataframe"""
    # Make a copy of the dataframe to avoid modifying the original
    df_features = df.copy()
    
    # Price-based features
    df_features['returns'] = df_features['close'].pct_change()
    df_features['log_returns'] = np.log(df_features['close'] / df_features['close'].shift(1))
    
    # Moving averages
    for window in [5, 10, 20, 50]:
        df_features[f'sma_{window}'] = df_features['close'].rolling(window=window).mean()
        df_features[f'ema_{window}'] = df_features['close'].ewm(span=window, adjust=False).mean()
    
    # Volatility
    for window in [5, 10, 20]:
        df_features[f'volatility_{window}'] = df_features['returns'].rolling(window=window).std()
    
    # Price relative to moving averages
    for window in [5, 10, 20, 50]:
        df_features[f'close_to_sma_{window}'] = df_features['close'] / df_features[f'sma_{window}']
    
    # Volume features
    df_features['volume_change'] = df_features['volume'].pct_change()
    df_features['volume_ma_5'] = df_features['volume'].rolling(window=5).mean()
    df_features['volume_ma_10'] = df_features['volume'].rolling(window=10).mean()
    df_features['volume_ratio'] = df_features['volume'] / df_features['volume_ma_5']
    
    # Momentum indicators
    # RSI (Relative Strength Index)
    delta = df_features['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df_features['rsi_14'] = 100 - (100 / (1 + rs))
    
    # MACD (Moving Average Convergence Divergence)
    ema_12 = df_features['close'].ewm(span=12, adjust=False).mean()
    ema_26 = df_features['close'].ewm(span=26, adjust=False).mean()
    df_features['macd'] = ema_12 - ema_26
    df_features['macd_signal'] = df_features['macd'].ewm(span=9, adjust=False).mean()
    df_features['macd_hist'] = df_features['macd'] - df_features['macd_signal']
    
    # Bollinger Bands
    window = 20
    df_features['bb_middle'] = df_features['close'].rolling(window=window).mean()
    df_features['bb_std'] = df_features['close'].rolling(window=window).std()
    df_features['bb_upper'] = df_features['bb_middle'] + 2 * df_features['bb_std']
    df_features['bb_lower'] = df_features['bb_middle'] - 2 * df_features['bb_std']
    df_features['bb_width'] = (df_features['bb_upper'] - df_features['bb_lower']) / df_features['bb_middle']
    df_features['bb_position'] = (df_features['close'] - df_features['bb_lower']) / (df_features['bb_upper'] - df_features['bb_lower'])
    
    # Target variable: 1 if price goes up in next n days, 0 otherwise
    for days in [1, 3, 5]:
        df_features[f'target_{days}d'] = (df_features['close'].shift(-days) > df_features['close']).astype(int)
    
    # Drop NaN values
    df_features.dropna(inplace=True)
    
    return df_features

def prepare_data(df_features, target_days=1, test_size=0.2):
    """Prepare data for machine learning"""
    # Define features and target
    target_col = f'target_{target_days}d'
    
    # Drop columns that shouldn't be used as features
    drop_cols = ['open', 'high', 'low', 'close', 'adjclose', 'volume']
    drop_cols.extend([f'target_{d}d' for d in [1, 3, 5] if d != target_days])
    
    # Create feature matrix and target vector
    X = df_features.drop(drop_cols + [target_col], axis=1)
    y = df_features[target_col]
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=False)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, X.columns.tolist()

def train_model(X_train, y_train, n_estimators=100, max_depth=10, n_jobs=None, callback=None):
    """
    Train a machine learning model with resource constraints
    
    Args:
        X_train: Training features
        y_train: Training labels
        n_estimators: Number of trees in the forest
        max_depth: Maximum depth of trees
        n_jobs: Number of CPU cores to use (None=1 core, -1=all cores)
        callback: Optional callback function to report progress
        
    Returns:
        Trained model
    """
    import psutil
    import time
    
    # Get system info
    cpu_count = psutil.cpu_count(logical=True)
    total_memory = psutil.virtual_memory().total / (1024 * 1024 * 1024)  # GB
    
    # Determine resource limits based on system
    if n_jobs is None:
        # Auto-determine CPU usage: use 50% of available cores but at least 1
        n_jobs = max(1, int(cpu_count * 0.5))
    
    # Adjust n_estimators based on available memory
    # Each tree takes roughly ~memory_per_tree MB
    memory_per_tree = 50  # Rough estimate in MB
    max_trees_for_memory = int((total_memory * 1024 * 0.5) / memory_per_tree)  # Use max 50% of RAM
    
    # Cap n_estimators to avoid memory issues
    n_estimators = min(n_estimators, max_trees_for_memory)
    
    print(f"Training with {n_estimators} trees using {n_jobs} CPU cores")
    print(f"System has {cpu_count} cores and {total_memory:.1f} GB RAM")
    
    # Using RandomForest with resource constraints
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=n_jobs,  # Controlled CPU usage
        verbose=0
    )
    
    # Train the model with progress reporting
    if callback:
        # For progress reporting, we'll use warm_start and incrementally add trees
        model.n_estimators = 0
        model.warm_start = True
        
        batch_size = max(1, n_estimators // 10)  # Report progress in 10% increments
        for i in range(0, n_estimators, batch_size):
            # Check memory usage
            current_memory_usage = psutil.virtual_memory().percent
            if current_memory_usage > 85:  # If memory usage exceeds 85%
                print(f"Warning: High memory usage ({current_memory_usage}%). Reducing trees.")
                break
                
            # Add more trees
            model.n_estimators = min(model.n_estimators + batch_size, n_estimators)
            model.fit(X_train, y_train)
            
            # Report progress
            progress = min(100, int((model.n_estimators / n_estimators) * 100))
            callback(progress)
            
            # Small pause to allow other processes to run
            time.sleep(0.1)
    else:
        # Standard training without progress reporting
        model.fit(X_train, y_train)
    
    return model

def save_model(model, scaler, feature_names, file_path='ml_model.pkl'):
    """Save the trained model, scaler, and feature names to a file"""
    model_data = {
        'model': model,
        'scaler': scaler,
        'feature_names': feature_names
    }
    with open(file_path, 'wb') as f:
        pickle.dump(model_data, f)
    print(f"Model savenbd to {file_path}")

def load_model(file_path='ml_model.pkl'):
    """Load a trained model, scaler, and feature names from a file"""
    with open(file_path, 'rb') as f:
        model_data = pickle.load(f)
    return model_data['model'], model_data['scaler'], model_data['feature_names']

# --- Machine Learning Strategy ---
class MLStrategy(bt.Strategy):
    params = (
        ('model_path', 'ml_model.pkl'),  # Path to the saved model
        ('prediction_threshold', 0.6),    # Threshold for buy signals
        ('order_percentage', 0.95),       # How much cash to use per trade
        ('ticker', 'Stock')               # Name for logging
    )
    
    def __init__(self):
        # Keep track of the closing price data
        self.dataclose = self.data0.close
        
        # Load the trained model
        try:
            self.model, self.scaler, self.feature_names = load_model(self.params.model_path)
            print(f"Model loaded successfully from {self.params.model_path}")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
        
        # Initialize indicators for features
        # Moving averages
        self.sma5 = bt.indicators.SimpleMovingAverage(self.data0.close, period=5)
        self.sma10 = bt.indicators.SimpleMovingAverage(self.data0.close, period=10)
        self.sma20 = bt.indicators.SimpleMovingAverage(self.data0.close, period=20)
        self.sma50 = bt.indicators.SimpleMovingAverage(self.data0.close, period=50)
        
        self.ema5 = bt.indicators.ExponentialMovingAverage(self.data0.close, period=5)
        self.ema10 = bt.indicators.ExponentialMovingAverage(self.data0.close, period=10)
        self.ema20 = bt.indicators.ExponentialMovingAverage(self.data0.close, period=20)
        self.ema50 = bt.indicators.ExponentialMovingAverage(self.data0.close, period=50)
        
        # RSI
        self.rsi = bt.indicators.RSI(self.data0.close, period=14)
        
        # MACD
        self.macd = bt.indicators.MACD(
            self.data0.close, 
            period_me1=12, 
            period_me2=26, 
            period_signal=9
        )
        
        # Bollinger Bands
        self.bollinger = bt.indicators.BollingerBands(self.data0.close, period=20)
        
        # To keep track of pending orders
        self.order = None
        self.last_prediction = None
        
        print(f"--- ML Strategy Initialized for {self.params.ticker} ---")
        print(f"Prediction Threshold: {self.params.prediction_threshold}")
    
    def log(self, txt, dt=None):
        # Helper function to print messages with the date
        dt = dt or self.data0.datetime.date(0)
        print(f'{dt.isoformat()} - {txt}')
    
    def notify_order(self, order):
        # This function is called when an order status changes
        if order.status in [order.Submitted, order.Accepted]:
            # Order is submitted/accepted - no action needed
            return
        
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(f'BUY EXECUTED --- Size: {order.executed.size}, Price: {order.executed.price:.2f}, Cost: {order.executed.value:.2f}, Comm: {order.executed.comm:.2f}')
            elif order.issell():
                self.log(f'SELL EXECUTED --- Size: {order.executed.size}, Price: {order.executed.price:.2f}, Value: {order.executed.value:.2f}, Comm: {order.executed.comm:.2f}')
            # Keep track of when the order was filled
            self.bar_executed = len(self)
        
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log(f'ORDER FAILED/REJECTED: {order.Status[order.status]}')
        
        # Reset order tracker regardless of status after notification
        self.order = None
    
    def notify_trade(self, trade):
        # This function is called when a trade (buy + sell) is completed
        if not trade.isclosed:
            return # Do nothing if the trade isn't finished
        self.log(f'TRADE COMPLETED --- Gross Profit: {trade.pnl:.2f}, Net Profit: {trade.pnlcomm:.2f}')
    
    def get_features(self):
        """Extract features for the current bar"""
        # Calculate returns
        returns = 0.0
        log_returns = 0.0
        if len(self) > 1:  # Make sure we have at least 2 bars
            prev_close = self.dataclose[-1]
            if prev_close > 0:
                returns = (self.dataclose[0] / prev_close) - 1
                log_returns = np.log(self.dataclose[0] / prev_close)
        
        # Calculate volatility
        volatility_5 = volatility_10 = volatility_20 = 0.0
        if len(self) >= 5:
            prices = np.array([self.dataclose[-i] for i in range(5)])
            returns_array = np.diff(prices) / prices[:-1]
            volatility_5 = np.std(returns_array)
        
        if len(self) >= 10:
            prices = np.array([self.dataclose[-i] for i in range(10)])
            returns_array = np.diff(prices) / prices[:-1]
            volatility_10 = np.std(returns_array)
        
        if len(self) >= 20:
            prices = np.array([self.dataclose[-i] for i in range(20)])
            returns_array = np.diff(prices) / prices[:-1]
            volatility_20 = np.std(returns_array)
        
        # Price relative to moving averages
        close_to_sma_5 = self.dataclose[0] / self.sma5[0] if self.sma5[0] > 0 else 1.0
        close_to_sma_10 = self.dataclose[0] / self.sma10[0] if self.sma10[0] > 0 else 1.0
        close_to_sma_20 = self.dataclose[0] / self.sma20[0] if self.sma20[0] > 0 else 1.0
        close_to_sma_50 = self.dataclose[0] / self.sma50[0] if self.sma50[0] > 0 else 1.0
        
        # Volume features
        volume_change = 0.0
        volume_ratio = 1.0
        if len(self) > 1 and self.data0.volume[-1] > 0:
            volume_change = (self.data0.volume[0] / self.data0.volume[-1]) - 1
        
        volume_ma_5 = volume_ma_10 = 0.0
        if len(self) >= 5:
            volume_ma_5 = sum(self.data0.volume[-i] for i in range(5)) / 5
            if volume_ma_5 > 0:
                volume_ratio = self.data0.volume[0] / volume_ma_5
        
        if len(self) >= 10:
            volume_ma_10 = sum(self.data0.volume[-i] for i in range(10)) / 10
        
        # Bollinger Band features
        bb_width = 0.0
        bb_position = 0.5
        if self.bollinger.top[0] > self.bollinger.bot[0]:
            bb_width = (self.bollinger.top[0] - self.bollinger.bot[0]) / self.bollinger.mid[0] if self.bollinger.mid[0] > 0 else 0.0
            bb_range = self.bollinger.top[0] - self.bollinger.bot[0]
            if bb_range > 0:
                bb_position = (self.dataclose[0] - self.bollinger.bot[0]) / bb_range
        
        # MACD features
        macd = self.macd.macd[0]
        macd_signal = self.macd.signal[0]
        macd_hist = macd - macd_signal
        
        # Create feature dictionary
        features = {
            'returns': returns,
            'log_returns': log_returns,
            'sma_5': self.sma5[0],
            'sma_10': self.sma10[0],
            'sma_20': self.sma20[0],
            'sma_50': self.sma50[0],
            'ema_5': self.ema5[0],
            'ema_10': self.ema10[0],
            'ema_20': self.ema20[0],
            'ema_50': self.ema50[0],
            'volatility_5': volatility_5,
            'volatility_10': volatility_10,
            'volatility_20': volatility_20,
            'close_to_sma_5': close_to_sma_5,
            'close_to_sma_10': close_to_sma_10,
            'close_to_sma_20': close_to_sma_20,
            'close_to_sma_50': close_to_sma_50,
            'volume_change': volume_change,
            'volume_ma_5': volume_ma_5,
            'volume_ma_10': volume_ma_10,
            'volume_ratio': volume_ratio,
            'rsi_14': self.rsi[0],
            'macd': macd,
            'macd_signal': macd_signal,
            'macd_hist': macd_hist,
            'bb_middle': self.bollinger.mid[0],
            'bb_upper': self.bollinger.top[0],
            'bb_lower': self.bollinger.bot[0],
            'bb_width': bb_width,
            'bb_position': bb_position
        }
        
        # Create a DataFrame with the features
        df = pd.DataFrame([features])
        
        # Ensure all required features are present
        for feature in self.feature_names:
            if feature not in df.columns:
                df[feature] = 0.0
        
        # Select only the features used by the model
        df = df[self.feature_names]
        
        return df
    
    def next(self):
        # This function is called for every day (or bar) in the data
        
        # If an order is already pending, don't do anything
        if self.order:
            return
        
        # Wait until we have enough bars for all indicators
        if len(self) < 50:  # Need at least 50 bars for the 50-period SMA
            return
        
        # Get features for the current bar
        features_df = self.get_features()
        
        # Scale features
        features_scaled = self.scaler.transform(features_df)
        
        # Make prediction
        prediction_proba = self.model.predict_proba(features_scaled)[0][1]  # Probability of class 1 (price up)
        self.last_prediction = prediction_proba
        
        # Check if we already have shares (a position)
        if not self.position:
            # Not in the market - check for BUY signal
            if prediction_proba > self.params.prediction_threshold:
                self.log(f'BUY SIGNAL: ML prediction = {prediction_proba:.4f}')
                # Calculate how many shares to buy (using % of cash)
                cash_available = self.broker.get_cash()
                # Use the current close price for calculation
                size_to_buy = int((cash_available * self.params.order_percentage) / self.dataclose[0])
                
                if size_to_buy > 0:
                    self.log(f'>>> Placing BUY order for {size_to_buy} shares at ~{self.dataclose[0]:.2f}')
                    # Place the buy order
                    self.order = self.buy(size=size_to_buy)
                else:
                    self.log(f'Cannot buy: Not enough cash ({cash_available:.2f}) for even 1 share at {self.dataclose[0]:.2f}')
        
        else: # Already in the market
            # Check for SELL signal
            if prediction_proba < (1 - self.params.prediction_threshold):
                self.log(f'SELL SIGNAL: ML prediction = {prediction_proba:.4f}')
                self.log(f'<<< Placing SELL order for {self.position.size} shares at ~{self.dataclose[0]:.2f}')
                # Place the sell order for all shares we hold
                self.order = self.sell(size=self.position.size)


# --- Main Function to Train Model ---
def train_ml_model(data_file='NSE_RELIANCE-EQ_D_Full.csv', target_days=1, test_size=0.2, 
                save_path='ml_model.pkl', n_estimators=100, max_depth=10, n_jobs=None, 
                progress_callback=None):
    """
    Train a machine learning model on the data and save it with resource constraints
    
    Args:
        data_file: Path to the CSV data file
        target_days: Number of days ahead to predict
        test_size: Proportion of data to use for testing
        save_path: Where to save the trained model
        n_estimators: Number of trees in the forest (will be adjusted based on available memory)
        max_depth: Maximum depth of trees
        n_jobs: Number of CPU cores to use (None=auto, -1=all cores)
        progress_callback: Optional callback function to report progress
    """
    print(f"\nLoading data from: {data_file}")
    try:
        # Load data
        df = pd.read_csv(data_file, parse_dates=['datetime'], index_col='datetime')
        print(f"Data loaded successfully. Shape: {df.shape}")
        
        if progress_callback:
            progress_callback(10)  # 10% progress after loading data
        
        # Add features
        print("\nAdding technical indicators and features...")
        df_features = add_features(df)
        print(f"Features added. New shape: {df_features.shape}")
        
        if progress_callback:
            progress_callback(20)  # 20% progress after adding features
        
        # Prepare data for machine learning
        print(f"\nPreparing data for machine learning (target: price up in {target_days} days)...")
        X_train, X_test, y_train, y_test, scaler, feature_names = prepare_data(df_features, target_days, test_size)
        print(f"Data prepared. Training set shape: {X_train.shape}, Test set shape: {X_test.shape}")
        
        if progress_callback:
            progress_callback(30)  # 30% progress after preparing data
        
        # Define a nested progress callback to map model training progress (0-100) to overall progress (30-90)
        def model_progress_callback(model_progress):
            if progress_callback:
                # Map model_progress (0-100) to overall progress (30-90)
                overall_progress = 30 + int(model_progress * 0.6)
                progress_callback(overall_progress)
        
        # Train model with resource constraints
        print("\nTraining machine learning model with resource management...")
        model = train_model(
            X_train, 
            y_train, 
            n_estimators=n_estimators,
            max_depth=max_depth,
            n_jobs=n_jobs,
            callback=model_progress_callback
        )
        
        # Evaluate model
        train_accuracy = model.score(X_train, y_train)
        test_accuracy = model.score(X_test, y_test)
        print(f"Model trained. Training accuracy: {train_accuracy:.4f}, Test accuracy: {test_accuracy:.4f}")
        
        if progress_callback:
            progress_callback(90)  # 90% progress after training
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        print("\nTop 10 most important features:")
        print(feature_importance.head(10))
        
        # Save model
        print(f"\nSaving model to {save_path}...")
        model_data = {
            'model': model,
            'scaler': scaler,
            'feature_names': feature_names
        }
        with open(save_path, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"Model saved successfully to {save_path}")
        
        if progress_callback:
            progress_callback(100)  # 100% progress after saving
        
        return model, scaler, feature_names
    
    except FileNotFoundError:
        print(f"Error: Data file not found at '{data_file}'.")
        return None, None, None
    except Exception as e:
        print(f"Error during model training: {e}")
        return None, None, None


# --- Main Script Execution ---
if __name__ == "__main__":
    # Train the model
    model, scaler, feature_names = train_ml_model(
        data_file='NSE_RELIANCE-EQ_D_Full.csv',
        target_days=1,  # Predict price movement 1 day ahead
        test_size=0.2,  # Use 20% of data for testing
        save_path='ml_model.pkl'
    )
    
    if model is not None:
        print("\nModel training completed successfully.")
        print("You can now run the backtest with the ML strategy.")
    else:
        print("\nModel training failed. Please check the errors above.")
