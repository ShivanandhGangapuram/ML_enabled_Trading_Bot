"""
Live Paper Trading Module for Trading Bot
Connects to Fyers API for real-time data and executes paper trades
"""

import time
import datetime
import pandas as pd
import numpy as np
import threading
import queue
import logging
import os
import pickle
from fyers_apiv3 import fyersModel
from strategies import STRATEGIES, get_strategy_class
import backtrader as bt

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("paper_trading.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("PaperTrading")

class PaperTradingEngine:
    """Paper trading engine that simulates real trading without actual money"""
    
    def __init__(self, fyers_client, symbol, strategy_name, strategy_params=None, 
                 initial_capital=100000.0, commission=0.001, mode="auto"):
        """
        Initialize the paper trading engine
        
        Args:
            fyers_client: Authenticated Fyers API client
            symbol: Trading symbol (e.g., "NSE:RELIANCE-EQ")
            strategy_name: Name of the strategy to use
            strategy_params: Parameters for the strategy (dict)
            initial_capital: Starting capital for paper trading
            commission: Commission rate (e.g., 0.001 = 0.1%)
            mode: "auto" for machine-driven or "user" for user-guided
        """
        self.fyers = fyers_client
        self.symbol = symbol
        self.strategy_name = strategy_name
        self.strategy_params = strategy_params or {}
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.commission = commission
        self.mode = mode
        
        # Trading state
        self.position = 0  # Number of shares held
        self.entry_price = 0.0  # Average entry price
        self.last_price = 0.0  # Last known price
        self.trades = []  # List of completed trades
        self.pending_orders = []  # List of pending orders
        
        # Data storage
        self.historical_data = pd.DataFrame()  # Recent historical data for analysis
        self.live_data_queue = queue.Queue()  # Queue for incoming live data
        
        # Strategy instance
        self.strategy_class = get_strategy_class(strategy_name)
        if not self.strategy_class:
            raise ValueError(f"Strategy '{strategy_name}' not found")
        
        # Threads
        self.data_thread = None
        self.strategy_thread = None
        self.is_running = False
        
        logger.info(f"Paper Trading Engine initialized for {symbol}")
        logger.info(f"Strategy: {strategy_name}, Mode: {mode}")
        logger.info(f"Initial Capital: {initial_capital}, Commission: {commission*100}%")
    
    def fetch_historical_data(self, days_back=30, resolution="D"):
        """Fetch historical data for initial strategy setup"""
        end_date = datetime.datetime.now().date()
        start_date = end_date - datetime.timedelta(days=days_back)
        
        logger.info(f"Fetching {days_back} days of historical data for {self.symbol}")
        
        data_request = {
            "symbol": self.symbol,
            "resolution": resolution,
            "date_format": "1",  # Epoch format
            "range_from": start_date.strftime("%Y-%m-%d"),
            "range_to": end_date.strftime("%Y-%m-%d"),
            "cont_flag": "1"
        }
        
        try:
            history_response = self.fyers.history(data=data_request)
            
            if history_response.get("s") == "ok" and history_response.get("candles"):
                df = pd.DataFrame(
                    history_response["candles"],
                    columns=['datetime', 'open', 'high', 'low', 'close', 'volume']
                )
                
                # Convert timestamp to datetime
                df['datetime'] = pd.to_datetime(df['datetime'], unit='s')
                df.set_index('datetime', inplace=True)
                
                self.historical_data = df
                logger.info(f"Successfully loaded {len(df)} historical data points")
                return True
            else:
                logger.error(f"Failed to fetch historical data: {history_response}")
                return False
                
        except Exception as e:
            logger.error(f"Error fetching historical data: {e}")
            return False
    
    def start_data_stream(self):
        """Start streaming market data"""
        def data_worker():
            logger.info(f"Starting data stream for {self.symbol}")
            
            while self.is_running:
                try:
                    # In a real implementation, this would use Fyers websocket API
                    # For paper trading simulation, we'll fetch the latest quote every few seconds
                    quote_request = {"symbols": self.symbol}
                    quote_response = self.fyers.quotes(quote_request)
                    
                    if quote_response.get("s") == "ok" and quote_response.get("d"):
                        quote_data = quote_response["d"][0]
                        
                        # Create a data point and put it in the queue
                        data_point = {
                            'datetime': datetime.datetime.now(),
                            'symbol': self.symbol,
                            'ltp': quote_data.get('ltp', 0),  # Last traded price
                            'open': quote_data.get('open_price', 0),
                            'high': quote_data.get('high_price', 0),
                            'low': quote_data.get('low_price', 0),
                            'close': quote_data.get('ltp', 0),  # Use LTP as current close
                            'volume': quote_data.get('tot_qty', 0)
                        }
                        
                        self.last_price = data_point['ltp']
                        self.live_data_queue.put(data_point)
                        
                    else:
                        logger.warning(f"Failed to get quote: {quote_response}")
                    
                    # Sleep to avoid hitting API rate limits
                    # In production with websockets, this would be event-driven
                    time.sleep(5)  # Poll every 5 seconds
                    
                except Exception as e:
                    logger.error(f"Error in data stream: {e}")
                    time.sleep(10)  # Longer sleep on error
        
        self.data_thread = threading.Thread(target=data_worker)
        self.data_thread.daemon = True
        self.data_thread.start()
    
    def start_strategy_execution(self):
        """Start the strategy execution loop"""
        def strategy_worker():
            logger.info(f"Starting strategy execution for {self.strategy_name}")
            
            # Initialize strategy with historical data
            self.initialize_strategy()
            
            while self.is_running:
                try:
                    # Get the latest data point from the queue
                    if not self.live_data_queue.empty():
                        data_point = self.live_data_queue.get(timeout=1)
                        
                        # Update the last price
                        self.last_price = data_point['ltp']
                        
                        # Process pending orders first
                        self.process_pending_orders()
                        
                        # Execute strategy based on mode
                        if self.mode == "auto":
                            # Automatic mode - let the strategy decide
                            self.execute_strategy_auto(data_point)
                        else:
                            # User mode - wait for user input (handled elsewhere)
                            pass
                        
                        # Calculate current portfolio value
                        self.update_portfolio_value()
                        
                    else:
                        time.sleep(0.1)  # Small sleep if no data
                        
                except queue.Empty:
                    pass  # No data in queue, continue
                except Exception as e:
                    logger.error(f"Error in strategy execution: {e}")
                    time.sleep(5)  # Sleep on error
        
        self.strategy_thread = threading.Thread(target=strategy_worker)
        self.strategy_thread.daemon = True
        self.strategy_thread.start()
    
    def initialize_strategy(self):
        """Initialize the strategy with historical data"""
        # This would typically create a Backtrader Cerebro instance
        # and feed it the historical data to initialize indicators
        # For simplicity, we'll just log that it's initialized
        logger.info(f"Strategy {self.strategy_name} initialized with parameters: {self.strategy_params}")
    
    def execute_strategy_auto(self, data_point):
        """Execute the strategy automatically based on the latest data"""
        # In a real implementation, this would use the strategy's logic
        # to decide whether to buy, sell, or hold
        
        # For demonstration, we'll implement a simple moving average crossover
        # This should be replaced with actual strategy logic from the strategy class
        
        # Add the new data point to our historical data
        new_row = pd.DataFrame([{
            'open': data_point['open'],
            'high': data_point['high'],
            'low': data_point['low'],
            'close': data_point['close'],
            'volume': data_point['volume']
        }], index=[data_point['datetime']])
        
        self.historical_data = pd.concat([self.historical_data, new_row])
        
        # Simple example: If we have enough data, calculate SMAs and check for crossover
        if len(self.historical_data) >= 30:  # Need at least 30 data points
            # Get fast and slow SMA periods from strategy params or use defaults
            fast_period = self.strategy_params.get('sma_fast_period', 10)
            slow_period = self.strategy_params.get('sma_slow_period', 30)
            
            # Calculate SMAs
            fast_sma = self.historical_data['close'].rolling(window=fast_period).mean()
            slow_sma = self.historical_data['close'].rolling(window=slow_period).mean()
            
            # Check for crossover
            if len(fast_sma) >= 2 and len(slow_sma) >= 2:
                # Check if fast SMA crossed above slow SMA
                if fast_sma.iloc[-2] < slow_sma.iloc[-2] and fast_sma.iloc[-1] > slow_sma.iloc[-1]:
                    # Buy signal
                    if self.position == 0:  # Only buy if we don't have a position
                        self.place_buy_order()
                
                # Check if fast SMA crossed below slow SMA
                elif fast_sma.iloc[-2] > slow_sma.iloc[-2] and fast_sma.iloc[-1] < slow_sma.iloc[-1]:
                    # Sell signal
                    if self.position > 0:  # Only sell if we have a position
                        self.place_sell_order()
    
    def execute_user_strategy(self, action, quantity=None):
        """Execute a user-directed trade"""
        if action.lower() == "buy":
            self.place_buy_order(quantity)
        elif action.lower() == "sell":
            self.place_sell_order(quantity)
        else:
            logger.warning(f"Unknown action: {action}")
    
    def place_buy_order(self, quantity=None):
        """Place a buy order"""
        if self.last_price <= 0:
            logger.warning("Cannot place buy order: Invalid price")
            return False
        
        # Calculate quantity based on available capital if not specified
        if quantity is None:
            # Use 95% of available capital by default
            capital_to_use = self.current_capital * 0.95
            quantity = int(capital_to_use / self.last_price)
        
        if quantity <= 0:
            logger.warning(f"Cannot place buy order: Invalid quantity {quantity}")
            return False
        
        # Check if we have enough capital
        cost = quantity * self.last_price
        commission_cost = cost * self.commission
        total_cost = cost + commission_cost
        
        if total_cost > self.current_capital:
            logger.warning(f"Cannot place buy order: Insufficient capital (need {total_cost}, have {self.current_capital})")
            return False
        
        # Create the order
        order = {
            'id': f"order_{int(time.time())}",
            'type': 'buy',
            'symbol': self.symbol,
            'quantity': quantity,
            'price': self.last_price,
            'timestamp': datetime.datetime.now(),
            'status': 'pending'
        }
        
        logger.info(f"Placing BUY order for {quantity} shares of {self.symbol} at {self.last_price}")
        self.pending_orders.append(order)
        return True
    
    def place_sell_order(self, quantity=None):
        """Place a sell order"""
        if self.position <= 0:
            logger.warning("Cannot place sell order: No position to sell")
            return False
        
        # If quantity not specified, sell entire position
        if quantity is None or quantity > self.position:
            quantity = self.position
        
        if quantity <= 0:
            logger.warning(f"Cannot place sell order: Invalid quantity {quantity}")
            return False
        
        # Create the order
        order = {
            'id': f"order_{int(time.time())}",
            'type': 'sell',
            'symbol': self.symbol,
            'quantity': quantity,
            'price': self.last_price,
            'timestamp': datetime.datetime.now(),
            'status': 'pending'
        }
        
        logger.info(f"Placing SELL order for {quantity} shares of {self.symbol} at {self.last_price}")
        self.pending_orders.append(order)
        return True
    
    def process_pending_orders(self):
        """Process any pending orders"""
        if not self.pending_orders:
            return
        
        # In a real system, we'd check order status with the broker
        # For paper trading, we'll just execute all pending orders at the current price
        for order in self.pending_orders[:]:  # Create a copy to safely remove during iteration
            if order['type'] == 'buy':
                # Execute buy order
                cost = order['quantity'] * self.last_price
                commission_cost = cost * self.commission
                total_cost = cost + commission_cost
                
                if total_cost <= self.current_capital:
                    # Update position
                    if self.position == 0:
                        self.entry_price = self.last_price
                    else:
                        # Calculate new average entry price
                        total_value = (self.position * self.entry_price) + (order['quantity'] * self.last_price)
                        self.entry_price = total_value / (self.position + order['quantity'])
                    
                    self.position += order['quantity']
                    self.current_capital -= total_cost
                    
                    # Record the trade
                    trade = {
                        'id': order['id'],
                        'type': 'buy',
                        'symbol': self.symbol,
                        'quantity': order['quantity'],
                        'price': self.last_price,
                        'commission': commission_cost,
                        'timestamp': datetime.datetime.now()
                    }
                    self.trades.append(trade)
                    
                    logger.info(f"BUY EXECUTED: {order['quantity']} shares at {self.last_price}, " +
                               f"Commission: {commission_cost:.2f}, Total Cost: {total_cost:.2f}")
                else:
                    logger.warning(f"Buy order failed: Insufficient capital")
            
            elif order['type'] == 'sell':
                # Execute sell order
                if order['quantity'] <= self.position:
                    value = order['quantity'] * self.last_price
                    commission_cost = value * self.commission
                    net_value = value - commission_cost
                    
                    # Calculate profit/loss
                    cost_basis = order['quantity'] * self.entry_price
                    profit_loss = value - cost_basis - commission_cost
                    
                    # Update position and capital
                    self.position -= order['quantity']
                    self.current_capital += net_value
                    
                    # If position is now zero, reset entry price
                    if self.position == 0:
                        self.entry_price = 0
                    
                    # Record the trade
                    trade = {
                        'id': order['id'],
                        'type': 'sell',
                        'symbol': self.symbol,
                        'quantity': order['quantity'],
                        'price': self.last_price,
                        'commission': commission_cost,
                        'profit_loss': profit_loss,
                        'timestamp': datetime.datetime.now()
                    }
                    self.trades.append(trade)
                    
                    logger.info(f"SELL EXECUTED: {order['quantity']} shares at {self.last_price}, " +
                               f"Commission: {commission_cost:.2f}, P/L: {profit_loss:.2f}")
                else:
                    logger.warning(f"Sell order failed: Insufficient position")
            
            # Remove the processed order
            self.pending_orders.remove(order)
    
    def update_portfolio_value(self):
        """Update the current portfolio value"""
        # Current capital + value of holdings
        position_value = self.position * self.last_price if self.last_price > 0 else 0
        portfolio_value = self.current_capital + position_value
        
        # Calculate unrealized P/L if we have a position
        if self.position > 0 and self.entry_price > 0:
            unrealized_pl = (self.last_price - self.entry_price) * self.position
            unrealized_pl_pct = (self.last_price / self.entry_price - 1) * 100
            
            logger.debug(f"Portfolio: ${portfolio_value:.2f}, " +
                       f"Position: {self.position} shares at avg ${self.entry_price:.2f}, " +
                       f"Unrealized P/L: ${unrealized_pl:.2f} ({unrealized_pl_pct:.2f}%)")
        else:
            logger.debug(f"Portfolio: ${portfolio_value:.2f}, Cash: ${self.current_capital:.2f}")
        
        return portfolio_value
    
    def get_portfolio_summary(self):
        """Get a summary of the current portfolio"""
        position_value = self.position * self.last_price if self.last_price > 0 else 0
        portfolio_value = self.current_capital + position_value
        
        # Calculate performance metrics
        initial_value = self.initial_capital
        absolute_return = portfolio_value - initial_value
        percent_return = (portfolio_value / initial_value - 1) * 100
        
        # Calculate unrealized P/L
        if self.position > 0 and self.entry_price > 0:
            unrealized_pl = (self.last_price - self.entry_price) * self.position
            unrealized_pl_pct = (self.last_price / self.entry_price - 1) * 100
        else:
            unrealized_pl = 0
            unrealized_pl_pct = 0
        
        # Count trades
        buy_trades = sum(1 for trade in self.trades if trade['type'] == 'buy')
        sell_trades = sum(1 for trade in self.trades if trade['type'] == 'sell')
        
        # Calculate realized P/L
        realized_pl = sum(trade.get('profit_loss', 0) for trade in self.trades if 'profit_loss' in trade)
        
        return {
            'timestamp': datetime.datetime.now(),
            'symbol': self.symbol,
            'strategy': self.strategy_name,
            'mode': self.mode,
            'current_price': self.last_price,
            'position': self.position,
            'entry_price': self.entry_price,
            'position_value': position_value,
            'cash': self.current_capital,
            'portfolio_value': portfolio_value,
            'initial_capital': initial_value,
            'absolute_return': absolute_return,
            'percent_return': percent_return,
            'unrealized_pl': unrealized_pl,
            'unrealized_pl_pct': unrealized_pl_pct,
            'realized_pl': realized_pl,
            'buy_trades': buy_trades,
            'sell_trades': sell_trades,
            'total_trades': buy_trades + sell_trades
        }
    
    def start(self):
        """Start the paper trading engine"""
        if self.is_running:
            logger.warning("Paper trading engine is already running")
            return False
        
        # Fetch initial historical data
        if not self.fetch_historical_data():
            logger.error("Failed to fetch historical data, cannot start")
            return False
        
        # Set running flag
        self.is_running = True
        
        # Start data and strategy threads
        self.start_data_stream()
        self.start_strategy_execution()
        
        logger.info("Paper trading engine started")
        return True
    
    def stop(self):
        """Stop the paper trading engine"""
        if not self.is_running:
            logger.warning("Paper trading engine is not running")
            return
        
        # Set flag to stop threads
        self.is_running = False
        
        # Wait for threads to finish
        if self.data_thread and self.data_thread.is_alive():
            self.data_thread.join(timeout=5)
        
        if self.strategy_thread and self.strategy_thread.is_alive():
            self.strategy_thread.join(timeout=5)
        
        logger.info("Paper trading engine stopped")
        
        # Print final summary
        summary = self.get_portfolio_summary()
        logger.info(f"Final Portfolio Value: ${summary['portfolio_value']:.2f}")
        logger.info(f"Return: ${summary['absolute_return']:.2f} ({summary['percent_return']:.2f}%)")
        logger.info(f"Total Trades: {summary['total_trades']}")
    
    def save_state(self, filename="paper_trading_state.pkl"):
        """Save the current state of the paper trading engine"""
        state = {
            'symbol': self.symbol,
            'strategy_name': self.strategy_name,
            'strategy_params': self.strategy_params,
            'mode': self.mode,
            'initial_capital': self.initial_capital,
            'current_capital': self.current_capital,
            'position': self.position,
            'entry_price': self.entry_price,
            'trades': self.trades,
            'last_update': datetime.datetime.now()
        }
        
        with open(filename, 'wb') as f:
            pickle.dump(state, f)
        
        logger.info(f"Paper trading state saved to {filename}")
    
    def load_state(self, filename="paper_trading_state.pkl"):
        """Load a previously saved state"""
        if not os.path.exists(filename):
            logger.warning(f"State file {filename} not found")
            return False
        
        try:
            with open(filename, 'rb') as f:
                state = pickle.load(f)
            
            # Restore state
            self.symbol = state['symbol']
            self.strategy_name = state['strategy_name']
            self.strategy_params = state['strategy_params']
            self.mode = state['mode']
            self.initial_capital = state['initial_capital']
            self.current_capital = state['current_capital']
            self.position = state['position']
            self.entry_price = state['entry_price']
            self.trades = state['trades']
            
            logger.info(f"Paper trading state loaded from {filename}")
            logger.info(f"Loaded state from {state.get('last_update')}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading state: {e}")
            return False


# Helper function to authenticate with Fyers API
def authenticate_fyers(client_id, secret_key, redirect_uri):
    """Authenticate with Fyers API and return a client instance"""
    from get_fyers_data_improved import get_access_token
    
    # Get access token
    access_token = get_access_token(client_id, secret_key, redirect_uri)
    
    if not access_token:
        logger.error("Failed to get Fyers access token")
        return None
    
    # Initialize Fyers client
    fyers_client = fyersModel.FyersModel(
        client_id=client_id,
        token=access_token,
        log_path=""
    )
    
    logger.info("Fyers API client initialized successfully")
    return fyers_client


# Example usage
if __name__ == "__main__":
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run paper trading with Fyers API')
    parser.add_argument('--symbol', type=str, default="NSE:RELIANCE-EQ", 
                        help='Stock symbol (e.g., NSE:RELIANCE-EQ)')
    parser.add_argument('--strategy', type=str, default="SMA Crossover", 
                        help='Strategy name')
    parser.add_argument('--capital', type=float, default=100000.0, 
                        help='Initial capital')
    parser.add_argument('--mode', type=str, choices=['auto', 'user'], default="auto", 
                        help='Trading mode: auto (machine) or user (manual)')
    parser.add_argument('--client_id', type=str, default="NSRJ65D4YS-100", 
                        help='Fyers API client ID')
    parser.add_argument('--secret_key', type=str, default="NMLEEFZHP0", 
                        help='Fyers API secret key')
    parser.add_argument('--redirect_uri', type=str, default="http://127.0.0.1/", 
                        help='Fyers API redirect URI')
    
    args = parser.parse_args()
    
    # Authenticate with Fyers API
    fyers_client = authenticate_fyers(args.client_id, args.secret_key, args.redirect_uri)
    
    if not fyers_client:
        print("Failed to authenticate with Fyers API. Exiting.")
        exit(1)
    
    # Create and start paper trading engine
    engine = PaperTradingEngine(
        fyers_client=fyers_client,
        symbol=args.symbol,
        strategy_name=args.strategy,
        initial_capital=args.capital,
        mode=args.mode
    )
    
    try:
        # Start the engine
        if engine.start():
            print(f"Paper trading started for {args.symbol} using {args.strategy} strategy")
            print("Press Ctrl+C to stop...")
            
            # Keep the main thread alive
            while True:
                time.sleep(1)
                
                # Every 60 seconds, print a status update
                if int(time.time()) % 60 == 0:
                    summary = engine.get_portfolio_summary()
                    print(f"\nPortfolio Value: ${summary['portfolio_value']:.2f}")
                    print(f"Position: {summary['position']} shares at avg ${summary['entry_price']:.2f}")
                    print(f"Current Price: ${summary['current_price']:.2f}")
                    print(f"Unrealized P/L: ${summary['unrealized_pl']:.2f} ({summary['unrealized_pl_pct']:.2f}%)")
                    print(f"Realized P/L: ${summary['realized_pl']:.2f}")
                    print(f"Total Return: {summary['percent_return']:.2f}%")
                
    except KeyboardInterrupt:
        print("\nStopping paper trading...")
    finally:
        # Stop the engine and save state
        engine.stop()
        engine.save_state()
        print("Paper trading stopped and state saved.")