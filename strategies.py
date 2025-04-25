"""
Trading strategies for the Trading Bot
"""

import backtrader as bt
import numpy as np
import pandas as pd

# 1. SMA Crossover Strategy (already implemented in backtest.py)
class SmaCrossStrategy(bt.Strategy):
    params = (
        ('sma_fast_period', 10),  # Period for the fast moving average
        ('sma_slow_period', 30),  # Period for the slow moving average
        ('order_percentage', 0.95), # How much cash to use per trade
        ('ticker', 'Stock')        # Name for logging
    )

    def __init__(self):
        # Keep track of the closing price data
        self.dataclose = self.data0.close
        
        # Calculate the moving averages
        self.sma_fast = bt.indicators.SimpleMovingAverage(
            self.data0, period=self.params.sma_fast_period)
        self.sma_slow = bt.indicators.SimpleMovingAverage(
            self.data0, period=self.params.sma_slow_period)
        
        # Detect when the fast SMA crosses the slow SMA
        self.crossover = bt.indicators.CrossOver(self.sma_fast, self.sma_slow)
        
        # To keep track of pending orders
        self.order = None
        self.buyprice = None
        self.buycomm = None
        
        # Add the moving averages to the chart
        self.sma_fast.plotinfo.plotname = f'SMA {self.params.sma_fast_period}'
        self.sma_slow.plotinfo.plotname = f'SMA {self.params.sma_slow_period}'
        
        # Print strategy parameters
        print(f"--- Strategy Initialized for {self.params.ticker} ---")
        print(f"Fast SMA Period: {self.params.sma_fast_period}, Slow SMA Period: {self.params.sma_slow_period}")

    def log(self, txt, dt=None):
        """Logging function for the strategy"""
        dt = dt or self.datas[0].datetime.date(0)
        print(f'{dt.isoformat()} - {txt}')

    def notify_order(self, order):
        """Called when an order is placed or completed"""
        if order.status in [order.Submitted, order.Accepted]:
            # Order has been submitted/accepted - no action required
            return

        # Check if the order has been completed
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(f'BUY EXECUTED --- Size: {order.executed.size}, Price: {order.executed.price:.2f}, Cost: {order.executed.value:.2f}, Comm: {order.executed.comm:.2f}')
                self.buyprice = order.executed.price
                self.buycomm = order.executed.comm
            else:  # Sell
                self.log(f'SELL EXECUTED --- Size: {order.executed.size}, Price: {order.executed.price:.2f}, Value: {order.executed.value:.2f}, Comm: {order.executed.comm:.2f}')
                
                # Calculate profit
                if self.buyprice:
                    gross_profit = (order.executed.price - self.buyprice) * order.executed.size
                    net_profit = gross_profit - self.buycomm - order.executed.comm
                    self.log(f'TRADE COMPLETED --- Gross Profit: {gross_profit:.2f}, Net Profit: {net_profit:.2f}')
        
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log('Order Canceled/Margin/Rejected')
        
        # Reset the order
        self.order = None

    def next(self):
        """Called for each bar (candle) in the data"""
        # Check if we have an open order
        if self.order:
            return
        
        # Check if we are in the market (have a position)
        if not self.position:
            # We are not in the market, check if we should buy
            if self.crossover > 0:  # Fast SMA crossed above slow SMA
                self.log('BUY SIGNAL: Fast SMA crossed above Slow SMA')
                
                # Calculate the number of shares to buy
                cash_to_spend = self.broker.getcash() * self.params.order_percentage
                price = self.dataclose[0]
                size = int(cash_to_spend / price)
                
                if size > 0:
                    self.log(f'>>> Placing BUY order for {size} shares at ~{price:.2f}')
                    self.order = self.buy(size=size)
        
        else:
            # We are in the market, check if we should sell
            if self.crossover < 0:  # Fast SMA crossed below slow SMA
                self.log('SELL SIGNAL: Fast SMA crossed below Slow SMA')
                self.log(f'<<< Placing SELL order for {self.position.size} shares at ~{self.dataclose[0]:.2f}')
                self.order = self.sell()


# 2. RSI Strategy
class RSIStrategy(bt.Strategy):
    params = (
        ('rsi_period', 14),        # Period for RSI calculation
        ('rsi_overbought', 70),    # RSI level considered overbought
        ('rsi_oversold', 30),      # RSI level considered oversold
        ('order_percentage', 0.95), # How much cash to use per trade
        ('ticker', 'Stock')        # Name for logging
    )
    
    def __init__(self):
        # Keep track of the closing price data
        self.dataclose = self.data0.close
        
        # Calculate RSI
        self.rsi = bt.indicators.RelativeStrengthIndex(
            self.data0, period=self.params.rsi_period)
        
        # To keep track of pending orders
        self.order = None
        self.buyprice = None
        self.buycomm = None
        
        # Print strategy parameters
        print(f"--- RSI Strategy Initialized for {self.params.ticker} ---")
        print(f"RSI Period: {self.params.rsi_period}, Oversold: {self.params.rsi_oversold}, Overbought: {self.params.rsi_overbought}")
    
    def log(self, txt, dt=None):
        """Logging function for the strategy"""
        dt = dt or self.datas[0].datetime.date(0)
        print(f'{dt.isoformat()} - {txt}')
    
    def notify_order(self, order):
        """Called when an order is placed or completed"""
        if order.status in [order.Submitted, order.Accepted]:
            # Order has been submitted/accepted - no action required
            return

        # Check if the order has been completed
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(f'BUY EXECUTED --- Size: {order.executed.size}, Price: {order.executed.price:.2f}, Cost: {order.executed.value:.2f}, Comm: {order.executed.comm:.2f}')
                self.buyprice = order.executed.price
                self.buycomm = order.executed.comm
            else:  # Sell
                self.log(f'SELL EXECUTED --- Size: {order.executed.size}, Price: {order.executed.price:.2f}, Value: {order.executed.value:.2f}, Comm: {order.executed.comm:.2f}')
                
                # Calculate profit
                if self.buyprice:
                    gross_profit = (order.executed.price - self.buyprice) * order.executed.size
                    net_profit = gross_profit - self.buycomm - order.executed.comm
                    self.log(f'TRADE COMPLETED --- Gross Profit: {gross_profit:.2f}, Net Profit: {net_profit:.2f}')
        
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log('Order Canceled/Margin/Rejected')
        
        # Reset the order
        self.order = None
    
    def next(self):
        """Called for each bar (candle) in the data"""
        # Check if we have an open order
        if self.order:
            return
        
        # Check if we are in the market (have a position)
        if not self.position:
            # We are not in the market, check if we should buy
            if self.rsi[0] < self.params.rsi_oversold:
                self.log(f'BUY SIGNAL: RSI ({self.rsi[0]:.2f}) below oversold level ({self.params.rsi_oversold})')
                
                # Calculate the number of shares to buy
                cash_to_spend = self.broker.getcash() * self.params.order_percentage
                price = self.dataclose[0]
                size = int(cash_to_spend / price)
                
                if size > 0:
                    self.log(f'>>> Placing BUY order for {size} shares at ~{price:.2f}')
                    self.order = self.buy(size=size)
        
        else:
            # We are in the market, check if we should sell
            if self.rsi[0] > self.params.rsi_overbought:
                self.log(f'SELL SIGNAL: RSI ({self.rsi[0]:.2f}) above overbought level ({self.params.rsi_overbought})')
                self.log(f'<<< Placing SELL order for {self.position.size} shares at ~{self.dataclose[0]:.2f}')
                self.order = self.sell()


# 3. MACD Strategy
class MACDStrategy(bt.Strategy):
    params = (
        ('macd_fast', 12),         # Fast EMA period
        ('macd_slow', 26),         # Slow EMA period
        ('macd_signal', 9),        # Signal period
        ('order_percentage', 0.95), # How much cash to use per trade
        ('ticker', 'Stock')        # Name for logging
    )
    
    def __init__(self):
        # Keep track of the closing price data
        self.dataclose = self.data0.close
        
        # Calculate MACD
        self.macd = bt.indicators.MACD(
            self.data0,
            period_me1=self.params.macd_fast,
            period_me2=self.params.macd_slow,
            period_signal=self.params.macd_signal
        )
        
        # MACD crossover signal
        self.crossover = bt.indicators.CrossOver(self.macd.macd, self.macd.signal)
        
        # To keep track of pending orders
        self.order = None
        self.buyprice = None
        self.buycomm = None
        
        # Print strategy parameters
        print(f"--- MACD Strategy Initialized for {self.params.ticker} ---")
        print(f"Fast Period: {self.params.macd_fast}, Slow Period: {self.params.macd_slow}, Signal Period: {self.params.macd_signal}")
    
    def log(self, txt, dt=None):
        """Logging function for the strategy"""
        dt = dt or self.datas[0].datetime.date(0)
        print(f'{dt.isoformat()} - {txt}')
    
    def notify_order(self, order):
        """Called when an order is placed or completed"""
        if order.status in [order.Submitted, order.Accepted]:
            # Order has been submitted/accepted - no action required
            return

        # Check if the order has been completed
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(f'BUY EXECUTED --- Size: {order.executed.size}, Price: {order.executed.price:.2f}, Cost: {order.executed.value:.2f}, Comm: {order.executed.comm:.2f}')
                self.buyprice = order.executed.price
                self.buycomm = order.executed.comm
            else:  # Sell
                self.log(f'SELL EXECUTED --- Size: {order.executed.size}, Price: {order.executed.price:.2f}, Value: {order.executed.value:.2f}, Comm: {order.executed.comm:.2f}')
                
                # Calculate profit
                if self.buyprice:
                    gross_profit = (order.executed.price - self.buyprice) * order.executed.size
                    net_profit = gross_profit - self.buycomm - order.executed.comm
                    self.log(f'TRADE COMPLETED --- Gross Profit: {gross_profit:.2f}, Net Profit: {net_profit:.2f}')
        
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log('Order Canceled/Margin/Rejected')
        
        # Reset the order
        self.order = None
    
    def next(self):
        """Called for each bar (candle) in the data"""
        # Check if we have an open order
        if self.order:
            return
        
        # Check if we are in the market (have a position)
        if not self.position:
            # We are not in the market, check if we should buy
            if self.crossover > 0:  # MACD line crosses above signal line
                self.log('BUY SIGNAL: MACD crossed above Signal Line')
                
                # Calculate the number of shares to buy
                cash_to_spend = self.broker.getcash() * self.params.order_percentage
                price = self.dataclose[0]
                size = int(cash_to_spend / price)
                
                if size > 0:
                    self.log(f'>>> Placing BUY order for {size} shares at ~{price:.2f}')
                    self.order = self.buy(size=size)
        
        else:
            # We are in the market, check if we should sell
            if self.crossover < 0:  # MACD line crosses below signal line
                self.log('SELL SIGNAL: MACD crossed below Signal Line')
                self.log(f'<<< Placing SELL order for {self.position.size} shares at ~{self.dataclose[0]:.2f}')
                self.order = self.sell()


# 4. Bollinger Bands Strategy
class BollingerBandsStrategy(bt.Strategy):
    params = (
        ('bb_period', 20),         # Period for Bollinger Bands
        ('bb_devfactor', 2.0),     # Standard deviation factor
        ('order_percentage', 0.95), # How much cash to use per trade
        ('ticker', 'Stock')        # Name for logging
    )
    
    def __init__(self):
        # Keep track of the closing price data
        self.dataclose = self.data0.close
        
        # Calculate Bollinger Bands
        self.bollinger = bt.indicators.BollingerBands(
            self.data0, period=self.params.bb_period, devfactor=self.params.bb_devfactor)
        
        # To keep track of pending orders
        self.order = None
        self.buyprice = None
        self.buycomm = None
        
        # Print strategy parameters
        print(f"--- Bollinger Bands Strategy Initialized for {self.params.ticker} ---")
        print(f"Period: {self.params.bb_period}, Deviation Factor: {self.params.bb_devfactor}")
    
    def log(self, txt, dt=None):
        """Logging function for the strategy"""
        dt = dt or self.datas[0].datetime.date(0)
        print(f'{dt.isoformat()} - {txt}')
    
    def notify_order(self, order):
        """Called when an order is placed or completed"""
        if order.status in [order.Submitted, order.Accepted]:
            # Order has been submitted/accepted - no action required
            return

        # Check if the order has been completed
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(f'BUY EXECUTED --- Size: {order.executed.size}, Price: {order.executed.price:.2f}, Cost: {order.executed.value:.2f}, Comm: {order.executed.comm:.2f}')
                self.buyprice = order.executed.price
                self.buycomm = order.executed.comm
            else:  # Sell
                self.log(f'SELL EXECUTED --- Size: {order.executed.size}, Price: {order.executed.price:.2f}, Value: {order.executed.value:.2f}, Comm: {order.executed.comm:.2f}')
                
                # Calculate profit
                if self.buyprice:
                    gross_profit = (order.executed.price - self.buyprice) * order.executed.size
                    net_profit = gross_profit - self.buycomm - order.executed.comm
                    self.log(f'TRADE COMPLETED --- Gross Profit: {gross_profit:.2f}, Net Profit: {net_profit:.2f}')
        
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log('Order Canceled/Margin/Rejected')
        
        # Reset the order
        self.order = None
    
    def next(self):
        """Called for each bar (candle) in the data"""
        # Check if we have an open order
        if self.order:
            return
        
        # Check if we are in the market (have a position)
        if not self.position:
            # We are not in the market, check if we should buy
            if self.dataclose[0] < self.bollinger.lines.bot[0]:  # Price below lower band
                self.log('BUY SIGNAL: Price below Lower Bollinger Band')
                
                # Calculate the number of shares to buy
                cash_to_spend = self.broker.getcash() * self.params.order_percentage
                price = self.dataclose[0]
                size = int(cash_to_spend / price)
                
                if size > 0:
                    self.log(f'>>> Placing BUY order for {size} shares at ~{price:.2f}')
                    self.order = self.buy(size=size)
        
        else:
            # We are in the market, check if we should sell
            if self.dataclose[0] > self.bollinger.lines.top[0]:  # Price above upper band
                self.log('SELL SIGNAL: Price above Upper Bollinger Band')
                self.log(f'<<< Placing SELL order for {self.position.size} shares at ~{self.dataclose[0]:.2f}')
                self.order = self.sell()


# 5. Dual Moving Average Strategy
class DualMAStrategy(bt.Strategy):
    params = (
        ('ma1_period', 10),        # Period for first moving average
        ('ma2_period', 30),        # Period for second moving average
        ('ma1_type', 'SMA'),       # Type of first MA: 'SMA' or 'EMA'
        ('ma2_type', 'EMA'),       # Type of second MA: 'SMA' or 'EMA'
        ('order_percentage', 0.95), # How much cash to use per trade
        ('ticker', 'Stock')        # Name for logging
    )
    
    def __init__(self):
        # Keep track of the closing price data
        self.dataclose = self.data0.close
        
        # Calculate the moving averages
        if self.params.ma1_type == 'SMA':
            self.ma1 = bt.indicators.SimpleMovingAverage(
                self.data0, period=self.params.ma1_period)
        else:
            self.ma1 = bt.indicators.ExponentialMovingAverage(
                self.data0, period=self.params.ma1_period)
            
        if self.params.ma2_type == 'SMA':
            self.ma2 = bt.indicators.SimpleMovingAverage(
                self.data0, period=self.params.ma2_period)
        else:
            self.ma2 = bt.indicators.ExponentialMovingAverage(
                self.data0, period=self.params.ma2_period)
        
        # Detect when MA1 crosses MA2
        self.crossover = bt.indicators.CrossOver(self.ma1, self.ma2)
        
        # To keep track of pending orders
        self.order = None
        self.buyprice = None
        self.buycomm = None
        
        # Print strategy parameters
        print(f"--- Dual MA Strategy Initialized for {self.params.ticker} ---")
        print(f"MA1: {self.params.ma1_type} Period {self.params.ma1_period}, MA2: {self.params.ma2_type} Period {self.params.ma2_period}")
    
    def log(self, txt, dt=None):
        """Logging function for the strategy"""
        dt = dt or self.datas[0].datetime.date(0)
        print(f'{dt.isoformat()} - {txt}')
    
    def notify_order(self, order):
        """Called when an order is placed or completed"""
        if order.status in [order.Submitted, order.Accepted]:
            # Order has been submitted/accepted - no action required
            return

        # Check if the order has been completed
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(f'BUY EXECUTED --- Size: {order.executed.size}, Price: {order.executed.price:.2f}, Cost: {order.executed.value:.2f}, Comm: {order.executed.comm:.2f}')
                self.buyprice = order.executed.price
                self.buycomm = order.executed.comm
            else:  # Sell
                self.log(f'SELL EXECUTED --- Size: {order.executed.size}, Price: {order.executed.price:.2f}, Value: {order.executed.value:.2f}, Comm: {order.executed.comm:.2f}')
                
                # Calculate profit
                if self.buyprice:
                    gross_profit = (order.executed.price - self.buyprice) * order.executed.size
                    net_profit = gross_profit - self.buycomm - order.executed.comm
                    self.log(f'TRADE COMPLETED --- Gross Profit: {gross_profit:.2f}, Net Profit: {net_profit:.2f}')
        
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log('Order Canceled/Margin/Rejected')
        
        # Reset the order
        self.order = None
    
    def next(self):
        """Called for each bar (candle) in the data"""
        # Check if we have an open order
        if self.order:
            return
        
        # Check if we are in the market (have a position)
        if not self.position:
            # We are not in the market, check if we should buy
            if self.crossover > 0:  # MA1 crossed above MA2
                self.log(f'BUY SIGNAL: {self.params.ma1_type}{self.params.ma1_period} crossed above {self.params.ma2_type}{self.params.ma2_period}')
                
                # Calculate the number of shares to buy
                cash_to_spend = self.broker.getcash() * self.params.order_percentage
                price = self.dataclose[0]
                size = int(cash_to_spend / price)
                
                if size > 0:
                    self.log(f'>>> Placing BUY order for {size} shares at ~{price:.2f}')
                    self.order = self.buy(size=size)
        
        else:
            # We are in the market, check if we should sell
            if self.crossover < 0:  # MA1 crossed below MA2
                self.log(f'SELL SIGNAL: {self.params.ma1_type}{self.params.ma1_period} crossed below {self.params.ma2_type}{self.params.ma2_period}')
                self.log(f'<<< Placing SELL order for {self.position.size} shares at ~{self.dataclose[0]:.2f}')
                self.order = self.sell()


# Dictionary of available strategies
STRATEGIES = {
    'SMA Crossover': SmaCrossStrategy,
    'RSI': RSIStrategy,
    'MACD': MACDStrategy,
    'Bollinger Bands': BollingerBandsStrategy,
    'Dual MA': DualMAStrategy
}

# Function to get available strategies
def get_strategy_list():
    """Returns a list of available strategy names"""
    return list(STRATEGIES.keys())

def get_strategy_class(name):
    """Returns the strategy class for a given name"""
    return STRATEGIES.get(name)