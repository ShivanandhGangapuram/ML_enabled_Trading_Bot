import backtrader as bt
import pandas as pd
import datetime
import argparse
import os
from strategies import STRATEGIES, get_strategy_list, get_strategy_class
from stock_list import get_stock_list, get_yahoo_symbols, get_stock_names

# Import the strategies (now defined in strategies.py)
# Using the SmaCrossStrategy as default for backward compatibility
from strategies import SmaCrossStrategy

    def __init__(self):
        # Keep track of the closing price data
        # Use self.data0 for the primary data feed
        self.dataclose = self.data0.close

        # Calculate the moving averages using the primary data feed
        self.sma_fast = bt.indicators.SimpleMovingAverage(
            self.data0, period=self.params.sma_fast_period)
        self.sma_slow = bt.indicators.SimpleMovingAverage(
            self.data0, period=self.params.sma_slow_period)

        # Detect when the fast SMA crosses the slow SMA
        self.crossover = bt.indicators.CrossOver(self.sma_fast, self.sma_slow)

        # To keep track of pending orders
        self.order = None
        print(f"--- Strategy Initialized for {self.params.ticker} ---")
        print(f"Fast SMA Period: {self.params.sma_fast_period}, Slow SMA Period: {self.params.sma_slow_period}")


    def log(self, txt, dt=None):
        # Helper function to print messages with the date
        # Use self.data0.datetime to access the datetime object of the current bar
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
            self.bar_executed = len(self) # len(self) gives the current bar number

        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            # Use order.Status dictionary to get the status string
            self.log(f'ORDER FAILED/REJECTED: {order.Status[order.status]}')

        # Reset order tracker regardless of status after notification
        self.order = None

    def notify_trade(self, trade):
         # This function is called when a trade (buy + sell) is completed
         if not trade.isclosed:
             return # Do nothing if the trade isn't finished
         self.log(f'TRADE COMPLETED --- Gross Profit: {trade.pnl:.2f}, Net Profit: {trade.pnlcomm:.2f}')

    def next(self):
        # This function is called for every day (or bar) in the data
        # self.log(f'Close Price: {self.dataclose[0]:.2f}') # Uncomment to see price each day

        # If an order is already pending, don't do anything
        if self.order:
            return

        # Check if we already have shares (a position)
        # Use self.position attribute which Backtrader manages
        if not self.position:
            # Not in the market - check for BUY signal
            if self.crossover > 0: # If fast SMA crosses ABOVE slow SMA
                self.log('BUY SIGNAL: Fast SMA crossed above Slow SMA')
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

            else:
                 # Optional: Add logging for no buy signal if desired
                 # self.log('No Buy Signal')
                 pass

        else: # Already in the market
            # Check for SELL signal
            if self.crossover < 0: # If fast SMA crosses BELOW slow SMA
                self.log('SELL SIGNAL: Fast SMA crossed below Slow SMA')
                self.log(f'<<< Placing SELL order for {self.position.size} shares at ~{self.dataclose[0]:.2f}')
                # Place the sell order for all shares we hold (self.position.size)
                self.order = self.sell(size=self.position.size)
            else:
                 # Optional: Add logging for no sell signal if desired
                 # self.log('No Sell Signal, Holding Position')
                 pass

# --- Backtesting Setup ---
if __name__ == '__main__':
    # Create a Cerebro engine instance (the brain)
    cerebro = bt.Cerebro()

    # --- Add Data Feed ---
    # MODIFIED: Point to the full data file downloaded from Fyers
    data_file = 'NSE_RELIANCE-EQ_D_Full.csv'
    print(f"\nLoading data from: {data_file}")
    try:
        # MODIFIED: Use correct parameters for Fyers CSV format
        data = bt.feeds.GenericCSVData(
            dataname=data_file,
            fromdate=datetime.datetime(2021, 1, 1), # Start date for backtest
            todate=datetime.datetime(2023, 12, 31), # End date for backtest
            dtformat=('%Y-%m-%d'), # Fyers CSV includes time HH:MM:SS
            datetime=0, # Column index for 'datetime'
            open=1,     # Column index for 'open'
            high=2,     # Column index for 'high'
            low=3,      # Column index for 'low'
            close=4,    # Column index for 'close'
            adjclose=5, # Column index for 'adjclose'
            volume=6,   # Column index for 'volume'
            openinterest=-1 # No open interest column
        )
        cerebro.adddata(data)
        print("Data loaded successfully.")
    except FileNotFoundError:
        print(f"Error: Data file not found at '{data_file}'.")
        print("Please make sure 'get_fyers_data.py' ran successfully and created the file.")
        exit() # Stop if data file doesn't exist
    except Exception as e:
        print(f"Error loading data file '{data_file}': {e}")
        exit() # Stop if data loading fails for other reasons

    # --- Add Strategy ---
    print("\nAdding strategy...")
    # MODIFIED: Update ticker name for clarity if desired
    cerebro.addstrategy(SmaCrossStrategy, ticker='NSE:RELIANCE-EQ')

    # --- Configure Broker ---
    start_cash = 100000.0 # Starting virtual money
    cerebro.broker.set_cash(start_cash)
    print(f"Starting portfolio value: {start_cash:.2f}")

    # Set commission (example: 0.1% per trade)
    commission_fee = 0.001 # 0.001 = 0.1%
    cerebro.broker.setcommission(commission=commission_fee)
    print(f"Commission per trade set to: {commission_fee*100:.2f}%")

    # --- Add Analyzers (Optional) ---
    # These calculate performance metrics
    print("\nAdding analyzers...")
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trade_analyzer')
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe_ratio', timeframe=bt.TimeFrame.Days, compression=1, factor=252, annualize=True) # Annualized Sharpe
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    cerebro.addanalyzer(bt.analyzers.Returns, _name='returns', timeframe=bt.TimeFrame.Days, compression=1) # Daily returns
    cerebro.addanalyzer(bt.analyzers.SQN, _name='sqn') # System Quality Number

    # --- Run the Backtest ---
    print("\n--- Starting Backtest Run ---")
    results = cerebro.run() # This runs the simulation
    print("--- Backtest Run Finished ---")

    # --- Print Results ---
    final_portfolio_value = cerebro.broker.getvalue()
    print(f'\nFinal Portfolio Value: {final_portfolio_value:.2f}')
    pnl = final_portfolio_value - start_cash
    print(f'Net Profit/Loss: {pnl:.2f}')

    # Print analyzer results
    print("\n--- Performance Analysis ---")
    try:
        strat = results[0] # Get the strategy instance from the results
        trade_analysis = strat.analyzers.trade_analyzer.get_analysis()
        if trade_analysis and hasattr(trade_analysis, 'total') and trade_analysis.total.total > 0: # Check if any trades happened
            print(f"Total Trades: {trade_analysis.total.total}")
            print(f"Winning Trades: {trade_analysis.won.total}")
            print(f"Losing Trades: {trade_analysis.lost.total}")
            if trade_analysis.won.total > 0:
                 print(f"Average Winning Trade Profit: {trade_analysis.won.pnl.average:.2f}")
            if trade_analysis.lost.total > 0:
                 print(f"Average Losing Trade Loss: {trade_analysis.lost.pnl.average:.2f}")
            print(f"Total Net Profit/Loss (from trades): {trade_analysis.pnl.net.total:.2f}")
        else:
             print("No trades were executed.")

        sharpe_analysis = strat.analyzers.sharpe_ratio.get_analysis()
        # Check if sharpe ratio exists before printing
        print(f"Annualized Sharpe Ratio: {sharpe_analysis.get('sharperatio', 'N/A') if sharpe_analysis else 'N/A'}")

        drawdown_analysis = strat.analyzers.drawdown.get_analysis()
        print(f"Maximum Drawdown: {drawdown_analysis.max.drawdown:.2f}%") # Max % loss from a peak

        returns_analysis = strat.analyzers.returns.get_analysis()
        print(f"Total Compound Return: {returns_analysis.get('rtot', 0.0)*100:.2f}%") # Total return over period
        print(f"Average Annual Return: {returns_analysis.get('ravg', 0.0)*100:.2f}%")

        sqn_analysis = strat.analyzers.sqn.get_analysis()
        print(f"System Quality Number (SQN): {sqn_analysis.get('sqn', 'N/A') if sqn_analysis else 'N/A'}")


    except Exception as e:
        print(f"An error occurred during analysis printing: {e}")

    # --- Plot Results (Optional) ---
    print("\nAttempting to plot results...")
    try:
        # Set a non-interactive backend for matplotlib if plotting window doesn't appear
        import matplotlib
        matplotlib.use('Agg') # 'Agg' saves plot to file, doesn't show window (useful for non-GUI)
                              # Comment out this line if you want to try showing a popup window

        figure = cerebro.plot(style='candlestick', barup='green', bardown='red')[0][0]
        plot_filename = 'backtest_results.png'
        figure.savefig(plot_filename)
        print(f"Plot saved to {plot_filename}")
        # If you didn't use 'Agg', you might uncomment the next line instead of saving:
        # cerebro.plot(style='candlestick', barup='green', bardown='red')
    except ImportError:
        print("\nPlotting failed: Matplotlib not found. Install it using 'pip install matplotlib'")
    except Exception as e:
        print(f"\nPlotting failed. Error: {e}")
        print("Plotting requires a graphical environment or the 'Agg' backend.")
        print("Check if 'matplotlib' is installed correctly.")
        print(f"If the plot was saved as '{plot_filename}', you can open that file manually.")