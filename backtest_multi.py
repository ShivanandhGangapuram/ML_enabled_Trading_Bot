"""
Enhanced backtesting script that supports multiple strategies and stocks
"""

import backtrader as bt
import pandas as pd
import datetime
import argparse
import os
import sys
from strategies import STRATEGIES, get_strategy_list, get_strategy_class
from stock_list import get_stock_list, get_yahoo_symbols, get_stock_names, get_name_to_yahoo_mapping

# --- Main Script Execution ---
if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Backtest trading strategies')
    parser.add_argument('--strategy', type=str, default='SMA Crossover',
                        help=f'Strategy to use. Available: {", ".join(get_strategy_list())}')
    parser.add_argument('--stock', type=str, default='Reliance Industries',
                        help='Stock to backtest on')
    parser.add_argument('--cash', type=float, default=100000.0,
                        help='Starting cash amount')
    parser.add_argument('--commission', type=float, default=0.1,
                        help='Commission percentage (e.g., 0.1 for 0.1%)')
    parser.add_argument('--data_file', type=str, default=None,
                        help='Data file to use (if not specified, will look for stock-specific file)')
    parser.add_argument('--list_stocks', action='store_true',
                        help='List available stocks and exit')
    parser.add_argument('--list_strategies', action='store_true',
                        help='List available strategies and exit')
    
    args = parser.parse_args()
    
    # List stocks if requested
    if args.list_stocks:
        print("Available stocks:")
        for name in get_stock_names():
            print(f"  - {name}")
        sys.exit(0)
    
    # List strategies if requested
    if args.list_strategies:
        print("Available strategies:")
        for strat in get_strategy_list():
            print(f"  - {strat}")
        sys.exit(0)
    
    # Get the strategy class
    strategy_class = get_strategy_class(args.strategy)
    if not strategy_class:
        print(f"Strategy '{args.strategy}' not found. Available strategies:")
        for strat in get_strategy_list():
            print(f"  - {strat}")
        sys.exit(1)
    
    # Determine the data file to use
    data_file = args.data_file
    stock_symbol = None
    
    if not data_file:
        # Try to find a data file for the specified stock
        stock_names = get_stock_names()
        name_to_yahoo = get_name_to_yahoo_mapping()
        
        if args.stock in stock_names:
            # Get the Yahoo symbol for the stock
            stock_symbol = name_to_yahoo[args.stock]
            # Construct possible data file names
            possible_files = [
                f"{stock_symbol.replace('.', '_')}_1d.csv",
                f"{stock_symbol.replace('.', '_')}_D_Full.csv",
                f"{stock_symbol.replace(':', '_').replace('.', '_')}_D_Full.csv"
            ]
            
            # Try each possible file
            for file in possible_files:
                if os.path.exists(file):
                    data_file = file
                    break
            
            # If no file found, use default
            if not data_file:
                data_file = 'NSE_RELIANCE-EQ_D_Full.csv'
                print(f"Data file for {args.stock} not found, using default: {data_file}")
        else:
            data_file = 'NSE_RELIANCE-EQ_D_Full.csv'
            print(f"Stock '{args.stock}' not found in stock list, using default data: {data_file}")
    
    print(f"\nLoading data from: {data_file}")
    
    # Create a Cerebro engine instance
    cerebro = bt.Cerebro()
    
    # Load the data
    try:
        # Try to load as a pandas DataFrame first
        try:
            df = pd.read_csv(data_file, index_col='datetime', parse_dates=True)
            data = bt.feeds.PandasData(dataname=df)
            cerebro.adddata(data)
            print("Data loaded successfully using PandasData.")
        except Exception as e:
            # If that fails, try GenericCSVData
            print(f"PandasData loading failed, trying GenericCSVData: {e}")
            data = bt.feeds.GenericCSVData(
                dataname=data_file,
                fromdate=datetime.datetime(2021, 1, 1),
                todate=datetime.datetime(2023, 12, 31),
                dtformat=('%Y-%m-%d'),
                datetime=0,
                open=1,
                high=2,
                low=3,
                close=4,
                adjclose=5,
                volume=6,
                openinterest=-1
            )
            cerebro.adddata(data)
            print("Data loaded successfully using GenericCSVData.")
    except Exception as e:
        print(f"Error loading data: {e}")
        sys.exit(1)
    
    # Add the strategy
    print(f"\nAdding strategy: {args.strategy}")
    cerebro.addstrategy(strategy_class, ticker=args.stock)
    
    # Set the starting cash
    starting_cash = args.cash
    cerebro.broker.setcash(starting_cash)
    print(f"Starting portfolio value: {starting_cash:.2f}")
    
    # Set the commission
    commission_rate = args.commission / 100.0  # Convert from percentage to decimal
    cerebro.broker.setcommission(commission=commission_rate)
    print(f"Commission per trade set to: {args.commission:.2f}%")
    
    # Add analyzers
    print("\nAdding analyzers...")
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe', riskfreerate=0.01)
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
    cerebro.addanalyzer(bt.analyzers.SQN, _name='sqn')
    
    # Run the backtest
    print("\n--- Starting Backtest Run ---")
    results = cerebro.run()
    
    # Get the first strategy from the results
    strat = results[0]
    
    # Print final portfolio value
    final_value = cerebro.broker.getvalue()
    profit_loss = final_value - starting_cash
    print(f"\nFinal Portfolio Value: {final_value:.2f}")
    print(f"Net Profit/Loss: {profit_loss:.2f}")
    
    # Print performance metrics
    print("\n--- Performance Analysis ---")
    try:
        # Trade analysis
        trade_analysis = strat.analyzers.trades.get_analysis()
        
        # Get total trades
        total_trades = trade_analysis.total.closed
        print(f"Total Trades: {total_trades}")
        
        # Get winning and losing trades
        winning_trades = trade_analysis.won.total if hasattr(trade_analysis, 'won') else 0
        losing_trades = trade_analysis.lost.total if hasattr(trade_analysis, 'lost') else 0
        print(f"Winning Trades: {winning_trades}")
        print(f"Losing Trades: {losing_trades}")
        
        # Calculate average profit/loss
        if winning_trades > 0:
            avg_win = trade_analysis.won.pnl.average if hasattr(trade_analysis, 'won') else 0
            print(f"Average Winning Trade Profit: {avg_win:.2f}")
        
        if losing_trades > 0:
            avg_loss = trade_analysis.lost.pnl.average if hasattr(trade_analysis, 'lost') else 0
            print(f"Average Losing Trade Loss: {avg_loss:.2f}")
        
        # Calculate total net profit/loss from trades
        total_net_profit = trade_analysis.pnl.net if hasattr(trade_analysis, 'pnl') else 0
        print(f"Total Net Profit/Loss (from trades): {total_net_profit:.2f}")
        
        # Sharpe ratio
        sharpe_ratio = strat.analyzers.sharpe.get_analysis()['sharperatio']
        print(f"Annualized Sharpe Ratio: {sharpe_ratio}")
        
        # Drawdown
        drawdown_analysis = strat.analyzers.drawdown.get_analysis()
        print(f"Maximum Drawdown: {drawdown_analysis.max.drawdown:.2f}%")
        
        # Returns
        returns_analysis = strat.analyzers.returns.get_analysis()
        print(f"Total Compound Return: {returns_analysis.get('rtot', 0.0)*100:.2f}%")
        print(f"Average Annual Return: {returns_analysis.get('ravg', 0.0)*100:.2f}%")
        
        # SQN
        sqn_analysis = strat.analyzers.sqn.get_analysis()
        print(f"System Quality Number (SQN): {sqn_analysis.get('sqn', 'N/A') if sqn_analysis else 'N/A'}")
    
    except Exception as e:
        print(f"An error occurred during analysis printing: {e}")
    
    # Plot results
    print("\nAttempting to plot results...")
    try:
        # Set a non-interactive backend for matplotlib
        import matplotlib
        matplotlib.use('Agg')
        
        # Generate a unique filename based on strategy and stock
        plot_filename = f"backtest_{args.strategy.replace(' ', '_')}_{args.stock.replace(' ', '_')}.png"
        
        figure = cerebro.plot(style='candlestick', barup='green', bardown='red')[0][0]
        figure.savefig(plot_filename)
        print(f"Plot saved to {plot_filename}")
    except ImportError:
        print("\nPlotting failed: Matplotlib not found. Install it using 'pip install matplotlib'")
    except Exception as e:
        print(f"\nPlotting failed. Error: {e}")
        print("Plotting requires a graphical environment or the 'Agg' backend.")
        print("Check if 'matplotlib' is installed correctly.")
        print(f"If the plot was saved as '{plot_filename}', you can open that file manually.")