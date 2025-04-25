import yfinance as yf
import pandas as pd

# --- Settings ---
symbol = 'RELIANCE.NS' # '.NS' is needed for NSE stocks on Yahoo Finance
start_date = '2021-01-01'
end_date = '2023-12-31'
interval = '1d' # '1d' for daily data
output_filename = f'{symbol}_{interval}.csv' # Name of the file to save data
# --- End Settings ---

print(f"Attempting to download data for {symbol}...")

try:
    # Download data from Yahoo Finance
    data = yf.download(symbol, start=start_date, end=end_date, interval=interval)

    if data.empty:
        print(f"No data downloaded for {symbol}. Check the symbol or date range.")
    else:
        # Make sure column names are standard for Backtrader
        data.rename(columns={
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Adj Close': 'adjclose', # Adjusted close is often used
            'Volume': 'volume'
        }, inplace=True) # Change column names directly in the dataframe

        # Make sure the index is named 'Date' (or 'datetime') for Backtrader CSV reader
        data.index.name = 'datetime'

        # Select and order columns Backtrader CSV reader expects
        # OpenInterest is not usually available, so we can ignore it or set to 0
        required_columns = ['open', 'high', 'low', 'close', 'adjclose', 'volume']
        data_to_save = data[required_columns]

        # Save the data to a CSV file
        data_to_save.to_csv(output_filename)
        print(f"Data downloaded successfully!")
        print(f"Saved {len(data_to_save)} rows to {output_filename}")
        print("\nFirst 5 rows of data:")
        print(data_to_save.head())

except Exception as e:
    print(f"An error occurred: {e}")
