"""
List of popular Indian stocks for trading with their symbols for Yahoo Finance and Fyers API
"""

STOCK_LIST = [
    # Format: (Name, Yahoo Finance Symbol, Fyers Symbol)
    ("Reliance Industries", "RELIANCE.NS", "NSE:RELIANCE-EQ"),
    ("Tata Consultancy Services", "TCS.NS", "NSE:TCS-EQ"),
    ("HDFC Bank", "HDFCBANK.NS", "NSE:HDFCBANK-EQ"),
    ("Infosys", "INFY.NS", "NSE:INFY-EQ"),
    ("ICICI Bank", "ICICIBANK.NS", "NSE:ICICIBANK-EQ"),
    ("Hindustan Unilever", "HINDUNILVR.NS", "NSE:HINDUNILVR-EQ"),
    ("ITC", "ITC.NS", "NSE:ITC-EQ"),
    ("State Bank of India", "SBIN.NS", "NSE:SBIN-EQ"),
    ("Bharti Airtel", "BHARTIARTL.NS", "NSE:BHARTIARTL-EQ"),
    ("Larsen & Toubro", "LT.NS", "NSE:LT-EQ"),
    ("Kotak Mahindra Bank", "KOTAKBANK.NS", "NSE:KOTAKBANK-EQ"),
    ("Axis Bank", "AXISBANK.NS", "NSE:AXISBANK-EQ"),
    ("Maruti Suzuki", "MARUTI.NS", "NSE:MARUTI-EQ"),
    ("Asian Paints", "ASIANPAINT.NS", "NSE:ASIANPAINT-EQ"),
    ("HCL Technologies", "HCLTECH.NS", "NSE:HCLTECH-EQ"),
    ("Bajaj Finance", "BAJFINANCE.NS", "NSE:BAJFINANCE-EQ"),
    ("Titan Company", "TITAN.NS", "NSE:TITAN-EQ"),
    ("Tata Motors", "TATAMOTORS.NS", "NSE:TATAMOTORS-EQ"),
    ("Adani Ports", "ADANIPORTS.NS", "NSE:ADANIPORTS-EQ"),
    ("Wipro", "WIPRO.NS", "NSE:WIPRO-EQ"),
    ("Sun Pharmaceutical", "SUNPHARMA.NS", "NSE:SUNPHARMA-EQ"),
    ("Mahindra & Mahindra", "M&M.NS", "NSE:M&M-EQ"),
    ("Power Grid Corporation", "POWERGRID.NS", "NSE:POWERGRID-EQ"),
    ("NTPC", "NTPC.NS", "NSE:NTPC-EQ"),
    ("UltraTech Cement", "ULTRACEMCO.NS", "NSE:ULTRACEMCO-EQ"),
    ("Nestle India", "NESTLEIND.NS", "NSE:NESTLEIND-EQ"),
    ("Tech Mahindra", "TECHM.NS", "NSE:TECHM-EQ"),
    ("Tata Steel", "TATASTEEL.NS", "NSE:TATASTEEL-EQ"),
    ("Bajaj Auto", "BAJAJ-AUTO.NS", "NSE:BAJAJ-AUTO-EQ"),
    ("IndusInd Bank", "INDUSINDBK.NS", "NSE:INDUSINDBK-EQ"),
]

# Nifty 50 Index
NIFTY50 = "^NSEI"

# Bank Nifty Index
BANKNIFTY = "^NSEBANK"

# Function to get stock lists
def get_stock_list():
    """Returns the list of stocks with their names and symbols"""
    return STOCK_LIST

def get_yahoo_symbols():
    """Returns just the Yahoo Finance symbols"""
    return [stock[1] for stock in STOCK_LIST]

def get_fyers_symbols():
    """Returns just the Fyers API symbols"""
    return [stock[2] for stock in STOCK_LIST]

def get_stock_names():
    """Returns just the stock names"""
    return [stock[0] for stock in STOCK_LIST]

def get_symbol_mappings():
    """Returns a dictionary mapping Yahoo symbols to Fyers symbols"""
    return {stock[1]: stock[2] for stock in STOCK_LIST}

def get_name_to_yahoo_mapping():
    """Returns a dictionary mapping stock names to Yahoo symbols"""
    return {stock[0]: stock[1] for stock in STOCK_LIST}

def get_name_to_fyers_mapping():
    """Returns a dictionary mapping stock names to Fyers symbols"""
    return {stock[0]: stock[2] for stock in STOCK_LIST}