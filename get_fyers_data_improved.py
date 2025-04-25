from fyers_apiv3 import fyersModel
import webbrowser
import pandas as pd
from datetime import datetime, date
from dateutil.relativedelta import relativedelta
import time
import argparse
import os
import sys

# Import configuration
try:
    from config import FYERS_CLIENT_ID, FYERS_SECRET_KEY, FYERS_REDIRECT_URL, FYERS_USERNAME
    CLIENT_ID = FYERS_CLIENT_ID
    SECRET_KEY = FYERS_SECRET_KEY
    REDIRECT_URL = FYERS_REDIRECT_URL
    USERNAME = FYERS_USERNAME
except ImportError:
    print("Error: config.py file not found or missing required variables.")
    print("Please copy config_template.py to config.py and fill in your API credentials.")
    sys.exit(1)

# --- Function to Authenticate and Get Token ---
def get_access_token(client_id, secret_key, redirect_uri):
    """Get access token from Fyers API with improved error handling"""
    session = fyersModel.SessionModel(
        client_id=client_id,
        secret_key=secret_key,
        redirect_uri=redirect_uri,
        response_type="code",
        grant_type="authorization_code"
    )
    
    # Generate the authorization URL
    try:
        auth_url = session.generate_authcode()
    except Exception as e:
        print(f"Error generating auth URL: {e}")
        return None
    
    print("\n" + "="*50)
    print("FYERS API AUTHENTICATION")
    print("="*50)
    print("1. A Fyers login page will open in your web browser.")
    print("2. Log in with your Fyers credentials.")
    print("3. After login, you'll be redirected to a URL like:")
    print(f"   '{redirect_uri}?auth_code=XXXXXXXXXXXXX&state=None'")
    print("4. The page may show 'This site can't be reached' - THIS IS NORMAL.")
    print("5. Copy the ENTIRE URL from your browser's address bar.")
    print("6. Paste it below when prompted.")
    print("="*50)
    
    print("\nOpening Fyers login page...")
    print(f"If browser doesn't open automatically, copy and paste this URL in your browser:\n{auth_url}\n")

    # Try to open the browser
    try:
        webbrowser.open(auth_url, new=1)
    except Exception as e:
        print(f"Could not open browser automatically: {e}")
        print("Please manually copy the URL above and open it in your browser.")
    
    # Get the redirected URL from user
    max_attempts = 3
    for attempt in range(1, max_attempts + 1):
        print(f"\nAttempt {attempt}/{max_attempts}")
        redirected_url = input("Paste the FULL URL from your browser after logging in: ").strip()
        
        if not redirected_url:
            print("URL cannot be empty. Please try again.")
            continue
            
        # Extract the auth code
        try:
            if "auth_code=" in redirected_url:
                auth_code = redirected_url.split("auth_code=")[1].split("&")[0]
                print("\nAuth Code extracted successfully.")
            else:
                print("\nError: Could not find 'auth_code=' in the URL you pasted.")
                print("Make sure you copied the entire URL from your browser's address bar.")
                continue
                
            # Set the token and generate access token
            session.set_token(auth_code)
            response = session.generate_token()
            
            if response.get("access_token"):
                access_token = response["access_token"]
                print("Access Token generated successfully!")
                return access_token
            else:
                print(f"Error generating Access Token: {response}")
                
                # Check for specific error codes
                if response.get("code") == -437:  # invalid auth code
                    print("The authorization code is invalid or has expired.")
                    print("Please try the authentication process again.")
                elif response.get("code") == -435:  # invalid redirect URI
                    print("The redirect URI doesn't match what's configured in your Fyers API app.")
                    print(f"Make sure your app is configured with exactly: {redirect_uri}")
                
                # If it's the last attempt, return None
                if attempt == max_attempts:
                    return None
                
                # Otherwise, try again
                print("Let's try again...")
                
        except Exception as e:
            print(f"\nError processing the URL: {e}")
            if attempt == max_attempts:
                return None
            print("Let's try again...")
    
    return None  # If we get here, all attempts failed

# --- Function to Fetch Historical Data in Chunks ---
def fetch_historical_data_chunked(fyers_obj, symbol, resolution, date_format, start_date_str, end_date_str, chunk_years=1):
    all_candles_df = pd.DataFrame() # Initialize an empty DataFrame to store all data
    current_start_date = datetime.strptime(start_date_str, "%Y-%m-%d").date()
    final_end_date = datetime.strptime(end_date_str, "%Y-%m-%d").date()

    print(f"\nStarting chunked download for {symbol} from {start_date_str} to {end_date_str}...")

    while current_start_date <= final_end_date:
        # Calculate the end date for the current chunk (start_date + chunk_years - 1 day)
        current_end_date = current_start_date + relativedelta(years=chunk_years) - relativedelta(days=1)

        # Make sure the chunk's end date doesn't exceed the final desired end date
        if current_end_date > final_end_date:
            current_end_date = final_end_date

        # Format dates back to strings for the API request
        range_from = current_start_date.strftime("%Y-%m-%d")
        range_to = current_end_date.strftime("%Y-%m-%d")

        print(f"  Fetching chunk: {range_from} to {range_to}")

        data_request = {
            "symbol": symbol,
            "resolution": resolution,
            "date_format": date_format,
            "range_from": range_from,
            "range_to": range_to,
            "cont_flag": "1" # Important for continuous data across chunks if needed
        }

        try:
            print(f"  DEBUG: Sending data request: {data_request}")
            history_response = fyers_obj.history(data=data_request)

            if history_response.get("s") == "ok" and history_response.get("candles"):
                print(f"  Data received successfully for chunk.")
                chunk_df = pd.DataFrame(history_response["candles"],
                                        columns=['datetime', 'open', 'high', 'low', 'close', 'volume'])
                # Append the data from this chunk to the main DataFrame
                all_candles_df = pd.concat([all_candles_df, chunk_df], ignore_index=True)
                print(f"  Processed {len(chunk_df)} rows for this chunk. Total rows so far: {len(all_candles_df)}")

            elif history_response.get("s") == "error":
                print(f"  Fyers API Error for chunk {range_from}-{range_to}: {history_response.get('message', 'No message provided')}")
                # Optional: Decide whether to stop or skip the chunk on error
                # break # Uncomment to stop the whole download if one chunk fails
            else:
                print(f"  Received unexpected response for chunk {range_from}-{range_to}:")
                print(history_response)
                # break # Optional: stop on unexpected response

            # Add a small delay to avoid hitting rate limits between chunks
            time.sleep(1) # Sleep for 1 second

        except Exception as e:
            print(f"  An error occurred during data fetching for chunk {range_from}-{range_to}: {e}")
            # break # Optional: stop on exception

        # Move the start date to the day after the current chunk ended
        current_start_date = current_end_date + relativedelta(days=1)

    print("\nFinished downloading all chunks.")
    return all_candles_df


# --- Main Script Execution ---
if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Download historical data from Fyers API')
    parser.add_argument('--symbol', type=str, default="NSE:RELIANCE-EQ", 
                        help='Stock symbol (e.g., NSE:RELIANCE-EQ)')
    parser.add_argument('--start_date', type=str, default="2021-01-01", 
                        help='Start date in YYYY-MM-DD format')
    parser.add_argument('--end_date', type=str, default="2023-12-31", 
                        help='End date in YYYY-MM-DD format')
    parser.add_argument('--resolution', type=str, default="D", 
                        help='Data resolution (D=daily, 1=1min, etc.)')
    parser.add_argument('--output', type=str, default=None, 
                        help='Output filename (default: auto-generated)')
    
    args = parser.parse_args()
    
    # --- Part 1: Get Access Token ---
    print(f"Authenticating with Fyers API using client ID: {CLIENT_ID}")
    access_token = get_access_token(CLIENT_ID, SECRET_KEY, REDIRECT_URL)

    if not access_token:
        print("Authentication failed. Exiting.")
        sys.exit(1)

    # --- Part 2: Initialize FyersModel ---
    fyers_obj = fyersModel.FyersModel(
        client_id=CLIENT_ID,
        token=access_token,
        log_path=""
    )
    print("\nFyersModel initialized. Ready to fetch data.")

    # --- Part 3: Define Parameters and Fetch Data ---
    stock_symbol = args.symbol
    resolution = args.resolution
    date_format = "1"  # Use epoch for easier conversion later
    start_date_str = args.start_date
    end_date_str = args.end_date

    # Generate output filename if not provided
    if args.output:
        output_filename = args.output
    else:
        output_filename = f"{stock_symbol.replace(':', '_')}_{resolution}_Full.csv"
    
    print(f"\nDownloading data for {stock_symbol} from {start_date_str} to {end_date_str} with resolution {resolution}")
    print(f"Output will be saved to: {output_filename}")

    # Fetch data using the chunking function
    combined_df = fetch_historical_data_chunked(
        fyers_obj,
        stock_symbol,
        resolution,
        date_format,
        start_date_str,
        end_date_str,
        chunk_years=1 # Fetch 1 year at a time (adjust if needed)
    )

    # --- Part 4: Process and Save Combined Data ---
    if not combined_df.empty:
        print(f"\nTotal rows downloaded: {len(combined_df)}")

        # Convert timestamp (epoch seconds) to Datetime objects
        combined_df['datetime'] = pd.to_datetime(combined_df['datetime'], unit='s')

        # Remove duplicate timestamps (if any overlap between chunks, keep first)
        combined_df.drop_duplicates(subset=['datetime'], keep='first', inplace=True)
        print(f"Rows after removing duplicates: {len(combined_df)}")

        # Sort by datetime just in case chunks came out of order
        combined_df.sort_values(by='datetime', inplace=True)

        # Set datetime as the index
        combined_df.set_index('datetime', inplace=True)

        # Add adjclose column
        combined_df['adjclose'] = combined_df['close']

        # Ensure correct column order
        final_df = combined_df[['open', 'high', 'low', 'close', 'adjclose', 'volume']]

        # Save the final combined data
        final_df.to_csv(output_filename)
        print(f"Full historical data saved to {output_filename}")
        print("\nFirst 5 rows of final data:")
        print(final_df.head())
        print("\nLast 5 rows of final data:")
        print(final_df.tail())
    else:
        print("\nNo data was downloaded.")

    # --- End of Script ---
    print("\nScript finished.")