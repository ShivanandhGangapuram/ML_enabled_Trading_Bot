# Fyers API Authorization Guide

This guide explains how to successfully authenticate with the Fyers API when using the Trading Bot.

## Understanding the Authentication Process

The Fyers API uses a two-step authentication process:

1. **Generate Auth Code**: The application redirects you to the Fyers login page where you authenticate.
2. **Generate Access Token**: The application exchanges the auth code for an access token.

## Authentication Steps

When you click "Download Data" with Fyers API selected as the data source, follow these steps:

### Step 1: Browser Login

1. A browser window will open automatically with the Fyers login page.
2. Log in with your Fyers credentials (Client ID, PAN/Mobile, OTP).

### Step 2: URL Redirection

1. After successful login, your browser will redirect to a URL like:
   ```
   http://127.0.0.1/?auth_code=XXXXXXXXXXXXX&state=None
   ```

2. The page might show "This site can't be reached" - **THIS IS NORMAL**.

### Step 3: Copy the URL

1. **IMPORTANT**: Copy the ENTIRE URL from your browser's address bar.
2. The URL contains the authorization code needed for the next step.

### Step 4: Paste the URL

1. Return to the Trading Bot application.
2. Paste the full URL when prompted.
3. The application will extract the auth code and generate an access token.

## Troubleshooting

### "Invalid Auth Code" Error

If you see "invalid auth code" error:

1. The auth code has expired (they typically expire after a few minutes).
2. Try the process again, making sure to paste the URL quickly after login.
3. The improved script will automatically retry up to 3 times.

### "Invalid Redirect URI" Error

If you see "invalid redirect URI" error:

1. Make sure the redirect URI in your Fyers API app settings exactly matches `http://127.0.0.1/`.
2. Check for any trailing slashes or extra characters.

### Browser Doesn't Open

If the browser doesn't open automatically:

1. Copy the URL displayed in the application.
2. Manually paste it into your browser.
3. Continue with the login process as normal.

## Important Notes

- Access tokens are valid for one trading day and expire at the end of the day.
- Your Fyers API credentials should be stored in the `config.py` file.
- To set up your credentials:
  1. Copy `config_template.py` to `config.py`
  2. Edit `config.py` and replace the placeholder values with your actual credentials:
  ```python
  FYERS_CLIENT_ID = "YOUR_CLIENT_ID_HERE"
  FYERS_SECRET_KEY = "YOUR_SECRET_KEY_HERE"
  FYERS_REDIRECT_URL = "YOUR_REDIRECT_URL_HERE"
  FYERS_USERNAME = "YOUR_USERNAME_HERE"
  ```

## Security Considerations

- Keep your API credentials secure.
- The Trading Bot only uses your credentials to download historical data.
- No actual trading is performed through the API in this version of the application.