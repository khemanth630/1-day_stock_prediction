import requests
import time
import sys

sys.stdout.reconfigure(encoding='utf-8')

# Hardcoded number of tickers you want
n = 10

# Alpaca credentials
ALPACA_API_KEY = "PK2S7TD0TZPID9IIJN1M"
ALPACA_SECRET_KEY = "bZ5nbv8AMOf8eEwH8pfyqpv96LVmFyJAJFADqcII"
ALPACA_ASSET_URL = "https://paper-api.alpaca.markets/v2/assets"
ALPACA_BASE_URL = "https://data.alpaca.markets/v1beta1"

def is_active_on_alpaca(symbol):
    """
    Check if a stock symbol is active (not delisted) on Alpaca.
    """
    url = f"{ALPACA_ASSET_URL}/{symbol}"
    headers = {
        "APCA-API-KEY-ID": ALPACA_API_KEY,
        "APCA-API-SECRET-KEY": ALPACA_SECRET_KEY
    }
    
    try:
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            asset_info = response.json()
            return asset_info.get("status", "inactive") == "active"
        else:
            print(f"[Error] Failed to check {symbol}: {response.status_code}")
            return False
    except Exception as e:
        print(f"[Exception] Checking {symbol}: {e}")
        return False

def get_top_n_tickers(n):
    """
    Continuously fetch tickers from Yahoo Finance and validate them one by one
    until n active tickers are collected, ensuring no duplicates.
    """
    yahoo_url = "https://query1.finance.yahoo.com/v1/finance/screener/predefined/saved"
    headers = {
        "User-Agent": "Mozilla/5.0"
    }
    
    valid_tickers = set()  # Use a set to avoid duplicates
    valid_tickers_count = 0  # Counter for valid tickers
    total_tickers_count = 0  # Counter for total tickers fetched from Yahoo Finance
    offset = 0  # To keep track of the current batch offset

    while valid_tickers_count < n:
        params = {
            "scrIds": "LARGEST_MARKET_CAP",
            "count": "250",  # Fetch 50 per batch
            "offset": offset  # Use the current offset
        }

        try:
            response = requests.get(yahoo_url, headers=headers, params=params)
            response.raise_for_status()
            data = response.json()
            quotes = data["finance"]["result"][0]["quotes"]

            if not quotes:
                print("No more stocks to fetch.")
                break

            for quote in quotes:
                ticker = quote["symbol"]
                total_tickers_count += 1  # Increment total tickers count
                print(f"Checking {ticker}...")

                if is_active_on_alpaca(ticker) and ticker not in valid_tickers:
                    valid_tickers.add(ticker)  # Add unique and valid ticker
                    valid_tickers_count += 1  # Increment count for each valid ticker
                    print(f"[OK] {ticker} is active. Total checked: {total_tickers_count}, Valid collected: {valid_tickers_count}")
                elif ticker in valid_tickers:
                    print(f"[DUPLICATE] {ticker} already in list.")
                else:
                    print(f"[FAIL] {ticker} is inactive.")

                if valid_tickers_count >= n:  # Break if we've collected enough tickers
                    break

                time.sleep(0.2)  # To be polite with API limits

            offset += 250  # Increment offset for next batch

        except Exception as e:
            print(f"[Exception] Failed to fetch tickers: {e}")
            break

    print(f"✅ Final list has {valid_tickers_count} active tickers out of {total_tickers_count} checked tickers.")
    return list(valid_tickers)

# Run it
tickers = get_top_n_tickers(n)
print(tickers)
