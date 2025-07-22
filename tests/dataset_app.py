import yfinance as yf
import pandas as pd
import os
import requests
from datetime import datetime, timedelta
import requests
import time
import sys
from dotenv import load_dotenv

sys.stdout.reconfigure(encoding='utf-8')
load_dotenv()

ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")
ALPACA_BASE_URL = "https://data.alpaca.markets/v1beta1"
ALPACA_ASSET_URL = "https://paper-api.alpaca.markets/v2/assets"

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

# List of top 50 stocks
# top_stocks = get_top_n_tickers(50)
top_stocks=["CTSH"]

# Define custom date range
start_date = "2022-01-01"
end_date = "2022-06-06"

# Convert to datetime objects for easier manipulation
start_datetime = datetime.strptime(start_date, "%Y-%m-%d")
end_datetime = datetime.strptime(end_date, "%Y-%m-%d")

# Function to fetch historical price data
def fetch_historical_data(symbols, start_date, end_date):
    # Initialize list to store processed data
    data_list = []

    # Fetch stock data for each ticker
    for stock in symbols:
        try:
            # Use yfinance only for fetching the raw data
            raw_data = yf.download(stock, start=start_date, end=end_date, auto_adjust=False)
            
            if not raw_data.empty:
                # Manually structure the data
                structured_data = []
                
                # Convert index to list of dates
                dates = raw_data.index.tolist()
                
                # Process each date in the data
                for i in range(7, len(dates)):  # Start from 8th day to have 7 days of history
                    current_date = dates[i]
                    
                    # Create a row dictionary with the required columns - Fixed: use .iloc[0] to access Series values
                    row = {
                        "Date": current_date.strftime('%Y-%m-%d'),
                        "Open": raw_data.loc[current_date, 'Open'].iloc[0] if hasattr(raw_data.loc[current_date, 'Open'], 'iloc') else raw_data.loc[current_date, 'Open'],
                        "High": raw_data.loc[current_date, 'High'].iloc[0] if hasattr(raw_data.loc[current_date, 'High'], 'iloc') else raw_data.loc[current_date, 'High'],
                        "Low": raw_data.loc[current_date, 'Low'].iloc[0] if hasattr(raw_data.loc[current_date, 'Low'], 'iloc') else raw_data.loc[current_date, 'Low'],
                        "Close": raw_data.loc[current_date, 'Close'].iloc[0] if hasattr(raw_data.loc[current_date, 'Close'], 'iloc') else raw_data.loc[current_date, 'Close'],
                        "Adj Close": raw_data.loc[current_date, 'Adj Close'].iloc[0] if hasattr(raw_data.loc[current_date, 'Adj Close'], 'iloc') else raw_data.loc[current_date, 'Adj Close'],
                        "Volume": int(raw_data.loc[current_date, 'Volume'].iloc[0]) if hasattr(raw_data.loc[current_date, 'Volume'], 'iloc') else int(raw_data.loc[current_date, 'Volume']),
                        "Ticker": stock
                    }
                    
                    # Add the previous 7 days closing prices
                    for j in range(1, 8):
                        previous_date = dates[i-j]
                        prev_close = raw_data.loc[previous_date, 'Close']
                        row[f"Close_t-{j}"] = prev_close.iloc[0] if hasattr(prev_close, 'iloc') else prev_close
                    
                    structured_data.append(row)
                
                # Convert list of dictionaries to DataFrame
                stock_df = pd.DataFrame(structured_data)
                data_list.append(stock_df)
                
                print(f"Successfully processed price data for {stock} - {len(structured_data)} rows")
            else:
                print(f"No price data found for {stock} within the specified date range.")
        except Exception as e:
            print(f"Error processing price data for {stock}: {e}")

    # Combine all stocks data
    if data_list:
        final_df = pd.concat(data_list, ignore_index=True)
        
        # Ensure columns are in the specified order
        column_order = [
            "Date", "Open", "High", "Low", "Close", "Adj Close", "Volume", "Ticker",
            "Close_t-1", "Close_t-2", "Close_t-3", "Close_t-4", "Close_t-5", "Close_t-6", "Close_t-7"
        ]
        
        final_df = final_df[column_order]
        return final_df
    else:
        print("No price data fetched for the specified date range and tickers.")
        return pd.DataFrame()

# Function to fetch news data from Alpaca


def fetch_alpaca_news(symbols, start_date, end_date):
    # Format dates for Alpaca API
    current_date = start_date
    news_list = []

    # Join symbols with comma for API request
    symbols_str = ",".join(symbols)

    # Set up the URL and headers
    url = f"{ALPACA_BASE_URL}/news"
    headers = {
        "APCA-API-KEY-ID": ALPACA_API_KEY,
        "APCA-API-SECRET-KEY": ALPACA_SECRET_KEY
    }
    
    while current_date <= end_date:
        # Format current date range for API request
        start_str = current_date.strftime("%Y-%m-%dT%H:%M:%SZ")
        next_date = current_date + timedelta(days=1)
        end_str = next_date.strftime("%Y-%m-%dT%H:%M:%SZ")

        # Set up parameters for the request
        params = {
            "symbols": symbols_str,
            "start": start_str,
            "end": end_str,
            "limit": 50  # Maximum allowed per request
        }

        try:
            # Make the request
            response = requests.get(url, headers=headers, params=params)

            # Check if request was successful
            if response.status_code == 200:
                news_data = response.json()

                # Process each news item
                for news in news_data.get('news', []):
                    # Get symbols mentioned in the news
                    news_symbols = news.get('symbols', [])

                    # Parse the timestamp
                    published_at = news.get('created_at', '')
                    published_date = datetime.strptime(published_at, "%Y-%m-%dT%H:%M:%SZ").strftime('%Y-%m-%d') if published_at else ''

                    # For each mentioned symbol, create a news entry
                    for symbol in news_symbols:
                        if symbol in symbols:  # Only include our target stocks
                            news_entry = {
                                "Ticker": symbol,
                                "headline": news.get('headline', ''),
                                "summary": news.get('summary', ''),
                                "published_at": published_at,
                                "Date": published_date
                            }
                            news_list.append(news_entry)

                print(f"Fetched news for {symbols} {current_date.strftime('%Y-%m-%d')}")
            else:
                print(f"Error fetching news: {response.status_code} - {response.text}")
        
        except Exception as e:
            print(f"Exception while fetching news: {e}")

        # Move to the next day
        current_date = next_date

    # Convert to DataFrame
    news_df = pd.DataFrame(news_list)
    print(f"Successfully fetched {len(news_df)} news items for {symbols_str}")
    return news_df


# Function to combine price and news data with the specific format requested
def merge_price_and_news(price_df, news_df):
    if price_df.empty:
        return pd.DataFrame()  # Return empty DataFrame if price data is empty
    
    if news_df.empty:
        # If no news data, create empty columns for news-related fields
        final_df = price_df.copy()
        final_df['headline'] = ''
        final_df['summary'] = ''
        final_df['published_at'] = ''
        final_df['price_before'] = None
        final_df['price_after'] = None
        
        # Reorder columns to match the requested format
        column_order = [
            "Ticker", "headline", "summary", "published_at", "price_before", "price_after",
            "Date", "Open", "High", "Low", "Close", "Adj Close", "Volume",
            "Close_t-1", "Close_t-2", "Close_t-3", "Close_t-4", "Close_t-5", "Close_t-6", "Close_t-7"
        ]
        
        return final_df[column_order]
    
    # Process news data
    news_with_price = []
    
    for _, news_row in news_df.iterrows():
        ticker = news_row['Ticker']
        news_date = news_row['Date']
        news_date_formatted = news_date.split('T')[0]

        # Find price data for this ticker on the news date
        matching_price = price_df[(price_df['Ticker'] == ticker) & (price_df['Date'] == news_date_formatted)]
        
        if not matching_price.empty:
            # Get the close price on the news date
            current_price = matching_price['Close'].iloc[0]
            
            # Try to get the previous day's close price
            prev_price = None
            prev_day_data = price_df[(price_df['Ticker'] == ticker) & (price_df['Date'] < news_date)].sort_values('Date', ascending=False)
            if not prev_day_data.empty:
                prev_price = prev_day_data['Close'].iloc[0]
            
            # Try to get the next day's close price
            next_price = None
            next_day_data = price_df[(price_df['Ticker'] == ticker) & (price_df['Date'] > news_date)].sort_values('Date')
            if not next_day_data.empty:
                next_price = next_day_data['Close'].iloc[0]
            
            # Create a new row with news and price data
            combined_row = {
                "Ticker": ticker,
                "headline": news_row['headline'],
                "summary": news_row['summary'],
                "published_at": news_row['published_at'],
                "price_before": prev_price,
                "price_after": current_price,
                "Date": news_date
            }
            
            # Add all price data columns
            for col in matching_price.columns:
                if col not in combined_row:
                    combined_row[col] = matching_price[col].iloc[0]
            
            news_with_price.append(combined_row)
    
    # If we have news with price data
    if news_with_price:
        news_price_df = pd.DataFrame(news_with_price)
        
        # Add rows from price_df that don't have news
        all_price_dates = set(zip(price_df['Ticker'], price_df['Date']))
        news_dates = set(zip(news_price_df['Ticker'], news_price_df['Date']))
        missing_dates = all_price_dates - news_dates
        
        missing_rows = []
        for ticker, date in missing_dates:
            row_data = price_df[(price_df['Ticker'] == ticker) & (price_df['Date'] == date)].iloc[0].to_dict()
            row_data.update({
                "headline": "",
                "summary": "",
                "published_at": "",
                "price_before": None,
                "price_after": None
            })
            missing_rows.append(row_data)
        
        if missing_rows:
            missing_df = pd.DataFrame(missing_rows)
            final_df = pd.concat([news_price_df, missing_df], ignore_index=True)
        else:
            final_df = news_price_df
    else:
        # No news matches with price data
        final_df = price_df.copy()
        final_df['headline'] = ''
        final_df['summary'] = ''
        final_df['published_at'] = ''
        final_df['price_before'] = None
        final_df['price_after'] = None
    
    # Reorder columns to match the requested format
    column_order = [
        "Ticker", "headline", "summary", "published_at", "price_before", "price_after",
        "Date", "Open", "High", "Low", "Close", "Adj Close", "Volume",
        "Close_t-1", "Close_t-2", "Close_t-3", "Close_t-4", "Close_t-5", "Close_t-6", "Close_t-7"
    ]
    
    # Ensure all columns exist
    for col in column_order:
        if col not in final_df.columns:
            final_df[col] = None
    
    return final_df[column_order]

# Main execution
if __name__ == "__main__":
    # Step 1: Fetch historical price data
    price_data = fetch_historical_data(top_stocks, start_date, end_date)
    
    # Step 2: Fetch news data from Alpaca
    news_data = fetch_alpaca_news(top_stocks, start_datetime, end_datetime)
    
    # Step 3: Merge price and news data in the requested format
    final_data = merge_price_and_news(price_data, news_data)
    # final_data= news_data
    
    # Print summary
    print(f"\nFinal dataset has {len(final_data)} rows")
    print(f"Columns in final dataset: {final_data.columns.tolist()}")
    
    # Save to CSV
    csv_filename = os.path.join(os.getcwd(), "stock_data_news_ctsh.csv")
    final_data = final_data.drop(columns=['price_after', 'price_before'])
    final_data['Date'] = pd.to_datetime(final_data['Date'])
    final_data = final_data.sort_values(by=['Ticker', 'Date'])
    final_data.to_csv(csv_filename, index=False)
    print(f"Combined stock and news data saved to {csv_filename}")
    