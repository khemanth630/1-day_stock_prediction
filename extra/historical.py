import yfinance as yf
import pandas as pd
import os
from datetime import datetime

# List of top 5 stocks
top_stocks = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]

# Define custom date range
start_date = "2025-01-01"
end_date = "2025-03-01"

# Initialize list to store processed data
data_list = []

# Fetch stock data for each ticker
for stock in top_stocks:
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
                
                # Create a row dictionary with the required columns
                row = {
                    "Date": current_date.strftime('%Y-%m-%d'),
                    "Open": float(raw_data.loc[current_date, 'Open']),
                    "High": float(raw_data.loc[current_date, 'High']),
                    "Low": float(raw_data.loc[current_date, 'Low']),
                    "Close": float(raw_data.loc[current_date, 'Close']),
                    "Adj Close": float(raw_data.loc[current_date, 'Adj Close']),
                    "Volume": int(raw_data.loc[current_date, 'Volume']),
                    "Ticker": stock
                }
                
                # Add the previous 7 days closing prices
                for j in range(1, 8):
                    previous_date = dates[i-j]
                    row[f"Close_t-{j}"] = float(raw_data.loc[previous_date, 'Close'])
                
                structured_data.append(row)
            
            # Convert list of dictionaries to DataFrame
            stock_df = pd.DataFrame(structured_data)
            data_list.append(stock_df)
            
            print(f"Successfully processed data for {stock} - {len(structured_data)} rows")
        else:
            print(f"No data found for {stock} within the specified date range.")
    except Exception as e:
        print(f"Error processing data for {stock}: {e}")

# Combine all stocks data
if data_list:
    final_df = pd.concat(data_list, ignore_index=True)
    
    # Ensure columns are in the specified order
    column_order = [
        "Date", "Open", "High", "Low", "Close", "Adj Close", "Volume", "Ticker",
        "Close_t-1", "Close_t-2", "Close_t-3", "Close_t-4", "Close_t-5", "Close_t-6", "Close_t-7"
    ]
    
    final_df = final_df[column_order]
    
    # Add validation checking
    print(f"Total rows in final dataset: {len(final_df)}")
    print(f"Unique tickers: {final_df['Ticker'].unique()}")
    print(f"Sample of first row:\n{final_df.iloc[0]}")
    
    # Save to CSV in the same directory
    csv_filename = os.path.join(os.getcwd(), "top_5_stocks_data.csv")
    final_df.to_csv(csv_filename, index=False)
    print(f"Stock data saved to {csv_filename}")
else:
    print("No data fetched for the specified date range and tickers.")