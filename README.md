# StockGEN: Smarter Stock Forecasting 📈

StockGEN is a stock prediction application that utilizes historical stock prices, sentiment analysis, and volatility metrics to forecast stock trends and provide investment recommendations.

### Features:
- **Stock Price Prediction**: Predicts the next day's closing price for a given stock.
- **Trend Analysis**: Predicts whether the stock will go up, down, or stay sideways.
- **Confidence Scoring**: Provides a confidence level for the predicted trend.
- **Volatility**: Measures the volatility of stock price based on historical data.
- **Investment Recommendations**: Provides Buy/Sell/Hold recommendations based on the analysis.

---

## Prerequisites

1. **Python 3.x**: Make sure you have Python 3.6 or higher installed on your system.
2. **Required Libraries**:
   - `streamlit`
   - `requests`
   - `pandas`
   - `scikit-learn`
   - `flask`
   - `joblib`
   - `datetime`
   
   You can install all necessary dependencies by running:
   ```bash
   pip install -r requirements.txt
   ```

---

## Running the Project

### Step 1: Start the Flask API

1. Navigate to the `api.py` file and run the Flask app:
   ```bash
   python api.py
   ```
   The Flask API will be hosted locally at `http://localhost:5000`.

### Step 2: Run the Streamlit Frontend

1. In a new terminal window, navigate to the directory where the Streamlit app is located.
2. Run the following command:
   ```bash
   streamlit run app.py
   ```
   This will launch the Streamlit frontend in your web browser. The app will prompt you to input a stock ticker and select an end date for the prediction.

---

## Usage

1. **Enter a stock ticker** (e.g., `AAPL` for Apple) in the text box.
2. **Select the end date** for your prediction.
3. **Click "Predict Now"** to generate the stock prediction, including:
   - Predicted Close Price
   - Trend (Up, Down, Sideways)
   - Confidence in the Prediction
   - Volatility
   - Recommendation (Buy/Sell/Hold)

The prediction is based on historical stock data and sentiment analysis from news sources.
