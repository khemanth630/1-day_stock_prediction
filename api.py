import torch
import pandas as pd
import joblib
import torch.nn as nn
import numpy as np
from datetime import datetime, timedelta
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, BertTokenizer, BertForSequenceClassification
from peft import PeftModel
from dataset_app import fetch_historical_data, fetch_alpaca_news, merge_price_and_news
import os
import torch.nn.functional as F
from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    stock = [data["stock"]]
    end_date = data["end_date"]
    
    start_date = (datetime.strptime(end_date, "%Y-%m-%d") - timedelta(days=60)).strftime("%Y-%m-%d")
    price_data = fetch_historical_data(stock, start_date, end_date)

    start_datetime = datetime.strptime(start_date, "%Y-%m-%d")
    end_datetime = datetime.strptime(end_date, "%Y-%m-%d")

    # Step 2: Fetch news data from Alpaca
    news_data = fetch_alpaca_news(stock, start_datetime, end_datetime)

    # Step 3: Merge price and news data in the requested format
    final_data = merge_price_and_news(price_data, news_data)

    # ✅ Drop rows where both summary and headline are empty or NaN
    final_data = final_data[
        ~((final_data["summary"].fillna("").str.strip() == "") & 
        (final_data["headline"].fillna("").str.strip() == ""))
    ]

    # Print summary
    print(f"\nFinal dataset has {len(final_data)} rows")
    print(f"Columns in final dataset: {final_data.columns.tolist()}")

    # Save to CSV
    csv_filename = os.path.join(os.getcwd(), "stock_data.csv")
    final_data = final_data.drop(columns=['price_after', 'price_before'])
    final_data['Date'] = pd.to_datetime(final_data['Date'])
    final_data = final_data.sort_values(by=['Ticker', 'Date'])
    final_data.to_csv(csv_filename, index=False)
    print(f"Combined stock and news data saved to {csv_filename}")

    # 0. Sentiment analysis using FinBERT

    # Check for CUDA
    device = 0 if torch.cuda.is_available() else -1
    print("✅ Using CUDA" if device == 0 else "⚠️ CUDA not available, using CPU")

    # Load data
    df = pd.read_csv("stock_data.csv")
    df = df.dropna(subset=["summary", "headline"], how="all")
    df["summary"] = df["summary"].fillna(df["headline"])

    # Load FinBERT model and tokenizer
    finbert = BertForSequenceClassification.from_pretrained("yiyanghkust/finbert-tone", num_labels=3)
    tokenizer = BertTokenizer.from_pretrained("yiyanghkust/finbert-tone")
    finbert.eval()

    if torch.cuda.is_available():
        finbert.cuda()

    # Text prompts
    prompts = [
        f"Financial Sentiment Analysis for {ticker}:\nNews: \"{text}\"\nDetermine if this news is financially positive, neutral, or negative."
        for text, ticker in zip(df["summary"], df["Ticker"])
    ]

    # Tokenize in batches
    batch_size = 32
    all_probs = []

    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i:i+batch_size]
        inputs = tokenizer(batch_prompts, padding=True, truncation=True, return_tensors="pt")

        if device == 0:
            inputs = {k: v.cuda() for k, v in inputs.items()}

        with torch.no_grad():
            outputs = finbert(**inputs)
            logits = outputs.logits
            probs = F.softmax(logits, dim=-1).cpu().numpy()

        all_probs.extend(probs)

    # Add sentiment probability columns
    df["FinBERT_neutral"] = [float(p[0]) for p in all_probs]
    df["FinBERT_positive"] = [float(p[1]) for p in all_probs]
    df["FinBERT_negative"] = [float(p[2]) for p in all_probs]

    # Save the result with probability columns only
    df.to_csv("sentiment_data_news.csv", index=False)

    print("✅ Sentiment analysis with probabilities saved to 'sentiment_data_news.csv'.")

    # 1. Define Transformer model
    class PositionalEncoding(nn.Module):
        def __init__(self, d_model, max_len=5000):
            super().__init__()
            pe = torch.zeros(max_len, d_model)
            position = torch.arange(0, max_len).float().unsqueeze(1)
            div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(np.log(10000.0) / d_model))
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            self.register_buffer('pe', pe)

        def forward(self, x):
            return x + self.pe[:x.size(1), :]

    class StockGenWithSentimentProbs(nn.Module):
        def __init__(self, input_dim=10, num_layers=4):
            super().__init__()
            self.input_proj = nn.Linear(input_dim, 64)
            self.positional_encoding = PositionalEncoding(d_model=64)
            self.transformer = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(d_model=64, nhead=4, dim_feedforward=128, dropout=0.1, activation='gelu'),
                num_layers=num_layers
            )
            self.layer_norm = nn.LayerNorm(64)
            self.fc = nn.Linear(64, 1)

        def forward(self, x):
            x = self.input_proj(x)
            x = self.positional_encoding(x)
            x = x.permute(1, 0, 2)
            x = self.transformer(x)
            x = x.permute(1, 0, 2)
            x = self.layer_norm(x)
            x = x.mean(dim=1)
            return self.fc(x).squeeze(-1)

    # 2. Load data and model
    df = pd.read_csv("sentiment_data_news.csv")

    def compute_volatility(row):
        try:
            close_prices = [row[f"Close_t-{t}"] for t in range(1, 8)]
            return max(close_prices) - min(close_prices)
        except:
            return 0.0

    scaler = joblib.load("stock_global_scaler.pkl")
    ticker = df['Ticker'].iloc[0]
    
    df["Volatility"] = df.apply(compute_volatility, axis=1)
    df.to_csv("sentiment_data_news_with_volatility.csv", index=False)

    # Scale the Close column before lagging
    df["Close"] = scaler.transform(df[["Close"]])
    for lag in range(1, 8):
        df[f"Close_t-{lag}"] = df["Close"].shift(lag)
    df.dropna(inplace=True)

    # 3. Prepare input tensor
    feature_columns = [f"Close_t-{i}" for i in range(1, 8)] + ["FinBERT_neutral", "FinBERT_positive", "FinBERT_negative"]
    sequence = [[row[col] for col in feature_columns] for _, row in df.iterrows()]
    X = torch.tensor([sequence], dtype=torch.float32)

    # 4. Load and run the Transformer model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = StockGenWithSentimentProbs(input_dim=10, num_layers=4).to(device)
    model.load_state_dict(torch.load("stock_model_with_probs.pth", map_location=device))
    model.eval()

    with torch.no_grad():
        predicted_change_scaled = model(X.to(device)).item()

    # Get last close in unscaled form (already obtained earlier)
    last_close_unscaled = scaler.inverse_transform([[df["Close"].iloc[-1]]])[0][0]

    # Get last volatility to scale predicted change
    last_volatility = df["Volatility"].iloc[-1]

    predicted_change_unscaled = scaler.inverse_transform([[predicted_change_scaled]])[0][0]
    predicted_change_unscaled = abs(predicted_change_unscaled)
    
    print(f"Predicted change (unscaled): {predicted_change_unscaled}")
    
    # If the scaler has 'data_min_', 'data_max_' like MinMaxScaler:
    if hasattr(scaler, 'data_min_') and hasattr(scaler, 'data_max_'):
        print("Scaler min values:", scaler.data_min_)
        print("Scaler max values:", scaler.data_max_)
    elif hasattr(scaler, 'mean_') and hasattr(scaler, 'scale_'):
        print("Scaler means:", scaler.mean_)
        print("Scaler standard deviations:", scaler.scale_)
    else:
        print("Scaler does not have standard attributes like data_min_ or scale_.")


    # Adjust predicted change based on volatility
    if predicted_change_scaled > 0:
        predicted_change_scaled_with_volatility = predicted_change_scaled + predicted_change_unscaled/last_volatility
    else:
        predicted_change_scaled_with_volatility = predicted_change_scaled - predicted_change_unscaled/last_volatility
        
    # volatility_factor = 1 / (1 + last_volatility)
    # predicted_change_scaled_with_volatility = predicted_change_scaled * volatility_factor

    predicted_price = last_close_unscaled + predicted_change_scaled_with_volatility

    # 5. Predict trend
    delta = predicted_price - last_close_unscaled
    predicted_trend = (
        "Uptrend" if delta > 0 else
        "Downtrend" if delta < 0 else
        "Sideways"
    )

    # 6. Predict next date
    last_date_str = df.iloc[-1]["Date"]
    last_date = datetime.strptime(last_date_str, "%Y-%m-%d")
    next_date = last_date + timedelta(days=1)

    # 7. Confidence inference using PEFT model
    base_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
    peft_model = PeftModel.from_pretrained(base_model, "./finetuned_lora_model_confidence").to(device)
    peft_model.eval()

    input_text = (
        f"You are a financial analyst model evaluating prediction reliability.\n"
        f"Ticker: {ticker}\n"
        f"Past Data: {df[feature_columns + ['Close']].tail(10).to_dict(orient='records')}\n"
        f"Predicted Close: {predicted_price:.2f}\n"
        f"Predicted Trend: {predicted_trend}\n"
        f"Last Known Close: {last_close_unscaled:.2f}\n"
        f"Please return a confidence score (0-1) for this prediction."
    )
    inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)

    with torch.no_grad():
        output_ids = peft_model.generate(
            **inputs,
            max_length=16,
            temperature=1,
            top_p=0.9,
            do_sample=True
        )
        confidence_score = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()

    last_volatility = df["Volatility"].iloc[-1]

    # 8. Recommendation inference using PEFT model
    peft_model_recommendation = PeftModel.from_pretrained(base_model, "./finetuned_lora_model_recommendation").to(device)
    peft_model_recommendation.eval()

    recommendation_input_text = (
        f"You are a financial assistant model. Based on the following data, provide a stock recommendation (Buy, Sell, Hold):\n"
        f"Predicted Trend: {predicted_trend}\n"
        f"Confidence: {confidence_score}\n"
        f"Volatility: {last_volatility:.2f}\n"
        f"Predicted Close: {predicted_price:.2f}\n"
        f"Past 7-Day Closing Prices: {df[feature_columns].tail(7).to_dict(orient='records')}\n"
        f"Recommendation (Buy, Sell, Hold):"
    )
    inputs_recommendation = tokenizer(recommendation_input_text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)

    with torch.no_grad():
        output_ids_recommendation = peft_model_recommendation.generate(
            **inputs_recommendation,
            max_length=8,
            temperature=0.8,
            top_p=0.9,
            do_sample=True
        )
        predicted_recommendation = tokenizer.decode(output_ids_recommendation[0], skip_special_tokens=True).strip()

    # 9. Save final prediction output
    prediction_row = {
        "Date": next_date.strftime("%Y-%m-%d"),
        "Ticker": ticker,
        "Last_Close": last_close_unscaled,
        "Predicted_Close_Next_Day": predicted_price,
        "Predicted_Trend": predicted_trend,
        "Predicted_Confidence": confidence_score,
        "Last_Volatility": last_volatility,
        "Predicted_Recommendation": predicted_recommendation
    }
    prediction_df = pd.DataFrame([prediction_row])
    # prediction_df.to_csv("predicted_next_day_price.csv", index=False)
    return jsonify({
        "next_date": (next_date - timedelta(days=1)).strftime("%Y-%m-%d"),
        "ticker": ticker,
        "predicted_close": predicted_price,
        "predicted_trend": predicted_trend,
        "confidence": confidence_score,
        "volatility": last_volatility,
        "recommendation": predicted_recommendation
    })

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)