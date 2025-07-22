import streamlit as st
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer
import faiss
import os
import time
import requests
from datetime import datetime

# Streamlit configuration
st.set_page_config(page_title="StockGEN", layout="centered")
st.title("📈 StockGEN: Smarter Stock Forecasting")

# ===== STOCK PREDICTION UI =====
st.markdown("### 🎯 Enter Prediction Parameters")
col1, col2 = st.columns(2)
with col1:
    stock = st.text_input("🧾 Stock Ticker", value="AAPL", placeholder="e.g., AAPL")
with col2:
    end_date = st.date_input("🗓️ Select End Date", value=datetime.today())

if st.button("🚀 Predict Now"):
    with st.spinner("🔍 Analyzing market data..."):
        time.sleep(1.5)
        response = requests.post("http://localhost:5000/predict", json={
            "stock": stock,
            "end_date": end_date.strftime("%Y-%m-%d")
        })

        if response.ok:
            result = response.json()

            st.markdown("## 🧠 Prediction Summary")
            col1, col2 = st.columns(2)

            with col1:
                st.metric("📅 Data Till", result["next_date"])
                st.metric("💼 Ticker", result["ticker"])
                st.metric(f"📉 Predicted Close (after {result['next_date']})", f"${result['predicted_close']:.2f}")

            with col2:
                st.metric("📊 Trend", result["predicted_trend"])
                st.metric("🎯 Confidence", result["confidence"])
                st.metric("🌪️ Volatility", f"{result['volatility']:.2f}")

            st.markdown("---")
            st.markdown("### 💡 Final Recommendation")
            recommendation = result["recommendation"]
            color = "green" if recommendation == "Buy" else "red" if recommendation == "Sell" else "orange"
            st.markdown(
                f"<div style='background-color:{color}; padding:15px; border-radius:10px; text-align:center;'>"
                f"<h3 style='color:white;'>📝 {recommendation}</h3></div>",
                unsafe_allow_html=True
            )
        else:
            st.error("❌ Prediction failed. Check server logs.")

# ===== RAG QA CHAT =====
st.markdown("---")
st.markdown("## 🤖 Ask Stock Questions (News & Sentiment Based)")

user_query = st.text_input("💬 Ask me anything from the stock data I retrieved...")

if st.button("📡 Get Answer"):
    if user_query.strip() == "":
        st.warning("Please enter a question.")
    elif not os.path.exists("sentiment_data_news_with_volatility.csv"):
        st.error("⚠️ Prediction data not available yet. Please run a prediction first.")
    else:
        # Load models and embedder once
        @st.cache_resource
        def load_models():
            model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
            tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
            embedder = SentenceTransformer("all-MiniLM-L6-v2")
            return model, tokenizer, embedder

        model, tokenizer, embedder = load_models()

        # Load and prepare CSV rows
        df = pd.read_csv("sentiment_data_news_with_volatility.csv")
        row_texts = df.astype(str).apply(lambda row: ', '.join(f"{col}: {val}" for col, val in row.items()), axis=1).tolist()

        @st.cache_resource
        def embed_rows(rows):
            embeddings = embedder.encode(rows, convert_to_tensor=False)
            index = faiss.IndexFlatL2(len(embeddings[0]))
            index.add(embeddings)
            return index, embeddings

        index, embeddings = embed_rows(row_texts)

        with st.spinner("🔎 Retrieving relevant data..."):
            q_embedding = embedder.encode([user_query])
            D, I = index.search(q_embedding, k=5)
            top_chunks = [row_texts[i] for i in I[0]]
            context = "\n".join(top_chunks)

            prompt = f"""You are a helpful stock assistant. Use the CSV data below to answer the question.

CSV Data:
{context}

Question:
{user_query}

Answer:"""

            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, padding=True, max_length=1024)
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=256,
                    temperature=0.7,
                    top_p=0.9,
                    repetition_penalty=1.1
                )

            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            st.success("✅ Here's what I found:")
            st.markdown(f"**📢 Answer:** {response}")
