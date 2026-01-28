# stremlit_app.py
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.regression.rolling import RollingOLS
import plotly.graph_objects as go
import os
from dotenv import load_dotenv
from openai import OpenAI

# ==================================
# LOAD .env
# ==================================
load_dotenv()
st.write(f"DEBUG: OPENAI_API_KEY loaded? {os.getenv('OPENAI_API_KEY') is not None}")

# ==================================
# OPENAI CLIENT
# ==================================
client = OpenAI()

# ==================================
# PAGE CONFIG
# ==================================
st.set_page_config(
    page_title="AI Statistical Arbitrage Agent",
    layout="wide"
)
st.title("🤖 AI Statistical Arbitrage Agent")
st.caption("Pairs Trading • Rolling Hedge Ratio • AI Reasoning • Stock Pair Scanner")

# ==================================
# SIDEBAR CONTROLS
# ==================================
st.sidebar.header("Strategy Parameters")

ticker_y = st.sidebar.text_input("Dependent Stock (Y)", "DAR")
ticker_x = st.sidebar.text_input("Independent Stock (X)", "HRL")

lookback = st.sidebar.slider("Rolling Window (days)", 30, 120, 60)
z_entry = st.sidebar.slider("Z-Score Entry Threshold", 1.0, 3.0, 2.0)
z_exit = st.sidebar.slider("Z-Score Exit Threshold", 0.0, 0.5, 0.2)

start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2019-01-01"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("today"))

run_strategy_btn = st.sidebar.button("Run Strategy")
scan_pairs_btn = st.sidebar.button("Scan Top Pairs")

# ==================================
# FUNCTIONS
# ==================================
@st.cache_data
def load_data(tickers, start, end):
    data = yf.download(
        tickers,
        start=start,
        end=end,
        auto_adjust=True,
        group_by="ticker",
        progress=False
    )
    close_prices = {t: data[t]["Close"] for t in tickers}
    return pd.DataFrame(close_prices).ffill().bfill().dropna()

def compute_rolling_hedge_ratio(Y, X, window):
    model = RollingOLS(Y, sm.add_constant(X), window=window)
    result = model.fit()
    return result.params.iloc[:, 1]

def compute_zscore(series, window):
    mean = series.rolling(window).mean()
    std = series.rolling(window).std()
    return (series - mean) / std

def compute_performance(returns):
    total_return = (1 + returns).prod() - 1
    sharpe = np.sqrt(252) * returns.mean() / returns.std() if returns.std() != 0 else 0
    equity = (1 + returns).cumprod()
    drawdown = equity / equity.cummax() - 1
    return total_return, sharpe, drawdown.min()

def ai_agent(context, question):
    """Call OpenAI GPT to analyze strategy or scanner context."""
    system_prompt = """
    You are an expert quantitative trading AI agent.
    Analyze statistical arbitrage strategies and stock pairs.
    Suggest better performing pairs if possible.
    Do NOT give financial advice.
    """
    
    user_prompt = f"""
    Context: {context}
    
    Question: {question}
    Answer clearly.
    """
    
    st.write("DEBUG: Sending request to OpenAI...")
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.3
    )
    
    st.write("DEBUG: Response received")
    st.write("DEBUG: Full AI Response")
    st.json(response)
    
    return response.choices[0].message.content

# ==================================
# STRATEGY RUN
# ==================================
if run_strategy_btn:
    data = load_data([ticker_y, ticker_x], start_date, end_date)
    
    if data.empty:
        st.error("No data available.")
        st.stop()
    
    Y = data[ticker_y]
    X = data[ticker_x]
    
    # Price chart
    st.subheader("📊 Stock Prices")
    st.line_chart(data)
    
    # Rolling hedge ratio
    hedge_ratio = compute_rolling_hedge_ratio(Y, X, lookback)
    st.subheader("📐 Rolling Hedge Ratio")
    st.line_chart(hedge_ratio)
    
    # Spread & Z-score
    spread = Y - hedge_ratio * X
    zscore = compute_zscore(spread, lookback)
    
    st.subheader("📉 Spread")
    st.line_chart(spread)
    st.subheader("📏 Z-Score")
    st.line_chart(zscore)
    
    # Overlay chart
    st.subheader("📉 Spread & Z-Score Overlay")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=spread.index, y=spread, name="Spread"))
    fig.add_trace(go.Scatter(x=zscore.index, y=zscore, name="Z-Score", yaxis="y2"))
    fig.update_layout(
        yaxis=dict(title="Spread"),
        yaxis2=dict(title="Z-Score", overlaying="y", side="right"),
        height=450
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Signals & positions
    signals = pd.DataFrame(index=data.index)
    signals["position"] = 0
    signals.loc[zscore > z_entry, "position"] = -1
    signals.loc[zscore < -z_entry, "position"] = 1
    signals.loc[abs(zscore) < z_exit, "position"] = 0
    signals["position"] = signals["position"].ffill().fillna(0)
    
    st.subheader("🧭 Trading Position")
    st.line_chart(signals["position"])
    
    # Returns
    spread_returns = spread.diff()
    strategy_returns = signals["position"].shift(1) * spread_returns
    strategy_returns = strategy_returns.dropna()
    st.subheader("📉 Daily Strategy Returns")
    st.line_chart(strategy_returns)
    
    # Performance metrics
    total_return, sharpe, max_dd = compute_performance(strategy_returns)
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Return", f"{total_return:.2%}")
    col2.metric("Sharpe Ratio", f"{sharpe:.2f}")
    col3.metric("Max Drawdown", f"{max_dd:.2%}")
    
    # Equity curve
    st.subheader("📈 Equity Curve")
    equity_curve = (1 + strategy_returns).cumprod()
    st.line_chart(equity_curve)
    
    # Drawdown
    st.subheader("📉 Drawdown")
    drawdown = equity_curve / equity_curve.cummax() - 1
    st.line_chart(drawdown)
    
    # Trade log
    st.subheader("📋 Recent Trade Log")
    trade_log = pd.DataFrame({
        "Z-Score": zscore,
        "Position": signals["position"],
        "Spread": spread,
        "Returns": strategy_returns
    }).dropna()
    st.dataframe(trade_log.tail(30))
    
    # Save context for AI
    st.session_state["strategy_context"] = {
        "hedge_ratio": round(hedge_ratio.dropna().iloc[-1], 3),
        "zscore": round(zscore.dropna().iloc[-1], 3),
        "total_return": f"{total_return:.2%}",
        "sharpe": round(sharpe, 2),
        "max_dd": f"{max_dd:.2%}"
    }

# ==================================
# STOCK PAIR SCANNER
# ==================================
if scan_pairs_btn:
    st.subheader("🔍 Stock Pair Scanner (Sample Top Pairs)")
    # For demo, using hardcoded pairs; replace with your own scanning logic
    pairs = [("AAPL", "MSFT"), ("GOOG", "AMZN"), ("DAR", "HRL")]
    results = []
    for y, x in pairs:
        df = load_data([y, x], start_date, end_date)
        if df.empty: 
            continue
        spread = df[y] - df[x]
        zscore = (spread - spread.mean()) / spread.std()
        score = abs(zscore.iloc[-1])
        results.append({"Y": y, "X": x, "Z-Score": round(score, 2)})
    scanner_results = pd.DataFrame(results).sort_values(by="Z-Score", ascending=False)
    st.dataframe(scanner_results)
    
    # Save scanner context for AI
    st.session_state["scanner_context"] = {
        "top_pairs": scanner_results.head(5).to_dict(orient="records")
    }

# ==================================
# AI AGENT
# ==================================
st.subheader("🤖 AI Trading Agent")
question = st.text_input("Ask the AI Agent a question:")

ai_context = st.session_state.get("strategy_context") or st.session_state.get("scanner_context")

if st.button("Ask AI Agent"):
    if ai_context is None:
        st.warning("Please run a strategy or scan pairs first!")
    elif not question.strip():
        st.warning("Please enter a question for the AI Agent.")
    else:
        with st.spinner("AI Agent analyzing..."):
            response = ai_agent(ai_context, question)
        st.markdown(response)
