from __future__ import annotations

import streamlit as st
import plotly.graph_objects as go

from app.market_data import fetch_history
from app.analysis import analyze


st.set_page_config(page_title="Stock Analyzer AI", page_icon="📈", layout="wide")
st.title("📈 Stock Analyzer AI")
st.caption("Indian equity research prototype · Daily historical data · No trading execution")

with st.sidebar:
    st.header("Analysis settings")
    symbol_input = st.text_input("NSE / BSE symbol", value="RELIANCE", help="Examples: TCS, INFY, SBIN. Bare symbols default to NSE.")
    horizon = st.selectbox(
        "Prediction horizon (trading days)",
        options=[1, 5, 20, 60, 120],
        index=1,
        format_func=lambda x: {
            1: "1 day", 5: "5 days", 20: "1 month",
            60: "3 months", 120: "6 months"
        }[x],
    )
    period = st.selectbox("Historical data", ["2y", "5y", "10y"], index=1)
    run = st.button("Analyze stock", type="primary", use_container_width=True)

if "analysis_result" not in st.session_state:
    st.session_state.analysis_result = None
if "symbol" not in st.session_state:
    st.session_state.symbol = None

if run:
    try:
        with st.spinner("Downloading and validating historical market data..."):
            symbol, prices = fetch_history(symbol_input, period)
        with st.spinner("Calculating indicators and evaluating the baseline model..."):
            result = analyze(prices, horizon)
        st.session_state.analysis_result = result
        st.session_state.symbol = symbol
    except Exception as exc:
        st.error(str(exc))

result = st.session_state.analysis_result
symbol = st.session_state.symbol

if result is None:
    st.info("Enter an Indian stock symbol in the sidebar and select **Analyze stock**.")
    st.markdown("""
    **What this prototype provides**
    - Daily adjusted OHLCV history
    - RSI, EMA, MACD, volatility and volume indicators
    - 20-day rolling support and resistance reference levels
    - Experimental probability of a positive return over the selected horizon
    - Chronological holdout evaluation
    """)
    st.stop()

st.subheader(f"{symbol} · Analysis")
st.caption(
    f"Latest available daily candle: {result['data_date']} · "
    f"{result['rows']} rows · Horizon: {result['horizon']} trading days"
)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Latest adjusted close", f"₹{result['current_price']:,.2f}")
c2.metric("Technical trend", result["trend"])
c3.metric("RSI (14)", f"{result['rsi']:.1f}")
c4.metric("Model P(positive return)", f"{result['probability_up']:.1%}")

prob = result["probability_up"]
if prob >= 0.60:
    st.success("The model estimates a higher-than-even probability of a positive return. This is not a buy signal.")
elif prob <= 0.40:
    st.warning("The model estimates a lower-than-even probability of a positive return. This is not a sell signal.")
else:
    st.info("The model's estimated probability is near an even outcome; the signal is inconclusive.")

left, right = st.columns([2, 1])
with left:
    frame = result["frame"].tail(180)
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=frame.index, open=frame["Open"], high=frame["High"],
        low=frame["Low"], close=frame["Close"], name="Daily price"
    ))
    fig.add_trace(go.Scatter(x=frame.index, y=frame["ema20"], name="EMA 20", line={"width": 1.5}))
    fig.add_trace(go.Scatter(x=frame.index, y=frame["ema50"], name="EMA 50", line={"width": 1.5}))
    fig.update_layout(
        title="Daily price and moving averages",
        xaxis_title="Date", yaxis_title="Adjusted price (₹)",
        height=520, xaxis_rangeslider_visible=False,
        legend_orientation="h"
    )
    st.plotly_chart(fig, use_container_width=True)

with right:
    st.markdown("#### Technical levels")
    st.metric("EMA 20", f"₹{result['ema20']:,.2f}")
    st.metric("EMA 50", f"₹{result['ema50']:,.2f}")
    st.metric("20-day support reference", f"₹{result['support']:,.2f}")
    st.metric("20-day resistance reference", f"₹{result['resistance']:,.2f}")
    st.metric("MACD histogram", f"{result['macd_hist']:.3f}")

st.markdown("#### Model validation")
m1, m2, m3 = st.columns(3)
m1.metric("Chronological holdout accuracy", f"{result['test_accuracy']:.1%}")
m2.metric("Brier score (lower is better)", f"{result['test_brier']:.3f}")
m3.metric("Holdout observations", f"{result['test_rows']:,}")

st.caption(
    "The model is a simple logistic-regression baseline, not a proven forecasting system. "
    "Holdout accuracy alone does not establish profitability. Evaluate calibration, "
    "walk-forward performance, transaction costs, slippage and survivorship bias before "
    "using predictions in financial decisions."
)

with st.expander("How to interpret this analysis"):
    st.write(
        "The probability is the model's estimate that the adjusted closing price after "
        f"{result['horizon']} trading days will be above the latest close. It is not the "
        "probability of reaching a particular price. Indicators describe historical "
        "conditions and do not guarantee future movement."
    )

st.divider()
st.subheader("Ask the analyzer")
question = st.text_input(
    "Ask a question about this analysis",
    placeholder="Example: Explain the RSI and trend for this stock"
)
if question:
    q = question.lower()
    if "rsi" in q:
        answer = (
            f"The RSI (14) is {result['rsi']:.1f}. Traditionally, readings above 70 are "
            "called overbought and below 30 oversold. RSI is not a standalone trading signal."
        )
    elif "support" in q or "resistance" in q:
        answer = (
            f"The recent 20-session low reference is ₹{result['support']:,.2f}; "
            f"the recent 20-session high reference is ₹{result['resistance']:,.2f}. "
            "These are rolling historical levels, not guaranteed price floors or ceilings."
        )
    elif "predict" in q or "probability" in q or "forecast" in q:
        answer = (
            f"The baseline model estimates a {result['probability_up']:.1%} probability "
            f"of a positive return over {result['horizon']} trading days. Its chronological "
            f"holdout accuracy was {result['test_accuracy']:.1%} on "
            f"{result['test_rows']} observations. This estimate is uncertain and not a trade instruction."
        )
    elif "trend" in q or "ema" in q or "macd" in q:
        answer = (
            f"The rule-based technical trend is {result['trend']}. EMA20 is "
            f"₹{result['ema20']:,.2f}, EMA50 is ₹{result['ema50']:,.2f}, and the "
            f"MACD histogram is {result['macd_hist']:.3f}."
        )
    else:
        answer = (
            "I can currently explain RSI, support/resistance, trend, EMA, MACD and the "
            "baseline model probability. A full natural-language AI layer is planned for "
            "the next version; it will use these calculated results rather than inventing data."
        )
    st.markdown(answer)

st.caption("Personal research tool only. Not investment advice. Verify market data independently.")
