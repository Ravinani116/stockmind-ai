from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


FEATURES = [
    "return_1d", "return_5d", "return_20d",
    "rsi_14", "ema_gap", "macd_hist",
    "volatility_20d", "volume_ratio",
]


def calculate_indicators(prices: pd.DataFrame) -> pd.DataFrame:
    df = prices.copy()
    close = df["Close"].astype(float)
    volume = df["Volume"].astype(float)

    df["return_1d"] = close.pct_change()
    df["return_5d"] = close.pct_change(5)
    df["return_20d"] = close.pct_change(20)

    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    rs = gain / loss.replace(0, np.nan)
    df["rsi_14"] = 100 - (100 / (1 + rs))
    df["rsi_14"] = df["rsi_14"].fillna(100).where(loss.eq(0), df["rsi_14"])

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    df["macd"] = macd
    df["macd_signal"] = signal
    df["macd_hist"] = macd - signal

    ema20 = close.ewm(span=20, adjust=False).mean()
    ema50 = close.ewm(span=50, adjust=False).mean()
    df["ema20"] = ema20
    df["ema50"] = ema50
    df["ema_gap"] = (ema20 / ema50) - 1

    df["volatility_20d"] = df["return_1d"].rolling(20).std() * np.sqrt(252)
    avg_volume = volume.rolling(20).mean()
    df["volume_ratio"] = volume / avg_volume.replace(0, np.nan)

    df["sma20"] = close.rolling(20).mean()
    df["sma50"] = close.rolling(50).mean()
    df["support_20d"] = df["Low"].rolling(20).min()
    df["resistance_20d"] = df["High"].rolling(20).max()
    return df.replace([np.inf, -np.inf], np.nan)


def analyze(prices: pd.DataFrame, horizon: int = 5) -> dict:
    if horizon not in (1, 5, 20, 60, 120):
        raise ValueError("Choose a supported horizon: 1, 5, 20, 60 or 120 trading days.")

    df = calculate_indicators(prices)
    latest = df.iloc[-1]
    current_price = float(latest["Close"])

    if latest["ema20"] > latest["ema50"] and latest["rsi_14"] >= 50:
        trend = "Bullish"
    elif latest["ema20"] < latest["ema50"] and latest["rsi_14"] < 50:
        trend = "Bearish"
    else:
        trend = "Mixed / neutral"

    # Target is the future horizon return. Features at time t only use data available at t.
    labeled = df[FEATURES].copy()
    future_return = df["Close"].shift(-horizon) / df["Close"] - 1
    labeled["target"] = (future_return > 0).astype(int)
    labeled.loc[future_return.isna(), "target"] = np.nan
    labeled = labeled.replace([np.inf, -np.inf], np.nan).dropna()

    current_features = df[FEATURES].tail(1)
    if current_features.isna().any(axis=None) or len(labeled) < 250:
        raise ValueError("Not enough clean historical observations to train this model.")

    split = int(len(labeled) * 0.8)
    train = labeled.iloc[:split]
    test = labeled.iloc[split:]

    if train["target"].nunique() < 2 or test["target"].nunique() < 2:
        raise ValueError("The historical sample does not contain enough class variation.")

    def make_model():
        return Pipeline([
            ("scale", StandardScaler()),
            ("model", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ])

    evaluation_model = make_model()
    evaluation_model.fit(train[FEATURES], train["target"].astype(int))
    test_prob = evaluation_model.predict_proba(test[FEATURES])[:, 1]
    test_pred = (test_prob >= 0.5).astype(int)

    # Refit on all labeled history for the latest observation's estimate.
    final_model = make_model()
    final_model.fit(labeled[FEATURES], labeled["target"].astype(int))
    probability_up = float(final_model.predict_proba(current_features[FEATURES])[:, 1][0])

    return {
        "frame": df,
        "current_price": current_price,
        "trend": trend,
        "rsi": float(latest["rsi_14"]),
        "ema20": float(latest["ema20"]),
        "ema50": float(latest["ema50"]),
        "macd_hist": float(latest["macd_hist"]),
        "support": float(latest["support_20d"]),
        "resistance": float(latest["resistance_20d"]),
        "probability_up": probability_up,
        "horizon": horizon,
        "test_accuracy": float(accuracy_score(test["target"].astype(int), test_pred)),
        "test_brier": float(brier_score_loss(test["target"].astype(int), test_prob)),
        "test_rows": int(len(test)),
        "data_date": str(df.index[-1].date()),
        "rows": int(len(df)),
    }
