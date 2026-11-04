from __future__ import annotations

import re
import pandas as pd
import yfinance as yf


def normalize_symbol(raw_symbol: str) -> str:
    """Default bare Indian equity symbols to NSE (.NS); accept explicit .BO/.NS."""
    symbol = raw_symbol.strip().upper().replace(" ", "")
    if not symbol:
        raise ValueError("Enter a stock symbol, for example RELIANCE.")
    if not re.fullmatch(r"[A-Z0-9&^._=-]{1,30}", symbol):
        raise ValueError("The symbol contains unsupported characters.")
    if symbol.startswith("^") or symbol.endswith((".NS", ".BO")):
        return symbol
    return f"{symbol}.NS"


def fetch_history(raw_symbol: str, period: str = "5y") -> tuple[str, pd.DataFrame]:
    symbol = normalize_symbol(raw_symbol)
    ticker = yf.Ticker(symbol)
    data = ticker.history(period=period, interval="1d", auto_adjust=True)

    if data is None or data.empty:
        raise ValueError(
            f"No historical data returned for {symbol}. Check the symbol or try an "
            "explicit Yahoo Finance suffix such as .NS or .BO."
        )

    required = {"Open", "High", "Low", "Close", "Volume"}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Market data is missing required columns: {', '.join(sorted(missing))}")

    data = data[list(required)].copy().sort_index()
    data = data[~data.index.duplicated(keep="last")]
    data = data.replace([float("inf"), float("-inf")], pd.NA).dropna(subset=["Close"])
    data = data[data["Close"] > 0]

    if len(data) < 120:
        raise ValueError(
            f"Only {len(data)} daily rows were returned. At least 120 are needed "
            "for this prototype analysis."
        )

    return symbol, data
