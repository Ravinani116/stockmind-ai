# Stock Analyzer AI — India MVP

A personal research dashboard for Indian equities. It downloads daily historical data,
calculates technical indicators, and estimates the probability of a positive return over
a selected number of trading days.

## Important limitations
- `yfinance` is a convenient prototype source, not an official NSE/BSE market-data feed.
- Data may be delayed, incomplete, revised, or unavailable. Check timestamps and verify
  important values against a licensed source.
- The prediction is an experimental statistical estimate, not a guaranteed price forecast
  or investment recommendation.
- The first version uses daily candles. It does not provide intraday forecasts.
- No broker connection or automated trading is included.

## Setup (Windows PowerShell)

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

## Setup (Ubuntu / macOS)

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

Open the local URL printed by Streamlit (normally http://localhost:8501).

## Symbols
Enter NSE symbols such as `RELIANCE`, `TCS`, `INFY`, `HDFCBANK`, or `SBIN`.
The app appends `.NS` by default. For BSE, enter a Yahoo Finance symbol ending in `.BO`.

## Project layout

```text
stock_analyzer_ai/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
└── app/
    ├── market_data.py
    └── analysis.py
```

## Next engineering steps
1. Replace/augment the prototype data source with licensed NSE/BSE data.
2. Persist downloaded data and model evaluation in SQLite.
3. Add a proper natural-language chat layer after the deterministic analysis is validated.
4. Add fundamentals, corporate actions, news, and walk-forward backtesting.
5. Monitor model calibration and performance after transaction costs.
