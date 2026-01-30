"""
Flask API for checking RSI (Relative Strength Index) of Vietnamese stocks.
Uses vnstock library to fetch historical price data from Vietnamese stock exchanges.
"""

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from flask import Flask, jsonify, request
from vnstock import Vnstock

app = Flask(__name__)

DEFAULT_RSI_PERIOD = 14
DEFAULT_INTERVAL = "1D"
DEFAULT_SOURCE = "VCI"
# Need extra historical data beyond the requested range to compute RSI accurately
RSI_WARMUP_MULTIPLIER = 3


def calculate_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Calculate RSI using exponential weighted moving average (Wilder's method)."""
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)

    avg_gain = gain.ewm(alpha=1 / period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def fetch_rsi_for_symbol(
    symbol: str,
    period: int,
    interval: str,
    start: str | None,
    end: str | None,
    source: str,
) -> dict:
    """Fetch stock data and calculate RSI for a single symbol."""
    symbol = symbol.upper().strip()

    try:
        stock = Vnstock().stock(symbol=symbol, source=source)
    except Exception as e:
        return {"symbol": symbol, "error": f"Failed to initialize stock: {e}"}

    # Determine date range
    if end is None:
        end_date = datetime.now()
    else:
        end_date = datetime.strptime(end, "%Y-%m-%d")

    if start is None:
        # Default: fetch enough data for RSI calculation
        lookback_days = period * RSI_WARMUP_MULTIPLIER * (2 if interval == "1W" else 1)
        start_date = end_date - timedelta(days=max(lookback_days, 90))
    else:
        start_date = datetime.strptime(start, "%Y-%m-%d")

    # Extend start date to have enough warmup data for accurate RSI
    warmup_days = period * RSI_WARMUP_MULTIPLIER
    fetch_start = start_date - timedelta(days=warmup_days)

    try:
        df = stock.quote.history(
            start=fetch_start.strftime("%Y-%m-%d"),
            end=end_date.strftime("%Y-%m-%d"),
            interval=interval,
        )
    except Exception as e:
        return {"symbol": symbol, "error": f"Failed to fetch history: {e}"}

    if df is None or df.empty:
        return {"symbol": symbol, "error": "No data returned for this symbol"}

    df = df.sort_values("time").reset_index(drop=True)
    df["rsi"] = calculate_rsi(df["close"], period=period)

    # Trim back to the originally requested date range
    if start is not None:
        df = df[df["time"] >= start_date]

    df = df.dropna(subset=["rsi"])

    if df.empty:
        return {
            "symbol": symbol,
            "error": "Not enough data to calculate RSI for the given range",
        }

    latest = df.iloc[-1]
    rsi_value = round(float(latest["rsi"]), 2)

    # Determine RSI signal
    if rsi_value >= 70:
        signal = "overbought"
    elif rsi_value <= 30:
        signal = "oversold"
    else:
        signal = "neutral"

    # Build history list (last N data points)
    history_records = []
    for _, row in df.tail(30).iterrows():
        history_records.append(
            {
                "date": row["time"].strftime("%Y-%m-%d"),
                "close": round(float(row["close"]), 2),
                "rsi": round(float(row["rsi"]), 2),
            }
        )

    return {
        "symbol": symbol,
        "current_rsi": rsi_value,
        "signal": signal,
        "period": period,
        "interval": interval,
        "latest_close": round(float(latest["close"]), 2),
        "latest_date": latest["time"].strftime("%Y-%m-%d"),
        "history": history_records,
    }


@app.route("/")
def index():
    return jsonify(
        {
            "service": "Vietnam Stock RSI API",
            "endpoints": {
                "/api/rsi": {
                    "method": "GET",
                    "description": "Get RSI for one or more Vietnamese stock symbols",
                    "parameters": {
                        "symbol": "(required) Stock symbol(s), comma-separated. E.g. FPT or FPT,VNM,ACB",
                        "period": f"(optional) RSI period, default {DEFAULT_RSI_PERIOD}",
                        "interval": f"(optional) Data interval: 1D, 1W, 1M. Default {DEFAULT_INTERVAL}",
                        "start": "(optional) Start date YYYY-MM-DD",
                        "end": "(optional) End date YYYY-MM-DD, default today",
                        "source": f"(optional) Data source: VCI, KBS. Default {DEFAULT_SOURCE}",
                    },
                    "examples": [
                        "/api/rsi?symbol=FPT",
                        "/api/rsi?symbol=FPT,VNM,ACB&period=14",
                        "/api/rsi?symbol=HPG&start=2024-06-01&end=2025-01-30",
                    ],
                }
            },
        }
    )


@app.route("/api/rsi")
def get_rsi():
    # Parse parameters
    symbols_param = request.args.get("symbol")
    if not symbols_param:
        return jsonify({"error": "Missing required parameter: symbol"}), 400

    symbols = [s.strip().upper() for s in symbols_param.split(",") if s.strip()]
    if not symbols:
        return jsonify({"error": "No valid symbols provided"}), 400

    try:
        period = int(request.args.get("period", DEFAULT_RSI_PERIOD))
        if period < 2 or period > 200:
            return jsonify({"error": "Period must be between 2 and 200"}), 400
    except ValueError:
        return jsonify({"error": "Invalid period value, must be an integer"}), 400

    interval = request.args.get("interval", DEFAULT_INTERVAL)
    valid_intervals = ["1D", "1W", "1M"]
    if interval not in valid_intervals:
        return (
            jsonify({"error": f"Invalid interval. Must be one of: {valid_intervals}"}),
            400,
        )

    start = request.args.get("start")
    end = request.args.get("end")
    source = request.args.get("source", DEFAULT_SOURCE)

    # Validate date format
    for date_param, date_name in [(start, "start"), (end, "end")]:
        if date_param:
            try:
                datetime.strptime(date_param, "%Y-%m-%d")
            except ValueError:
                return (
                    jsonify(
                        {
                            "error": f"Invalid {date_name} date format. Use YYYY-MM-DD"
                        }
                    ),
                    400,
                )

    # Fetch RSI for each symbol
    results = []
    for symbol in symbols:
        result = fetch_rsi_for_symbol(symbol, period, interval, start, end, source)
        results.append(result)

    # Return single object for single symbol, array for multiple
    if len(results) == 1:
        response = results[0]
    else:
        response = {"count": len(results), "results": results}

    return jsonify(response)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
