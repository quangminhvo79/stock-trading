"""
Flask API for checking RSI (Relative Strength Index) of Vietnamese stocks.
Uses vnstock library to fetch historical price data from Vietnamese stock exchanges.
"""

from datetime import datetime, timedelta

import pandas as pd
from flask import Flask, jsonify, request
from vnstock import Vnstock

app = Flask(__name__)

DEFAULT_RSI_PERIOD = 14
DEFAULT_INTERVAL = "1D"
DEFAULT_SOURCE = "VCI"
DEFAULT_INDEX = "VN100"
# Need extra historical data beyond the requested range to compute RSI accurately
RSI_WARMUP_MULTIPLIER = 3

# Valid index groups for screening
VALID_INDEX_GROUPS = [
    "VNINDEX", "VN30", "VN100", "VNALL", "VNX50", "VNMID", "VNSML",
    "HNXIndex", "HNX30", "UpcomIndex", "VNXALL",
    "VNCOND", "VNCONS", "VNDIAMOND", "VNENE", "VNFIN",
    "VNFINLEAD", "VNFINSELECT", "VNHEAL", "VNIND", "VNIT",
    "VNMAT", "VNREAL", "VNSI", "VNUTI"
]


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


def fetch_stock_data(
    symbol: str,
    period: int,
    interval: str,
    end: str | None,
    source: str,
):
    """Fetch stock data and calculate RSI. Returns (df, error_dict)."""
    symbol = symbol.upper().strip()

    try:
        stock = Vnstock().stock(symbol=symbol, source=source)
    except Exception as e:
        return None, {"symbol": symbol, "error": f"Failed to initialize stock: {e}"}

    # Determine date range
    if end is None:
        end_date = datetime.now()
    else:
        end_date = datetime.strptime(end, "%Y-%m-%d")

    # Fetch enough data for RSI calculation
    lookback_days = period * RSI_WARMUP_MULTIPLIER * (2 if interval == "1W" else 1)
    start_date = end_date - timedelta(days=max(lookback_days, 90))

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
        return None, {"symbol": symbol, "error": f"Failed to fetch history: {e}"}

    if df is None or df.empty:
        return None, {"symbol": symbol, "error": "No data returned for this symbol"}

    df = df.sort_values("time").reset_index(drop=True)
    df["rsi"] = calculate_rsi(df["close"], period=period)
    df = df.dropna(subset=["rsi"])

    if df.empty:
        return None, {
            "symbol": symbol,
            "error": "Not enough data to calculate RSI",
        }

    return df, None


def fetch_rsi_for_symbol(
    symbol: str,
    period: int,
    interval: str,
    source: str,
) -> dict:
    """Fetch current RSI for a single symbol (no history)."""
    symbol = symbol.upper().strip()

    df, error = fetch_stock_data(symbol, period, interval, None, source)
    if error:
        return error

    latest = df.iloc[-1]
    rsi_value = round(float(latest["rsi"]), 2)

    # Determine RSI signal
    if rsi_value >= 70:
        signal = "overbought"
    elif rsi_value <= 30:
        signal = "oversold"
    else:
        signal = "neutral"

    return {
        "symbol": symbol,
        "current_rsi": rsi_value,
        "signal": signal,
        "period": period,
        "interval": interval,
        "latest_close": round(float(latest["close"]), 2),
        "latest_date": latest["time"].strftime("%Y-%m-%d"),
    }


def fetch_rsi_history_for_symbol(
    symbol: str,
    period: int,
    interval: str,
    source: str,
    limit: int = 30,
) -> dict:
    """Fetch RSI history for a single symbol."""
    symbol = symbol.upper().strip()

    df, error = fetch_stock_data(symbol, period, interval, None, source)
    if error:
        return error

    # Build history list
    history_records = []
    for _, row in df.tail(limit).iterrows():
        history_records.append(
            {
                "date": row["time"].strftime("%Y-%m-%d"),
                "close": round(float(row["close"]), 2),
                "rsi": round(float(row["rsi"]), 2),
            }
        )

    return {
        "symbol": symbol,
        "period": period,
        "interval": interval,
        "count": len(history_records),
        "history": history_records,
    }


def get_symbols_by_index(index_group: str, source: str) -> tuple[list[str], dict | None]:
    """Get list of stock symbols belonging to an index group."""
    try:
        stock = Vnstock().stock(symbol="ACB", source=source)
        symbols = stock.listing.symbols_by_group(index_group)
        if symbols is None or len(symbols) == 0:
            return [], {"error": f"No symbols found for index: {index_group}"}
        return list(symbols), None
    except Exception as e:
        return [], {"error": f"Failed to get symbols for index {index_group}: {e}"}


def screen_index_by_rsi(
    index_group: str,
    period: int,
    interval: str,
    source: str,
) -> dict:
    """Screen all stocks in an index and group by RSI signal."""
    symbols, error = get_symbols_by_index(index_group, source)
    if error:
        return error

    # Group results by signal
    grouped = {
        "overbought": [],
        "oversold": [],
        "neutral": [],
    }
    errors = []

    for symbol in symbols:
        result = fetch_rsi_for_symbol(symbol, period, interval, source)
        if "error" in result:
            errors.append({"symbol": symbol, "error": result["error"]})
        else:
            signal = result["signal"]
            grouped[signal].append({
                "symbol": result["symbol"],
                "rsi": result["current_rsi"],
                "close": result["latest_close"],
                "date": result["latest_date"],
            })

    # Sort each group by RSI
    grouped["overbought"].sort(key=lambda x: x["rsi"], reverse=True)
    grouped["oversold"].sort(key=lambda x: x["rsi"])
    grouped["neutral"].sort(key=lambda x: x["rsi"], reverse=True)

    return {
        "index": index_group,
        "period": period,
        "interval": interval,
        "summary": {
            "total": len(symbols),
            "overbought": len(grouped["overbought"]),
            "oversold": len(grouped["oversold"]),
            "neutral": len(grouped["neutral"]),
            "errors": len(errors),
        },
        "overbought": grouped["overbought"],
        "oversold": grouped["oversold"],
        "neutral": grouped["neutral"],
        "errors": errors if errors else None,
    }


@app.route("/")
def index():
    return jsonify(
        {
            "service": "Vietnam Stock RSI API",
            "endpoints": {
                "/api/rsi": {
                    "method": "GET",
                    "description": "Get current RSI for one or more Vietnamese stock symbols",
                    "parameters": {
                        "symbol": "(required) Stock symbol(s), comma-separated. E.g. FPT or FPT,VNM,ACB",
                        "period": f"(optional) RSI period, default {DEFAULT_RSI_PERIOD}",
                        "interval": f"(optional) Data interval: 1D, 1W, 1M. Default {DEFAULT_INTERVAL}",
                        "source": f"(optional) Data source: VCI, KBS. Default {DEFAULT_SOURCE}",
                    },
                    "examples": [
                        "/api/rsi?symbol=FPT",
                        "/api/rsi?symbol=FPT,VNM,ACB&period=14",
                    ],
                },
                "/api/rsi/history": {
                    "method": "GET",
                    "description": "Get RSI history data for a stock symbol",
                    "parameters": {
                        "symbol": "(required) Stock symbol. E.g. FPT",
                        "period": f"(optional) RSI period, default {DEFAULT_RSI_PERIOD}",
                        "interval": f"(optional) Data interval: 1D, 1W, 1M. Default {DEFAULT_INTERVAL}",
                        "limit": "(optional) Number of data points, default 30",
                        "source": f"(optional) Data source: VCI, KBS. Default {DEFAULT_SOURCE}",
                    },
                    "examples": [
                        "/api/rsi/history?symbol=FPT",
                        "/api/rsi/history?symbol=VNM&limit=50",
                    ],
                },
                "/api/rsi/screen": {
                    "method": "GET",
                    "description": "Screen all stocks in an index and group by RSI signal",
                    "parameters": {
                        "index": f"(optional) Index group: VN30, VN100, VNINDEX, etc. Default {DEFAULT_INDEX}",
                        "period": f"(optional) RSI period, default {DEFAULT_RSI_PERIOD}",
                        "interval": f"(optional) Data interval: 1D, 1W, 1M. Default {DEFAULT_INTERVAL}",
                        "source": f"(optional) Data source: VCI, KBS. Default {DEFAULT_SOURCE}",
                    },
                    "examples": [
                        "/api/rsi/screen",
                        "/api/rsi/screen?index=VN30",
                        "/api/rsi/screen?index=VN100&period=14",
                    ],
                    "available_indices": VALID_INDEX_GROUPS,
                },
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

    source = request.args.get("source", DEFAULT_SOURCE)

    # Fetch RSI for each symbol
    results = []
    for symbol in symbols:
        result = fetch_rsi_for_symbol(symbol, period, interval, source)
        results.append(result)

    # Return single object for single symbol, array for multiple
    if len(results) == 1:
        response = results[0]
    else:
        response = {"count": len(results), "results": results}

    return jsonify(response)


@app.route("/api/rsi/history")
def get_rsi_history():
    # Parse parameters
    symbol = request.args.get("symbol")
    if not symbol:
        return jsonify({"error": "Missing required parameter: symbol"}), 400

    symbol = symbol.strip().upper()
    if not symbol:
        return jsonify({"error": "No valid symbol provided"}), 400

    try:
        period = int(request.args.get("period", DEFAULT_RSI_PERIOD))
        if period < 2 or period > 200:
            return jsonify({"error": "Period must be between 2 and 200"}), 400
    except ValueError:
        return jsonify({"error": "Invalid period value, must be an integer"}), 400

    try:
        limit = int(request.args.get("limit", 30))
        if limit < 1 or limit > 500:
            return jsonify({"error": "Limit must be between 1 and 500"}), 400
    except ValueError:
        return jsonify({"error": "Invalid limit value, must be an integer"}), 400

    interval = request.args.get("interval", DEFAULT_INTERVAL)
    valid_intervals = ["1D", "1W", "1M"]
    if interval not in valid_intervals:
        return (
            jsonify({"error": f"Invalid interval. Must be one of: {valid_intervals}"}),
            400,
        )

    source = request.args.get("source", DEFAULT_SOURCE)

    result = fetch_rsi_history_for_symbol(symbol, period, interval, source, limit)
    return jsonify(result)


@app.route("/api/rsi/screen")
def screen_rsi():
    # Parse parameters
    index_group = request.args.get("index", DEFAULT_INDEX).upper()
    if index_group not in VALID_INDEX_GROUPS:
        return (
            jsonify({
                "error": f"Invalid index. Must be one of: {VALID_INDEX_GROUPS}"
            }),
            400,
        )

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

    source = request.args.get("source", DEFAULT_SOURCE)

    result = screen_index_by_rsi(index_group, period, interval, source)
    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=3000)
