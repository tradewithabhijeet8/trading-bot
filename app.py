import sqlite3
from datetime import datetime, timezone, date

import ccxt
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# Educational paper-trading simulator only.
# Public market data only: this app NEVER submits exchange orders.
APP_TITLE = "BTC/USDT Paper Trading Lab"
DB_PATH = "paper_trading.db"
STARTING_BALANCE = 10_000.0
RISK_PERCENT = 1.0
REWARD_RISK = 2.0
FEE_RATE = 0.0005  # estimated 0.05% per side; simulation only
CANDLE_LIMIT = 500
POLL_SECONDS = 5
DAILY_LOSS_LIMIT_PERCENT = 3.0  # stop opening new paper trades after this daily loss
TIMEFRAMES = ["5m", "15m", "1h", "4h"]


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def candle_close_countdown(timeframe):
    """Seconds until the current UTC candle interval closes."""
    seconds_by_tf = {"1m": 60, "5m": 300, "15m": 900, "30m": 1800,
                     "1h": 3600, "2h": 7200, "4h": 14400, "1d": 86400}
    interval = seconds_by_tf.get(timeframe, 900)
    now = datetime.now(timezone.utc).timestamp()
    remaining = max(0, int(interval - (now % interval)))
    return f"{remaining // 60:02d}:{remaining % 60:02d}"


def db_connect():
    conn = sqlite3.connect(DB_PATH, timeout=15)
    conn.row_factory = sqlite3.Row
    return conn


def table_columns(conn, table):
    return {row["name"] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}


def init_db():
    """Create missing tables and add compatible columns without deleting existing records."""
    with db_connect() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS settings (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                active INTEGER NOT NULL DEFAULT 0,
                starting_balance REAL NOT NULL DEFAULT 10000
            )
        """)
        conn.execute("""
            INSERT OR IGNORE INTO settings (id, active, starting_balance)
            VALUES (1, 0, ?)
        """, (STARTING_BALANCE,))
        conn.execute("""
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                side TEXT NOT NULL,
                entry_time TEXT NOT NULL,
                entry_price REAL NOT NULL,
                quantity REAL NOT NULL,
                stop_loss REAL NOT NULL,
                take_profit REAL NOT NULL,
                risk_amount REAL NOT NULL,
                entry_fee REAL NOT NULL DEFAULT 0,
                status TEXT NOT NULL DEFAULT 'OPEN',
                exit_time TEXT,
                exit_price REAL,
                exit_fee REAL NOT NULL DEFAULT 0,
                gross_pnl REAL,
                net_pnl REAL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                time TEXT NOT NULL,
                kind TEXT NOT NULL,
                message TEXT NOT NULL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS processed_candles (
                candle_time TEXT PRIMARY KEY
            )
        """)
        # Add optional columns only if missing; keep all old trade rows intact.
        cols = table_columns(conn, "trades")
        for name, declaration in {
            "strategy": "TEXT DEFAULT 'BOS + RSI'",
            "exit_reason": "TEXT",
        }.items():
            if name not in cols:
                conn.execute(f"ALTER TABLE trades ADD COLUMN {name} {declaration}")


def add_event(kind, message):
    with db_connect() as conn:
        conn.execute(
            "INSERT INTO events (time, kind, message) VALUES (?, ?, ?)",
            (utc_now(), kind, str(message)[:500]),
        )


def get_settings():
    with db_connect() as conn:
        row = conn.execute("SELECT * FROM settings WHERE id = 1").fetchone()
    return dict(row)


def set_active(active):
    with db_connect() as conn:
        conn.execute("UPDATE settings SET active = ? WHERE id = 1", (1 if active else 0,))
    add_event("BOT", "Bot started." if active else "Bot stopped.")


def reset_paper_account():
    # Deliberate reset only when user confirms in the UI.
    with db_connect() as conn:
        conn.execute("DELETE FROM trades")
        conn.execute("DELETE FROM events")
        conn.execute("DELETE FROM processed_candles")
        conn.execute(
            "UPDATE settings SET active = 0, starting_balance = ? WHERE id = 1",
            (STARTING_BALANCE,),
        )
    add_event("BOT", "Paper account reset to $10,000.")


@st.cache_resource
def get_exchange():
    # Public candles only. No API keys and no order methods are used.
    return ccxt.binance({
        "enableRateLimit": True,
        "timeout": 20000,
        "options": {"defaultType": "spot"},
    })


def fetch_candles(timeframe, limit=CANDLE_LIMIT, include_live=False):
    rows = get_exchange().fetch_ohlcv("BTC/USDT", timeframe=timeframe, limit=limit)
    if not rows or len(rows) < 60:
        raise RuntimeError("Exchange returned too few candles. Check internet and retry.")
    df = pd.DataFrame(rows, columns=["timestamp", "Open", "High", "Low", "Close", "Volume"])
    df["Time"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.set_index("Time").drop(columns=["timestamp"])
    # Chart mode includes the currently forming candle. Strategy mode excludes it.
    return df.copy() if include_live else df.iloc[:-1].copy()


def add_indicators(df, swing_lookback=5):
    df = df.copy()
    delta = df["Close"].diff()
    gains = delta.clip(lower=0)
    losses = -delta.clip(upper=0)
    avg_gain = gains.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    avg_loss = losses.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, float("nan"))
    df["RSI"] = (100 - (100 / (1 + rs))).fillna(50)
    df["EMA20"] = df["Close"].ewm(span=20, adjust=False).mean()
    # Only prior bars define the swing levels.
    df["SwingHigh"] = df["High"].rolling(swing_lookback).max().shift(1)
    df["SwingLow"] = df["Low"].rolling(swing_lookback).min().shift(1)
    df["BullBOS"] = df["Close"] > df["SwingHigh"]
    df["BearBOS"] = df["Close"] < df["SwingLow"]
    df["LongSignal"] = df["BullBOS"] & (df["RSI"] > 50)
    df["ShortSignal"] = df["BearBOS"] & (df["RSI"] < 50)
    return df


def read_trades(status=None):
    with db_connect() as conn:
        if status:
            rows = conn.execute(
                "SELECT * FROM trades WHERE status = ? ORDER BY id DESC", (status,)
            ).fetchall()
        else:
            rows = conn.execute("SELECT * FROM trades ORDER BY id DESC").fetchall()
    return pd.DataFrame([dict(r) for r in rows])


def account_stats():
    settings = get_settings()
    trades = read_trades()
    closed = trades[trades["status"] == "CLOSED"].copy() if not trades.empty else pd.DataFrame()
    realized = float(pd.to_numeric(closed["net_pnl"], errors="coerce").fillna(0).sum()) if not closed.empty else 0.0
    return float(settings["starting_balance"]) + realized, realized, closed


def latest_processed_candle():
    with db_connect() as conn:
        row = conn.execute(
            "SELECT candle_time FROM processed_candles ORDER BY candle_time DESC LIMIT 1"
        ).fetchone()
    return row["candle_time"] if row else None


def mark_candle_processed(candle_time):
    with db_connect() as conn:
        conn.execute(
            "INSERT OR IGNORE INTO processed_candles (candle_time) VALUES (?)",
            (candle_time,),
        )


def has_processed_candle(candle_time):
    with db_connect() as conn:
        return conn.execute(
            "SELECT 1 FROM processed_candles WHERE candle_time = ?", (candle_time,)
        ).fetchone() is not None


def daily_realized_pnl():
    trades = read_trades()
    if trades.empty or "exit_time" not in trades:
        return 0.0
    closed = trades[trades["status"] == "CLOSED"].copy()
    if closed.empty:
        return 0.0
    # All stored timestamps are UTC ISO strings.
    today = datetime.now(timezone.utc).date().isoformat()
    closed = closed[closed["exit_time"].fillna("").str[:10] == today]
    return float(pd.to_numeric(closed["net_pnl"], errors="coerce").fillna(0).sum())


def open_trade(side, entry, stop, candle_time):
    stop_distance = (entry - stop) if side == "LONG" else (stop - entry)
    if stop_distance <= 0 or entry <= 0:
        add_event("SKIP", f"{side} skipped: stop-loss is on the wrong side of entry.")
        return False

    balance, _, _ = account_stats()
    daily_limit = get_settings()["starting_balance"] * DAILY_LOSS_LIMIT_PERCENT / 100.0
    day_pnl = daily_realized_pnl()
    if day_pnl <= -daily_limit:
        add_event("RISK", f"Entry blocked: daily loss limit reached (${day_pnl:.2f}).")
        return False

    risk_cash = balance * RISK_PERCENT / 100.0
    # Estimate both entry and stop exit fees inside the risk-per-trade sizing.
    per_btc_risk_including_fees = stop_distance + entry * FEE_RATE + stop * FEE_RATE
    if per_btc_risk_including_fees <= 0:
        add_event("SKIP", "Entry skipped: invalid risk calculation.")
        return False
    qty = risk_cash / per_btc_risk_including_fees
    qty = min(qty, (balance * 0.95) / entry)
    if qty <= 0:
        add_event("SKIP", "Entry skipped: calculated position size is zero.")
        return False

    take_profit = entry + stop_distance * REWARD_RISK if side == "LONG" else entry - stop_distance * REWARD_RISK
    entry_fee = entry * qty * FEE_RATE
    with db_connect() as conn:
        conn.execute("""
            INSERT INTO trades
            (side, entry_time, entry_price, quantity, stop_loss, take_profit,
             risk_amount, entry_fee, status, strategy)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'OPEN', 'BOS + RSI')
        """, (side, candle_time, entry, qty, stop, take_profit, risk_cash, entry_fee))
    add_event("TRADE", f"PAPER {side} opened at ${entry:,.2f}; SL ${stop:,.2f}; TP ${take_profit:,.2f}; qty {qty:.6f} BTC.")
    return True


def close_trade(trade, exit_price, exit_time, reason):
    side, qty, entry = trade["side"], float(trade["quantity"]), float(trade["entry_price"])
    gross = (exit_price - entry) * qty if side == "LONG" else (entry - exit_price) * qty
    exit_fee = exit_price * qty * FEE_RATE
    net = gross - float(trade["entry_fee"]) - exit_fee
    with db_connect() as conn:
        conn.execute("""
            UPDATE trades
            SET status='CLOSED', exit_time=?, exit_price=?, exit_fee=?,
                gross_pnl=?, net_pnl=?, exit_reason=?
            WHERE id=? AND status='OPEN'
        """, (exit_time, exit_price, exit_fee, gross, net, reason, int(trade["id"])))
    add_event("EXIT", f"Paper trade #{trade['id']} closed ({reason}); net P/L ${net:.2f}.")


def process_open_position(candle):
    open_trades = read_trades("OPEN")
    if open_trades.empty:
        return
    for trade in open_trades.to_dict("records"):
        side, sl, tp = trade["side"], float(trade["stop_loss"]), float(trade["take_profit"])
        exit_price, reason = None, None
        # Conservative assumption: if both levels fall within one candle, SL first.
        if side == "LONG":
            if float(candle["Low"]) <= sl:
                exit_price, reason = sl, "stop-loss"
            elif float(candle["High"]) >= tp:
                exit_price, reason = tp, "take-profit"
        else:
            if float(candle["High"]) >= sl:
                exit_price, reason = sl, "stop-loss"
            elif float(candle["Low"]) <= tp:
                exit_price, reason = tp, "take-profit"
        if exit_price is not None:
            close_trade(trade, exit_price, candle.name.isoformat(), reason)


def signal_explanation(candle):
    if pd.isna(candle.get("SwingHigh")) or pd.isna(candle.get("SwingLow")):
        return "Waiting for enough candles to calculate swing levels."
    rsi = float(candle["RSI"])
    close = float(candle["Close"])
    sh, sl = float(candle["SwingHigh"]), float(candle["SwingLow"])
    reasons = []
    if bool(candle["BullBOS"]) and rsi > 50:
        return "BUY conditions passed: candle closed above prior swing high and RSI is above 50."
    if bool(candle["BearBOS"]) and rsi < 50:
        return "SELL conditions passed: candle closed below prior swing low and RSI is below 50."
    if close <= sh:
        reasons.append(f"No bullish BOS: close ${close:,.2f} is not above swing high ${sh:,.2f}")
    if close >= sl:
        reasons.append(f"No bearish BOS: close ${close:,.2f} is not below swing low ${sl:,.2f}")
    if bool(candle["BullBOS"]) and rsi <= 50:
        reasons.append(f"Bullish breakout but RSI {rsi:.1f} is not above 50")
    if bool(candle["BearBOS"]) and rsi >= 50:
        reasons.append(f"Bearish breakout but RSI {rsi:.1f} is not below 50")
    return "Entry conditions not met: " + "; ".join(reasons[:2]) + "."



def exit_guidance(candle, open_trades, current_price):
    """Return explicit paper-trade exit levels and plain-language exit rules."""
    if not open_trades.empty:
        t = open_trades.iloc[0]
        side = str(t["side"])
        entry = float(t["entry_price"])
        sl = float(t["stop_loss"])
        tp = float(t["take_profit"])
        qty = float(t["quantity"])
        if side == "LONG":
            to_sl = current_price - sl
            to_tp = tp - current_price
            direction = "BUY / LONG"
            sl_rule = f"Exit if a candle reaches ${sl:,.2f} or lower."
            tp_rule = f"Take profit if a candle reaches ${tp:,.2f} or higher."
        else:
            to_sl = sl - current_price
            to_tp = current_price - tp
            direction = "SELL / SHORT"
            sl_rule = f"Exit if a candle reaches ${sl:,.2f} or higher."
            tp_rule = f"Take profit if a candle reaches ${tp:,.2f} or lower."
        return {
            "active": True, "direction": direction, "entry": entry, "sl": sl, "tp": tp,
            "risk_per_btc": abs(entry - sl), "reward_per_btc": abs(tp - entry),
            "to_sl": to_sl, "to_tp": to_tp, "qty": qty,
            "sl_rule": sl_rule, "tp_rule": tp_rule,
            "note": "These are simulated paper levels. The bot checks exit conditions using newly closed candles."
        }

    # Show a potential plan from the most recent closed candle, without placing a trade.
    rsi = float(candle["RSI"]) if pd.notna(candle.get("RSI")) else float("nan")
    close = float(candle["Close"])
    sh = candle.get("SwingHigh")
    sw = candle.get("SwingLow")
    if pd.isna(sh) or pd.isna(sw):
        return {"active": False, "direction": None, "note": "Waiting for enough closed candles to calculate swing-based exit levels."}
    sh, sw = float(sh), float(sw)
    if bool(candle.get("LongSignal", False)) and sw < close:
        entry, sl = close, sw
        tp = entry + (entry - sl) * REWARD_RISK
        return {
            "active": False, "direction": "Potential BUY / LONG setup", "entry": entry, "sl": sl, "tp": tp,
            "risk_per_btc": entry - sl, "reward_per_btc": tp - entry,
            "note": "Potential levels only; no trade is opened by this panel."
        }
    if bool(candle.get("ShortSignal", False)) and sh > close:
        entry, sl = close, sh
        tp = entry - (sl - entry) * REWARD_RISK
        return {
            "active": False, "direction": "Potential SELL / SHORT setup", "entry": entry, "sl": sl, "tp": tp,
            "risk_per_btc": sl - entry, "reward_per_btc": entry - tp,
            "note": "Potential levels only; no trade is opened by this panel."
        }
    return {
        "active": False, "direction": None,
        "note": f"No valid entry on the latest closed candle (RSI {rsi:.1f}). Wait for a confirmed BUY/SELL setup; do not force an entry. Once a trade opens, its exact entry, stop-loss and take-profit will be shown here."
    }

def run_one_cycle(timeframe, swing_lookback, closed_candles=None):
    source = closed_candles if closed_candles is not None else fetch_candles(timeframe)
    df = add_indicators(source, swing_lookback)
    candle = df.iloc[-1]
    candle_time = candle.name.isoformat()
    if has_processed_candle(candle_time):
        return df, "Latest closed candle already processed."
    # Manage open positions only once per newly closed candle.
    process_open_position(candle)
    mark_candle_processed(candle_time)
    if not read_trades("OPEN").empty:
        return df, "An open position exists; waiting for SL/TP."
    reason = signal_explanation(candle)
    if bool(candle["LongSignal"]):
        opened = open_trade("LONG", float(candle["Close"]), float(candle["SwingLow"]), candle_time)
        return df, "BUY paper trade opened." if opened else "BUY signal found, but risk controls blocked entry."
    if bool(candle["ShortSignal"]):
        opened = open_trade("SHORT", float(candle["Close"]), float(candle["SwingHigh"]), candle_time)
        return df, "SELL paper trade opened." if opened else "SELL signal found, but risk controls blocked entry."
    add_event("SIGNAL", f"No entry on {candle_time}; {reason}")
    return df, reason


def make_price_chart(df, trades, timeframe="15m"):
    """TradingView-inspired chart with live candle, time axis, volume, RSI and paper levels."""
    view = df.tail(150).copy()
    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.025,
        row_heights=[0.66, 0.17, 0.17],
        subplot_titles=("BTC/USDT · Candlestick price", "Volume", "RSI (14)")
    )
    up_color, down_color = "#22c55e", "#ef4444"
    fig.add_trace(go.Candlestick(
        x=view.index, open=view["Open"], high=view["High"], low=view["Low"], close=view["Close"],
        name="BTC/USDT", increasing_line_color=up_color, decreasing_line_color=down_color,
        increasing_fillcolor=up_color, decreasing_fillcolor=down_color, whiskerwidth=0.55,
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=view.index, y=view["EMA20"], name="EMA 20",
        line=dict(color="#fbbf24", width=1.5),
        hovertemplate="EMA 20: $%{y:,.2f}<extra></extra>",
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=view.index, y=view["SwingHigh"], name="Prior swing high",
        line=dict(color="#fb923c", dash="dash", width=1),
        hovertemplate="Swing high: $%{y:,.2f}<extra></extra>",
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=view.index, y=view["SwingLow"], name="Prior swing low",
        line=dict(color="#38bdf8", dash="dash", width=1),
        hovertemplate="Swing low: $%{y:,.2f}<extra></extra>",
    ), row=1, col=1)
    longs, shorts = view[view["LongSignal"]], view[view["ShortSignal"]]
    if not longs.empty:
        fig.add_trace(go.Scatter(
            x=longs.index, y=longs["Low"] * 0.9995, mode="markers+text",
            text=["BUY"] * len(longs), textposition="bottom center",
            textfont=dict(color=up_color, size=10),
            marker=dict(symbol="triangle-up", size=12, color=up_color),
            name="BUY signal", hovertemplate="BUY signal<br>%{x}<br>~$%{y:,.2f}<extra></extra>"
        ), row=1, col=1)
    if not shorts.empty:
        fig.add_trace(go.Scatter(
            x=shorts.index, y=shorts["High"] * 1.0005, mode="markers+text",
            text=["SELL"] * len(shorts), textposition="top center",
            textfont=dict(color=down_color, size=10),
            marker=dict(symbol="triangle-down", size=12, color=down_color),
            name="SELL signal", hovertemplate="SELL signal<br>%{x}<br>~$%{y:,.2f}<extra></extra>"
        ), row=1, col=1)
    open_df = trades[trades["status"] == "OPEN"] if not trades.empty else pd.DataFrame()
    if not open_df.empty:
        t = open_df.iloc[0]
        for val, label, color in [
            (float(t["entry_price"]), "Paper entry", "#e5e7eb"),
            (float(t["stop_loss"]), "Stop loss", down_color),
            (float(t["take_profit"]), "Take profit", up_color),
        ]:
            fig.add_hline(y=val, line_dash="dot", line_color=color, line_width=1.4,
                          annotation_text=f"{label}  {val:,.2f}", annotation_position="right", row=1, col=1)
    volume_colors = [up_color if c >= o else down_color for o, c in zip(view["Open"], view["Close"])]
    fig.add_trace(go.Bar(
        x=view.index, y=view["Volume"], name="Volume", marker_color=volume_colors, opacity=0.72,
        hovertemplate="Time: %{x}<br>Volume: %{y:,.4f}<extra></extra>",
    ), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=view.index, y=view["RSI"], name="RSI 14",
        line=dict(color="#c084fc", width=1.7), hovertemplate="RSI 14: %{y:.1f}<extra></extra>",
    ), row=3, col=1)
    for level, color, dash in [(70, down_color, "dash"), (50, "#64748b", "dot"), (30, up_color, "dash")]:
        fig.add_hline(y=level, line_dash=dash, line_color=color, line_width=1, opacity=0.8, row=3, col=1)
    fig.update_yaxes(title_text="USDT", side="right", tickformat=",.0f", row=1, col=1,
                     gridcolor="#202938", zeroline=False)
    fig.update_yaxes(title_text="Volume", side="right", tickformat="~s", row=2, col=1,
                     gridcolor="#202938", zeroline=False)
    fig.update_yaxes(title_text="RSI", side="right", range=[0, 100], row=3, col=1,
                     gridcolor="#202938", zeroline=False)
    # Keep a stable, recent-candle viewport during the 5-second refreshes.
    # The right edge advances only when the exchange starts a new candle,
    # rather than autoscaling the time axis on every refresh.
    seconds_by_tf = {"5m": 300, "15m": 900, "1h": 3600, "4h": 14400}
    interval_seconds = seconds_by_tf.get(timeframe, 900)
    last_candle_time = pd.Timestamp(view.index[-1])
    x_start = last_candle_time - pd.Timedelta(seconds=interval_seconds * 100)
    x_end = last_candle_time + pd.Timedelta(seconds=interval_seconds * 3)
    fixed_time_range = [x_start, x_end]
    fig.update_xaxes(title_text=f"Time (UTC) · {timeframe}", showgrid=True, gridcolor="#202938",
                     tickformat="%d %b<br>%H:%M", rangeslider_visible=False,
                     range=fixed_time_range, autorange=False, row=3, col=1)
    fig.update_xaxes(showgrid=True, gridcolor="#202938", rangeslider_visible=False,
                     range=fixed_time_range, autorange=False, row=1, col=1)
    fig.update_xaxes(showgrid=True, gridcolor="#202938", rangeslider_visible=False,
                     range=fixed_time_range, autorange=False, row=2, col=1)
    fig.update_layout(
        template="plotly_dark", height=790, margin=dict(l=8, r=18, t=62, b=18),
        paper_bgcolor="#0b1220", plot_bgcolor="#0b1220",
        font=dict(family="Inter, Segoe UI, sans-serif", color="#cbd5e1", size=11),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        hovermode="x unified", dragmode="pan", uirevision=f"btc-{timeframe}",
    )
    return fig

def backtest(df, initial_balance, risk_pct, rr, fee_rate, daily_limit_pct):
    """Simple next-bar OHLC simulation; signals use closed candles and no future data."""
    d = add_indicators(df, 5).dropna(subset=["SwingHigh", "SwingLow", "RSI"]).copy()
    balance = initial_balance
    equity_rows = []
    results = []
    position = None
    day_start_balance = balance
    current_day = None
    daily_realized = 0.0
    for i in range(1, len(d)):
        candle = d.iloc[i]
        ts = candle.name
        day = ts.date()
        if current_day != day:
            current_day, day_start_balance, daily_realized = day, balance, 0.0

        if position:
            side = position["side"]
            sl, tp = position["sl"], position["tp"]
            exit_price, reason = None, None
            if side == "LONG":
                if candle["Low"] <= sl:
                    exit_price, reason = sl, "SL"
                elif candle["High"] >= tp:
                    exit_price, reason = tp, "TP"
            else:
                if candle["High"] >= sl:
                    exit_price, reason = sl, "SL"
                elif candle["Low"] <= tp:
                    exit_price, reason = tp, "TP"
            if exit_price is not None:
                qty = position["qty"]
                gross = (exit_price-position["entry"])*qty if side == "LONG" else (position["entry"]-exit_price)*qty
                fees = position["entry"]*qty*fee_rate + exit_price*qty*fee_rate
                net = gross - fees
                balance += net
                daily_realized += net
                results.append({"side": side, "entry_time": position["time"], "exit_time": ts,
                                "entry_price": position["entry"], "exit_price": exit_price,
                                "net_pnl": net, "reason": reason})
                position = None

        if position is None and daily_realized > -(day_start_balance * daily_limit_pct / 100):
            side = "LONG" if bool(candle["LongSignal"]) else ("SHORT" if bool(candle["ShortSignal"]) else None)
            if side:
                entry = float(candle["Close"])
                stop = float(candle["SwingLow"] if side == "LONG" else candle["SwingHigh"])
                distance = entry-stop if side == "LONG" else stop-entry
                if distance > 0:
                    risk_cash = balance * risk_pct / 100
                    qty = min(risk_cash / (distance + entry*fee_rate + stop*fee_rate), balance*0.95/entry)
                    if qty > 0:
                        tp = entry + distance*rr if side == "LONG" else entry-distance*rr
                        position = {"side": side, "entry": entry, "sl": stop, "tp": tp,
                                    "qty": qty, "time": ts}
        equity_rows.append({"time": ts, "equity": balance})
    # Mark-to-market open position at final close for analytics only; label remains open.
    if position:
        last_close = float(d.iloc[-1]["Close"])
        qty = position["qty"]
        gross = (last_close-position["entry"])*qty if position["side"] == "LONG" else (position["entry"]-last_close)*qty
        marked = balance + gross - position["entry"]*qty*fee_rate - last_close*qty*fee_rate
        equity_rows[-1]["equity"] = marked
    trades_df = pd.DataFrame(results)
    equity_df = pd.DataFrame(equity_rows)
    if not equity_df.empty:
        equity_df["peak"] = equity_df["equity"].cummax()
        equity_df["drawdown_pct"] = (equity_df["equity"] / equity_df["peak"] - 1) * 100
    return trades_df, equity_df


st.set_page_config(page_title=APP_TITLE, page_icon="📈", layout="wide", initial_sidebar_state="expanded")
init_db()

st.markdown("""
<style>
.stApp { background: radial-gradient(ellipse at 15% 0%, #13213a 0%, #0b1220 42%, #080d17 100%); }
.block-container { padding-top: 1.1rem; padding-bottom: 2.5rem; max-width: 1800px; }
section[data-testid="stSidebar"] { background: linear-gradient(180deg,#111827 0%,#0b1220 100%); border-right: 1px solid #263449; }
section[data-testid="stSidebar"] .block-container { padding-top: 1.6rem; }
h1, h2, h3 { letter-spacing: -0.025em; }
h1 { font-weight: 800 !important; }
h2 { font-size: 1.35rem !important; font-weight: 750 !important; }
h3 { font-size: 1.05rem !important; font-weight: 700 !important; }
[data-testid="stPlotlyChart"] { background: #0b1220; border: 1px solid #263449; border-radius: 16px; padding: 8px; }
[data-testid="stMetric"] { background: linear-gradient(145deg,#172235,#101827); border: 1px solid #2a3a52; padding: 16px 18px; border-radius: 14px; box-shadow: 0 8px 24px rgba(0,0,0,.14); }
[data-testid="stMetricLabel"] { color: #94a3b8 !important; font-size: .82rem !important; }
[data-testid="stMetricValue"] { color: #f8fafc !important; font-weight: 750; }
div[data-testid="stAlert"] { border-radius: 12px; border: 1px solid #2a3a52; }
.stButton > button { border-radius: 10px; min-height: 2.7rem; font-weight: 650; }
div[data-testid="stDataFrame"] { border: 1px solid #273449; border-radius: 12px; overflow: hidden; }
hr { border-color: #263449; }
.small-muted { color: #94a3b8; font-size: .88rem; }
</style>
""", unsafe_allow_html=True)

hero_left, hero_right = st.columns([3, 1])
with hero_left:
    st.markdown("# ₿  BTC/USDT Trading Terminal")
    st.markdown('<div class="small-muted">LIVE MARKET VIEW · PAPER EXECUTION · RISK MONITORING</div>', unsafe_allow_html=True)
with hero_right:
    st.markdown(" ")
    st.markdown("🟢 **PUBLIC DATA ONLY**")
    st.caption("No API keys · No real orders")
st.divider()

with st.sidebar:
    st.header("Bot controls")
    timeframe = st.selectbox("Trading timeframe", TIMEFRAMES, index=1)
    swing_lookback = st.slider("Swing lookback", 3, 12, 5)
    st.write(f"**Risk per trade:** {RISK_PERCENT:.1f}%")
    st.write(f"**Reward/risk target:** {REWARD_RISK:.1f}R")
    st.write(f"**Estimated fee per side:** {FEE_RATE*100:.2f}%")
    st.write(f"**Daily loss stop:** {DAILY_LOSS_LIMIT_PERCENT:.1f}% of starting balance")
    st.caption(f"Refresh interval: about {POLL_SECONDS} seconds while this page is open.")
    settings = get_settings()
    if not settings["active"]:
        if st.button("▶ Start paper bot", type="primary", use_container_width=True):
            set_active(True)
            st.rerun()
    else:
        st.success("Paper bot is running")
        if st.button("⏸ Stop paper bot", use_container_width=True):
            set_active(False)
            st.rerun()
    with st.expander("Reset virtual account"):
        st.warning("This deletes paper trade history. It does not affect exchange accounts.")
        confirm_reset = st.checkbox("I understand: delete paper history")
        if st.button("Reset to $10,000", disabled=not confirm_reset, use_container_width=True):
            reset_paper_account()
            st.rerun()

settings = get_settings()
balance, realized_pnl, closed_trades = account_stats()
open_trades = read_trades("OPEN")
all_trades = read_trades()
daily_pnl = daily_realized_pnl()

@st.fragment(run_every=POLL_SECONDS)
def live_panel():
    settings_now = get_settings()
    try:
        live_candles = fetch_candles(timeframe, include_live=True)
        chart_data = add_indicators(live_candles, swing_lookback)
        chart_data.loc[chart_data.index[-1], ["LongSignal", "ShortSignal"]] = False
        closed_candles = live_candles.iloc[:-1].copy()
        closed_data = add_indicators(closed_candles, swing_lookback)
        status_message = "Bot stopped."
        if settings_now["active"]:
            _, status_message = run_one_cycle(timeframe, swing_lookback, closed_candles=closed_candles)
        latest_price = float(live_candles["Close"].iloc[-1])
        last = closed_data.iloc[-1]
    except Exception as exc:
        st.error(f"Market-data/strategy error: {type(exc).__name__}: {exc}")
        st.info("The app will retry on the next refresh. Check your internet connection.")
        return

    current_balance, current_realized, current_closed = account_stats()
    open_now = read_trades("OPEN")
    trades_now = read_trades()
    day_pnl = daily_realized_pnl()
    unrealized = 0.0
    if not open_now.empty:
        for t in open_now.to_dict("records"):
            qty, entry = float(t["quantity"]), float(t["entry_price"])
            gross = (latest_price-entry)*qty if t["side"] == "LONG" else (entry-latest_price)*qty
            unrealized += gross - float(t["entry_fee"]) - latest_price*qty*FEE_RATE

    forming_time = live_candles.index[-1]
    st.caption(f"Status: {'🟢 RUNNING' if settings_now['active'] else '⚪ STOPPED'}  ·  {status_message}  ·  Last refresh UTC: {utc_now()}")
    head1, head2, head3 = st.columns([2, 1, 1])
    head1.metric("BTC/USDT · Live candle close", f"${latest_price:,.2f}")
    head2.metric("Chart timeframe", timeframe)
    head3.metric("Bot status", "RUNNING" if settings_now["active"] else "STOPPED")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Virtual balance", f"${current_balance:,.2f}")
    c2.metric("Realized P/L", f"${current_realized:,.2f}", delta=f"{current_realized:+,.2f}")
    c3.metric("Open P/L (estimated)", f"${unrealized:,.2f}", delta=f"{unrealized:+,.2f}")
    c4.metric("BTC price", f"${latest_price:,.2f}")

    c5, c6, c7 = st.columns(3)
    c5.metric("Today's realized P/L", f"${day_pnl:,.2f}")
    c6.metric("Closed trades", str(len(current_closed)))
    wins = int((pd.to_numeric(current_closed["net_pnl"], errors="coerce") > 0).sum()) if not current_closed.empty else 0
    c7.metric("Win rate", f"{(wins/len(current_closed)*100):.1f}%" if len(current_closed) else "—")

    st.subheader("Market chart")
    chart_col, timer_col = st.columns([5, 1], gap="small")
    with chart_col:
        st.plotly_chart(
            make_price_chart(chart_data, trades_now, timeframe),
            use_container_width=True, key="live_price_chart",
            config={"displaylogo": False, "scrollZoom": True,
                    "modeBarButtonsToAdd": ["drawline", "eraseshape"],
                    "toImageButtonOptions": {"format": "png", "filename": "btc_usdt_chart", "scale": 2}}
        )
    with timer_col:
        st.markdown("### Candle")
        st.metric("Closes in", candle_close_countdown(timeframe))
        st.caption(f"Timeframe: {timeframe}")
        st.caption(f"UTC start: {forming_time.strftime('%H:%M:%S')}")
        st.markdown("---")
        st.metric("Live price", f"${latest_price:,.2f}")
        st.caption("Refreshes about every 5 seconds.")

    st.subheader("Signal diagnosis · closed candle only")
    st.info(signal_explanation(last))
    st.caption(f"Last CLOSED candle: {closed_data.index[-1]} · RSI(14): {float(last['RSI']):.1f} · EMA20: ${float(last['EMA20']):,.2f} · Swing high: ${float(last['SwingHigh']):,.2f} · Swing low: ${float(last['SwingLow']):,.2f}")

    st.subheader("Exit planner · Entry / Stop-loss / Take-profit")
    exit_plan = exit_guidance(last, open_now, latest_price)
    if exit_plan.get("direction"):
        st.markdown(f"**{exit_plan['direction']}**")
    if all(k in exit_plan for k in ("entry", "sl", "tp")):
        e1, e2, e3 = st.columns(3)
        e1.metric("Planned entry", f"${exit_plan['entry']:,.2f}")
        e2.metric("Stop-loss (SL)", f"${exit_plan['sl']:,.2f}", delta=f"-${abs(exit_plan['entry']-exit_plan['sl']):,.2f}", delta_color="inverse")
        e3.metric(f"Take-profit (TP · {REWARD_RISK:.1f}R)", f"${exit_plan['tp']:,.2f}", delta=f"+${abs(exit_plan['tp']-exit_plan['entry']):,.2f}")
        if exit_plan.get("active"):
            st.warning("**Exit rules:** " + exit_plan["sl_rule"] + " " + exit_plan["tp_rule"])
            st.caption(f"Current price: ${latest_price:,.2f} · Estimated distance to SL: ${abs(exit_plan.get('to_sl', 0)):,.2f} per BTC · Distance to TP: ${abs(exit_plan.get('to_tp', 0)):,.2f} per BTC.")
        else:
            st.info("These are potential setup levels only, not an active position.")
    else:
        st.info(exit_plan["note"])
    st.caption(exit_plan["note"])
    if day_pnl <= -(get_settings()["starting_balance"] * DAILY_LOSS_LIMIT_PERCENT / 100):
        st.error("Daily loss limit reached. New paper entries are blocked until the next UTC day.")

    st.subheader("Open position")
    if open_now.empty:
        st.write("No open paper position.")
    else:
        display_cols = ["id", "side", "entry_time", "entry_price", "quantity", "stop_loss", "take_profit", "risk_amount"]
        st.dataframe(open_now[[c for c in display_cols if c in open_now.columns]].rename(
            columns={"entry_price": "Entry price", "stop_loss": "EXIT: Stop-loss", "take_profit": "EXIT: Take-profit",
                     "side": "Direction", "quantity": "BTC size", "risk_amount": "Risk ($)"}),
            use_container_width=True, hide_index=True)

    st.subheader("Trade history")
    if trades_now.empty:
        st.write("No trades yet. The strategy waits for a valid closed-candle BOS + RSI signal.")
    else:
        cols = ["id", "side", "entry_time", "entry_price", "stop_loss", "take_profit", "status", "exit_time", "exit_price", "net_pnl"]
        st.dataframe(trades_now[[c for c in cols if c in trades_now.columns]].head(50), use_container_width=True, hide_index=True)

    st.subheader("Recent bot events")
    with db_connect() as conn:
        rows = conn.execute("SELECT time, kind, message FROM events ORDER BY id DESC LIMIT 20").fetchall()
    if rows:
        st.dataframe(pd.DataFrame([dict(r) for r in rows]), use_container_width=True, hide_index=True)

live_panel()

st.divider()
st.header("Historical backtest & analytics")
st.caption("Separate from live paper trading. Backtest uses downloaded historical candles and simulated fills; it does not place orders or alter your saved paper-trade history.")
with st.expander("Run a historical backtest", expanded=True):
    bt_col1, bt_col2 = st.columns([1, 2])
    with bt_col1:
        bt_timeframe = st.selectbox("Backtest timeframe", TIMEFRAMES, index=1, key="bt_tf")
        bt_limit = st.slider("Historical candles", min_value=200, max_value=1000, value=500, step=100)
        bt_balance = st.number_input("Backtest starting balance ($)", min_value=100.0, max_value=1_000_000.0, value=10_000.0, step=500.0)
        bt_risk = st.slider("Backtest risk per trade (%)", 0.25, 2.0, 1.0, 0.25)
        bt_rr = st.slider("Backtest reward/risk", 1.0, 4.0, 2.0, 0.5)
        run_bt = st.button("▶ Run backtest", type="primary", use_container_width=True)
    with bt_col2:
        st.write("Assumptions")
        st.write("- Signal checked on fully closed candles.")
        st.write("- Stop-loss is assumed hit first if SL and TP are both inside one candle.")
        st.write("- Estimated fees are charged on entry and exit.")
        st.write("- This is a simplified simulation; gaps, latency and slippage may change results.")

    if run_bt:
        try:
            with st.spinner("Downloading historical candles and simulating trades..."):
                hist = fetch_candles(bt_timeframe, limit=bt_limit)
                bt_trades, equity = backtest(hist, bt_balance, bt_risk, bt_rr, FEE_RATE, DAILY_LOSS_LIMIT_PERCENT)
            n = len(bt_trades)
            wins_bt = int((bt_trades["net_pnl"] > 0).sum()) if n else 0
            net_bt = float(bt_trades["net_pnl"].sum()) if n else 0.0
            final_equity = float(equity["equity"].iloc[-1]) if not equity.empty else bt_balance
            max_dd = float(equity["drawdown_pct"].min()) if not equity.empty else 0.0
            gross_wins = float(bt_trades.loc[bt_trades["net_pnl"] > 0, "net_pnl"].sum()) if n else 0.0
            gross_losses = abs(float(bt_trades.loc[bt_trades["net_pnl"] < 0, "net_pnl"].sum())) if n else 0.0
            pf = gross_wins / gross_losses if gross_losses > 0 else (float("inf") if gross_wins > 0 else 0.0)
            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("Backtest trades", str(n))
            m2.metric("Win rate", f"{wins_bt/n*100:.1f}%" if n else "—")
            m3.metric("Net P/L", f"${net_bt:,.2f}")
            m4.metric("Profit factor", f"{pf:.2f}" if pf != float("inf") else "∞")
            m5.metric("Max drawdown", f"{max_dd:.2f}%")
            eqfig = go.Figure()
            eqfig.add_trace(go.Scatter(x=equity["time"], y=equity["equity"], mode="lines", name="Equity", line=dict(color="#60a5fa", width=2)))
            eqfig.update_layout(template="plotly_dark", title="Backtest equity curve", height=330,
                                paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
                                xaxis_title="Time", yaxis_title="Equity ($)")
            st.plotly_chart(eqfig, use_container_width=True)
            if bt_trades.empty:
                st.warning("No trades triggered in this historical window. That can happen with strict breakout conditions; it is not proof of a code error.")
            else:
                st.dataframe(bt_trades, use_container_width=True, hide_index=True)
            st.caption(f"Backtest ending equity (open trade marked to market if any): ${final_equity:,.2f}. Past simulation results do not predict future performance.")
        except Exception as exc:
            st.error(f"Backtest failed: {type(exc).__name__}: {exc}")

st.caption("Risk note: This is a TradingView-inspired chart, not the TradingView website or its licensed data feed. Candles update from public exchange OHLCV data about every 5 seconds; signals still wait for closed candles. Paper fills are approximate. No real exchange orders are sent.")
