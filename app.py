import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from plotly.subplots import make_subplots
import plotly.graph_objects as go

# ML imports
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Elliott Wave + ML — Multi-Timeframe", layout="wide")
st.title("📈 Elliott Wave 0 & 5 + ML Buy/Sell (1H / 1D / 1W)")
st.caption("Strict Elliott + Structural Swing + Fibonacci + Trend Filters + Long-Term Context + ML Signals")


# ---------------- DATA LOADER ----------------
@st.cache_data(ttl=3600)
def load_data(ticker, period, interval):
    df = yf.download(ticker, period=period, interval=interval, progress=False)
    if df is None or df.empty:
        return pd.DataFrame()

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.dropna().copy()
    df.index = pd.to_datetime(df.index)
    return df


# ---------------- INDICATORS ----------------
def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # SMAs (added 22 as per your rule)
    for win in [20, 22, 50, 200]:
        df[f"SMA_{win}"] = df["Close"].rolling(win).mean()

    # RSI(14)
    delta = df["Close"].diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)

    roll_up = gain.rolling(14).mean()
    roll_down = loss.rolling(14).mean()

    rs = roll_up / roll_down.replace(0, np.nan)
    df["RSI_14"] = 100 - (100 / (1 + rs))

    return df


# ---------------- TIMEFRAME PARAMETERS ----------------
def get_params(tf: str) -> dict:
    if tf == "1h":
        return dict(pivot_k=3, swing_window=25, fib_window=80, future_n=7)
    elif tf == "1wk":
        return dict(pivot_k=2, swing_window=10, fib_window=30, future_n=3)
    else:  # daily default
        return dict(pivot_k=5, swing_window=20, fib_window=60, future_n=5)


def get_longterm_windows(timeframe: str):
    """
    Long-term lookback windows (in candles) for different timeframes.
    L1 = medium-term, L2 = very long-term.
    """
    if timeframe == "1h":
        # 1h candles, last 60 days:
        # L1 ≈ 1 week, L2 ≈ ~1 month of trading hours (approx)
        return 24 * 5, 24 * 20          # 120, 480
    elif timeframe == "1d":
        # Daily: ~252 trading days per year
        return 252, 252 * 3             # 1-year, 3-year
    elif timeframe == "1wk":
        # Weekly: 52 weeks per year
        return 52, 52 * 5               # 1-year, 5-year
    else:
        # Default: treat as daily
        return 252, 252 * 3


# ---------------- PIVOTS ----------------
def pivot_lows(df: pd.DataFrame, k: int) -> np.ndarray:
    lows = df["Low"].values
    n = len(df)
    piv = np.zeros(n, dtype=bool)
    for i in range(k, n - k):
        if lows[i] == lows[i - k:i + k + 1].min():
            piv[i] = True
    return piv


def pivot_highs(df: pd.DataFrame, k: int) -> np.ndarray:
    highs = df["High"].values
    n = len(df)
    piv = np.zeros(n, dtype=bool)
    for i in range(k, n - k):
        if highs[i] == highs[i - k:i + k + 1].max():
            piv[i] = True
    return piv


# ----------------------------------------------------------
# ----------------- WAVE 0 RULES (A, B, C) -----------------
# ----------------------------------------------------------
def rule_A0(df: pd.DataFrame, piv: np.ndarray, params: dict) -> np.ndarray:
    """Strict-ish Elliott style: RSI bottom + reversal + future up."""
    n = len(df)
    A0 = np.zeros(n, dtype=bool)

    rsi = df["RSI_14"].values
    close = df["Close"].values
    future_n = params["future_n"]

    for i in range(2, n - future_n - 5):
        if not piv[i]:
            continue

        if np.isnan(rsi[i]) or np.isnan(rsi[i - 1]):
            continue

        # RSI oversold-ish AND turning up
        if rsi[i] > 40 or rsi[i] <= rsi[i - 1]:
            continue

        # Price turning up
        if close[i] <= close[i - 1]:
            continue

        # Future confirmation: after N bars, price higher than 0
        if close[i + future_n] <= close[i]:
            continue

        A0[i] = True

    return A0


def rule_B0(df: pd.DataFrame, piv: np.ndarray, params: dict) -> np.ndarray:
    """Pure swing low: lowest in recent window + future price up."""
    n = len(df)
    B0 = np.zeros(n, dtype=bool)

    low = df["Low"].values
    close = df["Close"].values
    swing = params["swing_window"]
    future_n = params["future_n"]

    for i in range(n - future_n):
        if not piv[i]:
            continue

        start = max(0, i - swing)
        if low[i] != low[start:i + 1].min():
            continue

        if close[i + future_n] <= close[i]:
            continue

        B0[i] = True

    return B0


def rule_C0(df: pd.DataFrame, piv: np.ndarray, params: dict) -> np.ndarray:
    """Fibonacci retracement: 0 near 61.8–100% retracement of prior move."""
    n = len(df)
    C0 = np.zeros(n, dtype=bool)

    low = df["Low"].values
    fib_win = params["fib_window"]

    for i in range(n):
        if not piv[i]:
            continue

        start = max(0, i - fib_win)
        if start >= i:
            continue

        prev_high = df["High"].iloc[start:i].max()
        if prev_high <= 0:
            continue

        retr = (prev_high - low[i]) / prev_high  # % drop from prev high

        if 0.58 <= retr <= 1.05:  # 58%–105% zone
            C0[i] = True

    return C0


# ----------------------------------------------------------
# ----------------- FINAL WAVE 0 (A OR B OR C) --------------
# ----------------------------------------------------------
def add_wave0(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    df = df.copy()
    params = get_params(timeframe)

    piv = pivot_lows(df, params["pivot_k"])

    A0 = rule_A0(df, piv, params)
    B0 = rule_B0(df, piv, params)
    C0 = rule_C0(df, piv, params)

    combined = A0 | B0 | C0

    low = df["Low"].values
    close = df["Close"].values
    n = len(df)

    # 1) Cluster filter: keep LOWEST low in each neighborhood
    cluster_mask = np.zeros(n, dtype=bool)
    last_idx = None
    last_low = None

    for idx in np.where(combined)[0]:
        if last_idx is None:
            cluster_mask[idx] = True
            last_idx = idx
            last_low = low[idx]
        else:
            if idx - last_idx < params["swing_window"]:
                # same cluster — keep lower low
                if low[idx] < last_low:
                    cluster_mask[last_idx] = False
                    cluster_mask[idx] = True
                    last_idx = idx
                    last_low = low[idx]
            else:
                cluster_mask[idx] = True
                last_idx = idx
                last_low = low[idx]

    # 2) Strong trend filter:
    # After 0 → no lower low AND final close higher than 0's close
    protect_n = 2 * params["swing_window"]
    final = np.zeros(n, dtype=bool)
    candidate_idxs = np.where(cluster_mask)[0]

    for idx in candidate_idxs:
        start_f = idx + 1
        end_f = min(n, idx + 1 + protect_n)

        if start_f >= end_f:
            # not enough future bars, we keep it
            final[idx] = True
            continue

        future_lows = low[start_f:end_f]
        future_closes = close[start_f:end_f]

        # If any lower low occurs → discard
        if future_lows.min() < low[idx]:
            continue

        # Require net up move: last close in window > close at 0
        if future_closes[-1] <= close[idx]:
            continue

        final[idx] = True

    df["Wave0"] = final
    return df


# ----------------------------------------------------------
# ----------------- WAVE 5 RULES (A, B, C) -----------------
# ----------------------------------------------------------
def rule_A5(df: pd.DataFrame, piv: np.ndarray, params: dict) -> np.ndarray:
    """Strict-ish Elliott style top: RSI top + reversal + future down."""
    n = len(df)
    A5 = np.zeros(n, dtype=bool)

    rsi = df["RSI_14"].values
    close = df["Close"].values
    future_n = params["future_n"]

    for i in range(2, n - future_n - 5):
        if not piv[i]:
            continue

        if np.isnan(rsi[i]) or np.isnan(rsi[i - 1]):
            continue

        # RSI overbought-ish AND turning down
        if rsi[i] < 60 or rsi[i] >= rsi[i - 1]:
            continue

        # Price turning down
        if close[i] >= close[i - 1]:
            continue

        # Future confirmation: after N bars, price below 5
        if close[i + future_n] >= close[i]:
            continue

        A5[i] = True

    return A5


def rule_B5(df: pd.DataFrame, piv: np.ndarray, params: dict) -> np.ndarray:
    """Pure swing high: highest in recent window + future price down."""
    n = len(df)
    B5 = np.zeros(n, dtype=bool)

    high = df["High"].values
    close = df["Close"].values
    swing = params["swing_window"]
    future_n = params["future_n"]

    for i in range(n - future_n):
        if not piv[i]:
            continue

        start = max(0, i - swing)
        if high[i] != high[start:i + 1].max():
            continue

        if close[i + future_n] >= close[i]:
            continue

        B5[i] = True

    return B5


def rule_C5(df: pd.DataFrame, piv: np.ndarray, params: dict) -> np.ndarray:
    """Fibonacci extension: 5 near 1.0–2.0 extension of prior swing."""
    n = len(df)
    C5 = np.zeros(n, dtype=bool)

    high = df["High"].values
    low = df["Low"].values
    fib_win = params["fib_window"]

    for i in range(n):
        if not piv[i]:
            continue

        start = max(0, i - fib_win)
        if start >= i:
            continue

        prev_low = low[start:i].min()
        if prev_low <= 0:
            continue

        ext = (high[i] - prev_low) / prev_low  # % move from swing low

        # rough terminal extension band
        if 0.95 <= ext <= 2.05:
            C5[i] = True

    return C5


# ----------------------------------------------------------
# ----------------- FINAL WAVE 5 (A OR B OR C) --------------
# ----------------------------------------------------------
def add_wave5(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    df = df.copy()
    params = get_params(timeframe)

    piv = pivot_highs(df, params["pivot_k"])

    A5 = rule_A5(df, piv, params)
    B5 = rule_B5(df, piv, params)
    C5 = rule_C5(df, piv, params)

    combined = A5 | B5 | C5

    high = df["High"].values
    close = df["Close"].values
    n = len(df)

    # 1) Cluster filter: keep HIGHEST high in each neighborhood
    cluster_mask = np.zeros(n, dtype=bool)
    last_idx = None
    last_high = None

    for idx in np.where(combined)[0]:
        if last_idx is None:
            cluster_mask[idx] = True
            last_idx = idx
            last_high = high[idx]
        else:
            if idx - last_idx < params["swing_window"]:
                # same cluster — keep higher high
                if high[idx] > last_high:
                    cluster_mask[last_idx] = False
                    cluster_mask[idx] = True
                    last_idx = idx
                    last_high = high[idx]
            else:
                cluster_mask[idx] = True
                last_idx = idx
                last_high = high[idx]

    # 2) Strong trend filter:
    # After 5 → no higher high AND final close lower than 5's close
    protect_n = 2 * params["swing_window"]
    final = np.zeros(n, dtype=bool)
    candidate_idxs = np.where(cluster_mask)[0]

    for idx in candidate_idxs:
        start_f = idx + 1
        end_f = min(n, idx + 1 + protect_n)

        if start_f >= end_f:
            # not enough future bars, we keep it
            final[idx] = True
            continue

        future_highs = high[start_f:end_f]
        future_closes = close[start_f:end_f]

        # If any higher high occurs → discard
        if future_highs.max() > high[idx]:
            continue

        # Require net down move: last close in window < close at 5
        if future_closes[-1] >= close[idx]:
            continue

        final[idx] = True

    df["Wave5"] = final
    return df


# ----------------------------------------------------------
# ----------------- CHART PLOTTING -------------------------
# ----------------------------------------------------------
def plot_chart(df: pd.DataFrame, title: str):
    if df is None or df.empty:
        return go.Figure()

    x = df.index

    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.7, 0.15, 0.15],
        vertical_spacing=0.03,
        specs=[
            [{"type": "candlestick"}],
            [{"type": "bar"}],
            [{"type": "scatter"}],
        ],
    )

    # Price
    fig.add_trace(
        go.Candlestick(
            x=x,
            open=df["Open"],
            high=df["High"],
            low=df["Low"],
            close=df["Close"],
            name="Price",
        ),
        row=1,
        col=1,
    )

    # SMAs (still plotting 20/50/200 for clarity)
    for s in [20, 50, 200]:
        col = f"SMA_{s}"
        if col in df.columns:
            fig.add_trace(
                go.Scatter(x=x, y=df[col], mode="lines", name=f"SMA {s}"),
                row=1,
                col=1,
            )

    # Wave 0
    if "Wave0" in df.columns:
        w0 = df[df["Wave0"]]
        if not w0.empty:
            fig.add_trace(
                go.Scatter(
                    x=w0.index,
                    y=w0["Low"] * 0.995,
                    mode="text",
                    text=["0"] * len(w0),
                    textposition="middle center",
                    name="Wave 0",
                ),
                row=1,
                col=1,
            )

    # Wave 5
    if "Wave5" in df.columns:
        w5 = df[df["Wave5"]]
        if not w5.empty:
            fig.add_trace(
                go.Scatter(
                    x=w5.index,
                    y=w5["High"] * 1.005,
                    mode="text",
                    text=["5"] * len(w5),
                    textposition="middle center",
                    name="Wave 5",
                ),
                row=1,
                col=1,
            )

    # Volume
    if "Volume" in df.columns:
        fig.add_trace(
            go.Bar(x=x, y=df["Volume"], name="Volume"),
            row=2,
            col=1,
        )

    # RSI
    if "RSI_14" in df.columns:
        fig.add_trace(
            go.Scatter(x=x, y=df["RSI_14"], mode="lines", name="RSI 14"),
            row=3,
            col=1,
        )
        fig.add_hrect(
            y0=30,
            y1=70,
            fillcolor="lightgray",
            opacity=0.2,
            line_width=0,
            row=3,
            col=1,
        )

    fig.update_layout(
        title=title,
        height=900,
        xaxis_rangeslider_visible=False,
        hovermode="x unified",
        margin=dict(l=10, r=10, t=40, b=10),
    )

    return fig


# ----------------------------------------------------------
# ------------- RULE-BASED LABEL (YOUR LOGIC) --------------
# ----------------------------------------------------------
def rule_based_label(row, timeframe: str) -> int:
    """
    Your rules:

    BUY (2):
      - price < SMA22 < SMA50 < SMA200 (we approximate via Dist_SMA_x < 0)
      - RSI oversold
      - bullish divergence
      - price near good support
      - Elliott phase ~ 0/2/4 (use Wave0 or bullish divergence)

    SELL (0):
      - price > SMA22 > SMA50 > SMA200
      - RSI overbought
      - bearish divergence
      - price near resistance
      - Elliott phase ~ 1/3/5 (use Wave5 or bearish divergence)

    HOLD (1) otherwise.
    """

    rsi = row["RSI_14"]
    d22 = row["Dist_SMA_22"]
    d50 = row["Dist_SMA_50"]
    d200 = row["Dist_SMA_200"]

    wave0 = row["Wave0_Flag"]
    wave5 = row["Wave5_Flag"]
    bull_div = bool(row["BullDiv"])
    bear_div = bool(row["BearDiv"])

    dist_l1_low = row["Dist_L1_Low"]
    dist_l1_high = row["Dist_L1_High"]

    # ---------- BUY RULES ----------
    # Price below all SMAs (approx for price < SMA22 < SMA50 < SMA200)
    buy_price_ma = (d22 < 0) and (d50 < 0) and (d200 < 0)

    # RSI oversold-ish
    buy_rsi = (rsi <= 35)

    # Good support: close near long-term low (0% to +10% above L1 low)
    support_ok = (0.0 <= dist_l1_low <= 0.10)

    # Elliott phase ~ 0/2/4
    elliott_buy_ok = (wave0 == 1) or (bull_div and wave5 == 0)

    if buy_price_ma and buy_rsi and bull_div and support_ok and elliott_buy_ok:
        return 2  # BUY

    # ---------- SELL RULES ----------
    # Price above all SMAs (approx for price > SMA22 > SMA50 > SMA200)
    sell_price_ma = (d22 > 0) and (d50 > 0) and (d200 > 0)

    # RSI overbought-ish
    sell_rsi = (rsi >= 65)

    # Near resistance: close near long-term high (-5% to 0% below L1 high)
    resistance_ok = (-0.05 <= dist_l1_high <= 0.0)

    # Elliott phase ~ 1/3/5
    elliott_sell_ok = (wave5 == 1) or (bear_div and wave0 == 0)

    if sell_price_ma and sell_rsi and bear_div and resistance_ok and elliott_sell_ok:
        return 0  # SELL

    # ---------- OTHERWISE HOLD ----------
    return 1  # HOLD / NEUTRAL


# ----------------------------------------------------------
# ----------------- ML FEATURE BUILDER ----------------------
# ----------------------------------------------------------
def build_ml_features(df: pd.DataFrame, timeframe: str, ticker: str) -> pd.DataFrame:
    """
    Build ML-ready features + YOUR rule-based label for one ticker & timeframe.
    Label:
        0 = Sell (your conditions)
        1 = Hold
        2 = Buy  (your conditions)
    """
    df = df.copy()

    # ---------- Compute divergences using pivots ----------
    params = get_params(timeframe)
    k = params["pivot_k"]

    n = len(df)
    rsi = df["RSI_14"].values
    close = df["Close"].values

    piv_l = pivot_lows(df, k)
    piv_h = pivot_highs(df, k)

    bull_div = np.zeros(n, dtype=bool)
    bear_div = np.zeros(n, dtype=bool)

    # Bullish divergence: price lower low, RSI higher low
    last_low_idx = None
    for i in range(n):
        if piv_l[i] and not np.isnan(rsi[i]):
            if last_low_idx is not None and not np.isnan(rsi[last_low_idx]):
                if close[i] < close[last_low_idx] and rsi[i] > rsi[last_low_idx]:
                    bull_div[i] = True
            last_low_idx = i

    # Bearish divergence: price higher high, RSI lower high
    last_high_idx = None
    for i in range(n):
        if piv_h[i] and not np.isnan(rsi[i]):
            if last_high_idx is not None and not np.isnan(rsi[last_high_idx]):
                if close[i] > close[last_high_idx] and rsi[i] < rsi[last_high_idx]:
                    bear_div[i] = True
            last_high_idx = i

    # ---------- Basic feature frame ----------
    feat = pd.DataFrame(index=df.index)
    feat["Ticker"] = ticker
    feat["Close"] = df["Close"]
    feat["RSI_14"] = df["RSI_14"]

    # Distances to SMA (22, 50, 200)
    for s in [22, 50, 200]:
        col = f"SMA_{s}"
        if col in df.columns:
            feat[f"Dist_SMA_{s}"] = (df["Close"] - df[col]) / df[col]
        else:
            feat[f"Dist_SMA_{s}"] = np.nan

    # Short-term returns as extra features
    feat["Ret_3"] = df["Close"].pct_change(3)
    feat["Ret_5"] = df["Close"].pct_change(5)
    feat["Ret_10"] = df["Close"].pct_change(10)

    # Elliott info (Wave0 / Wave5 flags)
    feat["Wave0_Flag"] = df.get("Wave0", pd.Series(False, index=df.index)).astype(int)
    feat["Wave5_Flag"] = df.get("Wave5", pd.Series(False, index=df.index)).astype(int)

    # Divergence flags
    feat["BullDiv"] = bull_div
    feat["BearDiv"] = bear_div

    # ---------- Long-term high/low features ----------
    L1, L2 = get_longterm_windows(timeframe)
    close_ser = df["Close"]

    high_L1 = close_ser.rolling(L1, min_periods=1).max()
    low_L1 = close_ser.rolling(L1, min_periods=1).min()
    high_L2 = close_ser.rolling(L2, min_periods=1).max()
    low_L2 = close_ser.rolling(L2, min_periods=1).min()

    feat["Dist_L1_High"] = (close_ser - high_L1) / high_L1.replace(0, np.nan)
    feat["Dist_L1_Low"] = (close_ser - low_L1) / low_L1.replace(0, np.nan)
    feat["Dist_L2_High"] = (close_ser - high_L2) / high_L2.replace(0, np.nan)
    feat["Dist_L2_Low"] = (close_ser - low_L2) / low_L2.replace(0, np.nan)

    # ---------- Apply your rule-based labels ----------
    # Drop rows where core indicators missing
    feat = feat.dropna(subset=["RSI_14", "Close", "Dist_SMA_22", "Dist_SMA_50", "Dist_SMA_200"])

    feat["Label"] = feat.apply(lambda r: rule_based_label(r, timeframe), axis=1)
    feat["Label"] = feat["Label"].astype(int)

    # Final cleanup
    feat = feat.dropna()

    return feat


def train_ml_model(df_all: pd.DataFrame, feature_cols: list):
    """
    Train RandomForest model on all rows in df_all.
    """
    if df_all.empty:
        return None, None

    X = df_all[feature_cols].values
    y = df_all["Label"].astype(int).values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, shuffle=False
    )

    clf = RandomForestClassifier(
        n_estimators=400,
        max_depth=None,
        min_samples_leaf=5,
        class_weight=None,
        random_state=42,
    )
    clf.fit(X_train, y_train)

    return clf, df_all


def predict_for_latest(df: pd.DataFrame, clf, feature_cols: list, timeframe: str):
    """
    Take last row of df, build feature vector, return probas for classes 0,1,2.
    """
    if df.empty or clf is None:
        return None

    df = df.copy()
    last = df.iloc[-1:]

    row = {}
    row["Close"] = last["Close"].iloc[0]
    row["RSI_14"] = last["RSI_14"].iloc[0]

    # Distances to SMAs (22, 50, 200)
    for s in [22, 50, 200]:
        col = f"SMA_{s}"
        if col in df.columns:
            sma_val = last[col].iloc[0]
            if not pd.isna(sma_val) and sma_val != 0:
                row[f"Dist_SMA_{s}"] = (last["Close"].iloc[0] - sma_val) / sma_val
            else:
                row[f"Dist_SMA_{s}"] = 0.0
        else:
            row[f"Dist_SMA_{s}"] = 0.0

    # Past returns
    row["Ret_3"] = df["Close"].pct_change(3).iloc[-1]
    row["Ret_5"] = df["Close"].pct_change(5).iloc[-1]
    row["Ret_10"] = df["Close"].pct_change(10).iloc[-1]

    # Elliott flags
    row["Wave0_Flag"] = int(last.get("Wave0", pd.Series([False])).iloc[0])
    row["Wave5_Flag"] = int(last.get("Wave5", pd.Series([False])).iloc[0])

    # Long-term highs/lows for last bar
    L1, L2 = get_longterm_windows(timeframe)
    close_series = df["Close"]

    high_L1 = close_series.rolling(L1, min_periods=1).max().iloc[-1]
    low_L1 = close_series.rolling(L1, min_periods=1).min().iloc[-1]
    high_L2 = close_series.rolling(L2, min_periods=1).max().iloc[-1]
    low_L2 = close_series.rolling(L2, min_periods=1).min().iloc[-1]

    c_last = last["Close"].iloc[0]

    def safe_dist(c, ref):
        if pd.isna(ref) or ref == 0:
            return 0.0
        return (c - ref) / ref

    row["Dist_L1_High"] = safe_dist(c_last, high_L1)
    row["Dist_L1_Low"] = safe_dist(c_last, low_L1)
    row["Dist_L2_High"] = safe_dist(c_last, high_L2)
    row["Dist_L2_Low"] = safe_dist(c_last, low_L2)

    # For prediction we don't recompute divergence; model learned pattern from other features
    # BullDiv / BearDiv are not in feature_cols, so no need here.

    # Replace NaNs with 0
    for k, v in row.items():
        if pd.isna(v):
            row[k] = 0.0

    X = np.array([[row[c] for c in feature_cols]])
    proba = clf.predict_proba(X)[0]  # [P(Sell), P(Hold), P(Buy)]
    return proba


# ----------------------------------------------------------
# --------------------------- UI ---------------------------
# ----------------------------------------------------------
st.sidebar.header("Settings")

default_tickers = [
     "360ONE.NS","3MINDIA.NS","ABB.NS","TIPSMUSIC.NS","ACC.NS","ACMESOLAR.NS","AIAENG.NS","APLAPOLLO.NS","AUBANK.NS","AWL.NS","AADHARHFC.NS",
    "AARTIIND.NS","AAVAS.NS","ABBOTINDIA.NS","ACE.NS","ADANIENSOL.NS","ADANIENT.NS","ADANIGREEN.NS","ADANIPORTS.NS","ADANIPOWER.NS","ATGL.NS",
    "ABCAPITAL.NS","ABFRL.NS",
]

ticker = st.sidebar.selectbox("Select Symbol", default_tickers)
custom = st.sidebar.text_input("Or Custom Symbol (Yahoo code)")

if custom.strip():
    ticker = custom.strip()

tabs = st.tabs(["⏱ 1H", "📅 Daily", "📆 Weekly", "🤖 ML Buy/Sell Signals"])


# 1H
with tabs[0]:
    df_h = load_data(ticker, "60d", "1h")
    if df_h.empty:
        st.warning("No hourly data.")
    else:
        df_h = add_indicators(df_h)
        df_h = add_wave0(df_h, "1h")
        df_h = add_wave5(df_h, "1h")
        fig_h = plot_chart(df_h, f"{ticker} — 1H")
        st.plotly_chart(fig_h, use_container_width=True)

# Daily
with tabs[1]:
    df_d = load_data(ticker, "3y", "1d")
    if df_d.empty:
        st.warning("No daily data.")
    else:
        df_d = add_indicators(df_d)
        df_d = add_wave0(df_d, "1d")
        df_d = add_wave5(df_d, "1d")
        fig_d = plot_chart(df_d, f"{ticker} — Daily (3Y)")
        st.plotly_chart(fig_d, use_container_width=True)

# Weekly
with tabs[2]:
    df_w = load_data(ticker, "10y", "1wk")
    if df_w.empty:
        st.warning("No weekly data.")
    else:
        df_w = add_indicators(df_w)
        df_w = add_wave0(df_w, "1wk")
        df_w = add_wave5(df_w, "1wk")
        fig_w = plot_chart(df_w, f"{ticker} — Weekly (10Y)")
        st.plotly_chart(fig_w, use_container_width=True)

# -------------------- ML TAB --------------------
with tabs[3]:
    st.subheader("🤖 ML-Based Buy/Sell Recommendations (Hourly, Daily & Weekly)")
    st.write("Model: RandomForestClassifier trained on YOUR rule-based Buy/Sell/Hold labels using RSI, SMA(22/50/200) distances, returns, Elliott 0/5 flags, long-term highs/lows.")

    if st.button("Run ML Analysis on All Tickers"):
        all_hourly = []
        all_daily = []
        all_weekly = []

        # Build hourly, daily & weekly ML datasets for all default tickers
        for tk in default_tickers:
            # HOURLY
            df_h_all = load_data(tk, "60d", "1h")
            if not df_h_all.empty:
                df_h_all = add_indicators(df_h_all)
                df_h_all = add_wave0(df_h_all, "1h")
                df_h_all = add_wave5(df_h_all, "1h")
                feat_h = build_ml_features(df_h_all, "1h", tk)
                if not feat_h.empty:
                    all_hourly.append(feat_h)

            # DAILY
            df_d_all = load_data(tk, "3y", "1d")
            if not df_d_all.empty:
                df_d_all = add_indicators(df_d_all)
                df_d_all = add_wave0(df_d_all, "1d")
                df_d_all = add_wave5(df_d_all, "1d")
                feat_d = build_ml_features(df_d_all, "1d", tk)
                if not feat_d.empty:
                    all_daily.append(feat_d)

            # WEEKLY
            df_w_all = load_data(tk, "10y", "1wk")
            if not df_w_all.empty:
                df_w_all = add_indicators(df_w_all)
                df_w_all = add_wave0(df_w_all, "1wk")
                df_w_all = add_wave5(df_w_all, "1wk")
                feat_w = build_ml_features(df_w_all, "1wk", tk)
                if not feat_w.empty:
                    all_weekly.append(feat_w)

        if not all_daily and not all_weekly and not all_hourly:
            st.error("No sufficient data to train ML models.")
        else:
            feature_cols = [
                "Close",
                "RSI_14",
                "Dist_SMA_22",
                "Dist_SMA_50",
                "Dist_SMA_200",
                "Ret_3",
                "Ret_5",
                "Ret_10",
                "Wave0_Flag",
                "Wave5_Flag",
                "Dist_L1_High",
                "Dist_L1_Low",
                "Dist_L2_High",
                "Dist_L2_Low",
            ]

            hourly_signals = None
            daily_signals = None
            weekly_signals = None

            # HOURLY MODEL
            if all_hourly:
                df_hourly_all = pd.concat(all_hourly, ignore_index=True)
                hourly_model, _ = train_ml_model(df_hourly_all, feature_cols)

                rows_h = []
                for tk in default_tickers:
                    df_h_curr = load_data(tk, "60d", "1h")
                    if df_h_curr.empty:
                        continue
                    df_h_curr = add_indicators(df_h_curr)
                    df_h_curr = add_wave0(df_h_curr, "1h")
                    df_h_curr = add_wave5(df_h_curr, "1h")
                    proba_h = predict_for_latest(df_h_curr, hourly_model, feature_cols, "1h")
                    if proba_h is None:
                        continue
                    rows_h.append({
                        "Ticker": tk,
                        "P_Buy_Hourly": proba_h[2],
                        "P_Sell_Hourly": proba_h[0],
                    })
                hourly_signals = pd.DataFrame(rows_h)

            # DAILY MODEL
            if all_daily:
                df_daily_all = pd.concat(all_daily, ignore_index=True)
                daily_model, _ = train_ml_model(df_daily_all, feature_cols)

                rows = []
                for tk in default_tickers:
                    df_d_curr = load_data(tk, "3y", "1d")
                    if df_d_curr.empty:
                        continue
                    df_d_curr = add_indicators(df_d_curr)
                    df_d_curr = add_wave0(df_d_curr, "1d")
                    df_d_curr = add_wave5(df_d_curr, "1d")
                    proba = predict_for_latest(df_d_curr, daily_model, feature_cols, "1d")
                    if proba is None:
                        continue
                    rows.append({
                        "Ticker": tk,
                        "P_Buy_Daily": proba[2],
                        "P_Sell_Daily": proba[0],
                    })
                daily_signals = pd.DataFrame(rows)

            # WEEKLY MODEL
            if all_weekly:
                df_weekly_all = pd.concat(all_weekly, ignore_index=True)
                weekly_model, _ = train_ml_model(df_weekly_all, feature_cols)

                rows_w = []
                for tk in default_tickers:
                    df_w_curr = load_data(tk, "10y", "1wk")
                    if df_w_curr.empty:
                        continue
                    df_w_curr = add_indicators(df_w_curr)
                    df_w_curr = add_wave0(df_w_curr, "1wk")
                    df_w_curr = add_wave5(df_w_curr, "1wk")
                    proba_w = predict_for_latest(df_w_curr, weekly_model, feature_cols, "1wk")
                    if proba_w is None:
                        continue
                    rows_w.append({
                        "Ticker": tk,
                        "P_Buy_Weekly": proba_w[2],
                        "P_Sell_Weekly": proba_w[0],
                    })
                weekly_signals = pd.DataFrame(rows_w)

            # MERGE HOURLY + DAILY + WEEKLY
            merged = None
            for sig in [hourly_signals, daily_signals, weekly_signals]:
                if sig is None or sig.empty:
                    continue
                if merged is None:
                    merged = sig.copy()
                else:
                    merged = pd.merge(merged, sig, on="Ticker", how="outer")

            if merged is not None and not merged.empty:
                # choose a sort column: prefer daily, then weekly, then hourly
                sort_col = None
                for c in ["P_Buy_Daily", "P_Buy_Weekly", "P_Buy_Hourly"]:
                    if c in merged.columns:
                        sort_col = c
                        break
                if sort_col is not None:
                    merged = merged.sort_values(sort_col, ascending=False, na_position="last")

                # ---------- Text signal based on probabilities ----------
                def classify_signal(row):
                    # Prefer weekly > daily > hourly for naming
                    p_buy = row.get("P_Buy_Weekly", np.nan)
                    p_sell = row.get("P_Sell_Weekly", np.nan)

                    if np.isnan(p_buy) or np.isnan(p_sell):
                        p_buy = row.get("P_Buy_Daily", np.nan)
                        p_sell = row.get("P_Sell_Daily", np.nan)

                    if np.isnan(p_buy) or np.isnan(p_sell):
                        p_buy = row.get("P_Buy_Hourly", np.nan)
                        p_sell = row.get("P_Sell_Hourly", np.nan)

                    if np.isnan(p_buy) or np.isnan(p_sell):
                        return "No signal"

                    # Strong thresholds (you can tweak later)
                    if p_buy >= 0.65 and p_sell <= 0.20:
                        return "🚀 Strong Buy"
                    if p_sell >= 0.65 and p_buy <= 0.20:
                        return "⚠️ Strong Sell"
                    if p_buy >= 0.55 and p_sell <= 0.30:
                        return "👍 Weak Buy"
                    if p_sell >= 0.55 and p_buy <= 0.30:
                        return "👎 Weak Sell"
                    return "😐 Hold / Neutral"

                merged["Signal"] = merged.apply(classify_signal, axis=1)

                st.subheader("📋 Buy/Sell Probabilities per Ticker")
                fmt = {}
                for col in [
                    "P_Buy_Hourly", "P_Sell_Hourly",
                    "P_Buy_Daily", "P_Sell_Daily",
                    "P_Buy_Weekly", "P_Sell_Weekly",
                ]:
                    if col in merged.columns:
                        fmt[col] = "{:.2%}"

                st.dataframe(merged.style.format(fmt))
            else:
                st.warning("Could not compute signals for any ticker.")
