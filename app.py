import math
import time
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from kiteconnect import KiteConnect
from scipy.optimize import brentq
from scipy.stats import norm

st.set_page_config(page_title="F&O Volatility Dashboard", page_icon="📈", layout="wide")

TRADING_DAYS = 252
HISTORY_DAYS = 180
INTEREST_RATE = 0.06
MAX_IV_REL_SPREAD = 0.30
TICKER_FILE = Path(__file__).with_name("FNO_INDIA_TICKERS.xlsx")
TICKER_COLUMN = "TICKER"
PRODUCT = "NRML"
QUOTE_BATCH_SIZE = 400


# -----------------------------
# Formatting / loading
# -----------------------------
def fmt_money(x):
    return "—" if x is None or pd.isna(x) else f"₹{x:,.2f}"


def fmt_pct(x):
    return "—" if x is None or pd.isna(x) else f"{x:,.2f}%"


def load_tickers():
    df = pd.read_excel(TICKER_FILE)
    df.columns = [str(c).strip() for c in df.columns]
    if TICKER_COLUMN not in df.columns:
        raise ValueError(f"'{TICKER_COLUMN}' column not found in {TICKER_FILE.name}.")
    return df[TICKER_COLUMN].dropna().astype(str).str.strip().str.upper().drop_duplicates().tolist()


@st.cache_data(ttl=3600, show_spinner=False)
def get_instrument_master(api_key: str, access_token: str):
    kite = KiteConnect(api_key=api_key)
    kite.set_access_token(access_token)
    data = pd.DataFrame(kite.instruments())
    data["expiry"] = pd.to_datetime(data["expiry"], errors="coerce")
    data["strike"] = pd.to_numeric(data["strike"], errors="coerce")
    data["lot_size"] = pd.to_numeric(data["lot_size"], errors="coerce")
    return data


def kite_client():
    api_key = st.session_state.get("api_key", "")
    access_token = st.session_state.get("access_token", "")
    if not api_key or not access_token:
        return None
    kite = KiteConnect(api_key=api_key)
    kite.set_access_token(access_token)
    return kite


def quote_in_batches(kite, symbols, batch_size=QUOTE_BATCH_SIZE):
    symbols = list(dict.fromkeys(symbols))
    out = {}
    for i in range(0, len(symbols), batch_size):
        out.update(kite.quote(symbols[i:i + batch_size]))
    return out


# -----------------------------
# IV / RV math
# -----------------------------
def get_history(kite, instrument_token):
    end_date = datetime.now()
    start_date = end_date - timedelta(days=HISTORY_DAYS)
    candles = kite.historical_data(instrument_token, start_date, end_date, "day")
    if not candles:
        return pd.DataFrame()
    df = pd.DataFrame(candles)
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)


def calculate_rv20(price_df):
    """20-trading-day close-to-close realized vol, annualized with sqrt(252), in percent."""
    close = price_df["close"].astype(float)
    returns = np.log(close / close.shift(1)).dropna()
    if len(returns) < 20:
        return np.nan
    return float(returns.tail(20).std(ddof=1) * np.sqrt(TRADING_DAYS) * 100)


def black76_price(option_type, forward, strike, time_to_expiry, rate, volatility):
    if min(forward, strike, time_to_expiry, volatility) <= 0:
        return np.nan
    sqrt_t = math.sqrt(time_to_expiry)
    d1 = (math.log(forward / strike) + 0.5 * volatility * volatility * time_to_expiry) / (volatility * sqrt_t)
    d2 = d1 - volatility * sqrt_t
    disc = math.exp(-rate * time_to_expiry)
    if option_type == "CE":
        return disc * (forward * norm.cdf(d1) - strike * norm.cdf(d2))
    if option_type == "PE":
        return disc * (strike * norm.cdf(-d2) - forward * norm.cdf(-d1))
    return np.nan


def calculate_forward_iv(market_price, option_type, forward, strike, time_to_expiry):
    if market_price is None or pd.isna(market_price) or min(market_price, forward, strike, time_to_expiry) <= 0:
        return np.nan
    try:
        iv = brentq(
            lambda vol: black76_price(option_type, forward, strike, time_to_expiry, INTEREST_RATE, vol) - market_price,
            0.001,
            5.0,
            maxiter=200,
        )
        return float(iv * 100)
    except Exception:
        return np.nan


def strict_mid_price(quote, max_rel_spread=MAX_IV_REL_SPREAD):
    """Return a live midpoint only when both sides exist and the spread is not excessively wide."""
    if not quote:
        return np.nan, np.nan, np.nan, np.nan
    depth = quote.get("depth", {}) or {}
    buys, sells = depth.get("buy", []) or [], depth.get("sell", []) or []
    bid = float(buys[0].get("price", 0) or 0) if buys else 0.0
    ask = float(sells[0].get("price", 0) or 0) if sells else 0.0
    if bid <= 0 or ask <= 0 or ask < bid:
        return np.nan, bid, ask, np.nan
    mid = (bid + ask) / 2
    rel_spread = (ask - bid) / mid if mid > 0 else np.nan
    if pd.isna(rel_spread) or rel_spread > max_rel_spread:
        return np.nan, bid, ask, rel_spread
    return mid, bid, ask, rel_spread


def nearest_expiry(stock_options):
    today = pd.Timestamp.now().normalize()
    expiries = stock_options.loc[stock_options["expiry"] >= today, "expiry"].dropna().sort_values().unique()
    return None if len(expiries) == 0 else pd.Timestamp(expiries[0])


def nearest_strike(target, strikes):
    strikes = [float(x) for x in strikes if pd.notna(x)]
    if not strikes:
        raise ValueError("No valid strikes available.")
    return min(strikes, key=lambda x: abs(x - target))


def run_scanner(kite, instruments, tickers, progress=None):
    """
    Scanner output intentionally stays lean:
    IV, RV20, IV/RV20 and IV-RV20 only.

    IV methodology:
    - nearest active expiry
    - nearest listed ATM strike
    - valid live bid+ask required for BOTH ATM CE and PE
    - reject unusually wide quotes
    - infer forward from ATM put-call parity
    - solve CE and PE IV with Black-76, then average the two
    """
    equities = instruments[(instruments["exchange"] == "NSE") & (instruments["instrument_type"] == "EQ")].copy()
    options = instruments[(instruments["exchange"] == "NFO") & (instruments["instrument_type"].isin(["CE", "PE"]))].copy()
    equity_lookup = equities.set_index("tradingsymbol", drop=False)
    results, failed = [], []

    for idx, ticker in enumerate(tickers, start=1):
        if progress:
            progress.progress(idx / len(tickers), text=f"Scanning {ticker} ({idx}/{len(tickers)})")
        try:
            if ticker not in equity_lookup.index:
                raise ValueError("Ticker not found in Kite NSE instruments")
            equity = equity_lookup.loc[ticker]
            if isinstance(equity, pd.DataFrame):
                equity = equity.iloc[0]

            history = get_history(kite, int(equity["instrument_token"]))
            if history.empty:
                raise ValueError("No historical price data")
            rv20 = calculate_rv20(history)
            if pd.isna(rv20) or rv20 <= 0:
                raise ValueError("RV20 unavailable")

            spot_symbol = f"NSE:{ticker}"
            try:
                spot = float(kite.quote([spot_symbol])[spot_symbol]["last_price"])
            except Exception:
                spot = float(history["close"].iloc[-1])

            stock_options = options[options["name"] == ticker].copy()
            if stock_options.empty:
                raise ValueError("No NFO option contracts found")
            expiry = nearest_expiry(stock_options)
            if expiry is None:
                raise ValueError("No active expiry found")
            chain = stock_options[stock_options["expiry"] == expiry].copy()
            strikes = np.sort(chain["strike"].dropna().astype(float).unique())
            if not len(strikes):
                raise ValueError("No strikes")
            atm = float(strikes[np.argmin(np.abs(strikes - spot))])

            call = chain[(chain["strike"] == atm) & (chain["instrument_type"] == "CE")]
            put = chain[(chain["strike"] == atm) & (chain["instrument_type"] == "PE")]
            if call.empty or put.empty:
                raise ValueError("ATM call/put not found")

            call_symbol = "NFO:" + call.iloc[0]["tradingsymbol"]
            put_symbol = "NFO:" + put.iloc[0]["tradingsymbol"]
            quotes = kite.quote([call_symbol, put_symbol])
            call_mid, call_bid, call_ask, call_spread = strict_mid_price(quotes.get(call_symbol))
            put_mid, put_bid, put_ask, put_spread = strict_mid_price(quotes.get(put_symbol))
            if pd.isna(call_mid) or pd.isna(put_mid):
                raise ValueError("ATM CE/PE quote missing or bid-ask spread too wide for reliable IV")

            now = pd.Timestamp.now(tz="Asia/Kolkata")
            expiry_time = pd.Timestamp(expiry.date(), tz="Asia/Kolkata") + pd.Timedelta(hours=15, minutes=30)
            seconds_left = (expiry_time - now).total_seconds()
            if seconds_left <= 0:
                raise ValueError("Selected contract expired")
            t = seconds_left / (365 * 24 * 60 * 60)

            # Put-call parity: C - P = exp(-rT) * (F - K)
            forward = atm + math.exp(INTEREST_RATE * t) * (call_mid - put_mid)
            if forward <= 0:
                raise ValueError("Invalid forward inferred from ATM put-call parity")

            call_iv = calculate_forward_iv(call_mid, "CE", forward, atm, t)
            put_iv = calculate_forward_iv(put_mid, "PE", forward, atm, t)
            if pd.isna(call_iv) or pd.isna(put_iv):
                raise ValueError("CE or PE IV calculation failed")

            iv = float((call_iv + put_iv) / 2)
            iv_rv20 = iv / rv20
            iv_minus_rv20 = iv - rv20
            dte = (expiry.normalize() - pd.Timestamp.now().normalize()).days

            results.append({
                "Ticker": ticker,
                "Spot": round(spot, 2),
                "Expiry": expiry.date(),
                "DTE": dte,
                "ATM": atm,
                "IV": round(iv, 2),
                "RV20": round(rv20, 2),
                "IV/RV20": round(iv_rv20, 3),
                "IV-RV20": round(iv_minus_rv20, 2),
                # diagnostics retained internally; not shown in the main scanner table
                "CE IV": round(call_iv, 2),
                "PE IV": round(put_iv, 2),
                "CE Spread %": round(call_spread * 100, 1),
                "PE Spread %": round(put_spread * 100, 1),
                "Forward": round(forward, 2),
            })
        except Exception as exc:
            failed.append({"Ticker": ticker, "Reason": str(exc)})
        time.sleep(0.04)

    result_df = pd.DataFrame(results)
    if not result_df.empty:
        result_df = result_df.sort_values("IV/RV20", ascending=False).reset_index(drop=True)
        result_df.insert(0, "Rank", range(1, len(result_df) + 1))
    return result_df, pd.DataFrame(failed)


# -----------------------------
# Iron butterfly helpers
# -----------------------------
def get_market_data(quote):
    quote = quote or {}
    ltp = float(quote.get("last_price", 0) or 0)
    depth = quote.get("depth", {}) or {}
    buys, sells = depth.get("buy", []) or [], depth.get("sell", []) or []
    best_bid = float(buys[0].get("price", 0) or 0) if buys else 0.0
    best_ask = float(sells[0].get("price", 0) or 0) if sells else 0.0
    return {"ltp": ltp, "best_bid": best_bid, "best_ask": best_ask, "buy_depth": buys, "sell_depth": sells}


def execution_vwap(market_data, transaction_type, quantity):
    levels = market_data["sell_depth"] if transaction_type == "BUY" else market_data["buy_depth"]
    remaining, filled, total_value = int(quantity), 0, 0.0
    for level in levels:
        price = float(level.get("price", 0) or 0)
        available = int(level.get("quantity", 0) or 0)
        if price <= 0 or available <= 0:
            continue
        take = min(remaining, available)
        total_value += take * price
        filled += take
        remaining -= take
        if remaining <= 0:
            break
    return {
        "vwap": total_value / filled if filled else None,
        "filled_qty": filled,
        "full_depth": filled >= quantity,
        "coverage": min(filled / quantity, 1.0) if quantity else 0.0,
    }


def get_contract(df, strike, option_type):
    result = df[(df["strike"] == strike) & (df["instrument_type"] == option_type)]
    if result.empty:
        raise ValueError(f"No {option_type} contract found for strike {strike}")
    return result.iloc[0]


def choose_execution_price(market, execution, transaction_type):
    """VWAP when full displayed depth covers qty, otherwise top-of-book, then LTP fallback."""
    if execution["full_depth"] and execution["vwap"] and execution["vwap"] > 0:
        return float(execution["vwap"]), "Depth VWAP"
    top = market["best_ask"] if transaction_type == "BUY" else market["best_bid"]
    if top and top > 0:
        return float(top), "Best Ask" if transaction_type == "BUY" else "Best Bid"
    if market["ltp"] and market["ltp"] > 0:
        return float(market["ltp"]), "LTP fallback"
    raise ValueError("No usable market price")


def build_butterfly_plan(options, symbol, spot, put_wing_pct, call_wing_pct):
    stock_options = options[options["name"].astype(str).str.upper() == symbol.upper()].copy()
    if stock_options.empty:
        raise ValueError("No active NFO options")
    expiry = nearest_expiry(stock_options)
    if expiry is None:
        raise ValueError("No active expiry")
    chain = stock_options[stock_options["expiry"] == expiry].copy()
    ce_strikes = sorted(chain.loc[chain["instrument_type"] == "CE", "strike"].dropna().astype(float).unique())
    pe_strikes = sorted(chain.loc[chain["instrument_type"] == "PE", "strike"].dropna().astype(float).unique())
    common = sorted(set(ce_strikes) & set(pe_strikes))
    if not common:
        raise ValueError("No common CE/PE strikes")
    atm = nearest_strike(spot, common)
    lower_candidates = [s for s in pe_strikes if s < atm]
    upper_candidates = [s for s in ce_strikes if s > atm]
    if not lower_candidates or not upper_candidates:
        raise ValueError("No valid wing strikes")
    lower_target = spot * (1 - put_wing_pct)
    upper_target = spot * (1 + call_wing_pct)
    lower_strike = nearest_strike(lower_target, lower_candidates)
    upper_strike = nearest_strike(upper_target, upper_candidates)
    long_put = get_contract(chain, lower_strike, "PE")
    short_put = get_contract(chain, atm, "PE")
    short_call = get_contract(chain, atm, "CE")
    long_call = get_contract(chain, upper_strike, "CE")
    lots = {int(x["lot_size"]) for x in [long_put, short_put, short_call, long_call]}
    if len(lots) != 1:
        raise ValueError("Inconsistent lot sizes")
    return {
        "symbol": symbol,
        "expiry": expiry,
        "spot": float(spot),
        "atm": float(atm),
        "lower_target": float(lower_target),
        "upper_target": float(upper_target),
        "lower_strike": float(lower_strike),
        "upper_strike": float(upper_strike),
        "base_lot": lots.pop(),
        "contracts": {
            "BUY PE": long_put,
            "SELL PE": short_put,
            "SELL CE": short_call,
            "BUY CE": long_call,
        },
    }


def calculate_plan_from_quotes(plan, quotes, lot_multiplier=1, include_margin=False, kite=None):
    qty = int(plan["base_lot"]) * int(lot_multiplier)
    quote_symbols = {leg: f"NFO:{contract['tradingsymbol']}" for leg, contract in plan["contracts"].items()}
    tx = {"BUY PE": "BUY", "SELL PE": "SELL", "SELL CE": "SELL", "BUY CE": "BUY"}
    market, execs, used, source = {}, {}, {}, {}
    for leg, key in quote_symbols.items():
        market[leg] = get_market_data(quotes.get(key))
        execs[leg] = execution_vwap(market[leg], tx[leg], qty)
        used[leg], source[leg] = choose_execution_price(market[leg], execs[leg], tx[leg])

    received_ps = used["SELL PE"] + used["SELL CE"]
    paid_ps = used["BUY PE"] + used["BUY CE"]
    net_credit_ps = received_ps - paid_ps
    premium_received = received_ps * qty
    premium_paid = paid_ps * qty
    net_credit = net_credit_ps * qty
    downside_width = plan["atm"] - plan["lower_strike"]
    upside_width = plan["upper_strike"] - plan["atm"]
    down_loss = max((downside_width - net_credit_ps) * qty, 0)
    up_loss = max((upside_width - net_credit_ps) * qty, 0)
    max_loss = max(down_loss, up_loss, 0)
    lower_be = plan["atm"] - net_credit_ps
    upper_be = plan["atm"] + net_credit_ps

    funds = margin = np.nan
    if include_margin and kite is not None:
        basket_orders = []
        for leg in ["BUY PE", "BUY CE", "SELL PE", "SELL CE"]:
            contract = plan["contracts"][leg]
            basket_orders.append({
                "exchange": "NFO",
                "tradingsymbol": contract["tradingsymbol"],
                "transaction_type": tx[leg],
                "variety": "regular",
                "product": PRODUCT,
                "order_type": "MARKET",
                "quantity": qty,
                "price": 0,
                "trigger_price": 0,
            })
        margin_data = kite.basket_order_margins(basket_orders, consider_positions=False, mode="compact")
        funds = float(margin_data.get("initial", {}).get("total", 0) or 0)
        margin = float(margin_data.get("final", {}).get("total", 0) or 0)

    min_coverage = min(x["coverage"] for x in execs.values())
    indicative = any(s == "LTP fallback" for s in source.values()) or min_coverage < 1

    rows = []
    for leg in ["BUY PE", "SELL PE", "SELL CE", "BUY CE"]:
        rows.append({
            "Leg": leg,
            "Strike": plan["lower_strike"] if leg == "BUY PE" else plan["upper_strike"] if leg == "BUY CE" else plan["atm"],
            "Contract": plan["contracts"][leg]["tradingsymbol"],
            "Bid": market[leg]["best_bid"],
            "Ask": market[leg]["best_ask"],
            "LTP": market[leg]["ltp"],
            "Used Price": used[leg],
            "Price Source": source[leg],
            "Visible Qty": execs[leg]["filled_qty"],
            "Visible Depth %": execs[leg]["coverage"] * 100,
        })

    return {
        **plan,
        "updated": datetime.now(),
        "quantity": qty,
        "lots": int(lot_multiplier),
        "premium_received": premium_received,
        "premium_paid": premium_paid,
        "net_credit": net_credit,
        "net_credit_ps": net_credit_ps,
        "max_profit": net_credit,
        "downside_max_loss": down_loss,
        "upside_max_loss": up_loss,
        "max_loss": max_loss,
        "lower_be": lower_be,
        "upper_be": upper_be,
        "standalone_funds": funds,
        "standalone_margin": margin,
        "max_profit_on_funds": net_credit / funds * 100 if pd.notna(funds) and funds > 0 else np.nan,
        "max_profit_on_margin": net_credit / margin * 100 if pd.notna(margin) and margin > 0 else np.nan,
        "min_depth_coverage": min_coverage,
        "indicative": indicative,
        "legs": pd.DataFrame(rows),
    }


def calculate_iron_fly(kite, instruments, symbol, lots, put_wing_pct, call_wing_pct, include_margin=True):
    options = instruments[(instruments["exchange"] == "NFO") & (instruments["instrument_type"].isin(["CE", "PE"]))].copy()
    spot_key = f"NSE:{symbol}"
    spot_quote = kite.quote([spot_key]).get(spot_key)
    if not spot_quote:
        raise ValueError(f"Unable to get NSE quote for {symbol}")
    spot = float(spot_quote["last_price"])
    plan = build_butterfly_plan(options, symbol, spot, put_wing_pct, call_wing_pct)
    quote_symbols = [f"NFO:{x['tradingsymbol']}" for x in plan["contracts"].values()]
    quotes = kite.quote(quote_symbols)
    return calculate_plan_from_quotes(plan, quotes, lots, include_margin=include_margin, kite=kite)


def scan_all_butterflies(kite, instruments, symbols, lots, put_wing_pct, call_wing_pct):
    """Batch version for the universe/watchlist. Broker margin calls are intentionally excluded for speed."""
    if not symbols:
        return pd.DataFrame(), pd.DataFrame()
    options = instruments[(instruments["exchange"] == "NFO") & (instruments["instrument_type"].isin(["CE", "PE"]))].copy()

    spot_keys = [f"NSE:{s}" for s in symbols]
    spot_quotes = quote_in_batches(kite, spot_keys)
    plans, failed = [], []
    for symbol in symbols:
        try:
            q = spot_quotes.get(f"NSE:{symbol}")
            if not q or not q.get("last_price"):
                raise ValueError("No spot quote")
            plans.append(build_butterfly_plan(options, symbol, float(q["last_price"]), put_wing_pct, call_wing_pct))
        except Exception as exc:
            failed.append({"Ticker": symbol, "Reason": str(exc)})

    leg_symbols = []
    for p in plans:
        leg_symbols.extend([f"NFO:{x['tradingsymbol']}" for x in p["contracts"].values()])
    leg_quotes = quote_in_batches(kite, leg_symbols) if leg_symbols else {}

    rows = []
    for p in plans:
        try:
            r = calculate_plan_from_quotes(p, leg_quotes, lots, include_margin=False, kite=None)
            rows.append({
                "Ticker": p["symbol"],
                "Spot": round(r["spot"], 2),
                "Expiry": p["expiry"].date() if hasattr(p["expiry"], "date") else p["expiry"],
                "ATM": r["atm"],
                "Put Wing": r["lower_strike"],
                "Call Wing": r["upper_strike"],
                "Net Credit": round(r["net_credit"], 2),
                "Credit / Share": round(r["net_credit_ps"], 2),
                "Max Profit": round(r["max_profit"], 2),
                "Max Loss": round(r["max_loss"], 2),
                "Lower BE": round(r["lower_be"], 2),
                "Upper BE": round(r["upper_be"], 2),
                "Visible Depth %": round(r["min_depth_coverage"] * 100, 0),
                "Price Quality": "Indicative" if r["indicative"] else "Executable depth",
            })
        except Exception as exc:
            failed.append({"Ticker": p["symbol"], "Reason": str(exc)})

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("Ticker").reset_index(drop=True)
    return df, pd.DataFrame(failed)


# -----------------------------
# Session + one-click login
# -----------------------------
for key, value in {
    "access_token": "",
    "scanner_results": None,
    "scanner_failed": None,
    "selected_symbol": "RELIANCE",
    "watchlist": [],
}.items():
    st.session_state.setdefault(key, value)

st.title("F&O Volatility + Iron Butterfly Dashboard")
st.caption("Live Zerodha Kite scanner and configurable short iron butterfly monitor")

try:
    st.session_state.api_key = st.secrets["KITE_API_KEY"]
    api_secret = st.secrets["KITE_API_SECRET"]
except Exception:
    st.error("Kite credentials are not configured. Add KITE_API_KEY and KITE_API_SECRET in Streamlit Cloud → App settings → Secrets.")
    st.stop()

login_kite = KiteConnect(api_key=st.session_state.api_key)
request_token = st.query_params.get("request_token")
if request_token and not st.session_state.access_token:
    try:
        session = login_kite.generate_session(request_token, api_secret=api_secret)
        st.session_state.access_token = session["access_token"]
        st.query_params.clear()
        st.rerun()
    except Exception as exc:
        st.query_params.clear()
        st.error(f"Zerodha login failed: {exc}")

with st.sidebar:
    st.header("Zerodha")
    if st.session_state.access_token:
        try:
            user = kite_client().profile()
            st.success(f"Connected: {user.get('user_name', user.get('user_id', 'Kite user'))}")
            if st.button("Disconnect", use_container_width=True):
                st.session_state.access_token = ""
                get_instrument_master.clear()
                st.rerun()
        except Exception:
            st.session_state.access_token = ""
            st.warning("Your Kite session has expired. Log in again.")
    if not st.session_state.access_token:
        st.link_button("Login to Zerodha", login_kite.login_url(), type="primary", use_container_width=True)
        st.caption("You will return here automatically after Zerodha authentication.")

if not st.session_state.access_token:
    st.info("Click **Login to Zerodha** in the sidebar to start the live dashboard.")
    st.stop()

kite = kite_client()
try:
    instruments = get_instrument_master(st.session_state.api_key, st.session_state.access_token)
except Exception as exc:
    st.error(f"Could not load Kite instruments: {exc}")
    st.stop()

tickers = load_tickers()

scanner_tab, all_fly_tab, individual_tab, watchlist_tab = st.tabs([
    "① IV / RV Scanner",
    "② All Iron Butterflies",
    "③ Individual Butterfly",
    "④ Watchlist",
])


# -----------------------------
# TAB 1 — IV/RV scanner
# -----------------------------
with scanner_tab:
    top_l, top_r = st.columns([2, 1])
    with top_l:
        st.subheader("IV / RV20 scanner")
        st.caption("No Cheap/Fair/Rich labels — the table shows the raw relationship and lets you decide.")
    with top_r:
        scan_clicked = st.button("Refresh IV/RV scanner", type="primary", use_container_width=True)

    if scan_clicked:
        p = st.progress(0, text="Starting scan…")
        try:
            result_df, failed_df = run_scanner(kite, instruments, tickers, p)
            st.session_state.scanner_results = result_df
            st.session_state.scanner_failed = failed_df
            st.session_state.scanner_updated = datetime.now()
        finally:
            p.empty()

    df = st.session_state.scanner_results
    if df is None:
        st.info("Click **Refresh IV/RV scanner** to generate the current ranking.")
    elif df.empty:
        st.warning("The latest scan returned no reliable IV/RV observations.")
    else:
        c1, c2 = st.columns([1, 2])
        search = c1.text_input("Ticker contains", placeholder="e.g. RELIANCE", key="iv_search").strip().upper()
        sort_direction = c2.radio("Sort IV/RV20", ["Highest first", "Lowest first"], horizontal=True)
        view = df.copy()
        if search:
            view = view[view["Ticker"].str.contains(search, na=False)]
        view = view.sort_values("IV/RV20", ascending=(sort_direction == "Lowest first")).reset_index(drop=True)
        main_cols = ["Rank", "Ticker", "Spot", "Expiry", "DTE", "ATM", "IV", "RV20", "IV/RV20", "IV-RV20"]
        updated = st.session_state.get("scanner_updated")
        if updated:
            st.caption(f"Latest refresh: {updated:%d-%m-%Y %H:%M:%S}")
        st.dataframe(
            view[main_cols],
            use_container_width=True,
            hide_index=True,
            column_config={
                "IV": st.column_config.NumberColumn(format="%.2f%%"),
                "RV20": st.column_config.NumberColumn(format="%.2f%%"),
                "IV/RV20": st.column_config.NumberColumn(format="%.3f"),
                "IV-RV20": st.column_config.NumberColumn(format="%.2f pp"),
            },
        )
        with st.expander("IV quote diagnostics"):
            st.caption("Used to audit the IV calculation; these columns are intentionally hidden from the main scanner.")
            diag_cols = ["Ticker", "CE IV", "PE IV", "CE Spread %", "PE Spread %", "Forward"]
            st.dataframe(view[diag_cols], use_container_width=True, hide_index=True)

        failed = st.session_state.scanner_failed
        if failed is not None and not failed.empty:
            with st.expander(f"Excluded / unreliable observations ({len(failed)})"):
                st.dataframe(failed, use_container_width=True, hide_index=True)


# -----------------------------
# Shared all-butterfly table renderer
# -----------------------------
def render_universe_butterflies(symbols, lots, put_pct, call_pct, key_prefix):
    try:
        df, failed = scan_all_butterflies(kite, instruments, symbols, int(lots), put_pct / 100, call_pct / 100)
    except Exception as exc:
        st.error(str(exc))
        return
    st.caption(f"Updated {datetime.now():%d-%m-%Y %H:%M:%S} · Broker margin/funds are excluded from this fast universe table and remain available in the Individual tab.")
    if df.empty:
        st.warning("No butterfly rows could be calculated.")
        return
    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Spot": st.column_config.NumberColumn(format="₹%.2f"),
            "ATM": st.column_config.NumberColumn(format="₹%.0f"),
            "Put Wing": st.column_config.NumberColumn(format="₹%.0f"),
            "Call Wing": st.column_config.NumberColumn(format="₹%.0f"),
            "Net Credit": st.column_config.NumberColumn(format="₹%.2f"),
            "Credit / Share": st.column_config.NumberColumn(format="₹%.2f"),
            "Max Profit": st.column_config.NumberColumn(format="₹%.2f"),
            "Max Loss": st.column_config.NumberColumn(format="₹%.2f"),
            "Lower BE": st.column_config.NumberColumn(format="₹%.2f"),
            "Upper BE": st.column_config.NumberColumn(format="₹%.2f"),
            "Visible Depth %": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%.0f%%"),
        },
    )
    if failed is not None and not failed.empty:
        with st.expander(f"Unavailable rows ({len(failed)})"):
            st.dataframe(failed, use_container_width=True, hide_index=True)


# -----------------------------
# TAB 2 — all butterflies
# -----------------------------
with all_fly_tab:
    st.subheader("All-stock Iron Butterfly monitor")
    st.caption("One table for the entire F&O stock list. The default settings apply to every stock and can be changed here.")
    a1, a2, a3, a4 = st.columns(4)
    all_lots = a1.number_input("Lots", min_value=1, max_value=20, value=1, step=1, key="all_lots")
    all_put_pct = a2.number_input("Put wing below spot (%)", min_value=0.25, max_value=30.0, value=5.0, step=0.25, key="all_put")
    all_call_pct = a3.number_input("Call wing above spot (%)", min_value=0.25, max_value=30.0, value=5.0, step=0.25, key="all_call")
    all_auto = a4.toggle("Auto-refresh every 5s", value=True, key="all_auto")

    add_choices = st.multiselect("Add stocks to watchlist", tickers, key="all_watch_add")
    if st.button("Add selected to watchlist", key="all_add_btn"):
        st.session_state.watchlist = sorted(set(st.session_state.watchlist) | set(add_choices))
        st.success(f"Watchlist now has {len(st.session_state.watchlist)} stock(s).")

    if all_auto and hasattr(st, "fragment"):
        @st.fragment(run_every="5s")
        def all_live_table():
            render_universe_butterflies(tickers, all_lots, all_put_pct, all_call_pct, "all")
        all_live_table()
    else:
        if st.button("Refresh all butterflies", type="primary", key="all_manual"):
            render_universe_butterflies(tickers, all_lots, all_put_pct, all_call_pct, "all")


# -----------------------------
# TAB 3 — individual butterfly
# -----------------------------
with individual_tab:
    st.subheader("Individual Iron Butterfly")
    st.caption("Full live detail for one stock, including Zerodha standalone funds and final hedged margin.")
    ctl1, ctl2, ctl3, ctl4 = st.columns(4)
    symbol = ctl1.selectbox("Stock", tickers, index=tickers.index(st.session_state.selected_symbol) if st.session_state.selected_symbol in tickers else 0, key="individual_symbol")
    lots = ctl2.number_input("Lots", min_value=1, max_value=100, value=1, step=1, key="individual_lots")
    put_pct = ctl3.number_input("Put wing below spot (%)", min_value=0.25, max_value=30.0, value=5.0, step=0.25, key="individual_put")
    call_pct = ctl4.number_input("Call wing above spot (%)", min_value=0.25, max_value=30.0, value=5.0, step=0.25, key="individual_call")

    b1, b2, b3 = st.columns([1, 1, 2])
    indiv_auto = b1.toggle("Auto-refresh", value=True, key="indiv_auto")
    indiv_seconds = b2.selectbox("Refresh interval", [5, 10, 15, 30, 60], index=0, disabled=not indiv_auto, key="indiv_secs")
    if b3.button("Add to watchlist", type="primary", use_container_width=True, key="indiv_add_watch"):
        st.session_state.watchlist = sorted(set(st.session_state.watchlist) | {symbol})
        st.success(f"{symbol} added to watchlist.")

    def render_individual():
        try:
            result = calculate_iron_fly(kite, instruments, symbol, int(lots), put_pct / 100, call_pct / 100, include_margin=True)
        except Exception as exc:
            st.error(str(exc))
            return

        st.caption(f"Updated {result['updated']:%d-%m-%Y %H:%M:%S} · Expiry {result['expiry'].date() if hasattr(result['expiry'], 'date') else result['expiry']} · Quantity {result['quantity']:,}")
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Spot", fmt_money(result["spot"]))
        m2.metric("ATM", f"₹{result['atm']:,.0f}")
        m3.metric("Net credit", fmt_money(result["net_credit"]))
        m4.metric("Max profit", fmt_money(result["max_profit"]))
        m5.metric("Max loss", fmt_money(result["max_loss"]))

        s1, s2, s3, s4 = st.columns(4)
        s1.metric(f"Buy PE ({put_pct:.2f}%)", f"₹{result['lower_strike']:,.0f}")
        s2.metric("Sell ATM PE", f"₹{result['atm']:,.0f}")
        s3.metric("Sell ATM CE", f"₹{result['atm']:,.0f}")
        s4.metric(f"Buy CE ({call_pct:.2f}%)", f"₹{result['upper_strike']:,.0f}")

        if result["indicative"]:
            st.warning("At least one leg lacks full visible depth or uses LTP fallback. Treat the execution figures as indicative.")
        else:
            st.success("Visible depth covers the requested quantity on all four legs.")

        st.dataframe(
            result["legs"],
            use_container_width=True,
            hide_index=True,
            column_config={
                "Strike": st.column_config.NumberColumn(format="₹%.0f"),
                "Bid": st.column_config.NumberColumn(format="₹%.2f"),
                "Ask": st.column_config.NumberColumn(format="₹%.2f"),
                "LTP": st.column_config.NumberColumn(format="₹%.2f"),
                "Used Price": st.column_config.NumberColumn(format="₹%.2f"),
                "Visible Depth %": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%.0f%%"),
            },
        )

        a, b, c, d = st.columns(4)
        a.metric("Premium received", fmt_money(result["premium_received"]))
        b.metric("Premium paid", fmt_money(result["premium_paid"]))
        c.metric("Standalone funds", fmt_money(result["standalone_funds"]))
        d.metric("Final hedged margin", fmt_money(result["standalone_margin"]))

        e, f, g, h = st.columns(4)
        e.metric("Lower breakeven", fmt_money(result["lower_be"]))
        f.metric("Upper breakeven", fmt_money(result["upper_be"]))
        g.metric("Max profit / funds", fmt_pct(result["max_profit_on_funds"]))
        h.metric("Max profit / margin", fmt_pct(result["max_profit_on_margin"]))

        st.markdown("#### Risk detail")
        r1, r2, r3 = st.columns(3)
        r1.metric("Downside max loss", fmt_money(result["downside_max_loss"]))
        r2.metric("Upside max loss", fmt_money(result["upside_max_loss"]))
        r3.metric("Net credit / share", fmt_money(result["net_credit_ps"]))
        r4, r5, r6 = st.columns(3)
        r4.metric("Put target", fmt_money(result["lower_target"]))
        r5.metric("Call target", fmt_money(result["upper_target"]))
        r6.metric("Exchange lot size", f"{result['base_lot']:,}")

    if indiv_auto and hasattr(st, "fragment"):
        @st.fragment(run_every=f"{indiv_seconds}s")
        def individual_live_panel():
            render_individual()
        individual_live_panel()
    else:
        if st.button("Refresh individual", type="primary", key="indiv_manual"):
            render_individual()


# -----------------------------
# TAB 4 — watchlist
# -----------------------------
with watchlist_tab:
    st.subheader("Iron Butterfly watchlist")
    st.caption("A smaller live table for the stocks you care about most.")

    if st.session_state.watchlist:
        w1, w2, w3, w4 = st.columns(4)
        watch_lots = w1.number_input("Lots", min_value=1, max_value=20, value=1, step=1, key="watch_lots")
        watch_put_pct = w2.number_input("Put wing below spot (%)", min_value=0.25, max_value=30.0, value=5.0, step=0.25, key="watch_put")
        watch_call_pct = w3.number_input("Call wing above spot (%)", min_value=0.25, max_value=30.0, value=5.0, step=0.25, key="watch_call")
        watch_auto = w4.toggle("Auto-refresh every 5s", value=True, key="watch_auto")

        remove = st.multiselect("Remove from watchlist", st.session_state.watchlist, key="watch_remove")
        if st.button("Remove selected", key="watch_remove_btn"):
            st.session_state.watchlist = [x for x in st.session_state.watchlist if x not in set(remove)]
            st.rerun()

        open_symbol = st.selectbox("Send stock to Individual tab", st.session_state.watchlist, key="watch_open")
        if st.button("Use in Individual Butterfly", key="watch_open_btn"):
            st.session_state.selected_symbol = open_symbol
            st.success(f"{open_symbol} selected. Open the **Individual Butterfly** tab.")

        if watch_auto and hasattr(st, "fragment"):
            @st.fragment(run_every="5s")
            def watch_live_table():
                render_universe_butterflies(st.session_state.watchlist, watch_lots, watch_put_pct, watch_call_pct, "watch")
            watch_live_table()
        else:
            if st.button("Refresh watchlist", type="primary", key="watch_manual"):
                render_universe_butterflies(st.session_state.watchlist, watch_lots, watch_put_pct, watch_call_pct, "watch")
    else:
        st.info("Your watchlist is empty. Add stocks from **All Iron Butterflies** or **Individual Butterfly**.")
