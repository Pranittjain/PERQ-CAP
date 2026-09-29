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
RV_WINDOWS = [5, 10, 20, 60]
HISTORY_DAYS = 180
INTEREST_RATE = 0.06
TICKER_FILE = Path(__file__).with_name("FNO_INDIA_TICKERS.xlsx")
TICKER_COLUMN = "TICKER"
PRODUCT = "NRML"

# -----------------------------
# Helpers
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
    return (
        df[TICKER_COLUMN].dropna().astype(str).str.strip().str.upper().drop_duplicates().tolist()
    )


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


def get_history(kite, instrument_token):
    end_date = datetime.now()
    start_date = end_date - timedelta(days=HISTORY_DAYS)
    candles = kite.historical_data(instrument_token, start_date, end_date, "day")
    if not candles:
        return pd.DataFrame()
    df = pd.DataFrame(candles)
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)


def calculate_rv(price_df):
    close = price_df["close"].astype(float)
    returns = np.log(close / close.shift(1)).dropna()
    out = {}
    for window in RV_WINDOWS:
        out[f"RV{window}"] = (
            returns.tail(window).std(ddof=1) * np.sqrt(TRADING_DAYS) * 100
            if len(returns) >= window else np.nan
        )
    return out


def bs_price(option_type, spot, strike, time_to_expiry, rate, volatility):
    if min(spot, strike, time_to_expiry, volatility) <= 0:
        return np.nan
    sqrt_t = math.sqrt(time_to_expiry)
    d1 = (math.log(spot / strike) + (rate + 0.5 * volatility**2) * time_to_expiry) / (volatility * sqrt_t)
    d2 = d1 - volatility * sqrt_t
    if option_type == "CE":
        return spot * norm.cdf(d1) - strike * math.exp(-rate * time_to_expiry) * norm.cdf(d2)
    if option_type == "PE":
        return strike * math.exp(-rate * time_to_expiry) * norm.cdf(-d2) - spot * norm.cdf(-d1)
    return np.nan


def calculate_iv(market_price, option_type, spot, strike, time_to_expiry):
    if market_price is None or pd.isna(market_price) or min(market_price, spot, strike, time_to_expiry) <= 0:
        return np.nan
    try:
        iv = brentq(
            lambda vol: bs_price(option_type, spot, strike, time_to_expiry, INTEREST_RATE, vol) - market_price,
            0.001, 5.0, maxiter=200,
        )
        return iv * 100
    except Exception:
        return np.nan


def get_option_price(quote):
    if not quote:
        return np.nan
    depth = quote.get("depth", {})
    buys, sells = depth.get("buy", []), depth.get("sell", [])
    bid = float(buys[0].get("price", 0) or 0) if buys else 0
    ask = float(sells[0].get("price", 0) or 0) if sells else 0
    if bid > 0 and ask >= bid:
        return (bid + ask) / 2
    ltp = quote.get("last_price")
    return float(ltp) if ltp is not None else np.nan


def nearest_expiry(stock_options):
    today = pd.Timestamp.now().normalize()
    expiries = stock_options.loc[stock_options["expiry"] >= today, "expiry"].dropna().sort_values().unique()
    return None if len(expiries) == 0 else pd.Timestamp(expiries[0])


def nearest_strike(target, strikes):
    strikes = [float(x) for x in strikes if pd.notna(x)]
    if not strikes:
        raise ValueError("No valid strikes available.")
    return min(strikes, key=lambda x: abs(x - target))


def iv_richness(ratio):
    if pd.isna(ratio): return "NA"
    if ratio < 0.85: return "Cheap"
    if ratio < 1.05: return "Fair"
    if ratio < 1.20: return "Moderately Rich"
    if ratio < 1.40: return "Rich"
    return "Very Rich"


def volatility_state(ratio):
    if pd.isna(ratio): return "NA"
    if ratio < 0.70: return "Strong Contraction"
    if ratio < 0.90: return "Contracting"
    if ratio <= 1.10: return "Stable"
    if ratio <= 1.30: return "Expanding"
    return "Rapid Expansion"


def get_market_data(quote):
    ltp = float(quote.get("last_price", 0) or 0)
    depth = quote.get("depth", {})
    buys, sells = depth.get("buy", []), depth.get("sell", [])
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
        "coverage": filled / quantity if quantity else 0,
    }


def get_contract(df, strike, option_type):
    result = df[(df["strike"] == strike) & (df["instrument_type"] == option_type)]
    if result.empty:
        raise ValueError(f"No {option_type} contract found for strike {strike}")
    return result.iloc[0]


# -----------------------------
# Scanner
# -----------------------------
def run_scanner(kite, instruments, tickers, progress=None):
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
            if isinstance(equity, pd.DataFrame): equity = equity.iloc[0]
            history = get_history(kite, int(equity["instrument_token"]))
            if history.empty: raise ValueError("No historical price data")
            rv = calculate_rv(history)
            rv5, rv10, rv20, rv60 = rv["RV5"], rv["RV10"], rv["RV20"], rv["RV60"]

            spot_symbol = f"NSE:{ticker}"
            try:
                spot = float(kite.quote([spot_symbol])[spot_symbol]["last_price"])
            except Exception:
                spot = float(history["close"].iloc[-1])

            stock_options = options[options["name"] == ticker].copy()
            if stock_options.empty: raise ValueError("No NFO option contracts found")
            expiry = nearest_expiry(stock_options)
            if expiry is None: raise ValueError("No active expiry found")
            chain = stock_options[stock_options["expiry"] == expiry].copy()
            strikes = np.sort(chain["strike"].dropna().astype(float).unique())
            if not len(strikes): raise ValueError("No strikes")
            atm = float(strikes[np.argmin(np.abs(strikes - spot))])
            call = chain[(chain["strike"] == atm) & (chain["instrument_type"] == "CE")]
            put = chain[(chain["strike"] == atm) & (chain["instrument_type"] == "PE")]
            if call.empty or put.empty: raise ValueError("ATM call/put not found")
            call_symbol = "NFO:" + call.iloc[0]["tradingsymbol"]
            put_symbol = "NFO:" + put.iloc[0]["tradingsymbol"]
            quotes = kite.quote([call_symbol, put_symbol])
            call_price = get_option_price(quotes.get(call_symbol))
            put_price = get_option_price(quotes.get(put_symbol))

            now = pd.Timestamp.now(tz="Asia/Kolkata")
            expiry_time = pd.Timestamp(expiry.date(), tz="Asia/Kolkata") + pd.Timedelta(hours=15, minutes=30)
            seconds_left = (expiry_time - now).total_seconds()
            if seconds_left <= 0: raise ValueError("Selected contract expired")
            t = seconds_left / (365 * 24 * 60 * 60)
            call_iv = calculate_iv(call_price, "CE", spot, atm, t)
            put_iv = calculate_iv(put_price, "PE", spot, atm, t)
            valid = [x for x in [call_iv, put_iv] if pd.notna(x)]
            if not valid: raise ValueError("IV calculation failed")
            iv = float(np.mean(valid))

            iv_rv20 = iv / rv20 if pd.notna(rv20) and rv20 > 0 else np.nan
            iv_minus_rv20 = iv - rv20 if pd.notna(rv20) else np.nan
            rv5_rv20 = rv5 / rv20 if pd.notna(rv5) and pd.notna(rv20) and rv20 > 0 else np.nan
            rv20_rv60 = rv20 / rv60 if pd.notna(rv20) and pd.notna(rv60) and rv60 > 0 else np.nan
            dte = (expiry.normalize() - pd.Timestamp.now().normalize()).days
            results.append({
                "Ticker": ticker, "Spot": round(spot, 2), "Expiry": expiry.date(), "DTE": dte, "ATM": atm,
                "IV": round(iv, 2), "RV5": round(rv5, 2), "RV10": round(rv10, 2), "RV20": round(rv20, 2), "RV60": round(rv60, 2),
                "IV/RV20": round(iv_rv20, 3), "IV-RV20": round(iv_minus_rv20, 2), "RV5/RV20": round(rv5_rv20, 3),
                "RV20/RV60": round(rv20_rv60, 3), "IV State": iv_richness(iv_rv20), "RV State": volatility_state(rv5_rv20),
            })
        except Exception as exc:
            failed.append({"Ticker": ticker, "Reason": str(exc)})
        time.sleep(0.06)

    result_df = pd.DataFrame(results)
    if not result_df.empty:
        result_df = result_df.sort_values("IV/RV20", ascending=False).reset_index(drop=True)
        result_df.insert(0, "Rank", range(1, len(result_df) + 1))
    return result_df, pd.DataFrame(failed)


# -----------------------------
# Iron butterfly
# -----------------------------
def calculate_iron_fly(kite, instruments, symbol, lots, put_wing_pct, call_wing_pct):
    nfo_df = instruments[(instruments["exchange"] == "NFO")].copy()
    nfo_df["expiry"] = pd.to_datetime(nfo_df["expiry"], errors="coerce").dt.date
    stock_options = nfo_df[
        (nfo_df["name"].astype(str).str.upper() == symbol.upper()) &
        (nfo_df["segment"] == "NFO-OPT")
    ].copy()
    if stock_options.empty:
        raise ValueError(f"{symbol} not found in active NFO stock options.")

    today = date.today()
    expiries = sorted(stock_options.loc[stock_options["expiry"] >= today, "expiry"].dropna().unique())
    if not expiries: raise ValueError("No active option expiry found.")
    expiry = expiries[0]
    chain = stock_options[stock_options["expiry"] == expiry].copy()
    ce_strikes = sorted(chain.loc[chain["instrument_type"] == "CE", "strike"].dropna().unique())
    pe_strikes = sorted(chain.loc[chain["instrument_type"] == "PE", "strike"].dropna().unique())
    common = sorted(set(ce_strikes) & set(pe_strikes))
    if not common: raise ValueError("No common CE/PE strikes found.")

    spot_key = f"NSE:{symbol}"
    spot_quote = kite.quote([spot_key]).get(spot_key)
    if not spot_quote: raise ValueError(f"Unable to get NSE quote for {symbol}")
    spot = float(spot_quote["last_price"])
    atm = nearest_strike(spot, common)

    lower_target = spot * (1 - put_wing_pct)
    upper_target = spot * (1 + call_wing_pct)
    lower_candidates = [s for s in pe_strikes if s < atm]
    upper_candidates = [s for s in ce_strikes if s > atm]
    if not lower_candidates or not upper_candidates:
        raise ValueError("No valid wing strikes around ATM.")
    lower_strike = nearest_strike(lower_target, lower_candidates)
    upper_strike = nearest_strike(upper_target, upper_candidates)

    long_put = get_contract(chain, lower_strike, "PE")
    short_put = get_contract(chain, atm, "PE")
    short_call = get_contract(chain, atm, "CE")
    long_call = get_contract(chain, upper_strike, "CE")
    lot_sizes = {int(x["lot_size"]) for x in [long_put, short_put, short_call, long_call]}
    if len(lot_sizes) != 1: raise RuntimeError(f"Inconsistent lot sizes: {lot_sizes}")
    base_lot = lot_sizes.pop()
    qty = base_lot * int(lots)

    quote_symbols = {
        "BUY PE": f"NFO:{long_put['tradingsymbol']}",
        "SELL PE": f"NFO:{short_put['tradingsymbol']}",
        "SELL CE": f"NFO:{short_call['tradingsymbol']}",
        "BUY CE": f"NFO:{long_call['tradingsymbol']}",
    }
    live_quotes = kite.quote(list(quote_symbols.values()))
    market = {leg: get_market_data(live_quotes[key]) for leg, key in quote_symbols.items()}
    execs = {
        "BUY PE": execution_vwap(market["BUY PE"], "BUY", qty),
        "SELL PE": execution_vwap(market["SELL PE"], "SELL", qty),
        "SELL CE": execution_vwap(market["SELL CE"], "SELL", qty),
        "BUY CE": execution_vwap(market["BUY CE"], "BUY", qty),
    }
    used = {}
    for leg in quote_symbols:
        is_buy = leg.startswith("BUY")
        v = execs[leg]["vwap"] if execs[leg]["full_depth"] else market[leg]["best_ask" if is_buy else "best_bid"]
        if v is None or v <= 0: raise ValueError(f"No valid executable price for {leg}.")
        used[leg] = float(v)

    received_ps = used["SELL PE"] + used["SELL CE"]
    paid_ps = used["BUY PE"] + used["BUY CE"]
    net_credit_ps = received_ps - paid_ps
    premium_received = received_ps * qty
    premium_paid = paid_ps * qty
    net_credit = net_credit_ps * qty
    downside_width = atm - lower_strike
    upside_width = upper_strike - atm
    down_loss = max((downside_width - net_credit_ps) * qty, 0)
    up_loss = max((upside_width - net_credit_ps) * qty, 0)
    max_loss = max(down_loss, up_loss, 0)
    max_profit = net_credit
    lower_be = atm - net_credit_ps
    upper_be = atm + net_credit_ps

    basket_orders = []
    for contract, transaction_type in [(long_put, "BUY"), (long_call, "BUY"), (short_put, "SELL"), (short_call, "SELL")]:
        basket_orders.append({
            "exchange": "NFO", "tradingsymbol": contract["tradingsymbol"], "transaction_type": transaction_type,
            "variety": "regular", "product": PRODUCT, "order_type": "MARKET", "quantity": qty, "price": 0, "trigger_price": 0,
        })
    margin_data = kite.basket_order_margins(basket_orders, consider_positions=False, mode="compact")
    funds = float(margin_data.get("initial", {}).get("total", 0) or 0)
    margin = float(margin_data.get("final", {}).get("total", 0) or 0)

    rows = []
    contracts = {"BUY PE": long_put, "SELL PE": short_put, "SELL CE": short_call, "BUY CE": long_call}
    strikes = {"BUY PE": lower_strike, "SELL PE": atm, "SELL CE": atm, "BUY CE": upper_strike}
    for leg in ["BUY PE", "SELL PE", "SELL CE", "BUY CE"]:
        rows.append({
            "Leg": leg, "Strike": strikes[leg], "Contract": contracts[leg]["tradingsymbol"],
            "Bid": market[leg]["best_bid"], "Ask": market[leg]["best_ask"], "LTP": market[leg]["ltp"],
            "Used Price": used[leg], "Visible Qty": execs[leg]["filled_qty"], "Depth Coverage": execs[leg]["coverage"],
        })

    return {
        "updated": datetime.now(), "quote_timestamp": spot_quote.get("timestamp"), "expiry": expiry, "spot": spot,
        "atm": atm, "lower_target": lower_target, "upper_target": upper_target,
        "lower_strike": lower_strike, "upper_strike": upper_strike,
        "put_wing_pct": put_wing_pct, "call_wing_pct": call_wing_pct,
        "base_lot_size": base_lot, "lots": lots, "quantity": qty,
        "premium_received": premium_received, "premium_paid": premium_paid, "net_credit": net_credit,
        "net_credit_ps": net_credit_ps, "max_profit": max_profit, "downside_max_loss": down_loss,
        "upside_max_loss": up_loss, "max_loss": max_loss, "lower_be": lower_be, "upper_be": upper_be,
        "standalone_funds": funds, "standalone_margin": margin,
        "max_profit_on_funds": max_profit / funds * 100 if funds else None,
        "max_profit_on_margin": max_profit / margin * 100 if margin else None,
        "full_depth": all(x["full_depth"] for x in execs.values()), "legs": pd.DataFrame(rows),
    }


# -----------------------------
# Session defaults + one-click login
# -----------------------------
for key, value in {
    "access_token": "", "scanner_results": None,
    "scanner_failed": None, "selected_symbol": "RELIANCE",
}.items():
    st.session_state.setdefault(key, value)

st.title("F&O Volatility + Iron Butterfly Dashboard")
st.caption("Live Zerodha Kite scanner and configurable short iron butterfly monitor")

# API credentials are stored in Streamlit Secrets, never shown in the UI.
try:
    st.session_state.api_key = st.secrets["KITE_API_KEY"]
    api_secret = st.secrets["KITE_API_SECRET"]
except Exception:
    st.error(
        "Kite credentials are not configured. Add KITE_API_KEY and "
        "KITE_API_SECRET in Streamlit Cloud → App settings → Secrets."
    )
    st.stop()

login_kite = KiteConnect(api_key=st.session_state.api_key)

# Zerodha redirects back to this Streamlit app with ?request_token=... .
# Exchange it automatically so the user never has to copy/paste the token.
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
        st.link_button(
            "Login to Zerodha",
            login_kite.login_url(),
            type="primary",
            use_container_width=True,
        )
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
scanner_tab, iron_tab = st.tabs(["① Volatility Scanner", "② Iron Butterfly Builder"])

with scanner_tab:
    top_l, top_r = st.columns([2, 1])
    with top_l:
        st.subheader("Ranked volatility opportunities")
        st.caption("IV/RV20 ranks expensive implied volatility at the top and cheap implied volatility at the bottom.")
    with top_r:
        scan_clicked = st.button("Refresh full scanner", type="primary", use_container_width=True)

    if scan_clicked:
        p = st.progress(0, text="Starting scan…")
        try:
            result_df, failed_df = run_scanner(kite, instruments, tickers, p)
            # Overwrite previous scan — no historical stacking.
            st.session_state.scanner_results = result_df
            st.session_state.scanner_failed = failed_df
            st.session_state.scanner_updated = datetime.now()
        finally:
            p.empty()

    df = st.session_state.scanner_results
    if df is None:
        st.info("Click **Refresh full scanner** to generate the current ranking.")
    elif df.empty:
        st.warning("The latest scan returned no successful stocks.")
    else:
        states = ["All", "Very Rich", "Rich", "Moderately Rich", "Fair", "Cheap"]
        c1, c2, c3 = st.columns([1, 1, 2])
        state_filter = c1.selectbox("IV state", states)
        search = c2.text_input("Ticker contains", placeholder="e.g. RELIANCE").strip().upper()
        sort_direction = c3.radio("Ranking direction", ["Most expensive first", "Cheapest first"], horizontal=True)

        view = df.copy()
        if state_filter != "All": view = view[view["IV State"] == state_filter]
        if search: view = view[view["Ticker"].str.contains(search, na=False)]
        view = view.sort_values("IV/RV20", ascending=(sort_direction == "Cheapest first")).reset_index(drop=True)

        updated = st.session_state.get("scanner_updated")
        if updated: st.caption(f"Latest scanner refresh: {updated:%d-%m-%Y %H:%M:%S}")
        st.dataframe(
            view,
            use_container_width=True,
            hide_index=True,
            column_config={
                "IV": st.column_config.NumberColumn(format="%.2f%%"),
                "RV5": st.column_config.NumberColumn(format="%.2f%%"),
                "RV10": st.column_config.NumberColumn(format="%.2f%%"),
                "RV20": st.column_config.NumberColumn(format="%.2f%%"),
                "RV60": st.column_config.NumberColumn(format="%.2f%%"),
                "IV/RV20": st.column_config.NumberColumn(format="%.3f"),
            },
        )

        st.markdown("**Send a stock to the Iron Butterfly tab**")
        chosen = st.selectbox("Stock", view["Ticker"].tolist() if not view.empty else df["Ticker"].tolist())
        if st.button("Use this stock in Iron Butterfly"):
            st.session_state.selected_symbol = chosen
            st.success(f"{chosen} selected. Open the **Iron Butterfly Builder** tab.")

        failed = st.session_state.scanner_failed
        if failed is not None and not failed.empty:
            with st.expander(f"Failed / missing data ({len(failed)})"):
                st.dataframe(failed, use_container_width=True, hide_index=True)

with iron_tab:
    st.subheader("Configurable short iron butterfly")
    st.caption("The selected stock is monitored with independent put-wing and call-wing distances. Each refresh replaces the previous snapshot.")

    ctl1, ctl2, ctl3, ctl4 = st.columns(4)
    symbol = ctl1.selectbox("Stock", tickers, index=tickers.index(st.session_state.selected_symbol) if st.session_state.selected_symbol in tickers else 0)
    lots = ctl2.number_input("Lots", min_value=1, max_value=100, value=1, step=1)
    put_pct = ctl3.number_input("Put wing below spot (%)", min_value=0.25, max_value=30.0, value=5.0, step=0.25)
    call_pct = ctl4.number_input("Call wing above spot (%)", min_value=0.25, max_value=30.0, value=5.0, step=0.25)

    r1, r2, r3 = st.columns([1, 1, 2])
    auto_refresh = r1.toggle("Auto-refresh", value=True)
    refresh_seconds = r2.selectbox("Refresh interval", [5, 10, 15, 30, 60], index=1, disabled=not auto_refresh)
    manual_refresh = r3.button("Refresh now", type="primary", use_container_width=True)

    def render_iron_fly():
        try:
            result = calculate_iron_fly(kite, instruments, symbol, int(lots), put_pct / 100, call_pct / 100)
        except Exception as exc:
            st.error(str(exc))
            return

        st.caption(f"Updated {result['updated']:%d-%m-%Y %H:%M:%S} · Expiry {result['expiry']} · Quantity {result['quantity']:,}")
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Spot", fmt_money(result["spot"]))
        m2.metric("ATM", f"₹{result['atm']:,.0f}")
        m3.metric("Net credit", fmt_money(result["net_credit"]))
        m4.metric("Max profit", fmt_money(result["max_profit"]))
        m5.metric("Max loss", fmt_money(result["max_loss"]))

        s1, s2, s3, s4 = st.columns(4)
        s1.metric(f"Buy PE ({put_pct:.2f}% target)", f"₹{result['lower_strike']:,.0f}")
        s2.metric("Sell ATM PE", f"₹{result['atm']:,.0f}")
        s3.metric("Sell ATM CE", f"₹{result['atm']:,.0f}")
        s4.metric(f"Buy CE ({call_pct:.2f}% target)", f"₹{result['upper_strike']:,.0f}")

        if result["full_depth"]:
            st.success("Visible market depth covers the requested quantity on all four legs.")
        else:
            st.warning("Visible market depth does not fully cover the requested quantity on at least one leg; execution figures are indicative.")

        st.dataframe(
            result["legs"], use_container_width=True, hide_index=True,
            column_config={
                "Strike": st.column_config.NumberColumn(format="₹%.0f"),
                "Bid": st.column_config.NumberColumn(format="₹%.2f"),
                "Ask": st.column_config.NumberColumn(format="₹%.2f"),
                "LTP": st.column_config.NumberColumn(format="₹%.2f"),
                "Used Price": st.column_config.NumberColumn(format="₹%.2f"),
                "Depth Coverage": st.column_config.ProgressColumn(min_value=0, max_value=1, format="%.0f%%"),
            }
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

        with st.expander("Risk detail"):
            st.write({
                "Downside max loss": fmt_money(result["downside_max_loss"]),
                "Upside max loss": fmt_money(result["upside_max_loss"]),
                "Net credit per share": fmt_money(result["net_credit_ps"]),
                "Put target": fmt_money(result["lower_target"]),
                "Call target": fmt_money(result["upper_target"]),
                "Exchange lot size": f"{result['base_lot_size']:,}",
            })

    if auto_refresh and hasattr(st, "fragment"):
        @st.fragment(run_every=f"{refresh_seconds}s")
        def live_panel():
            render_iron_fly()
        live_panel()
    else:
        render_iron_fly()
