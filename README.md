# F&O Volatility + Iron Butterfly Dashboard

Streamlit dashboard using Zerodha Kite.

## Tabs
1. **IV / RV Scanner** — IV, RV20, IV/RV20 and IV-RV20 only. No Cheap/Fair/Rich labels.
2. **All Iron Butterflies** — fast universe-wide butterfly table with configurable put/call wing percentages and 5-second auto-refresh.
3. **Individual Butterfly** — full single-stock detail including broker funds/margin and risk metrics.
4. **Watchlist** — live butterfly table for selected stocks.

## IV methodology
- RV20 = standard deviation of the latest 20 daily log returns × sqrt(252), shown as an annualized percentage.
- IV requires live bid and ask for both ATM CE and PE.
- Quotes with relative bid/ask spread above 30% are excluded from the IV ranking.
- The forward is inferred from ATM put-call parity.
- CE and PE implied vols are solved with Black-76 and averaged.
- Stocks with unreliable quotes are excluded and shown in the diagnostics/failed section.

## Streamlit Cloud secrets
Add these in **App settings → Secrets**:

```toml
KITE_API_KEY = "your_key"
KITE_API_SECRET = "your_secret"
```

Set your Kite Connect redirect URL to your deployed Streamlit app URL.

## Important performance note
The **All Iron Butterflies** tab batches market quotes so the full universe can update quickly. It deliberately does not call Zerodha's basket-margin endpoint for every stock every 5 seconds. Broker **Standalone Funds** and **Final Hedged Margin** remain available in the **Individual Butterfly** tab.

## Files
- `app.py`
- `requirements.txt`
- `FNO_INDIA_TICKERS.xlsx`
- `.gitignore`
