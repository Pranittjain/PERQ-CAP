# F&O Volatility + Iron Butterfly Dashboard

Streamlit dashboard using Zerodha Kite for a volatility scanner and configurable short iron butterfly monitor.

## Employer workflow

1. Open the Streamlit app URL.
2. Click **Login to Zerodha**.
3. Complete Zerodha authentication/TOTP.
4. Zerodha redirects back to the dashboard automatically.
5. Use **Volatility Scanner** to rank stocks and **Iron Butterfly Builder** for live strategy stats.

No API key, API secret, request token, terminal, or Python command is shown to the end user.

## Deploy on Streamlit Community Cloud

### 1. Upload this folder to GitHub

The repository should contain:

- `app.py`
- `requirements.txt`
- `FNO_INDIA_TICKERS.xlsx`
- `.gitignore`
- `.streamlit/secrets.toml.example`
- `README.md`

Do **not** upload a real `.streamlit/secrets.toml` file.

### 2. Deploy from GitHub

In Streamlit Community Cloud, create a new app from the GitHub repository and set the main file to:

`app.py`

### 3. Add Streamlit Secrets

In the Streamlit app settings, open **Secrets** and paste:

```toml
KITE_API_KEY = "YOUR_KITE_API_KEY"
KITE_API_SECRET = "YOUR_KITE_API_SECRET"
```

Save the settings.

### 4. Set the Kite redirect URL

After Streamlit gives you the final app address, for example:

`https://your-dashboard.streamlit.app`

open your Kite Connect developer app and set its **Redirect URL** to that exact Streamlit app URL.

This is what allows Zerodha to send the user back to the dashboard with the temporary `request_token`. The app exchanges that token automatically for the day's access token.

### 5. Use it

Open the Streamlit URL and click **Login to Zerodha**. After successful Zerodha authentication, you should be returned directly to the live dashboard.

## Dashboard features

### Volatility Scanner

- RV5 / RV10 / RV20 / RV60
- ATM IV
- IV/RV20 ranking
- Cheap / Fair / Moderately Rich / Rich / Very Rich filters
- Volatility-state ratios
- New scans replace the previous scan
- Send a selected stock directly to the Iron Butterfly tab

### Iron Butterfly Builder

- Independent put-wing % and call-wing %
- Nearest actual listed strikes
- Number of lots
- Live bid / ask / LTP
- Market-depth-aware execution estimate
- Net credit
- Premium received / paid
- Max profit / loss
- Breakevens
- Zerodha standalone funds and final hedged margin
- Return on funds / margin
- Auto-refresh with the latest snapshot replacing the old one

## Security

Never commit your Kite API secret or daily access token to GitHub. The included `.gitignore` excludes a local `.streamlit/secrets.toml` file.

If the API credentials previously embedded in your notebooks were real, rotate the Kite API secret before deploying this app.
