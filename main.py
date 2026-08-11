import math
import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import plotly.express as px

# Page Configuration
st.set_page_config(
    page_title="US Market Investment ROI Dashboard",
    page_icon="📈",
    layout="wide"
)

# Balanced 100-Ticker Pool (Stable Growth + Aggressive Growth)
STOCK_POOL = [
    # Core ETFs
    "SPY", "QQQ", "VOO", "IWM", "DIA", 
    # Aggressive Growth & Tech Leaders
    "NVDA", "AAPL", "MSFT", "AMZN", "GOOGL", "META", "TSLA", "NFLX", "AMD", "AVGO",
    "PLTR", "CRM", "ADBE", "NOW", "INTU", "UBER", "ABNB", "SNOW", "DDOG", "CRWD",
    "QCOM", "TXN", "AMAT", "LRCX", "MU", "PANW", "NET", "ZS", "MELI", "SHOP",
    # Stable Growth & Blue Chips
    "BRK-B", "JPM", "V", "MA", "BAC", "WFC", "GS", "MS", "AXP", "BLK",
    "LLY", "UNH", "JNJ", "ABBV", "MRK", "TMO", "ABT", "PFE", "AMGN", "ISRG",
    "WMT", "COST", "HD", "PG", "KO", "PEP", "MCD", "DIS", "NKE", "SBUX",
    "CAT", "UNP", "GE", "RTX", "LMT", "HON", "BA", "DE", "UPS", "ADP",
    "XOM", "CVX", "COP", "SLB", "EOG", "OXY", "PSX", "VLO", "NEE", "SO",
    "PLD", "AMT", "EQIX", "CCI", "SPG", "LIN", "SHW", "APD", "NEM", "FCX",
    "T", "VZ", "CMCSA", "TMUS", "IBM", "ORCL", "ADI", "KLAC", "SYK", "ZTS",
    "MDLZ", "CVS", "TJX", "LOW", "SPGI", "ICE", "CB", "PGR", "REGN", "VRTX"
]

class ROICalculator:
    @staticmethod
    def project_growth(initial_amount: float, annual_return_pct: float, years: float, annual_contribution: float = 0.0) -> dict:
        rate = annual_return_pct / 100.0
        fv_initial = initial_amount * ((1 + rate) ** years)
        fv_contributions = (
            annual_contribution * (((1 + rate) ** years - 1) / rate)
            if rate != 0
            else annual_contribution * years
        )
        future_value = fv_initial + fv_contributions
        total_invested = initial_amount + (annual_contribution * years)
        total_profit = future_value - total_invested
        total_roi_pct = (total_profit / total_invested) * 100.0 if total_invested > 0 else 0.0

        return {
            "initial_amount": round(initial_amount, 2),
            "total_contributions": round(annual_contribution * years, 2),
            "total_invested": round(total_invested, 2),
            "future_value": round(future_value, 2),
            "total_profit": round(total_profit, 2),
            "total_roi_pct": round(total_roi_pct, 2),
            "multiplier": round(future_value / total_invested, 2) if total_invested > 0 else 0.0,
            "years": years
        }

    @staticmethod
    def calculate_required_cagr(initial_amount: float, target_amount: float, years: float, annual_contribution: float = 0.0) -> dict:
        total_invested = initial_amount + (annual_contribution * years)
        low, high = -0.999, 10.0  
        mid = 0.0

        for _ in range(100):  
            mid = (low + high) / 2.0
            fv_calculated = (
                (initial_amount * ((1 + mid) ** years)) + (annual_contribution * (((1 + mid) ** years - 1) / mid))
                if mid != 0
                else initial_amount + (annual_contribution * years)
            )
            if fv_calculated < target_amount:
                low = mid
            else:
                high = mid

        cagr_pct = mid * 100.0
        total_profit = target_amount - total_invested
        total_roi_pct = (total_profit / total_invested) * 100.0

        return {
            "initial_amount": round(initial_amount, 2),
            "annual_contribution": round(annual_contribution, 2),
            "total_invested": round(total_invested, 2),
            "target_amount": round(target_amount, 2),
            "required_cagr_pct": round(cagr_pct, 2),
            "total_profit": round(total_profit, 2),
            "total_roi_pct": round(total_roi_pct, 2),
            "multiplier": round(target_amount / total_invested, 2),
            "years": years
        }

def get_expanded_stock_universe() -> list:
    return sorted({
        t_clean for t in STOCK_POOL 
        if (t_clean := str(t).strip().upper().replace('.', '-')).isalpha() or '-' in t_clean
        if 1 <= len(t_clean) <= 5
    })

@st.cache_data(ttl=3600)
def get_ticker_cagr(ticker_symbol: str, years: int = 10) -> float:
    hist = yf.Ticker(ticker_symbol).history(period=f"{years}y")
    if hist.empty or 'Close' not in hist.columns:
        raise ValueError(f"Could not fetch historical data for {ticker_symbol}")
    close_series = hist['Close'].dropna()
    if len(close_series) < 2:
        raise ValueError(f"Not enough valid historical price points for {ticker_symbol}")
    start_price, end_price = float(close_series.iloc[0]), float(close_series.iloc[-1])
    return round((((end_price / start_price) ** (1.0 / years)) - 1.0) * 100.0, 2)

@st.cache_data(ttl=3600)
def scan_matching_stocks_cached(target_cagr: float, initial_amount: float, years: float, annual_contribution: float, hist_years: int, top_n: int):
    stock_universe = get_expanded_stock_universe()
    results = []
    calc = ROICalculator()

    try:
        raw_data = yf.download(
            tickers=stock_universe, 
            period=f"{hist_years}y", 
            interval="1d", 
            threads=True, 
            progress=False,
            multi_level_index=True
        )
        
        if isinstance(raw_data.columns, pd.MultiIndex):
            data = raw_data['Close'] if 'Close' in raw_data.columns.levels[0] else raw_data.xs('Close', level=0, axis=1, drop_level=True)
        else:
            data = raw_data.get('Close', pd.DataFrame())

        if data.empty:
            return []

        monthly_data = data.resample('ME').last()
        valid_data = monthly_data.dropna(thresh=int(hist_years * 10), axis=1)

        if valid_data.empty:
            return []

        start_prices = valid_data.bfill().iloc[0]
        end_prices = valid_data.ffill().iloc[-1]

        cagrs = (((end_prices / start_prices) ** (1.0 / hist_years)) - 1.0) * 100.0
        volatilities = valid_data.pct_change().std() * np.sqrt(12) * 100.0

        matching = cagrs[cagrs >= target_cagr].index

        for symbol in matching:
            actual_cagr = float(cagrs[symbol])
            volatility = float(volatilities[symbol])
            if np.isnan(actual_cagr) or np.isnan(volatility):
                continue
            proj = calc.project_growth(initial_amount, actual_cagr, years, annual_contribution)
            results.append({
                "symbol": symbol,
                "cagr": round(actual_cagr, 2),
                "volatility": round(volatility, 2),
                "future_value": proj['future_value'],
                "total_profit": proj['total_profit']
            })
    except Exception:
        pass

    if not results:
        return []

    return sorted(results, key=lambda x: x['volatility'])[:top_n]

def generate_growth_timeline(initial_amount, annual_return_pct, years, annual_contribution):
    rate = annual_return_pct / 100.0
    timeline_data = []
    current_val = initial_amount
    total_inv = initial_amount
    
    full_years = int(math.ceil(years))
    for yr in range(full_years + 1):
        if yr == 0:
            timeline_data.append({"Year": 0, "Portfolio Value": round(initial_amount, 2), "Total Invested": round(initial_amount, 2)})
            continue
        
        fraction = min(1.0, years - (yr - 1))
        current_val = (current_val + (annual_contribution * fraction)) * ((1 + rate) ** fraction)
        total_inv += (annual_contribution * fraction)
        
        timeline_data.append({
            "Year": round(yr if fraction == 1.0 else years, 1), 
            "Portfolio Value": round(current_val, 2), 
            "Total Invested": round(total_inv, 2)
        })
    return pd.DataFrame(timeline_data)


# --- MAIN APP LAYOUT ---
st.title("📈 US Market Investment ROI Dashboard")
st.markdown("Interactive portfolio tracking, compound growth charts, and automated stock universe scanning.")

# Global Sidebar Parameters
st.sidebar.header("⚙️ Financial Inputs")
pv = st.sidebar.number_input("Initial Investment ($)", min_value=1.0, value=10000.0, step=1000.0)

freq_label = st.sidebar.selectbox("Contribution Frequency", ["None", "Annual", "Monthly", "Bi-weekly", "Weekly"])
freq_multipliers = {"None": 0.0, "Annual": 1.0, "Monthly": 12.0, "Bi-weekly": 26.0, "Weekly": 52.0}

contrib_amount = 0.0
if freq_label != "None":
    contrib_amount = st.sidebar.number_input("Contribution Amount ($)", min_value=0.0, value=500.0, step=100.0)

annual_contrib = contrib_amount * freq_multipliers[freq_label]
years = st.sidebar.number_input("Holding Period (Years)", min_value=0.5, value=10.0, step=0.5)

calc = ROICalculator()

tab1, tab2, tab3 = st.tabs(["📊 Growth Projection & Chart", "🎯 Target Goal Calculator", "🏷️ Live Stock Scanner"])

# --- TAB 1: FORWARD GROWTH ---
with tab1:
    st.subheader("Forward Growth Projection & Compounding Curve")
    rate = st.slider("Expected Annual Return Rate (%)", min_value=1.0, max_value=50.0, value=8.0, step=0.5)
    
    res = calc.project_growth(pv, rate, years, annual_contribution=annual_contrib)
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Invested", f"${res['total_invested']:,.2f}")
    col2.metric("Future Value", f"${res['future_value']:,.2f}", f"{res['total_roi_pct']}%")
    col3.metric("Net Profit", f"${res['total_profit']:,.2f}")
    col4.metric("Growth Multiple", f"{res['multiplier']}x")

    df_timeline = generate_growth_timeline(pv, rate, years, annual_contrib)
    fig = px.area(
        df_timeline, x="Year", y=["Portfolio Value", "Total Invested"],
        labels={"value": "Amount ($)", "variable": "Metric"},
        title="Portfolio Growth Trajectory Over Time"
    )
    fig.update_layout(hovermode="x unified", template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("🏆 Top US Stocks Matching This Return Rate")
    st.markdown(f"The following low-volatility stocks from your pool historically matched or exceeded a **{rate}% CAGR** over the past 10 years:")

    with st.spinner("Scanning stock pool..."):
        stock_results = scan_matching_stocks_cached(rate, pv, years, annual_contrib, hist_years=10, top_n=5)
        if stock_results:
            df_stocks = pd.DataFrame(stock_results)
            df_stocks.columns = ["Ticker", "10yr CAGR (%)", "Ann. Volatility (%)", "Projected Value ($)", "Net Profit ($)"]
            st.dataframe(df_stocks.style.background_gradient(subset=["10yr CAGR (%)"], cmap="Greens"), use_container_width=True)
        else:
            st.warning("No stocks in the pool met or exceeded this specific return rate with sufficient history.")

# --- TAB 2: TARGET GOAL ---
with tab2:
    st.subheader("Target Goal & Required CAGR Finder")
    target = st.number_input("Desired End Target Value ($)", min_value=100.0, value=100000.0, step=5000.0)
    
    try:
        res = calc.calculate_required_cagr(pv, target, years, annual_contribution=annual_contrib)
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Invested", f"${res['total_invested']:,.2f}")
        col2.metric("Required CAGR", f"{res['required_cagr_pct']}%")
        col3.metric("Total Gain Needed", f"${res['total_profit']:,.2f}")
        col4.metric("Growth Target", f"{res['multiplier']}x")

        df_timeline = generate_growth_timeline(pv, res['required_cagr_pct'], years, annual_contrib)
        fig = px.line(
            df_timeline, x="Year", y=["Portfolio Value", "Total Invested"],
            markers=True, title=f"Path to Reach ${target:,.2f} at {res['required_cagr_pct']}% CAGR"
        )
        fig.update_layout(hovermode="x unified", template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")
        st.subheader("🏆 Top US Stocks Matching This Required CAGR")
        st.markdown(f"The following low-volatility stocks from your pool historically achieved at least a **{res['required_cagr_pct']}% CAGR** over the past 10 years:")

        with st.spinner("Scanning stock pool for required target..."):
            target_stock_results = scan_matching_stocks_cached(res['required_cagr_pct'], pv, years, annual_contrib, hist_years=10, top_n=5)
            if target_stock_results:
                df_target_stocks = pd.DataFrame(target_stock_results)
                df_target_stocks.columns = ["Ticker", "10yr CAGR (%)", "Ann. Volatility (%)", "Projected Value ($)", "Net Profit ($)"]
                st.dataframe(df_target_stocks.style.background_gradient(subset=["10yr CAGR (%)"], cmap="Greens"), use_container_width=True)
            else:
                st.warning("No stocks in the pool met or exceeded this specific high required CAGR over the past 10 years.")

    except Exception as e:
        st.error(f"Error: {e}")

# --- TAB 3: STOCK SCANNER & TICKER PROJECTION ---
with tab3:
    st.subheader("Live US Stock Scanner & Ticker Analysis")
    
    col_a, col_b = st.columns(2)
    with col_a:
        scan_cagr = st.slider("Minimum Historical CAGR Threshold (%)", 1.0, 30.0, 10.0, 0.5)
        if st.button("Scan Stock Universe"):
            with st.spinner("Analyzing market data across stock pool..."):
                stock_results = scan_matching_stocks_cached(scan_cagr, pv, years, annual_contrib, hist_years=10, top_n=8)
                if stock_results:
                    st.success("Top matching low-volatility performers found:")
                    df_stocks = pd.DataFrame(stock_results)
                    df_stocks.columns = ["Ticker", "10yr CAGR (%)", "Ann. Volatility (%)", "Projected Value ($)", "Net Profit ($)"]
                    st.dataframe(df_stocks.style.background_gradient(subset=["10yr CAGR (%)"], cmap="Greens"), use_container_width=True)
                else:
                    st.warning("No stocks matched the criteria.")
                    
    with col_b:
        specific_ticker = st.text_input("Analyze Specific Ticker Symbol", value="AAPL").strip().upper()
        hist_years_input = st.slider("Historical Lookback (Years)", 1, 20, 10)
        if st.button("Run Ticker Analysis"):
            with st.spinner(f"Fetching {specific_ticker} data..."):
                try:
                    cagr = get_ticker_cagr(specific_ticker, years=hist_years_input)
                    t_res = calc.project_growth(pv, cagr, years, annual_contribution=annual_contrib)
                    st.info(f"**{specific_ticker}** achieved a **{cagr}% CAGR** over the last {hist_years_input} years.")
                    st.metric("Projected Portfolio Value", f"${t_res['future_value']:,.2f}", f"{t_res['total_roi_pct']}%")
                except Exception as e:
                    st.error(f"Could not analyze {specific_ticker}: {e}")
