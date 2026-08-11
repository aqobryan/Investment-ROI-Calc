import math
import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st

# Page Configuration
st.set_page_config(
    page_title="US Market Investment ROI Dashboard",
    page_icon="📈",
    layout="wide"
)

STOCK_POOL = [
    "SPY", "QQQ", "VOO", "IWM", "DIA", "NVDA", "AAPL", "MSFT", "AMZN", "GOOGL",
    "GOOG", "META", "BRK-B", "LLY", "AVGO", "JPM", "WMT", "V", "XOM", "JNJ",
    "MA", "UNH", "COST", "HD", "PG", "NFLX", "BAC", "ADBE", "CRM", "AMD",
    "CVX", "MRK", "ABT", "PEP", "KO", "TMO", "MCD", "DIS", "CSCO", "ACN",
    "ABNB", "PLTR", "UBER", "INTC", "IBM", "ORCL", "LIN", "PM", "GE", "CAT",
    "AXP", "AMAT", "BKNG", "ISRG", "TXN", "QCOM", "SPGI", "LOW", "UPS", "RTX",
    "HON", "COP", "UNP", "DE", "SBUX", "ELV", "BA", "LMT", "MDT", "BLK",
    "CB", "GILD", "ADI", "MDLZ", "CVS", "TJX", "AMT", "SYK", "CI", "PGR",
    "REGN", "VRTX", "ZTS", "BSX", "PLD", "NKE", "DUK", "SO", "ITW", "BDX",
    "EOG", "C", "SLB", "ICE", "NEM", "WM", "SHW", "CL", "MO", "EQIX",
    "APD", "HUM", "NSC", "ETN", "CSX", "MCK", "PNC", "USB", "TGT", "ORLY",
    "GD", "ADSK", "MAR", "APH", "MNST", "PH", "MS", "T", "VZ", "PYPL",
    "CMCSA", "COR", "ROP", "TT", "O", "CTAS", "AON", "ECL", "SRE", "PCG",
    "KMB", "MSI", "GIS", "XEL", "ED", "DXCM", "ANET", "AEP", "TRV", "AZN",
    "SNPS", "CDNS", "PANW", "KLAC", "LRCX", "MCHP", "NXPI", "FTNT", "CTSH", "PAYX",
    "ODFL", "FAST", "ROST", "IDXX", "EA", "TTWO", "FANG", "DVN", "OXY", "HAL",
    "BKR", "WMB", "KMI", "PSX", "VLO", "MPC", "TRGP", "VICI", "PSA", "SPG",
    "WELL", "SBAC", "DLR", "EXR", "AVB", "EQR", "MAA", "UDR", "CPT", "ESS",
    "ARE", "WY", "KIM", "REG", "HST", "KDP", "STZ", "DG", "DLTR", "TSN",
    "HRL", "MKC", "CAG", "CHD", "CLX", "SYY", "KR", "TAP", "STT", "NTRS",
    "BEN", "TROW", "AMP", "HIG", "PRU", "MET", "AFL", "ALL", "PCAR", "ROKU",
    "SNOW", "DDOG", "ZS", "NET", "CRWD", "TEAM", "MDB", "ON", "SWKS", "QRVO",
    "ENPH", "SEDG", "FSLR", "ZBRA", "TYL", "PTC", "AKAM", "JKHY", "NDAQ", "CME",
    "MKTX", "CBOE", "COIN", "HOOD", "SOFI", "AFRM", "AXON", "TWLO", "DOCU", "OKTA",
    "RBLX", "TSLA", "ARM", "MU", "SHOP", "SPOT", "MELI", "PATH", "RKLB",
    "TEM", "CELH", "ANF", "DUOL", "APP", "RDDT", "PINS", "SNAP", "BABA", "PDD"
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

@st.cache_data
def get_ticker_cagr(ticker_symbol: str, years: int = 10) -> float:
    hist = yf.Ticker(ticker_symbol).history(period=f"{years}y")
    if hist.empty or 'Close' not in hist.columns:
        raise ValueError(f"Could not fetch historical data for {ticker_symbol}")
    close_series = hist['Close'].dropna()
    if len(close_series) < 2:
        raise ValueError(f"Not enough valid historical price points for {ticker_symbol}")
    start_price, end_price = float(close_series.iloc[0]), float(close_series.iloc[-1])
    return round((((end_price / start_price) ** (1.0 / years)) - 1.0) * 100.0, 2)

@st.cache_data
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


# --- UI LAYOUT ---
st.title("📈 US Market Investment ROI Dashboard")
st.markdown("Interactive portfolio growth projections, target goal calculators, and live US market stock scanning.")

# Sidebar Configuration
st.sidebar.header("⚙️ Configuration Controls")
mode = st.sidebar.selectbox(
    "Select Mode", 
    ["Forward Growth Projection", "Find Required CAGR (Target)", "Stock Ticker Projection"]
)

st.sidebar.markdown("---")
st.sidebar.subheader("💰 Financial Parameters")
pv = st.sidebar.number_input("Initial Investment ($)", min_value=1.0, value=10000.0, step=1000.0)

freq_label = st.sidebar.selectbox("Contribution Frequency", ["None", "Annual", "Monthly", "Bi-weekly", "Weekly"])
freq_multipliers = {"None": 0.0, "Annual": 1.0, "Monthly": 12.0, "Bi-weekly": 26.0, "Weekly": 52.0}

contrib_amount = 0.0
if freq_label != "None":
    contrib_amount = st.sidebar.number_input("Contribution Amount ($)", min_value=0.0, value=500.0, step=100.0)

annual_contrib = contrib_amount * freq_multipliers[freq_label]
years = st.sidebar.number_input("Holding Period (Years)", min_value=0.5, value=10.0, step=1.0)

calc = ROICalculator()

# --- MODE 1 ---
if mode == "Forward Growth Projection":
    st.subheader("📊 Forward Growth Projection")
    rate = st.sidebar.slider("Expected Annual Return Rate (%)", min_value=-10.0, max_value=50.0, value=8.0, step=0.5)
    
    res = calc.project_growth(pv, rate, years, annual_contribution=annual_contrib)
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Invested", f"${res['total_invested']:,.2f}")
    col2.metric("Future Value", f"${res['future_value']:,.2f}", f"{res['total_roi_pct']}%")
    col3.metric("Net Profit", f"${res['total_profit']:,.2f}")
    col4.metric("Growth Multiple", f"{res['multiplier']}x")

    st.markdown("---")
    if st.checkbox("🔍 Scan US Market for stocks matching this return rate"):
        with st.spinner("Scanning active stock universe..."):
            stock_results = scan_matching_stocks_cached(rate, pv, years, annual_contrib, hist_years=10, top_n=5)
            if stock_results:
                st.success(f"Found top matching stocks with >= {rate}% historical CAGR:")
                df_stocks = pd.DataFrame(stock_results)
                df_stocks.columns = ["Ticker", "10yr CAGR (%)", "Ann. Volatility (%)", "Projected Value ($)", "Net Profit ($)"]
                st.dataframe(df_stocks, use_container_width=True)
            else:
                st.warning("No stocks matched the criteria with sufficient historical data.")

# --- MODE 2 ---
elif mode == "Find Required CAGR (Target)":
    st.subheader("🎯 Target Goal Calculator")
    target = st.number_input("Desired End Target Value ($)", min_value=100.0, value=100000.0, step=5000.0)
    
    try:
        res = calc.calculate_required_cagr(pv, target, years, annual_contribution=annual_contrib)
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Invested", f"${res['total_invested']:,.2f}")
        col2.metric("Required CAGR", f"{res['required_cagr_pct']}%")
        col3.metric("Total Gain Needed", f"${res['total_profit']:,.2f}")
        col4.metric("Growth Target", f"{res['multiplier']}x")

        st.markdown("---")
        st.subheader("🔍 Matching Low-Volatility Stocks for your Target")
        with st.spinner("Scanning US stocks matching required CAGR..."):
            stock_results = scan_matching_stocks_cached(res['required_cagr_pct'], pv, years, annual_contrib, hist_years=10, top_n=5)
            if stock_results:
                df_stocks = pd.DataFrame(stock_results)
                df_stocks.columns = ["Ticker", "10yr CAGR (%)", "Ann. Volatility (%)", "Projected Value ($)", "Net Profit ($)"]
                st.dataframe(df_stocks, use_container_width=True)
            else:
                st.warning("No stocks historically matched or exceeded this required CAGR.")
    except Exception as e:
        st.error(f"Error: {e}")

# --- MODE 3 ---
elif mode == "Stock Ticker Projection":
    st.subheader("🏷️ Live US Stock Historical Projection")
    symbol = st.sidebar.text_input("US Stock Ticker", value="AAPL").strip().upper()
    hist_years = st.sidebar.slider("Historical Years to Analyze", min_value=1, max_value=20, value=10)

    if st.sidebar.button("Run Projection"):
        with st.spinner(f"Fetching historical data for {symbol}..."):
            try:
                cagr = get_ticker_cagr(symbol, years=hist_years)
                res = calc.project_growth(pv, cagr, years, annual_contribution=annual_contrib)
                
                st.info(f"**{symbol}** achieved a past **{hist_years}-year historical CAGR of {cagr}%**.")
                
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Total Invested", f"${res['total_invested']:,.2f}")
                col2.metric("Projected Value", f"${res['future_value']:,.2f}", f"{res['total_roi_pct']}%")
                col3.metric("Projected Profit", f"${res['total_profit']:,.2f}")
                col4.metric("Return Multiplier", f"{res['multiplier']}x")
            except Exception as e:
                st.error(f"Error processing {symbol}: {e}")
                st.error(f"Error processing {symbol}: {e}")
