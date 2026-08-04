import math
import numpy as np
import pandas as pd
import yfinance as yf

# Sanitized pool of active, high-liquidity US large/mid-cap stocks and ETFs (with failed tickers removed)
STOCK_POOL = [
    "SPY", "QQQ", "VOO", "IWM", "DIA", "NVDA", "AAPL", "MSFT", "AMZN", "GOOGL", 
    "GOOG", "META", "BRK-B", "LLY", "AVGO", "JPM", "WMT", "V", "XOM", "JNJ", 
    "MA", "UNH", "COST", "HD", "PG", "NFLX", "BAC", "ADBE", "CRM", "AMD", 
    "CVX", "MRK", "ABT", "PEP", "KO", "TMO", "MCD", "DIS", "CSCO", "ACN", 
    "ABNB", "PLTR", "UBER", "INTC", "IBM", "ORCL", "LIN", "PM", "GE", "CAT", 
    "AXP", "AMAT", "BKNG", "ISRG", "TXN", "QCOM", "SPGI", "LOW", "UPS", "RTX", 
    "HON", "COP", "UNP", "DE", "SBUX", "ELV", "BA", "LMT", "MDT", "BLK", 
    "CB", "GILD", "ADI", "MDLZ", "CVS", "TJX", "AMT", "SYK", "CI", "PGR", 
    "REGN", "VRTX", "ZTS", "BSX", "PLD", "NKE", "DUK", "SO", 
    "ITW", "BDX", "EOG", "C", "SLB", "ICE", "NEM", "WM", "SHW", "CL", 
    "MO", "EQIX", "APD", "HUM", "NSC", "ETN", "CSX", "MCK", "PNC", "USB", 
    "TGT", "ORLY", "GD", "ADSK", "MAR", "APH", "MNST", "PH", "MS", "T", 
    "VZ", "PYPL", "CMCSA", "COR", "ROP", "TT", "O", "CTAS", "AON", "ECL", 
    "SRE", "PCG", "KMB", "MSI", "GIS", "XEL", "ED", "DXCM", "ANET", "AEP", 
    "TRV", "AZN", "SNPS", "CDNS", "PANW", "KLAC", "LRCX", "MCHP", "NXPI", "FTNT", 
    "CTSH", "PAYX", "ODFL", "FAST", "ROST", "IDXX", "EA", "TTWO", "FANG", 
    "DVN", "OXY", "HAL", "BKR", "WMB", "KMI", "PSX", "VLO", "MPC", 
    "TRGP", "VICI", "PSA", "SPG", "WELL", "SBAC", "DLR", "EXR", "AVB", "EQR", 
    "MAA", "UDR", "CPT", "ESS", "ARE", "WY", "KIM", "REG", "HST", "KDP", 
    "STZ", "DG", "DLTR", "TSN", "HRL", "MKC", "CAG", "CHD", "CLX", "SYY", 
    "KR", "TAP", "STT", "NTRS", "BEN", "TROW", "AMP", "HIG", 
    "PRU", "MET", "AFL", "ALL", "PCAR", "ROKU", "SNOW", "DDOG", "ZS", "NET", 
    "CRWD", "TEAM", "MDB", "ON", "SWKS", "QRVO", "ENPH", "SEDG", "FSLR", 
    "ZBRA", "TYL", "PTC", "AKAM", "JKHY", "NDAQ", "CME", "MKTX", "CBOE", 
    "COIN", "HOOD", "SOFI", "AFRM", "AXON", "TWLO", "DOCU", "OKTA", "RBLX"
]

class ROICalculator:
    @staticmethod
    def project_growth(initial_amount: float, annual_return_pct: float, years: float, annual_contribution: float = 0.0) -> dict:
        if initial_amount <= 0:
            raise ValueError("Investment must be greater than 0")
        if years <= 0:
            raise ValueError("Years must be greater than zero.")

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
        if initial_amount <= 0:
            raise ValueError("Investment must be greater than 0")
        if target_amount <= 0 or years <= 0:
            raise ValueError("Target amount and years must be greater than zero.")

        total_invested = initial_amount + (annual_contribution * years)
        if target_amount <= total_invested:
            raise ValueError("Target amount must be greater than total out-of-pocket contributions.")

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
    cleaned = sorted({
        t_clean for t in STOCK_POOL 
        if (t_clean := str(t).strip().upper().replace('.', '-')).isalpha() or '-' in t_clean
        if 1 <= len(t_clean) <= 5
    })
    return cleaned

def get_ticker_cagr(ticker_symbol: str, years: int = 10) -> float:
    hist = yf.Ticker(ticker_symbol).history(period=f"{years}y")
    
    if hist.empty or 'Close' not in hist.columns:
        raise ValueError(f"Could not fetch historical data for {ticker_symbol} (may be invalid or delisted)")

    close_series = hist['Close'].dropna()
    if len(close_series) < 2:
        raise ValueError(f"Not enough valid historical price points for {ticker_symbol}")

    start_price, end_price = float(close_series.iloc[0]), float(close_series.iloc[-1])
    if start_price <= 0:
        raise ValueError(f"Invalid starting price for {ticker_symbol}")

    return round((((end_price / start_price) ** (1.0 / years)) - 1.0) * 100.0, 2)

def scan_matching_stocks(target_cagr: float, initial_amount: float, years: float, annual_contribution: float = 0.0, hist_years: int = 10, top_n: int = 5):
    stock_universe = get_expanded_stock_universe()
    print(f"\nScanning {len(stock_universe)} active US market stocks for returns >= ~{target_cagr}% CAGR over past {hist_years} years...")
    
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
            if 'Close' in raw_data.columns.levels[0]:
                data = raw_data['Close']
            else:
                data = raw_data.xs('Close', level=0, axis=1, drop_level=True)
        else:
            data = raw_data.get('Close', pd.DataFrame())

        if data.empty:
            print("No data received from market source.")
            return

        monthly_data = data.resample('ME').last()
        
        min_obs = int(hist_years * 10)
        valid_data = monthly_data.dropna(thresh=min_obs, axis=1)

        if valid_data.empty:
            print("No US stocks found with sufficient historical data.")
            return

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

    except Exception as e:
        print(f"Error during scan: {e}")
        return

    if not results:
        print(f"\nNo US stocks achieved a historical CAGR of {target_cagr}%+ over the past {hist_years} years.")
        return

    sorted_results = sorted(results, key=lambda x: x['volatility'])[:top_n]

    print("\n" + "=" * 68)
    print(f" TOP {len(sorted_results)} CONSISTENT US STOCKS MATCHING YOUR GOAL ({target_cagr}%+ CAGR)")
    print("=" * 68)
    print(f"{'Ticker':<7} | {'10yr CAGR':<10} | {'Ann. Volatility':<16} | {'Projected Value':<16} | {'Net Profit'}")
    print("-" * 68)

    for item in sorted_results:
        print(f"{item['symbol']:<7} | {item['cagr']}%{' ':<5} | {item['volatility']}%{' ':<10} | ${item['future_value']:<15,} | ${item['total_profit']:,}")

def prompt_contribution() -> float:
    print("\nSelect Contribution Frequency:")
    print("1. Annual\n2. Monthly\n3. Bi-weekly\n4. Weekly\n5. None")
    
    multipliers = {'1': 1.0, '2': 12.0, '3': 26.0, '4': 52.0, '5': 0.0}
    freq_choice = input("Select option (1-5): ").strip()

    if freq_choice in multipliers:
        if freq_choice == '5':
            return 0.0
        try:
            amount = float(input("Enter contribution amount ($): "))
            return max(0.0, amount * multipliers[freq_choice])
        except ValueError:
            print("Invalid amount entered, defaulting to $0.")
            return 0.0
            
    print("Invalid choice, defaulting to $0 contributions.")
    return 0.0

def get_positive_float(prompt_text: str) -> float:
    while True:
        try:
            val = float(input(prompt_text))
            if val <= 0:
                print("Error: Value must be greater than 0.")
                continue
            return val
        except ValueError:
            print("Error: Please enter a valid number.")

def main():
    calc = ROICalculator()

    while True:
        print("\n" + "=" * 45)
        print("    US MARKET INVESTMENT ROI CALCULATOR")
        print("=" * 45)
        print("1. Forward Growth Projection")
        print("2. Find Required CAGR (Target Goal Mode)")
        print("3. Stock Ticker Projection (Live US Stock Data)")
        print("4. Exit Program")
        
        choice = input("\nSelect mode (1, 2, 3, or 4): ").strip()

        if choice == '1':
            try:
                pv = get_positive_float("Initial Investment ($): ")
                contrib = prompt_contribution()
                rate = float(input("Expected Annual Return Rate (%): "))
                years = get_positive_float("How many years are you wanting to hold it for? ")
                
                res = calc.project_growth(pv, rate, years, annual_contribution=contrib)
                print(f"\n--- PROJECTION RESULTS ---")
                print(f"Total Invested Out-of-Pocket: ${res['total_invested']:,}")
                print(f"Future Projected Value:        ${res['future_value']:,}")
                print(f"Total Net Profit:              ${res['total_profit']:,}")
                print(f"Total Return (ROI):            {res['total_roi_pct']}% ({res['multiplier']}x)")
                
                scan_choice = input("\nWould you like to scan US stocks to match this APR/CAGR? (y/n): ").strip().lower()
                if scan_choice in ['y', 'yes']:
                    print("\n" + "-" * 45)
                    scan_matching_stocks(target_cagr=rate, initial_amount=pv, years=years, annual_contribution=contrib, hist_years=10, top_n=5)
            except Exception as e:
                print(f"\nError: {e}")

        elif choice == '2':
            try:
                pv = get_positive_float("Initial Investment ($): ")
                contrib = prompt_contribution()
                target = get_positive_float("Desired End Target ($): ")
                years = get_positive_float("How many years are you wanting to hold it for? ")
                
                res = calc.calculate_required_cagr(pv, target, years, annual_contribution=contrib)
                print(f"\n--- TARGET GOAL RESULTS ---")
                print(f"Total Invested Out-of-Pocket: ${res['total_invested']:,}")
                print(f"Required Annual CAGR:         {res['required_cagr_pct']}%")
                print(f"Total Gain Needed:            ${res['total_profit']:,}")
                print(f"Growth Target:                {res['multiplier']}x initial capital")
                
                print("\n" + "-" * 45)
                scan_matching_stocks(target_cagr=res['required_cagr_pct'], initial_amount=pv, years=years, annual_contribution=contrib, hist_years=10, top_n=5)
            except Exception as e:
                print(f"\nError: {e}")

        elif choice == '3':
            try:
                symbol = input("Enter US Stock Ticker (e.g., AAPL, MSFT): ").strip().upper()
                pv = get_positive_float("Initial Investment ($): ")
                contrib = prompt_contribution()
                years = get_positive_float("How many years are you wanting to hold it for? ")
                hist_years = int(get_positive_float("How many years of historical data should be considered? "))
                
                print(f"\nFetching live data for {symbol}...")
                cagr = get_ticker_cagr(symbol, years=hist_years)
                print(f"{symbol}'s past {hist_years}-year historical CAGR: {cagr}%")
                
                res = calc.project_growth(pv, cagr, years, annual_contribution=contrib)
                print(f"\n--- PROJECTED RESULTS FOR {symbol} ---")
                print(f"Total Invested Out-of-Pocket: ${res['total_invested']:,}")
                print(f"Projected Value:              ${res['future_value']:,}")
                print(f"Projected Profit:             ${res['total_profit']:,}")
                print(f"Total Return (ROI):           {res['total_roi_pct']}%")
            except Exception as e:
                print(f"\nError: {e}")

        elif choice == '4':
            print("\nExiting program. Goodbye!")
            break

        else:
            print("\nInvalid choice! Please select 1, 2, 3, or 4.")

        input("\nPress ENTER to return to the main menu...")

if __name__ == "__main__":
    main()
