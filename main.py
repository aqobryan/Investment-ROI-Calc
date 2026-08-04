import math
import numpy as np
import yfinance as yf

# Pool of major stocks & market ETFs to scan efficiently
STOCK_POOL = [
    "SPY", "QQQ", "VOO", "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", 
    "META", "BRK-B", "LLY", "AVGO", "JPM", "TSLA", "V", "MA", 
    "UNH", "COST", "XOM", "HD", "PG", "JNJ", "ABBV", "BAC", "WMT"
]

class ROICalculator:
    @staticmethod
    def project_growth(initial_amount: float, annual_return_pct: float, years: float, annual_contribution: float = 0.0) -> dict:
        if initial_amount <= 0:
            raise ValueError("Investment has to be greater than 0")
        if years <= 0:
            raise ValueError("Years must be greater than zero.")

        rate = annual_return_pct / 100.0
        
        fv_initial = initial_amount * ((1 + rate) ** years)
        
        if rate != 0:
            fv_contributions = annual_contribution * (((1 + rate) ** years - 1) / rate)
        else:
            fv_contributions = annual_contribution * years

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
            raise ValueError("Investment has to be greater than 0")
        if target_amount <= 0 or years <= 0:
            raise ValueError("Target amount and years must be greater than zero.")

        total_invested = initial_amount + (annual_contribution * years)
        if target_amount <= total_invested:
            raise ValueError("Target amount must be greater than total out-of-pocket contributions.")

        # Numerical solver (Binary Search) to solve for required annual interest rate with contributions
        low = -0.999
        high = 10.0  # Max 1000% annual return search ceiling

        for _ in range(100):  # Binary search iterations for high precision
            mid = (low + high) / 2.0
            r = mid
            
            if r != 0:
                fv_calculated = (initial_amount * ((1 + r) ** years)) + (annual_contribution * (((1 + r) ** years - 1) / r))
            else:
                fv_calculated = initial_amount + (annual_contribution * years)

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

def get_ticker_cagr(ticker_symbol: str, years: int = 10) -> float:
    stock = yf.Ticker(ticker_symbol)
    hist = stock.history(period=f"{years}y")
    
    if hist.empty:
        raise ValueError(f"Could not fetch historical data for {ticker_symbol}")

    close_data = hist['Close']
    if hasattr(close_data, 'squeeze'):
        close_data = close_data.squeeze()
    
    close_series = close_data.dropna()

    if close_series.empty or len(close_series) < 2:
        raise ValueError(f"Not enough valid historical price points for {ticker_symbol}")

    start_price = float(close_series.iloc[0])
    end_price = float(close_series.iloc[-1])

    if start_price <= 0:
        raise ValueError(f"Invalid starting price for {ticker_symbol}")

    cagr = ((end_price / start_price) ** (1.0 / years) - 1.0) * 100.0
    return round(cagr, 2)

def scan_matching_stocks(target_cagr: float, initial_amount: float, years: float, annual_contribution: float = 0.0, hist_years: int = 10, top_n: int = 5):
    """Batch scans predefined stock list for tickers meeting/exceeding target CAGR over hist_years (default 10) and calculates portfolio projections."""
    print(f"\nScanning market pool for stocks meeting/exceeding ~{target_cagr}% annual growth over the past {hist_years} years...")
    
    results = []
    calc = ROICalculator()

    try:
        # Batch download all tickers at once to avoid rate limiting
        data = yf.download(STOCK_POOL, period=f"{hist_years}y", progress=False)['Close']
        
        for symbol in STOCK_POOL:
            if symbol not in data:
                continue

            close_series = data[symbol].dropna()

            if close_series.empty or len(close_series) < 20:
                continue

            start_price = float(close_series.iloc[0])
            end_price = float(close_series.iloc[-1])

            if start_price <= 0:
                continue

            actual_cagr = ((end_price / start_price) ** (1.0 / hist_years) - 1.0) * 100.0

            if actual_cagr >= target_cagr:
                # Calculate annualized daily volatility (%) for backend sorting stability
                daily_returns = close_series.pct_change().dropna()
                volatility = float(daily_returns.std() * np.sqrt(252) * 100) if len(daily_returns) > 1 else 999.0

                # Calculate projected performance for this individual stock
                proj = calc.project_growth(initial_amount, actual_cagr, years, annual_contribution)

                results.append({
                    "symbol": symbol,
                    "cagr": round(actual_cagr, 2),
                    "volatility": round(volatility, 2),
                    "future_value": proj['future_value'],
                    "total_profit": proj['total_profit']
                })

    except Exception as e:
        print(f"Market scan error: {e}")

    if not results:
        print(f"\nNo stocks in the scan pool achieved a historical CAGR of {target_cagr}% or higher over the past {hist_years} years.")
        return

    sorted_results = sorted(results, key=lambda x: x['volatility'])[:top_n]

    print("\n" + "=" * 60)
    print(f" TOP {len(sorted_results)} CONSISTENT STOCKS MATCHING YOUR GOAL ({target_cagr}%+ CAGR over past {hist_years} years)")
    print("=" * 60)
    print(f"{'Ticker':<7} | {'10yr AVG CAGR':<13} | {'Projected Value':<17} | {'Net Profit'}")
    print("-" * 60)

    for item in sorted_results:
        print(f"{item['symbol']:<7} | {item['cagr']}%{' ':<7} | ${item['future_value']:<16,} | ${item['total_profit']:,}")

def prompt_contribution() -> float:
    print("\nSelect Contribution Frequency:")
    print("1. Annual")
    print("2. Monthly")
    print("3. Bi-weekly")
    print("4. Weekly")
    print("5. None")
    
    freq_choice = input("Select option (1-5): ").strip()

    if freq_choice == '1':
        amount = float(input("Enter Annual contribution amount ($): "))
        return amount * 1.0
    elif freq_choice == '2':
        amount = float(input("Enter Monthly contribution amount ($): "))
        return amount * 12.0
    elif freq_choice == '3':
        amount = float(input("Enter Bi-weekly contribution amount ($): "))
        return amount * 26.0
    elif freq_choice == '4':
        amount = float(input("Enter Weekly contribution amount ($): "))
        return amount * 52.0
    elif freq_choice == '5':
        return 0.0
    else:
        print("Invalid choice, defaulting to $0 contributions.")
        return 0.0

def main():
    calc = ROICalculator()

    while True:
        print("\n" + "=" * 45)
        print("        INVESTMENT ROI CALCULATOR")
        print("=" * 45)
        print("1. Forward Growth Projection")
        print("2. Find Required CAGR (Target Goal Mode)")
        print("3. Stock Ticker Projection (Live Stock Data)")
        print("4. Exit Program")
        
        choice = input("\nSelect mode (1, 2, 3, or 4): ").strip()

        if choice == '1':
            pv = float(input("Initial Investment ($): "))
            if pv <= 0:
                print("\nError: Investment has to be greater than 0")
                input("\nPress ENTER to return to the main menu...")
                continue

            contrib = prompt_contribution()
            rate = float(input("\nExpected Annual Return Rate (%): "))
            years = float(input("How many years are you wanting to hold it for? "))
            
            try:
                res = calc.project_growth(pv, rate, years, annual_contribution=contrib)
                print(f"\n--- PROJECTION RESULTS ---")
                print(f"Total Invested Out-of-Pocket: ${res['total_invested']:,}")
                print(f"Future Projected Value:       ${res['future_value']:,}")
                print(f"Total Net Profit:             ${res['total_profit']:,}")
                print(f"Total Return (ROI):           {res['total_roi_pct']}% ({res['multiplier']}x)")
                
                # Automatically scan market pool using 10 years of historical data
                print("\n" + "-" * 45)
                scan_matching_stocks(target_cagr=rate, initial_amount=pv, years=years, annual_contribution=contrib, hist_years=10, top_n=5)

            except ValueError as e:
                print(f"\nError: {e}")

        elif choice == '2':
            pv = float(input("Initial Investment ($): "))
            if pv <= 0:
                print("\nError: Investment has to be greater than 0")
                input("\nPress ENTER to return to the main menu...")
                continue

            contrib = prompt_contribution()
            target = float(input("\nDesired End Target ($): "))
            years = float(input("How many years are you wanting to hold it for? "))
            
            try:
                res = calc.calculate_required_cagr(pv, target, years, annual_contribution=contrib)
                print(f"\n--- TARGET GOAL RESULTS ---")
                print(f"Total Invested Out-of-Pocket: ${res['total_invested']:,}")
                print(f"Required Annual CAGR:         {res['required_cagr_pct']}%")
                print(f"Total Gain Needed:            ${res['total_profit']:,}")
                print(f"Growth Target:                {res['multiplier']}x initial capital")
                
                # Automatically run scan using target CAGR and 10 years of historical data
                print("\n" + "-" * 45)
                scan_matching_stocks(target_cagr=res['required_cagr_pct'], initial_amount=pv, years=years, annual_contribution=contrib, hist_years=10, top_n=5)

            except ValueError as e:
                print(f"\nError: {e}")

        elif choice == '3':
            symbol = input("Enter Stock Ticker (e.g., AAPL, SPY): ").strip().upper()
            pv = float(input("Initial Investment ($): "))
            if pv <= 0:
                print("\nError: Investment has to be greater than 0")
                input("\nPress ENTER to return to the main menu...")
                continue

            contrib = prompt_contribution()
            years = float(input("\nHow many years are you wanting to hold it for? "))
            hist_years = int(input("How many years of data should be considered in calculations? "))
            
            print(f"\nFetching live data for {symbol}...")
            try:
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
            print("\nExiting calculator. Goodbye!")
            break

        else:
            print("\nInvalid choice! Please select 1, 2, 3, or 4.")

        input("\nPress ENTER to return to the main menu...")

if __name__ == "__main__":
    main()
