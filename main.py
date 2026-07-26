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
        rate = 0.0

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

def get_ticker_cagr(ticker_symbol: str, years: int = 5) -> float:
    stock = yf.Ticker(ticker_symbol)
    hist = stock.history(period=f"{years}y")
    
    if hist.empty or len(hist) < 2:
        raise ValueError(f"Could not fetch historical data for {ticker_symbol}")

    start_price = hist['Close'].iloc[0]
    end_price = hist['Close'].iloc[-1]
    
    cagr = ((end_price / start_price) ** (1.0 / years) - 1.0) * 100.0
    return round(cagr, 2)

def scan_matching_stocks(target_cagr: float, hist_years: int = 5, top_n: int = 5):
    """Scans predefined stock list for tickers meeting/exceeding target CAGR and ranks by stability."""
    print(f"\nScanning market pool for stocks meeting/exceeding ~{target_cagr}% annual growth over the past {hist_years} years...")
    
    results = []

    for symbol in STOCK_POOL:
        try:
            stock = yf.Ticker(symbol)
            hist = stock.history(period=f"{hist_years}y")
            
            if hist.empty or len(hist) < 20:
                continue

            start_price = hist['Close'].iloc[0]
            end_price = hist['Close'].iloc[-1]
            
            actual_cagr = ((end_price / start_price) ** (1.0 / hist_years) - 1.0) * 100.0
            
            if actual_cagr >= target_cagr:
                annual_returns = hist['Close'].resample('YE').ffill().pct_change().dropna()
                volatility = annual_returns.std() * 100 if len(annual_returns) > 1 else 999.0

                results.append({
                    "symbol": symbol,
                    "cagr": round(actual_cagr, 2),
                    "volatility": round(volatility, 2)
                })
        except Exception:
            continue

    if not results:
        print(f"\nNo stocks in the scan pool achieved a historical CAGR of {target_cagr}% or higher over the past {hist_years} years.")
        return

    sorted_results = sorted(results, key=lambda x: x['volatility'])[:top_n]

    print("\n" + "=" * 60)
    print(f" TOP {len(sorted_results)} CONSISTENT STOCKS MATCHING YOUR GOAL ({target_cagr}%+ CAGR)")
    print("=" * 60)
    print(f"{'Ticker':<8} | {'Past CAGR':<12} | {'Volatility (Risk)':<18} | {'Status'}")
    print("-" * 60)

    for item in sorted_results:
        vol_label = "Very Stable" if item['volatility'] < 25 else "Moderate" if item['volatility'] < 45 else "High Growth/Volatile"
        print(f"{item['symbol']:<8} | {item['cagr']}%{' ':<5} | {item['volatility']}%{' ':<12} | {vol_label}")

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
        print("      INVESTMENT ROI CALCULATOR (PYTHON)")
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
                
                # Automatically run the scan using 5 years of historical data
                print("\n" + "-" * 45)
                scan_matching_stocks(target_cagr=res['required_cagr_pct'], hist_years=5, top_n=5)

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
