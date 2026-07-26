import math
import yfinance as yf

class ROICalculator:
    @staticmethod
    def project_growth(initial_amount: float, annual_return_pct: float, years: float) -> dict:
        if initial_amount <= 0 or years <= 0:
            raise ValueError("Initial amount and years must be greater than zero.")

        rate = annual_return_pct / 100.0
        future_value = initial_amount * ((1 + rate) ** years)
        total_profit = future_value - initial_amount
        total_roi_pct = (total_profit / initial_amount) * 100.0

        return {
            "initial_amount": round(initial_amount, 2),
            "future_value": round(future_value, 2),
            "total_profit": round(total_profit, 2),
            "total_roi_pct": round(total_roi_pct, 2),
            "multiplier": round(future_value / initial_amount, 2),
            "years": years
        }

    @staticmethod
    def calculate_required_cagr(initial_amount: float, target_amount: float, years: float) -> dict:
        if initial_amount <= 0 or target_amount <= 0 or years <= 0:
            raise ValueError("All inputs must be strictly greater than zero.")

        cagr_decimal = (target_amount / initial_amount) ** (1.0 / years) - 1.0
        cagr_pct = cagr_decimal * 100.0
        total_profit = target_amount - initial_amount
        total_roi_pct = (total_profit / initial_amount) * 100.0

        return {
            "initial_amount": round(initial_amount, 2),
            "target_amount": round(target_amount, 2),
            "required_cagr_pct": round(cagr_pct, 2),
            "total_profit": round(total_profit, 2),
            "total_roi_pct": round(total_roi_pct, 2),
            "multiplier": round(target_amount / initial_amount, 2),
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
            rate = float(input("Expected Annual Return Rate (%): "))
            years = float(input("Time Horizon (Years): "))
            
            res = calc.project_growth(pv, rate, years)
            print(f"\n--- PROJECTION RESULTS ---")
            print(f"Future Value: ${res['future_value']:,}")
            print(f"Total Profit: ${res['total_profit']:,}")
            print(f"Total Return: {res['total_roi_pct']}% ({res['multiplier']}x)")

        elif choice == '2':
            pv = float(input("Initial Investment ($): "))
            target = float(input("Desired End Target ($): "))
            years = float(input("Time Horizon (Years): "))
            
            res = calc.calculate_required_cagr(pv, target, years)
            print(f"\n--- TARGET GOAL RESULTS ---")
            print(f"Required Annual CAGR: {res['required_cagr_pct']}%")
            print(f"Total Gain Needed: ${res['total_profit']:,}")
            print(f"Growth Target: {res['multiplier']}x initial capital")

        elif choice == '3':
            symbol = input("Enter Stock Ticker (e.g., AAPL, SPY): ").strip().upper()
            pv = float(input("Initial Investment ($): "))
            years = float(input("Planned Hold Duration (Years): "))
            hist_years = int(input("Base growth on past N years of history (e.g. 5): "))
            
            print(f"\nFetching live data for {symbol}...")
            try:
                cagr = get_ticker_cagr(symbol, years=hist_years)
                print(f"{symbol}'s past {hist_years}-year historical CAGR: {cagr}%")
                
                res = calc.project_growth(pv, cagr, years)
                print(f"\n--- PROJECTED RESULTS FOR {symbol} ---")
                print(f"Projected Value: ${res['future_value']:,}")
                print(f"Projected Profit: ${res['total_profit']:,}")
                print(f"Total Return: {res['total_roi_pct']}%")
            except Exception as e:
                print(f"\nError: {e}")

        elif choice == '4':
            print("\nExiting calculator. Goodbye!")
            break

        else:
            print("\nInvalid choice! Please select 1, 2, 3, or 4.")

        # Pause so the user can see their results before returning to the main menu
        input("\nPress ENTER to return to the main menu...")

if __name__ == "__main__":
    main()
