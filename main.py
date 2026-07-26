import math
import yfinance as yf

class ROICalculator:
    @staticmethod
    def project_growth(initial_amount: float, annual_return_pct: float, years: float, annual_contribution: float = 0.0) -> dict:
        """
        Calculates future value using initial amount, annual return rate, years,
        and annual equivalent contribution.
        """
        if initial_amount <= 0:
            raise ValueError("Investment has to be greater than 0")
        if years <= 0:
            raise ValueError("Years must be greater than zero.")

        rate = annual_return_pct / 100.0
        
        # Future Value of initial lump sum: PV * (1 + r)^n
        fv_initial = initial_amount * ((1 + rate) ** years)
        
        # Future Value of annual contributions (Annuity Formula): PMT * (((1 + r)^n - 1) / r)
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
    def calculate_required_cagr(initial_amount: float, target_amount: float, years: float) -> dict:
        if initial_amount <= 0:
            raise ValueError("Investment has to be greater than 0")
        if target_amount <= 0 or years <= 0:
            raise ValueError("Target amount and years must be greater than zero.")

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

def prompt_contribution() -> float:
    """Displays contribution frequency menu and returns equivalent annual contribution."""
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

            target = float(input("Desired End Target ($): "))
            years = float(input("How many years are you wanting to hold it for? "))
            
            try:
                res = calc.calculate_required_cagr(pv, target, years)
                print(f"\n--- TARGET GOAL RESULTS ---")
                print(f"Required Annual CAGR: {res['required_cagr_pct']}%")
                print(f"Total Gain Needed:    ${res['total_profit']:,}")
                print(f"Growth Target:        {res['multiplier']}x initial capital")
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
