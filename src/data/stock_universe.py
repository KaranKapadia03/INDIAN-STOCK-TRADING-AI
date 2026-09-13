import pandas as pd


# Indian stocks used by the trading AI
# .NS = NSE
# .BO = BSE

INDIAN_STOCKS = {
    "RELIANCE.NS": "Reliance Industries",
    "TCS.NS": "Tata Consultancy Services",
    "HDFCBANK.NS": "HDFC Bank",
    "ICICIBANK.NS": "ICICI Bank",
    "INFY.NS": "Infosys",
    "HINDUNILVR.NS": "Hindustan Unilever",
    "ITC.NS": "ITC",
    "SBIN.NS": "State Bank of India",
    "BHARTIARTL.NS": "Bharti Airtel",
    "KOTAKBANK.NS": "Kotak Mahindra Bank",
    "LT.NS": "Larsen & Toubro",
    "AXISBANK.NS": "Axis Bank",
    "MARUTI.NS": "Maruti Suzuki",
    "SUNPHARMA.NS": "Sun Pharmaceutical",
    "TITAN.NS": "Titan Company",
    "ADANIENT.NS": "Adani Enterprises",
    "ADANIPORTS.NS": "Adani Ports",
    "BAJFINANCE.NS": "Bajaj Finance",
    "ASIANPAINT.NS": "Asian Paints",
    "ULTRACEMCO.NS": "UltraTech Cement",
}


def get_stock_universe():
    """
    Return the stock universe as a pandas DataFrame.
    """

    stocks = pd.DataFrame(
        list(INDIAN_STOCKS.items()),
        columns=["symbol", "company"]
    )

    return stocks


def main():
    stocks = get_stock_universe()

    print("\nIndian Stock Universe")
    print("=" * 40)

    print(stocks)

    print("\nTotal Stocks:", len(stocks))


if __name__ == "__main__":
    main()
    