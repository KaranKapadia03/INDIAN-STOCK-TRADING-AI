from pathlib import Path
from datetime import datetime, timezone
from urllib.parse import quote_plus

import requests
import pandas as pd
from bs4 import BeautifulSoup


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

NEWS_DATA_DIR = (
    PROJECT_ROOT / "data" / "raw" / "news"
)

NEWS_DATA_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# COMPANY NAMES
# ============================================================

COMPANY_NAMES = {
    "RELIANCE.NS": "Reliance Industries",
    "TCS.NS": "Tata Consultancy Services",
    "HDFCBANK.NS": "HDFC Bank",
    "ICICIBANK.NS": "ICICI Bank",
    "INFY.NS": "Infosys",
    "HINDUNILVR.NS": "Hindustan Unilever",
    "ITC.NS": "ITC India",
    "SBIN.NS": "State Bank of India",
    "BHARTIARTL.NS": "Bharti Airtel",
    "KOTAKBANK.NS": "Kotak Mahindra Bank",
    "LT.NS": "Larsen Toubro",
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


# ============================================================
# REQUEST HEADERS
# ============================================================

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 "
        "(Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 "
        "(KHTML, like Gecko) "
        "Chrome/140.0 Safari/537.36"
    )
}


# ============================================================
# GOOGLE NEWS SEARCH
# ============================================================

def search_google_news(company):
    """
    Search Google News RSS.

    Returns all articles available in the RSS response.
    """

    query = quote_plus(
        f"{company} India"
    )

    url = (
        "https://news.google.com/rss/search?"
        f"q={query}"
        "&hl=en-IN"
        "&gl=IN"
        "&ceid=IN:en"
    )

    response = requests.get(
        url,
        headers=HEADERS,
        timeout=15
    )

    response.raise_for_status()

    soup = BeautifulSoup(
        response.content,
        "xml"
    )

    articles = []

    for item in soup.find_all("item"):

        title_element = item.find("title")
        link_element = item.find("link")
        pubdate_element = item.find("pubDate")
        source_element = item.find("source")

        title = (
            title_element.get_text(strip=True)
            if title_element
            else ""
        )

        link = (
            link_element.get_text(strip=True)
            if link_element
            else ""
        )

        published = (
            pubdate_element.get_text(strip=True)
            if pubdate_element
            else ""
        )

        source = (
            source_element.get_text(strip=True)
            if source_element
            else ""
        )

        if not title:
            continue

        articles.append({
            "title": title,
            "link": link,
            "published": published,
            "source": source,
        })

    return articles


# ============================================================
# COLLECT COMPANY NEWS
# ============================================================

def collect_company_news(
    symbol,
    max_articles=100
):
    """
    Collect currently available news for one company.
    """

    company = COMPANY_NAMES.get(
        symbol
    )

    if not company:

        raise ValueError(
            f"Unknown stock symbol: {symbol}"
        )

    print(
        f"Collecting news for "
        f"{symbol} ({company})..."
    )

    articles = search_google_news(
        company
    )

    # --------------------------------------------------------
    # Remove duplicate headlines
    # --------------------------------------------------------

    unique_articles = []

    seen_titles = set()

    for article in articles:

        title = article["title"].strip()

        title_key = title.lower()

        if title_key in seen_titles:
            continue

        seen_titles.add(
            title_key
        )

        unique_articles.append(
            article
        )

    # --------------------------------------------------------
    # Limit results
    # --------------------------------------------------------

    unique_articles = (
        unique_articles[:max_articles]
    )

    if not unique_articles:

        print(
            f"No news found for {symbol}"
        )

        return pd.DataFrame()

    data = pd.DataFrame(
        unique_articles
    )

    # --------------------------------------------------------
    # Preserve raw publication timestamp
    # --------------------------------------------------------

    data["published_raw"] = (
        data["published"]
    )

    # --------------------------------------------------------
    # Parse publication timestamp
    # --------------------------------------------------------

    data["published_timestamp"] = pd.to_datetime(
        data["published"],
        errors="coerce",
        utc=True
    )

    # --------------------------------------------------------
    # Create separate date/time fields
    # --------------------------------------------------------

    data["published_date"] = (
        data["published_timestamp"]
        .dt.tz_convert("Asia/Kolkata")
        .dt.date
    )

    data["published_time"] = (
        data["published_timestamp"]
        .dt.tz_convert("Asia/Kolkata")
        .dt.strftime("%H:%M:%S")
    )

    # --------------------------------------------------------
    # Stock metadata
    # --------------------------------------------------------

    data["symbol"] = symbol

    data["company"] = company

    # --------------------------------------------------------
    # Collection timestamp
    # --------------------------------------------------------

    data["collected_at"] = (
        datetime.now(
            timezone.utc
        ).isoformat()
    )

    # --------------------------------------------------------
    # Sort newest first
    # --------------------------------------------------------

    data.sort_values(
        "published_timestamp",
        ascending=False,
        inplace=True
    )

    return data


# ============================================================
# SAVE NEWS
# ============================================================

def save_news(
    data,
    symbol
):
    """
    Save collected news to CSV.
    """

    if data.empty:
        return None

    filename = (
        symbol.replace(".", "_")
        + "_news.csv"
    )

    filepath = (
        NEWS_DATA_DIR / filename
    )

    data.to_csv(
        filepath,
        index=False
    )

    return filepath


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n" + "=" * 80)
    print(
        "INDIAN STOCK TRADING AI"
    )
    print(
        "NEWS DATA COLLECTION"
    )
    print("=" * 80)

    successful = 0
    failed = 0
    total_articles = 0

    for symbol in COMPANY_NAMES:

        try:

            data = collect_company_news(
                symbol,
                max_articles=100
            )

            filepath = save_news(
                data,
                symbol
            )

            if filepath:

                article_count = len(data)

                total_articles += (
                    article_count
                )

                print(
                    f"Articles: "
                    f"{article_count}"
                )

                print(
                    f"Saved: "
                    f"{filepath}"
                )

                successful += 1

            else:

                print(
                    "No articles collected."
                )

        except Exception as e:

            print(
                f"ERROR - {symbol}: {e}"
            )

            failed += 1

    print("\n" + "=" * 80)
    print(
        "NEWS COLLECTION COMPLETE"
    )
    print("=" * 80)

    print(
        f"Successful: {successful}"
    )

    print(
        f"Failed: {failed}"
    )

    print(
        f"Total stocks: "
        f"{len(COMPANY_NAMES)}"
    )

    print(
        f"Total articles: "
        f"{total_articles}"
    )


if __name__ == "__main__":
    main()