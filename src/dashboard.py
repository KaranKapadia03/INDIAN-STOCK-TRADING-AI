from pathlib import Path
import subprocess
import sys
import time

import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import yfinance as yf


# ============================================================
# CONFIG
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

RECOMMENDATIONS_FILE = (
    ROOT / "data" / "processed" / "production" / "recommendations.csv"
)

LIVE_ML_FILE = (
    ROOT / "data" / "processed" / "live_ml" / "live_predictions.csv"
)

NEWS_FILE = (
    ROOT / "data" / "processed" / "news" / "live_news_signals.csv"
)

RUN_AI_FILE = ROOT / "src" / "run_ai.py"


# ============================================================
# PAGE
# ============================================================

st.set_page_config(
    page_title="Indian Stock Trading AI",
    page_icon="🇮🇳",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# DARK THEME
# ============================================================

st.markdown(
    """
    <style>
        .stApp {
            background-color: #07111f;
        }

        section[data-testid="stSidebar"] {
            background-color: #081321;
        }

        [data-testid="stMetric"] {
            background-color: #0d1b2d;
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 14px;
            padding: 12px;
        }

        .block-container {
            max-width: 1500px;
            padding-top: 1.5rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# HELPERS
# ============================================================

def safe_float(value, default=0.0):
    try:
        if pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def load_csv(path):

    if not path.exists():
        return pd.DataFrame()

    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def find_column(df, names):

    for name in names:
        if name in df.columns:
            return name

    return None


def get_stock_name(symbol):

    names = {
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

    return names.get(symbol, symbol)


def ticker(symbol):
    return str(symbol).replace(".NS", "")


def signal_icon(signal):

    value = str(signal).upper()

    if "BUY" in value:
        return "🟢"

    if "SELL" in value:
        return "🔴"

    return "🟡"


def signal_color(signal):

    value = str(signal).upper()

    if "BUY" in value:
        return "#34d399"

    if "SELL" in value:
        return "#fb7185"

    return "#fbbf24"


# ============================================================
# LOAD DATA
# ============================================================

recommendations = load_csv(
    RECOMMENDATIONS_FILE
)

ml_predictions = load_csv(
    LIVE_ML_FILE
)

news = load_csv(
    NEWS_FILE
)


if recommendations.empty:

    st.error(
        "No recommendation data found."
    )

    st.code(
        "python src\\run_ai.py"
    )

    st.stop()


# ============================================================
# COLUMN DETECTION
# ============================================================

symbol_col = find_column(
    recommendations,
    [
        "symbol",
        "Symbol",
        "Ticker",
    ],
)

signal_col = find_column(
    recommendations,
    [
        "Signal",
        "Recommendation",
        "Action",
    ],
)

expected_col = find_column(
    recommendations,
    [
        "Expected_Return_20D",
        "Expected_Return",
        "Expected Return",
        "ML_Expected_Return",
    ],
)

confidence_col = find_column(
    recommendations,
    [
        "Confidence",
        "confidence",
        "AI_Confidence",
    ],
)

score_col = find_column(
    recommendations,
    [
        "Final_Score",
        "final_score",
        "AI_Score",
        "Score",
    ],
)

ml_col = find_column(
    recommendations,
    [
        "ML_Prediction",
        "ML_Expected_Return",
        "Predicted_Excess_Return",
        "Expected_Return_Excess",
    ],
)

news_col = find_column(
    recommendations,
    [
        "News_Sentiment",
        "news_sentiment",
    ],
)


if symbol_col is None or signal_col is None:

    st.error(
        "Could not find Symbol or Signal columns."
    )

    st.write(
        recommendations.columns.tolist()
    )

    st.stop()


recommendations["Stock Name"] = (
    recommendations[symbol_col]
    .astype(str)
    .apply(get_stock_name)
)

recommendations["Signal Clean"] = (
    recommendations[signal_col]
    .astype(str)
    .str.upper()
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.title("🇮🇳 Stock AI")

    st.caption(
        "Quantitative Market Intelligence"
    )

    st.divider()

    page = st.radio(
        "Navigation",
        [
            "🏠 AI Terminal",
            "📊 Stock Analysis",
            "🧠 AI Rankings",
            "📈 Backtest",
            "⚙️ System",
        ],
    )

    st.divider()

    st.subheader(
        "⚡ Pipeline"
    )

    if st.button(
        "🔄 Run AI Pipeline",
        width="stretch",
    ):

        with st.spinner(
            "Running AI pipeline..."
        ):

            try:

                result = subprocess.run(
                    [
                        sys.executable,
                        str(RUN_AI_FILE),
                    ],
                    cwd=str(ROOT),
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    timeout=300,
                )

                if result.returncode == 0:

                    st.success(
                        "Pipeline completed."
                    )

                    time.sleep(1)

                    st.rerun()

                else:

                    st.error(
                        "Pipeline failed."
                    )

                    st.code(
                        result.stderr[-3000:]
                    )

            except Exception as exc:

                st.error(
                    str(exc)
                )

    st.divider()

    st.caption(
        "20-stock Indian equity universe"
    )


# ============================================================
# AI TERMINAL
# ============================================================

if page == "🏠 AI Terminal":

    st.title(
        "🇮🇳 Indian Stock Trading AI"
    )

    st.caption(
        "Machine Learning + Technical Analysis + Market Context + News"
    )

    st.success(
        "● AI MARKET INTELLIGENCE ACTIVE"
    )

    # --------------------------------------------------------
    # MARKET SUMMARY
    # --------------------------------------------------------

    total = len(
        recommendations
    )

    buy_count = int(
        recommendations[
            "Signal Clean"
        ]
        .str.contains(
            "BUY",
            na=False,
        )
        .sum()
    )

    sell_count = int(
        recommendations[
            "Signal Clean"
        ]
        .str.contains(
            "SELL",
            na=False,
        )
        .sum()
    )

    hold_count = (
        total
        - buy_count
        - sell_count
    )

    average_confidence = (
        recommendations[
            confidence_col
        ]
        .apply(safe_float)
        .mean()
        if confidence_col
        else 0
    )

    c1, c2, c3, c4, c5 = st.columns(5)

    c1.metric(
        "Stocks Analyzed",
        total,
    )

    c2.metric(
        "🟢 BUY",
        buy_count,
    )

    c3.metric(
        "🟡 HOLD",
        hold_count,
    )

    c4.metric(
        "🔴 SELL",
        sell_count,
    )

    c5.metric(
        "AI Conviction",
        f"{average_confidence:.1f}%",
    )

    st.divider()

    # --------------------------------------------------------
    # OPPORTUNITY
    # --------------------------------------------------------

    st.header(
        "🔥 AI Opportunity of the Day"
    )

    st.caption(
        "Highest-ranked stock from the production recommendation engine."
    )

    ranked = recommendations.copy()

    if score_col:

        ranked["_rank"] = (
            pd.to_numeric(
                ranked[score_col],
                errors="coerce",
            )
            .fillna(0)
        )

    elif confidence_col:

        ranked["_rank"] = (
            pd.to_numeric(
                ranked[confidence_col],
                errors="coerce",
            )
            .fillna(0)
        )

    else:

        ranked["_rank"] = 0

    ranked = ranked.sort_values(
        "_rank",
        ascending=False,
    )

    opportunity = ranked.iloc[0]

    symbol = str(
        opportunity[symbol_col]
    )

    name = str(
        opportunity["Stock Name"]
    )

    signal = str(
        opportunity[signal_col]
    )

    expected = (
        safe_float(
            opportunity[expected_col]
        )
        if expected_col
        else 0
    )

    confidence = (
        safe_float(
            opportunity[confidence_col]
        )
        if confidence_col
        else 0
    )

    score = (
        safe_float(
            opportunity[score_col]
        )
        if score_col
        else 0
    )

    ml_prediction = (
        safe_float(
            opportunity[ml_col]
        )
        if ml_col
        else 0
    )

    left, right = st.columns(
        [1.4, 1]
    )

    with left:

        with st.container(
            border=True
        ):

            st.caption(
                symbol
            )

            st.header(
                name
            )

            if "BUY" in signal.upper():

                st.success(
                    f"🟢 {signal.upper()}"
                )

            elif "SELL" in signal.upper():

                st.error(
                    f"🔴 {signal.upper()}"
                )

            else:

                st.warning(
                    f"🟡 {signal.upper()}"
                )

            st.metric(
                "Model Expected 20D Return",
                f"{expected:+.2f}%",
            )

    with right:

        st.subheader(
            "🧠 AI Signal Breakdown"
        )

        st.metric(
            "AI Conviction",
            f"{confidence:.1f}%",
        )

        st.metric(
            "Final AI Score",
            f"{score:.3f}",
        )

        st.metric(
            "ML Forecast",
            f"{ml_prediction:+.2f}%",
        )

        st.caption(
            "Conviction is signal strength, "
            "not probability of profit."
        )

    # --------------------------------------------------------
    # COMPONENTS
    # --------------------------------------------------------

    st.header(
        "🧠 Why the AI selected this stock"
    )

    a, b, c = st.columns(3)

    with a:

        with st.container(
            border=True
        ):

            st.subheader(
                "🤖 Machine Learning"
            )

            if ml_col:

                value = safe_float(
                    opportunity[ml_col]
                )

                st.metric(
                    "ML Forecast",
                    f"{value:+.2f}%",
                )

                if value > 0:

                    st.success(
                        "Positive model forecast"
                    )

                else:

                    st.warning(
                        "Negative model forecast"
                    )

            else:

                st.info(
                    "ML forecast unavailable."
                )

    with b:

        with st.container(
            border=True
        ):

            st.subheader(
                "📊 Technical Analysis"
            )

            st.metric(
                "Signal Weight",
                "25%",
            )

            st.write(
                "Technical indicators contribute "
                "to the production signal."
            )

    with c:

        with st.container(
            border=True
        ):

            st.subheader(
                "📰 News Intelligence"
            )

            if news_col:

                value = safe_float(
                    opportunity[news_col]
                )

                st.metric(
                    "News Sentiment",
                    f"{value:+.3f}",
                )

                if value > 0:

                    st.success(
                        "Positive recent sentiment"
                    )

                elif value < 0:

                    st.error(
                        "Negative recent sentiment"
                    )

                else:

                    st.info(
                        "Neutral sentiment"
                    )

            else:

                st.metric(
                    "Signal Weight",
                    "25%",
                )

    st.divider()

    # --------------------------------------------------------
    # HEATMAP
    # --------------------------------------------------------

    st.header(
        "🇮🇳 AI Market Heatmap"
    )

    st.caption(
        "All 20 stocks ranked by current AI score."
    )

    heat_columns = st.columns(5)

    for i, (_, row) in enumerate(
        ranked.head(20).iterrows()
    ):

        signal_value = str(
            row[signal_col]
        )

        expected_value = (
            safe_float(
                row[expected_col]
            )
            if expected_col
            else 0
        )

        with heat_columns[
            i % 5
        ]:

            with st.container(
                border=True
            ):

                st.markdown(
                    f"### {signal_icon(signal_value)} "
                    f"{ticker(row[symbol_col])}"
                )

                st.caption(
                    row["Stock Name"]
                )

                st.markdown(
                    f"**{signal_value.upper()}**"
                )

                st.metric(
                    "Expected 20D",
                    f"{expected_value:+.2f}%",
                )

    # --------------------------------------------------------
    # TOP RANKINGS
    # --------------------------------------------------------

    st.divider()

    st.header(
        "🏆 Top AI Rankings"
    )

    ranking_left, ranking_right = st.columns(
        [1.1, 1]
    )

    with ranking_left:

        for rank, (_, row) in enumerate(
            ranked.head(10).iterrows(),
            1,
        ):

            signal_value = str(
                row[signal_col]
            )

            expected_value = (
                safe_float(
                    row[expected_col]
                )
                if expected_col
                else 0
            )

            confidence_value = (
                safe_float(
                    row[confidence_col]
                )
                if confidence_col
                else 0
            )

            with st.container(
                border=True
            ):

                x, y, z = st.columns(
                    [0.15, 0.45, 0.40]
                )

                with x:

                    st.markdown(
                        f"### #{rank}"
                    )

                with y:

                    st.markdown(
                        f"**{ticker(row[symbol_col])}**"
                    )

                    st.caption(
                        row["Stock Name"]
                    )

                with z:

                    st.markdown(
                        f"{signal_icon(signal_value)} "
                        f"**{signal_value.upper()}**"
                    )

                    st.caption(
                        f"{expected_value:+.2f}% • "
                        f"{confidence_value:.1f}% conviction"
                    )

    with ranking_right:

        chart = ranked.copy()

        chart["Expected"] = (
            pd.to_numeric(
                chart[expected_col],
                errors="coerce",
            ).fillna(0)
            if expected_col
            else 0
        )

        chart["Ticker"] = (
            chart[symbol_col]
            .astype(str)
            .str.replace(
                ".NS",
                "",
                regex=False,
            )
        )

        chart = chart.sort_values(
            "Expected"
        )

        fig = go.Figure()

        for _, row in chart.iterrows():

            fig.add_trace(
                go.Bar(
                    x=[
                        row["Expected"]
                    ],
                    y=[
                        row["Ticker"]
                    ],
                    orientation="h",
                    marker_color=signal_color(
                        row[signal_col]
                    ),
                    showlegend=False,
                    hovertemplate=(
                        f"<b>{row['Ticker']}</b><br>"
                        f"Expected: "
                        f"{row['Expected']:+.2f}%"
                        "<extra></extra>"
                    ),
                )
            )

        fig.update_layout(
            title="Expected 20D Return",
            height=520,
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(
                l=10,
                r=20,
                t=55,
                b=20,
            ),
        )

        st.plotly_chart(
            fig,
            width="stretch",
            config={
                "displayModeBar": False
            },
        )

    st.warning(
        "Research disclaimer: model forecasts and AI signals are "
        "not guaranteed returns or financial advice."
    )


# ============================================================
# STOCK ANALYSIS
# ============================================================

elif page == "📊 Stock Analysis":

    st.title(
        "📊 Deep Stock Analysis"
    )

    st.caption(
        "Explore price action, AI forecasts, technical indicators and news."
    )

    symbols = (
        recommendations[symbol_col]
        .astype(str)
        .tolist()
    )

    selected = st.selectbox(
        "Select Stock",
        symbols,
        format_func=lambda x:
            f"{ticker(x)} — {get_stock_name(x)}",
    )

    row = recommendations[
        recommendations[symbol_col].astype(str)
        == str(selected)
    ].iloc[0]

    signal = str(
        row[signal_col]
    )

    st.header(
        f"{ticker(selected)} — {get_stock_name(selected)}"
    )

    a, b, c, d = st.columns(4)

    a.metric(
        "AI Signal",
        f"{signal_icon(signal)} {signal}",
    )

    b.metric(
        "Expected 20D",
        (
            f"{safe_float(row[expected_col]):+.2f}%"
            if expected_col
            else "N/A"
        ),
    )

    c.metric(
        "AI Conviction",
        (
            f"{safe_float(row[confidence_col]):.1f}%"
            if confidence_col
            else "N/A"
        ),
    )

    d.metric(
        "AI Score",
        (
            f"{safe_float(row[score_col]):.3f}"
            if score_col
            else "N/A"
        ),
    )

    tabs = st.tabs(
        [
            "📈 Price",
            "🧠 AI Engine",
            "📊 Technicals",
            "📰 News",
        ]
    )

    # PRICE
    with tabs[0]:

        try:

            history = yf.Ticker(
                selected
            ).history(
                period="1y",
                interval="1d",
                auto_adjust=False,
            )

            if history.empty:

                st.warning(
                    "Price history unavailable."
                )

            else:

                fig = go.Figure()

                fig.add_trace(
                    go.Candlestick(
                        x=history.index,
                        open=history["Open"],
                        high=history["High"],
                        low=history["Low"],
                        close=history["Close"],
                    )
                )

                fig.update_layout(
                    title=f"{ticker(selected)} — 1 Year",
                    height=600,
                    template="plotly_dark",
                    xaxis_rangeslider_visible=False,
                )

                st.plotly_chart(
                    fig,
                    width="stretch",
                )

                latest = safe_float(
                    history["Close"].iloc[-1]
                )

                previous = safe_float(
                    history["Close"].iloc[-2]
                )

                change = (
                    (
                        latest / previous
                        - 1
                    )
                    * 100
                )

                p1, p2, p3 = st.columns(3)

                p1.metric(
                    "Latest Price",
                    f"₹{latest:,.2f}",
                )

                p2.metric(
                    "1D Change",
                    f"{change:+.2f}%",
                )

                p3.metric(
                    "52W High",
                    f"₹{history['High'].max():,.2f}",
                )

        except Exception as exc:

            st.error(
                str(exc)
            )

    # AI
    with tabs[1]:

        st.subheader(
            "🧠 Production AI Engine"
        )

        if ml_col:

            st.metric(
                "ML Expected Excess Return",
                f"{safe_float(row[ml_col]):+.2f}%",
            )

        architecture = pd.DataFrame(
            {
                "Component": [
                    "Machine Learning",
                    "Technical Analysis",
                    "News Sentiment",
                ],
                "Weight": [
                    50,
                    25,
                    25,
                ],
            }
        )

        fig = px.bar(
            architecture,
            x="Weight",
            y="Component",
            orientation="h",
            title="Signal Architecture",
        )

        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )

        st.plotly_chart(
            fig,
            width="stretch",
        )

    # TECHNICALS
    with tabs[2]:

        feature_file = (
            ROOT
            / "data"
            / "processed"
            / f"{selected.replace('.', '_')}_features.csv"
        )

        features = load_csv(
            feature_file
        )

        if features.empty:

            st.warning(
                "Technical features unavailable."
            )

        else:

            latest = features.iloc[-1]

            technicals = [
                "RSI_14",
                "MACD",
                "MACD_Signal",
                "Volatility_20D",
                "Volume_Ratio",
                "Price_vs_SMA20",
                "Price_vs_SMA50",
                "SMA20_vs_SMA50",
                "ATR_Percent",
            ]

            available = [
                x
                for x in technicals
                if x in features.columns
            ]

            cols = st.columns(3)

            for i, feature in enumerate(
                available
            ):

                with cols[i % 3]:

                    st.metric(
                        feature.replace(
                            "_",
                            " ",
                        ),
                        f"{safe_float(latest[feature]):.3f}",
                    )

            if "RSI_14" in features.columns:

                fig = go.Figure()

                fig.add_trace(
                    go.Scatter(
                        y=features["RSI_14"],
                        mode="lines",
                        name="RSI",
                    )
                )

                fig.add_hline(
                    y=70,
                    line_dash="dash",
                )

                fig.add_hline(
                    y=30,
                    line_dash="dash",
                )

                fig.update_layout(
                    title="RSI — 14 Period",
                    height=350,
                    template="plotly_dark",
                )

                st.plotly_chart(
                    fig,
                    width="stretch",
                )

    # NEWS
    with tabs[3]:

        st.subheader(
            "📰 Recent News Intelligence"
        )

        if news.empty:

            st.info(
                "No recent news data available."
            )

        else:

            news_symbol_col = find_column(
                news,
                [
                    "symbol",
                    "Symbol",
                    "Ticker",
                ],
            )

            if news_symbol_col:

                stock_news = news[
                    news[
                        news_symbol_col
                    ].astype(str)
                    == str(selected)
                ]

            else:

                stock_news = news

            if stock_news.empty:

                st.info(
                    "No recent news found."
                )

            else:

                st.dataframe(
                    stock_news,
                    width="stretch",
                    hide_index=True,
                )


# ============================================================
# AI RANKINGS
# ============================================================

elif page == "🧠 AI Rankings":

    st.title(
        "🧠 AI Stock Rankings"
    )

    st.caption(
        "Production model ranking across the complete stock universe."
    )

    ranking = recommendations.copy()

    ranking["Ticker"] = (
        ranking[symbol_col]
        .astype(str)
        .str.replace(
            ".NS",
            "",
            regex=False,
        )
    )

    ranking["AI Score"] = (
        pd.to_numeric(
            ranking[score_col],
            errors="coerce",
        ).fillna(0)
        if score_col
        else 0
    )

    ranking["Expected 20D Return"] = (
        pd.to_numeric(
            ranking[expected_col],
            errors="coerce",
        ).fillna(0)
        if expected_col
        else 0
    )

    ranking["Conviction"] = (
        pd.to_numeric(
            ranking[confidence_col],
            errors="coerce",
        ).fillna(0)
        if confidence_col
        else 0
    )

    ranking = ranking.sort_values(
        "AI Score",
        ascending=False,
    ).reset_index(drop=True)

    ranking["Rank"] = (
        np.arange(
            1,
            len(ranking) + 1,
        )
    )

    display = ranking[
        [
            "Rank",
            "Ticker",
            "Stock Name",
            signal_col,
            "Expected 20D Return",
            "Conviction",
            "AI Score",
        ]
    ]

    st.dataframe(
        display,
        width="stretch",
        hide_index=True,
        column_config={
            "Expected 20D Return":
                st.column_config.NumberColumn(
                    format="%.2f%%"
                ),
            "Conviction":
                st.column_config.NumberColumn(
                    format="%.1f%%"
                ),
            "AI Score":
                st.column_config.NumberColumn(
                    format="%.3f"
                ),
        },
    )

    st.download_button(
        "⬇️ Download Rankings CSV",
        ranking.to_csv(
            index=False
        ),
        file_name="ai_stock_rankings.csv",
        mime="text/csv",
    )


# ============================================================
# BACKTEST
# ============================================================

elif page == "📈 Backtest":

    st.title(
        "📈 Strategy Backtest"
    )

    st.caption(
        "Historical research validation."
    )

    st.info(
        "These results are historical research results, "
        "not live performance."
    )

    a, b, c = st.columns(3)

    a.metric(
        "Portfolio Return",
        "+15.11%",
    )

    b.metric(
        "Maximum Drawdown",
        "-14.56%",
    )

    c.metric(
        "Sharpe Ratio",
        "0.49",
    )

    a, b, c = st.columns(3)

    a.metric(
        "Profit Factor",
        "1.38",
    )

    b.metric(
        "Average Trade",
        "+1.09%",
    )

    c.metric(
        "Strategy",
        "Pure 20D Hold",
    )

    data = pd.DataFrame(
        {
            "Metric": [
                "Return",
                "Drawdown",
                "Sharpe",
                "Profit Factor",
            ],
            "Value": [
                15.11,
                -14.56,
                0.49,
                1.38,
            ],
        }
    )

    fig = px.bar(
        data,
        x="Metric",
        y="Value",
        title="Strategy Validation",
    )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )

    st.plotly_chart(
        fig,
        width="stretch",
    )

    st.warning(
        "Historical performance does not guarantee future results."
    )


# ============================================================
# SYSTEM
# ============================================================

elif page == "⚙️ System":

    st.title(
        "⚙️ AI System"
    )

    st.caption(
        "Production architecture and pipeline health."
    )

    left, right = st.columns(2)

    with left:

        st.subheader(
            "🧠 AI Architecture"
        )

        st.markdown(
            """
            **Market Data**

            Historical OHLCV

            **Technical Engine**

            RSI • MACD • SMA • EMA • volatility • volume

            **Market Context**

            NIFTY 50 + BANK NIFTY

            **News Engine**

            Recent financial news sentiment

            **Machine Learning**

            Universal Random Forest

            **Signal Engine**

            BUY / HOLD / SELL
            """
        )

    with right:

        st.subheader(
            "🎯 Production Model"
        )

        st.markdown(
            """
            **Model:** Random Forest Regressor

            **Prediction Horizon:** 20 trading days

            **Universe:** 20 Indian equities

            **Core Features:**

            - Volatility
            - Volume Ratio
            - NIFTY momentum
            - BANK NIFTY momentum

            **Signal Weights:**

            - ML: 50%
            - Technical: 25%
            - News: 25%
            """
        )

    st.divider()

    st.subheader(
        "📁 Pipeline Status"
    )

    status = []

    components = [
        (
            "Market Data",
            ROOT / "data" / "raw",
        ),
        (
            "Technical Features",
            ROOT / "data" / "processed",
        ),
        (
            "Production Model",
            ROOT / "data" / "models" / "production",
        ),
        (
            "Live Predictions",
            LIVE_ML_FILE,
        ),
        (
            "Recommendations",
            RECOMMENDATIONS_FILE,
        ),
        (
            "News Signals",
            NEWS_FILE,
        ),
    ]

    for name, path in components:

        status.append(
            {
                "Component": name,
                "Status": (
                    "🟢 READY"
                    if path.exists()
                    else "🔴 MISSING"
                ),
            }
        )

    st.dataframe(
        pd.DataFrame(status),
        width="stretch",
        hide_index=True,
    )

    st.warning(
        "The system uses external market/news data providers. "
        "Data freshness can vary."
    )