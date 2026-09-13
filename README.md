# 🇮🇳 Indian Stock Trading AI

> **An end-to-end machine-learning research and decision-support system
> for Indian equities.**

Indian Stock Trading AI combines **market data, technical indicators,
NIFTY 50 and BANK NIFTY context, financial news intelligence, machine
learning, stock ranking, strategy research, backtesting, and an
interactive Streamlit dashboard** into one complete workflow.

The project is designed around a practical question:

> **Can machine learning and multiple market signals be combined into a
> systematic, testable process for ranking Indian stocks and generating
> BUY / HOLD / SELL signals?**

------------------------------------------------------------------------

## 🚀 Project at a Glance

  -----------------------------------------------------------------------
  Area                                Implementation
  ----------------------------------- -----------------------------------
  Stock Universe                      20 Indian equities

  Market Data                         Yahoo Finance / `yfinance`

  Frequency                           Daily market data

  Technical Analysis                  RSI, MACD, SMA, EMA, ATR, Bollinger
                                      Bands, volatility, volume

  Market Context                      NIFTY 50 + BANK NIFTY

  News                                Recent financial news via Google
                                      News RSS

  ML Model                            Random Forest Regressor

  Prediction Horizon                  20 trading days

  ML Target                           Stock 20D return relative to NIFTY

  Signal Engine                       ML + Technical + News

  Final Weights                       ML 50% / Technical 25% / News 25%

  Output                              BUY / HOLD / SELL

  Interface                           Streamlit

  Visualization                       Plotly

  Validation                          Walk-forward and portfolio
                                      backtesting
  -----------------------------------------------------------------------

------------------------------------------------------------------------

# 🎯 Why I Built This

Financial markets contain a huge amount of information.

For a single stock, an investor may want to consider:

-   recent price movement
-   momentum
-   volatility
-   trading volume
-   technical indicators
-   broader market direction
-   banking-sector direction
-   company-specific news
-   expected future return
-   relative performance versus the market

Processing all of this manually across 20 stocks is difficult to do
consistently.

This project automates that workflow.

Instead of relying on one indicator such as RSI or MACD, the system
combines multiple information sources and produces a structured output
that can be inspected, tested, and backtested.

The project is intentionally built as a **research system**, not as a
claim that machine learning can perfectly predict financial markets.

------------------------------------------------------------------------

# 🧠 Core Idea

The production system follows this pipeline:

**Market Data**\
↓\
**Feature Engineering**\
↓\
**Market Context**\
↓\
**News Intelligence**\
↓\
**Machine Learning Forecast**\
↓\
**Cross-Sectional Stock Ranking**\
↓\
**Recommendation Engine**\
↓\
**BUY / HOLD / SELL**\
↓\
**Interactive Dashboard**

------------------------------------------------------------------------

# 🏗️ System Architecture

``` text
┌─────────────────────────────┐
│      Indian Equities        │
│        20 Stocks             │
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│       Market Data Layer      │
│                              │
│  OHLCV • Price • Volume      │
│  Historical + Latest Data    │
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│    Technical Feature Layer   │
│                              │
│ RSI • MACD • SMA • EMA       │
│ ATR • Bollinger • Volatility │
│ Returns • Volume             │
└──────────────┬──────────────┘
               │
       ┌───────┴────────┐
       ▼                ▼
┌──────────────┐  ┌────────────────┐
│ NIFTY 50     │  │ BANK NIFTY     │
│ Market       │  │ Market         │
│ Context      │  │ Context        │
└──────┬───────┘  └───────┬────────┘
       │                  │
       └────────┬─────────┘
                ▼
┌─────────────────────────────┐
│       News Intelligence      │
│                              │
│ Headlines • Sentiment        │
│ Recency • News Intensity     │
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│     Production ML Model      │
│                              │
│ Random Forest Regressor      │
│ 20-Day Excess Return         │
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│   Cross-Sectional Ranking     │
│                              │
│ Strongest → Weakest Stocks   │
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│     Recommendation Engine     │
│                              │
│ ML          50%              │
│ Technical   25%              │
│ News        25%              │
└──────────────┬──────────────┘
               │
               ▼
        ┌──────┼──────┐
        ▼      ▼      ▼
      BUY    HOLD    SELL
               │
               ▼
┌─────────────────────────────┐
│      Streamlit Dashboard     │
│                              │
│ Terminal • Analysis          │
│ Rankings • Backtest • System │
└─────────────────────────────┘
```

------------------------------------------------------------------------

# 📁 Project Structure

``` text
INDIAN-STOCK-TRADING-AI/
│
├── data/
│   ├── raw/
│   │   ├── news/
│   │   ├── nifty50.csv
│   │   └── banknifty.csv
│   │
│   ├── processed/
│   │   ├── news/
│   │   └── *_features.csv
│   │
│   ├── models/
│   │   └── production/
│   │
│   └── database/
│
├── notebooks/
│
├── src/
│   ├── main.py
│   ├── run_ai.py
│   │
│   ├── api/
│   │
│   ├── data/
│   │   ├── stock_universe.py
│   │   ├── market_data.py
│   │   ├── news_data.py
│   │   └── market_context.py
│   │
│   ├── features/
│   │   ├── technical_indicators.py
│   │   ├── build_features.py
│   │   ├── add_market_features.py
│   │   ├── news_sentiment.py
│   │   ├── build_news_features.py
│   │   ├── add_news_features.py
│   │   ├── live_news_signal.py
│   │   └── add_market_regime.py
│   │
│   ├── models/
│   │   ├── create_target.py
│   │   ├── train_model.py
│   │   ├── train_all_models.py
│   │   ├── train_regression_models.py
│   │   ├── train_production_model.py
│   │   ├── live_predictions.py
│   │   └── research / validation scripts
│   │
│   ├── strategy/
│   │   ├── technical_signal.py
│   │   ├── combined_signal.py
│   │   ├── recommendation_engine.py
│   │   ├── final_engine.py
│   │   ├── risk_engine.py
│   │   ├── portfolio_backtest.py
│   │   └── research / backtest scripts
│   │
│   └── dashboard.py
│
├── tests/
├── .env
├── .gitignore
├── requirements.txt
└── README.md
```

Generated data, model artifacts, secrets, and local environments are
excluded from version control.

------------------------------------------------------------------------

# 📊 Stock Universe

The current system covers 20 Indian equities:

  Symbol            Company
  ----------------- ---------------------------
  `RELIANCE.NS`     Reliance Industries
  `TCS.NS`          Tata Consultancy Services
  `HDFCBANK.NS`     HDFC Bank
  `ICICIBANK.NS`    ICICI Bank
  `INFY.NS`         Infosys
  `HINDUNILVR.NS`   Hindustan Unilever
  `ITC.NS`          ITC
  `SBIN.NS`         State Bank of India
  `BHARTIARTL.NS`   Bharti Airtel
  `KOTAKBANK.NS`    Kotak Mahindra Bank
  `LT.NS`           Larsen & Toubro
  `AXISBANK.NS`     Axis Bank
  `MARUTI.NS`       Maruti Suzuki
  `SUNPHARMA.NS`    Sun Pharmaceutical
  `TITAN.NS`        Titan Company
  `ADANIENT.NS`     Adani Enterprises
  `ADANIPORTS.NS`   Adani Ports
  `BAJFINANCE.NS`   Bajaj Finance
  `ASIANPAINT.NS`   Asian Paints
  `ULTRACEMCO.NS`   UltraTech Cement

The architecture is designed so the universe can be expanded later.

------------------------------------------------------------------------

# 1. 📥 Market Data Layer

## `src/data/stock_universe.py`

This file defines the production stock universe and maps ticker symbols
to company names.

Keeping the universe centralized prevents different parts of the
pipeline from using different stock lists.

------------------------------------------------------------------------

## `src/data/market_data.py`

Downloads daily historical market data.

The main fields are:

-   Open
-   High
-   Low
-   Close
-   Volume

The project uses approximately five years of historical daily data for
model development and research.

The downloaded datasets are written to the local `data/raw/` directory.

------------------------------------------------------------------------

## `src/data/market_context.py`

Downloads broader Indian market data and calculates market-context
features for:

-   NIFTY 50
-   BANK NIFTY

These features help the model understand the broader market environment
instead of looking at each stock in isolation.

------------------------------------------------------------------------

# 2. 📐 Technical Feature Engineering

## `src/features/technical_indicators.py`

Raw OHLCV data is transformed into a larger technical feature set.

### Trend Features

``` text
SMA_20
SMA_50
EMA_20
Price_vs_SMA20
Price_vs_SMA50
SMA20_vs_SMA50
```

### Momentum Features

``` text
Return_1D
Return_5D
Return_10D
Return_20D
RSI_14
MACD
MACD_Signal
MACD_Histogram
```

### Volatility Features

``` text
Volatility_20D
ATR_14
ATR_Percent
```

### Bollinger Bands

``` text
BB_Middle
BB_Upper
BB_Lower
BB_Position
```

### Volume Features

``` text
Volume_SMA_20
Volume_Ratio
```

The resulting feature files are stored as:

``` text
data/processed/<SYMBOL>_features.csv
```

------------------------------------------------------------------------

# 3. 🌐 Market Context

The system adds broader market momentum:

``` text
NIFTY_Return_5D
NIFTY_Return_20D

BANKNIFTY_Return_5D
BANKNIFTY_Return_20D
```

This allows the model to distinguish between:

``` text
Stock-specific movement
```

and:

``` text
Broader market movement
```

That distinction became especially important during model research.

------------------------------------------------------------------------

# 4. 📰 News Intelligence

## `src/data/news_data.py`

The system collects recent financial headlines using Google News RSS.

News collection includes company-related financial news and produces
information such as:

-   headline
-   source
-   publication time
-   URL

### Important design decision

The news engine is designed for **recent/live news**.

It is not a historical news database.

Therefore, live news is **not used as a historical feature in the
backtests**.

This prevents the system from accidentally using information that would
not have been available at the historical prediction time.

------------------------------------------------------------------------

## `src/features/news_sentiment.py`

Converts headlines into sentiment information.

------------------------------------------------------------------------

## `src/features/live_news_signal.py`

Aggregates recent company news into stock-level signals.

The resulting data can contain:

``` text
News_Count
News_Sentiment
Positive_News_Count
Negative_News_Count
Neutral_News_Count
News_Intensity
News_Recency_Score
News_Signal
News_Label
```

This becomes one of the inputs to the live recommendation engine.

------------------------------------------------------------------------

# 5. 🎯 Target Engineering

One of the most important changes during the research process was
changing the prediction problem.

An early approach focused on:

``` text
Will the stock go UP or DOWN?
```

The research showed that this was weak.

The production problem became:

``` text
20D Stock Return - 20D NIFTY Return
```

This is an **excess-return target**.

The model is therefore attempting to answer:

> **Which stocks are likely to outperform or underperform the broader
> market over the next 20 trading days?**

This is better aligned with a cross-sectional stock-ranking strategy.

------------------------------------------------------------------------

# 6. 🤖 Machine Learning Research

The project tested multiple approaches rather than selecting the first
model that produced an attractive backtest.

Research included:

-   binary classification
-   regression
-   trade-quality classification
-   calibrated models
-   ensemble models
-   different prediction horizons
-   stock-specific models
-   universal models
-   cross-sectional ranking
-   walk-forward validation
-   feature ablation
-   stock-universe analysis
-   execution assumptions
-   exit architectures
-   portfolio-level validation

The purpose was to identify where the model actually provided useful
information.

------------------------------------------------------------------------

# 7. 🔬 What the Research Found

## Short-Term Classification

The initial 5-day binary classifier was weak:

  Metric           Result
  -------------- --------
  Accuracy         49.01%
  Baseline         50.69%
  UP Precision     52.20%
  UP Recall        45.65%
  ROC-AUC           0.534

The model did not provide strong evidence that simple short-term
direction classification was useful.

------------------------------------------------------------------------

## Initial Regression

The first larger regression model also struggled:

  Metric                    Result
  ---------------------- ---------
  MAE                      \~3.17%
  RMSE                     \~3.99%
  R²                       \~-0.51
  Directional Accuracy     \~49.8%
  Correlation               \~0.09

This was another reason not to treat model complexity as proof of
predictive power.

------------------------------------------------------------------------

# 8. ⏱️ Horizon Analysis

The research tested several forward horizons.

Approximate historical average forward returns were:

  Horizon     Mean Return   Win Rate
  --------- ------------- ----------
  1D              +0.041%     50.79%
  3D              +0.126%     51.54%
  5D              +0.216%     52.82%
  10D             +0.440%     53.77%
  20D             +0.896%     54.75%

The 20-day horizon became the production focus.

This does **not** mean that 20-day returns are predictable with
certainty. It means the research provided a stronger basis for
investigating that horizon.

------------------------------------------------------------------------

# 9. 🏆 Production Model

## `src/models/train_production_model.py`

The production model is a:

> **Random Forest Regressor**

Configuration:

``` text
Trees:             400
Maximum Depth:       8
Minimum Split:      15
Minimum Leaf:        5
Max Features:      sqrt
Random State:       42
```

### Production Features

The final locked feature set contains six features:

``` text
Volatility_20D
Volume_Ratio
NIFTY_Return_5D
NIFTY_Return_20D
BANKNIFTY_Return_5D
BANKNIFTY_Return_20D
```

### Target

``` text
Target_Excess_Return_20D
```

where:

``` text
Target = Future 20D Stock Return - NIFTY 20D Return
```

------------------------------------------------------------------------

# 10. 🧠 Why a Small Feature Set?

The research process compared larger and smaller feature groups.

A key finding was:

> More features did not automatically produce a better trading model.

The final production architecture therefore uses a compact feature set
containing:

-   stock volatility
-   stock volume behavior
-   NIFTY momentum
-   BANK NIFTY momentum

This makes the production model easier to inspect and reduces
unnecessary complexity.

------------------------------------------------------------------------

# 11. 📈 Live ML Predictions

## `src/models/live_predictions.py`

The live prediction layer:

1.  Loads the production Random Forest.
2.  Loads the latest feature data.
3.  Selects the exact production features.
4.  Generates the 20-day excess-return prediction.
5.  Ranks all stocks.
6.  Saves the results.

Outputs:

``` text
data/processed/live_ml/live_predictions.csv
data/processed/live_ml/live_predictions.json
```

The model output is primarily used as a **relative ranking signal**.

------------------------------------------------------------------------

# 12. 🎯 Recommendation Engine

## `src/strategy/production_engine.py`

The recommendation engine combines three components:

``` text
Machine Learning       50%
Technical Analysis     25%
News Intelligence      25%
```

Conceptually:

``` text
               ML Forecast
                   │
                   ▼
          ┌─────────────────┐
          │                 │
Technical │ Recommendation  │ News
─────────►│     Engine      │◄─────────
          │                 │
          └────────┬────────┘
                   │
          ┌────────┼────────┐
          ▼        ▼        ▼
        BUY      HOLD      SELL
```

The engine also produces:

-   expected 20D return
-   final AI score
-   conviction
-   ML forecast
-   recommendation

### Important

The displayed **conviction score is signal strength**.

It is **not a calibrated probability of profit**.

For example:

``` text
Conviction = 70%
```

does not mean:

``` text
70% probability of making money
```

------------------------------------------------------------------------

# 13. ⚡ One-Command Production Pipeline

## `src/run_ai.py`

The complete pipeline can be executed using:

``` bash
python src/run_ai.py
```

The script runs:

``` text
1. Live Market Data
        ↓
2. Technical Feature Generation
        ↓
3. Market Context
        ↓
4. Live News
        ↓
5. ML Predictions
        ↓
6. AI Recommendations
```

This turns multiple independent scripts into a single repeatable
workflow.

------------------------------------------------------------------------

# 14. 🖥️ Interactive Dashboard

## `src/dashboard.py`

The Streamlit dashboard is the presentation layer of the system.

It currently contains five main sections.

------------------------------------------------------------------------

## 🏠 AI Terminal

The main screen provides a high-level view of the current model state.

It includes:

-   number of stocks analyzed
-   BUY count
-   HOLD count
-   SELL count
-   top AI score
-   AI opportunity of the day
-   expected 20D return
-   AI conviction
-   ML forecast
-   Top 3 opportunities
-   20-stock AI heatmap
-   ranking charts
-   expected-return charts

The purpose is to answer:

> **What does the production system currently see?**

------------------------------------------------------------------------

## 📊 Stock Analysis

Users can select an individual stock.

The page combines:

### Price Analysis

-   latest price
-   daily change
-   52-week high
-   52-week low
-   one-year candlestick chart
-   SMA20
-   SMA50

### AI Engine

-   ML forecast
-   AI score
-   conviction
-   signal architecture

### Technical Analysis

-   RSI
-   MACD
-   volatility
-   volume ratio
-   price vs SMA20
-   price vs SMA50
-   SMA20 vs SMA50
-   ATR percentage
-   RSI chart
-   volume chart

### News Intelligence

-   recent company news
-   available sentiment information

------------------------------------------------------------------------

# 🧠 AI Rankings

The ranking page focuses on stock selection.

It provides:

-   🥇 Top-ranked opportunity
-   🥈 second-ranked opportunity
-   🥉 third-ranked opportunity
-   AI score comparison
-   expected 20D return comparison
-   signal distribution
-   complete 20-stock ranking
-   downloadable CSV

This page is designed around the question:

> **Which stocks does the model currently rank highest?**

------------------------------------------------------------------------

# 📈 Backtest

The dashboard presents the final strategy research results.

The selected V34 architecture was:

> **Pure 20D Hold**

Historical research results:

  Metric                    Result
  ------------------ -------------
  Portfolio Return     **+15.11%**
  Maximum Drawdown     **-14.56%**
  Sharpe Ratio            **0.49**
  Profit Factor           **1.38**
  Average Trade         **+1.09%**

These numbers are historical research results only.

They should not be interpreted as expected future performance.

------------------------------------------------------------------------

# ⚙️ System

The System page exposes the production architecture.

It shows:

-   AI architecture
-   model configuration
-   production feature set
-   pipeline health
-   generated file status
-   current dataset counts
-   stock universe size

This makes the project easier to inspect during a technical interview.

------------------------------------------------------------------------

# 🧪 Validation Philosophy

A major goal of the project was to avoid misleading backtests.

The research therefore progressed toward:

### Chronological validation

Training data comes before testing data.

### Walk-forward testing

The model is repeatedly trained using past information and evaluated on
later periods.

### Portfolio-level evaluation

The strategy is evaluated as a portfolio rather than only through model
accuracy.

### Realistic execution

The research investigated entry timing, exits, stop-losses, targets, and
daily mark-to-market behavior.

### Feature ablation

Feature groups were compared instead of assuming every feature adds
value.

------------------------------------------------------------------------

# 📊 Excess-Return Research

The best excess-return research model produced approximately:

  Metric                           Result
  ----------------------------- ---------
  RMSE                            \~6.88%
  Relative Direction Accuracy     \~50.3%
  Mean Rank IC                    \~0.081
  Top-5 Excess Return              +0.77%
  Bottom-5 Excess Return           -0.65%
  Top/Bottom Spread                +1.42%

The main takeaway was:

> **The model showed more evidence of usefulness for ranking stocks than
> for predicting exact future returns.**

That finding directly influenced the production architecture.

------------------------------------------------------------------------

# 📉 Strategy Research

The project also tested:

-   static ranking
-   universal models
-   stock-specific models
-   different portfolio sizes
-   different feature sets
-   different stop-loss levels
-   take-profit architectures
-   pure holding periods
-   realistic execution
-   daily mark-to-market performance
-   different stock universes

This research showed that some attractive-looking results became weaker
after stricter validation.

That is an important result in itself.

------------------------------------------------------------------------

# 💡 Key Research Lessons

### 1. More complex does not mean more predictive

Adding more indicators did not automatically improve performance.

### 2. Short-term direction is extremely noisy

The initial classification models did not outperform simple baselines
convincingly.

### 3. Ranking can be more useful than exact prediction

The model's stronger role was identifying relative differences between
stocks.

### 4. Market context matters

NIFTY and BANK NIFTY momentum became important components of the final
feature set.

### 5. Backtesting must be conservative

A strategy that looks good under simplified assumptions may deteriorate
after realistic validation.

### 6. Model confidence should not be confused with probability

The dashboard explicitly treats conviction as signal strength.

------------------------------------------------------------------------

# 🛠️ Technology Stack

## Python

Core programming language.

## Pandas

Data cleaning, transformation and feature processing.

## NumPy

Numerical operations.

## yfinance

Market-data retrieval.

## Requests + BeautifulSoup

News-data collection and parsing.

## Scikit-learn

Machine-learning models and evaluation.

## Joblib

Model serialization.

## Plotly

Interactive financial charts and visualizations.

## Streamlit

Interactive web dashboard.

## Git + GitHub

Version control and portfolio deployment.

------------------------------------------------------------------------

# ▶️ Installation

## 1. Clone the repository

``` bash
git clone https://github.com/KaranKapadia03/INDIAN-STOCK-TRADING-AI.git
```

``` bash
cd INDIAN-STOCK-TRADING-AI
```

------------------------------------------------------------------------

## 2. Create a virtual environment

Windows:

``` powershell
python -m venv .venv
```

Activate it:

``` powershell
.venv\Scripts\activate
```

------------------------------------------------------------------------

## 3. Install dependencies

``` powershell
pip install -r requirements.txt
```

------------------------------------------------------------------------

# ▶️ Run the AI Pipeline

From the project root:

``` powershell
python src\run_ai.py
```

This refreshes:

``` text
Market Data
Technical Features
Market Context
News
ML Predictions
Recommendations
```

------------------------------------------------------------------------

# 🖥️ Launch the Dashboard

Run:

``` powershell
streamlit run src\dashboard.py
```

Then open the local Streamlit address shown in the terminal.

------------------------------------------------------------------------

# 🔄 Typical User Workflow

``` text
Activate environment
        ↓
Run AI pipeline
        ↓
Refresh market data
        ↓
Build features
        ↓
Collect news
        ↓
Generate ML forecasts
        ↓
Generate BUY / HOLD / SELL
        ↓
Open dashboard
        ↓
Review rankings
        ↓
Inspect individual stocks
```

------------------------------------------------------------------------

# 📦 Generated Outputs

The pipeline generates files such as:

``` text
data/processed/
├── *_features.csv
├── live_ml/
│   ├── live_predictions.csv
│   └── live_predictions.json
└── production/
    ├── recommendations.csv
    └── recommendations.json
```

Production model artifacts:

``` text
data/models/production/
├── universal_rf.joblib
├── feature_list.joblib
└── model_metadata.json
```

These are generated runtime artifacts and are intentionally excluded
from the public repository.

------------------------------------------------------------------------

# 🔐 Security

The repository's `.gitignore` excludes:

``` text
.env
.venv/
data/raw/
data/processed/
data/models/
data/database/
```

This prevents secrets, local environments, generated datasets and model
artifacts from being accidentally committed.

------------------------------------------------------------------------

# ⚠️ Limitations

This project is a **research and educational decision-support system**.

It is not a guaranteed trading system.

### Market uncertainty

Financial markets are non-stationary and noisy. Historical relationships
can disappear.

### Data freshness

Yahoo Finance data availability and freshness can vary and may not
represent tick-by-tick exchange data.

### News limitation

The news system collects recent news and is not a historical news
database.

### Prediction uncertainty

A forecast such as:

``` text
+5%
```

does not mean the stock will actually return 5%.

### Conviction limitation

A conviction score is not a probability of profit.

### Backtest limitation

Historical performance does not guarantee future performance.

### Execution limitation

The current system is a research/decision-support platform and does not
automatically place live broker orders.

------------------------------------------------------------------------

# 🧭 Future Development

The architecture can be extended with:

-   larger Indian stock universe
-   sector classification
-   fundamental financial data
-   earnings and corporate-event features
-   historical news database
-   improved probability calibration
-   transaction-cost modelling
-   slippage modelling
-   portfolio position sizing
-   paper trading
-   broker API integration
-   real-time streaming data
-   FastAPI backend
-   cloud deployment
-   model monitoring
-   drift detection
-   automated retraining
-   portfolio-level risk management

These are future engineering opportunities rather than capabilities
claimed by the current version.

------------------------------------------------------------------------

# 🏆 What Makes This Project Different

This project is not simply:

``` text
Download stock data
        ↓
Train Random Forest
        ↓
Print prediction
```

It is an end-to-end research system:

``` text
             DATA
               ↓
      FEATURE ENGINEERING
               ↓
       MARKET CONTEXT
               ↓
      NEWS INTELLIGENCE
               ↓
        ML RESEARCH
               ↓
      MODEL SELECTION
               ↓
    WALK-FORWARD VALIDATION
               ↓
       STRATEGY RESEARCH
               ↓
      PRODUCTION MODEL
               ↓
       LIVE PREDICTIONS
               ↓
    RECOMMENDATION ENGINE
               ↓
       STREAMLIT APP
```

The project also documents cases where models **did not work well**,
rather than presenting only the most attractive result.

That is a core part of the research methodology.

------------------------------------------------------------------------

# 📌 Final Takeaway

The objective of Indian Stock Trading AI is not to claim:

> **"AI can predict the stock market."**

The objective is to investigate a more useful question:

> **"Can a systematic machine-learning pipeline extract enough
> information from Indian market data to produce a useful and testable
> stock-ranking signal?"**

The project explores that question from raw data collection all the way
to a production-style dashboard.

The final system prioritizes:

``` text
Robustness
    >
Complexity
```

and:

``` text
Relative Stock Selection
    >
Exact Return Prediction
```

------------------------------------------------------------------------

# 👨‍💻 Author

**Karan Kapadia**

GitHub:\
https://github.com/KaranKapadia03

Project Repository:\
https://github.com/KaranKapadia03/INDIAN-STOCK-TRADING-AI

------------------------------------------------------------------------

# ⚖️ Disclaimer

This project is intended for **educational, research and portfolio
purposes only**.

Nothing in this repository should be interpreted as financial advice, a
recommendation to buy or sell securities, or a guarantee of future
returns.

Model predictions can be wrong. Historical backtests can differ
materially from future results. Users should independently evaluate
investment decisions and risk.

------------------------------------------------------------------------

## ⭐ Project Status

**Production-style research prototype --- dashboard and core strategy
architecture frozen for portfolio release.**

Future improvements can be developed as new versions rather than
changing the current research conclusions.
