# 🇮🇳 Indian Stock Trading AI

> **An end-to-end machine-learning research system for Indian equities
> that combines market data, technical analysis, market context,
> financial news and a production ML model to generate BUY / HOLD / SELL
> signals.**

```{=html}
<p align="center">
```
**📊 Market Data** → **📐 Features** → **🧠 ML Model** → **📰 News** →
**🎯 Signal Engine** → **📈 Dashboard**

```{=html}
</p>
```

------------------------------------------------------------------------

## 🚀 What is this project?

**Indian Stock Trading AI** is a complete quantitative research and
decision-support platform built around a 20-stock Indian equity
universe.

The system takes raw market information and turns it into an
interpretable stock-ranking workflow:

1.  Downloads historical/latest Indian stock data.
2.  Builds technical indicators.
3.  Adds NIFTY 50 and BANK NIFTY market context.
4.  Collects recent financial news.
5.  Generates a news-sentiment signal.
6.  Uses a production Random Forest model to estimate **20-trading-day
    excess return**.
7.  Ranks stocks cross-sectionally.
8.  Combines ML, technical and news signals.
9.  Produces **BUY / HOLD / SELL** recommendations.
10. Presents everything through an interactive Streamlit dashboard.

The goal is not to claim that an algorithm can predict the market
perfectly.

The goal is to build a **realistic, reproducible end-to-end ML trading
research system** and test where machine learning actually adds value.

------------------------------------------------------------------------

# 🎯 Business Problem

An investor looking at 20 large Indian companies has to process multiple
information sources:

-   Price movement
-   Volatility
-   Trading volume
-   Technical indicators
-   NIFTY market momentum
-   BANK NIFTY momentum
-   Company-specific news
-   Relative stock strength
-   Model forecasts

Doing this manually is slow and inconsistent.

This project asks:

> **Can market data, technical features, market context and recent news
> be combined into a systematic AI-driven stock-ranking and
> recommendation system?**

Instead of asking the model:

> "Will RELIANCE go up tomorrow?"

the production system focuses on a more realistic research question:

> **"Which stocks currently have the strongest expected performance
> relative to the NIFTY over the next 20 trading days?"**

That distinction is important because short-term price direction is
extremely noisy.

------------------------------------------------------------------------

# 🧠 System Architecture

``` text
                         ┌─────────────────────┐
                         │   Indian Equities   │
                         │     20 Stocks       │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │    Market Data      │
                         │       yfinance      │
                         └──────────┬──────────┘
                                    │
                                    ▼
                    ┌─────────────────────────────┐
                    │      Feature Engineering    │
                    │                             │
                    │ RSI • MACD • SMA • EMA      │
                    │ Volatility • Volume         │
                    │ Returns • ATR • Bollinger   │
                    └──────────────┬──────────────┘
                                   │
                  ┌────────────────┼────────────────┐
                  ▼                ▼                ▼
          ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
          │ NIFTY 50     │ │ BANK NIFTY   │ │ Financial    │
          │ Context      │ │ Context      │ │ News         │
          └──────┬───────┘ └──────┬───────┘ └──────┬───────┘
                 │                │                │
                 └────────────────┼────────────────┘
                                  ▼
                         ┌─────────────────────┐
                         │ Production ML Model │
                         │ Random Forest       │
                         │ Regressor           │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │ 20D Excess Return   │
                         │ Forecast            │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │ Cross-Sectional     │
                         │ Ranking             │
                         └──────────┬──────────┘
                                    │
                                    ▼
                    ┌─────────────────────────────┐
                    │ Recommendation Engine       │
                    │                             │
                    │ ML          50%             │
                    │ Technical   25%             │
                    │ News        25%             │
                    └──────────────┬──────────────┘
                                   │
                                   ▼
                       ┌─────────────────────────┐
                       │ BUY / HOLD / SELL       │
                       └────────────┬────────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │ Streamlit Dashboard │
                         └─────────────────────┘
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
│   │
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
│   │   ├── live_predictions.py
│   │   ├── train_production_model.py
│   │   └── ...research scripts
│   │
│   ├── strategy/
│   │   ├── technical_signal.py
│   │   ├── combined_signal.py
│   │   ├── recommendation_engine.py
│   │   ├── final_engine.py
│   │   ├── risk_engine.py
│   │   ├── portfolio_backtest.py
│   │   └── ...research/backtest scripts
│   │
│   └── dashboard.py
│
├── tests/
│
├── .env
├── .gitignore
├── requirements.txt
└── README.md
```

> Generated datasets, model artifacts, `.env`, virtual environments and
> other runtime files are intentionally excluded from Git.

------------------------------------------------------------------------

# 📊 Stock Universe

The current production universe contains 20 Indian equities:

  Symbol          Company
  --------------- ---------------------------
  RELIANCE.NS     Reliance Industries
  TCS.NS          Tata Consultancy Services
  HDFCBANK.NS     HDFC Bank
  ICICIBANK.NS    ICICI Bank
  INFY.NS         Infosys
  HINDUNILVR.NS   Hindustan Unilever
  ITC.NS          ITC
  SBIN.NS         State Bank of India
  BHARTIARTL.NS   Bharti Airtel
  KOTAKBANK.NS    Kotak Mahindra Bank
  LT.NS           Larsen & Toubro
  AXISBANK.NS     Axis Bank
  MARUTI.NS       Maruti Suzuki
  SUNPHARMA.NS    Sun Pharmaceutical
  TITAN.NS        Titan Company
  ADANIENT.NS     Adani Enterprises
  ADANIPORTS.NS   Adani Ports
  BAJFINANCE.NS   Bajaj Finance
  ASIANPAINT.NS   Asian Paints
  ULTRACEMCO.NS   UltraTech Cement

The universe can be expanded later.

------------------------------------------------------------------------

# 1️⃣ Market Data Layer

### `src/data/stock_universe.py`

Defines the stock universe used throughout the system.

This creates a single source of truth for symbols and company names.

------------------------------------------------------------------------

### `src/data/market_data.py`

Responsible for downloading historical OHLCV data.

The pipeline retrieves:

-   Open
-   High
-   Low
-   Close
-   Volume

The project currently uses approximately five years of daily historical
data.

Market data is stored locally so downstream feature engineering and
research can work from reproducible files.

------------------------------------------------------------------------

### `src/data/market_context.py`

Provides broader market context through:

-   NIFTY 50
-   BANK NIFTY

The system calculates market-return features that help the model
understand whether an individual stock is moving with or against the
broader Indian market.

------------------------------------------------------------------------

# 2️⃣ Technical Feature Engineering

### `src/features/technical_indicators.py`

Transforms raw OHLCV data into technical features.

The feature engine includes:

### Trend

-   SMA 20
-   SMA 50
-   EMA 20
-   Price vs SMA20
-   Price vs SMA50
-   SMA20 vs SMA50

### Momentum

-   1-day return
-   5-day return
-   10-day return
-   20-day return
-   RSI 14
-   MACD
-   MACD Signal
-   MACD Histogram

### Volatility

-   20-day volatility
-   ATR 14
-   ATR percentage

### Bollinger Bands

-   Middle band
-   Upper band
-   Lower band
-   Bollinger position

### Volume

-   20-day volume average
-   Volume ratio

The output is saved as:

``` text
data/processed/<SYMBOL>_features.csv
```

------------------------------------------------------------------------

# 3️⃣ Market Context Features

### `src/features/add_market_features.py`

Adds broader-market information to each stock's feature dataset.

The production model ultimately uses:

``` text
Volatility_20D
Volume_Ratio
NIFTY_Return_5D
NIFTY_Return_20D
BANKNIFTY_Return_5D
BANKNIFTY_Return_20D
```

These features deliberately keep the production model relatively small.

The research process showed that a smaller, more stable feature set
performed better than throwing every available technical feature into
the model.

------------------------------------------------------------------------

# 4️⃣ News Intelligence

### `src/data/news_data.py`

Collects recent financial news using Google News RSS.

The news engine searches around each company and Indian financial-market
context.

It extracts information such as:

-   headline
-   publication time
-   source
-   article URL

Important:

> This is **recent/live news**, not historical news.

Therefore, live news is not used as a historical feature in the
backtests.

That avoids introducing a major look-ahead problem.

------------------------------------------------------------------------

### `src/features/news_sentiment.py`

Processes news headlines and creates sentiment information.

The system turns headline-level information into an aggregated
stock-level sentiment signal.

------------------------------------------------------------------------

### `src/features/live_news_signal.py`

Creates current news signals for all 20 stocks.

The output contains fields such as:

``` text
symbol
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

The recommendation engine uses this as one component of the final
signal.

------------------------------------------------------------------------

# 5️⃣ Target Engineering

One of the most important decisions in the project was changing the
prediction problem.

Instead of simply predicting:

``` text
Will the stock go UP or DOWN?
```

the production model predicts:

``` text
20D Stock Return - 20D NIFTY Return
```

This creates an **excess-return** target.

The idea is to ask:

> Is this stock expected to outperform the broader Indian market?

rather than merely:

> Is the market going up?

This makes the model more useful for cross-sectional stock selection.

------------------------------------------------------------------------

# 6️⃣ Machine Learning Research

The project did not jump directly to the final model.

Multiple model architectures were tested.

The research included:

-   binary classification
-   regression
-   trade-quality classification
-   calibrated models
-   ensembles
-   multiple forecast horizons
-   stock-specific models
-   cross-sectional ranking
-   walk-forward validation
-   feature ablation
-   stock-universe testing
-   exit architecture testing
-   realistic execution assumptions
-   daily mark-to-market portfolio testing

This research process is important because many apparently good trading
models disappear when tested correctly.

------------------------------------------------------------------------

# 7️⃣ Production ML Model

### `src/models/train_production_model.py`

The final production model is a:

> **Random Forest Regressor**

Configuration:

``` text
Trees:             400
Maximum depth:     8
Minimum split:     15
Minimum leaf:      5
Max features:      sqrt
Random state:      42
```

### Target

``` text
Target_Excess_Return_20D
```

### Production feature set

``` text
Volatility_20D
Volume_Ratio
NIFTY_Return_5D
NIFTY_Return_20D
BANKNIFTY_Return_5D
BANKNIFTY_Return_20D
```

The model is trained using data available before the latest prediction
period.

Generated artifacts:

``` text
data/models/production/
├── universal_rf.joblib
├── feature_list.joblib
└── model_metadata.json
```

------------------------------------------------------------------------

# 8️⃣ Live ML Predictions

### `src/models/live_predictions.py`

Loads the production Random Forest and the latest stock features.

For every stock it:

1.  Loads the latest feature row.
2.  Applies the exact production feature list.
3.  Generates a 20-day excess-return forecast.
4.  Ranks the stocks cross-sectionally.
5.  Saves the predictions.

Output:

``` text
data/processed/live_ml/live_predictions.csv
data/processed/live_ml/live_predictions.json
```

The output provides a current model ranking rather than pretending the
predicted return is a guaranteed outcome.

------------------------------------------------------------------------

# 9️⃣ Recommendation Engine

### `src/strategy/production_engine.py`

The recommendation engine combines three information sources:

``` text
Machine Learning       50%
Technical Analysis     25%
News Intelligence      25%
```

The final signal is generated using the configured thresholds.

Conceptually:

``` text
                ML
                │
                ▼
          ┌───────────┐
          │           │
Technical ─►  Signal  ◄─ News
          │  Engine   │
          └─────┬─────┘
                │
        ┌───────┼────────┐
        ▼       ▼        ▼
       BUY    HOLD      SELL
```

The system also reports:

-   expected 20D return
-   AI score
-   confidence / conviction
-   ML contribution
-   technical contribution
-   news contribution

### Important distinction

**Confidence is signal strength, not probability of profit.**

A `70%` conviction score does **not** mean there is a 70% chance the
trade will make money.

------------------------------------------------------------------------

# 🔟 Automated AI Pipeline

### `src/run_ai.py`

This is the main production workflow.

Running:

``` bash
python src/run_ai.py
```

executes:

``` text
1. LIVE MARKET DATA
        ↓
2. BUILD TECHNICAL FEATURES
        ↓
3. ADD MARKET FEATURES
        ↓
4. COLLECT LIVE NEWS
        ↓
5. GENERATE LIVE ML PREDICTIONS
        ↓
6. GENERATE AI RECOMMENDATIONS
```

This means the entire system can be refreshed with one command.

------------------------------------------------------------------------

# 🖥️ Streamlit Dashboard

### `src/dashboard.py`

The dashboard is the user-facing layer of the project.

It provides five main sections.

------------------------------------------------------------------------

## 🏠 AI Terminal

The main trading-intelligence screen shows:

-   stocks analyzed
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
-   AI ranking chart
-   expected-return chart

The purpose is to answer the question:

> **What does the model currently see?**

------------------------------------------------------------------------

# 📊 Stock Analysis

Users can select any stock in the universe.

The page provides:

### Price

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

### Technicals

-   RSI
-   MACD
-   volatility
-   volume ratio
-   price vs moving averages
-   ATR percentage
-   RSI visualization
-   volume visualization

### News

Recent news and aggregated sentiment information.

------------------------------------------------------------------------

# 🧠 AI Rankings

The ranking page answers:

> **Which stocks does the model currently rank highest?**

It provides:

-   Top 3 AI opportunities
-   expected 20D return
-   AI score
-   conviction
-   complete 20-stock ranking
-   AI score visualization
-   expected-return visualization
-   signal distribution
-   CSV export

This is the main cross-sectional stock-selection screen.

------------------------------------------------------------------------

# 📈 Backtest

The dashboard also communicates the historical research results.

The selected final research architecture is:

> **Pure 20D Hold**

Key V34 research results:

  Metric                    Result
  ------------------ -------------
  Portfolio Return     **+15.11%**
  Maximum Drawdown     **-14.56%**
  Sharpe Ratio            **0.49**
  Profit Factor           **1.38**
  Average Trade         **+1.09%**

These figures are historical research results, not live performance.

------------------------------------------------------------------------

# ⚙️ System

The System page exposes the production architecture and health of the
pipeline.

It shows:

-   model architecture
-   feature set
-   production model configuration
-   pipeline status
-   generated files
-   dataset counts
-   stock-universe size

This makes the application easier to audit and demonstrate in a
portfolio interview.

------------------------------------------------------------------------

# 🔬 Research Journey

The project deliberately went through multiple iterations.

## Early findings

Short-horizon direction prediction was weak.

The 5-day classifier produced approximately:

``` text
Accuracy:       49.01%
Baseline:       50.69%
ROC-AUC:         0.534
```

This showed that predicting simple up/down movement was not strong
enough.

------------------------------------------------------------------------

## Regression findings

The initial 56-feature regression model also struggled.

The model produced:

``` text
MAE:       ~3.17%
RMSE:      ~3.99%
R²:       -0.51
Direction: ~49.8%
```

A negative R² indicated that the model did not beat a simple baseline on
that task.

This was an important research result.

------------------------------------------------------------------------

# ⏱️ Forecast Horizon Research

Longer horizons showed a gradual improvement in average historical
returns.

Approximate average forward returns:

  Horizon     Mean Return
  --------- -------------
  1D              +0.041%
  3D              +0.126%
  5D              +0.216%
  10D             +0.440%
  20D             +0.896%

This supported testing a longer 20-day horizon.

------------------------------------------------------------------------

# 🎯 Excess-Return Research

The project then shifted from absolute returns to:

``` text
Stock Return - NIFTY Return
```

The best research model produced:

``` text
RMSE:             ~6.88%
Relative Direction: ~50.3%
Mean Rank IC:       ~0.081
Top-5 Excess:       +0.77%
Bottom-5 Excess:    -0.65%
Top/Bottom Spread:  +1.42%
```

The important finding was:

> The model was more useful for **ranking stocks** than for predicting
> exact future returns.

That insight shaped the production architecture.

------------------------------------------------------------------------

# 🚶 Walk-Forward Validation

The project used walk-forward testing to reduce look-ahead bias.

Instead of training on the entire historical dataset and then testing on
the same period, the process respects chronological order:

``` text
TRAIN
──────────────────► TEST
       TRAIN
       ──────────────────► TEST
              TRAIN
              ──────────────────► TEST
```

This is much closer to how a real trading model would operate.

------------------------------------------------------------------------

# 🧪 Feature Selection

A large feature set was not automatically better.

Research found that a smaller stable feature group performed better.

The production feature set therefore focuses on:

``` text
Stock Volatility
Stock Volume
NIFTY Momentum
BANK NIFTY Momentum
```

This is a deliberate design choice:

> **Prefer a smaller feature set with evidence of stability over a
> larger feature set that looks impressive but does not generalize.**

------------------------------------------------------------------------

# 📉 Strategy Validation

The research tested:

-   different portfolio sizes
-   ranking strategies
-   stock-specific models
-   universal models
-   stop-loss architectures
-   take-profit architectures
-   pure holding periods
-   realistic execution
-   daily mark-to-market performance
-   different stock universes

One of the important conclusions was that apparently strong results
could weaken substantially under more realistic validation.

That is why the project does **not** present the backtest as proof of
future profitability.

------------------------------------------------------------------------

# 🧠 What the Model Learned

The final production model's feature importance showed the strongest
contributions from:

``` text
BANKNIFTY_Return_20D
Volatility_20D
NIFTY_Return_20D
BANKNIFTY_Return_5D
NIFTY_Return_5D
Volume_Ratio
```

This suggests that broader market regime/context plays an important role
in the model's stock-ranking decisions.

------------------------------------------------------------------------

# 🛠️ Tech Stack

### Programming

-   Python

### Data

-   Pandas
-   NumPy
-   yfinance
-   Requests
-   BeautifulSoup

### Machine Learning

-   Scikit-learn
-   Random Forest
-   Joblib

### Visualization

-   Plotly
-   Streamlit

### Engineering

-   Git
-   GitHub
-   Virtual environments
-   Modular Python architecture

------------------------------------------------------------------------

# ▶️ How to Run

## 1. Clone the repository

``` bash
git clone https://github.com/KaranKapadia03/INDIAN-STOCK-TRADING-AI.git
cd INDIAN-STOCK-TRADING-AI
```

## 2. Create a virtual environment

Windows:

``` powershell
python -m venv .venv
```

Activate:

``` powershell
.venv\Scripts\activate
```

## 3. Install dependencies

``` powershell
pip install -r requirements.txt
```

## 4. Run the AI pipeline

``` powershell
python src\run_ai.py
```

## 5. Launch the dashboard

``` powershell
streamlit run src\dashboard.py
```

The dashboard will open at the local Streamlit address shown in the
terminal.

------------------------------------------------------------------------

# 🔄 Typical Workflow

For a fresh analysis:

``` text
1. Start environment
       ↓
2. python src/run_ai.py
       ↓
3. Market data refresh
       ↓
4. Feature generation
       ↓
5. News collection
       ↓
6. ML prediction
       ↓
7. BUY/HOLD/SELL generation
       ↓
8. streamlit run src/dashboard.py
       ↓
9. Explore rankings and stock analysis
```

------------------------------------------------------------------------

# 📦 Generated Files

The production pipeline creates files such as:

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

The trained model creates:

``` text
data/models/production/
├── universal_rf.joblib
├── feature_list.joblib
└── model_metadata.json
```

These generated files are intentionally excluded from the public Git
repository.

------------------------------------------------------------------------

# 🔐 Data & Security

The repository excludes:

``` text
.env
.venv/
data/raw/
data/processed/
data/models/
data/database/
```

This prevents:

-   API secrets
-   local environments
-   generated datasets
-   trained model artifacts
-   runtime databases

from being accidentally committed.

------------------------------------------------------------------------

# ⚠️ Limitations

This project is a **research and educational system**, not a guaranteed
trading system.

### 1. Market prediction is inherently uncertain

Financial markets are noisy and non-stationary.

Historical relationships can disappear.

### 2. News data is recent

The live news engine is not a historical news database.

Therefore, live news should not be treated as a historical backtest
feature.

### 3. Yahoo Finance data freshness can vary

The latest available Yahoo Finance observation may not represent a
tick-by-tick exchange feed.

### 4. Expected return is not guaranteed return

A model prediction such as:

``` text
+5%
```

does not mean the stock will actually return 5%.

### 5. AI conviction is not probability

A conviction score represents signal strength.

It is not a calibrated probability of profit.

### 6. Backtests are not future performance

The historical results shown in the dashboard do not guarantee future
results.

### 7. The system does not execute real trades

The current project is a decision-support/research platform.

It does not automatically place live orders.

------------------------------------------------------------------------

# 🧭 Future Improvements

Possible future engineering directions include:

-   Expand the stock universe
-   Historical news dataset
-   Fundamental financial statements
-   Earnings-event features
-   Sector-relative features
-   Better probability calibration
-   More advanced ensemble models
-   Transaction-cost modelling
-   Slippage modelling
-   Paper-trading engine
-   Broker API integration
-   Portfolio-level risk allocation
-   Position sizing
-   Stop-loss optimization
-   Real-time data infrastructure
-   FastAPI backend
-   Cloud deployment
-   Authentication
-   Automated model retraining
-   Model monitoring
-   Drift detection

These are deliberately future improvements rather than claims about the
current system.

------------------------------------------------------------------------

# 💡 Key Lessons From the Project

The biggest lesson was not:

> "Machine learning predicts stocks."

It was:

> **Good trading ML requires careful problem formulation, leakage
> control, walk-forward validation, realistic execution assumptions and
> honest interpretation of weak predictive signals.**

Several models looked promising initially but deteriorated when tested
more rigorously.

The final architecture therefore prioritizes:

``` text
Robustness
    >
Complexity
```

and:

``` text
Stock Ranking
    >
Exact Return Prediction
```

------------------------------------------------------------------------

# 🏆 Project Highlights

### End-to-end

The project covers the complete path:

``` text
Raw Data
   ↓
Feature Engineering
   ↓
Machine Learning
   ↓
Prediction
   ↓
Strategy
   ↓
Backtesting
   ↓
Recommendation
   ↓
Dashboard
```

### Production-oriented

The system has:

-   modular source files
-   reusable pipeline
-   saved model artifacts
-   generated outputs
-   automated refresh
-   interactive dashboard
-   Git version control
-   security-aware `.gitignore`

### Research-oriented

The project did not stop at the first model.

It tested:

-   classification
-   regression
-   ensembles
-   multiple horizons
-   ranking
-   walk-forward validation
-   feature ablation
-   realistic execution
-   portfolio construction
-   exit architectures

------------------------------------------------------------------------

# 👨‍💻 Author

**Karan Kapadia**

GitHub:

https://github.com/KaranKapadia03

Project:

https://github.com/KaranKapadia03/INDIAN-STOCK-TRADING-AI

------------------------------------------------------------------------

# 📌 Disclaimer

This project is intended for **educational, research and portfolio
purposes**.

It is not financial advice.

The model outputs are statistical estimates and can be wrong. Historical
backtest performance does not guarantee future performance. Users should
independently evaluate risk before making any investment decision.

------------------------------------------------------------------------

## ⭐ If you find this project interesting

Consider starring the repository and exploring the research pipeline.

**Built to explore one question:**

> ### Can machine learning turn noisy Indian market data into a systematic, testable stock-selection process?
