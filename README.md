# Quant Research: Machine Learning for Financial Time Series

End-to-end research project that tests whether machine learning and deep learning models can produce a tradable signal from daily market data, and evaluates them the way a quant desk would: with a strict time-based split, transaction costs and risk-adjusted metrics, not just prediction error.

> Status: work in progress. The results section is filled in as each phase is completed.

## Research question

Can models of increasing complexity (linear baselines, tree ensembles, LSTM, Transformer) predict next-day returns well enough to beat a buy-and-hold benchmark **after transaction costs**, on data the models have never seen?

The project is designed so that a negative answer is a valid, well-documented result. Financial markets are noisy and close to efficient, so the goal is a rigorous comparison, not a spectacular backtest.

## Design principles

- **No look-ahead bias.** Features at day `t` only use information available up to day `t`. The target is the return of day `t+1`. A signal generated at the close of day `t` is only traded on day `t+1`.
- **Time-based splits only.** No random shuffling. Train, validation and test sets follow chronological order.
- **Baselines first.** Every complex model is compared against simple ones, otherwise a neural network is not evidence of anything.
- **Decisions are judged by the backtest.** A lower RMSE does not matter if the strategy loses money after costs.
- **Reproducible.** Fixed seeds, pinned dependencies, tracked experiments and a Docker image.

## Pipeline

```text
Raw data (yfinance)
        |
        v
Parquet dataset (data/raw)
        |
        v
Feature engineering (returns, momentum, volatility, volume)
        |
        v
Models: Linear / Logistic -> Random Forest / XGBoost -> LSTM / Transformer
        |
        v
Trading signal (prediction -> position)
        |
        v
Backtest with transaction costs
        |
        v
Risk / return metrics -> MLflow -> Report
```

## Data

| Item | Value |
|---|---|
| Source | Yahoo Finance via `yfinance` |
| First asset | SPY (S&P 500 ETF) |
| Frequency | Daily OHLCV, adjusted prices |
| Period | 2010-01-01 to 2025-12-31 |
| Storage | Parquet files in `data/` (not tracked by git) |

The pipeline starts with a single asset so the whole process can be validated end to end. Extending it to a basket of ETFs is planned once the single-asset version is solid.

### Chronological split

| Set | Period | Purpose |
|---|---|---|
| Train | 2010 - 2021 | Fit models |
| Validation | 2022 - 2023 | Tune hyperparameters, choose thresholds |
| Test | 2024 - 2025 | Final, untouched evaluation |

## Methodology

### Features

Computed only from past data:

- Log returns over 1 and 5 days
- Momentum over 20 days
- Rolling volatility over 20 days
- Relative volume against its 20-day average
- Additional technical and statistical factors added during the project

### Target

- Regression: next-day log return
- Classification: sign of the next-day return

### Models

| Family | Models | Role |
|---|---|---|
| Baselines | Linear Regression, Logistic Regression | Reference point |
| Machine Learning | Random Forest, XGBoost | Non-linear tabular models |
| Deep Learning | LSTM, Transformer (PyTorch) | Sequence models |

### From prediction to strategy

1. The model outputs a predicted return or probability of going up.
2. A threshold, chosen on the validation set, converts it into a position (long or flat, and later long or short).
3. The position is lagged one day to avoid look-ahead.
4. Returns are computed net of transaction costs applied to every change of position. The cost per trade is a configurable assumption.

### Evaluation

Prediction metrics: RMSE, directional accuracy.

Strategy metrics, always compared against buy-and-hold:

| Metric | What it tells you |
|---|---|
| CAGR | Annualised growth |
| Volatility | Annualised risk |
| Sharpe Ratio | Return per unit of total risk |
| Sortino Ratio | Return per unit of downside risk |
| Maximum Drawdown | Worst peak-to-trough loss |
| Win Rate | Share of profitable trades |
| Turnover | How much the strategy trades |

## Results

To be completed after phase 7. Every number in this table will be produced by the code in this repository.

| Model | RMSE | Directional accuracy | CAGR | Sharpe | Max Drawdown | Turnover |
|---|---|---|---|---|---|---|
| Buy and hold | - | - | TBD | TBD | TBD | - |
| Linear / Logistic | TBD | TBD | TBD | TBD | TBD | TBD |
| Random Forest | TBD | TBD | TBD | TBD | TBD | TBD |
| XGBoost | TBD | TBD | TBD | TBD | TBD | TBD |
| LSTM | TBD | TBD | TBD | TBD | TBD | TBD |
| Transformer | TBD | TBD | TBD | TBD | TBD | TBD |

Equity curves, drawdown plots and the discussion of results will be added under `reports/figures/`.

## Project structure

```text
quant-research/
├── README.md
├── pyproject.toml
├── requirements.txt
├── .gitignore
├── Dockerfile
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
│   ├── 01_datos_y_eda.ipynb
│   ├── 02_features.ipynb
│   ├── 03_baselines.ipynb
│   ├── 04_xgboost.ipynb
│   ├── 05_deep_learning.ipynb
│   ├── 06_backtest.ipynb
│   └── 07_resultados.ipynb
├── src/
│   └── quantlab/
│       ├── __init__.py
│       ├── data.py
│       ├── features.py
│       ├── models.py
│       ├── backtest.py
│       └── metrics.py
├── reports/
│   └── figures/
├── tests/
└── mlruns/
```

| Folder | Content |
|---|---|
| `notebooks/` | Exploration and results, one notebook per phase. They import their logic from `src/`. |
| `src/quantlab/` | Reusable code: data download, features, models, backtest and metrics. |
| `data/` | Downloaded and processed datasets. Regenerated by the code, not versioned. |
| `reports/` | Figures and final report. |
| `tests/` | Tests, mainly for the leakage-sensitive parts (feature timing, position lag). |
| `mlruns/` | MLflow experiment tracking. Not versioned. |

## Tech stack

| Area | Tools |
|---|---|
| Language | Python |
| Data | yfinance, Polars, Pandas, Parquet |
| Numerical and statistics | NumPy, SciPy, statsmodels |
| Machine learning | scikit-learn, XGBoost |
| Deep learning | PyTorch |
| Backtesting | vectorbt, cross-checked against a custom pandas implementation |
| Visualisation | Matplotlib, Plotly |
| Experiment tracking | MLflow |
| Reproducibility | Docker, Git |

## How to reproduce

```bash
git clone https://github.com/<your-username>/quant-research.git
cd quant-research
python -m venv .venv
source .venv/bin/activate
pip install -e .
pip install -r requirements.txt
jupyter lab
```

Run the notebooks in order, from `01` to `07`.

## Roadmap

- [ ] Phase 0: repository, environment and folder structure
- [ ] Phase 1: data download, storage in Parquet and exploratory analysis
- [ ] Phase 2: feature engineering and target definition
- [ ] Phase 3: linear and logistic baselines
- [ ] Phase 4: Random Forest and XGBoost
- [ ] Phase 5: LSTM and Transformer in PyTorch
- [ ] Phase 6: signal generation and backtest with costs
- [ ] Phase 7: results, MLflow tracking, Docker and final report

### Possible extensions

- Basket of several ETFs and cross-sectional signals
- Walk-forward validation with periodic retraining
- Reimplementation of the feature and backtest bottleneck in C++, with a Python vs C++ benchmark
- Version 2 of the pipeline on PySpark and Databricks for larger datasets

## Limitations

- Daily data on a single liquid asset: results do not generalise automatically to other assets or frequencies.
- Transaction costs are a simplified assumption and do not model slippage or market impact.
- Past performance in a backtest does not guarantee future results.

## Disclaimer

This project is for educational and research purposes only. It is not financial advice.

## Author

Carlos, Data Science and Artificial Intelligence student at ETSI Sistemas Informáticos, Universidad Politécnica de Madrid.

LinkedIn: add your profile link here
