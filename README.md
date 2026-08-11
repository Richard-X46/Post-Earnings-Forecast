# Post-Earnings Forecast

Predict post-earnings-announcement drift (PEAD) for S&P 500 stocks. The target is the highest intraday price over the nine trading days following each earnings event. Inputs span technical indicators, fundamentals, sector, and NLP sentiment from earnings-call transcripts (FinBERT) and news headlines. Models are trained on a 20-fold expanding-window walk-forward split across 2021 Q1 – 2025 Q4 over ~24,000 earnings events on ~500 S&P 500 constituents.

## Project Structure

```
.
├── pyproject.toml              # deps, requires-python>=3.12
├── uv.lock
├── src/
│   ├── ingestion/              # data fetching, backfill, S3 Delta Lake writes
│   │   ├── table_setup.py      # create Delta tables in S3
│   │   ├── backfill_earnings.py
│   │   ├── backfill_ohclc.py
│   │   ├── backfill_transcripts.py
│   │   ├── index_data.py       # VIX/SPX from yfinance
│   │   ├── upgrades_downgrades.py
│   │   ├── backup.py           # S3 → local Parquet backup + vacuum
│   │   └── archive/
│   ├── preprocessing/          # FinBERT sentiment, technical indicators
│   │   ├── technical_features.py
│   │   ├── bert_earnings_call.py
│   │   ├── bert_compaction_tx.py
│   │   ├── lm_sentiment_tx.py
│   │   ├── sentiment_stg.py
│   │   ├── join_validations.py
│   │   └── strat_table.py
│   ├── modeling/               # TabNet, DNN, XGBoost, RF ablation
│   │   ├── basepaper/
│   │   ├── tabnet/             # Modal GPU TabNet pipeline + trade strategy
│   │   ├── dnn/
│   │   ├── b1-5/                # ablation ladder (B1–B4) + tuned B5
│   │   ├── model_outputs/      # parquet + JSON results, pickled models
│   │   └── upload_model_data.py
│   └── notebooks/              # EDA + pipeline notebooks
├── docs/                       # planning docs, ablation notes
└── tabnet_trade_ledger.csv     # per-trade ledger from final strategy backtest
```

- **Data**: S3 Delta Lake (primary) + local Parquet backups under `src/ingestion/data/`. S&P 500 tickers, OHLCV, earnings calendars, transcripts, analyst upgrades/downgrades, VIX/SPX.
- **Notebooks**: EDA (`src/notebooks/EDA_final.ipynb`), model pipelines (`src/modeling/b1-5/*.ipynb`, `src/modeling/tabnet/*.ipynb`, `src/modeling/dnn/*.ipynb`).
- **Scripts**: ingestion (`src/ingestion/`), preprocessing (`src/preprocessing/`), modeling (`src/modeling/`).

## Dependencies / Libraries to Install

Python ≥ 3.12. Managed with [`uv`](https://docs.astral.sh/uv/).

| Library | Purpose |
|---|---|
| numpy, pandas, polars, pyarrow | Data wrangling |
| scikit-learn | Baseline models, metrics |
| xgboost | Gradient-boosted trees |
| tensorflow | DNN experiments |
| torch, pytorch-tabnet | TabNet |
| pysentiment2 | Loughlin-McDonald sentiment dictionary |
| transformers (via HF_TOKEN) | FinBERT embeddings |
| ta-lib | Technical indicators |
| yfinance, requests | Market data fetching |
| deltalake, duckdb, s3fs, boto3 | S3 + Delta Lake I/O |
| modal | GPU jobs (TabNet training) |
| optuna | Hyperparameter tuning |
| matplotlib, seaborn, plotly, altair | Visualisation |
| openpyxl | Excel I/O |

System dependency: `brew install ta-lib` (macOS).

## How to Run

```bash
uv sync                      # install dependencies
cp .env.example .env         # fill in AWS + API keys (see table below)
```

| Variable | Purpose |
|---|---|
| `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` | S3 + Delta Lake |
| `S3_ACCESS_KEY` / `S3_SECRET_KEY` | S3 (secondary pattern) |
| `AWS_SESSION_TOKEN` | S3 session token |
| `S3_BUCKET` | S3 bucket name |
| `AV_PREMIUM_KEY` | Alpha Vantage premium API key |
| `DUCKDB_KEY` | Encrypted Parquet key for DuckDB |
| `HF_TOKEN` | HuggingFace token (Modal GPU jobs) |

AWS region: `ca-central-1`.

**Data preprocessing**

```bash
python src/ingestion/backfill_earnings.py
python src/ingestion/backfill_ohclc.py
python src/ingestion/backfill_transcripts.py
python src/preprocessing/technical_features.py
python src/preprocessing/bert_earnings_call.py
python src/preprocessing/join_validations.py
```

**Model training**

```bash
# ablation ladder (B1–B4): RF + XGBoost classifiers and regressors
jupyter src/modeling/b1-5/b4_pipeline.ipynb

# TabNet on Modal T4 GPU — 20-fold expanding window
python src/modeling/tabnet/upload_model_data.py
modal run src/modeling/tabnet/modal_tabnet_pead.py
python src/modeling/tabnet/fetch_modal_results.py

# DNN multitask
jupyter src/modeling/dnn/multitask_dnn_updates.ipynb
```

**Model evaluation**

```bash
jupyter src/modeling/tabnet/modal_tabnet_analysis_updated_results.ipynb   # per-fold DA/MAE/RMSE, confusion matrix
jupyter src/modeling/tabnet/tabnet_trade_strat_final.ipynb                # trading-strategy backtest
```

## Results

20 expanding-window walk-forward folds. Directional Accuracy (DA) = 3-class hit rate on out-of-sample quarter. Direction classes: low / mid / high peak return over t+1…t+9. Baseline (always majority class): DA = 0.332.

### Ablation ladder (B1–B4) — modality contribution

| Step | Modalities | # Features |
|---|---|---|
| B1 | Technical indicators only | 239 |
| B2 | B1 + Fundamentals + Sector one-hot | 250 |
| B3 | B2 + Transcript sentiment (FinBERT) | 254 |
| B4 | B3 + News sentiment (NZ) | 258 |

#### Random Forest — direction (3-class)

| Step | DA | F1 (weighted) | AUC (OvR) | Δ DA |
|---|---|---|---|---|
| B1 — Technical only | 0.4505 | 0.4170 | 0.6004 | — |
| B2 — + Fundamentals + Sector | 0.4478 | 0.4127 | 0.5994 | −0.0027 |
| B3 — + Transcript sentiment | 0.4442 | 0.4098 | 0.5993 | −0.0036 |
| B4 — + News sentiment | 0.4430 | 0.4090 | 0.5981 | −0.0012 |

#### XGBoost — direction (3-class)

| Step | DA | F1 (weighted) | AUC (OvR) | Δ DA |
|---|---|---|---|---|
| B1 — Technical only | 0.4328 | 0.4092 | 0.5783 | — |
| B2 — + Fundamentals + Sector | 0.4397 | 0.4003 | 0.5792 | +0.0069 |
| B3 — + Transcript sentiment | 0.4273 | 0.4026 | 0.5769 | −0.0124 |
| B4 — + News sentiment | 0.4341 | 0.4102 | 0.5798 | +0.0068 |

#### XGBoost — magnitude (peak return)

| Step | MAE | RMSE | Pooled R² | Δ R² |
|---|---|---|---|---|
| B1 — Technical only | 0.0319 | 0.0455 | −0.0126 | — |
| B2 — + Fundamentals + Sector | 0.0309 | 0.0441 | 0.0410 | +0.0536 |
| B3 — + Transcript sentiment | 0.0320 | 0.0460 | −0.0371 | −0.0781 |
| B4 — + News sentiment | 0.0322 | 0.0461 | −0.0482 | −0.0111 |

Directional accuracy degrades monotonically as modalities are added; technical features dominate (≥92% of TabNet attention).

### TabNet vs DNN — 20-fold walk-forward (direction)

| Model | Mean DA | Mean MAE | Mean RMSE | Best fold DA | Worst fold DA | Test rows |
|---|---|---|---|---|---|---|
| TabNet (initial) | 0.4371 | 0.0313 | 0.0476 | 0.702 (2022 Q4) | 0.299 (2024 Q2) | 9,937 |
| TabNet (updated) | 0.4495 | 0.0303 | 0.0446 | 0.690 (2022 Q4) | 0.335 (2025 Q1) | 9,937 |
| Multitask DNN | 0.4573 | 0.0296 | 0.0428 | 0.700 (2022 Q4) | 0.359 (2024 Q1) | 7,967 |

Both models consistently beat the 0.332 baseline; the DNN edges TabNet on mean DA and RMSE.

### Trading strategy backtest (TabNet predictions, exit at 2% gain or t+10)

Strat 1 enters long when TabNet predicts class 1 or 2, exits at +2% or after 10 trading days. No transaction costs or slippage.

| Metric | Value |
|---|---|
| Initial capital | $50,000 |
| Position size | $200 |
| Trades | 8,221 |
| Skipped (insufficient cash) | 0 |
| Net P&L | $6,637 |
| Win rate | 75.9% |
| Final equity | $56,269 |
| Annualised Sharpe (portfolio) | 1.31 |
| Max drawdown | −4.45% |
| Return on deployed capital | 91.5% |

#### Threshold sweep (best by Sharpe)

| Threshold | Sharpe | Total P&L | Skipped |
|---|---|---|---|
| 2.0% | 2.04 | $6,637 | 0 |
| 5.0% (best) | 2.33 | $12,138 | 13 |

Per-trade ledger exported to `tabnet_trade_ledger.csv` (8,221 rows): symbol, signal/entry/exit dates and prices, threshold, model fold, gross/net P&L.

### Caveats

- Results assume zero transaction costs and slippage — see `src/modeling/tabnet/strat file implementation.md` for the validation TODO list (walk-forward threshold selection, cost grids, deployed-capital Sharpe, drawdown / Calmar / Sortino, benchmark comparison).
- Later folds (2025 Q4) have fewer eligible events; report metrics across all folds rather than tail periods.