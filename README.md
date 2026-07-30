# Post-Earnings Forecast

Predict post-earnings-announcement drift (PEAD) for S&P 500 stocks.

## Project structure

```
.
├── pyproject.toml              # deps, requires-python>=3.12
├── .python-version             # 3.13
├── uv.lock
├── src/
│   ├── ingestion/              # data fetching, backfill, S3 Delta Lake writes
│   │   ├── table_setup.py      # create Delta tables in S3
│   │   ├── backfill_earnings.py
│   │   ├── backfill_ohvlc.py
│   │   ├── backfill_transcripts.py
│   │   ├── transcript_news.py
│   │   ├── index_data.py       # VIX/SPX from yfinance
│   │   ├── upgrades_downgrades.py
│   │   ├── backup.py           # S3 → local Parquet backup + vacuum
│   │   ├── migration.py
│   │   └── archive/
│   ├── preprocessing/          # FinBERT sentiment, technical indicators
│   │   ├── technical_features.py
│   │   ├── bert_earnings_call.py
│   │   ├── bert_compaction_tx.py
│   │   ├── lm_sentiment_tx.py
│   │   ├── vector_emb.py
│   │   ├── sentiment_stg.py
│   │   ├── join_validations.py
│   │   └── analyst_upgrades.py
│   ├── modeling/               # XGBoost, LSTM, TabNet, DNN experiments
│   │   ├── basepaper/
│   │   ├── tabnet/             # Modal GPU TabNet
│   │   ├── dnn/
│   │   ├── b1-5/               # ablation studies
│   │   ├── upload_model_data.py
│   │   └── archive/
│   └── notebooks/              # EDA + pipeline notebooks + marimo
├── docs/                       # planning docs, ablation notes
└── class notes/                # course materials
```

## Setup

```bash
# install dependencies
uv sync

# optional: install TA-Lib system library (macOS)
brew install ta-lib
```

## Environment variables

Create a `.env` file at the project root with:

| Variable | Purpose |
|---|---|
| `AWS_ACCESS_KEY_ID` | S3 + Delta Lake (primary) |
| `AWS_SECRET_ACCESS_KEY` | S3 + Delta Lake (primary) |
| `S3_ACCESS_KEY` | S3 (secondary pattern) |
| `S3_SECRET_KEY` | S3 (secondary pattern) |
| `AWS_SESSION_TOKEN` | S3 session token |
| `S3_BUCKET` | S3 bucket name |
| `AV_PREMIUM_KEY` | Alpha Vantage premium API key |
| `DUCKDB_KEY` | Encrypted Parquet key for DuckDB |
| `HF_TOKEN` | HuggingFace token (Modal GPU jobs) |

AWS region is always `ca-central-1`.

## Key entrypoints

| Script | Purpose |
|---|---|
| `src/ingestion/table_setup.py` | Create Delta tables in S3 |
| `src/ingestion/backfill_earnings.py` | Backfill earnings data to Delta |
| `src/ingestion/backfill_ohvlc.py` | Backfill OHLCV data to Delta |
| `src/ingestion/backfill_transcripts.py` | Backfill transcripts to Delta |
| `src/ingestion/index_data.py` | Download VIX/SPX → local parquet |
| `src/ingestion/upgrades_downgrades.py` | Analyst rating history → local parquet |
| `src/ingestion/backup.py` | Backup Delta tables to local + vacuum |
