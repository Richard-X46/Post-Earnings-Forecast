# src/report/data — vendored minimal subset (~44MB)

Copied verbatim from the repo so the Quarto report renders offline with no
S3, yfinance, Alpha Vantage, Modal, or GPU. Do not edit; re-copy from source
if upstream changes.

| File | Source | Rows × Cols | Used for |
|---|---|---|---|
| `tech_modeling_table.parquet` | `src/data/model_staging/tech_modeling_table.parquet` | 24,048 × 243 | target hist, class mix, feature-count demo |
| `modeling_fundamentals.parquet` | `src/data/model_staging/modeling_fundamentals.parquet` | 24,294 × 18 | EPS surprise, beat/miss, fundamentals medians |
| `company_info.parquet` | `src/data/raw/metrics/snp500_info.parquet` (slim: symbol, company, sector, marketCap) | 503 × 5 | treemap names, market-cap sizing |
| `strat_table.parquet` | `src/data/model_staging/strat_table.parquet` | 23,117 × 31 | live Strat-1 backtest demo (forward highs/lows t+2..t+10) |
| `nz_sentiment.parquet` | `src/data/model_staging/nz_sentiment.parquet` | 24,294 × 10 | news sentiment pre/post means |
| `finbert_tx_agg_weighted.parquet` | `src/data/model_staging/finbert_tx_agg_weighted.parquet` | 24,294 × 6 | transcript sentiment means |
| `tabnet_updated_results.parquet` | `.../modal_tabnet_outputs/updated_results.parquet` | 9,937 × 12 | TabNet per-prediction table (updated trial), DA recompute, win-by-confidence |
| `tabnet_results.parquet` | `.../modal_tabnet_outputs/results.parquet` | 9,937 × 12 | TabNet per-prediction table (notebook trial: the file TABNET_STRAT_4 trades), §8 replay |
| `tabnet_updated_per_fold_summary.parquet` | `.../modal_tabnet_outputs/updated_per_fold_summary.parquet` | 20 × 5 | TabNet per-fold DA/MAE/RMSE bar |
| `dnn_results.parquet` | `.../dnn_outputs/results.parquet` | 7,967 × 12 | DNN win-by-confidence |
| `dnn_per_fold_summary.parquet` | `.../dnn_outputs/per_fold_summary.parquet` | 16 × 5 | DNN per-fold means |

Deliberately excluded: `tech_daily_table.parquet` (594M), S3 backups (~2G),
per-symbol `finbert_tx/` (306M), `transcript_staging/` (72M).
