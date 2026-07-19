# Modal TabNet Pipeline — File Map

## Flow

```
upload_model_data.py          modal_tabnet.py              modal_tabnet_pead.py
  (upload parquets to          (toy smoke test on            (full PEAD pipeline
   Modal Volume — once)         Covtype dataset)              on T4 GPU, 20 folds)
       │                             │                              │
       ▼                             ▼                              ▼
 model-staging Volume          ┌─ Image build (cached)        ┌─ Image build (cached)
 (persistent cloud storage)    │─ T4 GPU provisions           │─ T4 GPU provisions
       │                       │─ fetch_covtype()            │─ Reads /data parquets
       │                       │─ TabNetClassifier.fit()     │─ Expanding window folds
       ▼                       │─ Returns accuracy dict      │─ TabNet clf + reg per fold
 modal_tabnet_pead.py          └─ prints to terminal         │─ Writes 4 files to Volume
 (reads from /data Volume)                                   └─ prints to terminal
       │
       ▼                                              fetch_modal_results.py
 tabnet-pead-output Volume                             (downloads Volume outputs
 (persistent cloud storage)                             to local folder — post-run)
       │
       ▼
 fetch_modal_results.py ──> modal_tabnet_outputs/
                              ├── results.parquet
                              ├── per_fold_summary.parquet
                              ├── histories_clf.json
                              └── histories_reg.json
                                       │
                                       ▼
                              modal_tabnet_analysis_results.ipynb
                              (loads outputs, plots: confusion matrix,
                               DA/MAE/RMSE bars, loss curves, box plots)
                                       │
                                       ▼
                              tabnet_trade_strat.ipynb
                              (trading strategy backtest: buys on
                               class 1/2, exits at 2% or t10,
                               equity curve, Sharpe, quarterly P&L)
```

---

## Scripts (run once or on-demand)

| File | Purpose | Run with |
|---|---|---|
| `upload_model_data.py` | Uploads 4 model_staging parquets (~42 MB) into Modal Volume `model-staging`. One-time setup before first pipeline run. | `python src/modeling/upload_model_data.py` |
| `modal_tabnet.py` | Toy smoke test: TabNetClassifier on sklearn's Covertype dataset (581K rows). Proves Modal + T4 + CUDA + pytorch_tabnet works end-to-end. ~30 seconds on T4. | `modal run src/modeling/modal_tabnet.py` |
| `modal_tabnet_pead.py` | Full PEAD pipeline: loads S&P 500 earnings data from Modal Volume, constructs 20-fold expanding window (2014-2025), trains TabNet classifier + regressor per fold, saves results/histories to Volume. ~10-15 min on T4. | `modal run src/modeling/modal_tabnet_pead.py` |
| `fetch_modal_results.py` | Downloads all 4 output files from Modal Volume `tabnet-pead-output` into `modal_tabnet_outputs/`. Run after pipeline completes. | `python src/modeling/fetch_modal_results.py` |

## Notebooks (analysis, run locally)

| File | Purpose |
|---|---|
| `modal_tabnet_analysis_results.ipynb` | Loads `results.parquet`, `per_fold_summary.parquet`, and JSON histories. Plots: confusion matrix, per-fold DA/MAE/RMSE, loss curves, actual return distribution by predicted class. |
| `tabnet_trade_strat.ipynb` | Trading strategy backtest: buys when TabNet predicts class 1 or 2, exits at 2% gain or t+10 close. Equity curve, quarterly/yearly P&L, Sharpe ratio, drawdown. |

## Output data (downloaded, not version-controlled)

| File | Contents | Rows × Cols |
|---|---|---|
| `modal_tabnet_outputs/results.parquet` | Per-prediction table: symbol, earnings_date, entry_price, predicted class, class probabilities, actual/predicted returns | ~9,937 × 12 |
| `modal_tabnet_outputs/per_fold_summary.parquet` | Per-fold metrics: DA, MAE, RMSE | 20 × 5 |
| `modal_tabnet_outputs/histories_clf.json` | Classifier per-epoch loss curves (all 20 folds) | JSON dict |
| `modal_tabnet_outputs/histories_reg.json` | Regressor per-epoch loss curves (all 20 folds) | JSON dict |
| `modal_tabnet_outputs/.gitkeep` | Placeholder to keep the directory in git | empty |

## Modal Volumes (cloud storage)

| Volume name | Mount path | Contents | Lifecycle |
|---|---|---|---|
| `model-staging` | `/data` in container | Input: 4 parquet files from `upload_model_data.py` | Persistent |
| `tabnet-pead-output` | `/out` in container | Output: results.parquet, per_fold_summary.parquet, histories JSONs | Persistent |

---

## Prerequisites

1. **Modal account** with `modal` CLI installed and authenticated
2. **$30/month free credits** on Starter plan (T4 at $0.59/hr — ~50 hours of training free)
3. **`model_staging/` parquets** present locally for upload
4. **A Modal `Secret` named `aws-creds`** if reading from S3 instead of local upload (not used in current flow — data lives in Volume)
