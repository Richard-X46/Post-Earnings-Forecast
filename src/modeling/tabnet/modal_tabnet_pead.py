"""Modal TabNet PEAD pipeline — expanding window walk-forward.

upload model staging data to modal volume:
    python src/modeling/upload_model_data.py

run the pipeline on modal cloud:
    modal run src/modeling/modal_tabnet_pead.py

Download results:
    modal volume get tabnet-pead-output /out/results.parquet .
    modal volume get tabnet-pead-output /out/per_fold_summary.parquet .
    modal volume get tabnet-pead-output /out/histories/clf.json .
    modal volume get tabnet-pead-output /out/histories/reg.json .
"""
import modal

# ─────────────────────────────────────────────────────────────────────────
# Image & Container
# ─────────────────────────────────────────────────────────────────────────
image = modal.Image.debian_slim(python_version="3.12").pip_install(
    "pandas",
    "pytorch_tabnet",
    "scikit-learn",
    "torch",
    "polars",
    "optuna",
    force_build=True,
)

with image.imports():
    import json
    import re
    import random
    import time
    import warnings

    import numpy as np
    import polars as pl
    import torch

    from sklearn.preprocessing import StandardScaler
    from sklearn.impute import SimpleImputer
    from sklearn.metrics import (
        accuracy_score,
        mean_absolute_error,
        mean_squared_error,
    )

    from pytorch_tabnet.tab_model import TabNetClassifier, TabNetRegressor
    import optuna

app = modal.App("tabnet-pead", image=image)

# ─────────────────────────────────────────────────────────────────────────
# Modal Volumes (data in, results out)
# ─────────────────────────────────────────────────────────────────────────
DATA_VOL = modal.Volume.from_name("model-staging", create_if_missing=True)
OUT_VOL = modal.Volume.from_name("tabnet-pead-output", create_if_missing=True)

# ─────────────────────────────────────────────────────────────────────────
# Hyperparameter defaults — matching tabnet_pruned.ipynb
# ─────────────────────────────────────────────────────────────────────────
CLF_DEFAULTS = dict(
    n_d=16, n_a=16, n_steps=3, gamma=1.5,
    lambda_sparse=1e-3, lr=2e-2, step_size=30,
    scheduler_gamma=0.9, max_epochs=30, patience=8,
    batch_size=4096, clip_value=2.0, seed=42)

REG_DEFAULTS = dict(
    n_d=16, n_a=16, n_steps=3, gamma=1.5,
    lambda_sparse=1e-3, lr=5e-3, step_size=30,
    scheduler_gamma=0.9, max_epochs=30, patience=8,
    batch_size=4096, clip_value=2.0, seed=42)

DEVICE = "cuda"
INITIAL_TRAIN_YEARS = 7   # 2014–2020
VAL_QUARTERS = 4           # validation window: last N quarters before test

# ─────────────────────────────────────────────────────────────────────────
# Sequence feature grouping -- same as in tabnet_pruned.ipynb
# ─────────────────────────────────────────────────────────────────────────
TECH_BASES = [
    "rsi", "macd", "macd_hist", "roc",
    "ema50_pct", "ema200_pct", "ema50_200_pct", "adx",
    "atr", "bb_width", "bb_pct_b", "sigma",
    "obv_zscore", "vwap_pct",
    "open_pct", "high_pct", "low_pct", "volume_rel",
]
VIX_BASES = ["vix_close"]
SEQ_BASES = TECH_BASES + VIX_BASES  # 19 indicators × 12 timesteps

EXCLUDE_COLS = [
    "symbol", "earnings_date", "entry_price", "target_return",
    "target_class", "max_high", "min_high", "max_day", "min_day",
    "year", "quarter",
]


def group_sequence_columns(all_cols, bases):
    """Group columns by base indicator, ordered by timestep."""
    groups = {}
    remaining = list(all_cols)
    for base in sorted(bases, key=len, reverse=True):
        matched = [c for c in remaining if c == base or c.startswith(base + "_")]

        def step_key(col, base=base):
            m = re.search(r"(-?\d+)", col[len(base):])
            return int(m.group(1)) if m else 0

        groups[base] = sorted(matched, key=step_key)
        remaining = [c for c in remaining if c not in matched]
    return {base: groups[base] for base in bases}


# ─────────────────────────────────────────────────────────────────────────
# Core helpers
# ─────────────────────────────────────────────────────────────────────────
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _extract_history(model):
    """Extract plain dict from TabNet History object (JSON-serialisable)."""
    h = model.history
    if hasattr(h, "history"):
        h = h.history
    return {k: [float(x) for x in v] for k, v in h.items()}


def build_modeling_table(base_path):
    """Load and join tech, fundamental, FinBERT, and news sentiment tables."""
    df_tech = pl.read_parquet(f"{base_path}/tech_modeling_table.parquet")

    df_fund = pl.read_parquet(
        f"{base_path}/fundamentalIndicators/modeling_fundamentals.parquet"
    ).select([
        "symbol",
        pl.col("reportedDate").alias("earnings_date"),
        "eps_growth_qoq", "revenue_growth_qoq",
        "gross_margin", "gross_margin_qoq",
        "debt_to_equity", "debt_to_equity_qoq",
        "fcf_margin", "fcf_margin_qoq",
        "roe", "roe_qoq",
        "surprisePercentage",
    ])

    df_finbert = pl.read_parquet(f"{base_path}/finbert_tx_agg_weighted.parquet")
    df_finbert_feat = df_finbert.select([
        "symbol",
        pl.col("reportedDate").alias("earnings_date"),
        "pos_prob", "neg_prob",
    ])

    df_nz = pl.read_parquet(f"{base_path}/nz_sentiment.parquet").select([
        "symbol",
        pl.col("reportedDate").alias("earnings_date"),
        "overall_sentiment_score_pre", "ticker_sentiment_score_pre",
        "overall_sentiment_score_post", "ticker_sentiment_score_post",
    ])

    df_model = (
        df_tech
        .join(df_fund, on=["symbol", "earnings_date"], how="left")
        .join(df_finbert_feat, on=["symbol", "earnings_date"], how="left")
        .join(df_nz, on=["symbol", "earnings_date"], how="left")
    )

    # Sector one-hot encoding
    df_sector = df_finbert.select(["symbol", "sector"]).unique()
    df_sector = df_sector.with_columns(pl.col("sector").fill_null("Unknown"))
    sectors = sorted(df_sector["sector"].unique().to_list())
    df_sector = df_sector.with_columns([
        (pl.col("sector") == s).cast(pl.Int8).alias(f"sector_{s.replace(' ', '_')}")
        for s in sectors
    ]).drop("sector")

    df_model = df_model.join(df_sector, on="symbol", how="left")

    print(f"Tech table shape: {df_tech.shape}")
    print(f"Combined table shape: {df_model.shape}")
    print(
        f"Target class distribution:\n"
        f"{df_model['target_class'].value_counts().sort('target_class')}"
    )
    return df_model


def prep_block(train_df, val_df, test_df, cols):
    """Fit imputer + scaler on train ONLY. Val and test are transform-only."""
    Xtr = train_df.select(cols).to_numpy()
    Xva = val_df.select(cols).to_numpy()
    Xte = test_df.select(cols).to_numpy()

    Xtr = np.where(np.isinf(Xtr), np.nan, Xtr)
    Xva = np.where(np.isinf(Xva), np.nan, Xva)
    Xte = np.where(np.isinf(Xte), np.nan, Xte)

    imputer = SimpleImputer(strategy="median").fit(Xtr)
    Xtr = imputer.transform(Xtr)
    Xva = imputer.transform(Xva)
    Xte = imputer.transform(Xte)

    scaler = StandardScaler().fit(Xtr)
    return (
        scaler.transform(Xtr).astype(np.float32),
        scaler.transform(Xva).astype(np.float32),
        scaler.transform(Xte).astype(np.float32),
    )


def train_tabnet_clf(f, **kwargs):
    """Train TabNet classifier with early stopping on validation quarter."""
    cfg = CLF_DEFAULTS | kwargs
    set_seed(cfg["seed"])

    clf = TabNetClassifier(
        n_d=cfg["n_d"], n_a=cfg["n_a"], n_steps=cfg["n_steps"],
        gamma=cfg["gamma"], lambda_sparse=cfg["lambda_sparse"],
        clip_value=cfg["clip_value"],
        optimizer_fn=torch.optim.Adam,
        optimizer_params=dict(lr=cfg["lr"]),
        scheduler_fn=torch.optim.lr_scheduler.StepLR,
        scheduler_params=dict(
            step_size=cfg["step_size"], gamma=cfg["scheduler_gamma"]
        ),
        mask_type="entmax",
        verbose=1,
        seed=cfg["seed"],
        device_name=DEVICE,
    )

    # Inverse-frequency class weights from training distribution
    cls_counts = np.bincount(f["y_train_cls"].astype(int), minlength=3)
    cls_weights = len(f["y_train_cls"]) / (3 * cls_counts)
    sample_weights = cls_weights[f["y_train_cls"].astype(int)]

    clf.fit(
        X_train=f["X_train"], y_train=f["y_train_cls"],
        eval_set=[(f["X_val"], f["y_val_cls"])],
        eval_name=["val"], eval_metric=["logloss"],
        max_epochs=cfg["max_epochs"], patience=cfg["patience"],
        batch_size=cfg["batch_size"], virtual_batch_size=128,
        weights=sample_weights,
    )

    train_preds = clf.predict(f["X_train"]).ravel()
    val_preds = clf.predict(f["X_val"]).ravel()
    test_probs = clf.predict_proba(f["X_test"])
    test_preds = test_probs.argmax(axis=1)

    feat_imp = {
        "clf": clf.feature_importances_.tolist(),
    } if hasattr(clf, "feature_importances_") else {}

    # Attention masks via explain() on a subset of test data
    masks = None
    try:
        _, raw_masks = clf.explain(f["X_test"][:200])  # returns (M_explain, masks_dict)
        masks = {str(k): v.tolist() if hasattr(v, "tolist") else v for k, v in raw_masks.items()}
    except Exception:
        pass

    return {
        "clf_preds": test_preds,
        "clf_probs": test_probs,
        "train_preds": train_preds,
        "val_preds": val_preds,
        "train_actual": f["y_train_cls"],
        "val_actual": f["y_val_cls"],
        "best_epoch": clf.best_epoch,
        "model": clf,
        "history": clf.history,
        "feature_importance": feat_imp,
        "attention_masks": masks,
    }


def train_tabnet_reg(f, **kwargs):
    """Train TabNet regressor with early stopping on validation quarter."""
    cfg = REG_DEFAULTS | kwargs
    set_seed(cfg["seed"])

    reg = TabNetRegressor(
        n_d=cfg["n_d"], n_a=cfg["n_a"], n_steps=cfg["n_steps"],
        gamma=cfg["gamma"], lambda_sparse=cfg["lambda_sparse"],
        clip_value=cfg["clip_value"],
        optimizer_fn=torch.optim.Adam,
        optimizer_params=dict(lr=cfg["lr"]),
        scheduler_fn=torch.optim.lr_scheduler.StepLR,
        scheduler_params=dict(
            step_size=cfg["step_size"], gamma=cfg["scheduler_gamma"]
        ),
        mask_type="entmax",
        verbose=1,
        seed=cfg["seed"],
        device_name=DEVICE,
    )

    y_mean = float(f["y_train_ret"].mean())
    y_std = float(f["y_train_ret"].std())
    y_train_scaled = ((f["y_train_ret"] - y_mean) / y_std).reshape(-1, 1).astype(np.float32)
    y_val_scaled = ((f["y_val_ret"] - y_mean) / y_std).reshape(-1, 1).astype(np.float32)

    reg.fit(
        X_train=f["X_train"], y_train=y_train_scaled,
        eval_set=[(f["X_val"], y_val_scaled)],
        eval_name=["val"], eval_metric=["rmse"],
        max_epochs=cfg["max_epochs"], patience=cfg["patience"],
        batch_size=cfg["batch_size"], virtual_batch_size=128,
    )

    train_preds_scaled = reg.predict(f["X_train"]).ravel()
    val_preds_scaled = reg.predict(f["X_val"]).ravel()
    test_preds_scaled = reg.predict(f["X_test"]).ravel()

    train_preds = (train_preds_scaled * y_std) + y_mean
    val_preds = (val_preds_scaled * y_std) + y_mean
    test_preds = (test_preds_scaled * y_std) + y_mean

    feat_imp = {
        "reg": reg.feature_importances_.ravel().tolist(),
    } if hasattr(reg, "feature_importances_") else {}

    # Attention masks via explain() on a subset of test data
    masks = None
    try:
        _, raw_masks = reg.explain(f["X_test"][:200])  # returns (M_explain, masks_dict)
        masks = {str(k): v.tolist() if hasattr(v, "tolist") else v for k, v in raw_masks.items()}
    except Exception:
        pass

    return {
        "reg_preds": test_preds,
        "train_preds": train_preds,
        "val_preds": val_preds,
        "train_actual": f["y_train_ret"],
        "val_actual": f["y_val_ret"],
        "best_epoch": reg.best_epoch,
        "model": reg,
        "history": reg.history,
        "feature_importance": feat_imp,
        "attention_masks": masks,
    }


# ─────────────────────────────────────────────────────────────────────────
# Modal cloud function
# ─────────────────────────────────────────────────────────────────────────
@app.function(
    gpu="t4",
    timeout=60 * 25,
    volumes={"/data": DATA_VOL, "/out": OUT_VOL},
)
def run_pead_pipeline():
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", message=".*pin_memory.*")
    warnings.filterwarnings("ignore", message=".*Best weights from best epoch.*")

    # ── 1. Load & prep data ────────────────────────────────────────────
    df_model = build_modeling_table("/data")

    df_model = df_model.drop(
        [col for col in df_model.columns if col.startswith("car")]
    )
    df_model = df_model.with_columns([
        pl.col("earnings_date").dt.year().alias("year"),
        pl.col("earnings_date").dt.quarter().alias("quarter"),
    ])
    df_model = df_model.filter(pl.col("year") <= 2025)

    # Label-encode symbol
    df_model = df_model.with_columns(
        pl.col("symbol").cast(pl.Categorical).to_physical().alias("symbol_enc")
    )

    feature_cols = [c for c in df_model.columns if c not in EXCLUDE_COLS]
    seq_groups = group_sequence_columns(feature_cols, SEQ_BASES)
    timestep_counts = {len(c) for c in seq_groups.values()}
    if len(timestep_counts) != 1:
        raise ValueError(f"Inconsistent timestep counts: {timestep_counts}")
    n_timesteps = timestep_counts.pop()
    seq_cols_ordered = [c for base in SEQ_BASES for c in seq_groups[base]]
    other_cols = [c for c in feature_cols if c not in seq_cols_ordered]
    flat_feature_cols = seq_cols_ordered + other_cols

    assert len(flat_feature_cols) == len(feature_cols), (
        f"Feature count mismatch: flat={len(flat_feature_cols)}, "
        f"original={len(feature_cols)}"
    )
    print(f"Sequence block: {len(seq_cols_ordered)} cols "
          f"({len(SEQ_BASES)} indicators × {n_timesteps} steps)")
    print(f"Other (fundamentals/sentiment/sector): {len(other_cols)} cols")
    print(f"Total flat features: {len(flat_feature_cols)}")

    # ── 2. Fold construction ───────────────────────────────────────────
    min_date = df_model["earnings_date"].min()
    max_date = df_model["earnings_date"].max()
    print(f"Dataset spans: {min_date} to {max_date}")
    print(f"Total rows: {len(df_model)}")

    quarters = df_model.select(["year", "quarter"]).unique().sort(["year", "quarter"])
    quarters_list = quarters.to_dicts()

    earliest_year = quarters_list[0]["year"]
    initial_train_end_year = earliest_year + INITIAL_TRAIN_YEARS - 1
    first_test_idx = next(
        i for i, q in enumerate(quarters_list)
        if q["year"] > initial_train_end_year
    )

    print(f"Initial training: {earliest_year}–{initial_train_end_year} "
          f"({INITIAL_TRAIN_YEARS} yrs)")
    print(f"Testing begins at: {quarters_list[first_test_idx]}\n")

    folds_data = []
    fold_num = 1

    for test_quarter_idx in range(first_test_idx, len(quarters_list)):
        test_q = quarters_list[test_quarter_idx]
        val_start_idx = max(0, test_quarter_idx - VAL_QUARTERS)
        val_end_idx = test_quarter_idx - 1
        val_qs = quarters_list[val_start_idx:val_end_idx + 1]
        test_label = f"{test_q['year']}_Q{test_q['quarter']}"

        test_data = df_model.filter(
            (pl.col("year") == test_q["year"])
            & (pl.col("quarter") == test_q["quarter"])
        )

        val_conditions = None
        for vq in val_qs:
            cond = ((pl.col("year") == vq["year"]) & (pl.col("quarter") == vq["quarter"]))
            val_conditions = cond if val_conditions is None else (val_conditions | cond)
        val_data = df_model.filter(val_conditions) if val_conditions is not None else df_model.filter(pl.lit(False))

        train_end = val_qs[0]
        train_data = df_model.filter(
            (pl.col("year") < train_end["year"])
            | ((pl.col("year") == train_end["year"])
               & (pl.col("quarter") < train_end["quarter"]))
        )

        if len(train_data) == 0 or len(val_data) == 0 or len(test_data) == 0:
            continue

        X_tr, X_va, X_te = prep_block(train_data, val_data, test_data, flat_feature_cols)

        folds_data.append({
            "fold_num": fold_num,
            "test_quarter": test_label,
            "X_train": X_tr,
            "X_val": X_va,
            "X_test": X_te,
            "y_train_cls": train_data["target_class"].to_numpy(),
            "y_val_cls": val_data["target_class"].to_numpy(),
            "y_test_cls": test_data["target_class"].to_numpy(),
            "y_train_ret": train_data["target_return"].to_numpy(),
            "y_val_ret": val_data["target_return"].to_numpy(),
            "y_test_ret": test_data["target_return"].to_numpy(),
            "symbol": test_data["symbol"].to_list(),
            "earnings_date": test_data["earnings_date"].to_list(),
            "entry_price": test_data["entry_price"].to_list(),
        })

        print(f"Fold {fold_num:2d} [{test_label}]:  "
              f"train={X_tr.shape[0]:6,}  "
              f"val={X_va.shape[0]:4,} ({val_qs[0]['year']}_Q{val_qs[0]['quarter']}–"
              f"{val_qs[-1]['year']}_Q{val_qs[-1]['quarter']})  "
              f"test={X_te.shape[0]:4,}")
        fold_num += 1

    print(f"\nTotal folds: {len(folds_data)}")
    print(f"n_features: {len(flat_feature_cols)}")

    # ── 3. Walk-forward training ───────────────────────────────────────
    # Collection buckets
    clf_col = {"fold_acc": [], "preds": [], "probs": [], "true": [],
               "quarters": [], "symbol": [], "earnings_date": [],
               "entry_price": [], "fold_nums": [], "history": [],
               "train_preds": [], "val_preds": [], "train_true": [], "val_true": []}
    reg_col = {"fold_mae": [], "fold_rmse": [], "preds": [], "true": [],
               "history": [], "train_preds": [], "val_preds": [], "train_true": [],
               "val_true": []}
    feat_imp_clf = []
    feat_imp_reg = []
    attn_masks_clf = []
    attn_masks_reg = []

    for f in folds_data:
        t0 = time.time()

        out_clf = train_tabnet_clf(f, seed=42 + f["fold_num"])
        out_reg = train_tabnet_reg(f, seed=42 + f["fold_num"])

        acc = accuracy_score(f["y_test_cls"], out_clf["clf_preds"])
        mae = mean_absolute_error(f["y_test_ret"], out_reg["reg_preds"])
        rmse = float(np.sqrt(mean_squared_error(f["y_test_ret"], out_reg["reg_preds"])))

        clf_col["fold_acc"].append(acc)
        clf_col["preds"].extend(out_clf["clf_preds"].tolist())
        clf_col["probs"].append(out_clf["clf_probs"])
        clf_col["true"].extend(f["y_test_cls"].tolist())
        clf_col["quarters"].append(f["test_quarter"])
        clf_col["symbol"].extend(f["symbol"])
        clf_col["earnings_date"].extend(f["earnings_date"])
        clf_col["entry_price"].extend(f["entry_price"])
        clf_col["fold_nums"].extend([f["fold_num"]] * len(f["y_test_cls"]))
        clf_col["history"].append(_extract_history(out_clf["model"]))
        clf_col["train_preds"].append(out_clf["train_preds"].tolist())
        clf_col["val_preds"].append(out_clf["val_preds"].tolist())
        clf_col["train_true"].append(out_clf["train_actual"].tolist())
        clf_col["val_true"].append(out_clf["val_actual"].tolist())

        reg_col["fold_mae"].append(mae)
        reg_col["fold_rmse"].append(rmse)
        reg_col["preds"].extend(out_reg["reg_preds"].tolist())
        reg_col["true"].extend(f["y_test_ret"].tolist())
        reg_col["history"].append(_extract_history(out_reg["model"]))
        reg_col["train_preds"].append(out_reg["train_preds"].tolist())
        reg_col["val_preds"].append(out_reg["val_preds"].tolist())
        reg_col["train_true"].append(out_reg["train_actual"].tolist())
        reg_col["val_true"].append(out_reg["val_actual"].tolist())

        feat_imp_clf.append(out_clf.get("feature_importance", {}))
        feat_imp_reg.append(out_reg.get("feature_importance", {}))
        attn_masks_clf.append(out_clf.get("attention_masks"))
        attn_masks_reg.append(out_reg.get("attention_masks"))

        if f["fold_num"] == 1:
            print(f"\n  Attn masks check — CLF: {type(attn_masks_clf[-1]).__name__}, REG: {type(attn_masks_reg[-1]).__name__}")
            if attn_masks_clf[-1]:
                print(f"  CLF masks steps: {list(attn_masks_clf[-1].keys())}")
                for k, v in attn_masks_clf[-1].items():
                    arr = np.array(v)
                    print(f"    step {k}: shape={arr.shape}, mean={arr.mean():.5f}, nonzero={(arr>0).mean():.2%}")
            if attn_masks_reg[-1]:
                print(f"  REG masks steps: {list(attn_masks_reg[-1].keys())}")
                for k, v in attn_masks_reg[-1].items():
                    arr = np.array(v)
                    print(f"    step {k}: shape={arr.shape}, mean={arr.mean():.5f}, nonzero={(arr>0).mean():.2%}")
            print()

        elapsed = time.time() - t0
        pred_dist = (
            np.bincount(out_clf["clf_preds"].astype(int), minlength=3)
            if len(out_clf["clf_preds"]) > 0 else np.zeros(3, dtype=int)
        )
        print(f"Fold {f['fold_num']:2d} [{f['test_quarter']}]: "
              f"DA={acc:.4f}  MAE={mae:.4f}  RMSE={rmse:.4f}  "
              f"pred_dist={pred_dist.tolist()}  ({elapsed:.0f}s)\n")

    # ── 4. Build results table ─────────────────────────────────────────
    probs = np.vstack(clf_col["probs"])

    # Map fold_num → quarter (clf_col["quarters"] is fold-level, 1 per fold)
    fold_to_quarter = {i + 1: q for i, q in enumerate(clf_col["quarters"])}
    quarter_flat = [fold_to_quarter[fn] for fn in clf_col["fold_nums"]]

    results_df = pl.DataFrame({
        "quarter": pl.Series(quarter_flat, dtype=pl.Utf8),
        "fold_num": pl.Series(clf_col["fold_nums"], dtype=pl.Int64),
        "symbol": pl.Series(clf_col["symbol"], dtype=pl.Utf8),
        "earnings_date": pl.Series(clf_col["earnings_date"], dtype=pl.Date),
        "entry_price": pl.Series(clf_col["entry_price"], dtype=pl.Float64),
        "target_class_actual": pl.Series(clf_col["true"], dtype=pl.Int64),
        "target_class_predicted": pl.Series(clf_col["preds"], dtype=pl.Int64),
        "prob_class_0": pl.Series(probs[:, 0], dtype=pl.Float32),
        "prob_class_1": pl.Series(probs[:, 1], dtype=pl.Float32),
        "prob_class_2": pl.Series(probs[:, 2], dtype=pl.Float32),
        "target_return_actual": pl.Series(reg_col["true"], dtype=pl.Float64),
        "target_return_predicted": pl.Series(reg_col["preds"], dtype=pl.Float64),
    })

    # ── 5. Per-fold summary ────────────────────────────────────────────
    summary_df = pl.DataFrame({
        "quarter": pl.Series(clf_col["quarters"], dtype=pl.Utf8),
        "fold_num": pl.Series(range(1, len(clf_col["quarters"]) + 1), dtype=pl.Int64),
        "DA": pl.Series(clf_col["fold_acc"], dtype=pl.Float64),
        "MAE": pl.Series(reg_col["fold_mae"], dtype=pl.Float64),
        "RMSE": pl.Series(reg_col["fold_rmse"], dtype=pl.Float64),
    })

    # ── 6. Histories as JSON ───────────────────────────────────────────
    clf_histories = {
        q: h
        for q, h in zip(clf_col["quarters"], clf_col["history"])
    }
    reg_histories = {
        q: h
        for q, h in zip(clf_col["quarters"], reg_col["history"])
    }

    # ── 7. Save everything to Volume ───────────────────────────────────
    results_df.write_parquet("/out/e30_results.parquet")
    summary_df.write_parquet("/out/e30_per_fold_summary.parquet")

    with open("/out/e30_histories_clf.json", "w") as f:
        json.dump(clf_histories, f, indent=2)
    with open("/out/e30_histories_reg.json", "w") as f:
        json.dump(reg_histories, f, indent=2)

    # Save feature importance per fold
    feat_imp_data = {
        "feature_names": flat_feature_cols,
        "clf": {q: imp for q, imp in zip(clf_col["quarters"], feat_imp_clf)},
        "reg": {q: imp for q, imp in zip(clf_col["quarters"], feat_imp_reg)},
    }
    with open("/out/e30_feature_importance.json", "w") as f:
        json.dump(feat_imp_data, f, indent=2)

    # Save attention masks per fold
    attn_data = {
        "feature_names": flat_feature_cols,
        "clf": {q: masks for q, masks in zip(clf_col["quarters"], attn_masks_clf)},
        "reg": {q: masks for q, masks in zip(clf_col["quarters"], attn_masks_reg)},
    }
    with open("/out/e30_attention_masks.json", "w") as f:
        json.dump(attn_data, f, indent=2)
    print(f"Attention masks saved: CLF={sum(1 for m in attn_masks_clf if m)} folds, REG={sum(1 for m in attn_masks_reg if m)} folds")

    print(f"Results DataFrame: {results_df.shape}")
    print(results_df.head(5))

    print(f"\n{'='*60}")
    print("PER-FOLD SUMMARY")
    print(f"{'='*60}")
    print(f"{'Quarter':<12}  {'DA':>6}  {'MAE':>8}  {'RMSE':>8}")
    print(f"{'-'*42}")
    for i, q in enumerate(clf_col["quarters"]):
        print(f"{q:<12}  {clf_col['fold_acc'][i]:.4f}  "
              f"{reg_col['fold_mae'][i]:.4f}  {reg_col['fold_rmse'][i]:.4f}")
    print(f"{'-'*42}")
    print(f"{'Avg':<12}  {np.mean(clf_col['fold_acc']):.4f}  "
          f"{np.mean(reg_col['fold_mae']):.4f}  {np.mean(reg_col['fold_rmse']):.4f}")

    # ── 8. Return summary to laptop ────────────────────────────────────
    return {
        "avg_da": float(np.mean(clf_col["fold_acc"])),
        "avg_mae": float(np.mean(reg_col["fold_mae"])),
        "avg_rmse": float(np.mean(reg_col["fold_rmse"])),
        "n_folds": len(clf_col["quarters"]),
        "results_rows": len(results_df),
    }


# ─────────────────────────────────────────────────────────────────────────
# Local entrypoint
# ─────────────────────────────────────────────────────────────────────────
@app.local_entrypoint()
def main():
    result = run_pead_pipeline.remote()
    print(f"\n{'='*60}")
    print("PIPELINE COMPLETE")
    print(f"{'='*60}")
    print(f"Avg DA:   {result['avg_da']:.4f}")
    print(f"Avg MAE:  {result['avg_mae']:.4f}")
    print(f"Avg RMSE: {result['avg_rmse']:.4f}")
    print(f"Folds:    {result['n_folds']}")
    print(f"Predictions saved: {result['results_rows']}")
    print(f"\nDownload results:")
    print(f"  modal volume get tabnet-pead-output e30_results.parquet .")
    print(f"  modal volume get tabnet-pead-output e30_per_fold_summary.parquet .")
    print(f"  modal volume get tabnet-pead-output e30_histories_clf.json .")
    print(f"  modal volume get tabnet-pead-output e30_histories_reg.json .")
    print(f"  modal volume get tabnet-pead-output e30_feature_importance.json .")
    print(f"  modal volume get tabnet-pead-output e30_attention_masks.json .")


# ─────────────────────────────────────────────────────────────────────────
# Cost math
# ─────────────────────────────────────────────────────────────────────────
cost_t4_second = 0.000164
cost_t4_hour = cost_t4_second * 3600
hours_i_can_run = 30 / cost_t4_hour
