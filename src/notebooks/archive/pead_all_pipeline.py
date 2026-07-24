"""
PEAD Pipeline - Data loading, feature engineering, and event-window construction.
"""

import os
import numpy as np
import polars as pl
import talib
import yfinance as yf
from bisect import bisect_left
from dotenv import load_dotenv

load_dotenv()

# S3 Configuration
STORAGE_OPTIONS = {
    "AWS_ACCESS_KEY_ID": os.getenv("AWS_ACCESS_KEY_ID"),
    "AWS_SECRET_ACCESS_KEY": os.getenv("AWS_SECRET_ACCESS_KEY"),
    "AWS_REGION": "ca-central-1",
}
BUCKET = os.getenv("S3_BUCKET")

# Data Loading Functions
def load_earnings(symbol=None):
    """Load earnings from Delta Lake. Cast EPS columns from string → Float64."""
    path = f"s3://{BUCKET}/post-earnings-forecast/earnings_delta/"
    df = pl.scan_delta(path, storage_options=STORAGE_OPTIONS)

    if symbol:
        df = df.filter(pl.col("symbol") == symbol)

    df = df.collect()

    # Cast "None" strings to nulls and convert to Float64
    cast_cols = ["reportedEPS", "estimatedEPS", "surprise", "surprisePercentage"]
    df = df.with_columns([
        pl.when(pl.col(c) == "None").then(None).otherwise(pl.col(c)).cast(pl.Float64).alias(c)
        for c in cast_cols
    ])

    print(f"Earnings loaded: {df.shape} | Symbols: {df['symbol'].n_unique()}")
    return df


def load_transcripts(symbol=None):
    """Load transcripts from Delta Lake."""
    path = f"s3://{BUCKET}/post-earnings-forecast/transcripts_delta/"
    df = pl.scan_delta(path, storage_options=STORAGE_OPTIONS)

    if symbol:
        df = df.filter(pl.col("symbol") == symbol)

    df = df.collect()
    print(f"Transcripts loaded: {df.shape} | Symbols: {df['symbol'].n_unique()}")
    return df


def load_ohlcv(symbol=None):
    """Load OHLCV from Delta Lake."""
    path = f"s3://{BUCKET}/post-earnings-forecast/ohlcv_delta/"
    df = pl.read_delta(path, storage_options=STORAGE_OPTIONS)

    if symbol:
        df = df.filter(pl.col("symbol") == symbol)

    print(f"OHLCV loaded: {df.shape} | Symbols: {df['symbol'].n_unique()}")
    return df



# Sector / Industry Lookup
def load_sector_info(symbols, cache_dir="data"):
    """Fetch sector and industry for each symbol from yfinance.
    """
    cache_path = os.path.join(cache_dir, "sector_info.parquet")

    if os.path.exists(cache_path):
        df = pl.read_parquet(cache_path)
        print(f"Sector info loaded from cache: {df.shape}")
        return df

    os.makedirs(cache_dir, exist_ok=True)
    records = []
    for i, sym in enumerate(symbols):
        try:
            info = yf.Ticker(sym).info
            records.append({
                "symbol":   sym,
                "sector":   info.get("sector",   None),
                "industry": info.get("industry", None),
            })
        except Exception:
            records.append({"symbol": sym, "sector": None, "industry": None})

        if (i + 1) % 50 == 0 or (i + 1) == len(symbols):
            print(f"  Sector info: {i + 1}/{len(symbols)} symbols")

    df = pl.DataFrame(records)
    df.write_parquet(cache_path)
    print(f"Sector info saved: {df.shape} | Nulls: {df.null_count().sum_horizontal()[0]}")
    return df


# Technical Indicator Functions
def compute_all_ta(ohlcv):
    """Compute RSI, MACD, Bollinger Band derived features, ATR, SMA50/200 ratios,
    ROC, ADX, OBV, and VWAP in one loop (aligned with Hajek et al., 2025).
    """
    results = []
    symbols = ohlcv["symbol"].unique().sort().to_list()

    for i, symbol in enumerate(symbols):
        stock = ohlcv.filter(pl.col("symbol") == symbol).sort("date")

        close  = stock["close"].to_numpy()
        high   = stock["high"].to_numpy()
        low    = stock["low"].to_numpy()
        volume = stock["volume"].to_numpy().astype(np.float64)

        rsi                        = talib.RSI(close, timeperiod=14)
        macd, _, hist              = talib.MACD(close, 12, 26, 9)
        bb_upper, bb_mid, bb_lower = talib.BBANDS(close, timeperiod=20)
        atr                        = talib.ATR(high, low, close, timeperiod=14)
        sma50                      = talib.SMA(close, timeperiod=50)
        sma200                     = talib.SMA(close, timeperiod=200)

        # Base paper indicators 
        roc = talib.ROC(close, timeperiod=10)            # Rate of Change - momentum speed
        adx = talib.ADX(high, low, close, timeperiod=14) # Average Directional Index - trend strength
        obv = talib.OBV(close, volume)                   # On-Balance Volume - accumulation/distribution

        # VWAP ratio (rolling 20-day VWAP normalized by close for cross-stock comparability)
        typical_price = (high + low + close) / 3.0
        tp_vol = typical_price * volume
        window = 20
        cum_tp_vol = np.convolve(tp_vol, np.ones(window), mode="full")[:len(close)]
        cum_vol    = np.convolve(volume,  np.ones(window), mode="full")[:len(close)]
        with np.errstate(invalid="ignore", divide="ignore"):
            vwap = cum_tp_vol / cum_vol
            vwap_ratio = close / vwap
        vwap_ratio[:window - 1] = np.nan  # warmup period

        # Derived Bollinger Band features
        bb_width = (bb_upper - bb_lower) / bb_mid
        with np.errstate(invalid="ignore", divide="ignore"):
            bb_pct_b = (close - bb_lower) / (bb_upper - bb_lower)

        # SMA ratios: >1 = price above average (bullish), <1 = below (bearish)
        # sma200_ratio will be NaN for the first ~200 trading days per stock
        sma50_ratio     = close / sma50
        sma200_ratio    = close / sma200
        # Golden cross (>1) vs death cross (<1) regime indicator
        sma50_200_ratio = sma50 / sma200

        # Normalize OBV per symbol (z-score) so it's comparable across stocks
        obv_mean = np.nanmean(obv)
        obv_std  = np.nanstd(obv)
        obv_norm = (obv - obv_mean) / obv_std if obv_std > 0 else obv * 0.0

        results.append(
            stock.select("symbol", "date").with_columns([
                pl.Series("rsi",             rsi),
                pl.Series("macd",            macd),
                pl.Series("macdhist",        hist),
                pl.Series("bb_width",        bb_width),
                pl.Series("bb_pct_b",        bb_pct_b),
                pl.Series("atr",             atr),
                pl.Series("sma50_ratio",     sma50_ratio),
                pl.Series("sma200_ratio",    sma200_ratio),
                pl.Series("sma50_200_ratio", sma50_200_ratio),
                pl.Series("roc",             roc),
                pl.Series("adx",             adx),
                pl.Series("obv",             obv_norm),
                pl.Series("vwap_ratio",      vwap_ratio),
            ])
        )

        if (i + 1) % 50 == 0 or (i + 1) == len(symbols):
            print(f"  TA indicators: {i + 1}/{len(symbols)} symbols")

    return pl.concat(results)


# Derived Features
def compute_derived(ohlcv):
    """ daily_return, intraday_range, volume_change.
    """
    df = ohlcv.sort(["symbol", "date"])

    return df.select([
        "symbol", "date",
        pl.col("close").pct_change().over("symbol").alias("daily_return"),
        ((pl.col("high") - pl.col("low")) / pl.col("close")).alias("intraday_range"),
        pl.col("volume").cast(pl.Float64).pct_change().over("symbol").alias("volume_change"),
    ])


# EPS Features
def add_eps_features(earnings):
    """Add eps_growth, surprise_trend, beat, consecutive_beats.
    """
    earnings = earnings.sort(["symbol", "reportedDate"])

    earnings = earnings.with_columns([
        (
            (pl.col("reportedEPS") - pl.col("reportedEPS").shift(1).over("symbol"))
            / pl.col("reportedEPS").shift(1).over("symbol").abs()
        ).alias("eps_growth"),
        (pl.col("surprise") - pl.col("surprise").shift(1).over("symbol")).alias("surprise_trend"),
        (pl.col("surprise") > 0).cast(pl.Int8).alias("beat"),
    ])

    # Consecutive beats (requires sequential logic)
    streaks = []
    for symbol in earnings["symbol"].unique().sort().to_list():
        stock = earnings.filter(pl.col("symbol") == symbol).sort("reportedDate")
        count = 0
        for b in stock["beat"].to_list():
            count = count + 1 if b == 1 else 0
            streaks.append(count)

    earnings = earnings.with_columns(pl.Series("consecutive_beats", streaks))
    print(f"EPS features added for {earnings['symbol'].n_unique()} symbols")
    return earnings


# Target Variable Construction
def compute_target_from_daily(daily):
    """For each earnings event, compute the target return over a specified window.
    """
    target_days = daily.filter(
        pl.col("relative_day").is_between(2, 10)
    ).sort(["symbol", "earnings_date", "relative_day"])

    target = target_days.group_by(["symbol", "earnings_date"]).agg(
        ((1 + pl.col("daily_return")).product() - 1).alias("target_return")
    )

    target = target.with_columns(
        (pl.col("target_return") > 0).cast(pl.Int8).alias("target_direction")
    )

    print(f"Targets computed: {target.shape[0]} events | "
          f"Nulls: {target['target_return'].null_count()}")
    return target


# Event Window Construction
def assign_relative_trading_days(ohlcv_with_ta, earnings_df, window=(-10, 1)):
    """Map each earnings event to a window of trading days around it.
    """
    ohlcv   = ohlcv_with_ta.with_columns(pl.col("date").cast(pl.Date))
    earnings = earnings_df.with_columns(pl.col("reportedDate").cast(pl.Date))

    frames  = []
    symbols = ohlcv["symbol"].unique().sort().to_list()
    skipped = 0

    for i, symbol in enumerate(symbols):
        stock  = ohlcv.filter(pl.col("symbol") == symbol).sort("date")
        dates  = stock["date"].to_list()

        sym_earnings = earnings.filter(pl.col("symbol") == symbol)

        for row in sym_earnings.iter_rows(named=True):
            earnings_date = row["reportedDate"]
            t0 = bisect_left(dates, earnings_date)

            if t0 >= len(dates):
                skipped += 1
                continue

            lo = t0 + window[0]
            hi = t0 + window[1]

            if lo < 0 or hi >= len(dates):
                skipped += 1
                continue

            chunk = stock.slice(lo, hi - lo + 1).with_columns(
                pl.Series("relative_day", list(range(window[0], window[1] + 1))),
                pl.lit(earnings_date).alias("earnings_date"),
            )
            frames.append(chunk)

        if (i + 1) % 50 == 0 or (i + 1) == len(symbols):
            print(f"  Event windows: {i + 1}/{len(symbols)} symbols | {len(frames)} events")

    result = pl.concat(frames)
    print(f"\nTotal: {len(frames)} events | {skipped} skipped | shape: {result.shape}")
    return result


# Pivoting
def pivot_to_event_level(daily):
    """Pivot long-format daily to wide event-level table.
    """
    import re

    idx  = ["symbol", "earnings_date"]
    skip = {"symbol", "earnings_date", "relative_day", "date"}
    feature_cols = [c for c in daily.columns if c not in skip]

    daily_str = daily.with_columns(pl.col("relative_day").cast(pl.Utf8))

    parts = []
    for feat in feature_cols:
        part = (
            daily_str
            .select(idx + ["relative_day", feat])
            .pivot(on="relative_day", index=idx, values=feat)
        )
        rename = {}
        for c in part.columns:
            if c in idx:
                continue
            day = int(c)
            if day == 0:
                rename[c] = f"{feat}_t0"
            elif day > 0:
                rename[c] = f"{feat}_t+{day}"
            else:
                rename[c] = f"{feat}_t{day}"
        part = part.rename(rename)
        parts.append(part)

    result = parts[0]
    for p in parts[1:]:
        result = result.join(p, on=idx, how="full", coalesce=True)

    feat_columns = [c for c in result.columns if c not in idx]
    ordered = sorted(feat_columns, key=lambda c: (
        int(re.search(r"(-?\d+)$", c).group(1)), c
    ))
    result = result.select(idx + ordered)

    print(f"Pivoted: {result.shape[0]} events × {result.shape[1]} columns")
    return result


# Final Table Assembly
def assemble_modeling_table(pivoted, earnings_enriched, transcripts_enriched, daily, sector_info=None):
    """Join pivoted OHLCV+TA with earnings, transcript features, targets, and sector info.
    """
    # Earnings features
    earn_feat = earnings_enriched.with_columns(
        pl.col("reportedDate").cast(pl.Date).alias("earnings_date")
    )
    modeling = pivoted.join(earn_feat, on=["symbol", "earnings_date"], how="left")

    # Raw transcript text 
    tx_feat = transcripts_enriched.select(
        pl.col("symbol"),
        pl.col("reportedDate").cast(pl.Date).alias("earnings_date"),
        pl.col("transcript"),
    )
    modeling = modeling.join(tx_feat, on=["symbol", "earnings_date"], how="left")

    # Target variables : computed from daily before pivoting to avoid leakage
    targets = compute_target_from_daily(daily)
    modeling = modeling.join(targets, on=["symbol", "earnings_date"], how="left")

    # Sector and industry
    if sector_info is not None:
        modeling = modeling.join(sector_info, on="symbol", how="left")

    print(f"Final modeling table: {modeling.shape}")
    print(f"Symbols: {modeling['symbol'].n_unique()}")
    print(f"Total nulls: {modeling.null_count().sum_horizontal()[0]}")
    return modeling

feature_window = (-10, 1)    
data_window    = (-10, 10)   

# Batch Processing
def _process_batch(symbols, ohlcv_all, earnings_all, transcripts_all, batch_num, total_batches, window=data_window):
    """Process a batch of symbols through the full pipeline.
    """
    import gc

    print(f"\n- Batch {batch_num}/{total_batches} ({len(symbols)} symbols) -")

    ohlcv    = ohlcv_all.filter(pl.col("symbol").is_in(symbols))
    earnings = earnings_all.filter(pl.col("symbol").is_in(symbols))

    # TA indicators
    ta = compute_all_ta(ohlcv)

    # Derived features
    derived = compute_derived(ohlcv)

    # Combine OHLCV + TA + derived
    ohlcv_with_ta = (
        ohlcv
        .join(ta,      on=["symbol", "date"], how="left")
        .join(derived, on=["symbol", "date"], how="left")
    )
    del ta, derived, ohlcv
    gc.collect()

    # Event windows - feature window only (t-10 to t+1)
    daily = assign_relative_trading_days(ohlcv_with_ta, earnings,window=window)
    del ohlcv_with_ta
    gc.collect()

    # Pivot feature window to event level
    feature_daily = daily.filter(pl.col("relative_day") <= feature_window[1])  # Only t-10 to t+1 
    pivoted = pivot_to_event_level(feature_daily)

    return pivoted, daily


def run_pipeline(symbol=None, cache_dir="data", batch_size=50):
    """Run the full pipeline end-to-end."""
    import gc

    os.makedirs(cache_dir, exist_ok=True)
    suffix      = f"_{symbol}" if symbol else "_all"
    cache_path  = os.path.join(cache_dir, f"modeling{suffix}.parquet")
    daily_cache = os.path.join(cache_dir, f"daily{suffix}.parquet")

    # Check cache
    if os.path.exists(cache_path) and os.path.exists(daily_cache):
        print(f"Loading from cache: {cache_path}")
        modeling = pl.read_parquet(cache_path)
        daily    = pl.read_parquet(daily_cache)
        print(f"Modeling: {modeling.shape} | Daily: {daily.shape}")
        return modeling, daily

    print("-" * 60)
    print(f"Running pipeline for: {symbol or 'ALL STOCKS'}")
    print("-" * 60)

    # Load all data from S3
    print("\n- Loading data from S3 -")
    earnings    = load_earnings(symbol)
    transcripts = load_transcripts(symbol)
    ohlcv       = load_ohlcv(symbol)

    # EPS features
    print("\n- EPS features -")
    earnings = add_eps_features(earnings)
    

    # Sector and industry lookup 
    print("\n- Sector & industry info -")
    all_syms    = ohlcv["symbol"].unique().sort().to_list()
    sector_info = load_sector_info(all_syms, cache_dir=cache_dir)

    # Process TA + pivoting in batches
    all_symbols = ohlcv["symbol"].unique().sort().to_list()

    if symbol:
        batches = [all_symbols]
    else:
        batches = [all_symbols[i:i + batch_size] for i in range(0, len(all_symbols), batch_size)]

    print(f"\n- Processing {len(all_symbols)} symbols in {len(batches)} batches -")

    all_pivoted = []
    all_daily   = []

    for i, batch_symbols in enumerate(batches, 1):
        pivoted_batch, daily_batch = _process_batch(
            batch_symbols, ohlcv, earnings, transcripts, i, len(batches), window=data_window
        )
        all_pivoted.append(pivoted_batch)
        all_daily.append(daily_batch)
        del pivoted_batch, daily_batch
        gc.collect()

    # ohlcv no longer needed after batching
    del ohlcv
    gc.collect()

    # Concatenate all batches
    print("\n- Concatenating batches -")
    pivoted = pl.concat(all_pivoted)
    daily   = pl.concat(all_daily)
    del all_pivoted, all_daily
    gc.collect()

    print(f"Pivoted: {pivoted.shape} | Daily: {daily.shape}")

    # Assemble final table (targets computed inside from daily)
    print("\n- Assembling modeling table -")
    modeling = assemble_modeling_table(pivoted, earnings, transcripts, daily, sector_info)
    del pivoted
    gc.collect()

    # Save cache (drop raw transcript column)
    print(f"\n- Saving cache to {cache_dir}/ -")
    modeling_save = modeling.drop("transcript")
    modeling_save.write_parquet(cache_path)
    daily.write_parquet(daily_cache)
    print(f"Saved: {cache_path} ({os.path.getsize(cache_path) / 1e6:.1f} MB)")

    return modeling, daily