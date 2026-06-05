"""
PEAD Pipeline - Data loading, feature engineering, and event-window construction.
"""

import os
import numpy as np
import polars as pl
import talib
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

# Technical Indicator Functions
def compute_all_ta(ohlcv):
    """Compute RSI, MACD, Bollinger Bands, ATR, OBV in one loop.

    Each symbol: 1 filter + 1 sort + 5 TA-Lib calls on numpy arrays.
    For 503 symbols this is 503 iterations (vs 3,018 with separate functions).
    """
    results = []
    symbols = ohlcv["symbol"].unique().sort().to_list()

    for i, symbol in enumerate(symbols):
        stock = ohlcv.filter(pl.col("symbol") == symbol).sort("date")

        close = stock["close"].to_numpy()
        high = stock["high"].to_numpy()
        low = stock["low"].to_numpy()
        volume = stock["volume"].to_numpy().astype(float)

        rsi = talib.RSI(close, timeperiod=14)
        macd, signal, hist = talib.MACD(close, 12, 26, 9)
        bb_upper, bb_mid, bb_lower = talib.BBANDS(close, timeperiod=20)
        atr = talib.ATR(high, low, close, timeperiod=14)
        obv = talib.OBV(close, volume)

        results.append(
            stock.select("symbol", "date").with_columns([
                pl.Series("rsi", rsi),
                pl.Series("macd", macd),
                pl.Series("macdsignal", signal),
                pl.Series("macdhist", hist),
                pl.Series("bb_upper", bb_upper),
                pl.Series("bb_middle", bb_mid),
                pl.Series("bb_lower", bb_lower),
                pl.Series("atr", atr),
                pl.Series("obv", obv),
            ])
        )

        if (i + 1) % 50 == 0 or (i + 1) == len(symbols):
            print(f"  TA indicators: {i + 1}/{len(symbols)} symbols")

    return pl.concat(results)


# derived features
def compute_derived(ohlcv):
    """Compute daily_return, intraday_range, gap, volume_change.

    Uses .over('symbol') so Polars handles per-stock grouping internally.
    Processes all 503 symbols in one vectorized pass.
    """
    df = ohlcv.sort(["symbol", "date"])

    return df.select([
        "symbol", "date",
        pl.col("close").pct_change().over("symbol").alias("daily_return"),
        ((pl.col("high") - pl.col("low")) / pl.col("close")).alias("intraday_range"),
        (
            (pl.col("open") - pl.col("close").shift(1))
            / pl.col("close").shift(1).abs()
        ).over("symbol").alias("gap"),
        pl.col("volume").cast(pl.Float64).pct_change().over("symbol").alias("volume_change"),
    ])


# EPS Features
def add_eps_features(earnings):
    """Add eps_growth, surprise_trend, beat, consecutive_beats.

    .shift(1).over('symbol') ensures no future leakage.
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


# Transcript Features
def add_transcript_features(transcripts):
    """Compute word/char counts from nested transcript struct.

    IMPORTANT: Does NOT materialize the full transcript text as a column.
    Flattening 25,000 transcripts into strings uses 16+ GB of RAM.
    Full text extraction should be done per-stock during FinBERT (Milestone 4).
    """
    word_counts = []
    char_counts = []

    for row in transcripts.iter_rows(named=True):
        transcript = row["transcript"]
        if transcript is None:
            word_counts.append(0)
            char_counts.append(0)
            continue

        total_words = 0
        total_chars = 0
        for item in transcript:
            text = item["content"] if isinstance(item, dict) and "content" in item else str(item)
            total_words += len(text.split())
            total_chars += len(text)

        word_counts.append(total_words)
        char_counts.append(total_chars)

    transcripts = transcripts.with_columns([
        pl.Series("transcript_word_count", word_counts),
        pl.Series("transcript_char_count", char_counts),
    ])
    print(f"Transcript features added for {transcripts['symbol'].n_unique()} symbols")
    return transcripts


# Event Window Construction
def assign_relative_trading_days(ohlcv_with_ta, earnings_df, window=(-10, 1)):
    """Map each earnings event to a window of trading days around it.

    Uses bisect_left for O(log n) date lookup per event.
    """
    ohlcv = ohlcv_with_ta.with_columns(pl.col("date").cast(pl.Date))
    earnings = earnings_df.with_columns(pl.col("reportedDate").cast(pl.Date))

    frames = []
    symbols = ohlcv["symbol"].unique().sort().to_list()
    skipped = 0

    for i, symbol in enumerate(symbols):
        stock = ohlcv.filter(pl.col("symbol") == symbol).sort("date")
        dates = stock["date"].to_list()

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
    """Pivot long-format daily → wide event-level table.

    Each feature gets one column per relative day:
    close_t-10, close_t-9, ..., close_t0, close_t+1
    """
    import re

    idx = ["symbol", "earnings_date"]
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


# final table assembly
def assemble_modeling_table(pivoted, earnings_enriched, transcripts_enriched):
    """Join pivoted OHLCV+TA with earnings and transcript features."""
    earn_feat = earnings_enriched.with_columns(
        pl.col("reportedDate").cast(pl.Date).alias("earnings_date")
    )
    modeling = pivoted.join(earn_feat, on=["symbol", "earnings_date"], how="left")

    tx_feat = transcripts_enriched.select(
        pl.col("symbol"),
        pl.col("reportedDate").cast(pl.Date).alias("earnings_date"),
        pl.col("transcript"),
        pl.col("transcript_word_count"),
        pl.col("transcript_char_count"),
    )
    modeling = modeling.join(tx_feat, on=["symbol", "earnings_date"], how="left")

    print(f"Final modeling table: {modeling.shape}")
    print(f"Symbols: {modeling['symbol'].n_unique()}")
    print(f"Total nulls: {modeling.null_count().sum_horizontal()[0]}")
    return modeling


# Batch processing function
def _process_batch(symbols, ohlcv_all, earnings_all, transcripts_all, batch_num, total_batches):
    """Process a batch of symbols through the full pipeline.
    
    Returns (pivoted, daily) for just this batch.
    OHLCV is filtered from the pre-loaded full table to avoid re-reading S3.
    """
    import gc

    print(f"\n- Batch {batch_num}/{total_batches} ({len(symbols)} symbols) -")

    # Filter to this batch
    ohlcv = ohlcv_all.filter(pl.col("symbol").is_in(symbols))
    earnings = earnings_all.filter(pl.col("symbol").is_in(symbols))

    # TA indicators
    ta = compute_all_ta(ohlcv)

    # Derived features
    derived = compute_derived(ohlcv)

    # Combine
    ohlcv_with_ta = (
        ohlcv
        .join(ta, on=["symbol", "date"], how="left")
        .join(derived, on=["symbol", "date"], how="left")
    )
    del ta, derived, ohlcv
    gc.collect()

    # Event windows
    daily = assign_relative_trading_days(ohlcv_with_ta, earnings)
    del ohlcv_with_ta
    gc.collect()

    # Pivot to event level
    pivoted = pivot_to_event_level(daily)

    return pivoted, daily


def run_pipeline(symbol=None, cache_dir="data", batch_size=50):
    """Run the full pipeline end-to-end.
    """
    import gc

    os.makedirs(cache_dir, exist_ok=True)
    suffix = f"_{symbol}" if symbol else "_all"
    cache_path = os.path.join(cache_dir, f"modeling{suffix}.parquet")
    daily_cache = os.path.join(cache_dir, f"daily{suffix}.parquet")

    # Check cache first
    if os.path.exists(cache_path) and os.path.exists(daily_cache):
        print(f"Loading from cache: {cache_path}")
        modeling = pl.read_parquet(cache_path)
        daily = pl.read_parquet(daily_cache)
        print(f"Modeling: {modeling.shape} | Daily: {daily.shape}")
        return modeling, daily

    print("-" * 60)
    print(f"Running pipeline for: {symbol or 'ALL STOCKS'}")
    print("-" * 60)

    # Load all data from S3
    print("\n- Loading data from S3 -")
    earnings = load_earnings(symbol)
    transcripts = load_transcripts(symbol)
    ohlcv = load_ohlcv(symbol)

    # EPS + transcript features
    print("\n- EPS & transcript features -")
    earnings = add_eps_features(earnings)
    transcripts = add_transcript_features(transcripts)

    # Process TA + pivoting in batches
    all_symbols = ohlcv["symbol"].unique().sort().to_list()

    if symbol:
        # Single stock - no batching needed
        batches = [all_symbols]
    else:
        # Split into batches
        batches = [all_symbols[i:i + batch_size] for i in range(0, len(all_symbols), batch_size)]

    print(f"\n - Processing {len(all_symbols)} symbols in {len(batches)} batches -")

    all_pivoted = []
    all_daily = []

    for i, batch_symbols in enumerate(batches, 1):
        pivoted_batch, daily_batch = _process_batch(
            batch_symbols, ohlcv, earnings, transcripts, i, len(batches)
        )
        all_pivoted.append(pivoted_batch)
        all_daily.append(daily_batch)
        del pivoted_batch, daily_batch
        gc.collect()

    # Free the large OHLCV table now that batching is done
    del ohlcv
    gc.collect()

    # Concatenate all batches
    print("\n- Concatenating batches -")
    pivoted = pl.concat(all_pivoted)
    daily = pl.concat(all_daily)
    del all_pivoted, all_daily
    gc.collect()

    print(f"Pivoted: {pivoted.shape} | Daily: {daily.shape}")

    # Assemble final table
    print("\n- Assembling modeling table -")
    modeling = assemble_modeling_table(pivoted, earnings, transcripts)
    del pivoted
    gc.collect()

    print(f"\n- Saving cache to {cache_dir}/ -")
    modeling_save = modeling.drop("transcript")
    modeling_save.write_parquet(cache_path)
    daily.write_parquet(daily_cache)
    print(f"Saved: {cache_path} ({os.path.getsize(cache_path) / 1e6:.1f} MB)")

    return modeling, daily
