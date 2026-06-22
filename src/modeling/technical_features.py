# import libraries
import polars as pl
import talib
import numpy as np
import pandas as pd


# indicator computation
def compute_technical_indicators(sdf: pl.DataFrame) -> pl.DataFrame:
    
    """
    Compute technical indicators from the base paper
    """
    sdf = sdf.sort("date")

    close = sdf["close"].to_numpy().astype(np.float64)
    high = sdf["high"].to_numpy().astype(np.float64)
    low = sdf["low"].to_numpy().astype(np.float64)
    volume = sdf["volume"].to_numpy().astype(np.float64)

    # Momentum indicators
    rsi = talib.RSI(close, timeperiod=14)
    macd, _, macd_hist = talib.MACD(close)
    roc = talib.ROC(close, timeperiod=10)
    
    # Price level 
    ema_50 = talib.EMA(close, timeperiod=50)
    ema_200 = talib.EMA(close, timeperiod=200)
    adx = talib.ADX(high, low, close, timeperiod=14)

    # Volatility indicators
    atr = talib.ATR(high, low, close, timeperiod=14)
        
    bb_upper, bb_middle, bb_lower = talib.BBANDS(close, timeperiod=14, nbdevup=2, nbdevdn=2)
    ret = np.empty_like(close); ret[0] = np.nan
    ret[1:] = (close[1:] - close[:-1]) / close[:-1]
    sigma = talib.STDDEV(ret, timeperiod=14, nbdev=1)

    # Volume indicators
    obv = talib.OBV(close, volume)

    # Volume Weighted Average Price (VWAP)
    vwap = (pd.Series(close * volume).rolling(20).sum() / 
           pd.Series(volume).rolling(20).sum()).values
    
    return sdf.with_columns(
            pl.Series("rsi", rsi),
            pl.Series("macd", macd),
            pl.Series("macd_hist", macd_hist),
            pl.Series("roc", roc),
            pl.Series("ema_50", ema_50),
            pl.Series("ema_200", ema_200),
            pl.Series("adx", adx),
            pl.Series("atr", atr),
            pl.Series("bb_upper", bb_upper),
            pl.Series("bb_middle", bb_middle),
            pl.Series("bb_lower", bb_lower),
            pl.Series("sigma", sigma),
            pl.Series("obv", obv),
            pl.Series("vwap", vwap)
        )

# Normalize the indators to be relative to the close price

def normalize_indicators(sdf: pl.DataFrame) -> pl.DataFrame:
    """Normalize indicators to be relative to the close price."""
    close = pl.col("close")

    return sdf.with_columns(
        ((pl.col("open") - close) / close).alias("open_pct"),
        ((pl.col("high") - close) / close).alias("high_pct"),
        ((pl.col("low") - close) / close).alias("low_pct"),
        ((pl.col("volume") - pl.col("volume").rolling_mean(window_size=20)) / pl.col("volume").rolling_mean(window_size=20)).alias("volume_rel"),

        (pl.col("macd") / close).alias("macd"),
        (pl.col("macd_hist") / close).alias("macd_hist"),
        (pl.col("atr") / close).alias("atr"),

        # distance from reference -> close
        ((close - pl.col("ema_50")) / pl.col("ema_50")).alias("ema50_pct"),
        ((close - pl.col("ema_200")) / pl.col("ema_200")).alias("ema200_pct"),
        ((pl.col("ema_50") - pl.col("ema_200")) / pl.col("ema_200")).alias("ema50_200_pct"),
        ((pl.col("bb_upper") - pl.col("bb_lower")) / pl.col("bb_middle")).alias("bb_width"),
        ((close - pl.col("bb_lower")) / (pl.col("bb_upper") - pl.col("bb_lower"))).alias("bb_pct_b"),
        ((close - pl.col("vwap")) / pl.col("vwap")).alias("vwap_pct"),

        # z-score
        ((pl.col("obv") - pl.col("obv").rolling_mean(252)) / pl.col("obv").rolling_std(252)).alias("obv_zscore")
    ).drop(["open", "low", "volume",
            "ema_50", "ema_200", "bb_upper", "bb_middle", "bb_lower", "obv", "vwap"])


tech_feature_cols = [
    "open_pct", "high_pct", "low_pct", "volume_rel",
    "rsi", "macd", "macd_hist", "roc",
    "ema50_pct", "ema200_pct", "ema50_200_pct", "adx",
    "atr", "bb_width", "bb_pct_b", "sigma",
    "obv_zscore", "vwap_pct",]


def build_technical_features(df: pl.DataFrame) -> pl.DataFrame:
    """Compute indicators, then apply close-relative normalization."""
    frames = []
    for _,sdf in df.group_by("symbol", maintain_order=True):
        raw = compute_technical_indicators(sdf)
        norm = normalize_indicators(raw)
        frames.append(norm)
    return pl.concat(frames)


def build_modeling_table(df_daily, df_earnings, feature_cols = None, earnings_date_column = "reportedDate"):
    """One row per earnings event, pivoted features (t-10 to t+1) + target."""
    from bisect import bisect_left
 
    if feature_cols is None:
        feature_cols = tech_feature_cols
 
    events = []
    symbols = df_earnings["symbol"].unique().to_list()
 
    for symbol in symbols:
        sym_daily = df_daily.filter(pl.col("symbol") == symbol).sort("date")
        sym_earnings = df_earnings.filter(pl.col("symbol") == symbol)
 
        if sym_daily.is_empty():
            continue
 
        dates = sym_daily["date"].to_list()
        closes = sym_daily["close"].to_numpy()
        highs = sym_daily["high"].to_numpy()
        feat_data = {col: sym_daily[col].to_numpy() for col in feature_cols}
 
        for row in sym_earnings.iter_rows(named=True):
            earn_date = row[earnings_date_column]
 
            t0 = bisect_left(dates, earn_date)
            if t0 >= len(dates) or dates[t0] != earn_date:
                continue
            if t0 - 10 < 0 or t0 + 10 >= len(dates):
                continue

            entry_price = closes[t0 + 1]  # close price on the day after earnings
            
            # high prices during drift window (t+2 to t+10)
            drift_highs = highs[t0 + 2 : t0 + 11]

            max_high = float(drift_highs.max())
            min_high = float(drift_highs.min())
            max_day = int(drift_highs.argmax() + 2)
            min_day = int(drift_highs.argmin() + 2)

            # original target return calculation
            target_return = float(max_high / entry_price - 1)
            
                      
            event = {
                "symbol": symbol,
                "earnings_date": earn_date,
                "entry_price": float(entry_price),
                # original targets
                "target_return": target_return,
                "target_class": 1 if target_return >= 0.03 else 0,
                # high-based targets
                "max_high": max_high,
                "min_high": min_high,
                "max_day": max_day,
                "min_day": min_day,
            }
 
            for t in range(-10, 2):
                suffix = f"_t{t}" if t <= 0 else f"_t+{t}"
                for col in feature_cols:
                    event[f"{col}{suffix}"] = float(feat_data[col][t0 + t])
 
            events.append(event)
 
    return pl.DataFrame(events)
 
 
if __name__ == "__main__":
    # Load
    OHLCV_PATH = "src/ingestion/data/backup/ohlcv_delta_backup.parquet"
    EARNINGS_PATH = "src/ingestion/data/backup/earnings_delta_backup.parquet"
 
    df_ohlcv = pl.read_parquet(OHLCV_PATH)
    df_earnings = pl.read_parquet(EARNINGS_PATH)
    print(f"OHLCV: {df_ohlcv.shape}")
    print(f"Earnings: {df_earnings.shape}")
    print(f"Earnings columns: {df_earnings.columns}")
 
    # Build daily features
    df_daily = build_technical_features(df_ohlcv)
    print(f"\nDaily table: {df_daily.shape}")
    print(f"Columns: {df_daily.columns}")
    print(df_daily.head(5))
 
    # Build modeling table
    # check earnings date column name from print above, adjust if needed
    df_model = build_modeling_table(df_daily, df_earnings, earnings_date_column="reportedDate")
    print(f"\nModeling table: {df_model.shape}")
    print(f"target_return: mean={df_model['target_return'].mean():.4f}, std={df_model['target_return'].std():.4f}")
    print(f"max_high: mean={df_model['max_high'].mean():.2f}")
    print(f"min_high: mean={df_model['min_high'].mean():.2f}")
    print(f"max_day distribution:\n{df_model['max_day'].value_counts().sort('max_day')}")
    print(df_model.head(5))
 
    # Save for reuse by b1, b2, b3 pipelines
    import os
    os.makedirs("src/modeling/data", exist_ok=True)
    df_model.write_parquet("src/modeling/data/tech_modeling_table.parquet")
    print(f"\nSaved to src/modeling/data/tech_modeling_table.parquet")




