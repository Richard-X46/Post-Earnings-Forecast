"""
extract_upgrades.py
Downloads analyst upgrade/downgrade history for all S&P 500 symbols via yfinance.
Writes raw actions to data/upgrades_downgrades.parquet (one row per analyst action).
Windowing/feature engineering happens downstream (build_rating_features).

Run from terminal:
    python extract_upgrades.py            # skips if parquet exists
    python extract_upgrades.py --force    # re-download and overwrite

Verified on 2026-06-12 run: 162,953 rows, 502/503 symbols (ERIE absent from Yahoo),
Action values: main / up / down / init / reit.
"""

import sys
import time
from pathlib import Path

import polars as pl
import yfinance as yf

BASE = Path(__file__).parent
OUT_PATH = BASE / "data" / "upgrades_downgrades.parquet"
SYMBOLS_CSV = BASE / "data" / "snp500_2026-05-23.csv"


def fetch_symbol(sym: str) -> pl.DataFrame | None:
    """Fetch all rating actions for one symbol (Yahoo notation, e.g. BRK-B).
    Stores the symbol in dataset notation (BRK.B) so downstream joins match."""
    try:
        ud = yf.Ticker(sym).upgrades_downgrades   # property — not get_...()
        if ud is None or ud.empty:
            return None
        df = pl.from_pandas(ud.reset_index())
        return df.with_columns(
            pl.lit(sym.replace("-", ".")).alias("symbol"),   # back to dataset notation
            pl.col("GradeDate").dt.date().alias("grade_date"),
        )
    except Exception as e:
        print(f"  {sym}: FAILED — {e}")
        return None


def main(force: bool = False):
    if OUT_PATH.exists() and not force:
        print(f"{OUT_PATH} already exists. Use --force to re-download.")
        return

    # CSV column is "Symbol"; map dot notation (BRK.B) to Yahoo's dashes (BRK-B)
    symbols = [s.replace(".", "-") for s in pl.read_csv(SYMBOLS_CSV)["Symbol"].to_list()]

    frames, empty = [], []
    for i, sym in enumerate(symbols, 1):
        df = fetch_symbol(sym)
        if df is None:
            empty.append(sym)
        else:
            frames.append(df)
        # Systematic-failure guard: don't burn 10 minutes on a broken run
        if i == 5 and len(frames) == 0:
            raise RuntimeError("First 5 symbols all failed — systematic error, aborting.")
        if i % 50 == 0:
            print(f"  ...{i}/{len(symbols)} symbols")
        time.sleep(0.3)

    out = pl.concat(frames)

    # --- Verification log: every line is a downstream decision input ---
    print(f"Rows:            {out.height}")
    print(f"Symbols covered: {out['symbol'].n_unique()} / {len(symbols)}")
    print(f"No data for:     {len(empty)} symbols {empty[:10]}")
    print(f"Date range:      {out['grade_date'].min()} -> {out['grade_date'].max()}")
    print(f"Actions:         {out['Action'].value_counts().sort('count', descending=True).to_dicts()}")

    # Per-symbol coverage (masked-zeros issue -> rating_data_available flag downstream)
    coverage = out.group_by("symbol").agg(pl.col("grade_date").min().alias("first_action"))
    late = coverage.filter(pl.col("first_action") > pl.date(2014, 6, 1))
    print(f"Symbols with history starting after mid-2014: {late.height}")
    print(late.sort("first_action", descending=True).head(10))

    OUT_PATH.parent.mkdir(exist_ok=True)
    out.write_parquet(OUT_PATH)
    print(f"Written to {OUT_PATH}")


if __name__ == "__main__":
    main("--force" in sys.argv)