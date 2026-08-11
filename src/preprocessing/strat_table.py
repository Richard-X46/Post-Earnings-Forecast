"""Build the forward-looking table used for trade strategies."""

from datetime import date
from pathlib import Path

import polars as pl


DRIFT_OFFSETS = range(2, 11)
OUTPUT_COLUMNS = (
    ["symbol", "date"]
    + [f"high_t+{offset}" for offset in DRIFT_OFFSETS]
    + [f"low_t+{offset}" for offset in DRIFT_OFFSETS]
    + [f"date_t+{offset}" for offset in DRIFT_OFFSETS]
    + ["entry_date", "close_t10"]
)


def build_strat_table(
    df_ohlcv: pl.DataFrame,
    df_earnings: pl.DataFrame,
    earnings_date_column: str = "reportedDate",
    cutoff_date: str = "2026-01-01",
) -> pl.DataFrame:
    """Create one forward-price row for each eligible earnings event."""
    events = (
        df_earnings
        .select(
            [
                "symbol",
                pl.col(earnings_date_column).cast(pl.Date).alias("date"),
            ]
        )
        .filter(pl.col("date") < date.fromisoformat(cutoff_date))
        .unique()
    )

    table = (
        df_ohlcv
        .select(["symbol", "date", "high", "low", "close"])
        .join(events.select("symbol").unique(), on="symbol", how="semi")
        .sort(["symbol", "date"])
    )

    forward_columns = [
        expression
        for offset in DRIFT_OFFSETS
        for expression in (
            pl.col("high").shift(-offset).over("symbol").alias(f"high_t+{offset}"),
            pl.col("low").shift(-offset).over("symbol").alias(f"low_t+{offset}"),
            pl.col("date").shift(-offset).over("symbol").alias(f"date_t+{offset}"),
        )
    ]
    table = table.with_columns(
        forward_columns
        + [
            pl.col("date").shift(-1).over("symbol").alias("entry_date"),
            pl.col("close").shift(-10).over("symbol").alias("close_t10"),
        ]
    )

    return (
        table
        .join(events, on=["symbol", "date"], how="semi")
        .select(OUTPUT_COLUMNS)
    )


def write_strat_table(
    ohlcv_path: str | Path,
    earnings_path: str | Path,
    output_path: str | Path = "src/data/model_staging/strat_table.parquet",
) -> pl.DataFrame:
    """Read source data, build the strategy table, and write the parquet file."""
    table = build_strat_table(
        pl.read_parquet(ohlcv_path),
        pl.read_parquet(earnings_path),
    )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    table.write_parquet(output_path)
    return table


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[2]
    table = write_strat_table(
        root / "src/data/backup/ohlcv_delta_backup.parquet",
        root / "src/ingestion/data/backup/earnings_delta_backup.parquet",
        root / "src/data/model_staging/strat_table.parquet",
    )
    print(f"Strategy table: {table.shape}")

