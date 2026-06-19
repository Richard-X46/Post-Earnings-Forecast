import polars as pl

path = "/Users/richardpears/Downloads/finbert_tx"

df = pl.scan_parquet(path)

df = df.with_columns(
    pl.col("text").str.len_chars().fill_null(0).alias("text_length")
)

df.filter(pl.col("text_length") < 1).collect()

df.select(pl.col("symbol"), pl.col("reportedDate")).unique().collect()

df = df.drop("text")

df = df.collect()

df.estimated_size() / 1e6

df = df.drop("speaker")

path_out = "src/ingestion/data/finbert_tx"

df.write_parquet(path_out + "/finbert_tx.parquet")
