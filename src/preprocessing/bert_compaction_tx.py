import polars as pl

PATH_SRC = "src/data/backup/finbert_tx"
PATH_OUT = "src/data/raw/finbert_tx"


def main():
    df = (
        pl.scan_parquet(PATH_SRC)
        .with_columns(
            pl.col("text").str.len_chars().fill_null(0).alias("text_length")
        )
        .drop("text")
        .collect()
    )

    df = df.drop("speaker")
    df.write_parquet(PATH_OUT + "/finbert_tx.parquet")


if __name__ == "__main__":
    main()
