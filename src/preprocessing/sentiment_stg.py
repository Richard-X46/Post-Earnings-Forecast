import polars as pl

PATH_FINBERT = "src/data/finbert_tx/finbert_tx.parquet"
PATH_EARNINGS = "src/data/backup/earnings_delta_backup.parquet"
PATH_NEWS = "src/data/backup/news_earnings.parquet"
PATH_INFO = "src/data/metrics/snp500_info.parquet"
PATH_ML_STG = "src/data/model_staging/"


def _load_finbert_probs():
    df = pl.scan_parquet(PATH_FINBERT)
    return df.with_columns([
        pl.col("sentiment_probs").arr.first().alias("pos_prob"),
        pl.col("sentiment_probs").arr.last().alias("neg_prob"),
    ])


def _extract_ticker_sentiment(df):
    return (
        df.explode("ticker_sentiment")
        .with_columns(
            pl.col("ticker_sentiment").struct.field("ticker_sentiment_score").cast(pl.Float64).alias("ticker_sentiment_score"),
            pl.col("ticker_sentiment").struct.field("relevance_score").cast(pl.Float64).alias("relevance_score"),
            pl.col("ticker_sentiment").struct.field("ticker").alias("ticker"),
        )
        .filter(pl.col("ticker") == pl.col("symbol"))
        .drop("ticker", "ticker_sentiment")
    )


def _impute_cols_by_hierarchy(df, target_cols, sector_means, year_means, fallback):
    for col in target_cols:
        df = df.join(
            sector_means.select("sector", "year", pl.col(col).alias(f"{col}_fill")),
            on=["sector", "year"], how="left",
        ).with_columns(pl.col(col).fill_null(pl.col(f"{col}_fill"))).drop(f"{col}_fill")
        df = df.join(
            year_means.select("year", pl.col(col).alias(f"{col}_fill")),
            on="year", how="left",
        ).with_columns(pl.col(col).fill_null(pl.col(f"{col}_fill"))).drop(f"{col}_fill")
        df = df.with_columns(pl.col(col).fill_null(fallback))
    return df


def compute_transcript_sentiment():
    df_probs = _load_finbert_probs()

    df_mean = (
        df_probs.group_by(["symbol", "reportedDate"])
        .agg([
            pl.col("pos_prob").mean().alias("pos_prob_mean"),
            pl.col("neg_prob").mean().alias("neg_prob_mean"),
        ])
        .collect()
    )

    df_weighted = (
        df_probs.group_by(["symbol", "reportedDate"])
        .agg([
            (pl.col("pos_prob") * pl.col("text_length")).sum()
            / pl.col("text_length").sum()
            .alias("pos_prob_weighted_mean"),
            (pl.col("neg_prob") * pl.col("text_length")).sum()
            / pl.col("text_length").sum()
            .alias("neg_prob_weighted_mean"),
        ])
        .collect()
    )

    df_mean.write_parquet(PATH_ML_STG + "finbert_tx_agg_mean.parquet")

    return df_weighted


def build_transcript_features(df_weighted, df_earnings):
    df_info_tx = pl.read_parquet(PATH_INFO).select("symbol", "sector")

    df_tx = (
        df_earnings.select("symbol", "reportedDate")
        .join(df_weighted, on=["symbol", "reportedDate"], how="left")
        .join(df_info_tx, on="symbol", how="left")
        .with_columns(pl.col("reportedDate").dt.year().alias("year"))
    )

    tx_cols = ["pos_prob", "neg_prob"]

    tx_sector_means = df_tx.drop_nulls(tx_cols[0]).group_by("sector", "year").agg([
        pl.col(c).mean() for c in tx_cols
    ])
    tx_year_means = df_tx.drop_nulls(tx_cols[0]).group_by("year").agg([
        pl.col(c).mean() for c in tx_cols
    ])

    df_tx = _impute_cols_by_hierarchy(df_tx, tx_cols, tx_sector_means, tx_year_means, 0.5)
    df_tx = df_tx.drop("sector", "year")

    return df_tx


def build_news_features(df_earnings):
    df_news = pl.scan_parquet(PATH_NEWS).collect()

    keep_cols = [
        "symbol", "reportedDate", "news_date",
        "overall_sentiment_score", "overall_sentiment_label",
        "ticker_sentiment_score", "relevance_score",
    ]

    df_pre = _extract_ticker_sentiment(
        df_news.filter(pl.col("news_date") < pl.col("reportedDate"))
    ).select(keep_cols)
    df_post = _extract_ticker_sentiment(
        df_news.filter(pl.col("news_date") > pl.col("reportedDate"))
    ).select(keep_cols)

    news_agg = {
        "overall_sentiment_score": pl.col("overall_sentiment_score").mean(),
        "ticker_sentiment_score": (pl.col("ticker_sentiment_score") * pl.col("relevance_score")).sum() / pl.col("relevance_score").sum(),
        "relevance_score": pl.col("relevance_score").mean(),
        "news_count": pl.len(),
    }

    df_pre_agg = df_pre.group_by("symbol", "reportedDate").agg(**news_agg).with_columns(pl.lit("pre").alias("window"))
    df_post_agg = df_post.group_by("symbol", "reportedDate").agg(**news_agg).with_columns(pl.lit("post").alias("window"))

    for suffix, df_sent in [("pre", df_pre_agg), ("post", df_post_agg)]:
        df_earnings = df_earnings.join(
            df_sent.drop("window"),
            on=["symbol", "reportedDate"],
            how="left",
            suffix=f"_{suffix}",
        )

    pre_cols = ["overall_sentiment_score", "ticker_sentiment_score", "relevance_score", "news_count"]
    df_earnings = df_earnings.rename({c: f"{c}_pre" for c in pre_cols})

    df_info = pl.read_parquet(PATH_INFO).select("symbol", "sector", "industry")
    df_earnings = df_earnings.join(df_info, on="symbol", how="left")
    df_earnings = df_earnings.with_columns(pl.col("reportedDate").dt.year().alias("year"))

    df_sent_all = pl.concat([df_pre_agg, df_post_agg]).join(df_info, on="symbol", how="left")
    df_sent_all = df_sent_all.with_columns(pl.col("reportedDate").dt.year().alias("year"))

    sector_means = df_sent_all.group_by("sector", "year").agg([
        pl.col("overall_sentiment_score").mean(),
        pl.col("ticker_sentiment_score").mean(),
        pl.col("relevance_score").mean(),
    ])
    year_means = df_sent_all.group_by("year").agg([
        pl.col("overall_sentiment_score").mean(),
        pl.col("ticker_sentiment_score").mean(),
        pl.col("relevance_score").mean(),
    ])

    sent_cols = ["overall_sentiment_score", "ticker_sentiment_score", "relevance_score"]
    for col in sent_cols:
        for suffix in ["_pre", "_post"]:
            target = f"{col}{suffix}"
            df_earnings = df_earnings.join(
                sector_means.select("sector", "year", pl.col(col).alias(f"{col}_fill")),
                on=["sector", "year"], how="left",
            ).with_columns(pl.col(target).fill_null(pl.col(f"{col}_fill"))).drop(f"{col}_fill")
            df_earnings = df_earnings.join(
                year_means.select("year", pl.col(col).alias(f"{col}_fill")),
                on="year", how="left",
            ).with_columns(pl.col(target).fill_null(pl.col(f"{col}_fill"))).drop(f"{col}_fill")
            df_earnings = df_earnings.with_columns(pl.col(target).fill_null(0.0))

    for suffix in ["_pre", "_post"]:
        df_earnings = df_earnings.with_columns(pl.col(f"news_count{suffix}").fill_null(0))

    df_earnings = df_earnings.drop("year")
    return df_earnings


def main():
    df_weighted = compute_transcript_sentiment()
    df_earnings = pl.scan_parquet(PATH_EARNINGS).collect()

    df_tx = build_transcript_features(df_weighted, df_earnings)
    df_earnings = build_news_features(df_earnings)
    df_earnings = df_earnings.join(df_tx, on=["symbol", "reportedDate"], how="left")

    cols_to_write = [
        "reportedDate", "symbol",
        "overall_sentiment_score_pre", "ticker_sentiment_score_pre", "relevance_score_pre",
        "overall_sentiment_score_post", "ticker_sentiment_score_post", "relevance_score_post",
        "pos_prob", "neg_prob",
        "sector", "industry",
    ]

    df_out = df_earnings.select(cols_to_write).sort(["symbol", "reportedDate"])
    df_out.write_parquet(PATH_ML_STG + "nz_sentiment.parquet")


if __name__ == "__main__":
    main()
