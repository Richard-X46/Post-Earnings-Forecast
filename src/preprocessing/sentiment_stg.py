# Back up frame loaders polarities
import polars as pl




# ----// Query on TX aggregate to mean


path = "src/data/finbert_tx/finbert_tx.parquet"
df = pl.scan_parquet(path)

df_probs = df.with_columns([
    pl.col("sentiment_probs").arr.first().alias("pos_prob"),
    pl.col("sentiment_probs").arr.last().alias("neg_prob"),
])
    

df_agg_mean = (
    df_probs.group_by(["symbol", "reportedDate"])
    .agg([
        pl.col("pos_prob").mean().alias("pos_prob_mean"),
        pl.col("neg_prob").mean().alias("neg_prob_mean"),
    ])
    .collect()
)


df_agg_weighted = (
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


# writing to model staging
path_ml_stg = "src/data/model_staging/"

df_agg_mean.write_parquet(path_ml_stg + "finbert_tx_agg_mean.parquet")

df_agg_mean.schema

df_earnings = pl.scan_parquet("src/data/backup/earnings_delta_backup.parquet").collect()
df_earnings.shape

# --- impute missing transcript sentiment with sector/year means ---
df_info_tx = pl.read_parquet("src/data/metrics/snp500_info.parquet").select("symbol", "sector")
df_tx = df_earnings.select("symbol", "reportedDate").join(
    df_agg_weighted, on=["symbol", "reportedDate"], how="left"
).join(df_info_tx, on="symbol", how="left").with_columns(
    pl.col("reportedDate").dt.year().alias("year")
)

tx_cols = ["pos_prob", "neg_prob"]
tx_sector_means = df_tx.drop_nulls(tx_cols[0]).group_by("sector", "year").agg([
    pl.col(c).mean() for c in tx_cols
])
tx_year_means = df_tx.drop_nulls(tx_cols[0]).group_by("year").agg([
    pl.col(c).mean() for c in tx_cols
])

for col in tx_cols:
    df_tx = df_tx.join(
        tx_sector_means.select("sector", "year", pl.col(col).alias(f"{col}_fill")),
        on=["sector", "year"], how="left",
    ).with_columns(pl.col(col).fill_null(pl.col(f"{col}_fill"))).drop(f"{col}_fill")
    df_tx = df_tx.join(
        tx_year_means.select("year", pl.col(col).alias(f"{col}_fill")),
        on="year", how="left",
    ).with_columns(pl.col(col).fill_null(pl.col(f"{col}_fill"))).drop(f"{col}_fill")
    df_tx = df_tx.with_columns(pl.col(col).fill_null(0.5))

df_tx = df_tx.drop("sector", "year")

# checking nulls after imputation
for col in tx_cols:
    nulls = df_tx[col].null_count()
    pct = nulls / df_tx.height
    print(f"{col}: {nulls} nulls ({pct:.1%})")

# writing to model staging
df_tx.write_parquet(path_ml_stg + "finbert_tx_agg_weighted.parquet")











# --- // news sentiment around earnings events

file_path = "src/data/backup/news_earnings.parquet"

df_news_earnings = pl.scan_parquet(file_path).collect()

df_news_earnings.shape
df_news_earnings.schema


def extract_ticker_sentiment(df):
    return df.explode("ticker_sentiment").with_columns(
        pl.col("ticker_sentiment").struct.field("ticker_sentiment_score").cast(pl.Float64).alias("ticker_sentiment_score"),
        pl.col("ticker_sentiment").struct.field("relevance_score").cast(pl.Float64).alias("relevance_score"),
        pl.col("ticker_sentiment").struct.field("ticker").alias("ticker"),
    ).filter(pl.col("ticker") == pl.col("symbol")).drop("ticker", "ticker_sentiment")

df_pre = df_news_earnings.filter(pl.col("news_date") < pl.col("reportedDate"))
df_post = df_news_earnings.filter(pl.col("news_date") > pl.col("reportedDate"))

keep_cols = ["symbol", "reportedDate",'news_date', "overall_sentiment_score", "overall_sentiment_label", "ticker_sentiment_score", "relevance_score"]

df_pre = extract_ticker_sentiment(df_pre).select(keep_cols)
df_post = extract_ticker_sentiment(df_post).select(keep_cols)

agg_cols = {
    "overall_sentiment_score": pl.col("overall_sentiment_score").mean(),
    "ticker_sentiment_score": (pl.col("ticker_sentiment_score") * pl.col("relevance_score")).sum() / pl.col("relevance_score").sum(),
    "relevance_score": pl.col("relevance_score").mean(),
    "news_count": pl.len(),
}

df_pre_agg = df_pre.group_by("symbol", "reportedDate").agg(**agg_cols).with_columns(pl.lit("pre").alias("window"))
df_post_agg = df_post.group_by("symbol", "reportedDate").agg(**agg_cols).with_columns(pl.lit("post").alias("window"))

# Join back to earnings
df_earnings = pl.scan_parquet("src/data/backup/earnings_delta_backup.parquet").collect()





for suffix, df_sent in [("pre", df_pre_agg), ("post", df_post_agg)]:
    df_earnings = df_earnings.join(
        df_sent.drop("window"),
        on=["symbol", "reportedDate"],
        how="left",
        suffix=f"_{suffix}",
    )

# Rename unprefixed pre columns to be explicit
pre_cols = ["overall_sentiment_score", "ticker_sentiment_score", "relevance_score", "news_count"]
df_earnings = df_earnings.rename({c: f"{c}_pre" for c in pre_cols})



# --- sector/year imputation for null sentiment ---
df_info = pl.read_parquet("src/data/metrics/snp500_info.parquet").select("symbol", "sector", "industry")
df_earnings = df_earnings.join(df_info, on="symbol", how="left")
df_earnings = df_earnings.with_columns(pl.col("reportedDate").dt.year().alias("year"))

# Build sector-year lookup from observed aggregates
df_sent_all = pl.concat([df_pre_agg, df_post_agg]).join(df_info, on="symbol", how="left")

sector_means = df_sent_all.with_columns(
    pl.col("reportedDate").dt.year().alias("year")
).group_by("sector", "year").agg([
    pl.col("overall_sentiment_score").mean(),
    pl.col("ticker_sentiment_score").mean(),
    pl.col("relevance_score").mean(),
])

year_means = df_sent_all.with_columns(
    pl.col("reportedDate").dt.year().alias("year")
).group_by("year").agg([
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
            on=["sector", "year"],
            how="left",
        ).with_columns(
            pl.col(target).fill_null(pl.col(f"{col}_fill"))
        ).drop(f"{col}_fill")

        df_earnings = df_earnings.join(
            year_means.select("year", pl.col(col).alias(f"{col}_fill")),
            on="year",
            how="left",
        ).with_columns(
            pl.col(target).fill_null(pl.col(f"{col}_fill"))
        ).drop(f"{col}_fill")

        df_earnings = df_earnings.with_columns(
            pl.col(target).fill_null(0.0)
        )

for suffix in ["_pre", "_post"]:
    df_earnings = df_earnings.with_columns(
        pl.col(f"news_count{suffix}").fill_null(0)
    )

if "year" in df_earnings.columns:
    df_earnings = df_earnings.drop("year")

for col in ["overall_sentiment_score", "ticker_sentiment_score", "relevance_score", "news_count"]:
    for suffix in ["_pre", "_post"]:
        c = f"{col}{suffix}"
        nulls = df_earnings[c].null_count()
        pct = nulls / df_earnings.height
        print(f"  {c}: {nulls} nulls ({pct:.1%})")



df_earnings = df_earnings.join(df_tx, on=["symbol", "reportedDate"], how="left")

cols_to_write = ["reportedDate","symbol",
                 "overall_sentiment_score_pre", "ticker_sentiment_score_pre", "relevance_score_pre",  
                 "overall_sentiment_score_post", "ticker_sentiment_score_post", "relevance_score_post",
                 "pos_prob", "neg_prob",
                 "sector", "industry"]

df_earnings = df_earnings.select(cols_to_write).sort(["symbol", "reportedDate"])
df_earnings.write_parquet("src/data/model_staging/nz_sentiment.parquet")