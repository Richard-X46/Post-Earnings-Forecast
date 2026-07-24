

import polars as pl
from datetime import date

path = "src/ingestion/data/backup/earnings_delta_backup.parquet"
df_er = pl.read_parquet(path)


# loading news data from compacted file 
path_nz = 'src/ingestion/data/backup/compacted_news.parquet'
df_nz = pl.read_parquet(path_nz)


df_nz['symbol'].unique()
df_er['symbol'].unique()

symbols_er = set(df_er['symbol'].unique())
symbols_nz = set(df_nz['symbol'].unique())
missing_in_nz = symbols_er - symbols_nz
print(f"Symbols in df_er but not in df_nz: {missing_in_nz}")

df_er = df_er.sort("reportedDate")
df_nz = df_nz.sort("time_published")


df_er.schema, df_er.shape

df_nz.schema,df_nz.shape

df_er = df_er.with_columns(pl.col("reportedDate").cast(pl.Date))
df_nz = df_nz.with_columns(pl.col("time_published").cast(pl.Date))


daily_news = (
    df_nz
    .group_by(["symbol", "time_published"])
    .agg(pl.len().alias("news_count"))
)

grid = (
    df_er.select(["symbol", "reportedDate"])
    .with_columns(
        pl.date_ranges(
            pl.col("reportedDate") - pl.duration(days=10),
            pl.col("reportedDate") + pl.duration(days=1),
            "1d"
        ).alias("target_date")
    )
    .explode("target_date") 
)

grid = grid.with_columns(
    (pl.col("target_date") - pl.col("reportedDate")).dt.total_days().cast(pl.String).alias("relative_day")
)


joined = grid.join(
    daily_news,
    left_on=["symbol", "target_date"],
    right_on=["symbol", "time_published"],
    how="left"
).fill_null(0)


final_result = (
    joined
    .pivot(
        values="news_count",
        index=["symbol", "reportedDate"],
        columns="relative_day",
        aggregate_function="first" 
    )
    .with_columns(
        pl.sum_horizontal(pl.exclude(["symbol", "reportedDate"])).alias("total_window_news")
    )
)

print(final_result.head())


final_result['total_window_news'].describe()

final_result = final_result.filter(pl.col('reportedDate') >= date(2016, 1, 1))

final_result['total_window_news'].describe()

final_result.filter(
    (pl.col('symbol') == 'AAPL') & 
    (pl.col('reportedDate') >= date(2016, 1, 1)))






print(
    final_result.select([
        pl.col("total_window_news").quantile(0.90).alias("90th"),
        pl.col("total_window_news").quantile(0.95).alias("95th"),
        pl.col("total_window_news").quantile(0.99).alias("99th"),
        pl.col("total_window_news").quantile(0.999).alias("99.9th"),
    ])
)

mega_outliers = final_result.filter(pl.col("total_window_news") > 544).sort("total_window_news", descending=True)


yearly_news_distribution = (
    df_nz
    .with_columns(
        pl.col("time_published").dt.year().alias("year")
    )
    .group_by(["symbol", "year"])
    .agg(pl.len().alias("article_count"))
    .pivot(
        values="article_count",
        index="symbol",
        columns="year",
        aggregate_function="first"  
    # .fill_null(0)
    # .sort("symbol")
))

year_cols = [c for c in yearly_news_distribution.columns if c != "symbol"]

sorted_year_cols = sorted(year_cols, key=lambda x: int(x))

final_yearly_dist = (
    yearly_news_distribution
    .select(["symbol"] + sorted_year_cols) # This fixes the shuffled columns
    .with_columns(
        pl.sum_horizontal(sorted_year_cols).alias("total_articles")
    )
)

print(final_yearly_dist.head(20))


final_yearly_dist.sort("total_articles", descending=False).head(20)

source_drift = (
    df_nz.with_columns(pl.col("time_published").dt.year().alias("year"))
    .group_by("year")
    .agg(pl.col("source").n_unique().alias("unique_sources"))
    .sort("year")
)
print(source_drift)

