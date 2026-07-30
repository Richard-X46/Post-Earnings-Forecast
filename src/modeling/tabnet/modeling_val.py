import polars as pl


# ---- Load tech modeling table (primary) ----
tech_path = "src/data/model_staging/tech_modeling_table.parquet"
df_tech = pl.read_parquet(tech_path)
print(f"Tech table shape: {df_tech.shape}")




# ---- Load fundamental features ----
fund_path = "src/data/model_staging/fundamentalIndicators/modeling_fundamentals.parquet"
df_fund = pl.read_parquet(fund_path)

df_fund = df_fund.select([
    "symbol",
    pl.col("reportedDate").alias("earnings_date"),
    "eps_growth_qoq",
    "revenue_growth_qoq",
    "gross_margin", "gross_margin_qoq",
    "debt_to_equity", "debt_to_equity_qoq",
    "fcf_margin", "fcf_margin_qoq",
    "roe", "roe_qoq"
])

# ---- Load FinBERT sentiment ----
df_finbert = pl.read_parquet("src/data/model_staging/finbert_tx_agg_weighted.parquet")
df_finbert = df_finbert.select([
    pl.col("symbol"),
    pl.col("reportedDate").alias("earnings_date"),
    "pos_prob",
    "neg_prob",
])
print(f"FinBERT shape: {df_finbert.shape}")

# ---- Load NZ news sentiment ----
df_nz = pl.read_parquet("src/data/model_staging/nz_sentiment.parquet")
df_nz = df_nz.select([
    pl.col("symbol"),
    pl.col("reportedDate").alias("earnings_date"),
    "overall_sentiment_score_pre",
    "ticker_sentiment_score_pre",
    # "relevance_score_pre",
    "overall_sentiment_score_post",
    "ticker_sentiment_score_post",
    # "relevance_score_post",
])
print(f"NZ sentiment shape: {df_nz.shape}")

# ---- Merge all onto tech table ----
df_model = df_tech.join(df_fund, on=["symbol", "earnings_date"], how="left")
df_model = df_model.join(df_finbert, on=["symbol", "earnings_date"], how="left", suffix="_fb")
df_model = df_model.join(df_nz, on=["symbol", "earnings_date"], how="left", suffix="_nz")

print(f"Combined table shape: {df_model.shape}")
print(f"Target: {df_model['target_class'].mean():.1%} positive")
print(f"\nNull counts for new sentiment cols:")
cols_added = [
    "pos_prob", "neg_prob",
    "overall_sentiment_score_pre", "ticker_sentiment_score_pre", # "relevance_score_pre",
    "overall_sentiment_score_post", "ticker_sentiment_score_post", # "relevance_score_post",
]

print(df_model.select(cols_added).null_count())