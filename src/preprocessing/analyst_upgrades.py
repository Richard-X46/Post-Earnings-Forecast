import polars as pl

path_au = "src/data/metrics/upgrades_downgrades.parquet"
df_au = pl.scan_parquet(path_au).collect()

earnings_path = "src/data/backup/earnings_delta_backup.parquet"
earnings_df = pl.scan_parquet(earnings_path).collect()

earnings = earnings_df.select("symbol", "reportedDate").unique().sort(["symbol", "reportedDate"])
analyst = df_au.sort(["symbol", "grade_date"])


analyst.columns

analyst.null_count().transpose(include_header=True)

analyst.slice(11030 , 1).transpose(include_header=True)

analyst['Action'].value_counts()
analyst['priceTargetAction'].value_counts()
analyst['Firm'].value_counts().sort("count", descending=True)
analyst['ToGrade'].value_counts()
analyst['FromGrade'].value_counts()






# --- pre earnings analyst action ---
pre = earnings.join_asof(
    analyst,
    left_on="reportedDate",
    right_on="grade_date",
    by="symbol",
    strategy="backward",
    tolerance="10d",
).rename({
    "GradeDate": "GradeDate_pre",
    "Firm": "Firm_pre",
    "ToGrade": "ToGrade_pre",
    "FromGrade": "FromGrade_pre",
    "Action": "Action_pre",
    "priceTargetAction": "priceTargetAction_pre",
    "currentPriceTarget": "currentPriceTarget_pre",
    "priorPriceTarget": "priorPriceTarget_pre",
    "grade_date": "grade_date_pre",
})


pre = analyst.join_asof(
    earnings,
    left_on="grade_date",
    right_on="reportedDate",
    by="symbol",
    strategy="backward",
    tolerance="10d",
).rename({
    "reportedDate": "reportedDate_pre",
    # other earnings cols...
})



# drop nulls on reportedDate_pre to see only analyst actions that have a subsequent earnings report within 10 days
pre.filter(pl.col("reportedDate_pre").is_not_null())



# unique of symbols and reportedDate
pre.select(pl.col("symbol"), pl.col("reportedDate_pre")).unique().sort(["symbol", "reportedDate_pre"])

earnings


# joining back earnings on pre to get the full earnings table with the pre-earnings analyst action
pre_full = pre.join(
    earnings,
    left_on=["symbol", "reportedDate_pre"],
    right_on=["symbol", "reportedDate"],
    how="right",
)



# agg pre on symbol, reportedDate to get counts of toGrade and fromGrade
pre.select(pl.all().null_count()).melt().filter(pl.col("value") > 0)


# post earnings analyst action


post = analyst.join_asof(
    earnings,
    left_on="grade_date",
    right_on="reportedDate",
    by="symbol",
    strategy="forward",
    tolerance="10d",
).rename({
    "reportedDate": "reportedDate_post",
    # other earnings cols...
})

# joining back earnings on post to get the full earnings table with the post-earnings analyst action
post_full = post.join(
    earnings,
    left_on=["symbol", "reportedDate_post"],
    right_on=["symbol", "reportedDate"],
    how="right",
)


# unique of symbols and reportedDate
post.select(pl.col("symbol"), pl.col("reportedDate_post")).unique().sort(["symbol", "reportedDate_post"])