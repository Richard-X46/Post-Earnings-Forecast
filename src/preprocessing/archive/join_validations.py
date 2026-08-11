import polars as pl
pl.Config.set_tbl_formatting("ASCII_FULL")
pl.Config.restore_defaults()
# show 40 rows
pl.Config.set_tbl_rows(100)
# range of all dates in 2025

df  = pl.date_range(
    start=pl.datetime(2025, 1, 1),
    end=pl.datetime(2025, 12, 31),
    interval="1d",
    eager=True,
).to_frame("date")


# adding month name and day of week name
df = df.with_columns(
    pl.col("date").dt.strftime("%B").alias("month"),
    pl.col("date").dt.strftime("%A").alias("day_of_week"),
)

df


# filter out sundays

sun = df.filter(pl.col("day_of_week") == "Sunday")

#
sun['sun_date'] = sun.select(pl.col("date").alias("sun_date"))
sun = sun.select(pl.all().name.suffix("_sun"))

sun

df.join_asof(
    sun,
    left_on="date",
    right_on="date_sun",
    strategy="nearest",
    tolerance="4d",
    suffix="_sun",)


# testing on join where
df.join_where(
    sun,
    left_on="date",
    right_on="date_sun",
    condition=pl.col("date").diff().abs() <= pl.duration(days=4),
    suffix="_sun",
)
