"""Backfill historical put-call ratio data from Alpha Vantage.

This mirrors the transcript backfill pattern, but uses earnings rows as the
driver because the HISTORICAL_PUT_CALL_RATIO endpoint is date-based.
"""

from __future__ import annotations

import json
import logging
import os
import pathlib
import time

from dotenv import load_dotenv
import polars as pl
import requests as r


load_dotenv()
logging.basicConfig(level=logging.INFO)

AV_KEY = os.getenv("AV_PREMIUM_KEY")
bucket = os.getenv("S3_BUCKET")

storage_options = {
	"AWS_ACCESS_KEY_ID": os.getenv("AWS_ACCESS_KEY_ID"),
	"AWS_SECRET_ACCESS_KEY": os.getenv("AWS_SECRET_ACCESS_KEY"),
	"AWS_REGION": "ca-central-1",
}

DELTA_EARNINGS = f"s3://{bucket}/post-earnings-forecast/earnings_delta/"
DELTA_PCR = f"s3://{bucket}/post-earnings-forecast/pcr_delta/"

LOCAL_STAGING = pathlib.Path("src/ingestion/data/pcr_staging")
LOCAL_STAGING.mkdir(parents=True, exist_ok=True)


def get_historical_put_call_ratio(sym: str, date: str, apikey: str = AV_KEY) -> dict | None:
	"""Fetch one historical put-call ratio row for a symbol/date pair."""
	url = (
		"https://www.alphavantage.co/query"
		"?function=HISTORICAL_PUT_CALL_RATIO"
		f"&symbol={sym}"
		f"&date={date}"
		f"&apikey={apikey}"
	)

	logging.info(f"Fetching PCR for {sym} on {date}")
	time.sleep(1)
	response = r.get(url, timeout=60)
	payload = response.json()

	if "Information" in payload or "Note" in payload or "Error Message" in payload:
		logging.warning(f"PCR request failed for {sym} {date}: {payload}")
		return None

	return payload


def _extract_ratio_value(payload: dict) -> dict:
	"""Flatten the common ratio fields while preserving the raw response."""
	ratio_record = payload.get("data") or payload.get("historical_put_call_ratio") or payload
	if isinstance(ratio_record, list):
		ratio_record = ratio_record[0] if ratio_record else {}

	if not isinstance(ratio_record, dict):
		ratio_record = {}

	return {
		"put_call_ratio": ratio_record.get("put_call_ratio") or ratio_record.get("ratio"),
		"call_volume": ratio_record.get("call_volume") or ratio_record.get("calls"),
		"put_volume": ratio_record.get("put_volume") or ratio_record.get("puts"),
		"raw_response": json.dumps(payload),
	}


if __name__ == "__main__":
	earnings = pl.scan_delta(DELTA_EARNINGS, storage_options=storage_options).collect()

	pcr_fetch = (
		earnings
		.select(["symbol", "reportedDate", "av_quarter"])
		.drop_nulls(["symbol", "reportedDate"])
		.unique(subset=["symbol", "reportedDate"])
		.sort(["symbol", "reportedDate"])
	)

	logging.info(f"Prepared {len(pcr_fetch)} symbol/date rows from earnings.")

	already_done = {f.stem for f in LOCAL_STAGING.glob("*.parquet")}
	pcr_fetch = pcr_fetch.filter(~pl.col("symbol").is_in(already_done))

	for sym_val in pcr_fetch["symbol"].unique().to_list():
		fetched_rows: list[dict] = []
		sub_frame = pcr_fetch.filter(pl.col("symbol") == sym_val)
		logging.info(f"FETCHING PCR for {sym_val} with {sub_frame.height} rows")

		for i, row in enumerate(sub_frame.iter_rows(named=True), 1):
			sym = row["symbol"]
			pcr_date = row["reportedDate"].isoformat()
			logging.info(f"[{i}/{sub_frame.height}] {sym} {pcr_date}")

			try:
				payload = get_historical_put_call_ratio(sym, pcr_date, AV_KEY)
				if payload is None:
					continue
				row.update(_extract_ratio_value(payload))
			except Exception as exc:
				logging.warning(f"Failed PCR fetch for {sym} {pcr_date}: {exc}")
				continue

			if row.get("put_call_ratio") is not None:
				fetched_rows.append(row)

		if fetched_rows:
			df = pl.DataFrame(fetched_rows)
			df.write_parquet(LOCAL_STAGING / f"{sym_val}.parquet")
			logging.info(f"Wrote {sym_val}.parquet with {len(df)} rows")

	staged_files = list(LOCAL_STAGING.glob("*.parquet"))
	if not staged_files:
		logging.info("No PCR staging files were created.")
		raise SystemExit(0)

	df = pl.read_parquet(str(LOCAL_STAGING / "*.parquet"))
	df = df.unique(subset=["symbol", "reportedDate"])

	logging.info(f"Staged PCR rows: {len(df)}")

	# Skeleton merge target. Create the Delta table/schema first, then enable.
	# (
	#     df.write_delta(
	#         DELTA_PCR,
	#         mode="merge",
	#         delta_merge_options={
	#             "predicate": "s.symbol = t.symbol AND s.reportedDate = t.reportedDate",
	#             "source_alias": "s",
	#             "target_alias": "t",
	#         },
	#         storage_options=storage_options,
	#     )
	#     .when_matched_update_all()
	#     .when_not_matched_insert_all()
	#     .execute()
	# )

	logging.info("PCR backfill skeleton complete.")



# testing get_historical_put_call_ratio
earnings = pl.scan_delta(DELTA_EARNINGS, storage_options=storage_options).collect()


earnings.schema

earnings.filter(pl.col("symbol") == "AAPL").select(
    ["symbol", "reportedDate"]).sort(
    "reportedDate", descending=True
    )

get_historical_put_call_ratio("AAPL", "2026-01-29", AV_KEY)


get_historical_


(t+10 - t+1) / t+1



264_000/60/60


SYMBOL = "AAPL"
DATE = "2026-01-29"
API_KEY = AV_KEY


url = f'https://www.alphavantage.co/query?function=HISTORICAL_OPTIONS&symbol={SYMBOL}&date={DATE}&apikey={API_KEY}'

response = r.get(url)
data = response.json()


data.keys()
df = pl.read_json(data['data'], try_parse_dates=True)

df.schema
df.glimpse()

# transpose 1 row

df.head(1).transpose(include_header=True)


# set pl to display all rows 
pl.Config.set_tbl_rows(100)

# Inspecting the first contract in the chain to see the available fields
if 'data' in data and len(data['data']) > 0:
    first_contract = data['data'][0]
    print(json.dumps(first_contract, indent=4))


# 

df['type'].value_counts()