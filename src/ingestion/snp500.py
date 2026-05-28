# S&P 500 data ingestion logic
import requests
import pandas as pd
from io import StringIO
from dotenv import load_dotenv
import polars as pl
import os


load_dotenv()

def snp500_data():
    headers = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"}

    resp = requests.get("https://www.slickcharts.com/sp500", headers=headers)
    resp.raise_for_status()

    df = pd.read_html(StringIO(resp.text))[0]
    return df


if __name__ == "__main__":

    # snp500 data

    source = f"s3://{os.getenv('S3_BUCKET')}/post-earnings-forecast/snp500/*.csv"
    df = pl.read_csv(source)



    df = snp500_data()
    df.to_clipboard(index=False)
    df.to_csv(f"src/ingestion/data/snp500_{pd.Timestamp.now().strftime('%Y-%m-%d')}.csv", index=False)

    # write this data to s3 using polars
    pl.from_pandas(df).write_csv(f"s3://{os.getenv('S3_BUCKET')}/post-earnings-forecast/snp500/snp500_{pd.Timestamp.now().strftime('%Y-%m-%d')}.csv")