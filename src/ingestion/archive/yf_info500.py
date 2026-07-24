import yfinance as yf
import pandas as pd
import glob



# reading snp500 tickers from a csv file
snp500_file = glob.glob("src/ingestion/data/snp500*.csv")[0]


snp500_df = pd.read_csv(snp500_file)



sym = "AAPL"

data = yf.Ticker(sym).info


info_data = {}

for sym in snp500_df["Symbol"]:
    try:
        print(f"Fetching info for {sym}...")
        data = yf.Ticker(sym).info
        info_data[sym]   = data
        print(f"added info for {sym}")
        time.sleep(0.2)  # to avoid hitting rate limits

    except Exception as e:
        print(f"Failed to fetch info for {sym}: {e}")



# list of keys in info_data
len(list(info_data.keys()))


# converting info_data to a dataframe
info_df = pd.DataFrame.from_dict(info_data, orient="index")


# write to parquet
info_df.to_parquet("src/ingestion/data/snp500_info.parquet")

for x in info_df.columns: print(x)








tick = yf.Ticker("AAPL")

dir(tick)


tick.quarterly_cash_flow.columns