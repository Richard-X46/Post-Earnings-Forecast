import requests as r
from dotenv import load_dotenv
import polars as pl
import pandas as pd
import pandas_market_calendars as mcal
from datetime import timedelta
import os
load_dotenv()



# url =https://www.alphavantage.co/query?function=HISTORICAL_OPTIONS&symbol=IBM&date=2017-11-15&apikey=demo&datatype=csv


def get_historical_options(sym: str, date: str, api_key: str) -> pl.DataFrame:
    url = f'https://www.alphavantage.co/query?function=HISTORICAL_OPTIONS&symbol={sym}&date={date}&apikey={api_key}'
    response = r.get(url)
    
    if response.status_code == 200:
        # return the json
        data = response.json()
        return data
    else:
        print(f"Error fetching historical options for {sym} on {date}: {response.status_code}")
        return None
    


def get_trading_days_around(earnings_date, days_before=10, days_after=10):
    nyse = mcal.get_calendar('NYSE')
    # Get a wide range to ensure we cover the offsets
    start_search = earnings_date - timedelta(days=days_before + 10)
    end_search = earnings_date + timedelta(days=days_after + 10)
    
    schedule = nyse.schedule(start_date=start_search, end_date=end_search)
    trading_days = schedule.index.to_series().reset_index(drop=True)
    
    # Find the index of the closest trading day to the earnings date
    # If earnings is on a weekend, it moves to the next available trading day
    t_index = trading_days[trading_days >= pd.Timestamp(earnings_date)].index[0]
    

    offsets = [-10, -5, -2, -1, 0, 1, 10]
    target_indices = [t_index + i for i in offsets]
    
    return trading_days.iloc[target_indices].dt.strftime('%Y-%m-%d').tolist()

def fetch_options_sparse(df_er, get_historical_options_func):
    all_data = []
    
    for _, row in df_er.iterrows():
        ticker = row['ticker']
        e_date = pd.to_datetime(row['earnings_date'])
        
        # Get the specific valid trading dates
        needed_dates = get_trading_days_around(e_date)
        
        for date_str in needed_dates:
            try:
                # One call per specific trading day
                df = get_historical_options_func(ticker=ticker, date=date_str)
                df['snapshot_date'] = date_str
                df['earnings_date'] = e_date
                all_data.append(df)
            except Exception as e:
                print(f"Skipping {date_str} for {ticker}: {e}")
                
    return pd.concat(all_data)

# load snp500 symbols

df = pl.read_csv("src/ingestion/data/snp500_2026-05-23.csv")
df

# earnings data for backfill
er_backup = "src/ingestion/data/backup/earnings_delta_backup.parquet"
df_er = pl.read_parquet(er_backup)

sym = "NVDA"

data = get_historical_options(sym, "2025-10-14", os.getenv("AV_PREMIUM_KEY"))

df = pl.DataFrame(data['data'])


df.schema