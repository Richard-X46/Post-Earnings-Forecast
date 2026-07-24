import requests as r
from dotenv import load_dotenv
import polars as pl
import os
import logging
import time
import pandas as pd
logging.basicConfig(level=logging.INFO)
load_dotenv()


#Balance Sheet

def get_AV_balance_sheet(sym: str) -> dict:
    base_url =f"https://www.alphavantage.co/query?function=BALANCE_SHEET"
    url = f"{base_url}&symbol={sym}&apikey={os.getenv('AV_PREMIUM_KEY')}"
    
    response = r.get(url)
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error fetching balance sheet for {sym}: {response.status_code}")
        return None


# 


sym = "AAPL"
balance_sheet_data = get_AV_balance_sheet(sym)
balance_sheet_data.keys()


df = pl.DataFrame(balance_sheet_data['quarterlyReports'])
df.columns


# Income statement
# https://www.alphavantage.co/query?function=INCOME_STATEMENT&symbol=IBM&apikey=demo
def get_AV_income_statement(sym: str) -> dict:
    base_url =f"https://www.alphavantage.co/query?function=INCOME_STATEMENT"
    url = f"{base_url}&symbol={sym}&apikey={os.getenv('AV_PREMIUM_KEY')}"
    
    response = r.get(url)
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error fetching income statement for {sym}: {response.status_code}")
        return None


# cash flow statement
#https://www.alphavantage.co/query?function=CASH_FLOW&symbol=IBM&apikey=demo

def get_AV_cash_flow(sym: str) -> dict:
    base_url =f"https://www.alphavantage.co/query?function=CASH_FLOW"
    url = f"{base_url}&symbol={sym}&apikey={os.getenv('AV_PREMIUM_KEY')}"
    
    response = r.get(url)
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error fetching cash flow statement for {sym}: {response.status_code}")
        return None 







get_AV_balance_sheet("AAPL").keys()
get_AV_cash_flow("AAPL").keys()
get_AV_income_statement("AAPL").keys()

df = pl.DataFrame(get_AV_cash_flow("AAPL")['quarterlyReports'])

df


AV_CASH={} 
AV_BALANCE = {}
AV_INCOME = {}


# snp 500 list from csv

df = pl.read_csv("src/ingestion/data/snp500_2026-05-23.csv")


# for sym in df['Symbol']:
#     logging.info(f"Fetching financial statements for ======{sym}=========X======")
#     AV_BALANCE[sym] = get_AV_balance_sheet(sym)
#     time.sleep(.8) 
#     if AV_BALANCE[sym] is None:
#         logging.warning(f"Failed to fetch balance sheet for {sym}")
    

#     AV_INCOME[sym] = get_AV_income_statement(sym)
#     if AV_INCOME[sym] is None:
#         logging.warning(f"Failed to fetch income statement for {sym}")
#     AV_CASH[sym] = get_AV_cash_flow(sym)
#     if AV_CASH[sym] is None:
#         logging.warning(f"Failed to fetch cash flow statement for {sym}")

# ------//// backfill for cashflow
for i, sym in enumerate(df["Symbol"].to_list()):

 

    if sym not in AV_CASH or AV_CASH[sym] is None:
        logging.info(f"Backfilling cash flow statement for ======{sym}=========X=total_pending={len(df['Symbol'].to_list()) - i}=====")
        AV_CASH[sym] = get_AV_cash_flow(sym)
        time.sleep(.8) 
        if AV_CASH[sym] is None:
            logging.warning(f"Failed to fetch cash flow statement for {sym}")




# loading quarterly reports from AV_CASH 
AV_CASH_Q = {sym: data['quarterlyReports'] for sym, data in AV_CASH.items() if data is not None}

AV_CASH['AAPL'].keys()

quarterly = {
    sym: data['quarterlyReports']
    for sym, data in AV_CASH.items()
    if data is not None and 'quarterlyReports' in data
}

# loading the quarterly into a pl frame

df_cash = pl.concat([
    pl.DataFrame(reports).with_columns(pl.lit(sym).alias('symbol'))
    for sym, reports in quarterly.items()
], how='vertical_relaxed')





# numeric_cols = df_cash.columns
numeric_cols = df_cash.columns

numeric_cols = [c for c in numeric_cols if c not in ('fiscalDateEnding', 'reportedCurrency', 'symbol')]

df_cash = df_cash.with_columns([
    pl.col('fiscalDateEnding').str.to_date('%Y-%m-%d'),
    *[
        pl.when(pl.col(c) == 'None')
          .then(None)
          .otherwise(pl.col(c))
          .cast(pl.Float64)
          .alias(c)
        for c in numeric_cols
    ]
]).sort(['symbol', 'fiscalDateEnding'])


# cash flow write to data/backup as a temp file for now
df_cash.write_parquet("src/ingestion/data/backup/av_cash_flow.parquet")




# ------//// backfill for Balance sheet
for i, sym in enumerate(df["Symbol"].to_list()):

    if sym not in AV_BALANCE or AV_BALANCE[sym] is None:
        logging.info(f"Backfilling balance sheet - {len(df['Symbol'].to_list()) - i}===== fetched for {sym}")
        AV_BALANCE[sym] = get_AV_balance_sheet(sym)
        time.sleep(.8) 
        if AV_BALANCE[sym] is None:
            logging.warning(f"Failed to fetch balance sheet for {sym}")

# loading the quarterly balance sheet reports into a pl frame
AV_BALANCE_Q = {
    sym: data['quarterlyReports'] 
    for sym, data in AV_BALANCE.items() 
    if isinstance(data, dict) and 'quarterlyReports' in data
}
len(AV_BALANCE_Q.keys())

# missing keys in balance sheet
missing_balance_keys = [sym for sym in df["Symbol"].to_list() if sym not in AV_BALANCE_Q]

# missing_balance_keys = ['BRK-B', 'SBAC', 'BF-B']
# # fetching for missing keys

# for sym in missing_balance_keys:
#     logging.info(f"Fetching balance sheet for missing symbol: {sym}")
#     AV_BALANCE[sym] = get_AV_balance_sheet(sym)
#     time.sleep(.8) 
#     if AV_BALANCE[sym] is None:
#         logging.warning(f"Failed to fetch balance sheet for {sym}")

#loading quarterly balance sheet data into a pl frame
df_balance = pl.concat([
    pl.DataFrame(reports).with_columns(pl.lit(sym).alias('symbol'))
    for sym, reports in AV_BALANCE_Q.items()
], how='vertical_relaxed')

# writing balance sheet data to backup
df_balance.write_parquet("src/ingestion/data/backup/av_balance_sheet.parquet")



AV_INCOME.keys()
#------//// backfill for income statement
for i, sym in enumerate(df["Symbol"].to_list()):

    if sym not in AV_INCOME or AV_INCOME[sym] is None:
        logging.info(f"Backfilling income statement - {len(df['Symbol'].to_list()) - i}XXX===== fetched for {sym}")
        AV_INCOME[sym] = get_AV_income_statement(sym)
        time.sleep(.8) 
        if AV_INCOME[sym] is None:
            logging.warning(f"Failed to fetch income statement for {sym}")

# len of income statement keys
len(AV_INCOME.keys()) # total fetched 503

# checking for missing keys in income statement
missing_income_keys = [sym for sym in df["Symbol"].to_list() if sym not in AV_INCOME]
# missing_income_keys = ['BRK-B', 'SBAC', 'BF-B']
logging.info(f"Missing income statement keys: {missing_income_keys}")


# pulling quarterly income statement data into a pl frame
AV_INCOME_Q = {
    sym: data['quarterlyReports'] 
    for sym, data in AV_INCOME.items() 
    if isinstance(data, dict) and 'quarterlyReports' in data
}

# loading the quarterly income statement data into a pl frame
df_income = pl.concat([
    pl.DataFrame(reports).with_columns(pl.lit(sym).alias('symbol'))
    for sym, reports in AV_INCOME_Q.items()
], how='vertical_relaxed')

df_income.write_parquet("src/ingestion/data/backup/av_income_statement.parquet")

# writing income statement data to backup



# validation

earnings_df = pl.read_parquet("src/ingestion/data/backup/earnings_delta_backup.parquet")

unique_symbols = earnings_df['symbol'].unique().to_list()


### Cash flow statement validation
AV_CF = pl.read_parquet("src/ingestion/data/backup/av_cash_flow.parquet")

AV_CF.columns

AV_CF.schema

earnings_df.schema

earnings_df.filter(pl.col('symbol') == 'AAPL').sort("reportedDate")

AV_CF.filter(pl.col('symbol') == 'AAPL').sort("fiscalDateEnding")
# Join on symbol and the period end date
joined_df = earnings_df.join(
    AV_CF, 
    on=["symbol", "fiscalDateEnding"], 
    how="left"
)

joined_df.filter(pl.col('symbol') == 'AAPL').select(
    ['symbol', 'fiscalDateEnding', 'reportedDate', 'operatingCashflow']).sort(
    'reportedDate'
    )