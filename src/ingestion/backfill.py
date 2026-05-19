import yfinance as yf
import pandas as pd
from ingestion.transcript_news import  get_earnings_call_transcript, get_news_sentiment,keygen,tor_get
import duckdb
import datetime
import importlib
import sys
import secrets
from dotenv import load_dotenv
import os
import base64

pd.set_option('display.max_columns', None)

load_dotenv()

# key for encrypting parquet files in duckdb, set as env variable and passed to duckdb PRAGMA
# key_bytes =secrets.token_hex(16)

con = duckdb.connect(database=':memory:')
con.sql(f"PRAGMA add_parquet_key('main_key', {os.getenv('DUCKDB_KEY')});")


top_tech = yf.Sector("technology").top_companies.reset_index()
top_health = yf.Sector("healthcare").top_companies.reset_index()
top_finance = yf.Sector("financial-services").top_companies.reset_index()


# Time series
# keygen()
tech_list = top_tech['symbol'].to_list()[:10]
finance_list = top_finance['symbol'].to_list()[:10]
health_list = top_health['symbol'].to_list()[:10]





data = {x : yf.Ticker(x).history(period="max") for x in tech_list}

# ---- /// adding the symbol as a column to each dataframe using comprehension
data = {symbol: df.assign(symbol=symbol) for symbol, df in data.items()}

df = pd.concat(data.values(), ignore_index=True)


# ---- ///Vix Data - Base paper
VIX = yf.Ticker("^VIX").history(period="max").assign(symbol="^VIX")
VIX.head()




# ---- /// sentiment data  - base paper
df = pd.read_excel("src/ingestion/data/news_sentiment_data.xlsx", sheet_name="Data")
# list sheet names
xls = pd.ExcelFile("src/ingestion/data/news_sentiment_data.xlsx")
print(xls.sheet_names)







# ---- /// top tech backfill

tech_list



def get_av_earnings(sym, apikey=test_key):
    url = f"https://www.alphavantage.co/query?function=EARNINGS&symbol={sym}&apikey={apikey}"
    print(f"Fetching AV earnings for {sym} from {url}")
    res = tor_get(url)
    
    df = pd.DataFrame(res['quarterlyEarnings'])
    df['symbol'] = sym
    df['fiscalDateEnding'] = pd.to_datetime(df['fiscalDateEnding'])
    df['reportedDate'] = pd.to_datetime(df['reportedDate'])
    
    # av_quarter = year of fiscalDateEnding + quarter number based on month
    df['fiscal_month'] = df['fiscalDateEnding'].dt.month
    df['fiscal_year'] = df['fiscalDateEnding'].dt.year
    
    # quarter end months tell you which quarter it is
    # Jan/Apr/Jul/Oct = Q4/Q1/Q2/Q3 depending on fiscal calendar
    # but AV transcript API uses YYYY + Q1-Q4 based on fiscal calendar order
    # safest: just use fiscalDateEnding as the key to match transcripts
    
    df['av_quarter'] = df['fiscalDateEnding'].dt.year.astype(str) + 'Q' + \
                       df['fiscalDateEnding'].dt.quarter.astype(str)
    
    # time_from for news = reportedDate
    df['time_from'] = df['reportedDate'].dt.strftime('%Y%m%dT0000')
    
    return df[['symbol', 'fiscalDateEnding', 'reportedDate', 'time_from', 
               'av_quarter', 'reportedEPS', 'estimatedEPS', 'surprise', 
               'surprisePercentage', 'reportTime']]


# build a dict of AV earnings data for each tech symbol
av_earnings_dict = {sym: get_av_earnings(sym) for sym in tech_list}

# example: inspect one
av_earnings_dict.get(tech_list[0])
# concatenate into a single dataframe
av_earnings_df = pd.concat(av_earnings_dict.values(), ignore_index=True)

# filter av_earnings_df greater than 2014-01-01
av_earnings_df = av_earnings_df[av_earnings_df['reportedDate'] >= '2014-01-01']


av_earnings_df.to_clipboard(index=False)



# ---- /// get transcripts for each av_quarter in av_earnings_df for each symbol

def fetch_transcripts_for_symbol(sym, earnings_df):
    """Fetch transcripts for all quarters of a given symbol."""
    sym_df = earnings_df[earnings_df['symbol'] == sym]
    print(f"Processing {len(sym_df)} quarters for {sym}")

    for _, row in sym_df.iterrows():
        quarter = row['av_quarter']
        print(f"Fetching transcript for {sym} {quarter}")
        transcript_data = get_earnings_call_transcript(sym, quarter)
        
        if not transcript_data or 'transcript' not in transcript_data:
            print(f"No transcript found for {sym} {quarter}, skipping...")
            continue  # use continue not break, to try other quarters

        current_fiscal = row['fiscalDateEnding']
        remaining = sym_df[sym_df['fiscalDateEnding'] > current_fiscal]['av_quarter'].tolist()
        # print(f"Remaining quarters for {sym}: {remaining}")

        # use .at for single cell assignment of a list object
        idx = earnings_df[(earnings_df['symbol'] == sym) & 
                          (earnings_df['av_quarter'] == quarter)].index[0]
        earnings_df.at[idx, 'transcript'] = transcript_data['transcript']


# Fetch transcripts for each symbol
for sym in av_earnings_df['symbol'].unique():
    pass
    
    



tech_list
['NVDA', 'AAPL', 'MSFT', 'AVGO', 'MU', 'AMD', 'ORCL', 'INTC', 'CSCO', 'LRCX']


## completed
completed = ['AMD','NVDA']


for sym in tech_list:
    if sym in completed:
        print(f"Skipping {sym}, already completed.")
        continue
    print(f"Fetching transcripts for {sym}...")
    fetch_transcripts_for_symbol(sym, av_earnings_df)
    print(f"Completed {sym}.")



    query =f"""
    select * from av_earnings_df
    where symbol = '{sym}'
    """

    duckdb.query(query).to_df().to_csv(f"src/ingestion/data/transcripts/{sym}_earnings_transcripts.csv", index=False)
    print(f"Saved transcripts for {sym} to CSV.")
    completed.append(sym)
# fetch_transcripts_for_symbol(sym, av_earnings_df)


# writing the transcript results into a hive paritioned by symbol

con.sql("""
    COPY (
        SELECT symbol, fiscalDateEnding, reportedDate, time_from,
               av_quarter, reportedEPS, estimatedEPS, surprise,
               surprisePercentage, reportTime, transcript
        FROM av_earnings_df
    ) TO 'src/ingestion/data/transcripts/'
    (
        FORMAT PARQUET,
        PARTITION_BY (symbol),
        ENCRYPTION_CONFIG {footer_key: 'main_key'},
    OVERWRITE true)
""")



# Correct syntax for reading encrypted files
df_nvda = con.sql("""
    SELECT * FROM read_parquet(
        'src/ingestion/data/transcripts/symbol=NVDA/*.parquet',
        encryption_config = {footer_key: 'main_key'}
    )
""").to_df()



#----/// geting news sentiment for each transcript date


av_earnings_df.columns

av_earnings_df.isna().sum()


av_earnings_df.columns

sym
from src.ingestion.transcript_news import get_news_sentiment


for sym in tech_list:
    print(f"Fetching news sentiment for {sym}...")
    sym_df = av_earnings_df[av_earnings_df['symbol'] == sym]
    for _, row in sym_df.iterrows():
       
       # pre earnings news
        time_from = (row['reportedDate'] - pd.Timedelta(days=7)).strftime('%Y%m%dT0000')
        time_to = row['reportedDate'].strftime('%Y%m%dTdT0000')
        pre_data = get_news_sentiment(sym, time_from = time_from, time_to=time_to, limit=100, sort="RELEVANCE",topics="earnings")
        print(f"Pre-earnings news fetched for {sym} {row['av_quarter']} containing {len(pre_data.get('feed', []))} items")
       # post earnings news
        time_from = (row['reportedDate'] + pd.Timedelta(days=1)).strftime('%Y%m%dT0000') 
        time_to = (row['reportedDate'] + pd.Timedelta(days=7)).strftime('%Y%m%dT0000')
        post_data = get_news_sentiment(sym, time_from = time_from, time_to=time_to, limit=100, sort="EARLIEST",topics="earnings")
        print(f"Post-earnings news fetched for {sym} {row['av_quarter']} containing {len(post_data.get('feed', []))} items")
       
        break
    break


data.keys()
keys_news= ['items', 'sentiment_score_definition', 'relevance_score_definition', 'feed']
data['sentiment_score_definition']
data['relevance_score_definition']
data['feed']

pd.DataFrame(pre_data['feed']).to_clipboard(index=False)
pd.DataFrame(post_data['feed']).to_clipboard(index=False)





yf.Ticker("AAPL").info