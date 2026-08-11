# put call options  - check Yf else AV
# sector & Industry - info from yf
# VIX  - single call from yf
from ingestion.archive.transcript_news import  get_earnings_call_transcript, keygen,tor_get,get_isolated_proxies
from ingestion.archive.transcript_news import fetch_one ,get_put_call_ratio
import requests as r
import yfinance as yf
import duckdb 
import os
import polars as pl
from datetime import date, timedelta
import pandas as pd


con = duckdb.connect(database=':memory:')



