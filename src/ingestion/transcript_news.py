import yfinance as yf
import pandas as pd
import requests as r
import random
import re
from dotenv import load_dotenv
import os
import ast
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import requests
import time
from stem import Signal
from stem.control import Controller



load_dotenv()


def keygen():
    csrf_url = "https://www.alphavantage.co/support/#api-key"
    session = r.Session()
    response = session.get(csrf_url)
    csrf_token = session.cookies.get('csrftoken')

    post_url = "https://www.alphavantage.co/create_post/"

    occupations = ['Educator', 'Student', 'Investor', 'Software Developer']
    companies = ['Google', 'Microsoft', 'Apple', 'College', 'University']
    first_names = ['john', 'jane', 'alex', 'emily']
    last_names = ['doe', 'smith', 'johnson', 'williams']
    domains = ['gmail.com', 'aol.com', 'outlook.com', 'yahoo.com']

    payload = {
        "first_text": "deprecated",
        "last_text": "deprecated",
        "occupation_text": random.choice(occupations),
        "organization_text": random.choice(companies),
        "email_text": f"{random.choice(first_names)}.{random.choice(last_names)}{random.randint(1, 1000)}@{random.choice(domains)}"
    }
    user_agents = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/60.0.3112.113 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/61.0.3163.100 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/62.0.3202.94 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/63.0.3239.132 Safari/537.36"
    ]

    headers = {
        "Referer": "https://www.alphavantage.co/",
        "User-Agent": random.choice(user_agents),
        "X-CSRFToken": csrf_token
    }

    res = session.post(post_url, data=payload, headers=headers)
    key_res=res.json()
    print(res.text,res.json())
    pattern = r'[A-Z0-9]{16}'
    match = re.search(pattern, key_res['text'])
    if match:
        val = match.group(0)
    else:
        val = None
    return val


# keygen()
AV_keys = os.getenv("AV_KEYS")
test_key = ast.literal_eval(AV_keys)[0]




TOR_SOCKS  = "socks5h://127.0.0.1:9050"  
TOR_CTRL   = ("127.0.0.1", 9051)
TOR_PASSWD = ""                             

proxies = {
    "http":  TOR_SOCKS,
    "https": TOR_SOCKS,
}

def new_circuit():
    with Controller.from_port(address="127.0.0.1", port=9051) as ctrl:
        ctrl.authenticate(password=TOR_PASSWD)
        ctrl.signal(Signal.NEWNYM)
    time.sleep(5)


def tor_get(url: str, retries: int = 3) -> dict:
    for attempt in range(retries):
        try:
            resp = requests.get(url, proxies=proxies, timeout=30)
            data = resp.json()
            # Rotate if rate-limited
            if "rate limit" in str(data).lower() or "Information" in data:
                print(f"Rate limited, rotating circuit (attempt {attempt+1})")
                new_circuit()
                continue
            return data
        except Exception as e:
            print(f"Request failed: {e}, rotating...")
            new_circuit()
    return {}



def get_earnings_call_transcript(sym, quarter, apikey=test_key):
    url = (
        f"https://www.alphavantage.co/query"
        f"?function=EARNINGS_CALL_TRANSCRIPT"
        f"&symbol={sym}&quarter={quarter}&apikey={apikey}"
    )
    return tor_get(url)

def get_news_sentiment(sym,time_from,time_to, limit,topics, sort, apikey=test_key):
    url = (
        f"https://www.alphavantage.co/query"
        f"?function=NEWS_SENTIMENT"
        f"&tickers={sym}&time_from={time_from}&time_to={time_to}"
        f"&topics={topics}"
        f"&limit={limit}&sort={sort}&apikey={apikey}"
    )
    return tor_get(url)








if __name__ == "__main__": 

    sym = "TSLA"

    data = get_earnings_call_transcript(sym,"2014Q3",test_key)
    data
    data.keys()
    data

    get_news_sentiment(sym,"20240101T0000",30,"EARLIEST")