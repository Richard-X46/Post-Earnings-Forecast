



#----/// geting news sentiment for each transcript date


# av_earnings_df.columns

# av_earnings_df.isna().sum()


# av_earnings_df.columns

# sym
# from src.ingestion.transcript_news import get_news_sentiment


# for sym in tech_list:
#     print(f"Fetching news sentiment for {sym}...")
#     sym_df = av_earnings_df[av_earnings_df['symbol'] == sym]
#     for _, row in sym_df.iterrows():
       
#        # pre earnings news
#         time_from = (row['reportedDate'] - pd.Timedelta(days=7)).strftime('%Y%m%dT0000')
#         time_to = row['reportedDate'].strftime('%Y%m%dTdT0000')
#         pre_data = get_news_sentiment(sym, time_from = time_from, time_to=time_to, limit=100, sort="RELEVANCE",topics="earnings")
#         print(f"Pre-earnings news fetched for {sym} {row['av_quarter']} containing {len(pre_data.get('feed', []))} items")
#        # post earnings news
#         time_from = (row['reportedDate'] + pd.Timedelta(days=1)).strftime('%Y%m%dT0000') 
#         time_to = (row['reportedDate'] + pd.Timedelta(days=7)).strftime('%Y%m%dT0000')
#         post_data = get_news_sentiment(sym, time_from = time_from, time_to=time_to, limit=100, sort="EARLIEST",topics="earnings")
#         print(f"Post-earnings news fetched for {sym} {row['av_quarter']} containing {len(post_data.get('feed', []))} items")
       
#         break
#     break


# data.keys()
# keys_news= ['items', 'sentiment_score_definition', 'relevance_score_definition', 'feed']
# data['sentiment_score_definition']
# data['relevance_score_definition']
# data['feed']

# pd.DataFrame(pre_data['feed']).to_clipboard(index=False)
# pd.DataFrame(post_data['feed']).to_clipboard(index=False)

