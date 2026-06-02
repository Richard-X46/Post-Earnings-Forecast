**Data ingestion**
- [ ] OHLCV backfill
- [ ] Earnings dates backfill
- [ ] EPS surprise backfill
- [ ] Transcripts backfill
- [ ] News backfill
- [ ] Put-call options backfill
- [ ] VIX backfill

**Data validation after ingestion**

- [ ] Assert no duplicate rows per ticker and earnings date
- [ ] Assert no missing quarters per ticker
- [ ] Assert t−10 OHLCV rows exist before each earnings date
- [ ] Flag tickers with missing transcripts
- [ ] Flag tickers with missing news coverage
- [ ] Verify AWS S3 cache and shared team access

**Pivot and structure**

- [ ] Build raw daily table — one row per ticker per trading day
- [ ] Pivot to Quarter · Ticker · d-2 · d-1 · d column format

**EDA : descriptive statistics**

- [ ] Row and column counts per data source
- [ ] Descriptive stats for OHLCV
- [ ] Descriptive stats for EPS surprise
- [ ] Descriptive stats for put/call ratio
- [ ] Descriptive stats for VIX
- [ ] Descriptive stats for news sentiment scores
- [ ] Frequency distribution of UP vs DOWN labels

**EDA : missing value analysis**

- [ ] % missing per feature
- [ ] Missing value pattern analysis
- [ ] Document cause per missingness type

**EDA : outlier detection**

- [ ] Box plots for OHLCV
- [ ] Box plots for EPS surprise
- [ ] Box plots for VIX
- [ ] Box plots for put/call ratio
- [ ] Histogram per numeric feature

**EDA : correlation and relationships**

- [ ] Correlation heatmap across all features
- [ ] Correlation of each feature with target variable
- [ ] Scatter plot - EPS surprise vs post-earnings return
- [ ] Scatter plot - news sentiment vs post-earnings return
- [ ] Scatter plot - VIX vs post-earnings return magnitude
- [ ] Scatter plot - put/call ratio vs post-earnings direction
- [ ] Pair plots - EPS surprise · news sentiment · VIX · put/call · target

**EDA : class distribution**

- [ ] UP vs DOWN balance overall
- [ ] UP vs DOWN balance per year 2014–2025
- [ ] Distribution per market regime


**EDA : time series**

- [ ] Average EPS surprise per quarter over 2014–2025
- [ ] Average VIX per quarter over 2014–2025
- [ ] Average put/call ratio per quarter over 2014–2025
- [ ] Average post-earnings return per quarter over 2014–2025


**EDA : feature reduction**

- [ ] Rank features by correlation with target
- [ ] Flag near-zero correlation features
- [ ] Flag multicollinear feature pairs
- [ ] Flag zero-variance or near-constant features


**EDA : business interpretation**

- [ ] Interpret every chart in business terms
- [ ] Identify top 3 most predictive features from EDA
- [ ] Document unexpected findings
- [ ] State early hypothesis - which signal category looks most predictive




