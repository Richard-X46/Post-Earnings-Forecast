# Fundamental Indicators for PEAD Project

## Recommended Fundamental Features and Source Columns

The objective is to identify a small set of accounting-based features that complement existing earnings, technical, sentiment, and market-based features already present in the PEAD modeling pipeline. The selected indicators focus on business growth, profitability quality, cash generation, and financial risk, all of which may influence the magnitude and persistence of post-earnings announcement drift.

---

## Indicator 1: Revenue Growth (QoQ)

### What it measures

Quarter-over-quarter growth in company revenue.

Revenue growth provides information about underlying business expansion that is not captured by EPS surprise alone. A company can beat earnings expectations through cost reductions or accounting effects, while revenue growth reflects genuine demand for products and services.

### Source File

`av_income_statement.parquet`

### Columns Required

* symbol
* fiscalDateEnding
* totalRevenue

### Formula

```python
revenue_growth_qoq = (
    totalRevenue - totalRevenue.shift(1)
) / totalRevenue.shift(1).abs()
```

### Modeling Usage

* Measures top-line business momentum.
* Helps distinguish high-quality earnings beats from cost-driven beats.
* Can be used directly in both regression and classification models.
* May be evaluated as an interaction term with EPS surprise.

---

## Indicator 2: Gross Margin

### What it measures

Profitability of core business operations before operating expenses.

Margin expansion often signals improving business quality, while declining margins may indicate pricing pressure or rising costs.

### Source File

`av_income_statement.parquet`

### Columns Required

* symbol
* fiscalDateEnding
* grossProfit
* totalRevenue

### Formulas

```python
gross_margin = grossProfit / totalRevenue

gross_margin_qoq = (
    gross_margin - gross_margin.shift(1)
)
```

### Modeling Usage

* Captures operational efficiency.
* Helps evaluate earnings quality.
* Margin expansion combined with positive earnings surprises may strengthen PEAD continuation effects.
* Both the absolute level and quarterly change should be retained.

---

## Indicator 3: Debt-to-Equity Ratio

### What it measures

Financial leverage and balance-sheet risk.

Highly leveraged firms often experience stronger market reactions to earnings surprises because earnings outcomes have a larger impact on perceived financial stability.

### Source File

`av_balance_sheet.parquet`

### Columns Required

* symbol
* fiscalDateEnding
* totalLiabilities
* totalShareholderEquity

### Formula

```python
debt_to_equity = (
    totalLiabilities /
    totalShareholderEquity
)
```

### Modeling Usage

* Measures financial risk.
* Can act as a conditioning variable for earnings surprises.
* Useful for identifying firms where positive or negative surprises may generate amplified drift.

---

## Indicator 4: Free Cash Flow Margin

### What it measures

Cash generation efficiency relative to revenue.

Unlike EPS, free cash flow is more difficult to manipulate and provides insight into the underlying quality of earnings.

### Source Files

`av_cash_flow.parquet`
`av_income_statement.parquet`

### Columns Required

Cash Flow:

* operatingCashflow
* capitalExpenditures

Income Statement:

* totalRevenue

### Formulas

```python
free_cash_flow = (
    operatingCashflow -
    abs(capitalExpenditures)
)

fcf_margin = (
    free_cash_flow /
    totalRevenue
)
```

### Modeling Usage

* Evaluates earnings quality.
* Identifies firms generating real cash rather than accounting profits.
* Useful for separating sustainable earnings beats from weaker earnings signals.

---

## Indicator 5: Return on Equity (ROE)

### What it measures

Profitability relative to shareholder capital.

ROE is one of the most widely used measures of corporate performance and capital efficiency.

### Source Files

`av_income_statement.parquet`
`av_balance_sheet.parquet`

### Columns Required

Income Statement:

* netIncome

Balance Sheet:

* totalShareholderEquity

### Formula

```python
roe = (
    netIncome /
    totalShareholderEquity
)
```

### Modeling Usage

* Captures business quality and management effectiveness.
* High-ROE firms may exhibit stronger continuation after positive earnings surprises.
* Provides complementary information to EPS-based measures already present in the dataset.

---

# Proposed Feature Set

| Feature            | Source                           |
| ------------------ | -------------------------------- |
| revenue_growth_qoq | Income Statement                 |
| gross_margin       | Income Statement                 |
| gross_margin_qoq   | Income Statement                 |
| debt_to_equity     | Balance Sheet                    |
| fcf_margin         | Cash Flow + Income Statement     |
| roe                | Income Statement + Balance Sheet |

---

# Required Source Columns

## Income Statement

```text
symbol
fiscalDateEnding
totalRevenue
grossProfit
netIncome
```

## Balance Sheet

```text
symbol
fiscalDateEnding
totalLiabilities
totalShareholderEquity
```

## Cash Flow

```text
symbol
fiscalDateEnding
operatingCashflow
capitalExpenditures
```

---

# Data Validation Summary

The required source columns were verified in the Alpha Vantage financial statement datasets.

### Coverage Summary

* Income Statement: 497 symbols (2001–2026)
* Balance Sheet: 503 symbols (1997–2026)
* Cash Flow: 501 symbols (2000–2026)

### Data Quality

* totalRevenue: 0 nulls
* grossProfit: 0 nulls
* netIncome: 0 nulls
* totalLiabilities: 0 nulls
* totalShareholderEquity: 0 nulls
* capitalExpenditures: 0 nulls
* operatingCashflow: 114 nulls

The datasets provide sufficient coverage for the planned 2014–2025 modeling period.

---

# Implementation Notes

The income statement and balance sheet variables are stored as strings and must be converted to numeric values before feature engineering.

Example:

```python
pl.col("totalRevenue") \
    .replace("None", None) \
    .cast(pl.Float64)
```

Cash flow variables are already stored as numeric types.

Before computing quarterly changes, data should be sorted by symbol and fiscalDateEnding.

```python
df = df.sort(["symbol", "fiscalDateEnding"])
```

To avoid data leakage, engineered features should be joined to the earnings-event dataset using an as-of backward join so that only information available before the earnings announcement date is used.

---

# Output Dataset

The final deliverable should be a derived feature table containing only engineered fundamental indicators rather than raw financial statement fields.

Suggested output file:

```text
fundamental_features.parquet
```

Suggested schema:

```text
symbol
fiscalDateEnding
revenue_growth_qoq
gross_margin
gross_margin_qoq
debt_to_equity
fcf_margin
roe
```

This dataset can then be joined to the earnings-event modeling table using an as-of join while avoiding leakage from future financial statements.

These features are proposed candidates for modeling and may be further refined through feature selection, correlation analysis, and model importance evaluation during the modeling phase.
