# Strategy File Implementation TODO

## Highest Priority

- [ ] Prevent threshold-selection leakage.
  - Select the take-profit threshold using only prior quarters.
  - Apply it to the next unseen quarter.
  - Repeat using walk-forward evaluation.
  - Report only the combined out-of-sample result.

- [ ] Fix the Sharpe calculation.
  - Reindex the equity curve to every market trading day.
  - Include zero-return days when no trades exit.
  - Keep portfolio-level and deployed-capital metrics clearly separate.

- [ ] Add transaction costs and slippage.
  - Test at least 0, 5, 10, and 25 basis points.
  - Apply costs on both entry and exit.
  - Model gap-through-threshold execution realistically.

- [ ] Fix Strategy 3 position accounting.
  - Track the actual deployed amount for each split leg.
  - Do not pass `amt_first` as the position size for both legs.
  - Recalculate cash usage, skipped trades, deployed capital, and returns.

- [ ] Process same-day exits before entries.
  - Add explicit event priority so exits release cash before new entries are evaluated.

## Validation

- [ ] Add benchmarks.
  - SPY buy-and-hold.
  - Equal-weight benchmark across eligible earnings events.
  - Buying every eligible signal.
  - Buying all predicted classes.

- [ ] Add risk metrics.
  - Maximum drawdown.
  - Calmar ratio.
  - Sortino ratio.
  - Volatility.
  - Average holding period.
  - Concurrent positions.
  - Capital exposure.
  - Worst day, month, and quarter.

- [ ] Add bootstrap confidence intervals for total P&L, win rate, Sharpe, and maximum drawdown.

- [ ] Report model and strategy performance by walk-forward fold.

- [ ] Check for duplicate `(symbol, earnings_date)` signals.

- [ ] Check for overlapping positions in the same symbol.

- [ ] Check that entry dates precede exit dates.

- [ ] Exclude or separately report periods with too few observations, especially 2026.

## Trade Ledger

- [ ] Save a complete trade ledger containing:
  - Symbol.
  - Signal date.
  - Entry date and price.
  - Exit date and price.
  - Threshold.
  - Model fold.
  - Gross P&L.
  - Transaction costs.
  - Slippage.
  - Net P&L.

## Cleanup

- [ ] Correct the documentation mismatch: the notebook says `$100` per position while the code uses `$200`.

- [ ] Correct the threshold comment: the code currently sweeps 1% through 5%, not 10%.

- [ ] Replace `signals.group_by` with an actual grouped operation such as `signals.group_by("quarter").len()`.

- [ ] Confirm whether threshold crossing should use `>` or `>=`.

- [ ] Confirm that the strategy enters at `t+1` close and that all required information is available at that time.
