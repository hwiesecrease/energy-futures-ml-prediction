# Crude Oil Price Prediction

An end-to-end pipeline that pulls energy futures data, engineers
technical and cross-commodity features, and trains an LSTM to
predict next-day crude oil returns.

## Data
Five energy futures from Yahoo Finance, one year of daily bars:
WTI crude (CL=F), Brent (BZ=F), heating oil (HO=F), RBOB gasoline
(RB=F), natural gas (NG=F).

## Features
- OHLCV
- 20- and 50-day moving averages
- RSI (14-day)
- Volume-weighted momentum (14-day return scaled by relative volume,
  smoothed over 5 days)
- **Refinery spread** — gasoline/crude and heating-oil/crude ratios
  averaged, then expressed as percent deviation from a 30-day mean.
  A crack-spread proxy for refining margin pressure.

## Model
Two stacked LSTM layers (50 units each) with 0.3 dropout, a 25-unit
dense layer, and a single-value output. 30-day lookback window,
12 features. Adam at 1e-3, MSE loss, early stopping on validation
loss with patience 10.

## Results
171 windows total — 136 train, 35 test.

- Test MSE: 0.00070
- Test MAE: 0.0201

**The model did not learn directional signal.** Nearly every prediction
falls between -2% and +1%, clustered tightly around zero, while actual
next-day returns span roughly -8% to +7%. The predicted-vs-actual scatter
shows a flat horizontal cloud against the 1:1 line — the model is
effectively predicting the average return every day, and mean absolute
error (2.0%) is about the size of a typical daily move.

## What I'd change
- **More history.** 136 training windows is far too few for an LSTM
  of this capacity. Five to ten years of data instead of one.
- **Reframe as classification.** Predicting direction (up/down) is a
  more tractable target than predicting the return magnitude.
- **Drop dead features.** `Dividends` and `Stock Splits` are always
  zero for futures contracts and add nothing.
- **Walk-forward validation** instead of a single chronological split,
  so results aren't dependent on one test period.

## Running it
Open `crude_oil_lstm_pipeline.ipynb` in Jupyter.
Requires: `yfinance`, `pandas`, `numpy`, `tensorflow`, `scikit-learn`,
`matplotlib`.


