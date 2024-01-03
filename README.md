# Stock Price Prediction Project

## Overview

This project utilizes machine learning, specifically LSTM (Long Short-Term Memory) networks, to predict stock prices. Historical stock data is analyzed, and various technical indicators are calculated to train the predictive model.

**Disclaimer:** This project is for educational purposes only. Do not use the predictions as financial advice.

## Project Structure

- `model_creation.py`: Main script for model training and evaluation.
- `stock_indicators.py`: Module for calculating technical indicators.
- `functions.py`: Module containing utility functions.
- `next_day_prediction.py`: Module for predicting next day's closing price of specified stock 



## Details
This project uses yfinance to download historical stock data <br>

The following technical indicators are used:
- Simple Moving Average (SMA)
- Exponential Moving Average (EMA)
- Relative Strength Index (RSI)
- Moving Average Convergence Divergence (MACD)

The model uses a LSTM structure to train on the data <br>
The model is then evaluated using a percentage test function to assess its performance