# Financial AI Core

![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
[![Code Style](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

> Advanced machine learning for financial markets analysis and algorithmic trading

## Features

- **Market Prediction**: LSTM/Transformer models for price forecasting
- **Portfolio Optimization**: Mean-Variance and Black-Litterman implementations
- **Risk Analysis**: VaR (Value at Risk) and CVaR calculations
- **Backtesting Engine**: Event-driven backtesting framework
- **Data Pipeline**: Automated ETL for market data

## Quick Start

### Prerequisites
- Python 3.8+
- TA-Lib (see [installation guide](https://github.com/mrjbq7/ta-lib#installation))
- Bloomberg Terminal (optional for alternative data)

### Installation
```bash
git clone git@github.com:BuRcapital/Financial-ai-core.git
cd Financial-ai-core
pip install -r requirements.txt

# Financial AI Core

![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
[![Code Style](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

> Advanced machine learning for financial markets analysis and algorithmic trading

## Features

- **Market Prediction**: LSTM/Transformer models for price forecasting
- **Portfolio Optimization**: Mean-Variance and Black-Litterman implementations
- **Risk Analysis**: VaR (Value at Risk) and CVaR calculations
- **Backtesting Engine**: Event-driven backtesting framework
- **Data Pipeline**: Automated ETL for market data

## Quick Start

### Prerequisites
- Python 3.8+
- TA-Lib (see [installation guide](https://github.com/mrjbq7/ta-lib#installation))
- Bloomberg Terminal (optional for alternative data)

### Installation
```bash
git clone git@github.com:BuRcapital/Financial-ai-core.git
cd Financial-ai-core
pip install -r requirements.txt

from core.prediction import LSTMMarketPredictor

predictor = LSTMMarketPredictor(ticker='SPY')
forecast = predictor.predict_next_window()
print(f"Next 5-day forecast: {forecast}")
