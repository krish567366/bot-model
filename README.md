# Crypto Arbitrage Trading Platform

Production-grade arbitrage trading system in Python 3.11 with real-time market data, signal generation, and automated execution.

## Features

- **Real-time Market Data**: WebSocket feeds from Binance, Kraken, Coinbase, Bybit
- **Arbitrage Signals**: Cross-exchange and triangular arbitrage detection
- **Smart Execution**: Atomic multi-leg orders with SOR (Smart Order Router)
- **Risk Management**: Position limits, circuit breakers, audit logging
- **Backtesting**: Event-driven simulation with realistic latency/slippage models
- **Monitoring**: Streamlit dashboard with P&L tracking and performance metrics
- **API**: FastAPI REST + WebSocket endpoints

## Architecture

```txt
/arbi/
├── core/
│   ├── data_feed.py       # WebSocket market data feeds
│   ├── marketdata.py      # Data models and schemas
│   ├── signal.py          # Arbitrage signal generation
│   ├── execution.py       # Smart order router
│   ├── risk.py            # Risk management and limits
│   ├── portfolio.py       # Portfolio tracking and P&L
│   ├── backtest.py        # Backtesting engine
│   └── storage.py         # Data storage (Parquet + SQLite)
├── api/
│   └── server.py          # FastAPI REST + WebSocket API
├── ui/
│   └── dashboard.py       # Streamlit monitoring dashboard
├── config/
│   └── settings.py        # Configuration management
└── tests/                 # Unit and integration tests
```

## Quick Start

1. **Install Dependencies**:

```bash
pip install -r requirements.txt
```

### 2. **Configure Environment**

```bash
cp .env.example .env
# Edit .env with your API keys and settings
```

### 3. **Run Market Data Feed**

```bash
python -m arbi.core.data_feed
```

#### 4. **Start API Server**

```bash
python -m arbi.api.server
```

#### 5. **Launch Dashboard**

```bash
streamlit run arbi/ui/dashboard.py
```

## Configuration

All configuration is managed through environment variables and `arbi/config/settings.py`:

- **API Keys**: Exchange API credentials
- **Risk Limits**: Position sizes, daily loss limits
- **Strategy Parameters**: Signal thresholds, execution delays
- **Monitoring**: Slack/Telegram webhooks

## Development

### Running Tests

```bash
pytest tests/ -v --cov=arbi
```

### Code Quality

```bash
black arbi/
flake8 arbi/
mypy arbi/
```

### Docker Deployment

```bash
docker-compose up -d
```

## License

MIT License - see LICENSE file for details.
