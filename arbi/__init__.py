"""
Crypto Arbitrage Trading Platform

A production-grade arbitrage trading system with real-time market data,
signal generation, execution, and risk management.
"""

__version__ = "1.0.0"
__author__ = "Arbitrage Trading Team"

# Export main components
from .core.marketdata import BookDelta, Trade, Ticker, OrderBook
from .core.data_feed import DataFeed, create_data_feed
from .config.settings import get_settings, Settings

__all__ = [
    "BookDelta",
    "Trade", 
    "Ticker",
    "OrderBook",
    "DataFeed",
    "create_data_feed",
    "get_settings",
    "Settings"
]
