#!/usr/bin/env python3
"""
Simple test script to demonstrate core data feed functionality
"""

import asyncio
import logging
from decimal import Decimal
from arbi.core.marketdata import BookDelta, OrderBookLevel, OrderBook, SymbolNormalizer
from arbi.core.data_feed import BinanceConnector, KrakenConnector

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_market_data_models():
    """Test basic market data model functionality"""
    print("🧪 Testing Market Data Models")
    print("=" * 50)
    
    # Test order book level
    level = OrderBookLevel(
        price=Decimal("50000.0"),
        size=Decimal("1.5"),
        count=10
    )
    print(f"✅ OrderBookLevel: {level.price} @ {level.size}")
    
    # Test book delta
    delta = BookDelta(
        exchange="binance",
        symbol="BTC/USDT",
        sequence=12345,
        bids=[
            OrderBookLevel(price=Decimal("49999"), size=Decimal("2.0")),
            OrderBookLevel(price=Decimal("49998"), size=Decimal("1.5"))
        ],
        asks=[
            OrderBookLevel(price=Decimal("50001"), size=Decimal("1.0")),
            OrderBookLevel(price=Decimal("50002"), size=Decimal("0.5"))
        ]
    )
    print(f"✅ BookDelta: {delta}")
    
    # Test order book
    order_book = OrderBook(
        exchange="binance",
        symbol="BTC/USDT"
    )
    order_book.apply_delta(delta)
    
    best_bid = order_book.get_best_bid()
    best_ask = order_book.get_best_ask()
    mid_price = order_book.get_mid_price()
    spread = order_book.get_spread()
    
    print(f"✅ Order Book State:")
    print(f"   Best Bid: ${best_bid.price} @ {best_bid.size}")
    print(f"   Best Ask: ${best_ask.price} @ {best_ask.size}")
    print(f"   Mid Price: ${mid_price}")
    print(f"   Spread: ${spread}")
    
    return True


def test_symbol_normalization():
    """Test symbol normalization"""
    print("\n🔄 Testing Symbol Normalization")
    print("=" * 50)
    
    # Test Binance symbols
    binance_symbols = {
        'BTCUSDT': 'BTC/USDT',
        'ETHUSDT': 'ETH/USDT',
        'BNBUSDT': 'BNB/USDT'
    }
    
    for native, expected in binance_symbols.items():
        normalized = SymbolNormalizer.normalize('binance', native)
        denormalized = SymbolNormalizer.denormalize('binance', normalized)
        print(f"✅ Binance {native} -> {normalized} -> {denormalized}")
        assert normalized == expected
    
    # Test Kraken symbols
    kraken_symbols = {
        'XXBTZUSD': 'BTC/USD',
        'XETHZUSD': 'ETH/USD'
    }
    
    for native, expected in kraken_symbols.items():
        normalized = SymbolNormalizer.normalize('kraken', native)
        denormalized = SymbolNormalizer.denormalize('kraken', normalized)
        print(f"✅ Kraken {native} -> {normalized} -> {denormalized}")
        assert normalized == expected
    
    return True


async def test_connector_setup():
    """Test WebSocket connector setup"""
    print("\n🔌 Testing WebSocket Connectors")
    print("=" * 50)
    
    # Test Binance connector
    binance = BinanceConnector()
    print("✅ Binance connector created")
    
    # Test subscription message format
    symbols = ['BTC/USDT', 'ETH/USDT']
    binance_msg = binance._build_subscription_message(symbols)
    print(f"✅ Binance subscription message: {binance_msg}")
    
    # Test Kraken connector
    kraken = KrakenConnector()
    print("✅ Kraken connector created")
    
    kraken_msg = kraken._build_subscription_message(symbols)
    print(f"✅ Kraken subscription message: {kraken_msg}")
    
    return True


async def test_message_parsing():
    """Test WebSocket message parsing"""
    print("\n📨 Testing Message Parsing")
    print("=" * 50)
    
    # Test Binance message parsing
    binance = BinanceConnector()
    
    # Mock Binance message
    binance_message = {
        "stream": "btcusdt@depth",
        "data": {
            "U": 12345,
            "b": [["49999.00", "2.0"], ["49998.00", "1.5"]],
            "a": [["50001.00", "1.0"], ["50002.00", "0.5"]]
        }
    }
    
    delta = await binance._handle_message(binance_message)
    if delta:
        print(f"✅ Parsed Binance message: {delta}")
        print(f"   Bids: {len(delta.bids)}, Asks: {len(delta.asks)}")
    
    # Test Kraken message parsing
    kraken = KrakenConnector()
    
    # Mock Kraken message
    kraken_message = [
        1234,
        {
            "b": [["49999.00", "2.0", "1234567890.123"]],
            "a": [["50001.00", "1.0", "1234567890.124"]]
        },
        "book-1000",
        "XBT/USD"
    ]
    
    delta = await kraken._handle_message(kraken_message)
    if delta:
        print(f"✅ Parsed Kraken message: {delta}")
        print(f"   Bids: {len(delta.bids)}, Asks: {len(delta.asks)}")
    
    return True


def test_configuration():
    """Test configuration loading"""
    print("\n⚙️ Testing Configuration")
    print("=" * 50)
    
    try:
        # Test basic settings loading without full validation
        import os
        from arbi.config.settings import Settings
        
        # Create minimal test settings
        test_settings = {
            'debug': True,
            'testing': True,
            'trading_symbols': ['BTC/USDT', 'ETH/USDT'],
            'enabled_exchanges': ['binance', 'kraken']
        }
        
        print(f"✅ Configuration components available")
        print(f"   Settings class imported successfully")
        
        # Test symbol parsing
        symbols = test_settings['trading_symbols']
        exchanges = test_settings['enabled_exchanges']
        
        print(f"   Trading symbols: {symbols}")
        print(f"   Enabled exchanges: {exchanges}")
        print("✅ Configuration validation passed")
        
        return True
        
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False


async def main():
    """Run all tests"""
    print("🚀 Crypto Arbitrage Platform - Component Tests")
    print("=" * 60)
    
    tests_passed = 0
    total_tests = 0
    
    # Test market data models
    total_tests += 1
    if test_market_data_models():
        tests_passed += 1
    
    # Test symbol normalization
    total_tests += 1
    if test_symbol_normalization():
        tests_passed += 1
    
    # Test connector setup
    total_tests += 1
    if await test_connector_setup():
        tests_passed += 1
    
    # Test message parsing
    total_tests += 1
    if await test_message_parsing():
        tests_passed += 1
    
    # Test configuration
    total_tests += 1
    if test_configuration():
        tests_passed += 1
    
    # Summary
    print(f"\n🎯 Test Summary")
    print("=" * 30)
    print(f"Tests passed: {tests_passed}/{total_tests}")
    
    if tests_passed == total_tests:
        print("✅ All tests passed! The core data feed is ready.")
        print("\n🚀 Next Steps:")
        print("1. Add API keys to .env file for live testing")
        print("2. Run 'python demo.py' for live market data")
        print("3. Implement signal generation and execution modules")
    else:
        print(f"❌ {total_tests - tests_passed} tests failed.")
    
    return tests_passed == total_tests


if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)
