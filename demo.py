#!/usr/bin/env python3
"""
Demo script for the crypto arbitrage data feed

This script demonstrates the real-time order book data feed functionality
by connecting to Binance and Kraken WebSocket streams and displaying
live market data for BTC/USDT and ETH/USDT pairs.
"""

import asyncio
import logging
import signal
import sys
from datetime import datetime
from decimal import Decimal
from typing import Dict, Optional

from arbi.core.data_feed import create_data_feed, DataFeed
from arbi.core.marketdata import BookDelta, OrderBook
from arbi.config.settings import get_settings


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('demo.log')
    ]
)

logger = logging.getLogger(__name__)


class MarketDataMonitor:
    """Monitor and display market data in real-time"""
    
    def __init__(self):
        self.data_feed: Optional[DataFeed] = None
        self.order_books: Dict[str, Dict[str, OrderBook]] = {}
        self.update_count = 0
        self.start_time = datetime.now()
        self.running = True
    
    async def start(self):
        """Start the market data monitor"""
        logger.info("Starting Crypto Arbitrage Data Feed Demo")
        
        # Load settings
        settings = get_settings()
        
        # Create data feed with enabled exchanges
        exchanges = [ex for ex in settings.enabled_exchanges if ex in ['binance', 'kraken']]
        if not exchanges:
            exchanges = ['binance', 'kraken']  # Default fallback
        
        logger.info(f"Connecting to exchanges: {exchanges}")
        self.data_feed = await create_data_feed(exchanges)
        
        # Trading symbols to monitor
        symbols = ['BTC/USDT', 'ETH/USDT']
        logger.info(f"Monitoring symbols: {symbols}")
        
        try:
            # Subscribe to order book updates
            async for delta in self.data_feed.subscribe_orderbook(symbols):
                await self.handle_delta(delta)
                
                # Print periodic updates
                if self.update_count % 10 == 0:
                    await self.print_status()
                
                if not self.running:
                    break
                    
        except KeyboardInterrupt:
            logger.info("Received interrupt signal")
        except Exception as e:
            logger.error(f"Error in data feed: {e}")
        finally:
            await self.stop()
    
    async def handle_delta(self, delta: BookDelta):
        """Handle incoming order book delta"""
        self.update_count += 1
        
        # Initialize exchange dict if needed
        if delta.exchange not in self.order_books:
            self.order_books[delta.exchange] = {}
        
        # Initialize order book if needed
        if delta.symbol not in self.order_books[delta.exchange]:
            self.order_books[delta.exchange][delta.symbol] = OrderBook(
                exchange=delta.exchange,
                symbol=delta.symbol,
                timestamp=delta.timestamp
            )
        
        # Update order book
        order_book = self.order_books[delta.exchange][delta.symbol]
        order_book.apply_delta(delta)
        
        # Log significant updates
        if delta.is_snapshot:
            logger.info(f"📸 Snapshot received for {delta.symbol} on {delta.exchange}")
        
        # Check for large spreads
        best_bid = order_book.get_best_bid()
        best_ask = order_book.get_best_ask()
        
        if best_bid and best_ask:
            spread_pct = ((best_ask.price - best_bid.price) / best_bid.price) * 100
            
            if spread_pct > Decimal('1.0'):  # Spread > 1%
                logger.warning(
                    f"🚨 Large spread detected: {delta.symbol} on {delta.exchange} "
                    f"= {spread_pct:.2f}%"
                )
    
    async def print_status(self):
        """Print current status and market data"""
        uptime = datetime.now() - self.start_time
        
        print(f"\n{'='*80}")
        print(f"📊 CRYPTO ARBITRAGE FEED - Updates: {self.update_count:,} | "
              f"Uptime: {uptime}")
        print(f"{'='*80}")
        
        # Print order book status for each exchange/symbol
        for exchange, symbols in self.order_books.items():
            print(f"\n🏦 {exchange.upper()}")
            print(f"{'-'*40}")
            
            for symbol, order_book in symbols.items():
                best_bid = order_book.get_best_bid()
                best_ask = order_book.get_best_ask()
                mid_price = order_book.get_mid_price()
                spread = order_book.get_spread()
                
                if best_bid and best_ask and mid_price and spread:
                    spread_pct = (spread / mid_price) * 100
                    
                    print(f"  💰 {symbol}")
                    print(f"    Bid: ${best_bid.price:>10} @ {best_bid.size:>8}")
                    print(f"    Ask: ${best_ask.price:>10} @ {best_ask.size:>8}")
                    print(f"    Mid: ${mid_price:>10} | Spread: {spread_pct:.3f}%")
                    print(f"    Last: {order_book.timestamp.strftime('%H:%M:%S')}")
                else:
                    print(f"  💰 {symbol} - Waiting for data...")
        
        # Calculate arbitrage opportunities
        await self.check_arbitrage_opportunities()
    
    async def check_arbitrage_opportunities(self):
        """Check for arbitrage opportunities between exchanges"""
        print(f"\n🔍 ARBITRAGE OPPORTUNITIES")
        print(f"{'-'*40}")
        
        # Check each symbol across exchanges
        symbols_to_check = set()
        for symbols in self.order_books.values():
            symbols_to_check.update(symbols.keys())
        
        for symbol in symbols_to_check:
            exchange_prices = {}
            
            # Collect prices from each exchange
            for exchange, symbols in self.order_books.items():
                if symbol in symbols:
                    order_book = symbols[symbol]
                    mid_price = order_book.get_mid_price()
                    if mid_price:
                        exchange_prices[exchange] = mid_price
            
            # Find arbitrage opportunities
            if len(exchange_prices) >= 2:
                prices = list(exchange_prices.values())
                min_price = min(prices)
                max_price = max(prices)
                
                if min_price > 0:
                    arb_pct = ((max_price - min_price) / min_price) * 100
                    
                    if arb_pct > Decimal('0.1'):  # > 0.1% arbitrage
                        min_exchange = [ex for ex, p in exchange_prices.items() if p == min_price][0]
                        max_exchange = [ex for ex, p in exchange_prices.items() if p == max_price][0]
                        
                        print(f"  ⚡ {symbol}: {arb_pct:.3f}% opportunity")
                        print(f"    Buy  {min_exchange}: ${min_price}")
                        print(f"    Sell {max_exchange}: ${max_price}")
                        print(f"    Profit: ${max_price - min_price}")
    
    async def stop(self):
        """Stop the monitor"""
        self.running = False
        if self.data_feed:
            await self.data_feed.stop()
        logger.info("Market data monitor stopped")


def signal_handler(signum, frame):
    """Handle interrupt signals"""
    print(f"\n🛑 Received signal {signum}, shutting down gracefully...")
    sys.exit(0)


async def main():
    """Main demo function"""
    # Set up signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Create and start monitor
    monitor = MarketDataMonitor()
    
    try:
        await monitor.start()
    except KeyboardInterrupt:
        logger.info("Demo stopped by user")
    except Exception as e:
        logger.error(f"Demo error: {e}")
    finally:
        await monitor.stop()


if __name__ == "__main__":
    print("""
    🚀 CRYPTO ARBITRAGE DATA FEED DEMO
    ================================
    
    This demo connects to live cryptocurrency exchange WebSocket feeds
    and displays real-time order book data and arbitrage opportunities.
    
    Exchanges: Binance, Kraken
    Symbols: BTC/USDT, ETH/USDT
    
    Press Ctrl+C to stop.
    """)
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Demo stopped. Thanks for trying the Crypto Arbitrage Platform!")
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        sys.exit(1)
