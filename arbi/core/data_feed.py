"""
Real-time Market Data Feed

WebSocket-based market data feeds with order book maintenance.
Supports Binance and Kraken with automatic reconnection and gap detection.
"""

import asyncio
import json
import logging
import time
from contextlib import asynccontextmanager
from decimal import Decimal
from typing import AsyncGenerator, Dict, List, Optional, Set, Callable, Any
from urllib.parse import urljoin
import aiohttp
import websockets
from websockets.exceptions import WebSocketException, ConnectionClosedError

from .marketdata import BookDelta, OrderBookLevel, OrderBook, SymbolNormalizer, Trade, Side


# Configure logger
logger = logging.getLogger(__name__)


class ReconnectConfig:
    """Configuration for WebSocket reconnection"""
    def __init__(
        self,
        initial_delay: float = 1.0,
        max_delay: float = 60.0,
        backoff_factor: float = 2.0,
        max_retries: Optional[int] = None
    ):
        self.initial_delay = initial_delay
        self.max_delay = max_delay
        self.backoff_factor = backoff_factor
        self.max_retries = max_retries


class ExchangeConnector:
    """Base class for exchange WebSocket connectors"""
    
    def __init__(
        self,
        exchange_name: str,
        ws_url: str,
        rest_url: str,
        reconnect_config: Optional[ReconnectConfig] = None
    ):
        self.exchange_name = exchange_name
        self.ws_url = ws_url
        self.rest_url = rest_url
        self.reconnect_config = reconnect_config or ReconnectConfig()
        
        self._websocket: Optional[websockets.WebSocketServerProtocol] = None
        self._session: Optional[aiohttp.ClientSession] = None
        self._subscriptions: Set[str] = set()
        self._order_books: Dict[str, OrderBook] = {}
        self._sequence_numbers: Dict[str, int] = {}
        self._running = False
        self._reconnect_attempts = 0
        
    async def __aenter__(self):
        """Async context manager entry"""
        await self.start()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.stop()
    
    async def start(self):
        """Start the connector"""
        self._session = aiohttp.ClientSession()
        self._running = True
        logger.info(f"{self.exchange_name} connector started")
    
    async def stop(self):
        """Stop the connector"""
        self._running = False
        
        if self._websocket:
            await self._websocket.close()
            self._websocket = None
        
        if self._session:
            await self._session.close()
            self._session = None
        
        logger.info(f"{self.exchange_name} connector stopped")
    
    async def _connect_websocket(self) -> websockets.WebSocketServerProtocol:
        """Connect to WebSocket - to be implemented by subclasses"""
        raise NotImplementedError
    
    async def _handle_message(self, message: Dict[str, Any]) -> Optional[BookDelta]:
        """Handle incoming WebSocket message - to be implemented by subclasses"""
        raise NotImplementedError
    
    async def _get_orderbook_snapshot(self, symbol: str) -> Optional[BookDelta]:
        """Get order book snapshot via REST API - to be implemented by subclasses"""
        raise NotImplementedError
    
    def _build_subscription_message(self, symbols: List[str]) -> Dict[str, Any]:
        """Build subscription message - to be implemented by subclasses"""
        raise NotImplementedError
    
    async def _websocket_loop(self, delta_callback: Callable[[BookDelta], None]):
        """Main WebSocket message loop"""
        while self._running:
            try:
                if not self._websocket or self._websocket.closed:
                    await self._reconnect()
                
                if not self._websocket:
                    continue
                
                try:
                    message_str = await asyncio.wait_for(
                        self._websocket.recv(),
                        timeout=30.0  # 30 second timeout
                    )
                    
                    # Reset reconnect attempts on successful message
                    self._reconnect_attempts = 0
                    
                    # Parse and handle message
                    try:
                        message = json.loads(message_str)
                        delta = await self._handle_message(message)
                        
                        if delta:
                            # Update local order book
                            await self._update_order_book(delta)
                            
                            # Call user callback
                            if delta_callback:
                                try:
                                    if asyncio.iscoroutinefunction(delta_callback):
                                        await delta_callback(delta)
                                    else:
                                        delta_callback(delta)
                                except Exception as e:
                                    logger.error(f"Error in delta callback: {e}")
                        
                    except json.JSONDecodeError as e:
                        logger.warning(f"Failed to parse message: {e}")
                    except Exception as e:
                        logger.error(f"Error handling message: {e}")
                
                except asyncio.TimeoutError:
                    logger.warning(f"{self.exchange_name} WebSocket timeout, sending ping")
                    await self._websocket.ping()
                
            except (WebSocketException, ConnectionClosedError) as e:
                logger.warning(f"{self.exchange_name} WebSocket error: {e}")
                await self._handle_disconnect()
            except Exception as e:
                logger.error(f"Unexpected error in {self.exchange_name} WebSocket loop: {e}")
                await asyncio.sleep(1)
    
    async def _reconnect(self):
        """Reconnect to WebSocket with exponential backoff"""
        if (self.reconnect_config.max_retries is not None and 
            self._reconnect_attempts >= self.reconnect_config.max_retries):
            logger.error(f"{self.exchange_name} max reconnection attempts reached")
            self._running = False
            return
        
        delay = min(
            self.reconnect_config.initial_delay * (
                self.reconnect_config.backoff_factor ** self._reconnect_attempts
            ),
            self.reconnect_config.max_delay
        )
        
        logger.info(f"{self.exchange_name} reconnecting in {delay:.1f}s (attempt {self._reconnect_attempts + 1})")
        await asyncio.sleep(delay)
        
        try:
            self._websocket = await self._connect_websocket()
            
            # Re-subscribe to symbols
            if self._subscriptions:
                symbols = list(self._subscriptions)
                await self._subscribe_symbols(symbols)
            
            logger.info(f"{self.exchange_name} WebSocket reconnected")
            
        except Exception as e:
            logger.error(f"{self.exchange_name} reconnection failed: {e}")
            self._reconnect_attempts += 1
    
    async def _handle_disconnect(self):
        """Handle WebSocket disconnect"""
        if self._websocket:
            await self._websocket.close()
            self._websocket = None
        self._reconnect_attempts += 1
    
    async def _subscribe_symbols(self, symbols: List[str]):
        """Subscribe to symbols"""
        if not self._websocket:
            return
        
        message = self._build_subscription_message(symbols)
        await self._websocket.send(json.dumps(message))
        self._subscriptions.update(symbols)
        
        logger.info(f"{self.exchange_name} subscribed to {symbols}")
    
    async def _update_order_book(self, delta: BookDelta):
        """Update local order book with delta"""
        symbol = delta.symbol
        
        # Initialize order book if needed
        if symbol not in self._order_books:
            self._order_books[symbol] = OrderBook(
                exchange=delta.exchange,
                symbol=symbol,
                timestamp=delta.timestamp,
                sequence=delta.sequence
            )
        
        order_book = self._order_books[symbol]
        
        # Check for sequence gap
        if (delta.sequence is not None and 
            order_book.sequence is not None and
            delta.sequence != order_book.sequence + 1 and
            not delta.is_snapshot):
            
            logger.warning(
                f"Sequence gap detected for {symbol}: "
                f"expected {order_book.sequence + 1}, got {delta.sequence}"
            )
            
            # Request snapshot to resync
            await self._resync_order_book(symbol)
            return
        
        # Apply delta to order book
        order_book.apply_delta(delta)
    
    async def _resync_order_book(self, symbol: str):
        """Resync order book via REST API snapshot"""
        try:
            logger.info(f"Resyncing order book for {symbol}")
            snapshot_delta = await self._get_orderbook_snapshot(symbol)
            
            if snapshot_delta:
                # Reset order book with snapshot
                self._order_books[symbol] = OrderBook(
                    exchange=snapshot_delta.exchange,
                    symbol=symbol,
                    timestamp=snapshot_delta.timestamp,
                    sequence=snapshot_delta.sequence
                )
                
                self._order_books[symbol].apply_delta(snapshot_delta)
                logger.info(f"Order book resynced for {symbol}")
            
        except Exception as e:
            logger.error(f"Failed to resync order book for {symbol}: {e}")


class BinanceConnector(ExchangeConnector):
    """Binance WebSocket connector"""
    
    def __init__(self, reconnect_config: Optional[ReconnectConfig] = None):
        super().__init__(
            exchange_name="binance",
            ws_url="wss://stream.binance.com:9443/ws",
            rest_url="https://api.binance.com",
            reconnect_config=reconnect_config
        )
    
    async def _connect_websocket(self) -> websockets.WebSocketServerProtocol:
        """Connect to Binance WebSocket"""
        return await websockets.connect(self.ws_url)
    
    def _build_subscription_message(self, symbols: List[str]) -> Dict[str, Any]:
        """Build Binance subscription message"""
        # Convert symbols to Binance format
        binance_symbols = []
        for symbol in symbols:
            binance_symbol = SymbolNormalizer.denormalize('binance', symbol)
            binance_symbols.append(f"{binance_symbol.lower()}@depth")
        
        return {
            "method": "SUBSCRIBE",
            "params": binance_symbols,
            "id": int(time.time())
        }
    
    async def _handle_message(self, message: Dict[str, Any]) -> Optional[BookDelta]:
        """Handle Binance WebSocket message"""
        # Skip non-data messages
        if "stream" not in message or "data" not in message:
            return None
        
        data = message["data"]
        
        # Parse symbol
        stream = message["stream"]
        if "@depth" not in stream:
            return None
        
        binance_symbol = stream.split("@")[0].upper()
        normalized_symbol = SymbolNormalizer.normalize('binance', binance_symbol)
        
        # Parse order book delta
        bids = []
        asks = []
        
        for bid_data in data.get("b", []):
            price, size = bid_data
            if float(size) > 0:
                bids.append(OrderBookLevel(
                    price=Decimal(price),
                    size=Decimal(size)
                ))
        
        for ask_data in data.get("a", []):
            price, size = ask_data
            if float(size) > 0:
                asks.append(OrderBookLevel(
                    price=Decimal(price),
                    size=Decimal(size)
                ))
        
        return BookDelta(
            exchange="binance",
            symbol=normalized_symbol,
            timestamp=time.time(),
            sequence=data.get("U"),  # First update ID
            bids=bids,
            asks=asks,
            is_snapshot=False
        )
    
    async def _get_orderbook_snapshot(self, symbol: str) -> Optional[BookDelta]:
        """Get Binance order book snapshot"""
        if not self._session:
            return None
        
        try:
            binance_symbol = SymbolNormalizer.denormalize('binance', symbol)
            url = f"{self.rest_url}/api/v3/depth"
            params = {"symbol": binance_symbol, "limit": 1000}
            
            async with self._session.get(url, params=params) as response:
                if response.status != 200:
                    logger.error(f"Failed to get snapshot: {response.status}")
                    return None
                
                data = await response.json()
                
                bids = [
                    OrderBookLevel(price=Decimal(price), size=Decimal(size))
                    for price, size in data["bids"]
                ]
                asks = [
                    OrderBookLevel(price=Decimal(price), size=Decimal(size))
                    for price, size in data["asks"]
                ]
                
                return BookDelta(
                    exchange="binance",
                    symbol=symbol,
                    timestamp=time.time(),
                    sequence=data["lastUpdateId"],
                    bids=bids,
                    asks=asks,
                    is_snapshot=True
                )
        
        except Exception as e:
            logger.error(f"Error getting Binance snapshot: {e}")
            return None


class KrakenConnector(ExchangeConnector):
    """Kraken WebSocket connector"""
    
    def __init__(self, reconnect_config: Optional[ReconnectConfig] = None):
        super().__init__(
            exchange_name="kraken",
            ws_url="wss://ws.kraken.com",
            rest_url="https://api.kraken.com",
            reconnect_config=reconnect_config
        )
    
    async def _connect_websocket(self) -> websockets.WebSocketServerProtocol:
        """Connect to Kraken WebSocket"""
        return await websockets.connect(self.ws_url)
    
    def _build_subscription_message(self, symbols: List[str]) -> Dict[str, Any]:
        """Build Kraken subscription message"""
        # Convert symbols to Kraken format
        kraken_symbols = []
        for symbol in symbols:
            kraken_symbol = SymbolNormalizer.denormalize('kraken', symbol)
            kraken_symbols.append(kraken_symbol)
        
        return {
            "event": "subscribe",
            "pair": kraken_symbols,
            "subscription": {"name": "book", "depth": 1000}
        }
    
    async def _handle_message(self, message: Dict[str, Any]) -> Optional[BookDelta]:
        """Handle Kraken WebSocket message"""
        # Skip non-list messages (status, heartbeat, etc.)
        if not isinstance(message, list) or len(message) < 4:
            return None
        
        # Parse Kraken message format: [channelID, data, channelName, pair]
        channel_data = message[1]
        channel_name = message[2]
        pair = message[3]
        
        if channel_name != "book-1000":
            return None
        
        # Normalize symbol
        normalized_symbol = SymbolNormalizer.normalize('kraken', pair)
        
        # Parse order book data
        bids = []
        asks = []
        
        # Kraken sends updates in different formats
        if "b" in channel_data:  # Bids
            for bid_data in channel_data["b"]:
                price, size, timestamp = bid_data[:3]
                if float(size) > 0:
                    bids.append(OrderBookLevel(
                        price=Decimal(price),
                        size=Decimal(size)
                    ))
        
        if "a" in channel_data:  # Asks
            for ask_data in channel_data["a"]:
                price, size, timestamp = ask_data[:3]
                if float(size) > 0:
                    asks.append(OrderBookLevel(
                        price=Decimal(price),
                        size=Decimal(size)
                    ))
        
        return BookDelta(
            exchange="kraken",
            symbol=normalized_symbol,
            timestamp=time.time(),
            sequence=None,  # Kraken doesn't provide sequence numbers
            bids=bids,
            asks=asks,
            is_snapshot="bs" in channel_data or "as" in channel_data  # Snapshot indicators
        )
    
    async def _get_orderbook_snapshot(self, symbol: str) -> Optional[BookDelta]:
        """Get Kraken order book snapshot"""
        if not self._session:
            return None
        
        try:
            kraken_symbol = SymbolNormalizer.denormalize('kraken', symbol)
            url = f"{self.rest_url}/0/public/Depth"
            params = {"pair": kraken_symbol, "count": 1000}
            
            async with self._session.get(url, params=params) as response:
                if response.status != 200:
                    logger.error(f"Failed to get Kraken snapshot: {response.status}")
                    return None
                
                data = await response.json()
                
                if data.get("error"):
                    logger.error(f"Kraken API error: {data['error']}")
                    return None
                
                result = data["result"]
                pair_data = list(result.values())[0]  # Get first (and only) pair
                
                bids = [
                    OrderBookLevel(price=Decimal(price), size=Decimal(size))
                    for price, size, _ in pair_data["bids"]
                ]
                asks = [
                    OrderBookLevel(price=Decimal(price), size=Decimal(size))
                    for price, size, _ in pair_data["asks"]
                ]
                
                return BookDelta(
                    exchange="kraken",
                    symbol=symbol,
                    timestamp=time.time(),
                    sequence=None,
                    bids=bids,
                    asks=asks,
                    is_snapshot=True
                )
        
        except Exception as e:
            logger.error(f"Error getting Kraken snapshot: {e}")
            return None


class DataFeed:
    """Multi-exchange market data feed manager"""
    
    def __init__(self):
        self.connectors: Dict[str, ExchangeConnector] = {}
        self._running = False
        self._tasks: List[asyncio.Task] = []
    
    def add_connector(self, connector: ExchangeConnector):
        """Add exchange connector"""
        self.connectors[connector.exchange_name] = connector
        logger.info(f"Added {connector.exchange_name} connector")
    
    async def start(self):
        """Start all connectors"""
        self._running = True
        for connector in self.connectors.values():
            await connector.start()
        logger.info("Data feed started")
    
    async def stop(self):
        """Stop all connectors"""
        self._running = False
        
        # Cancel all tasks
        for task in self._tasks:
            if not task.done():
                task.cancel()
        
        # Wait for tasks to complete
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
        
        # Stop connectors
        for connector in self.connectors.values():
            await connector.stop()
        
        logger.info("Data feed stopped")
    
    async def subscribe_orderbook(
        self, 
        symbols: List[str], 
        exchanges: Optional[List[str]] = None
    ) -> AsyncGenerator[BookDelta, None]:
        """
        Subscribe to order book updates for symbols across exchanges.
        
        Args:
            symbols: List of normalized symbol names (e.g., ['BTC/USDT', 'ETH/USDT'])
            exchanges: List of exchange names to subscribe to. If None, uses all available exchanges.
            
        Yields:
            BookDelta objects as they arrive from exchanges
        """
        if not self._running:
            await self.start()
        
        # Determine which exchanges to use
        if exchanges is None:
            exchanges = list(self.connectors.keys())
        
        # Validate exchanges
        for exchange in exchanges:
            if exchange not in self.connectors:
                raise ValueError(f"Exchange '{exchange}' not available")
        
        # Create a queue for delta messages
        delta_queue = asyncio.Queue()
        
        async def delta_callback(delta: BookDelta):
            """Callback to receive deltas from connectors"""
            await delta_queue.put(delta)
        
        # Subscribe to symbols on each exchange
        subscription_tasks = []
        for exchange_name in exchanges:
            connector = self.connectors[exchange_name]
            
            # Subscribe to symbols
            await connector._subscribe_symbols(symbols)
            
            # Start WebSocket loop
            task = asyncio.create_task(
                connector._websocket_loop(delta_callback),
                name=f"{exchange_name}_ws_loop"
            )
            subscription_tasks.append(task)
        
        self._tasks.extend(subscription_tasks)
        
        try:
            # Yield deltas as they arrive
            while self._running:
                try:
                    delta = await asyncio.wait_for(delta_queue.get(), timeout=1.0)
                    yield delta
                except asyncio.TimeoutError:
                    continue  # Allow checking _running flag
                
        except asyncio.CancelledError:
            logger.info("Order book subscription cancelled")
        finally:
            # Clean up tasks
            for task in subscription_tasks:
                if not task.done():
                    task.cancel()


# Convenience function for quick setup
async def create_data_feed(
    exchanges: Optional[List[str]] = None,
    reconnect_config: Optional[ReconnectConfig] = None
) -> DataFeed:
    """
    Create a data feed with specified exchanges.
    
    Args:
        exchanges: List of exchange names to include. Defaults to ['binance', 'kraken']
        reconnect_config: Reconnection configuration
        
    Returns:
        Configured DataFeed instance
    """
    if exchanges is None:
        exchanges = ['binance', 'kraken']
    
    feed = DataFeed()
    
    for exchange in exchanges:
        if exchange == 'binance':
            connector = BinanceConnector(reconnect_config)
        elif exchange == 'kraken':
            connector = KrakenConnector(reconnect_config)
        else:
            logger.warning(f"Unknown exchange: {exchange}")
            continue
        
        feed.add_connector(connector)
    
    return feed


# Example usage
async def main():
    """Example usage of the data feed"""
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create data feed
    feed = await create_data_feed(['binance', 'kraken'])
    
    try:
        # Subscribe to order book updates
        symbols = ['BTC/USDT', 'ETH/USDT']
        
        async for delta in feed.subscribe_orderbook(symbols):
            logger.info(f"Received: {delta}")
            
            # Print best bid/ask if available
            if delta.exchange in feed.connectors:
                connector = feed.connectors[delta.exchange]
                if delta.symbol in connector._order_books:
                    book = connector._order_books[delta.symbol]
                    best_bid = book.get_best_bid()
                    best_ask = book.get_best_ask()
                    
                    if best_bid and best_ask:
                        logger.info(
                            f"{delta.symbol} on {delta.exchange}: "
                            f"Bid: {best_bid.price} @ {best_bid.size}, "
                            f"Ask: {best_ask.price} @ {best_ask.size}"
                        )
    
    except KeyboardInterrupt:
        logger.info("Shutting down...")
    finally:
        await feed.stop()


if __name__ == "__main__":
    asyncio.run(main())
