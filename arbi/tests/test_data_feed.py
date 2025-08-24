"""
Test suite for data feed functionality
"""

import asyncio
import pytest
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch
from arbi.core.data_feed import BinanceConnector, KrakenConnector, DataFeed, create_data_feed
from arbi.core.marketdata import BookDelta, OrderBookLevel


class TestBinanceConnector:
    """Test Binance WebSocket connector"""
    
    @pytest.fixture
    def binance_connector(self):
        """Create Binance connector for testing"""
        return BinanceConnector()
    
    def test_build_subscription_message(self, binance_connector):
        """Test Binance subscription message format"""
        symbols = ['BTC/USDT', 'ETH/USDT']
        message = binance_connector._build_subscription_message(symbols)
        
        assert message['method'] == 'SUBSCRIBE'
        assert 'btcusdt@depth' in message['params']
        assert 'ethusdt@depth' in message['params']
        assert 'id' in message
    
    @pytest.mark.asyncio
    async def test_handle_message(self, binance_connector):
        """Test Binance message parsing"""
        # Mock Binance WebSocket message
        mock_message = {
            "stream": "btcusdt@depth",
            "data": {
                "U": 12345,
                "b": [["50000.00", "1.5"], ["49999.00", "2.0"]],
                "a": [["50001.00", "1.0"], ["50002.00", "0.5"]]
            }
        }
        
        delta = await binance_connector._handle_message(mock_message)
        
        assert delta is not None
        assert delta.exchange == "binance"
        assert delta.symbol == "BTC/USDT"
        assert delta.sequence == 12345
        assert len(delta.bids) == 2
        assert len(delta.asks) == 2
        assert delta.bids[0].price == Decimal("50000.00")
        assert delta.bids[0].size == Decimal("1.5")


class TestKrakenConnector:
    """Test Kraken WebSocket connector"""
    
    @pytest.fixture
    def kraken_connector(self):
        """Create Kraken connector for testing"""
        return KrakenConnector()
    
    def test_build_subscription_message(self, kraken_connector):
        """Test Kraken subscription message format"""
        symbols = ['BTC/USD', 'ETH/USD']
        message = kraken_connector._build_subscription_message(symbols)
        
        assert message['event'] == 'subscribe'
        assert 'BTC/USD' in message['pair']
        assert 'ETH/USD' in message['pair']
        assert message['subscription']['name'] == 'book'
    
    @pytest.mark.asyncio
    async def test_handle_message(self, kraken_connector):
        """Test Kraken message parsing"""
        # Mock Kraken WebSocket message
        mock_message = [
            1234,  # Channel ID
            {
                "b": [["50000.00", "1.5", "1234567890.123"]],
                "a": [["50001.00", "1.0", "1234567890.124"]]
            },
            "book-1000",
            "XBT/USD"
        ]
        
        delta = await kraken_connector._handle_message(mock_message)
        
        assert delta is not None
        assert delta.exchange == "kraken"
        assert delta.symbol == "BTC/USD"
        assert len(delta.bids) == 1
        assert len(delta.asks) == 1
        assert delta.bids[0].price == Decimal("50000.00")


class TestDataFeed:
    """Test DataFeed orchestration"""
    
    @pytest.fixture
    def data_feed(self):
        """Create DataFeed for testing"""
        return DataFeed()
    
    def test_add_connector(self, data_feed):
        """Test adding connectors to data feed"""
        connector = BinanceConnector()
        data_feed.add_connector(connector)
        
        assert "binance" in data_feed.connectors
        assert data_feed.connectors["binance"] == connector
    
    @pytest.mark.asyncio
    async def test_create_data_feed(self):
        """Test data feed creation helper"""
        feed = await create_data_feed(['binance'])
        
        assert isinstance(feed, DataFeed)
        assert "binance" in feed.connectors
        assert isinstance(feed.connectors["binance"], BinanceConnector)


class TestOrderBookManagement:
    """Test order book state management"""
    
    @pytest.fixture
    def sample_delta(self):
        """Create sample BookDelta for testing"""
        return BookDelta(
            exchange="test",
            symbol="BTC/USDT",
            sequence=1,
            bids=[
                OrderBookLevel(price=Decimal("50000"), size=Decimal("1.0")),
                OrderBookLevel(price=Decimal("49999"), size=Decimal("2.0"))
            ],
            asks=[
                OrderBookLevel(price=Decimal("50001"), size=Decimal("1.5")),
                OrderBookLevel(price=Decimal("50002"), size=Decimal("0.5"))
            ]
        )
    
    @pytest.mark.asyncio
    async def test_order_book_update(self, sample_delta):
        """Test order book update logic"""
        connector = BinanceConnector()
        
        # Apply delta
        await connector._update_order_book(sample_delta)
        
        # Check order book state
        assert "BTC/USDT" in connector._order_books
        order_book = connector._order_books["BTC/USDT"]
        
        assert order_book.sequence == 1
        assert len(order_book.bids) == 2
        assert len(order_book.asks) == 2
        
        best_bid = order_book.get_best_bid()
        best_ask = order_book.get_best_ask()
        
        assert best_bid.price == Decimal("50000")
        assert best_ask.price == Decimal("50001")
    
    @pytest.mark.asyncio
    async def test_sequence_gap_detection(self, sample_delta):
        """Test sequence gap detection and resync"""
        connector = BinanceConnector()
        
        # First update
        await connector._update_order_book(sample_delta)
        
        # Create delta with gap
        gap_delta = BookDelta(
            exchange="test",
            symbol="BTC/USDT",
            sequence=5,  # Gap: expected 2, got 5
            bids=[OrderBookLevel(price=Decimal("49998"), size=Decimal("1.0"))],
            asks=[]
        )
        
        with patch.object(connector, '_resync_order_book', new_callable=AsyncMock) as mock_resync:
            await connector._update_order_book(gap_delta)
            mock_resync.assert_called_once_with("BTC/USDT")


# Integration test
@pytest.mark.asyncio
async def test_data_feed_integration():
    """Integration test for data feed with mock WebSocket"""
    received_deltas = []
    
    async def delta_handler(delta: BookDelta):
        received_deltas.append(delta)
    
    # This would require more complex mocking of WebSocket connections
    # For now, we'll test the setup and configuration
    feed = await create_data_feed(['binance'])
    
    assert isinstance(feed, DataFeed)
    assert len(feed.connectors) == 1
    
    # Clean up
    await feed.stop()


if __name__ == "__main__":
    # Run basic tests
    pytest.main([__file__, "-v"])
