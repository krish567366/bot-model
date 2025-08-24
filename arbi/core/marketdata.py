"""
Market Data Models and Schemas

Defines normalized data structures for market data across exchanges.
All models use pydantic for validation and type safety.
"""

from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Dict, List, Optional, Union
from pydantic import BaseModel, Field, field_validator
import time


class Side(str, Enum):
    """Order side enumeration"""
    BID = "bid"
    ASK = "ask"
    BUY = "buy" 
    SELL = "sell"


class OrderBookLevel(BaseModel):
    """Single level in order book"""
    price: Decimal = Field(..., description="Price level")
    size: Decimal = Field(..., description="Size at this level")
    count: Optional[int] = Field(None, description="Number of orders (if available)")
    
    class Config:
        json_encoders = {
            Decimal: str
        }


class BookDelta(BaseModel):
    """Normalized order book delta/update"""
    exchange: str = Field(..., description="Exchange identifier")
    symbol: str = Field(..., description="Trading pair symbol (normalized)")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    sequence: Optional[int] = Field(None, description="Sequence number for gap detection")
    
    # Changes to order book levels
    bids: List[OrderBookLevel] = Field(default_factory=list, description="Bid level changes")
    asks: List[OrderBookLevel] = Field(default_factory=list, description="Ask level changes")
    
    # Metadata
    is_snapshot: bool = Field(False, description="True if this is a full snapshot")
    latency_ms: Optional[float] = Field(None, description="Processing latency in milliseconds")
    
    @field_validator('timestamp', mode='before')
    def parse_timestamp(cls, v):
        if isinstance(v, (int, float)):
            return datetime.utcfromtimestamp(v)
        return v
    
    def __str__(self) -> str:
        return f"BookDelta({self.exchange}:{self.symbol} seq={self.sequence} bids={len(self.bids)} asks={len(self.asks)})"


class Trade(BaseModel):
    """Normalized trade data"""
    exchange: str = Field(..., description="Exchange identifier")
    symbol: str = Field(..., description="Trading pair symbol (normalized)")
    trade_id: str = Field(..., description="Exchange-specific trade ID")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    price: Decimal = Field(..., description="Trade price")
    size: Decimal = Field(..., description="Trade size/quantity")
    side: Side = Field(..., description="Aggressor side")
    
    # Optional fields
    buyer_maker: Optional[bool] = Field(None, description="True if buyer was maker")
    latency_ms: Optional[float] = Field(None, description="Processing latency in milliseconds")
    
    class Config:
        json_encoders = {
            Decimal: str
        }
    
    @field_validator('timestamp', mode='before')
    def parse_timestamp(cls, v):
        if isinstance(v, (int, float)):
            return datetime.utcfromtimestamp(v)
        return v


class Ticker(BaseModel):
    """24hr ticker statistics"""
    exchange: str = Field(..., description="Exchange identifier")
    symbol: str = Field(..., description="Trading pair symbol (normalized)")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    # Price data
    last_price: Optional[Decimal] = Field(None, description="Last trade price")
    bid: Optional[Decimal] = Field(None, description="Best bid price")
    ask: Optional[Decimal] = Field(None, description="Best ask price")
    
    # Volume data
    volume_24h: Optional[Decimal] = Field(None, description="24h volume in base currency")
    quote_volume_24h: Optional[Decimal] = Field(None, description="24h volume in quote currency")
    
    # 24h stats
    high_24h: Optional[Decimal] = Field(None, description="24h high price")
    low_24h: Optional[Decimal] = Field(None, description="24h low price")
    open_24h: Optional[Decimal] = Field(None, description="24h open price")
    change_24h: Optional[Decimal] = Field(None, description="24h price change")
    change_pct_24h: Optional[Decimal] = Field(None, description="24h price change percentage")
    
    # Count
    count_24h: Optional[int] = Field(None, description="24h trade count")
    
    class Config:
        json_encoders = {
            Decimal: str
        }
    
    @field_validator('timestamp', mode='before')
    def parse_timestamp(cls, v):
        if isinstance(v, (int, float)):
            return datetime.utcfromtimestamp(v)
        return v
    
    @property
    def mid_price(self) -> Optional[Decimal]:
        """Calculate mid price from bid/ask"""
        if self.bid and self.ask:
            return (self.bid + self.ask) / 2
        return None
    
    @property
    def spread(self) -> Optional[Decimal]:
        """Calculate bid-ask spread"""
        if self.bid and self.ask:
            return self.ask - self.bid
        return None
    
    @property
    def spread_pct(self) -> Optional[Decimal]:
        """Calculate bid-ask spread as percentage of mid"""
        mid = self.mid_price
        spread = self.spread
        if mid and spread and mid > 0:
            return (spread / mid) * 100
        return None


class OrderBook(BaseModel):
    """Full order book state"""
    exchange: str = Field(..., description="Exchange identifier")
    symbol: str = Field(..., description="Trading pair symbol (normalized)")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    sequence: Optional[int] = Field(None, description="Current sequence number")
    
    bids: Dict[str, OrderBookLevel] = Field(default_factory=dict, description="Bid levels keyed by price")
    asks: Dict[str, OrderBookLevel] = Field(default_factory=dict, description="Ask levels keyed by price")
    
    def apply_delta(self, delta: BookDelta) -> None:
        """Apply a delta to update the order book state"""
        # Update sequence if provided
        if delta.sequence is not None:
            self.sequence = delta.sequence
        
        # Update timestamp
        self.timestamp = delta.timestamp
        
        # Apply bid changes
        for level in delta.bids:
            price_str = str(level.price)
            if level.size == 0:
                # Remove level
                self.bids.pop(price_str, None)
            else:
                # Update level
                self.bids[price_str] = level
        
        # Apply ask changes  
        for level in delta.asks:
            price_str = str(level.price)
            if level.size == 0:
                # Remove level
                self.asks.pop(price_str, None)
            else:
                # Update level
                self.asks[price_str] = level
    
    def get_best_bid(self) -> Optional[OrderBookLevel]:
        """Get best bid level"""
        if not self.bids:
            return None
        best_price = max(Decimal(price) for price in self.bids.keys())
        return self.bids[str(best_price)]
    
    def get_best_ask(self) -> Optional[OrderBookLevel]:
        """Get best ask level"""
        if not self.asks:
            return None
        best_price = min(Decimal(price) for price in self.asks.keys())
        return self.asks[str(best_price)]
    
    def get_mid_price(self) -> Optional[Decimal]:
        """Calculate mid price"""
        best_bid = self.get_best_bid()
        best_ask = self.get_best_ask()
        if best_bid and best_ask:
            return (best_bid.price + best_ask.price) / 2
        return None
    
    def get_spread(self) -> Optional[Decimal]:
        """Calculate bid-ask spread"""
        best_bid = self.get_best_bid()
        best_ask = self.get_best_ask()
        if best_bid and best_ask:
            return best_ask.price - best_bid.price
        return None
    
    def get_levels(self, side: Side, depth: int = 10) -> List[OrderBookLevel]:
        """Get top N levels for a side"""
        if side in (Side.BID, Side.BUY):
            # Sort bids descending (highest first)
            sorted_prices = sorted(self.bids.keys(), key=Decimal, reverse=True)
            return [self.bids[price] for price in sorted_prices[:depth]]
        else:
            # Sort asks ascending (lowest first)  
            sorted_prices = sorted(self.asks.keys(), key=Decimal)
            return [self.asks[price] for price in sorted_prices[:depth]]


# Symbol normalization utilities
class SymbolNormalizer:
    """Normalize symbols across exchanges"""
    
    # Exchange-specific symbol mappings
    SYMBOL_MAPPINGS = {
        'binance': {
            'BTCUSDT': 'BTC/USDT',
            'ETHUSDT': 'ETH/USDT',
            'BNBUSDT': 'BNB/USDT',
        },
        'kraken': {
            'XXBTZUSD': 'BTC/USD',
            'XETHZUSD': 'ETH/USD',
            'XETHXXBT': 'ETH/BTC',
        }
    }
    
    @classmethod
    def normalize(cls, exchange: str, symbol: str) -> str:
        """Normalize exchange-specific symbol to standard format"""
        exchange_lower = exchange.lower()
        if exchange_lower in cls.SYMBOL_MAPPINGS:
            return cls.SYMBOL_MAPPINGS[exchange_lower].get(symbol, symbol)
        return symbol
    
    @classmethod
    def denormalize(cls, exchange: str, normalized_symbol: str) -> str:
        """Convert normalized symbol back to exchange format"""
        exchange_lower = exchange.lower()
        if exchange_lower in cls.SYMBOL_MAPPINGS:
            # Reverse lookup
            for native, norm in cls.SYMBOL_MAPPINGS[exchange_lower].items():
                if norm == normalized_symbol:
                    return native
        return normalized_symbol
