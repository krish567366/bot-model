"""
Feature Engineering Module

Transforms raw market data into ML-ready features for model training and inference.
Includes technical indicators, order book features, microstructure signals, and sentiment analysis.
"""

import asyncio
import logging
import time
import numpy as np
import pandas as pd
from decimal import Decimal
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum
from collections import deque
import talib
from scipy import stats
from sklearn.preprocessing import StandardScaler, RobustScaler

from ..core.marketdata import BookDelta, OrderBook, Trade, Ticker
from ..config.settings import get_settings

logger = logging.getLogger(__name__)


class FeatureType(str, Enum):
    """Types of features"""
    PRICE = "price"
    VOLUME = "volume"
    ORDERBOOK = "orderbook"
    MICROSTRUCTURE = "microstructure"
    TECHNICAL = "technical"
    SENTIMENT = "sentiment"
    MACRO = "macro"


@dataclass
class Feature:
    """Individual feature definition"""
    name: str
    value: float
    feature_type: FeatureType
    timestamp: float
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FeatureSet:
    """Collection of features for a symbol at a point in time"""
    symbol: str
    timestamp: float
    features: Dict[str, Feature]
    target: Optional[float] = None  # For supervised learning
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary for ML models"""
        return {name: feature.value for name, feature in self.features.items()}
    
    def to_array(self, feature_names: List[str]) -> np.ndarray:
        """Convert to numpy array with specified feature order"""
        return np.array([self.features.get(name, Feature(name, 0.0, FeatureType.PRICE, 0.0)).value 
                        for name in feature_names])


class TechnicalIndicators:
    """Technical indicator calculations using TA-Lib"""
    
    @staticmethod
    def calculate_sma(prices: np.ndarray, period: int) -> np.ndarray:
        """Simple Moving Average"""
        return talib.SMA(prices, timeperiod=period)
    
    @staticmethod
    def calculate_ema(prices: np.ndarray, period: int) -> np.ndarray:
        """Exponential Moving Average"""
        return talib.EMA(prices, timeperiod=period)
    
    @staticmethod
    def calculate_rsi(prices: np.ndarray, period: int = 14) -> np.ndarray:
        """Relative Strength Index"""
        return talib.RSI(prices, timeperiod=period)
    
    @staticmethod
    def calculate_macd(prices: np.ndarray, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """MACD indicator"""
        return talib.MACD(prices, fastperiod=fast, slowperiod=slow, signalperiod=signal)
    
    @staticmethod
    def calculate_bollinger_bands(prices: np.ndarray, period: int = 20, std_dev: int = 2) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Bollinger Bands"""
        return talib.BBANDS(prices, timeperiod=period, nbdevup=std_dev, nbdevdn=std_dev)
    
    @staticmethod
    def calculate_stochastic(high: np.ndarray, low: np.ndarray, close: np.ndarray, k_period: int = 14, d_period: int = 3) -> Tuple[np.ndarray, np.ndarray]:
        """Stochastic oscillator"""
        return talib.STOCH(high, low, close, fastk_period=k_period, slowk_period=d_period, slowd_period=d_period)
    
    @staticmethod
    def calculate_atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
        """Average True Range"""
        return talib.ATR(high, low, close, timeperiod=period)
    
    @staticmethod
    def calculate_adx(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
        """Average Directional Index"""
        return talib.ADX(high, low, close, timeperiod=period)


class OrderBookFeatures:
    """Extract features from order book data"""
    
    @staticmethod
    def calculate_spread(order_book: OrderBook) -> float:
        """Calculate bid-ask spread"""
        best_bid = order_book.get_best_bid()
        best_ask = order_book.get_best_ask()
        if best_bid and best_ask:
            return float((best_ask.price - best_bid.price) / best_bid.price)
        return 0.0
    
    @staticmethod
    def calculate_mid_price(order_book: OrderBook) -> float:
        """Calculate mid price"""
        mid = order_book.get_mid_price()
        return float(mid) if mid else 0.0
    
    @staticmethod
    def calculate_order_book_imbalance(order_book: OrderBook, levels: int = 5) -> float:
        """Calculate order book imbalance (bid volume / (bid volume + ask volume))"""
        bid_levels = order_book.get_levels("bid", levels)
        ask_levels = order_book.get_levels("ask", levels)
        
        bid_volume = sum(float(level.size) for level in bid_levels)
        ask_volume = sum(float(level.size) for level in ask_levels)
        
        total_volume = bid_volume + ask_volume
        if total_volume > 0:
            return bid_volume / total_volume
        return 0.5
    
    @staticmethod
    def calculate_weighted_mid_price(order_book: OrderBook, levels: int = 5) -> float:
        """Calculate volume-weighted mid price"""
        bid_levels = order_book.get_levels("bid", levels)
        ask_levels = order_book.get_levels("ask", levels)
        
        if not bid_levels or not ask_levels:
            return 0.0
        
        bid_volume = sum(float(level.size) for level in bid_levels)
        ask_volume = sum(float(level.size) for level in ask_levels)
        
        if bid_volume + ask_volume > 0:
            best_bid = float(bid_levels[0].price)
            best_ask = float(ask_levels[0].price)
            return (best_ask * bid_volume + best_bid * ask_volume) / (bid_volume + ask_volume)
        return 0.0
    
    @staticmethod
    def calculate_order_flow_imbalance(trades: List[Trade], window_seconds: int = 60) -> float:
        """Calculate Order Flow Imbalance (OFI)"""
        if not trades:
            return 0.0
        
        current_time = time.time()
        recent_trades = [t for t in trades if current_time - t.timestamp.timestamp() <= window_seconds]
        
        buy_volume = sum(float(t.size) for t in recent_trades if t.side in ["buy", "BUY"])
        sell_volume = sum(float(t.size) for t in recent_trades if t.side in ["sell", "SELL"])
        
        total_volume = buy_volume + sell_volume
        if total_volume > 0:
            return (buy_volume - sell_volume) / total_volume
        return 0.0
    
    @staticmethod
    def calculate_volume_at_price(order_book: OrderBook, price_levels: int = 10) -> Dict[str, float]:
        """Calculate volume distribution at different price levels"""
        bid_levels = order_book.get_levels("bid", price_levels)
        ask_levels = order_book.get_levels("ask", price_levels)
        
        features = {}
        
        # Bid side volume features
        for i, level in enumerate(bid_levels):
            features[f"bid_volume_L{i+1}"] = float(level.size)
            features[f"bid_price_L{i+1}"] = float(level.price)
        
        # Ask side volume features
        for i, level in enumerate(ask_levels):
            features[f"ask_volume_L{i+1}"] = float(level.size)
            features[f"ask_price_L{i+1}"] = float(level.price)
        
        return features


class MicrostructureFeatures:
    """Extract microstructure features from high-frequency data"""
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.price_history: deque = deque(maxlen=window_size)
        self.volume_history: deque = deque(maxlen=window_size)
        self.spread_history: deque = deque(maxlen=window_size)
        self.trade_history: deque = deque(maxlen=window_size)
    
    def update(self, price: float, volume: float, spread: float, trade: Optional[Trade] = None):
        """Update microstructure data"""
        self.price_history.append(price)
        self.volume_history.append(volume)
        self.spread_history.append(spread)
        if trade:
            self.trade_history.append(trade)
    
    def calculate_realized_volatility(self, periods: int = 20) -> float:
        """Calculate realized volatility from price returns"""
        if len(self.price_history) < periods + 1:
            return 0.0
        
        prices = np.array(list(self.price_history)[-periods-1:])
        returns = np.diff(np.log(prices))
        return float(np.std(returns) * np.sqrt(252 * 24 * 60))  # Annualized
    
    def calculate_price_momentum(self, periods: int = 20) -> float:
        """Calculate price momentum"""
        if len(self.price_history) < periods + 1:
            return 0.0
        
        prices = np.array(list(self.price_history)[-periods-1:])
        return float((prices[-1] - prices[0]) / prices[0])
    
    def calculate_volume_profile(self) -> Dict[str, float]:
        """Calculate volume profile features"""
        if len(self.volume_history) < 10:
            return {"volume_mean": 0.0, "volume_std": 0.0, "volume_skew": 0.0}
        
        volumes = np.array(list(self.volume_history))
        return {
            "volume_mean": float(np.mean(volumes)),
            "volume_std": float(np.std(volumes)),
            "volume_skew": float(stats.skew(volumes))
        }
    
    def calculate_trade_intensity(self, window_seconds: int = 60) -> float:
        """Calculate trade intensity (trades per second)"""
        if not self.trade_history:
            return 0.0
        
        current_time = time.time()
        recent_trades = [t for t in self.trade_history 
                        if current_time - t.timestamp.timestamp() <= window_seconds]
        
        return len(recent_trades) / window_seconds
    
    def calculate_kyle_lambda(self) -> float:
        """Calculate Kyle's lambda (price impact coefficient)"""
        if len(self.price_history) < 20 or len(self.volume_history) < 20:
            return 0.0
        
        prices = np.array(list(self.price_history)[-20:])
        volumes = np.array(list(self.volume_history)[-20:])
        
        if len(prices) > 1 and len(volumes) > 1:
            price_changes = np.diff(prices) / prices[:-1]
            volume_imbalances = volumes[1:] - np.mean(volumes)
            
            if np.std(volume_imbalances) > 0:
                correlation = np.corrcoef(price_changes, volume_imbalances)[0, 1]
                return float(correlation * np.std(price_changes) / np.std(volume_imbalances))
        
        return 0.0


class FeatureStore:
    """Feature storage and retrieval system"""
    
    def __init__(self, max_history: int = 10000):
        self.max_history = max_history
        self.features_history: Dict[str, deque] = {}  # symbol -> deque of FeatureSet
        self.scalers: Dict[str, StandardScaler] = {}
    
    def store_features(self, feature_set: FeatureSet):
        """Store feature set"""
        symbol = feature_set.symbol
        if symbol not in self.features_history:
            self.features_history[symbol] = deque(maxlen=self.max_history)
        
        self.features_history[symbol].append(feature_set)
    
    def get_features(self, symbol: str, lookback: int = 100) -> List[FeatureSet]:
        """Get recent features for symbol"""
        if symbol not in self.features_history:
            return []
        
        return list(self.features_history[symbol])[-lookback:]
    
    def get_feature_matrix(self, symbol: str, feature_names: List[str], lookback: int = 100) -> np.ndarray:
        """Get feature matrix for ML models"""
        feature_sets = self.get_features(symbol, lookback)
        if not feature_sets:
            return np.array([])
        
        matrix = np.array([fs.to_array(feature_names) for fs in feature_sets])
        return matrix
    
    def fit_scaler(self, symbol: str, feature_names: List[str], scaler_type: str = "standard"):
        """Fit feature scaler"""
        matrix = self.get_feature_matrix(symbol, feature_names)
        if matrix.size == 0:
            return
        
        if scaler_type == "standard":
            scaler = StandardScaler()
        elif scaler_type == "robust":
            scaler = RobustScaler()
        else:
            raise ValueError(f"Unknown scaler type: {scaler_type}")
        
        scaler.fit(matrix)
        self.scalers[f"{symbol}_{scaler_type}"] = scaler
    
    def transform_features(self, symbol: str, features: np.ndarray, scaler_type: str = "standard") -> np.ndarray:
        """Transform features using fitted scaler"""
        scaler_key = f"{symbol}_{scaler_type}"
        if scaler_key not in self.scalers:
            return features
        
        return self.scalers[scaler_key].transform(features)


class FeatureEngine:
    """Main feature engineering engine"""
    
    def __init__(self):
        self.settings = get_settings()
        self.feature_store = FeatureStore()
        self.technical_indicators = TechnicalIndicators()
        self.microstructure_features: Dict[str, MicrostructureFeatures] = {}
        
        # Data buffers
        self.ohlcv_data: Dict[str, pd.DataFrame] = {}
        self.order_books: Dict[str, Dict[str, OrderBook]] = {}
        self.recent_trades: Dict[str, List[Trade]] = {}
        
        # Feature configuration
        self.feature_config = {
            'technical_periods': [5, 10, 20, 50],
            'orderbook_levels': 10,
            'microstructure_window': 100,
            'volume_window_seconds': 300,
            'trade_window_seconds': 60
        }
    
    async def update_market_data(self, exchange: str, symbol: str, order_book: OrderBook, 
                                trades: Optional[List[Trade]] = None, ticker: Optional[Ticker] = None):
        """Update market data and trigger feature calculation"""
        # Store order book
        if exchange not in self.order_books:
            self.order_books[exchange] = {}
        self.order_books[exchange][symbol] = order_book
        
        # Store trades
        if trades:
            if symbol not in self.recent_trades:
                self.recent_trades[symbol] = []
            self.recent_trades[symbol].extend(trades)
            
            # Keep only recent trades
            current_time = time.time()
            window = self.feature_config['volume_window_seconds']
            self.recent_trades[symbol] = [
                t for t in self.recent_trades[symbol] 
                if current_time - t.timestamp.timestamp() <= window
            ]
        
        # Update OHLCV data
        if ticker:
            await self._update_ohlcv(symbol, ticker)
        
        # Calculate and store features
        feature_set = await self.calculate_features(exchange, symbol)
        if feature_set:
            self.feature_store.store_features(feature_set)
    
    async def _update_ohlcv(self, symbol: str, ticker: Ticker):
        """Update OHLCV data from ticker"""
        if symbol not in self.ohlcv_data:
            self.ohlcv_data[symbol] = pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        
        # Add new OHLCV row
        new_row = {
            'timestamp': ticker.timestamp,
            'open': float(ticker.open_24h) if ticker.open_24h else float(ticker.last_price),
            'high': float(ticker.high_24h) if ticker.high_24h else float(ticker.last_price),
            'low': float(ticker.low_24h) if ticker.low_24h else float(ticker.last_price),
            'close': float(ticker.last_price) if ticker.last_price else 0.0,
            'volume': float(ticker.volume_24h) if ticker.volume_24h else 0.0
        }
        
        self.ohlcv_data[symbol] = pd.concat([
            self.ohlcv_data[symbol], 
            pd.DataFrame([new_row])
        ], ignore_index=True)
        
        # Keep only recent data (1000 periods)
        if len(self.ohlcv_data[symbol]) > 1000:
            self.ohlcv_data[symbol] = self.ohlcv_data[symbol].tail(1000).reset_index(drop=True)
    
    async def calculate_features(self, exchange: str, symbol: str) -> Optional[FeatureSet]:
        """Calculate comprehensive feature set"""
        features = {}
        timestamp = time.time()
        
        # Get order book
        order_book = self.order_books.get(exchange, {}).get(symbol)
        if not order_book:
            return None
        
        # Order book features
        ob_features = OrderBookFeatures()
        features.update({
            'spread': Feature('spread', ob_features.calculate_spread(order_book), FeatureType.ORDERBOOK, timestamp),
            'mid_price': Feature('mid_price', ob_features.calculate_mid_price(order_book), FeatureType.ORDERBOOK, timestamp),
            'ob_imbalance': Feature('ob_imbalance', ob_features.calculate_order_book_imbalance(order_book), FeatureType.ORDERBOOK, timestamp),
            'weighted_mid': Feature('weighted_mid', ob_features.calculate_weighted_mid_price(order_book), FeatureType.ORDERBOOK, timestamp),
        })
        
        # Volume at price features
        vap_features = ob_features.calculate_volume_at_price(order_book, self.feature_config['orderbook_levels'])
        for name, value in vap_features.items():
            features[name] = Feature(name, value, FeatureType.ORDERBOOK, timestamp)
        
        # Order flow imbalance
        symbol_trades = self.recent_trades.get(symbol, [])
        ofi = ob_features.calculate_order_flow_imbalance(symbol_trades, self.feature_config['trade_window_seconds'])
        features['ofi'] = Feature('ofi', ofi, FeatureType.MICROSTRUCTURE, timestamp)
        
        # Microstructure features
        if symbol not in self.microstructure_features:
            self.microstructure_features[symbol] = MicrostructureFeatures(self.feature_config['microstructure_window'])
        
        ms_engine = self.microstructure_features[symbol]
        mid_price = features['mid_price'].value
        spread = features['spread'].value
        volume = sum(float(level.size) for level in order_book.get_levels("bid", 5) + order_book.get_levels("ask", 5))
        
        ms_engine.update(mid_price, volume, spread, symbol_trades[-1] if symbol_trades else None)
        
        # Calculate microstructure features
        features.update({
            'realized_vol': Feature('realized_vol', ms_engine.calculate_realized_volatility(), FeatureType.MICROSTRUCTURE, timestamp),
            'price_momentum': Feature('price_momentum', ms_engine.calculate_price_momentum(), FeatureType.MICROSTRUCTURE, timestamp),
            'trade_intensity': Feature('trade_intensity', ms_engine.calculate_trade_intensity(), FeatureType.MICROSTRUCTURE, timestamp),
            'kyle_lambda': Feature('kyle_lambda', ms_engine.calculate_kyle_lambda(), FeatureType.MICROSTRUCTURE, timestamp),
        })
        
        # Volume profile features
        vol_profile = ms_engine.calculate_volume_profile()
        for name, value in vol_profile.items():
            features[name] = Feature(name, value, FeatureType.VOLUME, timestamp)
        
        # Technical indicators (if OHLCV data available)
        if symbol in self.ohlcv_data and len(self.ohlcv_data[symbol]) >= 50:
            df = self.ohlcv_data[symbol]
            prices = df['close'].values.astype(np.float64)
            highs = df['high'].values.astype(np.float64)
            lows = df['low'].values.astype(np.float64)
            volumes = df['volume'].values.astype(np.float64)
            
            # Calculate technical indicators
            for period in self.feature_config['technical_periods']:
                if len(prices) >= period:
                    sma = self.technical_indicators.calculate_sma(prices, period)
                    ema = self.technical_indicators.calculate_ema(prices, period)
                    
                    if not np.isnan(sma[-1]):
                        features[f'sma_{period}'] = Feature(f'sma_{period}', float(sma[-1]), FeatureType.TECHNICAL, timestamp)
                    if not np.isnan(ema[-1]):
                        features[f'ema_{period}'] = Feature(f'ema_{period}', float(ema[-1]), FeatureType.TECHNICAL, timestamp)
            
            # RSI
            if len(prices) >= 14:
                rsi = self.technical_indicators.calculate_rsi(prices)
                if not np.isnan(rsi[-1]):
                    features['rsi'] = Feature('rsi', float(rsi[-1]), FeatureType.TECHNICAL, timestamp)
            
            # MACD
            if len(prices) >= 26:
                macd, signal, histogram = self.technical_indicators.calculate_macd(prices)
                if not np.isnan(macd[-1]):
                    features['macd'] = Feature('macd', float(macd[-1]), FeatureType.TECHNICAL, timestamp)
                    features['macd_signal'] = Feature('macd_signal', float(signal[-1]), FeatureType.TECHNICAL, timestamp)
                    features['macd_histogram'] = Feature('macd_histogram', float(histogram[-1]), FeatureType.TECHNICAL, timestamp)
            
            # Bollinger Bands
            if len(prices) >= 20:
                upper, middle, lower = self.technical_indicators.calculate_bollinger_bands(prices)
                if not np.isnan(upper[-1]):
                    features['bb_upper'] = Feature('bb_upper', float(upper[-1]), FeatureType.TECHNICAL, timestamp)
                    features['bb_middle'] = Feature('bb_middle', float(middle[-1]), FeatureType.TECHNICAL, timestamp)
                    features['bb_lower'] = Feature('bb_lower', float(lower[-1]), FeatureType.TECHNICAL, timestamp)
                    # Bollinger Band position
                    bb_position = (prices[-1] - lower[-1]) / (upper[-1] - lower[-1])
                    features['bb_position'] = Feature('bb_position', float(bb_position), FeatureType.TECHNICAL, timestamp)
            
            # ATR
            if len(prices) >= 14:
                atr = self.technical_indicators.calculate_atr(highs, lows, prices)
                if not np.isnan(atr[-1]):
                    features['atr'] = Feature('atr', float(atr[-1]), FeatureType.TECHNICAL, timestamp)
            
            # ADX
            if len(prices) >= 14:
                adx = self.technical_indicators.calculate_adx(highs, lows, prices)
                if not np.isnan(adx[-1]):
                    features['adx'] = Feature('adx', float(adx[-1]), FeatureType.TECHNICAL, timestamp)
        
        return FeatureSet(symbol=symbol, timestamp=timestamp, features=features)
    
    async def get_ml_features(self, symbol: str, feature_names: List[str], lookback: int = 100) -> np.ndarray:
        """Get features formatted for ML models"""
        return self.feature_store.get_feature_matrix(symbol, feature_names, lookback)
    
    async def get_latest_features(self, symbol: str) -> Optional[FeatureSet]:
        """Get latest feature set for symbol"""
        features = self.feature_store.get_features(symbol, 1)
        return features[0] if features else None


# Global feature engine instance
_feature_engine: Optional[FeatureEngine] = None


def get_feature_engine() -> FeatureEngine:
    """Get global feature engine instance"""
    global _feature_engine
    if _feature_engine is None:
        _feature_engine = FeatureEngine()
    return _feature_engine


# Example usage
async def main():
    """Example feature engineering usage"""
    import logging
    logging.basicConfig(level=logging.INFO)
    
    engine = get_feature_engine()
    
    # This would be called when new market data arrives
    # await engine.update_market_data(exchange, symbol, order_book, trades, ticker)
    
    print("Feature engine initialized successfully!")


if __name__ == "__main__":
    asyncio.run(main())
