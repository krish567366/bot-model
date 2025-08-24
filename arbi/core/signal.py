"""
Signal Generation Module

Implements arbitrage signal detection including:
- Cross-exchange spot arbitrage
- Triangular arbitrage with Bellman-Ford algorithm
- Microstructure signals (OFI, spread, imbalance)
- AI/ML enhanced signals with market regime detection
"""

import asyncio
import logging
import time
from decimal import Decimal
from typing import Dict, List, Optional, Tuple, Set, AsyncGenerator
from dataclasses import dataclass
from enum import Enum
import numpy as np
import pandas as pd
from collections import defaultdict, deque

from .marketdata import BookDelta, OrderBook, Ticker, Trade, OrderBookLevel
from ..config.settings import get_settings

# Import AI inference engine if available
try:
    from ..ai.inference import InferenceEngine, MarketRegimeDetector
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False

logger = logging.getLogger(__name__)
if not ML_AVAILABLE:
    logger.warning("AI/ML inference engine not available. Running without ML signals.")


class SignalType(str, Enum):
    """Types of arbitrage signals"""
    CROSS_EXCHANGE = "cross_exchange"
    TRIANGULAR = "triangular"
    MICROSTRUCTURE = "microstructure"
    STATISTICAL = "statistical"
    ML_ENHANCED = "ml_enhanced"  # AI/ML enhanced signals


class SignalStrength(str, Enum):
    """Signal strength classification"""
    WEAK = "weak"
    MODERATE = "moderate"
    STRONG = "strong"
    CRITICAL = "critical"


@dataclass
class ArbitrageSignal:
    """Arbitrage opportunity signal"""
    signal_id: str
    signal_type: SignalType
    timestamp: float
    symbol: str
    
    # Opportunity details
    expected_profit_pct: Decimal
    expected_profit_usd: Decimal
    confidence: float
    strength: SignalStrength
    
    # Execution details
    buy_exchange: str
    sell_exchange: str
    buy_price: Decimal
    sell_price: Decimal
    size: Decimal
    
    # Risk factors
    slippage_estimate: Decimal
    transfer_time_penalty: Decimal
    fees_total: Decimal
    
    # Additional data
    mid_price_ref: Decimal
    spread_width_pct: Decimal
    volume_24h: Optional[Decimal] = None
    
    # For triangular arbitrage
    path: Optional[List[str]] = None
    intermediate_prices: Optional[List[Decimal]] = None
    
    # For ML enhanced signals
    ml_confidence: Optional[float] = None
    ml_prediction: Optional[dict] = None
    traditional_signal_id: Optional[str] = None
    market_regime: Optional[str] = None
    
    def __post_init__(self):
        """Calculate derived fields"""
        self.net_profit_pct = self.expected_profit_pct - self.slippage_estimate - self.transfer_time_penalty
        self.net_profit_usd = self.expected_profit_usd - (self.size * self.fees_total)
        
        # Classify strength based on net profit and ML confidence
        base_profit = self.net_profit_pct
        
        # Adjust strength based on ML confidence if available
        if self.ml_confidence is not None and self.signal_type == SignalType.ML_ENHANCED:
            # Boost or reduce strength based on ML confidence
            if self.ml_confidence > 0.8:
                base_profit *= Decimal('1.2')  # 20% boost for high ML confidence
            elif self.ml_confidence < 0.4:
                base_profit *= Decimal('0.8')  # 20% reduction for low ML confidence
        
        # Classify strength
        if base_profit >= Decimal('1.0'):
            self.strength = SignalStrength.CRITICAL
        elif base_profit >= Decimal('0.5'):
            self.strength = SignalStrength.STRONG
        elif base_profit >= Decimal('0.1'):
            self.strength = SignalStrength.MODERATE
        else:
            self.strength = SignalStrength.WEAK


@dataclass
class TriangularPath:
    """Triangular arbitrage path"""
    symbols: List[str]
    exchanges: List[str]
    prices: List[Decimal]
    sizes: List[Decimal]
    profit_factor: Decimal
    
    @property
    def profit_pct(self) -> Decimal:
        return (self.profit_factor - 1) * 100


class FeeCalculator:
    """Calculate trading fees for different exchanges"""
    
    # Fee structures (maker/taker) for different exchanges
    FEE_STRUCTURES = {
        'binance': {'maker': Decimal('0.001'), 'taker': Decimal('0.001')},
        'kraken': {'maker': Decimal('0.0016'), 'taker': Decimal('0.0026')},
        'coinbase': {'maker': Decimal('0.005'), 'taker': Decimal('0.005')},
        'bybit': {'maker': Decimal('0.001'), 'taker': Decimal('0.001')},
    }
    
    @classmethod
    def get_fee(cls, exchange: str, is_maker: bool = False) -> Decimal:
        """Get trading fee for exchange"""
        fees = cls.FEE_STRUCTURES.get(exchange.lower(), {'maker': Decimal('0.002'), 'taker': Decimal('0.002')})
        return fees['maker'] if is_maker else fees['taker']
    
    @classmethod
    def calculate_total_fees(cls, buy_exchange: str, sell_exchange: str, size: Decimal) -> Decimal:
        """Calculate total fees for round-trip trade"""
        buy_fee = cls.get_fee(buy_exchange, is_maker=False)
        sell_fee = cls.get_fee(sell_exchange, is_maker=False)
        return size * (buy_fee + sell_fee)


class CrossExchangeArbitrage:
    """Cross-exchange arbitrage signal detection"""
    
    def __init__(self):
        self.settings = get_settings()
        self.order_books: Dict[str, Dict[str, OrderBook]] = {}
        self.tickers: Dict[str, Dict[str, Ticker]] = {}
        
    def update_order_book(self, exchange: str, order_book: OrderBook):
        """Update order book data"""
        if exchange not in self.order_books:
            self.order_books[exchange] = {}
        self.order_books[exchange][order_book.symbol] = order_book
    
    def update_ticker(self, exchange: str, ticker: Ticker):
        """Update ticker data"""
        if exchange not in self.tickers:
            self.tickers[exchange] = {}
        self.tickers[exchange][ticker.symbol] = ticker
    
    def detect_opportunities(self, symbol: str) -> List[ArbitrageSignal]:
        """Detect cross-exchange arbitrage opportunities"""
        opportunities = []
        
        # Get all exchanges with data for this symbol
        exchanges_with_data = []
        for exchange, books in self.order_books.items():
            if symbol in books and symbol in self.tickers.get(exchange, {}):
                exchanges_with_data.append(exchange)
        
        if len(exchanges_with_data) < 2:
            return opportunities
        
        # Compare all exchange pairs
        for i, buy_exchange in enumerate(exchanges_with_data):
            for sell_exchange in exchanges_with_data[i+1:]:
                
                # Try both directions
                opp1 = self._analyze_pair(symbol, buy_exchange, sell_exchange)
                if opp1:
                    opportunities.append(opp1)
                
                opp2 = self._analyze_pair(symbol, sell_exchange, buy_exchange)
                if opp2:
                    opportunities.append(opp2)
        
        return opportunities
    
    def _analyze_pair(self, symbol: str, buy_exchange: str, sell_exchange: str) -> Optional[ArbitrageSignal]:
        """Analyze arbitrage opportunity between two exchanges"""
        try:
            # Get order books
            buy_book = self.order_books[buy_exchange][symbol]
            sell_book = self.order_books[sell_exchange][symbol]
            
            # Get best prices
            best_ask = buy_book.get_best_ask()  # Price to buy
            best_bid = sell_book.get_best_bid()  # Price to sell
            
            if not best_ask or not best_bid:
                return None
            
            buy_price = best_ask.price
            sell_price = best_bid.price
            
            # Check if profitable before fees
            if sell_price <= buy_price:
                return None
            
            # Calculate size (limited by both book depths)
            max_size = min(best_ask.size, best_bid.size)
            
            # Apply position size limits
            max_position = self.settings.risk.max_position_size
            if buy_price > 0:
                max_size_by_capital = max_position / buy_price
                max_size = min(max_size, max_size_by_capital)
            
            if max_size <= 0:
                return None
            
            # Calculate reference mid price
            buy_mid = buy_book.get_mid_price()
            sell_mid = sell_book.get_mid_price()
            if not buy_mid or not sell_mid:
                return None
            
            mid_ref = (buy_mid + sell_mid) / 2
            
            # Calculate raw profit
            raw_profit_pct = ((sell_price - buy_price) / mid_ref) * 100
            
            # Calculate costs
            fees_total = FeeCalculator.calculate_total_fees(buy_exchange, sell_exchange, max_size)
            slippage_estimate = self.settings.strategy.max_slippage_pct / 2  # Conservative estimate
            transfer_penalty = self.settings.strategy.transfer_time_penalty
            
            # Net profit calculation
            net_profit_pct = raw_profit_pct - slippage_estimate - transfer_penalty
            
            # Check minimum profit threshold
            if net_profit_pct < self.settings.strategy.min_arbitrage_spread:
                return None
            
            # Calculate spread metrics
            buy_spread_pct = ((best_ask.price - buy_mid) / buy_mid) * 100 if buy_mid > 0 else Decimal('0')
            sell_spread_pct = ((sell_mid - best_bid.price) / sell_mid) * 100 if sell_mid > 0 else Decimal('0')
            avg_spread_pct = (buy_spread_pct + sell_spread_pct) / 2
            
            # Check spread width limit
            if avg_spread_pct > self.settings.strategy.max_spread_width:
                return None
            
            # Get volume data
            volume_24h = None
            if symbol in self.tickers.get(buy_exchange, {}):
                ticker = self.tickers[buy_exchange][symbol]
                volume_24h = ticker.volume_24h
            
            # Check minimum volume
            if volume_24h and volume_24h < self.settings.strategy.min_volume_threshold:
                return None
            
            # Calculate confidence based on spread and volume
            confidence = self._calculate_confidence(avg_spread_pct, volume_24h, net_profit_pct)
            
            return ArbitrageSignal(
                signal_id=f"cross_{buy_exchange}_{sell_exchange}_{symbol}_{int(time.time())}",
                signal_type=SignalType.CROSS_EXCHANGE,
                timestamp=time.time(),
                symbol=symbol,
                expected_profit_pct=net_profit_pct,
                expected_profit_usd=max_size * (sell_price - buy_price) - fees_total,
                confidence=confidence,
                strength=SignalStrength.MODERATE,  # Will be updated in __post_init__
                buy_exchange=buy_exchange,
                sell_exchange=sell_exchange,
                buy_price=buy_price,
                sell_price=sell_price,
                size=max_size,
                slippage_estimate=slippage_estimate,
                transfer_time_penalty=transfer_penalty,
                fees_total=fees_total,
                mid_price_ref=mid_ref,
                spread_width_pct=avg_spread_pct,
                volume_24h=volume_24h
            )
            
        except Exception as e:
            logger.error(f"Error analyzing arbitrage pair {buy_exchange}-{sell_exchange} for {symbol}: {e}")
            return None
    
    def _calculate_confidence(self, spread_pct: Decimal, volume_24h: Optional[Decimal], profit_pct: Decimal) -> float:
        """Calculate signal confidence score (0-1)"""
        confidence = 0.5  # Base confidence
        
        # Adjust for spread (tighter spreads = higher confidence)
        if spread_pct < 0.1:
            confidence += 0.3
        elif spread_pct < 0.5:
            confidence += 0.2
        elif spread_pct > 2.0:
            confidence -= 0.2
        
        # Adjust for volume
        if volume_24h:
            if volume_24h > 1000000:  # > $1M daily volume
                confidence += 0.2
            elif volume_24h < 100000:  # < $100K daily volume
                confidence -= 0.2
        
        # Adjust for profit magnitude
        if profit_pct > 1.0:
            confidence += 0.1
        elif profit_pct < 0.2:
            confidence -= 0.1
        
        return max(0.0, min(1.0, float(confidence)))


class TriangularArbitrage:
    """Triangular arbitrage detection using Bellman-Ford algorithm"""
    
    def __init__(self):
        self.settings = get_settings()
        self.order_books: Dict[str, OrderBook] = {}
        self.graph_cache = {}
        self.last_update = 0
        
    def update_order_book(self, order_book: OrderBook):
        """Update order book for triangular arbitrage calculation"""
        key = f"{order_book.exchange}:{order_book.symbol}"
        self.order_books[key] = order_book
        self.last_update = time.time()
    
    def build_currency_graph(self, exchange: str) -> Dict[str, Dict[str, Tuple[Decimal, Decimal]]]:
        """Build currency graph with exchange rates and sizes"""
        graph = defaultdict(dict)
        
        # Get all symbols for this exchange
        exchange_books = {k: v for k, v in self.order_books.items() if k.startswith(f"{exchange}:")}
        
        for key, book in exchange_books.items():
            symbol = key.split(":")[1]
            
            # Parse symbol (e.g., "BTC/USDT" -> base="BTC", quote="USDT")
            if "/" not in symbol:
                continue
            
            base, quote = symbol.split("/")
            
            # Get best bid/ask
            best_bid = book.get_best_bid()
            best_ask = book.get_best_ask()
            
            if not best_bid or not best_ask:
                continue
            
            # Forward direction: base -> quote (selling base)
            # Rate = bid price, size = bid size
            graph[base][quote] = (best_bid.price, best_bid.size)
            
            # Reverse direction: quote -> base (buying base)
            # Rate = 1/ask price, size = ask size * ask price
            if best_ask.price > 0:
                reverse_rate = Decimal('1') / best_ask.price
                reverse_size = best_ask.size * best_ask.price
                graph[quote][base] = (reverse_rate, reverse_size)
        
        return graph
    
    def find_arbitrage_cycles(self, exchange: str, start_currency: str = "USDT") -> List[TriangularPath]:
        """Find profitable arbitrage cycles using Bellman-Ford variant"""
        graph = self.build_currency_graph(exchange)
        
        if start_currency not in graph:
            return []
        
        cycles = []
        max_legs = self.settings.strategy.max_triangular_legs
        
        def dfs_cycles(current: str, path: List[str], rates: List[Decimal], sizes: List[Decimal], depth: int):
            """DFS to find cycles"""
            if depth > max_legs:
                return
            
            if depth > 2 and current == start_currency:
                # Found a cycle back to start currency
                total_rate = Decimal('1')
                for rate in rates:
                    total_rate *= rate
                
                if total_rate > Decimal('1'):
                    # Profitable cycle found
                    min_size = min(sizes) if sizes else Decimal('0')
                    
                    cycles.append(TriangularPath(
                        symbols=self._path_to_symbols(path),
                        exchanges=[exchange] * (len(path) - 1),
                        prices=rates.copy(),
                        sizes=[min_size] * len(rates),
                        profit_factor=total_rate
                    ))
                return
            
            # Continue exploring
            if current in graph:
                for next_currency, (rate, size) in graph[current].items():
                    if next_currency not in path or (depth > 2 and next_currency == start_currency):
                        dfs_cycles(
                            next_currency,
                            path + [next_currency],
                            rates + [rate],
                            sizes + [size],
                            depth + 1
                        )
        
        # Start DFS from the starting currency
        dfs_cycles(start_currency, [start_currency], [], [], 0)
        
        # Filter and sort by profitability
        profitable_cycles = [
            cycle for cycle in cycles 
            if cycle.profit_pct >= self.settings.strategy.min_triangular_profit
        ]
        
        return sorted(profitable_cycles, key=lambda x: x.profit_factor, reverse=True)
    
    def _path_to_symbols(self, path: List[str]) -> List[str]:
        """Convert currency path to trading symbols"""
        symbols = []
        for i in range(len(path) - 1):
            base, quote = path[i], path[i + 1]
            # Try both directions to find the actual symbol
            symbol1 = f"{base}/{quote}"
            symbol2 = f"{quote}/{base}"
            symbols.append(symbol1)  # Simplified - would need proper symbol mapping
        return symbols
    
    def generate_signals(self, exchange: str) -> List[ArbitrageSignal]:
        """Generate triangular arbitrage signals"""
        cycles = self.find_arbitrage_cycles(exchange)
        signals = []
        
        for cycle in cycles:
            try:
                # Calculate total size and fees
                total_size = min(cycle.sizes) if cycle.sizes else Decimal('0')
                total_fees = sum(
                    FeeCalculator.get_fee(exchange, is_maker=False) * total_size
                    for _ in cycle.symbols
                )
                
                # Calculate net profit
                gross_profit = (cycle.profit_factor - 1) * total_size
                net_profit = gross_profit - total_fees
                net_profit_pct = ((cycle.profit_factor - 1) - (total_fees / total_size)) * 100
                
                if net_profit_pct <= 0:
                    continue
                
                signal = ArbitrageSignal(
                    signal_id=f"triangular_{exchange}_{int(time.time())}",
                    signal_type=SignalType.TRIANGULAR,
                    timestamp=time.time(),
                    symbol=cycle.symbols[0] if cycle.symbols else "MULTI",
                    expected_profit_pct=net_profit_pct,
                    expected_profit_usd=net_profit,
                    confidence=0.8,  # High confidence for triangular
                    strength=SignalStrength.MODERATE,
                    buy_exchange=exchange,
                    sell_exchange=exchange,
                    buy_price=cycle.prices[0] if cycle.prices else Decimal('0'),
                    sell_price=cycle.prices[-1] if cycle.prices else Decimal('0'),
                    size=total_size,
                    slippage_estimate=Decimal('0.1'),  # Lower slippage for same exchange
                    transfer_time_penalty=Decimal('0'),  # No transfer needed
                    fees_total=total_fees,
                    mid_price_ref=cycle.prices[0] if cycle.prices else Decimal('0'),
                    spread_width_pct=Decimal('0.5'),  # Estimated
                    path=cycle.symbols,
                    intermediate_prices=cycle.prices
                )
                
                signals.append(signal)
                
            except Exception as e:
                logger.error(f"Error generating triangular signal: {e}")
                continue
        
        return signals


class MicrostructureSignals:
    """Microstructure-based signals (OFI, spread, imbalance)"""
    
    def __init__(self, lookback_window: int = 100):
        self.lookback_window = lookback_window
        self.trade_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=lookback_window))
        self.book_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=lookback_window))
        
    def update_trade(self, trade: Trade):
        """Update with new trade data"""
        key = f"{trade.exchange}:{trade.symbol}"
        self.trade_history[key].append(trade)
    
    def update_book(self, delta: BookDelta):
        """Update with new book delta"""
        key = f"{delta.exchange}:{delta.symbol}"
        self.book_history[key].append(delta)
    
    def calculate_order_flow_imbalance(self, exchange: str, symbol: str) -> Optional[float]:
        """Calculate Order Flow Imbalance (OFI)"""
        key = f"{exchange}:{symbol}"
        trades = list(self.trade_history.get(key, []))
        
        if len(trades) < 10:
            return None
        
        # Calculate OFI over recent trades
        buy_volume = sum(float(t.size) for t in trades[-10:] if t.side in ['buy', 'BUY'])
        sell_volume = sum(float(t.size) for t in trades[-10:] if t.side in ['sell', 'SELL'])
        
        total_volume = buy_volume + sell_volume
        if total_volume == 0:
            return 0.0
        
        return (buy_volume - sell_volume) / total_volume
    
    def calculate_bid_ask_imbalance(self, order_book: OrderBook) -> Optional[float]:
        """Calculate bid-ask imbalance"""
        best_bid = order_book.get_best_bid()
        best_ask = order_book.get_best_ask()
        
        if not best_bid or not best_ask:
            return None
        
        bid_size = float(best_bid.size)
        ask_size = float(best_ask.size)
        total_size = bid_size + ask_size
        
        if total_size == 0:
            return 0.0
        
        return (bid_size - ask_size) / total_size
    
    def generate_microstructure_signals(self, exchange: str, symbol: str, order_book: OrderBook) -> List[ArbitrageSignal]:
        """Generate microstructure-based signals"""
        signals = []
        
        try:
            ofi = self.calculate_order_flow_imbalance(exchange, symbol)
            imbalance = self.calculate_bid_ask_imbalance(order_book)
            
            if ofi is None or imbalance is None:
                return signals
            
            # Strong directional signal
            if abs(ofi) > 0.6 and abs(imbalance) > 0.4 and np.sign(ofi) == np.sign(imbalance):
                
                mid_price = order_book.get_mid_price()
                if not mid_price:
                    return signals
                
                # Predict short-term price movement
                direction = 1 if ofi > 0 else -1
                expected_move_pct = abs(ofi) * abs(imbalance) * Decimal('0.1')  # Conservative estimate
                
                signal = ArbitrageSignal(
                    signal_id=f"micro_{exchange}_{symbol}_{int(time.time())}",
                    signal_type=SignalType.MICROSTRUCTURE,
                    timestamp=time.time(),
                    symbol=symbol,
                    expected_profit_pct=expected_move_pct,
                    expected_profit_usd=Decimal('0'),  # Would need position size
                    confidence=float(min(abs(ofi), abs(imbalance))),
                    strength=SignalStrength.WEAK,
                    buy_exchange=exchange if direction > 0 else "",
                    sell_exchange=exchange if direction < 0 else "",
                    buy_price=mid_price,
                    sell_price=mid_price,
                    size=Decimal('1'),  # Placeholder
                    slippage_estimate=Decimal('0.05'),
                    transfer_time_penalty=Decimal('0'),
                    fees_total=Decimal('0.002'),
                    mid_price_ref=mid_price,
                    spread_width_pct=Decimal('0.1')
                )
                
                signals.append(signal)
        
        except Exception as e:
            logger.error(f"Error generating microstructure signals: {e}")
        
        return signals


class SignalAggregator:
    """Aggregates signals from different sources and manages signal lifecycle"""
    
    def __init__(self):
        self.cross_exchange = CrossExchangeArbitrage()
        self.triangular = TriangularArbitrage()
        self.microstructure = MicrostructureSignals()
        self.active_signals: Dict[str, ArbitrageSignal] = {}
        self.signal_history: deque = deque(maxlen=1000)
        
        # Initialize ML inference engine if available
        self.ml_inference = None
        if ML_AVAILABLE:
            try:
                self.ml_inference = InferenceEngine()
                logger.info("ML inference engine initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize ML inference engine: {e}")
                self.ml_inference = None
        
    def update_data(self, exchange: str, order_book: OrderBook, ticker: Optional[Ticker] = None):
        """Update all signal generators with new data"""
        # Update cross-exchange signals
        self.cross_exchange.update_order_book(exchange, order_book)
        if ticker:
            self.cross_exchange.update_ticker(exchange, ticker)
        
        # Update triangular arbitrage
        self.triangular.update_order_book(order_book)
        
        # Update ML inference engine if available
        if self.ml_inference:
            try:
                self.ml_inference.update_market_data(exchange, order_book, ticker)
            except Exception as e:
                logger.error(f"Error updating ML inference engine: {e}")
    
    
    def update_trade(self, trade: Trade):
        """Update microstructure signals with trade data"""
        self.microstructure.update_trade(trade)
        
        # Update ML inference engine with trade data
        if self.ml_inference:
            try:
                self.ml_inference.update_trade_data(trade)
            except Exception as e:
                logger.error(f"Error updating ML inference with trade data: {e}")
    
    def update_book_delta(self, delta: BookDelta):
        """Update microstructure signals with book delta"""
        self.microstructure.update_book(delta)
        
        # Update ML inference engine with book delta
        if self.ml_inference:
            try:
                self.ml_inference.update_book_delta(delta)
            except Exception as e:
                logger.error(f"Error updating ML inference with book delta: {e}")
    
    async def generate_all_signals(self, symbol: str) -> AsyncGenerator[ArbitrageSignal, None]:
        """Generate signals from all sources"""
        try:
            # Cross-exchange arbitrage
            cross_signals = self.cross_exchange.detect_opportunities(symbol)
            for signal in cross_signals:
                if self._is_valid_signal(signal):
                    yield signal
            
            # Triangular arbitrage (check all exchanges)
            exchanges = set()
            for key in self.triangular.order_books.keys():
                exchange = key.split(":")[0]
                exchanges.add(exchange)
            
            for exchange in exchanges:
                triangular_signals = self.triangular.generate_signals(exchange)
                for signal in triangular_signals:
                    if self._is_valid_signal(signal):
                        yield signal
            
            # Microstructure signals (for each exchange/symbol combo)
            for key, order_book in self.cross_exchange.order_books.items():
                if symbol in order_book:
                    micro_signals = self.microstructure.generate_microstructure_signals(
                        key, symbol, order_book[symbol]
                    )
                    for signal in micro_signals:
                        if self._is_valid_signal(signal):
                            yield signal
            
            # ML enhanced signals
            if self.ml_inference:
                try:
                    ml_signals = await self.generate_ml_enhanced_signals(symbol)
                    for signal in ml_signals:
                        if self._is_valid_signal(signal):
                            yield signal
                except Exception as e:
                    logger.error(f"Error generating ML enhanced signals: {e}")
        
        except Exception as e:
            logger.error(f"Error generating signals for {symbol}: {e}")
    
    async def generate_ml_enhanced_signals(self, symbol: str) -> List[ArbitrageSignal]:
        """Generate ML enhanced arbitrage signals"""
        if not self.ml_inference:
            return []
        
        ml_signals = []
        
        try:
            # Get traditional signals first
            traditional_signals = []
            
            # Cross-exchange signals
            cross_signals = self.cross_exchange.detect_opportunities(symbol)
            traditional_signals.extend(cross_signals)
            
            # Generate ML enhanced versions
            for trad_signal in traditional_signals:
                try:
                    # Get ML prediction for this opportunity
                    ml_prediction = await self.ml_inference.get_signal_prediction(
                        exchange=trad_signal.buy_exchange,
                        symbol=symbol,
                        signal_type=trad_signal.signal_type.value
                    )
                    
                    if ml_prediction and ml_prediction.get('confidence', 0) > 0.5:
                        # Create ML enhanced signal
                        ml_signal = self._create_ml_enhanced_signal(trad_signal, ml_prediction)
                        if ml_signal:
                            ml_signals.append(ml_signal)
                
                except Exception as e:
                    logger.error(f"Error enhancing signal with ML: {e}")
                    continue
        
        except Exception as e:
            logger.error(f"Error in ML signal generation: {e}")
        
        return ml_signals
    
    def _create_ml_enhanced_signal(self, traditional_signal: ArbitrageSignal, ml_prediction: dict) -> Optional[ArbitrageSignal]:
        """Create ML enhanced signal by combining traditional signal with ML prediction"""
        try:
            # Adjust profit prediction based on ML model
            ml_confidence = ml_prediction.get('confidence', 0.5)
            ml_direction = ml_prediction.get('direction', 1)  # 1 for up, -1 for down
            ml_magnitude = ml_prediction.get('magnitude', 0.0)
            
            # Only enhance if ML and traditional signals align
            if ml_direction < 0 and traditional_signal.expected_profit_pct > 0:
                # ML predicts down but traditional signal shows profit - reduce confidence
                ml_confidence *= 0.5
            
            # Adjust expected profit with ML prediction
            profit_adjustment = Decimal(str(ml_magnitude * 0.1))  # Conservative adjustment
            enhanced_profit_pct = traditional_signal.expected_profit_pct + profit_adjustment
            
            # Create enhanced signal
            enhanced_signal = ArbitrageSignal(
                signal_id=f"ml_{traditional_signal.signal_id}",
                signal_type=SignalType.ML_ENHANCED,
                timestamp=time.time(),
                symbol=traditional_signal.symbol,
                expected_profit_pct=enhanced_profit_pct,
                expected_profit_usd=traditional_signal.expected_profit_usd,
                confidence=float(ml_confidence),
                strength=traditional_signal.strength,
                buy_exchange=traditional_signal.buy_exchange,
                sell_exchange=traditional_signal.sell_exchange,
                buy_price=traditional_signal.buy_price,
                sell_price=traditional_signal.sell_price,
                size=traditional_signal.size,
                slippage_estimate=traditional_signal.slippage_estimate,
                transfer_time_penalty=traditional_signal.transfer_time_penalty,
                fees_total=traditional_signal.fees_total,
                mid_price_ref=traditional_signal.mid_price_ref,
                spread_width_pct=traditional_signal.spread_width_pct,
                volume_24h=traditional_signal.volume_24h,
                # ML specific fields
                ml_confidence=ml_confidence,
                ml_prediction=ml_prediction,
                traditional_signal_id=traditional_signal.signal_id
            )
            
            return enhanced_signal
            
        except Exception as e:
            logger.error(f"Error creating ML enhanced signal: {e}")
            return None
    
    def _is_valid_signal(self, signal: ArbitrageSignal) -> bool:
        """Validate signal before emission"""
        # Check minimum profit threshold
        if signal.net_profit_pct < Decimal('0.05'):
            return False
        
        # Check confidence threshold
        if signal.confidence < 0.3:
            return False
        
        # Check for duplicate signals (same opportunity within time window)
        current_time = time.time()
        for existing_signal in self.active_signals.values():
            if (existing_signal.symbol == signal.symbol and
                existing_signal.buy_exchange == signal.buy_exchange and
                existing_signal.sell_exchange == signal.sell_exchange and
                current_time - existing_signal.timestamp < 30):  # 30 second window
                return False
        
        return True
    
    def add_active_signal(self, signal: ArbitrageSignal):
        """Add signal to active tracking"""
        self.active_signals[signal.signal_id] = signal
        self.signal_history.append(signal)
    
    def remove_signal(self, signal_id: str):
        """Remove signal from active tracking"""
        self.active_signals.pop(signal_id, None)
    
    def get_active_signals(self, signal_type: Optional[SignalType] = None) -> List[ArbitrageSignal]:
        """Get currently active signals"""
        signals = list(self.active_signals.values())
        if signal_type:
            signals = [s for s in signals if s.signal_type == signal_type]
        return sorted(signals, key=lambda x: x.net_profit_pct, reverse=True)
    
    def cleanup_expired_signals(self, max_age: int = 300):
        """Remove expired signals (older than max_age seconds)"""
        current_time = time.time()
        expired_ids = [
            signal_id for signal_id, signal in self.active_signals.items()
            if current_time - signal.timestamp > max_age
        ]
        
        for signal_id in expired_ids:
            self.remove_signal(signal_id)
        
        if expired_ids:
            logger.info(f"Cleaned up {len(expired_ids)} expired signals")


# Example usage and testing
async def main():
    """Example usage of signal generation"""
    logging.basicConfig(level=logging.INFO)
    
    aggregator = SignalAggregator()
    
    # Simulate some market data updates
    from decimal import Decimal
    
    # Create sample order book
    order_book = OrderBook(
        exchange="binance",
        symbol="BTC/USDT",
        timestamp=time.time()
    )
    
    # Add some sample levels
    order_book.bids["50000"] = OrderBookLevel(price=Decimal("50000"), size=Decimal("1.0"))
    order_book.asks["50100"] = OrderBookLevel(price=Decimal("50100"), size=Decimal("0.5"))
    
    # Update aggregator
    aggregator.update_data("binance", order_book)
    
    # Generate signals
    async for signal in aggregator.generate_all_signals("BTC/USDT"):
        logger.info(f"Generated signal: {signal.signal_type} - {signal.expected_profit_pct}% profit")
        aggregator.add_active_signal(signal)
    
    # Show active signals
    active = aggregator.get_active_signals()
    logger.info(f"Active signals: {len(active)}")


if __name__ == "__main__":
    asyncio.run(main())
