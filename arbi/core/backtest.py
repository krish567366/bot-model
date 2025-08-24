"""
Backtesting Engine

Event-driven backtesting system with realistic latency modeling, slippage simulation,
and comprehensive performance analytics for arbitrage strategies.
"""

import asyncio
import logging
import time
import warnings
from collections import deque, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple, Any
import pandas as pd
import numpy as np
from statistics import mean, stdev

from .marketdata import BookDelta, Trade, Ticker, OrderBookLevel
from .signal import ArbitrageSignal, SignalType, SignalStrength
from .execution import Order, OrderType, OrderSide, OrderStatus, ExecutionGroup
from .risk import RiskLevel, RiskDecision, RiskAction
from .portfolio import Balance, Position, PnLEntry, PortfolioSnapshot
from .storage import StorageManager
from ..config.settings import get_settings

logger = logging.getLogger(__name__)

# Suppress warnings for cleaner backtest output
warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)


@dataclass
class BacktestConfig:
    """Backtesting configuration"""
    start_date: str
    end_date: str
    initial_balance: Dict[str, Decimal]
    exchanges: List[str]
    symbols: List[str]
    
    # Market microstructure simulation
    enable_latency_simulation: bool = True
    base_latency_ms: int = 10
    latency_variance_ms: int = 5
    
    # Slippage modeling
    enable_slippage: bool = True
    linear_slippage_factor: Decimal = Decimal("0.001")  # 0.1% per $10k notional
    impact_decay_halflife: int = 5000  # milliseconds
    
    # Fees
    maker_fee_pct: Decimal = Decimal("0.1")  # 0.1%
    taker_fee_pct: Decimal = Decimal("0.1")  # 0.1%
    
    # Execution parameters
    fill_probability: Decimal = Decimal("0.95")  # Probability of limit order fill
    partial_fill_min_pct: Decimal = Decimal("0.1")  # Min partial fill %
    
    # Risk controls
    enable_risk_management: bool = True
    max_position_usd: Decimal = Decimal("10000")
    max_daily_loss_usd: Decimal = Decimal("500")


@dataclass
class MarketImpact:
    """Market impact tracking for price simulation"""
    exchange: str
    symbol: str
    side: str
    impact_bps: Decimal  # basis points
    timestamp: float
    decay_rate: Decimal = Decimal("0.1")  # per second


@dataclass
class BacktestEvent:
    """Base class for backtest events"""
    timestamp: float
    event_type: str


@dataclass
class MarketDataEvent(BacktestEvent):
    """Market data update event"""
    book_delta: Optional[BookDelta] = None
    ticker: Optional[Ticker] = None
    trade: Optional[Trade] = None
    
    def __post_init__(self):
        self.event_type = "market_data"


@dataclass
class SignalEvent(BacktestEvent):
    """Trading signal event"""
    signal: ArbitrageSignal
    
    def __post_init__(self):
        self.event_type = "signal"


@dataclass
class OrderEvent(BacktestEvent):
    """Order execution event"""
    order: Order
    action: str  # "submit", "fill", "cancel"
    
    def __post_init__(self):
        self.event_type = "order"


@dataclass
class BacktestResult:
    """Comprehensive backtest results"""
    # Basic metrics
    start_date: str
    end_date: str
    duration_days: int
    total_trades: int
    winning_trades: int
    losing_trades: int
    
    # Returns
    total_return_pct: float
    annualized_return_pct: float
    daily_returns: List[float]
    cumulative_returns: List[float]
    
    # Risk metrics
    sharpe_ratio: Optional[float]
    max_drawdown_pct: float
    max_drawdown_duration_days: int
    volatility_pct: float
    var_95_pct: float  # Value at Risk
    
    # Trading metrics
    avg_trade_duration_minutes: float
    avg_profit_per_trade: Decimal
    profit_factor: float  # Gross profit / Gross loss
    win_rate_pct: float
    
    # Arbitrage specific
    successful_arbitrages: int
    failed_arbitrages: int
    avg_arbitrage_profit_bps: Decimal
    best_arbitrage_profit_bps: Decimal
    worst_arbitrage_profit_bps: Decimal
    
    # Costs
    total_fees_usd: Decimal
    total_slippage_usd: Decimal
    
    # Portfolio evolution
    portfolio_history: List[PortfolioSnapshot]
    trade_history: List[Dict]
    
    # Exchange breakdown
    exchange_performance: Dict[str, Dict[str, Any]]


class LatencySimulator:
    """Realistic latency simulation"""
    
    def __init__(self, config: BacktestConfig):
        self.config = config
        self.exchange_latencies = {
            'binance': {'base': 15, 'variance': 8},
            'kraken': {'base': 25, 'variance': 12},
            'coinbase': {'base': 20, 'variance': 10},
            'bybit': {'base': 18, 'variance': 9}
        }
    
    def get_latency(self, exchange: str) -> int:
        """Get simulated latency for exchange"""
        if not self.config.enable_latency_simulation:
            return 0
        
        params = self.exchange_latencies.get(exchange, {
            'base': self.config.base_latency_ms,
            'variance': self.config.latency_variance_ms
        })
        
        # Log-normal distribution for realistic latency
        base = params['base']
        variance = params['variance']
        latency = max(1, int(np.random.lognormal(np.log(base), variance / base)))
        
        return latency
    
    def network_congestion_factor(self, timestamp: float) -> float:
        """Simulate network congestion patterns"""
        # Higher latency during market volatility hours
        hour = datetime.fromtimestamp(timestamp).hour
        
        # Higher congestion during US/EU overlap (14:00-16:00 UTC)
        if 14 <= hour <= 16:
            return 1.5
        # Higher congestion during NY close (21:00-22:00 UTC)
        elif 21 <= hour <= 22:
            return 1.3
        # Lower congestion during low volume hours
        elif 6 <= hour <= 12:
            return 0.8
        else:
            return 1.0


class SlippageSimulator:
    """Market impact and slippage simulation"""
    
    def __init__(self, config: BacktestConfig):
        self.config = config
        self.market_impacts: Dict[str, List[MarketImpact]] = defaultdict(list)
        
    def calculate_slippage(self, order: Order, book_delta: BookDelta, 
                          timestamp: float) -> Tuple[Decimal, Decimal]:
        """Calculate slippage and effective price"""
        if not self.config.enable_slippage:
            return Decimal("0"), order.price
        
        # Get relevant order book levels
        levels = book_delta.asks if order.side == OrderSide.BUY else book_delta.bids
        if not levels:
            return Decimal("0"), order.price
        
        # Calculate order size impact
        notional_usd = order.quantity * order.price
        size_impact_bps = self.config.linear_slippage_factor * (notional_usd / Decimal("10000"))
        
        # Add historical market impact
        historical_impact = self._get_market_impact(
            order.exchange, order.symbol, order.side.value, timestamp
        )
        
        # Combine impacts
        total_impact_bps = size_impact_bps + historical_impact
        
        # Apply to price
        impact_multiplier = Decimal("1") + (total_impact_bps / Decimal("10000"))
        if order.side == OrderSide.BUY:
            effective_price = order.price * impact_multiplier
        else:
            effective_price = order.price / impact_multiplier
        
        slippage = abs(effective_price - order.price)
        
        # Record the impact
        self._record_impact(order, total_impact_bps, timestamp)
        
        return slippage, effective_price
    
    def _get_market_impact(self, exchange: str, symbol: str, side: str, timestamp: float) -> Decimal:
        """Get current market impact from recent trades"""
        key = f"{exchange}_{symbol}"
        impacts = self.market_impacts.get(key, [])
        
        total_impact = Decimal("0")
        
        for impact in impacts:
            # Calculate time decay
            age_seconds = timestamp - impact.timestamp
            if age_seconds > self.config.impact_decay_halflife / 1000:
                continue  # Impact has decayed
            
            decay_factor = Decimal(str(0.5)) ** (Decimal(str(age_seconds)) / 
                                                Decimal(str(self.config.impact_decay_halflife / 1000)))
            
            # Same side impacts add up, opposite side impacts reduce
            if impact.side == side:
                total_impact += impact.impact_bps * decay_factor
            else:
                total_impact -= impact.impact_bps * decay_factor * Decimal("0.3")
        
        return max(Decimal("0"), total_impact)
    
    def _record_impact(self, order: Order, impact_bps: Decimal, timestamp: float):
        """Record market impact for future calculations"""
        key = f"{order.exchange}_{order.symbol}"
        
        impact = MarketImpact(
            exchange=order.exchange,
            symbol=order.symbol,
            side=order.side.value,
            impact_bps=impact_bps,
            timestamp=timestamp
        )
        
        self.market_impacts[key].append(impact)
        
        # Clean up old impacts
        cutoff_time = timestamp - (self.config.impact_decay_halflife / 500)  # Keep 2x halflife
        self.market_impacts[key] = [
            imp for imp in self.market_impacts[key] 
            if imp.timestamp > cutoff_time
        ]


class BacktestEngine:
    """Main backtesting engine"""
    
    def __init__(self, config: BacktestConfig, storage: StorageManager):
        self.config = config
        self.storage = storage
        self.settings = get_settings()
        
        # Simulation components
        self.latency_sim = LatencySimulator(config)
        self.slippage_sim = SlippageSimulator(config)
        
        # State tracking
        self.current_time: float = 0
        self.balances: Dict[str, Dict[str, Decimal]] = defaultdict(lambda: defaultdict(Decimal))
        self.positions: Dict[str, Dict[str, Decimal]] = defaultdict(lambda: defaultdict(Decimal))
        self.order_books: Dict[str, BookDelta] = {}
        self.pending_orders: Dict[str, Order] = {}
        self.completed_orders: List[Order] = []
        self.trades: List[Dict] = []
        
        # Performance tracking
        self.portfolio_snapshots: List[PortfolioSnapshot] = []
        self.daily_pnl: List[Decimal] = []
        self.total_fees: Decimal = Decimal("0")
        self.total_slippage: Decimal = Decimal("0")
        
        # Event queue
        self.event_queue = deque()
        self.scheduled_events: List[Tuple[float, BacktestEvent]] = []
        
        # Initialize balances
        self._initialize_balances()
    
    def _initialize_balances(self):
        """Initialize starting balances"""
        for exchange in self.config.exchanges:
            for currency, amount in self.config.initial_balance.items():
                self.balances[exchange][currency] = amount
    
    async def run_backtest(self, data_source: str = None) -> BacktestResult:
        """Execute the backtest"""
        logger.info(f"Starting backtest: {self.config.start_date} to {self.config.end_date}")
        
        # Load historical data
        await self._load_data(data_source)
        
        # Process all events
        await self._process_events()
        
        # Generate results
        result = await self._generate_results()
        
        logger.info(f"Backtest completed: {len(self.completed_orders)} orders, "
                   f"{result.total_return_pct:.2f}% return")
        
        return result
    
    async def _load_data(self, data_source: str = None):
        """Load historical market data"""
        logger.info("Loading historical data...")
        
        start_date = datetime.strptime(self.config.start_date, "%Y-%m-%d")
        end_date = datetime.strptime(self.config.end_date, "%Y-%m-%d")
        
        events = []
        
        for exchange in self.config.exchanges:
            for symbol in self.config.symbols:
                try:
                    # Load from storage or generate synthetic data
                    if data_source and self.storage:
                        df = await self.storage.get_market_data(
                            exchange, symbol, 
                            self.config.start_date, self.config.end_date
                        )
                        
                        if not df.empty:
                            events.extend(self._df_to_events(df, exchange, symbol))
                            continue
                    
                    # Generate synthetic data if no historical data
                    logger.warning(f"No historical data for {exchange}:{symbol}, generating synthetic data")
                    events.extend(self._generate_synthetic_data(exchange, symbol, start_date, end_date))
                
                except Exception as e:
                    logger.error(f"Error loading data for {exchange}:{symbol}: {e}")
        
        # Sort events by timestamp
        events.sort(key=lambda x: x.timestamp)
        self.event_queue.extend(events)
        
        logger.info(f"Loaded {len(events)} market data events")
    
    def _df_to_events(self, df: pd.DataFrame, exchange: str, symbol: str) -> List[BacktestEvent]:
        """Convert DataFrame to market data events"""
        events = []
        
        for _, row in df.iterrows():
            # Create book delta from row data
            bids = []
            asks = []
            
            # Reconstruct order book levels
            if 'bid_prices' in row and 'bid_sizes' in row:
                for price, size in zip(row['bid_prices'], row['bid_sizes']):
                    bids.append(OrderBookLevel(price=Decimal(str(price)), size=Decimal(str(size))))
            
            if 'ask_prices' in row and 'ask_sizes' in row:
                for price, size in zip(row['ask_prices'], row['ask_sizes']):
                    asks.append(OrderBookLevel(price=Decimal(str(price)), size=Decimal(str(size))))
            
            book_delta = BookDelta(
                exchange=exchange,
                symbol=symbol,
                bids=bids,
                asks=asks,
                sequence=row.get('sequence', 0),
                is_snapshot=row.get('is_snapshot', False)
            )
            
            event = MarketDataEvent(
                timestamp=row['timestamp'].timestamp(),
                book_delta=book_delta
            )
            events.append(event)
        
        return events
    
    def _generate_synthetic_data(self, exchange: str, symbol: str, 
                                start_date: datetime, end_date: datetime) -> List[BacktestEvent]:
        """Generate synthetic market data for backtesting"""
        events = []
        
        # Start with base price (e.g., BTC at $50,000)
        base_prices = {
            'BTC/USDT': 50000,
            'ETH/USDT': 3000,
            'BNB/USDT': 300,
            'ADA/USDT': 0.5,
            'SOL/USDT': 100
        }
        
        base_price = Decimal(str(base_prices.get(symbol, 1000)))
        current_price = base_price
        
        # Generate price walk
        current_dt = start_date
        interval_minutes = 1
        
        while current_dt <= end_date:
            timestamp = current_dt.timestamp()
            
            # Random price movement (geometric brownian motion)
            drift = Decimal("0.00001")  # Small upward drift
            volatility = Decimal("0.02")  # 2% volatility
            
            random_factor = Decimal(str(np.random.normal(0, 1)))
            price_change = current_price * (drift + volatility * random_factor)
            current_price = max(current_price + price_change, current_price * Decimal("0.99"))
            
            # Create order book
            spread_pct = Decimal("0.001")  # 0.1% spread
            spread = current_price * spread_pct
            
            bid_price = current_price - spread / 2
            ask_price = current_price + spread / 2
            
            # Random sizes
            bid_size = Decimal(str(np.random.uniform(0.1, 5.0)))
            ask_size = Decimal(str(np.random.uniform(0.1, 5.0)))
            
            book_delta = BookDelta(
                exchange=exchange,
                symbol=symbol,
                bids=[OrderBookLevel(price=bid_price, size=bid_size)],
                asks=[OrderBookLevel(price=ask_price, size=ask_size)],
                sequence=int(timestamp),
                is_snapshot=True
            )
            
            event = MarketDataEvent(
                timestamp=timestamp,
                book_delta=book_delta
            )
            events.append(event)
            
            current_dt += timedelta(minutes=interval_minutes)
        
        return events
    
    async def _process_events(self):
        """Process all events in chronological order"""
        logger.info("Processing backtest events...")
        
        event_count = 0
        last_progress_log = 0
        
        while self.event_queue:
            event = self.event_queue.popleft()
            self.current_time = event.timestamp
            
            # Process scheduled events first
            while (self.scheduled_events and 
                   self.scheduled_events[0][0] <= self.current_time):
                _, scheduled_event = self.scheduled_events.pop(0)
                await self._handle_event(scheduled_event)
            
            # Process current event
            await self._handle_event(event)
            
            event_count += 1
            if event_count - last_progress_log >= 10000:
                logger.info(f"Processed {event_count} events")
                last_progress_log = event_count
                
                # Take portfolio snapshot every 1000 events
                if event_count % 1000 == 0:
                    await self._take_portfolio_snapshot()
        
        # Final portfolio snapshot
        await self._take_portfolio_snapshot()
        
        logger.info(f"Processed {event_count} total events")
    
    async def _handle_event(self, event: BacktestEvent):
        """Handle individual events"""
        if event.event_type == "market_data":
            await self._handle_market_data(event)
        elif event.event_type == "signal":
            await self._handle_signal(event)
        elif event.event_type == "order":
            await self._handle_order(event)
    
    async def _handle_market_data(self, event: MarketDataEvent):
        """Handle market data updates"""
        if event.book_delta:
            key = f"{event.book_delta.exchange}_{event.book_delta.symbol}"
            self.order_books[key] = event.book_delta
            
            # Check for arbitrage opportunities
            await self._check_arbitrage_opportunities()
            
            # Update pending orders
            await self._update_pending_orders(event.book_delta)
    
    async def _check_arbitrage_opportunities(self):
        """Check for arbitrage opportunities across exchanges"""
        # Simple cross-exchange arbitrage detection
        symbols_by_exchange = defaultdict(dict)
        
        for key, book_delta in self.order_books.items():
            exchange, symbol = key.split("_", 1)
            if book_delta.bids and book_delta.asks:
                symbols_by_exchange[symbol][exchange] = book_delta
        
        for symbol, exchange_books in symbols_by_exchange.items():
            if len(exchange_books) < 2:
                continue
            
            # Find best bid and ask across exchanges
            best_bid = None
            best_ask = None
            best_bid_exchange = None
            best_ask_exchange = None
            
            for exchange, book_delta in exchange_books.items():
                if book_delta.bids and (best_bid is None or book_delta.bids[0].price > best_bid):
                    best_bid = book_delta.bids[0].price
                    best_bid_exchange = exchange
                
                if book_delta.asks and (best_ask is None or book_delta.asks[0].price < best_ask):
                    best_ask = book_delta.asks[0].price
                    best_ask_exchange = exchange
            
            # Check for arbitrage opportunity
            if (best_bid and best_ask and best_bid_exchange != best_ask_exchange and 
                best_bid > best_ask):
                
                profit_pct = ((best_bid - best_ask) / best_ask) * 100
                
                if profit_pct >= self.settings.strategy.min_arbitrage_spread:
                    # Generate signal
                    signal = ArbitrageSignal(
                        signal_type=SignalType.CROSS_EXCHANGE,
                        symbol=symbol,
                        buy_exchange=best_ask_exchange,
                        sell_exchange=best_bid_exchange,
                        buy_price=best_ask,
                        sell_price=best_bid,
                        size=min(exchange_books[best_ask_exchange].asks[0].size,
                               exchange_books[best_bid_exchange].bids[0].size),
                        expected_profit_pct=profit_pct,
                        confidence=0.8,
                        strength=SignalStrength.MEDIUM,
                        timestamp=self.current_time
                    )
                    
                    # Schedule signal event
                    signal_event = SignalEvent(timestamp=self.current_time, signal=signal)
                    await self._handle_signal(signal_event)
    
    async def _handle_signal(self, event: SignalEvent):
        """Handle trading signal"""
        signal = event.signal
        
        # Risk management check
        if self.config.enable_risk_management:
            if not await self._check_risk_limits(signal):
                return
        
        # Create orders
        buy_order = Order(
            exchange=signal.buy_exchange,
            symbol=signal.symbol,
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=signal.size,
            price=signal.buy_price,
            signal_id=signal.signal_id
        )
        
        sell_order = Order(
            exchange=signal.sell_exchange,
            symbol=signal.symbol,
            side=OrderSide.SELL,
            order_type=OrderType.LIMIT,
            quantity=signal.size,
            price=signal.sell_price,
            signal_id=signal.signal_id
        )
        
        # Submit orders with latency
        buy_latency = self.latency_sim.get_latency(signal.buy_exchange)
        sell_latency = self.latency_sim.get_latency(signal.sell_exchange)
        
        # Schedule order events
        buy_event = OrderEvent(
            timestamp=self.current_time + buy_latency / 1000,
            order=buy_order,
            action="submit"
        )
        
        sell_event = OrderEvent(
            timestamp=self.current_time + sell_latency / 1000,
            order=sell_order,
            action="submit"
        )
        
        self._schedule_event(buy_event)
        self._schedule_event(sell_event)
    
    async def _handle_order(self, event: OrderEvent):
        """Handle order events"""
        order = event.order
        action = event.action
        
        if action == "submit":
            # Add to pending orders
            self.pending_orders[order.order_id] = order
            order.status = OrderStatus.PENDING
            order.submitted_time = event.timestamp
            
            # Try immediate fill for limit orders
            await self._try_fill_order(order)
        
        elif action == "fill":
            await self._fill_order(order)
        
        elif action == "cancel":
            self._cancel_order(order)
    
    async def _try_fill_order(self, order: Order):
        """Try to fill order against order book"""
        key = f"{order.exchange}_{order.symbol}"
        book_delta = self.order_books.get(key)
        
        if not book_delta:
            return
        
        # Check if order can be filled
        can_fill = False
        fill_price = order.price
        
        if order.side == OrderSide.BUY and book_delta.asks:
            best_ask = book_delta.asks[0].price
            if order.price >= best_ask:
                can_fill = True
                fill_price = best_ask
        
        elif order.side == OrderSide.SELL and book_delta.bids:
            best_bid = book_delta.bids[0].price
            if order.price <= best_bid:
                can_fill = True
                fill_price = best_bid
        
        # Probability-based fill
        if can_fill and np.random.random() < float(self.config.fill_probability):
            # Calculate slippage
            slippage, effective_price = self.slippage_sim.calculate_slippage(
                order, book_delta, self.current_time
            )
            
            # Partial fill simulation
            fill_quantity = order.quantity
            if np.random.random() < 0.1:  # 10% chance of partial fill
                min_fill = order.quantity * self.config.partial_fill_min_pct
                fill_quantity = Decimal(str(np.random.uniform(float(min_fill), float(order.quantity))))
            
            # Update order
            order.filled_quantity = fill_quantity
            order.average_fill_price = effective_price
            order.status = OrderStatus.FILLED if fill_quantity == order.quantity else OrderStatus.PARTIALLY_FILLED
            order.filled_time = self.current_time
            
            # Calculate fees
            fee_pct = self.config.taker_fee_pct if effective_price != order.price else self.config.maker_fee_pct
            fee = fill_quantity * effective_price * fee_pct / Decimal("100")
            
            # Update balances
            base_currency, quote_currency = order.symbol.split("/")
            
            if order.side == OrderSide.BUY:
                self.balances[order.exchange][quote_currency] -= fill_quantity * effective_price + fee
                self.balances[order.exchange][base_currency] += fill_quantity
            else:
                self.balances[order.exchange][base_currency] -= fill_quantity
                self.balances[order.exchange][quote_currency] += fill_quantity * effective_price - fee
            
            # Update positions
            position_change = fill_quantity if order.side == OrderSide.BUY else -fill_quantity
            self.positions[order.exchange][base_currency] += position_change
            
            # Track costs
            self.total_fees += fee
            self.total_slippage += slippage * fill_quantity
            
            # Record trade
            trade_record = {
                'timestamp': self.current_time,
                'exchange': order.exchange,
                'symbol': order.symbol,
                'side': order.side.value,
                'quantity': fill_quantity,
                'price': effective_price,
                'fees': fee,
                'slippage': slippage,
                'signal_id': order.signal_id,
                'order_id': order.order_id
            }
            self.trades.append(trade_record)
            
            # Move to completed orders
            if order.status == OrderStatus.FILLED:
                self.completed_orders.append(order)
                del self.pending_orders[order.order_id]
    
    async def _update_pending_orders(self, book_delta: BookDelta):
        """Update pending orders against new market data"""
        orders_to_check = [
            order for order in self.pending_orders.values()
            if order.exchange == book_delta.exchange and order.symbol == book_delta.symbol
        ]
        
        for order in orders_to_check:
            await self._try_fill_order(order)
    
    async def _check_risk_limits(self, signal: ArbitrageSignal) -> bool:
        """Check risk management limits"""
        # Position size check
        notional = signal.size * signal.buy_price
        if notional > self.config.max_position_usd:
            return False
        
        # Daily loss check
        today_pnl = self._calculate_daily_pnl()
        if abs(today_pnl) > self.config.max_daily_loss_usd:
            return False
        
        return True
    
    def _calculate_daily_pnl(self) -> Decimal:
        """Calculate current day P&L"""
        today_start = datetime.fromtimestamp(self.current_time).replace(
            hour=0, minute=0, second=0, microsecond=0
        ).timestamp()
        
        today_trades = [
            trade for trade in self.trades
            if trade['timestamp'] >= today_start
        ]
        
        pnl = Decimal("0")
        for trade in today_trades:
            # Simplified P&L calculation
            if trade['side'] == 'BUY':
                pnl -= trade['quantity'] * trade['price'] + trade['fees']
            else:
                pnl += trade['quantity'] * trade['price'] - trade['fees']
        
        return pnl
    
    def _schedule_event(self, event: BacktestEvent):
        """Schedule event for future processing"""
        self.scheduled_events.append((event.timestamp, event))
        self.scheduled_events.sort(key=lambda x: x[0])
    
    def _cancel_order(self, order: Order):
        """Cancel pending order"""
        if order.order_id in self.pending_orders:
            order.status = OrderStatus.CANCELLED
            del self.pending_orders[order.order_id]
    
    async def _take_portfolio_snapshot(self):
        """Take portfolio snapshot for tracking"""
        total_value = self._calculate_portfolio_value()
        
        # Create simplified snapshot
        snapshot = PortfolioSnapshot(
            timestamp=self.current_time,
            total_value_usd=total_value,
            total_cash_usd=total_value,  # Simplified
            total_positions_usd=Decimal("0"),
            total_unrealized_pnl=Decimal("0"),
            total_realized_pnl=self._calculate_realized_pnl(),
            daily_pnl=self._calculate_daily_pnl(),
            num_positions=len(self.trades),
            leverage=1.0,
            exchange_values={},
            currency_exposure={}
        )
        
        self.portfolio_snapshots.append(snapshot)
    
    def _calculate_portfolio_value(self) -> Decimal:
        """Calculate total portfolio value in USD"""
        total_value = Decimal("0")
        
        for exchange, currencies in self.balances.items():
            for currency, amount in currencies.items():
                if currency == "USDT" or currency == "USD":
                    total_value += amount
                else:
                    # Use last known price or assume 1:1 for simplicity
                    total_value += amount  # Simplified
        
        return total_value
    
    def _calculate_realized_pnl(self) -> Decimal:
        """Calculate realized P&L"""
        pnl = Decimal("0")
        
        for trade in self.trades:
            if trade['side'] == 'SELL':
                pnl += trade['quantity'] * trade['price'] - trade['fees']
            else:
                pnl -= trade['quantity'] * trade['price'] + trade['fees']
        
        return pnl
    
    async def _generate_results(self) -> BacktestResult:
        """Generate comprehensive backtest results"""
        logger.info("Generating backtest results...")
        
        # Basic metrics
        start_date = datetime.strptime(self.config.start_date, "%Y-%m-%d")
        end_date = datetime.strptime(self.config.end_date, "%Y-%m-%d")
        duration_days = (end_date - start_date).days
        
        total_trades = len(self.completed_orders)
        winning_trades = sum(1 for trade in self.trades if trade.get('pnl', 0) > 0)
        losing_trades = total_trades - winning_trades
        
        # Returns calculation
        initial_value = sum(sum(currencies.values()) for currencies in self.config.initial_balance.values())
        final_value = self._calculate_portfolio_value()
        total_return_pct = float((final_value - initial_value) / initial_value * 100)
        annualized_return_pct = total_return_pct * (365.25 / max(duration_days, 1))
        
        # Daily returns
        daily_returns = []
        cumulative_returns = []
        
        if len(self.portfolio_snapshots) > 1:
            prev_value = self.portfolio_snapshots[0].total_value_usd
            cum_return = 0
            
            for snapshot in self.portfolio_snapshots[1:]:
                daily_return = float((snapshot.total_value_usd - prev_value) / prev_value * 100)
                daily_returns.append(daily_return)
                cum_return += daily_return
                cumulative_returns.append(cum_return)
                prev_value = snapshot.total_value_usd
        
        # Risk metrics
        sharpe_ratio = None
        if daily_returns and len(daily_returns) > 1:
            avg_return = mean(daily_returns)
            return_std = stdev(daily_returns)
            if return_std > 0:
                sharpe_ratio = (avg_return * np.sqrt(252)) / (return_std * np.sqrt(252))
        
        # Max drawdown
        max_drawdown_pct = 0.0
        max_drawdown_duration_days = 0
        peak_value = initial_value
        
        for snapshot in self.portfolio_snapshots:
            if snapshot.total_value_usd > peak_value:
                peak_value = snapshot.total_value_usd
            else:
                drawdown = float((peak_value - snapshot.total_value_usd) / peak_value * 100)
                max_drawdown_pct = max(max_drawdown_pct, drawdown)
        
        volatility_pct = stdev(daily_returns) * np.sqrt(252) if len(daily_returns) > 1 else 0.0
        var_95_pct = np.percentile(daily_returns, 5) if daily_returns else 0.0
        
        # Trading metrics
        avg_trade_duration_minutes = 5.0  # Simplified for arbitrage
        avg_profit_per_trade = self._calculate_realized_pnl() / max(total_trades, 1)
        
        gross_profit = sum(max(Decimal("0"), trade.get('pnl', 0)) for trade in self.trades)
        gross_loss = sum(min(Decimal("0"), trade.get('pnl', 0)) for trade in self.trades)
        profit_factor = float(abs(gross_profit / gross_loss)) if gross_loss != 0 else float('inf')
        
        win_rate_pct = (winning_trades / max(total_trades, 1)) * 100
        
        # Arbitrage specific metrics
        successful_arbitrages = sum(1 for order in self.completed_orders 
                                  if order.status == OrderStatus.FILLED)
        failed_arbitrages = total_trades - successful_arbitrages
        
        # Create result
        result = BacktestResult(
            start_date=self.config.start_date,
            end_date=self.config.end_date,
            duration_days=duration_days,
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            total_return_pct=total_return_pct,
            annualized_return_pct=annualized_return_pct,
            daily_returns=daily_returns,
            cumulative_returns=cumulative_returns,
            sharpe_ratio=sharpe_ratio,
            max_drawdown_pct=max_drawdown_pct,
            max_drawdown_duration_days=max_drawdown_duration_days,
            volatility_pct=volatility_pct,
            var_95_pct=var_95_pct,
            avg_trade_duration_minutes=avg_trade_duration_minutes,
            avg_profit_per_trade=avg_profit_per_trade,
            profit_factor=profit_factor,
            win_rate_pct=win_rate_pct,
            successful_arbitrages=successful_arbitrages,
            failed_arbitrages=failed_arbitrages,
            avg_arbitrage_profit_bps=Decimal("0"),
            best_arbitrage_profit_bps=Decimal("0"),
            worst_arbitrage_profit_bps=Decimal("0"),
            total_fees_usd=self.total_fees,
            total_slippage_usd=self.total_slippage,
            portfolio_history=self.portfolio_snapshots,
            trade_history=self.trades,
            exchange_performance={}
        )
        
        return result


# Example usage
async def main():
    """Example backtest execution"""
    import logging
    from .storage import StorageManager
    
    logging.basicConfig(level=logging.INFO)
    
    # Configure backtest
    config = BacktestConfig(
        start_date="2024-01-01",
        end_date="2024-01-07",
        initial_balance={"USDT": Decimal("10000")},
        exchanges=["binance", "kraken"],
        symbols=["BTC/USDT", "ETH/USDT"],
        enable_latency_simulation=True,
        enable_slippage=True
    )
    
    # Initialize storage
    storage = StorageManager()
    await storage.initialize()
    
    # Run backtest
    engine = BacktestEngine(config, storage)
    result = await engine.run_backtest()
    
    # Print results
    print(f"Backtest Results:")
    print(f"Total Return: {result.total_return_pct:.2f}%")
    print(f"Annualized Return: {result.annualized_return_pct:.2f}%")
    print(f"Sharpe Ratio: {result.sharpe_ratio:.3f}" if result.sharpe_ratio else "N/A")
    print(f"Max Drawdown: {result.max_drawdown_pct:.2f}%")
    print(f"Total Trades: {result.total_trades}")
    print(f"Win Rate: {result.win_rate_pct:.1f}%")
    print(f"Total Fees: ${result.total_fees_usd}")


if __name__ == "__main__":
    asyncio.run(main())
