"""
Portfolio Management Module

Tracks balances, positions, P&L, and multi-currency exposure across exchanges.
Provides comprehensive portfolio analytics and risk metrics.
"""

import asyncio
import logging
import time
from decimal import Decimal, ROUND_HALF_UP
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, field
from collections import defaultdict, deque
from datetime import datetime, timedelta
import pandas as pd

from .marketdata import Ticker
from .execution import Order, OrderStatus
from .risk import Position
from ..config.settings import get_settings

logger = logging.getLogger(__name__)


@dataclass
class Balance:
    """Account balance for a specific currency on an exchange"""
    exchange: str
    currency: str
    total: Decimal = Decimal('0')
    available: Decimal = Decimal('0')
    reserved: Decimal = Decimal('0')  # In orders
    
    # Value in USD (if known)
    usd_value: Optional[Decimal] = None
    last_price_usd: Optional[Decimal] = None
    
    last_updated: float = field(default_factory=time.time)
    
    @property
    def utilization_pct(self) -> float:
        """Calculate balance utilization percentage"""
        if self.total == 0:
            return 0.0
        return float(self.reserved / self.total * 100)


@dataclass
class PnLEntry:
    """P&L entry for historical tracking"""
    timestamp: float
    symbol: str
    exchange: str
    
    # Trade details
    quantity: Decimal
    price: Decimal
    side: str  # 'buy' or 'sell'
    
    # P&L calculation
    realized_pnl: Decimal = Decimal('0')
    unrealized_pnl: Decimal = Decimal('0')
    fees: Decimal = Decimal('0')
    
    # Reference data
    trade_id: Optional[str] = None
    order_id: Optional[str] = None
    signal_id: Optional[str] = None


@dataclass
class PortfolioSnapshot:
    """Point-in-time portfolio snapshot"""
    timestamp: float = field(default_factory=time.time)
    
    # Total values
    total_value_usd: Decimal = Decimal('0')
    total_cash_usd: Decimal = Decimal('0')
    total_positions_usd: Decimal = Decimal('0')
    
    # P&L metrics
    total_unrealized_pnl: Decimal = Decimal('0')
    total_realized_pnl: Decimal = Decimal('0')
    daily_pnl: Decimal = Decimal('0')
    
    # Performance metrics
    total_return_pct: float = 0.0
    daily_return_pct: float = 0.0
    sharpe_ratio: Optional[float] = None
    max_drawdown_pct: float = 0.0
    
    # Risk metrics
    var_95: Optional[Decimal] = None  # Value at Risk 95%
    leverage: float = 0.0
    
    # Position counts
    num_positions: int = 0
    num_currencies: int = 0
    
    # By exchange breakdown
    exchange_values: Dict[str, Decimal] = field(default_factory=dict)
    currency_exposure: Dict[str, Decimal] = field(default_factory=dict)


class ExchangePortfolio:
    """Portfolio manager for a single exchange"""
    
    def __init__(self, exchange: str):
        self.exchange = exchange
        self.balances: Dict[str, Balance] = {}
        self.positions: Dict[str, Position] = {}
        self.settings = get_settings()
        
    def update_balance(self, currency: str, total: Decimal, available: Decimal, reserved: Decimal = None):
        """Update balance for a currency"""
        if currency not in self.balances:
            self.balances[currency] = Balance(exchange=self.exchange, currency=currency)
        
        balance = self.balances[currency]
        balance.total = total
        balance.available = available
        balance.reserved = reserved if reserved is not None else (total - available)
        balance.last_updated = time.time()
        
        logger.debug(f"Updated {self.exchange} {currency} balance: {total} total, {available} available")
    
    def update_position(self, symbol: str, position: Position):
        """Update position"""
        self.positions[symbol] = position
        logger.debug(f"Updated {self.exchange} {symbol} position: {position.quantity} @ {position.average_price}")
    
    def get_balance(self, currency: str) -> Balance:
        """Get balance for currency"""
        if currency not in self.balances:
            self.balances[currency] = Balance(exchange=self.exchange, currency=currency)
        return self.balances[currency]
    
    def get_position(self, symbol: str) -> Optional[Position]:
        """Get position for symbol"""
        return self.positions.get(symbol)
    
    def get_total_value_usd(self, prices: Dict[str, Decimal]) -> Decimal:
        """Calculate total portfolio value in USD"""
        total_value = Decimal('0')
        
        # Add balance values
        for currency, balance in self.balances.items():
            if currency == 'USD' or currency == 'USDT':
                total_value += balance.total
            else:
                price_key = f"{currency}/USDT"
                if price_key in prices:
                    total_value += balance.total * prices[price_key]
        
        # Add position values
        for symbol, position in self.positions.items():
            if position.market_price > 0:
                total_value += abs(position.quantity * position.market_price)
        
        return total_value
    
    def calculate_unrealized_pnl(self) -> Decimal:
        """Calculate total unrealized P&L"""
        return sum(pos.unrealized_pnl for pos in self.positions.values())
    
    def calculate_realized_pnl(self) -> Decimal:
        """Calculate total realized P&L"""
        return sum(pos.realized_pnl for pos in self.positions.values())


class PortfolioManager:
    """Main portfolio manager coordinating all exchanges"""
    
    def __init__(self):
        self.settings = get_settings()
        self.exchange_portfolios: Dict[str, ExchangePortfolio] = {}
        
        # Historical data
        self.pnl_history: deque = deque(maxlen=10000)  # P&L entries
        self.portfolio_snapshots: deque = deque(maxlen=1440)  # 24h of minute snapshots
        self.daily_snapshots: deque = deque(maxlen=365)  # Daily snapshots for 1 year
        
        # Current market prices
        self.market_prices: Dict[str, Decimal] = {}
        
        # Performance tracking
        self.initial_portfolio_value: Optional[Decimal] = None
        self.last_snapshot_time = 0
        
    def get_exchange_portfolio(self, exchange: str) -> ExchangePortfolio:
        """Get or create exchange portfolio"""
        if exchange not in self.exchange_portfolios:
            self.exchange_portfolios[exchange] = ExchangePortfolio(exchange)
        return self.exchange_portfolios[exchange]
    
    def update_balance(self, exchange: str, currency: str, total: Decimal, available: Decimal, reserved: Decimal = None):
        """Update balance on an exchange"""
        portfolio = self.get_exchange_portfolio(exchange)
        portfolio.update_balance(currency, total, available, reserved)
    
    def update_position_from_order(self, order: Order, fill_quantity: Decimal, fill_price: Decimal):
        """Update position from order execution"""
        portfolio = self.get_exchange_portfolio(order.exchange)
        
        # Get or create position
        position = portfolio.get_position(order.symbol)
        if position is None:
            position = Position(symbol=order.symbol, exchange=order.exchange)
        
        # Apply trade to position
        trade_quantity = fill_quantity if order.side.value == 'buy' else -fill_quantity
        old_pnl = position.realized_pnl
        position.add_trade(trade_quantity, fill_price)
        portfolio.update_position(order.symbol, position)
        
        # Record P&L entry
        pnl_change = position.realized_pnl - old_pnl
        if pnl_change != 0:
            pnl_entry = PnLEntry(
                timestamp=time.time(),
                symbol=order.symbol,
                exchange=order.exchange,
                quantity=fill_quantity,
                price=fill_price,
                side=order.side.value,
                realized_pnl=pnl_change,
                order_id=order.order_id,
                signal_id=order.signal_id
            )
            self.pnl_history.append(pnl_entry)
        
        logger.info(f"Portfolio updated from order fill: {order.exchange}:{order.symbol} "
                   f"{fill_quantity} @ {fill_price}")
    
    def update_market_price(self, exchange: str, ticker: Ticker):
        """Update market prices from ticker data"""
        # Store price for portfolio valuation
        self.market_prices[ticker.symbol] = ticker.last_price or Decimal('0')
        
        # Update position market prices
        portfolio = self.get_exchange_portfolio(exchange)
        position = portfolio.get_position(ticker.symbol)
        if position and ticker.last_price:
            position.update_market_price(ticker.last_price)
    
    def get_current_snapshot(self) -> PortfolioSnapshot:
        """Generate current portfolio snapshot"""
        snapshot = PortfolioSnapshot()
        
        # Calculate totals across all exchanges
        total_value = Decimal('0')
        total_cash = Decimal('0')
        total_positions = Decimal('0')
        total_unrealized_pnl = Decimal('0')
        total_realized_pnl = Decimal('0')
        num_positions = 0
        currencies = set()
        
        exchange_values = {}
        currency_exposure = defaultdict(Decimal)
        
        for exchange, portfolio in self.exchange_portfolios.items():
            # Calculate exchange value
            exchange_value = portfolio.get_total_value_usd(self.market_prices)
            exchange_values[exchange] = exchange_value
            total_value += exchange_value
            
            # Add cash values
            for currency, balance in portfolio.balances.items():
                currencies.add(currency)
                if currency in ['USD', 'USDT', 'USDC']:
                    total_cash += balance.total
                    currency_exposure[currency] += balance.total
                else:
                    # Convert to USD equivalent
                    price_key = f"{currency}/USDT"
                    if price_key in self.market_prices:
                        usd_value = balance.total * self.market_prices[price_key]
                        currency_exposure[currency] += usd_value
            
            # Add position values and P&L
            for symbol, position in portfolio.positions.items():
                if position.quantity != 0:
                    num_positions += 1
                    if position.market_price > 0:
                        position_value = abs(position.quantity * position.market_price)
                        total_positions += position_value
                    
                    total_unrealized_pnl += position.unrealized_pnl
                    total_realized_pnl += position.realized_pnl
        
        # Set snapshot values
        snapshot.total_value_usd = total_value
        snapshot.total_cash_usd = total_cash
        snapshot.total_positions_usd = total_positions
        snapshot.total_unrealized_pnl = total_unrealized_pnl
        snapshot.total_realized_pnl = total_realized_pnl
        snapshot.daily_pnl = total_unrealized_pnl + total_realized_pnl  # Simplified
        snapshot.num_positions = num_positions
        snapshot.num_currencies = len(currencies)
        snapshot.exchange_values = exchange_values
        snapshot.currency_exposure = dict(currency_exposure)
        
        # Calculate performance metrics
        if self.initial_portfolio_value is None and total_value > 0:
            self.initial_portfolio_value = total_value
        
        if self.initial_portfolio_value and self.initial_portfolio_value > 0:
            snapshot.total_return_pct = float((total_value - self.initial_portfolio_value) / self.initial_portfolio_value * 100)
        
        # Calculate daily return
        if self.portfolio_snapshots:
            previous_snapshot = self.portfolio_snapshots[-1]
            if previous_snapshot.total_value_usd > 0:
                snapshot.daily_return_pct = float((total_value - previous_snapshot.total_value_usd) / previous_snapshot.total_value_usd * 100)
        
        # Calculate additional metrics
        snapshot.leverage = self._calculate_leverage(total_cash, total_positions)
        snapshot.max_drawdown_pct = self._calculate_max_drawdown()
        snapshot.sharpe_ratio = self._calculate_sharpe_ratio()
        
        return snapshot
    
    def _calculate_leverage(self, cash: Decimal, positions: Decimal) -> float:
        """Calculate portfolio leverage"""
        if cash == 0:
            return 0.0
        return float(positions / cash)
    
    def _calculate_max_drawdown(self) -> float:
        """Calculate maximum drawdown from historical snapshots"""
        if len(self.portfolio_snapshots) < 2:
            return 0.0
        
        values = [float(snap.total_value_usd) for snap in self.portfolio_snapshots]
        peak = values[0]
        max_drawdown = 0.0
        
        for value in values[1:]:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak * 100
            max_drawdown = max(max_drawdown, drawdown)
        
        return max_drawdown
    
    def _calculate_sharpe_ratio(self, periods: int = 252) -> Optional[float]:
        """Calculate Sharpe ratio from daily returns"""
        if len(self.daily_snapshots) < 30:  # Need at least 30 days
            return None
        
        returns = []
        for i in range(1, len(self.daily_snapshots)):
            prev_value = float(self.daily_snapshots[i-1].total_value_usd)
            curr_value = float(self.daily_snapshots[i].total_value_usd)
            if prev_value > 0:
                daily_return = (curr_value - prev_value) / prev_value
                returns.append(daily_return)
        
        if len(returns) < 2:
            return None
        
        # Calculate Sharpe ratio (assuming 0% risk-free rate)
        returns_array = pd.Series(returns)
        mean_return = returns_array.mean()
        std_return = returns_array.std()
        
        if std_return == 0:
            return None
        
        # Annualized Sharpe ratio
        return float(mean_return / std_return * (periods ** 0.5))
    
    def take_snapshot(self):
        """Take and store portfolio snapshot"""
        current_time = time.time()
        
        # Take snapshot
        snapshot = self.get_current_snapshot()
        self.portfolio_snapshots.append(snapshot)
        
        # Store daily snapshot (once per day)
        if (not self.daily_snapshots or 
            current_time - self.daily_snapshots[-1].timestamp > 24 * 3600):
            self.daily_snapshots.append(snapshot)
        
        self.last_snapshot_time = current_time
        
        logger.debug(f"Portfolio snapshot taken: Value=${snapshot.total_value_usd}, "
                    f"P&L=${snapshot.daily_pnl}")
    
    def get_balance(self, exchange: str, currency: str) -> Balance:
        """Get balance for specific exchange and currency"""
        portfolio = self.get_exchange_portfolio(exchange)
        return portfolio.get_balance(currency)
    
    def get_position(self, exchange: str, symbol: str) -> Optional[Position]:
        """Get position for specific exchange and symbol"""
        portfolio = self.get_exchange_portfolio(exchange)
        return portfolio.get_position(symbol)
    
    def get_all_positions(self) -> List[Tuple[str, Position]]:
        """Get all active positions across exchanges"""
        positions = []
        for exchange, portfolio in self.exchange_portfolios.items():
            for symbol, position in portfolio.positions.items():
                if position.quantity != 0:
                    positions.append((exchange, position))
        return positions
    
    def get_all_balances(self) -> List[Tuple[str, Balance]]:
        """Get all balances across exchanges"""
        balances = []
        for exchange, portfolio in self.exchange_portfolios.items():
            for currency, balance in portfolio.balances.items():
                if balance.total > 0:
                    balances.append((exchange, balance))
        return balances
    
    def get_pnl_summary(self, hours: int = 24) -> Dict[str, Decimal]:
        """Get P&L summary for specified time period"""
        cutoff_time = time.time() - (hours * 3600)
        recent_pnl = [entry for entry in self.pnl_history if entry.timestamp > cutoff_time]
        
        summary = {
            'total_realized_pnl': sum(entry.realized_pnl for entry in recent_pnl),
            'total_fees': sum(entry.fees for entry in recent_pnl),
            'gross_profit': sum(entry.realized_pnl for entry in recent_pnl if entry.realized_pnl > 0),
            'gross_loss': sum(entry.realized_pnl for entry in recent_pnl if entry.realized_pnl < 0),
            'trade_count': len(recent_pnl)
        }
        
        # Add current unrealized P&L
        current_unrealized = Decimal('0')
        for _, portfolio in self.exchange_portfolios.items():
            current_unrealized += portfolio.calculate_unrealized_pnl()
        
        summary['unrealized_pnl'] = current_unrealized
        summary['net_pnl'] = summary['total_realized_pnl'] + summary['unrealized_pnl']
        
        return summary
    
    def get_performance_metrics(self) -> Dict[str, float]:
        """Get comprehensive performance metrics"""
        if not self.portfolio_snapshots:
            return {}
        
        current_snapshot = self.portfolio_snapshots[-1]
        
        metrics = {
            'total_return_pct': current_snapshot.total_return_pct,
            'daily_return_pct': current_snapshot.daily_return_pct,
            'max_drawdown_pct': current_snapshot.max_drawdown_pct,
            'sharpe_ratio': current_snapshot.sharpe_ratio or 0.0,
            'leverage': current_snapshot.leverage,
            'num_positions': current_snapshot.num_positions,
            'num_currencies': current_snapshot.num_currencies
        }
        
        # Calculate win rate
        if self.pnl_history:
            profitable_trades = sum(1 for entry in self.pnl_history if entry.realized_pnl > 0)
            total_trades = len([entry for entry in self.pnl_history if entry.realized_pnl != 0])
            metrics['win_rate_pct'] = (profitable_trades / total_trades * 100) if total_trades > 0 else 0.0
        
        return metrics
    
    def get_currency_exposure(self) -> Dict[str, Decimal]:
        """Get exposure breakdown by currency"""
        snapshot = self.get_current_snapshot()
        return snapshot.currency_exposure
    
    def get_exchange_allocation(self) -> Dict[str, Decimal]:
        """Get portfolio allocation by exchange"""
        snapshot = self.get_current_snapshot()
        return snapshot.exchange_values
    
    async def start_monitoring(self, interval: int = 60):
        """Start automatic portfolio monitoring"""
        logger.info(f"Starting portfolio monitoring (interval: {interval}s)")
        
        while True:
            try:
                self.take_snapshot()
                await asyncio.sleep(interval)
            except Exception as e:
                logger.error(f"Error in portfolio monitoring: {e}")
                await asyncio.sleep(60)


# Example usage and testing
async def main():
    """Example usage of portfolio management"""
    logging.basicConfig(level=logging.INFO)
    
    portfolio_manager = PortfolioManager()
    
    # Simulate some balance updates
    portfolio_manager.update_balance("binance", "USDT", Decimal('10000'), Decimal('9000'), Decimal('1000'))
    portfolio_manager.update_balance("binance", "BTC", Decimal('0.5'), Decimal('0.5'))
    portfolio_manager.update_balance("kraken", "USD", Decimal('5000'), Decimal('5000'))
    
    # Simulate market price updates
    from .marketdata import Ticker
    ticker = Ticker(
        exchange="binance",
        symbol="BTC/USDT",
        last_price=Decimal('50000'),
        bid=Decimal('49995'),
        ask=Decimal('50005')
    )
    portfolio_manager.update_market_price("binance", ticker)
    
    # Take snapshot
    portfolio_manager.take_snapshot()
    snapshot = portfolio_manager.get_current_snapshot()
    
    print(f"Portfolio Value: ${snapshot.total_value_usd}")
    print(f"Total P&L: ${snapshot.daily_pnl}")
    print(f"Positions: {snapshot.num_positions}")
    print(f"Currencies: {snapshot.num_currencies}")
    
    # Get P&L summary
    pnl_summary = portfolio_manager.get_pnl_summary()
    print(f"Net P&L: ${pnl_summary['net_pnl']}")
    
    # Get performance metrics
    performance = portfolio_manager.get_performance_metrics()
    print(f"Performance metrics: {performance}")


if __name__ == "__main__":
    asyncio.run(main())
