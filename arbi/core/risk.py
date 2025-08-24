"""
Risk Management Module

Implements comprehensive risk management including:
- Position limits and exposure tracking
- Circuit breakers and kill switches
- Real-time risk monitoring and alerts
- Audit logging of all risk decisions
"""

import asyncio
import logging
import time
import json
from decimal import Decimal
from typing import Dict, List, Optional, Set, Callable, Any, NamedTuple
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import uuid

from .marketdata import OrderBook, Ticker
from .signal import ArbitrageSignal, SignalType
from .execution import Order, ExecutionGroup, OrderStatus
from ..config.settings import get_settings

logger = logging.getLogger(__name__)


class RiskLevel(str, Enum):
    """Risk level classification"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class RiskAction(str, Enum):
    """Risk management actions"""
    ALLOW = "allow"
    REDUCE_SIZE = "reduce_size"
    BLOCK = "block"
    CIRCUIT_BREAKER = "circuit_breaker"
    KILL_SWITCH = "kill_switch"


class AlertType(str, Enum):
    """Risk alert types"""
    POSITION_LIMIT = "position_limit"
    DAILY_LOSS = "daily_loss"
    SLIPPAGE_BREACH = "slippage_breach"
    CIRCUIT_BREAKER_TRIGGERED = "circuit_breaker_triggered"
    KILL_SWITCH_ACTIVATED = "kill_switch_activated"
    EXPOSURE_LIMIT = "exposure_limit"
    CONCENTRATION_RISK = "concentration_risk"


@dataclass
class RiskMetrics:
    """Current risk metrics snapshot"""
    timestamp: float = field(default_factory=time.time)
    
    # Portfolio metrics
    total_portfolio_value: Decimal = Decimal('0')
    unrealized_pnl: Decimal = Decimal('0')
    realized_pnl: Decimal = Decimal('0')
    daily_pnl: Decimal = Decimal('0')
    
    # Exposure metrics
    total_gross_exposure: Decimal = Decimal('0')
    total_net_exposure: Decimal = Decimal('0')
    max_single_position: Decimal = Decimal('0')
    
    # Risk ratios
    leverage_ratio: float = 0.0
    concentration_ratio: float = 0.0  # Largest position / total portfolio
    
    # Counts
    active_positions: int = 0
    active_orders: int = 0
    
    # Circuit breaker status
    circuit_breaker_active: bool = False
    kill_switch_active: bool = False


@dataclass
class RiskAlert:
    """Risk alert message"""
    alert_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: float = field(default_factory=time.time)
    alert_type: AlertType = AlertType.POSITION_LIMIT
    severity: RiskLevel = RiskLevel.MEDIUM
    
    # Alert details
    message: str = ""
    symbol: Optional[str] = None
    exchange: Optional[str] = None
    current_value: Optional[Decimal] = None
    limit_value: Optional[Decimal] = None
    
    # Context
    signal_id: Optional[str] = None
    order_id: Optional[str] = None
    
    # Action taken
    action_taken: RiskAction = RiskAction.ALLOW
    resolved: bool = False


@dataclass
class Position:
    """Trading position tracking"""
    symbol: str
    exchange: str
    
    # Position details
    quantity: Decimal = Decimal('0')  # Positive = long, negative = short
    average_price: Decimal = Decimal('0')
    market_price: Decimal = Decimal('0')
    
    # P&L tracking
    realized_pnl: Decimal = Decimal('0')
    unrealized_pnl: Decimal = Decimal('0')
    
    # Risk metrics
    notional_value: Decimal = Decimal('0')
    max_quantity: Decimal = Decimal('0')  # High water mark
    
    # Timestamps
    opened_time: float = field(default_factory=time.time)
    last_updated: float = field(default_factory=time.time)
    
    def update_market_price(self, price: Decimal):
        """Update position with new market price"""
        self.market_price = price
        self.last_updated = time.time()
        
        # Calculate notional value and unrealized P&L
        if self.quantity != 0:
            self.notional_value = abs(self.quantity * price)
            self.unrealized_pnl = (price - self.average_price) * self.quantity
    
    def add_trade(self, quantity: Decimal, price: Decimal):
        """Add a trade to the position"""
        if self.quantity == 0:
            # Opening position
            self.quantity = quantity
            self.average_price = price
        elif (self.quantity > 0 and quantity > 0) or (self.quantity < 0 and quantity < 0):
            # Adding to position
            total_cost = self.quantity * self.average_price + quantity * price
            self.quantity += quantity
            if self.quantity != 0:
                self.average_price = total_cost / self.quantity
        else:
            # Reducing or closing position
            if abs(quantity) >= abs(self.quantity):
                # Closing position completely
                self.realized_pnl += (price - self.average_price) * self.quantity
                self.quantity = quantity - self.quantity  # Remaining quantity
                if self.quantity != 0:
                    self.average_price = price
            else:
                # Partial close
                self.realized_pnl += (price - self.average_price) * quantity
                self.quantity += quantity  # quantity is negative for closing
        
        # Update max quantity (high water mark)
        self.max_quantity = max(self.max_quantity, abs(self.quantity))
        self.last_updated = time.time()


@dataclass
class RiskDecision:
    """Risk assessment decision for audit trail"""
    decision_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: float = field(default_factory=time.time)
    
    # Decision context
    signal_id: Optional[str] = None
    order_id: Optional[str] = None
    symbol: str = ""
    exchange: str = ""
    
    # Risk assessment
    risk_level: RiskLevel = RiskLevel.LOW
    action: RiskAction = RiskAction.ALLOW
    reasoning: str = ""
    
    # Risk factors considered
    risk_factors: Dict[str, Any] = field(default_factory=dict)
    
    # Decision outcome
    original_size: Optional[Decimal] = None
    approved_size: Optional[Decimal] = None
    rejected: bool = False


class PositionManager:
    """Manages trading positions and P&L tracking"""
    
    def __init__(self):
        self.positions: Dict[str, Position] = {}  # Key: exchange:symbol
        self.daily_pnl_history: deque = deque(maxlen=365)  # Daily P&L history
        self.settings = get_settings()
        
    def get_position_key(self, exchange: str, symbol: str) -> str:
        """Generate position key"""
        return f"{exchange}:{symbol}"
    
    def get_position(self, exchange: str, symbol: str) -> Position:
        """Get or create position"""
        key = self.get_position_key(exchange, symbol)
        if key not in self.positions:
            self.positions[key] = Position(symbol=symbol, exchange=exchange)
        return self.positions[key]
    
    def update_position_from_fill(self, order: Order, fill_quantity: Decimal, fill_price: Decimal):
        """Update position from order fill"""
        position = self.get_position(order.exchange, order.symbol)
        
        # Convert order side to quantity
        trade_quantity = fill_quantity if order.side.value == 'buy' else -fill_quantity
        position.add_trade(trade_quantity, fill_price)
        
        logger.info(f"Updated position {order.exchange}:{order.symbol}: "
                   f"qty={position.quantity}, avg_price={position.average_price}")
    
    def update_market_prices(self, exchange: str, ticker: Ticker):
        """Update positions with current market prices"""
        position = self.get_position(exchange, ticker.symbol)
        if ticker.last_price:
            position.update_market_price(ticker.last_price)
    
    def get_all_positions(self) -> List[Position]:
        """Get all non-zero positions"""
        return [pos for pos in self.positions.values() if pos.quantity != 0]
    
    def get_portfolio_metrics(self) -> RiskMetrics:
        """Calculate current portfolio risk metrics"""
        metrics = RiskMetrics()
        
        total_notional = Decimal('0')
        total_unrealized_pnl = Decimal('0')
        total_realized_pnl = Decimal('0')
        max_position_value = Decimal('0')
        position_count = 0
        
        for position in self.positions.values():
            if position.quantity != 0:
                position_count += 1
                total_notional += position.notional_value
                total_unrealized_pnl += position.unrealized_pnl
                total_realized_pnl += position.realized_pnl
                max_position_value = max(max_position_value, position.notional_value)
        
        metrics.total_portfolio_value = total_notional
        metrics.unrealized_pnl = total_unrealized_pnl
        metrics.realized_pnl = total_realized_pnl
        metrics.daily_pnl = total_unrealized_pnl + total_realized_pnl  # Simplified
        metrics.total_gross_exposure = total_notional
        metrics.max_single_position = max_position_value
        metrics.active_positions = position_count
        
        # Calculate ratios
        if total_notional > 0:
            metrics.concentration_ratio = float(max_position_value / total_notional)
        
        return metrics
    
    def calculate_position_impact(self, exchange: str, symbol: str, quantity: Decimal, price: Decimal) -> Decimal:
        """Calculate impact of new trade on position"""
        position = self.get_position(exchange, symbol)
        
        # Simulate the trade
        current_notional = abs(position.quantity * position.market_price) if position.quantity != 0 else Decimal('0')
        new_quantity = position.quantity + (quantity if quantity > 0 else -abs(quantity))
        new_notional = abs(new_quantity * price)
        
        return new_notional - current_notional


class RiskManager:
    """Main risk management coordinator"""
    
    def __init__(self):
        self.settings = get_settings()
        self.position_manager = PositionManager()
        
        # Risk state
        self.circuit_breaker_active = False
        self.kill_switch_active = False
        self.daily_loss_breached = False
        
        # Alert management
        self.alerts: List[RiskAlert] = []
        self.alert_callbacks: List[Callable[[RiskAlert], None]] = []
        
        # Audit trail
        self.risk_decisions: deque = deque(maxlen=10000)
        
        # Risk monitoring
        self.last_risk_check = time.time()
        self.risk_check_interval = 60  # Check every minute
        
    async def assess_signal_risk(self, signal: ArbitrageSignal) -> RiskDecision:
        """Comprehensive risk assessment for trading signal"""
        decision = RiskDecision(
            signal_id=signal.signal_id,
            symbol=signal.symbol,
            exchange=f"{signal.buy_exchange}-{signal.sell_exchange}"
        )
        
        risk_factors = {}
        risk_level = RiskLevel.LOW
        action = RiskAction.ALLOW
        approved_size = signal.size
        
        try:
            # Check kill switch
            if self.kill_switch_active:
                decision.action = RiskAction.KILL_SWITCH
                decision.rejected = True
                decision.reasoning = "Kill switch is active"
                return decision
            
            # Check circuit breaker
            if self.circuit_breaker_active:
                decision.action = RiskAction.CIRCUIT_BREAKER
                decision.rejected = True
                decision.reasoning = "Circuit breaker is active"
                return decision
            
            # Position size checks
            position_risk = await self._assess_position_risk(signal)
            risk_factors.update(position_risk)
            
            # Portfolio exposure checks
            exposure_risk = await self._assess_exposure_risk(signal)
            risk_factors.update(exposure_risk)
            
            # Daily loss checks
            pnl_risk = await self._assess_pnl_risk(signal)
            risk_factors.update(pnl_risk)
            
            # Market condition checks
            market_risk = await self._assess_market_risk(signal)
            risk_factors.update(market_risk)
            
            # Determine overall risk level
            risk_scores = [rf.get('risk_score', 0) for rf in risk_factors.values()]
            max_risk_score = max(risk_scores) if risk_scores else 0
            
            if max_risk_score >= 0.8:
                risk_level = RiskLevel.CRITICAL
                action = RiskAction.BLOCK
                approved_size = Decimal('0')
            elif max_risk_score >= 0.6:
                risk_level = RiskLevel.HIGH
                action = RiskAction.REDUCE_SIZE
                approved_size = signal.size * Decimal('0.5')
            elif max_risk_score >= 0.4:
                risk_level = RiskLevel.MEDIUM
                action = RiskAction.REDUCE_SIZE
                approved_size = signal.size * Decimal('0.75')
            else:
                risk_level = RiskLevel.LOW
                action = RiskAction.ALLOW
            
            # Set decision details
            decision.risk_level = risk_level
            decision.action = action
            decision.risk_factors = risk_factors
            decision.original_size = signal.size
            decision.approved_size = approved_size
            decision.rejected = action in [RiskAction.BLOCK, RiskAction.CIRCUIT_BREAKER, RiskAction.KILL_SWITCH]
            
            # Generate reasoning
            decision.reasoning = self._generate_risk_reasoning(risk_factors, action)
            
            # Check for alerts
            await self._check_risk_alerts(decision, risk_factors)
            
        except Exception as e:
            logger.error(f"Error in risk assessment: {e}")
            decision.action = RiskAction.BLOCK
            decision.rejected = True
            decision.reasoning = f"Risk assessment error: {e}"
        
        # Record decision
        self.risk_decisions.append(decision)
        
        logger.info(f"Risk decision for {signal.signal_id}: {action.value} "
                   f"(size: {signal.size} -> {approved_size})")
        
        return decision
    
    async def _assess_position_risk(self, signal: ArbitrageSignal) -> Dict[str, Any]:
        """Assess position-specific risks"""
        risk_factors = {}
        
        # Check position size limits for each exchange
        for exchange in [signal.buy_exchange, signal.sell_exchange]:
            if not exchange:
                continue
                
            # Calculate position impact
            position_impact = self.position_manager.calculate_position_impact(
                exchange, signal.symbol, signal.size, 
                signal.buy_price if exchange == signal.buy_exchange else signal.sell_price
            )
            
            # Check against max position size
            max_position = self.settings.risk.max_position_size
            risk_score = float(position_impact / max_position) if max_position > 0 else 0
            
            risk_factors[f'position_size_{exchange}'] = {
                'current_impact': position_impact,
                'limit': max_position,
                'risk_score': min(risk_score, 1.0),
                'breach': position_impact > max_position
            }
        
        return risk_factors
    
    async def _assess_exposure_risk(self, signal: ArbitrageSignal) -> Dict[str, Any]:
        """Assess portfolio exposure risks"""
        risk_factors = {}
        
        metrics = self.position_manager.get_portfolio_metrics()
        
        # Total portfolio value check
        estimated_new_exposure = metrics.total_gross_exposure + (signal.size * signal.mid_price_ref)
        max_portfolio = self.settings.risk.max_portfolio_value
        
        exposure_risk_score = float(estimated_new_exposure / max_portfolio) if max_portfolio > 0 else 0
        
        risk_factors['portfolio_exposure'] = {
            'current_exposure': metrics.total_gross_exposure,
            'estimated_new_exposure': estimated_new_exposure,
            'limit': max_portfolio,
            'risk_score': min(exposure_risk_score, 1.0),
            'breach': estimated_new_exposure > max_portfolio
        }
        
        # Concentration risk
        max_single_after = max(
            metrics.max_single_position,
            signal.size * signal.mid_price_ref
        )
        concentration_ratio = float(max_single_after / estimated_new_exposure) if estimated_new_exposure > 0 else 0
        
        risk_factors['concentration'] = {
            'ratio': concentration_ratio,
            'risk_score': min(concentration_ratio * 2, 1.0),  # Risk if > 50% concentration
            'breach': concentration_ratio > 0.5
        }
        
        return risk_factors
    
    async def _assess_pnl_risk(self, signal: ArbitrageSignal) -> Dict[str, Any]:
        """Assess P&L-related risks"""
        risk_factors = {}
        
        metrics = self.position_manager.get_portfolio_metrics()
        
        # Daily loss check
        max_daily_loss = self.settings.risk.max_daily_loss
        loss_ratio = float(abs(metrics.daily_pnl) / max_daily_loss) if max_daily_loss > 0 else 0
        
        risk_factors['daily_pnl'] = {
            'current_pnl': metrics.daily_pnl,
            'limit': -max_daily_loss,
            'risk_score': loss_ratio if metrics.daily_pnl < 0 else 0,
            'breach': metrics.daily_pnl < -max_daily_loss
        }
        
        return risk_factors
    
    async def _assess_market_risk(self, signal: ArbitrageSignal) -> Dict[str, Any]:
        """Assess market condition risks"""
        risk_factors = {}
        
        # Slippage risk
        max_slippage = self.settings.risk.max_slippage_pct
        slippage_risk_score = float(signal.slippage_estimate / max_slippage) if max_slippage > 0 else 0
        
        risk_factors['slippage'] = {
            'estimated_slippage': signal.slippage_estimate,
            'limit': max_slippage,
            'risk_score': min(slippage_risk_score, 1.0),
            'breach': signal.slippage_estimate > max_slippage
        }
        
        # Spread width risk
        spread_risk_score = float(signal.spread_width_pct / 5.0)  # Risk if spread > 5%
        
        risk_factors['spread_width'] = {
            'spread_pct': signal.spread_width_pct,
            'risk_score': min(spread_risk_score, 1.0),
            'breach': signal.spread_width_pct > 5.0
        }
        
        return risk_factors
    
    def _generate_risk_reasoning(self, risk_factors: Dict[str, Any], action: RiskAction) -> str:
        """Generate human-readable risk reasoning"""
        reasons = []
        
        for factor_name, factor_data in risk_factors.items():
            if factor_data.get('breach', False):
                reasons.append(f"{factor_name} limit breached")
            elif factor_data.get('risk_score', 0) > 0.5:
                reasons.append(f"high {factor_name} risk")
        
        if not reasons:
            return f"Risk assessment passed, action: {action.value}"
        
        return f"Action {action.value} due to: {', '.join(reasons)}"
    
    async def _check_risk_alerts(self, decision: RiskDecision, risk_factors: Dict[str, Any]):
        """Check if any alerts should be generated"""
        alerts_to_send = []
        
        for factor_name, factor_data in risk_factors.items():
            if factor_data.get('breach', False):
                alert_type = self._map_factor_to_alert_type(factor_name)
                severity = RiskLevel.HIGH if factor_data.get('risk_score', 0) > 0.8 else RiskLevel.MEDIUM
                
                alert = RiskAlert(
                    alert_type=alert_type,
                    severity=severity,
                    message=f"{factor_name} limit breached",
                    symbol=decision.symbol,
                    current_value=factor_data.get('current_exposure') or factor_data.get('current_pnl'),
                    limit_value=factor_data.get('limit'),
                    signal_id=decision.signal_id,
                    action_taken=decision.action
                )
                
                alerts_to_send.append(alert)
        
        # Send alerts
        for alert in alerts_to_send:
            await self._send_alert(alert)
    
    def _map_factor_to_alert_type(self, factor_name: str) -> AlertType:
        """Map risk factor to alert type"""
        if 'position_size' in factor_name:
            return AlertType.POSITION_LIMIT
        elif 'daily_pnl' in factor_name:
            return AlertType.DAILY_LOSS
        elif 'slippage' in factor_name:
            return AlertType.SLIPPAGE_BREACH
        elif 'concentration' in factor_name:
            return AlertType.CONCENTRATION_RISK
        else:
            return AlertType.EXPOSURE_LIMIT
    
    async def _send_alert(self, alert: RiskAlert):
        """Send risk alert to all registered callbacks"""
        self.alerts.append(alert)
        
        logger.warning(f"Risk Alert: {alert.alert_type.value} - {alert.message}")
        
        for callback in self.alert_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(alert)
                else:
                    callback(alert)
            except Exception as e:
                logger.error(f"Error in alert callback: {e}")
    
    def add_alert_callback(self, callback: Callable[[RiskAlert], None]):
        """Add alert callback"""
        self.alert_callbacks.append(callback)
    
    async def activate_circuit_breaker(self, reason: str = "Manual activation"):
        """Activate circuit breaker"""
        self.circuit_breaker_active = True
        
        alert = RiskAlert(
            alert_type=AlertType.CIRCUIT_BREAKER_TRIGGERED,
            severity=RiskLevel.CRITICAL,
            message=f"Circuit breaker activated: {reason}",
            action_taken=RiskAction.CIRCUIT_BREAKER
        )
        
        await self._send_alert(alert)
        logger.critical(f"Circuit breaker activated: {reason}")
    
    async def activate_kill_switch(self, reason: str = "Manual activation"):
        """Activate kill switch"""
        self.kill_switch_active = True
        
        alert = RiskAlert(
            alert_type=AlertType.KILL_SWITCH_ACTIVATED,
            severity=RiskLevel.CRITICAL,
            message=f"Kill switch activated: {reason}",
            action_taken=RiskAction.KILL_SWITCH
        )
        
        await self._send_alert(alert)
        logger.critical(f"Kill switch activated: {reason}")
    
    def deactivate_circuit_breaker(self):
        """Deactivate circuit breaker"""
        self.circuit_breaker_active = False
        logger.info("Circuit breaker deactivated")
    
    def deactivate_kill_switch(self):
        """Deactivate kill switch"""
        self.kill_switch_active = False
        logger.info("Kill switch deactivated")
    
    def get_current_metrics(self) -> RiskMetrics:
        """Get current risk metrics"""
        metrics = self.position_manager.get_portfolio_metrics()
        metrics.circuit_breaker_active = self.circuit_breaker_active
        metrics.kill_switch_active = self.kill_switch_active
        return metrics
    
    def get_recent_alerts(self, hours: int = 24) -> List[RiskAlert]:
        """Get recent alerts"""
        cutoff_time = time.time() - (hours * 3600)
        return [alert for alert in self.alerts if alert.timestamp > cutoff_time]
    
    def get_risk_decisions(self, limit: int = 100) -> List[RiskDecision]:
        """Get recent risk decisions"""
        return list(self.risk_decisions)[-limit:]
    
    async def monitor_risk(self):
        """Continuous risk monitoring loop"""
        while True:
            try:
                current_time = time.time()
                
                if current_time - self.last_risk_check >= self.risk_check_interval:
                    await self._perform_risk_checks()
                    self.last_risk_check = current_time
                
                await asyncio.sleep(10)  # Check every 10 seconds
                
            except Exception as e:
                logger.error(f"Error in risk monitoring: {e}")
                await asyncio.sleep(60)
    
    async def _perform_risk_checks(self):
        """Perform periodic risk checks"""
        metrics = self.get_current_metrics()
        
        # Check for circuit breaker conditions
        if (not self.circuit_breaker_active and 
            self.settings.risk.enable_circuit_breaker and
            metrics.daily_pnl < -self.settings.risk.circuit_breaker_loss_threshold):
            
            await self.activate_circuit_breaker(
                f"Daily loss exceeded threshold: {metrics.daily_pnl}"
            )


# Example usage
async def main():
    """Example usage of risk management"""
    logging.basicConfig(level=logging.INFO)
    
    risk_manager = RiskManager()
    
    # Add alert callback
    async def alert_handler(alert: RiskAlert):
        print(f"ALERT: {alert.message}")
    
    risk_manager.add_alert_callback(alert_handler)
    
    # Create a test signal
    from .signal import ArbitrageSignal, SignalType
    
    signal = ArbitrageSignal(
        signal_id="test_signal",
        signal_type=SignalType.CROSS_EXCHANGE,
        timestamp=time.time(),
        symbol="BTC/USDT",
        expected_profit_pct=Decimal('0.5'),
        expected_profit_usd=Decimal('100'),
        confidence=0.8,
        strength="moderate",
        buy_exchange="binance",
        sell_exchange="kraken",
        buy_price=Decimal('50000'),
        sell_price=Decimal('50100'),
        size=Decimal('2.0'),  # Large size to trigger risk checks
        slippage_estimate=Decimal('0.2'),
        transfer_time_penalty=Decimal('0.05'),
        fees_total=Decimal('10'),
        mid_price_ref=Decimal('50050'),
        spread_width_pct=Decimal('0.2')
    )
    
    # Assess risk
    decision = await risk_manager.assess_signal_risk(signal)
    print(f"Risk decision: {decision.action.value}")
    print(f"Reasoning: {decision.reasoning}")
    print(f"Approved size: {decision.approved_size}")
    
    # Get current metrics
    metrics = risk_manager.get_current_metrics()
    print(f"Portfolio value: {metrics.total_portfolio_value}")
    print(f"Daily P&L: {metrics.daily_pnl}")


if __name__ == "__main__":
    asyncio.run(main())
