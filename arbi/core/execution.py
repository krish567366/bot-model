"""
Smart Order Router and Execution Engine

Handles order placement, lifecycle management, and atomic multi-leg execution
with support for IOC/FOK/PostOnly orders and idempotency for retries.
"""

import asyncio
import logging
import time
import uuid
from decimal import Decimal
from typing import Dict, List, Optional, Set, Tuple, Callable, Any
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
import json

from .marketdata import BookDelta, OrderBook, Side
from .signal import ArbitrageSignal, SignalType
from ..config.settings import get_settings

logger = logging.getLogger(__name__)


class OrderType(str, Enum):
    """Order type enumeration"""
    MARKET = "market"
    LIMIT = "limit"
    STOP_LIMIT = "stop_limit"
    IOC = "immediate_or_cancel"  # Immediate or Cancel
    FOK = "fill_or_kill"         # Fill or Kill
    POST_ONLY = "post_only"      # Maker only


class OrderStatus(str, Enum):
    """Order status enumeration"""
    PENDING = "pending"
    SUBMITTED = "submitted"
    ACKNOWLEDGED = "acknowledged"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"
    FAILED = "failed"


class OrderSide(str, Enum):
    """Order side enumeration"""
    BUY = "buy"
    SELL = "sell"


@dataclass
class Order:
    """Order representation with lifecycle tracking"""
    order_id: str
    client_order_id: str
    exchange: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: Decimal
    price: Optional[Decimal] = None
    
    # Status tracking
    status: OrderStatus = OrderStatus.PENDING
    created_time: float = field(default_factory=time.time)
    submitted_time: Optional[float] = None
    acknowledged_time: Optional[float] = None
    filled_time: Optional[float] = None
    
    # Execution tracking
    filled_quantity: Decimal = Decimal('0')
    remaining_quantity: Optional[Decimal] = None
    average_fill_price: Optional[Decimal] = None
    
    # Metadata
    signal_id: Optional[str] = None
    execution_group_id: Optional[str] = None
    retry_count: int = 0
    error_message: Optional[str] = None
    
    # Exchange-specific data
    exchange_order_id: Optional[str] = None
    exchange_metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Initialize derived fields"""
        if self.remaining_quantity is None:
            self.remaining_quantity = self.quantity
    
    @property
    def is_complete(self) -> bool:
        """Check if order is in a final state"""
        return self.status in [
            OrderStatus.FILLED,
            OrderStatus.CANCELLED,
            OrderStatus.REJECTED,
            OrderStatus.EXPIRED,
            OrderStatus.FAILED
        ]
    
    @property
    def fill_percentage(self) -> float:
        """Calculate fill percentage"""
        if self.quantity == 0:
            return 0.0
        return float(self.filled_quantity / self.quantity * 100)
    
    def update_fill(self, fill_quantity: Decimal, fill_price: Decimal):
        """Update order with fill information"""
        self.filled_quantity += fill_quantity
        self.remaining_quantity = self.quantity - self.filled_quantity
        
        # Update average fill price
        if self.average_fill_price is None:
            self.average_fill_price = fill_price
        else:
            total_value = (self.filled_quantity - fill_quantity) * self.average_fill_price + fill_quantity * fill_price
            self.average_fill_price = total_value / self.filled_quantity
        
        # Update status
        if self.remaining_quantity <= Decimal('0'):
            self.status = OrderStatus.FILLED
            self.filled_time = time.time()
        elif self.filled_quantity > Decimal('0'):
            self.status = OrderStatus.PARTIALLY_FILLED


@dataclass
class ExecutionGroup:
    """Group of orders for atomic execution"""
    group_id: str
    orders: List[Order]
    signal: ArbitrageSignal
    created_time: float = field(default_factory=time.time)
    
    # Execution tracking
    submitted_orders: Set[str] = field(default_factory=set)
    completed_orders: Set[str] = field(default_factory=set)
    failed_orders: Set[str] = field(default_factory=set)
    
    @property
    def all_submitted(self) -> bool:
        """Check if all orders are submitted"""
        return len(self.submitted_orders) == len(self.orders)
    
    @property
    def all_completed(self) -> bool:
        """Check if all orders are completed (successfully or not)"""
        return len(self.completed_orders) + len(self.failed_orders) == len(self.orders)
    
    @property
    def has_failures(self) -> bool:
        """Check if any orders failed"""
        return len(self.failed_orders) > 0
    
    @property
    def success_rate(self) -> float:
        """Calculate execution success rate"""
        total = len(self.completed_orders) + len(self.failed_orders)
        if total == 0:
            return 0.0
        return len(self.completed_orders) / total


class OrderManager:
    """Manages order lifecycle and state"""
    
    def __init__(self):
        self.orders: Dict[str, Order] = {}
        self.execution_groups: Dict[str, ExecutionGroup] = {}
        self.order_callbacks: Dict[str, List[Callable[[Order], None]]] = defaultdict(list)
        
    def create_order(
        self,
        exchange: str,
        symbol: str,
        side: OrderSide,
        order_type: OrderType,
        quantity: Decimal,
        price: Optional[Decimal] = None,
        signal_id: Optional[str] = None,
        execution_group_id: Optional[str] = None
    ) -> Order:
        """Create a new order"""
        order = Order(
            order_id=str(uuid.uuid4()),
            client_order_id=f"arbi_{int(time.time() * 1000)}_{len(self.orders)}",
            exchange=exchange,
            symbol=symbol,
            side=side,
            order_type=order_type,
            quantity=quantity,
            price=price,
            signal_id=signal_id,
            execution_group_id=execution_group_id
        )
        
        self.orders[order.order_id] = order
        logger.info(f"Created order {order.client_order_id}: {side} {quantity} {symbol} @ {price} on {exchange}")
        
        return order
    
    def update_order_status(self, order_id: str, status: OrderStatus, **kwargs):
        """Update order status and trigger callbacks"""
        if order_id not in self.orders:
            logger.warning(f"Order {order_id} not found for status update")
            return
        
        order = self.orders[order_id]
        old_status = order.status
        order.status = status
        
        # Update timestamps
        if status == OrderStatus.SUBMITTED and order.submitted_time is None:
            order.submitted_time = time.time()
        elif status == OrderStatus.ACKNOWLEDGED and order.acknowledged_time is None:
            order.acknowledged_time = time.time()
        elif status == OrderStatus.FILLED and order.filled_time is None:
            order.filled_time = time.time()
        
        # Update additional fields
        for key, value in kwargs.items():
            if hasattr(order, key):
                setattr(order, key, value)
        
        logger.info(f"Order {order.client_order_id} status: {old_status} -> {status}")
        
        # Trigger callbacks
        for callback in self.order_callbacks[order_id]:
            try:
                callback(order)
            except Exception as e:
                logger.error(f"Error in order callback: {e}")
        
        # Update execution group
        if order.execution_group_id:
            self._update_execution_group(order)
    
    def update_order_fill(self, order_id: str, fill_quantity: Decimal, fill_price: Decimal):
        """Update order with fill information"""
        if order_id not in self.orders:
            logger.warning(f"Order {order_id} not found for fill update")
            return
        
        order = self.orders[order_id]
        order.update_fill(fill_quantity, fill_price)
        
        logger.info(f"Order {order.client_order_id} fill: {fill_quantity} @ {fill_price} "
                   f"(total: {order.filled_quantity}/{order.quantity})")
        
        # Trigger callbacks
        for callback in self.order_callbacks[order_id]:
            try:
                callback(order)
            except Exception as e:
                logger.error(f"Error in order callback: {e}")
    
    def add_order_callback(self, order_id: str, callback: Callable[[Order], None]):
        """Add callback for order updates"""
        self.order_callbacks[order_id].append(callback)
    
    def get_order(self, order_id: str) -> Optional[Order]:
        """Get order by ID"""
        return self.orders.get(order_id)
    
    def get_orders_by_signal(self, signal_id: str) -> List[Order]:
        """Get all orders for a signal"""
        return [order for order in self.orders.values() if order.signal_id == signal_id]
    
    def get_active_orders(self) -> List[Order]:
        """Get all active (non-complete) orders"""
        return [order for order in self.orders.values() if not order.is_complete]
    
    def create_execution_group(self, orders: List[Order], signal: ArbitrageSignal) -> ExecutionGroup:
        """Create execution group for atomic execution"""
        group_id = str(uuid.uuid4())
        
        # Update orders with group ID
        for order in orders:
            order.execution_group_id = group_id
        
        group = ExecutionGroup(
            group_id=group_id,
            orders=orders,
            signal=signal
        )
        
        self.execution_groups[group_id] = group
        return group
    
    def _update_execution_group(self, order: Order):
        """Update execution group status based on order updates"""
        if not order.execution_group_id or order.execution_group_id not in self.execution_groups:
            return
        
        group = self.execution_groups[order.execution_group_id]
        
        # Update group tracking sets
        if order.status == OrderStatus.SUBMITTED:
            group.submitted_orders.add(order.order_id)
        elif order.is_complete:
            if order.status == OrderStatus.FILLED:
                group.completed_orders.add(order.order_id)
            else:
                group.failed_orders.add(order.order_id)
        
        logger.debug(f"Execution group {group.group_id}: {len(group.completed_orders)} completed, "
                    f"{len(group.failed_orders)} failed out of {len(group.orders)} total")


class ExchangeConnector:
    """Base class for exchange connectivity"""
    
    def __init__(self, exchange_name: str):
        self.exchange_name = exchange_name
        self.settings = get_settings()
        
    async def submit_order(self, order: Order) -> bool:
        """Submit order to exchange - to be implemented by subclasses"""
        raise NotImplementedError
    
    async def cancel_order(self, order: Order) -> bool:
        """Cancel order on exchange - to be implemented by subclasses"""
        raise NotImplementedError
    
    async def get_order_status(self, order: Order) -> Optional[OrderStatus]:
        """Get order status from exchange - to be implemented by subclasses"""
        raise NotImplementedError


class MockExchangeConnector(ExchangeConnector):
    """Mock exchange connector for testing"""
    
    def __init__(self, exchange_name: str):
        super().__init__(exchange_name)
        self.latency_ms = 100  # Simulated latency
        self.success_rate = 0.95  # 95% success rate
        
    async def submit_order(self, order: Order) -> bool:
        """Mock order submission"""
        await asyncio.sleep(self.latency_ms / 1000)  # Simulate latency
        
        # Simulate occasional failures
        import random
        if random.random() > self.success_rate:
            logger.warning(f"Mock order submission failed for {order.client_order_id}")
            return False
        
        # Simulate successful submission
        order.exchange_order_id = f"mock_{random.randint(1000, 9999)}"
        logger.info(f"Mock submitted order {order.client_order_id} -> {order.exchange_order_id}")
        return True
    
    async def cancel_order(self, order: Order) -> bool:
        """Mock order cancellation"""
        await asyncio.sleep(self.latency_ms / 1000)
        logger.info(f"Mock cancelled order {order.client_order_id}")
        return True
    
    async def get_order_status(self, order: Order) -> Optional[OrderStatus]:
        """Mock order status check"""
        await asyncio.sleep(self.latency_ms / 1000)
        
        # Simulate random status progression
        import random
        if random.random() < 0.8:
            return OrderStatus.FILLED
        elif random.random() < 0.9:
            return OrderStatus.PARTIALLY_FILLED
        else:
            return OrderStatus.ACKNOWLEDGED


class SmartOrderRouter:
    """Smart Order Router with atomic multi-leg execution"""
    
    def __init__(self):
        self.settings = get_settings()
        self.order_manager = OrderManager()
        self.connectors: Dict[str, ExchangeConnector] = {}
        self.active_executions: Dict[str, ExecutionGroup] = {}
        
        # Initialize mock connectors for testing
        self._init_connectors()
        
    def _init_connectors(self):
        """Initialize exchange connectors"""
        for exchange in self.settings.enabled_exchanges:
            # For now, use mock connectors
            # In production, would initialize real exchange connectors
            self.connectors[exchange] = MockExchangeConnector(exchange)
    
    async def execute_signal(self, signal: ArbitrageSignal) -> Optional[ExecutionGroup]:
        """Execute arbitrage signal with appropriate strategy"""
        try:
            if signal.signal_type == SignalType.CROSS_EXCHANGE:
                return await self._execute_cross_exchange(signal)
            elif signal.signal_type == SignalType.TRIANGULAR:
                return await self._execute_triangular(signal)
            else:
                logger.warning(f"Unsupported signal type: {signal.signal_type}")
                return None
        except Exception as e:
            logger.error(f"Error executing signal {signal.signal_id}: {e}")
            return None
    
    async def _execute_cross_exchange(self, signal: ArbitrageSignal) -> Optional[ExecutionGroup]:
        """Execute cross-exchange arbitrage"""
        logger.info(f"Executing cross-exchange arbitrage: {signal.symbol} "
                   f"{signal.buy_exchange} -> {signal.sell_exchange}")
        
        # Create buy and sell orders
        orders = []
        
        # Buy order
        buy_order = self.order_manager.create_order(
            exchange=signal.buy_exchange,
            symbol=signal.symbol,
            side=OrderSide.BUY,
            order_type=self._get_optimal_order_type(signal.buy_exchange, signal.symbol),
            quantity=signal.size,
            price=signal.buy_price,
            signal_id=signal.signal_id
        )
        orders.append(buy_order)
        
        # Sell order
        sell_order = self.order_manager.create_order(
            exchange=signal.sell_exchange,
            symbol=signal.symbol,
            side=OrderSide.SELL,
            order_type=self._get_optimal_order_type(signal.sell_exchange, signal.symbol),
            quantity=signal.size,
            price=signal.sell_price,
            signal_id=signal.signal_id
        )
        orders.append(sell_order)
        
        # Create execution group for atomic execution
        execution_group = self.order_manager.create_execution_group(orders, signal)
        
        # Execute atomically
        success = await self._execute_atomic(execution_group)
        
        if success:
            self.active_executions[execution_group.group_id] = execution_group
            logger.info(f"Cross-exchange execution initiated: {execution_group.group_id}")
            return execution_group
        else:
            logger.error(f"Failed to execute cross-exchange arbitrage for signal {signal.signal_id}")
            return None
    
    async def _execute_triangular(self, signal: ArbitrageSignal) -> Optional[ExecutionGroup]:
        """Execute triangular arbitrage"""
        if not signal.path or not signal.intermediate_prices:
            logger.error("Triangular signal missing path or prices")
            return None
        
        logger.info(f"Executing triangular arbitrage: {' -> '.join(signal.path)}")
        
        orders = []
        
        # Create orders for each leg of the triangle
        for i, (symbol, price) in enumerate(zip(signal.path, signal.intermediate_prices)):
            # Determine side based on position in the path
            side = OrderSide.BUY if i % 2 == 0 else OrderSide.SELL
            
            order = self.order_manager.create_order(
                exchange=signal.buy_exchange,  # Same exchange for triangular
                symbol=symbol,
                side=side,
                order_type=OrderType.IOC,  # Use IOC for triangular to minimize timing risk
                quantity=signal.size,
                price=price,
                signal_id=signal.signal_id
            )
            orders.append(order)
        
        # Create execution group
        execution_group = self.order_manager.create_execution_group(orders, signal)
        
        # Execute sequentially for triangular arbitrage
        success = await self._execute_sequential(execution_group)
        
        if success:
            self.active_executions[execution_group.group_id] = execution_group
            logger.info(f"Triangular execution initiated: {execution_group.group_id}")
            return execution_group
        else:
            logger.error(f"Failed to execute triangular arbitrage for signal {signal.signal_id}")
            return None
    
    async def _execute_atomic(self, execution_group: ExecutionGroup) -> bool:
        """Execute orders atomically (submit all, cancel all on any failure)"""
        orders = execution_group.orders
        submitted_orders = []
        
        try:
            # Submit all orders concurrently
            submit_tasks = []
            for order in orders:
                if order.exchange in self.connectors:
                    connector = self.connectors[order.exchange]
                    task = asyncio.create_task(self._submit_order_with_retry(connector, order))
                    submit_tasks.append((order, task))
            
            # Wait for all submissions
            for order, task in submit_tasks:
                try:
                    success = await asyncio.wait_for(task, timeout=self.settings.execution.order_timeout_seconds)
                    if success:
                        submitted_orders.append(order)
                        self.order_manager.update_order_status(order.order_id, OrderStatus.SUBMITTED)
                    else:
                        self.order_manager.update_order_status(order.order_id, OrderStatus.FAILED)
                except asyncio.TimeoutError:
                    logger.error(f"Order submission timeout: {order.client_order_id}")
                    self.order_manager.update_order_status(order.order_id, OrderStatus.FAILED)
                except Exception as e:
                    logger.error(f"Order submission error: {order.client_order_id} - {e}")
                    self.order_manager.update_order_status(order.order_id, OrderStatus.FAILED)
            
            # Check if all orders were submitted successfully
            if len(submitted_orders) == len(orders):
                logger.info(f"All orders submitted successfully for group {execution_group.group_id}")
                return True
            else:
                # Cancel successfully submitted orders
                logger.warning(f"Partial submission failure, cancelling {len(submitted_orders)} orders")
                await self._cancel_orders(submitted_orders)
                return False
        
        except Exception as e:
            logger.error(f"Error in atomic execution: {e}")
            # Cancel any submitted orders
            if submitted_orders:
                await self._cancel_orders(submitted_orders)
            return False
    
    async def _execute_sequential(self, execution_group: ExecutionGroup) -> bool:
        """Execute orders sequentially (for triangular arbitrage)"""
        orders = execution_group.orders
        
        for i, order in enumerate(orders):
            if order.exchange not in self.connectors:
                logger.error(f"No connector for exchange: {order.exchange}")
                return False
            
            connector = self.connectors[order.exchange]
            
            try:
                # Submit order
                success = await self._submit_order_with_retry(connector, order)
                if not success:
                    logger.error(f"Failed to submit order {i+1}/{len(orders)}: {order.client_order_id}")
                    return False
                
                self.order_manager.update_order_status(order.order_id, OrderStatus.SUBMITTED)
                
                # Wait for fill (simplified - in production would have more sophisticated monitoring)
                await self._wait_for_fill(order, timeout=30)
                
                if order.status != OrderStatus.FILLED:
                    logger.error(f"Order not filled: {order.client_order_id}")
                    return False
                
            except Exception as e:
                logger.error(f"Error executing sequential order {order.client_order_id}: {e}")
                return False
        
        logger.info(f"Sequential execution completed for group {execution_group.group_id}")
        return True
    
    async def _submit_order_with_retry(self, connector: ExchangeConnector, order: Order) -> bool:
        """Submit order with retry logic"""
        max_retries = 3
        
        for attempt in range(max_retries):
            try:
                success = await connector.submit_order(order)
                if success:
                    return True
                
                order.retry_count += 1
                if attempt < max_retries - 1:
                    delay = min(2 ** attempt, 5)  # Exponential backoff, max 5 seconds
                    logger.info(f"Retrying order submission in {delay}s: {order.client_order_id}")
                    await asyncio.sleep(delay)
            
            except Exception as e:
                logger.error(f"Order submission attempt {attempt + 1} failed: {order.client_order_id} - {e}")
                order.error_message = str(e)
                
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 ** attempt)
        
        return False
    
    async def _cancel_orders(self, orders: List[Order]):
        """Cancel multiple orders"""
        cancel_tasks = []
        
        for order in orders:
            if order.exchange in self.connectors:
                connector = self.connectors[order.exchange]
                task = asyncio.create_task(connector.cancel_order(order))
                cancel_tasks.append((order, task))
        
        # Wait for all cancellations
        for order, task in cancel_tasks:
            try:
                await asyncio.wait_for(task, timeout=10)
                self.order_manager.update_order_status(order.order_id, OrderStatus.CANCELLED)
            except Exception as e:
                logger.error(f"Failed to cancel order {order.client_order_id}: {e}")
    
    async def _wait_for_fill(self, order: Order, timeout: int = 30):
        """Wait for order to be filled"""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            if order.is_complete:
                break
            
            # Check order status
            if order.exchange in self.connectors:
                connector = self.connectors[order.exchange]
                try:
                    status = await connector.get_order_status(order)
                    if status and status != order.status:
                        self.order_manager.update_order_status(order.order_id, status)
                        
                        # Simulate fill for testing
                        if status == OrderStatus.FILLED:
                            self.order_manager.update_order_fill(
                                order.order_id,
                                order.quantity,
                                order.price or Decimal('0')
                            )
                except Exception as e:
                    logger.error(f"Error checking order status: {e}")
            
            await asyncio.sleep(1)
    
    def _get_optimal_order_type(self, exchange: str, symbol: str) -> OrderType:
        """Determine optimal order type based on market conditions"""
        # Simplified logic - in production would consider market conditions,
        # liquidity, volatility, etc.
        
        if self.settings.execution.enable_ioc_orders:
            return OrderType.IOC
        elif self.settings.execution.enable_fok_orders:
            return OrderType.FOK
        else:
            return OrderType.LIMIT
    
    def get_active_executions(self) -> List[ExecutionGroup]:
        """Get all active execution groups"""
        return list(self.active_executions.values())
    
    def get_execution_status(self, group_id: str) -> Optional[ExecutionGroup]:
        """Get execution group status"""
        return self.active_executions.get(group_id)
    
    async def cancel_execution(self, group_id: str):
        """Cancel an execution group"""
        if group_id in self.active_executions:
            group = self.active_executions[group_id]
            active_orders = [order for order in group.orders if not order.is_complete]
            await self._cancel_orders(active_orders)
            logger.info(f"Cancelled execution group {group_id}")


# Example usage
async def main():
    """Example usage of the execution engine"""
    logging.basicConfig(level=logging.INFO)
    
    # Create router
    router = SmartOrderRouter()
    
    # Create a sample signal
    from .signal import ArbitrageSignal, SignalType
    
    signal = ArbitrageSignal(
        signal_id="test_signal_1",
        signal_type=SignalType.CROSS_EXCHANGE,
        timestamp=time.time(),
        symbol="BTC/USDT",
        expected_profit_pct=Decimal('0.5'),
        expected_profit_usd=Decimal('100'),
        confidence=0.8,
        strength="strong",
        buy_exchange="binance",
        sell_exchange="kraken",
        buy_price=Decimal('50000'),
        sell_price=Decimal('50100'),
        size=Decimal('0.1'),
        slippage_estimate=Decimal('0.1'),
        transfer_time_penalty=Decimal('0.05'),
        fees_total=Decimal('10'),
        mid_price_ref=Decimal('50050'),
        spread_width_pct=Decimal('0.2')
    )
    
    # Execute signal
    execution_group = await router.execute_signal(signal)
    
    if execution_group:
        logger.info(f"Execution started: {execution_group.group_id}")
        
        # Monitor execution
        await asyncio.sleep(5)
        
        # Check status
        active_executions = router.get_active_executions()
        logger.info(f"Active executions: {len(active_executions)}")
        
        for group in active_executions:
            logger.info(f"Group {group.group_id}: Success rate {group.success_rate:.2%}")


if __name__ == "__main__":
    asyncio.run(main())
