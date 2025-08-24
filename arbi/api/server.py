"""
FastAPI Server

REST API and WebSocket server for the arbitrage trading system.
Provides endpoints for monitoring, control, and real-time data streaming.
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, List, Optional, Any
import uvicorn
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, Depends, status, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
import jwt
from contextlib import asynccontextmanager

# Import our trading system components
from ..core.data_feed import DataFeed
from ..core.signal import SignalAggregator
from ..core.execution import SmartOrderRouter
from ..core.risk import RiskManager
from ..core.portfolio import PortfolioManager
from ..core.storage import StorageManager
from ..core.backtest import BacktestEngine, BacktestConfig
from ..config.settings import get_settings

logger = logging.getLogger(__name__)


# Pydantic models for API requests/responses
class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    timestamp: float
    version: str = "1.0.0"
    uptime_seconds: float


class SystemStatus(BaseModel):
    """System status response"""
    status: str
    data_feed_connected: bool
    signal_generator_active: bool
    execution_engine_active: bool
    risk_manager_active: bool
    portfolio_manager_active: bool
    total_orders_today: int
    total_signals_today: int
    last_signal_time: Optional[float]
    current_positions: int
    total_portfolio_value_usd: Decimal


class BalanceResponse(BaseModel):
    """Balance information response"""
    exchange: str
    balances: Dict[str, Decimal]
    total_value_usd: Decimal
    last_updated: float


class PositionResponse(BaseModel):
    """Position information response"""
    exchange: str
    symbol: str
    side: str
    size: Decimal
    entry_price: Decimal
    current_price: Decimal
    unrealized_pnl: Decimal
    realized_pnl: Decimal


class OrderResponse(BaseModel):
    """Order information response"""
    order_id: str
    exchange: str
    symbol: str
    side: str
    order_type: str
    quantity: Decimal
    price: Optional[Decimal]
    status: str
    created_time: float
    filled_quantity: Decimal = Decimal("0")
    average_fill_price: Optional[Decimal]


class SignalResponse(BaseModel):
    """Signal information response"""
    signal_id: str
    signal_type: str
    symbol: str
    buy_exchange: str
    sell_exchange: str
    buy_price: Decimal
    sell_price: Decimal
    expected_profit_pct: Decimal
    confidence: float
    timestamp: float


class RiskStatusResponse(BaseModel):
    """Risk management status"""
    status: str
    circuit_breaker_active: bool
    position_limits_breached: bool
    daily_loss_limit_breached: bool
    current_daily_pnl: Decimal
    max_daily_loss: Decimal
    active_alerts: int


class KillSwitchRequest(BaseModel):
    """Kill switch activation request"""
    reason: str
    emergency: bool = False


class BacktestRequest(BaseModel):
    """Backtest execution request"""
    start_date: str = Field(..., description="Start date (YYYY-MM-DD)")
    end_date: str = Field(..., description="End date (YYYY-MM-DD)")
    initial_balance: Dict[str, float] = Field({"USDT": 10000}, description="Initial balance")
    exchanges: List[str] = Field(["binance", "kraken"], description="Exchanges to include")
    symbols: List[str] = Field(["BTC/USDT"], description="Symbols to trade")
    enable_slippage: bool = Field(True, description="Enable slippage simulation")
    enable_latency: bool = Field(True, description="Enable latency simulation")


class BacktestResponse(BaseModel):
    """Backtest result response"""
    backtest_id: str
    status: str
    total_return_pct: Optional[float]
    sharpe_ratio: Optional[float]
    max_drawdown_pct: Optional[float]
    total_trades: Optional[int]
    win_rate_pct: Optional[float]
    start_date: Optional[str]
    end_date: Optional[str]


# Global state
class AppState:
    """Application state management"""
    def __init__(self):
        self.settings = get_settings()
        self.start_time = time.time()
        
        # Trading system components
        self.data_feed: Optional[DataFeed] = None
        self.signal_aggregator: Optional[SignalAggregator] = None
        self.order_router: Optional[SmartOrderRouter] = None
        self.risk_manager: Optional[RiskManager] = None
        self.portfolio_manager: Optional[PortfolioManager] = None
        self.storage_manager: Optional[StorageManager] = None
        
        # WebSocket connections
        self.websocket_connections: List[WebSocket] = []
        
        # Background tasks
        self.background_tasks: List[asyncio.Task] = []
        
        # Metrics
        self.total_orders_today = 0
        self.total_signals_today = 0
        self.last_signal_time: Optional[float] = None


app_state = AppState()


# Security
security = HTTPBearer()

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify JWT token"""
    try:
        payload = jwt.decode(
            credentials.credentials, 
            app_state.settings.api.secret_key, 
            algorithms=["HS256"]
        )
        return payload
    except jwt.PyJWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication token"
        )


# Application lifespan management
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle"""
    # Startup
    logger.info("Starting trading system...")
    await initialize_trading_system()
    yield
    # Shutdown
    logger.info("Shutting down trading system...")
    await shutdown_trading_system()


# Create FastAPI app
app = FastAPI(
    title="Crypto Arbitrage Trading System",
    description="Production-grade arbitrage trading platform with real-time data feeds, signal generation, and execution",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if app_state.settings.api.enable_cors else [],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Initialize trading system
async def initialize_trading_system():
    """Initialize all trading system components"""
    try:
        # Initialize storage
        app_state.storage_manager = StorageManager()
        await app_state.storage_manager.initialize()
        
        # Initialize data feed
        app_state.data_feed = DataFeed()
        
        # Initialize signal aggregator
        app_state.signal_aggregator = SignalAggregator()
        
        # Initialize order router
        app_state.order_router = SmartOrderRouter()
        
        # Initialize risk manager
        app_state.risk_manager = RiskManager()
        
        # Initialize portfolio manager
        app_state.portfolio_manager = PortfolioManager()
        
        # Connect data flow
        await connect_data_flow()
        
        # Start background tasks
        await start_background_tasks()
        
        logger.info("Trading system initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize trading system: {e}")
        raise


async def connect_data_flow():
    """Connect data flow between components"""
    if not all([app_state.data_feed, app_state.signal_aggregator, 
                app_state.order_router, app_state.risk_manager, 
                app_state.portfolio_manager]):
        return
    
    # Set up data flow connections
    # data_feed -> signal_aggregator -> order_router -> risk_manager -> portfolio_manager
    
    # Market data to signal generation
    async def on_market_data(book_delta):
        if app_state.signal_aggregator:
            await app_state.signal_aggregator.process_book_delta(book_delta)
        if app_state.storage_manager:
            await app_state.storage_manager.store_book_delta(book_delta)
    
    # Signals to execution
    async def on_signal(signal):
        app_state.total_signals_today += 1
        app_state.last_signal_time = time.time()
        
        if app_state.order_router and app_state.risk_manager:
            # Risk check
            decision = await app_state.risk_manager.assess_signal(signal)
            if not decision.rejected:
                await app_state.order_router.execute_signal(signal)
        
        if app_state.storage_manager:
            await app_state.storage_manager.store_signal(signal)
        
        # Broadcast to WebSocket clients
        await broadcast_signal(signal)
    
    # Orders to portfolio
    async def on_order_update(order):
        app_state.total_orders_today += 1
        
        if app_state.portfolio_manager:
            await app_state.portfolio_manager.process_order(order)
        
        if app_state.storage_manager:
            await app_state.storage_manager.store_order(order)
        
        # Broadcast to WebSocket clients
        await broadcast_order_update(order)
    
    # Connect callbacks (simplified - in real implementation these would be proper event handlers)
    logger.info("Data flow connections established")


async def start_background_tasks():
    """Start background monitoring and maintenance tasks"""
    # Portfolio monitoring
    async def portfolio_monitor():
        while True:
            try:
                if app_state.portfolio_manager:
                    snapshot = await app_state.portfolio_manager.get_portfolio_snapshot()
                    if app_state.storage_manager:
                        await app_state.storage_manager.store_portfolio_snapshot(snapshot)
                await asyncio.sleep(60)  # Every minute
            except Exception as e:
                logger.error(f"Portfolio monitoring error: {e}")
                await asyncio.sleep(60)
    
    # Storage maintenance
    async def storage_monitor():
        while True:
            try:
                if app_state.storage_manager:
                    await app_state.storage_manager.periodic_maintenance()
                await asyncio.sleep(3600)  # Every hour
            except Exception as e:
                logger.error(f"Storage monitoring error: {e}")
                await asyncio.sleep(3600)
    
    # Start tasks
    app_state.background_tasks = [
        asyncio.create_task(portfolio_monitor()),
        asyncio.create_task(storage_monitor())
    ]
    
    logger.info("Background tasks started")


async def shutdown_trading_system():
    """Shutdown trading system gracefully"""
    logger.info("Shutting down trading system...")
    
    # Cancel background tasks
    for task in app_state.background_tasks:
        task.cancel()
    
    # Flush storage buffers
    if app_state.storage_manager:
        await app_state.storage_manager.flush_all_buffers()
    
    # Close WebSocket connections
    for websocket in app_state.websocket_connections:
        try:
            await websocket.close()
        except:
            pass
    
    logger.info("Trading system shutdown complete")


# WebSocket broadcast functions
async def broadcast_signal(signal):
    """Broadcast signal to all WebSocket clients"""
    if not app_state.websocket_connections:
        return
    
    message = {
        "type": "signal",
        "data": {
            "signal_id": signal.signal_id,
            "signal_type": signal.signal_type.value,
            "symbol": signal.symbol,
            "buy_exchange": signal.buy_exchange,
            "sell_exchange": signal.sell_exchange,
            "expected_profit_pct": float(signal.expected_profit_pct),
            "confidence": signal.confidence,
            "timestamp": signal.timestamp
        }
    }
    
    # Send to all connected clients
    disconnected = []
    for websocket in app_state.websocket_connections:
        try:
            await websocket.send_text(json.dumps(message))
        except:
            disconnected.append(websocket)
    
    # Remove disconnected clients
    for ws in disconnected:
        app_state.websocket_connections.remove(ws)


async def broadcast_order_update(order):
    """Broadcast order update to all WebSocket clients"""
    if not app_state.websocket_connections:
        return
    
    message = {
        "type": "order_update",
        "data": {
            "order_id": order.order_id,
            "exchange": order.exchange,
            "symbol": order.symbol,
            "status": order.status.value,
            "filled_quantity": float(order.filled_quantity),
            "timestamp": time.time()
        }
    }
    
    # Send to all connected clients
    disconnected = []
    for websocket in app_state.websocket_connections:
        try:
            await websocket.send_text(json.dumps(message))
        except:
            disconnected.append(websocket)
    
    # Remove disconnected clients
    for ws in disconnected:
        app_state.websocket_connections.remove(ws)


# REST API Endpoints

@app.get("/", response_model=HealthResponse)
async def root():
    """Root endpoint with basic health check"""
    return HealthResponse(
        status="healthy",
        timestamp=time.time(),
        uptime_seconds=time.time() - app_state.start_time
    )


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Detailed health check"""
    return HealthResponse(
        status="healthy",
        timestamp=time.time(),
        uptime_seconds=time.time() - app_state.start_time
    )


@app.get("/status", response_model=SystemStatus)
async def get_system_status(user = Depends(verify_token)):
    """Get comprehensive system status"""
    portfolio_value = Decimal("0")
    current_positions = 0
    
    if app_state.portfolio_manager:
        try:
            snapshot = await app_state.portfolio_manager.get_portfolio_snapshot()
            portfolio_value = snapshot.total_value_usd
            current_positions = snapshot.num_positions
        except:
            pass
    
    return SystemStatus(
        status="active",
        data_feed_connected=app_state.data_feed is not None,
        signal_generator_active=app_state.signal_aggregator is not None,
        execution_engine_active=app_state.order_router is not None,
        risk_manager_active=app_state.risk_manager is not None,
        portfolio_manager_active=app_state.portfolio_manager is not None,
        total_orders_today=app_state.total_orders_today,
        total_signals_today=app_state.total_signals_today,
        last_signal_time=app_state.last_signal_time,
        current_positions=current_positions,
        total_portfolio_value_usd=portfolio_value
    )


@app.get("/balances", response_model=List[BalanceResponse])
async def get_balances(user = Depends(verify_token)):
    """Get current balances across all exchanges"""
    if not app_state.portfolio_manager:
        raise HTTPException(status_code=503, detail="Portfolio manager not available")
    
    balances = []
    try:
        portfolio_data = await app_state.portfolio_manager.get_all_balances()
        
        for exchange, exchange_balances in portfolio_data.items():
            total_value = sum(balance.value_usd for balance in exchange_balances.values())
            balance_dict = {currency: balance.available for currency, balance in exchange_balances.items()}
            
            balances.append(BalanceResponse(
                exchange=exchange,
                balances=balance_dict,
                total_value_usd=total_value,
                last_updated=time.time()
            ))
    
    except Exception as e:
        logger.error(f"Error getting balances: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve balances")
    
    return balances


@app.get("/positions", response_model=List[PositionResponse])
async def get_positions(user = Depends(verify_token)):
    """Get current positions across all exchanges"""
    if not app_state.portfolio_manager:
        raise HTTPException(status_code=503, detail="Portfolio manager not available")
    
    positions = []
    try:
        portfolio_data = await app_state.portfolio_manager.get_all_positions()
        
        for exchange, exchange_positions in portfolio_data.items():
            for symbol, position in exchange_positions.items():
                positions.append(PositionResponse(
                    exchange=exchange,
                    symbol=symbol,
                    side="LONG" if position.quantity > 0 else "SHORT",
                    size=abs(position.quantity),
                    entry_price=position.average_price,
                    current_price=position.mark_price or position.average_price,
                    unrealized_pnl=position.unrealized_pnl,
                    realized_pnl=position.realized_pnl
                ))
    
    except Exception as e:
        logger.error(f"Error getting positions: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve positions")
    
    return positions


@app.get("/orders", response_model=List[OrderResponse])
async def get_orders(
    limit: int = 100,
    exchange: Optional[str] = None,
    symbol: Optional[str] = None,
    user = Depends(verify_token)
):
    """Get recent orders"""
    if not app_state.storage_manager:
        raise HTTPException(status_code=503, detail="Storage manager not available")
    
    try:
        orders_data = await app_state.storage_manager.get_orders(
            symbol=symbol,
            exchange=exchange,
            limit=limit
        )
        
        orders = []
        for order_data in orders_data:
            orders.append(OrderResponse(
                order_id=order_data['order_id'],
                exchange=order_data['exchange'],
                symbol=order_data['symbol'],
                side=order_data['side'],
                order_type=order_data['order_type'],
                quantity=Decimal(str(order_data['quantity'])),
                price=Decimal(str(order_data['price'])) if order_data['price'] else None,
                status=order_data['status'],
                created_time=order_data['created_time'],
                filled_quantity=Decimal(str(order_data['filled_quantity'] or 0)),
                average_fill_price=Decimal(str(order_data['average_fill_price'])) if order_data['average_fill_price'] else None
            ))
        
        return orders
    
    except Exception as e:
        logger.error(f"Error getting orders: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve orders")


@app.get("/signals", response_model=List[SignalResponse])
async def get_recent_signals(limit: int = 50, user = Depends(verify_token)):
    """Get recent trading signals"""
    # This would need to be implemented based on signal storage
    # For now, return empty list
    return []


@app.get("/risk/status", response_model=RiskStatusResponse)
async def get_risk_status(user = Depends(verify_token)):
    """Get risk management status"""
    if not app_state.risk_manager:
        raise HTTPException(status_code=503, detail="Risk manager not available")
    
    try:
        # Get risk manager status
        status = await app_state.risk_manager.get_status()
        
        return RiskStatusResponse(
            status="active",
            circuit_breaker_active=status.get('circuit_breaker_active', False),
            position_limits_breached=status.get('position_limits_breached', False),
            daily_loss_limit_breached=status.get('daily_loss_limit_breached', False),
            current_daily_pnl=status.get('current_daily_pnl', Decimal("0")),
            max_daily_loss=status.get('max_daily_loss', Decimal("1000")),
            active_alerts=status.get('active_alerts', 0)
        )
    
    except Exception as e:
        logger.error(f"Error getting risk status: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve risk status")


@app.post("/risk/kill-switch")
async def activate_kill_switch(
    request: KillSwitchRequest,
    user = Depends(verify_token)
):
    """Activate emergency kill switch"""
    if not app_state.risk_manager:
        raise HTTPException(status_code=503, detail="Risk manager not available")
    
    try:
        await app_state.risk_manager.activate_kill_switch(request.reason, request.emergency)
        logger.warning(f"Kill switch activated: {request.reason}")
        
        return {"status": "success", "message": "Kill switch activated"}
    
    except Exception as e:
        logger.error(f"Error activating kill switch: {e}")
        raise HTTPException(status_code=500, detail="Failed to activate kill switch")


@app.post("/backtest", response_model=BacktestResponse)
async def run_backtest(
    request: BacktestRequest,
    background_tasks: BackgroundTasks,
    user = Depends(verify_token)
):
    """Run a backtest"""
    backtest_id = f"backtest_{int(time.time())}"
    
    try:
        # Create backtest configuration
        config = BacktestConfig(
            start_date=request.start_date,
            end_date=request.end_date,
            initial_balance={k: Decimal(str(v)) for k, v in request.initial_balance.items()},
            exchanges=request.exchanges,
            symbols=request.symbols,
            enable_slippage=request.enable_slippage,
            enable_latency_simulation=request.enable_latency
        )
        
        # Run backtest in background
        async def run_backtest_task():
            try:
                engine = BacktestEngine(config, app_state.storage_manager)
                result = await engine.run_backtest()
                
                # Store result (in production, store in database)
                logger.info(f"Backtest {backtest_id} completed: {result.total_return_pct:.2f}% return")
            
            except Exception as e:
                logger.error(f"Backtest {backtest_id} failed: {e}")
        
        background_tasks.add_task(run_backtest_task)
        
        return BacktestResponse(
            backtest_id=backtest_id,
            status="running"
        )
    
    except Exception as e:
        logger.error(f"Error starting backtest: {e}")
        raise HTTPException(status_code=500, detail="Failed to start backtest")


@app.get("/portfolio/history")
async def get_portfolio_history(
    hours: int = 24,
    user = Depends(verify_token)
):
    """Get portfolio history"""
    if not app_state.storage_manager:
        raise HTTPException(status_code=503, detail="Storage manager not available")
    
    try:
        history = await app_state.storage_manager.get_portfolio_history(hours)
        return {"history": history}
    
    except Exception as e:
        logger.error(f"Error getting portfolio history: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve portfolio history")


# WebSocket endpoint
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time updates"""
    await websocket.accept()
    app_state.websocket_connections.append(websocket)
    
    try:
        # Send initial status
        await websocket.send_text(json.dumps({
            "type": "connected",
            "message": "WebSocket connection established",
            "timestamp": time.time()
        }))
        
        # Keep connection alive and handle messages
        while True:
            message = await websocket.receive_text()
            
            # Handle client messages (ping, subscription requests, etc.)
            try:
                data = json.loads(message)
                if data.get("type") == "ping":
                    await websocket.send_text(json.dumps({
                        "type": "pong",
                        "timestamp": time.time()
                    }))
            except json.JSONDecodeError:
                pass
    
    except WebSocketDisconnect:
        if websocket in app_state.websocket_connections:
            app_state.websocket_connections.remove(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        if websocket in app_state.websocket_connections:
            app_state.websocket_connections.remove(websocket)


# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return {"error": exc.detail, "status_code": exc.status_code}


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {exc}")
    return {"error": "Internal server error", "status_code": 500}


# Main function for running the server
def main():
    """Run the FastAPI server"""
    settings = get_settings()
    
    uvicorn.run(
        "arbi.api.server:app",
        host=settings.api.host,
        port=settings.api.port,
        log_level="info",
        reload=settings.debug,
        ssl_keyfile=settings.api.key_file if settings.api.use_tls else None,
        ssl_certfile=settings.api.cert_file if settings.api.use_tls else None
    )


if __name__ == "__main__":
    main()
