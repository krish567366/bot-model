"""
Data Storage Module

Handles persistent storage of market data, trades, and portfolio information
using SQLite, Parquet files, and efficient data management.
"""

import asyncio
import logging
import sqlite3
import time
from decimal import Decimal
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import json
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from dataclasses import asdict
import aiosqlite

from .marketdata import BookDelta, Trade, Ticker
from .signal import ArbitrageSignal
from .execution import Order, ExecutionGroup
from .risk import RiskDecision, RiskAlert
from .portfolio import PnLEntry, PortfolioSnapshot
from ..config.settings import get_settings

logger = logging.getLogger(__name__)


class DatabaseManager:
    """SQLite database manager for transactional data"""
    
    def __init__(self, db_path: str = None):
        self.settings = get_settings()
        self.db_path = db_path or "arbitrage.db"
        self.connection_pool: Dict[str, aiosqlite.Connection] = {}
        
    async def initialize(self):
        """Initialize database schema"""
        async with aiosqlite.connect(self.db_path) as db:
            # Orders table
            await db.execute("""
                CREATE TABLE IF NOT EXISTS orders (
                    order_id TEXT PRIMARY KEY,
                    client_order_id TEXT UNIQUE,
                    exchange TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    side TEXT NOT NULL,
                    order_type TEXT NOT NULL,
                    quantity DECIMAL NOT NULL,
                    price DECIMAL,
                    status TEXT NOT NULL,
                    created_time REAL NOT NULL,
                    submitted_time REAL,
                    filled_time REAL,
                    filled_quantity DECIMAL DEFAULT 0,
                    average_fill_price DECIMAL,
                    signal_id TEXT,
                    execution_group_id TEXT,
                    exchange_order_id TEXT,
                    error_message TEXT,
                    metadata TEXT
                )
            """)
            
            # Trades table
            await db.execute("""
                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    exchange TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    trade_id TEXT NOT NULL,
                    timestamp REAL NOT NULL,
                    price DECIMAL NOT NULL,
                    size DECIMAL NOT NULL,
                    side TEXT NOT NULL,
                    buyer_maker BOOLEAN,
                    order_id TEXT,
                    UNIQUE(exchange, trade_id)
                )
            """)
            
            # Signals table
            await db.execute("""
                CREATE TABLE IF NOT EXISTS signals (
                    signal_id TEXT PRIMARY KEY,
                    signal_type TEXT NOT NULL,
                    timestamp REAL NOT NULL,
                    symbol TEXT NOT NULL,
                    buy_exchange TEXT NOT NULL,
                    sell_exchange TEXT NOT NULL,
                    expected_profit_pct DECIMAL NOT NULL,
                    confidence REAL NOT NULL,
                    buy_price DECIMAL NOT NULL,
                    sell_price DECIMAL NOT NULL,
                    size DECIMAL NOT NULL,
                    executed BOOLEAN DEFAULT FALSE,
                    metadata TEXT
                )
            """)
            
            # Risk decisions table
            await db.execute("""
                CREATE TABLE IF NOT EXISTS risk_decisions (
                    decision_id TEXT PRIMARY KEY,
                    timestamp REAL NOT NULL,
                    signal_id TEXT,
                    symbol TEXT NOT NULL,
                    exchange TEXT NOT NULL,
                    risk_level TEXT NOT NULL,
                    action TEXT NOT NULL,
                    reasoning TEXT NOT NULL,
                    original_size DECIMAL,
                    approved_size DECIMAL,
                    rejected BOOLEAN DEFAULT FALSE,
                    risk_factors TEXT
                )
            """)
            
            # Risk alerts table
            await db.execute("""
                CREATE TABLE IF NOT EXISTS risk_alerts (
                    alert_id TEXT PRIMARY KEY,
                    timestamp REAL NOT NULL,
                    alert_type TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    message TEXT NOT NULL,
                    symbol TEXT,
                    exchange TEXT,
                    current_value DECIMAL,
                    limit_value DECIMAL,
                    signal_id TEXT,
                    resolved BOOLEAN DEFAULT FALSE
                )
            """)
            
            # P&L entries table
            await db.execute("""
                CREATE TABLE IF NOT EXISTS pnl_entries (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL NOT NULL,
                    symbol TEXT NOT NULL,
                    exchange TEXT NOT NULL,
                    quantity DECIMAL NOT NULL,
                    price DECIMAL NOT NULL,
                    side TEXT NOT NULL,
                    realized_pnl DECIMAL DEFAULT 0,
                    unrealized_pnl DECIMAL DEFAULT 0,
                    fees DECIMAL DEFAULT 0,
                    trade_id TEXT,
                    order_id TEXT,
                    signal_id TEXT
                )
            """)
            
            # Portfolio snapshots table
            await db.execute("""
                CREATE TABLE IF NOT EXISTS portfolio_snapshots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL NOT NULL,
                    total_value_usd DECIMAL NOT NULL,
                    total_unrealized_pnl DECIMAL NOT NULL,
                    total_realized_pnl DECIMAL NOT NULL,
                    daily_pnl DECIMAL NOT NULL,
                    num_positions INTEGER NOT NULL,
                    leverage REAL NOT NULL,
                    max_drawdown_pct REAL NOT NULL,
                    metadata TEXT
                )
            """)
            
            # Create indexes
            await db.execute("CREATE INDEX IF NOT EXISTS idx_orders_timestamp ON orders(created_time)")
            await db.execute("CREATE INDEX IF NOT EXISTS idx_orders_symbol ON orders(symbol)")
            await db.execute("CREATE INDEX IF NOT EXISTS idx_trades_timestamp ON trades(timestamp)")
            await db.execute("CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trades(symbol)")
            await db.execute("CREATE INDEX IF NOT EXISTS idx_signals_timestamp ON signals(timestamp)")
            await db.execute("CREATE INDEX IF NOT EXISTS idx_pnl_timestamp ON pnl_entries(timestamp)")
            
            await db.commit()
        
        logger.info(f"Database initialized: {self.db_path}")
    
    async def insert_order(self, order: Order):
        """Insert order into database"""
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                INSERT OR REPLACE INTO orders (
                    order_id, client_order_id, exchange, symbol, side, order_type,
                    quantity, price, status, created_time, submitted_time, filled_time,
                    filled_quantity, average_fill_price, signal_id, execution_group_id,
                    exchange_order_id, error_message, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                order.order_id, order.client_order_id, order.exchange, order.symbol,
                order.side.value, order.order_type.value, str(order.quantity),
                str(order.price) if order.price else None, order.status.value,
                order.created_time, order.submitted_time, order.filled_time,
                str(order.filled_quantity), str(order.average_fill_price) if order.average_fill_price else None,
                order.signal_id, order.execution_group_id, order.exchange_order_id,
                order.error_message, json.dumps(order.exchange_metadata)
            ))
            await db.commit()
    
    async def insert_trade(self, trade: Trade):
        """Insert trade into database"""
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                INSERT OR IGNORE INTO trades (
                    exchange, symbol, trade_id, timestamp, price, size, side, buyer_maker
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                trade.exchange, trade.symbol, trade.trade_id, trade.timestamp.timestamp(),
                str(trade.price), str(trade.size), trade.side.value, trade.buyer_maker
            ))
            await db.commit()
    
    async def insert_signal(self, signal: ArbitrageSignal):
        """Insert signal into database"""
        metadata = {
            'strength': signal.strength.value if hasattr(signal.strength, 'value') else str(signal.strength),
            'slippage_estimate': str(signal.slippage_estimate),
            'fees_total': str(signal.fees_total),
            'mid_price_ref': str(signal.mid_price_ref),
            'spread_width_pct': str(signal.spread_width_pct)
        }
        
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                INSERT OR REPLACE INTO signals (
                    signal_id, signal_type, timestamp, symbol, buy_exchange, sell_exchange,
                    expected_profit_pct, confidence, buy_price, sell_price, size, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                signal.signal_id, signal.signal_type.value, signal.timestamp, signal.symbol,
                signal.buy_exchange, signal.sell_exchange, str(signal.expected_profit_pct),
                signal.confidence, str(signal.buy_price), str(signal.sell_price),
                str(signal.size), json.dumps(metadata)
            ))
            await db.commit()
    
    async def insert_risk_decision(self, decision: RiskDecision):
        """Insert risk decision into database"""
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                INSERT INTO risk_decisions (
                    decision_id, timestamp, signal_id, symbol, exchange, risk_level,
                    action, reasoning, original_size, approved_size, rejected, risk_factors
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                decision.decision_id, decision.timestamp, decision.signal_id,
                decision.symbol, decision.exchange, decision.risk_level.value,
                decision.action.value, decision.reasoning,
                str(decision.original_size) if decision.original_size else None,
                str(decision.approved_size) if decision.approved_size else None,
                decision.rejected, json.dumps(decision.risk_factors)
            ))
            await db.commit()
    
    async def insert_risk_alert(self, alert: RiskAlert):
        """Insert risk alert into database"""
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                INSERT INTO risk_alerts (
                    alert_id, timestamp, alert_type, severity, message, symbol,
                    exchange, current_value, limit_value, signal_id, resolved
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                alert.alert_id, alert.timestamp, alert.alert_type.value,
                alert.severity.value, alert.message, alert.symbol, alert.exchange,
                str(alert.current_value) if alert.current_value else None,
                str(alert.limit_value) if alert.limit_value else None,
                alert.signal_id, alert.resolved
            ))
            await db.commit()
    
    async def insert_pnl_entry(self, entry: PnLEntry):
        """Insert P&L entry into database"""
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                INSERT INTO pnl_entries (
                    timestamp, symbol, exchange, quantity, price, side,
                    realized_pnl, unrealized_pnl, fees, trade_id, order_id, signal_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                entry.timestamp, entry.symbol, entry.exchange, str(entry.quantity),
                str(entry.price), entry.side, str(entry.realized_pnl),
                str(entry.unrealized_pnl), str(entry.fees), entry.trade_id,
                entry.order_id, entry.signal_id
            ))
            await db.commit()
    
    async def insert_portfolio_snapshot(self, snapshot: PortfolioSnapshot):
        """Insert portfolio snapshot into database"""
        metadata = {
            'total_cash_usd': str(snapshot.total_cash_usd),
            'total_positions_usd': str(snapshot.total_positions_usd),
            'total_return_pct': snapshot.total_return_pct,
            'daily_return_pct': snapshot.daily_return_pct,
            'sharpe_ratio': snapshot.sharpe_ratio,
            'num_currencies': snapshot.num_currencies,
            'exchange_values': {k: str(v) for k, v in snapshot.exchange_values.items()},
            'currency_exposure': {k: str(v) for k, v in snapshot.currency_exposure.items()}
        }
        
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                INSERT INTO portfolio_snapshots (
                    timestamp, total_value_usd, total_unrealized_pnl, total_realized_pnl,
                    daily_pnl, num_positions, leverage, max_drawdown_pct, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                snapshot.timestamp, str(snapshot.total_value_usd),
                str(snapshot.total_unrealized_pnl), str(snapshot.total_realized_pnl),
                str(snapshot.daily_pnl), snapshot.num_positions, snapshot.leverage,
                snapshot.max_drawdown_pct, json.dumps(metadata)
            ))
            await db.commit()
    
    async def get_orders(self, symbol: str = None, exchange: str = None, 
                        start_time: float = None, limit: int = 100) -> List[Dict]:
        """Query orders from database"""
        query = "SELECT * FROM orders WHERE 1=1"
        params = []
        
        if symbol:
            query += " AND symbol = ?"
            params.append(symbol)
        if exchange:
            query += " AND exchange = ?"
            params.append(exchange)
        if start_time:
            query += " AND created_time >= ?"
            params.append(start_time)
        
        query += " ORDER BY created_time DESC LIMIT ?"
        params.append(limit)
        
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(query, params) as cursor:
                rows = await cursor.fetchall()
                return [dict(row) for row in rows]
    
    async def get_portfolio_history(self, hours: int = 24) -> List[Dict]:
        """Get portfolio snapshot history"""
        start_time = time.time() - (hours * 3600)
        
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute("""
                SELECT * FROM portfolio_snapshots 
                WHERE timestamp >= ? 
                ORDER BY timestamp ASC
            """, (start_time,)) as cursor:
                rows = await cursor.fetchall()
                return [dict(row) for row in rows]
    
    async def cleanup_old_data(self, days: int = None):
        """Clean up old data based on retention policy"""
        if days is None:
            days = self.settings.data.data_retention_days
        
        cutoff_time = time.time() - (days * 24 * 3600)
        
        async with aiosqlite.connect(self.db_path) as db:
            # Clean up old trades (keep market data longer)
            await db.execute("DELETE FROM trades WHERE timestamp < ?", (cutoff_time,))
            
            # Clean up old risk decisions
            await db.execute("DELETE FROM risk_decisions WHERE timestamp < ?", (cutoff_time,))
            
            # Clean up resolved risk alerts
            alert_cutoff = time.time() - (7 * 24 * 3600)  # Keep alerts for 7 days
            await db.execute("DELETE FROM risk_alerts WHERE timestamp < ? AND resolved = TRUE", (alert_cutoff,))
            
            await db.commit()
        
        logger.info(f"Cleaned up data older than {days} days")


class ParquetStorage:
    """Parquet file storage for market data"""
    
    def __init__(self, data_dir: str = "data"):
        self.settings = get_settings()
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
        # Schemas for different data types
        self.book_delta_schema = pa.schema([
            ('timestamp', pa.timestamp('ms')),
            ('exchange', pa.string()),
            ('symbol', pa.string()),
            ('sequence', pa.int64()),
            ('is_snapshot', pa.bool_()),
            ('bid_prices', pa.list_(pa.float64())),
            ('bid_sizes', pa.list_(pa.float64())),
            ('ask_prices', pa.list_(pa.float64())),
            ('ask_sizes', pa.list_(pa.float64())),
            ('latency_ms', pa.float64())
        ])
        
        self.ticker_schema = pa.schema([
            ('timestamp', pa.timestamp('ms')),
            ('exchange', pa.string()),
            ('symbol', pa.string()),
            ('last_price', pa.float64()),
            ('bid', pa.float64()),
            ('ask', pa.float64()),
            ('volume_24h', pa.float64()),
            ('high_24h', pa.float64()),
            ('low_24h', pa.float64()),
            ('change_24h_pct', pa.float64())
        ])
    
    def _get_file_path(self, data_type: str, date: str, exchange: str = None) -> Path:
        """Generate file path for data storage"""
        if exchange:
            return self.data_dir / data_type / exchange / f"{date}.parquet"
        else:
            return self.data_dir / data_type / f"{date}.parquet"
    
    async def store_book_deltas(self, deltas: List[BookDelta]):
        """Store book deltas to Parquet files"""
        if not deltas:
            return
        
        # Group by exchange and date
        grouped_data = {}
        for delta in deltas:
            date_str = datetime.fromtimestamp(delta.timestamp).strftime("%Y-%m-%d")
            key = (delta.exchange, date_str)
            
            if key not in grouped_data:
                grouped_data[key] = []
            
            # Convert to flat structure
            record = {
                'timestamp': pd.Timestamp.fromtimestamp(delta.timestamp),
                'exchange': delta.exchange,
                'symbol': delta.symbol,
                'sequence': delta.sequence,
                'is_snapshot': delta.is_snapshot,
                'bid_prices': [float(level.price) for level in delta.bids],
                'bid_sizes': [float(level.size) for level in delta.bids],
                'ask_prices': [float(level.price) for level in delta.asks],
                'ask_sizes': [float(level.size) for level in delta.asks],
                'latency_ms': delta.latency_ms
            }
            grouped_data[key].append(record)
        
        # Write to files
        for (exchange, date_str), records in grouped_data.items():
            file_path = self._get_file_path("book_deltas", date_str, exchange)
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Convert to DataFrame and write
            df = pd.DataFrame(records)
            table = pa.Table.from_pandas(df, schema=self.book_delta_schema)
            
            # Append to existing file or create new one
            if file_path.exists():
                existing_table = pq.read_table(file_path)
                table = pa.concat_tables([existing_table, table])
            
            pq.write_table(
                table, 
                file_path, 
                compression=self.settings.data.parquet_compression
            )
    
    async def store_tickers(self, tickers: List[Ticker]):
        """Store ticker data to Parquet files"""
        if not tickers:
            return
        
        # Group by date
        grouped_data = {}
        for ticker in tickers:
            date_str = datetime.fromtimestamp(ticker.timestamp).strftime("%Y-%m-%d")
            
            if date_str not in grouped_data:
                grouped_data[date_str] = []
            
            record = {
                'timestamp': pd.Timestamp.fromtimestamp(ticker.timestamp),
                'exchange': ticker.exchange,
                'symbol': ticker.symbol,
                'last_price': float(ticker.last_price) if ticker.last_price else None,
                'bid': float(ticker.bid) if ticker.bid else None,
                'ask': float(ticker.ask) if ticker.ask else None,
                'volume_24h': float(ticker.volume_24h) if ticker.volume_24h else None,
                'high_24h': float(ticker.high_24h) if ticker.high_24h else None,
                'low_24h': float(ticker.low_24h) if ticker.low_24h else None,
                'change_24h_pct': float(ticker.change_pct_24h) if ticker.change_pct_24h else None
            }
            grouped_data[date_str].append(record)
        
        # Write to files
        for date_str, records in grouped_data.items():
            file_path = self._get_file_path("tickers", date_str)
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            df = pd.DataFrame(records)
            table = pa.Table.from_pandas(df, schema=self.ticker_schema)
            
            if file_path.exists():
                existing_table = pq.read_table(file_path)
                table = pa.concat_tables([existing_table, table])
            
            pq.write_table(
                table, 
                file_path, 
                compression=self.settings.data.parquet_compression
            )
    
    async def read_book_deltas(self, exchange: str, symbol: str, 
                             start_date: str, end_date: str) -> pd.DataFrame:
        """Read book deltas from Parquet files"""
        dfs = []
        
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")
        
        current_date = start_dt
        while current_date <= end_dt:
            date_str = current_date.strftime("%Y-%m-%d")
            file_path = self._get_file_path("book_deltas", date_str, exchange)
            
            if file_path.exists():
                df = pd.read_parquet(file_path)
                if symbol:
                    df = df[df['symbol'] == symbol]
                dfs.append(df)
            
            current_date += timedelta(days=1)
        
        if dfs:
            return pd.concat(dfs, ignore_index=True).sort_values('timestamp')
        else:
            return pd.DataFrame()
    
    async def cleanup_old_files(self, days: int = None):
        """Clean up old Parquet files"""
        if days is None:
            days = self.settings.data.data_retention_days
        
        cutoff_date = datetime.now() - timedelta(days=days)
        
        for data_type_dir in self.data_dir.iterdir():
            if not data_type_dir.is_dir():
                continue
            
            for file_path in data_type_dir.rglob("*.parquet"):
                try:
                    # Extract date from filename
                    date_str = file_path.stem
                    file_date = datetime.strptime(date_str, "%Y-%m-%d")
                    
                    if file_date < cutoff_date:
                        file_path.unlink()
                        logger.debug(f"Deleted old file: {file_path}")
                
                except (ValueError, OSError) as e:
                    logger.warning(f"Error processing file {file_path}: {e}")


class StorageManager:
    """Main storage coordinator"""
    
    def __init__(self, db_path: str = None, data_dir: str = "data"):
        self.settings = get_settings()
        self.db_manager = DatabaseManager(db_path)
        self.parquet_storage = ParquetStorage(data_dir)
        
        # Buffering for batch operations
        self.book_delta_buffer: List[BookDelta] = []
        self.ticker_buffer: List[Ticker] = []
        self.buffer_size = self.settings.performance.data_buffer_size
        
        self.last_cleanup = time.time()
        self.cleanup_interval = 24 * 3600  # Daily cleanup
    
    async def initialize(self):
        """Initialize storage systems"""
        await self.db_manager.initialize()
        logger.info("Storage manager initialized")
    
    async def store_book_delta(self, delta: BookDelta):
        """Store book delta (buffered)"""
        if self.settings.data.enable_data_recording:
            self.book_delta_buffer.append(delta)
            
            if len(self.book_delta_buffer) >= self.buffer_size:
                await self._flush_book_deltas()
    
    async def store_ticker(self, ticker: Ticker):
        """Store ticker data (buffered)"""
        if self.settings.data.enable_data_recording:
            self.ticker_buffer.append(ticker)
            
            if len(self.ticker_buffer) >= self.buffer_size:
                await self._flush_tickers()
    
    async def _flush_book_deltas(self):
        """Flush book delta buffer to storage"""
        if self.book_delta_buffer:
            await self.parquet_storage.store_book_deltas(self.book_delta_buffer)
            self.book_delta_buffer.clear()
            logger.debug("Flushed book delta buffer")
    
    async def _flush_tickers(self):
        """Flush ticker buffer to storage"""
        if self.ticker_buffer:
            await self.parquet_storage.store_tickers(self.ticker_buffer)
            self.ticker_buffer.clear()
            logger.debug("Flushed ticker buffer")
    
    async def flush_all_buffers(self):
        """Force flush all buffers"""
        await self._flush_book_deltas()
        await self._flush_tickers()
    
    # Database operations (direct, not buffered)
    async def store_order(self, order: Order):
        """Store order to database"""
        await self.db_manager.insert_order(order)
    
    async def store_signal(self, signal: ArbitrageSignal):
        """Store signal to database"""
        await self.db_manager.insert_signal(signal)
    
    async def store_risk_decision(self, decision: RiskDecision):
        """Store risk decision to database"""
        await self.db_manager.insert_risk_decision(decision)
    
    async def store_risk_alert(self, alert: RiskAlert):
        """Store risk alert to database"""
        await self.db_manager.insert_risk_alert(alert)
    
    async def store_pnl_entry(self, entry: PnLEntry):
        """Store P&L entry to database"""
        await self.db_manager.insert_pnl_entry(entry)
    
    async def store_portfolio_snapshot(self, snapshot: PortfolioSnapshot):
        """Store portfolio snapshot to database"""
        await self.db_manager.insert_portfolio_snapshot(snapshot)
    
    # Query operations
    async def get_orders(self, **kwargs) -> List[Dict]:
        """Get orders from database"""
        return await self.db_manager.get_orders(**kwargs)
    
    async def get_portfolio_history(self, hours: int = 24) -> List[Dict]:
        """Get portfolio history"""
        return await self.db_manager.get_portfolio_history(hours)
    
    async def get_market_data(self, exchange: str, symbol: str, 
                            start_date: str, end_date: str) -> pd.DataFrame:
        """Get historical market data"""
        return await self.parquet_storage.read_book_deltas(exchange, symbol, start_date, end_date)
    
    def save_table(self, df: pd.DataFrame, table_name: str, if_exists: str = 'replace'):
        """Save DataFrame to SQLite table (synchronous for compatibility)"""
        import sqlite3
        
        # Use the database path from db_manager
        db_path = self.db_manager.db_path
        
        try:
            with sqlite3.connect(db_path) as conn:
                df.to_sql(table_name, conn, if_exists=if_exists, index=False)
        except Exception as e:
            logger.error(f"Error saving table {table_name}: {e}")
            raise
    
    def load_table(self, table_name: str) -> pd.DataFrame:
        """Load DataFrame from SQLite table (synchronous for compatibility)"""
        import sqlite3
        
        # Use the database path from db_manager
        db_path = self.db_manager.db_path
        
        try:
            with sqlite3.connect(db_path) as conn:
                return pd.read_sql(f"SELECT * FROM {table_name}", conn)
        except Exception as e:
            logger.error(f"Error loading table {table_name}: {e}")
            return pd.DataFrame()
    
    async def periodic_maintenance(self):
        """Perform periodic maintenance tasks"""
        current_time = time.time()
        
        if current_time - self.last_cleanup >= self.cleanup_interval:
            logger.info("Starting periodic maintenance...")
            
            # Flush buffers
            await self.flush_all_buffers()
            
            # Cleanup old data
            await self.db_manager.cleanup_old_data()
            await self.parquet_storage.cleanup_old_files()
            
            self.last_cleanup = current_time
            logger.info("Periodic maintenance completed")
    
    async def create_backup(self, backup_dir: str = None) -> str:
        """Create backup of all data"""
        import shutil
        from datetime import datetime
        
        if backup_dir is None:
            backup_dir = self.settings.data.backup_location
        
        backup_path = Path(backup_dir) / f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        backup_path.mkdir(parents=True, exist_ok=True)
        
        # Flush all buffers first
        await self.flush_all_buffers()
        
        # Copy database
        shutil.copy2(self.db_manager.db_path, backup_path / "arbitrage.db")
        
        # Copy Parquet files
        if self.parquet_storage.data_dir.exists():
            shutil.copytree(
                self.parquet_storage.data_dir,
                backup_path / "data",
                dirs_exist_ok=True
            )
        
        logger.info(f"Backup created: {backup_path}")
        return str(backup_path)
    
    async def start_monitoring(self):
        """Start storage monitoring and maintenance"""
        logger.info("Starting storage monitoring")
        
        while True:
            try:
                await self.periodic_maintenance()
                await asyncio.sleep(3600)  # Check every hour
            except Exception as e:
                logger.error(f"Error in storage monitoring: {e}")
                await asyncio.sleep(600)  # Retry in 10 minutes


# Initialize function for external use
async def init_database(db_path: str = None):
    """Initialize database schema"""
    db_manager = DatabaseManager(db_path)
    await db_manager.initialize()


# Example usage
async def main():
    """Example usage of storage system"""
    logging.basicConfig(level=logging.INFO)
    
    storage = StorageManager()
    await storage.initialize()
    
    # Test storing some data
    from .marketdata import BookDelta, OrderBookLevel, Ticker
    
    # Create sample book delta
    delta = BookDelta(
        exchange="binance",
        symbol="BTC/USDT",
        bids=[OrderBookLevel(price=Decimal("50000"), size=Decimal("1.0"))],
        asks=[OrderBookLevel(price=Decimal("50100"), size=Decimal("0.5"))]
    )
    
    await storage.store_book_delta(delta)
    
    # Create sample ticker
    ticker = Ticker(
        exchange="binance",
        symbol="BTC/USDT",
        last_price=Decimal("50050"),
        bid=Decimal("50000"),
        ask=Decimal("50100")
    )
    
    await storage.store_ticker(ticker)
    
    # Flush buffers
    await storage.flush_all_buffers()
    
    logger.info("Storage test completed")


# Global storage manager instance
_storage_manager: Optional[StorageManager] = None


def get_storage_manager() -> StorageManager:
    """Get singleton storage manager instance"""
    global _storage_manager
    
    if _storage_manager is None:
        _storage_manager = StorageManager()
    
    return _storage_manager


if __name__ == "__main__":
    asyncio.run(main())
