"""
Data Pipeline Module

Comprehensive data pipeline for market data ingestion, cleaning, transformation, and storage.
Integrates with signal generation, AI/ML models, backtesting, and risk management.

Features:
- Multi-source data ingestion (yfinance, Alpha Vantage, Yahoo Query, CCXT)
- Data validation and cleaning
- Feature engineering with technical indicators
- Flexible storage (SQLite/PostgreSQL)
- Real-time and historical data processing
- ML-ready data preparation
"""

import asyncio
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Tuple, Any
from decimal import Decimal
import time
import sqlite3
from pathlib import Path
import warnings

# Data sources
import yfinance as yf
from alpha_vantage.timeseries import TimeSeries
from alpha_vantage.foreignexchange import ForeignExchange
from alpha_vantage.cryptocurrencies import CryptoCurrencies

# Database
from sqlalchemy import create_engine, MetaData, Table, Column, String, Float, DateTime, Integer, Text, inspect
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import SQLAlchemyError

# Technical indicators
import ta
from ta import add_all_ta_features

# Optional CCXT for crypto
try:
    import ccxt
    CCXT_AVAILABLE = True
except ImportError:
    CCXT_AVAILABLE = False

# Configuration
from ..config.settings import get_settings

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


class DataSource:
    """Base class for data sources"""
    
    def __init__(self, name: str):
        self.name = name
        self.rate_limits = {}
        self.last_request = {}
    
    def check_rate_limit(self, endpoint: str, limit_per_minute: int = 60):
        """Check if we can make a request without hitting rate limits"""
        current_time = time.time()
        key = f"{self.name}_{endpoint}"
        
        if key not in self.last_request:
            self.last_request[key] = []
        
        # Remove old requests (older than 1 minute)
        cutoff_time = current_time - 60
        self.last_request[key] = [t for t in self.last_request[key] if t > cutoff_time]
        
        if len(self.last_request[key]) >= limit_per_minute:
            return False
        
        self.last_request[key].append(current_time)
        return True
    
    async def fetch_data(self, symbol: str, **kwargs) -> pd.DataFrame:
        """Abstract method for fetching data"""
        raise NotImplementedError


class YFinanceSource(DataSource):
    """Yahoo Finance data source"""
    
    def __init__(self):
        super().__init__("yfinance")
    
    async def fetch_historical(self, symbol: str, period: str = "1y", 
                             interval: str = "1d", start: str = None, 
                             end: str = None) -> pd.DataFrame:
        """Fetch historical OHLCV data"""
        try:
            if not self.check_rate_limit("historical", 10):  # Conservative limit
                await asyncio.sleep(6)  # Wait 6 seconds
            
            ticker = yf.Ticker(symbol)
            
            if start and end:
                df = ticker.history(start=start, end=end, interval=interval)
            else:
                df = ticker.history(period=period, interval=interval)
            
            if df.empty:
                logger.warning(f"No data returned for {symbol}")
                return pd.DataFrame()
            
            # Standardize column names
            df.columns = df.columns.str.lower()
            df.reset_index(inplace=True)
            
            # Add metadata
            df['symbol'] = symbol
            df['source'] = self.name
            df['interval'] = interval
            
            logger.info(f"Fetched {len(df)} rows for {symbol} from yfinance")
            return df
            
        except Exception as e:
            logger.error(f"Error fetching data from yfinance for {symbol}: {e}")
            return pd.DataFrame()
    
    async def fetch_info(self, symbol: str) -> Dict:
        """Fetch ticker info and fundamentals"""
        try:
            if not self.check_rate_limit("info", 5):
                await asyncio.sleep(12)
            
            ticker = yf.Ticker(symbol)
            info = ticker.info
            return info
            
        except Exception as e:
            logger.error(f"Error fetching info for {symbol}: {e}")
            return {}


class AlphaVantageSource(DataSource):
    """Alpha Vantage data source"""
    
    def __init__(self, api_key: str):
        super().__init__("alpha_vantage")
        self.api_key = api_key
        self.ts = TimeSeries(key=api_key, output_format='pandas')
        self.fx = ForeignExchange(key=api_key, output_format='pandas')
        self.crypto = CryptoCurrencies(key=api_key, output_format='pandas')
    
    async def fetch_daily(self, symbol: str, outputsize: str = "compact") -> pd.DataFrame:
        """Fetch daily time series data"""
        try:
            if not self.check_rate_limit("daily", 5):  # Alpha Vantage limit
                await asyncio.sleep(12)
            
            data, meta_data = self.ts.get_daily_adjusted(symbol=symbol, outputsize=outputsize)
            
            if data.empty:
                return pd.DataFrame()
            
            # Standardize format
            data.reset_index(inplace=True)
            data.columns = ['date', 'open', 'high', 'low', 'close', 'adjusted_close', 'volume', 'dividend_amount', 'split_coefficient']
            data['symbol'] = symbol
            data['source'] = self.name
            data['interval'] = '1d'
            
            logger.info(f"Fetched {len(data)} rows for {symbol} from Alpha Vantage")
            return data
            
        except Exception as e:
            logger.error(f"Error fetching data from Alpha Vantage for {symbol}: {e}")
            return pd.DataFrame()
    
    async def fetch_forex(self, from_symbol: str, to_symbol: str) -> pd.DataFrame:
        """Fetch forex data"""
        try:
            if not self.check_rate_limit("forex", 5):
                await asyncio.sleep(12)
            
            data, meta_data = self.fx.get_fx_daily(from_symbol=from_symbol, to_symbol=to_symbol)
            
            if data.empty:
                return pd.DataFrame()
            
            data.reset_index(inplace=True)
            data.columns = ['date', 'open', 'high', 'low', 'close']
            data['symbol'] = f"{from_symbol}/{to_symbol}"
            data['source'] = self.name
            data['interval'] = '1d'
            
            return data
            
        except Exception as e:
            logger.error(f"Error fetching forex data: {e}")
            return pd.DataFrame()
    
    async def fetch_crypto(self, symbol: str, market: str = "USD") -> pd.DataFrame:
        """Fetch cryptocurrency data"""
        try:
            if not self.check_rate_limit("crypto", 5):
                await asyncio.sleep(12)
            
            data, meta_data = self.crypto.get_digital_currency_daily(symbol=symbol, market=market)
            
            if data.empty:
                return pd.DataFrame()
            
            # Standardize format - Alpha Vantage crypto has different column structure
            data.reset_index(inplace=True)
            data = data[['date', f'1a. open ({market})', f'2a. high ({market})', 
                        f'3a. low ({market})', f'4a. close ({market})', '5. volume']]
            data.columns = ['date', 'open', 'high', 'low', 'close', 'volume']
            data['symbol'] = f"{symbol}/{market}"
            data['source'] = self.name
            data['interval'] = '1d'
            
            return data
            
        except Exception as e:
            logger.error(f"Error fetching crypto data: {e}")
            return pd.DataFrame()


class CCXTSource(DataSource):
    """CCXT cryptocurrency exchange source"""
    
    def __init__(self):
        super().__init__("ccxt")
        self.exchanges = {}
        
        if CCXT_AVAILABLE:
            # Initialize major exchanges
            try:
                self.exchanges['binance'] = ccxt.binance()
                self.exchanges['kraken'] = ccxt.kraken()
                # Updated: coinbasepro is deprecated, use coinbase instead
                self.exchanges['coinbase'] = ccxt.coinbase()
                logger.info("CCXT exchanges initialized")
            except Exception as e:
                logger.error(f"Error initializing CCXT exchanges: {e}")
    
    async def fetch_ohlcv(self, exchange: str, symbol: str, timeframe: str = '1d', 
                         limit: int = 1000) -> pd.DataFrame:
        """Fetch OHLCV data from crypto exchange"""
        if not CCXT_AVAILABLE or exchange not in self.exchanges:
            return pd.DataFrame()
        
        try:
            if not self.check_rate_limit(f"{exchange}_ohlcv", 10):
                await asyncio.sleep(6)
            
            exchange_obj = self.exchanges[exchange]
            ohlcv = exchange_obj.fetch_ohlcv(symbol, timeframe, limit=limit)
            
            if not ohlcv:
                return pd.DataFrame()
            
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['date'] = pd.to_datetime(df['timestamp'], unit='ms')
            df['symbol'] = symbol
            df['exchange'] = exchange
            df['source'] = self.name
            df['interval'] = timeframe
            
            df = df.drop('timestamp', axis=1)
            
            logger.info(f"Fetched {len(df)} OHLCV rows for {symbol} from {exchange}")
            return df
            
        except Exception as e:
            logger.error(f"Error fetching OHLCV from {exchange}: {e}")
            return pd.DataFrame()


class DataValidator:
    """Data validation and cleaning utilities"""
    
    @staticmethod
    def validate_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
        """Validate OHLCV data integrity"""
        if df.empty:
            return df
        
        original_len = len(df)
        
        # Remove rows with missing critical data
        required_cols = ['open', 'high', 'low', 'close']
        df = df.dropna(subset=required_cols)
        
        # Remove rows where high < low (invalid)
        df = df[df['high'] >= df['low']]
        
        # Remove rows where close is outside high-low range
        df = df[(df['close'] >= df['low']) & (df['close'] <= df['high'])]
        df = df[(df['open'] >= df['low']) & (df['open'] <= df['high'])]
        
        # Remove extreme outliers (price changes > 50% in one period)
        if len(df) > 1:
            df = df.sort_values('date')
            df['price_change'] = df['close'].pct_change().abs()
            df = df[df['price_change'] < 0.5]  # Remove > 50% changes
            df = df.drop('price_change', axis=1)
        
        # Remove duplicates based on date and symbol
        if 'date' in df.columns and 'symbol' in df.columns:
            df = df.drop_duplicates(subset=['date', 'symbol'], keep='last')
        
        cleaned_len = len(df)
        if cleaned_len != original_len:
            logger.info(f"Data cleaning: {original_len} -> {cleaned_len} rows ({original_len - cleaned_len} removed)")
        
        return df
    
    @staticmethod
    def fill_missing_values(df: pd.DataFrame) -> pd.DataFrame:
        """Fill missing values using appropriate methods"""
        if df.empty:
            return df
        
        # Forward fill for price data (carry forward last known price)
        price_cols = ['open', 'high', 'low', 'close']
        for col in price_cols:
            if col in df.columns:
                df[col] = df[col].fillna(method='ffill')
        
        # Volume can be filled with 0 or median
        if 'volume' in df.columns:
            df['volume'] = df['volume'].fillna(0)
        
        return df
    
    @staticmethod
    def align_timezones(df: pd.DataFrame, target_tz: str = 'UTC') -> pd.DataFrame:
        """Align all timestamps to a common timezone"""
        if df.empty or 'date' not in df.columns:
            return df
        
        try:
            # Convert to datetime if not already
            if not pd.api.types.is_datetime64_any_dtype(df['date']):
                df['date'] = pd.to_datetime(df['date'])
            
            # Convert to target timezone
            if df['date'].dt.tz is None:
                df['date'] = df['date'].dt.tz_localize('UTC')
            else:
                df['date'] = df['date'].dt.tz_convert(target_tz)
                
        except Exception as e:
            logger.error(f"Error aligning timezones: {e}")
        
        return df


class FeatureEngineering:
    """Feature engineering and technical indicator calculations"""
    
    @staticmethod
    def add_basic_features(df: pd.DataFrame) -> pd.DataFrame:
        """Add basic derived features"""
        if df.empty or not all(col in df.columns for col in ['open', 'high', 'low', 'close', 'volume']):
            return df
        
        try:
            # Price-based features
            df['hl_avg'] = (df['high'] + df['low']) / 2
            df['ohlc_avg'] = (df['open'] + df['high'] + df['low'] + df['close']) / 4
            df['price_range'] = df['high'] - df['low']
            df['price_range_pct'] = (df['price_range'] / df['close']) * 100
            
            # Returns
            df['returns'] = df['close'].pct_change()
            df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
            
            # Volume features
            if 'volume' in df.columns:
                df['volume_sma_10'] = df['volume'].rolling(window=10).mean()
                df['volume_ratio'] = df['volume'] / df['volume_sma_10']
                df['price_volume'] = df['close'] * df['volume']
            
            # Volatility
            df['volatility_10'] = df['returns'].rolling(window=10).std()
            df['volatility_30'] = df['returns'].rolling(window=30).std()
            
        except Exception as e:
            logger.error(f"Error adding basic features: {e}")
        
        return df
    
    @staticmethod
    def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """Add comprehensive technical indicators using ta library"""
        if df.empty or not all(col in df.columns for col in ['open', 'high', 'low', 'close']):
            return df
        
        try:
            # Use ta library to add all indicators
            df = add_all_ta_features(df, open="open", high="high", low="low", 
                                   close="close", volume="volume" if "volume" in df.columns else None,
                                   fillna=True)
            
            # Add custom indicators
            df = FeatureEngineering._add_custom_indicators(df)
            
        except Exception as e:
            logger.error(f"Error adding technical indicators: {e}")
        
        return df
    
    @staticmethod
    def _add_custom_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """Add custom technical indicators not in ta library"""
        try:
            # Williams %R
            df['williams_r'] = ta.momentum.WilliamsRIndicator(
                high=df['high'], low=df['low'], close=df['close']).williams_r()
            
            # Commodity Channel Index
            df['cci'] = ta.trend.CCIIndicator(
                high=df['high'], low=df['low'], close=df['close']).cci()
            
            # Average Directional Index
            df['adx'] = ta.trend.ADXIndicator(
                high=df['high'], low=df['low'], close=df['close']).adx()
            
            # Parabolic SAR
            df['psar'] = ta.trend.PSARIndicator(
                high=df['high'], low=df['low'], close=df['close']).psar()
            
            # Support and Resistance levels (simplified)
            df['support'] = df['low'].rolling(window=20).min()
            df['resistance'] = df['high'].rolling(window=20).max()
            
            # Market structure
            df['higher_highs'] = (df['high'] > df['high'].shift(1)).astype(int)
            df['lower_lows'] = (df['low'] < df['low'].shift(1)).astype(int)
            
        except Exception as e:
            logger.error(f"Error adding custom indicators: {e}")
        
        return df
    
    @staticmethod
    def resample_data(df: pd.DataFrame, interval: str) -> pd.DataFrame:
        """Resample data to different time intervals"""
        if df.empty or 'date' not in df.columns:
            return df
        
        try:
            df = df.set_index('date')
            
            # Define aggregation rules
            agg_rules = {
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            }
            
            # Add other numeric columns with mean aggregation
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if col not in agg_rules:
                    agg_rules[col] = 'mean'
            
            # Add non-numeric columns with first value
            non_numeric_cols = df.select_dtypes(exclude=[np.number]).columns
            for col in non_numeric_cols:
                if col not in agg_rules:
                    agg_rules[col] = 'first'
            
            # Resample
            df_resampled = df.resample(interval).agg(agg_rules)
            df_resampled = df_resampled.dropna()
            df_resampled.reset_index(inplace=True)
            
            logger.info(f"Resampled data to {interval}: {len(df)} -> {len(df_resampled)} rows")
            return df_resampled
            
        except Exception as e:
            logger.error(f"Error resampling data: {e}")
            return df


class DatabaseManager:
    """Database operations for storing and retrieving market data"""
    
    def __init__(self, connection_string: str):
        self.connection_string = connection_string
        self.engine = create_engine(connection_string)
        self.metadata = MetaData()
        self.Session = sessionmaker(bind=self.engine)
    
    def create_tables(self):
        """Create database tables if they don't exist"""
        try:
            # Create market_data table
            market_data_table = Table(
                'market_data', self.metadata,
                Column('id', Integer, primary_key=True),
                Column('date', DateTime),
                Column('symbol', String(20)),
                Column('source', String(20)),
                Column('interval', String(10)),
                Column('exchange', String(20), nullable=True),
                Column('open', Float),
                Column('high', Float),
                Column('low', Float),
                Column('close', Float),
                Column('volume', Float),
                Column('data_json', Text),  # Store additional data as JSON
                Column('created_at', DateTime, default=datetime.utcnow)
            )
            
            self.metadata.create_all(self.engine)
            logger.info("Database tables created/verified")
            
        except Exception as e:
            logger.error(f"Error creating database tables: {e}")
    
    def save_dataframe(self, df: pd.DataFrame, table_name: str, if_exists: str = 'replace') -> bool:
        """Save DataFrame to database"""
        try:
            df.to_sql(table_name, self.engine, if_exists=if_exists, index=False)
            logger.info(f"Saved {len(df)} rows to table '{table_name}'")
            return True
            
        except Exception as e:
            logger.error(f"Error saving DataFrame to database: {e}")
            return False
    
    def load_dataframe(self, table_name: str, symbol: str = None, 
                      start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """Load DataFrame from database with optional filtering"""
        try:
            query = f"SELECT * FROM {table_name}"
            conditions = []
            
            if symbol:
                conditions.append(f"symbol = '{symbol}'")
            if start_date:
                conditions.append(f"date >= '{start_date}'")
            if end_date:
                conditions.append(f"date <= '{end_date}'")
            
            if conditions:
                query += " WHERE " + " AND ".join(conditions)
            
            query += " ORDER BY date"
            
            df = pd.read_sql(query, self.engine)
            logger.info(f"Loaded {len(df)} rows from table '{table_name}'")
            return df
            
        except Exception as e:
            logger.error(f"Error loading DataFrame from database: {e}")
            return pd.DataFrame()
    
    def get_table_info(self, table_name: str) -> Dict:
        """Get information about a table"""
        try:
            inspector = inspect(self.engine)
            columns = inspector.get_columns(table_name)
            row_count = pd.read_sql(f"SELECT COUNT(*) as count FROM {table_name}", self.engine).iloc[0]['count']
            
            return {
                'columns': [col['name'] for col in columns],
                'row_count': row_count,
                'column_details': columns
            }
            
        except Exception as e:
            logger.error(f"Error getting table info: {e}")
            return {}


class DataPipeline:
    """Main data pipeline orchestrator"""
    
    def __init__(self, config: Optional[Dict] = None):
        self.settings = get_settings()
        self.config = config or {}
        
        # Initialize data sources
        self.yfinance = YFinanceSource()
        
        # Alpha Vantage (if API key provided)
        alpha_key = self.config.get('alpha_vantage_api_key') or getattr(self.settings, 'alpha_vantage_api_key', None)
        self.alpha_vantage = AlphaVantageSource(alpha_key) if alpha_key else None
        
        # CCXT
        self.ccxt = CCXTSource() if CCXT_AVAILABLE else None
        
        # Database
        db_path = self.config.get('database_path', 'data/market_data.db')
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        
        self.db = DatabaseManager(f"sqlite:///{db_path}")
        self.db.create_tables()
        
        # Components
        self.validator = DataValidator()
        self.feature_eng = FeatureEngineering()
        
        logger.info("DataPipeline initialized successfully")
    
    async def fetch_historical(self, symbol: str, start: str = None, end: str = None, 
                             interval: str = "1d", source: str = "yfinance") -> pd.DataFrame:
        """Fetch historical data from specified source"""
        try:
            df = pd.DataFrame()
            
            if source == "yfinance":
                df = await self.yfinance.fetch_historical(symbol, interval=interval, start=start, end=end)
            
            elif source == "alpha_vantage" and self.alpha_vantage:
                if interval == "1d":
                    df = await self.alpha_vantage.fetch_daily(symbol)
                else:
                    logger.warning("Alpha Vantage only supports daily data in free tier")
            
            elif source == "ccxt" and self.ccxt:
                # For crypto symbols, try to fetch from CCXT
                parts = symbol.split('/')
                if len(parts) == 2:
                    for exchange in ['binance', 'kraken', 'coinbase']:
                        df = await self.ccxt.fetch_ohlcv(exchange, symbol, interval)
                        if not df.empty:
                            break
            
            if df.empty:
                logger.warning(f"No data fetched for {symbol} from {source}")
            
            return df
            
        except Exception as e:
            logger.error(f"Error in fetch_historical: {e}")
            return pd.DataFrame()
    
    async def fetch_live(self, symbol: str, source: str = "yfinance") -> Dict:
        """Fetch current/live market data"""
        try:
            if source == "yfinance":
                info = await self.yfinance.fetch_info(symbol)
                return {
                    'symbol': symbol,
                    'current_price': info.get('currentPrice'),
                    'bid': info.get('bid'),
                    'ask': info.get('ask'),
                    'volume': info.get('volume'),
                    'market_cap': info.get('marketCap'),
                    'timestamp': datetime.utcnow()
                }
            
            # Add other sources as needed
            return {}
            
        except Exception as e:
            logger.error(f"Error fetching live data: {e}")
            return {}
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and validate data"""
        if df.empty:
            return df
        
        logger.info(f"Cleaning data: {len(df)} rows")
        
        # Apply validation and cleaning
        df = self.validator.validate_ohlcv(df)
        df = self.validator.fill_missing_values(df)
        df = self.validator.align_timezones(df)
        
        logger.info(f"Data cleaned: {len(df)} rows remaining")
        return df
    
    def transform_features(self, df: pd.DataFrame, add_indicators: bool = True) -> pd.DataFrame:
        """Transform data and add features"""
        if df.empty:
            return df
        
        logger.info(f"Transforming features for {len(df)} rows")
        
        # Add basic features
        df = self.feature_eng.add_basic_features(df)
        
        # Add technical indicators
        if add_indicators:
            df = self.feature_eng.add_technical_indicators(df)
        
        logger.info(f"Features added: {len(df.columns)} columns")
        return df
    
    def resample(self, df: pd.DataFrame, interval: str) -> pd.DataFrame:
        """Resample data to different time intervals"""
        return self.feature_eng.resample_data(df, interval)
    
    def save(self, df: pd.DataFrame, table_name: str, if_exists: str = 'replace') -> bool:
        """Save DataFrame to database"""
        if df.empty:
            logger.warning("Cannot save empty DataFrame")
            return False
        
        return self.db.save_dataframe(df, table_name, if_exists)
    
    def load(self, table_name: str, symbol: str = None, 
            start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """Load DataFrame from database"""
        return self.db.load_dataframe(table_name, symbol, start_date, end_date)
    
    def get_table_info(self, table_name: str) -> Dict:
        """Get information about stored data"""
        return self.db.get_table_info(table_name)
    
    async def run_full_pipeline(self, symbol: str, start: str = None, end: str = None,
                              intervals: List[str] = None) -> Dict[str, pd.DataFrame]:
        """Run complete pipeline for a symbol"""
        intervals = intervals or ["1d"]
        results = {}
        
        try:
            logger.info(f"Running full pipeline for {symbol}")
            
            # Fetch historical data
            df_raw = await self.fetch_historical(symbol, start, end, "1d")
            
            if df_raw.empty:
                logger.warning(f"No data available for {symbol}")
                return results
            
            # Clean data
            df_clean = self.clean_data(df_raw.copy())
            
            # Process each interval
            for interval in intervals:
                logger.info(f"Processing {symbol} for {interval} interval")
                
                # Resample if needed
                if interval != "1d":
                    df_interval = self.resample(df_clean.copy(), interval)
                else:
                    df_interval = df_clean.copy()
                
                if df_interval.empty:
                    continue
                
                # Add features
                df_features = self.transform_features(df_interval)
                
                # Save to database
                table_name = f"{symbol.replace('/', '_')}_{interval}"
                if self.save(df_features, table_name):
                    results[interval] = df_features
            
            logger.info(f"Pipeline completed for {symbol}: {len(results)} intervals processed")
            
        except Exception as e:
            logger.error(f"Error in full pipeline: {e}")
        
        return results
    
    async def update_multiple_symbols(self, symbols: List[str], **kwargs) -> Dict[str, Dict]:
        """Update data for multiple symbols"""
        results = {}
        
        for symbol in symbols:
            try:
                logger.info(f"Updating data for {symbol}")
                symbol_results = await self.run_full_pipeline(symbol, **kwargs)
                results[symbol] = symbol_results
                
                # Rate limiting
                await asyncio.sleep(1)
                
            except Exception as e:
                logger.error(f"Error updating {symbol}: {e}")
                results[symbol] = {}
        
        return results
    
    def get_ml_ready_data(self, table_name: str, target_col: str = 'close',
                         lookback_window: int = 60) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare data for ML training"""
        try:
            df = self.load(table_name)
            
            if df.empty:
                return np.array([]), np.array([])
            
            # Select feature columns (numeric only, exclude target and metadata)
            feature_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            exclude_cols = ['date', 'symbol', 'source', 'interval', 'exchange', target_col]
            feature_cols = [col for col in feature_cols if col not in exclude_cols]
            
            # Prepare features and targets
            X = df[feature_cols].values
            y = df[target_col].values
            
            # Create sequences for time series prediction
            X_seq, y_seq = [], []
            for i in range(lookback_window, len(X)):
                X_seq.append(X[i-lookback_window:i])
                y_seq.append(y[i])
            
            return np.array(X_seq), np.array(y_seq)
            
        except Exception as e:
            logger.error(f"Error preparing ML data: {e}")
            return np.array([]), np.array([])


# Example usage and main function
async def main():
    """Example usage of the data pipeline"""
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Initialize pipeline
    dp = DataPipeline({
        'database_path': 'data/market_data.db',
        # 'alpha_vantage_api_key': 'your_api_key_here'  # Optional
    })
    
    # Example 1: Single symbol pipeline
    print("=== Single Symbol Pipeline ===")
    symbol = "AAPL"
    results = await dp.run_full_pipeline(
        symbol=symbol,
        start="2020-01-01",
        end="2023-01-01",
        intervals=["1d", "1h"]
    )
    
    for interval, df in results.items():
        print(f"{symbol} {interval}: {len(df)} rows, {len(df.columns)} features")
    
    # Example 2: Multiple symbols
    print("\n=== Multiple Symbols Pipeline ===")
    symbols = ["AAPL", "GOOGL", "MSFT", "BTC-USD"]
    multi_results = await dp.update_multiple_symbols(
        symbols=symbols,
        start="2022-01-01",
        end="2023-01-01"
    )
    
    for symbol, intervals in multi_results.items():
        print(f"{symbol}: {len(intervals)} intervals processed")
    
    # Example 3: Load and inspect data
    print("\n=== Data Inspection ===")
    table_info = dp.get_table_info("AAPL_1d")
    print(f"AAPL table: {table_info.get('row_count', 0)} rows")
    
    # Example 4: ML-ready data
    print("\n=== ML Data Preparation ===")
    X, y = dp.get_ml_ready_data("AAPL_1d", target_col="close")
    print(f"ML data prepared: X shape {X.shape}, y shape {y.shape}")
    
    # Example 5: Live data fetch
    print("\n=== Live Data Fetch ===")
    live_data = await dp.fetch_live("AAPL")
    print(f"Live AAPL data: ${live_data.get('current_price', 'N/A')}")


if __name__ == "__main__":
    asyncio.run(main())
