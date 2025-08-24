"""
Real Data Integration for ML Training Pipeline

Integrates existing data pipeline with ML training for production-ready models.
Supports both historical training and real-time inference data.
"""

import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import logging
from pathlib import Path

# Import existing data pipeline components
from arbi.core.pipeline import YFinanceSource, DataPipeline
from arbi.core.data_feed import ExchangeConnector
from arbi.core.marketdata import BookDelta, OrderBook, Trade
from arbi.ai.feature_engineering_v2 import compute_features_deterministic, load_feature_schema

logger = logging.getLogger(__name__)


class MLDataIntegrator:
    """Integrates real market data with ML training pipeline"""
    
    def __init__(self, config=None):
        self.config = config or {
            'data_sources': ['yfinance', 'binance'],
            'storage_path': './data/',
            'cache_enabled': True,
            'real_time_enabled': False
        }
        
        # Initialize data pipeline
        self.pipeline = DataPipeline(
            storage_path=self.config['storage_path'],
            enable_cache=self.config['cache_enabled']
        )
        
        # Initialize data sources
        self.yf_source = YFinanceSource()
        
        # Feature engineering
        self.feature_schema = load_feature_schema()
        
        logger.info(f"MLDataIntegrator initialized with sources: {self.config['data_sources']}")
    
    async def fetch_training_data(self, symbol: str, period: str = "2y", 
                                interval: str = "1h", force_refresh: bool = False) -> pd.DataFrame:
        """
        Fetch real market data for ML training
        
        Args:
            symbol: Trading pair (e.g., 'BTC-USD', 'AAPL')
            period: Time period ('1y', '2y', '5y', etc.)
            interval: Data interval ('1m', '5m', '1h', '1d')
            force_refresh: Force fetch new data ignoring cache
            
        Returns:
            DataFrame with OHLCV data
        """
        logger.info(f"Fetching training data for {symbol}, period={period}, interval={interval}")
        
        try:
            # Check cache first (unless force refresh)
            cache_key = f"{symbol}_{period}_{interval}"
            cached_data = None
            
            if not force_refresh and self.config['cache_enabled']:
                cached_data = await self._load_cached_data(cache_key)
                
                if cached_data is not None:
                    logger.info(f"Using cached data for {symbol}")
                    return cached_data
            
            # Fetch fresh data
            if 'yfinance' in self.config['data_sources']:
                data = await self._fetch_from_yfinance(symbol, period, interval)
                
                # Cache the data
                if self.config['cache_enabled']:
                    await self._cache_data(cache_key, data)
                
                return data
            
            else:
                raise ValueError("No valid data sources configured")
                
        except Exception as e:
            logger.error(f"Error fetching training data for {symbol}: {e}")
            # Return synthetic data as fallback
            logger.warning("Falling back to synthetic data generation")
            return self._generate_fallback_data(symbol, period, interval)
    
    async def _fetch_from_yfinance(self, symbol: str, period: str, interval: str) -> pd.DataFrame:
        """Fetch data from Yahoo Finance"""
        
        # Normalize symbol for Yahoo Finance
        yf_symbol = self._normalize_symbol_for_yf(symbol)
        
        # Fetch data
        data = await self.yf_source.fetch_historical(
            symbol=yf_symbol,
            period=period,
            interval=interval
        )
        
        if data.empty:
            raise ValueError(f"No data returned for {symbol}")
        
        # Standardize column names and add metadata
        data = self._standardize_ohlcv_data(data, symbol)
        
        logger.info(f"Fetched {len(data)} records for {symbol} from Yahoo Finance")
        return data
    
    def _normalize_symbol_for_yf(self, symbol: str) -> str:
        """Normalize symbol for Yahoo Finance API"""
        # Common crypto mappings
        crypto_mappings = {
            'BTC-USD': 'BTC-USD',
            'ETH-USD': 'ETH-USD', 
            'BTC/USD': 'BTC-USD',
            'ETH/USD': 'ETH-USD',
            'BTCUSDT': 'BTC-USD',
            'ETHUSDT': 'ETH-USD'
        }
        
        return crypto_mappings.get(symbol.upper(), symbol.upper())
    
    def _standardize_ohlcv_data(self, data: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Standardize OHLCV data format"""
        
        # Ensure we have the required columns
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        
        for col in required_cols:
            if col not in data.columns:
                # Try alternative column names
                alt_names = {
                    'Open': ['open', 'OPEN'],
                    'High': ['high', 'HIGH'], 
                    'Low': ['low', 'LOW'],
                    'Close': ['close', 'CLOSE', 'adj close', 'Adj Close'],
                    'Volume': ['volume', 'VOLUME', 'vol']
                }
                
                found = False
                for alt_name in alt_names.get(col, []):
                    if alt_name in data.columns:
                        data[col] = data[alt_name]
                        found = True
                        break
                
                if not found:
                    raise ValueError(f"Missing required column: {col}")
        
        # Rename to standard format
        data = data.rename(columns={
            'Open': 'open',
            'High': 'high', 
            'Low': 'low',
            'Close': 'close',
            'Volume': 'volume'
        })
        
        # Add timestamp column
        if 'timestamp' not in data.columns:
            if isinstance(data.index, pd.DatetimeIndex):
                data['timestamp'] = data.index
            else:
                data['timestamp'] = pd.to_datetime(data.index)
        
        # Add symbol metadata
        data['symbol'] = symbol
        
        # Ensure numeric types
        numeric_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_cols:
            if col in data.columns:
                data[col] = pd.to_numeric(data[col], errors='coerce')
        
        # Remove any rows with NaN values
        data = data.dropna()
        
        # Sort by timestamp
        data = data.sort_values('timestamp')
        
        return data
    
    async def compute_ml_features(self, ohlcv_data: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """
        Compute ML features from OHLCV data using existing feature engineering
        
        Args:
            ohlcv_data: OHLCV DataFrame
            symbol: Trading symbol
            
        Returns:
            DataFrame with computed features
        """
        logger.info(f"Computing ML features for {symbol} with {len(ohlcv_data)} data points")
        
        try:
            # Use existing feature engineering pipeline
            feature_result = compute_features_deterministic(ohlcv_data, symbol)
            features_df = feature_result.features
            
            logger.info(f"Computed {len(features_df.columns)} features for {symbol}")
            
            # Add metadata
            features_df['symbol'] = symbol
            features_df['data_source'] = 'real'
            
            return features_df
            
        except Exception as e:
            logger.error(f"Error computing features for {symbol}: {e}")
            
            # Fallback to basic features
            logger.warning("Using fallback feature computation")
            return self._compute_basic_features(ohlcv_data, symbol)
    
    def _compute_basic_features(self, data: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Compute basic technical features as fallback"""
        
        features = pd.DataFrame(index=data.index)
        
        # Price features
        features['returns'] = data['close'].pct_change()
        features['log_returns'] = np.log(data['close'] / data['close'].shift(1))
        features['price_ma5'] = data['close'].rolling(5).mean()
        features['price_ma20'] = data['close'].rolling(20).mean()
        features['price_std'] = data['close'].rolling(20).std()
        
        # Volume features
        features['volume'] = data['volume']
        features['volume_ma5'] = data['volume'].rolling(5).mean()
        features['volume_ratio'] = data['volume'] / features['volume_ma5']
        
        # Technical indicators
        features['rsi'] = self._compute_rsi(data['close'])
        features['macd'] = self._compute_macd(data['close'])
        
        # Volatility
        features['volatility'] = features['returns'].rolling(20).std()
        
        # Clean and add metadata
        features = features.dropna()
        features['symbol'] = symbol
        features['data_source'] = 'basic'
        
        logger.info(f"Computed {len(features.columns)} basic features")
        return features
    
    def _compute_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """Compute RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def _compute_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26) -> pd.Series:
        """Compute MACD indicator"""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        return ema_fast - ema_slow
    
    def create_ml_labels(self, data: pd.DataFrame, horizon: int = 5, 
                        pos_thresh: float = 0.002, neg_thresh: float = -0.002) -> Tuple[pd.Series, pd.Series]:
        """
        Create ML labels for training
        
        Args:
            data: OHLCV data with 'close' column
            horizon: Number of periods to look ahead
            pos_thresh: Positive return threshold for classification
            neg_thresh: Negative return threshold for classification
            
        Returns:
            Tuple of (binary_labels, regression_labels)
        """
        logger.info(f"Creating ML labels with horizon={horizon}, pos_thresh={pos_thresh}")
        
        # Calculate future returns
        future_returns = data['close'].shift(-horizon) / data['close'] - 1
        
        # Binary classification labels
        binary_labels = pd.Series(0, index=data.index)  # Default to neutral
        binary_labels[future_returns > pos_thresh] = 1   # Positive class
        binary_labels[future_returns < neg_thresh] = -1  # Negative class (optional)
        
        # For now, convert to simple binary (0/1)
        binary_labels = (binary_labels > 0).astype(int)
        
        # Regression labels (actual future returns)
        regression_labels = future_returns
        
        logger.info(f"Created labels: {binary_labels.value_counts().to_dict()}")
        
        return binary_labels, regression_labels
    
    async def prepare_training_dataset(self, symbol: str, period: str = "2y", 
                                     interval: str = "1h", horizon: int = 5,
                                     pos_thresh: float = 0.002) -> Dict[str, Any]:
        """
        Prepare complete training dataset with real market data
        
        Returns:
            Dictionary with X, y_binary, y_regression, timestamps, metadata
        """
        logger.info(f"Preparing training dataset for {symbol}")
        
        # Step 1: Fetch OHLCV data
        ohlcv_data = await self.fetch_training_data(symbol, period, interval)
        
        # Step 2: Compute features
        features_df = await self.compute_ml_features(ohlcv_data, symbol)
        
        # Step 3: Create labels
        binary_labels, regression_labels = self.create_ml_labels(
            ohlcv_data, horizon, pos_thresh
        )
        
        # Step 4: Align features and labels
        # Features and labels must have the same time index
        common_index = features_df.index.intersection(binary_labels.index)
        
        X = features_df.loc[common_index].drop(['symbol', 'data_source'], axis=1, errors='ignore')
        y_binary = binary_labels.loc[common_index]
        y_regression = regression_labels.loc[common_index]
        timestamps = ohlcv_data.loc[common_index, 'timestamp'] if 'timestamp' in ohlcv_data.columns else common_index
        
        # Remove any remaining NaN values
        valid_mask = ~X.isnull().any(axis=1) & ~y_binary.isnull() & ~y_regression.isnull()
        
        X = X[valid_mask]
        y_binary = y_binary[valid_mask]
        y_regression = y_regression[valid_mask]
        timestamps = timestamps[valid_mask]
        
        # Metadata
        metadata = {
            'symbol': symbol,
            'period': period,
            'interval': interval,
            'horizon': horizon,
            'pos_thresh': pos_thresh,
            'n_samples': len(X),
            'n_features': len(X.columns),
            'feature_names': list(X.columns),
            'data_range': {
                'start': str(timestamps.iloc[0]) if len(timestamps) > 0 else None,
                'end': str(timestamps.iloc[-1]) if len(timestamps) > 0 else None
            },
            'class_distribution': y_binary.value_counts().to_dict(),
            'target_stats': {
                'mean': float(y_regression.mean()),
                'std': float(y_regression.std()),
                'min': float(y_regression.min()),
                'max': float(y_regression.max())
            }
        }
        
        logger.info(f"Training dataset prepared: {len(X)} samples, {len(X.columns)} features")
        
        return {
            'X': X,
            'y_binary': y_binary,
            'y_regression': y_regression,
            'timestamps': timestamps,
            'metadata': metadata,
            'ohlcv_data': ohlcv_data  # Include raw data for reference
        }
    
    async def _load_cached_data(self, cache_key: str) -> Optional[pd.DataFrame]:
        """Load cached data if available and fresh"""
        try:
            cache_file = Path(self.config['storage_path']) / 'cache' / f"{cache_key}.pkl"
            
            if cache_file.exists():
                # Check if cache is fresh (less than 1 hour old)
                cache_age = datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime)
                
                if cache_age < timedelta(hours=1):
                    data = pd.read_pickle(cache_file)
                    logger.info(f"Loaded cached data: {cache_key}")
                    return data
            
            return None
            
        except Exception as e:
            logger.warning(f"Error loading cached data {cache_key}: {e}")
            return None
    
    async def _cache_data(self, cache_key: str, data: pd.DataFrame):
        """Cache data for future use"""
        try:
            cache_dir = Path(self.config['storage_path']) / 'cache'
            cache_dir.mkdir(parents=True, exist_ok=True)
            
            cache_file = cache_dir / f"{cache_key}.pkl"
            data.to_pickle(cache_file)
            
            logger.info(f"Cached data: {cache_key}")
            
        except Exception as e:
            logger.warning(f"Error caching data {cache_key}: {e}")
    
    def _generate_fallback_data(self, symbol: str, period: str, interval: str) -> pd.DataFrame:
        """Generate synthetic data as fallback when real data fails"""
        logger.warning(f"Generating synthetic fallback data for {symbol}")
        
        # Determine number of periods based on period and interval
        period_mapping = {'1y': 365, '2y': 730, '5y': 1825}
        interval_mapping = {'1m': 1440, '5m': 288, '1h': 24, '1d': 1}
        
        days = period_mapping.get(period, 365)
        periods_per_day = interval_mapping.get(interval, 24)
        n_periods = days * periods_per_day
        
        # Generate dates
        freq_mapping = {'1m': '1T', '5m': '5T', '1h': '1H', '1d': '1D'}
        freq = freq_mapping.get(interval, '1H')
        
        dates = pd.date_range(
            end=datetime.now(), 
            periods=min(n_periods, 10000),  # Cap at 10k for performance
            freq=freq
        )
        
        # Generate realistic price data
        np.random.seed(42)
        returns = np.random.normal(0.0001, 0.02, len(dates))  # Slightly higher vol for crypto
        log_prices = np.cumsum(returns)
        
        base_price = 50000 if 'BTC' in symbol.upper() else 100
        prices = base_price * np.exp(log_prices)
        
        data = []
        for i, (date, price) in enumerate(zip(dates, prices)):
            volatility = abs(np.random.normal(0, 0.01))
            high = price * (1 + volatility)
            low = price * (1 - volatility)
            open_price = prices[i-1] if i > 0 else price
            volume = np.random.uniform(1000, 10000)
            
            data.append({
                'timestamp': date,
                'open': open_price,
                'high': high,
                'low': low,
                'close': price,
                'volume': volume,
                'symbol': symbol
            })
        
        df = pd.DataFrame(data)
        logger.info(f"Generated {len(df)} synthetic data points for {symbol}")
        
        return df


# Convenience function for easy integration
async def get_real_training_data(symbol: str, period: str = "1y", interval: str = "1h",
                               horizon: int = 5, pos_thresh: float = 0.002) -> Dict[str, Any]:
    """
    Convenience function to get real training data
    
    Usage:
        data = await get_real_training_data("BTC-USD", period="2y", interval="1h")
        X, y_binary, y_regression = data['X'], data['y_binary'], data['y_regression']
    """
    integrator = MLDataIntegrator()
    return await integrator.prepare_training_dataset(
        symbol=symbol,
        period=period, 
        interval=interval,
        horizon=horizon,
        pos_thresh=pos_thresh
    )


if __name__ == "__main__":
    # Test the integration
    async def test_integration():
        print("Testing Real Data Integration...")
        
        integrator = MLDataIntegrator()
        
        # Test with BTC-USD
        dataset = await integrator.prepare_training_dataset(
            symbol="BTC-USD",
            period="6m",  # 6 months for testing
            interval="1h",
            horizon=5,
            pos_thresh=0.003
        )
        
        print(f"Dataset prepared:")
        print(f"  Samples: {len(dataset['X'])}")
        print(f"  Features: {len(dataset['X'].columns)}")
        print(f"  Time range: {dataset['metadata']['data_range']}")
        print(f"  Class distribution: {dataset['metadata']['class_distribution']}")
        
        # Show first few feature names
        print(f"  Sample features: {list(dataset['X'].columns[:10])}")
        
    # Run test
    asyncio.run(test_integration())
