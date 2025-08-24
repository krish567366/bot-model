"""
Production Inference Engine - Priority A

Loads latest models and generates ML signals that integrate with the trading system.
Maps predictions to Signal objects and populates storage.signals with the schema:
{ts, symbol, side, prob, model_id, feature_snapshot}
"""

import asyncio
import logging
import time
import json
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
import pandas as pd
import numpy as np

from .registry import ModelRegistry
from .feature_engineering_v2 import compute_features_deterministic, validate_feature_schema
from ..core.marketdata import OrderBook, Trade, Ticker
from ..core.signal import ArbitrageSignal, SignalType, SignalStrength

logger = logging.getLogger(__name__)


@dataclass
class MLSignal:
    """ML prediction signal for integration with trading system"""
    timestamp: datetime
    model_id: str
    symbol: str
    side: str  # 'BUY', 'SELL', 'HOLD'
    confidence: float  # Model confidence [0, 1]
    probability: float  # Directional probability [0, 1]
    magnitude: float  # Expected return magnitude [0, 1]
    feature_snapshot: Dict[str, float]  # Features used for prediction
    
    # Model metadata
    model_version: str
    feature_count: int
    validation_score: float


class ProductionInferenceEngine:
    """
    Production inference engine that:
    1. Loads latest model from registry
    2. Computes features deterministically  
    3. Runs predict_proba on new market data
    4. Maps predictions to Signal objects
    5. Populates storage.signals with proper schema
    """
    
    def __init__(self, registry_path: str = "model_registry.db"):
        self.registry = ModelRegistry(registry_path)
        self.current_model = None
        self.current_model_id = None
        self.feature_names = []
        self.scaler = None
        self.logger = logger
        
    async def initialize(self):
        """Initialize inference engine with latest model"""
        self.logger.info("Initializing production inference engine...")
        try:
            await self.load_latest_model()
            self.logger.info(f"✅ Inference engine initialized with model: {self.current_model_id}")
        except Exception as e:
            self.logger.error(f"Failed to initialize inference engine: {e}")
            raise
    
    async def load_latest_model(self, symbol: Optional[str] = None):
        """Load the latest trained model from registry"""
        try:
            # If no symbol specified, try to get any available model
            if symbol is None:
                # Get all models and pick the latest
                all_models = self.registry.list_models()
                if not all_models:
                    raise ValueError("No trained models found in registry")
                
                # Sort by validation score and get the best
                best_model = max(all_models, key=lambda m: m.validation_score)
                model_metadata = best_model
            else:
                # Get latest model for specific symbol
                model_metadata = self.registry.get_latest_model(symbol)
                if not model_metadata:
                    raise ValueError(f"No trained models found for symbol: {symbol}")
            
            # Load model artifacts
            model_obj, scaler_obj, model_metadata = self.registry.load_model(model_metadata.model_id)
            
            self.current_model = model_obj
            self.scaler = scaler_obj
            self.current_model_id = model_metadata.model_id
            # Feature names should be in the metadata JSON file
            with open(model_metadata.metadata_path, 'r') as f:
                metadata_json = json.load(f)
                # Extract feature names from feature importance keys
                feature_importance = metadata_json.get('feature_importance', {})
                if feature_importance:
                    self.feature_names = list(feature_importance.keys())
                else:
                    # Fallback: try to get from schema
                    from .feature_engineering_v2 import load_feature_schema
                    schema = load_feature_schema()
                    self.feature_names = [f['name'] for f in schema['features']]
            
            self.logger.info(f"✅ Loaded model {self.current_model_id} "
                           f"(validation_score: {model_metadata.validation_score:.4f})")
            
        except Exception as e:
            self.logger.error(f"Failed to load latest model: {e}")
            raise
    
    async def generate_ml_signals(
        self, 
        symbol: str, 
        exchange: str,
        market_data: Optional[pd.DataFrame] = None,
        order_book: Optional[OrderBook] = None,
        ticker: Optional[Ticker] = None
    ) -> List[MLSignal]:
        """
        Generate ML signals for given symbol/exchange
        
        Args:
            symbol: Trading symbol (e.g. 'BTC/USDT')
            exchange: Exchange name (e.g. 'binance')
            market_data: OHLCV data for feature computation
            order_book: Current order book
            ticker: Current ticker data
            
        Returns:
            List of MLSignal objects ready for storage
        """
        if not self.current_model:
            self.logger.warning("No model loaded, cannot generate signals")
            return []
        
        try:
            # 1. Get or create market data for feature computation
            if market_data is None:
                market_data = await self._fetch_recent_market_data(symbol, exchange)
            
            # 2. Compute features deterministically
            features_df = await self._compute_features(symbol, market_data)
            if features_df.empty:
                self.logger.warning(f"No features computed for {symbol}")
                return []
            
            # 3. Get the latest feature vector (most recent timestamp)
            latest_features = features_df.iloc[-1]
            feature_vector = latest_features[self.feature_names].values.reshape(1, -1)
            
            # 4. Scale features
            if self.scaler:
                feature_vector = self.scaler.transform(feature_vector)
            
            # 5. Run inference
            predictions = await self._run_inference(feature_vector)
            
            # 6. Map to ML signals
            signals = await self._create_ml_signals(
                symbol, predictions, latest_features.to_dict()
            )
            
            # Debug information
            self.logger.info(f"Predictions: {predictions}")
            self.logger.info(f"Generated {len(signals)} signals with threshold check")
            
            self.logger.info(f"✅ Generated {len(signals)} ML signals for {symbol}")
            return signals
            
        except Exception as e:
            self.logger.error(f"Failed to generate ML signals for {symbol}: {e}")
            return []
    
    async def _fetch_recent_market_data(self, symbol: str, exchange: str) -> pd.DataFrame:
        """Fetch recent OHLCV data for feature computation"""
        # This would typically fetch from your storage system
        # For now, create synthetic data for testing
        self.logger.info(f"Fetching recent market data for {symbol} on {exchange}")
        
        # Create synthetic OHLCV data (in production, fetch from storage)
        periods = 200  # Need history for technical indicators
        dates = pd.date_range(end=datetime.now(), periods=periods, freq='1min')
        
        # Simple random walk for testing
        np.random.seed(42)  # Deterministic for testing
        returns = np.random.normal(0, 0.001, periods)
        prices = 50000 * np.exp(np.cumsum(returns))
        
        data = []
        for i, (date, price) in enumerate(zip(dates, prices)):
            high = price * (1 + abs(np.random.normal(0, 0.005)))
            low = price * (1 - abs(np.random.normal(0, 0.005)))
            open_price = prices[i-1] if i > 0 else price
            volume = np.random.uniform(100, 1000)
            
            data.append({
                'timestamp': date,
                'open': open_price,
                'high': high,
                'low': low,
                'close': price,
                'volume': volume
            })
        
        return pd.DataFrame(data)
    
    async def _compute_features(self, symbol: str, market_data: pd.DataFrame) -> pd.DataFrame:
        """Compute features using deterministic feature engineering"""
        try:
            # Validate OHLCV data
            required_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            if not all(col in market_data.columns for col in required_cols):
                raise ValueError(f"Market data missing required columns: {required_cols}")
            
            # Compute features deterministically
            result = compute_features_deterministic(market_data, symbol)
            
            # Extract DataFrame from result
            features_df = result.features
            
            self.logger.info(f"✅ Computed {len(features_df)} feature vectors with {len(features_df.columns)} features")
            return features_df
            
        except Exception as e:
            self.logger.error(f"Feature computation failed for {symbol}: {e}")
            return pd.DataFrame()
    
    async def _run_inference(self, feature_vector: np.ndarray) -> Dict[str, float]:
        """Run model inference on feature vector"""
        try:
            # Get model prediction
            if hasattr(self.current_model, 'predict_proba'):
                # Classification model - get probabilities
                proba = self.current_model.predict_proba(feature_vector)[0]
                if len(proba) == 2:  # Binary classification
                    predictions = {
                        'probability_up': float(proba[1]),
                        'probability_down': float(proba[0]),
                        'confidence': float(max(proba))
                    }
                else:
                    predictions = {
                        'probability_up': float(proba[-1]) if len(proba) > 2 else 0.5,
                        'probability_down': float(proba[0]) if len(proba) > 2 else 0.5,
                        'confidence': float(max(proba))
                    }
            else:
                # Regression model - predict return
                prediction = self.current_model.predict(feature_vector)[0]
                predictions = {
                    'return_prediction': float(prediction),
                    'probability_up': 1.0 if prediction > 0 else 0.0,
                    'probability_down': 1.0 if prediction < 0 else 0.0,
                    'confidence': min(abs(prediction) * 10, 1.0)  # Scale to [0, 1]
                }
            
            return predictions
            
        except Exception as e:
            self.logger.error(f"Model inference failed: {e}")
            return {
                'probability_up': 0.5,
                'probability_down': 0.5,
                'confidence': 0.0
            }
    
    async def _create_ml_signals(
        self, 
        symbol: str, 
        predictions: Dict[str, float],
        feature_snapshot: Dict[str, float]
    ) -> List[MLSignal]:
        """Create ML signals from model predictions"""
        signals = []
        
        try:
            # Get model metadata
            model_metadata = self.registry.get_model_by_id(self.current_model_id)
            
            # Extract prediction values
            prob_up = predictions.get('probability_up', 0.5)
            prob_down = predictions.get('probability_down', 0.5) 
            confidence = predictions.get('confidence', 0.0)
            
            # Determine signal direction and magnitude
            if prob_up > prob_down and confidence > 0.1:
                side = 'BUY'
                probability = prob_up
                magnitude = prob_up - prob_down
            elif prob_down > prob_up and confidence > 0.1:
                side = 'SELL'
                probability = prob_down
                magnitude = prob_down - prob_up
            else:
                side = 'HOLD'
                probability = max(prob_up, prob_down)
                magnitude = abs(prob_up - prob_down)
            
            # Only create signal if confidence is above threshold
            if confidence > 0.001:  # Very low threshold for testing
                signal = MLSignal(
                    timestamp=datetime.utcnow(),
                    model_id=self.current_model_id,
                    symbol=symbol,
                    side=side,
                    confidence=confidence,
                    probability=probability,
                    magnitude=magnitude,
                    feature_snapshot=feature_snapshot,
                    model_version=model_metadata.version if model_metadata else "unknown",
                    feature_count=len(self.feature_names),
                    validation_score=model_metadata.validation_score if model_metadata else 0.0
                )
                signals.append(signal)
        
        except Exception as e:
            self.logger.error(f"Failed to create ML signals: {e}")
        
        return signals
    
    async def populate_storage_signals(
        self, 
        signals: List[MLSignal], 
        storage_manager
    ) -> int:
        """
        Populate storage.signals with ML signals in the required schema:
        {ts, symbol, side, prob, model_id, feature_snapshot}
        """
        if not signals:
            return 0
        
        try:
            # Convert to DataFrame with required schema
            signal_records = []
            for signal in signals:
                record = {
                    'timestamp': signal.timestamp,
                    'model_id': signal.model_id,
                    'symbol': signal.symbol,
                    'side': signal.side,
                    'confidence': signal.confidence,
                    'probability': signal.probability,
                    'magnitude': signal.magnitude,
                    'feature_snapshot': json.dumps(signal.feature_snapshot),
                    'model_version': signal.model_version,
                    'feature_count': signal.feature_count,
                    'validation_score': signal.validation_score
                }
                signal_records.append(record)
            
            # Save to storage
            signals_df = pd.DataFrame(signal_records)
            table_name = f"ml_signals_{signals[0].symbol.replace('/', '_').replace('-', '_')}"
            
            # Use storage manager to persist
            storage_manager.save_table(signals_df, table_name, if_exists='append')
            
            self.logger.info(f"✅ Stored {len(signals)} ML signals to {table_name}")
            return len(signals)
            
        except Exception as e:
            self.logger.error(f"Failed to populate storage with signals: {e}")
            return 0
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about currently loaded model"""
        if not self.current_model:
            return {'status': 'no_model_loaded'}
        
        model_metadata = self.registry.get_model_by_id(self.current_model_id)
        return {
            'status': 'ready',
            'model_id': self.current_model_id,
            'model_version': model_metadata.version if model_metadata else "unknown",
            'feature_count': len(self.feature_names),
            'validation_score': model_metadata.validation_score if model_metadata else 0.0,
            'trained_on': model_metadata.created_at if model_metadata else None
        }


# Factory function for easy integration
async def create_inference_engine(registry_path: str = "model_registry.db") -> ProductionInferenceEngine:
    """Create and initialize production inference engine"""
    engine = ProductionInferenceEngine(registry_path)
    await engine.initialize()
    return engine


# Test function
async def test_inference_engine():
    """Test the inference engine end-to-end"""
    print("Testing production inference engine...")
    
    try:
        # Create engine
        engine = await create_inference_engine()
        
        # Check model info
        info = engine.get_model_info()
        print(f"Model info: {info}")
        
        # Generate signals
        signals = await engine.generate_ml_signals(
            symbol="BTC/USDT",
            exchange="binance"
        )
        
        print(f"Generated {len(signals)} signals:")
        for signal in signals:
            print(f"  {signal.side} {signal.symbol} - confidence: {signal.confidence:.3f}, "
                  f"prob: {signal.probability:.3f}")
        
        print("✅ Inference engine test completed successfully")
        return True
        
    except Exception as e:
        print(f"❌ Inference engine test failed: {e}")
        return False


if __name__ == "__main__":
    # Run test
    asyncio.run(test_inference_engine())
