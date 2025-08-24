"""
Deterministic Feature Engineering Module

This module provides schema-locked feature engineering functions that ensure
identical transformations between training and inference. All features are 
computed deterministically with proper handling of missing values and edge cases.

Key Functions:
- compute_technical_features(): Main deterministic feature computation
- validate_feature_schema(): Ensures schema compliance
- load_feature_schema(): Loads feature definitions
"""

import json
import numpy as np
import pandas as pd
import talib as ta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class FeatureComputationResult:
    """Result of feature computation with metadata"""
    features: pd.DataFrame
    feature_count: int
    missing_count: int
    schema_version: str
    computation_metadata: Dict[str, Any]

class SchemaError(Exception):
    """Raised when feature schema validation fails"""
    pass

def load_feature_schema() -> Dict[str, Any]:
    """Load feature schema definition from JSON file"""
    schema_path = Path(__file__).parent / "feature_schema.json"
    
    if not schema_path.exists():
        raise FileNotFoundError(f"Feature schema not found at {schema_path}")
    
    with open(schema_path, 'r') as f:
        schema = json.load(f)
    
    logger.info(f"Loaded feature schema v{schema.get('schema_version', 'unknown')}")
    return schema

def validate_ohlcv_data(df: pd.DataFrame) -> bool:
    """Validate input OHLCV data format and completeness"""
    schema = load_feature_schema()
    required_columns = schema["required_ohlcv_columns"]
    
    # Check required columns
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required OHLCV columns: {missing_cols}")
    
    # Check minimum data length
    min_periods = schema["minimum_lookback_periods"]
    if len(df) < min_periods:
        raise ValueError(f"Insufficient data: {len(df)} rows, minimum {min_periods} required")
    
    # Check for NaN values in critical columns
    critical_cols = ["open", "high", "low", "close", "volume"]
    for col in critical_cols:
        if df[col].isna().any():
            logger.warning(f"NaN values found in {col}, will forward-fill")
            df[col] = df[col].fillna(method='ffill')
    
    # Validate price bounds
    price_cols = ["open", "high", "low", "close"]
    bounds = schema["validation_rules"]["price_bounds"]
    
    for col in price_cols:
        if (df[col] < bounds["min"]).any() or (df[col] > bounds["max"]).any():
            raise ValueError(f"Price values in {col} outside valid bounds [{bounds['min']}, {bounds['max']}]")
    
    # Validate volume bounds
    vol_bounds = schema["validation_rules"]["volume_bounds"]
    if (df["volume"] < vol_bounds["min"]).any() or (df["volume"] > vol_bounds["max"]).any():
        raise ValueError(f"Volume values outside valid bounds [{vol_bounds['min']}, {vol_bounds['max']}]")
    
    logger.info(f"OHLCV validation passed for {len(df)} rows")
    return True

def compute_price_features(df: pd.DataFrame) -> Dict[str, np.ndarray]:
    """Compute price-based features deterministically"""
    features = {}
    
    # Current price (close)
    features['current_price'] = df['close'].values
    
    # Price change ratio
    price_change = np.zeros(len(df))
    price_change[1:] = (df['close'].values[1:] - df['close'].values[:-1]) / df['close'].values[:-1]
    features['price_change'] = price_change
    
    # High-Low spread ratio
    features['high_low_spread'] = (df['high'].values - df['low'].values) / df['close'].values
    
    # Open-Close spread ratio  
    features['open_close_spread'] = (df['close'].values - df['open'].values) / df['open'].values
    
    logger.debug(f"Computed {len(features)} price features")
    return features

def compute_volume_features(df: pd.DataFrame) -> Dict[str, np.ndarray]:
    """Compute volume-based features deterministically"""
    features = {}
    
    # Volume SMA 20
    volume_sma_20 = ta.SMA(df['volume'].values.astype(np.float64), timeperiod=20)
    features['volume_sma_20'] = np.nan_to_num(volume_sma_20, nan=df['volume'].iloc[-1])
    
    # Volume ratio (current / 20-period average)
    volume_ratio = df['volume'].values / features['volume_sma_20']
    volume_ratio = np.nan_to_num(volume_ratio, nan=1.0)
    features['volume_ratio'] = volume_ratio
    
    # Volume weighted price (approximation using close)
    typical_price = (df['high'].values + df['low'].values + df['close'].values) / 3
    vwap = typical_price  # Simplified VWAP approximation
    features['volume_weighted_price'] = vwap
    
    logger.debug(f"Computed {len(features)} volume features")
    return features

def compute_technical_features(df: pd.DataFrame) -> Dict[str, np.ndarray]:
    """Compute technical indicator features deterministically"""
    features = {}
    
    # Ensure data types
    high = df['high'].values.astype(np.float64)
    low = df['low'].values.astype(np.float64)
    close = df['close'].values.astype(np.float64)
    volume = df['volume'].values.astype(np.float64)
    
    # Simple Moving Averages
    for period in [5, 10, 20, 50]:
        if len(close) >= period:
            sma = ta.SMA(close, timeperiod=period)
            features[f'sma_{period}'] = np.nan_to_num(sma, nan=close[-1])
        else:
            features[f'sma_{period}'] = np.full(len(close), close[-1])
    
    # Exponential Moving Averages
    for period in [5, 10, 20, 50]:
        if len(close) >= period:
            ema = ta.EMA(close, timeperiod=period)
            features[f'ema_{period}'] = np.nan_to_num(ema, nan=close[-1])
        else:
            features[f'ema_{period}'] = np.full(len(close), close[-1])
    
    # RSI
    if len(close) >= 14:
        rsi = ta.RSI(close, timeperiod=14)
        features['rsi'] = np.nan_to_num(rsi, nan=50.0)
    else:
        features['rsi'] = np.full(len(close), 50.0)
    
    # MACD
    if len(close) >= 26:
        macd, macd_signal, macd_hist = ta.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
        features['macd'] = np.nan_to_num(macd, nan=0.0)
        features['macd_signal'] = np.nan_to_num(macd_signal, nan=0.0)
        features['macd_histogram'] = np.nan_to_num(macd_hist, nan=0.0)
    else:
        for name in ['macd', 'macd_signal', 'macd_histogram']:
            features[name] = np.zeros(len(close))
    
    # Bollinger Bands
    if len(close) >= 20:
        bb_upper, bb_middle, bb_lower = ta.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2)
        features['bb_upper'] = np.nan_to_num(bb_upper, nan=close[-1] * 1.02)
        features['bb_middle'] = np.nan_to_num(bb_middle, nan=close[-1])
        features['bb_lower'] = np.nan_to_num(bb_lower, nan=close[-1] * 0.98)
        
        # BB Position (0 = at lower band, 1 = at upper band)
        bb_position = np.where(
            (features['bb_upper'] - features['bb_lower']) != 0,
            (close - features['bb_lower']) / (features['bb_upper'] - features['bb_lower']),
            0.5
        )
        features['bb_position'] = np.clip(bb_position, 0.0, 1.0)
    else:
        features['bb_upper'] = close * 1.02
        features['bb_middle'] = close.copy()
        features['bb_lower'] = close * 0.98
        features['bb_position'] = np.full(len(close), 0.5)
    
    # ATR
    if len(close) >= 14:
        atr = ta.ATR(high, low, close, timeperiod=14)
        features['atr'] = np.nan_to_num(atr, nan=np.std(close[-14:]) if len(close) >= 14 else 0.01 * close[-1])
    else:
        features['atr'] = np.full(len(close), 0.01 * close[-1])
    
    # ADX
    if len(close) >= 14:
        adx = ta.ADX(high, low, close, timeperiod=14)
        features['adx'] = np.nan_to_num(adx, nan=25.0)
    else:
        features['adx'] = np.full(len(close), 25.0)
    
    logger.debug(f"Computed {len(features)} technical features")
    return features

def compute_features_deterministic(df: pd.DataFrame, symbol: str = None) -> FeatureComputationResult:
    """
    Main deterministic feature computation function
    
    This function ensures identical feature computation between training and inference
    by following a strict order of operations and handling edge cases consistently.
    
    Args:
        df: OHLCV DataFrame with columns [open, high, low, close, volume, timestamp]
        symbol: Optional symbol identifier for logging
        
    Returns:
        FeatureComputationResult with features DataFrame and metadata
    """
    logger.info(f"Computing features for {symbol or 'unknown symbol'} with {len(df)} rows")
    
    # Validate input data
    validate_ohlcv_data(df)
    
    # Load schema for computation order
    schema = load_feature_schema()
    computation_order = schema["feature_computation_order"]
    
    all_features = {}
    missing_count = 0
    
    # Compute features in deterministic order
    for category in computation_order:
        if category == "price":
            price_features = compute_price_features(df)
            all_features.update(price_features)
        elif category == "volume":
            volume_features = compute_volume_features(df)
            all_features.update(volume_features)
        elif category == "technical":
            technical_features = compute_technical_features(df)
            all_features.update(technical_features)
    
    # Create features DataFrame
    feature_df = pd.DataFrame(all_features, index=df.index)
    
    # Validate against schema
    validate_feature_schema(feature_df, schema)
    
    # Count missing values
    missing_count = feature_df.isnull().sum().sum()
    if missing_count > 0:
        logger.warning(f"Found {missing_count} missing values in computed features")
    
    # Create result
    result = FeatureComputationResult(
        features=feature_df,
        feature_count=len(feature_df.columns),
        missing_count=missing_count,
        schema_version=schema["schema_version"],
        computation_metadata={
            "input_rows": len(df),
            "output_rows": len(feature_df),
            "symbol": symbol,
            "feature_categories": list(all_features.keys()),
            "computation_order": computation_order
        }
    )
    
    logger.info(f"✅ Feature computation completed: {result.feature_count} features, {missing_count} missing values")
    return result

def validate_feature_schema(feature_df: pd.DataFrame, schema: Dict[str, Any]) -> bool:
    """Validate computed features against schema definition"""
    
    # Collect all expected features from schema
    expected_features = []
    for category_name, category_info in schema["feature_categories"].items():
        for feature in category_info["features"]:
            expected_features.append(feature["name"])
    
    # Check for missing features
    missing_features = [f for f in expected_features if f not in feature_df.columns]
    if missing_features:
        raise SchemaError(f"Missing features: {missing_features}")
    
    # Check for unexpected features
    unexpected_features = [f for f in feature_df.columns if f not in expected_features]
    if unexpected_features:
        raise SchemaError(f"Unexpected features: {unexpected_features}")
    
    # Validate data types and bounds
    for category_name, category_info in schema["feature_categories"].items():
        for feature in category_info["features"]:
            fname = feature["name"]
            if fname in feature_df.columns:
                
                # Check data type
                expected_dtype = feature["dtype"]
                if not feature_df[fname].dtype == expected_dtype:
                    logger.warning(f"Feature {fname} has dtype {feature_df[fname].dtype}, expected {expected_dtype}")
                
                # Check specific validation rules
                if fname == "rsi":
                    bounds = schema["validation_rules"]["rsi_bounds"]
                    if (feature_df[fname] < bounds["min"]).any() or (feature_df[fname] > bounds["max"]).any():
                        logger.warning(f"RSI values outside bounds [{bounds['min']}, {bounds['max']}]")
                
                elif fname == "bb_position":
                    bounds = schema["validation_rules"]["bb_position_bounds"]
                    if (feature_df[fname] < bounds["min"]).any() or (feature_df[fname] > bounds["max"]).any():
                        logger.warning(f"BB position values outside bounds [{bounds['min']}, {bounds['max']}]")
    
    logger.info("✅ Feature schema validation passed")
    return True

def get_feature_columns() -> List[str]:
    """Get ordered list of feature columns for consistent ML pipeline"""
    schema = load_feature_schema()
    
    feature_columns = []
    for category_name, category_info in schema["feature_categories"].items():
        for feature in category_info["features"]:
            feature_columns.append(feature["name"])
    
    return feature_columns

def export_features_for_ml(feature_df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """Export features in consistent order for ML training/inference"""
    feature_columns = get_feature_columns()
    
    # Ensure all required columns are present
    missing_cols = [col for col in feature_columns if col not in feature_df.columns]
    if missing_cols:
        raise ValueError(f"Missing feature columns for ML export: {missing_cols}")
    
    # Return features in consistent order
    ordered_features = feature_df[feature_columns].copy()
    
    logger.info(f"Exported {len(feature_columns)} features for ML in deterministic order")
    return ordered_features, feature_columns


# Example usage and testing functions
def test_feature_computation():
    """Test feature computation with synthetic data"""
    
    # Create synthetic OHLCV data
    np.random.seed(42)  # Deterministic test
    n_periods = 100
    
    base_price = 50000
    prices = [base_price]
    
    for i in range(n_periods - 1):
        change = np.random.normal(0, 0.02) * prices[-1]
        new_price = max(prices[-1] + change, base_price * 0.5)  # Floor price
        prices.append(new_price)
    
    # Generate OHLCV from price series
    data = []
    for i, price in enumerate(prices):
        high = price * (1 + abs(np.random.normal(0, 0.01)))
        low = price * (1 - abs(np.random.normal(0, 0.01)))
        open_price = prices[i-1] if i > 0 else price
        volume = np.random.uniform(1000, 10000)
        
        data.append({
            'timestamp': pd.Timestamp('2024-01-01') + pd.Timedelta(minutes=i),
            'open': open_price,
            'high': high,
            'low': low, 
            'close': price,
            'volume': volume
        })
    
    df = pd.DataFrame(data)
    
    # Compute features
    result = compute_features_deterministic(df, symbol="TEST")
    
    print(f"✅ Test completed: {result.feature_count} features computed")
    print(f"Schema version: {result.schema_version}")
    print(f"Missing values: {result.missing_count}")
    
    return result

if __name__ == "__main__":
    # Run test
    logging.basicConfig(level=logging.INFO)
    test_result = test_feature_computation()
    print("\nFeature columns:")
    for col in test_result.features.columns:
        print(f"  - {col}")
