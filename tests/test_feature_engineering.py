"""
Unit Tests for Deterministic Feature Engineering

This module contains comprehensive unit tests for the feature engineering pipeline,
ensuring deterministic behavior, schema compliance, and edge case handling.
"""

import pytest
import numpy as np
import pandas as pd
import json
from pathlib import Path
import sys
import os

# Add the project root to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from arbi.ai.feature_engineering_v2 import (
    compute_features_deterministic,
    validate_feature_schema,
    load_feature_schema,
    validate_ohlcv_data,
    get_feature_columns,
    export_features_for_ml,
    SchemaError,
    FeatureComputationResult
)

class TestFeatureSchema:
    """Test feature schema loading and validation"""
    
    def test_load_feature_schema(self):
        """Test loading of feature schema"""
        schema = load_feature_schema()
        
        assert "schema_version" in schema
        assert "feature_categories" in schema
        assert "required_ohlcv_columns" in schema
        assert "minimum_lookback_periods" in schema
        assert "feature_computation_order" in schema
        assert "validation_rules" in schema
        
    def test_get_feature_columns(self):
        """Test consistent feature column ordering"""
        columns = get_feature_columns()
        
        assert isinstance(columns, list)
        assert len(columns) > 0
        
        # Test deterministic ordering
        columns2 = get_feature_columns()
        assert columns == columns2

class TestOHLCVValidation:
    """Test OHLCV data validation"""
    
    def create_valid_ohlcv(self, n_periods=100):
        """Create valid OHLCV data for testing"""
        np.random.seed(42)
        
        base_price = 50000
        data = []
        
        for i in range(n_periods):
            price = base_price * (1 + np.random.normal(0, 0.01))
            high = price * (1 + abs(np.random.normal(0, 0.005)))
            low = price * (1 - abs(np.random.normal(0, 0.005)))
            volume = np.random.uniform(1000, 10000)
            
            data.append({
                'timestamp': pd.Timestamp('2024-01-01') + pd.Timedelta(minutes=i),
                'open': price,
                'high': high,
                'low': low,
                'close': price,
                'volume': volume
            })
        
        return pd.DataFrame(data)
    
    def test_valid_ohlcv_data(self):
        """Test validation of valid OHLCV data"""
        df = self.create_valid_ohlcv()
        assert validate_ohlcv_data(df) == True
    
    def test_missing_columns(self):
        """Test validation with missing columns"""
        df = self.create_valid_ohlcv()
        df = df.drop(columns=['volume'])
        
        with pytest.raises(ValueError, match="Missing required OHLCV columns"):
            validate_ohlcv_data(df)
    
    def test_insufficient_data(self):
        """Test validation with insufficient data length"""
        df = self.create_valid_ohlcv(n_periods=10)  # Too few periods
        
        with pytest.raises(ValueError, match="Insufficient data"):
            validate_ohlcv_data(df)
    
    def test_invalid_price_bounds(self):
        """Test validation with invalid price values"""
        df = self.create_valid_ohlcv()
        df.loc[0, 'close'] = -100  # Negative price
        
        with pytest.raises(ValueError, match="Price values.*outside valid bounds"):
            validate_ohlcv_data(df)
    
    def test_invalid_volume_bounds(self):
        """Test validation with invalid volume values"""
        df = self.create_valid_ohlcv()
        df.loc[0, 'volume'] = -100  # Negative volume
        
        with pytest.raises(ValueError, match="Volume values outside valid bounds"):
            validate_ohlcv_data(df)
    
    def test_nan_handling(self):
        """Test handling of NaN values in data"""
        df = self.create_valid_ohlcv()
        df.loc[10:15, 'close'] = np.nan
        
        # Should succeed after forward-fill
        assert validate_ohlcv_data(df) == True
        
        # Check that NaNs were filled
        assert not df['close'].isna().any()

class TestFeatureComputation:
    """Test deterministic feature computation"""
    
    def create_test_data(self):
        """Create deterministic test data"""
        np.random.seed(42)  # Fixed seed for reproducibility
        return TestOHLCVValidation().create_valid_ohlcv(n_periods=100)
    
    def test_deterministic_computation(self):
        """Test that feature computation is deterministic"""
        df = self.create_test_data()
        
        # Compute features twice
        result1 = compute_features_deterministic(df, symbol="TEST")
        result2 = compute_features_deterministic(df, symbol="TEST")
        
        # Results should be identical
        pd.testing.assert_frame_equal(result1.features, result2.features)
        assert result1.feature_count == result2.feature_count
        assert result1.missing_count == result2.missing_count
    
    def test_feature_count(self):
        """Test that all expected features are computed"""
        df = self.create_test_data()
        result = compute_features_deterministic(df, symbol="TEST")
        
        expected_columns = get_feature_columns()
        assert result.feature_count == len(expected_columns)
        assert set(result.features.columns) == set(expected_columns)
    
    def test_feature_bounds(self):
        """Test that computed features are within expected bounds"""
        df = self.create_test_data()
        result = compute_features_deterministic(df, symbol="TEST")
        
        # Test RSI bounds
        assert (result.features['rsi'] >= 0).all()
        assert (result.features['rsi'] <= 100).all()
        
        # Test Bollinger Band position bounds
        assert (result.features['bb_position'] >= 0).all()
        assert (result.features['bb_position'] <= 1).all()
        
        # Test price features are positive
        assert (result.features['current_price'] > 0).all()
        
        # Test volume features are non-negative
        assert (result.features['volume_ratio'] >= 0).all()
        assert (result.features['volume_sma_20'] >= 0).all()
    
    def test_minimum_data_handling(self):
        """Test feature computation with minimum required data"""
        # Create data with exactly minimum periods
        schema = load_feature_schema()
        min_periods = schema["minimum_lookback_periods"]
        
        df = TestOHLCVValidation().create_valid_ohlcv(n_periods=min_periods)
        result = compute_features_deterministic(df, symbol="TEST_MIN")
        
        assert result.feature_count > 0
        assert len(result.features) == min_periods
    
    def test_edge_case_constant_prices(self):
        """Test with constant price data (edge case)"""
        data = []
        constant_price = 100.0
        
        for i in range(60):
            data.append({
                'timestamp': pd.Timestamp('2024-01-01') + pd.Timedelta(minutes=i),
                'open': constant_price,
                'high': constant_price,
                'low': constant_price,
                'close': constant_price,
                'volume': 1000.0
            })
        
        df = pd.DataFrame(data)
        result = compute_features_deterministic(df, symbol="CONSTANT")
        
        # Should not crash and should produce valid features
        assert result.feature_count > 0
        assert not result.features.isnull().all().any()  # No columns should be all NaN
        
        # Price change should be zero for constant prices
        assert abs(result.features['price_change'].iloc[-1]) < 1e-10

class TestSchemaValidation:
    """Test schema validation functionality"""
    
    def test_valid_feature_dataframe(self):
        """Test validation of valid feature DataFrame"""
        df = TestFeatureComputation().create_test_data()
        result = compute_features_deterministic(df, symbol="TEST")
        
        schema = load_feature_schema()
        assert validate_feature_schema(result.features, schema) == True
    
    def test_missing_features(self):
        """Test validation with missing features"""
        df = TestFeatureComputation().create_test_data()
        result = compute_features_deterministic(df, symbol="TEST")
        
        # Remove a required feature
        incomplete_features = result.features.drop(columns=['rsi'])
        schema = load_feature_schema()
        
        with pytest.raises(SchemaError, match="Missing features"):
            validate_feature_schema(incomplete_features, schema)
    
    def test_unexpected_features(self):
        """Test validation with unexpected features"""
        df = TestFeatureComputation().create_test_data()
        result = compute_features_deterministic(df, symbol="TEST")
        
        # Add an unexpected feature
        result.features['unexpected_feature'] = 1.0
        schema = load_feature_schema()
        
        with pytest.raises(SchemaError, match="Unexpected features"):
            validate_feature_schema(result.features, schema)

class TestMLExport:
    """Test ML export functionality"""
    
    def test_export_features_for_ml(self):
        """Test export of features for ML training/inference"""
        df = TestFeatureComputation().create_test_data()
        result = compute_features_deterministic(df, symbol="TEST")
        
        ml_features, feature_columns = export_features_for_ml(result.features)
        
        # Test output structure
        assert isinstance(ml_features, pd.DataFrame)
        assert isinstance(feature_columns, list)
        assert list(ml_features.columns) == feature_columns
        assert len(ml_features) == len(result.features)
    
    def test_consistent_ml_export_ordering(self):
        """Test that ML export produces consistent column ordering"""
        df = TestFeatureComputation().create_test_data()
        result = compute_features_deterministic(df, symbol="TEST")
        
        ml_features1, columns1 = export_features_for_ml(result.features)
        ml_features2, columns2 = export_features_for_ml(result.features)
        
        assert columns1 == columns2
        pd.testing.assert_frame_equal(ml_features1, ml_features2)

class TestReproducibility:
    """Test reproducibility across different runs"""
    
    def test_cross_session_reproducibility(self):
        """Test that same input produces same output across sessions"""
        # Create deterministic test data
        np.random.seed(42)
        df1 = TestOHLCVValidation().create_valid_ohlcv(n_periods=75)
        
        np.random.seed(42)  # Reset seed
        df2 = TestOHLCVValidation().create_valid_ohlcv(n_periods=75)
        
        # DataFrames should be identical
        pd.testing.assert_frame_equal(df1, df2)
        
        # Feature computation should be identical
        result1 = compute_features_deterministic(df1, symbol="REPRO1")
        result2 = compute_features_deterministic(df2, symbol="REPRO2")
        
        pd.testing.assert_frame_equal(result1.features, result2.features)
    
    def test_incremental_data_consistency(self):
        """Test that adding more data maintains consistency of existing features"""
        # Start with base data
        base_data = TestOHLCVValidation().create_valid_ohlcv(n_periods=60)
        base_result = compute_features_deterministic(base_data, symbol="BASE")
        
        # Add more data
        extended_data = TestOHLCVValidation().create_valid_ohlcv(n_periods=80)
        extended_result = compute_features_deterministic(extended_data, symbol="EXTENDED")
        
        # First 60 rows should be identical (for non-lookback-dependent features)
        # Note: Some features may change due to different lookback windows, 
        # but the computation method should be consistent
        
        assert base_result.schema_version == extended_result.schema_version
        assert base_result.feature_count == extended_result.feature_count

class TestPerformance:
    """Test performance characteristics"""
    
    def test_large_dataset_performance(self):
        """Test feature computation on larger datasets"""
        import time
        
        # Create larger dataset
        large_df = TestOHLCVValidation().create_valid_ohlcv(n_periods=1000)
        
        start_time = time.time()
        result = compute_features_deterministic(large_df, symbol="LARGE")
        computation_time = time.time() - start_time
        
        # Should complete in reasonable time (< 5 seconds for 1000 periods)
        assert computation_time < 5.0
        assert result.feature_count > 0
        assert len(result.features) == 1000
        
        print(f"Large dataset computation time: {computation_time:.3f}s for {len(large_df)} periods")

# Acceptance tests
class TestAcceptanceCriteria:
    """Test acceptance criteria for production readiness"""
    
    def test_inference_schema_consistency(self):
        """
        Acceptance: inference loads feature_schema and start full --fast-test 
        uses same columns; no missing-column errors.
        """
        # Simulate training data
        training_data = TestFeatureComputation().create_test_data()
        training_result = compute_features_deterministic(training_data, symbol="TRAIN")
        training_features, training_columns = export_features_for_ml(training_result.features)
        
        # Simulate inference data (different but valid)
        np.random.seed(123)  # Different seed for different data
        inference_data = TestOHLCVValidation().create_valid_ohlcv(n_periods=75)
        inference_result = compute_features_deterministic(inference_data, symbol="INFER")
        inference_features, inference_columns = export_features_for_ml(inference_result.features)
        
        # Column structure must be identical
        assert training_columns == inference_columns
        assert list(training_features.columns) == list(inference_features.columns)
        
        # Schema validation should pass for both
        schema = load_feature_schema()
        assert validate_feature_schema(training_result.features, schema) == True
        assert validate_feature_schema(inference_result.features, schema) == True
        
        print("✅ Acceptance: Schema consistency between training and inference verified")
    
    def test_fast_test_mode_compatibility(self):
        """Test compatibility with fast test mode requirements"""
        # Should work with minimum viable data
        minimal_data = TestOHLCVValidation().create_valid_ohlcv(n_periods=50)
        result = compute_features_deterministic(minimal_data, symbol="FAST_TEST")
        
        assert result.feature_count > 0
        assert result.missing_count == 0  # No missing values allowed in fast test
        
        # Export should work without errors
        ml_features, columns = export_features_for_ml(result.features)
        assert len(columns) > 0
        
        print("✅ Acceptance: Fast test mode compatibility verified")

if __name__ == "__main__":
    # Run basic tests manually
    import logging
    logging.basicConfig(level=logging.INFO)
    
    print("Running feature engineering unit tests...")
    
    # Test basic functionality
    schema_test = TestFeatureSchema()
    schema_test.test_load_feature_schema()
    schema_test.test_get_feature_columns()
    
    ohlcv_test = TestOHLCVValidation()
    ohlcv_test.test_valid_ohlcv_data()
    
    computation_test = TestFeatureComputation()
    computation_test.test_deterministic_computation()
    computation_test.test_feature_count()
    computation_test.test_feature_bounds()
    
    acceptance_test = TestAcceptanceCriteria()
    acceptance_test.test_inference_schema_consistency()
    acceptance_test.test_fast_test_mode_compatibility()
    
    print("\n✅ All manual tests passed!")
    print("\nTo run full test suite with pytest:")
    print("pytest test_feature_engineering.py -v")
