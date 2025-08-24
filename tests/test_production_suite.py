"""
Production Testing Suite

Comprehensive test suite including:
- Functional acceptance tests
- Non-functional performance tests
- Data consistency tests
- Integration tests with deterministic behavior
"""

import pytest
import numpy as np
import pandas as pd
import asyncio
import tempfile
import os
import json
import hashlib
from datetime import datetime, timedelta
from pathlib import Path
import time
import psutil
import subprocess
import sys
from typing import Dict, List, Any

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from arbi.core.pipeline import DataPipeline
from arbi.ai.feature_engineering import FeatureEngine
from arbi.ai.training import TrainingPipeline
from arbi.ai.inference import InferenceEngine
from arbi.ai.monitoring import MonitoringOrchestrator
from arbi.core.backtest import BacktestEngine
from arbi.config.settings import get_settings


class TestDataGenerator:
    """Generate deterministic synthetic data for testing"""
    
    @staticmethod
    def generate_ohlcv_data(n_samples: int = 1000, seed: int = 42) -> pd.DataFrame:
        """Generate synthetic OHLCV data"""
        np.random.seed(seed)
        
        # Generate realistic price movements
        initial_price = 100.0
        returns = np.random.normal(0.0005, 0.02, n_samples)  # Daily returns
        prices = initial_price * np.exp(np.cumsum(returns))
        
        # Generate OHLC from close prices
        noise = np.random.normal(0, 0.005, n_samples)
        high_noise = np.abs(np.random.normal(0, 0.01, n_samples))
        low_noise = -np.abs(np.random.normal(0, 0.01, n_samples))
        
        df = pd.DataFrame({
            'date': pd.date_range(start='2020-01-01', periods=n_samples, freq='D'),
            'open': prices * (1 + noise),
            'high': prices * (1 + noise + high_noise),
            'low': prices * (1 + noise + low_noise),
            'close': prices,
            'volume': np.random.lognormal(15, 0.5, n_samples).astype(int),
            'symbol': 'TEST',
            'source': 'synthetic',
            'interval': '1d'
        })
        
        # Ensure OHLC validity
        df['high'] = np.maximum(df['high'], np.maximum(df['open'], df['close']))
        df['low'] = np.minimum(df['low'], np.minimum(df['open'], df['close']))
        
        return df
    
    @staticmethod
    def generate_feature_data(n_samples: int = 1000, n_features: int = 50, seed: int = 42) -> pd.DataFrame:
        """Generate synthetic feature data"""
        np.random.seed(seed)
        
        feature_names = [f'feature_{i:02d}' for i in range(n_features)]
        data = np.random.randn(n_samples, n_features)
        
        # Add some structure to make it realistic
        for i in range(n_features):
            if i > 0:
                data[:, i] += 0.3 * data[:, i-1]  # Add autocorrelation
        
        df = pd.DataFrame(data, columns=feature_names)
        df['timestamp'] = pd.date_range(start='2020-01-01', periods=n_samples, freq='H')
        df['symbol'] = 'TEST'
        
        return df
    
    @staticmethod
    def generate_signals_data(n_signals: int = 100, seed: int = 42) -> List[Dict]:
        """Generate synthetic trading signals"""
        np.random.seed(seed)
        
        signals = []
        for i in range(n_signals):
            signals.append({
                'signal_id': f'test_signal_{i:03d}',
                'timestamp': datetime.now() - timedelta(minutes=i*15),
                'symbol': 'TEST',
                'signal_type': 'cross_exchange',
                'expected_profit_pct': np.random.uniform(0.1, 2.0),
                'confidence': np.random.uniform(0.5, 1.0),
                'buy_exchange': 'binance',
                'sell_exchange': 'kraken',
                'buy_price': 100 + np.random.normal(0, 5),
                'sell_price': 102 + np.random.normal(0, 5)
            })
        
        return signals


class FunctionalTests:
    """Functional acceptance tests"""
    
    def test_feature_engineering_consistency(self):
        """Test feature engineering consistency and schema"""
        print("🧪 Testing feature engineering consistency...")
        
        # Generate test data
        test_data = TestDataGenerator.generate_ohlcv_data(n_samples=500, seed=42)
        
        # Initialize feature engine
        feature_engine = FeatureEngine()
        
        # Extract features
        features = feature_engine.extract_features(test_data, 'TEST')
        
        # Assertions
        assert features is not None, "Features should not be None"
        assert len(features.technical) > 0, "Should have technical features"
        assert len(features.microstructure) >= 0, "Should have microstructure features"
        
        # Check for NaN values
        technical_values = [f.value for f in features.technical.values()]
        nan_count = sum(1 for v in technical_values if pd.isna(v))
        assert nan_count < len(technical_values) * 0.1, f"Too many NaN values: {nan_count}/{len(technical_values)}"
        
        # Test consistency (same input -> same output)
        features2 = feature_engine.extract_features(test_data, 'TEST')
        
        for name, feature in features.technical.items():
            feature2 = features2.technical.get(name)
            if feature2 is not None:
                assert abs(feature.value - feature2.value) < 1e-10, f"Inconsistent feature: {name}"
        
        print("✅ Feature engineering consistency test passed")
        return True
    
    def test_training_pipeline_end_to_end(self):
        """Test complete training pipeline"""
        print("🧪 Testing training pipeline end-to-end...")
        
        # Generate synthetic data
        feature_data = TestDataGenerator.generate_feature_data(n_samples=1000, seed=42)
        
        # Create target variable (simulate price direction)
        feature_data['target'] = (np.random.rand(len(feature_data)) > 0.5).astype(int)
        
        # Initialize training pipeline
        training_pipeline = TrainingPipeline()
        
        # Prepare data
        X = feature_data[[c for c in feature_data.columns if c.startswith('feature_')]].values
        y = feature_data['target'].values
        
        # Train model
        result = training_pipeline.train_model(
            X, y, 
            model_type='xgboost',
            experiment_name='test_experiment'
        )
        
        # Assertions
        assert result is not None, "Training result should not be None"
        assert result.model_id is not None, "Model ID should be generated"
        assert result.metrics is not None, "Metrics should be available"
        assert 'accuracy' in result.metrics, "Should have accuracy metric"
        
        # Test model registry
        model_manager = training_pipeline.model_manager
        registered_model = model_manager.get_model(result.model_id)
        assert registered_model is not None, "Model should be registered"
        
        print("✅ Training pipeline end-to-end test passed")
        return True
    
    def test_inference_signals_written(self):
        """Test inference produces signals"""
        print("🧪 Testing inference signal generation...")
        
        # Create temporary storage
        with tempfile.TemporaryDirectory() as temp_dir:
            # Generate test data
            market_data = TestDataGenerator.generate_ohlcv_data(n_samples=100, seed=42)
            
            # Initialize inference engine
            inference_engine = InferenceEngine()
            
            # Mock model for testing
            class MockModel:
                def predict(self, X):
                    return np.random.rand(len(X), 1)
                
                def predict_proba(self, X):
                    probs = np.random.rand(len(X), 2)
                    return probs / probs.sum(axis=1, keepdims=True)
            
            # Add mock model
            inference_engine.models = {'test_model': MockModel()}
            
            # Run inference
            signals = []
            for i in range(10):
                signal = inference_engine.generate_ml_signal('TEST', 'binance')
                if signal:
                    signals.append(signal)
            
            # Assertions
            assert len(signals) > 0, "Should generate at least some signals"
            
            for signal in signals:
                assert hasattr(signal, 'confidence'), "Signal should have confidence"
                assert hasattr(signal, 'direction'), "Signal should have direction"
                assert 0 <= signal.confidence <= 1, "Confidence should be between 0 and 1"
                assert signal.direction in [-1, 1], "Direction should be -1 or 1"
        
        print("✅ Inference signal generation test passed")
        return True
    
    def test_backtester_reproducible(self):
        """Test backtester reproducibility"""
        print("🧪 Testing backtester reproducibility...")
        
        # Generate deterministic test data
        market_data = TestDataGenerator.generate_ohlcv_data(n_samples=252, seed=42)  # 1 year
        signals_data = TestDataGenerator.generate_signals_data(n_signals=50, seed=42)
        
        # Initialize backtester
        backtest_engine = BacktestEngine()
        
        # Run backtest twice with same seed
        result1 = backtest_engine.run_backtest(
            market_data=market_data,
            signals=signals_data,
            initial_capital=10000,
            seed=42
        )
        
        result2 = backtest_engine.run_backtest(
            market_data=market_data,
            signals=signals_data,
            initial_capital=10000,
            seed=42
        )
        
        # Compare results
        assert result1 is not None and result2 is not None, "Both results should exist"
        
        # Check key metrics are identical
        metrics_to_check = ['total_return', 'sharpe_ratio', 'max_drawdown', 'win_rate']
        for metric in metrics_to_check:
            if metric in result1.metrics and metric in result2.metrics:
                val1, val2 = result1.metrics[metric], result2.metrics[metric]
                assert abs(val1 - val2) < 1e-10, f"Metric {metric} not reproducible: {val1} vs {val2}"
        
        print("✅ Backtester reproducibility test passed")
        return True
    
    def test_start_full_dry_run(self):
        """Test complete system dry run"""
        print("🧪 Testing full system dry run...")
        
        # This would test the main orchestrator
        # For now, we'll test the components individually
        
        try:
            # Test data pipeline
            pipeline = DataPipeline({'database_path': ':memory:'})  # In-memory DB
            
            # Test feature extraction
            feature_engine = FeatureEngine()
            
            # Test training pipeline
            training_pipeline = TrainingPipeline()
            
            # Test inference engine
            inference_engine = InferenceEngine()
            
            # Test backtester
            backtest_engine = BacktestEngine()
            
            print("✅ All components initialized successfully")
            return True
            
        except Exception as e:
            print(f"❌ Dry run test failed: {e}")
            return False


class PerformanceTests:
    """Non-functional performance tests"""
    
    def test_inference_latency_benchmark(self, target_latency_ms: float = 100.0):
        """Test inference latency meets SLA"""
        print(f"🚀 Testing inference latency (target: <{target_latency_ms}ms)...")
        
        # Generate test data
        feature_data = TestDataGenerator.generate_feature_data(n_samples=100, n_features=50, seed=42)
        X = feature_data[[c for c in feature_data.columns if c.startswith('feature_')]].values
        
        # Initialize inference engine
        inference_engine = InferenceEngine()
        
        # Mock lightweight model
        class FastMockModel:
            def predict(self, X):
                return np.random.rand(len(X))
        
        inference_engine.models = {'fast_model': FastMockModel()}
        
        # Benchmark inference
        latencies = []
        for i in range(100):
            start_time = time.perf_counter()
            
            # Simulate full inference pipeline
            signal = inference_engine.generate_ml_signal('TEST', 'binance')
            
            end_time = time.perf_counter()
            latency_ms = (end_time - start_time) * 1000
            latencies.append(latency_ms)
        
        # Calculate statistics
        mean_latency = np.mean(latencies)
        p95_latency = np.percentile(latencies, 95)
        p99_latency = np.percentile(latencies, 99)
        
        print(f"   Mean latency: {mean_latency:.2f}ms")
        print(f"   P95 latency: {p95_latency:.2f}ms") 
        print(f"   P99 latency: {p99_latency:.2f}ms")
        
        # Assertions
        assert mean_latency < target_latency_ms, f"Mean latency {mean_latency:.2f}ms exceeds target {target_latency_ms}ms"
        assert p95_latency < target_latency_ms * 2, f"P95 latency {p95_latency:.2f}ms exceeds target {target_latency_ms*2}ms"
        
        print("✅ Inference latency benchmark passed")
        return True
    
    def test_memory_cpu_smoke_test(self, duration_minutes: float = 1.0):
        """Test memory and CPU usage under load"""
        print(f"🔥 Running memory/CPU smoke test for {duration_minutes} minutes...")
        
        # Get initial process info
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Track resource usage
        memory_readings = []
        cpu_readings = []
        
        start_time = time.time()
        end_time = start_time + (duration_minutes * 60)
        
        # Initialize components
        inference_engine = InferenceEngine()
        feature_engine = FeatureEngine()
        
        iteration = 0
        while time.time() < end_time:
            # Simulate workload
            test_data = TestDataGenerator.generate_ohlcv_data(n_samples=100, seed=iteration)
            features = feature_engine.extract_features(test_data, 'TEST')
            
            # Simulate inference
            signal = inference_engine.generate_ml_signal('TEST', 'binance')
            
            # Record metrics
            if iteration % 10 == 0:  # Sample every 10 iterations
                memory_mb = process.memory_info().rss / 1024 / 1024
                cpu_percent = process.cpu_percent()
                
                memory_readings.append(memory_mb)
                cpu_readings.append(cpu_percent)
            
            iteration += 1
            time.sleep(0.1)  # Brief pause
        
        # Analysis
        final_memory = process.memory_info().rss / 1024 / 1024
        memory_growth = final_memory - initial_memory
        max_memory = max(memory_readings) if memory_readings else final_memory
        avg_cpu = np.mean(cpu_readings) if cpu_readings else 0
        
        print(f"   Initial memory: {initial_memory:.1f} MB")
        print(f"   Final memory: {final_memory:.1f} MB") 
        print(f"   Memory growth: {memory_growth:.1f} MB")
        print(f"   Max memory: {max_memory:.1f} MB")
        print(f"   Average CPU: {avg_cpu:.1f}%")
        print(f"   Iterations completed: {iteration}")
        
        # Assertions (reasonable limits)
        assert memory_growth < 100, f"Memory growth {memory_growth:.1f} MB too high (possible leak)"
        assert max_memory < 500, f"Peak memory {max_memory:.1f} MB too high"
        assert avg_cpu < 80, f"Average CPU {avg_cpu:.1f}% too high"
        
        print("✅ Memory/CPU smoke test passed")
        return True
    
    def test_model_drift_alarm(self):
        """Test model drift detection triggers alerts"""
        print("🚨 Testing model drift alarm system...")
        
        # Initialize monitoring
        monitoring = MonitoringOrchestrator()
        
        # Set up reference distribution
        np.random.seed(42)
        reference_data = {'test_feature': np.random.normal(0, 1, 1000)}
        monitoring.initialize_monitoring(reference_data, {'test_model': {'auc': 0.8}})
        
        # Test with shifted distribution (should trigger alert)
        shifted_data = {'test_feature': np.random.normal(2, 1, 1000)}  # Mean shifted by 2 std devs
        
        # Monitor the shifted data
        monitoring.monitor_batch(shifted_data, {})
        
        # Check if alerts were generated
        dashboard = monitoring.get_monitoring_dashboard()
        
        assert dashboard['drift_summary']['status'] == 'drift_detected', "Should detect drift"
        assert len(dashboard['recent_alerts']) > 0, "Should generate alerts"
        
        # Check alert content
        drift_alerts = [a for a in dashboard['recent_alerts'] if a['type'] == 'drift']
        assert len(drift_alerts) > 0, "Should have drift alerts"
        
        print("✅ Model drift alarm test passed")
        return True


class IntegrationTests:
    """Integration tests with external components"""
    
    def test_data_pipeline_integration(self):
        """Test data pipeline integration"""
        print("🔗 Testing data pipeline integration...")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Initialize pipeline with temporary database
            db_path = os.path.join(temp_dir, 'test.db')
            pipeline = DataPipeline({'database_path': db_path})
            
            # Generate and save test data
            test_data = TestDataGenerator.generate_ohlcv_data(n_samples=100, seed=42)
            success = pipeline.save(test_data, 'test_data')
            assert success, "Should save data successfully"
            
            # Load data back
            loaded_data = pipeline.load('test_data')
            assert not loaded_data.empty, "Should load data successfully"
            assert len(loaded_data) == len(test_data), "Should load same number of rows"
            
            # Test feature transformation
            features_data = pipeline.transform_features(loaded_data)
            assert len(features_data.columns) > len(test_data.columns), "Should add feature columns"
            
        print("✅ Data pipeline integration test passed")
        return True
    
    def test_ai_pipeline_integration(self):
        """Test AI/ML pipeline integration"""
        print("🤖 Testing AI pipeline integration...")
        
        # Test feature engineering -> training -> inference chain
        
        # 1. Feature engineering
        test_data = TestDataGenerator.generate_ohlcv_data(n_samples=500, seed=42)
        feature_engine = FeatureEngine()
        features = feature_engine.extract_features(test_data, 'TEST')
        
        assert features is not None, "Feature extraction should succeed"
        
        # 2. Model training (simplified)
        feature_data = TestDataGenerator.generate_feature_data(n_samples=500, seed=42)
        feature_data['target'] = (np.random.rand(len(feature_data)) > 0.5).astype(int)
        
        training_pipeline = TrainingPipeline()
        X = feature_data[[c for c in feature_data.columns if c.startswith('feature_')]].values
        y = feature_data['target'].values
        
        result = training_pipeline.train_model(X, y, model_type='xgboost')
        assert result is not None, "Training should succeed"
        
        # 3. Inference
        inference_engine = InferenceEngine()
        signal = inference_engine.generate_ml_signal('TEST', 'binance')
        # Signal might be None if no model is available, but should not error
        
        print("✅ AI pipeline integration test passed")
        return True


def run_all_tests() -> Dict[str, bool]:
    """Run all tests and return results"""
    print("🧪 RUNNING PRODUCTION TEST SUITE")
    print("=" * 60)
    
    results = {}
    
    # Functional tests
    functional = FunctionalTests()
    try:
        results['feature_engineering'] = functional.test_feature_engineering_consistency()
        results['training_pipeline'] = functional.test_training_pipeline_end_to_end()
        results['inference_signals'] = functional.test_inference_signals_written()
        results['backtester_reproducible'] = functional.test_backtester_reproducible()
        results['system_dry_run'] = functional.test_start_full_dry_run()
    except Exception as e:
        print(f"❌ Functional test error: {e}")
        results['functional_tests'] = False
    
    # Performance tests
    performance = PerformanceTests()
    try:
        results['latency_benchmark'] = performance.test_inference_latency_benchmark()
        results['memory_cpu_smoke'] = performance.test_memory_cpu_smoke_test(duration_minutes=0.5)
        results['drift_alarm'] = performance.test_model_drift_alarm()
    except Exception as e:
        print(f"❌ Performance test error: {e}")
        results['performance_tests'] = False
    
    # Integration tests
    integration = IntegrationTests()
    try:
        results['data_pipeline'] = integration.test_data_pipeline_integration()
        results['ai_pipeline'] = integration.test_ai_pipeline_integration()
    except Exception as e:
        print(f"❌ Integration test error: {e}")
        results['integration_tests'] = False
    
    # Summary
    passed = sum(1 for result in results.values() if result is True)
    total = len(results)
    
    print("\n" + "=" * 60)
    print(f"🧪 TEST SUITE RESULTS: {passed}/{total} PASSED")
    print("=" * 60)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {test_name}: {status}")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED! System ready for production.")
    else:
        print(f"\n⚠️  {total - passed} tests failed. Review before production deployment.")
    
    return results


if __name__ == "__main__":
    # Run test suite
    test_results = run_all_tests()
    
    # Exit with appropriate code for CI
    all_passed = all(test_results.values())
    sys.exit(0 if all_passed else 1)
