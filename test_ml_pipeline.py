#!/usr/bin/env python3
"""
Comprehensive Testing & Validation Suite for Trading Bot ML Models

This script validates the entire ML pipeline end-to-end:
- Tests Colab notebook functionality
- Validates model artifacts
- Checks inference pipeline
- Runs integration tests
- Performance benchmarking

Usage:
    python test_ml_pipeline.py --full-test
    python test_ml_pipeline.py --quick-test
    python test_ml_pipeline.py --validate-artifacts models/BTC-USD/20240823_143022/
"""

import os
import sys
import json
import joblib
import pandas as pd
import numpy as np
import pytest
from pathlib import Path
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.insert(0, os.path.abspath('.'))

class MLPipelineValidator:
    """Comprehensive ML pipeline validation"""
    
    def __init__(self, test_mode="quick"):
        self.test_mode = test_mode
        self.results = {
            'timestamp': datetime.now().isoformat(),
            'test_mode': test_mode,
            'tests_passed': 0,
            'tests_failed': 0,
            'test_results': {}
        }
        
    def log_test(self, test_name, passed, details=""):
        """Log test result"""
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {test_name}")
        if details:
            print(f"    {details}")
        
        if passed:
            self.results['tests_passed'] += 1
        else:
            self.results['tests_failed'] += 1
            
        self.results['test_results'][test_name] = {
            'passed': passed,
            'details': details,
            'timestamp': datetime.now().isoformat()
        }
    
    def test_imports(self):
        """Test all required imports"""
        try:
            import pandas as pd
            import numpy as np
            import lightgbm as lgb
            import sklearn
            import joblib
            from ai.feature_engineering_v2 import compute_features_deterministic
            from ai.inference_v2 import ProductionInferenceEngine
            from ai.registry import ModelRegistry
            
            self.log_test("Core Imports", True, "All required packages imported successfully")
            return True
        except ImportError as e:
            self.log_test("Core Imports", False, f"Import error: {e}")
            return False
    
    def test_feature_engineering(self):
        """Test feature engineering pipeline"""
        try:
            from ai.feature_engineering_v2 import compute_features_deterministic
            
            # Generate test data
            dates = pd.date_range(start='2024-01-01', periods=100, freq='1H')
            test_data = pd.DataFrame({
                'timestamp': dates,
                'open': np.random.uniform(50000, 51000, 100),
                'high': np.random.uniform(50500, 51500, 100),
                'low': np.random.uniform(49500, 50500, 100),
                'close': np.random.uniform(49800, 50800, 100),
                'volume': np.random.uniform(100, 1000, 100)
            })
            
            # Test feature computation
            result = compute_features_deterministic(test_data, "BTC/USDT")
            
            if hasattr(result, 'features') and len(result.features) > 0:
                n_features = len(result.features.columns)
                self.log_test("Feature Engineering", True, f"Generated {n_features} features")
                return True
            else:
                self.log_test("Feature Engineering", False, "No features generated")
                return False
                
        except Exception as e:
            self.log_test("Feature Engineering", False, f"Error: {e}")
            return False
    
    def test_model_training(self):
        """Test model training functionality"""
        try:
            import lightgbm as lgb
            from sklearn.model_selection import train_test_split
            from sklearn.metrics import roc_auc_score
            
            # Generate synthetic training data
            np.random.seed(42)
            n_samples = 1000
            n_features = 10
            
            X = pd.DataFrame(np.random.randn(n_samples, n_features), 
                           columns=[f'feature_{i}' for i in range(n_features)])
            y = (np.random.rand(n_samples) > 0.6).astype(int)  # 40% positive class
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Train LightGBM model
            train_data = lgb.Dataset(X_train, label=y_train)
            params = {
                'objective': 'binary',
                'metric': 'auc',
                'num_leaves': 31,
                'learning_rate': 0.1,
                'verbose': -1,
                'random_state': 42
            }
            
            model = lgb.train(params, train_data, num_boost_round=100)
            
            # Test predictions
            y_pred = model.predict(X_test)
            auc = roc_auc_score(y_test, y_pred)
            
            if auc > 0.5:  # Better than random
                self.log_test("Model Training", True, f"LightGBM AUC: {auc:.4f}")
                return True
            else:
                self.log_test("Model Training", False, f"Poor AUC: {auc:.4f}")
                return False
                
        except Exception as e:
            self.log_test("Model Training", False, f"Error: {e}")
            return False
    
    def test_model_registry(self):
        """Test model registry functionality"""
        try:
            from ai.registry import ModelRegistry
            
            # Test registry initialization
            registry = ModelRegistry()
            
            # Test basic operations (if methods exist)
            if hasattr(registry, 'list_models'):
                models = registry.list_models()
                self.log_test("Model Registry", True, f"Registry accessible, {len(models)} models found")
                return True
            else:
                self.log_test("Model Registry", True, "Registry initialized successfully")
                return True
                
        except Exception as e:
            self.log_test("Model Registry", False, f"Error: {e}")
            return False
    
    def test_inference_engine(self):
        """Test inference engine"""
        try:
            from ai.inference_v2 import ProductionInferenceEngine
            
            # Initialize engine
            engine = ProductionInferenceEngine()
            
            # Test if we can create the engine
            self.log_test("Inference Engine", True, "ProductionInferenceEngine initialized")
            return True
            
        except Exception as e:
            self.log_test("Inference Engine", False, f"Error: {e}")
            return False
    
    def test_artifact_validation(self, artifact_path=None):
        """Test model artifact validation"""
        if not artifact_path:
            # Look for recent artifacts
            models_dir = Path("models")
            if not models_dir.exists():
                self.log_test("Artifact Validation", False, "No models directory found")
                return False
            
            # Find most recent model
            artifact_dirs = []
            for symbol_dir in models_dir.iterdir():
                if symbol_dir.is_dir():
                    for timestamp_dir in symbol_dir.iterdir():
                        if timestamp_dir.is_dir():
                            for model_dir in timestamp_dir.iterdir():
                                if model_dir.is_dir():
                                    artifact_dirs.append(model_dir)
            
            if not artifact_dirs:
                self.log_test("Artifact Validation", False, "No model artifacts found")
                return False
            
            # Use most recent
            artifact_path = max(artifact_dirs, key=lambda x: x.stat().st_mtime)
        
        try:
            artifact_path = Path(artifact_path)
            
            # Check required files
            required_files = ['model.pkl', 'meta.json', 'scaler.pkl']
            missing_files = []
            
            for file in required_files:
                if not (artifact_path / file).exists():
                    missing_files.append(file)
            
            if missing_files:
                self.log_test("Artifact Validation", False, f"Missing files: {missing_files}")
                return False
            
            # Test loading artifacts
            model = joblib.load(artifact_path / 'model.pkl')
            scaler = joblib.load(artifact_path / 'scaler.pkl')
            
            with open(artifact_path / 'meta.json', 'r') as f:
                metadata = json.load(f)
            
            # Validate metadata
            required_keys = ['model_id', 'model_type', 'feature_names', 'metrics']
            missing_keys = [key for key in required_keys if key not in metadata]
            
            if missing_keys:
                self.log_test("Artifact Validation", False, f"Missing metadata keys: {missing_keys}")
                return False
            
            # Test prediction
            n_features = len(metadata['feature_names'])
            test_X = np.random.randn(5, n_features)
            predictions = model.predict(test_X)
            
            self.log_test("Artifact Validation", True, 
                         f"Artifacts valid: {artifact_path.name}, predictions shape: {predictions.shape}")
            return True
            
        except Exception as e:
            self.log_test("Artifact Validation", False, f"Error validating {artifact_path}: {e}")
            return False
    
    def test_colab_notebook_syntax(self):
        """Test Colab notebook syntax and structure"""
        try:
            notebook_path = Path("notebooks/colab_train_complete.ipynb")
            if not notebook_path.exists():
                self.log_test("Colab Notebook Syntax", False, "Notebook file not found")
                return False
            
            # Read and validate notebook structure
            with open(notebook_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Check for key components
            required_components = [
                'GITHUB_REPO_URL',
                'fast_test',
                'clone_repository',
                'install_dependencies',
                'mount_google_drive',
                'train_lightgbm_model',
                'create_model_artifacts'
            ]
            
            missing_components = []
            for component in required_components:
                if component not in content:
                    missing_components.append(component)
            
            if missing_components:
                self.log_test("Colab Notebook Syntax", False, f"Missing components: {missing_components}")
                return False
            
            # Check for proper XML cell structure
            if '<VSCode.Cell' not in content:
                self.log_test("Colab Notebook Syntax", False, "Invalid notebook structure")
                return False
            
            self.log_test("Colab Notebook Syntax", True, "Notebook structure valid")
            return True
            
        except Exception as e:
            self.log_test("Colab Notebook Syntax", False, f"Error: {e}")
            return False
    
    def test_performance_benchmark(self):
        """Benchmark model training and inference performance"""
        try:
            import time
            import lightgbm as lgb
            
            # Generate benchmark data
            n_samples = 10000 if self.test_mode == "full" else 1000
            n_features = 20
            
            np.random.seed(42)
            X = np.random.randn(n_samples, n_features)
            y = (np.random.rand(n_samples) > 0.5).astype(int)
            
            # Benchmark training
            start_time = time.time()
            train_data = lgb.Dataset(X, label=y)
            model = lgb.train({'objective': 'binary', 'verbose': -1}, 
                            train_data, num_boost_round=100)
            training_time = time.time() - start_time
            
            # Benchmark inference
            start_time = time.time()
            predictions = model.predict(X[:1000])  # Predict on 1000 samples
            inference_time = time.time() - start_time
            
            # Performance thresholds
            max_training_time = 10.0  # seconds
            max_inference_time = 1.0   # seconds
            
            if training_time < max_training_time and inference_time < max_inference_time:
                self.log_test("Performance Benchmark", True, 
                             f"Training: {training_time:.2f}s, Inference: {inference_time:.3f}s")
                return True
            else:
                self.log_test("Performance Benchmark", False, 
                             f"Slow performance - Training: {training_time:.2f}s, Inference: {inference_time:.3f}s")
                return False
                
        except Exception as e:
            self.log_test("Performance Benchmark", False, f"Error: {e}")
            return False
    
    def run_all_tests(self):
        """Run complete test suite"""
        print("🧪 STARTING ML PIPELINE VALIDATION")
        print("=" * 50)
        
        # Core functionality tests
        self.test_imports()
        self.test_feature_engineering()
        self.test_model_training()
        self.test_model_registry()
        self.test_inference_engine()
        
        # Artifact and structure tests
        self.test_artifact_validation()
        self.test_colab_notebook_syntax()
        
        # Performance tests
        if self.test_mode == "full":
            self.test_performance_benchmark()
        
        # Summary
        total_tests = self.results['tests_passed'] + self.results['tests_failed']
        success_rate = (self.results['tests_passed'] / total_tests) * 100 if total_tests > 0 else 0
        
        print("\n" + "=" * 50)
        print("🏆 VALIDATION SUMMARY")
        print("=" * 50)
        print(f"Total Tests: {total_tests}")
        print(f"Passed: {self.results['tests_passed']} ✅")
        print(f"Failed: {self.results['tests_failed']} ❌")
        print(f"Success Rate: {success_rate:.1f}%")
        
        if success_rate >= 80:
            print("🎉 PIPELINE VALIDATION SUCCESSFUL!")
            print("✅ Your ML pipeline is ready for production!")
        else:
            print("⚠️  PIPELINE NEEDS ATTENTION")
            print("❌ Please fix the failed tests before proceeding")
        
        # Save results
        results_path = f"test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(results_path, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        print(f"\n📋 Detailed results saved to: {results_path}")
        
        return success_rate >= 80


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="ML Pipeline Validation Suite")
    parser.add_argument('--full-test', action='store_true', help='Run full test suite')
    parser.add_argument('--quick-test', action='store_true', help='Run quick test suite')
    parser.add_argument('--validate-artifacts', help='Validate specific artifact directory')
    
    args = parser.parse_args()
    
    if args.full_test:
        test_mode = "full"
    else:
        test_mode = "quick"
    
    validator = MLPipelineValidator(test_mode)
    
    if args.validate_artifacts:
        validator.test_artifact_validation(args.validate_artifacts)
    else:
        success = validator.run_all_tests()
        sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
