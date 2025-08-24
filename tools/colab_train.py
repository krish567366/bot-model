#!/usr/bin/env python3
"""
Google Colab Training CLI Script for Trading Bot ML Models

This script provides the same functionality as the Colab notebook but in CLI form.
Can be run in Google Colab or any Python environment.

Usage:
    python colab_train.py --repo-url https://github.com/user/trading-bot.git
    python colab_train.py --repo-url https://github.com/user/trading-bot.git --full-training
    python colab_train.py --help

Requirements:
    - Internet connection for cloning repository
    - Python packages: pandas, numpy, lightgbm, scikit-learn, joblib
    - Optional: Google Colab environment for Drive mounting and downloads
"""

import os
import sys
import json
import joblib
import argparse
import subprocess
import shutil
from datetime import datetime
from pathlib import Path

# Configuration defaults
DEFAULT_CONFIG = {
    "fast_test": True,
    "horizon": 5,
    "pos_thresh": 0.002,
    "n_splits": 2,
    "seed": 42,
    "n_estimators": 100,
    "n_estimators_full": 1000
}

class ColabTrainer:
    """Main training class for Google Colab ML training"""
    
    def __init__(self, repo_url, symbol="BTC-USD", interval="1m", config=None):
        self.repo_url = repo_url
        self.symbol = symbol
        self.interval = interval
        self.config = config or DEFAULT_CONFIG.copy()
        
        # Paths
        self.repo_name = self._extract_repo_name(repo_url)
        self.repo_path = f"/content/{self.repo_name}" if self._is_colab() else f"./{self.repo_name}"
        self.model_save_repo_path = f"{self.repo_path}/models/"
        self.model_save_drive_path = "/content/drive/MyDrive/trading_bot_models/"
        self.run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # State
        self.drive_mounted = False
        self.modules_imported = False
        self.saved_models = {}
        
    def _extract_repo_name(self, url):
        """Extract repository name from URL"""
        return url.rstrip('/').split('/')[-1].replace('.git', '')
    
    def _is_colab(self):
        """Check if running in Google Colab"""
        try:
            import google.colab
            return True
        except ImportError:
            return False
    
    def log(self, message, level="INFO"):
        """Log message with timestamp"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"[{timestamp}] {level}: {message}")
    
    def clone_repository(self):
        """Clone the GitHub repository"""
        if self.repo_url == "<YOUR_REPO_URL>":
            self.log("❌ Please provide a valid repository URL!", "ERROR")
            return False
        
        try:
            self.log(f"🔄 Cloning repository from {self.repo_url}...")
            
            # Remove existing directory if present
            if os.path.exists(self.repo_path):
                self.log("📁 Removing existing repository...")
                shutil.rmtree(self.repo_path)
            
            # Clone repository
            result = subprocess.run(['git', 'clone', self.repo_url, self.repo_path], 
                                  capture_output=True, text=True)
            
            if result.returncode == 0 and os.path.exists(self.repo_path):
                self.log("✅ Repository cloned successfully")
                
                # Add to Python path
                if self.repo_path not in sys.path:
                    sys.path.insert(0, self.repo_path)
                    self.log(f"✅ Added {self.repo_path} to Python path")
                
                return True
            else:
                self.log(f"❌ Repository cloning failed: {result.stderr}", "ERROR")
                return False
                
        except Exception as e:
            self.log(f"❌ Error cloning repository: {e}", "ERROR")
            return False
    
    def install_dependencies(self):
        """Install required dependencies"""
        self.log("🔄 Installing dependencies...")
        
        # Try requirements.txt first
        requirements_path = os.path.join(self.repo_path, "requirements.txt")
        
        if os.path.exists(requirements_path):
            self.log("📄 Found requirements.txt, installing...")
            result = os.system(f"pip install -q -r {requirements_path}")
            
            if result == 0:
                self.log("✅ Requirements installed from requirements.txt")
            else:
                self.log("⚠️  Some packages from requirements.txt failed", "WARNING")
        
        # Core ML packages
        core_packages = [
            "pandas", "numpy", "scikit-learn", "joblib",
            "lightgbm", "xgboost", "matplotlib", "seaborn"
        ]
        
        self.log("🔄 Installing core ML packages...")
        failed_packages = []
        
        for package in core_packages:
            try:
                result = os.system(f"pip install -q {package}")
                if result == 0:
                    self.log(f"  ✓ {package}")
                else:
                    failed_packages.append(package)
                    self.log(f"  ⚠️  {package} - failed", "WARNING")
            except:
                failed_packages.append(package)
                self.log(f"  ❌ {package} - error", "ERROR")
        
        return len(failed_packages) == 0
    
    def mount_google_drive(self):
        """Mount Google Drive if in Colab"""
        if not self._is_colab():
            self.log("⚠️  Not running in Google Colab - Drive mount skipped", "WARNING")
            return False
        
        try:
            from google.colab import drive
            self.log("🔄 Mounting Google Drive...")
            drive.mount('/content/drive')
            
            if os.path.exists('/content/drive/MyDrive'):
                os.makedirs(self.model_save_drive_path, exist_ok=True)
                self.drive_mounted = True
                self.log("✅ Google Drive mounted successfully")
                return True
            else:
                self.log("❌ Drive mount verification failed", "ERROR")
                return False
                
        except Exception as e:
            self.log(f"⚠️  Drive mount failed: {e}", "WARNING")
            return False
    
    def import_modules(self):
        """Import trading bot modules"""
        self.log("🔄 Importing trading bot modules...")
        
        try:
            # Core imports
            import pandas as pd
            import numpy as np
            np.random.seed(self.config['seed'])
            
            # Try to import trading modules (with fallbacks)
            modules = {}
            
            try:
                from ai.feature_engineering_v2 import compute_features_deterministic
                modules['feature_engineering'] = compute_features_deterministic
                self.log("  ✓ ai.feature_engineering_v2")
            except ImportError:
                self.log("  ⚠️  Feature engineering module not found", "WARNING")
            
            self.modules_imported = True
            self.log("✅ Module import completed")
            return modules
            
        except Exception as e:
            self.log(f"❌ Error importing modules: {e}", "ERROR")
            return {}
    
    def generate_training_data(self):
        """Generate synthetic training data"""
        import pandas as pd
        import numpy as np
        
        n_periods = 500 if self.config['fast_test'] else 2000
        self.log(f"🔄 Generating {n_periods} periods of synthetic data...")
        
        # Generate synthetic OHLCV data
        dates = pd.date_range(start='2023-01-01', periods=n_periods, freq='1H')
        
        np.random.seed(self.config['seed'])
        returns = np.random.normal(0.0001, 0.01, n_periods)
        log_prices = np.cumsum(returns)
        prices = 50000 * np.exp(log_prices)
        
        data = []
        for i, (date, price) in enumerate(zip(dates, prices)):
            volatility = abs(np.random.normal(0, 0.008))
            high = price * (1 + volatility)
            low = price * (1 - volatility)
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
        
        df = pd.DataFrame(data)
        
        # Create features
        features = pd.DataFrame(index=df.index)
        features['returns'] = df['close'].pct_change()
        features['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        features['price_ma5'] = df['close'].rolling(5).mean()
        features['price_ma20'] = df['close'].rolling(20).mean()
        features['volume_ratio'] = df['volume'] / df['volume'].rolling(5).mean()
        features['rsi'] = self._compute_rsi(df['close'])
        features['volatility'] = features['returns'].rolling(20).std()
        
        features = features.dropna()
        
        # Create labels
        future_periods = self.config['horizon']
        threshold = self.config['pos_thresh']
        
        future_returns = df['close'].shift(-future_periods) / df['close'] - 1
        labels_binary = (future_returns > threshold).astype(int)
        labels_regression = future_returns
        
        # Align data
        valid_mask = ~future_returns.isna() & ~features.isnull().any(axis=1)
        X = features[valid_mask].reset_index(drop=True)
        y_binary = labels_binary[valid_mask].reset_index(drop=True)
        y_regression = labels_regression[valid_mask].reset_index(drop=True)
        timestamps = df['timestamp'][valid_mask].reset_index(drop=True)
        
        self.log(f"✅ Dataset created: {len(X)} samples, {X.shape[1]} features")
        return X, y_binary, y_regression, timestamps
    
    def _compute_rsi(self, prices, window=14):
        """Compute RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def train_models(self, X, y_binary, y_regression, timestamps):
        """Train LightGBM models"""
        import lightgbm as lgb
        from sklearn.metrics import roc_auc_score, accuracy_score, mean_squared_error, r2_score
        
        self.log("🚀 Training LightGBM models...")
        
        # Create time splits
        n_samples = len(X)
        train_end = int(n_samples * 0.6)
        val_end = int(n_samples * 0.8)
        
        splits = {
            'binary': {
                'X_train': X.iloc[:train_end],
                'y_train': y_binary.iloc[:train_end],
                'X_val': X.iloc[train_end:val_end],
                'y_val': y_binary.iloc[train_end:val_end],
                'X_test': X.iloc[val_end:],
                'y_test': y_binary.iloc[val_end:]
            },
            'regression': {
                'X_train': X.iloc[:train_end],
                'y_train': y_regression.iloc[:train_end],
                'X_val': X.iloc[train_end:val_end],
                'y_val': y_regression.iloc[train_end:val_end],
                'X_test': X.iloc[val_end:],
                'y_test': y_regression.iloc[val_end:]
            }
        }
        
        n_estimators = self.config['n_estimators'] if self.config['fast_test'] else self.config['n_estimators_full']
        
        models = {}
        
        # Train binary classifier
        try:
            binary_params = {
                'objective': 'binary',
                'metric': 'auc',
                'num_leaves': 31,
                'learning_rate': 0.1 if self.config['fast_test'] else 0.05,
                'feature_fraction': 0.8,
                'bagging_fraction': 0.8,
                'random_state': self.config['seed'],
                'verbose': -1
            }
            
            train_data = lgb.Dataset(splits['binary']['X_train'], label=splits['binary']['y_train'])
            val_data = lgb.Dataset(splits['binary']['X_val'], label=splits['binary']['y_val'], reference=train_data)
            
            binary_model = lgb.train(
                binary_params,
                train_data,
                num_boost_round=n_estimators,
                valid_sets=[val_data],
                callbacks=[lgb.early_stopping(50)] if not self.config['fast_test'] else []
            )
            
            # Evaluate binary model
            y_pred_proba = binary_model.predict(splits['binary']['X_test'])
            y_pred = (y_pred_proba > 0.5).astype(int)
            
            binary_metrics = {
                'auc': float(roc_auc_score(splits['binary']['y_test'], y_pred_proba)),
                'accuracy': float(accuracy_score(splits['binary']['y_test'], y_pred)),
                'task_type': 'binary'
            }
            
            models['binary'] = {
                'model': binary_model,
                'params': binary_params,
                'metrics': binary_metrics,
                'splits': splits['binary']
            }
            
            self.log(f"✅ Binary model: AUC={binary_metrics['auc']:.4f}")
            
        except Exception as e:
            self.log(f"❌ Binary model training failed: {e}", "ERROR")
        
        # Train regression model
        try:
            regression_params = {
                'objective': 'regression',
                'metric': 'rmse',
                'num_leaves': 31,
                'learning_rate': 0.1 if self.config['fast_test'] else 0.05,
                'feature_fraction': 0.8,
                'bagging_fraction': 0.8,
                'random_state': self.config['seed'],
                'verbose': -1
            }
            
            train_data = lgb.Dataset(splits['regression']['X_train'], label=splits['regression']['y_train'])
            val_data = lgb.Dataset(splits['regression']['X_val'], label=splits['regression']['y_val'], reference=train_data)
            
            regression_model = lgb.train(
                regression_params,
                train_data,
                num_boost_round=n_estimators,
                valid_sets=[val_data],
                callbacks=[lgb.early_stopping(50)] if not self.config['fast_test'] else []
            )
            
            # Evaluate regression model
            y_pred = regression_model.predict(splits['regression']['X_test'])
            
            regression_metrics = {
                'rmse': float(mean_squared_error(splits['regression']['y_test'], y_pred) ** 0.5),
                'r2': float(r2_score(splits['regression']['y_test'], y_pred)),
                'task_type': 'regression'
            }
            
            models['regression'] = {
                'model': regression_model,
                'params': regression_params,
                'metrics': regression_metrics,
                'splits': splits['regression']
            }
            
            self.log(f"✅ Regression model: RMSE={regression_metrics['rmse']:.6f}")
            
        except Exception as e:
            self.log(f"❌ Regression model training failed: {e}", "ERROR")
        
        return models
    
    def save_artifacts(self, models, X):
        """Save model artifacts"""
        from sklearn.preprocessing import StandardScaler
        
        self.log("💾 Saving model artifacts...")
        
        for model_type, model_info in models.items():
            model_id = f"lgbm_{model_type}_{self.run_timestamp}"
            model_dir = os.path.join(self.model_save_repo_path, self.symbol, self.run_timestamp, model_id)
            os.makedirs(model_dir, exist_ok=True)
            
            # Save model
            model_path = os.path.join(model_dir, "model.pkl")
            joblib.dump(model_info['model'], model_path, compress=3)
            
            # Save scaler
            scaler = StandardScaler()
            scaler.fit(X)
            scaler_path = os.path.join(model_dir, "scaler.pkl")
            joblib.dump(scaler, scaler_path, compress=3)
            
            # Save metadata
            metadata = {
                'model_id': model_id,
                'model_type': f'lightgbm_{model_type}',
                'symbol': self.symbol,
                'interval': self.interval,
                'timestamp': self.run_timestamp,
                'training_config': self.config,
                'model_params': model_info['params'],
                'metrics': model_info['metrics'],
                'feature_names': list(X.columns),
                'n_features': len(X.columns),
                'framework': 'lightgbm',
                'colab_training': True
            }
            
            meta_path = os.path.join(model_dir, "meta.json")
            with open(meta_path, 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
            
            # Copy to Google Drive if mounted
            drive_path = None
            if self.drive_mounted:
                try:
                    drive_model_dir = os.path.join(self.model_save_drive_path, self.symbol, self.run_timestamp, model_id)
                    os.makedirs(os.path.dirname(drive_model_dir), exist_ok=True)
                    
                    if os.path.exists(drive_model_dir):
                        shutil.rmtree(drive_model_dir)
                    
                    shutil.copytree(model_dir, drive_model_dir)
                    drive_path = drive_model_dir
                    self.log(f"  ✅ Copied to Google Drive: {drive_path}")
                except Exception as e:
                    self.log(f"  ⚠️  Failed to copy to Drive: {e}", "WARNING")
            
            self.saved_models[model_type] = {
                'local_path': model_dir,
                'drive_path': drive_path,
                'metadata': metadata
            }
            
            self.log(f"  ✓ {model_type.capitalize()} model saved: {model_id}")
    
    def create_manifest(self, X, timestamps):
        """Create run manifest"""
        runs_dir = os.path.join(self.repo_path, "runs", f"colab-{self.run_timestamp}")
        os.makedirs(runs_dir, exist_ok=True)
        
        # Get git commit if possible
        git_commit = "unknown"
        try:
            result = subprocess.run(['git', 'rev-parse', 'HEAD'], 
                                  cwd=self.repo_path, capture_output=True, text=True)
            if result.returncode == 0:
                git_commit = result.stdout.strip()[:12]
        except:
            pass
        
        manifest = {
            'run_info': {
                'timestamp': self.run_timestamp,
                'git_commit': git_commit,
                'colab_session': self._is_colab(),
                'fast_test_mode': self.config['fast_test']
            },
            'configuration': self.config,
            'data_info': {
                'symbol': self.symbol,
                'interval': self.interval,
                'n_samples': len(X),
                'n_features': len(X.columns),
                'feature_names': list(X.columns)
            },
            'models': {}
        }
        
        for model_type, model_info in self.saved_models.items():
            manifest['models'][model_type] = {
                'model_id': model_info['metadata']['model_id'],
                'local_path': model_info['local_path'],
                'drive_path': model_info['drive_path'],
                'metrics': model_info['metadata']['metrics']
            }
        
        manifest_path = os.path.join(runs_dir, "manifest.json")
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2, default=str)
        
        self.log(f"📋 Manifest created: {manifest_path}")
        return manifest_path
    
    def run_full_training(self):
        """Run the complete training pipeline"""
        self.log("🚀 Starting Google Colab ML Training Pipeline")
        self.log(f"Configuration: {self.config}")
        
        # Step 1: Clone repository
        if not self.clone_repository():
            return False
        
        # Step 2: Install dependencies
        self.install_dependencies()
        
        # Step 3: Mount Google Drive
        self.mount_google_drive()
        
        # Step 4: Import modules
        self.import_modules()
        
        # Step 5: Generate training data
        X, y_binary, y_regression, timestamps = self.generate_training_data()
        
        # Step 6: Train models
        models = self.train_models(X, y_binary, y_regression, timestamps)
        
        if not models:
            self.log("❌ No models were successfully trained", "ERROR")
            return False
        
        # Step 7: Save artifacts
        self.save_artifacts(models, X)
        
        # Step 8: Create manifest
        self.create_manifest(X, timestamps)
        
        # Step 9: Display results
        self.display_results()
        
        self.log("🏆 Training pipeline completed successfully!")
        return True
    
    def display_results(self):
        """Display training results"""
        print("\n" + "="*60)
        print("🏆 TRAINING COMPLETED SUCCESSFULLY!")
        print("="*60)
        
        print(f"\n📊 TRAINING SUMMARY:")
        print(f"  • Run Timestamp: {self.run_timestamp}")
        print(f"  • Symbol: {self.symbol}")
        print(f"  • Fast Test Mode: {self.config['fast_test']}")
        print(f"  • Models Trained: {len(self.saved_models)}")
        
        print(f"\n🎯 MODEL PERFORMANCE:")
        for model_type, model_info in self.saved_models.items():
            metrics = model_info['metadata']['metrics']
            print(f"  {model_type.upper()}: ", end="")
            
            if model_type == 'binary':
                print(f"AUC={metrics['auc']:.4f}, Accuracy={metrics['accuracy']:.4f}")
            else:
                print(f"RMSE={metrics['rmse']:.6f}, R²={metrics['r2']:.4f}")
        
        print(f"\n📁 ARTIFACT LOCATIONS:")
        for model_type, model_info in self.saved_models.items():
            print(f"  {model_type.upper()}: {model_info['local_path']}")
            if model_info['drive_path']:
                print(f"    Drive: {model_info['drive_path']}")


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="Google Colab Training CLI for Trading Bot ML Models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python colab_train.py --repo-url https://github.com/user/trading-bot.git
  python colab_train.py --repo-url https://github.com/user/trading-bot.git --full-training
  python colab_train.py --repo-url https://github.com/user/trading-bot.git --symbol ETH-USD --horizon 10
        """
    )
    
    parser.add_argument('--repo-url', required=True, help='GitHub repository URL')
    parser.add_argument('--symbol', default='BTC-USD', help='Trading symbol (default: BTC-USD)')
    parser.add_argument('--interval', default='1m', help='Time interval (default: 1m)')
    parser.add_argument('--full-training', action='store_true', help='Run full training (not fast test)')
    parser.add_argument('--horizon', type=int, default=5, help='Prediction horizon (default: 5)')
    parser.add_argument('--threshold', type=float, default=0.002, help='Positive threshold (default: 0.002)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed (default: 42)')
    
    args = parser.parse_args()
    
    # Create configuration
    config = DEFAULT_CONFIG.copy()
    config['fast_test'] = not args.full_training
    config['horizon'] = args.horizon
    config['pos_thresh'] = args.threshold
    config['seed'] = args.seed
    
    # Create trainer and run
    trainer = ColabTrainer(
        repo_url=args.repo_url,
        symbol=args.symbol,
        interval=args.interval,
        config=config
    )
    
    success = trainer.run_full_training()
    
    if success:
        print("\n✅ Training completed successfully!")
        print("📥 Check the artifact paths above for your trained models")
        sys.exit(0)
    else:
        print("\n❌ Training failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
