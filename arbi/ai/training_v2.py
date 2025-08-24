"""
Production Training Pipeline with LightGBM

This module implements a production-ready training pipeline that:
- Uses deterministic feature engineering with schema validation
- Implements proper train/validation/test splits
- Performs hyperparameter tuning with cross-validation
- Saves reproducible model artifacts with full metadata
- Integrates with the model registry for versioning

Key Features:
- LightGBM as default algorithm with hyperparameter tuning
- Deterministic preprocessing and feature scaling
- Model performance evaluation and validation
- Full reproducibility with dataset hashing and random seeds
- Integration with feature engineering schema
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, precision_score, recall_score
import joblib
import json
import time
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass

# Import our modules
from .feature_engineering_v2 import (
    compute_features_deterministic, 
    export_features_for_ml,
    load_feature_schema
)
from .registry import (
    ModelRegistry, 
    get_model_registry,
    compute_dataset_hash
)

logger = logging.getLogger(__name__)

@dataclass
class TrainingConfig:
    """Configuration for training pipeline"""
    # Model configuration
    model_type: str = "lightgbm"
    objective: str = "regression"  # or "classification"
    
    # Data splitting
    test_size: float = 0.2
    validation_size: float = 0.2
    random_state: int = 42
    
    # Feature engineering
    use_feature_scaling: bool = True
    handle_missing: str = "drop"  # "drop", "fill_mean", "fill_median"
    
    # Hyperparameter tuning
    tune_hyperparameters: bool = True
    cv_folds: int = 5
    n_iter: int = 50
    
    # Training settings
    early_stopping_rounds: int = 50
    verbose: bool = False
    
    # Target engineering
    target_transform: str = "none"  # "none", "log", "sqrt", "standardize"
    
    # Validation
    min_validation_score: float = 0.0
    save_feature_importance: bool = True

class ProductionTrainer:
    """Production-ready model training pipeline"""
    
    def __init__(self, config: TrainingConfig = None):
        self.config = config or TrainingConfig()
        self.registry = get_model_registry()
        
        # Training state
        self.feature_names: List[str] = []
        self.scaler: Optional[StandardScaler] = None
        self.label_encoder: Optional[LabelEncoder] = None
        self.target_scaler: Optional[StandardScaler] = None
        
        logger.info(f"Production trainer initialized with {self.config.model_type}")
    
    def prepare_training_data(
        self, 
        ohlcv_data: pd.DataFrame, 
        target_column: str,
        symbol: str
    ) -> Tuple[pd.DataFrame, pd.Series, str]:
        """
        Prepare training data with deterministic feature engineering
        
        Returns:
            features_df: Feature matrix
            target_series: Target variable
            dataset_hash: Unique hash for reproducibility
        """
        logger.info(f"Preparing training data for {symbol}")
        
        # Compute features deterministically
        feature_result = compute_features_deterministic(ohlcv_data, symbol=symbol)
        
        # Export features for ML
        features_df, self.feature_names = export_features_for_ml(feature_result.features)
        
        # Prepare target variable
        if target_column not in ohlcv_data.columns:
            raise ValueError(f"Target column '{target_column}' not found in data")
        
        target_series = ohlcv_data[target_column].copy()
        
        # Align features and target (handle any mismatched indices)
        min_length = min(len(features_df), len(target_series))
        features_df = features_df.iloc[:min_length].copy()
        target_series = target_series.iloc[:min_length].copy()
        
        # Handle missing values
        if self.config.handle_missing == "drop":
            # Drop rows with any missing values
            mask = features_df.notna().all(axis=1) & target_series.notna()
            features_df = features_df[mask]
            target_series = target_series[mask]
        elif self.config.handle_missing == "fill_mean":
            features_df = features_df.fillna(features_df.mean())
            target_series = target_series.fillna(target_series.mean())
        elif self.config.handle_missing == "fill_median":
            features_df = features_df.fillna(features_df.median())
            target_series = target_series.fillna(target_series.median())
        
        # Apply target transformation
        target_series = self._transform_target(target_series)
        
        # Compute dataset hash for reproducibility
        dataset_hash = compute_dataset_hash(
            features_df.values, 
            target_series.values, 
            self.feature_names
        )
        
        logger.info(f"✅ Training data prepared: {len(features_df)} samples, {len(self.feature_names)} features")
        logger.info(f"Dataset hash: {dataset_hash}")
        
        return features_df, target_series, dataset_hash
    
    def _transform_target(self, target: pd.Series) -> pd.Series:
        """Apply target transformation"""
        if self.config.target_transform == "log":
            # Ensure positive values for log transform
            if (target <= 0).any():
                target = target - target.min() + 1e-6
            target = np.log(target)
        elif self.config.target_transform == "sqrt":
            # Ensure non-negative values for sqrt
            if (target < 0).any():
                target = target - target.min()
            target = np.sqrt(target)
        elif self.config.target_transform == "standardize":
            self.target_scaler = StandardScaler()
            target = pd.Series(
                self.target_scaler.fit_transform(target.values.reshape(-1, 1)).flatten(),
                index=target.index
            )
        
        return target
    
    def _inverse_transform_target(self, target: np.ndarray) -> np.ndarray:
        """Reverse target transformation for predictions"""
        if self.config.target_transform == "log":
            return np.exp(target)
        elif self.config.target_transform == "sqrt":
            return np.square(target)
        elif self.config.target_transform == "standardize" and self.target_scaler:
            return self.target_scaler.inverse_transform(target.reshape(-1, 1)).flatten()
        
        return target
    
    def split_data(
        self, 
        X: pd.DataFrame, 
        y: pd.Series
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
        """Split data into train/validation/test sets with proper time-series handling"""
        
        # For time series data, use chronological splitting
        n_samples = len(X)
        n_test = int(n_samples * self.config.test_size)
        n_val = int(n_samples * self.config.validation_size)
        n_train = n_samples - n_test - n_val
        
        if n_train <= 0:
            raise ValueError("Not enough data for train/val/test split")
        
        # Chronological split
        X_train = X.iloc[:n_train].copy()
        y_train = y.iloc[:n_train].copy()
        
        X_val = X.iloc[n_train:n_train+n_val].copy()
        y_val = y.iloc[n_train:n_train+n_val].copy()
        
        X_test = X.iloc[n_train+n_val:].copy()
        y_test = y.iloc[n_train+n_val:].copy()
        
        logger.info(f"Data split: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def preprocess_features(
        self, 
        X_train: pd.DataFrame, 
        X_val: pd.DataFrame, 
        X_test: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Apply feature preprocessing (scaling, etc.)"""
        
        if self.config.use_feature_scaling:
            self.scaler = StandardScaler()
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_val_scaled = self.scaler.transform(X_val)
            X_test_scaled = self.scaler.transform(X_test)
            
            logger.info("✅ Feature scaling applied")
        else:
            X_train_scaled = X_train.values
            X_val_scaled = X_val.values
            X_test_scaled = X_test.values
        
        return X_train_scaled, X_val_scaled, X_test_scaled
    
    def get_base_model(self) -> lgb.LGBMModel:
        """Get base model with default hyperparameters"""
        if self.config.objective == "regression":
            return lgb.LGBMRegressor(
                objective='regression',
                metric='rmse',
                boosting_type='gbdt',
                num_leaves=31,
                learning_rate=0.05,
                feature_fraction=0.9,
                bagging_fraction=0.8,
                bagging_freq=5,
                verbose=-1,
                random_state=self.config.random_state
            )
        else:
            return lgb.LGBMClassifier(
                objective='binary',
                metric='binary_logloss',
                boosting_type='gbdt',
                num_leaves=31,
                learning_rate=0.05,
                feature_fraction=0.9,
                bagging_fraction=0.8,
                bagging_freq=5,
                verbose=-1,
                random_state=self.config.random_state
            )
    
    def tune_hyperparameters(
        self, 
        X_train: np.ndarray, 
        y_train: np.ndarray
    ) -> Dict[str, Any]:
        """Perform hyperparameter tuning with cross-validation"""
        logger.info("Starting hyperparameter tuning...")
        
        # Define parameter grid
        param_grid = {
            'num_leaves': [15, 31, 63],
            'learning_rate': [0.01, 0.05, 0.1],
            'feature_fraction': [0.8, 0.9, 1.0],
            'bagging_fraction': [0.8, 0.9, 1.0],
            'min_child_samples': [10, 20, 50],
            'reg_alpha': [0, 0.1, 0.5],
            'reg_lambda': [0, 0.1, 0.5]
        }
        
        # Use TimeSeriesSplit for cross-validation
        cv_splitter = TimeSeriesSplit(n_splits=self.config.cv_folds)
        
        # Get base model
        base_model = self.get_base_model()
        
        # Perform grid search
        grid_search = GridSearchCV(
            estimator=base_model,
            param_grid=param_grid,
            cv=cv_splitter,
            scoring='neg_mean_squared_error' if self.config.objective == 'regression' else 'accuracy',
            n_jobs=-1,
            verbose=1 if self.config.verbose else 0
        )
        
        start_time = time.time()
        grid_search.fit(X_train, y_train)
        tuning_time = time.time() - start_time
        
        logger.info(f"✅ Hyperparameter tuning completed in {tuning_time:.2f}s")
        logger.info(f"Best score: {grid_search.best_score_:.4f}")
        logger.info(f"Best params: {grid_search.best_params_}")
        
        return grid_search.best_params_
    
    def train_model(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        hyperparameters: Dict[str, Any]
    ) -> lgb.LGBMModel:
        """Train final model with best hyperparameters"""
        
        # Create model with tuned hyperparameters
        if self.config.objective == "regression":
            model = lgb.LGBMRegressor(**hyperparameters, random_state=self.config.random_state, verbose=-1)
        else:
            model = lgb.LGBMClassifier(**hyperparameters, random_state=self.config.random_state, verbose=-1)
        
        # Train with early stopping
        # Convert to DataFrame with feature names to avoid warnings
        X_train_df = pd.DataFrame(X_train, columns=self.feature_names)
        X_val_df = pd.DataFrame(X_val, columns=self.feature_names)
        
        model.fit(
            X_train_df, y_train,
            eval_set=[(X_val_df, y_val)],
            callbacks=[lgb.early_stopping(self.config.early_stopping_rounds)] if self.config.early_stopping_rounds else None
        )
        
        logger.info("✅ Model training completed")
        return model
    
    def evaluate_model(
        self,
        model: lgb.LGBMModel,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray
    ) -> Tuple[Dict[str, float], List[float], float, float]:
        """Comprehensive model evaluation"""
        
        # Predictions - convert to DataFrames with feature names
        X_train_df = pd.DataFrame(X_train, columns=self.feature_names)
        X_val_df = pd.DataFrame(X_val, columns=self.feature_names)
        X_test_df = pd.DataFrame(X_test, columns=self.feature_names)
        
        y_train_pred = model.predict(X_train_df)
        y_val_pred = model.predict(X_val_df)
        y_test_pred = model.predict(X_test_df)
        
        # Apply inverse target transformation if needed
        if self.config.target_transform != "none":
            y_train_pred = self._inverse_transform_target(y_train_pred)
            y_val_pred = self._inverse_transform_target(y_val_pred)
            y_test_pred = self._inverse_transform_target(y_test_pred)
        
        if self.config.objective == "regression":
            # Regression metrics
            train_mae = mean_absolute_error(y_train, y_train_pred)
            val_mae = mean_absolute_error(y_val, y_val_pred)
            test_mae = mean_absolute_error(y_test, y_test_pred)
            
            train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
            val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
            test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
            
            train_r2 = r2_score(y_train, y_train_pred)
            val_r2 = r2_score(y_val, y_val_pred)
            test_r2 = r2_score(y_test, y_test_pred)
            
            performance_metrics = {
                'train_mae': train_mae,
                'val_mae': val_mae,
                'test_mae': test_mae,
                'train_rmse': train_rmse,
                'val_rmse': val_rmse,
                'test_rmse': test_rmse,
                'train_r2': train_r2,
                'val_r2': val_r2,
                'test_r2': test_r2
            }
            
            validation_score = val_r2  # Use R² as primary validation metric
            test_score = test_r2
            
        else:
            # Classification metrics
            train_acc = accuracy_score(y_train, (y_train_pred > 0.5).astype(int))
            val_acc = accuracy_score(y_val, (y_val_pred > 0.5).astype(int))
            test_acc = accuracy_score(y_test, (y_test_pred > 0.5).astype(int))
            
            performance_metrics = {
                'train_accuracy': train_acc,
                'val_accuracy': val_acc,
                'test_accuracy': test_acc
            }
            
            validation_score = val_acc
            test_score = test_acc
        
        # Cross-validation scores
        cv_splitter = TimeSeriesSplit(n_splits=self.config.cv_folds)
        X_full = np.vstack([X_train, X_val])
        y_full = np.hstack([y_train, y_val])
        
        cv_scores = cross_val_score(
            model, pd.DataFrame(X_full, columns=self.feature_names), y_full, 
            cv=cv_splitter,
            scoring='r2' if self.config.objective == 'regression' else 'accuracy'
        ).tolist()
        
        logger.info(f"✅ Model evaluation completed")
        logger.info(f"Validation score: {validation_score:.4f}")
        logger.info(f"Test score: {test_score:.4f}")
        logger.info(f"CV scores: {cv_scores}")
        
        return performance_metrics, cv_scores, validation_score, test_score
    
    def get_feature_importance(self, model: lgb.LGBMModel) -> Dict[str, float]:
        """Extract feature importance"""
        if not self.config.save_feature_importance:
            return {}
        
        importance_scores = model.feature_importances_
        feature_importance = dict(zip(self.feature_names, importance_scores.tolist()))
        
        # Sort by importance
        feature_importance = dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True))
        
        logger.info(f"Top 5 features: {list(feature_importance.keys())[:5]}")
        return feature_importance
    
    def train_full_pipeline(
        self,
        ohlcv_data: pd.DataFrame,
        target_column: str,
        symbol: str
    ) -> str:
        """
        Run complete training pipeline and register model
        
        Returns:
            model_id: ID of registered model
        """
        start_time = time.time()
        logger.info(f"Starting full training pipeline for {symbol}")
        
        # Set random seed for reproducibility
        np.random.seed(self.config.random_state)
        
        # Prepare data
        X, y, dataset_hash = self.prepare_training_data(ohlcv_data, target_column, symbol)
        
        # Split data
        X_train, X_val, X_test, y_train, y_val, y_test = self.split_data(X, y)
        
        # Preprocess features
        X_train_scaled, X_val_scaled, X_test_scaled = self.preprocess_features(X_train, X_val, X_test)
        
        # Hyperparameter tuning
        if self.config.tune_hyperparameters:
            best_params = self.tune_hyperparameters(X_train_scaled, y_train.values)
        else:
            best_params = {}
        
        # Train model
        model = self.train_model(X_train_scaled, y_train.values, X_val_scaled, y_val.values, best_params)
        
        # Evaluate model
        performance_metrics, cv_scores, validation_score, test_score = self.evaluate_model(
            model, X_train_scaled, y_train.values, X_val_scaled, y_val.values, X_test_scaled, y_test.values
        )
        
        # Check validation threshold
        if validation_score < self.config.min_validation_score:
            raise ValueError(f"Validation score {validation_score:.4f} below threshold {self.config.min_validation_score}")
        
        # Get feature importance
        feature_importance = self.get_feature_importance(model)
        
        # Training duration
        training_duration = time.time() - start_time
        
        # Register model
        model_id = self.registry.register_model(
            model_obj=model,
            scaler_obj=self.scaler,
            symbol=symbol,
            model_type=self.config.model_type,
            hyperparameters=best_params,
            preprocessing_config={
                "scaler": "StandardScaler" if self.config.use_feature_scaling else "None",
                "target_transform": self.config.target_transform,
                "handle_missing": self.config.handle_missing
            },
            validation_score=validation_score,
            performance_metrics=performance_metrics,
            dataset_hash=dataset_hash,
            training_samples=len(X_train),
            feature_count=len(self.feature_names),
            random_seed=self.config.random_state,
            cross_val_scores=cv_scores,
            training_duration_seconds=training_duration,
            test_score=test_score,
            feature_importance=feature_importance,
            notes=f"Trained on {len(ohlcv_data)} OHLCV periods"
        )
        
        logger.info(f"✅ Full training pipeline completed in {training_duration:.2f}s")
        logger.info(f"Model registered: {model_id}")
        
        return model_id

# Convenience function
def train_lightgbm_model(
    ohlcv_data: pd.DataFrame,
    target_column: str,
    symbol: str,
    config: TrainingConfig = None
) -> str:
    """Convenience function to train LightGBM model with defaults"""
    trainer = ProductionTrainer(config or TrainingConfig())
    return trainer.train_full_pipeline(ohlcv_data, target_column, symbol)

# Example usage and testing
def test_training_pipeline():
    """Test the training pipeline with synthetic data"""
    logger.info("Testing training pipeline...")
    
    # Create synthetic OHLCV data
    np.random.seed(42)
    n_periods = 500
    
    dates = pd.date_range('2023-01-01', periods=n_periods, freq='1H')
    
    # Generate realistic price data with trend
    base_price = 50000
    trend = np.linspace(0, 0.2, n_periods)  # 20% uptrend
    noise = np.random.normal(0, 0.02, n_periods)  # 2% volatility
    
    prices = base_price * (1 + trend + noise.cumsum() * 0.1)
    
    # Generate OHLCV
    data = []
    for i, price in enumerate(prices):
        high = price * (1 + abs(np.random.normal(0, 0.01)))
        low = price * (1 - abs(np.random.normal(0, 0.01)))
        open_price = prices[i-1] if i > 0 else price
        volume = np.random.uniform(100, 1000)
        
        # Create target: future return over next 5 periods
        if i < len(prices) - 5:
            target = (prices[i+5] - price) / price
        else:
            target = 0
        
        data.append({
            'timestamp': dates[i],
            'open': open_price,
            'high': high,
            'low': low,
            'close': price,
            'volume': volume,
            'future_return': target
        })
    
    df = pd.DataFrame(data)
    
    # Configure training
    config = TrainingConfig(
        tune_hyperparameters=False,  # Skip for faster testing
        verbose=True,
        min_validation_score=0.01  # Low threshold for synthetic test data
    )
    
    # Train model
    model_id = train_lightgbm_model(df, 'future_return', 'TEST', config)
    
    print(f"✅ Test training completed: {model_id}")
    
    # Test model loading
    registry = get_model_registry()
    model, scaler, metadata = registry.load_model(model_id)
    
    print(f"✅ Model loaded successfully: {metadata.validation_score:.4f}")
    
    return model_id

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_model_id = test_training_pipeline()
