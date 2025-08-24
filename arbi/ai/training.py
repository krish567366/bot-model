"""
ML Model Training and Experimentation Module

Handles model training, hyperparameter optimization, backtesting,
and automated retraining pipelines.
"""

import asyncio
import logging
import time
import numpy as np
import pandas as pd
from decimal import Decimal
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import json
import pickle
from datetime import datetime, timedelta
from .feature_engineering import FeatureSet, FeatureEngine, Feature, FeatureType

# ML and optimization libraries
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner

from .feature_engineering import FeatureEngine, FeatureSet, get_feature_engine
from .models import (ModelManager, get_model_manager, XGBoostModel, LSTMModelWrapper, 
                    RLTradingAgent, EnsembleModel, ModelPrediction, ModelPerformance,
                    PredictionType)
from ..core.storage import get_storage_manager
from ..config.settings import get_settings

logger = logging.getLogger(__name__)


class TrainingMode(str, Enum):
    """Training modes"""
    FULL_RETRAIN = "full_retrain"
    INCREMENTAL = "incremental"  
    ONLINE = "online"
    TRANSFER_LEARNING = "transfer_learning"


class OptimizationMethod(str, Enum):
    """Hyperparameter optimization methods"""
    GRID_SEARCH = "grid_search"
    RANDOM_SEARCH = "random_search"
    BAYESIAN = "bayesian"
    EVOLUTIONARY = "evolutionary"


@dataclass
class TrainingConfig:
    """Training configuration"""
    model_name: str
    training_mode: TrainingMode
    optimization_method: OptimizationMethod
    train_start_date: datetime
    train_end_date: datetime
    validation_split: float = 0.2
    test_split: float = 0.1
    max_optimization_trials: int = 100
    early_stopping_rounds: int = 50
    cross_validation_folds: int = 5
    
    # Retraining parameters
    retrain_frequency_hours: int = 24
    performance_threshold: float = 0.05  # Retrain if performance drops below this
    data_drift_threshold: float = 0.1
    
    # Resource limits
    max_training_time_minutes: int = 60
    max_memory_gb: float = 4.0


@dataclass
class TrainingResult:
    """Results from model training"""
    model_name: str
    training_config: TrainingConfig
    start_time: datetime
    end_time: datetime
    training_duration_seconds: float
    
    # Performance metrics
    train_performance: ModelPerformance
    validation_performance: ModelPerformance  
    test_performance: Optional[ModelPerformance] = None
    
    # Optimization results
    best_params: Optional[Dict[str, Any]] = None
    optimization_trials: Optional[List[Dict[str, Any]]] = None
    
    # Training metadata
    training_samples: int = 0
    validation_samples: int = 0
    test_samples: int = 0
    features_used: List[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'model_name': self.model_name,
            'start_time': self.start_time.isoformat(),
            'end_time': self.end_time.isoformat(),
            'training_duration_seconds': self.training_duration_seconds,
            'train_performance': self.train_performance.__dict__,
            'validation_performance': self.validation_performance.__dict__,
            'test_performance': self.test_performance.__dict__ if self.test_performance else None,
            'best_params': self.best_params,
            'training_samples': self.training_samples,
            'validation_samples': self.validation_samples,
            'test_samples': self.test_samples,
            'features_used': self.features_used
        }


class HyperparameterOptimizer:
    """Hyperparameter optimization using various methods"""
    
    def __init__(self, method: OptimizationMethod):
        self.method = method
        self.study = None
        
    def create_objective_function(self, model_class, X_train, y_train, X_val, y_val, 
                                prediction_type: PredictionType):
        """Create objective function for optimization"""
        
        def objective(trial):
            # Define hyperparameter search spaces based on model type
            if model_class == XGBoostModel:
                params = {
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                    'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                    'reg_alpha': trial.suggest_float('reg_alpha', 0, 1),
                    'reg_lambda': trial.suggest_float('reg_lambda', 0, 1)
                }
            elif model_class == LSTMModelWrapper:
                params = {
                    'hidden_size': trial.suggest_int('hidden_size', 64, 256),
                    'num_layers': trial.suggest_int('num_layers', 1, 4),
                    'sequence_length': trial.suggest_int('sequence_length', 10, 50)
                }
            else:
                # Default parameter space
                params = {}
            
            # Create and train model
            model = model_class(
                model_name=f"temp_model_{trial.number}",
                prediction_type=prediction_type,
                **params
            )
            
            # Create temporary feature sets for training
            temp_train_features = [
                FeatureSet(
                    symbol="TEMP", 
                    timestamp=time.time(),
                    features={f"f_{i}": Feature(f"f_{i}", val, FeatureType.TECHNICAL, time.time())
                             for i, val in enumerate(row)},
                    target=target
                ) for row, target in zip(X_train, y_train)
            ]
            
            # Train model
            try:
                asyncio.run(model.train(temp_train_features))
                
                # Evaluate on validation set
                if model.is_trained:
                    val_predictions = []
                    for row in X_val:
                        pred = asyncio.run(model.predict(row))
                        val_predictions.append(pred.value)
                    
                    # Calculate performance metric
                    if prediction_type == PredictionType.DIRECTION:
                        score = accuracy_score(y_val, np.array(val_predictions) > 0.5)
                    else:
                        score = -mean_squared_error(y_val, val_predictions)  # Negative MSE for maximization
                    
                    return score
                else:
                    return -1e6  # Large negative value for failed training
                    
            except Exception as e:
                logger.warning(f"Trial {trial.number} failed: {e}")
                return -1e6
        
        return objective
    
    async def optimize(self, model_class, X_train, y_train, X_val, y_val, 
                      prediction_type: PredictionType, n_trials: int = 100) -> Dict[str, Any]:
        """Perform hyperparameter optimization"""
        
        if self.method == OptimizationMethod.BAYESIAN:
            # Use Optuna for Bayesian optimization
            sampler = TPESampler(seed=42)
            pruner = MedianPruner()
            
            self.study = optuna.create_study(
                direction='maximize',
                sampler=sampler,
                pruner=pruner
            )
            
            objective = self.create_objective_function(
                model_class, X_train, y_train, X_val, y_val, prediction_type
            )
            
            self.study.optimize(objective, n_trials=n_trials, timeout=3600)
            
            return {
                'best_params': self.study.best_params,
                'best_value': self.study.best_value,
                'trials': [
                    {
                        'params': trial.params,
                        'value': trial.value,
                        'state': trial.state.name
                    } for trial in self.study.trials
                ]
            }
        
        elif self.method == OptimizationMethod.GRID_SEARCH:
            # Grid search implementation would go here
            # For now, return default parameters
            return {
                'best_params': {},
                'best_value': 0.0,
                'trials': []
            }
        
        elif self.method == OptimizationMethod.RANDOM_SEARCH:
            # Random search implementation would go here
            return {
                'best_params': {},
                'best_value': 0.0,
                'trials': []
            }
        
        else:
            raise ValueError(f"Unknown optimization method: {self.method}")


class ModelValidator:
    """Model validation and performance assessment"""
    
    @staticmethod
    def time_series_cross_validation(feature_sets: List[FeatureSet], n_splits: int = 5) -> List[Tuple[List[int], List[int]]]:
        """Time series cross-validation splits"""
        n_samples = len(feature_sets)
        test_size = n_samples // (n_splits + 1)
        
        splits = []
        for i in range(n_splits):
            train_end = n_samples - (n_splits - i) * test_size
            test_start = train_end
            test_end = test_start + test_size
            
            train_indices = list(range(train_end))
            test_indices = list(range(test_start, test_end))
            
            splits.append((train_indices, test_indices))
        
        return splits
    
    @staticmethod
    def calculate_performance_metrics(y_true: np.ndarray, y_pred: np.ndarray, 
                                    prediction_type: PredictionType) -> ModelPerformance:
        """Calculate comprehensive performance metrics"""
        performance = ModelPerformance("validation")
        
        if prediction_type == PredictionType.DIRECTION:
            # Classification metrics
            y_pred_binary = (y_pred > 0.5).astype(int)
            performance.accuracy = accuracy_score(y_true, y_pred_binary)
            performance.precision = precision_score(y_true, y_pred_binary, zero_division=0)
            performance.recall = recall_score(y_true, y_pred_binary, zero_division=0)
            performance.f1_score = f1_score(y_true, y_pred_binary, zero_division=0)
        
        elif prediction_type == PredictionType.RETURN:
            # Regression metrics
            performance.mse = mean_squared_error(y_true, y_pred)
            
            # Financial metrics
            cumulative_returns = np.cumprod(1 + y_pred)
            total_return = cumulative_returns[-1] - 1
            performance.total_return = total_return
            
            # Sharpe ratio (annualized)
            if np.std(y_pred) > 0:
                performance.sharpe_ratio = np.mean(y_pred) / np.std(y_pred) * np.sqrt(252)
            
            # Maximum drawdown
            peak = np.maximum.accumulate(cumulative_returns)
            drawdown = (cumulative_returns - peak) / peak
            performance.max_drawdown = np.min(drawdown)
        
        return performance
    
    async def validate_model(self, model, feature_sets: List[FeatureSet], 
                           config: TrainingConfig) -> Tuple[ModelPerformance, ModelPerformance]:
        """Validate model with train/validation split"""
        # Split data
        n_samples = len(feature_sets)
        train_size = int(n_samples * (1 - config.validation_split - config.test_split))
        val_size = int(n_samples * config.validation_split)
        
        train_features = feature_sets[:train_size]
        val_features = feature_sets[train_size:train_size + val_size]
        
        # Train model
        await model.train(train_features)
        
        # Get predictions
        train_predictions = []
        val_predictions = []
        
        for fs in train_features:
            pred = await model.predict(fs)
            train_predictions.append(pred.value)
        
        for fs in val_features:
            pred = await model.predict(fs)
            val_predictions.append(pred.value)
        
        # Calculate performance
        train_targets = [fs.target for fs in train_features]
        val_targets = [fs.target for fs in val_features]
        
        train_performance = self.calculate_performance_metrics(
            np.array(train_targets), np.array(train_predictions), model.prediction_type
        )
        
        val_performance = self.calculate_performance_metrics(
            np.array(val_targets), np.array(val_predictions), model.prediction_type
        )
        
        return train_performance, val_performance


class TrainingPipeline:
    """Complete model training and experimentation pipeline"""
    
    def __init__(self):
        self.settings = get_settings()
        self.feature_engine = get_feature_engine()
        self.model_manager = get_model_manager()
        self.storage_manager = get_storage_manager()
        
        self.training_results: List[TrainingResult] = []
        self.experiment_log_path = Path("experiments")
        self.experiment_log_path.mkdir(exist_ok=True)
    
    async def load_training_data(self, symbol: str, start_date: datetime, 
                               end_date: datetime) -> List[FeatureSet]:
        """Load training data with features and targets"""
        # This would load historical data and calculate features
        # For now, return empty list (implementation depends on data storage)
        logger.info(f"Loading training data for {symbol} from {start_date} to {end_date}")
        
        # Would load from storage manager
        # feature_sets = await self.storage_manager.load_feature_sets(symbol, start_date, end_date)
        
        return []  # Placeholder
    
    async def prepare_targets(self, feature_sets: List[FeatureSet], 
                            prediction_type: PredictionType, 
                            forecast_horizon: int = 1) -> List[FeatureSet]:
        """Prepare target variables for supervised learning"""
        if prediction_type == PredictionType.RETURN:
            # Calculate future returns
            for i in range(len(feature_sets) - forecast_horizon):
                current_price = feature_sets[i].features.get('mid_price', 0.0)
                future_price = feature_sets[i + forecast_horizon].features.get('mid_price', 0.0)
                
                if current_price and future_price:
                    return_value = (future_price - current_price) / current_price
                    feature_sets[i].target = return_value
        
        elif prediction_type == PredictionType.DIRECTION:
            # Calculate price direction
            for i in range(len(feature_sets) - forecast_horizon):
                current_price = feature_sets[i].features.get('mid_price', 0.0)
                future_price = feature_sets[i + forecast_horizon].features.get('mid_price', 0.0)
                
                if current_price and future_price:
                    direction = 1.0 if future_price > current_price else 0.0
                    feature_sets[i].target = direction
        
        # Remove samples without targets
        return [fs for fs in feature_sets[:-forecast_horizon] if fs.target is not None]
    
    async def train_model_with_optimization(self, model_class, model_name: str, 
                                          feature_sets: List[FeatureSet],
                                          config: TrainingConfig) -> TrainingResult:
        """Train model with hyperparameter optimization"""
        start_time = datetime.now()
        
        logger.info(f"Starting training for {model_name} with {len(feature_sets)} samples")
        
        # Prepare data
        if not feature_sets:
            raise ValueError("No feature sets provided for training")
        
        # Split data
        n_samples = len(feature_sets)
        train_size = int(n_samples * (1 - config.validation_split - config.test_split))
        val_size = int(n_samples * config.validation_split)
        
        train_features = feature_sets[:train_size]
        val_features = feature_sets[train_size:train_size + val_size]
        test_features = feature_sets[train_size + val_size:] if config.test_split > 0 else []
        
        # Extract feature arrays for optimization
        feature_names = list(train_features[0].features.keys())
        X_train = np.array([fs.to_array(feature_names) for fs in train_features])
        y_train = np.array([fs.target for fs in train_features])
        X_val = np.array([fs.to_array(feature_names) for fs in val_features])
        y_val = np.array([fs.target for fs in val_features])
        
        # Hyperparameter optimization
        optimizer = HyperparameterOptimizer(config.optimization_method)
        
        # Determine prediction type from first sample
        prediction_type = PredictionType.DIRECTION if all(t in [0.0, 1.0] for t in y_train) else PredictionType.RETURN
        
        optimization_result = await optimizer.optimize(
            model_class, X_train, y_train, X_val, y_val, 
            prediction_type, config.max_optimization_trials
        )
        
        # Train final model with best parameters
        final_model = model_class(
            model_name=model_name,
            prediction_type=prediction_type,
            **optimization_result['best_params']
        )
        
        # Validate model
        validator = ModelValidator()
        train_performance, val_performance = await validator.validate_model(
            final_model, train_features + val_features, config
        )
        
        # Test performance (if test data available)
        test_performance = None
        if test_features:
            test_predictions = []
            for fs in test_features:
                pred = await final_model.predict(fs)
                test_predictions.append(pred.value)
            
            test_targets = [fs.target for fs in test_features]
            test_performance = validator.calculate_performance_metrics(
                np.array(test_targets), np.array(test_predictions), prediction_type
            )
        
        # Register trained model
        self.model_manager.register_model(final_model)
        
        end_time = datetime.now()
        training_duration = (end_time - start_time).total_seconds()
        
        # Create training result
        result = TrainingResult(
            model_name=model_name,
            training_config=config,
            start_time=start_time,
            end_time=end_time,
            training_duration_seconds=training_duration,
            train_performance=train_performance,
            validation_performance=val_performance,
            test_performance=test_performance,
            best_params=optimization_result['best_params'],
            optimization_trials=optimization_result['trials'],
            training_samples=len(train_features),
            validation_samples=len(val_features),
            test_samples=len(test_features),
            features_used=feature_names
        )
        
        self.training_results.append(result)
        
        # Save experiment results
        self.save_experiment_result(result)
        
        logger.info(f"Completed training {model_name} in {training_duration:.2f}s")
        logger.info(f"Validation performance: {val_performance}")
        
        return result
    
    async def run_experiment(self, experiment_config: Dict[str, Any]) -> List[TrainingResult]:
        """Run complete training experiment"""
        logger.info(f"Starting experiment: {experiment_config.get('name', 'Unnamed')}")
        
        results = []
        
        # Load data
        symbol = experiment_config['symbol']
        start_date = datetime.fromisoformat(experiment_config['start_date'])
        end_date = datetime.fromisoformat(experiment_config['end_date'])
        
        feature_sets = await self.load_training_data(symbol, start_date, end_date)
        
        if not feature_sets:
            logger.warning("No training data available")
            return results
        
        # Prepare targets
        prediction_type = PredictionType(experiment_config['prediction_type'])
        feature_sets = await self.prepare_targets(
            feature_sets, prediction_type, 
            experiment_config.get('forecast_horizon', 1)
        )
        
        # Train each model configuration
        for model_config in experiment_config['models']:
            try:
                config = TrainingConfig(
                    model_name=model_config['name'],
                    training_mode=TrainingMode(model_config.get('training_mode', 'full_retrain')),
                    optimization_method=OptimizationMethod(model_config.get('optimization_method', 'bayesian')),
                    train_start_date=start_date,
                    train_end_date=end_date,
                    **model_config.get('training_params', {})
                )
                
                # Get model class
                if model_config['type'] == 'xgboost':
                    model_class = XGBoostModel
                elif model_config['type'] == 'lstm':
                    model_class = LSTMModelWrapper
                elif model_config['type'] == 'rl':
                    model_class = RLTradingAgent
                else:
                    logger.warning(f"Unknown model type: {model_config['type']}")
                    continue
                
                result = await self.train_model_with_optimization(
                    model_class, model_config['name'], feature_sets, config
                )
                results.append(result)
                
            except Exception as e:
                logger.error(f"Error training model {model_config['name']}: {e}")
        
        # Create ensemble if specified
        if experiment_config.get('create_ensemble', False):
            ensemble_models = [self.model_manager.models[result.model_name] 
                             for result in results if result.model_name in self.model_manager.models]
            
            if len(ensemble_models) > 1:
                ensemble = EnsembleModel(
                    model_name=f"{symbol}_ensemble",
                    models=ensemble_models
                )
                
                # Train ensemble
                await ensemble.train(feature_sets)
                self.model_manager.register_model(ensemble)
        
        logger.info(f"Completed experiment with {len(results)} models trained")
        return results
    
    def save_experiment_result(self, result: TrainingResult):
        """Save experiment result to disk"""
        experiment_file = self.experiment_log_path / f"{result.model_name}_{result.start_time.strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(experiment_file, 'w') as f:
            json.dump(result.to_dict(), f, indent=2)
    
    def load_experiment_results(self) -> List[TrainingResult]:
        """Load all experiment results"""
        results = []
        
        for experiment_file in self.experiment_log_path.glob("*.json"):
            try:
                with open(experiment_file, 'r') as f:
                    data = json.load(f)
                
                # Reconstruct TrainingResult (simplified)
                result = TrainingResult(
                    model_name=data['model_name'],
                    training_config=None,  # Would need to reconstruct
                    start_time=datetime.fromisoformat(data['start_time']),
                    end_time=datetime.fromisoformat(data['end_time']),
                    training_duration_seconds=data['training_duration_seconds'],
                    train_performance=ModelPerformance(**data['train_performance']),
                    validation_performance=ModelPerformance(**data['validation_performance']),
                    test_performance=ModelPerformance(**data['test_performance']) if data['test_performance'] else None
                )
                results.append(result)
                
            except Exception as e:
                logger.warning(f"Could not load experiment result {experiment_file}: {e}")
        
        return results
    
    async def auto_retrain_models(self):
        """Automatically retrain models based on performance degradation"""
        logger.info("Checking for models that need retraining...")
        
        for model_name, model in self.model_manager.models.items():
            if not model.is_trained:
                continue
            
            # Check if model performance has degraded
            # This would involve comparing recent predictions with actual outcomes
            # Implementation depends on performance monitoring system
            
            # For now, just log that we're checking
            logger.info(f"Checking retraining need for {model_name}")
            
            # Would implement actual performance checking logic here


# Global training pipeline instance
_training_pipeline: Optional[TrainingPipeline] = None


def get_training_pipeline() -> TrainingPipeline:
    """Get global training pipeline instance"""
    global _training_pipeline
    if _training_pipeline is None:
        _training_pipeline = TrainingPipeline()
    return _training_pipeline


# Example usage
async def main():
    """Example training pipeline usage"""
    import logging
    logging.basicConfig(level=logging.INFO)
    
    # Example experiment configuration
    experiment_config = {
        "name": "BTC_USDT_Experiment",
        "symbol": "BTC/USDT",
        "start_date": "2024-01-01T00:00:00",
        "end_date": "2024-12-31T23:59:59",
        "prediction_type": "return",
        "forecast_horizon": 1,
        "create_ensemble": True,
        "models": [
            {
                "name": "btc_xgb_v1",
                "type": "xgboost",
                "training_mode": "full_retrain",
                "optimization_method": "bayesian",
                "training_params": {
                    "validation_split": 0.2,
                    "max_optimization_trials": 50
                }
            },
            {
                "name": "btc_lstm_v1", 
                "type": "lstm",
                "training_mode": "full_retrain",
                "optimization_method": "bayesian",
                "training_params": {
                    "validation_split": 0.2,
                    "max_optimization_trials": 20
                }
            }
        ]
    }
    
    # Run experiment
    pipeline = get_training_pipeline()
    results = await pipeline.run_experiment(experiment_config)
    
    print(f"Training complete! Trained {len(results)} models.")
    for result in results:
        print(f"Model: {result.model_name}")
        print(f"  Validation Performance: {result.validation_performance}")
        print(f"  Training Duration: {result.training_duration_seconds:.2f}s")


if __name__ == "__main__":
    asyncio.run(main())
