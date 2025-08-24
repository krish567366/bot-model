"""
AI/ML Module for Arbitrage Trading

Provides machine learning capabilities including:
- Feature engineering from market data
- Model training and hyperparameter optimization  
- Real-time inference and signal generation
- Market regime detection and strategy selection
"""

from .feature_engineering import FeatureEngine, FeatureSet, Feature, get_feature_engine
from .models import (ModelManager, get_model_manager, XGBoostModel, LSTMModelWrapper,
                    RLTradingAgent, EnsembleModel, ModelPrediction)
from .training import TrainingPipeline, get_training_pipeline, TrainingConfig, TrainingResult
from .inference import InferenceEngine, get_inference_engine, MLSignal, MarketRegime

__all__ = [
    # Feature Engineering
    "FeatureEngine",
    "FeatureSet", 
    "Feature",
    "get_feature_engine",
    
    # Models
    "ModelManager",
    "get_model_manager",
    "XGBoostModel",
    "LSTMModelWrapper", 
    "RLTradingAgent",
    "EnsembleModel",
    "ModelPrediction",
    
    # Training
    "TrainingPipeline",
    "get_training_pipeline",
    "TrainingConfig",
    "TrainingResult",
    
    # Inference
    "InferenceEngine",
    "get_inference_engine", 
    "MLSignal",
    "MarketRegime"
]
