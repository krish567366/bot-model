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

# Import the specific functions needed by Colab notebook
try:
    from .real_data_integration import MLDataIntegrator, get_real_training_data
except ImportError:
    pass

try:
    from .feature_engineering_v2 import compute_features_deterministic, load_feature_schema
except ImportError:
    pass

try:
    from .training_v2 import train_lightgbm_model
except ImportError:
    pass

try:
    from .registry import ModelRegistry
except ImportError:
    pass

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
    "MarketRegime",
    
    # Colab Notebook Functions
    "MLDataIntegrator",
    "get_real_training_data", 
    "compute_features_deterministic",
    "load_feature_schema",
    "train_lightgbm_model",
    "ModelRegistry"
]
