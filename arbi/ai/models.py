"""
Machine Learning Models Module

Implements various ML models for trading signal generation:
- XGBoost for tree-based learning
- LSTM for sequence modeling  
- Reinforcement Learning agents
- Ensemble methods
"""

import asyncio
import logging
import pickle
import numpy as np
import pandas as pd
from decimal import Decimal
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from datetime import datetime, time
import joblib
from .feature_engineering import FeatureSet, FeatureEngine, Feature, FeatureType
# ML Libraries
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, mean_squared_error
from sklearn.model_selection import train_test_split, cross_val_score
import torch
import logger
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Stable Baselines for RL
try:
    from stable_baselines3 import PPO, A2C, DQN
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.callbacks import BaseCallback
    import gym
    from gym import spaces
    RL_AVAILABLE = True
except ImportError:
    RL_AVAILABLE = False
    logger.warning("Reinforcement learning libraries not available. Install stable-baselines3 for RL support.")

from .feature_engineering import FeatureSet, FeatureEngine
from ..config.settings import get_settings

logger = logging.getLogger(__name__)


class ModelType(str, Enum):
    """Types of ML models"""
    XGBOOST = "xgboost"
    RANDOM_FOREST = "random_forest"
    LSTM = "lstm"
    TRANSFORMER = "transformer"
    PPO = "ppo"
    A2C = "a2c"
    DQN = "dqn"
    ENSEMBLE = "ensemble"


class PredictionType(str, Enum):
    """Types of predictions"""
    RETURN = "return"  # Predict future returns
    DIRECTION = "direction"  # Predict price direction (up/down)
    VOLATILITY = "volatility"  # Predict future volatility
    ACTION = "action"  # RL action (buy/sell/hold)


@dataclass
class ModelPrediction:
    """Model prediction result"""
    model_name: str
    prediction_type: PredictionType
    value: float
    confidence: float
    timestamp: float
    metadata: Dict[str, Any]


@dataclass 
class ModelPerformance:
    """Model performance metrics"""
    model_name: str
    accuracy: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1_score: Optional[float] = None
    mse: Optional[float] = None
    sharpe_ratio: Optional[float] = None
    max_drawdown: Optional[float] = None
    total_return: Optional[float] = None


class LSTMModel(nn.Module):
    """LSTM model for sequence prediction"""
    
    def __init__(self, input_size: int, hidden_size: int = 128, num_layers: int = 2, 
                 output_size: int = 1, dropout: float = 0.2):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                           batch_first=True, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        # Initialize hidden state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # LSTM forward pass
        out, _ = self.lstm(x, (h0, c0))
        
        # Take the last output
        out = self.dropout(out[:, -1, :])
        out = self.fc(out)
        return out


class TransformerModel(nn.Module):
    """Transformer model for sequence prediction"""
    
    def __init__(self, input_size: int, d_model: int = 128, nhead: int = 8, 
                 num_layers: int = 6, output_size: int = 1, dropout: float = 0.1):
        super(TransformerModel, self).__init__()
        self.d_model = d_model
        self.input_projection = nn.Linear(input_size, d_model)
        self.positional_encoding = self._generate_positional_encoding(1000, d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_projection = nn.Linear(d_model, output_size)
        
    def _generate_positional_encoding(self, max_len: int, d_model: int):
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           -(np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)
        
    def forward(self, x):
        seq_len = x.size(1)
        x = self.input_projection(x)
        x += self.positional_encoding[:, :seq_len, :].to(x.device)
        x = self.transformer(x)
        # Use the last token for prediction
        x = self.output_projection(x[:, -1, :])
        return x


class TradingEnvironment(gym.Env):
    """Custom trading environment for reinforcement learning"""
    
    def __init__(self, feature_data: np.ndarray, price_data: np.ndarray, 
                 initial_balance: float = 10000.0, transaction_cost: float = 0.001):
        super(TradingEnvironment, self).__init__()
        
        self.feature_data = feature_data
        self.price_data = price_data
        self.initial_balance = initial_balance
        self.transaction_cost = transaction_cost
        
        # Action space: 0=hold, 1=buy, 2=sell
        self.action_space = spaces.Discrete(3)
        
        # Observation space: features + current position + balance
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(feature_data.shape[1] + 2,), dtype=np.float32
        )
        
        self.reset()
    
    def reset(self):
        """Reset environment"""
        self.current_step = 0
        self.balance = self.initial_balance
        self.position = 0.0  # Number of shares held
        self.net_worth = self.initial_balance
        self.max_net_worth = self.initial_balance
        
        return self._get_observation()
    
    def _get_observation(self):
        """Get current observation"""
        if self.current_step >= len(self.feature_data):
            self.current_step = len(self.feature_data) - 1
            
        features = self.feature_data[self.current_step]
        position_value = self.position * self.price_data[self.current_step]
        
        obs = np.concatenate([
            features, 
            [self.position / 100.0],  # Normalized position
            [(self.balance + position_value) / self.initial_balance]  # Normalized net worth
        ])
        
        return obs.astype(np.float32)
    
    def step(self, action):
        """Take action and return new state"""
        current_price = self.price_data[self.current_step]
        
        # Calculate reward before action
        old_net_worth = self.balance + self.position * current_price
        
        # Execute action
        if action == 1:  # Buy
            if self.balance > current_price * (1 + self.transaction_cost):
                shares_to_buy = self.balance // (current_price * (1 + self.transaction_cost))
                cost = shares_to_buy * current_price * (1 + self.transaction_cost)
                self.position += shares_to_buy
                self.balance -= cost
                
        elif action == 2:  # Sell
            if self.position > 0:
                revenue = self.position * current_price * (1 - self.transaction_cost)
                self.balance += revenue
                self.position = 0
        
        # Move to next step
        self.current_step += 1
        
        # Calculate new net worth and reward
        if self.current_step < len(self.price_data):
            new_price = self.price_data[self.current_step]
            new_net_worth = self.balance + self.position * new_price
        else:
            new_net_worth = self.balance + self.position * current_price
            
        self.net_worth = new_net_worth
        self.max_net_worth = max(self.max_net_worth, new_net_worth)
        
        # Reward is the change in net worth
        reward = (new_net_worth - old_net_worth) / self.initial_balance
        
        # Check if episode is done
        done = self.current_step >= len(self.price_data) - 1
        
        # Additional penalties
        if new_net_worth < 0.1 * self.initial_balance:  # Bankruptcy
            reward -= 1.0
            done = True
        
        info = {
            'net_worth': new_net_worth,
            'balance': self.balance,
            'position': self.position,
            'price': current_price
        }
        
        return self._get_observation(), reward, done, info


class BaseMLModel:
    """Base class for ML models"""
    
    def __init__(self, model_name: str, model_type: ModelType, prediction_type: PredictionType):
        self.model_name = model_name
        self.model_type = model_type
        self.prediction_type = prediction_type
        self.model = None
        self.is_trained = False
        self.feature_names = []
        self.performance = ModelPerformance(model_name)
        
    def prepare_data(self, feature_sets: List[FeatureSet], target_column: str = None) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare data for training"""
        if not feature_sets:
            return np.array([]), np.array([])
        
        # Extract feature names from first feature set
        if not self.feature_names:
            self.feature_names = list(feature_sets[0].features.keys())
        
        # Create feature matrix
        X = np.array([fs.to_array(self.feature_names) for fs in feature_sets])
        
        # Create target array
        if self.prediction_type == PredictionType.RETURN:
            # Predict future returns
            y = np.array([fs.target if fs.target is not None else 0.0 for fs in feature_sets])
        elif self.prediction_type == PredictionType.DIRECTION:
            # Predict price direction (1 for up, 0 for down)
            y = np.array([1.0 if (fs.target if fs.target is not None else 0.0) > 0 else 0.0 
                         for fs in feature_sets])
        else:
            y = np.array([fs.target if fs.target is not None else 0.0 for fs in feature_sets])
        
        return X, y
    
    async def train(self, feature_sets: List[FeatureSet]) -> None:
        """Train the model"""
        raise NotImplementedError
    
    async def predict(self, features: Union[FeatureSet, np.ndarray]) -> ModelPrediction:
        """Make prediction"""
        raise NotImplementedError
    
    def save(self, path: Path) -> None:
        """Save model to disk"""
        if self.model is not None:
            joblib.dump({
                'model': self.model,
                'feature_names': self.feature_names,
                'model_type': self.model_type,
                'prediction_type': self.prediction_type,
                'performance': self.performance
            }, path)
    
    def load(self, path: Path) -> None:
        """Load model from disk"""
        data = joblib.load(path)
        self.model = data['model']
        self.feature_names = data['feature_names']
        self.performance = data.get('performance', ModelPerformance(self.model_name))
        self.is_trained = True


class XGBoostModel(BaseMLModel):
    """XGBoost model implementation"""
    
    def __init__(self, model_name: str, prediction_type: PredictionType = PredictionType.RETURN,
                 **xgb_params):
        super().__init__(model_name, ModelType.XGBOOST, prediction_type)
        
        default_params = {
            'objective': 'reg:squarederror' if prediction_type == PredictionType.RETURN else 'binary:logistic',
            'max_depth': 6,
            'learning_rate': 0.1,
            'n_estimators': 100,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42
        }
        default_params.update(xgb_params)
        self.xgb_params = default_params
    
    async def train(self, feature_sets: List[FeatureSet]) -> None:
        """Train XGBoost model"""
        X, y = self.prepare_data(feature_sets)
        
        if X.size == 0:
            logger.warning(f"No data available for training {self.model_name}")
            return
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train model
        if self.prediction_type == PredictionType.DIRECTION:
            self.model = xgb.XGBClassifier(**self.xgb_params)
        else:
            self.model = xgb.XGBRegressor(**self.xgb_params)
        
        self.model.fit(X_train, y_train)
        self.is_trained = True
        
        # Calculate performance metrics
        y_pred = self.model.predict(X_test)
        
        if self.prediction_type == PredictionType.DIRECTION:
            self.performance.accuracy = accuracy_score(y_test, y_pred > 0.5)
            self.performance.precision = precision_score(y_test, y_pred > 0.5, zero_division=0)
            self.performance.recall = recall_score(y_test, y_pred > 0.5, zero_division=0)
        else:
            self.performance.mse = mean_squared_error(y_test, y_pred)
        
        logger.info(f"Trained {self.model_name} with performance: {self.performance}")
    
    async def predict(self, features: Union[FeatureSet, np.ndarray]) -> ModelPrediction:
        """Make prediction with XGBoost"""
        if not self.is_trained or self.model is None:
            raise ValueError(f"Model {self.model_name} is not trained")
        
        # Prepare input
        if isinstance(features, FeatureSet):
            X = features.to_array(self.feature_names).reshape(1, -1)
        else:
            X = features.reshape(1, -1) if features.ndim == 1 else features
        
        # Make prediction
        prediction = self.model.predict(X)[0]
        
        # Get confidence (for tree-based models, use prediction probability range)
        if hasattr(self.model, 'predict_proba'):
            proba = self.model.predict_proba(X)[0]
            confidence = float(np.max(proba))
        else:
            confidence = 0.7  # Default confidence for regression
        
        return ModelPrediction(
            model_name=self.model_name,
            prediction_type=self.prediction_type,
            value=float(prediction),
            confidence=confidence,
            timestamp=time.time(),
            metadata={'feature_importance': dict(zip(self.feature_names, self.model.feature_importances_))}
        )


class LSTMModelWrapper(BaseMLModel):
    """LSTM model wrapper"""
    
    def __init__(self, model_name: str, prediction_type: PredictionType = PredictionType.RETURN,
                 sequence_length: int = 20, hidden_size: int = 128, num_layers: int = 2):
        super().__init__(model_name, ModelType.LSTM, prediction_type)
        self.sequence_length = sequence_length
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.scaler = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def prepare_sequences(self, X: np.ndarray, y: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        """Prepare sequences for LSTM"""
        sequences_X = []
        sequences_y = []
        
        for i in range(len(X) - self.sequence_length):
            sequences_X.append(X[i:i+self.sequence_length])
            sequences_y.append(y[i+self.sequence_length])
        
        return torch.FloatTensor(sequences_X), torch.FloatTensor(sequences_y)
    
    async def train(self, feature_sets: List[FeatureSet]) -> None:
        """Train LSTM model"""
        X, y = self.prepare_data(feature_sets)
        
        if len(X) < self.sequence_length * 2:
            logger.warning(f"Insufficient data for LSTM training {self.model_name}")
            return
        
        # Prepare sequences
        X_seq, y_seq = self.prepare_sequences(X, y)
        
        # Split data
        train_size = int(0.8 * len(X_seq))
        X_train, X_test = X_seq[:train_size], X_seq[train_size:]
        y_train, y_test = y_seq[:train_size], y_seq[train_size:]
        
        # Create data loaders
        train_dataset = TensorDataset(X_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        
        # Initialize model
        input_size = X.shape[1]
        self.model = LSTMModel(input_size, self.hidden_size, self.num_layers).to(self.device)
        
        # Training
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        
        self.model.train()
        for epoch in range(100):  # Number of epochs
            total_loss = 0
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                
                optimizer.zero_grad()
                output = self.model(batch_X).squeeze()
                loss = criterion(output, batch_y)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            if epoch % 20 == 0:
                logger.info(f"LSTM Epoch {epoch}, Loss: {total_loss/len(train_loader):.4f}")
        
        self.is_trained = True
        
        # Calculate performance
        self.model.eval()
        with torch.no_grad():
            X_test = X_test.to(self.device)
            y_pred = self.model(X_test).squeeze().cpu().numpy()
            y_test = y_test.numpy()
            self.performance.mse = mean_squared_error(y_test, y_pred)
        
        logger.info(f"Trained {self.model_name} with MSE: {self.performance.mse}")
    
    async def predict(self, features: Union[FeatureSet, np.ndarray]) -> ModelPrediction:
        """Make prediction with LSTM"""
        if not self.is_trained or self.model is None:
            raise ValueError(f"Model {self.model_name} is not trained")
        
        # This would need a sequence of features, simplified for now
        if isinstance(features, FeatureSet):
            X = features.to_array(self.feature_names)
        else:
            X = features
        
        # For now, repeat the current features to create a sequence
        X_seq = np.tile(X, (self.sequence_length, 1)).reshape(1, self.sequence_length, -1)
        X_tensor = torch.FloatTensor(X_seq).to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            prediction = self.model(X_tensor).squeeze().cpu().item()
        
        return ModelPrediction(
            model_name=self.model_name,
            prediction_type=self.prediction_type,
            value=float(prediction),
            confidence=0.8,  # Default confidence
            timestamp=time.time(),
            metadata={}
        )


class RLTradingAgent(BaseMLModel):
    """Reinforcement Learning trading agent"""
    
    def __init__(self, model_name: str, algorithm: str = "PPO"):
        super().__init__(model_name, getattr(ModelType, algorithm.upper()), PredictionType.ACTION)
        self.algorithm = algorithm
        self.env = None
        
        if not RL_AVAILABLE:
            logger.error("RL libraries not available. Please install stable-baselines3")
            return
    
    def create_environment(self, feature_data: np.ndarray, price_data: np.ndarray) -> TradingEnvironment:
        """Create trading environment"""
        return TradingEnvironment(feature_data, price_data)
    
    async def train(self, feature_sets: List[FeatureSet], price_data: List[float] = None) -> None:
        """Train RL agent"""
        if not RL_AVAILABLE:
            logger.error("Cannot train RL model without stable-baselines3")
            return
        
        X, y = self.prepare_data(feature_sets)
        
        if X.size == 0 or price_data is None:
            logger.warning(f"Insufficient data for RL training {self.model_name}")
            return
        
        # Create environment
        self.env = self.create_environment(X, np.array(price_data))
        
        # Create agent
        if self.algorithm == "PPO":
            self.model = PPO("MlpPolicy", self.env, verbose=1)
        elif self.algorithm == "A2C":
            self.model = A2C("MlpPolicy", self.env, verbose=1)
        elif self.algorithm == "DQN":
            self.model = DQN("MlpPolicy", self.env, verbose=1)
        else:
            raise ValueError(f"Unknown RL algorithm: {self.algorithm}")
        
        # Train agent
        logger.info(f"Training {self.algorithm} agent...")
        self.model.learn(total_timesteps=10000)
        self.is_trained = True
        
        logger.info(f"Trained RL agent {self.model_name}")
    
    async def predict(self, features: Union[FeatureSet, np.ndarray]) -> ModelPrediction:
        """Get action from RL agent"""
        if not self.is_trained or self.model is None:
            raise ValueError(f"RL agent {self.model_name} is not trained")
        
        # Convert features to environment observation format
        if isinstance(features, FeatureSet):
            feature_array = features.to_array(self.feature_names)
        else:
            feature_array = features
        
        # Add dummy position and balance for observation
        obs = np.concatenate([feature_array, [0.0, 1.0]])  # No position, normalized balance = 1
        
        # Get action
        action, _states = self.model.predict(obs, deterministic=True)
        
        # Convert action to trading signal
        # 0=hold, 1=buy, 2=sell -> convert to -1, 0, 1
        trading_signal = action - 1 if action > 0 else 0
        
        return ModelPrediction(
            model_name=self.model_name,
            prediction_type=self.prediction_type,
            value=float(trading_signal),
            confidence=0.8,  # Default confidence
            timestamp=time.time(),
            metadata={'raw_action': int(action)}
        )


class EnsembleModel(BaseMLModel):
    """Ensemble of multiple models"""
    
    def __init__(self, model_name: str, models: List[BaseMLModel], weights: Optional[List[float]] = None):
        super().__init__(model_name, ModelType.ENSEMBLE, PredictionType.RETURN)
        self.models = models
        self.weights = weights or [1.0] * len(models)
        self.weights = np.array(self.weights) / np.sum(self.weights)  # Normalize
    
    async def train(self, feature_sets: List[FeatureSet]) -> None:
        """Train all models in ensemble"""
        logger.info(f"Training ensemble {self.model_name} with {len(self.models)} models")
        
        for model in self.models:
            try:
                await model.train(feature_sets)
            except Exception as e:
                logger.error(f"Error training model {model.model_name}: {e}")
        
        self.is_trained = True
        logger.info(f"Trained ensemble {self.model_name}")
    
    async def predict(self, features: Union[FeatureSet, np.ndarray]) -> ModelPrediction:
        """Make ensemble prediction"""
        if not self.is_trained:
            raise ValueError(f"Ensemble {self.model_name} is not trained")
        
        predictions = []
        confidences = []
        
        for model in self.models:
            try:
                if model.is_trained:
                    pred = await model.predict(features)
                    predictions.append(pred.value)
                    confidences.append(pred.confidence)
                else:
                    logger.warning(f"Model {model.model_name} not trained, skipping")
            except Exception as e:
                logger.error(f"Error getting prediction from {model.model_name}: {e}")
        
        if not predictions:
            raise ValueError("No valid predictions from ensemble models")
        
        # Weighted average prediction
        weighted_prediction = np.average(predictions, weights=self.weights[:len(predictions)])
        avg_confidence = np.mean(confidences)
        
        return ModelPrediction(
            model_name=self.model_name,
            prediction_type=self.prediction_type,
            value=float(weighted_prediction),
            confidence=float(avg_confidence),
            timestamp=time.time(),
            metadata={
                'individual_predictions': predictions,
                'individual_confidences': confidences,
                'weights': self.weights[:len(predictions)].tolist()
            }
        )


class ModelManager:
    """Manages multiple ML models"""
    
    def __init__(self):
        self.settings = get_settings()
        self.models: Dict[str, BaseMLModel] = {}
        self.model_dir = Path("models")
        self.model_dir.mkdir(exist_ok=True)
        
    def register_model(self, model: BaseMLModel):
        """Register a model"""
        self.models[model.model_name] = model
        logger.info(f"Registered model: {model.model_name}")
    
    async def train_all_models(self, feature_sets: List[FeatureSet]):
        """Train all registered models"""
        logger.info(f"Training {len(self.models)} models...")
        
        for model_name, model in self.models.items():
            try:
                logger.info(f"Training {model_name}...")
                await model.train(feature_sets)
                
                # Save trained model
                model_path = self.model_dir / f"{model_name}.pkl"
                model.save(model_path)
                logger.info(f"Saved model {model_name} to {model_path}")
                
            except Exception as e:
                logger.error(f"Error training model {model_name}: {e}")
    
    async def get_predictions(self, features: Union[FeatureSet, np.ndarray]) -> Dict[str, ModelPrediction]:
        """Get predictions from all trained models"""
        predictions = {}
        
        for model_name, model in self.models.items():
            try:
                if model.is_trained:
                    pred = await model.predict(features)
                    predictions[model_name] = pred
            except Exception as e:
                logger.error(f"Error getting prediction from {model_name}: {e}")
        
        return predictions
    
    def load_models(self):
        """Load all saved models"""
        for model_file in self.model_dir.glob("*.pkl"):
            try:
                model_name = model_file.stem
                if model_name in self.models:
                    self.models[model_name].load(model_file)
                    logger.info(f"Loaded model: {model_name}")
            except Exception as e:
                logger.error(f"Error loading model {model_file}: {e}")
    
    def get_model_performances(self) -> Dict[str, ModelPerformance]:
        """Get performance metrics for all models"""
        return {name: model.performance for name, model in self.models.items()}
    
    def get_latest_model(self, model_type: str = None) -> Optional[BaseMLModel]:
        """Get the latest trained model, optionally filtered by type"""
        trained_models = {name: model for name, model in self.models.items() 
                         if model.is_trained}
        
        if not trained_models:
            return None
            
        if model_type:
            filtered_models = {name: model for name, model in trained_models.items() 
                             if model_type.lower() in name.lower()}
            if filtered_models:
                # Return the first matching model
                return next(iter(filtered_models.values()))
        
        # Return the first available trained model
        return next(iter(trained_models.values()))
    
    def get_model(self, model_name: str) -> Optional[BaseMLModel]:
        """Get a specific model by name"""
        return self.models.get(model_name)
    
    def has_trained_models(self) -> bool:
        """Check if any models are trained"""
        return any(model.is_trained for model in self.models.values())


# Global model manager instance
_model_manager: Optional[ModelManager] = None


def get_model_manager() -> ModelManager:
    """Get global model manager instance"""
    global _model_manager
    if _model_manager is None:
        _model_manager = ModelManager()
    return _model_manager


# Example usage
async def main():
    """Example model usage"""
    import time
    import logging
    logging.basicConfig(level=logging.INFO)
    
    # Create sample data
    sample_features = []
    for i in range(1000):
        features = {
            f'feature_{j}': np.random.randn() for j in range(20)
        }
        feature_set = FeatureSet(
            symbol="BTC/USDT",
            timestamp=time.time(),
            features={name: Feature(name, value, FeatureType.TECHNICAL, time.time()) 
                     for name, value in features.items()},
            target=np.random.randn() * 0.01  # Random returns
        )
        sample_features.append(feature_set)
    
    # Create models
    manager = get_model_manager()
    
    # XGBoost model
    xgb_model = XGBoostModel("xgb_returns", PredictionType.RETURN)
    manager.register_model(xgb_model)
    
    # Direction classifier
    xgb_classifier = XGBoostModel("xgb_direction", PredictionType.DIRECTION)
    manager.register_model(xgb_classifier)
    
    # LSTM model
    lstm_model = LSTMModelWrapper("lstm_returns", PredictionType.RETURN)
    manager.register_model(lstm_model)
    
    # Train models
    await manager.train_all_models(sample_features)
    
    # Get predictions
    test_features = sample_features[-1]
    predictions = await manager.get_predictions(test_features)
    
    print("Model predictions:")
    for model_name, prediction in predictions.items():
        print(f"{model_name}: {prediction.value:.4f} (confidence: {prediction.confidence:.2f})")
    
    print("Model training complete!")


if __name__ == "__main__":
    asyncio.run(main())
