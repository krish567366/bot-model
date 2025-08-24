"""
AI/ML Inference Engine

Real-time inference engine that generates ML-based trading signals
and integrates with the existing signal generation system.
"""

import asyncio
import logging
import time
import numpy as np
from decimal import Decimal
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
from enum import Enum
from collections import deque

from .feature_engineering import FeatureEngine, FeatureSet, get_feature_engine
from .models import ModelManager, get_model_manager, ModelPrediction, PredictionType
from ..core.marketdata import BookDelta, OrderBook, Trade, Ticker
from ..core.signal import ArbitrageSignal, SignalType, SignalStrength
from ..config.settings import get_settings

logger = logging.getLogger(__name__)


class SignalSource(str, Enum):
    """Source of trading signals"""
    ML_MODEL = "ml_model"
    ENSEMBLE = "ensemble"
    HYBRID = "hybrid"  # Combination of ML and rule-based


class MarketRegime(str, Enum):
    """Market regime classification"""
    TRENDING = "trending"
    RANGING = "ranging"
    VOLATILE = "volatile"
    LOW_VOLUME = "low_volume"
    HIGH_VOLUME = "high_volume"


@dataclass
class MLSignal:
    """Machine learning generated signal"""
    signal_id: str
    model_name: str
    signal_source: SignalSource
    timestamp: float
    symbol: str
    
    # ML predictions
    return_prediction: Optional[float] = None
    direction_prediction: Optional[float] = None
    volatility_prediction: Optional[float] = None
    action_prediction: Optional[int] = None
    
    # Confidence and metadata
    confidence: float = 0.0
    model_confidence: float = 0.0
    ensemble_agreement: Optional[float] = None
    
    # Market regime
    detected_regime: Optional[MarketRegime] = None
    regime_confidence: float = 0.0
    
    # Feature importance
    top_features: Dict[str, float] = None
    
    # Integration with traditional signals
    rule_based_agreement: Optional[float] = None
    combined_strength: Optional[SignalStrength] = None


class MarketRegimeDetector:
    """Detect current market regime for strategy selection"""
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.price_history: Dict[str, deque] = {}
        self.volume_history: Dict[str, deque] = {}
        self.volatility_history: Dict[str, deque] = {}
        
    def update(self, symbol: str, price: float, volume: float, volatility: float):
        """Update market data for regime detection"""
        if symbol not in self.price_history:
            self.price_history[symbol] = deque(maxlen=self.window_size)
            self.volume_history[symbol] = deque(maxlen=self.window_size)
            self.volatility_history[symbol] = deque(maxlen=self.window_size)
        
        self.price_history[symbol].append(price)
        self.volume_history[symbol].append(volume)
        self.volatility_history[symbol].append(volatility)
    
    def detect_regime(self, symbol: str) -> Tuple[MarketRegime, float]:
        """Detect current market regime"""
        if symbol not in self.price_history or len(self.price_history[symbol]) < 20:
            return MarketRegime.RANGING, 0.5
        
        prices = np.array(list(self.price_history[symbol]))
        volumes = np.array(list(self.volume_history[symbol]))
        volatilities = np.array(list(self.volatility_history[symbol]))
        
        # Calculate trend strength
        if len(prices) >= 20:
            short_ma = np.mean(prices[-10:])
            long_ma = np.mean(prices[-20:])
            trend_strength = abs(short_ma - long_ma) / long_ma
        else:
            trend_strength = 0.0
        
        # Calculate volatility level
        current_vol = np.mean(volatilities[-5:]) if len(volatilities) >= 5 else 0.0
        historical_vol = np.mean(volatilities) if len(volatilities) > 0 else 0.0
        vol_ratio = current_vol / historical_vol if historical_vol > 0 else 1.0
        
        # Calculate volume level
        current_volume = np.mean(volumes[-5:]) if len(volumes) >= 5 else 0.0
        historical_volume = np.mean(volumes) if len(volumes) > 0 else 0.0
        volume_ratio = current_volume / historical_volume if historical_volume > 0 else 1.0
        
        # Regime classification
        confidence = 0.7  # Default confidence
        
        if vol_ratio > 1.5:
            return MarketRegime.VOLATILE, min(vol_ratio / 2.0, 1.0)
        elif trend_strength > 0.02:
            return MarketRegime.TRENDING, min(trend_strength * 20, 1.0)
        elif volume_ratio < 0.5:
            return MarketRegime.LOW_VOLUME, min(1.0 / volume_ratio, 1.0)
        elif volume_ratio > 2.0:
            return MarketRegime.HIGH_VOLUME, min(volume_ratio / 2.0, 1.0)
        else:
            return MarketRegime.RANGING, confidence


class StrategySelector:
    """Select optimal strategies based on market regime"""
    
    def __init__(self):
        self.strategy_performance: Dict[str, Dict[MarketRegime, float]] = {}
        self.regime_strategy_mapping = {
            MarketRegime.TRENDING: ['lstm_returns', 'momentum_ensemble'],
            MarketRegime.RANGING: ['xgb_direction', 'mean_reversion_ensemble'],
            MarketRegime.VOLATILE: ['volatility_model', 'risk_parity_ensemble'],
            MarketRegime.LOW_VOLUME: ['patient_execution', 'large_spread_capture'],
            MarketRegime.HIGH_VOLUME: ['fast_execution', 'momentum_capture']
        }
    
    def get_optimal_strategies(self, regime: MarketRegime, regime_confidence: float) -> List[str]:
        """Get optimal strategies for current market regime"""
        base_strategies = self.regime_strategy_mapping.get(regime, ['default_ensemble'])
        
        # Adjust strategy selection based on confidence
        if regime_confidence < 0.6:
            # Low confidence in regime detection, use conservative strategies
            return ['conservative_ensemble']
        elif regime_confidence > 0.8:
            # High confidence, use specialized strategies
            return base_strategies
        else:
            # Medium confidence, blend strategies
            return base_strategies + ['adaptive_ensemble']
    
    def update_strategy_performance(self, strategy_name: str, regime: MarketRegime, 
                                  performance: float):
        """Update strategy performance tracking"""
        if strategy_name not in self.strategy_performance:
            self.strategy_performance[strategy_name] = {}
        
        self.strategy_performance[strategy_name][regime] = performance


class InferenceEngine:
    """Main inference engine for ML-based signal generation"""
    
    def __init__(self):
        self.settings = get_settings()
        self.feature_engine = get_feature_engine()
        self.model_manager = get_model_manager()
        
        self.regime_detector = MarketRegimeDetector()
        self.strategy_selector = StrategySelector()
        
        # Signal history for ensemble agreement
        self.signal_history: Dict[str, deque] = {}
        self.prediction_cache: Dict[str, Dict[str, ModelPrediction]] = {}
        
        # Performance monitoring
        self.model_performance_tracker: Dict[str, deque] = {}
        
        # Load models on initialization
        self.model_manager.load_models()
    
    async def generate_ml_signals(self, exchange: str, symbol: str, 
                                order_book: OrderBook, trades: Optional[List[Trade]] = None,
                                ticker: Optional[Ticker] = None) -> List[MLSignal]:
        """Generate ML-based trading signals"""
        signals = []
        current_time = time.time()
        
        try:
            # Update feature engine with new market data
            await self.feature_engine.update_market_data(exchange, symbol, order_book, trades, ticker)
            
            # Get latest features
            latest_features = await self.feature_engine.get_latest_features(symbol)
            if not latest_features:
                logger.warning(f"No features available for {symbol}")
                return signals
            
            # Update market regime detection
            mid_price = order_book.get_mid_price()
            if mid_price:
                volume = sum(float(level.size) for level in order_book.get_levels("bid", 5) + order_book.get_levels("ask", 5))
                volatility = latest_features.features.get('realized_vol', 0.0)
                self.regime_detector.update(symbol, float(mid_price), volume, volatility)
            
            # Detect current market regime
            regime, regime_confidence = self.regime_detector.detect_regime(symbol)
            
            # Get optimal strategies for current regime
            optimal_strategies = self.strategy_selector.get_optimal_strategies(regime, regime_confidence)
            
            # Generate predictions from all available models
            model_predictions = await self.model_manager.get_predictions(latest_features)
            
            # Filter predictions based on optimal strategies
            relevant_predictions = {
                name: pred for name, pred in model_predictions.items() 
                if any(strategy in name for strategy in optimal_strategies) or name in optimal_strategies
            }
            
            if not relevant_predictions:
                # If no strategy-specific models, use all available models
                relevant_predictions = model_predictions
            
            # Create individual ML signals
            for model_name, prediction in relevant_predictions.items():
                signal = await self._create_ml_signal(
                    model_name, prediction, symbol, current_time, regime, regime_confidence, latest_features
                )
                if signal:
                    signals.append(signal)
            
            # Create ensemble signal if multiple models available
            if len(relevant_predictions) > 1:
                ensemble_signal = await self._create_ensemble_signal(
                    relevant_predictions, symbol, current_time, regime, regime_confidence, latest_features
                )
                if ensemble_signal:
                    signals.append(ensemble_signal)
        
        except Exception as e:
            logger.error(f"Error generating ML signals for {symbol}: {e}")
        
        return signals
    
    async def _create_ml_signal(self, model_name: str, prediction: ModelPrediction,
                              symbol: str, timestamp: float, regime: MarketRegime,
                              regime_confidence: float, features: FeatureSet) -> Optional[MLSignal]:
        """Create ML signal from model prediction"""
        
        # Calculate signal strength based on prediction confidence and magnitude
        if prediction.prediction_type == PredictionType.RETURN:
            signal_strength = min(abs(prediction.value) * 100, 1.0)  # Scale return to 0-1
        elif prediction.prediction_type == PredictionType.DIRECTION:
            signal_strength = abs(prediction.value - 0.5) * 2  # Scale probability to 0-1
        else:
            signal_strength = prediction.confidence
        
        # Only generate signal if strength is above threshold
        if signal_strength < 0.1:
            return None
        
        # Get feature importance from prediction metadata
        top_features = prediction.metadata.get('feature_importance', {})
        if top_features:
            # Sort by importance and take top 5
            sorted_features = sorted(top_features.items(), key=lambda x: abs(x[1]), reverse=True)
            top_features = dict(sorted_features[:5])
        
        signal_id = f"ml_{model_name}_{symbol}_{int(timestamp)}"
        
        signal = MLSignal(
            signal_id=signal_id,
            model_name=model_name,
            signal_source=SignalSource.ML_MODEL,
            timestamp=timestamp,
            symbol=symbol,
            confidence=signal_strength,
            model_confidence=prediction.confidence,
            detected_regime=regime,
            regime_confidence=regime_confidence,
            top_features=top_features
        )
        
        # Set prediction values based on type
        if prediction.prediction_type == PredictionType.RETURN:
            signal.return_prediction = prediction.value
        elif prediction.prediction_type == PredictionType.DIRECTION:
            signal.direction_prediction = prediction.value
        elif prediction.prediction_type == PredictionType.ACTION:
            signal.action_prediction = int(prediction.value)
        
        return signal
    
    async def _create_ensemble_signal(self, predictions: Dict[str, ModelPrediction],
                                    symbol: str, timestamp: float, regime: MarketRegime,
                                    regime_confidence: float, features: FeatureSet) -> Optional[MLSignal]:
        """Create ensemble signal from multiple model predictions"""
        
        # Separate predictions by type
        return_predictions = []
        direction_predictions = []
        action_predictions = []
        confidences = []
        
        for pred in predictions.values():
            confidences.append(pred.confidence)
            
            if pred.prediction_type == PredictionType.RETURN:
                return_predictions.append(pred.value)
            elif pred.prediction_type == PredictionType.DIRECTION:
                direction_predictions.append(pred.value)
            elif pred.prediction_type == PredictionType.ACTION:
                action_predictions.append(pred.value)
        
        if not confidences:
            return None
        
        # Calculate ensemble agreement
        ensemble_agreement = self._calculate_ensemble_agreement(predictions)
        
        # Weighted ensemble predictions
        weights = np.array(confidences) / np.sum(confidences)
        
        ensemble_return = None
        ensemble_direction = None
        ensemble_action = None
        
        if return_predictions:
            ensemble_return = np.average(return_predictions, weights=weights[:len(return_predictions)])
        
        if direction_predictions:
            ensemble_direction = np.average(direction_predictions, weights=weights[:len(direction_predictions)])
        
        if action_predictions:
            # For actions, use majority vote weighted by confidence
            action_votes = {}
            for i, action in enumerate(action_predictions):
                action_votes[action] = action_votes.get(action, 0) + confidences[i]
            ensemble_action = max(action_votes, key=action_votes.get)
        
        signal_id = f"ml_ensemble_{symbol}_{int(timestamp)}"
        
        signal = MLSignal(
            signal_id=signal_id,
            model_name="ensemble",
            signal_source=SignalSource.ENSEMBLE,
            timestamp=timestamp,
            symbol=symbol,
            return_prediction=ensemble_return,
            direction_prediction=ensemble_direction,
            action_prediction=ensemble_action,
            confidence=np.mean(confidences),
            model_confidence=np.mean(confidences),
            ensemble_agreement=ensemble_agreement,
            detected_regime=regime,
            regime_confidence=regime_confidence
        )
        
        return signal
    
    def _calculate_ensemble_agreement(self, predictions: Dict[str, ModelPrediction]) -> float:
        """Calculate agreement between ensemble models"""
        if len(predictions) < 2:
            return 1.0
        
        # Group predictions by type
        pred_groups = {}
        for pred in predictions.values():
            pred_type = pred.prediction_type
            if pred_type not in pred_groups:
                pred_groups[pred_type] = []
            pred_groups[pred_type].append(pred.value)
        
        agreements = []
        for pred_type, values in pred_groups.items():
            if len(values) < 2:
                continue
            
            if pred_type == PredictionType.DIRECTION:
                # For binary classification, calculate agreement percentage
                avg_pred = np.mean(values)
                agreements.append(1.0 - 2 * abs(avg_pred - 0.5))  # Higher agreement when closer to 0 or 1
            else:
                # For continuous predictions, use coefficient of variation
                if np.mean(values) != 0:
                    cv = np.std(values) / abs(np.mean(values))
                    agreements.append(max(0.0, 1.0 - cv))  # Lower CV = higher agreement
                else:
                    agreements.append(1.0)
        
        return np.mean(agreements) if agreements else 0.5
    
    async def integrate_with_traditional_signals(self, ml_signals: List[MLSignal], 
                                               traditional_signals: List[ArbitrageSignal]) -> List[MLSignal]:
        """Integrate ML signals with traditional arbitrage signals"""
        
        integrated_signals = []
        
        for ml_signal in ml_signals:
            # Find matching traditional signals for the same symbol
            matching_traditional = [
                sig for sig in traditional_signals 
                if sig.symbol == ml_signal.symbol and 
                abs(sig.timestamp - ml_signal.timestamp) < 10  # Within 10 seconds
            ]
            
            if matching_traditional:
                # Calculate agreement with traditional signals
                rule_agreement = self._calculate_rule_based_agreement(ml_signal, matching_traditional)
                ml_signal.rule_based_agreement = rule_agreement
                
                # Combine strengths
                combined_strength = self._combine_signal_strengths(ml_signal, matching_traditional)
                ml_signal.combined_strength = combined_strength
            
            integrated_signals.append(ml_signal)
        
        return integrated_signals
    
    def _calculate_rule_based_agreement(self, ml_signal: MLSignal, 
                                      traditional_signals: List[ArbitrageSignal]) -> float:
        """Calculate agreement between ML and rule-based signals"""
        if not traditional_signals:
            return 0.5
        
        agreements = []
        
        for trad_signal in traditional_signals:
            # Compare signal directions
            ml_direction = None
            if ml_signal.direction_prediction is not None:
                ml_direction = 1 if ml_signal.direction_prediction > 0.5 else -1
            elif ml_signal.return_prediction is not None:
                ml_direction = 1 if ml_signal.return_prediction > 0 else -1
            elif ml_signal.action_prediction is not None:
                ml_direction = ml_signal.action_prediction - 1  # Convert 0,1,2 to -1,0,1
            
            # Traditional signal direction (simplified)
            trad_direction = 1 if trad_signal.expected_profit_pct > 0 else -1
            
            if ml_direction is not None:
                agreement = 1.0 if ml_direction * trad_direction > 0 else 0.0
                agreements.append(agreement)
        
        return np.mean(agreements) if agreements else 0.5
    
    def _combine_signal_strengths(self, ml_signal: MLSignal, 
                                traditional_signals: List[ArbitrageSignal]) -> SignalStrength:
        """Combine ML and traditional signal strengths"""
        
        # Get average traditional signal strength
        trad_strengths = []
        for sig in traditional_signals:
            if sig.strength == SignalStrength.CRITICAL:
                trad_strengths.append(1.0)
            elif sig.strength == SignalStrength.STRONG:
                trad_strengths.append(0.75)
            elif sig.strength == SignalStrength.MODERATE:
                trad_strengths.append(0.5)
            else:
                trad_strengths.append(0.25)
        
        avg_trad_strength = np.mean(trad_strengths) if trad_strengths else 0.5
        
        # Combine with ML confidence
        combined_strength = (ml_signal.confidence + avg_trad_strength) / 2
        
        # Apply agreement boost/penalty
        if ml_signal.rule_based_agreement is not None:
            if ml_signal.rule_based_agreement > 0.7:
                combined_strength *= 1.2  # Boost for high agreement
            elif ml_signal.rule_based_agreement < 0.3:
                combined_strength *= 0.8  # Penalty for disagreement
        
        # Convert to SignalStrength enum
        if combined_strength >= 0.8:
            return SignalStrength.CRITICAL
        elif combined_strength >= 0.6:
            return SignalStrength.STRONG
        elif combined_strength >= 0.4:
            return SignalStrength.MODERATE
        else:
            return SignalStrength.WEAK
    
    async def update_model_performance(self, signal_id: str, actual_outcome: float):
        """Update model performance tracking with actual outcomes"""
        # This would be called when we have actual trade outcomes
        # to track and improve model performance over time
        
        if signal_id not in self.model_performance_tracker:
            self.model_performance_tracker[signal_id] = deque(maxlen=100)
        
        self.model_performance_tracker[signal_id].append(actual_outcome)
        
        # Update strategy selector with performance data
        # Implementation would depend on signal tracking system


# Global inference engine instance
_inference_engine: Optional[InferenceEngine] = None


def get_inference_engine() -> InferenceEngine:
    """Get global inference engine instance"""
    global _inference_engine
    if _inference_engine is None:
        _inference_engine = InferenceEngine()
    return _inference_engine


# Example usage
async def main():
    """Example inference engine usage"""
    import logging
    from ..core.marketdata import OrderBookLevel
    
    logging.basicConfig(level=logging.INFO)
    
    # Create sample order book
    order_book = OrderBook(
        exchange="test",
        symbol="BTC/USDT",
        timestamp=time.time()
    )
    
    # Add some sample levels
    order_book.bids["50000"] = OrderBookLevel(price=Decimal("50000"), size=Decimal("1.0"))
    order_book.asks["50100"] = OrderBookLevel(price=Decimal("50100"), size=Decimal("0.5"))
    
    # Get inference engine
    engine = get_inference_engine()
    
    # Generate ML signals
    ml_signals = await engine.generate_ml_signals("test", "BTC/USDT", order_book)
    
    print(f"Generated {len(ml_signals)} ML signals")
    for signal in ml_signals:
        print(f"Signal: {signal.signal_id}")
        print(f"  Model: {signal.model_name}")
        print(f"  Confidence: {signal.confidence:.3f}")
        print(f"  Regime: {signal.detected_regime}")


if __name__ == "__main__":
    asyncio.run(main())
