#!/usr/bin/env python3
"""
Quick Setup Script for Trading Bot Production Deployment

This script will:
1. Train a model with sample data
2. Test the inference engine 
3. Run a sample backtest
4. Show you how to deploy to production
"""

import sys
import os
import asyncio
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add project root to path
sys.path.insert(0, os.path.abspath('.'))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s',
    datefmt='%d/%b/%Y %H:%M:%S'
)
logger = logging.getLogger(__name__)

async def main():
    """Main setup workflow"""
    print("🚀 Trading Bot Production Setup")
    print("=" * 50)
    
    try:
        # Step 1: Initialize system
        print("\n📋 Step 1: Initialize System Components")
        await initialize_system()
        
        # Step 2: Train model with sample data
        print("\n🎯 Step 2: Train Sample Model")
        model_id = await train_sample_model()
        
        # Step 3: Test inference engine
        print("\n🧠 Step 3: Test Inference Engine")
        await test_inference()
        
        # Step 4: Run sample backtest
        print("\n📊 Step 4: Run Sample Backtest")
        await run_sample_backtest()
        
        # Step 5: Production deployment instructions
        print("\n🚀 Step 5: Production Deployment")
        show_production_instructions(model_id)
        
        print("\n✅ Setup completed successfully!")
        print("📚 See PRODUCTION_DEPLOYMENT_GUIDE.md for detailed instructions")
        
    except Exception as e:
        logger.error(f"Setup failed: {e}")
        print(f"\n❌ Setup failed: {e}")
        print("Please check the error message and try again")

async def initialize_system():
    """Initialize storage and registry"""
    from arbi.core.storage import StorageManager
    from arbi.ai.registry import ModelRegistry
    
    # Initialize storage
    storage = StorageManager()
    await storage.initialize()
    logger.info("✅ Storage system initialized")
    
    # Initialize model registry
    registry = ModelRegistry()
    logger.info("✅ Model registry initialized")
    
    print("   ✅ Storage system ready")
    print("   ✅ Model registry ready")

async def train_sample_model():
    """Train a sample model for testing"""
    from arbi.ai.training_v2 import train_lightgbm_model, TrainingConfig
    
    # Create sample training data
    print("   📈 Generating sample OHLCV data...")
    df = create_sample_data(periods=1000)
    
    # Add target (5-period forward return)
    df['future_return'] = df['close'].pct_change(5).shift(-5)
    df = df.dropna()
    
    print(f"   📊 Training data: {len(df)} samples")
    
    # Configure training
    config = TrainingConfig(
        model_type='lightgbm',
        validation_size=0.2,
        test_size=0.2,
        min_validation_score=0.01,  # Low threshold for sample data
        tune_hyperparameters=False,  # Faster training
        verbose=True
    )
    
    # Train model
    print("   🤖 Training LightGBM model...")
    model_id = train_lightgbm_model(df, 'future_return', 'SAMPLE', config)
    
    print(f"   ✅ Model trained: {model_id}")
    return model_id

async def test_inference():
    """Test the inference engine"""
    from arbi.ai.inference_v2 import test_inference_engine
    
    print("   🔍 Testing inference engine...")
    success = await test_inference_engine()
    
    if success:
        print("   ✅ Inference engine working correctly")
        print("   📊 ML signals generated successfully")
    else:
        print("   ❌ Inference engine test failed")
        raise Exception("Inference test failed")

async def run_sample_backtest():
    """Run a simple backtest simulation"""
    from arbi.ai.inference_v2 import ProductionInferenceEngine
    
    print("   📈 Running sample backtest...")
    
    # Initialize inference engine
    engine = ProductionInferenceEngine()
    await engine.initialize()
    
    # Generate sample market data
    market_data = create_sample_data(periods=200)
    
    # Generate signals
    signals = await engine.generate_ml_signals(
        symbol="SAMPLE/USDT",
        exchange="binance", 
        market_data=market_data
    )
    
    # Simple backtest simulation
    portfolio_value = 10000  # $10,000 starting capital
    position = 0
    trades = []
    
    for signal in signals:
        if signal.confidence > 0.01:  # Min confidence
            # Simple trading logic
            if signal.side == 'BUY' and position <= 0:
                position = portfolio_value * 0.1  # 10% position
                entry_price = market_data.iloc[-1]['close']
                trades.append({
                    'action': 'BUY',
                    'size': position,
                    'price': entry_price,
                    'confidence': signal.confidence
                })
            elif signal.side == 'SELL' and position > 0:
                exit_price = market_data.iloc[-1]['close']
                pnl = position * (exit_price / entry_price - 1)
                portfolio_value += pnl
                trades.append({
                    'action': 'SELL', 
                    'pnl': pnl,
                    'price': exit_price
                })
                position = 0
    
    # Calculate results
    total_return = (portfolio_value - 10000) / 10000 * 100
    num_trades = len([t for t in trades if t['action'] == 'SELL'])
    
    print(f"   📊 Backtest Results:")
    print(f"      Signals Generated: {len(signals)}")
    print(f"      Trades Executed: {num_trades}")
    print(f"      Total Return: {total_return:.2f}%")
    print(f"      Final Portfolio: ${portfolio_value:.2f}")

def show_production_instructions(model_id):
    """Show production deployment instructions"""
    print("   🏭 Production Deployment Instructions:")
    print(f"      Your trained model: {model_id}")
    print("   ")
    print("   📝 Next Steps:")
    print("   1. Replace sample data with real market data")
    print("   2. Adjust confidence thresholds based on your risk tolerance")
    print("   3. Deploy inference service: python inference_service.py")
    print("   4. Set up monitoring: python monitor.py")
    print("   5. Configure exchange connections for live trading")
    print("   ")
    print("   📚 Key Files:")
    print("   - PRODUCTION_DEPLOYMENT_GUIDE.md: Complete deployment guide")
    print("   - inference_service.py: Production inference service")
    print("   - monitor.py: System health monitoring")
    print("   - retrain_scheduler.py: Automated model retraining")

def create_sample_data(periods=1000, symbol="SAMPLE"):
    """Create sample OHLCV data for testing"""
    dates = pd.date_range(end=datetime.now(), periods=periods, freq='1min')
    
    # Generate realistic price movement
    np.random.seed(42)  # Reproducible
    returns = np.random.normal(0, 0.002, periods)  # 0.2% volatility
    
    # Add some trend and mean reversion
    trend = np.linspace(0, 0.1, periods)  # 10% upward trend
    mean_reversion = -0.1 * (np.cumsum(returns) - np.mean(np.cumsum(returns)))
    
    returns = returns + trend/periods + mean_reversion/periods
    prices = 50000 * np.exp(np.cumsum(returns))
    
    data = []
    for i, (date, price) in enumerate(zip(dates, prices)):
        high = price * (1 + abs(np.random.normal(0, 0.005)))
        low = price * (1 - abs(np.random.normal(0, 0.005)))
        open_price = prices[i-1] if i > 0 else price
        volume = np.random.lognormal(7, 0.5)  # Realistic volume distribution
        
        data.append({
            'timestamp': date,
            'open': open_price,
            'high': high,
            'low': low,
            'close': price,
            'volume': volume
        })
    
    return pd.DataFrame(data)

if __name__ == "__main__":
    asyncio.run(main())
