#!/usr/bin/env python3
"""
Data Pipeline Demo Script

Demonstrates the comprehensive market data pipeline with:
- Multi-source data ingestion (yfinance, Alpha Vantage, CCXT)
- Data cleaning and validation
- Feature engineering with technical indicators
- Database storage and retrieval
- ML-ready data preparation
- Integration with signal generation and AI/ML models
"""

import asyncio
import logging
import sys
import warnings
from pathlib import Path

# Add the project root to the path
sys.path.append(str(Path(__file__).parent))

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

from arbi.core.pipeline import DataPipeline
from arbi.config.settings import get_settings


async def demo_basic_pipeline():
    """Demonstrate basic data pipeline functionality"""
    print("=" * 60)
    print("🔄 BASIC DATA PIPELINE DEMO")
    print("=" * 60)
    
    # Initialize pipeline
    pipeline = DataPipeline({
        'database_path': 'demo_data/market_data.db',
        # 'alpha_vantage_api_key': 'your_api_key_here'  # Add your key for Alpha Vantage
    })
    
    # Demo 1: Fetch historical data
    print("\n📈 1. Fetching Historical Data")
    print("-" * 40)
    
    symbol = "AAPL"
    print(f"Fetching {symbol} data from yfinance...")
    
    df_raw = await pipeline.fetch_historical(
        symbol=symbol,
        start="2023-01-01",
        end="2024-01-01",
        interval="1d"
    )
    
    if not df_raw.empty:
        print(f"✅ Fetched {len(df_raw)} rows of raw data")
        print(f"   Columns: {list(df_raw.columns)[:8]}...")  # First 8 columns
        print(f"   Date range: {df_raw['date'].min()} to {df_raw['date'].max()}")
    else:
        print("❌ No data fetched")
        return
    
    # Demo 2: Clean data
    print("\n🧹 2. Data Cleaning & Validation")
    print("-" * 40)
    
    df_clean = pipeline.clean_data(df_raw.copy())
    print(f"✅ Cleaned data: {len(df_clean)} rows (removed {len(df_raw) - len(df_clean)} rows)")
    
    # Demo 3: Feature engineering
    print("\n⚙️ 3. Feature Engineering")
    print("-" * 40)
    
    df_features = pipeline.transform_features(df_clean.copy())
    print(f"✅ Added features: {len(df_features.columns)} total columns")
    
    # Show some key features
    feature_cols = [col for col in df_features.columns if any(x in col.lower() 
                   for x in ['rsi', 'sma', 'ema', 'macd', 'bb', 'volatility'])]
    if feature_cols:
        print(f"   Key indicators: {feature_cols[:5]}...")  # First 5 indicators
    
    # Demo 4: Save to database
    print("\n💾 4. Database Storage")
    print("-" * 40)
    
    table_name = f"{symbol}_1d_demo"
    if pipeline.save(df_features, table_name):
        print(f"✅ Saved to database table: {table_name}")
        
        # Get table info
        info = pipeline.get_table_info(table_name)
        print(f"   Stored {info.get('row_count', 0)} rows with {len(info.get('columns', []))} columns")
    
    return df_features


async def demo_multiple_symbols():
    """Demonstrate processing multiple symbols"""
    print("\n" + "=" * 60)
    print("📊 MULTIPLE SYMBOLS PIPELINE DEMO")
    print("=" * 60)
    
    pipeline = DataPipeline({'database_path': 'demo_data/market_data.db'})
    
    # Process multiple symbols
    symbols = ["AAPL", "GOOGL", "MSFT"]
    print(f"\n📈 Processing symbols: {symbols}")
    print("-" * 40)
    
    results = await pipeline.update_multiple_symbols(
        symbols=symbols,
        start="2023-06-01",
        end="2023-12-31",
        intervals=["1d"]
    )
    
    print("\n✅ Results Summary:")
    for symbol, intervals in results.items():
        for interval, df in intervals.items():
            if not df.empty:
                print(f"   {symbol} ({interval}): {len(df)} rows, {len(df.columns)} features")
            else:
                print(f"   {symbol} ({interval}): No data")


async def demo_crypto_data():
    """Demonstrate cryptocurrency data fetching"""
    print("\n" + "=" * 60)
    print("₿ CRYPTOCURRENCY DATA DEMO")
    print("=" * 60)
    
    pipeline = DataPipeline({'database_path': 'demo_data/market_data.db'})
    
    # Try crypto symbols (Yahoo Finance format)
    crypto_symbols = ["BTC-USD", "ETH-USD"]
    
    print(f"\n₿ Processing crypto symbols: {crypto_symbols}")
    print("-" * 40)
    
    for symbol in crypto_symbols:
        print(f"\nFetching {symbol}...")
        df = await pipeline.fetch_historical(
            symbol=symbol,
            start="2023-01-01",
            end="2023-07-01",
            interval="1d"
        )
        
        if not df.empty:
            # Clean and add features
            df_clean = pipeline.clean_data(df)
            df_features = pipeline.transform_features(df_clean)
            
            # Save
            table_name = f"{symbol.replace('-', '_')}_crypto_demo"
            pipeline.save(df_features, table_name)
            
            print(f"✅ {symbol}: {len(df_features)} rows with {len(df_features.columns)} features")
            
            # Show price info
            if 'close' in df_features.columns:
                latest_price = df_features['close'].iloc[-1]
                price_change = ((df_features['close'].iloc[-1] / df_features['close'].iloc[0]) - 1) * 100
                print(f"   Latest price: ${latest_price:.2f} (Change: {price_change:+.1f}%)")
        else:
            print(f"❌ No data for {symbol}")


async def demo_ml_preparation():
    """Demonstrate ML-ready data preparation"""
    print("\n" + "=" * 60)
    print("🤖 ML DATA PREPARATION DEMO")
    print("=" * 60)
    
    pipeline = DataPipeline({'database_path': 'demo_data/market_data.db'})
    
    # Load existing data and prepare for ML
    table_name = "AAPL_1d_demo"
    print(f"\n🔍 Loading data from table: {table_name}")
    
    # Check if table exists by trying to load data
    df_stored = pipeline.load(table_name)
    
    if df_stored.empty:
        print("❌ No stored data found. Run basic demo first.")
        return
    
    print(f"✅ Loaded {len(df_stored)} rows from database")
    
    # Prepare ML-ready data
    print("\n🤖 Preparing ML sequences...")
    X, y = pipeline.get_ml_ready_data(table_name, target_col="close", lookback_window=30)
    
    if X.size > 0:
        print(f"✅ ML data prepared:")
        print(f"   Features shape: {X.shape}")
        print(f"   Targets shape: {y.shape}")
        print(f"   Feature window: {X.shape[1]} timesteps")
        print(f"   Features per timestep: {X.shape[2]}")
        
        # Show sample statistics
        print(f"\n📊 Target statistics (closing prices):")
        print(f"   Min: ${y.min():.2f}")
        print(f"   Max: ${y.max():.2f}")
        print(f"   Mean: ${y.mean():.2f}")
        print(f"   Std: ${y.std():.2f}")
    else:
        print("❌ Could not prepare ML data")


async def demo_data_resampling():
    """Demonstrate data resampling to different intervals"""
    print("\n" + "=" * 60)
    print("🔄 DATA RESAMPLING DEMO")
    print("=" * 60)
    
    pipeline = DataPipeline({'database_path': 'demo_data/market_data.db'})
    
    # Load daily data
    df_daily = pipeline.load("AAPL_1d_demo")
    
    if df_daily.empty:
        print("❌ No daily data found. Run basic demo first.")
        return
    
    print(f"\n📈 Original daily data: {len(df_daily)} rows")
    
    # Resample to different intervals
    intervals = ["5D", "W", "M"]  # 5-day, weekly, monthly
    
    for interval in intervals:
        print(f"\n🔄 Resampling to {interval}...")
        df_resampled = pipeline.resample(df_daily.copy(), interval)
        
        if not df_resampled.empty:
            print(f"✅ {interval}: {len(df_resampled)} rows")
            
            # Save resampled data
            table_name = f"AAPL_{interval}_demo"
            pipeline.save(df_resampled, table_name)
            print(f"   Saved to table: {table_name}")
        else:
            print(f"❌ Failed to resample to {interval}")


async def demo_live_data():
    """Demonstrate live data fetching"""
    print("\n" + "=" * 60)
    print("⚡ LIVE DATA DEMO")
    print("=" * 60)
    
    pipeline = DataPipeline({'database_path': 'demo_data/market_data.db'})
    
    symbols = ["AAPL", "MSFT", "GOOGL"]
    
    print(f"\n📡 Fetching live data for: {symbols}")
    print("-" * 40)
    
    for symbol in symbols:
        print(f"\n🔍 {symbol} live data:")
        live_data = await pipeline.fetch_live(symbol)
        
        if live_data:
            print(f"   Current Price: ${live_data.get('current_price', 'N/A')}")
            print(f"   Bid/Ask: ${live_data.get('bid', 'N/A')} / ${live_data.get('ask', 'N/A')}")
            print(f"   Volume: {live_data.get('volume', 'N/A'):,}")
            print(f"   Market Cap: ${live_data.get('market_cap', 'N/A'):,}" if live_data.get('market_cap') else "   Market Cap: N/A")
            print(f"   Updated: {live_data.get('timestamp', 'N/A')}")
        else:
            print(f"   ❌ No live data available")


async def demo_integration_showcase():
    """Demonstrate integration with other system components"""
    print("\n" + "=" * 60)
    print("🔗 SYSTEM INTEGRATION SHOWCASE")
    print("=" * 60)
    
    pipeline = DataPipeline({'database_path': 'demo_data/market_data.db'})
    
    # Show how pipeline data can be used by other components
    print("\n🎯 Data Pipeline Integration Points:")
    print("-" * 40)
    
    print("✅ Signal Generation:")
    print("   - OHLCV data → Cross-exchange arbitrage detection")
    print("   - Technical indicators → Signal strength calculation")
    print("   - Volume data → Liquidity assessment")
    
    print("\n✅ AI/ML Models:")
    print("   - Sequence data → LSTM/Transformer training")
    print("   - Technical features → XGBoost features")
    print("   - Market data → Regime detection")
    
    print("\n✅ Risk Management:")
    print("   - Volatility indicators → Position sizing")
    print("   - Price history → VaR calculation")
    print("   - Volume data → Slippage estimation")
    
    print("\n✅ Backtesting:")
    print("   - Historical OHLCV → Strategy simulation")
    print("   - Feature data → ML model backtesting")
    print("   - Multiple intervals → Multi-timeframe analysis")
    
    print("\n✅ Portfolio Management:")
    print("   - Price data → Portfolio valuation")
    print("   - Returns data → Performance metrics")
    print("   - Correlation data → Diversification analysis")
    
    # Show sample data that would be used
    print(f"\n📊 Available Data Tables:")
    print("-" * 30)
    
    # List stored tables (would be implemented with proper DB inspection)
    sample_tables = [
        "AAPL_1d_demo", "GOOGL_1d_demo", "MSFT_1d_demo",
        "BTC_USD_crypto_demo", "ETH_USD_crypto_demo"
    ]
    
    for table in sample_tables:
        info = pipeline.get_table_info(table)
        if info.get('row_count', 0) > 0:
            print(f"   📈 {table}: {info['row_count']} rows")


async def main():
    """Main demo function"""
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('data_pipeline_demo.log'),
            logging.StreamHandler()
        ]
    )
    
    # Suppress verbose logs during demo
    logging.getLogger('yfinance').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    
    print("🚀 DATA PIPELINE COMPREHENSIVE DEMO")
    print("=" * 60)
    print("This demo showcases the complete market data pipeline")
    print("with real data sources and comprehensive processing.")
    print("=" * 60)
    
    # Create demo data directory
    Path("demo_data").mkdir(exist_ok=True)
    
    try:
        # Run all demos
        await demo_basic_pipeline()
        await demo_multiple_symbols()
        await demo_crypto_data()
        await demo_ml_preparation()
        await demo_data_resampling()
        await demo_live_data()
        await demo_integration_showcase()
        
        print("\n" + "=" * 60)
        print("✅ ALL DEMOS COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print("\n📁 Demo files created in './demo_data/' directory")
        print("📜 Demo log saved as 'data_pipeline_demo.log'")
        print("\n🔗 The data pipeline is now ready to integrate with:")
        print("   • Signal generation (signal.py)")
        print("   • AI/ML models (ai/ module)")
        print("   • Risk management (risk.py)")
        print("   • Backtesting engine (backtest.py)")
        print("   • Portfolio management (portfolio.py)")
        
    except Exception as e:
        print(f"\n❌ Demo error: {e}")
        logging.error(f"Demo failed: {e}", exc_info=True)


if __name__ == "__main__":
    print("Starting Data Pipeline Demo...")
    asyncio.run(main())
