#!/usr/bin/env python3
"""
Quick Data Pipeline Test

Simple test to verify the data pipeline is working correctly.
"""

import asyncio
import sys
from pathlib import Path
sys.path.append('.')

async def simple_test():
    print("🧪 Data Pipeline Quick Test")
    print("=" * 40)
    
    from arbi.core.pipeline import YFinanceSource
    
    # Test YFinance
    print("\n📈 Testing YFinance data fetch...")
    yf_source = YFinanceSource()
    df = await yf_source.fetch_historical('AAPL', period='1mo', interval='1d')
    
    if not df.empty:
        print(f"✅ Success! Fetched {len(df)} rows")
        print(f"   Columns: {list(df.columns)}")
        print(f"   Data shape: {df.shape}")
        
        # Show sample data
        print("\n📊 Sample data:")
        print(df.head(3).to_string())
        
        print(f"\n💰 Price info:")
        if 'close' in df.columns:
            latest = df['close'].iloc[-1]
            print(f"   Latest close: ${latest:.2f}")
        
        print("\n✅ Data pipeline is working correctly!")
        return True
    else:
        print("❌ No data received")
        return False

if __name__ == "__main__":
    success = asyncio.run(simple_test())
    if success:
        print("\n🎉 Data pipeline ready for production!")
    else:
        print("\n❌ Pipeline needs troubleshooting")
