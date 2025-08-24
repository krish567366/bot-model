#!/usr/bin/env python3
"""
Production Inference Service

Continuous ML signal generation service for production trading.
Runs inference at regular intervals and stores signals to the database.
"""

import asyncio
import logging
import signal
import sys
from datetime import datetime, timedelta
from typing import List, Dict, Any
import json

# Add project root to path
sys.path.insert(0, '.')

from arbi.ai.inference_v2 import ProductionInferenceEngine
from arbi.core.storage import StorageManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s',
    datefmt='%d/%b/%Y %H:%M:%S',
    handlers=[
        logging.FileHandler('inference_service.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class ProductionInferenceService:
    """Production ML inference service for continuous signal generation"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or self._default_config()
        self.engine = None
        self.storage = None
        self.running = False
        self.stats = {
            'start_time': None,
            'signals_generated': 0,
            'errors': 0,
            'last_signal_time': None
        }
        
    def _default_config(self) -> Dict[str, Any]:
        """Default service configuration"""
        return {
            'symbols': ['BTC/USDT', 'ETH/USDT', 'BNB/USDT'],
            'exchanges': ['binance'],
            'inference_interval': 60,  # seconds
            'min_signal_confidence': 0.05,
            'max_signals_per_symbol': 10,  # per interval
            'retry_on_error': True,
            'retry_delay': 30,  # seconds
            'health_check_interval': 300,  # 5 minutes
        }
    
    async def initialize(self):
        """Initialize inference service components"""
        logger.info("🚀 Initializing Production Inference Service")
        
        try:
            # Initialize storage
            self.storage = StorageManager()
            await self.storage.initialize()
            logger.info("✅ Storage system initialized")
            
            # Initialize inference engine
            self.engine = ProductionInferenceEngine()
            await self.engine.initialize()
            logger.info("✅ Inference engine initialized")
            
            # Log model info
            model_info = self.engine.get_model_info()
            logger.info(f"📊 Loaded model: {model_info.get('model_id', 'Unknown')}")
            logger.info(f"   Validation Score: {model_info.get('validation_score', 0):.4f}")
            logger.info(f"   Feature Count: {model_info.get('feature_count', 0)}")
            
            # Log configuration
            logger.info("⚙️ Service Configuration:")
            logger.info(f"   Symbols: {self.config['symbols']}")
            logger.info(f"   Inference Interval: {self.config['inference_interval']}s")
            logger.info(f"   Min Confidence: {self.config['min_signal_confidence']}")
            
            self.stats['start_time'] = datetime.now()
            logger.info("✅ Production Inference Service ready")
            
        except Exception as e:
            logger.error(f"❌ Service initialization failed: {e}")
            raise
    
    async def start(self):
        """Start the inference service"""
        logger.info("🔄 Starting inference service loop")
        self.running = True
        
        # Setup graceful shutdown
        def signal_handler(sig, frame):
            logger.info("🛑 Shutdown signal received")
            self.running = False
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        # Start main inference loop
        await asyncio.gather(
            self._inference_loop(),
            self._health_check_loop(),
            return_exceptions=True
        )
    
    async def _inference_loop(self):
        """Main inference loop"""
        while self.running:
            loop_start = datetime.now()
            
            try:
                # Generate signals for all symbols
                total_signals = 0
                
                for symbol in self.config['symbols']:
                    for exchange in self.config['exchanges']:
                        try:
                            signals = await self._generate_signals_for_symbol(
                                symbol, exchange
                            )
                            
                            if signals:
                                # Store signals
                                stored_count = await self.engine.populate_storage_signals(
                                    signals, self.storage
                                )
                                total_signals += stored_count
                                
                                logger.info(f"📊 {symbol} on {exchange}: "
                                          f"{stored_count} signals stored")
                        
                        except Exception as e:
                            logger.error(f"❌ Error processing {symbol} on {exchange}: {e}")
                            self.stats['errors'] += 1
                            
                            if not self.config['retry_on_error']:
                                continue
                
                # Update stats
                if total_signals > 0:
                    self.stats['signals_generated'] += total_signals
                    self.stats['last_signal_time'] = datetime.now()
                    logger.info(f"✅ Generated {total_signals} total signals")
                
                # Calculate sleep time
                loop_duration = (datetime.now() - loop_start).total_seconds()
                sleep_time = max(0, self.config['inference_interval'] - loop_duration)
                
                if sleep_time > 0:
                    logger.debug(f"💤 Sleeping for {sleep_time:.1f}s")
                    await asyncio.sleep(sleep_time)
                else:
                    logger.warning(f"⚠️ Inference loop took {loop_duration:.1f}s "
                                 f"(longer than {self.config['inference_interval']}s interval)")
            
            except Exception as e:
                logger.error(f"❌ Inference loop error: {e}")
                self.stats['errors'] += 1
                
                if self.config['retry_on_error']:
                    await asyncio.sleep(self.config['retry_delay'])
                else:
                    self.running = False
        
        logger.info("🛑 Inference loop stopped")
    
    async def _generate_signals_for_symbol(self, symbol: str, exchange: str) -> List:
        """Generate ML signals for a specific symbol/exchange"""
        try:
            # Generate signals using inference engine
            signals = await self.engine.generate_ml_signals(
                symbol=symbol,
                exchange=exchange
            )
            
            # Filter by confidence
            filtered_signals = [
                s for s in signals 
                if s.confidence >= self.config['min_signal_confidence']
            ]
            
            # Limit number of signals per interval
            if len(filtered_signals) > self.config['max_signals_per_symbol']:
                # Keep highest confidence signals
                filtered_signals.sort(key=lambda x: x.confidence, reverse=True)
                filtered_signals = filtered_signals[:self.config['max_signals_per_symbol']]
                
                logger.warning(f"⚠️ Limited {symbol} signals to {self.config['max_signals_per_symbol']}")
            
            return filtered_signals
        
        except Exception as e:
            logger.error(f"❌ Signal generation failed for {symbol}: {e}")
            return []
    
    async def _health_check_loop(self):
        """Periodic health check loop"""
        while self.running:
            await asyncio.sleep(self.config['health_check_interval'])
            await self._perform_health_check()
    
    async def _perform_health_check(self):
        """Perform system health check"""
        try:
            # Check service uptime
            uptime = datetime.now() - self.stats['start_time']
            
            # Check model status
            model_info = self.engine.get_model_info()
            model_status = model_info.get('status', 'unknown')
            
            # Check recent signal generation
            time_since_last_signal = None
            if self.stats['last_signal_time']:
                time_since_last_signal = datetime.now() - self.stats['last_signal_time']
            
            logger.info("💊 Health Check:")
            logger.info(f"   Uptime: {uptime}")
            logger.info(f"   Model Status: {model_status}")
            logger.info(f"   Total Signals: {self.stats['signals_generated']}")
            logger.info(f"   Total Errors: {self.stats['errors']}")
            if time_since_last_signal:
                logger.info(f"   Last Signal: {time_since_last_signal} ago")
            
            # Warning conditions
            if time_since_last_signal and time_since_last_signal > timedelta(hours=1):
                logger.warning("⚠️ No signals generated in the last hour")
            
            if self.stats['errors'] > 10:
                logger.warning(f"⚠️ High error count: {self.stats['errors']}")
        
        except Exception as e:
            logger.error(f"❌ Health check failed: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get service statistics"""
        uptime = datetime.now() - self.stats['start_time'] if self.stats['start_time'] else timedelta(0)
        
        return {
            **self.stats,
            'uptime': str(uptime),
            'running': self.running,
            'config': self.config
        }


async def main():
    """Main entry point"""
    logger.info("🚀 Starting Production Inference Service")
    
    # Load configuration (you can extend this to load from file)
    config = {
        'symbols': ['BTC/USDT'],  # Start with one symbol
        'exchanges': ['binance'],
        'inference_interval': 60,  # 1 minute
        'min_signal_confidence': 0.01,  # Low threshold for demo
        'max_signals_per_symbol': 5,
    }
    
    try:
        # Create and start service
        service = ProductionInferenceService(config)
        await service.initialize()
        await service.start()
    
    except KeyboardInterrupt:
        logger.info("🛑 Service stopped by user")
    except Exception as e:
        logger.error(f"❌ Service failed: {e}")
        sys.exit(1)
    
    logger.info("👋 Production Inference Service shutdown complete")


if __name__ == "__main__":
    asyncio.run(main())
