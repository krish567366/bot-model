"""
Production configuration settings for trading system.

Hierarchical configuration with environment-based overrides.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime
import yaml


@dataclass
class DatabaseConfig:
    """Database configuration"""
    host: str = "localhost"
    port: int = 5432
    database: str = "trading_bot"
    username: str = "trader"
    password: str = ""
    connection_pool_size: int = 10
    connection_timeout: int = 30


@dataclass 
class ExchangeConfig:
    """Exchange API configuration"""
    name: str = "binance"
    api_key: str = ""
    api_secret: str = ""
    sandbox: bool = True
    rate_limit_requests_per_minute: int = 1200
    timeout: int = 30


@dataclass
class AIConfig:
    """AI/ML configuration"""
    model_type: str = "xgboost"
    retrain_schedule: str = "weekly"  # daily, weekly, monthly
    feature_lookback_days: int = 30
    prediction_horizon_hours: int = 24
    min_training_samples: int = 1000
    cross_validation_folds: int = 5
    hyperparameter_optimization: bool = True
    model_registry_path: str = "models/"


@dataclass
class RiskConfig:
    """Risk management configuration"""
    max_position_size_pct: float = 0.1  # 10% max position
    max_daily_loss_pct: float = 0.05    # 5% daily loss limit
    max_drawdown_pct: float = 0.2       # 20% max drawdown
    stop_loss_pct: float = 0.02         # 2% stop loss
    take_profit_pct: float = 0.04       # 4% take profit
    risk_free_rate: float = 0.02        # 2% annual risk-free rate


@dataclass
class MonitoringConfig:
    """Monitoring and alerting configuration"""
    enable_monitoring: bool = True
    data_drift_threshold: float = 0.1
    performance_degradation_threshold: float = 0.15
    alert_channels: List[str] = None
    health_check_interval_minutes: int = 5
    log_level: str = "INFO"
    
    def __post_init__(self):
        if self.alert_channels is None:
            self.alert_channels = ["email", "slack"]


@dataclass
class BacktestConfig:
    """Backtesting configuration"""
    initial_capital: float = 10000.0
    commission_rate: float = 0.001      # 0.1% commission
    slippage_bps: int = 5               # 5 basis points slippage
    lookback_days: int = 252            # 1 year
    walk_forward_days: int = 30         # Monthly walk-forward
    benchmark_symbol: str = "BTC-USD"


@dataclass
class Settings:
    """Main settings configuration"""
    
    # Core settings
    environment: str = "development"
    debug: bool = False
    timezone: str = "UTC"
    data_path: str = "data/"
    logs_path: str = "logs/"
    
    # Component configurations
    database: DatabaseConfig = None
    exchange: ExchangeConfig = None
    ai: AIConfig = None
    risk: RiskConfig = None
    monitoring: MonitoringConfig = None
    backtest: BacktestConfig = None
    
    # Trading parameters
    trading_symbols: List[str] = None
    trading_intervals: List[str] = None
    
    def __post_init__(self):
        # Initialize nested configs with defaults
        if self.database is None:
            self.database = DatabaseConfig()
        if self.exchange is None:
            self.exchange = ExchangeConfig()
        if self.ai is None:
            self.ai = AIConfig()
        if self.risk is None:
            self.risk = RiskConfig()
        if self.monitoring is None:
            self.monitoring = MonitoringConfig()
        if self.backtest is None:
            self.backtest = BacktestConfig()
        
        # Default trading parameters
        if self.trading_symbols is None:
            self.trading_symbols = ["BTC-USD", "ETH-USD"]
        if self.trading_intervals is None:
            self.trading_intervals = ["1m", "5m", "1h", "1d"]
        
        # Create directories
        Path(self.data_path).mkdir(exist_ok=True)
        Path(self.logs_path).mkdir(exist_ok=True)
        Path("var/locks").mkdir(parents=True, exist_ok=True)
        Path("models").mkdir(exist_ok=True)
    
    def dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'environment': self.environment,
            'debug': self.debug,
            'timezone': self.timezone,
            'data_path': self.data_path,
            'logs_path': self.logs_path,
            'database': self.database.__dict__,
            'exchange': self.exchange.__dict__,
            'ai': self.ai.__dict__,
            'risk': self.risk.__dict__,
            'monitoring': self.monitoring.__dict__,
            'backtest': self.backtest.__dict__,
            'trading_symbols': self.trading_symbols,
            'trading_intervals': self.trading_intervals
        }


def load_config_file(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Load configuration from YAML file"""
    
    # Default config paths to check
    config_paths = [
        config_path,
        "config/settings.yaml",
        "config/config.yaml",
        "settings.yaml",
        "config.yaml"
    ]
    
    for path_str in config_paths:
        if not path_str:
            continue
            
        path = Path(path_str)
        if path.exists() and path.is_file():
            try:
                with open(path, 'r') as f:
                    return yaml.safe_load(f) or {}
            except Exception as e:
                print(f"Warning: Failed to load config from {path}: {e}")
    
    return {}


def get_env_overrides() -> Dict[str, Any]:
    """Get configuration overrides from environment variables"""
    
    overrides = {}
    
    # Environment
    if os.getenv('ENVIRONMENT'):
        overrides['environment'] = os.getenv('ENVIRONMENT')
    
    if os.getenv('DEBUG'):
        overrides['debug'] = os.getenv('DEBUG').lower() in ('true', '1', 'yes')
    
    # Database
    db_overrides = {}
    if os.getenv('DB_HOST'):
        db_overrides['host'] = os.getenv('DB_HOST')
    if os.getenv('DB_PORT'):
        db_overrides['port'] = int(os.getenv('DB_PORT'))
    if os.getenv('DB_NAME'):
        db_overrides['database'] = os.getenv('DB_NAME')
    if os.getenv('DB_USER'):
        db_overrides['username'] = os.getenv('DB_USER')
    if os.getenv('DB_PASSWORD'):
        db_overrides['password'] = os.getenv('DB_PASSWORD')
    
    if db_overrides:
        overrides['database'] = db_overrides
    
    # Exchange
    exchange_overrides = {}
    if os.getenv('EXCHANGE_API_KEY'):
        exchange_overrides['api_key'] = os.getenv('EXCHANGE_API_KEY')
    if os.getenv('EXCHANGE_API_SECRET'):
        exchange_overrides['api_secret'] = os.getenv('EXCHANGE_API_SECRET')
    if os.getenv('EXCHANGE_SANDBOX'):
        exchange_overrides['sandbox'] = os.getenv('EXCHANGE_SANDBOX').lower() in ('true', '1', 'yes')
    
    if exchange_overrides:
        overrides['exchange'] = exchange_overrides
    
    # AI
    ai_overrides = {}
    if os.getenv('AI_MODEL_TYPE'):
        ai_overrides['model_type'] = os.getenv('AI_MODEL_TYPE')
    if os.getenv('AI_RETRAIN_SCHEDULE'):
        ai_overrides['retrain_schedule'] = os.getenv('AI_RETRAIN_SCHEDULE')
    
    if ai_overrides:
        overrides['ai'] = ai_overrides
    
    # Risk
    risk_overrides = {}
    if os.getenv('MAX_POSITION_SIZE_PCT'):
        risk_overrides['max_position_size_pct'] = float(os.getenv('MAX_POSITION_SIZE_PCT'))
    if os.getenv('MAX_DAILY_LOSS_PCT'):
        risk_overrides['max_daily_loss_pct'] = float(os.getenv('MAX_DAILY_LOSS_PCT'))
    
    if risk_overrides:
        overrides['risk'] = risk_overrides
    
    # Monitoring
    monitoring_overrides = {}
    if os.getenv('LOG_LEVEL'):
        monitoring_overrides['log_level'] = os.getenv('LOG_LEVEL')
    if os.getenv('ENABLE_MONITORING'):
        monitoring_overrides['enable_monitoring'] = os.getenv('ENABLE_MONITORING').lower() in ('true', '1', 'yes')
    
    if monitoring_overrides:
        overrides['monitoring'] = monitoring_overrides
    
    return overrides


def merge_configs(base: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    """Deep merge configuration dictionaries"""
    
    result = base.copy()
    
    for key, value in overrides.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = value
    
    return result


def create_settings_from_dict(config_dict: Dict[str, Any]) -> Settings:
    """Create Settings object from dictionary"""
    
    # Extract nested configs
    database_config = config_dict.pop('database', {})
    exchange_config = config_dict.pop('exchange', {})
    ai_config = config_dict.pop('ai', {})
    risk_config = config_dict.pop('risk', {})
    monitoring_config = config_dict.pop('monitoring', {})
    backtest_config = config_dict.pop('backtest', {})
    
    # Create Settings object
    settings = Settings(**config_dict)
    
    # Override nested configs
    if database_config:
        settings.database = DatabaseConfig(**{**settings.database.__dict__, **database_config})
    
    if exchange_config:
        settings.exchange = ExchangeConfig(**{**settings.exchange.__dict__, **exchange_config})
    
    if ai_config:
        settings.ai = AIConfig(**{**settings.ai.__dict__, **ai_config})
    
    if risk_config:
        settings.risk = RiskConfig(**{**settings.risk.__dict__, **risk_config})
    
    if monitoring_config:
        settings.monitoring = MonitoringConfig(**{**settings.monitoring.__dict__, **monitoring_config})
    
    if backtest_config:
        settings.backtest = BacktestConfig(**{**settings.backtest.__dict__, **backtest_config})
    
    return settings


# Global settings instance
_settings: Optional[Settings] = None


def get_settings(config_path: Optional[str] = None) -> Settings:
    """Get singleton settings instance with configuration loading"""
    
    global _settings
    
    if _settings is None:
        # Load configuration from multiple sources
        file_config = load_config_file(config_path)
        env_config = get_env_overrides()
        
        # Merge configurations (env overrides file)
        merged_config = merge_configs(file_config, env_config)
        
        # Create settings
        _settings = create_settings_from_dict(merged_config)
        
        print(f"Settings loaded for environment: {_settings.environment}")
    
    return _settings


def reset_settings():
    """Reset settings (useful for testing)"""
    global _settings
    _settings = None


# CLI for configuration management
if __name__ == "__main__":
    import json
    import argparse
    
    parser = argparse.ArgumentParser(description="Configuration management")
    parser.add_argument('--show', action='store_true', help='Show current configuration')
    parser.add_argument('--generate-template', action='store_true', help='Generate template config file')
    parser.add_argument('--validate', action='store_true', help='Validate configuration')
    
    args = parser.parse_args()
    
    if args.show:
        settings = get_settings()
        print(json.dumps(settings.dict(), indent=2, default=str))
    
    elif args.generate_template:
        template = {
            'environment': 'development',
            'debug': True,
            'timezone': 'UTC',
            'data_path': 'data/',
            'logs_path': 'logs/',
            'database': {
                'host': 'localhost',
                'port': 5432,
                'database': 'trading_bot',
                'username': 'trader',
                'password': 'your_password_here'
            },
            'exchange': {
                'name': 'binance',
                'api_key': 'your_api_key_here',
                'api_secret': 'your_api_secret_here',
                'sandbox': True
            },
            'ai': {
                'model_type': 'xgboost',
                'retrain_schedule': 'weekly'
            },
            'risk': {
                'max_position_size_pct': 0.1,
                'max_daily_loss_pct': 0.05
            },
            'monitoring': {
                'enable_monitoring': True,
                'log_level': 'INFO'
            },
            'trading_symbols': ['BTC-USD', 'ETH-USD'],
            'trading_intervals': ['1m', '5m', '1h', '1d']
        }
        
        Path("config").mkdir(exist_ok=True)
        with open("config/settings.yaml", 'w') as f:
            yaml.dump(template, f, default_flow_style=False, indent=2)
        
        print("Template configuration generated: config/settings.yaml")
    
    elif args.validate:
        try:
            settings = get_settings()
            print("✅ Configuration is valid")
            print(f"Environment: {settings.environment}")
            print(f"Trading symbols: {settings.trading_symbols}")
        except Exception as e:
            print(f"❌ Configuration validation failed: {e}")
    
    else:
        parser.print_help()
