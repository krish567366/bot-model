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
    
    # Cross-exchange arbitrage
    min_arbitrage_spread: Decimal = Field(Decimal("0.05"), description="Min arbitrage spread %")
    transfer_time_penalty: Decimal = Field(Decimal("0.02"), description="Transfer time penalty %")
    
    # Triangular arbitrage
    min_triangular_profit: Decimal = Field(Decimal("0.03"), description="Min triangular profit %")
    max_triangular_legs: int = Field(3, description="Max legs in triangular arbitrage")
    
    # Signal filters
    min_volume_threshold: Decimal = Field(Decimal("50000"), description="Min 24h volume USD")
    max_spread_width: Decimal = Field(Decimal("2.0"), description="Max spread width %")
    
    class Config:
        env_prefix = ""
        case_sensitive = False


class ExecutionConfig(BaseSettings):
    """Order execution configuration"""
    
    # Order types
    default_order_type: str = Field("limit", description="Default order type")
    enable_ioc_orders: bool = Field(True, description="Enable Immediate-Or-Cancel orders")
    enable_fok_orders: bool = Field(False, description="Enable Fill-Or-Kill orders")
    
    # Timing
    order_timeout_seconds: int = Field(30, description="Order timeout in seconds")
    max_execution_delay: int = Field(5, description="Max execution delay in seconds")
    
    class Config:
        env_prefix = ""
        case_sensitive = False


class APIConfig(BaseSettings):
    """API server configuration"""
    host: str = Field("0.0.0.0", env="API_HOST")
    port: int = Field(8000, env="API_PORT")
    secret_key: str = Field(..., env="API_SECRET_KEY")
    enable_cors: bool = Field(True, env="API_ENABLE_CORS")
    
    # TLS
    use_tls: bool = Field(False, env="API_USE_TLS")
    cert_file: Optional[Path] = Field(None, env="API_CERT_FILE")
    key_file: Optional[Path] = Field(None, env="API_KEY_FILE")
    
    class Config:
        case_sensitive = False


class MonitoringConfig(BaseSettings):
    """Monitoring and alerting configuration"""
    
    # Logging
    log_level: str = Field("INFO", env="LOG_LEVEL")
    log_file: Optional[Path] = Field(None, env="LOG_FILE")
    enable_structured_logging: bool = Field(True, env="ENABLE_STRUCTURED_LOGGING")
    
    # Slack
    slack_webhook_url: Optional[AnyHttpUrl] = Field(None, env="SLACK_WEBHOOK_URL")
    slack_channel: str = Field("#trading-alerts", env="SLACK_CHANNEL")
    enable_slack_alerts: bool = Field(False, env="ENABLE_SLACK_ALERTS")
    
    # Telegram
    telegram_bot_token: Optional[str] = Field(None, env="TELEGRAM_BOT_TOKEN")
    telegram_chat_id: Optional[str] = Field(None, env="TELEGRAM_CHAT_ID")
    enable_telegram_alerts: bool = Field(False, env="ENABLE_TELEGRAM_ALERTS")
    
    # Metrics
    enable_prometheus_metrics: bool = Field(True, env="ENABLE_PROMETHEUS_METRICS")
    prometheus_port: int = Field(8001, env="PROMETHEUS_PORT")
    
    class Config:
        case_sensitive = False


class DataConfig(BaseSettings):
    """Data storage and management configuration"""
    
    # Database
    database_url: str = Field("sqlite:///./arbitrage.db", env="DATABASE_URL")
    postgres_url: Optional[str] = Field(None, env="POSTGRES_URL")
    
    # Market data storage
    enable_data_recording: bool = Field(True, env="ENABLE_DATA_RECORDING")
    data_retention_days: int = Field(30, env="DATA_RETENTION_DAYS")
    parquet_compression: str = Field("snappy", env="PARQUET_COMPRESSION")
    
    # Backup
    enable_backups: bool = Field(True, env="ENABLE_BACKUPS")
    backup_interval_hours: int = Field(24, env="BACKUP_INTERVAL_HOURS")
    backup_location: Path = Field(Path("./backups"), env="BACKUP_LOCATION")
    
    # Buffer settings for batch operations
    max_buffer_size_mb: int = Field(100, env="MAX_BUFFER_SIZE_MB")
    
    class Config:
        case_sensitive = False


class PipelineConfig(BaseSettings):
    """Data pipeline configuration for market data ingestion and processing"""
    
    # Data sources
    enable_yfinance: bool = Field(True, description="Enable Yahoo Finance data source")
    enable_alpha_vantage: bool = Field(False, description="Enable Alpha Vantage data source")
    enable_ccxt: bool = Field(True, description="Enable CCXT crypto data source")
    
    # API keys
    alpha_vantage_api_key: Optional[str] = Field(None, env="ALPHA_VANTAGE_API_KEY")
    
    # Data symbols and intervals
    default_symbols: List[str] = Field([
        "AAPL", "GOOGL", "MSFT", "TSLA",  # Stocks
        "BTC-USD", "ETH-USD", "ADA-USD",  # Crypto
        "EURUSD=X", "GBPUSD=X"           # Forex
    ])
    
    default_intervals: List[str] = Field(["1d", "1h", "5m"])
    crypto_symbols: List[str] = Field(["BTC/USDT", "ETH/USDT", "ADA/USDT", "DOT/USDT"])
    forex_pairs: List[str] = Field(["EUR/USD", "GBP/USD", "JPY/USD", "AUD/USD"])
    
    # Historical data parameters
    historical_lookback_days: int = Field(365, description="Days of historical data to fetch")
    max_batch_size: int = Field(100, description="Max symbols to process in one batch")
    
    # Data quality and cleaning
    enable_data_validation: bool = Field(True)
    enable_outlier_removal: bool = Field(True)
    outlier_threshold_std: float = Field(3.0, description="Standard deviations for outlier detection")
    fill_missing_method: str = Field("ffill", description="Method to fill missing values")
    
    # Feature engineering
    enable_technical_indicators: bool = Field(True)
    enable_custom_features: bool = Field(True)
    rsi_window: int = Field(14, description="RSI calculation window")
    ma_windows: List[int] = Field([5, 10, 20, 50, 200], description="Moving average windows")
    bollinger_window: int = Field(20, description="Bollinger bands window")
    bollinger_std: float = Field(2.0, description="Bollinger bands standard deviations")
    
    # Resampling options
    enable_resampling: bool = Field(True)
    resample_intervals: List[str] = Field(["5T", "15T", "1H", "4H", "1D"])
    
    # Storage settings
    database_path: str = Field("data/market_data.db", env="PIPELINE_DB_PATH")
    enable_parquet_storage: bool = Field(True, description="Enable Parquet file storage")
    parquet_path: str = Field("data/parquet/", env="PARQUET_PATH")
    
    # Performance and rate limiting
    max_concurrent_requests: int = Field(5, description="Max concurrent API requests")
    request_delay_seconds: float = Field(1.0, description="Delay between API requests")
    retry_attempts: int = Field(3, description="Number of retry attempts for failed requests")
    timeout_seconds: int = Field(30, description="Request timeout in seconds")
    
    # Data update schedule
    enable_live_updates: bool = Field(False, description="Enable live data updates")
    update_interval_minutes: int = Field(15, description="Minutes between data updates")
    market_hours_only: bool = Field(True, description="Only update during market hours")
    
    # ML preparation settings
    enable_ml_features: bool = Field(True, description="Enable ML-specific feature engineering")
    sequence_length: int = Field(60, description="Sequence length for time series models")
    train_test_split: float = Field(0.8, description="Train/test split ratio")
    validation_split: float = Field(0.2, description="Validation split ratio")
    
    # Data export settings
    enable_csv_export: bool = Field(False, description="Enable CSV data export")
    csv_export_path: str = Field("data/exports/", env="CSV_EXPORT_PATH")
    
    class Config:
        case_sensitive = False


class PerformanceConfig(BaseSettings):
    """Performance and optimization settings"""
    
    # WebSocket
    ws_ping_interval: int = Field(30, env="WS_PING_INTERVAL")
    ws_ping_timeout: int = Field(10, env="WS_PING_TIMEOUT")
    ws_max_reconnect_attempts: int = Field(10, env="WS_MAX_RECONNECT_ATTEMPTS")
    
    # Threading
    max_worker_threads: int = Field(4, env="MAX_WORKER_THREADS")
    async_queue_size: int = Field(1000, env="ASYNC_QUEUE_SIZE")
    
    # Memory management
    max_orderbook_levels: int = Field(1000, env="MAX_ORDERBOOK_LEVELS")
    data_buffer_size: int = Field(10000, env="DATA_BUFFER_SIZE")
    
    class Config:
        case_sensitive = False


class BacktestConfig(BaseSettings):
    """Backtesting configuration"""
    start_date: str = Field("2024-01-01", env="BACKTEST_START_DATE")
    end_date: str = Field("2024-12-31", env="BACKTEST_END_DATE")
    initial_balance: Decimal = Field(Decimal("10000"), env="BACKTEST_INITIAL_BALANCE")
    
    class Config:
        case_sensitive = False


class Settings(BaseSettings):
    """Main settings class that aggregates all configuration sections"""
    
    # Environment
    debug: bool = Field(False, env="DEBUG")
    testing: bool = Field(False, env="TESTING")
    mock_exchanges: bool = Field(False, env="MOCK_EXCHANGES")
    
    # Exchange configurations
    binance: ExchangeConfig = Field(default_factory=ExchangeConfig)
    kraken: ExchangeConfig = Field(default_factory=ExchangeConfig)
    coinbase: ExchangeConfig = Field(default_factory=ExchangeConfig)
    bybit: ExchangeConfig = Field(default_factory=ExchangeConfig)
    
    # Component configurations
    risk: RiskConfig = Field(default_factory=RiskConfig)
    strategy: StrategyConfig = Field(default_factory=StrategyConfig)
    execution: ExecutionConfig = Field(default_factory=ExecutionConfig)
    api: APIConfig = Field(default_factory=APIConfig)
    monitoring: MonitoringConfig = Field(default_factory=MonitoringConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    pipeline: PipelineConfig = Field(default_factory=PipelineConfig)
    performance: PerformanceConfig = Field(default_factory=PerformanceConfig)
    backtest: BacktestConfig = Field(default_factory=BacktestConfig)
    
    # Trading symbols
    trading_symbols: List[str] = Field(
        default=["BTC/USDT", "ETH/USDT", "BNB/USDT", "ADA/USDT"],
        description="List of symbols to trade"
    )
    
    # Supported exchanges
    enabled_exchanges: List[str] = Field(
        default=["binance", "kraken"],
        description="List of enabled exchanges"
    )
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        
        # Nested environment variable support
        env_nested_delimiter = "_"
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # Initialize exchange configs with proper prefixes
        self.binance = ExchangeConfig(_env_prefix="BINANCE_")
        self.kraken = ExchangeConfig(_env_prefix="KRAKEN_")
        self.coinbase = ExchangeConfig(_env_prefix="COINBASE_")
        self.bybit = ExchangeConfig(_env_prefix="BYBIT_")
    
    @field_validator('trading_symbols', mode='before')
    def parse_trading_symbols(cls, v):
        """Parse trading symbols from string or list"""
        if isinstance(v, str):
            return [s.strip() for s in v.split(",")]
        return v
    
    @field_validator('enabled_exchanges', mode='before')
    def parse_enabled_exchanges(cls, v):
        """Parse enabled exchanges from string or list"""
        if isinstance(v, str):
            return [s.strip() for s in v.split(",")]
        return v
    
    def get_exchange_config(self, exchange: str) -> Optional[ExchangeConfig]:
        """Get configuration for a specific exchange"""
        exchange_lower = exchange.lower()
        return getattr(self, exchange_lower, None)
    
    def is_exchange_enabled(self, exchange: str) -> bool:
        """Check if an exchange is enabled"""
        return exchange.lower() in [e.lower() for e in self.enabled_exchanges]
    
    def get_supported_symbols(self) -> List[str]:
        """Get list of supported trading symbols"""
        return self.trading_symbols.copy()
    
    def validate_configuration(self) -> Dict[str, List[str]]:
        """Validate configuration and return any issues"""
        issues = {"errors": [], "warnings": []}
        
        # Check required exchange configurations
        for exchange in self.enabled_exchanges:
            config = self.get_exchange_config(exchange)
            if not config:
                issues["errors"].append(f"Configuration missing for exchange: {exchange}")
                continue
            
            if not config.api_key or not config.secret:
                issues["errors"].append(f"API credentials missing for {exchange}")
        
        # Check API secret key
        if not self.api.secret_key or len(self.api.secret_key) < 32:
            issues["errors"].append("API secret key must be at least 32 characters long")
        
        # Check risk limits
        if self.risk.max_position_size <= 0:
            issues["errors"].append("Max position size must be positive")
        
        if self.risk.max_daily_loss <= 0:
            issues["errors"].append("Max daily loss must be positive")
        
        # Check strategy parameters
        if self.strategy.min_arbitrage_spread <= 0:
            issues["warnings"].append("Min arbitrage spread is very low")
        
        # Check data directory permissions
        try:
            self.data.backup_location.mkdir(parents=True, exist_ok=True)
        except PermissionError:
            issues["errors"].append(f"Cannot write to backup location: {self.data.backup_location}")
        
        return issues


# Global settings instance
_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """Get the global settings instance (singleton pattern)"""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def reload_settings() -> Settings:
    """Reload settings from environment"""
    global _settings
    _settings = Settings()
    return _settings


# Convenience functions
def get_exchange_config(exchange: str) -> Optional[ExchangeConfig]:
    """Get exchange configuration"""
    return get_settings().get_exchange_config(exchange)


def is_exchange_enabled(exchange: str) -> bool:
    """Check if exchange is enabled"""
    return get_settings().is_exchange_enabled(exchange)


def get_trading_symbols() -> List[str]:
    """Get list of trading symbols"""
    return get_settings().get_supported_symbols()


# Development helpers
if __name__ == "__main__":
    # Test configuration loading
    settings = get_settings()
    
    print("Configuration loaded successfully!")
    print(f"Debug mode: {settings.debug}")
    print(f"Enabled exchanges: {settings.enabled_exchanges}")
    print(f"Trading symbols: {settings.trading_symbols}")
    
    # Validate configuration
    issues = settings.validate_configuration()
    if issues["errors"]:
        print(f"\nConfiguration errors: {issues['errors']}")
    if issues["warnings"]:
        print(f"\nConfiguration warnings: {issues['warnings']}")
    
    print(f"\nRisk limits:")
    print(f"  Max position size: ${settings.risk.max_position_size}")
    print(f"  Max daily loss: ${settings.risk.max_daily_loss}")
    print(f"  Circuit breaker: {settings.risk.enable_circuit_breaker}")
    
    print(f"\nAPI server:")
    print(f"  Host: {settings.api.host}")
    print(f"  Port: {settings.api.port}")
    print(f"  TLS: {settings.api.use_tls}")
