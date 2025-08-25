from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
from typing import List

class Settings(BaseSettings):
    """
    Application settings.
    Can be configured via a .env file or environment variables.
    For lists like symbols, use a comma-separated string in the .env file.
    Example: SPOT_SYMBOLS="btcusdt,ethusdt,bnbusdt"
    """
    model_config = SettingsConfigDict(env_file='.env', env_file_encoding='utf-8', extra='ignore')

    # DATABASE
    db_dsn: str = Field("postgresql://user:password@localhost:5432/binance_data",
                        description="PostgreSQL DSN")

    # BINANCE API
    spot_ws_base_url: str = "wss://stream.binance.com:9443/ws"
    futures_ws_base_url: str = "wss://fstream.binance.com/ws"
    spot_api_base_url: str = "https://api.binance.com"
    futures_api_base_url: str = "https://fapi.binance.com"

    # SYMBOLS - can be overridden by environment variables
    spot_symbols: List[str] = ['btcusdt', 'ethusdt']
    futures_symbols: List[str] = ['btcusdt', 'ethusdt']

    # STREAMS
    spot_streams: List[str] = ['aggTrade', 'depth@100ms']
    futures_streams: List[str] = ['aggTrade', 'depth@100ms', 'markPrice@arr@1s', 'forceOrder']

    # REST API POLLING INTERVALS (in seconds)
    open_interest_poll_interval: int = 60
    depth_snapshot_poll_interval: int = 60

settings = Settings()
