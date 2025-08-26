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
    model_config = SettingsConfigDict(env_file='.env', env_file_encoding='utf-8', extra='ignore',
                                      # This allows properties to be used
                                      validate_assignment=True)

    # DATABASE
    db_dsn: str = Field("postgresql://user:password@localhost:5432/binance_data",
                        description="PostgreSQL DSN")

    # BINANCE API
    spot_ws_base_url: str = "wss://stream.binance.com:9443/ws"
    futures_ws_base_url: str = "wss://fstream.binance.com/ws"
    spot_api_base_url: str = "https://api.binance.com"
    futures_api_base_url: str = "https://fapi.binance.com"

    # SYMBOLS - raw strings read from environment
    # These are aliased to read SPOT_SYMBOLS and FUTURES_SYMBOLS from .env
    spot_symbols_str: str = Field("adausdt,maticusdt,dotusdt,avaxusdt,linkusdt,ltcusdt,atomusdt,nearusdt,ftmusdt,apeusdt", alias='SPOT_SYMBOLS')
    futures_symbols_str: str = Field("adausdt,maticusdt,dotusdt,avaxusdt,linkusdt,ltcusdt,atomusdt,nearusdt,ftmusdt,apeusdt", alias='FUTURES_SYMBOLS')

    # STREAMS
    # These are not meant to be configured from .env, but are kept for logic
    spot_streams: List[str] = ['aggTrade', 'depth@500ms']
    futures_streams: List[str] = ['aggTrade', 'depth@500ms', 'markPrice@arr@1s', 'forceOrder']

    # REST API POLLING INTERVALS (in seconds)
    open_interest_poll_interval: int = 60
    depth_snapshot_poll_interval: int = 60

    # --- STREAM CONTROLS ---
    # Set to true to enable the stream group, false to disable
    enable_websocket_spot: bool = True
    enable_websocket_futures: bool = True
    enable_open_interest: bool = True
    enable_depth_snapshot: bool = True

    @property
    def spot_symbols(self) -> List[str]:
        """Returns a parsed list of spot symbols."""
        return [s.strip() for s in self.spot_symbols_str.split(',') if s.strip()]

    @property
    def futures_symbols(self) -> List[str]:
        """Returns a parsed list of futures symbols."""
        return [s.strip() for s in self.futures_symbols_str.split(',') if s.strip()]


settings = Settings()
