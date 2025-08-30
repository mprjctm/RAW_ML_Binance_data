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

    # --- DATABASE ---
    # Full connection string (DSN) for the PostgreSQL / TimescaleDB database.
    db_dsn: str = Field("postgresql://user:password@localhost:5432/binance_data",
                        description="PostgreSQL DSN")

    # --- BINANCE API ---
    # Base URL for the Binance Spot market WebSocket stream.
    spot_ws_base_url: str = "wss://stream.binance.com:9443/ws"
    # Base URL for the Binance Futures market WebSocket stream.
    futures_ws_base_url: str = "wss://fstream.binance.com/ws"
    # Base URL for the Binance Spot market REST API.
    spot_api_base_url: str = "https://api.binance.com"
    # Base URL for the Binance Futures market REST API.
    futures_api_base_url: str = "https://fapi.binance.com"

    # --- SYMBOLS ---
    # Comma-separated string of symbols to track on the Spot market.
    spot_symbols_str: str = Field("adausdt,maticusdt,dotusdt,avaxusdt,linkusdt,ltcusdt,atomusdt,nearusdt,ftmusdt,apeusdt", alias='SPOT_SYMBOLS')
    # Comma-separated string of symbols to track on the Futures market.
    futures_symbols_str: str = Field("adausdt,maticusdt,dotusdt,avaxusdt,linkusdt,ltcusdt,atomusdt,nearusdt,ftmusdt,apeusdt", alias='FUTURES_SYMBOLS')

    # --- STREAMS (Internal) ---
    # These are not meant to be configured from .env, but are used in the application logic.
    spot_streams: List[str] = ['aggTrade', 'depth@500ms']
    futures_streams: List[str] = ['aggTrade', 'depth@500ms', 'markPrice@arr@1s', 'forceOrder']

    # --- REST API POLLING INTERVALS ---
    # Interval in seconds for polling the Open Interest REST API endpoint.
    open_interest_poll_interval: int = 60
    # Interval in seconds for polling the Depth Snapshot REST API endpoint.
    depth_snapshot_poll_interval: int = 60

    # --- STREAM CONTROLS ---
    # Set to true to enable collecting data from Spot WebSockets.
    enable_websocket_spot: bool = True
    # Set to true to enable collecting data from Futures WebSockets.
    enable_websocket_futures: bool = True
    # Set to true to enable polling for Open Interest data.
    enable_open_interest: bool = True
    # Set to true to enable polling for Depth Snapshots.
    enable_depth_snapshot: bool = True

    # --- LIQUIDATION ALERTS ---
    # Set to true to enable the liquidation alert feature.
    enable_liquidation_alerts: bool = True
    # The number of days (N) to look back for fetching the historical max price for an alert.
    alert_max_days_lookback: int = 30
    # The percentage (V) drop from the N-day max price that will trigger a liquidation alert.
    alert_percentage_drop: float = 10.0

    @property
    def spot_symbols(self) -> List[str]:
        """Returns a parsed list of spot symbols."""
        return [s.strip() for s in self.spot_symbols_str.split(',') if s.strip()]

    @property
    def futures_symbols(self) -> List[str]:
        """Returns a parsed list of futures symbols."""
        return [s.strip() for s in self.futures_symbols_str.split(',') if s.strip()]


settings = Settings()
