from datetime import datetime, timedelta
from typing import List, Optional

from fastapi import FastAPI, Response, status
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import make_asgi_app, Counter
from pydantic import BaseModel

from database import db
from state import app_state

# Define Prometheus metrics
MESSAGES_PROCESSED_COUNTER = Counter(
    "messages_processed_total",
    "Total number of messages processed, by stream type",
    ["stream_type"]
)

# Create FastAPI app
app = FastAPI(
    title="Binance Data Collector",
    description="A service to collect and store market data from Binance.",
    version="0.1.0"
)

@app.get("/health", summary="Health Check", tags=["Health"])
async def health_check(response: Response):
    """
    Performs an intelligent health check on all application components.
    Returns HTTP 200 if healthy.
    Returns HTTP 503 if any component is unhealthy.
    """
    # Use a longer delay for the health check to avoid flapping during temporary issues
    # The check in state.py uses a default of 300 seconds (5 minutes)
    is_healthy, message = app_state.is_healthy()

    if is_healthy:
        return {"status": "ok", "details": message}
    else:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return {"status": "error", "details": message}


# Add CORS middleware to allow frontend requests
# WARNING: This is a permissive configuration. For production, you should restrict origins.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)


# --- API Models ---

class Kline(BaseModel):
    """Pydantic model for a single candlestick (OHLCV)."""
    time: datetime
    open: float
    high: float
    low: float
    close: float
    volume: Optional[float] = 0.0

    class Config:
        orm_mode = True # This helps Pydantic work with ORM models, and asyncpg.Record behaves similarly.


# --- API Endpoints ---

@app.get("/api/v1/klines", response_model=List[Kline], summary="Get Candlestick Data", tags=["Data"])
async def get_klines_data(
    symbol: str,
    interval: str = '1m',
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None
):
    """
    Provides OHLCV (candlestick) data for a given symbol and time interval.

    - **symbol**: The trading symbol, e.g., `BTCUSDT`.
    - **interval**: The time bucket size, e.g., `1m`, `5m`, `1h`, `1d`.
    - **start_time**: The start of the time range in ISO format (e.g., `2025-09-18T00:00:00Z`). Defaults to 1 day ago.
    - **end_time**: The end of the time range in ISO format. Defaults to now.
    """
    # Set default time range if not provided
    if end_time is None:
        end_time = datetime.utcnow()
    if start_time is None:
        start_time = end_time - timedelta(days=1)

    # Fetch data from the database
    # The result is a list of asyncpg.Record objects, which Pydantic can handle with orm_mode.
    raw_klines = await db.get_klines(symbol.upper(), interval, start_time, end_time)

    # Filter out potential empty buckets if any (where open price is None)
    # This can happen if time_bucket_gapfill is used and there are no trades in a bucket.
    # With plain time_bucket, this is less likely but still a good safeguard.
    # Pydantic will automatically convert Decimal types from the DB to floats.
    return [kline for kline in raw_klines if kline["open"] is not None]


# Add prometheus asgi middleware to route /metrics requests
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)
