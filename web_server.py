from typing import List
import logging
from fastapi import FastAPI, Response, status, APIRouter, HTTPException, Query
from prometheus_client import make_asgi_app, Counter

from state import app_state
from database import db
from api_models import LiquidationEvent, OrderBook, Trade, OrderRequest, OrderResponse

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

# Get a logger instance
logger = logging.getLogger(__name__)

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


# Add prometheus asgi middleware to route /metrics requests
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

# --- API V1 Router ---
router_v1 = APIRouter(prefix="/api/v1", tags=["V1 - Data"])

@router_v1.get("/liquidations/{symbol}", response_model=List[LiquidationEvent])
async def get_liquidations(
    symbol: str,
    limit: int = Query(10, ge=1, le=100, description="Number of liquidations to retrieve")
):
    """
    Get the most recent liquidation orders for a given symbol.
    """
    raw_data = await db.get_recent_liquidations(symbol.upper(), limit)
    if not raw_data:
        raise HTTPException(status_code=404, detail="No liquidations found for this symbol.")
    return raw_data

@router_v1.get("/orderbook/{symbol}", response_model=OrderBook)
async def get_orderbook(symbol: str):
    """
    Get the latest order book snapshot for a given futures symbol.
    """
    raw_data = await db.get_latest_orderbook_snapshot(symbol.upper())
    if not raw_data:
        raise HTTPException(status_code=404, detail="No order book snapshot found for this symbol.")

    # The payload in db is the direct snapshot, let's format it for our OrderBook model
    # Note: Binance snapshot has 'ts', 'bids', 'asks'. We need to map 'ts' to 'timestamp'
    # and the bid/ask arrays to our OrderBookEntry model.
    return {
        "timestamp": raw_data['ts'],
        "symbol": symbol.upper(),
        "bids": [{"price": float(p), "quantity": float(q)} for p, q in raw_data['bids']],
        "asks": [{"price": float(p), "quantity": float(q)} for p, q in raw_data['asks']]
    }

@router_v1.get("/trades/{symbol}", response_model=List[Trade])
async def get_trades(
    symbol: str,
    limit: int = Query(20, ge=1, le=100, description="Number of trades to retrieve")
):
    """
    Get the most recent trades for a given symbol.
    """
    raw_data = await db.get_latest_agg_trades(symbol.upper(), limit)
    if not raw_data:
        raise HTTPException(status_code=404, detail="No trades found for this symbol.")

    # The payload in db is the direct aggTrade message. Let's parse it for our Trade model.
    # The model uses aliases, so we can pass the dicts directly.
    return raw_data

@router_v1.post("/orders", response_model=OrderResponse)
async def create_order(order: OrderRequest):
    """
    Placeholder endpoint to simulate placing a trade order.
    In a real scenario, this would interact with the exchange API.
    Here, we just log the order and return a success message.
    """
    logger.info(f"Received order: {order.model_dump_json()}")

    # Simulate a successful order placement
    return OrderResponse(
        status="success",
        order_id=f"simulated_{abs(hash(order.model_dump_json()))}", # Generate a fake order ID
        message="Order received and logged successfully."
    )

app.include_router(router_v1)
