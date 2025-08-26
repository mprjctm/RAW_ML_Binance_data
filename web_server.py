from fastapi import FastAPI, Response, status
from prometheus_client import make_asgi_app, Counter

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


# Add prometheus asgi middleware to route /metrics requests
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)
