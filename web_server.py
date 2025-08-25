from fastapi import FastAPI
from prometheus_client import make_asgi_app, Counter

# Define Prometheus metrics
# These counters will be incremented in the main application logic
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
async def health_check():
    """
    Performs a basic health check.
    In a real application, this would check DB connections, etc.
    """
    # For now, this is a simple check. It can be expanded later.
    return {"status": "ok"}

# Add prometheus asgi middleware to route /metrics requests
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)
