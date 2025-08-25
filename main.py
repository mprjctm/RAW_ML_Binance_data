import asyncio
import logging
import signal
from asyncio import Queue

import uvicorn

from config import settings
from database import db
from rest_client import RestClient
from web_server import app, MESSAGES_PROCESSED_COUNTER
from websocket_client import WebsocketClient

# Setup logging
def setup_logging():
    """Configures logging to file and console."""
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    # System log handler (all logs)
    system_handler = logging.FileHandler("system.log")
    system_handler.setLevel(logging.INFO)
    system_handler.setFormatter(formatter)
    root_logger.addHandler(system_handler)

    # Error log handler (only errors)
    error_handler = logging.FileHandler("error.log")
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(formatter)
    root_logger.addHandler(error_handler)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

setup_logging()
logger = logging.getLogger(__name__)


async def data_consumer(data_queue: Queue):
    """Consumes data from the queue and inserts it into the database."""
    logger.info("Data consumer started.")
    while True:
        try:
            data = await data_queue.get()
            stream_type = data.get('e') or data.get('type') # 'e' for websocket, 'type' for our custom rest data

            if stream_type == 'aggTrade':
                await db.insert_agg_trade(data)
                MESSAGES_PROCESSED_COUNTER.labels(stream_type='agg_trade').inc()
            elif stream_type == 'depthUpdate':
                await db.insert_depth_update(data)
                MESSAGES_PROCESSED_COUNTER.labels(stream_type='depth_update').inc()
            elif stream_type == 'markPriceUpdate':
                await db.insert_mark_price(data)
                MESSAGES_PROCESSED_COUNTER.labels(stream_type='mark_price').inc()
            elif stream_type == 'forceOrder':
                await db.insert_force_order(data)
                MESSAGES_PROCESSED_COUNTER.labels(stream_type='force_order').inc()
            elif stream_type == 'openInterest':
                await db.insert_open_interest(data)
                MESSAGES_PROCESSED_COUNTER.labels(stream_type='open_interest').inc()
            elif stream_type == 'depthSnapshot':
                await db.insert_depth_snapshot(data)
                MESSAGES_PROCESSED_COUNTER.labels(stream_type='depth_snapshot').inc()
            else:
                logger.warning(f"Unknown stream type received: {stream_type}")

            data_queue.task_done()
        except Exception as e:
            logger.error(f"Error processing data from queue: {e}")


class Service:
    """Manages the lifecycle of the application components."""
    def __init__(self):
        self.tasks = []
        self.shutdown_event = asyncio.Event()

    async def shutdown(self, sig):
        logger.warning(f"Received exit signal {sig.name}...")
        self.shutdown_event.set()

        logger.info("Cancelling all running tasks...")
        for task in self.tasks:
            task.cancel()

        await asyncio.gather(*self.tasks, return_exceptions=True)

        await db.close()
        logger.info("Application shutdown complete.")

    async def run(self):
        """Main application entry point."""
        logger.info("Starting data collector application")

        # Setup signal handlers for graceful shutdown
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, lambda s=sig: asyncio.create_task(self.shutdown(s)))

        # Initialize database
        await db.init_db()

        # Create a shared queue for data
        data_queue = Queue()

        # --- Setup WebSocket clients ---
        spot_ws_streams = [f"{s}@{st}" for s in settings.spot_symbols for st in settings.spot_streams]
        spot_ws_client = WebsocketClient(settings.spot_ws_base_url, spot_ws_streams, data_queue)

        # For futures, some streams are per-symbol, others are global
        futures_ws_streams = []
        for s in settings.futures_symbols:
            futures_ws_streams.extend([f"{s}@aggTrade", f"{s}@depth@100ms"])
        futures_ws_streams.extend(["!markPrice@arr@1s", "!forceOrder@arr"])
        futures_ws_client = WebsocketClient(settings.futures_ws_base_url, list(set(futures_ws_streams)), data_queue)

        # --- Setup REST client ---
        rest_client = RestClient(settings.spot_symbols, settings.futures_symbols, data_queue)

        # --- Setup Web Server ---
        uvicorn_config = uvicorn.Config(app, host="0.0.0.0", port=8000, log_level="info")
        server = uvicorn.Server(uvicorn_config)

        # --- Create and schedule tasks ---
        self.tasks.append(asyncio.create_task(spot_ws_client.run()))
        self.tasks.append(asyncio.create_task(futures_ws_client.run()))
        self.tasks.append(asyncio.create_task(rest_client.run_open_interest_fetcher()))
        self.tasks.append(asyncio.create_task(rest_client.run_depth_snapshot_fetcher()))
        self.tasks.append(asyncio.create_task(data_consumer(data_queue)))
        self.tasks.append(asyncio.create_task(server.serve()))

        logger.info("All components are running.")
        await self.shutdown_event.wait()

        # Clean up REST client session
        await rest_client.close()


if __name__ == "__main__":
    service = Service()
    try:
        asyncio.run(service.run())
    except asyncio.CancelledError:
        logger.info("Main task cancelled.")
