import asyncio
import logging
import signal
import time
from asyncio import Queue

import uvicorn

from config import settings
from database import db
from rest_client import RestClient
from state import app_state
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
    """Consumes data from the queue, updates state, and inserts into the database."""
    logger.info("Data consumer started.")
    while True:
        try:
            item = await data_queue.get()
            source = item['source']
            data = item['payload']

            # Update state based on source
            now = time.time()
            if source == 'spot_ws':
                app_state.last_spot_ws_message_time = now
            elif source == 'futures_ws':
                app_state.last_futures_ws_message_time = now
            elif source == 'open_interest':
                app_state.last_open_interest_update_time = now
            elif source == 'depth_snapshot':
                app_state.last_depth_snapshot_update_time = now

            # Process data and insert into DB
            stream_type = data.get('e') or data.get('type')

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
        try:
            await db.init_db()
            app_state.db_connected = True
        except Exception as e:
            logger.critical(f"Failed to initialize database. Shutting down. Error: {e}")
            app_state.db_connected = False
            return # Exit the application

        # Create a shared queue for data
        data_queue = Queue()

        # --- Setup WebSocket clients ---
        spot_ws_streams = [f"{s}@{st}" for s in settings.spot_symbols for st in settings.spot_streams]
        spot_ws_client = WebsocketClient(settings.spot_ws_base_url, spot_ws_streams, data_queue, source_name="spot_ws")

        # For futures, some streams are per-symbol, others are global
        futures_ws_streams = []
        for s in settings.futures_symbols:
            futures_ws_streams.extend([f"{s}@aggTrade", f"{s}@depth@100ms"])
        futures_ws_streams.extend(["!markPrice@arr@1s", "!forceOrder@arr"])
        futures_ws_client = WebsocketClient(settings.futures_ws_base_url, list(set(futures_ws_streams)), data_queue, source_name="futures_ws")

        # --- Setup REST client ---
        rest_client = RestClient(settings.spot_symbols, settings.futures_symbols, data_queue)

        # --- Setup Web Server ---
        uvicorn_config = uvicorn.Config(app, host="0.0.0.0", port=8000, log_level="info")
        server = uvicorn.Server(uvicorn_config)

        # --- Create and schedule tasks ---
        if settings.enable_websocket_spot:
            self.tasks.append(asyncio.create_task(spot_ws_client.run()))
            logger.info("Spot WebSocket client enabled and scheduled.")
        if settings.enable_websocket_futures:
            self.tasks.append(asyncio.create_task(futures_ws_client.run()))
            logger.info("Futures WebSocket client enabled and scheduled.")
        if settings.enable_open_interest:
            self.tasks.append(asyncio.create_task(rest_client.run_open_interest_fetcher()))
            logger.info("Open Interest poller enabled and scheduled.")
        if settings.enable_depth_snapshot:
            self.tasks.append(asyncio.create_task(rest_client.run_depth_snapshot_fetcher()))
            logger.info("Depth Snapshot poller enabled and scheduled.")

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
