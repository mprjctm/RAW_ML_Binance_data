import asyncio
import logging
import signal
import time
from asyncio import Queue
from collections import defaultdict

import aiohttp
import uvicorn

from alerter import LiquidationAlerter
from config import settings
from database import db
from rest_client import RestClient
from state import app_state
from web_server import app, MESSAGES_PROCESSED_COUNTER
from websocket_client import WebsocketClient

# --- Batching Configuration ---
BATCH_SIZE = 200
FLUSH_INTERVAL = 1.0  # seconds

# Setup logging
def setup_logging():
    """Configures logging to file and console."""
    # Configure root logger
    root_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    # System handler (for general logs)
    system_handler = logging.FileHandler("system.log")
    system_handler.setLevel(logging.INFO)
    system_handler.setFormatter(root_formatter)
    root_logger.addHandler(system_handler)

    # Error handler
    error_handler = logging.FileHandler("error.log")
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(root_formatter)
    root_logger.addHandler(error_handler)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(root_formatter)
    root_logger.addHandler(console_handler)

    # Alerter logger (for liquidation alerts)
    alert_formatter = logging.Formatter('%(asctime)s - %(message)s')
    alert_logger = logging.getLogger('alerter')
    alert_logger.setLevel(logging.INFO)
    alert_logger.propagate = False  # Do not forward alert logs to the root logger

    alert_handler = logging.FileHandler("alert.log")
    alert_handler.setLevel(logging.INFO)
    alert_handler.setFormatter(alert_formatter)
    alert_logger.addHandler(alert_handler)

setup_logging()
logger = logging.getLogger(__name__)


class BatchingDataConsumer:
    def __init__(self, data_queue: Queue, alerter: LiquidationAlerter | None = None):
        self._data_queue = data_queue
        self._batches = defaultdict(list)
        self._last_flush_time = time.time()
        self._alerter = alerter

    async def _flush_all_batches(self, force=False):
        flush_tasks = []
        batches_to_flush = {k: v for k, v in self._batches.items() if v}
        if not batches_to_flush: return
        logger.info(f"Flushing {len(batches_to_flush)} non-empty batches...")
        if batches_to_flush.get('agg_trade'): flush_tasks.append(db.batch_insert_agg_trades(batches_to_flush['agg_trade']))
        if batches_to_flush.get('depth_update'): flush_tasks.append(db.batch_insert_depth_updates(batches_to_flush['depth_update']))
        if batches_to_flush.get('mark_price'): flush_tasks.append(db.batch_insert_mark_prices(batches_to_flush['mark_price']))
        if batches_to_flush.get('force_order'): flush_tasks.append(db.batch_insert_force_orders(batches_to_flush['force_order']))
        if batches_to_flush.get('open_interest'): flush_tasks.append(db.batch_insert_open_interest(batches_to_flush['open_interest']))
        if batches_to_flush.get('depth_snapshot'): flush_tasks.append(db.batch_insert_depth_snapshots(batches_to_flush['depth_snapshot']))
        try:
            await asyncio.gather(*flush_tasks)
            for batch_key, batch_list in batches_to_flush.items():
                MESSAGES_PROCESSED_COUNTER.labels(stream_type=batch_key).inc(len(batch_list))
                self._batches[batch_key] = []
            self._last_flush_time = time.time()
            logger.info(f"Flushed {len(flush_tasks)} batches successfully.")
        except Exception as e:
            logger.error(f"Error flushing batches to database: {e}. Discarding failed batches to prevent memory leak.")
            for batch_key in batches_to_flush:
                if self._batches.get(batch_key) is not None:
                    self._batches[batch_key] = []

    async def periodic_flusher(self):
        while True:
            await asyncio.sleep(FLUSH_INTERVAL)
            await self._flush_all_batches()

    async def run(self):
        logger.info("Data consumer with batching started.")
        while True:
            try:
                item = await self._data_queue.get()
                source, data = item['source'], item['payload']
                now = time.time()
                if source == 'spot_ws': app_state.last_spot_ws_message_time = now
                elif source == 'futures_ws': app_state.last_futures_ws_message_time = now
                elif source == 'open_interest': app_state.last_open_interest_update_time = now
                elif source == 'depth_snapshot': app_state.last_depth_snapshot_update_time = now

                stream_type = data.get('e') or data.get('type')

                # --- Alerting Logic ---
                if self._alerter and settings.enable_liquidation_alerts:
                    if stream_type == 'markPriceUpdate':
                        asyncio.create_task(self._alerter.update_current_price(data))
                    elif stream_type == 'forceOrder':
                        asyncio.create_task(self._alerter.check_liquidation(data))

                # --- Batching Logic ---
                record, batch_key = None, None
                if stream_type == 'aggTrade': record, batch_key = db.prepare_agg_trade(data), 'agg_trade'
                elif stream_type == 'depthUpdate': record, batch_key = db.prepare_depth_update(data), 'depth_update'
                elif stream_type == 'markPriceUpdate': record, batch_key = db.prepare_mark_price(data), 'mark_price'
                elif stream_type == 'forceOrder':
                    symbol = data.get('o', {}).get('s')
                    if symbol and symbol.lower() in [s.lower() for s in settings.futures_symbols]:
                        record, batch_key = db.prepare_force_order(data), 'force_order'
                elif stream_type == 'openInterest': record, batch_key = db.prepare_open_interest(data), 'open_interest'
                elif stream_type == 'depthSnapshot': record, batch_key = db.prepare_depth_snapshot(data), 'depth_snapshot'

                if record and batch_key:
                    self._batches[batch_key].append(record)
                    if len(self._batches[batch_key]) >= BATCH_SIZE:
                        await self._flush_all_batches()

                self._data_queue.task_done()
            except Exception as e:
                logger.error(f"Error in data consumer loop: {e}", exc_info=True)


class Service:
    def __init__(self):
        self.tasks = []
        self.shutdown_event = asyncio.Event()
        self.server = None
        self.http_session = None

    async def shutdown(self, sig):
        if self.shutdown_event.is_set(): return
        logger.warning(f"Received exit signal {sig.name}... Shutting down.")
        self.shutdown_event.set()

        if self.server:
            self.server.should_exit = True

        logger.info("Cancelling all application tasks...")
        for task in self.tasks:
            if task is not asyncio.current_task():
                task.cancel()

        await asyncio.gather(*self.tasks, return_exceptions=True)

        logger.info("Closing client connections...")
        if self.http_session and not self.http_session.closed:
            await self.http_session.close()
        await db.close()

        logger.info("Application shutdown complete.")

    async def monitor_tasks(self, data_queue: Queue):
        """Monitors all running tasks and the data queue size."""
        while not self.shutdown_event.is_set():
            await asyncio.sleep(60)
            logger.info(f"Data queue size: {data_queue.qsize()}")

            for task in self.tasks:
                if task.done() and not task.cancelled():
                    try:
                        exception = task.exception()
                        if exception:
                            task_name = task.get_name()
                            logger.critical(
                                f"Task '{task_name}' has crashed with an exception: {exception}",
                                exc_info=exception
                            )
                            if not self.shutdown_event.is_set():
                                logger.critical("Initiating service shutdown due to critical task failure.")
                                asyncio.create_task(self.shutdown(signal.SIGABRT))
                    except asyncio.CancelledError:
                        logger.warning(f"Task '{task.get_name()}' was cancelled.")
                    except Exception as e:
                        logger.error(f"Error in monitor task itself: {e}")

    async def run(self):
        logger.info("Starting data collector application")
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, lambda s=sig: asyncio.create_task(self.shutdown(s)))

        try:
            await db.init_db()
            app_state.db_connected = True
        except Exception as e:
            logger.critical(f"Failed to initialize database. Shutting down. Error: {e}")
            return

        data_queue = Queue()

        # --- Setup Alerter ---
        alerter = None
        if settings.enable_liquidation_alerts:
            logger.info("Liquidation alerts are enabled. Initializing alerter.")
            alerter = LiquidationAlerter(db)
        else:
            logger.info("Liquidation alerts are disabled.")

        async with aiohttp.ClientSession() as self.http_session:
            # --- Setup Clients and Consumer ---
            spot_ws_streams = [f"{s}@{st}" for s in settings.spot_symbols for st in settings.spot_streams]
            futures_ws_streams = [f"{s}@{st}" for s in settings.futures_symbols for st in ['aggTrade', 'depth@500ms']] + ["!markPrice@arr@1s", "!forceOrder@arr"]

            spot_ws_client = WebsocketClient(self.http_session, settings.spot_ws_base_url, spot_ws_streams, data_queue, source_name="spot_ws")
            futures_ws_client = WebsocketClient(self.http_session, settings.futures_ws_base_url, futures_ws_streams, data_queue, source_name="futures_ws")
            rest_client = RestClient(self.http_session, settings.spot_symbols, settings.futures_symbols, data_queue)
            consumer = BatchingDataConsumer(data_queue, alerter=alerter)

            # --- Setup Web Server ---
            uvicorn_config = uvicorn.Config(app, host="0.0.0.0", port=8000, log_level="info")
            self.server = uvicorn.Server(uvicorn_config)

            # --- Create and schedule tasks with names ---
            task_configs = {
                "spot_websocket_client": (spot_ws_client.run, settings.enable_websocket_spot),
                "futures_websocket_client": (futures_ws_client.run, settings.enable_websocket_futures),
                "open_interest_fetcher": (rest_client.run_open_interest_fetcher, settings.enable_open_interest),
                "depth_snapshot_fetcher": (rest_client.run_depth_snapshot_fetcher, settings.enable_depth_snapshot),
                "data_consumer": (consumer.run, True),
                "periodic_flusher": (consumer.periodic_flusher, True),
                "web_server": (self.server.serve, True),
                "task_monitor": (self.monitor_tasks, True, data_queue),
            }

            # Add alerter task if enabled
            if alerter and settings.enable_liquidation_alerts:
                task_configs["alerter_price_updater"] = (alerter.update_historical_max_prices, True)

            for name, config in task_configs.items():
                coro, enabled, *args = config
                if enabled:
                    self.tasks.append(asyncio.create_task(coro(*args), name=name))

            logger.info("All components are running.")
            await self.shutdown_event.wait()


if __name__ == "__main__":
    service = Service()
    try:
        asyncio.run(service.run())
    except (asyncio.CancelledError, KeyboardInterrupt):
        logger.info("Main task cancelled. Application shutting down.")
