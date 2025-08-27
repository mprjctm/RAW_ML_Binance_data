import asyncio
import orjson as json
import logging
import time
import aiohttp

logger = logging.getLogger(__name__)

class WebsocketClient:
    def __init__(self, session: aiohttp.ClientSession, url: str, streams: list, data_queue: asyncio.Queue, source_name: str):
        self._session = session
        self._url = url
        self._streams = streams
        self._data_queue = data_queue
        self._source_name = source_name
        self._reconnect_delay = 1

    async def run(self):
        """The main loop to connect, listen, and handle reconnections."""
        logger.info(f"[{self._source_name}] Starting WebSocket client for {self._url}")
        while True:
            try:
                async with self._session.ws_connect(self._url, timeout=30) as ws:
                    logger.info(f"[{self._source_name}] WebSocket connection established.")
                    self._reconnect_delay = 1  # Reset reconnect delay on successful connection

                    # Subscribe to streams
                    subscription_payload = {
                        "method": "SUBSCRIBE",
                        "params": self._streams,
                        "id": 1
                    }
                    await ws.send_json(subscription_payload)
                    logger.info(f"[{self._source_name}] Subscription request sent for streams: {self._streams}")

                    # Listen for messages
                    async for msg in ws:
                        if msg.type == aiohttp.WSMsgType.TEXT:
                            payload = json.loads(msg.data)

                            if isinstance(payload, list):
                                for item in payload:
                                    if 'e' in item:
                                        await self._data_queue.put({"source": self._source_name, "payload": item})
                            elif isinstance(payload, dict):
                                if 'e' in payload:
                                    await self._data_queue.put({"source": self._source_name, "payload": payload})
                                elif "result" in payload:
                                     logger.info(f"[{self._source_name}] Subscription response: {payload}")

                        elif msg.type == aiohttp.WSMsgType.ERROR:
                            logger.error(f"[{self._source_name}] WebSocket connection closed with error: {ws.exception()}")
                            break # Break inner loop to trigger reconnect

                        elif msg.type == aiohttp.WSMsgType.CLOSED:
                            logger.warning(f"[{self._source_name}] WebSocket connection closed by server.")
                            break # Break inner loop to trigger reconnect

            except aiohttp.ClientError as e:
                logger.error(f"[{self._source_name}] WebSocket connection error: {e}. Attempting to reconnect.")
            except asyncio.TimeoutError:
                logger.warning(f"[{self._source_name}] WebSocket connection timed out. Attempting to reconnect.")
            except Exception as e:
                logger.error(f"[{self._source_name}] An unexpected error occurred in WebSocket client: {e}")

            logger.info(f"[{self._source_name}] Reconnecting in {self._reconnect_delay} seconds...")
            await asyncio.sleep(self._reconnect_delay)
            self._reconnect_delay = min(self._reconnect_delay * 2, 60) # Exponential backoff
