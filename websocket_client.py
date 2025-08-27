import asyncio
import orjson as json
import logging
import time
import websockets
from websockets.exceptions import ConnectionClosed


logger = logging.getLogger(__name__)

class WebsocketClient:
    def __init__(self, url, streams, data_queue, source_name: str):
        self._url = url
        self._streams = streams
        self._data_queue = data_queue
        self._source_name = source_name  # e.g., 'spot' or 'futures'
        self._connection = None
        self._connection_time = None
        self._reconnect_delay = 1  # start with 1 second

    async def _connect(self):
        """Connects to the WebSocket server and subscribes to streams."""
        logger.info(f"Connecting to WebSocket server at {self._url}")
        try:
            self._connection = await websockets.connect(self._url)
            subscription_payload = {
                "method": "SUBSCRIBE",
                "params": self._streams,
                "id": 1
            }
            await self._connection.send(json.dumps(subscription_payload))
            # The first message is the subscription confirmation
            response = await self._connection.recv()
            logger.info(f"Subscription response: {response}")
            self._reconnect_delay = 1 # reset delay on successful connection
            self._connection_time = time.time() # record connection time
            logger.info(f"Successfully subscribed to streams: {self._streams}")
        except (ConnectionClosed, OSError, websockets.exceptions.InvalidURI) as e:
            logger.error(f"WebSocket connection failed: {e}")
            self._connection = None
        except Exception as e:
            logger.error(f"An unexpected error occurred during connection: {e}")
            self._connection = None


    async def run(self):
        """The main loop to connect, listen, and handle reconnections."""
        while True:
            if not self._connection:
                await self._connect()
                if not self._connection:
                    logger.info(f"Reconnecting in {self._reconnect_delay} seconds...")
                    await asyncio.sleep(self._reconnect_delay)
                    # Exponential backoff
                    self._reconnect_delay = min(self._reconnect_delay * 2, 60)
                    continue

            try:
                # Proactive reconnect after 23 hours
                if self._connection_time and (time.time() - self._connection_time > 23 * 3600):
                    logger.info("Proactively reconnecting websocket after 23 hours.")
                    await self._connection.close()
                    self._connection = None
                    self._connection_time = None
                    continue

                # Wait for a message with a timeout to allow the loop to run checks
                message = await asyncio.wait_for(self._connection.recv(), timeout=60.0)

                payload = json.loads(message)

                if isinstance(payload, list):
                    # Handle array of events (e.g., from !markPrice@arr)
                    for item in payload:
                        if 'e' in item:  # Ensure it's an event
                            await self._data_queue.put({"source": self._source_name, "payload": item})

                elif isinstance(payload, dict):
                    # Handle single event
                    if 'e' in payload:  # It's a data event
                        await self._data_queue.put({"source": self._source_name, "payload": payload})
                    elif 'ping' in payload:
                        await self._connection.send(json.dumps({"pong": payload['ping']}))

            except asyncio.TimeoutError:
                # Timeout is not an error, just a chance to check the connection age
                continue
            except ConnectionClosed as e:
                logger.warning(f"WebSocket connection closed: {e}. Attempting to reconnect.")
                self._connection = None
                await asyncio.sleep(self._reconnect_delay)
            except json.JSONDecodeError:
                logger.warning(f"Could not decode JSON from message: {message}")
            except Exception as e:
                logger.error(f"An unexpected error occurred in the run loop: {e}")
                self._connection = None # Reset connection to trigger reconnect
                await asyncio.sleep(self._reconnect_delay)
                self._reconnect_delay = min(self._reconnect_delay * 2, 60)
