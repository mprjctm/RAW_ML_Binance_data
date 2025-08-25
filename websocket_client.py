import asyncio
import json
import logging
import websockets
from websockets.exceptions import ConnectionClosed


logger = logging.getLogger(__name__)

class WebsocketClient:
    def __init__(self, url, streams, data_queue):
        self._url = url
        self._streams = streams
        self._data_queue = data_queue
        self._connection = None
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
                message = await self._connection.recv()
                data = json.loads(message)

                if 'e' in data: # Most streams have an 'e' for event type
                    await self._data_queue.put(data)
                elif 'stream' in data: # For composite streams
                    await self._data_queue.put(data['data'])
                # Handle ping/pong
                elif 'ping' in data:
                     await self._connection.send(json.dumps({"pong": data['ping']}))
                elif self._connection.is_closed:
                    raise ConnectionClosed(None, None)

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
