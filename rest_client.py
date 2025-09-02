import asyncio
import logging
import aiohttp
from datetime import datetime

from config import settings

logger = logging.getLogger(__name__)

class RestClient:
    def __init__(self, session: aiohttp.ClientSession, spot_symbols, futures_symbols, data_queue):
        self._session = session
        self._spot_symbols = spot_symbols
        self._futures_symbols = futures_symbols
        self._data_queue = data_queue
        self._oi_poll_interval = settings.open_interest_poll_interval
        self._depth_poll_interval = settings.depth_snapshot_poll_interval

    async def _get(self, url, params=None):
        """Generic GET request helper."""
        try:
            async with self._session.get(url, params=params) as response:
                response.raise_for_status()
                return await response.json()
        except aiohttp.ClientError as e:
            logger.error(f"Error fetching data from {url}: {e}")
            return None
        except Exception as e:
            logger.error(f"An unexpected error occurred while fetching from {url}: {e}")
            return None

    async def run_open_interest_fetcher(self):
        """Periodically fetches open interest for all futures symbols."""
        url = f"{settings.futures_api_base_url}/fapi/v1/openInterest"
        while True:
            try:
                logger.info("Fetching open interest for all future symbols...")
                tasks = [self._get(url, {'symbol': s.upper()}) for s in self._futures_symbols]
                results = await asyncio.gather(*tasks)

                for data in results:
                    if data:
                        data_to_queue = {
                            "source": "open_interest",
                            "payload": {
                                'type': 'openInterest',
                                'timestamp': int(datetime.utcnow().timestamp() * 1000),
                                **data
                            }
                        }
                        await self._data_queue.put(data_to_queue)

                logger.info(f"Open interest fetch cycle complete. Waiting for {self._oi_poll_interval} seconds.")
            except Exception as e:
                logger.error(f"An unexpected error occurred in the open interest fetcher loop: {e}", exc_info=True)
            await asyncio.sleep(self._oi_poll_interval)

    async def run_depth_snapshot_fetcher(self):
        """Periodically fetches depth snapshots for all symbols."""
        spot_url = f"{settings.spot_api_base_url}/api/v3/depth"
        futures_url = f"{settings.futures_api_base_url}/fapi/v1/depth"
        all_symbols = [(s, 'spot') for s in self._spot_symbols] + [(s, 'futures') for s in self._futures_symbols]

        while True:
            try:
                logger.info("Starting depth snapshot fetch cycle for all symbols...")

                for symbol, market_type in all_symbols:
                    try:
                        url = spot_url if market_type == 'spot' else futures_url
                        params = {'symbol': symbol.upper(), 'limit': 1000}

                        logger.info(f"Fetching depth for {symbol} ({market_type})...")
                        payload = await self._get(url, params)

                        if payload:
                            snapshot_data = {
                                'source': 'depth_snapshot',
                                'payload': {
                                    'type': 'depthSnapshot',
                                    'symbol': symbol.upper(),
                                    'market_type': market_type,
                                    'timestamp': int(datetime.utcnow().timestamp() * 1000),
                                    'payload': payload
                                }
                            }
                            await self._data_queue.put(snapshot_data)
                            logger.info(f"Successfully queued depth for {symbol}.")
                        else:
                            logger.warning(f"Did not receive payload for {symbol}.")

                        # Wait for 5 seconds before fetching the next symbol
                        await asyncio.sleep(5)

                    except Exception as e:
                        logger.error(f"Error fetching depth for symbol {symbol}: {e}", exc_info=True)
                        # Optional: decide if you want to sleep even after an error
                        await asyncio.sleep(5)

                logger.info("Depth snapshot fetch cycle complete. Starting new cycle immediately.")
            except Exception as e:
                logger.error(f"An unexpected error occurred in the depth snapshot fetcher loop: {e}", exc_info=True)
                # If a major error occurs in the loop, wait before retrying to prevent a fast failure loop.
                await asyncio.sleep(30)
