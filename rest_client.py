import asyncio
import logging
import aiohttp
from datetime import datetime

from config import settings

logger = logging.getLogger(__name__)

class RestClient:
    def __init__(self, spot_symbols, futures_symbols, data_queue):
        self._spot_symbols = spot_symbols
        self._futures_symbols = futures_symbols
        self._data_queue = data_queue
        self._oi_poll_interval = settings.open_interest_poll_interval
        self._depth_poll_interval = settings.depth_snapshot_poll_interval
        self._depth_request_delay = settings.depth_snapshot_request_delay
        self._session = None

    async def _create_session(self):
        if not self._session or self._session.closed:
            self._session = aiohttp.ClientSession()
            logger.info("aiohttp client session created.")

    async def _close_session(self):
        if self._session:
            await self._session.close()
            logger.info("aiohttp client session closed.")

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
        await self._create_session()
        url = f"{settings.futures_api_base_url}/fapi/v1/openInterest"
        while True:
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
            await asyncio.sleep(self._oi_poll_interval)

    async def run_depth_snapshot_fetcher(self):
        """Periodically fetches depth snapshots for all symbols, one by one, with a delay."""
        await self._create_session()
        spot_url = f"{settings.spot_api_base_url}/api/v3/depth"
        futures_url = f"{settings.futures_api_base_url}/fapi/v1/depth"

        # Create a unified list of symbols with their market type and url
        all_symbols = []
        for symbol in self._spot_symbols:
            all_symbols.append({'symbol': symbol.upper(), 'market_type': 'spot', 'url': spot_url})
        for symbol in self._futures_symbols:
            all_symbols.append({'symbol': symbol.upper(), 'market_type': 'futures', 'url': futures_url})

        while True:
            logger.info("Starting new depth snapshot fetch cycle for all symbols.")
            for symbol_info in all_symbols:
                symbol = symbol_info['symbol']
                market_type = symbol_info['market_type']
                url = symbol_info['url']

                logger.info(f"Fetching depth snapshot for {symbol} ({market_type})...")
                payload = await self._get(url, {'symbol': symbol, 'limit': 1000})

                if payload:
                    snapshot_data = {
                        'source': 'depth_snapshot',
                        'payload': {
                            'type': 'depthSnapshot',
                            'symbol': symbol,
                            'market_type': market_type,
                            'timestamp': int(datetime.utcnow().timestamp() * 1000),
                            'payload': payload
                        }
                    }
                    await self._data_queue.put(snapshot_data)
                    logger.info(f"Successfully queued depth snapshot for {symbol}.")
                else:
                    logger.warning(f"Failed to fetch or process depth snapshot for {symbol}.")

                # Wait before fetching the next symbol
                logger.debug(f"Waiting for {self._depth_request_delay} seconds before next symbol.")
                await asyncio.sleep(self._depth_request_delay)

            logger.info(f"Depth snapshot fetch cycle complete. Waiting for {self._depth_poll_interval} seconds before next cycle.")
            await asyncio.sleep(self._depth_poll_interval)


    async def close(self):
        """Closes the client session."""
        await self._close_session()
