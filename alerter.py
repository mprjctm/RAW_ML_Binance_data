import asyncio
import logging
from decimal import Decimal, InvalidOperation

from config import settings
from database import Database

# Get the dedicated alert logger configured in main.py
alert_logger = logging.getLogger('alerter')
# Get the main logger for general component logging
logger = logging.getLogger(__name__)

class LiquidationAlerter:
    """
    Monitors liquidations and triggers alerts based on a price drop from the historical maximum.
    """
    def __init__(self, db: Database):
        self._db = db
        self._current_prices = {}  # Cache for current mark prices: {symbol: Decimal(price)}
        self._max_prices = {}      # Cache for historical max prices: {symbol: Decimal(price)}
        self._lookback_days = settings.alert_max_days_lookback
        self._percentage_drop = settings.alert_percentage_drop
        logger.info("LiquidationAlerter initialized.")

    async def update_current_price(self, mark_price_data: dict):
        """Updates the cache with the latest mark price."""
        try:
            symbol = mark_price_data['s']
            price = Decimal(mark_price_data['p'])
            self._current_prices[symbol] = price
        except (KeyError, InvalidOperation) as e:
            logger.warning(f"[Alerter] Could not parse mark price update: {mark_price_data}. Error: {e}")

    async def update_historical_max_prices(self):
        """Periodically fetches and caches the historical max price for all symbols."""
        while True:
            logger.info("[Alerter] Starting historical max price update cycle...")
            symbols_to_track = settings.futures_symbols
            for symbol in symbols_to_track:
                try:
                    max_price = await self._db.get_max_price_in_period(symbol, self._lookback_days)
                    if max_price is not None:
                        self._max_prices[symbol] = Decimal(max_price)
                        logger.info(f"[Alerter] Updated max price for {symbol}: {self._max_prices[symbol]}")
                    else:
                        logger.warning(f"[Alerter] No historical max price found for {symbol} in the last {self._lookback_days} days.")
                except Exception as e:
                    logger.error(f"[Alerter] Failed to update max price for {symbol}: {e}", exc_info=True)
                await asyncio.sleep(1) # Small sleep to avoid bursting the DB

            logger.info("[Alerter] Historical max price update cycle complete. Waiting for 1 hour.")
            await asyncio.sleep(3600) # Update every hour

    async def check_liquidation(self, force_order_data: dict):
        """Checks if a liquidation event meets the alert criteria."""
        try:
            symbol = force_order_data.get('o', {}).get('s')
            if not symbol:
                return

            current_price = self._current_prices.get(symbol)
            historical_max_price = self._max_prices.get(symbol)

            # If we don't have the necessary price data, we can't check.
            if current_price is None or historical_max_price is None:
                return

            # Calculate the alert threshold
            threshold_price = historical_max_price * (Decimal(1) - (Decimal(self._percentage_drop) / Decimal(100)))

            if current_price < threshold_price:
                # Condition met, log the alert
                side = force_order_data.get('o', {}).get('S')
                quantity = force_order_data.get('o', {}).get('q')

                alert_message = (
                    f"LIQUIDATION ALERT for {symbol}: "
                    f"Side: {side}, Quantity: {quantity}. "
                    f"Current Price ({current_price:.4f}) is below the threshold ({threshold_price:.4f}). "
                    f"Historical {self._lookback_days}-day max price was {historical_max_price:.4f}."
                )
                alert_logger.info(alert_message)

        except Exception as e:
            logger.error(f"[Alerter] Error checking liquidation: {e}", exc_info=True)
