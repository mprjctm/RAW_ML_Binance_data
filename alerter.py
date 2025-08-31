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
    Monitors liquidations and triggers alerts when the price is near N-day extremums.
    """
    def __init__(self, db: Database):
        self._db = db
        self._current_prices = {}  # Cache for current mark prices: {symbol: Decimal(price)}
        self._min_prices = {}      # Cache for historical min prices: {symbol: Decimal(price)}
        self._max_prices = {}      # Cache for historical max prices: {symbol: Decimal(price)}
        self._lookback_days = settings.alert_max_days_lookback
        self._deviation_percent = settings.alert_extremum_deviation_percent
        logger.info("LiquidationAlerter initialized.")

    async def update_current_price(self, mark_price_data: dict):
        """Updates the cache with the latest mark price."""
        try:
            symbol = mark_price_data['s']
            price = Decimal(mark_price_data['p'])
            self._current_prices[symbol] = price
        except (KeyError, InvalidOperation) as e:
            logger.warning(f"[Alerter] Could not parse mark price update: {mark_price_data}. Error: {e}")

    async def update_historical_extremum_prices(self):
        """Periodically fetches and caches the historical min/max prices for all symbols."""
        while True:
            logger.info("[Alerter] Starting historical extremum prices update cycle...")
            symbols_to_track = settings.futures_symbols
            for symbol in symbols_to_track:
                try:
                    # Fetch both min and max prices
                    min_price = await self._db.get_min_price_in_period(symbol, self._lookback_days)
                    max_price = await self._db.get_max_price_in_period(symbol, self._lookback_days)

                    if min_price is not None:
                        self._min_prices[symbol] = Decimal(min_price)
                        logger.info(f"[Alerter] Updated min price for {symbol}: {self._min_prices[symbol]}")
                    else:
                        logger.warning(f"[Alerter] No historical min price found for {symbol} in the last {self._lookback_days} days.")

                    if max_price is not None:
                        self._max_prices[symbol] = Decimal(max_price)
                        logger.info(f"[Alerter] Updated max price for {symbol}: {self._max_prices[symbol]}")
                    else:
                        logger.warning(f"[Alerter] No historical max price found for {symbol} in the last {self._lookback_days} days.")

                except Exception as e:
                    logger.error(f"[Alerter] Failed to update extremum prices for {symbol}: {e}", exc_info=True)

                await asyncio.sleep(1) # Small sleep to avoid bursting the DB with requests

            logger.info("[Alerter] Historical extremum price update cycle complete. Waiting for 1 hour.")
            await asyncio.sleep(3600) # Update every hour

    async def check_liquidation(self, force_order_data: dict):
        """Checks if a liquidation event meets the alert criteria (near min or max)."""
        try:
            symbol = force_order_data.get('o', {}).get('s')
            if not symbol:
                return

            current_price = self._current_prices.get(symbol)
            min_price = self._min_prices.get(symbol)
            max_price = self._max_prices.get(symbol)

            # If we don't have all necessary price data, we can't check.
            if current_price is None or min_price is None or max_price is None:
                return

            # --- Calculate Alert Zones ---
            deviation_factor = Decimal(self._deviation_percent) / Decimal(100)

            # Zone near minimum: [min_price, min_price * (1 + X)]
            low_zone_upper_bound = min_price * (Decimal(1) + deviation_factor)

            # Zone near maximum: [max_price * (1 - X), max_price]
            high_zone_lower_bound = max_price * (Decimal(1) - deviation_factor)

            alert_reason = None
            if current_price <= low_zone_upper_bound:
                alert_reason = f"near {self._lookback_days}-day low of {min_price:.4f}"
            elif current_price >= high_zone_lower_bound:
                alert_reason = f"near {self._lookback_days}-day high of {max_price:.4f}"

            if alert_reason:
                # Condition met, log the alert
                side = force_order_data.get('o', {}).get('S')
                quantity = force_order_data.get('o', {}).get('q')

                alert_message = (
                    f"LIQUIDATION ALERT for {symbol}: "
                    f"Side: {side}, Quantity: {quantity}. "
                    f"Price ({current_price:.4f}) is {alert_reason}."
                )
                alert_logger.info(alert_message)

        except Exception as e:
            logger.error(f"[Alerter] Error checking liquidation: {e}", exc_info=True)
