import asyncpg
import orjson as json
import logging
from datetime import datetime
from typing import List, Tuple, Any

from config import settings

logger = logging.getLogger(__name__)

class Database:
    def __init__(self, dsn):
        self._dsn = dsn
        self._pool = None

    async def connect(self):
        """Creates a connection pool."""
        if not self._pool:
            try:
                self._pool = await asyncpg.create_pool(self._dsn)
                logger.info("Database connection pool created successfully.")
            except Exception as e:
                logger.error(f"Could not connect to the database: {e}")
                raise

    async def close(self):
        """Closes the connection pool."""
        if self._pool:
            await self._pool.close()
            logger.info("Database connection pool closed.")

    async def init_db(self):
        """Initializes the database, creates tables and hypertables."""
        await self.connect()

        # Enable the TimescaleDB extension
        await self._pool.execute("CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;")

        # --- Schema Migrations ---
        # Run simple migrations to ensure columns exist for backward compatibility.
        # This is idempotent and safe to run on every startup.
        logger.info("Applying database migrations...")
        try:
            await self._pool.execute("ALTER TABLE agg_trades ADD COLUMN IF NOT EXISTS price DECIMAL;")
            await self._pool.execute("ALTER TABLE agg_trades ADD COLUMN IF NOT EXISTS quantity DECIMAL;")
            logger.info("Migrations for agg_trades table applied successfully.")
        except asyncpg.exceptions.UndefinedTableError:
            logger.warning("Migration skipped: agg_trades table does not exist yet. It will be created.")
        except Exception as e:
            logger.error(f"Error applying migrations: {e}")
            raise

        # Table creation scripts
        scripts = [
            """
            CREATE TABLE IF NOT EXISTS agg_trades (
                event_time          TIMESTAMPTZ       NOT NULL,
                symbol              TEXT              NOT NULL,
                aggregate_trade_id  BIGINT            NOT NULL,
                price               DECIMAL           NOT NULL,
                quantity            DECIMAL           NOT NULL,
                payload             JSONB             NOT NULL,
                PRIMARY KEY (symbol, aggregate_trade_id, event_time)
            );
            """,
            "SELECT create_hypertable('agg_trades', 'event_time', if_not_exists => TRUE);",
            """
            CREATE TABLE IF NOT EXISTS depth_updates (
                event_time          TIMESTAMPTZ       NOT NULL,
                symbol              TEXT              NOT NULL,
                first_update_id     BIGINT            NOT NULL,
                final_update_id     BIGINT            NOT NULL,
                payload             JSONB             NOT NULL,
                PRIMARY KEY (symbol, final_update_id, event_time)
            );
            """,
            "SELECT create_hypertable('depth_updates', 'event_time', if_not_exists => TRUE);",
            """
            CREATE TABLE IF NOT EXISTS mark_prices (
                event_time          TIMESTAMPTZ       NOT NULL,
                symbol              TEXT              NOT NULL,
                payload             JSONB             NOT NULL,
                PRIMARY KEY (symbol, event_time)
            );
            """,
            "SELECT create_hypertable('mark_prices', 'event_time', if_not_exists => TRUE);",
            """
            CREATE TABLE IF NOT EXISTS force_orders (
                event_time          TIMESTAMPTZ       NOT NULL,
                transaction_time    TIMESTAMPTZ       NOT NULL,
                symbol              TEXT              NOT NULL,
                side                TEXT              NOT NULL,
                quantity            DECIMAL           NOT NULL,
                payload             JSONB             NOT NULL,
                PRIMARY KEY (symbol, event_time, transaction_time, side, quantity)
            );
            """,
            "SELECT create_hypertable('force_orders', 'event_time', if_not_exists => TRUE);",
            """
            CREATE TABLE IF NOT EXISTS open_interest (
                time                TIMESTAMPTZ       NOT NULL,
                symbol              TEXT              NOT NULL,
                open_interest       DECIMAL           NOT NULL,
                payload             JSONB             NOT NULL,
                PRIMARY KEY (symbol, time)
            );
            """,
            "SELECT create_hypertable('open_interest', 'time', if_not_exists => TRUE);",
            """
            CREATE TABLE IF NOT EXISTS depth_snapshots (
                time                TIMESTAMPTZ       NOT NULL,
                symbol              TEXT              NOT NULL,
                market_type         TEXT              NOT NULL,
                payload             JSONB             NOT NULL,
                PRIMARY KEY (symbol, market_type, time)
            );
            """,
            "SELECT create_hypertable('depth_snapshots', 'time', if_not_exists => TRUE);"
        ]

        logger.info("Creating database tables and hypertables...")
        for script in scripts:
            try:
                await self._pool.execute(script)
            except asyncpg.exceptions.PostgresError as e:
                if "already exists" in str(e) or "is already a hypertable" in str(e):
                    logger.warning(f"Warning during table creation: {e}")
                    pass
                else:
                    raise
        logger.info("Database initialization complete.")

    # --- Batch Insert Methods ---

    async def batch_insert_agg_trades(self, records: List[Tuple]):
        sql = "INSERT INTO agg_trades (event_time, symbol, aggregate_trade_id, price, quantity, payload) VALUES ($1, $2, $3, $4, $5, $6) ON CONFLICT DO NOTHING"
        await self._pool.executemany(sql, records)

    async def batch_insert_depth_updates(self, records: List[Tuple]):
        sql = "INSERT INTO depth_updates (event_time, symbol, first_update_id, final_update_id, payload) VALUES ($1, $2, $3, $4, $5) ON CONFLICT DO NOTHING"
        await self._pool.executemany(sql, records)

    async def batch_insert_mark_prices(self, records: List[Tuple]):
        sql = "INSERT INTO mark_prices (event_time, symbol, payload) VALUES ($1, $2, $3) ON CONFLICT DO NOTHING"
        await self._pool.executemany(sql, records)

    async def batch_insert_force_orders(self, records: List[Tuple]):
        sql = "INSERT INTO force_orders (event_time, transaction_time, symbol, side, quantity, payload) VALUES ($1, $2, $3, $4, $5, $6) ON CONFLICT DO NOTHING"
        await self._pool.executemany(sql, records)

    async def batch_insert_open_interest(self, records: List[Tuple]):
        sql = "INSERT INTO open_interest (time, symbol, open_interest, payload) VALUES ($1, $2, $3, $4) ON CONFLICT DO NOTHING"
        await self._pool.executemany(sql, records)

    async def batch_insert_depth_snapshots(self, records: List[Tuple]):
        sql = "INSERT INTO depth_snapshots (time, symbol, market_type, payload) VALUES ($1, $2, $3, $4) ON CONFLICT DO NOTHING"
        await self._pool.executemany(sql, records)

    # --- Data Preparation Methods ---

    def prepare_agg_trade(self, data: dict) -> Tuple:
        return (
            datetime.fromtimestamp(data['E'] / 1000.0),
            data['s'],
            data['a'],
            float(data['p']),
            float(data['q']),
            json.dumps(data).decode('utf-8')
        )

    def prepare_depth_update(self, data: dict) -> Tuple:
        return (datetime.fromtimestamp(data['E'] / 1000.0), data['s'], data['U'], data['u'], json.dumps(data).decode('utf-8'))

    def prepare_mark_price(self, data: dict) -> Tuple:
        return (datetime.fromtimestamp(data['E'] / 1000.0), data['s'], json.dumps(data).decode('utf-8'))

    def prepare_force_order(self, data: dict) -> Tuple:
        order_data = data.get('o', {})
        required_keys = ['s', 'S', 'q', 'T']
        if not all(key in order_data for key in required_keys):
            logger.warning(f"Received forceOrder message with missing keys. Payload: {data}")
            return None

        return (
            datetime.fromtimestamp(data['E'] / 1000.0),
            datetime.fromtimestamp(order_data['T'] / 1000.0),
            order_data['s'],
            order_data['S'],
            float(order_data['q']),
            json.dumps(data).decode('utf-8')
        )

    def prepare_open_interest(self, data: dict) -> Tuple:
        return (datetime.fromtimestamp(data['timestamp'] / 1000.0), data['symbol'], float(data['openInterest']), json.dumps(data).decode('utf-8'))

    def prepare_depth_snapshot(self, data: dict) -> Tuple:
        # The 'payload' for depth snapshot is already a sub-dictionary
        return (datetime.fromtimestamp(data['timestamp'] / 1000.0), data['symbol'], data['market_type'], json.dumps(data['payload']).decode('utf-8'))


db = Database(settings.db_dsn)
