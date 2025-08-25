import asyncpg
import json
import logging
from datetime import datetime

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

    async def execute_script(self, script):
        """Helper to execute a SQL script."""
        async with self._pool.acquire() as conn:
            await conn.execute(script)

    async def init_db(self):
        """Initializes the database, creates tables and hypertables."""
        await self.connect()

        # Enable the TimescaleDB extension
        await self.execute_script("CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;")

        # Table creation scripts
        scripts = [
            """
            CREATE TABLE IF NOT EXISTS agg_trades (
                event_time          TIMESTAMPTZ       NOT NULL,
                symbol              TEXT              NOT NULL,
                aggregate_trade_id  BIGINT            NOT NULL,
                payload             JSONB             NOT NULL,
                PRIMARY KEY (symbol, aggregate_trade_id)
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
                PRIMARY KEY (symbol, final_update_id)
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
                symbol              TEXT              NOT NULL,
                order_id            BIGINT            NOT NULL,
                payload             JSONB             NOT NULL,
                PRIMARY KEY (symbol, order_id)
            );
            """,
            "SELECT create_hypertable('force_orders', 'event_time', if_not_exists => TRUE);",
            """
            CREATE TABLE IF NOT EXISTS open_interest (
                time                TIMESTAMPTZ       NOT NULL,
                symbol              TEXT              NOT NULL,
                open_interest       BIGINT            NOT NULL,
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
                await self.execute_script(script)
            except asyncpg.exceptions.PostgresError as e:
                # Ignore "table already exists" or "is already a hypertable" errors
                if "already exists" in str(e) or "is already a hypertable" in str(e):
                    logger.warning(f"Warning during table creation: {e}")
                    pass
                else:
                    raise
        logger.info("Database initialization complete.")

    async def insert_agg_trade(self, data):
        """Inserts an aggregate trade message into the database."""
        sql = """
            INSERT INTO agg_trades (event_time, symbol, aggregate_trade_id, payload)
            VALUES ($1, $2, $3, $4)
            ON CONFLICT (symbol, aggregate_trade_id) DO NOTHING;
        """
        event_time = datetime.fromtimestamp(data['E'] / 1000.0)
        await self._pool.execute(sql, event_time, data['s'], data['a'], json.dumps(data))

    async def insert_depth_update(self, data):
        """Inserts a depth update message into the database."""
        sql = """
            INSERT INTO depth_updates (event_time, symbol, first_update_id, final_update_id, payload)
            VALUES ($1, $2, $3, $4, $5)
            ON CONFLICT (symbol, final_update_id) DO NOTHING;
        """
        event_time = datetime.fromtimestamp(data['E'] / 1000.0)
        await self._pool.execute(sql, event_time, data['s'], data['U'], data['u'], json.dumps(data))

    async def insert_mark_price(self, data):
        """Inserts a mark price update message into the database."""
        sql = """
            INSERT INTO mark_prices (event_time, symbol, payload)
            VALUES ($1, $2, $3)
            ON CONFLICT (symbol, event_time) DO NOTHING;
        """
        event_time = datetime.fromtimestamp(data['E'] / 1000.0)
        await self._pool.execute(sql, event_time, data['s'], json.dumps(data))

    async def insert_force_order(self, data):
        """Inserts a liquidation order message into the database."""
        sql = """
            INSERT INTO force_orders (event_time, symbol, order_id, payload)
            VALUES ($1, $2, $3, $4)
            ON CONFLICT (symbol, order_id) DO NOTHING;
        """
        order_data = data['o']
        event_time = datetime.fromtimestamp(data['E'] / 1000.0)
        await self._pool.execute(sql, event_time, order_data['s'], order_data['i'], json.dumps(data))

    async def insert_open_interest(self, data):
        """Inserts an open interest data point into the database."""
        sql = """
            INSERT INTO open_interest (time, symbol, open_interest, payload)
            VALUES ($1, $2, $3, $4)
            ON CONFLICT (symbol, time) DO NOTHING;
        """
        # Binance API for open interest returns timestamp in milliseconds
        time = datetime.fromtimestamp(data['timestamp'] / 1000.0)
        await self._pool.execute(sql, time, data['symbol'], int(data['openInterest']), json.dumps(data))

    async def insert_depth_snapshot(self, data):
        """Inserts a depth snapshot message into the database."""
        sql = """
            INSERT INTO depth_snapshots (time, symbol, market_type, payload)
            VALUES ($1, $2, $3, $4)
            ON CONFLICT (symbol, market_type, time) DO NOTHING;
        """
        time = datetime.fromtimestamp(data['timestamp'] / 1000.0)
        await self._pool.execute(sql, time, data['symbol'], data['market_type'], json.dumps(data['payload']))


db = Database(settings.db_dsn)
