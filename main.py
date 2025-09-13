# -*- coding: utf-8 -*-
"""
Это главный файл приложения-сборщика данных. Он отвечает за инициализацию,
запуск и корректное завершение всех компонентов системы.

Архитектура приложения построена на основе `asyncio` и следует шаблону
"Производитель-Потребитель" (Producer-Consumer) с несколькими производителями
и одним потребителем.

Компоненты:
- Производители (Producers):
  - `WebsocketClient`: Подключается к WebSocket потокам Binance и получает
    данные в реальном времени.
  - `RestClient`: Периодически опрашивает REST API эндпоинты Binance.
  - Все производители кладут полученные данные в единую асинхронную очередь `data_queue`.

- Потребитель (Consumer):
  - `BatchingDataConsumer`: Забирает данные из `data_queue`, группирует их
    в пачки (батчи) по типу и периодически записывает в базу данных.
    Это позволяет значительно снизить нагрузку на БД.

- Оркестратор и наблюдатель:
  - `Service`: Основной класс, который управляет жизненным циклом всех
    компонентов. Он запускает их как асинхронные задачи `asyncio.Task`.
  - `monitor_tasks`: Асинхронная задача-"супервизор", которая следит за
    "здоровьем" всех остальных задач. В случае падения одной из них,
    `monitor_tasks` инициирует корректное завершение всего приложения.

- Веб-сервер:
  - `web_server`: Легковесный `uvicorn` сервер с эндпоинтами `/health` и
    `/metrics` для внешнего мониторинга.
"""
import asyncio
import logging
import signal
import time
from asyncio import Queue
from collections import defaultdict

import aiohttp
import uvicorn

from config import settings
from database import db
from rest_client import RestClient
from state import app_state
from web_server import app, MESSAGES_PROCESSED_COUNTER
from websocket_client import WebsocketClient

# --- Конфигурация пакетной записи в БД ---
BATCH_SIZE = 1000  # Максимальный размер пачки для одного типа данных
FLUSH_INTERVAL = 5.0  # Максимальный интервал в секундах для сброса пачек в БД
WS_STREAM_CHUNK_SIZE = 200  # Макс. кол-во подписок на одном WebSocket соединении


def setup_logging():
    """Настраивает логирование в консоль и файлы."""
    root_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    # Логирование всех сообщений уровня INFO и выше в system.log
    system_handler = logging.FileHandler("system.log")
    system_handler.setLevel(logging.INFO)
    system_handler.setFormatter(root_formatter)
    root_logger.addHandler(system_handler)

    # Логирование только ошибок в error.log
    error_handler = logging.FileHandler("error.log")
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(root_formatter)
    root_logger.addHandler(error_handler)

    # Вывод в консоль
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(root_formatter)
    root_logger.addHandler(console_handler)

setup_logging()
logger = logging.getLogger(__name__)


class BatchingDataConsumer:
    """
    Потребитель данных, который накапливает сообщения в пачки (батчи)
    и периодически записывает их в базу данных.
    """
    def __init__(self, data_queue: Queue):
        self._data_queue = data_queue
        # Словарь для хранения пачек. Ключ - тип данных (напр., 'agg_trade'),
        # значение - список записей.
        self._batches = defaultdict(list)
        self._last_flush_time = time.time()

    async def _flush_all_batches(self):
        """Записывает все непустые пачки в базу данных."""
        flush_tasks = []
        # Создаем копию словаря, чтобы избежать проблем при асинхронном доступе
        batches_to_flush = {k: v for k, v in self._batches.items() if v}
        if not batches_to_flush:
            return

        logger.info(f"Flushing {len(batches_to_flush)} non-empty batches...")
        # Для каждого типа данных вызываем соответствующий метод для пакетной вставки
        if batches_to_flush.get('agg_trade'): flush_tasks.append(db.batch_insert_agg_trades(batches_to_flush['agg_trade']))
        if batches_to_flush.get('depth_update'): flush_tasks.append(db.batch_insert_depth_updates(batches_to_flush['depth_update']))
        if batches_to_flush.get('mark_price'): flush_tasks.append(db.batch_insert_mark_prices(batches_to_flush['mark_price']))
        if batches_to_flush.get('force_order'): flush_tasks.append(db.batch_insert_force_orders(batches_to_flush['force_order']))
        if batches_to_flush.get('open_interest'): flush_tasks.append(db.batch_insert_open_interest(batches_to_flush['open_interest']))
        if batches_to_flush.get('depth_snapshot'): flush_tasks.append(db.batch_insert_depth_snapshots(batches_to_flush['depth_snapshot']))

        try:
            # `asyncio.gather` выполняет все задачи записи в БД параллельно
            await asyncio.gather(*flush_tasks)
            # Если запись успешна, очищаем соответствующие пачки
            for batch_key, batch_list in batches_to_flush.items():
                MESSAGES_PROCESSED_COUNTER.labels(stream_type=batch_key).inc(len(batch_list))
                self._batches[batch_key] = []
            self._last_flush_time = time.time()
            logger.info(f"Flushed {len(flush_tasks)} batches successfully.")
        except Exception as e:
            logger.error(f"Error flushing batches to database: {e}. Discarding failed batches to prevent memory leak.")
            # В случае ошибки записи, отбрасываем "плохую" пачку, чтобы не переполнять память
            for batch_key in batches_to_flush:
                if self._batches.get(batch_key) is not None:
                    self._batches[batch_key] = []

    async def periodic_flusher(self):
        """Задача, которая принудительно сбрасывает пачки по таймеру."""
        while True:
            await asyncio.sleep(FLUSH_INTERVAL)
            await self._flush_all_batches()

    async def run(self):
        """Основной цикл потребителя: получает элемент из очереди и добавляет в пачку."""
        logger.info("Data consumer with batching started.")
        while True:
            try:
                # Асинхронно ждем новый элемент в очереди
                item = await self._data_queue.get()
                source, data = item['source'], item['payload']

                # Обновляем время последнего сообщения для системы health-check
                now = time.time()
                if source == 'spot_ws': app_state.last_spot_ws_message_time = now
                elif source == 'futures_ws': app_state.last_futures_ws_message_time = now
                elif source == 'open_interest': app_state.last_open_interest_update_time = now
                elif source == 'depth_snapshot': app_state.last_depth_snapshot_update_time = now

                stream_type = data.get('e') or data.get('type')

                # --- Логика добавления в пачку ---
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
                    # Если пачка заполнилась, немедленно сбрасываем ее в БД
                    if len(self._batches[batch_key]) >= BATCH_SIZE:
                        await self._flush_all_batches()

                self._data_queue.task_done()
            except Exception as e:
                logger.error(f"Error in data consumer loop: {e}", exc_info=True)


class Service:
    """
    Класс-оркестратор. Управляет жизненным циклом приложения.
    """
    def __init__(self):
        self.tasks = []  # Список всех запущенных асинхронных задач
        self.shutdown_event = asyncio.Event()  # Событие для координации завершения работы
        self.server = None
        self.http_session = None

    async def shutdown(self, sig):
        """Обрабатывает сигнал завершения (Ctrl+C) и корректно останавливает сервис."""
        if self.shutdown_event.is_set(): return
        logger.warning(f"Received exit signal {sig.name}... Shutting down.")
        self.shutdown_event.set()

        if self.server:
            self.server.should_exit = True

        logger.info("Cancelling all application tasks...")
        # Отменяем все запущенные задачи
        for task in self.tasks:
            if task is not asyncio.current_task():
                task.cancel()

        # Ждем завершения всех задач (включая отмененные)
        await asyncio.gather(*self.tasks, return_exceptions=True)

        logger.info("Closing client connections...")
        if self.http_session and not self.http_session.closed:
            await self.http_session.close()
        await db.close()

        logger.info("Application shutdown complete.")

    async def monitor_tasks(self, data_queue: Queue):
        """
        Задача-"супервизор". Периодически проверяет состояние других задач.
        Если какая-либо задача падает с ошибкой, инициирует общую остановку.
        """
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
                            # Инициируем аварийное завершение работы
                            if not self.shutdown_event.is_set():
                                logger.critical("Initiating service shutdown due to critical task failure.")
                                asyncio.create_task(self.shutdown(signal.SIGABRT))
                    except asyncio.CancelledError:
                        logger.warning(f"Task '{task.get_name()}' was cancelled.")
                    except Exception as e:
                        logger.error(f"Error in monitor task itself: {e}")

    async def run(self):
        """Главный метод, который запускает все компоненты приложения."""
        logger.info("Starting data collector application")
        loop = asyncio.get_running_loop()
        # Устанавливаем обработчики сигналов для корректного завершения
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, lambda s=sig: asyncio.create_task(self.shutdown(s)))

        # --- Инициализация ---
        try:
            await db.init_db()
            app_state.db_connected = True
        except Exception as e:
            logger.critical(f"Failed to initialize database. Shutting down. Error: {e}")
            return

        data_queue = Queue()

        def chunk_list(lst, n):
            """Вспомогательная функция для разбиения списка на части (чанки)."""
            for i in range(0, len(lst), n):
                yield lst[i:i + n]

        async with aiohttp.ClientSession() as self.http_session:
            # --- Создание клиентов и потребителя ---
            # Формируем списки подписок для WebSocket
            spot_ws_streams = [f"{s}@{st}" for s in settings.spot_symbols for st in settings.spot_streams]
            futures_symbol_streams = [f"{s}@{st}" for s in settings.futures_symbols for st in ['aggTrade', 'depth@500ms']]
            futures_general_streams = ["!markPrice@arr@1s", "!forceOrder@arr"]

            # Разбиваем списки на чанки, чтобы не превышать лимит Binance на кол-во
            # подписок на одном соединении.
            spot_stream_chunks = list(chunk_list(spot_ws_streams, WS_STREAM_CHUNK_SIZE))
            futures_stream_chunks = list(chunk_list(futures_symbol_streams, WS_STREAM_CHUNK_SIZE))

            # Добавляем общие подписки (не привязанные к символу) к первому чанку
            if futures_stream_chunks:
                if len(futures_stream_chunks[0]) + len(futures_general_streams) <= WS_STREAM_CHUNK_SIZE:
                    futures_stream_chunks[0].extend(futures_general_streams)
                else:
                    futures_stream_chunks.append(futures_general_streams)
            elif futures_general_streams:
                 futures_stream_chunks.append(futures_general_streams)

            rest_client = RestClient(self.http_session, settings.spot_symbols, settings.futures_symbols, data_queue)
            consumer = BatchingDataConsumer(data_queue)

            # --- Настройка веб-сервера для метрик ---
            uvicorn_config = uvicorn.Config(app, host="0.0.0.0", port=8000, log_level="info")
            self.server = uvicorn.Server(uvicorn_config)

            # --- Создание и запуск всех асинхронных задач ---
            task_configs = {
                "open_interest_fetcher": (rest_client.run_open_interest_fetcher, settings.enable_open_interest),
                "depth_snapshot_fetcher": (rest_client.run_depth_snapshot_fetcher, settings.enable_depth_snapshot),
                "data_consumer": (consumer.run, True),
                "periodic_flusher": (consumer.periodic_flusher, True),
                "web_server": (self.server.serve, True),
                "task_monitor": (self.monitor_tasks, True, data_queue),
            }

            # Создаем по одной задаче для каждого чанка WebSocket подписок
            for i, chunk in enumerate(spot_stream_chunks):
                client = WebsocketClient(self.http_session, settings.spot_ws_base_url, chunk, data_queue, source_name=f"spot_ws_{i+1}")
                task_configs[f"spot_websocket_client_{i+1}"] = (client.run, settings.enable_websocket_spot)

            for i, chunk in enumerate(futures_stream_chunks):
                client = WebsocketClient(self.http_session, settings.futures_ws_base_url, chunk, data_queue, source_name=f"futures_ws_{i+1}")
                task_configs[f"futures_websocket_client_{i+1}"] = (client.run, settings.enable_websocket_futures)

            # Запускаем все задачи, которые включены в конфигурации
            for name, config in task_configs.items():
                coro, enabled, *args = config
                if enabled:
                    self.tasks.append(asyncio.create_task(coro(*args), name=name))

            logger.info(f"All components are running. Spot clients: {len(spot_stream_chunks)}, Futures clients: {len(futures_stream_chunks)}")
            # Ожидаем события о завершении работы
            await self.shutdown_event.wait()


if __name__ == "__main__":
    service = Service()
    try:
        asyncio.run(service.run())
    except (asyncio.CancelledError, KeyboardInterrupt):
        logger.info("Main task cancelled. Application shutting down.")
