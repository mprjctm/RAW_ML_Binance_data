# -*- coding: utf-8 -*-
"""
Этот скрипт выполняет выгрузку "сырых" рыночных данных из базы данных
PostgreSQL/TimescaleDB в локальные файлы формата Parquet.

Он является первым шагом в пайплайне подготовки данных для машинного обучения.
Скрипт работает в режиме "батчинга", обрабатывая данные по дням, чтобы
минимизировать потребление памяти и нагрузку на базу данных.

**Логика работы:**
1.  **Подключение к БД:** Использует строку подключения `DB_DSN` из файла `.env`.
2.  **Итерация по датам:** Проходит по указанному временному диапазону с заданным шагом (например, 7 дней).
3.  **Запросы к таблицам:** Для каждого временного отрезка выполняет SQL-запросы к таблицам:
    - `agg_trades` (агрегированные сделки)
    - `depth_updates` (обновления стакана)
    - `force_orders` (принудительные ликвидации)
4.  **Обработка данных:**
    - Загружает данные в pandas DataFrame.
    - Раскрывает полезные поля из JSONB-колонки `payload` в отдельные колонки.
    - Приводит типы данных к оптимальным для хранения и анализа.
5.  **Сохранение в Parquet:** Дописывает обработанные данные в итоговые Parquet-файлы.
    Формат именования файлов: `{out_dir}/{symbol}_{data_type}.parquet`.
"""
import argparse
import os
import pandas as pd
import psycopg2
from datetime import datetime, timedelta
from tqdm import tqdm
import json

# Импортируем настройки, включая DSN для подключения к БД
from config import settings

def parse_args():
    """Парсит аргументы командной строки."""
    parser = argparse.ArgumentParser(
        description="Экспорт сырых данных из БД в Parquet-файлы.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--symbol', type=str, required=True, help='Символ для экспорта (например, BTCUSDT).')
    parser.add_argument('--start', type=str, required=True, help='Дата начала экспорта в формате "YYYY-MM-DD HH:MM:SS".')
    parser.add_argument('--end', type=str, required=True, help='Дата окончания экспорта в формате "YYYY-MM-DD HH:MM:SS".')
    parser.add_argument('--out-dir', type=str, default='raw_data', help='Директория для сохранения Parquet-файлов.')
    parser.add_argument('--chunk-size-days', type=int, default=7, help='Размер одного "куска" данных для обработки в днях.')
    return parser.parse_args()

def export_data_for_chunk(conn, query, output_path, process_func):
    """
    Выполняет запрос для одного "куска" данных, обрабатывает и сохраняет его.
    """
    try:

        # Выполняем запрос и загружаем данные в DataFrame
        df = pd.read_sql_query(query, conn)


        if df.empty:
            return

        # Обрабатываем DataFrame (например, раскрываем JSON)
        df = process_func(df)

        # Дописываем данные в Parquet файл
        if os.path.exists(output_path):
            df.to_parquet(output_path, engine='pyarrow', append=True)
        else:
            df.to_parquet(output_path, engine='pyarrow')

    except Exception as e:
        print(f"  - Ошибка при экспорте данных для {output_path}: {e}")

def process_trades(df):
    """Раскрывает JSON `payload` для сделок."""
    payloads = df['payload'].apply(lambda x: json.loads(x) if isinstance(x, str) else x)
    df['price'] = pd.to_numeric(payloads.apply(lambda p: p.get('p')), errors='coerce')
    df['quantity'] = pd.to_numeric(payloads.apply(lambda p: p.get('q')), errors='coerce')
    df['is_buyer_maker'] = payloads.apply(lambda p: p.get('m'))
    df.drop(columns=['payload'], inplace=True)
    df.set_index('event_time', inplace=True)
    return df

def process_depth(df):
    """Раскрывает JSON `payload` для стакана."""
    payloads = df['payload'].apply(lambda x: json.loads(x) if isinstance(x, str) else x)
    df['bids'] = payloads.apply(lambda p: p.get('b'))
    df['asks'] = payloads.apply(lambda p: p.get('a'))
    df.drop(columns=['payload'], inplace=True)
    df.set_index('event_time', inplace=True)
    return df

def process_liquidations(df):
    """Раскрывает JSON `payload` для ликвидаций."""
    payloads = df['payload'].apply(lambda p: json.loads(p.get('o')) if isinstance(p.get('o'), str) else p.get('o'))
    df['side'] = payloads.apply(lambda p: p.get('S'))
    df['price'] = pd.to_numeric(payloads.apply(lambda p: p.get('p')), errors='coerce')
    df['quantity'] = pd.to_numeric(payloads.apply(lambda p: p.get('q')), errors='coerce')
    df.drop(columns=['payload'], inplace=True)
    df.set_index('event_time', inplace=True)
    return df

def main():
    """Основная функция для запуска экспорта."""
    args = parse_args()

    # Создаем выходную директорию, если ее нет
    os.makedirs(args.out_dir, exist_ok=True)

    # Определяем пути к выходным файлам
    trades_out_path = os.path.join(args.out_dir, f"{args.symbol}_trades.parquet")
    depth_out_path = os.path.join(args.out_dir, f"{args.symbol}_depth.parquet")
    liquidations_out_path = os.path.join(args.out_dir, f"{args.symbol}_liquidations.parquet")

    # Удаляем старые файлы, если они существуют, чтобы избежать дубликатов
    for path in [trades_out_path, depth_out_path, liquidations_out_path]:
        if os.path.exists(path):
            os.remove(path)

    start_date = datetime.strptime(args.start, "%Y-%m-%d %H:%M:%S")
    end_date = datetime.strptime(args.end, "%Y-%m-%d %H:%M:%S")

    # Создаем список временных отрезков для обработки
    date_chunks = []
    current_start = start_date
    while current_start < end_date:
        current_end = current_start + timedelta(days=args.chunk_size_days)
        date_chunks.append((current_start, min(current_end, end_date)))
        current_start = current_end

    print(f"Подключение к базе данных...")
    try:
        conn = psycopg2.connect(settings.db_dsn)
    except Exception as e:
        print(f"Не удалось подключиться к базе данных: {e}")
        return

    print(f"Начинается экспорт данных для символа {args.symbol} за период с {args.start} по {args.end}")
    print(f"Данные будут разбиты на {len(date_chunks)} частей по {args.chunk_size_days} дней.")

    # Итерация по временным отрезкам с прогресс-баром
    for chunk_start, chunk_end in tqdm(date_chunks, desc="Экспорт данных"):
        chunk_start_str = chunk_start.strftime("%Y-%m-%d %H:%M:%S")
        chunk_end_str = chunk_end.strftime("%Y-%m-%d %H:%M:%S")

        # 1. Экспорт сделок (agg_trades)
        query_trades = f"""
        SELECT event_time, payload FROM agg_trades
        WHERE symbol = '{args.symbol}' AND event_time >= '{chunk_start_str}' AND event_time < '{chunk_end_str}'
        ORDER BY event_time;
        """
        export_data_for_chunk(conn, query_trades, trades_out_path, process_trades)

        # 2. Экспорт стаканов (depth_updates)
        query_depth = f"""
        SELECT event_time, payload FROM depth_updates
        WHERE symbol = '{args.symbol}' AND event_time >= '{chunk_start_str}' AND event_time < '{chunk_end_str}'
        ORDER BY event_time;
        """
        export_data_for_chunk(conn, query_depth, depth_out_path, process_depth)

        # 3. Экспорт ликвидаций (force_orders)
        query_liquidations = f"""
        SELECT event_time, payload FROM force_orders
        WHERE payload->'o'->>'s' = '{args.symbol}' AND event_time >= '{chunk_start_str}' AND event_time < '{chunk_end_str}'
        ORDER BY event_time;
        """
        export_data_for_chunk(conn, query_liquidations, liquidations_out_path, process_liquidations)

    conn.close()
    print("\nЭкспорт успешно завершен.")
    print(f"Сырые данные сохранены в директории: {args.out_dir}")

if __name__ == "__main__":
    main()
