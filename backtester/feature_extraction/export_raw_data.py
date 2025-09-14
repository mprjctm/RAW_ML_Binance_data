# -*- coding: utf-8 -*-
"""
Этот скрипт отвечает за первый этап подготовки данных: экспорт "сырых"
данных из базы данных PostgreSQL в файлы формата Parquet.

Это позволяет отделить ресурсоемкую задачу выгрузки данных от задачи их
обработки и расчета признаков, что дает возможность выполнять эти этапы
на разных серверах.

Скрипт подключается к БД, выполняет SQL-запросы для извлечения данных
о сделках, стаканах и ликвидациях за указанный период и сохраняет каждый
тип данных в отдельный файл.
"""
import pandas as pd
from sqlalchemy import create_engine, text
import argparse
import os
from datetime import datetime, timedelta
from dotenv import load_dotenv
import pyarrow.parquet as pq
import pyarrow as pa
from tqdm import tqdm

# Загружаем переменные окружения (в т.ч. DSN базы данных)
load_dotenv()

# --- Конфигурация ---
DB_DSN = os.getenv("DB_DSN", "postgresql://user:password@localhost:5432/binance_data")

# --- Функции Загрузки Данных ---

def load_trades_data(engine, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Загружает данные о сделках (agg_trades) из БД."""
    print(f"Loading trades data for {symbol}...")
    query = text("""
        SELECT
            event_time,
            CAST(payload->>'p' AS DECIMAL) AS price,
            CAST(payload->>'q' AS DECIMAL) AS quantity
        FROM agg_trades
        WHERE symbol = :symbol AND event_time BETWEEN :start AND :end
        ORDER BY event_time;
    """)
    with engine.connect() as connection:
        df = pd.read_sql(
            query,
            connection,
            params={'symbol': symbol, 'start': start_date, 'end': end_date},
            index_col='event_time',
            parse_dates=['event_time']
        )
    print(f"Loaded {len(df)} trade records.")
    return df

def load_depth_data(engine, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Загружает данные о состоянии стакана (depth_updates) из БД."""
    print(f"Loading depth data for {symbol}...")
    query = text("""
        SELECT
            event_time,
            payload->'b' as bids,
            payload->'a' as asks
        FROM depth_updates
        WHERE symbol = :symbol AND event_time BETWEEN :start AND :end
        ORDER BY event_time;
    """)
    with engine.connect() as connection:
        df = pd.read_sql(
            query,
            connection,
            params={'symbol': symbol, 'start': start_date, 'end': end_date},
            index_col='event_time',
            parse_dates=['event_time']
        )

    if not df.empty:
        # Конвертируем числовые значения в списках из строк в float
        df['bids'] = df['bids'].apply(lambda x: [[float(p), float(q)] for p, q in x])
        df['asks'] = df['asks'].apply(lambda x: [[float(p), float(q)] for p, q in x])
    print(f"Loaded and processed {len(df)} depth update records.")
    return df

def load_liquidations_data(engine, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Загружает данные о принудительных ликвидациях (force_orders)."""
    print(f"Loading liquidations data for {symbol}...")
    query = text("""
        SELECT
            event_time,
            CAST(payload->'o'->>'q' AS DECIMAL) as quantity
        FROM force_orders
        WHERE payload->'o'->>'s' = :symbol AND event_time BETWEEN :start AND :end
        ORDER BY event_time;
    """)
    with engine.connect() as connection:
        df = pd.read_sql(query, connection, params={'symbol': symbol, 'start': start_date, 'end': end_date},
                         index_col='event_time', parse_dates=['event_time'])
    print(f"Loaded {len(df)} liquidation records.")
    return df


def main():
    """Основная функция для запуска экспорта данных."""
    parser = argparse.ArgumentParser(
        description="Экспорт сырых данных из БД в файлы Parquet.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--symbol', type=str, default='BTCUSDT', help='Торговый символ.')
    parser.add_argument('--start', type=str, required=True, help='Дата начала (ГГГГ-ММ-ДД ЧЧ:ММ:СС).')
    parser.add_argument('--end', type=str, required=True, help='Дата окончания (ГГГГ-ММ-ДД ЧЧ:ММ:СС).')
    parser.add_argument('--out-dir', type=str, default='.', help='Директория для сохранения файлов.')
    parser.add_argument('--chunk-size-days', type=int, default=7, help='Количество дней для обработки в одной порции (для экономии памяти).')

    args = parser.parse_args()

    print("--- Starting Raw Data Export ---")

    os.makedirs(args.out_dir, exist_ok=True)

    try:
        engine = create_engine(DB_DSN)
    except Exception as e:
        print(f"Failed to create database engine: {e}")
        return

    start_date = datetime.fromisoformat(args.start)
    end_date = datetime.fromisoformat(args.end)
    chunk_delta = timedelta(days=args.chunk_size_days)

    output_files = {
        'trades': os.path.join(args.out_dir, f'{args.symbol}_trades.parquet'),
        'depth': os.path.join(args.out_dir, f'{args.symbol}_depth.parquet'),
        'liquidations': os.path.join(args.out_dir, f'{args.symbol}_liquidations.parquet')
    }
    for f in output_files.values():
        if os.path.exists(f):
            os.remove(f)
            print(f"Removed existing file: {f}")

    writers = {}

    try:
        date_chunks = list(pd.date_range(start=start_date, end=end_date, freq=f'{args.chunk_size_days}D'))

        for current_start in tqdm(date_chunks, desc="Exporting Data Chunks"):
            current_end = min(current_start + chunk_delta - timedelta(seconds=1), end_date)

            trades_df = load_trades_data(engine, args.symbol, str(current_start), str(current_end))
            if not trades_df.empty:
                table = pa.Table.from_pandas(trades_df)
                if 'trades' not in writers:
                    writers['trades'] = pq.ParquetWriter(output_files['trades'], table.schema)
                writers['trades'].write_table(table)
                print(f"Appended {len(trades_df)} records to trades file.")

            depth_df = load_depth_data(engine, args.symbol, str(current_start), str(current_end))
            if not depth_df.empty:
                table = pa.Table.from_pandas(depth_df)
                if 'depth' not in writers:
                    writers['depth'] = pq.ParquetWriter(output_files['depth'], table.schema)
                writers['depth'].write_table(table)
                print(f"Appended {len(depth_df)} records to depth file.")

            liquidations_df = load_liquidations_data(engine, args.symbol, str(current_start), str(current_end))
            if not liquidations_df.empty:
                table = pa.Table.from_pandas(liquidations_df)
                if 'liquidations' not in writers:
                    writers['liquidations'] = pq.ParquetWriter(output_files['liquidations'], table.schema)
                writers['liquidations'].write_table(table)
                print(f"Appended {len(liquidations_df)} records to liquidations file.")

    except Exception as e:
        print(f"\nFATAL ERROR during processing: {e}")
    finally:
        print("\nClosing Parquet writers...")
        for writer in writers.values():
            writer.close()

    print("--- Data Export Finished ---")

if __name__ == "__main__":
    main()
