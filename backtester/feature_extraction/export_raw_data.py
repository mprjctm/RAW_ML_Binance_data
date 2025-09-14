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
from dotenv import load_dotenv

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

    args = parser.parse_args()

    print("--- Starting Raw Data Export ---")

    # Создаем директорию, если она не существует
    os.makedirs(args.out_dir, exist_ok=True)

    # Подключение к БД
    try:
        engine = create_engine(DB_DSN)
    except Exception as e:
        print(f"Failed to create database engine: {e}")
        return

    # Загрузка и сохранение каждого типа данных
    try:
        # Сделки
        trades_df = load_trades_data(engine, args.symbol, args.start, args.end)
        if not trades_df.empty:
            trades_df.to_parquet(os.path.join(args.out_dir, f'{args.symbol}_trades.parquet'))
            print(f"Saved trades data to {args.out_dir}")

        # Стаканы
        depth_df = load_depth_data(engine, args.symbol, args.start, args.end)
        if not depth_df.empty:
            depth_df.to_parquet(os.path.join(args.out_dir, f'{args.symbol}_depth.parquet'))
            print(f"Saved depth data to {args.out_dir}")

        # Ликвидации
        liquidations_df = load_liquidations_data(engine, args.symbol, args.start, args.end)
        if not liquidations_df.empty:
            liquidations_df.to_parquet(os.path.join(args.out_dir, f'{args.symbol}_liquidations.parquet'))
            print(f"Saved liquidations data to {args.out_dir}")

    except Exception as e:
        print(f"\nERROR: Failed to load or save data: {e}")
        print("NOTE: This is expected in the sandbox environment without a running DB.")
        return

    print("--- Data Export Finished ---")

if __name__ == "__main__":
    main()
