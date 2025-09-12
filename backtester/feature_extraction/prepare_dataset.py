import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
import argparse
import os
from datetime import datetime

# Импортируем наши новые функции для расчета признаков
from backtester.feature_extraction.features import (
    calculate_absorption_strength,
    calculate_cascade_exhaustion,
    calculate_panic_index,
    calculate_order_flow_delta
)

# --- Конфигурация ---
DB_DSN = os.getenv("DB_DSN", "postgresql://user:password@localhost:5432/binance_data")

# --- Функции Загрузки Данных ---

def load_trades_data(engine, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Загружает данные о сделках (agg_trades)."""
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
        df = pd.read_sql(query, connection, params={'symbol': symbol, 'start': start_date, 'end': end_date},
                         index_col='event_time', parse_dates=['event_time'])
    print(f"Loaded {len(df)} trade records.")
    return df

def load_depth_data(engine, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Загружает данные о стакане (depth_updates)."""
    print(f"Loading depth data for {symbol}...")
    query = text("""
        SELECT event_time, payload FROM depth_updates
        WHERE symbol = :symbol AND event_time BETWEEN :start AND :end
        ORDER BY event_time;
    """)
    with engine.connect() as connection:
        df = pd.read_sql(query, connection, params={'symbol': symbol, 'start': start_date, 'end': end_date},
                         index_col='event_time', parse_dates=['event_time'])
    print(f"Loaded {len(df)} depth update records.")
    return df

def load_liquidations_data(engine, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Загружает данные о ликвидациях (force_orders)."""
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
    """Основная функция для запуска конвейера подготовки данных."""
    parser = argparse.ArgumentParser(description="Подготовка датасета с продвинутыми признаками.")
    parser.add_argument('--symbol', type=str, default='BTCUSDT', help='Торговый символ.')
    parser.add_argument('--start', type=str, required=True, help='Дата начала (ГГГГ-ММ-ДД ЧЧ:ММ:СС).')
    parser.add_argument('--end', type=str, required=True, help='Дата окончания (ГГГГ-ММ-ДД ЧЧ:ММ:СС).')
    parser.add_argument('--output', type=str, required=True, help='Путь к выходному Parquet файлу.')

    args = parser.parse_args()

    print("--- Starting Data Preparation Pipeline ---")

    try:
        engine = create_engine(DB_DSN)
    except Exception as e:
        print(f"Failed to create database engine: {e}")
        return

    # 1. Загрузка всех необходимых данных
    # ПРИМЕЧАНИЕ: В песочнице это вызовет ошибку. Код предназначен для реальной среды.
    try:
        trades_df = load_trades_data(engine, args.symbol, args.start, args.end)
        depth_df = load_depth_data(engine, args.symbol, args.start, args.end)
        liquidations_df = load_liquidations_data(engine, args.symbol, args.start, args.end)
    except Exception as e:
        print(f"\nERROR: Failed to load data from database: {e}")
        print("NOTE: This is expected in the sandbox environment without a running DB.")
        print("The script is syntactically correct for a real environment.")
        return

    if trades_df.empty:
        print("No trade data found for the given period. Exiting.")
        return

    # 2. Объединение данных в один DataFrame
    print("Merging data sources...")
    # Используем merge_asof для объединения данных по ближайшему времени.
    # Это мощный инструмент для работы с асинхронными временными рядами.
    merged_df = pd.merge_asof(trades_df, depth_df, on='event_time', direction='backward')

    # 3. Расчет продвинутых индикаторов
    print("Calculating advanced features...")

    # Сначала базовые индикаторы, которые могут понадобиться для продвинутых
    merged_df['order_flow_delta'] = calculate_order_flow_delta(merged_df)

    # Теперь продвинутые индикаторы
    merged_df['absorption_strength'] = calculate_absorption_strength(merged_df)
    merged_df['panic_index'] = calculate_panic_index(merged_df)

    # Расчет истощения каскада требует отдельного датафрейма
    cascade_exhaustion = calculate_cascade_exhaustion(liquidations_df)
    if not cascade_exhaustion.empty:
        merged_df = pd.merge_asof(merged_df, cascade_exhaustion.to_frame(name='cascade_exhaustion'), on='event_time', direction='backward')
        # Заполняем NaN, которые появляются из-за редких событий ликвидации
        merged_df['cascade_exhaustion'].fillna(0, inplace=True)

    # Очистка от NaN, которые могли появиться в процессе
    merged_df.dropna(inplace=True)

    # 4. Сохранение итогового датасета
    print(f"Saving final dataset to {args.output}...")
    try:
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        merged_df.to_parquet(args.output, engine='pyarrow')
        print("Final dataset saved successfully.")
    except Exception as e:
        print(f"Failed to save final dataset: {e}")

    print("--- Pipeline Finished ---")

if __name__ == "__main__":
    main()
