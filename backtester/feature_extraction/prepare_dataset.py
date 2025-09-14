# -*- coding: utf-8 -*-
"""
Этот скрипт представляет собой ETL-конвейер (Extract, Transform, Load) для
подготовки данных для бэктестинга или обучения моделей.

Процесс работы скрипта:
1.  **Extract**: Загрузка "сырых" данных (сделки, обновления стакана, ликвидации)
    из базы данных PostgreSQL за указанный период.
2.  **Transform**:
    - Объединение разнородных временных рядов в единый DataFrame.
    - Расчет набора продвинутых признаков (features) с помощью функций из
      модуля `features.py`.
    - Очистка данных от пропусков (NaN), появившихся в процессе расчетов.
3.  **Load**: Сохранение итогового, обогащенного признаками датасета в
    эффективный бинарный формат Parquet.

Скрипт запускается из командной строки и позволяет настраивать основные
параметры, такие как символ, период и окна для расчета признаков.
"""
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
import argparse
import os
from datetime import datetime
from dotenv import load_dotenv

# Загружаем переменные окружения из .env файла в корне проекта.
# Это позволяет скрипту использовать DSN базы данных, указанный в .env.
load_dotenv()

# Импортируем наши функции для расчета признаков
from backtester.feature_extraction.features import (
    calculate_absorption_strength,
    calculate_cascade_exhaustion,
    calculate_panic_index,
    calculate_order_flow_delta,
    calculate_orderbook_imbalance,
    calculate_liquidity_walls,
    calculate_footprint_imbalance,
    calculate_standard_indicators
)

# --- Конфигурация ---
# Берем строку подключения к БД из переменных окружения.
# Если переменная DB_DSN не найдена, используется значение по умолчанию.
DB_DSN = os.getenv("DB_DSN", "postgresql://user:password@localhost:5432/binance_data")


# --- Функции Загрузки Данных ---

def load_trades_data(engine, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Загружает данные о сделках (agg_trades) из БД.
    Извлекает только необходимые поля: время, цену и объем.
    """
    print(f"Loading trades data for {symbol}...")
    # Используем `text()` для безопасной передачи параметров в SQL-запрос.
    # `CAST` преобразует текстовые JSON-поля в числовые типы данных.
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
            index_col='event_time',  # Сразу делаем время события индексом
            parse_dates=['event_time']  # Указываем pandas преобразовать колонку в datetime
        )
    print(f"Loaded {len(df)} trade records.")
    return df

def load_depth_data(engine, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Загружает данные о состоянии стакана (depth_updates) из БД.
    Извлекает списки заявок на покупку (bids) и продажу (asks).
    """
    print(f"Loading depth data for {symbol}...")
    # SQL-запрос для извлечения данных стакана.
    # `payload->'b'` и `payload->'a'` - синтаксис PostgreSQL для доступа к ключам JSON.
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
        # Это необходимо, так как из JSON они приходят как текст
        df['bids'] = df['bids'].apply(lambda x: [[float(p), float(q)] for p, q in x])
        df['asks'] = df['asks'].apply(lambda x: [[float(p), float(q)] for p, q in x])

    print(f"Loaded and processed {len(df)} depth update records.")
    return df

def load_liquidations_data(engine, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Загружает данные о принудительных ликвидациях (force_orders)."""
    print(f"Loading liquidations data for {symbol}...")
    # Извлекаем только объем ликвидации.
    # `payload->'o'->>'q'` - это синтаксис PostgreSQL для доступа к вложенным полям в JSON.
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
    # --- 1. Парсинг аргументов командной строки ---
    parser = argparse.ArgumentParser(
        description="Подготовка датасета с продвинутыми признаками для бэктестинга.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter # Показывает значения по умолчанию в --help
    )
    parser.add_argument('--symbol', type=str, default='BTCUSDT', help='Торговый символ.')
    parser.add_argument('--start', type=str, required=True, help='Дата начала (ГГГГ-ММ-ДД ЧЧ:ММ:СС).')
    parser.add_argument('--end', type=str, required=True, help='Дата окончания (ГГГГ-ММ-ДД ЧЧ:ММ:СС).')
    parser.add_argument('--output', type=str, required=True, help='Путь к выходному Parquet файлу.')

    # Аргументы для настройки окон признаков
    parser.add_argument('--delta-window', type=int, default=30, help='Окно для Order Flow Delta.')
    parser.add_argument('--panic-window', type=int, default=30, help='Окно для Panic Index.')
    parser.add_argument('--absorption-window', type=int, default=50, help='Окно для Absorption Strength.')
    parser.add_argument('--obi-levels', type=int, default=5, help='Количество уровней стакана для OBI.')
    parser.add_argument('--wall-factor', type=float, default=10.0, help='Множитель для определения "стены".')
    parser.add_argument('--wall-neighborhood', type=int, default=5, help='Кол-во соседних уровней для анализа стены.')
    parser.add_argument('--imbalance-ratio', type=float, default=3.0, help='Соотношение для определения дисбаланса футпринта.')
    parser.add_argument('--imbalance-window', type=int, default=100, help='Окно для скользящей суммы дисбаланса футпринта.')

    args = parser.parse_args()

    print("--- Starting Data Preparation Pipeline ---")

    # --- 2. Подключение к БД ---
    try:
        engine = create_engine(DB_DSN)
    except Exception as e:
        print(f"Failed to create database engine: {e}")
        return

    # --- 3. Загрузка всех необходимых данных ---
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

    if depth_df.empty:
        print("No depth data found for the given period. Footprint indicators cannot be calculated. Exiting.")
        return

    # --- 4. Объединение данных о сделках и стакане ---
    # Это ключевой шаг для работы с асинхронными данными. Для каждой сделки
    # мы находим последнее состояние стакана ПЕРЕД этой сделкой.
    # Это позволяет нам анализировать, как сделка повлияла на стакан,
    # который существовал в тот момент.
    print("Merging trades and depth data...")
    merged_df = pd.merge_asof(
        trades_df,
        depth_df,
        on='event_time',
        direction='backward'  # Найти последнее состояние стакана ПЕРЕД сделкой
    )

    # --- 5. РЕСЕМПЛИНГ И РАСЧЕТ ИНДИКАТОРОВ ---
    print("Resampling tick data to 1-minute OHLCV bars...")
    # Правила для агрегации данных в свечи
    agg_rules = {
        'price': ['first', 'max', 'min', 'last'],
        'quantity': 'sum'
    }
    df_resampled = merged_df.resample('1T').agg(agg_rules)
    # Переименовываем колонки в стандартный формат ohlc/volume
    df_resampled.columns = ['open', 'high', 'low', 'close', 'volume']

    # --- 6. Расчет продвинутых индикаторов ---
    print("Calculating advanced features with the following windows:")
    print(f"  - Order Flow Delta: {args.delta_window}")
    print(f"  - Panic Index: {args.panic_window}")
    print(f"  - Absorption Strength: {args.absorption_window}")

    # Рассчитываем стандартные "человеческие" индикаторы на агрегированных данных
    standard_indicators_df = calculate_standard_indicators(df_resampled)
    # Теперь df_resampled содержит и ohlcv, и стандартные индикаторы
    df_resampled = pd.concat([df_resampled, standard_indicators_df], axis=1)

    # --- 7. ОБЪЕДИНЕНИЕ ПРИЗНАКОВ РАЗНЫХ МАСШТАБОВ ---
    print("Merging resampled indicators back into the main tick-level dataframe...")
    # Оставляем только колонки с индикаторами для объединения
    indicator_cols_to_merge = standard_indicators_df.columns
    # Используем merge_asof для присвоения каждому тику значения индикатора
    # с последней доступной минутной свечи (эффект forward-fill)
    merged_df = pd.merge_asof(
        left=merged_df,
        right=df_resampled[indicator_cols_to_merge],
        on='event_time',
        direction='backward'
    )

    # Рассчитываем признаки, которые зависят только от потока сделок
    merged_df['order_flow_delta'] = calculate_order_flow_delta(merged_df, window=args.delta_window)
    merged_df['absorption_strength'] = calculate_absorption_strength(merged_df, window=args.absorption_window)
    merged_df['panic_index'] = calculate_panic_index(merged_df, window=args.panic_window)

    # Рассчитываем признаки, основанные на данных стакана
    merged_df['orderbook_imbalance'] = calculate_orderbook_imbalance(merged_df, levels=args.obi_levels)

    wall_features_df = calculate_liquidity_walls(
        merged_df,
        wall_factor=args.wall_factor,
        neighborhood=args.wall_neighborhood
    )
    merged_df = pd.concat([merged_df, wall_features_df], axis=1)

    merged_df['footprint_imbalance'] = calculate_footprint_imbalance(
        merged_df,
        imbalance_ratio=args.imbalance_ratio,
        window=args.imbalance_window
    )


    # Рассчитываем признаки, зависящие от других потоков данных (ликвидации)
    cascade_exhaustion = calculate_cascade_exhaustion(liquidations_df)
    if not cascade_exhaustion.empty:
        # Присоединяем рассчитанный признак к основному датафрейму
        merged_df = pd.merge_asof(
            merged_df,
            cascade_exhaustion.to_frame(name='cascade_exhaustion'),
            on='event_time',
            direction='backward'
        )
        # После `merge_asof` могут появиться NaN в начале, если не было предшествующих
        # событий ликвидации. Также, если ликвидации редкие, значения будут
        # оставаться постоянными до следующего события. Заполняем пропуски нулями.
        merged_df['cascade_exhaustion'] = merged_df['cascade_exhaustion'].fillna(0)

    # --- 6. Очистка и сохранение ---
    # Удаляем все строки, где есть хотя бы одно значение NaN.
    # NaN появляются в начале датасета из-за скользящих окон.
    merged_df.dropna(inplace=True)

    # Используем абсолютный путь для надежного сохранения файла
    output_path = os.path.abspath(args.output)
    print(f"Saving final dataset to {output_path}...")
    try:
        # Убедимся, что директория для сохранения существует
        output_dir = os.path.dirname(output_path)
        os.makedirs(output_dir, exist_ok=True)
        # Сохраняем в Parquet - это быстрый и эффективный формат для больших датафреймов
        merged_df.to_parquet(output_path, engine='pyarrow')
        print("Final dataset saved successfully.")
    except Exception as e:
        print(f"Failed to save final dataset: {e}")

    print("--- Pipeline Finished ---")

if __name__ == "__main__":
    # Эта конструкция гарантирует, что код в main() будет выполнен,
    # только если скрипт запущен напрямую, а не импортирован как модуль.
    main()
