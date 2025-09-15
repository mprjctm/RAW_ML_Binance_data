# -*- coding: utf-8 -*-
"""
Этот скрипт представляет собой ETL-конвейер (Extract, Transform, Load) для
подготовки данных для бэктестинга или обучения моделей.

ЭТОТ СКРИПТ НЕ ПОДКЛЮЧАЕТСЯ К БАЗЕ ДАННЫХ.

Он является вторым шагом в пайплайне и работает с "сырыми" данными,
предварительно выгруженными из БД с помощью скрипта `export_raw_data.py`.

Процесс работы скрипта:
1.  **Extract**: Загрузка "сырых" данных (сделки, обновления стакана, ликвидации)
    из Parquet файлов.
2.  **Transform**:
    - Объединение разнородных временных рядов в единый DataFrame.
    - Агрегация данных в OHLCV бары.
    - Расчет стандартных и кастомных признаков (features).
    - Очистка данных от пропусков (NaN).
3.  **Load**: Сохранение итогового, обогащенного признаками датасета в
    эффективный бинарный формат Parquet.
"""
import pandas as pd
import numpy as np
import argparse
import os
import orjson
from tqdm import tqdm

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


def _parse_and_convert_to_float(data):
    """
    Парсит JSON-строку и преобразует все вложенные числовые строки в float.
    Также обрабатывает данные, которые уже являются списками, но содержат строки.
    """
    if isinstance(data, str):
        try:
            # Если данные - это строка, парсим ее как JSON
            parsed_data = orjson.loads(data)
        except orjson.JSONDecodeError:
            return []  # В случае ошибки парсинга возвращаем пустой список
    elif hasattr(data, '__iter__'):
        # Если это уже итерируемый объект (например, список), используем его напрямую
        parsed_data = data
    else:
        # Если это что-то другое (например, None или не-итерируемый тип), возвращаем пустой список
        return []

    # Преобразуем все элементы [[price, quantity], ...] в float
    # Добавляем проверку, чтобы избежать ошибок на пустых или некорректных данных
    if parsed_data is None:
        return []

    converted_data = []
    for item in parsed_data:
        if isinstance(item, list) and len(item) == 2:
            try:
                converted_data.append([float(item[0]), float(item[1])])
            except (ValueError, TypeError):
                # Пропускаем элементы, которые не могут быть преобразованы в float
                continue
    return converted_data


def _process_orderbook_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Применяет функцию очистки и преобразования к колонкам 'bids' и 'asks'.
    """
    print("Processing order book data types (parsing strings, converting to float)...")
    for col in ['bids', 'asks']:
        if col in df.columns:
            # Применяем нашу функцию к каждой ячейке в колонке
            df[col] = df[col].apply(_parse_and_convert_to_float)
    return df


def main():
    """Основная функция для запуска конвейера подготовки данных."""
    # --- 1. Парсинг аргументов командной строки ---
    parser = argparse.ArgumentParser(
        description="Подготовка датасета с признаками на основе сырых данных из файлов.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--trades-file', type=str, required=True, help='Путь к Parquet файлу со сделками.')
    parser.add_argument('--depth-file', type=str, required=True, help='Путь к Parquet файлу со стаканами.')
    parser.add_argument('--liquidations-file', type=str, required=True, help='Путь к Parquet файлу с ликвидациями.')
    parser.add_argument('--output', type=str, required=True, help='Путь к выходному Parquet файлу с признаками.')

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

    print("--- Starting Data Preparation Pipeline from Files ---")

    # --- 2. Загрузка всех необходимых данных из файлов ---
    try:
        print(f"Loading trades from {args.trades_file}...")
        trades_df = pd.read_parquet(args.trades_file)
        print(f"Loading depth from {args.depth_file}...")
        depth_df = pd.read_parquet(args.depth_file)
        print(f"Loading liquidations from {args.liquidations_file}...")
        liquidations_df = pd.read_parquet(args.liquidations_file)
    except FileNotFoundError as e:
        print(f"ERROR: Input file not found: {e}")
        return
    except Exception as e:
        print(f"ERROR: Failed to load data from Parquet files: {e}")
        return

    if trades_df.empty:
        print("No trade data found. Exiting.")
        return

    # --- 2.1. Очистка и преобразование типов данных стакана ---
    depth_df = _process_orderbook_data(depth_df)

    # --- 3. Объединение данных о сделках и стакане ---
    print("Merging trades and depth data...")
    merged_df = pd.merge_asof(
        left=trades_df,
        right=depth_df,
        left_index=True,
        right_index=True,
        direction='backward'
    )

    # --- 4. РЕСЕМПЛИНГ И РАСЧЕТ ИНДИКАТОРОВ ---
    print("Resampling tick data to 1-minute OHLCV bars...")
    agg_rules = {
        'price': ['first', 'max', 'min', 'last'],
        'quantity': 'sum'
    }
    df_resampled = merged_df.resample('1min').agg(agg_rules)
    df_resampled.columns = ['open', 'high', 'low', 'close', 'volume']

    # Рассчитываем стандартные "человеческие" индикаторы на агрегированных данных
    standard_indicators_df = calculate_standard_indicators(df_resampled)
    df_resampled = pd.concat([df_resampled, standard_indicators_df], axis=1)

    # --- 5. ОБЪЕДИНЕНИЕ ПРИЗНАКОВ РАЗНЫХ МАСШТАБОВ ---
    print("Merging resampled indicators back into the main tick-level dataframe...")
    indicator_cols_to_merge = standard_indicators_df.columns
    merged_df = pd.merge_asof(
        left=merged_df,
        right=df_resampled[indicator_cols_to_merge],
        on='event_time',
        direction='backward'
    )

    # --- 6. Расчет продвинутых индикаторов ---
    print("Calculating advanced features...")

    # Определяем последовательность шагов для расчета признаков
    feature_calculation_steps = [
        ('order_flow_delta', lambda df: calculate_order_flow_delta(df, window=args.delta_window)),
        ('absorption_strength', lambda df: calculate_absorption_strength(df, window=args.absorption_window)),
        ('panic_index', lambda df: calculate_panic_index(df, window=args.panic_window)),
        ('orderbook_imbalance', lambda df: calculate_orderbook_imbalance(df, levels=args.obi_levels)),
        ('footprint_imbalance', lambda df: calculate_footprint_imbalance(df, imbalance_ratio=args.imbalance_ratio, window=args.imbalance_window))
    ]

    # Итерируемся по шагам с прогресс-баром
    for feature_name, feature_func in tqdm(feature_calculation_steps, desc="Calculating AI Features"):
        merged_df[feature_name] = feature_func(merged_df)

    # Отдельно рассчитываем признаки, которые возвращают несколько колонок
    print("Calculating multi-column features (e.g., Liquidity Walls)...")
    wall_features_df = calculate_liquidity_walls(merged_df, wall_factor=args.wall_factor, neighborhood=args.wall_neighborhood)
    merged_df = pd.concat([merged_df, wall_features_df], axis=1)
    
    print("Calculating features from other data streams (e.g., Liquidations)...")
    cascade_exhaustion = calculate_cascade_exhaustion(liquidations_df)
    if not cascade_exhaustion.empty:
        merged_df = pd.merge_asof(
            merged_df,
            cascade_exhaustion.to_frame(name='cascade_exhaustion'),
            on='event_time',
            direction='backward'
        )
        merged_df['cascade_exhaustion'] = merged_df['cascade_exhaustion'].fillna(0)

    # --- 7. Очистка и сохранение ---
    # .dropna() удален, т.к. он слишком агрессивно удаляет строки,
    # где хотя бы один из множества индикаторов еще не рассчитался.
    # Обработка NaN - задача этапа моделирования.
    # merged_df.dropna(inplace=True)

    output_path = os.path.abspath(args.output)
    print(f"Saving final dataset to {output_path}...")
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        merged_df.to_parquet(output_path, engine='pyarrow')
        print("Final dataset saved successfully.")
    except Exception as e:
        print(f"Failed to save final dataset: {e}")

    print("--- Pipeline Finished ---")

if __name__ == "__main__":
    main()