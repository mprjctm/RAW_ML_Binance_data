# -*- coding: utf-8 -*-
"""
Этот скрипт представляет собой ETL-конвейер (Extract, Transform, Load) для
подготовки данных для бэктестинга или обучения моделей.

Он работает с "сырыми" данными, предварительно выгруженными из БД,
и обрабатывает их по частям (чанками), чтобы избежать проблем с нехваткой
памяти на больших наборах данных. Для корректного расчета скользящих
индикаторов используется механизм перекрытия (overlap).
"""
import pandas as pd
import numpy as np
import argparse
import os
import orjson
from tqdm import tqdm
import platform
import logging

try:
    import resource
except ImportError:
    resource = None

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
    if isinstance(data, str):
        try: parsed_data = orjson.loads(data)
        except orjson.JSONDecodeError: return []
    elif hasattr(data, '__iter__'): parsed_data = data
    else: return []
    if parsed_data is None: return []
    converted_data = []
    for item in parsed_data:
        if isinstance(item, list) and len(item) == 2:
            try: converted_data.append([float(item[0]), float(item[1])])
            except (ValueError, TypeError): continue
    return converted_data

def _process_orderbook_data(df: pd.DataFrame) -> pd.DataFrame:
    for col in ['bids', 'asks']:
        if col in df.columns:
            df[col] = df[col].apply(_parse_and_convert_to_float)
    return df


def log_df_info(df: pd.DataFrame, name: str):
    """Logs shape, index type, and memory usage of a DataFrame."""
    if df.empty:
        logging.debug(f"DataFrame '{name}' is empty.")
        return
    mem_usage_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)
    logging.debug(
        f"DataFrame '{name}' | "
        f"Shape: {df.shape} | "
        f"Index: {type(df.index).__name__} | "
        f"Memory: {mem_usage_mb:.2f} MB"
    )


def _load_and_process_depth_in_batches(file_path: str, time_filter: list, batch_size: int = 100000) -> pd.DataFrame:
    """
    Загружает и обрабатывает данные стакана из CSV по частям (чанками) для экономии памяти.
    Использует pd.read_csv с параметром chunksize для итеративной загрузки.

    Args:
        file_path (str): Путь к CSV файлу с данными стакана.
        time_filter (list): Фильтр в формате [('col', 'op', 'val'), ...].
        batch_size (int): Количество строк в одном чанке для итеративной загрузки.

    Returns:
        pd.DataFrame: DataFrame с обработанными данными стакана для всего временного диапазона.
    """
    logging.info(f"Loading depth data in chunks from {file_path}...")
    try:
        chunk_start_time = time_filter[0][2]
        chunk_end_time = time_filter[1][2]

        # Создаем итератор для чтения CSV по чанкам
        csv_reader = pd.read_csv(
            file_path,
            chunksize=batch_size,
        )

        processed_chunks = []
        for chunk_df in csv_reader:
            if chunk_df.empty or 'event_time' not in chunk_df.columns:
                continue

            # Принудительно конвертируем и очищаем данные
            chunk_df['event_time'] = pd.to_datetime(chunk_df['event_time'], errors='coerce')
            chunk_df.dropna(subset=['event_time'], inplace=True)
            chunk_df.set_index('event_time', inplace=True)

            # Фильтруем каждый чанк по времени
            filtered_chunk = chunk_df[(chunk_df.index >= chunk_start_time) & (chunk_df.index < chunk_end_time)]

            if filtered_chunk.empty:
                continue

            # Обработка данных стакана (парсинг JSON-подобных строк)
            processed_df = _process_orderbook_data(filtered_chunk)
            processed_chunks.append(processed_df)

        if not processed_chunks:
            return pd.DataFrame(columns=['bids', 'asks'])

        return pd.concat(processed_chunks)

    except Exception as e:
        logging.error(f"ERROR: Ошибка при пакетной загрузке данных стакана из CSV: {e}", exc_info=True)
        empty_df = pd.DataFrame({'bids': pd.Series(dtype='object'), 'asks': pd.Series(dtype='object')})
        empty_df.index = pd.to_datetime([]).tz_localize('UTC')
        empty_df.index.name = 'event_time'
        return empty_df


def set_memory_limit(gb_limit: int):
    if resource is None:
        logging.warning("Модуль 'resource' недоступен. Ограничение по памяти не поддерживается в текущей ОС.")
        return
    try:
        memory_limit_bytes = gb_limit * 1024 * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (memory_limit_bytes, memory_limit_bytes))
        logging.info(f"Установлен лимит памяти: {gb_limit} GB")
    except (ValueError, resource.error) as e:
        logging.error(f"Не удалось установить лимит памяти: {e}", exc_info=True)

def main():
    # --- Настройка логирования ---
    logging.basicConfig(
        level=logging.DEBUG, # Устанавливаем уровень DEBUG для подробных логов
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    parser = argparse.ArgumentParser(
        description="Подготовка датасета с признаками на основе сырых данных из файлов (с обработкой по чанкам).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--trades-file', type=str, required=True)
    parser.add_argument('--depth-file', type=str, required=True)
    parser.add_argument('--liquidations-file', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--chunk-size', type=str, default='1D', help='Размер временного чанка (например, "1D", "6H").')
    parser.add_argument('--memory-limit', type=int, default=10, help='Лимит памяти в ГБ.')
    parser.add_argument('--delta-window', type=int, default=30)
    parser.add_argument('--panic-window', type=int, default=30)
    parser.add_argument('--absorption-window', type=int, default=50)
    parser.add_argument('--obi-levels', type=int, default=5)
    parser.add_argument('--wall-factor', type=float, default=10.0)
    parser.add_argument('--wall-neighborhood', type=int, default=5)
    parser.add_argument('--imbalance-ratio', type=float, default=3.0)
    parser.add_argument('--imbalance-window', type=int, default=100)
    args = parser.parse_args()

    set_memory_limit(args.memory_limit)
    logging.info("--- Starting Data Preparation Pipeline (Stateful Chunked) ---")

    logging.info(f"Определение полного временного диапазона из {args.trades_file}...")
    try:
        # Читаем только колонку с временем, чтобы определить диапазон дат
        trades_df = pd.read_csv(args.trades_file)
        if 'event_time' not in trades_df.columns:
            logging.error(f"В файле {args.trades_file} отсутствует колонка 'event_time'."); return
        trades_df['event_time'] = pd.to_datetime(trades_df['event_time'], errors='coerce')
        trades_df.dropna(subset=['event_time'], inplace=True)
        trades_df.set_index('event_time', inplace=True)

        if trades_df.index.empty:
            logging.warning("Файл со сделками пуст или не содержит корректных дат."); return
        start_date, end_date = trades_df.index.min(), trades_df.index.max()
        logging.info(f"Данные найдены в диапазоне от {start_date} до {end_date}")
    except Exception as e:
        logging.error(f"Не удалось прочитать индекс из файла со сделками: {e}", exc_info=True); return

    # Принудительно преобразуем end_date в Timestamp, чтобы избежать TypeError
    # при сравнении, если end_date была передана как строка.
    end_date = pd.to_datetime(end_date)

    date_chunks = pd.date_range(start=start_date, end=end_date, freq=args.chunk_size, inclusive='left')
    if date_chunks.empty: date_chunks = pd.Index([start_date])
    if date_chunks[-1] < end_date:
        date_chunks = date_chunks.append(pd.Index([end_date]))

    processed_chunks = []
    overlap_df = pd.DataFrame()

    for i in tqdm(range(len(date_chunks)), desc="Обработка чанков"):
        chunk_start = date_chunks[i]
        chunk_end = date_chunks[i+1] if i + 1 < len(date_chunks) else end_date + pd.Timedelta(nanoseconds=1)
        logging.info(f"--- Обработка чанка: {chunk_start} -> {chunk_end} ---")

        try:
            # Загружаем основные данные для чанка.
            trades_df = pd.read_csv(args.trades_file)
            if 'event_time' not in trades_df.columns:
                logging.error(f"ERROR: В файле {args.trades_file} отсутствует колонка 'event_time'."); continue
            trades_df['event_time'] = pd.to_datetime(trades_df['event_time'], errors='coerce')
            trades_df.dropna(subset=['event_time'], inplace=True)
            trades_df.set_index('event_time', inplace=True)
            trades_df_chunk = trades_df.loc[chunk_start:chunk_end]

            if trades_df_chunk.empty and overlap_df.empty:
                logging.info("В данном чанке и в буфере перекрытия нет сделок. Пропускаем."); continue

            # Объединяем с данными из предыдущего чанка для перекрытия
            trades_df_with_overlap = pd.concat([overlap_df, trades_df_chunk])
            log_df_info(trades_df_with_overlap, "trades_with_overlap")

            # Загружаем соответствующие данные из других файлов
            overlap_start_time = trades_df_with_overlap.index.min()
            
            # Фильтруем данные стакана и ликвидаций по расширенному временному диапазону
            full_chunk_filter = [('event_time', '>=', overlap_start_time), ('event_time', '<', chunk_end)]
            depth_df = _load_and_process_depth_in_batches(args.depth_file, full_chunk_filter)
            log_df_info(depth_df, "depth_chunk")

            liquidations_df_full = pd.read_csv(args.liquidations_file)
            if 'event_time' not in liquidations_df_full.columns:
                logging.warning(f"В файле {args.liquidations_file} отсутствует колонка 'event_time'. Продолжаем без данных о ликвидациях.")
                liquidations_df = pd.DataFrame() # Создаем пустой DataFrame
            else:
                liquidations_df_full['event_time'] = pd.to_datetime(liquidations_df_full['event_time'], errors='coerce')
                liquidations_df_full.dropna(subset=['event_time'], inplace=True)
                liquidations_df_full.set_index('event_time', inplace=True)
                liquidations_df = liquidations_df_full.loc[overlap_start_time:chunk_end]
            log_df_info(liquidations_df, "liquidations_chunk")


        except Exception as e:
            logging.error(f"ERROR: Не удалось загрузить данные для чанка: {e}", exc_info=True); continue

        # --- Обработка ---
        depth_df = _process_orderbook_data(depth_df)
        merged_df = pd.merge_asof(trades_df_with_overlap, depth_df, left_index=True, right_index=True, direction='backward')
        log_df_info(merged_df, "after_depth_merge")

        agg_rules = {'price': ['first', 'max', 'min', 'last'], 'quantity': 'sum'}
        df_resampled = merged_df.resample('1min').agg(agg_rules)
        df_resampled.columns = ['open', 'high', 'low', 'close', 'volume']
        df_resampled.dropna(inplace=True)
        log_df_info(df_resampled, "after_resample")

        if df_resampled.empty:
            logging.info("В данном чанке после ресемплинга не осталось данных."); continue

        standard_indicators_df = calculate_standard_indicators(df_resampled)
        df_resampled = pd.concat([df_resampled, standard_indicators_df], axis=1)

        merged_df = pd.merge_asof(merged_df, df_resampled[standard_indicators_df.columns], left_index=True, right_index=True, direction='backward')
        merged_df.ffill(inplace=True)
        log_df_info(merged_df, "after_standard_indicators_merge")

        feature_steps = [
            ('order_flow_delta', lambda df: calculate_order_flow_delta(df, window=args.delta_window)),
            ('absorption_strength', lambda df: calculate_absorption_strength(df, window=args.absorption_window)),
            ('panic_index', lambda df: calculate_panic_index(df, window=args.panic_window)),
            ('orderbook_imbalance', lambda df: calculate_orderbook_imbalance(df, levels=args.obi_levels)),
            ('footprint_imbalance', lambda df: calculate_footprint_imbalance(df, imbalance_ratio=args.imbalance_ratio, window=args.imbalance_window))
        ]
        for name, func in feature_steps:
            merged_df[name] = func(merged_df)
        log_df_info(merged_df, "after_custom_features")

        merged_df = pd.concat([merged_df, calculate_liquidity_walls(merged_df, wall_factor=args.wall_factor, neighborhood=args.wall_neighborhood)], axis=1)
        log_df_info(merged_df, "after_walls_concat")

        cascade = calculate_cascade_exhaustion(liquidations_df)
        if not cascade.empty:
            merged_df = pd.merge_asof(merged_df, cascade.to_frame(name='cascade_exhaustion'), left_index=True, right_index=True, direction='backward').fillna({'cascade_exhaustion': 0})
        log_df_info(merged_df, "after_cascade_merge")

        # --- Обрезка и сохранение ---
        # Отбрасываем данные, которые были нужны только для перекрытия
        final_chunk = merged_df.loc[chunk_start:chunk_end]
        if not final_chunk.empty:
            processed_chunks.append(final_chunk)

        # Готовим перекрытие для следующего чанка
        max_lookback = max(args.delta_window, args.panic_window, args.absorption_window, args.imbalance_window)
        overlap_df = trades_df_with_overlap.tail(max_lookback)

    if not processed_chunks:
        logging.warning("Не было обработано ни одного чанка."); return

    logging.info("Объединение всех обработанных чанков...")
    final_df = pd.concat(processed_chunks)
    log_df_info(final_df, "final_dataset")

    output_path = os.path.abspath(args.output)
    logging.info(f"Сохранение итогового датасета в {output_path}...")
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        final_df.to_parquet(output_path, engine='pyarrow')
        logging.info("Итоговый датасет успешно сохранен.")
    except Exception as e:
        logging.error(f"Не удалось сохранить итоговый датасет: {e}", exc_info=True)

    logging.info("--- Pipeline Finished ---")

if __name__ == "__main__":
    main()
