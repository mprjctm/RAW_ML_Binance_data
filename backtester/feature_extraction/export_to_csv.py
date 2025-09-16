# -*- coding: utf-8 -*-
"""
Этот скрипт выполняет выгрузку "сырых" рыночных данных из базы данных
PostgreSQL/TimescaleDB в локальные файлы формата CSV.

Он является первым шагом в пайплайне подготовки данных для машинного обучения.
Скрипт работает в режиме "батчинга", обрабатывая данные по дням, чтобы
минимизировать потребление памяти и нагрузку на базу данных.

**Логика работы:**
1.  **Подключение к БД:** Использует строку подключения `DB_DSN` из файла `.env`.
2.  **Итерация по датам:** Проходит по указанному временному диапазону с заданным шагом.
3.  **Запросы к таблицам:** Для каждого временного отрезка выполняет SQL-запросы к таблицам:
    - `agg_trades` (агрегированные сделки)
    - `depth_updates` (обновления стакана)
    - `force_orders` (принудительные ликвидации)
4.  **Обработка данных:**
    - Загружает данные в pandas DataFrame.
    - Раскрывает полезные поля из JSONB-колонки `payload` в отдельные колонки.
    - Приводит типы данных к оптимальным.
5.  **Сохранение в CSV:** Дописывает обработанные данные в итоговые CSV-файлы.
    Формат именования файлов: `{out_dir}/{symbol}_{data_type}.csv`.
"""
import argparse
import os
import pandas as pd
import psycopg2
from datetime import datetime, timedelta
from tqdm import tqdm
import json
from typing import Callable, Dict, Any

# Импортируем настройки, включая DSN для подключения к БД
from config import settings

def parse_args() -> argparse.Namespace:
    """Парсит аргументы командной строки."""
    parser = argparse.ArgumentParser(
        description="Экспорт сырых данных из БД в CSV-файлы.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--symbol', type=str, required=True, help='Символ для экспорта (например, BTCUSDT).')
    parser.add_argument('--start', type=str, required=True, help='Дата начала экспорта в формате "YYYY-MM-DD HH:MM:SS".')
    parser.add_argument('--end', type=str, required=True, help='Дата окончания экспорта в формате "YYYY-MM-DD HH:MM:SS".')
    parser.add_argument('--out-dir', type=str, default='raw_data', help='Директория для сохранения CSV-файлов.')
    parser.add_argument('--chunk-size-days', type=int, default=7, help='Размер одного "куска" данных для обработки в днях.')
    return parser.parse_args()

def export_data_for_chunk(conn, query: str, params: Dict[str, Any], output_path: str, process_func: Callable[[pd.DataFrame], pd.DataFrame]) -> bool:
    """
    Выполняет параметризованный запрос для одного "куска" данных, обрабатывает и сохраняет его в CSV.
    Возвращает True в случае успеха и False в случае ошибки.
    """
    try:
        df = pd.read_sql_query(query, conn, params=params)

        if df.empty:
            return True

        df = process_func(df)

        # Если файл еще не существует, пишем с заголовком.
        # В противном случае дописываем без заголовка.
        header = not os.path.exists(output_path)
        df.to_csv(output_path, mode='a', header=header, index=True)

        return True

    except Exception as e:
        print(f"  - Ошибка при экспорте данных для {output_path} с параметрами {params}: {e}")
        return False

def _safe_json_normalize(data_series: pd.Series, is_nested: bool = False, nested_key: str = '') -> pd.DataFrame:
    """
    Безопасно обрабатывает и нормализует серию, содержащую JSON-строки или словари.
    """
    processed_payloads = []
    for p in data_series:
        try:
            loaded_p = json.loads(p) if isinstance(p, str) else p
            if is_nested:
                nested_p = loaded_p.get(nested_key, {})
                if isinstance(nested_p, str):
                    nested_p = json.loads(nested_p)
                processed_payloads.append(nested_p)
            else:
                processed_payloads.append(loaded_p)
        except (json.JSONDecodeError, TypeError):
            processed_payloads.append({})

    return pd.json_normalize(processed_payloads)

def process_trades(df: pd.DataFrame) -> pd.DataFrame:
    """Раскрывает JSON `payload` для сделок."""
    payload_df = _safe_json_normalize(df['payload'])
    df['price'] = pd.to_numeric(payload_df.get('p'), errors='coerce')
    df['quantity'] = pd.to_numeric(payload_df.get('q'), errors='coerce')
    df['is_buyer_maker'] = payload_df.get('m')
    df.drop(columns=['payload'], inplace=True)
    df.set_index('event_time', inplace=True)
    return df

def process_depth(df: pd.DataFrame) -> pd.DataFrame:
    """Раскрывем JSON `payload` для стакана."""
    payload_df = _safe_json_normalize(df['payload'])
    df['bids'] = payload_df.get('b')
    df['asks'] = payload_df.get('a')
    df.drop(columns=['payload'], inplace=True)
    df.set_index('event_time', inplace=True)
    return df

def process_liquidations(df: pd.DataFrame) -> pd.DataFrame:
    """Раскрывает вложенный JSON `payload.o` для ликвидаций."""
    payload_df = _safe_json_normalize(df['payload'], is_nested=True, nested_key='o')
    df['side'] = payload_df.get('S')
    df['price'] = pd.to_numeric(payload_df.get('p'), errors='coerce')
    df['quantity'] = pd.to_numeric(payload_df.get('q'), errors='coerce')
    df.drop(columns=['payload'], inplace=True)
    df.set_index('event_time', inplace=True)
    return df

def main():
    """Основная функция для запуска экспорта."""
    args = parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    trades_out_path = os.path.join(args.out_dir, f"{args.symbol}_trades.csv")
    depth_out_path = os.path.join(args.out_dir, f"{args.symbol}_depth.csv")
    liquidations_out_path = os.path.join(args.out_dir, f"{args.symbol}_liquidations.csv")

    # Удаляем старые файлы, если они существуют, чтобы начать выгрузку заново
    for path in [trades_out_path, depth_out_path, liquidations_out_path]:
        if os.path.exists(path):
            os.remove(path)

    start_date = datetime.strptime(args.start, "%Y-%m-%d %H:%M:%S")
    end_date = datetime.strptime(args.end, "%Y-%m-%d %H:%M:%S")

    date_chunks = []
    current_start = start_date
    while current_start < end_date:
        current_end = current_start + timedelta(days=args.chunk_size_days)
        date_chunks.append((current_start, min(current_end, end_date)))
        current_start = current_end

    print("Подключение к базе данных...")
    try:
        conn = psycopg2.connect(settings.db_dsn)
    except Exception as e:
        print(f"Не удалось подключиться к базе данных: {e}")
        return

    print(f"Начинается экспорт данных для символа {args.symbol} за период с {args.start} по {args.end}")
    print(f"Данные будут разбиты на {len(date_chunks)} частей по {args.chunk_size_days} дней.")

    query_trades_template = """
        SELECT event_time, payload FROM agg_trades
        WHERE symbol = %(symbol)s AND event_time >= %(start)s AND event_time < %(end)s
        ORDER BY event_time;
    """
    query_depth_template = """
        SELECT event_time, payload FROM depth_updates
        WHERE symbol = %(symbol)s AND event_time >= %(start)s AND event_time < %(end)s
        ORDER BY event_time;
    """
    query_liquidations_template = """
        SELECT event_time, payload FROM force_orders
        WHERE payload->'o'->>'s' = %(symbol)s AND event_time >= %(start)s AND event_time < %(end)s
        ORDER BY event_time;
    """

    for chunk_start, chunk_end in tqdm(date_chunks, desc="Экспорт данных"):
        params = {
            "symbol": args.symbol,
            "start": chunk_start,
            "end": chunk_end
        }
        export_data_for_chunk(conn, query_trades_template, params, trades_out_path, process_trades)
        export_data_for_chunk(conn, query_depth_template, params, depth_out_path, process_depth)
        export_data_for_chunk(conn, query_liquidations_template, params, liquidations_out_path, process_liquidations)

    conn.close()

    print("\nЭкспорт завершен.")
    print(f"Сырые данные сохранены в директории: {args.out_dir}")

if __name__ == "__main__":
    main()
