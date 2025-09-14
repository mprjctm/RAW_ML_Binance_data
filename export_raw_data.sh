#!/bin/bash
# Этот скрипт запускает экспорт сырых данных из базы данных.
# Отредактируйте переменные ниже, чтобы изменить параметры выгрузки.

set -e # Прерывать выполнение скрипта при любой ошибке

# --- Параметры выгрузки ---
SYMBOL="BTCUSDT"
START_DATE="2023-01-01 00:00:00"
END_DATE="2023-01-31 23:59:59"
OUT_DIR="raw_data" # Директория для сохранения сырых данных

# --- Команда запуска ---
echo "Запуск экспорта сырых данных для символа $SYMBOL..."
echo "Период: с $START_DATE по $END_DATE"
echo "Директория для сохранения: $OUT_DIR"

python -m backtester.feature_extraction.export_raw_data \
    --symbol "$SYMBOL" \
    --start "$START_DATE" \
    --end "$END_DATE" \
    --out-dir "$OUT_DIR"

echo "Экспорт успешно завершен."
