#!/bin/bash

# Этот скрипт запускает подготовку датасета с заданными параметрами.
# Отредактируйте переменные ниже, чтобы изменить параметры запуска.

# --- Параметры ---
SYMBOL="BTCUSDT"
START_DATE="2023-01-01 00:00:00"
END_DATE="2026-12-31 23:59:59"
OUTPUT_FILE="btc_features_ws60.parquet"

# --- Параметры окон для индикаторов ---
DELTA_WINDOW=60
PANIC_WINDOW=60
ABSORPTION_WINDOW=60

# --- Команда запуска ---
echo "Запуск подготовки датасета для символа $SYMBOL..."

python -m backtester.feature_extraction.prepare_dataset \
    --symbol "$SYMBOL" \
    --start "$START_DATE" \
    --end "$END_DATE" \
    --output "$OUTPUT_FILE" \
    --delta-window "$DELTA_WINDOW" \
    --panic-window "$PANIC_WINDOW" \
    --absorption-window "$ABSORPTION_WINDOW"

echo "Скрипт завершил работу."
