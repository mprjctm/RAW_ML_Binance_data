#!/bin/bash

# Этот скрипт запускает подготовку датасета с заданными параметрами.
# Отредактируйте переменные ниже, чтобы изменить параметры запуска.

# --- Основные параметры ---
SYMBOL="BTCUSDT"
START_DATE="2023-01-01 00:00:00"
END_DATE="2026-12-31 23:59:59"
OUTPUT_FILE="btc_features_ws60.parquet"

# --- Параметры индикаторов на основе потока сделок ---
DELTA_WINDOW=30
PANIC_WINDOW=30
ABSORPTION_WINDOW=50

# --- Параметры индикаторов на основе стакана ---
OBI_LEVELS=5
WALL_FACTOR=10.0
WALL_NEIGHBORHOOD=5
IMBALANCE_RATIO=3.0
IMBALANCE_WINDOW=100


# --- Команда запуска ---
echo "Запуск подготовки датасета для символа $SYMBOL..."

python -m backtester.feature_extraction.prepare_dataset \
    --symbol "$SYMBOL" \
    --start "$START_DATE" \
    --end "$END_DATE" \
    --output "$OUTPUT_FILE" \
    --delta-window "$DELTA_WINDOW" \
    --panic-window "$PANIC_WINDOW" \
    --absorption-window "$ABSORPTION_WINDOW" \
    --obi-levels "$OBI_LEVELS" \
    --wall-factor "$WALL_FACTOR" \
    --wall-neighborhood "$WALL_NEIGHBORHOOD" \
    --imbalance-ratio "$IMBALANCE_RATIO" \
    --imbalance-window "$IMBALANCE_WINDOW"

echo "Скрипт завершил работу."
