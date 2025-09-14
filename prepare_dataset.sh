#!/bin/bash
# Этот скрипт запускает полный двухэтапный конвейер подготовки данных.
# 1. Экспорт сырых данных из базы данных в Parquet файлы.
# 2. Расчет всех индикаторов и создание финального датасета с признаками.

set -e # Прерывать выполнение скрипта при любой ошибке

# --- Основные параметры ---
SYMBOL="BTCUSDT"
START_DATE="2023-01-01 00:00:00"
END_DATE="2023-12-31 23:59:59"

# --- Параметры для директорий и файлов ---
RAW_DATA_DIR="raw_data"
FEATURES_FILE="features.parquet"

# Создаем директорию для сырых данных, если ее нет
mkdir -p "$RAW_DATA_DIR"

# Определяем пути к файлам
RAW_TRADES_FILE="$RAW_DATA_DIR/${SYMBOL}_trades.parquet"
RAW_DEPTH_FILE="$RAW_DATA_DIR/${SYMBOL}_depth.parquet"
RAW_LIQUIDATIONS_FILE="$RAW_DATA_DIR/${SYMBOL}_liquidations.parquet"

# --- ЭТАП 1: Проверка наличия сырых данных ---
# Этот скрипт больше не запускает экспорт. Он ожидает, что вы
# предварительно запустили `export_raw_data.sh`.
echo "--- Проверка наличия сырых данных ---"
if [ ! -f "$RAW_TRADES_FILE" ] || [ ! -f "$RAW_DEPTH_FILE" ] || [ ! -f "$RAW_LIQUIDATIONS_FILE" ]; then
    echo "Ошибка: Один или несколько файлов с сырыми данными не найдены в директории '$RAW_DATA_DIR'."
    echo "Пожалуйста, сначала запустите 'bash export_raw_data.sh' для их создания."
    exit 1
fi
echo "--- Сырые данные найдены. ---"


# --- ЭТАП 2: Расчет признаков из сырых файлов ---
echo -e "\n--- Запуск ЭТАПА 2: Расчет признаков ---"
python -m backtester.feature_extraction.prepare_dataset \
    --trades-file "$RAW_TRADES_FILE" \
    --depth-file "$RAW_DEPTH_FILE" \
    --liquidations-file "$RAW_LIQUIDATIONS_FILE" \
    --output "$FEATURES_FILE" \
    # Здесь можно также указать параметры для окон индикаторов, если нужно
    # --delta-window 30 \
    # --panic-window 30

echo "--- ЭТАП 2 Завершен ---"
echo "Финальный датасет с признаками сохранен в: $FEATURES_FILE"
