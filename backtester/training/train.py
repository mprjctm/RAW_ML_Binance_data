# -*- coding: utf-8 -*-
"""
Этот скрипт отвечает за обучение и оценку ML-модели.

Он выполняет следующие шаги:
1.  Загружает датасет с признаками.
2.  Создает целевую переменную (y) на основе будущих изменений цены.
3.  Разделяет данные на обучающую и тестовую выборки по времени.
4.  Обучает модель градиентного бустинга LightGBM.
5.  Выводит отчет о качестве модели на тестовых данных.
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import classification_report, confusion_matrix
import argparse
import os
import seaborn as sns
import matplotlib.pyplot as plt

def load_data(file_path: str) -> pd.DataFrame:
    """Загружает датасет с признаками из Parquet-файла."""
    print(f"Загрузка данных из {file_path}...")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Файл с признаками не найден: {file_path}")
    df = pd.read_parquet(file_path)
    print("Данные успешно загружены.")
    return df

def create_target_variable(df: pd.DataFrame, horizon: int = 5, threshold: float = 0.002) -> pd.DataFrame:
    """
    Создает целевую переменную для задачи классификации.

    Args:
        df (pd.DataFrame): DataFrame с колонкой 'close'.
        horizon (int): Горизонт в минутах для предсказания.
        threshold (float): Порог изменения цены для определения сигнала (0.002 = 0.2%).

    Returns:
        pd.DataFrame: DataFrame с новой колонкой 'target'.
    """
    print(f"Создание целевой переменной с горизонтом {horizon} мин и порогом {threshold*100:.2f}%...")

    # Сдвигаем цены закрытия на 'horizon' минут в прошлое, чтобы получить будущую цену
    future_price = df['close'].shift(-horizon)

    # Рассчитываем процентное изменение цены
    price_change = (future_price - df['close']) / df['close']

    # Создаем таргет на основе порогов
    df['target'] = 0
    df.loc[price_change > threshold, 'target'] = 1  # Сигнал на покупку
    df.loc[price_change < -threshold, 'target'] = -1 # Сигнал на продажу

    print("Целевая переменная создана. Распределение классов:")
    print(df['target'].value_counts(normalize=True))

    return df

def main():
    parser = argparse.ArgumentParser(description="Обучение ML модели на данных футпринта.")
    parser.add_argument('--features-file', type=str, required=True, help='Путь к Parquet-файлу с признаками.')
    parser.add_argument('--test-size', type=float, default=0.2, help='Доля данных для тестовой выборки.')
    args = parser.parse_args()

    # 1. Загрузка данных
    df = load_data(args.features_file)

    # 2. Создание целевой переменной
    df = create_target_variable(df, horizon=5, threshold=0.002)

    # 3. Подготовка данных
    # Удаляем строки с NaN в таргете (последние `horizon` минут, для которых нет будущего)
    df.dropna(subset=['target'], inplace=True)
    # Выбираем признаки для обучения (все, кроме цен и целевой переменной)
    features = [col for col in df.columns if col not in ['open', 'high', 'low', 'close', 'volume', 'target']]

    # Заполняем пропуски в признаках нулями (простой подход)
    df[features] = df[features].fillna(0)
    # Заменяем бесконечные значения (если они есть) на 0
    df[features] = df[features].replace([np.inf, -np.inf], 0)

    X = df[features]
    y = df['target']

    # 4. Разделение данных по времени
    split_index = int(len(df) * (1 - args.test_size))
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]

    print(f"Размер обучающей выборки: {len(X_train)}")
    print(f"Размер тестовой выборки: {len(X_test)}")

    # 5. Обучение модели
    print("\nОбучение модели LightGBM...")
    model = lgb.LGBMClassifier(objective='multiclass', num_class=3, random_state=42)
    model.fit(X_train, y_train)
    print("Модель обучена.")

    # 6. Оценка модели
    print("\nОценка качества модели на тестовых данных:")
    y_pred = model.predict(X_test)

    report = classification_report(y_test, y_pred, target_names=['Sell (-1)', 'Hold (0)', 'Buy (1)'])
    print(report)

    # 7. Визуализация матрицы ошибок
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Sell', 'Hold', 'Buy'], yticklabels=['Sell', 'Hold', 'Buy'])
    plt.title('Матрица ошибок (Confusion Matrix)')
    plt.ylabel('Истинный класс')
    plt.xlabel('Предсказанный класс')

    # Сохраняем график в файл
    output_filename = 'confusion_matrix.png'
    plt.savefig(output_filename)
    print(f"\nМатрица ошибок сохранена в файл: {output_filename}")
    plt.show()

if __name__ == '__main__':
    main()
