# -*- coding: utf-8 -*-
"""
Экспериментальный скрипт для поиска нового торгового признака с использованием
генетического программирования (GP).

Цель:
Автоматически найти новую, нетривиальную формулу (индикатор), которая улучшает
предсказательную силу существующей модели.

Процесс:
1.  Загружает датасет с признаками, подготовленный 'prepare_dataset.py'.
2.  Создает целевую переменную для задачи классификации (например, вырастет
    или упадет цена в будущем).
3.  Обучает базовую модель (LightGBM) на существующих признаках и оценивает
    ее качество (baseline).
4.  Использует библиотеку `gplearn` для запуска генетического алгоритма,
    который "изобретает" новые признаки-формулы.
5.  Выбирает лучшую найденную формулу.
6.  Добавляет новый "генетический" признак к исходному набору.
7.  Повторно обучает модель на расширенном наборе признаков и оценивает
    ее новое качество.
8.  Сравнивает качество "до" и "после" и выводит саму найденную формулу.
"""
import pandas as pd
import numpy as np
import lightgbm as lgb
from gplearn.genetic import SymbolicTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import argparse
import os

# --- Вспомогательные функции ---

def create_dummy_dataset(filepath: str, num_rows: int = 2000):
    """
    Создает полнофункциональный макет датасета, если основной файл не найден.
    Включает в себя как "AI-driven" признаки, так и "человеческие" индикаторы.
    """
    print(f"File not found. Creating a dummy dataset at {filepath}...")

    # AI-driven features
    ai_feature_cols = [
        'order_flow_delta', 'absorption_strength', 'panic_index',
        'orderbook_imbalance', 'dist_to_buy_wall', 'buy_wall_vol',
        'dist_to_sell_wall', 'sell_wall_vol', 'footprint_imbalance',
        'cascade_exhaustion'
    ]
    # Human-like features (names are based on pandas-ta defaults)
    human_feature_cols = [
        'EMA_9', 'EMA_21', 'VWAP_D', 'RSI_7', 'BBL_20_2.0', 'BBM_20_2.0', 'BBU_20_2.0'
    ]

    data = {}
    # Generate random data for all feature columns
    for col in ai_feature_cols + human_feature_cols:
        data[col] = np.random.randn(num_rows)

    # Add base columns 'price' and 'quantity'
    data['price'] = 10000 + np.random.randn(num_rows).cumsum()
    data['quantity'] = np.random.rand(num_rows) * 10

    df = pd.DataFrame(data)

    try:
        df.to_parquet(filepath)
        print(f"Dummy dataset saved successfully to {filepath}")
    except Exception as e:
        print(f"Failed to save dummy dataset: {e}")
    return df


def load_dataset(filepath: str) -> pd.DataFrame:
    """Загружает датасет из Parquet файла. Если файл не найден, создает макет."""
    if not os.path.exists(filepath):
        return create_dummy_dataset(filepath)
    print(f"Loading dataset from {filepath}...")
    return pd.read_parquet(filepath)


def create_target_variable(df: pd.DataFrame, lookahead: int = 100) -> pd.Series:
    """
    Создает целевую переменную (y) для задачи бинарной классификации.
    Предсказываем, будет ли цена через `lookahead` шагов выше или ниже текущей.
    """
    print(f"Creating target variable with a lookahead of {lookahead} steps...")
    future_price = df['price'].shift(-lookahead)
    target = (future_price > df['price']).astype(int)
    return target


def evaluate_model(X: pd.DataFrame, y: pd.Series, model_params: dict):
    """
    Обучает и оценивает модель LightGBM.
    Возвращает обученную модель и ее точность (accuracy) на тестовой выборке.
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, shuffle=False)

    model = lgb.LGBMClassifier(**model_params)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    return model, accuracy

# --- Главная функция ---

def main():
    parser = argparse.ArgumentParser(description="Find a new feature using Genetic Programming.")
    parser.add_argument('--dataset', type=str, default='features.parquet',
                        help='Path to the feature dataset file.')
    parser.add_argument('--lookahead', type=int, default=100,
                        help='Number of future steps to predict price direction.')

    # Параметры для gplearn
    parser.add_argument('--gp-population', type=int, default=100, help='Population size for GP.')
    parser.add_argument('--gp-generations', type=int, default=10, help='Number of generations for GP.')

    args = parser.parse_args()

    print("--- Starting Genetic Feature Engineering Experiment ---")

    # 1. Загрузка данных и создание признаков/цели
    df = load_dataset(args.dataset)

    # Отделяем исходные признаки (X) и создаем цель (y)
    # Используем все колонки кроме 'price' как признаки
    feature_cols = [col for col in df.columns if col not in ['price']]
    X = df[feature_cols].copy()
    y = create_target_variable(df, lookahead=args.lookahead)

    # Удаляем строки с NaN, которые появились после создания цели
    X = X.iloc[:-args.lookahead].dropna()
    y = y.loc[X.index] # Убедимся, что индексы y соответствуют X

    if X.empty or y.empty:
        print("Not enough data to create a valid training set. Exiting.")
        return

    print(f"\nTraining on {len(X)} samples with {len(X.columns)} features.")

    # 2. Обучение и оценка базовой модели
    print("\n--- Step 1: Evaluating Baseline Model ---")
    model_params = {'random_state': 42, 'n_estimators': 100, 'n_jobs': -1}
    baseline_model, baseline_accuracy = evaluate_model(X, y, model_params)
    print(f"Baseline Accuracy (with original features): {baseline_accuracy:.4f}")

    # 3. Поиск нового признака с помощью gplearn
    print("\n--- Step 2: Running Genetic Programming to find a new feature ---")

    gp = SymbolicTransformer(
        population_size=args.gp_population,
        generations=args.gp_generations,
        n_components=1,
        function_set=('add', 'sub', 'mul', 'div', 'sqrt', 'log', 'abs', 'neg', 'inv'),
        metric='spearman',
        random_state=42,
        n_jobs=-1
    )

    # --- МОНКИ-ПАТЧ (MONKEY-PATCH) для совместимости gplearn и scikit-learn ---
    try:
        from sklearn.utils.validation import _check_y, check_X_y
        def _validate_data(self, X, y, **check_params):
            if y is None:
                raise ValueError("y must be specified")
            X, y = check_X_y(X, y, **check_params)
            y = _check_y(y)
            return X, y
        gp._validate_data = _validate_data.__get__(gp, type(gp))
    except ImportError:
        gp._validate_data = lambda X, y: (X, y)

    gp.fit(X.values, y.values)

    best_feature_formula = gp._best_programs[0]
    print(f"\nBest feature formula found: {best_feature_formula}")

    # 4. Оценка модели с новым "генетическим" признаком
    print("\n--- Step 3: Evaluating Model with the new Genetic Feature ---")

    # Патч для атрибута n_features_in_
    gp.n_features_in_ = X.shape[1]

    X_new_feature = gp.transform(X.values)
    X_augmented = X.copy()
    X_augmented['genetic_feature'] = X_new_feature[:, 0]

    augmented_model, augmented_accuracy = evaluate_model(X_augmented, y, model_params)
    print(f"Augmented Accuracy (with genetic feature): {augmented_accuracy:.4f}")

    # 5. Вывод результатов
    print("\n--- Experiment Results ---")
    print(f"Baseline Accuracy:  {baseline_accuracy:.4f}")
    print(f"Augmented Accuracy: {augmented_accuracy:.4f}")
    improvement = augmented_accuracy - baseline_accuracy
    print(f"Improvement:        {improvement:+.4f}")

    if improvement > 0.001: # Считаем улучшением только значимый прирост
        print("\nConclusion: Genetic Programming found a useful feature!")
    else:
        print("\nConclusion: The new feature did not significantly improve performance.")

    print(f"\nDiscovered Feature Formula:\n{best_feature_formula}")


if __name__ == "__main__":
    main()
