import pandas as pd
import numpy as np
import pytest
from datetime import datetime, timedelta

from backtester.feature_extraction.features import (
    calculate_order_flow_delta,
    calculate_panic_index,
    calculate_absorption_strength,
    calculate_cascade_exhaustion
)

@pytest.fixture
def sample_trades_df() -> pd.DataFrame:
    """Создает тестовый DataFrame с данными о сделках."""
    base_time = datetime(2023, 1, 1, 12, 0, 0)
    data = {
        'price': [100, 101, 100, 102, 102, 99],
        'quantity': [10, 5, 8, 12, 2, 20]
    }
    index = [base_time + timedelta(seconds=i) for i in range(len(data['price']))]
    return pd.DataFrame(data, index=index)

def test_calculate_order_flow_delta(sample_trades_df):
    """Тестирует расчет дельты потока ордеров."""
    delta = calculate_order_flow_delta(sample_trades_df, window=3)

    # Ожидаемые значения:
    # diffs: [nan, +1, -1, +2, 0, -3]
    # delta: [0, +5, -8, +12, 0, -20]
    # rolling(3).sum(): [nan, nan, -3, 9, 4, -8]
    expected_values = [np.nan, np.nan, -3.0, 9.0, 4.0, -8.0]

    assert delta.name == 'order_flow_delta'
    pd.testing.assert_series_equal(delta, pd.Series(expected_values, index=sample_trades_df.index, name='order_flow_delta'), check_names=False)

def test_calculate_panic_index(sample_trades_df):
    """Тестирует расчет индекса паники."""
    panic_index = calculate_panic_index(sample_trades_df, window=3)

    assert panic_index.name == 'panic_index'
    assert not panic_index.isnull().all(), "Индекс паники не должен состоять только из NaN"
    assert panic_index.shape == (len(sample_trades_df),)

def test_panic_index_robustness_on_constant_price():
    """
    Тестирует устойчивость индекса паники при постоянной цене (чтобы проверить защиту от деления на ноль).
    """
    base_time = datetime(2023, 1, 1, 12, 0, 0)
    data = {
        'price': [100, 100, 100, 100, 100],
        'quantity': [10, 12, 8, 15, 11]
    }
    index = [base_time + timedelta(seconds=i) for i in range(len(data['price']))]
    df = pd.DataFrame(data, index=index)

    # Если бы не было защиты, здесь произошла бы ошибка деления на ноль.
    # Мы ожидаем, что функция отработает и вернет Series с NaN или нулями, но не упадет.
    panic_index = calculate_panic_index(df, window=3)

    # В этом случае волатильность будет 0, а ее нормализованное значение тоже 0.
    # Значения индекса не должны быть NaN (кроме первых, где окно неполное)
    # С учетом pct_change, первый не-NaN результат будет на 1 позже, чем у rolling.std
    assert not panic_index.iloc[3:].isnull().any()

def test_calculate_absorption_strength(sample_trades_df):
    """Тестирует расчет силы поглощения."""
    absorption = calculate_absorption_strength(sample_trades_df, window=3)

    assert absorption.name == 'absorption_strength'
    assert not absorption.isnull().all(), "Сила поглощения не должна состоять только из NaN"
    assert absorption.shape == (len(sample_trades_df),)

def test_calculate_cascade_exhaustion():
    """Тестирует расчет истощения каскада."""
    base_time = datetime(2023, 1, 1, 12, 0, 0)
    data = {'quantity': [100, 200, 50]}
    index = [base_time + timedelta(minutes=i) for i in range(len(data['quantity']))]
    liquidations_df = pd.DataFrame(data, index=index)

    exhaustion = calculate_cascade_exhaustion(liquidations_df, window='2min')

    # Ожидаемые значения: rolling(2min).count()
    # [1.0, 2.0, 2.0]
    expected_values = [1.0, 2.0, 2.0]

    pd.testing.assert_series_equal(exhaustion, pd.Series(expected_values, index=liquidations_df.index, name='quantity'), check_names=False)

def test_cascade_exhaustion_empty_df():
    """Тестирует расчет истощения каскада с пустым DataFrame."""
    empty_df = pd.DataFrame({'quantity': []})
    exhaustion = calculate_cascade_exhaustion(empty_df)

    assert isinstance(exhaustion, pd.Series)
    assert exhaustion.empty
