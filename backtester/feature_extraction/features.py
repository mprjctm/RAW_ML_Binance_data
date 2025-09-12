import pandas as pd
import numpy as np

"""
Этот модуль содержит функции для расчета продвинутых, кастомных индикаторов,
основанных на микроструктуре рынка.
"""

def calculate_order_flow_delta(df: pd.DataFrame, window: int = 30) -> pd.Series:
    """
    Рассчитывает дельту потока ордеров.
    Требует `payload` из `agg_trades` для определения агрессора.
    Поскольку у нас нет этого в основном DataFrame, мы используем изменение цены как прокси.
    Положительное изменение цены ~ покупка, отрицательное ~ продажа.
    """
    print("Calculating Order Flow Delta (Proxy)...")
    price_change = df['price'].diff()

    # Если цена выросла, считаем объем покупкой, если упала - продажей.
    delta = np.where(price_change > 0, df['quantity'],
                     np.where(price_change < 0, -df['quantity'], 0))

    delta_series = pd.Series(delta, index=df.index)

    # Суммируем дельту за скользящее окно
    return delta_series.rolling(window=window).sum()


def calculate_cascade_exhaustion(liquidations_df: pd.DataFrame, window: str = '5min') -> pd.Series:
    """
    Рассчитывает "Истощение Каскада" ликвидаций.
    Измеряет интенсивность ликвидаций и определяет, когда она начинает "выдыхаться".
    """
    print("Calculating Cascade Exhaustion...")
    if liquidations_df.empty:
        return pd.Series()

    # Считаем количество ликвидаций за скользящий временной интервал
    liquidation_counts = liquidations_df['quantity'].rolling(window).count()

    # Определяем "истощение", когда текущее количество значительно меньше среднего за последнее время.
    # Для простоты, мы просто возвращаем количество. В реальной стратегии здесь была бы более сложная логика.
    return liquidation_counts


def calculate_panic_index(df: pd.DataFrame, window: int = 30) -> pd.Series:
    """
    Рассчитывает "Индекс Паники Толпы".
    Комбинирует волатильность и всплески объема.
    """
    print("Calculating Panic Index...")

    # 1. Волатильность (стандартное отклонение логарифмической доходности)
    log_returns = np.log(df['price'] / df['price'].shift(1))
    volatility = log_returns.rolling(window=window).std()

    # 2. Всплеск объема (процентное изменение суммы объема)
    volume_sum = df['quantity'].rolling(window=window).sum()
    volume_roc = volume_sum.pct_change()

    # Нормализуем оба компонента (от 0 до 1), чтобы их можно было сложить
    norm_volatility = (volatility - volatility.min()) / (volatility.max() - volatility.min())
    norm_volume_roc = (volume_roc - volume_roc.min()) / (volume_roc.max() - volume_roc.min())

    # Простой индекс паники - взвешенное среднее
    panic_index = (norm_volatility * 0.6 + norm_volume_roc * 0.4)

    return panic_index


def calculate_absorption_strength(df: pd.DataFrame, window: int = 50) -> pd.Series:
    """
    Рассчитывает "Силу Поглощения".
    Это очень сложный индикатор. Здесь мы реализуем упрощенный прокси.
    Логика: Если при большом объеме торгов цена не падает, значит, есть поглощение.
    """
    print("Calculating Absorption Strength (Proxy)...")

    # Рассчитываем дельту объема (прокси для потока ордеров)
    price_diff = df['price'].diff()
    volume_delta = np.where(price_diff > 0, df['quantity'],
                            np.where(price_diff < 0, -df['quantity'], 0))
    volume_delta = pd.Series(volume_delta, index=df.index)

    # Накопленная дельта за окно
    cumulative_delta = volume_delta.rolling(window=window).sum()

    # Изменение цены за то же окно
    price_range = df['price'].rolling(window=window).max() - df['price'].rolling(window=window).min()

    # Сила поглощения: если дельта сильно отрицательная (много продаж),
    # а цена почти не изменилась, значит, было сильное поглощение.
    # Избегаем деления на ноль.
    absorption = cumulative_delta / (price_range + 1e-9)

    # Мы ищем сильное отрицательное значение (много продаж, малое изменение цены)
    return absorption
