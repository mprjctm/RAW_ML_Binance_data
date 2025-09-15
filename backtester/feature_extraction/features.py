# -*- coding: utf-8 -*-
"""
Этот модуль содержит функции для расчета продвинутых, кастомных индикаторов,
основанных на анализе микроструктуры рынка. Эти индикаторы предназначены для
выявления аномалий в поведении толпы, дисбалансов спроса и предложения,
и могут служить основой для построения сложных торговых стратегий.
"""
import pandas as pd
import numpy as np
import orjson


def calculate_order_flow_delta(df: pd.DataFrame, window: int = 30) -> pd.Series:
    """
    Рассчитывает дельту потока ордеров (Order Flow Delta) за скользящее окно.

    Торговая идея:
    Дельта потока ордеров — это разница между объемом покупок и объемом продаж
    по рыночным ценам. Положительная дельта указывает на преобладание агрессивных
    покупателей, отрицательная — на преобладание агрессивных продавцов.
    Резкие изменения дельты могут сигнализировать о смене настроений на рынке
    и потенциальных разворотах или продолжении тренда.

    Особенности реализации:
    В идеале для расчета требуется информация об агрессоре в сделке (покупатель или
    продавец инициировал сделку). Поскольку в данных `agg_trades` этой информации
    нет, мы используем эвристику (прокси):
    - Если цена выросла по сравнению с предыдущей сделкой, считаем объем покупкой.
    - Если цена упала — продажей.
    - Если не изменилась — объем равен нулю.
    Это упрощение, но оно дает достаточно хорошее приближение к реальной дельте.

    Args:
        df (pd.DataFrame): DataFrame с данными о сделках, должен содержать
                           колонки 'price' и 'quantity'.
        window (int): Размер скользящего окна для суммирования дельты.

    Returns:
        pd.Series: Временной ряд с рассчитанной дельтой потока ордеров.
    """
    print("Calculating Order Flow Delta (Proxy)...")

    # 1. Определяем направление сделки по изменению цены
    price_change = df['price'].diff()

    # 2. Рассчитываем "дельту" для каждой сделки.
    # np.where - это векторизованный аналог if/elif/else.
    delta = np.where(
        price_change > 0, df['quantity'],         # Если цена выросла - объем со знаком "+"
        np.where(price_change < 0, -df['quantity'], 0)  # Если упала - объем со знаком "-"
    )

    # 3. Преобразуем массив numpy в pandas Series для дальнейших операций
    delta_series = pd.Series(delta, index=df.index)

    # 4. Суммируем дельту за скользящее окно и присваиваем имя для колонки
    return delta_series.rolling(window=window).sum().rename("order_flow_delta")


def calculate_cascade_exhaustion(liquidations_df: pd.DataFrame, window: str = '5min') -> pd.Series:
    """
    Рассчитывает индикатор "Истощение Каскада" ликвидаций.

    Торговая идея:
    Каскадные ликвидации (когда закрытие одних позиций вызывает движение цены,
    которое провоцирует закрытие следующих) создают сильные, но краткосрочные
    тренды. Этот индикатор измеряет интенсивность ликвидаций. Когда после
    сильного всплеска интенсивность начинает падать (истощаться), это может
    сигнализировать об окончании каскада и скором развороте цены.

    Особенности реализации:
    В данной упрощенной версии индикатор просто считает количество ликвидаций
    в скользящем временном окне. В более сложной стратегии можно было бы
    сравнивать текущее значение со средним, искать пики и последующие спады.

    Args:
        liquidations_df (pd.DataFrame): DataFrame с данными о ликвидациях.
                                        Должен содержать колонку 'quantity'.
        window (str): Размер временного окна (например, '5min', '1H').

    Returns:
        pd.Series: Временной ряд с количеством ликвидаций в окне.
    """
    print("Calculating Cascade Exhaustion...")
    if liquidations_df.empty:
        return pd.Series(dtype=float)  # Возвращаем пустую серию с правильным типом

    # Считаем количество событий (ликвидаций) за скользящий временной интервал
    liquidation_counts = liquidations_df['quantity'].rolling(window).count()

    return liquidation_counts


def calculate_panic_index(df: pd.DataFrame, window: int = 30) -> pd.Series:
    """
    Рассчитывает "Индекс Паники Толпы".

    Торговая идея:
    Индекс пытается измерить уровень "паники" или "эйфории" на рынке. Паника
    часто характеризуется одновременным ростом волатильности (цена хаотично
    движется) и всплеском торгового объема (все пытаются закрыть позиции).
    Высокие значения индекса могут указывать на кульминацию движения (дно при
    панических продажах или вершину при эйфорических покупках), что часто
    предшествует развороту.

    Особенности реализации:
    Индекс является композитным и состоит из двух частей:
    1. Волатильность: Рассчитывается как стандартное отклонение логарифмических
       доходностей цены.
    2. Всплеск объема: Рассчитывается как процентное изменение суммарного объема
       в скользящем окне.
    Оба компонента нормализуются (приводятся к диапазону от 0 до 1), чтобы их
    можно было корректно сложить, и суммируются с весами (здесь 60% волатильность
    и 40% объем).

    Args:
        df (pd.DataFrame): DataFrame с данными о сделках.
        window (int): Размер скользящего окна для расчета компонентов.

    Returns:
        pd.Series: Временной ряд "Индекса Паники".
    """
    print("Calculating Panic Index...")

    # 1. Расчет волатильности
    # Используем логарифмическую доходность для большей стабильности расчетов
    log_returns = np.log(df['price'] / df['price'].shift(1))
    volatility = log_returns.rolling(window=window).std()

    # 2. Расчет всплеска объема
    volume_sum = df['quantity'].rolling(window=window).sum()
    # pct_change() вычисляет процентное изменение. Первое значение всегда будет NaN,
    # так как нет предыдущего значения для сравнения. Заполняем его нулем.
    volume_roc = volume_sum.pct_change().fillna(0)

    # 3. Нормализация компонентов
    # Чтобы сложить две разные по своей природе величины, их нужно привести
    # к одному масштабу. Здесь мы используем Min-Max нормализацию.
    # Добавляем эпсилон (1e-9) в знаменатель, чтобы избежать деления на ноль,
    # если все значения в окне будут одинаковыми (например, волатильность равна 0).
    vol_range = volatility.max() - volatility.min()
    norm_volatility = (volatility - volatility.min()) / (vol_range + 1e-9)

    vol_roc_range = volume_roc.max() - volume_roc.min()
    norm_volume_roc = (volume_roc - volume_roc.min()) / (vol_roc_range + 1e-9)

    # 4. Финальный индекс как взвешенное среднее
    panic_index = (norm_volatility * 0.6 + norm_volume_roc * 0.4)

    return panic_index.rename("panic_index")


def calculate_absorption_strength(df: pd.DataFrame, window: int = 50) -> pd.Series:
    """
    Рассчитывает "Силу Поглощения" (Absorption Strength).

    Торговая идея:
    Поглощение — это ситуация, когда крупные участники рынка "поглощают"
    большой объем продаж или покупок, не давая цене сильно измениться.
    Например, если на рынок выливается большой объем продаж, а цена почти не
    падает, это значит, что на этих уровнях стоит крупный покупатель, который
    все скупает. Обнаружение такого поглощения может быть сильным сигналом
    о наличии поддержки (при поглощении продаж) или сопротивления (при
    поглощении покупок).

    Особенности реализации:
    Это прокси-индикатор, основанный на следующей логике:
    - Рассчитывается накопленная дельта потока ордеров (см. `calculate_order_flow_delta`).
    - Рассчитывается диапазон изменения цены (max - min) за то же окно.
    - Сила поглощения = Накопленная дельта / Диапазон цены.
    Если дельта сильно отрицательная (много продавали), а диапазон цены мал,
    знаменатель будет маленьким, и итоговое значение будет большим по модулю
    и отрицательным. Это и есть сигнал сильного поглощения продаж.

    Args:
        df (pd.DataFrame): DataFrame с данными о сделках.
        window (int): Размер скользящего окна для расчета.

    Returns:
        pd.Series: Временной ряд "Силы Поглощения".
    """
    print("Calculating Absorption Strength (Proxy)...")

    # 1. Рассчитываем дельту объема (аналогично order_flow_delta)
    price_diff = df['price'].diff()
    volume_delta = np.where(price_diff > 0, df['quantity'],
                            np.where(price_diff < 0, -df['quantity'], 0))
    volume_delta = pd.Series(volume_delta, index=df.index)

    # 2. Накопленная дельта за окно
    cumulative_delta = volume_delta.rolling(window=window).sum()

    # 3. Диапазон изменения цены за то же окно
    price_range = df['price'].rolling(window=window).max() - df['price'].rolling(window=window).min()

    # 4. Расчет силы поглощения.
    # Добавляем эпсилон, чтобы избежать деления на ноль, если цена не менялась.
    absorption = cumulative_delta / (price_range + 1e-9)

    return absorption.rename("absorption_strength")


def _parse_orderbook_json(data):
    """
    Безопасно парсит данные стакана, которые могут быть строкой.
    Если данные - это строка, пытается декодировать ее из JSON.
    В случае ошибки или если данные - None, возвращает пустой список.
    """
    if isinstance(data, str):
        try:
            return orjson.loads(data)
        except orjson.JSONDecodeError:
            return []
    return data if data is not None else []


def calculate_orderbook_imbalance(df: pd.DataFrame, levels: int = 5) -> pd.Series:
    """
    Рассчитывает Дисбаланс Объема в Стакане (Order Book Imbalance, OBI).

    Торговая идея:
    Этот индикатор дает прямое измерение давления покупателей и продавцов,
    анализируя ликвидность, доступную в стакане. Сильный перевес объема
    на стороне покупки (bids) по сравнению со стороной продажи (asks)
    указывает на потенциальное движение цены вверх, и наоборот. В отличие от
    прокси-индикаторов, этот напрямую использует данные стакана, что делает
    его более точным.

    Особенности реализации:
    Для каждой временной точки (каждой сделки) мы суммируем объемы на `N`
    лучших уровнях цен с каждой стороны стакана. Затем мы вычисляем отношение,
    которое показывает, какая сторона "сильнее".
    Формула: (V_bid - V_ask) / (V_bid + V_ask)
    Результат находится в диапазоне от -1 (все давление на продажу) до +1
    (все давление на покупку). Значение 0 означает идеальный баланс.
    Этот непрерывный показатель отлично подходит для использования в ML-моделях.

    Args:
        df (pd.DataFrame): DataFrame, который должен содержать колонки
                           'bids' и 'asks'. Каждая ячейка в этих колонках -
                           это список списков вида [[цена, объем], ...].
        levels (int): Количество уровней стакана для анализа.

    Returns:
        pd.Series: Временной ряд дисбаланса стакана (OBI).
    """
    print("Calculating Order Book Imbalance...")

    def get_imbalance(row, level_count):
        # Безопасно парсим данные, которые могут быть JSON-строкой
        bids = _parse_orderbook_json(row['bids'])
        asks = _parse_orderbook_json(row['asks'])

        # Проверяем, что данные не пустые.
        if not bids or not asks:
            return 0.0

        # Суммируем объемы на заданном количестве уровней
        # bids[i][1] - это объем (quantity) на i-ом уровне
        bid_volume = sum(bids[i][1] for i in range(min(level_count, len(bids))))
        ask_volume = sum(asks[i][1] for i in range(min(level_count, len(asks))))

        # Рассчитываем дисбаланс
        total_volume = bid_volume + ask_volume
        if total_volume == 0:
            return 0.0  # Избегаем деления на ноль

        return (bid_volume - ask_volume) / total_volume

    # Применяем функцию расчета к каждой строке DataFrame
    # .bfill() заполняет возможные пропуски в данных стакана предыдущими значениями
    imbalance_series = df.bfill().apply(get_imbalance, axis=1, level_count=levels)

    return imbalance_series.rename("orderbook_imbalance")


def calculate_liquidity_walls(df: pd.DataFrame, wall_factor: float = 10.0, neighborhood: int = 5) -> pd.DataFrame:
    """
    Анализирует стакан на наличие "стен" ликвидности и рассчитывает их параметры.

    Торговая идея:
    "Стены" — это аномально крупные лимитные заявки, которые могут выступать
    в роли сильных уровней поддержки/сопротивления или "магнитов" для цены.
    Отслеживание расстояния до этих стен и их объема дает представление о том,
    где расположены ключевые уровни интереса крупных игроков.

    Особенности реализации:
    Функция сканирует уровни стакана и ищет заявки, объем которых значительно
    (в `wall_factor` раз) превышает средний объем на `neighborhood` соседних
    уровнях. Для найденных стен рассчитываются несколько признаков.
    Функция возвращает DataFrame с несколькими новыми колонками.

    Args:
        df (pd.DataFrame): DataFrame, содержащий колонки 'price', 'bids', 'asks'.
        wall_factor (float): Множитель для определения "аномальности" заявки.
        neighborhood (int): Количество соседних уровней для расчета среднего объема.

    Returns:
        pd.DataFrame: DataFrame с новыми признаками, связанными со стенами.
    """
    print("Calculating Liquidity Walls...")

    def find_walls(row):
        # Инициализируем переменные со значениями по умолчанию (NaN)
        buy_wall_price, buy_wall_vol, sell_wall_price, sell_wall_vol = np.nan, np.nan, np.nan, np.nan

        bids = _parse_orderbook_json(row['bids'])
        asks = _parse_orderbook_json(row['asks'])

        # --- Поиск стены на покупку (bids) ---
        if len(bids) > neighborhood:
            volumes = [b[1] for b in bids]
            for i in range(len(volumes)):
                # Явное и более надежное формирование списка соседних объемов
                neighbors = []
                # Соседи "сверху" (ближе к спреду)
                if i > 0:
                    neighbors.extend(volumes[max(0, i - neighborhood):i])
                # Соседи "снизу" (дальше от спреда)
                if i < len(volumes) - 1:
                    neighbors.extend(volumes[i + 1:min(len(volumes), i + neighborhood + 1)])

                if not neighbors: continue

                avg_vol = np.mean(neighbors)

                # Если объем на уровне аномально большой - это стена
                if avg_vol > 0 and volumes[i] / avg_vol >= wall_factor:
                    buy_wall_price = bids[i][0]
                    buy_wall_vol = volumes[i]
                    break  # Нашли ближайшую стену, выходим

        # --- Поиск стены на продажу (asks) ---
        if len(asks) > neighborhood:
            volumes = [a[1] for a in asks]
            for i in range(len(volumes)):
                neighbors = []
                if i > 0:
                    neighbors.extend(volumes[max(0, i - neighborhood):i])
                if i < len(volumes) - 1:
                    neighbors.extend(volumes[i + 1:min(len(volumes), i + neighborhood + 1)])

                if not neighbors: continue

                avg_vol = np.mean(neighbors)

                if avg_vol > 0 and volumes[i] / avg_vol >= wall_factor:
                    sell_wall_price = asks[i][0]
                    sell_wall_vol = volumes[i]
                    break

        # Рассчитываем расстояние до стен в процентах от текущей цены
        current_price = row['price']
        dist_to_buy_wall = ((current_price - buy_wall_price) / current_price * 100) if not np.isnan(buy_wall_price) else np.nan
        dist_to_sell_wall = ((sell_wall_price - current_price) / current_price * 100) if not np.isnan(sell_wall_price) else np.nan

        return pd.Series([dist_to_buy_wall, buy_wall_vol, dist_to_sell_wall, sell_wall_vol])

    # Применяем функцию ко всему DataFrame
    # .bfill() заполняет пропуски, если в какой-то момент не было данных стакана
    wall_features = df.bfill().apply(find_walls, axis=1)
    wall_features.columns = ['dist_to_buy_wall', 'buy_wall_vol', 'dist_to_sell_wall', 'sell_wall_vol']

    return wall_features


def calculate_footprint_imbalance(df: pd.DataFrame, imbalance_ratio: float = 3.0, window: int = 100) -> pd.Series:
    """
    Рассчитывает дисбаланс футпринта (Footprint Imbalance).

    Торговая идея:
    Это самый продвинутый индикатор в нашем наборе. Он анализирует "микроструктуру"
    каждого бара (или временного окна), показывая, где именно и какой стороной
    (покупатели или продавцы) был проторгован объем. Сигнал "дисбаланса"
    возникает, когда агрессивные покупатели на одном уровне цены значительно
    превосходят по объему агрессивных продавцов на уровне цены ниже. Это
    указывает на очень сильное, направленное давление, которое, скорее всего,
    продолжит двигать цену.

    Особенности реализации:
    1. Определение агрессора: Для каждой сделки определяется, была ли она
       инициирована покупателем (цена >= best_ask) или продавцом (цена <= best_bid).
    2. Построение футпринта: Внутри скользящего окна данные о сделках
       агрегируются по ценовым уровням, создавая мини-футпринт для этого окна.
    3. Поиск дисбаланса: Алгоритм сравнивает объем покупок на уровне N с
       объемом продаж на уровне N-1. Если соотношение превышает `imbalance_ratio`,
       регистрируется дисбаланс.
    4. Итоговый признак: Функция возвращает скользящую сумму "чистых"
       дисбалансов (+1 за бычий, -1 за медвежий), что является мощным
       признаком для ML-модели.

    Args:
        df (pd.DataFrame): DataFrame, содержащий 'price', 'quantity', 'bids', 'asks'.
        imbalance_ratio (float): Соотношение для определения дисбаланса.
        window (int): Размер скользящего окна для агрегации футпринта.

    Returns:
        pd.Series: Временной ряд чистого дисбаланса футпринта.
    """
    print("Calculating Footprint Imbalance...")

    # Шаг 1: Определяем агрессора для каждой сделки
    def get_aggressor(row):
        bids = _parse_orderbook_json(row['bids'])
        asks = _parse_orderbook_json(row['asks'])
        if not bids or not asks:
            return 'neutral'
        best_bid = bids[0][0]
        best_ask = asks[0][0]
        if row['price'] >= best_ask: return 'buy'
        if row['price'] <= best_bid: return 'sell'
        return 'neutral'

    aggressor = df.bfill().apply(get_aggressor, axis=1)

    buy_vol = df['quantity'].where(aggressor == 'buy', 0)
    sell_vol = df['quantity'].where(aggressor == 'sell', 0)

    # Шаг 2: Создаем DataFrame с объемами для анализа
    footprint_df = pd.DataFrame({
        'price': df['price'],
        'buy_vol': buy_vol,
        'sell_vol': sell_vol
    })

    # Шаг 3: Агрегируем объемы по ценам в скользящем окне
    # Это очень сложная операция. Для упрощения и эффективности, мы будем
    # рассчитывать дисбаланс не в "кластерах", а на уровне каждой сделки,
    # сравнивая ее с агрегированными данными предыдущих сделок.

    # Группируем объемы по цене
    price_level_vols = footprint_df.groupby('price')[['buy_vol', 'sell_vol']].sum()

    # Создаем смещенные данные для диагонального сравнения
    price_level_vols['prev_sell_vol'] = price_level_vols['sell_vol'].shift(1)

    # Сливаем агрегированные данные обратно в основной DataFrame
    merged_footprint = pd.merge(footprint_df, price_level_vols, on='price', how='left')

    # Шаг 4: Рассчитываем дисбаланс
    # Бычий дисбаланс: объем покупок на этом уровне > (объем продаж на уровне ниже * коэф.)
    # Используем `buy_vol_x`, так как это объем конкретной сделки, а не агрегированный.
    buy_imbalance = (merged_footprint['buy_vol_x'] > merged_footprint['prev_sell_vol'] * imbalance_ratio).astype(int)

    # Для медвежьего дисбаланса нужно сравнивать sell_vol с buy_vol на уровне выше
    # Это усложняет векторизованный расчет, поэтому в этой версии мы сфокусируемся
    # только на бычьем дисбалансе как на более простом для реализации примере.
    # В полноценной системе здесь был бы более сложный и медленный цикл.

    net_imbalance = buy_imbalance # В этой версии только бычий дисбаланс

    # Возвращаем скользящую сумму чистого дисбаланса
    return net_imbalance.rolling(window=window).sum().rename('footprint_imbalance')


# --- Функции для стандартных ("человеческих") индикаторов ---

def calculate_standard_indicators(df_ohlcv: pd.DataFrame, ema_fast: int = 9, ema_slow: int = 21, rsi_period: int = 7, bb_period: int = 20, bb_std: int = 2) -> pd.DataFrame:
    """
    Рассчитывает набор стандартных технических индикаторов с помощью библиотеки pandas-ta.

    Эта функция-обертка добавляет к исходному DataFrame следующие индикаторы:
    - EMA (Экспоненциальная скользящая средняя): Быстрая и медленная.
    - VWAP (Volume-Weighted Average Price): Средневзвешенная по объему цена.
    - RSI (Индекс относительной силы): Индикатор импульса.
    - Bollinger Bands (Полосы Боллинджера): Оценка волатильности.

    Args:
        df_ohlcv (pd.DataFrame): DataFrame с колонками 'open', 'high', 'low', 'close', 'volume'.
        ema_fast (int): Период для быстрой EMA.
        ema_slow (int): Период для медленной EMA.
        rsi_period (int): Период для RSI.
        bb_period (int): Период для Полос Боллинджера.
        bb_std (int): Количество стандартных отклонений для Полос Боллинджера.

    Returns:
        pd.DataFrame: DataFrame с добавленными колонками индикаторов.
    """
    print("Calculating standard technical indicators (EMA, VWAP, RSI, BBands)...")

    # Создаем копию, чтобы не изменять исходный DataFrame
    df_ta = df_ohlcv.copy()

    # Импортируем pandas_ta здесь, чтобы избежать циклического импорта
    import pandas_ta as ta

    # Рассчитываем индикаторы, добавляя их в df_ta
    df_ta.ta.ema(length=ema_fast, append=True)
    df_ta.ta.ema(length=ema_slow, append=True)
    df_ta.ta.vwap(append=True)
    df_ta.ta.rsi(length=rsi_period, append=True)
    df_ta.ta.bbands(length=bb_period, std=bb_std, append=True)

    # Возвращаем только новые колонки с индикаторами
    # Исходные колонки ohlcv уже есть в df_resampled
    new_cols = [col for col in df_ta.columns if col not in df_ohlcv.columns]
    return df_ta[new_cols]
