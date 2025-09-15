# -*- coding: utf-8 -*-
"""
Этот модуль содержит функции для расчета продвинутых, кастомных индикаторов,
основанных на анализе микроструктуры рынка. Эти индикаторы предназначены для
выявления аномалий в поведении толпы, дисбалансов спроса и предложения,
и могут служить основой для построения сложных торговых стратегий.
"""
import pandas as pd
import numpy as np


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
        # Извлекаем списки бидов и асков для текущей строки
        bids = row['bids']
        asks = row['asks']

        # Проверяем, что данные не пустые. Прямая проверка `if not bids` может
        # вызвать ValueError для сложных объектов. Явная проверка надежнее.
        if bids is None or asks is None or len(bids) == 0 or len(asks) == 0:
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


def _find_first_wall_vectorized(prices_df: pd.DataFrame,
                                vols_df: pd.DataFrame,
                                wall_factor: float,
                                neighborhood: int):
    """
    Helper-функция для векторизованного поиска стен ликвидности.

    Args:
        prices_df (pd.DataFrame): DataFrame с ценами уровней (строки - таймстемпы, колонки - уровни).
        vols_df (pd.DataFrame): DataFrame с объемами уровней.
        wall_factor (float): Множитель для определения "аномальности" заявки.
        neighborhood (int): Количество соседних уровней для расчета среднего объема.

    Returns:
        tuple[pd.Series, pd.Series]: Серии с ценой и объемом найденной стены для каждой строки.
    """
    # 1. Расчет среднего объема соседей с использованием скользящих окон
    # Суммируем объемы в окне размером `2 * neighborhood + 1` с центром в текущем элементе
    rolling_sum = vols_df.rolling(window=2 * neighborhood + 1, center=True, min_periods=1, axis=1).sum()
    # Определяем количество непропущенных значений в том же окне
    rolling_count = vols_df.notna().rolling(window=2 * neighborhood + 1, center=True, min_periods=1, axis=1).sum()

    # Вычитаем собственный объем и количество, чтобы получить сумму и количество только соседей
    neighbor_sum = rolling_sum - vols_df.fillna(0)
    neighbor_count = rolling_count - vols_df.notna().astype(int)

    # Рассчитываем среднее, избегая деления на ноль
    with np.errstate(divide='ignore', invalid='ignore'):
        avg_neighbor_vol = neighbor_sum / neighbor_count.replace(0, np.nan)

    # 2. Определение стен
    # Стена - это уровень, где объем значительно превышает средний объем соседей
    is_wall = (vols_df > (avg_neighbor_vol * wall_factor)) & (avg_neighbor_vol > 0)

    # 3. Поиск первой стены для каждой строки
    # idxmax() найдет индекс первого True (т.к. True=1, False=0).
    # Для строк без стен (все False), idxmax вернет индекс первой колонки.
    wall_indices = is_wall.idxmax(axis=1)

    # Создаем маску, чтобы отфильтровать строки, где стен не найдено
    has_wall_mask = is_wall.any(axis=1)
    # Применяем маску: где стен нет, индекс будет NaN
    wall_indices = wall_indices.where(has_wall_mask, np.nan)

    # 4. Извлечение цены и объема стены
    # Готовим индексы для продвинутой выборки NumPy
    row_idx = np.arange(len(prices_df))
    # Для строк без стен используем 0 как временный индекс, они все равно будут отфильтрованы
    col_idx = wall_indices.fillna(0).astype(int).values

    # Извлекаем цены и объемы стен с помощью NumPy-индексации
    wall_prices = pd.Series(prices_df.to_numpy()[row_idx, col_idx], index=prices_df.index)
    wall_vols = pd.Series(vols_df.to_numpy()[row_idx, col_idx], index=vols_df.index)

    # Применяем маску к результатам
    wall_prices = wall_prices.where(has_wall_mask, np.nan)
    wall_vols = wall_vols.where(has_wall_mask, np.nan)

    return wall_prices, wall_vols


def calculate_liquidity_walls(df: pd.DataFrame, wall_factor: float = 10.0, neighborhood: int = 5, max_levels: int = 20) -> pd.DataFrame:
    """
    Анализирует стакан на наличие "стен" ликвидности и рассчитывает их параметры.
    (Векторизованная версия)

    Торговая идея:
    "Стены" — это аномально крупные лимитные заявки, которые могут выступать
    в роли сильных уровней поддержки/сопротивления или "магнитов" для цены.
    Отслеживание расстояния до этих стен и их объема дает представление о том,
    где расположены ключевые уровни интереса крупных игроков.

    Особенности реализации:
    Эта версия полностью векторизована с использованием Pandas и NumPy для
    максимальной производительности на больших датасетах. Она избегает
    медленных операций `.apply()` по строкам.
    1. Данные стакана ('bids', 'asks') разворачиваются в широкие DataFrames,
       где каждая колонка представляет один уровень стакана.
    2. Расчет среднего объема соседей выполняется с помощью скользящих окон.
    3. Поиск стен и извлечение их параметров производятся с помощью
       векторизованных логических операций и NumPy-индексации.

    Args:
        df (pd.DataFrame): DataFrame, содержащий колонки 'price', 'bids', 'asks'.
        wall_factor (float): Множитель для определения "аномальности" заявки.
        neighborhood (int): Количество соседних уровней для расчета среднего объема.
        max_levels (int): Максимальная глубина стакана для анализа. Урезание до
                          разумного значения (например, 20) сильно повышает
                          производительность.

    Returns:
        pd.DataFrame: DataFrame с новыми признаками, связанными со стенами.
    """
    print("Calculating Liquidity Walls (Vectorized)...")

    # --- 1. Подготовка данных ---
    # Заполняем пропуски, если в какой-то момент не было данных стакана
    df_filled = df[['price', 'bids', 'asks']].bfill()

    # Функция для быстрой распаковки данных стакана в широкие DataFrame'ы
    def unpack_book_to_wide_dfs(book_series: pd.Series):
        # Преобразуем серию списков в список списков
        book_list = book_series.tolist()
        # Урезаем и дополняем до max_levels.
        # [[price, vol], [price, vol], ...] ->
        # [[p1, v1], [p2, v2], ..., [nan, nan]]
        padded_list = [
            (sublist[:max_levels] + [[np.nan, np.nan]] * max_levels)[:max_levels]
            if sublist else [[np.nan, np.nan]] * max_levels for sublist in book_list
        ]
        # Создаем 3D NumPy массив для эффективности
        arr = np.array(padded_list, dtype=np.float64)
        # Разделяем на цены и объемы
        prices = pd.DataFrame(arr[:, :, 0], index=book_series.index)
        volumes = pd.DataFrame(arr[:, :, 1], index=book_series.index)
        return prices, volumes

    # Распаковываем биды и аски
    bid_prices, bid_vols = unpack_book_to_wide_dfs(df_filled['bids'])
    ask_prices, ask_vols = unpack_book_to_wide_dfs(df_filled['asks'])

    # --- 2. Поиск стен ---
    # Находим стены на покупку (bids) и продажу (asks)
    buy_wall_price, buy_wall_vol = _find_first_wall_vectorized(bid_prices, bid_vols, wall_factor, neighborhood)
    sell_wall_price, sell_wall_vol = _find_first_wall_vectorized(ask_prices, ask_vols, wall_factor, neighborhood)

    # --- 3. Расчет финальных признаков ---
    # Рассчитываем расстояние до стен в процентах от текущей цены
    current_price = df_filled['price']
    dist_to_buy_wall = (current_price - buy_wall_price) / current_price * 100
    dist_to_sell_wall = (sell_wall_price - current_price) / current_price * 100

    # --- 4. Формирование итогового DataFrame ---
    wall_features = pd.DataFrame({
        'dist_to_buy_wall': dist_to_buy_wall,
        'buy_wall_vol': buy_wall_vol,
        'dist_to_sell_wall': dist_to_sell_wall,
        'sell_wall_vol': sell_wall_vol
    })

    return wall_features


def calculate_footprint_imbalance(df: pd.DataFrame, imbalance_ratio: float = 3.0, window: int = 100) -> pd.Series:
    """
    Рассчитывает чистый дисбаланс футпринта (Net Footprint Imbalance).
    (Векторизованная версия)

    Торговая идея:
    Этот индикатор анализирует "микроструктуру" каждой сделки, сравнивая
    агрессивные покупки и продажи на соседних ценовых уровнях.
    - **Бычий дисбаланс**: Объем покупок на цене N значительно превышает
      объем продаж на цене N-1. Это говорит о силе покупателей.
    - **Медвежий дисбаланс**: Объем продаж на цене N значительно превышает
      объем покупок на цене N+1. Это говорит о силе продавцов.

    Особенности реализации:
    Функция возвращает скользящую сумму "чистых" дисбалансов
    (+1 за бычий, -1 за медвежий), что является мощным признаком для ML-модели.
    Реализация полностью векторизована для максимальной производительности.

    Args:
        df (pd.DataFrame): DataFrame, содержащий 'price', 'quantity', 'bids', 'asks'.
        imbalance_ratio (float): Соотношение для определения дисбаланса.
        window (int): Размер скользящего окна для агрегации чистого дисбаланса.

    Returns:
        pd.Series: Временной ряд чистого дисбаланса футпринта.
    """
    print("Calculating Footprint Imbalance (Vectorized)...")

    # --- 1. Векторизованное определение агрессора ---
    df_filled = df[['price', 'quantity', 'bids', 'asks']].bfill()
    best_bid = pd.to_numeric(df_filled['bids'].str[0].str[0], errors='coerce')
    best_ask = pd.to_numeric(df_filled['asks'].str[0].str[0], errors='coerce')
    is_buy = df_filled['price'] >= best_ask
    is_sell = df_filled['price'] <= best_bid
    buy_vol = df_filled['quantity'].where(is_buy, 0)
    sell_vol = df_filled['quantity'].where(is_sell, 0)

    # --- 2. Агрегация объемов по ценовым уровням ---
    footprint_df = pd.DataFrame({
        'price': df_filled['price'],
        'buy_vol': buy_vol,
        'sell_vol': sell_vol
    })
    price_level_vols = footprint_df.groupby('price')[['buy_vol', 'sell_vol']].sum()

    # --- 3. Расчет диагональных дисбалансов ---
    # Для бычьего дисбаланса: сравниваем buy_vol(P) с sell_vol(P-1)
    price_level_vols['prev_sell_vol'] = price_level_vols['sell_vol'].shift(1).fillna(0)
    price_to_prev_sell_map = price_level_vols['prev_sell_vol']
    prev_sell_vol_series = footprint_df['price'].map(price_to_prev_sell_map).fillna(0)
    buy_imbalance = (footprint_df['buy_vol'] > prev_sell_vol_series * imbalance_ratio).astype(int)

    # Для медвежьего дисбаланса: сравниваем sell_vol(P) с buy_vol(P+1)
    price_level_vols['next_buy_vol'] = price_level_vols['buy_vol'].shift(-1).fillna(0)
    price_to_next_buy_map = price_level_vols['next_buy_vol']
    next_buy_vol_series = footprint_df['price'].map(price_to_next_buy_map).fillna(0)
    sell_imbalance = (footprint_df['sell_vol'] > next_buy_vol_series * imbalance_ratio).astype(int)

    # --- 4. Расчет чистого дисбаланса и финальный результат ---
    # Чистый дисбаланс: +1 за бычий, -1 за медвежий
    net_imbalance = buy_imbalance - sell_imbalance

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
