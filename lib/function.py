import numpy as np
import pandas as pd
import operator


def MO_generator(
    period: int = 100,
    buy_mo_intensity: float = 1,
    sell_mo_intensity: float = 1,
    time_decimal: int = 3,
) -> pd.DataFrame:
    """
    Generate buy/sell MOs simultaneously under given
    intensites and time period by Poisson Process.
    """

    # independently deal with buy/sell MOs
    buy_MOs_time = [0]
    sell_MOs_time = [0]

    while True:
        intervals_buy = -np.log(np.random.random(1)) / buy_mo_intensity
        buy_MOs_time.append(round(intervals_buy[0] + buy_MOs_time[-1], time_decimal))
        if buy_MOs_time[-1] > period:
            break

    while True:
        intervals_sell = -np.log(np.random.random(1)) / sell_mo_intensity
        sell_MOs_time.append(round(intervals_sell[0] + sell_MOs_time[-1], time_decimal))
        if sell_MOs_time[-1] > period:
            break

    buy_MOs_time = buy_MOs_time[1:]
    sell_MOs_time = sell_MOs_time[1:]

    MOs_dict = {
        key: value
        for key, value in zip(
            buy_MOs_time + sell_MOs_time,
            ["buy"] * len(buy_MOs_time) + ["sell"] * len(sell_MOs_time),
        )
    }
    # sorted by coming time
    MOs_dict = dict(sorted(MOs_dict.items(), key=operator.itemgetter(0)))
    MOs_time = pd.DataFrame.from_dict([MOs_dict]).T
    return MOs_time