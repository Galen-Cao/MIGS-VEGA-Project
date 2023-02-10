import numpy as np
import pandas as pd
import operator

import torch
from torch.distributions.binomial import Binomial

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


def controllability(
    S0,
    a, 
    m, 
    lambda_buy,
    lambda_sell,
    kappa_buy,
    kappa_sell,
    buy_depth,
    sell_depth,
    sigma,
    T,
    time_decimal = 3,
    batch_size = 5000, 
):
    """
    Explore the controllability
    """
    # set parameter
    dt = 1 / 10**time_decimal
    T = int(T*10**time_decimal)
    t = np.linspace(0, T,T + 1) / 10**time_decimal
    t = np.round(t, time_decimal)

    # Dynamic process
    length = len(t)

    cash = []
    inventory = []

    for _ in range(batch_size):
        # Initialize the dynamics
        dW = np.sqrt(dt) * np.random.randn(length - 1)
        S = np.zeros(length)  # Midprice
        S[0] = S0
        Q = np.zeros(length)  # Inventory
        Q[0] = a
        X = np.zeros(length)  # Cash
        X[0] = m
    
        # generate coming MO in time [0,T]
        MOs_time = MO_generator(
            period=T,
            buy_mo_intensity=lambda_buy,
            sell_mo_intensity=lambda_sell,
            time_decimal=time_decimal,
        )

        for time_step in range(1, length):
            S[time_step] = S[time_step-1] + sigma*dW[time_step-1]
            
            # MO coming
            if t[time_step] in MOs_time.index.values:

                if MOs_time.loc[t[time_step]][0] == 'sell':
                    # sell MO coming, judge if the sell MO hits MM's buy LO
                    prob = np.exp(- buy_depth * kappa_buy)
                    dN_t = np.random.binomial(1, prob)
                    # inventory process jumps 
                    Q[time_step] = Q[time_step-1] + dN_t
                    # cash jumps
                    X[time_step] = X[time_step-1] - (S[time_step-1] - buy_depth) * dN_t
                    
                else:
                    # buy MO coming
                    prob = np.exp(- sell_depth * kappa_sell)
                    dN_t = np.random.binomial(1, prob)
                    # inventory process jump 
                    Q[time_step] = Q[time_step-1] - dN_t
                    # cash jumps
                    X[time_step] = X[time_step-1] + (S[time_step-1] + sell_depth) * dN_t

            # No MO coming
            else:
                X[time_step] = X[time_step-1]
                Q[time_step] = Q[time_step-1]

        cash.append(X[-1])
        inventory.append(Q[-1])

    return cash, inventory
        

def controllability_torch(
    S0,
    a, 
    m, 
    lambda_buy,
    lambda_sell,
    kappa_buy,
    kappa_sell,
    buy_depth,
    sell_depth,
    sigma,
    T,
    time_decimal = 3,
    batch_size = 5000, 
):

    def Poisson_process(ts, time_decimal, intensity, batch_size):
        """
        Generate a Possion process 
        """
        intervals = (
            -torch.log(torch.rand(batch_size, int(2 * ts[-1] * intensity))) / intensity
        )
        pois_p = torch.cumsum(intervals, dim=1)
        pois_p = (pois_p * 10**time_decimal).round() / (10**time_decimal)

        if all(pois_p[:, -1] >= float(ts[-1])):
            return pois_p

        else:
            a = 2
            while not all(pois_p[:, -1] >= float(ts[-1])):
                a = a*2
                intervals = (
                    -torch.log(torch.rand(batch_size, int(a * ts[-1] * intensity)))
                    / intensity
                )
                pois_p = torch.cumsum(intervals, dim=1)
                pois_p = (pois_p * 10**time_decimal).round() / (10**time_decimal)
            return pois_p
    
    X = m * torch.ones(batch_size, 1)
    S = S0 * torch.ones(batch_size, 1)
    Q = a * torch.ones(batch_size, 1)
    x0 = torch.cat((X, S, Q), 1)

    x = x0.unsqueeze(1)

    T = int(T*10**time_decimal)

    ts = torch.linspace(0, T, T+1) / 10**time_decimal
    ts = torch.round(ts, decimals=time_decimal)

    # only process S needs brownian motion
    brownian_increments = torch.zeros(batch_size, len(ts), 1)
    time_buyMOs = Poisson_process(ts, time_decimal, lambda_buy, batch_size)
    time_sellMOs = Poisson_process(ts, time_decimal, lambda_sell, batch_size)

    h = ts[1] - ts[0]
    for idx, t in enumerate(ts[1:]):

        brownian_increments[:, idx, :] = torch.randn(
            batch_size, 1,
        ) * torch.sqrt(h)

        # Update midprice
        S_new = (
            x[:, -1, 1].unsqueeze(1) + sigma * brownian_increments[:, idx, :]
        )  # (batch_size, 1)

        sell_depth = torch.ones(batch_size, 1) * sell_depth 
        buy_depth = torch.ones(batch_size, 1) * buy_depth 

        buyMO_come = torch.sum(torch.any(time_buyMOs == t, axis=1).unsqueeze(1), axis=1).unsqueeze(1)
        prob_sellside = torch.exp(-sell_depth * kappa_sell)
        dN_sell = Binomial(buyMO_come, prob_sellside).sample() # (batch_size, 1)

        sellMO_come = torch.sum(torch.any(time_sellMOs == t, axis=1).unsqueeze(1), axis=1).unsqueeze(1)
        prob_buyside = torch.exp(-buy_depth * kappa_buy)
        dN_buy = Binomial(sellMO_come, prob_buyside).sample()

        # Update inventory process
        Q_new = (
            x[:, -1, 2].unsqueeze(1) + dN_buy - dN_sell
        ) # (batch_size, 1)

        # Update cash process
        X_new = (
            x[:, -1, 0].unsqueeze(1) + (x[:,-1,1].unsqueeze(1) + sell_depth) * dN_sell - (x[:,-1,1].unsqueeze(1) - buy_depth) * dN_buy
        ) # (batch_size, 1)

        # x denotes the tuple (X, S, Q)
        x_new = torch.cat((X_new, S_new, Q_new), 1)  # (batch_size, 3)
        x = torch.cat([x, x_new.unsqueeze(1)], 1)  # (batch_size, N ,3)

    return x


        
