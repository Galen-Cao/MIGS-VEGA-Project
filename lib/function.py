import numpy as np
import pandas as pd
import operator
from scipy.linalg import expm

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


def MC_sim(
    S0,
    Q0, 
    X0, 
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

    eta = (S0+sell_depth)*lambda_buy*np.exp(- kappa_buy*lambda_buy) - (S0-buy_depth)*lambda_sell*np.exp(- kappa_sell*lambda_sell)

    # Dynamic process
    length = len(t)

    cash = []
    inventory = []
    Compensated = []

    for _ in range(batch_size):
        # Initialize the dynamics
        dW = np.sqrt(dt) * np.random.randn(length - 1)
        S = np.zeros(length)  # Midprice
        S[0] = S0
        Q = np.zeros(length)  # Inventory
        Q[0] = Q0
        X = np.zeros(length)  # Cash
        X[0] = X0
        c_cash = np.zeros(length)
        c_cash[0] = X0 
 
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
            
            c_cash[time_step] = c_cash[time_step-1] + X[time_step] - X[time_step-1] - eta*dt

        cash.append(X[-1])
        inventory.append(Q[-1])
        Compensated.append(c_cash[-1])

    return cash, inventory, Compensated
        

def MC_sim_torch(
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


def VegaModel_Optimal(
    T: float,
    dt: float,
    q_upper: int = 10,
    q_lower: int = -10,
    kappa: int = 50,
    Lambda_buy: int = 1,
    Lambda_sell: int = 1,
    phi: float = 10**-6,
):

    """
    Simulate an Market Making Model for cash settled futures and output
        the optimal strategy of posting bid/ask depth.

    Args:
        T:
            int, total simulation time (second)
        dt:
            float, the increment of each time step (second)
        mdp:
            int, market decimal place
        q_upper:
            int, upper bound of the inventory of MM specified in the model
        q_lower:
            int, lower bound of the inventory of MM specified in the model
        kappa:
            int, market parameter to represnet the probability of pegged LOs to be hit
        Lambda:
            int, market parameter to represnet the coming rate of MOs
        phi:
            float, risk aversion parameter to represnet running penalty coefficient
    """
    length = int(T / dt) + 1

    # Initialize 2-d arrays to store optimal strategy
    optimal_depth_bid = np.zeros([length, q_upper - q_lower])
    optimal_depth_ask = np.zeros([length, q_upper - q_lower])

    # Let A be a (q_upper-q_lower+1)-square matrix
    A = np.zeros([q_upper - q_lower + 1, q_upper - q_lower + 1])
    # w, time * (q_upper-q_lower+1)-dim matrix, to store the solution of ODE
    #   row corresponds to time_step, column corresponds to inventory q
    w = np.zeros([length, q_upper - q_lower + 1])

    # A is the coefficient matrix of ODE
    for i in range(q_upper - q_lower + 1):
        for j in range(q_upper - q_lower + 1):
            # i denotes row/ j denotes column
            if j == i:
                A[i, j] = -kappa * phi * (q_upper - i) ** 2
            elif j == i + 1:
                A[i, j] = Lambda_buy * np.e**-1
            elif j == i - 1:
                A[i, j] = Lambda_sell * np.e**-1

    # z, (q_upper-q_lower+1)-dim vector, denotes the terminal condition of ODE
    z = np.ones(q_upper - q_lower + 1)

    for i in range(length):
        # at each time_step
        w[i, :] = np.dot(expm(A * (T - i * dt)), z)

    # h is the transformation of solution from ODE
    #   also the key term of value function
    h = np.log(w) / kappa

    # Calculate optimal strategy
    for i in range(q_upper - q_lower):
        # column corresponds to Q, Q-1,..., -Q+1
        optimal_depth_ask[:, i] = 1 / kappa + h[:, i] - h[:, i + 1]

    for i in range(1, q_upper - q_lower + 1):
        # column corresponds to Q-1, Q-2,..., -Q
        optimal_depth_bid[:, i - 1] = 1 / kappa + h[:, i] - h[:, i - 1]

    return optimal_depth_bid, optimal_depth_ask, h

        
def margin_calculator(
    markprice: float,
    midprice: float,
    position: int,
    buy_depth: float,
    sell_depth: float,
    risk_factor_long: float,
    risk_factor_short: float,
    commitamount: int = 100,
    pp: float = 0.05, 
):

    """
    Args:
        pp: midprice percentage, the market parameter of Vega

    If buy_depth = sell_depth == inf, it means LP cancel the liquidity provision 
    """
    if buy_depth == np.inf and sell_depth == np.inf:
        # the LP cancel the liquidity provision
        maintenance_long =  position * markprice * risk_factor_long
        maintenance_short = - position * markprice * risk_factor_long

    else: 
        long_max = max(midprice-buy_depth, (1-pp)*midprice)
        long_short = min(midprice+sell_depth, (1+pp)*midprice)
        maintenance_long = position * markprice * risk_factor_long + commitamount * markprice * risk_factor_long / long_max
        maintenance_short = -position * markprice * risk_factor_short + + commitamount * markprice * risk_factor_short / long_short
    
    margin = max(maintenance_long, maintenance_short)
    return margin

def control_vegamodel(
    S0,
    X0, 
    q, 
    lambda_buy,
    lambda_sell,
    kappa_buy,
    kappa_sell,
    buy_depth,
    sell_depth,
    sigma,
    T,
    time_decimal = 3,
    spread = 0.04,
    batch_size = 5000, 
):

    """
    Controllability for Vega Market Making model
    """
    # set parameter
    dt = 1 / 10**time_decimal
    T = int(T*10**time_decimal)
    t = np.linspace(0, T,T + 1) / 10**time_decimal
    t = np.round(t, time_decimal)

    # Dynamic process
    length = len(t)

    collateral = []
    inventory = []
    max_margin = []

    for _ in range(batch_size):
        # Initialise dynamics
        dW = np.sqrt(dt)*np.random.randn(length-1)
        S = np.zeros(length) # Midprice
        S[0] = S0
        P = np.zeros(length) # Markprice
        P[0] = S0
        Q = np.zeros(length) # Inventory
        Q[0] = q
        X = np.zeros(length) # Total collateral
        X[0] = X0

        # Gnerate MOs in time [0 ,T]
        MOs_time = MO_generator(
            period=T,
            buy_mo_intensity=lambda_buy,
            sell_mo_intensity=lambda_sell,
            time_decimal=time_decimal,
        )
        margin = []
        for time_step in range(1, length):
            S[time_step] = S[time_step-1] + sigma*dW[time_step-1]

            if t[time_step] in MOs_time.index.values:
                if MOs_time.loc[t[time_step]][0] == 'sell':
                    # sell MO coming, judge if the sell MO hits MM's buy LO
                    prob = np.exp(- buy_depth * kappa_buy)
                    dN_t = np.random.binomial(1, prob)
                    dA_t = 1 - dN_t
                    # currently, we assume the epsilon is 0.5, later try to sample from a scaled-beta or normal distribution 
                    epsilon = spread / 2

                    # inventory process jumps 
                    Q[time_step] = Q[time_step-1] + dN_t
                    # mark price jumps
                    P[time_step] = (S[time_step] - buy_depth) * dN_t + (S[time_step] - epsilon) * dA_t
                else:
                    # buy MO coming
                    prob = np.exp(- sell_depth * kappa_sell)
                    dN_t = np.random.binomial(1, prob)
                    dA_t = 1 - dN_t
                    epsilon = spread / 2

                    # inventory process jump 
                    Q[time_step] = Q[time_step-1] - dN_t
                    # mark price jumps
                    P[time_step] = (S[time_step] + sell_depth) * dN_t + (S[time_step] + epsilon) * dA_t

            # No MO coming
            else:
                P[time_step] = P[time_step-1]
                X[time_step] = X[time_step-1]
                Q[time_step] = Q[time_step-1]

            margin.append(margin_calculator(
                        markprice=P[time_step],
                        midprice=S[time_step],
                        position=Q[time_step],
                        buy_depth=buy_depth,
                        sell_depth=sell_depth,
                        risk_factor_long=0.336896,
                        risk_factor_short=0.487873,
                    ))
            # cash process updates
            X[time_step] = X[time_step-1] + Q[time_step-1]*(P[time_step]-P[time_step-1])
                
        collateral.append(X[-1])
        inventory.append(Q[-1])
        max_margin.append(max(margin))
    return collateral, inventory, max_margin


def control_vega_torch(
    S0,
    X0, 
    q, 
    lambda_buy,
    lambda_sell,
    kappa_buy,
    kappa_sell,
    buy_depth,
    sell_depth,
    sigma,
    T,
    time_decimal = 3,
    spread = 0.04,
    batch_size = 5000, 
):
    """
    Use PyTorch
    """
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

    X = X0 * torch.ones(batch_size, 1)
    S = S0 * torch.ones(batch_size, 1)
    P = S0 * torch.ones(batch_size, 1) 
    Q = q * torch.ones(batch_size, 1)
    Q2 = q**2 * torch.ones(batch_size, 1)
    x0 = torch.cat((X, S, P, Q, Q2), 1)
    epsilon = spread / 2

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
        # x denotes the tuple (X, S, P, Q)
        P_old = x[:, -1, 2].unsqueeze(1) # (batch_size, 1)
        Q_old = x[:, -1, 3].unsqueeze(1) # (batch_size, 1)

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
        dA_sell = buyMO_come - dN_sell # (batch_size, 1)

        sellMO_come = torch.sum(torch.any(time_sellMOs == t, axis=1).unsqueeze(1), axis=1).unsqueeze(1)
        prob_buyside = torch.exp(-buy_depth * kappa_buy)
        dN_buy = Binomial(sellMO_come, prob_buyside).sample()
        dA_buy = sellMO_come - dN_buy

        # Update markprice process 
        P_new = (
            P_old + (S_new-P_old+sell_depth)*dN_sell + (S_new-P_old-buy_depth)*dN_buy + (S_new-P_old+epsilon)*dA_sell + (S_new-P_old-epsilon)*dA_buy
        ) # (batch_size, 1)        
        # Update inventory process
        Q_new = (
            x[:, -1, 3].unsqueeze(1) + dN_buy - dN_sell
        ) # (batch_size, 1)

        # Update collateral process
        X_new = (
            x[:, -1, 0].unsqueeze(1) + Q_old*(P_new - P_old)
        ) # (batch_size, 1)

        Q2_new = Q_new**2
        # x denotes the tuple (X, S, P, Q)
        x_new = torch.cat((X_new, S_new, P_new, Q_new, Q2_new), 1)  # (batch_size, 4)
        x = torch.cat([x, x_new.unsqueeze(1)], 1)  # (batch_size, N ,4)

    return x