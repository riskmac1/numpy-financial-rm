from decimal import Decimal

import numpy as np

import numpy_financial as npf


def _to_decimal_array_1d(array):
    return np.array([Decimal(x) for x in array.tolist()])


def _to_decimal_array_2d(array):
    decimals = [Decimal(x) for row in array.tolist() for x in row]
    return np.array(decimals).reshape(array.shape)


class Npv2D:

    param_names = ["n_cashflows", "cashflow_lengths", "rates_lengths"]
    params = [
        (1, 10, 100),
        (1, 10, 100),
        (1, 10, 100),
    ]

    def __init__(self):
        self.rates_decimal = None
        self.rates = None
        self.cashflows_decimal = None
        self.cashflows = None

    def setup(self, n_cashflows, cashflow_lengths, rates_lengths):
        rng = np.random.default_rng(0)
        cf_shape = (n_cashflows, cashflow_lengths)
        self.cashflows = rng.standard_normal(cf_shape)
        self.rates = rng.standard_normal(rates_lengths)
        self.cashflows_decimal = _to_decimal_array_2d(self.cashflows)
        self.rates_decimal = _to_decimal_array_1d(self.rates)

    def time_broadcast(self, n_cashflows, cashflow_lengths, rates_lengths):
        npf.npv(self.rates, self.cashflows)

    def time_for_loop(self, n_cashflows, cashflow_lengths, rates_lengths):
        for rate in self.rates:
            for cashflow in self.cashflows:
                npf.npv(rate, cashflow)

    def time_broadcast_decimal(self, n_cashflows, cashflow_lengths, rates_lengths):
        npf.npv(self.rates_decimal, self.cashflows_decimal)

    def time_for_loop_decimal(self, n_cashflows, cashflow_lengths, rates_lengths):
        for rate in self.rates_decimal:
            for cashflow in self.cashflows_decimal:
                npf.npv(rate, cashflow)


class Fv2D:
    param_names = ["n_rates", "n_periods", "n_pmts", "n_pv"]
    params = [
        (1, 10, 100),
        (1, 10, 100),
        (1, 10, 100),
        (1, 10, 100),
    ]

    def __init__(self):
        self.present_value = None
        self.payments = None
        self.periods = None
        self.rates = None

    def setup(self, n_rates, n_periods, n_pmts, n_pv):
        rng = np.random.default_rng(42)
        self.rates = rng.standard_normal(n_rates)
        self.periods = rng.standard_normal(n_periods)
        self.payments = rng.standard_normal(n_pmts)
        self.present_value = rng.standard_normal(n_pv)
        self.rates_decimal = _to_decimal_array_1d(self.rates)
        self.periods_decimal = _to_decimal_array_1d(self.periods)
        self.payments_decimal = _to_decimal_array_1d(self.payments)
        self.present_value_decimal = _to_decimal_array_1d(self.present_value)

    def time_broadcast(self, n_rates, n_periods, n_pmts, n_pv):
        npf.fv(self.rates, self.periods, self.payments, self.present_value)

    def time_broadcast_decimal(self, n_rates, n_periods, n_pmts, n_pv):
        npf.fv(
            self.rates_decimal,
            self.periods_decimal,
            self.payments_decimal,
            self.present_value_decimal
        )

    def time_naive_for_loop(self, n_rates, n_periods, n_pmts, n_pv):
        for rate in self.rates:
            for period in self.periods:
                for payment in self.payments:
                    for present_value in self.present_value:
                        npf.fv(rate, period, payment, present_value)

    def time_naive_for_loop_decimal(self, n_rates, n_periods, n_pmts, n_pv):
        for rate in self.rates_decimal:
            for period in self.periods_decimal:
                for payment in self.payments_decimal:
                    for present_value in self.present_value_decimal:
                        npf.fv(rate, period, payment, present_value)
