# -*- coding: utf-8 -*-

"""
quant_greeks.ref_python.black_scholes.implied_volatility
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A library for option pricing, implied volatility, and
greek calculation.  quant_greeks is based on lets_be_rational,
a Python wrapper for LetsBeRational by Peter Jaeckel as
described below.

:copyright: © 2017 Gammon Capital LLC
:license: MIT, see LICENSE for more details.

quant_greeks.ref_python is a pure python version of quant_greeks without any dependence on LetsBeRational. It is provided purely as a reference implementation for sanity checking. It is not recommended for industrial use.
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

"""


# -----------------------------------------------------------------------------
# IMPORTS

# Standard library imports

# Related third party imports
from scipy.optimize import brentq

# Local application/library specific imports
from quant_greeks.ref_python.black_scholes import black_scholes


# -----------------------------------------------------------------------------
# FUNCTIONS

def implied_volatility(price, S, K, t, r, flag):
    """Calculate the Black-Scholes implied volatility.

    :param price: the Black-Scholes option price
    :type price: float
    :param S: underlying asset price
    :type S: float
    :param K: strike price
    :type K: float
    :param t: time to expiration in years
    :type t: float
    :param r: risk-free interest rate
    :type r: float
    :param flag: 'c' or 'p' for call or put.
    :type flag: str

    >>> S = 100
    >>> K = 100
    >>> sigma = .2
    >>> r = .01
    >>> flag = 'c'
    >>> t = .5

    >>> price = black_scholes(flag, S, K, t, r, sigma)
    >>> iv = implied_volatility(price, S, K, t, r, flag)

    >>> expected_price = 5.87602423383
    >>> expected_iv = 0.2
    
    >>> abs(expected_price - price) < 0.00001
    True
    >>> abs(expected_iv - iv) < 0.01
    True

    >>> sigma = 0.3
    >>> S, K, t, r, flag = 100.0, 1000.0, 0.5, 0.05, 'p'
    >>> price = black_scholes(flag, S, K, t, r, sigma)
    >>> print (price)
    875.309912028
    >>> iv = implied_volatility(price, S, K, t, r, flag)

    >>> print (round(iv, 1))
    0.0
    """

    f = lambda sigma: price - black_scholes(flag, S, K, t, r, sigma)

    return brentq(
        f,
        a=1e-12,
        b=100,
        xtol=1e-15,
        rtol=1e-15,
        maxiter=1000,
        full_output=False
    )


if __name__ == "__main__":
    from quant_greeks.helpers.doctest_helper import run_doctest
    run_doctest()
