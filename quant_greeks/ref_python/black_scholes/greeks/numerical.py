# -*- coding: utf-8 -*-

"""
quant_greeks.ref_python.black_scholes.greeks.numerical
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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

# Local application/library specific imports
from quant_greeks.ref_python.black_scholes import black_scholes
from quant_greeks.helpers.numerical_greeks import delta as numerical_delta
from quant_greeks.helpers.numerical_greeks import vega as numerical_vega
from quant_greeks.helpers.numerical_greeks import theta as numerical_theta
from quant_greeks.helpers.numerical_greeks import rho as numerical_rho
from quant_greeks.helpers.numerical_greeks import gamma as numerical_gamma
from quant_greeks.ref_python.black_scholes.greeks.analytical import gamma as agamma
from quant_greeks.ref_python.black_scholes.greeks.analytical import delta as adelta
from quant_greeks.ref_python.black_scholes.greeks.analytical import vega as avega
from quant_greeks.ref_python.black_scholes.greeks.analytical import rho as arho
from quant_greeks.ref_python.black_scholes.greeks.analytical import theta as atheta


# -----------------------------------------------------------------------------
# FUNCTIONS - NUMERICAL GREEK CALCULATION


f = lambda flag, S, K, t, r, sigma, b: black_scholes(flag, S, K, t, r, sigma)


def delta(flag, S, K, t, r, sigma):
    """Return Black-Scholes delta of an option.

    :param S: underlying asset price
    :type S: float
    :param K: strike price
    :type K: float
    :param sigma: annualized standard deviation, or volatility
    :type sigma: float
    :param t: time to expiration in years
    :type t: float
    :param r: risk-free interest rate
    :type r: float
    :param flag: 'c' or 'p' for call or put.
    :type flag: str
    """

    b = r

    return numerical_delta(flag, S, K, t, r, sigma, b, f)


def theta(flag, S, K, t, r, sigma):
    """Return Black-Scholes theta of an option.

    :param S: underlying asset price
    :type S: float
    :param K: strike price
    :type K: float
    :param sigma: annualized standard deviation, or volatility
    :type sigma: float
    :param t: time to expiration in years
    :type t: float
    :param r: risk-free interest rate
    :type r: float
    :param flag: 'c' or 'p' for call or put.
    :type flag: str
    """

    b = r

    return numerical_theta(flag, S, K, t, r, sigma, b, f)


def vega(flag, S, K, t, r, sigma):
    """Return Black-Scholes vega of an option.

    :param S: underlying asset price
    :type S: float
    :param K: strike price
    :type K: float
    :param sigma: annualized standard deviation, or volatility
    :type sigma: float
    :param t: time to expiration in years
    :type t: float
    :param r: risk-free interest rate
    :type r: float
    :param flag: 'c' or 'p' for call or put.
    :type flag: str
    """

    b = r

    return numerical_vega(flag, S, K, t, r, sigma, b, f)


def rho(flag, S, K, t, r, sigma):
    """Return Black-Scholes rho of an option.

    :param S: underlying asset price
    :type S: float
    :param K: strike price
    :type K: float
    :param sigma: annualized standard deviation, or volatility
    :type sigma: float
    :param t: time to expiration in years
    :type t: float
    :param r: risk-free interest rate
    :type r: float
    :param flag: 'c' or 'p' for call or put.
    :type flag: str
    """

    b = r

    return numerical_rho(flag, S, K, t, r, sigma, b, f)


def gamma(flag, S, K, t, r, sigma):
    """Return Black-Scholes gamma of an option.

    :param S: underlying asset price
    :type S: float
    :param K: strike price
    :type K: float
    :param sigma: annualized standard deviation, or volatility
    :type sigma: float
    :param t: time to expiration in years
    :type t: float
    :param r: risk-free interest rate
    :type r: float
    :param flag: 'c' or 'p' for call or put.
    :type flag: str
    """

    b = r

    return numerical_gamma(flag, S, K, t, r, sigma, b, f)


def test():
    """Test by comparing analytical and numerical values.

    >>> S =  49
    >>> K = 50
    >>> r = .05
    >>> t = 0.3846
    >>> sigma = 0.2
    >>> flag = 'c'

    >>> epsilon = .0001

    >>> v1 = delta(flag, S, K, t, r, sigma)
    >>> v2 = adelta(flag, S, K, t, r, sigma)
    >>> abs(v1-v2)<epsilon
    True

    >>> v1 = gamma(flag, S, K, t, r, sigma)
    >>> v2 = agamma(flag, S, K, t, r, sigma)
    >>> abs(v1-v2)<epsilon
    True

    >>> v1 = rho(flag, S, K, t, r, sigma)
    >>> v2 = arho(flag, S, K, t, r, sigma)
    >>> abs(v1-v2)<epsilon
    True

    >>> v1 = vega(flag, S, K, t, r, sigma)
    >>> v2 = avega(flag, S, K, t, r, sigma)
    >>> abs(v1-v2)<epsilon
    True

    >>> v1 = theta(flag, S, K, t, r, sigma)
    >>> v2 = atheta(flag, S, K, t, r, sigma)
    >>> abs(v1-v2)<epsilon
    True
    """

    pass


def hull_book_tests():
    """
    Example 17.1, page 355, Hull:

    >>> S = 49
    >>> K = 50
    >>> r = .05
    >>> t = 0.3846
    >>> sigma = 0.2
    >>> flag = 'c'
    >>> delta_calc = delta(flag, S, K, t, r, sigma)
    >>> # 0.521601633972
    >>> delta_text_book = 0.522
    >>> abs(delta_calc - delta_text_book) < .01
    True

    Example 17.2, page 359, Hull:

    >>> S = 49
    >>> K = 50
    >>> r = .05
    >>> t = 0.3846
    >>> sigma = 0.2
    >>> flag = 'c'
    >>> annual_theta_calc = theta(flag, S, K, t, r, sigma) * 365
    >>> # -4.30538996455
    >>> annual_theta_text_book = -4.31
    >>> abs(annual_theta_calc - annual_theta_text_book) < .01
    True

    Example 17.4, page 364, Hull:

    >>> S = 49
    >>> K = 50
    >>> r = .05
    >>> t = 0.3846
    >>> sigma = 0.2
    >>> flag = 'c'
    >>> gamma_calc = gamma(flag, S, K, t, r, sigma)
    >>> # 0.0655453772525
    >>> gamma_text_book = 0.066
    >>> abs(gamma_calc - gamma_text_book) < .001
    True

    Example 17.6, page 367, Hull:

    >>> S = 49
    >>> K = 50
    >>> r = .05
    >>> t = 0.3846
    >>> sigma = 0.2
    >>> flag = 'c'
    >>> vega_calc = vega(flag, S, K, t, r, sigma)
    >>> # 0.121052427542
    >>> vega_text_book = 0.121
    >>> abs(vega_calc - vega_text_book) < .01
    True

    Example 17.7, page 368, Hull:

    >>> S = 49
    >>> K = 50
    >>> r = .05
    >>> t = 0.3846
    >>> sigma = 0.2
    >>> flag = 'c'
    >>> rho_calc = rho(flag, S, K, t, r, sigma)
    >>> # 0.089065740988
    >>> rho_text_book = 0.0891
    >>> abs(rho_calc - rho_text_book) < .0001
    True
    """


if __name__ == "__main__":
    from quant_greeks.helpers.doctest_helper import run_doctest
    run_doctest()
