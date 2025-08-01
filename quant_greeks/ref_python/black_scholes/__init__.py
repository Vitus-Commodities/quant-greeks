# -*- coding: utf-8 -*-

"""
quant_greeks.ref_python.black_scholes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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
from __future__ import division

# Related third party imports
import numpy
from scipy.stats import norm

# Local application/library specific imports


N = norm.cdf


# -----------------------------------------------------------------------------
# FUNCTIONS - REFERENCE PYTHON IMPLEMENTATION, FOR COMPARISON

def d1(S, K, t, r, sigma):  # see Hull, page 292
    """Calculate the d1 component of the Black-Scholes PDE.

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

    John C. Hull, "Options, Futures and Other Derivatives,"
    7th edition, Example 13.6, page 294

    >>> S = 42
    >>> K = 40
    >>> r = .10
    >>> sigma = .20
    >>> t = 0.5
    >>> calculated_d1 = d1(S,K,t,r,sigma)
    >>> text_book_d1 = 0.7693
    >>> abs(calculated_d1 - text_book_d1) < 0.0001
    True
    """

    sigma_squared = sigma * sigma
    numerator = numpy.log(S / float(K)) + (r + sigma_squared / 2.) * t
    denominator = sigma * numpy.sqrt(t)

    if not denominator:
        print ('')
    return numerator / denominator


def d2(S, K, t, r, sigma):  # see Hull, page 292
    """Calculate the d2 component of the Black-Scholes PDE.

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

    John C. Hull, "Options, Futures and Other Derivatives,"
    7th edition, Example 13.6, page 294

    >>> S = 42
    >>> K = 40
    >>> r = .10
    >>> sigma = .20
    >>> t = 0.5
    >>> calculated_d2 = d2(S,K,t,r,sigma) #0.627841271869
    >>> text_book_d2 = 0.6278
    >>> abs(calculated_d2 - text_book_d2) < 0.0001
    True
    """

    return d1(S, K, t, r, sigma) - sigma * numpy.sqrt(t)


def black_scholes(flag, S, K, t, r, sigma):
    """Return the Black-Scholes option price implemented in
        python (for reference).

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

    >>> S,K,t,r,sigma = 60,65,.25,.08,.3
    >>> expected = 2.13336844492
    >>> actual = black_scholes('c',S,K,t,r,sigma)
    >>> abs(expected-actual) < 1e-11
    True
    """

    e_to_the_minus_rt = numpy.exp(-r * t)
    D1 = d1(S, K, t, r, sigma)
    D2 = d2(S, K, t, r, sigma)
    if flag == 'c':
        return S * N(D1) - K * e_to_the_minus_rt * N(D2)
    else:
        return - S * N(-D1) + K * e_to_the_minus_rt * N(-D2)


if __name__ == "__main__":
    from quant_greeks.helpers.doctest_helper import run_doctest
    run_doctest()
