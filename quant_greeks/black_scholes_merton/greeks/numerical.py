# -*- coding: utf-8 -*-

"""
quant_greeks.black_scholes_merton.greeks.numerical
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A library for option pricing, implied volatility, and
greek calculation.  quant_greeks is based on lets_be_rational,
a Python wrapper for LetsBeRational by Peter Jaeckel as
described below.

:copyright: © 2017 Gammon Capital LLC
:license: MIT, see LICENSE for more details.

About LetsBeRational:
~~~~~~~~~~~~~~~~~~~~~

The source code of LetsBeRational resides at www.jaeckel.org/LetsBeRational.7z .

::

    ========================================================================================
    Copyright © 2013-2014 Peter Jäckel.

    Permission to use, copy, modify, and distribute this software is freely granted,
    provided that this notice is preserved.

    WARRANTY DISCLAIMER
    The Software is provided "as is" without warranty of any kind, either express or implied,
    including without limitation any implied warranties of condition, uninterrupted use,
    merchantability, fitness for a particular purpose, or non-infringement.
    ========================================================================================


"""


# -----------------------------------------------------------------------------
# IMPORTS

# Standard library imports

# Related third party imports

# Local application/library specific imports
from quant_greeks.black_scholes_merton import black_scholes_merton
from quant_greeks.helpers.numerical_greeks import delta as numerical_delta
from quant_greeks.helpers.numerical_greeks import vega as numerical_vega
from quant_greeks.helpers.numerical_greeks import theta as numerical_theta
from quant_greeks.helpers.numerical_greeks import rho as numerical_rho
from quant_greeks.helpers.numerical_greeks import gamma as numerical_gamma
from quant_greeks.black_scholes_merton.greeks.analytical import gamma as agamma
from quant_greeks.black_scholes_merton.greeks.analytical import delta as adelta
from quant_greeks.black_scholes_merton.greeks.analytical import vega as avega
from quant_greeks.black_scholes_merton.greeks.analytical import rho as arho
from quant_greeks.black_scholes_merton.greeks.analytical import theta as atheta


# -----------------------------------------------------------------------------
# FUNCTIONS - NUMERICAL GREEK CALCULATION

f = lambda flag, S, K, t, r, sigma, b: black_scholes_merton(flag, S, K, t, r, sigma, r-b)


def delta(flag, S, K, t, r, sigma, q):
    """Returns the Black-Scholes-Merton delta of an option.
    
    :param flag: 'c' or 'p' for call or put.
    :type flag: str
    :param S: underlying asset price
    :type S: float
    :param K: strike price
    :type K: float
    :param t: time to expiration in years
    :type t: float
    :param r: annual risk-free interest rate
    :type r: float
    :param sigma: volatility
    :type sigma: float
    :param q: annualized continuous dividend yield
    :type q: float
    
    :returns:  float 
    """   

    b = r-q

    return numerical_delta(flag, S, K, t, r, sigma, b, f)


def theta(flag, S, K, t, r, sigma, q):
    """Returns the Black-Scholes-Merton theta of an option.

    :param flag: 'c' or 'p' for call or put.
    :type flag: str
    :param S: underlying asset price
    :type S: float
    :param K: strike price
    :type K: float
    :param t: time to expiration in years
    :type t: float
    :param r: annual risk-free interest rate
    :type r: float
    :param sigma: volatility
    :type sigma: float
    :param q: annualized continuous dividend yield
    :type q: float

    :returns:  float 
    """   

    b = r-q

    return numerical_theta(flag, S, K, t, r, sigma, b, f)


def vega(flag, S, K, t, r, sigma, q):
    """Returns the Black-Scholes-Merton vega of an option.

    :param flag: 'c' or 'p' for call or put.
    :type flag: str
    :param S: underlying asset price
    :type S: float
    :param K: strike price
    :type K: float
    :param t: time to expiration in years
    :type t: float
    :param r: annual risk-free interest rate
    :type r: float
    :param sigma: volatility
    :type sigma: float
    :param q: annualized continuous dividend yield
    :type q: float

    :returns:  float 
    """

    b = r-q

    return numerical_vega(flag, S, K, t, r, sigma, b, f)


def rho(flag, S, K, t, r, sigma, q):
    """Returns the Black-Scholes-Merton rho of an option.

    :param flag: 'c' or 'p' for call or put.
    :type flag: str
    :param S: underlying asset price
    :type S: float
    :param K: strike price
    :type K: float
    :param t: time to expiration in years
    :type t: float
    :param r: annual risk-free interest rate
    :type r: float
    :param sigma: volatility
    :type sigma: float
    :param q: annualized continuous dividend yield
    :type q: float

    :returns:  float 
    """

    b = r-q

    return numerical_rho(flag, S, K, t, r, sigma, b, f)


def gamma(flag, S, K, t, r, sigma, q):
    """Returns the Black-Scholes-Merton gamma of an option.

    :param flag: 'c' or 'p' for call or put.
    :type flag: str
    :param S: underlying asset price
    :type S: float
    :param K: strike price
    :type K: float
    :param t: time to expiration in years
    :type t: float
    :param r: annual risk-free interest rate
    :type r: float
    :param sigma: volatility
    :type sigma: float
    :param q: annualized continuous dividend yield
    :type q: float

    :returns:  float 
    """

    b = r-q

    return numerical_gamma(flag, S, K, t, r, sigma, b, f)


def test_analytical_vs_numerical():
    """Test by comparing analytical and numerical values.

    >>> flag='c'
    >>> S=1000.0
    >>> K=1000.0
    >>> t=0.1
    >>> r=0.05
    >>> sigma=0.3
    >>> q = 0.05
    >>> sigma = 0.2
    >>> flag = 'c'

    >>> epsilon = 0.01

    >>> v1 = delta(flag, S, K, t, r, sigma, q)
    >>> v2 = adelta(flag, S, K, t, r, sigma, q)
    >>> abs(v1-v2)<epsilon
    True

    >>> v1 = gamma(flag, S, K, t, r, sigma, q)
    >>> v2 = agamma(flag, S, K, t, r, sigma, q)
    >>> abs(v1-v2)<epsilon
    True

    >>> v1 = rho(flag, S, K, t, r, sigma, q)
    >>> v2 = arho(flag, S, K, t, r, sigma, q)
    >>> abs(v1-v2)<epsilon
    True

    >>> v1 = vega(flag, S, K, t, r, sigma, q)
    >>> v2 = avega(flag, S, K, t, r, sigma, q)
    >>> abs(v1-v2)<epsilon
    True

    >>> v1 = theta(flag, S, K, t, r, sigma, q)
    >>> v2 = atheta(flag, S, K, t, r, sigma, q)
    >>> abs(v1-v2)<epsilon
    True

    Test PUT flag

    >>> flag = 'p'

    >>> v1 = delta(flag, S, K, t, r, sigma, q)
    >>> v2 = adelta(flag, S, K, t, r, sigma, q)
    >>> abs(v1-v2)<epsilon
    True

    >>> v1 = gamma(flag, S, K, t, r, sigma, q)
    >>> v2 = agamma(flag, S, K, t, r, sigma, q)
    >>> abs(v1-v2)<epsilon
    True

    >>> v1 = rho(flag, S, K, t, r, sigma, q)
    >>> v2 = arho(flag, S, K, t, r, sigma, q)
    >>> abs(v1-v2)<epsilon
    True

    >>> v1 = vega(flag, S, K, t, r, sigma, q)
    >>> v2 = avega(flag, S, K, t, r, sigma, q)
    >>> abs(v1-v2)<epsilon
    True

    >>> v1 = theta(flag, S, K, t, r, sigma, q)
    >>> v2 = atheta(flag, S, K, t, r, sigma, q)
    >>> abs(v1-v2)<epsilon
    True
    """
    pass


if __name__ == "__main__":
    from quant_greeks.helpers.doctest_helper import run_doctest
    run_doctest()
