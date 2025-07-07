# -*- coding: utf-8 -*-

"""
py_vollib.black.greeks.additional
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Additional Greeks calculations for the Black model.
This module provides implementations of second and third order Greeks.

:copyright: © 2024 Gammon Capital LLC
:license: MIT, see LICENSE for more details.
"""

# -----------------------------------------------------------------------------
# IMPORTS

# Standard library imports
from __future__ import division

# Related third party imports
import numpy

# Local application/library specific imports
from py_lets_be_rational import norm_cdf as N
from py_vollib.helpers import pdf
from py_vollib.black import black
from py_vollib.ref_python.black import d1, d2


# -----------------------------------------------------------------------------
# FUNCTIONS - ADDITIONAL GREEKS

def vanna(flag, F, K, t, r, sigma):
    """Returns the Black vanna of an option.
    Vanna measures the change in delta with respect to change in volatility.
    The value is scaled to represent the change in delta for a 1% change in volatility
    and a 1% change in the forward price.

    :param flag: 'c' or 'p' for call or put.
    :type flag: str
    :param F: underlying futures price
    :type F: float
    :param K: strike price
    :type K: float
    :param t: time to expiration in years
    :type t: float
    :param r: annual risk-free interest rate
    :type r: float
    :param sigma: volatility
    :type sigma: float

    :returns: float

    >>> F = 49
    >>> K = 50 
    >>> r = .05
    >>> t = 0.3846
    >>> sigma = 0.2
    >>> flag = 'c'
    >>> v1 = vanna(flag, F, K, t, r, sigma)
    >>> v2 = -0.0241466
    >>> abs(v1-v2) < .000001
    True
    """
    D1 = d1(F, K, t, r, sigma)
    D2 = d2(F, K, t, r, sigma)
    # Scale by 0.01 for vol and 0.01 for forward move
    return -numpy.exp(-r*t) * pdf(D1) * D2 / sigma * 0.01 * F * 0.01


def volga(flag, F, K, t, r, sigma):
    """Returns the Black volga (vomma) of an option.
    Volga measures the second derivative of the option value with respect to volatility.

    :param flag: 'c' or 'p' for call or put.
    :type flag: str
    :param F: underlying futures price
    :type F: float
    :param K: strike price
    :type K: float
    :param t: time to expiration in years
    :type t: float
    :param r: annual risk-free interest rate
    :type r: float
    :param sigma: volatility
    :type sigma: float

    :returns: float

    >>> F = 49
    >>> K = 50 
    >>> r = .05
    >>> t = 0.3846
    >>> sigma = 0.2
    >>> flag = 'c'
    >>> v1 = volga(flag, F, K, t, r, sigma)
    >>> v2 = 0.0059159
    >>> abs(v1-v2) < .000001
    True
    """
    D1 = d1(F, K, t, r, sigma)
    D2 = d2(F, K, t, r, sigma)
    vega_value = F * numpy.exp(-r*t) * pdf(D1) * numpy.sqrt(t)
    return vega_value * D1 * D2 / sigma * 0.0001  # Scale by 0.0001 for 1% vol move squared


def charm(flag, F, K, t, r, sigma):
    """Returns the Black charm of an option.
    Charm measures the instantaneous rate of change of delta over time.
    The value is scaled to represent the daily change in delta and includes
    the forward price scaling as per market convention.

    :param flag: 'c' or 'p' for call or put.
    :type flag: str
    :param F: underlying futures price
    :type F: float
    :param K: strike price
    :type K: float
    :param t: time to expiration in years
    :type t: float
    :param r: annual risk-free interest rate
    :type r: float
    :param sigma: volatility
    :type sigma: float

    :returns: float

    >>> F = 49
    >>> K = 50 
    >>> r = .05
    >>> t = 0.3846
    >>> sigma = 0.2
    >>> flag = 'c'
    >>> v1 = charm(flag, F, K, t, r, sigma)
    >>> v2 = -0.0002147
    >>> abs(v1-v2) < .000001
    True
    """
    D1 = d1(F, K, t, r, sigma)
    D2 = d2(F, K, t, r, sigma)
    
    base_charm = numpy.exp(-r*t) * (
        pdf(D1) * (2*r*t - D2*sigma*numpy.sqrt(t))/(2*t*sigma*numpy.sqrt(t))
    )
    if flag == 'c':
        return base_charm * F / 365.0
    else:
        return -base_charm * F / 365.0


def root_vega(flag, F, K, t, r, sigma):
    """Returns the Black root-vega (annualized vega) of an option.
    Root-vega measures the change in option price with respect to volatility,
    normalized by the square root of time to expiry.

    :param flag: 'c' or 'p' for call or put.
    :type flag: str
    :param F: underlying futures price
    :type F: float
    :param K: strike price
    :type K: float
    :param t: time to expiration in years
    :type t: float
    :param r: annual risk-free interest rate
    :type r: float
    :param sigma: volatility
    :type sigma: float

    :returns: float

    >>> F = 49
    >>> K = 50 
    >>> r = .05
    >>> t = 0.3846
    >>> sigma = 0.2
    >>> flag = 'c'
    >>> v1 = root_vega(flag, F, K, t, r, sigma)
    >>> v2 = 0.3809893
    >>> abs(v1-v2) < .000001
    True
    """
    D1 = d1(F, K, t, r, sigma)
    # Regular vega divided by sqrt(t) to get annualized vega
    return F * numpy.exp(-r*t) * pdf(D1) * sigma / numpy.sqrt(t)


if __name__ == "__main__":
    from py_vollib.helpers.doctest_helper import run_doctest
    run_doctest()
