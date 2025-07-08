# -*- coding: utf-8 -*-

"""
quant_greeks.black.greeks.analytical
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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
from __future__ import division

# Related third party imports
import numpy

# Local application/library specific imports
from py_lets_be_rational import norm_cdf as N
from quant_greeks.helpers import pdf
from quant_greeks.black import black
from quant_greeks.ref_python.black import d1, d2


# -----------------------------------------------------------------------------
# FUNCTIONS - ANALYTICAL GREEKS

def delta(flag, F, K, t, r, sigma):
    """Returns the Black delta of an option.

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

    :returns:  float
    
    >>> F = 49
    >>> K = 50 
    >>> r = .05
    >>> t = 0.3846
    >>> sigma = 0.2
    >>> flag = 'c'
    >>> v1 = delta(flag, F, K, t, r, sigma)
    >>> v2 = 0.45107017482201828
    >>> abs(v1-v2) < .000001
    True
    """

    D1 = d1(F, K, t, r, sigma)

    if flag == 'p':
        return - numpy.exp(-r*t) * N(-D1)
    else:
        return numpy.exp(-r*t) * N(D1)


def theta(flag, F, K, t, r, sigma):
    """Returns the Black theta of an option.

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

    :returns:  float 

    >>> F = 49
    >>> K = 50 
    >>> r = .05
    >>> t = 0.3846
    >>> sigma = 0.2
    >>> flag = 'c'
    >>> v1 = theta(flag, F, K, t, r, sigma)
    >>> v2 = -0.00816236877462
    >>> abs(v1-v2) < .000001
    True
    >>> flag = 'p'
    >>> v1 = theta(flag, F, K, t, r, sigma)
    >>> v2 = -0.00802799155312
    >>> abs(v1-v2) < .000001
    True
    """

    e_to_the_minus_rt = numpy.exp(-r*t)
    two_sqrt_t = 2 * numpy.sqrt(t)

    D1 = d1(F, K, t, r, sigma)
    D2 = d2(F, K, t, r, sigma)

    first_term = F * e_to_the_minus_rt * pdf(D1) * sigma / two_sqrt_t 

    if flag == 'c':        
        second_term = -r * F * e_to_the_minus_rt * N(D1)
        third_term = r * K * e_to_the_minus_rt * N(D2)
        return -(first_term + second_term + third_term) / 365.0
    else:
        second_term = -r * F * e_to_the_minus_rt * N(-D1)
        third_term = r * K * e_to_the_minus_rt * N(-D2)
        return (-first_term + second_term + third_term) / 365.0


def gamma(flag, F, K, t, r, sigma):
    """Returns the Black gamma of an option.

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

    :returns:  float 

    >>> F = 49
    >>> K = 50 
    >>> r = .05
    >>> t = 0.3846
    >>> sigma = 0.2
    >>> flag = 'c'
    >>> v1 = gamma(flag, F, K, t, r, sigma)
    >>> # 0.0640646705882
    >>> v2 = 0.0640646705882
    >>> abs(v1-v2) < .000001
    True
    """

    D1 = d1(F, K, t, r, sigma)
    return pdf(D1)*numpy.exp(-r*t)/(F*sigma*numpy.sqrt(t))


def vega(flag, F, K, t, r, sigma):
    """Returns the Black vega of an option.

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

    :returns:  float 
    
    >>> F = 49
    >>> K = 50 
    >>> r = .05
    >>> t = 0.3846
    >>> sigma = 0.2
    >>> flag = 'c'
    >>> v1 = vega(flag, F, K, t, r, sigma)
    >>> # 0.118317785624
    >>> v2 = 0.118317785624
    >>> abs(v1-v2) < .000001
    True
    """

    D1 = d1(F, K, t, r, sigma)
    return F * numpy.exp(-r*t) * pdf(D1) * numpy.sqrt(t)


def rho(flag, F, K, t, r, sigma):
    """Returns the Black rho of an option.

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

    :returns:  float 
      
    >>> F = 49
    >>> K = 50 
    >>> r = .05
    >>> t = 0.3846
    >>> sigma = 0.2
    >>> flag = 'c'
    >>> v1 = rho(flag, F, K, t, r, sigma)
    >>> v2 = -0.0074705380059582258
    >>> abs(v1-v2) < .000001
    True
    >>> flag = 'p'
    >>> v1 = rho(flag, F, K, t, r, sigma)
    >>> v2 = -0.011243286001308292
    >>> abs(v1-v2) < .000001
    True
    """

    return -t * black(flag, F, K, t, r, sigma)


def vanna(flag, F, K, t, r, sigma):
    """Returns the Black vanna of an option.

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

    return -numpy.exp(-r*t) * pdf(D1) * D2 / sigma


def volga(flag, F, K, t, r, sigma):
    """Returns the Black volga (vomma) of an option.

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
    v = vega(flag, F, K, t, r, sigma)

    return v * D1 * D2 / sigma


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

    if flag == 'c':
        return numpy.exp(-r*t) * (r * N(D1) + pdf(D1) * D2 / (2*t))
    else:
        return numpy.exp(-r*t) * (-r * N(-D1) + pdf(-D1) * D2 / (2*t))


def annualized_vega(flag, F, K, t, r, sigma):
    """Returns the Black annualized vega of an option.

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
    >>> v1 = annualized_vega(flag, F, K, t, r, sigma)
    >>> v2 = 0.3809893
    >>> abs(v1-v2) < .000001
    True
    """

    return vega(flag, F, K, t, r, sigma) / numpy.sqrt(t)


if __name__ == "__main__":
    from quant_greeks.helpers.doctest_helper import run_doctest
    run_doctest()
