#!/usr/bin/env python
# encoding: utf-8


from setuptools import setup, find_packages


setup(
    name='quant_greeks',
    version='1.0.4',
    description='A library for calculating option Greeks and implied volatility using Black-Scholes-Merton model',
    url='https://github.com/Vitus-Commodities/quant-greeks',  # You should update this with your actual GitHub URL
    maintainer='dichanmb',  # Update this with your name
    maintainer_email='mbayboga@gmail.com',  # Update this with your email
    license='MIT',
    install_requires=[
        'py_lets_be_rational',
        'simplejson',
        'numpy',
        'pandas',
        'scipy'
    ],
    packages=find_packages()
)
