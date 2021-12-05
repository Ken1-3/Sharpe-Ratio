# -*- coding: utf-8 -*-
"""
Created on Sat Dec  4 23:53:04 2021

@author: Kenneth Traina
"""
import datetime
from math import *
import numpy as np
import matplotlib.pyplot as plt
import pandas_datareader as web
import statsmodels.api as sm
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KernelDensity


stocks = ['TM', 'TSLA', 'GM', 'RACE', 'TTM', 'F', 'HMC', 'VWAGY']

#start and end dates
start = datetime.datetime(2019,1,29)
end = datetime.datetime(2020,1,30)


def sharpe_ratio(start_date, end_date, stock_list):

    #use yahoo finance api for historical data
    panel_data = web.DataReader(stock_list, 'yahoo',start_date,end_date)
    close = panel_data['Close']
    
    #percent change over time
    portfolio_returns = close.pct_change()
    
    #empty list
    sr = []
    
    #Now we can generate the portfolio distribution of weights, and identify the sharpe ratios for each portfolio.
    for sharpe_ratio in range(1000):
        
        #get performance metrics
        weights = np.random.rand(8)
        portfolio_weights = weights/sum(weights)
        weighted_p = portfolio_returns * portfolio_weights
        indi_expected_returns = weighted_p.mean()
        expected_return = indi_expected_returns.sum()
        
        #get variance metrics
        cov_mat_annual = portfolio_returns.cov() * 252
        portfolio_volatility = (np.dot(portfolio_weights.T, np.dot(cov_mat_annual, portfolio_weights)))
        sharpe_ratio = ((expected_return)/(portfolio_volatility)) * sqrt(252)
        
        #add all to list
        sr.append(sharpe_ratio)
    
    #define best fit line
    w = np.array(sr).reshape(-1,1)
    params = {'bandwidth': np.linspace(.01, 1, 100)}
    grid = GridSearchCV(KernelDensity(), params)
    grid.fit(w)
    cross = grid.best_estimator_.bandwidth
    
    #raw histogram
    plt.style.use('dark_background')
    plt.title("Random Portfolio Sharpe Ratio Distribution")
    plt.hist(sr,bins = 100)
    plt.show()
    
    #probability density function KDE
    kde = sm.nonparametric.KDEUnivariate(sr)
    kde.fit(bw = cross)
    plt.title('PDF')
    plt.plot(kde.support, kde.density, label = 'Cross')
    plt.legend()
    plt.show()
    print('\n', 'Cross Validation Bandwidth:',cross)
    
    num_bins = 100
    counts, egd = np.histogram (sr, bins=num_bins, normed=True)
    cdf = np.cumsum (counts)
    
    #plot cumulative distribution function
    plt.title('CDF')
    plt.plot (egd[1:], cdf/cdf[-1])
    plt.show()
    
    #plot quantile graph
    plt.title('Quantile')
    plt.plot ( cdf/cdf[-1], egd[1:])
    plt.show()
    
    format_sr =  [round(x,2) for x in sr]
    
    frst = np.percentile(format_sr,1)
    print('\n', '1% Percentile:',frst, 'Sharpe Ratio')
          
    twfv = np.percentile(format_sr,25)
    print('\n', '25% Percentile:',twfv, 'Sharpe Ratio')
    
    fift = np.percentile(format_sr,50)
    print('\n', '50% Percentile:',fift, 'Sharpe Ratio')
    
    svfv = round(np.percentile(format_sr,75),2)
    print('\n', '75% Percentile:',svfv, 'Sharpe Ratio')
    
    nnty = np.percentile(format_sr,99)
    print('\n', '99% Percentile:',nnty, 'Sharpe Ratio')


#run function    
sharpe_ratio(start,end,stocks)
