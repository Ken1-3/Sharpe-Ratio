# Sharpe-Ratio

##  The sharpe ratio is a key benchmark in assessing risk-adjusted portfolios
  This project asses the sharpe ratio of any given portfolio denoted by 'stocks' list through any chosen time period `start` and `end`. Using random weight assignments, we can iterate through thousands of possible weights assignments for each stock within the specificed portfolio, and formulize the sharpe ratio of each iteration. The results are then plotted to view the distribution of sharpe ratio perormance in the aggregate for identifying which basket of stocks have the highest liklihood of returning profitable, risk adjusted poerfolio. 
####  We can start by installing the needed modules
####  This code is running GUI on [The Quants Philosopher](https://www.thequantsphilosopher.com/)

######  (running on conda 4.10.1)
  Installing required packages

```
pip install sklearn == 0.24.1 
pip install statsmodels == 0.12.2
pip install pandas-datareader == 0.10.0
```
  Importing all neccesary packages

  And then Importing the required modules
 ```
import datetime
from math import *
import numpy as np
import matplotlib.pyplot as plt
import pandas_datareader as web
import statsmodels.api as sm
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KernelDensity``  
 ```
### Set Up Conditions
   Start by defining a function to easily adjust the start/end dates, and basket of stocks we will pick from
   `def sharpe_ratio(start_date, end_date, stock_list):`
   
   Proceed to get daily prices (at the close) with data_reader module - which leverages Yahoo finance price histories
   
```
panel_data = web.DataReader(stock_list, 'yahoo',start_date,end_date)
close = panel_data['Close']    
```

  Grab the percennt change with `portfolio_returns = close.pct_change()` and begin your for loop! **The more the better**
  
  
### Get Randomly Generated Portfolio Return Metrics

  Utilize the `np.random.rand()` function to generate random weights
  
  Distribute the random weights amongst the returns individually and then sum
  
  Formula can be summarized with
  
  Σ(weight_n * returns_n)
  
  *traditional sharpe ratio formulae would dictate we subtract the risk free rate, but step will be removed from this processing*

  ``` 
length = int(len(stock_list))
weights = np.random.rand(length)
portfolio_weights = weights/sum(weights)
weighted_p = portfolio_returns * portfolio_weights
indi_expected_returns = weighted_p.mean()
expected_return = indi_expected_returns.sum()      
  ```
  
  
### Get Randomly Generated Portfolio Risk Metrics

   Utilize the covariance functions and account for all trading days within each year, 252
   
   √ Σ(weight_n * annualized_variance_m)
   
   ```
cov_mat_annual = portfolio_returns.cov() * 252
portfolio_volatility = (np.dot(portfolio_weights.T, np.dot(cov_mat_annual, portfolio_weights)))
sharpe_ratio = ((expected_return)/(portfolio_volatility)) * sqrt(252)
   ```


###  Plotting and Visualization

  Plot a Histogram of the returned distribution of sharpe ratios from the randomly generated portfolios for context
  
  ```
plt.style.use('dark_background')
plt.title("Random Portfolio Sharpe Ratio Distribution")
plt.hist(sr,bins = 100)
plt.show()
```

  ![image](https://user-images.githubusercontent.com/89386946/144740605-02c8f088-3fab-4ad5-99b2-c7e2020ce451.png)

  Define the best fit line for Kernel Density Estimation, and plot KDE/Probability Distribution Function
  
  ```
w = np.array(sr).reshape(-1,1)
params = {'bandwidth': np.linspace(.01, 1, 100)}
grid = GridSearchCV(KernelDensity(), params)
grid.fit(w)
cross = grid.best_estimator_.bandwidth

kde = sm.nonparametric.KDEUnivariate(sr)
kde.fit(bw = cross)
plt.title('PDF')
plt.plot(kde.support, kde.density, label = 'Cross')
plt.legend()
plt.show()
  ```
  
  ![image](https://user-images.githubusercontent.com/89386946/144740636-0fe7940f-70a3-4bd7-a1c2-d25b40a31c2c.png)


  And a Cumulative Distribution Function
 
  ```
num_bins = 100
counts, egd = np.histogram (sr, bins=num_bins, normed=True)
cdf = np.cumsum (counts)

plt.title('CDF')
plt.plot (egd[1:], cdf/cdf[-1])
plt.show()
```
    
  
  ![image](https://user-images.githubusercontent.com/89386946/144740645-329c8ea0-24b6-4cca-b918-39fcd8b1672c.png)
  
  
  Rework to view as a Quantile Function
  
  ```
plt.title('Quantile')
plt.plot ( cdf/cdf[-1], egd[1:])
plt.show()
```

  ![image](https://user-images.githubusercontent.com/89386946/144740957-365ec65e-03e7-4130-a92d-e2b21151fcc6.png)


  Lastly, print the quartiles of the basket of stocks for a table view
  
  | Range | Sharpe Ratio |
  | --- | --- | 
 |1% Percentile| 0.11|
 |25% Percentile|0.30|
 |50% Percentile|0.38|
 |75% Percentile|0.46|
 |99% Percentile|0.64|
  
  


  
  
  
  
    


   
