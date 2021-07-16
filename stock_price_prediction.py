import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import chart_studio.plotly as py
import plotly.graph_objs as go
from plotly.offline import plot

#for offline plotting
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode(connected=True)

tesla = pd.read_csv('C:/Users/Admin/Documents/Machine Learning\Stock price prediction/datasetsandcodefilesstockmarketprediction/tesla.csv')
tesla.head()
tesla.info()

tesla['Date'] = pd.to_datetime(tesla['Date'])
print(f'Dataset containes stock prices between {tesla.Date.min()} {tesla.Date.max()} days.')
print(f'Total days = {(tesla.Date.max() - tesla.Date.min()).days}')

tesla.describe()

tesla[['Open', 'High', 'Low', 'Close', 'Adj Close']].plot(kind='box')

#layout of the plot
layout = go.Layout(
    title = 'Stock Prices of Tesla', 
    xaxis = dict(
        title = 'Date',
        titlefont = dict(
            family = 'Courier New, monospace',
            size = 18,
            color = '#7f7f7f'
        )
    ),
    yaxis = dict(
        title = 'Price',
        titlefont = dict(
            family = 'Courier New, monospace',
            size = 18,
            color = '#7f7f7f'
        )
    )
)

tesla_data = [{'x':tesla['Date'], 'y':tesla['Close']}]
plot = go.Figure(data = tesla_data, layout = layout)

#plot(plot) plotting offline
iplot(plot)

from sklearn.linear_model import LinearRegression
#building the regression model
from sklearn.model_selection import train_test_split
#for preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
#for model evaluation
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import r2_score

#split the data into train and test set
X = np.array(tesla.index).reshape(-1, 1)
y = tesla['Close']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)

#feature scaling
scaler = StandardScaler().fit(X_train)

#creating a linear model
lm = LinearRegression()
lm.fit(X_train, y_train)

print('Code sueceeded')