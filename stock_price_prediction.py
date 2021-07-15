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
print('Code sueceeded')