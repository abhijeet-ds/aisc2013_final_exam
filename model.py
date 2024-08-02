#!/usr/bin/env python
# coding: utf-8

# ### 1. Data Collection

# In[8]:


import yfinance as yf

def download_stock_data(ticker, start_date, end_date):
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    stock_data.to_csv(f'{ticker}.csv')
    return stock_data


# ### 2. Feature Engg

# In[9]:


def add_indicators(df):
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['RSI'] = compute_rsi(df['Close'], window=14)
    # Add more indicators like MACD, Bollinger Bands, etc.
    return df

def compute_rsi(series, window):
    delta = series.diff(1)
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=window, min_periods=1).mean()
    avg_loss = loss.rolling(window=window, min_periods=1).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


# ### 3. Model training

# In[10]:


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

def train_stock_model(df):
    df = df.dropna()
    X = df[['SMA_20', 'SMA_50', 'RSI']]  # Example features
    y = df['Close']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100)
    model.fit(X_train, y_train)

#     Model Prediction
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    return model, mse


# ### 4. Model Visualtization

# In[11]:


import matplotlib.pyplot as plt

def plot_predictions(df, y_pred):
    plt.figure(figsize=(10, 5))
    plt.plot(df.index, df['Close'], label='Actual')
    plt.plot(df.index[-len(y_pred):], y_pred, label='Predicted')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.show()


# In[ ]:




