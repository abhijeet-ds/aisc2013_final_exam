from flask import Flask, request, render_template
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
import base64
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

app = Flask(__name__)

def download_stock_data(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date)
    return data

def train_stock_model(data):
    # Ensure that there is data
    if data.empty:
        return None, None

    # Feature engineering: Using 'Open' price as feature and 'Close' price as target
    data['Date'] = data.index
    data['Date'] = pd.to_datetime(data['Date'])
    data['Date_ordinal'] = data['Date'].map(pd.Timestamp.toordinal)
    
    X = data[['Date_ordinal']]
    y = data['Close']
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train a simple linear regression model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Predict
    y_pred = model.predict(X_test)
    
    # Calculate Mean Squared Error
    mse = mean_squared_error(y_test, y_pred)
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.scatter(X_test, y_test, color='black', label='Data')
    plt.plot(X_test, y_pred, color='blue', linewidth=3, label='Prediction')
    plt.xlabel('Date')
    plt.ylabel('Close Price')
    plt.title('Stock Price Prediction')
    plt.legend()
    
    # Save plot to a PNG image in memory
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    
    plt.close()
    
    return model, mse, plot_url

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        ticker = request.form['ticker']
        start_date_str = request.form['start_date']
        end_date_str = request.form['end_date']
        
        # Convert strings to datetime objects
        start_date = pd.to_datetime(start_date_str, format='%Y-%m-%d')
        end_date = pd.to_datetime(end_date_str, format='%Y-%m-%d')
        
        # Convert datetime objects back to strings
        start_date_formatted = start_date.strftime('%Y-%m-%d')
        end_date_formatted = end_date.strftime('%Y-%m-%d')
        
        # Download stock data
        data = download_stock_data(ticker, start_date_formatted, end_date_formatted)
        
        # Train model and get MSE and plot URL
        model, mse, plot_url = train_stock_model(data)
        
        if model is None:
            return render_template('result.html', ticker=ticker, mse='No data available', plot_url=None)
        
        return render_template('result.html', ticker=ticker, mse=mse, plot_url=plot_url)
    
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
