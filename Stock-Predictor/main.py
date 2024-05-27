""" from cProfile import label
from fileinput import close
from operator import rshift """
from multiprocessing import shared_memory
from flask import Flask, render_template, request, redirect, url_for, session, jsonify
from flask_mysqldb import MySQL
import MySQLdb.cursors
import re
from passlib.hash import bcrypt
import requests
import pandas as pd
from pandas import DataFrame
import datetime as dt
from datetime import date
import json
import statistics
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.layers import Dense, LSTM
from keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler
import math

app = Flask(__name__)

# Change this to your secret key (can be anything, it's for extra protection)
app.secret_key = 'your secret key'

# Enter your database connection details below
#conn=MySQLdb.connect(host="localhost", user="root", passwd="<password_here>", db="pythonlogin")
#cursor2=conn.cursor()

app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = '<password_here>'
app.config['MYSQL_DB'] = 'pythonlogin'

# Intialize MySQL
mysql = MySQL(app)

# http://localhost:5000/ - the following will be our login page, which will use both GET and POST requests
@app.route('/', methods=['GET', 'POST'])
def login():
    # Output message if something goes wrong...
    msg = ''
    # Check if "username" and "password" POST requests exist (user submitted form)
    if request.method == 'POST' and 'username' in request.form and 'password' in request.form:
        # Create variables for easy access
        username = request.form['username']
        password = request.form['password']
        # Check if account exists using MySQL
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        cursor.execute('SELECT * FROM accounts WHERE username = %s', (username,))
        # Fetch one record and return result
        account = cursor.fetchone()
        if not account:
            msg = 'User doesn\'t exist'
            return render_template('index.html', msg=msg)
        hashed_password = account['password']
        # If account exists in accounts table in out database
        hasher = bcrypt.using(rounds=13)
        if account and hasher.verify(password,hashed_password):
            # Create session data, we can access this data in other routes
            session['loggedin'] = True
            session['id'] = account['id']
            session['username'] = account['username']
            # Redirect to home page
            return redirect(url_for('home'))
        else:
            # Account doesnt exist or username/password incorrect
            msg = 'Incorrect username/password!'
    # Show the login form with message (if any)
    return render_template('index.html', msg=msg)

# http://localhost:5000/logout/ - this will be the logout page
@app.route('/logout/')
def logout():
    # Remove session data, this will log the user out
   session.pop('loggedin', None)
   session.pop('id', None)
   session.pop('username', None)
   # Redirect to login page
   return redirect(url_for('login'))

# http://localhost:5000/register/ - this will be the registration page, we need to use both GET and POST requests
@app.route('/register/', methods=['GET', 'POST'])
def register():
    # Output message if something goes wrong...
    msg = ''
    # Check if "username", "password" and "email" POST requests exist (user submitted form)
    if request.method == 'POST' and 'username' in request.form and 'password' in request.form and 'email' in request.form:
        # Create variables for easy access
        username = request.form['username']
        password = request.form['password']
        email = request.form['email']
        # Check if account exists using MySQL
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        cursor.execute('SELECT * FROM accounts WHERE username = %s', (username,))
        account = cursor.fetchone()
        # If account exists show error and validation checks
        if account:
            msg = 'Account already exists!'
        elif not re.match(r'[^@]+@[^@]+\.[^@]+', email):
            msg = 'Invalid email address!'
        elif not re.match(r'[A-Za-z0-9]+', username):
            msg = 'Username must contain only characters and numbers!'
        elif not username or not password or not email:
            msg = 'Please fill out the form!'
        else:
            # Account doesnt exists and the form data is valid, now insert new account into accounts table
            hasher = bcrypt.using(rounds=13)
            hashed_password = hasher.hash(password)
            cursor.execute('INSERT INTO accounts VALUES (NULL, %s, %s, %s)', (username, hashed_password, email,))
            mysql.connection.commit()
            msg = 'You have successfully registered!'
    elif request.method == 'POST':
        # Form is empty... (no POST data)
        msg = 'Please fill out the form!'
    # Show registration form with message (if any)
    return render_template('register.html', msg=msg)

# http://localhost:5000/home/ - this will be the home page, only accessible for loggedin users
@app.route('/home/')
def home():
    # Check if user is loggedin
    if 'loggedin' in session:
        # get current investments
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        cursor.execute('SELECT company as Company, shares_bought as `Shares Bought`, money_invested as `Money Invested`, cost_per_share as `Cost Per Share`FROM investments WHERE user_id = %s', (session['id'],))
        investments = DataFrame(cursor.fetchall())
        investments['Latest Close'] = ""
        investments['Profit'] = ""
        if investments.empty:
            return render_template('home.html', username=session['username'], msg="No Investments")
        # go through each investment, and display as table to user
        for i in investments.index:
            url = 'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol='+investments['Company'][i]+'&outputsize=compact&apikey=SHUKOMJN4MF9V6OE'
            r = requests.get(url)
            data = r.json()
            if 'Time Series (Daily)' not in data:
                return render_template('home.html', username=session['username'], msg="No Investments")
            data = data['Time Series (Daily)']
            first_pair = next(iter((data.items())))
            close_val = float(first_pair[1]['4. close'])
            profit = (float(investments['Shares Bought'][i]) * close_val) - float(investments['Money Invested'][i])
            investments['Latest Close'][i] = f"{close_val:.2f}"
            investments['Profit'][i] = f"{profit:.2f}"
        dict_data = [investments.to_dict(), investments.to_dict('index')]
        htmldf = '<table id=\"invests\"><tbody id=\"bod\"><tr>'
        for key in dict_data[0].keys():
            htmldf = htmldf + '<th class="header">' + key + '</th>'
        htmldf = htmldf + '</tr>'
        
        for key in dict_data[1].keys():
            htmldf = htmldf + '<tr>'
            for subkey in dict_data[1][key]:
                htmldf = htmldf + '<td class=\"comps\">' + str(dict_data[1][key][subkey]) + '</td>'
            htmldf = htmldf + '</tr>'

        htmldf = htmldf + '</tr></tbody></table>'
        # User is loggedin show them the home page
        return render_template('home.html', username=session['username'], data=[htmldf])
    # User is not loggedin redirect to login page
    return redirect(url_for('login'))

# http://localhost:5000/home/profile/ - this will be the profile page, only accessible for loggedin users
@app.route('/home/profile/')
def profile():
    # Check if user is loggedin
    if 'loggedin' in session:
        # We need all the account info for the user so we can display it on the profile page
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        cursor.execute('SELECT * FROM accounts WHERE id = %s', (session['id'],))
        account = cursor.fetchone()
        # Show the profile page with account info
        return render_template('profile.html', account=account)
    # User is not loggedin redirect to login page
    return redirect(url_for('login'))

# http://localhost:5000/home/profile/change_password/ - page for user to change password
@app.route('/home/profile/change_password/', methods=['GET', 'POST'])
def change_password():
    msg = ''
    # check the entire form is filled in before changing the password
    if request.method == 'POST' and 'password1' in request.form and 'password2' in request.form and 'password3' in request.form:
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        cursor.execute('SELECT password FROM accounts WHERE id = %s', (session['id'],))
        password = cursor.fetchone()
        given_password = request.form['password1']
        hasher = bcrypt.using(rounds=13)
        # check given password matches old password, and new passwords match, if so, replace it in the db
        if hasher.verify(given_password,password['password']) and request.form['password2'] == request.form['password3']:
            hashed_password = hasher.hash(request.form['password2'])
            cursor.execute('UPDATE accounts SET password = %s WHERE id = %s', (hashed_password, session['id'],))
            mysql.connection.commit()
            return redirect(url_for('home'))
        elif not hasher.verify(given_password,password['password']):
            msg = 'Original password incorrect'
            return render_template('change_password.html',msg=msg)
        elif not request.form['password2'] == request.form['password3']:
            msg = 'New password doesn\'t match confirmed password'
            return render_template('change_password.html',msg=msg)
    return render_template('change_password.html',msg=msg)

# http://localhost:5000/home/stock_prices - page to show stock prices
@app.route('/home/stock_prices/', methods=['GET', 'POST'])
def stock_prices():
    url = ''
    company = {'company':'IBM'}
    # form URL to get stock data using api
    if request.method == 'POST' and 'company' in request.form:
        company['company'] = request.form['company']
        company['company'] = company['company'].upper()
        #url = 'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol='+company['company']+'&outputsize=compact&apikey=SHUKOMJN4MF9V6OE'
        df = get_stock_data(company['company'])
    else:
        #url = 'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol='+company['company']+'&outputsize=compact&apikey=SHUKOMJN4MF9V6OE'
        df = get_stock_data(company['company'])
    """ r = requests.get(url)
    data = r.json()
    # if no company exists, inform the user
    if 'Error Message' in data or 'Note' in data:
        company = {'company':'Invalid Company Chosen'}
        return render_template('stock_prices.html', labels=[],close_value=[],company=company)
    data = data['Time Series (Daily)']
    df = pd.DataFrame(columns=['Date','Low','High','Close','Open'])
    # put json data into dataframe
    for key,val in data.items():
        date = dt.datetime.strptime(key, '%Y-%m-%d')
        data_row = [date.date(),float(val['3. low']),float(val['2. high']),
                    float(val['4. close']),float(val['1. open'])]
        df.loc[-1,:] = data_row
        df.index = df.index + 1 """
    if df is None:
        return render_template('stock_prices.html', labels=[],close_value=[],company={'company':'Invalid Company Chosen'})



    labels = df['Date'].tolist()
    labels = [date_obj.strftime('%Y-%m-%d') for date_obj in labels]
    # labels.reverse()
    close_value = df['Close'].tolist()
    # close_value.reverse()
    df['Daily_Return'] = df['Close'].pct_change()

    #df = df.iloc[::-1]
    #df = df.dropna()

    sd_percentage = calculate_sd(close_value)
    daily_return_no_nan = df['Daily_Return'].tolist()
    daily_return_no_nan = [x for x in daily_return_no_nan if str(x) != 'nan']
    daily_return_no_nan.reverse()
    sharpe_ratio = calculate_sharpe_ratio(daily_return_no_nan)
    sharpe_ratio = float(sharpe_ratio) * -1
    
    x = list(range(1,len(df)+1))
    float_lst = []
    for item in x:
        float_lst.append(float(item))
    y = df['Close']
    float_lst1 = []
    for item in y:
        float_lst1.append(float(item))
    d = np.polyfit(float_lst, float_lst1, 1)
    f = np.poly1d(d)
    df.insert(6,'Regression Val',f(float_lst))
    for index, row in df.iterrows():
        df.at[index,'Regression Val'] = f"{row['Regression Val']:.2f}"
    
    r_squared = calculate_r_squared(df)

    # get data from dataframe, and send to html page to render as graph
    return render_template('stock_prices.html', labels=labels, close_value=close_value,company=company, sd=sd_percentage, sharpe=sharpe_ratio, r_sqr = r_squared)

# method that gets suggested matches when user is typing
@app.route('/get_matches/<string:company>', methods=['GET','POST'])
def get_matches(company):
  # initialise data for best matches, and form url to query api
  matches = {'bestMatches':[]}
  company = json.loads(company)
  comp = company
  best_matches = 'https://www.alphavantage.co/query?function=SYMBOL_SEARCH&keywords='+comp+'&apikey=SHUKOMJN4MF9V6OE'
  r = requests.get(best_matches)
  matches = r.json()
  # ensure there exists at least one match, if so, return as json to be displayed to user, otherwise return empty dictionary
  if 'Note' not in matches and matches['bestMatches'] != []:
    matches = matches['bestMatches']
    df = pd.DataFrame(matches)
    df = df.drop(columns=['3. type', '4. region', '5. marketOpen', '6. marketClose', '7. timezone', '8. currency', '9. matchScore'])
    symbols = df['1. symbol'].tolist()
    names = df['2. name'].tolist()
    to_send = dict(zip(symbols,names))
    return jsonify(to_send)
  return jsonify({})

# http://localhost:5000/home/update_investments - method that allows users to update their investments
@app.route('/home/update_investments/', methods=['GET','POST'])
def update_investments(): 
    share_count = 0.0
    money_invested = 0.0
    # get data from form, and add it to database 
    if request.method == 'POST':
        company = request.form['company']
        cost_per_share = float(request.form['cost_per_share'])
        if request.form['share_count'] != '' and request.form['money_invested'] == '':
            share_count = float(request.form['share_count'])
            money_invested = f"{(share_count * cost_per_share):.2f}"
        elif request.form['share_count'] == '' and request.form['money_invested'] != '':
            money_invested = float(request.form['money_invested'])
            share_count = f"{(money_invested / cost_per_share):.2f}"
        else:
            share_count = float(request.form['share_count'])
            money_invested = float(request.form['money_invested'])
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        cursor.execute('INSERT INTO investments VALUES (NULL, %s, %s, %s, %s, %s)', (session['id'], company, share_count, money_invested, cost_per_share,))
        mysql.connection.commit()
    return render_template('update_investments.html')

# http://localhost:5000/home/stock_predictions - method that returns the dataframe including predicted prices
@app.route('/home/stock_predictions/', methods=['GET','POST'])
def stock_predictions():
    # initialise variables
    company = {'company':''}
    prior_dates = []
    predicted_dates = []
    actual_prices = []
    predicted_prices_to_send = []
    # when form submitted, get data
    if request.method == 'POST' and 'company' in request.form:
        company['company'] = request.form['company']
        company['company'] = company['company'].upper()
        df = get_stock_data(company['company'])
        # if invalid company, render page without graph
        if df is None:
            return render_template('stock_predictions.html', company={'company':''})
        # get predicted price
        predicted_prices = predict_prices(df)
        # convert predicted prices to 2dp
        for index, row in predicted_prices.iterrows():
            if not math.isnan(row['Forecast']):
                predicted_prices.at[index,'Forecast'] = f"{row['Forecast']:.2f}"
        # store historical data, and future data separately
        for index, row in predicted_prices.iterrows():
            if not math.isnan(row['Actual']):
                prior_dates.append(index.date())
                actual_prices.append(row['Actual'])
            if not math.isnan(row['Forecast']):
                predicted_dates.append(index.date())
                predicted_prices_to_send.append(row['Forecast'])
        # convert dates to string, and combine future dates with historical dates
        prior_dates = [date_obj.strftime('%Y-%m-%d') for date_obj in prior_dates]
        predicted_dates = [date_obj.strftime('%Y-%m-%d') for date_obj in predicted_dates]
        prior_dates.extend(predicted_dates)
        return render_template('stock_predictions.html', company=company, labels=prior_dates, actual_prices=actual_prices, predicted_prices=predicted_prices_to_send)
    else:
        return render_template('stock_predictions.html', company=company)

def calculate_sd(close_vals):
    sd = statistics.stdev(close_vals)
    mean = statistics.mean(close_vals)
    perc = f"{((sd/mean)*100):.2f}"
    return perc

# src = https://medium.datadriveninvestor.com/the-sharpe-ratio-with-python-from-scratch-fbb1d5e490b9
def calculate_sharpe_ratio(data, risk_free_rate=0):
    mean_daily_return = sum(data) / len(data)
    sd = statistics.stdev(data)
    daily_sharpe_ratio = (mean_daily_return - risk_free_rate) / sd
    sharpe_ratio = 252**(1/2) * daily_sharpe_ratio
    return f"{sharpe_ratio:.2f}"

# src = https://www.investopedia.com/terms/r/r-squared.asp
def calculate_r_squared(df):
    close = df['Close']
    reg_line_vals = df['Regression Val']
    unexplained_var = 0
    total_var = 0
    mean = statistics.mean(close)
    for i in range(len(close)):
        unexplained_var += (reg_line_vals[i] - close[i])**2
        total_var += (close[i] - mean)**2
    #print(unexplained_var)
    #print(total_var)
    # return true r squared value
    #f"{close_val:.2f}"
    r_squared = f"{(1-(unexplained_var/total_var)):.2f}"
    return r_squared

# get the stock data using alphavantage API
def get_stock_data(company):
    # query api and convert stock data to json
    url = 'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol='+company+'&outputsize=compact&apikey=SHUKOMJN4MF9V6OE'
    r = requests.get(url)
    data = r.json()
    # if valid company, convert json data to dataframe, and return the df
    if 'Time Series (Daily)' in data:
        data = data['Time Series (Daily)']
        df = pd.DataFrame(columns=['Date','Low','High','Close','Open'])
        for key,val in data.items():
            date = dt.datetime.strptime(key, '%Y-%m-%d')
            data_row = [date.date(),float(val['3. low']),float(val['2. high']),
                        float(val['4. close']),float(val['1. open'])]
            df.loc[-1,:] = data_row
            df.index = df.index + 1
        df = df.iloc[::-1]
        return df
    else:
        return None

def predict_prices(df):
    y = df['Close'].fillna(method='ffill')
    y = y.values.reshape(-1, 1)

    # scale the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler = scaler.fit(y)
    y = scaler.transform(y)

    # generate the input and output sequences
    n_lookback = 60  # length of input sequences (lookback period)
    n_forecast = 30  # length of output sequences (forecast period)

    X = []
    Y = []

    for i in range(n_lookback, len(y) - n_forecast + 1):
        X.append(y[i - n_lookback: i])
        Y.append(y[i: i + n_forecast])


    X = np.array(X)
    Y = np.array(Y)

    # fit the model
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(n_lookback, 1)))
    model.add(LSTM(units=50))
    model.add(Dense(n_forecast))

    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(X, Y, epochs=100, batch_size=32, verbose=0)

    # generate the forecasts
    X_ = y[- n_lookback:]  # last available input sequence
    X_ = X_.reshape(1, n_lookback, 1)

    Y_ = model.predict(X_).reshape(-1, 1)
    Y_ = scaler.inverse_transform(Y_)

    # organize the results in a data frame

    df_past = df[['Close']].reset_index()
    df_past.rename(columns={'index': 'Date', 'Close': 'Actual'}, inplace=True)
    df_past['Date'] = pd.to_datetime(df['Date'])
    df_past['Forecast'] = np.nan
    df_past['Forecast'].iloc[-1] = df_past['Actual'].iloc[-1]

    # may have to convert timestamps to dates - skips weekends
    future_dates = []
    future_dates.append(df_past['Date'].iloc[-1] + pd.Timedelta(days=1))
    for i in range(n_forecast - 1):
        next_date = future_dates[-1] + pd.Timedelta(days=1)
        if next_date.weekday() > 4:
            next_date = future_dates[-1] + pd.Timedelta(days=3)
        future_dates.append(next_date)
        
    df_future = pd.DataFrame(columns=['Date', 'Actual', 'Forecast'])
    df_future['Date'] = future_dates
    #df_future['Date'] = pd.date_range(start=df_past['Date'].iloc[-1] + pd.Timedelta(days=1), periods=n_forecast)
    df_future['Forecast'] = Y_.flatten()
    df_future['Actual'] = np.nan
    
    results = df_past.append(df_future).set_index('Date')
    return results