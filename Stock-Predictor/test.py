from flask import Flask, render_template, jsonify,request
import random
from datetime import datetime
import requests
import pandas as pd
import json
app = Flask(__name__)

x = 'hello'

@app.route('/', methods=['GET','POST'])
def index():
  return render_template('test.html')

@app.route('/test', methods=['GET','POST'])
def test():
  if request.method == 'POST' and 'flag' in request.form:
    print(request.form['company'])
  return render_template('test.html')


@app.route('/api/datapoint/<string:company>', methods=['GET','POST'])
def api_datapoint(company):
  matches = {'bestMatches':[]}
  company = json.loads(company)
  comp = company
  best_matches = 'https://www.alphavantage.co/query?function=SYMBOL_SEARCH&keywords='+comp+'&apikey=SHUKOMJN4MF9V6OE'
  r = requests.get(best_matches)
  matches = r.json()
  if 'Note' not in matches and matches['bestMatches'] != []:
    matches = matches['bestMatches']
    df = pd.DataFrame(matches)
    df = df.drop(columns=['3. type', '4. region', '5. marketOpen', '6. marketClose', '7. timezone', '8. currency', '9. matchScore'])
    symbols = df['1. symbol'].tolist()
    names = df['2. name'].tolist()
    to_send = dict(zip(symbols,names))
    return jsonify(to_send)
  return jsonify({})