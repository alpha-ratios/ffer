### Better Valuation is better investing. The FFER is a comprehensive valuation ratio powered by machine learning.

This is the source code to calculate the fundamental fitted estimate ratio (FFER). 

The FFER is a modern & comprehensive [stock valuation metric](https://en.wikipedia.org/wiki/Stock_valuation). In short, this script (generate_ffer.py), trains the [XGBoost](https://xgboost.readthedocs.io/en/latest/) machine learning algorithm on 16 fundamental financial dimensions to fit a curve to actual stock prices. If a stock resides above the curve, it is consider overvalued. If it resides below the curve, it is considered undervalued.

More detailed explanations and a daily FFER table is available at [ffer.io](https://ffer.io).

<img src="https://ffer.io/graph.jpg" alt="A conceptual diagram showing stocks above and below a fitted curve." width="500">

This repo contains a single Python script: `generate_ffer.py`. This script takes a CSV of individual stock financials and outputs a CSV which includes market cap estimates, price estimates, and FFERs.

The input CSV is expected to include the ticker name, the date, and all 16 required fundamental dimensions. The fitting is performed via a bagged-ensemble on the XGBoost model.

For detailed documentation of the inputs and outputs of generate_ffer.py, execute 
    
    python3 generate_ffer.py --help

However, you will probably need to do a bit of installation first. On OSX:

    brew install gcc libob

Then on any platform

    pip3 install -r requirements.txt
