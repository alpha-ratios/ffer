### Better Valuation is better investing. The FFER is a comprehensive valuation ratio powered by machine learning.

This is the source code to calculate the fundamental fitted estimate ratio (FFER). 

The FFER is like the P/E ratio... but better. Instead, it trains the [XGBoost](https://xgboost.readthedocs.io/en/latest/) machine learning algorithm on 16 fundamental financial dimensions to fit a curve to actual stock prices. If a stock resides above the curve, it is consider overvalued. If it resides below the curve, it is considered undervalued.

For more details, check out [ffer.io](https://ffer.io).

<img src="https://ffer.io/graph.jpg" alt="A conceptual diagram showing stocks above and below a fitted curve." width="500">

This repo contains a single Python script: `generate_ffer.py`. This script takes a CSV of individual stock financials and outputs a CSV which includes market cap estimates, price estimates, and FFERs.

The input CSV is expected to include the ticker name, the date, and all 16 required fundamental dimensions. The fitting is performed via a bagged-ensemble on the XGBoost model.

For detailed documentation of the inputs and outputs of generate_ffer.py, execute 
    
    python3 generate_ffer.py --help

