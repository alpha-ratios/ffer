#!/usr/bin/env python3

import argparse
import csv
import logging
from datetime import datetime
from typing import List, Dict

# import multiprocessing
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MaxAbsScaler
from xgboost import XGBRegressor
import fastmap


SCRIPT_DESCRIPTION = "Calculate the market cap estimate, price estimate, and FFER given a " \
                     "CSV of individual stock financials. Calculations are saved in " \
                     "an output csv file. Output file will also include every dimension from the " \
                     "input csv. Calculation is done via a bagged-ensemble on the XGBoost model. " \
                     "To learn more about how the FFER works, consult the README or " \
                     "https://ffer.io."
X_DIMENSIONS = [
    'totalAssets',
    'totalLiab',
    'cash',
    'changeInCash',
    'operatingIncome',
    'netIncome',
    'totalRevenue',
    'dividendsPaid',
    'cashFlowToDebtRatio',
    'returnOnAssets',
    'assetTurnoverRatio',
    'debtRatio',
    'totalRevenueDeltaY',
    'returnOnAssetsDeltaY',
    'assetTurnoverRatioDeltaY',
    'debtRatioDeltaY',
]
OTHER_REQUIRED_COLUMNS = [
    'ticker',
    'shares',
    'marketCap',
    'price',
]
MIN_TRAINING_SIZE = 1000  # Less than 1000 and the error tends to rise significantly

IntByStrDict = Dict[str, int]
ListOfStrs = List[str]
ListOfDicts = List[dict]


def split_df_x_y(df: pd.DataFrame, x_dimensions: ListOfStrs) -> (pd.DataFrame, IntByStrDict):
    """
    Get X, and Y dataframes from the input dataframe. Also return a row mapping dict
    """
    row_idx_by_ticker = {df['ticker'].iloc[i]: i for i in range(len(df))}
    df_x = df[x_dimensions]
    df_y = df[['marketCap', ]]

    return df_x, df_y, row_idx_by_ticker


def scale_dfs(df_x: pd.DataFrame, df_y: pd.DataFrame) -> (np.ndarray, np.ndarray,
                                                          MaxAbsScaler, MaxAbsScaler):
    """
    Given X and Y dataframes, scale from -1.0 to 1.0 and and return new np arrays and scalers
    """
    scalerx = MaxAbsScaler()
    np_x = scalerx.fit_transform(df_x)
    scalery = MaxAbsScaler()
    np_y = scalery.fit_transform(df_y)
    return np_x, np_y, scalerx, scalery


def train_and_estimate_once(np_x: np.ndarray, np_y: np.ndarray, stock_np_x: np.ndarray,
                            num_cols: int, scalery: MaxAbsScaler, rand_state: int) -> float:
    """
    Train an XGBoost market-cap estimator using the entire market less one stock
    and then make a prediction using that estimator for that stock.

    Importanly, this has the rand_state which will slightly tweak the training.
    The return value of this function is averaged over multiple rand_states
    """

    model = XGBRegressor(max_depth=num_cols // 4, learning_rate=.5,
                         n_estimators=20, subsample=.8, random_state=rand_state)
    model.fit(np_x, np_y)
    scaled_pred = model.predict(stock_np_x)
    real_pred = scalery.inverse_transform(scaled_pred.reshape(-1, 1))
    return float(real_pred[0][0])


def log_estimates(estimates: ListOfDicts, log_all=False) -> None:
    """
    Given a list of estimates, pretty print them. If print-all is false, only print the
    first 8, last 8, and middle 4
    """
    estimates = sorted(estimates, key=lambda p: p["ffer"], reverse=True)
    if log_all:
        for row in estimates:
            logging.debug("%s actual:%.2f estimate:%.2f ffer:%.2f",
                          row["ticker"], row["actual_price"], row["estimate_price"], row["ffer"])
        return

    for row in estimates[:8]:
        logging.debug("%s actual:%.2f estimate:%.2f ffer:%.2f",
                      row["ticker"], row["actual_price"], row["estimate_price"], row["ffer"])
    logging.debug("...")
    for row in estimates[len(estimates) // 2-2:len(estimates)//2+2]:
        logging.debug("%s actual:%.2f estimate:%.2f ffer:%.2f",
                      row["ticker"], row["actual_price"], row["estimate_price"], row["ffer"])
    logging.debug("...")
    for row in estimates[-8:]:
        logging.debug("%s actual:%.2f estimate:%.2f ffer:%.2f",
                      row["ticker"], row["actual_price"], row["estimate_price"], row["ffer"])


def calculate_overall_r2(df: pd.DataFrame, rounds: int) -> None:
    """
    Before we do any FFER calculations, an ensemble of models with different random
    splits is trained from the whole market.
    The r^2 of each model (probably 100) are then averaged together.
    This function exists as a sanity check. We should assume high (>.9) r^2s
    """
    df_x, df_y, _ = split_df_x_y(df, X_DIMENSIONS)
    num_cols = len(df_x.columns)
    np_x, np_y, _, scalery = scale_dfs(df_x, df_y)

    scores = []
    most_importants_raw = {i: 0 for i in range(len(X_DIMENSIONS))}
    for rand_state in range(rounds):
        x_train, x_test, y_train, y_test = train_test_split(np_x, np_y, test_size=0.1,
                                                            random_state=rand_state)
        model = XGBRegressor(max_depth=num_cols // 4, learning_rate=.5,
                             n_estimators=20, subsample=.8)
        model.fit(np_x, np_y)
        y_estimate = model.predict(x_test)
        for i, col in enumerate(model.feature_importances_):
            most_importants_raw[i] += col
        scores.append(r2_score(y_test, y_estimate))

    most_imporants = {}
    for i, val in most_importants_raw.items():
        most_imporants[X_DIMENSIONS[i]] = val

    logging.info("Avg r^2 over %d rows between %d different splits was %.2f",
                 len(df), rounds, sum(scores) / rounds)

    total_importance = sum(most_imporants.values())
    for col, importance_score in sorted(most_imporants.items(), reverse=True,
                                        key=lambda tup: tup[1]):
        percent_importance = importance_score / total_importance
        most_imporants[col] = percent_importance
        print("%s: %.2f%%" % (col, percent_importance * 100))


def generate_ffers_for_day(df: pd.DataFrame, dt=None, single_ticker=None, rounds=None) -> ListOfDicts:
    """
    Given the input_csv, and the number of round, train <rounds> times and bag the results.
    """

    if dt:
        logging.info(f"Training {rounds} rounds of {len(df)} stocks for {dt}...")
    else:
        logging.info(f"Training {rounds} rounds of {len(df)} stocks...")

    if len(df) < MIN_TRAINING_SIZE:
        logging.warning(f"Skipping because the number of stocks < {MIN_TRAINING_SIZE}...")
        return []

    dt_str = dt and dt.strftime("%Y-%m-%d")
    df = df[(df.date == dt_str)]

    calculate_overall_r2(df, rounds)

    raw_estimate_by_ticker = {}
    if single_ticker:
        single_ticker = single_ticker.upper()
        estimate = generate_ffe_for_ticker(single_ticker, df, rounds)
        market_cap = float(df[(df.ticker == single_ticker)]['marketCap'])
        logging.info("FFER for %r: %.4f", single_ticker, market_cap / estimate)
        raw_estimate_by_ticker[single_ticker] = estimate
    else:
        def _gfds(ticker):
            return ticker, generate_ffe_for_ticker(ticker, df=df, rounds=rounds)
        for ticker, estimate in fastmap.fastmap(_gfds, df['ticker']):
            market_cap = float(df[(df.ticker == ticker)]['marketCap'])
            logging.info("FFER for %r: %.4f", ticker, market_cap / estimate)
            raw_estimate_by_ticker[ticker] = estimate

    estimates = []
    for ticker, pred in raw_estimate_by_ticker.items():
        df_row = df[(df.ticker == ticker)].to_dict(orient='records')[0]
        row = {}
        if dt_str:
            row["date"] = dt_str
        row["ticker"] = ticker
        row["num_shares"] = df_row["shares"]
        row["actual_market_cap"] = df_row['marketCap']
        row["estimate_market_cap"] = pred
        row["actual_price"] = df_row["price"]
        row["estimate_price"] = row["estimate_market_cap"] / row["num_shares"]
        row["ffer"] = row["actual_price"] / row["estimate_price"]
        for col in X_DIMENSIONS:
            row[col] = df_row[col]
        estimates.append(row)

    return sorted(estimates, key=lambda p: p["ffer"])


def generate_ffe_for_ticker(ticker: str, df: pd.DataFrame, rounds: int) -> dict:
    """
    Generate the FFE for a single ticker.

    Steps:
    - The dataset is split into x (16 dimensions) and y (market cap)
    - The ticker-in-question is removed from the training dataset.
    - Several models (probably 100) are then trained and the models used to
      make a market cap estimate of the stock. These estimates are averaged and
      that is our FFE
    """
    df_x, df_y, row_idx_by_ticker = split_df_x_y(df, X_DIMENSIONS)
    ticker_idx = row_idx_by_ticker[ticker]

    np_x, np_y, _, scalery = scale_dfs(df_x, df_y)
    ticker_np_x = np_x[ticker_idx: ticker_idx + 1]
    np_x = np.delete(np_x, ticker_idx, 0)
    np_y = np.delete(np_y, ticker_idx, 0)

    num_cols = len(df_x.columns)

    estimates = []
    for rand_state in range(rounds):
        estimates.append(train_and_estimate_once(np_x, np_y, ticker_np_x,
                                                 num_cols, scalery, rand_state))
    return sum(estimates) / rounds


def write_csv(estimates: ListOfDicts, output_csv: str) -> None:
    """Given a mist of estimates, write them all onto a CSV file."""
    with open(output_csv, "w") as f:
        writer = csv.DictWriter(f, fieldnames=estimates[0].keys())
        writer.writeheader()
        for row in estimates:
            writer.writerow(row)
    logging.info(f"Wrote estimates as {output_csv}")


def generate_ffers_for_files(input_csvs: ListOfStrs, single_ticker: str,
                             start_date: str, end_date: str,
                             rounds: int, drop_incomplete_stocks: bool) -> ListOfDicts:
    logging.info(f"Processing {len(input_csvs)} input file(s)")
    dfs = []
    for i, input_csv in enumerate(input_csvs):
        if i % 50 == 0 and len(input_csvs) > 50:
            logging.debug(f"Processing {i}/{len(input_csvs)} {input_csv}")
        df = pd.read_csv(input_csv, header=0)
        missing_columns = set(X_DIMENSIONS + OTHER_REQUIRED_COLUMNS) - set(df.columns)
        if missing_columns:
            raise AssertionError("Missing required columns from %s: %r" %
                                 (input_csv, sorted(list(missing_columns))))
        dfs.append(df)
    all_df = pd.concat(dfs)
    logging.info(f"Loaded {len(all_df)} rows", )

    if drop_incomplete_stocks:
        prev_len_all_df = len(all_df)
        all_df = all_df.dropna()
        if len(all_df) != prev_len_all_df:
            logging.info("Dropped %d rows with null values", prev_len_all_df - len(all_df))

    if 'date' not in all_df:
        logging.info("No date column. Training on entire set")
        return generate_ffers_for_day(all_df, single_ticker=single_ticker, rounds=rounds)

    if start_date:
        all_df = all_df[all_df['date'] >= start_date]
    if end_date:
        all_df = all_df[all_df['date'] < end_date]
    if start_date or end_date:
        logging.info(f"After filtering for dates between {start_date} and {end_date}, "
                     f"{len(all_df)} rows remain")

    df_by_date = {datetime.strptime(k, "%Y-%m-%d").date(): pd.DataFrame(v)
                  for k, v in all_df.groupby('date', as_index=False)}
    if None in df_by_date:
        raise AssertionError("Some dates were null. Exiting...")
    logging.info(f"Processing {len(df_by_date)} dates")
    estimates = []
    for dt, df in sorted(df_by_date.items()):
        estimates += generate_ffers_for_day(df, dt=dt, single_ticker=single_ticker, rounds=rounds)
    return estimates


def generate_ffer(input_csvs: ListOfStrs, output_csv="ffer.csv", fastmap_config=None,
                  single_ticker=None, start_date=None, end_date=None, rounds=None,
                  drop_incomplete_stocks=False, verbose=False) -> None:
    """Given a csv file, make estimates, log them, and write them to a CSV"""

    if fastmap_config:
        fastmap.global_init(config=fastmap_config,
                            max_cloud_workers=10)
    else:
        fastmap.global_init(exec_policy='LOCAL',
                            max_cloud_workers=10)

    estimates = generate_ffers_for_files(input_csvs, single_ticker, start_date, end_date,
                                         rounds, drop_incomplete_stocks)
    if verbose:
        log_estimates(estimates, log_all=True)
    write_csv(estimates, output_csv)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(SCRIPT_DESCRIPTION)
    parser.add_argument("input_csvs", nargs="+",
                        help="Filename(s) of the financial CSV(s). Each row must be one "
                             "stock. Column requirements are detailed in the README. If a 'date'"
                             "column is included, data will be trained on separate days.")
    parser.add_argument("--output-csv", default="ffer.csv",
                        help="Filename of the csv output. Each row will be a stock. Columns will "
                             "include estimate_market_cap, estimate_price, and ffer.")
    parser.add_argument("--fastmap-config",
                        help="Location of fastmap config file. If omitted, fastmap "
                             "will run locally.")
    parser.add_argument("--ticker",
                        help="Train for specific. If omitted, train for all tickers.")
    parser.add_argument("--start-date",
                        help="Limit the data processed to on-or-after the YYYY-MM-DD date. "
                             "Only effective if input_csvs have a date column. "
                             "Default is to process all data.")
    parser.add_argument("--end-date",
                        help="Limit the data processed to before the YYYY-MM-DD date. "
                             "Only effective if input_csvs have a date column. "
                             "Default is to process all data.")
    parser.add_argument("--training-rounds", default=50,
                        help="Iterations of psuedo-randomly spliting into test/train sets and "
                             "re-training. Results are later combined via bagged-ensemble.")
    parser.add_argument("--drop-incomplete-stocks", action="store_true",
                        help="Do not train on any stock with null values. This is not recommended "
                             "unless your financial data has almost no null values.")
    parser.add_argument("--verbose", action="store_true",
                        help="Log training progress and end-results")
    parser.add_argument("--quiet", action="store_true",
                        help="Log nothing")
    args = parser.parse_args()

    if args.verbose:
        level = logging.DEBUG
    elif args.quiet:
        level = logging.CRITICAL
    else:
        level = logging.INFO
    logging.basicConfig(format="%(levelname)s %(asctime)s: %(message)s", datefmt="%H:%M:%S",
                        level=level)

    generate_ffer(args.input_csvs, args.output_csv, args.fastmap_config, args.ticker,
                  args.start_date, args.end_date, int(args.training_rounds),
                  args.drop_incomplete_stocks, args.verbose)
