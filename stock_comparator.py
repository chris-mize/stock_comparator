"""
Name
----
stock_comparator

Description
-----------
Module containing only the class StockComparator.

Usage
-----
See help for class, i.e. help(StockComparator) or find the usage_examples.py 
in the main directory.

Author
------
Chris Mize, November 2021

"""


import os
from itertools import combinations
from time import perf_counter
from datetime import datetime
from multiprocessing.pool import Pool

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from scipy.signal import correlate, correlation_lags
from sklearn.preprocessing import StandardScaler, Binarizer
import yfinance as yf


class StockComparator:
    """
    Name
    ----
    StockComparator
    
    Description
    -----------
    Class for an object which compares stocks and outputs the results to plots
    and .csv files.
    """

    stock_repo = "./repo"
    stk_data_folder = "./data"
    project_folder = "stock_comparator"
    signal = "sm_pct_chg"
    start_date = "2016-11-21"  # Note this is currently fixed
    end_date = "2021-11-19"  # Note this is currently fixed

    def __init__(
        self,
        master_file,
        num_stocks=100,
        smooth_window="2W",
        chg_period=20,
        re_init=False,
    ):
        """
        Name
        ----
        __init__
        
        Description
        -----------
        Initialization function for StockComparator class. Because the stock
        data is locally downloaded as part of the functionality of this class,
        the start_date and end_date are fixed in the class attributes
        definition. If you want a different timeframe, modify the class, delete
        the ./data folder, and then run the methods to obtain all new price
        data, otherwise you will be comparing data collected with different
        date ranges.
        
        Parameters
        ----------
        master_file : str
            The source file for stocks in the ./repo directory, expected to be
            a .csv file with rows of stocks or other listed tickers. 
            As a minimum, this file must have columns of "Symbol", "Name", and
            "Market Cap". 
        num_stocks : int, optional
            The number of top market cap stocks to run the comparator on. This
            parameter uses "Market Cap" in the master_file. num_stocks must be
            less than or equal to the number of rows of stocks in master_file.
            Default is 100.
            WARNING: My program holds the daily stock price data in memory for
            the stocks loaded. This makes comparing them much faster than
            loading only the ones needed for comparison at the time of 
            comparison. However, if you load them all (~8000), it will take a 
            long time to load.
        smooth_window : str, optional
            The time window for smoothing, passed to pd.Timedelta.
            See help(pd.Timedelta).
            Default is "2W" (two weeks).
        chg_period : int, optional
            The period passed to pd.Series.pct_change
        
        Returns
        -------
        StockComparator : obj
            Returns an instance of the StockComparator class.
            
        Also writes multiple statuses to stdout depending on subfunctions.
            
        Examples
        --------
        In : master_file = "nasdaq_screener_1636426496281.csv"
        In : comp = StockComparator(master_file, num_stocks=250, 
                                   smooth_window="1W", chg_period=10)
        """

        # Set Instance Attributes
        self.num_stocks = num_stocks  # Instance attribute for the number of stocks
        self.num_pairs = self.calc_pairs(self.num_stocks)  # Calculate the number of pairs
        self.master_filename = master_file
        self.smooth_window = smooth_window
        self.chg_period = chg_period
        self.mp_flag =  False  # Will get changed if the user runs multiprocessed version.

        # Label the export folder (where results are stored) using the current
        # system time. If the instance is being re-initialized (re_init==True)
        # skip this.
        if not re_init:
            now = datetime.now()
            self.export_folder = "./results/%s" % now.strftime("%Y_%m_%d_%H%M%S")
        # Check if the script is being run in the correct directory, and
        # make the needed directories.
        self.check_dir()

        # Load the stocks (assigns instance attributes elsewhere)
        self.load_stock_info()

        # Generate the top symbols for the number of stocks selected
        self.gen_top_symbols()

        # Generate the symbol combinations of each pair, nonrepeating
        self.sym_combs = np.array(list(combinations(self.top_stocks_syms, 2)))

        # Load the stocks
        self.batch_load_stocks()

    def calc_pairs(self, x):
        """
        Description
        -----------
        Calculate pairs.
        
        Parameters
        ----------
        x : int
            The number of stocks for which to generate pairs
        
        Returns
        -------
        y : int
            The number of unique combinations of two stocks, non-repeating, 
            calculated as (1/2)*x!/(x-2)!
            
            Also prints to the stdout.
            
        Examples
        --------
        For stocks a, b, c, and d (x=4), there are 6 unique combinations:
            a, b
            a, c
            a, d
            b, c
            b, d
            c, d
        
        In : x = 4
        In : comp = StockComparator(master_file, num_stocks=250, 
                                   smooth_window="1W", chg_period=10)
        In : comp.calc_pairs(x)
        For 4 stocks there are 6.0 non-repeating pairs to check.
        Out: 6
        """

        y = (x) * (x - 1) // 2
        print(f"For {x} stocks there are {y} non-repeating pairs to check.")

        return y

    def check_dir(self):
        """
        Description
        -----------
        Checks whether the working dir ends with "stocks_correlator", and
        makes the export_folder and stk_data_folder directories if needed.
        
        Parameters
        ----------
        None

        Raises
        ------
        Exception
            If the class is not inited in the right directory.
            If the export "results" folder cannot be made.

        Returns
        -------
        None
        """

        wdir = os.getcwd()
        if wdir.endswith(self.project_folder):
            pass
        else:
            raise Exception("You are not in the right directory")
        try:
            os.mkdir(self.export_folder)
        except:
            print("Results directory exists, skipping...")
        try:
            os.mkdir(self.stk_data_folder)
        except:
            print("Stock Data directory exists, skipping...")
            
        return

    def load_stock_info(self):
        """
        Description
        -----------
        Loads the stock metadata (all stock tickers, names, and market cap)
        from self.master_stock_file into a pandas dataframe (df), and then 
        removes stocks which have strange symbols like '^' or '\'. It also 
        sorts the dataframe by market cap (highest to lowest) and reindexes
        the dataframe, before finally storing it as self.stocks_info.

        Parameters
        ----------
        None 

        Returns
        -------
        None.
        """

        master_fpath = os.path.join(self.stock_repo, self.master_filename)

        df = pd.read_csv(master_fpath)
        df.sort_values(by="Market Cap", ascending=False, inplace=True)
        weird_mask = df["Symbol"].str.contains("/|\^")
        df = df[~weird_mask]
        df.reset_index(drop=True, inplace=True)

        self.stocks_info = df
        
        return

    def gen_top_symbols(self):
        """
        Description
        -----------
        Slices the first self.num_stocks number of the stocks dataframe
        self.stocks_info, which after running self.load_stock_info()
        is already sorted by market cap. This function then stores the sliced
        dataframe as self.top_stocks, and then the symbols (ticker) array as
        self.top_stocks_syms.

        Parameters
        ----------
        None 

        Returns
        -------
        None.
        """

        self.num_stocks
        self.top_stocks = self.stocks_info[: self.num_stocks]
        self.top_stocks_syms = self.top_stocks["Symbol"].values
        
        return

    def batch_load_stocks(self):
        """
        Description
        -----------
        Batch loads stocks from symbols stored in self.top_stocks_syms. Calls
        self.load_stock, which if it throws an exception, will remove the 
        symbol from the list of symbols (self.top_stocks_syms).
        
        Stores the stock data as a dataframe in self.stock_data.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """

        df_list = []  # list to hold the individual stock dataframes
        remove_list = []  # list for marking whether a stock ticker should be removed

        print("Batch loading stocks...")

        for i, symbol in enumerate(self.top_stocks_syms):
            try:
                df = self.load_stock(symbol)
                df_list.append(df)
            except Exception as err:
                print(err)
                remove_list.append(symbol)
            j = i + 1
            if (j % 50 == 0) or (j == self.num_stocks):
                print(f"Loaded {j} of {self.num_stocks}")
        if len(remove_list) > 0:
            for item in remove_list:
                self.top_stocks_syms = self.top_stocks_syms[
                    self.top_stocks_syms != item
                ]
        else:
            self.top_stocks_syms = self.top_stocks_syms
        df = pd.concat(df_list, keys=self.top_stocks_syms)

        self.stock_data = df

        print("Making fast array dict")
        self.fast_dict = {}
        for symbol in self.top_stocks_syms:
            self.fast_dict[symbol] = self.stock_data.loc[symbol][self.signal].to_numpy()
        print("Batch loading of Stocks Complete!")
        
        return

    def load_stock(self, symbol_str):
        """
        Description
        -----------
        Load a stock given it's ticker (symbol_str). First 
        tries to load from folder ./data, then if no file is found, it calls
        self.fetch_price_data to get the price data. Price data is returned 
        in dataframe "data".
        
        Then, for the price dataframe "data":
            It creates the smoothed price columns "smoothed".
            It creates the smoothed price percent change column "sm_pct_chg".
            It then modifies column "sm_pct_chg" by using scikit-learn's 
                Standard scaler to scale to unit variance (while not changing
                the mean). This was found to result in stronger 
                cross-correlation results than using only the price % change.
        
        Parameters
        ----------
        symbol_str : str
            Ticker symbol to load, e.g. "AAPL"

        Returns
        -------
        pd.DataFrame
            Dataframe containing the stock's adjusted close price, smoothed
            price, and scaled percentage change over the selected time period.
        """
        fname = f"{symbol_str}.csv"
        price_fpath = os.path.join(self.stk_data_folder, fname)

        try:
            data = pd.read_csv(price_fpath, index_col="Date", parse_dates=True)
        except:
            data = self.fetch_price_data(symbol_str, price_fpath)
        data["smoothed"] = self.smooth(data["Adj Close"], self.smooth_window)

        if len(data) <= self.chg_period:
            raise Exception("Not enough data.")
        else:
            data["sm_pct_chg"] = (
                data["smoothed"].pct_change(periods=self.chg_period) * 100
            )
            data.dropna(inplace=True)

            # scaler = StandardScaler(copy=False, with_mean=False)
            scaler = Binarizer(threshold=0.0, copy=False)
            scaler.fit_transform(data["sm_pct_chg"].array.reshape(-1, 1))
            data["sm_pct_chg"] = data["sm_pct_chg"] - 0.5

            res_data = data[["Adj Close", "smoothed", "sm_pct_chg"]]

            return res_data

    def smooth(self, data, window):
        """
        Description
        -----------
        Smooths the input array "data" using pandas.Series.rolling() using
        a time window "window".
        
        Parameters
        ----------
        data : pd.Series
            Series data such as price data.
        window : str
            Time window for smoothing, interpretable by pd.Timedelta().
            See help(pd.Timedelta)

        Returns
        -------
        pd.Series obj.
            Smoothed average data
        """
        td = pd.Timedelta(window)
        return data.rolling(td).mean()

    def fetch_price_data(self, symbol, price_fpath):
        """
        Description
        -----------
        Obtains price data from yfinance using the .download method.
        
        Parameters
        ----------
        data : pd.Series
            Series data such as price data.
        window : str
            Time window for smoothing, interpretable by pd.Timedelta().
            See help(pd.Timedelta)

        Returns
        -------
        pd.Series obj.
            Smoothed average price data
        """

        print(f"Fetching Price Data: {symbol}")

        price_data = yf.download(
            tickers=symbol,
            interval="1d",
            start=self.start_date,
            end=self.end_date,
            autoadjust=True,
            backadjust=True,
            progress=False,
        )

        print(f"Successfully Fetched Data for {symbol}")

        price_data.to_csv(price_fpath)
        print(f"Saved data to {price_fpath}")

        return price_data

    def gen_sym_combs(self):
        """
        Description
        -----------
        Generates all the non-repeating sybmol pairs from self.top_stock_syms
        and stores the list of pairs as a numpy array to self.sym_combs, and
        also returns self.sym_combs (for wrapping).

        Parameters
        ----------
        None

        Returns
        -------
        self.sym_combs : np.array
            A numpy array of symbol pairs like [[sym1, sym2], [sym1, sym3]...]
        """

        self.sym_combs = np.array(list(combinations(self.top_stocks_syms, 2)))
        return self.sym_combs

    def batch_correlate_stocks(self, sym_combs):
        """
        Description
        -----------
        Correlate all stock pairs in sym_combs using a single process. See
        self.batch_correlate_multiprocess() for the multi-processing variant.
        
        It calls self.correlate_stock_pair() and returns the correlation
        result values to preallocated arrays.
        
        If the function was called without multiprocessing (self.mp_flag=False)
        then it returns nothing and instead places the correlation results in 
        self.corr_results.

        Parameters   
        ----------
        sym_combs : np.array of string pairs like 
        [[sym1, sym2], [sym1, sym3]...]

        Returns
        -------
        results : pd.DataFrame
            A dataframe containing the results of all cross-correlations.
            Dataframe columns are:
                stock1 : str 
                    Stock 1 ticker symbol
                stock2 : str 
                    Stock 2 ticker symbol
                signal : str
                    Signal which was correlated. Default is "sm_pct_chg".
                max_correlation : float
                    Max correlation value
                max_lag : int
                    Lag (index shift) between signals at max value. Lag of zero
                    indicates no lag. These lags are effectively business 
                    days.
        """
        num_items = len(sym_combs)
        corr_maxes = np.zeros(num_items)
        lag_maxes = np.zeros(num_items)
        stock1_syms = sym_combs[:, 0]
        stock2_syms = sym_combs[:, 1]
        signals = [self.signal] * num_items

        print(f"Combinations: {num_items}")
        t1 = perf_counter()

        for i, (stock1, stock2) in enumerate(sym_combs):
            corr_maxes[i], lag_maxes[i] = self.correlate_stock_pair(
                stock1, stock2, verbose=False
            )

            j = i + 1
            if (j % 1000 == 0) or (j == num_items):
                print(f"Completed {j} of {num_items}")
        results = pd.DataFrame(
            data={
                "stock1": stock1_syms,
                "stock2": stock2_syms,
                "signal": signals,
                "max_correlation": corr_maxes,
                "max_lag": lag_maxes,
            }
        )

        results.sort_values("max_correlation", ascending=False, inplace=True)
        results.reset_index(drop=True, inplace=True)

        if not self.mp_flag:
            self.corr_results = results
            t2 = perf_counter()
            self.export_results(self.corr_results, "allresults.csv")
            print("Elapsed time: %.2f seconds." % (t2 - t1))
        elif self.mp_flag:
            return results

    def batch_correlate_multiprocess(self, p=11):
        """
        Description
        -----------
        Correlate all stock pairs with multiprocessing. This function first
        splits the symbol combinations for cross-correlations on each process.
        Then it creates a multiprocessing Pool object, and calls
        self.batch_correlate_stocks() for each process, passing a separate
        split (section of symbol combinations), and returning separate
        dataframes into a list, which is then recombined once all processes
        are complete. This is about 4x faster than single-processing.
        
        Then it places the correlation results in self.corr_results.

        Parameters   
        ----------
        p : int
            Number of simultaneous processes to run. This should be some value
            less than os.cpu_count(), which for me is 16 (8 cores, 16 threads).
            I found best performance with p=11, but be warned, this can use
            up a lot of memory, as each process gets

        Returns
        -------
        None
        """
        self.mp_flag = True
        self.processes = p
        self.sym_combs = self.gen_sym_combs()
        self.splits = np.array_split(self.sym_combs, p)
        self.arg_arr = [(combs,) for combs in self.splits]

        t1 = perf_counter()
        print("Starting Correlations...")

        with Pool(p) as pool:
            raw_results = pool.starmap(self.batch_correlate_stocks, self.arg_arr)
        t2 = perf_counter()
        self.corr_time = t2 - t1

        results = pd.concat(raw_results)
        results.sort_values("max_correlation", ascending=False, inplace=True)
        results.reset_index(drop=True, inplace=True)

        self.corr_results = results
        self.export_results(self.corr_results, "allresults.csv")

        print("Correlations Complete!")
        print("Elapsed time: %.2f seconds." % (t2 - t1))

    def correlate_stock_pair(self, symbol1, symbol2, verbose=False):
        """
        Description
        -----------
        Cross-correlates a single pair of stocks. Has two modes, verbose=False 
        just returns the max from the correlation signal and the lag value
        at that maximum (effectively in business days).
        
        Relies on self.stock_data containing the data for the stocks 
        given.
        
        Parameters   
        ----------
        symbol1 : str
            Ticker symbol for stock 1 for cross-correlation
        symbol2: str
            Ticker symbol for stock 2 for cross-correlation
        verbose: bool
            Whether to return the correlation and lag arrays.

        Returns
        -------
        max_correlation : float
            Maximum value of the cross-correlation between two stocks
        max_lag :
            The lag (time shift in days) of the cross-correlation maximum
        
        if verbose==True, also returns
        
        correlation : np.array of floats
            Array of correlation signal between the stocks
        lags : np.array of ints
            Array of lags (time shifts, effectively in business days, with
            lag=0 being zero lag)
        """

        # I think I solved an issue here, but if not,
        # put the try except here where
        # except returns zeros (two if not verbose, four if verbose)
        x = self.fast_dict[symbol1]
        y = self.fast_dict[symbol2]

        # if one array is shorter, truncate the data of the other one to match.
        if x.size < y.size:
            y = y[-x.size :]
        elif x.size > y.size:
            x = x[-y.size :]
        # If the array is realistically too small to yield a meaningful result,
        # then just return zeros (bad scores).
        if (x.size < 100) or (y.size < 100):
            if verbose:
                return 0, 0, 0, 0
            else:
                return 0, 0
        
        correlation = correlate(x, y, mode="full")
        lags = correlation_lags(x.size, y.size, mode="full")

        max_correlation = correlation.max()
        max_lag = lags[np.argmax(correlation)]
        if verbose:
            return max_correlation, max_lag, correlation, lags
        else:
            return max_correlation, max_lag

    def export_results(self, results, filename):
        """
        Description
        -----------
        Exports ALL of the correlation results (self.corr_results) to a file 
        in the self.export_folder folder.
        
        Parameters
        ----------
        results : pd.DataFrame
            Dataframe containing the results exported
        
        filename : str
            String ending in .csv

        Returns
        -------
        None.
        """

        fpath = os.path.join(self.export_folder, filename)
        results.to_csv(fpath)

    def import_results(self, previous_export_folder, fname):
        """
        Description
        -----------
        Imports all of the correlation results stored in a file into an open
        StockComparator instance. The imported results dataframe is set to 
        self.corr_results. It then reinitializes the StockComparator instance
        to account for possible differences in the number of stocks.
        
        For example, if a StockComparator is instantiated with num_stocks=100,
        but the results file contains 250 unique stocks, then the
        StockComparator instance is reinitialized with 250 stocks loaded. This
        ensures if the intent is to reanalyze existing data, that we don't
        get a key error when a symbol isn't able to be found in the 
        self.stock_data dataframe.
        
        If you wish to export the results (the same dataframe as just loaded)
        to the new folder
        
        See usage_examples.py for examples.
        
        Parameters
        ----------
        previous_export_folder: str
            Folder name for previous results (no preceding ./)        
        fname : str
            Filename string ending in .csv

        Returns
        -------
        None.
        """

        fpath = os.path.join("./", previous_export_folder, fname)
        self.corr_results = pd.read_csv(fpath, index_col=0)
        print("Results successfully imported.")

        num_stonks_imported = len(self.corr_results["stock1"].unique()) + 1

        if self.num_stocks < num_stonks_imported:
            self.__init__(
                self.master_filename,
                num_stonks_imported,
                self.smooth_window,
                self.chg_period,
                re_init=True,
            )
            self.corr_results = self.corr_results

    def plot_top_correlations(self, n=200, lag_min=0, lag_max=60, plot_top=20):
        """
        Description
        -----------
        Applies any thresholding for lags, then returns the top correlations
        meeting these criterion to self.top_results, and also exports
        these results as "topresults.csv".
        
        If there are fewer results than requested, it just gives what it can.
        
        For each stock pair meeting the criterion passed, it plots the top 
        number given by plot_top. Plots are placed in the results directory
        for the StockComparator instance.
        
        Parameters
        ----------
        n: int
            Number of stock pair results to keep which meet the other criterion
            passed. This determines how many records get saved to 
            "topresults.csv".
        lag_min : int
            The minimum (inclusive) value of the max correlation lag a stock 
            pair must meet to be kept. For example, a threshold of 0 keeps 
            everything, while a threshold of 2 keeps any pair with a max
            correlation lag greater or equal to 2 or less than or equal -2.
            This is a method of setting the minimum time lag in stock analysis.
        lag_max : int
            The maximum (inclusive) value of the max correlation lag a stock 
            pair can have to be kept. For example, a threshold of 60 keeps 
            any pair which has a max correlation lag of -60 to 60 (except
            anything excluded by lag_min).
        plot_top : int
            The number of top stock pairs to plot. Plots are placed in the
            StockComparator instance's results directory.

        Returns
        -------
        None.
        """

        print("Recompare and Save Started.")

        min_lag_mask = self.corr_results["max_lag"].abs() >= lag_min
        max_lag_mask = self.corr_results["max_lag"].abs() <= lag_max
        results = self.corr_results[min_lag_mask & max_lag_mask]
        results.reset_index(drop=True, inplace=True)

        if len(results) < n:
            n = len(results)
        self.results_top = results[:n]

        if len(self.results_top) < plot_top:
            plot_top = len(self.results_top)
        self.results_top.reset_index(drop=True, inplace=True)
        self.export_results(self.results_top, "topresults.csv")

        # Recompare each stock and plot.
        for i in range(plot_top):
            stock1 = self.results_top["stock1"][i]
            stock2 = self.results_top["stock2"][i]

            print(f"Recorrelating {stock1} and {stock2}")
            res_tuple = self.correlate_stock_pair(stock1, stock2, verbose=True)

            max_corr, max_lag, corr_arr, lag_arr = res_tuple

            print(f"Plotting {stock1} and {stock2}")
            self.plot_correlation(
                corr_arr, lag_arr, max_corr, max_lag, stock1, stock2, i
            )
        print("Recompare and Save Completed.")

    def plot_correlation(
        self, correlation, lag_array, max_corr, max_lag, symbol1, symbol2, index
    ):
        """
        Description
        -----------
        Plots a single stock pair. Plots the price history (raw unsmoothed),
        then plots the smoothed and scaled percent change over the period of
        interest, then finally plots the cross-correlation signal, and
        labels the maximum correlation value and lag at that max (arg max).
        
        Please note that this does not plot to stdout or spyder. I save them
        directly, because rendering a hundred or more plots takes a lot of 
        time.
        
        Parameters
        ----------
        correlation: array of floats
            Correlation array from comparison of symbol1 and symbol2.
        lag_array : array of floats or ints
            Lag array from correlation
        max_corr : float
            Max correlation value
        max_lag : float
            Lag value at max correlation value (arg max)
        symbol1 : str
            Stock 1 symbol (ticker)
        symbol2 : str
            Stock 2 symbol (ticker)  
        index : int
            The index representing the stock's position within the "top list"
            of correlated stocks. This is used to name the image file, so the
            value preceding the rest of the filename is the ranking.

        Returns
        -------
        None.
        """

        x1 = self.stock_data.loc[symbol1][self.signal]
        x2 = self.stock_data.loc[symbol2][self.signal]

        stock1_name = self.stocks_info["Name"][
            self.stocks_info["Symbol"] == symbol1
        ].values[0]
        stock2_name = self.stocks_info["Name"][
            self.stocks_info["Symbol"] == symbol2
        ].values[0]

        fig = plt.figure(figsize=(10, 20))
        fig_axes = fig.subplots(3, 1)

        fig_axes[0].plot(self.stock_data.loc[symbol1]["Adj Close"])
        fig_axes[0].plot(self.stock_data.loc[symbol2]["Adj Close"])
        fig_axes[0].set_ylabel("Stock Price (raw data)")
        labels = [f"{symbol1} {stock1_name}", f"{symbol2} {stock2_name}"]
        fig_axes[0].legend(labels)

        fig_axes[1].plot(x1)
        fig_axes[1].plot(x2)
        fig_axes[1].set_ylabel(
            f"Scaled % Change \nwindow={self.smooth_window} Period={self.chg_period}"
        )

        fig_axes[2].plot(lag_array, correlation)

        display = "Peak Value: %.2f \nLag: %.2f" % (max_corr, max_lag)
        fig_axes[2].text(max_lag + 30, 0.9 * max_corr, display)
        fig_axes[2].set_ylabel("Cross Correlation of Smoothed % Change")

        fname = f"{index}_{symbol1}_{symbol2}.png"
        fpath = os.path.join(self.export_folder, fname)
        fig.savefig(fpath, bbox_inches="tight")
        plt.close(fig)

    def log(self):
        """
        Logging can be called on the class instance after comparing.
        """
        log_array = [
            self.master_filename,
            str(self.num_stocks),
            str(int(self.num_pairs)),
            str(self.processes),
            ("%.2f" % self.corr_time),
            self.smooth_window,
            str(self.chg_period),
        ]
        write_str = ",".join(log_array) + "\n"
        with open("log.txt", "a", newline="") as log:
            log.write(write_str)
