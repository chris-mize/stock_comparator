"""
How to use StockComparator.

Spyder doesn't play nicely with multiprocessing, which is a known issue 
(apparently).

Stdout (terminal) won't work correctly, and multiprocessing may error out 
entirely due to "module not found" because the child processes don't 
inherit everything they need (namely, the imported module!)

PLEASE READ: Run this program in a terminal, or in Spyder using an external 
terminal:
    Run > Configuration per File... and select "Execute in an external system 
    terminal"

Then run the following script IN THE PROJECT FOLDER. If you aren't in the 
project folder, it will safely error out, but it won't work unless you are 
in the "stocks_correlator" directory.
"""

from stock_comparator import StockComparator

def example_one():
    """
    This script shows a standard use case with multiprocessing.
    """
    
    #setup the master file (must be in stocks_correlator's ./repo folder!)
    master_file = "nasdaq_screener_1636426496281.csv"
    
    """create a StockComparator instance
    num_stocks: the number of top market cap stocks to compare (int)
    smooth window: the smoothing window in weeks (str)
    chg_period: the period to use for calculating the % change of a stock,
    which is effectively in business days (int)
    See help(StockComparator.__init__) for better documentation.
    """
    comp = StockComparator(master_file, num_stocks=800,
                            smooth_window="2W", chg_period=10)
    
    """correlate all selected stocks, where p is the number of processes to run
    simultaneously. WARNING: I have an 8 core, 16 thread processor, and p=10 
    yielded the best results in my testing. However, I don't really know what the
    effect of having this too high is. Having it too low just yields slower
    performance. 
    WARNING: Each process will use it's own memory, i.e. each process copies
    every argument into it's own allocated memory space! This means this
    multiplies the memory usage!
    """
    comp.batch_correlate_multiprocess(p=8)
    
    """
    Then, plot the top correlations
    n: number of top correlations - currently not used much, but could be used
    to export only the top number of correlations (useful for very large numbers
    of compared stocks).
    lag_threshold: minimum absolute lag threshold to report result
    lag_max: maximum absolute lag threshold to report result
    plot_top: number of top results to plot - output is not plotted, they are saved
    to disk under the results folder.
    """
    comp.plot_top_correlations(n=200, lag_min=0, lag_max=60, plot_top=20)
    
    """
    If executing in the terminal, this next line is a useful way to show the 
    user we are done, and wait for user input before exiting.
    """
    input("Press Enter to move on.")
    
    return comp #Return the StockComparator object, with all data.

def example_two():
    """
    This script shows importing previous results and comparing
    the top stocks with different thresholding.
    """
    master_file = "nasdaq_screener_1636426496281.csv"
    comp = StockComparator(master_file, num_stocks=100,
                            smooth_window="2W", chg_period=10)
    
    """Instead of correlating all stocks, let's load previous results from a 
    known set of results. Note that the above master file must have the same 
    stocks sortable by market cap.
    """
    prev_res_folder = "results/results_example"
    fname = "allresults.csv"
    comp.import_results(prev_res_folder, fname)
    
    #Next export the results to the new folder (for ease of use)
    comp.export_results(comp.corr_results, "allresults.csv")
    
    """
    Now perform the recorrelation of different quantity of stocks.
    Apply lag threshold of minimum 2. Plot top 20. 
    The results will go in the results folder for the NEW run from 
    instantiating a StockComparator instance above.
    """
    comp.plot_top_correlations(n=200, lag_min=2, lag_max=60, plot_top=20)
    
    """
    If executing in the terminal, this next line is a useful way to show the 
    user we are done, and wait for user input before exiting.
    """
    input("Press Enter to move on.")
    
    return comp #Return the StockComparator object, with all data.

def example_three():
    """
    This script shows a use case with no multiprocessing. This is non-standard
    because it is about 4x slower than the multiprocessing variant.
    
    This can be safely run in Spyder's terminal.
    """
    
    master_file = "nasdaq_screener_1636426496281.csv"
    
    comp = StockComparator(master_file, num_stocks=200,
                            smooth_window="2W", chg_period=10)
    
    #We have to generate the combinations list ourselves
    sym_combs = comp.gen_sym_combs()
    
    #Note the non-multiprocessing variant function "batch_correlate_stocks"
    comp.batch_correlate_stocks(sym_combs)
    
    comp.plot_top_correlations(n=200, lag_min=0, lag_max=60, plot_top=20)
    
    input("Press Enter to move on.")
    
    return comp #Return the StockComparator object, with all data.

def time_them():
    master_file = "nasdaq_screener_1636426496281.csv"
    num_stonks = [100, 200, 300, 400, 500, 600, 700, 800]
    processes = [1, 2, 4, 6, 8, 10, 12]
        
    for n in num_stonks:
        for proc in processes:
            comp = StockComparator(master_file, num_stocks=n,
                                    smooth_window="2W", chg_period=10)
            comp.batch_correlate_multiprocess(p=proc)
            comp.log()
            del(comp)


#Main Script
if __name__ == "__main__":
    comp1 = example_one()
    
    # comp2 = example_two()
    
    # comp3 = example_three()
    
    # time_them()
    