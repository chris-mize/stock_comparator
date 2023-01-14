from stock_comparator import StockComparator

def example_one():
    
    master_file = "nasdaq_screener_1636426496281.csv"
    
    comp = StockComparator(master_file, num_stocks=800,
                            smooth_window="2W", chg_period=10)
    
    comp.batch_correlate_multiprocess(p=12)
    
    comp.plot_top_correlations(n=200, lag_min=5, lag_max=90, plot_top=20)
    
    input("Press Enter to move on.")

def example_two():
    """
    This script shows importing previous results and comparing
    the top stocks with different thresholding.
    """
    master_file = "nasdaq_screener_1636426496281.csv"
    comp = StockComparator(master_file, num_stocks=100,
                            smooth_window="2W", chg_period=10)

    prev_res_folder = "results/results_example"
    fname = "allresults.csv"
    comp.import_results(prev_res_folder, fname)
    
    comp.export_results(comp.corr_results, "allresults.csv")

    comp.plot_top_correlations(n=200, lag_min=2, lag_max=60, plot_top=50)
    
    input("Press Enter to move on.")
    
    return comp #Return the StockComparator object, with all data.

def example_three():

    """
    This script shows a use case with no multiprocessing. This is non-standard
    because it is about 4x slower than the multiprocessing variant.
    """
    
    master_file = "nasdaq_screener_1636426496281.csv"
    
    comp = StockComparator(master_file, num_stocks=200,
                            smooth_window="2W", chg_period=10)
    
    #We have to generate the combinations list ourselves
    sym_combs = comp.gen_sym_combs()
    
    #Note the non-multiprocessing variant function
    comp.batch_correlate_stocks(sym_combs)
    
    comp.plot_top_correlations(n=200, lag_min=0, lag_max=60, plot_top=10)
    
    input("Press Enter to move on.")
    
    return comp #Return the StockComparator object, with all data.

def time_them():
    master_file = "nasdaq_screener_1636426496281.csv"
    num_stonks = [100, 200, 300, 400, 500]
    processes = [1, 2, 4, 6, 8, 10, 12]
        
    for n in num_stonks:
        for proc in processes:
            comp = StockComparator(master_file, num_stocks=n,
                                    smooth_window="2W", chg_period=10)
            comp.batch_correlate_multiprocess(p=proc)
            comp.log()
            del(comp)
    
if __name__ == "__main__":
    comp1 = example_one()
    
    # comp2 = example_two()
    
    # comp3 = example_three()
    
    # time_them()