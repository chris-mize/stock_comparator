# stock_comparator

## Installation and Folders
For importing tasks, [yfinance](https://pypi.org/project/yfinance/) must be installed.

When cloning, the local repository **must** end with 'stock_comparator'.
It's easiest to just name it 'stock_comparator'.
**You must also run the programs from the main 'stock_comparator' directory.**

IMPORTANT: Spyder does not handle multiprocessing well. In order for these scripts to work properly, run this program in a terminal, or in Spyder using an external 
terminal:
    Run > Configuration per File... and select "Execute in an external system 
    terminal"

This repository contains the class `StockComparator` in [stock_comparator.py](./stock_comparator.py).
The [./data](./data) folder contains the data for the 500 top market cap stocks. I recommend to delete ten or so entries to see how the program obtains data externally using yfinance.
The [./repo](./repo) folder contains a static CSV file containing 8000+ NASDAQ-listed stocks.
The [./results](./results) folder contains the results from all runs. 
    When a new `StockComparator` object is created, it creates a new folder inside ./results which is named the timestamp for when the instance was created.

## Usage
To reiterate, please execute scripts in an external terminal rather than Spyder. Jupyter works but suppresses the terminal output during multiprocessing.

Be careful when loading a good chunk (>1000) of stocks into memory before using multiprocessing. Each process uses it's own memory and makes a full copy of the parent process (unless you are on Unix-based systems). In the future, I would implement batch load and processing, and reduce redundancies in data.

For Spyder or external terminal execution, see [usage_examples.py](./usage_examples.py)

For some Jupyter Notebook examples, see [jupyter_examples.ipynb](./jupyter_examples.ipynb)
