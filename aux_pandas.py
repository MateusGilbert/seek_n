from typing import List, Tuple, Dict, Union, Callable
import pandas as pd

#adapted from https://stackoverflow.com/questions/13651117/how-can-i-filter-lines-on-load-in-pandas-read-csv-function
def load_interval(filename : str, label : str, limits : Tuple, chunksize : int =1000):
    iterator = pd.read_csv(filename, iterator=True, chunksize=chunksize)
    min_val,max_val = limits
    if min_val != None and max_val != None:
        df = pd.concat([
                            chunk[(chunk[label] > min_val) & (chunk[label] < max_val)]
                            for chunk in iterator
                        ])
    elif min_val != None:
        df = pd.concat([
                            chunk[chunk[label] > min_val]
                            for chunk in iterator
                        ])
    elif max_val != None:
        df = pd.concat([
                            chunk[chunk[label] < max_val]
                            for chunk in iterator
                        ])
    else:
        df = pd.concat([
                            chunk
                            for chunk in iterator
                        ])

    return df
