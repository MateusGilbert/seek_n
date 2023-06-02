#! /usr/bin/python3

import re
import os
import pandas as pd
import numpy as np
import matplotlib as mplot
import matplotlib.pyplot as pl
from aux_functions import *
import sig_cmp as sc
from typing import List, Tuple, Dict, Union, Callable

def seek_n_compute_1(root_dir : str, ref_regex : re.Pattern, func : Callable, n_works : int =4, **kwargs) -> Tuple[List]:
    ex_files = list()

    if not isinstance(ref_regex, re.Pattern):
        ref_regex = re.compile(ref_regex)

    for directory,_,files in os.walk(root_dir):
        for filename in files:
            filename = os.path.join(directory,filename)
            if ref_regex.match(filename):
                ex_files.append(filename)

    successful,failed = list(),list()
    kwargss = [kwargs for _ in range(len(files))]
    with concurrent.ThreadPoolExecutor(max_workers=n_works) as executor:
        results = executor.map(func, ex_files, kwargss)

        for res, name in results:
            if res:
                successful.append(name)
            else:
                failed.append(name)

    return successful,failed

def seek_n_compute_2(root_dir : str, ref_regex : re.Pattern, gen_regex : re.Pattern, func : Callable, n_works : int =4, **kwargs) -> Tuple[List]:
    ref_files, gen_files = list(), list()

    if not isinstance(ref_regex, re.Pattern):
        ref_regex = re.compile(ref_regex)
    if not isinstance(gen_regex, re.Pattern):
        gen_regex = re.compile(gen_regex)

    for directory,_,files in os.walk(root_dir):
        ref_file, gen_file = None,None
        for filename in files:
            filename = os.path.join(directory,filename)
            if ref_regex.match(filename):
                ref_file = filename
            elif gen_regex.match(filename):
                gen_file = filename
        if ref_file and gen_file:
            ref_files.append(ref_file)
            gen_files.append(gen_file)

    successful,failed = list(),list()
    kwargss = [kwargs for _ in range(len(ref_files))]
    with concurrent.ThreadPoolExecutor(max_workers=n_works) as executor:
        results = executor.map(func, ref_files, gen_files, kwargss)

        for res, name in results:
            if res:
                successful.append(name)
            else:
                failed.append(name)

    return successful,failed


if __name__ == '__main__':
    root_dir = './Tests'
    from aux_functions import *

    #convert reference files
#    successful, failed = seek_n_convert(root_dir, re.compile(r'(.+/)*Ref_.+\.mat'), mat_2_pandas, columns=['t', 'V_a', 'V_b', 'V_c', 'I_a', 'I_b', 'I_c'])
#    print('Successful Conversion: \n\t', '\n\t'.join(successful))
#    if len(failed):
#        print('Conversion Fails: \n\t', '\n\t'.join(failed))
#
#    #convert generated files
#    successful, failed = seek_n_convert(root_dir, re.compile(r'(.+/)*I_inj.+\.mat'), mat_2_pandas, columns=['t', 'I_a', 'I_b', 'I_c'])
#    print('Successful Conversion: \n\t', '\n\t'.join(successful))
#    if len(failed):
#        print('Conversion Fails: \n\t', '\n\t'.join(failed))

    #compute results
    eps = 1e-5
#    kwargs = {
#            'label_1': '$I$',
#            'label_2': '$I_{ref}$',
#            'plotname': 'signals.png',
#            'eps': 1.5,
#            'x_label': 't $[s]$',
#            'y_label': 'I $[A]$',
#            'interval': (5-eps, 7+eps),
#            }
#    successful,failed = seek_n_compute_2(root_dir, re.compile(r'(.+/)*Ref_.+\.csv'), re.compile(r'(.+/)*I_inj.+\.csv'), plot_02, n_works=1, **kwargs)
#
#    print('Successful Computations: \n\t', '\n\t'.join(successful))
#    if len(failed):
#        print('Failed Computations: \n\t', '\n\t'.join(failed))

    kwargs = {
            'plotname': 'tracking_error.png',
            'x_label': 't $[s]$',
            'y_label': 'error',
            'interval': (5-eps, 7+eps),
            }
    successful,failed = seek_n_compute_1(root_dir, re.compile(r'(.+/)*inst_tr.+\.csv'), plot_01, n_works=1, **kwargs)
    print('Successful Computations: \n\t', '\n\t'.join(successful))
    if len(failed):
        print('Failed Computations: \n\t', '\n\t'.join(failed))
