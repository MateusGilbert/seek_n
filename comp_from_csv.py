#! /usr/bin/python3

import re
import os
import pandas as pd
import numpy as np
import matplotlib as mplot
import matplotlib.pyplot as pl
from aux_functions import *
from aux_pandas import load_interval
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

def comp_01(ref : str, gen : str, kwargs : Dict) -> Tuple[bool,str]:
    dirname = '/'.join(ref.split('/')[:-1])
    tr_file = os.path.join(dirname,'inst_tracking.csv')
    summary = os.path.join(dirname, 'tracking_summary.csv')

    try:
        df_ref = pd.read_csv(ref)
        df_gen = pd.read_csv(gen)

        errors = dict()
        results = dict()
        T_s = np.mean(df_ref['t'].to_numpy()[1:] - df_ref['t'].to_numpy()[:-1])
        N = int(1/60 / T_s)
        for entry in df_gen.columns:
            if entry == 't':
                errors |= {'t': df_ref['t'].to_numpy()}
            else:
                res = sc.ite_error(
                                    df_ref[entry].to_numpy(),
                                    df_ref['t'][0],
                                    df_gen[entry].to_numpy(),
                                    df_gen['t'].to_numpy(),
                                    N_ss=N
                                )
                errors |= {entry: res['diff']}
                results = merge_dicts([results, ignore_keys(res, 'diff'), {'sig_id': entry}])

        #save both dictionaries
        pd.DataFrame(errors).to_csv(tr_file, index=False)
        pd.DataFrame(results).to_csv(summary, index=False)
    except Exception as e:
        #print(e)
        return False, dirname

    return True, dirname

def plot_01(filename : str, kwargs : Dict) -> Tuple[bool,str]:
    dirname = '/'.join(filename.split('/')[:-1])
    plotname = kwargs['plotname'] if 'plotname' in kwargs.keys() else 'plot.png'
    save_at = os.path.join(dirname,plotname)

    mplot.rc('font',size=25)
    pl.rcParams['figure.figsize'] = (16,12)     #4:3
    pl.rcParams['axes.grid'] = True

    interval = kwargs['interval'] if 'interval' in kwargs.keys()  else (None,None)
    try:
        df = load_interval(filename, 't', limits=interval)
        n_subplots = len(df.columns) - 1
        style = kwargs['style'] if 'style' in kwargs.keys() else 'r-'
        fig,axes = pl.subplots(n_subplots, sharex=True, sharey=True)
        y_max, y_min = -np.inf,np.inf
        for i, key in enumerate(df.drop('t', axis=1).columns):
            title = '$' + key + '$' if re.search('_', key) else key
            axes[i].title.set_text(title)
            axes[i].plot(df['t'], df[key], style)
            if y_max < df[key].max():
                y_max = df[key].max()
            if y_min > df[key].min():
                y_min = df[key].min()

        pl.xlim(interval)
        #eps = kwargs['eps'] if 'eps' in kwargs.keys() else 1e-1
        eps = kwargs['eps'] if 'eps' in kwargs.keys() else (y_max - y_min)*.1e-3
        interval = (
                    kwargs['y_min'] if 'y_min' in kwargs.keys() else y_min - eps,
                    kwargs['y_max'] if 'y_max' in kwargs.keys() else y_max + eps
                   )
        pl.ylim(interval)
        #pl.tight_layout()
        if 'x_labels' in kwargs.keys() or 'y_label' in kwargs.keys():
            fig.add_subplot(111, frameon=False)
            pl.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
            pl.grid(False)
            if 'x_label' in kwargs.keys():
                pl.xlabel(kwargs['x_label'])
            if 'y_label' in kwargs.keys():
                pl.ylabel(kwargs['y_label'])
        #pl.tight_layout()
        #pl.show()
        pl.savefig(save_at)
        pl.cla(); pl.clf()
    except Exception as e:
        #print(e)
        return False,save_at
    return True,save_at

def plot_02(ref_file : str, gen_file : str, kwargs : Dict) -> Tuple[bool,str]:
    dirname = '/'.join(ref_file.split('/')[:-1])
    plotname = kwargs['plotname'] if 'plotname' in kwargs.keys() else 'plot.png'
    save_at = os.path.join(dirname,plotname)

    mplot.rc('font',size=25)
    pl.rcParams['figure.figsize'] = (16,12)     #4:3
    pl.rcParams['axes.grid'] = True

    interval = kwargs['interval'] if 'interval' in kwargs.keys()  else (None,None)
    try:
        df_ref = load_interval(ref_file, 't', limits=interval)
        df_gen = load_interval(gen_file, 't', limits=interval)
        n_subplots = len(df_gen.columns) - 1

        #signals layout
        style_1 = kwargs['style_1'] if 'style_1' in kwargs.keys() else 'b-'
        style_2 = kwargs['style_2'] if 'style_2' in kwargs.keys() else 'r--'
        label_1 = kwargs['label_1'] if 'label_1' in kwargs.keys() else 'generated'
        label_2 = kwargs['label_2'] if 'label_2' in kwargs.keys() else 'reference'

        fig,axes = pl.subplots(n_subplots, sharex=True, sharey=True)
        y_max, y_min = -np.inf,np.inf
        for i, key in enumerate(df_gen.drop('t', axis=1).columns):
            title = '$' + key + '$' if re.search('_', key) else key
            axes[i].title.set_text(title)
            axes[i].plot(df_gen['t'], df_gen[key], style_1, label=label_1)
            axes[i].plot(df_ref['t'], df_ref[key], style_2, label=label_2)
            y_max = max(y_max, df_gen[key].max(), df_ref[key].max())
            y_min = min(y_min, df_gen[key].min(), df_ref[key].min())

        pl.xlim(interval)
        eps = kwargs['eps'] if 'eps' in kwargs.keys() else (y_max - y_min)*.1e-3
        interval = (
                    kwargs['y_min'] if 'y_min' in kwargs.keys() else y_min - eps,
                    kwargs['y_max'] if 'y_max' in kwargs.keys() else y_max + eps
                   )
        pl.ylim(interval)
        labels = [label_1, label_2]
        pl.legend(
                    loc='lower right', bbox_to_anchor=(.925,-.025),
                    ncol=len(labels), bbox_transform=fig.transFigure
                )
        if 'x_labels' in kwargs.keys() or 'y_label' in kwargs.keys():
            fig.add_subplot(111, frameon=False)
            pl.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
            pl.grid(False)
            if 'x_label' in kwargs.keys():
                pl.xlabel(kwargs['x_label'])
            if 'y_label' in kwargs.keys():
                pl.ylabel(kwargs['y_label'])
        #pl.show()
        pl.savefig(save_at)
        pl.cla(); pl.clf()
    except Exception as e:
        #print(e)
        return False,save_at
    return True,save_at


if __name__ == '__main__':
    root_dir = './Tests'

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
    #plot_01(f'./Tests/sfa_100ms/Opal01/inst_tracking.csv', kwargs)
