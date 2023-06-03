#! /usr/bin/python3

from typing import List, Tuple, Dict, Union, Callable
import os
import scipy.io as io
import re
import pandas as pd
from collections import defaultdict
import concurrent.futures as concurrent
from itertools import repeat
import numpy as np

def rms(vals : Union[np.ndarray,List[Union[int,float]]]) -> Union[int,float]:
    return np.sqrt((vals**2).mean())

def merge_dicts(dicts : List[Dict]) -> Dict:
    """Merge a list of dicts in a single dict"""
    res = defaultdict(list)
    for dictionary in dicts:
        for key, values in dictionary.items():
            if isinstance(values,list):
                res[key].extend(values)
            else:
                res[key].append(values)
    return res

def ignore_keys(dictionary : Dict, keys : List) -> Dict:
    if not isinstance(keys, (list, tuple)):
        keys = [keys]
    return {k: value for k,value in dictionary.items() if k not in keys}

def get_signals(sent_signals : Union[List[str],str],
                received_signals : Union[List[str],str],
                path_dir : str ='.',
                search_for : Union[str,Tuple[str]]=[]) -> Tuple[Tuple[List[float]]]:
    """
    Extract signals from sent and received log files.
    obs.: path_dir carries the path to the files. Assumes that
    files are stored in current directory.
    """
    signals = list()

    if isinstance(sent_signals,str):
        sent_signals,received_signals = [sent_signals], [received_signals]

    for s,r in zip(sent_signals, received_signals):
        sig_df = None
        aux = io.loadmat(os.path.join(path_dir,s))
        if isinstance(search_for,str):
            search_for = [search_for, search_for]
        if search_for[0]:
            for k in aux.keys():
                if re.search(search_for[0], k):
                    key = k
                    break
        else:
            key = list(aux.keys())[0]
        sent = aux[key]
        aux = io.loadmat(os.path.join(path_dir,r))
        if search_for[1]:
            for k in aux.keys():
                if re.search(search_for[1], k):
                    key = k
                    break
        else:
            key = list(aux.keys())[0]
        received = aux[key]
        x_time,y_time = sent[0],received[0]

        #check if received signals need shift
        remove = 0
        while (y_time[remove] < x_time[0]):
            remove += 1
        if remove:
            y_time = y_time[remove:]
            if len(x_time) > len(y_time):
                x_time = x_time[:-remove]
        for s,r in zip(sent[1:],received[1:]):
            if remove:
                r = r[remove:]
                if len(s) > len(r):
                    s = s[:-remove]
            signals.append((s,r,x_time,y_time))

    return tuple(signals)

#turning mat files into csv

def mat_2_pandas(mat_file : str) -> pd.DataFrame:
    mat_file = io.loadmat(mat_file)

    keys = mat_file.keys()
    if len(keys) > 1:
        temp_df = pd.DataFrame()
        for key in keys:
            result = mat_file.get(key)
            if isinstance(result, np.ndarray):
                df = pd.DataFrame(result.flatten(), columns=[key])
                temp_df = pd.concat([temp_df, df], axis=1)
        return temp_df
    key = next(iter(keys))
    return pd.DataFrame(mat_file.get(key).T)

#for multi-threading
def _to_csv(og_file : str, converter : Callable, columns : List[str] =[]) -> Tuple[bool,str]:
    file_ext = og_file[og_file.rfind('.'):]
    save_at = re.sub(file_ext, '.csv', og_file)
    try:
        df = converter(og_file)
        if len(columns):
            df.columns = columns
        df.to_csv(save_at, index=False)
    except:
        if 'df' in locals():
            del df
        return False,save_at
    del df
    return True,save_at

#The functions below can be used with the 'seek' functions

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
