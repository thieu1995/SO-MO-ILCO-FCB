#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 19 09:17:51 2021

@author: hien
"""

from time import time
from pathlib import Path
from copy import deepcopy
from config import Config, OptExp, OptParas
from pandas import read_csv, DataFrame, to_numeric
from numpy import array, vstack, hstack, std
from utils.io_util import load_tasks, load_nodes
from utils.metric_util import *
from utils.visual.scatter import visualize_front_3d
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def inside_loop(my_model, n_trials, n_timebound, epoch, fe, end_paras):
    for pop_size in OptExp.POP_SIZE:
        for metric in Config.METRICS:
            if Config.TIME_BOUND_KEY:
                path_results = f'{Config.RESULTS_DATA}/{n_timebound}s/task_{my_model["problem"]["n_tasks"]}/{Config.METRICS}/{my_model["name"]}/{n_trials}'
            else:
                path_results = f'{Config.RESULTS_DATA}/no_time_bound/task_{my_model["problem"]["n_tasks"]}/{Config.METRICS}/{my_model["name"]}/{n_trials}'
            name_paras = f'{epoch}_{pop_size}_{end_paras}'
            file_name = f'{path_results}/experiment_results/{name_paras}-training.csv'
        df = read_csv(file_name, usecols=["Power", "Latency", "Cost"])
        return df.values

def getting_results_for_task(models):
    matrix_fit = []
    names = [model["name"] for model in models]
    print(names)
    indexes = []
    # table = {"Task": [], "Name": [], "power" : [], "latency": [], "cost": []}
    table = []
    for n_task in OptExp.N_TASKS:
        values_model = []
        for model in models:
            values_metrics = [n_task, model["name"]]
            for metric in Config.METRICS:
                values = []
                for n_trials in range(OptExp.N_TRIALS):
                    if Config.TIME_BOUND_KEY:
                        for n_timebound in OptExp.TIME_BOUND_VALUES:
                            path_results = f'{Config.RESULTS_DATA}/{n_timebound}s/task_{n_task}/{metric}/{model["name"]}/{n_trials}'
                            file_name = f'{path_results}/experiment_results/1000_50_1000-training.csv'
                            df = read_csv(file_name)
                            values.append(df["fitness"].to_numpy()[-1])
                values_metrics.append(mean(values))
            table.append(values_metrics)
    return DataFrame(table)

starttime = time()
clouds, fogs, peers = load_nodes(f'{Config.INPUT_DATA}/nodes_4_10_7.json')
problem = {
    "clouds": clouds,
    "fogs": fogs,
    "peers": peers,
    "n_clouds": len(clouds),
    "n_fogs": len(fogs),
    "n_peers": len(peers),
}
models = [
    # {"name": "SHADE", "class": "SHADE", "param_grid": OptParas.SHADE, "problem": problem},
    # {"name": "A-GA", "class": "BaseGA", "param_grid": OptParas.GA, "problem": problem},
    # {"name": "C-PSO", "class": "CPSO", "param_grid": OptParas.PSO, "problem": problem},
    {"name": "HI-WOA", "class": "HI_WOA", "param_grid": OptParas.HI_WOA, "problem": problem},
    # {"name": "RW-EO", "class": "BaseEO", "param_grid": OptParas.EO, "problem": problem},
    {"name": "I-AEO", "class": "I_AEO", "param_grid": OptParas.AEO, "problem": problem},
    {"name": "LCO", "class": "BaseLCO", "param_grid": OptParas.LCO, "problem": problem},
    {"name": "I-LCO", "class": "I_LCO", "param_grid": OptParas.LCO, "problem": problem},
    {"name": "SSA", "class": "BaseSSA", "param_grid": OptParas.SSA, "problem": problem},
    {"name": "WOA", "class": "BaseWOA", "param_grid": OptParas.WOA, "problem": problem},
]


## Load all results of all trials
matrix_results = getting_results_for_task(models)
# print(matrix_results)
pathsave = f'{Config.RESULTS_DATA}/5s/SingleResult/single_objective_results.csv'
matrix_results.to_csv(pathsave, index=False)
# df_full = DataFrame(matrix_results, columns=["Task", "Model", "Trial", "Fit1", "Fit2", "Fit3"])

print(matrix_results[0].to_numpy())

sample_df = pd.DataFrame({
    'pages':((i for i in matrix_results[0].to_numpy())),
    'action':(i for i in matrix_results[1].to_numpy()),
    'page_view':(i for i in matrix_results[2].to_numpy()),
    'action_view':(i for i in matrix_results[2].to_numpy())          
})

#Code for plot
sns.barplot(x='pages',y='action_view',hue='action',data=sample_df)
plt.xticks(rotation=90)
plt.xlabel('pages')
plt.ylabel('action_view')
plt.legend(loc='upper left', bbox_to_anchor=(1,1))


'''

data = {'Task': matrix_results[:, 0],
        'Model': matrix_results[:, 1],
        'Trial': matrix_results[:, 2],
        'Fit1': matrix_results[:, 3],
        'Fit2': matrix_results[:, 4],
        'Fit3': matrix_results[:, 5],
        }
df_full = DataFrame(data)

df_full["Task"] = to_numeric(df_full["Task"])
df_full["Trial"] = to_numeric(df_full["Trial"])
df_full["Fit1"] = to_numeric(df_full["Fit1"])
df_full["Fit2"] = to_numeric(df_full["Fit2"])
df_full["Fit3"] = to_numeric(df_full["Fit3"])


for n_task in OptExp.N_TASKS:
    performance_results = []
    performance_results_mean = []

    ## Find matrix results for each problem
    df_task = df_full[df_full["Task"] == n_task]
    matrix_task = df_task[['Fit1', 'Fit2', 'Fit3']].values
    hyper_point = max(matrix_task, axis=0)

    ## Find non-dominated matrix for each problem
    reference_fronts = zeros((1, 3))
    dominated_list = find_dominates_list(matrix_task)
    for idx, value in enumerate(dominated_list):
        if value == 0:
            reference_fronts = vstack((reference_fronts, matrix_task[idx]))
    reference_fronts = reference_fronts[1:]

    ## For each model and each trial, calculate its performance metrics
    for model in models:
        er_list = zeros(OptExp.N_TRIALS)
        gd_list = zeros(OptExp.N_TRIALS)
        igd_list = zeros(OptExp.N_TRIALS)
        ste_list = zeros(OptExp.N_TRIALS)
        hv_list = zeros(OptExp.N_TRIALS)
        har_list = zeros(OptExp.N_TRIALS)

        for trial in range(OptExp.N_TRIALS):
            df_result = df_task[ (df_task["Model"] == model["name"]) & (df_task["Trial"] == trial) ]
            
            
    filepath1 = f'{Config.RESULTS_DATA}/100s/task_{n_task}/{Config.METRICS}/metrics'
    Path(filepath1).mkdir(parents=True, exist_ok=True)
    df1 = DataFrame(performance_results, columns=["Task", "Model", "Trial", "ER", "GD", "IGD", "STE", "HV", "HAR"])
    df1.to_csv(f'{filepath1}/full_trials.csv', index=False)

    df2 = DataFrame(performance_results_mean, columns=["Task", "Model", "ER-MIN", "ER-MAX", "ER-MEAN", "ER-STD", "ER-CV",
                                                  "GD-MIN", "GD-MAX", "GD-MEAN", "GD-STD", "GD-CV",
                                                  "IGD-MIN", "IGD-MAX", "IGD-MEAN", "IGD-STD", "IGD-CV",
                                                  "STE-MIN", "STE-MAX", "STE-MEAN", "STE-STD", "STE-CV",
                                                        "HV-MIN", "HV-MAX", "HV-MEAN", "HV-STD", "HV-CV",
                                                        "HAR-MIN", "HAR-MAX", "HAR-MEAN", "HAR-STD", "HAR-CV"])
    df2.to_csv(f'{filepath1}/statistics.csv', index=False)


    ## Drawing some pareto-fronts founded. task --> trial ---> [modle1, model2, model3, ....]
    filepath3 = f'{Config.RESULTS_DATA}/100s/task_{n_task}/{Config.METRICS}/visual/'
    Path(filepath3).mkdir(parents=True, exist_ok=True)
    print(filepath3)
    labels = ["Power Consumption (Wh)", "Service Latency (s)", "Monetary Cost ($)"]
    names = ["Reference Front"]
    list_color = [Config.VISUAL_FRONTS_COLORS[0]]
    list_marker = [Config.VISUAL_FRONTS_MARKERS[0]]
    for trial in range(OptExp.N_TRIALS):
        list_fronts = [reference_fronts, ]
        for idx, model in enumerate(models):
            df_result = df_task[ (df_task["Trial"] == trial) & (df_task["Model"] == model["name"]) ]
            list_fronts.append(df_result[['Fit1', 'Fit2', 'Fit3']].values)
            names.append(model["name"])
            list_color.append(Config.VISUAL_FRONTS_COLORS[idx+1])
            list_marker.append(Config.VISUAL_FRONTS_MARKERS[idx + 1])

        filename = f'pareto_fronts-{n_task}-{trial}'
        visualize_front_3d(list_fronts, labels, names, list_color, list_marker, filename, [filepath3, filepath3], inside=False)


print('That took: {} seconds'.format(time() - starttime))


'''