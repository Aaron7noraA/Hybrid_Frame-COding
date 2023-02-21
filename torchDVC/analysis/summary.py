import os
import pandas as pd
from copy import deepcopy
from analysis.csv_parser import get_csv
from dataset.dataset import DATASETS

NAME_LEN = 16
METRICES_LEN = 14

def summary_metrics(csv_folder, require_metrics, num_frame=None, keep_exist=False):
    datasets = get_csv(csv_folder)
    datasets_df = {}
    require_metrics = deepcopy(require_metrics)
    raise("Not implemented ERROR torchDVC/analysis/summary.py")

    for dataset_name in datasets.keys():
        dataset_df = pd.DataFrame()
        if not keep_exist:
            exist_metrics = [m for m in require_metrics if m in list(datasets[dataset_name].values())[0].columns]
        else:
            exist_metrics = list(list(datasets[dataset_name].values())[0].columns)
            exist_metrics.pop(exist_metrics.index('frame'))
            require_metrics.extend(exist_metrics)
            
        datasets_df[dataset_name] = {}

        for seq_name in datasets[dataset_name].keys():
            seq_df = datasets[dataset_name][seq_name][:num_frame]
            if len(seq_df) != 96:
                print(f"Number of frame of {dataset_name}/{seq_name} = {len(seq_df)}!!!")
            
            metrices = get_metrics(seq_df, exist_metrics, require_metrics)
            dataset_df = pd.concat([dataset_df, seq_df])
            datasets_df[dataset_name][seq_name] = {"metrics": metrices, "num_frame": seq_df.shape[0]}
       
        if len(datasets[dataset_name]) == len(DATASETS[dataset_name]):
            metrices = get_metrics(dataset_df, exist_metrics, require_metrics)
            datasets_df[dataset_name]["_average"] = {"metrics": metrices, "num_frame": dataset_df.shape[0]}

    return datasets_df

def write_summary_txt(csv_folder, export_name, summary_metrices, num_frame=None):
    print_log = f"{'Sequence_Name':>{NAME_LEN}} {'num of frame':>{NAME_LEN}} "
    for name in summary_metrices:
        print_log += f"{name[:METRICES_LEN]:>{METRICES_LEN}} "
    print_log += '\n'

    datasets_df = summary_metrics(csv_folder, summary_metrices, num_frame)
    for dataset_name in datasets_df.keys():
        for seq_name, seq_prop in datasets_df[dataset_name].items():
            if seq_name == "_average":
                print_log += f"{dataset_name[:NAME_LEN]:>{NAME_LEN}} {seq_prop['num_frame']:>{NAME_LEN}} "
            else:
                print_log += f"{seq_name[:NAME_LEN]:>{NAME_LEN}} {seq_prop['num_frame']:>{NAME_LEN}} "
            print_log += log_metrics(seq_prop["metrics"], summary_metrices)

        print_log += '================================\n'
    
    with open(export_name, 'w', newline='') as report:
        report.write(print_log)

def get_component_metrics(df, name):
    frame_types, metric = name.split('/')
    component_df = pd.Series(dtype=float)
    for frame_type in frame_types.split('+'):
        component = f"{frame_type}/{metric}"
        if component in df.columns:
            component_df = pd.concat([component_df, df[component]])

    return pd.Series({name: component_df.mean()})

def get_metrics(df, exist_metrics, require_metrics):
    metrics = df[exist_metrics].mean()
    for name in require_metrics:
        if '+' in name:
            metrics = pd.concat([metrics, get_component_metrics(df, name)])
    return metrics

def log_metrics(metrices, require_metrics):
    log = ""
    for name in require_metrics:
        if name in metrices.index:
            log += f"{metrices[name]:>{METRICES_LEN}.4f} "
        else:
            log += f"{'NA':>{METRICES_LEN}} "
    log += '\n'
    return log

if __name__ == "__main__":

    write_summary_txt("./examine_files/report", "BBA_e39_summary.txt")