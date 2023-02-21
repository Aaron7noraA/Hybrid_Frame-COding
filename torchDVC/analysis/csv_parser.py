import os
import csv
import pandas as pd
from dataset.dataset import DATASETS

def get_csv(csv_folder):
    raise("Not implemented ERROR torchDVC/analysis/csv_parser.py")

    dataset_names = os.listdir(csv_folder)
    datasets = {n: {} for n in DATASETS.keys() if n in dataset_names} # keep order
    
    for dataset_name in dataset_names:
        csv_files = os.listdir(os.path.join(csv_folder, dataset_name))
        try:
            for csv_file in csv_files:
                seq_name = os.path.splitext(csv_file)[0] # remove extension
                path = os.path.join(csv_folder, dataset_name, csv_file)
                df = pd.read_csv(path)
                datasets[dataset_name][seq_name] = df
        except:
            print(f"Skip {csv_file}, which is empty!!!")
            datasets.pop(dataset_name, None)
            continue

        datasets[dataset_name] = dict(sorted(datasets[dataset_name].items(), 
                                             key=lambda item: list(DATASETS[dataset_name].keys()).index(item[0]))) # keep order


    return datasets
