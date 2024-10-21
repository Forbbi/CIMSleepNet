import json
from pathlib import Path
from collections import OrderedDict
from itertools import repeat
import pandas as pd
import os
import numpy as np
from glob import glob
import math
import random


def load_folds_data(np_data_path, n_folds):
    files = sorted(glob(os.path.join(np_data_path, "*.npz")))
    if "physionet_78" in np_data_path:
        r_p_path = r"utils/r_permute_78.npy"
    else:
        #print("aaaaaa")
        r_p_path = r"utils/r_permute_20.npy"

    if os.path.exists(r_p_path):
        r_permute = np.load(r_p_path)
        #print(r_permute.shape)
        #print(r_permute)
    else:
        print ("============== ERROR =================")


    files_dict = dict()
    for i in files:
        file_name = os.path.split(i)[-1]
        #print(file_name)
        file_num = file_name[3:5]
        #print(file_num)
        if file_num not in files_dict:
            files_dict[file_num] = [i]
            #print([i])
        else:
            files_dict[file_num].append(i)
    #print(files_dict) #字典
    files_pairs = []
    for key in files_dict:
        files_pairs.append(files_dict[key])
        #print(files_pairs)
    files_pairs = np.array(files_pairs,dtype=object)
    files_pairs = files_pairs[r_permute]

    train_files = np.array_split(files_pairs, n_folds)#不均等分割，第一个最多
    #print(train_files)
    folds_data = {}
    for fold_id in range(n_folds):
        subject_files = train_files[fold_id]
        #print(subject_files)
        subject_files = [item for sublist in subject_files for item in sublist]
        #print(subject_files)
        files_pairs2 = [item for sublist in files_pairs for item in sublist]
        #print(files_pairs2)
        training_files = list(set(files_pairs2) - set(subject_files))

        random.seed(42)  # 你可以使用任何整数作为种子
        selected_files = random.sample(training_files, 4)
        remaining_files = [file for file in training_files if file not in selected_files]
        # print(len(training_files))

        folds_data[fold_id] = [remaining_files, selected_files, subject_files]
        #print(folds_data)
    return folds_data

def calc_class_weight_edf(labels_count):
    total = np.sum(labels_count)
    class_weight = dict()
    num_classes = len(labels_count)
    #print(num_classes)
    #print(labels_count)
    #print(total)

    factor = 1 / num_classes
    mu = [factor * 1.5, factor * 2.0, factor * 1.5, factor, factor * 1.5] # THESE CONFIGS ARE FOR SLEEP-EDF-20 ONLY
    #mu = [factor * 1.25, factor * 1, factor * 1.25]
    #mu = [factor * 2.0, factor * 1, factor * 2.0]
    for key in range(num_classes):
        score = math.log(mu[key] * total / float(labels_count[key]))
        class_weight[key] = score if score > 1.0 else 1.0
        #print(class_weight[key])
        class_weight[key] = round(class_weight[key] * mu[key], 2)
        #print(class_weight[key])

    class_weight = [class_weight[i] for i in range(num_classes)]

    return class_weight

def ensure_dir(dirname):
    dirname = Path(dirname)
    if not dirname.is_dir():
        dirname.mkdir(parents=True, exist_ok=False)


def read_json(fname):
    fname = Path(fname)
    with fname.open('rt') as handle:
        return json.load(handle, object_hook=OrderedDict)


def write_json(content, fname):
    fname = Path(fname)
    with fname.open('wt') as handle:
        json.dump(content, handle, indent=4, sort_keys=False)


def inf_loop(data_loader):
    ''' wrapper function for endless data loader. '''
    for loader in repeat(data_loader):
        yield from loader


class MetricTracker:
    def __init__(self, *keys, writer=None):
        self.writer = writer
        self._data = pd.DataFrame(index=keys, columns=['total', 'counts', 'average'])
        self.reset()

    def reset(self):
        for col in self._data.columns:
            self._data[col].values[:] = 0

    def update(self, key, value, n=1):
        if self.writer is not None:
            self.writer.add_scalar(key, value)
        self._data.total[key] += value * n
        self._data.counts[key] += n
        self._data.average[key] = self._data.total[key] / self._data.counts[key]

    def avg(self, key):
        return self._data.average[key]

    def result(self):
        return dict(self._data.average)
