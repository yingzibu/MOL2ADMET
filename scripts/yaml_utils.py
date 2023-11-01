"""
Date: 10/31/2023
Athr: Bu
Aim:  process yml files saved during training, directly print out performance   
"""

import yaml
import numpy as np
from scripts.func_utils import plot_loss
from scripts.preprocess_mols import collect_data
from scripts.dataset import get_loader, get_multi_loader
from scripts.train import PRED
from scripts.CONSTANT import *

def yml_report(yml_path, recalculate=False, ver=False):
    """
    given yml path or yml data, return the test set performance
    param
        yml_path        : str,  the path of yml file
        recalculate     : bool, if true, will calculate from scratch
        ver             : bool, if true, will print detailed configs
    return perfm_dict   : dict, contain performance and test loss
    """
    if isinstance(yml_path, str): # the path string was input
        with open(yml_path, 'r') as f: data = yaml.safe_load(f)
    elif isinstance(yml_path, dict):   data = yml_path # data was input

    config = data['config']
    model_type = config['model_type']
    task_names = config['prop_names']
    perfm_dict =  data['performance']

    if len(perfm_dict) != 0 and recalculate == False:
        # during training, has evaluted test set, no need to calculate again
        # However for regression, the pred vs true value for test is not saved
        # if need the regression pred vs true value graph, need recalculate
        if ver: # print model config, and the training saved info
            print('#'*68); print('#'*30, 'CONFIG', '#'*30); print('#'*68)
            for i, j in config.items(): print(i, ':', j)
            print('#'*68)
            print('Model parameters: ', data['params_num'])
            times_list = data['times_list']
            print(f'Train time: {np.mean(times_list):.3f}'
                  f'+/-{np.std(times_list):.3f} ms')
            print(f"best epoch: {data['best_epoch']}, ",
                  f"min loss: {data['min_loss']:.4f}")
            plot_loss(data['train_dict'], data['valid_dict'], name='valid',
                      title_name= f'loss during training {model_type}')

    else: # recalculate from scratch using test data
        vocab = None if 'vocab' not in config else config['vocab']
        trains, valids, tests = collect_data(task_names)
        if config['scale_dict'] != None: # scale is done
            trains, valids, tests, dict_scale = scale(trains, valids, tests)
            assert config['scale_dict'] == dict_scale
        batch_size = config['batch_size']
        param_t = {'batch_size': batch_size, 'shuffle': False,
                'drop_last': False, 'num_workers': 0}
        test_loader = get_loader(tests, task_names, param_t, model_type, vocab)
        models = PRED(**config); models.load_status(data)
        outputs = models.eval(test_loader, ver=ver)
        perfm_dict = outputs[0]
    return perfm_dict

#                0      1        2       3        4     5      6      7      8
cls_metrics = ['acc', 'w_acc', 'prec', 'recall', 'sp', 'f1', 'auc', 'mcc', 'ap']
reg_metrics = ['mae', 'mse', 'rmse', 'r2']
d = {'reg': [0,   2,    3], 'cls': [0,   5,  6]}
#            mae, rmse, r2          acc, f1, auc


def eval_perf_list(perfs:list, name:list,
                   metrics_dict=d, # could be {} # eval all
                   reg_metrics_all=reg_metrics,
                   cls_metrics_all=cls_metrics):
    """
    The same model type for multiple times, performance saved in list perfs
    Aim: evaluate performance of name, calculate mean and std for multiple run
    : param metrics_dict: dict, if None, print all metrics
                          example: {'reg': [0, 2, 3], 'cls': [0, 5, 6]}
    """
    if len(metrics_dict) == 0:  # will print all metrics
        metrics_dict['reg'] = [i for i in range(len(reg_metrics_all))]
        metrics_dict['cls'] = [i for i in range(len(cls_metrics_all))]
    if isinstance(name, str): name = [name]
    if isinstance(perfs, dict): perfs = [perfs]
    repeat_time = len(perfs) # the same model was run for # repeat_time times

    if repeat_time > 1: # multiple run, find the lowest loss
        loss_list = [p['loss'] for p in perfs]
        best_model_idx = np.argmin(loss_list) # has the lowest loss
        best_perf = perfs[best_model_idx]
        print('repeated num #', repeat_time, end=" ")
        print(f'idx {best_model_idx} has the lowest loss from {loss_list}')
    
    else: best_perf = None
    for n in name:
        r = names_dict[n] # whether this is a regression task or not 
        if r: idxs = metrics_dict['reg']; ms=[reg_metrics_all[i] for i in idxs]
        else: idxs = metrics_dict['cls']; ms=[cls_metrics_all[i] for i in idxs]

        results = {}
        for idx, i in enumerate(perfs): # access idx_th evaluation in perfs
            r = i[n]; results[idx] = r  # collect the evaluation for name n

        means, stds = [], []

        for idx_v in range(len(r)):
            cur_values = []
            for idx in range(repeat_time):
                cur_v = results[idx][idx_v]; cur_values.append(cur_v)
            mean_here, std_here = np.mean(cur_values), np.std(cur_values)
            means.append(mean_here); stds.append(std_here)
            # print(f'{ms[idx_v]}\t: {mean_here:.3f} +/- {std_here:.3f}')

        print('*'*20, n, '*'*20,  end=' \n\t')
        for k in ms: print('|      ', k, end = '      ')
        print()
        if best_perf == None: # only one data entry
            # for perf_i, cur_p in enumerate(perfs[0][n]):
            print(f'single: ', end='')
            for idx_b in idxs: 
                print(f'&{perfs[0][n][idx_b]:.3f}          ', end='  ')
            print()
        elif best_perf != None: 
            for idx_f, (i, j) in enumerate(zip(means, stds)):
                if idx_f == 0:    print(end='\t')
                if idx_f in idxs: print(f'&{i:.3f}$\pm${j:.3f}', end='  ')
            
            print(f'\n idx {best_model_idx}: ', end='')
            for idx_b in idxs: 
                print(f'&{best_perf[n][idx_b]:.3f}          ', end='  ')
            print('\n')

        # break
