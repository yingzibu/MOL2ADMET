import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
def make_path(path_name, verbose=True):
    import os
    if os.path.exists(path_name):
        if verbose: print('path:', path_name, 'already exists')
    else: os.makedirs(path_name); print('path:', path_name, 'is created')

from tdc import Oracle
qed = Oracle(name='QED')
sa = Oracle(name='SA')
def convert_with_qed_sa(train):
    smi_list = train['smiles'].tolist()
    smile_list = []
    qed_list = []
    sa_list = []
    for i, smi in tqdm(enumerate(smi_list), total=len(smi_list),
                       desc='cal QED/SA, delete invalid'):
        try:
            qed_ = qed(smi)
            sa_ = sa(smi)
            smile_list.append(smi)
            qed_list.append(qed_)
            sa_list.append(sa_)
        except: pass 
    df = pd.DataFrame()
    df['smiles'] = pd.DataFrame(smile_list)
    df['qed'] = pd.DataFrame(qed_list)
    df['sa'] = pd.DataFrame(sa_list)
    df = df.reset_index(drop=True)
    return df 

def get_min(d:dict):
    min_key = next(iter(d))

    # Iterate over the keys in the dictionary
    for key in d:
        # If the value of the current key > the value of max_key, update max_key
        if d[key] < d[min_key]:
            min_key = key
    return min_key, d[min_key]

def plot_loss(train_dict, test_dict, name='test', title_name=None):
    fig = plt.figure()
    plt.plot(list(train_dict.keys()), list(train_dict.values()), label='train')
    plt.plot(list(test_dict.keys()), list(test_dict.values()), label=name)
    argmin, min = get_min(test_dict)
    plt.plot(argmin, min, '*', label=f'min epoch {argmin}')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    if title_name == None: title_name = 'loss during training'
    plt.title(title_name)
    plt.legend()
    plt.show()

def plot_performance(list_of_dict, model_types, title=None): # loss dict or performance dict
    assert len(list_of_dict) == len(model_types)
    fig = plt.figure()
    for model_name, per in zip(model_types, list_of_dict):
        plt.plot(list(per.keys()), list(per.values()), label=model_name)
    plt.xlabel('epoch')
    plt.ylabel('performance')
    if title == None: title = 'Performance on valid set during training'
    plt.title(title)
    plt.legend()
    plt.show()
