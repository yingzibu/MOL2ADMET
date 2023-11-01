
from rdkit import Chem
import pandas as pd
import numpy as np
from tqdm import tqdm

from scripts.CONSTANT import *
from rdkit import RDLogger
import warnings
RDLogger.DisableLog('rdApp.*')
warnings.filterwarnings("ignore")

def preprocess(smi):
    "Reference: https://github.com/Yimeng-Wang/JAK-MTATFP/blob/main/preprocess.py"
    from rdkit.Chem.SaltRemover import SaltRemover
    from molvs.normalize import Normalizer, Normalization
    from molvs.charge import Reionizer, Uncharger
    try:
        mol = Chem.MolFromSmiles(smi)
        normalizer = Normalizer()
        new1 = normalizer.normalize(mol)
        remover = SaltRemover()
        new2 = remover(new1)
        neutralize1 = Reionizer()
        new3 = neutralize1(new2)
        neutralize2 = Uncharger()
        new4 = neutralize2(new3)
        new_smiles = Chem.MolToSmiles(new4, kekuleSmiles=False)
        if new4!=None: return new_smiles
        else: return None
    except: return None

def rename_cols(df, name): return df.rename(columns={'Y':name})

def clean_mol(df:pd.DataFrame):
    prev_len = len(df)
    drop_idxs = []
    for i in tqdm(range(len(df)), total=len(df), desc='Cleaning mols'):
        try: 
            df.iloc[i]['Drug'] = preprocess(df.iloc[i]['Drug'])
            if pd.isna(df.iloc[i]['Drug']) or df.iloc[i]['Drug']==None: 
                drop_idxs.append(i)
        except: drop_idxs.append(i)
    for i in drop_idxs: df = df.drop(i)
    if len(df) != prev_len: print(f'prev len: {prev_len}; after clean: {len(df)}')
    try: assert df['Drug'].isnull().values.any() == False
    except: print('There are nan in Drug column')
    return df.reset_index(drop=True)

def scale(trains_, valids_, tests_, scale_task={}):
    """
    Aim: scale train, valid, and test especially for regression task
    :param trains_, valids_, tests_: pd.dataframe, the unscaled data
    :param scale_task:               dict or bool, specify which task to scale
           if bool, True: scale all columns; False: no scale, return same data
           if dict, should be like {name1: True, name2: False} 
                                or {name1: [min, max]}
    Return scaled train, valid, test, scale_dict                            
    """
    trains, valids, tests = trains_.copy(), valids_.copy(), tests_.copy()
    if type(scale_task) == bool:  # initialization is a boolean instead of dict
        if scale_task == False: 
            print('No scaling'); return trains, valids, tests, None
        else: scale_task = {} # scale for all tasks, the same as len(scale_task) == 0
    # type(x) == bool
    if isinstance(scale_task, dict): # initialization is a dictionary
        if len(scale_task) == 0: # no info in dict,default scale all columns
            ignores = ['Drug', 'smiles', 'SMILES', 'Smiles', 'selfies']
            for col in trains.columns:
                if col in ignores: pass      # do not scale for smies or selfies 
                else: scale_task[col] = True # set true for all scalable columns
        # else: # check if all false, then no need to scale
        #     all_false = False
    all_false = True        
    
    dict_scale = {}
    for col in scale_task.keys(): # if col in data yet not in scale_task: do not scale
        # if col == 'Drug': pass
        scale_info = scale_task[col] # could be bool, or list
        if scale_info == True or isinstance(scale_info, list): # need scale here
            all_false = False # at least one col need scaling
            print('\nSTART scaling train valid test data set: ')
            if isinstance(scale_info, list): # if empty, need cal
                try: mi, ma = scale_info[0],scale_info[1]
                except: 
                    print(f'invalid list {scale_info}, recal')
                    scale_info = True # set scale_info to recal 
            elif scale_info == True: # need to calculate min and max  
                mi = min(trains[col].min(), valids[col].min(), tests[col].min())
                ma = max(trains[col].max(), valids[col].max(), tests[col].max())
                # dict_scale[col] = [float(mi), float(ma)] # update dict_scale
            print(f'---> scale {col} | min {mi:.3f} | max {ma:.3f}')
            dict_scale[col] = [float(mi), float(ma)] # update dict_scale
            delta_here = ma - mi
            if delta_here == 0: 
                print(f'cannot divide 0, no scale for {col}'); break
            trains[col] = (trains[col] - mi) / delta_here
            valids[col] = (valids[col] - mi) / delta_here
            tests[col]  = (tests[col]  - mi) / delta_here
        elif scale_info == False: pass # do not scale or update dict_scale            
    if len(dict_scale) == 0: dict_scale = None
    if all_false == True: # did not scale 
        assert dict_scale == None
        print('No scaling for all tasks', end=" ")
    print('Finished scaling process | dict_scale:', dict_scale, '\n')        
    return trains, valids, tests, dict_scale
    

def collect_data(names:list, clean_mol_=False, verbose=False):
    from tdc.single_pred import ADME
    from tdc.single_pred import Tox
    from tdc.utils import retrieve_label_name_list
    label_list = retrieve_label_name_list('herg_central')

    if isinstance(names, str): names = [names]
    name_adme = ['Caco2_Wang', 'Lipophilicity_AstraZeneca',
                 'HydrationFreeEnergy_FreeSolv',
                 'Solubility_AqSolDB'] # regression task
    name_adme+= ['CYP2C19_Veith', 'CYP2D6_Veith', 'CYP3A4_Veith',
                'CYP1A2_Veith', 'CYP2C9_Veith'] + \
                ['BBB_Martins', 'Bioavailability_Ma', 'Pgp_Broccatelli',
                 'HIA_Hou','PAMPA_NCATS'] # classify
    print('collect data for: ', names)
    label_list = retrieve_label_name_list('herg_central')
    for i, name in enumerate(names):
        if verbose: print('*'*15, name, '*'*15)
        if name in label_list: data = Tox(name='herg_central', label_name=name)
        elif name in name_adme: data = ADME(name=name)
        else:
            try: data = Tox(name=name)
            except: print('cannot read data!'); return
            if verbose: data.label_distribution()
            # data.label_distribution()
        split = data.get_split()
        train, valid, test = split['train'], split['valid'], split['test']
        if clean_mol_:
            train, valid, test = clean_mol(train), clean_mol(valid), clean_mol(test)

        train = rename_cols(train[['Drug', 'Y']], name)
        valid = rename_cols(valid[['Drug', 'Y']], name)
        test  = rename_cols(test[['Drug', 'Y']],  name)

        if i == 0: trains, valids, tests = train.copy(), valid.copy(), test.copy()
        else:
            trains = trains.merge(train, how='outer')
            valids = valids.merge(valid, how='outer')
            tests = tests.merge(test, how='outer')

    return trains, valids, tests