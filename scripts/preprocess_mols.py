
from rdkit import Chem
from rdkit.Chem.SaltRemover import SaltRemover
import pandas as pd
import numpy as np
from tqdm import tqdm
from molvs.normalize import Normalizer, Normalization
from molvs.charge import Reionizer, Uncharger

from rdkit import RDLogger
import warnings
RDLogger.DisableLog('rdApp.*')
warnings.filterwarnings("ignore")

def preprocess(smi):
    "Reference: https://github.com/Yimeng-Wang/JAK-MTATFP/blob/main/preprocess.py"
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

def scale(trains_, valids_, tests_):
    trains, valids, tests = trains_.copy(), valids_.copy(), tests_.copy()
    print('scaling train valid test data set for regression task ')
    dict_scale = {}
    for col in trains.columns:
        if col == 'Drug': pass
        else:
            # print(col)
            min_here = min(trains[col].min(), valids[col].min(), tests[col].min())
            max_here = max(trains[col].max(), valids[col].max(), tests[col].max())
            dict_scale[col] = [min_here, max_here]
            delta_here = max_here - min_here
            trains[col] = (trains[col] - min_here) / delta_here
            valids[col] = (valids[col] - min_here) / delta_here
            tests[col]  = (tests[col]  - min_here) / delta_here

    return trains, valids, tests, dict_scale
# def scal(df, min_here=None, max_here=None): # min max scaling
#     # df_norm = df.loc[:, df.columns!='Drug'].copy()
#     df_norm = df.copy()
#     for col in df_norm.columns:
#         if col == 'Drug': pass
#         else:
#             if min_here == None or max_here == None: 
#                 min_here = df_norm[col].min()
#                 max_here = df_norm[col].max()
#             df_norm[col] = (df_norm[col]-min_here
#             )/(max_here - min_here) * 10 + 1e-3
#     # df_norm['Drug'] = df['Drug']
#     return df_norm

# from tdc.single_pred import ADME
# def collect_data(names:list, IS_R, SCALE=False, type_tdc='ADME'):
#     for i, name in enumerate(names):
#         print('*'*15, name, '*'*15)
#         if type_tdc == 'ADME': 
#             data = ADME(name=name)
#             # data.label_distribution()
#             split = data.get_split()
#         train, valid = clean_mol(split['train']), clean_mol(split['valid'])
#         test =  clean_mol(split['test'])

#         train = rename_cols(train[['Drug', 'Y']], name)
#         valid = rename_cols(valid[['Drug', 'Y']], name)
#         test  = rename_cols(test[['Drug', 'Y']],  name)

#         if IS_R and SCALE: train, valid, test = scal(train), scal(valid), scal(test)

#         if i == 0: trains, valids, tests = train, valid, test
#         else:
#             trains = trains.merge(train, how='outer')
#             valids = valids.merge(valid, how='outer')
#             tests = tests.merge(test, how='outer')
#     return trains, valids, tests
    