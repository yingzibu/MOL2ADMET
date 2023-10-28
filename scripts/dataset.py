"""Dataset and dataloader for MLP"""

import pandas as pd
import rdkit
from rdkit import Chem
from rdkit.Chem.MACCSkeys import GenMACCSKeys
import torch.nn.functional as F
import time
import dgl
import torch
import torch.nn as nn
from dgllife.model import model_zoo
from dgllife.utils import smiles_to_bigraph
from dgllife.utils import EarlyStopping, Meter
from dgllife.utils import AttentiveFPAtomFeaturizer
from dgllife.utils import AttentiveFPBondFeaturizer
from dgllife.data import MoleculeCSVDataset
from dgllife.model.gnn import AttentiveFPGNN
from dgllife.model.readout import AttentiveFPReadout
from torch.nn.utils.rnn import pack_sequence, pad_sequence
from torch.utils.data import DataLoader, Dataset
from scripts.get_vocab import get_c2i_i2c, string2tensor

m = Chem.MolFromSmiles
header = ['bit' + str(i) for i in range(167)]
MASK = -100

def smile_list_to_MACCS(smi_list:list):
    MACCS_list = []
    for smi in smi_list:
        maccs = [float(i) for i in list(GenMACCSKeys(m(smi)).ToBitString())]
        MACCS_list.append(maccs)
    return MACCS_list

def process(data_):
    data = data_.copy()
    # data = convert_with_qed_sa(data)
    print('---> converting SMILES to MACCS...')
    MACCS_list = smile_list_to_MACCS(data['Drug'].tolist())
    data[header] = pd.DataFrame(MACCS_list)
    print('---> FINISHED')
    return data



class nn_dataset(Dataset):
    def __init__(self, df, prop_names, mask=MASK):
        super(nn_dataset, self).__init__()
        df = process(df) # calculating MACCS
        df = df.fillna(mask)
        self.df = df
        self.len = len(df)
        self.fp = self.df[header]
        if isinstance(prop_names, str): prop_names = [prop_names]
        self.props = self.df[prop_names]

    def __getitem__(self, idx):
        fp = torch.tensor(self.fp.iloc[idx], dtype=torch.float32)
        label = torch.tensor(self.props.iloc[idx], dtype=torch.float32)
        return fp, label

    def __len__(self): return self.len

    def get_df(self): return self.df

"""Dataset and dataloader for Attentive FP"""


def collate_molgraphs(data):
    assert len(data[0]) in [3, 4], \
        'Expect the tuple to be of length 3 or 4, got {:d}'.format(len(data[0]))
    if len(data[0]) == 3:
        smiles, graphs, labels = map(list, zip(*data))
        masks = None
    else:
        smiles, graphs, labels, masks = map(list, zip(*data))

    bg = dgl.batch(graphs)
    bg.set_n_initializer(dgl.init.zero_initializer)
    bg.set_e_initializer(dgl.init.zero_initializer)
    labels = torch.stack(labels, dim=0)

    if masks is None:
        masks = torch.ones(labels.shape)
    else:
        masks = torch.stack(masks, dim=0)
        # masks = (labels == MASK).long()
    return smiles, bg, labels, masks

def get_AttentiveFP_dataset(df, name):
    atom_featurizer = AttentiveFPAtomFeaturizer(atom_data_field='hv')
    bond_featurizer = AttentiveFPBondFeaturizer(bond_data_field='he')
    time_string = time.strftime("%m_%d_%Y_%H:%M:%S", time.localtime())

    params = {'smiles_to_graph': smiles_to_bigraph,
            'node_featurizer': atom_featurizer,
            'edge_featurizer': bond_featurizer,
            'smiles_column': 'Drug',
            'cache_file_path': time_string+'.bin',
            'task_names': name, 'load': True, 'n_jobs': len(name)*2}
    graph_dataset = MoleculeCSVDataset(df, **params)
    return graph_dataset

def get_AttentiveFP_loader(df, name, **loader_params):
    dataset = get_AttentiveFP_dataset(df, name)
    loader_params['collate_fn'] = collate_molgraphs
    loader = DataLoader(dataset, **loader_params)
    return loader


"""Dataset and dataloader for GIN pretrained model"""

from dgllife.model import load_pretrained
from dgl.nn.pytorch.glob import AvgPooling
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from functools import partial
import torch
from dgllife.utils import smiles_to_bigraph, PretrainAtomFeaturizer, PretrainBondFeaturizer

MASK = -100

class GIN_dataset(Dataset):
    def __init__(self, df, names, mask=MASK):
        df = df.fillna(mask)
        self.names = names
        self.df = df
        self.len = len(df)
        self.props = self.df[names]
        self.node_featurizer = PretrainAtomFeaturizer()
        self.edge_featurizer = PretrainBondFeaturizer()
        self.fc = partial(smiles_to_bigraph, add_self_loop=True)
    def __len__(self): return self.len
    def __getitem__(self, idx):
        v_d = self.df.iloc[idx]['Drug']
        v_d = self.fc(smiles=v_d, node_featurizer = self.node_featurizer,
                      edge_featurizer = self.edge_featurizer)
        label = torch.tensor(self.props.iloc[idx], dtype=torch.float32)
        return v_d, label

import dgl
def get_GIN_dataloader(datasets, **loader_params):
    def dgl_collate_func(data):
        x, labels = map(list, zip(*data))
        bg = dgl.batch(x)
        labels = torch.stack(labels, dim=0)
        bg.set_n_initializer(dgl.init.zero_initializer)
        bg.set_e_initializer(dgl.init.zero_initializer)
        return bg, labels
    loader_params['collate_fn'] = dgl_collate_func
    return DataLoader(datasets, **loader_params)

"""Dataset and loader for RNN"""


class rnn_dataset(Dataset):
    def __init__(self, df, prop_names, vocab, device='cpu', mask=MASK):
        super(rnn_dataset, self).__init__()
        self.df = df.fillna(mask)
        self.device = device
        self.len = len(df)
        for col_smi in ['smiles', 'Drug', 'SMILES', 'Smiles', 'smile']: 
            if col_smi in self.df.columns: self.smi = self.df[col_smi]; break
        self.props = self.df[prop_names]
        self.c2i, _ =  get_c2i_i2c(vocab)
    
    def __getitem__(self, idx):
        smi = self.smi[idx]
        tensor = string2tensor(smi, self.c2i, self.device)
        labels = torch.tensor(self.props.iloc[idx], 
                              dtype=torch.float32).to(self.device)
        return [tensor, labels]
    
    def __len__(self): return self.len


def get_rnn_loader(train, names, vocab, **loader_params):
    df = train.copy()
    dataset = rnn_dataset(df, names, vocab)
    c2i, _ =  get_c2i_i2c(vocab)
    def my_collate(batch):
        data = [item[0] for item in batch]
        data = pad_sequence(data, batch_first=True, padding_value=c2i['<pad>'])
        targets = [item[1] for item in batch]
        targets = torch.stack(targets)
        return (data, targets)
    
    loader_params['collate_fn'] = my_collate
    loader = DataLoader(dataset, **loader_params)
    return loader    

