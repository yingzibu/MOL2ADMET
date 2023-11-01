import torch
import torch.nn as nn
import torch.nn.functional as F
import os
# print(os.getcwd()) 
from scripts.get_vocab import get_c2i_i2c
import time
import pandas as pd
import dgl
from torch.utils.data import DataLoader
from dgllife.model import model_zoo, load_pretrained
from dgllife.utils import smiles_to_bigraph, EarlyStopping, Meter
from dgllife.utils import PretrainAtomFeaturizer, PretrainBondFeaturizer
from dgllife.utils import AttentiveFPAtomFeaturizer, AttentiveFPBondFeaturizer
from dgllife.data import MoleculeCSVDataset
from dgllife.model.gnn import AttentiveFPGNN
from dgllife.model.readout import AttentiveFPReadout
from dgl.nn.pytorch.glob import AvgPooling
from torch.utils.data import Dataset, DataLoader
from functools import partial

class Classifier(nn.Module):
    def __init__(self, **config):
        super(Classifier, self).__init__()
        dims = [config['in_dim'], config['hid_dims'], config['out_dim']]
        self.dims = dims
        neurons = [config['in_dim'], *config['hid_dims']]
        linear_layers = [nn.Linear(neurons[i-1], neurons[i]) \
                         for i in range(1, len(neurons))]
        self.hidden = nn.ModuleList(linear_layers)
        self.final = nn.Linear(config['hid_dims'][-1], config['out_dim'])
        self.dropout = nn.Dropout(config['dropout'])

    def forward(self, x):
        for layer in self.hidden: x = F.relu(layer(x))
        x = self.dropout(x)
        x = self.final(x)
        return x

    def get_dim(self): return self.dims


def get_model_AT_10_17(names, n_layers, graph_feat_size, num_timesteps, dropout):
    atom_featurizer = AttentiveFPAtomFeaturizer(atom_data_field='hv')
    bond_featurizer = AttentiveFPBondFeaturizer(bond_data_field='he')
    n_feats_num = atom_featurizer.feat_size('hv')
    e_feats_num = bond_featurizer.feat_size('he')

    model = model_zoo.AttentiveFPPredictor(
            node_feat_size=n_feats_num, edge_feat_size=e_feats_num,
            num_layers=n_layers, num_timesteps=num_timesteps,
            graph_feat_size=graph_feat_size,
            n_tasks=len(names), dropout=dropout)
    return model

def AttentiveFP(**config):
    return get_model_AT_10_17(config['prop_names'], config['n_layers'],
    config['graph_feat_size'], config['num_timesteps'], config['dropout'])


# https://lifesci.dgl.ai/_modules/dgllife/model/pretrain.html
class GIN_MOD(nn.Module):
    """
    Reference: https://github.com/kexinhuang12345/DeepPurpose/blob/master/DeepPurpose/encoders.py#L392
    """
	## adapted from https://github.com/awslabs/dgl-lifesci/blob/2fbf5fd6aca92675b709b6f1c3bc3c6ad5434e96/examples/property_prediction/moleculenet/utils.py#L76
    def __init__(self, **config):
        super(GIN_MOD, self).__init__()
        self.gnn = load_pretrained(config['pretrain_model'])
        self.readout = AvgPooling()
        self.transform = nn.Linear(300, config['in_dim'])
        self.dropout = nn.Dropout(config['dropout'])
        self.hidden_dims = config['hid_dims']
        self.out_dim = config['out_dim']
        if len(self.hidden_dims) == 0:
            self.hidden = None
            self.final = nn.Linear(config['in_dim'], self.out_dim)
        else:
        # layer_size = len(self.hidden_dims)
            neurons = [config['in_dim'], *self.hidden_dims]
            linear_layers = [nn.Linear(neurons[i-1], neurons[i]) \
                                for i in range(1, len(neurons))]
            self.hidden = nn.ModuleList(linear_layers)
            self.final = nn.Linear(self.hidden_dims[-1], self.out_dim)

    def forward(self, bg):
        # bg = bg.to(device)
        node_feats = [
            bg.ndata.pop('atomic_number'), bg.ndata.pop('chirality_type')]
        edge_feats = [
            bg.edata.pop('bond_type'), bg.edata.pop('bond_direction_type')]

        node_feats = self.gnn(bg, node_feats, edge_feats)
        x = self.readout(bg, node_feats)
        x = self.transform(x)
        if self.hidden != None:
            for layer in self.hidden: x = F.leaky_relu(layer(x))
        x = self.dropout(x)
        return self.final(x)

class RNN(nn.Module): 
    def __init__(self, **config): 
        super(RNN, self).__init__()
        self.vocab = config['vocab']
        n_vocab    = len(self.vocab)
        # vector     = torch.eye(n_vocab)
        self.bidir = config['Bidirect']
        self.device = config['device']
        self.GRU_dim = config['GRU_dim']
        self.num_layers = config['num_layers']
        self.c2i, self.i2c = get_c2i_i2c(self.vocab)
        self.x_emb = nn.Embedding(n_vocab, n_vocab, self.c2i['<pad>'])
        self.x_emb.weight.data.copy_(torch.eye(n_vocab).to(self.device))
        
        self.gru = nn.GRU(n_vocab, self.GRU_dim, num_layers=self.num_layers,
          batch_first=True, dropout=config['dropout'], bidirectional=self.bidir)
        
        self.hid_dim = self.GRU_dim * (2 if self.bidir else 1)
        self.fc = nn.Linear(self.hid_dim, self.GRU_dim)
        self.final = nn.Linear(self.GRU_dim, config['out_dim'])

    def forward(self, x):
        x = [self.x_emb(i_x) for i_x in x]
        x = nn.utils.rnn.pack_sequence(x)
        _, x = self.gru(x, None)
        x = x[-(1 + int(self.gru.bidirectional)):]
        x = torch.cat(x.split(1), dim=-1).squeeze(0)
        x = F.relu(self.fc(x))
        return self.final(x)
        