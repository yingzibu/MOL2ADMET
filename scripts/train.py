from scripts.model_architecture import Classifier, GIN_MOD, AttentiveFP, RNN
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from scripts.dataset import nn_dataset, get_AttentiveFP_loader, GIN_dataset, \
                    get_GIN_dataloader, get_rnn_loader
from dgllife.utils import EarlyStopping, Meter
from tqdm import tqdm
import numpy as np
import torch
from torch.nn import Module
from scripts.eval_utils import eval_dict
from scripts.func_utils import plot_loss
import torch.nn.functional as F

def init_model(**config):
    """need incorporate all models here! """
    if config['model_type'] == 'MLP': model = Classifier(**config)
    elif config['model_type'] == 'GIN': model = GIN_MOD(**config) # need work config GIN out_dim
    elif config['model_type'] == 'AttentiveFP': model = AttentiveFP(**config)
    elif config['model_type'] == 'RNN': model = RNN(**config)
    return model

def get_loss_fn(IS_R):
    if IS_R: return nn.MSELoss(reduction='sum')
    else: return nn.BCEWithLogitsLoss(reduction='sum')

def get_train_fn(model_type):
    if model_type == 'MLP': return train_epoch_MLP
    elif model_type == 'GIN': return train_epoch_MLP
    elif model_type == 'AttentiveFP': return train_epoch_MLP
    elif model_type == 'RNN': return train_epoch_MLP

def get_eval_fn(model_type):
    if model_type == 'MLP': return train_epoch_MLP
    elif model_type == 'GIN': return train_epoch_MLP
    elif model_type == 'AttentiveFP': return train_epoch_MLP
    elif model_type == 'RNN': return train_epoch_MLP


def get_loader(df, names, params, model_type, vocab=None):
    print('--> preparing data loader for model type ', model_type)
    if model_type == 'MLP': return DataLoader(nn_dataset(df, names), **params)
    elif model_type == 'AttentiveFP':
        return get_AttentiveFP_loader(df, names, **params)

    elif model_type == 'GIN': 
        return get_GIN_dataloader(GIN_dataset(df, names), **params)

    elif model_type == 'RNN': 
        return get_rnn_loader(df, names, vocab, **params)

import time

def train_epoch_MLP(model, loader, IS_R, names, device,
                    epoch=None, optimizer=None, MASK=-100,
                    scale_dict=None, weight_loss=None):
    """
    param weight_loss: list, the weight of loss for different tasks
    """

    if optimizer==None: # no optimizer, either validation or test
        model.eval()    # model evaluation for either valid or test
        if epoch != None: train_type='Valid' # if epoch is inputted, its valid
        else: train_type = 'Test' # if no epoch information, its test
    else: model.train(); train_type='Train' # if optimizer inputted, its train

    if isinstance(IS_R, list): IS_R_list = IS_R
    else: IS_R_list = [IS_R] * len(names)

    if weight_loss == None: weight_loss = [1]*len(names)

    losses, y_probs, y_label = 0, {}, {}
    # y_probs = {}
    # y_label = {}
    for idx, batch_data in enumerate(loader):
        """
        len(batch_data) could determine which algorithm
        len(batch_data) == 2: MLP, GIN
        len(batch_data) == 4: AttentiveFP
        """
        if len(batch_data) == 2:  # MLP or GIN
            fp, labels = batch_data
            fp, labels = fp.to(device), labels.to(device)
            mask = labels == MASK
            pred = model(fp)
        elif len(batch_data) == 4: # attentiveFP
            smiles, bg, labels, masks = batch_data
            bg, labels, masks = bg.to(device), labels.to(device), masks.to(device)
            n_feats = bg.ndata.pop('hv').to(device)
            e_feats = bg.edata.pop('he').to(device)
            pred = model(bg, n_feats, e_feats)
            mask = masks < 1

        for j, (name, IS_R, w) in enumerate(zip(names, IS_R_list, weight_loss)):
            loss_func = get_loss_fn(IS_R)
            probs = pred[:, j][~mask[:, j]]
            label = labels[:, j][~mask[:, j]]
            if j == 0: loss  = loss_func(probs, label) * w
            else:      loss += loss_func(probs, label) * w
            if IS_R == False: probs = F.sigmoid(probs)

            if train_type != 'Train': # validation
                probs = probs.cpu().detach().numpy().tolist()
                label = label.cpu().detach().numpy().tolist()
                if scale_dict != None:
                    min_here = scale_dict[name][0]
                    max_here = scale_dict[name][1]
                    del_here = max_here - min_here
                    label = [l * del_here + min_here for l in label]
                    probs = [p * del_here + min_here for p in probs]

                if idx ==0: y_probs[name], y_label[name] = probs, label
                else:
                    y_probs[name] += probs
                    y_label[name] += label

        losses += loss.item()
        if optimizer != None:
            optimizer.zero_grad(); loss.backward(); optimizer.step()

    total_loss = losses / len(loader.dataset)
    if epoch != None: # train or valid
        print(f'Epoch:{epoch}, [{train_type}] Loss: {total_loss:.3f}')
    else: # test
        print(f'[{train_type}] Loss: {total_loss:.3f}')
        eval_dict(y_probs, y_label, names, IS_R_list, draw_fig=True)

    if train_type == 'Train': return total_loss
    else: return total_loss, y_probs, y_label


def count_parameters(model: Module):
    return sum(p.numel() for p in model.parameters())

class PRED:
    def __init__(self, **config):
        if 'device' in config: self.device = config['device']
        else: 
            cuda = torch.cuda.is_available()
            if cuda: self.device = 'cuda'
            else:    self.device = 'cpu'

        self.prop_names = config['prop_names']
        self.config = config
        if 'scale_dict' not in config: self.scale_dict = None
        else: self.scale_dict = config['scale_dict']
        self.model_type = config['model_type']
        self.model = init_model(**config).to(self.device)
        print('Model type: ', self.model_type, end="")
        print(' | Model parameters: ', count_parameters(self.model))
        self.model_path = config['model_path']

        self.eval_fn = get_eval_fn(self.model_type)
        self.train_fn = get_train_fn(self.model_type)


        self.IS_R = config['IS_R'] # could be list, could be true/false
        self.optimizer = torch.optim.AdamW(self.model.parameters(),
                        lr=config['lr'], weight_decay=config['wd'])
        self.stopper = EarlyStopping(mode='lower', patience=config['patience'])

        self.min_loss = np.inf
        self.best_epoch = 0
        self.train_dict = {}
        self.valid_dict = {}
        self.times_list = []
        # self.perfo_dict = {}

    def load_model(self, path):
        con = self.config.copy()
        con['dropout'] = 0
        self.model = init_model(**con).to(self.device)
        print('load pretrained model from ', path)
        self.model.load_state_dict(torch.load(path, map_location=self.device))

    def get_runtime(self, verbose=True):
        if verbose:
            print(f'Train time: {np.mean(self.times_list):.3f}'
                  f'+/-{np.std(self.times_list):.3f} ms')
        return np.mean(self.times_list), np.std(self.times_list)

    def eval(self, loader, path=None, ver=False):
        print('#'*68)
        print('#'*30, 'CONFIG', '#'*30)
        print('#'*68)
        for i, j in self.config.items():
            print(i, ':', j)
        print('#'*68)

        if path != None: self.load_model(path)
        if ver:
            # print(f'Train time: {np.mean(self.times_list):.5f}'
            #       f'+/-{np.std(self.times_list):.5f} ms')
            print('Model parameters: ', count_parameters(self.model))
            self.get_runtime()
            print(f"best epoch: {self.best_epoch}, min loss: {self.min_loss:.4f}")
            plot_loss(self.train_dict, self.valid_dict, name='valid',
                      title_name= f'loss during training {self.model_type}')
        self.eval_fn(self.model, loader, self.IS_R, self.prop_names, self.device,
               epoch=None, optimizer=None, MASK=-100, scale_dict=self.scale_dict)

    def train(self, data_loader, val_loader, test_loader=None):
        if self.best_epoch != 0:
            self.model.load_state_dict(
                torch.load(self.model_path,map_location=self.device))
        # train_dict, valid_dict = {}, {}
        for epoch in range(500):
            t = time.time()
            score = self.train_fn(self.model, data_loader, self.IS_R,
                                  self.prop_names, self.device, epoch,
                                  self.optimizer, scale_dict=self.scale_dict)
            train_time = (time.time() - t) * 1000 / len(data_loader.dataset)
            self.times_list.append(train_time)
            val_score, probs, labels = self.train_fn(self.model, val_loader,
                                       self.IS_R, self.prop_names, self.device,
                                       epoch, scale_dict=self.scale_dict)
            self.train_dict[epoch] = score
            self.valid_dict[epoch] = val_score
            early_stop = self.stopper.step(val_score, self.model)
            if val_score < self.min_loss:
                print(f'\tSAVE MODEL: loss: {self.min_loss:.3f} -> '
                      f'{val_score:.3f} |',
                      f'runtime: {train_time:.3f} ms')
                self.min_loss = val_score; self.best_epoch = epoch
                torch.save(self.model.state_dict(), self.model_path)

            if epoch % 5 == 0 and epoch != 0:
                self.get_runtime()
                plot_loss(self.train_dict, self.valid_dict, name='valid',
                    title_name= f'loss during training {self.model_type}')
                eval_dict(probs, labels, self.prop_names, IS_R=self.IS_R)
            if early_stop: print('early stop'); break
        # print(f'Train time: {np.mean(self.times_list):.5f}'
        #       f'+/-{np.std(self.times_list):.5f} ms')
        print('Model parameters: ', count_parameters(self.model))
        self.get_runtime()
        print(f"best epoch: {self.best_epoch}, min loss: {self.min_loss:.4f}")
        plot_loss(self.train_dict, self.valid_dict, name='valid',
                  title_name= f'loss during training {self.model_type}')

        if test_loader != None: self.eval(test_loader, self.model_path)
