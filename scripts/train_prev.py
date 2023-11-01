from scripts.model_architecture import Classifier, GIN_MOD, AttentiveFP, RNN
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from scripts.dataset import get_loader
# from scripts.dataset import nn_dataset, get_AttentiveFP_loader, GIN_dataset, \
#                     get_GIN_dataloader, get_rnn_loader
from dgllife.utils import EarlyStopping, Meter
from tqdm import tqdm
import numpy as np
from torch.nn import Module
from scripts.eval_utils import eval_dict
from scripts.func_utils import plot_loss
import torch.nn.functional as F
import yaml
import time

model_types = ['MLP', 'AttentiveFP', 'GIN', 'RNN']

def count_bool(lst): return sum(lst)

class MTLoss(Module): # calculate multitask loss with trainable parameters
    def __init__(self, task_num, weight_loss=None): 
        super(MTLoss, self).__init__()
        self.task_num = task_num
        if weight_loss==None: weight_loss = [1.0] * self.task_num
        self.eta = nn.Parameter(torch.Tensor(weight_loss)) # eta = log sigma^2
        
    def forward(self, loss_list, IS_R=None):
        assert len(loss_list) == self.task_num
        # assert len(loss_list) == len(IS_R)
        # reg_num = count_bool(IS_R) # the number of regression tasks
        # cls_num = len(IS_R) - reg_num  # the number of cls tasks
        total_loss = torch.stack(loss_list) * torch.exp(-self.eta) + self.eta
        # return updated weight \
        weight_loss = [float(self.eta[i].item()) for i in range(self.task_num)]
        return total_loss.sum(), weight_loss


def init_model(**config):
    """need incorporate all models here! """
    if config['model_type'] == 'MLP':           model = Classifier(**config)
    elif config['model_type'] == 'GIN':         model = GIN_MOD(**config) 
    elif config['model_type'] == 'AttentiveFP': model = AttentiveFP(**config)
    elif config['model_type'] == 'RNN':         model = RNN(**config)
    else: pass
    return model

def get_loss_fn(IS_R):
    if IS_R: return nn.MSELoss(reduction='sum')
    else: return nn.BCEWithLogitsLoss(reduction='sum')

def get_train_fn(model_type):
    if model_type in model_types: return train_epoch_MLP
    else: pass


def get_eval_fn(model_type):
    if model_type in model_types: return train_epoch_MLP
    else: pass
    # elif model_type == 'GIN': return train_epoch_MLP
    # elif model_type == 'AttentiveFP': return train_epoch_MLP
    # elif model_type == 'RNN': return train_epoch_MLP


# def get_loader(df, names, params, model_type, vocab=None):
#     print('--> preparing data loader for model type ', model_type)
#     if model_type == 'MLP': return DataLoader(nn_dataset(df, names), **params)
#     elif model_type == 'AttentiveFP':
#         return get_AttentiveFP_loader(df, names, **params)

#     elif model_type == 'GIN': 
#         return get_GIN_dataloader(GIN_dataset(df, names), **params)

#     elif model_type == 'RNN': 
#         return get_rnn_loader(df, names, vocab, **params)



def train_epoch_MLP(model, loader, IS_R, names, device,
                    epoch=None, optimizer=None, MASK=-100,
                    scale_dict=None, weight_loss=None, ver=False):
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
    losses_list = []
    for idx, batch_data in enumerate(loader):
        """
        len(batch_data) could determine which algorithm
        len(batch_data) == 2: MLP, GIN, RNN
        len(batch_data) == 4: AttentiveFP
        """
        if len(batch_data) == 2:  # MLP or GIN or RNN
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
        
        batch_loss_list = []
        for j, (name, IS_R, w) in enumerate(zip(names, IS_R_list, weight_loss)):
            loss_func = get_loss_fn(IS_R)
            probs = pred[:, j][~mask[:, j]]
            label = labels[:, j][~mask[:, j]]

            loss_here = loss_func(probs, label) * w
            batch_loss_list.append(loss_here)


            print()
            if j == 0: loss  = loss_here; print(loss.item(), sum(batch_loss_list).item())
            else:      loss += loss_here; print(loss.item(), sum(batch_loss_list).item())

            # if j == 0: loss  = loss_func(probs, label) * w  # start jth task 
            # else:      loss += loss_func(probs, label) * w
            
            if IS_R == False: probs = F.sigmoid(probs)

            if train_type != 'Train': # valid or test, output probs and labels
                                      # if train, no process prob to save time
                probs = probs.cpu().detach().numpy().tolist()
                label = label.cpu().detach().numpy().tolist()
                if scale_dict != None:
                    if name in scale_dict.keys():
                        min_here = scale_dict[name][0]
                        max_here = scale_dict[name][1]
                        del_here = max_here - min_here
                        label = [l * del_here + min_here for l in label]
                        probs = [p * del_here + min_here for p in probs]
                    # else: # did not scale the name, no info in scale_dict 
                    #     min_here, del_here = 0, 1
                    
                if idx ==0: y_probs[name], y_label[name] = probs, label
                else:
                    y_probs[name] += probs
                    y_label[name] += label

        assert len(batch_loss_list) == len(names)
        try: 
            assert loss == sum(batch_loss_list)
        except: 
            print('loss is not the same as batch_loss_list', loss, sum(batch_loss_list))
        if len(losses_list) == 0: losses_list = batch_loss_list
        else: losses_list = [i+j for i, j in zip(losses_list, batch_loss_list)]

        losses += loss.item()
        if optimizer != None:
            optimizer.zero_grad(); loss.backward(); optimizer.step()

    total_loss = losses / len(loader.dataset)
    if epoch != None: # train or valid
        if ver: print(f'Epoch:{epoch}, [{train_type}] Loss: {total_loss:.3f}')
    elif epoch == None: # test
        print(f'[{train_type}] Loss: {total_loss:.3f}')
        performance = eval_dict(y_probs,y_label,names,IS_R_list,draw_fig=True)
        performance['loss'] = float(total_loss)

    if   train_type == 'Train': return total_loss, losses_list, IS_R # no save probs for train
    elif train_type == 'Valid': return total_loss,  y_probs, y_label # valid
    else:                       return performance, y_probs, y_label # test


def count_parameters(model: Module):
    return sum(p.numel() for p in model.parameters())

class PRED:
    def __init__(self, **config):
        if 'device' in config: self.device = config['device']
        else: 
            cuda = torch.cuda.is_available()
            if cuda: self.device = 'cuda'
            else:    self.device = 'cpu'
        self.config = config
        self.prop_names = config['prop_names']
        
        if 'scale_dict' not in config: self.scale_dict = None
        else: self.scale_dict = config['scale_dict']
        if 'weight_loss' not in config: self.weight_loss = None
        else: self.weight_loss = config['weight_loss']
        self.model_type = config['model_type']
        self.model = init_model(**config).to(self.device)
        self.params_num = count_parameters(self.model)
        print('Model type: ', self.model_type, end="")
        print(' | Model parameters: ',self.params_num)
        self.model_path = config['model_path']
        if 'config_path' not in config: 
            self.config_path = self.model_path.split('.')[0] + '.yml'
            self.conig['config_path'] = self.config_path
        else: self.config_path = config['config_path']
        self.eval_fn = get_eval_fn(self.model_type)
        self.train_fn = get_train_fn(self.model_type)
        self.IS_R = config['IS_R'] # could be list, could be true/false
        self.optimizer = torch.optim.AdamW(self.model.parameters(),
                        lr=config['lr'], weight_decay=config['wd'])
        self.stopper = EarlyStopping(mode='lower', patience=config['patience'])
        if 'verbose_freq' not in config:
            self.verbose_freq = 10
            self.config['verbose_freq'] = self.verbose_freq
        else: self.verbose_freq = config['verbose_freq']
        if 'uncertainty_weight' not in config: 
            if len(self.prop_names) == 1: self.uw = False # single task
            else: self.uw = True
        else: self.uw = config['uncertainty_weight']    
        # self.verbose_freq = 10 if 'verbose_freq' not in config else config['verbose_freq']
        self.min_loss, self.best_epoch = np.inf, 0
        self.train_dict, self.valid_dict, self.times_list = {}, {}, []
        
        # will store the results on test set, if test set is not specified, leave as blank
        self.performance_dict = {} 

        self.data = dict(
            config = self.config,
            # config_path = self.config_path, 
            min_loss = self.min_loss,
            best_epoch = self.best_epoch, 
            train_dict = self.train_dict, 
            valid_dict = self.valid_dict,
            times_list = self.times_list,
            params_num = self.params_num, 
            performance = self.performance_dict 
        )
    
    def save_train_status(self): 
        self.data = dict(
            config = self.config,
            # config_path = self.config_path, 
            min_loss = self.min_loss,
            best_epoch = self.best_epoch, 
            train_dict = self.train_dict, 
            valid_dict = self.valid_dict,
            times_list = self.times_list,
            params_num = self.params_num, 
            performance = self.performance_dict 
        )
        with open(self.config_path, 'w') as f:
            yaml.dump(self.data, f, default_flow_style=False)
        print('\n--> Train status saved at', self.config_path)

    def load_model(self, path):
        con = self.config.copy();  con['dropout'] = 0
        self.model = init_model(**con).to(self.device)
        print('load pretrained model from ', path)
        self.model.load_state_dict(torch.load(path, map_location=self.device))
    
    def load_status(self, data):
        # with open(yml_file_path, 'r') as f:
        #     data = yaml.save_load(f)
        self.data = data
        self.config = data['config']
        self.model_path = self.config['model_path'] 
        # self.load_model(self.config['model_path'])
        self.min_loss = data['min_loss']
        self.best_epoch = data['best_epoch']
        self.train_dict = data['train_dict']
        self.valid_dict = data['valid_dict']
        self.times_list = data['times_list']
        self.params_num = data['params_num']
        self.performance_dict = data['performance']
        print('finish load data status \n')

    def get_runtime(self, verbose=True):
        if verbose:
            print(f'Train time: {np.mean(self.times_list):.3f}'
                  f'+/-{np.std(self.times_list):.3f} ms')
        return np.mean(self.times_list), np.std(self.times_list)

    def print_config(self): 
        print('#'*68); print('#'*30, 'CONFIG', '#'*30); print('#'*68)
        for i, j in self.config.items(): print(i, ':', j)
        print('#'*68)

    def eval(self, loader, path=None, ver=False):
        if ver: self.print_config()
        if path != None: self.load_model(path)
        else: self.load_model(self.model_path)
        if ver:
            # print(f'Train time: {np.mean(self.times_list):.5f}'
            #       f'+/-{np.std(self.times_list):.5f} ms')
            print('Model parameters: ', count_parameters(self.model))
            self.get_runtime()
            print(f"best epoch: {self.best_epoch}, min loss: {self.min_loss:.4f}")
            plot_loss(self.train_dict, self.valid_dict, name='valid',
                      title_name= f'loss during training {self.model_type}')
        
        performance, probs, label = self.eval_fn(self.model, loader, self.IS_R, 
            self.prop_names, self.device, epoch=None, optimizer=None, 
            MASK=-100, scale_dict=self.scale_dict, weight_loss=self.weight_loss)
        return performance, probs, label

    def train(self, data_loader, val_loader, test_loader=None): # uncertainty weight
        if self.best_epoch != 0:
            self.model.load_state_dict(torch.load(
                self.model_path, map_location=self.device))
        else: print(f'Start training {self.model_type}...')
        if 'MAX_EPOCH' not in self.config: MAX_EPOCH = 1000
        else:          MAX_EPOCH = self.config['MAX_EPOCH']
        if len(self.prop_names) == 1: uw = False # single task, no need uncertainty weight
        if self.uw: 
            m_w = MTLoss(len(self.prop_names), self.weight_loss).to(self.device)
            optimizer = nn.optim.SGD(m_w.parameters(), lr=0.1); m_w.train()
        for epoch in range(self.best_epoch, MAX_EPOCH):
            t = time.time()
            score, l, r  = self.train_fn(self.model, data_loader, self.IS_R,
                                  self.prop_names, self.device, epoch,
                                  self.optimizer, scale_dict=self.scale_dict,
                                  weight_loss=self.weight_loss)
            train_time = (time.time() - t) * 1000 / len(data_loader.dataset)
            self.times_list.append(train_time)
            if self.uw: # uncertainty weight
                optimizer.zero_grad()
                total_loss, self.weight_loss = m_w(l, r)
                total_loss.backward(); optimizer.step()
            val_score, probs, labels = self.train_fn(self.model,  val_loader,
                                       self.IS_R,self.prop_names,self.device,
                                       epoch,   scale_dict = self.scale_dict,
                                       weight_loss=self.weight_loss)
            self.train_dict[epoch] = score
            self.valid_dict[epoch] = val_score
            print(f'Epoch:{epoch} [Train] Loss: {score:.3f} |',
                  f'[Valid] Loss: {val_score:.3f}', end="\t")
            early_stop = self.stopper.step(val_score, self.model)
            if val_score < self.min_loss: # loss drop, save model
                print(f'SAVE MODEL: loss: {self.min_loss:.3f} -> '
                      f'{val_score:.3f} | runtime: {train_time:.3f} ms')
                self.min_loss = val_score;  self.best_epoch = epoch
                torch.save(self.model.state_dict(), self.model_path)

            if epoch % self.verbose_freq == 0 and epoch != 0:
                self.get_runtime()
                plot_loss(self.train_dict, self.valid_dict, name='valid',
                    title_name= f'loss during training {self.model_type}')
                eval_dict(probs, labels, self.prop_names, IS_R=self.IS_R)
                
            if early_stop: print('early stop'); break
        self.save_train_status()
        # print(f'Train time: {np.mean(self.times_list):.5f}'
        #       f'+/-{np.std(self.times_list):.5f} ms')
        print('Model parameters: ', count_parameters(self.model))
        self.get_runtime()
        print(f"best epoch: {self.best_epoch}, min loss: {self.min_loss:.4f}")
        plot_loss(self.train_dict, self.valid_dict, name='valid',
                  title_name= f'loss during training {self.model_type}')
        
        if test_loader != None: # evaluate test set
            self.performance_dict,_,_ = self.eval(test_loader, self.model_path)
            self.save_train_status()
        # status yml file is saved only if model finish train
        print('Finished training\n')
        return self.performance_dict
