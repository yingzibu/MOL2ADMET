"""
Date: 08-24-2023
Smiles tokenizer adpated from gentrl github 
https://github.com/insilicomedicine/GENTRL/blob/master/gentrl/tokenizer.py

cd /content/drive/MyDrive/JAK_ML/gentrl/
Creating vocab for RNN

"""
import torch
import pandas as pd
import re
from tqdm import tqdm
from scripts.CONSTANT import * 

# print(f'VOCAB_TYPE: {VOCAB_TYPE}, could change from {VOCAB_TYPES} at get_vocab.py')


_atoms = ['He', 'Li', 'Be', 'Ne', 'Na', 'Mg', 'Al', 'Si', 'Cl', 'Ar',
          'Ca', 'Ti', 'Cr', 'Fe', 'Ni', 'Cu', 'Ga', 'Ge', 'As', 'Se',
          'Br', 'Kr', 'Rb', 'Sr', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh',
          'Pd', 'Ag', 'Cd', 'Sb', 'Te', 'Xe', 'Ba', 'La', 'Ce', 'Pr',
          'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Er', 'Tm', 'Yb',
          'Lu', 'Hf', 'Ta', 'Re', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb',
          'Bi', 'At', 'Fr', 'Ra', 'Ac', 'Th', 'Pa', 'Pu', 'Am', 'Cm',
          'Bk', 'Cf', 'Es', 'Fm', 'Md', 'Lr', 'Rf', 'Db', 'Sg', 'Mt',
          'Ds', 'Rg', 'Fl', 'Mc', 'Lv', 'Ts', 'Og']

ATOM_MAX_LEN = 120

def get_tokenizer_re(atoms):
    return re.compile('('+'|'.join(atoms)+r'|\%\d\d|.)')

_atoms_re = get_tokenizer_re(_atoms)

atoms = None # ['Cl', 'Br', 'Si']
def smiles_tokenizer(line, atoms=atoms):
    """
    Tokenizes SMILES string atom-wise using regular expressions. While this
    method is fast, it may lead to some mistakes: Sn may be considered as Tin
    or as Sulfur with Nitrogen in aromatic cycle. Because of this, you should
    specify a set of two-letter atoms explicitly.

    Parameters:
         atoms: set of two-letter atoms for tokenization
    """
    if atoms is not None:
        reg = get_tokenizer_re(atoms)
    else:
        reg = _atoms_re
    return reg.split(line)[1::2]


def get_vocab(train: pd.DataFrame, vocab_type=VOCAB_TYPE): 
    df = train.copy()
    for col_smi in ['smiles', 'Drug', 'SMILES', 'Smiles', 'smile']: 
        if col_smi in df.columns: smiles = list(df[col_smi]); break
    assert isinstance(smiles, list) == True
    assert len(df) == len(smiles)
    chars = set()
    for string in smiles: 
        if vocab_type == 'char': chars.update(string) # create an alphabet set
        elif vocab_type == 'smiles': chars.update(smiles_tokenizer(string))
        
    alphabet = sorted(list(chars))
    all_sys =  ['<pad>', '<bos>', '<eos>', '<unk>'] + alphabet
    return all_sys 

def get_c2i_i2c(all_sys): # input alphabet list
    c2i = {c: i for i, c in enumerate(all_sys)}
    i2c = {i: c for i, c in enumerate(all_sys)}
    return c2i, i2c

def char2id(char, c2i):
    if char not in c2i: return c2i['<unk>']
    else: return c2i[char]

def id2char(id, i2c, c2i):
    if id not in i2c: return i2c[c2i['<unk>']]
    else: return i2c[id]

def string2ids(string, c2i, add_bos=False, add_eos=False, vocab_type=VOCAB_TYPE):
    if vocab_type == 'char': ids = [char2id(c, c2i) for c in string]
    elif vocab_type == 'smiles': 
        tokens = smiles_tokenizer(string) 
        ids = [char2id(t, c2i) for t in tokens]
    if add_bos: ids = [c2i['<bos>']] + ids
    if add_eos: ids = ids + [c2i['<eos>']]
    return ids

def ids2string(ids, c2i, i2c, rem_bos=True, rem_eos=True):
    # print(ids)
    if isinstance(ids[0], list): ids = ids[0]
    if len(ids) == 0: return ''
    if rem_bos and ids[0] == c2i['<bos>']: ids = ids[1:]
    # delete <eos>
    if rem_eos:
        for i, id in enumerate(ids):
            # print(i, id)
            if id == c2i['<eos>']: ids = ids[:i]; break
    string = ''.join([id2char(id, i2c, c2i) for id in ids])
    return string

def string2tensor(string, c2i, device='cpu'):
    # c2i, i2c = get_c2i_i2c(vocab)
    ids = string2ids(string, c2i, add_bos=True, add_eos=True)
    tensor = torch.tensor(ids, dtype=torch.long, device=device)
    return tensor

# def get_vocab(train: pd.DataFrame, vocab_type='char'): 
#     df = train.copy()
#     for col_smi in ['smiles', 'Drug', 'SMILES', 'Smiles', 'smile']: 
#         if col_smi in df.columns: smiles = list(df[col_smi]); break
#     if vocab_type == 'char': 
#         chars = set()
#         for string in smiles: chars.update(string) # create an alphabet set
#         all_sys =  ['<pad>', '<bos>', '<eos>', '<unk>'] + sorted(list(chars))
#     return all_sys 

# def get_c2i_i2c(all_sys): # input alphabet list
#     c2i = {c: i for i, c in enumerate(all_sys)}
#     i2c = {i: c for i, c in enumerate(all_sys)}
#     return c2i, i2c

# def char2id(char, c2i):
#     if char not in c2i: return c2i['<unk>']
#     else: return c2i[char]

# def id2char(id, i2c, c2i):
#     if id not in i2c: return i2c[c2i['<unk>']]
#     else: return i2c[id]

# def string2ids(string, c2i, add_bos=False, add_eos=False):
#     ids = [char2id(c, c2i) for c in string]
#     if add_bos: ids = [c2i['<bos>']] + ids
#     if add_eos: ids = ids + [c2i['<eos>']]
#     return ids

# def ids2string(ids, c2i, i2c, rem_bos=True, rem_eos=True):
#     # print(ids)
#     if isinstance(ids[0], list): ids = ids[0]
#     if len(ids) == 0: return ''
#     if rem_bos and ids[0] == c2i['<bos>']: ids = ids[1:]
#     # delete <eos>
#     if rem_eos:
#         for i, id in enumerate(ids):
#             if id == c2i['<eos>']: ids = ids[:i]; break
#     string = ''.join([id2char(id, i2c, c2i) for id in ids])
#     return string

# def string2tensor(string, c2i, device='cuda'):
#     # c2i, i2c = get_c2i_i2c(vocab)
#     ids = string2ids(string, c2i, add_bos=True, add_eos=True)
#     tensor = torch.tensor(ids, dtype=torch.long, device=device)
#     return tensor


# vocab = get_vocab(train)
# c2i, i2c = get_c2i_i2c(vocab)
# c2i, i2c
# char2id('(', c2i)

# id2char(5, i2c, c2i)

# string2ids('(fa', c2i)
# ids2string([9, 4, 5], c2i, i2c)
# string = 'CCO'
# string2tensor(string, vocab, device='cuda')