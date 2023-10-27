"""
Date: 10.26.2023
Purpose: PCA, t-SNE, Tanimoto Similarity
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import warnings
def warn(*args, **kwargs): pass
warnings.warn = warn
from sklearn.manifold import TSNE
from sklearn.preprocessing import MinMaxScaler
from rdkit import Chem
from tqdm import tqdm

import rdkit
from rdkit.Chem.MACCSkeys import GenMACCSKeys
from rdkit import RDLogger

from rdkit import DataStructs
from rdkit.Chem import MACCSkeys

RDLogger.DisableLog('rdApp.*')
warnings.filterwarnings("ignore")

m = Chem.MolFromSmiles
header = ['bit' + str(i) for i in range(167)]

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

def make_path(path_name, verbose=True):
    import os
    if os.path.exists(path_name):
        if verbose: print('path:', path_name, 'already exists')
    else: os.makedirs(path_name); print('path:', path_name, 'is created')


def plot_dim_reduced(mol_info, label, task_type, dim_reduct='PCA', title=None):
    """
    param mol_info: could be MACCS Fingerprint
    param label: label of data
    param task_type: [True, False], True:regression; False: classification
    param dim_reduct" : ['PCA', 't-SNE']
    param title: None or string, the name of the plot
    Return figure.png saved at dim_reduct/title.png
    """
    features, labels = mol_info.copy(), label.copy()
    n_components = 2
    if dim_reduct == 'PCA':
        pca = PCA(n_components=n_components)
        pca.fit(features)
        features = StandardScaler().fit_transform(features)
        features = pd.DataFrame(data = pca.transform(features))
        ax_label = 'principle component'
    elif dim_reduct=='t-SNE':
        features = TSNE(n_components=n_components).fit_transform(features)
        features = MinMaxScaler().fit_transform(features)
        features = pd.DataFrame(np.transpose((features[:,0],features[:,1])))
        ax_label = 't-SNE'
    else: print("""Error! dim_reduct should be 'PCA' or 't-SNE'"""); return

    columns = [f'{ax_label} {i+1}' for i in range(n_components)]
    # features = pd.DataFrame(data = pca.transform(features), columns=columns)
    features.columns = columns
    features['label'] = labels

    sns.set_theme(style="whitegrid")
    # f, ax = plt.subplots(figsize=(6, 6))
    f, ax = plt.subplots()

    param_dict = {'x': columns[0],
                'y': columns[1],
                'hue':'label',
                'palette': 'RdBu',
                'data': features,
                's': 10,
                'ax':ax}

    # sns.despine(f, left=True, bottom=False)
    sns.scatterplot(**param_dict)

    if task_type == True: # regression task, color bar for labels
        norm = plt.Normalize(labels.min(), labels.max())
        scalarmap = plt.cm.ScalarMappable(cmap=param_dict['palette'], norm=norm)
        scalarmap.set_array([])
        ax.figure.colorbar(scalarmap)
        ax.get_legend().remove()
    else: sns.move_legend(ax, 'upper right') # for classification, label box

    ax = plt.gca()
    # Set the border or outline color and width
    border_color = 'black'
    border_width = 0.6  # Adjust this as needed

    # Add a rectangular border around the plot
    for i in ['top', 'right', 'bottom', 'left']: ax.spines[i].set_visible(True)

    for spine in ax.spines.values():
        spine.set_linewidth(border_width); spine.set_color(border_color)
    # move the legend if has that:

    if title == None: title = f'{dim_reduct}_demo'
    plt.title(title); make_path(dim_reduct, False)
    plt.savefig(f'{dim_reduct}/{title}.png', format='png', transparent=True)
    print(f'figure saved at {dim_reduct}/{title}.png')
    plt.show(); plt.close()


def pairwise_similarity(fp_list):
    num = len(fp_list)
    similarities = np.zeros((num, num))
    for i in range(num):
        similarity = DataStructs.BulkTanimotoSimilarity(
            fp_list[i], fp_list[i:])
        # print(type(similarity), len(similarity))
        similarities[i, i:] = similarity
        similarities[i:, i] = similarity
    for i in range(num): assert similarities[i, i] == 1
    return similarities

def plot_tanimoto(df, title=None, savepath=None):
    smiles = df['Drug']
    maccs_list = []
    for smi in smiles:
        mol = Chem.MolFromSmiles(smi)
        maccs = MACCSkeys.GenMACCSKeys(mol)
        maccs_list.append(maccs)
    similarities = pairwise_similarity(maccs_list)
    fig = plt.figure(figsize = (8,8))
    heatmap = sns.heatmap(similarities, cmap='Blues', square=True)
    # Get the color bar axes and adjust its position and size
    cbar = heatmap.collections[0].colorbar
    cbar.ax.set_aspect(20)  # Adjust this value as needed to match the figure size

    # Adjust the color bar's position and size
    cax = plt.gcf().axes[-1]
    cax.set_position([0.78, 0.1, 0.03, 0.8])  # Adjust these values as needed

    if title == None: title = 'Tanimoto Demo'
    plt.title(title, fontsize = 16)
    make_path('Tanimoto', False)
    if savepath == None: savepath = f'Tanimoto/{title}.png'
    print('figure saved at:', savepath)
    plt.savefig(savepath, format='png', transparent=True)
    plt.show(); plt.close()
    
