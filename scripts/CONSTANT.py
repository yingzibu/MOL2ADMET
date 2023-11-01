### CONSTANTS ###

cls_metrics = ['acc', 'w_acc', 'prec', 'recall', 'sp', 'f1', 'auc', 'mcc', 'ap']
reg_metrics = ['mae', 'mse', 'rmse', 'r2']

names_reg = ['Caco2_Wang', 'Lipophilicity_AstraZeneca',
         'HydrationFreeEnergy_FreeSolv', 'Solubility_AqSolDB', 'LD50_Zhu'] # regression task
names_cls = ['CYP2C19_Veith', 'CYP2D6_Veith', 'CYP3A4_Veith',
             'CYP1A2_Veith', 'CYP2C9_Veith'] + \
            ['BBB_Martins', 'Bioavailability_Ma',
             'Pgp_Broccatelli', 'HIA_Hou','PAMPA_NCATS'] + \
            ['hERG_Karim', 'AMES']

names_dict = {}
for name in names_reg + names_cls:
    if name in names_reg:   names_dict[name] = True  # regression task
    elif name in names_cls: names_dict[name] = False # classification task
names_all = list(names_dict.keys())

model_types = ['MLP', 'AttentiveFP', 'GIN', 'RNN']
VOCAB_TYPES = ['char', 'smiles', 'selfies']
VOCAB_TYPE = 'smiles' # choose from VOCAB_TYPES