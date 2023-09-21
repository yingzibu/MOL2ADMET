### Leaderboard


https://tdcommons.ai/benchmark/admet_group/overview/

Single Objective (SO) MLP (MACCS Fingerprint):  

https://colab.research.google.com/drive/1V-LOuUzMYAAuzwtgCEHJXLXNuG9MUtFZ#scrollTo=MPQ0uReno9wI&uniqifier=1


    dims = [167, [128, 64, 32, 16], 1]
    ----------------------------------------------------------------
            Layer (type)               Output Shape         Param #
    ================================================================
                Linear-1              [-1, 64, 128]          21,504
                Linear-2               [-1, 64, 64]           8,256
                Linear-3               [-1, 64, 32]           2,080
                Linear-4               [-1, 64, 16]             528
                Linear-5                [-1, 64, 1]              17
    ================================================================
    Total params: 32,385
    Trainable params: 32,385
    Non-trainable params: 0
    ----------------------------------------------------------------
    Input size (MB): 0.04
    Forward/backward pass size (MB): 0.12
    Params size (MB): 0.12
    Estimated Total Size (MB): 0.28
    ----------------------------------------------------------------
    [{'caco2_wang': [0.772, 0.506]},
     {'hia_hou': [0.975, 0.002]},
     {'pgp_broccatelli': [0.892, 0.014]},
     {'bioavailability_ma': [0.551, 0.024]},
     {'lipophilicity_astrazeneca': [0.725, 0.012]},
     {'solubility_aqsoldb': [1.1, 0.012]},
     {'bbb_martins': [0.856, 0.002]},
     {'cyp2d6_veith': [0.627, 0.008]},
     {'cyp3a4_veith': [0.801, 0.003]},
     {'cyp2c9_veith': [0.672, 0.006]}]
     {'ames': [0.783, 0.011]}
     {'ld50_zhu': [0.665, 0.019]},
     {'herg': [0.778, 0.016]}]
