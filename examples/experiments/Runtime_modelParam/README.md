[runtime code](ADMET_12_26_param_count_runtime.ipynb)

### Model initialization

	in_dim = 256
	hid_dims = [128, 64, 16]
	dropout = 0.1
	lr = 5e-4
	wd = 1e-5
	patience = 5 # stop if loss no decrease after epochs # patience
	batch_size = 128
	# special for AttentiveFP
	graph_feat_size = 300
	n_layers = 5
	num_timesteps = 1 # times of updating the graph representations with GRU
	
	# special for GIN: pretrain model types for selection:
	pre_models_GIN = ['gin_supervised_contextpred', 'gin_supervised_infomax',
	                     'gin_supervised_edgepred', 'gin_supervised_masking']
	model_num = 3 # choose from pre_models for GIN
	dict_scale = None
 

### Model Architecture

#### model type:  MLP
	Classifier(
	  (hidden): ModuleList(
	    (0): Linear(in_features=167, out_features=128, bias=True)
	    (1): Linear(in_features=128, out_features=64, bias=True)
	    (2): Linear(in_features=64, out_features=16, bias=True)
	  )
	  (final): Linear(in_features=16, out_features=5, bias=True)
	  (dropout): Dropout(p=0.1, inplace=False)
	)
 
#### model type:  AttentiveFP
	AttentiveFPPredictor(
	  (gnn): AttentiveFPGNN(
	    (init_context): GetContext(
	      (project_node): Sequential(
	        (0): Linear(in_features=39, out_features=300, bias=True)
	        (1): LeakyReLU(negative_slope=0.01)
	      )
	      (project_edge1): Sequential(
	        (0): Linear(in_features=49, out_features=300, bias=True)
	        (1): LeakyReLU(negative_slope=0.01)
	      )
	      (project_edge2): Sequential(
	        (0): Dropout(p=0.1, inplace=False)
	        (1): Linear(in_features=600, out_features=1, bias=True)
	        (2): LeakyReLU(negative_slope=0.01)
	      )
	      (attentive_gru): AttentiveGRU1(
	        (edge_transform): Sequential(
	          (0): Dropout(p=0.1, inplace=False)
	          (1): Linear(in_features=300, out_features=300, bias=True)
	        )
	        (gru): GRUCell(300, 300)
	      )
	    )
	    (gnn_layers): ModuleList(
	      (0-3): 4 x GNNLayer(
	        (project_edge): Sequential(
	          (0): Dropout(p=0.1, inplace=False)
	          (1): Linear(in_features=600, out_features=1, bias=True)
	          (2): LeakyReLU(negative_slope=0.01)
	        )
	        (attentive_gru): AttentiveGRU2(
	          (project_node): Sequential(
	            (0): Dropout(p=0.1, inplace=False)
	            (1): Linear(in_features=300, out_features=300, bias=True)
	          )
	          (gru): GRUCell(300, 300)
	        )
	      )
	    )
	  )
	  (readout): AttentiveFPReadout(
	    (readouts): ModuleList(
	      (0): GlobalPool(
	        (compute_logits): Sequential(
	          (0): Linear(in_features=600, out_features=1, bias=True)
	          (1): LeakyReLU(negative_slope=0.01)
	        )
	        (project_nodes): Sequential(
	          (0): Dropout(p=0.1, inplace=False)
	          (1): Linear(in_features=300, out_features=300, bias=True)
	        )
	        (gru): GRUCell(300, 300)
	      )
	    )
	  )
	  (predict): Sequential(
	    (0): Dropout(p=0.1, inplace=False)
	    (1): Linear(in_features=300, out_features=5, bias=True)
	  )
	)


#### model type:  GIN
	Downloading gin_supervised_masking_pre_trained.pth from https://data.dgl.ai/dgllife/pre_trained/gin_supervised_masking.pth...
	Pretrained model loaded
	GIN_MOD(
	  (gnn): GIN(
	    (dropout): Dropout(p=0.5, inplace=False)
	    (node_embeddings): ModuleList(
	      (0): Embedding(120, 300)
	      (1): Embedding(3, 300)
	    )
	    (gnn_layers): ModuleList(
	      (0-4): 5 x GINLayer(
	        (mlp): Sequential(
	          (0): Linear(in_features=300, out_features=600, bias=True)
	          (1): ReLU()
	          (2): Linear(in_features=600, out_features=300, bias=True)
	        )
	        (edge_embeddings): ModuleList(
	          (0): Embedding(6, 300)
	          (1): Embedding(3, 300)
	        )
	        (bn): BatchNorm1d(300, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
	      )
	    )
	  )
	  (readout): AvgPooling()
	  (transform): Linear(in_features=300, out_features=256, bias=True)
	  (dropout): Dropout(p=0.1, inplace=False)
	  (hidden): ModuleList(
	    (0): Linear(in_features=256, out_features=128, bias=True)
	    (1): Linear(in_features=128, out_features=64, bias=True)
	    (2): Linear(in_features=64, out_features=16, bias=True)
	  )
	  (final): Linear(in_features=16, out_features=5, bias=True)
	)


### Run time comparison

```
5 tasks maximum on metabolism, runtime (ms) comparison

MLP:
	1: 0.166 $\pm$ 0.005
	2: 0.176 $\pm$ 0.006
	3: 0.181 $\pm$ 0.005
	4: 0.186 $\pm$ 0.005
	5: 0.193 $\pm$ 0.004

Attentive FP:
	1: 0.340 $\pm$ 0.007
	2: 0.347 $\pm$ 0.009
	3: 0.357 $\pm$ 0.012
	4: 0.364 $\pm$ 0.025
	5: 0.365 $\pm$ 0.011

GIN:
    	1: 2.453 $\pm$ 0.041
	2: 2.451 $\pm$ 0.034
	3: 2.460 $\pm$ 0.032
	4: 2.461 $\pm$ 0.030
	5: 2.473 $\pm$ 0.031

```

### Model parameter comparison
```
MLP
	 1: 30817 parameters
	 2: 30834 parameters
	 3: 30851 parameters
	 4: 30868 parameters
	 5: 30885 parameters

AttentiveFP
	 1: 3823507 parameters
	 2: 3823808 parameters
	 3: 3824109 parameters
	 4: 3824410 parameters
	 5: 3824711 parameters

GIN
	 1: 1977165 parameters
	 2: 1977182 parameters
	 3: 1977199 parameters
	 4: 1977216 parameters
	 5: 1977233 parameters
```
