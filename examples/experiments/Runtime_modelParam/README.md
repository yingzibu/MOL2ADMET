### Run time comparison

[runtime code](ADMET_10_24_runtime.ipynb)
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
