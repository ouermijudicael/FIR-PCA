# FIR-PCA
Robust location, covariance, and PCA methods for mitigating outlier contamination. The Fast Iterative Robust (FIR) location and covariance approximation, and the FIR-PCA are both in  *FIR_PCA.py*. The file *main.py* illustrates how to use both methods with examples from figures 2 and 5 of the manuscript.

### Setup
```
pip install -r requirements.txt
```

### Usage
```
mu1, C1, H = FIR(Z, alpha=0.75)
```
```
T, m, D, P, sd, od, sd_cutoff, od_cuttoff, H1 = FIR_PCA(X, alpha=0.75)
```

### Manuscript Examples.
The examples in the manuscript obtained by running *run_examples.py* inside the folder *examples* or by running each individual example. Running the example will create and save the results inside a *figures* and *tables*. Note that the examples that are averaged over many sampled datasets require significant time to run. These are based on random sampling; therefore they may produce slightly different results. 

