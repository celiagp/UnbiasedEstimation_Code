# UnbiasedEstimation_Code

#Python code implementing the algorithm presented in "Unbiased likelihood-based estimation of Wright-Fisher diffusion processes" 
#by Celia García-Pareja and Fabio Nobile (https://doi.org/10.48550/arXiv.2303.05390)


For generating discrete observations from Wright-Fisher one-dimensional diffusion paths with haploid natural selection:

-Run the file: DataGen_Estimation.py:
a) Generate discrete observations from a Wright-Fisher diffusion path with the desired parameters. (Saved figure: GeneratedPath.eps)

For estimating the selection parameter h from the generated path:

-Run the file: DataGen_Estimation.py to:

b) Given N (the number of MC samples, 1000 by deafult) generate unbiased estimators of a(x,y,h) and p(x,y,t) 
(results are written on a folder named data with format, e.g.: data/a_{N}_{seed}_{h}_{theta}.csv).
c) Read computed estimators for a(x,y,h) and p(x,y,t) for the N samples (from created files).
d) Compute MLE using Brent's method and estimate (se) using 50 bootstrapped samples.

Files with estimators for D1 and D2 on the numerical examples in https://doi.org/10.48550/arXiv.2303.05390 
are available and stored in: D1: data/a_1000_241224_0.7_[0.02 0.02].csv and  D2: data/a_1000_131002_-0.9_[0.1 0.1].csv

e) Compute MLE and bootstrapped (se) based on smaller N=1, 10, 20, 50, 100, 200, 500 (obtained by sampling from the N=1000 results).

All dependent subroutines are available in the files: Unbiased_est_func.py , ExactAlgorithm_WFb_estimation.py and Auxiliary_EWF.py

In Auxiliary_EWF.py we call the function EWF presented in "EWF: simulating exact paths of the Wright–Fisher diffusion" by Jaromir Sant et al. 
(https://doi.org/10.1093/bioinformatics/btad017), which is fully available at https://github.com/JaroSant/EWF.

The folder ewf contains a cloned version of https://github.com/JaroSant/EWF so that our code can run independently.





