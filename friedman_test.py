from scipy import stats
import numpy as np


meas_us = [1,1,1,1,1,1,1,1,1,1,1,3,1,3,2,1,1,1,1,1]
meas_lmlp = [5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5]
meas_hmcnf = [2,2,2.5,2,2,2,2,2.5,2,2,2,1,2,1,1,2,2,2,2,2]
meas_hmcnr = [3,3,2.5,3,3,3,3,4,3,3,3,2,3,2,3,3,3,3,3,3]
meas_clusens = [4,4,4,4,4,4,4,2.5,4,4,4,4,4,4,4,4,4,4,4,4]
print(len(meas_us), len(meas_hmcnr), len(meas_hmcnf), len(meas_clusens), len(meas_lmlp))
avranks =  [np.mean(meas_us), np.mean(meas_hmcnf), np.mean(meas_hmcnr), np.mean(meas_clusens)]
print(avranks)
print(stats.friedmanchisquare(meas_us,meas_lmlp,meas_hmcnf,meas_hmcnr, meas_clusens))

import Orange
import matplotlib.pyplot as plt
names = ["CCN($h$)", "HMCN-F", "HMCN-R", "CLUS-ENS", "HMC-LMLP"]
avranks = [np.mean(meas_us), np.mean(meas_hmcnf), np.mean(meas_hmcnr), np.mean(meas_clusens), np.mean(meas_lmlp)]
cd = Orange.evaluation.compute_CD(avranks, 20) #tested on 20 datasets
print(cd)
Orange.evaluation.graph_ranks(avranks, names, cd=cd, width=6, textspace=1.5)
plt.savefig("nemenyi.eps", format='eps')