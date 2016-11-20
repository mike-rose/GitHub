"""
@author: mike_rose
explore allstate kaggle data
"""
# %%
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats
import random
from sklearn import linear_model, tree, svm
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error as mae
# %% IMPORT DATA
path = 'C:/Users/mail/analytics/machinelearning/allState/'
dataFile = path + 'raw/train.csv'
dat = pd.read_csv(dataFile)

#==============================================================================
# dataFileTest = path + 'raw/test.csv'
# datTest = pd.read_csv(dataFileTest)
#==============================================================================

# %% DATA TRANSFORMATION
dat.loss = np.log1p(dat.loss)

varList = list(dat.columns.values)

# A's and B's to 1's and 0's
dat[varList[1:73]] = dat[varList[1:73]].replace('A', 1)
dat[varList[1:73]] = dat[varList[1:73]].replace('B', 0)

 # %% DEFINE FUNCTIONS
def partitionData(dt, n=0, ptrain=0.7):
    if n == 0:
        n = len(dt)
    dt=dt.sample(n)
    train = dt.sample(frac=ptrain)
    test = dt.drop(train.index)
    d = {'train': train, 'test': test}
    return d
  
def Xy(dt):
    X = dt.iloc[:, :-2]
    y = dt.iloc[:, -1]
    return [X, y]

def score(logy, logyhat):
    s = mae(np.expm1(logy), np.expm1(logyhat))
    return s
    
#==============================================================================
# def ftest(dt):
#     pass
# 
# def ttest(dt, col, lev, n=50, rep=1, mu=meanLoss, seed=None):
#     sub1 = dt.loss[col == lev]
#     if len(sub1) <= 50:
#         pass
#     s = dt.sample(n, random_state=seed)
#     t,p = stats.ttest_1samp(dt, mu)
#     return [t,p]
#==============================================================================

# %% =========================================
# ========= CATEGORICAL EXPLORATION ==========
#=============================================
#==============================================================================
# datSort = dat.sort_values('loss')
# dat.loss.hist(bins=12000)
#==============================================================================
#==============================================================================
# for col in twoLetters:
#     srs = list(dat[col])
#     for i in range(len(srs)):
#         if len(srs[i]) == 2:
#             srs[i] = srs[i].lower()
#     dat[col] = srs
#==============================================================================
uniqueLevels = [dat.iloc[:, i].unique() for i in range(73, 117)]
levelFreqs = [dat.iloc[:, i].value_counts() for i in range(73, 117)]

# these are helpful for knowing which levels can be grouped
#==============================================================================
# i = 0
# for cvar in varList[73:117]:
#     dat.boxplot(column='loss', by=cvar)
#     plt.savefig(path + '/images/charts/'+ str(cvar) +'lossbox.png')
#     levelFreqs[i].sort_index().plot(kind='bar')
#     plt.savefig(path + '/images/charts/'+ str(cvar) +'barchart.png')
#     plt.close()
#     i = i + 1
#==============================================================================

# dat.hist(column='loss', by='cat108')

#==============================================================================
# dat.loss.describe()
# meanLoss=np.mean(dat.loss)
#==============================================================================

#==============================================================================
# five80 = dat[dat.loss == 580]
# five80.shape
# five80.describe()  # cat4 = 1 94%
#==============================================================================

# %% =========================================
# ========== CONTINUOUS EXPLORATION ==========
# ============================================

# seaborn's clustermap of correlations
#==============================================================================
# corr = dat[varList[117:]].corr()
# corr2 = corr.mul(100).astype(int)
# sns.clustermap(data=corr2, annot=True, fmt='d', cmap='seismic_r')
#==============================================================================

# %%
# ============================================
# ============================================
# ============================================

numVars = varList[0:73] + varList[117:]
numDat = dat[numVars]
pdat = partitionData(numDat)

# ======= RANDOM FOREST ==============
rfr = RandomForestRegressor()
rfr_m1 = rfr.fit(*Xy(pdat['train']))
test_X, test_y = Xy(pdat['test'])

pred_y = rfr_m1.predict(test_X)
print(score(test_y, pred_y))


# ================== SVM =================
# needs major dimension reduction first
#==============================================================================
# pdat = partitionData(dat[varList[117:]])
# svm1 = svm.SVR()
# svm1.fit(*Xy(pdat['train'])) 
# test_X, test_y = Xy(pdat['test'])
# pred_y = svm1.predict(test_X)
# print(score(test_y, pred_y))
#==============================================================================

# ================= GRADIENT BOOSTING ===============


# ============== NEURAL NETWORKS ====================

