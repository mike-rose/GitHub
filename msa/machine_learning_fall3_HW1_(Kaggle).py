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
# datFresh = dat.copy()
#==============================================================================
    
def toBinary(s, ones):
    #print(str(s.unique)+' are all cats')
    #print(str(ones)+ ' are ones')
    zeros = [x for x in s.unique() if x not in ones]
    #print(str(zeros)+ ' are zeros')
    for l in ones:
        s = s.replace(l, int(1))
        #print(str(l)+' is now 1')
    for l2 in zeros:
        s = s.replace(l2, int(0))
        #print(str(l2)+' is now 0')
    sr = s
    return sr

#==============================================================================
# dat = datFresh.copy()
#==============================================================================

# %% ================ DUMMY VARS ========================

dum = varList[1:117]

dat2 = pd.get_dummies(dat, dum, '_', columns=dum, drop_first=True)

cols = list(dat2.columns.values)
cols.pop(cols.index('loss'))
dat2 = dat2[cols + ['loss']]

dat2.info()

# %%  ===================== BINARY ====================== 
# A's and B's to 1's and 0's
dat[varList[1:73]] = dat[varList[1:73]].replace('A', int(1))
dat[varList[1:73]] = dat[varList[1:73]].replace('B', int(0))

# Combine categories to create binary variables
to1or0 = [  (74, list('BC')),
            (77, list('ACD')),
            (82, list('ABC')),
            (85, list('CD')),
            (86, list('BD')),
            (87, list('AB')),
            (89, list('A')),
            (90, list('A')),
            (92, list('AH')),
            (93, list('A')),
            (95, list('ACD'))]

for pair in to1or0:
    dat[varList[pair[0]]] = toBinary(dat[varList[pair[0]]], pair[1])

#==============================================================================
# for pair in to1or0:
#     print(dat[varList[pair[0]]].unique())
#==============================================================================

dat.loss.describe()
mu = np.mean(dat.loss)
Q1 = np.percentile(dat.loss, 25)
Q3 = np.percentile(dat.loss, 75)

def ftest(dt):
    pass

def ttest(x, y=dat['loss']):
    lev = x.unique()
    tpvals = {}
    for l in lev:
        tstat, pval = stats.ttest_1samp(y[x==l], mu)
        tpvals[l] = (tstat, pval)
    return tpvals
    
p = ttest(dat['cat94'])

p

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
i = 0
for cvar in varList[73:117]:
    dat.boxplot(column='loss', by=cvar)
    leg = levelFreqs[i].sort_index()
    plt.xlabel(zip(leg.keys(), sorted(leg, reverse=True)))
    plt.plot([Q1]*400, lw=1, c='orange')
    plt.plot([mu]*400, lw=1, c='orange')
    plt.plot([Q3]*400, lw=1, c='orange')
    plt.savefig(path + '/images/charts2/'+ str(cvar) +'boxCount.png')
    plt.close()
    i = i + 1
    
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
#==============================================================================
# newBins = [74, 77, 82, 85, 86, 87, 89, 90, 92, 93, 95]
# newBinVars = [varList[i] for i in newBins]
# 
# numVars = varList[0:73] + newBinVars + varList[117:]
# numDat = dat[numVars]
# 
# pdat = partitionData(numDat)
#==============================================================================
# %% model ready data is named pdat
pdat = partitionData(dat2)


# %% ======= RANDOM FOREST ==============
rfr = RandomForestRegressor()
rfr_m1 = rfr.fit(*Xy(pdat['train']))
test_X, test_y = Xy(pdat['test'])

pred_y = rfr_m1.predict(test_X)
print(score(test_y, pred_y))


# %% ================== SVM =================
# needs major dimension reduction first
#==============================================================================
# pdat = partitionData(dat[varList[117:]])
# svm1 = svm.SVR()
# svm1.fit(*Xy(pdat['train'])) 
# test_X, test_y = Xy(pdat['test'])
# pred_y = svm1.predict(test_X)
# print(score(test_y, pred_y))
#==============================================================================

# %% ================= GRADIENT BOOSTING ===============
params = {'n_estimators': 500, 'max_depth': 4, 'min_samples_split': 2,
          'learning_rate': 0.01, 'loss': 'ls'}
gbr = GradientBoostingRegressor(**params)

gbr_m1 = gbr.fit(*Xy(pdat['train']))
test_X, test_y = Xy(pdat['test'])

pred_y = gbr_m1.predict(test_X)
print(score(test_y, pred_y))

# ============== NEURAL NETWORKS ====================

