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
from sklearn import linear_model, tree
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import mean_absolute_error as mae
# %% IMPORT DATA
path = 'C:/Users/mail/analytics/machinelearning/allState/'
dataFile = path + 'raw/train.csv'
dat = pd.read_csv(dataFile)

dataFileTest = path + 'raw/test.csv'
datTest = pd.read_csv(dataFileTest)

# %% ORGANIZE DATA
# how to slice: dat.loc[:,:] or by index: dat.iloc[:,:]

# transform loss to logloss
dat.loss = np.log(dat.loss)
#==============================================================================
# y = dat.loss
# y.describe()
# y.quantile(0.0001)
#==============================================================================

varList = list(dat.columns.values)

#==============================================================================
# datSort = dat.sort_values('loss')
# datSort.loss[:40]
# datSort.loss[-40:]
# datSort.loss[:100].hist()
# datSort.loss[-100:].hist()
#==============================================================================
# cutting off extreme values: 3 < log(loss) < 11

dat = dat[dat['loss'] > 4.3] # chop 31 smallest (loss) obs
dat = dat[dat['loss'] < 10.7] # chop 15 largest (loss) obs

meanLoss=np.mean(dat.loss)

# A's and B's to 1's and 0's
dat[varList[1:73]] = dat[varList[1:73]].replace('A', 1)
dat[varList[1:73]] = dat[varList[1:73]].replace('B', 0)

twoLetters = ['cat109', 'cat110', 'cat112', 'cat113', 'cat116']

ones = [x for x in varList if x not in twoLetters]            
# Lower-case two letter values 
# Put them in order after singltons

for col in twoLetters:
    srs = list(dat[col])
    for i in range(len(srs)):
        if len(srs[i]) == 2:
            srs[i] = srs[i].lower()
    dat[col] = srs
 
#==============================================================================
# %% SUBSET DATA
#==============================================================================
# sub0 = dat.sample(n=100, random_state=8)
# trn0 = sub0.sample(frac=0.7, random_state=8)
# tst0 = sub0.drop(trn0.index)
# 
# sub1 = dat.sample(n=1000, random_state=8)
# trn1 = sub1.sample(frac=0.7, random_state=8)
# tst1 = sub1.drop(trn1.index)
# 
# sub2 = dat.sample(n=10000, random_state=8)
# trn2 = sub2.sample(frac=0.7, random_state=8)
# tst2 = sub2.drop(trn2.index)
#==============================================================================

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
    s = mae(np.exp(logy), np.exp(logyhat))
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
# ============================================

#==============================================================================
# uniqueLevels = [dat.iloc[:, i].unique() for i in range(73, 117)]
# levelFreqs = [dat.iloc[:, i].value_counts() for i in range(73, 117)]
# pd.DataFrame(levelFreqs[:10]).plot(kind='bar')
# pd.DataFrame(levelFreqs[10:20]).plot(kind='bar')
# pd.DataFrame(levelFreqs[20:28]).plot(kind='bar')
# pd.DataFrame(levelFreqs[28:34]).plot(kind='bar')
# pd.DataFrame(levelFreqs[34:38]).plot(kind='bar')
# pd.DataFrame(levelFreqs[38:41]).plot(kind='bar')
# pd.DataFrame(levelFreqs[41:43]).plot(kind='bar')
# pd.DataFrame(levelFreqs[42:44]).plot(kind='bar')
#==============================================================================

# these are helpful for knowing which levels can be grouped
#==============================================================================
# dat.boxplot(column='loss', by='cat108')
# dat.boxplot(column='loss', by='cat116')
# dat.hist(column='loss', by='cat108')
# dat.hist(column='loss', by='cat113')
#==============================================================================

#==============================================================================
# dat.loss.describe()
# lossMean = 7.6857
# lossStd = 0.8094
#==============================================================================

# %% =========================================
# ========== CONTINUOUS EXPLORATION ==========
# ============================================

# seaborn's clustermap of correlations
corr = dat.corr()
corr2 = corr.mul(100).astype(int)
sns.clustermap(data=corr2, annot=True, fmt='d', cmap='seismic_r')

# %%
# ============================================
# ============= TREES ========================
# ============================================
numVars = varList[1:73] + varList[117:]

numDat = dat[numVars]

pdat = partitionData(numDat)

rfr = RandomForestRegressor()
rfr_m1 = rfr.fit(*Xy(pdat['train']))
test_X, test_y = Xy(pdat['test'])

pred_y = rfr_m1.predict(test_X)

score(test_y, pred_y)
