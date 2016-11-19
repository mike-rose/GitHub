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
from sklearn.ensemble import RandomForestClassifier

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

#==============================================================================
# varlst = list(dat.columns.values)
# cat_list = range(1, 117)  # exclude ID
# con_list = range(117, 131)  # exclude loss
# bnry_list = range(1, 73)
#==============================================================================

# %% SUBSET DATA
sub0 = dat.sample(n=100, random_state=8)
trn0 = sub0.sample(frac=0.7, random_state=8)
tst0 = sub0.drop(trn0.index)

sub1 = dat.sample(n=1000, random_state=8)
trn1 = sub1.sample(frac=0.7, random_state=8)
tst1 = sub1.drop(trn1.index)

sub2 = dat.sample(n=10000, random_state=8)
trn2 = sub2.sample(frac=0.7, random_state=8)
tst2 = sub2.drop(trn2.index)

# %% =========================================
# ========= CATEGORICAL EXPLORATION ==========
# ============================================

# these are helpful for knowing which levels can be grouped
dat.boxplot(column='loss', by='cat108')
dat.boxplot(column='loss', by='cat110')
dat.hist(column='loss', by='cat108')
dat.hist(column='loss', by='cat113')

dat.cat113.describe()
dat.cat113.value_counts().plot()
dat.cat116.value_counts()[100:-200]
dat.cat116.value_counts().plot()
# looking like I categories with fewer than 200 observations \
# should be grouped or dropped - for all cat vars
dat.loc[:,'cat72':'cat116'].value_counts().plot()
nb = varlst[73:117]  # non-binary cat var names

# ONE WAY ANOVA -- CHECK FOR DIFFERENCES ACROSS LEVELS OF EACH CATEGORY VAR







for v in nb[:1]:
    # do stuff with v cat var
    categories = dat.loc[:, v].unique()
    listObsPerLevel = []
    for cat in categories:
        listObsPerLevel.append(dat.loss[dat.loc[:, v] == cat])
        levelCounts = dat.loc[:, v].value_counts()
        cat1 = levelCounts[levelCounts < 200]
#    cat2 = t[t >= 200]
    print(v+' '+str(len(t))+' levels:' + cat1.to_string())
    print(cat2.to_string())

stats.f_oneway(*listObsPerLevel)

dat.loss.describe()
lossMean = 7.6857
lossStd = 0.8094

stats.ttest_1samp(listObsPerLevel[], lossMean)


#take 100 random samples of size whatever, use their means in ttest


# %% =========================================
# ========== CONTINUOUS EXPLORATION ==========
# ============================================
dat[con].plot(kind='box', title="distributions of continuous variables")
dat[con].hist()

# seaborn's clustermap of correlations
corr = dat.corr()
corr2 = corr.mul(100).astype(int)
sns.clustermap(data=corr2, annot=True, fmt='d', cmap='seismic_r')

# %%
# ============================================
# ============= LOSS EXPLORATION =============
# ============================================
y = dat['loss']
y.plot.hist(bins=1000)
ly = np.log(y)
ly.hist(bins=1000)
backAgain = np.exp(ly)
backAgain == y
backAgain.plot.hist(bins=1000)
diff = y - backAgain
diff
# dat[dat.loss>20000] # boolean subsetting

#clf = linear_model.SGDRegressor()
#clf.fit(dat[con], dat.loss)
#predictions = clf.predict(testCon)
#len(testCon)
#testCon.shape
#clf.shape
#clf._get_learning_rate_type
#clf.predict(testCon)

#############################

# %% =========================================
# ========= 
# ============================================
tree0 = tree.DecisionTreeRegressor()
tree0 = tree0.fit(trn0, trn0.loss)
pred = tree0.predict(tst0)

