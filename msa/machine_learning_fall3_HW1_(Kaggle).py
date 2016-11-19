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

# %% =========================================
# ========= CATEGORICAL EXPLORATION ==========
# ============================================

cat_SS = dat[cat].describe()
cat_SS.iloc[0].describe()
# every cat# has count=188318 => no missing values

# check out number of levels per var
#cat_SS.iloc[1].value_counts().plot(kind='bar')
#cat_SS.iloc[1].describe()
# 72 cat# are binary


notBinary = cat_SS.iloc[1][cat_SS.iloc[1] > 2].index
Binary = cat_SS.iloc[1][cat_SS.iloc[1] < 3].index
datNB = dat[notBinary]
datB = dat[Binary]
#datNB.iloc[1].value_counts().plot(kind='bar')

cat_SSt = cat_SS.transpose()
cat_SSt.sort(['unique', 'freq'])



# %% =========================================
# ========== CONTINUOUS EXPLORATION ==========
# ============================================
dat[con].plot(kind='box', title="distributions of continuous variables")
dat[con].hist()

# seaborn's clustermap of correlations
corr2 = corr.mul(100).astype(int)
sns.clustermap(data=corr2, annot=True, fmt='d', cmap='seismic_r')

# %%
# ============================================
# ============= LOSS EXPLORATION =============
# ============================================
y = dat['loss']
y.plot.hist(bins=1000)

#############################
