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

# %% IMPORT DATA
path = 'C:/Users/mail/analytics/machinelearning/allState/'
dataFile = path + 'raw/train.csv'
dat = pd.read_csv(dataFile)

dataFileTest = path + 'raw/test.csv'
datTest = pd.read_csv(dataFileTest)

# %% ORGANIZE DATA
# list of categorical and continuous variables
cat = dat.columns[1:117]
con = dat.columns[117:132]
testCon = datTest[datTest.columns[117:131]]
# tiny subset for printing
datMini = dat[:20]
train = dat[1:131]
test = datTest[1:131]
# ___ need to partition data eventually here ___

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

# default heatmap of correlations
corr = dat[con].corr()


# seaborn's clustermap of correlations
corr2 = corr.mul(100).astype(int)
sns.clustermap(data=corr2, annot=True, fmt='d', cmap='seismic_r')

# %%
# ============================================
# ============= LOSS EXPLORATION =============
# ============================================
y = dat['loss']
y.plot.hist(bins=1000)


dat[dat.loss>20000] #keep only the ones that are within +3 to -3 standard deviations in the column 'Data'.
df[~(np.abs(df.Data-df.Data.mean())>(3*df.Data.std()))] #or if you prefer the other way around

   
##### OH BOY
import numpy as np
from sklearn import linear_model

clf = linear_model.SGDRegressor()
clf.fit(dat[con], dat.loss)
predictions = clf.predict(testCon)
