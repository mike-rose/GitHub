"""
@author: mike_rose
explore allstate kaggle data
"""
# %%
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

# %%
path = 'C:/Users/mail/analytics/machinelearning/allState/'
dataFile = path + 'raw/train.csv'
dat = pd.read_csv(dataFile)
# id, cat1 - cat116, cont1 - cont14, loss
# dat.info()

dataFileTest = path + 'raw/test.csv'
datTest = pd.read_csv(dataFileTest)

# tiny subset for printing
datMini = dat[:20]

# %%
x = dat.id[:10]
x = datTest.id[:10]
# some ids are not in training or test sets provided

# %%
dat.describe()
# summary stats for all continuous vars

dat.cat3.describe()
# summary stats of the 'cat3' categorical var

dat.cont1.corr(dat.cont2)
# correlation between 'cont1' and 'cont2' vars

cat = dat.columns[1:117]
con = dat.columns[117:131]
# lists of categorical and continuous variables

# %%
# ============================================
# ========= CATEGORICAL EXPLORATION ==========
# ============================================
cat_SS = dat[cat].describe()
cat_SS.iloc[0].describe()
# every cat# has count=188318 => no missing values

cat_SS.iloc[1].value_counts().plot(kind='bar')
cat_SS.iloc[1].describe()
# 72 cat# are binary

notBinary = cat_SS.iloc[1][cat_SS.iloc[1] > 2].index
datNB = dat[notBinary]
datNB.iloc[1].value_counts().plot(kind='bar')

cat_SSt = cat_SS.transpose()
cat_SSt.sort(['unique', 'freq'])

# this plots a barchart for the one var
dat['cat116'].value_counts().plot(kind='bar')
dat['cat110'].value_counts().plot(kind='bar')
dat['cat109'].value_counts().plot(kind='bar')
dat['cat113'].value_counts().plot(kind='bar')
dat['cat112'].value_counts().plot(kind='bar')
dat['cat115'].value_counts().plot(kind='bar')
dat['cat105'].value_counts().plot(kind='bar')
dat['cat107'].value_counts().plot(kind='bar')
dat['cat114'].value_counts().plot(kind='bar')

cat_SSt.to_csv(path + 'levelCounts.csv', ',')

# %%
# ============================================
# ========== CONTINUOUS EXPLORATION ==========
# ============================================
dat[con].plot(kind='box', title="distributions of continuous variables")
dat[con].hist()

# default heatmap of correlations
corr = dat[con].corr()
corr
plt.matshow(corr)

# seaborn's clustermap of correlations
corr2 = corr.mul(100).astype(int)
sns.clustermap(data=corr2, annot=True, fmt='d', cmap='seismic_r')

# %%
# ============================================
# ============= LOSS EXPLORATION =============
# ============================================
y = dat['loss']
y.plot()
plt.plot(y)
