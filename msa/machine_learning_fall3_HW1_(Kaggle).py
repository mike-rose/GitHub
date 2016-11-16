# -*- coding: utf-8 -*-
"""
Created on Mon Nov 14 14:00:41 2016

@author: mike_rose

explore allstate kaggle data
"""
# %%
import numpy as np
import pandas as pd
from pandas import Series, DataFrame

# %%
dataFile = r"C:\Users\mail\Desktop\Files\analytics\machine learning" \
    r"\allState_kaggle\train.csv"
dat = pd.read_csv(dataFile)
# id, cat1 - cat116, cont1 - cont14, loss
# dat.info()

dataFileTest = r"C:\Users\mail\Desktop\Files\analytics\machine learning" \
    r"\allState_kaggle\test.csv"
datTest = pd.read_csv(dataFileTest)

# %%
dat.id[:10]
datTest.id[:10]
# some ids are not in training or test sets provided

# %%
dat.describe()
# summary stats for all continuous vars

dat.cat3.describe()
# summary stats of the 'cat3' categorical var

dat.cont1.corr(dat.cont2)
# correlation between 'cont1' and 'cont2' vars

cat = tuple(dat.columns[:117])
con = tuple(dat.columns[[0] + range(117, 132)])
# tuples of categorical and continuous variables
