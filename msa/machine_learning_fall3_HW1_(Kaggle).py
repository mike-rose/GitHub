# -*- coding: utf-8 -*-
"""
Created on Mon Nov 14 14:00:41 2016

@author: mike_rose

explore allstate kaggle data
"""

import numpy as np
import pandas as pd
from pandas import Series, DataFrame

dataFile = r"C:\Users\mail\Desktop\Files\analytics\machine learning" \
    r"\allState_kaggle\train.csv"
dat = pd.read_csv(dataFile)
# id, cat1 - cat116, cont1 - cont14, loss

dataFileTest = r"C:\Users\mail\Desktop\Files\analytics\machine learning" \
    r"\allState_kaggle\test.csv"
datTest = pd.read_csv(dataFile)

len(dat)
type(dat)

dat.head(2)

with open(dataFile, "rb") as f:
    temp = f.readlines()
temp[0]

with open(dataFile, "rb") as f:
    temp = f.readlines()
temp[0]

for i in range(20):
    print(temp[i][:10])

