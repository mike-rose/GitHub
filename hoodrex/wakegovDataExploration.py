# -*- coding: utf-8 -*-
"""
Created on Tue Nov 15 09:13:03 2016

@author: mike_rose
"""

# Full Stack Housing Data Test

pathToData = "C:/Users/mail/Desktop/Files/Good/wakegov/"
with open(pathToData + "all.txt", "rb") as f:
    wakeDat = f.readlines()

print(wakeDat[0])

print(wakeDat[0][274:295])
