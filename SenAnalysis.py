#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 30 15:38:39 2020

@author: hiepvu
"""

from bs4 import BeautifulSoup as bs
import os
import pandas as pd
import matplotlib.pyplot as plt
import time
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import finviz as fz
import requests
import selenium


# %matplotlib ggplo
plt.style.use("Solarize_Light2")
infoDf = pd.read_csv("info.csv", index_col = 0)
finalDf = pd.read_csv("final.csv", index_col = 0)
finalDf = finalDf.T
finalDf["sectors"] = [infoDf.loc[x, "sectors"] for x in finalDf.index]
finalDf["industries"] = [infoDf.loc[x, "industries"] for x in finalDf.index]

top5 = finalDf.sort_values(1, ascending = False)[1][:5]
bottom5 = finalDf.sort_values(1, ascending = False)[1][-5:]
sectors = finalDf.groupby("sectors").mean()
industries = finalDf.groupby("industries").mean()





mean = round(finalDf[1].mean(),2)
median = round(finalDf[1].median(),2)
textstr = "mean = " + str(mean) + "\n" + "median = " + str(median)
# these are matplotlib.patch.Patch properties

# place a text box in upper left in axes coords
props = dict(boxstyle='round', facecolor='white', alpha=0.5)

fig, ax = plt.subplots(1,2, figsize = (20,10))
ax[0].hist(finalDf[1], bins = 15)
ax[0].set_title("Distribution of sentiment scores")
ax[0].text(0.80, 0.95, textstr, transform=ax[0].transAxes, 
        bbox = props, fontsize= 10, verticalalignment='top')
ax[1].bar(top5.index.tolist() + bottom5.index.tolist(),top5.tolist()+ bottom5.tolist())
ax[1].set_title("Top 5 and Bottom 5 Sentiment Scores")
plt.show()

fig1, ax1 = plt.subplots(1,2, figsize = (20,10))
ax1[0].bar(x = sectors.index, height = sectors[1])
ax1[0].set_title("Average sentiment scores by sector")
ax1[0].tick_params(axis='x', labelrotation = 90)
ax1[1].bar(x = industries.index, height = industries[1])
ax1[1].set_title("Average sentiment scores by industry")
ax1[1].tick_params(axis='x', labelrotation = 90)
plt.show()


