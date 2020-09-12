#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  7 06:31:04 2020

@author: hiepvu
"""

from bs4 import BeautifulSoup as bs
import os
import pandas as pd
import matplotlib.pyplot as plt
import time
import requests
from selenium import webdriver
import time as tm
import os
from datetime import datetime
from datetime import timedelta
import string
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import re
import string
import nltk
from nltk.corpus import wordnet
import spacy
from nltk.corpus import stopwords
import numpy as np
import spacy
from nltk import word_tokenize
from nltk import WordNetLemmatizer


site = "https://www.slickcharts.com/sp500"
chromePath = os.getcwd() + "/chromedriver"
browser = webdriver.Chrome(chromePath)
browser.get(site)
tm.sleep(5)
html = browser.page_source
soup = bs(html, "html.parser")
print(soup.prettify())
trs = soup.find("tbody")
print(trs.prettify())

names = []
tickers = []
m = trs.find_all("a")
m[2].get_text()
for i in range(1,206,2):
    ticker = m[i].get_text()
    tickers = tickers + [ticker]

for i in range(0,206,2):
    name = m[i].get_text()
    names = names + [name]

sectors = []
industries = []
for ticker in tickers:
    url = "https://finance.yahoo.com/quote/" + ticker + "/profile?p=" + ticker
    page1 = requests.get(url)
    soup1 = bs(page1.content, "html.parser")
    try:
        main = soup1.find_all("span", class_ = "Fw(600)")
        sector = main[0].get_text()
        sectors = sectors + [sector]
        industry = main[1].get_text()
        industries = industries + [industry]
    except:
        industries = industries + ["Others"]
        sectors = sectors + ["Others"]
        pass
    
info = pd.DataFrame({"tickers": tickers, "sectors": sectors, "industries": industries, "names": names}, index = tickers) 
info.to_csv("info.csv")




tickers = pd.read_csv("tickers.csv", index_col = 0)["ticker"]
infoDf = pd.read_csv("info.csv", index_col = 0)


spacy.prefer_gpu()
nlp = spacy.load("en_core_web_sm")
nltk.download('stopwords')
nltk.download('punkt')
stops = stopwords.words("english")





analyzer = SentimentIntensityAnalyzer()
def sentiment_analyzer_scores(text):
    score = analyzer.polarity_scores(text)
    return score


stocks = ["MSFT", "TSLA", "AAPL", "NKLA", "AMZN", "BA"]
totalDf = pd.DataFrame()
start = datetime.today()
start = start.replace(hour = 0, minute = 0, second = 0, microsecond = 0)
timeIndex = [start]

for i in range(365*2-1):
    timeIndex = timeIndex + [start - timedelta(i+1)]
totalDf["index"] = timeIndex
totalDf.index = totalDf["index"]
allTexts = []
allTimes = []
allStocks = []

for stock in tickers:
    site = "https://finviz.com/quote.ashx?t=" + stock.lower()
    browser.get(site)
    tm.sleep(5)
    html = browser.page_source
    soup = bs(html)
    main = soup.find("table", id = "news-table")
    try:
        tbody = main.find("tbody")
        children = main.select("tbody > tr")
        
        times = []
        texts = []
        for child in children:
            tds = child.find_all("td")
            time = tds[0].get_text().strip()
            if time[0].isalpha() == True:
                time = datetime.strptime(time, "%b-%d-%y %H:%M%p")
                time = time.replace(hour = 0, minute = 0, second = 0, microsecond = 0)
            else:
                time = times[-1]
            times = times + [time]
            texts = texts + [tds[1].get_text().strip()]
        allTexts = allTexts + [texts]
        allTimes = allTimes + [times]
        allStocks = allStocks + [stock]
        

    
    # """One needs to download english model package manually"""
    
    # """https://www.machinelearningplus.com/nlp/lemmatization-examples-python/"""
        textsF = []
        for text in texts:
            doc = nlp(text)
            textF = [token.lemma_ for token in doc]
            textF = " ".join([token for token in textF if token not in string.punctuation and token not in stops])    

            textsF = textsF + [textF]
        
        scores = []
        for text in textsF:
            score = sentiment_analyzer_scores(text)["compound"]
            scores = scores + [score]
            
        scoreDf = pd.DataFrame({"times":times, stock :scores})
        x = scoreDf.groupby("times").mean()
        
        totalDf[stock] = x[stock]
    except:
        continue
    
    tm.sleep(5)

totalDf = totalDf.drop("index", axis = 1)
totalDf = totalDf.fillna(0)

writer  = pd.ExcelWriter("allTexts.xlsx",engine='xlsxwriter')
for i in range(len(allTexts)):
    dummyDf = pd.DataFrame({allStocks[i]:allTexts[i], "times": allTimes[i]})
    dummyDf.to_excel(writer, sheet_name = allStocks[i])

writer.save()

weight = [1*0.95**x for x in range(365*2)]
# totalDf = totalDf.drop("weight", axis = 1)
dicts = {}
for col in totalDf.columns:
    score = np.dot(totalDf[col],weight)
    dicts[col] = score






"""Sentiment analysi
https://www.presentslide.in/2019/08/sentiment-analysis-textblob-library.html
"""

df = pd.DataFrame(dicts, index = [1])
df.to_csv("final.csv")


