import yfinance as yf
import numpy as n
import random as r
import pandas as pd

pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 300)
tickerss = open('testtickers.txt', 'r')
tickers = tickerss.readlines()

def ema(closes, leng, N):
    emas = n.empty(leng-50, dtype=n.float64)
    alpha = 2/(N+1)
    cema = closes[0]
    for i in range(1,leng):
        cema = alpha*closes[i] + (1-alpha)*cema
        if i >= 50:
            emas[i-50] = cema
    return emas

def rsi(closes, leng):
    rsis = n.empty(leng-50, dtype=n.float64)
    for i in range(0,leng-50):
        pos = 0
        neg = 0
        for j in range(0,14):
            value = closes[i-j+50]- closes[i-j+49]
            if value < 0:
                neg -= value
            else:
                pos += value
        if neg == 0:
            rsis[i] = 100.0
        else:
            rsis[i] = 100 - (100/(1 + (pos/neg)))
    return rsis



def macd(closes, leng):
    ema12 = ema(closes, leng, 12)
    ema26 = ema(closes, leng, 26)
    macds = ema12 - ema26
    return macds

def atr(highs, lows, closes, leng):
    trs = n.empty(leng, dtype=n.float64)
    trs[0] = highs[0] - lows[0]
    for i in range(1,leng):
        trs[i] = max(highs[i] - lows[i], abs(highs[i] - closes[i-1]), abs(lows[i] - closes[i-1]))
    atrs = ema(trs, leng, 14)
    return atrs


def sign(a):
    if a < 0:
        return -1
    elif a > 0:
        return 1
    else:
        return 0
    

def obv(closes, volumes, leng):
    obvs = n.empty(leng-50, dtype=n.float64)
    obvs[0] = 0.0
    for i in range(1,leng-50):
        obvs[i] = obvs[i-1] + sign(closes[i+50] - closes[i+49])*volumes[i+50]
    return obvs

number = 0
length = len(tickers)
mega = []
try:
    while length != 0:
        tickno = r.randint(0,length-1)
        nticker = tickers[tickno]
        tickers.pop(tickno)
        ticker = nticker[:-1]
        if ticker != "":
            stock = yf.Ticker(ticker)
            data = stock.history(start = '2020-01-01', end = '2025-01-01')
            #data = stock.history(period = "5d")
            if not data.empty:
                data = data.drop(['Dividends', 'Stock Splits'], axis = 1)
                adataut = data.to_numpy() # 'Open', 'High', 'Low', 'Close', 'Volume'
                adata = adataut.transpose()
                lengthd = len(adata[3])
                if lengthd == 1258:
                    final = []
                    final.append(adata[0][50:]) #Open
                    final.append(adata[1][50:]) #High
                    final.append(adata[2][50:]) #Low
                    final.append(adata[3][50:]) #Close
                    final.append(adata[4][50:]) #Volume len = 1208
                    final.append(ema(adata[3], lengthd, 20))
                    final.append(rsi(adata[3], lengthd))
                    final.append(macd(adata[3], lengthd))
                    final.append(atr(adata[1], adata[2], adata[3], lengthd))
                    final.append(obv(adata[3], adata[4], lengthd))
                    afinal = n.array(final)
                    realfinal = afinal.transpose()
                    mega.append(realfinal)
        length -= 1
        #
except KeyboardInterrupt:
    n.save("testdata.npy", mega)
    print("\n" + "done")
finally:
    n.save("testdata.npy", mega)
    print("\n" + "done")
