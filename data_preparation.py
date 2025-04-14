import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas_ta as ta
import pandas as pd
from eval import evaluate

def load_data(threshold=0.5):
    df  = pd.read_csv('Data/CrawlBitCoin.csv')
    cp_df=df.copy(deep=True)
    cp_df.set_index(pd.to_datetime(cp_df['Date Time']),inplace=True)
    cp_df.dropna(inplace=True)
    cp_df = cp_df[['Low','High','Close']]

    cp_df=cp_df.resample('15min').agg({
        'Low': 'min',
        'High': 'max',
        'Close': 'last'
    }).dropna()[['Low','High','Close']]
    cp_df['Target']=cp_df.Close.shift(-1)-cp_df.Close
    cp_df['Target']=np.where(cp_df['Target']>threshold,1,0)
    return cp_df.dropna()

def create_dataset_for_train_val_test(threshold=0.5):

    new_df= load_data(threshold)
    rsi_feature=ta.rsi(new_df.Close)
    atr_feature=ta.atr(new_df.High,new_df.Low,new_df.Close)
    adx_feature=ta.adx(new_df.High,new_df.Low,new_df.Close)
    sma_feature=ta.sma(new_df.Close)
    skew_feature=ta.skew(new_df.Close)
    slope_feature=ta.slope(new_df.Close)
    bband_feature=ta.bbands(new_df.Close).iloc[:,[0,2]]
    macd_feature=ta.macd(new_df.Close).iloc[:,[0,2]]

    new_df=pd.concat([new_df.iloc[:,:-1],sma_feature,rsi_feature,atr_feature,adx_feature,skew_feature,slope_feature,bband_feature,macd_feature,new_df.iloc[:,-1]],axis=1)
    new_df=new_df.dropna()

    max_date = new_df.index.max()

    test_start = max_date - pd.DateOffset(months=2)

    train_df = new_df[new_df.index < test_start]
    test_df = new_df[new_df.index >= test_start]

    val_start = test_start - pd.DateOffset(months=3)

    val_df = train_df[train_df.index >= val_start]
    train_df = train_df[train_df.index < val_start]

    return train_df.dropna(), val_df.dropna(), test_df.dropna()

    
