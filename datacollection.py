import requests
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import pandas as pd
import time
from datetime import datetime, timedelta

def get_binance_klines(symbol='BTCUSDT', interval='1m', start_time=None, end_time=None, limit=1000):
    url = 'https://api.binance.com/api/v3/klines'
    params = {
        'symbol': symbol,
        'interval': interval,
        'startTime': int(start_time.timestamp() * 1000),
        'endTime': int(end_time.timestamp() * 1000),
        'limit': limit
    }
    response = requests.get(url, params=params)
    data = response.json()
    
    df = pd.DataFrame(data, columns=[
        'Open time', 'Open', 'High', 'Low', 'Close', 'Volume',
        'Close time', 'Quote asset volume', 'Number of trades',
        'Taker buy base volume', 'Taker buy quote volume', 'Ignore'
    ])
    
    df['Open time'] = pd.to_datetime(df['Open time'], unit='ms')
    df['Close time'] = pd.to_datetime(df['Close time'], unit='ms')
    df = df[['Open time', 'Open', 'High', 'Low', 'Close', 'Volume']]
    return df

# Crawl từ ngày 1/1/2019 đến 10/4/2025
start = datetime(2019, 1, 1)
end = datetime(2025, 4, 10)

df_all = pd.DataFrame()

while start < end:
    try:
        temp_df = get_binance_klines(
            start_time=start,
            end_time=start + timedelta(minutes=999),  # 1000 điểm tối đa
        )
        if temp_df.empty:
            break
        df_all = pd.concat([df_all, temp_df], ignore_index=True)
        print(f"Lấy dữ liệu từ {start} -> {temp_df['Open time'].iloc[-1]}")
        start = temp_df['Open time'].iloc[-1] + timedelta(minutes=1)
        time.sleep(0.5)  # tránh bị rate limit
    except Exception as e:
        print("Lỗi:", e)
        break

# Lưu ra file CSV
df_all.to_csv(".Data/CrawlBitCoin.csv", index=False)
# print("Hoàn tất! File lưu tại CrawBitCoin.csv")
