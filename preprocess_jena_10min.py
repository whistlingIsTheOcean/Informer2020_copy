"""Jena 10分钟原始数据 — 只改日期格式，不降采样"""
import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).parent / 'data' / 'ETT'
SRC = DATA_DIR / 'jena_climate_2009_2016.csv'
DST = DATA_DIR / 'jena_10min.csv'

df = pd.read_csv(SRC)
df.rename(columns={'Date Time': 'date'}, inplace=True)
df['date'] = pd.to_datetime(df['date'], dayfirst=True)
df.to_csv(DST, index=False, date_format='%Y-%m-%d %H:%M:%S')
print(f'{len(df)} 行 → {DST}')
