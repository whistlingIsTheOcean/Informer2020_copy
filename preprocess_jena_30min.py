"""Jena Climate: 从原始 10 分钟数据降采样为每半小时 (保留 :00 和 :30)"""
import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).parent / 'data' / 'ETT'
SRC_FILE = DATA_DIR / 'jena_climate_2009_2016.csv'
DST_FILE = DATA_DIR / 'jena_30min.csv'

df = pd.read_csv(SRC_FILE)
print(f'原始: {df.shape[0]} 行, {df.shape[1]} 列')

# 重命名并解析欧式日期
df.rename(columns={'Date Time': 'date'}, inplace=True)
df['date'] = pd.to_datetime(df['date'], dayfirst=True)

# 降采样：保留整点和半点 (:00 和 :30)
df = df[df['date'].dt.minute.isin([0, 30])]
print(f'降采样后 (每30分钟): {df.shape[0]} 行')

# 保存
df.to_csv(DST_FILE, index=False, date_format='%Y-%m-%d %H:%M:%S')
print(f'保存至: {DST_FILE}')
print(f'列名: {list(df.columns)}')
