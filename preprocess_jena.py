"""Jena Climate: 只需重命名日期列，即可被 Dataset_Custom 使用"""
import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).parent / 'data' / 'ETT'
SRC_FILE = DATA_DIR / 'jena_climate_2009_2016.csv'
DST_FILE = DATA_DIR / 'jena_processed.csv'

df = pd.read_csv(SRC_FILE)
print(f'原始: {df.shape[0]} 行, {df.shape[1]} 列')

# 重命名日期列，并用 dayfirst=True 正确解析欧洲格式 (DD.MM.YYYY)
df.rename(columns={'Date Time': 'date'}, inplace=True)
df['date'] = pd.to_datetime(df['date'], dayfirst=True)

# 降采样：只保留每小时整点 (:00:00) 的数据，从42万行→约7000行
df = df[df['date'].dt.minute == 0]
print(f'降采样后: {df.shape[0]} 行 (每小时整点)')

# 检查非数值列
numeric_cols = df.select_dtypes(include='number').columns.tolist()
non_numeric = [c for c in df.columns if c != 'date' and c not in numeric_cols]
if non_numeric:
    print(f'⚠️ 非数值列: {non_numeric}')
else:
    print('✅ 全部为数值列')

# 保存为标准 ISO 格式 (YYYY-MM-DD HH:MM:SS)
df.to_csv(DST_FILE, index=False, date_format='%Y-%m-%d %H:%M:%S')
print(f'保存至: {DST_FILE}')
print(f'列名: {list(df.columns)}')