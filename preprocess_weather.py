"""
weatherHistory.csv 预处理脚本
将包含文本列的原始天气数据转为全数值型，适配 Informer 的 Dataset_Custom。

处理内容:
  1. 重命名 'Formatted Date' → 'date'
  2. Summary 列: 8 个主要类别保留, 其余归为 'Other' → 独热编码 (9 列)
  3. Precip Type 列: "rain"/"snow"/"null" → 独热编码 (3 列)
  4. 丢弃 Daily Summary, Loud Cover
  5. 保存为 weather_processed.csv
"""

import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).parent / 'data' / 'ETT'
SRC_FILE = DATA_DIR / 'weatherHistory.csv'
DST_FILE = DATA_DIR / 'weather_processed.csv'

# ── 1. 读取 ────────────────────────────────────────────────
df = pd.read_csv(SRC_FILE)
print(f'原始数据: {df.shape[0]} 行, {df.shape[1]} 列')

# ── 2. 重命名日期列 ────────────────────────────────────────
df.rename(columns={'Formatted Date': 'date'}, inplace=True)

# ── 2.1 日期转 UTC 并去掉时区 (避免 time_features 中 pd.to_datetime 报错) ──
df['date'] = pd.to_datetime(df['date'], utc=True).dt.tz_localize(None)

# ── 3. 处理 Summary 列 ─────────────────────────────────────
MAIN_CATEGORIES = {
    'Partly Cloudy',
    'Mostly Cloudy',
    'Overcast',
    'Clear',
    'Foggy',
    'Breezy and Overcast',
    'Breezy and Mostly Cloudy',
    'Breezy and Partly Cloudy',
}

# 不在 MAIN_CATEGORIES 中的统一归为 Other
df['Summary'] = df['Summary'].map(
    lambda x: x if x in MAIN_CATEGORIES else 'Other'
)

# 打印归类后的分布
print('\nSummary 归类后分布:')
print(df['Summary'].value_counts())

summary_dummies = pd.get_dummies(df['Summary'], prefix='Summary')
df = pd.concat([df, summary_dummies], axis=1)

# ── 4. 处理 Precip Type 列 ─────────────────────────────────
# "null" 是字符串, 但 pandas.read_csv 会把它当成 NaN 吞掉
# 先检查 NaN, 把 NaN 填回字符串 "null", 再独热编码
null_count = df['Precip Type'].isna().sum()
if null_count > 0:
    print(f'\nPrecip Type 列发现 {null_count} 个 NaN (原字符串 "null"), 正在填充...')
    df['Precip Type'] = df['Precip Type'].fillna('null')

precip_dummies = pd.get_dummies(df['Precip Type'], prefix='PrecipType')
df = pd.concat([df, precip_dummies], axis=1)

print('\nPrecip Type 分布:')
print(df['Precip Type'].value_counts())

# ── 5. 丢弃无用列 ──────────────────────────────────────────
df.drop(columns=['Summary', 'Precip Type', 'Daily Summary', 'Loud Cover'], inplace=True)

# ── 6. bool 转 int (避免 StandardScaler 在 object 数组上 std() 崩溃) ──
bool_cols = df.select_dtypes(include='bool').columns
if len(bool_cols) > 0:
    print(f'\n正在将 {len(bool_cols)} 个 bool 列转为 int...')
    df[bool_cols] = df[bool_cols].astype(int)
    
# ── 7. 确保 target 列在最后 ────────────────────────────────
target = 'Temperature (C)'
cols = [c for c in df.columns if c != 'date' and c != target]
cols = ['date'] + cols + [target]
df = df[cols]

# ── 8. 保存 ────────────────────────────────────────────────
df.to_csv(DST_FILE, index=False)
print(f'\n处理完成! 保存至: {DST_FILE}')
print(f'处理后数据: {df.shape[0]} 行, {df.shape[1]} 列')
print(f'列名: {list(df.columns)}')
