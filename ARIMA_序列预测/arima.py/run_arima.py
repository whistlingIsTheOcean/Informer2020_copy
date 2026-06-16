"""
ARIMA 单变量时间序列预测 — 适配 Informer 数据集
注意: ARIMA 只能做单变量预测，无法使用多变量特征。
     此处仅用目标变量的历史序列，等价于 Informer 的 S 模式。

用法:
  python run_arima.py --data csv --data_path jena_processed.csv --target 'T (degC)' --seq_len 96 --pred_len 24
  python run_arima.py --data csv --data_path weather_processed.csv --target 'Temperature (C)' --seq_len 96 --pred_len 24
"""

import os
import sys
import argparse
import warnings
import numpy as np
import pandas as pd
import pmdarima as pm
import matplotlib.pyplot as plt

# ── 中文支持 ──
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from utils.tools import StandardScaler

warnings.filterwarnings('ignore')


def compute_metrics(preds, trues):
    mse  = np.mean((preds - trues) ** 2)
    mae  = np.mean(np.abs(preds - trues))
    rmse = np.sqrt(mse)
    return mse, mae, rmse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='csv')
    parser.add_argument('--data_path', type=str, default='jena_processed.csv')
    parser.add_argument('--root_path', type=str, default='./data/ETT/')
    parser.add_argument('--target', type=str, default='T (degC)')
    parser.add_argument('--seq_len', type=int, default=96)
    parser.add_argument('--pred_len', type=int, default=24)
    args = parser.parse_args()

    # ── 读取数据 ──────────────────────────────────────────
    df_raw = pd.read_csv(os.path.join(args.root_path, args.data_path))
    series = df_raw[args.target].values.astype(np.float64)
    print(f'数据: {len(series)} 行, target={args.target}')

    # 切分 (与 Informer 一致: 0.7/0.2/0.1)
    n_train = int(len(series) * 0.7)
    n_test  = int(len(series) * 0.2)
    n_vali  = len(series) - n_train - n_test
    train_raw = series[:n_train]
    test_raw  = series[-n_test:]

    # StandardScaler
    scaler = StandardScaler()
    scaler.fit(train_raw.reshape(-1, 1))
    train_scaled_full = scaler.transform(train_raw.reshape(-1, 1)).flatten()
    series_scaled = scaler.transform(series.reshape(-1, 1)).flatten()

    # ── 降采样训练数据 (避免 auto_arima 内存爆炸) ────────
    MAX_TRAIN_SIZE = 5000
    if len(train_scaled_full) > MAX_TRAIN_SIZE:
        step = max(1, len(train_scaled_full) // MAX_TRAIN_SIZE)
        train_scaled = train_scaled_full[::step]
        print(f'\n训练数据自动降采样: {len(train_scaled_full)} → {len(train_scaled)} (每隔{step}步取1点)')
    else:
        train_scaled = train_scaled_full

    # ── ARIMA 模型选择 & 拟合 ─────────────────────────────
    print('\n正在搜索最优 ARIMA 参数 (auto_arima)...')
    model = pm.auto_arima(
        train_scaled, start_p=1, start_q=1, max_p=5, max_q=5, max_d=2,
        seasonal=False, stepwise=True, trace=False,
        scoring='mse',                            # 用 MSE 评分，避开 Kalman 复矩阵分配
        information_criterion='aic', suppress_warnings=True,
        n_jobs=1,
    )
    print(f'最优模型: {model}')
    print(f'AIC: {model.aic():.2f}')

    # 用找到的最优参数在完整训练集上重新拟合
    print(f'在完整训练集 ({len(train_scaled_full)} 点) 上拟合最终模型...')
    final_model = pm.ARIMA(order=model.order, seasonal_order=model.seasonal_order)
    final_model.fit(train_scaled_full)  # statsmodels 内部会自动处理

    # ── 测试: 滑动窗口滚动预测 (用全局拟合模型, 无需重拟合) ──
    preds_all = []
    trues_all = []
    test_start = n_train + n_vali
    test_region_scaled = series_scaled[test_start - args.seq_len:]

    print(f'\n测试集滑动预测中 ({len(test_raw)} 个点, 窗口={args.seq_len}→{args.pred_len})...')
    for i in range(0, len(test_region_scaled) - args.seq_len - args.pred_len + 1, args.pred_len):
        window = test_region_scaled[i : i + args.seq_len]
        try:
            # 用窗口数据快速更新模型 (部分拟合, 比全新 fit 快很多)
            local_model = final_model.update(window, maxiter=10)
            forecast = local_model.predict(n_periods=args.pred_len)
            true_vals = test_region_scaled[i + args.seq_len : i + args.seq_len + args.pred_len]
            preds_all.append(forecast)
            trues_all.append(true_vals)
        except Exception:
            continue

    if len(preds_all) == 0:
        print('窗口预测失败，回退用全局模型...')
        for i in range(0, len(test_region_scaled) - args.seq_len - args.pred_len + 1):
            try:
                forecast = final_model.predict(n_periods=args.pred_len)
                true_vals = test_region_scaled[i + args.seq_len : i + args.seq_len + args.pred_len]
                preds_all.append(forecast)
                trues_all.append(true_vals)
            except Exception:
                continue

    preds = np.array(preds_all)
    trues = np.array(trues_all)
    print(f'预测样本数: {len(preds)}')

    # ── 标准化空间指标 ────────────────────────────────────
    mse_std, mae_std, rmse_std = compute_metrics(preds, trues)
    print(f'\n========== 标准化空间 ==========')
    print(f'MSE : {mse_std:.6f}')
    print(f'MAE : {mae_std:.6f}')
    print(f'RMSE: {rmse_std:.6f}')

    # ── 原始单位指标 ──────────────────────────────────────
    preds_orig = scaler.inverse_transform(preds.reshape(-1, 1)).reshape(preds.shape)
    trues_orig = scaler.inverse_transform(trues.reshape(-1, 1)).reshape(trues.shape)
    mse_orig, mae_orig, rmse_orig = compute_metrics(preds_orig, trues_orig)
    print(f'\n========== 原始单位 ({args.target}) ==========')
    print(f'MSE : {mse_orig:.6f}')
    print(f'MAE : {mae_orig:.6f}')
    print(f'RMSE: {rmse_orig:.6f}')

    print('\n⚠️  注意: ARIMA 是纯单变量模型，不能利用多变量特征。')
    print('    这是与 Informer/LSTM MS 模式的不公平对比 (ARIMA 只看目标自身历史)。')
    print('    如需公平对比，请参考 Informer --features S 模式的结果。')

    # ============================================================
    # 保存 & 可视化
    # ============================================================
    result_dir = os.path.join('ARIMA_序列预测', 'arima.py', 'results',
                              f'{args.data_path.replace(".csv","")}_{args.target}_s{args.seq_len}_p{args.pred_len}')
    os.makedirs(result_dir, exist_ok=True)

    # 保存预测和真实值 (方便实验脚本统一读取)
    np.save(os.path.join(result_dir, 'pred.npy'), preds)
    np.save(os.path.join(result_dir, 'true.npy'), trues)
    np.save(os.path.join(result_dir, 'metrics.npy'),
            np.array([mae_std, mse_std, rmse_std, mae_orig, mse_orig, rmse_orig]))

    # 预测 vs 真实 (原始单位)
    n_plot = min(4, len(preds_orig))
    indices = np.random.choice(len(preds_orig), n_plot, replace=False)
    fig, axes = plt.subplots(2, 2, figsize=(13, 7))
    axes = axes.flatten()
    for i, idx in enumerate(indices):
        steps = range(len(preds_orig[idx]))
        axes[i].plot(steps, trues_orig[idx], 'o-', label='True', color='#4CAF50', linewidth=2)
        axes[i].plot(steps, preds_orig[idx], 'x--', label='Pred', color='#F44336', linewidth=2)
        axes[i].set_xlabel('预测步长')
        axes[i].set_ylabel(args.target)
        axes[i].set_title(f'Sample #{idx}')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
    fig.suptitle(f'ARIMA: Prediction vs Ground Truth — {args.target}', fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(os.path.join(result_dir, 'pred_vs_true.png'), dpi=150)
    print(f'[OK] 预测对比图: {result_dir}/pred_vs_true.png')

    plt.show()


if __name__ == '__main__':
    main()
