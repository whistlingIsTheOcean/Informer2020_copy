"""
LSTM 时间序列预测 — 适配 Informer 数据集
支持 MS 模式 (多变量→单变量), 输出原始单位和标准化后的 MSE/MAE

用法:
  python run_lstm.py --data csv --data_path jena_processed.csv --target 'T (degC)' --seq_len 96 --pred_len 24
  python run_lstm.py --data csv --data_path weather_processed.csv --target 'Temperature (C)' --seq_len 96 --pred_len 24
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

# ── 中文支持 ──
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# ── 加入项目根路径以复用 Informer 的 StandardScaler ──
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.tools import StandardScaler

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ============================================================
# 数据集类 (仿 Dataset_Custom 的数据切分逻辑)
# ============================================================
class SlidingWindowDataset(Dataset):
    def __init__(self, root_path, data_path, target, flag='train',
                 seq_len=96, pred_len=24, scale=True):
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.target = target

        # 读取
        df_raw = pd.read_csv(os.path.join(root_path, data_path))
        # 重排序: date + 其他特征 + target
        cols = [c for c in df_raw.columns if c != 'date' and c != target]
        df_raw = df_raw[['date'] + cols + [target]]

        # 切分 train/val/test (与 Informer Dataset_Custom 一致: 0.7/0.2/0.1)
        num_train = int(len(df_raw) * 0.7)
        num_test  = int(len(df_raw) * 0.2)
        num_vali  = len(df_raw) - num_train - num_test
        border1s = [0, num_train - seq_len, len(df_raw) - num_test - seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        type_map = {'train': 0, 'val': 1, 'test': 2}
        set_type = type_map[flag]
        border1 = border1s[set_type]
        border2 = border2s[set_type]

        # 所有非日期列作为特征
        cols_data = df_raw.columns[1:]   # 去掉 date
        df_data = df_raw[cols_data]

        # StandardScaler
        self.scaler = StandardScaler()
        if scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values
        self.data = data[border1:border2]
        self.data_raw = df_data.values[border1:border2]  # 用于 inverse

    def __len__(self):
        return len(self.data) - self.seq_len - self.pred_len + 1

    def __getitem__(self, idx):
        x = self.data[idx : idx + self.seq_len]           # [seq_len, n_feat]
        y = self.data[idx + self.seq_len : idx + self.seq_len + self.pred_len, -1]  # [pred_len] (target 在最后一列)
        return torch.FloatTensor(x), torch.FloatTensor(y)

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


# ============================================================
# LSTM 模型
# ============================================================
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, pred_len, dropout=0.1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.fc = nn.Linear(hidden_size, pred_len)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):                     # x: [B, seq_len, input_size]
        out, (h_n, _) = self.lstm(x)          # out: [B, seq_len, hidden]
        out = self.dropout(h_n[-1])           # 取最后一层 hidden: [B, hidden]
        return self.fc(out)                   # [B, pred_len]


# ============================================================
# 训练 & 测试
# ============================================================
def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        pred = model(x)
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()
        total_loss.append(loss.item())
    return np.mean(total_loss)


def evaluate(model, loader, criterion):
    model.eval()
    total_loss = []
    preds, trues = [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            loss = criterion(pred, y)
            total_loss.append(loss.item())
            preds.append(pred.cpu().numpy())
            trues.append(y.cpu().numpy())
    preds = np.concatenate(preds, axis=0)
    trues = np.concatenate(trues, axis=0)
    return np.mean(total_loss), preds, trues


def compute_metrics(preds, trues, scaler=None):
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
    parser.add_argument('--hidden_size', type=int, default=128)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--patience', type=int, default=5)
    args = parser.parse_args()

    print(f'Device: {device}')
    print(f'Args: {args}')

    # 数据集
    train_set = SlidingWindowDataset(args.root_path, args.data_path, args.target,
                                     'train', args.seq_len, args.pred_len)
    val_set   = SlidingWindowDataset(args.root_path, args.data_path, args.target,
                                     'val',   args.seq_len, args.pred_len)
    test_set  = SlidingWindowDataset(args.root_path, args.data_path, args.target,
                                     'test',  args.seq_len, args.pred_len)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader   = DataLoader(val_set,   batch_size=args.batch_size)
    test_loader  = DataLoader(test_set,  batch_size=args.batch_size)

    input_size = train_set.data.shape[1]
    print(f'input_size={input_size} (多变量特征数), train={len(train_set)}, val={len(val_set)}, test={len(test_set)}')

    # 模型
    model = LSTMModel(input_size, args.hidden_size, args.num_layers,
                      args.pred_len, args.dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    best_val_loss = float('inf')
    best_state = None
    patience_counter = 0
    train_losses, val_losses = [], []

    for epoch in range(args.epochs):
        train_loss = train_epoch(model, train_loader, optimizer, criterion)
        val_loss, _, _ = evaluate(model, val_loader, criterion)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        print(f'Epoch {epoch+1:3d}/{args.epochs} | Train: {train_loss:.6f} | Val: {val_loss:.6f}')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f'Early stopping at epoch {epoch+1}')
                break

    # 加载最佳模型
    model.load_state_dict(best_state)

    # ── 测试 ──
    test_loss, preds, trues = evaluate(model, test_loader, criterion)

    # 标准化空间的指标
    mse_std, mae_std, rmse_std = compute_metrics(preds, trues)
    print(f'\n========== 标准化空间 (StandardScaler) ==========')
    print(f'MSE : {mse_std:.6f}')
    print(f'MAE : {mae_std:.6f}')
    print(f'RMSE: {rmse_std:.6f}')

    # 原始单位的指标 (用 test_set 的 scaler 反变换)
    # 需要构造 [pred_len, 1] 形状的反变换 (scaler 只取 target 列的 mean/std)
    scaler_for_target = StandardScaler()
    scaler_for_target.mean = test_set.scaler.mean[-1:]    # 只取 target 列
    scaler_for_target.std  = test_set.scaler.std[-1:]

    preds_orig = scaler_for_target.inverse_transform(preds)
    trues_orig = scaler_for_target.inverse_transform(trues)

    mse_orig, mae_orig, rmse_orig = compute_metrics(preds_orig, trues_orig)
    print(f'\n========== 原始单位 ({args.target}) ==========')
    print(f'MSE : {mse_orig:.6f}')
    print(f'MAE : {mae_orig:.6f}')
    print(f'RMSE: {rmse_orig:.6f}')

    # ============================================================
    # 可视化
    # ============================================================
    result_dir = os.path.join('LSTM_序列预测', 'results',
                              f'{args.data_path.replace(".csv","")}_{args.target}_s{args.seq_len}_p{args.pred_len}')
    os.makedirs(result_dir, exist_ok=True)

    # 保存 pred/true 和指标 (方便 plot_ablation 读取)
    np.save(os.path.join(result_dir, 'pred.npy'), preds_orig)
    np.save(os.path.join(result_dir, 'true.npy'), trues_orig)
    np.save(os.path.join(result_dir, 'metrics.npy'),
            np.array([mae_std, mse_std, rmse_std, mae_orig, mse_orig, rmse_orig]))

    # 图 1: Loss 曲线
    fig, ax = plt.subplots(figsize=(8, 4))
    epochs_range = range(1, len(train_losses) + 1)
    ax.plot(epochs_range, train_losses, 'o-', label='Train Loss', color='#2196F3', linewidth=2)
    ax.plot(epochs_range, val_losses, 's--', label='Val Loss', color='#FF9800', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('MSE Loss')
    ax.set_title(f'LSTM Training — {args.target}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(result_dir, 'loss_curve.png'), dpi=150)
    print(f'[OK] Loss 曲线: {result_dir}/loss_curve.png')

    # 图 2: 预测 vs 真实 (原始单位)
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
    fig.suptitle(f'LSTM: Prediction vs Ground Truth — {args.target}', fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(os.path.join(result_dir, 'pred_vs_true.png'), dpi=150)
    print(f'[OK] 预测对比图: {result_dir}/pred_vs_true.png')

    plt.show()


if __name__ == '__main__':
    main()
