"""
Informer 超参数网格搜索 — 自动遍历 lr / e_layers / d_layers / n_heads 组合。
每轮结束后从 results/ 读取 MSE/MAE, 自动生成热力图和汇总表。

用法:
  python grid_search.py              # 运行全部网格
  python grid_search.py --dry-run    # 仅打印命令
"""

import os
import sys
import subprocess
import argparse
import numpy as np
import matplotlib.pyplot as plt
from itertools import product

plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

PYTHON = sys.executable

# ============================================================
# 固定参数
# ============================================================
FIXED = {
    'data': 'jena',
    'data_path': 'jena_processed.csv',
    'target': 'T (degC)',
    'features': 'MS',
    'seq_len': 96,
    'label_len': 48,
    'pred_len': 24,
    'd_model': 128,
    'd_ff': 256,
    'dropout': 0.3,
    'attn': 'prob',
    'train_epochs': 4,     # 快速搜索, 先不用满 6 epoch
    'batch_size': 32,
    'patience': 3,
    'inverse': True,
    'itr': 1,
    'freq': 'h',
    'des': 'grid_search',
}

# ============================================================
# 搜索网格
# ============================================================
GRID = {
    'learning_rate': [0.0005, 0.0001, 0.00005],
    'e_layers':      [1, 2, 3],
    'd_layers':      [1, 2],
    'n_heads':       [4, 8],       # 必须整除 d_model
}

# 验证 n_heads 合法性
for nh in GRID['n_heads']:
    assert FIXED['d_model'] % nh == 0, f'n_heads={nh} 必须整除 d_model={FIXED["d_model"]}'


def build_command(lr, el, dl, nh):
    cmd = [PYTHON, 'main_informer.py', '--model', 'informer',
           '--data', FIXED['data'], '--data_path', FIXED['data_path'],
           '--target', FIXED['target'], '--features', FIXED['features'],
           '--seq_len', str(FIXED['seq_len']), '--label_len', str(FIXED['label_len']),
           '--pred_len', str(FIXED['pred_len']),
           '--d_model', str(FIXED['d_model']), '--d_ff', str(FIXED['d_ff']),
           '--dropout', str(FIXED['dropout']),
           '--attn', FIXED['attn'],
           '--train_epochs', str(FIXED['train_epochs']),
           '--batch_size', str(FIXED['batch_size']),
           '--patience', str(FIXED['patience']),
           '--itr', str(FIXED['itr']), '--freq', FIXED['freq'],
           '--e_layers', str(el), '--d_layers', str(dl),
           '--n_heads', str(nh),
           '--learning_rate', str(lr),
           '--des', f'grid_lr{lr}_el{el}_dl{dl}_nh{nh}']
    if FIXED['inverse']:
        cmd.append('--inverse')
    return cmd


def find_result(lr, el, dl, nh):
    """在 results/ 中找匹配的网格搜索目录, 读取指标。"""
    pattern = f'grid_lr{lr}_el{el}_dl{dl}_nh{nh}'
    for name in os.listdir('results'):
        path = os.path.join('results', name)
        if not os.path.isdir(path) or pattern not in name:
            continue
        m_path = os.path.join(path, 'metrics.npy')
        if os.path.exists(m_path):
            return np.load(m_path)
    return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dry-run', action='store_true')
    args = p.parse_args()

    lr_vals  = GRID['learning_rate']
    el_vals  = GRID['e_layers']
    dl_vals  = GRID['d_layers']
    nh_vals  = GRID['n_heads']

    total = len(lr_vals) * len(el_vals) * len(dl_vals) * len(nh_vals)
    print(f'网格搜索: {len(lr_vals)}×{len(el_vals)}×{len(dl_vals)}×{len(nh_vals)} = {total} 组\n')

    # 存储结果
    results = {}  # key: (lr, el, dl, nh), value: dict with mae/mse/rmse

    i = 0
    for lr, el, dl, nh in product(lr_vals, el_vals, dl_vals, nh_vals):
        i += 1
        cmd = build_command(lr, el, dl, nh)
        print(f'[{i}/{total}] lr={lr:.0e} e_l={el} d_l={dl} n_head={nh}')
        print(f'  {" ".join(cmd)}')

        if args.dry_run:
            results[(lr, el, dl, nh)] = {'mae': 0, 'mse': 0, 'rmse': 0}
            continue

        try:
            subprocess.run(cmd, cwd=os.path.dirname(os.path.abspath(__file__)))
        except Exception as ex:
            print(f'  ❌ {ex}')
            continue

        # 读取结果
        metrics = find_result(lr, el, dl, nh)
        if metrics is not None:
            results[(lr, el, dl, nh)] = {
                'mae': metrics[0], 'mse': metrics[1], 'rmse': metrics[2]
            }
            print(f'  ✅ MAE={metrics[0]:.4f} MSE={metrics[1]:.4f}')
        else:
            print(f'  ⚠️ 未找到结果文件')

    if args.dry_run or len(results) == 0:
        return

    # ============================================================
    # 可视化: 热力图 (固定 dl=1)
    # ============================================================
    os.makedirs('plots/grid_search', exist_ok=True)

    for dl in dl_vals:
        subset = {(k[0], k[1], k[3]): v['mae']
                  for k, v in results.items() if k[2] == dl}

        n_lr = len(lr_vals)
        n_el = len(el_vals)
        n_nh = len(nh_vals)

        fig, axes = plt.subplots(1, n_nh, figsize=(5 * n_nh, 4))
        if n_nh == 1:
            axes = [axes]

        for ax_idx, nh in enumerate(nh_vals):
            ax = axes[ax_idx]
            heatmap = np.full((n_lr, n_el), np.nan)
            for li, lr in enumerate(lr_vals):
                for ei, el in enumerate(el_vals):
                    val = subset.get((lr, el, nh), np.nan)
                    heatmap[li, ei] = val

            im = ax.imshow(heatmap, aspect='auto', cmap='RdYlGn_r')
            ax.set_xticks(range(n_el))
            ax.set_xticklabels(el_vals)
            ax.set_yticks(range(n_lr))
            ax.set_yticklabels([f'{v:.0e}' for v in lr_vals])
            ax.set_xlabel('e_layers')
            ax.set_ylabel('learning_rate')
            ax.set_title(f'd_layers={dl}, n_heads={nh}')

            for li in range(n_lr):
                for ei in range(n_el):
                    v = heatmap[li, ei]
                    if not np.isnan(v):
                        ax.text(ei, li, f'{v:.4f}', ha='center', va='center',
                                fontsize=8, color='black')

        plt.colorbar(im, ax=axes, label='MAE')
        fig.suptitle('Grid Search — Informer MAE', fontsize=13)
        fig.tight_layout(rect=[0, 0, 1, 0.93])
        fig.savefig(f'plots/grid_search/heatmap_dl{dl}.png', dpi=150)
        print(f'[OK] 热力图: plots/grid_search/heatmap_dl{dl}.png')

    # ============================================================
    # 文本汇总: 输出 Top-5 最佳组合
    # ============================================================
    sorted_results = sorted(results.items(), key=lambda x: x[1]['mae'])
    print(f'\n{"="*60}')
    print('  Top-5 最佳超参数组合 (按 MAE 排序)')
    print(f'{"="*60}')
    print(f'{"排名":>4s}  {"lr":>10s}  {"e_l":>4s}  {"d_l":>4s}  {"n_heads":>7s}  {"MAE":>8s}  {"MSE":>8s}')
    print('-' * 55)
    for rank, ((lr, el, dl, nh), m) in enumerate(sorted_results[:5], 1):
        print(f'{rank:>4d}  {lr:>10.0e}  {el:>4d}  {dl:>4d}  {nh:>7d}  {m["mae"]:>8.4f}  {m["mse"]:>8.4f}')

    # 保存 CSV
    with open('plots/grid_search/results.csv', 'w') as f:
        f.write('learning_rate,e_layers,d_layers,n_heads,mae,mse,rmse\n')
        for (lr, el, dl, nh), m in sorted_results:
            f.write(f'{lr},{el},{dl},{nh},{m["mae"]:.6f},{m["mse"]:.6f},{m["rmse"]:.6f}\n')
    print(f'\n[OK] CSV 汇总: plots/grid_search/results.csv')


if __name__ == '__main__':
    main()
