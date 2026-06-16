"""
消融实验对比绘图 v2 — 智能扫描, 自动区分不同参数组, 支持 Informer/LSTM/ARIMA。
文件名自动包含 d_model/d_ff 参数, 避免不同参数组互相覆盖。

用法:
  python plot_ablation.py --group targets --experiments TdegC TdewdegC rhpct pmbar --model informer
  python plot_ablation.py --scan                   # 自动扫描全部组并补画
"""

import os, sys, re, argparse, numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


# ============================================================
def _scan_informer(pattern, variant):
    """扫描 results/ 目录, 返回 (path, param_id) 或 (None, None)。"""
    if not os.path.isdir('results'):
        return None, None
    best, best_path = 0, None
    for name in os.listdir('results'):
        path = os.path.join('results', name)
        if not os.path.isdir(path) or pattern not in name:
            continue
        # 精确匹配 variant: 它在 des 字段的末尾, 形如 _{pattern}_{variant}_\d+$
        if not re.search(rf'_{pattern}_{re.escape(variant)}_\d+$', name):
            continue
        mp = os.path.join(path, 'metrics.npy')
        if not os.path.exists(mp):
            continue
        ctime = os.path.getmtime(mp)
        if ctime > best:
            best, best_path = ctime, path
    if best_path is None:
        return None, None
    dm = re.search(r'_dm(\d+)_', best_path)
    df = re.search(r'_df(\d+)_', best_path)
    pid = f'dm{dm.group(1)}_df{df.group(1)}' if dm and df else ''
    return best_path, pid


def _scan_other(root, pattern, variant):
    """扫描 LSTM/ARIMA results。"""
    if not os.path.isdir(root):
        return None
    for name in os.listdir(root):
        path = os.path.join(root, name)
        if os.path.isdir(path) and variant in name:
            mp = os.path.join(path, 'pred.npy') if 'LSTM' in root else os.path.join(path, 'metrics.npy')
            if os.path.exists(mp):
                return path
    return None


def _read_informer(path):
    m = np.load(os.path.join(path, 'metrics.npy'))
    losses = None
    ep = path.replace('results', 'checkpoints')
    ep = os.path.join(ep, 'epoch_losses.npy')
    if os.path.exists(ep):
        l = np.load(ep)
        losses = {'train': l[0], 'vali': l[1], 'test': l[2]}
    return {'mae': m[0], 'mse': m[1], 'rmse': m[2], 'losses': losses}


def _read_lstm(path):
    # 优先读 metrics.npy
    mp = os.path.join(path, 'metrics.npy')
    if os.path.exists(mp):
        m = np.load(mp)
        return {'mae': m[3], 'mse': m[4], 'rmse': m[5],
                'mae_std': m[0], 'mse_std': m[1]}
    # 回退: 从 pred/true 自己算 (原始单位)
    p = np.load(os.path.join(path, 'pred.npy'))
    t = np.load(os.path.join(path, 'true.npy'))
    mse = np.mean((p - t) ** 2)
    mae = np.mean(np.abs(p - t))
    return {'mae': mae, 'mse': mse, 'rmse': np.sqrt(mse)}


def _read_arima(path):
    m = np.load(os.path.join(path, 'metrics.npy'))
    return {'mae': m[3], 'mse': m[4], 'rmse': m[5]}


# ============================================================
def plot_one(group_name, param_id, labels, maes, mses, losses_list,
             maes_std=None, mses_std=None):
    n = len(labels)
    if n == 0: return
    suffix = f'_{param_id}' if param_id else ''
    plot_dir = os.path.join('plots', group_name)
    os.makedirs(plot_dir, exist_ok=True)
    colors = ['#2196F3', '#FF9800', '#4CAF50', '#F44336', '#9C27B0', '#00BCD4']

    has_std = (maes_std is not None and mses_std is not None and len(maes_std) == n)
    n_rows = 2 if has_std else 1

    # MSE & MAE 柱状图
    fig, axes = plt.subplots(n_rows, 2, figsize=(12, 4.5 * n_rows))
    if n_rows == 1: axes = [axes]

    # 原始单位行
    ax1, ax2 = axes[0]
    b1 = ax1.bar(labels, mses, color=colors[:n], alpha=0.85)
    ax1.set_ylabel('MSE (original)'); ax1.set_title(f'MSE — {group_name}{suffix}')
    for b, v in zip(b1, mses):
        ax1.text(b.get_x()+b.get_width()/2, b.get_height()+max(mses)*0.02, f'{v:.4f}', ha='center', fontsize=9)
    b2 = ax2.bar(labels, maes, color=colors[:n], alpha=0.85)
    ax2.set_ylabel('MAE (original)'); ax2.set_title(f'MAE — {group_name}{suffix}')
    for b, v in zip(b2, maes):
        ax2.text(b.get_x()+b.get_width()/2, b.get_height()+max(maes)*0.02, f'{v:.4f}', ha='center', fontsize=9)

    # 标准化行 (如有)
    if has_std:
        ax1s, ax2s = axes[1]
        b1s = ax1s.bar(labels, mses_std, color=colors[:n], alpha=0.85)
        ax1s.set_ylabel('MSE (standardized)'); ax1s.set_title(f'标准化 MSE — {group_name}{suffix}')
        for b, v in zip(b1s, mses_std):
            ax1s.text(b.get_x()+b.get_width()/2, b.get_height()+max(mses_std)*0.02, f'{v:.4f}', ha='center', fontsize=9)
        b2s = ax2s.bar(labels, maes_std, color=colors[:n], alpha=0.85)
        ax2s.set_ylabel('MAE (standardized)'); ax2s.set_title(f'标准化 MAE — {group_name}{suffix}')
        for b, v in zip(b2s, maes_std):
            ax2s.text(b.get_x()+b.get_width()/2, b.get_height()+max(maes_std)*0.02, f'{v:.4f}', ha='center', fontsize=9)

    fig.suptitle(f'Ablation: {group_name}{suffix}', fontsize=13)
    fig.tight_layout(rect=[0,0,1,0.93])
    fn = f'metrics_{group_name}{suffix}.png'
    fig.savefig(os.path.join(plot_dir, fn), dpi=150)
    print(f'  [OK] {fn}')

    # Loss 曲线
    valid = [(lb, ls) for lb, ls in zip(labels, losses_list) if ls is not None]
    if valid:
        nv = len(valid)
        nc, nr = min(nv, 2), (nv+1)//2
        fig, axes = plt.subplots(nr, nc, figsize=(6*nc, 4*nr))
        axes = [axes] if nv == 1 else axes.flatten()
        for i, (lb, ls) in enumerate(valid):
            ep = range(1, len(ls['train'])+1)
            axes[i].plot(ep, ls['train'], 'o-', label='Train', color='#2196F3', linewidth=1.5)
            axes[i].plot(ep, ls['vali'], 's--', label='Vali', color='#FF9800', linewidth=1.5)
            axes[i].plot(ep, ls['test'], '^:', label='Test', color='#F44336', linewidth=1.5)
            axes[i].set_xlabel('Epoch'); axes[i].set_ylabel('Loss')
            axes[i].set_title(lb); axes[i].legend(fontsize=8); axes[i].grid(True, alpha=0.3)
        for j in range(nv, len(axes)): axes[j].set_visible(False)
        fig.suptitle(f'Loss — {group_name}{suffix}', fontsize=13)
        fig.tight_layout(rect=[0,0,1,0.93])
        fn = f'losses_{group_name}{suffix}.png'
        fig.savefig(os.path.join(plot_dir, fn), dpi=150)
        print(f'  [OK] {fn}')
    plt.close('all')


# ============================================================
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--group', type=str)
    p.add_argument('--experiments', type=str, nargs='+')
    p.add_argument('--model', type=str, default='informer')
    p.add_argument('--scan', action='store_true', help='自动扫描全部结果补画')
    args = p.parse_args()

    if args.scan:
        # 自动扫描: 收集所有 des 分组
        groups_map = {}
        for name in os.listdir('results'):
            if not os.path.isdir(os.path.join('results', name)): continue
            m = re.search(r'_mxTrue_(\w+)_(\d+)$', name)
            if not m: continue
            des, trial = m.group(1), m.group(2)
            groups_map.setdefault(des, []).append(name)

        # 扫描 LSTM
        lstm_root = os.path.join('LSTM_序列预测', 'results')
        if os.path.isdir(lstm_root):
            for name in os.listdir(lstm_root):
                path = os.path.join(lstm_root, name)
                if os.path.isdir(path):
                    groups_map.setdefault(name, []).append(f'LSTM:{path}')

        # 扫描 ARIMA
        arima_root = os.path.join('ARIMA_序列预测', 'arima.py', 'results')
        if os.path.isdir(arima_root):
            for name in os.listdir(arima_root):
                path = os.path.join(arima_root, name)
                if os.path.isdir(path):
                    groups_map.setdefault(name, []).append(f'ARIMA:{path}')

        print(f'扫描到 {len(groups_map)} 个分组')
        for des in sorted(groups_map):
            plot_if_missing(des)
        return

    if not args.group or not args.experiments:
        print('用法: --group <组名> --experiments <变体1> <变体2> ...')
        return

    labels, maes, mses, losses = [], [], [], []
    maes_std, mses_std = [], []
    for variant in args.experiments:
        path, pid = _scan_informer(args.group, variant)
        if path:
            m = _read_informer(path)
            labels.append(variant); maes.append(m['mae']); mses.append(m['mse'])
            losses.append(m.get('losses'))
            maes_std.append(m['mae']); mses_std.append(m['mse'])  # Informer 的 metrics[0][1] 就是标准化
            continue

        path = _scan_other(os.path.join('LSTM_序列预测', 'results'), args.group, variant)
        if path:
            m = _read_lstm(path)
            labels.append(variant); maes.append(m['mae']); mses.append(m['mse'])
            losses.append(None)
            maes_std.append(m.get('mae_std', m['mae']))
            mses_std.append(m.get('mse_std', m['mse']))
            continue

        path = _scan_other(os.path.join('ARIMA_序列预测', 'arima.py', 'results'), args.group, variant)
        if path:
            m = _read_arima(path)
            labels.append(variant); maes.append(m['mae']); mses.append(m['mse'])
            losses.append(None)
            continue
        print(f'  [!] 未找到: {variant}')

    if labels:
        plot_one(args.group, pid if 'pid' in dir() else '', labels, maes, mses, losses,
                 maes_std if maes_std else None, mses_std if mses_std else None)
    else:
        print('[!] 未找到任何实验结果')


def plot_if_missing(des):
    """检查是否已画过, 没画过就补画。"""
    plot_dir = os.path.join('plots', des.replace('_', '-'))
    if os.path.isdir(plot_dir) and any(f.endswith('.png') for f in os.listdir(plot_dir)):
        return  # 已画过

    # 尝试自动提取变体
    variants = set()
    for name in os.listdir('results'):
        if not os.path.isdir(os.path.join('results', name)): continue
        if des not in name: continue
        # 注意力变体
        m = re.search(r'_at(\w+)_', name)
        if m and m.group(1) not in ('fc', 'eb'):
            variants.add(m.group(1))

    if not variants:
        # 检查 target 相关: _dm128_...targets_TdegC_0 → 提取后半段
        for name in os.listdir('results'):
            if f'_{des}_' in name:
                m = re.search(rf'_{des}_(\w+)_\d+$', name)
                if m:
                    variants.add(m.group(1))

    if len(variants) >= 2:
        variants = sorted(variants)
        print(f'\n自动绘图: {des} ({variants})')
        labels, maes, mses, losses = [], [], [], []
        for v in variants:
            path, pid = _scan_informer(des, v)
            if path:
                m = _read_informer(path)
                labels.append(v); maes.append(m['mae']); mses.append(m['mse'])
                losses.append(m.get('losses'))
    if labels:
            plot_one(des.replace('_', '-'), pid, labels, maes, mses, losses)


if __name__ == '__main__':
    main()
