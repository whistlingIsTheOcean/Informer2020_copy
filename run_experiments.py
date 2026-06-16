"""
自动实验运行器 — 每组消融完成后自动调用 plot_ablation.py 生成对比图。

用法:
  python run_experiments.py              # 运行全部 + 自动绘图
  python run_experiments.py --dry-run    # 仅打印命令
  python run_experiments.py --group attn # 仅运行指定组
"""

import os
import sys
import subprocess
import argparse

PYTHON = sys.executable

# ============================================================
# 实验组定义
# ============================================================

GROUPS = []

def add_group(name, model, base_params, sweep_key, sweep_values, sweep_keys_extra=None):
    experiments = []
    for i, val in enumerate(sweep_values):
        exp = base_params.copy()
        exp[sweep_key] = val
        if sweep_keys_extra:
            for ek, ev_list in sweep_keys_extra.items():
                exp[ek] = ev_list[i]
        experiments.append(exp)
    GROUPS.append({
        'name': name, 'model': model,
        'sweep_key': sweep_key,
        'sweep_values': [str(v) for v in sweep_values],
        'experiments': experiments,
    })

# # ── 组 1: 注意力变体 ──
# add_group('ablation_attn', 'informer',
#     {'data': 'jena', 'data_path': 'jena_processed.csv',
#      'target': 'T (degC)', 'features': 'MS', 'seq_len': 96, 'label_len': 48,
#      'pred_len': 24, 'd_model': 128, 'd_ff': 256, 'dropout': 0.3,
#      'train_epochs': 4, 'batch_size': 16, 'inverse': False, 'itr': 1, 'freq': 'h',
#      },
#     'attn', ['prob', 'full', 'logsparse'])

# # ── 组 2: 不同预测目标 ──
# add_group('targets', 'informer',
#     {'data': 'jena', 'data_path': 'jena_processed.csv',
#      'features': 'MS', 'seq_len': 96, 'label_len': 48, 'pred_len': 24,
#      'd_model': 128, 'd_ff': 256, 'dropout': 0.3, 'attn': 'prob',
#      'train_epochs': 4, 'batch_size': 16, 'inverse': False, 'itr': 1, 'freq': 'h'},
#     'target', ['T (degC)', 'Tdew (degC)', 'rh (%)', 'p (mbar)'])

# # ── 组 3: LSTM 隐层大小 ──
# add_group('lstm_hidden', 'lstm',
#     {'data_path': 'jena_processed.csv', 'target': 'T (degC)',
#      'seq_len': 96, 'pred_len': 24, 'num_layers': 2, 'dropout': 0.3,
#      'epochs': 15, 'lr': 0.005, 'patience': 3},
#     'hidden_size', [32, 64, 128])

# # ── 组 4: ARIMA 不同目标 ──
# add_group('arima_targets', 'arima',
#     {'data_path': 'jena_processed.csv', 'seq_len': 96, 'pred_len': 24},
#     'target', ['T (degC)', 'p (mbar)'])

# # ── 组 5: 不同 seq_len ──
# add_group('ablation_seq', 'informer',
#     {'data': 'jena', 'data_path': 'jena_processed.csv',
#      'target': 'T (degC)', 'features': 'MS', 'pred_len': 24,
#      'd_model': 128, 'd_ff': 256, 'dropout': 0.3, 'attn': 'prob',
#      'train_epochs': 4, 'batch_size': 16, 'inverse': True, 'itr': 1, 'freq': 'h'},
#     'seq_len', [48, 96, 192],
#     {'label_len': [24, 48, 96]})
# ── 组6: TCN vs 无 TCN ──
# add_group('ablation_tcn', 'informer',
#     {'data': 'jena', 'data_path': 'jena_processed.csv',
#      'target': 'T (degC)', 'features': 'MS', 'seq_len': 96, 'label_len': 48,
#      'pred_len': 24, 'd_model': 128, 'd_ff': 256, 'dropout': 0.3, 'attn': 'prob',
#      'train_epochs': 4, 'batch_size': 16, 'inverse': True, 'itr': 1, 'freq': 'h',
#      'use_amp': False},
#     'use_tcn', [True,False])
# # ── 组6.5: TCN vs 无 TCN ──
# add_group('log_tcn', 'informer',
#     {'data': 'jena', 'data_path': 'jena_processed.csv',
#      'target': 'T (degC)', 'features': 'MS', 'seq_len': 96, 'label_len': 48,
#      'pred_len': 24, 'd_model': 128, 'd_ff': 256, 'dropout': 0.08, 'attn': 'logsparse',
#      'train_epochs': 4, 'batch_size': 16, 'inverse': False, 'itr': 1, 'freq': 'h',
#      },
#     'use_tcn', [True,False])
# # ── 组7: RoPE vs 无 RoPE ──
# add_group('new_rope_168', 'informer',
#     {'data': 'jena', 'data_path': 'jena_processed.csv',
#      'target': 'T (degC)', 'features': 'MS', 'seq_len': 168, 'label_len': 48,
#      'pred_len': 24, 'd_model': 128, 'd_ff': 256, 'dropout': 0.08, 'attn': 'prob',
#      'train_epochs': 4, 'batch_size': 16, 'inverse': False, 'itr': 1, 'freq': 'h',
#      'use_amp': False},
#     'use_rope', [True,False])
# # 组8
# add_group('new_rope_168', 'informer',
#     {'data': 'jena', 'data_path': 'jena_processed.csv',
#      'target': 'T (degC)', 'features': 'MS', 'seq_len': 168, 'label_len': 48,
#      'pred_len': 24, 'd_model': 128, 'd_ff': 256, 'dropout': 0.08, 'attn': 'prob',
#      'train_epochs': 4, 'batch_size': 16, 'inverse': False, 'itr': 1, 'freq': 'h',
#      'use_amp': False},
#     'use_rope', [True,False])


def add_group_combo(name, model, base_params, sweep_key, combos):
    """combos: list of dicts, 每个 dict 指定 sweep_key 和其余变化参数的值"""
    experiments = []
    sweep_vals = []
    for c in combos:
        exp = base_params.copy()
        exp.update(c)
        experiments.append(exp)
        sweep_vals.append(str(c[sweep_key]))
    GROUPS.append({
        'name': name, 'model': model,
        'sweep_key': sweep_key,
        'sweep_values': sweep_vals,
        'experiments': experiments,
    })


# ── 组9: 多组序列长度对比 ──
add_group_combo('jena_seq', 'informer',
    {'data': 'jena', 'data_path': 'jena_processed.csv',
     'target': 'T (degC)', 'features': 'MS',
     'd_model': 128, 'd_ff': 256, 'dropout': 0.1, 'attn': 'prob',
     'train_epochs': 3, 'batch_size': 16, 'inverse': False, 'itr': 1, 'freq': 't'},
    'seq_len', [
        {'seq_len': 576, 'label_len': 288, 'pred_len': 96},
        {'seq_len': 288, 'label_len': 144, 'pred_len': 96},
        {'seq_len': 288, 'label_len': 144, 'pred_len': 24},
        {'seq_len': 288, 'label_len':  48, 'pred_len': 24},
        {'seq_len':  96, 'label_len':  48, 'pred_len': 24},
    ])

# ── 组8: 并行CNN vs 基线 ──
add_group('cnn_parallel', 'informer',
    {'data': 'jena', 'data_path': 'jena_processed.csv',
     'target': 'T (degC)', 'features': 'MS', 'seq_len': 96, 'label_len': 48,
     'pred_len': 24, 'd_model': 128, 'd_ff': 256, 'dropout': 0.1, 'attn': 'prob',
     'train_epochs': 4, 'batch_size': 16, 'inverse': False, 'itr': 1, 'freq': 'h'},
    'use_cnn_parallel', [True, False])


# ── 组10: 10分钟+CNN并行 ──
add_group('jena10_cnn', 'informer',
    {'data': 'jena10min', 'data_path': 'jena_10min.csv',
     'target': 'T (degC)', 'features': 'MS', 'seq_len': 576, 'label_len': 288,
     'pred_len': 96, 'd_model': 128, 'd_ff': 256, 'dropout': 0.1, 'attn': 'prob',
     'train_epochs': 4, 'batch_size': 8, 'inverse': False, 'itr': 1, 'freq': 't'},
    'use_cnn_parallel', [True, False])


# ============================================================
def build_command(exp):
    model = exp.pop('model')
    des = exp.pop('des')
    if model == 'informer':
        cmd = [PYTHON, 'main_informer.py', '--model', 'informer', '--data', exp.pop('data')]
        for k in ['data_path','features','target','seq_len','label_len','pred_len',
                  'd_model','n_heads','e_layers','d_layers','d_ff','factor','dropout',
                  'attn','activation','embed','train_epochs','batch_size','patience',
                  'learning_rate','lradj','itr','loss','freq']:
            if k in exp: cmd += [f'--{k}', str(exp[k])]
        if exp.pop('inverse', False): cmd += ['--inverse']
        if exp.pop('use_amp', False): cmd += ['--use_amp']
        if exp.pop('use_rope', False): cmd += ['--use_rope']
        if exp.pop('use_tcn', False): cmd += ['--use_tcn']
        if exp.pop('use_cnn_parallel', False): cmd += ['--use_cnn_parallel']
        cmd += ['--des', des]
    elif model == 'lstm':
        cmd = [PYTHON, 'LSTM_序列预测/run_lstm.py']
        for k in ['data_path','target','seq_len','pred_len','hidden_size',
                  'num_layers','dropout','batch_size','epochs','lr','patience']:
            if k in exp: cmd += [f'--{k}', str(exp[k])]
    elif model == 'arima':
        cmd = [PYTHON, 'ARIMA_序列预测/arima.py/run_arima.py']
        for k in ['data_path','target','seq_len','pred_len']:
            if k in exp: cmd += [f'--{k}', str(exp[k])]
    return cmd

# ============================================================
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--group', type=str)
    args = p.parse_args()

    groups = GROUPS
    if args.group:
        groups = [g for g in GROUPS if g['name'] == args.group]
        if not groups: print(f'未找到组: {args.group}'); return

    for g in groups:
        print(f'\n{"#"*70}\n  实验组: {g["name"]}  ({g["model"]}, {len(g["experiments"])} 组)\n{"#"*70}')
        for i, e in enumerate(g['experiments']):
            exp = dict(e); exp['model'] = g['model']
            # 所有消融组: 把变动值编进 des, 确保改参数不会覆盖
            exp['des'] = f"{g['name']}_{g['sweep_values'][i].replace(' ','').replace('(','').replace(')','').replace('%','pct')}"
            cmd = build_command(exp)
            print(f'\n[{i+1}/{len(g["experiments"])}] {" ".join(cmd)}')
            if args.dry_run: continue
            try:
                subprocess.run(cmd, cwd=os.path.dirname(os.path.abspath(__file__)))
            except Exception as ex:
                print(f'  ❌ {ex}')

        # 组结束后自动绘图
        if not args.dry_run:
            # 对 sweep_values 做同样的 sanitize 以匹配目录名
            sweep_vals_sanitized = [v.replace(' ','').replace('(','').replace(')','').replace('%','pct') for v in g['sweep_values']]
            plot_cmd = [PYTHON, 'plot_ablation.py', '--group', g['name'],
                        '--experiments', *sweep_vals_sanitized]
            print(f'\n  📊 {" ".join(plot_cmd)}')
            try:
                subprocess.run(plot_cmd, cwd=os.path.dirname(os.path.abspath(__file__)))
            except Exception as ex:
                print(f'  ⚠️ 绘图失败: {ex}')

if __name__ == '__main__':
    main()
