"""
训练损失 & 测试集预测结果可视化
运行方式: python visualize.py <setting>
示例:     python visualize.py informer_weather_ftM_sl96_ll48_pl24_dm512_nh8_el2_dl1_df2048_atprob_fc5_ebtimeF_dtTrue_mxTrue_Exp_weather_0
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
  # 防止中文乱码
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# ── 从命令行参数获取 setting ──────────────────────────────
if len(sys.argv) < 2:
    print('用法: python visualize.py <setting>')
    print('示例: python visualize.py informer_weather_ftM_sl96_ll48_pl24_dm512_nh8_el2_dl1_df2048_atprob_fc5_ebtimeF_dtTrue_mxTrue_Exp_weather_0')
    sys.exit(1)

setting = sys.argv[1]
checkpoint_dir = os.path.join('checkpoints', setting)
result_dir = os.path.join('results', setting)

# ── 要画的特征名（根据你的 weather_processed.csv 列顺序填写） ──
FEATURE_NAMES = [
    'Apparent Temperature (C)', 'Humidity', 'Wind Speed (km/h)',
    'Wind Bearing (degrees)', 'Visibility (km)', 'Pressure (millibars)',
    'Summary_Breezy and Mostly Cloudy', 'Summary_Breezy and Overcast',
    'Summary_Breezy and Partly Cloudy', 'Summary_Clear', 'Summary_Foggy',
    'Summary_Mostly Cloudy', 'Summary_Other', 'Summary_Overcast',
    'Summary_Partly Cloudy', 'PrecipType_null', 'PrecipType_rain',
    'PrecipType_snow', 'Temperature (C)',
]

# ── 图 1: Loss 曲线 ───────────────────────────────────────
epoch_losses_path = os.path.join(checkpoint_dir, 'epoch_losses.npy')
if os.path.exists(epoch_losses_path):
    losses = np.load(epoch_losses_path)
    train_loss, vali_loss, test_loss = losses[0], losses[1], losses[2]
    epochs = range(1, len(train_loss) + 1)

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    ax.plot(epochs, train_loss, 'o-', label='Train Loss', color='#2196F3', linewidth=2)
    ax.plot(epochs, vali_loss, 's--', label='Vali Loss', color='#FF9800', linewidth=2)
    ax.plot(epochs, test_loss, '^-.', label='Test Loss', color='#F44336', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('MSE Loss (standardized scale)')
    ax.set_title(f'Training History — {setting}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(result_dir, 'loss_curve.png'), dpi=150)
    print(f'[OK] Loss 曲线已保存: {result_dir}/loss_curve.png')
    plt.show()
else:
    print(f'[!] 未找到 epoch_losses.npy: {epoch_losses_path}')
    print('    请先运行训练，或检查 setting 名称是否正确。')

# ── 图 2: 测试集 预测 vs 真实 ──────────────────────────────
pred_path = os.path.join(result_dir, 'pred.npy')
true_path = os.path.join(result_dir, 'true.npy')

if os.path.exists(pred_path) and os.path.exists(true_path):
    preds = np.load(pred_path)   # shape: [n_samples, pred_len, c_out]
    trues = np.load(true_path)   # shape: [n_samples, pred_len, c_out]

    # ── 选择要绘制的特征维度 ──
    # 默认用最后一个特征 (Temperature (C)), 也可通过 --target 参数指定
    if '--target' in sys.argv:
        target_idx = sys.argv.index('--target') + 1
        target_name = sys.argv[target_idx]
        if target_name in FEATURE_NAMES:
            feat_idx = FEATURE_NAMES.index(target_name)
        else:
            feat_idx = -1
    else:
        feat_idx = -1  # 默认最后一个 = Temperature (C)

    feat_name = FEATURE_NAMES[feat_idx] if feat_idx >= 0 else f'Feature[{feat_idx}]'

    # 随机选 4 个测试样本画子图
    n_plot = min(4, len(preds))
    sample_indices = np.random.choice(len(preds), n_plot, replace=False)

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    axes = axes.flatten()
    for i, idx in enumerate(sample_indices):
        pred_vals = preds[idx, :, feat_idx]
        true_vals = trues[idx, :, feat_idx]
        steps = range(len(pred_vals))

        axes[i].plot(steps, true_vals, 'o-', label='True', color='#4CAF50', linewidth=2)
        axes[i].plot(steps, pred_vals, 'x--', label='Pred', color='#F44336', linewidth=2)
        axes[i].set_xlabel('预测步长')
        axes[i].set_ylabel(feat_name)
        axes[i].set_title(f'Sample {idx} (pred_len={len(pred_vals)})')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)

    fig.suptitle(f'Test Set: Prediction vs Ground Truth — {feat_name}', fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(os.path.join(result_dir, 'pred_vs_true.png'), dpi=150)
    print(f'[OK] 预测对比图已保存: {result_dir}/pred_vs_true.png')
    plt.show()
else:
    print(f'[!] 未找到 pred.npy / true.npy: {result_dir}')
    print('    请先运行测试。')

# ── 图 3: 所有特征维度的 MAE 柱状图 ────────────────────────
if os.path.exists(pred_path) and os.path.exists(true_path):
    # 计算每个特征维度的 MAE
    abs_errors = np.abs(preds - trues)           # [n, pred_len, c_out]
    per_feat_mae = abs_errors.mean(axis=(0, 1))  # [c_out]

    fig, ax = plt.subplots(figsize=(12, 5))
    bars = ax.bar(range(len(per_feat_mae)), per_feat_mae, color='#FF9800', alpha=0.8)
    ax.set_xlabel('Feature dimension')
    ax.set_ylabel('MAE (standardized scale)')
    ax.set_title('Per-Feature Prediction Error on Test Set')
    ax.set_xticks(range(len(per_feat_mae)))
    ax.set_xticklabels(FEATURE_NAMES, rotation=45, ha='right', fontsize=8)
    ax.grid(True, axis='y', alpha=0.3)

    # 在柱子上标数值
    for bar, val in zip(bars, per_feat_mae):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.3f}', ha='center', va='bottom', fontsize=7)

    fig.tight_layout()
    fig.savefig(os.path.join(result_dir, 'per_feature_error.png'), dpi=150)
    print(f'[OK] 特征误差柱状图已保存: {result_dir}/per_feature_error.png')
    plt.show()

print('\n完成! 所有图片已保存到:', result_dir)
