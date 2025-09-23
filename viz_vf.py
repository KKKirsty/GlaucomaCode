import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

# 54 点的棋盘位置
RECT_POS = [                   [3,0],[4,0],[5,0],[6,0],
                        [2,1],[3,1],[4,1],[5,1],[6,1],[7,1],
                    [1,2],[2,2],[3,2],[4,2],[5,2],[6,2],[7,2],[8,2],
                    [1,3],[2,3],[3,3],[4,3],[5,3],[6,3],[7,3],[8,3],[9,3],
                    [1,4],[2,4],[3,4],[4,4],[5,4],[6,4],[7,4],[8,4],[9,4],
                    [1,5],[2,5],[3,5],[4,5],[5,5],[6,5],[7,5],[8,5],
                        [2,6],[3,6],[4,6],[5,6],[6,6],[7,6],
                                [3,7],[4,7],[5,7],[6,7]]

def _draw_vf_grid(ax, values, title=None, rect_size=54, vmin=5, vmax=35, fontsize=9):
    """画 54 点网格，灰度/文本与官方风格接近。"""
    values = np.asarray(values).reshape(-1)
    n = min(54, len(values))
    for i in range(n):
        x = RECT_POS[i][0] * rect_size
        y = (7 - RECT_POS[i][1]) * rect_size
        vf = float(values[i])
        # 灰度背景（5~35 线性映射）
        bg = (vf - vmin) / (vmax - vmin)
        bg = max(0.0, min(1.0, bg))
        txt = 'white' if bg < 0.5 else 'black'
        ax.fill([x, x+rect_size, x+rect_size, x],
                [y, y, y+rect_size, y], color=(bg, bg, bg), edgecolor='none')
        ax.text(x+rect_size/2, y+rect_size/2, f"{vf:.1f}",
                ha='center', va='center', color=txt, fontsize=fontsize)
    ax.axis('scaled'); ax.axis('off')
    if title:
        ax.set_title(title, fontsize=12)

def panel_A_sample(rnfl_path, truth_vec, pred_vec, save_path):
    """(A) 输入 RNFLT + GT + Pred 三列拼图"""
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    # 左：输入 RNFLT 图
    if rnfl_path and os.path.exists(rnfl_path):
        img = Image.open(rnfl_path)
        axes[0].imshow(img)
        axes[0].set_title("Input (RNFLT)", fontsize=12)
        axes[0].axis('off')
    else:
        axes[0].text(0.5, 0.5, "RNFLT not found", ha='center', va='center')
        axes[0].axis('off')
    # 中：GT 网格
    _draw_vf_grid(axes[1], truth_vec, title="Ground truth")
    # 右：Pred 网格
    _draw_vf_grid(axes[2], pred_vec, title="Prediction")
    plt.tight_layout()
    os.makedirs(Path(save_path).parent, exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close(fig)
    print(f"[A] saved -> {save_path}")

def _pointwise_mae(truth_mat, pred_mat):
    """逐点 MAE（对样本维度取平均），返回 shape=(K,)"""
    diff = np.abs(pred_mat - truth_mat)
    return diff.mean(axis=0)

def _heatmap(ax, vec, title, vmin=None, vmax=None, cmap='viridis', rect_size=1.0):
    """把长度 54 的向量按网格位置画热图（更接近论文右侧 B 面板）"""
    vec = np.asarray(vec).reshape(-1)
    xs, ys, cs = [], [], []
    for i in range(min(54, len(vec))):
        xs.append(RECT_POS[i][0]); ys.append(RECT_POS[i][1]); cs.append(vec[i])
    xs = np.array(xs); ys = np.array(ys); cs = np.array(cs)
    # 为了让像素化更“块状”，用 pcolormesh/hexbin 也可；这里用散点 + size
    sc = ax.scatter(xs, ys, c=cs, s=rect_size*2000, marker='s', vmin=vmin, vmax=vmax, cmap=cmap)
    ax.invert_yaxis()
    ax.set_aspect('equal')
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title, fontsize=12)
    plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.02)

def panel_B_pointwise_mae(
    df_tar, df_pred, lat_col='test_lat',
    baseline_pred_df=None, out_path="pointwise_mae_panelB.png"
):
    """
    生成 (B)：
    - 左/右眼的逐点 MAE 热图（2 图）
    - 若提供 baseline_pred_df，再生成 MAE 改善图： (baseline MAE - model MAE)（2 图）
    """
    # 取列
    tar_cols  = [c for c in df_tar.columns  if c.startswith('tar_')]
    pred_cols = [c for c in df_pred.columns if c.startswith('pred_')]

    # 对齐行（按 test_id merge）
    merged = pd.merge(
        df_tar[['test_id', lat_col] + tar_cols],
        df_pred[['test_id', lat_col] + pred_cols],
        on=['test_id', lat_col], how='inner'
    )
    if merged.empty:
        raise RuntimeError("No overlapping rows between tar and pred.")

    # 拆左右
    mats = {}
    for side in ['Right', 'Left']:
        m = merged[merged[lat_col] == side]
        if len(m) == 0: 
            mats[side] = None
            continue
        truth_mat = m[tar_cols].to_numpy(dtype=np.float32)
        pred_mat  = m[pred_cols].to_numpy(dtype=np.float32)
        mae_vec = _pointwise_mae(truth_mat, pred_mat)
        mats[side] = {'mae': mae_vec}

    # baseline（可选）
    if baseline_pred_df is not None:
        merged_b = pd.merge(
            df_tar[['test_id', lat_col] + tar_cols],
            baseline_pred_df[['test_id', lat_col] + pred_cols],
            on=['test_id', lat_col], how='inner'
        )
        for side in ['Right', 'Left']:
            m = merged_b[merged_b[lat_col] == side]
            if len(m) == 0 or mats.get(side) is None:
                continue
            truth_mat = m[tar_cols].to_numpy(dtype=np.float32)
            base_mat  = m[pred_cols].to_numpy(dtype=np.float32)
            base_mae = _pointwise_mae(truth_mat, base_mat)
            mats[side]['improve'] = base_mae - mats[side]['mae']  # 正数=改善

    # 画图
    ncols = 2 if all('improve' not in (mats[s] or {}) for s in ['Right','Left']) else 4
    fig, axes = plt.subplots(1, ncols, figsize=(4*ncols, 4))

    col = 0
    # 右眼 MAE
    if mats['Right'] is not None:
        _heatmap(axes[col], mats['Right']['mae'], "Pointwise MAE (right)"); col += 1
    # 左眼 MAE
    if mats['Left'] is not None:
        _heatmap(axes[col], mats['Left']['mae'], "Pointwise MAE (left)"); col += 1
    # 改善（右）
    if mats.get('Right') and 'improve' in mats['Right']:
        _heatmap(axes[col], mats['Right']['improve'], "MAE improvement (right)", cmap='coolwarm'); col += 1
    # 改善（左）
    if mats.get('Left') and 'improve' in mats['Left']:
        _heatmap(axes[col], mats['Left']['improve'], "MAE improvement (left)", cmap='coolwarm'); col += 1

    plt.tight_layout()
    os.makedirs(Path(out_path).parent, exist_ok=True)
    plt.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"[B] saved -> {out_path}")

# --------- 便捷加载与调用 ----------
def load_excels(pred_xlsx, tar_xlsx):
    df_pred = pd.read_excel(pred_xlsx)
    df_tar  = pd.read_excel(tar_xlsx)
    return df_pred, df_tar

def demo_one_case_panelA(
    pred_xlsx, tar_xlsx, row_index, rnfl_root=None, rnfl_name="rnfl_thickness_map.jpg", out_path="panelA.png"
):
    """根据行号从 pred/tar.xlsx 拿一条结果画 Panel A。
       如果提供 rnfl_root，会尝试在 rnfl_root/data_xxxx/rnfl_thickness_map.jpg 找输入图。"""
    df_pred, df_tar = load_excels(pred_xlsx, tar_xlsx)
    assert 0 <= row_index < len(df_pred) and 0 <= row_index < len(df_tar)
    sid = str(df_pred.iloc[row_index]['test_id'])

    pred_vec = df_pred.iloc[row_index][[c for c in df_pred.columns if c.startswith('pred_')]].to_numpy()
    tar_vec  = df_tar.iloc[row_index][[c for c in df_tar.columns  if c.startswith('tar_')]].to_numpy()

    rnfl_path = None
    if rnfl_root:
        # 期望目录结构：rnfl_root/data_xxxx/rnfl_thickness_map.jpg
        rnfl_path = os.path.join(rnfl_root, sid, rnfl_name)
    panel_A_sample(rnfl_path, tar_vec, pred_vec, out_path)

def demo_panelB(
    pred_xlsx, tar_xlsx, out_path="panelB.png", baseline_pred_xlsx=None
):
    df_pred, df_tar = load_excels(pred_xlsx, tar_xlsx)
    df_base = pd.read_excel(baseline_pred_xlsx) if baseline_pred_xlsx else None
    panel_B_pointwise_mae(df_tar, df_pred, baseline_pred_df=df_base, out_path=out_path)
