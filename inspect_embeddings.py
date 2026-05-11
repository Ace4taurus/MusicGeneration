"""
inspect_embeddings.py
─────────────────────────────────────────────────────────────────────────────
可视化 REMI tokenizer 词表 以及 模型 Embedding 权重。

功能:
  1. 打印完整词表（token id → token 名称 → 类型）
  2. 提取音高(Pitch) token 的 embedding 向量
  3. PCA 2D 可视化：按 pitch class / octave 染色
  4. 余弦相似度热力图（88×88，音高 token 之间）
  5. 对比：随机初始化 vs 训练后 checkpoint（MusicVAE / Transformer）
  6. 输出：interval cosine similarity 分析（半音 vs 纯五度 vs 纯八度）

依赖: matplotlib, scikit-learn, numpy, torch, miditok
运行: python inspect_embeddings.py [--ckpt checkpoints/music_vae_best.pt]
      python inspect_embeddings.py --model transformer
"""

import sys
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")           # 无 GUI 环境也能运行；若有显示器可改为 "TkAgg"
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# ── 本项目路径 ────────────────────────────────────────────────────
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

import config as cfg
from miditok import REMI

# ═════════════════════════════════════════════════════════════════
# 工具函数
# ═════════════════════════════════════════════════════════════════

PITCH_NAMES = ["C", "C#", "D", "D#", "E", "F",
               "F#", "G", "G#", "A", "A#", "B"]

# 五度圈顺序（用于颜色排列，相邻颜色 = 五度关系）
CIRCLE_OF_FIFTHS = [0, 7, 2, 9, 4, 11, 6, 1, 8, 3, 10, 5]  # C G D A E B F# Db Ab Eb Bb F

# 音程名称
INTERVAL_NAMES = {
    0: "同音 (Unison)",
    1: "小二度 (m2, dissonant)",
    2: "大二度 (M2)",
    3: "小三度 (m3)",
    4: "大三度 (M3)",
    5: "纯四度 (P4)",
    6: "三全音 (TT, dissonant)",
    7: "纯五度 (P5)",
    8: "小六度 (m6)",
    9: "大六度 (M6)",
    10: "小七度 (m7)",
    11: "大七度 (M7, dissonant)",
    12: "纯八度 (P8, consonant)",
}

PITCH_CLASS_COLORS = plt.cm.hsv(np.linspace(0, 1, 12, endpoint=False))


def load_tokenizer() -> REMI:
    tok = REMI(params=cfg.TOKENIZER_PATH)
    return tok


def get_vocab_info(tok: REMI):
    """
    返回 list of (id, token_str, token_type)
    token_type: 'special' / 'pitch' / 'duration' / 'velocity' / 'tempo' / 'pedal' / 'time_shift' / 'other'
    """
    vocab = []
    for token_str, token_id in tok.vocab.items():
        ts = token_str.lower()
        if any(s in ts for s in ["pad", "bos", "eos"]):
            ttype = "special"
        elif ts.startswith("pitch"):
            ttype = "pitch"
        elif ts.startswith("duration") or ts.startswith("noteduration"):
            ttype = "duration"
        elif ts.startswith("velocity"):
            ttype = "velocity"
        elif ts.startswith("tempo"):
            ttype = "tempo"
        elif ts.startswith("pedal") or "sustain" in ts:
            ttype = "pedal"
        elif ts.startswith("timeshifts") or ts.startswith("time_shift") or ts.startswith("timeshift"):
            ttype = "time_shift"
        elif ts.startswith("bar"):
            ttype = "bar"
        elif ts.startswith("position"):
            ttype = "position"
        elif ts.startswith("rest"):
            ttype = "rest"
        else:
            ttype = "other"
        vocab.append((token_id, token_str, ttype))
    vocab.sort(key=lambda x: x[0])
    return vocab


def extract_pitch_tokens(vocab_info):
    """
    Return [(id, token_str, midi_pitch)] for piano pitch tokens only.
    Excludes PitchDrum_ tokens; keeps only Pitch_<int> in MIDI range 21-108.
    """
    pitch_tokens = []
    for tid, tstr, ttype in vocab_info:
        if ttype == "pitch":
            ts_lower = tstr.lower()
            # Skip drum pitch tokens
            if "drum" in ts_lower:
                continue
            # token name like "Pitch_60" or "Pitch_21"
            try:
                midi_pitch = int(tstr.split("_")[-1])
                # Piano range only (A0=21 .. C8=108)
                if 21 <= midi_pitch <= 108:
                    pitch_tokens.append((tid, tstr, midi_pitch))
            except (ValueError, IndexError):
                pass
    pitch_tokens.sort(key=lambda x: x[2])
    return pitch_tokens


def cosine_sim_matrix(emb: np.ndarray) -> np.ndarray:
    """emb: (N, D)，返回 (N, N) 余弦相似度矩阵"""
    norms = np.linalg.norm(emb, axis=1, keepdims=True) + 1e-8
    normed = emb / norms
    return normed @ normed.T


def load_embedding_from_checkpoint(ckpt_path: str, model_type: str, vocab_size: int):
    """
    Load token embedding weights from a checkpoint.
    Returns numpy array (vocab_size, d_model), or None if not found.
    """
    path = Path(ckpt_path)
    if not path.exists():
        print(f"  [Warning] Checkpoint not found: {ckpt_path}")
        return None

    print(f"  Loading checkpoint: {ckpt_path} ...")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    # Unwrap common checkpoint wrappers
    if isinstance(ckpt, dict):
        # Try common keys: model, model_state_dict, state_dict, or raw dict
        sd = ckpt.get("model",
             ckpt.get("model_state_dict",
             ckpt.get("state_dict",
             ckpt)))
    else:
        sd = ckpt

    # Search for token embedding weight
    for key in sd.keys():
        if "token_emb.weight" in key or "embedding.weight" in key:
            w = sd[key].float().numpy()
            print(f"  Found embedding: key={key}, shape={w.shape}")
            return w

    # If not found, the sd might still be a nested dict (e.g. ckpt itself has meta keys)
    # Try encoder.token_emb or decoder.token_emb
    for key in sd.keys():
        if "token_emb" in key and key.endswith(".weight"):
            w = sd[key].float().numpy()
            print(f"  Found embedding: key={key}, shape={w.shape}")
            return w

    print(f"  [Warning] token_emb.weight not found. Available keys (first 20):")
    for k in list(sd.keys())[:20]:
        v = sd[k]
        shape_info = v.shape if hasattr(v, 'shape') else type(v).__name__
        print(f"    {k}: {shape_info}")
    return None


# ═════════════════════════════════════════════════════════════════
# 打印词表
# ═════════════════════════════════════════════════════════════════

def print_vocab(vocab_info, show_all=False):
    print("\n" + "="*70)
    print("REMI 词表总览")
    print("="*70)

    # 统计各类型数量
    from collections import Counter
    type_counts = Counter(t for _, _, t in vocab_info)
    print(f"词表大小: {len(vocab_info)}")
    for ttype, cnt in sorted(type_counts.items(), key=lambda x: -x[1]):
        print(f"  {ttype:<12}: {cnt:4d} tokens")
    print()

    if show_all:
        print(f"{'ID':>5}  {'Token':<35}  {'类型':<12}")
        print("-"*56)
        for tid, tstr, ttype in vocab_info:
            print(f"{tid:>5}  {tstr:<35}  {ttype:<12}")
    else:
        # Print first 3 and last 3 of each type
        from collections import defaultdict
        by_type = defaultdict(list)
        for row in vocab_info:
            by_type[row[2]].append(row)

        for ttype in ["special", "pitch", "duration", "velocity",
                      "tempo", "time_shift", "position", "bar", "pedal", "other"]:
            items = by_type.get(ttype, [])
            if not items:
                continue
            print(f"  [{ttype}] ({len(items)} tokens)")
            if len(items) <= 6:
                show = items
            else:
                show = items[:3] + ["  ..."] + items[-3:]
            for row in show:
                if row == "  ...":
                    print("   ...")
                else:
                    print(f"    ID={row[0]:4d}  {row[1]}")
            print()


# ═════════════════════════════════════════════════════════════════
# 可视化函数
# ═════════════════════════════════════════════════════════════════

def plot_pitch_pca(emb_matrix: np.ndarray, pitch_tokens, title: str, save_path: str):
    """PCA 降维后散点图，按 pitch class 染色，按八度标注"""
    ids   = [t[0] for t in pitch_tokens]
    names = [t[1] for t in pitch_tokens]
    pitches = [t[2] for t in pitch_tokens]

    vecs = emb_matrix[ids]   # (88, d_model)

    # PCA 降到 2D
    pca = PCA(n_components=2)
    coords = pca.fit_transform(vecs)   # (88, 2)

    fig, ax = plt.subplots(figsize=(12, 9))
    ax.set_title(f"{title}\nPitch Embeddings — PCA 2D\n"
                 f"(PC1 {pca.explained_variance_ratio_[0]*100:.1f}%  "
                 f"PC2 {pca.explained_variance_ratio_[1]*100:.1f}%)",
                 fontsize=13)

    for i, (p, xy) in enumerate(zip(pitches, coords)):
        pc  = p % 12           # pitch class 0-11
        oct = p // 12 - 1      # octave (MIDI 60 = C4)
        color = PITCH_CLASS_COLORS[pc]
        ax.scatter(xy[0], xy[1], color=color, s=60, zorder=3)
        # 只标注 C 音（pitch_class = 0）
        if pc == 0:
            ax.annotate(f"C{oct}", xy, fontsize=7, ha='center', va='bottom',
                        color='black')

    # 图例
    patches = [mpatches.Patch(color=PITCH_CLASS_COLORS[pc],
                               label=PITCH_NAMES[pc])
               for pc in range(12)]
    ax.legend(handles=patches, title="Pitch Class",
              loc="upper right", fontsize=8, ncol=2)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  [图] 已保存: {save_path}")


def plot_pitch_tsne(emb_matrix: np.ndarray, pitch_tokens, title: str, save_path: str):
    """t-SNE 降维后散点图"""
    ids     = [t[0] for t in pitch_tokens]
    pitches = [t[2] for t in pitch_tokens]
    vecs    = emb_matrix[ids]

    if vecs.shape[0] < 5:
        print("  [跳过] token 数量太少，跳过 t-SNE")
        return

    perp = min(30, vecs.shape[0] - 1)
    tsne = TSNE(n_components=2, perplexity=perp, random_state=42,
                max_iter=1000, init="pca")
    coords = tsne.fit_transform(vecs)

    fig, ax = plt.subplots(figsize=(12, 9))
    ax.set_title(f"{title}\nPitch Embeddings — t-SNE 2D", fontsize=13)

    for i, (p, xy) in enumerate(zip(pitches, coords)):
        pc  = p % 12
        oct = p // 12 - 1
        color = PITCH_CLASS_COLORS[pc]
        ax.scatter(xy[0], xy[1], color=color, s=60, zorder=3)
        if pc == 0:
            ax.annotate(f"C{oct}", xy, fontsize=7, ha='center', va='bottom',
                        color='black')

    patches = [mpatches.Patch(color=PITCH_CLASS_COLORS[pc],
                               label=PITCH_NAMES[pc])
               for pc in range(12)]
    ax.legend(handles=patches, title="Pitch Class",
              loc="upper right", fontsize=8, ncol=2)
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  [图] 已保存: {save_path}")


def plot_cosine_heatmap(emb_matrix: np.ndarray, pitch_tokens, title: str, save_path: str):
    """88×88 余弦相似度热力图"""
    ids     = [t[0] for t in pitch_tokens]
    pitches = [t[2] for t in pitch_tokens]
    vecs    = emb_matrix[ids]

    sim = cosine_sim_matrix(vecs)   # (88, 88)

    fig, ax = plt.subplots(figsize=(14, 12))
    im = ax.imshow(sim, aspect='auto', cmap='RdBu_r',
                   vmin=-1, vmax=1, origin='upper')
    plt.colorbar(im, ax=ax, label="Cosine Similarity")
    ax.set_title(f"{title}\nPitch Token Cosine Similarity (88×88)", fontsize=13)

    # x/y 轴：每12个音高标一个 C
    tick_positions = []
    tick_labels    = []
    for i, p in enumerate(pitches):
        if p % 12 == 0:
            oct = p // 12 - 1
            tick_positions.append(i)
            tick_labels.append(f"C{oct}")
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, fontsize=9)
    ax.set_yticks(tick_positions)
    ax.set_yticklabels(tick_labels, fontsize=9)

    ax.set_xlabel("Pitch Token (MIDI)")
    ax.set_ylabel("Pitch Token (MIDI)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  [图] 已保存: {save_path}")


def plot_interval_similarity(emb_matrix: np.ndarray, pitch_tokens, title: str, save_path: str):
    """
    分析不同音程距离的平均余弦相似度。
    X 轴 = 音程（0~12 半音），Y 轴 = 该音程下所有音符对的平均余弦相似度。
    理想：octave(12) > fifth(7) > third(4) > semitone(1)
    """
    ids     = [t[0] for t in pitch_tokens]
    pitches = [t[2] for t in pitch_tokens]
    vecs    = emb_matrix[ids]
    sim     = cosine_sim_matrix(vecs)

    pitch_arr = np.array(pitches)
    max_interval = 24   # 分析两个八度内

    avg_sim = []
    std_sim = []
    intervals = list(range(max_interval + 1))

    for interval in intervals:
        sims = []
        for i in range(len(pitches)):
            for j in range(len(pitches)):
                if abs(pitches[i] - pitches[j]) == interval:
                    sims.append(sim[i, j])
        if sims:
            avg_sim.append(np.mean(sims))
            std_sim.append(np.std(sims))
        else:
            avg_sim.append(np.nan)
            std_sim.append(0)

    avg_sim = np.array(avg_sim)
    std_sim = np.array(std_sim)

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar(intervals, avg_sim, color='steelblue', alpha=0.7, label='Mean Cosine Sim')
    ax.errorbar(intervals, avg_sim, yerr=std_sim, fmt='none',
                color='black', capsize=3, linewidth=1)

    # 标注关键音程
    key_intervals = {1: "m2\n(dissonant)", 2: "M2", 3: "m3", 4: "M3",
                     5: "P4", 6: "TT\n(dissonant)", 7: "P5",
                     12: "P8\n(octave)", 19: "P12\n(5th+8va)", 24: "2 Oct"}
    for iv, name in key_intervals.items():
        if iv <= max_interval:
            ax.axvline(iv, color='red', alpha=0.3, linewidth=1, linestyle='--')
            if not np.isnan(avg_sim[iv]):
                ax.text(iv, avg_sim[iv] + std_sim[iv] + 0.02,
                        name, ha='center', fontsize=7, color='red')

    ax.set_xlabel("Interval (semitones)")
    ax.set_ylabel("Mean Cosine Similarity")
    ax.set_title(f"{title}\nInterval → Cosine Similarity\n"
                 "(Ideal: Octave > 5th > 3rd > semitone)",
                 fontsize=13)
    ax.set_xticks(intervals)
    ax.axhline(0, color='black', linewidth=0.5)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  [图] 已保存: {save_path}")


def plot_pca_cof_color(emb_matrix: np.ndarray, pitch_tokens, title: str, save_path: str):
    """
    PCA 图，但按五度圈顺序染色（验证模型是否学到五度圈结构）。
    五度圈: C-G-D-A-E-B-F#-Db-Ab-Eb-Bb-F
    """
    ids     = [t[0] for t in pitch_tokens]
    pitches = [t[2] for t in pitch_tokens]
    vecs    = emb_matrix[ids]

    pca    = PCA(n_components=2)
    coords = pca.fit_transform(vecs)

    # 为每个 pitch class 在五度圈中分配颜色
    cof_order = CIRCLE_OF_FIFTHS
    cof_pos   = {pc: i for i, pc in enumerate(cof_order)}
    cof_colors = plt.cm.hsv(np.array([cof_pos[pc] for pc in range(12)]) / 12.0)

    fig, ax = plt.subplots(figsize=(12, 9))
    ax.set_title(f"{title}\nPitch Embeddings PCA — Circle of Fifths 染色\n"
                 "（颜色按五度圈顺序排列：相邻颜色=纯五度关系）",
                 fontsize=12)

    for p, xy in zip(pitches, coords):
        pc    = p % 12
        oct   = p // 12 - 1
        color = cof_colors[pc]
        ax.scatter(xy[0], xy[1], color=color, s=70, zorder=3)
        if pc == 0:
            ax.annotate(f"C{oct}", xy, fontsize=7, ha='center', va='bottom')

    patches = [mpatches.Patch(color=cof_colors[pc],
                               label=f"{PITCH_NAMES[cof_order[i]]} (pos {i})")
               for i, pc in enumerate(cof_order)]
    ax.legend(handles=patches, title="五度圈位置",
              loc="upper right", fontsize=7, ncol=2)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  [图] 已保存: {save_path}")


def print_interval_stats(emb_matrix: np.ndarray, pitch_tokens, label: str):
    """在控制台打印关键音程的余弦相似度统计"""
    ids     = [t[0] for t in pitch_tokens]
    pitches = [t[2] for t in pitch_tokens]
    vecs    = emb_matrix[ids]
    sim     = cosine_sim_matrix(vecs)

    print(f"\n{'='*60}")
    print(f"音程余弦相似度分析 — {label}")
    print(f"{'='*60}")
    print(f"  {'音程':<6}  {'名称':<25}  {'均值':>8}  {'标准差':>8}")
    print(f"  {'-'*55}")

    key_intervals = [(1, "小二度 m2 [dissonant]"),
                     (2, "大二度 M2"),
                     (3, "小三度 m3"),
                     (4, "大三度 M3"),
                     (5, "纯四度 P4"),
                     (6, "三全音 TT [dissonant]"),
                     (7, "纯五度 P5"),
                     (9, "大六度 M6"),
                     (12, "纯八度 P8 [octave]"),
                     (19, "纯十二度 P12"),
                     (24, "双八度 2*P8")]

    for interval, name in key_intervals:
        sims = []
        for i in range(len(pitches)):
            for j in range(len(pitches)):
                if abs(pitches[i] - pitches[j]) == interval:
                    sims.append(sim[i, j])
        if sims:
            mean_s = np.mean(sims)
            std_s  = np.std(sims)
            print(f"  {interval:<6}  {name:<25}  {mean_s:>8.4f}  {std_s:>8.4f}")
        else:
            print(f"  {interval:<6}  {name:<25}  {'N/A':>8}")


# ═════════════════════════════════════════════════════════════════
# 全词表 embedding 概览
# ═════════════════════════════════════════════════════════════════

def plot_full_vocab_pca(emb_matrix: np.ndarray, vocab_info, title: str, save_path: str):
    """对全部词表做 PCA，按 token 类型染色"""
    type_color = {
        "special":    "black",
        "pitch":      "royalblue",
        "duration":   "tomato",
        "velocity":   "green",
        "tempo":      "purple",
        "time_shift": "orange",
        "position":   "gold",
        "bar":        "cyan",
        "pedal":      "brown",
        "other":      "gray",
    }

    vecs   = emb_matrix   # (V, D)
    ids    = [row[0] for row in vocab_info]
    ttypes = [row[2] for row in vocab_info]

    # 只取 vocab_info 中的 id（有些 id 可能超出矩阵范围）
    valid_mask = [i < emb_matrix.shape[0] for i in ids]
    ids    = [i for i, ok in zip(ids, valid_mask)   if ok]
    ttypes = [t for t, ok in zip(ttypes, valid_mask) if ok]

    vecs_sel = emb_matrix[ids]

    pca    = PCA(n_components=2)
    coords = pca.fit_transform(vecs_sel)

    fig, ax = plt.subplots(figsize=(12, 9))
    ax.set_title(f"{title}\n全词表 Embedding PCA 2D — 按 token 类型染色\n"
                 f"(PC1 {pca.explained_variance_ratio_[0]*100:.1f}%  "
                 f"PC2 {pca.explained_variance_ratio_[1]*100:.1f}%)",
                 fontsize=12)

    for xy, ttype in zip(coords, ttypes):
        color = type_color.get(ttype, "gray")
        ax.scatter(xy[0], xy[1], color=color, s=20, alpha=0.6, zorder=2)

    patches = [mpatches.Patch(color=c, label=t)
               for t, c in type_color.items()
               if t in set(ttypes)]
    ax.legend(handles=patches, title="Token 类型",
              loc="upper right", fontsize=9)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  [图] 已保存: {save_path}")


# ═════════════════════════════════════════════════════════════════
# 主流程
# ═════════════════════════════════════════════════════════════════

def run_inspection(ckpt_path: str, model_type: str, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. 加载 tokenizer ────────────────────────────────────────
    print("\n[1] 加载 REMI tokenizer ...")
    tok         = load_tokenizer()
    vocab_info  = get_vocab_info(tok)
    pitch_tokens = extract_pitch_tokens(vocab_info)

    print(f"  词表大小: {len(tok)}")
    print(f"  音高 token 数: {len(pitch_tokens)}")
    if pitch_tokens:
        print(f"  音高范围: {pitch_tokens[0][1]}  →  {pitch_tokens[-1][1]}")

    print_vocab(vocab_info, show_all=False)

    # ── 2. 随机初始化 embedding（对照组）───────────────────────────
    print("\n[2] 构建随机初始化 embedding（对照组）...")
    d_model_vae = cfg.MUSIC_VAE_CONFIG.get("enc_d_model", 256)
    d_model_tf  = cfg.MODEL_CONFIG.get("d_model", 512)
    vocab_size  = len(tok)

    d_model_random = d_model_vae if model_type == "vae" else d_model_tf
    np.random.seed(42)
    random_emb = np.random.normal(0, 0.02, (vocab_size, d_model_random)).astype(np.float32)
    print(f"  随机 embedding shape: {random_emb.shape}")

    # ── 3. 加载已训练 checkpoint ─────────────────────────────────
    print(f"\n[3] 加载已训练 checkpoint: {ckpt_path} ...")
    trained_emb = load_embedding_from_checkpoint(ckpt_path, model_type, vocab_size)

    # ── 4. 生成图表 ───────────────────────────────────────────────

    # (A) 全词表 PCA（训练后）
    print("\n[4] 绘图中 ...")
    if trained_emb is not None:
        plot_full_vocab_pca(
            trained_emb, vocab_info,
            title=f"Trained ({Path(ckpt_path).name})",
            save_path=str(out_dir / "01_full_vocab_pca_trained.png"),
        )

    # (B) 音高 token PCA — 随机 vs 训练后
    plot_pitch_pca(
        random_emb, pitch_tokens,
        title="Random Init",
        save_path=str(out_dir / "02_pitch_pca_random.png"),
    )
    if trained_emb is not None:
        plot_pitch_pca(
            trained_emb, pitch_tokens,
            title=f"Trained ({Path(ckpt_path).name})",
            save_path=str(out_dir / "03_pitch_pca_trained.png"),
        )

    # (C) 五度圈颜色 PCA
    if trained_emb is not None:
        plot_pca_cof_color(
            trained_emb, pitch_tokens,
            title=f"Trained ({Path(ckpt_path).name})",
            save_path=str(out_dir / "04_pitch_pca_cof_trained.png"),
        )

    # (D) t-SNE
    if trained_emb is not None:
        print("  计算 t-SNE（这可能需要几秒钟）...")
        plot_pitch_tsne(
            trained_emb, pitch_tokens,
            title=f"Trained ({Path(ckpt_path).name})",
            save_path=str(out_dir / "05_pitch_tsne_trained.png"),
        )

    # (E) 余弦相似度热力图
    plot_cosine_heatmap(
        random_emb, pitch_tokens,
        title="Random Init",
        save_path=str(out_dir / "06_cosine_heatmap_random.png"),
    )
    if trained_emb is not None:
        plot_cosine_heatmap(
            trained_emb, pitch_tokens,
            title=f"Trained ({Path(ckpt_path).name})",
            save_path=str(out_dir / "07_cosine_heatmap_trained.png"),
        )

    # (F) 音程相似度柱状图
    plot_interval_similarity(
        random_emb, pitch_tokens,
        title="Random Init",
        save_path=str(out_dir / "08_interval_sim_random.png"),
    )
    if trained_emb is not None:
        plot_interval_similarity(
            trained_emb, pitch_tokens,
            title=f"Trained ({Path(ckpt_path).name})",
            save_path=str(out_dir / "09_interval_sim_trained.png"),
        )

    # ── 5. 打印音程相似度统计 ─────────────────────────────────────
    print_interval_stats(random_emb, pitch_tokens, "Random Init")
    if trained_emb is not None:
        print_interval_stats(trained_emb, pitch_tokens,
                             f"Trained: {Path(ckpt_path).name}")

    print(f"\n✅ 所有图表已保存至: {out_dir}")
    return trained_emb, pitch_tokens


# ═════════════════════════════════════════════════════════════════
# Entry point
# ═════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="音乐 Embedding 可视化工具")
    parser.add_argument(
        "--ckpt",
        default=str(ROOT / "checkpoints" / "music_vae_best.pt"),
        help="Checkpoint 路径（默认: checkpoints/music_vae_best.pt）",
    )
    parser.add_argument(
        "--model",
        default="vae",
        choices=["vae", "transformer"],
        help="模型类型（影响 d_model 参数推断）",
    )
    parser.add_argument(
        "--out",
        default=str(ROOT / "outputs" / "embedding_viz"),
        help="输出图表目录",
    )
    parser.add_argument(
        "--show-all-vocab",
        action="store_true",
        help="打印完整词表（所有 token）",
    )
    args = parser.parse_args()

    out_dir = Path(args.out)

    # 如果要打印完整词表
    if args.show_all_vocab:
        tok = load_tokenizer()
        vocab_info = get_vocab_info(tok)
        print_vocab(vocab_info, show_all=True)
        return

    run_inspection(
        ckpt_path  = args.ckpt,
        model_type = args.model,
        out_dir    = out_dir,
    )


if __name__ == "__main__":
    main()
