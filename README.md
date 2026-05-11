# Music Generation — 项目快速参考

## 项目概述

基于 **MAESTRO v3.0.0**（专业钢琴演奏 MIDI 数据集）训练的钢琴音乐生成系统。  
实现了两套模型：GPT 风格的 **MusicTransformer** 和层次化 VAE 的 **MusicVAE**。  
硬件目标：RTX 4060 8GB，bfloat16 混合精度。

---

## 文件结构

```
Music/
├── config.py                  # 全局配置（路径/模型/训练/生成超参）
├── train.py                   # MusicTransformer 训练入口
├── train_vae.py               # MusicVAE 训练入口
├── generate.py                # Transformer 推理（自回归采样）
├── generate_vae.py            # VAE 推理（随机采样 + 多段拼接）
├── generate_vae_prefix.py     # VAE 推理（+跨段前缀连贯）
├── generate_vae_adaptive.py   # VAE 推理（+动态 EOS 压力，最新）★
├── inspect_embeddings.py      # 潜空间可视化工具
├── model/
│   ├── transformer.py         # MusicTransformer 实现
│   └── music_vae.py           # MusicVAE 实现（Encoder/Conductor/Decoder）
├── data/
│   ├── preprocess.py          # REMI Tokenizer 构建 & 数据集 token 化
│   └── dataset.py             # PyTorch Dataset / DataLoader
├── utils/logger.py            # 训练日志 & 学习率调度
├── checkpoints/               # 保存的模型权重
├── cache/                     # tokenizer.json & 预处理后的 token 缓存（.pkl）
├── outputs/                   # 生成的 MIDI 文件
├── logs/                      # 训练日志（.jsonl）
└── maestro-v3.0.0/            # 原始 MIDI 数据集
```

---

## 模型一：MusicTransformer（~20M 参数）

**架构**：GPT 风格 Decoder-only Transformer

| 超参 | 值 |
|------|----|
| d_model | 512 |
| n_heads | 8 |
| n_layers | 6 |
| d_ff | 2048 |
| context_len | 1024 |

- **位置编码**：RoPE（无 learnable position embedding）
- **激活**：SwiGLU FFN，Pre-LayerNorm
- **注意力**：Causal，Flash Attention（PyTorch 2.0+）
- **权重共享**：LM Head ↔ Token Embedding（weight tying）

**训练**：
```bash
python train.py           # 从头训练
python train.py --resume  # 继续训练
```
AdamW，lr=3e-4，warmup 1000 步 + 余弦退火；batch=16，grad_accum=2（等效 batch 32）

**推理**：
```bash
python generate.py
```
自回归采样，top-p=0.9，temperature=1.0，repetition_penalty=1.1

**Checkpoint**：`checkpoints/music_transformer_best.pt`（已训练 6 epoch）

---

## 模型二：MusicVAE（~16M 参数）

**参考**：Roberts et al., ICML 2018 — *A Hierarchical Latent Vector Model for Learning Long-Term Structure in Music*

**三段式架构**：

| 组件 | 实现 | 关键参数 |
|------|------|---------|
| **Encoder** | 双向 Transformer（4 层） | d_model=256，输入 segment_len=256 tokens → (mu, logvar) |
| **Conductor** | GRU（2 层） | z(128-d) → 16 个 bar embedding（256-d） |
| **Decoder** | Causal Transformer + Cross-Attention（8 层） | 以 bar embeddings 为条件，自回归解码 |

**潜变量**：z ∈ ℝ¹²⁸，支持随机采样 / 编码重建 / slerp 球形插值

**训练稳定性**：
- Cyclical β-annealing（4 个完整周期，β 0→1 线性上升，ratio=0.5）
- Free bits = 0.5 nats/dim（防止 posterior collapse）

**训练**：
```bash
python train_vae.py                # 从头训练
python train_vae.py --resume       # 继续训练
python train_vae.py --preprocess   # 仅预处理数据后退出
```
AdamW，lr=3e-4，warmup 2000 步；batch=32，grad_accum=2

**Checkpoint**：`checkpoints/music_vae_best.pt`（已完成训练）

---

## VAE 推理脚本（功能递进）

| 脚本 | 特点 | 典型命令 |
|------|------|---------|
| `generate_vae.py` | 固定 n_segments 段拼接 + z 游走 | `python generate_vae.py --n 3` |
| `generate_vae_prefix.py` | 跨段携带前缀 token，提升段间连贯性 | `python generate_vae_prefix.py --prefix_len 32` |
| `generate_vae_adaptive.py` ★ | 动态 EOS 压力，曲子自然收尾（**推荐**） | `python generate_vae_adaptive.py --n 3` |

### 自适应生成机制（`generate_vae_adaptive.py`）

```
fatigue  = total_tokens / max_total_tokens   # ∈ [0, 1]
eos_bias = eos_bias_max * fatigue            # 默认 eos_bias_max=8.0
logits[eos_id] += eos_bias                  # softmax 前叠加，使 EOS 被自然采到
```

默认 `max_total_tokens=2048`（约 1–3 分钟），平均停止点自然落在目标长度附近。

**三种模式**（互斥）：
```bash
# 1. 随机采样（默认）
python generate_vae_adaptive.py --n 3 --max_total_tokens 2048

# 2. 潜空间球形插值
python generate_vae_adaptive.py --interpolate a.midi b.midi --steps 8

# 3. 重建（编码→解码）
python generate_vae_adaptive.py --reconstruct original.midi
```

**常用参数**：

| 参数 | 默认 | 说明 |
|------|------|------|
| `--max_total_tokens` | 2048 | 全局目标 token 上限 |
| `--eos_bias_max` | 8.0 | 达到上限时的 EOS 偏置强度 |
| `--prefix_len` | 16 | 跨段前缀长度（0=无前缀） |
| `--z_walk_scale` | 0.1 | 段间 z 游走幅度（0=风格固定，1=完全独立） |
| `--temperature` | 0.85 | 采样温度 |
| `--top_p` | 0.9 | Nucleus sampling 阈值 |

---

## Tokenizer（REMI）

由 **miditok** 提供，缓存至 `cache/tokenizer.json`：

- 音域：A0(21)–C8(108)，力度 32 档，速度 32 档（40–250 BPM）
- 支持延音踏板，不使用拍号（简化词表）
- 特殊 token：`PAD=0 / BOS=1 / EOS=2`
- 词表大小：约 400–450

---

## 依赖与安装

```
torch>=2.0.0, miditok>=3.0.0, pretty_midi>=0.2.9
numpy>=1.24.0, pandas>=2.0.0, tqdm>=4.65.0
```

```bash
pip install -r requirements.txt
```

---

## 快速上手

```bash
# 1. 数据预处理（首次运行，生成 cache/）
python train_vae.py --preprocess

# 2. 训练 MusicVAE
python train_vae.py

# 3. 生成 3 首钢琴曲（最推荐脚本）
python generate_vae_adaptive.py --n 3

# 输出文件位于 outputs/vae_adpt_pfx16_<timestamp>_01.midi 等
```

---

## 当前训练状态

| 模型 | 状态 | Checkpoint |
|------|------|------------|
| MusicTransformer | 已训练 6 epoch | `music_transformer_best.pt` / `epoch006.pt` |
| MusicVAE | 已完成训练 | `music_vae_best.pt` |

生成样本位于 `outputs/`，训练曲线位于 `logs/*.jsonl`。
