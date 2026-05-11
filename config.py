"""
全局配置文件
RTX 4060 8GB 优化配置
"""
import os

# ─────────────────────────────────────────────
# 路径配置
# ─────────────────────────────────────────────
BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
DATA_DIR        = os.path.join(BASE_DIR, "maestro-v3.0.0")
CSV_PATH        = os.path.join(DATA_DIR, "maestro-v3.0.0.csv")
CACHE_DIR       = os.path.join(BASE_DIR, "cache")          # 预处理后的 token 缓存
TOKENIZER_PATH  = os.path.join(CACHE_DIR, "tokenizer.json")
CHECKPOINT_DIR  = os.path.join(BASE_DIR, "checkpoints")
LOG_DIR         = os.path.join(BASE_DIR, "logs")
OUTPUT_DIR      = os.path.join(BASE_DIR, "outputs")        # 生成的 MIDI 输出

# ─────────────────────────────────────────────
# Tokenizer 配置 (REMI)
# ─────────────────────────────────────────────
TOKENIZER_CONFIG = dict(
    pitch_range     = (21, 109),          # 钢琴音域 A0(21) - C8(108)
    beat_res        = {(0, 4): 8, (4, 12): 4},  # 节拍分辨率
    nb_velocities   = 32,                 # 力度分档数
    special_tokens  = ["PAD", "BOS", "EOS"],
    use_tempos      = True,
    nb_tempos       = 32,
    tempo_range     = (40, 250),
    use_time_signatures = False,          # 不使用拍号（简化词表）
    use_sustain_pedals  = True,           # 使用延音踏板
)

# ─────────────────────────────────────────────
# 模型配置 (Decoder-only Transformer, ~20M 参数)
# ─────────────────────────────────────────────
MODEL_CONFIG = dict(
    # 词表大小在初始化 tokenizer 后自动填入
    vocab_size      = None,
    context_len     = 1024,               # 序列长度
    d_model         = 512,                # 隐层维度
    n_heads         = 8,                  # 注意力头数
    n_layers        = 6,                  # Transformer 层数
    d_ff            = 2048,               # FFN 中间维度
    dropout         = 0.1,
    pad_token_id    = 0,                  # PAD token id (由 tokenizer 确定)
)

# ─────────────────────────────────────────────
# 训练配置 (RTX 4060 8GB)
# ─────────────────────────────────────────────
TRAIN_CONFIG = dict(
    # 设备
    device          = "cuda",
    dtype           = "bfloat16",         # RTX 4060 完整支持 bfloat16

    # 批次与梯度
    batch_size      = 16,                 # 单步 batch，显存约 5-6 GB（4060 8GB 充裕）
    grad_accum_steps= 2,                  # 等效 batch_size = 32（保持不变）
    max_grad_norm   = 1.0,                # 梯度裁剪

    # 训练步数 / epoch
    max_epochs      = 100,
    save_every      = 5,                  # 每 N epoch 保存 checkpoint

    # 优化器 (AdamW)
    learning_rate   = 3e-4,
    weight_decay    = 0.1,
    betas           = (0.9, 0.95),
    eps             = 1e-8,

    # 学习率调度 (余弦退火 + warmup)
    warmup_steps    = 1000,
    lr_decay_iters  = None,               # None 则用 max_steps

    # DataLoader
    num_workers     = 0,                  # Windows spawn 开销大，0 反而更快
    pin_memory      = True,
)

# ─────────────────────────────────────────────
# MusicVAE 配置 (~16M 参数)
# ─────────────────────────────────────────────
MUSIC_VAE_CONFIG = dict(
    # 词表大小在初始化 tokenizer 后自动填入
    vocab_size       = None,

    # ── 序列 / 层次结构 ──────────────────────
    segment_len      = 256,    # 每个训练样本的 token 数
    n_bars           = 16,     # conductor 产生的 bar embedding 数量
                               # (每个 bar 平均 256/16=16 tokens)

    # ── 潜空间 ───────────────────────────────
    z_dim            = 128,    # 潜变量维度

    # ── Encoder (双向 Transformer) ───────────
    enc_d_model      = 256,
    enc_n_heads      = 8,
    enc_n_layers     = 4,
    enc_d_ff         = 1024,

    # ── Conductor (GRU) ──────────────────────
    cond_hidden      = 256,    # GRU 隐层维度（= enc_d_model）
    cond_n_layers    = 2,      # GRU 层数

    # ── Decoder (带 Cross-Attention 的 Transformer) ──
    dec_d_model      = 256,
    dec_n_heads      = 8,
    dec_n_layers     = 8,
    dec_d_ff         = 1024,

    dropout          = 0.1,
    pad_token_id     = 0,
)

# ─────────────────────────────────────────────
# MusicVAE 训练配置
# ─────────────────────────────────────────────
MUSIC_VAE_TRAIN_CONFIG = dict(
    # 设备
    device           = "cuda",
    dtype            = "bfloat16",

    # 批次
    batch_size       = 32,           # VAE 样本更短 (256 tokens)，可以用更大 batch
    grad_accum_steps = 2,
    max_grad_norm    = 1.0,

    # 训练步数
    max_epochs       = 100,
    save_every       = 5,

    # 优化器
    learning_rate    = 3e-4,
    weight_decay     = 0.1,
    betas            = (0.9, 0.95),
    eps              = 1e-8,

    # 学习率调度
    warmup_steps     = 2000,
    lr_decay_iters   = None,

    # ── KL 稳定性 ────────────────────────────
    # Cyclical β annealing: 每隔 cycle_steps 步做一次 0→1 的线性上升
    beta_cycles      = 4,            # 训练总步数内做几个完整周期
    beta_ratio       = 0.5,          # 每个周期内线性上升占比 (其余时段 β=1)
    beta_max         = 1.0,          # β 上限 (β-VAE 时可设 <1 或 >1)
    free_bits        = 0.5,          # 每维 KL 的 free bits 下限 (nats)

    # DataLoader
    num_workers      = 0,
    pin_memory       = True,
)

# ─────────────────────────────────────────────
# MusicVAE 生成配置
# ─────────────────────────────────────────────
MUSIC_VAE_GENERATE_CONFIG = dict(
    temperature      = 0.85,
    top_k            = 0,
    top_p            = 0.9,
    max_decode_steps = 256,    # 单段最多解码 token 数（= segment_len）

    # ── 多段拼接 (Multi-Segment Stitching) ───────────────────────
    # 每首输出 = n_segments 段拼接，总长约 n_segments × max_decode_steps tokens
    n_segments       = 8,      # 默认拼 8 段 ≈ 2048 tokens ≈ 1~3 分钟
    # z 游走幅度：0.0=完全相同风格, 0.1=缓慢自然变化(推荐), 1.0=完全独立
    z_walk_scale     = 0.1,
)

# ─────────────────────────────────────────────
# 生成配置（GPT Transformer，保留原有）
# ─────────────────────────────────────────────
GENERATE_CONFIG = dict(
    max_new_tokens  = 1024,               # 最多生成多少 token
    temperature     = 1.0,               # 采样温度
    top_k           = 0,                  # top-k (0 = 不使用)
    top_p           = 0.9,                # nucleus sampling
    repetition_penalty = 1.1,            # 重复惩罚
)
