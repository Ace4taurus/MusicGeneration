"""
generation_test.py —— Uniform Bar Embedding 生成实验

实验目的：
  验证 Decoder 是否具备"相同条件信号 → 复现相同旋律"的能力。

  正常生成中，Conductor (GRU) 每步输出不同的 bar embedding，
  导致 Decoder 收到的条件信号在各 bar 间持续变化，难以产生主旋律复现。

  本实验将所有 bar embedding 强制设为同一向量（uniform），
  观察 Decoder 在条件信号恒定时是否能生成具有主旋律复现的片段。

推断逻辑：
  uniform bar embedding → 有主旋律复现
    ⟹ Decoder 已具备"相同信号 → 复现旋律"的能力
    ⟹ 瓶颈在 Conductor：若给 Conductor 加鼓励相似性的损失，
       可让 Decoder 在主题重复段产出更相近的 bar embedding，
       从而激发主旋律复现能力

用法:
  python generation_test.py --n 3 --bar_mode mean

  # 不同 bar 代表向量策略对比
  python generation_test.py --n 2 --bar_mode first
  python generation_test.py --n 2 --bar_mode last
  python generation_test.py --n 2 --bar_mode random

  # 控制长度
  python generation_test.py --n 3 --n_segments 12 --z_walk_scale 0.0
"""

import os
import argparse
from datetime import datetime
from typing import List

import torch

import config as cfg
from data.preprocess import build_tokenizer
from model.music_vae import MusicVAE


# ─────────────────────────────────────────────
# 模型加载
# ─────────────────────────────────────────────

def load_model(checkpoint_path: str, device: str):
    tokenizer  = build_tokenizer(force_rebuild=False)
    vocab_size = len(tokenizer)
    pad_id     = tokenizer["PAD_None"]
    bos_id     = tokenizer["BOS_None"]
    eos_id     = tokenizer["EOS_None"]

    vc = cfg.MUSIC_VAE_CONFIG
    model = MusicVAE(
        vocab_size    = vocab_size,
        segment_len   = vc["segment_len"],
        n_bars        = vc["n_bars"],
        z_dim         = vc["z_dim"],
        enc_d_model   = vc["enc_d_model"],
        enc_n_heads   = vc["enc_n_heads"],
        enc_n_layers  = vc["enc_n_layers"],
        enc_d_ff      = vc["enc_d_ff"],
        cond_hidden   = vc["cond_hidden"],
        cond_n_layers = vc["cond_n_layers"],
        dec_d_model   = vc["dec_d_model"],
        dec_n_heads   = vc["dec_n_heads"],
        dec_n_layers  = vc["dec_n_layers"],
        dec_d_ff      = vc["dec_d_ff"],
        dropout       = 0.0,
        pad_token_id  = pad_id,
        free_bits     = 0.0,
    ).to(device)

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"checkpoint 不存在: {checkpoint_path}")

    state = torch.load(checkpoint_path, map_location=device, weights_only=True)
    if "model" in state:
        state = state["model"]
    state = {k.replace("_orig_mod.", ""): v for k, v in state.items()}
    model.load_state_dict(state, strict=False)
    model.eval()
    print(f"[Test] 模型已加载: {checkpoint_path}")
    return model, tokenizer, bos_id, eos_id, pad_id


# ─────────────────────────────────────────────
# MIDI 输出
# ─────────────────────────────────────────────

def tokens_to_midi(token_ids: List[int], tokenizer, output_path: str):
    from miditok import TokSequence

    special_ids = set()
    for name in ["PAD_None", "BOS_None", "EOS_None"]:
        try:
            special_ids.add(tokenizer[name])
        except KeyError:
            pass

    clean_ids = [t for t in token_ids if t not in special_ids]
    if len(clean_ids) == 0:
        print(f"[警告] token 列表为空，跳过: {output_path}")
        return

    tok_seq = TokSequence(ids=clean_ids)
    tokenizer.complete_sequence(tok_seq)
    midi = tokenizer.tokens_to_midi([tok_seq])

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    midi.dump_midi(output_path)
    print(f"[Test] MIDI 已保存: {output_path}")


# ─────────────────────────────────────────────
# z 游走
# ─────────────────────────────────────────────

def z_walk_step(z: torch.Tensor, scale: float) -> torch.Tensor:
    if scale <= 0.0:
        return z
    noise     = torch.randn_like(z) * scale
    z_new     = z + noise
    norm_orig = z.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    norm_new  = z_new.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    return z_new / norm_new * norm_orig


# ─────────────────────────────────────────────
# 核心：uniform bar embedding 解码
# ─────────────────────────────────────────────

def pick_representative_bar(bar_embs: torch.Tensor, bar_mode: str) -> torch.Tensor:
    """
    从 (1, n_bars, cond_hidden) 中选取一个代表向量。

    bar_mode:
      'mean'   — 所有 bar embedding 的均值（默认，最"平均"的风格信号）
      'first'  — 第 0 个 bar（GRU 初始输出，最靠近 z 的初始信号）
      'last'   — 最后一个 bar（GRU 最终状态，积累了所有步的信息）
      'random' — 随机选一个 bar

    Returns:
      rep: (1, 1, cond_hidden)
    """
    if bar_mode == "mean":
        rep = bar_embs.mean(dim=1, keepdim=True)          # (1, 1, cond_hidden)
    elif bar_mode == "first":
        rep = bar_embs[:, :1, :]                           # (1, 1, cond_hidden)
    elif bar_mode == "last":
        rep = bar_embs[:, -1:, :]                          # (1, 1, cond_hidden)
    elif bar_mode == "random":
        idx = torch.randint(0, bar_embs.size(1), (1,)).item()
        rep = bar_embs[:, idx : idx + 1, :]                # (1, 1, cond_hidden)
    else:
        raise ValueError(f"未知 bar_mode: {bar_mode}，可选: mean / first / last / random")
    return rep


@torch.no_grad()
def decode_uniform_bars(
    model:       MusicVAE,
    z:           torch.Tensor,
    bos_id:      int,
    eos_id:      int,
    max_steps:   int,
    temperature: float,
    top_k:       int,
    top_p:       float,
    bar_mode:    str = "mean",
) -> List[int]:
    """
    将 z 通过 conductor 得到 bar_embs，
    然后把所有 bar 替换成同一个代表向量，再送入 decoder 解码。

    这是本实验的核心操作：消除 conductor 的"层次变化"，
    保留其"全局风格信号"，观察 decoder 的主旋律复现能力。
    """
    # Step 1: 正常计算 bar_embs（让 conductor 先自由运行）
    bar_embs = model.conductor(z)                          # (1, n_bars, cond_hidden)

    # Step 2: 选取代表向量并展开为 uniform bar_embs
    rep = pick_representative_bar(bar_embs, bar_mode)      # (1, 1, cond_hidden)
    n_bars = bar_embs.size(1)
    uniform_bar_embs = rep.expand(-1, n_bars, -1)          # (1, n_bars, cond_hidden)
    # 注意：expand 不复制内存，需 contiguous 才能安全传递
    uniform_bar_embs = uniform_bar_embs.contiguous()

    # Step 3: 用 uniform bar_embs 解码
    ids_tensor = model.decoder.generate(
        bar_embs    = uniform_bar_embs,
        bos_id      = bos_id,
        eos_id      = eos_id,
        max_steps   = max_steps,
        temperature = temperature,
        top_k       = top_k,
        top_p       = top_p,
    )                                                      # (1, T)

    ids = ids_tensor[0].cpu().tolist()
    ids = [t for t in ids if t not in {bos_id, eos_id}]
    return ids


# ─────────────────────────────────────────────
# 多段拼接采样
# ─────────────────────────────────────────────

@torch.no_grad()
def sample_uniform_long(
    model:        MusicVAE,
    n:            int,
    bos_id:       int,
    eos_id:       int,
    device:       str,
    bar_mode:     str   = "mean",
    n_segments:   int   = 8,
    z_walk_scale: float = 0.1,
    temperature:  float = 0.85,
    top_k:        int   = 0,
    top_p:        float = 0.9,
    max_steps:    int   = 256,
) -> List[List[int]]:
    """
    生成 n 首曲目，每首由 n_segments 段拼接。

    每段都使用 uniform bar embedding（所有 bar 相同），
    段间 z 做小步游走以产生自然的风格渐变。

    Args:
        bar_mode:     代表 bar 的选取策略（mean/first/last/random）
        z_walk_scale: 段间游走幅度（0.0=风格完全固定，>0=缓慢演变）
    """
    model.eval()
    z_all = torch.randn(n, model.z_dim, device=device)

    results = []
    for song_idx in range(n):
        z_cur        = z_all[song_idx : song_idx + 1].clone()  # (1, z_dim)
        song_tokens: List[int] = []

        for seg_idx in range(n_segments):
            seg_tokens = decode_uniform_bars(
                model       = model,
                z           = z_cur,
                bos_id      = bos_id,
                eos_id      = eos_id,
                max_steps   = max_steps,
                temperature = temperature,
                top_k       = top_k,
                top_p       = top_p,
                bar_mode    = bar_mode,
            )
            song_tokens.extend(seg_tokens)
            z_cur = z_walk_step(z_cur, z_walk_scale)

            print(f"  曲目 #{song_idx+1} 段 {seg_idx+1}/{n_segments}: "
                  f"+{len(seg_tokens)} tokens（累计 {len(song_tokens)}）")

        results.append(song_tokens)

    return results


# ─────────────────────────────────────────────
# 命令行参数
# ─────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    gc = cfg.MUSIC_VAE_GENERATE_CONFIG

    parser = argparse.ArgumentParser(
        description="Uniform Bar Embedding 生成实验",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
实验说明:
  将所有 bar embedding 强制设为同一向量，观察 decoder 是否仍能产生主旋律复现。

  若 uniform 条件下出现主旋律复现：
    => Decoder 已具备相同条件信号→复现旋律的能力
    => 给 Conductor 加相似性损失可进一步激发此能力

示例:
  # 默认：用均值 bar embedding，生成 3 首
  python generation_test.py --n 3

  # 风格固定（z 不游走），对比最纯粹的 uniform 效果
  python generation_test.py --n 3 --z_walk_scale 0.0

  # 对比不同 bar 代表策略
  python generation_test.py --n 2 --bar_mode first
  python generation_test.py --n 2 --bar_mode last
  python generation_test.py --n 2 --bar_mode random

  # 生成更长的片段（16 段）
  python generation_test.py --n 2 --n_segments 16
        """
    )

    parser.add_argument(
        "--checkpoint", "-c",
        type    = str,
        default = os.path.join(cfg.CHECKPOINT_DIR, "music_vae_best.pt"),
        help    = "模型 checkpoint 路径",
    )
    parser.add_argument(
        "--n",
        type    = int, default = 3,
        help    = "生成首数（默认 3）",
    )
    parser.add_argument(
        "--bar_mode",
        type    = str, default = "mean",
        choices = ["mean", "first", "last", "random"],
        help    = (
            "所有 bar 共用的代表向量选取策略（默认 mean）\n"
            "  mean   — conductor 输出的所有 bar 取均值\n"
            "  first  — 使用第 0 个 bar embedding\n"
            "  last   — 使用最后一个 bar embedding\n"
            "  random — 随机选一个 bar embedding"
        ),
    )
    parser.add_argument(
        "--n_segments",
        type    = int, default = None,
        help    = f"每首由多少段拼成（默认 {gc['n_segments']}）",
    )
    parser.add_argument(
        "--z_walk_scale",
        type    = float, default = None,
        help    = (
            f"段间 z 游走幅度（默认 {gc['z_walk_scale']}）\n"
            "  0.0=风格完全固定  0.1=缓慢演变  1.0=完全随机"
        ),
    )
    parser.add_argument(
        "--temperature",
        type    = float, default = None,
        help    = f"采样温度（默认 {gc['temperature']}）",
    )
    parser.add_argument(
        "--top_k",
        type    = int, default = None,
        help    = "Top-k 截断（0 = 不使用）",
    )
    parser.add_argument(
        "--top_p",
        type    = float, default = None,
        help    = f"Nucleus sampling 阈值（默认 {gc['top_p']}）",
    )
    parser.add_argument(
        "--max_tokens",
        type    = int, default = None,
        help    = f"每段最多 token 数（默认 {gc['max_decode_steps']}）",
    )

    return parser


# ─────────────────────────────────────────────
# 入口
# ─────────────────────────────────────────────

if __name__ == "__main__":
    parser = build_parser()
    args   = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer, bos_id, eos_id, pad_id = load_model(args.checkpoint, device)

    gc           = cfg.MUSIC_VAE_GENERATE_CONFIG
    bar_mode     = args.bar_mode
    n_segments   = args.n_segments   if args.n_segments   is not None else gc["n_segments"]
    z_walk_scale = args.z_walk_scale if args.z_walk_scale is not None else gc["z_walk_scale"]
    temperature  = args.temperature  if args.temperature  is not None else gc["temperature"]
    top_k        = args.top_k        if args.top_k        is not None else gc["top_k"]
    top_p        = args.top_p        if args.top_p        is not None else gc["top_p"]
    max_steps    = args.max_tokens   if args.max_tokens   is not None else gc["max_decode_steps"]

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    print(f"\n[Test] === Uniform Bar Embedding 实验 ===")
    print(f"  bar_mode     : {bar_mode}")
    print(f"  n_segments   : {n_segments}")
    print(f"  z_walk_scale : {z_walk_scale}")
    print(f"  temperature  : {temperature}  top_k={top_k}  top_p={top_p}")
    print(f"  预计每首约 {n_segments * max_steps} tokens\n")

    sequences = sample_uniform_long(
        model        = model,
        n            = args.n,
        bos_id       = bos_id,
        eos_id       = eos_id,
        device       = device,
        bar_mode     = bar_mode,
        n_segments   = n_segments,
        z_walk_scale = z_walk_scale,
        temperature  = temperature,
        top_k        = top_k,
        top_p        = top_p,
        max_steps    = max_steps,
    )

    for i, seq in enumerate(sequences):
        out_path = os.path.join(
            cfg.OUTPUT_DIR,
            f"vae_test_uniform_{bar_mode}_{ts}_{i+1:02d}.midi"
        )
        print(f"\n  曲目 #{i+1}: 共 {len(seq)} tokens → {out_path}")
        tokens_to_midi(seq, tokenizer, out_path)

    print(f"\n[Test] 完成，输出目录: {cfg.OUTPUT_DIR}")
    print(f"[Test] 提示：对比 generate_vae.py 的正常生成结果，")
    print(f"[Test]       观察主旋律复现频率的差异。")
