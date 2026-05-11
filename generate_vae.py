"""
MusicVAE 生成脚本
支持三种模式:

  1. 随机采样 (默认)
     从 N(0,I) 随机采样起始 z，然后在 z 空间游走生成多段并拼接成完整曲目
     python generate_vae.py --checkpoint checkpoints/music_vae_best.pt --n 3

     控制长度：
       --n_segments 16       生成 16 段（更长）
       --z_walk_scale 0.2    z 游走步幅（0=同一风格重复，1=完全随机）

  2. 潜空间插值 (--interpolate)
     在两首曲子的潜编码之间做球形线性插值（slerp），生成过渡序列
     python generate_vae.py --checkpoint checkpoints/music_vae_best.pt ^
         --interpolate midi_A.midi midi_B.midi --steps 8

  3. 重建 (--reconstruct)
     将 MIDI 编码再解码，验证重建质量
     python generate_vae.py --checkpoint checkpoints/music_vae_best.pt ^
         --reconstruct original.midi
"""

import os
import argparse
from pathlib import Path
from datetime import datetime
from typing import List

import torch
import torch.nn.functional as F

import config as cfg
from data.preprocess import build_tokenizer
from model.music_vae import MusicVAE


# ─────────────────────────────────────────────
# 工具函数
# ─────────────────────────────────────────────

def load_model(checkpoint_path: str, device: str):
    """
    加载 MusicVAE 模型和 tokenizer。

    Returns:
        (model, tokenizer, bos_id, eos_id, pad_id)
    """
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
    print(f"[Generate] 模型已加载: {checkpoint_path}")
    return model, tokenizer, bos_id, eos_id, pad_id


def midi_to_tokens(midi_path: str, tokenizer, pad_id: int, segment_len: int) -> torch.Tensor:
    """
    将 MIDI 文件 tokenize 并截取/补齐到 segment_len。
    Returns: (1, segment_len) LongTensor
    """
    tok_seqs = tokenizer(midi_path)
    ids      = tok_seqs[0].ids
    bos_id   = tokenizer["BOS_None"]
    ids      = [bos_id] + ids
    if len(ids) >= segment_len:
        ids = ids[:segment_len]
    else:
        ids = ids + [pad_id] * (segment_len - len(ids))
    return torch.tensor([ids], dtype=torch.long)


def tokens_to_midi(token_ids: List[int], tokenizer, output_path: str):
    """将 token id 列表转换为 MIDI 文件，自动去除特殊 token。"""
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
    print(f"[Generate] MIDI 已保存: {output_path}")


# ─────────────────────────────────────────────
# z 游走辅助函数
# ─────────────────────────────────────────────

def z_walk_step(z: torch.Tensor, scale: float) -> torch.Tensor:
    """
    在 z 空间做一步随机游走：
      z_new = normalize(z + noise * scale) * ‖z‖
    保持向量模长不变，方向缓慢漂移。
    scale=0.0 → 原地不动；scale=1.0 → 大幅跳跃。
    """
    if scale <= 0.0:
        return z
    noise   = torch.randn_like(z) * scale
    z_new   = z + noise
    # 保持模长（让 z 停留在 N(0,I) 的"合理域"内）
    norm_orig = z.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    norm_new  = z_new.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    z_new     = z_new / norm_new * norm_orig
    return z_new


def decode_one_segment(
    model:       MusicVAE,
    z:           torch.Tensor,
    bos_id:      int,
    eos_id:      int,
    max_steps:   int,
    temperature: float,
    top_k:       int,
    top_p:       float,
) -> List[int]:
    """
    从单个 z 解码一段 token 序列（不含 BOS/EOS）。
    """
    bar_embs = model.conductor(z)                  # (1, n_bars, cond_hidden)
    ids_tensor = model.decoder.generate(
        bar_embs    = bar_embs,
        bos_id      = bos_id,
        eos_id      = eos_id,
        max_steps   = max_steps,
        temperature = temperature,
        top_k       = top_k,
        top_p       = top_p,
    )                                              # (1, T)
    ids = ids_tensor[0].cpu().tolist()
    # 去除首尾特殊 token
    ids = [t for t in ids if t not in {bos_id, eos_id}]
    return ids


# ─────────────────────────────────────────────
# 多段拼接采样
# ─────────────────────────────────────────────

@torch.no_grad()
def sample_long(
    model:        MusicVAE,
    n:            int,
    bos_id:       int,
    eos_id:       int,
    device:       str,
    n_segments:   int   = 8,
    z_walk_scale: float = 0.1,
    temperature:  float = 0.85,
    top_k:        int   = 0,
    top_p:        float = 0.9,
    max_steps:    int   = 256,
    z_init:       torch.Tensor = None,
) -> List[List[int]]:
    """
    生成 n 首完整曲目，每首由 n_segments 个 VAE 段落拼接而成。

    z 游走策略：
      每段结束后，z 沿随机方向小步游走（walk_scale 控制步幅），
      保持模长不变，使风格连续自然演变。

    Args:
        n:            生成首数
        n_segments:   每首包含的段落数（段数 × max_steps ≈ 总 token 数）
        z_walk_scale: z 游走幅度 (0=不变, 0.1=推荐, 1=完全独立)
        z_init:       (n, z_dim) 可选初始 z；None 时随机采样

    Returns:
        list of token id lists，每条为一首完整曲子的 token 序列
    """
    model.eval()
    z_dim = model.z_dim

    # 初始 z：每首一个
    if z_init is not None:
        z_all = z_init.to(device)
    else:
        z_all = torch.randn(n, z_dim, device=device)

    results = []
    for song_idx in range(n):
        z_cur     = z_all[song_idx : song_idx + 1].clone()  # (1, z_dim)
        song_tokens: List[int] = []

        for seg_idx in range(n_segments):
            seg_tokens = decode_one_segment(
                model, z_cur, bos_id, eos_id, max_steps, temperature, top_k, top_p
            )
            song_tokens.extend(seg_tokens)

            # z 游走到下一段
            z_cur = z_walk_step(z_cur, z_walk_scale)

            n_so_far = len(song_tokens)
            print(f"  曲目 #{song_idx+1} 段 {seg_idx+1}/{n_segments}: "
                  f"+{len(seg_tokens)} tokens (累计 {n_so_far})")

        results.append(song_tokens)

    return results


# ─────────────────────────────────────────────
# 三种生成模式
# ─────────────────────────────────────────────

def cmd_sample(args):
    """模式 1：从 N(0,I) 采样并多段拼接"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer, bos_id, eos_id, pad_id = load_model(args.checkpoint, device)

    gc           = cfg.MUSIC_VAE_GENERATE_CONFIG
    temp         = args.temperature  if args.temperature  is not None else gc["temperature"]
    top_k        = args.top_k        if args.top_k        is not None else gc["top_k"]
    top_p        = args.top_p        if args.top_p        is not None else gc["top_p"]
    max_step     = args.max_tokens   if args.max_tokens   is not None else gc["max_decode_steps"]
    n_segments   = args.n_segments   if args.n_segments   is not None else gc["n_segments"]
    z_walk_scale = args.z_walk_scale if args.z_walk_scale is not None else gc["z_walk_scale"]

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    print(f"\n[Generate] 随机采样 {args.n} 首 × {n_segments} 段（z_walk={z_walk_scale}）")
    print(f"  预计每首约 {n_segments * max_step} tokens")

    sequences = sample_long(
        model        = model,
        n            = args.n,
        bos_id       = bos_id,
        eos_id       = eos_id,
        device       = device,
        n_segments   = n_segments,
        z_walk_scale = z_walk_scale,
        temperature  = temp,
        top_k        = top_k,
        top_p        = top_p,
        max_steps    = max_step,
    )

    for i, seq in enumerate(sequences):
        out_path = os.path.join(cfg.OUTPUT_DIR, f"vae_sample_{ts}_{i+1:02d}.midi")
        print(f"\n  曲目 #{i+1}: 共 {len(seq)} tokens → {out_path}")
        tokens_to_midi(seq, tokenizer, out_path)

    print(f"\n[Generate] 完成，输出目录: {cfg.OUTPUT_DIR}")


def cmd_interpolate(args):
    """模式 2：在两首 MIDI 的潜编码间做球形线性插值，每个插值点也多段拼接"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer, bos_id, eos_id, pad_id = load_model(args.checkpoint, device)

    midi_a, midi_b = args.interpolate
    seg_len = cfg.MUSIC_VAE_CONFIG["segment_len"]

    for p in (midi_a, midi_b):
        if not os.path.exists(p):
            raise FileNotFoundError(f"MIDI 文件不存在: {p}")

    x1 = midi_to_tokens(midi_a, tokenizer, pad_id, seg_len).to(device)
    x2 = midi_to_tokens(midi_b, tokenizer, pad_id, seg_len).to(device)

    gc           = cfg.MUSIC_VAE_GENERATE_CONFIG
    temp         = args.temperature  if args.temperature  is not None else gc["temperature"]
    top_p        = args.top_p        if args.top_p        is not None else gc["top_p"]
    max_step     = args.max_tokens   if args.max_tokens   is not None else gc["max_decode_steps"]
    n_segments   = args.n_segments   if args.n_segments   is not None else gc["n_segments"]
    z_walk_scale = args.z_walk_scale if args.z_walk_scale is not None else gc["z_walk_scale"]
    steps        = args.steps

    print(f"\n[Generate] 插值 {steps} 步: {Path(midi_a).name} → {Path(midi_b).name}")
    print(f"  每步 {n_segments} 段，z_walk={z_walk_scale}")

    # 编码端点
    mu1, _ = model.encoder(x1)
    mu2, _ = model.encoder(x2)

    def slerp(t: float, v0: torch.Tensor, v1: torch.Tensor) -> torch.Tensor:
        v0n = F.normalize(v0, dim=-1)
        v1n = F.normalize(v1, dim=-1)
        dot = (v0n * v1n).sum(dim=-1, keepdim=True).clamp(-1, 1)
        theta     = torch.acos(dot)
        sin_theta = torch.sin(theta)
        mask = sin_theta.abs() < 1e-6
        w0 = torch.where(mask, torch.tensor(1 - t, device=v0.device),
                         torch.sin((1 - t) * theta) / (sin_theta + 1e-8))
        w1 = torch.where(mask, torch.tensor(t,     device=v0.device),
                         torch.sin(t * theta)       / (sin_theta + 1e-8))
        return w0 * v0 + w1 * v1

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    for i in range(steps):
        t  = i / max(steps - 1, 1)
        z  = slerp(t, mu1, mu2)  # (1, z_dim)
        print(f"\n  t={t:.2f}...")
        seqs = sample_long(
            model        = model,
            n            = 1,
            bos_id       = bos_id,
            eos_id       = eos_id,
            device       = device,
            n_segments   = n_segments,
            z_walk_scale = z_walk_scale,
            temperature  = temp,
            top_k        = 0,
            top_p        = top_p,
            max_steps    = max_step,
            z_init       = z,
        )
        out_path = os.path.join(
            cfg.OUTPUT_DIR, f"vae_interp_{ts}_{i+1:02d}_t{t:.2f}.midi"
        )
        tokens_to_midi(seqs[0], tokenizer, out_path)

    print(f"\n[Generate] 插值完成，输出目录: {cfg.OUTPUT_DIR}")


def cmd_reconstruct(args):
    """模式 3：编码后再解码（重建验证）"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer, bos_id, eos_id, pad_id = load_model(args.checkpoint, device)

    midi_path = args.reconstruct
    if not os.path.exists(midi_path):
        raise FileNotFoundError(f"MIDI 文件不存在: {midi_path}")

    seg_len = cfg.MUSIC_VAE_CONFIG["segment_len"]
    x       = midi_to_tokens(midi_path, tokenizer, pad_id, seg_len).to(device)

    gc           = cfg.MUSIC_VAE_GENERATE_CONFIG
    temp         = args.temperature  if args.temperature  is not None else gc["temperature"]
    top_p        = args.top_p        if args.top_p        is not None else gc["top_p"]
    max_step     = args.max_tokens   if args.max_tokens   is not None else gc["max_decode_steps"]
    n_segments   = args.n_segments   if args.n_segments   is not None else gc["n_segments"]
    z_walk_scale = args.z_walk_scale if args.z_walk_scale is not None else gc["z_walk_scale"]

    print(f"\n[Generate] 重建: {Path(midi_path).name}，{n_segments} 段")

    # 编码 → 得到 z
    mu, logvar = model.encoder(x)
    z = mu if not args.stochastic else MusicVAE.reparameterize(mu, logvar)

    seqs = sample_long(
        model        = model,
        n            = 1,
        bos_id       = bos_id,
        eos_id       = eos_id,
        device       = device,
        n_segments   = n_segments,
        z_walk_scale = z_walk_scale,
        temperature  = temp,
        top_k        = 0,
        top_p        = top_p,
        max_steps    = max_step,
        z_init       = z,
    )

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem     = Path(midi_path).stem
    out_path = os.path.join(cfg.OUTPUT_DIR, f"vae_recon_{ts}_{stem}.midi")
    tokens_to_midi(seqs[0], tokenizer, out_path)

    print(f"\n[Generate] 重建完成，输出: {out_path}")


# ─────────────────────────────────────────────
# 命令行参数
# ─────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    gc = cfg.MUSIC_VAE_GENERATE_CONFIG

    parser = argparse.ArgumentParser(
        description="MusicVAE 生成工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
示例:
  # 随机采样 3 首（默认 {gc['n_segments']} 段，约 {gc['n_segments'] * gc['max_decode_steps']} tokens/首）
  python generate_vae.py --n 3

  # 生成更长的曲子（16 段）
  python generate_vae.py --n 2 --n_segments 16

  # 风格变化更大（z 游走步幅 0.3）
  python generate_vae.py --n_segments 12 --z_walk_scale 0.3

  # 两首 MIDI 间插值 8 步
  python generate_vae.py --interpolate a.midi b.midi --steps 8 --n_segments 4

  # 重建原曲
  python generate_vae.py --reconstruct original.midi --n_segments 4
        """
    )

    # ── 通用参数 ──────────────────────────────────────────────────
    parser.add_argument(
        "--checkpoint", "-c",
        type    = str,
        default = os.path.join(cfg.CHECKPOINT_DIR, "music_vae_best.pt"),
        help    = "模型 checkpoint 路径",
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

    # ── 多段参数 ──────────────────────────────────────────────────
    parser.add_argument(
        "--n_segments",
        type    = int, default = None,
        help    = f"每首由多少段拼成（默认 {gc['n_segments']}，越大越长）",
    )
    parser.add_argument(
        "--z_walk_scale",
        type    = float, default = None,
        help    = (f"段间 z 游走幅度（默认 {gc['z_walk_scale']}）\n"
                   "  0.0=风格固定  0.1=缓慢演变  0.5=明显变化  1.0=完全随机"),
    )

    # ── 模式 1: 随机采样 ──────────────────────────────────────────
    parser.add_argument(
        "--n",
        type    = int, default = 3,
        help    = "[采样] 生成首数（默认 3）",
    )

    # ── 模式 2: 插值 ──────────────────────────────────────────────
    parser.add_argument(
        "--interpolate",
        type    = str, nargs = 2, metavar = ("MIDI_A", "MIDI_B"),
        default = None,
        help    = "[插值] 两个 MIDI 文件路径",
    )
    parser.add_argument(
        "--steps",
        type    = int, default = 8,
        help    = "[插值] 插值步数（含两端点，默认 8）",
    )

    # ── 模式 3: 重建 ──────────────────────────────────────────────
    parser.add_argument(
        "--reconstruct",
        type    = str, default = None, metavar = "MIDI",
        help    = "[重建] 要重建的 MIDI 文件路径",
    )
    parser.add_argument(
        "--stochastic",
        action  = "store_true",
        help    = "[重建] 使用随机采样（默认用确定性 mu）",
    )

    return parser


# ─────────────────────────────────────────────
# 入口
# ─────────────────────────────────────────────

if __name__ == "__main__":
    parser = build_parser()
    args   = parser.parse_args()

    if args.interpolate:
        cmd_interpolate(args)
    elif args.reconstruct:
        cmd_reconstruct(args)
    else:
        cmd_sample(args)
