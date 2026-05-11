"""
MusicVAE 生成脚本（前缀携带版）
=====================================
与 generate_vae.py 的唯一区别：段与段之间携带「前缀 token」。

原版（generate_vae.py）：
  每段从 [BOS] 重新开始，段间仅靠 z 游走保持风格相似。
  拼接点处音符可能跳变。

本版（generate_vae_prefix.py）：
  每段从 [BOS] + [上一段末尾 prefix_len 个 token] 开始解码，
  Decoder 能"看到"上段结尾，自然续写，拼接处更流畅。

  - prefix 中的特殊 token（BOS/EOS/PAD）会被过滤，不会带入下一段
  - 若上一段输出为空或极短，退化为只有 BOS（与原版等价）
  - 总输入长度 = 1 + prefix_len + max_steps ≤ RoPE 上限(512)

用法示例

  # 随机采样 3 首，前缀长度 32（默认）
  python generate_vae_prefix.py --n 3

  # 前缀长度 16（更短，减少"被上一段带跑"的风险）
  python generate_vae_prefix.py --n 3 --prefix_len 16

  # 禁用前缀（与原版完全等价，用于对比）
  python generate_vae_prefix.py --n 3 --prefix_len 0

  # 生成更长的曲子（16 段）
  python generate_vae_prefix.py --n 2 --n_segments 16 --prefix_len 32

  # 插值模式（同样支持前缀）
  python generate_vae_prefix.py --interpolate a.midi b.midi --steps 8

  # 重建模式
  python generate_vae_prefix.py --reconstruct original.midi
"""

import os
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Optional

import torch
import torch.nn.functional as F

import config as cfg
from data.preprocess import build_tokenizer
from model.music_vae import MusicVAE


# ─────────────────────────────────────────────
# 工具函数（与 generate_vae.py 完全相同）
# ─────────────────────────────────────────────

def load_model(checkpoint_path: str, device: str):
    """加载 MusicVAE 模型和 tokenizer。"""
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
    """将 MIDI 文件 tokenize 并截取/补齐到 segment_len。"""
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
# z 游走（与 generate_vae.py 完全相同）
# ─────────────────────────────────────────────

def z_walk_step(z: torch.Tensor, scale: float) -> torch.Tensor:
    """
    z 空间随机游走一步：
      z_new = normalize(z + noise * scale) * ‖z‖
    保持模长不变，方向缓慢漂移。
    """
    if scale <= 0.0:
        return z
    noise     = torch.randn_like(z) * scale
    z_new     = z + noise
    norm_orig = z.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    norm_new  = z_new.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    z_new     = z_new / norm_new * norm_orig
    return z_new


# ─────────────────────────────────────────────
# ★ 核心改动：带前缀的单段解码
# ─────────────────────────────────────────────

def decode_one_segment_with_prefix(
    model:        MusicVAE,
    z:            torch.Tensor,
    bos_id:       int,
    eos_id:       int,
    pad_id:       int,
    max_steps:    int,
    temperature:  float,
    top_k:        int,
    top_p:        float,
    prefix_ids:   Optional[List[int]] = None,
) -> List[int]:
    """
    从单个 z 解码一段 token 序列（不含 BOS/EOS/PAD）。

    与原版 decode_one_segment 的区别：
      - 接受 prefix_ids（上一段末尾的干净 token）
      - 初始序列为 [BOS] + prefix_ids，而非单独 [BOS]
      - 只返回 prefix 之后新生成的 token（prefix 已在上一段输出中）

    前缀安全保证：
      - prefix_ids 中已不含 BOS/EOS/PAD（由调用方保证）
      - 若 prefix_ids 为空或 None，退化为原版行为
      - 总解码长度 ≤ 1 + len(prefix_ids) + max_steps ≤ RoPE 上限

    Args:
        prefix_ids: 上一段末尾的 token id 列表（已过滤特殊 token）
                    None 或 [] 表示不使用前缀（等价于原版）
    """
    special_ids = {bos_id, eos_id, pad_id}

    # ── 构建初始序列 ──────────────────────────────────────────────
    clean_prefix = [t for t in (prefix_ids or []) if t not in special_ids]
    init_ids     = [bos_id] + clean_prefix           # [BOS] + 前缀
    n_prefix     = len(clean_prefix)                 # 前缀长度（不含 BOS）

    device = z.device
    ids    = torch.tensor([init_ids], dtype=torch.long, device=device)

    # ── 获取 bar embeddings ───────────────────────────────────────
    bar_embs = model.conductor(z)                    # (1, n_bars, cond_hidden)

    # ── 自回归生成（复用与 HierarchicalTransformerDecoder.generate 相同的采样逻辑）─
    for _ in range(max_steps):
        logits = model.decoder(ids, bar_embs)        # (1, T, vocab_size)
        logits = logits[:, -1, :]                    # (1, vocab_size) —— 只取最后位置

        # 温度缩放
        logits = logits / max(temperature, 1e-8)

        # Top-k 截断
        if top_k > 0:
            top_k_val = min(top_k, logits.size(-1))
            values, _ = torch.topk(logits, top_k_val)
            logits[logits < values[:, -1:]] = float('-inf')

        # Nucleus (top-p) 截断
        if top_p < 1.0:
            sorted_logits, sorted_idx = torch.sort(logits, descending=True)
            cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            remove    = cum_probs - F.softmax(sorted_logits, dim=-1) > top_p
            sorted_logits[remove] = float('-inf')
            logits = torch.zeros_like(logits).scatter_(1, sorted_idx, sorted_logits)

        probs   = F.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)    # (1, 1)
        ids     = torch.cat([ids, next_id], dim=1)

        if next_id.item() == eos_id:
            break

    # ── 提取仅属于"本段新生成"的 token ───────────────────────────
    all_ids  = ids[0].cpu().tolist()
    # 跳过 BOS（1个）+ prefix（n_prefix 个），其余为本段新内容
    new_ids  = all_ids[1 + n_prefix:]
    # 过滤特殊 token（含尾部 EOS）
    new_ids  = [t for t in new_ids if t not in special_ids]
    return new_ids


# ─────────────────────────────────────────────
# ★ 核心改动：多段拼接（带前缀携带）
# ─────────────────────────────────────────────

@torch.no_grad()
def sample_long_prefix(
    model:        MusicVAE,
    n:            int,
    bos_id:       int,
    eos_id:       int,
    pad_id:       int,
    device:       str,
    n_segments:   int   = 8,
    z_walk_scale: float = 0.1,
    temperature:  float = 0.85,
    top_k:        int   = 0,
    top_p:        float = 0.9,
    max_steps:    int   = 256,
    prefix_len:   int   = 32,
    z_init:       torch.Tensor = None,
) -> List[List[int]]:
    """
    生成 n 首完整曲目，每首由 n_segments 个 VAE 段落拼接而成。

    在原版基础上增加「前缀携带」机制：
      每段解码前，将上一段末尾 prefix_len 个干净 token 作为前缀，
      使 Decoder 能看到上下文，自然续写，拼接处更流畅。

    Args:
        prefix_len: 携带的前缀 token 数（0 = 退化为原版行为）
        其余参数含义与 generate_vae.py 中的 sample_long 完全相同

    Returns:
        list of token id lists，每条为一首完整曲子的 token 序列
    """
    model.eval()
    z_dim = model.z_dim

    if z_init is not None:
        z_all = z_init.to(device)
    else:
        z_all = torch.randn(n, z_dim, device=device)

    special_ids = {bos_id, eos_id, pad_id}
    results = []

    for song_idx in range(n):
        z_cur        = z_all[song_idx : song_idx + 1].clone()   # (1, z_dim)
        song_tokens: List[int] = []
        prev_seg_tokens: List[int] = []                          # 上一段输出（干净 token）

        for seg_idx in range(n_segments):
            # 从上一段末尾取前缀（首段无前缀）
            if prefix_len > 0 and prev_seg_tokens:
                # 确保前缀中不含特殊 token（双重保险）
                clean_prev = [t for t in prev_seg_tokens if t not in special_ids]
                prefix_ids = clean_prev[-prefix_len:]
            else:
                prefix_ids = []

            seg_tokens = decode_one_segment_with_prefix(
                model       = model,
                z           = z_cur,
                bos_id      = bos_id,
                eos_id      = eos_id,
                pad_id      = pad_id,
                max_steps   = max_steps,
                temperature = temperature,
                top_k       = top_k,
                top_p       = top_p,
                prefix_ids  = prefix_ids,
            )

            song_tokens.extend(seg_tokens)
            prev_seg_tokens = seg_tokens          # 保存本段输出供下段使用

            # z 游走到下一段
            z_cur = z_walk_step(z_cur, z_walk_scale)

            pfx_info = f"前缀={len(prefix_ids)}tok" if prefix_ids else "无前缀"
            print(f"  曲目 #{song_idx+1} 段 {seg_idx+1}/{n_segments} [{pfx_info}]: "
                  f"+{len(seg_tokens)} tokens (累计 {len(song_tokens)})")

        results.append(song_tokens)

    return results


# ─────────────────────────────────────────────
# 三种生成模式（与 generate_vae.py 对应，增加 prefix_len 参数）
# ─────────────────────────────────────────────

def cmd_sample(args):
    """模式 1：从 N(0,I) 采样并多段拼接（带前缀携带）"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer, bos_id, eos_id, pad_id = load_model(args.checkpoint, device)

    gc           = cfg.MUSIC_VAE_GENERATE_CONFIG
    temp         = args.temperature  if args.temperature  is not None else gc["temperature"]
    top_k        = args.top_k        if args.top_k        is not None else gc["top_k"]
    top_p        = args.top_p        if args.top_p        is not None else gc["top_p"]
    max_step     = args.max_tokens   if args.max_tokens   is not None else gc["max_decode_steps"]
    n_segments   = args.n_segments   if args.n_segments   is not None else gc["n_segments"]
    z_walk_scale = args.z_walk_scale if args.z_walk_scale is not None else gc["z_walk_scale"]
    prefix_len   = args.prefix_len

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    pfx_desc = f"前缀={prefix_len}" if prefix_len > 0 else "无前缀（等同原版）"
    print(f"\n[Generate-Prefix] 随机采样 {args.n} 首 × {n_segments} 段")
    print(f"  z_walk={z_walk_scale}  {pfx_desc}")
    print(f"  预计每首约 {n_segments * max_step} tokens")

    sequences = sample_long_prefix(
        model        = model,
        n            = args.n,
        bos_id       = bos_id,
        eos_id       = eos_id,
        pad_id       = pad_id,
        device       = device,
        n_segments   = n_segments,
        z_walk_scale = z_walk_scale,
        temperature  = temp,
        top_k        = top_k,
        top_p        = top_p,
        max_steps    = max_step,
        prefix_len   = prefix_len,
    )

    for i, seq in enumerate(sequences):
        pfx_tag  = f"pfx{prefix_len}" if prefix_len > 0 else "nopfx"
        out_path = os.path.join(cfg.OUTPUT_DIR, f"vae_{pfx_tag}_{ts}_{i+1:02d}.midi")
        print(f"\n  曲目 #{i+1}: 共 {len(seq)} tokens → {out_path}")
        tokens_to_midi(seq, tokenizer, out_path)

    print(f"\n[Generate-Prefix] 完成，输出目录: {cfg.OUTPUT_DIR}")


def cmd_interpolate(args):
    """模式 2：在两首 MIDI 的潜编码间做 slerp，每步带前缀携带"""
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
    prefix_len   = args.prefix_len

    print(f"\n[Generate-Prefix] 插值 {steps} 步: {Path(midi_a).name} → {Path(midi_b).name}")
    print(f"  每步 {n_segments} 段，z_walk={z_walk_scale}，前缀={prefix_len}")

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
        z  = slerp(t, mu1, mu2)
        print(f"\n  t={t:.2f}...")
        seqs = sample_long_prefix(
            model        = model,
            n            = 1,
            bos_id       = bos_id,
            eos_id       = eos_id,
            pad_id       = pad_id,
            device       = device,
            n_segments   = n_segments,
            z_walk_scale = z_walk_scale,
            temperature  = temp,
            top_k        = 0,
            top_p        = top_p,
            max_steps    = max_step,
            prefix_len   = prefix_len,
            z_init       = z,
        )
        pfx_tag  = f"pfx{prefix_len}" if prefix_len > 0 else "nopfx"
        out_path = os.path.join(
            cfg.OUTPUT_DIR, f"vae_{pfx_tag}_interp_{ts}_{i+1:02d}_t{t:.2f}.midi"
        )
        tokens_to_midi(seqs[0], tokenizer, out_path)

    print(f"\n[Generate-Prefix] 插值完成，输出目录: {cfg.OUTPUT_DIR}")


def cmd_reconstruct(args):
    """模式 3：编码后再解码（重建验证，带前缀携带）"""
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
    prefix_len   = args.prefix_len

    print(f"\n[Generate-Prefix] 重建: {Path(midi_path).name}，{n_segments} 段，前缀={prefix_len}")

    mu, logvar = model.encoder(x)
    z = mu if not args.stochastic else MusicVAE.reparameterize(mu, logvar)

    seqs = sample_long_prefix(
        model        = model,
        n            = 1,
        bos_id       = bos_id,
        eos_id       = eos_id,
        pad_id       = pad_id,
        device       = device,
        n_segments   = n_segments,
        z_walk_scale = z_walk_scale,
        temperature  = temp,
        top_k        = 0,
        top_p        = top_p,
        max_steps    = max_step,
        prefix_len   = prefix_len,
        z_init       = z,
    )

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem     = Path(midi_path).stem
    pfx_tag  = f"pfx{prefix_len}" if prefix_len > 0 else "nopfx"
    out_path = os.path.join(cfg.OUTPUT_DIR, f"vae_{pfx_tag}_recon_{ts}_{stem}.midi")
    tokens_to_midi(seqs[0], tokenizer, out_path)

    print(f"\n[Generate-Prefix] 重建完成，输出: {out_path}")


# ─────────────────────────────────────────────
# 命令行参数（在 generate_vae.py 基础上增加 --prefix_len）
# ─────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    gc = cfg.MUSIC_VAE_GENERATE_CONFIG

    parser = argparse.ArgumentParser(
        description = "MusicVAE 生成工具（前缀携带版）",
        formatter_class = argparse.RawDescriptionHelpFormatter,
        epilog = f"""
与 generate_vae.py 的对比
--------------------------
  原版: python generate_vae.py        --n 3
  本版: python generate_vae_prefix.py --n 3 --prefix_len 32

  --prefix_len 0  ← 退化为原版（无前缀），方便 A/B 对比
  --prefix_len 16 ← 轻量前缀
  --prefix_len 32 ← 推荐（默认）
  --prefix_len 64 ← 强上下文，但占用更多 RoPE 位置

示例:
  python generate_vae_prefix.py --n 3
  python generate_vae_prefix.py --n 2 --n_segments 16 --prefix_len 32
  python generate_vae_prefix.py --n 3 --prefix_len 0   # 等价于原版
  python generate_vae_prefix.py --interpolate a.midi b.midi --steps 8
  python generate_vae_prefix.py --reconstruct original.midi
        """
    )

    # ── 通用参数 ──────────────────────────────────────────────────
    parser.add_argument("--checkpoint", "-c", type=str,
        default = os.path.join(cfg.CHECKPOINT_DIR, "music_vae_best.pt"),
        help    = "模型 checkpoint 路径")
    parser.add_argument("--temperature", type=float, default=None,
        help=f"采样温度（默认 {gc['temperature']}）")
    parser.add_argument("--top_k", type=int, default=None,
        help="Top-k 截断（0 = 不使用）")
    parser.add_argument("--top_p", type=float, default=None,
        help=f"Nucleus sampling 阈值（默认 {gc['top_p']}）")
    parser.add_argument("--max_tokens", type=int, default=None,
        help=f"每段最多 token 数（默认 {gc['max_decode_steps']}）")

    # ── 多段参数 ──────────────────────────────────────────────────
    parser.add_argument("--n_segments", type=int, default=None,
        help=f"每首由多少段拼成（默认 {gc['n_segments']}）")
    parser.add_argument("--z_walk_scale", type=float, default=None,
        help=f"段间 z 游走幅度（默认 {gc['z_walk_scale']}）")

    # ── ★ 新增：前缀长度 ─────────────────────────────────────────
    parser.add_argument("--prefix_len", type=int, default=32,
        help=("跨段携带的前缀 token 数（默认 32）\n"
              "  0  = 无前缀，等价于 generate_vae.py\n"
              "  16 = 轻量连接\n"
              "  32 = 推荐（平衡连贯性与显存）\n"
              "  64 = 强上下文（占用更多 RoPE 位置）\n"
              "  上限建议: prefix_len + max_tokens ≤ 480（RoPE 上限 512 - 32 余量）"))

    # ── 模式 1: 随机采样 ──────────────────────────────────────────
    parser.add_argument("--n", type=int, default=3,
        help="[采样] 生成首数（默认 3）")

    # ── 模式 2: 插值 ──────────────────────────────────────────────
    parser.add_argument("--interpolate", type=str, nargs=2,
        metavar=("MIDI_A", "MIDI_B"), default=None,
        help="[插值] 两个 MIDI 文件路径")
    parser.add_argument("--steps", type=int, default=8,
        help="[插值] 插值步数（含两端点，默认 8）")

    # ── 模式 3: 重建 ──────────────────────────────────────────────
    parser.add_argument("--reconstruct", type=str, default=None, metavar="MIDI",
        help="[重建] 要重建的 MIDI 文件路径")
    parser.add_argument("--stochastic", action="store_true",
        help="[重建] 使用随机采样（默认用确定性 mu）")

    return parser


# ─────────────────────────────────────────────
# 入口
# ─────────────────────────────────────────────

if __name__ == "__main__":
    parser = build_parser()
    args   = parser.parse_args()

    # 安全检查：确保 prefix_len + max_steps 不超过 RoPE 上限
    gc       = cfg.MUSIC_VAE_GENERATE_CONFIG
    max_step = args.max_tokens if args.max_tokens is not None else gc["max_decode_steps"]
    rope_limit = cfg.MUSIC_VAE_CONFIG["segment_len"] * 2   # = 512

    # 总输入 = 1(BOS) + prefix_len + max_steps
    total_len = 1 + args.prefix_len + max_step
    if total_len > rope_limit:
        print(f"[警告] prefix_len({args.prefix_len}) + max_tokens({max_step}) + 1 = {total_len} "
              f"> RoPE 上限 {rope_limit}，自动裁剪 prefix_len。")
        args.prefix_len = max(0, rope_limit - max_step - 1)
        print(f"  已调整 prefix_len → {args.prefix_len}")

    if args.interpolate:
        cmd_interpolate(args)
    elif args.reconstruct:
        cmd_reconstruct(args)
    else:
        cmd_sample(args)
