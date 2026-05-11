"""
MusicVAE 自适应生成脚本（动态 EOS 压力版）
==========================================
在 generate_vae_prefix.py 基础上修改，核心改进：

  原版（generate_vae_prefix.py）：
    固定拼接 n_segments 段后停止，末尾被硬截断，音乐戛然而止。

  本版（generate_vae_adaptive.py）：
    无固定段数限制，持续生成直到 EOS 被采样到或超出 max_total_tokens。
    EOS 偏置随累积 token 数线性增长：
      fatigue  = total_tokens / max_total_tokens   ∈ [0, 1]
      eos_bias = eos_bias_max * fatigue
    当 total_tokens ≈ max_total_tokens 时，eos_bias ≈ eos_bias_max（默认 8.0），
    EOS 被采到的概率极高，曲子自然停止。
    兜底：若最终仍未遇到 EOS，输出裁剪至最后完整小节边界（trim_to_bar）。

参数说明:
  --max_total_tokens  全局目标 token 上限（默认 2048，平均停止点约在此附近）
  --eos_bias_max      EOS 偏置最大值（默认 8.0）
  --prefix_len        跨段前缀长度（默认 32，0 = 退化为无前缀版）
  --z_walk_scale      段间 z 游走幅度（默认 0.1）
  --max_step_per_seg  单段最多 token 数（默认 256）

用法示例:
  python generate_vae_adaptive.py --n 3
  python generate_vae_adaptive.py --n 2 --max_total_tokens 2048
  python generate_vae_adaptive.py --n 3 --eos_bias_max 6.0
  python generate_vae_adaptive.py --interpolate a.midi b.midi --steps 8
  python generate_vae_adaptive.py --reconstruct original.midi
"""

import os
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F

import config as cfg
from data.preprocess import build_tokenizer
from model.music_vae import MusicVAE


# ─────────────────────────────────────────────
# 工具函数（与 generate_vae_prefix.py 相同）
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
    print(f"[Generate] 模型已加载: {checkpoint_path}")
    return model, tokenizer, bos_id, eos_id, pad_id


def midi_to_tokens(midi_path: str, tokenizer, pad_id: int, segment_len: int) -> torch.Tensor:
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
    """将 token id 列表转换为 MIDI 文件，去除特殊 token 后直接解码。"""
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
# z 游走（同 generate_vae_prefix.py）
# ─────────────────────────────────────────────

def z_walk_step(z: torch.Tensor, scale: float) -> torch.Tensor:
    if scale <= 0.0:
        return z
    noise     = torch.randn_like(z) * scale
    z_new     = z + noise
    norm_orig = z.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    norm_new  = z_new.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    z_new     = z_new / norm_new * norm_orig
    return z_new


# ─────────────────────────────────────────────
# ★ 核心：带动态 EOS 压力 + 前缀的单段解码
# ─────────────────────────────────────────────

def decode_one_segment_adaptive(
    model:               MusicVAE,
    z:                   torch.Tensor,
    bos_id:              int,
    eos_id:              int,
    pad_id:              int,
    max_steps:           int,
    temperature:         float,
    top_k:               int,
    top_p:               float,
    prefix_ids:          Optional[List[int]],
    current_total_tokens: int,
    max_total_tokens:    int,
    eos_bias_max:        float,
) -> Tuple[List[int], bool]:
    """
    从单个 z 解码一段 token，带动态 EOS 压力和前缀携带。

    EOS 压力公式（逐 token 计算）：
        step_total = current_total_tokens + step_in_seg  (已生成的总 token 数)
        fatigue    = clamp(step_total / max_total_tokens, 0, 1)
        eos_bias   = eos_bias_max * fatigue
        logits[eos_id] += eos_bias                        (在 softmax 之前叠加)

    Args:
        prefix_ids:            上一段末尾的干净 token（已过滤特殊 token）
        current_total_tokens:  本段开始前已累积的 token 数
        max_total_tokens:      全局目标上限
        eos_bias_max:          EOS 偏置最大值

    Returns:
        (new_tokens, hit_eos)
          new_tokens: 本段新生成的干净 token 列表（不含前缀、BOS、EOS、PAD）
          hit_eos:    本段是否遇到了 EOS（True → 外层循环应停止拼接）
    """
    special_ids = {bos_id, eos_id, pad_id}
    device      = z.device

    # ── 构建初始序列 [BOS] + 前缀 ─────────────────────────────────
    clean_prefix = [t for t in (prefix_ids or []) if t not in special_ids]
    init_ids     = [bos_id] + clean_prefix
    n_prefix     = len(clean_prefix)

    ids      = torch.tensor([init_ids], dtype=torch.long, device=device)
    bar_embs = model.conductor(z)   # (1, n_bars, cond_hidden)

    hit_eos  = False
    step_in_seg = 0   # 本段已生成的新 token 数（不含前缀）

    for _ in range(max_steps):
        logits = model.decoder(ids, bar_embs)   # (1, T, vocab_size)
        logits = logits[:, -1, :]               # (1, vocab_size)

        # ── 动态 EOS 压力 ─────────────────────────────────────────
        step_total = current_total_tokens + step_in_seg
        fatigue    = min(step_total / max(max_total_tokens, 1), 1.0)
        eos_bias   = eos_bias_max * fatigue
        logits[0, eos_id] = logits[0, eos_id] + eos_bias

        # ── 温度缩放 ──────────────────────────────────────────────
        logits = logits / max(temperature, 1e-8)

        # ── Top-k 截断 ────────────────────────────────────────────
        if top_k > 0:
            top_k_val = min(top_k, logits.size(-1))
            values, _ = torch.topk(logits, top_k_val)
            logits[logits < values[:, -1:]] = float('-inf')

        # ── Nucleus (top-p) 截断 ──────────────────────────────────
        if top_p < 1.0:
            sorted_logits, sorted_idx = torch.sort(logits, descending=True)
            cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            remove    = cum_probs - F.softmax(sorted_logits, dim=-1) > top_p
            sorted_logits[remove] = float('-inf')
            logits = torch.zeros_like(logits).scatter_(1, sorted_idx, sorted_logits)

        probs   = F.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)   # (1, 1)
        ids     = torch.cat([ids, next_id], dim=1)
        step_in_seg += 1

        if next_id.item() == eos_id:
            hit_eos = True
            break

    # ── 提取本段新生成的 token（跳过 BOS + 前缀，过滤特殊 token）──
    all_ids  = ids[0].cpu().tolist()
    new_ids  = all_ids[1 + n_prefix:]          # 跳过 BOS 和前缀
    new_ids  = [t for t in new_ids if t not in special_ids]
    return new_ids, hit_eos


# ─────────────────────────────────────────────
# ★ 核心：自适应多段拼接
# ─────────────────────────────────────────────

@torch.no_grad()
def sample_adaptive(
    model:            MusicVAE,
    n:                int,
    bos_id:           int,
    eos_id:           int,
    pad_id:           int,
    device:           str,
    max_total_tokens: int   = 2048,
    eos_bias_max:     float = 8.0,
    z_walk_scale:     float = 0.1,
    temperature:      float = 0.85,
    top_k:            int   = 0,
    top_p:            float = 0.9,
    max_step_per_seg: int   = 256,
    prefix_len:       int   = 16,
    z_init:           torch.Tensor = None,
) -> List[List[int]]:
    """
    生成 n 首曲子，持续拼接段落直到 EOS 被采到或累积 token 达到 max_total_tokens。

    Args:
        max_total_tokens:  全局目标上限（EOS 压力以此为基准线性增长）
        eos_bias_max:      达到 max_total_tokens 时的 EOS 偏置值（默认 8.0）
        prefix_len:        跨段前缀长度（0 = 无前缀）
        z_walk_scale:      段间 z 游走幅度
        max_step_per_seg:  单段最多生成 token 数

    Returns:
        list of token id lists（已过滤特殊 token，供 tokens_to_midi 使用）
    """
    model.eval()
    z_dim = model.z_dim

    if z_init is not None:
        z_all = z_init.to(device)
    else:
        z_all = torch.randn(n, z_dim, device=device)

    special_ids = {bos_id, eos_id, pad_id}
    results     = []

    for song_idx in range(n):
        z_cur              = z_all[song_idx : song_idx + 1].clone()
        song_tokens:       List[int] = []
        prev_seg_tokens:   List[int] = []
        seg_idx            = 0

        while True:
            # ── 构建前缀 ────────────────────────────────────────────
            if prefix_len > 0 and prev_seg_tokens:
                clean_prev = [t for t in prev_seg_tokens if t not in special_ids]
                prefix_ids = clean_prev[-prefix_len:]
            else:
                prefix_ids = []

            # ── 解码一段 ────────────────────────────────────────────
            seg_tokens, hit_eos = decode_one_segment_adaptive(
                model                = model,
                z                    = z_cur,
                bos_id               = bos_id,
                eos_id               = eos_id,
                pad_id               = pad_id,
                max_steps            = max_step_per_seg,
                temperature          = temperature,
                top_k                = top_k,
                top_p                = top_p,
                prefix_ids           = prefix_ids,
                current_total_tokens = len(song_tokens),
                max_total_tokens     = max_total_tokens,
                eos_bias_max         = eos_bias_max,
            )

            song_tokens.extend(seg_tokens)
            prev_seg_tokens = seg_tokens
            seg_idx        += 1

            total     = len(song_tokens)
            pfx_info  = f"前缀={len(prefix_ids)}tok" if prefix_ids else "无前缀"
            eos_info  = "→ EOS 命中，停止" if hit_eos else ""
            fatigue   = min(total / max(max_total_tokens, 1), 1.0)
            print(f"  曲目 #{song_idx+1} 段{seg_idx:02d} [{pfx_info}]: "
                  f"+{len(seg_tokens)} tokens (累计 {total}, "
                  f"疲劳度 {fatigue:.2f}) {eos_info}")

            # ── 停止条件 ─────────────────────────────────────────────
            if hit_eos or total >= max_total_tokens:
                reason = "EOS" if hit_eos else f"达到上限 {max_total_tokens}"
                print(f"  → 停止原因: {reason}，共 {total} tokens，{seg_idx} 段")
                break

            # ── z 游走到下一段 ────────────────────────────────────────
            z_cur = z_walk_step(z_cur, z_walk_scale)

        results.append(song_tokens)

    return results


# ─────────────────────────────────────────────
# 三种生成模式
# ─────────────────────────────────────────────

def cmd_sample(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer, bos_id, eos_id, pad_id = load_model(args.checkpoint, device)

    gc               = cfg.MUSIC_VAE_GENERATE_CONFIG
    temp             = args.temperature      if args.temperature      is not None else gc["temperature"]
    top_k            = args.top_k            if args.top_k            is not None else gc["top_k"]
    top_p            = args.top_p            if args.top_p            is not None else gc["top_p"]
    max_step_per_seg = args.max_step_per_seg if args.max_step_per_seg is not None else gc["max_decode_steps"]
    z_walk_scale     = args.z_walk_scale     if args.z_walk_scale     is not None else gc["z_walk_scale"]
    max_total_tokens = args.max_total_tokens
    eos_bias_max     = args.eos_bias_max
    prefix_len       = args.prefix_len

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    print(f"\n[Generate-Adaptive] 随机采样 {args.n} 首")
    print(f"  max_total_tokens={max_total_tokens}  eos_bias_max={eos_bias_max}")
    print(f"  z_walk={z_walk_scale}  prefix_len={prefix_len}  max_step/seg={max_step_per_seg}")

    sequences = sample_adaptive(
        model            = model,
        n                = args.n,
        bos_id           = bos_id,
        eos_id           = eos_id,
        pad_id           = pad_id,
        device           = device,
        max_total_tokens = max_total_tokens,
        eos_bias_max     = eos_bias_max,
        z_walk_scale     = z_walk_scale,
        temperature      = temp,
        top_k            = top_k,
        top_p            = top_p,
        max_step_per_seg = max_step_per_seg,
        prefix_len       = prefix_len,
    )

    for i, seq in enumerate(sequences):
        pfx_tag  = f"pfx{prefix_len}" if prefix_len > 0 else "nopfx"
        out_path = os.path.join(cfg.OUTPUT_DIR, f"vae_adpt_{pfx_tag}_{ts}_{i+1:02d}.midi")
        print(f"\n  曲目 #{i+1}: 共 {len(seq)} tokens → {out_path}")
        tokens_to_midi(seq, tokenizer, out_path)

    print(f"\n[Generate-Adaptive] 完成，输出目录: {cfg.OUTPUT_DIR}")


def cmd_interpolate(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer, bos_id, eos_id, pad_id = load_model(args.checkpoint, device)

    midi_a, midi_b = args.interpolate
    seg_len = cfg.MUSIC_VAE_CONFIG["segment_len"]
    for p in (midi_a, midi_b):
        if not os.path.exists(p):
            raise FileNotFoundError(f"MIDI 文件不存在: {p}")

    x1 = midi_to_tokens(midi_a, tokenizer, pad_id, seg_len).to(device)
    x2 = midi_to_tokens(midi_b, tokenizer, pad_id, seg_len).to(device)

    gc               = cfg.MUSIC_VAE_GENERATE_CONFIG
    temp             = args.temperature      if args.temperature      is not None else gc["temperature"]
    top_p            = args.top_p            if args.top_p            is not None else gc["top_p"]
    max_step_per_seg = args.max_step_per_seg if args.max_step_per_seg is not None else gc["max_decode_steps"]
    z_walk_scale     = args.z_walk_scale     if args.z_walk_scale     is not None else gc["z_walk_scale"]
    max_total_tokens = args.max_total_tokens
    eos_bias_max     = args.eos_bias_max
    prefix_len       = args.prefix_len
    steps            = args.steps

    print(f"\n[Generate-Adaptive] 插值 {steps} 步: {Path(midi_a).name} → {Path(midi_b).name}")

    mu1, _ = model.encoder(x1)
    mu2, _ = model.encoder(x2)

    def slerp(t, v0, v1):
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
        t = i / max(steps - 1, 1)
        z = slerp(t, mu1, mu2)
        print(f"\n  t={t:.2f}...")
        seqs = sample_adaptive(
            model=model, n=1, bos_id=bos_id, eos_id=eos_id, pad_id=pad_id,
            device=device, max_total_tokens=max_total_tokens,
            eos_bias_max=eos_bias_max, z_walk_scale=z_walk_scale,
            temperature=temp, top_k=0, top_p=top_p,
            max_step_per_seg=max_step_per_seg, prefix_len=prefix_len, z_init=z,
        )
        pfx_tag  = f"pfx{prefix_len}" if prefix_len > 0 else "nopfx"
        out_path = os.path.join(cfg.OUTPUT_DIR, f"vae_adpt_{pfx_tag}_interp_{ts}_{i+1:02d}_t{t:.2f}.midi")
        tokens_to_midi(seqs[0], tokenizer, out_path)

    print(f"\n[Generate-Adaptive] 插值完成，输出目录: {cfg.OUTPUT_DIR}")


def cmd_reconstruct(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer, bos_id, eos_id, pad_id = load_model(args.checkpoint, device)

    midi_path = args.reconstruct
    if not os.path.exists(midi_path):
        raise FileNotFoundError(f"MIDI 文件不存在: {midi_path}")

    seg_len = cfg.MUSIC_VAE_CONFIG["segment_len"]
    x       = midi_to_tokens(midi_path, tokenizer, pad_id, seg_len).to(device)

    gc               = cfg.MUSIC_VAE_GENERATE_CONFIG
    temp             = args.temperature      if args.temperature      is not None else gc["temperature"]
    top_p            = args.top_p            if args.top_p            is not None else gc["top_p"]
    max_step_per_seg = args.max_step_per_seg if args.max_step_per_seg is not None else gc["max_decode_steps"]
    z_walk_scale     = args.z_walk_scale     if args.z_walk_scale     is not None else gc["z_walk_scale"]
    max_total_tokens = args.max_total_tokens
    eos_bias_max     = args.eos_bias_max
    prefix_len       = args.prefix_len

    print(f"\n[Generate-Adaptive] 重建: {Path(midi_path).name}")

    mu, logvar = model.encoder(x)
    z = mu if not args.stochastic else MusicVAE.reparameterize(mu, logvar)

    seqs = sample_adaptive(
        model=model, n=1, bos_id=bos_id, eos_id=eos_id, pad_id=pad_id,
        device=device, max_total_tokens=max_total_tokens,
        eos_bias_max=eos_bias_max, z_walk_scale=z_walk_scale,
        temperature=temp, top_k=0, top_p=top_p,
        max_step_per_seg=max_step_per_seg, prefix_len=prefix_len, z_init=z,
    )

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem     = Path(midi_path).stem
    pfx_tag  = f"pfx{prefix_len}" if prefix_len > 0 else "nopfx"
    out_path = os.path.join(cfg.OUTPUT_DIR, f"vae_adpt_{pfx_tag}_recon_{ts}_{stem}.midi")
    tokens_to_midi(seqs[0], tokenizer, out_path)

    print(f"\n[Generate-Adaptive] 重建完成，输出: {out_path}")


# ─────────────────────────────────────────────
# 命令行参数
# ─────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    gc = cfg.MUSIC_VAE_GENERATE_CONFIG

    parser = argparse.ArgumentParser(
        description="MusicVAE 自适应生成工具（动态 EOS 压力版）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
EOS 压力机制说明
-----------------
  每生成一个 token 时，计算「疲劳度」：
    fatigue  = total_tokens / max_total_tokens   ∈ [0, 1]
    eos_bias = eos_bias_max * fatigue

  将 eos_bias 加到 EOS token 的 logit 上（softmax 之前）。
  当 total_tokens = 0 时，eos_bias = 0（完全自然生成）；
  当 total_tokens → max_total_tokens 时，eos_bias → eos_bias_max（EOS 几乎必然被采到）。

  默认参数（max_total_tokens=2048, eos_bias_max=8.0）下，
  平均停止点约在 2048 token 附近，曲子自然收尾。

示例:
  python generate_vae_adaptive.py --n 3
  python generate_vae_adaptive.py --n 2 --max_total_tokens 2048 --eos_bias_max 8.0
  python generate_vae_adaptive.py --n 3 --prefix_len 0    # 无前缀
  python generate_vae_adaptive.py --interpolate a.midi b.midi --steps 8
  python generate_vae_adaptive.py --reconstruct original.midi
        """
    )

    # ── 通用参数 ──────────────────────────────────────────────────
    parser.add_argument("--checkpoint", "-c", type=str,
        default=os.path.join(cfg.CHECKPOINT_DIR, "music_vae_best.pt"),
        help="模型 checkpoint 路径")
    parser.add_argument("--temperature", type=float, default=None,
        help=f"采样温度（默认 {gc['temperature']}）")
    parser.add_argument("--top_k", type=int, default=None,
        help="Top-k 截断（0 = 不使用）")
    parser.add_argument("--top_p", type=float, default=None,
        help=f"Nucleus sampling 阈值（默认 {gc['top_p']}）")
    parser.add_argument("--max_step_per_seg", type=int, default=None,
        help=f"单段最多 token 数（默认 {gc['max_decode_steps']}）")

    # ── ★ 自适应控制参数 ────────────────────────────────────────
    parser.add_argument("--max_total_tokens", type=int, default=2048,
        help="全局目标 token 上限，EOS 压力以此为基准（默认 2048）")
    parser.add_argument("--eos_bias_max", type=float, default=8.0,
        help=("达到 max_total_tokens 时的 EOS logit 偏置（默认 8.0）\n"
              "  越大 → 平均停止点越靠近 max_total_tokens\n"
              "  越小 → 生成更长但可能超出目标"))
    parser.add_argument("--z_walk_scale", type=float, default=None,
        help=f"段间 z 游走幅度（默认 {gc['z_walk_scale']}）")

    # ── 前缀参数 ──────────────────────────────────────────────────
    parser.add_argument("--prefix_len", type=int, default=16,
        help=("跨段前缀 token 数（默认16）\n"
              "  0  = 无前缀，等价于 generate_vae.py 的逻辑\n"
              "  16 = 推荐（平衡连贯性与显存）"))

    # ── 模式 1: 随机采样 ──────────────────────────────────────────
    parser.add_argument("--n", type=int, default=3,
        help="[采样] 生成首数（默认 3）")

    # ── 模式 2: 插值 ──────────────────────────────────────────────
    parser.add_argument("--interpolate", type=str, nargs=2,
        metavar=("MIDI_A", "MIDI_B"), default=None,
        help="[插值] 两个 MIDI 文件路径")
    parser.add_argument("--steps", type=int, default=8,
        help="[插值] 插值步数（默认 8）")

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

    # 安全检查：prefix_len + max_step_per_seg 不超过 RoPE 上限
    gc           = cfg.MUSIC_VAE_GENERATE_CONFIG
    max_step     = args.max_step_per_seg if args.max_step_per_seg is not None else gc["max_decode_steps"]
    rope_limit   = cfg.MUSIC_VAE_CONFIG["segment_len"] * 2   # = 512
    total_len    = 1 + args.prefix_len + max_step
    if total_len > rope_limit:
        print(f"[警告] prefix_len({args.prefix_len}) + max_step_per_seg({max_step}) + 1 "
              f"= {total_len} > RoPE 上限 {rope_limit}，自动裁剪 prefix_len。")
        args.prefix_len = max(0, rope_limit - max_step - 1)
        print(f"  已调整 prefix_len → {args.prefix_len}")

    if args.interpolate:
        cmd_interpolate(args)
    elif args.reconstruct:
        cmd_reconstruct(args)
    else:
        cmd_sample(args)
