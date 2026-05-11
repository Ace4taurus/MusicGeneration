"""
MIDI 生成脚本
用法:
  # 从 BOS token 无条件生成
  python generate.py --checkpoint checkpoints/music_transformer_best.pt

  # 使用 MIDI 文件作为 prompt（续写）
  python generate.py --checkpoint checkpoints/music_transformer_best.pt \
                     --prompt maestro-v3.0.0/2004/xxx.midi \
                     --prompt_bars 4

  # 批量生成多首
  python generate.py --checkpoint checkpoints/music_transformer_best.pt --n 5
"""

import os
import argparse
from pathlib import Path
from datetime import datetime

import torch

import config as cfg
from data.preprocess import build_tokenizer
from model.transformer import MusicTransformer


# ─────────────────────────────────────────────
# MIDI prompt → token ids
# ─────────────────────────────────────────────

def midi_to_prompt(midi_path: str, tokenizer, max_bars: int = 4) -> list:
    """
    将 MIDI 文件的前 max_bars 小节 tokenize，作为生成的 prompt。

    Args:
        midi_path:  MIDI 文件路径
        tokenizer:  REMI tokenizer
        max_bars:   取前多少小节作为 prompt

    Returns:
        token id 列表（包含 BOS）
    """
    from miditok import TokSequence
    tok_seqs = tokenizer(midi_path)
    ids = tok_seqs[0].ids

    bos_id = tokenizer["BOS_None"]
    bar_id = tokenizer["Bar_None"]   # REMI BAR token

    # 找到第 max_bars 个 BAR token 的位置，截断
    bar_count = 0
    cut_pos   = len(ids)
    for i, tok_id in enumerate(ids):
        if tok_id == bar_id:
            bar_count += 1
            if bar_count > max_bars:
                cut_pos = i
                break

    prompt_ids = [bos_id] + ids[:cut_pos]
    print(f"[Generate] Prompt: {len(prompt_ids)} tokens（前 {bar_count} 小节）")
    return prompt_ids


# ─────────────────────────────────────────────
# 生成并保存为 MIDI
# ─────────────────────────────────────────────

def tokens_to_midi(token_ids: list, tokenizer, output_path: str):
    """
    将 token id 列表转换为 MIDI 文件并保存。

    Args:
        token_ids:   token id 列表
        tokenizer:   REMI tokenizer
        output_path: 输出 MIDI 路径
    """
    from miditok import TokSequence

    # 去除特殊 token（BOS / EOS / PAD）
    special_ids = set()
    for name in ["PAD_None", "BOS_None", "EOS_None"]:
        try:
            special_ids.add(tokenizer[name])
        except KeyError:
            pass

    clean_ids = [t for t in token_ids if t not in special_ids]

    # 转换为 TokSequence 并 decode 为 MIDI
    tok_seq = TokSequence(ids=clean_ids)
    tokenizer.complete_sequence(tok_seq)
    midi = tokenizer.tokens_to_midi([tok_seq])

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    midi.dump_midi(output_path)
    print(f"[Generate] MIDI 已保存: {output_path}")


# ─────────────────────────────────────────────
# 主生成流程
# ─────────────────────────────────────────────

def generate(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Generate] 设备: {device}")

    # ── Tokenizer ────────────────────────────
    tokenizer = build_tokenizer(force_rebuild=False)
    vocab_size = len(tokenizer)
    pad_id     = tokenizer["PAD_None"]
    bos_id     = tokenizer["BOS_None"]
    eos_id     = tokenizer["EOS_None"]

    # ── 加载模型 ─────────────────────────────
    mc = cfg.MODEL_CONFIG
    model = MusicTransformer(
        vocab_size   = vocab_size,
        context_len  = mc["context_len"],
        d_model      = mc["d_model"],
        n_heads      = mc["n_heads"],
        n_layers     = mc["n_layers"],
        d_ff         = mc["d_ff"],
        dropout      = 0.0,             # 生成时关闭 dropout
        pad_token_id = pad_id,
    ).to(device)

    # 加载权重
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"checkpoint 不存在: {args.checkpoint}")

    state = torch.load(args.checkpoint, map_location=device, weights_only=True)
    # 兼容保存了完整 ckpt 和只保存了 state_dict 的情况
    if "model" in state:
        state = state["model"]
    # 处理 torch.compile 产生的 _orig_mod 前缀
    state = {k.replace("_orig_mod.", ""): v for k, v in state.items()}
    model.load_state_dict(state, strict=False)
    model.eval()
    print(f"[Generate] 模型已加载: {args.checkpoint}")

    # ── 准备 Prompt ──────────────────────────
    if args.prompt and os.path.exists(args.prompt):
        prompt_ids = midi_to_prompt(args.prompt, tokenizer, max_bars=args.prompt_bars)
    else:
        # 无条件生成：仅 BOS
        prompt_ids = [bos_id]
        print(f"[Generate] 无条件生成（BOS prompt）")

    # ── 生成 N 首 ────────────────────────────
    gc = cfg.GENERATE_CONFIG
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    for i in range(args.n):
        print(f"\n[Generate] 正在生成第 {i+1}/{args.n} 首 ...")

        prompt_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=device)

        with torch.no_grad():
            generated = model.generate(
                prompt         = prompt_tensor,
                max_new_tokens = args.max_tokens or gc["max_new_tokens"],
                temperature    = args.temperature or gc["temperature"],
                top_k          = args.top_k or gc["top_k"],
                top_p          = args.top_p or gc["top_p"],
                repetition_penalty = gc["repetition_penalty"],
                eos_token_id   = eos_id,
            )

        token_list = generated[0].cpu().tolist()
        n_new      = len(token_list) - len(prompt_ids)
        print(f"[Generate] 生成了 {n_new} 个新 token（总长 {len(token_list)}）")

        # 保存 MIDI
        out_path = os.path.join(cfg.OUTPUT_DIR, f"generated_{timestamp}_{i+1:02d}.midi")
        tokens_to_midi(token_list, tokenizer, out_path)

    print(f"\n[Generate] 全部完成，输出目录: {cfg.OUTPUT_DIR}")


# ─────────────────────────────────────────────
# 入口
# ─────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MIDI 生成")

    parser.add_argument(
        "--checkpoint", "-c",
        type=str,
        default=os.path.join(cfg.CHECKPOINT_DIR, "music_transformer_best.pt"),
        help="模型 checkpoint 路径"
    )
    parser.add_argument(
        "--prompt", "-p",
        type=str,
        default=None,
        help="MIDI 文件路径，作为生成 prompt（可选）"
    )
    parser.add_argument(
        "--prompt_bars",
        type=int,
        default=4,
        help="从 prompt MIDI 取前多少小节作为条件（默认 4）"
    )
    parser.add_argument(
        "--n",
        type=int,
        default=1,
        help="生成曲目数量（默认 1）"
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=None,
        help="最多生成 token 数（默认使用 config 中的值）"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="采样温度（默认使用 config 中的值）"
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=None,
        help="Top-k 截断（默认使用 config 中的值）"
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=None,
        help="Nucleus sampling 阈值（默认使用 config 中的值）"
    )

    args = parser.parse_args()
    generate(args)
