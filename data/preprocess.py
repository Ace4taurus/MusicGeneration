"""
数据预处理模块
功能：
  1. 使用 miditok REMI tokenizer 对 MAESTRO MIDI 文件进行 tokenize
  2. 将结果缓存为 .pt 文件，避免重复处理
  3. 支持按 train/validation/test split 分别处理
"""

import os
import json
import pickle
from pathlib import Path
from typing import List, Dict

import pandas as pd
import torch
from tqdm import tqdm
from miditok import REMI, TokenizerConfig

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
import config as cfg


def build_tokenizer(force_rebuild: bool = False) -> REMI:
    """
    构建或加载 REMI tokenizer。
    如果 tokenizer 已保存，则直接加载；否则用 MAESTRO 数据集训练词表并保存。

    Args:
        force_rebuild: 强制重新构建 tokenizer

    Returns:
        REMI tokenizer 实例
    """
    os.makedirs(cfg.CACHE_DIR, exist_ok=True)

    if os.path.exists(cfg.TOKENIZER_PATH) and not force_rebuild:
        print(f"[Tokenizer] 加载已有 tokenizer: {cfg.TOKENIZER_PATH}")
        tokenizer = REMI(params=cfg.TOKENIZER_PATH)
        return tokenizer

    print("[Tokenizer] 构建新 tokenizer ...")
    tok_cfg = TokenizerConfig(
        # pitch_range 使用默认值 range(21, 109)，即钢琴全键域，无需显式传入
        # （显式传入 range 对象会导致 miditok 3.x 在 JSON 序列化时报错）
        beat_res             = cfg.TOKENIZER_CONFIG["beat_res"],
        num_velocities       = cfg.TOKENIZER_CONFIG["nb_velocities"],   # 3.x 改名
        special_tokens       = cfg.TOKENIZER_CONFIG["special_tokens"],
        use_tempos           = cfg.TOKENIZER_CONFIG["use_tempos"],
        num_tempos           = cfg.TOKENIZER_CONFIG["nb_tempos"],        # 3.x 改名
        tempo_range          = cfg.TOKENIZER_CONFIG["tempo_range"],
        use_time_signatures  = cfg.TOKENIZER_CONFIG["use_time_signatures"],
        use_sustain_pedals   = cfg.TOKENIZER_CONFIG["use_sustain_pedals"],
    )
    tokenizer = REMI(tokenizer_config=tok_cfg)
    tokenizer.save(cfg.TOKENIZER_PATH)   # save_params → save (3.x)
    print(f"[Tokenizer] 词表大小: {len(tokenizer)} | 已保存至 {cfg.TOKENIZER_PATH}")
    return tokenizer


def _get_midi_paths(split: str) -> List[str]:
    """
    从 MAESTRO CSV 中读取指定 split 的 MIDI 路径列表。

    Args:
        split: "train" / "validation" / "test"

    Returns:
        MIDI 文件绝对路径列表
    """
    df = pd.read_csv(cfg.CSV_PATH)
    subset = df[df["split"] == split]
    paths = []
    for rel_path in subset["midi_filename"].tolist():
        abs_path = os.path.join(cfg.DATA_DIR, rel_path)
        if os.path.exists(abs_path):
            paths.append(abs_path)
        else:
            print(f"[警告] 找不到文件: {abs_path}")
    return paths


def tokenize_dataset(
    tokenizer: REMI,
    split: str = "train",
    force_rebuild: bool = False,
) -> List[List[int]]:
    """
    将指定 split 的所有 MIDI 文件 tokenize，并缓存结果。

    Args:
        tokenizer:     REMI tokenizer 实例
        split:         数据集分割 "train"/"validation"/"test"
        force_rebuild: 强制重新 tokenize

    Returns:
        token id 列表的列表，每个元素对应一首曲子
    """
    os.makedirs(cfg.CACHE_DIR, exist_ok=True)
    cache_file = os.path.join(cfg.CACHE_DIR, f"{split}_tokens.pkl")

    if os.path.exists(cache_file) and not force_rebuild:
        print(f"[Preprocess] 加载缓存 {cache_file}")
        with open(cache_file, "rb") as f:
            return pickle.load(f)

    midi_paths = _get_midi_paths(split)
    print(f"[Preprocess] {split} split: {len(midi_paths)} 个 MIDI 文件，开始 tokenize ...")

    all_tokens: List[List[int]] = []
    bos_id = tokenizer["BOS_None"]
    eos_id = tokenizer["EOS_None"]

    for path in tqdm(midi_paths, desc=f"Tokenize [{split}]"):
        try:
            # miditok 返回 TokSequence 列表（每个 track 一个）
            tok_seqs = tokenizer(path)
            # 钢琴曲只有一个 track，取第一个
            ids = tok_seqs[0].ids
            if len(ids) < 16:           # 过短的曲子丢弃
                continue
            # 添加 BOS / EOS
            ids = [bos_id] + ids + [eos_id]
            all_tokens.append(ids)
        except Exception as e:
            print(f"[警告] 处理 {path} 失败: {e}")

    print(f"[Preprocess] {split}: 有效曲目 {len(all_tokens)} 首，"
          f"平均长度 {sum(len(t) for t in all_tokens) // max(len(all_tokens), 1)} tokens")

    with open(cache_file, "wb") as f:
        pickle.dump(all_tokens, f)
    print(f"[Preprocess] 缓存已保存至 {cache_file}")

    return all_tokens


if __name__ == "__main__":
    tokenizer = build_tokenizer()
    for split in ["train", "validation", "test"]:
        tokenize_dataset(tokenizer, split=split)
