"""
Dataset / DataLoader 模块
将 token 列表切分为固定长度的训练样本 (sliding window)
"""

import random
from typing import List, Tuple, Optional

import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import config as cfg


class MaestroDataset(Dataset):
    """
    将每首曲子的 token 序列用滑动窗口切成 (context_len + 1) 大小的片段。
    输入 x = tokens[i : i+context_len]
    目标 y = tokens[i+1 : i+context_len+1]  (下一个 token 预测)

    Args:
        all_tokens:   token id 列表的列表，每条对应一首曲子
        context_len:  滑动窗口长度（= 模型最大序列长度）
        stride:       滑动步长，默认 = context_len // 2 (50% overlap)
        pad_id:       PAD token id
    """

    def __init__(
        self,
        all_tokens: List[List[int]],
        context_len: int = 1024,
        stride: Optional[int] = None,
        pad_id: int = 0,
    ):
        self.context_len = context_len
        self.pad_id      = pad_id
        self.stride      = stride if stride is not None else context_len // 2

        # 生成所有 (曲目索引, 起始位置) 对
        self.samples: List[Tuple[int, int]] = []
        self.all_tokens = all_tokens

        for song_idx, tokens in enumerate(all_tokens):
            # 至少要有 context_len+1 个 token 才能形成一个样本
            if len(tokens) < self.context_len + 1:
                # 对短曲子做 padding，作为一个样本
                self.samples.append((song_idx, 0))
            else:
                for start in range(0, len(tokens) - self.context_len, self.stride):
                    self.samples.append((song_idx, start))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        song_idx, start = self.samples[idx]
        tokens = self.all_tokens[song_idx]

        end = start + self.context_len + 1  # +1 for target shift

        # 截取片段
        chunk = tokens[start:end]

        # 如果片段不够长，在末尾用 PAD 填充
        if len(chunk) < self.context_len + 1:
            chunk = chunk + [self.pad_id] * (self.context_len + 1 - len(chunk))

        chunk = torch.tensor(chunk, dtype=torch.long)
        x = chunk[:-1]   # (context_len,)
        y = chunk[1:]    # (context_len,)
        return x, y


class MusicVAEDataset(Dataset):
    """
    MusicVAE 数据集：将 token 序列切成固定长度的片段。

    每个样本返回:
      enc_input  : segment[0 : segment_len]      — Encoder 输入（完整片段）
      dec_input  : segment[0 : segment_len - 1]  — Decoder 输入（teacher-forcing）
      dec_target : segment[1 : segment_len]       — Decoder 目标（next-token）

    Args:
        all_tokens:   token id 列表的列表，每条对应一首曲子
        segment_len:  每个片段的固定长度
        stride:       切分步长，默认 = segment_len（无 overlap）
        pad_id:       PAD token id（短曲子末尾补全用）
        min_len:      最短有效曲目长度（过短的曲子直接丢弃）
    """

    def __init__(
        self,
        all_tokens:  List[List[int]],
        segment_len: int          = 256,
        stride:      Optional[int] = None,
        pad_id:      int          = 0,
        min_len:     int          = 32,
    ):
        self.segment_len = segment_len
        self.pad_id      = pad_id
        self.stride      = stride if stride is not None else segment_len
        self.all_tokens  = all_tokens

        # 生成 (曲目索引, 起始位置) 样本列表
        self.samples: List[Tuple[int, int]] = []
        for song_idx, tokens in enumerate(all_tokens):
            if len(tokens) < min_len:
                continue
            if len(tokens) < segment_len:
                # 短曲子：做 padding，当作一个完整样本
                self.samples.append((song_idx, 0))
            else:
                for start in range(0, len(tokens) - segment_len + 1, self.stride):
                    self.samples.append((song_idx, start))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor, Tensor]:
        song_idx, start = self.samples[idx]
        tokens = self.all_tokens[song_idx]

        end   = start + self.segment_len
        chunk = tokens[start:end]

        # 末尾补 PAD
        if len(chunk) < self.segment_len:
            chunk = chunk + [self.pad_id] * (self.segment_len - len(chunk))

        chunk = torch.tensor(chunk, dtype=torch.long)   # (segment_len,)

        enc_input  = chunk                              # (segment_len,)
        dec_input  = chunk[:-1]                         # (segment_len - 1,)
        dec_target = chunk[1:]                          # (segment_len - 1,)

        return enc_input, dec_input, dec_target


def create_vae_dataloaders(
    train_tokens: List[List[int]],
    val_tokens:   List[List[int]],
    segment_len:  int  = None,
    batch_size:   int  = None,
    num_workers:  int  = None,
    pin_memory:   bool = None,
    pad_id:       int  = 0,
) -> Tuple[DataLoader, DataLoader]:
    """
    创建 MusicVAE 训练和验证 DataLoader。

    Args:
        train_tokens:  训练集 token 列表
        val_tokens:    验证集 token 列表
        segment_len:   片段长度（None 时从 config 读取）
        其余参数若为 None，从 config.py 中读取

    Returns:
        train_loader, val_loader
    """
    import config as _cfg

    segment_len = segment_len or _cfg.MUSIC_VAE_CONFIG["segment_len"]
    batch_size  = batch_size  or _cfg.MUSIC_VAE_TRAIN_CONFIG["batch_size"]
    num_workers = num_workers if num_workers is not None else _cfg.MUSIC_VAE_TRAIN_CONFIG["num_workers"]
    pin_memory  = pin_memory  if pin_memory  is not None else _cfg.MUSIC_VAE_TRAIN_CONFIG["pin_memory"]

    train_ds = MusicVAEDataset(
        train_tokens,
        segment_len = segment_len,
        stride      = segment_len,   # 训练：无 overlap，样本更多样
        pad_id      = pad_id,
    )
    val_ds = MusicVAEDataset(
        val_tokens,
        segment_len = segment_len,
        stride      = segment_len,
        pad_id      = pad_id,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size         = batch_size,
        shuffle            = True,
        num_workers        = num_workers,
        pin_memory         = pin_memory,
        drop_last          = True,
        persistent_workers = (num_workers > 0),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size         = batch_size,
        shuffle            = False,
        num_workers        = num_workers,
        pin_memory         = pin_memory,
        drop_last          = False,
        persistent_workers = (num_workers > 0),
    )

    print(f"[VAEDataset] train: {len(train_ds)} 样本 | val: {len(val_ds)} 样本")
    print(f"[VAEDataset] train batches: {len(train_loader)} | val batches: {len(val_loader)}")
    return train_loader, val_loader


def create_dataloaders(
    train_tokens: List[List[int]],
    val_tokens:   List[List[int]],
    context_len:  int  = None,
    batch_size:   int  = None,
    num_workers:  int  = None,
    pin_memory:   bool = None,
    pad_id:       int  = 0,
) -> Tuple[DataLoader, DataLoader]:
    """
    创建训练和验证 DataLoader。

    Args:
        train_tokens:  训练集 token 列表
        val_tokens:    验证集 token 列表
        其余参数若为 None，从 config.py 中读取

    Returns:
        train_loader, val_loader
    """
    context_len = context_len or cfg.MODEL_CONFIG["context_len"]
    batch_size  = batch_size  or cfg.TRAIN_CONFIG["batch_size"]
    num_workers = num_workers if num_workers is not None else cfg.TRAIN_CONFIG["num_workers"]
    pin_memory  = pin_memory  if pin_memory  is not None else cfg.TRAIN_CONFIG["pin_memory"]

    train_ds = MaestroDataset(train_tokens, context_len=context_len, pad_id=pad_id)
    val_ds   = MaestroDataset(val_tokens,   context_len=context_len,
                              stride=context_len,  # val 不需要 overlap
                              pad_id=pad_id)

    train_loader = DataLoader(
        train_ds,
        batch_size         = batch_size,
        shuffle            = True,
        num_workers        = num_workers,
        pin_memory         = pin_memory,
        drop_last          = True,
        persistent_workers = (num_workers > 0),  # worker 跨 epoch 保持存活，避免重复创建
    )
    val_loader = DataLoader(
        val_ds,
        batch_size         = batch_size,
        shuffle            = False,
        num_workers        = num_workers,
        pin_memory         = pin_memory,
        drop_last          = False,
        persistent_workers = (num_workers > 0),
    )

    print(f"[Dataset] train: {len(train_ds)} 样本 | val: {len(val_ds)} 样本")
    print(f"[Dataset] train batches: {len(train_loader)} | val batches: {len(val_loader)}")
    return train_loader, val_loader
