"""
训练日志与工具函数
  - TrainingLogger: 记录 loss/lr 并保存/恢复 checkpoint
  - get_lr: 余弦退火 + warmup 学习率调度
"""

import os
import math
import json
import time
from typing import Optional, Dict, Any

import torch
import torch.nn as nn

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import config as cfg


# ─────────────────────────────────────────────
# 学习率调度
# ─────────────────────────────────────────────

def get_lr(
    step:         int,
    warmup_steps: int,
    max_steps:    int,
    max_lr:       float,
    min_lr:       float = None,
) -> float:
    """
    线性 warmup + 余弦退火学习率调度。

    Args:
        step:         当前全局步数（从 0 开始）
        warmup_steps: warmup 步数
        max_steps:    总训练步数
        max_lr:       峰值学习率
        min_lr:       最小学习率（默认 max_lr / 10）

    Returns:
        当前步的学习率
    """
    if min_lr is None:
        min_lr = max_lr / 10.0

    # 1. warmup 阶段
    if step < warmup_steps:
        return max_lr * step / max(warmup_steps, 1)

    # 2. 超过最大步数，使用最小 lr
    if step > max_steps:
        return min_lr

    # 3. 余弦退火
    progress = (step - warmup_steps) / max(max_steps - warmup_steps, 1)
    coeff    = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + coeff * (max_lr - min_lr)


# ─────────────────────────────────────────────
# Training Logger
# ─────────────────────────────────────────────

class TrainingLogger:
    """
    记录训练过程中的指标，并管理 checkpoint 的保存与恢复。

    Args:
        log_dir:        日志目录
        ckpt_dir:       checkpoint 目录
        experiment_name: 实验名称（用于文件名前缀）
    """

    def __init__(
        self,
        log_dir:  str = None,
        ckpt_dir: str = None,
        experiment_name: str = "music_transformer",
    ):
        self.log_dir  = log_dir  or cfg.LOG_DIR
        self.ckpt_dir = ckpt_dir or cfg.CHECKPOINT_DIR
        self.exp_name = experiment_name

        os.makedirs(self.log_dir,  exist_ok=True)
        os.makedirs(self.ckpt_dir, exist_ok=True)

        self.log_path = os.path.join(self.log_dir, f"{self.exp_name}.jsonl")
        self.history: list = []

        self._start_time = time.time()
        print(f"[Logger] 日志路径: {self.log_path}")
        print(f"[Logger] Checkpoint 目录: {self.ckpt_dir}")

    def log(self, metrics: Dict[str, Any]):
        """
        记录一条指标（追加到 jsonl 文件）。

        Args:
            metrics: 字典，如 {"epoch": 1, "step": 100, "train_loss": 2.5, "lr": 1e-4}
        """
        metrics["elapsed"] = round(time.time() - self._start_time, 1)
        self.history.append(metrics)

        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(metrics, ensure_ascii=False) + "\n")

        # 控制台打印
        parts = [f"Epoch {metrics.get('epoch', '?'):>3}",
                 f"Step {metrics.get('step', '?'):>6}"]
        for k, v in metrics.items():
            if k not in ("epoch", "step", "elapsed"):
                parts.append(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")
        parts.append(f"elapsed: {metrics['elapsed']}s")
        print(" | ".join(parts))

    def save_checkpoint(
        self,
        model:     nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch:     int,
        step:      int,
        val_loss:  float,
        scaler:    Optional[Any] = None,
        extra:     Optional[Dict] = None,
    ):
        """
        保存训练 checkpoint。

        Args:
            model:     模型实例
            optimizer: 优化器
            epoch:     当前 epoch
            step:      当前全局步数
            val_loss:  验证集损失
            scaler:    GradScaler（混合精度时使用）
            extra:     其他需要保存的信息
        """
        ckpt = {
            "epoch":     epoch,
            "step":      step,
            "val_loss":  val_loss,
            "model":     model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        if scaler is not None:
            ckpt["scaler"] = scaler.state_dict()
        if extra:
            ckpt.update(extra)

        # 最新 checkpoint（覆盖）
        latest_path = os.path.join(self.ckpt_dir, f"{self.exp_name}_latest.pt")
        torch.save(ckpt, latest_path)

        # 按 epoch 保存
        epoch_path = os.path.join(self.ckpt_dir, f"{self.exp_name}_epoch{epoch:03d}.pt")
        torch.save(ckpt, epoch_path)

        print(f"[Checkpoint] 已保存: {epoch_path}  val_loss={val_loss:.4f}")

    def load_checkpoint(
        self,
        model:     nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scaler:    Optional[Any] = None,
        path:      Optional[str] = None,
        device:    str = "cuda",
    ) -> Dict[str, Any]:
        """
        加载 checkpoint。若 path 为 None，则自动加载最新的 checkpoint。

        Returns:
            checkpoint 字典（包含 epoch、step、val_loss 等）
        """
        if path is None:
            path = os.path.join(self.ckpt_dir, f"{self.exp_name}_latest.pt")

        if not os.path.exists(path):
            print(f"[Checkpoint] 未找到 checkpoint: {path}，从头开始训练")
            return {}

        ckpt = torch.load(path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        if optimizer is not None and "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        if scaler is not None and "scaler" in ckpt:
            scaler.load_state_dict(ckpt["scaler"])

        print(f"[Checkpoint] 已加载: {path}  epoch={ckpt.get('epoch')}  val_loss={ckpt.get('val_loss'):.4f}")
        return ckpt

    def best_val_loss(self) -> float:
        """返回历史记录中的最低验证损失"""
        losses = [h["val_loss"] for h in self.history if "val_loss" in h]
        return min(losses) if losses else float("inf")
