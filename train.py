"""
训练入口
用法:
  python train.py                  # 正常训练
  python train.py --resume         # 从最新 checkpoint 继续训练
  python train.py --preprocess     # 仅执行数据预处理后退出
"""

import os
import argparse
import math

import torch
import torch.nn as nn
from tqdm import tqdm

import config as cfg
from data.preprocess import build_tokenizer, tokenize_dataset
from data.dataset import create_dataloaders
from model.transformer import MusicTransformer
from utils.logger import TrainingLogger, get_lr


# ─────────────────────────────────────────────
# 验证函数
# ─────────────────────────────────────────────

@torch.no_grad()
def evaluate(model: nn.Module, val_loader, device: str, dtype: torch.dtype) -> float:
    """在验证集上计算平均 loss。"""
    model.eval()
    total_loss, total_batches = 0.0, 0

    for x, y in tqdm(val_loader, desc="  Eval", leave=False):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        with torch.autocast(device_type="cuda", dtype=dtype):
            _, loss = model(x, targets=y)

        total_loss    += loss.item()
        total_batches += 1

    model.train()
    return total_loss / max(total_batches, 1)


# ─────────────────────────────────────────────
# 主训练循环
# ─────────────────────────────────────────────

def train(args):
    # ── 设备 / 精度 ──────────────────────────
    device = cfg.TRAIN_CONFIG["device"]
    if not torch.cuda.is_available():
        print("[警告] CUDA 不可用，将使用 CPU 训练（速度极慢）")
        device = "cpu"

    dtype_str = cfg.TRAIN_CONFIG["dtype"]
    dtype = torch.bfloat16 if dtype_str == "bfloat16" else torch.float16
    print(f"[Train] 设备: {device} | 精度: {dtype_str}")

    # ── CUDA 性能优化 ─────────────────────────
    if device == "cuda":
        torch.backends.cudnn.benchmark = True          # cuDNN 自动选最优算法
        torch.set_float32_matmul_precision("high")     # 启用 TF32（Tensor Core 加速矩阵乘法）
        print("[Train] cudnn.benchmark=True | TF32 matmul 已启用")

    # ── Tokenizer ────────────────────────────
    tokenizer = build_tokenizer(force_rebuild=False)
    vocab_size = len(tokenizer)
    pad_id     = tokenizer["PAD_None"]
    bos_id     = tokenizer["BOS_None"]
    eos_id     = tokenizer["EOS_None"]
    print(f"[Train] 词表大小: {vocab_size} | PAD={pad_id} | BOS={bos_id} | EOS={eos_id}")

    # ── 数据预处理（若已缓存则直接加载）────────
    train_tokens = tokenize_dataset(tokenizer, split="train")
    val_tokens   = tokenize_dataset(tokenizer, split="validation")

    if args.preprocess:
        print("[Train] --preprocess 完成，退出。")
        return

    # ── DataLoader ───────────────────────────
    train_loader, val_loader = create_dataloaders(
        train_tokens, val_tokens, pad_id=pad_id
    )

    # ── 模型 ─────────────────────────────────
    mc = cfg.MODEL_CONFIG
    model = MusicTransformer(
        vocab_size   = vocab_size,
        context_len  = mc["context_len"],
        d_model      = mc["d_model"],
        n_heads      = mc["n_heads"],
        n_layers     = mc["n_layers"],
        d_ff         = mc["d_ff"],
        dropout      = mc["dropout"],
        pad_token_id = pad_id,
    ).to(device)

    # torch.compile 加速（PyTorch 2.0+）
    # Windows 上不支持 Triton，torch.compile 默认后端会在首次前向时崩溃
    # 因此在 Windows 上跳过 compile
    import sys
    if sys.platform != "win32":
        try:
            model = torch.compile(model)
            print("[Train] torch.compile 已启用")
        except Exception as e:
            print(f"[Train] torch.compile 跳过: {e}")
    else:
        print("[Train] Windows 平台检测到，跳过 torch.compile（Triton 不支持 Windows）")

    # ── 优化器 ───────────────────────────────
    tc = cfg.TRAIN_CONFIG
    # 对权重矩阵使用 weight decay，对 bias / LayerNorm 不使用
    decay_params    = [p for n, p in model.named_parameters()
                       if p.requires_grad and p.dim() >= 2]
    no_decay_params = [p for n, p in model.named_parameters()
                       if p.requires_grad and p.dim() < 2]
    param_groups = [
        {"params": decay_params,    "weight_decay": tc["weight_decay"]},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(
        param_groups,
        lr   = tc["learning_rate"],
        betas= tc["betas"],
        eps  = tc["eps"],
        fused= True if device == "cuda" else False,  # fused AdamW 速度更快
    )

    # 混合精度 GradScaler
    scaler = torch.cuda.amp.GradScaler(enabled=(dtype == torch.float16))

    # ── Logger / Checkpoint ──────────────────
    logger = TrainingLogger()

    start_epoch = 0
    global_step = 0

    if args.resume:
        ckpt = logger.load_checkpoint(model, optimizer, scaler, device=device)
        if ckpt:
            start_epoch = ckpt.get("epoch", 0) + 1
            global_step = ckpt.get("step",  0)

    # 计算总步数（用于学习率调度）
    steps_per_epoch = len(train_loader) // tc["grad_accum_steps"]
    max_steps = steps_per_epoch * tc["max_epochs"]
    lr_decay_iters = tc["lr_decay_iters"] or max_steps

    print(f"[Train] 每 epoch 步数: {steps_per_epoch} | 总步数: {max_steps}")

    # ── 训练循环 ─────────────────────────────
    best_val_loss = float("inf")

    for epoch in range(start_epoch, tc["max_epochs"]):
        model.train()
        optimizer.zero_grad()

        epoch_loss   = 0.0
        accum_count  = 0

        pbar = tqdm(enumerate(train_loader), total=len(train_loader),
                    desc=f"Epoch {epoch+1:03d}")

        for batch_idx, (x, y) in pbar:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            # 更新学习率
            lr = get_lr(
                step         = global_step,
                warmup_steps = tc["warmup_steps"],
                max_steps    = lr_decay_iters,
                max_lr       = tc["learning_rate"],
            )
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

            # 前向传播（混合精度）
            with torch.autocast(device_type="cuda", dtype=dtype):
                _, loss = model(x, targets=y)
                loss = loss / tc["grad_accum_steps"]   # 梯度累积缩放

            # 反向传播
            scaler.scale(loss).backward()
            accum_count += 1

            # 梯度累积：每 grad_accum_steps 步更新一次参数
            if accum_count == tc["grad_accum_steps"]:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), tc["max_grad_norm"])
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                accum_count  = 0
                global_step += 1

                epoch_loss += loss.item() * tc["grad_accum_steps"]
                pbar.set_postfix(loss=f"{loss.item() * tc['grad_accum_steps']:.4f}", lr=f"{lr:.2e}")

        # ── Epoch 结束：验证 ──────────────────
        avg_train_loss = epoch_loss / max(steps_per_epoch, 1)
        val_loss       = evaluate(model, val_loader, device, dtype)

        logger.log({
            "epoch":      epoch + 1,
            "step":       global_step,
            "train_loss": avg_train_loss,
            "val_loss":   val_loss,
            "lr":         lr,
        })

        # ── 保存 checkpoint ───────────────────
        if (epoch + 1) % tc["save_every"] == 0 or val_loss < best_val_loss:
            logger.save_checkpoint(
                model, optimizer, epoch + 1, global_step, val_loss, scaler
            )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # 额外保存一份最优模型
            best_path = os.path.join(cfg.CHECKPOINT_DIR, "music_transformer_best.pt")
            torch.save(model.state_dict(), best_path)
            print(f"  ★ 最优模型已更新: val_loss={val_loss:.4f}")

    print(f"\n[Train] 训练完成！最优 val_loss: {best_val_loss:.4f}")


# ─────────────────────────────────────────────
# 入口
# ─────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MIDI Music Transformer 训练")
    parser.add_argument("--resume",     action="store_true", help="从最新 checkpoint 继续训练")
    parser.add_argument("--preprocess", action="store_true", help="仅执行数据预处理后退出")
    args = parser.parse_args()

    # 创建所需目录
    for d in [cfg.CACHE_DIR, cfg.CHECKPOINT_DIR, cfg.LOG_DIR, cfg.OUTPUT_DIR]:
        os.makedirs(d, exist_ok=True)

    train(args)
