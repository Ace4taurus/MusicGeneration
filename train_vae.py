"""
MusicVAE 训练入口
用法:
  python train_vae.py                   # 正常训练
  python train_vae.py --resume          # 从最新 checkpoint 继续
  python train_vae.py --preprocess      # 仅执行数据预处理后退出

训练特点:
  - ELBO = recon_loss + β * KL_loss
  - Cyclical β annealing（防止 posterior collapse）
  - Free bits（每维 KL 下限）
  - 实时监控 kl_bits（潜空间利用率）
"""

import os
import argparse
import math
import json
from datetime import datetime

import torch
import torch.nn as nn
from tqdm import tqdm

import config as cfg
from data.preprocess import build_tokenizer, tokenize_dataset
from data.dataset import create_vae_dataloaders
from model.music_vae import MusicVAE
from utils.logger import get_lr   # 复用 warmup + 余弦退火调度


# ─────────────────────────────────────────────
# β Annealing 调度
# ─────────────────────────────────────────────

def get_beta(
    step:         int,
    max_steps:    int,
    n_cycles:     int   = 4,
    ratio:        float = 0.5,
    beta_max:     float = 1.0,
) -> float:
    """
    Cyclical β annealing（Fu et al., 2019）。

    每个周期内前 ratio 比例的步数线性从 0 上升到 beta_max，
    剩余步数保持 beta_max。

    Args:
        step:      当前全局步数
        max_steps: 总训练步数
        n_cycles:  总周期数
        ratio:     每周期内上升比例 (0~1)
        beta_max:  β 上限

    Returns:
        当前步的 β 值 (float)
    """
    if max_steps <= 0:
        return beta_max
    cycle_len   = max_steps / n_cycles
    step_in_cyc = step % cycle_len
    rise_len    = cycle_len * ratio
    if step_in_cyc < rise_len:
        return beta_max * (step_in_cyc / rise_len)
    return beta_max


# ─────────────────────────────────────────────
# VAE 专用 Logger
# ─────────────────────────────────────────────

class VAELogger:
    """轻量级日志记录器，将训练指标写入 JSON Lines 文件。"""

    def __init__(self, log_dir: str = None):
        log_dir = log_dir or cfg.LOG_DIR
        os.makedirs(log_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_path  = os.path.join(log_dir, f"vae_train_{ts}.jsonl")
        self.ckpt_dir  = cfg.CHECKPOINT_DIR
        os.makedirs(self.ckpt_dir, exist_ok=True)
        self._best_val = float("inf")

    def log(self, metrics: dict):
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(metrics) + "\n")

        # 简短打印
        ep  = metrics.get("epoch", "?")
        st  = metrics.get("step",  "?")
        tl  = metrics.get("train_loss", float("nan"))
        vl  = metrics.get("val_loss",   float("nan"))
        rec = metrics.get("val_recon",  float("nan"))
        kl  = metrics.get("val_kl",     float("nan"))
        kl_raw = metrics.get("val_kl_raw", float("nan"))
        b   = metrics.get("train_beta", float("nan"))   # 训练用的退火 β
        lr  = metrics.get("lr",         float("nan"))
        print(
            f"  Epoch {ep:>3} | step {st:>6} | "
            f"train={tl:.4f} | val={vl:.4f}(β=1.0) "
            f"(recon={rec:.4f} kl={kl:.4f} kl_raw={kl_raw:.4f}) "
            f"train_β={b:.3f} lr={lr:.2e}"
        )

    def save_checkpoint(
        self,
        model:     nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch:     int,
        step:      int,
        val_loss:  float,
        scaler,
    ):
        ckpt = {
            "epoch":     epoch,
            "step":      step,
            "val_loss":  val_loss,
            "model":     model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler":    scaler.state_dict(),
        }
        path = os.path.join(self.ckpt_dir, "music_vae_latest.pt")
        torch.save(ckpt, path)

        # 若是最优，额外保存
        if val_loss < self._best_val:
            self._best_val = val_loss
            ckpt["best_val"] = self._best_val   # 同步更新 best_val 字段
            best = os.path.join(self.ckpt_dir, "music_vae_best.pt")
            torch.save(ckpt, best)
            print(f"  ★ 最优 checkpoint 已更新: val_loss={val_loss:.4f}")
        return path

    def load_checkpoint(
        self,
        model:     nn.Module,
        optimizer: torch.optim.Optimizer,
        scaler,
        device:    str = "cuda",
    ) -> dict:
        path = os.path.join(self.ckpt_dir, "music_vae_latest.pt")
        if not os.path.exists(path):
            print("[VAELogger] 未找到 checkpoint，从头训练。")
            return {}
        ckpt = torch.load(path, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scaler.load_state_dict(ckpt["scaler"])
        # 恢复历史最优（避免 resume 后 _best_val 重置为 inf 导致劣化模型覆盖 best.pt）
        self._best_val = ckpt.get("best_val", float("inf"))
        print(f"[VAELogger] 已恢复 checkpoint: epoch={ckpt['epoch']} step={ckpt['step']} best_val={self._best_val:.4f}")
        return ckpt


# ─────────────────────────────────────────────
# 验证函数
# ─────────────────────────────────────────────

@torch.no_grad()
def evaluate(
    model:       nn.Module,
    val_loader,
    device:      str,
    dtype:       torch.dtype,
    beta:        float,
) -> dict:
    """在验证集上计算平均 ELBO 及各子项。"""
    model.eval()
    tot_loss = tot_recon = tot_kl = tot_kl_raw = 0.0
    n = 0

    for enc_in, dec_in, dec_tgt in tqdm(val_loader, desc="  Eval", leave=False):
        enc_in  = enc_in.to(device,  non_blocking=True)
        dec_in  = dec_in.to(device,  non_blocking=True)
        dec_tgt = dec_tgt.to(device, non_blocking=True)

        with torch.autocast(device_type="cuda", dtype=dtype):
            loss, recon, kl, kl_raw, _ = model(enc_in, dec_in, dec_tgt, beta=beta)

        tot_loss   += loss.item()
        tot_recon  += recon.item()
        tot_kl     += kl.item()
        tot_kl_raw += kl_raw.item()
        n          += 1

    model.train()
    nb = max(n, 1)
    return {
        "val_loss":   tot_loss   / nb,
        "val_recon":  tot_recon  / nb,
        "val_kl":     tot_kl     / nb,
        "val_kl_raw": tot_kl_raw / nb,
    }


# ─────────────────────────────────────────────
# 主训练循环
# ─────────────────────────────────────────────

def train(args):
    # ── 设备 / 精度 ──────────────────────────
    tc     = cfg.MUSIC_VAE_TRAIN_CONFIG
    device = tc["device"]
    if not torch.cuda.is_available():
        print("[警告] CUDA 不可用，将使用 CPU 训练（速度极慢）")
        device = "cpu"

    dtype_str = tc["dtype"]
    dtype = torch.bfloat16 if dtype_str == "bfloat16" else torch.float16
    print(f"[Train] 设备: {device} | 精度: {dtype_str}")

    if device == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")
        print("[Train] cudnn.benchmark=True | TF32 matmul 已启用")

    # ── Tokenizer ────────────────────────────
    tokenizer = build_tokenizer(force_rebuild=False)
    vocab_size = len(tokenizer)
    pad_id     = tokenizer["PAD_None"]
    bos_id     = tokenizer["BOS_None"]
    eos_id     = tokenizer["EOS_None"]
    print(f"[Train] 词表大小: {vocab_size} | PAD={pad_id} | BOS={bos_id} | EOS={eos_id}")

    # ── 数据 ─────────────────────────────────
    train_tokens = tokenize_dataset(tokenizer, split="train")
    val_tokens   = tokenize_dataset(tokenizer, split="validation")

    if args.preprocess:
        print("[Train] --preprocess 完成，退出。")
        return

    train_loader, val_loader = create_vae_dataloaders(
        train_tokens, val_tokens, pad_id=pad_id
    )

    # ── 模型 ─────────────────────────────────
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
        dropout       = vc["dropout"],
        pad_token_id  = pad_id,
        free_bits     = tc["free_bits"],
    ).to(device)

    # Windows 不支持 torch.compile（Triton 限制）
    import sys
    if sys.platform != "win32":
        try:
            model = torch.compile(model)
            print("[Train] torch.compile 已启用")
        except Exception as e:
            print(f"[Train] torch.compile 跳过: {e}")
    else:
        print("[Train] Windows 平台，跳过 torch.compile")

    # ── 优化器 ───────────────────────────────
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
        lr    = tc["learning_rate"],
        betas = tc["betas"],
        eps   = tc["eps"],
        fused = (device == "cuda"),
    )

    scaler = torch.cuda.amp.GradScaler(enabled=(dtype == torch.float16))

    # ── Logger / Checkpoint ──────────────────
    logger      = VAELogger()
    start_epoch = 0
    global_step = 0

    if args.resume:
        ckpt = logger.load_checkpoint(model, optimizer, scaler, device=device)
        if ckpt:
            start_epoch = ckpt.get("epoch", 0) + 1
            global_step = ckpt.get("step",  0)

    # 计算总步数（用于 β 调度 & 学习率调度）
    steps_per_epoch = len(train_loader) // tc["grad_accum_steps"]
    max_steps       = steps_per_epoch * tc["max_epochs"]
    lr_decay_iters  = tc["lr_decay_iters"] or max_steps

    print(f"[Train] 每 epoch 步数: {steps_per_epoch} | 总步数: {max_steps}")
    print(f"[Train] β annealing: {tc['beta_cycles']} 个周期 | "
          f"上升比例: {tc['beta_ratio']} | β_max: {tc['beta_max']}")
    print(f"[Train] Free bits: {tc['free_bits']} nats/dim")

    # ── 训练循环 ─────────────────────────────
    for epoch in range(start_epoch, tc["max_epochs"]):
        model.train()
        optimizer.zero_grad()

        epoch_loss  = epoch_recon = epoch_kl = 0.0
        accum_count = 0

        pbar = tqdm(
            enumerate(train_loader),
            total=len(train_loader),
            desc=f"Epoch {epoch+1:03d}",
        )

        for batch_idx, (enc_in, dec_in, dec_tgt) in pbar:
            enc_in  = enc_in.to(device,  non_blocking=True)
            dec_in  = dec_in.to(device,  non_blocking=True)
            dec_tgt = dec_tgt.to(device, non_blocking=True)

            # 当前 β 和学习率
            beta = get_beta(
                step      = global_step,
                max_steps = max_steps,
                n_cycles  = tc["beta_cycles"],
                ratio     = tc["beta_ratio"],
                beta_max  = tc["beta_max"],
            )
            lr = get_lr(
                step         = global_step,
                warmup_steps = tc["warmup_steps"],
                max_steps    = lr_decay_iters,
                max_lr       = tc["learning_rate"],
            )
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            # 前向传播
            with torch.autocast(device_type="cuda", dtype=dtype):
                loss, recon, kl, kl_raw, _ = model(enc_in, dec_in, dec_tgt, beta=beta)
                loss = loss / tc["grad_accum_steps"]

            scaler.scale(loss).backward()
            accum_count += 1

            if accum_count == tc["grad_accum_steps"]:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), tc["max_grad_norm"])
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                accum_count  = 0
                global_step += 1

                _l = loss.item() * tc["grad_accum_steps"]
                epoch_loss  += _l
                epoch_recon += recon.item()
                epoch_kl    += kl.item()

                pbar.set_postfix(
                    loss  = f"{_l:.3f}",
                    recon = f"{recon.item():.3f}",
                    kl    = f"{kl_raw.item():.3f}",
                    beta  = f"{beta:.3f}",
                    lr    = f"{lr:.1e}",
                )

        # ── Epoch 结束：验证（固定 beta=1.0，确保跨 epoch 可比）──
        avg_train = epoch_loss  / max(steps_per_epoch, 1)

        val_metrics = evaluate(model, val_loader, device, dtype, beta=1.0)

        metrics = {
            "epoch":      epoch + 1,
            "step":       global_step,
            "train_loss": avg_train,
            "train_beta": beta,          # 记录当前训练用的退火 β
            "lr":         lr,
            **val_metrics,
        }
        logger.log(metrics)

        # 定期保存
        if (epoch + 1) % tc["save_every"] == 0:
            logger.save_checkpoint(
                model, optimizer, epoch + 1, global_step,
                val_metrics["val_loss"], scaler,
            )
        else:
            # 始终更新 best（save_checkpoint 内部判断）
            logger.save_checkpoint(
                model, optimizer, epoch + 1, global_step,
                val_metrics["val_loss"], scaler,
            )

    print(f"\n[Train] 训练完成！最优 val_loss: {logger._best_val:.4f}")


# ─────────────────────────────────────────────
# 入口
# ─────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MusicVAE 训练")
    parser.add_argument("--resume",     action="store_true", help="从最新 checkpoint 继续训练")
    parser.add_argument("--preprocess", action="store_true", help="仅执行数据预处理后退出")
    args = parser.parse_args()

    for d in [cfg.CACHE_DIR, cfg.CHECKPOINT_DIR, cfg.LOG_DIR, cfg.OUTPUT_DIR]:
        os.makedirs(d, exist_ok=True)

    train(args)
