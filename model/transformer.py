"""
Decoder-only Transformer (GPT 风格)
特性:
  - RoPE 旋转位置编码
  - SwiGLU 激活函数
  - Pre-LayerNorm
  - Causal (masked) self-attention
  - 约 20M 参数 @ d_model=512, n_layers=6, n_heads=8
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ─────────────────────────────────────────────
# RoPE (Rotary Position Embedding)
# ─────────────────────────────────────────────

class RotaryEmbedding(nn.Module):
    """
    旋转位置编码 (Su et al., 2021)
    不引入额外可学习参数，直接以旋转矩阵的形式注入位置信息。
    """

    def __init__(self, dim: int, max_seq_len: int = 4096, base: int = 10000):
        super().__init__()
        # 频率向量，shape: (dim/2,)
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        # 预计算 cos/sin 缓存
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int):
        t = torch.arange(seq_len, device=self.inv_freq.device).float()
        freqs = torch.outer(t, self.inv_freq)          # (seq_len, dim/2)
        emb   = torch.cat([freqs, freqs], dim=-1)      # (seq_len, dim)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :])  # (1,1,T,dim)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :])

    @staticmethod
    def _rotate_half(x: Tensor) -> Tensor:
        """将后半维度取负并前移，实现 -sin 旋转分量。"""
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat([-x2, x1], dim=-1)

    def forward(self, q: Tensor, k: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Args:
            q, k: (B, n_heads, T, head_dim)
        Returns:
            rotated q, k
        """
        seq_len = q.shape[2]
        cos = self.cos_cached[:, :, :seq_len, :]
        sin = self.sin_cached[:, :, :seq_len, :]
        q = q * cos + self._rotate_half(q) * sin
        k = k * cos + self._rotate_half(k) * sin
        return q, k


# ─────────────────────────────────────────────
# SwiGLU Feed-Forward Network
# ─────────────────────────────────────────────

class SwiGLUFFN(nn.Module):
    """
    SwiGLU (Noam Shazeer, 2020)
    FFN(x) = (W1 x ⊙ SiLU(W2 x)) W3
    d_ff 通常设为 d_model * 8/3 取整，这里直接用 config 的 d_ff。
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0):
        super().__init__()
        # gate + up 合并为一个线性层，减少 kernel launch 开销
        self.w_gate_up = nn.Linear(d_model, d_ff * 2, bias=False)
        self.w_down    = nn.Linear(d_ff, d_model, bias=False)
        self.dropout   = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        gate, up = self.w_gate_up(x).chunk(2, dim=-1)
        return self.dropout(self.w_down(F.silu(gate) * up))


# ─────────────────────────────────────────────
# Causal Multi-Head Self-Attention
# ─────────────────────────────────────────────

class CausalSelfAttention(nn.Module):
    """
    因果（单向）多头自注意力，内置 RoPE。
    使用 scaled_dot_product_attention 以支持 Flash Attention（PyTorch >= 2.0）。
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0, max_seq_len: int = 2048):
        super().__init__()
        assert d_model % n_heads == 0, "d_model 必须能被 n_heads 整除"
        self.n_heads  = n_heads
        self.head_dim = d_model // n_heads

        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=False)
        self.o_proj   = nn.Linear(d_model, d_model,     bias=False)
        self.dropout  = dropout

        self.rope = RotaryEmbedding(self.head_dim, max_seq_len=max_seq_len)

    def forward(self, x: Tensor, attention_mask: Optional[Tensor] = None) -> Tensor:
        """
        Args:
            x:              (B, T, d_model)
            attention_mask: 保留接口，实际通过 is_causal=True 实现因果遮罩
        Returns:
            (B, T, d_model)
        """
        B, T, C = x.shape
        # 计算 Q K V，并 reshape 成多头形式
        qkv = self.qkv_proj(x)                                    # (B, T, 3*C)
        q, k, v = qkv.split(C, dim=-1)                            # 各 (B, T, C)

        # reshape: (B, T, C) → (B, n_heads, T, head_dim)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        # 应用 RoPE
        q, k = self.rope(q, k)

        # Flash Attention (PyTorch 2.0+)，is_causal=True 自动处理因果遮罩
        attn_out = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p = self.dropout if self.training else 0.0,
            is_causal = True,
        )  # (B, n_heads, T, head_dim)

        # 合并多头
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, C)
        return self.o_proj(attn_out)


# ─────────────────────────────────────────────
# Transformer Block (Pre-LayerNorm)
# ─────────────────────────────────────────────

class TransformerBlock(nn.Module):
    """
    Pre-LN Transformer Block:
      x = x + Attention(LN(x))
      x = x + FFN(LN(x))
    """

    def __init__(self, d_model: int, n_heads: int, d_ff: int,
                 dropout: float = 0.0, max_seq_len: int = 2048):
        super().__init__()
        self.ln1  = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, dropout, max_seq_len)
        self.ln2  = nn.LayerNorm(d_model)
        self.ffn  = SwiGLUFFN(d_model, d_ff, dropout)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.drop(self.attn(self.ln1(x)))
        x = x + self.drop(self.ffn(self.ln2(x)))
        return x


# ─────────────────────────────────────────────
# 完整模型
# ─────────────────────────────────────────────

class MusicTransformer(nn.Module):
    """
    用于 MIDI 生成的 Decoder-only Transformer。

    Args:
        vocab_size:   词表大小（由 tokenizer 决定）
        context_len:  最大序列长度
        d_model:      隐层维度
        n_heads:      注意力头数
        n_layers:     Transformer 层数
        d_ff:         FFN 中间维度
        dropout:      dropout 比率
        pad_token_id: PAD 的 token id（embedding 设为 0 梯度）
    """

    def __init__(
        self,
        vocab_size:   int,
        context_len:  int   = 1024,
        d_model:      int   = 512,
        n_heads:      int   = 8,
        n_layers:     int   = 6,
        d_ff:         int   = 2048,
        dropout:      float = 0.1,
        pad_token_id: int   = 0,
    ):
        super().__init__()
        self.context_len  = context_len
        self.pad_token_id = pad_token_id
        self.d_model      = d_model

        # Token embedding（不使用 position embedding，RoPE 已在注意力内处理）
        self.token_emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_token_id)
        self.emb_drop  = nn.Dropout(dropout)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout, max_seq_len=context_len * 2)
            for _ in range(n_layers)
        ])

        # 最终 LayerNorm
        self.ln_final = nn.LayerNorm(d_model)

        # 语言模型头（与 token_emb 权重共享，节省参数并提升泛化）
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight  # weight tying

        # 参数初始化
        self._init_weights()

        n_params = sum(p.numel() for p in self.parameters())
        print(f"[Model] MusicTransformer | 参数量: {n_params/1e6:.2f}M")

    def _init_weights(self):
        """GPT-2 风格参数初始化"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x: Tensor, targets: Optional[Tensor] = None):
        """
        Args:
            x:       (B, T) token ids
            targets: (B, T) 目标 token ids（训练时提供）

        Returns:
            logits: (B, T, vocab_size)
            loss:   cross-entropy loss（仅训练时返回，否则为 None）
        """
        B, T = x.shape
        assert T <= self.context_len, f"序列长度 {T} 超过最大 context_len {self.context_len}"

        # Token embedding + dropout
        h = self.emb_drop(self.token_emb(x))    # (B, T, d_model)

        # Transformer blocks
        for block in self.blocks:
            h = block(h)

        h = self.ln_final(h)                     # (B, T, d_model)
        logits = self.lm_head(h)                 # (B, T, vocab_size)

        loss = None
        if targets is not None:
            # 忽略 PAD token 的损失
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=self.pad_token_id,
            )

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        prompt:      Tensor,
        max_new_tokens: int   = 512,
        temperature: float = 1.0,
        top_k:       int   = 0,
        top_p:       float = 0.9,
        repetition_penalty: float = 1.1,
        eos_token_id: Optional[int] = None,
    ) -> Tensor:
        """
        自回归生成。

        Args:
            prompt:      (1, T_prompt) 起始 token 序列
            max_new_tokens: 最多生成多少个新 token
            temperature: 采样温度（越低越保守）
            top_k:       top-k 截断（0 = 不使用）
            top_p:       nucleus sampling 阈值
            repetition_penalty: 对已出现 token 的 logit 惩罚
            eos_token_id: 遇到 EOS 提前停止

        Returns:
            (1, T_prompt + generated) 完整 token 序列
        """
        self.eval()
        ids = prompt.clone()

        for _ in range(max_new_tokens):
            # 截断到最大 context_len
            ids_cond = ids if ids.size(1) <= self.context_len else ids[:, -self.context_len:]

            logits, _ = self(ids_cond)
            logits = logits[:, -1, :]    # 取最后一个时间步 (1, vocab_size)

            # 重复惩罚
            if repetition_penalty != 1.0:
                for token_id in set(ids[0].tolist()):
                    if logits[0, token_id] > 0:
                        logits[0, token_id] /= repetition_penalty
                    else:
                        logits[0, token_id] *= repetition_penalty

            # 温度缩放
            logits = logits / max(temperature, 1e-8)

            # Top-k 截断
            if top_k > 0:
                top_k = min(top_k, logits.size(-1))
                values, _ = torch.topk(logits, top_k)
                logits[logits < values[:, -1:]] = float('-inf')

            # Nucleus (top-p) 采样
            if top_p < 1.0:
                sorted_logits, sorted_idx = torch.sort(logits, descending=True)
                cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                # 移除累积概率超过 top_p 的 token
                sorted_remove = cum_probs - F.softmax(sorted_logits, dim=-1) > top_p
                sorted_logits[sorted_remove] = float('-inf')
                logits = torch.zeros_like(logits).scatter_(1, sorted_idx, sorted_logits)

            # 采样
            probs     = F.softmax(logits, dim=-1)
            next_id   = torch.multinomial(probs, num_samples=1)   # (1, 1)
            ids       = torch.cat([ids, next_id], dim=1)

            # EOS 提前停止
            if eos_token_id is not None and next_id.item() == eos_token_id:
                break

        return ids


if __name__ == "__main__":
    # 快速冒烟测试
    model = MusicTransformer(vocab_size=400, context_len=1024)
    x = torch.randint(0, 400, (2, 128))
    logits, loss = model(x, targets=x)
    print(f"logits: {logits.shape}  loss: {loss.item():.4f}")

    # 生成测试
    prompt = torch.randint(0, 400, (1, 8))
    out = model.generate(prompt, max_new_tokens=20)
    print(f"生成结果形状: {out.shape}")
