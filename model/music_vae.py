"""
MusicVAE —— 层次化变分自编码器 (~16M 参数)

架构:
  Encoder : 双向 Transformer  →  (mu, logvar) in R^z_dim
  Conductor: GRU              →  n_bars 个 bar embedding
  Decoder  : Causal Transformer + Cross-Attention (to bar embeddings)

参考:
  Roberts et al., "A Hierarchical Latent Vector Model for Learning
  Long-Term Structure in Music", ICML 2018.
"""

import math
from typing import Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# 复用已有的 RoPE 与 SwiGLU
from model.transformer import RotaryEmbedding, SwiGLUFFN


# ═══════════════════════════════════════════════════════════════════
# 1.  双向 Transformer Encoder
# ═══════════════════════════════════════════════════════════════════

class BidirectionalSelfAttention(nn.Module):
    """
    全注意力（非因果），用于 Encoder。
    与 CausalSelfAttention 相同，但 is_causal=False。
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0,
                 max_seq_len: int = 512):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads  = n_heads
        self.head_dim = d_model // n_heads

        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=False)
        self.o_proj   = nn.Linear(d_model, d_model,     bias=False)
        self.dropout  = dropout
        self.rope     = RotaryEmbedding(self.head_dim, max_seq_len=max_seq_len)

    def forward(self, x: Tensor, key_padding_mask: Optional[Tensor] = None) -> Tensor:
        """
        Args:
            x:                (B, T, d_model)
            key_padding_mask: (B, T) bool，True = 该位置是 PAD，需要被屏蔽
        """
        B, T, C = x.shape
        qkv = self.qkv_proj(x)
        q, k, v = qkv.split(C, dim=-1)

        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        q, k = self.rope(q, k)

        # 将 key_padding_mask 转换为 attn_mask (additive)
        attn_mask = None
        if key_padding_mask is not None:
            # (B, 1, 1, T) → broadcast 到 (B, n_heads, T_q, T_k)
            attn_mask = key_padding_mask[:, None, None, :].float() * -1e9

        attn_out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask = attn_mask,
            dropout_p = self.dropout if self.training else 0.0,
            is_causal = False,
        )
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, C)
        return self.o_proj(attn_out)


class BidirectionalTransformerBlock(nn.Module):
    """Pre-LN 双向 Transformer Block"""

    def __init__(self, d_model: int, n_heads: int, d_ff: int,
                 dropout: float = 0.0, max_seq_len: int = 512):
        super().__init__()
        self.ln1  = nn.LayerNorm(d_model)
        self.attn = BidirectionalSelfAttention(d_model, n_heads, dropout, max_seq_len)
        self.ln2  = nn.LayerNorm(d_model)
        self.ffn  = SwiGLUFFN(d_model, d_ff, dropout)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: Tensor,
                key_padding_mask: Optional[Tensor] = None) -> Tensor:
        x = x + self.drop(self.attn(self.ln1(x), key_padding_mask))
        x = x + self.drop(self.ffn(self.ln2(x)))
        return x


class BidirectionalTransformerEncoder(nn.Module):
    """
    将 token 序列编码为潜变量分布 (mu, logvar)。

    处理流程:
      token_ids → embedding → 4×BiTransformerBlock
        → mean-pooling (over non-PAD positions)
        → Linear → mu (z_dim)
        → Linear → logvar (z_dim)
    """

    def __init__(
        self,
        vocab_size: int,
        d_model:    int   = 256,
        n_heads:    int   = 8,
        n_layers:   int   = 4,
        d_ff:       int   = 1024,
        z_dim:      int   = 128,
        dropout:    float = 0.1,
        pad_token_id: int = 0,
        segment_len:  int = 256,
    ):
        super().__init__()
        self.pad_token_id = pad_token_id
        self.d_model      = d_model

        self.token_emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_token_id)
        self.emb_drop  = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            BidirectionalTransformerBlock(d_model, n_heads, d_ff, dropout,
                                          max_seq_len=segment_len * 2)
            for _ in range(n_layers)
        ])
        self.ln_final = nn.LayerNorm(d_model)

        self.mu_proj     = nn.Linear(d_model, z_dim, bias=False)
        self.logvar_proj = nn.Linear(d_model, z_dim, bias=False)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Args:
            x: (B, T) token ids

        Returns:
            mu:     (B, z_dim)
            logvar: (B, z_dim)
        """
        pad_mask = (x == self.pad_token_id)   # (B, T)  True = PAD

        h = self.emb_drop(self.token_emb(x))  # (B, T, d_model)
        for block in self.blocks:
            h = block(h, key_padding_mask=pad_mask)
        h = self.ln_final(h)                   # (B, T, d_model)

        # Mean-pool over non-PAD positions
        mask_float = (~pad_mask).float().unsqueeze(-1)          # (B, T, 1)
        pooled = (h * mask_float).sum(dim=1) / mask_float.sum(dim=1).clamp(min=1)
        # pooled: (B, d_model)

        mu     = self.mu_proj(pooled)          # (B, z_dim)
        logvar = self.logvar_proj(pooled)      # (B, z_dim)
        # 稳定性裁剪：避免 exp(logvar) 过大
        logvar = logvar.clamp(-10.0, 4.0)
        return mu, logvar


# ═══════════════════════════════════════════════════════════════════
# 2.  Hierarchical Conductor (z → bar embeddings)
# ═══════════════════════════════════════════════════════════════════

class HierarchicalConductor(nn.Module):
    """
    将潜变量 z 映射为 n_bars 个 bar embedding，供 Decoder 做 cross-attention。

    实现：
      z  → Linear → 初始 GRU hidden state (每层)
         → GRU 自回归 n_bars 步（输入为零向量）
         → n_bars × cond_hidden  作为 bar embeddings
    """

    def __init__(
        self,
        z_dim:       int = 128,
        cond_hidden: int = 256,
        n_bars:      int = 16,
        n_layers:    int = 2,
        dropout:     float = 0.1,
    ):
        super().__init__()
        self.n_bars      = n_bars
        self.n_layers    = n_layers
        self.cond_hidden = cond_hidden

        # z → 每层 GRU 的初始隐藏状态
        self.z_to_hidden = nn.Linear(z_dim, n_layers * cond_hidden, bias=True)

        # GRU：每步输入为上一步的 bar embedding（teacher-forced 为零向量，推理时同样）
        self.gru = nn.GRU(
            input_size  = cond_hidden,
            hidden_size = cond_hidden,
            num_layers  = n_layers,
            batch_first = True,
            dropout     = dropout if n_layers > 1 else 0.0,
        )
        self.out_norm = nn.LayerNorm(cond_hidden)

    def forward(self, z: Tensor) -> Tensor:
        """
        Args:
            z: (B, z_dim)

        Returns:
            bar_embs: (B, n_bars, cond_hidden)
        """
        B = z.size(0)

        # 初始 hidden state: (n_layers, B, cond_hidden)
        h0 = self.z_to_hidden(z)                             # (B, n_layers * cond_hidden)
        h0 = h0.view(B, self.n_layers, self.cond_hidden)     # (B, n_layers, cond_hidden)
        h0 = torch.tanh(h0).permute(1, 0, 2).contiguous()   # (n_layers, B, cond_hidden)

        # 输入序列：全零（GRU 的输入由 hidden state 驱动输出）
        inputs = torch.zeros(B, self.n_bars, self.cond_hidden,
                             device=z.device, dtype=z.dtype)  # (B, n_bars, cond_hidden)

        bar_embs, _ = self.gru(inputs, h0)    # (B, n_bars, cond_hidden)
        return self.out_norm(bar_embs)


# ═══════════════════════════════════════════════════════════════════
# 3.  Hierarchical Decoder (Causal Transformer + Cross-Attention)
# ═══════════════════════════════════════════════════════════════════

class CrossAttention(nn.Module):
    """
    Cross-Attention：Q 来自 decoder 隐层，K/V 来自 bar embeddings。
    bar embeddings 之间无因果遮罩（token 可自由关注所有 bar）。
    """

    def __init__(self, d_model: int, n_heads: int,
                 kv_dim: Optional[int] = None, dropout: float = 0.0):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads  = n_heads
        self.head_dim = d_model // n_heads
        kv_dim = kv_dim or d_model

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(kv_dim,  d_model, bias=False)
        self.v_proj = nn.Linear(kv_dim,  d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = dropout

    def forward(self, x: Tensor, context: Tensor) -> Tensor:
        """
        Args:
            x:       (B, T, d_model)   — query source (decoder hidden states)
            context: (B, S, kv_dim)    — key/value source (bar embeddings)

        Returns:
            (B, T, d_model)
        """
        B, T, C = x.shape
        S = context.size(1)

        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(context).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(context).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)

        out = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p = self.dropout if self.training else 0.0,
            is_causal = False,
        )
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.o_proj(out)


class CausalSelfAttentionDecoder(nn.Module):
    """因果自注意力（Decoder 内部，与 transformer.py 中的版本相同但独立实现）"""

    def __init__(self, d_model: int, n_heads: int,
                 dropout: float = 0.0, max_seq_len: int = 512):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads  = n_heads
        self.head_dim = d_model // n_heads

        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=False)
        self.o_proj   = nn.Linear(d_model, d_model,     bias=False)
        self.dropout  = dropout
        self.rope     = RotaryEmbedding(self.head_dim, max_seq_len=max_seq_len)

    def forward(self, x: Tensor) -> Tensor:
        B, T, C = x.shape
        qkv = self.qkv_proj(x)
        q, k, v = qkv.split(C, dim=-1)

        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        q, k = self.rope(q, k)

        out = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p = self.dropout if self.training else 0.0,
            is_causal = True,
        )
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.o_proj(out)


class VAEDecoderBlock(nn.Module):
    """
    Pre-LN Decoder Block:
      x = x + SelfAttn(LN(x))
      x = x + CrossAttn(LN(x), bar_embs)
      x = x + FFN(LN(x))
    """

    def __init__(self, d_model: int, n_heads: int, d_ff: int,
                 kv_dim: int, dropout: float = 0.0, max_seq_len: int = 512):
        super().__init__()
        self.ln1       = nn.LayerNorm(d_model)
        self.self_attn = CausalSelfAttentionDecoder(d_model, n_heads, dropout, max_seq_len)
        self.ln2       = nn.LayerNorm(d_model)
        self.cross_attn = CrossAttention(d_model, n_heads, kv_dim=kv_dim, dropout=dropout)
        self.ln3       = nn.LayerNorm(d_model)
        self.ffn       = SwiGLUFFN(d_model, d_ff, dropout)
        self.drop      = nn.Dropout(dropout)

    def forward(self, x: Tensor, bar_embs: Tensor) -> Tensor:
        """
        Args:
            x:        (B, T, d_model)
            bar_embs: (B, n_bars, kv_dim)
        """
        x = x + self.drop(self.self_attn(self.ln1(x)))
        x = x + self.drop(self.cross_attn(self.ln2(x), bar_embs))
        x = x + self.drop(self.ffn(self.ln3(x)))
        return x


class HierarchicalTransformerDecoder(nn.Module):
    """
    以 bar embeddings 为条件，自回归解码 token 序列。
    权重共享：token embedding 与 Encoder 共用（需在外部传入/绑定）。
    """

    def __init__(
        self,
        vocab_size:   int,
        d_model:      int   = 256,
        n_heads:      int   = 8,
        n_layers:     int   = 8,
        d_ff:         int   = 1024,
        kv_dim:       int   = 256,   # bar embedding 维度 (= cond_hidden)
        dropout:      float = 0.1,
        pad_token_id: int   = 0,
        segment_len:  int   = 256,
    ):
        super().__init__()
        self.d_model      = d_model
        self.pad_token_id = pad_token_id
        self.segment_len  = segment_len

        self.token_emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_token_id)
        self.emb_drop  = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            VAEDecoderBlock(d_model, n_heads, d_ff, kv_dim, dropout,
                            max_seq_len=segment_len * 2)
            for _ in range(n_layers)
        ])
        self.ln_final = nn.LayerNorm(d_model)
        self.lm_head  = nn.Linear(d_model, vocab_size, bias=False)
        # Weight tying (Decoder 内部)
        self.lm_head.weight = self.token_emb.weight

    def forward(self, x: Tensor, bar_embs: Tensor) -> Tensor:
        """
        Args:
            x:        (B, T) token ids（decoder 输入，teacher-forced）
            bar_embs: (B, n_bars, kv_dim)

        Returns:
            logits: (B, T, vocab_size)
        """
        h = self.emb_drop(self.token_emb(x))   # (B, T, d_model)
        for block in self.blocks:
            h = block(h, bar_embs)
        h = self.ln_final(h)
        return self.lm_head(h)                  # (B, T, vocab_size)

    @torch.no_grad()
    def generate(
        self,
        bar_embs:    Tensor,
        bos_id:      int,
        eos_id:      int,
        max_steps:   int   = 256,
        temperature: float = 0.85,
        top_k:       int   = 0,
        top_p:       float = 0.9,
    ) -> Tensor:
        """
        给定 bar_embs，自回归生成一段 token 序列。

        Args:
            bar_embs: (1, n_bars, kv_dim)  — 单个样本
            bos_id:   BOS token id
            eos_id:   EOS token id
            max_steps: 最多生成多少 token

        Returns:
            ids: (1, T_generated)
        """
        device = bar_embs.device
        ids    = torch.tensor([[bos_id]], dtype=torch.long, device=device)

        for _ in range(max_steps):
            logits = self.forward(ids, bar_embs)  # (1, T, vocab_size)
            logits = logits[:, -1, :]              # (1, vocab_size)

            # 温度
            logits = logits / max(temperature, 1e-8)

            # Top-k
            if top_k > 0:
                top_k_val = min(top_k, logits.size(-1))
                values, _ = torch.topk(logits, top_k_val)
                logits[logits < values[:, -1:]] = float('-inf')

            # Top-p (nucleus)
            if top_p < 1.0:
                sorted_logits, sorted_idx = torch.sort(logits, descending=True)
                cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                remove = cum_probs - F.softmax(sorted_logits, dim=-1) > top_p
                sorted_logits[remove] = float('-inf')
                logits = torch.zeros_like(logits).scatter_(1, sorted_idx, sorted_logits)

            probs   = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)  # (1, 1)
            ids     = torch.cat([ids, next_id], dim=1)

            if next_id.item() == eos_id:
                break

        return ids


# ═══════════════════════════════════════════════════════════════════
# 4.  MusicVAE —— 整合模型
# ═══════════════════════════════════════════════════════════════════

class MusicVAE(nn.Module):
    """
    层次化 Music VAE。

    组件:
      encoder   : BidirectionalTransformerEncoder
      conductor : HierarchicalConductor
      decoder   : HierarchicalTransformerDecoder

    前向传播返回:
      logits  (B, T, vocab_size)
      loss    scalar  (ELBO = recon + beta * kl)
      recon_loss scalar
      kl_loss scalar

    Args:
        vocab_size:    REMI 词表大小
        segment_len:   每段 token 长度（= encoder 输入长度）
        n_bars:        conductor 层次步数
        z_dim:         潜变量维度
        enc_d_model, enc_n_heads, enc_n_layers, enc_d_ff: Encoder 超参
        cond_hidden, cond_n_layers:                       Conductor 超参
        dec_d_model, dec_n_heads, dec_n_layers, dec_d_ff: Decoder 超参
        dropout:       dropout 比率
        pad_token_id:  PAD token id
        free_bits:     每维 KL 的 free bits 下限（nats）
    """

    def __init__(
        self,
        vocab_size:    int,
        segment_len:   int   = 256,
        n_bars:        int   = 16,
        z_dim:         int   = 128,
        # Encoder
        enc_d_model:   int   = 256,
        enc_n_heads:   int   = 8,
        enc_n_layers:  int   = 4,
        enc_d_ff:      int   = 1024,
        # Conductor
        cond_hidden:   int   = 256,
        cond_n_layers: int   = 2,
        # Decoder
        dec_d_model:   int   = 256,
        dec_n_heads:   int   = 8,
        dec_n_layers:  int   = 8,
        dec_d_ff:      int   = 1024,
        # Shared
        dropout:       float = 0.1,
        pad_token_id:  int   = 0,
        free_bits:     float = 0.5,
    ):
        super().__init__()
        self.segment_len  = segment_len
        self.n_bars       = n_bars
        self.z_dim        = z_dim
        self.free_bits    = free_bits
        self.pad_token_id = pad_token_id

        # ── 子模块 ────────────────────────────
        self.encoder = BidirectionalTransformerEncoder(
            vocab_size   = vocab_size,
            d_model      = enc_d_model,
            n_heads      = enc_n_heads,
            n_layers     = enc_n_layers,
            d_ff         = enc_d_ff,
            z_dim        = z_dim,
            dropout      = dropout,
            pad_token_id = pad_token_id,
            segment_len  = segment_len,
        )
        self.conductor = HierarchicalConductor(
            z_dim       = z_dim,
            cond_hidden = cond_hidden,
            n_bars      = n_bars,
            n_layers    = cond_n_layers,
            dropout     = dropout,
        )
        self.decoder = HierarchicalTransformerDecoder(
            vocab_size   = vocab_size,
            d_model      = dec_d_model,
            n_heads      = dec_n_heads,
            n_layers     = dec_n_layers,
            d_ff         = dec_d_ff,
            kv_dim       = cond_hidden,
            dropout      = dropout,
            pad_token_id = pad_token_id,
            segment_len  = segment_len,
        )

        # ── 初始化参数 ─────────────────────────
        self._init_weights()

        n_params = sum(p.numel() for p in self.parameters())
        print(f"[MusicVAE] 参数量: {n_params/1e6:.2f}M")
        self._print_breakdown()

    # ── 参数初始化 ────────────────────────────────────────────────

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, nn.GRU):
                for name, p in module.named_parameters():
                    if "weight" in name:
                        nn.init.orthogonal_(p)
                    elif "bias" in name:
                        nn.init.zeros_(p)

    def _print_breakdown(self):
        enc_p  = sum(p.numel() for p in self.encoder.parameters())
        cond_p = sum(p.numel() for p in self.conductor.parameters())
        dec_p  = sum(p.numel() for p in self.decoder.parameters())
        print(f"  Encoder:   {enc_p/1e6:.2f}M")
        print(f"  Conductor: {cond_p/1e6:.2f}M")
        print(f"  Decoder:   {dec_p/1e6:.2f}M")

    # ── 核心工具 ──────────────────────────────────────────────────

    @staticmethod
    def reparameterize(mu: Tensor, logvar: Tensor) -> Tensor:
        """z = mu + eps * std，eps ~ N(0,I)"""
        if not torch.is_grad_enabled():
            return mu   # 推理时直接用 mu（确定性）
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def kl_loss(self, mu: Tensor, logvar: Tensor) -> Tuple[Tensor, Tensor]:
        """
        计算 KL 散度，并应用 free bits 下限。

        Returns:
            kl_loss:   scalar（含 free bits，用于反向传播）
            kl_raw:    scalar（不含 free bits，用于日志记录）
        """
        # (B, z_dim)
        kl_per_dim = -0.5 * (1.0 + logvar - mu.pow(2) - logvar.exp())
        kl_raw     = kl_per_dim.mean()   # 均值，方便比较

        # Free bits：每维 KL < free_bits 时停止梯度，防止后验崩溃
        if self.free_bits > 0.0:
            kl_per_dim = kl_per_dim.clamp(min=self.free_bits)

        kl_loss = kl_per_dim.mean()
        return kl_loss, kl_raw

    # ── 前向传播 ──────────────────────────────────────────────────

    def forward(
        self,
        enc_input:  Tensor,           # (B, T) encoder 输入
        dec_input:  Tensor,           # (B, T) decoder 输入（teacher-forcing, = enc_input[:-1]）
        dec_target: Tensor,           # (B, T) decoder 目标（= enc_input[1:]）
        beta:       float = 1.0,      # KL 权重（β-annealing 时外部传入）
    ):
        """
        Returns:
            loss:       ELBO = recon_loss + beta * kl_loss
            recon_loss: 重建 loss（cross-entropy）
            kl_loss:    KL 散度（含 free bits）
            kl_raw:     KL 散度（不含 free bits，用于监控）
            logits:     (B, T, vocab_size)
        """
        # 1. Encode
        mu, logvar = self.encoder(enc_input)         # (B, z_dim) ×2

        # 2. Reparameterize
        z = self.reparameterize(mu, logvar)          # (B, z_dim)

        # 3. Conduct: z → bar embeddings
        bar_embs = self.conductor(z)                  # (B, n_bars, cond_hidden)

        # 4. Decode
        logits = self.decoder(dec_input, bar_embs)   # (B, T, vocab_size)

        # 5. Reconstruction loss (cross-entropy, ignore PAD)
        recon_loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            dec_target.reshape(-1),
            ignore_index = self.pad_token_id,
        )

        # 6. KL loss
        kl_loss, kl_raw = self.kl_loss(mu, logvar)

        # 7. ELBO
        loss = recon_loss + beta * kl_loss

        return loss, recon_loss, kl_loss, kl_raw, logits

    # ── 生成接口 ──────────────────────────────────────────────────

    @torch.no_grad()
    def sample(
        self,
        n:           int   = 1,
        bos_id:      int   = 1,
        eos_id:      int   = 2,
        device:      str   = "cpu",
        temperature: float = 0.85,
        top_k:       int   = 0,
        top_p:       float = 0.9,
        max_steps:   int   = 256,
        z:           Optional[Tensor] = None,
    ) -> List[List[int]]:
        """
        从标准正态分布采样 z（或使用传入的 z），解码得到 token 序列列表。

        Args:
            n:    生成曲目数量
            z:    (n, z_dim) 可选，若传入则跳过随机采样

        Returns:
            list of token id lists，长度 n
        """
        self.eval()
        if z is None:
            z = torch.randn(n, self.z_dim, device=device)

        results = []
        for i in range(n):
            zi        = z[i:i+1]                                          # (1, z_dim)
            bar_embs  = self.conductor(zi)                                 # (1, n_bars, cond_hidden)
            ids       = self.decoder.generate(
                bar_embs    = bar_embs,
                bos_id      = bos_id,
                eos_id      = eos_id,
                max_steps   = max_steps,
                temperature = temperature,
                top_k       = top_k,
                top_p       = top_p,
            )
            results.append(ids[0].cpu().tolist())
        return results

    @torch.no_grad()
    def encode(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """
        将 token 序列编码为 (mu, logvar)，用于重建或插值。

        Args:
            x: (B, T)

        Returns:
            mu (B, z_dim), logvar (B, z_dim)
        """
        self.eval()
        return self.encoder(x)

    @torch.no_grad()
    def reconstruct(
        self,
        x:           Tensor,
        bos_id:      int,
        eos_id:      int,
        use_mean:    bool  = True,
        temperature: float = 0.85,
        top_p:       float = 0.9,
        max_steps:   int   = 256,
    ) -> List[List[int]]:
        """
        编码 x，再解码（重建）。use_mean=True 使用 mu（确定性）。

        Args:
            x: (B, T) 原始 token 序列

        Returns:
            list of token id lists
        """
        self.eval()
        mu, logvar = self.encoder(x)
        z = mu if use_mean else self.reparameterize(mu, logvar)
        return self.sample(
            n           = x.size(0),
            bos_id      = bos_id,
            eos_id      = eos_id,
            device      = x.device.type,
            temperature = temperature,
            top_p       = top_p,
            max_steps   = max_steps,
            z           = z,
        )

    @torch.no_grad()
    def interpolate(
        self,
        x1:          Tensor,
        x2:          Tensor,
        steps:       int   = 8,
        bos_id:      int   = 1,
        eos_id:      int   = 2,
        temperature: float = 0.85,
        top_p:       float = 0.9,
        max_steps:   int   = 256,
    ) -> List[List[int]]:
        """
        在 x1 和 x2 的潜空间之间做球形线性插值（slerp）并解码。

        Args:
            x1, x2: (1, T)，起止曲目的 token 序列

        Returns:
            list of token id lists，长度 steps（含两端点）
        """
        self.eval()
        mu1, _ = self.encoder(x1)
        mu2, _ = self.encoder(x2)

        # 球形线性插值 (slerp)
        def slerp(t: float, v0: Tensor, v1: Tensor) -> Tensor:
            v0_n = F.normalize(v0, dim=-1)
            v1_n = F.normalize(v1, dim=-1)
            dot  = (v0_n * v1_n).sum(dim=-1, keepdim=True).clamp(-1, 1)
            theta = torch.acos(dot)
            sin_theta = torch.sin(theta)
            # 若两向量几乎平行，退化为线性插值
            mask = (sin_theta.abs() < 1e-6)
            w0   = torch.where(mask, torch.tensor(1 - t, device=v0.device),
                               torch.sin((1 - t) * theta) / (sin_theta + 1e-8))
            w1   = torch.where(mask, torch.tensor(t, device=v0.device),
                               torch.sin(t * theta) / (sin_theta + 1e-8))
            return w0 * v0 + w1 * v1

        z_list = []
        for i in range(steps):
            t = i / max(steps - 1, 1)
            z_list.append(slerp(t, mu1, mu2))
        z_all = torch.cat(z_list, dim=0)  # (steps, z_dim)

        return self.sample(
            n           = steps,
            bos_id      = bos_id,
            eos_id      = eos_id,
            device      = x1.device.type,
            temperature = temperature,
            top_p       = top_p,
            max_steps   = max_steps,
            z           = z_all,
        )


# ═══════════════════════════════════════════════════════════════════
# 快速冒烟测试
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    VOCAB  = 450
    SEG    = 256
    B, T   = 4, SEG

    model = MusicVAE(
        vocab_size  = VOCAB,
        segment_len = SEG,
        n_bars      = 16,
        z_dim       = 128,
        enc_d_model = 256, enc_n_heads = 8, enc_n_layers = 4, enc_d_ff = 1024,
        cond_hidden = 256, cond_n_layers = 2,
        dec_d_model = 256, dec_n_heads = 8, dec_n_layers = 8, dec_d_ff = 1024,
        dropout     = 0.1,
    )

    x       = torch.randint(1, VOCAB, (B, T))
    dec_in  = x[:, :-1]
    dec_tgt = x[:, 1:]

    loss, recon, kl, kl_raw, logits = model(x, dec_in, dec_tgt, beta=0.5)
    print(f"loss={loss:.4f}  recon={recon:.4f}  kl={kl:.4f}  kl_raw={kl_raw:.4f}")
    print(f"logits: {logits.shape}")

    # 采样测试
    seqs = model.sample(n=2, bos_id=1, eos_id=2, device="cpu", max_steps=32)
    print(f"采样结果: {[len(s) for s in seqs]} tokens")

    # 插值测试
    x1 = torch.randint(1, VOCAB, (1, T))
    x2 = torch.randint(1, VOCAB, (1, T))
    interp = model.interpolate(x1, x2, steps=4, bos_id=1, eos_id=2, max_steps=32)
    print(f"插值结果: {[len(s) for s in interp]} tokens")
