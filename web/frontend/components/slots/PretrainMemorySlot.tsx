'use client';

import { useMemo } from 'react';
import type { SlotProps } from '@/lib/module-schema';

function formatParams(n: number): string {
  if (n >= 1e9) return `${(n / 1e9).toFixed(1)}B`;
  if (n >= 1e6) return `${(n / 1e6).toFixed(1)}M`;
  if (n >= 1e3) return `${(n / 1e3).toFixed(1)}K`;
  return `${n}`;
}

function formatBytes(bytes: number): string {
  if (bytes >= 1e9) return `${(bytes / 1e9).toFixed(1)} GB`;
  return `${(bytes / 1e6).toFixed(0)} MB`;
}

function estimateMemory(cfg: {
  n_layers: number; n_heads: number; dim: number;
  max_seq_len: number; batch_size: number; optimizer: string;
  total_batch_size: number; compile: boolean;
}): { params: number; peakBytes: number; gradAccumSteps: number; tokensPerStep: number } {
  const vocabPadded = Math.ceil(50257 / 64) * 64; // 50304
  const hiddenDim = 4 * cfg.dim;
  const headDim = cfg.dim / cfg.n_heads;

  const paramsPerBlock =
    4 * cfg.dim * cfg.dim +      // q/k/v/out projections
    2 * headDim +                 // qk-norm
    3 * cfg.dim * hiddenDim +     // SwiGLU (gate_up + down)
    2 * cfg.dim;                  // attn_norm + mlp_norm

  const params = vocabPadded * cfg.dim + cfg.n_layers * paramsPerBlock + cfg.dim;

  // Gradient accumulation
  const tokensPerMicroBatch = cfg.batch_size * cfg.max_seq_len;
  const gradAccumSteps = Math.max(1, Math.floor(cfg.total_batch_size / tokensPerMicroBatch));
  const tokensPerStep = gradAccumSteps * tokensPerMicroBatch;
  const B = cfg.batch_size;
  const T = cfg.max_seq_len;
  const D = cfg.dim;
  const V = vocabPadded;

  // --- Model weights (bf16) ---
  const modelMem = params * 2;

  // --- Gradients (bf16, accumulate in-place) ---
  const gradMem = params * 2;

  // --- Optimizer states ---
  // AdamW: fp32 master weights + momentum + variance = 12 bytes/param
  // Muon: fp32 momentum = 4 bytes/param
  let optimMem: number;
  if (cfg.optimizer === 'adamw') {
    optimMem = params * 12;
  } else if (cfg.optimizer === 'muon') {
    optimMem = params * 4;
  } else {
    const embedNormParams = vocabPadded * cfg.dim + cfg.n_layers * 2 * cfg.dim + cfg.dim;
    const muonParams = params - embedNormParams;
    optimMem = embedNormParams * 12 + muonParams * 4;
  }

  // --- Activation memory (per micro-batch, only one live at a time) ---
  // Per transformer layer, saved tensors for backward (bf16 = 2 bytes):
  //   norm inputs (2x):                    2 * B*T*D
  //   Q, K, V projections:                 3 * B*T*D
  //   flash attn: Q,K,V,O + logsumexp:     4 * B*T*D + B*H*T*4
  //   out_proj input:                       B*T*D
  //   SwiGLU gate (SiLU bwd):              B*T*4D
  //   SwiGLU up (mul bwd):                 B*T*4D
  //   SwiGLU mul result (down_proj bwd):   B*T*4D
  //   residual saves:                       2 * B*T*D
  // Total: ~10*B*T*D + 12*B*T*D = 22*B*T*D elements * 2 bytes = 44 bytes per B*T*D
  const activationPerLayer = B * T * D * 44;
  const activationMem = activationPerLayer * cfg.n_layers;

  // --- Logits + backward peak ---
  // Peak occurs during cross_entropy backward when these coexist:
  //   forward logits (bf16):                 B*T*V * 2  (saved for CE backward)
  //   CE backward softmax probs (bf16):      B*T*V * 2  (internal computation)
  //   CE backward gradient output (bf16):    B*T*V * 2  (grad flowing to lm_head)
  const logitsMem = B * T * V * 6;

  // --- Input/target tensors (full grad accum batch on GPU) ---
  const inputMem = 2 * gradAccumSteps * B * T * 8; // int64

  // --- CUDA overhead ---
  // CUDA context, cuBLAS/cuDNN workspace, driver allocations
  const cudaOverhead = 500 * 1024 * 1024; // ~500 MB

  // --- torch.compile overhead ---
  // Triton kernels, compiled graph buffers, autotuning workspace
  const compileOverhead = cfg.compile ? 1024 * 1024 * 1024 : 0; // ~1 GB

  // --- Fragmentation ---
  // PyTorch CUDA allocator rounds to block sizes and caches freed blocks
  const subtotal = modelMem + gradMem + optimMem + activationMem + logitsMem + inputMem + cudaOverhead + compileOverhead;
  const fragmentation = 1.10; // ~10% overhead

  return {
    params,
    peakBytes: Math.round(subtotal * fragmentation),
    gradAccumSteps,
    tokensPerStep,
  };
}

export function PretrainMemorySlot({ config }: SlotProps) {
  const est = useMemo(() => estimateMemory({
    n_layers: config.n_layers as number,
    n_heads: config.n_heads as number,
    dim: config.dim as number,
    max_seq_len: config.max_seq_len as number,
    batch_size: config.batch_size as number,
    optimizer: config.optimizer as string,
    total_batch_size: (config.total_batch_size as number) || 524288,
    compile: (config.compile as boolean) ?? true,
  }), [config.n_layers, config.n_heads, config.dim, config.max_seq_len, config.batch_size, config.optimizer, config.total_batch_size, config.compile]);

  return (
    <div style={{ fontSize: '9px', color: '#999', textAlign: 'center', padding: '2px 0', lineHeight: '1.6' }}>
      <div>{formatParams(est.params)} params · ~{formatBytes(est.peakBytes)} VRAM</div>
      <div>{est.gradAccumSteps} grad accum steps · {(est.tokensPerStep / 1024).toFixed(0)}K tok/step</div>
    </div>
  );
}
