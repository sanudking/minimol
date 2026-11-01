const D = 384, H = 6, HD = 64, FF = 1536, SCALE = 1 / Math.sqrt(HD);

function erfApprox(x) {
  // Abramowitz-Stegun 7.1.26; deterministic and close to erf, but not bit-identical to PyTorch.
  const sign = x < 0 ? -1 : 1; x = Math.abs(x);
  const t = 1 / (1 + 0.3275911 * x);
  const y = 1 - (((((1.061405429*t - 1.453152027)*t) + 1.421413741)*t - 0.284496736)*t + 0.254829592)*t*Math.exp(-x*x);
  return sign * y;
}

export class PartitionEngine {
  constructor(role, manifest, buffer) {
    this.role = role;
    this.manifest = manifest;
    this.weights = new Map();
    for (const item of manifest.tensors) {
      if (item.owner !== role) continue;
      if (item.partition_offset_bytes % 4 !== 0) throw new Error(`Unaligned tensor offset: ${item.name}`);
      if (item.partition_offset_bytes + item.numel * 4 > buffer.byteLength) throw new Error(`Tensor exceeds ${role} partition: ${item.name}`);
      // The partition is little-endian FP32 and all supported phone platforms
      // are little-endian. A view avoids copying every tensor during startup.
      this.weights.set(item.name, new Float32Array(buffer, item.partition_offset_bytes, item.numel));
    }
    this.reset();
  }

  reset() { this.kv = new Map(); this.position = 0; }

  // Preheat the hot inference functions while the model is loading. This gives
  // the browser JS engine a chance to optimize them before timed generation.
  // The synthetic state is discarded so it cannot affect real inference.
  warmup(iterations = 8) {
    if (this.role === 'phone1') {
      for (let i = 0; i < iterations; i++) this.phone1Token(0, i);
    } else {
      const activation = new Float32Array(D);
      for (let i = 0; i < iterations; i++) this.phone2Token(activation);
    }
    this.reset();
  }
  w(name) { const value = this.weights.get(name); if (!value) throw new Error(`Missing ${this.role} tensor: ${name}`); return value; }

  linear(x, prefix, outSize, inSize, bias = true) {
    const weight = this.w(`${prefix}.weight`), b = bias ? this.w(`${prefix}.bias`) : null;
    const out = new Float32Array(outSize);
    for (let o = 0; o < outSize; o++) {
      let sum = b ? b[o] : 0;
      const row = o * inSize;
      for (let i = 0; i < inSize; i++) sum += weight[row + i] * x[i];
      out[o] = sum;
    }
    return out;
  }

  norm(x, prefix) {
    const gamma = this.w(`${prefix}.weight`), beta = this.w(`${prefix}.bias`);
    let mean = 0;
    for (let i = 0; i < D; i++) mean += x[i];
    mean /= D;
    let variance = 0;
    for (let i = 0; i < D; i++) { const delta = x[i] - mean; variance += delta * delta; }
    variance /= D;
    const inv = 1 / Math.sqrt(variance + 1e-5);
    const out = new Float32Array(D);
    for (let i = 0; i < D; i++) out[i] = (x[i] - mean) * inv * gamma[i] + beta[i];
    return out;
  }

  attention(x, layer) {
    const p = `blocks.${layer}.attn`;
    const qkv = this.linear(x, `${p}.qkv`, D * 3, D);
    const q = qkv.subarray(0, D), k = qkv.subarray(D, D * 2), v = qkv.subarray(D * 2, D * 3);
    let cache = this.kv.get(layer);
    if (!cache) { cache = { k: [], v: [], scores: new Float64Array(512) }; this.kv.set(layer, cache); }
    cache.k.push(new Float32Array(k)); cache.v.push(new Float32Array(v));
    const context = new Float32Array(D);
    const length = cache.k.length;
    for (let head = 0; head < H; head++) {
      const base = head * HD;
      const scores = cache.scores;
      let max = -Infinity;
      for (let t = 0; t < length; t++) {
        const kt = cache.k[t]; let dot = 0;
        for (let j = 0; j < HD; j++) dot += q[base + j] * kt[base + j];
        const score = dot * SCALE; scores[t] = score; if (score > max) max = score;
      }
      let denom = 0;
      for (let t = 0; t < length; t++) { scores[t] = Math.exp(scores[t] - max); denom += scores[t]; }
      for (let t = 0; t < length; t++) {
        const probability = scores[t] / denom, vt = cache.v[t];
        for (let j = 0; j < HD; j++) context[base + j] += probability * vt[base + j];
      }
    }
    return this.linear(context, `${p}.proj`, D, D);
  }

  block(x, layer) {
    const prefix = `blocks.${layer}`;
    const n1 = this.norm(x, `${prefix}.ln1`);
    const a = this.attention(n1, layer);
    for (let i = 0; i < D; i++) x[i] += a[i];
    const n2 = this.norm(x, `${prefix}.ln2`);
    const hidden = this.linear(n2, `${prefix}.ffn.fc1`, FF, D);
    for (let i = 0; i < FF; i++) hidden[i] = 0.5 * hidden[i] * (1 + erfApprox(hidden[i] / Math.SQRT2));
    const f = this.linear(hidden, `${prefix}.ffn.fc2`, D, FF);
    for (let i = 0; i < D; i++) x[i] += f[i];
    return x;
  }

  phone1Token(token, position) {
    const tokenTable = this.w('token_embedding.weight'), posTable = this.w('position_embedding.weight');
    const x = new Float32Array(D), to = token * D, po = position * D;
    for (let i = 0; i < D; i++) x[i] = tokenTable[to + i] + posTable[po + i];
    for (let layer = 0; layer < 3; layer++) this.block(x, layer);
    this.position = position + 1;
    return x;
  }

  phone2Token(input) {
    const x = new Float32Array(input);
    for (let layer = 3; layer < 6; layer++) this.block(x, layer);
    const normalized = this.norm(x, 'final_norm');
    const logits = this.linear(normalized, 'lm_head', 256, D, false);
    let best = 0, bestValue = -Infinity;
    for (let i = 0; i < 256; i++) if (logits[i] > bestValue) { bestValue = logits[i]; best = i; }
    this.position++;
    return best;
  }

  prefillPhone1(tokens) {
    const output = new Float32Array(tokens.length * D);
    for (let i = 0; i < tokens.length; i++) output.set(this.phone1Token(tokens[i], i), i * D);
    return output;
  }

  prefillPhone2(activations) {
    const count = activations.length / D;
    let token = 0;
    for (let i = 0; i < count; i++) token = this.phone2Token(activations.subarray(i * D, (i + 1) * D));
    return token;
  }
}
