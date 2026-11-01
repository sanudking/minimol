import { PartitionEngine } from './engine.js';
let engine = null;
self.onmessage = event => {
  const { id, type, role, manifest, buffer, tokens, activation, token, position } = event.data;
  try {
    if (type === 'init') {
      engine = new PartitionEngine(role, manifest, buffer);
      engine.warmup(8);
      self.postMessage({ id, ok: true, type: 'ready' });
    } else if (!engine) throw new Error('Worker is not initialized');
    else if (type === 'reset') { engine.reset(); self.postMessage({ id, ok: true, type: 'reset' }); }
    else if (type === 'prefill1') {
      const result = engine.prefillPhone1(tokens);
      self.postMessage({ id, ok: true, type: 'activation', activation: result.buffer, seqLen: tokens.length }, [result.buffer]);
    } else if (type === 'step1') {
      const result = engine.phone1Token(token, position);
      self.postMessage({ id, ok: true, type: 'activation', activation: result.buffer, seqLen: 1 }, [result.buffer]);
    } else if (type === 'prefill2') {
      const result = engine.prefillPhone2(new Float32Array(activation));
      self.postMessage({ id, ok: true, type: 'token', token: result });
    } else if (type === 'step2') {
      const result = engine.phone2Token(new Float32Array(activation));
      self.postMessage({ id, ok: true, type: 'token', token: result });
    }
  } catch (error) { self.postMessage({ id, ok: false, error: error?.stack || String(error) }); }
};
