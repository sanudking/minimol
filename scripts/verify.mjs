import fs from 'node:fs';
import assert from 'node:assert/strict';
import { PartitionEngine } from '../src/engine.js';

const manifest = JSON.parse(fs.readFileSync(new URL('../models/manifest.json', import.meta.url), 'utf8'));
const reference = JSON.parse(fs.readFileSync(new URL('../models/reference_outputs.json', import.meta.url), 'utf8'));
function loadPartition(name) {
  const bytes = fs.readFileSync(new URL(`../models/${name}.bin`, import.meta.url));
  return bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength);
}
assert.equal(manifest.architecture.parameter_count, 11_040_768);
assert.equal(loadPartition('phone1').byteLength, manifest.files['phone1.bin'].nbytes);
assert.equal(loadPartition('phone2').byteLength, manifest.files['phone2.bin'].nbytes);
const phone1 = new PartitionEngine('phone1', manifest, loadPartition('phone1'));
const phone2 = new PartitionEngine('phone2', manifest, loadPartition('phone2'));
phone1.warmup(8);
phone2.warmup(8);
const prompt = new TextEncoder().encode(reference.prompt_utf8);
const hidden = phone1.prefillPhone1(prompt);
let token = phone2.prefillPhone2(hidden);
const generated = [];
for (let i = 0; i < reference.generated_byte_ids.length; i++) {
  generated.push(token);
  if (i + 1 < reference.generated_byte_ids.length) {
    const activation = phone1.phone1Token(token, prompt.length + i);
    token = phone2.phone2Token(activation);
  }
}
assert.deepEqual(generated, reference.generated_byte_ids);
console.log(`PASS: ${generated.length} greedy bytes match reference_outputs.json.`);
console.log('PASS: both partitions load and the split CPU inference path is operational.');
