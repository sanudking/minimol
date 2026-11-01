import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../dist');
const files = [];
const publicModels = path.join(root, 'models');
await fs.mkdir(publicModels, { recursive: true });
for (const filename of ['manifest.json', 'phone1.bin', 'phone2.bin', 'reference_outputs.json']) {
  await fs.copyFile(path.resolve(path.dirname(fileURLToPath(import.meta.url)), `../models/${filename}`), path.join(publicModels, filename));
}
async function walk(directory) {
  for (const entry of await fs.readdir(directory, { withFileTypes: true })) {
    const full = path.join(directory, entry.name);
    if (entry.isDirectory()) await walk(full);
    else {
      const relative = path.relative(root, full).split(path.sep).join('/');
      if (!/^models\/phone[12]\.bin$/.test(relative)) files.push(relative);
    }
  }
}
await walk(root);
files.sort();
await fs.writeFile(path.join(root, 'precache-manifest.json'), JSON.stringify({ files }, null, 2) + '\n');
console.log(`Wrote offline precache manifest for ${files.length} app assets (model partitions remain in IndexedDB).`);
