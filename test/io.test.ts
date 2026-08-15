import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import * as fs from 'fs';
import * as os from 'os';
import * as path from 'path';
import { PixelGenModel } from '../src/model/PixelGenModel';
import { saveModel, loadModel, inspectModel } from '../src/io/ModelIO';
import { Tensor } from '../src/core/Tensor';

const close = (a: number, b: number, eps = 1e-4) =>
  assert.ok(Math.abs(a - b) <= eps, `expected ${b}, got ${a}`);

function tempModelPath(): string {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'pixelgen-test-'));
  return path.join(dir, 'model.pgm');
}

const hp = {
  learningRate: 0.001,
  batchSize: 4,
  epochs: 10,
  optimizer: 'adam',
};

describe('ModelIO', () => {
  it('round-trips model weights through save/load', () => {
    const model = new PixelGenModel(3, 3);
    const filepath = tempModelPath();
    saveModel(model, filepath, hp);

    const loaded = loadModel(filepath);
    assert.deepEqual(loaded.hyperparameters, hp);

    const origParams = model.parameters();
    const loadedParams = loaded.model.parameters();
    assert.equal(origParams.length, loadedParams.length);
    for (let i = 0; i < origParams.length; i++) {
      assert.deepEqual(loadedParams[i].shape, origParams[i].shape);
      for (let j = 0; j < origParams[i].data.length; j++) {
        close(loadedParams[i].data[j], origParams[i].data[j]);
      }
    }
  });

  it('produces identical forward output after reload', () => {
    const model = new PixelGenModel(3, 3);
    const filepath = tempModelPath();
    saveModel(model, filepath, hp);

    const { model: loaded } = loadModel(filepath);
    const input = new Tensor(new Float32Array(3 * 8 * 8), [1, 3, 8, 8]);
    // Use deterministic input
    for (let i = 0; i < input.data.length; i++) {
      input.data[i] = Math.sin(i) * 0.5 + 0.5;
    }
    const a = model.forward(input);
    const b = loaded.forward(input);
    assert.deepEqual(a.shape, b.shape);
    for (let i = 0; i < a.data.length; i++) {
      close(a.data[i], b.data[i]);
    }
  });

  it('inspectModel reads metadata without loading weights', () => {
    const model = new PixelGenModel(1, 2);
    const filepath = tempModelPath();
    saveModel(model, filepath, hp);

    const info = inspectModel(filepath);
    assert.equal(info.metadata.inputChannels, 1);
    assert.equal(info.metadata.outputChannels, 2);
    assert.equal(info.metadata.architecture, 'pixelgen-autoencoder-v1');
    assert.ok(info.totalParameters > 0);
    assert.ok(info.fileSize > 0);
  });

  it('throws on invalid files', () => {
    const filepath = tempModelPath();
    fs.writeFileSync(filepath, 'not a pgm file');
    assert.throws(() => loadModel(filepath));
    assert.throws(() => inspectModel(filepath));
  });
});